module CoarseGrainingEnergyFluxesGPUExt

# `@kernel`/`@index`/`@Const` are imported bare (not qualified as KA.@kernel etc.) as a verified,
# necessary exception: KernelAbstractions' `@kernel` macro does AST pattern-matching on the literal
# unqualified `@Const`/`@index` syntax during its own expansion — qualifying them breaks precompilation
# with a real MethodError (confirmed directly, not assumed), so this is not a style choice.
using KernelAbstractions: KernelAbstractions as KA, @kernel, @index, @Const
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF
using FlowGeometries: FlowGeometries

# GPUBackend: apply the precomputed footprint with one GPU thread per output cell. The footprint
# (offset + weight arrays) and the mask are moved to the device once; the kernel mirrors the serial
# per-row logic exactly (so on the KernelAbstractions CPU backend it matches the serial result).
# `is_zerofill` selects the masking branch as a plain Bool (kernels can't dispatch on types).

@kernel function _cgef_filter_kernel!(
    out, @Const(field), @Const(mask), @Const(di), @Const(dj), @Const(w), @Const(ptr),
    nbands::Int, periodic_x::Bool, periodic_y::Bool, is_zerofill::Bool,
)
    i, j = @index(Global, NTuple)
    T = eltype(out)
    Nx, Ny = size(out)
    if i <= Nx && j <= Ny
        if mask[i, j]
            b = nbands == 1 ? 1 : j
            lo = ptr[b]
            hi = ptr[b+1] - 1
            ws = zero(T)
            wn = zero(T)
            for k in lo:hi
                jj = j + dj[k]
                inbounds = true
                if jj < 1 || jj > Ny
                    if periodic_y
                        jj = mod1(jj, Ny)
                    else
                        inbounds = false
                    end
                end
                if inbounds
                    ii = i + di[k]
                    if ii < 1 || ii > Nx
                        if periodic_x
                            ii = mod1(ii, Nx)
                        else
                            inbounds = false
                        end
                    end
                    if inbounds
                        active = mask[ii, jj]
                        wk = w[k]
                        if is_zerofill
                            wn += wk
                            if active
                                ws += wk * field[ii, jj]
                            end
                        elseif active
                            wn += wk
                            ws += wk * field[ii, jj]
                        end
                    end
                end
            end
            out[i, j] = wn > T(1e-15) ? ws / wn : zero(T)
        else
            out[i, j] = zero(T)
        end
    end
end

# Cached scattered/nonuniform-axis variant: `ii_arr`/`jj_arr` hold ABSOLUTE neighbour indices
# (periodic wrap already resolved at footprint-build time, per `ScatteredCache`'s convention), so
# there's no offset arithmetic or bounds/wrap branch here — mirrors `apply_footprint_row!`'s cached
# branch.
@kernel function _cgef_filter_kernel_scattered!(
    out, @Const(field), @Const(mask), @Const(ii_arr), @Const(jj_arr), @Const(w), @Const(ptr),
    is_zerofill::Bool,
)
    i, j = @index(Global, NTuple)
    T = eltype(out)
    Nx, Ny = size(out)
    if i <= Nx && j <= Ny
        if mask[i, j]
            t = i + (j - 1) * Nx
            lo = ptr[t]
            hi = ptr[t+1] - 1
            ws = zero(T)
            wn = zero(T)
            for k in lo:hi
                ii = ii_arr[k]
                jj = jj_arr[k]
                active = mask[ii, jj]
                wk = w[k]
                if is_zerofill
                    wn += wk
                    if active
                        ws += wk * field[ii, jj]
                    end
                elseif active
                    wn += wk
                    ws += wk * field[ii, jj]
                end
            end
            out[i, j] = wn > T(1e-15) ? ws / wn : zero(T)
        else
            out[i, j] = zero(T)
        end
    end
end

# The streaming kernel calls the SAME traversal the host does — `Connectivity.fold_within` compiles
# inside a `@kernel`: it threads its accumulator by value, and on a `StructuredGrid` the query is index
# arithmetic over `metric_window` with no scratch and no spatial index. The window, the two periodic
# conventions and the distance therefore have one implementation, not one per backend.
@kernel function _cgef_filter_kernel_traversal!(
    out, @Const(field), @Const(mask), grid, mt, kernel, scale, rad, is_cartesian::Bool, is_zerofill::Bool,
)
    i, j = @index(Global, NTuple)
    T = eltype(out)
    Nx, Ny = size(out)
    if i <= Nx && j <= Ny
        if mask[i, j]
            # `Filtering._ball_fold` is the host's own entry point: it picks the image convention for a
            # `StructuredGrid` and omits the argument for a curvilinear mesh, which has no axis to tile.
            ws, wn = CGEF.Filtering._ball_fold(
                (zero(T), zero(T)), grid, Int(i), Int(j), rad, is_cartesian, mt,
            ) do acc, J, d
                a, b = acc
                wk = CGEF.Kernels.kernel_weight(kernel, T(d), scale) * FlowGeometries.Grids.area(grid, J[1], J[2])
                active = mask[J[1], J[2]]
                # No `return` anywhere: `@kernel` rejects one even inside a closure.
                if is_zerofill
                    (active ? a + wk * field[J[1], J[2]] : a, b + wk)
                elseif active
                    (a + wk * field[J[1], J[2]], b + wk)
                else
                    acc
                end
            end
            out[i, j] = wn > T(1e-15) ? ws / wn : zero(T)
        else
            out[i, j] = zero(T)
        end
    end
end

# Separable Gaussian: two kernel launches with a device-resident `row_pass` between them, mirroring
# the CPU row-pass/column-pass bodies so results are bit-identical. A thread's column pass reads
# `row_pass[:, jj]` for rows other than its own, so the driver `KA.synchronize`s between the two
# launches and they cannot be fused the way the disk-truncated kernels above are.
#
# The weight table is a vector when the axis spacing is constant and a `(2·lim+1) × N` matrix when it
# is not (see `Filtering._sepw`), so the rank is dispatched on here as it is on the host — indexing a
# matrix linearly would read down its first column and produce a wrong answer rather than an error.
@inline _gpu_sepw(g::AbstractVector, ::Int, k::Int) = @inbounds g[k]
@inline _gpu_sepw(g::AbstractMatrix, i::Int, k::Int) = @inbounds g[k, i]

@kernel function _cgef_separable_row_pass_kernel!(row_pass, @Const(masked_input), @Const(gx), di_lim::Int, periodic_x::Bool)
    i, j = @index(Global, NTuple)
    T = eltype(row_pass)
    Nx, Ny = size(row_pass)
    if i <= Nx && j <= Ny
        s = zero(T)
        for ddi in (-di_lim):di_lim
            ii = i + ddi
            inbounds = true
            if ii < 1 || ii > Nx
                if periodic_x
                    ii = mod1(ii, Nx)
                else
                    inbounds = false
                end
            end
            if inbounds
                s += _gpu_sepw(gx, i, ddi + di_lim + 1) * masked_input[ii, j]
            end
        end
        row_pass[i, j] = s
    end
end

@kernel function _cgef_separable_column_pass_kernel!(out, @Const(row_pass), @Const(gy), dj_lim::Int, periodic_y::Bool)
    i, j = @index(Global, NTuple)
    T = eltype(out)
    Nx, Ny = size(out)
    if i <= Nx && j <= Ny
        s = zero(T)
        for ddj in (-dj_lim):dj_lim
            jj = j + ddj
            inbounds = true
            if jj < 1 || jj > Ny
                if periodic_y
                    jj = mod1(jj, Ny)
                else
                    inbounds = false
                end
            end
            if inbounds
                s += _gpu_sepw(gy, j, ddj + dj_lim + 1) * row_pass[i, jj]
            end
        end
        out[i, j] = s
    end
end

move(dev, x) = (y = KA.allocate(dev, eltype(x), size(x)); copyto!(y, x); y)

"""
    GPUResident

A footprint whose device-side arrays — tables, mask, and any intermediate buffers — already live on
the device. `plan_filter` builds one through [`CGEF.Filtering.prepare_workspace`](@ref), so a plan
transfers its footprint ONCE and every later `filter_apply!` goes straight to the kernel launches.
Nothing here depends on the field or the mask strategy, which are the only things that change between
calls.
"""
abstract type GPUResident end

# Banded (uniform-axis) footprint: per-band offset/weight tables.
struct GPUBandedFootprint{F, DI, DJ, VW, VP, MA} <: GPUResident
    fp::F
    di::DI
    dj::DJ
    w::VW
    ptr::VP
    maskd::MA
end

# Scattered footprint with a materialized neighbour cache: absolute neighbour indices per point.
struct GPUScatteredCached{F, II, JJ, VW, VP, MA} <: GPUResident
    fp::F
    ii::II
    jj::JJ
    w::VW
    ptr::VP
    maskd::MA
end

# Streaming footprint: no cache, so the kernel re-derives each point's neighbourhood from the grid
# itself. Coordinates and the cell measure are read through the grid rather than uploaded separately.
struct GPUStreaming{F, G, MT, MA} <: GPUResident
    fp::F
    grid::G
    topology::MT
    maskd::MA
end

# Separable Gaussian: the two weight tables, the two scratch planes the passes need, and whichever
# denominator the footprint was built with — the `ZeroFill` one is the rank-1 outer product, constant
# across calls, so it is formed here rather than rebuilt every apply.
struct GPUSeparable{F, GX, GY, AT, IR, DE, MA} <: GPUResident
    fp::F
    gx::GX
    gy::GY
    masked_input::AT
    row_pass::AT
    invrenorm::IR   # dense Deformable denominator, or nothing
    denom::DE       # rank-1 ZeroFill denominator, or nothing
    maskd::MA
end

_maskd(dev, grid) = move(dev, Array{Bool}(FlowGeometries.Grids.mask(grid)))

function CGEF.Filtering.prepare_workspace(
    b::CGEF.ComputationalBackends.GPUBackend, grid::FlowGeometries.Grids.AbstractGrid,
    fp::CGEF.Filtering.FilterFootprint,
)
    dev = b.backend
    return GPUBandedFootprint(fp, move(dev, fp.di), move(dev, fp.dj), move(dev, fp.w), move(dev, fp.ptr), _maskd(dev, grid))
end

function CGEF.Filtering.prepare_workspace(
    b::CGEF.ComputationalBackends.GPUBackend, grid::FlowGeometries.Grids.AbstractGrid,
    fp::CGEF.Filtering.ScatteredFilterPlan,
)
    dev = b.backend
    maskd = _maskd(dev, grid)
    cache = fp.cache
    cache === nothing || return GPUScatteredCached(
        fp, move(dev, cache.ii), move(dev, cache.jj), move(dev, cache.w), move(dev, cache.ptr), maskd,
    )
    # The kernel runs the grid's own ball query, so the grid goes where the kernel does — FlowGeometries'
    # `Adapt` support does the move, and on the CPU device it is the identity.
    #
    # NOT the plan's topology: on a curvilinear grid that one carries a k-d tree, and `Adapt` refuses to
    # move a spatial index rather than leave the device holding a host pointer. A device query is the
    # unindexed scan.
    return GPUStreaming(
        fp, KA.adapt(dev, grid), FlowGeometries.Connectivity.MetricTopology(grid), maskd,
    )
end

function CGEF.Filtering.prepare_workspace(
    b::CGEF.ComputationalBackends.GPUBackend, grid::FlowGeometries.Grids.AbstractGrid,
    fp::CGEF.Filtering.SeparableGaussianFootprint{T},
) where {T<:AbstractFloat}
    dev = b.backend
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    invrenorm = fp.invrenorm === nothing ? nothing : move(dev, fp.invrenorm)
    denom = if fp.invrenorm === nothing
        xp = move(dev, fp.Nx_profile)
        yp = move(dev, fp.Ny_profile)
        d = KA.allocate(dev, T, Nx, Ny)
        d .= reshape(xp, Nx, 1) .* reshape(yp, 1, Ny)
        d
    else
        nothing
    end
    return GPUSeparable(
        fp, move(dev, fp.gx), move(dev, fp.gy),
        KA.allocate(dev, T, Nx, Ny), KA.allocate(dev, T, Nx, Ny),
        invrenorm, denom, _maskd(dev, grid),
    )
end

# Dispatch on which resident footprint `prepare_workspace` produced — uniform axes get the
# offset-based kernel, nonuniform axes the cached or streaming scattered kernel.
function _run_gpu_kernel!(dev, out, field, ws::GPUBandedFootprint, grid, periodic_x::Bool, periodic_y::Bool, is_zerofill::Bool)
    Nx, Ny = size(out)
    _cgef_filter_kernel!(dev)(
        out, field, ws.maskd, ws.di, ws.dj, ws.w, ws.ptr, ws.fp.nbands,
        periodic_x, periodic_y, is_zerofill; ndrange = (Nx, Ny),
    )
end

function _run_gpu_kernel!(dev, out, field, ws::GPUScatteredCached, grid, periodic_x::Bool, periodic_y::Bool, is_zerofill::Bool)
    Nx, Ny = size(out)
    _cgef_filter_kernel_scattered!(dev)(out, field, ws.maskd, ws.ii, ws.jj, ws.w, ws.ptr, is_zerofill; ndrange = (Nx, Ny))
end

function _run_gpu_kernel!(dev, out, field, ws::GPUStreaming, grid, periodic_x::Bool, periodic_y::Bool, is_zerofill::Bool)
    Nx, Ny = size(out)
    fp = ws.fp
    _cgef_filter_kernel_traversal!(dev)(
        out, field, ws.maskd, ws.grid, ws.topology, fp.kernel, fp.scale, fp.rad, fp.is_cartesian,
        is_zerofill; ndrange = (Nx, Ny),
    )
end

# Separable Gaussian fast path: two kernel launches (row-pass, then column-pass) separated by a
# `KA.synchronize` — the column pass reads rows other than its own, so they cannot be fused the way
# the disk-truncated kernels are. The mask-strategy branch is already baked into which denominator the
# footprint carries, exactly as on the host.
function _run_gpu_kernel!(dev, out, field, ws::GPUSeparable, grid, periodic_x::Bool, periodic_y::Bool, is_zerofill::Bool)
    T = eltype(out)
    Nx, Ny = size(out)
    fp = ws.fp
    ws.masked_input .= T.(ws.maskd) .* field
    _cgef_separable_row_pass_kernel!(dev)(ws.row_pass, ws.masked_input, ws.gx, fp.di_lim, fp.periodic_x; ndrange = (Nx, Ny))
    KA.synchronize(dev)
    _cgef_separable_column_pass_kernel!(dev)(out, ws.row_pass, ws.gy, fp.dj_lim, fp.periodic_y; ndrange = (Nx, Ny))
    KA.synchronize(dev)
    if ws.invrenorm === nothing
        out .= ifelse.(ws.denom .> T(1e-15), out ./ ws.denom, zero(T))
    else
        out .*= ws.invrenorm
    end
    out .= ifelse.(ws.maskd, out, zero(T))
    return out
end

# Whatever the caller supplied, end up with a device-resident footprint: a plan built for this backend
# already holds one, anything else (no plan, or a plan built for another backend) is uploaded here.
@inline _resident(::CGEF.ComputationalBackends.GPUBackend, grid, ws::GPUResident, kernel, scale, mask_strategy) = ws
@inline _resident(b::CGEF.ComputationalBackends.GPUBackend, grid, ws, kernel, scale, mask_strategy) =
    CGEF.Filtering.prepare_workspace(b, grid, ws)
@inline _resident(b::CGEF.ComputationalBackends.GPUBackend, grid, ::Nothing, kernel, scale, mask_strategy) =
    CGEF.Filtering.prepare_workspace(b, grid, CGEF.Filtering.build_footprint(grid, kernel, scale; mask_strategy = mask_strategy))

# Only `mask` and `isperiodic` are read from the grid here, and both already work for a
# `CurvilinearGrid` (`isperiodic` falls back to `false` — a curvilinear mesh carries no periodicity
# flags — through `AbstractGrid`'s default method).
function CGEF.Filtering.gpu_filter_field!(
    gpu_backend::CGEF.ComputationalBackends.GPUBackend,
    out::AbstractMatrix{T},
    field::AbstractMatrix{T},
    grid::Union{FlowGeometries.Grids.StructuredGrid{G,T,2}, FlowGeometries.Grids.CurvilinearGrid{T,G}},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
    workspace,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    dev = gpu_backend.backend
    if workspace isa CGEF.Filtering.PrefixSumTopHatPlan
        # Phase 1 is a running scan along axis 1, so it is not a per-point-parallel device kernel, and
        # the plan's buffers are host arrays. Run it on the host: at O(N·dj_lim) against the device
        # kernels' O(N·di_lim·dj_lim) it is still the fastest path for this kernel and grid. A device
        # implementation needs a parallel (Blelloch) scan.
        return CGEF.Filtering.apply_prefixsum_tophat!(out, field, grid, workspace, mask_strategy)
    end
    ws = _resident(gpu_backend, grid, workspace, kernel, scale, mask_strategy)
    ws isa GPUSeparable && CGEF.Filtering._separable_check_strategy(ws.fp, mask_strategy)
    is_zerofill = mask_strategy isa CGEF.Filtering.ZeroFill
    _run_gpu_kernel!(
        dev, out, field, ws, grid,
        FlowGeometries.Grids.isperiodic(grid, 1), FlowGeometries.Grids.isperiodic(grid, 2), is_zerofill,
    )
    KA.synchronize(dev)
    return out
end

end # module
