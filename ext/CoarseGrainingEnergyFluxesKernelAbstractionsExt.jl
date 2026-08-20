module CoarseGrainingEnergyFluxesKernelAbstractionsExt

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

# Batch axis, carried by ONE kernel body rather than a duplicated "batched" twin. Every kernel here is
# launched with a batch extent, and `size(A, 3)` is 1 for a 2-D array, so the unbatched case is the
# Nb == 1 case rather than separate code. `_at`/`_set!` dispatch on array rank, which resolves at compile
# time, so a 2-D field costs exactly what direct indexing did.
#
# The footprint tables, weight tables and mask are spatial-only and shared across `b` untouched: only
# the field and output accesses carry the index. A batch therefore becomes one launch of
# `prod(spatial)*Nb` work items instead of `Nb` launches of `prod(spatial)`, which is what matters when a
# single slice does not fill the device — a 64² slice is 4k work items.
@inline _at(A::AbstractArray{<:Any,2}, i::Int, j::Int, ::Int) = @inbounds A[i, j]
@inline _at(A::AbstractArray{<:Any,3}, i::Int, j::Int, b::Int) = @inbounds A[i, j, b]
@inline _set!(A::AbstractArray{<:Any,2}, i::Int, j::Int, ::Int, v) = @inbounds A[i, j] = v
@inline _set!(A::AbstractArray{<:Any,3}, i::Int, j::Int, b::Int, v) = @inbounds A[i, j, b] = v

# CartesianIndex form, for the engines that walk a dimension-generic index. Matching ranks means the
# array is unbatched; a higher rank means the trailing axis is the batch.
@inline _atI(A::AbstractArray{<:Any,N}, I::CartesianIndex{N}, ::Int) where {N} = @inbounds A[I]
@inline _atI(A::AbstractArray{<:Any,M}, I::CartesianIndex{N}, b::Int) where {M,N} = @inbounds A[I, b]
@inline _setI!(A::AbstractArray{<:Any,N}, I::CartesianIndex{N}, ::Int, v) where {N} = @inbounds A[I] = v
@inline _setI!(A::AbstractArray{<:Any,M}, I::CartesianIndex{N}, b::Int, v) where {M,N} = @inbounds A[I, b] = v

@kernel function _cgef_filter_kernel!(
    out, @Const(field), @Const(mask), @Const(di), @Const(dj), @Const(w), @Const(ptr),
    nbands::Int, periodic_x::Bool, periodic_y::Bool, is_zerofill::Bool,
)
    i, j, b = @index(Global, NTuple)
    T = eltype(out)
    Nx, Ny, Nb = size(out, 1), size(out, 2), size(out, 3)
    if i <= Nx && j <= Ny && b <= Nb
        if mask[i, j]
            band = nbands == 1 ? 1 : j
            lo = ptr[band]
            hi = ptr[band+1] - 1
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
                                ws += wk * _at(field, ii, jj, b)
                            end
                        elseif active
                            wn += wk
                            ws += wk * _at(field, ii, jj, b)
                        end
                    end
                end
            end
            _set!(out, i, j, b, wn > T(1e-15) ? ws / wn : zero(T))
        else
            _set!(out, i, j, b, zero(T))
        end
    end
end

# 1-D / true-3-D grids: the offset/weight table is translation-invariant, so the kernel walks it per
# point exactly as the serial `_footprint_nd_point` does. Written over a linear index and reconstructed
# through `CartesianIndices` so one kernel serves every `N` — a per-dimension kernel would be three
# copies of the same arithmetic.
@kernel function _cgef_filter_kernel_nd!(
    out, @Const(field), @Const(mask), @Const(offsets), @Const(w),
    dims, periodic, is_zerofill::Bool,
)
    lin, b = @index(Global, NTuple)
    T = eltype(out)
    nspatial = prod(dims)
    Nb = size(out, length(dims) + 1)
    if lin <= nspatial && b <= Nb
        I = CartesianIndices(dims)[lin]
        if mask[I]
            Ti = Tuple(I)
            ws = zero(T)
            wn = zero(T)
            for k in eachindex(offsets)
                off = offsets[k]
                # Validity first, in a plain loop: a closure that ASSIGNED to `valid` would capture and
                # box it, which a device kernel cannot do. The `ntuple` below therefore only reads.
                valid = true
                for d in eachindex(dims)
                    jj = Ti[d] + off[d]
                    if (jj < 1 || jj > dims[d]) && !periodic[d]
                        valid = false
                    end
                end
                if valid
                    J = ntuple(Val(length(dims))) do d
                        jj = Ti[d] + off[d]
                        (jj < 1 || jj > dims[d]) ? mod1(jj, dims[d]) : jj
                    end
                    JI = CartesianIndex(J)
                    active = mask[JI]
                    wk = w[k]
                    if is_zerofill
                        wn += wk
                        if active
                            ws += wk * _atI(field, JI, b)
                        end
                    elseif active
                        wn += wk
                        ws += wk * _atI(field, JI, b)
                    end
                end
            end
            _setI!(out, I, b, wn > T(1e-15) ? ws / wn : zero(T))
        else
            _setI!(out, I, b, zero(T))
        end
    end
end

# Device analogue of the threaded ext's `_omt_driver`: apply a caller-supplied closure once per index.
# `apply_separable_nd!` is written against this driver abstraction, so the separable `N`-pass
# engine runs on the device with no separate implementation — the passes stay ordered because each
# kernel launch is synchronized before the next.
@kernel function _cgef_driver_kernel!(f, indices)
    i = @index(Global, Linear)
    if i <= length(indices)
        f(indices[i])
    end
end

struct GPUDriver{D}
    dev::D
end
@inline function (drv::GPUDriver)(f::F, indices) where {F}
    _cgef_driver_kernel!(drv.dev)(f, indices; ndrange = length(indices))
    KA.synchronize(drv.dev)
    return nothing
end

# Cached ND scattered footprint: `nbrs` holds resolved neighbour multi-indices (periodic wrap already
# applied at build time), so like the 2-D cached kernel there is no offset or wrap arithmetic here.
@kernel function _cgef_filter_kernel_nd_cached!(
    out, @Const(field), @Const(mask), @Const(nbrs), @Const(w), @Const(ptr), dims, is_zerofill::Bool,
)
    lin, b = @index(Global, NTuple)
    T = eltype(out)
    nspatial = prod(dims)
    Nb = size(out, length(dims) + 1)
    if lin <= nspatial && b <= Nb
        I = CartesianIndices(dims)[lin]
        if mask[I]
            ws = zero(T)
            wn = zero(T)
            for k in ptr[lin]:(ptr[lin+1] - 1)
                JI = CartesianIndex(nbrs[k])
                active = mask[JI]
                wk = w[k]
                if is_zerofill
                    wn += wk
                    if active
                        ws += wk * _atI(field, JI, b)
                    end
                elseif active
                    wn += wk
                    ws += wk * _atI(field, JI, b)
                end
            end
            _setI!(out, I, b, wn > T(1e-15) ? ws / wn : zero(T))
        else
            _setI!(out, I, b, zero(T))
        end
    end
end

# Node sets: the footprint is already a CSR adjacency, so there is no window to derive and the kernel
# is a flat gather over each node's stored neighbour block. Index space is 1-D and carries no grid
# dimensionality, which is why this serves a node set of any embedding dimension.
# Node gather with the batch axis carried the same way as the structured kernels: the neighbour lists,
# weights and mask are per-node and shared across `b`, so only the field/output accesses take the index.
# `size(A, 2)` is 1 for a node vector, so an unbatched apply is the `Nb == 1` case of this launch.
@inline _at1(A::AbstractVector, t::Int, ::Int) = @inbounds A[t]
@inline _at1(A::AbstractMatrix, t::Int, b::Int) = @inbounds A[t, b]
@inline _set1!(A::AbstractVector, t::Int, ::Int, v) = @inbounds A[t] = v
@inline _set1!(A::AbstractMatrix, t::Int, b::Int, v) = @inbounds A[t, b] = v

@kernel function _cgef_filter_kernel_node!(
    out, @Const(field), @Const(mask), @Const(nbrs), @Const(w), @Const(ptr), is_zerofill::Bool,
)
    t, b = @index(Global, NTuple)
    T = eltype(out)
    Nn, Nb = size(out, 1), size(out, 2)
    if t <= Nn && b <= Nb
        if mask[t]
            ws = zero(T)
            wn = zero(T)
            for k in ptr[t]:(ptr[t+1] - 1)
                j = nbrs[k]
                wk = w[k]
                active = mask[j]
                if is_zerofill
                    wn += wk
                    if active
                        ws += wk * _at1(field, j, b)
                    end
                elseif active
                    wn += wk
                    ws += wk * _at1(field, j, b)
                end
            end
            _set1!(out, t, b, wn > T(1e-15) ? ws / wn : zero(T))
        else
            _set1!(out, t, b, zero(T))
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
    i, j, b = @index(Global, NTuple)
    T = eltype(out)
    Nx, Ny, Nb = size(out, 1), size(out, 2), size(out, 3)
    if i <= Nx && j <= Ny && b <= Nb
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
                        ws += wk * _at(field, ii, jj, b)
                    end
                elseif active
                    wn += wk
                    ws += wk * _at(field, ii, jj, b)
                end
            end
            _set!(out, i, j, b, wn > T(1e-15) ? ws / wn : zero(T))
        else
            _set!(out, i, j, b, zero(T))
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
    i, j, bi = @index(Global, NTuple)
    T = eltype(out)
    Nx, Ny, Nb = size(out, 1), size(out, 2), size(out, 3)
    if i <= Nx && j <= Ny && bi <= Nb
        if mask[i, j]
            # `Filtering._ball_fold` is the host's own entry point: it picks the image convention for a
            # `StructuredGrid` and omits the argument for a curvilinear mesh, which has no axis to tile.
            # The fold closure only READS `bi`; assigning to a captured variable would box it, which a
            # device kernel cannot do.
            ws, wn = CGEF.Filtering._ball_fold(
                (zero(T), zero(T)), grid, Int(i), Int(j), rad, is_cartesian, mt,
            ) do acc, J, d
                a, b = acc
                wk = CGEF.Kernels.kernel_weight(kernel, T(d), scale) * FlowGeometries.Grids.area(grid, J[1], J[2])
                active = mask[J[1], J[2]]
                # No `return` anywhere: `@kernel` rejects one even inside a closure.
                if is_zerofill
                    (active ? a + wk * _at(field, J[1], J[2], bi) : a, b + wk)
                elseif active
                    (a + wk * _at(field, J[1], J[2], bi), b + wk)
                else
                    acc
                end
            end
            _set!(out, i, j, bi, wn > T(1e-15) ? ws / wn : zero(T))
        else
            _set!(out, i, j, bi, zero(T))
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
@inline _gpu_sepw(g::AbstractMatrix, i::Int, k::Int) = @inbounds g[i, k]

# The separable weight tables are indexed by POSITION along the pass axis only, so like the footprint
# tables above they are shared across `b` and the batch index only reaches the field and output.
@kernel function _cgef_separable_row_pass_kernel!(row_pass, @Const(masked_input), @Const(gx), di_lim::Int, periodic_x::Bool)
    i, j, b = @index(Global, NTuple)
    T = eltype(row_pass)
    Nx, Ny, Nb = size(row_pass, 1), size(row_pass, 2), size(row_pass, 3)
    if i <= Nx && j <= Ny && b <= Nb
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
                s += _gpu_sepw(gx, i, ddi + di_lim + 1) * _at(masked_input, ii, j, b)
            end
        end
        _set!(row_pass, i, j, b, s)
    end
end

@kernel function _cgef_separable_column_pass_kernel!(out, @Const(row_pass), @Const(gy), dj_lim::Int, periodic_y::Bool)
    i, j, b = @index(Global, NTuple)
    T = eltype(out)
    Nx, Ny, Nb = size(out, 1), size(out, 2), size(out, 3)
    if i <= Nx && j <= Ny && b <= Nb
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
                s += _gpu_sepw(gy, j, ddj + dj_lim + 1) * _at(row_pass, i, jj, b)
            end
        end
        _set!(out, i, j, b, s)
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

# Separable Gaussian on a 1-D / true-3-D grid: the whole footprint is rebuilt with device arrays, so
# the shared `N`-pass engine can run against it unchanged.
struct GPUSeparableND{F} <: GPUResident
    fp::F
end

# Cached ND scattered footprint, device-resident.
struct GPUScatteredNDCached{F, VN, VW, VP, MA} <: GPUResident
    fp::F
    nbrs::VN
    w::VW
    ptr::VP
    maskd::MA
end

# 1-D / true-3-D translation-invariant offset table.
struct GPUFootprintND{F, VO, VW, MA} <: GPUResident
    fp::F
    offsets::VO
    w::VW
    maskd::MA
end

# Node set: the CSR adjacency and its weights, moved once. No coordinates and no topology are needed
# on the device — the neighbourhood is already resolved into indices at build time.
struct GPUNodeFootprint{F, VN, VW, VP, MA} <: GPUResident
    fp::F
    nbrs::VN
    w::VW
    ptr::VP
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

# A device kernel cannot run the grid's own ball query for a nonuniform ND axis without the topology
# machinery the 2-D streaming path carries, so the GPU uses the MATERIALIZED neighbour list. When the
# plan was built streaming (`NeverCache`), it is rebuilt cached here — an explicit, documented memory
# cost of choosing the device, not a silent change of engine: the numbers are identical either way.
function CGEF.Filtering.prepare_workspace(
    b::CGEF.ComputationalBackends.GPUBackend, grid::FlowGeometries.Grids.AbstractGrid,
    fp::CGEF.Filtering.NDScatteredFilterPlan,
)
    dev = b.backend
    cached = fp.cache === nothing ?
        CGEF.Filtering.build_footprint(grid, fp.kernel, fp.scale;
            cache_strategy = CGEF.Filtering.AlwaysCache()) : fp
    cache = cached.cache
    return GPUScatteredNDCached(
        cached, move(dev, cache.nbrs), move(dev, cache.w), move(dev, cache.ptr), _maskd(dev, grid),
    )
end

function CGEF.Filtering.prepare_workspace(
    b::CGEF.ComputationalBackends.GPUBackend, grid::FlowGeometries.Grids.AbstractGrid,
    fp::CGEF.Filtering.SeparableFootprintND{N,T},
) where {N, T<:AbstractFloat}
    dev = b.backend
    return GPUSeparableND(CGEF.Filtering.SeparableFootprintND(
        map(g -> move(dev, g), fp.g), fp.lim, fp.periodic,
        fp.profiles === nothing ? nothing : map(pv -> move(dev, pv), fp.profiles),
        fp.invrenorm === nothing ? nothing : move(dev, fp.invrenorm),
        fp.masked, move(dev, fp.masked_input), move(dev, fp.scratch),
    ))
end

function CGEF.Filtering.prepare_workspace(
    b::CGEF.ComputationalBackends.GPUBackend, grid::FlowGeometries.Grids.AbstractGrid,
    fp::CGEF.Filtering.FilterFootprintND,
)
    dev = b.backend
    return GPUFootprintND(fp, move(dev, fp.offsets), move(dev, fp.w), _maskd(dev, grid))
end

function CGEF.Filtering.prepare_workspace(
    b::CGEF.ComputationalBackends.GPUBackend, grid::FlowGeometries.Grids.AbstractGrid,
    fp::CGEF.Filtering.NodeFilterPlan,
)
    dev = b.backend
    return GPUNodeFootprint(fp, move(dev, fp.nbrs), move(dev, fp.w), move(dev, fp.ptr), _maskd(dev, grid))
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
    fp::CGEF.Filtering.SeparableFootprint{T},
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
# One method for both ranks. A 3-D output against a banded footprint is unambiguously a batch — the
# banded engine exists only for 2-D structured grids, so the extra axis cannot be spatial — and a 2-D
# output is the `Nb == 1` case of the same launch.
function _run_gpu_kernel!(dev, out, field, ws::GPUBandedFootprint, grid, periodic_x::Bool, periodic_y::Bool, is_zerofill::Bool)
    _cgef_filter_kernel!(dev)(
        out, field, ws.maskd, ws.di, ws.dj, ws.w, ws.ptr, ws.fp.nbands,
        periodic_x, periodic_y, is_zerofill;
        ndrange = (size(out, 1), size(out, 2), size(out, 3)),
    )
end

# The banded engine is the one whose device kernel carries a batch index, so it is the one that can take
# the fused launch. Every other engine falls to the slice loop in `filter_apply_batched!` rather than a
# kernel that would silently ignore the batch axis.
# `plan.footprint` on a GPU plan is the DEVICE-resident footprint that `prepare_workspace` produced, not
# the host `FilterFootprint` — testing for the host type here silently disables the fused path.
# Structured grids pre-upload at `plan_filter` time so the plan already holds a device-resident
# footprint; a node grid does not, so its plan still holds the host `NodeFilterPlan` and `_resident`
# uploads at apply time. Both spellings therefore have to be accepted, or the fused path is silently
# disabled for exactly the grid architecture that needs it most.
CGEF.Filtering._gpu_batched_supported(plan::CGEF.Filtering.PhysicalFilterPlan) =
    plan.footprint isa Union{GPUBandedFootprint, GPUSeparable, GPUNodeFootprint, GPUFootprintND, GPUSeparableND,
                             GPUScatteredCached, GPUScatteredNDCached, GPUStreaming,
                             CGEF.Filtering.NodeFilterPlan}

# Node grids: the spatial rank is 1, so a batch is a `(Nnodes, Nb)` matrix rather than a 3-D array.
function CGEF.Filtering.gpu_filter_field_batched!(
    gpu_backend::CGEF.ComputationalBackends.GPUBackend,
    out::AbstractArray{T,2},
    field::AbstractArray{T,2},
    grid::FlowGeometries.Grids.UnstructuredGrid,
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
    workspace,
) where {T<:AbstractFloat}
    dev = gpu_backend.backend
    ws = _resident(gpu_backend, grid, workspace, kernel, scale, mask_strategy)
    _run_gpu_kernel!(dev, out, field, ws, grid, false, false, mask_strategy isa CGEF.Filtering.ZeroFill)
    KA.synchronize(dev)
    return out
end

# Any structured rank: `filter_apply_batched!` has already checked that the leading axes are the grid and
# the trailing one is the batch, so the rank does not need pinning here.
function CGEF.Filtering.gpu_filter_field_batched!(
    gpu_backend::CGEF.ComputationalBackends.GPUBackend,
    out::AbstractArray{T},
    field::AbstractArray{T},
    grid::Union{FlowGeometries.Grids.StructuredGrid{G,T}, FlowGeometries.Grids.CurvilinearGrid{T,G}},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
    workspace,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    dev = gpu_backend.backend
    ws = _resident(gpu_backend, grid, workspace, kernel, scale, mask_strategy)
    # Same guard the per-slice apply performs: a separable footprint is built for one mask strategy and
    # silently means something else under the other.
    ws isa GPUSeparable && CGEF.Filtering._separable_check_strategy(ws.fp, mask_strategy)
    # A 1-D grid has no direction 2, so it cannot be queried unconditionally. The ND engines read the
    # grid's own `periodic_flags` anyway and ignore these two.
    nspatial = length(FlowGeometries.Grids.size_tuple(grid))
    px = FlowGeometries.Grids.isperiodic(grid, 1)
    py = nspatial >= 2 ? FlowGeometries.Grids.isperiodic(grid, 2) : false
    _run_gpu_kernel!(dev, out, field, ws, grid, px, py, mask_strategy isa CGEF.Filtering.ZeroFill)
    KA.synchronize(dev)
    return out
end

function _run_gpu_kernel!(dev, out, field, ws::GPUScatteredNDCached, grid, periodic_x::Bool, periodic_y::Bool, is_zerofill::Bool)
    dims = FlowGeometries.Grids.size_tuple(grid)
    _cgef_filter_kernel_nd_cached!(dev)(
        out, field, ws.maskd, ws.nbrs, ws.w, ws.ptr, dims, is_zerofill;
        ndrange = (prod(dims), size(out, length(dims) + 1)),
    )
end

function _run_gpu_kernel!(dev, out, field, ws::GPUSeparableND, grid, periodic_x::Bool, periodic_y::Bool, is_zerofill::Bool)
    strategy = is_zerofill ? CGEF.Filtering.ZeroFill() : CGEF.Filtering.Deformable()
    CGEF.Filtering.apply_separable_nd!(out, field, grid, ws.fp, strategy, GPUDriver(dev))
end

function _run_gpu_kernel!(dev, out, field, ws::GPUFootprintND, grid, periodic_x::Bool, periodic_y::Bool, is_zerofill::Bool)
    dims = FlowGeometries.Grids.size_tuple(grid)
    # Spatial extent flattened, batch as the second launch axis. `size(out, length(dims)+1)` is 1 for an
    # unbatched output, so this is one launch shape for both cases.
    _cgef_filter_kernel_nd!(dev)(
        out, field, ws.maskd, ws.offsets, ws.w,
        dims, FlowGeometries.Grids.periodic_flags(grid), is_zerofill;
        ndrange = (prod(dims), size(out, length(dims) + 1)),
    )
end

function _run_gpu_kernel!(dev, out, field, ws::GPUNodeFootprint, grid, periodic_x::Bool, periodic_y::Bool, is_zerofill::Bool)
    _cgef_filter_kernel_node!(dev)(
        out, field, ws.maskd, ws.nbrs, ws.w, ws.ptr, is_zerofill;
        ndrange = (size(out, 1), size(out, 2)),
    )
end

function _run_gpu_kernel!(dev, out, field, ws::GPUScatteredCached, grid, periodic_x::Bool, periodic_y::Bool, is_zerofill::Bool)
    _cgef_filter_kernel_scattered!(dev)(
        out, field, ws.maskd, ws.ii, ws.jj, ws.w, ws.ptr, is_zerofill;
        ndrange = (size(out, 1), size(out, 2), size(out, 3)),
    )
end

function _run_gpu_kernel!(dev, out, field, ws::GPUStreaming, grid, periodic_x::Bool, periodic_y::Bool, is_zerofill::Bool)
    fp = ws.fp
    _cgef_filter_kernel_traversal!(dev)(
        out, field, ws.maskd, ws.grid, ws.topology, fp.kernel, fp.scale, fp.rad, fp.is_cartesian,
        is_zerofill; ndrange = (size(out, 1), size(out, 2), size(out, 3)),
    )
end

# Separable Gaussian fast path: two kernel launches (row-pass, then column-pass) separated by a
# `KA.synchronize` — the column pass reads rows other than its own, so they cannot be fused the way
# the disk-truncated kernels are. The mask-strategy branch is already baked into which denominator the
# footprint carries, exactly as on the host.
# The resident footprint's pass buffers are sized for a single slice, and the column pass must read the
# whole row pass, so a batch needs its own intermediates. Two device allocations buy 2 launches instead
# of 2*Nb; the spatial-only mask and denominator broadcast against the trailing batch axis unchanged.
@inline _sep_buffers(dev, ws::GPUSeparable, ::AbstractArray{<:Any,2}) = (ws.masked_input, ws.row_pass)
@inline _sep_buffers(dev, ws::GPUSeparable, out::AbstractArray{T,3}) where {T} =
    (KA.allocate(dev, T, size(out)...), KA.allocate(dev, T, size(out)...))

function _run_gpu_kernel!(dev, out, field, ws::GPUSeparable, grid, periodic_x::Bool, periodic_y::Bool, is_zerofill::Bool)
    T = eltype(out)
    nd = (size(out, 1), size(out, 2), size(out, 3))
    fp = ws.fp
    masked, rowp = _sep_buffers(dev, ws, out)
    masked .= T.(ws.maskd) .* field
    _cgef_separable_row_pass_kernel!(dev)(rowp, masked, ws.gx, fp.di_lim, fp.periodic_x; ndrange = nd)
    KA.synchronize(dev)
    _cgef_separable_column_pass_kernel!(dev)(out, rowp, ws.gy, fp.dj_lim, fp.periodic_y; ndrange = nd)
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

# 1-D and true-3-D structured grids. `N` is unconstrained; the 2-D method above is more specific and
# wins for N=2, exactly as the threaded hooks are arranged.
function CGEF.Filtering.gpu_filter_field!(
    gpu_backend::CGEF.ComputationalBackends.GPUBackend,
    out::AbstractArray{T,N},
    field::AbstractArray{T,N},
    grid::FlowGeometries.Grids.StructuredGrid{G,T,N},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
    workspace,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}, N}
    dev = gpu_backend.backend
    ws = _resident(gpu_backend, grid, workspace, kernel, scale, mask_strategy)
    _run_gpu_kernel!(
        dev, out, field, ws, grid, false, false, mask_strategy isa CGEF.Filtering.ZeroFill,
    )
    KA.synchronize(dev)
    return out
end

# Node sets. Separate from the 2-D method because the output is a vector and the footprint carries its
# own adjacency, so none of the periodic-wrap or window machinery above applies.
function CGEF.Filtering.gpu_filter_field!(
    gpu_backend::CGEF.ComputationalBackends.GPUBackend,
    out::AbstractVector{T},
    field::AbstractVector{T},
    grid::FlowGeometries.Grids.UnstructuredGrid{T},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
    workspace,
) where {T<:AbstractFloat}
    dev = gpu_backend.backend
    ws = _resident(gpu_backend, grid, workspace, kernel, scale, mask_strategy)
    _run_gpu_kernel!(
        dev, out, field, ws, grid, false, false, mask_strategy isa CGEF.Filtering.ZeroFill,
    )
    KA.synchronize(dev)
    return out
end

end # module
