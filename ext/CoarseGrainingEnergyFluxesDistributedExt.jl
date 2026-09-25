module CoarseGrainingEnergyFluxesDistributedExt

using Distributed: Distributed
using SharedArrays: SharedArrays
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF
using FlowGeometries: FlowGeometries
using FlowTransformBindings: FlowTransformBindings as FTB

# DistributedBackend: build the footprint once, then fill output rows across worker processes into a
# SharedArray — a single shared-memory node, not a multi-node decomposition. Rows write disjoint
# output columns, so the result is identical to serial. With no extra workers the `@distributed` loop
# runs on the caller. Works for any row-decomposable 2D grid, structured or curvilinear.
function CGEF.Filtering.distributed_filter_field!(
    out::AbstractMatrix{T},
    field::AbstractMatrix,
    grid::Union{FlowGeometries.Grids.StructuredGrid{T,G,2}, FlowGeometries.Grids.CurvilinearGrid{T,G}},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
    workspace,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    fp = workspace === nothing ? CGEF.Filtering.build_footprint(grid, kernel, scale; mask_strategy = mask_strategy) : workspace
    CGEF.Filtering._check_strategy(fp, mask_strategy)
    if fp isa CGEF.Filtering.SeparableFootprint
        return _distributed_apply_separable!(out, field, grid, fp)
    elseif fp isa CGEF.Filtering.PrefixSumTopHatPlan
        # The prefix table lives in the plan (ordinary process-local arrays, not a SharedArray), so a
        # `@distributed` loop over rows would build it in the workers' own address spaces and the
        # caller would see nothing. The prefix pass is O(N) — negligible against the O(N·dj_lim) apply
        # — so run this path locally in full rather than pretending to distribute it.
        return CGEF.Filtering.apply_prefixsum_tophat!(out, field, grid, fp, mask_strategy)
    end
    periodic_x = FlowGeometries.Grids.isperiodic(grid, 1)
    periodic_y = FlowGeometries.Grids.isperiodic(grid, 2)
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    s_out = SharedArrays.SharedArray{T}(Nx, Ny)
    fill!(s_out, zero(T))
    CGEF.Filtering.prepare_row_apply!(fp, field, grid)
    @sync Distributed.@distributed for j in 1:Ny
        CGEF.Filtering.apply_footprint_row!(s_out, field, grid, fp, mask_strategy, periodic_x, periodic_y, j)
    end
    copyto!(out, s_out)
    return out
end

# Separable Gaussian. `masked_input` is copied into a `SharedArray` before either loop starts, so the
# row pass needs no communication — each worker's rows read only their own column. The column pass is
# then row-parallel against the completed `row_pass` SharedArray. Both call the same per-row/per-column
# bodies as serial, so results are bit-identical.
function _distributed_apply_separable!(
    out::AbstractMatrix{T}, field::AbstractMatrix, grid::FlowGeometries.Grids.StructuredGrid, fp::CGEF.Filtering.SeparableFootprint{T},
) where {T<:AbstractFloat}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    mask = FlowGeometries.Grids.mask(grid)
    gx, gy = fp.gx, fp.gy
    di_lim, dj_lim = fp.di_lim, fp.dj_lim
    periodic_x, periodic_y = fp.periodic_x, fp.periodic_y

    masked_input = SharedArrays.SharedArray{T}(Nx, Ny)
    masked_input .= mask .* field   # fused into the SharedArray; `copyto!` would materialize a temporary
    row_pass = SharedArrays.SharedArray{T}(Nx, Ny)
    @sync Distributed.@distributed for j in 1:Ny
        CGEF.Filtering._separable_row_pass_at!(row_pass, masked_input, gx, di_lim, periodic_x, Nx, j)
    end

    s_out = SharedArrays.SharedArray{T}(Nx, Ny)
    @sync Distributed.@distributed for j in 1:Ny
        CGEF.Filtering._separable_column_pass_at!(s_out, row_pass, gy, dj_lim, periodic_y, Nx, Ny, j)
    end
    CGEF.Filtering._separable_normalize_and_mask!(s_out, fp, mask, Nx, Ny)
    copyto!(out, s_out)
    return out
end


# Distributed analogue of the driver the serial and threaded backends pass to
# `apply_separable_nd!`, so the `N`-pass engine has one implementation across all three.
@inline function _dist_driver(f::F, indices) where {F}
    @sync Distributed.@distributed for i in eachindex(indices)
        f(indices[i])
    end
    return nothing
end

_shared_like(a::AbstractArray{T}) where {T} =
    (sh = SharedArrays.SharedArray{T}(size(a)); copyto!(sh, a); sh)

# 1-D and true-3-D grids. `N` is unconstrained, so the 2-D method above is more specific and still wins
# for N=2. Point-indexed footprints decompose over linear indices; the separable engine instead has
# its plan-local pass buffers REBUILT as SharedArrays, because a `@distributed` loop writing the plan's
# own arrays would write them in the workers' address spaces and the caller would see nothing.
function CGEF.Filtering.distributed_filter_field!(
    out::AbstractArray{T,N},
    field::AbstractArray,
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
    workspace,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}, N}
    fp = workspace === nothing ? CGEF.Filtering.build_footprint(grid, kernel, scale; mask_strategy = mask_strategy) : workspace
    CGEF.Filtering._check_strategy(fp, mask_strategy)
    dims = FlowGeometries.Grids.size_tuple(grid)
    mask = FlowGeometries.Grids.mask(grid)

    if fp isa CGEF.Filtering.PrefixSumTopHat3DPlan
        # A running scan along axis 1 is not a per-point-parallel decomposition, and the plan's buffers
        # are local arrays. Run it in full locally, as the 2-D prefix-sum path does: at O(N·w_y·w_z)
        # against the distributed ball walk's O(N·w³) it is still the faster path.
        return CGEF.Filtering.apply_prefixsum_tophat_3d!(out, field, grid, fp, mask_strategy)
    end

    if fp isa CGEF.Filtering.SeparableFootprintND
        sfp = CGEF.Filtering.SeparableFootprintND(
            fp.g, fp.lim, fp.periodic, fp.profiles, fp.invrenorm, fp.strategy, fp.masked, fp.bound,
            _shared_like(fp.masked_input), _shared_like(fp.scratch),
        )
        s_out = SharedArrays.SharedArray{T}(dims)
        CGEF.Filtering.apply_separable_nd!(s_out, field, grid, sfp, mask_strategy, _dist_driver)
        copyto!(out, s_out)
        return out
    end

    s_out = SharedArrays.SharedArray{T}(dims)
    fill!(s_out, zero(T))
    cart = CartesianIndices(dims)
    # The point functions decide which targets the strategy filters.
    if fp isa CGEF.Filtering.FilterFootprintND
        periodic = FlowGeometries.Grids.periodic_flags(grid)
        @sync Distributed.@distributed for lin in 1:length(s_out)
            I = cart[lin]
            s_out[I] = CGEF.Filtering._footprint_nd_point(field, fp, mask_strategy, dims, periodic, mask, I)
        end
    elseif fp.cache !== nothing
        lin_idx = LinearIndices(dims)
        cache = fp.cache
        exterior = fp.exterior
        @sync Distributed.@distributed for lin in 1:length(s_out)
            I = cart[lin]
            s_out[I] = CGEF.Filtering._footprint_nd_point_cached(field, cache, exterior, mask_strategy, mask, lin_idx, I)
        end
    else
        @sync Distributed.@distributed for lin in 1:length(s_out)
            I = cart[lin]
            s_out[I] = CGEF.Filtering._footprint_nd_point_streaming(field, grid, fp, mask_strategy, mask, I)
        end
    end
    copyto!(out, s_out)
    return out
end

# Node sets. A node grid's real-space footprint is always a `NodeFilterPlan`, so unlike the 2-D method
# there is no separable or prefix-sum variant to branch on. Nodes carry no row structure, so the
# decomposition is over `eachindex(out)` directly; node `t` writes only `out[t]`, so the SharedArray
# needs no reduction and the result is identical to serial.
function CGEF.Filtering.distributed_filter_field!(
    out::AbstractVector{T},
    field::AbstractVector,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
    fp::CGEF.Filtering.NodeFilterPlan,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    n = length(out)
    s_out = SharedArrays.SharedArray{T}(n)
    fill!(s_out, zero(T))
    @sync Distributed.@distributed for t in 1:n
        s_out[t] = CGEF.Filtering._footprint_node_point(field, grid, fp, mask_strategy, t)
    end
    copyto!(out, s_out)
    return out
end

CGEF.Pipeline.batch_alloc_shared(::Type{T}, dims::Integer...) where {T} =
    (sh = SharedArrays.SharedArray{T}(dims); fill!(sh, zero(T)); sh)

# A sweep on a worker runs FastTransforms on one OpenMP thread there, the processes carrying the
# parallelism. The scope is set inside the loop body because a scoped value does not cross processes.
_serial_ft(f) = Base.ScopedValues.with(f, FTB.FASTTRANSFORMS_THREADS => 1)

# Batch-parallel PIPELINE sweeps across worker processes — one shared-memory node, not a multi-node
# decomposition. Slices write disjoint views of the batched result, so no synchronization is needed and
# the assembled result is identical to serial.
#
# The storage must be shared: a `@distributed` loop over plain arrays would write them in the workers'
# own address spaces and the caller would see zeros. That is a hard error rather than a silent copy.
function CGEF.Pipeline.distributed_coarse_grain_batch!(batch, u, v, w, grid, valR, ctx)
    _require_shared(batch.Π, "coarse_grain_batch!")
    n = length(batch.slices)
    @sync Distributed.@distributed for t in 1:n
        _serial_ft(() -> CGEF.Pipeline.coarse_grain_batch_slice!(batch, u, v, w, grid, valR, ctx, t, 1))
    end
    return batch
end

# Ragged batch across workers. Shapes differ per slice, so each slice owns its own result and there is
# no shared batched tensor to write into — every result's storage must be shared instead.
function CGEF.Pipeline.distributed_coarse_grain_slices!(results, us, vs, ws, grids, ctx)
    for r in results
        _require_shared(r.Π, "coarse_grain_slices!")
    end
    n = length(results)
    @sync Distributed.@distributed for t in 1:n
        _serial_ft(() -> CGEF.Pipeline.coarse_grain_slice_serial!(results, us, vs, ws, grids, ctx, t))
    end
    return results
end

function _require_shared(A, fname::AbstractString)
    parent(A) isa SharedArrays.SharedArray || throw(ArgumentError(
        "DistributedBackend needs shared result storage for $fname; allocate it with " *
        "`alloc = CoarseGrainingEnergyFluxes.Pipeline.batch_alloc_shared`, or use SerialBackend()/ThreadedBackend()",
    ))
    return nothing
end

# ---------------------------------------------------------------------------
# Spectral filtering of a Cartesian node set by nonuniform FFT, across worker processes
# ---------------------------------------------------------------------------
#
# Each worker holds the library plan over its block of the points in its own memory. The analysis
# `F_k = Σⱼ wⱼ cⱼ exp(−i k⋅xⱼ)` is a sum over points, so the workers' partial analyses, written into their
# columns of a shared array, add to the whole spectrum. The caller multiplies it by the transfer
# function, and each worker evaluates the filtered series at its own points into the shared output.
# A worker reads and writes its block through buffers in its inner backend's memory, one copy per field.

# The block of `1:n` the `k`-th of `nw` workers transforms.
_block(k::Int, nw::Int, n::Int) = (div((k - 1) * n, nw) + 1):div(k * n, nw)

"""
    DistributedNUFFTGridPlan

The scale-independent half of a nonuniform-FFT filter plan divided among the worker processes: each
worker's grid plan and buffers, held on that worker, its block of the points, and the arrays the
workers and the caller share: the field, the workers' partial spectra, the filtered spectrum and the
output, and the same set with a trailing batch axis when planned for one.
"""
struct DistributedNUFFTGridPlan{T, S<:SharedArrays.SharedArray{T}, P, BT, VD <: AbstractVector{Distributed.Future}, VI <: AbstractVector{<:Integer}, VU <: AbstractVector{UnitRange{Int}}} <: CGEF.Filtering.AbstractGridPlan
    parts::VD
    workers::VI
    blocks::VU
    npts::Int
    nb::Int
    bounded::Bool
    masked::Bool
    field::S
    out::S
    shared::P        # (; partial, spectrum): (mode_size..., nw) and mode_size
    batched::BT      # (; field, out, partial, spectrum) with a batch axis, or nothing
end

"""
    DistributedNUFFTFilterPlan

A [`DistributedNUFFTGridPlan`](@ref) with the transfer function on the caller and, under `Deformable`,
each worker's inverse local mass held on that worker (`nothing` otherwise).
"""
struct DistributedNUFFTFilterPlan{GP<:DistributedNUFFTGridPlan, TR, R, MS<:CGEF.Filtering.AbstractMaskStrategy} <:
       CGEF.Filtering.AbstractFilterPlan
    grid_plan::GP
    transfer::TR
    invrenorm::R
    strategy::MS
end

CGEF.Filtering.plan_strategy(plan::DistributedNUFFTFilterPlan) = plan.strategy
CGEF.Filtering.spectral_scratch(::DistributedNUFFTGridPlan) = nothing

Base.show(io::IO, gp::DistributedNUFFTGridPlan) =
    print(io, "DistributedNUFFTGridPlan(", gp.npts, " points on ", length(gp.workers), " workers)")
Base.show(io::IO, plan::DistributedNUFFTFilterPlan) = print(io, "DistributedNUFFTFilterPlan(", plan.grid_plan, ")")

# ── On each worker ──────────────────────────────────────────────────────────────────────────────────
# A future's value lives on the worker that computed it, so `fetch` there returns the worker's own
# grid plan and buffers.

function _part_plan(nufft, grid, backend, batch, block::UnitRange{Int})
    gp = CGEF.Filtering._nufft_grid_plan(nufft, grid; backend = backend, batch = batch, points = collect(block))
    sc = CGEF.Filtering._nufft_scratch(gp)
    T = eltype(gp.weights)
    n = length(block)
    stage = CGEF.Filtering._allocate(backend, T, (n,))
    bstage = gp.batched === nothing ? nothing : CGEF.Filtering._allocate(backend, T, (n, gp.nb))
    return (; gp, sc, stage, bstage, block)
end

_part_info(f) = (p = fetch(f); (FTB.mode_size(p.gp.plan), p.gp.bounded))
_part_transfer(f, kernel, scale) = Array(CGEF.Filtering._nufft_transfer(fetch(f).gp, kernel, scale))

# A worker's modes into the host memory the shared arrays live in.
_host(A::Array) = A
_host(A::AbstractArray) = Array(A)

# The plan, values, modes and field buffer for one field or for the batch.
_part_bufs(p, batched::Bool) = batched ? (p.gp.batched, p.sc.batched.values, p.sc.batched.modes, p.bstage) :
                                         (p.gp.plan, p.sc.values, p.sc.modes, p.stage)

# This worker's block of the shared `src` into `dst`, column by column.
function _read_block!(dst, src::SharedArrays.SharedArray, block::UnitRange{Int})
    s, n, N = SharedArrays.sdata(src), length(block), size(src, 1)
    for b in 1:size(src, 2)
        copyto!(dst, (b - 1) * n + 1, s, (b - 1) * N + first(block), n)
    end
    return dst
end

function _write_block!(dst::SharedArrays.SharedArray, src, block::UnitRange{Int})
    d, n, N = SharedArrays.sdata(dst), length(block), size(dst, 1)
    for b in 1:size(dst, 2)
        copyto!(d, (b - 1) * N + first(block), src, (b - 1) * n + 1, n)
    end
    return dst
end

function _part_analysis!(f, field, partial, k::Int, batched::Bool)
    p = fetch(f)
    plan, values, modes, stage = _part_bufs(p, batched)
    CGEF.Filtering._load_weighted!(values, _read_block!(stage, field, p.block), p.gp)
    FTB.nufft_type1!(modes, plan, values)
    copyto!(selectdim(SharedArrays.sdata(partial), ndims(partial), k), _host(modes))
    return nothing
end

# The analysis of the quadrature weights over the active points, for `Deformable`'s local mass.
function _part_mass!(f, partial, k::Int)
    p = fetch(f)
    gp, sc = p.gp, p.sc
    gp.mask === nothing ? (sc.values .= gp.weights) : (@. sc.values = gp.mask * gp.weights)
    FTB.nufft_type1!(sc.modes, gp.plan, sc.values)
    copyto!(selectdim(SharedArrays.sdata(partial), ndims(partial), k), _host(sc.modes))
    return nothing
end

# This worker's inverse local mass, from the filtered spectrum of the weights; it stays on the worker.
function _part_invrenorm(f, spectrum)
    p = fetch(f)
    gp, sc = p.gp, p.sc
    copyto!(sc.modes, SharedArrays.sdata(spectrum))
    FTB.nufft_type2!(sc.values, gp.plan, sc.modes)
    return gp.mask === nothing ? CGEF.Filtering._inv_mass.(sc.values) :
           ifelse.(gp.mask, CGEF.Filtering._inv_mass.(sc.values), zero(eltype(sc.values)))
end

function _part_synthesis!(f, spectrum, out, invrenorm, batched::Bool)
    p = fetch(f)
    plan, values, modes, stage = _part_bufs(p, batched)
    copyto!(modes, SharedArrays.sdata(spectrum))
    FTB.nufft_type2!(values, plan, modes)
    invrenorm === nothing ? (stage .= values) : (stage .= values .* fetch(invrenorm))
    _write_block!(out, stage, p.block)
    return nothing
end

# ── On the caller ───────────────────────────────────────────────────────────────────────────────────

_shared(::Type{T}, dims, pids) where {T} =
    (s = SharedArrays.SharedArray{T}(dims; pids = pids); fill!(s, zero(T)); s)

function CGEF.Filtering.distributed_nufft_grid_plan(
    nufft, grid::FlowGeometries.Grids.UnstructuredGrid{T};
    backend::CGEF.ComputationalBackends.AbstractDistributedBackend,
    batch::Union{Nothing,Integer} = nothing,
) where {T<:AbstractFloat}
    ws = Distributed.workers()
    nw = length(ws)
    npts = length(FlowGeometries.Grids.coordinates(grid, 1))
    nw <= npts || throw(ArgumentError("$npts points cannot be divided among $nw workers"))
    inner = CGEF.ComputationalBackends.local_backend(backend)
    blocks = [_block(k, nw, npts) for k in 1:nw]
    parts = [Distributed.remotecall(_part_plan, w, nufft, grid, inner, batch, blocks[k]) for (k, w) in enumerate(ws)]
    # Every build has finished, and a worker's failure is raised here, before any array is shared.
    infos = Vector{Any}(undef, nw)
    @sync for (k, w) in enumerate(ws)
        @async infos[k] = Distributed.remotecall_fetch(_part_info, w, parts[k])
    end
    ms, bounded = first(infos)
    pids = union([Distributed.myid()], ws)
    nb = batch === nothing ? 0 : Int(batch)
    shared = (partial = _shared(Complex{T}, (ms..., nw), pids), spectrum = _shared(Complex{T}, ms, pids))
    batched = nb == 0 ? nothing :
        (field = _shared(T, (npts, nb), pids), out = _shared(T, (npts, nb), pids),
         partial = _shared(Complex{T}, (ms..., nb, nw), pids), spectrum = _shared(Complex{T}, (ms..., nb), pids))
    return DistributedNUFFTGridPlan(
        parts, ws, blocks, npts, nb, bounded, !all(FlowGeometries.Grids.mask(grid)),
        _shared(T, (npts,), pids), _shared(T, (npts,), pids), shared, batched,
    )
end

# One call per worker, concurrently.
function _each_part(fn, gp::DistributedNUFFTGridPlan, args...)
    @sync for (k, w) in enumerate(gp.workers)
        @async Distributed.remotecall_wait(fn, w, gp.parts[k], args..., k)
    end
    return nothing
end

# The whole spectrum: the sum of the workers' partial analyses.
function _sum_partial!(spectrum, partial)
    s = SharedArrays.sdata(spectrum)
    sum!(reshape(s, size(s)..., 1), SharedArrays.sdata(partial))
    return spectrum
end

function CGEF.Filtering.distributed_nufft_filter_plan(
    gp::DistributedNUFFTGridPlan{T}, kernel::CGEF.Kernels.AbstractFilterKernel, scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
) where {T}
    transfer = Distributed.remotecall_fetch(_part_transfer, first(gp.workers), first(gp.parts), kernel, scale)
    invrenorm = if mask_strategy isa CGEF.Filtering.Deformable && (gp.masked || gp.bounded)
        # `filter(mask)` through the same division of the points; the box beyond a bounded record is
        # inactive.
        _each_part(_part_mass!, gp, gp.shared.partial)
        SharedArrays.sdata(_sum_partial!(gp.shared.spectrum, gp.shared.partial)) .*= transfer
        [Distributed.remotecall(_part_invrenorm, w, gp.parts[k], gp.shared.spectrum) for (k, w) in enumerate(gp.workers)]
    else
        nothing
    end
    return DistributedNUFFTFilterPlan(gp, transfer, invrenorm, mask_strategy)
end

_invrenorm(plan::DistributedNUFFTFilterPlan, k::Int) = plan.invrenorm === nothing ? nothing : plan.invrenorm[k]

function _synthesize_parts!(plan::DistributedNUFFTFilterPlan, sh, batched::Bool)
    gp = plan.grid_plan
    @sync for (k, w) in enumerate(gp.workers)
        @async Distributed.remotecall_wait(_part_synthesis!, w, gp.parts[k], sh.spectrum, sh.out, _invrenorm(plan, k),
                                           batched)
    end
    return sh.out
end

function _analyze_parts!(plan::DistributedNUFFTFilterPlan, sh, field, batched::Bool)
    gp = plan.grid_plan
    copyto!(sh.field, field)
    @sync for (k, w) in enumerate(gp.workers)
        @async Distributed.remotecall_wait(_part_analysis!, w, gp.parts[k], sh.field, sh.partial, k, batched)
    end
    return _sum_partial!(sh.spectrum, sh.partial)
end

_single(gp::DistributedNUFFTGridPlan) =
    (field = gp.field, out = gp.out, partial = gp.shared.partial, spectrum = gp.shared.spectrum)

function _batch(gp::DistributedNUFFTGridPlan, A::AbstractMatrix)
    gp.batched === nothing && throw(ArgumentError(
        "this spectral plan was not built for a batch; pass `batch = nb` to `plan_filter`"))
    size(A) == (gp.npts, gp.nb) || throw(DimensionMismatch(
        "the plan was built for $(gp.npts) points × a batch of $(gp.nb); got $(size(A))"))
    return gp.batched
end

function _apply!(out, field, plan::DistributedNUFFTFilterPlan, sh, batched::Bool)
    size(out) == size(field) || throw(DimensionMismatch("got out $(size(out)) and field $(size(field))"))
    s = SharedArrays.sdata(_analyze_parts!(plan, sh, field, batched))
    s .*= plan.transfer
    return copyto!(out, _synthesize_parts!(plan, sh, batched))
end

CGEF.Filtering.filter_apply!(out::AbstractVector, field::AbstractVector, plan::DistributedNUFFTFilterPlan) =
    _apply!(out, field, plan, _single(plan.grid_plan), false)

CGEF.Filtering._batched_fields(outs, plan::DistributedNUFFTFilterPlan) =
    plan.grid_plan.batched !== nothing && ndims(first(outs)) == 2

CGEF.Filtering.filter_apply_batched!(out::AbstractMatrix, field::AbstractMatrix, plan::DistributedNUFFTFilterPlan) =
    _apply!(out, field, plan, _batch(plan.grid_plan, field), true)

# Analysis depends on the field alone, so a sweep runs it once and each scale only multiplies by its own
# transfer function and evaluates back to the points.
CGEF.Filtering.analyze_buffer(plan::DistributedNUFFTFilterPlan, ::AbstractVector) =
    similar(SharedArrays.sdata(plan.grid_plan.shared.spectrum))
CGEF.Filtering.analyze_buffer(plan::DistributedNUFFTFilterPlan, field::AbstractMatrix) =
    (plan.grid_plan.batched === nothing || size(field) != (plan.grid_plan.npts, plan.grid_plan.nb)) ? nothing :
        similar(SharedArrays.sdata(plan.grid_plan.batched.spectrum))

_sh(plan::DistributedNUFFTFilterPlan, A::AbstractVector) = (_single(plan.grid_plan), false)
_sh(plan::DistributedNUFFTFilterPlan, A::AbstractMatrix) = (_batch(plan.grid_plan, A), true)

function CGEF.Filtering.filter_analyze!(F̂::AbstractArray, field::AbstractVecOrMat, plan::DistributedNUFFTFilterPlan)
    sh, batched = _sh(plan, field)
    return copyto!(F̂, SharedArrays.sdata(_analyze_parts!(plan, sh, field, batched)))
end

function CGEF.Filtering.filter_synthesize!(out::AbstractVecOrMat, F̂::AbstractArray, plan::DistributedNUFFTFilterPlan)
    sh, batched = _sh(plan, out)
    SharedArrays.sdata(sh.spectrum) .= F̂ .* plan.transfer
    return copyto!(out, _synthesize_parts!(plan, sh, batched))
end

end # module
