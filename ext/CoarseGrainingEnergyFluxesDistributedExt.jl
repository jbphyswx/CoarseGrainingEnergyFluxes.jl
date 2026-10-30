module CoarseGrainingEnergyFluxesDistributedExt

using Distributed: Distributed
using SharedArrays: SharedArrays
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF
using FlowGeometries: FlowGeometries
using FlowTransformBindings: FlowTransformBindings as FTB

# A real-space apply divided among the worker processes of one machine. The plan sends each worker its
# copy of the footprint and the grid once, held there in a `Future` as the share its inner backend runs,
# with the pass buffers every worker writes in memory the processes share. An apply copies the field into
# the shared field, runs each pass as one contiguous block of its units per worker on that worker's inner
# backend, and copies the shared output out; a pass starts once every worker has finished the one before.
# On a device inner backend a worker's share is one pass on its device (`AbstractProcessShare`), whose
# intermediate buffers stay there.

"""
    DistributedFootprint

A real-space footprint divided among the worker processes: the footprint with its pass buffers in shared
memory, the passes each worker runs, each worker's share with the grid and the inner backend, held on that
worker, and the field and output arrays the processes share.
"""
struct DistributedFootprint{
    FP, PS<:Val, VF<:AbstractVector{Distributed.Future}, VW<:AbstractVector{Int}, S<:SharedArrays.SharedArray,
}
    fp::FP
    passes::PS
    parts::VF
    workers::VW
    field::S
    out::S
end

CGEF.Filtering._host_footprint(ws::DistributedFootprint) = ws.fp

_hold(fp, grid, inner) = (; fp = CGEF.Filtering._share_workspace(inner, grid, fp), grid, inner)
_ready(f) = (fetch(f); nothing)

function CGEF.Filtering.prepare_workspace(
    b::CGEF.ComputationalBackends.DistributedBackend, grid::FlowGeometries.Grids.AbstractGrid{G,T},
    fp::CGEF.Filtering._SharedFootprint,
) where {G<:FlowGeometries.Geometry.AbstractGeometry, T<:AbstractFloat}
    ws = Distributed.workers()
    pids = union([Distributed.myid()], ws)
    inner = CGEF.ComputationalBackends.local_backend(b)
    held = _held(inner, fp, pids)
    parts = [Distributed.remotecall(_hold, w, held, grid, inner) for w in ws]
    # Every copy has arrived, and a worker's failure is raised here.
    @sync for (k, w) in enumerate(ws)
        @async Distributed.remotecall_fetch(_ready, w, parts[k])
    end
    dims = size(FlowGeometries.Grids.mask(grid))
    # The caller holds the shared copy, so its shared buffers live as long as the plan.
    return DistributedFootprint(held, CGEF.Filtering._process_passes(inner, fp), parts, ws, _shared(T, dims, pids),
                                _shared(T, dims, pids))
end

# The footprint the workers receive: its pass buffers shared where the workers' host passes write them; a
# device share copies the footprint to its device and runs every pass but the last there.
_held(_, fp, pids) = _with_shared(fp, pids)
_held(::CGEF.ComputationalBackends.GPUBackend, fp, _) = fp

# A footprint's pass buffers in memory the processes `pids` share; a footprint whose passes write `out`
# alone keeps its own.
_with_shared(fp, _) = fp
_with_shared(fp::CGEF.Filtering.SeparableFootprint, pids) = CGEF.Filtering.SeparableFootprint(
    fp.gx, fp.gy, fp.di_lim, fp.dj_lim, fp.periodic_x, fp.periodic_y, fp.Nx_profile, fp.Ny_profile,
    fp.invrenorm, fp.strategy, fp.masked, fp.bound, _shared_like(fp.masked_input, pids),
    _shared_like(fp.row_pass, pids),
)
_with_shared(fp::CGEF.Filtering.SeparableFootprintND, pids) = CGEF.Filtering.SeparableFootprintND(
    fp.g, fp.lim, fp.periodic, fp.profiles, fp.invrenorm, fp.strategy, fp.masked, fp.bound,
    _shared_like(fp.masked_input, pids), _shared_like(fp.scratch, pids),
)
_with_shared(fp::CGEF.Filtering.PrefixSumTopHatPlan, pids) = CGEF.Filtering.PrefixSumTopHatPlan(
    fp.grid_plan, CGEF.Filtering.PrefixSumScratch([_shared_scan(CGEF.Filtering._prefixsum_num(fp.scratch), pids)],
                                                  _shared_values(fp.scratch.cell_values, pids)),
    fp.rad, fp.dj_lim, fp.uniform_axis, fp.bound, fp.invden, fp.hw, fp.wcell, fp.edge,
)
_with_shared(fp::CGEF.Filtering.PrefixSumTopHat3DPlan, pids) = CGEF.Filtering.PrefixSumTopHat3DPlan(
    CGEF.Filtering.PrefixSum3DScratch(_shared_scan(fp.scratch.prefix, pids), _shared_values(fp.scratch.cell_values, pids)),
    fp.rad, fp.dj_lim, fp.dk_lim, fp.periodic, fp.masked, fp.bound, fp.strategy, fp.wcell, fp.invden, fp.edge,
)

_shared_like(a::AbstractArray{T}, pids) where {T} = (s = _shared(T, size(a), pids); copyto!(s, a); s)

# A prefix-sum scratch's cell values in shared memory, where it holds them: the 2-D scratch's slot 1, the
# 3-D scratch's array.
_shared_values(::Nothing, _) = nothing
_shared_values(v::AbstractVector{<:AbstractMatrix}, pids) = [_shared_like(first(v), pids)]
_shared_values(v::AbstractArray{<:AbstractFloat,3}, pids) = _shared_like(v, pids)

# A compensated scan of `P`'s size in shared memory. Its leading entry along axis 1 is the empty prefix,
# which the fills leave at zero.
_shared_scan(P::CGEF.Filtering.CompensatedScan{T}, pids) where {T} =
    CGEF.Filtering.CompensatedScan(_shared(T, size(P.hi), pids), _shared(T, size(P.lo), pids))

function CGEF.Filtering.distributed_filter_field!(
    out::AbstractArray, field::AbstractArray, grid::FlowGeometries.Grids.AbstractGrid,
    kernel::CGEF.Kernels.AbstractFilterKernel, scale::AbstractFloat,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy, ws::DistributedFootprint,
)
    CGEF.Filtering._check_strategy(ws.fp, mask_strategy)
    copyto!(SharedArrays.sdata(ws.field), field)
    _passes!(ws, mask_strategy, Val(1), ws.passes)
    return copyto!(out, SharedArrays.sdata(ws.out))
end

CGEF.Filtering.distributed_filter_field!(
    out::AbstractArray, field::AbstractArray, grid::FlowGeometries.Grids.AbstractGrid,
    kernel::CGEF.Kernels.AbstractFilterKernel, scale::AbstractFloat,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy, fp,
) = throw(ArgumentError(
    "DistributedBackend divides the real-space footprints among workers; the $(nameof(typeof(fp))) engine " *
    "runs in one process. Name a footprint engine (`engine = FootprintEngine()`) or plan with a local backend."))

_passes!(ws, strategy, ::Val{n}, ::Val{n}) where {n} = _pass!(ws, strategy, Val(n))

function _passes!(ws, strategy, ::Val{p}, ::Val{n}) where {p,n}
    _pass!(ws, strategy, Val(p))
    return _passes!(ws, strategy, Val(p + 1), Val(n))
end

function _pass!(ws::DistributedFootprint, strategy, pass::Val)
    nk = length(ws.workers)
    @sync for (k, w) in enumerate(ws.workers)
        @async Distributed.remotecall_wait(_run_share, w, ws.parts[k], ws.field, ws.out, strategy, pass, k, nk)
    end
    return nothing
end

# On worker `k` of `nk`: its held copy's share of the pass, from and into the shared arrays.
function _run_share(part, field, out, strategy, pass, k, nk)
    h = fetch(part)
    CGEF.Filtering._share_pass!(SharedArrays.sdata(out), SharedArrays.sdata(field), h.grid, h.fp, strategy, pass,
                                k, nk, h.inner)
    return nothing
end

# A result the worker processes write, maps and per-scale values alike, lives in memory every process of
# the machine shares.
CGEF.Pipeline._result_maps(::CGEF.ComputationalBackends.AbstractDistributedBackend, ::Type{T}, dims::Dims) where {T} =
    _shared(T, dims, Int[])
CGEF.Pipeline._result_values(::CGEF.ComputationalBackends.AbstractDistributedBackend, ::Type{T}, dims::Dims) where {T} =
    _shared(T, dims, Int[])

# Runs `f` with FastTransforms on one OpenMP thread. Call it on the worker: a scoped value does not
# cross processes.
_serial_ft(f) = Base.ScopedValues.with(f, FTB.FASTTRANSFORMS_THREADS => 1)

# Each worker sweeps a contiguous block of slices into disjoint views of the shared batched result. The
# result arrays must be backed by a `SharedArray`: a worker's writes to a plain array stay on the worker.
function _batch_block!(batch, u, v, w, grid, valR, ctx, block::UnitRange{Int})
    _serial_ft() do
        CGEF.Pipeline._run_base!(batch, u, v, w, grid, valR, ctx, block)
    end
    return nothing
end

function CGEF.Pipeline.distributed_coarse_grain_batch!(batch, u, v, w, grid, valR, ctx)
    _require_shared(batch, "coarse_grain_batch!")
    n = length(batch.slices)
    pids = Distributed.workers()
    nw = min(n, length(pids))
    @sync for k in 1:nw
        @async Distributed.remotecall_wait(_batch_block!, pids[k], batch, u, v, w, grid, valR, ctx,
                                           CGEF.Filtering._share_block(k, nw, n))
    end
    return batch
end

# Ragged batch across workers: every slice's result must be backed by `SharedArray`s.
function CGEF.Pipeline.distributed_coarse_grain_slices!(results, us, vs, ws, grids, ctx)
    for r in results
        _require_shared(r, "coarse_grain_slices!")
    end
    n = length(results)
    @sync Distributed.@distributed for t in 1:n
        _serial_ft(() -> CGEF.Pipeline.coarse_grain_slice_serial!(results, us, vs, ws, grids, ctx, t))
    end
    return results
end

# A worker writes a result's maps and its per-scale values, and only a `SharedArray` carries those
# writes back to the caller.
function _require_shared(r, fname::AbstractString)
    all(A -> parent(A) isa SharedArrays.SharedArray,
        (r.Π, r.scales, r.cumulative_energy, r.wavenumber, r.filtering_spectrum)) || throw(ArgumentError(
        "DistributedBackend writes $fname's results on the worker processes, so every array of the " *
        "result has to be shared with them; build it with `execution = DistributedBackend()` " *
        "(`CoarseGrainBatchResult(grid, Nscales, batch_size; execution)`), or use " *
        "SerialBackend()/ThreadedBackend()",
    ))
    return nothing
end

# ---------------------------------------------------------------------------
# Spectral filtering of a Cartesian node set by nonuniform FFT, across worker processes
# ---------------------------------------------------------------------------
#
# Each worker holds the NUFFT plan over its block of the points. The analysis `F_k = Σⱼ wⱼ cⱼ exp(−i k⋅xⱼ)`
# is a sum over points, so the workers' partial analyses, one column each of a shared array, sum to the
# whole spectrum. The caller multiplies it by the transfer function; each worker evaluates the result at
# its own points.

"""
    DistributedNUFFTGridPlan

The scale-independent half of a nonuniform-FFT filter plan divided among the worker processes: each
worker's grid plan and buffers, held on that worker, its block of the points, the arrays the workers
and the caller share (the field, the workers' partial spectra, the filtered spectrum and the output,
and the same set with a trailing batch axis when planned for one), `i k_d` along each direction of the
mode array on the caller, the distance every image of a point lies beyond the other points, and the
backend it was built for.
"""
struct DistributedNUFFTGridPlan{T, S<:SharedArrays.SharedArray{T}, P, BT, VD <: AbstractVector{Distributed.Future}, VI <: AbstractVector{<:Integer}, VU <: AbstractVector{UnitRange{Int}}, TP<:Tuple, M<:AbstractArray{Bool}, IK<:Tuple, B<:CGEF.ComputationalBackends.AbstractDistributedBackend} <: CGEF.Filtering.AbstractGridPlan
    parts::VD
    workers::VI
    blocks::VU
    npts::Int
    nb::Int
    topology::TP     # the grid's; a `Bounded` direction is padded past the domain
    reach::T
    mask::M          # the grid's
    field::S
    out::S
    shared::P        # (; partial, spectrum): (mode_size..., nw) and mode_size
    batched::BT      # (; field, out, partial, spectrum) with a batch axis, or nothing
    ik::IK
    execution::B
end

"""
    DistributedNUFFTFilterPlan

A [`DistributedNUFFTGridPlan`](@ref) with the transfer function on the caller and, under `Deformable`,
each worker's inverse local mass and its derivatives along each direction held on that worker (`nothing`
otherwise).
"""
struct DistributedNUFFTFilterPlan{GP<:DistributedNUFFTGridPlan, TR, R, DM, MS<:CGEF.Filtering.AbstractMaskStrategy} <:
       CGEF.Filtering.AbstractFilterPlan
    grid_plan::GP
    transfer::TR
    invrenorm::R
    dmass::DM        # per direction, each worker's `∂_d M`; or nothing
    strategy::MS
end

CGEF.Filtering.series_form(plan::DistributedNUFFTFilterPlan) = CGEF.Filtering._ratio_form(plan.invrenorm)
CGEF.Filtering.plan_strategy(plan::DistributedNUFFTFilterPlan) = plan.strategy
CGEF.Filtering.plan_execution(plan::DistributedNUFFTFilterPlan) = plan.grid_plan.execution
CGEF.Filtering.spectral_scratch(::DistributedNUFFTGridPlan) = nothing

Base.show(io::IO, gp::DistributedNUFFTGridPlan) =
    print(io, "DistributedNUFFTGridPlan(", gp.npts, " points on ", length(gp.workers), " workers)")
Base.show(io::IO, plan::DistributedNUFFTFilterPlan) = print(io, "DistributedNUFFTFilterPlan(", plan.grid_plan, ")")

# ── On each worker ──────────────────────────────────────────────────────────────────────────────────
# A future's value lives on the worker that computed it, so `fetch` there returns the worker's own
# grid plan and buffers.

function _part_plan(nufft, grid, backend, batch, block::UnitRange{Int}, reach)
    gp = CGEF.Filtering._nufft_grid_plan(nufft, grid; execution = backend, batch = batch, points = collect(block),
                                         box = CGEF.Filtering._NodeBox(reach))
    sc = CGEF.Filtering._nufft_scratch(gp)
    T = eltype(gp.weights)
    n = length(block)
    stage = CGEF.Filtering._allocate(backend, T, (n,))
    bstage = gp.batched === nothing ? nothing : CGEF.Filtering._allocate(backend, T, (n, gp.nb))
    return (; gp, sc, stage, bstage, block)
end

_part_modes(f) = FTB.mode_size(fetch(f).gp.plan)
_part_transfer(f, kernel, scale) = Array(CGEF.Filtering._nufft_transfer(fetch(f).gp, kernel, scale))
_part_ik(f) = map(Array, fetch(f).gp.ik)
_part_reach(f) = fetch(f).gp.reach

# A worker's modes into the host memory the shared arrays live in.
_host(A::Array) = A
_host(A::AbstractArray) = Array(A)

# Whether a call carries one field or the plan's batch.
struct _One end
struct _Batch end

# The plan, values, modes and field buffer for one field or for the batch.
_part_bufs(p, ::_Batch) = (p.gp.batched, p.sc.batched.values, p.sc.batched.modes, p.bstage)
_part_bufs(p, ::_One) = (p.gp.plan, p.sc.values, p.sc.modes, p.stage)

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

function _part_analysis!(f, field, partial, k::Int, mode)
    p = fetch(f)
    plan, values, modes, stage = _part_bufs(p, mode)
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

function _part_synthesis!(f, spectrum, out, invrenorm, mode)
    p = fetch(f)
    plan, values, modes, stage = _part_bufs(p, mode)
    copyto!(modes, SharedArrays.sdata(spectrum))
    FTB.nufft_type2!(values, plan, modes)
    invrenorm === nothing ? (stage .= values) : (stage .= values .* fetch(invrenorm))
    _write_block!(out, stage, p.block)
    return nothing
end

# This worker's synthesis of the shared spectrum at its points, kept on the worker: `∂_d M` of the mass.
function _part_values(f, spectrum)
    p = fetch(f)
    copyto!(p.sc.modes, SharedArrays.sdata(spectrum))
    FTB.nufft_type2!(p.sc.values, p.gp.plan, p.sc.modes)
    return copy(p.sc.values)
end

# `∂_d ū` at this worker's points from the shared `i k_d Ĝ F̂`, the quotient rule's `(∂_d N − ū ∂_d M)/M` where
# the plan renormalizes, `ū` read from the shared field.
function _part_derivative!(f, spectrum, ū, out, invrenorm, dmass, mode)
    p = fetch(f)
    plan, values, modes, stage = _part_bufs(p, mode)
    copyto!(modes, SharedArrays.sdata(spectrum))
    FTB.nufft_type2!(values, plan, modes)
    if invrenorm === nothing
        stage .= values
    else
        _read_block!(stage, ū, p.block)
        stage .= (values .- stage .* fetch(dmass)) .* fetch(invrenorm)
    end
    _write_block!(out, stage, p.block)
    return nothing
end

# ── On the caller ───────────────────────────────────────────────────────────────────────────────────

_shared(::Type{T}, dims, pids) where {T} =
    (s = SharedArrays.SharedArray{T}(dims; pids = pids); fill!(s, zero(T)); s)

function CGEF.Filtering.distributed_nufft_grid_plan(
    nufft, grid::FlowGeometries.Grids.UnstructuredGrid{T,G,D}, reach::T;
    execution::CGEF.ComputationalBackends.AbstractDistributedBackend,
    batch::Union{Nothing,Integer} = nothing,
) where {T<:AbstractFloat, G, D}
    ws = Distributed.workers()
    nw = length(ws)
    npts = length(FlowGeometries.Grids.coordinates(grid, 1))
    nw <= npts || throw(ArgumentError("$npts points cannot be divided among $nw workers"))
    inner = CGEF.ComputationalBackends.local_backend(execution)
    blocks = [CGEF.Filtering._share_block(k, nw, npts) for k in 1:nw]
    parts = [Distributed.remotecall(_part_plan, w, nufft, grid, inner, batch, blocks[k], reach)
             for (k, w) in enumerate(ws)]
    # Every build has finished, and a worker's failure is raised here, before any array is shared.
    modes = Vector{NTuple{D,Int}}(undef, nw)
    @sync for (k, w) in enumerate(ws)
        @async modes[k] = Distributed.remotecall_fetch(_part_modes, w, parts[k])
    end
    ms = first(modes)
    pids = union([Distributed.myid()], ws)
    nb = batch === nothing ? 0 : Int(batch)
    shared = (partial = _shared(Complex{T}, (ms..., nw), pids), spectrum = _shared(Complex{T}, ms, pids))
    batched = batch === nothing ? nothing :
        (field = _shared(T, (npts, nb), pids), out = _shared(T, (npts, nb), pids),
         partial = _shared(Complex{T}, (ms..., nb, nw), pids), spectrum = _shared(Complex{T}, (ms..., nb), pids))
    ik = Distributed.remotecall_fetch(_part_ik, first(ws), first(parts))
    held = T(Distributed.remotecall_fetch(_part_reach, first(ws), first(parts)))
    return DistributedNUFFTGridPlan(
        parts, ws, blocks, npts, nb, FlowGeometries.Grids.topology(grid), held, FlowGeometries.Grids.mask(grid),
        _shared(T, (npts,), pids), _shared(T, (npts,), pids), shared, batched, ik, execution,
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
    CGEF.Filtering._check_unbounded_reach(kernel, gp.topology)
    CGEF.Filtering._check_reach(gp.reach, kernel, scale, Val(length(gp.ik)))
    transfer = Distributed.remotecall_fetch(_part_transfer, first(gp.workers), first(gp.parts), kernel, scale)
    invrenorm, dmass = CGEF.Filtering._divides_by_mass(mask_strategy, gp.mask, gp.topology) ?
        _window_mass(gp, transfer) : (nothing, nothing)
    return DistributedNUFFTFilterPlan(gp, transfer, invrenorm, dmass, mask_strategy)
end

CGEF.Filtering.rate_available(::DistributedNUFFTFilterPlan, kernel::CGEF.Kernels.AbstractFilterKernel) =
    CGEF.Kernels.has_transfer_rate(kernel)
function CGEF.Filtering.rate_plan(plan::DistributedNUFFTFilterPlan, kernel::CGEF.Kernels.AbstractFilterKernel,
                                  scale::AbstractFloat)
    gp = plan.grid_plan
    rate = Distributed.remotecall_fetch(_part_transfer, first(gp.workers), first(gp.parts),
                                        CGEF.Kernels.TransferRate(kernel), scale)
    return DistributedNUFFTFilterPlan(gp, rate, plan.invrenorm, plan.dmass, plan.strategy)
end

# Each worker's `1/M` and `∂_d M` at its points, held there, `M = filter(mask)` through the same division of
# the points; the padding beyond a bounded domain is inactive. Every worker reads the shared spectrum
# before the next synthesis overwrites it.
function _window_mass(gp::DistributedNUFFTGridPlan, transfer)
    _each_part(_part_mass!, gp, gp.shared.partial)
    spectrum = SharedArrays.sdata(gp.shared.spectrum)
    M̂ = copy(SharedArrays.sdata(_sum_partial!(gp.shared.spectrum, gp.shared.partial)))
    on_parts(fn) = (fs = [Distributed.remotecall(fn, w, gp.parts[k], gp.shared.spectrum)
                          for (k, w) in enumerate(gp.workers)]; foreach(wait, fs); fs)
    spectrum .= M̂ .* transfer
    invmass = on_parts(_part_invrenorm)
    dmass = map(gp.ik) do ik
        spectrum .= M̂ .* transfer .* ik
        on_parts(_part_values)
    end
    return invmass, dmass
end

_invrenorm(plan::DistributedNUFFTFilterPlan, k::Int) = plan.invrenorm === nothing ? nothing : plan.invrenorm[k]
_dmass(plan::DistributedNUFFTFilterPlan, k::Int, d::Int) = plan.dmass === nothing ? nothing : plan.dmass[d][k]

function _synthesize_parts!(plan::DistributedNUFFTFilterPlan, sh, mode)
    gp = plan.grid_plan
    @sync for (k, w) in enumerate(gp.workers)
        @async Distributed.remotecall_wait(_part_synthesis!, w, gp.parts[k], sh.spectrum, sh.out, _invrenorm(plan, k),
                                           mode)
    end
    return sh.out
end

function _analyze_parts!(plan::DistributedNUFFTFilterPlan, sh, field, mode)
    gp = plan.grid_plan
    copyto!(sh.field, field)
    @sync for (k, w) in enumerate(gp.workers)
        @async Distributed.remotecall_wait(_part_analysis!, w, gp.parts[k], sh.field, sh.partial, k, mode)
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

function _apply!(out, field, plan::DistributedNUFFTFilterPlan, sh, mode)
    size(out) == size(field) || throw(DimensionMismatch("got out $(size(out)) and field $(size(field))"))
    s = SharedArrays.sdata(_analyze_parts!(plan, sh, field, mode))
    s .*= plan.transfer
    return copyto!(out, _synthesize_parts!(plan, sh, mode))
end

CGEF.Filtering.filter_apply!(out::AbstractVector, field::AbstractVector, plan::DistributedNUFFTFilterPlan) =
    _apply!(out, field, plan, _single(plan.grid_plan), _One())

CGEF.Filtering._batched_fields(outs, plan::DistributedNUFFTFilterPlan) =
    plan.grid_plan.batched !== nothing && ndims(first(outs)) == 2

CGEF.Filtering.filter_apply_batched!(out::AbstractMatrix, field::AbstractMatrix, plan::DistributedNUFFTFilterPlan) =
    _apply!(out, field, plan, _batch(plan.grid_plan, field), _Batch())

# Analysis depends on the field alone, so a sweep runs it once and each scale only multiplies by its own
# transfer function and evaluates back to the points.
CGEF.Filtering.analyze_buffer(plan::DistributedNUFFTFilterPlan, ::AbstractVector) =
    similar(SharedArrays.sdata(plan.grid_plan.shared.spectrum))
CGEF.Filtering.analyze_buffer(plan::DistributedNUFFTFilterPlan, field::AbstractMatrix) = CGEF.Filtering._batch_analysis(
    plan.grid_plan.batched, size(field, 2), _ -> plan.grid_plan.nb,
    () -> similar(SharedArrays.sdata(plan.grid_plan.batched.spectrum)),
)

_sh(plan::DistributedNUFFTFilterPlan, A::AbstractVector) = (_single(plan.grid_plan), _One())
_sh(plan::DistributedNUFFTFilterPlan, A::AbstractMatrix) = (_batch(plan.grid_plan, A), _Batch())

function CGEF.Filtering.filter_analyze!(F̂::AbstractArray, field::AbstractVecOrMat, plan::DistributedNUFFTFilterPlan)
    sh, mode = _sh(plan, field)
    return copyto!(F̂, SharedArrays.sdata(_analyze_parts!(plan, sh, field, mode)))
end

function CGEF.Filtering.filter_synthesize!(out::AbstractVecOrMat, F̂::AbstractArray, plan::DistributedNUFFTFilterPlan)
    sh, mode = _sh(plan, out)
    SharedArrays.sdata(sh.spectrum) .= F̂ .* plan.transfer
    return copyto!(out, _synthesize_parts!(plan, sh, mode))
end

# The shared field, free once analyzed, carries `ū` to the workers for the quotient rule.
function CGEF.Filtering.filter_synthesize_derivative!(
    out::AbstractVecOrMat, F̂::AbstractArray, ū::AbstractVecOrMat, plan::DistributedNUFFTFilterPlan, d::Int,
)
    sh, mode = _sh(plan, out)
    gp = plan.grid_plan
    SharedArrays.sdata(sh.spectrum) .= F̂ .* plan.transfer .* gp.ik[d]
    copyto!(sh.field, ū)
    @sync for (k, w) in enumerate(gp.workers)
        @async Distributed.remotecall_wait(_part_derivative!, w, gp.parts[k], sh.spectrum, sh.field, sh.out,
                                           _invrenorm(plan, k), _dmass(plan, k, d), mode)
    end
    return copyto!(out, sh.out)
end

end # module
