module CoarseGrainingEnergyFluxesAbstractFFTsDistributedExt

using AbstractFFTs: AbstractFFTs
using LinearAlgebra: LinearAlgebra as LA
using Distributed: Distributed
using SharedArrays: SharedArrays
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF
using FlowGeometries: FlowGeometries

# FFT spectral filtering on a uniform Cartesian grid, across worker processes. Each worker transforms a
# block of columns along direction 1, then a block of rows of the half spectrum along direction 2,
# multiplies them by the transfer function, and inverts in reverse order. The four unnormalized
# transforms scale the field by `Px·Py`, which the workers' transfer function divides out.

# Each array plans through the AbstractFFTs provider of its own type, with the options that provider's
# extension declares for `nt` threads.
_rfft_plan(A::AbstractArray, nt::Int, options) = AbstractFFTs.plan_rfft(A, 1; CGEF.Filtering._fft_options(A, nt, options)...)
_brfft_plan(A::AbstractArray, n::Int, nt::Int, options) =
    AbstractFFTs.plan_brfft(A, n, 1; CGEF.Filtering._fft_options(A, nt, options)...)
_fft2_plan(A::AbstractArray, nt::Int, options) = AbstractFFTs.plan_fft!(A, 2; CGEF.Filtering._fft_options(A, nt, options)...)
_bfft2_plan(A::AbstractArray, nt::Int, options) = AbstractFFTs.plan_bfft!(A, 2; CGEF.Filtering._fft_options(A, nt, options)...)

# ── On each worker ──────────────────────────────────────────────────────────────────────────────────
# A future's value lives on the worker that computed it, so `fetch` there returns the worker's own
# buffers and transforms.

# One worker's buffers and transforms for `nf` fields: its columns `cols` of the transform grid, their
# half spectrum, and its rows `rows` of the half spectrum across every column, with a host copy of those
# rows where the inner backend's memory is a device's.
function _part(backend, ::Type{T}, P::NTuple{2,Int}, cols::UnitRange{Int}, rows::UnitRange{Int}, nf::Int, options) where {T}
    Px, Py = P
    nt = CGEF.Filtering._library_threads(backend)
    a = CGEF.Filtering._allocate(backend, T, (Px, length(cols) * nf))
    b = CGEF.Filtering._allocate(backend, Complex{T}, (Px ÷ 2 + 1, length(cols) * nf))
    r = CGEF.Filtering._allocate(backend, Complex{T}, (length(rows), Py, nf))
    rh = r isa Array ? r : zeros(Complex{T}, size(r))
    return (; a, b, r, rh, cols, rows, nf, fwd1 = _rfft_plan(a, nt, options), inv1 = _brfft_plan(b, Px, nt, options),
            fwd2 = _fft2_plan(r, nt, options), inv2 = _bfft2_plan(r, nt, options))
end

_parts(backend, ::Type{T}, P, cols, rows, nb::Int, options) where {T} =
    (single = _part(backend, T, P, cols, rows, 1, options),
     batched = nb == 0 ? nothing : _part(backend, T, P, cols, rows, nb, options))

_ready(f) = (fetch(f); nothing)
_hold(backend, x) = CGEF.Filtering._on_backend(backend, x)
# Whether a call carries one field or the plan's batch, and the worker's buffers for it.
struct _One end
struct _Batch end
_pick(f, ::_One) = fetch(f).single
_pick(f, ::_Batch) = fetch(f).batched

# Columns `cols` of every field of the shared `S` to and from the column-major buffer `buf`.
function _read_cols!(buf, S::SharedArrays.SharedArray, cols::UnitRange{Int}, nf::Int)
    s, m, n = SharedArrays.sdata(S), size(S, 1), length(cols)
    for f in 1:nf
        copyto!(buf, (f - 1) * m * n + 1, s, (f - 1) * m * size(S, 2) + (first(cols) - 1) * m + 1, m * n)
    end
    return buf
end

function _write_cols!(S::SharedArrays.SharedArray, buf, cols::UnitRange{Int}, nf::Int)
    s, m, n = SharedArrays.sdata(S), size(S, 1), length(cols)
    for f in 1:nf
        copyto!(s, (f - 1) * m * size(S, 2) + (first(cols) - 1) * m + 1, buf, (f - 1) * m * n + 1, m * n)
    end
    return S
end

function _cols_forward!(f, A, B, mode, _)
    p = _pick(f, mode)
    LA.mul!(p.b, p.fwd1, _read_cols!(p.a, A, p.cols, p.nf))
    _write_cols!(B, p.b, p.cols, p.nf)
    return nothing
end

function _cols_inverse!(f, B, C, mode, _)
    p = _pick(f, mode)
    LA.mul!(p.a, p.inv1, _read_cols!(p.b, B, p.cols, p.nf))
    _write_cols!(C, p.a, p.cols, p.nf)
    return nothing
end

# The rows of the half spectrum along the second direction: forward, and with `transfer` (this worker's
# rows of the scaled transfer function) multiplied and inverted; `forward` and `transfer` select the
# steps.
function _rows!(f, B, transfer, mode, forward::Bool, k::Int)
    p = _pick(f, mode)
    rows_of_B = view(SharedArrays.sdata(B), p.rows, :, 1:p.nf)
    p.rh .= rows_of_B
    p.r === p.rh || copyto!(p.r, p.rh)
    forward && (p.fwd2 * p.r)
    if transfer !== nothing
        p.r .*= fetch(transfer[k])
        p.inv2 * p.r
    end
    p.r === p.rh || copyto!(p.rh, p.r)
    rows_of_B .= p.rh
    return nothing
end

# ── On the caller ───────────────────────────────────────────────────────────────────────────────────

"""
    DistributedFFTGridPlan

The scale-independent half of an FFT filter plan divided among the worker processes: each worker's
buffers and transforms, held on that worker, its blocks of columns and rows, the grid's transform
layout and mask, the shared transform-grid field, half spectrum and output, and the same set with
a trailing batch axis when planned for one, and `i k` along each axis of the half spectrum.
"""
struct DistributedFFTGridPlan{T, L, M, B<:CGEF.ComputationalBackends.AbstractLocalBackend, S, BT, IK<:Tuple} <:
       CGEF.Filtering.AbstractGridPlan
    parts::Vector{Distributed.Future}
    workers::Vector{Int}
    rows::Vector{UnitRange{Int}}
    layout::L        # `_fft_layout(grid, reach)`
    mask::M          # the grid's mask, or nothing on a grid whose mask is `AllActive`
    execution::B     # the workers' inner execution backend
    single::S        # (; A, B, C) shared arrays for one field
    batched::BT      # the same with a trailing batch axis, or nothing
    nb::Int
    ik::IK           # `Filtering._fft_ik(layout)`
end

"""
    DistributedFFTFilterPlan

A [`DistributedFFTGridPlan`](@ref) with each worker's rows of the transfer function, held on that
worker, and the `Deformable` inverse local mass and its derivatives along each axis on the caller
(`nothing` otherwise).
"""
struct DistributedFFTFilterPlan{GP<:DistributedFFTGridPlan, R, DM, MS<:CGEF.Filtering.AbstractMaskStrategy} <:
       CGEF.Filtering.AbstractFilterPlan
    grid_plan::GP
    transfer::Vector{Distributed.Future}
    invrenorm::R
    dmass::DM
    strategy::MS
end

CGEF.Filtering.series_form(plan::DistributedFFTFilterPlan) = CGEF.Filtering._ratio_form(plan.invrenorm)
CGEF.Filtering.plan_strategy(plan::DistributedFFTFilterPlan) = plan.strategy
CGEF.Filtering.plan_execution(plan::DistributedFFTFilterPlan) =
    CGEF.ComputationalBackends.DistributedBackend(plan.grid_plan.execution)
CGEF.Filtering.spectral_scratch(::DistributedFFTGridPlan) = nothing

Base.show(io::IO, gp::DistributedFFTGridPlan) =
    print(io, "DistributedFFTGridPlan(", gp.layout.dims, " on ", length(gp.workers), " workers)")
Base.show(io::IO, plan::DistributedFFTFilterPlan) = print(io, "DistributedFFTFilterPlan(", plan.grid_plan, ")")

_shared(::Type{T}, dims, pids) where {T} =
    (s = SharedArrays.SharedArray{T}(dims; pids = pids); fill!(s, zero(T)); s)

function CGEF.Filtering.distributed_fft_grid_plan(
    grid::FlowGeometries.Grids.StructuredGrid{T}, reach::T;
    execution::CGEF.ComputationalBackends.AbstractDistributedBackend,
    batch::Union{Nothing,Integer} = nothing,
    fft_options::NamedTuple = (;),
) where {T<:AbstractFloat}
    layout = CGEF.Filtering._fft_layout(grid, reach)
    Px, Py = layout.P
    nk = Px ÷ 2 + 1
    # Every worker takes at least one column and one row.
    ws = Distributed.workers()[1:min(Distributed.nworkers(), Py, nk)]
    nw = length(ws)
    inner = CGEF.ComputationalBackends.local_backend(execution)
    nb = batch === nothing ? 0 : Int(batch)
    rows = [CGEF.Filtering._share_block(k, nw, nk) for k in 1:nw]
    parts = [Distributed.remotecall(_parts, w, inner, T, layout.P, CGEF.Filtering._share_block(k, nw, Py), rows[k], nb,
                                    fft_options)
             for (k, w) in enumerate(ws)]
    @sync for (k, w) in enumerate(ws)
        @async Distributed.remotecall_fetch(_ready, w, parts[k])
    end
    pids = union([Distributed.myid()], ws)
    set(nf) = (A = _shared(T, (Px, Py, nf), pids), B = _shared(Complex{T}, (nk, Py, nf), pids),
               C = _shared(T, (Px, Py, nf), pids))
    mask = CGEF.Filtering._plan_mask(FlowGeometries.Grids.mask(grid), identity, CGEF.ComputationalBackends.SerialBackend())
    single = set(1)
    batched = batch === nothing ? nothing : set(nb)
    ik = CGEF.Filtering._fft_ik(layout)
    return DistributedFFTGridPlan{T, typeof(layout), typeof(mask), typeof(inner), typeof(single), typeof(batched),
                                  typeof(ik)}(parts, ws, rows, layout, mask, inner, single, batched, nb, ik)
end

function _each(fn, gp::DistributedFFTGridPlan, args...)
    @sync for (k, w) in enumerate(gp.workers)
        @async Distributed.remotecall_wait(fn, w, gp.parts[k], args..., k)
    end
    return nothing
end

# `mask · field` into the grid's region of the transform-grid field, whose padding stays zero.
function _stage!(A::SharedArrays.SharedArray, field::AbstractArray, gp::DistributedFFTGridPlan)
    Nx, Ny = gp.layout.dims
    region = view(SharedArrays.sdata(A), 1:Nx, 1:Ny, :)
    F = reshape(field, Nx, Ny, :)
    gp.mask === nothing ? (region .= F) : (region .= gp.mask .* F)
    return A
end

function _unstage!(out::AbstractArray, C::SharedArrays.SharedArray, gp::DistributedFFTGridPlan, invrenorm)
    Nx, Ny = gp.layout.dims
    region = view(SharedArrays.sdata(C), 1:Nx, 1:Ny, :)
    O = reshape(out, Nx, Ny, :)
    invrenorm === nothing ? (O .= region) : (O .= region .* invrenorm)
    return out
end

function _filter!(out, field, gp::DistributedFFTGridPlan, sh, transfer, invrenorm, mode)
    _stage!(sh.A, field, gp)
    _each(_cols_forward!, gp, sh.A, sh.B, mode)
    _each(_rows!, gp, sh.B, transfer, mode, true)
    _each(_cols_inverse!, gp, sh.B, sh.C, mode)
    return _unstage!(out, sh.C, gp, invrenorm)
end

function CGEF.Filtering.distributed_fft_filter_plan(
    gp::DistributedFFTGridPlan{T}, kernel::CGEF.Kernels.AbstractFilterKernel, scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
) where {T}
    CGEF.Filtering._check_reach(gp.layout.reach, kernel, scale, Val(2))
    transfer = _held_transfer(gp, kernel, scale)
    invrenorm, dmass = CGEF.Filtering._divides_by_mass(mask_strategy, gp.mask, gp.layout.topology) ?
        _window_mass(gp, transfer, T) : (nothing, nothing)
    return DistributedFFTFilterPlan(gp, transfer, invrenorm, dmass, mask_strategy)
end

# Each worker's rows of the transfer, scaled by the transform's length, held where that worker transforms.
function _held_transfer(gp::DistributedFFTGridPlan{T}, kernel, scale::T) where {T}
    layout = gp.layout
    scaled = CGEF.Filtering._fft_transfer(kernel, layout.topology, layout, scale) ./ T(layout.P[1] * layout.P[2])
    return [Distributed.remotecall(_hold, w, gp.execution, scaled[gp.rows[k], :]) for (k, w) in enumerate(gp.workers)]
end

CGEF.Filtering.rate_available(::DistributedFFTFilterPlan, kernel::CGEF.Kernels.AbstractFilterKernel) =
    CGEF.Kernels.has_transfer_rate(kernel)
CGEF.Filtering.rate_plan(plan::DistributedFFTFilterPlan, kernel::CGEF.Kernels.AbstractFilterKernel, scale::AbstractFloat) =
    DistributedFFTFilterPlan(plan.grid_plan, _held_transfer(plan.grid_plan, CGEF.Kernels.TransferRate(kernel), scale),
                             plan.invrenorm, plan.dmass, plan.strategy)

# `1/M` and `∂M/∂x`, `∂M/∂y` on the caller, `M = filter(mask)` through the same transforms; the padding beyond
# a bounded axis is inactive.
function _window_mass(gp::DistributedFFTGridPlan, transfer, ::Type{T}) where {T}
    dims, sh = gp.layout.dims, gp.single
    _stage!(sh.A, gp.mask === nothing ? ones(T, dims) : T.(gp.mask), gp)
    _each(_cols_forward!, gp, sh.A, sh.B, _One())
    _each(_rows!, gp, sh.B, nothing, _One(), true)
    M̂ = copy(SharedArrays.sdata(sh.B))
    function synth(ik)
        SharedArrays.sdata(sh.B) .= M̂ .* ik
        _each(_rows!, gp, sh.B, transfer, _One(), false)
        _each(_cols_inverse!, gp, sh.B, sh.C, _One())
        return _unstage!(zeros(T, dims), sh.C, gp, nothing)
    end
    M = synth(one(T))
    active = gp.mask === nothing ? trues(dims) : gp.mask
    return ifelse.(active, CGEF.Filtering._inv_mass.(M), zero(T)), map(synth, gp.ik)
end

_shared_for(gp::DistributedFFTGridPlan, ::AbstractMatrix) = (gp.single, _One())
function _shared_for(gp::DistributedFFTGridPlan, A::AbstractArray{<:Any,3})
    gp.batched === nothing && throw(ArgumentError(
        "this spectral plan was not built for a batch; pass `batch = nb` to `plan_filter`"))
    size(A) == (gp.layout.dims..., gp.nb) || throw(DimensionMismatch(
        "the plan was built for $(gp.layout.dims) × a batch of $(gp.nb); got $(size(A))"))
    return (gp.batched, _Batch())
end

function CGEF.Filtering.filter_apply!(out::AbstractMatrix, field::AbstractMatrix, plan::DistributedFFTFilterPlan)
    gp = plan.grid_plan
    size(out) == size(field) == gp.layout.dims || throw(DimensionMismatch(
        "the plan was built for $(gp.layout.dims); got out $(size(out)) and field $(size(field))"))
    return _filter!(out, field, gp, gp.single, plan.transfer, plan.invrenorm, _One())
end

CGEF.Filtering._batched_fields(outs, plan::DistributedFFTFilterPlan) =
    plan.grid_plan.batched !== nothing && ndims(first(outs)) == 3

function CGEF.Filtering.filter_apply_batched!(
    out::AbstractArray{<:Any,3}, field::AbstractArray{<:Any,3}, plan::DistributedFFTFilterPlan,
)
    size(out) == size(field) || throw(DimensionMismatch("got out $(size(out)) and field $(size(field))"))
    sh, mode = _shared_for(plan.grid_plan, field)
    return _filter!(out, field, plan.grid_plan, sh, plan.transfer, plan.invrenorm, mode)
end

# Analysis is scale-independent, so a sweep transforms each field once and then only multiplies by each
# scale's transfer function and inverts.
function CGEF.Filtering.analyze_buffer(plan::DistributedFFTFilterPlan, field::AbstractArray)
    gp = plan.grid_plan
    make() = similar(SharedArrays.sdata(first(_shared_for(gp, field)).B))
    ndims(field) == 3 && return CGEF.Filtering._batch_analysis(gp.batched, size(field, 3), _ -> gp.nb, make)
    return make()
end

function CGEF.Filtering.filter_analyze!(F̂::AbstractArray, field::AbstractArray, plan::DistributedFFTFilterPlan)
    gp = plan.grid_plan
    sh, mode = _shared_for(gp, field)
    _stage!(sh.A, field, gp)
    _each(_cols_forward!, gp, sh.A, sh.B, mode)
    _each(_rows!, gp, sh.B, nothing, mode, true)
    return copyto!(F̂, SharedArrays.sdata(sh.B))
end

function CGEF.Filtering.filter_synthesize!(out::AbstractArray, F̂::AbstractArray, plan::DistributedFFTFilterPlan)
    gp = plan.grid_plan
    sh, mode = _shared_for(gp, out)
    copyto!(SharedArrays.sdata(sh.B), F̂)
    _each(_rows!, gp, sh.B, plan.transfer, mode, false)
    _each(_cols_inverse!, gp, sh.B, sh.C, mode)
    return _unstage!(out, sh.C, gp, plan.invrenorm)
end

# `i k_d` multiplies the half spectrum on the caller, which commutes with the workers' transfer function.
function CGEF.Filtering.filter_synthesize_derivative!(
    out::AbstractArray, F̂::AbstractArray, ū::AbstractArray, plan::DistributedFFTFilterPlan, d::Int,
)
    gp = plan.grid_plan
    sh, mode = _shared_for(gp, out)
    SharedArrays.sdata(sh.B) .= F̂ .* gp.ik[d]
    _each(_rows!, gp, sh.B, plan.transfer, mode, false)
    _each(_cols_inverse!, gp, sh.B, sh.C, mode)
    _unstage!(out, sh.C, gp, nothing)
    return CGEF.Filtering._quotient_rule!(out, ū, plan.invrenorm, plan.dmass, d)
end

end # module
