module CoarseGrainingEnergyFluxesFFTWDistributedExt

using FFTW: FFTW
using LinearAlgebra: LinearAlgebra as LA
using Distributed: Distributed
using SharedArrays: SharedArrays
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF
using FlowGeometries: FlowGeometries

# FFT spectral filtering on a uniform Cartesian grid, across worker processes. The two-direction
# real-to-complex transform is a transform along each direction in turn: each worker transforms a block
# of the columns along the first direction, then a block of the rows of the half spectrum along the
# second, multiplies them by the transfer function, and inverts in the reverse order. The transform-grid
# field, its half spectrum and the output are shared arrays; a worker copies its block into buffers in its
# inner backend's memory, where its own transforms run. The four unnormalized transforms scale the field
# by `Px·Py`, which the workers' transfer function divides out.

# The block of `1:n` the `k`-th of `nw` workers transforms.
_block(k::Int, nw::Int, n::Int) = (div((k - 1) * n, nw) + 1):div(k * n, nw)

# FFTW plans a host array at a planner thread count; any other array plans through the AbstractFFTs
# provider of its own type.
_rfft_plan(A::Array, nt::Int) = FFTW.plan_rfft(A, 1; num_threads = nt)
_rfft_plan(A::AbstractArray, ::Int) = FFTW.plan_rfft(A, 1)
_brfft_plan(A::Array, n::Int, nt::Int) = FFTW.plan_brfft(A, n, 1; num_threads = nt)
_brfft_plan(A::AbstractArray, n::Int, ::Int) = FFTW.plan_brfft(A, n, 1)
_fft2_plan(A::Array, nt::Int) = FFTW.plan_fft!(A, 2; num_threads = nt)
_fft2_plan(A::AbstractArray, ::Int) = FFTW.plan_fft!(A, 2)
_bfft2_plan(A::Array, nt::Int) = FFTW.plan_bfft!(A, 2; num_threads = nt)
_bfft2_plan(A::AbstractArray, ::Int) = FFTW.plan_bfft!(A, 2)

# ── On each worker ──────────────────────────────────────────────────────────────────────────────────
# A future's value lives on the worker that computed it, so `fetch` there returns the worker's own
# buffers and transforms.

# One worker's buffers and transforms for `nf` fields: its columns `cols` of the transform grid, their
# half spectrum, and its rows `rows` of the half spectrum across every column, with a host copy of those
# rows where the inner backend's memory is a device's.
function _part(backend, ::Type{T}, P::NTuple{2,Int}, cols::UnitRange{Int}, rows::UnitRange{Int}, nf::Int) where {T}
    Px, Py = P
    nt = CGEF.Filtering._library_threads(backend)
    a = CGEF.Filtering._allocate(backend, T, (Px, length(cols) * nf))
    b = CGEF.Filtering._allocate(backend, Complex{T}, (Px ÷ 2 + 1, length(cols) * nf))
    r = CGEF.Filtering._allocate(backend, Complex{T}, (length(rows), Py, nf))
    rh = r isa Array ? r : zeros(Complex{T}, size(r))
    return (; a, b, r, rh, cols, rows, nf, fwd1 = _rfft_plan(a, nt), inv1 = _brfft_plan(b, Px, nt),
            fwd2 = _fft2_plan(r, nt), inv2 = _bfft2_plan(r, nt))
end

_parts(backend, ::Type{T}, P, cols, rows, nb::Int) where {T} =
    (single = _part(backend, T, P, cols, rows, 1), batched = nb == 0 ? nothing : _part(backend, T, P, cols, rows, nb))

_ready(f) = (fetch(f); nothing)
_hold(backend, x) = CGEF.Filtering._on_backend(backend, x)
_pick(f, batched::Bool) = (p = fetch(f); batched ? p.batched : p.single)

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

function _cols_forward!(f, A, B, batched::Bool, _)
    p = _pick(f, batched)
    LA.mul!(p.b, p.fwd1, _read_cols!(p.a, A, p.cols, p.nf))
    _write_cols!(B, p.b, p.cols, p.nf)
    return nothing
end

function _cols_inverse!(f, B, C, batched::Bool, _)
    p = _pick(f, batched)
    LA.mul!(p.a, p.inv1, _read_cols!(p.b, B, p.cols, p.nf))
    _write_cols!(C, p.a, p.cols, p.nf)
    return nothing
end

# The rows of the half spectrum along the second direction: forward, and with `transfer` (this worker's
# rows of the scaled transfer function) multiplied and inverted; `forward` and `transfer` select the
# steps.
function _rows!(f, B, transfer, batched::Bool, forward::Bool, k::Int)
    p = _pick(f, batched)
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
layout and mask, and the shared transform-grid field, half spectrum and output, and the same set with
a trailing batch axis when planned for one.
"""
struct DistributedFFTGridPlan{T, L, M, B<:CGEF.ComputationalBackends.AbstractLocalBackend, S, BT} <:
       CGEF.Filtering.AbstractGridPlan
    parts::Vector{Distributed.Future}
    workers::Vector{Int}
    rows::Vector{UnitRange{Int}}
    layout::L        # `_fft_layout(grid)`
    mask::M          # the grid's mask, or nothing when fully active
    backend::B       # the workers' inner backend
    single::S        # (; A, B, C) shared arrays for one field
    batched::BT      # the same with a trailing batch axis, or nothing
    nb::Int
end

"""
    DistributedFFTFilterPlan

A [`DistributedFFTGridPlan`](@ref) with each worker's rows of the transfer function, held on that
worker, and the `Deformable` inverse local mass on the caller (`nothing` otherwise).
"""
struct DistributedFFTFilterPlan{GP<:DistributedFFTGridPlan, R, MS<:CGEF.Filtering.AbstractMaskStrategy} <:
       CGEF.Filtering.AbstractFilterPlan
    grid_plan::GP
    transfer::Vector{Distributed.Future}
    invrenorm::R
    strategy::MS
end

CGEF.Filtering.plan_strategy(plan::DistributedFFTFilterPlan) = plan.strategy
CGEF.Filtering.spectral_scratch(::DistributedFFTGridPlan) = nothing

Base.show(io::IO, gp::DistributedFFTGridPlan) =
    print(io, "DistributedFFTGridPlan(", gp.layout.dims, " on ", length(gp.workers), " workers)")
Base.show(io::IO, plan::DistributedFFTFilterPlan) = print(io, "DistributedFFTFilterPlan(", plan.grid_plan, ")")

_shared(::Type{T}, dims, pids) where {T} =
    (s = SharedArrays.SharedArray{T}(dims; pids = pids); fill!(s, zero(T)); s)

function CGEF.Filtering.distributed_fft_grid_plan(
    grid::FlowGeometries.Grids.StructuredGrid{T};
    backend::CGEF.ComputationalBackends.AbstractDistributedBackend,
    batch::Union{Nothing,Integer} = nothing,
) where {T<:AbstractFloat}
    layout = CGEF.Filtering._fft_layout(grid)
    Px, Py = layout.P
    nk = Px ÷ 2 + 1
    # Every worker takes at least one column and one row.
    ws = Distributed.workers()[1:min(Distributed.nworkers(), Py, nk)]
    nw = length(ws)
    inner = CGEF.ComputationalBackends.local_backend(backend)
    nb = batch === nothing ? 0 : Int(batch)
    rows = [_block(k, nw, nk) for k in 1:nw]
    parts = [Distributed.remotecall(_parts, w, inner, T, layout.P, _block(k, nw, Py), rows[k], nb)
             for (k, w) in enumerate(ws)]
    @sync for (k, w) in enumerate(ws)
        @async Distributed.remotecall_fetch(_ready, w, parts[k])
    end
    pids = union([Distributed.myid()], ws)
    set(nf) = (A = _shared(T, (Px, Py, nf), pids), B = _shared(Complex{T}, (nk, Py, nf), pids),
               C = _shared(T, (Px, Py, nf), pids))
    m = FlowGeometries.Grids.mask(grid)
    mask = all(m) ? nothing : m
    single = set(1)
    return DistributedFFTGridPlan{T, typeof(layout), typeof(mask), typeof(inner), typeof(single),
                                  Union{Nothing, typeof(single)}}(
        parts, ws, rows, layout, mask, inner, single, nb == 0 ? nothing : set(nb), nb)
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

function _filter!(out, field, gp::DistributedFFTGridPlan, sh, transfer, invrenorm, batched::Bool)
    _stage!(sh.A, field, gp)
    _each(_cols_forward!, gp, sh.A, sh.B, batched)
    _each(_rows!, gp, sh.B, transfer, batched, true)
    _each(_cols_inverse!, gp, sh.B, sh.C, batched)
    return _unstage!(out, sh.C, gp, invrenorm)
end

function CGEF.Filtering.distributed_fft_filter_plan(
    gp::DistributedFFTGridPlan{T}, kernel::CGEF.Kernels.AbstractFilterKernel, scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
) where {T}
    (; kx, ky, P) = gp.layout
    scaled = T[CGEF.Kernels.spectral_transfer(kernel, sqrt(kx[i]^2 + ky[j]^2), scale) / (P[1] * P[2])
               for i in eachindex(kx), j in eachindex(ky)]
    transfer = [Distributed.remotecall(_hold, w, gp.backend, scaled[gp.rows[k], :]) for (k, w) in enumerate(gp.workers)]
    invrenorm = if mask_strategy isa CGEF.Filtering.Deformable && (gp.mask !== nothing || gp.layout.padded)
        # `filter(mask)` through the same transforms; the padding beyond a bounded axis is inactive.
        dims = gp.layout.dims
        indicator = gp.mask === nothing ? ones(T, dims) : T.(gp.mask)
        renorm = _filter!(zeros(T, dims), indicator, gp, gp.single, transfer, nothing, false)
        active = gp.mask === nothing ? trues(dims) : gp.mask
        ifelse.(active, CGEF.Filtering._inv_mass.(renorm), zero(T))
    else
        nothing
    end
    return DistributedFFTFilterPlan(gp, transfer, invrenorm, mask_strategy)
end

function _shared_for(gp::DistributedFFTGridPlan, A::AbstractArray)
    ndims(A) == 2 && return (gp.single, false)
    gp.batched === nothing && throw(ArgumentError(
        "this spectral plan was not built for a batch; pass `batch = nb` to `plan_filter`"))
    size(A) == (gp.layout.dims..., gp.nb) || throw(DimensionMismatch(
        "the plan was built for $(gp.layout.dims) × a batch of $(gp.nb); got $(size(A))"))
    return (gp.batched, true)
end

function CGEF.Filtering.filter_apply!(out::AbstractMatrix, field::AbstractMatrix, plan::DistributedFFTFilterPlan)
    gp = plan.grid_plan
    size(out) == size(field) == gp.layout.dims || throw(DimensionMismatch(
        "the plan was built for $(gp.layout.dims); got out $(size(out)) and field $(size(field))"))
    return _filter!(out, field, gp, gp.single, plan.transfer, plan.invrenorm, false)
end

CGEF.Filtering._batched_fields(outs, plan::DistributedFFTFilterPlan) =
    plan.grid_plan.batched !== nothing && ndims(first(outs)) == 3

function CGEF.Filtering.filter_apply_batched!(
    out::AbstractArray{<:Any,3}, field::AbstractArray{<:Any,3}, plan::DistributedFFTFilterPlan,
)
    size(out) == size(field) || throw(DimensionMismatch("got out $(size(out)) and field $(size(field))"))
    sh, batched = _shared_for(plan.grid_plan, field)
    return _filter!(out, field, plan.grid_plan, sh, plan.transfer, plan.invrenorm, batched)
end

# Analysis is scale-independent, so a sweep transforms each field once and then only multiplies by each
# scale's transfer function and inverts.
function CGEF.Filtering.analyze_buffer(plan::DistributedFFTFilterPlan, field::AbstractArray)
    gp = plan.grid_plan
    (ndims(field) == 3 && (gp.batched === nothing || size(field, 3) != gp.nb)) && return nothing
    sh, _ = _shared_for(gp, field)
    return similar(SharedArrays.sdata(sh.B))
end

function CGEF.Filtering.filter_analyze!(F̂::AbstractArray, field::AbstractArray, plan::DistributedFFTFilterPlan)
    gp = plan.grid_plan
    sh, batched = _shared_for(gp, field)
    _stage!(sh.A, field, gp)
    _each(_cols_forward!, gp, sh.A, sh.B, batched)
    _each(_rows!, gp, sh.B, nothing, batched, true)
    return copyto!(F̂, SharedArrays.sdata(sh.B))
end

function CGEF.Filtering.filter_synthesize!(out::AbstractArray, F̂::AbstractArray, plan::DistributedFFTFilterPlan)
    gp = plan.grid_plan
    sh, batched = _shared_for(gp, out)
    copyto!(SharedArrays.sdata(sh.B), F̂)
    _each(_rows!, gp, sh.B, plan.transfer, batched, false)
    _each(_cols_inverse!, gp, sh.B, sh.C, batched)
    return _unstage!(out, sh.C, gp, plan.invrenorm)
end

end # module
