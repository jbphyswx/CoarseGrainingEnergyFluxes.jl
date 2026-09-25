# ---------------------------------------------------------------------------
# Spectral filtering of a Cartesian node set by nonuniform FFT
# ---------------------------------------------------------------------------
#
# A quadrature estimate of the field's Fourier coefficients on a box of periods L = (L₁, …, L_D), a
# multiply by the kernel's transfer function, and the series evaluated back at the points:
#
#   type 1:    F_k = Σⱼ wⱼ cⱼ exp(-i k⋅xⱼ),   wⱼ = Aⱼ / ∏ L_d
#   multiply:  F_k ← Ĝ(|k|, ℓ) F_k
#   type 2:    c̄ⱼ = Σ_k F_k exp(+i k⋅xⱼ)
#
# `Aⱼ` is the grid's measure of point j, so F_k is the quadrature rule for the box's Fourier coefficient
# of c. A direction the grid declares periodic takes the grid's period. Any other is padded as FFTW pads
# a bounded axis: the record, the points' extent plus one spacing, holds N modes, and the box holds
# 2·nextprod((2,3,5), N) at the same spacing, so the wrap-around path between two points is at least
# the record long.
#
# An axis of even count N carries the frequencies |k| ≤ N/2 with weight ½ at ±N/2, so the series is
# real for a real field and equals the FFT's on a lattice of N points. The values are real, and the
# transforms hold the half k₁ ≥ 0.
#
# Masking (Knutsson & Westin 1993): `ZeroFill` filters `mask·field`; `Deformable` also divides by
# `filter(mask)`, run through the same pipeline once per plan. A masked point contributes exactly zero,
# whatever the field holds there.

const _NUFFTRoute = Union{SpectralBackends.AbstractAutoSpectralBackend, SpectralBackends.AbstractNUFFTSpectralBackend}

"""
    NUFFTGridPlan

The half of a nonuniform-FFT filter plan the filter scale does not reach: the transform over the
node set (and one over a trailing batch of `nb` fields when planned for it), the box periods, the
record's mode count per direction, the quadrature weights, the mask, the execution backend whose
memory holds them, and the points this process transforms (`nothing` for all). A transform holds the
working state of its own execution, so a concurrent driver needs its own grid plan per worker.
"""
struct NUFFTGridPlan{
    T<:AbstractFloat, D, P, PB, VT<:AbstractVector{T}, MK, B<:ComputationalBackends.AbstractExecutionBackend, O,
} <: AbstractGridPlan
    plan::P
    batched::PB          # the plan with `ntrans = nb`, or nothing
    nb::Int
    period::NTuple{D,T}
    counts::NTuple{D,Int}
    weights::VT          # Aⱼ / ∏ L_d over this process's points
    mask::MK             # the grid's mask over this process's points, or nothing when fully active
    bounded::Bool
    backend::B
    own::O               # this process's points, or nothing for all of them
    npts::Int            # the grid's point count
end

"""
    NUFFTScratch

The transient half of a nonuniform-FFT filter plan: the values and modes one execution writes, and the
same pair with a trailing batch axis for a batched plan. One per concurrent worker.
"""
struct NUFFTScratch{V<:AbstractVector, F<:AbstractArray, BT} <: AbstractFilterScratch
    values::V
    modes::F
    batched::BT          # (; values, modes) with a trailing batch axis, or nothing
end

"""
    NUFFTFilterPlan

Scattered-Cartesian spectral filter plan: the shared [`NUFFTGridPlan`](@ref), the transfer function on
its modes, the `Deformable` inverse local mass (or `nothing`), and the [`NUFFTScratch`](@ref) the apply
writes through.
"""
struct NUFFTFilterPlan{
    T<:AbstractFloat, GP<:NUFFTGridPlan{T}, A<:AbstractArray{T}, R, MS<:AbstractMaskStrategy, SC<:NUFFTScratch,
} <: AbstractFilterPlan
    grid_plan::GP
    transfer::A
    invrenorm::R
    strategy::MS
    scratch::SC
end

plan_strategy(plan::NUFFTFilterPlan) = plan.strategy

Base.show(io::IO, plan::NUFFTFilterPlan) =
    print(io, "NUFFTFilterPlan(", plan.grid_plan.plan, ")")

# The record's mode count per direction: a lattice's own count per axis, or the count a uniform
# density of the same points gives over the spans.
function _record_counts(x::NTuple{D,AbstractVector}, spans::NTuple{D,Real}) where {D}
    npts = length(first(x))
    distinct = map(v -> length(unique(v)), x)
    prod(distinct) == npts && return distinct
    ρ = (npts / prod(spans))^(1 / D)
    return map(s -> max(2, round(Int, s * ρ)), spans)
end

# An even count N runs over |k| ≤ N/2, one mode more than N.
_mode_count(N::Int) = isodd(N) ? N : N + 1
_edge_weight(::Type{T}, f::Int, N::Int) where {T} = (iseven(N) && abs(f) == N ÷ 2) ? T(1 // 2) : one(T)

# `points` is the share of the points this plan transforms, `_owned(backend, npts)` unless given.
function _nufft_grid_plan(
    nufft, grid::FlowGeometries.Grids.UnstructuredGrid{T,G,D};
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    batch::Union{Nothing,Integer} = nothing,
    points::Union{Nothing,AbstractVector{Int}} = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractCartesianGeometry{T}, D}
    1 <= D <= 3 || throw(ArgumentError(
        "nonuniform-FFT spectral filtering takes 1, 2 or 3 coordinates; this grid has $D"))
    x = ntuple(d -> FlowGeometries.Grids.coordinates(grid, d), Val(D))
    npts = length(first(x))
    npts > 0 || throw(ArgumentError("nonuniform-FFT spectral filtering needs at least one point."))
    periodic = ntuple(d -> FlowGeometries.Grids.isperiodic(grid, d), Val(D))
    lo = map(minimum, x)
    extent = map((v, l) -> maximum(v) - l, x, lo)
    spans = ntuple(d -> periodic[d] ? T(FlowGeometries.Grids.period(grid, d)) : extent[d], Val(D))
    all(>(0), spans) || throw(ArgumentError(
        "nonuniform-FFT spectral filtering needs a box of positive size; the points span $spans. A " *
        "point set with no extent in a direction needs `periodic` and `period` declared for it."))
    record = _record_counts(x, spans)
    # A bounded direction's record spacing is its extent over the count less one, so a lattice of N
    # points spans exactly N spacings.
    counts = ntuple(d -> periodic[d] ? record[d] : 2 * nextprod((2, 3, 5), record[d]), Val(D))
    L = ntuple(d -> periodic[d] ? spans[d] : counts[d] * extent[d] / (record[d] - 1), Val(D))
    nmodes = map(_mode_count, counts)
    # This process's points, their weights and mask go where `backend` runs, and the library plans in
    # that memory. Type 1 is a sum over points, so the processes' partial spectra sum to the whole one.
    own = points === nothing ? _owned(backend, npts) : points
    part(v) = own === nothing ? v : v[own]
    xb = map(v -> _on_backend(backend, T.(part(v))), x)
    kw = (; period = L, origin = lo, nthreads = _library_threads(backend))
    plan = FlowTransformBindings.plan_nufft(nufft, T, xb, nmodes; kw...)
    nb = batch === nothing ? 0 : Int(batch)
    batched = nb == 0 ? nothing : FlowTransformBindings.plan_nufft(nufft, T, xb, nmodes; ntrans = nb, kw...)
    weights = _on_backend(backend, T.(part(FlowGeometries.Grids.measure(grid)) ./ prod(L)))
    m = FlowGeometries.Grids.mask(grid)
    return NUFFTGridPlan(plan, batched, nb, L, counts, weights, all(m) ? nothing : _on_backend(backend, part(m)),
                         !all(periodic), backend, own === nothing ? nothing : _on_backend(backend, own), npts)
end

# This process's points of a field every process holds.
@inline _local(field::AbstractVector, ::Nothing) = field
@inline _local(field::AbstractVector, own) = view(field, own)
@inline _local(field::AbstractMatrix, ::Nothing) = field
@inline _local(field::AbstractMatrix, own) = view(field, own, :)

# Type 1 of this process's points, summed over the processes into the whole spectrum.
function _analysis!(modes, p, values, gp::NUFFTGridPlan)
    FlowTransformBindings.nufft_type1!(modes, p, values)
    return _sum_across!(gp.backend, modes)
end

function _nufft_scratch(gp::NUFFTGridPlan)
    b = gp.batched
    return NUFFTScratch(
        FlowTransformBindings.allocate_values(gp.plan), FlowTransformBindings.allocate_modes(gp.plan),
        b === nothing ? nothing :
            (values = FlowTransformBindings.allocate_values(b), modes = FlowTransformBindings.allocate_modes(b)),
    )
end

# `Ĝ(|k|, ℓ)` on the plan's modes, with the split-Nyquist weights.
function _nufft_transfer(gp::NUFFTGridPlan{T,D}, kernel::Kernels.AbstractFilterKernel, scale::T) where {T,D}
    p = gp.plan
    freqs = ntuple(d -> FlowTransformBindings.mode_frequencies(p, d), Val(D))
    transfer = Array{T}(undef, FlowTransformBindings.mode_size(p))
    for I in CartesianIndices(transfer)
        k2 = zero(T)
        w = one(T)
        for d in 1:D
            f = freqs[d][I[d]]
            k2 += (T(2π) * f / gp.period[d])^2
            w *= _edge_weight(T, f, gp.counts[d])
        end
        transfer[I] = w * Kernels.spectral_transfer(kernel, sqrt(k2), scale, Val(D))
    end
    return _on_backend(gp.backend, transfer)
end

# `w .* field`, or `mask .* (w .* field)`: a `Bool` is a strong zero, so a masked point adds nothing
# even where the field is NaN.
function _load_weighted!(c::AbstractArray, field::AbstractArray, gp::NUFFTGridPlan{T}) where {T}
    if gp.mask === nothing
        @. c = T(gp.weights * field)
    else
        @. c = T(gp.mask * (gp.weights * field))
    end
    return c
end

# This process's filtered values into `out`, which each process then holds whole.
_store_filtered!(out, c, invrenorm, gp::NUFFTGridPlan) = _store_filtered!(out, c, invrenorm, gp.own, gp.backend)
_store_filtered!(out, c, ::Nothing, ::Nothing, _) = (out .= c; out)
_store_filtered!(out, c, invrenorm::AbstractVector, ::Nothing, _) = (out .= c .* invrenorm; out)
function _store_filtered!(out, c, invrenorm, own, backend)
    fill!(out, zero(eltype(out)))
    _store_filtered!(_local(out, own), c, invrenorm, nothing, backend)
    return _sum_across!(backend, out)
end

# The order `AutoSpectralBackend` tries the libraries in.
const _NUFFT_LIBRARY_ORDER = (FlowTransformBindings.NonuniformFFTsBackend(), FlowTransformBindings.FINUFFTBackend())

"""
    _nufft_library(spectral_backend) -> tag or nothing

The FlowTransformBindings tag that runs a nonuniform-FFT filter: the one named, or for
`AutoSpectralBackend` NonuniformFFTs when it is loaded and FINUFFT when only it is; `nothing` when
neither is loaded.
"""
_nufft_library(b::Union{FlowTransformBindings.FINUFFTBackend, FlowTransformBindings.NonuniformFFTsBackend}) = b
function _nufft_library(::SpectralBackends.AbstractAutoSpectralBackend)
    for b in _NUFFT_LIBRARY_ORDER
        FlowTransformBindings.is_available(b) && return b
    end
    return nothing
end
_nufft_library(t::SpectralBackends.AbstractNUFFTSpectralBackend) = throw(ArgumentError(
    "$(nameof(typeof(t))) names no NUFFT library; pass FlowTransformBindings.NonuniformFFTsBackend() " *
    "(`using NonuniformFFTs`) or FlowTransformBindings.FINUFFTBackend() (`using FINUFFT`)."))

spectral_scratch(gp::NUFFTGridPlan) = _nufft_scratch(gp)

"""
    distributed_nufft_grid_plan(nufft, grid; backend::DistributedBackend, batch) -> AbstractGridPlan
    distributed_nufft_filter_plan(grid_plan, kernel, scale, mask_strategy) -> AbstractFilterPlan

A nonuniform-FFT spectral plan whose points are divided among the worker processes, each holding the
library plan over its block. Methods in the Distributed extension.
"""
function distributed_nufft_grid_plan end
function distributed_nufft_filter_plan end

distributed_nufft_grid_plan(args...; kwargs...) = throw(ArgumentError(
    "DistributedBackend is unavailable — run `using Distributed, SharedArrays` (or use SerialBackend())."))

# The grid plan for `backend`: over this process's points, or divided among the worker processes.
_nufft_grid_plan_for(nufft, grid, backend::ComputationalBackends.AbstractDistributedBackend, batch) =
    distributed_nufft_grid_plan(nufft, grid; backend = backend, batch = batch)
_nufft_grid_plan_for(nufft, grid, backend::ComputationalBackends.AbstractExecutionBackend, batch) =
    _nufft_grid_plan(nufft, grid; backend = backend, batch = batch)

function spectral_grid_plan(
    spectral_backend::_NUFFTRoute, grid::FlowGeometries.Grids.UnstructuredGrid{T,G},
    ::Kernels.AbstractFilterKernel;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    batch::Union{Nothing,Integer} = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractCartesianGeometry{T}}
    nufft = _nufft_library(spectral_backend)
    nufft === nothing && return nothing
    return _nufft_grid_plan_for(nufft, grid, backend, batch)
end

"""
    spectral_filter_plan(spectral_backend, grid::UnstructuredGrid{Cartesian}, kernel, scale;
                         mask_strategy = ZeroFill(), backend = AutoBackend(), batch = nothing)

`batch = nb` adds a transform with `ntrans = nb`, which `filter_apply_batched!` runs over an
`(npts, nb)` array in one execution per direction. The library runs on `_library_threads(backend)`
threads.
"""
function spectral_filter_plan(
    spectral_backend::_NUFFTRoute,
    grid::FlowGeometries.Grids.UnstructuredGrid{T,G},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    batch::Union{Nothing,Integer} = nothing,
    grid_plan::Union{Nothing,AbstractGridPlan} = nothing,
    scratch::Union{Nothing,NUFFTScratch} = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractCartesianGeometry{T}}
    gp = if grid_plan === nothing
        nufft = _nufft_library(spectral_backend)
        nufft === nothing && _no_spectral_backend(spectral_backend, grid)
        _nufft_grid_plan_for(nufft, grid, backend, batch)
    else
        grid_plan
    end
    return _nufft_filter_plan(gp, scratch, kernel, scale, mask_strategy)
end

_nufft_filter_plan(gp::AbstractGridPlan, _, kernel, scale, mask_strategy) =
    distributed_nufft_filter_plan(gp, kernel, scale, mask_strategy)

function _nufft_filter_plan(
    gp::NUFFTGridPlan{T}, scratch, kernel::Kernels.AbstractFilterKernel, scale::T, mask_strategy::AbstractMaskStrategy,
) where {T}
    sc = scratch === nothing ? _nufft_scratch(gp) : scratch
    transfer = _nufft_transfer(gp, kernel, scale)
    invrenorm = if mask_strategy isa Deformable && (gp.mask !== nothing || gp.bounded)
        # `filter(mask)`, through the plan's own pipeline; the box beyond a bounded record is inactive.
        if gp.mask === nothing
            sc.values .= gp.weights
        else
            @. sc.values = gp.mask * gp.weights
        end
        _analysis!(sc.modes, gp.plan, sc.values, gp)
        sc.modes .*= transfer
        FlowTransformBindings.nufft_type2!(sc.values, gp.plan, sc.modes)
        gp.mask === nothing ? _inv_mass.(sc.values) : ifelse.(gp.mask, _inv_mass.(sc.values), zero(T))
    else
        nothing
    end
    return NUFFTFilterPlan(gp, transfer, invrenorm, mask_strategy, sc)
end

function filter_apply!(out::AbstractVector{T}, field::AbstractVector, plan::NUFFTFilterPlan{T}) where {T}
    gp, sc = plan.grid_plan, plan.scratch
    _load_weighted!(sc.values, _local(field, gp.own), gp)
    _analysis!(sc.modes, gp.plan, sc.values, gp)
    sc.modes .*= plan.transfer
    FlowTransformBindings.nufft_type2!(sc.values, gp.plan, sc.modes)
    return _store_filtered!(out, sc.values, plan.invrenorm, gp)
end

# Analysis depends on the field alone, so a sweep runs it once and each scale only multiplies by its own
# transfer function and evaluates back to the points.
analyze_buffer(plan::NUFFTFilterPlan, ::AbstractVector) = similar(plan.scratch.modes)

function filter_analyze!(F̂::AbstractArray, field::AbstractVector, plan::NUFFTFilterPlan)
    gp, sc = plan.grid_plan, plan.scratch
    _load_weighted!(sc.values, _local(field, gp.own), gp)
    return _analysis!(F̂, gp.plan, sc.values, gp)
end

function filter_synthesize!(out::AbstractVector{T}, F̂::AbstractArray, plan::NUFFTFilterPlan{T}) where {T}
    gp, sc = plan.grid_plan, plan.scratch
    sc.modes .= F̂ .* plan.transfer
    FlowTransformBindings.nufft_type2!(sc.values, gp.plan, sc.modes)
    return _store_filtered!(out, sc.values, plan.invrenorm, gp)
end

# A trailing batch axis: `nb` fields on the same points, one `ntrans = nb` execution per direction.

_batched_fields(outs, plan::NUFFTFilterPlan) = plan.grid_plan.batched !== nothing && ndims(first(outs)) == 2

function _batch_buffers(plan::NUFFTFilterPlan, field::AbstractMatrix)
    gp, p = plan.grid_plan, plan.scratch.batched
    (gp.batched === nothing || p === nothing) && throw(ArgumentError(
        "this spectral plan was not built for a batch; pass `batch = nb` to `plan_filter`"))
    size(field) == (gp.npts, gp.nb) || throw(DimensionMismatch(
        "the plan was built for $(gp.npts) points × a batch of $(gp.nb); got $(size(field))"))
    return gp.batched, p
end

function filter_apply_batched!(out::AbstractMatrix{T}, field::AbstractMatrix, plan::NUFFTFilterPlan{T}) where {T}
    size(out) == size(field) || throw(DimensionMismatch(
        "filter_apply_batched! got out $(size(out)) and field $(size(field))"))
    gp = plan.grid_plan
    b, p = _batch_buffers(plan, field)
    _load_weighted!(p.values, _local(field, gp.own), gp)
    _analysis!(p.modes, b, p.values, gp)
    p.modes .*= plan.transfer
    FlowTransformBindings.nufft_type2!(p.values, b, p.modes)
    return _store_filtered!(out, p.values, plan.invrenorm, gp)
end

analyze_buffer(plan::NUFFTFilterPlan, field::AbstractMatrix) =
    (plan.scratch.batched === nothing || size(field) != (plan.grid_plan.npts, plan.grid_plan.nb)) ? nothing :
        similar(plan.scratch.batched.modes)

function filter_analyze!(F̂::AbstractArray, field::AbstractMatrix, plan::NUFFTFilterPlan)
    gp = plan.grid_plan
    b, p = _batch_buffers(plan, field)
    _load_weighted!(p.values, _local(field, gp.own), gp)
    return _analysis!(F̂, b, p.values, gp)
end

function filter_synthesize!(out::AbstractMatrix{T}, F̂::AbstractArray, plan::NUFFTFilterPlan{T}) where {T}
    b, p = _batch_buffers(plan, out)
    p.modes .= F̂ .* plan.transfer
    FlowTransformBindings.nufft_type2!(p.values, b, p.modes)
    return _store_filtered!(out, p.values, plan.invrenorm, plan.grid_plan)
end
