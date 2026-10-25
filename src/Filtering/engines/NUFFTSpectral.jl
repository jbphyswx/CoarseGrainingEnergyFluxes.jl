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
# of c. A direction the grid declares periodic takes the grid's period. Any other is padded as
# `_fft_layout` pads a bounded axis: the length the nodes cover (`FlowGeometries.Grids.domain_length`)
# holds the N modes of its N nodes (`FlowGeometries.Grids.node_counts`), and the box holds
# 2·nextprod((2,3,5), N) at the same spacing, so the wrap-around path between two points is at least
# that length.
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
mode count per direction, the quadrature weights, the mask, the execution backend whose
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
    execution::B
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
plan_execution(plan::NUFFTFilterPlan) = plan.grid_plan.execution

Base.show(io::IO, plan::NUFFTFilterPlan) =
    print(io, "NUFFTFilterPlan(", plan.grid_plan.plan, ")")

# An even count N runs over |k| ≤ N/2, one mode more than N.
_mode_count(N::Int) = isodd(N) ? N : N + 1
_edge_weight(::Type{T}, f::Int, N::Int) where {T} = (iseven(N) && abs(f) == N ÷ 2) ? T(1 // 2) : one(T)

"""
    _KernelBox(radius, cutoff)

The box of a kernel sum ([`NUFFTSumFootprint`](@ref)): a bounded direction padded by the kernel's radius,
so every image of a point lies at least `radius` from the others, and every mode with `|k| ≤ cutoff`.
"""
struct _KernelBox{T<:AbstractFloat}
    radius::T
    cutoff::T
end

# The box periods and mode counts. A spectral filter's box holds a mode per node along a periodic direction
# and a bounded one padded as `_fft_layout` pads an axis; a kernel sum's holds the modes up to the cutoff,
# an odd count per direction.
function _nufft_box(::Nothing, grid, periodic::NTuple{D,Bool}, spans::Tuple{T,Vararg{T}}) where {D,T}
    nodes = FlowGeometries.Grids.node_counts(grid)
    counts = ntuple(d -> periodic[d] ? nodes[d] : 2 * nextprod((2, 3, 5), nodes[d]), Val(D))
    L = ntuple(d -> periodic[d] ? spans[d] : counts[d] * T(FlowGeometries.Grids.domain_length(grid, d)) / nodes[d],
               Val(D))
    return L, counts
end

function _nufft_box(b::_KernelBox, _, periodic::NTuple{D,Bool}, spans::Tuple{T,Vararg{T}}) where {D,T}
    L = ntuple(d -> periodic[d] ? spans[d] : spans[d] + T(b.radius), Val(D))
    return L, map(Ld -> 2 * floor(Int, T(b.cutoff) * Ld / T(2π)) + 1, L)
end

# `points` is the share of the points this plan transforms, `_owned(execution, npts)` unless given. `box`
# is `nothing` for a spectral filter or a `_KernelBox`; `tol` is the transforms' tolerance.
function _nufft_grid_plan(
    nufft, grid::FlowGeometries.Grids.UnstructuredGrid{T,G,D};
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    batch::Union{Nothing,Integer} = nothing,
    points::Union{Nothing,AbstractVector{Int}} = nothing,
    box::Union{Nothing,_KernelBox} = nothing,
    tol::Real = FlowTransformBindings.default_tolerance(T),
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
    L, counts = _nufft_box(box, grid, periodic, spans)
    nmodes = map(_mode_count, counts)
    # This process's points, their weights and mask go where `execution` runs, and the library plans in
    # that memory. Type 1 is a sum over points, so the processes' partial spectra sum to the whole one.
    own = points === nothing ? _owned(execution, npts) : points
    part(v) = own === nothing ? v : v[own]
    xb = map(v -> _on_backend(execution, T.(part(v))), x)
    kw = (; period = L, origin = lo, nthreads = _library_threads(execution), tol = tol)
    plan = FlowTransformBindings.plan_nufft(nufft, T, xb, nmodes; kw...)
    nb = batch === nothing ? 0 : Int(batch)
    batched = nb == 0 ? nothing : FlowTransformBindings.plan_nufft(nufft, T, xb, nmodes; ntrans = nb, kw...)
    weights = _on_backend(execution, T.(part(FlowGeometries.Grids.measure(grid)) ./ prod(L)))
    m = FlowGeometries.Grids.mask(grid)
    return NUFFTGridPlan(plan, batched, nb, L, counts, weights, all(m) ? nothing : _on_backend(execution, part(m)),
                         !all(periodic), execution, own === nothing ? nothing : _on_backend(execution, own), npts)
end

# This process's points of a field every process holds.
@inline _local(field::AbstractVector, ::Nothing) = field
@inline _local(field::AbstractVector, own) = view(field, own)
@inline _local(field::AbstractMatrix, ::Nothing) = field
@inline _local(field::AbstractMatrix, own) = view(field, own, :)

# Type 1 of this process's points, summed over the processes into the whole spectrum.
function _analysis!(modes, p, values, gp::NUFFTGridPlan)
    FlowTransformBindings.nufft_type1!(modes, p, values)
    return _sum_across!(gp.execution, modes)
end

function _nufft_scratch(gp::NUFFTGridPlan)
    b = gp.batched
    return NUFFTScratch(
        FlowTransformBindings.allocate_values(gp.plan), FlowTransformBindings.allocate_modes(gp.plan),
        b === nothing ? nothing :
            (values = FlowTransformBindings.allocate_values(b), modes = FlowTransformBindings.allocate_modes(b)),
    )
end

# `Ĝ(k, ℓ)` on the plan's modes, with the split-Nyquist weights.
function _nufft_transfer(gp::NUFFTGridPlan{T,D}, kernel::Kernels.AbstractFilterKernel, scale::T) where {T,D}
    p = gp.plan
    freqs = ntuple(d -> FlowTransformBindings.mode_frequencies(p, d), Val(D))
    transfer = Array{T}(undef, FlowTransformBindings.mode_size(p))
    for I in CartesianIndices(transfer)
        k = ntuple(d -> T(2π) * freqs[d][I[d]] / gp.period[d], Val(D))
        w = prod(ntuple(d -> _edge_weight(T, freqs[d][I[d]], gp.counts[d]), Val(D)))
        transfer[I] = w * Kernels.spectral_transfer(kernel, k, scale)
    end
    return _on_backend(gp.execution, transfer)
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
_store_filtered!(out, c, invrenorm, gp::NUFFTGridPlan) = _store_filtered!(out, c, invrenorm, gp.own, gp.execution)
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
    distributed_nufft_grid_plan(nufft, grid; execution::DistributedBackend, batch) -> AbstractGridPlan
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
    distributed_nufft_grid_plan(nufft, grid; execution = backend, batch = batch)
_nufft_grid_plan_for(nufft, grid, backend::ComputationalBackends.AbstractExecutionBackend, batch) =
    _nufft_grid_plan(nufft, grid; execution = backend, batch = batch)

function spectral_grid_plan(
    spectral_backend::_NUFFTRoute, grid::FlowGeometries.Grids.UnstructuredGrid{T,G},
    ::Kernels.AbstractFilterKernel;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    batch::Union{Nothing,Integer} = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractCartesianGeometry{T}}
    nufft = _nufft_library(spectral_backend)
    nufft === nothing && return nothing
    return _nufft_grid_plan_for(nufft, grid, execution, batch)
end

"""
    spectral_filter_plan(spectral_backend, grid::UnstructuredGrid{Cartesian}, kernel, scale;
                         mask_strategy = ZeroFill(), execution = AutoBackend(), batch = nothing,
                         grid_plan = nothing, scratch = nothing) -> AbstractFilterPlan

The nonuniform-FFT filter plan over a Cartesian node set, built on `grid_plan` and `scratch` when
given. `batch = nb` adds a transform with `ntrans = nb`, which `filter_apply_batched!` runs over an
`(npts, nb)` array in one execution per direction. The library runs on `_library_threads(execution)`
threads.
"""
function spectral_filter_plan(
    spectral_backend::_NUFFTRoute,
    grid::FlowGeometries.Grids.UnstructuredGrid{T,G},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    batch::Union{Nothing,Integer} = nothing,
    grid_plan::Union{Nothing,AbstractGridPlan} = nothing,
    scratch::Union{Nothing,NUFFTScratch} = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractCartesianGeometry{T}}
    gp = if grid_plan === nothing
        nufft = _nufft_library(spectral_backend)
        nufft === nothing && _no_spectral_backend(spectral_backend, grid)
        _nufft_grid_plan_for(nufft, grid, execution, batch)
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
        # `filter(mask)`, through the plan's own pipeline; the padding beyond a bounded domain is inactive.
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

# ---------------------------------------------------------------------------
# A radial kernel's sum over a Cartesian node set, by nonuniform FFT
# ---------------------------------------------------------------------------

"""
    NUFFTSumFootprint

The real-space filter over a Cartesian grid's cells as points, `Σⱼ G(|x − xⱼ|) Aⱼ fⱼ / Σⱼ G(|x − xⱼ|) Aⱼ`
over the cells the strategy counts, its sums taken by nonuniform FFT. On a box whose bounded directions
are padded by the kernel's radius, so that every image of a point lies past the radius from every other,

    Σⱼ G(x − xⱼ) Aⱼ cⱼ = Σ_k Ĝ(k) e^{ik⋅x} Σⱼ (Aⱼ / ∏L) cⱼ e^{−ik⋅xⱼ}

(Potts & Steidl 2003), with `|k|` up to [`Kernels.transfer_cutoff`](@ref). It is the node CSR engine's
sum with the kernel continued past its truncation radius, where it is below
`Kernels.GAUSSIAN_TRUNCATION_TOL` of its peak, and the transforms run at that tolerance.

`invden` is the reciprocal of the denominator, zero where it vanishes or, under `Deformable`, at an
inactive node. `bound` is whether the two strategies' denominators differ, which they do on a masked grid.

# References
- Potts & Steidl (2003), *SIAM J. Sci. Comput.* 24(6), "Fast summation at nonequispaced knots by NFFTs".
"""
struct NUFFTSumFootprint{
    T<:AbstractFloat, GP<:NUFFTGridPlan{T}, A<:AbstractArray{T}, V<:AbstractVector{T},
    MS<:AbstractMaskStrategy, SC<:NUFFTScratch,
}
    grid_plan::GP
    transfer::A
    invden::V
    strategy::MS
    bound::Bool
    scratch::SC
end

Base.show(io::IO, fp::NUFFTSumFootprint) = print(io, "NUFFTSumFootprint(", fp.grid_plan.plan, ")")

_driven_footprint(::NUFFTSumFootprint) = true

"""
    nufft_sum_footprint(grid, kernel, scale; mask_strategy = ZeroFill(), execution = SerialBackend())
        -> NUFFTSumFootprint

The [`NUFFTSumFootprint`](@ref) of a kernel with a [`Kernels.transfer_cutoff`](@ref) over the cells of a
Cartesian grid with no lattice of its own, a node set or a curvilinear mesh, read and written in the
grid's linear cell order. The transforms run on the NUFFT library `AutoSpectralBackend` takes, on
`_library_threads(execution)` threads.
"""
nufft_sum_footprint(
    grid::FlowGeometries.Grids.AbstractGrid{G,T}, kernel::Kernels.AbstractFilterKernel, scale::T; kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractCartesianGeometry{T}} =
    _nufft_sum_footprint(_node_set(grid), kernel, scale; kwargs...)

function _nufft_sum_footprint(
    grid::FlowGeometries.Grids.UnstructuredGrid{T,G,D}, kernel::Kernels.AbstractFilterKernel, scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractCartesianGeometry{T}, D}
    nufft = _nufft_library(SpectralBackends.AutoSpectralBackend())
    box = _KernelBox(T(Kernels.kernel_radius(kernel, scale, Val(D))), T(Kernels.transfer_cutoff(kernel, scale)))
    gp = _nufft_grid_plan(nufft, grid; execution, box, tol = T(Kernels.GAUSSIAN_TRUNCATION_TOL))
    sc = _nufft_scratch(gp)
    transfer = _nufft_transfer(gp, kernel, scale)
    # The denominator is the same sum over the weights of the nodes the strategy counts.
    deformable = mask_strategy isa Deformable && gp.mask !== nothing
    deformable ? (@. sc.values = gp.mask * gp.weights) : (sc.values .= gp.weights)
    _analysis!(sc.modes, gp.plan, sc.values, gp)
    sc.modes .*= transfer
    FlowTransformBindings.nufft_type2!(sc.values, gp.plan, sc.modes)
    invden = deformable ? ifelse.(gp.mask, _inv_mass.(sc.values), zero(T)) : _inv_mass.(sc.values)
    return NUFFTSumFootprint(gp, transfer, invden, mask_strategy, gp.mask !== nothing, sc)
end

@noinline _nufft_sum_strategy_mismatch() = throw(ArgumentError(
    "NUFFTSumFootprint is being applied with a different mask strategy than it was built for, on a masked " *
    "grid, where the two normalize differently. Rebuild the plan with the `mask_strategy` you intend to " *
    "apply with.",
))

function _check_strategy(fp::NUFFTSumFootprint, strategy::AbstractMaskStrategy)
    (!fp.bound || typeof(strategy) === typeof(fp.strategy)) || _nufft_sum_strategy_mismatch()
    return nothing
end

# The library runs the transforms on the plan's threads, so a driver has nothing to divide. `field` and
# `out` are read and written in linear cell order.
function apply_footprint!(
    out::AbstractArray, field::AbstractArray, grid, fp::NUFFTSumFootprint{T}, strategy::AbstractMaskStrategy,
    driver::D = _sep_serial,
) where {T, D}
    _check_strategy(fp, strategy)
    gp, sc = fp.grid_plan, fp.scratch
    c, w, m = sc.values, gp.weights, gp.mask
    # A `Bool` is a strong zero: an inactive cell adds nothing even where the field is NaN.
    @inbounds for s in eachindex(c)
        c[s] = m === nothing ? T(w[s] * field[s]) : T(m[s] * (w[s] * field[s]))
    end
    _analysis!(sc.modes, gp.plan, c, gp)
    sc.modes .*= fp.transfer
    FlowTransformBindings.nufft_type2!(c, gp.plan, sc.modes)
    invden = fp.invden
    @inbounds for s in eachindex(c)
        out[s] = c[s] * invden[s]
    end
    return out
end

_apply_serial!(out, field, grid, fp::NUFFTSumFootprint, strategy) = apply_footprint!(out, field, grid, fp, strategy)
