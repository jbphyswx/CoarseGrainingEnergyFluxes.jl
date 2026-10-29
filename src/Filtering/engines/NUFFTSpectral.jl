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
# `_bounded_length` modes at the same spacing, so every image of a point lies at least the grid plan's
# reach beyond the other points.
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
mode count per direction, the quadrature weights, the mask, the grid's topology, the distance every image
of a point lies beyond the other points (`Inf` with no bounded direction), the execution backend whose
memory holds them, the points this process transforms (`nothing` for all), and `i k_d` along each
direction of the mode array. A transform holds the working state of its own execution, so a concurrent
driver needs its own grid plan per worker.
"""
struct NUFFTGridPlan{
    T<:AbstractFloat, D, P, PB, VT<:AbstractVector{T}, MK, TP<:NTuple{D,FlowGeometries.Grids.AbstractTopology},
    B<:ComputationalBackends.AbstractExecutionBackend, O, IK<:NTuple{D,AbstractArray{Complex{T}}},
} <: AbstractGridPlan
    plan::P
    batched::PB          # the plan with `ntrans = nb`, or nothing
    nb::Int
    period::NTuple{D,T}
    counts::NTuple{D,Int}
    weights::VT          # Aⱼ / ∏ L_d over this process's points
    mask::MK             # the grid's mask over this process's points, or nothing when fully active
    topology::TP         # a `Bounded` direction is padded past the domain
    reach::T
    execution::B
    own::O               # this process's points, or nothing for all of them
    npts::Int            # the grid's point count
    ik::IK               # `i k_d` laid along direction `d` of the mode array, in the plan's memory
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
its modes, the `Deformable` inverse local mass and its derivative along each direction (or `nothing`
for both), and the [`NUFFTScratch`](@ref) the apply writes through.
"""
struct NUFFTFilterPlan{
    T<:AbstractFloat, GP<:NUFFTGridPlan{T}, A<:AbstractArray{T}, R, DM, MS<:AbstractMaskStrategy, SC<:NUFFTScratch,
} <: AbstractFilterPlan
    grid_plan::GP
    transfer::A
    invrenorm::R
    dmass::DM      # `∂_d M` over this process's points per direction, with `invrenorm`; or nothing
    strategy::MS
    scratch::SC
end

series_form(plan::NUFFTFilterPlan) = _ratio_form(plan.invrenorm)

rate_available(::NUFFTFilterPlan, kernel::Kernels.AbstractFilterKernel) = Kernels.has_transfer_rate(kernel)
rate_plan(plan::NUFFTFilterPlan{T}, kernel::Kernels.AbstractFilterKernel, scale::T) where {T} = NUFFTFilterPlan(
    plan.grid_plan, _nufft_transfer(plan.grid_plan, Kernels.TransferRate(kernel), scale), plan.invrenorm, plan.dmass,
    plan.strategy, plan.scratch,
)

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

# A spectral filter's box at the mode counts the caller gives, over the grid's periods.
struct _ModeCounts{D}
    counts::NTuple{D,Int}
end

# A spectral filter's box with a mode per node, its bounded directions padded for a kernel reaching `reach`.
struct _NodeBox{T<:AbstractFloat}
    reach::T
end

_mode_box(::Nothing, reach::AbstractFloat) = _NodeBox(reach)
_mode_box(counts::NTuple{D,Integer}, ::AbstractFloat) where {D} = _ModeCounts{D}(map(Int, counts))

function _nufft_box(b::_ModeCounts{D}, _, periodic::NTuple{D,Bool}, spans::Tuple{T,Vararg{T}}) where {D,T}
    all(periodic) || throw(ArgumentError(
        "mode counts span a periodic box; this node set is bounded in direction $(findfirst(!, periodic))"))
    return spans, b.counts
end

# The box periods and mode counts. A spectral filter's box holds a mode per node along a periodic direction
# and a bounded one padded as `_fft_layout` pads an axis; a kernel sum's holds the modes up to the cutoff,
# an odd count per direction.
function _nufft_box(b::_NodeBox, grid, periodic::NTuple{D,Bool}, spans::Tuple{T,Vararg{T}}) where {D,T}
    nodes = FlowGeometries.Grids.node_counts(grid)
    Δ = ntuple(d -> T(FlowGeometries.Grids.domain_length(grid, d)) / nodes[d], Val(D))
    counts = ntuple(d -> periodic[d] ? nodes[d] : _bounded_length(nodes[d], spans[d], Δ[d], T(b.reach)), Val(D))
    L = ntuple(d -> periodic[d] ? spans[d] : counts[d] * Δ[d], Val(D))
    return L, counts
end

function _nufft_box(b::_KernelBox, _, periodic::NTuple{D,Bool}, spans::Tuple{T,Vararg{T}}) where {D,T}
    L = ntuple(d -> periodic[d] ? spans[d] : spans[d] + T(b.radius), Val(D))
    return L, map(Ld -> 2 * floor(Int, T(b.cutoff) * Ld / T(2π)) + 1, L)
end

# `points` is the share of the points this plan transforms, `_owned(execution, npts)` unless given. `box`
# is a spectral filter's `_NodeBox` or `_ModeCounts`, or a kernel sum's `_KernelBox`; `tol` is the
# transforms' tolerance. The form taking `k` returns `k(plan)`, called where the library plans' types are
# concrete.
_nufft_grid_plan(nufft, grid::FlowGeometries.Grids.UnstructuredGrid; kwargs...) =
    _nufft_grid_plan(identity, nufft, grid; kwargs...)

function _nufft_grid_plan(
    k::K, nufft, grid::FlowGeometries.Grids.UnstructuredGrid{T,G,D};
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    batch::Union{Nothing,Integer} = nothing,
    points::Union{Nothing,AbstractVector{Int}} = nothing,
    box::Union{_NodeBox,_KernelBox,_ModeCounts},
    tol::Union{Nothing,Real} = nothing,
) where {K, T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractCartesianGeometry{T}, D}
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
    held = minimum(ntuple(d -> periodic[d] ? T(Inf) : L[d] - spans[d], Val(D)))
    nmodes = map(_mode_count, counts)
    # This process's points, their weights and mask go where `execution` runs, and the library plans in
    # that memory. Type 1 is a sum over points, so the processes' partial spectra sum to the whole one.
    own = points === nothing ? _owned(execution, npts) : points
    part(v) = own === nothing ? v : v[own]
    xb = map(v -> _on_backend(execution, T.(part(v))), x)
    kw = (; period = L, origin = lo, nthreads = _library_threads(execution),
            tol = tol === nothing ? FlowTransformBindings.default_tolerance(T) : T(tol))
    nb = batch === nothing ? 0 : Int(batch)
    weights = _on_backend(execution, T.(part(FlowGeometries.Grids.measure(grid)) ./ prod(L)))
    mask = _plan_mask(FlowGeometries.Grids.mask(grid), part, execution)
    tp = FlowGeometries.Grids.topology(grid)
    owned = own === nothing ? nothing : _on_backend(execution, own)
    return FlowTransformBindings.plan_nufft(nufft, T, xb, nmodes; kw...) do plan
        ik = ntuple(d -> _ik(plan, d, L[d], Val(D), execution), Val(D))
        _batched_nufft(nufft, T, xb, nmodes, kw, batch) do batched
            k(NUFFTGridPlan(plan, batched, nb, L, counts, weights, mask, tp, held, execution, owned, npts, ik))
        end
    end
end

# `i k_d = 2πi f / L_d` on the plan's modes along direction `d`, laid along that axis of the mode array, in
# `execution`'s memory.
function _ik(plan, d::Int, L::T, ::Val{D}, execution) where {T,D}
    f = FlowTransformBindings.mode_frequencies(plan, d)
    ik = [complex(zero(T), T(2π) * T(fr) / L) for fr in f]
    return _on_backend(execution, reshape(ik, ntuple(e -> e == d ? length(f) : 1, Val(D))))
end

# `f` of the plan over a batch of `batch` fields, `f(nothing)` for none.
_batched_nufft(f::F, _, ::Type, _, _, _, ::Nothing) where {F} = f(nothing)
_batched_nufft(f::F, nufft, ::Type{T}, xb, nmodes, kw::NamedTuple, batch::Integer) where {F,T} =
    FlowTransformBindings.plan_nufft(f, nufft, T, xb, nmodes; ntrans = Int(batch), kw...)

# The grid's mask over this process's points in `execution`'s memory, `nothing` on a grid with no
# inactive cell.
_plan_mask(::FlowGeometries.Grids.AllActive, part, execution) = nothing
_plan_mask(m::AbstractArray{Bool}, part, execution) = _on_backend(execution, part(m))

# Whether a box with directions of topologies `tp` pads past a bounded domain, from the tags' types.
@inline _padded(tp::Tuple) = any(t -> t isa FlowGeometries.Grids.Bounded, tp)

# Whether a mask, or a plan's `nothing` for none, can hold an inactive point, from its type.
@inline _masked(::Nothing) = false
@inline _masked(::FlowGeometries.Grids.AllActive) = false
@inline _masked(::AbstractArray{Bool}) = true

# Whether `strategy` divides by `filter(mask)`: `Deformable` where a point is masked or the box pads past
# a bounded domain, whose padding is inactive. Elsewhere every window holds the kernel's whole mass.
@inline _divides_by_mass(::AbstractMaskStrategy, mask, tp::Tuple) = false
@inline _divides_by_mass(::Deformable, mask, tp::Tuple) = _masked(mask) || _padded(tp)

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

# FlowTransformBindings allocates every buffer with a trailing batch axis; a single field's are the plain
# vector and mode array sharing that memory.
function _nufft_scratch(gp::NUFFTGridPlan)
    b = gp.batched
    m = FlowTransformBindings.allocate_modes(gp.plan)
    return NUFFTScratch(
        vec(FlowTransformBindings.allocate_values(gp.plan)), dropdims(m; dims = ndims(m)),
        b === nothing ? nothing :
            (values = FlowTransformBindings.allocate_values(b), modes = FlowTransformBindings.allocate_modes(b)),
    )
end

# `Ĝ(k, ℓ)` on the plan's modes, with the split-Nyquist weights, into a host array.
function _nufft_transfer!(transfer::Array{T}, gp::NUFFTGridPlan{T,D}, kernel::Kernels.AbstractFilterKernel,
                          scale::T) where {T,D}
    p = gp.plan
    freqs = ntuple(d -> FlowTransformBindings.mode_frequencies(p, d), Val(D))
    for I in CartesianIndices(transfer)
        k = ntuple(d -> T(2π) * freqs[d][I[d]] / gp.period[d], Val(D))
        w = prod(ntuple(d -> _edge_weight(T, freqs[d][I[d]], gp.counts[d]), Val(D)))
        transfer[I] = w * Kernels.spectral_transfer(kernel, k, scale)
    end
    return transfer
end

_nufft_transfer(gp::NUFFTGridPlan{T}, kernel::Kernels.AbstractFilterKernel, scale::T) where {T} =
    _on_backend(gp.execution,
                _nufft_transfer!(Array{T}(undef, FlowTransformBindings.mode_size(gp.plan)), gp, kernel, scale))

"""
    rescale!(plan::NUFFTFilterPlan, kernel, scale) -> plan

`plan` refilled with the transfer function of `kernel` at `scale`, its transforms and scratch kept. A plan
that divides by the filtered mask's mass (`Deformable` on a masked or bounded node set) holds a mass of its
own scale and raises, as does a kernel reaching past the grid plan's padding.
"""
function rescale!(plan::NUFFTFilterPlan{T}, kernel::Kernels.AbstractFilterKernel, scale::T) where {T<:AbstractFloat}
    _validate_scale(scale)
    _no_window_mass(plan.invrenorm)
    _check_box(plan.grid_plan, kernel, scale)
    _store_transfer!(plan.transfer, plan.grid_plan, kernel, scale)
    return plan
end

# Raises where `kernel` at `scale` wraps around the grid plan's box.
function _check_box(gp::NUFFTGridPlan{T,D}, kernel::Kernels.AbstractFilterKernel, scale::T) where {T,D}
    _check_unbounded_reach(kernel, gp.topology)
    return _check_reach(gp.reach, kernel, scale, Val(D))
end

_no_window_mass(::Nothing) = nothing
_no_window_mass(_) = throw(ArgumentError(
    "this plan divides by the filtered mask's mass at its own scale; build a plan at the new scale"))

_store_transfer!(transfer::Array{T}, gp, kernel, scale::T) where {T} = _nufft_transfer!(transfer, gp, kernel, scale)
_store_transfer!(transfer::AbstractArray{T}, gp, kernel, scale::T) where {T} =
    copyto!(transfer, _nufft_transfer!(Array{T}(undef, size(transfer)), gp, kernel, scale))

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

"""
    _nufft_library(spectral_backend) -> tag or nothing

The FlowTransformBindings tag that runs a nonuniform-FFT filter: the one named, or for
`AutoSpectralBackend` NonuniformFFTs when it is loaded and FINUFFT when only it is; `nothing` when
neither is loaded.
"""
_nufft_library(b::Union{FlowTransformBindings.FINUFFTBackend, FlowTransformBindings.NonuniformFFTsBackend}) = b
_nufft_library(::SpectralBackends.AbstractAutoSpectralBackend) =
    FlowTransformBindings.is_available(FlowTransformBindings.NonuniformFFTsBackend()) ?
        FlowTransformBindings.NonuniformFFTsBackend() :
    FlowTransformBindings.is_available(FlowTransformBindings.FINUFFTBackend()) ?
        FlowTransformBindings.FINUFFTBackend() : nothing
_nufft_library(t::SpectralBackends.AbstractNUFFTSpectralBackend) = throw(ArgumentError(
    "$(nameof(typeof(t))) names no NUFFT library; pass FlowTransformBindings.NonuniformFFTsBackend() " *
    "(`using NonuniformFFTs`) or FlowTransformBindings.FINUFFTBackend() (`using FINUFFT`)."))

spectral_scratch(gp::NUFFTGridPlan) = _nufft_scratch(gp)

"""
    distributed_nufft_grid_plan(nufft, grid, reach; execution::DistributedBackend, batch) -> AbstractGridPlan
    distributed_nufft_filter_plan(grid_plan, kernel, scale, mask_strategy) -> AbstractFilterPlan

A nonuniform-FFT spectral plan whose points are divided among the worker processes, each holding the
library plan over its block of the box padded for `reach`. Methods in the Distributed extension.
"""
function distributed_nufft_grid_plan end
function distributed_nufft_filter_plan end

distributed_nufft_grid_plan(args...; kwargs...) = throw(ArgumentError(
    "DistributedBackend is unavailable — run `using Distributed, SharedArrays` (or use SerialBackend())."))

# `k` of the grid plan for `backend`, its bounded directions padded for `reach`: over this process's points,
# or divided among the worker processes. `modes` is `nothing` for a mode per node, or the mode counts per
# direction; `tol` is `nothing` for the library's default tolerance at the grid's precision.
_nufft_grid_plan_for(nufft, grid, backend, batch, modes, tol, reach) =
    _nufft_grid_plan_for(identity, nufft, grid, backend, batch, modes, tol, reach)
_nufft_grid_plan_for(k::K, nufft, grid, backend::ComputationalBackends.AbstractDistributedBackend, batch,
                     ::Nothing, ::Nothing, reach) where {K} =
    k(distributed_nufft_grid_plan(nufft, grid, reach; execution = backend, batch = batch))
_nufft_grid_plan_for(k::K, nufft, grid, backend::ComputationalBackends.AbstractExecutionBackend, batch, modes, tol,
                     reach) where {K} =
    _nufft_grid_plan(k, nufft, grid; execution = backend, batch = batch, box = _mode_box(modes, reach), tol = tol)
_nufft_grid_plan_for(_, _, _, ::ComputationalBackends.AbstractDistributedBackend, _, _, _, _) = throw(ArgumentError(
    "a nonuniform-FFT plan divided among worker processes takes a mode per node at the default tolerance; " *
    "drop `modes` and `tol`"))

function spectral_grid_plan(
    spectral_backend::_NUFFTRoute, grid::FlowGeometries.Grids.UnstructuredGrid{T,G,D},
    kernel::Kernels.AbstractFilterKernel;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    batch::Union{Nothing,Integer} = nothing,
    modes::Union{Nothing,Tuple} = nothing,
    tol::Union{Nothing,Real} = nothing,
    max_scale::Real = 0,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractCartesianGeometry{T}, D}
    nufft = _nufft_library(spectral_backend)
    nufft === nothing && return nothing
    return _nufft_grid_plan_for(nufft, grid, execution, batch, modes, tol, _kernel_reach(kernel, max_scale, T, Val(D)))
end

"""
    spectral_filter_plan(spectral_backend, grid::UnstructuredGrid{Cartesian}, kernel, scale;
                         mask_strategy = ZeroFill(), execution = AutoBackend(), batch = nothing,
                         modes = nothing, tol = nothing, grid_plan = nothing, scratch = nothing)
        -> AbstractFilterPlan

The nonuniform-FFT filter plan over a Cartesian node set, built on `grid_plan` and `scratch` when
given, otherwise on a box whose bounded directions are padded for `kernel` at `scale`. `batch = nb` adds
a transform with `ntrans = nb`, which `filter_apply_batched!` runs over an `(npts, nb)` array in one
execution per direction. `modes`, one count per direction of a periodic node set, sets the modes the box
holds; `nothing` takes a mode per node. `tol` is the transforms' relative accuracy, `nothing` for
`FlowTransformBindings.default_tolerance` of the grid's precision. The library runs on
`_library_threads(execution)` threads; [`rescale!`](@ref) refills the plan for another scale.
`SharpSpectralKernel` raises on a node set with a bounded direction.
"""
function spectral_filter_plan(
    spectral_backend::_NUFFTRoute,
    grid::FlowGeometries.Grids.UnstructuredGrid{T,G,D},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    batch::Union{Nothing,Integer} = nothing,
    modes::Union{Nothing,Tuple} = nothing,
    tol::Union{Nothing,Real} = nothing,
    grid_plan::Union{Nothing,AbstractGridPlan} = nothing,
    scratch::Union{Nothing,NUFFTScratch} = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractCartesianGeometry{T}, D}
    grid_plan === nothing || return _nufft_filter_plan(grid_plan, scratch, kernel, scale, mask_strategy)
    nufft = _nufft_library(spectral_backend)
    nufft === nothing && _no_spectral_backend(spectral_backend, grid)
    reach = _kernel_reach(kernel, scale, T, Val(D))
    return _nufft_grid_plan_for(nufft, grid, execution, batch, modes, tol, reach) do gp
        _nufft_filter_plan(gp, scratch, kernel, scale, mask_strategy)
    end
end

_nufft_filter_plan(gp::AbstractGridPlan, _, kernel, scale, mask_strategy) =
    distributed_nufft_filter_plan(gp, kernel, scale, mask_strategy)

function _nufft_filter_plan(
    gp::NUFFTGridPlan{T}, scratch, kernel::Kernels.AbstractFilterKernel, scale::T, mask_strategy::AbstractMaskStrategy,
) where {T}
    _check_box(gp, kernel, scale)
    sc = scratch === nothing ? _nufft_scratch(gp) : scratch
    transfer = _nufft_transfer(gp, kernel, scale)
    invrenorm, dmass = _divides_by_mass(mask_strategy, gp.mask, gp.topology) ?
        _window_mass(gp, sc, transfer) : (nothing, nothing)
    return NUFFTFilterPlan(gp, transfer, invrenorm, dmass, mask_strategy, sc)
end

# `1/M` and `∂_d M` over this process's points, `M = filter(mask)` through the plan's own pipeline; the
# padding beyond a bounded domain is inactive, and an inactive point's `1/M` is zero.
function _window_mass(gp::NUFFTGridPlan{T,D}, sc::NUFFTScratch, transfer) where {T,D}
    if gp.mask === nothing
        sc.values .= gp.weights
    else
        @. sc.values = gp.mask * gp.weights
    end
    M̂ = _analysis!(similar(sc.modes), gp.plan, sc.values, gp)
    sc.modes .= M̂ .* transfer
    FlowTransformBindings.nufft_type2!(sc.values, gp.plan, sc.modes)
    invmass = gp.mask === nothing ? _inv_mass.(sc.values) : ifelse.(gp.mask, _inv_mass.(sc.values), zero(T))
    dmass = ntuple(Val(D)) do d
        sc.modes .= M̂ .* transfer .* gp.ik[d]
        FlowTransformBindings.nufft_type2!(sc.values, gp.plan, sc.modes)
        copy(sc.values)
    end
    return invmass, dmass
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

function filter_synthesize_derivative!(
    out::AbstractVector{T}, F̂::AbstractArray, ū::AbstractVector, plan::NUFFTFilterPlan{T}, d::Int,
) where {T}
    gp, sc = plan.grid_plan, plan.scratch
    sc.modes .= F̂ .* plan.transfer .* gp.ik[d]
    FlowTransformBindings.nufft_type2!(sc.values, gp.plan, sc.modes)
    _less_mass_slope!(sc.values, _local(ū, gp.own), plan.dmass, d)
    return _store_filtered!(out, sc.values, plan.invrenorm, gp)
end

function filter_synthesize_derivative!(
    out::AbstractMatrix{T}, F̂::AbstractArray, ū::AbstractMatrix, plan::NUFFTFilterPlan{T}, d::Int,
) where {T}
    b, p = _batch_buffers(plan, out)
    gp = plan.grid_plan
    p.modes .= F̂ .* plan.transfer .* gp.ik[d]
    FlowTransformBindings.nufft_type2!(p.values, b, p.modes)
    _less_mass_slope!(p.values, _local(ū, gp.own), plan.dmass, d)
    return _store_filtered!(out, p.values, plan.invrenorm, gp)
end

# The quotient rule's numerator, `∂_d N − ū ∂_d M`; a plan with no window mass has nothing to subtract.
@inline _less_mass_slope!(c, _, ::Nothing, ::Int) = c
@inline _less_mass_slope!(c, ū, dmass::Tuple, d::Int) = (c .-= ū .* dmass[d]; c)

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
    _batch_analysis(plan.scratch.batched, size(field, 2), _ -> plan.grid_plan.nb, () -> similar(plan.scratch.batched.modes))

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
[`Kernels.truncation_tol`](@ref) of its peak, and the transforms run at that tolerance.

`invden` is the reciprocal of the denominator, zero where it vanishes or, under `Deformable`, at an
inactive node, and `dden` the denominator's derivative along each direction, which the quotient rule of
the filtered field's derivative reads. `bound` is whether the two strategies' denominators differ, which
they do on a masked grid.

# References
- Potts & Steidl (2003), *SIAM J. Sci. Comput.* 24(6), "Fast summation at nonequispaced knots by NFFTs".
"""
struct NUFFTSumFootprint{
    T<:AbstractFloat, GP<:NUFFTGridPlan{T}, A<:AbstractArray{T}, V<:AbstractVector{T}, DD<:Tuple,
    MS<:AbstractMaskStrategy, SC<:NUFFTScratch,
}
    grid_plan::GP
    transfer::A
    invden::V
    dden::DD
    strategy::MS
    bound::Bool
    scratch::SC
end

Base.show(io::IO, fp::NUFFTSumFootprint) = print(io, "NUFFTSumFootprint(", fp.grid_plan.plan, ")")

_driven_footprint(::NUFFTSumFootprint) = true
_engine_of(::NUFFTSumFootprint) = NUFFTSumEngine()

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
    nufft_sum_footprint(identity, grid, kernel, scale; kwargs...)

# `k` of the footprint, called where the library plans' types are concrete.
nufft_sum_footprint(
    k::K, grid::FlowGeometries.Grids.AbstractGrid{G,T}, kernel::Kernels.AbstractFilterKernel, scale::T; kwargs...,
) where {K, T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractCartesianGeometry{T}} =
    _nufft_sum_footprint(k, _node_set(grid), kernel, scale; kwargs...)

function _nufft_sum_footprint(
    k::K, grid::FlowGeometries.Grids.UnstructuredGrid{T,G,D}, kernel::Kernels.AbstractFilterKernel, scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(), tol::T = Kernels.truncation_tol(T)
) where {K, T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractCartesianGeometry{T}, D}
    nufft = _nufft_library(SpectralBackends.AutoSpectralBackend())
    box = _KernelBox(T(Kernels.kernel_radius(kernel, scale, Val(D))), T(Kernels.transfer_cutoff(kernel, scale)))
    return _nufft_grid_plan(nufft, grid; execution, box, tol = tol) do gp
        k(_nufft_sum_from(gp, kernel, scale, mask_strategy))
    end
end

function _nufft_sum_from(gp::NUFFTGridPlan{T,D}, kernel, scale, mask_strategy::AbstractMaskStrategy) where {T,D}
    sc = _nufft_scratch(gp)
    transfer = _nufft_transfer(gp, kernel, scale)
    # The denominator is the same sum over the weights of the nodes the strategy counts.
    deformable = mask_strategy isa Deformable && gp.mask !== nothing
    _mass_density!(sc.values, mask_strategy, gp)
    M̂ = _analysis!(similar(sc.modes), gp.plan, sc.values, gp)
    sc.modes .= M̂ .* transfer
    FlowTransformBindings.nufft_type2!(sc.values, gp.plan, sc.modes)
    invden = deformable ? ifelse.(gp.mask, _inv_mass.(sc.values), zero(T)) : _inv_mass.(sc.values)
    dden = ntuple(Val(D)) do d
        sc.modes .= M̂ .* transfer .* gp.ik[d]
        FlowTransformBindings.nufft_type2!(sc.values, gp.plan, sc.modes)
        copy(sc.values)
    end
    return NUFFTSumFootprint(gp, transfer, invden, dden, mask_strategy, gp.mask !== nothing, sc)
end

# The weights of the nodes a strategy's window mass counts: every node's under `ZeroFill`, the active ones' under
# `Deformable`.
_mass_density!(c, ::AbstractMaskStrategy, gp::NUFFTGridPlan) = (c .= gp.weights)
_mass_density!(c, ::Deformable, gp::NUFFTGridPlan) = _masked_weights!(c, gp.mask, gp.weights)
_masked_weights!(c, ::Nothing, w) = (c .= w)
_masked_weights!(c, mask, w) = (@. c = mask * w)

# The kernel sum is the ratio of the field's series to the denominator's, so a flux under
# `SpectralDerivative()` reuses one analysis for the filtered field and its derivatives.
series_form(::PhysicalFilterPlan{<:NUFFTSumFootprint}) = SeriesRatio()

# Its derivative in the filter width by the same series: the transfer `∂_ℓĜ` over its analyses, divided by the window
# mass it holds, with `∂_ℓM` from `mass_analysis`.
rate_available(::PhysicalFilterPlan{<:NUFFTSumFootprint}, kernel::Kernels.AbstractFilterKernel) =
    Kernels.has_transfer_rate(kernel)
curvature_available(plan::PhysicalFilterPlan{<:NUFFTSumFootprint}, kernel::Kernels.AbstractFilterKernel) =
    rate_available(plan, Kernels.TransferRate(kernel))

function rate_plan(plan::PhysicalFilterPlan{<:NUFFTSumFootprint{T}}, kernel::Kernels.AbstractFilterKernel,
                   scale::T) where {T<:AbstractFloat}
    fp = plan.footprint
    rate = NUFFTSumFootprint(fp.grid_plan, _nufft_transfer(fp.grid_plan, Kernels.TransferRate(kernel), scale), fp.invden,
                             fp.dden, fp.strategy, fp.bound, fp.scratch)
    return PhysicalFilterPlan(rate, plan.grid, plan.strategy, plan.kernel, plan.scale, plan.execution)
end

"""
    mass_analysis(plan) -> M̂

The analysis of the density whose kernel sum is the window mass of the kernel-sum `plan`: every node's weight under
`ZeroFill`, the active nodes' under `Deformable`.
"""
function mass_analysis(plan::PhysicalFilterPlan{<:NUFFTSumFootprint})
    fp = plan.footprint
    gp, sc = fp.grid_plan, fp.scratch
    _mass_density!(sc.values, fp.strategy, gp)
    return _analysis!(similar(sc.modes), gp.plan, sc.values, gp)
end

analyze_buffer(plan::PhysicalFilterPlan{<:NUFFTSumFootprint}, ::AbstractVector) = similar(plan.footprint.scratch.modes)

function filter_analyze!(F̂::AbstractArray, field::AbstractVector, plan::PhysicalFilterPlan{<:NUFFTSumFootprint})
    fp = plan.footprint
    _check_strategy(fp, plan.strategy)
    _load_weighted!(fp.scratch.values, field, fp.grid_plan)
    return _analysis!(F̂, fp.grid_plan.plan, fp.scratch.values, fp.grid_plan)
end

function filter_synthesize!(out::AbstractVector, F̂::AbstractArray, plan::PhysicalFilterPlan{<:NUFFTSumFootprint})
    fp = plan.footprint
    fp.scratch.modes .= F̂ .* fp.transfer
    FlowTransformBindings.nufft_type2!(fp.scratch.values, fp.grid_plan.plan, fp.scratch.modes)
    out .= fp.scratch.values .* fp.invden
    return out
end

function filter_synthesize_derivative!(
    out::AbstractVector, F̂::AbstractArray, ū::AbstractVector, plan::PhysicalFilterPlan{<:NUFFTSumFootprint}, d::Int,
)
    fp = plan.footprint
    c = fp.scratch.values
    fp.scratch.modes .= F̂ .* fp.transfer .* fp.grid_plan.ik[d]
    FlowTransformBindings.nufft_type2!(c, fp.grid_plan.plan, fp.scratch.modes)
    @. out = (c - ū * fp.dden[d]) * fp.invden
    return out
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
    c = sc.values
    _load_weighted!(c, vec(field), gp)
    _analysis!(sc.modes, gp.plan, c, gp)
    sc.modes .*= fp.transfer
    FlowTransformBindings.nufft_type2!(c, gp.plan, sc.modes)
    vec(out) .= c .* fp.invden
    return out
end

_apply_serial!(out, field, grid, fp::NUFFTSumFootprint, strategy) = apply_footprint!(out, field, grid, fp, strategy)
