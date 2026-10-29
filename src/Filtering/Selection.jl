# ---------------------------------------------------------------------------
# Sweep families: one grid plan and one scratch shared by every scale
# ---------------------------------------------------------------------------

"""
    FilterPlanFamily(grid_plan, plans, scratch[, rates, rate_order])

The plans for a sweep of scales over one grid, with the scale-independent
[`AbstractGridPlan`](@ref) and the per-apply [`AbstractFilterScratch`](@ref) they share; either is
`nothing` for an engine with nothing to share. `rates` holds each plan's [`scale_rate`](@ref) when the family
was built with [`ScaleRates`](@ref) or [`ScaleCurvatures`](@ref), the tag `rate_order` holds, and is `nothing`
otherwise. A family is an `AbstractVector` of its plans.

The plans of one family share one scratch, so they must not be applied concurrently; give each
concurrent worker its own family.
"""
struct FilterPlanFamily{
    P, GP<:Union{Nothing,AbstractGridPlan}, PV<:AbstractVector{P},
    SC<:Union{Nothing,AbstractFilterScratch}, RT<:Union{Nothing,AbstractVector}, O<:AbstractScaleRates,
} <: AbstractVector{P}
    grid_plan::GP
    plans::PV
    scratch::SC
    rates::RT
    rate_order::O
end

FilterPlanFamily(grid_plan::GP, plans::PV, scratch::SC, rates::RT = nothing,
                 rate_order::O = NoScaleRates()) where {P, GP, PV<:AbstractVector{P}, SC, RT, O} =
    FilterPlanFamily{P,GP,PV,SC,RT,O}(grid_plan, plans, scratch, rates, rate_order)

Base.size(f::FilterPlanFamily) = size(f.plans)
Base.@propagate_inbounds Base.getindex(f::FilterPlanFamily, i::Int) = f.plans[i]
Base.IndexStyle(::Type{<:FilterPlanFamily}) = Base.IndexLinear()

Base.show(io::IO, f::FilterPlanFamily) =
    print(io, "FilterPlanFamily(", length(f.plans), " scales, ",
          f.grid_plan === nothing ? "no shared grid plan" : nameof(typeof(f.grid_plan)), ")")
Base.show(io::IO, ::MIME"text/plain", f::FilterPlanFamily) = show(io, f)

# `(grid_plan, scratch)` an engine shares across the scales of a sweep, each `nothing` independently.
# Its methods follow `build_footprint`'s dispatch, so the per-scale builder takes what they return.
_sweep_shared(grid, kernel, mask_strategy, method; kwargs...) =
    _spectral_sweep_shared(grid, kernel, mask_strategy, method; kwargs...)

function _spectral_sweep_shared(
    grid, kernel, mask_strategy, method;
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    kwargs...,
)
    method isa Spectral || return (nothing, nothing)
    gp = spectral_grid_plan(transform, grid, kernel; mask_strategy = mask_strategy, kwargs...)
    return (gp, gp === nothing ? nothing : spectral_scratch(gp))
end

function _sweep_shared(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    kernel::Kernels.TopHatKernel,
    mask_strategy::AbstractMaskStrategy,
    method::AbstractFilterMethod;
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    method isa Spectral && return _spectral_sweep_shared(grid, kernel, mask_strategy, method; kwargs...)
    FlowGeometries.Grids.measure_factors(grid) === nothing && return (nothing, nothing)
    gp = _build_prefixsum_grid_plan(grid; mask_strategy = mask_strategy)
    return (gp, _prefixsum_scratch(gp, grid, kernel))
end

function _sweep_shared(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    kernel::SeparableKernel,
    mask_strategy::AbstractMaskStrategy,
    method::AbstractFilterMethod;
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    method isa Spectral && return _spectral_sweep_shared(grid, kernel, mask_strategy, method; kwargs...)
    FlowGeometries.Grids.measure_factors(grid) === nothing && return (nothing, nothing)
    return (nothing, _separable_scratch(grid))
end

"""
    plan_filter_sweep(grid, kernel, scales; mask_strategy = ZeroFill(), method = nothing,
                      cache_byte_budget = default_cache_byte_budget(), scale_rates = NoScaleRates(),
                      kwargs...) -> FilterPlanFamily

One [`plan_filter`](@ref) per scale in `scales`, all sharing one grid-determined half of the engine
and one per-apply scratch. For the prefix-sum top-hat engine the shared half is the measure prefix
scans, the ordered axis-1 coordinates and the mask scan. `method = nothing` takes the grid's default method.
`scale_rates = ScaleRates()` also builds each real-space plan's [`scale_rate`](@ref), which a sweep's exact
filtering spectrum reads, so a family reused across calls builds them once; `ScaleCurvatures()` builds their
second derivatives too, which the spectral slopes read.

Forwards every other [`plan_filter`](@ref) keyword. `cache_byte_budget` bounds the neighbour caches of
the whole sweep: each scale's plan is given what the scales before it left.

```julia
family = plan_filter_sweep(grid, TopHatKernel(), scales)
for (s, plan) in zip(scales, family)
    filter_apply!(out, field, plan)
end
```
"""
function plan_filter_sweep(
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scales::AbstractVector;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    method::Union{Nothing,AbstractFilterMethod} = nothing,
    cache_byte_budget::Integer = default_cache_byte_budget(),
    scale_rates::AbstractScaleRates = NoScaleRates(),
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    meth = method === nothing ? _default_method(grid) : method
    gp, sc = _all_transform_engines(grid, kernel, scales, meth, kwargs) ? (nothing, nothing) :
             _sweep_shared(grid, kernel, mask_strategy, meth; max_scale = maximum(scales; init = zero(T)), kwargs...)
    kw = (; mask_strategy = mask_strategy, method = meth, _shared_keywords(gp, sc)..., kwargs...)
    plans = _plan_each_scale(grid, kernel, scales, kw, Int(cache_byte_budget))
    return FilterPlanFamily(gp, plans, sc, _family_rates(scale_rates, plans), scale_rates)
end

# The shared half as `plan_filter` keywords: only those the engine takes, since `plan_filter` raises on
# any other.
_shared_keywords(::Nothing, ::Nothing) = (;)
_shared_keywords(gp, ::Nothing) = (; grid_plan = gp)
_shared_keywords(gp, sc) = (; grid_plan = gp, scratch = sc)

# One plan per scale, each given the neighbour-cache budget the scales before it left.
function _plan_each_scale(
    grid::FlowGeometries.Grids.AbstractGrid{G,T}, kernel, scales, kw::NamedTuple, budget::Int,
) where {G,T}
    left = Ref(budget)
    return map(scales) do s
        p = plan_filter(grid, kernel, T(s); _budget_keywords(grid, left[])..., kw...)
        left[] -= _cache_bytes(p)
        p
    end
end

_budget_keywords(grid, left::Int) = _takes_cache_budget(grid) ? (; cache_byte_budget = max(left, 0)) : (;)

# Whether every scale of a real-space sweep takes a transform engine, which reads none of the shared
# half the real-space engines build.
function _all_transform_engines(grid::FlowGeometries.Grids.AbstractGrid{G,T}, kernel, scales, method,
                                kwargs) where {G,T}
    method isa RealSpace || return false
    resolved = _resolve_backend(get(kwargs, :execution, ComputationalBackends.AutoBackend()), grid)
    engine = get(kwargs, :engine, AutoEngine())
    return all(s -> _engine(engine, grid, kernel, T(s), resolved) isa Union{PaddedFFTEngine,ZonalFFTEngine}, scales)
end

# The grids whose real-space plans hold an optional neighbour cache under a byte budget.
_takes_cache_budget(::Union{FlowGeometries.Grids.StructuredGrid, FlowGeometries.Grids.CurvilinearGrid}) = true
_takes_cache_budget(::FlowGeometries.Grids.AbstractGrid) = false

# The bytes a plan's neighbour cache holds, counted as the budget counts them.
_cache_bytes(plan::PhysicalFilterPlan) = _cache_bytes(plan.footprint)
_cache_bytes(fp::ScatteredFilterPlan{T}) where {T} =
    fp.cache === nothing ? 0 : length(fp.cache.w) * _cache_entry_bytes(Val(2), T)
_cache_bytes(fp::NDScatteredFilterPlan{N,T}) where {N,T} =
    fp.cache === nothing ? 0 : length(fp.cache.w) * _cache_entry_bytes(Val(N), T)
_cache_bytes(_) = 0

"""
    _default_method(grid) -> AbstractFilterMethod

The method [`plan_filter`](@ref) and [`plan_filter_sweep`](@ref) take on `grid` when the caller names
none: `RealSpace()` on every grid, the compact-kernel convolution that defines coarse graining (Aluie
2019). The transfer-function filter is reached only by an explicit `method = Spectral()`.
"""
_default_method(::FlowGeometries.Grids.AbstractGrid) = RealSpace()

# Grids whose parallel backends decompose over rows through `apply_footprint_row!`.
_row_parallelizable(::FlowGeometries.Grids.StructuredGrid{T,G,2}) where {G,T} = true
_row_parallelizable(::FlowGeometries.Grids.CurvilinearGrid) = true
_row_parallelizable(::FlowGeometries.Grids.AbstractGrid) = false

"""
    _is_flat_cell(grid) -> Bool

Whether `grid` names a cell by a single integer, `FlowGeometries.Grids.FlatCells()`: a node set, or a
spherical pixelization whose cells carry one id. The node CSR engine reads such a grid only through
`fold_within(grid, t)`, `measure(grid, j)` and `isactive(grid, t)`. A `CartesianCells()` grid names a
cell by an index tuple and takes the rectilinear or curvilinear engines.
"""
@inline _is_flat_cell(grid::FlowGeometries.Grids.AbstractGrid) =
    FlowGeometries.Grids.cell_address(grid) === FlowGeometries.Grids.FlatCells()

# Dispatch handle for the trait: a method that must vary with it takes this as its first argument.
@inline _cell_address(grid::FlowGeometries.Grids.AbstractGrid) =
    FlowGeometries.Grids.cell_address(grid)

# Grids with point-indexed footprints, whose parallel backends split over points: each point writes
# only its own cell.
_nd_parallelizable(::FlowGeometries.Grids.StructuredGrid{T,G,1}) where {G,T} = true
_nd_parallelizable(::FlowGeometries.Grids.StructuredGrid{T,G,3}) where {G,T} = true
_nd_parallelizable(grid::FlowGeometries.Grids.AbstractGrid) = _is_flat_cell(grid)

# Whether `grid` has a parallel path for a concrete backend.
_backend_supported(grid::FlowGeometries.Grids.AbstractGrid, ::ComputationalBackends.SerialBackend) = true
_backend_supported(grid::FlowGeometries.Grids.AbstractGrid, ::ComputationalBackends.ThreadedBackend) = _row_parallelizable(grid) || _nd_parallelizable(grid)
_backend_supported(grid::FlowGeometries.Grids.AbstractGrid, ::ComputationalBackends.DistributedBackend) =
    _row_parallelizable(grid) || _nd_parallelizable(grid)
_backend_supported(grid::FlowGeometries.Grids.AbstractGrid, ::ComputationalBackends.GPUBackend) =
    _row_parallelizable(grid) || _nd_parallelizable(grid)
_backend_supported(grid::FlowGeometries.Grids.AbstractGrid, ::ComputationalBackends.MPIBackend) =
    _row_parallelizable(grid) || _nd_parallelizable(grid)

# Whether a capability's extension is loaded: `false` here, `true` from the extension that provides it.
# `_HostFFT` is an AbstractFFTs implementation for host arrays (FFTW), `_DeviceFFT` FFTs of device arrays
# through their own AbstractFFTs implementation, `_Threads` the threaded applies (OhMyThreads).
abstract type _Capability end
struct _HostFFT <: _Capability end
struct _DeviceFFT <: _Capability end
struct _Threads <: _Capability end
_available(::_Capability) = false

# The planner keywords an FFT of arrays like `A` takes: the caller's `fft_options`, passed to the provider's
# `plan_*` as they stand, after the thread count where the provider's extension declares one.
_fft_options(::AbstractArray, ::Int, options::NamedTuple) = options

# `fft_options` as a builder's keyword, none when empty: a transform library that plans no FFT of its own
# takes no such keyword.
_fft_keywords(::NamedTuple{()}) = (;)
_fft_keywords(options::NamedTuple) = (; fft_options = options)

# Whether an FFT provider's extension is loaded for arrays in `execution`'s memory.
_fft_provided(execution) = _host_memory(execution) ? _available(_HostFFT()) : _available(_DeviceFFT())

# A transform runs where it evaluates the same sum as the kernel's real-space engine in fewer operations.
# Counts are per apply, a fused multiply-add counting one: one per tap of the direct sum and of each
# separable pass; for the prefix-sum top-hat seven per entry of its numerator scan (a compensated
# addition) and four per cell and band (two scan differences, their sum, the weighted accumulate);
# `2.5 n log₂ n` per real transform of length `n` (forward or inverse); and per spectral value four for a
# complex product, two for a real weight times a complex value accumulated.
_rfft_ops(n::Integer) = 2.5 * n * log2(n)
_prefixsum_ops(ncell::Integer, bands::Integer) = Float64(ncell) * 8 + 4.0 * bands

# Transform length along one axis: the axis itself where it wraps, else the least even 2·3·5-smooth length
# holding the data and the widest offset `w` between two of its cells.
_padded_length(N::Int, w::Int, periodic::Bool) = periodic ? N : _even_smooth_length(N + min(w, N - 1))

"""
    _padded_halfwidths(grid, kernel, scale) -> NTuple{N,Int}

The half-widths in cells of the direct footprint the padded transform samples on a uniform Cartesian
grid: the ball of `kernel_radius` for a radial kernel, the separable engine's stencil for a separable
one. A bounded axis reaches no farther than its extent.
"""
function _padded_halfwidths(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, kernel::Kernels.AbstractFilterKernel, scale::T,
) where {T<:AbstractFloat, N, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    rad = Kernels.kernel_radius(kernel, scale, Val(N))
    return _clamp_bounded(grid, ntuple(d -> ceil(Int, rad / _axis_step(grid, d)), Val(N)))
end

_padded_halfwidths(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, kernel::SeparableKernel, scale::T,
) where {T<:AbstractFloat, N, G<:FlowGeometries.Geometry.CartesianGeometry{T}} =
    _clamp_bounded(grid, _separable_halfwidths(grid, kernel, scale))

# A `CellOverlap` top-hat reaches every cell its support touches, half a cell past the radius.
function _padded_halfwidths(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, kernel::Kernels.TopHatKernel{Kernels.CellOverlap}, scale::T,
) where {T<:AbstractFloat, N, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    rad = Kernels.kernel_radius(kernel, scale, Val(N))
    return _clamp_bounded(grid, ntuple(d -> ceil(Int, rad / _axis_step(grid, d) + T(1) / 2), Val(N)))
end

function _clamp_bounded(grid::FlowGeometries.Grids.StructuredGrid, w::NTuple{N,Int}) where {N}
    dims = FlowGeometries.Grids.size_tuple(grid)
    return ntuple(d -> FlowGeometries.Grids.isperiodic(grid, d) ? w[d] : min(w[d], dims[d] - 1), Val(N))
end

@inline _axis_step(grid::FlowGeometries.Grids.StructuredGrid{T}, d::Int) where {T} =
    abs(T(step(FlowGeometries.Grids.coordinates(grid, d))))

# The lattice offsets of a uniform grid within distance `rad` across the directions `ds`, a bounded
# direction reaching no farther than its extent.
function _ball_offsets(grid::FlowGeometries.Grids.StructuredGrid{T}, rad::T, ds::NTuple{K,Int}) where {T,K}
    dims = FlowGeometries.Grids.size_tuple(grid)
    reach(d) = (r = floor(Int, rad / _axis_step(grid, d)); FlowGeometries.Grids.isperiodic(grid, d) ? r : min(r, dims[d] - 1))
    d1, rest = first(ds), Base.tail(ds)
    count = 0
    for o in CartesianIndices(map(d -> (-reach(d)):reach(d), rest))
        r2 = rad^2 - sum(ntuple(k -> (o[k] * _axis_step(grid, rest[k]))^2, Val(K - 1)); init = zero(T))
        r2 < 0 && continue
        h = floor(Int, sqrt(r2) / _axis_step(grid, d1))
        count += 2 * (FlowGeometries.Grids.isperiodic(grid, d1) ? h : min(h, dims[d1] - 1)) + 1
    end
    return count
end

"""
    _padded_fft_ops(grid, kernel, scale) -> (transform, realspace)

The operations of one apply on a uniform Cartesian grid by the zero-padded transform and by the
real-space engine `build_footprint` takes for `kernel` ([`_real_space_ops`](@ref)).
"""
function _padded_fft_ops(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, kernel::Kernels.AbstractFilterKernel, scale::T,
) where {T<:AbstractFloat, N, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    dims = FlowGeometries.Grids.size_tuple(grid)
    w = _padded_halfwidths(grid, kernel, scale)
    P = ntuple(d -> _padded_length(dims[d], w[d], FlowGeometries.Grids.isperiodic(grid, d)), Val(N))
    transform = 2 * _rfft_ops(prod(P)) + 4 * (P[1] ÷ 2 + 1) * prod(Base.tail(P); init = 1)
    return transform, _real_space_ops(grid, kernel, scale)
end

"""
    _real_space_ops(grid, kernel, scale) -> Float64

The operations of one apply on a uniform Cartesian grid by the real-space engine for `kernel`: the
direct sum over the ball of radius `kernel_radius`, the prefix-sum top-hat (in two and three
directions), or the separable passes.
"""
function _real_space_ops(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, kernel::Kernels.AbstractFilterKernel, scale::T,
) where {T<:AbstractFloat, N, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    rad = Kernels.kernel_radius(kernel, scale, Val(N))
    return Float64(_ball_offsets(grid, rad, ntuple(identity, Val(N)))) * prod(FlowGeometries.Grids.size_tuple(grid))
end

# Two directions: a band per row of the disk.
function _real_space_ops(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2}, kernel::Kernels.TopHatKernel, scale::T,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    n = prod(FlowGeometries.Grids.size_tuple(grid))
    rad = Kernels.kernel_radius(kernel, scale)
    return _prefixsum_ops(n, n * _ball_offsets(grid, rad, (2,))) +
           _edge_ops(kernel, n, rad, (_axis_step(grid, 1), _axis_step(grid, 2)))
end

# Three directions: a band per `(j, k)` offset in the disk across directions 2 and 3.
function _real_space_ops(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,3}, kernel::Kernels.TopHatKernel, scale::T,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    n = prod(FlowGeometries.Grids.size_tuple(grid))
    rad = Kernels.kernel_radius(kernel, scale)
    return _prefixsum_ops(n, n * _ball_offsets(grid, rad, (2, 3))) +
           _edge_ops(kernel, n, rad, ntuple(d -> _axis_step(grid, d), Val(3)))
end

# A `CellOverlap` top-hat stores each cell's value once and adds a tap of the direct sum per cell its support's
# boundary cuts: a circle of radius `R` crosses about `4R/Δ` lines of each direction, a sphere about `2πR²` over
# each face area.
_edge_ops(::Kernels.AbstractFilterKernel, n, rad::T, Δ) where {T} = zero(T)
_edge_ops(::Kernels.TopHatKernel{Kernels.CellOverlap}, n, rad::T, Δ::NTuple{2}) where {T} =
    T(n) * (1 + 4 * rad * (1 / Δ[1] + 1 / Δ[2]))
_edge_ops(::Kernels.TopHatKernel{Kernels.CellOverlap}, n, rad::T, Δ::NTuple{3}) where {T} =
    T(n) * (1 + T(2π) * rad^2 * (1 / (Δ[2] * Δ[3]) + 1 / (Δ[1] * Δ[3]) + 1 / (Δ[1] * Δ[2])))

function _real_space_ops(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, kernel::SeparableKernel, scale::T,
) where {T<:AbstractFloat, N, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    w = _clamp_bounded(grid, _separable_halfwidths(grid, kernel, scale))
    return T(prod(FlowGeometries.Grids.size_tuple(grid))) * sum(d -> 2w[d] + 1, 1:N)
end

"""
    _zonal_fft_ops(grid, kernel, scale) -> (transform, realspace)

The operations of one apply on a sphere of uniform longitude by the transform along longitude and by
the real-space engine: the prefix-sum top-hat for `TopHatKernel`, the banded sum otherwise. Per pair of
latitude rows the taps are the longitude offsets within great-circle distance `rad`,
`sin²(Δλ/2) ≤ (sin²(rad/2R) − sin²(Δφ/2)) / (cos φ cos φ′)`, and a pair with any is a band.
"""
function _zonal_fft_ops(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2}, kernel::Kernels.AbstractFilterKernel, scale::T,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.SphericalGeometry{T}}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    R = T(FlowGeometries.Geometry.radius(FlowGeometries.Grids.grid_geometry(grid)))
    lat = FlowGeometries.Grids.coordinates(grid, 2)
    dλ = abs(T(step(FlowGeometries.Grids.coordinates(grid, 1))))
    rad = _query_radius(grid, kernel, scale)
    min_dφ = R * T(FlowGeometries.Grids.minimum_spacing(grid, 2))
    dj_lim = (isfinite(min_dφ) && min_dφ > 0) ? min(Ny - 1, ceil(Int, rad / min_dφ)) : 0
    h = sin(min(rad / (2R), T(π) / 2))^2
    taps = 0
    pairs = 0
    bands = 0
    reach = 0
    for j in 1:Ny, jj in max(1, j - dj_lim):min(Ny, j + dj_lim)
        pairs += 1
        s = sin((lat[jj] - lat[j]) / 2)^2
        s <= h || continue
        bands += 1
        c = cos(lat[j]) * cos(lat[jj])
        half = (c <= 0 || h - s >= c) ? Nx : floor(Int, 2 * asin(sqrt((h - s) / c)) / dλ)
        taps += min(Nx, 2 * half + 1)
        reach = max(reach, min(half, Nx - 1))
    end
    Px = FlowGeometries.Grids.isperiodic(grid, 1) ? Nx : _even_smooth_length(Nx + reach)
    realspace = kernel isa Kernels.TopHatKernel ?
        _prefixsum_ops(Nx * Ny, Nx * bands) + _edge_ops(kernel, Nx * Ny, rad, (R * dλ, min_dφ)) :
        T(taps) * Nx
    return 2 * Ny * _rfft_ops(Px) + 2 * (Px ÷ 2 + 1) * pairs, realspace
end

# The padded transform evaluates the direct footprint sum of any kernel on a uniform Cartesian grid.
_takes_padded_fft(grid, kernel, scale, resolved) = false
_takes_padded_fft(grid::FlowGeometries.Grids.StructuredGrid, kernel, scale, resolved) =
    _with_spacings(sp -> _takes_padded_fft(sp, grid, kernel, scale, resolved), grid)
_takes_padded_fft(::Tuple, grid, kernel, scale, resolved) = false
function _takes_padded_fft(
    ::NTuple{N,FlowGeometries.Axes.UniformSpacing}, grid::FlowGeometries.Grids.StructuredGrid{T,G,N},
    kernel::Kernels.AbstractFilterKernel, scale, resolved,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}, N}
    _fft_provided(resolved) || return false
    transform, realspace = _padded_fft_ops(grid, kernel, T(scale))
    return transform < realspace
end

# The transform along longitude evaluates the sum on a sphere whose longitude is uniform (a ring or a
# regional box), latitude any axis, when the weights depend on the longitude offset alone.
_takes_zonal_fft(grid, kernel, scale, resolved) = false
_takes_zonal_fft(grid::FlowGeometries.Grids.StructuredGrid, kernel, scale, resolved) =
    _with_spacings(sp -> _takes_zonal_fft(sp, grid, kernel, scale, resolved), grid)
_takes_zonal_fft(::Tuple, grid, kernel, scale, resolved) = false
function _takes_zonal_fft(
    ::Tuple{FlowGeometries.Axes.UniformSpacing,Any}, grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    kernel::Kernels.AbstractFilterKernel, scale, resolved,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.SphericalGeometry{T}}
    (_fft_provided(resolved) && _zonal_weights(grid, kernel, T(scale))) || return false
    transform, realspace = _zonal_fft_ops(grid, kernel, T(scale))
    return transform < realspace
end

# The Cartesian grids filtered over their cells as points: node sets and curvilinear meshes.
const _CartesianPoints{T,G} = Union{FlowGeometries.Grids.UnstructuredGrid{T,G}, FlowGeometries.Grids.CurvilinearGrid{T,G}}

# A top-hat over Cartesian points takes the binned engine on the host: its plan is one sweep over `O(n)`
# memory, its apply reads `O((r/h)^{D−1})` bin runs and partial bins per target, where the neighbour list
# holds `O((r/h)^D)` entries and is built in two passes over them. A curvilinear mesh's cells under
# `CellOverlap` add the cells near the disk's edge by their area inside it.
_takes_binned_tophat(grid, kernel) = false
_takes_binned_tophat(::_CartesianPoints{T,G}, ::Kernels.TopHatKernel) where {T, G<:FlowGeometries.Geometry.AbstractCartesianGeometry{T}} = true

# A kernel with a transfer cutoff over Cartesian points takes the nonuniform-FFT sum where building its plan
# and applying it once costs fewer operations than building the neighbour list and applying it once, or
# where the list's bytes exceed the cache budget.
_takes_nufft_sum(grid, kernel, scale) = false
function _takes_nufft_sum(
    grid::_CartesianPoints{T,G}, kernel::Kernels.AbstractFilterKernel, scale,
) where {T, G<:FlowGeometries.Geometry.AbstractCartesianGeometry{T}}
    K = Kernels.transfer_cutoff(kernel, T(scale))
    (K === nothing || _nufft_library(SpectralBackends.AutoSpectralBackend()) === nothing) && return false
    transform, entries, build = _nufft_sum_ops(grid, kernel, T(scale), K)
    return 2 * transform < build + entries || entries * (sizeof(Int) + sizeof(T)) > default_cache_byte_budget()
end

"""
    _nufft_sum_ops(grid, kernel, scale, cutoff) -> (transform, entries, build)

The operations of one apply over Cartesian points by the nonuniform-FFT sum, and the neighbour list's
entries (one operation each per apply) and the operations of its build. The sum is two transforms of
`M·w^D` spreading and `2.5 N log₂ N` over the twice-oversampled box of `n_d = 2⌊K L_d/2π⌋ + 1` modes, `w`
the spreading width `⌈−log₁₀ tol⌉ + 1` and `L_d` the box of [`_KernelBox`](@ref); building its plan is one
such apply, the window masses. The list holds each point's neighbours at the mean density,
`V_D r^D · M/ΣA`, and is built from the point bins ([`_binned_csr`](@ref)) in two passes over the points of
the bins within reach, `D + 1` operations each (a squared distance and its test), and a weight and a store
per entry.
"""
function _nufft_sum_ops(grid, kernel, scale::T, K::T, tol::T = Kernels.truncation_tol(T)) where {T<:AbstractFloat}
    D = FlowGeometries.Grids.ncoordinates(grid)
    M = length(FlowGeometries.Grids.mask(grid))
    R = T(Kernels.kernel_radius(kernel, scale, Val(D)))
    ball = D == 1 ? 2R : D == 2 ? T(π) * R^2 : T(4π) / 3 * R^3
    entries = Float64(M) * min(M, ball * M / sum(FlowGeometries.Grids.measure(grid)))
    _, h, nb, _, _ = FlowGeometries.Grids._binned_lattice(grid, nothing, Val(D))
    lines, _, wout = _binned_lines(h, R, R)
    candidates = Float64(M) * M / prod(nb) * sum(q -> 2 * wout[q] + 1, eachindex(lines))
    build = 2 * (D + 1) * candidates + 2 * entries
    n = ntuple(_kernel_dim(grid)) do d
        x = FlowGeometries.Grids.coordinates(grid, d)
        L = FlowGeometries.Grids.isperiodic(grid, d) ? T(FlowGeometries.Grids.period(grid, d)) :
            maximum(x) - minimum(x) + R
        2 * floor(Int, K * L / T(2π)) + 1
    end
    w = ceil(Int, -log10(tol)) + 1
    transform = 2 * (Float64(M) * w^D + _rfft_ops(prod(2 .* n))) + prod(n)
    return transform, entries, build
end

"""
    _select_engine(grid, kernel, scale, resolved) -> AbstractEngine

The engine `AutoEngine` takes on the resolved backend. On a backend whose applies run in this process'
loops: the binned top-hat and the nonuniform-FFT sum over Cartesian points; there and on a GPU backend,
the padded and zonal transforms over rectilinear grids where an FFT library serves the backend's memory;
each where it applies and costs fewer operations than the footprint, the footprint otherwise.
[`check_setup`](@ref) reports from the same choice.
"""
function _select_engine(grid, kernel, scale, resolved::ComputationalBackends.AbstractExecutionBackend)
    if _host_backend(resolved)
        _takes_binned_tophat(grid, kernel) && return BinnedTopHatEngine()
        _takes_nufft_sum(grid, kernel, scale) && return NUFFTSumEngine()
    end
    if _transform_backend(resolved)
        _takes_padded_fft(grid, kernel, scale, resolved) && return PaddedFFTEngine()
        _takes_zonal_fft(grid, kernel, scale, resolved) && return ZonalFFTEngine()
    end
    return FootprintEngine()
end

_engine(::AutoEngine, grid, kernel, scale, resolved) = _select_engine(grid, kernel, scale, resolved)
_engine(e::AbstractEngine, grid, kernel, scale, resolved) = e

# `k` of the footprint of `engine` for the spacing tuple `sp` ([`_with_spacings`](@ref)), called where the
# footprint's type is concrete; `kwargs` reach the grid's own `build_footprint`.
_build_engine(k::K, ::PaddedFFTEngine, sp, grid, kernel, scale, strategy, resolved; fft_options::NamedTuple = (;),
              kwargs...) where {K} =
    k(padded_fft_footprint(sp, grid, kernel, scale; mask_strategy = strategy, execution = resolved, fft_options))
_build_engine(k::K, ::ZonalFFTEngine, sp, grid, kernel, scale, strategy, resolved; fft_options::NamedTuple = (;),
              kwargs...) where {K} =
    k(zonal_fft_footprint(sp, grid, kernel, scale; mask_strategy = strategy, execution = resolved, fft_options))
_build_engine(k::K, ::BinnedTopHatEngine, _, grid, kernel, scale, strategy, resolved; kwargs...) where {K} =
    k(binned_tophat_plan(grid, kernel, scale; mask_strategy = strategy, kwargs...))
_build_engine(k::K, ::NUFFTSumEngine, _, grid, kernel, scale, strategy, resolved; kwargs...) where {K} =
    nufft_sum_footprint(k, grid, kernel, scale; mask_strategy = strategy, execution = resolved)
_build_engine(k::K, ::FootprintEngine, ::Nothing, grid, kernel, scale, strategy, resolved; kwargs...) where {K} =
    k(build_footprint(grid, kernel, scale; kwargs...))
_build_engine(k::K, ::FootprintEngine, sp::Tuple, grid, kernel, scale, strategy, resolved; kwargs...) where {K} =
    k(_structured_footprint(sp, grid, kernel, scale; kwargs...))

# A radial kernel's weights depend on the offsets alone; a `HighOrderKernel`'s do where one correction
# serves each latitude row.
_zonal_weights(grid, kernel::Kernels.AbstractFilterKernel, scale) = Kernels.is_radial(kernel)
_zonal_weights(grid, kernel::Kernels.HighOrderKernel, scale) = _highorder_shares_rows(grid, kernel, scale)


# The backends whose applies run in this process's own loops, where the host engines
# (`_driven_footprint`) run.
_host_backend(b::ComputationalBackends.AbstractExecutionBackend) =
    b isa Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend}

# The backends the padded and zonal transforms apply on: the host loops' and a GPU backend's, whose
# extension applies them by broadcasts and library transforms (`_resident_apply!`).
_transform_backend(b::ComputationalBackends.AbstractExecutionBackend) =
    _host_backend(b) || b isa ComputationalBackends.AbstractGPUBackend

# Resolves `AutoBackend` against this package's extensions and `grid`.
# `ComputationalBackends.resolve_backend(::AutoBackend)` is ComputationalBackends' own signature, shared
# by all its consumers.
@inline function _resolve_backend(
    backend::ComputationalBackends.AbstractExecutionBackend, grid::FlowGeometries.Grids.AbstractGrid,
)
    backend isa ComputationalBackends.AbstractAutoBackend ||
        return ComputationalBackends.resolve_backend(backend)
    threaded = ComputationalBackends.ThreadedBackend()
    return (Threads.nthreads() > 1 && _available(_Threads()) && _backend_supported(grid, threaded)) ?
        threaded : ComputationalBackends.SerialBackend()
end

# Raises for an explicit non-serial `backend` that `grid` has no parallel path for; `AutoBackend()`
# resolves to serial there instead.
function _check_backend_compatible(grid::FlowGeometries.Grids.AbstractGrid, backend::ComputationalBackends.AbstractExecutionBackend)
    if !(backend isa ComputationalBackends.AutoBackend) && !(backend isa ComputationalBackends.SerialBackend) && !_backend_supported(grid, _resolve_backend(backend, grid))
        throw(ArgumentError(
            "execution = $(typeof(backend)) has no parallel path on $(typeof(grid)). Pass " *
            "`execution = SerialBackend()` to run serially, or `execution = AutoBackend()` to let the " *
            "library choose.",
        ))
    end
    return nothing
end

"""
    plan_filter(grid, kernel, scale; mask_strategy = ZeroFill(), execution = AutoBackend(),
                method = _default_method(grid), transform = AutoSpectralBackend(), kwargs...)

The plan for a grid with no rectilinear or curvilinear engine of its own; `method` defaults to
[`_default_method`](@ref). In real space a flat-cell grid (see [`_is_flat_cell`](@ref)) has the node
CSR engine over its own ball query, and any other grid raises. Under `method = Spectral()` the grid is
filtered over its cells by the nonuniform transform of its geometry: NUFFT for a Cartesian one, NUFSHT
for a spherical one.

`RealSpace()` applies the kernel with its compact support. A transform is exact for a band-limited
field and its per-apply cost is independent of the filter scale; its support is global.
"""
function plan_filter(
    grid::FlowGeometries.Grids.AbstractGrid,
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    method::AbstractFilterMethod = _default_method(grid),
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    engine::AbstractEngine = AutoEngine(),
    kwargs...,
) where {T<:AbstractFloat}
    _validate_scale(scale)
    method isa Spectral && return spectral_filter_plan(
        transform, grid, kernel, scale; mask_strategy = mask_strategy, execution = execution, kwargs...,
    )
    isempty(kwargs) || _unused_keywords(kwargs, method)
    _check_backend_compatible(grid, execution)
    resolved = _resolve_backend(execution, grid)
    return _physical_plan(resolved, engine, grid, kernel, scale, mask_strategy, (; execution = resolved))
end

# The real-space plan on the resolved backend, behind a function barrier: `_resolve_backend` reads the
# thread count. `kw` reaches the grid's own `build_footprint`.
_physical_plan(
    resolved::ComputationalBackends.AbstractExecutionBackend, engine::AbstractEngine, grid, kernel, scale,
    mask_strategy::AbstractMaskStrategy, kw::NamedTuple,
) = _engine_plan(_engine(engine, grid, kernel, scale, resolved), resolved, grid, kernel, scale, mask_strategy, kw)

# The plan of the engine `e` the selection chose, which `AutoEngine` reads from values: a barrier, inside
# which the footprint is built and wrapped once per spacing tuple, each call on concrete types.
_engine_plan(
    e::AbstractEngine, resolved::ComputationalBackends.AbstractExecutionBackend, grid, kernel, scale,
    mask_strategy::AbstractMaskStrategy, kw::NamedTuple,
) = _with_spacings(grid) do sp
    _build_engine(e, sp, grid, kernel, scale, mask_strategy, resolved; kw...) do fp
        PhysicalFilterPlan(prepare_workspace(resolved, grid, fp), grid, mask_strategy, kernel, scale, resolved)
    end
end

# A curvilinear grid's real-space engine is the scattered per-point footprint, whose neighbour cache
# `cache_strategy` controls.
function plan_filter(
    grid::FlowGeometries.Grids.CurvilinearGrid{T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    method::AbstractFilterMethod = RealSpace(),
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    cache_strategy::AbstractCacheStrategy = AutoCache(),
    cache_byte_budget::Integer = default_cache_byte_budget(),
    engine::AbstractEngine = AutoEngine(),
    kwargs...,
) where {T<:AbstractFloat}
    _validate_scale(scale)
    if method isa Spectral
        # Over its cells as a node set (engines/NodeSpectral).
        return spectral_filter_plan(transform, grid, kernel, scale; mask_strategy = mask_strategy, execution = execution, kwargs...)
    end
    isempty(kwargs) || _unused_keywords(kwargs, method)
    resolved = _resolve_backend(execution, grid)
    _check_backend_compatible(grid, execution)
    return _physical_plan(resolved, engine, grid, kernel, scale, mask_strategy,
                          (; cache_strategy = cache_strategy, cache_byte_budget = cache_byte_budget))
end

"""
    filter_apply!(out, field, plan) -> out

Apply the prebuilt `plan` (see [`plan_filter`](@ref)) to `field`, writing `out`, on the backend the
plan was built for. Every backend applies the footprint stored in `plan`.
"""
function filter_apply!(out::AbstractArray, field::AbstractArray, plan::PhysicalFilterPlan)
    if plan.execution isa ComputationalBackends.SerialBackend
        return _apply_serial!(out, field, plan.grid, plan.footprint, plan.strategy)
    elseif plan.execution isa ComputationalBackends.ThreadedBackend
        return threaded_filter_field!(out, field, plan.grid, plan.kernel, plan.scale, plan.strategy, plan.footprint)
    elseif plan.execution isa ComputationalBackends.DistributedBackend
        return distributed_filter_field!(out, field, plan.grid, plan.kernel, plan.scale, plan.strategy, plan.footprint)
    elseif plan.execution isa ComputationalBackends.GPUBackend
        return gpu_filter_field!(plan.execution, out, field, plan.grid, plan.kernel, plan.scale, plan.strategy, plan.footprint)
    elseif plan.execution isa ComputationalBackends.MPIBackend
        return mpi_filter_field!(plan.execution, out, field, plan.grid, plan.kernel, plan.scale, plan.strategy, plan.footprint)
    else
        throw(ArgumentError("Unsupported execution backend: $(typeof(plan.execution))"))
    end
end

# Whether the device extension has a fused batch kernel for `plan`'s engine; every other engine takes
# the slice loop.
_gpu_batched_supported(::AbstractFilterPlan) = false

"""
    filter_apply_batched!(out, field, plan) -> out

Apply `plan` to each slice along a trailing batch axis: `out` and `field` are `(spatial..., Nb)` over
the plan's grid. On a device with a fused kernel for the engine this is one launch of
`prod(spatial) * Nb` work items; otherwise it loops over the slices. For several separate arrays on
one grid, see [`filter_apply_batch!`](@ref).
"""
function filter_apply_batched!(out::AbstractArray, field::AbstractArray, plan::AbstractFilterPlan)
    spatial = FlowGeometries.Grids.size_tuple(plan.grid)
    valR = Val(length(spatial))
    _check_batched_shape(out, "out", spatial, valR)
    _check_batched_shape(field, "field", spatial, valR)
    size(out) == size(field) || throw(DimensionMismatch(
        "filter_apply_batched! got out $(size(out)) and field $(size(field))",
    ))
    if plan.execution isa ComputationalBackends.GPUBackend && _gpu_batched_supported(plan)
        gpu_filter_field_batched!(
            plan.execution, out, field, plan.grid, plan.kernel, plan.scale, plan.strategy, plan.footprint,
        )
        return out
    end
    d = ndims(out)
    for b in axes(out, d)
        filter_apply!(selectdim(out, d, b), selectdim(field, d, b), plan)
    end
    return out
end

function _check_batched_shape(
    A::AbstractArray, name::AbstractString, spatial::NTuple{R,Int}, ::Val{R},
) where {R}
    ndims(A) == R + 1 || throw(DimensionMismatch(
        "$name has $(ndims(A)) dimensions; filter_apply_batched! expects $R spatial + 1 batch",
    ))
    ntuple(i -> size(A, i), Val(R)) == spatial || throw(DimensionMismatch(
        "$name's leading dimensions $(ntuple(i -> size(A, i), Val(R))) do not match grid shape $spatial",
    ))
    return nothing
end

"""
    analyze_buffer(plan, field) -> F̂ or nothing
    filter_analyze!(F̂, field, plan) -> F̂
    filter_synthesize!(out, F̂, plan) -> out

A spectral apply in two halves: `filter_analyze!`, the forward transform, and `filter_synthesize!`,
the multiply by `Ĝ(|k|, ℓ)` and the inverse transform. Only the second depends on the scale, so a sweep
analyzes each field once and synthesizes it at every scale: `F̂` analyzed with one plan may be
synthesized with the plan for any other scale over the same grid.

`analyze_buffer` returns a buffer for `F̂`, or `nothing` for an engine with no scale-independent half;
callers then use [`filter_apply!`](@ref).
"""
function analyze_buffer end
function filter_analyze! end
function filter_synthesize! end

# Real-space engines have no scale-independent half to hoist.
analyze_buffer(::AbstractFilterPlan, ::AbstractArray) = nothing

# A batch's analysis buffer, `make()`, for a plan holding `batched`, its transforms over a batch of
# `width(batched)` fields; `nothing` for a plan holding none, which applies field by field. A batch of
# another width raises, as the apply does.
@inline _batch_analysis(::Nothing, ::Integer, _, _) = nothing
@inline function _batch_analysis(batched, got::Integer, width::W, make::F) where {W,F}
    nb = width(batched)
    got == nb || throw(DimensionMismatch("the plan was built for a batch of $nb fields; got $got"))
    return make()
end

function gpu_filter_field_batched!(args...; kwargs...)
    throw(ArgumentError("GPUBackend is unavailable — run `using KernelAbstractions` (or use SerialBackend())."))
end

# Several separate fields on one device footprint: their launches are enqueued back to back and the
# host waits once, at the end of the batch.
function gpu_filter_fields!(args...; kwargs...)
    throw(ArgumentError("GPUBackend is unavailable — run `using KernelAbstractions` (or use SerialBackend())."))
end

"""
    filter_apply_batch!(outs, fields, plan) -> outs

Apply one prebuilt `plan` to several separate fields on its grid, writing `outs[i]` from `fields[i]`;
the result equals [`filter_apply!`](@ref) per field. The scattered footprints derive each target
point's neighbourhood once for the whole batch.

`outs` and `fields` are equal-length collections of matching arrays: an `NTuple{K,V}` for a batch size
known at compile time (stack accumulators), or an `AbstractVector`. Arrays with a trailing batch axis
go through [`filter_apply_batched!`](@ref). For independent fields on different grids, see
[`filter_slices!`](@ref).
"""
function filter_apply_batch!(outs, fields, plan::PhysicalFilterPlan)
    _batched_fields(outs, plan) && return filter_apply_batch_trailing!(outs, fields, plan)
    if plan.execution isa ComputationalBackends.SerialBackend
        return _apply_serial_batch!(outs, fields, plan.grid, plan.footprint, plan.strategy)
    elseif plan.execution isa ComputationalBackends.ThreadedBackend
        return threaded_filter_fields!(outs, fields, plan.grid, plan.kernel, plan.scale, plan.strategy, plan.footprint)
    elseif plan.execution isa ComputationalBackends.GPUBackend
        return gpu_filter_fields!(
            plan.execution, outs, fields, plan.grid, plan.kernel, plan.scale, plan.strategy, plan.footprint,
        )
    else
        # Distributed and MPI apply field by field.
        for k in eachindex(outs)
            filter_apply!(outs[k], fields[k], plan)
        end
        return outs
    end
end

function filter_apply_batch!(outs, fields, plan::AbstractFilterPlan)
    _batched_fields(outs, plan) && return filter_apply_batch_trailing!(outs, fields, plan)
    for k in eachindex(outs)
        filter_apply!(outs[k], fields[k], plan)
    end
    return outs
end

# Whether the fields carry a trailing batch axis: one dimension more than the plan's grid.
@inline function _batched_fields(outs, plan::PhysicalFilterPlan)
    R = length(FlowGeometries.Grids.size_tuple(plan.grid))
    return ndims(first(outs)) == R + 1
end

# A spectral plan type with no method of its own has no trailing-batch form.
@inline _batched_fields(outs, ::AbstractFilterPlan) = false

function filter_apply_batch_trailing!(outs, fields, plan::AbstractFilterPlan)
    for k in eachindex(outs)
        filter_apply_batched!(outs[k], fields[k], plan)
    end
    return outs
end

"""
    filter_fields!(outs, fields, grid, kernel, scale; mask_strategy = ZeroFill(), filter_plan = nothing,
                   execution = AutoBackend())

Filter several fields on one grid at one kernel and scale with a single plan, `filter_plan` or one
built by [`plan_filter`](@ref), through [`filter_apply_batch!`](@ref). `outs` and `fields` are
indexable collections of matching arrays, such as a tuple of velocity components.
"""
function filter_fields!(
    outs,
    fields,
    grid::FlowGeometries.Grids.StructuredGrid{T,G},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    filter_plan::Union{Nothing,AbstractFilterPlan} = nothing,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
) where {G<:FlowGeometries.Geometry.AbstractGeometry{T}} where {T<:AbstractFloat}
    plan = filter_plan === nothing ? plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, execution = execution) : filter_plan
    return filter_apply_batch!(outs, fields, plan)
end
