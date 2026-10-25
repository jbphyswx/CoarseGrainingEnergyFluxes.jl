# ---------------------------------------------------------------------------
# Sweep families: one grid plan and one scratch shared by every scale
# ---------------------------------------------------------------------------

"""
    FilterPlanFamily(grid_plan, plans, scratch)

The plans for a sweep of scales over one grid, with the scale-independent
[`AbstractGridPlan`](@ref) and the per-apply [`AbstractFilterScratch`](@ref) they share; either is
`nothing` for an engine with nothing to share. A family is an `AbstractVector` of its plans.

The plans of one family share one scratch, so they must not be applied concurrently; give each
concurrent worker its own family.
"""
struct FilterPlanFamily{
    P, GP<:Union{Nothing,AbstractGridPlan}, PV<:AbstractVector{P},
    SC<:Union{Nothing,AbstractFilterScratch},
} <: AbstractVector{P}
    grid_plan::GP
    plans::PV
    scratch::SC
end

FilterPlanFamily(grid_plan::GP, plans::PV, scratch::SC) where {
    P, GP, PV<:AbstractVector{P}, SC,
} = FilterPlanFamily{P,GP,PV,SC}(grid_plan, plans, scratch)

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
    return (gp, _prefixsum_scratch(gp, grid))
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
                      cache_byte_budget = default_cache_byte_budget(), kwargs...) -> FilterPlanFamily

One [`plan_filter`](@ref) per scale in `scales`, all sharing one grid-determined half of the engine
and one per-apply scratch. For the prefix-sum top-hat engine the shared half is the measure prefix
scans, the extended axis and the mask scan. `method = nothing` takes the grid's default method.

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
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    meth = method === nothing ? _default_method(grid) : method
    gp, sc = _all_transform_engines(grid, kernel, scales, meth, kwargs) ? (nothing, nothing) :
             _sweep_shared(grid, kernel, mask_strategy, meth; kwargs...)
    # Only the keywords the engine takes are passed; `plan_filter` raises on any other.
    shared = gp === nothing && sc === nothing ? (;) :
             sc === nothing ? (; grid_plan = gp) : (; grid_plan = gp, scratch = sc)
    left = Int(cache_byte_budget)
    plans = map(scales) do s
        budget = _takes_cache_budget(grid) ? (; cache_byte_budget = max(left, 0)) : (;)
        p = plan_filter(grid, kernel, T(s); mask_strategy = mask_strategy, method = meth,
                        shared..., budget..., kwargs...)
        left -= _cache_bytes(p)
        p
    end
    return FilterPlanFamily(gp, plans, sc)
end

# Whether every scale of a real-space sweep takes a transform engine, which reads none of the shared
# half the real-space engines build.
function _all_transform_engines(grid::FlowGeometries.Grids.AbstractGrid{G,T}, kernel, scales, method,
                                kwargs) where {G,T}
    method isa RealSpace || return false
    _host_backend(_resolve_backend(get(kwargs, :execution, ComputationalBackends.AutoBackend()), grid)) || return false
    return all(s -> _takes_padded_fft(grid, kernel, T(s)) || _takes_zonal_fft(grid, kernel, T(s)), scales)
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

@inline _fftw_available() =
    Base.get_extension(parentmodule(@__MODULE__), :CoarseGrainingEnergyFluxesFFTWExt) !== nothing

# A transform runs where it evaluates the same sum as the kernel's real-space engine in fewer operations.
# Counts are per apply, a fused multiply-add counting one: one per tap of the direct sum and of each
# separable pass; for the prefix-sum top-hat seven per entry of its numerator scan (a compensated
# addition) and four per cell and band (two scan differences, their sum, the weighted accumulate);
# `2.5 n log₂ n` per real transform of length `n` (forward or inverse); and per spectral value four for a
# complex product, two for a real weight times a complex value accumulated.
_rfft_ops(n::Integer) = 2.5 * n * log2(n)
_prefixsum_ops(ncell::Integer, nrep::Integer, bands::Integer) = Float64(ncell) * (7nrep + 1) + 4.0 * bands

# Transform length along one axis: the axis itself where it wraps, else the least 2·3·5-smooth length
# holding the data and the widest offset `w` between two of its cells.
_padded_length(N::Int, w::Int, periodic::Bool) = periodic ? N : nextprod((2, 3, 5), N + min(w, N - 1))

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

# Two directions: a band per row of the disk, along a tripled axis 1 where it is periodic.
function _real_space_ops(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2}, kernel::Kernels.TopHatKernel, scale::T,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    n = prod(FlowGeometries.Grids.size_tuple(grid))
    nrep = FlowGeometries.Grids.isperiodic(grid, 1) ? 3 : 1
    return _prefixsum_ops(n, nrep, n * _ball_offsets(grid, Kernels.kernel_radius(kernel, scale), (2,)))
end

# Three directions: a band per `(j, k)` offset in the disk across directions 2 and 3.
function _real_space_ops(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,3}, kernel::Kernels.TopHatKernel, scale::T,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    n = prod(FlowGeometries.Grids.size_tuple(grid))
    return _prefixsum_ops(n, 1, n * _ball_offsets(grid, Kernels.kernel_radius(kernel, scale), (2, 3)))
end

function _real_space_ops(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, kernel::SeparableKernel, scale::T,
) where {T<:AbstractFloat, N, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    w = _clamp_bounded(grid, _separable_halfwidths(grid, kernel, scale))
    return Float64(prod(FlowGeometries.Grids.size_tuple(grid))) * sum(d -> 2w[d] + 1, 1:N)
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
    Px = FlowGeometries.Grids.isperiodic(grid, 1) ? Nx : nextprod((2, 3, 5), Nx + reach)
    realspace = kernel isa Kernels.TopHatKernel ? _prefixsum_ops(Nx * Ny, Nx > 1 ? 3 : 1, Nx * bands) :
                Float64(taps) * Nx
    return 2 * Ny * _rfft_ops(Px) + 2 * (Px ÷ 2 + 1) * pairs, realspace
end

# The padded transform evaluates the direct footprint sum of any kernel on a uniform Cartesian grid.
_takes_padded_fft(grid, kernel, scale) = false
function _takes_padded_fft(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N,S,TP,<:NTuple{N,AbstractRange}},
    kernel::Kernels.AbstractFilterKernel, scale,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}, N, S, TP}
    _fftw_available() || return false
    transform, realspace = _padded_fft_ops(grid, kernel, T(scale))
    return transform < realspace
end

# The transform along longitude evaluates the sum on a sphere whose longitude is a uniform `Range` (a
# ring or a regional box), latitude any axis, when the weights depend on the longitude offset alone.
_takes_zonal_fft(grid, kernel, scale) = false
function _takes_zonal_fft(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2,S,TP,<:Tuple{AbstractRange,<:AbstractVector}},
    kernel::Kernels.AbstractFilterKernel, scale,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.SphericalGeometry{T}, S, TP}
    (_zonal_weights(grid, kernel, T(scale)) && _fftw_available()) || return false
    transform, realspace = _zonal_fft_ops(grid, kernel, T(scale))
    return transform < realspace
end

# The Cartesian grids filtered over their cells as points: node sets and curvilinear meshes.
const _CartesianPoints{T,G} = Union{FlowGeometries.Grids.UnstructuredGrid{T,G}, FlowGeometries.Grids.CurvilinearGrid{T,G}}

# A top-hat over Cartesian points takes the binned engine on the host: its plan is one sweep over `O(n)`
# memory, its apply reads `O((r/h)^{D−1})` bin runs and partial bins per target, where the neighbour list
# holds `O((r/h)^D)` entries and is built in two passes over them.
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
function _nufft_sum_ops(grid, kernel, scale::T, K::T) where {T<:AbstractFloat}
    D = FlowGeometries.Grids.ncoordinates(grid)
    M = length(FlowGeometries.Grids.mask(grid))
    R = T(Kernels.kernel_radius(kernel, scale, Val(D)))
    ball = D == 1 ? 2R : D == 2 ? T(π) * R^2 : T(4π) / 3 * R^3
    entries = Float64(M) * min(M, ball * M / sum(FlowGeometries.Grids.measure(grid)))
    _, h, nb, _, _ = _binned_lattice(grid, Val(D))
    lines, _, wout = _binned_lines(h, R, R)
    candidates = Float64(M) * M / prod(nb) * sum(q -> 2 * wout[q] + 1, eachindex(lines))
    build = 2 * (D + 1) * candidates + 2 * entries
    n = ntuple(D) do d
        x = FlowGeometries.Grids.coordinates(grid, d)
        L = FlowGeometries.Grids.isperiodic(grid, d) ? T(FlowGeometries.Grids.period(grid, d)) :
            maximum(x) - minimum(x) + R
        2 * floor(Int, K * L / T(2π)) + 1
    end
    w = ceil(Int, -log10(Kernels.GAUSSIAN_TRUNCATION_TOL)) + 1
    transform = 2 * (Float64(M) * w^D + _rfft_ops(prod(2 .* n))) + prod(n)
    return transform, entries, build
end

# The engine of a grid filtered over its cells as points: on the host, the binned top-hat or the
# nonuniform-FFT sum where they apply, and the grid's own `build_footprint` otherwise.
function _point_set_footprint(grid, kernel, scale, mask_strategy, resolved; kwargs...)
    if _host_backend(resolved)
        _takes_binned_tophat(grid, kernel) && return binned_tophat_plan(grid, kernel, scale; mask_strategy)
        _takes_nufft_sum(grid, kernel, scale) &&
            return nufft_sum_footprint(grid, kernel, scale; mask_strategy, execution = resolved)
    end
    return build_footprint(grid, kernel, scale; kwargs...)
end

# A radial kernel's weights depend on the offsets alone; a `HighOrderKernel`'s do where one correction
# serves each latitude row.
_zonal_weights(grid, kernel::Kernels.AbstractFilterKernel, scale) = Kernels.is_radial(kernel)
_zonal_weights(grid, kernel::Kernels.HighOrderKernel, scale) = _highorder_shares_rows(grid, kernel, scale)

@inline _threading_available() =
    Base.get_extension(parentmodule(@__MODULE__), :CoarseGrainingEnergyFluxesOhMyThreadsExt) !== nothing

# The backends whose applies run in this process's own loops, where the host engines
# (`_driven_footprint`) run.
_host_backend(b::ComputationalBackends.AbstractExecutionBackend) =
    b isa Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend}

# Resolves `AutoBackend` against this package's extensions and `grid`.
# `ComputationalBackends.resolve_backend(::AutoBackend)` is ComputationalBackends' own signature, shared
# by all its consumers.
@inline function _resolve_backend(
    backend::ComputationalBackends.AbstractExecutionBackend, grid::FlowGeometries.Grids.AbstractGrid,
)
    backend isa ComputationalBackends.AbstractAutoBackend ||
        return ComputationalBackends.resolve_backend(backend)
    threaded = ComputationalBackends.ThreadedBackend()
    return (Threads.nthreads() > 1 && _threading_available() && _backend_supported(grid, threaded)) ?
        threaded : ComputationalBackends.SerialBackend()
end

# Raises for an explicit non-serial `backend` that `grid` has no parallel path for; `AutoBackend()`
# resolves to serial there instead.
function _check_backend_compatible(grid::FlowGeometries.Grids.AbstractGrid, backend::ComputationalBackends.AbstractExecutionBackend)
    if !(backend isa ComputationalBackends.AutoBackend) && !(backend isa ComputationalBackends.SerialBackend) && !_backend_supported(grid, _resolve_backend(backend, grid))
        throw(ArgumentError(
            "execution = $(typeof(backend)) was requested explicitly, but $(typeof(grid)) has no " *
            "matching parallel hook for it — there is no way to honor this request. Pass " *
            "`execution = SerialBackend()` explicitly if serial execution is acceptable, or " *
            "`execution = AutoBackend()` to let the library choose.",
        ))
    end
    return nothing
end

"""
    build_footprint(grid, kernel, scale; kwargs...) -> NodeFilterPlan

Real-space footprint for a flat-cell grid (see [`_is_flat_cell`](@ref)): every cell's neighbours within
`kernel_radius` and their weights. A Cartesian node set's are read from its bins
([`_binned_csr`](@ref)), each image of a cell across a periodic direction at its own distance; any other
grid's come from `Connectivity.fold_within`, the grid's own metric ball query on the cell list
`Connectivity.default_sweep_topology` returns, so the neighbourhood follows the geometry's distance and
periodic wrap. Raises for a grid whose cells are an index tuple and that no rectilinear or curvilinear
engine matches.
"""
build_footprint(
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}} =
    _build_footprint_flat(_cell_address(grid), grid, kernel, scale; kwargs...)

# Reached only by a `CartesianCells` grid that no rectilinear or curvilinear `build_footprint` matches.
@noinline _build_footprint_flat(
    ::FlowGeometries.Grids.CartesianCells, grid, kernel, scale; kwargs...,
) = throw(ArgumentError(
    "no real-space engine is implemented for $(nameof(typeof(grid))). Its cells are named by an " *
    "index tuple, so the flat-cell node engine does not apply, and no rectilinear or curvilinear " *
    "engine matches it either. Use `method = Spectral()` if a transform targets this grid.",
))

function _build_footprint_flat(
    ::FlowGeometries.Grids.FlatCells,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    return NodeFilterPlan(_node_list(grid, kernel, scale, _build_backend(execution))...)
end

# Whether a flat-cell grid's neighbour list is read from its bins, every image of a cell across a periodic
# direction at its own distance: a Cartesian node set. Any other is built over the grid's own ball query.
_binned_list(::FlowGeometries.Grids.UnstructuredGrid{T,G}) where {T, G<:FlowGeometries.Geometry.AbstractCartesianGeometry{T}} = true
_binned_list(_) = false

function _node_list(grid, kernel, scale, backend)
    _binned_list(grid) && return _binned_csr(grid, kernel, scale, backend)
    built, perm = _build_order(grid)
    return _renumbered(_node_csr(built, kernel, scale, backend), perm)
end

# The backend a node plan's build runs on: the host's threads where the plan's apply is threaded, each
# thread taking rows as it finishes since their neighbour counts differ; the calling thread otherwise.
_build_backend(::ComputationalBackends.AbstractThreadedBackend) = FlowGeometries.Execution.BalancedThreads()
_build_backend(::ComputationalBackends.AbstractExecutionBackend) = ComputationalBackends.SerialBackend()

# The order a node plan is built in. A point cloud's ids need not follow its points through space, and a
# ball query then reads its candidates from all over memory, so the plan is built over the nodes in a
# space-filling order (`spatial_order`) and renumbered back. A pixelization's ids already do.
_build_order(grid::FlowGeometries.Grids.UnstructuredGrid) =
    (perm = FlowGeometries.Grids.spatial_order(grid); (FlowGeometries.Grids.reorder(grid, perm), perm))
_build_order(grid) = (grid, nothing)

# A list built over the nodes in the order `perm` lists them, with its rows and neighbours named by the
# grid's own ids: built node `s` is node `perm[s]`.
_renumbered(csr::Tuple, ::Nothing) = csr
function _renumbered((bnbrs, bw, bptr)::Tuple, perm::AbstractVector{<:Integer})
    n = length(perm)
    ptr = Vector{Int}(undef, n + 1)
    ptr[1] = 1
    len = Vector{Int}(undef, n)
    @inbounds for s in 1:n
        len[perm[s]] = bptr[s + 1] - bptr[s]
    end
    @inbounds for t in 1:n
        ptr[t + 1] = ptr[t] + len[t]
    end
    nbrs, w = similar(bnbrs), similar(bw)
    @inbounds for s in 1:n
        o = ptr[perm[s]] - bptr[s]
        for k in bptr[s]:(bptr[s + 1] - 1)
            nbrs[k + o] = perm[bnbrs[k]]
            w[k + o] = bw[k]
        end
    end
    return nbrs, w, ptr
end

# Every node's neighbours within the kernel's radius and their weights, as CSR in the grid's node order:
# one pass counts each row, the next fills it, so every row is written where the counts put it and the
# rows run on `backend` independently.
function _node_csr(
    grid::FlowGeometries.Grids.AbstractGrid{G,T}, kernel::Kernels.AbstractFilterKernel, scale::T, backend,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    dim = _kernel_dim(grid)
    rad = Kernels.kernel_radius(kernel, scale, dim)
    n = length(FlowGeometries.Grids.mask(grid))
    mt = _query_topology(grid, rad)
    # `active_only = false` keeps the plan mask-independent, the mask entering at apply time;
    # `self = true` includes the centre cell.
    query(f::F, acc, t, sc) where {F} = FlowGeometries.Connectivity.fold_within(
        f, acc, grid, t; ball = rad, self = true, active_only = false, topology = mt, scratch = sc,
    )
    ptr = Vector{Int}(undef, n + 1)
    ptr[1] = 1
    FlowGeometries.Execution.run_chunks(n, backend) do rng
        sc = FlowGeometries.Connectivity.ball_scratch(mt)
        @inbounds for t in rng
            ptr[t + 1] = query((m, _, _) -> m + 1, 0, t, sc)
        end
    end
    @inbounds for t in 1:n
        ptr[t + 1] += ptr[t]
    end
    nbrs = Vector{Int}(undef, ptr[n + 1] - 1)
    w = Vector{T}(undef, ptr[n + 1] - 1)
    FlowGeometries.Execution.run_chunks(n, backend) do rng
        sc = FlowGeometries.Connectivity.ball_scratch(mt)
        @inbounds for t in rng
            query(ptr[t], t, sc) do k, j, d
                nbrs[k] = j
                w[k] = Kernels.kernel_weight(kernel, T(d), scale, dim) * FlowGeometries.Grids.measure(grid, j)
                return k + 1
            end
        end
    end
    return nbrs, w, ptr
end

# A `HighOrderKernel`'s weights are corrected about each node ([`_highorder_targets`](@ref)), and a node
# plan stores them all.
function _build_footprint_flat(
    ::FlowGeometries.Grids.FlatCells,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.HighOrderKernel,
    scale::T;
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    rad = _query_radius(grid, kernel, scale)
    mt = _query_topology(grid, rad)
    scratch = FlowGeometries.Connectivity.ball_scratch()
    fold = (f, acc, t) -> FlowGeometries.Connectivity.fold_within(
        f, acc, grid, t; ball = rad, self = true, active_only = false, topology = mt, scratch = scratch,
    )
    _, _, kept = _highorder_targets(grid, kernel, scale, rad, fold, nothing, Returns(true))
    return NodeFilterPlan(kept...)
end

"""
    apply_footprint!(out, field, grid, fp::NodeFilterPlan, strategy) -> out

Weighted mean over each node's stored neighbourhood, with the same two mask conventions the structured
engines use: `ZeroFill` keeps a masked neighbour in the denominator, contributes nothing for it and
filters a masked node too; `Deformable` drops it from both sums and leaves a masked node at zero.
"""
function apply_footprint!(
    out::AbstractVector{T}, field::AbstractVector, grid::FlowGeometries.Grids.AbstractGrid,
    fp::NodeFilterPlan{T}, strategy::AbstractMaskStrategy,
) where {T<:AbstractFloat}
    @inbounds for t in eachindex(out)
        out[t] = _footprint_node_point(field, grid, fp, strategy, t)
    end
    return out
end

# Node `t`'s filtered value, shared by the serial and parallel drivers. A flat-cell grid has no lattice
# past its cells, so no exterior mass enters the normalization.
@inline function _footprint_node_point(
    field::AbstractVector, grid::FlowGeometries.Grids.AbstractGrid,
    fp::NodeFilterPlan{T}, strategy::AbstractMaskStrategy, t::Integer,
) where {T<:AbstractFloat}
    _filters_target(strategy, FlowGeometries.Grids.isactive(grid, t)) || return zero(T)
    ws = zero(T)
    wn = zero(T)
    @inbounds for k in fp.ptr[t]:(fp.ptr[t+1] - 1)
        j = fp.nbrs[k]
        wj = fp.w[k]
        active = FlowGeometries.Grids.isactive(grid, j)
        if strategy isa ZeroFill
            wn += wj
            active && (ws += wj * field[j])
        else
            active || continue
            wn += wj
            ws += wj * field[j]
        end
    end
    return _normalized(ws, wn)
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
    kwargs...,
) where {T<:AbstractFloat}
    _validate_scale(scale)
    method isa Spectral && return spectral_filter_plan(
        transform, grid, kernel, scale; mask_strategy = mask_strategy, execution = execution, kwargs...,
    )
    isempty(kwargs) || _unused_keywords(kwargs, method)
    _check_backend_compatible(grid, execution)
    resolved = _resolve_backend(execution, grid)
    fp = _point_set_footprint(grid, kernel, scale, mask_strategy, resolved; execution = resolved)
    return PhysicalFilterPlan(prepare_workspace(resolved, grid, fp), grid, mask_strategy, kernel, scale, resolved)
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
    fp = _point_set_footprint(grid, kernel, scale, mask_strategy, resolved;
                              cache_strategy = cache_strategy, cache_byte_budget = cache_byte_budget)
    return PhysicalFilterPlan(prepare_workspace(resolved, grid, fp), grid, mask_strategy, kernel, scale, resolved)
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
