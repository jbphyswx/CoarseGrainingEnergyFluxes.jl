module Filtering

using FlowGeometries: FlowGeometries
using ..Kernels: Kernels
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends
using StaticArrays: StaticArrays as SA

export AbstractMaskStrategy, ZeroFill, Deformable
export AbstractFilterMethod, RealSpace, Spectral, AutoMethod
export AbstractCacheStrategy, AutoCache, AlwaysCache, NeverCache
export filter_field!, filter_fields!, filter_slices!
export AbstractFilterPlan, plan_filter, filter_apply!, filter_apply_batch!

# Kernels that factor per axis, `G(x₁,…,x_N) = ∏ G(x_d)`, and so take the two-pass `O(N·Σwᵈ)` engine
# instead of the `O(N·∏wᵈ)` footprint one. A `Union` rather than an abstract supertype because
# `GaussianKernel` is separable *incidentally* (its radial form factors) while `HighOrderKernel` is
# separable *by definition* and has no radial form at all — they share no useful supertype, only this
# property. `Kernels.is_separable` is the trait; this is the dispatch handle.
const SeparableKernel = Union{Kernels.GaussianKernel, Kernels.HighOrderKernel}

# ---------------------------------------------------------------------------
# Masking strategy (singleton types — specializable, unlike Symbol dispatch)
# ---------------------------------------------------------------------------

"""
    AbstractMaskStrategy

How masked (inactive) cells enter the filter normalization.
"""
abstract type AbstractMaskStrategy end

"""
    ZeroFill <: AbstractMaskStrategy

Excluded cells are treated as zero-valued: they contribute to the denominator (kernel weight) but
zero to the numerator. The kernel is homogeneous (same shape everywhere), which preserves domain
averages and commutation with derivatives (the Storer 2022 / Aluie 2019 "fixed kernel" mode).
"""
struct ZeroFill <: AbstractMaskStrategy end

"""
    Deformable <: AbstractMaskStrategy

Masked cells are excluded from BOTH numerator and denominator, so the kernel is renormalized over the
locally-included area only ("deformable kernel"). Excluded cells are genuinely dropped, but the kernel
becomes inhomogeneous near a mask boundary (breaks the strict commutation theorems).
"""
struct Deformable <: AbstractMaskStrategy end

# ---------------------------------------------------------------------------
# Filtering method: physical direct-sum (default) vs spectral (FFT/SHT/NUFFT via extensions)
# ---------------------------------------------------------------------------

"""
    AbstractFilterMethod

How the convolution is evaluated: [`RealSpace`](@ref) (physical-space footprint, any grid/mask) or
[`Spectral`](@ref) (transform-space multiply — FFT for uniform periodic Cartesian, spherical
harmonics for the uniform sphere, NUFFT/NUFSHT for scattered points; provided by extensions).
"""
abstract type AbstractFilterMethod end

"Physical-space direct-sum convolution (works on any grid, mask, and geometry)."
struct RealSpace <: AbstractFilterMethod end

"""
    Spectral <: AbstractFilterMethod

Transform-space filtering (kernel applied as a multiply on the transformed field). Requires a
spectral extension and a compatible grid (e.g. `using FFTW` for a uniform, periodic Cartesian grid).
"""
struct Spectral <: AbstractFilterMethod end

"""
    AutoMethod <: AbstractFilterMethod

Pick the engine from real capability, the same contract [`AutoCache`](@ref) and `AutoBackend` follow:
[`Spectral`](@ref) only where a transform is available AND exact for this grid, otherwise
[`RealSpace`](@ref).

Selects [`Spectral`](@ref) only where every axis is periodic and uniform and a transform backend is
loaded — the case where a periodic transform is the exact filter. Otherwise [`RealSpace`](@ref).
"""
struct AutoMethod <: AbstractFilterMethod end

"""
    AbstractFilterPlan

A prebuilt filter (grid + kernel + scale + mask strategy + backend) that can be applied to many
fields without redoing setup. Physical-space backends precompute a `FilterFootprint`; the spectral
extensions (FFTW/FINUFFT/SHT) hold cached transform plans. Declared here rather than alongside
`PhysicalFilterPlan` further down so that `filter_field!`'s `filter_plan::Union{Nothing,
AbstractFilterPlan}` keyword annotation, just below, can name it.
"""
abstract type AbstractFilterPlan end

# A plan owns FFTW/FINUFFT/SHT plan objects, whose own `show` walks the library's internal plan tree —
# 6 KB of output for a 16×16 transform, and a call into the C library from wherever the plan happens to
# be printed, including a worker that does not own it. The type name is what a plan usefully prints.
Base.show(io::IO, p::AbstractFilterPlan) = print(io, nameof(typeof(p)), "(…)")
Base.show(io::IO, ::MIME"text/plain", p::AbstractFilterPlan) = show(io, p)


# ---------------------------------------------------------------------------
# Cache strategy: how much of the nonuniform-axis/curvilinear/ND real-space footprint to precompute
# and store, vs. recompute on the fly at apply time (singleton types — specializable, same idiom as
# AbstractMaskStrategy).
# ---------------------------------------------------------------------------

"""
    AbstractCacheStrategy

Whether a real-space footprint over a genuinely nonuniform axis (`StructuredGrid` with a `Vector`
axis, `CurvilinearGrid`, or ND with a non-`Range` axis) stores its full per-point neighbour list, or
recomputes it on the fly at apply time. The per-point neighbour/weight computation itself is always
the same either way (there is no shared translation-invariant table to exploit on a nonuniform axis,
unlike the `FilterFootprint` fast path) — this only controls whether that computation's RESULT is
kept in memory for reuse across separate future `filter_apply!` calls, or re-derived each time.
"""
abstract type AbstractCacheStrategy end

"""
    AutoCache <: AbstractCacheStrategy

Build and store the full per-point neighbour-list cache only if its estimated size is under
`cache_byte_budget` (default [`DEFAULT_CACHE_BYTE_BUDGET`](@ref)); otherwise fall back to recomputing
neighbours on the fly at apply time. This is the only cache-strategy knob most callers ever need —
it caches whenever doing so is affordable, which is strictly better than not caching whenever a
plan will be reused across more than one `filter_apply!` call.
"""
struct AutoCache <: AbstractCacheStrategy end

"""
    AlwaysCache <: AbstractCacheStrategy

Force building the full per-point neighbour-list cache regardless of `cache_byte_budget` — for a
caller who knows more memory is available than the conservative default budget assumes.
"""
struct AlwaysCache <: AbstractCacheStrategy end

"""
    NeverCache <: AbstractCacheStrategy

Force recomputing neighbours on the fly at every `filter_apply!`/`filter_apply_batch!` call, never
storing the cache — for a genuine, known memory ceiling `AutoCache`'s budget check doesn't already
account for (e.g. a GPU's device memory budget, or deliberately running many large plans
concurrently). Not a speed/memory preference: for any workflow that reuses a plan across more than
one call, caching is strictly better whenever it fits in memory, so this should be reached for only
when a specific external memory constraint is known, not by default.
"""
struct NeverCache <: AbstractCacheStrategy end

"""
    DEFAULT_CACHE_BYTE_BUDGET

Default byte budget for [`AutoCache`](@ref)'s size check on a real-space footprint cache (256 MiB).
"""
const DEFAULT_CACHE_BYTE_BUDGET = 256 * 1024^2

# ---------------------------------------------------------------------------
# Extension hook points
# ---------------------------------------------------------------------------
# Fallbacks that error until the relevant backend extension is loaded. Each execution-backend
# extension overrides its hook; the public `filter_field!` dispatches here based on the resolved
# backend. (Backend TYPES live in `Backends`; these are the per-backend filtering implementations.)

function threaded_filter_field!(args...; kwargs...)
    throw(ArgumentError("ThreadedBackend is unavailable — run `using OhMyThreads` (or use SerialBackend())."))
end

# Batched form: several fields sharing one grid/kernel/scale. Separate from the single-field hook
# because the scattered footprints derive each point's neighbour list on the fly, and that derivation
# is shared across the batch — a per-field loop would repeat it once per field.
function threaded_filter_fields!(args...; kwargs...)
    throw(ArgumentError("ThreadedBackend is unavailable — run `using OhMyThreads` (or use SerialBackend())."))
end

# Slice-parallel form: many INDEPENDENT problems, one plan each. A different axis from
# `threaded_filter_fields!`, which shares one grid across several fields.
function threaded_filter_slices!(args...; kwargs...)
    throw(ArgumentError("ThreadedBackend is unavailable — run `using OhMyThreads` (or use SerialBackend())."))
end

# Padded-FFT real-space engine (FFTW extension). Zero-padding makes the transform compute the LINEAR
# convolution, so unlike a periodic transform it holds on bounded and masked domains.
function padded_fft_footprint(args...; kwargs...)
    throw(ArgumentError("The padded-FFT real-space engine needs FFTW — run `using FFTW`."))
end

function distributed_filter_field!(args...; kwargs...)
    throw(ArgumentError("DistributedBackend is unavailable — run `using Distributed` (or use SerialBackend())."))
end

function gpu_filter_field!(args...; kwargs...)
    throw(ArgumentError("GPUBackend is unavailable — run `using KernelAbstractions` + a GPU backend (or use SerialBackend())."))
end

function mpi_filter_field!(args...; kwargs...)
    throw(ArgumentError("MPIBackend is unavailable — run `using MPI` (or use SerialBackend())."))
end

# Build a spectral filter plan. Every spectral backend is a thin transform adapter that overrides
# this for its grid type (forward transform → multiply by `spectral_transfer` → inverse transform):
#   FFTW    StructuredGrid{Cartesian}     (uniform periodic Cartesian)
#   FINUFFT UnstructuredGrid{Cartesian}   (scattered / non-uniform Cartesian)
#   SHT     StructuredGrid{Spherical}     (uniform spherical, Gauss–Legendre × equiangular)
#   NUFSHT  UnstructuredGrid{Spherical}   (scattered spherical)
# Errors until a compatible extension is loaded.
function spectral_filter_plan(spectral_backend, grid, kernel, scale; kwargs...)
    throw(ArgumentError(
        "Spectral filtering with $(typeof(spectral_backend)) is unavailable for $(typeof(grid)) — " *
        "load a spectral backend (`using FFTW` uniform Cartesian, `using FINUFFT` scattered " *
        "Cartesian, `using FastSphericalHarmonics` uniform spherical, `using NUFSHT` scattered " *
        "spherical).",
    ))
end

# ---------------------------------------------------------------------------
# Public Filtering API
# ---------------------------------------------------------------------------

"""
    filter_field!(out, field, grid, kernel, scale; mask_strategy=ZeroFill(), filter_plan=nothing, backend=AutoBackend())

Filter a field on a grid using `kernel` at characteristic full width `scale` (ℓ), writing the
result to `out` (returned).

# Keyword Arguments
- `mask_strategy::AbstractMaskStrategy=ZeroFill()`: masking strategy — `ZeroFill()` (excluded cells count
  in the denominator as zero; the kernel stays homogeneous) or `Deformable()` (excluded cells dropped
  from numerator and denominator; renormalized over the locally-included area).

  Near a boundary the footprint is truncated, and both strategies inherit the same shape distortion
  from that: measured on a straight coast, Gaussian at `ℓ = 16` cells, one cell inshore the footprint's
  centroid sits 0.21ℓ offshore and its width is 62% of the interior value (75% at `ℓ/4` inshore, 90% at
  `ℓ/2`, 100% at `ℓ`). Points within `≈ℓ` of a boundary are contaminated either way.

  They differ on the footprint's **mass**:

  - `ZeroFill` leaves it at the truncated value, so the kernel is position-independent and filtering
    **commutes with spatial derivatives** — the step the flux budget is derived by. A uniform field
    ≡ 1 then filters to 0.543 one cell from the coast, 0.948 at `ℓ/2`, 0.9996 at `ℓ`.
  - `Deformable` divides it out, so a constant filters to 1.000 everywhere, at the cost of a
    position-dependent kernel that no longer commutes with derivatives.

  `ZeroFill` is the default because the flux budget is a statement about commuting operators; prefer
  `Deformable` when a locally unbiased amplitude near a coast matters more than a budget that closes.

  Neither conserves the ACTIVE-cell integral on a masked domain, and they fail differently. `ZeroFill`
  conserves it over the whole domain exactly (7e-18 relative, unmasked periodic) but smears part of it
  onto masked cells, which report zero; `Deformable` renormalizes that away and tracks the active-cell
  integral better — 4.8e-4 relative drift against `ZeroFill`'s 1.1e-2 on a masked periodic grid. On a
  bounded grid the domain edge costs both about 1e-2 at `ℓ = 6Δx`. A closed energy budget wants a
  periodic unmasked domain; otherwise expect an `O(ℓ/L)` boundary residual.
- `filter_plan::Union{Nothing,AbstractFilterPlan}=nothing`: a prebuilt [`plan_filter`](@ref) result to
  reuse instead of building one from scratch — the zero-(re)allocation entry point for a repeated
  sweep (many timesteps/fields over the same grid/kernel/scale). When supplied, `mask_strategy`/
  `backend`/`method` are ignored (already baked into the plan); build it once with `plan_filter` and
  pass it here on every subsequent call.
- `backend::AbstractExecutionBackend=AutoBackend()`: execution backend (SerialBackend,
  ThreadedBackend, GPUBackend, …). Ignored when `filter_plan` is supplied.

For spherical grids the longitude footprint wraps only when the grid is periodic (`isperiodic`);
distances use the great-circle (Haversine) metric.

# Examples
```julia
geom = CartesianGeometry()
grid = StructuredGrid(geom, 0.0:1000.0:99_000.0, 0.0:1000.0:99_000.0, mask)
out = zeros(100, 100)
filter_field!(out, field, grid, TopHatKernel(), 5000.0; mask_strategy = Deformable())
```
"""
function filter_field!(
    out::AbstractArray{T},
    field::AbstractArray,
    grid::FlowGeometries.Grids.AbstractGrid,
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    filter_plan::Union{Nothing,AbstractFilterPlan} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    method::AbstractFilterMethod = RealSpace(),
) where {T<:AbstractFloat}
    plan = filter_plan === nothing ?
        plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend, method = method) : filter_plan
    return filter_apply!(out, field, plan)
end

# 3D volume filtering: horizontal filtering layer by layer. The plan is built once and reused across
# every layer on every backend, which is why this dispatches here rather than recursing into the 2D
# method per layer with no plan to pass.
function filter_field!(
    out::AbstractArray{T,3},
    field::AbstractArray{<:Any,3},
    grid::FlowGeometries.Grids.StructuredGrid{<:FlowGeometries.Geometry.AbstractGeometry,T,2},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    filter_plan::Union{Nothing,AbstractFilterPlan} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    method::AbstractFilterMethod = RealSpace(),
) where {T<:AbstractFloat}
    plan = filter_plan === nothing ?
        plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend, method = method) : filter_plan
    for k in axes(field, 3)
        filter_apply!(view(out, :, :, k), view(field, :, :, k), plan)
    end
    return out
end

# ---------------------------------------------------------------------------
# Serial physical-space convolution: precomputed footprint + single apply loop
# ---------------------------------------------------------------------------

"""
    FilterFootprint{T}

Precomputed convolution footprint for a structured grid + kernel + scale. The in-support neighbour
offsets `(di, dj)` and their geometric weights `w = kernel_weight(distance) * cell_area` are stored
in a flat CSR-like layout, grouped into axis-2 (`y`) bands (`ptr[b]:ptr[b+1]-1`). For Cartesian grids
the footprint is translation-invariant → a single band; for a spherical grid it is invariant in `x`
(longitude) → one band per `y` (latitude) value. The weights are mask-independent (geometry only);
masking is applied when the footprint is convolved with a field.
"""
struct FilterFootprint{T<:AbstractFloat, VI<:AbstractVector{Int}, VT<:AbstractVector{T}}
    di::VI    # axis-1 (x) index offset
    dj::VI    # axis-2 (y) index offset
    w::VT       # kernel_weight(distance) * cell area
    ptr::VI   # band b's entries: ptr[b]:ptr[b+1]-1
    nbands::Int        # 1 (Cartesian) or Ny (spherical)
end

@inline _band(::FilterFootprint, ::FlowGeometries.Grids.StructuredGrid{<:FlowGeometries.Geometry.CartesianGeometry}, j::Integer) = 1
@inline _band(::FilterFootprint, ::FlowGeometries.Grids.StructuredGrid{<:FlowGeometries.Geometry.SphericalGeometry}, j::Integer) = j

"""
    ScatteredCache{T}

The full per-target-point neighbour list for a [`ScatteredFilterPlan`](@ref) (absolute neighbour
indices + weights), built only when the plan's [`AbstractCacheStrategy`](@ref) calls for it. Same
CSR-style layout as before: target `t = i + (j-1)*Nx`, entries `ptr[t]:ptr[t+1]-1`.
"""
struct ScatteredCache{T<:AbstractFloat, VI<:AbstractVector{Int}, VT<:AbstractVector{T}}
    ii::VI    # absolute neighbour axis-1 (x) index (periodic wrap already resolved)
    jj::VI    # absolute neighbour axis-2 (y) index
    w::VT       # kernel_weight(distance) * cell area
    ptr::VI   # target t = i + (j-1)*Nx; entries ptr[t]:ptr[t+1]-1
end

"""
    ScatteredFilterPlan{T,K}

Real-space footprint for a genuinely nonuniform 2D `StructuredGrid` axis, or a `CurvilinearGrid`
(exactly the `periodic_x = periodic_y = false` case of the same candidate-window/distance-gate
computation). `FilterFootprint` is a translation-invariant cache — the SAME index offset (and its
weight) is reused for every target point — which is only valid on a uniform axis; here that
assumption is false (offset `+3` means a different physical displacement depending on where you
start), so there is no way to share one offset/weight set across points. That does NOT mean the
result must be stored, though: since the search window (`di_lim`/`dj_lim`) is already a global scalar
bound (not per-point), the exact same candidate enumeration + `distance`/`kernel_weight` gate that
determines a point's neighbours can be re-run identically at apply time from these few scalars alone
— `cache` holds the materialized [`ScatteredCache`](@ref) only when the plan's cache strategy decided
to build it (see [`AbstractCacheStrategy`](@ref)), `nothing` otherwise (apply-time recomputation).
"""
struct ScatteredFilterPlan{T<:AbstractFloat, K<:Kernels.AbstractFilterKernel, C<:Union{Nothing,ScatteredCache{T}}, MT}
    kernel::K
    scale::T
    rad::T
    di_lim::Int
    dj_lim::Int
    periodic_x::Bool
    periodic_y::Bool
    x_period::T
    y_period::T
    is_cartesian::Bool
    cache::C
    topology::MT   # built once; both the cache build and the streaming apply query through it
end

# Whether a ball query on this grid should carry a spatial index is upstream's call, not this
# package's: a separable window already bounds a `StructuredGrid`, and a curvilinear mesh has none.
@inline _query_topology(grid::FlowGeometries.Grids.AbstractGrid, ball) =
    FlowGeometries.Connectivity.default_sweep_topology(grid, ball)

# Estimated cache byte size for the AutoCache budget check — the same conservative bounding-box
# entry count the old unconditional `sizehint!` always paid for, now only materialized if it fits.
@inline _scattered_cache_bytes(Nx::Integer, Ny::Integer, di_lim::Integer, dj_lim::Integer, ::Type{T}) where {T} =
    Nx * Ny * (2*di_lim + 1) * (2*dj_lim + 1) * (2*sizeof(Int) + sizeof(T))

@inline function _should_cache(cache_strategy::AbstractCacheStrategy, est_bytes::Integer, cache_byte_budget::Integer)
    cache_strategy isa AlwaysCache && return true
    cache_strategy isa NeverCache && return false
    return est_bytes <= cache_byte_budget   # AutoCache
end

"""
    _scattered_window_bounds(grid::StructuredGrid{...,2}, rad) -> (di_lim, dj_lim, periodic_x, periodic_y, x_period, y_period, is_cartesian)

The widest window the grid's own ball query will scan, taken from `Connectivity.metric_window` rather
than re-derived here. On a rectilinear grid the window depends on the row, not the column, so one
evaluation per row covers the grid.
"""
function _scattered_window_bounds(
    grid::FlowGeometries.Grids.StructuredGrid{G,T,2}, rad::T,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    periodic_x = FlowGeometries.Grids.isperiodic(grid, 1)
    # Axis 2 is a Cartesian y (periodicity meaningful — a doubly-periodic box is standard) or a
    # spherical latitude (periodicity meaningless — wrapping past a pole is not an index wrap, so no
    # spherical grid sets it). Read it generically rather than assuming per geometry.
    periodic_y = FlowGeometries.Grids.isperiodic(grid, 2)

    _, Ny = FlowGeometries.Grids.size_tuple(grid)
    mt = FlowGeometries.Connectivity.MetricTopology(grid)
    di_lim = 0
    dj_lim = 0
    for j in 1:Ny
        w = FlowGeometries.Connectivity.metric_window(grid, (1, j), rad, mt)
        di_lim = max(di_lim, w[1])
        dj_lim = max(dj_lim, w[2])
    end

    # A wrapped candidate's raw stored coordinate sits a full period away from the target on a
    # periodic CARTESIAN axis (e.g. index Nx is `Lx` meters from index 1, not adjacent to it), so
    # the plain Euclidean `distance` below would reject every genuinely-close wrapped neighbor unless
    # shifted back by one period first. A periodic SPHERICAL axis needs no such shift: great-circle
    # distance is built from `cos`/`sin` of the raw longitude, which is already exactly 2π-periodic
    # regardless of the literal angle value. `x_period`/`y_period` mirror the same "extent + one
    # cell spacing" convention `StructuredGrid`'s own constructor uses to derive its periodic cell width.
    is_cartesian = G <: FlowGeometries.Geometry.CartesianGeometry{T}
    # The grid's stored wrap length is authoritative: `n·|Δ|` for a uniform axis, caller-supplied for a
    # stretched one, whose seam gap its samples do not determine.
    x_period = (periodic_x && is_cartesian) ? T(FlowGeometries.Grids.period(grid, 1)) : zero(T)
    y_period = (periodic_y && is_cartesian) ? T(FlowGeometries.Grids.period(grid, 2)) : zero(T)
    return di_lim, dj_lim, periodic_x, periodic_y, x_period, y_period, is_cartesian
end

# Folds `acc = f(acc, iin, jjn, d)` over every cell within `rad` of `(i, j)`, the centre included.
@inline function _scattered_foldl(
    f::F, acc, grid, target, i::Integer, j::Integer, Nx::Integer, Ny::Integer,
    di_lim::Integer, dj_lim::Integer, periodic_x::Bool, periodic_y::Bool,
    x_period::T, y_period::T, is_cartesian::Bool, rad::T, mt, scratch = nothing,
) where {F, T<:AbstractFloat}
    # The window bound, the periodic convention and the distance are all properties of the grid, so the
    # traversal is the grid's. The scalar arguments above are still taken because the plan stores them
    # for its cache-size estimate.
    return _ball_fold(acc, grid, Int(i), Int(j), rad, is_cartesian, mt, scratch) do a, J, d
        f(a, J[1], J[2], d)
    end
end

# `AllImages` is the torus convention a convolution needs: past `rad = L/2` a cell contributes through
# several images, each at its own displacement. A curvilinear grid has no axis to tile along, so its
# query takes no `images` argument.
#
# `scratch` is the candidate buffer an INDEXED query fills; without one it allocates a fresh list per
# call. A separable window has none to reuse, so the structured form ignores it. One buffer per task.
@inline _ball_fold(
    f::F, acc, grid::FlowGeometries.Grids.StructuredGrid, i::Int, j::Int, rad, is_cartesian::Bool, mt,
    scratch = nothing,
) where {F} = FlowGeometries.Connectivity.fold_within(
    f, acc, grid, i, j;
    ball = rad, self = true, active_only = false, topology = mt,
    images = is_cartesian ? FlowGeometries.Connectivity.AllImages() :
                            FlowGeometries.Connectivity.NearestImage(),
)

@inline _ball_fold(f::F, acc, grid, i::Int, j::Int, rad, ::Bool, mt, scratch = nothing) where {F} =
    FlowGeometries.Connectivity.fold_within(
        f, acc, grid, i, j;
        ball = rad, self = true, active_only = false, topology = mt, scratch = scratch,
    )

"""
    _build_footprint_scattered(grid, kernel, scale; cache_strategy=AutoCache(), cache_byte_budget=DEFAULT_CACHE_BYTE_BUDGET) -> ScatteredFilterPlan

Build the compact nonuniform-axis plan (O(1) scalar metadata) and, only if `cache_strategy` calls for
it, the full per-point [`ScatteredCache`](@ref) — correct for any spacing pattern (Cartesian or
spherical, uniform or not), since it never assumes translation invariance. The search window comes
from `_scattered_window_bounds`, i.e. the grid's own `Connectivity.metric_window`; the exact
`d <= rad` check still gates inclusion, so a loose bound only costs iterations, never a missed cell.
"""
function _build_footprint_scattered(
    grid::FlowGeometries.Grids.StructuredGrid{G,T,2},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    cache_strategy::AbstractCacheStrategy = AutoCache(),
    cache_byte_budget::Integer = DEFAULT_CACHE_BYTE_BUDGET,
    kwargs...,   # accepts (and ignores) mask_strategy — only the separable-Gaussian path needs it
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    rad = Kernels.kernel_radius(kernel, scale)
    di_lim, dj_lim, periodic_x, periodic_y, x_period, y_period, is_cartesian =
        _scattered_window_bounds(grid, rad)
    mt = _query_topology(grid, rad)

    cache = if _should_cache(cache_strategy, _scattered_cache_bytes(Nx, Ny, di_lim, dj_lim, T), cache_byte_budget)
        ii = Int[]
        jj = Int[]
        w = T[]
        sizehint!(ii, Nx * Ny * (2*di_lim + 1) * (2*dj_lim + 1))
        sizehint!(jj, Nx * Ny * (2*di_lim + 1) * (2*dj_lim + 1))
        sizehint!(w, Nx * Ny * (2*di_lim + 1) * (2*dj_lim + 1))
        ptr = Vector{Int}(undef, Nx * Ny + 1)
        ptr[1] = 1
        sc = FlowGeometries.Connectivity.ball_scratch()
        for j in 1:Ny, i in 1:Nx # column-major target order: t = i + (j-1)*Nx, increasing monotonically
            t = i + (j - 1) * Nx
            target = FlowGeometries.Grids.coords(SA.SVector, grid, i, j)
            _scattered_foldl(
                nothing, grid, target, i, j, Nx, Ny, di_lim, dj_lim, periodic_x, periodic_y, x_period, y_period, is_cartesian, rad, mt, sc,
            ) do _, iin, jjn, d
                push!(ii, iin)
                push!(jj, jjn)
                push!(w, Kernels.kernel_weight(kernel, d, scale) * FlowGeometries.Grids.area(grid, iin, jjn))
                nothing
            end
            ptr[t+1] = length(ii) + 1
        end
        ScatteredCache(ii, jj, w, ptr)
    else
        nothing
    end
    return ScatteredFilterPlan(kernel, scale, rad, di_lim, dj_lim, periodic_x, periodic_y, x_period, y_period, is_cartesian, cache, mt)
end

"""
    _build_footprint_curvilinear(grid, kernel, scale; cache_strategy=AutoCache(), cache_byte_budget=DEFAULT_CACHE_BYTE_BUDGET) -> ScatteredFilterPlan

Compact plan (and, if `cache_strategy` calls for it, the full per-point cache) for a
`FlowGeometries.Grids.CurvilinearGrid`. Enumeration goes through the grid's own ball query, as
it does for a `StructuredGrid`; what differs is the SIZE ESTIMATE the cache budget is checked against.
`Connectivity.metric_window` bounds a window from per-axis spacing, and a curvilinear mesh has no
separable axes to bound with, so the estimate comes from the smallest adjacent-node spacing in each
index direction instead. It feeds no computation — only the `AutoCache` decision — so being loose
costs a cache that would have fit, never a wrong answer.
"""
function _build_footprint_curvilinear(
    grid::FlowGeometries.Grids.CurvilinearGrid{T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    cache_strategy::AbstractCacheStrategy = AutoCache(),
    cache_byte_budget::Integer = DEFAULT_CACHE_BYTE_BUDGET,
    kwargs...,   # accepts (and ignores) mask_strategy — irrelevant here, no separable-Gaussian path
                 # for CurvilinearGrid (non-Cartesian-only concern), but accepted for a uniform call
                 # signature with the StructuredGrid `build_footprint` methods.
) where {T<:AbstractFloat}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    geo = FlowGeometries.Grids.grid_geometry(grid)
    rad = Kernels.kernel_radius(kernel, scale)

    # Smallest adjacent-node spacing in each index direction (walk both directions of the 2D mesh) —
    # an O(N) one-time pass to learn the mesh's spacing, not part of the O(N·M) storage question.
    min_di = T(Inf)
    min_dj = T(Inf)
    for j in 1:Ny, i in 1:Nx
        c = FlowGeometries.Grids.coords(SA.SVector, grid, i, j)
        if i < Nx
            d = FlowGeometries.Geometry.distance(geo, c, FlowGeometries.Grids.coords(SA.SVector, grid, i + 1, j))
            d > 0 && (min_di = min(min_di, d))
        end
        if j < Ny
            d = FlowGeometries.Geometry.distance(geo, c, FlowGeometries.Grids.coords(SA.SVector, grid, i, j + 1))
            d > 0 && (min_dj = min(min_dj, d))
        end
    end
    di_lim = isfinite(min_di) && min_di > 0 ? ceil(Int, rad / min_di) : 0
    dj_lim = isfinite(min_dj) && min_dj > 0 ? ceil(Int, rad / min_dj) : 0
    mt = _query_topology(grid, rad)

    cache = if _should_cache(cache_strategy, _scattered_cache_bytes(Nx, Ny, di_lim, dj_lim, T), cache_byte_budget)
        ii = Int[]
        jj = Int[]
        w = T[]
        sizehint!(ii, Nx * Ny * (2*di_lim + 1) * (2*dj_lim + 1))
        sizehint!(jj, Nx * Ny * (2*di_lim + 1) * (2*dj_lim + 1))
        sizehint!(w, Nx * Ny * (2*di_lim + 1) * (2*dj_lim + 1))
        ptr = Vector{Int}(undef, Nx * Ny + 1)
        ptr[1] = 1
        sc = FlowGeometries.Connectivity.ball_scratch()
        for j in 1:Ny, i in 1:Nx # column-major target order: t = i + (j-1)*Nx
            t = i + (j - 1) * Nx
            target = FlowGeometries.Grids.coords(SA.SVector, grid, i, j)
            _scattered_foldl(
                nothing, grid, target, i, j, Nx, Ny, di_lim, dj_lim, false, false, zero(T), zero(T), false, rad, mt, sc,
            ) do _, iin, jjn, d
                push!(ii, iin)
                push!(jj, jjn)
                push!(w, Kernels.kernel_weight(kernel, d, scale) * FlowGeometries.Grids.area(grid, iin, jjn))
                nothing
            end
            ptr[t+1] = length(ii) + 1
        end
        ScatteredCache(ii, jj, w, ptr)
    else
        nothing
    end
    return ScatteredFilterPlan(kernel, scale, rad, di_lim, dj_lim, false, false, zero(T), zero(T), false, cache, mt)
end

"""
    build_footprint(grid::CurvilinearGrid, kernel, scale; kwargs...) -> ScatteredFilterPlan

Real-space direct-sum footprint for a curvilinear grid (see [`_build_footprint_curvilinear`](@ref)).
"""
build_footprint(
    grid::FlowGeometries.Grids.CurvilinearGrid{T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    kwargs...,
) where {T<:AbstractFloat} = _build_footprint_curvilinear(grid, kernel, scale; kwargs...)

"""
    FilterFootprintND{N, T}

General N-dimensional footprint: in-support neighbour offsets (`NTuple{N,Int}`) and their geometric
weights `w = kernel_weight(distance) · cell_measure`. Used for 1D and 3D (Cartesian,
translation-invariant ⇒ a single offset set); the 2D path uses the optimized per-row
`FilterFootprint`.
"""
struct FilterFootprintND{N, T<:AbstractFloat, VO<:AbstractVector{NTuple{N,Int}}, VT<:AbstractVector{T}}
    offsets::VO
    w::VT
end

"""
    build_footprint(grid, kernel, scale) -> FilterFootprint

Fast path — real multiple dispatch, not a runtime check: both axes are `AbstractRange`, a
compile-time proof of constant spacing, so the footprint is genuinely translation-invariant and can
be shared via a single (Cartesian) or per-latitude-band (spherical) offset/weight cache. Spacing is
read via `step(...)` directly from the axis that's already proven uniform by its type — not from the
geometry's separately-stored `dx`/`dy` scalar, so there's no possibility of the two disagreeing.
"""
function build_footprint(
    # `StructuredGrid{G,T,N,TP,C,…}`: `TP` is the per-direction topology and `C` the coordinates, so the
    # axis constraint belongs in slot FIVE. Naming `TP` rather than leaving it implicit is what keeps
    # this from silently constraining the topology instead — a signature that then matches nothing and
    # sends every grid to the general path with no error anywhere.
    grid::FlowGeometries.Grids.StructuredGrid{G,T,2,TP,<:Tuple{AbstractRange,AbstractRange}},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}, TP<:NTuple{2,FlowGeometries.Grids.AbstractTopology}}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    rad = Kernels.kernel_radius(kernel, scale)

    di = Int[]
    dj = Int[]
    w = T[]
    ptr = Int[1]

    if G <: FlowGeometries.Geometry.CartesianGeometry{T}
        dx = step(FlowGeometries.Grids.coordinates(grid, 1))
        dy = step(FlowGeometries.Grids.coordinates(grid, 2))
        A = FlowGeometries.Grids.area(grid, 1, 1)   # uniform Cartesian cell area
        di_lim = dx > 0 ? ceil(Int, rad / dx) : 0
        dj_lim = dy > 0 ? ceil(Int, rad / dy) : 0
        # Exact window size (a single shared translation-invariant footprint, not per grid point):
        # every candidate offset in this rectangle is visited exactly once below.
        sizehint!(di, (2*di_lim + 1) * (2*dj_lim + 1))
        sizehint!(dj, (2*di_lim + 1) * (2*dj_lim + 1))
        sizehint!(w, (2*di_lim + 1) * (2*dj_lim + 1))
        for ddj in -dj_lim:dj_lim, ddi in -di_lim:di_lim
            d = sqrt((ddi * dx)^2 + (ddj * dy)^2)
            if d <= rad
                push!(di, ddi)
                push!(dj, ddj)
                push!(w, Kernels.kernel_weight(kernel, T(d), scale) * A)
            end
        end
        push!(ptr, length(di) + 1)
        return FilterFootprint(di, dj, w, ptr, 1)
    else
        R = FlowGeometries.Geometry.radius(FlowGeometries.Grids.grid_geometry(grid))
        dλ = step(FlowGeometries.Grids.coordinates(grid, 1))
        dφ = step(FlowGeometries.Grids.coordinates(grid, 2))
        dj_lim = dφ > 0 ? ceil(Int, rad / (R * dφ)) : 0
        # Per band, not one global bound: the longitude window widens as cosφ→0, so a pole-worst-case
        # bound over-reserves every band away from the poles. Pass 1 records each band's window and its
        # entry count; pass 2 reuses them.
        #
        # The window must hold for every row the ball reaches, not just the target's — a row nearer the
        # pole spans more longitude for the same radius — which is what `metric_window` gives, taking
        # the smallest cosφ over the latitude window. Capped at one turn besides: longitude identifies
        # rather than tiles, so a ring contributes each of its `Nx` cells at most once.
        di_lims = Vector{Int}(undef, Ny)
        total_entries = 0
        for j in 1:Ny
            dl = FlowGeometries.Connectivity.metric_window(grid, (1, j), rad)[1]
            di_lims[j] = min(dl, Nx ÷ 2)
            for ddj in -dj_lim:dj_lim
                jj = j + ddj
                (1 <= jj <= Ny) || continue
                total_entries += min(2 * dl + 1, Nx)
            end
        end
        sizehint!(di, total_entries)
        sizehint!(dj, total_entries)
        sizehint!(w, total_entries)
        for j in 1:Ny
            φ = FlowGeometries.Grids.coordinates(grid, 2)[j]
            di_lim = di_lims[j]
            for ddj in -dj_lim:dj_lim
                jj = j + ddj
                (1 <= jj <= Ny) || continue
                φ2 = FlowGeometries.Grids.coordinates(grid, 2)[jj]
                A = FlowGeometries.Grids.area(grid, 1, jj)   # spherical cell area depends only on latitude
                # Asymmetric by one when the window closes the ring: `-dl:dl` is `2dl+1` offsets, and at
                # `dl = Nx÷2` on an even ring that is `Nx+1` — the antipodal column visited from both
                # sides. Dropping the upper end leaves each of the `Nx` columns exactly once.
                hi_i = (2 * di_lim + 1 > Nx) ? di_lim - 1 : di_lim
                for ddi in -di_lim:hi_i
                    # Great-circle distance with Δλ = ddi·dλ (longitude-translation-invariant).
                    d = FlowGeometries.Geometry.distance(
                        FlowGeometries.Grids.grid_geometry(grid),
                        SA.SVector{2,T}(zero(T), φ),
                        SA.SVector{2,T}(T(ddi) * dλ, φ2),
                    )
                    if d <= rad
                        push!(di, ddi)
                        push!(dj, ddj)
                        push!(w, Kernels.kernel_weight(kernel, d, scale) * A)
                    end
                end
            end
            push!(ptr, length(di) + 1)
        end
        return FilterFootprint(di, dj, w, ptr, Ny)
    end
end

"""
    build_footprint(grid, kernel, scale; kwargs...) -> ScatteredFilterPlan

General path: at least one axis is a plain (non-`Range`) `AbstractVector`, which makes no type-level
uniformity guarantee — its values might happen to be evenly spaced, but nothing proves it, so no
assumption is made and the always-correct per-point plan is built instead. (Less specific than
the method above, so Julia only reaches this one when the fast method's constraint doesn't match.)
"""
function build_footprint(
    grid::FlowGeometries.Grids.StructuredGrid{G,T,2},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    return _build_footprint_scattered(grid, kernel, scale; kwargs...)
end

# ---------------------------------------------------------------------------
# Exact top-hat filtering on any rectilinear 2D `StructuredGrid`, in O(N·dj_lim) rather than
# O(N·di_lim·dj_lim). Requires `TopHatKernel`, whose weight is constant inside its support, so the
# weighted window sum is a plain interval sum. Two properties of a rectilinear grid make that sum O(1):
#
#  1. The cell measure is separable, `area[i,j] == wx[i]*wy[j]`:
#       Cartesian  Δx_i · Δy_j                    → wx[i]=Δx_i,  wy[j]=Δy_j
#       Spherical  R²·cos(φ_j)·Δλ_i·Δφ_j          → wx[i]=Δλ_i,  wy[j]=R²cos(φ_j)Δφ_j
#     so the window sum factors as `Σ_jj wy[jj] · Σ_ii field[ii,jj]·wx[ii]`, the inner term being an
#     interval sum along one row — O(1) from a per-row prefix sum.
#
#  2. At a fixed row offset the in-support axis-1 indices form one contiguous interval whose endpoints
#     are monotone in the target index, so a two-pointer sweep finds them in O(1) amortized.
#     Cartesian: `|Δx| ≤ √(rad²−Δy²)`. Spherical: `cos d = sinφ₁sinφ₂ + cosφ₁cosφ₂·cosΔλ` decreases
#     monotonically in `|Δλ|` on [0,π], so `d ≤ rad` ⟺ `|Δλ| ≤ acos((cos rad − sinφ₁sinφ₂)/(cosφ₁cosφ₂))`.
# ---------------------------------------------------------------------------

"""
    PrefixSumTopHatPlan{T,VT,MT,DT,BT,WX,WY}

Exact `O(N·dj_lim)` top-hat footprint for a rectilinear 2D `StructuredGrid` — see the section comment
above for the derivation. Holds the separable measure factors (`wx`,`wy`), the extended axis-1
coordinate array the two-pointer sweep walks (tripled when axis 1 is periodic, so a wrapped support
interval is still one contiguous run), and preallocated scratch so `filter_apply!` allocates nothing.

`prefix_den` (Deformable masking) is the prefix sum of `mask·wx` — mask-only, so it is built ONCE at
plan-build time, mirroring `SeparableFootprint`/`FFTWFilterPlan`'s `invrenorm` convention.
`ZeroFill`'s denominator is mask-independent and needs only the 1-D `prefix_wx`.
"""
struct PrefixSumTopHatPlan{
    T<:AbstractFloat,
    VT<:AbstractVector{T},
    MT<:AbstractMatrix{T},
    DT<:Union{Nothing,AbstractMatrix{T}},
    BT<:AbstractVector{Int},
    WX<:AbstractVector{T},
    WY<:AbstractVector{T},
}
    rad::T
    dj_lim::Int
    periodic_x::Bool
    periodic_y::Bool
    masked::Bool        # grid has inactive cells, so Deformable genuinely needs `prefix_den`
    x_period::T
    y_period::T
    # The grid's own measure factors, whatever it stores them as: a uniform axis carries one number
    # and a length, so these are not pinned to a dense vector.
    wx::WX              # axis-1 cell width;  measure[i,j] == wx[i]*wy[j]
    wy::WY              # axis-2 measure factor
    xe::VT              # extended axis-1 coordinates (nrep*Nx), strictly increasing
    src::BT             # xe[k] belongs to real axis-1 index src[k]
    prefix_wx::VT       # cumulative Σ wx along the extended axis (nrep*Nx+1) — ZeroFill denominator
    prefix_num::MT      # per-row cumulative Σ mask·field·wx (nrep*Nx+1 × Ny), refilled each apply
    prefix_den::DT      # per-row cumulative Σ mask·wx (nrep*Nx+1 × Ny), or nothing (Deformable only)
    den::MT             # per-point denominator accumulator (Nx × Ny); row-disjoint, so thread-safe
end

"""
    _rectilinear_measure_factors(grid) -> (wx, wy)

The grid's own per-axis measure factors, so that `measure[i, j] == wx[i] * wy[j]` exactly. Read from
the stored `SeparableMeasure` rather than rebuilt here: rebuilding has to reproduce every convention
the measure was constructed under, and a degenerate direction is where that fails — a single-latitude
grid measures arc length `R·Δλ` and a single-longitude one `R·Δφ`, neither of which is the `R²cosφ`
area form.
"""
function _rectilinear_measure_factors(
    grid::FlowGeometries.Grids.StructuredGrid{G,T,2},
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    mf = FlowGeometries.Grids.measure_factors(grid)
    mf === nothing && throw(ArgumentError(
        "the prefix-sum top-hat path needs a separable cell measure, but this grid's measure is dense",
    ))
    return mf[1], mf[2]
end

"""
    _build_prefixsum_tophat(grid, kernel::TopHatKernel, scale; mask_strategy=ZeroFill(), kwargs...)

Build the exact `O(N·dj_lim)` prefix-sum top-hat plan. `kwargs...` absorbs (and ignores)
`cache_strategy`/`cache_byte_budget` — there is no per-point neighbour list here to cache at all.
"""
function _build_prefixsum_tophat(
    grid::FlowGeometries.Grids.StructuredGrid{G,T,2},
    kernel::Kernels.TopHatKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    rad = Kernels.kernel_radius(kernel, scale)
    periodic_x = FlowGeometries.Grids.isperiodic(grid, 1)
    periodic_y = FlowGeometries.Grids.isperiodic(grid, 2)
    is_cartesian = G <: FlowGeometries.Geometry.CartesianGeometry{T}

    # The grid's own stored wrap length per direction — never re-derived from the samples, or the two
    # real-space engines wrap at different lengths on a stretched axis.
    x_period = (periodic_x && Nx > 1) ? T(FlowGeometries.Grids.period(grid, 1)) : zero(T)
    y_period = (periodic_y && Ny > 1) ? T(FlowGeometries.Grids.period(grid, 2)) : zero(T)

    wx, wy = _rectilinear_measure_factors(grid)

    # Axis-2 band bound from the axis's own minimum gap, converted to a physical distance.
    min_dy = FlowGeometries.Grids.minimum_spacing(grid, 2)
    dy_phys = is_cartesian ? min_dy : FlowGeometries.Geometry.radius(FlowGeometries.Grids.grid_geometry(grid)) * min_dy
    dj_lim = (isfinite(dy_phys) && dy_phys > 0) ? min(Ny - 1, ceil(Int, rad / dy_phys)) : 0

    # Extended axis-1 coordinates: one copy when non-periodic; three (shifted −P, 0, +P) when periodic,
    # so a wrapped support interval stays a single contiguous run and the two-pointer stays monotone.
    # The sweep searches `xe` by value (`x[i] ± hw`), so no index offset into the replicas is needed.
    # `xe` must be ascending — both the prefix sums and the sweep depend on it — while the axis itself
    # may be stored descending, so order the extension by coordinate and carry the real axis index in
    # `src`. Each replica is internally sorted and the period exceeds the axis extent, so concatenating
    # in shift order keeps `xe` globally sorted.
    nrep = (periodic_x && Nx > 1) ? 3 : 1
    ne = nrep * Nx
    ord = issorted(FlowGeometries.Grids.coordinates(grid, 1)) ? collect(1:Nx) : sortperm(FlowGeometries.Grids.coordinates(grid, 1))
    xe = Vector{T}(undef, ne)
    src = Vector{Int}(undef, ne)
    @inbounds for r in 0:(nrep - 1), t in 1:Nx
        k = r * Nx + t
        i = ord[t]
        xe[k] = FlowGeometries.Grids.coordinates(grid, 1)[i] + (nrep == 3 ? (r - 1) * x_period : zero(T))
        src[k] = i
    end

    prefix_wx = Vector{T}(undef, ne + 1)
    prefix_wx[1] = zero(T)
    @inbounds for k in 1:ne
        prefix_wx[k + 1] = prefix_wx[k] + wx[src[k]]
    end

    prefix_num = zeros(T, ne + 1, Ny)
    den = zeros(T, Nx, Ny)
    masked = !all(FlowGeometries.Grids.mask(grid))

    # Deformable's denominator depends only on the mask, so build it once here, never per apply.
    prefix_den = if mask_strategy isa Deformable && masked
        P = zeros(T, ne + 1, Ny)
        @inbounds for j in 1:Ny
            acc = zero(T)
            for k in 1:ne
                i = src[k]
                acc += FlowGeometries.Grids.mask(grid)[i, j] ? wx[i] : zero(T)
                P[k + 1, j] = acc
            end
        end
        P
    else
        nothing
    end

    return PrefixSumTopHatPlan(
        rad, dj_lim, periodic_x, periodic_y, masked, x_period, y_period,
        wx, wy, xe, src, prefix_wx, prefix_num, prefix_den, den,
    )
end

"""
    prefixsum_fill_numerator!(fp, field, grid)

The single O(N) per-apply pass: refill `fp.prefix_num` (cumulative `mask·field·wx` per row) from the
current field. Must run before any `apply_prefixsum_tophat_row!` call for that field — both the serial
and threaded drivers guarantee this.
"""
function prefixsum_fill_numerator!(
    fp::PrefixSumTopHatPlan{T}, field::AbstractMatrix, grid::FlowGeometries.Grids.StructuredGrid,
) where {T<:AbstractFloat}
    _, Ny = FlowGeometries.Grids.size_tuple(grid)
    for j in 1:Ny
        prefixsum_fill_numerator_row!(fp, field, grid, j)
    end
    return nothing
end

"""
    prefixsum_fill_numerator_row!(fp, field, grid, j)

Fill row `j`'s numerator prefix scan. Writes only column `j` of `fp.prefix_num`, so rows are mutually
independent and this may be run concurrently across `j` (the threaded backend does exactly that).
"""
function prefixsum_fill_numerator_row!(
    fp::PrefixSumTopHatPlan{T}, field::AbstractMatrix, grid::FlowGeometries.Grids.StructuredGrid, j::Integer,
) where {T<:AbstractFloat}
    ne = length(fp.xe)
    src, P, mask, wx = fp.src, fp.prefix_num, FlowGeometries.Grids.mask(grid), fp.wx
    acc = zero(T)
    @inbounds for k in 1:ne
        i = src[k]
        acc += mask[i, j] ? T(field[i, j]) * wx[i] : zero(T)
        P[k + 1, j] = acc
    end
    return nothing
end

# A plan built for `ZeroFill` on a masked grid holds no `prefix_den`, so it cannot serve a
# renormalizing strategy. Checked once per apply, not per row.
#
# `@noinline`, and the message interpolates nothing: interpolating `typeof(strategy)` pulls Base's
# dynamically-dispatched `show(::DataType)` into the call graph, so `JET.@test_opt` on any caller
# reports runtime dispatch even though the throw never executes.
@noinline function _prefixsum_strategy_mismatch()
    throw(ArgumentError(
        "PrefixSumTopHatPlan was built for ZeroFill masking, so it holds no renormalization data, but " *
        "is being applied with a renormalizing (non-ZeroFill) mask strategy on a masked grid. Rebuild " *
        "the plan with the same `mask_strategy` you intend to apply with.",
    ))
end

@inline function _prefixsum_check_strategy(fp::PrefixSumTopHatPlan, strategy::AbstractMaskStrategy)
    if !(strategy isa ZeroFill) && fp.masked && fp.prefix_den === nothing
        _prefixsum_strategy_mismatch()
    end
    return nothing
end

"""
    apply_prefixsum_tophat_row!(out, grid, fp, strategy, j) -> out

Fill output row `j` from the (already current) prefix sums. For each row offset in the band, sweeps
axis 1 with two MONOTONE pointers — `O(1)` amortized per point, not a per-point binary search — so the
whole row costs `O(Nx·dj_lim)`. Touches only row `j` of `out`/`fp.den`, so rows may run concurrently.
"""
function apply_prefixsum_tophat_row!(
    out::AbstractMatrix{T}, grid::FlowGeometries.Grids.StructuredGrid{G,T,2}, fp::PrefixSumTopHatPlan{T},
    strategy::AbstractMaskStrategy, j::Integer,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    ne = length(fp.xe)
    x, xe, src, rad = FlowGeometries.Grids.coordinates(grid, 1), fp.xe, fp.src, fp.rad
    Pn, Pd, Pwx, den = fp.prefix_num, fp.prefix_den, fp.prefix_wx, fp.den
    use_mask_den = !(strategy isa ZeroFill) && Pd !== nothing
    y_t = FlowGeometries.Grids.coordinates(grid, 2)[j]

    @inbounds for i in 1:Nx
        out[i, j] = zero(T)
        den[i, j] = zero(T)
    end

    # Visit each contributing row EXACTLY ONCE. `nb` is capped at `Ny` because a periodic axis-2 with a
    # band wider than the grid would otherwise map two different offsets onto the same row via `mod1`
    # and count it twice; capping at `Ny` consecutive raw offsets keeps `mod1` a bijection onto `1:Ny`.
    # The axis-2 displacement is taken as the MINIMUM IMAGE, which is both the correct periodic
    # displacement and independent of whether the axis is stored ascending or descending.
    nb = min(2 * fp.dj_lim + 1, Ny)
    @inbounds for b in 0:(nb - 1)
        jj_raw = j - fp.dj_lim + b
        (fp.periodic_y || (1 <= jj_raw <= Ny)) || continue
        jj = mod1(jj_raw, Ny)
        dy = FlowGeometries.Grids.coordinates(grid, 2)[jj] - y_t
        if fp.periodic_y && fp.y_period > zero(T)
            dy -= fp.y_period * round(dy / fp.y_period)
        end
        hw = FlowGeometries.Connectivity.metric_band(grid, 1, y_t, y_t + dy, rad)
        hw < zero(T) && continue
        wyj = fp.wy[jj]

        if fp.periodic_x && ne > Nx && T(2) * hw >= fp.x_period
            # Support spans at least a full period: every cell of the row is included exactly once.
            # Summing over the tripled axis would triple-count, so use one replica's total directly.
            num_all = Pn[Nx + 1, jj] - Pn[1, jj]
            den_all = use_mask_den ? (Pd[Nx + 1, jj] - Pd[1, jj]) : (Pwx[Nx + 1] - Pwx[1])
            for i in 1:Nx
                out[i, j] += wyj * num_all
                den[i, j] += wyj * den_all
            end
        elseif !fp.periodic_x && ne == Nx && x isa AbstractRange && step(x) > zero(T)
            # Uniform ascending axis: the two pointers advance by exactly one per target, so the window
            # is a constant ±w and the walk collapses. `w` is found with the same `≤` the pointer loop
            # uses, so it makes the same boundary decision. Peeling the interior leaves it branch-free.
            w = 0
            while (w + 1) * step(x) <= hw
                w += 1
            end
            ilo = min(w + 1, Nx + 1)
            ihi = max(Nx - w, 0)
            @inbounds for i in 1:min(w, Nx)                     # left edge: lo clamps to 1
                hi = min(Nx, i + w)
                out[i, j] += wyj * (Pn[hi + 1, jj] - Pn[1, jj])
                den[i, j] += wyj * (use_mask_den ? (Pd[hi + 1, jj] - Pd[1, jj]) : (Pwx[hi + 1] - Pwx[1]))
            end
            if use_mask_den
                @inbounds @simd for i in ilo:ihi                # interior: no clamping at all
                    out[i, j] += wyj * (Pn[i + w + 1, jj] - Pn[i - w, jj])
                    den[i, j] += wyj * (Pd[i + w + 1, jj] - Pd[i - w, jj])
                end
            else
                @inbounds @simd for i in ilo:ihi
                    out[i, j] += wyj * (Pn[i + w + 1, jj] - Pn[i - w, jj])
                    den[i, j] += wyj * (Pwx[i + w + 1] - Pwx[i - w])
                end
            end
            @inbounds for i in max(ihi + 1, w + 1):Nx           # right edge: hi clamps to Nx
                lo = max(1, i - w)
                out[i, j] += wyj * (Pn[Nx + 1, jj] - Pn[lo, jj])
                den[i, j] += wyj * (use_mask_den ? (Pd[Nx + 1, jj] - Pd[lo, jj]) : (Pwx[Nx + 1] - Pwx[lo]))
            end
        else
            lo = 1
            hi = 0
            # Walk the TARGET index in ascending-coordinate order (`src[1:Nx]` is exactly that
            # permutation), not raw index order: the two pointers only ever advance forward, which
            # requires the target coordinate to increase monotonically. On a descending axis raw order
            # would decrease it and silently corrupt every interval.
            for t in 1:Nx
                i = src[t]
                xc = x[i]
                xlo = xc - hw
                xhi = xc + hw
                while lo <= ne && xe[lo] < xlo
                    lo += 1
                end
                while hi < ne && xe[hi + 1] <= xhi
                    hi += 1
                end
                if hi >= lo
                    out[i, j] += wyj * (Pn[hi + 1, jj] - Pn[lo, jj])
                    den[i, j] += wyj * (use_mask_den ? (Pd[hi + 1, jj] - Pd[lo, jj]) : (Pwx[hi + 1] - Pwx[lo]))
                end
            end
        end
    end

    @inbounds for i in 1:Nx
        out[i, j] = (FlowGeometries.Grids.isactive(grid, i, j) && den[i, j] > T(1e-15)) ? out[i, j] / den[i, j] : zero(T)
    end
    return out
end

"""
    apply_prefixsum_tophat!(out, field, grid, fp, strategy) -> out

Whole-grid exact top-hat convolve: one `O(N)` prefix-sum pass over the field, then one `O(N·dj_lim)`
two-pointer sweep.
"""
function apply_prefixsum_tophat!(
    out::AbstractMatrix{T}, field::AbstractMatrix, grid::FlowGeometries.Grids.StructuredGrid{G,T,2},
    fp::PrefixSumTopHatPlan{T}, strategy::AbstractMaskStrategy,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    _, Ny = FlowGeometries.Grids.size_tuple(grid)
    _prefixsum_check_strategy(fp, strategy)
    prefixsum_fill_numerator!(fp, field, grid)
    for j in 1:Ny
        apply_prefixsum_tophat_row!(out, grid, fp, strategy, j)
    end
    return out
end

"""
    apply_footprint!(out, field, grid, fp::PrefixSumTopHatPlan, strategy, periodic_x, periodic_y) -> out

`apply_footprint!`-shaped entry point for the prefix-sum plan, so the generic whole-grid convolve name
works uniformly across every footprint type. `periodic_x`/`periodic_y` are accepted for interface
compatibility but IGNORED: unlike the offset-based footprints, this plan captured its periodicity (and
the matching extended-coordinate layout) from the grid at build time, and cannot honour a different
choice at apply time.
"""
apply_footprint!(
    out::AbstractMatrix{T}, field::AbstractMatrix, grid::FlowGeometries.Grids.StructuredGrid{G,T,2},
    fp::PrefixSumTopHatPlan{T}, strategy::AbstractMaskStrategy, periodic_x::Bool, periodic_y::Bool,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}} =
    apply_prefixsum_tophat!(out, field, grid, fp, strategy)

"""
    build_footprint(grid::StructuredGrid{<:AbstractGeometry,T,2}, kernel::TopHatKernel, scale; kwargs...) -> PrefixSumTopHatPlan

Exact prefix-sum top-hat path for any rectilinear 2D `StructuredGrid` (see the section comment above).
More specific than the generic 2D methods (constrained on kernel type), so Julia selects it whenever
`kernel isa TopHatKernel`.
"""
function build_footprint(
    grid::FlowGeometries.Grids.StructuredGrid{G,T,2},
    kernel::Kernels.TopHatKernel,
    scale::T;
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    return _build_prefixsum_tophat(grid, kernel, scale; kwargs...)
end

# Range axes are a strict special case of "rectilinear", so they take the same exact prefix-sum path.
# This method exists to resolve what would otherwise be a genuine dispatch AMBIGUITY between the
# generic-kernel Range-axis method and the generic-axis `TopHatKernel` method above (neither is more
# specific than the other for the Range+TopHat combination) — it is strictly more specific than both.
function build_footprint(
    grid::FlowGeometries.Grids.StructuredGrid{G,T,2,TP,<:Tuple{AbstractRange,AbstractRange}},
    kernel::Kernels.TopHatKernel,
    scale::T;
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}, TP<:NTuple{2,FlowGeometries.Grids.AbstractTopology}}
    return _build_prefixsum_tophat(grid, kernel, scale; kwargs...)
end

# ---------------------------------------------------------------------------
# Separable Gaussian fast path. `exp(-α(Δx²+Δy²)/ℓ²)` factors as `Gx(Δx)·Gy(Δy)`, so a 2D convolution
# becomes a row pass then a column pass: O(N·r) against O(N·r²), up to the truncation-shape difference
# noted below. The factorization needs a rectilinear grid, not a uniform one — a stretched axis makes
# `Gx` depend on position as well as offset, which is a wider weight table (see
# [`_separable_axis_weights`](@ref)) rather than a different algorithm.
#
# Cartesian only: great-circle distance does not factor, so a spherical grid takes the per-latitude-band
# path instead.
# ---------------------------------------------------------------------------

"""
    SeparableFootprint{T}

Precomputed 1D Gaussian weight vectors (`gx`,`gy`) plus preallocated scratch buffers for the
row-pass/column-pass separable convolution. `invrenorm` (Deformable masking only) is the precomputed
reciprocal local kernel mass over active cells — the SAME separable machinery run once, at plan-build
time, on `Float.(mask)` instead of `field`, mirroring `FFTWFilterPlan`/`SHTFilterPlan`'s established
`invrenorm` pattern. `Nx_profile`/`Ny_profile` (ZeroFill masking) are the mask-INDEPENDENT denominator
profiles: `Σ w` over valid (in-bounds/periodic) offsets is itself separable into one Nx-length and
one Ny-length vector, since which offsets are valid depends only on `i` (resp. `j`) and periodicity,
never on the mask.

Note: unlike the disk-truncated (`d <= rad`) non-separable footprint, this truncates each axis
independently at the SAME per-axis `rad` (the Gaussian's own 1D marginal decays at the identical rate
`kernel_radius` was derived from) — a square window, not a disk. The Gaussian has no true hard
support (only a numerical truncation tolerance), so this is an equally valid truncation, just a
different shape — matched against `RealSpace` within a measured tolerance, not asserted bit-identical.
"""
struct SeparableFootprint{
    T<:AbstractFloat,
    VT<:AbstractVecOrMat{T},
    PVT<:Union{Nothing,AbstractVector{T}},
    MT<:AbstractMatrix{T},
    IMT<:Union{Nothing,AbstractMatrix{T}},
}
    gx::VT
    gy::VT
    di_lim::Int
    dj_lim::Int
    periodic_x::Bool
    periodic_y::Bool
    Nx_profile::PVT
    Ny_profile::PVT
    invrenorm::IMT
    masked::Bool          # whether the grid had any inactive cell when this was built
    masked_input::MT
    row_pass::MT
end

# The per-row and per-column bodies, factored out so the ThreadedBackend extension parallelizes over
# `j` through these same functions. The row pass at row `j` reads only `src[:, j]`, so it parallelizes
# over `j`; the column pass reads `row_pass[:, jj]` across rows, so it must run as a separate pass
# after every row's `row_pass` is written, and cannot be fused with it.
"""
    _sepw(g, i, k) -> T

Weight of stencil slot `k` at axis position `i`.

The Gaussian factorizes on ANY rectilinear grid — `exp(-α(Δx²+Δy²)/ℓ²) = Gx(Δx)·Gy(Δy)` needs no
constant spacing — but on a uniform axis `Gx` depends on the OFFSET alone, while on a stretched one it
depends on the position too. Both are the same convolution with a different weight table, so the two
are one code path distinguished by the table's rank: a vector is shared across positions and a matrix
is `(2·lim+1) × N`, column-major so each position's stencil is contiguous.
"""
@inline _sepw(g::AbstractVector, ::Int, k::Int) = @inbounds g[k]
@inline _sepw(g::AbstractMatrix, i::Int, k::Int) = @inbounds g[i, k]

@inline function _separable_row_pass_at!(
    row_pass::AbstractMatrix{T}, src::AbstractMatrix{T}, gx::AbstractVecOrMat{T},
    di_lim::Int, periodic_x::Bool, Nx::Int, j::Int,
) where {T<:AbstractFloat}
    # Taps outermost, position innermost. Accumulating one output point at a time makes the inner loop
    # an FP reduction, which cannot be reassociated and so never vectorizes; this way the inner loop is
    # a unit-stride axpy over `i` and does. Measured 1.29 → 0.30 ns per weighted add at Nx=512, w=32.
    @inbounds begin
        for i in 1:Nx
            row_pass[i, j] = zero(T)
        end
        for ddi in (-di_lim):di_lim
            k = ddi + di_lim + 1
            if periodic_x && abs(ddi) < Nx
                # A wrapped tap is two contiguous runs, each at a CONSTANT offset. Writing it as one
                # loop over `mod1` costs the vectorization, since the index is then data-dependent.
                if ddi >= 0
                    @simd for i in 1:(Nx - ddi)
                        row_pass[i, j] += _sepw(gx, i, k) * src[i + ddi, j]
                    end
                    @simd for i in (Nx - ddi + 1):Nx
                        row_pass[i, j] += _sepw(gx, i, k) * src[i + ddi - Nx, j]
                    end
                else
                    @simd for i in 1:(-ddi)
                        row_pass[i, j] += _sepw(gx, i, k) * src[i + ddi + Nx, j]
                    end
                    @simd for i in (-ddi + 1):Nx
                        row_pass[i, j] += _sepw(gx, i, k) * src[i + ddi, j]
                    end
                end
            elseif periodic_x
                # A tap wider than the axis wraps more than once, so it needs the general index.
                @simd for i in 1:Nx
                    row_pass[i, j] += _sepw(gx, i, k) * src[mod1(i + ddi, Nx), j]
                end
            else
                @simd for i in max(1, 1 - ddi):min(Nx, Nx - ddi)
                    row_pass[i, j] += _sepw(gx, i, k) * src[i + ddi, j]
                end
            end
        end
    end
    return nothing
end

@inline function _separable_column_pass_at!(
    dst::AbstractMatrix{T}, row_pass::AbstractMatrix{T}, gy::AbstractVecOrMat{T},
    dj_lim::Int, periodic_y::Bool, Nx::Int, Ny::Int, j::Int,
) where {T<:AbstractFloat}
    # Same inversion as the row pass, and here the weight is constant across `i` — it is indexed by the
    # output column `j` — so it hoists out of the inner loop entirely.
    @inbounds begin
        for i in 1:Nx
            dst[i, j] = zero(T)
        end
        for ddj in (-dj_lim):dj_lim
            jj = j + ddj
            if jj < 1 || jj > Ny
                periodic_y || continue
                jj = mod1(jj, Ny)
            end
            wt = _sepw(gy, j, ddj + dj_lim + 1)
            @simd for i in 1:Nx
                dst[i, j] += wt * row_pass[i, jj]
            end
        end
    end
    return nothing
end

# Row-pass (over axis 1) then column-pass (over axis 2) of `src` into `dst`, using `fp`'s scratch
# `row_pass` buffer — the single shared primitive both the plan-build-time `invrenorm` computation
# and every serial `apply_separable!` call use, so they can never drift out of sync with
# each other. The ThreadedBackend extension calls `_separable_row_pass_at!`/`_separable_column_pass_at!`
# directly (parallelized over `j`) instead of this serial driver.
function _separable_convolve!(dst::AbstractMatrix{T}, src::AbstractMatrix{T}, fp::SeparableFootprint{T}, Nx::Int, Ny::Int) where {T<:AbstractFloat}
    gx, gy = fp.gx, fp.gy
    di_lim, dj_lim = fp.di_lim, fp.dj_lim
    periodic_x, periodic_y = fp.periodic_x, fp.periodic_y
    row_pass = fp.row_pass
    for j in 1:Ny
        _separable_row_pass_at!(row_pass, src, gx, di_lim, periodic_x, Nx, j)
    end
    for j in 1:Ny
        _separable_column_pass_at!(dst, row_pass, gy, dj_lim, periodic_y, Nx, Ny, j)
    end
    return dst
end

"""
    _separable_axis_weights(x, lim, periodic, period, kernel, scale, wfac) -> AbstractVecOrMat

A separable kernel's per-axis weight table: `Kernels.kernel_profile(kernel, Δx, ℓ)` over the stencil,
in the layout [`_sepw`](@ref) reads. Any kernel with a 1-D profile takes this path — the Gaussian,
whose radial form happens to factor, and [`Kernels.HighOrderKernel`](@ref), which is separable by
definition and has no radial form at all.

Uniform axis: the displacement is `ddi·Δ` wherever the stencil sits, so one vector serves every
position. Stretched axis: the displacement depends on the position too, so the table gains a position
axis — `(2·lim+1) × N`, which is `O(N·lim)` against the `O(N·lim²)` of a per-point neighbour cache, and
leaves the apply at `O(N·lim)` instead of `O(N·lim²)`.

`lim` comes from the SMALLEST gap on the axis, so on a stretched axis a coarse region's stencil is
wider than it needs to be; those slots hold exact zeros rather than being trimmed, which keeps the
inner loop's bounds static. A periodic displacement carries the image offset, matching the tiling
convention the scattered engine uses.
"""
# Cell-averaged weights keep a discontinuous kernel's MASS exact on any grid, but they cannot recover
# the vanishing MOMENTS if a limb is thinner than a cell — there is simply no resolution there to
# distinguish it from a box. That is a warning rather than an error: the filter is still a valid
# normalized low-pass, it just is not the high-order one that was asked for.
_warn_unresolved_limbs(::Kernels.AbstractFilterKernel, _, ::Int, _) = nothing

function _warn_unresolved_limbs(
    kernel::Kernels.HighOrderKernel, x::AbstractVector, d::Int, scale::Real,
)
    length(x) < 2 && return nothing
    Δ = minimum(abs(x[i] - x[i - 1]) for i in (firstindex(x) + 1):lastindex(x))
    b = kernel.b_over_ℓ * scale
    (isfinite(Δ) && Δ > 0 && b < Δ) && @warn(
        "$(nameof(typeof(kernel))) at ℓ = $scale has limb width b = $b, below axis $d's spacing " *
        "$Δ — under one cell per limb, so the vanishing moments it is built for do not survive " *
        "discretization and the result is effectively a box. At b_over_ℓ = $(kernel.b_over_ℓ) this " *
        "needs ℓ ≥ $(Δ / kernel.b_over_ℓ).",
        maxlog = 1,
    )
    return nothing
end

function _separable_axis_weights(
    x::AbstractRange{T}, lim::Int, ::Bool, ::T, kernel::Kernels.AbstractFilterKernel, scale::T,
    ::AbstractVector{T},
) where {T<:AbstractFloat}
    Δ = T(step(x))
    # `profile_cell_average` is the point sample for every smooth kernel and the exact cell integral
    # for a discontinuous one — see `Kernels.profile_cell_average`.
    return [Kernels.profile_cell_average(kernel, T(ddi) * Δ, Δ, scale) for ddi in -lim:lim]
end

function _separable_axis_weights(
    x::AbstractVector{T}, lim::Int, periodic::Bool, period::T,
    kernel::Kernels.AbstractFilterKernel, scale::T, wfac::AbstractVector{T},
) where {T<:AbstractFloat}
    n = length(x)
    # POSITION-major, `(n, 2·lim+1)`: the passes hold a tap fixed and sweep position, so position must
    # be the contiguous axis or every inner loop gathers with stride `2·lim+1`.
    # An untouched slot is an exact zero: outside the domain.
    g = zeros(T, n, 2 * lim + 1)
    @inbounds for i in 1:n, ddi in -lim:lim
        ii = i + ddi
        shift = zero(T)
        if ii < 1 || ii > n
            periodic || continue
            shift = T(fld(ii - 1, n)) * period
            ii = mod1(ii, n)
        end
        # The NEIGHBOUR's measure factor, so the two passes together weight by `kernel · cell area` —
        # the same quantity the scattered engine forms as `kernel_weight(d) * area(grid, ii, jj)`.
        # A rectilinear measure is itself a product of per-axis factors, so it splits over the passes
        # exactly as the kernel does. On a uniform axis it is a constant that cancels in the
        # normalization, which is why the vector method above can leave it out.
        # `wfac[ii]` is the neighbour cell's width, so it doubles as the averaging window: the weight
        # is the kernel's INTEGRAL over that cell, not its value at the node. Identical to the point
        # sample for a smooth kernel (`profile_cell_average`'s default), and the difference between
        # working and not for a discontinuous one on a stretched axis.
        g[i, ddi + lim + 1] =
            Kernels.profile_cell_average(kernel, x[ii] + shift - x[i], wfac[ii], scale) * wfac[ii]
    end
    return g
end

"""
    _build_separable_footprint(grid, kernel::GaussianKernel, scale; mask_strategy=ZeroFill(), kwargs...) -> SeparableFootprint

Build the per-axis weight tables, the mask-dependent normalization data for `mask_strategy`, and the
preallocated scratch buffers for the separable path. `kwargs...` absorbs (and ignores)
`cache_strategy`/`cache_byte_budget` — there is no per-point neighbour list to cache here at all, so
those knobs (which only govern the scattered path) don't apply.
"""
function _build_separable_footprint(
    grid::FlowGeometries.Grids.StructuredGrid{G,T,2},
    kernel::SeparableKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    _warn_unresolved_limbs(kernel, FlowGeometries.Grids.coordinates(grid, 1), 1, scale)
    _warn_unresolved_limbs(kernel, FlowGeometries.Grids.coordinates(grid, 2), 2, scale)
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    rad = Kernels.kernel_radius(kernel, scale)
    # The smallest gap on each axis, so the stencil never under-covers: a coarser region's slots then
    # hold exact zeros rather than missing neighbours.
    dx = FlowGeometries.Grids.minimum_spacing(grid, 1)
    dy = FlowGeometries.Grids.minimum_spacing(grid, 2)
    di_lim = (isfinite(dx) && dx > 0) ? ceil(Int, rad / dx) : 0
    dj_lim = (isfinite(dy) && dy > 0) ? ceil(Int, rad / dy) : 0
    periodic_x = FlowGeometries.Grids.isperiodic(grid, 1)
    periodic_y = FlowGeometries.Grids.isperiodic(grid, 2)
    # The cell measure enters the weights, so the two passes together form `kernel · area` — the same
    # product the scattered engine builds per candidate. A rectilinear measure is separable by
    # construction, so it splits over the passes; anything else cannot take this path at all.
    mf = FlowGeometries.Grids.measure_factors(grid)
    mf === nothing && throw(ArgumentError(
        "the separable path needs a separable cell measure, but this grid's measure is dense",
    ))
    gx = _separable_axis_weights(
        FlowGeometries.Grids.coordinates(grid, 1), di_lim, periodic_x,
        T(FlowGeometries.Grids.period(grid, 1)), kernel, scale, convert(AbstractVector{T}, mf[1]),
    )
    gy = _separable_axis_weights(
        FlowGeometries.Grids.coordinates(grid, 2), dj_lim, periodic_y,
        T(FlowGeometries.Grids.period(grid, 2)), kernel, scale, convert(AbstractVector{T}, mf[2]),
    )

    masked_input = zeros(T, Nx, Ny)
    row_pass = zeros(T, Nx, Ny)
    fully_active = all(FlowGeometries.Grids.mask(grid))
    fp_partial = SeparableFootprint(gx, gy, di_lim, dj_lim, periodic_x, periodic_y, nothing, nothing, nothing, !fully_active, masked_input, row_pass)

    if fully_active || mask_strategy isa ZeroFill
        # `ZeroFill`'s denominator is the mask-independent geometric profile. A fully-active grid takes
        # the same branch whatever its strategy: with nothing excluded, `Deformable` coincides with it.
        Nx_profile = _separable_profile(di_lim, gx, Nx, periodic_x)
        Ny_profile = _separable_profile(dj_lim, gy, Ny, periodic_y)
        return SeparableFootprint(gx, gy, di_lim, dj_lim, periodic_x, periodic_y, Nx_profile, Ny_profile, nothing, !fully_active, masked_input, row_pass)
    else
        # Deformable: precompute invrenorm = 1/separable_convolve(Float.(mask)) ONCE — the mask never
        # changes across repeated `filter_apply!` calls on a fixed plan.
        maskf = T.(FlowGeometries.Grids.mask(grid))
        denom = zeros(T, Nx, Ny)
        _separable_convolve!(denom, maskf, fp_partial, Nx, Ny)
        invrenorm = similar(denom)
        @. invrenorm = ifelse(denom > T(1e-15), one(T) / denom, zero(T))
        return SeparableFootprint(gx, gy, di_lim, dj_lim, periodic_x, periodic_y, nothing, nothing, invrenorm, !fully_active, masked_input, row_pass)
    end
end

# ZeroFill's mask-independent denominator profile: Σ w over geometrically-valid offsets at each
# index — separable since validity depends only on the index/periodicity, never on the mask.
function _separable_profile(lim::Int, g::AbstractVecOrMat{T}, N::Int, periodic::Bool) where {T<:AbstractFloat}
    profile = zeros(T, N)
    @inbounds for i in 1:N
        s = zero(T)
        for dd in -lim:lim
            ii = i + dd
            valid = (ii >= 1 && ii <= N) || periodic
            valid && (s += _sepw(g, i, dd + lim + 1))
        end
        profile[i] = s
    end
    return profile
end

"""
    apply_separable!(out, field, grid, fp::SeparableFootprint, strategy) -> out

Apply the separable fast path: `masked_input = mask .* field` (the SAME numerator input for
both mask strategies — see the struct docstring), one shared row-pass/column-pass convolution, then
divide by whichever mask-strategy-specific denominator `fp` holds.
"""
function apply_separable!(
    out::AbstractMatrix{T}, field::AbstractMatrix, grid::FlowGeometries.Grids.StructuredGrid, fp::SeparableFootprint{T}, strategy::AbstractMaskStrategy,
) where {T<:AbstractFloat}
    _separable_check_strategy(fp, strategy)
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    mask = FlowGeometries.Grids.mask(grid)
    @. fp.masked_input = T(mask) * field
    _separable_convolve!(out, fp.masked_input, fp, Nx, Ny)
    _separable_normalize_and_mask!(out, fp, mask, Nx, Ny)
    return out
end

@noinline function _separable_strategy_mismatch()
    throw(ArgumentError(
        "SeparableFootprint was built for ZeroFill masking, so it holds only the rank-1 " *
        "profiles and no renormalization field, but is being applied with a renormalizing " *
        "(non-ZeroFill) mask strategy on a masked grid. Rebuild the plan with the same " *
        "`mask_strategy` you intend to apply with.",
    ))
end

# The denominator is fixed at build time — rank-1 profiles for ZeroFill, a dense `invrenorm` for
# Deformable — so the apply cannot honour a strategy the plan was not built for.
#
# `invrenorm === nothing` does not by itself mean ZeroFill: on a fully-active grid both strategies
# coincide and take the profiles. Hence the stored `masked` flag, rather than an `all(mask)` scan.
@inline function _separable_check_strategy(fp::SeparableFootprint, strategy::AbstractMaskStrategy)
    if !(strategy isa ZeroFill) && fp.masked && fp.invrenorm === nothing
        _separable_strategy_mismatch()
    end
    return nothing
end

# Shared denominator-normalize + mask-zero epilogue, used identically by the serial path above and
# the ThreadedBackend extension's parallel-row-pass/column-pass path — kept in one place so they can
# never drift out of sync (mirrors `_separable_convolve!`'s own "single shared primitive" role).
function _separable_normalize_and_mask!(
    out::AbstractMatrix{T}, fp::SeparableFootprint{T}, mask::AbstractMatrix{Bool}, Nx::Int, Ny::Int,
) where {T<:AbstractFloat}
    if fp.invrenorm !== nothing
        @. out *= fp.invrenorm
    else
        Nx_profile, Ny_profile = fp.Nx_profile, fp.Ny_profile
        @inbounds for j in 1:Ny, i in 1:Nx
            denom = Nx_profile[i] * Ny_profile[j]
            out[i, j] = denom > T(1e-15) ? out[i, j] / denom : zero(T)
        end
    end
    @inbounds for j in 1:Ny, i in 1:Nx
        mask[i, j] || (out[i, j] = zero(T))
    end
    return out
end

"""
    build_footprint(grid::StructuredGrid{Cartesian,T,2,TP,<:Tuple{AbstractRange,AbstractRange}}, kernel::GaussianKernel, scale; kwargs...) -> SeparableFootprint

Fast path for a `GaussianKernel` on a uniform (`Range`-axis) Cartesian grid — see the "Separable
Gaussian fast path" section above. More specific than the generic Range-axis method (constrained on
kernel type too), so Julia picks this one whenever `kernel isa GaussianKernel`.
"""
function build_footprint(
    grid::FlowGeometries.Grids.StructuredGrid{G,T,2,TP,<:Tuple{AbstractRange,AbstractRange}},
    kernel::SeparableKernel,
    scale::T;
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}, TP<:NTuple{2,FlowGeometries.Grids.AbstractTopology}}
    return _build_separable_footprint(grid, kernel, scale; kwargs...)
end

"""
    build_footprint(grid::StructuredGrid{Cartesian,T,2}, kernel::GaussianKernel, scale; kwargs...) -> SeparableFootprint

Separability does not require constant spacing: `exp(-α(Δx²+Δy²)/ℓ²)` factorizes on any rectilinear
grid, and a stretched axis only makes the per-axis weight depend on position as well as offset — see
[`_separable_axis_weights`](@ref). So a stretched Cartesian grid gets the same two-pass `O(N·(wx+wy))`
convolution rather than falling to the `O(N·wx·wy)` scattered engine, which for a Gaussian at `w = 20`
is a factor `(2w+1)/2` in operations and a much larger one in per-operation cost.

The Range-axis method above is strictly more specific and resolves what would otherwise be an
ambiguity with the generic Range-axis method; both build the same footprint.
"""
function build_footprint(
    grid::FlowGeometries.Grids.StructuredGrid{G,T,2},
    kernel::SeparableKernel,
    scale::T;
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    return _build_separable_footprint(grid, kernel, scale; kwargs...)
end

"""
    apply_footprint!(out, field, grid, fp, strategy, periodic_x, periodic_y)

Convolve `field` with a precomputed `fp` into `out`, applying the mask `strategy`. `out` and
`field` are 2D (a single layer). The masking branch specializes on the strategy type.
"""
function apply_footprint!(
    out::AbstractMatrix{T},
    field::AbstractMatrix,
    grid::FlowGeometries.Grids.StructuredGrid,
    fp::FilterFootprint{T},
    strategy::AbstractMaskStrategy,
    periodic_x::Bool,
    periodic_y::Bool,
) where {T<:AbstractFloat}
    _, Ny = FlowGeometries.Grids.size_tuple(grid)
    fill!(out, zero(T))
    for j in 1:Ny
        apply_footprint_row!(out, field, grid, fp, strategy, periodic_x, periodic_y, j)
    end
    return out
end

"""
    apply_footprint_row!(out, field, grid, fp, strategy, periodic_x, periodic_y, j)

Fill output row `j` (`out[:, j]`) from a precomputed footprint. Rows are independent (each writes a
disjoint column of the column-major output), so this is the unit of parallelism for the threaded /
distributed backends. Callers must `fill!(out, 0)` first (masked cells are left untouched here).
"""
@inline function _footprint_point(
    field::AbstractMatrix,
    grid::FlowGeometries.Grids.StructuredGrid,
    fp::FilterFootprint{T},
    lo::Int,
    hi::Int,
    strategy::AbstractMaskStrategy,
    periodic_x::Bool,
    periodic_y::Bool,
    Nx::Int,
    Ny::Int,
    i::Integer,
    j::Integer,
) where {T<:AbstractFloat}
    weighted_sum = zero(T)
    weight_norm = zero(T)
    @inbounds for k in lo:hi
        jj = j + fp.dj[k]
        if jj < 1 || jj > Ny
            periodic_y || continue
            jj = mod1(jj, Ny)
        end
        ii = i + fp.di[k]
        if ii < 1 || ii > Nx
            periodic_x || continue
            ii = mod1(ii, Nx)
        end
        active = FlowGeometries.Grids.isactive(grid, ii, jj)
        w = fp.w[k]
        if strategy isa ZeroFill
            # Excluded cells count in the denominator (as zero).
            weight_norm += w
            active && (weighted_sum += w * field[ii, jj])
        else
            # Deformable: masked cells excluded from numerator AND denominator.
            active || continue
            weight_norm += w
            weighted_sum += w * field[ii, jj]
        end
    end
    return weight_norm > T(1e-15) ? weighted_sum / weight_norm : zero(T)
end

# Unmasked: every entry contributes at every column, so the denominator is constant across the row
# interior and both mask strategies coincide. That lets the footprint index move to the outer loop,
# making the inner loop a contiguous axpy along x. Returns the interior columns handled here; the
# remaining edge columns (non-periodic x only) go through the general per-point path.
function _footprint_row_axpy!(
    out::AbstractMatrix{T},
    field::AbstractMatrix,
    grid::FlowGeometries.Grids.StructuredGrid,
    fp::FilterFootprint{T},
    lo::Int,
    hi::Int,
    periodic_x::Bool,
    periodic_y::Bool,
    Nx::Int,
    Ny::Int,
    j::Integer,
) where {T<:AbstractFloat}
    grid.mask isa FlowGeometries.Grids.AllActive || return (1, 0)
    wsum = zero(T)
    dilim = 0
    @inbounds for k in lo:hi
        jj = j + fp.dj[k]
        if (jj < 1 || jj > Ny) && !periodic_y
            continue
        end
        wsum += fp.w[k]
        dilim = max(dilim, abs(fp.di[k]))
    end
    ilo = periodic_x ? 1 : min(dilim + 1, Nx + 1)
    ihi = periodic_x ? Nx : max(Nx - dilim, 0)
    ihi >= ilo || return (1, 0)
    oc = view(out, :, j)
    @inbounds @simd for i in ilo:ihi
        oc[i] = zero(T)
    end
    @inbounds for k in lo:hi
        jj = j + fp.dj[k]
        if jj < 1 || jj > Ny
            periodic_y || continue
            jj = mod1(jj, Ny)
        end
        w = fp.w[k]
        d = fp.di[k]
        fc = view(field, :, jj)
        if periodic_x && abs(d) < Nx
            # Split the wrap into two constant-offset runs so the inner loop stays unit-stride.
            if d >= 0
                @simd for i in 1:(Nx-d)
                    oc[i] += w * fc[i+d]
                end
                @simd for i in (Nx-d+1):Nx
                    oc[i] += w * fc[i+d-Nx]
                end
            else
                @simd for i in 1:(-d)
                    oc[i] += w * fc[i+d+Nx]
                end
                @simd for i in (1-d):Nx
                    oc[i] += w * fc[i+d]
                end
            end
        elseif periodic_x
            @simd for i in ilo:ihi
                oc[i] += w * fc[mod1(i + d, Nx)]
            end
        else
            @simd for i in ilo:ihi
                oc[i] += w * fc[i+d]
            end
        end
    end

    if wsum > T(1e-15)
        @inbounds @simd for i in ilo:ihi
            oc[i] /= wsum # could be hoisted as * f, where f = inv(wsum)
        end
    else
        @inbounds @simd for i in ilo:ihi
            oc[i] = zero(T)
        end
    end
    return (ilo, ihi)
end

function apply_footprint_row!(
    out::AbstractMatrix{T},
    field::AbstractMatrix,
    grid::FlowGeometries.Grids.StructuredGrid,
    fp::FilterFootprint{T},
    strategy::AbstractMaskStrategy,
    periodic_x::Bool,
    periodic_y::Bool,
    j::Integer,
) where {T<:AbstractFloat}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    b = _band(fp, grid, j)
    lo = fp.ptr[b]
    hi = fp.ptr[b+1] - 1
    ilo, ihi = _footprint_row_axpy!(out, field, grid, fp, lo, hi, periodic_x, periodic_y, Nx, Ny, j)
    for i in 1:(ilo-1)
        FlowGeometries.Grids.isactive(grid, i, j) || continue
        out[i, j] = _footprint_point(field, grid, fp, lo, hi, strategy, periodic_x, periodic_y, Nx, Ny, i, j)
    end
    for i in (ihi+1):Nx
        FlowGeometries.Grids.isactive(grid, i, j) || continue
        out[i, j] = _footprint_point(field, grid, fp, lo, hi, strategy, periodic_x, periodic_y, Nx, Ny, i, j)
    end
    return out
end

"""
    apply_footprint!(out, field, grid, fp::ScatteredFilterPlan, strategy, periodic_x, periodic_y)

Whole-grid convolve using a [`ScatteredFilterPlan`](@ref) (the nonuniform-axis/curvilinear fallback).
`periodic_x`/`periodic_y` are accepted only for a uniform call signature with the `FilterFootprint`
method above — periodicity for this footprint kind lives in `fp` itself, not these arguments.
"""
function apply_footprint!(
    out::AbstractMatrix{T},
    field::AbstractMatrix,
    grid::FlowGeometries.Grids.AbstractGrid,
    fp::ScatteredFilterPlan{T},
    strategy::AbstractMaskStrategy,
    periodic_x::Bool,
    periodic_y::Bool,
) where {T<:AbstractFloat}
    _, Ny = FlowGeometries.Grids.size_tuple(grid)
    fill!(out, zero(T))
    for j in 1:Ny
        apply_footprint_row!(out, field, grid, fp, strategy, periodic_x, periodic_y, j)
    end
    return out
end

"""
    apply_footprint_row!(out, field, grid, fp::ScatteredFilterPlan, strategy, periodic_x, periodic_y, j)

Fill output row `j` from a [`ScatteredFilterPlan`](@ref): if `fp.cache !== nothing`, read the
precomputed per-point neighbour list (absolute `(ii,jj)` indices, periodic wrap already resolved at
build time); otherwise recompute each point's neighbours/weights on the fly from `fp`'s compact
scalar metadata. Both branches enumerate candidates through `_scattered_foldl`, so they are
bit-identical by construction rather than by convention. The accumulator is threaded through the
fold's return value rather than captured and mutated, which is what keeps the streaming branch free
of per-iteration allocation (verified by `@allocated` tests) without a second copy of the loop.
"""
function apply_footprint_row!(
    out::AbstractMatrix{T},
    field::AbstractMatrix,
    grid::FlowGeometries.Grids.AbstractGrid,
    fp::ScatteredFilterPlan{T},
    strategy::AbstractMaskStrategy,
    periodic_x::Bool,
    periodic_y::Bool,
    j::Integer,
) where {T<:AbstractFloat}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    cache = fp.cache
    if cache !== nothing
        for i in 1:Nx
            FlowGeometries.Grids.isactive(grid, i, j) || continue
            t = i + (j - 1) * Nx
            lo = cache.ptr[t]
            hi = cache.ptr[t+1] - 1
            weighted_sum = zero(T)
            weight_norm = zero(T)
            @inbounds for k in lo:hi
                ii = cache.ii[k]
                jj = cache.jj[k]
                active = FlowGeometries.Grids.isactive(grid, ii, jj)
                w = cache.w[k]
                if strategy isa ZeroFill
                    weight_norm += w
                    active && (weighted_sum += w * field[ii, jj])
                else
                    active || continue
                    weight_norm += w
                    weighted_sum += w * field[ii, jj]
                end
            end
            out[i, j] = weight_norm > T(1e-15) ? weighted_sum / weight_norm : zero(T)
        end
    else
        kernel = fp.kernel
        scale = fp.scale
        di_lim, dj_lim = fp.di_lim, fp.dj_lim
        fp_periodic_x, fp_periodic_y = fp.periodic_x, fp.periodic_y
        x_period, y_period = fp.x_period, fp.y_period
        is_cartesian = fp.is_cartesian
        rad = fp.rad
        # One candidate buffer for the row, not one per point: an indexed query fills it, and each row
        # is its own task under a row-parallel backend, so nothing is shared across tasks.
        sc = FlowGeometries.Connectivity.ball_scratch()
        for i in 1:Nx
            FlowGeometries.Grids.isactive(grid, i, j) || continue
            target = FlowGeometries.Grids.coords(SA.SVector, grid, i, j)
            weighted_sum, weight_norm = _scattered_foldl(
                (zero(T), zero(T)), grid, target, i, j, Nx, Ny, di_lim, dj_lim,
                fp_periodic_x, fp_periodic_y, x_period, y_period, is_cartesian, rad, fp.topology, sc,
            ) do acc, iin, jjn, d
                ws, wn = acc
                active = FlowGeometries.Grids.isactive(grid, iin, jjn)
                w = Kernels.kernel_weight(kernel, d, scale) * FlowGeometries.Grids.area(grid, iin, jjn)
                if strategy isa ZeroFill
                    return (active ? ws + w * field[iin, jjn] : ws, wn + w)
                else
                    active || return acc
                    return (ws + w * field[iin, jjn], wn + w)
                end
            end
            out[i, j] = weight_norm > T(1e-15) ? weighted_sum / weight_norm : zero(T)
        end
    end
    return out
end

# ---------------------------------------------------------------------------
# General N-dimensional engine (1D + 3D Cartesian); the 2D path uses the per-row engine above.
# ---------------------------------------------------------------------------

# One shared offset set for the whole grid, so it needs both uniform spacing (`AbstractRange` axes) and
# a position-independent metric. A spherical metric is position-dependent — arc length varies with
# latitude, and in 3D with radius — so a spherical 1D/3D grid takes the general path below.
"""
    SeparableFootprintND{N,T}

A separable kernel in `N` dimensions: one weight table per axis, applied as `N` successive 1-D
passes.

This is the same factorization the 2-D path uses, and the reason it matters grows with `N`. A
`FilterFootprintND` enumerates the whole `∏(2wᵈ+1)` box per point; `N` passes cost `∑(2wᵈ+1)`. At
`w = 20` in 3-D that is 68,921 multiply-adds per point against 123.

Weight tables follow [`_sepw`](@ref): a vector where the axis is uniform, an `Nᵈ × (2wᵈ+1)` matrix where
it is stretched.
"""
struct SeparableFootprintND{
    N, T<:AbstractFloat,
    GT<:NTuple{N,AbstractVecOrMat{T}},
    PVT<:Union{Nothing,NTuple{N,AbstractVector{T}}},
    AT<:AbstractArray{T,N},
    IAT<:Union{Nothing,AbstractArray{T,N}},
}
    g::GT
    lim::NTuple{N,Int}
    periodic::NTuple{N,Bool}
    profiles::PVT      # rank-1 ZeroFill denominator, one factor per axis
    invrenorm::IAT     # dense Deformable denominator, or nothing
    masked::Bool
    masked_input::AT
    scratch::AT
end

"""
    _sep_serial(f, indices)

The default pass driver: apply `f` to every index in order. Every output point of a pass is
independent, so a backend can substitute a parallel driver of the same shape and get a bit-identical
result — only the barrier BETWEEN passes is required.
"""
@inline function _sep_serial(f::F, indices) where {F}
    for I in indices
        f(I)
    end
    return nothing
end

@inline function _separable_pass!(
    dst::AbstractArray{T,N}, src::AbstractArray{T,N}, g::AbstractVecOrMat{T},
    lim::Int, periodic::Bool, dims::NTuple{N,Int}, ::Val{d}, driver::D,
) where {T<:AbstractFloat, N, d, D}
    n = dims[d]
    # Driven over COLUMNS, not points: taps move to the outer loop so the inner loop always walks
    # dimension 1 with unit stride, whichever axis the pass is along. A per-point form cannot do this —
    # for `d ≥ 2` it strides by `∏dims[1:d-1]` on every tap. Each column is written by exactly one task,
    # so this is as race-free as a per-point form and every backend gets the same answer.
    #
    # Rank 1 needs no separate branch: `Base.tail((n,))` is `()` and `CartesianIndices(())` holds one
    # 0-dimensional index, so the column pass runs once over the whole vector.
    driver(CartesianIndices(Base.tail(dims))) do J
        if d == 1
            _separable_col_pass!(dst, src, g, lim, periodic, dims[1], Tuple(J))
        else
            _separable_slab_pass!(dst, src, g, lim, periodic, n, dims[1], Tuple(J), Val(d))
        end
    end
    return dst
end

# One line along dimension `d > 1`. The weight is indexed by the OUTPUT position along `d`, which is
# fixed for this line, so it hoists out of the inner loop entirely and each tap is a plain axpy.
@inline function _separable_slab_pass!(
    dst::AbstractArray{T}, src::AbstractArray{T}, g::AbstractVecOrMat{T},
    lim::Int, periodic::Bool, n::Int, n1::Int, J::Tuple, ::Val{d},
) where {T<:AbstractFloat, d}
    jd = J[d - 1]
    dv = view(dst, :, J...)
    @inbounds begin
        for i in 1:n1
            dv[i] = zero(T)
        end
        for dd in (-lim):lim
            jj = jd + dd
            if jj < 1 || jj > n
                periodic || continue
                jj = mod1(jj, n)
            end
            wt = _sepw(g, jd, dd + lim + 1)
            sv = view(src, :, Base.setindex(J, jj, d - 1)...)
            @simd for i in 1:n1
                dv[i] += wt * sv[i]
            end
        end
    end
    return nothing
end

# One line along dimension 1, tap-outer. `dv`/`sv` are contiguous views, so each tap is an axpy.
@inline function _separable_col_pass!(
    dst::AbstractArray{T}, src::AbstractArray{T}, g::AbstractVecOrMat{T},
    lim::Int, periodic::Bool, n1::Int, J::Tuple,
) where {T<:AbstractFloat}
    dv = view(dst, :, J...)
    sv = view(src, :, J...)
    @inbounds begin
        for i in 1:n1
            dv[i] = zero(T)
        end
        for dd in (-lim):lim
            k = dd + lim + 1
            if periodic && abs(dd) < n1
                if dd >= 0
                    @simd for i in 1:(n1 - dd)
                        dv[i] += _sepw(g, i, k) * sv[i + dd]
                    end
                    @simd for i in (n1 - dd + 1):n1
                        dv[i] += _sepw(g, i, k) * sv[i + dd - n1]
                    end
                else
                    @simd for i in 1:(-dd)
                        dv[i] += _sepw(g, i, k) * sv[i + dd + n1]
                    end
                    @simd for i in (-dd + 1):n1
                        dv[i] += _sepw(g, i, k) * sv[i + dd]
                    end
                end
            elseif periodic
                @simd for i in 1:n1
                    dv[i] += _sepw(g, i, k) * sv[mod1(i + dd, n1)]
                end
            else
                @simd for i in max(1, 1 - dd):min(n1, n1 - dd)
                    dv[i] += _sepw(g, i, k) * sv[i + dd]
                end
            end
        end
    end
    return nothing
end

# The passes, unrolled by recursion on `Val(d)`. Buffers alternate and the last pass writes `dst`, so
# nothing aliases: pass 1 reads the caller's masked input and writes `scratch`, and every later pass
# reads what its predecessor just wrote. `masked_input` is free to be reused from pass 2 on.
# The two alternating intermediates are passed IN rather than taken from the footprint, because a batched
# apply needs batch-sized ones and the footprint's are sized for a single slice. `dims` is the driven
# array's shape, so trailing batch axes ride along; `Val(N)` is the number of PASSES, which stays at the
# grid's rank so no pass ever differences along a batch axis.
@inline _sep_pass_chain!(dst, src, fp, dims, ::Val{N}, ::Val{N}, driver::D, b1, b2) where {N,D} =
    _separable_pass!(dst, src, fp.g[N], fp.lim[N], fp.periodic[N], dims, Val(N), driver)

@inline function _sep_pass_chain!(dst, src, fp, dims, ::Val{N}, ::Val{d}, driver::D, b1, b2) where {N,d,D}
    buf = isodd(d) ? b1 : b2
    _separable_pass!(buf, src, fp.g[d], fp.lim[d], fp.periodic[d], dims, Val(d), driver)
    return _sep_pass_chain!(dst, buf, fp, dims, Val(N), Val(d + 1), driver, b1, b2)
end

# Unbatched callers keep the footprint's own buffers.
@inline _sep_pass_chain!(dst, src, fp, dims, vn::Val, vd::Val, driver::D = _sep_serial) where {D} =
    _sep_pass_chain!(dst, src, fp, dims, vn, vd, driver, fp.scratch, fp.masked_input)

function _build_separable_nd(
    grid::FlowGeometries.Grids.StructuredGrid{G,T,N},
    kernel::SeparableKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    kwargs...,
) where {T<:AbstractFloat, N, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    dims = FlowGeometries.Grids.size_tuple(grid)
    rad = Kernels.kernel_radius(kernel, scale)
    periodic = FlowGeometries.Grids.periodic_flags(grid)
    lim = ntuple(Val(N)) do d
        s = FlowGeometries.Grids.minimum_spacing(grid, d)
        (isfinite(s) && s > 0) ? ceil(Int, rad / s) : 0
    end
    mf = FlowGeometries.Grids.measure_factors(grid)
    mf === nothing && throw(ArgumentError(
        "the separable path needs a separable cell measure, but this grid's measure is dense",
    ))
    g = ntuple(Val(N)) do d
        _separable_axis_weights(
            FlowGeometries.Grids.coordinates(grid, d), lim[d], periodic[d],
            T(FlowGeometries.Grids.period(grid, d)), kernel, scale, convert(AbstractVector{T}, mf[d]),
        )
    end

    masked_input = zeros(T, dims)
    scratch = zeros(T, dims)
    fully_active = all(FlowGeometries.Grids.mask(grid))
    fp_partial = SeparableFootprintND(
        g, lim, periodic, nothing, nothing, !fully_active, masked_input, scratch,
    )
    if fully_active || mask_strategy isa ZeroFill
        # Mask-independent denominator: `Σ w` over geometrically valid offsets, which depends only on
        # the index and the wrapping, so it stays a product of one factor per axis.
        profiles = ntuple(d -> _separable_profile(lim[d], g[d], dims[d], periodic[d]), Val(N))
        return SeparableFootprintND(
            g, lim, periodic, profiles, nothing, !fully_active, masked_input, scratch,
        )
    end
    maskf = T.(FlowGeometries.Grids.mask(grid))
    denom = zeros(T, dims)
    copyto!(masked_input, maskf)
    _sep_pass_chain!(denom, masked_input, fp_partial, dims, Val(N), Val(1))
    invrenorm = similar(denom)
    @. invrenorm = ifelse(denom > T(1e-15), one(T) / denom, zero(T))
    return SeparableFootprintND(
        g, lim, periodic, nothing, invrenorm, !fully_active, masked_input, scratch,
    )
end

@inline function _separable_check_strategy(
    fp::SeparableFootprintND, strategy::AbstractMaskStrategy,
)
    if !(strategy isa ZeroFill) && fp.masked && fp.invrenorm === nothing
        _separable_strategy_mismatch()
    end
    return nothing
end

"""
    apply_separable_nd!(out, field, grid, fp, strategy, driver = _sep_serial)

Run the `N` separable passes and the pointwise normalization. `driver` supplies the per-pass index
sweep — see [`_sep_serial`](@ref); a threaded backend passes its own and gets the same answer, since
every point within a pass is independent and the passes themselves stay ordered.
"""
#
# The array rank is free while the footprint stays at the grid's rank `R`, so `out`/`field` may carry
# trailing batch axes. Everything below is driven over the ARRAY's shape, which is what folds a batch into
# the driven index space — one pass over the whole batch instead of one per slice, and on a device one
# launch instead of `Nb`. The pass count stays `R`, so no pass differences along a batch axis.
#
# The profile tables, the renormalization array and the mask are all spatial-only, so they are indexed with
# the leading `R` components of the driven index rather than the index itself.
function apply_separable_nd!(
    out::AbstractArray{T}, field::AbstractArray, grid::FlowGeometries.Grids.StructuredGrid,
    fp::SeparableFootprintND{R,T}, strategy::AbstractMaskStrategy, driver::D = _sep_serial,
) where {T<:AbstractFloat, R, D}
    _separable_check_strategy(fp, strategy)
    dims = size(out)
    mask = FlowGeometries.Grids.mask(grid)
    b1, b2 = _sep_nd_buffers(fp, out, Val(R))
    @. b2 = T(mask) * field
    _sep_pass_chain!(out, b2, fp, dims, Val(R), Val(1), driver, b1, b2)
    prof = fp.profiles
    inv = fp.invrenorm
    driver(CartesianIndices(dims)) do I
        @inbounds begin
            Is = CartesianIndex(ntuple(d -> I[d], Val(R)))
            if inv === nothing
                den = prod(ntuple(d -> prof[d][I[d]], Val(R)))
                out[I] = den > T(1e-15) ? out[I] / den : zero(T)
            else
                out[I] *= inv[Is]
            end
            mask[Is] || (out[I] = zero(T))
        end
    end
    return out
end

# The footprint's own pass buffers when the apply is unbatched; batch-sized ones otherwise, since the
# chain's intermediates must match the driven shape.
@inline function _sep_nd_buffers(fp, out::AbstractArray{T}, ::Val{R}) where {T,R}
    ndims(out) == R && return (fp.scratch, fp.masked_input)
    return (similar(out), similar(out))
end

# A Gaussian on a 1-D or true-3-D Cartesian grid is separable exactly as it is in 2-D, so it takes the
# `N`-pass engine rather than `FilterFootprintND`'s full-box enumeration. More specific than the
# generic-kernel methods below (constrained on the kernel too), and unconstrained in the axis types
# because separability does not require uniform spacing — see [`_separable_axis_weights`](@ref).
build_footprint(
    grid::FlowGeometries.Grids.StructuredGrid{G,T,1},
    kernel::SeparableKernel, scale::T; kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}} =
    _build_separable_nd(grid, kernel, scale; kwargs...)

build_footprint(
    grid::FlowGeometries.Grids.StructuredGrid{G,T,3},
    kernel::SeparableKernel, scale::T; kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}} =
    _build_separable_nd(grid, kernel, scale; kwargs...)

# Range axes AND a Gaussian is more specific than either of the two methods that would otherwise both
# apply (kernel-specific with free axes, axis-specific with a free kernel), so these resolve that pair.
# They route to the separable engine as well: uniform spacing makes the weight tables vectors instead
# of matrices, not a different algorithm.
build_footprint(
    grid::FlowGeometries.Grids.StructuredGrid{G,T,1,TP,<:Tuple{AbstractRange}},
    kernel::SeparableKernel, scale::T; kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}, TP<:NTuple{1,FlowGeometries.Grids.AbstractTopology}} =
    _build_separable_nd(grid, kernel, scale; kwargs...)

build_footprint(
    grid::FlowGeometries.Grids.StructuredGrid{G,T,3,TP,<:Tuple{AbstractRange,AbstractRange,AbstractRange}},
    kernel::SeparableKernel, scale::T; kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}, TP<:NTuple{3,FlowGeometries.Grids.AbstractTopology}} =
    _build_separable_nd(grid, kernel, scale; kwargs...)

build_footprint(
    grid::FlowGeometries.Grids.StructuredGrid{G,T,1,TP,<:Tuple{AbstractRange}},
    kernel::Kernels.AbstractFilterKernel, scale::T; kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}, TP<:NTuple{1,FlowGeometries.Grids.AbstractTopology}} = _build_footprint_nd(grid, kernel, scale)
build_footprint(
    grid::FlowGeometries.Grids.StructuredGrid{G,T,3,TP,<:Tuple{AbstractRange,AbstractRange,AbstractRange}},
    kernel::Kernels.AbstractFilterKernel, scale::T; kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}, TP<:NTuple{3,FlowGeometries.Grids.AbstractTopology}} = _build_footprint_nd(grid, kernel, scale)

# General path: at least one axis is a plain AbstractVector (no uniformity guarantee), OR the
# geometry is non-Cartesian (no translation-invariant fast path exists, see above) — less specific
# than the two Cartesian-only methods above, reached whenever they don't match.
build_footprint(grid::FlowGeometries.Grids.StructuredGrid{G,T,1}, kernel::Kernels.AbstractFilterKernel, scale::T; kwargs...) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}} =
    _build_footprint_nd_scattered(grid, kernel, scale; kwargs...)
build_footprint(grid::FlowGeometries.Grids.StructuredGrid{G,T,3}, kernel::Kernels.AbstractFilterKernel, scale::T; kwargs...) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}} =
    _build_footprint_nd_scattered(grid, kernel, scale; kwargs...)

function _build_footprint_nd(
    grid::FlowGeometries.Grids.StructuredGrid{G,T,N},
    kernel::Kernels.AbstractFilterKernel,
    scale::T,
) where {N, T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    rad = Kernels.kernel_radius(kernel, scale)
    # Real per-axis step, read from the axis itself (already proven uniform by its Range type via
    # the calling method's dispatch constraint) — not the geometry's separately-stored dx/dy/dz,
    # so there's no possibility of the two disagreeing.
    spacing = ntuple(d -> step(FlowGeometries.Grids.coordinates(grid, d)), N)
    A = FlowGeometries.Grids.measure(grid)[ntuple(_ -> 1, N)...]   # uniform Cartesian cell measure
    lim = ntuple(d -> spacing[d] > 0 ? ceil(Int, rad / spacing[d]) : 0, N)
    offsets = NTuple{N,Int}[]
    w = T[]
    # Exact window size (single shared translation-invariant footprint): every candidate offset in
    # this hyperrectangle is visited exactly once below.
    sizehint!(offsets, prod(2 .* lim .+ 1))
    sizehint!(w, prod(2 .* lim .+ 1))
    for off in CartesianIndices(ntuple(d -> (-lim[d]):lim[d], N))
        o = Tuple(off)
        d2 = zero(T)
        for d in 1:N
            d2 += (T(o[d]) * spacing[d])^2
        end
        dist = sqrt(d2)
        if dist <= rad
            push!(offsets, o)
            push!(w, Kernels.kernel_weight(kernel, dist, scale) * A)
        end
    end
    return FilterFootprintND(offsets, w)
end

"""
    NDScatteredCache{N, T}

The full per-target-point neighbour list for an [`NDScatteredFilterPlan`](@ref) (absolute neighbour
multi-indices + weights), built only when the plan's [`AbstractCacheStrategy`](@ref) calls for it.
"""
struct NDScatteredCache{N, T<:AbstractFloat, VO<:AbstractVector{NTuple{N,Int}}, VT<:AbstractVector{T}, VI<:AbstractVector{Int}}
    nbrs::VO
    w::VT
    ptr::VI   # target t = LinearIndices(dims)[I]; entries ptr[t]:ptr[t+1]-1
end

"""
    NDScatteredFilterPlan{N, T, K}

N-D (1D or 3D) analog of [`ScatteredFilterPlan`](@ref): compact scalar metadata (per-axis window
limits, periodicity/period, geometry flag) for when at least one of the N axes is a plain
`AbstractVector` (no type-level uniformity proof) — no translation-invariance assumption, correct
for any spacing pattern. `cache` holds the materialized [`NDScatteredCache`](@ref) only when the
plan's cache strategy decided to build it, `nothing` otherwise (apply-time recomputation).
"""
struct NDScatteredFilterPlan{N, T<:AbstractFloat, K<:Kernels.AbstractFilterKernel, C<:Union{Nothing,NDScatteredCache{N,T}}}
    kernel::K
    scale::T
    rad::T
    lim::NTuple{N,Int}
    periodic::NTuple{N,Bool}
    period::NTuple{N,T}
    is_cartesian::Bool
    cache::C
end

@inline _nd_scattered_cache_bytes(dims::NTuple{N,Int}, lim::NTuple{N,Int}, ::Type{T}) where {N,T} =
    prod(dims) * prod(2 .* lim .+ 1) * (sizeof(NTuple{N,Int}) + sizeof(T))

"""
    _nd_scattered_window_bounds(grid, rad) -> (lim, periodic, period, is_cartesian)

Compact per-axis scalar window-bound derivation for the nonuniform N-D (1D/3D) case — the same values
an [`NDScatteredFilterPlan`](@ref) stores. The window itself is `Connectivity.metric_window`, read from
the grid's cached axis statistics in O(1); nothing here scans the grid.
"""
function _nd_scattered_window_bounds(
    grid::FlowGeometries.Grids.StructuredGrid{G,T,N}, rad::T,
) where {N, T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    periodic = FlowGeometries.Grids.periodic_flags(grid)
    # Conservative per-axis index-radius bound, valid at every point of the grid; the exact `d <= rad`
    # check below still gates inclusion. `metric_window` reads it from the grid's cached axis statistics
    # in O(1) — converting a physical radius through the local metric (r·cosφ for longitude, r for
    # latitude, the radial axis already being distance) is the geometry's own arithmetic.
    lim = FlowGeometries.Connectivity.metric_window(grid, rad)

    # A wrapped candidate's raw stored coordinate sits a full period away from the target on a
    # periodic CARTESIAN axis, so plain Euclidean `distance` would reject every genuinely-close
    # wrapped neighbor unless shifted back by one period first (a periodic spherical x axis needs
    # no such shift — see `_build_footprint_scattered`'s identical point for the 2D case).
    is_cartesian = G <: FlowGeometries.Geometry.CartesianGeometry{T}
    period = ntuple(N) do d
        (is_cartesian && periodic[d]) ? T(FlowGeometries.Grids.period(grid, d)) : zero(T)
    end
    return lim, periodic, period, is_cartesian
end

function _build_footprint_nd_scattered(
    grid::FlowGeometries.Grids.StructuredGrid{G,T,N},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    cache_strategy::AbstractCacheStrategy = AutoCache(),
    cache_byte_budget::Integer = DEFAULT_CACHE_BYTE_BUDGET,
    kwargs...,   # accepts (and ignores) mask_strategy — only the 2D separable-Gaussian path needs it
) where {N, T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    dims = FlowGeometries.Grids.size_tuple(grid)
    rad = Kernels.kernel_radius(kernel, scale)
    lim, periodic, period, is_cartesian = _nd_scattered_window_bounds(grid, rad)

    cache = if _should_cache(cache_strategy, _nd_scattered_cache_bytes(dims, lim, T), cache_byte_budget)
        nbrs = NTuple{N,Int}[]
        w = T[]
        sizehint!(nbrs, prod(dims) * prod(2 .* lim .+ 1))
        sizehint!(w, prod(dims) * prod(2 .* lim .+ 1))
        lin = LinearIndices(dims)
        ptr = Vector{Int}(undef, prod(dims) + 1)
        ptr[1] = 1
        for I in CartesianIndices(dims)
            t = lin[I]
            _nd_foldl(nothing, grid, Tuple(I), dims, lim, periodic, period, is_cartesian, rad) do _, J, d
                push!(nbrs, J)
                push!(w, Kernels.kernel_weight(kernel, d, scale) * FlowGeometries.Grids.area(grid, J...))
                nothing
            end
            ptr[t+1] = length(nbrs) + 1
        end
        NDScatteredCache(nbrs, w, ptr)
    else
        nothing
    end
    return NDScatteredFilterPlan(kernel, scale, rad, lim, periodic, period, is_cartesian, cache)
end

# Folds `acc = f(acc, J, d)` over every in-support neighbour of `Ti`, the centre included. The ND
# counterpart of `_scattered_foldl`: the cache builder and the streaming apply both enumerate through
# it, so they agree by construction rather than by two copies being kept in step.
@inline function _nd_foldl(
    f::F, acc, grid::FlowGeometries.Grids.StructuredGrid{G,T,N}, Ti::NTuple{N,Int},
    dims::NTuple{N,Int}, lim::NTuple{N,Int}, periodic::NTuple{N,Bool}, period::NTuple{N,T},
    is_cartesian::Bool, rad::T,
) where {F, N, T<:AbstractFloat, G}
    geo = FlowGeometries.Grids.grid_geometry(grid)
    target = FlowGeometries.Grids.coords(SA.SVector, grid, Ti...)
    for off in CartesianIndices(ntuple(d -> (-lim[d]):lim[d], N))
        J = ntuple(N) do d
            jj = Ti[d] + off[d]
            (jj < 1 || jj > dims[d]) ? (periodic[d] ? mod1(jj, dims[d]) : 0) : jj
        end
        any(==(0), J) && continue
        shift = ntuple(N) do d
            jj = Ti[d] + off[d]
            jj < 1 ? -period[d] : (jj > dims[d] ? period[d] : zero(T))
        end
        neighbor = FlowGeometries.Grids.coords(SA.SVector, grid, J...)
        neighbor_shifted = is_cartesian ? (neighbor + SA.SVector{N,T}(shift)) : neighbor
        d = FlowGeometries.Geometry.distance(geo, target, neighbor_shifted)
        d <= rad || continue
        acc = f(acc, J, d)
    end
    return acc
end

# Shifted neighbour multi-index with per-axis periodic wrap; returns (index, in-bounds?).
@inline function _shift_index(I::NTuple{N,Int}, o::NTuple{N,Int}, dims::NTuple{N,Int}, periodic::NTuple{N,Bool}) where {N}
    J = ntuple(N) do d
        jj = I[d] + o[d]
        (jj < 1 || jj > dims[d]) ? (periodic[d] ? mod1(jj, dims[d]) : 0) : jj
    end
    return J, !any(==(0), J)
end

# Per-point kernel factored out of `apply_footprint_nd!` so a parallel (per-point-independent) loop
# can reuse the EXACT same arithmetic instead of duplicating it — see
# `CoarseGrainingEnergyFluxesOhMyThreadsExt`'s ND threaded hook.
@inline function _footprint_nd_point(
    field::AbstractArray, fp::FilterFootprintND{N,T}, strategy::AbstractMaskStrategy,
    dims::NTuple{N,Int}, periodic::NTuple{N,Bool}, mask, I::CartesianIndex{N},
) where {N, T<:AbstractFloat}
    Ti = Tuple(I)
    ws = zero(T)
    wn = zero(T)
    @inbounds for k in eachindex(fp.offsets)
        J, valid = _shift_index(Ti, fp.offsets[k], dims, periodic)
        valid || continue
        active = mask[J...]
        wk = fp.w[k]
        if strategy isa ZeroFill
            wn += wk
            active && (ws += wk * field[J...])
        elseif active
            wn += wk
            ws += wk * field[J...]
        end
    end
    return wn > T(1e-15) ? ws / wn : zero(T)
end

function apply_footprint_nd!(
    out::AbstractArray{T,N},
    field::AbstractArray,
    grid::FlowGeometries.Grids.StructuredGrid{G,T,N},
    fp::FilterFootprintND{N,T},
    strategy::AbstractMaskStrategy,
) where {N, T<:AbstractFloat, G}
    dims = FlowGeometries.Grids.size_tuple(grid)
    periodic = FlowGeometries.Grids.periodic_flags(grid)
    mask = FlowGeometries.Grids.mask(grid)
    fill!(out, zero(T))
    @inbounds for I in CartesianIndices(out)
        mask[I] || continue
        out[I] = _footprint_nd_point(field, fp, strategy, dims, periodic, mask, I)
    end
    return out
end

@inline function _footprint_nd_point_cached(
    field::AbstractArray, cache::NDScatteredCache{N,T}, strategy::AbstractMaskStrategy,
    mask, lin::LinearIndices{N}, I::CartesianIndex{N},
) where {N, T<:AbstractFloat}
    t = lin[I]
    lo = cache.ptr[t]
    hi = cache.ptr[t+1] - 1
    ws = zero(T)
    wn = zero(T)
    @inbounds for k in lo:hi
        J = cache.nbrs[k]
        active = mask[J...]
        wk = cache.w[k]
        if strategy isa ZeroFill
            wn += wk
            active && (ws += wk * field[J...])
        elseif active
            wn += wk
            ws += wk * field[J...]
        end
    end
    return wn > T(1e-15) ? ws / wn : zero(T)
end

# Streaming (no cache) per-point recompute, over the same enumeration the cache builder uses. The
# accumulator is threaded through the fold's return value rather than captured and mutated, which is
# what keeps this allocation-free.
function _footprint_nd_point_streaming(
    field::AbstractArray, grid::FlowGeometries.Grids.StructuredGrid{G,T,N}, fp::NDScatteredFilterPlan{N,T},
    strategy::AbstractMaskStrategy, mask, dims::NTuple{N,Int}, I::CartesianIndex{N},
) where {N, T<:AbstractFloat, G}
    kernel, scale = fp.kernel, fp.scale
    ws, wn = _nd_foldl(
        (zero(T), zero(T)), grid, Tuple(I), dims, fp.lim, fp.periodic, fp.period, fp.is_cartesian, fp.rad,
    ) do acc, J, d
        s, n = acc
        active = mask[J...]
        wk = Kernels.kernel_weight(kernel, d, scale) * FlowGeometries.Grids.area(grid, J...)
        if strategy isa ZeroFill
            return (active ? s + wk * field[J...] : s, n + wk)
        else
            active || return acc
            return (s + wk * field[J...], n + wk)
        end
    end
    return wn > T(1e-15) ? ws / wn : zero(T)
end

function apply_footprint_nd!(
    out::AbstractArray{T,N},
    field::AbstractArray,
    grid::FlowGeometries.Grids.StructuredGrid{G,T,N},
    fp::NDScatteredFilterPlan{N,T},
    strategy::AbstractMaskStrategy,
) where {N, T<:AbstractFloat, G}
    dims = FlowGeometries.Grids.size_tuple(grid)
    mask = FlowGeometries.Grids.mask(grid)
    fill!(out, zero(T))
    if fp.cache !== nothing
        lin = LinearIndices(dims)
        cache = fp.cache
        @inbounds for I in CartesianIndices(out)
            mask[I] || continue
            out[I] = _footprint_nd_point_cached(field, cache, strategy, mask, lin, I)
        end
    else
        @inbounds for I in CartesianIndices(out)
            mask[I] || continue
            out[I] = _footprint_nd_point_streaming(field, grid, fp, strategy, mask, dims, I)
        end
    end
    return out
end

# Dispatch the apply on the footprint kind.
"""
    NodeFilterPlan{T}

Real-space filter footprint for a node set: per-node CSR neighbour blocks with their geometric weights
`w = kernel_weight(d) · control volume`.

A node set has no axes, so there is no index window to bound a search with and the neighbourhood
cannot be re-derived per apply the way a structured grid's can. It is found once, at plan time, and
stored — which also means the search is paid once per plan rather than once per field, and a single
`compute_Π!` makes six to nine applies against one plan.

The node itself is included. `Connectivity.neighbors_within` excludes it, matching stencil semantics
where a cell is not its own neighbour, but a filter's zero offset is a genuine contribution.
"""
struct NodeFilterPlan{T<:AbstractFloat, VI<:AbstractVector{Int}, VT<:AbstractVector{T}}
    nbrs::VI
    w::VT
    ptr::VI
end

_apply_serial!(out, field, grid, fp::FilterFootprint, strategy) =
    apply_footprint!(out, field, grid, fp, strategy, FlowGeometries.Grids.isperiodic(grid, 1), FlowGeometries.Grids.isperiodic(grid, 2))
_apply_serial!(out, field, grid, fp::ScatteredFilterPlan, strategy) =
    apply_footprint!(out, field, grid, fp, strategy, FlowGeometries.Grids.isperiodic(grid, 1), FlowGeometries.Grids.isperiodic(grid, 2))
_apply_serial!(out, field, grid, fp::FilterFootprintND, strategy) =
    apply_footprint_nd!(out, field, grid, fp, strategy)
_apply_serial!(out, field, grid, fp::NDScatteredFilterPlan, strategy) =
    apply_footprint_nd!(out, field, grid, fp, strategy)
_apply_serial!(out, field, grid, fp::PrefixSumTopHatPlan, strategy) =
    apply_prefixsum_tophat!(out, field, grid, fp, strategy)
_apply_serial!(out, field, grid, fp::SeparableFootprintND, strategy) =
    apply_separable_nd!(out, field, grid, fp, strategy)
_apply_serial!(out, field, grid, fp::NodeFilterPlan, strategy) =
    apply_footprint!(out, field, grid, fp, strategy)
_apply_serial!(out, field, grid, fp::SeparableFootprint, strategy) =
    apply_separable!(out, field, grid, fp, strategy)

# ---------------------------------------------------------------------------
# Batched apply: derive each target point's neighbours and weights once and apply them to every field
# in the batch. `compute_Π!` makes 6-9 `filter_apply!` calls per scale against one plan, so this is
# where the redundancy is. Orthogonal to caching, which changes only how the weights are derived.
# ---------------------------------------------------------------------------

# Homogeneous-`NTuple` batches (compile-time-known K): `MVector` accumulators are stack-allocated,
# not heap — genuinely zero-allocation. `Vector` batches (runtime-known K, e.g. a variable number of
# quadratic-product terms): a small `Vector{T}` accumulator, allocated ONCE per row/point-loop (not
# per candidate or per target point), so its cost is O(Ny) or O(N) total, not part of the O(N·M) hot
# path. Both share `eachindex`/`fill!`/indexing, so the rest of the per-row/per-point logic below is
# written once, generic over which container `outs`/`fields` actually is.
@inline _batch_zeros(::NTuple{K,<:Any}, ::Type{T}) where {K,T<:AbstractFloat} = SA.MVector{K,T}(ntuple(_ -> zero(T), K))
@inline _batch_zeros(v::AbstractVector, ::Type{T}) where {T<:AbstractFloat} = zeros(T, length(v))

_apply_serial_batch!(outs, fields, grid, fp::FilterFootprint, strategy) =
    apply_footprint_batch!(outs, fields, grid, fp, strategy, FlowGeometries.Grids.isperiodic(grid, 1), FlowGeometries.Grids.isperiodic(grid, 2))
_apply_serial_batch!(outs, fields, grid, fp::ScatteredFilterPlan, strategy) =
    apply_footprint_batch!(outs, fields, grid, fp, strategy, FlowGeometries.Grids.isperiodic(grid, 1), FlowGeometries.Grids.isperiodic(grid, 2))
_apply_serial_batch!(outs, fields, grid, fp::FilterFootprintND, strategy) =
    apply_footprint_nd_batch!(outs, fields, grid, fp, strategy)
_apply_serial_batch!(outs, fields, grid, fp::NDScatteredFilterPlan, strategy) =
    apply_footprint_nd_batch!(outs, fields, grid, fp, strategy)
# The remaining footprints carry no per-point neighbour derivation for a batch to share: the separable
# Gaussians hold precomputed 1-D weight tables, `PrefixSumTopHatPlan`'s support intervals come from a
# two-pointer sweep that is already O(1) amortized, and `NodeFilterPlan` stores its adjacency outright.
# The only per-field work left really is per-field data, so a batch here IS the per-field loop.
function _apply_serial_batch!(
    outs, fields, grid,
    fp::Union{SeparableFootprint, SeparableFootprintND, PrefixSumTopHatPlan, NodeFilterPlan},
    strategy,
)
    for k in eachindex(outs)
        _apply_serial!(outs[k], fields[k], grid, fp, strategy)
    end
    return outs
end

function apply_footprint_batch!(
    outs, fields, grid::FlowGeometries.Grids.AbstractGrid, fp::FilterFootprint{T}, strategy::AbstractMaskStrategy,
    periodic_x::Bool, periodic_y::Bool,
) where {T<:AbstractFloat}
    _, Ny = FlowGeometries.Grids.size_tuple(grid)
    for out in outs
        fill!(out, zero(T))
    end
    for j in 1:Ny
        apply_footprint_row_batch!(outs, fields, grid, fp, strategy, periodic_x, periodic_y, j)
    end
    return outs
end

function apply_footprint_row_batch!(
    outs, fields, grid::FlowGeometries.Grids.StructuredGrid, fp::FilterFootprint{T}, strategy::AbstractMaskStrategy,
    periodic_x::Bool, periodic_y::Bool, j::Integer,
) where {T<:AbstractFloat}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    b = _band(fp, grid, j)
    lo = fp.ptr[b]
    hi = fp.ptr[b+1] - 1
    acc_ws = _batch_zeros(outs, T)
    acc_wn = _batch_zeros(outs, T)
    for i in 1:Nx
        FlowGeometries.Grids.isactive(grid, i, j) || continue
        fill!(acc_ws, zero(T))
        fill!(acc_wn, zero(T))
        @inbounds for k in lo:hi
            jj = j + fp.dj[k]
            if jj < 1 || jj > Ny
                periodic_y || continue
                jj = mod1(jj, Ny)
            end
            ii = i + fp.di[k]
            if ii < 1 || ii > Nx
                periodic_x || continue
                ii = mod1(ii, Nx)
            end
            active = FlowGeometries.Grids.isactive(grid, ii, jj)
            w = fp.w[k]
            if strategy isa ZeroFill
                for m in eachindex(fields)
                    acc_wn[m] += w
                end
                if active
                    for m in eachindex(fields)
                        acc_ws[m] += w * fields[m][ii, jj]
                    end
                end
            elseif active
                for m in eachindex(fields)
                    acc_wn[m] += w
                    acc_ws[m] += w * fields[m][ii, jj]
                end
            end
        end
        for m in eachindex(outs)
            outs[m][i, j] = acc_wn[m] > T(1e-15) ? acc_ws[m] / acc_wn[m] : zero(T)
        end
    end
    return outs
end

function apply_footprint_batch!(
    outs, fields, grid::FlowGeometries.Grids.AbstractGrid, fp::ScatteredFilterPlan{T}, strategy::AbstractMaskStrategy,
    periodic_x::Bool, periodic_y::Bool,
) where {T<:AbstractFloat}
    _, Ny = FlowGeometries.Grids.size_tuple(grid)
    for out in outs
        fill!(out, zero(T))
    end
    for j in 1:Ny
        apply_footprint_row_batch!(outs, fields, grid, fp, strategy, periodic_x, periodic_y, j)
    end
    return outs
end

function apply_footprint_row_batch!(
    outs, fields, grid::FlowGeometries.Grids.AbstractGrid, fp::ScatteredFilterPlan{T}, strategy::AbstractMaskStrategy,
    periodic_x::Bool, periodic_y::Bool, j::Integer,
) where {T<:AbstractFloat}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    acc_ws = _batch_zeros(outs, T)
    acc_wn = _batch_zeros(outs, T)
    cache = fp.cache
    if cache !== nothing
        for i in 1:Nx
            FlowGeometries.Grids.isactive(grid, i, j) || continue
            t = i + (j - 1) * Nx
            lo = cache.ptr[t]
            hi = cache.ptr[t+1] - 1
            fill!(acc_ws, zero(T))
            fill!(acc_wn, zero(T))
            @inbounds for k in lo:hi
                ii = cache.ii[k]
                jj = cache.jj[k]
                active = FlowGeometries.Grids.isactive(grid, ii, jj)
                w = cache.w[k]
                if strategy isa ZeroFill
                    for m in eachindex(fields)
                        acc_wn[m] += w
                    end
                    if active
                        for m in eachindex(fields)
                            acc_ws[m] += w * fields[m][ii, jj]
                        end
                    end
                elseif active
                    for m in eachindex(fields)
                        acc_wn[m] += w
                        acc_ws[m] += w * fields[m][ii, jj]
                    end
                end
            end
            for m in eachindex(outs)
                outs[m][i, j] = acc_wn[m] > T(1e-15) ? acc_ws[m] / acc_wn[m] : zero(T)
            end
        end
    else
        kernel = fp.kernel
        scale = fp.scale
        di_lim, dj_lim = fp.di_lim, fp.dj_lim
        fp_periodic_x, fp_periodic_y = fp.periodic_x, fp.periodic_y
        x_period, y_period = fp.x_period, fp.y_period
        is_cartesian = fp.is_cartesian
        rad = fp.rad
        sc = FlowGeometries.Connectivity.ball_scratch()   # per row; see the single-field row apply
        for i in 1:Nx
            FlowGeometries.Grids.isactive(grid, i, j) || continue
            target = FlowGeometries.Grids.coords(SA.SVector, grid, i, j)
            fill!(acc_ws, zero(T))
            fill!(acc_wn, zero(T))
            _scattered_foldl(
                nothing, grid, target, i, j, Nx, Ny, di_lim, dj_lim,
                fp_periodic_x, fp_periodic_y, x_period, y_period, is_cartesian, rad, fp.topology, sc,
            ) do _, iin, jjn, d
                active = FlowGeometries.Grids.isactive(grid, iin, jjn)
                w = Kernels.kernel_weight(kernel, d, scale) * FlowGeometries.Grids.area(grid, iin, jjn)
                if strategy isa ZeroFill
                    for m in eachindex(fields)
                        acc_wn[m] += w
                    end
                    if active
                        for m in eachindex(fields)
                            acc_ws[m] += w * fields[m][iin, jjn]
                        end
                    end
                elseif active
                    for m in eachindex(fields)
                        acc_wn[m] += w
                        acc_ws[m] += w * fields[m][iin, jjn]
                    end
                end
                nothing
            end
            for m in eachindex(outs)
                outs[m][i, j] = acc_wn[m] > T(1e-15) ? acc_ws[m] / acc_wn[m] : zero(T)
            end
        end
    end
    return outs
end

"""
    apply_footprint_nd_batch!(outs, fields, grid, fp, strategy) -> outs

Batched point-indexed apply over the whole grid. The batch shares each point's neighbour enumeration
across all `K` fields, so the enumeration is paid once rather than `K` times.
"""
function apply_footprint_nd_batch!(
    outs, fields, grid::FlowGeometries.Grids.StructuredGrid{G,T,N}, fp::Union{FilterFootprintND{N,T}, NDScatteredFilterPlan{N,T}}, strategy::AbstractMaskStrategy,
) where {N, T<:AbstractFloat, G}
    for out in outs
        fill!(out, zero(T))
    end
    return apply_footprint_nd_batch_over!(
        outs, fields, grid, fp, strategy, CartesianIndices(FlowGeometries.Grids.size_tuple(grid)),
    )
end

"""
    apply_footprint_nd_batch_over!(outs, fields, grid, fp, strategy, indices) -> outs

The batched apply restricted to `indices`. Each output point depends only on its own neighbourhood,
so a parallel backend can hand disjoint index blocks to different tasks and get the serial answer.
`outs` must already be zeroed — the caller owns that, since a block only writes its own points.
"""
function apply_footprint_nd_batch_over!(
    outs, fields, grid::FlowGeometries.Grids.StructuredGrid{G,T,N}, fp::FilterFootprintND{N,T}, strategy::AbstractMaskStrategy,
    indices,
) where {N, T<:AbstractFloat, G}
    dims = FlowGeometries.Grids.size_tuple(grid)
    periodic = FlowGeometries.Grids.periodic_flags(grid)
    mask = FlowGeometries.Grids.mask(grid)
    acc_ws = _batch_zeros(outs, T)
    acc_wn = _batch_zeros(outs, T)
    @inbounds for I in indices
        mask[I] || continue
        Ti = Tuple(I)
        fill!(acc_ws, zero(T))
        fill!(acc_wn, zero(T))
        for k in eachindex(fp.offsets)
            J, valid = _shift_index(Ti, fp.offsets[k], dims, periodic)
            valid || continue
            active = mask[J...]
            wk = fp.w[k]
            if strategy isa ZeroFill
                for m in eachindex(fields)
                    acc_wn[m] += wk
                end
                if active
                    for m in eachindex(fields)
                        acc_ws[m] += wk * fields[m][J...]
                    end
                end
            elseif active
                for m in eachindex(fields)
                    acc_wn[m] += wk
                    acc_ws[m] += wk * fields[m][J...]
                end
            end
        end
        for m in eachindex(outs)
            outs[m][I] = acc_wn[m] > T(1e-15) ? acc_ws[m] / acc_wn[m] : zero(T)
        end
    end
    return outs
end

# One streamed point of the ND batch. An `NTuple` batch has its width in the type, so it folds
# immutable accumulators through `_nd_foldl`'s `acc` and allocates nothing; unknown width keeps the
# mutable buffers. Both enumerate through `_nd_foldl`, so both match the cache builder.
@inline function _nd_stream_point!(
    outs::NTuple{K,<:AbstractArray}, fields::NTuple{K,<:AbstractArray},
    grid::FlowGeometries.Grids.StructuredGrid{G,T,N}, fp::NDScatteredFilterPlan{N,T},
    strategy::AbstractMaskStrategy, dims::NTuple{N,Int}, mask, I::CartesianIndex{N}, kernel, scale,
    _acc_ws, _acc_wn,
) where {K, N, T<:AbstractFloat, G}
    z = zero(SA.SVector{K,T})
    ws, wn = _nd_foldl(
        (z, z), grid, Tuple(I), dims, fp.lim, fp.periodic, fp.period, fp.is_cartesian, fp.rad,
    ) do a, J, d
        aws, awn = a
        active = mask[J...]
        wk = Kernels.kernel_weight(kernel, d, scale) * FlowGeometries.Grids.area(grid, J...)
        if strategy isa ZeroFill
            awn = awn .+ wk
            active && (aws = aws .+ wk .* SA.SVector{K,T}(ntuple(m -> fields[m][J...], Val(K))))
        elseif active
            awn = awn .+ wk
            aws = aws .+ wk .* SA.SVector{K,T}(ntuple(m -> fields[m][J...], Val(K)))
        end
        (aws, awn)
    end
    @inbounds for m in 1:K
        outs[m][I] = wn[m] > T(1e-15) ? ws[m] / wn[m] : zero(T)
    end
    return nothing
end

@inline function _nd_stream_point!(
    outs, fields, grid::FlowGeometries.Grids.StructuredGrid{G,T,N}, fp::NDScatteredFilterPlan{N,T},
    strategy::AbstractMaskStrategy, dims::NTuple{N,Int}, mask, I::CartesianIndex{N}, kernel, scale,
    acc_ws, acc_wn,
) where {N, T<:AbstractFloat, G}
    fill!(acc_ws, zero(T))
    fill!(acc_wn, zero(T))
    _nd_foldl(
        nothing, grid, Tuple(I), dims, fp.lim, fp.periodic, fp.period, fp.is_cartesian, fp.rad,
    ) do _, J, d
        active = mask[J...]
        wk = Kernels.kernel_weight(kernel, d, scale) * FlowGeometries.Grids.area(grid, J...)
        if strategy isa ZeroFill
            for m in eachindex(fields)
                acc_wn[m] += wk
            end
            if active
                for m in eachindex(fields)
                    acc_ws[m] += wk * fields[m][J...]
                end
            end
        elseif active
            for m in eachindex(fields)
                acc_wn[m] += wk
                acc_ws[m] += wk * fields[m][J...]
            end
        end
        nothing
    end
    @inbounds for m in eachindex(outs)
        outs[m][I] = acc_wn[m] > T(1e-15) ? acc_ws[m] / acc_wn[m] : zero(T)
    end
    return nothing
end

function apply_footprint_nd_batch_over!(
    outs, fields, grid::FlowGeometries.Grids.StructuredGrid{G,T,N}, fp::NDScatteredFilterPlan{N,T}, strategy::AbstractMaskStrategy,
    indices,
) where {N, T<:AbstractFloat, G}
    dims = FlowGeometries.Grids.size_tuple(grid)
    mask = FlowGeometries.Grids.mask(grid)
    acc_ws = _batch_zeros(outs, T)
    acc_wn = _batch_zeros(outs, T)
    if fp.cache !== nothing
        cache = fp.cache
        lin = LinearIndices(dims)
        @inbounds for I in indices
            mask[I] || continue
            t = lin[I]
            lo = cache.ptr[t]
            hi = cache.ptr[t+1] - 1
            fill!(acc_ws, zero(T))
            fill!(acc_wn, zero(T))
            for k in lo:hi
                J = cache.nbrs[k]
                active = mask[J...]
                wk = cache.w[k]
                if strategy isa ZeroFill
                    for m in eachindex(fields)
                        acc_wn[m] += wk
                    end
                    if active
                        for m in eachindex(fields)
                            acc_ws[m] += wk * fields[m][J...]
                        end
                    end
                elseif active
                    for m in eachindex(fields)
                        acc_wn[m] += wk
                        acc_ws[m] += wk * fields[m][J...]
                    end
                end
            end
            for m in eachindex(outs)
                outs[m][I] = acc_wn[m] > T(1e-15) ? acc_ws[m] / acc_wn[m] : zero(T)
            end
        end
    else
        kernel, scale = fp.kernel, fp.scale
        @inbounds for I in indices
            mask[I] || continue
            _nd_stream_point!(outs, fields, grid, fp, strategy, dims, mask, I, kernel, scale, acc_ws, acc_wn)
        end
    end
    return outs
end

# ---------------------------------------------------------------------------
# Reusable filter plans: build the footprint ONCE, apply to many fields/scales
# ---------------------------------------------------------------------------

"""
Physical-space plan: a precomputed footprint reused across all longitudes, fields, and layers — for
EVERY backend, not just serial. `kernel`/`scale` are retained only so the cached-footprint path can
still call each backend's row-parallel hook (which takes them positionally); they're not used to
rebuild the footprint once `footprint` is already built.
"""
struct PhysicalFilterPlan{FP, G<:FlowGeometries.Grids.AbstractGrid, S<:AbstractMaskStrategy, K<:Kernels.AbstractFilterKernel, T<:AbstractFloat, B<:ComputationalBackends.AbstractExecutionBackend} <: AbstractFilterPlan
    footprint::FP   # FilterFootprint (2D structured), FilterFootprintND (1D/3D), or ScatteredFilterPlan/NDScatteredFilterPlan (nonuniform/curvilinear)
    grid::G
    strategy::S
    kernel::K
    scale::T
    backend::B
end

"""
    prepare_workspace(backend, grid, footprint) -> workspace

Backend hook run ONCE by [`plan_filter`](@ref), whose result becomes the plan's stored workspace. The
default returns the footprint unchanged; a backend that needs its own residency — the GPU's device
buffers — returns something its apply step consumes directly, so no transfer is repeated per call.
"""
prepare_workspace(
    ::ComputationalBackends.AbstractExecutionBackend, ::FlowGeometries.Grids.AbstractGrid, fp,
) = fp

# Boundary-only validation (paid once per `plan_filter` call, not per grid point): a non-positive or
# non-finite filter scale is never physically meaningful and would otherwise surface later as a
# confusing NaN/zero-radius footprint deep in the call stack instead of a clear error at the API edge.
@inline function _validate_scale(scale::T) where {T<:AbstractFloat}
    isfinite(scale) && scale > zero(T) || throw(ArgumentError(
        "filter scale must be finite and positive, got $scale",
    ))
    return nothing
end

"""
    plan_filter(grid, kernel, scale; mask_strategy=ZeroFill(), backend=AutoBackend()) -> AbstractFilterPlan

Build a reusable filter plan: the footprint is precomputed ONCE regardless of backend (serial,
threaded, distributed, GPU, or MPI) and reused across every subsequent `filter_apply!` call — no
backend rebuilds it per call. Apply with `filter_apply!(out, field, plan)`.
"""
function plan_filter(
    grid::FlowGeometries.Grids.StructuredGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    method::AbstractFilterMethod = RealSpace(),
    spectral_backend::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    cache_strategy::AbstractCacheStrategy = AutoCache(),
    cache_byte_budget::Integer = DEFAULT_CACHE_BYTE_BUDGET,
    kwargs...,
) where {G<:FlowGeometries.Geometry.AbstractGeometry{T}} where {T<:AbstractFloat}
    _validate_scale(scale)
    if _resolve_method(grid, kernel, method) isa Spectral
        return spectral_filter_plan(spectral_backend, grid, kernel, scale; mask_strategy = mask_strategy, backend = backend, kwargs...)
    end
    resolved = _resolve_backend(backend, grid)
    _check_backend_compatible(grid, backend)
    fp = _padded_fft_applicable(grid, kernel, method) ?
        padded_fft_footprint(grid, kernel, scale; mask_strategy = mask_strategy) :
        build_footprint(grid, kernel, scale; mask_strategy = mask_strategy,
            cache_strategy = cache_strategy, cache_byte_budget = cache_byte_budget)
    return PhysicalFilterPlan(prepare_workspace(resolved, grid, fp), grid, mask_strategy, kernel, scale, resolved)
end

# The row-based parallel backends (Threaded/Distributed/GPU/MPI) decompose over rows of a 2D grid
# via `apply_footprint_row!`, which already works generically for CurvilinearGrid (a 2D grid using
# the scattered per-point footprint) as well as StructuredGrid.
_row_parallelizable(::FlowGeometries.Grids.StructuredGrid{G,T,2}) where {G,T} = true
_row_parallelizable(::FlowGeometries.Grids.CurvilinearGrid) = true
_row_parallelizable(::FlowGeometries.Grids.AbstractGrid) = false

# 1D/true-3D grids use point-indexed footprints, so their parallel hook iterates points rather than
# rows. Each output point reads neighbours and writes only its own cell, so that is equally valid.
# Threaded only; Distributed/GPU/MPI would need a domain decomposition for the ND case.
_nd_parallelizable(::FlowGeometries.Grids.StructuredGrid{G,T,1}) where {G,T} = true
_nd_parallelizable(::FlowGeometries.Grids.StructuredGrid{G,T,3}) where {G,T} = true
# A node set is point-indexed with no row structure, so it parallelizes on the same argument as the
# 1D/true-3D grids: `_footprint_node_point` writes only node `t`'s own cell.
_nd_parallelizable(::FlowGeometries.Grids.UnstructuredGrid) = true
_nd_parallelizable(::FlowGeometries.Grids.AbstractGrid) = false

# Whether `grid` can actually honor a specific concrete backend request.
_backend_supported(grid::FlowGeometries.Grids.AbstractGrid, ::ComputationalBackends.SerialBackend) = true
_backend_supported(grid::FlowGeometries.Grids.AbstractGrid, ::ComputationalBackends.ThreadedBackend) = _row_parallelizable(grid) || _nd_parallelizable(grid)
# Point-indexed grids decompose over linear indices into a SharedArray rather than over rows, so they
# are supported even though `_row_parallelizable` is false for them.
_backend_supported(grid::FlowGeometries.Grids.AbstractGrid, ::ComputationalBackends.DistributedBackend) =
    _row_parallelizable(grid) || _nd_parallelizable(grid)
# The device kernels for point-indexed footprints need no row decomposition, so GPU support follows
# `_nd_parallelizable` as well as `_row_parallelizable` — one kernel over a linear index serves 1-D,
# true-3-D and node sets alike.
_backend_supported(grid::FlowGeometries.Grids.AbstractGrid, ::ComputationalBackends.GPUBackend) =
    _row_parallelizable(grid) || _nd_parallelizable(grid)
# Point-indexed grids partition round-robin over linear indices and recombine with `Allreduce!`, so
# they are supported even though `_row_parallelizable` is false for them.
_backend_supported(grid::FlowGeometries.Grids.AbstractGrid, ::ComputationalBackends.MPIBackend) =
    _row_parallelizable(grid) || _nd_parallelizable(grid)

# `AutoBackend()` landing on serial is auto-selection working; an explicit non-serial request that
# cannot be honoured is an error, since the caller would otherwise run on believing they had the
# parallelism. Checked against the original `backend`, before `AutoBackend()` is resolved away.
@inline _fftw_available() =
    Base.get_extension(parentmodule(@__MODULE__), :CoarseGrainingEnergyFluxesFFTWExt) !== nothing

# A transform filters by PERIODIC convolution, so it reproduces the intended filter only where every
# axis wraps, and it needs constant spacing — which a `Range` axis proves at the type level.
_spectral_exact(grid::FlowGeometries.Grids.StructuredGrid{G,T,N}) where {
    T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}, N,
} =
    all(ntuple(d -> FlowGeometries.Grids.isperiodic(grid, d), N)) &&
    all(ntuple(d -> FlowGeometries.Grids.coordinates(grid, d) isa AbstractRange, N))
_spectral_exact(::FlowGeometries.Grids.AbstractGrid) = false

@inline _besselj1_available() =
    Base.get_extension(parentmodule(@__MODULE__), :CoarseGrainingEnergyFluxesSpecialFunctionsExt) !== nothing

# `AutoMethod` picks on real capability, never on a preference: a transform only where it is available
# AND exact for this grid. Every kernel wins there, including the top-hat — its prefix-sum engine is
# O(N) but with a large enough constant to lose 60x to a transform whose cost does not scale with the
# filter width at all (30.0 ms vs 0.5 ms at half-width 64 on a periodic 256^2 grid).
@inline function _resolve_method(grid, kernel, method::AbstractFilterMethod)
    method isa AutoMethod || return method
    (_fftw_available() && _spectral_exact(grid)) || return RealSpace()
    # The planar top-hat's transfer function is the Bessel-J₁ form, which only exists when the
    # SpecialFunctions extension is loaded; without it there is no spectral top-hat to select.
    kernel isa Kernels.TopHatKernel && !_besselj1_available() && return RealSpace()
    return Spectral()
end

# Which kernels have a factored real-space engine. A `SharpSpectralKernel` has none — its radial `sinc`
# does not separate — so it falls to the banded disk sum at O(N·w²) with a 10ℓ radius.
_has_fast_real_space_engine(::Kernels.TopHatKernel) = true       # O(N) prefix sum
_has_fast_real_space_engine(::SeparableKernel) = true            # O(N·(wx+wy)) separable
_has_fast_real_space_engine(::Kernels.AbstractFilterKernel) = false

# `AutoMethod` may evaluate a real-space filter by padded transform: it is the SAME linear convolution,
# valid on bounded and masked domains, at O(N log N) instead of O(N·w²) — 716.7 → 0.7 ms at half-width
# 40. Only for a kernel with no factored engine, and Cartesian only, since the padded transform assumes
# one translation-invariant footprint and a spherical grid's per-latitude bands are not that.
_padded_fft_applicable(grid, kernel, method::AbstractFilterMethod) = false
_padded_fft_applicable(
    grid::FlowGeometries.Grids.StructuredGrid{G,T,2,TP,<:Tuple{AbstractRange,AbstractRange}},
    kernel::Kernels.AbstractFilterKernel, ::AutoMethod,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}, TP} =
    !_has_fast_real_space_engine(kernel) && _fftw_available()

@inline _threading_available() =
    Base.get_extension(parentmodule(@__MODULE__), :CoarseGrainingEnergyFluxesOhMyThreadsExt) !== nothing

# Upstream leaves `resolve_backend(::AutoBackend)` to the consumer, since it cannot see whether this
# package's threading extension is loaded. Kept package-local rather than added as a method there:
# that signature is ComputationalBackends' own, so every consumer defining it would overwrite the rest.
@inline function _resolve_backend(
    backend::ComputationalBackends.AbstractExecutionBackend, grid::FlowGeometries.Grids.AbstractGrid,
)
    backend isa ComputationalBackends.AbstractAutoBackend ||
        return ComputationalBackends.resolve_backend(backend)
    threaded = ComputationalBackends.ThreadedBackend()
    # Auto must choose on REAL capability: threads available, the extension that implements them
    # loaded, AND this grid having a parallel path. Choosing a backend the grid cannot honor is what
    # silently produced serial execution under a reported parallel backend.
    return (Threads.nthreads() > 1 && _threading_available() && _backend_supported(grid, threaded)) ?
        threaded : ComputationalBackends.SerialBackend()
end

function _check_backend_compatible(grid::FlowGeometries.Grids.AbstractGrid, backend::ComputationalBackends.AbstractExecutionBackend)
    if !(backend isa ComputationalBackends.AutoBackend) && !(backend isa ComputationalBackends.SerialBackend) && !_backend_supported(grid, _resolve_backend(backend, grid))
        throw(ArgumentError(
            "backend = $(typeof(backend)) was requested explicitly, but $(typeof(grid)) has no " *
            "matching parallel hook for it — there is no way to honor this request. Pass " *
            "`backend = SerialBackend()` explicitly if serial execution is acceptable, or " *
            "`backend = AutoBackend()` to let the library choose.",
        ))
    end
    return nothing
end

"""
    build_footprint(grid::UnstructuredGrid, kernel, scale; kwargs...) -> NodeFilterPlan

Real-space footprint for a node set, from `Connectivity.fold_within` — the grid's own metric ball
query, so the neighbourhood honours the geometry's distance and any periodic wrap without this package
re-deriving either, and the fold hands back each neighbour's distance rather than making the weight
recompute it.

The sweep visits every node, which is where a spatial index pays for itself:
`Connectivity.default_sweep_topology` builds one when `NearestNeighbors` is loaded, taking the build
from `O(n²)` to `O(n log n)`, and returns the unindexed topology (same rows, linear scan) when it is
not. Either way it is paid ONCE per plan and reused by every `filter_apply!`, including the six to
nine a single `compute_Π!` makes.
"""
function build_footprint(
    grid::FlowGeometries.Grids.UnstructuredGrid{T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    kwargs...,
) where {T<:AbstractFloat}
    rad = Kernels.kernel_radius(kernel, scale)
    n = length(FlowGeometries.Grids.mask(grid))
    ptr = Vector{Int}(undef, n + 1)
    nbrs = Int[]
    w = T[]
    ptr[1] = 1
    mt = _query_topology(grid, rad)
    scratch = FlowGeometries.Connectivity.ball_scratch()
    for t in 1:n
        # `active_only = false`: which cells are masked is the mask STRATEGY's business at apply time,
        # exactly as it is for the structured engines, whose caches are likewise mask-independent.
        # `self = true` folds the centre at distance zero, where the kernel carries its largest weight.
        FlowGeometries.Connectivity.fold_within(
            nothing, grid, t; ball = rad, self = true, active_only = false,
            topology = mt, scratch = scratch,
        ) do _, j, d
            push!(nbrs, j)
            push!(w, Kernels.kernel_weight(kernel, T(d), scale) * FlowGeometries.Grids.measure(grid, j))
            return nothing
        end
        ptr[t+1] = length(nbrs) + 1
    end
    return NodeFilterPlan(nbrs, w, ptr)
end

"""
    apply_footprint!(out, field, grid, fp::NodeFilterPlan, strategy) -> out

Weighted mean over each node's stored neighbourhood, with the same two mask conventions the structured
engines use: `ZeroFill` keeps a masked neighbour in the denominator and contributes nothing for it,
`Deformable` drops it from both.
"""
function apply_footprint!(
    out::AbstractVector{T}, field::AbstractVector, grid::FlowGeometries.Grids.UnstructuredGrid,
    fp::NodeFilterPlan{T}, strategy::AbstractMaskStrategy,
) where {T<:AbstractFloat}
    @inbounds for t in eachindex(out)
        out[t] = _footprint_node_point(field, grid, fp, strategy, t)
    end
    return out
end

# Per-node kernel, factored out of the loop above so a parallel driver reuses the exact same
# arithmetic rather than duplicating it — node `t` reads neighbours and writes only its own cell, so
# the threaded result is bit-identical. Mirrors `_footprint_nd_point`'s role for the ND engines.
@inline function _footprint_node_point(
    field::AbstractVector, grid::FlowGeometries.Grids.UnstructuredGrid,
    fp::NodeFilterPlan{T}, strategy::AbstractMaskStrategy, t::Integer,
) where {T<:AbstractFloat}
    FlowGeometries.Grids.isactive(grid, t) || return zero(T)
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
    return wn > T(1e-15) ? ws / wn : zero(T)
end

# A node set now has a real-space engine as well as the spectral one, so it gets a `PhysicalFilterPlan`
# like every other grid. This method remains the fallback for any grid architecture that has neither.
function plan_filter(
    grid::FlowGeometries.Grids.AbstractGrid,
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    method::AbstractFilterMethod = Spectral(),
    spectral_backend::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    kwargs...,
) where {T<:AbstractFloat}
    _validate_scale(scale)
    method isa Spectral || throw(ArgumentError(
        "Only spectral filtering (`method = Spectral()`) is implemented for $(typeof(grid)). A " *
        "real-space engine needs a neighbour search out to the filter radius, which this grid " *
        "architecture provides no query for; use `Spectral()`, or a StructuredGrid, CurvilinearGrid " *
        "or UnstructuredGrid, which have real-space engines.",
    ))
    return spectral_filter_plan(spectral_backend, grid, kernel, scale; mask_strategy = mask_strategy, backend = backend, kwargs...)
end

"""
    plan_filter(grid::UnstructuredGrid, kernel, scale; method = Spectral(), …)

A node set supports both methods, and both plan in `O(n log n)`. `Spectral()` is the default: it is
exact for a band-limited field and its per-apply cost is independent of the filter scale, where the
real-space engine's grows with the neighbour count inside the ball. `RealSpace()` applies the kernel
as written, with compact support; a transform's support is global.
"""
function plan_filter(
    grid::FlowGeometries.Grids.UnstructuredGrid{T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    method::AbstractFilterMethod = Spectral(),
    spectral_backend::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    kwargs...,
) where {T<:AbstractFloat}
    _validate_scale(scale)
    method isa Spectral && return spectral_filter_plan(
        spectral_backend, grid, kernel, scale; mask_strategy = mask_strategy, backend = backend, kwargs...,
    )
    _check_backend_compatible(grid, backend)
    return PhysicalFilterPlan(
        build_footprint(grid, kernel, scale), grid, mask_strategy, kernel, scale,
        _resolve_backend(backend, grid),
    )
end

# Curvilinear grids have a genuine real-space direct-sum engine (the scattered per-point footprint),
# so — unlike the unstructured/spectral-only fallback above — they precompute a `PhysicalFilterPlan`.
# More specific than the `AbstractGrid` method, so it is chosen for a `CurvilinearGrid`.
function plan_filter(
    grid::FlowGeometries.Grids.CurvilinearGrid{T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    method::AbstractFilterMethod = RealSpace(),
    spectral_backend::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    cache_strategy::AbstractCacheStrategy = AutoCache(),
    cache_byte_budget::Integer = DEFAULT_CACHE_BYTE_BUDGET,
    kwargs...,
) where {T<:AbstractFloat}
    _validate_scale(scale)
    if method isa Spectral
        # No spectral backend targets a CurvilinearGrid (FINUFFT/NUFSHT are UnstructuredGrid-only),
        # so this raises the standard informative "spectral unavailable" error.
        return spectral_filter_plan(spectral_backend, grid, kernel, scale; mask_strategy = mask_strategy, backend = backend, kwargs...)
    end
    resolved = _resolve_backend(backend, grid)
    _check_backend_compatible(grid, backend)
    fp = build_footprint(grid, kernel, scale; cache_strategy = cache_strategy, cache_byte_budget = cache_byte_budget)
    return PhysicalFilterPlan(prepare_workspace(resolved, grid, fp), grid, mask_strategy, kernel, scale, resolved)
end

"""
    filter_apply!(out, field, plan) -> out

Apply a prebuilt [`plan_filter`](@ref) to a single 2D field, dispatching to whichever backend the
plan was built for — the footprint is ALWAYS the one cached in `plan`, never rebuilt here, for every
backend (serial, threaded, distributed, GPU, MPI).
"""
function filter_apply!(out::AbstractArray, field::AbstractArray, plan::PhysicalFilterPlan)
    if plan.backend isa ComputationalBackends.SerialBackend
        return _apply_serial!(out, field, plan.grid, plan.footprint, plan.strategy)
    elseif plan.backend isa ComputationalBackends.ThreadedBackend
        return threaded_filter_field!(out, field, plan.grid, plan.kernel, plan.scale, plan.strategy, plan.footprint)
    elseif plan.backend isa ComputationalBackends.DistributedBackend
        return distributed_filter_field!(out, field, plan.grid, plan.kernel, plan.scale, plan.strategy, plan.footprint)
    elseif plan.backend isa ComputationalBackends.GPUBackend
        return gpu_filter_field!(plan.backend, out, field, plan.grid, plan.kernel, plan.scale, plan.strategy, plan.footprint)
    elseif plan.backend isa ComputationalBackends.MPIBackend
        return mpi_filter_field!(out, field, plan.grid, plan.kernel, plan.scale, plan.strategy, plan.footprint)
    else
        throw(ArgumentError("Unsupported backend: $(typeof(plan.backend))"))
    end
end

"""
    filter_apply_batch!(outs, fields, plan::AbstractFilterPlan) -> outs

Apply `plan` to every field in `fields`, writing into the matching entry of `outs`, deriving each
target point's neighbour list/weight exactly ONCE and reusing it across the whole batch — not once
per field. `outs`/`fields` must be equal-length, matching-shape collections of arrays: an
`NTuple{K,V}` (single concrete array type `V`) for a compile-time-known batch size (fastest — see
`_batch_zeros`), or an `AbstractVector` for a runtime-determined batch size.
"""
# A fused device batch exists only for engines whose kernel carries a batch index; the extension
# overrides this for those. Never guess — an unfused engine must take the slice loop, not a wrong kernel.
_gpu_batched_supported(::AbstractFilterPlan) = false

"""
    filter_apply_batched!(out, field, plan) -> out

Apply `plan` across a **trailing batch axis**: `out` and `field` are `(spatial..., Nb)` over the plan's
grid. The filter gathers only along spatial axes, so slices are independent and the batch index is
carried through untouched.

On a device this issues ONE launch of `prod(spatial) * Nb` work items instead of `Nb` launches of
`prod(spatial)`, which is what matters when a single slice does not fill the device — a 64² slice is
4k work items. On the host it is the same work as slicing and looping, and exists so callers have one
shape-generic entry point rather than reimplementing the loop.

Differs from [`filter_apply_batch!`](@ref), which takes several *separate* fields sharing a grid; here
the batch is one contiguous array, which is what allows the fused launch.
"""
function filter_apply_batched!(out::AbstractArray, field::AbstractArray, plan::AbstractFilterPlan)
    spatial = FlowGeometries.Grids.size_tuple(plan.grid)
    valR = Val(length(spatial))
    _check_batched_shape(out, "out", spatial, valR)
    _check_batched_shape(field, "field", spatial, valR)
    size(out) == size(field) || throw(DimensionMismatch(
        "filter_apply_batched! got out $(size(out)) and field $(size(field))",
    ))
    if plan.backend isa ComputationalBackends.GPUBackend && _gpu_batched_supported(plan)
        gpu_filter_field_batched!(
            plan.backend, out, field, plan.grid, plan.kernel, plan.scale, plan.strategy, plan.footprint,
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

Split of a spectral apply into its two halves. A spectral filter is forward transform → multiply by
`Ĝ(|k|, ℓ)` → inverse transform, and only the multiply depends on the scale, so a sweep over S scales
needs the forward ONCE per field rather than once per (field, scale): `5 + 5S` transforms instead of
`10S`.

`analyze_buffer` returns `nothing` for an engine with no transform to share — a real-space filter does
all its work per scale — and callers fall back to [`filter_apply!`](@ref).

Plans for different scales over the same grid share a forward transform, so `F̂` produced with any one
of them may be synthesized with any other.
"""
function analyze_buffer end
function filter_analyze! end
function filter_synthesize! end

# Real-space engines have no scale-independent half to hoist.
analyze_buffer(::AbstractFilterPlan, ::AbstractArray) = nothing

function gpu_filter_field_batched!(args...; kwargs...)
    throw(ArgumentError("GPUBackend is unavailable — run `using KernelAbstractions` (or use SerialBackend())."))
end

function filter_apply_batch!(outs, fields, plan::PhysicalFilterPlan)
    _batched_fields(outs, plan) && return filter_apply_batch_trailing!(outs, fields, plan)
    if plan.backend isa ComputationalBackends.SerialBackend
        return _apply_serial_batch!(outs, fields, plan.grid, plan.footprint, plan.strategy)
    elseif plan.backend isa ComputationalBackends.ThreadedBackend
        return threaded_filter_fields!(outs, fields, plan.grid, plan.kernel, plan.scale, plan.strategy, plan.footprint)
    else
        # Distributed/GPU/MPI decompose the domain across workers or a device, so the batch hoist
        # would have to be expressed inside each one's own decomposition rather than around it. They
        # apply per field, each call still reusing the plan's footprint.
        for k in eachindex(outs)
            filter_apply!(outs[k], fields[k], plan)
        end
        return outs
    end
end

# Spectral plans have no per-point neighbour derivation to hoist out of a per-field loop — each apply
# is an independent transform pass — so batching has nothing to collapse and they apply sequentially.
function filter_apply_batch!(outs, fields, plan::AbstractFilterPlan)
    _batched_fields(outs, plan) && return filter_apply_batch_trailing!(outs, fields, plan)
    for k in eachindex(outs)
        filter_apply!(outs[k], fields[k], plan)
    end
    return outs
end

# When the fields themselves carry a trailing batch axis, each one goes through the batched entry point
# rather than the per-slice apply: the field axis stays a loop, but every field's whole batch is one pass.
# `_batched_fields` is checked against the plan's grid, so a true-3D field on a 3D grid is not mistaken
# for a batched 2D field.
@inline function _batched_fields(outs, plan::PhysicalFilterPlan)
    R = length(FlowGeometries.Grids.size_tuple(plan.grid))
    return ndims(first(outs)) == R + 1
end

# A spectral plan carries transforms rather than a grid, so it has no spatial rank to compare against and
# no trailing-batch form; it keeps the per-field loop.
@inline _batched_fields(outs, ::AbstractFilterPlan) = false

function filter_apply_batch_trailing!(outs, fields, plan::AbstractFilterPlan)
    for k in eachindex(outs)
        filter_apply_batched!(outs[k], fields[k], plan)
    end
    return outs
end

"""
    filter_fields!(outs, fields, grid, kernel, scale; mask_strategy=ZeroFill(), backend=AutoBackend())

Filter several fields that share the same grid/kernel/scale, building the footprint/plan ONCE and
applying it through [`filter_apply_batch!`](@ref), so each target point's neighbours are enumerated
once for the whole batch rather than once per field. `outs` and `fields` are indexable collections of
matching arrays — a tuple of velocity components, or a vector of them.
"""
function filter_fields!(
    outs,
    fields,
    grid::FlowGeometries.Grids.StructuredGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    filter_plan::Union{Nothing,AbstractFilterPlan} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
) where {G<:FlowGeometries.Geometry.AbstractGeometry{T}} where {T<:AbstractFloat}
    plan = filter_plan === nothing ? plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend) : filter_plan
    return filter_apply_batch!(outs, fields, plan)
end

# ---------------------------------------------------------------------------
# Slice-parallel apply: many independent problems, one plan each
# ---------------------------------------------------------------------------

"""
    filter_slices!(outs, fields, plans; backend = AutoBackend()) -> outs

Apply `plans[t]` to `fields[t]`, writing `outs[t]`, over a collection of **independent** slices.

This is a different parallel axis from [`filter_apply_batch!`](@ref), which shares one grid across
several fields: here each slice has its own grid, plan and point count, and slices share nothing, so
there is no synchronization at all. Where a workload has many slices, this is the outermost
race-free axis and the one that converts thread count into throughput — threading *within* one slice
saturates once the slice is small enough that per-task overhead dominates its work.

Each slice runs **serially inside**, whatever backend its own plan carries: nesting a threaded apply
under a threaded slice loop would have both levels claim the whole thread pool.
"""
function filter_slices!(
    outs, fields, plans;
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
)
    length(outs) == length(fields) == length(plans) || throw(DimensionMismatch(
        "filter_slices! got $(length(outs)) outputs, $(length(fields)) fields and $(length(plans)) plans",
    ))
    resolved = _resolve_slice_backend(backend)
    resolved isa ComputationalBackends.ThreadedBackend &&
        return threaded_filter_slices!(outs, fields, plans)
    for t in eachindex(plans)
        apply_slice_serial!(outs[t], fields[t], plans[t])
    end
    return outs
end

# Slices are independent for every grid architecture, so — unlike `_resolve_backend` — this needs no
# grid capability check: the parallelism is over the collection, not inside any one slice.
@inline function _resolve_slice_backend(backend::ComputationalBackends.AbstractExecutionBackend)
    backend isa ComputationalBackends.AbstractAutoBackend ||
        return ComputationalBackends.resolve_backend(backend)
    return (Threads.nthreads() > 1 && _threading_available()) ?
        ComputationalBackends.ThreadedBackend() : ComputationalBackends.SerialBackend()
end

"""
    apply_slice_serial!(out, field, plan) -> out

One slice, forced down the serial engine regardless of the backend recorded in `plan`. The slice
loop owns the parallelism; see [`filter_slices!`](@ref).
"""
@inline apply_slice_serial!(out, field, plan::PhysicalFilterPlan) =
    _apply_serial!(out, field, plan.grid, plan.footprint, plan.strategy)
# A spectral plan has no separate serial engine — its transform is already the whole apply.
@inline apply_slice_serial!(out, field, plan::AbstractFilterPlan) = filter_apply!(out, field, plan)

"""
    slice_costs(plans) -> Vector{Int}

Per-slice work proxy: the number of target points each plan writes. A slice's cost grows at least
linearly in this, so it is what a longest-first schedule should sort on.
"""
slice_costs(plans) = [_plan_npoints(p) for p in plans]

@inline _plan_npoints(p::PhysicalFilterPlan) = prod(FlowGeometries.Grids.size_tuple(p.grid))
@inline _plan_npoints(p::AbstractFilterPlan) = 1

end # module
