module Filtering

using ..Geometry: Geometry
using ..Grids: Grids
using ..Kernels: Kernels
using ..Backends: Backends
using StaticArrays: StaticArrays as SA

export AbstractMaskStrategy, ZeroFill, Deformable
export AbstractFilterMethod, DirectSum, Spectral
export filter_field!, filter_fields!
export AbstractFilterPlan, plan_filter, filter_apply!

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
the locally-included area only ("deformable kernel"). Excluded cells are genuinely dropped, but the kernel becomes
inhomogeneous near a mask boundary (breaks the strict commutation theorems).
"""
struct Deformable <: AbstractMaskStrategy end

# ---------------------------------------------------------------------------
# Filtering method: physical direct-sum (default) vs spectral (FFT/SHT/NUFFT via extensions)
# ---------------------------------------------------------------------------

"""
    AbstractFilterMethod

How the convolution is evaluated: [`DirectSum`](@ref) (physical-space footprint, any grid/mask) or
[`Spectral`](@ref) (transform-space multiply — FFT for uniform periodic Cartesian, spherical
harmonics for the uniform sphere, NUFFT/NUFSHT for scattered points; provided by extensions).
"""
abstract type AbstractFilterMethod end

"Physical-space direct-sum convolution (works on any grid, mask, and geometry)."
struct DirectSum <: AbstractFilterMethod end

"""
    Spectral <: AbstractFilterMethod

Transform-space filtering (kernel applied as a multiply on the transformed field). Requires a
spectral extension and a compatible grid (e.g. `using FFTW` for a uniform, periodic Cartesian grid).
"""
struct Spectral <: AbstractFilterMethod end

# ---------------------------------------------------------------------------
# Extension hook points
# ---------------------------------------------------------------------------
# Fallbacks that error until the relevant backend extension is loaded. Each execution-backend
# extension overrides its hook; the public `filter_field!` dispatches here based on the resolved
# backend. (Backend TYPES live in `Backends`; these are the per-backend filtering implementations.)

function threaded_filter_field!(args...; kwargs...)
    throw(ArgumentError("ThreadedBackend is unavailable — run `using OhMyThreads` (or use SerialBackend())."))
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
function spectral_filter_plan(grid, kernel, scale; kwargs...)
    throw(ArgumentError(
        "Spectral filtering is unavailable for $(typeof(grid)) — load a spectral backend " *
        "(`using FFTW` uniform Cartesian, `using FINUFFT` scattered Cartesian, " *
        "`using FastSphericalHarmonics` uniform spherical, `using NUFSHT` scattered spherical).",
    ))
end

# ---------------------------------------------------------------------------
# Public Filtering API
# ---------------------------------------------------------------------------

"""
    filter_field!(out, field, grid, kernel, scale; mask_strategy=Deformable(), workspace=nothing, backend=AutoBackend())

Filter a field on a grid using `kernel` at characteristic full width `scale` (ℓ), writing the
result to `out` (returned).

# Keyword Arguments
- `mask_strategy::AbstractMaskStrategy=Deformable()`: masking strategy — `ZeroFill()` (excluded cells count
  in the denominator as zero; homogeneous kernel) or `Deformable()` (excluded cells dropped from
  numerator and denominator; renormalized over the locally-included area).
- `workspace=nothing`: reserved for a precomputed footprint/plan (currently unused).
- `backend::AbstractExecutionBackend=AutoBackend()`: execution backend (SerialBackend,
  ThreadedBackend, GPUBackend, …).

For spherical grids the longitude footprint wraps only when the grid is periodic (`isperiodic`);
distances use the great-circle (Haversine) metric.

# Examples
```julia
geom = CartesianGeometry(1000.0, 1000.0)
grid = StructuredGrid(geom, lon, lat, mask)
out = zeros(100, 100)
filter_field!(out, field, grid, TopHatKernel(), 5000.0; mask_strategy = Deformable())
```
"""
function filter_field!(
    out::AbstractArray{T},
    field::AbstractArray,
    grid::Grids.AbstractGrid,
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = Deformable(),
    workspace = nothing,
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    method::AbstractFilterMethod = DirectSum(),
) where {T<:AbstractFloat}

    # Spectral methods route through a (cached) transform plan provided by an extension.
    if method isa Spectral
        return filter_apply!(out, field, plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend, method = method))
    end

    # 1. Resolve AutoBackend to a concrete backend instance.
    resolved = Backends.resolve_backend(backend)
    _check_backend_compatible(grid, backend)

    # 2. Dispatch to the appropriate execution backend (extensions override the hook functions).
    #    Distributed/GPU/MPI are latitude-row decomposed and thus 2D only; Threaded ALSO supports the
    #    1D/true-3D point-indexed (`FilterFootprintND`) representation via `_nd_parallelizable` — see
    #    `_backend_supported` above. A grid/backend combination with no matching hook always uses the
    #    serial engine (silently for AutoBackend, already rejected above for an explicit request).
    if resolved isa Backends.SerialBackend || !_backend_supported(grid, resolved)
        serial_filter_field!(out, field, grid, kernel, scale, mask_strategy, workspace)
    elseif resolved isa Backends.ThreadedBackend
        threaded_filter_field!(out, field, grid, kernel, scale, mask_strategy, workspace)
    elseif resolved isa Backends.DistributedBackend
        distributed_filter_field!(out, field, grid, kernel, scale, mask_strategy, workspace)
    elseif resolved isa Backends.GPUBackend
        gpu_filter_field!(resolved, out, field, grid, kernel, scale, mask_strategy, workspace)
    elseif resolved isa Backends.MPIBackend
        mpi_filter_field!(out, field, grid, kernel, scale, mask_strategy, workspace)
    else
        throw(ArgumentError("Unsupported backend: $(typeof(resolved))"))
    end

    return out
end

# 3D volume filtering: horizontal filtering applied layer-by-layer (2.5D). A single footprint is
# built once and reused across all depth layers.
function filter_field!(
    out::AbstractArray{T,3},
    field::AbstractArray{<:Any,3},
    grid::Grids.StructuredGrid{<:Geometry.AbstractGeometry,T,2},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = Deformable(),
    workspace = nothing,
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
) where {T<:AbstractFloat}
    resolved = Backends.resolve_backend(backend)
    if resolved isa Backends.SerialBackend
        fp = build_footprint(grid, kernel, scale)
        periodic = Grids.isperiodic(grid, 1)
        for k in axes(field, 3)
            apply_footprint!(view(out, :, :, k), view(field, :, :, k), grid, fp, mask_strategy, periodic)
        end
    else
        for k in axes(field, 3)
            filter_field!(view(out, :, :, k), view(field, :, :, k), grid, kernel, scale;
                mask_strategy = mask_strategy, workspace = workspace, backend = backend)
        end
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
in a flat CSR-like layout, grouped into latitude bands (`ptr[b]:ptr[b+1]-1`). For Cartesian grids
the footprint is translation-invariant → a single band; for spherical lat-lon it is invariant in
longitude → one band per latitude. The weights are mask-independent (geometry only); masking is
applied when the footprint is convolved with a field.
"""
struct FilterFootprint{T<:AbstractFloat, VI<:AbstractVector{Int}, VT<:AbstractVector{T}}
    di::VI    # longitude index offset
    dj::VI    # latitude index offset
    w::VT       # kernel_weight(distance) * cell area
    ptr::VI   # band b's entries: ptr[b]:ptr[b+1]-1
    nbands::Int        # 1 (Cartesian) or Nlat (spherical)
end

@inline _band(::FilterFootprint, ::Grids.StructuredGrid{<:Geometry.CartesianGeometry}, j::Integer) = 1
@inline _band(::FilterFootprint, ::Grids.StructuredGrid{<:Geometry.SphericalGeometry}, j::Integer) = j

"""
    FilterFootprintScattered{T}

Precomputed per-TARGET-POINT convolution footprint (absolute neighbour indices + weights), used
when a `StructuredGrid`'s axis spacing is genuinely nonuniform. `FilterFootprint` above is a
translation-invariant cache — the SAME index offset (and its weight) is reused for every target `i`
in a row/band — which is only valid when the index-to-physical-distance mapping is the same
everywhere, i.e. a uniform axis. For a nonuniform axis that assumption is false (offset `+3` means a
different physical displacement depending on where you start), so there is no way to share a single
offset/weight set across a row; each target point genuinely needs its own. This is still built ONCE
per (grid, kernel, scale) and reused across every subsequent `filter_apply!` call, exactly like
`FilterFootprint` — just without the translation-invariance memory saving, since that saving isn't
available on a nonuniform axis.
"""
struct FilterFootprintScattered{T<:AbstractFloat, VI<:AbstractVector{Int}, VT<:AbstractVector{T}}
    ii::VI    # absolute neighbour longitude/x index (periodic wrap already resolved)
    jj::VI    # absolute neighbour latitude/y index
    w::VT       # kernel_weight(distance) * cell area
    ptr::VI   # target t = i + (j-1)*Nlon; entries ptr[t]:ptr[t+1]-1
end

"""
    _build_footprint_scattered(grid, kernel, scale) -> FilterFootprintScattered

Build the per-point footprint by real distance checks at every target — correct for any spacing
pattern (Cartesian or spherical, uniform or not), since it never assumes translation invariance.
The per-target search window uses a conservative (safe, never under-covering) index-radius bound
derived from the SMALLEST gap found anywhere on each axis (via [`Grids._min_gap`](@ref)); the exact
`d <= rad` check below still gates inclusion, so a loose bound only costs extra iterations, never a
missed cell.
"""
function _build_footprint_scattered(
    grid::Grids.StructuredGrid{G,T,2},
    kernel::Kernels.AbstractFilterKernel,
    scale::T,
) where {T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}}
    Nlon, Nlat = Grids.size_tuple(grid)
    rad = Kernels.kernel_radius(kernel, scale)
    periodic_lon = Grids.isperiodic(grid, 1)

    min_dlat = Grids._min_gap(grid.lat)
    min_dlon = Grids._min_gap(grid.lon)
    if G <: Geometry.SphericalGeometry{T}
        R = grid.geometry.R
        dj_lim = isfinite(min_dlat) && min_dlat > 0 ? ceil(Int, rad / (R * min_dlat)) : 0
        cosφ_min = minimum((abs(cos(φ)) for φ in grid.lat if abs(cos(φ)) > T(1e-12)); init = one(T))
        di_lim = (isfinite(min_dlon) && min_dlon > 0 && cosφ_min > 0) ?
            ceil(Int, rad / (R * cosφ_min * min_dlon)) : 0
    else
        dj_lim = isfinite(min_dlat) && min_dlat > 0 ? ceil(Int, rad / min_dlat) : 0
        di_lim = isfinite(min_dlon) && min_dlon > 0 ? ceil(Int, rad / min_dlon) : 0
    end

    # A wrapped candidate's raw stored coordinate sits a full period away from the target on a
    # periodic CARTESIAN axis (e.g. index Nlon is `Lx` meters from index 1, not adjacent to it), so
    # the plain Euclidean `distance` below would reject every genuinely-close wrapped neighbor unless
    # shifted back by one period first. A periodic SPHERICAL axis needs no such shift: great-circle
    # distance is built from `cos`/`sin` of the raw longitude, which is already exactly 2π-periodic
    # regardless of the literal angle value. `lon_period` mirrors the same "extent + one cell
    # spacing" convention `StructuredGrid`'s own constructor uses to derive its periodic cell width.
    is_cartesian = G <: Geometry.CartesianGeometry{T}
    lon_period = (periodic_lon && is_cartesian) ?
        (grid.lon[end] - grid.lon[1] + (grid.lon[2] - grid.lon[1])) : zero(T)

    ii = Int[]
    jj = Int[]
    w = T[]
    # Conservative upper bound on total entries (every target's full search window), so push!
    # never needs to reallocate/copy mid-loop — an exact count would require redoing the distance
    # gate below for every candidate, which is the expensive part this cache exists to avoid paying twice.
    sizehint!(ii, Nlon * Nlat * (2*di_lim + 1) * (2*dj_lim + 1))
    sizehint!(jj, Nlon * Nlat * (2*di_lim + 1) * (2*dj_lim + 1))
    sizehint!(w, Nlon * Nlat * (2*di_lim + 1) * (2*dj_lim + 1))
    ptr = Vector{Int}(undef, Nlon * Nlat + 1)
    ptr[1] = 1
    for j in 1:Nlat, i in 1:Nlon # column-major target order: t = i + (j-1)*Nlon, increasing monotonically
        t = i + (j - 1) * Nlon
        target = Grids.coords(grid, i, j)
        j_lo = max(1, j - dj_lim)
        j_hi = min(Nlat, j + dj_lim)
        for jjn in j_lo:j_hi
            i_lo = i - di_lim
            i_hi = i + di_lim
            for ii_raw in i_lo:i_hi
                iin = ii_raw
                shift = zero(T)
                if iin < 1 || iin > Nlon
                    periodic_lon || continue
                    shift = iin < 1 ? -lon_period : lon_period
                    iin = mod1(iin, Nlon)
                end
                neighbor = Grids.coords(grid, iin, jjn)
                neighbor_shifted = (is_cartesian && !iszero(shift)) ?
                    (neighbor + SA.SVector{2,T}(shift, zero(T))) : neighbor
                d = Geometry.distance(grid.geometry, target, neighbor_shifted)
                d <= rad || continue
                push!(ii, iin)
                push!(jj, jjn)
                push!(w, Kernels.kernel_weight(kernel, d, scale) * Grids.area(grid, iin, jjn))
            end
        end
        ptr[t+1] = length(ii) + 1
    end
    return FilterFootprintScattered(ii, jj, w, ptr)
end

"""
    _build_footprint_curvilinear(grid, kernel, scale) -> FilterFootprintScattered

Per-target-point footprint for a [`Grids.CurvilinearGrid`](@ref). A curvilinear mesh stores its
coordinates as 2D arrays with no separable per-axis spacing, so there is no `lon[2]-lon[1]`-style
shortcut for the search radius (as the `StructuredGrid` scattered builder has); instead we walk BOTH
index directions to find the smallest physical spacing between adjacent nodes in each index
direction, giving a conservative (safe, never under-covering) per-direction index-radius bound. The
exact `distance(...) <= rad` gate still decides inclusion, so a loose bound only costs extra
iterations at build time. Reuses the same [`FilterFootprintScattered`](@ref) container and apply
path as the nonuniform-`StructuredGrid` case. `CurvilinearGrid` is treated as non-periodic.
"""
function _build_footprint_curvilinear(
    grid::Grids.CurvilinearGrid{T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T,
) where {T<:AbstractFloat}
    Nlon, Nlat = Grids.size_tuple(grid)
    geo = grid.geometry
    rad = Kernels.kernel_radius(kernel, scale)

    # Smallest adjacent-node spacing in each index direction (walk both directions of the 2D mesh).
    min_di = T(Inf)
    min_dj = T(Inf)
    for j in 1:Nlat, i in 1:Nlon
        c = Grids.coords(grid, i, j)
        if i < Nlon
            d = Geometry.distance(geo, c, Grids.coords(grid, i + 1, j))
            d > 0 && (min_di = min(min_di, d))
        end
        if j < Nlat
            d = Geometry.distance(geo, c, Grids.coords(grid, i, j + 1))
            d > 0 && (min_dj = min(min_dj, d))
        end
    end
    di_lim = isfinite(min_di) && min_di > 0 ? ceil(Int, rad / min_di) : 0
    dj_lim = isfinite(min_dj) && min_dj > 0 ? ceil(Int, rad / min_dj) : 0

    ii = Int[]
    jj = Int[]
    w = T[]
    # Conservative upper bound (see the analogous StructuredGrid scattered builder above for why
    # this is a sizehint!, not an exact preallocation: an exact count needs the distance gate below).
    sizehint!(ii, Nlon * Nlat * (2*di_lim + 1) * (2*dj_lim + 1))
    sizehint!(jj, Nlon * Nlat * (2*di_lim + 1) * (2*dj_lim + 1))
    sizehint!(w, Nlon * Nlat * (2*di_lim + 1) * (2*dj_lim + 1))
    ptr = Vector{Int}(undef, Nlon * Nlat + 1)
    ptr[1] = 1
    for j in 1:Nlat, i in 1:Nlon # column-major target order: t = i + (j-1)*Nlon
        t = i + (j - 1) * Nlon
        target = Grids.coords(grid, i, j)
        for jn in max(1, j - dj_lim):min(Nlat, j + dj_lim)
            for in_ in max(1, i - di_lim):min(Nlon, i + di_lim)
                neighbor = Grids.coords(grid, in_, jn)
                d = Geometry.distance(geo, target, neighbor)
                d <= rad || continue
                push!(ii, in_)
                push!(jj, jn)
                push!(w, Kernels.kernel_weight(kernel, d, scale) * Grids.area(grid, in_, jn))
            end
        end
        ptr[t+1] = length(ii) + 1
    end
    return FilterFootprintScattered(ii, jj, w, ptr)
end

"""
    build_footprint(grid::CurvilinearGrid, kernel, scale) -> FilterFootprintScattered

Real-space direct-sum footprint for a curvilinear grid (see [`_build_footprint_curvilinear`](@ref)).
"""
build_footprint(
    grid::Grids.CurvilinearGrid{T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T,
) where {T<:AbstractFloat} = _build_footprint_curvilinear(grid, kernel, scale)

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
    grid::Grids.StructuredGrid{G,T,2,<:Tuple{AbstractRange,AbstractRange}},
    kernel::Kernels.AbstractFilterKernel,
    scale::T,
) where {T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}}
    Nlon, Nlat = Grids.size_tuple(grid)
    rad = Kernels.kernel_radius(kernel, scale)
    di = Int[]
    dj = Int[]
    w = T[]
    ptr = Int[1]

    if G <: Geometry.CartesianGeometry{T}
        dx = step(grid.lon)
        dy = step(grid.lat)
        A = Grids.area(grid, 1, 1)   # uniform Cartesian cell area
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
        R = grid.geometry.R
        dλ = step(grid.lon)
        dφ = step(grid.lat)
        dj_lim = dφ > 0 ? ceil(Int, rad / (R * dφ)) : 0
        # Conservative upper bound across all Nlat row-appends below: di_lim widens as cosφ→0 near
        # the poles, so bound it using the smallest |cosφ| actually present on this grid.
        cosφ_min = minimum((abs(cos(φ)) for φ in grid.lat if abs(cos(φ)) > T(1e-12)); init = one(T))
        di_lim_max = (dλ > 0 && cosφ_min > 0) ? ceil(Int, rad / (R * cosφ_min * dλ)) : 0
        sizehint!(di, Nlat * (2*di_lim_max + 1) * (2*dj_lim + 1))
        sizehint!(dj, Nlat * (2*di_lim_max + 1) * (2*dj_lim + 1))
        sizehint!(w, Nlat * (2*di_lim_max + 1) * (2*dj_lim + 1))
        for j in 1:Nlat
            φ = grid.lat[j]
            cosφ = cos(φ)
            di_lim = (dλ > 0 && abs(cosφ) > T(1e-12)) ? ceil(Int, rad / (R * cosφ * dλ)) : 0
            for ddj in -dj_lim:dj_lim
                jj = j + ddj
                (1 <= jj <= Nlat) || continue
                φ2 = grid.lat[jj]
                A = Grids.area(grid, 1, jj)   # spherical cell area depends only on latitude
                for ddi in -di_lim:di_lim
                    # Great-circle distance with Δλ = ddi·dλ (longitude-translation-invariant).
                    d = Geometry.distance(
                        grid.geometry,
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
        return FilterFootprint(di, dj, w, ptr, Nlat)
    end
end

"""
    build_footprint(grid, kernel, scale) -> FilterFootprintScattered

General path: at least one axis is a plain (non-`Range`) `AbstractVector`, which makes no type-level
uniformity guarantee — its values might happen to be evenly spaced, but nothing proves it, so no
assumption is made and the always-correct per-point footprint is built instead. (Less specific than
the method above, so Julia only reaches this one when the fast method's constraint doesn't match.)
"""
function build_footprint(
    grid::Grids.StructuredGrid{G,T,2},
    kernel::Kernels.AbstractFilterKernel,
    scale::T,
) where {T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}}
    return _build_footprint_scattered(grid, kernel, scale)
end

"""
    apply_footprint!(out, field, grid, fp, strategy, periodic_lon)

Convolve `field` with a precomputed `fp` into `out`, applying the mask `strategy`. `out` and
`field` are 2D (a single layer). The masking branch specializes on the strategy type.
"""
function apply_footprint!(
    out::AbstractMatrix{T},
    field::AbstractMatrix,
    grid::Grids.StructuredGrid,
    fp::FilterFootprint{T},
    strategy::AbstractMaskStrategy,
    periodic_lon::Bool,
) where {T<:AbstractFloat}
    _, Nlat = Grids.size_tuple(grid)
    fill!(out, zero(T))
    for j in 1:Nlat
        apply_footprint_row!(out, field, grid, fp, strategy, periodic_lon, j)
    end
    return out
end

"""
    apply_footprint_row!(out, field, grid, fp, strategy, periodic_lon, j)

Fill output row `j` (`out[:, j]`) from a precomputed footprint. Rows are independent (each writes a
disjoint column of the column-major output), so this is the unit of parallelism for the threaded /
distributed backends. Callers must `fill!(out, 0)` first (masked cells are left untouched here).
"""
function apply_footprint_row!(
    out::AbstractMatrix{T},
    field::AbstractMatrix,
    grid::Grids.StructuredGrid,
    fp::FilterFootprint{T},
    strategy::AbstractMaskStrategy,
    periodic_lon::Bool,
    j::Integer,
) where {T<:AbstractFloat}
    Nlon, Nlat = Grids.size_tuple(grid)
    b = _band(fp, grid, j)
    lo = fp.ptr[b]
    hi = fp.ptr[b+1] - 1
    for i in 1:Nlon
        Grids.isactive(grid, i, j) || continue
        weighted_sum = zero(T)
        weight_norm = zero(T)
        @inbounds for k in lo:hi
            jj = j + fp.dj[k]
            (1 <= jj <= Nlat) || continue
            ii = i + fp.di[k]
            if ii < 1 || ii > Nlon
                periodic_lon || continue
                ii = mod1(ii, Nlon)
            end
            active = Grids.isactive(grid, ii, jj)
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
        out[i, j] = weight_norm > T(1e-15) ? weighted_sum / weight_norm : zero(T)
    end
    return out
end

"""
    apply_footprint!(out, field, grid, fp::FilterFootprintScattered, strategy, periodic_lon)

Whole-grid convolve using a per-point [`FilterFootprintScattered`](@ref) footprint (the nonuniform-axis
fallback). `periodic_lon` is accepted only for a uniform call signature with the `FilterFootprint` method
above — wrapping was already resolved into absolute indices at footprint-build time, so it's unused here.
"""
function apply_footprint!(
    out::AbstractMatrix{T},
    field::AbstractMatrix,
    grid::Grids.AbstractGrid,
    fp::FilterFootprintScattered{T},
    strategy::AbstractMaskStrategy,
    periodic_lon::Bool,
) where {T<:AbstractFloat}
    _, Nlat = Grids.size_tuple(grid)
    fill!(out, zero(T))
    for j in 1:Nlat
        apply_footprint_row!(out, field, grid, fp, strategy, periodic_lon, j)
    end
    return out
end

"""
    apply_footprint_row!(out, field, grid, fp::FilterFootprintScattered, strategy, periodic_lon, j)

Fill output row `j` from a precomputed per-point footprint — absolute `(ii,jj)` neighbour indices
(periodic wrap already resolved at build time), no offset arithmetic needed.
"""
function apply_footprint_row!(
    out::AbstractMatrix{T},
    field::AbstractMatrix,
    grid::Grids.AbstractGrid,
    fp::FilterFootprintScattered{T},
    strategy::AbstractMaskStrategy,
    periodic_lon::Bool,
    j::Integer,
) where {T<:AbstractFloat}
    Nlon, _ = Grids.size_tuple(grid)
    for i in 1:Nlon
        Grids.isactive(grid, i, j) || continue
        t = i + (j - 1) * Nlon
        lo = fp.ptr[t]
        hi = fp.ptr[t+1] - 1
        weighted_sum = zero(T)
        weight_norm = zero(T)
        @inbounds for k in lo:hi
            ii = fp.ii[k]
            jj = fp.jj[k]
            active = Grids.isactive(grid, ii, jj)
            w = fp.w[k]
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
    return out
end

# ---------------------------------------------------------------------------
# General N-dimensional engine (1D + 3D Cartesian); the 2D path uses the per-row engine above.
# ---------------------------------------------------------------------------

# Fast path: dispatched on ALL N axes being `AbstractRange` (compile-time proof of uniform spacing)
# AND `CartesianGeometry` — real multiple dispatch, no runtime check. Unlike the 2D per-row engine
# above (which has a genuine per-latitude-band spherical fast path), this translation-invariant
# single-offset-set scheme is fundamentally Cartesian-only: a spherical metric is position-dependent
# (arc length varies with latitude AND, in 3D, with radius), so it can never collapse to one shared
# offset set the way a flat Cartesian grid can. A spherical 1D/3D grid always takes the general path
# below instead — correct for any spacing pattern, just without a translation-invariant fast path
# (building a per-(latitude,radius)-band cache analogous to the 2D case is a possible future
# optimization, not yet implemented).
build_footprint(
    grid::Grids.StructuredGrid{G,T,1,<:Tuple{AbstractRange}},
    kernel::Kernels.AbstractFilterKernel, scale::T,
) where {T<:AbstractFloat, G<:Geometry.CartesianGeometry{T}} = _build_footprint_nd(grid, kernel, scale)
build_footprint(
    grid::Grids.StructuredGrid{G,T,3,<:Tuple{AbstractRange,AbstractRange,AbstractRange}},
    kernel::Kernels.AbstractFilterKernel, scale::T,
) where {T<:AbstractFloat, G<:Geometry.CartesianGeometry{T}} = _build_footprint_nd(grid, kernel, scale)

# General path: at least one axis is a plain AbstractVector (no uniformity guarantee), OR the
# geometry is non-Cartesian (no translation-invariant fast path exists, see above) — less specific
# than the two Cartesian-only methods above, reached whenever they don't match.
build_footprint(grid::Grids.StructuredGrid{G,T,1}, kernel::Kernels.AbstractFilterKernel, scale::T) where {T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}} =
    _build_footprint_nd_scattered(grid, kernel, scale)
build_footprint(grid::Grids.StructuredGrid{G,T,3}, kernel::Kernels.AbstractFilterKernel, scale::T) where {T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}} =
    _build_footprint_nd_scattered(grid, kernel, scale)

function _build_footprint_nd(
    grid::Grids.StructuredGrid{G,T,N},
    kernel::Kernels.AbstractFilterKernel,
    scale::T,
) where {N, T<:AbstractFloat, G<:Geometry.CartesianGeometry{T}}
    rad = Kernels.kernel_radius(kernel, scale)
    # Real per-axis step, read from the axis itself (already proven uniform by its Range type via
    # the calling method's dispatch constraint) — not the geometry's separately-stored dx/dy/dz,
    # so there's no possibility of the two disagreeing.
    spacing = ntuple(d -> step(grid.axes[d]), N)
    A = grid.measure[ntuple(_ -> 1, N)...]   # uniform Cartesian cell measure
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
    FilterFootprintNDScattered{N, T}

Per-TARGET-POINT N-dimensional footprint (absolute neighbour multi-indices + weights), the N-D
analog of [`FilterFootprintScattered`](@ref) for when at least one of the N axes is a plain
`AbstractVector` (no type-level uniformity proof) — no translation-invariance assumption, correct
for any spacing pattern. Built once per (grid, kernel, scale); reused across every subsequent
`filter_apply!` call.
"""
struct FilterFootprintNDScattered{N, T<:AbstractFloat, VO<:AbstractVector{NTuple{N,Int}}, VT<:AbstractVector{T}, VI<:AbstractVector{Int}}
    nbrs::VO
    w::VT
    ptr::VI   # target t = LinearIndices(dims)[I]; entries ptr[t]:ptr[t+1]-1
end

function _build_footprint_nd_scattered(
    grid::Grids.StructuredGrid{G,T,N},
    kernel::Kernels.AbstractFilterKernel,
    scale::T,
) where {N, T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}}
    dims = Grids.size_tuple(grid)
    rad = Kernels.kernel_radius(kernel, scale)
    periodic = grid.periodic
    # Conservative (safe, never under-covering) per-axis index-radius bound from the smallest gap
    # found anywhere on that axis — the exact `d <= rad` check below still gates inclusion. A raw
    # axis gap IS already physical distance for Cartesian, but for spherical lon/lat axes it's
    # ANGULAR (radians) — dividing a physical `rad` by a radian gap directly would blow the window up
    # by orders of magnitude (radians are O(1), meters are O(1e5-1e6)). Convert via the local metric
    # (r·cosφ for longitude, r for latitude; the radial axis — N=3 only — is already physical
    # distance, no conversion needed), using the smallest r/|cosφ| found on the grid to stay
    # conservative. Mirrors `_build_footprint_scattered`'s 2D spherical handling.
    lim = if G <: Geometry.CartesianGeometry{T}
        ntuple(N) do d
            g = Grids._min_gap(grid.axes[d])
            isfinite(g) && g > 0 ? ceil(Int, rad / g) : 0
        end
    else
        r_min = N == 3 ? minimum(abs, grid.axes[3]) : grid.geometry.R
        cosφ_min = minimum((abs(cos(φ)) for φ in grid.axes[2] if abs(cos(φ)) > T(1e-12)); init = one(T))
        min_dlon = Grids._min_gap(grid.axes[1])
        min_dlat = Grids._min_gap(grid.axes[2])
        lon_lim = (isfinite(min_dlon) && min_dlon > 0 && cosφ_min > 0) ?
            ceil(Int, rad / (r_min * cosφ_min * min_dlon)) : 0
        lat_lim = (isfinite(min_dlat) && min_dlat > 0) ? ceil(Int, rad / (r_min * min_dlat)) : 0
        if N == 3
            min_dr = Grids._min_gap(grid.axes[3])
            r_lim = (isfinite(min_dr) && min_dr > 0) ? ceil(Int, rad / min_dr) : 0
            (lon_lim, lat_lim, r_lim)
        else
            (lon_lim, lat_lim)
        end
    end

    # A wrapped candidate's raw stored coordinate sits a full period away from the target on a
    # periodic CARTESIAN axis, so plain Euclidean `distance` would reject every genuinely-close
    # wrapped neighbor unless shifted back by one period first (a periodic spherical lon axis needs
    # no such shift — see `_build_footprint_scattered`'s identical point for the 2D case).
    is_cartesian = G <: Geometry.CartesianGeometry{T}
    period = ntuple(N) do d
        (is_cartesian && periodic[d]) ?
            (grid.axes[d][end] - grid.axes[d][1] + (grid.axes[d][2] - grid.axes[d][1])) : zero(T)
    end

    nbrs = NTuple{N,Int}[]
    w = T[]
    # Conservative upper bound (every target's full search window) — an exact count would need the
    # distance gate below, which is the expensive part this cache exists to avoid paying twice.
    sizehint!(nbrs, prod(dims) * prod(2 .* lim .+ 1))
    sizehint!(w, prod(dims) * prod(2 .* lim .+ 1))
    lin = LinearIndices(dims)
    ptr = Vector{Int}(undef, prod(dims) + 1)
    ptr[1] = 1
    for I in CartesianIndices(dims)
        t = lin[I]
        Ti = Tuple(I)
        target = Grids.coords(grid, Ti...)
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
            neighbor = Grids.coords(grid, J...)
            neighbor_shifted = is_cartesian ? (neighbor + SA.SVector{N,T}(shift)) : neighbor
            d = Geometry.distance(grid.geometry, target, neighbor_shifted)
            d <= rad || continue
            push!(nbrs, J)
            push!(w, Kernels.kernel_weight(kernel, d, scale) * Grids.area(grid, J...))
        end
        ptr[t+1] = length(nbrs) + 1
    end
    return FilterFootprintNDScattered(nbrs, w, ptr)
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
    grid::Grids.StructuredGrid{G,T,N},
    fp::FilterFootprintND{N,T},
    strategy::AbstractMaskStrategy,
) where {N, T<:AbstractFloat, G}
    dims = Grids.size_tuple(grid)
    periodic = grid.periodic
    mask = grid.mask
    fill!(out, zero(T))
    @inbounds for I in CartesianIndices(out)
        mask[I] || continue
        out[I] = _footprint_nd_point(field, fp, strategy, dims, periodic, mask, I)
    end
    return out
end

@inline function _footprint_nd_point(
    field::AbstractArray, fp::FilterFootprintNDScattered{N,T}, strategy::AbstractMaskStrategy,
    mask, lin::LinearIndices{N}, I::CartesianIndex{N},
) where {N, T<:AbstractFloat}
    t = lin[I]
    lo = fp.ptr[t]
    hi = fp.ptr[t+1] - 1
    ws = zero(T)
    wn = zero(T)
    @inbounds for k in lo:hi
        J = fp.nbrs[k]
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
    grid::Grids.StructuredGrid{G,T,N},
    fp::FilterFootprintNDScattered{N,T},
    strategy::AbstractMaskStrategy,
) where {N, T<:AbstractFloat, G}
    dims = Grids.size_tuple(grid)
    mask = grid.mask
    lin = LinearIndices(dims)
    fill!(out, zero(T))
    @inbounds for I in CartesianIndices(out)
        mask[I] || continue
        out[I] = _footprint_nd_point(field, fp, strategy, mask, lin, I)
    end
    return out
end

"Serial filter: build the footprint once, then convolve (2D per-row engine, or general n-D engine)."
function serial_filter_field!(
    out::AbstractArray{T},
    field::AbstractArray,
    grid::Grids.StructuredGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T,
    strategy::AbstractMaskStrategy,
    workspace,
) where {T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}}
    return _apply_serial!(out, field, grid, build_footprint(grid, kernel, scale), strategy)
end

"Serial filter on a curvilinear grid: build the scattered per-point footprint once, then convolve."
function serial_filter_field!(
    out::AbstractMatrix{T},
    field::AbstractMatrix,
    grid::Grids.CurvilinearGrid{T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T,
    strategy::AbstractMaskStrategy,
    workspace,
) where {T<:AbstractFloat}
    return _apply_serial!(out, field, grid, build_footprint(grid, kernel, scale), strategy)
end

# Dispatch the apply on the footprint kind.
_apply_serial!(out, field, grid, fp::FilterFootprint, strategy) =
    apply_footprint!(out, field, grid, fp, strategy, Grids.isperiodic(grid, 1))
_apply_serial!(out, field, grid, fp::FilterFootprintScattered, strategy) =
    apply_footprint!(out, field, grid, fp, strategy, Grids.isperiodic(grid, 1))
_apply_serial!(out, field, grid, fp::FilterFootprintND, strategy) =
    apply_footprint_nd!(out, field, grid, fp, strategy)
_apply_serial!(out, field, grid, fp::FilterFootprintNDScattered, strategy) =
    apply_footprint_nd!(out, field, grid, fp, strategy)

# ---------------------------------------------------------------------------
# Reusable filter plans: build the footprint ONCE, apply to many fields/scales
# ---------------------------------------------------------------------------

"""
    AbstractFilterPlan

A prebuilt filter (grid + kernel + scale + mask strategy + backend) that can be applied to many
fields without redoing setup. Physical-space backends precompute a `FilterFootprint`; spectral
backends (FFTW/FINUFFT/SHT extensions, Phase 5) hold cached transform plans.
"""
abstract type AbstractFilterPlan end

"""
Physical-space plan: a precomputed footprint reused across all longitudes, fields, and layers — for
EVERY backend, not just serial. `kernel`/`scale` are retained only so the cached-footprint path can
still call each backend's row-parallel hook (which takes them positionally); they're not used to
rebuild the footprint once `footprint` is already built.
"""
struct PhysicalFilterPlan{FP, G<:Grids.AbstractGrid, S<:AbstractMaskStrategy, K<:Kernels.AbstractFilterKernel, T<:AbstractFloat, B<:Backends.AbstractExecutionBackend} <: AbstractFilterPlan
    footprint::FP   # FilterFootprint (2D structured), FilterFootprintND (1D/3D), or FilterFootprintScattered (nonuniform/curvilinear)
    grid::G
    strategy::S
    kernel::K
    scale::T
    backend::B
end

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
    plan_filter(grid, kernel, scale; mask_strategy=Deformable(), backend=AutoBackend()) -> AbstractFilterPlan

Build a reusable filter plan: the footprint is precomputed ONCE regardless of backend (serial,
threaded, distributed, GPU, or MPI) and reused across every subsequent `filter_apply!` call — no
backend rebuilds it per call. Apply with `filter_apply!(out, field, plan)`.
"""
function plan_filter(
    grid::Grids.StructuredGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = Deformable(),
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    method::AbstractFilterMethod = DirectSum(),
) where {G<:Geometry.AbstractGeometry{T}} where {T<:AbstractFloat}
    _validate_scale(scale)
    if method isa Spectral
        return spectral_filter_plan(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend)
    end
    resolved = Backends.resolve_backend(backend)
    _check_backend_compatible(grid, backend)
    fp = build_footprint(grid, kernel, scale)
    return PhysicalFilterPlan(fp, grid, mask_strategy, kernel, scale, resolved)
end

# The row-based parallel backends (Threaded/Distributed/GPU/MPI) decompose over rows of a 2D grid
# via `apply_footprint_row!`, which already works generically for CurvilinearGrid (a 2D grid using
# the scattered per-point footprint) as well as StructuredGrid.
_row_parallelizable(::Grids.StructuredGrid{G,T,2}) where {G,T} = true
_row_parallelizable(::Grids.CurvilinearGrid) = true
_row_parallelizable(::Grids.AbstractGrid) = false

# 1D/true-3D StructuredGrid use a DIFFERENT footprint representation (`FilterFootprintND`/
# `FilterFootprintNDScattered`, point-indexed via `CartesianIndices` rather than row-indexed) — but
# each output point is still fully independent of every other (reads neighbours, writes only its own
# cell), so a per-point-parallel Threaded hook is just as valid as the row-parallel one, only over a
# different iteration space. Only Threaded is implemented this way so far (see
# `CoarseGrainingEnergyFluxesOhMyThreadsExt`); Distributed/GPU/MPI still need a real domain
# decomposition (halo exchange or similar) for the ND case, which is a separate, larger effort.
_nd_parallelizable(::Grids.StructuredGrid{G,T,1}) where {G,T} = true
_nd_parallelizable(::Grids.StructuredGrid{G,T,3}) where {G,T} = true
_nd_parallelizable(::Grids.AbstractGrid) = false

# Whether `grid` can actually honor a specific concrete backend request.
_backend_supported(grid::Grids.AbstractGrid, ::Backends.SerialBackend) = true
_backend_supported(grid::Grids.AbstractGrid, ::Backends.ThreadedBackend) = _row_parallelizable(grid) || _nd_parallelizable(grid)
_backend_supported(grid::Grids.AbstractGrid, ::Backends.DistributedBackend) = _row_parallelizable(grid)
_backend_supported(grid::Grids.AbstractGrid, ::Backends.GPUBackend) = _row_parallelizable(grid)
_backend_supported(grid::Grids.AbstractGrid, ::Backends.MPIBackend) = _row_parallelizable(grid)

# `AutoBackend()` (the default) silently landing on serial for a grid/backend combination with no
# parallel hook is correct auto-selection, not a fallback — nothing specific was asked for, so
# nothing was overridden. But an EXPLICIT non-serial backend request that can't be honored is a real
# mismatch between what the caller asked for and what they'd get — a hard error, not a silently
# downgraded warning (a warning can go unread; the caller's code would keep running as if it got the
# parallelism it asked for). Checked against the ORIGINAL `backend` argument, before `AutoBackend()`
# gets resolved away, so this distinction is still visible.
function _check_backend_compatible(grid::Grids.AbstractGrid, backend::Backends.AbstractExecutionBackend)
    if !(backend isa Backends.AutoBackend) && !(backend isa Backends.SerialBackend) && !_backend_supported(grid, Backends.resolve_backend(backend))
        throw(ArgumentError(
            "backend = $(typeof(backend)) was requested explicitly, but $(typeof(grid)) has no " *
            "matching parallel hook for it — there is no way to honor this request. Pass " *
            "`backend = SerialBackend()` explicitly if serial execution is acceptable, or " *
            "`backend = AutoBackend()` to let the library choose.",
        ))
    end
    return nothing
end

# Non-structured grids (curvilinear / scattered) have no real-space footprint engine; their only
# filtering path is a transform-backed spectral plan (FINUFFT / NUFSHT extensions). Structured grids
# use the more specific method above, so this never shadows it.
function plan_filter(
    grid::Grids.AbstractGrid,
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = Deformable(),
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    method::AbstractFilterMethod = Spectral(),
) where {T<:AbstractFloat}
    _validate_scale(scale)
    method isa Spectral || throw(ArgumentError(
        "Only spectral filtering (`method = Spectral()`) is available for $(typeof(grid)); the " *
        "real-space direct-sum engine requires a StructuredGrid or CurvilinearGrid.",
    ))
    return spectral_filter_plan(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend)
end

# Curvilinear grids have a genuine real-space direct-sum engine (the scattered per-point footprint),
# so — unlike the unstructured/spectral-only fallback above — they precompute a `PhysicalFilterPlan`.
# More specific than the `AbstractGrid` method, so it is chosen for a `CurvilinearGrid`.
function plan_filter(
    grid::Grids.CurvilinearGrid{T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = Deformable(),
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    method::AbstractFilterMethod = DirectSum(),
) where {T<:AbstractFloat}
    _validate_scale(scale)
    if method isa Spectral
        # No spectral backend targets a CurvilinearGrid (FINUFFT/NUFSHT are UnstructuredGrid-only),
        # so this raises the standard informative "spectral unavailable" error.
        return spectral_filter_plan(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend)
    end
    resolved = Backends.resolve_backend(backend)
    _check_backend_compatible(grid, backend)
    fp = build_footprint(grid, kernel, scale)
    return PhysicalFilterPlan(fp, grid, mask_strategy, kernel, scale, resolved)
end

"""
    filter_apply!(out, field, plan) -> out

Apply a prebuilt [`plan_filter`](@ref) to a single 2D field, dispatching to whichever backend the
plan was built for — the footprint is ALWAYS the one cached in `plan`, never rebuilt here, for every
backend (serial, threaded, distributed, GPU, MPI).
"""
function filter_apply!(out::AbstractArray, field::AbstractArray, plan::PhysicalFilterPlan)
    if plan.backend isa Backends.SerialBackend || !_backend_supported(plan.grid, plan.backend)
        return _apply_serial!(out, field, plan.grid, plan.footprint, plan.strategy)
    elseif plan.backend isa Backends.ThreadedBackend
        return threaded_filter_field!(out, field, plan.grid, plan.kernel, plan.scale, plan.strategy, plan.footprint)
    elseif plan.backend isa Backends.DistributedBackend
        return distributed_filter_field!(out, field, plan.grid, plan.kernel, plan.scale, plan.strategy, plan.footprint)
    elseif plan.backend isa Backends.GPUBackend
        return gpu_filter_field!(plan.backend, out, field, plan.grid, plan.kernel, plan.scale, plan.strategy, plan.footprint)
    elseif plan.backend isa Backends.MPIBackend
        return mpi_filter_field!(out, field, plan.grid, plan.kernel, plan.scale, plan.strategy, plan.footprint)
    else
        throw(ArgumentError("Unsupported backend: $(typeof(plan.backend))"))
    end
end

"""
    filter_fields!(outs, fields, grid, kernel, scale; mask_strategy=Deformable(), backend=AutoBackend())

Filter several fields that share the same grid/kernel/scale, building the footprint/plan ONCE.
`outs` and `fields` are iterables of matching 2D arrays (e.g. tuples of velocity components).
"""
function filter_fields!(
    outs,
    fields,
    grid::Grids.StructuredGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = Deformable(),
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
) where {G<:Geometry.AbstractGeometry{T}} where {T<:AbstractFloat}
    plan = plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend)
    for (out, field) in zip(outs, fields)
        filter_apply!(out, field, plan)
    end
    return outs
end

end # module
