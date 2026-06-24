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
# Land-masking strategy (singleton types — specializable, unlike Symbol dispatch)
# ---------------------------------------------------------------------------

"""
    AbstractMaskStrategy

How land/dry cells enter the filter normalization.
"""
abstract type AbstractMaskStrategy end

"""
    ZeroFill <: AbstractMaskStrategy

Dry cells are treated as zero-valued water: they contribute to the denominator (kernel weight) but
zero to the numerator. The kernel is homogeneous (same shape everywhere), which preserves domain
averages and commutation with derivatives (the Storer 2022 / Aluie 2019 "fixed kernel" mode).
"""
struct ZeroFill <: AbstractMaskStrategy end

"""
    Deformable <: AbstractMaskStrategy

Dry cells are excluded from BOTH numerator and denominator, so the kernel is renormalized over the
local water area only ("deformable kernel"). Land is genuinely excluded, but the kernel becomes
inhomogeneous near coasts (breaks the strict commutation theorems).
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

function finufft_filter_field!(args...; kwargs...)
    throw(ArgumentError("FINUFFT filtering is unavailable — run `using FINUFFT`."))
end

# Build a spectral filter plan (overridden by FFTW / FastSphericalHarmonics / FINUFFT / NUFSHT
# extensions, dispatched on the grid geometry). Errors until a compatible extension is loaded.
function spectral_filter_plan(grid, kernel, scale; kwargs...)
    throw(ArgumentError(
        "Spectral filtering is unavailable for $(typeof(grid)) — load a spectral backend " *
        "(e.g. `using FFTW` for a uniform periodic Cartesian grid).",
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
- `mask_strategy::AbstractMaskStrategy=Deformable()`: land masking — `ZeroFill()` (dry cells count
  in the denominator as zero water; homogeneous kernel) or `Deformable()` (dry cells excluded from
  numerator and denominator; renormalized over local water).
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

    # 2. Dispatch to the appropriate execution backend (extensions override the hook functions).
    if resolved isa Backends.SerialBackend
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
struct FilterFootprint{T<:AbstractFloat}
    di::Vector{Int}    # longitude index offset
    dj::Vector{Int}    # latitude index offset
    w::Vector{T}       # kernel_weight(distance) * cell area
    ptr::Vector{Int}   # band b's entries: ptr[b]:ptr[b+1]-1
    nbands::Int        # 1 (Cartesian) or Nlat (spherical)
end

@inline _band(::FilterFootprint, ::Grids.StructuredGrid{<:Geometry.CartesianGeometry}, j::Integer) = 1
@inline _band(::FilterFootprint, ::Grids.StructuredGrid{<:Geometry.SphericalGeometry}, j::Integer) = j

"""
    FilterFootprintND{N, T}

General N-dimensional footprint: in-support neighbour offsets (`NTuple{N,Int}`) and their geometric
weights `w = kernel_weight(distance) · cell_measure`. Used for 1D and 3D (Cartesian,
translation-invariant ⇒ a single offset set); the 2D path uses the optimized per-row
`FilterFootprint`.
"""
struct FilterFootprintND{N, T<:AbstractFloat}
    offsets::Vector{NTuple{N,Int}}
    w::Vector{T}
end

"""
    build_footprint(grid, kernel, scale) -> FilterFootprint

Precompute the (type-stable) convolution footprint once; reusable across all longitudes, scales'
worth of fields, and depth layers.
"""
function build_footprint(
    grid::Grids.StructuredGrid{G,T,2},
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
        dx = grid.geometry.dx
        dy = grid.geometry.dy
        A = Grids.area(grid, 1, 1)   # uniform Cartesian cell area
        di_lim = dx > 0 ? ceil(Int, rad / dx) : 0
        dj_lim = dy > 0 ? ceil(Int, rad / dy) : 0
        for ddj in -dj_lim:dj_lim, ddi in -di_lim:di_lim
            d = sqrt((ddi * dx)^2 + (ddj * dy)^2)
            if d <= rad
                push!(di, ddi)
                push!(dj, ddj)
                push!(w, Kernels.kernel_weight(kernel, T(d), scale) * A)
            end
        end
        push!(ptr, length(di) + 1)
        return FilterFootprint{T}(di, dj, w, ptr, 1)
    else
        R = grid.geometry.R
        dλ = Nlon > 1 ? grid.lon[2] - grid.lon[1] : zero(T)
        dφ = Nlat > 1 ? grid.lat[2] - grid.lat[1] : zero(T)
        dj_lim = dφ > 0 ? ceil(Int, rad / (R * dφ)) : 0
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
        return FilterFootprint{T}(di, dj, w, ptr, Nlat)
    end
end

"""
    apply_footprint!(out, field, grid, fp, strategy, periodic_lon)

Convolve `field` with a precomputed `fp` into `out`, applying the land-mask `strategy`. `out` and
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
distributed backends. Callers must `fill!(out, 0)` first (dry cells are left untouched here).
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
        Grids.iswet(grid, i, j) || continue
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
            wet = Grids.iswet(grid, ii, jj)
            w = fp.w[k]
            if strategy isa ZeroFill
                # Dry cells count in the denominator (as zero water).
                weight_norm += w
                wet && (weighted_sum += w * field[ii, jj])
            else
                # Deformable: dry cells excluded from numerator AND denominator.
                wet || continue
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

@inline _cartesian_spacing(geo::Geometry.CartesianGeometry, d::Integer) = d == 1 ? geo.dx : (d == 2 ? geo.dy : geo.dz)

# Dispatched on N = 1 and N = 3 so it never collides with the N = 2 build_footprint above.
build_footprint(grid::Grids.StructuredGrid{G,T,1}, kernel::Kernels.AbstractFilterKernel, scale::T) where {T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}} =
    _build_footprint_nd(grid, kernel, scale)
build_footprint(grid::Grids.StructuredGrid{G,T,3}, kernel::Kernels.AbstractFilterKernel, scale::T) where {T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}} =
    _build_footprint_nd(grid, kernel, scale)

function _build_footprint_nd(
    grid::Grids.StructuredGrid{G,T,N},
    kernel::Kernels.AbstractFilterKernel,
    scale::T,
) where {N, T<:AbstractFloat, G<:Geometry.CartesianGeometry{T}}
    rad = Kernels.kernel_radius(kernel, scale)
    spacing = ntuple(d -> _cartesian_spacing(grid.geometry, d), N)
    A = grid.measure[ntuple(_ -> 1, N)...]   # uniform Cartesian cell measure
    lim = ntuple(d -> spacing[d] > 0 ? ceil(Int, rad / spacing[d]) : 0, N)
    offsets = NTuple{N,Int}[]
    w = T[]
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
    return FilterFootprintND{N,T}(offsets, w)
end

# Shifted neighbour multi-index with per-axis periodic wrap; returns (index, in-bounds?).
@inline function _shift_index(I::NTuple{N,Int}, o::NTuple{N,Int}, dims::NTuple{N,Int}, periodic::NTuple{N,Bool}) where {N}
    J = ntuple(N) do d
        jj = I[d] + o[d]
        (jj < 1 || jj > dims[d]) ? (periodic[d] ? mod1(jj, dims[d]) : 0) : jj
    end
    return J, !any(==(0), J)
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
        Ti = Tuple(I)
        ws = zero(T)
        wn = zero(T)
        for k in eachindex(fp.offsets)
            J, valid = _shift_index(Ti, fp.offsets[k], dims, periodic)
            valid || continue
            wet = mask[J...]
            wk = fp.w[k]
            if strategy isa ZeroFill
                wn += wk
                wet && (ws += wk * field[J...])
            elseif wet
                wn += wk
                ws += wk * field[J...]
            end
        end
        out[I] = wn > T(1e-15) ? ws / wn : zero(T)
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

# Dispatch the apply on the footprint kind.
_apply_serial!(out, field, grid, fp::FilterFootprint, strategy) =
    apply_footprint!(out, field, grid, fp, strategy, Grids.isperiodic(grid, 1))
_apply_serial!(out, field, grid, fp::FilterFootprintND, strategy) =
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

"Physical-space plan: a precomputed footprint reused across all longitudes, fields, and layers."
struct PhysicalFilterPlan{FP, G<:Grids.StructuredGrid, S<:AbstractMaskStrategy} <: AbstractFilterPlan
    footprint::FP   # FilterFootprint (2D) or FilterFootprintND (1D/3D)
    grid::G
    strategy::S
end

"Fallback plan for backends without a precomputed footprint yet — defers to `filter_field!`."
struct FallbackFilterPlan{G<:Grids.AbstractGrid, K<:Kernels.AbstractFilterKernel, T<:AbstractFloat, S<:AbstractMaskStrategy, B<:Backends.AbstractExecutionBackend} <: AbstractFilterPlan
    grid::G
    kernel::K
    scale::T
    strategy::S
    backend::B
end

"""
    plan_filter(grid, kernel, scale; mask_strategy=Deformable(), backend=AutoBackend()) -> AbstractFilterPlan

Build a reusable filter plan. For serial CPU execution this precomputes the footprint once (reused
across every field and depth layer); other backends get a deferring fallback plan (optimized in
later phases). Apply with `filter_apply!(out, field, plan)`.
"""
function plan_filter(
    grid::Grids.StructuredGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = Deformable(),
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    method::AbstractFilterMethod = DirectSum(),
) where {G<:Geometry.AbstractGeometry{T}} where {T<:AbstractFloat}
    if method isa Spectral
        return spectral_filter_plan(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend)
    end
    resolved = Backends.resolve_backend(backend)
    if resolved isa Backends.SerialBackend
        fp = build_footprint(grid, kernel, scale)
        return PhysicalFilterPlan(fp, grid, mask_strategy)
    else
        return FallbackFilterPlan(grid, kernel, scale, mask_strategy, resolved)
    end
end

"""
    filter_apply!(out, field, plan) -> out

Apply a prebuilt [`plan_filter`](@ref) to a single 2D field.
"""
filter_apply!(out::AbstractArray, field::AbstractArray, plan::PhysicalFilterPlan) =
    _apply_serial!(out, field, plan.grid, plan.footprint, plan.strategy)

filter_apply!(out::AbstractArray, field::AbstractArray, plan::FallbackFilterPlan) =
    filter_field!(out, field, plan.grid, plan.kernel, plan.scale; mask_strategy = plan.strategy, backend = plan.backend)

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
