module Grids

using ..Geometry: Geometry
using StaticArrays: StaticArrays as SA

export AbstractGrid, StructuredGrid, CurvilinearGrid, UnstructuredGrid
export coords, area, iswet, grid_geometry, size_tuple, isperiodic

"""
    AbstractGrid{G<:AbstractGeometry, T<:AbstractFloat}

Abstract supertype for all grid architectures (structured, curvilinear, unstructured).
"""
abstract type AbstractGrid{G<:Geometry.AbstractGeometry, T<:AbstractFloat} end

# Common interface queries
grid_geometry(grid::AbstractGrid) = grid.geometry

"""
    isperiodic(grid, dim) -> Bool

Whether axis `dim` (1 = lon/x, 2 = lat/y) is periodic, i.e. the filter footprint should wrap
across that boundary. Non-periodic by default; `StructuredGrid` carries explicit per-axis flags.
"""
isperiodic(::AbstractGrid, ::Integer) = false

# ---------------------------------------------------------------------------
# Structured Grid
# ---------------------------------------------------------------------------

"""
    StructuredGrid{G, T, N, V, AT, BT}

Structured (rectilinear) `N`-dimensional grid: one coordinate vector per axis (`axes`), an N-D cell
`measure` (length in 1D, area in 2D, volume in 3D), an N-D active `mask`, and per-axis `periodic`
flags. `N = 1, 2, 3`.

The first two axes carry the convenience aliases `grid.lon` / `grid.lat` (= `axes[1]` / `axes[2]`),
and `grid.areas` aliases `grid.measure`, so 2D code reads naturally.
"""
struct StructuredGrid{
    G<:Geometry.AbstractGeometry,
    T<:AbstractFloat,
    N,
    V<:AbstractVector{T},
    AT<:AbstractArray{T,N},
    BT<:AbstractArray{Bool,N},
} <: AbstractGrid{G, T}
    geometry::G
    axes::NTuple{N,V}        # coordinate vector per axis (axes[1]=lon/x, axes[2]=lat/y, axes[3]=z)
    measure::AT             # N-D cell measure (area in 2D, volume in 3D)
    mask::BT                # N-D active mask (true=water, false=land)
    periodic::NTuple{N,Bool} # per-axis periodicity for footprint wrapping
end

# Convenience field aliases: lon/lat for the first two axes, areas ≡ measure (keeps 2D code/users
# working). `getfield` is used internally to avoid recursion.
@inline function Base.getproperty(grid::StructuredGrid, name::Symbol)
    if name === :lon
        return @inbounds getfield(grid, :axes)[1]
    elseif name === :lat
        return @inbounds getfield(grid, :axes)[2]
    elseif name === :areas
        return getfield(grid, :measure)
    else
        return getfield(grid, name)
    end
end

size_tuple(grid::StructuredGrid) = size(grid.mask)

@inline isperiodic(grid::StructuredGrid, dim::Integer) = grid.periodic[dim]

# N-D accessors (a 2-index call on a 2D grid is the common case).
@inline function coords(grid::StructuredGrid{G,T,N}, I::Vararg{Integer,N}) where {G,T,N}
    return SA.SVector{N,T}(ntuple(d -> @inbounds(grid.axes[d][I[d]]), N))
end
@inline area(grid::StructuredGrid{G,T,N}, I::Vararg{Integer,N}) where {G,T,N} = grid.measure[I...]
@inline iswet(grid::StructuredGrid{G,T,N}, I::Vararg{Integer,N}) where {G,T,N} = grid.mask[I...]

# Auto-detect longitude periodicity: a spherical lon axis that closes the full 2π circle (to
# within one cell) is periodic; a regional lon span is NOT. Cartesian axes are opt-in only.
_auto_periodic_lon(::Geometry.CartesianGeometry, lon::AbstractVector) = false
function _auto_periodic_lon(::Geometry.SphericalGeometry, lon::AbstractVector{T}) where {T<:AbstractFloat}
    length(lon) > 2 || return false
    dλ = lon[2] - lon[1]
    iszero(dλ) && return false
    return isapprox(lon[end] - lon[1] + dλ, T(2π); atol = abs(dλ))
end

_periodic_tuple(p::NTuple{2,Bool}) = p
_periodic_tuple(p::Bool) = (p, false)
_periodic_tuple(p::Tuple{Bool}) = (p[1], false)

"""
    StructuredGrid(geometry, lon, lat, mask; periodic = nothing)

Build a structured (rectilinear) grid, pre-computing cell areas from the geometry.

`periodic` controls footprint wrapping per axis (`(lon, lat)`); pass a `Bool` (applied to lon) or
an `NTuple{2,Bool}`. When omitted, longitude periodicity is auto-detected — a spherical lon axis
spanning the full circle is treated as periodic, a regional span is not — and latitude is
non-periodic.
"""
function StructuredGrid(
    geometry::G,
    lon::AbstractVector,
    lat::AbstractVector,
    mask::AbstractMatrix{Bool};
    periodic = nothing,
) where {
    T<:AbstractFloat,
    G<:Geometry.AbstractGeometry{T}
}
    # Convert lon and lat vectors to the geometry float type T
    lon_T = convert(Vector{T}, lon)
    lat_T = convert(Vector{T}, lat)

    Nlon = length(lon_T)
    Nlat = length(lat_T)

    # Pre-allocate areas matrix
    areas = Matrix{T}(undef, Nlon, Nlat)

    # Populate cell areas
    if G <: Geometry.CartesianGeometry{T}
        # Cartesian cells are uniform
        A = Geometry.area_element(geometry)
        fill!(areas, A)
    else
        # Spherical cell area varies with latitude
        # Assuming lon/lat coordinates are cell centers, we estimate dλ and dφ
        # If coordinates are uniform, we can calculate standard dλ, dφ
        dλ = Nlon > 1 ? lon_T[2] - lon_T[1] : T(0)
        dφ = Nlat > 1 ? lat_T[2] - lat_T[1] : T(0)

        for j in 1:Nlat
            A_lat = Geometry.area_element(geometry, lat_T[j], dλ, dφ)
            for i in 1:Nlon
                areas[i, j] = A_lat
            end
        end
    end

    per = periodic === nothing ? (_auto_periodic_lon(geometry, lon_T), false) : _periodic_tuple(periodic)

    return StructuredGrid{G, T, 2, typeof(lon_T), typeof(areas), typeof(mask)}(
        geometry, (lon_T, lat_T), areas, mask, per,
    )
end

"""
    StructuredGrid(geometry, x, mask; periodic = false)

Build a 1D Cartesian grid (cell length `dx`). `periodic` is a `Bool` (default `false`).
"""
function StructuredGrid(
    geometry::G,
    x::AbstractVector,
    mask::AbstractVector{Bool};
    periodic::Bool = false,
) where {T<:AbstractFloat, G<:Geometry.CartesianGeometry{T}}
    x_T = convert(Vector{T}, x)
    measure = fill(geometry.dx, length(x_T))
    return StructuredGrid{G, T, 1, typeof(x_T), typeof(measure), typeof(mask)}(
        geometry, (x_T,), measure, mask, (periodic,),
    )
end

"""
    StructuredGrid(geometry, x, y, z, mask; periodic = (false, false, false))

Build a 3D Cartesian grid (cell volume `dx·dy·dz`; the geometry must carry a non-zero `dz`).
`periodic` is a `Bool` (applied to x) or an `NTuple{3,Bool}`.
"""
function StructuredGrid(
    geometry::G,
    x::AbstractVector,
    y::AbstractVector,
    z::AbstractVector,
    mask::AbstractArray{Bool,3};
    periodic = (false, false, false),
) where {T<:AbstractFloat, G<:Geometry.CartesianGeometry{T}}
    x_T = convert(Vector{T}, x)
    y_T = convert(Vector{T}, y)
    z_T = convert(Vector{T}, z)
    measure = fill(geometry.dx * geometry.dy * geometry.dz, size(mask))
    per = periodic isa Bool ? (periodic, false, false) : NTuple{3,Bool}(periodic)
    return StructuredGrid{G, T, 3, typeof(x_T), typeof(measure), typeof(mask)}(
        geometry, (x_T, y_T, z_T), measure, mask, per,
    )
end

# ---------------------------------------------------------------------------
# Curvilinear Grid
# ---------------------------------------------------------------------------

"""
    CurvilinearGrid{G, T, M, B}

Curvilinear grid where coordinates are 2D arrays (e.g. ROMS / WCOFS / Orthogonal Curvilinear).
"""
struct CurvilinearGrid{
    G<:Geometry.AbstractGeometry,
    T<:AbstractFloat,
    M<:AbstractMatrix{T},
    B<:AbstractMatrix{Bool}
} <: AbstractGrid{G, T}
    geometry::G
    lon::M       # X/λ coordinate array (Nlon × Nlat)
    lat::M       # Y/φ coordinate array (Nlon × Nlat)
    areas::M     # pre-computed cell areas (Nlon × Nlat)
    mask::B      # active mask (true=water, false=land)
end

size_tuple(grid::CurvilinearGrid) = size(grid.mask)

@inline function coords(grid::CurvilinearGrid{G,T}, i::Integer, j::Integer) where {G,T}
    return SA.SVector{2,T}(grid.lon[i, j], grid.lat[i, j])
end

@inline area(grid::CurvilinearGrid, i::Integer, j::Integer) = grid.areas[i, j]
@inline iswet(grid::CurvilinearGrid, i::Integer, j::Integer) = grid.mask[i, j]

# ---------------------------------------------------------------------------
# Unstructured Grid
# ---------------------------------------------------------------------------

"""
    UnstructuredGrid{G, T, V, B}

Unstructured mesh (e.g. radial data, finite volume, or triangular mesh) where coords are 1D vectors.
"""
struct UnstructuredGrid{
    G<:Geometry.AbstractGeometry,
    T<:AbstractFloat,
    V<:AbstractVector{T},
    B<:AbstractVector{Bool}
} <: AbstractGrid{G, T}
    geometry::G
    lon::V       # X/λ vector for each node (Nnodes)
    lat::V       # Y/φ vector for each node (Nnodes)
    areas::V     # Area of each grid cell / control volume (Nnodes)
    mask::B      # active mask (true=water, false=land) (Nnodes)
    neighbors::Vector{Vector{Int}} # Adjacency mapping for unstructured neighbor lookups
end

size_tuple(grid::UnstructuredGrid) = (length(grid.mask),)

@inline function coords(grid::UnstructuredGrid{G,T}, idx::Integer) where {G,T}
    return SA.SVector{2,T}(grid.lon[idx], grid.lat[idx])
end

@inline area(grid::UnstructuredGrid, idx::Integer) = grid.areas[idx]
@inline iswet(grid::UnstructuredGrid, idx::Integer) = grid.mask[idx]

end # module
