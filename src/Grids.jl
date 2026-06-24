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
    StructuredGrid{G, T, V, M, B}

Structured grid where coordinates are 1D vectors along each axis (e.g. regular latitude-longitude).
"""
struct StructuredGrid{
    G<:Geometry.AbstractGeometry,
    T<:AbstractFloat,
    V<:AbstractVector{T},
    M<:AbstractMatrix{T},
    B<:AbstractMatrix{Bool}
} <: AbstractGrid{G, T}
    geometry::G
    lon::V                  # 1D coordinate vector along X/λ
    lat::V                  # 1D coordinate vector along Y/φ
    areas::M                # 2D cell areas (Nlon × Nlat)
    mask::B                 # 2D active mask (true=water, false=land)
    periodic::NTuple{2,Bool} # per-axis periodicity (lon/x, lat/y) for footprint wrapping
end

size_tuple(grid::StructuredGrid) = size(grid.mask)

@inline function coords(grid::StructuredGrid{G,T}, i::Integer, j::Integer) where {G,T}
    return SA.SVector{2,T}(grid.lon[i], grid.lat[j])
end

@inline area(grid::StructuredGrid, i::Integer, j::Integer) = grid.areas[i, j]
@inline iswet(grid::StructuredGrid, i::Integer, j::Integer) = grid.mask[i, j]
@inline isperiodic(grid::StructuredGrid, dim::Integer) = grid.periodic[dim]

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

    return StructuredGrid{G, T, typeof(lon_T), typeof(areas), typeof(mask)}(
        geometry, lon_T, lat_T, areas, mask, per,
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
