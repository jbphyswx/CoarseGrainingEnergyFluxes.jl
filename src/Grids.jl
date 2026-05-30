module Grids

using ..Geometry
using StaticArrays

export AbstractGrid, StructuredGrid, CurvilinearGrid, UnstructuredGrid
export coords, area, iswet, grid_geometry, size_tuple

"""
    AbstractGrid{G<:AbstractGeometry, T<:AbstractFloat}

Abstract supertype for all grid architectures (structured, curvilinear, unstructured).
"""
abstract type AbstractGrid{G<:AbstractGeometry, T<:AbstractFloat} end

# Common interface queries
grid_geometry(grid::AbstractGrid) = grid.geometry

# ---------------------------------------------------------------------------
# Structured Grid
# ---------------------------------------------------------------------------

"""
    StructuredGrid{G, T, V, M, B}

Structured grid where coordinates are 1D vectors along each axis (e.g. regular latitude-longitude).
"""
struct StructuredGrid{
    G<:AbstractGeometry, 
    T<:AbstractFloat, 
    V<:AbstractVector{T}, 
    M<:AbstractMatrix{T}, 
    B<:AbstractMatrix{Bool}
} <: AbstractGrid{G, T}
    geometry::G
    lon::V       # 1D coordinate vector along X/λ
    lat::V       # 1D coordinate vector along Y/φ
    areas::M     # 2D cell areas (Nlon × Nlat) or (Nlat × Nlon)
    mask::B      # 2D active mask (true=water, false=land)
end

size_tuple(grid::StructuredGrid) = size(grid.mask)

@inline function coords(grid::StructuredGrid{G,T}, i::Integer, j::Integer) where {G,T}
    return SVector{2,T}(grid.lon[i], grid.lat[j])
end

@inline area(grid::StructuredGrid, i::Integer, j::Integer) = grid.areas[i, j]
@inline iswet(grid::StructuredGrid, i::Integer, j::Integer) = grid.mask[i, j]

# Helper constructor that pre-computes cell areas automatically from geometry and coordinates
function StructuredGrid(
    geometry::G,
    lon::V,
    lat::V,
    mask::B
) where {
    T<:AbstractFloat,
    G<:AbstractGeometry{T},
    V<:AbstractVector{T},
    B<:AbstractMatrix{Bool}
}
    Nlon = length(lon)
    Nlat = length(lat)
    
    # Pre-allocate areas matrix
    areas = Matrix{T}(undef, Nlon, Nlat)
    
    # Populate cell areas
    if G <: CartesianGeometry{T}
        # Cartesian cells are uniform
        A = area_element(geometry)
        fill!(areas, A)
    else
        # Spherical cell area varies with latitude
        # Assuming lon/lat coordinates are cell centers, we estimate dλ and dφ
        # If coordinates are uniform, we can calculate standard dλ, dφ
        dλ = Nlon > 1 ? lon[2] - lon[1] : T(0)
        dφ = Nlat > 1 ? lat[2] - lat[1] : T(0)
        
        for j in 1:Nlat
            A_lat = area_element(geometry, lat[j], dλ, dφ)
            for i in 1:Nlon
                areas[i, j] = A_lat
            end
        end
    end
    
    return StructuredGrid{G, T, V, typeof(areas), B}(geometry, lon, lat, areas, mask)
end

# ---------------------------------------------------------------------------
# Curvilinear Grid
# ---------------------------------------------------------------------------

"""
    CurvilinearGrid{G, T, M, B}

Curvilinear grid where coordinates are 2D arrays (e.g. ROMS / WCOFS / Orthogonal Curvilinear).
"""
struct CurvilinearGrid{
    G<:AbstractGeometry,
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
    return SVector{2,T}(grid.lon[i, j], grid.lat[j, j])
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
    G<:AbstractGeometry,
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
    return SVector{2,T}(grid.lon[idx], grid.lat[idx])
end

@inline area(grid::UnstructuredGrid, idx::Integer) = grid.areas[idx]
@inline iswet(grid::UnstructuredGrid, idx::Integer) = grid.mask[idx]

end # module
