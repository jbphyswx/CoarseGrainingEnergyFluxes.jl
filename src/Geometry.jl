module Geometry

using LinearAlgebra
using StaticArrays

export AbstractGeometry, CartesianGeometry, SphericalGeometry
export distance, area_element, to_planetary_cartesian, from_planetary_cartesian

"""
    AbstractGeometry{T<:AbstractFloat}

Abstract supertype for all coordinate systems and geometry metrics.

# Type Parameters
- `T`: Floating point type (Float32 or Float64) for coordinate calculations

# Implementations
- `CartesianGeometry{T}`: Cartesian coordinates with uniform grid spacing
- `SphericalGeometry{T}`: Spherical coordinates on a planet surface

# Examples
```julia
geom_cart = CartesianGeometry(1000.0, 1000.0)  # 1km x 1km grid
geom_sph = SphericalGeometry(6.371e6)          # Earth-like sphere
```
"""
abstract type AbstractGeometry{T<:AbstractFloat} end

"""
    CartesianGeometry{T<:AbstractFloat}

Represent Cartesian coordinates with grid spacings `dx`, `dy`, and optionally `dz`.

# Fields
- `dx::T`: Grid spacing in x-direction (meters)
- `dy::T`: Grid spacing in y-direction (meters)  
- `dz::T`: Grid spacing in z-direction (meters), zero for 2D grids

# Constructors
```julia
CartesianGeometry(dx, dy)        # 2D grid
CartesianGeometry(dx, dy, dz)    # 3D grid
```

# Examples
```julia
geom = CartesianGeometry(2000.0, 2000.0)  # 2km x 2km grid
area = area_element(geom)  # Returns 4e6 m²
```
"""
struct CartesianGeometry{T<:AbstractFloat} <: AbstractGeometry{T}
    dx::T
    dy::T
    dz::T
end

# Outer constructors for 2D Cartesian fallbacks
CartesianGeometry(dx::T, dy::T) where {T<:AbstractFloat} = CartesianGeometry{T}(dx, dy, zero(T))
CartesianGeometry{T}(dx, dy) where {T<:AbstractFloat} = CartesianGeometry{T}(convert(T, dx), convert(T, dy), zero(T))

"""
    SphericalGeometry{T<:AbstractFloat}

Represent spherical coordinates on a planet of radius `R`.

# Fields
- `R::T`: Planet radius in meters (default: 6.371e6 for Earth)

# Constructors
```julia
SphericalGeometry()               # Default Earth radius (6371 km)
SphericalGeometry(R)              # Custom radius in meters
SphericalGeometry{Float32}()        # Explicit type parameter
```

# Notes
- Uses Haversine formula for great-circle distance calculations
- Supports proper periodic boundary handling in longitude (0° ↔ 360°)
- Essential for global ocean/atmosphere calculations

# Examples
```julia
geom = SphericalGeometry()  # Earth-like sphere
p1 = SVector{2,Float64}(0.0, 0.0)  # (lon, lat) in radians
p2 = SVector{2,Float64}(0.0, π/2)  # North pole
d = distance(geom, p1, p2)  # Quarter circumference
```
"""
struct SphericalGeometry{T<:AbstractFloat} <: AbstractGeometry{T}
    R::T
end

SphericalGeometry() = SphericalGeometry(6.371e6)

# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------

"""
    distance(geo::AbstractGeometry, pt1, pt2)

Calculate the distance between two points in the given geometry.
- For `CartesianGeometry`, this is the Euclidean norm.
- For `SphericalGeometry`, this is the great-circle distance.
"""
@inline function distance(::CartesianGeometry{T}, pt1::SVector{N,T}, pt2::SVector{N,T}) where {N,T}
    return norm(pt1 - pt2)
end

@inline function distance(geo::SphericalGeometry{T}, coords1::SVector{2,T}, coords2::SVector{2,T}) where {T}
    # Haversine formula for great-circle distance (robust for both small and large distances)
    λ1, φ1 = coords1[1], coords1[2]
    λ2, φ2 = coords2[1], coords2[2]
    
    dλ = λ2 - λ1
    dφ = φ2 - φ1
    
    a = sin(dφ / T(2))^2 + cos(φ1) * cos(φ2) * sin(dλ / T(2))^2
    c = T(2) * atan(sqrt(a), sqrt(max(zero(T), one(T) - a)))
    return geo.R * c
end

# 3D spherical distance (with radial component)
@inline function distance(geo::SphericalGeometry{T}, coords1::SVector{3,T}, coords2::SVector{3,T}) where {T}
    # Transform spherical coords (λ, φ, r) to 3D planetary Cartesian first, then compute Euclidean distance
    p1 = spherical_to_planetary_position(geo, coords1)
    p2 = spherical_to_planetary_position(geo, coords2)
    return norm(p1 - p2)
end

# Helper to transform position coords (λ, φ, r) to Cartesian X, Y, Z
@inline function spherical_to_planetary_position(geo::SphericalGeometry{T}, coords::SVector{3,T}) where {T}
    λ, φ, r = coords[1], coords[2], coords[3]
    rad = geo.R + r
    X = rad * cos(φ) * cos(λ)
    Y = rad * cos(φ) * sin(λ)
    Z = rad * sin(φ)
    return SVector{3,T}(X, Y, Z)
end

@inline function spherical_to_planetary_position(geo::SphericalGeometry{T}, coords::SVector{2,T}) where {T}
    λ, φ = coords[1], coords[2]
    X = geo.R * cos(φ) * cos(λ)
    Y = geo.R * cos(φ) * sin(λ)
    Z = geo.R * sin(φ)
    return SVector{3,T}(X, Y, Z)
end

# ---------------------------------------------------------------------------
# Area Elements
# ---------------------------------------------------------------------------

"""
    area_element(geo::CartesianGeometry{T})
    area_element(geo::SphericalGeometry{T}, lat::T, dλ::T, dφ::T)

Compute local grid cell area.
"""
@inline area_element(geo::CartesianGeometry{T}) where {T} = geo.dx * geo.dy

@inline function area_element(geo::SphericalGeometry{T}, lat::T, dλ::T, dφ::T) where {T}
    return geo.R^2 * cos(lat) * dλ * dφ
end

# ---------------------------------------------------------------------------
# Coordinate Projections for Spherical Vector Fields
# ---------------------------------------------------------------------------

"""
    to_planetary_cartesian(geo::SphericalGeometry{T}, u_east, u_north, u_vertical, λ, φ)

Convert spherical local velocity components (East, North, Radial) into global planetary Cartesian X, Y, Z.
This is essential for spherical filtering to ensure commutativity with spatial derivatives.
"""
@inline function to_planetary_cartesian(
    geo::SphericalGeometry{T},
    u_east::Real,
    u_north::Real,
    u_vertical::Real,
    λ::Real,
    φ::Real
) where {T<:AbstractFloat}
    u_east_T = convert(T, u_east)
    u_north_T = convert(T, u_north)
    u_vertical_T = convert(T, u_vertical)
    λ_T = convert(T, λ)
    φ_T = convert(T, φ)
    
    sinλ, cosλ = sin(λ_T), cos(λ_T)
    sinφ, cosφ = sin(φ_T), cos(φ_T)
    
    ux = u_east_T * (-sinλ) + u_north_T * (-sinφ * cosλ) + u_vertical_T * (cosφ * cosλ)
    uy = u_east_T * (cosλ)  + u_north_T * (-sinφ * sinλ) + u_vertical_T * (cosφ * sinλ)
    uz =                      u_north_T * cosφ           + u_vertical_T * sinφ
    
    return SVector{3,T}(ux, uy, uz)
end

# 2D spherical version (assumes zero vertical velocity)
@inline function to_planetary_cartesian(
    geo::SphericalGeometry{T},
    u_east::Real,
    u_north::Real,
    λ::Real,
    φ::Real
) where {T<:AbstractFloat}
    return to_planetary_cartesian(geo, u_east, u_north, zero(T), λ, φ)
end

"""
    from_planetary_cartesian(geo::SphericalGeometry{T}, ux, uy, uz, λ, φ)

Convert global planetary Cartesian velocity components back to local East, North, Radial.
"""
@inline function from_planetary_cartesian(
    geo::SphericalGeometry{T},
    ux::Real,
    uy::Real,
    uz::Real,
    λ::Real,
    φ::Real
) where {T<:AbstractFloat}
    ux_T = convert(T, ux)
    uy_T = convert(T, uy)
    uz_T = convert(T, uz)
    λ_T = convert(T, λ)
    φ_T = convert(T, φ)
    
    sinλ, cosλ = sin(λ_T), cos(λ_T)
    sinφ, cosφ = sin(φ_T), cos(φ_T)
    
    u_east     = ux_T * (-sinλ) + uy_T * cosλ
    u_north    = ux_T * (-sinφ * cosλ) + uy_T * (-sinφ * sinλ) + uz_T * cosφ
    u_vertical = ux_T * (cosφ * cosλ)  + uy_T * (cosφ * sinλ)  + uz_T * sinφ
    
    return SVector{3,T}(u_east, u_north, u_vertical)
end

end # module
