module Geometry

using LinearAlgebra
using StaticArrays

export AbstractGeometry, CartesianGeometry, SphericalGeometry
export distance, area_element, to_planetary_cartesian, from_planetary_cartesian

"""
    AbstractGeometry{T<:AbstractFloat}

Abstract supertype for all coordinate systems and geometry metrics.
"""
abstract type AbstractGeometry{T<:AbstractFloat} end

"""
    CartesianGeometry{T<:AbstractFloat}

Represent Cartesian coordinates with grid spacings `dx`, `dy`, and optionally `dz`.
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
    ::SphericalGeometry{T},
    u_east::T,
    u_north::T,
    u_vertical::T,
    λ::T,
    φ::T
) where {T<:AbstractFloat}
    # Local-to-Global basis rotation matrix columns:
    # e_east = [-sin(λ), cos(λ), 0]
    # e_north = [-sin(φ)cos(λ), -sin(φ)sin(λ), cos(φ)]
    # e_radial = [cos(φ)cos(λ), cos(φ)sin(λ), sin(φ)]
    
    sinλ, cosλ = sin(λ), cos(λ)
    sinφ, cosφ = sin(φ), cos(φ)
    
    ux = u_east * (-sinλ) + u_north * (-sinφ * cosλ) + u_vertical * (cosφ * cosλ)
    uy = u_east * (cosλ)  + u_north * (-sinφ * sinλ) + u_vertical * (cosφ * sinλ)
    uz =                    u_north * cosφ           + u_vertical * sinφ
    
    return SVector{3,T}(ux, uy, uz)
end

# 2D spherical version (assumes zero vertical velocity)
@inline function to_planetary_cartesian(
    geo::SphericalGeometry{T},
    u_east::T,
    u_north::T,
    λ::T,
    φ::T
) where {T<:AbstractFloat}
    return to_planetary_cartesian(geo, u_east, u_north, zero(T), λ, φ)
end

"""
    from_planetary_cartesian(geo::SphericalGeometry{T}, ux, uy, uz, λ, φ)

Convert global planetary Cartesian velocity components back to local East, North, Radial.
"""
@inline function from_planetary_cartesian(
    ::SphericalGeometry{T},
    ux::T,
    uy::T,
    uz::T,
    λ::T,
    φ::T
) where {T<:AbstractFloat}
    sinλ, cosλ = sin(λ), cos(λ)
    sinφ, cosφ = sin(φ), cos(φ)
    
    u_east     = ux * (-sinλ) + uy * cosλ
    u_north    = ux * (-sinφ * cosλ) + uy * (-sinφ * sinλ) + uz * cosφ
    u_vertical = ux * (cosφ * cosλ)  + uy * (cosφ * sinλ)  + uz * sinφ
    
    return SVector{3,T}(u_east, u_north, u_vertical)
end

end # module
