module Geometry

using LinearAlgebra: LinearAlgebra as LA
using StaticArrays: StaticArrays as SA

export AbstractGeometry, CartesianGeometry, SphericalGeometry
export distance, area_element, volume_element, to_planetary_cartesian, from_planetary_cartesian
export nonuniform_first_derivative
export local_tangent_basis, project_to_tangent_plane

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
- Essential for global-scale calculations on a sphere

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
@inline function distance(::CartesianGeometry{T}, pt1::SA.SVector{N,T}, pt2::SA.SVector{N,T}) where {N,T}
    return LA.norm(pt1 - pt2)
end

@inline function distance(geo::SphericalGeometry{T}, coords1::SA.SVector{2,T}, coords2::SA.SVector{2,T}) where {T}
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
@inline function distance(geo::SphericalGeometry{T}, coords1::SA.SVector{3,T}, coords2::SA.SVector{3,T}) where {T}
    # Transform spherical coords (λ, φ, r) to 3D planetary Cartesian first, then compute Euclidean distance
    p1 = spherical_to_planetary_position(geo, coords1)
    p2 = spherical_to_planetary_position(geo, coords2)
    return LA.norm(p1 - p2)
end

# Helper to transform position coords (λ, φ, r) to Cartesian X, Y, Z. `r` is the ABSOLUTE radius
# (physical distance from the planet center) — not a depth/height offset from a reference radius,
# which would force picking an ocean-vs-atmosphere sign convention with no natural default.
@inline function spherical_to_planetary_position(geo::SphericalGeometry{T}, coords::SA.SVector{3,T}) where {T}
    λ, φ, r = coords[1], coords[2], coords[3]
    X = r * cos(φ) * cos(λ)
    Y = r * cos(φ) * sin(λ)
    Z = r * sin(φ)
    return SA.SVector{3,T}(X, Y, Z)
end

@inline function spherical_to_planetary_position(geo::SphericalGeometry{T}, coords::SA.SVector{2,T}) where {T}
    λ, φ = coords[1], coords[2]
    X = geo.R * cos(φ) * cos(λ)
    Y = geo.R * cos(φ) * sin(λ)
    Z = geo.R * sin(φ)
    return SA.SVector{3,T}(X, Y, Z)
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

"""
    volume_element(geo::CartesianGeometry{T})
    volume_element(geo::SphericalGeometry{T}, r::T, lat::T, dλ::T, dφ::T, dr::T)

Compute local grid cell volume. The spherical form generalizes [`area_element`](@ref) with the
LOCAL radius `r` at this level (not the fixed reference `geo.R`) — a genuine spherical-shell volume
element `r²·cosφ·dλ·dφ·dr`, needed once a grid has real multi-level radial structure instead of a
single reference sphere.
"""
@inline volume_element(geo::CartesianGeometry{T}) where {T} = geo.dx * geo.dy * geo.dz

@inline function volume_element(::SphericalGeometry{T}, r::T, lat::T, dλ::T, dφ::T, dr::T) where {T}
    return r^2 * cos(lat) * dλ * dφ * dr
end

# ---------------------------------------------------------------------------
# Nonuniform finite differences
# ---------------------------------------------------------------------------

"""
    nonuniform_first_derivative(f_m, f_0, f_p, h_m, h_p)

Standard 3-point, 2nd-order-accurate centered finite-difference approximation of the first
derivative at the middle node on a possibly *nonuniform* stencil. `f_m`, `f_0`, `f_p` are the
function values at the minus/center/plus nodes, and `h_m = x_0 - x_{-}`, `h_p = x_{+} - x_0 > 0` are
the (physical) left/right spacings.

Reduces exactly to the uniform central difference `(f_p - f_m)/(2h)` when `h_m == h_p == h`, and is
exact for linear and quadratic `f` for any `h_m`, `h_p`.
"""
@inline function nonuniform_first_derivative(f_m::T, f_0::T, f_p::T, h_m::T, h_p::T) where {T<:AbstractFloat}
    return (h_m^2 * f_p + (h_p^2 - h_m^2) * f_0 - h_p^2 * f_m) / (h_m * h_p * (h_m + h_p))
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

    return SA.SVector{3,T}(ux, uy, uz)
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

    return SA.SVector{3,T}(u_east, u_north, u_vertical)
end

# ---------------------------------------------------------------------------
# Local tangent-plane geometry (curvilinear / unstructured gradient reconstruction)
# ---------------------------------------------------------------------------

"""
    local_tangent_basis(geo, coords) -> (ê_east, ê_north)

Local physical East/North unit vectors at the point `coords`, expressed in the geometry's ambient
Cartesian frame. Used to build the 2D tangent-plane displacement of a stencil neighbour for
least-squares gradient reconstruction on grids with no separable coordinate axes (curvilinear,
unstructured).

- `CartesianGeometry`: trivial — `ê_east = (1, 0)`, `ê_north = (0, 1)` (the tangent plane *is* the
  `(x, y)` plane), returned as 2-vectors.
- `SphericalGeometry`: the exact local East/North unit vectors of the sphere at `(λ, φ)`, expressed
  in 3D planetary Cartesian coordinates (the same `[-sinλ, cosλ, 0]` / `[-sinφcosλ, -sinφsinλ, cosφ]`
  frame `to_planetary_cartesian` rotates through), returned as 3-vectors. No small-angle
  approximation.

Zero-allocation (returns a stack-allocated tuple of `SVector`s); safe to call per grid point.
"""
@inline function local_tangent_basis(::CartesianGeometry{T}, ::SA.SVector{2,T}) where {T}
    return (SA.SVector{2,T}(one(T), zero(T)), SA.SVector{2,T}(zero(T), one(T)))
end

@inline function local_tangent_basis(::SphericalGeometry{T}, coords::SA.SVector{2,T}) where {T}
    λ, φ = coords[1], coords[2]
    sinλ, cosλ = sin(λ), cos(λ)
    sinφ, cosφ = sin(φ), cos(φ)
    ê_east  = SA.SVector{3,T}(-sinλ, cosλ, zero(T))
    ê_north = SA.SVector{3,T}(-sinφ * cosλ, -sinφ * sinλ, cosφ)
    return (ê_east, ê_north)
end

"""
    project_to_tangent_plane(geo, center, neighbor) -> SVector{2,T}

Displacement of `neighbor` relative to `center`, projected into the local tangent plane at `center`
and returned as its `(East, North)` components — the physical-space displacement a curvilinear/
unstructured least-squares gradient stencil differences against.

- `CartesianGeometry`: exactly `neighbor - center` (the tangent plane is the coordinate plane).
- `SphericalGeometry`: the **exact 3D chord** `P(neighbor) - P(center)` (both mapped to planetary
  Cartesian via `spherical_to_planetary_position`) dotted onto the local East/North basis
  from [`local_tangent_basis`](@ref). This is an exact chord projection — not a small-angle/flat-Earth
  approximation — so it stays second-order consistent for the WLSQ reconstruction.

Zero-allocation (returns a stack-allocated `SVector{2,T}`); safe to call per stencil neighbour.
"""
@inline function project_to_tangent_plane(
    ::CartesianGeometry{T}, center::SA.SVector{2,T}, neighbor::SA.SVector{2,T},
) where {T}
    return neighbor - center
end

@inline function project_to_tangent_plane(
    geo::SphericalGeometry{T}, center::SA.SVector{2,T}, neighbor::SA.SVector{2,T},
) where {T}
    Pc = spherical_to_planetary_position(geo, center)
    Pn = spherical_to_planetary_position(geo, neighbor)
    chord = Pn - Pc
    ê_east, ê_north = local_tangent_basis(geo, center)
    return SA.SVector{2,T}(LA.dot(chord, ê_east), LA.dot(chord, ê_north))
end

end # module
