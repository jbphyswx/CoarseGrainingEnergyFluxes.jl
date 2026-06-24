module Derivatives

using ..Geometry: Geometry
using ..Grids: Grids

export AbstractStencilOrder, SecondOrderStencil
export ddx!, ddy!, ddz!

"""
    AbstractStencilOrder

Abstract supertype for spatial difference stencil orders.
"""
abstract type AbstractStencilOrder end

"""
    SecondOrderStencil <: AbstractStencilOrder

Standard 2nd-order centered difference stencil. Near boundaries and land,
it falls back dynamically to 1st-order one-sided differences to avoid contamination.
"""
struct SecondOrderStencil <: AbstractStencilOrder end

# ---------------------------------------------------------------------------
# X-derivative (ddx!)
# ---------------------------------------------------------------------------

"""
    ddx!(∂f∂x, f, grid)

Calculate spatial derivative of `f` in the horizontal x (Eastward/λ) direction, writing to `∂f∂x`.
"""
function ddx!(
    ∂f∂x::AbstractMatrix{T},
    f::AbstractMatrix{T},
    grid::Grids.StructuredGrid{Geometry.CartesianGeometry{T},T}
) where {T<:AbstractFloat}
    Nlon, Nlat = Grids.size_tuple(grid)
    dx = grid.geometry.dx
    
    for j in 1:Nlat
        for i in 1:Nlon
            if !Grids.iswet(grid, i, j)
                ∂f∂x[i, j] = zero(T)
                continue
            end
            
            # Check neighbors
            has_p = i < Nlon && Grids.iswet(grid, i+1, j)
            has_m = i > 1    && Grids.iswet(grid, i-1, j)
            
            if has_p && has_m
                # Standard 2nd-order centered difference
                ∂f∂x[i, j] = (f[i+1, j] - f[i-1, j]) / (T(2) * dx)
            elseif has_p
                # Forward difference (at boundary or near land)
                ∂f∂x[i, j] = (f[i+1, j] - f[i, j]) / dx
            elseif has_m
                # Backward difference (at boundary or near land)
                ∂f∂x[i, j] = (f[i, j] - f[i-1, j]) / dx
            else
                # Completely isolated point
                ∂f∂x[i, j] = zero(T)
            end
        end
    end
    return ∂f∂x
end

function ddx!(
    ∂f∂x::AbstractMatrix{T},
    f::AbstractMatrix{T},
    grid::Grids.StructuredGrid{Geometry.SphericalGeometry{T},T}
) where {T<:AbstractFloat}
    Nlon, Nlat = Grids.size_tuple(grid)
    R = grid.geometry.R
    dλ = Nlon > 1 ? grid.lon[2] - grid.lon[1] : T(0)
    
    for j in 1:Nlat
        φ = grid.lat[j]
        cosφ = cos(φ)
        
        # Avoid poles division by zero
        inv_denom = abs(cosφ) > T(1e-12) ? one(T) / (R * cosφ * dλ) : zero(T)
        
        for i in 1:Nlon
            if !Grids.iswet(grid, i, j)
                ∂f∂x[i, j] = zero(T)
                continue
            end
            
            # Periodic boundary handling for longitude (spherical grids wrap around)
            i_p = i < Nlon ? i + 1 : 1      # wrap to first point
            i_m = i > 1 ? i - 1 : Nlon      # wrap to last point
            
            has_p = Grids.iswet(grid, i_p, j)
            has_m = Grids.iswet(grid, i_m, j)
            
            if has_p && has_m
                ∂f∂x[i, j] = (f[i_p, j] - f[i_m, j]) / T(2) * inv_denom
            elseif has_p
                ∂f∂x[i, j] = (f[i_p, j] - f[i, j]) * inv_denom
            elseif has_m
                ∂f∂x[i, j] = (f[i, j] - f[i_m, j]) * inv_denom
            else
                ∂f∂x[i, j] = zero(T)
            end
        end
    end
    return ∂f∂x
end

# ---------------------------------------------------------------------------
# Y-derivative (ddy!)
# ---------------------------------------------------------------------------

"""
    ddy!(∂f∂y, f, grid)

Calculate spatial derivative of `f` in the vertical y (Northward/φ) direction, writing to `∂f∂y`.
"""
function ddy!(
    ∂f∂y::AbstractMatrix{T},
    f::AbstractMatrix{T},
    grid::Grids.StructuredGrid{Geometry.CartesianGeometry{T},T}
) where {T<:AbstractFloat}
    Nlon, Nlat = Grids.size_tuple(grid)
    dy = grid.geometry.dy
    
    for j in 1:Nlat
        for i in 1:Nlon
            if !Grids.iswet(grid, i, j)
                ∂f∂y[i, j] = zero(T)
                continue
            end
            
            # Check neighbors
            has_p = j < Nlat && Grids.iswet(grid, i, j+1)
            has_m = j > 1    && Grids.iswet(grid, i, j-1)
            
            if has_p && has_m
                # Standard 2nd-order centered difference
                ∂f∂y[i, j] = (f[i, j+1] - f[i, j-1]) / (T(2) * dy)
            elseif has_p
                # Forward difference (at boundary or near land)
                ∂f∂y[i, j] = (f[i, j+1] - f[i, j]) / dy
            elseif has_m
                # Backward difference (at boundary or near land)
                ∂f∂y[i, j] = (f[i, j] - f[i, j-1]) / dy
            else
                # Completely isolated point
                ∂f∂y[i, j] = zero(T)
            end
        end
    end
    return ∂f∂y
end

function ddy!(
    ∂f∂y::AbstractMatrix{T},
    f::AbstractMatrix{T},
    grid::Grids.StructuredGrid{Geometry.SphericalGeometry{T},T}
) where {T<:AbstractFloat}
    Nlon, Nlat = Grids.size_tuple(grid)
    R = grid.geometry.R
    dφ = Nlat > 1 ? grid.lat[2] - grid.lat[1] : T(0)
    inv_denom = one(T) / (R * dφ)
    
    for j in 1:Nlat
        for i in 1:Nlon
            if !Grids.iswet(grid, i, j)
                ∂f∂y[i, j] = zero(T)
                continue
            end
            
            has_p = j < Nlat && Grids.iswet(grid, i, j+1)
            has_m = j > 1    && Grids.iswet(grid, i, j-1)
            
            if has_p && has_m
                ∂f∂y[i, j] = (f[i, j+1] - f[i, j-1]) / T(2) * inv_denom
            elseif has_p
                ∂f∂y[i, j] = (f[i, j+1] - f[i, j]) * inv_denom
            elseif has_m
                ∂f∂y[i, j] = (f[i, j] - f[i, j-1]) * inv_denom
            else
                ∂f∂y[i, j] = zero(T)
            end
        end
    end
    return ∂f∂y
end

# ---------------------------------------------------------------------------
# Z-derivative (ddz!) - Supports 3D structures
# ---------------------------------------------------------------------------

"""
    ddz!(∂f∂z, f, grid)

Calculate spatial derivative of `f` in the vertical coordinate z, writing to `∂f∂z`.
"""
function ddz!(
    ∂f∂z::AbstractArray{T,3},
    f::AbstractArray{T,3},
    grid::Grids.StructuredGrid{Geometry.CartesianGeometry{T},T}
) where {T<:AbstractFloat}
    Nlon, Nlat, Ndepth = size(f)
    dz = grid.geometry.dz
    
    # Boundary/land-avoiding finite differences in vertical
    for k in 1:Ndepth
        for j in 1:Nlat
            for i in 1:Nlon
                if !Grids.iswet(grid, i, j)
                    ∂f∂z[i, j, k] = zero(T)
                    continue
                end
                
                has_p = k < Ndepth
                has_m = k > 1
                
                if has_p && has_m
                    ∂f∂z[i, j, k] = (f[i, j, k+1] - f[i, j, k-1]) / (T(2) * dz)
                elseif has_p
                    ∂f∂z[i, j, k] = (f[i, j, k+1] - f[i, j, k]) / dz
                elseif has_m
                    ∂f∂z[i, j, k] = (f[i, j, k] - f[i, j, k-1]) / dz
                else
                    ∂f∂z[i, j, k] = zero(T)
                end
            end
        end
    end
    return ∂f∂z
end

# ---------------------------------------------------------------------------
# True 3D Cartesian derivatives (3D grid + 3D mask)
#
# These dispatch on `StructuredGrid{Cartesian,T,3}` (N = 3), which is strictly more specific than the
# N-free 2.5D methods above, so a genuine 3D grid routes here while a 2D grid carrying a 3D field
# (layer-by-layer) keeps using the 2.5D methods. Each direction uses a 2nd-order centered difference
# that falls back to a one-sided difference at the domain edge or against a land cell, mirroring the
# 2D engine. Dry cells are written as exactly zero.
# ---------------------------------------------------------------------------

for (fn, dim, spacing) in (
    (:ddx!, 1, :(grid.geometry.dx)),
    (:ddy!, 2, :(grid.geometry.dy)),
    (:ddz!, 3, :(grid.geometry.dz)),
)
    @eval function $fn(
        ∂f::AbstractArray{T,3},
        f::AbstractArray{T,3},
        grid::Grids.StructuredGrid{Geometry.CartesianGeometry{T},T,3},
    ) where {T<:AbstractFloat}
        Nx, Ny, Nz = Grids.size_tuple(grid)
        h = $spacing
        mask = grid.mask
        d = $dim
        @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
            if !mask[i, j, k]
                ∂f[i, j, k] = zero(T)
                continue
            end
            ip = d == 1 ? i + 1 : i; jp = d == 2 ? j + 1 : j; kp = d == 3 ? k + 1 : k
            im = d == 1 ? i - 1 : i; jm = d == 2 ? j - 1 : j; km = d == 3 ? k - 1 : k
            lim = d == 1 ? Nx : d == 2 ? Ny : Nz
            idx = d == 1 ? i : d == 2 ? j : k
            has_p = idx < lim && mask[ip, jp, kp]
            has_m = idx > 1 && mask[im, jm, km]
            if has_p && has_m
                ∂f[i, j, k] = (f[ip, jp, kp] - f[im, jm, km]) / (T(2) * h)
            elseif has_p
                ∂f[i, j, k] = (f[ip, jp, kp] - f[i, j, k]) / h
            elseif has_m
                ∂f[i, j, k] = (f[i, j, k] - f[im, jm, km]) / h
            else
                ∂f[i, j, k] = zero(T)
            end
        end
        return ∂f
    end
end

end # module
