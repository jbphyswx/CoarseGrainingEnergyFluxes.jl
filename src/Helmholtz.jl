module Helmholtz

using ..Geometry
using ..Grids
using StaticArrays
using LinearAlgebra

export helmholtz_decompose!, solve_poisson!

"""
    solve_poisson!(Φ, RHS, grid; max_iter, tol, ω)

Solve the 2D Poisson equation ∇² Φ = RHS on wet points of `grid` using Successive Over-Relaxation (SOR)
with Neumann boundary conditions at land and grid boundaries.
"""
function solve_poisson!(
    Φ::AbstractMatrix{T},
    RHS::AbstractMatrix{T},
    grid::StructuredGrid{G,T};
    max_iter::Integer = 1000,
    tol::T = T(1e-5),
    ω::T = T(1.85), # Relaxation factor for SOR
    boundary::Symbol = :neumann
) where {T<:AbstractFloat, G<:AbstractGeometry{T}}
    Nlon, Nlat = size_tuple(grid)
    
    # Pre-calculate spatial factors
    if G <: CartesianGeometry{T}
        inv_dx2 = one(T) / (grid.geometry.dx^2)
        inv_dy2 = one(T) / (grid.geometry.dy^2)
        denom = T(2) * (inv_dx2 + inv_dy2)
    else
        # Spherical metrics: coords(grid, i, j) gives (lon, lat)
        R = grid.geometry.R
        dλ = Nlon > 1 ? grid.lon[2] - grid.lon[1] : T(0)
        dφ = Nlat > 1 ? grid.lat[2] - grid.lat[1] : T(0)
        inv_dλ2 = one(T) / (dλ^2)
        inv_dφ2 = one(T) / (dφ^2)
    end
    
    fill!(Φ, zero(T))
    
    for iter in 1:max_iter
        max_diff = zero(T)
        
        # Red-Black SOR sweeps for thread safety and fast convergence
        for color in 0:1
            # In Julia, column-major ordering means outer loop over j (lat), inner over i (lon)
            for j in 1:Nlat
                for i in 1:Nlon
                    # Check Red-Black color
                    if ((i + j) % 2) == color
                        iswet(grid, i, j) || continue
                        
                        # Fetch neighbors with Neumann or Dirichlet boundary conditions
                        if boundary === :neumann
                            Φ_ip = i < Nlon && iswet(grid, i+1, j) ? Φ[i+1, j] : Φ[i, j]
                            Φ_im = i > 1    && iswet(grid, i-1, j) ? Φ[i-1, j] : Φ[i, j]
                            Φ_jp = j < Nlat && iswet(grid, i, j+1) ? Φ[i, j+1] : Φ[i, j]
                            Φ_jm = j > 1    && iswet(grid, i, j-1) ? Φ[i, j-1] : Φ[i, j]
                        else # :dirichlet (zero flow potential on boundaries)
                            Φ_ip = i < Nlon && iswet(grid, i+1, j) ? Φ[i+1, j] : zero(T)
                            Φ_im = i > 1    && iswet(grid, i-1, j) ? Φ[i-1, j] : zero(T)
                            Φ_jp = j < Nlat && iswet(grid, i, j+1) ? Φ[i, j+1] : zero(T)
                            Φ_jm = j > 1    && iswet(grid, i, j-1) ? Φ[i, j-1] : zero(T)
                        end
                        
                        # Discretized Laplace step
                        if G <: CartesianGeometry{T}
                            # ∇² Φ = (Φ_ip - 2Φ + Φ_im)/dx² + (Φ_jp - 2Φ + Φ_jm)/dy²
                            rhs_val = RHS[i, j]
                            Φ_new = (inv_dx2 * (Φ_ip + Φ_im) + inv_dy2 * (Φ_jp + Φ_jm) - rhs_val) / denom
                        else
                            # Spherical Laplace operator
                            φ = grid.lat[j]
                            cosφ = cos(φ)
                            sinφ = sin(φ)
                            
                            term_λ = (Φ_ip + Φ_im) * inv_dλ2 / (R^2 * cosφ^2)
                            term_φ = ((Φ_jp + Φ_jm) * inv_dφ2 - sinφ * (Φ_jp - Φ_jm) / (T(2) * dφ * cosφ)) / R^2
                            
                            denom_sph = T(2) * inv_dλ2 / (R^2 * cosφ^2) + T(2) * inv_dφ2 / R^2
                            Φ_new = (term_λ + term_φ - RHS[i, j]) / denom_sph
                        end
                        
                        diff = Φ_new - Φ[i, j]
                        Φ[i, j] += ω * diff
                        max_diff = max(max_diff, abs(diff))
                    end
                end
            end
        end
        
        if max_diff < tol
            break
        end
    end
    
    return Φ
end

"""
    helmholtz_decompose!(u_rot, v_rot, u_div, v_div, u, v, grid; kwargs...)

Separate the horizontal velocity field `(u, v)` into rotational components `(u_rot, v_rot)`
(curl-free) and divergent components `(u_div, v_div)` (divergence-free).
"""
function helmholtz_decompose!(
    u_rot::AbstractMatrix{T},
    v_rot::AbstractMatrix{T},
    u_div::AbstractMatrix{T},
    v_div::AbstractMatrix{T},
    u::AbstractMatrix,
    v::AbstractMatrix,
    grid::StructuredGrid{G,T};
    max_iter::Integer = 1000,
    tol::T = T(1e-5),
    ω::T = T(1.85),
    boundary_χ::Symbol = :neumann,
    boundary_ψ::Symbol = :dirichlet
) where {T<:AbstractFloat, G<:AbstractGeometry{T}}
    Nlon, Nlat = size_tuple(grid)
    
    # 1. Pre-allocate intermediate potential and flux arrays
    div_f  = zeros(T, Nlon, Nlat)
    vort_f = zeros(T, Nlon, Nlat)
    χ      = zeros(T, Nlon, Nlat)  # Poloidal (divergent) potential
    ψ      = zeros(T, Nlon, Nlat)  # Toroidal (rotational) potential
    
    # Precompute grid spacings
    if G <: CartesianGeometry{T}
        dx = grid.geometry.dx
        dy = grid.geometry.dy
    else
        R = grid.geometry.R
        dλ = Nlon > 1 ? grid.lon[2] - grid.lon[1] : T(0)
        dφ = Nlat > 1 ? grid.lat[2] - grid.lat[1] : T(0)
    end
    
    # 2. Compute Divergence and Vorticity fields
    for j in 1:Nlat
        for i in 1:Nlon
            iswet(grid, i, j) || continue
            
            # Fetch neighbors with boundary boundary stencils (centered where possible)
            ip = i < Nlon && iswet(grid, i+1, j) ? i+1 : i
            im = i > 1    && iswet(grid, i-1, j) ? i-1 : i
            jp = j < Nlat && iswet(grid, i, j+1) ? j+1 : j
            jm = j > 1    && iswet(grid, i, j-1) ? j-1 : j
            
            if G <: CartesianGeometry{T}
                h_x = ip == im ? dx : (ip - im) * dx
                h_y = jp == jm ? dy : (jp - jm) * dy

                # Divergence: ∂u/∂x + ∂v/∂y
                dudx = (u[ip, j] - u[im, j]) / h_x
                dvdy = (v[i, jp] - v[i, jm]) / h_y
                div_f[i, j] = dudx + dvdy
                
                # Vorticity: ∂v/∂x - ∂u/∂y
                dvdx = (v[ip, j] - v[im, j]) / h_x
                dudy = (u[i, jp] - u[i, jm]) / h_y
                vort_f[i, j] = dvdx - dudy
            else
                φ = grid.lat[j]
                cosφ = cos(φ)
                
                # Spherical derivatives (h_λ and h_φ can be 0 for isolated wet points)
                h_λ = (ip - im) * dλ
                h_φ = (jp - jm) * dφ
                
                # Divergence: 1/(R cosφ) * [ ∂u/∂λ + ∂/∂φ(v cosφ) ]
                dudλ = ip == im ? zero(T) : (u[ip, j] - u[im, j]) / h_λ
                v_cos_jp = v[i, jp] * cos(grid.lat[jp])
                v_cos_jm = v[i, jm] * cos(grid.lat[jm])
                d_vcos_dφ = jp == jm ? zero(T) : (v_cos_jp - v_cos_jm) / h_φ
                
                div_f[i, j] = (dudλ + d_vcos_dφ) / (R * cosφ)
                
                # Vorticity: 1/(R cosφ) * [ ∂v/∂λ - ∂/∂φ(u cosφ) ]
                dvdλ = ip == im ? zero(T) : (v[ip, j] - v[im, j]) / h_λ
                u_cos_jp = u[i, jp] * cos(grid.lat[jp])
                u_cos_jm = u[i, jm] * cos(grid.lat[jm])
                d_ucos_dφ = jp == jm ? zero(T) : (u_cos_jp - u_cos_jm) / h_φ
                
                vort_f[i, j] = (dvdλ - d_ucos_dφ) / (R * cosφ)
            end
        end
    end
    
    # 2.5. Divergence Balancing for Neumann Solvability compatibility condition (Fredholm alternative)
    total_div = zero(T)
    total_area = zero(T)
    for j in 1:Nlat
        for i in 1:Nlon
            if iswet(grid, i, j)
                total_div += div_f[i, j] * area(grid, i, j)
                total_area += area(grid, i, j)
            end
        end
    end
    
    mean_div = total_div / total_area
    for j in 1:Nlat
        for i in 1:Nlon
            if iswet(grid, i, j)
                div_f[i, j] -= mean_div
            end
        end
    end

    # 3. Solve ∇² χ = Divergence and ∇² ψ = Vorticity
    solve_poisson!(χ, div_f, grid; max_iter=max_iter, tol=tol, ω=ω, boundary=boundary_χ)
    solve_poisson!(ψ, vort_f, grid; max_iter=max_iter, tol=tol, ω=ω, boundary=boundary_ψ)
    
    # 4. Compute divergent and rotational velocities from potentials
    for j in 1:Nlat
        for i in 1:Nlon
            if !iswet(grid, i, j)
                u_div[i, j] = zero(T)
                v_div[i, j] = zero(T)
                u_rot[i, j] = zero(T)
                v_rot[i, j] = zero(T)
                continue
            end
            
            ip = i < Nlon && iswet(grid, i+1, j) ? i+1 : i
            im = i > 1    && iswet(grid, i-1, j) ? i-1 : i
            jp = j < Nlat && iswet(grid, i, j+1) ? j+1 : j
            jm = j > 1    && iswet(grid, i, j-1) ? j-1 : j
            
            if G <: CartesianGeometry{T}
                h_x = ip == im ? dx : (ip - im) * dx
                h_y = jp == jm ? dy : (jp - jm) * dy

                # Divergent velocity u_div = ∇ χ
                u_div[i, j] = (χ[ip, j] - χ[im, j]) / h_x
                v_div[i, j] = (χ[i, jp] - χ[i, jm]) / h_y
                
                # Rotational velocity u_rot = ∇ × (ψ z) = [-∂ψ/∂y, ∂ψ/∂x]
                u_rot[i, j] = -(ψ[i, jp] - ψ[i, jm]) / h_y
                v_rot[i, j] = (ψ[ip, j] - ψ[im, j]) / h_x
            else
                φ = grid.lat[j]
                cosφ = cos(φ)
                h_λ = (ip - im) * dλ
                h_φ = (jp - jm) * dφ
                
                # u_div_λ = 1/(R cosφ) ∂χ/∂λ,  v_div_φ = 1/R ∂χ/∂φ
                u_div[i, j] = ip == im ? zero(T) : (χ[ip, j] - χ[im, j]) / (h_λ * R * cosφ)
                v_div[i, j] = jp == jm ? zero(T) : (χ[i, jp] - χ[i, jm]) / (h_φ * R)
                
                # u_rot_λ = -1/R ∂ψ/∂φ,  v_rot_φ = 1/(R cosφ) ∂ψ/∂λ
                u_rot[i, j] = jp == jm ? zero(T) : -(ψ[i, jp] - ψ[i, jm]) / (h_φ * R)
                v_rot[i, j] = ip == im ? zero(T) : (ψ[ip, j] - ψ[im, j]) / (h_λ * R * cosφ)
            end
        end
    end
    
    return u_rot, v_rot, u_div, v_div
end

end # module
