module Diagnostics

using ..Geometry: Geometry
using ..Grids: Grids
using ..Kernels: Kernels
using ..Filtering: Filtering
using ..Derivatives: Derivatives
using ..Backends: Backends

export ΠWorkspace, compute_Π!, compute_filtering_spectrum

"""
    ΠWorkspace{T, A}

Pre-allocated arrays for computing cross-scale energy flux Π to avoid heap allocations in scale loops.
"""
struct ΠWorkspace{T<:AbstractFloat, A<:AbstractArray{T}}
    # Filtered velocity components (local coordinates)
    u_filt::A
    v_filt::A
    w_filt::A
    
    # Planetary Cartesian velocities (if Spherical geometry is used)
    ux::A
    uy::A
    uz::A
    ux_filt::A
    uy_filt::A
    uz_filt::A
    
    # Filtered quadratic velocity products (planetary Cartesian if Spherical, else Cartesian)
    uu_filt::A
    uv_filt::A
    uw_filt::A
    vv_filt::A
    vw_filt::A
    ww_filt::A
    
    # Velocity derivatives / strain rate components (local coordinates)
    S_xx::A
    S_xy::A
    S_xz::A
    S_yy::A
    S_yz::A
    S_zz::A
    
    # Subfilter-scale stress components (local coordinates)
    τ_xx::A
    τ_xy::A
    τ_xz::A
    τ_yy::A
    τ_yz::A
    τ_zz::A
    
    # General temporary array for scratch work
    scratch::A
end

# Workspace constructor based on grid structure and float type
function ΠWorkspace(grid::Grids.AbstractGrid{G,T}; dims::Integer = 2) where {G, T<:AbstractFloat}
    sz = Grids.size_tuple(grid)
    A = Matrix{T}
    
    u_filt  = zeros(T, sz...)
    v_filt  = zeros(T, sz...)
    w_filt  = zeros(T, sz...)
    
    ux      = zeros(T, sz...)
    uy      = zeros(T, sz...)
    uz      = zeros(T, sz...)
    ux_filt = zeros(T, sz...)
    uy_filt = zeros(T, sz...)
    uz_filt = zeros(T, sz...)
    
    uu_filt = zeros(T, sz...)
    uv_filt = zeros(T, sz...)
    uw_filt = zeros(T, sz...)
    vv_filt = zeros(T, sz...)
    vw_filt = zeros(T, sz...)
    ww_filt = zeros(T, sz...)
    
    S_xx    = zeros(T, sz...)
    S_xy    = zeros(T, sz...)
    S_xz    = zeros(T, sz...)
    S_yy    = zeros(T, sz...)
    S_yz    = zeros(T, sz...)
    S_zz    = zeros(T, sz...)
    
    τ_xx    = zeros(T, sz...)
    τ_xy    = zeros(T, sz...)
    τ_xz    = zeros(T, sz...)
    τ_yy    = zeros(T, sz...)
    τ_yz    = zeros(T, sz...)
    τ_zz    = zeros(T, sz...)
    
    scratch = zeros(T, sz...)
    
    return ΠWorkspace{T, A}(
        u_filt, v_filt, w_filt,
        ux, uy, uz, ux_filt, uy_filt, uz_filt,
        uu_filt, uv_filt, uw_filt, vv_filt, vw_filt, ww_filt,
        S_xx, S_xy, S_xz, S_yy, S_yz, S_zz,
        τ_xx, τ_xy, τ_xz, τ_yy, τ_yz, τ_zz,
        scratch
    )
end

# ---------------------------------------------------------------------------
# Energy Flux (Π) Calculation
# ---------------------------------------------------------------------------

"""
    compute_Π!(Π, u, v, w, grid, kernel, scale; ρ₀=1025.0, workspace=nothing, backend=AutoBackend(), mask_strategy=:renormalize)

Compute the cross-scale kinetic energy flux Π = -ρ₀ S̄_ij τ_ij at filter scale ℓ.

This implements the coarse-graining framework of Aluie et al. (2018) for computing
energy transfer across scales in turbulent flows. Positive Π indicates forward cascade
(energy from large to small scales), negative Π indicates inverse cascade.

# Arguments
- `Π::AbstractMatrix{T}`: Output array for energy flux (modified in-place)
- `u::AbstractMatrix`: Eastward/zonal velocity component
- `v::AbstractMatrix`: Northward/meridional velocity component  
- `w::Union{Nothing,AbstractMatrix}`: Vertical velocity (nothing for 2D calculations)
- `grid::StructuredGrid`: Grid geometry and coordinates
- `kernel::AbstractFilterKernel`: Filter kernel
- `scale::T`: Filter scale ℓ in meters

# Keyword Arguments
- `ρ₀::T=1025.0`: Reference density (kg/m³), default seawater value
- `workspace=nothing`: Pre-allocated ΠWorkspace for intermediate arrays
- `backend::AbstractExecutionBackend=AutoBackend()`: Execution backend
- `mask_strategy::Symbol=:renormalize`: Land masking strategy

# Physics
The cross-scale energy flux is computed as:
```
Π = -ρ₀ * S̄_ij * τ_ij
```
where:
- `S̄_ij = 0.5 * (∂ū_i/∂x_j + ∂ū_j/∂x_i)` is the resolved strain rate tensor
- `τ_ij = [u_i*u_j]̄ - ū_i*ū_j` is the subfilter-scale (SFS) stress tensor
- Overbar denotes filtered quantities

For spherical geometry, velocity components are transformed to planetary Cartesian
coordinates before filtering to ensure commutativity with derivatives (Aluie 2019).

# Returns
- `Π`: Energy flux array (same as input), units of W/m³

# Examples
```julia
Π = zeros(100, 100)
compute_Π!(Π, u, v, nothing, grid, TopHatKernel(), 30000.0; ρ₀=1025.0)
# Π now contains energy flux at 30 km scale
```

# References
- Aluie et al. (2018): https://doi.org/10.1175/JPO-D-17-0100.1
- Aluie (2019): https://doi.org/10.1007/s13137-019-0123-9
"""
function compute_Π!(
    Π::AbstractMatrix{T},
    u::AbstractMatrix,
    v::AbstractMatrix,
    w::Union{Nothing, AbstractMatrix}, # nothing or zeros for 2D
    grid::Grids.StructuredGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    ρ₀::T = T(1025.0),
    workspace::Union{Nothing, ΠWorkspace} = nothing,
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    mask_strategy::Symbol = :renormalize
) where {T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}}
    
    # 1. Fetch or create pre-allocated workspace
    ws = workspace === nothing ? ΠWorkspace(grid) : workspace
    Nlon, Nlat = Grids.size_tuple(grid)
    
    has_w = w !== nothing
    
    if G <: Geometry.CartesianGeometry{T}
        # -------------------------------------------------------------------
        # Cartesian Case
        # -------------------------------------------------------------------
        # Filter velocity components
        Filtering.filter_field!(ws.u_filt, u, grid, kernel, scale; mask_strategy=mask_strategy, backend=backend)
        Filtering.filter_field!(ws.v_filt, v, grid, kernel, scale; mask_strategy=mask_strategy, backend=backend)
        if has_w
            Filtering.filter_field!(ws.w_filt, w, grid, kernel, scale; mask_strategy=mask_strategy, backend=backend)
        end
        
        # Filter products: u², uv, vv, etc.
        # uu
        @. ws.scratch = u * u
        Filtering.filter_field!(ws.uu_filt, ws.scratch, grid, kernel, scale; mask_strategy=mask_strategy, backend=backend)
        # uv
        @. ws.scratch = u * v
        Filtering.filter_field!(ws.uv_filt, ws.scratch, grid, kernel, scale; mask_strategy=mask_strategy, backend=backend)
        # vv
        @. ws.scratch = v * v
        Filtering.filter_field!(ws.vv_filt, ws.scratch, grid, kernel, scale; mask_strategy=mask_strategy, backend=backend)
        
        if has_w
            @. ws.scratch = u * w
            Filtering.filter_field!(ws.uw_filt, ws.scratch, grid, kernel, scale; mask_strategy=mask_strategy, backend=backend)
            @. ws.scratch = v * w
            Filtering.filter_field!(ws.vw_filt, ws.scratch, grid, kernel, scale; mask_strategy=mask_strategy, backend=backend)
            @. ws.scratch = w * w
            Filtering.filter_field!(ws.ww_filt, ws.scratch, grid, kernel, scale; mask_strategy=mask_strategy, backend=backend)
        end
        
        # Compute subfilter stresses: τ_ij = [u_i u_j]̄ - ū_i ū_j
        @. ws.τ_xx = ws.uu_filt - ws.u_filt * ws.u_filt
        @. ws.τ_xy = ws.uv_filt - ws.u_filt * ws.v_filt
        @. ws.τ_yy = ws.vv_filt - ws.v_filt * ws.v_filt
        if has_w
            @. ws.τ_xz = ws.uw_filt - ws.u_filt * ws.w_filt
            @. ws.τ_yz = ws.vw_filt - ws.v_filt * ws.w_filt
            @. ws.τ_zz = ws.ww_filt - ws.w_filt * ws.w_filt
        end
        
        # Compute strain rate tensor components: S̄_ij = 0.5 * (∂ū_i/∂x_j + ∂ū_j/∂x_i)
        # S_xx = ∂ū/∂x
        Derivatives.ddx!(ws.S_xx, ws.u_filt, grid)
        # S_yy = ∂v̄/∂y
        Derivatives.ddy!(ws.S_yy, ws.v_filt, grid)
        # S_xy = 0.5 * (∂ū/∂y + ∂v̄/∂x)
        Derivatives.ddy!(ws.S_xy, ws.u_filt, grid)
        Derivatives.ddx!(ws.scratch, ws.v_filt, grid)
        @. ws.S_xy = T(0.5) * (ws.S_xy + ws.scratch)
        
        if has_w
            # S_xz = 0.5 * (∂ū/∂z + ∂w̄/∂x) (assumes ∂/∂z is zero or handles vertical layers)
            Derivatives.ddx!(ws.S_xz, ws.w_filt, grid)
            @. ws.S_xz = T(0.5) * ws.S_xz
            
            # S_yz = 0.5 * (∂v̄/∂z + ∂w̄/∂y)
            Derivatives.ddy!(ws.S_yz, ws.w_filt, grid)
            @. ws.S_yz = T(0.5) * ws.S_yz
            
            # S_zz = ∂w̄/∂z = 0 (for standard 2.5D datasets)
            fill!(ws.S_zz, zero(T))
        end
        
    else
        # -------------------------------------------------------------------
        # Spherical Case (Aluie 2019 commutativity formulation)
        # -------------------------------------------------------------------
        # Transform local coordinates (u_east, v_north) to global Cartesian (u_X, u_Y, u_Z)
        for j in 1:Nlat
            φ = grid.lat[j]
            for i in 1:Nlon
                if Grids.iswet(grid, i, j)
                    λ = grid.lon[i]
                    u_val = u[i, j]
                    v_val = v[i, j]
                    w_val = has_w ? w[i, j] : zero(T)
                    
                    p_vel = Geometry.to_planetary_cartesian(grid.geometry, u_val, v_val, w_val, λ, φ)
                    ws.ux[i, j] = p_vel[1]
                    ws.uy[i, j] = p_vel[2]
                    ws.uz[i, j] = p_vel[3]
                else
                    ws.ux[i, j] = zero(T)
                    ws.uy[i, j] = zero(T)
                    ws.uz[i, j] = zero(T)
                end
            end
        end
        
        # Filter planetary Cartesian components
        Filtering.filter_field!(ws.ux_filt, ws.ux, grid, kernel, scale; mask_strategy=mask_strategy, backend=backend)
        Filtering.filter_field!(ws.uy_filt, ws.uy, grid, kernel, scale; mask_strategy=mask_strategy, backend=backend)
        Filtering.filter_field!(ws.uz_filt, ws.uz, grid, kernel, scale; mask_strategy=mask_strategy, backend=backend)
        
        # Filter planetary products: X-X, X-Y, X-Z, Y-Y, Y-Z, Z-Z
        @. ws.scratch = ws.ux * ws.ux
        Filtering.filter_field!(ws.uu_filt, ws.scratch, grid, kernel, scale; mask_strategy=mask_strategy, backend=backend)
        @. ws.scratch = ws.ux * ws.uy
        Filtering.filter_field!(ws.uv_filt, ws.scratch, grid, kernel, scale; mask_strategy=mask_strategy, backend=backend)
        @. ws.scratch = ws.ux * ws.uz
        Filtering.filter_field!(ws.uw_filt, ws.scratch, grid, kernel, scale; mask_strategy=mask_strategy, backend=backend)
        @. ws.scratch = ws.uy * ws.uy
        Filtering.filter_field!(ws.vv_filt, ws.scratch, grid, kernel, scale; mask_strategy=mask_strategy, backend=backend)
        @. ws.scratch = ws.uy * ws.uz
        Filtering.filter_field!(ws.vw_filt, ws.scratch, grid, kernel, scale; mask_strategy=mask_strategy, backend=backend)
        @. ws.scratch = ws.uz * ws.uz
        Filtering.filter_field!(ws.ww_filt, ws.scratch, grid, kernel, scale; mask_strategy=mask_strategy, backend=backend)
        
        # Transform filtered planetary velocities back to local coordinates (u_filt, v_filt, w_filt)
        for j in 1:Nlat
            φ = grid.lat[j]
            for i in 1:Nlon
                if Grids.iswet(grid, i, j)
                    λ = grid.lon[i]
                    l_vel = Geometry.from_planetary_cartesian(grid.geometry, ws.ux_filt[i, j], ws.uy_filt[i, j], ws.uz_filt[i, j], λ, φ)
                    ws.u_filt[i, j] = l_vel[1]
                    ws.v_filt[i, j] = l_vel[2]
                    ws.w_filt[i, j] = l_vel[3]
                else
                    ws.u_filt[i, j] = zero(T)
                    ws.v_filt[i, j] = zero(T)
                    ws.w_filt[i, j] = zero(T)
                end
            end
        end
        
        # Transform planetary filtered products to local stresses at each grid point
        # For orthogonal transformation of 2nd rank symmetric tensor: τ_local = R' * ( [u_i u_j]̄ - ū_i ū_j ) * R
        for j in 1:Nlat
            φ = grid.lat[j]
            sinφ, cosφ = sin(φ), cos(φ)
            for i in 1:Nlon
                if Grids.iswet(grid, i, j)
                    λ = grid.lon[i]
                    sinλ, cosλ = sin(λ), cos(λ)
                    
                    # 3x3 symmetric Cartesian SFS stress tensor
                    txx = ws.uu_filt[i, j] - ws.ux_filt[i, j] * ws.ux_filt[i, j]
                    txy = ws.uv_filt[i, j] - ws.ux_filt[i, j] * ws.uy_filt[i, j]
                    txz = ws.uw_filt[i, j] - ws.ux_filt[i, j] * ws.uz_filt[i, j]
                    tyy = ws.vv_filt[i, j] - ws.uy_filt[i, j] * ws.uy_filt[i, j]
                    tyz = ws.vw_filt[i, j] - ws.uy_filt[i, j] * ws.uz_filt[i, j]
                    tzz = ws.ww_filt[i, j] - ws.uz_filt[i, j] * ws.uz_filt[i, j]
                    
                    # Transform planetary stress tensor back to local coordinates
                    # using local rotation matrix R = [e_east, e_north, e_radial]
                    # We compute local stress elements:
                    # τ_ee = e_east' * T * e_east
                    # τ_nn = e_north' * T * e_north
                    # τ_en = e_east' * T * e_north
                    
                    # e_east = [-sinλ, cosλ, 0]
                    # e_north = [-sinφ*cosλ, -sinφ*sinλ, cosφ]
                    
                    # Temporary multiplication of T * e_east:
                    te_x = txx * (-sinλ) + txy * cosλ
                    te_y = txy * (-sinλ) + tyy * cosλ
                    te_z = txz * (-sinλ) + tyz * cosλ
                    
                    ws.τ_xx[i, j] = te_x * (-sinλ) + te_y * cosλ # τ_ee
                    
                    # Multiplication of T * e_north:
                    tn_x = txx * (-sinφ * cosλ) + txy * (-sinφ * sinλ) + txz * cosφ
                    tn_y = txy * (-sinφ * cosλ) + tyy * (-sinφ * sinλ) + tyz * cosφ
                    tn_z = txz * (-sinφ * cosλ) + tyz * (-sinφ * sinλ) + tzz * cosφ
                    
                    ws.τ_yy[i, j] = tn_x * (-sinφ * cosλ) + tn_y * (-sinφ * sinλ) + tn_z * cosφ # τ_nn
                    ws.τ_xy[i, j] = te_x * (-sinφ * cosλ) + te_y * (-sinφ * sinλ) + te_z * cosφ # τ_en
                    
                    if has_w
                        # e_radial = [cosφ*cosλ, cosφ*sinλ, sinφ]
                        tr_x = txx * (cosφ * cosλ) + txy * (cosφ * sinλ) + txz * sinφ
                        tr_y = txy * (cosφ * cosλ) + tyy * (cosφ * sinλ) + tyz * sinφ
                        tr_z = txz * (cosφ * cosλ) + tyz * (cosφ * sinλ) + tzz * sinφ
                        
                        ws.τ_xz[i, j] = te_x * (cosφ * cosλ) + te_y * (cosφ * sinλ) + te_z * sinφ # τ_er
                        ws.τ_yz[i, j] = tn_x * (cosφ * cosλ) + tn_y * (cosφ * sinλ) + tn_z * sinφ # τ_nr
                        ws.τ_zz[i, j] = tr_x * (cosφ * cosλ) + tr_y * (cosφ * sinλ) + tr_z * sinφ # τ_rr
                    end
                else
                    ws.τ_xx[i, j] = zero(T)
                    ws.τ_yy[i, j] = zero(T)
                    ws.τ_xy[i, j] = zero(T)
                    if has_w
                        ws.τ_xz[i, j] = zero(T)
                        ws.τ_yz[i, j] = zero(T)
                        ws.τ_zz[i, j] = zero(T)
                    end
                end
            end
        end
        
        # Compute Spherical Strain Rates (with geometry curvature correction terms)
        # S_ee = 1/(R cosφ) * ∂ū_e/∂λ - v̄_n * sinφ / (R cosφ)
        Derivatives.ddx!(ws.S_xx, ws.u_filt, grid)
        # S_nn = 1/R * ∂v̄_n/∂φ
        Derivatives.ddy!(ws.S_yy, ws.v_filt, grid)
        # S_en = 0.5 * ( 1/(R cosφ) ∂v̄_n/∂λ + 1/R ∂ū_e/∂φ + ū_e * sinφ / (R cosφ) )
        Derivatives.ddy!(ws.S_xy, ws.u_filt, grid)
        Derivatives.ddx!(ws.scratch, ws.v_filt, grid)
        
        R = grid.geometry.R
        for j in 1:Nlat
            φ = grid.lat[j]
            cosφ = cos(φ)
            sinφ = sin(φ)
            tan_fact = abs(cosφ) > T(1e-12) ? sinφ / (R * cosφ) : zero(T)
            
            for i in 1:Nlon
                if Grids.iswet(grid, i, j)
                    # S_ee correction
                    ws.S_xx[i, j] -= ws.v_filt[i, j] * tan_fact
                    
                    # S_en correction
                    ws.S_xy[i, j] = T(0.5) * (ws.S_xy[i, j] + ws.scratch[i, j] + ws.u_filt[i, j] * tan_fact)
                end
            end
        end
        
        if has_w
            # S_er = 0.5 * (∂ū_e/∂r + 1/(R cosφ) ∂w̄/∂λ) = 0.5 * 1/(R cosφ) ∂w̄/∂λ (if vertically flat layers)
            Derivatives.ddx!(ws.S_xz, ws.w_filt, grid)
            @. ws.S_xz = T(0.5) * ws.S_xz
            
            # S_nr = 0.5 * (∂v̄_n/∂r + 1/R ∂w̄/∂φ) = 0.5 * 1/R ∂w̄/∂φ
            Derivatives.ddy!(ws.S_yz, ws.w_filt, grid)
            @. ws.S_yz = T(0.5) * ws.S_yz
            
            # S_rr = ∂w̄/∂r = 0
            fill!(ws.S_zz, zero(T))
        end
    end
    
    # -----------------------------------------------------------------------
    # Tensor contraction: Π = -ρ₀ Σ_ij S̄_ij τ_ij
    # -----------------------------------------------------------------------
    # Since stress & strain rates are symmetric:
    # S̄_ij τ_ij = S_xx*τ_xx + 2*S_xy*τ_xy + S_yy*τ_yy (2D)
    # S̄_ij τ_ij = S_xx*τ_xx + 2*S_xy*τ_xy + S_yy*τ_yy + 2*S_xz*τ_xz + 2*S_yz*τ_yz + S_zz*τ_zz (3D)
    if has_w
        for j in 1:Nlat
            for i in 1:Nlon
                if Grids.iswet(grid, i, j)
                    sfs_sum = ws.S_xx[i, j] * ws.τ_xx[i, j] +
                              T(2) * ws.S_xy[i, j] * ws.τ_xy[i, j] +
                              ws.S_yy[i, j] * ws.τ_yy[i, j] +
                              T(2) * ws.S_xz[i, j] * ws.τ_xz[i, j] +
                              T(2) * ws.S_yz[i, j] * ws.τ_yz[i, j] +
                              ws.S_zz[i, j] * ws.τ_zz[i, j]
                    Π[i, j] = -ρ₀ * sfs_sum
                else
                    Π[i, j] = zero(T)
                end
            end
        end
    else
        for j in 1:Nlat
            for i in 1:Nlon
                if Grids.iswet(grid, i, j)
                    sfs_sum = ws.S_xx[i, j] * ws.τ_xx[i, j] +
                              T(2) * ws.S_xy[i, j] * ws.τ_xy[i, j] +
                              ws.S_yy[i, j] * ws.τ_yy[i, j]
                    Π[i, j] = -ρ₀ * sfs_sum
                else
                    Π[i, j] = zero(T)
                end
            end
        end
    end
    
    return Π
end

# ---------------------------------------------------------------------------
# Filtering Energy Spectrum E(ℓ)
# ---------------------------------------------------------------------------

"""
    compute_filtering_spectrum(u, v, w, grid, kernel, scales; ρ₀=1025.0, backend=AutoBackend(), mask_strategy=:renormalize)

Compute the filtering kinetic energy spectrum E(ℓ) = 0.5 * ρ₀ * ⟨|ū_ℓ|²⟩.

The filtering spectrum characterizes the distribution of kinetic energy across
scales through coarse-grained (filtered) velocities.

# Arguments
- `u::AbstractMatrix`: Eastward/zonal velocity component
- `v::AbstractMatrix`: Northward/meridional velocity component
- `w::Union{Nothing,AbstractMatrix}`: Vertical velocity (nothing for 2D)
- `grid::StructuredGrid`: Grid geometry with cell areas for weighting
- `kernel::AbstractFilterKernel`: Filter kernel
- `scales::AbstractVector`: Vector of filter scales ℓ in meters

# Keyword Arguments
- `ρ₀::T=1025.0`: Reference density (kg/m³)
- `backend::AbstractExecutionBackend=AutoBackend()`: Execution backend
- `mask_strategy::Symbol=:renormalize`: Land masking strategy

# Returns
- `spectrum::Vector{T}`: Energy spectrum values at each scale (m²/s²)

# Notes
The spectrum is computed as an area-weighted spatial average:
```
E(ℓ) = 0.5 * ρ₀ * ∫ |ū_ℓ|² dA / ∫ dA
```
where the integrals are over the wet domain.

# Examples
```julia
scales = collect(10000.0:10000.0:100000.0)  # 10-100 km
spectrum = compute_filtering_spectrum(u, v, grid, TopHatKernel(), scales)
# spectrum[i] is energy at scale scales[i]
```
"""
function compute_filtering_spectrum(
    u::AbstractMatrix,
    v::AbstractMatrix,
    w::Union{Nothing, AbstractMatrix},
    grid::Grids.StructuredGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scales::AbstractVector;
    ρ₀::T = T(1025.0),
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    mask_strategy::Symbol = :renormalize
) where {T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}}
    
    Nscales = length(scales)
    spectrum = zeros(T, Nscales)
    
    # Pre-allocate temporary workspace arrays
    Nlon, Nlat = Grids.size_tuple(grid)
    u_filt = zeros(T, Nlon, Nlat)
    v_filt = zeros(T, Nlon, Nlat)
    w_filt = w !== nothing ? zeros(T, Nlon, Nlat) : nothing
    
    # Precompute total wet grid area for spatial averaging
    total_area = zero(T)
    for j in 1:Nlat
        for i in 1:Nlon
            if Grids.iswet(grid, i, j)
                total_area += Grids.area(grid, i, j)
            end
        end
    end
    
    # Sweep through scales
    for s_idx in 1:Nscales
        ℓ = T(scales[s_idx])
        
        # Filter velocity fields at this scale
        Filtering.filter_field!(u_filt, u, grid, kernel, ℓ; mask_strategy=mask_strategy, backend=backend)
        Filtering.filter_field!(v_filt, v, grid, kernel, ℓ; mask_strategy=mask_strategy, backend=backend)
        if w !== nothing
            Filtering.filter_field!(w_filt, w, grid, kernel, ℓ; mask_strategy=mask_strategy, backend=backend)
        end
        
        # Compute spatial average energy: E(ℓ) = 0.5 * ρ₀ * ∫ |ū_ℓ|² dA / ∫ dA
        integrated_energy = zero(T)
        for j in 1:Nlat
            for i in 1:Nlon
                if Grids.iswet(grid, i, j)
                    vel2 = u_filt[i, j]^2 + v_filt[i, j]^2
                    if w !== nothing
                        vel2 += w_filt[i, j]^2
                    end
                    integrated_energy += vel2 * Grids.area(grid, i, j)
                end
            end
        end
        
        spectrum[s_idx] = T(0.5) * ρ₀ * integrated_energy / total_area
    end
    
    return spectrum
end

end # module
