module Diagnostics

using ..Geometry: Geometry
using ..Grids: Grids
using ..Kernels: Kernels
using ..Filtering: Filtering
using ..Derivatives: Derivatives
using ..Backends: Backends

export ΠWorkspace, compute_Π!, cumulative_energy, filtering_spectrum
export tau_decomposition

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
    compute_Π!(Π, u, v, w, grid, kernel, scale; ρ₀=1025.0, workspace=nothing, backend=AutoBackend(), mask_strategy=Deformable())

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
- `mask_strategy::AbstractMaskStrategy=Deformable()`: Land masking strategy (`ZeroFill()` or `Deformable()`)

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
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable()
) where {T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}}
    
    # 1. Fetch or create pre-allocated workspace
    ws = workspace === nothing ? ΠWorkspace(grid) : workspace
    Nlon, Nlat = Grids.size_tuple(grid)

    # Build the filter footprint/plan ONCE for this scale; reused for every velocity component and
    # quadratic product below (instead of rebuilding it on each of the ~9 filterings).
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend)
    
    has_w = w !== nothing
    
    if G <: Geometry.CartesianGeometry{T}
        # -------------------------------------------------------------------
        # Cartesian Case
        # -------------------------------------------------------------------
        # Filter velocity components
        Filtering.filter_apply!(ws.u_filt, u, plan)
        Filtering.filter_apply!(ws.v_filt, v, plan)
        if has_w
            Filtering.filter_apply!(ws.w_filt, w, plan)
        end
        
        # Filter products: u², uv, vv, etc.
        # uu
        @. ws.scratch = u * u
        Filtering.filter_apply!(ws.uu_filt, ws.scratch, plan)
        # uv
        @. ws.scratch = u * v
        Filtering.filter_apply!(ws.uv_filt, ws.scratch, plan)
        # vv
        @. ws.scratch = v * v
        Filtering.filter_apply!(ws.vv_filt, ws.scratch, plan)
        
        if has_w
            @. ws.scratch = u * w
            Filtering.filter_apply!(ws.uw_filt, ws.scratch, plan)
            @. ws.scratch = v * w
            Filtering.filter_apply!(ws.vw_filt, ws.scratch, plan)
            @. ws.scratch = w * w
            Filtering.filter_apply!(ws.ww_filt, ws.scratch, plan)
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
        Filtering.filter_apply!(ws.ux_filt, ws.ux, plan)
        Filtering.filter_apply!(ws.uy_filt, ws.uy, plan)
        Filtering.filter_apply!(ws.uz_filt, ws.uz, plan)
        
        # Filter planetary products: X-X, X-Y, X-Z, Y-Y, Y-Z, Z-Z
        @. ws.scratch = ws.ux * ws.ux
        Filtering.filter_apply!(ws.uu_filt, ws.scratch, plan)
        @. ws.scratch = ws.ux * ws.uy
        Filtering.filter_apply!(ws.uv_filt, ws.scratch, plan)
        @. ws.scratch = ws.ux * ws.uz
        Filtering.filter_apply!(ws.uw_filt, ws.scratch, plan)
        @. ws.scratch = ws.uy * ws.uy
        Filtering.filter_apply!(ws.vv_filt, ws.scratch, plan)
        @. ws.scratch = ws.uy * ws.uz
        Filtering.filter_apply!(ws.vw_filt, ws.scratch, plan)
        @. ws.scratch = ws.uz * ws.uz
        Filtering.filter_apply!(ws.ww_filt, ws.scratch, plan)
        
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

"""
    compute_Π!(Π::AbstractArray{T,3}, u, v, w, grid::StructuredGrid{Cartesian,T,3}, kernel, scale; ρ₀=1025, mask_strategy=Deformable(), backend=AutoBackend())

Full **three-dimensional** Cartesian cross-scale energy flux Π = -ρ₀ S̄_ij τ_ij with all nine strain
components (the diagonal `S_zz = ∂w̄/∂z` and the off-diagonals `S_xz, S_yz` carry genuine vertical
derivatives, unlike the 2.5D layer-by-layer path). The 3D grid carries a 3D mask, so dry cells are
handled per-cell in all three directions.

The contraction is the symmetric six-term sum
`S̄:τ = S_xx τ_xx + S_yy τ_yy + S_zz τ_zz + 2(S_xy τ_xy + S_xz τ_xz + S_yz τ_yz)`.

Dispatched on a 3D output array + 3D Cartesian grid (the 2D method takes an `AbstractMatrix`).
Spherical 3D (volumetric curvature terms) is not yet implemented. This diagnostic allocates its
temporaries; pass a reusable plan-driven path if it ever lands on a hot loop.
"""
function compute_Π!(
    Π::AbstractArray{T,3},
    u::AbstractArray{<:Any,3},
    v::AbstractArray{<:Any,3},
    w::AbstractArray{<:Any,3},
    grid::Grids.StructuredGrid{G,T,3},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    ρ₀::T = T(1025.0),
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
) where {T<:AbstractFloat, G<:Geometry.CartesianGeometry{T}}
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend)
    flt(f) = (o = zeros(T, size(f)); Filtering.filter_apply!(o, f, plan); o)

    # Filtered velocities and the six independent filtered quadratic products.
    ū = flt(u); v̄ = flt(v); w̄ = flt(w)
    uu = flt(u .* u); uv = flt(u .* v); uw = flt(u .* w)
    vv = flt(v .* v); vw = flt(v .* w); ww = flt(w .* w)

    # Subfilter stress τ_ij = ⟨u_i u_j⟩ - ū_i ū_j (symmetric, six components).
    τxx = uu .- ū .* ū; τxy = uv .- ū .* v̄; τxz = uw .- ū .* w̄
    τyy = vv .- v̄ .* v̄; τyz = vw .- v̄ .* w̄; τzz = ww .- w̄ .* w̄

    # Strain S̄_ij = ½(∂ū_i/∂x_j + ∂ū_j/∂x_i): three diagonals + three off-diagonals.
    Sxx = similar(ū); Derivatives.ddx!(Sxx, ū, grid)
    Syy = similar(ū); Derivatives.ddy!(Syy, v̄, grid)
    Szz = similar(ū); Derivatives.ddz!(Szz, w̄, grid)
    a = similar(ū); b = similar(ū)
    Derivatives.ddy!(a, ū, grid); Derivatives.ddx!(b, v̄, grid); Sxy = T(0.5) .* (a .+ b)
    Derivatives.ddz!(a, ū, grid); Derivatives.ddx!(b, w̄, grid); Sxz = T(0.5) .* (a .+ b)
    Derivatives.ddz!(a, v̄, grid); Derivatives.ddy!(b, w̄, grid); Syz = T(0.5) .* (a .+ b)

    mask = grid.mask
    @inbounds @. Π = ifelse(
        mask,
        -ρ₀ * (Sxx * τxx + Syy * τyy + Szz * τzz +
               T(2) * (Sxy * τxy + Sxz * τxz + Syz * τyz)),
        zero(T),
    )
    return Π
end

# ---------------------------------------------------------------------------
# Filtering Energy Spectrum E(ℓ)
# ---------------------------------------------------------------------------

"""
    cumulative_energy(u, v, w, grid, kernel, scales; ρ₀=1025.0, backend=AutoBackend(), mask_strategy=Deformable())

Cumulative coarse-grained kinetic energy `E(ℓ) = 0.5 ρ₀ ⟨|ū_ℓ|²⟩` at each filter scale
(Sadek & Aluie 2018, PRF, Eq. 15). This is the CUMULATIVE quantity; the filtering spectral DENSITY
(comparable to a Fourier energy spectrum) is its derivative w.r.t. filtering wavenumber — see
[`filtering_spectrum`](@ref).

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
- `mask_strategy::AbstractMaskStrategy=Deformable()`: Land masking strategy (`ZeroFill()` or `Deformable()`)

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
E = cumulative_energy(u, v, nothing, grid, TopHatKernel(), scales)
# E[i] is the cumulative coarse KE at scale scales[i]
```

# References
- Sadek & Aluie (2018), *Phys. Rev. Fluids* 3, 124610 — extracting the spectrum by filtering.
"""
function cumulative_energy(
    u::AbstractMatrix,
    v::AbstractMatrix,
    w::Union{Nothing, AbstractMatrix},
    grid::Grids.StructuredGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scales::AbstractVector;
    ρ₀::T = T(1025.0),
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable()
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
        plan = Filtering.plan_filter(grid, kernel, ℓ; mask_strategy=mask_strategy, backend=backend)

        # Filter velocity fields at this scale
        Filtering.filter_apply!(u_filt, u, plan)
        Filtering.filter_apply!(v_filt, v, plan)
        if w !== nothing
            Filtering.filter_apply!(w_filt, w, plan)
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

"""
    filtering_spectrum(u, v, w, grid, kernel, scales; ρ₀=1025.0, L=1, backend=AutoBackend(), mask_strategy=Deformable())
        -> (k_ℓ, Ẽ)

Filtering spectral DENSITY (Sadek & Aluie 2018, PRF, Eq. 14): the derivative of the cumulative
coarse-grained KE w.r.t. the filtering wavenumber `k_ℓ = L/ℓ`,

    Ẽ(k_ℓ) = d/dk_ℓ [ ½ρ₀⟨|ū_ℓ|²⟩ ] = -(ℓ²/L) d/dℓ[ ½ρ₀⟨|ū_ℓ|²⟩ ].

Unlike [`cumulative_energy`](@ref) (the cumulative quantity, Eq. 15), this is the spectral density
comparable to a Fourier energy spectrum. `L` is the region length: pass the domain size for the
Sadek–Aluie convention `k_ℓ = L/ℓ`; the default `L = 1` gives the FlowSieve convention `k_ℓ = 1/ℓ`.
`scales` need not be uniform. Returns the filtering wavenumbers `k_ℓ` and the density `Ẽ` per scale.

# References
- Sadek & Aluie (2018), *Phys. Rev. Fluids* 3, 124610.
"""
function filtering_spectrum(
    u::AbstractMatrix,
    v::AbstractMatrix,
    w::Union{Nothing, AbstractMatrix},
    grid::Grids.StructuredGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scales::AbstractVector;
    ρ₀::T = T(1025.0),
    L::Real = one(T),
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
) where {T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}}
    cum = cumulative_energy(u, v, w, grid, kernel, scales; ρ₀=ρ₀, backend=backend, mask_strategy=mask_strategy)
    kℓ = T(L) ./ T.(collect(scales))
    return kℓ, spectral_density(cum, kℓ)
end

"""
    spectral_density(C, k) -> dC/dk

Non-uniform finite-difference derivative of cumulative values `C` w.r.t. `k` (central in the
interior, one-sided at the ends). Returns zeros for fewer than two points.
"""
function spectral_density(C::AbstractVector{T}, k::AbstractVector) where {T<:AbstractFloat}
    n = length(C)
    g = zeros(T, n)
    n < 2 && return g
    @inbounds for i in 1:n
        if i == 1
            g[i] = (C[2] - C[1]) / (k[2] - k[1])
        elseif i == n
            g[i] = (C[n] - C[n-1]) / (k[n] - k[n-1])
        else
            g[i] = (C[i+1] - C[i-1]) / (k[i+1] - k[i-1])
        end
    end
    return g
end

# ---------------------------------------------------------------------------
# Subfilter-stress decomposition (Germano 1992): τ = Leonard + Cross + Reynolds
# ---------------------------------------------------------------------------

"""
    tau_decomposition(u, v, grid, kernel, scale; backend=AutoBackend(), mask_strategy=Deformable())
        -> (; L, C, R)

Split the 2D subfilter-scale stress `τ_ij = ⟨u_i u_j⟩ - ū_i ū_j` into Leonard, Cross, and Reynolds
contributions (Germano 1992, *JFM* 238, using generalized central moments so each piece is
individually Galilean-invariant). With `ū = G * u` the filtered velocity and `u' = u - ū` the
residual, and the generalized second moment `M(f, g) = (fg)‾ - f̄ ḡ`:

- Leonard  `L_ij = M(ū_i, ū_j)`            (resolved–resolved),
- Cross    `C_ij = M(ū_i, u'_j) + M(u'_i, ū_j)`,
- Reynolds `R_ij = M(u'_i, u'_j)`          (subfilter–subfilter; backscatter),

with `L + C + R = τ` exactly. Returns a named tuple of named tuples, each holding the symmetric
2D components `(; xx, xy, yy)` as arrays.
"""
function tau_decomposition(
    u::AbstractMatrix,
    v::AbstractMatrix,
    grid::Grids.StructuredGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
) where {T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}}
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend)
    # Filter operator (linear): allocate a fresh output (this is a diagnostic, not an inner loop).
    flt(f) = (o = zeros(T, size(f)); Filtering.filter_apply!(o, f, plan); o)

    ub = flt(u);  vb = flt(v)          # ū, v̄
    up = u .- ub; vp = v .- vb         # residuals u', v'
    ubb = flt(ub); vbb = flt(vb)       # double-filtered ū̄, v̄̄
    upb = flt(up); vpb = flt(vp)       # filtered residuals ū', v̄'

    # Generalized second moment M(f,g) = (fg)‾ - f̄ ḡ, with f̄ = flt(f).
    L = (
        xx = flt(ub .* ub) .- ubb .* ubb,
        xy = flt(ub .* vb) .- ubb .* vbb,
        yy = flt(vb .* vb) .- vbb .* vbb,
    )
    C = (
        xx = T(2) .* (flt(ub .* up) .- ubb .* upb),
        xy = (flt(ub .* vp) .- ubb .* vpb) .+ (flt(up .* vb) .- upb .* vbb),
        yy = T(2) .* (flt(vb .* vp) .- vbb .* vpb),
    )
    R = (
        xx = flt(up .* up) .- upb .* upb,
        xy = flt(up .* vp) .- upb .* vpb,
        yy = flt(vp .* vp) .- vpb .* vpb,
    )
    return (; L = L, C = C, R = R)
end

end # module
