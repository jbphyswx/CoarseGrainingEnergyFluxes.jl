module Diagnostics

using ..Geometry: Geometry
using ..Grids: Grids
using ..Kernels: Kernels
using ..Filtering: Filtering
using ..Derivatives: Derivatives
using ..Backends: Backends

export ΠWorkspace, compute_Π!, compute_Π_profile!, cumulative_energy, cumulative_energy!, filtering_spectrum, spectral_density, spectral_density!
export tau_decomposition, compute_Π_decomposed, tracer_variance_flux

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

# Workspace constructor based on grid structure and float type. `A` is inferred from what
# `zeros(T, sz...)` actually produces (Vector for a 1D grid, Matrix for 2D, Array{T,3} for 3D) —
# NOT hardcoded, since a 1D/3D grid's `sz` is a 1- or 3-tuple, not always 2D.
function ΠWorkspace(grid::Grids.AbstractGrid{G,T}) where {G, T<:AbstractFloat}
    sz = Grids.size_tuple(grid)

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
    
    return ΠWorkspace(
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

# Boundary-only (once per top-level call, not per grid point): a mismatched v/w would otherwise be
# silently truncated/ignored by CartesianIndices(u), not caught at all. Same idiom as the depth-count
# check in compute_Π_profile! below.
@inline function _validate_field_sizes(grid, Π::AbstractArray, u::AbstractArray, v = nothing, w = nothing)
    gsz = Grids.size_tuple(grid)
    size(u) == gsz || throw(DimensionMismatch("u has size $(size(u)), grid expects $gsz"))
    size(Π) == gsz || throw(DimensionMismatch("Π has size $(size(Π)), grid expects $gsz"))
    v === nothing || size(v) == gsz || throw(DimensionMismatch("v has size $(size(v)), grid expects $gsz"))
    w === nothing || size(w) == gsz || throw(DimensionMismatch("w has size $(size(w)), grid expects $gsz"))
    return nothing
end

"""
    compute_Π!(Π, u, v, w, grid, kernel, scale; workspace=nothing, backend=AutoBackend(), mask_strategy=Deformable())

Compute the cross-scale kinetic energy flux Π = -S̄_ij τ_ij at filter scale ℓ.

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
- `workspace=nothing`: Pre-allocated ΠWorkspace for intermediate arrays
- `backend::AbstractExecutionBackend=AutoBackend()`: Execution backend
- `mask_strategy::AbstractMaskStrategy=Deformable()`: Land masking strategy (`ZeroFill()` or `Deformable()`)

# Physics
The cross-scale energy flux is computed as:
```
Π = -S̄_ij * τ_ij
```
where:
- `S̄_ij = 0.5 * (∂ū_i/∂x_j + ∂ū_j/∂x_i)` is the resolved strain rate tensor
- `τ_ij = [u_i*u_j]̄ - ū_i*ū_j` is the subfilter-scale (SFS) stress tensor
- Overbar denotes filtered quantities

For spherical geometry, velocity components are transformed to planetary Cartesian
coordinates before filtering to ensure commutativity with derivatives (Aluie 2019).

# Physics regime: 2.5D thin-layer/quasi-geostrophic approximation when `w` is supplied
When `w !== nothing`, this method still computes only a SINGLE 2D layer's tensor: it includes the
cross terms `S_xz = ½∂ū/∂x, S_yz = ½∂v̄/∂y` in the strain contraction, but sets `S_zz = ∂w̄/∂z ≡ 0` and
never differentiates `u`/`v`/`w` in the vertical — there is no 3rd spatial dimension in the input
arrays for it to differentiate against. This is not a shortcut; it is the standard thin-layer (small
aspect ratio δ = H/L) / quasi-geostrophic scaling used throughout large-scale ocean and atmosphere
dynamics (Vallis, *Atmospheric and Oceanic Fluid Dynamics*, §5; Pedlosky, *Geophysical Fluid
Dynamics*, ch. 6), under which vertical shear terms are genuinely subdominant to horizontal gradients
— valid for the normal large-scale, stratified, rotating-flow regime this package targets, NOT for
homogeneous/isotropic 3D turbulence (e.g. boundary-layer or Rayleigh–Taylor studies), where filtering
genuinely blends all three directions and vertical derivatives are real, not assumed away. The
literature on "vertical structure via coarse-graining" (Aluie, Hecht & Vallis 2018, JPO; Buzzicotti,
Storer, Khatri, Griffies & Aluie 2023, JAMES) analyzes vertical structure by running this SAME 2D/2.5D
method independently at each depth level of a multi-level dataset and comparing/stacking the resulting
profiles — not by computing a coupled 3D tensor — so `Pipeline.coarse_grain_profile`/
[`compute_Π_profile!`](@ref) (looping this method per level) is the literature-matching way to get a
vertical-structure result. A genuinely coupled, all-nine-strain-component 3D method exists separately
for the true-3D Cartesian case (see the `AbstractArray{T,3}` `compute_Π!` method).

# Returns
- `Π`: Energy flux array (same as input), units of W/m³

# Examples
```julia
Π = zeros(100, 100)
compute_Π!(Π, u, v, nothing, grid, TopHatKernel(), 30000.0)
# Π now contains energy flux at 30 km scale
```

# References
- Aluie et al. (2018): https://doi.org/10.1175/JPO-D-17-0100.1
- Aluie (2019): https://doi.org/10.1007/s13137-019-0123-9
- Vallis, G.K., *Atmospheric and Oceanic Fluid Dynamics*, 2nd ed., Cambridge University Press, 2017.
- Pedlosky, J., *Geophysical Fluid Dynamics*, 2nd ed., Springer, 1987.
- Aluie, Hecht & Vallis (2018), *J. Phys. Oceanogr.* 48(2): https://doi.org/10.1175/JPO-D-17-0100.1
- Buzzicotti, Storer, Khatri, Griffies & Aluie (2023), *J. Adv. Model. Earth Syst.*:
  https://doi.org/10.1029/2021MS002583
"""
function compute_Π!(
    Π::AbstractMatrix{T},
    u::AbstractMatrix,
    v::AbstractMatrix,
    w::Union{Nothing, AbstractMatrix}, # nothing or zeros for 2D
    grid::Grids.StructuredGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    workspace::Union{Nothing, ΠWorkspace} = nothing,
    filter_plan::Union{Nothing, Filtering.AbstractFilterPlan} = nothing,
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable()
) where {T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}}
    _validate_field_sizes(grid, Π, u, v, w)
    ws = workspace === nothing ? ΠWorkspace(grid) : workspace
    # Build the filter footprint/plan ONCE for this scale; reused for every velocity component and
    # quadratic product below (instead of rebuilding it on each of the ~9 filterings), and — when the
    # caller passes a prebuilt `filter_plan` (e.g. a repeated call at a fixed scale, such as
    # `compute_Π_profile!`'s per-level loop) — reused ACROSS calls too, instead of rebuilding it fresh
    # every time despite `workspace` already being supplied (the footprint rebuild, not the workspace
    # scratch arrays, was the dominant per-call allocation until this was added).
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend) : filter_plan
    # A StructuredGrid's ddx!/ddy! need no prebuilt derivative cache (pass `nothing`).
    return _compute_Π_2d!(Π, u, v, w, grid, ws, plan, nothing)
end

"""
    compute_Π_profile!(Π, u, v, w, grid, kernel, scale; workspace=nothing, backend=AutoBackend(), mask_strategy=Deformable())

Depth-profile energy flux: given 3D `(lon, lat, depth)` velocity arrays, runs the 2D/2.5D
[`compute_Π!`](@ref) INDEPENDENTLY at each depth level — the literature-standard way to obtain
vertical structure via coarse-graining (Aluie, Hecht & Vallis 2018; Buzzicotti et al. 2023; see the
thin-layer/QG regime note on `compute_Π!`'s docstring) — writing each level's 2D result into the
matching slice of the 3D `Π` output. This is NOT a coupled 3D tensor computation (no vertical
derivatives are taken across levels); for that, see the true-3D `compute_Π!` method instead.
"""
function compute_Π_profile!(
    Π::AbstractArray{T,3},
    u::AbstractArray{T,3},
    v::AbstractArray{T,3},
    w::Union{Nothing, AbstractArray{T,3}},
    grid::Grids.StructuredGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    workspace::Union{Nothing, ΠWorkspace} = nothing,
    filter_plan::Union{Nothing, Filtering.AbstractFilterPlan} = nothing,
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
) where {T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}}
    Nlevels = size(u, 3)
    size(Π, 3) == Nlevels || throw(DimensionMismatch(
        "Π has $(size(Π, 3)) depth levels, u has $Nlevels",
    ))
    ws = workspace === nothing ? ΠWorkspace(grid) : workspace
    # Every level shares the SAME grid/kernel/scale, so the filter footprint is genuinely identical
    # across the whole loop — build it once here and pass it through, instead of `compute_Π!`
    # rebuilding it fresh at every level (the footprint rebuild, not the `ws` scratch arrays, was the
    # dominant per-call allocation before `filter_plan` reuse was added). A caller sweeping multiple
    # scales (`coarse_grain_profile`) can pass an externally prebuilt `filter_plan` to avoid rebuilding
    # across scale iterations too.
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend) : filter_plan
    for k in 1:Nlevels
        wk = w === nothing ? nothing : view(w, :, :, k)
        compute_Π!(
            view(Π, :, :, k), view(u, :, :, k), view(v, :, :, k), wk, grid, kernel, scale;
            workspace = ws, filter_plan = plan, backend = backend, mask_strategy = mask_strategy,
        )
    end
    return Π
end

# ---------------------------------------------------------------------------
# Shared 2D driver for the per-point tensor physics (rotation / SFS stress / strain contraction),
# called by BOTH the StructuredGrid and CurvilinearGrid `compute_Π!` methods. Both grids reach their
# geometry only through `Grids.coords`/`isactive` and the `ddx!`/`ddy!` operators, so this one kernel
# serves both — no duplicated tensor math. `deriv_plan` is `nothing` for a StructuredGrid (whose
# `ddx!`/`ddy!` take no cache) or a prebuilt `Derivatives.WLSQGradientPlan` for a CurvilinearGrid.
# ---------------------------------------------------------------------------

# Derivative dispatch: StructuredGrid ignores the (nothing) plan; CurvilinearGrid/UnstructuredGrid use
# their respective (index-offset vs node-index) WLSQ plans.
@inline _ddx!(out, f, grid::Grids.StructuredGrid, ::Nothing) = Derivatives.ddx!(out, f, grid)
@inline _ddy!(out, f, grid::Grids.StructuredGrid, ::Nothing) = Derivatives.ddy!(out, f, grid)
@inline _ddx!(out, f, grid::Grids.CurvilinearGrid, plan::Derivatives.WLSQGradientPlan) =
    Derivatives.ddx!(out, f, grid, plan)
@inline _ddy!(out, f, grid::Grids.CurvilinearGrid, plan::Derivatives.WLSQGradientPlan) =
    Derivatives.ddy!(out, f, grid, plan)
@inline _ddx!(out, f, grid::Grids.UnstructuredGrid, plan::Derivatives.UnstructuredWLSQGradientPlan) =
    Derivatives.ddx!(out, f, grid, plan)
@inline _ddy!(out, f, grid::Grids.UnstructuredGrid, plan::Derivatives.UnstructuredWLSQGradientPlan) =
    Derivatives.ddy!(out, f, grid, plan)

# Rotate a planetary-Cartesian symmetric SFS stress tensor (given as its xx/xy/xz/yy/yz/zz
# components) to the local (east, north, radial) frame at (λ,φ) — the exact algebraic transcription
# of the per-point rotation both `_compute_Π_2d!`'s spherical branch and the new 1D-node-indexed
# `_compute_Π_1d!` need (a spherical grid's tensor-rotation physics is identical regardless of how
# points happen to be indexed), so this shared scalar kernel avoids duplicating that algebra between
# the two array-shape-specific drivers. Always returns all six local components; callers that don't
# have a genuine radial/vertical velocity simply pass zero for `txz`/`tyz`/`tzz` and discard
# `τer`/`τnr`/`τrr` (matching the existing "not computed when !has_w" convention).
@inline function _rotate_stress_to_local_enr(
    txx::T, txy::T, txz::T, tyy::T, tyz::T, tzz::T, λ::T, φ::T,
) where {T<:AbstractFloat}
    sinφ, cosφ = sin(φ), cos(φ)
    sinλ, cosλ = sin(λ), cos(λ)
    # T * e_east, T * e_north, T * e_radial (e_east = [-sinλ,cosλ,0], e_north = [-sinφcosλ,-sinφsinλ,cosφ],
    # e_radial = [cosφcosλ,cosφsinλ,sinφ]).
    te_x = txx * (-sinλ) + txy * cosλ
    te_y = txy * (-sinλ) + tyy * cosλ
    te_z = txz * (-sinλ) + tyz * cosλ
    tn_x = txx * (-sinφ * cosλ) + txy * (-sinφ * sinλ) + txz * cosφ
    tn_y = txy * (-sinφ * cosλ) + tyy * (-sinφ * sinλ) + tyz * cosφ
    tn_z = txz * (-sinφ * cosλ) + tyz * (-sinφ * sinλ) + tzz * cosφ
    tr_x = txx * (cosφ * cosλ) + txy * (cosφ * sinλ) + txz * sinφ
    tr_y = txy * (cosφ * cosλ) + tyy * (cosφ * sinλ) + tyz * sinφ
    tr_z = txz * (cosφ * cosλ) + tyz * (cosφ * sinλ) + tzz * sinφ
    τee = te_x * (-sinλ) + te_y * cosλ
    τen = te_x * (-sinφ * cosλ) + te_y * (-sinφ * sinλ) + te_z * cosφ
    τer = te_x * (cosφ * cosλ) + te_y * (cosφ * sinλ) + te_z * sinφ
    τnn = tn_x * (-sinφ * cosλ) + tn_y * (-sinφ * sinλ) + tn_z * cosφ
    τnr = tn_x * (cosφ * cosλ) + tn_y * (cosφ * sinλ) + tn_z * sinφ
    τrr = tr_x * (cosφ * cosλ) + tr_y * (cosφ * sinλ) + tr_z * sinφ
    return τee, τen, τer, τnn, τnr, τrr
end

# Symmetric SFS tensor contraction S̄_ij τ_ij — the scalar sum shared by every `compute_Π!` driver's
# final step (2D contraction, or the full six-term 3D contraction when a vertical/radial component
# exists). Factored out so `_compute_Π_2d!` and `_compute_Π_1d!` share the identical arithmetic.
@inline _sfs_contraction(Sxx::T, Sxy::T, Syy::T, τxx::T, τxy::T, τyy::T) where {T<:AbstractFloat} =
    Sxx * τxx + T(2) * Sxy * τxy + Syy * τyy

@inline _sfs_contraction(
    Sxx::T, Sxy::T, Sxz::T, Syy::T, Syz::T, Szz::T, τxx::T, τxy::T, τxz::T, τyy::T, τyz::T, τzz::T,
) where {T<:AbstractFloat} =
    Sxx * τxx + T(2) * Sxy * τxy + Syy * τyy + T(2) * Sxz * τxz + T(2) * Syz * τyz + Szz * τzz

function _compute_Π_2d!(
    Π::AbstractMatrix{T},
    u::AbstractMatrix,
    v::AbstractMatrix,
    w::Union{Nothing, AbstractMatrix},
    grid::Grids.AbstractGrid{G,T},
    ws::ΠWorkspace,
    plan::Filtering.AbstractFilterPlan,
    deriv_plan,
) where {T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}}
    Nlon, Nlat = Grids.size_tuple(grid)
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
        _ddx!(ws.S_xx, ws.u_filt, grid, deriv_plan)
        # S_yy = ∂v̄/∂y
        _ddy!(ws.S_yy, ws.v_filt, grid, deriv_plan)
        # S_xy = 0.5 * (∂ū/∂y + ∂v̄/∂x)
        _ddy!(ws.S_xy, ws.u_filt, grid, deriv_plan)
        _ddx!(ws.scratch, ws.v_filt, grid, deriv_plan)
        @. ws.S_xy = T(0.5) * (ws.S_xy + ws.scratch)
        
        if has_w
            # S_xz = 0.5 * (∂ū/∂z + ∂w̄/∂x) (assumes ∂/∂z is zero or handles vertical layers)
            _ddx!(ws.S_xz, ws.w_filt, grid, deriv_plan)
            @. ws.S_xz = T(0.5) * ws.S_xz
            
            # S_yz = 0.5 * (∂v̄/∂z + ∂w̄/∂y)
            _ddy!(ws.S_yz, ws.w_filt, grid, deriv_plan)
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
            for i in 1:Nlon
                if Grids.isactive(grid, i, j)
                    λ, φ = Grids.coords(grid, i, j)
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
            for i in 1:Nlon
                if Grids.isactive(grid, i, j)
                    λ, φ = Grids.coords(grid, i, j)
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
        
        # Transform planetary filtered products to local stresses at each grid point, via the shared
        # `_rotate_stress_to_local_enr` scalar kernel (τ_local = R' * ( [u_i u_j]̄ - ū_i ū_j ) * R for
        # the orthogonal local rotation R = [e_east, e_north, e_radial]) — see that function for the
        # rotation algebra itself, kept in one place so the 1D `UnstructuredGrid` driver below shares it.
        for j in 1:Nlat
            for i in 1:Nlon
                if Grids.isactive(grid, i, j)
                    λ, φ = Grids.coords(grid, i, j)
                    # NOTE: txz/tyz/tzz are NOT gated on `has_w` — `ws.uz`/`uz_filt` (the planetary
                    # Cartesian Z-component) is generally nonzero even for a purely horizontal
                    # velocity (u_vertical=0 still rotates into a nonzero Z through cosφ/sinφ), so
                    # these cross terms are real, always-defined contributions that DO feed into the
                    # rotated τee/τen/τnn below — only τer/τnr/τrr (the genuine radial-direction
                    # stresses) are specifically a `has_w` (real vertical-velocity) quantity, and are
                    # the only components gated below.
                    txx = ws.uu_filt[i, j] - ws.ux_filt[i, j] * ws.ux_filt[i, j]
                    txy = ws.uv_filt[i, j] - ws.ux_filt[i, j] * ws.uy_filt[i, j]
                    tyy = ws.vv_filt[i, j] - ws.uy_filt[i, j] * ws.uy_filt[i, j]
                    txz = ws.uw_filt[i, j] - ws.ux_filt[i, j] * ws.uz_filt[i, j]
                    tyz = ws.vw_filt[i, j] - ws.uy_filt[i, j] * ws.uz_filt[i, j]
                    tzz = ws.ww_filt[i, j] - ws.uz_filt[i, j] * ws.uz_filt[i, j]
                    τee, τen, τer, τnn, τnr, τrr = _rotate_stress_to_local_enr(txx, txy, txz, tyy, tyz, tzz, λ, φ)
                    ws.τ_xx[i, j] = τee
                    ws.τ_yy[i, j] = τnn
                    ws.τ_xy[i, j] = τen
                    if has_w
                        ws.τ_xz[i, j] = τer
                        ws.τ_yz[i, j] = τnr
                        ws.τ_zz[i, j] = τrr
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
        _ddx!(ws.S_xx, ws.u_filt, grid, deriv_plan)
        # S_nn = 1/R * ∂v̄_n/∂φ
        _ddy!(ws.S_yy, ws.v_filt, grid, deriv_plan)
        # S_en = 0.5 * ( 1/(R cosφ) ∂v̄_n/∂λ + 1/R ∂ū_e/∂φ + ū_e * sinφ / (R cosφ) )
        _ddy!(ws.S_xy, ws.u_filt, grid, deriv_plan)
        _ddx!(ws.scratch, ws.v_filt, grid, deriv_plan)
        
        R = grid.geometry.R
        for j in 1:Nlat
            for i in 1:Nlon
                if Grids.isactive(grid, i, j)
                    _, φ = Grids.coords(grid, i, j)
                    cosφ = cos(φ)
                    sinφ = sin(φ)
                    tan_fact = abs(cosφ) > T(1e-12) ? sinφ / (R * cosφ) : zero(T)
                    # S_ee correction
                    ws.S_xx[i, j] -= ws.v_filt[i, j] * tan_fact
                    
                    # S_en correction
                    ws.S_xy[i, j] = T(0.5) * (ws.S_xy[i, j] + ws.scratch[i, j] + ws.u_filt[i, j] * tan_fact)
                end
            end
        end
        
        if has_w
            # S_er = 0.5 * (∂ū_e/∂r + 1/(R cosφ) ∂w̄/∂λ) = 0.5 * 1/(R cosφ) ∂w̄/∂λ (if vertically flat layers)
            _ddx!(ws.S_xz, ws.w_filt, grid, deriv_plan)
            @. ws.S_xz = T(0.5) * ws.S_xz
            
            # S_nr = 0.5 * (∂v̄_n/∂r + 1/R ∂w̄/∂φ) = 0.5 * 1/R ∂w̄/∂φ
            _ddy!(ws.S_yz, ws.w_filt, grid, deriv_plan)
            @. ws.S_yz = T(0.5) * ws.S_yz
            
            # S_rr = ∂w̄/∂r = 0
            fill!(ws.S_zz, zero(T))
        end
    end
    
    # -----------------------------------------------------------------------
    # Tensor contraction: Π = -Σ_ij S̄_ij τ_ij
    # -----------------------------------------------------------------------
    # Since stress & strain rates are symmetric:
    # S̄_ij τ_ij = S_xx*τ_xx + 2*S_xy*τ_xy + S_yy*τ_yy (2D)
    # S̄_ij τ_ij = S_xx*τ_xx + 2*S_xy*τ_xy + S_yy*τ_yy + 2*S_xz*τ_xz + 2*S_yz*τ_yz + S_zz*τ_zz (3D)
    if has_w
        for j in 1:Nlat
            for i in 1:Nlon
                if Grids.isactive(grid, i, j)
                    Π[i, j] = -_sfs_contraction(
                        ws.S_xx[i, j], ws.S_xy[i, j], ws.S_xz[i, j], ws.S_yy[i, j], ws.S_yz[i, j], ws.S_zz[i, j],
                        ws.τ_xx[i, j], ws.τ_xy[i, j], ws.τ_xz[i, j], ws.τ_yy[i, j], ws.τ_yz[i, j], ws.τ_zz[i, j],
                    )
                else
                    Π[i, j] = zero(T)
                end
            end
        end
    else
        for j in 1:Nlat
            for i in 1:Nlon
                if Grids.isactive(grid, i, j)
                    Π[i, j] = -_sfs_contraction(
                        ws.S_xx[i, j], ws.S_xy[i, j], ws.S_yy[i, j], ws.τ_xx[i, j], ws.τ_xy[i, j], ws.τ_yy[i, j],
                    )
                else
                    Π[i, j] = zero(T)
                end
            end
        end
    end

    return Π
end

# 1D node-indexed driver for `UnstructuredGrid` — the exact scalar physics of `_compute_Π_2d!`'s
# spherical/Cartesian branches, looped over `1:N` nodes instead of `(i,j)` pairs, reusing
# `_rotate_stress_to_local_enr`/`_sfs_contraction` so none of that algebra is duplicated.
function _compute_Π_1d!(
    Π::AbstractVector{T},
    u::AbstractVector,
    v::AbstractVector,
    w::Union{Nothing, AbstractVector},
    grid::Grids.UnstructuredGrid{T,G},
    ws::ΠWorkspace,
    plan::Filtering.AbstractFilterPlan,
    deriv_plan::Derivatives.UnstructuredWLSQGradientPlan,
) where {T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}}
    N = length(grid.mask)
    has_w = w !== nothing

    if G <: Geometry.CartesianGeometry{T}
        Filtering.filter_apply!(ws.u_filt, u, plan)
        Filtering.filter_apply!(ws.v_filt, v, plan)
        has_w && Filtering.filter_apply!(ws.w_filt, w, plan)

        @. ws.scratch = u * u
        Filtering.filter_apply!(ws.uu_filt, ws.scratch, plan)
        @. ws.scratch = u * v
        Filtering.filter_apply!(ws.uv_filt, ws.scratch, plan)
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

        @. ws.τ_xx = ws.uu_filt - ws.u_filt * ws.u_filt
        @. ws.τ_xy = ws.uv_filt - ws.u_filt * ws.v_filt
        @. ws.τ_yy = ws.vv_filt - ws.v_filt * ws.v_filt
        if has_w
            @. ws.τ_xz = ws.uw_filt - ws.u_filt * ws.w_filt
            @. ws.τ_yz = ws.vw_filt - ws.v_filt * ws.w_filt
            @. ws.τ_zz = ws.ww_filt - ws.w_filt * ws.w_filt
        end

        _ddx!(ws.S_xx, ws.u_filt, grid, deriv_plan)
        _ddy!(ws.S_yy, ws.v_filt, grid, deriv_plan)
        _ddy!(ws.S_xy, ws.u_filt, grid, deriv_plan)
        _ddx!(ws.scratch, ws.v_filt, grid, deriv_plan)
        @. ws.S_xy = T(0.5) * (ws.S_xy + ws.scratch)

        if has_w
            _ddx!(ws.S_xz, ws.w_filt, grid, deriv_plan)
            @. ws.S_xz = T(0.5) * ws.S_xz
            _ddy!(ws.S_yz, ws.w_filt, grid, deriv_plan)
            @. ws.S_yz = T(0.5) * ws.S_yz
            fill!(ws.S_zz, zero(T))
        end
    else
        # Spherical: local (u,v[,w]) -> planetary Cartesian, filter, rotate back — identical algebra
        # to `_compute_Π_2d!`'s spherical branch, node-indexed.
        for i in 1:N
            if Grids.isactive(grid, i)
                λ, φ = Grids.coords(grid, i)
                w_val = has_w ? w[i] : zero(T)
                p_vel = Geometry.to_planetary_cartesian(grid.geometry, u[i], v[i], w_val, λ, φ)
                ws.ux[i], ws.uy[i], ws.uz[i] = p_vel[1], p_vel[2], p_vel[3]
            else
                ws.ux[i] = zero(T); ws.uy[i] = zero(T); ws.uz[i] = zero(T)
            end
        end

        Filtering.filter_apply!(ws.ux_filt, ws.ux, plan)
        Filtering.filter_apply!(ws.uy_filt, ws.uy, plan)
        Filtering.filter_apply!(ws.uz_filt, ws.uz, plan)

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

        for i in 1:N
            if Grids.isactive(grid, i)
                λ, φ = Grids.coords(grid, i)
                l_vel = Geometry.from_planetary_cartesian(grid.geometry, ws.ux_filt[i], ws.uy_filt[i], ws.uz_filt[i], λ, φ)
                ws.u_filt[i], ws.v_filt[i], ws.w_filt[i] = l_vel[1], l_vel[2], l_vel[3]
            else
                ws.u_filt[i] = zero(T); ws.v_filt[i] = zero(T); ws.w_filt[i] = zero(T)
            end
        end

        for i in 1:N
            if Grids.isactive(grid, i)
                λ, φ = Grids.coords(grid, i)
                txx = ws.uu_filt[i] - ws.ux_filt[i] * ws.ux_filt[i]
                txy = ws.uv_filt[i] - ws.ux_filt[i] * ws.uy_filt[i]
                tyy = ws.vv_filt[i] - ws.uy_filt[i] * ws.uy_filt[i]
                txz = ws.uw_filt[i] - ws.ux_filt[i] * ws.uz_filt[i]
                tyz = ws.vw_filt[i] - ws.uy_filt[i] * ws.uz_filt[i]
                tzz = ws.ww_filt[i] - ws.uz_filt[i] * ws.uz_filt[i]
                τee, τen, τer, τnn, τnr, τrr = _rotate_stress_to_local_enr(txx, txy, txz, tyy, tyz, tzz, λ, φ)
                ws.τ_xx[i] = τee; ws.τ_yy[i] = τnn; ws.τ_xy[i] = τen
                if has_w
                    ws.τ_xz[i] = τer; ws.τ_yz[i] = τnr; ws.τ_zz[i] = τrr
                end
            else
                ws.τ_xx[i] = zero(T); ws.τ_yy[i] = zero(T); ws.τ_xy[i] = zero(T)
                if has_w
                    ws.τ_xz[i] = zero(T); ws.τ_yz[i] = zero(T); ws.τ_zz[i] = zero(T)
                end
            end
        end

        _ddx!(ws.S_xx, ws.u_filt, grid, deriv_plan)
        _ddy!(ws.S_yy, ws.v_filt, grid, deriv_plan)
        _ddy!(ws.S_xy, ws.u_filt, grid, deriv_plan)
        _ddx!(ws.scratch, ws.v_filt, grid, deriv_plan)

        R = grid.geometry.R
        for i in 1:N
            if Grids.isactive(grid, i)
                _, φ = Grids.coords(grid, i)
                cosφ, sinφ = cos(φ), sin(φ)
                tan_fact = abs(cosφ) > T(1e-12) ? sinφ / (R * cosφ) : zero(T)
                ws.S_xx[i] -= ws.v_filt[i] * tan_fact
                ws.S_xy[i] = T(0.5) * (ws.S_xy[i] + ws.scratch[i] + ws.u_filt[i] * tan_fact)
            end
        end

        if has_w
            _ddx!(ws.S_xz, ws.w_filt, grid, deriv_plan)
            @. ws.S_xz = T(0.5) * ws.S_xz
            _ddy!(ws.S_yz, ws.w_filt, grid, deriv_plan)
            @. ws.S_yz = T(0.5) * ws.S_yz
            fill!(ws.S_zz, zero(T))
        end
    end

    if has_w
        for i in 1:N
            Π[i] = Grids.isactive(grid, i) ? -_sfs_contraction(
                ws.S_xx[i], ws.S_xy[i], ws.S_xz[i], ws.S_yy[i], ws.S_yz[i], ws.S_zz[i],
                ws.τ_xx[i], ws.τ_xy[i], ws.τ_xz[i], ws.τ_yy[i], ws.τ_yz[i], ws.τ_zz[i],
            ) : zero(T)
        end
    else
        for i in 1:N
            Π[i] = Grids.isactive(grid, i) ?
                -_sfs_contraction(ws.S_xx[i], ws.S_xy[i], ws.S_yy[i], ws.τ_xx[i], ws.τ_xy[i], ws.τ_yy[i]) :
                zero(T)
        end
    end

    return Π
end

"""
    compute_Π!(Π, u, v, w, grid::UnstructuredGrid, kernel, scale; workspace=nothing, deriv_plan=nothing, backend=AutoBackend(), mask_strategy=Deformable(), method=Spectral())

Cross-scale kinetic energy flux Π = -S̄_ij τ_ij on a [`Grids.UnstructuredGrid`](@ref) (scattered
points, node-indexed) — the same physics as the 2D methods (planetary-Cartesian rotation for
spherical geometry), via `_compute_Π_1d!`. The resolved strain uses the node-indexed WLSQ
gradient (`Derivatives.ddx!`/`ddy!` with an `UnstructuredWLSQGradientPlan`). `UnstructuredGrid` has no
real-space (direct-sum) filtering engine yet — only spectral (FINUFFT/NUFSHT) — so `method` defaults
to `Spectral()`, unlike the other grid types' `DirectSum()` default.
"""
function compute_Π!(
    Π::AbstractVector{T},
    u::AbstractVector,
    v::AbstractVector,
    w::Union{Nothing, AbstractVector},
    grid::Grids.UnstructuredGrid{T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    workspace::Union{Nothing, ΠWorkspace} = nothing,
    deriv_plan::Union{Nothing, Derivatives.UnstructuredWLSQGradientPlan} = nothing,
    filter_plan::Union{Nothing, Filtering.AbstractFilterPlan} = nothing,
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    method::Filtering.AbstractFilterMethod = Filtering.Spectral(),
) where {T<:AbstractFloat}
    _validate_field_sizes(grid, Π, u, v, w)
    ws = workspace === nothing ? ΠWorkspace(grid) : workspace
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend, method=method) : filter_plan
    dplan = deriv_plan === nothing ? Derivatives.WLSQGradientPlan(grid) : deriv_plan
    return _compute_Π_1d!(Π, u, v, w, grid, ws, plan, dplan)
end

"""
    compute_Π!(Π, u, v, w, grid::CurvilinearGrid, kernel, scale; workspace=nothing, deriv_plan=nothing, backend=AutoBackend(), mask_strategy=Deformable())

Cross-scale kinetic energy flux Π = -S̄_ij τ_ij on a [`Grids.CurvilinearGrid`](@ref). Identical
physics to the `StructuredGrid` 2D method — it shares the same `_compute_Π_2d!` tensor kernel
— but the resolved strain uses the curvilinear WLSQ gradient (`Derivatives.ddx!`/`ddy!` with a
`WLSQGradientPlan`) and real-space filtering uses the scattered per-point footprint. Pass a prebuilt
`deriv_plan = WLSQGradientPlan(grid)` (and a reusable `workspace`) to avoid rebuilding them per call
across a scale sweep.
"""
function compute_Π!(
    Π::AbstractMatrix{T},
    u::AbstractMatrix,
    v::AbstractMatrix,
    w::Union{Nothing, AbstractMatrix},
    grid::Grids.CurvilinearGrid{T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    workspace::Union{Nothing, ΠWorkspace} = nothing,
    deriv_plan::Union{Nothing, Derivatives.WLSQGradientPlan} = nothing,
    filter_plan::Union{Nothing, Filtering.AbstractFilterPlan} = nothing,
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
) where {T<:AbstractFloat}
    _validate_field_sizes(grid, Π, u, v, w)
    ws = workspace === nothing ? ΠWorkspace(grid) : workspace
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend) : filter_plan
    dplan = deriv_plan === nothing ? Derivatives.WLSQGradientPlan(grid) : deriv_plan
    return _compute_Π_2d!(Π, u, v, w, grid, ws, plan, dplan)
end

"""
    compute_Π!(Π::AbstractArray{T,3}, u, v, w, grid::StructuredGrid{Cartesian,T,3}, kernel, scale; mask_strategy=Deformable(), backend=AutoBackend())

Full **three-dimensional** Cartesian cross-scale energy flux Π = -S̄_ij τ_ij with all nine strain
components (the diagonal `S_zz = ∂w̄/∂z` and the off-diagonals `S_xz, S_yz` carry genuine vertical
derivatives, unlike the 2.5D layer-by-layer path). The 3D grid carries a 3D mask, so dry cells are
handled per-cell in all three directions.

The contraction is the symmetric six-term sum
`S̄:τ = S_xx τ_xx + S_yy τ_yy + S_zz τ_zz + 2(S_xy τ_xy + S_xz τ_xz + S_yz τ_yz)`.

Dispatched on a 3D output array + 3D Cartesian grid (the 2D method takes an `AbstractMatrix`); see
the separate `StructuredGrid{Spherical,T,3}` method below for the spherical volumetric case (genuine
radius axis, real `∂/∂r`, full curvature-corrected strain). Pass a reusable `workspace`
(a [`ΠWorkspace`](@ref), dimension-generic) to avoid reallocating temporaries on every call — the
same "build once, reuse many" pattern the 2D driver uses, now that `ΠWorkspace` infers its array type
from the grid's actual shape instead of hardcoding `Matrix`.
"""
function compute_Π!(
    Π::AbstractArray{T,3},
    u::AbstractArray{<:Any,3},
    v::AbstractArray{<:Any,3},
    w::AbstractArray{<:Any,3},
    grid::Grids.StructuredGrid{G,T,3},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    workspace::Union{Nothing, ΠWorkspace} = nothing,
    filter_plan::Union{Nothing, Filtering.AbstractFilterPlan} = nothing,
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
) where {T<:AbstractFloat, G<:Geometry.CartesianGeometry{T}}
    _validate_field_sizes(grid, Π, u, v, w)
    ws = workspace === nothing ? ΠWorkspace(grid) : workspace
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend) : filter_plan

    # Filtered velocities and the six independent filtered quadratic products.
    Filtering.filter_apply!(ws.u_filt, u, plan)
    Filtering.filter_apply!(ws.v_filt, v, plan)
    Filtering.filter_apply!(ws.w_filt, w, plan)
    @. ws.scratch = u * u; Filtering.filter_apply!(ws.uu_filt, ws.scratch, plan)
    @. ws.scratch = u * v; Filtering.filter_apply!(ws.uv_filt, ws.scratch, plan)
    @. ws.scratch = u * w; Filtering.filter_apply!(ws.uw_filt, ws.scratch, plan)
    @. ws.scratch = v * v; Filtering.filter_apply!(ws.vv_filt, ws.scratch, plan)
    @. ws.scratch = v * w; Filtering.filter_apply!(ws.vw_filt, ws.scratch, plan)
    @. ws.scratch = w * w; Filtering.filter_apply!(ws.ww_filt, ws.scratch, plan)

    # Subfilter stress τ_ij = ⟨u_i u_j⟩ - ū_i ū_j (symmetric, six components).
    @. ws.τ_xx = ws.uu_filt - ws.u_filt * ws.u_filt
    @. ws.τ_xy = ws.uv_filt - ws.u_filt * ws.v_filt
    @. ws.τ_xz = ws.uw_filt - ws.u_filt * ws.w_filt
    @. ws.τ_yy = ws.vv_filt - ws.v_filt * ws.v_filt
    @. ws.τ_yz = ws.vw_filt - ws.v_filt * ws.w_filt
    @. ws.τ_zz = ws.ww_filt - ws.w_filt * ws.w_filt

    # Strain S̄_ij = ½(∂ū_i/∂x_j + ∂ū_j/∂x_i): three diagonals + three off-diagonals.
    Derivatives.ddx!(ws.S_xx, ws.u_filt, grid)
    Derivatives.ddy!(ws.S_yy, ws.v_filt, grid)
    Derivatives.ddz!(ws.S_zz, ws.w_filt, grid)
    Derivatives.ddy!(ws.S_xy, ws.u_filt, grid); Derivatives.ddx!(ws.scratch, ws.v_filt, grid)
    @. ws.S_xy = T(0.5) * (ws.S_xy + ws.scratch)
    Derivatives.ddz!(ws.S_xz, ws.u_filt, grid); Derivatives.ddx!(ws.scratch, ws.w_filt, grid)
    @. ws.S_xz = T(0.5) * (ws.S_xz + ws.scratch)
    Derivatives.ddz!(ws.S_yz, ws.v_filt, grid); Derivatives.ddy!(ws.scratch, ws.w_filt, grid)
    @. ws.S_yz = T(0.5) * (ws.S_yz + ws.scratch)

    mask = grid.mask
    @inbounds @. Π = ifelse(
        mask,
        -(ws.S_xx * ws.τ_xx + ws.S_yy * ws.τ_yy + ws.S_zz * ws.τ_zz +
          T(2) * (ws.S_xy * ws.τ_xy + ws.S_xz * ws.τ_xz + ws.S_yz * ws.τ_yz)),
        zero(T),
    )
    return Π
end

"""
    compute_Π!(Π::AbstractArray{T,3}, u, v, w, grid::StructuredGrid{Spherical,T,3}, kernel, scale; workspace=nothing, backend=AutoBackend(), mask_strategy=Deformable())

Full **three-dimensional spherical** cross-scale energy flux Π = -S̄_ij τ_ij: a genuine radius axis
`r[k]` (absolute distance from the planet center — see [`Grids.StructuredGrid`](@ref)'s 3D
constructor) and real vertical derivatives `∂/∂r`, unlike the 2.5D layer-by-layer path (which drops
the `u_r/r` curvature terms in `S_ee`/`S_nn` and the `S_er`/`S_nr`/`S_rr` radial strain entirely, since
it has no radial axis to differentiate against).

Velocities are rotated to planetary Cartesian for filtering (Aluie 2019 commutativity), then rotated
back to local (east, north, radial) — reusing the exact same `_rotate_stress_to_local_enr`/
`_sfs_contraction` kernels the 2D spherical driver uses, since that rotation algebra is
already fully 3×3-general (only the 2D/2.5D caller previously discarded the radial components). The
new physics here is the strain: the standard spherical strain-rate tensor in orthogonal curvilinear
coordinates (scale factors `h_λ = r cosφ, h_φ = r, h_r = 1`),

    S_ee = (1/(r cosφ))∂ū_e/∂λ - v̄_n·tanφ/r + w̄_r/r
    S_nn = (1/r)∂v̄_n/∂φ + w̄_r/r
    S_rr = ∂w̄_r/∂r
    S_en = ½[(1/(r cosφ))∂v̄_n/∂λ + (1/r)∂ū_e/∂φ + ū_e·tanφ/r]
    S_er = ½[(1/(r cosφ))∂w̄_r/∂λ + ∂ū_e/∂r - ū_e/r]
    S_nr = ½[(1/r)∂w̄_r/∂φ + ∂v̄_n/∂r - v̄_n/r]

where `∂/∂λ`/`∂/∂φ`/`∂/∂r` are [`Derivatives.ddx!`](@ref)/[`Derivatives.ddy!`](@ref)/[`Derivatives.ddz!`](@ref) (already
metric-scaled using the LOCAL `r[k]`, not the fixed reference radius). Pass a reusable `workspace`
to avoid reallocating temporaries on every call, exactly as the Cartesian 3D method does.
"""
function compute_Π!(
    Π::AbstractArray{T,3},
    u::AbstractArray{<:Any,3},
    v::AbstractArray{<:Any,3},
    w::AbstractArray{<:Any,3},
    grid::Grids.StructuredGrid{G,T,3},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    workspace::Union{Nothing, ΠWorkspace} = nothing,
    filter_plan::Union{Nothing, Filtering.AbstractFilterPlan} = nothing,
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
) where {T<:AbstractFloat, G<:Geometry.SphericalGeometry{T}}
    _validate_field_sizes(grid, Π, u, v, w)
    ws = workspace === nothing ? ΠWorkspace(grid) : workspace
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend) : filter_plan
    Nlon, Nlat, Nr = Grids.size_tuple(grid)

    # Rotate local (east, north, radial) velocity to planetary Cartesian at each point.
    @inbounds for k in 1:Nr, j in 1:Nlat, i in 1:Nlon
        if Grids.isactive(grid, i, j, k)
            λ, φ, _ = Grids.coords(grid, i, j, k)
            p_vel = Geometry.to_planetary_cartesian(grid.geometry, u[i, j, k], v[i, j, k], w[i, j, k], λ, φ)
            ws.ux[i, j, k] = p_vel[1]; ws.uy[i, j, k] = p_vel[2]; ws.uz[i, j, k] = p_vel[3]
        else
            ws.ux[i, j, k] = zero(T); ws.uy[i, j, k] = zero(T); ws.uz[i, j, k] = zero(T)
        end
    end

    Filtering.filter_apply!(ws.ux_filt, ws.ux, plan)
    Filtering.filter_apply!(ws.uy_filt, ws.uy, plan)
    Filtering.filter_apply!(ws.uz_filt, ws.uz, plan)
    @. ws.scratch = ws.ux * ws.ux; Filtering.filter_apply!(ws.uu_filt, ws.scratch, plan)
    @. ws.scratch = ws.ux * ws.uy; Filtering.filter_apply!(ws.uv_filt, ws.scratch, plan)
    @. ws.scratch = ws.ux * ws.uz; Filtering.filter_apply!(ws.uw_filt, ws.scratch, plan)
    @. ws.scratch = ws.uy * ws.uy; Filtering.filter_apply!(ws.vv_filt, ws.scratch, plan)
    @. ws.scratch = ws.uy * ws.uz; Filtering.filter_apply!(ws.vw_filt, ws.scratch, plan)
    @. ws.scratch = ws.uz * ws.uz; Filtering.filter_apply!(ws.ww_filt, ws.scratch, plan)

    # Rotate filtered planetary velocities back to local (east, north, radial).
    @inbounds for k in 1:Nr, j in 1:Nlat, i in 1:Nlon
        if Grids.isactive(grid, i, j, k)
            λ, φ, _ = Grids.coords(grid, i, j, k)
            l_vel = Geometry.from_planetary_cartesian(
                grid.geometry, ws.ux_filt[i, j, k], ws.uy_filt[i, j, k], ws.uz_filt[i, j, k], λ, φ,
            )
            ws.u_filt[i, j, k] = l_vel[1]; ws.v_filt[i, j, k] = l_vel[2]; ws.w_filt[i, j, k] = l_vel[3]
        else
            ws.u_filt[i, j, k] = zero(T); ws.v_filt[i, j, k] = zero(T); ws.w_filt[i, j, k] = zero(T)
        end
    end

    # Rotate filtered planetary quadratic products into the local (east,north,radial) stress tensor.
    @inbounds for k in 1:Nr, j in 1:Nlat, i in 1:Nlon
        if Grids.isactive(grid, i, j, k)
            λ, φ, _ = Grids.coords(grid, i, j, k)
            txx = ws.uu_filt[i, j, k] - ws.ux_filt[i, j, k] * ws.ux_filt[i, j, k]
            txy = ws.uv_filt[i, j, k] - ws.ux_filt[i, j, k] * ws.uy_filt[i, j, k]
            tyy = ws.vv_filt[i, j, k] - ws.uy_filt[i, j, k] * ws.uy_filt[i, j, k]
            txz = ws.uw_filt[i, j, k] - ws.ux_filt[i, j, k] * ws.uz_filt[i, j, k]
            tyz = ws.vw_filt[i, j, k] - ws.uy_filt[i, j, k] * ws.uz_filt[i, j, k]
            tzz = ws.ww_filt[i, j, k] - ws.uz_filt[i, j, k] * ws.uz_filt[i, j, k]
            τee, τen, τer, τnn, τnr, τrr = _rotate_stress_to_local_enr(txx, txy, txz, tyy, tyz, tzz, λ, φ)
            ws.τ_xx[i, j, k] = τee; ws.τ_xy[i, j, k] = τen; ws.τ_xz[i, j, k] = τer
            ws.τ_yy[i, j, k] = τnn; ws.τ_yz[i, j, k] = τnr; ws.τ_zz[i, j, k] = τrr
        else
            ws.τ_xx[i, j, k] = zero(T); ws.τ_xy[i, j, k] = zero(T); ws.τ_xz[i, j, k] = zero(T)
            ws.τ_yy[i, j, k] = zero(T); ws.τ_yz[i, j, k] = zero(T); ws.τ_zz[i, j, k] = zero(T)
        end
    end

    # Strain: ddx!/ddy!/ddz! are already metric-scaled (1/(r cosφ), 1/r, and a plain radial
    # derivative respectively, using the LOCAL r[k] at each level), so this gives the "flat" part of
    # each component; the curvature-correction terms are added in the loop below.
    Derivatives.ddx!(ws.S_xx, ws.u_filt, grid)
    Derivatives.ddy!(ws.S_yy, ws.v_filt, grid)
    Derivatives.ddz!(ws.S_zz, ws.w_filt, grid)
    Derivatives.ddy!(ws.S_xy, ws.u_filt, grid); Derivatives.ddx!(ws.scratch, ws.v_filt, grid)
    @. ws.S_xy = T(0.5) * (ws.S_xy + ws.scratch)
    Derivatives.ddz!(ws.S_xz, ws.u_filt, grid); Derivatives.ddx!(ws.scratch, ws.w_filt, grid)
    @. ws.S_xz = T(0.5) * (ws.S_xz + ws.scratch)
    Derivatives.ddz!(ws.S_yz, ws.v_filt, grid); Derivatives.ddy!(ws.scratch, ws.w_filt, grid)
    @. ws.S_yz = T(0.5) * (ws.S_yz + ws.scratch)

    @inbounds for k in 1:Nr, j in 1:Nlat, i in 1:Nlon
        if Grids.isactive(grid, i, j, k)
            _, φ, rk = Grids.coords(grid, i, j, k)
            cosφ = cos(φ); sinφ = sin(φ)
            tan_fact = abs(cosφ) > T(1e-12) ? sinφ / (rk * cosφ) : zero(T)
            inv_r = one(T) / rk
            u_e = ws.u_filt[i, j, k]; v_n = ws.v_filt[i, j, k]; w_r = ws.w_filt[i, j, k]
            ws.S_xx[i, j, k] += w_r * inv_r - v_n * tan_fact
            ws.S_yy[i, j, k] += w_r * inv_r
            ws.S_xy[i, j, k] += T(0.5) * u_e * tan_fact
            ws.S_xz[i, j, k] -= T(0.5) * u_e * inv_r
            ws.S_yz[i, j, k] -= T(0.5) * v_n * inv_r
        end
    end

    @inbounds for k in 1:Nr, j in 1:Nlat, i in 1:Nlon
        Π[i, j, k] = Grids.isactive(grid, i, j, k) ? -_sfs_contraction(
            ws.S_xx[i, j, k], ws.S_xy[i, j, k], ws.S_xz[i, j, k],
            ws.S_yy[i, j, k], ws.S_yz[i, j, k], ws.S_zz[i, j, k],
            ws.τ_xx[i, j, k], ws.τ_xy[i, j, k], ws.τ_xz[i, j, k],
            ws.τ_yy[i, j, k], ws.τ_yz[i, j, k], ws.τ_zz[i, j, k],
        ) : zero(T)
    end
    return Π
end

"""
    compute_Π!(Π::AbstractVector, u, grid::StructuredGrid{Cartesian,T,1}, kernel, scale; workspace=nothing, backend=AutoBackend(), mask_strategy=Deformable())

1D cross-scale energy flux Π = -S̄_xx τ_xx on a genuinely 1D `StructuredGrid` (a single scalar
velocity component `u` along one axis — the 1D analog of the 2D tensor contraction, which reduces to
a single term since there's only one strain/stress component). Not the 2D-with-singleton-dimension
case (which reuses the 2D methods directly).
"""
function compute_Π!(
    Π::AbstractVector{T},
    u::AbstractVector,
    grid::Grids.StructuredGrid{G,T,1},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    workspace::Union{Nothing, ΠWorkspace} = nothing,
    filter_plan::Union{Nothing, Filtering.AbstractFilterPlan} = nothing,
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
) where {T<:AbstractFloat, G<:Geometry.CartesianGeometry{T}}
    _validate_field_sizes(grid, Π, u)
    ws = workspace === nothing ? ΠWorkspace(grid) : workspace
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend) : filter_plan

    Filtering.filter_apply!(ws.u_filt, u, plan)
    @. ws.scratch = u * u
    Filtering.filter_apply!(ws.uu_filt, ws.scratch, plan)
    @. ws.τ_xx = ws.uu_filt - ws.u_filt * ws.u_filt

    Derivatives.ddx!(ws.S_xx, ws.u_filt, grid)

    mask = grid.mask
    @inbounds @. Π = ifelse(mask, -(ws.S_xx * ws.τ_xx), zero(T))
    return Π
end

# ---------------------------------------------------------------------------
# Filtering Energy Spectrum E(ℓ)
# ---------------------------------------------------------------------------

"""
    cumulative_energy!(spectrum, u, v, w, grid, kernel, scales; workspace=nothing, backend=AutoBackend(), mask_strategy=Deformable())

In-place [`cumulative_energy`](@ref): writes into the caller-supplied `spectrum` vector and, when
`workspace` (a [`ΠWorkspace`](@ref)) is supplied, reuses its `u_filt`/`v_filt`/`w_filt` scratch arrays
instead of allocating fresh ones — the same buffers `compute_Π!` already fills at each scale, so a
`coarse_grain!` sweep pays for this filtered-velocity scratch space once, not twice.
"""
function cumulative_energy!(
    spectrum::AbstractVector{T},
    u::AbstractArray,
    v::AbstractArray,
    w::Union{Nothing, AbstractArray},
    grid::Union{Grids.StructuredGrid{G,T}, Grids.CurvilinearGrid{T,G}, Grids.UnstructuredGrid{T,G}},
    kernel::Kernels.AbstractFilterKernel,
    scales::AbstractVector;
    workspace::Union{Nothing, ΠWorkspace} = nothing,
    filter_plans::Union{Nothing, AbstractVector} = nothing,
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    method::Filtering.AbstractFilterMethod = Filtering.DirectSum(),
) where {T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}}
    gsz = Grids.size_tuple(grid)
    size(u) == gsz || throw(DimensionMismatch("u has size $(size(u)), grid expects $gsz"))
    size(v) == gsz || throw(DimensionMismatch("v has size $(size(v)), grid expects $gsz"))
    w === nothing || size(w) == gsz || throw(DimensionMismatch("w has size $(size(w)), grid expects $gsz"))

    Nscales = length(scales)
    length(spectrum) == Nscales || throw(DimensionMismatch(
        "spectrum has length $(length(spectrum)), expected $Nscales (= length(scales))",
    ))

    ws = workspace === nothing ? ΠWorkspace(grid) : workspace
    u_filt, v_filt, w_filt = ws.u_filt, ws.v_filt, ws.w_filt

    # Dimension-generic active-cell iteration: `Tuple(I)...` splats to (i,) for a 1D UnstructuredGrid
    # or (i,j) for a 2D Structured/CurvilinearGrid, matching each grid's own `isactive`/`area` arity.
    idxs = CartesianIndices(u)

    # Precompute total active-cell area for spatial averaging
    total_area = zero(T)
    for I in idxs
        if Grids.isactive(grid, Tuple(I)...)
            total_area += Grids.area(grid, Tuple(I)...)
        end
    end
    total_area > zero(T) || throw(ArgumentError("grid has no active cells (all masked out)"))

    # Sweep through scales. When the caller (typically `coarse_grain!`, which already builds one
    # plan per scale for its own `compute_Π!` loop) supplies `filter_plans`, reuse those instead of
    # rebuilding the same footprint a second time — this was the dominant redundant allocation in a
    # `coarse_grain!` sweep (confirmed by measurement: a 3-scale sweep dropped from ~60KB to within a
    # few hundred bytes of the sum of the three `compute_Π!` calls' own footprint builds).
    for s_idx in 1:Nscales
        ℓ = T(scales[s_idx])
        plan = filter_plans === nothing ?
            Filtering.plan_filter(grid, kernel, ℓ; mask_strategy=mask_strategy, backend=backend, method=method) :
            filter_plans[s_idx]

        # Filter velocity fields at this scale
        Filtering.filter_apply!(u_filt, u, plan)
        Filtering.filter_apply!(v_filt, v, plan)
        if w !== nothing
            Filtering.filter_apply!(w_filt, w, plan)
        end

        # Compute spatial average specific energy: E(ℓ) = 0.5 * ∫ |ū_ℓ|² dA / ∫ dA
        integrated_energy = zero(T)
        for I in idxs
            if Grids.isactive(grid, Tuple(I)...)
                vel2 = u_filt[I]^2 + v_filt[I]^2
                if w !== nothing
                    vel2 += w_filt[I]^2
                end
                integrated_energy += vel2 * Grids.area(grid, Tuple(I)...)
            end
        end

        spectrum[s_idx] = T(0.5) * integrated_energy / total_area
    end

    return spectrum
end

"""
    cumulative_energy(u, v, w, grid, kernel, scales; backend=AutoBackend(), mask_strategy=Deformable())

Cumulative coarse-grained kinetic energy `E(ℓ) = 0.5 ⟨|ū_ℓ|²⟩` at each filter scale
(Sadek & Aluie 2018, PRF, Eq. 15). This is the CUMULATIVE quantity; the filtering spectral DENSITY
(comparable to a Fourier energy spectrum) is its derivative w.r.t. filtering wavenumber — see
[`filtering_spectrum`](@ref). Allocates a fresh `spectrum` vector each call; for a repeated sweep
(e.g. inside `coarse_grain!`), call [`cumulative_energy!`](@ref) directly with a reused buffer.

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
    u::AbstractArray,
    v::AbstractArray,
    w::Union{Nothing, AbstractArray},
    grid::Union{Grids.StructuredGrid{G,T}, Grids.CurvilinearGrid{T,G}, Grids.UnstructuredGrid{T,G}},
    kernel::Kernels.AbstractFilterKernel,
    scales::AbstractVector;
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    method::Filtering.AbstractFilterMethod = Filtering.DirectSum(),
) where {T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}}
    spectrum = zeros(T, length(scales))
    return cumulative_energy!(spectrum, u, v, w, grid, kernel, scales; backend=backend, mask_strategy=mask_strategy, method=method)
end

"""
    filtering_spectrum(u, v, w, grid, kernel, scales; L=1, backend=AutoBackend(), mask_strategy=Deformable())
        -> (k_ℓ, Ẽ)

Filtering spectral DENSITY (Sadek & Aluie 2018, PRF, Eq. 14): the derivative of the cumulative
coarse-grained KE w.r.t. the filtering wavenumber `k_ℓ = L/ℓ`,

    Ẽ(k_ℓ) = d/dk_ℓ [ ½⟨|ū_ℓ|²⟩ ] = -(ℓ²/L) d/dℓ[ ½⟨|ū_ℓ|²⟩ ].

Unlike [`cumulative_energy`](@ref) (the cumulative quantity, Eq. 15), this is the spectral density
comparable to a Fourier energy spectrum. `L` is the region length: pass the domain size for the
Sadek–Aluie convention `k_ℓ = L/ℓ`; the default `L = 1` gives the FlowSieve convention `k_ℓ = 1/ℓ`.
`scales` need not be uniform. Returns the filtering wavenumbers `k_ℓ` and the density `Ẽ` per scale.

# References
- Sadek & Aluie (2018), *Phys. Rev. Fluids* 3, 124610.
"""
function filtering_spectrum(
    u::AbstractArray,
    v::AbstractArray,
    w::Union{Nothing, AbstractArray},
    grid::Union{Grids.StructuredGrid{G,T}, Grids.CurvilinearGrid{T,G}, Grids.UnstructuredGrid{T,G}},
    kernel::Kernels.AbstractFilterKernel,
    scales::AbstractVector;
    L::Real = one(T),
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    method::Filtering.AbstractFilterMethod = Filtering.DirectSum(),
) where {T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}}
    cum = cumulative_energy(u, v, w, grid, kernel, scales; backend=backend, mask_strategy=mask_strategy, method=method)
    kℓ = T(L) ./ T.(scales)
    return kℓ, spectral_density(cum, kℓ)
end

"""
    spectral_density!(g, C, k) -> g

In-place [`spectral_density`](@ref): writes the non-uniform finite-difference derivative of `C`
w.r.t. `k` into the caller-supplied `g` (central in the interior, one-sided at the ends). Fills
zeros for fewer than two points.
"""
function spectral_density!(g::AbstractVector{T}, C::AbstractVector{T}, k::AbstractVector) where {T<:AbstractFloat}
    n = length(C)
    length(g) == n || throw(DimensionMismatch("g has length $(length(g)), expected $n (= length(C))"))
    n < 2 && (fill!(g, zero(T)); return g)
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

"""
    spectral_density(C, k) -> dC/dk

Non-uniform finite-difference derivative of cumulative values `C` w.r.t. `k` (central in the
interior, one-sided at the ends). Returns zeros for fewer than two points.
"""
function spectral_density(C::AbstractVector{T}, k::AbstractVector) where {T<:AbstractFloat}
    return spectral_density!(zeros(T, length(C)), C, k)
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
) where {T<:AbstractFloat, G<:Geometry.CartesianGeometry{T}}
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

# Rotate a planetary-Cartesian symmetric 2-tensor (given as its xx/xy/xz/yy/yz/zz components) to the
# local (east, north) frame at (λ,φ) — the same rotation `_compute_Π_2d!`'s spherical branch uses for
# τ, factored out here so `tau_decomposition` can apply it to L/C/R without duplicating the algebra.
@inline function _rotate_sym_to_local_en(
    txx::T, txy::T, txz::T, tyy::T, tyz::T, tzz::T, λ::T, φ::T,
) where {T<:AbstractFloat}
    sinφ, cosφ = sin(φ), cos(φ)
    sinλ, cosλ = sin(λ), cos(λ)
    te_x = txx * (-sinλ) + txy * cosλ
    te_y = txy * (-sinλ) + tyy * cosλ
    te_z = txz * (-sinλ) + tyz * cosλ
    τ_ee = te_x * (-sinλ) + te_y * cosλ
    tn_x = txx * (-sinφ * cosλ) + txy * (-sinφ * sinλ) + txz * cosφ
    tn_y = txy * (-sinφ * cosλ) + tyy * (-sinφ * sinλ) + tyz * cosφ
    tn_z = txz * (-sinφ * cosλ) + tyz * (-sinφ * sinλ) + tzz * cosφ
    τ_nn = tn_x * (-sinφ * cosλ) + tn_y * (-sinφ * sinλ) + tn_z * cosφ
    τ_en = te_x * (-sinφ * cosλ) + te_y * (-sinφ * sinλ) + te_z * cosφ
    return τ_ee, τ_en, τ_nn
end

"""
    tau_decomposition(u, v, grid::StructuredGrid{<:SphericalGeometry}, kernel, scale; ...) -> (; L, C, R)

Spherical counterpart of the Cartesian method above: like [`compute_Π!`](@ref)'s spherical branch,
the Leonard/Cross/Reynolds moments are formed in PLANETARY-CARTESIAN coordinates (so filtering
commutes with the moment/residual operations, Aluie 2019), then each of `L`, `C`, `R`'s resulting 3×3
symmetric tensor is rotated back to the local (east, north) frame at every grid point. `L+C+R = τ`
still holds exactly (the rotation is linear). Returns the same `(; L, C, R)` shape as the Cartesian
method — local `(; xx, xy, yy)` (≡ east-east/east-north/north-north) components.
"""
function tau_decomposition(
    u::AbstractMatrix,
    v::AbstractMatrix,
    grid::Grids.StructuredGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
) where {T<:AbstractFloat, G<:Geometry.SphericalGeometry{T}}
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend)
    flt(f) = (o = zeros(T, size(f)); Filtering.filter_apply!(o, f, plan); o)
    Nlon, Nlat = Grids.size_tuple(grid)

    # Local (u,v) -> planetary Cartesian (ux,uy,uz) at every point.
    ux = zeros(T, Nlon, Nlat); uy = zeros(T, Nlon, Nlat); uz = zeros(T, Nlon, Nlat)
    for j in 1:Nlat, i in 1:Nlon
        if Grids.isactive(grid, i, j)
            λ, φ = Grids.coords(grid, i, j)
            pc = Geometry.to_planetary_cartesian(grid.geometry, u[i, j], v[i, j], λ, φ)
            ux[i, j], uy[i, j], uz[i, j] = pc[1], pc[2], pc[3]
        end
    end

    uxb = flt(ux); uyb = flt(uy); uzb = flt(uz)             # ū (planetary Cartesian)
    uxp = ux .- uxb; uyp = uy .- uyb; uzp = uz .- uzb        # residuals u'
    uxbb = flt(uxb); uybb = flt(uyb); uzbb = flt(uzb)        # double-filtered ū̄
    uxpb = flt(uxp); uypb = flt(uyp); uzpb = flt(uzp)        # filtered residuals ū'

    # Generalized second moment M(f,g) = (fg)‾ - f̄ ḡ for each planetary-Cartesian tensor component.
    M(f, g, fb, gb) = flt(f .* g) .- fb .* gb
    Lxx = M(uxb, uxb, uxbb, uxbb); Lxy = M(uxb, uyb, uxbb, uybb); Lxz = M(uxb, uzb, uxbb, uzbb)
    Lyy = M(uyb, uyb, uybb, uybb); Lyz = M(uyb, uzb, uybb, uzbb); Lzz = M(uzb, uzb, uzbb, uzbb)
    Cxx = T(2) .* M(uxb, uxp, uxbb, uxpb)
    Cxy = M(uxb, uyp, uxbb, uypb) .+ M(uxp, uyb, uxpb, uybb)
    Cxz = M(uxb, uzp, uxbb, uzpb) .+ M(uxp, uzb, uxpb, uzbb)
    Cyy = T(2) .* M(uyb, uyp, uybb, uypb)
    Cyz = M(uyb, uzp, uybb, uzpb) .+ M(uyp, uzb, uypb, uzbb)
    Czz = T(2) .* M(uzb, uzp, uzbb, uzpb)
    Rxx = M(uxp, uxp, uxpb, uxpb); Rxy = M(uxp, uyp, uxpb, uypb); Rxz = M(uxp, uzp, uxpb, uzpb)
    Ryy = M(uyp, uyp, uypb, uypb); Ryz = M(uyp, uzp, uypb, uzpb); Rzz = M(uzp, uzp, uzpb, uzpb)

    Lee = zeros(T, Nlon, Nlat); Len = zeros(T, Nlon, Nlat); Lnn = zeros(T, Nlon, Nlat)
    Cee = zeros(T, Nlon, Nlat); Cen = zeros(T, Nlon, Nlat); Cnn = zeros(T, Nlon, Nlat)
    Ree = zeros(T, Nlon, Nlat); Ren = zeros(T, Nlon, Nlat); Rnn = zeros(T, Nlon, Nlat)
    for j in 1:Nlat, i in 1:Nlon
        if Grids.isactive(grid, i, j)
            λ, φ = Grids.coords(grid, i, j)
            Lee[i,j], Len[i,j], Lnn[i,j] = _rotate_sym_to_local_en(Lxx[i,j], Lxy[i,j], Lxz[i,j], Lyy[i,j], Lyz[i,j], Lzz[i,j], λ, φ)
            Cee[i,j], Cen[i,j], Cnn[i,j] = _rotate_sym_to_local_en(Cxx[i,j], Cxy[i,j], Cxz[i,j], Cyy[i,j], Cyz[i,j], Czz[i,j], λ, φ)
            Ree[i,j], Ren[i,j], Rnn[i,j] = _rotate_sym_to_local_en(Rxx[i,j], Rxy[i,j], Rxz[i,j], Ryy[i,j], Ryz[i,j], Rzz[i,j], λ, φ)
        end
    end
    return (;
        L = (xx = Lee, xy = Len, yy = Lnn),
        C = (xx = Cee, xy = Cen, yy = Cnn),
        R = (xx = Ree, xy = Ren, yy = Rnn),
    )
end

# ---------------------------------------------------------------------------
# Rotational / divergent (Helmholtz) decomposition of the energy flux
# ---------------------------------------------------------------------------

"""
    compute_Π_decomposed(u, v, u_rot, v_rot, grid, kernel, scale; backend=AutoBackend(), mask_strategy=Deformable())
        -> (; total, rotational, cross, divergent)

Split the 2D Cartesian cross-scale KE flux Π = -S̄_ij τ_ij into rotational-rotational (Π_RR),
divergent-divergent (Π_DD), and cross/interaction (Π_X — the "stimulated cascade" channel of
Barkan, Srinivasan & McWilliams 2024, JPO) parts, by decomposing **both sides** of the bilinear
contraction, not just the stress.

The Helmholtz decomposition itself is NOT recomputed here — pass the rotational (solenoidal,
divergence-free) part `(u_rot, v_rot)` from a Helmholtz solver (e.g. `HelmholtzDecomposition.jl`); the
divergent (irrotational) part is taken as the complement `(u, v) - (u_rot, v_rot)`. Writing
`u = uʳ + uᵈ`:

  - The strain S̄ is LINEAR in velocity, so it splits with **no cross term**: `S̄ = S̄ʳ + S̄ᵈ`.
  - The stress τ is BILINEAR (quadratic in velocity), so it splits into three pieces:
    `τ(u,u) = τ(uʳ,uʳ) + τ(uᵈ,uᵈ) + [τ(uʳ,uᵈ) + τ(uᵈ,uʳ)] = τʳʳ + τᵈᵈ + τ_X`.

Substituting both splits into `Π = -S̄:τ = -(S̄ʳ+S̄ᵈ):(τʳʳ+τᵈᵈ+τ_X)` and expanding the six resulting
terms into three physically named channels:

    Π_RR = -S̄ʳ:τʳʳ                                        (pure rotational-to-rotational cascade)
    Π_DD = -S̄ᵈ:τᵈᵈ                                        (pure divergent-to-divergent cascade)
    Π_X  = -(S̄ʳ:τᵈᵈ + S̄ᵈ:τʳʳ + S̄ʳ:τ_X + S̄ᵈ:τ_X)          (all rotational/divergent interaction terms)

so the channels sum **exactly** to the total flux, Π = Π_RR + Π_X + Π_DD — each piece constructed
directly (not as a residual), yet the identity holds by the same bilinearity/linearity argument.
An earlier version of this function computed all three channels by contracting the split stress
against the *full*, undecomposed strain S̄ (a one-sided split); that's only correct in the special
case S̄ᵈ ≡ 0, and silently wrong whenever the divergent part itself has nonzero strain.

Returns a named tuple of flux maps (W m⁻³): `rotational` = Π_RR, `divergent` = Π_DD, `cross` = Π_X.
"""
function compute_Π_decomposed(
    u::AbstractMatrix,
    v::AbstractMatrix,
    u_rot::AbstractMatrix,
    v_rot::AbstractMatrix,
    grid::Grids.StructuredGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
) where {T<:AbstractFloat, G<:Geometry.CartesianGeometry{T}}
    gsz = Grids.size_tuple(grid)
    size(u) == gsz || throw(DimensionMismatch("u has size $(size(u)), grid expects $gsz"))
    size(v) == gsz || throw(DimensionMismatch("v has size $(size(v)), grid expects $gsz"))
    size(u_rot) == gsz || throw(DimensionMismatch("u_rot has size $(size(u_rot)), grid expects $gsz"))
    size(v_rot) == gsz || throw(DimensionMismatch("v_rot has size $(size(v_rot)), grid expects $gsz"))
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend)
    flt(f) = (o = zeros(T, size(f)); Filtering.filter_apply!(o, f, plan); o)

    # Divergent (irrotational) part is the complement of the supplied rotational part.
    u_div = u .- u_rot
    v_div = v .- v_rot

    # Self subfilter stress τ(a,a): (xx, xy, yy) of τ_ij = ⟨a_i a_j⟩ - ā_i ā_j.
    function self_stress(a, b)
        ā = flt(a); b̄ = flt(b)
        return (xx = flt(a .* a) .- ā .* ā,
                xy = flt(a .* b) .- ā .* b̄,
                yy = flt(b .* b) .- b̄ .* b̄)
    end
    # Combined cross stress τ(a,b) + τ(b,a) for two DIFFERENT vector fields a=(a1,a2), b=(b1,b2):
    # the diagonal (xx, yy) components are automatically symmetric under a↔b (a1*b1 = b1*a1
    # pointwise), so each just doubles; the off-diagonal (xy) component genuinely needs both terms.
    function cross_stress(a1, a2, b1, b2)
        ā1 = flt(a1); ā2 = flt(a2); b̄1 = flt(b1); b̄2 = flt(b2)
        return (xx = T(2) .* (flt(a1 .* b1) .- ā1 .* b̄1),
                xy = (flt(a1 .* b2) .- ā1 .* b̄2) .+ (flt(b1 .* a2) .- b̄1 .* ā2),
                yy = T(2) .* (flt(a2 .* b2) .- ā2 .* b̄2))
    end

    τ_RR = self_stress(u_rot, v_rot)
    τ_DD = self_stress(u_div, v_div)
    τ_X  = cross_stress(u_rot, v_rot, u_div, v_div)

    # Strain S̄_xx = ∂ū/∂x, S̄_yy = ∂v̄/∂y, S̄_xy = ½(∂ū/∂y + ∂v̄/∂x), from a velocity pair.
    function strain(a, b)
        ā = flt(a); b̄ = flt(b)
        Sxx = similar(ā); Derivatives.ddx!(Sxx, ā, grid)
        Syy = similar(ā); Derivatives.ddy!(Syy, b̄, grid)
        p = similar(ā); q = similar(ā)
        Derivatives.ddy!(p, ā, grid); Derivatives.ddx!(q, b̄, grid)
        return (xx = Sxx, xy = T(0.5) .* (p .+ q), yy = Syy)
    end
    S_R = strain(u_rot, v_rot)
    S_D = strain(u_div, v_div)

    mask = grid.mask
    contract(S, τ) = ifelse.(mask, -(S.xx .* τ.xx .+ T(2) .* S.xy .* τ.xy .+ S.yy .* τ.yy), zero(T))

    Πrr = contract(S_R, τ_RR)
    Πdd = contract(S_D, τ_DD)
    Πx  = contract(S_R, τ_DD) .+ contract(S_D, τ_RR) .+ contract(S_R, τ_X) .+ contract(S_D, τ_X)
    return (; total = Πrr .+ Πx .+ Πdd, rotational = Πrr, cross = Πx, divergent = Πdd)
end

"""
    compute_Π_decomposed(u, v, w, u_rot, v_rot, w_rot, grid::StructuredGrid{Cartesian,T,3}, kernel, scale; backend=AutoBackend(), mask_strategy=Deformable())
        -> (; total, rotational, cross, divergent)

True three-dimensional analog of the 2D [`compute_Π_decomposed`](@ref) above: the same both-sides
(strain AND stress) rotational/divergent split — see that method's docstring for the derivation —
generalized to all six independent strain/stress tensor components, contracted the same way the
true-3D [`compute_Π!`](@ref) method does (nine-term symmetric contraction).
"""
function compute_Π_decomposed(
    u::AbstractArray{<:Any,3},
    v::AbstractArray{<:Any,3},
    w::AbstractArray{<:Any,3},
    u_rot::AbstractArray{<:Any,3},
    v_rot::AbstractArray{<:Any,3},
    w_rot::AbstractArray{<:Any,3},
    grid::Grids.StructuredGrid{G,T,3},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
) where {T<:AbstractFloat, G<:Geometry.CartesianGeometry{T}}
    gsz = Grids.size_tuple(grid)
    size(u) == gsz || throw(DimensionMismatch("u has size $(size(u)), grid expects $gsz"))
    size(v) == gsz || throw(DimensionMismatch("v has size $(size(v)), grid expects $gsz"))
    size(w) == gsz || throw(DimensionMismatch("w has size $(size(w)), grid expects $gsz"))
    size(u_rot) == gsz || throw(DimensionMismatch("u_rot has size $(size(u_rot)), grid expects $gsz"))
    size(v_rot) == gsz || throw(DimensionMismatch("v_rot has size $(size(v_rot)), grid expects $gsz"))
    size(w_rot) == gsz || throw(DimensionMismatch("w_rot has size $(size(w_rot)), grid expects $gsz"))
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend)
    flt(f) = (o = zeros(T, size(f)); Filtering.filter_apply!(o, f, plan); o)

    u_div = u .- u_rot
    v_div = v .- v_rot
    w_div = w .- w_rot

    # Self subfilter stress τ(a,a) for a velocity triple a=(a1,a2,a3): all six components.
    function self_stress(a1, a2, a3)
        ā1 = flt(a1); ā2 = flt(a2); ā3 = flt(a3)
        return (xx = flt(a1 .* a1) .- ā1 .* ā1, xy = flt(a1 .* a2) .- ā1 .* ā2, xz = flt(a1 .* a3) .- ā1 .* ā3,
                yy = flt(a2 .* a2) .- ā2 .* ā2, yz = flt(a2 .* a3) .- ā2 .* ā3, zz = flt(a3 .* a3) .- ā3 .* ā3)
    end
    # Combined cross stress τ(a,b)+τ(b,a) for two DIFFERENT velocity triples a=(a1,a2,a3), b=(b1,b2,b3).
    function cross_stress(a1, a2, a3, b1, b2, b3)
        ā1 = flt(a1); ā2 = flt(a2); ā3 = flt(a3); b̄1 = flt(b1); b̄2 = flt(b2); b̄3 = flt(b3)
        return (
            xx = T(2) .* (flt(a1 .* b1) .- ā1 .* b̄1),
            yy = T(2) .* (flt(a2 .* b2) .- ā2 .* b̄2),
            zz = T(2) .* (flt(a3 .* b3) .- ā3 .* b̄3),
            xy = (flt(a1 .* b2) .- ā1 .* b̄2) .+ (flt(b1 .* a2) .- b̄1 .* ā2),
            xz = (flt(a1 .* b3) .- ā1 .* b̄3) .+ (flt(b1 .* a3) .- b̄1 .* ā3),
            yz = (flt(a2 .* b3) .- ā2 .* b̄3) .+ (flt(b2 .* a3) .- b̄2 .* ā3),
        )
    end

    τ_RR = self_stress(u_rot, v_rot, w_rot)
    τ_DD = self_stress(u_div, v_div, w_div)
    τ_X  = cross_stress(u_rot, v_rot, w_rot, u_div, v_div, w_div)

    # Strain from a velocity triple (a,b,c): three diagonals + three off-diagonals.
    function strain(a, b, c)
        ā = flt(a); b̄ = flt(b); c̄ = flt(c)
        Sxx = similar(ā); Derivatives.ddx!(Sxx, ā, grid)
        Syy = similar(ā); Derivatives.ddy!(Syy, b̄, grid)
        Szz = similar(ā); Derivatives.ddz!(Szz, c̄, grid)
        p = similar(ā); q = similar(ā)
        Derivatives.ddy!(p, ā, grid); Derivatives.ddx!(q, b̄, grid)
        Sxy = T(0.5) .* (p .+ q)
        Derivatives.ddz!(p, ā, grid); Derivatives.ddx!(q, c̄, grid)
        Sxz = T(0.5) .* (p .+ q)
        Derivatives.ddz!(p, b̄, grid); Derivatives.ddy!(q, c̄, grid)
        Syz = T(0.5) .* (p .+ q)
        return (xx = Sxx, xy = Sxy, xz = Sxz, yy = Syy, yz = Syz, zz = Szz)
    end
    S_R = strain(u_rot, v_rot, w_rot)
    S_D = strain(u_div, v_div, w_div)

    mask = grid.mask
    contract(S, τ) = ifelse.(
        mask,
        -(S.xx .* τ.xx .+ S.yy .* τ.yy .+ S.zz .* τ.zz .+
          T(2) .* (S.xy .* τ.xy .+ S.xz .* τ.xz .+ S.yz .* τ.yz)),
        zero(T),
    )

    Πrr = contract(S_R, τ_RR)
    Πdd = contract(S_D, τ_DD)
    Πx  = contract(S_R, τ_DD) .+ contract(S_D, τ_RR) .+ contract(S_R, τ_X) .+ contract(S_D, τ_X)
    return (; total = Πrr .+ Πx .+ Πdd, rotational = Πrr, cross = Πx, divergent = Πdd)
end

# ---------------------------------------------------------------------------
# Cross-scale tracer-variance flux (scalar analog of Π; buoyancy ⇒ APE transfer)
# ---------------------------------------------------------------------------

"""
    tracer_variance_flux(u, v, θ, grid, kernel, scale; backend=AutoBackend(), mask_strategy=Deformable())
        -> Πθ

Cross-scale flux of the tracer variance ½⟨θ'²⟩ at filter scale ℓ (the scalar analog of the kinetic
energy flux Π; Aluie & Eyink):

    Πθ(x) = -∂_j θ̄ · τ_j(u, θ),   τ_j = ⟨u_j θ⟩ - ū_j θ̄  (the subfilter tracer flux),

with the same sign convention as [`compute_Π!`](@ref): `Πθ > 0` is a forward cascade of tracer
variance toward small scales, `Πθ < 0` an inverse cascade.

Taking `θ` to be the **buoyancy** `b = -g ρ'/ρ₀` makes this the cross-scale transfer of buoyancy
variance (the available-potential-energy-related transfer). Unlike the full Lees & Aluie (2019)
baropycnal work — which additionally requires the pressure field — this needs only `(u, v, θ)`.

Cartesian 2D (uses the physical-gradient derivatives `ddx!`/`ddy!`); spherical/3D deferred.
"""
function tracer_variance_flux(
    u::AbstractMatrix,
    v::AbstractMatrix,
    θ::AbstractMatrix,
    grid::Grids.StructuredGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
) where {T<:AbstractFloat, G<:Geometry.CartesianGeometry{T}}
    gsz = Grids.size_tuple(grid)
    size(u) == gsz || throw(DimensionMismatch("u has size $(size(u)), grid expects $gsz"))
    size(v) == gsz || throw(DimensionMismatch("v has size $(size(v)), grid expects $gsz"))
    size(θ) == gsz || throw(DimensionMismatch("θ has size $(size(θ)), grid expects $gsz"))
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend)
    flt(f) = (o = zeros(T, size(f)); Filtering.filter_apply!(o, f, plan); o)

    ū = flt(u); v̄ = flt(v); θ̄ = flt(θ)
    # Subfilter tracer flux τ_j = ⟨u_j θ⟩ - ū_j θ̄.
    τx = flt(u .* θ) .- ū .* θ̄
    τy = flt(v .* θ) .- v̄ .* θ̄

    # Resolved tracer gradient ∂_j θ̄.
    gx = similar(θ̄); Derivatives.ddx!(gx, θ̄, grid)
    gy = similar(θ̄); Derivatives.ddy!(gy, θ̄, grid)

    mask = grid.mask
    return ifelse.(mask, .-(τx .* gx .+ τy .* gy), zero(T))
end

"""
    tracer_variance_flux(u, v, w, θ, grid::StructuredGrid{Cartesian,T,3}, kernel, scale; backend=AutoBackend(), mask_strategy=Deformable())
        -> Πθ

True three-dimensional analog of the 2D [`tracer_variance_flux`](@ref) above: the subfilter tracer
flux gets a genuine vertical component `τ_z = ⟨wθ⟩ - w̄θ̄`, contracted against the resolved 3D
tracer gradient `∂_j θ̄` (all three components, including the real vertical derivative `∂θ̄/∂z`).
"""
function tracer_variance_flux(
    u::AbstractArray{<:Any,3},
    v::AbstractArray{<:Any,3},
    w::AbstractArray{<:Any,3},
    θ::AbstractArray{<:Any,3},
    grid::Grids.StructuredGrid{G,T,3},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
) where {T<:AbstractFloat, G<:Geometry.CartesianGeometry{T}}
    gsz = Grids.size_tuple(grid)
    size(u) == gsz || throw(DimensionMismatch("u has size $(size(u)), grid expects $gsz"))
    size(v) == gsz || throw(DimensionMismatch("v has size $(size(v)), grid expects $gsz"))
    size(w) == gsz || throw(DimensionMismatch("w has size $(size(w)), grid expects $gsz"))
    size(θ) == gsz || throw(DimensionMismatch("θ has size $(size(θ)), grid expects $gsz"))
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend)
    flt(f) = (o = zeros(T, size(f)); Filtering.filter_apply!(o, f, plan); o)

    ū = flt(u); v̄ = flt(v); w̄ = flt(w); θ̄ = flt(θ)
    # Subfilter tracer flux τ_j = ⟨u_j θ⟩ - ū_j θ̄, now with a genuine vertical component.
    τx = flt(u .* θ) .- ū .* θ̄
    τy = flt(v .* θ) .- v̄ .* θ̄
    τz = flt(w .* θ) .- w̄ .* θ̄

    # Resolved tracer gradient ∂_j θ̄, including the real vertical derivative.
    gx = similar(θ̄); Derivatives.ddx!(gx, θ̄, grid)
    gy = similar(θ̄); Derivatives.ddy!(gy, θ̄, grid)
    gz = similar(θ̄); Derivatives.ddz!(gz, θ̄, grid)

    mask = grid.mask
    return ifelse.(mask, .-(τx .* gx .+ τy .* gy .+ τz .* gz), zero(T))
end

end # module
