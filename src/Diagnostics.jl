module Diagnostics

using FlowGeometries: FlowGeometries
using ..Kernels: Kernels
using ..Filtering: Filtering
using ..Derivatives: Derivatives
using ComputationalBackends: ComputationalBackends

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

    # Three scratch arrays, so `filter_apply_batch!` can filter all 6 quadratic velocity products in
    # one pass. The spherical branch needs 6 simultaneous pre-filter buffers and only 3 are idle at
    # that point (`u_filt`/`v_filt`/`w_filt`, before the planetary→local transform overwrites them).
    scratch::A
    scratch2::A
    scratch3::A
end

# Workspace constructor based on grid structure and float type. `A` is inferred from what
# `zeros(T, sz...)` actually produces (Vector for a 1D grid, Matrix for 2D, Array{T,3} for 3D) —
# NOT hardcoded, since a 1D/3D grid's `sz` is a 1- or 3-tuple, not always 2D.
function ΠWorkspace(grid::FlowGeometries.Grids.AbstractGrid{G,T}) where {G, T<:AbstractFloat}
    sz = FlowGeometries.Grids.size_tuple(grid)

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
    scratch2 = zeros(T, sz...)
    scratch3 = zeros(T, sz...)

    return ΠWorkspace(
        u_filt, v_filt, w_filt,
        ux, uy, uz, ux_filt, uy_filt, uz_filt,
        uu_filt, uv_filt, uw_filt, vv_filt, vw_filt, ww_filt,
        S_xx, S_xy, S_xz, S_yy, S_yz, S_zz,
        τ_xx, τ_xy, τ_xz, τ_yy, τ_yz, τ_zz,
        scratch, scratch2, scratch3
    )
end

# ---------------------------------------------------------------------------
# Energy Flux (Π) Calculation
# ---------------------------------------------------------------------------

# Boundary-only (once per top-level call, not per grid point): a mismatched v/w would otherwise be
# silently truncated/ignored by CartesianIndices(u), not caught at all. Same idiom as the level-count
# check in compute_Π_profile! below.
@inline function _validate_field_sizes(grid, Π::AbstractArray, u::AbstractArray, v = nothing, w = nothing)
    gsz = FlowGeometries.Grids.size_tuple(grid)
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
- `mask_strategy::AbstractMaskStrategy=Deformable()`: Masking strategy (`ZeroFill()` or `Deformable()`)

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
method independently at each z level of a multi-level dataset and comparing/stacking the resulting
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
    grid::FlowGeometries.Grids.StructuredGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    workspace::Union{Nothing, ΠWorkspace} = nothing,
    filter_plan::Union{Nothing, Filtering.AbstractFilterPlan} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    deriv_plan::Union{Nothing, Derivatives.StencilPlan} = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    _validate_field_sizes(grid, Π, u, v, w)
    ws = workspace === nothing ? ΠWorkspace(grid) : workspace
    # One plan for this scale, shared by all ~9 filterings below. A caller repeating this at a fixed
    # scale can pass a prebuilt `filter_plan` to share it across calls as well.
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend) : filter_plan
    # The stencil weights depend only on the grid, so one table serves every derivative here and every
    # later call at any scale.
    dplan = deriv_plan === nothing ? Derivatives.StencilPlan(grid) : deriv_plan
    return _compute_Π!(Π, u, v, w, grid, ws, plan, dplan)
end

"""
    compute_Π_profile!(Π, u, v, w, grid, kernel, scale; workspace=nothing, backend=AutoBackend(), mask_strategy=Deformable())

Vertical-profile energy flux: given 3D `(x, y, z)` velocity arrays, runs the 2D/2.5D
[`compute_Π!`](@ref) INDEPENDENTLY at each z level — the literature-standard way to obtain
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
    grid::FlowGeometries.Grids.StructuredGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    workspace::Union{Nothing, ΠWorkspace} = nothing,
    filter_plan::Union{Nothing, Filtering.AbstractFilterPlan} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    deriv_plan::Union{Nothing, Derivatives.StencilPlan} = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    # Every level shares the grid, so one stencil table serves the whole loop.
    dplan = deriv_plan === nothing ? Derivatives.StencilPlan(grid) : deriv_plan
    Nlevels = size(u, 3)
    size(Π, 3) == Nlevels || throw(DimensionMismatch(
        "Π has $(size(Π, 3)) z levels, u has $Nlevels",
    ))
    ws = workspace === nothing ? ΠWorkspace(grid) : workspace
    # Every level shares the grid, kernel and scale, so one plan serves the whole loop. A caller
    # sweeping scales can pass one in to share it across iterations too.
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend) : filter_plan
    for k in 1:Nlevels
        wk = w === nothing ? nothing : view(w, :, :, k)
        compute_Π!(
            view(Π, :, :, k), view(u, :, :, k), view(v, :, :, k), wk, grid, kernel, scale;
            workspace = ws, filter_plan = plan, backend = backend, mask_strategy = mask_strategy,
            deriv_plan = dplan,
        )
    end
    return Π
end

# ---------------------------------------------------------------------------
# Shared 2D driver for the per-point tensor physics (rotation / SFS stress / strain contraction),
# called by BOTH the StructuredGrid and CurvilinearGrid `compute_Π!` methods. Both grids reach their
# geometry only through `FlowGeometries.Grids.coords`/`isactive` and the `ddx!`/`ddy!` operators, so this one kernel
# serves both — no duplicated tensor math. `deriv_plan` is a `Derivatives.StencilPlan` for a
# `StructuredGrid` and a `Connectivity.gradient_plan` for a curvilinear grid or a node set; either way
# it is geometry only, so one holds for every scale.
# ---------------------------------------------------------------------------

# Both tangent components of one field — which is what the strain tensor needs of every field it
# touches. A separable grid differences each direction independently; a curvilinear grid or a node set
# has no direction to difference along and fits both components at once from the same neighbour sweep,
# so asking for them together is one traversal there rather than two.
@inline function _grad2!(g1, g2, f, grid::FlowGeometries.Grids.StructuredGrid, ::Nothing)
    Derivatives.ddx!(g1, f, grid)
    Derivatives.ddy!(g2, f, grid)
    return nothing
end
@inline function _grad2!(g1, g2, f, grid::FlowGeometries.Grids.StructuredGrid, plan::Derivatives.StencilPlan)
    Derivatives.ddx!(g1, f, grid, plan)
    Derivatives.ddy!(g2, f, grid, plan)
    return nothing
end
@inline function _grad2!(g1, g2, f, _grid, plan::FlowGeometries.Discretization.GradientPlan)
    FlowGeometries.Discretization.gradient!(g1, g2, f, plan)
    return nothing
end

# Rotate a planetary-Cartesian symmetric stress to the local (east, north, radial) frame at (λ,φ),
# flattened to the component order this file's contractions read. A caller with no radial velocity
# passes zero for `txz`/`tyz`/`tzz` and discards `τer`/`τnr`/`τrr`.
@inline function _rotate_stress_to_local_enr(
    geo::FlowGeometries.Geometry.AbstractSphericalGeometry,
    txx::T, txy::T, txz::T, tyy::T, tyz::T, tzz::T, λ::T, φ::T,
) where {T<:AbstractFloat}
    τ = FlowGeometries.Geometry.tensor_to_local(geo, txx, tyy, tzz, txy, txz, tyz, λ, φ)
    return τ.λλ, τ.λφ, τ.λr, τ.φφ, τ.φr, τ.rr
end

# Symmetric SFS tensor contraction S̄_ij τ_ij — the scalar sum shared by every `compute_Π!` driver's
# final step (2D contraction, or the full six-term 3D contraction when a vertical/radial component
# exists). Factored out so `_compute_Π!` and `_compute_Π!` share the identical arithmetic.
@inline _sfs_contraction(Sxx::T, Sxy::T, Syy::T, τxx::T, τxy::T, τyy::T) where {T<:AbstractFloat} =
    Sxx * τxx + T(2) * Sxy * τxy + Syy * τyy

@inline _sfs_contraction(
    Sxx::T, Sxy::T, Sxz::T, Syy::T, Syz::T, Szz::T, τxx::T, τxy::T, τxz::T, τyy::T, τyz::T, τzz::T,
) where {T<:AbstractFloat} =
    Sxx * τxx + T(2) * Sxy * τxy + Syy * τyy + T(2) * Sxz * τxz + T(2) * Syz * τyz + Szz * τzz

# One driver for every point-indexed grid: the broadcasts are shape-agnostic and the explicit loops
# run over `CartesianIndices`, so a node-indexed `UnstructuredGrid` and an `(i,j)` 2D grid take the
# same code. The true-3D methods below are separate because their physics differs — real radial
# derivatives and curvature terms this 2.5D path drops by construction.
function _compute_Π!(
    Π::AbstractArray{T},
    u::AbstractArray,
    v::AbstractArray,
    w::Union{Nothing, AbstractArray},
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    ws::ΠWorkspace,
    plan::Filtering.AbstractFilterPlan,
    deriv_plan,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    has_w = w !== nothing

    if G <: FlowGeometries.Geometry.CartesianGeometry{T}
        # -------------------------------------------------------------------
        # Cartesian Case
        # -------------------------------------------------------------------
        # Filter velocity components — batched: one neighbour-list/weight derivation per point,
        # applied to all primitives at once (see `Filtering.filter_apply_batch!`), not once per field.
        if has_w
            Filtering.filter_apply_batch!((ws.u_filt, ws.v_filt, ws.w_filt), (u, v, w), plan)
        else
            Filtering.filter_apply_batch!((ws.u_filt, ws.v_filt), (u, v), plan)
        end

        # Filter products: u², uv, vv, etc. — also batched, in one pass. `ux`/`uy`/`uz`/`ux_filt`/
        # `uy_filt`/`uz_filt` are spherical-only fields, genuinely idle in this Cartesian branch, so
        # they're reused here purely as pre-filter product scratch (no new workspace fields needed).
        @. ws.ux = u * u
        @. ws.uy = u * v
        @. ws.uz = v * v
        if has_w
            @. ws.ux_filt = u * w
            @. ws.uy_filt = v * w
            @. ws.uz_filt = w * w
            Filtering.filter_apply_batch!(
                (ws.uu_filt, ws.uv_filt, ws.vv_filt, ws.uw_filt, ws.vw_filt, ws.ww_filt),
                (ws.ux, ws.uy, ws.uz, ws.ux_filt, ws.uy_filt, ws.uz_filt),
                plan,
            )
        else
            Filtering.filter_apply_batch!((ws.uu_filt, ws.uv_filt, ws.vv_filt), (ws.ux, ws.uy, ws.uz), plan)
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

        # Strain rate: S̄_ij = 0.5 * (∂ū_i/∂x_j + ∂ū_j/∂x_i). One gradient per velocity component.
        _grad2!(ws.S_xx, ws.S_xy, ws.u_filt, grid, deriv_plan)     # ∂ū/∂x, ∂ū/∂y
        _grad2!(ws.scratch, ws.S_yy, ws.v_filt, grid, deriv_plan)  # ∂v̄/∂x, ∂v̄/∂y
        @. ws.S_xy = T(0.5) * (ws.S_xy + ws.scratch)

        if has_w
            # S_xz = 0.5 * (∂ū/∂z + ∂w̄/∂x), S_yz = 0.5 * (∂v̄/∂z + ∂w̄/∂y); ∂/∂z is zero for a
            # level stack, leaving the horizontal gradient of w̄.
            _grad2!(ws.S_xz, ws.S_yz, ws.w_filt, grid, deriv_plan)
            @. ws.S_xz = T(0.5) * ws.S_xz
            @. ws.S_yz = T(0.5) * ws.S_yz

            # S_zz = ∂w̄/∂z = 0 (for standard 2.5D datasets)
            fill!(ws.S_zz, zero(T))
        end

    else
        # -------------------------------------------------------------------
        # Spherical Case (Aluie 2019 commutativity formulation)
        # -------------------------------------------------------------------
        # Transform local coordinates (u_east, v_north) to global Cartesian (u_X, u_Y, u_Z)
        for I in CartesianIndices(Π)
            let i = Tuple(I)
                if FlowGeometries.Grids.isactive(grid, i...)
                    λ, φ = FlowGeometries.Grids.coords(grid, i...)
                    u_val = u[I]
                    v_val = v[I]
                    w_val = has_w ? w[I] : zero(T)

                    p_vel = FlowGeometries.Geometry.vector_to_cartesian(FlowGeometries.Grids.grid_geometry(grid), u_val, v_val, w_val, λ, φ)
                    ws.ux[I] = p_vel[1]
                    ws.uy[I] = p_vel[2]
                    ws.uz[I] = p_vel[3]
                else
                    ws.ux[I] = zero(T)
                    ws.uy[I] = zero(T)
                    ws.uz[I] = zero(T)
                end
            end
        end

        # Filter planetary Cartesian components — batched (one derivation per point, not one per field).
        Filtering.filter_apply_batch!((ws.ux_filt, ws.uy_filt, ws.uz_filt), (ws.ux, ws.uy, ws.uz), plan)

        # Filter planetary products: X-X, X-Y, X-Z, Y-Y, Y-Z, Z-Z — also batched, in one pass. The 6
        # pre-filter product buffers reuse `u_filt`/`v_filt`/`w_filt` (genuinely idle here — the
        # "transform back to local coordinates" step below overwrites them with real values right
        # after, so nothing reads their stale product-scratch content) plus `scratch`/`scratch2`/
        # `scratch3` (the extra scratch fields added specifically so this fits in one batch).
        @. ws.u_filt = ws.ux * ws.ux
        @. ws.v_filt = ws.ux * ws.uy
        @. ws.w_filt = ws.ux * ws.uz
        @. ws.scratch = ws.uy * ws.uy
        @. ws.scratch2 = ws.uy * ws.uz
        @. ws.scratch3 = ws.uz * ws.uz
        Filtering.filter_apply_batch!(
            (ws.uu_filt, ws.uv_filt, ws.uw_filt, ws.vv_filt, ws.vw_filt, ws.ww_filt),
            (ws.u_filt, ws.v_filt, ws.w_filt, ws.scratch, ws.scratch2, ws.scratch3),
            plan,
        )

        # Transform filtered planetary velocities back to local coordinates (u_filt, v_filt, w_filt)
        for I in CartesianIndices(Π)
            let i = Tuple(I)
                if FlowGeometries.Grids.isactive(grid, i...)
                    λ, φ = FlowGeometries.Grids.coords(grid, i...)
                    l_vel = FlowGeometries.Geometry.vector_from_cartesian(FlowGeometries.Grids.grid_geometry(grid), ws.ux_filt[I], ws.uy_filt[I], ws.uz_filt[I], λ, φ)
                    ws.u_filt[I] = l_vel[1]
                    ws.v_filt[I] = l_vel[2]
                    ws.w_filt[I] = l_vel[3]
                else
                    ws.u_filt[I] = zero(T)
                    ws.v_filt[I] = zero(T)
                    ws.w_filt[I] = zero(T)
                end
            end
        end

        # Transform planetary filtered products to local stresses at each grid point, via the shared
        # `_rotate_stress_to_local_enr` scalar kernel (τ_local = R' * ( [u_i u_j]̄ - ū_i ū_j ) * R for
        # the orthogonal local rotation R = [e_east, e_north, e_radial]) — see that function for the
        # rotation algebra itself, kept in one place so the 1D `UnstructuredGrid` driver below shares it.
        # `txz`/`tyz`/`tzz` are not gated on `has_w`: the planetary Cartesian Z component is nonzero
        # even for a purely horizontal velocity, since that rotates into Z through cosφ/sinφ. Those
        # cross terms feed the rotated τee/τen/τnn. Only τer/τnr/τrr are genuinely radial.
        geo = FlowGeometries.Grids.grid_geometry(grid)
        for I in CartesianIndices(Π)
            let i = Tuple(I)
                if FlowGeometries.Grids.isactive(grid, i...)
                    λ, φ = FlowGeometries.Grids.coords(grid, i...)
                    txx = ws.uu_filt[I] - ws.ux_filt[I] * ws.ux_filt[I]
                    txy = ws.uv_filt[I] - ws.ux_filt[I] * ws.uy_filt[I]
                    tyy = ws.vv_filt[I] - ws.uy_filt[I] * ws.uy_filt[I]
                    txz = ws.uw_filt[I] - ws.ux_filt[I] * ws.uz_filt[I]
                    tyz = ws.vw_filt[I] - ws.uy_filt[I] * ws.uz_filt[I]
                    tzz = ws.ww_filt[I] - ws.uz_filt[I] * ws.uz_filt[I]
                    τee, τen, τer, τnn, τnr, τrr = _rotate_stress_to_local_enr(geo, txx, txy, txz, tyy, tyz, tzz, λ, φ)
                    ws.τ_xx[I] = τee
                    ws.τ_yy[I] = τnn
                    ws.τ_xy[I] = τen
                    if has_w
                        ws.τ_xz[I] = τer
                        ws.τ_yz[I] = τnr
                        ws.τ_zz[I] = τrr
                    end
                else
                    ws.τ_xx[I] = zero(T)
                    ws.τ_yy[I] = zero(T)
                    ws.τ_xy[I] = zero(T)
                    if has_w
                        ws.τ_xz[I] = zero(T)
                        ws.τ_yz[I] = zero(T)
                        ws.τ_zz[I] = zero(T)
                    end
                end
            end
        end

        # Compute Spherical Strain Rates (with geometry curvature correction terms)
        # S_ee = 1/(R cosφ) ∂ū_e/∂λ − v̄_n sinφ/(R cosφ);  S_nn = 1/R ∂v̄_n/∂φ
        # S_en = 0.5 ( 1/(R cosφ) ∂v̄_n/∂λ + 1/R ∂ū_e/∂φ + ū_e sinφ/(R cosφ) )
        _grad2!(ws.S_xx, ws.S_xy, ws.u_filt, grid, deriv_plan)
        _grad2!(ws.scratch, ws.S_yy, ws.v_filt, grid, deriv_plan)

        R = FlowGeometries.Geometry.radius(geo)
        for I in CartesianIndices(Π)
            let i = Tuple(I)
                if FlowGeometries.Grids.isactive(grid, i...)
                    _, φ = FlowGeometries.Grids.coords(grid, i...)
                    sinφ, cosφ = sincos(φ)
                    tan_fact = abs(cosφ) > T(1e-12) ? sinφ / (R * cosφ) : zero(T)
                    ws.S_xx[I] -= ws.v_filt[I] * tan_fact                    # S_ee correction
                    ws.S_xy[I] = T(0.5) * (ws.S_xy[I] + ws.scratch[I] + ws.u_filt[I] * tan_fact)  # S_en
                end
            end
        end

        if has_w
            # S_er = 0.5 (∂ū_e/∂r + 1/(R cosφ) ∂w̄/∂λ) and S_nr = 0.5 (∂v̄_n/∂r + 1/R ∂w̄/∂φ); with
            # vertically flat layers ∂/∂r drops and each is half the horizontal gradient of w̄.
            _grad2!(ws.S_xz, ws.S_yz, ws.w_filt, grid, deriv_plan)
            @. ws.S_xz = T(0.5) * ws.S_xz
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
    mask = FlowGeometries.Grids.mask(grid)
    if has_w
        @. Π = ifelse(mask, -_sfs_contraction(
            ws.S_xx, ws.S_xy, ws.S_xz, ws.S_yy, ws.S_yz, ws.S_zz,
            ws.τ_xx, ws.τ_xy, ws.τ_xz, ws.τ_yy, ws.τ_yz, ws.τ_zz,
        ), zero(T))
    else
        @. Π = ifelse(mask, -_sfs_contraction(
            ws.S_xx, ws.S_xy, ws.S_yy, ws.τ_xx, ws.τ_xy, ws.τ_yy,
        ), zero(T))
    end

    return Π
end


"""
    compute_Π!(Π, u, v, w, grid::UnstructuredGrid, kernel, scale; workspace=nothing, deriv_plan=nothing, backend=AutoBackend(), mask_strategy=Deformable(), method=Spectral())

Cross-scale kinetic energy flux Π = -S̄_ij τ_ij on a `FlowGeometries.Grids.UnstructuredGrid` (scattered
points, node-indexed) — the same physics as the 2D methods (planetary-Cartesian rotation for
spherical geometry), via `_compute_Π!`. The resolved strain uses the node-indexed WLSQ
gradient (`Connectivity.gradient_plan` + `Discretization.gradient!`). `method` defaults to
`Spectral()` here, unlike the other grid types' `RealSpace()` default: the transform is exact for a
band-limited field and its per-apply cost does not grow with the filter scale. `RealSpace()` applies
the kernel as written, with compact support; a transform's support is global.
"""
function compute_Π!(
    Π::AbstractVector{T},
    u::AbstractVector,
    v::AbstractVector,
    w::Union{Nothing, AbstractVector},
    grid::FlowGeometries.Grids.UnstructuredGrid{T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    workspace::Union{Nothing, ΠWorkspace} = nothing,
    deriv_plan::Union{Nothing, FlowGeometries.Discretization.GradientPlan} = nothing,
    filter_plan::Union{Nothing, Filtering.AbstractFilterPlan} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    method::Filtering.AbstractFilterMethod = Filtering.Spectral(),
) where {T<:AbstractFloat}
    _validate_field_sizes(grid, Π, u, v, w)
    ws = workspace === nothing ? ΠWorkspace(grid) : workspace
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend, method=method) : filter_plan
    dplan = deriv_plan === nothing ? FlowGeometries.Connectivity.gradient_plan(grid) : deriv_plan
    return _compute_Π!(Π, u, v, w, grid, ws, plan, dplan)
end

"""
    compute_Π!(Π, u, v, w, grid::CurvilinearGrid, kernel, scale; workspace=nothing, deriv_plan=nothing, backend=AutoBackend(), mask_strategy=Deformable())

Cross-scale kinetic energy flux Π = -S̄_ij τ_ij on a `FlowGeometries.Grids.CurvilinearGrid`. Identical
physics to the `StructuredGrid` 2D method — it shares the same `_compute_Π!` tensor kernel
— but the resolved strain uses the least-squares tangent-plane gradient
(`Discretization.gradient!` over a `Connectivity.gradient_plan`, both components from one neighbour
sweep) and real-space filtering uses the scattered per-point footprint. Pass a prebuilt
`deriv_plan = FG.Connectivity.gradient_plan(grid)` (and a reusable `workspace`) to avoid rebuilding them per call
across a scale sweep.
"""
function compute_Π!(
    Π::AbstractMatrix{T},
    u::AbstractMatrix,
    v::AbstractMatrix,
    w::Union{Nothing, AbstractMatrix},
    grid::FlowGeometries.Grids.CurvilinearGrid{T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    workspace::Union{Nothing, ΠWorkspace} = nothing,
    deriv_plan::Union{Nothing, FlowGeometries.Discretization.GradientPlan} = nothing,
    filter_plan::Union{Nothing, Filtering.AbstractFilterPlan} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
) where {T<:AbstractFloat}
    _validate_field_sizes(grid, Π, u, v, w)
    ws = workspace === nothing ? ΠWorkspace(grid) : workspace
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend) : filter_plan
    dplan = deriv_plan === nothing ? FlowGeometries.Connectivity.gradient_plan(grid) : deriv_plan
    return _compute_Π!(Π, u, v, w, grid, ws, plan, dplan)
end

"""
    compute_Π!(Π::AbstractArray{T,3}, u, v, w, grid::StructuredGrid{Cartesian,T,3}, kernel, scale; mask_strategy=Deformable(), backend=AutoBackend())

Full **three-dimensional** Cartesian cross-scale energy flux Π = -S̄_ij τ_ij with all nine strain
components (the diagonal `S_zz = ∂w̄/∂z` and the off-diagonals `S_xz, S_yz` carry genuine vertical
derivatives, unlike the 2.5D layer-by-layer path). The 3D grid carries a 3D mask, so masked cells are
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
    grid::FlowGeometries.Grids.StructuredGrid{G,T,3},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    workspace::Union{Nothing, ΠWorkspace} = nothing,
    filter_plan::Union{Nothing, Filtering.AbstractFilterPlan} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    deriv_plan::Union{Nothing, Derivatives.StencilPlan} = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    _validate_field_sizes(grid, Π, u, v, w)
    ws = workspace === nothing ? ΠWorkspace(grid) : workspace
    dplan = deriv_plan === nothing ? Derivatives.StencilPlan(grid) : deriv_plan
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend) : filter_plan

    # Filtered velocities and the six independent filtered quadratic products — batched (one
    # neighbour-list/weight derivation per point, applied to the whole group at once). `ux`/`uy`/`uz`/
    # `ux_filt`/`uy_filt`/`uz_filt` are spherical-only fields, genuinely idle here, reused purely as
    # pre-filter product scratch (no new workspace fields needed) — same pattern as `_compute_Π!`.
    Filtering.filter_apply_batch!((ws.u_filt, ws.v_filt, ws.w_filt), (u, v, w), plan)
    @. ws.ux = u * u
    @. ws.uy = u * v
    @. ws.uz = u * w
    @. ws.ux_filt = v * v
    @. ws.uy_filt = v * w
    @. ws.uz_filt = w * w
    Filtering.filter_apply_batch!(
        (ws.uu_filt, ws.uv_filt, ws.uw_filt, ws.vv_filt, ws.vw_filt, ws.ww_filt),
        (ws.ux, ws.uy, ws.uz, ws.ux_filt, ws.uy_filt, ws.uz_filt),
        plan,
    )

    # Subfilter stress τ_ij = ⟨u_i u_j⟩ - ū_i ū_j (symmetric, six components).
    @. ws.τ_xx = ws.uu_filt - ws.u_filt * ws.u_filt
    @. ws.τ_xy = ws.uv_filt - ws.u_filt * ws.v_filt
    @. ws.τ_xz = ws.uw_filt - ws.u_filt * ws.w_filt
    @. ws.τ_yy = ws.vv_filt - ws.v_filt * ws.v_filt
    @. ws.τ_yz = ws.vw_filt - ws.v_filt * ws.w_filt
    @. ws.τ_zz = ws.ww_filt - ws.w_filt * ws.w_filt

    # Strain S̄_ij = ½(∂ū_i/∂x_j + ∂ū_j/∂x_i): three diagonals + three off-diagonals.
    Derivatives.ddx!(ws.S_xx, ws.u_filt, grid, dplan)
    Derivatives.ddy!(ws.S_yy, ws.v_filt, grid, dplan)
    Derivatives.ddz!(ws.S_zz, ws.w_filt, grid, dplan)
    Derivatives.ddy!(ws.S_xy, ws.u_filt, grid, dplan); Derivatives.ddx!(ws.scratch, ws.v_filt, grid, dplan)
    @. ws.S_xy = T(0.5) * (ws.S_xy + ws.scratch)
    Derivatives.ddz!(ws.S_xz, ws.u_filt, grid, dplan); Derivatives.ddx!(ws.scratch, ws.w_filt, grid, dplan)
    @. ws.S_xz = T(0.5) * (ws.S_xz + ws.scratch)
    Derivatives.ddz!(ws.S_yz, ws.v_filt, grid, dplan); Derivatives.ddy!(ws.scratch, ws.w_filt, grid, dplan)
    @. ws.S_yz = T(0.5) * (ws.S_yz + ws.scratch)

    # A loop, not a broadcast: fusing thirteen arrays builds a `Broadcasted` wide enough to spill.
    mask = FlowGeometries.Grids.mask(grid)
    @inbounds for I in CartesianIndices(Π)
        Π[I] = mask[I] ? -_sfs_contraction(
            ws.S_xx[I], ws.S_xy[I], ws.S_xz[I], ws.S_yy[I], ws.S_yz[I], ws.S_zz[I],
            ws.τ_xx[I], ws.τ_xy[I], ws.τ_xz[I], ws.τ_yy[I], ws.τ_yz[I], ws.τ_zz[I],
        ) : zero(T)
    end
    return Π
end

"""
    compute_Π!(Π::AbstractArray{T,3}, u, v, w, grid::StructuredGrid{Spherical,T,3}, kernel, scale; workspace=nothing, backend=AutoBackend(), mask_strategy=Deformable())

Full **three-dimensional spherical** cross-scale energy flux Π = -S̄_ij τ_ij: a genuine radius axis
`r[k]` (absolute distance from the planet center — see `FlowGeometries.Grids.StructuredGrid`'s 3D
constructor) and real vertical derivatives `∂/∂r`, unlike the 2.5D layer-by-layer path (which drops
the `u_r/r` curvature terms in `S_ee`/`S_nn` and the `S_er`/`S_nr`/`S_rr` radial strain entirely, since
it has no radial axis to differentiate against).

Velocities are rotated to planetary Cartesian for filtering (Aluie 2019 commutativity), then rotated
back to local (east, north, radial), through the same `_rotate_stress_to_local_enr`/`_sfs_contraction`
kernels the 2D spherical driver uses — that rotation is fully 3×3-general, and the 2.5D caller simply
discards its radial components. What differs here is the strain: the spherical strain-rate tensor in
orthogonal curvilinear
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
    grid::FlowGeometries.Grids.StructuredGrid{G,T,3},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    workspace::Union{Nothing, ΠWorkspace} = nothing,
    filter_plan::Union{Nothing, Filtering.AbstractFilterPlan} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    deriv_plan::Union{Nothing, Derivatives.StencilPlan} = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.SphericalGeometry{T}}
    _validate_field_sizes(grid, Π, u, v, w)
    ws = workspace === nothing ? ΠWorkspace(grid) : workspace
    dplan = deriv_plan === nothing ? Derivatives.StencilPlan(grid) : deriv_plan
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend) : filter_plan
    Nx, Ny, Nr = FlowGeometries.Grids.size_tuple(grid)

    # Rotate local (east, north, radial) velocity to planetary Cartesian at each point.
    @inbounds for k in 1:Nr, j in 1:Ny, i in 1:Nx
        if FlowGeometries.Grids.isactive(grid, i, j, k)
            λ, φ, _ = FlowGeometries.Grids.coords(grid, i, j, k)
            p_vel = FlowGeometries.Geometry.vector_to_cartesian(FlowGeometries.Grids.grid_geometry(grid), u[i, j, k], v[i, j, k], w[i, j, k], λ, φ)
            ws.ux[i, j, k] = p_vel[1]; ws.uy[i, j, k] = p_vel[2]; ws.uz[i, j, k] = p_vel[3]
        else
            ws.ux[i, j, k] = zero(T); ws.uy[i, j, k] = zero(T); ws.uz[i, j, k] = zero(T)
        end
    end

    Filtering.filter_apply_batch!((ws.ux_filt, ws.uy_filt, ws.uz_filt), (ws.ux, ws.uy, ws.uz), plan)
    # Pre-filter product scratch reuses `u_filt`/`v_filt`/`w_filt` (idle until the "rotate back to
    # local" loop below overwrites them with real values) plus `scratch`/`scratch2`/`scratch3`.
    @. ws.u_filt = ws.ux * ws.ux
    @. ws.v_filt = ws.ux * ws.uy
    @. ws.w_filt = ws.ux * ws.uz
    @. ws.scratch = ws.uy * ws.uy
    @. ws.scratch2 = ws.uy * ws.uz
    @. ws.scratch3 = ws.uz * ws.uz
    Filtering.filter_apply_batch!(
        (ws.uu_filt, ws.uv_filt, ws.uw_filt, ws.vv_filt, ws.vw_filt, ws.ww_filt),
        (ws.u_filt, ws.v_filt, ws.w_filt, ws.scratch, ws.scratch2, ws.scratch3),
        plan,
    )

    # Rotate filtered planetary velocities back to local (east, north, radial).
    @inbounds for k in 1:Nr, j in 1:Ny, i in 1:Nx
        if FlowGeometries.Grids.isactive(grid, i, j, k)
            λ, φ, _ = FlowGeometries.Grids.coords(grid, i, j, k)
            l_vel = FlowGeometries.Geometry.vector_from_cartesian(
                FlowGeometries.Grids.grid_geometry(grid), ws.ux_filt[i, j, k], ws.uy_filt[i, j, k], ws.uz_filt[i, j, k], λ, φ,
            )
            ws.u_filt[i, j, k] = l_vel[1]; ws.v_filt[i, j, k] = l_vel[2]; ws.w_filt[i, j, k] = l_vel[3]
        else
            ws.u_filt[i, j, k] = zero(T); ws.v_filt[i, j, k] = zero(T); ws.w_filt[i, j, k] = zero(T)
        end
    end

    # Rotate filtered planetary quadratic products into the local (east,north,radial) stress tensor.
    geo = FlowGeometries.Grids.grid_geometry(grid)
    @inbounds for k in 1:Nr, j in 1:Ny, i in 1:Nx
        if FlowGeometries.Grids.isactive(grid, i, j, k)
            λ, φ, _ = FlowGeometries.Grids.coords(grid, i, j, k)
            txx = ws.uu_filt[i, j, k] - ws.ux_filt[i, j, k] * ws.ux_filt[i, j, k]
            txy = ws.uv_filt[i, j, k] - ws.ux_filt[i, j, k] * ws.uy_filt[i, j, k]
            tyy = ws.vv_filt[i, j, k] - ws.uy_filt[i, j, k] * ws.uy_filt[i, j, k]
            txz = ws.uw_filt[i, j, k] - ws.ux_filt[i, j, k] * ws.uz_filt[i, j, k]
            tyz = ws.vw_filt[i, j, k] - ws.uy_filt[i, j, k] * ws.uz_filt[i, j, k]
            tzz = ws.ww_filt[i, j, k] - ws.uz_filt[i, j, k] * ws.uz_filt[i, j, k]
            τee, τen, τer, τnn, τnr, τrr = _rotate_stress_to_local_enr(geo, txx, txy, txz, tyy, tyz, tzz, λ, φ)
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
    Derivatives.ddx!(ws.S_xx, ws.u_filt, grid, dplan)
    Derivatives.ddy!(ws.S_yy, ws.v_filt, grid, dplan)
    Derivatives.ddz!(ws.S_zz, ws.w_filt, grid, dplan)
    Derivatives.ddy!(ws.S_xy, ws.u_filt, grid, dplan); Derivatives.ddx!(ws.scratch, ws.v_filt, grid, dplan)
    @. ws.S_xy = T(0.5) * (ws.S_xy + ws.scratch)
    Derivatives.ddz!(ws.S_xz, ws.u_filt, grid, dplan); Derivatives.ddx!(ws.scratch, ws.w_filt, grid, dplan)
    @. ws.S_xz = T(0.5) * (ws.S_xz + ws.scratch)
    Derivatives.ddz!(ws.S_yz, ws.v_filt, grid, dplan); Derivatives.ddy!(ws.scratch, ws.w_filt, grid, dplan)
    @. ws.S_yz = T(0.5) * (ws.S_yz + ws.scratch)

    @inbounds for k in 1:Nr, j in 1:Ny, i in 1:Nx
        if FlowGeometries.Grids.isactive(grid, i, j, k)
            _, φ, rk = FlowGeometries.Grids.coords(grid, i, j, k)
            sinφ, cosφ = sincos(φ)
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

    @inbounds for k in 1:Nr, j in 1:Ny, i in 1:Nx
        Π[i, j, k] = FlowGeometries.Grids.isactive(grid, i, j, k) ? -_sfs_contraction(
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
    grid::FlowGeometries.Grids.StructuredGrid{G,T,1},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    workspace::Union{Nothing, ΠWorkspace} = nothing,
    filter_plan::Union{Nothing, Filtering.AbstractFilterPlan} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    deriv_plan::Union{Nothing, Derivatives.StencilPlan} = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    _validate_field_sizes(grid, Π, u)
    ws = workspace === nothing ? ΠWorkspace(grid) : workspace
    dplan = deriv_plan === nothing ? Derivatives.StencilPlan(grid) : deriv_plan
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend) : filter_plan

    @. ws.scratch = u * u
    Filtering.filter_apply_batch!((ws.u_filt, ws.uu_filt), (u, ws.scratch), plan)
    @. ws.τ_xx = ws.uu_filt - ws.u_filt * ws.u_filt

    Derivatives.ddx!(ws.S_xx, ws.u_filt, grid, dplan)

    mask = FlowGeometries.Grids.mask(grid)
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
    grid::Union{FlowGeometries.Grids.StructuredGrid{G,T}, FlowGeometries.Grids.CurvilinearGrid{T,G}, FlowGeometries.Grids.UnstructuredGrid{T,G}},
    kernel::Kernels.AbstractFilterKernel,
    scales::AbstractVector;
    workspace::Union{Nothing, ΠWorkspace} = nothing,
    filter_plans::Union{Nothing, AbstractVector} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    method::Filtering.AbstractFilterMethod = Filtering.RealSpace(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    gsz = FlowGeometries.Grids.size_tuple(grid)
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
        if FlowGeometries.Grids.isactive(grid, Tuple(I)...)
            total_area += FlowGeometries.Grids.area(grid, Tuple(I)...)
        end
    end
    total_area > zero(T) || throw(ArgumentError("grid has no active cells (all masked out)"))

    # Sweep through scales. When the caller (typically `coarse_grain!`, which already builds one
    # plan per scale for its own `compute_Π!` loop) supplies `filter_plans`, reuse those instead of
    # rebuilding the same footprint a second time — otherwise this becomes the dominant allocation in
    # a `coarse_grain!` sweep, since each footprint build costs far more than the rest of the loop body.
    for s_idx in 1:Nscales
        ℓ = T(scales[s_idx])
        plan = filter_plans === nothing ?
            Filtering.plan_filter(grid, kernel, ℓ; mask_strategy=mask_strategy, backend=backend, method=method) :
            filter_plans[s_idx]

        # Filter velocity fields at this scale — batched (one derivation per point, not one per field).
        if w !== nothing
            Filtering.filter_apply_batch!((u_filt, v_filt, w_filt), (u, v, w), plan)
        else
            Filtering.filter_apply_batch!((u_filt, v_filt), (u, v), plan)
        end

        # Compute spatial average specific energy: E(ℓ) = 0.5 * ∫ |ū_ℓ|² dA / ∫ dA
        integrated_energy = zero(T)
        for I in idxs
            if FlowGeometries.Grids.isactive(grid, Tuple(I)...)
                vel2 = u_filt[I]^2 + v_filt[I]^2
                if w !== nothing
                    vel2 += w_filt[I]^2
                end
                integrated_energy += vel2 * FlowGeometries.Grids.area(grid, Tuple(I)...)
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
    grid::Union{FlowGeometries.Grids.StructuredGrid{G,T}, FlowGeometries.Grids.CurvilinearGrid{T,G}, FlowGeometries.Grids.UnstructuredGrid{T,G}},
    kernel::Kernels.AbstractFilterKernel,
    scales::AbstractVector;
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    method::Filtering.AbstractFilterMethod = Filtering.RealSpace(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
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
    grid::Union{FlowGeometries.Grids.StructuredGrid{G,T}, FlowGeometries.Grids.CurvilinearGrid{T,G}, FlowGeometries.Grids.UnstructuredGrid{T,G}},
    kernel::Kernels.AbstractFilterKernel,
    scales::AbstractVector;
    L::Real = one(T),
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    method::Filtering.AbstractFilterMethod = Filtering.RealSpace(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
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
    grid::FlowGeometries.Grids.StructuredGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
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

# The (east, north) block of the same rotation, for callers with no radial component. Inlined, so the
# three unused contractions are dead code the compiler drops.
@inline function _rotate_sym_to_local_en(
    geo::FlowGeometries.Geometry.AbstractSphericalGeometry,
    txx::T, txy::T, txz::T, tyy::T, tyz::T, tzz::T, λ::T, φ::T,
) where {T<:AbstractFloat}
    τ = FlowGeometries.Geometry.tensor_to_local(geo, txx, tyy, tzz, txy, txz, tyz, λ, φ)
    return τ.λλ, τ.λφ, τ.φφ
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
    grid::FlowGeometries.Grids.StructuredGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.SphericalGeometry{T}}
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend)
    flt(f) = (o = zeros(T, size(f)); Filtering.filter_apply!(o, f, plan); o)
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)

    # Local (u,v) -> planetary Cartesian (ux,uy,uz) at every point.
    ux = zeros(T, Nx, Ny); uy = zeros(T, Nx, Ny); uz = zeros(T, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        if FlowGeometries.Grids.isactive(grid, i, j)
            λ, φ = FlowGeometries.Grids.coords(grid, i, j)
            pc = FlowGeometries.Geometry.vector_to_cartesian(FlowGeometries.Grids.grid_geometry(grid), u[i, j], v[i, j], λ, φ)
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

    Lee = zeros(T, Nx, Ny); Len = zeros(T, Nx, Ny); Lnn = zeros(T, Nx, Ny)
    Cee = zeros(T, Nx, Ny); Cen = zeros(T, Nx, Ny); Cnn = zeros(T, Nx, Ny)
    Ree = zeros(T, Nx, Ny); Ren = zeros(T, Nx, Ny); Rnn = zeros(T, Nx, Ny)
    geo = FlowGeometries.Grids.grid_geometry(grid)
    for j in 1:Ny, i in 1:Nx
        if FlowGeometries.Grids.isactive(grid, i, j)
            λ, φ = FlowGeometries.Grids.coords(grid, i, j)
            Lee[i,j], Len[i,j], Lnn[i,j] = _rotate_sym_to_local_en(geo, Lxx[i,j], Lxy[i,j], Lxz[i,j], Lyy[i,j], Lyz[i,j], Lzz[i,j], λ, φ)
            Cee[i,j], Cen[i,j], Cnn[i,j] = _rotate_sym_to_local_en(geo, Cxx[i,j], Cxy[i,j], Cxz[i,j], Cyy[i,j], Cyz[i,j], Czz[i,j], λ, φ)
            Ree[i,j], Ren[i,j], Rnn[i,j] = _rotate_sym_to_local_en(geo, Rxx[i,j], Rxy[i,j], Rxz[i,j], Ryy[i,j], Ryz[i,j], Rzz[i,j], λ, φ)
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
    grid::FlowGeometries.Grids.StructuredGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    # One stencil table for every derivative below; they differ only in direction and field.
    dplan = Derivatives.StencilPlan(grid)
    gsz = FlowGeometries.Grids.size_tuple(grid)
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
        Sxx = similar(ā); Derivatives.ddx!(Sxx, ā, grid, dplan)
        Syy = similar(ā); Derivatives.ddy!(Syy, b̄, grid, dplan)
        p = similar(ā); q = similar(ā)
        Derivatives.ddy!(p, ā, grid, dplan); Derivatives.ddx!(q, b̄, grid, dplan)
        return (xx = Sxx, xy = T(0.5) .* (p .+ q), yy = Syy)
    end
    S_R = strain(u_rot, v_rot)
    S_D = strain(u_div, v_div)

    mask = FlowGeometries.Grids.mask(grid)
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
    grid::FlowGeometries.Grids.StructuredGrid{G,T,3},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    # One stencil table for every derivative below; they differ only in direction and field.
    dplan = Derivatives.StencilPlan(grid)
    gsz = FlowGeometries.Grids.size_tuple(grid)
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
        Sxx = similar(ā); Derivatives.ddx!(Sxx, ā, grid, dplan)
        Syy = similar(ā); Derivatives.ddy!(Syy, b̄, grid, dplan)
        Szz = similar(ā); Derivatives.ddz!(Szz, c̄, grid, dplan)
        p = similar(ā); q = similar(ā)
        Derivatives.ddy!(p, ā, grid, dplan); Derivatives.ddx!(q, b̄, grid, dplan)
        Sxy = T(0.5) .* (p .+ q)
        Derivatives.ddz!(p, ā, grid, dplan); Derivatives.ddx!(q, c̄, grid, dplan)
        Sxz = T(0.5) .* (p .+ q)
        Derivatives.ddz!(p, b̄, grid, dplan); Derivatives.ddy!(q, c̄, grid, dplan)
        Syz = T(0.5) .* (p .+ q)
        return (xx = Sxx, xy = Sxy, xz = Sxz, yy = Syy, yz = Syz, zz = Szz)
    end
    S_R = strain(u_rot, v_rot, w_rot)
    S_D = strain(u_div, v_div, w_div)

    mask = FlowGeometries.Grids.mask(grid)
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

Cartesian and spherical, on a 2D grid; the true-3D Cartesian method is below. `ddx!`/`ddy!` supply the
physical gradient in either geometry.
"""
function tracer_variance_flux(
    u::AbstractMatrix,
    v::AbstractMatrix,
    θ::AbstractMatrix,
    grid::FlowGeometries.Grids.StructuredGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    # One stencil table for every derivative below; they differ only in direction and field.
    dplan = Derivatives.StencilPlan(grid)
    gsz = FlowGeometries.Grids.size_tuple(grid)
    size(u) == gsz || throw(DimensionMismatch("u has size $(size(u)), grid expects $gsz"))
    size(v) == gsz || throw(DimensionMismatch("v has size $(size(v)), grid expects $gsz"))
    size(θ) == gsz || throw(DimensionMismatch("θ has size $(size(θ)), grid expects $gsz"))
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend)

    θ̄ = zeros(T, gsz)
    ū = zeros(T, gsz)
    v̄ = zeros(T, gsz)
    uθ = zeros(T, gsz)
    vθ = zeros(T, gsz)
    @. uθ = u * θ
    @. vθ = v * θ
    τx = zeros(T, gsz)
    τy = zeros(T, gsz)
    Filtering.filter_apply_batch!((ū, v̄, θ̄, τx, τy), (u, v, θ, uθ, vθ), plan)

    # Subfilter tracer flux τ_j = ⟨u_j θ⟩ - ū_j θ̄.
    @. τx -= ū * θ̄
    @. τy -= v̄ * θ̄

    # Resolved tracer gradient ∂_j θ̄.
    gx = similar(θ̄); Derivatives.ddx!(gx, θ̄, grid, dplan)
    gy = similar(θ̄); Derivatives.ddy!(gy, θ̄, grid, dplan)

    mask = FlowGeometries.Grids.mask(grid)
    return ifelse.(mask, .-(τx .* gx .+ τy .* gy), zero(T))
end

"""
    tracer_variance_flux(u, v, θ, grid::StructuredGrid{<:SphericalGeometry}, kernel, scale; ...) -> Πθ

Spherical form of the tracer-variance flux. `τ_j = ⟨u_j θ⟩ - ū_j θ̄` is a vector, so — exactly as in
[`compute_Π!`](@ref) and [`tau_decomposition`](@ref) — the velocity is rotated to planetary Cartesian
before filtering (Aluie 2019 commutativity: component-wise filtering of a local east/north pair is not
a filtered vector, since the local basis turns from point to point), and the filtered flux is rotated
back to the local east/north frame to contract against `∂_j θ̄`. The scalar `θ` needs no rotation. The
radial component of `τ` is dropped, matching this 2-D shell's dropping of radial derivatives.
"""
function tracer_variance_flux(
    u::AbstractMatrix,
    v::AbstractMatrix,
    θ::AbstractMatrix,
    grid::FlowGeometries.Grids.StructuredGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.SphericalGeometry{T}}
    # One stencil table for every derivative below; they differ only in direction and field.
    dplan = Derivatives.StencilPlan(grid)
    gsz = FlowGeometries.Grids.size_tuple(grid)
    size(u) == gsz || throw(DimensionMismatch("u has size $(size(u)), grid expects $gsz"))
    size(v) == gsz || throw(DimensionMismatch("v has size $(size(v)), grid expects $gsz"))
    size(θ) == gsz || throw(DimensionMismatch("θ has size $(size(θ)), grid expects $gsz"))
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend)
    geo = FlowGeometries.Grids.grid_geometry(grid)

    ux = zeros(T, gsz); uy = zeros(T, gsz); uz = zeros(T, gsz)
    uxθ = zeros(T, gsz); uyθ = zeros(T, gsz); uzθ = zeros(T, gsz)
    @inbounds for I in CartesianIndices(u)
        i = Tuple(I)
        FlowGeometries.Grids.isactive(grid, i...) || continue
        λ, φ = FlowGeometries.Grids.coords(grid, i...)
        p = FlowGeometries.Geometry.vector_to_cartesian(geo, u[I], v[I], λ, φ)
        ux[I] = p[1]; uy[I] = p[2]; uz[I] = p[3]
        uxθ[I] = p[1] * θ[I]; uyθ[I] = p[2] * θ[I]; uzθ[I] = p[3] * θ[I]
    end

    θ̄ = zeros(T, gsz)
    ūx = zeros(T, gsz); ūy = zeros(T, gsz); ūz = zeros(T, gsz)
    τX = zeros(T, gsz); τY = zeros(T, gsz); τZ = zeros(T, gsz)
    Filtering.filter_apply_batch!(
        (ūx, ūy, ūz, θ̄, τX, τY, τZ), (ux, uy, uz, θ, uxθ, uyθ, uzθ), plan,
    )
    @. τX -= ūx * θ̄
    @. τY -= ūy * θ̄
    @. τZ -= ūz * θ̄

    # Rotate the planetary-Cartesian subfilter flux back to local (east, north, radial); the radial
    # component is not used, as the resolved gradient here has no radial part.
    τe = zeros(T, gsz); τn = zeros(T, gsz)
    @inbounds for I in CartesianIndices(u)
        i = Tuple(I)
        FlowGeometries.Grids.isactive(grid, i...) || continue
        λ, φ = FlowGeometries.Grids.coords(grid, i...)
        l = FlowGeometries.Geometry.vector_from_cartesian(geo, τX[I], τY[I], τZ[I], λ, φ)
        τe[I] = l[1]; τn[I] = l[2]
    end

    gx = similar(θ̄); Derivatives.ddx!(gx, θ̄, grid, dplan)
    gy = similar(θ̄); Derivatives.ddy!(gy, θ̄, grid, dplan)

    mask = FlowGeometries.Grids.mask(grid)
    return ifelse.(mask, .-(τe .* gx .+ τn .* gy), zero(T))
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
    grid::FlowGeometries.Grids.StructuredGrid{G,T,3},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    # One stencil table for every derivative below; they differ only in direction and field.
    dplan = Derivatives.StencilPlan(grid)
    gsz = FlowGeometries.Grids.size_tuple(grid)
    size(u) == gsz || throw(DimensionMismatch("u has size $(size(u)), grid expects $gsz"))
    size(v) == gsz || throw(DimensionMismatch("v has size $(size(v)), grid expects $gsz"))
    size(w) == gsz || throw(DimensionMismatch("w has size $(size(w)), grid expects $gsz"))
    size(θ) == gsz || throw(DimensionMismatch("θ has size $(size(θ)), grid expects $gsz"))
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend)

    uθ = u .* θ; vθ = v .* θ; wθ = w .* θ
    ū = zeros(T, gsz); v̄ = zeros(T, gsz); w̄ = zeros(T, gsz); θ̄ = zeros(T, gsz)
    τx = zeros(T, gsz); τy = zeros(T, gsz); τz = zeros(T, gsz)
    Filtering.filter_apply_batch!((ū, v̄, w̄, θ̄, τx, τy, τz), (u, v, w, θ, uθ, vθ, wθ), plan)

    # Subfilter tracer flux τ_j = ⟨u_j θ⟩ - ū_j θ̄, now with a genuine vertical component.
    @. τx -= ū * θ̄
    @. τy -= v̄ * θ̄
    @. τz -= w̄ * θ̄

    # Resolved tracer gradient ∂_j θ̄, including the real vertical derivative.
    gx = similar(θ̄); Derivatives.ddx!(gx, θ̄, grid, dplan)
    gy = similar(θ̄); Derivatives.ddy!(gy, θ̄, grid, dplan)
    gz = similar(θ̄); Derivatives.ddz!(gz, θ̄, grid, dplan)

    mask = FlowGeometries.Grids.mask(grid)
    return ifelse.(mask, .-(τx .* gx .+ τy .* gy .+ τz .* gz), zero(T))
end

"""
    tracer_variance_flux(u, v, w, θ, grid::StructuredGrid{<:SphericalGeometry,T,3}, kernel, scale; ...) -> Πθ

Volumetric spherical shell (lon, lat, radius): the 3D counterpart of the spherical 2D method, keeping
the radial component of both the subfilter tracer flux and the resolved gradient. Velocities are
rotated to planetary Cartesian for filtering and the filtered flux is rotated back to local (east,
north, radial), the same convention the true-3D [`compute_Π!`](@ref) uses.
"""
function tracer_variance_flux(
    u::AbstractArray{<:Any,3},
    v::AbstractArray{<:Any,3},
    w::AbstractArray{<:Any,3},
    θ::AbstractArray{<:Any,3},
    grid::FlowGeometries.Grids.StructuredGrid{G,T,3},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.SphericalGeometry{T}}
    # One stencil table for every derivative below; they differ only in direction and field.
    dplan = Derivatives.StencilPlan(grid)
    gsz = FlowGeometries.Grids.size_tuple(grid)
    size(u) == gsz || throw(DimensionMismatch("u has size $(size(u)), grid expects $gsz"))
    size(v) == gsz || throw(DimensionMismatch("v has size $(size(v)), grid expects $gsz"))
    size(w) == gsz || throw(DimensionMismatch("w has size $(size(w)), grid expects $gsz"))
    size(θ) == gsz || throw(DimensionMismatch("θ has size $(size(θ)), grid expects $gsz"))
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend)
    geo = FlowGeometries.Grids.grid_geometry(grid)

    ux = zeros(T, gsz); uy = zeros(T, gsz); uz = zeros(T, gsz)
    uxθ = zeros(T, gsz); uyθ = zeros(T, gsz); uzθ = zeros(T, gsz)
    @inbounds for I in CartesianIndices(u)
        i = Tuple(I)
        FlowGeometries.Grids.isactive(grid, i...) || continue
        λ, φ = FlowGeometries.Grids.coords(grid, i...)
        p = FlowGeometries.Geometry.vector_to_cartesian(geo, u[I], v[I], w[I], λ, φ)
        ux[I] = p[1]; uy[I] = p[2]; uz[I] = p[3]
        uxθ[I] = p[1] * θ[I]; uyθ[I] = p[2] * θ[I]; uzθ[I] = p[3] * θ[I]
    end

    θ̄ = zeros(T, gsz)
    ūx = zeros(T, gsz); ūy = zeros(T, gsz); ūz = zeros(T, gsz)
    τX = zeros(T, gsz); τY = zeros(T, gsz); τZ = zeros(T, gsz)
    Filtering.filter_apply_batch!(
        (ūx, ūy, ūz, θ̄, τX, τY, τZ), (ux, uy, uz, θ, uxθ, uyθ, uzθ), plan,
    )
    @. τX -= ūx * θ̄
    @. τY -= ūy * θ̄
    @. τZ -= ūz * θ̄

    τe = zeros(T, gsz); τn = zeros(T, gsz); τr = zeros(T, gsz)
    @inbounds for I in CartesianIndices(u)
        i = Tuple(I)
        FlowGeometries.Grids.isactive(grid, i...) || continue
        λ, φ = FlowGeometries.Grids.coords(grid, i...)
        l = FlowGeometries.Geometry.vector_from_cartesian(geo, τX[I], τY[I], τZ[I], λ, φ)
        τe[I] = l[1]; τn[I] = l[2]; τr[I] = l[3]
    end

    gx = similar(θ̄); Derivatives.ddx!(gx, θ̄, grid, dplan)
    gy = similar(θ̄); Derivatives.ddy!(gy, θ̄, grid, dplan)
    gz = similar(θ̄); Derivatives.ddz!(gz, θ̄, grid, dplan)

    mask = FlowGeometries.Grids.mask(grid)
    return ifelse.(mask, .-(τe .* gx .+ τn .* gy .+ τr .* gz), zero(T))
end

end # module
