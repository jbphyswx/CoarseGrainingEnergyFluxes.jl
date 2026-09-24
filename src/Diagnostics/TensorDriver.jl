# ---------------------------------------------------------------------------
# Shared 2D driver for the per-point tensor physics (rotation / SFS stress / strain contraction),
# called by BOTH the StructuredGrid and CurvilinearGrid `compute_Π!` methods. Both grids reach their
# geometry only through `FlowGeometries.Grids.coords`/`isactive` and the `ddx!`/`ddy!` operators, so this one kernel
# serves both — no duplicated tensor math. `deriv_plan` is a `Derivatives.StencilPlan` for a
# `StructuredGrid` and a `Operators.gradient_plan` for a curvilinear grid or a node set; either way
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
@inline function _grad2!(g1, g2, f, _grid, plan::FlowGeometries.Operators.GradientPlan)
    FlowGeometries.Operators.gradient!(g1, g2, f, plan)
    return nothing
end

"""
    output_grid(grid, mask_strategy) -> grid
    output_grid(grid, plan) -> grid

The grid a filter's outputs are read on. `ZeroFill` defines the filtered field at every cell, so its
outputs are differenced, rotated and contracted with every cell active; under `Deformable` a masked cell
is zero, and the outputs keep the grid's mask. The inputs a filter reads keep the mask either way.

A `deriv_plan` passed to a diagnostic differences filtered fields, so it is built on this grid:
`Derivatives.gradient_plan(output_grid(grid, mask_strategy))` on a curvilinear grid or a node set. A
[`Derivatives.StencilPlan`](@ref) does not depend on the mask and serves either grid.
"""
@inline output_grid(grid, ::Filtering.ZeroFill) = _unmasked(grid)
@inline output_grid(grid, ::Filtering.AbstractMaskStrategy) = grid
@inline output_grid(grid, plan::Filtering.AbstractFilterPlan) = output_grid(grid, Filtering.plan_strategy(plan))

# Whether `og` holds exactly `grid`'s active cells, so a plan built on either serves both.
@inline _same_cells(grid, og) = og === grid || all(FlowGeometries.Grids.mask(grid))

"""
    _derived_plan(plan, grid, og, kernel, scale, backend) -> plan

The plan for a field the first filter produced. Under `ZeroFill` that field is defined at every cell of
the output grid `og`, land included, so it is filtered with every cell active (a Germano moment's
second filter reads `ū` over land); where `og` holds the same cells as `grid`, `plan` serves.
"""
function _derived_plan(plan::Filtering.AbstractFilterPlan, grid, og, kernel, scale, backend)
    _same_cells(grid, og) && return plan
    return Filtering.plan_filter(
        og, kernel, scale;
        mask_strategy = Filtering.ZeroFill(), backend = backend, method = Filtering.plan_method(plan),
    )
end

@inline _unmasked(grid::FlowGeometries.Grids.AbstractGrid) = FlowGeometries.Grids.rebuild(
    grid, (; mask = FlowGeometries.Grids.AllActive(size(FlowGeometries.Grids.mask(grid)))),
)
@inline _unmasked(grid::FlowGeometries.Grids.RotatedGrid) =
    FlowGeometries.Grids.rebuild(grid, (; base = _unmasked(FlowGeometries.Grids.base_grid(grid))))

# The geometry-only derivative object of an architecture: a stencil table where there are axes to
# difference along, a least-squares tangent-plane gradient where there are not. Every diagnostic below
# builds its own through this, so one call site serves structured, curvilinear and flat-cell grids.
const AnyDerivPlan = Union{Derivatives.StencilPlan, FlowGeometries.Operators.GradientPlan}

_default_deriv_plan(grid::FlowGeometries.Grids.StructuredGrid) = Derivatives.StencilPlan(grid)
_default_deriv_plan(grid::FlowGeometries.Grids.AbstractGrid) = Derivatives.gradient_plan(grid)

@inline _resolve_deriv_plan(::Nothing, grid) = _default_deriv_plan(grid)
@inline _resolve_deriv_plan(plan::AnyDerivPlan, _grid) = plan

# The diagnostics that contract a two-component tangent vector or a 2×2 tangent tensor need a grid
# that resolves two directions. A 1-D grid carries one velocity component and one derivative, so it
# has no second component to supply; this names the call and the grid at the entry point.
@inline function _require_tangent_pair(grid::FlowGeometries.Grids.AbstractGrid, fname::AbstractString)
    d = FlowGeometries.Grids.ncoordinates(grid)
    d == 2 || throw(ArgumentError(
        "$fname contracts a two-component tangent field, so it needs a grid resolving two " *
        "directions; $(nameof(typeof(grid))) resolves $d",
    ))
    return nothing
end

# `tanφ / R` at one point, the curvature factor the spherical strain and curl corrections carry. Zero
# at the pole, where `h_λ = R cosφ` vanishes and the λ-derivative does not exist.
@inline function _tan_factor(
    geo::FlowGeometries.Geometry.AbstractSphericalGeometry{T}, φ::T,
) where {T<:AbstractFloat}
    sinφ, cosφ = sincos(φ)
    R = FlowGeometries.Geometry.radius(geo)
    return abs(cosφ) > T(1e-12) ? sinφ / (R * cosφ) : zero(T)
end

# Rotate a planetary-Cartesian symmetric stress to the local frame at (λ,φ). The result keeps the
# component keys `λλ`, `λφ`, `λr`, `φφ`, `φr`, `rr`, so a caller writing it into positional buffers
# names the component it wants for each slot. A caller with no radial velocity passes zero for
# `txz`/`tyz`/`tzz` and reads only the three keys carrying no `r`.
@inline _rotate_stress_to_local_enr(
    geo::FlowGeometries.Geometry.AbstractSphericalGeometry,
    txx::T, txy::T, txz::T, tyy::T, tyz::T, tzz::T, λ::T, φ::T,
) where {T<:AbstractFloat} =
    FlowGeometries.Geometry.tensor_to_local(geo, txx, tyy, tzz, txy, txz, tyz, λ, φ)

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
function _fill_stress_strain!(
    u::AbstractArray,
    v::AbstractArray,
    w::Union{Nothing, AbstractArray},
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    ws::ΠWorkspace,
    plan::Filtering.AbstractFilterPlan,
    deriv_plan,
    analyzed = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    has_w = w !== nothing
    cells = CartesianIndices(ws.u_filt)
    ograd = output_grid(grid, plan)

    if G <: FlowGeometries.Geometry.CartesianGeometry{T}
        # -------------------------------------------------------------------
        # Cartesian Case
        # -------------------------------------------------------------------
        # Filter velocity components — batched: one neighbour-list/weight derivation per point,
        # applied to all primitives at once (see `Filtering.filter_apply_batch!`), not once per field.
        if analyzed !== nothing
            # A sweep hoists the scale-independent forward transform out of the scale loop, so only the
            # per-scale synthesis is left here.
            _synthesize_all!(_velocity_outs(ws, has_w), analyzed.velocity, plan)
        elseif has_w
            Filtering.filter_apply_batch!((ws.u_filt, ws.v_filt, ws.w_filt), (u, v, w), plan)
        else
            Filtering.filter_apply_batch!((ws.u_filt, ws.v_filt), (u, v), plan)
        end

        # Filter products: u², uv, vv, etc. — also batched, in one pass. `ux`/`uy`/`uz`/`ux_filt`/
        # `uy_filt`/`uz_filt` are spherical-only fields, genuinely idle in this Cartesian branch, so
        # they're reused here purely as pre-filter product scratch (no new workspace fields needed).
        if analyzed !== nothing
            _synthesize_all!(_product_outs(ws, has_w), analyzed.product, plan)
        else
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
        _grad2!(ws.S_xx, ws.S_xy, ws.u_filt, ograd, deriv_plan)     # ∂ū/∂x, ∂ū/∂y
        _grad2!(ws.scratch, ws.S_yy, ws.v_filt, ograd, deriv_plan)  # ∂v̄/∂x, ∂v̄/∂y
        @. ws.S_xy = T(0.5) * (ws.S_xy + ws.scratch)

        if has_w
            # S_xz = 0.5 * (∂ū/∂z + ∂w̄/∂x), S_yz = 0.5 * (∂v̄/∂z + ∂w̄/∂y); ∂/∂z is zero for a
            # level stack, leaving the horizontal gradient of w̄.
            _grad2!(ws.S_xz, ws.S_yz, ws.w_filt, ograd, deriv_plan)
            @. ws.S_xz = T(0.5) * ws.S_xz
            @. ws.S_yz = T(0.5) * ws.S_yz

            # S_zz = ∂w̄/∂z = 0 (for standard 2.5D datasets)
            fill!(ws.S_zz, zero(T))
        end

    else
        # -------------------------------------------------------------------
        # Spherical Case (Aluie 2019 commutativity formulation)
        # -------------------------------------------------------------------
        # Transform local coordinates (u_east, v_north) to global Cartesian (u_X, u_Y, u_Z). This is a
        # function of the inputs and the grid, not of the scale, so a sweep does it once up front and
        # hands down a `SphericalAnalysis`; a standalone call does it here.
        analyzed isa SphericalAnalysis || _fill_planetary!(ws, u, v, w, grid)

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
        for I in cells
            let i = Tuple(I)
                if FlowGeometries.Grids.isactive(ograd, i...)
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
        for I in cells
            let i = Tuple(I)
                if FlowGeometries.Grids.isactive(ograd, i...)
                    λ, φ = FlowGeometries.Grids.coords(grid, i...)
                    txx = ws.uu_filt[I] - ws.ux_filt[I] * ws.ux_filt[I]
                    txy = ws.uv_filt[I] - ws.ux_filt[I] * ws.uy_filt[I]
                    tyy = ws.vv_filt[I] - ws.uy_filt[I] * ws.uy_filt[I]
                    txz = ws.uw_filt[I] - ws.ux_filt[I] * ws.uz_filt[I]
                    tyz = ws.vw_filt[I] - ws.uy_filt[I] * ws.uz_filt[I]
                    tzz = ws.ww_filt[I] - ws.uz_filt[I] * ws.uz_filt[I]
                    τl = _rotate_stress_to_local_enr(geo, txx, txy, txz, tyy, tyz, tzz, λ, φ)
                    τee, τen, τer = τl.λλ, τl.λφ, τl.λr
                    τnn, τnr, τrr = τl.φφ, τl.φr, τl.rr
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
        _grad2!(ws.S_xx, ws.S_xy, ws.u_filt, ograd, deriv_plan)
        _grad2!(ws.scratch, ws.S_yy, ws.v_filt, ograd, deriv_plan)

        R = FlowGeometries.Geometry.radius(geo)
        for I in cells
            let i = Tuple(I)
                if FlowGeometries.Grids.isactive(ograd, i...)
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
            _grad2!(ws.S_xz, ws.S_yz, ws.w_filt, ograd, deriv_plan)
            @. ws.S_xz = T(0.5) * ws.S_xz
            @. ws.S_yz = T(0.5) * ws.S_yz

            # S_rr = ∂w̄/∂r = 0
            fill!(ws.S_zz, zero(T))
        end
    end

    return nothing
end

# Π = −S̄:τ̄ from a filled workspace. Split from the fill above so the diagnostics that decompose the
# same `S̄` and `τ̄` — the strain/convergence pair, the Helmholtz channels — read one stress and one
# strain, whatever the metric built them.
#
# Since stress & strain rates are symmetric:
# S̄_ij τ_ij = S_xx*τ_xx + 2*S_xy*τ_xy + S_yy*τ_yy (2D)
# S̄_ij τ_ij = S_xx*τ_xx + 2*S_xy*τ_xy + S_yy*τ_yy + 2*S_xz*τ_xz + 2*S_yz*τ_yz + S_zz*τ_zz (3D)
function _contract_Π!(Π::AbstractArray{T}, ws::ΠWorkspace, grid, has_w::Bool) where {T<:AbstractFloat}
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

function _compute_Π!(
    Π::AbstractArray{T},
    u::AbstractArray,
    v::AbstractArray,
    w::Union{Nothing, AbstractArray},
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    ws::ΠWorkspace,
    plan::Filtering.AbstractFilterPlan,
    deriv_plan,
    analyzed = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    _fill_stress_strain!(u, v, w, grid, ws, plan, deriv_plan, analyzed)
    return _contract_Π!(Π, ws, output_grid(grid, plan), w !== nothing)
end


"""
    compute_Π!(Π, u, v, w, grid, kernel, scale; workspace=nothing, deriv_plan=nothing, backend=AutoBackend(), mask_strategy=ZeroFill(), method=RealSpace())

Cross-scale kinetic energy flux Π = -S̄_ij τ_ij on a `FlowGeometries.Grids.UnstructuredGrid` (scattered
points, node-indexed) — the same physics as the 2D methods (planetary-Cartesian rotation for
spherical geometry), via `_compute_Π!`. The resolved strain uses the node-indexed WLSQ
gradient (`Operators.gradient_plan` + `Operators.gradient!`). `method` defaults to
`RealSpace()` here. The transform is exact for a
band-limited field and its per-apply cost does not grow with the filter scale. `RealSpace()` applies
the kernel as written, with compact support; a transform's support is global.
"""
function compute_Π!(
    Π::AbstractVector{T},
    u::AbstractVector,
    v::AbstractVector,
    w::Union{Nothing, AbstractVector},
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    workspace::Union{Nothing, ΠWorkspace} = nothing,
    deriv_plan::Union{Nothing, FlowGeometries.Operators.GradientPlan} = nothing,
    filter_plan::Union{Nothing, Filtering.AbstractFilterPlan} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Filtering.AbstractFilterMethod = Filtering.RealSpace(),
    analyzed = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    _validate_field_sizes(grid, Π, u, v, w)
    ws = workspace === nothing ? ΠWorkspace(grid; has_w = w !== nothing) : workspace
    _check_workspace_w(ws, w)
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend, method=method) : filter_plan
    # The gradient is fitted over the cells the filtered field is read on.
    dplan = deriv_plan === nothing ? Derivatives.gradient_plan(output_grid(grid, plan)) : deriv_plan
    return _compute_Π!(Π, u, v, w, grid, ws, plan, dplan, analyzed)
end

"""
    compute_Π!(Π, u, v, w, grid::CurvilinearGrid, kernel, scale; workspace=nothing, deriv_plan=nothing, backend=AutoBackend(), mask_strategy=ZeroFill())

Cross-scale kinetic energy flux Π = -S̄_ij τ_ij on a `FlowGeometries.Grids.CurvilinearGrid`. Identical
physics to the `StructuredGrid` 2D method — it shares the same `_compute_Π!` tensor kernel
— but the resolved strain uses the least-squares tangent-plane gradient
(`Operators.gradient!` over a `Operators.gradient_plan`, both components from one neighbour
sweep) and real-space filtering uses the scattered per-point footprint. Pass a prebuilt
`deriv_plan = Derivatives.gradient_plan(output_grid(grid, mask_strategy))` (see [`output_grid`](@ref))
and a reusable `workspace` to avoid rebuilding them per call across a scale sweep.
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
    deriv_plan::Union{Nothing, FlowGeometries.Operators.GradientPlan} = nothing,
    filter_plan::Union{Nothing, Filtering.AbstractFilterPlan} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    analyzed = nothing,
) where {T<:AbstractFloat}
    _validate_field_sizes(grid, Π, u, v, w)
    ws = workspace === nothing ? ΠWorkspace(grid; has_w = w !== nothing) : workspace
    _check_workspace_w(ws, w)
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend) : filter_plan
    dplan = deriv_plan === nothing ? Derivatives.gradient_plan(output_grid(grid, plan)) : deriv_plan
    return _compute_Π!(Π, u, v, w, grid, ws, plan, dplan, analyzed)
end

"""
    compute_Π!(Π::AbstractArray{T,3}, u, v, w, grid::StructuredGrid{T,Cartesian,3}, kernel, scale; mask_strategy=ZeroFill(), backend=AutoBackend())

Full **three-dimensional** Cartesian cross-scale energy flux Π = -S̄_ij τ_ij with all nine strain
components (the diagonal `S_zz = ∂w̄/∂z` and the off-diagonals `S_xz, S_yz` carry genuine vertical
derivatives, unlike the 2.5D layer-by-layer path). The 3D grid carries a 3D mask, so masked cells are
handled per-cell in all three directions.

The contraction is the symmetric six-term sum
`S̄:τ = S_xx τ_xx + S_yy τ_yy + S_zz τ_zz + 2(S_xy τ_xy + S_xz τ_xz + S_yz τ_yz)`.

Dispatched on a 3D output array + 3D Cartesian grid (the 2D method takes an `AbstractMatrix`); see
the separate `StructuredGrid{T,Spherical,3}` method below for the spherical volumetric case (genuine
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
    grid::FlowGeometries.Grids.StructuredGrid{T,G,3},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    workspace::Union{Nothing, ΠWorkspace} = nothing,
    filter_plan::Union{Nothing, Filtering.AbstractFilterPlan} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    deriv_plan::Union{Nothing, Derivatives.StencilPlan} = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    _validate_field_sizes(grid, Π, u, v, w)
    ws = workspace === nothing ? ΠWorkspace(grid; has_w = w !== nothing) : workspace
    _check_workspace_w(ws, w)
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
    og = output_grid(grid, plan)
    Derivatives.ddx!(ws.S_xx, ws.u_filt, og, dplan)
    Derivatives.ddy!(ws.S_yy, ws.v_filt, og, dplan)
    Derivatives.ddz!(ws.S_zz, ws.w_filt, og, dplan)
    Derivatives.ddy!(ws.S_xy, ws.u_filt, og, dplan); Derivatives.ddx!(ws.scratch, ws.v_filt, og, dplan)
    @. ws.S_xy = T(0.5) * (ws.S_xy + ws.scratch)
    Derivatives.ddz!(ws.S_xz, ws.u_filt, og, dplan); Derivatives.ddx!(ws.scratch, ws.w_filt, og, dplan)
    @. ws.S_xz = T(0.5) * (ws.S_xz + ws.scratch)
    Derivatives.ddz!(ws.S_yz, ws.v_filt, og, dplan); Derivatives.ddy!(ws.scratch, ws.w_filt, og, dplan)
    @. ws.S_yz = T(0.5) * (ws.S_yz + ws.scratch)

    # A loop: fusing thirteen arrays in one broadcast builds a `Broadcasted` wide enough to spill.
    mask = FlowGeometries.Grids.mask(og)
    @inbounds for I in CartesianIndices(Π)
        Π[I] = mask[I] ? -_sfs_contraction(
            ws.S_xx[I], ws.S_xy[I], ws.S_xz[I], ws.S_yy[I], ws.S_yz[I], ws.S_zz[I],
            ws.τ_xx[I], ws.τ_xy[I], ws.τ_xz[I], ws.τ_yy[I], ws.τ_yz[I], ws.τ_zz[I],
        ) : zero(T)
    end
    return Π
end

"""
    compute_Π!(Π::AbstractArray{T,3}, u, v, w, grid::StructuredGrid{T,Spherical,3}, kernel, scale; workspace=nothing, backend=AutoBackend(), mask_strategy=ZeroFill())

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
    grid::FlowGeometries.Grids.StructuredGrid{T,G,3},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    workspace::Union{Nothing, ΠWorkspace} = nothing,
    filter_plan::Union{Nothing, Filtering.AbstractFilterPlan} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    deriv_plan::Union{Nothing, Derivatives.StencilPlan} = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.SphericalGeometry{T}}
    _validate_field_sizes(grid, Π, u, v, w)
    ws = workspace === nothing ? ΠWorkspace(grid; has_w = w !== nothing) : workspace
    _check_workspace_w(ws, w)
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
    og = output_grid(grid, plan)
    @inbounds for k in 1:Nr, j in 1:Ny, i in 1:Nx
        if FlowGeometries.Grids.isactive(og, i, j, k)
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
        if FlowGeometries.Grids.isactive(og, i, j, k)
            λ, φ, _ = FlowGeometries.Grids.coords(grid, i, j, k)
            txx = ws.uu_filt[i, j, k] - ws.ux_filt[i, j, k] * ws.ux_filt[i, j, k]
            txy = ws.uv_filt[i, j, k] - ws.ux_filt[i, j, k] * ws.uy_filt[i, j, k]
            tyy = ws.vv_filt[i, j, k] - ws.uy_filt[i, j, k] * ws.uy_filt[i, j, k]
            txz = ws.uw_filt[i, j, k] - ws.ux_filt[i, j, k] * ws.uz_filt[i, j, k]
            tyz = ws.vw_filt[i, j, k] - ws.uy_filt[i, j, k] * ws.uz_filt[i, j, k]
            tzz = ws.ww_filt[i, j, k] - ws.uz_filt[i, j, k] * ws.uz_filt[i, j, k]
            τl = _rotate_stress_to_local_enr(geo, txx, txy, txz, tyy, tyz, tzz, λ, φ)
            ws.τ_xx[i, j, k] = τl.λλ; ws.τ_xy[i, j, k] = τl.λφ; ws.τ_xz[i, j, k] = τl.λr
            ws.τ_yy[i, j, k] = τl.φφ; ws.τ_yz[i, j, k] = τl.φr; ws.τ_zz[i, j, k] = τl.rr
        else
            ws.τ_xx[i, j, k] = zero(T); ws.τ_xy[i, j, k] = zero(T); ws.τ_xz[i, j, k] = zero(T)
            ws.τ_yy[i, j, k] = zero(T); ws.τ_yz[i, j, k] = zero(T); ws.τ_zz[i, j, k] = zero(T)
        end
    end

    # Strain: ddx!/ddy!/ddz! are already metric-scaled (1/(r cosφ), 1/r, and a plain radial
    # derivative respectively, using the LOCAL r[k] at each level), so this gives the "flat" part of
    # each component; the curvature-correction terms are added in the loop below.
    Derivatives.ddx!(ws.S_xx, ws.u_filt, og, dplan)
    Derivatives.ddy!(ws.S_yy, ws.v_filt, og, dplan)
    Derivatives.ddz!(ws.S_zz, ws.w_filt, og, dplan)
    Derivatives.ddy!(ws.S_xy, ws.u_filt, og, dplan); Derivatives.ddx!(ws.scratch, ws.v_filt, og, dplan)
    @. ws.S_xy = T(0.5) * (ws.S_xy + ws.scratch)
    Derivatives.ddz!(ws.S_xz, ws.u_filt, og, dplan); Derivatives.ddx!(ws.scratch, ws.w_filt, og, dplan)
    @. ws.S_xz = T(0.5) * (ws.S_xz + ws.scratch)
    Derivatives.ddz!(ws.S_yz, ws.v_filt, og, dplan); Derivatives.ddy!(ws.scratch, ws.w_filt, og, dplan)
    @. ws.S_yz = T(0.5) * (ws.S_yz + ws.scratch)

    @inbounds for k in 1:Nr, j in 1:Ny, i in 1:Nx
        if FlowGeometries.Grids.isactive(og, i, j, k)
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
        Π[i, j, k] = FlowGeometries.Grids.isactive(og, i, j, k) ? -_sfs_contraction(
            ws.S_xx[i, j, k], ws.S_xy[i, j, k], ws.S_xz[i, j, k],
            ws.S_yy[i, j, k], ws.S_yz[i, j, k], ws.S_zz[i, j, k],
            ws.τ_xx[i, j, k], ws.τ_xy[i, j, k], ws.τ_xz[i, j, k],
            ws.τ_yy[i, j, k], ws.τ_yz[i, j, k], ws.τ_zz[i, j, k],
        ) : zero(T)
    end
    return Π
end

"""
    compute_Π!(Π::AbstractVector, u, grid::StructuredGrid{T,Cartesian,1}, kernel, scale; workspace=nothing, backend=AutoBackend(), mask_strategy=ZeroFill())

1D cross-scale energy flux Π = -S̄_xx τ_xx on a genuinely 1D `StructuredGrid` (a single scalar
velocity component `u` along one axis — the 1D analog of the 2D tensor contraction, which reduces to
a single term since there's only one strain/stress component). Not the 2D-with-singleton-dimension
case (which reuses the 2D methods directly).
"""
function compute_Π!(
    Π::AbstractVector{T},
    u::AbstractVector,
    grid::FlowGeometries.Grids.StructuredGrid{T,G,1},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    workspace::Union{Nothing, ΠWorkspace} = nothing,
    filter_plan::Union{Nothing, Filtering.AbstractFilterPlan} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    deriv_plan::Union{Nothing, Derivatives.StencilPlan} = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    _validate_field_sizes(grid, Π, u)
    # One axis carries one velocity component, so there is no vertical component for the workspace to
    # hold and none for a caller to supply.
    ws = workspace === nothing ? ΠWorkspace(grid) : workspace
    dplan = deriv_plan === nothing ? Derivatives.StencilPlan(grid) : deriv_plan
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend) : filter_plan

    @. ws.scratch = u * u
    Filtering.filter_apply_batch!((ws.u_filt, ws.uu_filt), (u, ws.scratch), plan)
    @. ws.τ_xx = ws.uu_filt - ws.u_filt * ws.u_filt

    og = output_grid(grid, plan)
    Derivatives.ddx!(ws.S_xx, ws.u_filt, og, dplan)

    mask = FlowGeometries.Grids.mask(og)
    @inbounds @. Π = ifelse(mask, -(ws.S_xx * ws.τ_xx), zero(T))
    return Π
end
