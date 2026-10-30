# ---------------------------------------------------------------------------
# Filtering, frame rotation, subfilter stress, strain and contraction shared by the flux diagnostics.
# `deriv_plan` is `Derivatives.derivative_plan(grid)`; a masked or undefined value is zero. The grids
# and frame plans these helpers take are a workspace's `Placement`'s, and `b` is the backend its loops
# run under.
# ---------------------------------------------------------------------------

# Both tangent components of one field's gradient, per unit distance.
@inline function _grad2!(g1, g2, f, grid::FlowGeometries.Grids.StructuredGrid,
                         plans::FlowGeometries.Operators.StencilPlans, b)
    FlowGeometries.Operators.derivative!(g1, f, grid, plans, 1; masked = zero(eltype(g1)), execution = b)
    FlowGeometries.Operators.derivative!(g2, f, grid, plans, 2; masked = zero(eltype(g2)), execution = b)
    return nothing
end
@inline function _grad2!(g1, g2, f, _grid, plan::FlowGeometries.Operators.GradientPlan, b)
    FlowGeometries.Operators.gradient!(g1, g2, f, plan; execution = b)
    return nothing
end

# The derivative of `f` along direction `d` of a structured grid, per unit distance.
@inline _dd!(out, f, grid::FlowGeometries.Grids.StructuredGrid, plans::FlowGeometries.Operators.StencilPlans,
             d::Int, b) =
    FlowGeometries.Operators.derivative!(out, f, grid, plans, d; masked = zero(eltype(out)), execution = b)

# All three components of one field's gradient on a structured volume, per unit distance.
@inline function _grad3!(g::NTuple{3,AbstractArray}, f, grid::FlowGeometries.Grids.StructuredGrid,
                         plans::FlowGeometries.Operators.StencilPlans, b)
    _dd!(g[1], f, grid, plans, 1, b)
    _dd!(g[2], f, grid, plans, 2, b)
    _dd!(g[3], f, grid, plans, 3, b)
    return nothing
end

# `S̄ = ½(∇ū + ∇ūᵀ)` of the filtered velocity `u` into the arrays `S`, ordered as
# `FlowGeometries.Operators.strain_components(grid)` names them: the diagonal, then the upper triangle row
# by row. The frame's turning is included; a cell is zero where it is inactive or a derivative it reads is
# undefined. `scratch` is one array per direction.
@inline _strain!(S::Tuple, u::Tuple, grid, plan, scratch::Tuple, b) = FlowGeometries.Operators.strain_rate!(
    NamedTuple{FlowGeometries.Operators.strain_components(grid)}(S), u, grid, plan;
    masked = zero(eltype(first(u))), scratch = scratch, execution = b,
)

# The local components `l` of the ambient Cartesian vector `p`, zero on inactive cells.
@inline _ambient_to_local!(l::NTuple{2}, p::NTuple{3}, frames, b) =
    (FlowGeometries.Operators.to_local!(l[1], l[2], nothing, p[1], p[2], p[3], frames; execution = b); l)
@inline _ambient_to_local!(l::NTuple{3}, p::NTuple{3}, frames, b) =
    (FlowGeometries.Operators.to_local!(l[1], l[2], l[3], p[1], p[2], p[3], frames; execution = b); l)

"""
    output_grid(grid, mask_strategy) -> grid
    output_grid(grid, plan) -> grid

The grid a filter's outputs are read on. `ZeroFill` defines the filtered field at every cell, so its
outputs are differenced, rotated and contracted with every cell active; under `Deformable` a masked cell
is zero, and the outputs keep the grid's mask. The inputs a filter reads keep the mask either way.

A `deriv_plan` passed to a diagnostic differences filtered fields, so it is built on this grid:
`Derivatives.derivative_plan(output_grid(grid, mask_strategy))`.
"""
@inline output_grid(grid, ::Filtering.ZeroFill) = _unmasked(grid)
@inline output_grid(grid, ::Filtering.AbstractMaskStrategy) = grid
@inline output_grid(grid, plan::Filtering.AbstractFilterPlan) = output_grid(grid, Filtering.plan_strategy(plan))
@inline output_grid(grid, ::SpinFilter) = output_grid(grid, Filtering.ZeroFill())

# Whether `og`, `grid`'s output grid, holds exactly `grid`'s active cells, so a plan built on either serves
# both. `output_grid` returns `grid` itself or `grid` with an `AllActive` mask, which is `grid`'s own type
# exactly when its mask is already `AllActive`.
@inline _same_cells(::G, ::G) where {G} = true
@inline _same_cells(_, _) = false

"""
    _derived_plan(plan, grid, og, kernel, scale) -> plan

The plan for a field the first filter produced, on `plan`'s backend. Under `ZeroFill` that field is
defined at every cell of the output grid `og`, land included, so it is filtered with every cell active
(a Germano moment's second filter reads `ū` over land); where `og` holds the same cells as `grid`, `plan`
serves.
"""
function _derived_plan(plan::Filtering.AbstractFilterPlan, grid, og, kernel, scale)
    _same_cells(grid, og) && return plan
    return Filtering.plan_filter(
        og, kernel, scale;
        mask_strategy = Filtering.ZeroFill(), execution = Filtering.plan_execution(plan),
        method = Filtering.plan_method(plan),
    )
end

@inline _unmasked(grid::FlowGeometries.Grids.AbstractGrid) = FlowGeometries.Grids.rebuild(
    grid, (; mask = FlowGeometries.Grids.AllActive(size(FlowGeometries.Grids.mask(grid)))),
)
@inline _unmasked(grid::FlowGeometries.Grids.RotatedGrid) =
    FlowGeometries.Grids.rebuild(grid, (; base = _unmasked(FlowGeometries.Grids.base_grid(grid))))

const AnyDerivPlan = Union{FlowGeometries.Operators.StencilPlans, FlowGeometries.Operators.GradientPlan}

# The curved surfaces whose diagnostics filter ambient Cartesian components and rotate back to the local
# frame: a sphere and a spheroid.
const _CurvedSurface{T} = Union{
    FlowGeometries.Geometry.AbstractSphericalGeometry{T}, FlowGeometries.Geometry.AbstractEllipsoidalGeometry{T},
}

@inline _resolve_deriv_plan(::Nothing, grid) = Derivatives.derivative_plan(grid)
@inline _resolve_deriv_plan(plan::AnyDerivPlan, _grid) = plan

# Throws unless `grid` resolves two directions, naming the call `fname` and the grid.
@inline function _require_tangent_pair(grid::FlowGeometries.Grids.AbstractGrid, fname::AbstractString)
    d = FlowGeometries.Grids.ncoordinates(grid)
    d == 2 || throw(ArgumentError(
        "$fname contracts a two-component tangent field, so it needs a grid resolving two " *
        "directions; $(nameof(typeof(grid))) resolves $d",
    ))
    return nothing
end

# The ambient Cartesian subfilter stress `τ = (pp)‾ − p̄ p̄` of the six filtered products, in place over
# them, then its local components into the arrays named `(λλ, φφ, rr, λφ, λr, φr)`, `nothing` where
# unwanted; zero on inactive cells.
function _local_stress!(λλ, φφ, rr, λφ, λr, φr, XX, YY, ZZ, XY, XZ, YZ, X̄, Ȳ, Z̄, frames, b)
    @. XX -= X̄ * X̄
    @. YY -= Ȳ * Ȳ
    @. ZZ -= Z̄ * Z̄
    @. XY -= X̄ * Ȳ
    @. XZ -= X̄ * Z̄
    @. YZ -= Ȳ * Z̄
    FlowGeometries.Operators.tensor_to_local!(λλ, φφ, rr, λφ, λr, φr, XX, YY, ZZ, XY, XZ, YZ, frames; execution = b)
    return nothing
end

# Symmetric SFS tensor contraction S̄_ij τ_ij, the two-dimensional and the six-term three-dimensional sum.
@inline _sfs_contraction(Sxx::T, Sxy::T, Syy::T, τxx::T, τxy::T, τyy::T) where {T<:AbstractFloat} =
    Sxx * τxx + T(2) * Sxy * τxy + Syy * τyy

@inline _sfs_contraction(
    Sxx::T, Sxy::T, Sxz::T, Syy::T, Syz::T, Szz::T, τxx::T, τxy::T, τxz::T, τyy::T, τyz::T, τzz::T,
) where {T<:AbstractFloat} =
    Sxx * τxx + T(2) * Sxy * τxy + Syy * τyy + T(2) * Sxz * τxz + T(2) * Syz * τyz + Szz * τzz

# `T̄_ee`, `T̄_en`, `T̄_nn` of the slice at offset `off` from a flux analysis: the filtered traceless part,
# `T̄_ee − T̄_nn + 2iT̄_en`, then the filtered trace `T̄_ee + T̄_nn` (with `w`, plus `i(w²)‾`).
function _spin_synthesize_flux!(ws::ΠWorkspace, sf::SpinFilter, w, off::Int)
    C = sf.coeffs
    uf, vf, uu, uv, vv = ws.u_filt, ws.v_filt, ws.uu_filt, ws.uv_filt, ws.vv_filt
    _spin_synthesize!(sf, C[1], 1)
    _spin_store!(sf) do k, z
        @inbounds uf[off + k] = real(z)
        @inbounds vf[off + k] = imag(z)
    end
    _spin_synthesize!(sf, C[2], 2)
    _spin_store!(sf) do k, z
        @inbounds uu[off + k] = real(z)
        @inbounds uv[off + k] = imag(z) / 2
    end
    _spin_synthesize!(sf, C[3], 0)
    _spin_trace!(uu, vv, ws.ww_filt, sf, w, off)
    _spin_synthesize_vertical!(ws, sf, w, off)
    return nothing
end

function _spin_trace!(uu, vv, _ww, sf::SpinFilter, ::Nothing, off::Int)
    _spin_store!(sf) do k, z
        @inbounds d = uu[off + k]
        @inbounds uu[off + k] = (real(z) + d) / 2
        @inbounds vv[off + k] = (real(z) - d) / 2
    end
    return nothing
end

function _spin_trace!(uu, vv, ww, sf::SpinFilter, ::AbstractArray, off::Int)
    _spin_store!(sf) do k, z
        @inbounds d = uu[off + k]
        @inbounds uu[off + k] = (real(z) + d) / 2
        @inbounds vv[off + k] = (real(z) - d) / 2
        @inbounds ww[off + k] = imag(z)
    end
    return nothing
end

# The radial row of `(u⊗u)‾`, `(wU)‾` by spin 1, and `w̄` by spin 0 (Aluie 2019, eqs. 150 and 158).
_spin_synthesize_vertical!(_ws, _sf, ::Nothing, _off) = nothing

function _spin_synthesize_vertical!(ws::ΠWorkspace, sf::SpinFilter, ::AbstractArray, off::Int)
    C = sf.coeffs
    uw, vw, wf = ws.uw_filt, ws.vw_filt, ws.w_filt
    _spin_synthesize!(sf, C[4], 1)
    _spin_store!(sf) do k, z
        @inbounds uw[off + k] = real(z)
        @inbounds vw[off + k] = imag(z)
    end
    _spin_synthesize!(sf, C[5], 0)
    _spin_store!(sf) do k, z
        @inbounds wf[off + k] = real(z)
    end
    return nothing
end

# The filtered velocity and products of every slice by the generalized convolution, at `kernel`'s scale
# `scale`, and under the series route each slice's strain; an `analyzed` sweep holds the slice's analysis
# already.
function _spin_flux_fields!(ws::ΠWorkspace, sf::SpinFilter, u, v, w, kernel, scale, analyzed, route)
    _spin_transfer!(sf, kernel, scale)
    n = sf.ncells
    for off in 0:n:(length(u) - n)
        analyzed isa SpinAnalysis || _spin_analyze_flux!(sf, u, v, w, off)
        _spin_synthesize_flux!(ws, sf, w, off)
        _spin_slice_strain!(ws, sf, w, off, route)
    end
    return nothing
end

_spin_slice_strain!(_ws, _sf, _w, _off, _route) = nothing

# The strain of the slice at offset `off` from the filtered velocity's spin-1 coefficients.
function _spin_slice_strain!(ws::ΠWorkspace, sf::SpinFilter, w, off::Int, ::SeriesStrain)
    _spin_strain!(ws.S_xx, ws.S_yy, ws.S_xy, sf, sf.coeffs[1], off)
    _spin_slice_vertical_strain!(ws, sf, w, off)
    return nothing
end

_spin_slice_vertical_strain!(_ws, _sf, ::Nothing, _off) = nothing

# The thin-layer `S_er = ½∂_e w̄`, `S_nr = ½∂_n w̄`, `S_rr = 0` from the vertical velocity's spin-0 coefficients.
function _spin_slice_vertical_strain!(ws::ΠWorkspace{T}, sf::SpinFilter, ::AbstractArray, off::Int) where {T}
    _spin_gradient!(ws.S_xz, ws.S_yz, sf, sf.coeffs[5], off, T(0.5))
    Szz = ws.S_zz
    FlowGeometries.Execution.run_indices(sf.ncells, sf.execution) do k
        @inbounds Szz[off + k] = zero(T)
    end
    return nothing
end

# Fill `ws`'s stress `τ_*` and strain `S_*` on any grid resolving two tangent directions, a node set
# included; the fields may carry trailing batch axes. `deriv_plan` is held where the workspace's loops run.
# A workspace holding the generalized convolution filters the velocity and the stress by it, at
# `kernel`'s scale `scale`, and `plan` may be the mask strategy alone (`_plan_for`).
function _fill_stress_strain!(
    u::AbstractArray,
    v::AbstractArray,
    w::Union{Nothing, AbstractArray},
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    ws::ΠWorkspace,
    plan::_PlanOrStrategy,
    deriv_plan,
    kernel::Kernels.AbstractFilterKernel,
    scale::T,
    analyzed = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    has_w = w !== nothing
    b = ws.at.execution
    og = _output(ws.at, plan)
    sf = _spin_of(ws.spin, plan)

    if sf !== nothing
        _spin_flux_fields!(ws, sf, u, v, w, kernel, scale, analyzed, deriv_plan)
        @. ws.τ_xx = ws.uu_filt - ws.u_filt * ws.u_filt
        @. ws.τ_xy = ws.uv_filt - ws.u_filt * ws.v_filt
        @. ws.τ_yy = ws.vv_filt - ws.v_filt * ws.v_filt
        if has_w
            @. ws.τ_xz = ws.uw_filt - ws.u_filt * ws.w_filt
            @. ws.τ_yz = ws.vw_filt - ws.v_filt * ws.w_filt
            @. ws.τ_zz = ws.ww_filt - ws.w_filt * ws.w_filt
        end
        _strain!((ws.S_xx, ws.S_yy, ws.S_xy), (ws.u_filt, ws.v_filt), og, deriv_plan, (ws.ux, ws.uy), b)
        if has_w
            # The thin-layer strain of the non-Cartesian branch below.
            _grad2!(ws.S_xz, ws.S_yz, ws.w_filt, og, deriv_plan, b)
            @. ws.S_xz = T(0.5) * ws.S_xz
            @. ws.S_yz = T(0.5) * ws.S_yz
            fill!(ws.S_zz, zero(T))
        end
    elseif G <: FlowGeometries.Geometry.CartesianGeometry{T}
        if analyzed !== nothing
            # `analyzed` holds a sweep's scale-independent forward transforms; only the synthesis is
            # per scale.
            _synthesize_all!(_velocity_outs(ws, has_w), analyzed.velocity, plan)
        elseif has_w
            Filtering.filter_apply_batch!((ws.u_filt, ws.v_filt, ws.w_filt), (u, v, w), plan)
        else
            Filtering.filter_apply_batch!((ws.u_filt, ws.v_filt), (u, v), plan)
        end

        # The ambient buffers, idle on a plane, hold the products before filtering.
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

        @. ws.τ_xx = ws.uu_filt - ws.u_filt * ws.u_filt
        @. ws.τ_xy = ws.uv_filt - ws.u_filt * ws.v_filt
        @. ws.τ_yy = ws.vv_filt - ws.v_filt * ws.v_filt
        if has_w
            @. ws.τ_xz = ws.uw_filt - ws.u_filt * ws.w_filt
            @. ws.τ_yz = ws.vw_filt - ws.v_filt * ws.w_filt
            @. ws.τ_zz = ws.ww_filt - ws.w_filt * ws.w_filt
        end

        # `ux`/`uy` are free once the products are filtered.
        _cartesian_strain!(ws, w, og, deriv_plan, b, Val(FlowGeometries.Grids.ncoordinates(grid)))

    else
        _ambient_stress_strain!(u, v, w, grid, ws, plan, deriv_plan, analyzed)
    end

    return nothing
end

# A sphere or a spheroid under the Cartesian-component filter: the ambient Cartesian components filtered
# (Aluie 2019), the stress rotated to the local frame, the strain by `route`. A `SphericalAnalysis` means
# `ws` already holds the components.
function _ambient_stress_strain!(u, v, w, grid, ws::ΠWorkspace, plan, route, analyzed)
    analyzed isa SphericalAnalysis || _to_ambient!(ws, u, v, w)
    F = _filtered_with_spectra!((ws.ux_filt, ws.uy_filt, ws.uz_filt), (ws.ux, ws.uy, ws.uz), plan, route)
    Filtering.filter_apply_batch!(_ambient_product_outs(ws), _ambient_products!(ws), plan)
    _ambient_local!(ws, F, w, grid, plan, route)
    return nothing
end

# The same from a sweep's analyses of the components and their products: each synthesized at this scale.
function _ambient_stress_strain!(_u, _v, w, grid, ws::ΠWorkspace, plan, route, a::NamedTuple{(:velocity, :product)})
    _synthesize_all!((ws.ux_filt, ws.uy_filt, ws.uz_filt), a.velocity, plan)
    _synthesize_all!(_ambient_product_outs(ws), a.product, plan)
    _ambient_local!(ws, a.velocity, w, grid, plan, route)
    return nothing
end

# The six products of the ambient components `ws.ux/uy/uz`, formed in the local-velocity buffers and the three
# scratch arrays, all free until the filtered velocity is rotated back; in the order of `_ambient_product_outs`.
function _ambient_products!(ws::ΠWorkspace)
    @. ws.u_filt = ws.ux * ws.ux
    @. ws.v_filt = ws.ux * ws.uy
    @. ws.w_filt = ws.ux * ws.uz
    @. ws.scratch = ws.uy * ws.uy
    @. ws.scratch2 = ws.uy * ws.uz
    @. ws.scratch3 = ws.uz * ws.uz
    return (ws.u_filt, ws.v_filt, ws.w_filt, ws.scratch, ws.scratch2, ws.scratch3)
end

@inline _ambient_product_outs(ws::ΠWorkspace) = (ws.uu_filt, ws.uv_filt, ws.uw_filt, ws.vv_filt, ws.vw_filt, ws.ww_filt)

# The filtered ambient components and products of `ws` into the local velocity, stress and strain; `F` the
# components' spectra under the series route.
function _ambient_local!(ws::ΠWorkspace, F, w, grid, plan, route)
    b = ws.at.execution
    # A horizontal velocity has an ambient Z component, so all six ambient products enter the local
    # `τ_ee`, `τ_en`, `τ_nn`; without `w` only the radial components drop.
    fr = _output_frames(ws.at, plan)
    has_w = w !== nothing
    FlowGeometries.Operators.to_local!(ws.u_filt, ws.v_filt, ws.w_filt, ws.ux_filt, ws.uy_filt, ws.uz_filt, fr;
                                       execution = b)
    _local_stress!(ws.τ_xx, ws.τ_yy, has_w ? ws.τ_zz : nothing, ws.τ_xy,
                   has_w ? ws.τ_xz : nothing, has_w ? ws.τ_yz : nothing,
                   ws.uu_filt, ws.vv_filt, ws.ww_filt, ws.uv_filt, ws.uw_filt, ws.vw_filt,
                   ws.ux_filt, ws.uy_filt, ws.uz_filt, fr, b)
    _ambient_strain!(ws, F, w, grid, _output(ws.at, plan), fr, route, plan, b)
    return nothing
end

# The strain in the local frame, its turning included: `S_ee = d_e ū_e − v̄_n tanφ/R`,
# `S_en = ½(d_n ū_e + d_e v̄_n + ū_e tanφ/R)`, `S_nn = d_n v̄_n`; with `w`, `S_er = ½(∂ū_e/∂r + ∂_e w̄)` and
# `S_nr = ½(∂v̄_n/∂r + ∂_n w̄)` with `∂/∂r = 0` on flat layers, and `S_rr = 0`.
function _ambient_strain!(ws::ΠWorkspace{T}, _F, w, _grid, og, _fr, deriv_plan, _plan, b) where {T}
    _strain!((ws.S_xx, ws.S_yy, ws.S_xy), (ws.u_filt, ws.v_filt), og, deriv_plan, (ws.scratch, ws.scratch2), b)
    if w !== nothing
        _grad2!(ws.S_xz, ws.S_yz, ws.w_filt, og, deriv_plan, b)
        @. ws.S_xz = T(0.5) * ws.S_xz
        @. ws.S_yz = T(0.5) * ws.S_yz
        fill!(ws.S_zz, zero(T))
    end
    return nothing
end

# The same from the filter's series (`_ambient_series_strain!`), through the strain's own buffers and
# `scratch`: `∂_e w̄ = r̂·∂_e p̄ + ū_e/R` and `∂_n w̄ = r̂·∂_n p̄ + v̄_n/R`. A sweep's analysis holds the ambient
# components across scales, and the stress holds the product buffers.
function _ambient_strain!(ws::ΠWorkspace{T}, F, w, grid, _og, fr, route::SeriesStrain, plan, b) where {T}
    Ge = (ws.S_xx, ws.S_xy, ws.S_xz)
    Gn = (ws.scratch, ws.S_yy, ws.S_yz)
    R = _sphere_radius(grid, T)
    _ambient_series_strain!((ws.S_xx, ws.S_yy, ws.S_xy), Ge, Gn, F, (ws.ux_filt, ws.uy_filt, ws.uz_filt),
                            ws.w_filt, fr, R, route, plan, b)
    if w !== nothing
        @. ws.S_xz = (Ge[3] + ws.u_filt / R) / 2
        @. ws.S_yz = (Gn[3] + ws.v_filt / R) / 2
        fill!(ws.S_zz, zero(T))
    end
    return nothing
end

# The strain of a Cartesian grid's filtered velocity by `deriv_plan`. On a plane a third component is the
# thin-layer vertical velocity, `S_xz = ½∂w̄/∂x`, `S_yz = ½∂w̄/∂y`, `S_zz = 0` with `∂/∂z = 0` on a level stack;
# in a volume every direction is resolved.
function _cartesian_strain!(ws::ΠWorkspace{T}, ::Nothing, og, deriv_plan, b, ::Val{2}) where {T}
    _strain!((ws.S_xx, ws.S_yy, ws.S_xy), (ws.u_filt, ws.v_filt), og, deriv_plan, (ws.ux, ws.uy), b)
    return ws
end

function _cartesian_strain!(ws::ΠWorkspace{T}, ::AbstractArray, og, deriv_plan, b, ::Val{2}) where {T}
    _cartesian_strain!(ws, nothing, og, deriv_plan, b, Val(2))
    _grad2!(ws.S_xz, ws.S_yz, ws.w_filt, og, deriv_plan, b)
    @. ws.S_xz = ws.S_xz / 2
    @. ws.S_yz = ws.S_yz / 2
    fill!(ws.S_zz, zero(T))
    return ws
end

function _cartesian_strain!(ws::ΠWorkspace{T}, ::AbstractArray, og, deriv_plan, b, ::Val{3}) where {T}
    _strain!((ws.S_xx, ws.S_yy, ws.S_zz, ws.S_xy, ws.S_xz, ws.S_yz),
             (ws.u_filt, ws.v_filt, ws.w_filt), og, deriv_plan, (ws.scratch, ws.ux, ws.uy), b)
    return ws
end

# `plan`'s series form where the caller filters by the spectral method, `NoSeries` in real space whatever
# engine evaluates it: Auto picks an engine by cost, and the engine changes the cost alone. The generalized
# convolution multiplies every degree by `Ĝ(n)`, a series.
@inline _spectral_form(plan::Filtering.AbstractFilterPlan) = _spectral_form(Filtering.plan_method(plan), plan)
@inline _spectral_form(::Filtering.AbstractMaskStrategy) = Filtering.NoSeries()
@inline _spectral_form(::SpinFilter) = Filtering.Series()
@inline _spectral_form(::Filtering.Spectral, plan) = Filtering.series_form(plan)
@inline _spectral_form(::Filtering.RealSpace, _) = Filtering.NoSeries()

Filtering.series_form(::SpinFilter) = Filtering.Series()

# What a diagnostic's derivative route reads: the workspace's generalized convolution where it filters the
# fields, the plan or mask strategy otherwise.
@inline _route_source(::Nothing, plan) = plan
@inline _route_source(sf::SpinFilter, plan) = _spin_of(sf, plan)

# The stress and the strain from the filter's series on a sphere: by the generalized convolution, `ð̄` and `ð` of
# the filtered velocity (`_spin_slice_strain!`); by the Cartesian components, each one's surface gradient.
function _fill_stress_strain!(
    u::AbstractArray,
    v::AbstractArray,
    w::Union{Nothing, AbstractArray},
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    ws::ΠWorkspace,
    plan::_PlanOrStrategy,
    route::SeriesStrain,
    kernel::Kernels.AbstractFilterKernel,
    scale::T,
    analyzed = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}}
    _series_sphere!(_spin_of(ws.spin, plan), u, v, w, grid, ws, plan, route, kernel, scale, analyzed)
    return nothing
end

@inline _series_sphere!(::Nothing, u, v, w, grid, ws, plan, route, _kernel, _scale, analyzed) =
    _ambient_stress_strain!(u, v, w, grid, ws, plan, route, analyzed)

function _series_sphere!(sf::SpinFilter, u, v, w, _grid, ws::ΠWorkspace, _plan, route, kernel, scale, analyzed)
    _spin_flux_fields!(ws, sf, u, v, w, kernel, scale, analyzed, route)
    @. ws.τ_xx = ws.uu_filt - ws.u_filt * ws.u_filt
    @. ws.τ_xy = ws.uv_filt - ws.u_filt * ws.v_filt
    @. ws.τ_yy = ws.vv_filt - ws.v_filt * ws.v_filt
    if w !== nothing
        @. ws.τ_xz = ws.uw_filt - ws.u_filt * ws.w_filt
        @. ws.τ_yz = ws.vw_filt - ws.v_filt * ws.w_filt
        @. ws.τ_zz = ws.ww_filt - ws.w_filt * ws.w_filt
    end
    return nothing
end

"""
    _derivative_route(derivative, deriv_plan, plan, grid) -> derivative plan or SeriesStrain()

How a diagnostic on `grid` differentiates the fields `plan` filters: a derivative plan of
[`output_grid`](@ref), or [`SeriesStrain`](@ref). A given `deriv_plan` stands for
[`Derivatives.PhysicalDerivative`](@ref); `SpectralDerivative()` with one, or on a plan with no series
form, raises.
"""
@inline _derivative_route(::Derivatives.AutoDerivative, ::Nothing, plan, grid) = _auto_route(_spectral_form(plan), plan, grid)
@inline _derivative_route(::Derivatives.AutoDerivative, deriv_plan, _, _) = deriv_plan
@inline _derivative_route(d::Derivatives.PhysicalDerivative, ::Nothing, plan, grid) =
    Derivatives.derivative_plan(output_grid(grid, plan); nodes = d.nodes)
@inline _derivative_route(::Derivatives.PhysicalDerivative, deriv_plan, _, _) = deriv_plan
@inline _derivative_route(::Derivatives.SpectralDerivative, ::Nothing, plan, _) = _series_route(Filtering.series_form(plan), plan)
@inline _derivative_route(::Derivatives.SpectralDerivative, _, _, _) = _spectral_with_plan()

@inline _auto_route(::Filtering.NoSeries, plan, grid) = Derivatives.derivative_plan(output_grid(grid, plan))
@inline _auto_route(::Filtering.AbstractSeriesForm, plan, _) = SeriesStrain(Filtering.series_gradient(plan))
@inline _series_route(::Filtering.AbstractSeriesForm, plan) = SeriesStrain(Filtering.series_gradient(plan))
@inline _series_route(::Filtering.NoSeries, plan) = _no_series(plan)

@noinline _spectral_with_plan() = throw(ArgumentError(
    "derivative = SpectralDerivative() differentiates the filter's own series, so a `deriv_plan` given with " *
    "it is never read: pass one or the other",
))
@noinline _no_series(plan) = throw(ArgumentError(
    "derivative = SpectralDerivative() needs a filter whose output is a series, and " *
    "$(nameof(typeof(plan))) has none: filter with `method = Spectral()` (or the NUFFT kernel sum over a " *
    "node set), or pass `derivative = PhysicalDerivative()`",
))

# The stress and strain of a Cartesian grid's fields filtered by a series plan: the velocity and its
# products analyzed once, the filtered fields synthesized from those spectra, and the strain the
# derivative of the filter's own output.
function _fill_stress_strain!(
    u::AbstractArray,
    v::AbstractArray,
    w::Union{Nothing, AbstractArray},
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    ws::ΠWorkspace,
    plan::Filtering.AbstractFilterPlan,
    ::SeriesStrain,
    kernel::Kernels.AbstractFilterKernel,
    scale::T,
    analyzed = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    has_w = w !== nothing
    a = analyzed === nothing ? analyze_sweep(u, v, w, grid, ws, plan) : analyzed
    vel = _velocity_outs(ws, has_w)
    _synthesize_all!(vel, a.velocity, plan)
    _synthesize_all!(_product_outs(ws, has_w), a.product, plan)
    @. ws.τ_xx = ws.uu_filt - ws.u_filt * ws.u_filt
    @. ws.τ_xy = ws.uv_filt - ws.u_filt * ws.v_filt
    @. ws.τ_yy = ws.vv_filt - ws.v_filt * ws.v_filt
    if has_w
        @. ws.τ_xz = ws.uw_filt - ws.u_filt * ws.w_filt
        @. ws.τ_yz = ws.vw_filt - ws.v_filt * ws.w_filt
        @. ws.τ_zz = ws.ww_filt - ws.w_filt * ws.w_filt
    end
    # The products are analyzed, so the ambient buffers are free for the cross derivatives.
    _series_strain!(ws, a.velocity, vel, plan, Val(FlowGeometries.Grids.ncoordinates(grid)))
    return nothing
end

@inline _series_derivative!(out, F̂, ū, plan, d::Int) = Filtering.filter_synthesize_derivative!(out, F̂, ū, plan, d)

# `S = ½(∂_j ū_i + ∂_i ū_j)` from the spectra and filtered fields of components `i` and `j`, through the
# scratch `a` and `b`.
function _series_shear!(S, (ûi, ūi), (ûj, ūj), i::Int, j::Int, plan, a, b)
    _series_derivative!(a, ûi, ūi, plan, j)
    _series_derivative!(b, ûj, ūj, plan, i)
    @. S = (a + b) / 2
    return S
end

# The filtered fields `outs` of `ins`, and under the series route the spectra they come from, one `nothing`
# per field otherwise.
function _filtered_with_spectra!(outs::Tuple, ins::Tuple, plan, _route)
    Filtering.filter_apply_batch!(outs, ins, plan)
    return map(_ -> nothing, ins)
end

function _filtered_with_spectra!(outs::Tuple, ins::Tuple, plan, ::SeriesStrain)
    F = map(f -> Filtering.filter_analyze!(Filtering.analyze_buffer(plan, f), f, plan), ins)
    _synthesize_all!(outs, F, plan)
    return F
end

# Both tangent components of the gradient of the filtered field `f̄`, whose field's spectrum is `F`:
# differenced by `dplan`, or from the filter's series.
@inline _grad2_route!(g1, g2, f̄, _F, og, dplan, b, _plan) = _grad2!(g1, g2, f̄, og, dplan, b)
@inline function _grad2_route!(g1, g2, f̄, F, _og, route::SeriesStrain, _b, plan)
    Filtering.filter_synthesize_gradient!(g1, g2, F, f̄, plan, route.gradient)
    return nothing
end

# The strain `S = (S_ee, S_nn, S_en)` of the tangent part of the filtered ambient Cartesian vector `p̄`, whose
# fields' spectra are `F`, from the filter's series: `S_ab = ½(e_a·∂_b p̄ + e_b·∂_a p̄) − (w̄/R)δ_ab` with `w̄ = r̂·p̄`
# on the sphere of radius `R`, the frame's turning included. `Ge` and `Gn` receive `(e_e, e_n, r̂)·∂_e p̄` and
# `(e_e, e_n, r̂)·∂_n p̄`.
function _ambient_series_strain!(S::NTuple{3,Any}, Ge::NTuple{3,Any}, Gn::NTuple{3,Any}, F::NTuple{3,Any},
                                 p̄::NTuple{3,Any}, w̄, frames, R, route::SeriesStrain, plan, b)
    foreach((ge, gn, f, x) -> Filtering.filter_synthesize_gradient!(ge, gn, f, x, plan, route.gradient), Ge, Gn, F, p̄)
    FlowGeometries.Operators.to_local!(Ge[1], Ge[2], Ge[3], Ge[1], Ge[2], Ge[3], frames; execution = b)
    FlowGeometries.Operators.to_local!(Gn[1], Gn[2], Gn[3], Gn[1], Gn[2], Gn[3], frames; execution = b)
    @. S[1] = Ge[1] - w̄ / R
    @. S[2] = Gn[2] - w̄ / R
    @. S[3] = (Ge[2] + Gn[1]) / 2
    return nothing
end

@inline _sphere_radius(grid, ::Type{T}) where {T} =
    T(FlowGeometries.Geometry.radius(FlowGeometries.Grids.grid_geometry(grid)))

# The gradient of the filtered scalar `f̄` the generalized convolution `sf` has just synthesized from its
# spin-0 coefficients `sf.coeffs[1]`: differenced by `dplan`, or from those coefficients.
@inline _spin_grad2!(ge, gn, f̄, _sf, og, dplan, b) = _grad2!(ge, gn, f̄, og, dplan, b)
@inline _spin_grad2!(ge, gn, _f̄, sf::SpinFilter, _og, ::SeriesStrain, _b) = _spin_gradient!(ge, gn, sf, sf.coeffs[1], 0)

# The strain `S = (S_ee, S_nn, S_en)` of the filtered tangent vector `(ā, b̄)` the generalized convolution `sf`
# has just synthesized from its spin-1 coefficients `sf.coeffs[1]`: differenced by `dplan` through `scratch`,
# or from those coefficients.
@inline _spin_strain2!(S::NTuple{3,Any}, ā, b̄, _sf, og, dplan, scratch, b) = _strain!(S, (ā, b̄), og, dplan, scratch, b)
@inline _spin_strain2!(S::NTuple{3,Any}, _ā, _b̄, sf::SpinFilter, _og, ::SeriesStrain, _scratch, _b) =
    _spin_strain!(S[1], S[2], S[3], sf, sf.coeffs[1], 0)

# The three components on a structured volume.
@inline _grad3_route!(g::NTuple{3,AbstractArray}, f̄, _F, og, dplan, b, _plan) = _grad3!(g, f̄, og, dplan, b)
@inline function _grad3_route!(g::NTuple{3,AbstractArray}, f̄, F, _og, ::SeriesStrain, _b, plan)
    _series_derivative!(g[1], F, f̄, plan, 1)
    _series_derivative!(g[2], F, f̄, plan, 2)
    _series_derivative!(g[3], F, f̄, plan, 3)
    return nothing
end

# `S_xx = ∂_x ā`, `S_yy = ∂_y b̄` and `S_xy = ½(∂_y ā + ∂_x b̄)` of a filtered pair `a = (Â, ā)`, `b = (B̂, b̄)`,
# through the scratch `s1` and `s2`.
function _series_pair!(Sxx, Syy, Sxy, a::Tuple{Any,Any}, b::Tuple{Any,Any}, plan, s1, s2)
    _series_derivative!(Sxx, a[1], a[2], plan, 1)
    _series_derivative!(Syy, b[1], b[2], plan, 2)
    _series_shear!(Sxy, a, b, 1, 2, plan, s1, s2)
    return Sxx
end

function _series_strain!(ws::ΠWorkspace, û::Tuple{Any,Any}, ū::Tuple{Any,Any}, plan, ::Val{2})
    _series_pair!(ws.S_xx, ws.S_yy, ws.S_xy, (û[1], ū[1]), (û[2], ū[2]), plan, ws.ux, ws.uy)
    return ws
end

# The thin-layer strain of a plane, as the differenced branch takes it: `∂/∂z = 0`.
function _series_strain!(ws::ΠWorkspace{T}, û::Tuple{Any,Any,Any}, ū::Tuple{Any,Any,Any}, plan, ::Val{2}) where {T}
    _series_strain!(ws, (û[1], û[2]), (ū[1], ū[2]), plan, Val(2))
    _series_derivative!(ws.S_xz, û[3], ū[3], plan, 1)
    _series_derivative!(ws.S_yz, û[3], ū[3], plan, 2)
    @. ws.S_xz = ws.S_xz / 2
    @. ws.S_yz = ws.S_yz / 2
    fill!(ws.S_zz, zero(T))
    return ws
end

function _series_strain!(ws::ΠWorkspace, û::Tuple{Any,Any,Any}, ū::Tuple{Any,Any,Any}, plan, ::Val{3})
    _series_derivative!(ws.S_xx, û[1], ū[1], plan, 1)
    _series_derivative!(ws.S_yy, û[2], ū[2], plan, 2)
    _series_derivative!(ws.S_zz, û[3], ū[3], plan, 3)
    _series_shear!(ws.S_xy, (û[1], ū[1]), (û[2], ū[2]), 1, 2, plan, ws.ux, ws.uy)
    _series_shear!(ws.S_xz, (û[1], ū[1]), (û[3], ū[3]), 1, 3, plan, ws.ux, ws.uy)
    _series_shear!(ws.S_yz, (û[2], ū[2]), (û[3], ū[3]), 2, 3, plan, ws.ux, ws.uy)
    return ws
end

# A flux contracts the two tangent components of a surface or plane; a Cartesian volume's three, given `w`.
@inline _require_flux_directions(grid, w) = _require_flux_directions(grid, w, Val(FlowGeometries.Grids.ncoordinates(grid)))
@inline _require_flux_directions(_, _, ::Val{2}) = nothing
@inline function _require_flux_directions(grid::FlowGeometries.Grids.AbstractGrid{G}, ::AbstractArray, ::Val{3}) where {G}
    G <: FlowGeometries.Geometry.CartesianGeometry || _require_tangent_pair(grid, "compute_Π!")
    return nothing
end
@inline _require_flux_directions(grid, _, ::Val) = _require_tangent_pair(grid, "compute_Π!")

# Π = −S̄:τ̄ from a workspace `_fill_stress_strain!` has filled, zero on inactive cells of `grid`, which
# is held where the workspace's arrays are.
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
    plan::_PlanOrStrategy,
    deriv_plan,
    kernel::Kernels.AbstractFilterKernel,
    scale::T,
    analyzed = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    _check_placement(ws.at, plan)
    _fill_stress_strain!(u, v, w, grid, ws, plan, _placed(ws.at, deriv_plan), kernel, scale, analyzed)
    return _contract_Π!(Π, ws, _output(ws.at, plan), w !== nothing)
end

"""
    _flux_setup(grid, Π, w, workspace, filter_plan, build; execution, mask_strategy, vector_filter, transform)
        -> (workspace, plan)

The workspace a flux runs on, `workspace` or one built for the vector filter `vector_filter` resolves to
and the batch shape of `Π`, and the plan it reads ([`_plan_for`](@ref)); `build()` builds a plan.
"""
function _flux_setup(
    grid, Π, w, workspace, filter_plan, build::F;
    execution, mask_strategy, vector_filter, transform,
) where {F}
    strategy = filter_plan === nothing ? mask_strategy : Filtering.plan_strategy(filter_plan)
    workspace === nothing || return workspace, _plan_for(workspace.spin, filter_plan, strategy, build)
    vf = Filtering.resolve_vector_filter(grid, strategy, vector_filter)
    plan = _setup_plan(filter_plan, vf, build)
    ws = ΠWorkspace(grid, _batch_dims(Π, grid), w; vector_filter = vf, transform = transform,
                    execution = _plan_backend(plan, execution))
    return ws, _plan_for(ws.spin, plan, strategy, build)
end

# The plan a new workspace is placed by: the caller's, or one `build()` makes unless the vector filter
# filters through its own transforms.
@inline _setup_plan(plan, _, _) = plan
@inline _setup_plan(::Nothing, ::Filtering.GeneralizedConvolution, _) = nothing
@inline _setup_plan(::Nothing, _, build::F) where {F} = build()

# The backend a plan, or a sweep's plans, run on; `execution` where there is none.
@inline _plan_backend(::Nothing, execution) = execution
@inline _plan_backend(plan::Filtering.AbstractFilterPlan, _) = Filtering.plan_execution(plan)
@inline _plan_backend(plans, execution) = isempty(plans) ? execution : Filtering.plan_execution(first(plans))

"""
    compute_Π!(Π, u, v, w, grid, kernel, scale; workspace=nothing, deriv_plan=nothing, filter_plan=nothing, execution=AutoBackend(), mask_strategy=ZeroFill(), method=RealSpace(), vector_filter=nothing, transform=AutoSpectralBackend(), derivative=AutoDerivative(), analyzed=nothing)

Cross-scale kinetic energy flux Π = -S̄_ij τ_ij on a grid with no structured axes — a
`FlowGeometries.Grids.CurvilinearGrid`, an `UnstructuredGrid` or a sphere pixelization — with the
physics of the `StructuredGrid` 2-D method, its vector filter on a sphere included. The grid resolves two
tangent directions, or on a Cartesian node set given `w` three. `derivative`
([`Derivatives.AbstractDerivative`](@ref)) takes the strain: by default the derivative of the filter's
own series where its output is one (`method = Spectral()`, and the generalized convolution on a sphere) and
the least-squares gradient of
[`Derivatives.derivative_plan`](@ref) otherwise. Fields may carry trailing batch axes past the grid's own.
`RealSpace()` applies the kernel as written, with compact support; a transform's support is global. A
prebuilt `deriv_plan = Derivatives.derivative_plan(output_grid(grid, mask_strategy))` (see
[`output_grid`](@ref)) and a reusable `workspace` serve every call of a scale sweep.
"""
function compute_Π!(
    Π::AbstractArray{T},
    u::AbstractArray,
    v::AbstractArray,
    w::Union{Nothing, AbstractArray},
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    workspace::Union{Nothing, ΠWorkspace} = nothing,
    deriv_plan::_DerivRoute{<:FlowGeometries.Operators.GradientPlan} = nothing,
    filter_plan::Union{Nothing, Filtering.AbstractFilterPlan} = nothing,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Filtering.AbstractFilterMethod = Filtering.RealSpace(),
    vector_filter::Union{Nothing, Filtering.AbstractVectorFilter} = nothing,
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    derivative::Derivatives.AbstractDerivative = Derivatives.AutoDerivative(),
    analyzed = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    _validate_field_sizes(grid, Π, u, v, w)
    _require_flux_directions(grid, w)
    ws, plan = _flux_setup(
        grid, Π, w, workspace, filter_plan,
        () -> Filtering.plan_filter(grid, kernel, scale; mask_strategy, execution, method, transform);
        execution, mask_strategy, vector_filter, transform,
    )
    _check_workspace_w(ws, w)
    route = _derivative_route(derivative, deriv_plan, _route_source(ws.spin, plan), grid)
    return _compute_Π!(Π, u, v, w, grid, ws, plan, route, kernel, scale, analyzed)
end

"""
    compute_Π!(Π::AbstractArray{T,3}, u, v, w, grid::StructuredGrid{T,Cartesian,3}, kernel, scale; workspace=nothing, filter_plan=nothing, deriv_plan=nothing, execution=AutoBackend(), mask_strategy=ZeroFill(), method=RealSpace())

Cross-scale energy flux Π = -S̄_ij τ_ij in a Cartesian volume, every strain component from derivatives
along all three axes and the mask applied per cell:
`S̄:τ = S_xx τ_xx + S_yy τ_yy + S_zz τ_zz + 2(S_xy τ_xy + S_xz τ_xz + S_yz τ_yz)`. A reusable
`workspace`, a [`ΠWorkspace`](@ref), serves every call.
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
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Filtering.AbstractFilterMethod = Filtering.RealSpace(),
    deriv_plan::_DerivRoute{<:FlowGeometries.Operators.StencilPlans} = nothing,
    derivative::Derivatives.AbstractDerivative = Derivatives.AutoDerivative(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    _validate_field_sizes(grid, Π, u, v, w)
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy, execution, method) : filter_plan
    ws = workspace === nothing ? ΠWorkspace(grid, (), w; execution = Filtering.plan_execution(plan)) : workspace
    _check_workspace_w(ws, w)
    _check_placement(ws.at, plan)
    route = _placed(ws.at, _derivative_route(derivative, deriv_plan, plan, grid))
    return _volume_flux!(Π, u, v, w, grid, ws, plan, route, kernel, scale)
end

function _volume_flux!(Π, u, v, w, grid, ws::ΠWorkspace, plan, route::SeriesStrain, kernel, scale)
    _fill_stress_strain!(u, v, w, grid, ws, plan, route, kernel, scale)
    return _contract_Π!(Π, ws, _output(ws.at, plan), true)
end

function _volume_flux!(Π, u, v, w, _grid, ws::ΠWorkspace, plan, dplan, _kernel, _scale)
    b = ws.at.execution
    og = _output(ws.at, plan)

    # The ambient buffers, idle on a Cartesian grid, hold the products before filtering.
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

    @. ws.τ_xx = ws.uu_filt - ws.u_filt * ws.u_filt
    @. ws.τ_xy = ws.uv_filt - ws.u_filt * ws.v_filt
    @. ws.τ_xz = ws.uw_filt - ws.u_filt * ws.w_filt
    @. ws.τ_yy = ws.vv_filt - ws.v_filt * ws.v_filt
    @. ws.τ_yz = ws.vw_filt - ws.v_filt * ws.w_filt
    @. ws.τ_zz = ws.ww_filt - ws.w_filt * ws.w_filt

    # The product buffers are free once filtered.
    _strain!((ws.S_xx, ws.S_yy, ws.S_zz, ws.S_xy, ws.S_xz, ws.S_yz),
             (ws.u_filt, ws.v_filt, ws.w_filt), og, dplan, (ws.scratch, ws.ux, ws.uy), b)
    return _contract_Π!(Π, ws, og, true)
end

"""
    compute_Π!(Π::AbstractArray{T,3}, u, v, w, grid::StructuredGrid{T,Spherical,3}, kernel, scale; workspace=nothing, filter_plan=nothing, deriv_plan=nothing, execution=AutoBackend(), mask_strategy=ZeroFill(), method=RealSpace())

Cross-scale energy flux Π = -S̄_ij τ_ij in a spherical shell, the grid's third axis the radius `r` from
the sphere's centre. The velocity is filtered in ambient Cartesian components (Aluie 2019) and rotated
back to the local (east, north, up) frame, where, with scale factors `h_λ = r cos φ`, `h_φ = r`,
`h_r = 1`,

    S_ee = d_λū_e − v̄_n tan φ/r + w̄_r/r
    S_nn = d_φv̄_n + w̄_r/r
    S_rr = ∂w̄_r/∂r
    S_en = ½(d_λv̄_n + d_φū_e + ū_e tan φ/r)
    S_er = ½(d_λw̄_r + ∂ū_e/∂r − ū_e/r)
    S_nr = ½(d_φw̄_r + ∂v̄_n/∂r − v̄_n/r)

with `d_λ = (1/(r cos φ))∂/∂λ` and `d_φ = (1/r)∂/∂φ` at each cell's own radius. A cell on a pole, where
`cos φ = 0`, is zero. A reusable `workspace` serves every call.
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
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Filtering.AbstractFilterMethod = Filtering.RealSpace(),
    deriv_plan::_DerivRoute{<:FlowGeometries.Operators.StencilPlans} = nothing,
    derivative::Derivatives.AbstractDerivative = Derivatives.AutoDerivative(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.SphericalGeometry{T}}
    _validate_field_sizes(grid, Π, u, v, w)
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy, execution, method) : filter_plan
    ws = workspace === nothing ? ΠWorkspace(grid, (), w; execution = Filtering.plan_execution(plan)) : workspace
    _check_workspace_w(ws, w)
    _check_placement(ws.at, plan)
    b = ws.at.execution
    og = _output(ws.at, plan)
    dplan = _placed(ws.at, _shell_route(_derivative_route(derivative, deriv_plan, plan, grid)))

    _to_ambient!(ws, u, v, w)
    Filtering.filter_apply_batch!((ws.ux_filt, ws.uy_filt, ws.uz_filt), (ws.ux, ws.uy, ws.uz), plan)
    Filtering.filter_apply_batch!(_ambient_product_outs(ws), _ambient_products!(ws), plan)

    fr = _output_frames(ws.at, plan)
    FlowGeometries.Operators.to_local!(ws.u_filt, ws.v_filt, ws.w_filt, ws.ux_filt, ws.uy_filt, ws.uz_filt, fr;
                                       execution = b)
    _local_stress!(ws.τ_xx, ws.τ_yy, ws.τ_zz, ws.τ_xy, ws.τ_xz, ws.τ_yz,
                   ws.uu_filt, ws.vv_filt, ws.ww_filt, ws.uv_filt, ws.uw_filt, ws.vw_filt,
                   ws.ux_filt, ws.uy_filt, ws.uz_filt, fr, b)
    _strain!((ws.S_xx, ws.S_yy, ws.S_zz, ws.S_xy, ws.S_xz, ws.S_yz),
             (ws.u_filt, ws.v_filt, ws.w_filt), og, dplan, (ws.scratch, ws.scratch2, ws.scratch3), b)
    return _contract_Π!(Π, ws, og, true)
end

# A spherical shell's filters, real space or the ambient components' scalar transforms, have no series form.
@inline _shell_route(dplan) = dplan
@noinline _shell_route(::SeriesStrain) = throw(ArgumentError(
    "a spherical shell's flux differences its filtered fields: pass `derivative = PhysicalDerivative()`"))

"""
    compute_Π!(Π::AbstractVector, u, grid::StructuredGrid{T,Cartesian,1}, kernel, scale; workspace=nothing, filter_plan=nothing, deriv_plan=nothing, execution=AutoBackend(), mask_strategy=ZeroFill(), method=RealSpace(), derivative=AutoDerivative())

Cross-scale energy flux Π = -S̄_xx τ_xx on a one-dimensional grid, from its single velocity component `u`,
`derivative` taking `S̄_xx` as on the plane.
"""
function compute_Π!(
    Π::AbstractVector{T},
    u::AbstractVector,
    grid::FlowGeometries.Grids.StructuredGrid{T,G,1},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    workspace::Union{Nothing, ΠWorkspace} = nothing,
    filter_plan::Union{Nothing, Filtering.AbstractFilterPlan} = nothing,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Filtering.AbstractFilterMethod = Filtering.RealSpace(),
    deriv_plan::_DerivRoute{<:FlowGeometries.Operators.StencilPlans} = nothing,
    derivative::Derivatives.AbstractDerivative = Derivatives.AutoDerivative(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    _validate_field_sizes(grid, Π, u)
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy, execution, method) : filter_plan
    ws = workspace === nothing ? ΠWorkspace(grid; execution = Filtering.plan_execution(plan)) : workspace
    _check_placement(ws.at, plan)
    og = _output(ws.at, plan)
    route = _placed(ws.at, _derivative_route(derivative, deriv_plan, plan, grid))

    @. ws.scratch = u * u
    _line_fields!(ws, u, plan, og, route)
    @. ws.τ_xx = ws.uu_filt - ws.u_filt * ws.u_filt

    mask = FlowGeometries.Grids.mask(og)
    @. Π = ifelse(mask, -(ws.S_xx * ws.τ_xx), zero(T))
    return Π
end

# `ū`, `(uu)‾` and `S̄_xx` of a line, `u²` in `ws.scratch`.
function _line_fields!(ws::ΠWorkspace, u, plan, og, dplan)
    Filtering.filter_apply_batch!((ws.u_filt, ws.uu_filt), (u, ws.scratch), plan)
    _dd!(ws.S_xx, ws.u_filt, og, dplan, 1, ws.at.execution)
    return ws
end

function _line_fields!(ws::ΠWorkspace, u, plan, _og, ::SeriesStrain)
    Fu = Filtering.filter_analyze!(Filtering.analyze_buffer(plan, u), u, plan)
    Fuu = Filtering.filter_analyze!(Filtering.analyze_buffer(plan, ws.scratch), ws.scratch, plan)
    Filtering.filter_synthesize!(ws.u_filt, Fu, plan)
    Filtering.filter_synthesize!(ws.uu_filt, Fuu, plan)
    Filtering.filter_synthesize_derivative!(ws.S_xx, Fu, ws.u_filt, plan, 1)
    return ws
end
