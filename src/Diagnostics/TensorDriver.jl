# ---------------------------------------------------------------------------
# Filtering, frame rotation, subfilter stress, strain and contraction shared by the flux diagnostics.
# `deriv_plan` is `Derivatives.derivative_plan(grid)`; a masked or undefined value is zero. The grids
# these helpers take are a workspace's `Placement` copies, and `b` is the backend its loops run under.
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
@inline _ambient_to_local!(l::NTuple{2}, p::NTuple{3}, grid, b) =
    (FlowGeometries.Operators.to_local!(l[1], l[2], nothing, p[1], p[2], p[3], grid; execution = b); l)
@inline _ambient_to_local!(l::NTuple{3}, p::NTuple{3}, grid, b) =
    (FlowGeometries.Operators.to_local!(l[1], l[2], l[3], p[1], p[2], p[3], grid; execution = b); l)

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

# Whether `og` holds exactly `grid`'s active cells, so a plan built on either serves both.
@inline _same_cells(grid, og) = og === grid || all(FlowGeometries.Grids.mask(grid))

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
function _local_stress!(λλ, φφ, rr, λφ, λr, φr, XX, YY, ZZ, XY, XZ, YZ, X̄, Ȳ, Z̄, grid, b)
    @. XX -= X̄ * X̄
    @. YY -= Ȳ * Ȳ
    @. ZZ -= Z̄ * Z̄
    @. XY -= X̄ * Ȳ
    @. XZ -= X̄ * Z̄
    @. YZ -= Ȳ * Z̄
    FlowGeometries.Operators.tensor_to_local!(λλ, φφ, rr, λφ, λr, φr, XX, YY, ZZ, XY, XZ, YZ, grid; execution = b)
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
# `scale`; an `analyzed` sweep holds the slice's analysis already.
function _spin_flux_fields!(ws::ΠWorkspace, sf::SpinFilter, u, v, w, kernel, scale, analyzed)
    _spin_transfer!(sf, kernel, scale)
    n = sf.ncells
    for off in 0:n:(length(u) - n)
        analyzed isa SpinAnalysis || _spin_analyze_flux!(sf, u, v, w, off)
        _spin_synthesize_flux!(ws, sf, w, off)
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
        _spin_flux_fields!(ws, sf, u, v, w, kernel, scale, analyzed)
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
        _strain!((ws.S_xx, ws.S_yy, ws.S_xy), (ws.u_filt, ws.v_filt), og, deriv_plan, (ws.ux, ws.uy), b)

        if has_w
            # S_xz = ½(∂ū/∂z + ∂w̄/∂x) and S_yz = ½(∂v̄/∂z + ∂w̄/∂y), with ∂/∂z = 0 on a level stack.
            _grad2!(ws.S_xz, ws.S_yz, ws.w_filt, og, deriv_plan, b)
            @. ws.S_xz = T(0.5) * ws.S_xz
            @. ws.S_yz = T(0.5) * ws.S_yz

            # S_zz = ∂w̄/∂z = 0 on a level stack.
            fill!(ws.S_zz, zero(T))
        end

    else
        # A sphere or a spheroid: filter the ambient Cartesian components (Aluie 2019). A
        # `SphericalAnalysis` means `ws` already holds them.
        analyzed isa SphericalAnalysis || _to_ambient!(ws, u, v, w)

        # The products are formed in the local-velocity buffers and the three scratch arrays, all free
        # until the filtered velocity is rotated back.
        Filtering.filter_apply_batch!((ws.ux_filt, ws.uy_filt, ws.uz_filt), (ws.ux, ws.uy, ws.uz), plan)
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

        # A horizontal velocity has an ambient Z component, so all six ambient products enter the local
        # `τ_ee`, `τ_en`, `τ_nn`; without `w` only the radial components drop.
        FlowGeometries.Operators.to_local!(ws.u_filt, ws.v_filt, ws.w_filt, ws.ux_filt, ws.uy_filt, ws.uz_filt, og;
                                           execution = b)
        _local_stress!(ws.τ_xx, ws.τ_yy, has_w ? ws.τ_zz : nothing, ws.τ_xy,
                       has_w ? ws.τ_xz : nothing, has_w ? ws.τ_yz : nothing,
                       ws.uu_filt, ws.vv_filt, ws.ww_filt, ws.uv_filt, ws.uw_filt, ws.vw_filt,
                       ws.ux_filt, ws.uy_filt, ws.uz_filt, og, b)

        # The strain in the local frame, its turning included: `S_ee = d_e ū_e − v̄_n tanφ/R`,
        # `S_en = ½(d_n ū_e + d_e v̄_n + ū_e tanφ/R)`, `S_nn = d_n v̄_n`.
        _strain!((ws.S_xx, ws.S_yy, ws.S_xy), (ws.u_filt, ws.v_filt), og, deriv_plan,
                 (ws.scratch, ws.scratch2), b)

        if has_w
            # S_er = ½(∂ū_e/∂r + 1/(R cosφ) ∂w̄/∂λ) and S_nr = ½(∂v̄_n/∂r + 1/R ∂w̄/∂φ), with ∂/∂r = 0
            # on flat layers.
            _grad2!(ws.S_xz, ws.S_yz, ws.w_filt, og, deriv_plan, b)
            @. ws.S_xz = T(0.5) * ws.S_xz
            @. ws.S_yz = T(0.5) * ws.S_yz

            # S_rr = ∂w̄/∂r = 0 on flat layers.
            fill!(ws.S_zz, zero(T))
        end
    end

    return nothing
end

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
    ws = workspace
    plan = filter_plan
    if ws === nothing
        vf = Filtering.resolve_vector_filter(grid, strategy, vector_filter)
        vf isa Filtering.GeneralizedConvolution || plan !== nothing || (plan = build())
        ws = ΠWorkspace(grid, _batch_dims(Π, grid), Val(w !== nothing); vector_filter = vf, transform = transform,
                        execution = plan === nothing ? execution : Filtering.plan_execution(plan))
    end
    return ws, _plan_for(ws.spin, plan, strategy, build)
end

"""
    compute_Π!(Π, u, v, w, grid, kernel, scale; workspace=nothing, deriv_plan=nothing, filter_plan=nothing, execution=AutoBackend(), mask_strategy=ZeroFill(), method=RealSpace(), vector_filter=nothing, transform=AutoSpectralBackend(), analyzed=nothing)

Cross-scale kinetic energy flux Π = -S̄_ij τ_ij on a grid with no structured axes — a
`FlowGeometries.Grids.CurvilinearGrid`, an `UnstructuredGrid` or a sphere pixelization — with the
physics of the `StructuredGrid` 2-D method, its vector filter on a sphere included. The resolved strain
uses the least-squares gradient of [`Derivatives.derivative_plan`](@ref), which needs a grid resolving two
tangent directions. Fields may carry trailing batch axes past the grid's own. `RealSpace()` applies the
kernel as written, with compact support; a transform's support is global. A prebuilt
`deriv_plan = Derivatives.derivative_plan(output_grid(grid, mask_strategy))` (see [`output_grid`](@ref))
and a reusable `workspace` serve every call of a scale sweep.
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
    deriv_plan::Union{Nothing, FlowGeometries.Operators.GradientPlan} = nothing,
    filter_plan::Union{Nothing, Filtering.AbstractFilterPlan} = nothing,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Filtering.AbstractFilterMethod = Filtering.RealSpace(),
    vector_filter::Union{Nothing, Filtering.AbstractVectorFilter} = nothing,
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    analyzed = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    _require_tangent_pair(grid, "compute_Π!")
    _validate_field_sizes(grid, Π, u, v, w)
    ws, plan = _flux_setup(
        grid, Π, w, workspace, filter_plan,
        () -> Filtering.plan_filter(grid, kernel, scale; mask_strategy, execution, method, transform);
        execution, mask_strategy, vector_filter, transform,
    )
    _check_workspace_w(ws, w)
    dplan = deriv_plan === nothing ? Derivatives.derivative_plan(output_grid(grid, plan)) : deriv_plan
    return _compute_Π!(Π, u, v, w, grid, ws, plan, dplan, kernel, scale, analyzed)
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
    deriv_plan::Union{Nothing, FlowGeometries.Operators.StencilPlans} = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    _validate_field_sizes(grid, Π, u, v, w)
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy, execution, method) : filter_plan
    ws = workspace === nothing ?
        ΠWorkspace(grid, (), Val(w !== nothing); execution = Filtering.plan_execution(plan)) : workspace
    _check_workspace_w(ws, w)
    _check_placement(ws.at, plan)
    b = ws.at.execution
    og = _output(ws.at, plan)
    dplan = _placed(ws.at, deriv_plan === nothing ? Derivatives.derivative_plan(output_grid(grid, plan)) : deriv_plan)

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
    deriv_plan::Union{Nothing, FlowGeometries.Operators.StencilPlans} = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.SphericalGeometry{T}}
    _validate_field_sizes(grid, Π, u, v, w)
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy, execution, method) : filter_plan
    ws = workspace === nothing ?
        ΠWorkspace(grid, (), Val(w !== nothing); execution = Filtering.plan_execution(plan)) : workspace
    _check_workspace_w(ws, w)
    _check_placement(ws.at, plan)
    b = ws.at.execution
    og = _output(ws.at, plan)
    dplan = _placed(ws.at, deriv_plan === nothing ? Derivatives.derivative_plan(output_grid(grid, plan)) : deriv_plan)

    _to_ambient!(ws, u, v, w)
    Filtering.filter_apply_batch!((ws.ux_filt, ws.uy_filt, ws.uz_filt), (ws.ux, ws.uy, ws.uz), plan)
    # The products are formed in the local-velocity buffers and the three scratch arrays, all free until
    # the filtered velocity is rotated back.
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

    FlowGeometries.Operators.to_local!(ws.u_filt, ws.v_filt, ws.w_filt, ws.ux_filt, ws.uy_filt, ws.uz_filt, og;
                                       execution = b)
    _local_stress!(ws.τ_xx, ws.τ_yy, ws.τ_zz, ws.τ_xy, ws.τ_xz, ws.τ_yz,
                   ws.uu_filt, ws.vv_filt, ws.ww_filt, ws.uv_filt, ws.uw_filt, ws.vw_filt,
                   ws.ux_filt, ws.uy_filt, ws.uz_filt, og, b)
    _strain!((ws.S_xx, ws.S_yy, ws.S_zz, ws.S_xy, ws.S_xz, ws.S_yz),
             (ws.u_filt, ws.v_filt, ws.w_filt), og, dplan, (ws.scratch, ws.scratch2, ws.scratch3), b)
    return _contract_Π!(Π, ws, og, true)
end

"""
    compute_Π!(Π::AbstractVector, u, grid::StructuredGrid{T,Cartesian,1}, kernel, scale; workspace=nothing, filter_plan=nothing, deriv_plan=nothing, execution=AutoBackend(), mask_strategy=ZeroFill(), method=RealSpace())

Cross-scale energy flux Π = -S̄_xx τ_xx on a one-dimensional grid, from its single velocity component `u`.
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
    deriv_plan::Union{Nothing, FlowGeometries.Operators.StencilPlans} = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    _validate_field_sizes(grid, Π, u)
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy, execution, method) : filter_plan
    ws = workspace === nothing ? ΠWorkspace(grid; execution = Filtering.plan_execution(plan)) : workspace
    _check_placement(ws.at, plan)
    og = _output(ws.at, plan)
    dplan = _placed(ws.at, deriv_plan === nothing ? Derivatives.derivative_plan(output_grid(grid, plan)) : deriv_plan)

    @. ws.scratch = u * u
    Filtering.filter_apply_batch!((ws.u_filt, ws.uu_filt), (u, ws.scratch), plan)
    @. ws.τ_xx = ws.uu_filt - ws.u_filt * ws.u_filt

    _dd!(ws.S_xx, ws.u_filt, og, dplan, 1, ws.at.execution)

    mask = FlowGeometries.Grids.mask(og)
    @. Π = ifelse(mask, -(ws.S_xx * ws.τ_xx), zero(T))
    return Π
end
