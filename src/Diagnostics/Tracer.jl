# ---------------------------------------------------------------------------
# Cross-scale tracer-variance flux (scalar analog of Π; buoyancy ⇒ APE transfer)
# ---------------------------------------------------------------------------

"""
    tracer_variance_flux(u, v, θ, grid, kernel, scale; backend=AutoBackend(), mask_strategy=ZeroFill())
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
    u::AbstractVecOrMat,
    v::AbstractVecOrMat,
    θ::AbstractVecOrMat,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    _require_tangent_pair(grid, "tracer_variance_flux")
    gsz = FlowGeometries.Grids.size_tuple(grid)
    size(u) == gsz || throw(DimensionMismatch("u has size $(size(u)), grid expects $gsz"))
    size(v) == gsz || throw(DimensionMismatch("v has size $(size(v)), grid expects $gsz"))
    size(θ) == gsz || throw(DimensionMismatch("θ has size $(size(θ)), grid expects $gsz"))
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend)
    dplan = _default_deriv_plan(output_grid(grid, plan))

    return tracer_variance_flux!(
        zeros(T, gsz), TracerFluxWorkspace(grid), u, v, θ, grid, kernel, scale;
        filter_plan = plan, deriv_plan = dplan,
    )
end

"""
    TracerFluxWorkspace(grid)

Scratch for [`tracer_variance_flux!`](@ref): the filtered fields, the two products, the subfilter
flux components and the resolved tracer gradient. Allocated once and reused.
"""
struct TracerFluxWorkspace{T<:AbstractFloat, A<:AbstractArray{T}}
    ū::A; v̄::A; θ̄::A
    uθ::A; vθ::A
    τx::A; τy::A
    gx::A; gy::A
end

function TracerFluxWorkspace(grid::FlowGeometries.Grids.AbstractGrid{G,T}) where {T<:AbstractFloat, G}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    z() = zeros(T, gsz)
    return TracerFluxWorkspace(z(), z(), z(), z(), z(), z(), z(), z(), z())
end

"""
    tracer_variance_flux!(Πθ, ws, u, v, θ, grid, kernel, scale; filter_plan=nothing, deriv_plan=nothing, ...) -> Πθ

In-place [`tracer_variance_flux`](@ref). With `ws`, `filter_plan` and `deriv_plan` all supplied, a
repeated evaluation — over timesteps or scales — allocates nothing.
"""
function tracer_variance_flux!(
    Πθ::AbstractVecOrMat{T},
    ws::TracerFluxWorkspace{T},
    u::AbstractVecOrMat,
    v::AbstractVecOrMat,
    θ::AbstractVecOrMat,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    filter_plan::Union{Nothing,Filtering.AbstractFilterPlan} = nothing,
    deriv_plan::Union{Nothing,AnyDerivPlan} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    _require_tangent_pair(grid, "tracer_variance_flux!")
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend) :
        filter_plan
    og = output_grid(grid, plan)
    dplan = _resolve_deriv_plan(deriv_plan, og)

    @. ws.uθ = u * θ
    @. ws.vθ = v * θ
    Filtering.filter_apply_batch!(
        (ws.ū, ws.v̄, ws.θ̄, ws.τx, ws.τy), (u, v, θ, ws.uθ, ws.vθ), plan,
    )

    # Subfilter tracer flux τ_j = ⟨u_j θ⟩ - ū_j θ̄.
    @. ws.τx -= ws.ū * ws.θ̄
    @. ws.τy -= ws.v̄ * ws.θ̄

    # Resolved tracer gradient ∂_j θ̄.
    _grad2!(ws.gx, ws.gy, ws.θ̄, og, dplan)

    mask = FlowGeometries.Grids.mask(og)
    @. Πθ = ifelse(mask, -(ws.τx * ws.gx + ws.τy * ws.gy), zero(T))
    return Πθ
end
