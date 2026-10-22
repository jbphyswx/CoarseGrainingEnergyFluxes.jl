# ---------------------------------------------------------------------------
# Cross-scale tracer-variance flux (scalar analog of Π; buoyancy ⇒ APE transfer)
# ---------------------------------------------------------------------------

"""
    tracer_variance_flux(u, v, θ, grid, kernel, scale; execution=AutoBackend(), mask_strategy=ZeroFill())
        -> Πθ

Cross-scale flux of the tracer variance ½⟨θ'²⟩ at filter scale ℓ (the scalar analog of the kinetic
energy flux Π; Aluie & Eyink):

    Πθ(x) = -∂_j θ̄ · τ_j(u, θ),   τ_j = ⟨u_j θ⟩ - ū_j θ̄  (the subfilter tracer flux),

with the same sign convention as [`compute_Π!`](@ref): `Πθ > 0` is a forward cascade of tracer
variance toward small scales, `Πθ < 0` an inverse cascade.

With `θ` the buoyancy `b = -g ρ'/ρ₀` this is the cross-scale transfer of buoyancy variance, tied to
available potential energy. This method serves a plane; a sphere and a volume have their own. The
result lives in the memory of `execution`.
"""
function tracer_variance_flux(
    u::AbstractVecOrMat,
    v::AbstractVecOrMat,
    θ::AbstractVecOrMat,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    _require_tangent_pair(grid, "tracer_variance_flux")
    gsz = FlowGeometries.Grids.size_tuple(grid)
    size(u) == gsz || throw(DimensionMismatch("u has size $(size(u)), grid expects $gsz"))
    size(v) == gsz || throw(DimensionMismatch("v has size $(size(v)), grid expects $gsz"))
    size(θ) == gsz || throw(DimensionMismatch("θ has size $(size(θ)), grid expects $gsz"))
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, execution=execution)
    dplan = Derivatives.derivative_plan(output_grid(grid, plan))
    ws = TracerFluxWorkspace(grid; execution = Filtering.plan_execution(plan))
    return tracer_variance_flux!(
        _zeros(ws.at, T, gsz), ws, u, v, θ, grid, kernel, scale;
        filter_plan = plan, deriv_plan = dplan,
    )
end

"""
    TracerFluxWorkspace(grid; execution = SerialBackend())

Scratch for [`tracer_variance_flux!`](@ref), in the memory of `execution` (see [`Placement`](@ref)): the
filtered fields, the two products, the subfilter flux components and the resolved tracer gradient.
Allocated once and reused.
"""
struct TracerFluxWorkspace{T<:AbstractFloat, A<:AbstractArray{T}, P<:Placement}
    ū::A; v̄::A; θ̄::A
    uθ::A; vθ::A
    τx::A; τy::A
    gx::A; gy::A
    at::P
end

function TracerFluxWorkspace(
    grid::FlowGeometries.Grids.AbstractGrid{G,T};
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
) where {T<:AbstractFloat, G}
    at = Placement(grid; execution = execution)
    gsz = FlowGeometries.Grids.size_tuple(grid)
    z() = _zeros(at, T, gsz)
    return TracerFluxWorkspace(z(), z(), z(), z(), z(), z(), z(), z(), z(), at)
end

"""
    tracer_variance_flux!(Πθ, ws, u, v, θ, grid, kernel, scale; filter_plan=nothing, deriv_plan=nothing, execution=AutoBackend(), mask_strategy=ZeroFill()) -> Πθ

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
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    _require_tangent_pair(grid, "tracer_variance_flux!")
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, execution = execution) :
        filter_plan
    _check_placement(ws.at, plan)
    og = _output(ws.at, plan)
    dplan = _placed(ws.at, _resolve_deriv_plan(deriv_plan, output_grid(grid, plan)))

    @. ws.uθ = u * θ
    @. ws.vθ = v * θ
    Filtering.filter_apply_batch!(
        (ws.ū, ws.v̄, ws.θ̄, ws.τx, ws.τy), (u, v, θ, ws.uθ, ws.vθ), plan,
    )

    # Subfilter tracer flux τ_j = ⟨u_j θ⟩ - ū_j θ̄.
    @. ws.τx -= ws.ū * ws.θ̄
    @. ws.τy -= ws.v̄ * ws.θ̄

    # Resolved tracer gradient ∂_j θ̄.
    _grad2!(ws.gx, ws.gy, ws.θ̄, og, dplan, ws.at.execution)

    mask = FlowGeometries.Grids.mask(og)
    @. Πθ = ifelse(mask, -(ws.τx * ws.gx + ws.τy * ws.gy), zero(T))
    return Πθ
end
