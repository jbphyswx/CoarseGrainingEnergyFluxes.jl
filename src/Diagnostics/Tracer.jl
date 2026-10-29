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
    method::Filtering.AbstractFilterMethod = Filtering.RealSpace(),
    derivative::Derivatives.AbstractDerivative = Derivatives.AutoDerivative(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    _require_tangent_pair(grid, "tracer_variance_flux")
    gsz = FlowGeometries.Grids.size_tuple(grid)
    size(u) == gsz || throw(DimensionMismatch("u has size $(size(u)), grid expects $gsz"))
    size(v) == gsz || throw(DimensionMismatch("v has size $(size(v)), grid expects $gsz"))
    size(θ) == gsz || throw(DimensionMismatch("θ has size $(size(θ)), grid expects $gsz"))
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy, execution, method)
    ws = TracerFluxWorkspace(grid; execution = Filtering.plan_execution(plan))
    return tracer_variance_flux!(
        _zeros(ws.at, T, gsz), ws, u, v, θ, grid, kernel, scale;
        filter_plan = plan, derivative = derivative,
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
    derivative::Derivatives.AbstractDerivative = Derivatives.AutoDerivative(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    _require_tangent_pair(grid, "tracer_variance_flux!")
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, execution = execution) :
        filter_plan
    _check_placement(ws.at, plan)
    og = _output(ws.at, plan)
    dplan = _placed(ws.at, _derivative_route(derivative, deriv_plan, plan, grid))

    @. ws.uθ = u * θ
    @. ws.vθ = v * θ
    F = _filtered_with_spectra!((ws.ū, ws.v̄, ws.θ̄, ws.τx, ws.τy), (u, v, θ, ws.uθ, ws.vθ), plan, dplan)

    # Subfilter tracer flux τ_j = ⟨u_j θ⟩ - ū_j θ̄.
    @. ws.τx -= ws.ū * ws.θ̄
    @. ws.τy -= ws.v̄ * ws.θ̄

    # Resolved tracer gradient ∂_j θ̄.
    _grad2_route!(ws.gx, ws.gy, ws.θ̄, F[3], og, dplan, ws.at.execution, plan)

    mask = FlowGeometries.Grids.mask(og)
    @. Πθ = ifelse(mask, -(ws.τx * ws.gx + ws.τy * ws.gy), zero(T))
    return Πθ
end

"""
    tracer_variance_flux(u, v, θ, grid::AbstractGrid{<:SphericalGeometry}, kernel, scale; vector_filter=nothing, transform=AutoSpectralBackend(), ...) -> Πθ

The tracer-variance flux on a sphere, on any grid resolving two tangent directions — structured,
curvilinear, a scattered node set or a sphere pixelization. The velocity and the vector `uθ` of the
subfilter flux `τ_j = ⟨u_j θ⟩ - ū_j θ̄` are filtered by `vector_filter`
([`Filtering.resolve_vector_filter`](@ref)). By default that is the generalized convolution of Aluie
(2019), whose spin transforms `transform` chooses; `θ̄` is then their spin-0 transform with the same
`Ĝ(n)`, so the flux reads one operator, and the filter plan is not built.
`Filtering.CartesianComponents()` forms `τ` in ambient Cartesian components and rotates it to the local
(east, north) frame, where its radial component drops, and the filter plan filters `θ`. `τ` is
contracted against `∂_j θ̄`.
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
    vector_filter::Union{Nothing, Filtering.AbstractVectorFilter} = nothing,
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    derivative::Derivatives.AbstractDerivative = Derivatives.AutoDerivative(),
) where {T<:AbstractFloat, G<:_CurvedSurface{T}}
    _require_tangent_pair(grid, "tracer_variance_flux")
    gsz = FlowGeometries.Grids.size_tuple(grid)
    size(u) == gsz || throw(DimensionMismatch("u has size $(size(u)), grid expects $gsz"))
    size(v) == gsz || throw(DimensionMismatch("v has size $(size(v)), grid expects $gsz"))
    size(θ) == gsz || throw(DimensionMismatch("θ has size $(size(θ)), grid expects $gsz"))
    vf = Filtering.resolve_vector_filter(grid, mask_strategy, vector_filter)
    if vf isa Filtering.GeneralizedConvolution
        ws = SphericalTracerFluxWorkspace(grid; execution = execution, vector_filter = vf, transform = transform)
        return tracer_variance_flux!(_zeros(ws.at, T, gsz), ws, u, v, θ, grid, kernel, scale;
                                     mask_strategy = mask_strategy, derivative = derivative)
    end
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy, execution, transform)
    ws = SphericalTracerFluxWorkspace(grid; execution = Filtering.plan_execution(plan), vector_filter = vf)
    return tracer_variance_flux!(_zeros(ws.at, T, gsz), ws, u, v, θ, grid, kernel, scale;
                                 filter_plan = plan, derivative = derivative)
end

"""
    SphericalTracerFluxWorkspace(grid; execution = SerialBackend(), vector_filter = nothing,
                                 transform = AutoSpectralBackend(), nufft = AutoSpectralBackend())

Scratch for the spherical [`tracer_variance_flux!`](@ref), in the memory of `execution` (see
[`Placement`](@ref)): the ambient Cartesian velocity, its filter, the three velocity–tracer products and
their filter, the filtered tracer, the local subfilter flux pair and the resolved tracer gradient.

On the surface of a sphere the workspace holds the transforms of the vector filter `vector_filter` names,
as [`ΠWorkspace`](@ref) does. Under the generalized convolution, the default, it holds the filtered
tangent velocity in place of the ambient triples.
"""
struct SphericalTracerFluxWorkspace{
    T<:AbstractFloat, A<:AbstractArray{T}, V<:Tuple{Vararg{A}}, B<:Tuple{Vararg{A}}, SF, P<:Placement,
}
    p::V        # ambient Cartesian velocity
    bp::B       # its filter, or the filtered tangent velocity
    pθ::V       # p_i θ, then the filtered product, then τ_i in ambient components
    θ̄::A
    τe::A; τn::A
    gx::A; gy::A
    spin::SF    # the generalized convolution, a `SpinFilter`, or nothing
    at::P
end

function SphericalTracerFluxWorkspace(
    grid::FlowGeometries.Grids.AbstractGrid{G,T};
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
    vector_filter::Union{Nothing,Filtering.AbstractVectorFilter} = nothing,
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    nufft::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
) where {T<:AbstractFloat, G<:_CurvedSurface{T}}
    at = Placement(grid; execution = execution)
    spin = _workspace_spin(grid, at, vector_filter, transform, nufft)
    gsz = FlowGeometries.Grids.size_tuple(grid)
    z() = _zeros(at, T, gsz)
    t(::Val{n}) where {n} = ntuple(_ -> z(), Val(n))
    spin === nothing && return SphericalTracerFluxWorkspace(
        t(Val(3)), t(Val(3)), t(Val(3)), z(), z(), z(), z(), z(), spin, at,
    )
    return SphericalTracerFluxWorkspace((), t(Val(2)), (), z(), z(), z(), z(), z(), spin, at)
end

_tracer_workspace(
    grid::FlowGeometries.Grids.AbstractGrid{<:_CurvedSurface}, execution,
    vector_filter, transform, nufft,
) = SphericalTracerFluxWorkspace(grid; execution, vector_filter, transform, nufft)

@inline _curl_scratch(ws::SphericalTracerFluxWorkspace) = (ws.gx, ws.gy)

"""
    tracer_variance_flux!(Πθ, ws::SphericalTracerFluxWorkspace, u, v, θ, grid, kernel, scale; ...) -> Πθ

In-place spherical [`tracer_variance_flux`](@ref), by the workspace's vector filter. The generalized
convolution reads no filter plan, and none is built for it when `filter_plan` is not given.
"""
function tracer_variance_flux!(
    Πθ::AbstractVecOrMat{T},
    ws::SphericalTracerFluxWorkspace{T},
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
    derivative::Derivatives.AbstractDerivative = Derivatives.AutoDerivative(),
) where {T<:AbstractFloat, G<:_CurvedSurface{T}}
    _require_tangent_pair(grid, "tracer_variance_flux!")
    plan = _plan_for(ws.spin, filter_plan, mask_strategy,
                     () -> Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, execution = execution))
    _check_placement(ws.at, plan)
    b = ws.at.execution
    og = _output(ws.at, plan)
    dplan = _placed(ws.at, _derivative_route(derivative, deriv_plan, _route_source(ws.spin, plan), grid))
    sf = _spin_of(ws.spin, plan)
    if sf === nothing
        Fθ = _cartesian_tracer_flux!(ws, u, v, θ, plan, _output_frames(ws.at, plan), b, dplan)
        _grad2_route!(ws.gx, ws.gy, ws.θ̄, Fθ, og, dplan, b, plan)
    else
        _spin_tracer_flux!(ws, sf, u, v, θ, kernel, scale, og, dplan, b)
    end
    omask = FlowGeometries.Grids.mask(og)
    @. Πθ = ifelse(omask, -(ws.τe * ws.gx + ws.τn * ws.gy), zero(T))
    return Πθ
end

# `θ̄` and the local subfilter flux `(τe, τn)` by the ambient Cartesian components; returns `θ`'s spectrum
# under the series route `route`, `nothing` otherwise.
function _cartesian_tracer_flux!(ws::SphericalTracerFluxWorkspace{T}, u, v, θ, plan, frames, b, route) where {T}
    p, bp, pθ = ws.p, ws.bp, ws.pθ
    _to_ambient!(p, u, v, nothing, ws.at.frames, b)
    mask = FlowGeometries.Grids.mask(ws.at.grid)
    for k in 1:3
        @. pθ[k] = ifelse(mask, p[k] * θ, zero(T))
    end
    Fθ = _filtered_with_spectra!((ws.θ̄,), (θ,), plan, route)
    Filtering.filter_apply_batch!(
        (bp[1], bp[2], bp[3], ws.gx, ws.gy, ws.τe),
        (p[1], p[2], p[3], pθ[1], pθ[2], pθ[3]), plan,
    )
    # `gx`, `gy` and `τe` hold the filtered products until the rotation and the gradient overwrite them.
    @. pθ[1] = ws.gx - bp[1] * ws.θ̄
    @. pθ[2] = ws.gy - bp[2] * ws.θ̄
    @. pθ[3] = ws.τe - bp[3] * ws.θ̄
    _ambient_to_local!((ws.τe, ws.τn), pθ, frames, b)
    return Fθ[1]
end

# The same by the generalized convolution: `θ` at spin 0, `u` and `θu` at spin 1, one `Ĝ(n)` for all; and
# `∇θ̄` by `dplan`.
function _spin_tracer_flux!(ws::SphericalTracerFluxWorkspace, sf::SpinFilter, u, v, θ, kernel, scale, og, dplan, b)
    ū = ws.bp
    _spin_transfer!(sf, kernel, scale)
    _spin_scalar!(ws.θ̄, sf, true, k -> @inbounds(θ[k]))
    _spin_grad2!(ws.gx, ws.gy, ws.θ̄, sf, og, dplan, b)
    _spin_vector!(ū[1], ū[2], u, v, sf, true)
    _spin_filter!(ws.τe, ws.τn, sf, 1, true, k -> @inbounds(θ[k] * complex(u[k], v[k])), 0)
    @. ws.τe -= ū[1] * ws.θ̄
    @. ws.τn -= ū[2] * ws.θ̄
    return nothing
end

"""
    TracerFlux3DWorkspace(grid; execution = SerialBackend())

Scratch for the true-3-D [`tracer_variance_flux!`](@ref), in the memory of `execution` (see
[`Placement`](@ref)): the velocity triple, its filter, the three velocity–tracer products and their
filter, the local subfilter flux, the filtered tracer and its three-component resolved gradient.
"""
struct TracerFlux3DWorkspace{T<:AbstractFloat, A<:AbstractArray{T,3}, P<:Placement}
    p::NTuple{3,A}      # the velocity triple, in ambient Cartesian components on a shell
    bp::NTuple{3,A}     # its filter
    pθ::NTuple{3,A}     # p_i θ, then the filtered product, then τ_i
    loc::NTuple{3,A}    # τ_i in the local frame
    g::NTuple{3,A}      # ∇θ̄
    θ̄::A
    at::P
end

function TracerFlux3DWorkspace(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,3};
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
) where {T<:AbstractFloat, G}
    at = Placement(grid; execution = execution)
    gsz = FlowGeometries.Grids.size_tuple(grid)
    z() = _zeros(at, T, gsz)
    t3() = (z(), z(), z())
    return TracerFlux3DWorkspace(t3(), t3(), t3(), t3(), t3(), z(), at)
end

"""
    tracer_variance_flux!(Πθ, ws::TracerFlux3DWorkspace, u, v, w, θ, grid, kernel, scale; ...) -> Πθ

In-place true-3-D [`tracer_variance_flux`](@ref), on a Cartesian volume or a spherical shell.
"""
function tracer_variance_flux!(
    Πθ::AbstractArray{T,3},
    ws::TracerFlux3DWorkspace{T},
    u::AbstractArray{<:Any,3},
    v::AbstractArray{<:Any,3},
    w::AbstractArray{<:Any,3},
    θ::AbstractArray{<:Any,3},
    grid::FlowGeometries.Grids.StructuredGrid{T,G,3},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    filter_plan::Union{Nothing,Filtering.AbstractFilterPlan} = nothing,
    deriv_plan::Union{Nothing,FlowGeometries.Operators.StencilPlans} = nothing,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    derivative::Derivatives.AbstractDerivative = Derivatives.AutoDerivative(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, execution = execution) :
        filter_plan
    _check_placement(ws.at, plan)
    b = ws.at.execution
    og = _output(ws.at, plan)
    dplan = _placed(ws.at, _derivative_route(derivative, deriv_plan, plan, grid))
    p, bp, pθ = ws.p, ws.bp, ws.pθ

    _fill_tau3_velocity!(p, u, v, w, ws.at.grid, ws.at.frames, b)
    for c in 1:3
        @. pθ[c] = p[c] * θ
    end
    F = _filtered_with_spectra!(
        (bp[1], bp[2], bp[3], ws.θ̄, ws.g[1], ws.g[2], ws.g[3]),
        (p[1], p[2], p[3], θ, pθ[1], pθ[2], pθ[3]), plan, dplan,
    )
    # The filtered products land in the gradient buffers, free until the gradient itself.
    for c in 1:3
        @. pθ[c] = ws.g[c] - bp[c] * ws.θ̄
    end
    _localize_triple!(ws.loc, pθ, og, _output_frames(ws.at, plan), b)

    _grad3_route!(ws.g, ws.θ̄, F[4], og, dplan, b, plan)
    mask = FlowGeometries.Grids.mask(og)
    @. Πθ = ifelse(mask,
        -(ws.loc[1] * ws.g[1] + ws.loc[2] * ws.g[2] + ws.loc[3] * ws.g[3]), zero(T))
    return Πθ
end

"""
    tracer_variance_flux(u, v, w, θ, grid::StructuredGrid{T,G,3}, kernel, scale; execution=AutoBackend(), mask_strategy=ZeroFill())
        -> Πθ

The tracer-variance flux in a volume: the subfilter flux has a vertical component `τ_z = ⟨wθ⟩ - w̄θ̄`,
and the resolved gradient `∂_j θ̄` all three components, the vertical derivative included. On a
spherical shell the velocity is filtered in ambient Cartesian components and the filtered flux rotated
to the local (east, north, up) frame.
"""
function tracer_variance_flux(
    u::AbstractArray{<:Any,3},
    v::AbstractArray{<:Any,3},
    w::AbstractArray{<:Any,3},
    θ::AbstractArray{<:Any,3},
    grid::FlowGeometries.Grids.StructuredGrid{T,G,3},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Filtering.AbstractFilterMethod = Filtering.RealSpace(),
    derivative::Derivatives.AbstractDerivative = Derivatives.AutoDerivative(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    for (nm, a) in (("u", u), ("v", v), ("w", w), ("θ", θ))
        size(a) == gsz || throw(DimensionMismatch("$nm has size $(size(a)), grid expects $gsz"))
    end
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy, execution, method)
    ws = TracerFlux3DWorkspace(grid; execution = Filtering.plan_execution(plan))
    return tracer_variance_flux!(
        _zeros(ws.at, T, gsz), ws, u, v, w, θ, grid, kernel, scale;
        filter_plan = plan, derivative = derivative,
    )
end
