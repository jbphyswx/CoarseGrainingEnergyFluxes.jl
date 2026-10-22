# ---------------------------------------------------------------------------
# Enstrophy flux (Rivera, Aluie & Ecke 2014, eq. 16)
# ---------------------------------------------------------------------------

# The curl of `(u, v)` into `ω`, `grid` and `dplan` held where the backend `b` runs; `g` is two
# scratch arrays.
@inline _curl_into!(ω, u, v, grid, dplan, g, b) =
    FlowGeometries.Operators.curl!(ω, (u, v), grid, dplan; masked = zero(eltype(ω)), scratch = g, execution = b)

"""
    vorticity!(ω, u, v, grid[, deriv_plan]; scratch = nothing, execution = SerialBackend()) -> ω

The vertical component of the relative vorticity on a grid resolving two tangent directions —
structured, curvilinear, a scattered node set or a sphere pixelization — zero on masked cells: the curl
in the local frame, `∂v/∂x − ∂u/∂y` on a plane and

    ζ = (1/(R cos φ))[∂v/∂λ − ∂(u cos φ)/∂φ] = ∂v/∂x − ∂u/∂y + u tan φ/R

on a sphere, with `∂/∂x`, `∂/∂y` derivatives per unit distance. A cell on a pole is zero. `scratch` is
two arrays the size of `ω`; with it and `deriv_plan` supplied, a call allocates nothing. The arrays live
in the memory of `execution`, whose loops take the curl; `grid` and `deriv_plan` are moved there.
"""
function vorticity!(
    ω::AbstractVecOrMat{T},
    u::AbstractVecOrMat,
    v::AbstractVecOrMat,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    deriv_plan::Union{Nothing,AnyDerivPlan} = nothing;
    scratch::Union{Nothing,NTuple{2,AbstractVecOrMat{T}}} = nothing,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    _require_tangent_pair(grid, "vorticity!")
    g = scratch === nothing ? (similar(ω), similar(ω)) : scratch
    all(s -> size(s) == size(ω), g) || throw(DimensionMismatch(
        "vorticity! scratch has sizes $(map(size, g)), expected $(size(ω)) each",
    ))
    at = Placement(grid; execution = execution)
    return _curl_into!(ω, u, v, at.grid, _placed(at, _resolve_deriv_plan(deriv_plan, grid)), g, at.execution)
end

"""
    vorticity(u, v, grid[, deriv_plan]; execution = SerialBackend()) -> ω

Allocating [`vorticity!`](@ref), `ω` in the memory of `execution`.
"""
function vorticity(
    u::AbstractVecOrMat, v::AbstractVecOrMat,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    deriv_plan::Union{Nothing,AnyDerivPlan} = nothing;
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    _require_tangent_pair(grid, "vorticity")
    at = Placement(grid; execution = execution)
    gsz = FlowGeometries.Grids.size_tuple(grid)
    ω = _zeros(at, T, gsz)
    return _curl_into!(ω, u, v, at.grid, _placed(at, _resolve_deriv_plan(deriv_plan, grid)),
                       (_zeros(at, T, gsz), _zeros(at, T, gsz)), at.execution)
end

"""
    enstrophy_flux(u, v, grid, kernel, scale; execution=AutoBackend(), mask_strategy=ZeroFill(), vector_filter=nothing, transform=AutoSpectralBackend()) -> Z

Cross-scale enstrophy flux (Rivera, Aluie & Ecke 2014, eq. 16),

```
Z_ℓ = −∂_j ω̄_ℓ · τ_ℓ(u_j, ω) ,   τ_j = (u_j ω)‾ − ū_j ω̄ ,   ω = ∂v/∂x − ∂u/∂y ,
```

the enstrophy analogue of [`compute_Π!`](@ref): positive means enstrophy moving to smaller scales.

# Gauge

This is the deformation (subtracted) form, the gauge `Π` uses: the resolved product `ū_j ω̄` is
subtracted, and the flux is pointwise Galilean-invariant. The unsubtracted `−∂_j ω̄ (u_j ω)‾` differs
from it by a transport divergence; the two share a spatial mean on a homogeneous domain and "differ
qualitatively as well as quantitatively" on an inhomogeneous or masked one (Aluie 2011; Aluie, Hecht &
Vallis 2018).

`Z` is [`tracer_variance_flux`](@ref) with `θ = ω` from [`vorticity!`](@ref); on a sphere
`vector_filter` and `transform` choose how it filters the vector `u ω`.

# References
- Rivera, M. K., Aluie, H., & Ecke, R. E. (2014). The direct enstrophy cascade of two-dimensional
  soap film flows. *Phys. Fluids* 26, 055105. doi:10.1063/1.4873579
"""
function enstrophy_flux(
    u::AbstractVecOrMat,
    v::AbstractVecOrMat,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    vector_filter::Union{Nothing, Filtering.AbstractVectorFilter} = nothing,
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    _require_tangent_pair(grid, "enstrophy_flux")
    ω = vorticity(u, v, grid, Derivatives.derivative_plan(grid); execution = execution)
    vector = G <: FlowGeometries.Geometry.CartesianGeometry ? (;) : (; vector_filter, transform)
    return tracer_variance_flux(u, v, ω, grid, kernel, scale;
                                execution = execution, mask_strategy = mask_strategy, vector...)
end

"""
    EnstrophyFluxWorkspace(grid; execution = SerialBackend(), vector_filter = nothing,
                           transform = AutoSpectralBackend(), nufft = AutoSpectralBackend())

Scratch for [`enstrophy_flux!`](@ref), in the memory of `execution` (see [`Placement`](@ref)): the
vorticity plus the tracer-flux scratch it is fed into, which on a sphere holds the transforms of the
vector filter `vector_filter` names ([`SphericalTracerFluxWorkspace`](@ref)).
"""
struct EnstrophyFluxWorkspace{T<:AbstractFloat, A<:AbstractArray{T}, W}
    ω::A
    tracer::W
end

function EnstrophyFluxWorkspace(
    grid::FlowGeometries.Grids.AbstractGrid{G,T};
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
    vector_filter::Union{Nothing,Filtering.AbstractVectorFilter} = nothing,
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    nufft::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
) where {T<:AbstractFloat, G}
    tracer = _tracer_workspace(grid, execution, vector_filter, transform, nufft)
    return EnstrophyFluxWorkspace(_zeros(tracer.at, T, FlowGeometries.Grids.size_tuple(grid)), tracer)
end

# The tracer-flux workspace of the grid's metric; a plane's filters no vector on a sphere.
_tracer_workspace(
    grid::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.CartesianGeometry}, execution,
    _vector_filter, _transform, _nufft,
) = TracerFluxWorkspace(grid; execution = execution)

# Two buffers of the tracer workspace, free while the curl is taken.
@inline _curl_scratch(ws::TracerFluxWorkspace) = (ws.uθ, ws.vθ)

"""
    enstrophy_flux!(Z, ws, u, v, grid, kernel, scale; filter_plan=nothing, deriv_plan=nothing, curl_plan=nothing, ...) -> Z

In-place [`enstrophy_flux`](@ref). `curl_plan` takes the vorticity of `(u, v)` on `grid`; `deriv_plan`
differences `ω̄` on [`output_grid`](@ref). Where the two grids hold the same cells, `deriv_plan` serves
both. With `ws` and every plan supplied, a repeated evaluation allocates nothing.
"""
function enstrophy_flux!(
    Z::AbstractVecOrMat{T},
    ws::EnstrophyFluxWorkspace{T},
    u::AbstractVecOrMat,
    v::AbstractVecOrMat,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    filter_plan::Union{Nothing,Filtering.AbstractFilterPlan} = nothing,
    deriv_plan::Union{Nothing,AnyDerivPlan} = nothing,
    curl_plan::Union{Nothing,AnyDerivPlan} = nothing,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    _require_tangent_pair(grid, "enstrophy_flux!")
    plan = _plan_for(_tracer_spin(ws.tracer), filter_plan, mask_strategy,
                     () -> Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, execution = execution))
    at = ws.tracer.at
    _check_placement(at, plan)
    og = output_grid(grid, plan)
    dplan = _resolve_deriv_plan(deriv_plan, og)
    _curl_into!(ws.ω, u, v, at.grid, _placed(at, _curl_plan(curl_plan, dplan, grid, og)),
                _curl_scratch(ws.tracer), at.execution)
    return tracer_variance_flux!(Z, ws.tracer, u, v, ws.ω, grid, kernel, scale;
                                 filter_plan = _built_plan(plan), mask_strategy = mask_strategy, deriv_plan = dplan)
end

# A tracer workspace's generalized convolution, `nothing` on a plane.
@inline _tracer_spin(::TracerFluxWorkspace) = nothing
@inline _tracer_spin(ws) = ws.spin

# The plan `_plan_for` gave, as a `filter_plan` keyword: `nothing` for a mask strategy.
@inline _built_plan(plan::Filtering.AbstractFilterPlan) = plan
@inline _built_plan(::Filtering.AbstractMaskStrategy) = nothing

# The vorticity differences the raw velocity over `grid`'s own cells.
@inline _curl_plan(plan::AnyDerivPlan, _dplan, _grid, _og) = plan
_curl_plan(::Nothing, dplan, grid, og) = _same_cells(grid, og) ? dplan : Derivatives.derivative_plan(grid)

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
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.SphericalGeometry{T}}
    _require_tangent_pair(grid, "tracer_variance_flux")
    gsz = FlowGeometries.Grids.size_tuple(grid)
    size(u) == gsz || throw(DimensionMismatch("u has size $(size(u)), grid expects $gsz"))
    size(v) == gsz || throw(DimensionMismatch("v has size $(size(v)), grid expects $gsz"))
    size(θ) == gsz || throw(DimensionMismatch("θ has size $(size(θ)), grid expects $gsz"))
    dplan = Derivatives.derivative_plan(output_grid(grid, mask_strategy))
    vf = Filtering.resolve_vector_filter(grid, mask_strategy, vector_filter)
    if vf isa Filtering.GeneralizedConvolution
        ws = SphericalTracerFluxWorkspace(grid; execution = execution, vector_filter = vf, transform = transform)
        return tracer_variance_flux!(_zeros(ws.at, T, gsz), ws, u, v, θ, grid, kernel, scale;
                                     mask_strategy = mask_strategy, deriv_plan = dplan)
    end
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy, execution, transform)
    ws = SphericalTracerFluxWorkspace(grid; execution = Filtering.plan_execution(plan), vector_filter = vf)
    return tracer_variance_flux!(_zeros(ws.at, T, gsz), ws, u, v, θ, grid, kernel, scale;
                                 filter_plan = plan, deriv_plan = dplan)
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
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}}
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
    grid::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry}, execution,
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
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}}
    _require_tangent_pair(grid, "tracer_variance_flux!")
    plan = _plan_for(ws.spin, filter_plan, mask_strategy,
                     () -> Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, execution = execution))
    _check_placement(ws.at, plan)
    b = ws.at.execution
    og = _output(ws.at, plan)
    dplan = _placed(ws.at, _resolve_deriv_plan(deriv_plan, output_grid(grid, plan)))
    sf = _spin_of(ws.spin, plan)
    if sf === nothing
        _cartesian_tracer_flux!(ws, u, v, θ, plan, og, b)
    else
        _spin_tracer_flux!(ws, sf, u, v, θ, kernel, scale)
    end
    _grad2!(ws.gx, ws.gy, ws.θ̄, og, dplan, b)
    omask = FlowGeometries.Grids.mask(og)
    @. Πθ = ifelse(omask, -(ws.τe * ws.gx + ws.τn * ws.gy), zero(T))
    return Πθ
end

# `θ̄` and the local subfilter flux `(τe, τn)` by the ambient Cartesian components.
function _cartesian_tracer_flux!(ws::SphericalTracerFluxWorkspace{T}, u, v, θ, plan, og, b) where {T}
    p, bp, pθ = ws.p, ws.bp, ws.pθ
    _to_ambient!(p, u, v, nothing, ws.at.grid, b)
    mask = FlowGeometries.Grids.mask(ws.at.grid)
    for k in 1:3
        @. pθ[k] = ifelse(mask, p[k] * θ, zero(T))
    end
    Filtering.filter_apply_batch!(
        (bp[1], bp[2], bp[3], ws.θ̄, ws.gx, ws.gy, ws.τe),
        (p[1], p[2], p[3], θ, pθ[1], pθ[2], pθ[3]), plan,
    )
    # `gx`, `gy` and `τe` hold the filtered products until the rotation and the gradient overwrite them.
    @. pθ[1] = ws.gx - bp[1] * ws.θ̄
    @. pθ[2] = ws.gy - bp[2] * ws.θ̄
    @. pθ[3] = ws.τe - bp[3] * ws.θ̄
    _ambient_to_local!((ws.τe, ws.τn), pθ, og, b)
    return nothing
end

# The same by the generalized convolution: `θ` at spin 0, `u` and `θu` at spin 1, one `Ĝ(n)` for all.
function _spin_tracer_flux!(ws::SphericalTracerFluxWorkspace, sf::SpinFilter, u, v, θ, kernel, scale)
    ū = ws.bp
    _spin_transfer!(sf, kernel, scale)
    _spin_scalar!(ws.θ̄, sf, true, k -> @inbounds(θ[k]))
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
    t(n) = ntuple(_ -> _zeros(at, T, gsz), n)
    return TracerFlux3DWorkspace(t(3), t(3), t(3), t(3), t(3), _zeros(at, T, gsz), at)
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
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, execution = execution) :
        filter_plan
    _check_placement(ws.at, plan)
    b = ws.at.execution
    og = _output(ws.at, plan)
    dplan = _placed(ws.at, deriv_plan === nothing ? Derivatives.derivative_plan(output_grid(grid, plan)) : deriv_plan)
    p, bp, pθ = ws.p, ws.bp, ws.pθ

    _fill_tau3_velocity!(p, u, v, w, ws.at.grid, b)
    for c in 1:3
        @. pθ[c] = p[c] * θ
    end
    Filtering.filter_apply_batch!(
        (bp[1], bp[2], bp[3], ws.θ̄, ws.g[1], ws.g[2], ws.g[3]),
        (p[1], p[2], p[3], θ, pθ[1], pθ[2], pθ[3]), plan,
    )
    # The filtered products land in the gradient buffers, free until the gradient itself.
    for c in 1:3
        @. pθ[c] = ws.g[c] - bp[c] * ws.θ̄
    end
    _localize_triple!(ws.loc, pθ, og, b)

    _grad3!(ws.g, ws.θ̄, og, dplan, b)
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
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    for (nm, a) in (("u", u), ("v", v), ("w", w), ("θ", θ))
        size(a) == gsz || throw(DimensionMismatch("$nm has size $(size(a)), grid expects $gsz"))
    end
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, execution = execution)
    ws = TracerFlux3DWorkspace(grid; execution = Filtering.plan_execution(plan))
    return tracer_variance_flux!(
        _zeros(ws.at, T, gsz), ws, u, v, w, θ, grid, kernel, scale;
        filter_plan = plan, deriv_plan = Derivatives.derivative_plan(output_grid(grid, plan)),
    )
end
