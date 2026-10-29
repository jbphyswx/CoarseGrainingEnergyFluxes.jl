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
in the memory of `execution`, whose loops take the curl; `grid` and `deriv_plan` are moved there on each
call unless they are there already.
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
    b = _placement_backend(execution, grid)
    return _curl_into!(ω, u, v, FlowGeometries.Execution.on_backend(b, grid),
                       FlowGeometries.Execution.on_backend(b, _resolve_deriv_plan(deriv_plan, grid)), g, b)
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
    b = _placement_backend(execution, grid)
    z() = Filtering._allocate(b, T, FlowGeometries.Grids.size_tuple(grid))
    return _curl_into!(z(), u, v, FlowGeometries.Execution.on_backend(b, grid),
                       FlowGeometries.Execution.on_backend(b, _resolve_deriv_plan(deriv_plan, grid)), (z(), z()), b)
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
`vector_filter` and `transform` choose how it filters the vector `u ω`. `derivative` takes `∇ω̄` as the
tracer flux takes `∇θ̄`. `ω` differences the unfiltered velocity under every `derivative`, at its node count:
a series of the raw field is exact only for an unmasked periodic band-limited one, and on a masked grid
rings at the coast.

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
    method::Filtering.AbstractFilterMethod = Filtering.RealSpace(),
    derivative::Derivatives.AbstractDerivative = Derivatives.AutoDerivative(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    _require_tangent_pair(grid, "enstrophy_flux")
    ω = vorticity(u, v, grid, Derivatives.derivative_plan(grid; nodes = _raw_nodes(derivative)); execution = execution)
    vector = G <: FlowGeometries.Geometry.CartesianGeometry ? (; method) : (; vector_filter, transform)
    return tracer_variance_flux(u, v, ω, grid, kernel, scale;
                                execution = execution, mask_strategy = mask_strategy, derivative = derivative,
                                vector...)
end

# The node count of the differences an unfiltered field takes under `derivative`.
@inline _raw_nodes(d::Derivatives.PhysicalDerivative) = d.nodes
@inline _raw_nodes(::Derivatives.AbstractDerivative) = Derivatives.PhysicalDerivative().nodes

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
    derivative::Derivatives.AbstractDerivative = Derivatives.AutoDerivative(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    _require_tangent_pair(grid, "enstrophy_flux!")
    plan = _plan_for(_tracer_spin(ws.tracer), filter_plan, mask_strategy,
                     () -> Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, execution = execution))
    at = ws.tracer.at
    _check_placement(at, plan)
    og = output_grid(grid, plan)
    nodes = _raw_nodes(derivative)
    dphys = deriv_plan === nothing ? Derivatives.derivative_plan(og; nodes) : deriv_plan
    _curl_into!(ws.ω, u, v, at.grid, _placed(at, _curl_plan(curl_plan, dphys, grid, og, nodes)),
                _curl_scratch(ws.tracer), at.execution)
    return tracer_variance_flux!(Z, ws.tracer, u, v, ws.ω, grid, kernel, scale;
                                 filter_plan = _built_plan(plan), mask_strategy = mask_strategy,
                                 deriv_plan = deriv_plan, derivative = derivative)
end

# A tracer workspace's generalized convolution, `nothing` on a plane.
@inline _tracer_spin(::TracerFluxWorkspace) = nothing
@inline _tracer_spin(ws) = ws.spin

# The plan `_plan_for` gave, as a `filter_plan` keyword: `nothing` for a mask strategy.
@inline _built_plan(plan::Filtering.AbstractFilterPlan) = plan
@inline _built_plan(::Filtering.AbstractMaskStrategy) = nothing

# The vorticity differences the raw velocity over `grid`'s own cells.
@inline _curl_plan(plan::AnyDerivPlan, _dplan, _grid, _og, _nodes) = plan
_curl_plan(::Nothing, dplan, grid, og, nodes::Int) =
    _same_cells(grid, og) ? dplan : Derivatives.derivative_plan(grid; nodes)
