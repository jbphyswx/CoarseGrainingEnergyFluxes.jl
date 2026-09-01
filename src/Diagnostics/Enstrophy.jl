# ---------------------------------------------------------------------------
# Enstrophy flux (Rivera, Aluie & Ecke 2014, eq. 16)
# ---------------------------------------------------------------------------

"""
    vorticity!(ω, u, v, grid[, deriv_plan]; scratch = nothing) -> ω

Vertical (radial) component of the relative vorticity on a grid resolving two tangent directions —
structured, curvilinear, a scattered node set or a sphere pixelization. Masked cells are zeroed, as
everywhere else. Uses the same gradient operator the flux diagnostics do, so `ω` and the gradients it
is later contracted against are consistent to the last bit.

On a Cartesian metric this is `ω = ∂v/∂x − ∂u/∂y`. On a sphere the curl in orthogonal curvilinear
coordinates carries the metric's own curvature term,

    ζ = (1/(R cosφ))[∂v/∂λ − ∂(u cosφ)/∂φ] = ∂v/∂x − ∂u/∂y + u tanφ/R ,

with `∂/∂x`, `∂/∂y` the distance derivatives the gradient operator returns. The spherical strain
[`compute_Π!`](@ref) contracts carries the same `tanφ/R` factor, so `ω` and `Π` share one gauge.

The two derivatives cannot be accumulated into one array, so a second full-size buffer is needed.
Pass `scratch` to supply it: a scale sweep calls this once per scale, and the in-place form is meant
to allocate nothing.
"""
function vorticity!(
    ω::AbstractVecOrMat{T},
    u::AbstractVecOrMat,
    v::AbstractVecOrMat,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    deriv_plan::Union{Nothing,AnyDerivPlan} = nothing;
    scratch::Union{Nothing,AbstractVecOrMat{T}} = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    _require_tangent_pair(grid, "vorticity!")
    dplan = _resolve_deriv_plan(deriv_plan, grid)
    tmp = scratch === nothing ? similar(ω) : scratch
    size(tmp) == size(ω) || throw(DimensionMismatch(
        "vorticity! scratch has size $(size(tmp)), expected $(size(ω))",
    ))
    # `ω` takes ∂v/∂x, `tmp` the ∂v/∂y it discards. The second call aliases both of its outputs onto
    # `tmp`: a gradient writes component 1 then component 2 at each cell and reads only the field, so
    # `tmp` is left holding ∂u/∂y and the curl needs no third buffer.
    _grad2!(ω, tmp, v, grid, dplan)
    _grad2!(tmp, tmp, u, grid, dplan)
    mask = FlowGeometries.Grids.mask(grid)
    @. ω = ifelse(mask, ω - tmp, zero(T))
    _add_curl_curvature!(ω, u, grid)
    return ω
end

# Cartesian: the curl is the plain antisymmetric derivative pair.
@inline _add_curl_curvature!(
    _ω, _u, ::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.CartesianGeometry},
) = nothing

function _add_curl_curvature!(
    ω::AbstractVecOrMat{T}, u::AbstractVecOrMat,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}}
    geo = FlowGeometries.Grids.grid_geometry(grid)
    @inbounds for I in CartesianIndices(ω)
        i = Tuple(I)
        FlowGeometries.Grids.isactive(grid, i...) || continue
        _, φ = FlowGeometries.Grids.coords(grid, i...)
        ω[I] += u[I] * _tan_factor(geo, φ)
    end
    return nothing
end

"""
    vorticity(u, v, grid[, deriv_plan]) -> ω

Allocating [`vorticity!`](@ref).
"""
function vorticity(
    u::AbstractVecOrMat, v::AbstractVecOrMat,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    deriv_plan::Union{Nothing,AnyDerivPlan} = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    return vorticity!(zeros(T, FlowGeometries.Grids.size_tuple(grid)), u, v, grid, deriv_plan)
end

"""
    enstrophy_flux(u, v, grid, kernel, scale; backend=AutoBackend(), mask_strategy=ZeroFill()) -> Z

Cross-scale enstrophy flux (Rivera, Aluie & Ecke 2014, eq. 16),

```
Z_ℓ = −∂_j ω̄_ℓ · τ_ℓ(u_j, ω) ,   τ_j = (u_j ω)‾ − ū_j ω̄ ,   ω = ∂v/∂x − ∂u/∂y ,
```

the enstrophy analogue of [`compute_Π!`](@ref): positive means enstrophy moving to smaller scales. In
2-D turbulence this is the quantity with a forward cascade while `Π` cascades inverse, so the two are
usually read together.

# Gauge

This is the **deformation (subtracted) form**, the same gauge `Π` uses: the resolved product `ū_j ω̄` is
subtracted, which is what makes it pointwise Galilean-invariant. The unsubtracted alternative
`−∂_j ω̄ (u_j ω)‾` differs from it by a transport divergence, and while the two share a spatial mean on
a homogeneous domain they "differ qualitatively as well as quantitatively" on an inhomogeneous or
masked one (Aluie 2011; Aluie, Hecht & Vallis 2018). Mixing gauges between `Π` and `Z` would make the
pair internally inconsistent, so only this one is provided.

Structurally `Z` is [`tracer_variance_flux`](@ref) with `θ = ω`, and that is how it is computed — the
enstrophy is the "variance" of the vorticity. The separate entry point exists because the caller should
not have to know to form `ω` with the matching derivative operator.

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
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    _require_tangent_pair(grid, "enstrophy_flux")
    dplan = _default_deriv_plan(grid)
    ω = vorticity(u, v, grid, dplan)
    return tracer_variance_flux(u, v, ω, grid, kernel, scale;
                                backend = backend, mask_strategy = mask_strategy)
end

"""
    EnstrophyFluxWorkspace(grid)

Scratch for [`enstrophy_flux!`](@ref): the vorticity plus the tracer-flux scratch it is fed into.
"""
struct EnstrophyFluxWorkspace{T<:AbstractFloat, A<:AbstractArray{T}, W}
    ω::A
    tracer::W
end

EnstrophyFluxWorkspace(grid::FlowGeometries.Grids.AbstractGrid{G,T}) where {T<:AbstractFloat, G} =
    EnstrophyFluxWorkspace(
        zeros(T, FlowGeometries.Grids.size_tuple(grid)), _tracer_workspace(grid),
    )

# The tracer flux the enstrophy is fed into is the one this metric needs: a plane filters the flux
# components as they stand, a sphere routes them through planetary Cartesian. The spherical methods
# of both of these are defined with that workspace, further down.
_tracer_workspace(grid::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.CartesianGeometry}) =
    TracerFluxWorkspace(grid)

# A buffer of the tracer workspace that is free while the curl is taken.
@inline _curl_scratch(ws::TracerFluxWorkspace) = ws.uθ

"""
    enstrophy_flux!(Z, ws, u, v, grid, kernel, scale; filter_plan=nothing, deriv_plan=nothing, ...) -> Z

In-place [`enstrophy_flux`](@ref). With `ws` and both plans supplied, a repeated evaluation allocates
nothing.
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
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    _require_tangent_pair(grid, "enstrophy_flux!")
    dplan = _resolve_deriv_plan(deriv_plan, grid)
    # A tracer-workspace buffer is free at this point (each is only written inside
    # `tracer_variance_flux!`), so it serves as the `∂u/∂y` scratch the curl needs.
    vorticity!(ws.ω, u, v, grid, dplan; scratch = _curl_scratch(ws.tracer))
    return tracer_variance_flux!(Z, ws.tracer, u, v, ws.ω, grid, kernel, scale;
                                 filter_plan = filter_plan, deriv_plan = dplan,
                                 backend = backend, mask_strategy = mask_strategy)
end

"""
    tracer_variance_flux(u, v, θ, grid::AbstractGrid{<:SphericalGeometry}, kernel, scale; ...) -> Πθ

Spherical form of the tracer-variance flux, on any grid resolving two tangent directions —
structured, curvilinear, a scattered node set or a sphere pixelization. `τ_j = ⟨u_j θ⟩ - ū_j θ̄` is a vector, so — exactly as in
[`compute_Π!`](@ref) and [`tau_decomposition`](@ref) — the velocity is rotated to planetary Cartesian
before filtering (Aluie 2019 commutativity: component-wise filtering of a local east/north pair is not
a filtered vector, since the local basis turns from point to point), and the filtered flux is rotated
back to the local east/north frame to contract against `∂_j θ̄`. The scalar `θ` needs no rotation. The
radial component of `τ` is dropped, matching this 2-D shell's dropping of radial derivatives.
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
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.SphericalGeometry{T}}
    _require_tangent_pair(grid, "tracer_variance_flux")
    gsz = FlowGeometries.Grids.size_tuple(grid)
    size(u) == gsz || throw(DimensionMismatch("u has size $(size(u)), grid expects $gsz"))
    size(v) == gsz || throw(DimensionMismatch("v has size $(size(v)), grid expects $gsz"))
    size(θ) == gsz || throw(DimensionMismatch("θ has size $(size(θ)), grid expects $gsz"))
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend)
    return tracer_variance_flux!(
        zeros(T, gsz), SphericalTracerFluxWorkspace(grid), u, v, θ, grid, kernel, scale;
        filter_plan = plan, deriv_plan = _default_deriv_plan(grid),
    )
end

"""
    SphericalTracerFluxWorkspace(grid)

Scratch for the spherical [`tracer_variance_flux!`](@ref): the planetary velocity, its filter, the
three velocity–tracer products and their filter, the filtered tracer, the local subfilter flux pair
and the resolved tracer gradient.
"""
struct SphericalTracerFluxWorkspace{T<:AbstractFloat, A<:AbstractArray{T}}
    p::NTuple{3,A}      # planetary velocity
    bp::NTuple{3,A}     # its filter
    pθ::NTuple{3,A}     # p_i θ, then the filtered product, then τ_i in planetary components
    θ̄::A
    τe::A; τn::A
    gx::A; gy::A
end

function SphericalTracerFluxWorkspace(
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    z() = zeros(T, gsz)
    t(n) = ntuple(_ -> z(), n)
    return SphericalTracerFluxWorkspace(t(3), t(3), t(3), z(), z(), z(), z(), z())
end

_tracer_workspace(grid::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.AbstractSphericalGeometry}) =
    SphericalTracerFluxWorkspace(grid)

@inline _curl_scratch(ws::SphericalTracerFluxWorkspace) = ws.gx

"""
    tracer_variance_flux!(Πθ, ws::SphericalTracerFluxWorkspace, u, v, θ, grid, kernel, scale; ...) -> Πθ

In-place spherical [`tracer_variance_flux`](@ref). One batched apply carries the whole flux: the three
planetary velocity components, the tracer, and the three velocity–tracer products.
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
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}}
    _require_tangent_pair(grid, "tracer_variance_flux!")
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend) :
        filter_plan
    dplan = _resolve_deriv_plan(deriv_plan, grid)
    geo = FlowGeometries.Grids.grid_geometry(grid)
    p, bp, pθ = ws.p, ws.bp, ws.pθ

    # `τ_j = ⟨u_j θ⟩ − ū_j θ̄` is a vector, so the velocity goes to planetary Cartesian before
    # filtering; the scalar `θ` needs no rotation.
    @inbounds for I in CartesianIndices(Πθ)
        i = Tuple(I)
        if FlowGeometries.Grids.isactive(grid, i...)
            λ, φ = FlowGeometries.Grids.coords(grid, i...)
            c = FlowGeometries.Geometry.vector_to_cartesian(geo, u[I], v[I], λ, φ)
            for k in 1:3
                p[k][I] = c[k]
                pθ[k][I] = c[k] * θ[I]
            end
        else
            for k in 1:3
                p[k][I] = zero(T); pθ[k][I] = zero(T)
            end
        end
    end

    Filtering.filter_apply_batch!(
        (bp[1], bp[2], bp[3], ws.θ̄, ws.gx, ws.gy, ws.τe),
        (p[1], p[2], p[3], θ, pθ[1], pθ[2], pθ[3]), plan,
    )
    # The filtered products land in three buffers that are free until the gradient; move them back
    # into `pθ` as the planetary subfilter flux.
    @. pθ[1] = ws.gx - bp[1] * ws.θ̄
    @. pθ[2] = ws.gy - bp[2] * ws.θ̄
    @. pθ[3] = ws.τe - bp[3] * ws.θ̄

    # Back to local (east, north); the radial component is unused, the resolved gradient here having
    # no radial part.
    @inbounds for I in CartesianIndices(Πθ)
        i = Tuple(I)
        if FlowGeometries.Grids.isactive(grid, i...)
            λ, φ = FlowGeometries.Grids.coords(grid, i...)
            l = FlowGeometries.Geometry.vector_from_cartesian(
                geo, pθ[1][I], pθ[2][I], pθ[3][I], λ, φ,
            )
            ws.τe[I] = l[1]; ws.τn[I] = l[2]
        else
            ws.τe[I] = zero(T); ws.τn[I] = zero(T)
        end
    end

    _grad2!(ws.gx, ws.gy, ws.θ̄, grid, dplan)
    mask = FlowGeometries.Grids.mask(grid)
    @. Πθ = ifelse(mask, -(ws.τe * ws.gx + ws.τn * ws.gy), zero(T))
    return Πθ
end

"""
    TracerFlux3DWorkspace(grid)

Scratch for the true-3-D [`tracer_variance_flux!`](@ref): the velocity triple, its filter, the three
velocity–tracer products and their filter, the local subfilter flux, the filtered tracer and its
three-component resolved gradient.
"""
struct TracerFlux3DWorkspace{T<:AbstractFloat, A<:AbstractArray{T,3}}
    p::NTuple{3,A}      # the velocity triple, planetary on a shell
    bp::NTuple{3,A}     # its filter
    pθ::NTuple{3,A}     # p_i θ, then the filtered product, then τ_i
    loc::NTuple{3,A}    # τ_i in the local frame
    g::NTuple{3,A}      # ∇θ̄
    θ̄::A
end

function TracerFlux3DWorkspace(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,3},
) where {T<:AbstractFloat, G}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    t(n) = ntuple(_ -> zeros(T, gsz), n)
    return TracerFlux3DWorkspace(t(3), t(3), t(3), t(3), t(3), zeros(T, gsz))
end

"""
    tracer_variance_flux!(Πθ, ws::TracerFlux3DWorkspace, u, v, w, θ, grid, kernel, scale; ...) -> Πθ

In-place true-3-D [`tracer_variance_flux`](@ref), on either metric. One batched apply carries the
whole flux: the three velocity components, the tracer, and the three velocity–tracer products.
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
    deriv_plan::Union{Nothing,Derivatives.StencilPlan} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend) :
        filter_plan
    dplan = deriv_plan === nothing ? Derivatives.StencilPlan(grid) : deriv_plan
    p, bp, pθ = ws.p, ws.bp, ws.pθ

    _fill_tau3_velocity!(p, u, v, w, grid, T)
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
    _localize_triple!(ws.loc, pθ, grid, T)

    Derivatives.ddx!(ws.g[1], ws.θ̄, grid, dplan)
    Derivatives.ddy!(ws.g[2], ws.θ̄, grid, dplan)
    Derivatives.ddz!(ws.g[3], ws.θ̄, grid, dplan)
    mask = FlowGeometries.Grids.mask(grid)
    @. Πθ = ifelse(mask,
        -(ws.loc[1] * ws.g[1] + ws.loc[2] * ws.g[2] + ws.loc[3] * ws.g[3]), zero(T))
    return Πθ
end

"""
    tracer_variance_flux(u, v, w, θ, grid::StructuredGrid{T,Cartesian,3}, kernel, scale; backend=AutoBackend(), mask_strategy=ZeroFill())
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
    grid::FlowGeometries.Grids.StructuredGrid{T,G,3},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
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
    tracer_variance_flux(u, v, w, θ, grid::StructuredGrid{T,<:SphericalGeometry,3}, kernel, scale; ...) -> Πθ

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
    grid::FlowGeometries.Grids.StructuredGrid{T,G,3},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
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
