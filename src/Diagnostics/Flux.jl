# ---------------------------------------------------------------------------
# Energy Flux (Π) Calculation
# ---------------------------------------------------------------------------

"""
    SeriesStrain(gradient = nothing)

The derivative route of a diagnostic whose filtered fields are differentiated from their filter's own series
([`Derivatives.SpectralDerivative`](@ref)), in the place of a derivative plan; a sweep passes it as
`deriv_plan` once resolved. `gradient` is the plan's [`Filtering.series_gradient`](@ref).
"""
struct SeriesStrain{G}
    gradient::G
end
SeriesStrain() = SeriesStrain(nothing)

# A `deriv_plan` keyword: a derivative plan, a sweep's resolved series route, or `nothing`.
const _DerivRoute{P} = Union{Nothing, P, SeriesStrain}

# Fields may carry trailing batch axes past the grid's rank: each field's leading axes are the grid's
# and every field has the same batch shape.
@inline function _validate_field_sizes(grid, Π::AbstractArray, u::AbstractArray, v = nothing, w = nothing)
    gsz = FlowGeometries.Grids.size_tuple(grid)
    valR = Val(length(gsz))
    _check_leading(Π, "Π", gsz, valR)
    _check_leading(u, "u", gsz, valR)
    bsz = _batch_dims(Π, valR)
    _batch_dims(u, valR) == bsz || throw(DimensionMismatch(
        "u's batch axes $(_batch_dims(u, valR)) do not match Π's $bsz",
    ))
    if v !== nothing
        _check_leading(v, "v", gsz, valR)
        _batch_dims(v, valR) == bsz || throw(DimensionMismatch(
            "v's batch axes $(_batch_dims(v, valR)) do not match Π's $bsz",
        ))
    end
    if w !== nothing
        _check_leading(w, "w", gsz, valR)
        _batch_dims(w, valR) == bsz || throw(DimensionMismatch(
            "w's batch axes $(_batch_dims(w, valR)) do not match Π's $bsz",
        ))
    end
    return nothing
end

# `Val`-typed ranks: `size(A)` sliced by a runtime range infers no fixed-size tuple and allocates.
@inline function _check_leading(
    A::AbstractArray, name::AbstractString, gsz::NTuple{R,Int}, ::Val{R},
) where {R}
    ndims(A) >= R || throw(DimensionMismatch(
        "$name has $(ndims(A)) dimensions, grid expects at least $R",
    ))
    ntuple(i -> size(A, i), Val(R)) == gsz || throw(DimensionMismatch(
        "$name's leading axes $(ntuple(i -> size(A, i), Val(R))) do not match grid shape $gsz",
    ))
    return nothing
end

# Trailing axes of a field beyond the grid's rank — the batch shape a workspace must be sized for.
@inline _batch_dims(A::AbstractArray, ::Val{R}) where {R} =
    ntuple(i -> size(A, R + i), Val(ndims(A) - R))
@inline _batch_dims(A::AbstractArray, grid) =
    _batch_dims(A, Val(length(FlowGeometries.Grids.size_tuple(grid))))

# The fields a flux computation filters, in the order their spectra are stored: the velocities and their
# products. None depends on the filter scale, so a sweep transforms them once and synthesizes per scale.
@inline _velocity_outs(ws::ΠWorkspace, has_w::Bool) =
    has_w ? (ws.u_filt, ws.v_filt, ws.w_filt) : (ws.u_filt, ws.v_filt)
@inline _velocity_ins(u, v, w, has_w::Bool) = has_w ? (u, v, w) : (u, v)
@inline _product_outs(ws::ΠWorkspace, has_w::Bool) =
    has_w ? (ws.uu_filt, ws.uv_filt, ws.vv_filt, ws.uw_filt, ws.vw_filt, ws.ww_filt) :
            (ws.uu_filt, ws.uv_filt, ws.vv_filt)
@inline _product_ins(ws::ΠWorkspace, has_w::Bool) =
    has_w ? (ws.ux, ws.uy, ws.uz, ws.ux_filt, ws.uy_filt, ws.uz_filt) : (ws.ux, ws.uy, ws.uz)

@inline function _synthesize_all!(outs::Tuple, spectra::Tuple, plan)
    for k in eachindex(outs)
        Filtering.filter_synthesize!(outs[k], spectra[k], plan)
    end
    return outs
end

"""
    SphericalAnalysis

Marker that `ws.ux`/`uy`/`uz` already hold the ambient Cartesian velocity components for this sweep.

The rotation into ambient Cartesian components depends on `(u, v, w, grid)` alone, and the per-scale
path does not write those three buffers, so one rotation serves the whole sweep.
"""
struct SphericalAnalysis end

# The ambient Cartesian components of the local (east, north, up) velocity into the three arrays of
# `p`, or into `ws.ux/uy/uz`; zero on inactive cells, the value the filter reads there. `frames` is the
# placement's frame plan of the cells the components are read on, held where the backend `b` runs.
@inline _to_ambient!(p::NTuple{3,AbstractArray}, u, v, w, frames::FlowGeometries.Operators.FramePlan, b) =
    FlowGeometries.Operators.to_planetary!(p[1], p[2], p[3], u, v, w, frames; execution = b)
@inline _to_ambient!(ws::ΠWorkspace, u, v, w) =
    _to_ambient!((ws.ux, ws.uy, ws.uz), u, v, w, ws.at.frames, ws.at.execution)

# On a sphere or a spheroid a sweep shares, across its scales, the spin analysis of the generalized
# convolution, or the rotation into ambient Cartesian components; a single `compute_Π!` call does either
# in its own pass.
function analyze_sweep(
    u, v, w, grid::FlowGeometries.Grids.AbstractGrid{G,T},
    ws::ΠWorkspace, plan::Filtering.AbstractFilterPlan,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractLonLatGeometry{T}}
    sf = _spin_of(ws.spin, plan)
    sf === nothing || return _spin_sweep_analysis(sf, u, v, w)
    # Only the two-coordinate branch shares the rotation; a shell (`coords` returns `(λ, φ, r)`)
    # rotates per scale.
    FlowGeometries.Grids.ncoordinates(grid) == 2 || return nothing
    _to_ambient!(ws, u, v, w)
    return SphericalAnalysis()
end

# A sweep reading no plan, whose workspace's generalized convolution filters every field (`_plan_for`).
analyze_sweep(u, v, w, _grid, ws::ΠWorkspace, strategy::Filtering.AbstractMaskStrategy) =
    _spin_sweep_analysis(_spin_of(ws.spin, strategy), u, v, w)

# The filter holds one slice's coefficients, so a sweep over fields with batch axes analyzes per scale.
function _spin_sweep_analysis(sf::SpinFilter, u, v, w)
    length(u) == sf.ncells || return nothing
    _spin_analyze_flux!(sf, u, v, w, 0)
    return SpinAnalysis()
end

"""
    analyze_sweep(u, v, w, grid, ws, plan) -> analysis or nothing

Forward-transform the raw inputs of a flux computation once, for reuse across every scale of a sweep.

A spectral filter is analyze → multiply by `Ĝ(|k|, ℓ)` → synthesize; only the multiply depends on the
scale, and every field a flux computation filters (the velocities and their pairwise products) is raw
input.

Returns `nothing` for an engine with no shareable analysis, such as a real-space filter; the caller
then runs the per-scale path. `plan` may be any one of the sweep's per-scale plans; they share a
forward transform. On a sphere the workspace's generalized convolution analyzes the velocity and its
products, and `plan` may be the mask strategy alone.
"""
function analyze_sweep(u, v, w, grid, ws::ΠWorkspace, plan::Filtering.AbstractFilterPlan)
    Filtering.analyze_buffer(plan, u) === nothing && return nothing
    has_w = w !== nothing
    _sweep_products!(ws, u, v, w, has_w)
    vel = map(f -> Filtering.filter_analyze!(Filtering.analyze_buffer(plan, f), f, plan),
              _velocity_ins(u, v, w, has_w))
    prod = map(f -> Filtering.filter_analyze!(Filtering.analyze_buffer(plan, f), f, plan),
               _product_ins(ws, has_w))
    return (velocity = vel, product = prod)
end

"""
    analyze_sweep!(a, u, v, w, grid, ws, plan) -> a

[`analyze_sweep`](@ref) into the spectra of `a`, an earlier result for fields of the same shape on a plan
sharing `plan`'s transforms.
"""
function analyze_sweep!(a::NamedTuple{(:velocity, :product)}, u, v, w, grid, ws::ΠWorkspace,
                        plan::Filtering.AbstractFilterPlan)
    has_w = w !== nothing
    _sweep_products!(ws, u, v, w, has_w)
    map((F̂, f) -> Filtering.filter_analyze!(F̂, f, plan), a.velocity, _velocity_ins(u, v, w, has_w))
    map((F̂, f) -> Filtering.filter_analyze!(F̂, f, plan), a.product, _product_ins(ws, has_w))
    return a
end

# The products borrow the per-scale path's scratch, which is free again once analyzed.
function _sweep_products!(ws::ΠWorkspace, u, v, w, has_w::Bool)
    @. ws.ux = u * u
    @. ws.uy = u * v
    @. ws.uz = v * v
    if has_w
        @. ws.ux_filt = u * w
        @. ws.uy_filt = v * w
        @. ws.uz_filt = w * w
    end
    return ws
end

"""
    compute_Π!(Π, u, v, w, grid, kernel, scale; workspace=nothing, filter_plan=nothing, execution=AutoBackend(), mask_strategy=ZeroFill(), method=RealSpace(), vector_filter=nothing, transform=AutoSpectralBackend(), deriv_plan=nothing, derivative=AutoDerivative(), analyzed=nothing)

The cross-scale kinetic energy flux `Π = −S̄_ij τ_ij` at filter scale `scale` (Aluie, Hecht & Vallis
2018): positive where energy passes from scales larger than `scale` to smaller ones.

`S̄_ij = ½(∂ū_i/∂x_j + ∂ū_j/∂x_i)` is the strain rate of the filtered velocity and
`τ_ij = (u_i u_j)‾ − ū_i ū_j` the subfilter stress. `derivative` ([`Derivatives.AbstractDerivative`](@ref))
takes the strain: by default the derivative of the filter's own series under `method = Spectral()` and
differences of the filtered field otherwise, by `deriv_plan` when given. On a sphere the velocity `ū` and the stress
`(u_i u_j)‾` are filtered by `vector_filter` ([`Filtering.resolve_vector_filter`](@ref)): by default the
generalized convolution of Aluie (2019) under `ZeroFill`, which commutes with the surface derivatives,
its spin transforms chosen by `transform`; `Filtering.CartesianComponents()` filters the ambient
Cartesian components and rotates them back to the local frame. `vector_filter` and `transform` build the
workspace when `workspace` is not given. The generalized convolution reads no scalar filter plan, and
none is built for it when `filter_plan` is not given.

`u` and `v` are the eastward and northward components. A given `w` is the vertical velocity of a thin
layer: `S_xz = ½∂w̄/∂x`, `S_yz = ½∂w̄/∂y` and `S_zz = 0`, with no vertical derivative, the thin-layer
scaling of Vallis (2017, §5) and Pedlosky (1987, ch. 6). Vertical structure is this method applied level
by level (Aluie, Hecht & Vallis 2018; Buzzicotti et al. 2023), as `Pipeline.coarse_grain_profile` does;
a 3-D grid takes the method that differentiates along all three axes.

Fields may carry trailing batch axes past the grid's two. Under `mask_strategy = ZeroFill()` the kernel
is the same at every position, so filtering commutes with derivatives; [`Filtering.filter_field!`](@ref)
describes each strategy at a boundary. `execution`, `mask_strategy` and `method` build the filter plan
when `filter_plan` is not given. A prebuilt `filter_plan`, `deriv_plan`
(`Derivatives.derivative_plan(output_grid(grid, mask_strategy))`) and `workspace` serve repeated calls;
the workspace lives in the memory of the plan's backend ([`ΠWorkspace`](@ref)) and the flux's own loops
run under the workspace's. `Π` and the fields live there too. `analyzed` takes a sweep's
[`analyze_sweep`](@ref) result.

Returns `Π`, the specific flux in m² s⁻³ (velocity in m s⁻¹, lengths in m); `ρ₀Π` is the flux per unit
volume, W m⁻³.

# Example
```julia
Π = zeros(100, 100)
compute_Π!(Π, u, v, nothing, grid, TopHatKernel(), 30000.0)
```

# References
- Aluie, Hecht & Vallis (2018), *J. Phys. Oceanogr.* 48(2): https://doi.org/10.1175/JPO-D-17-0100.1
- Aluie (2019): https://doi.org/10.1007/s13137-019-0123-9
- Buzzicotti, Storer, Khatri, Griffies & Aluie (2023), *J. Adv. Model. Earth Syst.*:
  https://doi.org/10.1029/2021MS002583
- Vallis, *Atmospheric and Oceanic Fluid Dynamics*, 2nd ed., Cambridge University Press, 2017.
- Pedlosky, *Geophysical Fluid Dynamics*, 2nd ed., Springer, 1987.
"""
function compute_Π!(
    Π::AbstractArray{T},
    u::AbstractArray,
    v::AbstractArray,
    w::Union{Nothing, AbstractArray}, # nothing or zeros for 2D
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    workspace::Union{Nothing, ΠWorkspace} = nothing,
    filter_plan::Union{Nothing, Filtering.AbstractFilterPlan} = nothing,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Filtering.AbstractFilterMethod = Filtering.RealSpace(),
    vector_filter::Union{Nothing, Filtering.AbstractVectorFilter} = nothing,
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    deriv_plan::_DerivRoute{<:FlowGeometries.Operators.StencilPlans} = nothing,
    derivative::Derivatives.AbstractDerivative = Derivatives.AutoDerivative(),
    # A sweep's `analyze_sweep` result, or `nothing`.
    analyzed = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    _validate_field_sizes(grid, Π, u, v, w)
    ws, plan = _flux_setup(
        grid, Π, w, workspace, filter_plan,
        () -> Filtering.plan_filter(grid, kernel, scale; mask_strategy, execution, method, transform);
        execution, mask_strategy, vector_filter, transform,
    )
    _check_workspace_w(ws, w)
    # The stencil weights depend only on the output grid, so one plan serves every scale.
    route = _derivative_route(derivative, deriv_plan, _route_source(ws.spin, plan), grid)
    return _compute_Π!(Π, u, v, w, grid, ws, plan, route, kernel, scale, analyzed)
end
