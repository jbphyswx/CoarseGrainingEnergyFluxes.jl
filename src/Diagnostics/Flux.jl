# ---------------------------------------------------------------------------
# Energy Flux (Π) Calculation
# ---------------------------------------------------------------------------

# Boundary-only (once per top-level call, not per grid point): a mismatched v/w would otherwise be
# silently truncated/ignored by CartesianIndices(u), not caught at all.
#
# Fields may carry trailing batch axes beyond the grid's rank, so what has to match is the LEADING axes
# plus the batch shape being common to every field — not the full size. Checking full equality would
# reject a batch; checking only the leading axes would let a `(Nx,Ny,3)` u pair with a `(Nx,Ny,5)` v and
# then silently truncate against whichever is shorter, which is the failure this guard exists to catch.
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

# `Val`-typed ranks throughout: slicing `size(A)` with a runtime range cannot infer a fixed-size tuple and
# allocates on every call, which is why these are built with `ntuple` at a statically known length.
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

# The fields a flux computation filters, in the order their spectra are stored. Every one of them is a
# RAW input — the velocities and their products — so none depends on the filter scale, which is what lets
# a sweep transform them once and only synthesize per scale.
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

Marker that `ws.ux`/`uy`/`uz` already hold the planetary-Cartesian velocity components for this sweep.

The spherical flux path works in planetary Cartesian coordinates, and the rotation into them is a
function of `(u, v, w, grid)` alone — a `coords` lookup and a trigonometric basis change at every
point, with no dependence on the filter scale. Those three buffers are not written again during a
scale, so one rotation serves the whole sweep.
"""
struct SphericalAnalysis end

"""
    _fill_planetary!(ws, u, v, w, grid) -> nothing
    _fill_planetary!((px, py, pz), u, v, w, grid) -> nothing

Rotate the local (east, north, up) velocity into planetary Cartesian components in `ws.ux/uy/uz`, or
in `px/py/pz`. Inactive cells are set to zero, which is the value the filter reads there.
"""
_fill_planetary!(ws::ΠWorkspace, u, v, w, grid::FlowGeometries.Grids.AbstractGrid) =
    _fill_planetary!((ws.ux, ws.uy, ws.uz), u, v, w, grid)

function _fill_planetary!(
    p::NTuple{3,AbstractArray}, u, v, w, grid::FlowGeometries.Grids.AbstractGrid{G,T},
) where {G, T<:AbstractFloat}
    has_w = w !== nothing
    geo = FlowGeometries.Grids.grid_geometry(grid)
    px, py, pz = p
    for I in CartesianIndices(px)
        let i = Tuple(I)
            if FlowGeometries.Grids.isactive(grid, i...)
                λ, φ = FlowGeometries.Grids.coords(grid, i...)
                p_vel = FlowGeometries.Geometry.vector_to_cartesian(
                    geo, u[I], v[I], has_w ? w[I] : zero(T), λ, φ,
                )
                px[I] = p_vel[1]
                py[I] = p_vel[2]
                pz[I] = p_vel[3]
            else
                px[I] = zero(T)
                py[I] = zero(T)
                pz[I] = zero(T)
            end
        end
    end
    return nothing
end

"""
    _planetary_to_local!(l, (px, py, pz), grid) -> l

The local (east, north[, up]) components of the planetary vector `(px, py, pz)` into the arrays of `l`,
two or three of them; zero on inactive cells. Each point reads all three planetary components before
it writes, so `l` may alias them.
"""
function _planetary_to_local!(
    l::Tuple, p::NTuple{3,AbstractArray}, grid::FlowGeometries.Grids.AbstractGrid{G,T},
) where {G, T<:AbstractFloat}
    geo = FlowGeometries.Grids.grid_geometry(grid)
    px, py, pz = p
    for I in CartesianIndices(px)
        i = Tuple(I)
        if FlowGeometries.Grids.isactive(grid, i...)
            λ, φ = FlowGeometries.Grids.coords(grid, i...)
            loc = FlowGeometries.Geometry.vector_from_cartesian(geo, px[I], py[I], pz[I], λ, φ)
            for c in eachindex(l)
                l[c][I] = loc[c]
            end
        else
            for c in eachindex(l)
                l[c][I] = zero(T)
            end
        end
    end
    return l
end

# Spherical grids: the shareable half is the rotation into planetary Cartesian, not a transform. Doing
# it here keeps it out of the scale loop; the per-scale path still runs it when no sweep analysis was
# made (a single `compute_Π!` call).
function analyze_sweep(
    u, v, w, grid::FlowGeometries.Grids.AbstractGrid{G,T},
    ws::ΠWorkspace, plan::Filtering.AbstractFilterPlan,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}}
    # Every grid whose flux takes the 2-D spherical branch, named by the count of directions it
    # resolves. A true-3-D spherical grid's `coords` returns `(λ, φ, r)` and it carries its own
    # rotation, so it keeps the per-scale path.
    FlowGeometries.Grids.ncoordinates(grid) == 2 || return nothing
    _fill_planetary!(ws, u, v, w, grid)
    return SphericalAnalysis()
end

"""
    analyze_sweep(u, v, w, grid, ws, plan) -> analysis or nothing

Forward-transform the raw inputs of a flux computation once, for reuse across every scale of a sweep.

A spectral filter is analyze → multiply by `Ĝ(|k|, ℓ)` → synthesize, and only the multiply depends on
the scale, while every field a flux computation filters is raw: the velocities and their pairwise
products. So a sweep over `S` scales needs `5 + 5S` transforms rather than `10S`.

Returns `nothing` for an engine with no shareable analysis — a real-space filter does all its work per
scale — and the caller then runs the ordinary per-scale path. `plan` may be any one of the sweep's
per-scale plans; they share a forward transform.
"""
function analyze_sweep(u, v, w, grid, ws::ΠWorkspace, plan::Filtering.AbstractFilterPlan)
    Filtering.analyze_buffer(plan, u) === nothing && return nothing
    has_w = w !== nothing
    # Products are formed into the same scratch the per-scale path uses; once analyzed, the spectra hold
    # everything the sweep needs and the scratch is free again.
    @. ws.ux = u * u
    @. ws.uy = u * v
    @. ws.uz = v * v
    if has_w
        @. ws.ux_filt = u * w
        @. ws.uy_filt = v * w
        @. ws.uz_filt = w * w
    end
    vel = map(f -> Filtering.filter_analyze!(Filtering.analyze_buffer(plan, f), f, plan),
              _velocity_ins(u, v, w, has_w))
    prod = map(f -> Filtering.filter_analyze!(Filtering.analyze_buffer(plan, f), f, plan),
               _product_ins(ws, has_w))
    return (velocity = vel, product = prod)
end

"""
    compute_Π!(Π, u, v, w, grid, kernel, scale; workspace=nothing, backend=AutoBackend(), mask_strategy=ZeroFill())

Compute the cross-scale kinetic energy flux Π = -S̄_ij τ_ij at filter scale ℓ.

This implements the coarse-graining framework of Aluie et al. (2018) for computing
energy transfer across scales in turbulent flows. Positive Π indicates forward cascade
(energy from large to small scales), negative Π indicates inverse cascade.

# Arguments
- `Π::AbstractMatrix{T}`: Output array for energy flux (modified in-place)
- `u::AbstractMatrix`: Eastward/zonal velocity component
- `v::AbstractMatrix`: Northward/meridional velocity component
- `w::Union{Nothing,AbstractMatrix}`: Vertical velocity (nothing for 2D calculations)
- `grid::StructuredGrid`: Grid geometry and coordinates
- `kernel::AbstractFilterKernel`: Filter kernel
- `scale::T`: Filter scale ℓ in meters

# Keyword Arguments
- `workspace=nothing`: Pre-allocated ΠWorkspace for intermediate arrays
- `backend::AbstractExecutionBackend=AutoBackend()`: Execution backend
- `mask_strategy::AbstractMaskStrategy=ZeroFill()`: Masking strategy (`ZeroFill()` or `Deformable()`).
  `ZeroFill` is the default because it keeps the kernel position-independent, so filtering commutes
  with spatial derivatives — the property the flux budget is derived by. See
  [`Filtering.filter_field!`](@ref) for the boundary artifacts of both choices.

# Physics
The cross-scale energy flux is computed as:
```
Π = -S̄_ij * τ_ij
```
where:
- `S̄_ij = 0.5 * (∂ū_i/∂x_j + ∂ū_j/∂x_i)` is the resolved strain rate tensor
- `τ_ij = [u_i*u_j]̄ - ū_i*ū_j` is the subfilter-scale (SFS) stress tensor
- Overbar denotes filtered quantities

For spherical geometry, velocity components are transformed to planetary Cartesian
coordinates before filtering to ensure commutativity with derivatives (Aluie 2019).

# Physics regime: 2.5D thin-layer/quasi-geostrophic approximation when `w` is supplied
When `w !== nothing`, this method still computes only a SINGLE 2D layer's tensor: it includes the
cross terms `S_xz = ½∂ū/∂x, S_yz = ½∂v̄/∂y` in the strain contraction, but sets `S_zz = ∂w̄/∂z ≡ 0` and
never differentiates `u`/`v`/`w` in the vertical — there is no 3rd spatial dimension in the input
arrays for it to differentiate against. This is not a shortcut; it is the standard thin-layer (small
aspect ratio δ = H/L) / quasi-geostrophic scaling used throughout large-scale ocean and atmosphere
dynamics (Vallis, *Atmospheric and Oceanic Fluid Dynamics*, §5; Pedlosky, *Geophysical Fluid
Dynamics*, ch. 6), under which vertical shear terms are genuinely subdominant to horizontal gradients
— valid for the normal large-scale, stratified, rotating-flow regime this package targets, NOT for
homogeneous/isotropic 3D turbulence (e.g. boundary-layer or Rayleigh–Taylor studies), where filtering
genuinely blends all three directions and vertical derivatives are real, not assumed away. The
literature on "vertical structure via coarse-graining" (Aluie, Hecht & Vallis 2018, JPO; Buzzicotti,
Storer, Khatri, Griffies & Aluie 2023, JAMES) analyzes vertical structure by running this SAME 2D/2.5D
method independently at each z level of a multi-level dataset and comparing/stacking the resulting
profiles — not by computing a coupled 3D tensor — so `Pipeline.coarse_grain_profile` (which sweeps this
method over the vertical axis as a batch) is the literature-matching way to get a vertical-structure
result. A genuinely coupled, all-nine-strain-component 3D method exists separately
for the true-3D Cartesian case (see the `AbstractArray{T,3}` `compute_Π!` method).

# Returns
- `Π`: the specific flux, per unit mass, in m² s⁻³ (velocity in m s⁻¹, lengths in m); `ρ₀·Π` is the
  flux per unit volume, W m⁻³

# Examples
```julia
Π = zeros(100, 100)
compute_Π!(Π, u, v, nothing, grid, TopHatKernel(), 30000.0)
# Π now contains energy flux at 30 km scale
```

# References
- Aluie et al. (2018): https://doi.org/10.1175/JPO-D-17-0100.1
- Aluie (2019): https://doi.org/10.1007/s13137-019-0123-9
- Vallis, G.K., *Atmospheric and Oceanic Fluid Dynamics*, 2nd ed., Cambridge University Press, 2017.
- Pedlosky, J., *Geophysical Fluid Dynamics*, 2nd ed., Springer, 1987.
- Aluie, Hecht & Vallis (2018), *J. Phys. Oceanogr.* 48(2): https://doi.org/10.1175/JPO-D-17-0100.1
- Buzzicotti, Storer, Khatri, Griffies & Aluie (2023), *J. Adv. Model. Earth Syst.*:
  https://doi.org/10.1029/2021MS002583
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
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    deriv_plan::Union{Nothing, Derivatives.StencilPlan} = nothing,
    # Spectra of the raw inputs from [`analyze_sweep`](@ref), when a sweep has hoisted the
    # scale-independent forward transform out of its scale loop.
    analyzed = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    # The grid's rank is fixed and the arrays' is not, so a field may carry trailing batch axes:
    # `(Nx, Ny, Nb)` against this rank-2 grid is a batch of slices. The filter applies, the elementwise
    # tensor algebra and the stencil derivatives all carry the trailing axes through.
    _validate_field_sizes(grid, Π, u, v, w)
    ws = workspace === nothing ?
        ΠWorkspace(grid, _batch_dims(Π, grid); has_w = w !== nothing) : workspace
    _check_workspace_w(ws, w)
    # One plan for this scale, shared by all ~9 filterings below. A caller repeating this at a fixed
    # scale can pass a prebuilt `filter_plan` to share it across calls as well.
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend) : filter_plan
    # The stencil weights depend only on the grid, so one table serves every derivative here and every
    # later call at any scale.
    dplan = deriv_plan === nothing ? Derivatives.StencilPlan(grid) : deriv_plan
    return _compute_Π!(Π, u, v, w, grid, ws, plan, dplan, analyzed)
end
