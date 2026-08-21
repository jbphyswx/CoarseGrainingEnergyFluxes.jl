module Diagnostics

using FlowGeometries: FlowGeometries
using ..Kernels: Kernels
using ..Filtering: Filtering
using ..Derivatives: Derivatives
using ComputationalBackends: ComputationalBackends

export ΠWorkspace, compute_Π!, cumulative_energy, cumulative_energy!, filtering_spectrum, spectral_density, spectral_density!
export tau_decomposition, compute_Π_decomposed, tracer_variance_flux
export compute_Π_strain_convergence, compute_Π_strain_convergence!, PiStrainWorkspace
export vorticity, vorticity!, enstrophy_flux, enstrophy_flux!, EnstrophyFluxWorkspace
export band_energies
export compressible_flux, compressible_flux!, FavreWorkspace, favre_filter!
export AbstractSpectrumPolicy, StrictSpectrum, ForceSpectrum, NoSpectrum

# ---------------------------------------------------------------------------
# Spectrum admissibility policy (singleton types — specializable, same idiom as AbstractMaskStrategy)
# ---------------------------------------------------------------------------

"""
    AbstractSpectrumPolicy

What to do when a filtering spectral density is asked for with a kernel whose `|Ĝ(k)|²` is not monotone
decreasing — [`StrictSpectrum`](@ref), [`ForceSpectrum`](@ref) or [`NoSpectrum`](@ref).

Sadek & Aluie (2018) eq. (21) guarantees `Ẽ(k_ℓ) ≥ 0` only when `d|Ĝ(k)|²/dk ≤ 0`. That condition is
**sufficient, not necessary**, and where it fails it tends to fail narrowly: the default `TopHatKernel`'s
`|Ĝ|²` falls to zero at `kℓ ≈ 7.66` and climbs back to only `0.0175` at `kℓ ≈ 10.27`, so the violation
sits in the far sub-filter tail at under 2% of the DC value while the rest of the curve is usable.
Hence three settings rather than a veto: the safe reading stays the default, and the other one stays
reachable.

`Π` and the cumulative energy carry no such condition and are unaffected by this choice.
"""
abstract type AbstractSpectrumPolicy end

"""
    StrictSpectrum <: AbstractSpectrumPolicy

Refuse to produce a spectral density for a kernel that fails [`Kernels.transfer_monotone`](@ref). The
default: either a density guaranteed non-negative, or an error naming the alternatives.
"""
struct StrictSpectrum <: AbstractSpectrumPolicy end

"""
    ForceSpectrum <: AbstractSpectrumPolicy

Compute the density regardless, warning once per session. The caller owns checking its sign — the right
setting when the kernel's non-monotone band sits outside the range of scales being interpreted.
"""
struct ForceSpectrum <: AbstractSpectrumPolicy end

"""
    NoSpectrum <: AbstractSpectrumPolicy

Skip the density entirely; the field is filled with `NaN` rather than a number that would read as
computed. `Π` and the cumulative energy are still produced.
"""
struct NoSpectrum <: AbstractSpectrumPolicy end

"""
    gate_spectrum(kernel, policy) -> Bool

Apply `policy` to `kernel`, returning whether a spectral density should be computed. Throws under
[`StrictSpectrum`](@ref) for a non-monotone kernel; warns once under [`ForceSpectrum`](@ref).
"""
function gate_spectrum end

gate_spectrum(kernel, ::StrictSpectrum) = (Kernels.check_spectrum_kernel(kernel); true)
gate_spectrum(::Any, ::NoSpectrum) = false

function gate_spectrum(kernel, ::ForceSpectrum)
    Kernels.transfer_monotone(kernel) || @warn(
        "ForceSpectrum with $(nameof(typeof(kernel))): its |Ĝ(k)|² is not monotone decreasing, so " *
        "Sadek & Aluie (2018) eq. (21) does not apply and the filtering spectral density is not " *
        "guaranteed non-negative. Check its sign before interpreting it.",
        maxlog = 1,
    )
    return true
end

"""
    ΠWorkspace{T, A}

Pre-allocated arrays for computing cross-scale energy flux Π to avoid heap allocations in scale loops.
"""
struct ΠWorkspace{T<:AbstractFloat, A<:AbstractArray{T}}
    # Filtered velocity components (local coordinates)
    u_filt::A
    v_filt::A
    w_filt::A

    # Planetary Cartesian velocities (if Spherical geometry is used)
    ux::A
    uy::A
    uz::A
    ux_filt::A
    uy_filt::A
    uz_filt::A

    # Filtered quadratic velocity products (planetary Cartesian if Spherical, else Cartesian)
    uu_filt::A
    uv_filt::A
    uw_filt::A
    vv_filt::A
    vw_filt::A
    ww_filt::A

    # Velocity derivatives / strain rate components (local coordinates)
    S_xx::A
    S_xy::A
    S_xz::A
    S_yy::A
    S_yz::A
    S_zz::A

    # Subfilter-scale stress components (local coordinates)
    τ_xx::A
    τ_xy::A
    τ_xz::A
    τ_yy::A
    τ_yz::A
    τ_zz::A

    # Three scratch arrays, so `filter_apply_batch!` can filter all 6 quadratic velocity products in
    # one pass. The spherical branch needs 6 simultaneous pre-filter buffers and only 3 are idle at
    # that point (`u_filt`/`v_filt`/`w_filt`, before the planetary→local transform overwrites them).
    scratch::A
    scratch2::A
    scratch3::A
end

# Workspace constructor based on grid structure and float type. `A` is inferred from what
# `zeros(T, sz...)` actually produces (Vector for a 1D grid, Matrix for 2D, Array{T,3} for 3D) —
# NOT hardcoded, since a 1D/3D grid's `sz` is a 1- or 3-tuple, not always 2D.
"""
    ΠWorkspace(grid)
    ΠWorkspace(grid, batch_size)

Scratch for a flux computation. Passing `batch_size` sizes every buffer as `(spatial..., batch...)` so a
whole batch of slices is held at once; the elementwise algebra then broadcasts over the trailing axes
unchanged, and a filter apply can cover the batch in one pass instead of one per slice.
"""
function ΠWorkspace(
    grid::FlowGeometries.Grids.AbstractGrid{G,T}, batch_size::Tuple = (),
) where {G, T<:AbstractFloat}
    sz = (FlowGeometries.Grids.size_tuple(grid)..., batch_size...)

    u_filt  = zeros(T, sz...)
    v_filt  = zeros(T, sz...)
    w_filt  = zeros(T, sz...)

    ux      = zeros(T, sz...)
    uy      = zeros(T, sz...)
    uz      = zeros(T, sz...)
    ux_filt = zeros(T, sz...)
    uy_filt = zeros(T, sz...)
    uz_filt = zeros(T, sz...)

    uu_filt = zeros(T, sz...)
    uv_filt = zeros(T, sz...)
    uw_filt = zeros(T, sz...)
    vv_filt = zeros(T, sz...)
    vw_filt = zeros(T, sz...)
    ww_filt = zeros(T, sz...)

    S_xx    = zeros(T, sz...)
    S_xy    = zeros(T, sz...)
    S_xz    = zeros(T, sz...)
    S_yy    = zeros(T, sz...)
    S_yz    = zeros(T, sz...)
    S_zz    = zeros(T, sz...)

    τ_xx    = zeros(T, sz...)
    τ_xy    = zeros(T, sz...)
    τ_xz    = zeros(T, sz...)
    τ_yy    = zeros(T, sz...)
    τ_yz    = zeros(T, sz...)
    τ_zz    = zeros(T, sz...)

    scratch = zeros(T, sz...)
    scratch2 = zeros(T, sz...)
    scratch3 = zeros(T, sz...)

    return ΠWorkspace(
        u_filt, v_filt, w_filt,
        ux, uy, uz, ux_filt, uy_filt, uz_filt,
        uu_filt, uv_filt, uw_filt, vv_filt, vw_filt, ww_filt,
        S_xx, S_xy, S_xz, S_yy, S_yz, S_zz,
        τ_xx, τ_xy, τ_xz, τ_yy, τ_yz, τ_zz,
        scratch, scratch2, scratch3
    )
end

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
- `Π`: Energy flux array (same as input), units of W/m³

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
#
# The GRID's rank is pinned and the arrays' is not, which is what lets a field carry trailing batch axes:
# `(Nx, Ny, Nb)` here is a batch of 2-D slices, while the same shape against a rank-3 grid is genuine
# volumetric data and takes the true-3D method. Binding the array rank instead — as this once did — makes
# a batch unrepresentable, and binding neither makes the two indistinguishable.
#
# Nothing below needs to know about the batch: the filter applies route themselves to a fused pass, the
# tensor algebra is elementwise so it broadcasts over trailing axes, and the stencil derivatives carry the
# extra axes through.
function compute_Π!(
    Π::AbstractArray{T},
    u::AbstractArray,
    v::AbstractArray,
    w::Union{Nothing, AbstractArray}, # nothing or zeros for 2D
    grid::FlowGeometries.Grids.StructuredGrid{G,T,2},
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
    _validate_field_sizes(grid, Π, u, v, w)
    ws = workspace === nothing ? ΠWorkspace(grid, _batch_dims(Π, grid)) : workspace
    # One plan for this scale, shared by all ~9 filterings below. A caller repeating this at a fixed
    # scale can pass a prebuilt `filter_plan` to share it across calls as well.
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend) : filter_plan
    # The stencil weights depend only on the grid, so one table serves every derivative here and every
    # later call at any scale.
    dplan = deriv_plan === nothing ? Derivatives.StencilPlan(grid) : deriv_plan
    return _compute_Π!(Π, u, v, w, grid, ws, plan, dplan, analyzed)
end

# ---------------------------------------------------------------------------
# Shared 2D driver for the per-point tensor physics (rotation / SFS stress / strain contraction),
# called by BOTH the StructuredGrid and CurvilinearGrid `compute_Π!` methods. Both grids reach their
# geometry only through `FlowGeometries.Grids.coords`/`isactive` and the `ddx!`/`ddy!` operators, so this one kernel
# serves both — no duplicated tensor math. `deriv_plan` is a `Derivatives.StencilPlan` for a
# `StructuredGrid` and a `Connectivity.gradient_plan` for a curvilinear grid or a node set; either way
# it is geometry only, so one holds for every scale.
# ---------------------------------------------------------------------------

# Both tangent components of one field — which is what the strain tensor needs of every field it
# touches. A separable grid differences each direction independently; a curvilinear grid or a node set
# has no direction to difference along and fits both components at once from the same neighbour sweep,
# so asking for them together is one traversal there rather than two.
@inline function _grad2!(g1, g2, f, grid::FlowGeometries.Grids.StructuredGrid, ::Nothing)
    Derivatives.ddx!(g1, f, grid)
    Derivatives.ddy!(g2, f, grid)
    return nothing
end
@inline function _grad2!(g1, g2, f, grid::FlowGeometries.Grids.StructuredGrid, plan::Derivatives.StencilPlan)
    Derivatives.ddx!(g1, f, grid, plan)
    Derivatives.ddy!(g2, f, grid, plan)
    return nothing
end
@inline function _grad2!(g1, g2, f, _grid, plan::FlowGeometries.Discretization.GradientPlan)
    FlowGeometries.Discretization.gradient!(g1, g2, f, plan)
    return nothing
end

# Rotate a planetary-Cartesian symmetric stress to the local (east, north, radial) frame at (λ,φ),
# flattened to the component order this file's contractions read. A caller with no radial velocity
# passes zero for `txz`/`tyz`/`tzz` and discards `τer`/`τnr`/`τrr`.
@inline function _rotate_stress_to_local_enr(
    geo::FlowGeometries.Geometry.AbstractSphericalGeometry,
    txx::T, txy::T, txz::T, tyy::T, tyz::T, tzz::T, λ::T, φ::T,
) where {T<:AbstractFloat}
    τ = FlowGeometries.Geometry.tensor_to_local(geo, txx, tyy, tzz, txy, txz, tyz, λ, φ)
    return τ.λλ, τ.λφ, τ.λr, τ.φφ, τ.φr, τ.rr
end

# Symmetric SFS tensor contraction S̄_ij τ_ij — the scalar sum shared by every `compute_Π!` driver's
# final step (2D contraction, or the full six-term 3D contraction when a vertical/radial component
# exists). Factored out so `_compute_Π!` and `_compute_Π!` share the identical arithmetic.
@inline _sfs_contraction(Sxx::T, Sxy::T, Syy::T, τxx::T, τxy::T, τyy::T) where {T<:AbstractFloat} =
    Sxx * τxx + T(2) * Sxy * τxy + Syy * τyy

@inline _sfs_contraction(
    Sxx::T, Sxy::T, Sxz::T, Syy::T, Syz::T, Szz::T, τxx::T, τxy::T, τxz::T, τyy::T, τyz::T, τzz::T,
) where {T<:AbstractFloat} =
    Sxx * τxx + T(2) * Sxy * τxy + Syy * τyy + T(2) * Sxz * τxz + T(2) * Syz * τyz + Szz * τzz

# One driver for every point-indexed grid: the broadcasts are shape-agnostic and the explicit loops
# run over `CartesianIndices`, so a node-indexed `UnstructuredGrid` and an `(i,j)` 2D grid take the
# same code. The true-3D methods below are separate because their physics differs — real radial
# derivatives and curvature terms this 2.5D path drops by construction.
function _compute_Π!(
    Π::AbstractArray{T},
    u::AbstractArray,
    v::AbstractArray,
    w::Union{Nothing, AbstractArray},
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    ws::ΠWorkspace,
    plan::Filtering.AbstractFilterPlan,
    deriv_plan,
    analyzed = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    has_w = w !== nothing

    if G <: FlowGeometries.Geometry.CartesianGeometry{T}
        # -------------------------------------------------------------------
        # Cartesian Case
        # -------------------------------------------------------------------
        # Filter velocity components — batched: one neighbour-list/weight derivation per point,
        # applied to all primitives at once (see `Filtering.filter_apply_batch!`), not once per field.
        if analyzed !== nothing
            # A sweep hoists the scale-independent forward transform out of the scale loop, so only the
            # per-scale synthesis is left here.
            _synthesize_all!(_velocity_outs(ws, has_w), analyzed.velocity, plan)
        elseif has_w
            Filtering.filter_apply_batch!((ws.u_filt, ws.v_filt, ws.w_filt), (u, v, w), plan)
        else
            Filtering.filter_apply_batch!((ws.u_filt, ws.v_filt), (u, v), plan)
        end

        # Filter products: u², uv, vv, etc. — also batched, in one pass. `ux`/`uy`/`uz`/`ux_filt`/
        # `uy_filt`/`uz_filt` are spherical-only fields, genuinely idle in this Cartesian branch, so
        # they're reused here purely as pre-filter product scratch (no new workspace fields needed).
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

        # Compute subfilter stresses: τ_ij = [u_i u_j]̄ - ū_i ū_j
        @. ws.τ_xx = ws.uu_filt - ws.u_filt * ws.u_filt
        @. ws.τ_xy = ws.uv_filt - ws.u_filt * ws.v_filt
        @. ws.τ_yy = ws.vv_filt - ws.v_filt * ws.v_filt
        if has_w
            @. ws.τ_xz = ws.uw_filt - ws.u_filt * ws.w_filt
            @. ws.τ_yz = ws.vw_filt - ws.v_filt * ws.w_filt
            @. ws.τ_zz = ws.ww_filt - ws.w_filt * ws.w_filt
        end

        # Strain rate: S̄_ij = 0.5 * (∂ū_i/∂x_j + ∂ū_j/∂x_i). One gradient per velocity component.
        _grad2!(ws.S_xx, ws.S_xy, ws.u_filt, grid, deriv_plan)     # ∂ū/∂x, ∂ū/∂y
        _grad2!(ws.scratch, ws.S_yy, ws.v_filt, grid, deriv_plan)  # ∂v̄/∂x, ∂v̄/∂y
        @. ws.S_xy = T(0.5) * (ws.S_xy + ws.scratch)

        if has_w
            # S_xz = 0.5 * (∂ū/∂z + ∂w̄/∂x), S_yz = 0.5 * (∂v̄/∂z + ∂w̄/∂y); ∂/∂z is zero for a
            # level stack, leaving the horizontal gradient of w̄.
            _grad2!(ws.S_xz, ws.S_yz, ws.w_filt, grid, deriv_plan)
            @. ws.S_xz = T(0.5) * ws.S_xz
            @. ws.S_yz = T(0.5) * ws.S_yz

            # S_zz = ∂w̄/∂z = 0 (for standard 2.5D datasets)
            fill!(ws.S_zz, zero(T))
        end

    else
        # -------------------------------------------------------------------
        # Spherical Case (Aluie 2019 commutativity formulation)
        # -------------------------------------------------------------------
        # Transform local coordinates (u_east, v_north) to global Cartesian (u_X, u_Y, u_Z)
        for I in CartesianIndices(Π)
            let i = Tuple(I)
                if FlowGeometries.Grids.isactive(grid, i...)
                    λ, φ = FlowGeometries.Grids.coords(grid, i...)
                    u_val = u[I]
                    v_val = v[I]
                    w_val = has_w ? w[I] : zero(T)

                    p_vel = FlowGeometries.Geometry.vector_to_cartesian(FlowGeometries.Grids.grid_geometry(grid), u_val, v_val, w_val, λ, φ)
                    ws.ux[I] = p_vel[1]
                    ws.uy[I] = p_vel[2]
                    ws.uz[I] = p_vel[3]
                else
                    ws.ux[I] = zero(T)
                    ws.uy[I] = zero(T)
                    ws.uz[I] = zero(T)
                end
            end
        end

        # Filter planetary Cartesian components — batched (one derivation per point, not one per field).
        Filtering.filter_apply_batch!((ws.ux_filt, ws.uy_filt, ws.uz_filt), (ws.ux, ws.uy, ws.uz), plan)

        # Filter planetary products: X-X, X-Y, X-Z, Y-Y, Y-Z, Z-Z — also batched, in one pass. The 6
        # pre-filter product buffers reuse `u_filt`/`v_filt`/`w_filt` (genuinely idle here — the
        # "transform back to local coordinates" step below overwrites them with real values right
        # after, so nothing reads their stale product-scratch content) plus `scratch`/`scratch2`/
        # `scratch3` (the extra scratch fields added specifically so this fits in one batch).
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

        # Transform filtered planetary velocities back to local coordinates (u_filt, v_filt, w_filt)
        for I in CartesianIndices(Π)
            let i = Tuple(I)
                if FlowGeometries.Grids.isactive(grid, i...)
                    λ, φ = FlowGeometries.Grids.coords(grid, i...)
                    l_vel = FlowGeometries.Geometry.vector_from_cartesian(FlowGeometries.Grids.grid_geometry(grid), ws.ux_filt[I], ws.uy_filt[I], ws.uz_filt[I], λ, φ)
                    ws.u_filt[I] = l_vel[1]
                    ws.v_filt[I] = l_vel[2]
                    ws.w_filt[I] = l_vel[3]
                else
                    ws.u_filt[I] = zero(T)
                    ws.v_filt[I] = zero(T)
                    ws.w_filt[I] = zero(T)
                end
            end
        end

        # Transform planetary filtered products to local stresses at each grid point, via the shared
        # `_rotate_stress_to_local_enr` scalar kernel (τ_local = R' * ( [u_i u_j]̄ - ū_i ū_j ) * R for
        # the orthogonal local rotation R = [e_east, e_north, e_radial]) — see that function for the
        # rotation algebra itself, kept in one place so the 1D `UnstructuredGrid` driver below shares it.
        # `txz`/`tyz`/`tzz` are not gated on `has_w`: the planetary Cartesian Z component is nonzero
        # even for a purely horizontal velocity, since that rotates into Z through cosφ/sinφ. Those
        # cross terms feed the rotated τee/τen/τnn. Only τer/τnr/τrr are genuinely radial.
        geo = FlowGeometries.Grids.grid_geometry(grid)
        for I in CartesianIndices(Π)
            let i = Tuple(I)
                if FlowGeometries.Grids.isactive(grid, i...)
                    λ, φ = FlowGeometries.Grids.coords(grid, i...)
                    txx = ws.uu_filt[I] - ws.ux_filt[I] * ws.ux_filt[I]
                    txy = ws.uv_filt[I] - ws.ux_filt[I] * ws.uy_filt[I]
                    tyy = ws.vv_filt[I] - ws.uy_filt[I] * ws.uy_filt[I]
                    txz = ws.uw_filt[I] - ws.ux_filt[I] * ws.uz_filt[I]
                    tyz = ws.vw_filt[I] - ws.uy_filt[I] * ws.uz_filt[I]
                    tzz = ws.ww_filt[I] - ws.uz_filt[I] * ws.uz_filt[I]
                    τee, τen, τer, τnn, τnr, τrr = _rotate_stress_to_local_enr(geo, txx, txy, txz, tyy, tyz, tzz, λ, φ)
                    ws.τ_xx[I] = τee
                    ws.τ_yy[I] = τnn
                    ws.τ_xy[I] = τen
                    if has_w
                        ws.τ_xz[I] = τer
                        ws.τ_yz[I] = τnr
                        ws.τ_zz[I] = τrr
                    end
                else
                    ws.τ_xx[I] = zero(T)
                    ws.τ_yy[I] = zero(T)
                    ws.τ_xy[I] = zero(T)
                    if has_w
                        ws.τ_xz[I] = zero(T)
                        ws.τ_yz[I] = zero(T)
                        ws.τ_zz[I] = zero(T)
                    end
                end
            end
        end

        # Compute Spherical Strain Rates (with geometry curvature correction terms)
        # S_ee = 1/(R cosφ) ∂ū_e/∂λ − v̄_n sinφ/(R cosφ);  S_nn = 1/R ∂v̄_n/∂φ
        # S_en = 0.5 ( 1/(R cosφ) ∂v̄_n/∂λ + 1/R ∂ū_e/∂φ + ū_e sinφ/(R cosφ) )
        _grad2!(ws.S_xx, ws.S_xy, ws.u_filt, grid, deriv_plan)
        _grad2!(ws.scratch, ws.S_yy, ws.v_filt, grid, deriv_plan)

        R = FlowGeometries.Geometry.radius(geo)
        for I in CartesianIndices(Π)
            let i = Tuple(I)
                if FlowGeometries.Grids.isactive(grid, i...)
                    _, φ = FlowGeometries.Grids.coords(grid, i...)
                    sinφ, cosφ = sincos(φ)
                    tan_fact = abs(cosφ) > T(1e-12) ? sinφ / (R * cosφ) : zero(T)
                    ws.S_xx[I] -= ws.v_filt[I] * tan_fact                    # S_ee correction
                    ws.S_xy[I] = T(0.5) * (ws.S_xy[I] + ws.scratch[I] + ws.u_filt[I] * tan_fact)  # S_en
                end
            end
        end

        if has_w
            # S_er = 0.5 (∂ū_e/∂r + 1/(R cosφ) ∂w̄/∂λ) and S_nr = 0.5 (∂v̄_n/∂r + 1/R ∂w̄/∂φ); with
            # vertically flat layers ∂/∂r drops and each is half the horizontal gradient of w̄.
            _grad2!(ws.S_xz, ws.S_yz, ws.w_filt, grid, deriv_plan)
            @. ws.S_xz = T(0.5) * ws.S_xz
            @. ws.S_yz = T(0.5) * ws.S_yz

            # S_rr = ∂w̄/∂r = 0
            fill!(ws.S_zz, zero(T))
        end
    end

    # -----------------------------------------------------------------------
    # Tensor contraction: Π = -Σ_ij S̄_ij τ_ij
    # -----------------------------------------------------------------------
    # Since stress & strain rates are symmetric:
    # S̄_ij τ_ij = S_xx*τ_xx + 2*S_xy*τ_xy + S_yy*τ_yy (2D)
    # S̄_ij τ_ij = S_xx*τ_xx + 2*S_xy*τ_xy + S_yy*τ_yy + 2*S_xz*τ_xz + 2*S_yz*τ_yz + S_zz*τ_zz (3D)
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


"""
    compute_Π!(Π, u, v, w, grid::UnstructuredGrid, kernel, scale; workspace=nothing, deriv_plan=nothing, backend=AutoBackend(), mask_strategy=ZeroFill(), method=Spectral())

Cross-scale kinetic energy flux Π = -S̄_ij τ_ij on a `FlowGeometries.Grids.UnstructuredGrid` (scattered
points, node-indexed) — the same physics as the 2D methods (planetary-Cartesian rotation for
spherical geometry), via `_compute_Π!`. The resolved strain uses the node-indexed WLSQ
gradient (`Connectivity.gradient_plan` + `Discretization.gradient!`). `method` defaults to
`Spectral()` here, unlike the other grid types' `RealSpace()` default: the transform is exact for a
band-limited field and its per-apply cost does not grow with the filter scale. `RealSpace()` applies
the kernel as written, with compact support; a transform's support is global.
"""
function compute_Π!(
    Π::AbstractVector{T},
    u::AbstractVector,
    v::AbstractVector,
    w::Union{Nothing, AbstractVector},
    grid::FlowGeometries.Grids.UnstructuredGrid{T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    workspace::Union{Nothing, ΠWorkspace} = nothing,
    deriv_plan::Union{Nothing, FlowGeometries.Discretization.GradientPlan} = nothing,
    filter_plan::Union{Nothing, Filtering.AbstractFilterPlan} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Filtering.AbstractFilterMethod = Filtering.Spectral(),
    analyzed = nothing,
) where {T<:AbstractFloat}
    _validate_field_sizes(grid, Π, u, v, w)
    ws = workspace === nothing ? ΠWorkspace(grid) : workspace
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend, method=method) : filter_plan
    dplan = deriv_plan === nothing ? FlowGeometries.Connectivity.gradient_plan(grid) : deriv_plan
    return _compute_Π!(Π, u, v, w, grid, ws, plan, dplan, analyzed)
end

"""
    compute_Π!(Π, u, v, w, grid::CurvilinearGrid, kernel, scale; workspace=nothing, deriv_plan=nothing, backend=AutoBackend(), mask_strategy=ZeroFill())

Cross-scale kinetic energy flux Π = -S̄_ij τ_ij on a `FlowGeometries.Grids.CurvilinearGrid`. Identical
physics to the `StructuredGrid` 2D method — it shares the same `_compute_Π!` tensor kernel
— but the resolved strain uses the least-squares tangent-plane gradient
(`Discretization.gradient!` over a `Connectivity.gradient_plan`, both components from one neighbour
sweep) and real-space filtering uses the scattered per-point footprint. Pass a prebuilt
`deriv_plan = FG.Connectivity.gradient_plan(grid)` (and a reusable `workspace`) to avoid rebuilding them per call
across a scale sweep.
"""
function compute_Π!(
    Π::AbstractMatrix{T},
    u::AbstractMatrix,
    v::AbstractMatrix,
    w::Union{Nothing, AbstractMatrix},
    grid::FlowGeometries.Grids.CurvilinearGrid{T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    workspace::Union{Nothing, ΠWorkspace} = nothing,
    deriv_plan::Union{Nothing, FlowGeometries.Discretization.GradientPlan} = nothing,
    filter_plan::Union{Nothing, Filtering.AbstractFilterPlan} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    analyzed = nothing,
) where {T<:AbstractFloat}
    _validate_field_sizes(grid, Π, u, v, w)
    ws = workspace === nothing ? ΠWorkspace(grid) : workspace
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend) : filter_plan
    dplan = deriv_plan === nothing ? FlowGeometries.Connectivity.gradient_plan(grid) : deriv_plan
    return _compute_Π!(Π, u, v, w, grid, ws, plan, dplan, analyzed)
end

"""
    compute_Π!(Π::AbstractArray{T,3}, u, v, w, grid::StructuredGrid{Cartesian,T,3}, kernel, scale; mask_strategy=ZeroFill(), backend=AutoBackend())

Full **three-dimensional** Cartesian cross-scale energy flux Π = -S̄_ij τ_ij with all nine strain
components (the diagonal `S_zz = ∂w̄/∂z` and the off-diagonals `S_xz, S_yz` carry genuine vertical
derivatives, unlike the 2.5D layer-by-layer path). The 3D grid carries a 3D mask, so masked cells are
handled per-cell in all three directions.

The contraction is the symmetric six-term sum
`S̄:τ = S_xx τ_xx + S_yy τ_yy + S_zz τ_zz + 2(S_xy τ_xy + S_xz τ_xz + S_yz τ_yz)`.

Dispatched on a 3D output array + 3D Cartesian grid (the 2D method takes an `AbstractMatrix`); see
the separate `StructuredGrid{Spherical,T,3}` method below for the spherical volumetric case (genuine
radius axis, real `∂/∂r`, full curvature-corrected strain). Pass a reusable `workspace`
(a [`ΠWorkspace`](@ref), dimension-generic) to avoid reallocating temporaries on every call — the
same "build once, reuse many" pattern the 2D driver uses, now that `ΠWorkspace` infers its array type
from the grid's actual shape instead of hardcoding `Matrix`.
"""
function compute_Π!(
    Π::AbstractArray{T,3},
    u::AbstractArray{<:Any,3},
    v::AbstractArray{<:Any,3},
    w::AbstractArray{<:Any,3},
    grid::FlowGeometries.Grids.StructuredGrid{G,T,3},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    workspace::Union{Nothing, ΠWorkspace} = nothing,
    filter_plan::Union{Nothing, Filtering.AbstractFilterPlan} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    deriv_plan::Union{Nothing, Derivatives.StencilPlan} = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    _validate_field_sizes(grid, Π, u, v, w)
    ws = workspace === nothing ? ΠWorkspace(grid) : workspace
    dplan = deriv_plan === nothing ? Derivatives.StencilPlan(grid) : deriv_plan
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend) : filter_plan

    # Filtered velocities and the six independent filtered quadratic products — batched (one
    # neighbour-list/weight derivation per point, applied to the whole group at once). `ux`/`uy`/`uz`/
    # `ux_filt`/`uy_filt`/`uz_filt` are spherical-only fields, genuinely idle here, reused purely as
    # pre-filter product scratch (no new workspace fields needed) — same pattern as `_compute_Π!`.
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

    # Subfilter stress τ_ij = ⟨u_i u_j⟩ - ū_i ū_j (symmetric, six components).
    @. ws.τ_xx = ws.uu_filt - ws.u_filt * ws.u_filt
    @. ws.τ_xy = ws.uv_filt - ws.u_filt * ws.v_filt
    @. ws.τ_xz = ws.uw_filt - ws.u_filt * ws.w_filt
    @. ws.τ_yy = ws.vv_filt - ws.v_filt * ws.v_filt
    @. ws.τ_yz = ws.vw_filt - ws.v_filt * ws.w_filt
    @. ws.τ_zz = ws.ww_filt - ws.w_filt * ws.w_filt

    # Strain S̄_ij = ½(∂ū_i/∂x_j + ∂ū_j/∂x_i): three diagonals + three off-diagonals.
    Derivatives.ddx!(ws.S_xx, ws.u_filt, grid, dplan)
    Derivatives.ddy!(ws.S_yy, ws.v_filt, grid, dplan)
    Derivatives.ddz!(ws.S_zz, ws.w_filt, grid, dplan)
    Derivatives.ddy!(ws.S_xy, ws.u_filt, grid, dplan); Derivatives.ddx!(ws.scratch, ws.v_filt, grid, dplan)
    @. ws.S_xy = T(0.5) * (ws.S_xy + ws.scratch)
    Derivatives.ddz!(ws.S_xz, ws.u_filt, grid, dplan); Derivatives.ddx!(ws.scratch, ws.w_filt, grid, dplan)
    @. ws.S_xz = T(0.5) * (ws.S_xz + ws.scratch)
    Derivatives.ddz!(ws.S_yz, ws.v_filt, grid, dplan); Derivatives.ddy!(ws.scratch, ws.w_filt, grid, dplan)
    @. ws.S_yz = T(0.5) * (ws.S_yz + ws.scratch)

    # A loop, not a broadcast: fusing thirteen arrays builds a `Broadcasted` wide enough to spill.
    mask = FlowGeometries.Grids.mask(grid)
    @inbounds for I in CartesianIndices(Π)
        Π[I] = mask[I] ? -_sfs_contraction(
            ws.S_xx[I], ws.S_xy[I], ws.S_xz[I], ws.S_yy[I], ws.S_yz[I], ws.S_zz[I],
            ws.τ_xx[I], ws.τ_xy[I], ws.τ_xz[I], ws.τ_yy[I], ws.τ_yz[I], ws.τ_zz[I],
        ) : zero(T)
    end
    return Π
end

"""
    compute_Π!(Π::AbstractArray{T,3}, u, v, w, grid::StructuredGrid{Spherical,T,3}, kernel, scale; workspace=nothing, backend=AutoBackend(), mask_strategy=ZeroFill())

Full **three-dimensional spherical** cross-scale energy flux Π = -S̄_ij τ_ij: a genuine radius axis
`r[k]` (absolute distance from the planet center — see `FlowGeometries.Grids.StructuredGrid`'s 3D
constructor) and real vertical derivatives `∂/∂r`, unlike the 2.5D layer-by-layer path (which drops
the `u_r/r` curvature terms in `S_ee`/`S_nn` and the `S_er`/`S_nr`/`S_rr` radial strain entirely, since
it has no radial axis to differentiate against).

Velocities are rotated to planetary Cartesian for filtering (Aluie 2019 commutativity), then rotated
back to local (east, north, radial), through the same `_rotate_stress_to_local_enr`/`_sfs_contraction`
kernels the 2D spherical driver uses — that rotation is fully 3×3-general, and the 2.5D caller simply
discards its radial components. What differs here is the strain: the spherical strain-rate tensor in
orthogonal curvilinear
coordinates (scale factors `h_λ = r cosφ, h_φ = r, h_r = 1`),

    S_ee = (1/(r cosφ))∂ū_e/∂λ - v̄_n·tanφ/r + w̄_r/r
    S_nn = (1/r)∂v̄_n/∂φ + w̄_r/r
    S_rr = ∂w̄_r/∂r
    S_en = ½[(1/(r cosφ))∂v̄_n/∂λ + (1/r)∂ū_e/∂φ + ū_e·tanφ/r]
    S_er = ½[(1/(r cosφ))∂w̄_r/∂λ + ∂ū_e/∂r - ū_e/r]
    S_nr = ½[(1/r)∂w̄_r/∂φ + ∂v̄_n/∂r - v̄_n/r]

where `∂/∂λ`/`∂/∂φ`/`∂/∂r` are [`Derivatives.ddx!`](@ref)/[`Derivatives.ddy!`](@ref)/[`Derivatives.ddz!`](@ref) (already
metric-scaled using the LOCAL `r[k]`, not the fixed reference radius). Pass a reusable `workspace`
to avoid reallocating temporaries on every call, exactly as the Cartesian 3D method does.
"""
function compute_Π!(
    Π::AbstractArray{T,3},
    u::AbstractArray{<:Any,3},
    v::AbstractArray{<:Any,3},
    w::AbstractArray{<:Any,3},
    grid::FlowGeometries.Grids.StructuredGrid{G,T,3},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    workspace::Union{Nothing, ΠWorkspace} = nothing,
    filter_plan::Union{Nothing, Filtering.AbstractFilterPlan} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    deriv_plan::Union{Nothing, Derivatives.StencilPlan} = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.SphericalGeometry{T}}
    _validate_field_sizes(grid, Π, u, v, w)
    ws = workspace === nothing ? ΠWorkspace(grid) : workspace
    dplan = deriv_plan === nothing ? Derivatives.StencilPlan(grid) : deriv_plan
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend) : filter_plan
    Nx, Ny, Nr = FlowGeometries.Grids.size_tuple(grid)

    # Rotate local (east, north, radial) velocity to planetary Cartesian at each point.
    @inbounds for k in 1:Nr, j in 1:Ny, i in 1:Nx
        if FlowGeometries.Grids.isactive(grid, i, j, k)
            λ, φ, _ = FlowGeometries.Grids.coords(grid, i, j, k)
            p_vel = FlowGeometries.Geometry.vector_to_cartesian(FlowGeometries.Grids.grid_geometry(grid), u[i, j, k], v[i, j, k], w[i, j, k], λ, φ)
            ws.ux[i, j, k] = p_vel[1]; ws.uy[i, j, k] = p_vel[2]; ws.uz[i, j, k] = p_vel[3]
        else
            ws.ux[i, j, k] = zero(T); ws.uy[i, j, k] = zero(T); ws.uz[i, j, k] = zero(T)
        end
    end

    Filtering.filter_apply_batch!((ws.ux_filt, ws.uy_filt, ws.uz_filt), (ws.ux, ws.uy, ws.uz), plan)
    # Pre-filter product scratch reuses `u_filt`/`v_filt`/`w_filt` (idle until the "rotate back to
    # local" loop below overwrites them with real values) plus `scratch`/`scratch2`/`scratch3`.
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

    # Rotate filtered planetary velocities back to local (east, north, radial).
    @inbounds for k in 1:Nr, j in 1:Ny, i in 1:Nx
        if FlowGeometries.Grids.isactive(grid, i, j, k)
            λ, φ, _ = FlowGeometries.Grids.coords(grid, i, j, k)
            l_vel = FlowGeometries.Geometry.vector_from_cartesian(
                FlowGeometries.Grids.grid_geometry(grid), ws.ux_filt[i, j, k], ws.uy_filt[i, j, k], ws.uz_filt[i, j, k], λ, φ,
            )
            ws.u_filt[i, j, k] = l_vel[1]; ws.v_filt[i, j, k] = l_vel[2]; ws.w_filt[i, j, k] = l_vel[3]
        else
            ws.u_filt[i, j, k] = zero(T); ws.v_filt[i, j, k] = zero(T); ws.w_filt[i, j, k] = zero(T)
        end
    end

    # Rotate filtered planetary quadratic products into the local (east,north,radial) stress tensor.
    geo = FlowGeometries.Grids.grid_geometry(grid)
    @inbounds for k in 1:Nr, j in 1:Ny, i in 1:Nx
        if FlowGeometries.Grids.isactive(grid, i, j, k)
            λ, φ, _ = FlowGeometries.Grids.coords(grid, i, j, k)
            txx = ws.uu_filt[i, j, k] - ws.ux_filt[i, j, k] * ws.ux_filt[i, j, k]
            txy = ws.uv_filt[i, j, k] - ws.ux_filt[i, j, k] * ws.uy_filt[i, j, k]
            tyy = ws.vv_filt[i, j, k] - ws.uy_filt[i, j, k] * ws.uy_filt[i, j, k]
            txz = ws.uw_filt[i, j, k] - ws.ux_filt[i, j, k] * ws.uz_filt[i, j, k]
            tyz = ws.vw_filt[i, j, k] - ws.uy_filt[i, j, k] * ws.uz_filt[i, j, k]
            tzz = ws.ww_filt[i, j, k] - ws.uz_filt[i, j, k] * ws.uz_filt[i, j, k]
            τee, τen, τer, τnn, τnr, τrr = _rotate_stress_to_local_enr(geo, txx, txy, txz, tyy, tyz, tzz, λ, φ)
            ws.τ_xx[i, j, k] = τee; ws.τ_xy[i, j, k] = τen; ws.τ_xz[i, j, k] = τer
            ws.τ_yy[i, j, k] = τnn; ws.τ_yz[i, j, k] = τnr; ws.τ_zz[i, j, k] = τrr
        else
            ws.τ_xx[i, j, k] = zero(T); ws.τ_xy[i, j, k] = zero(T); ws.τ_xz[i, j, k] = zero(T)
            ws.τ_yy[i, j, k] = zero(T); ws.τ_yz[i, j, k] = zero(T); ws.τ_zz[i, j, k] = zero(T)
        end
    end

    # Strain: ddx!/ddy!/ddz! are already metric-scaled (1/(r cosφ), 1/r, and a plain radial
    # derivative respectively, using the LOCAL r[k] at each level), so this gives the "flat" part of
    # each component; the curvature-correction terms are added in the loop below.
    Derivatives.ddx!(ws.S_xx, ws.u_filt, grid, dplan)
    Derivatives.ddy!(ws.S_yy, ws.v_filt, grid, dplan)
    Derivatives.ddz!(ws.S_zz, ws.w_filt, grid, dplan)
    Derivatives.ddy!(ws.S_xy, ws.u_filt, grid, dplan); Derivatives.ddx!(ws.scratch, ws.v_filt, grid, dplan)
    @. ws.S_xy = T(0.5) * (ws.S_xy + ws.scratch)
    Derivatives.ddz!(ws.S_xz, ws.u_filt, grid, dplan); Derivatives.ddx!(ws.scratch, ws.w_filt, grid, dplan)
    @. ws.S_xz = T(0.5) * (ws.S_xz + ws.scratch)
    Derivatives.ddz!(ws.S_yz, ws.v_filt, grid, dplan); Derivatives.ddy!(ws.scratch, ws.w_filt, grid, dplan)
    @. ws.S_yz = T(0.5) * (ws.S_yz + ws.scratch)

    @inbounds for k in 1:Nr, j in 1:Ny, i in 1:Nx
        if FlowGeometries.Grids.isactive(grid, i, j, k)
            _, φ, rk = FlowGeometries.Grids.coords(grid, i, j, k)
            sinφ, cosφ = sincos(φ)
            tan_fact = abs(cosφ) > T(1e-12) ? sinφ / (rk * cosφ) : zero(T)
            inv_r = one(T) / rk
            u_e = ws.u_filt[i, j, k]; v_n = ws.v_filt[i, j, k]; w_r = ws.w_filt[i, j, k]
            ws.S_xx[i, j, k] += w_r * inv_r - v_n * tan_fact
            ws.S_yy[i, j, k] += w_r * inv_r
            ws.S_xy[i, j, k] += T(0.5) * u_e * tan_fact
            ws.S_xz[i, j, k] -= T(0.5) * u_e * inv_r
            ws.S_yz[i, j, k] -= T(0.5) * v_n * inv_r
        end
    end

    @inbounds for k in 1:Nr, j in 1:Ny, i in 1:Nx
        Π[i, j, k] = FlowGeometries.Grids.isactive(grid, i, j, k) ? -_sfs_contraction(
            ws.S_xx[i, j, k], ws.S_xy[i, j, k], ws.S_xz[i, j, k],
            ws.S_yy[i, j, k], ws.S_yz[i, j, k], ws.S_zz[i, j, k],
            ws.τ_xx[i, j, k], ws.τ_xy[i, j, k], ws.τ_xz[i, j, k],
            ws.τ_yy[i, j, k], ws.τ_yz[i, j, k], ws.τ_zz[i, j, k],
        ) : zero(T)
    end
    return Π
end

"""
    compute_Π!(Π::AbstractVector, u, grid::StructuredGrid{Cartesian,T,1}, kernel, scale; workspace=nothing, backend=AutoBackend(), mask_strategy=ZeroFill())

1D cross-scale energy flux Π = -S̄_xx τ_xx on a genuinely 1D `StructuredGrid` (a single scalar
velocity component `u` along one axis — the 1D analog of the 2D tensor contraction, which reduces to
a single term since there's only one strain/stress component). Not the 2D-with-singleton-dimension
case (which reuses the 2D methods directly).
"""
function compute_Π!(
    Π::AbstractVector{T},
    u::AbstractVector,
    grid::FlowGeometries.Grids.StructuredGrid{G,T,1},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    workspace::Union{Nothing, ΠWorkspace} = nothing,
    filter_plan::Union{Nothing, Filtering.AbstractFilterPlan} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    deriv_plan::Union{Nothing, Derivatives.StencilPlan} = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    _validate_field_sizes(grid, Π, u)
    ws = workspace === nothing ? ΠWorkspace(grid) : workspace
    dplan = deriv_plan === nothing ? Derivatives.StencilPlan(grid) : deriv_plan
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend) : filter_plan

    @. ws.scratch = u * u
    Filtering.filter_apply_batch!((ws.u_filt, ws.uu_filt), (u, ws.scratch), plan)
    @. ws.τ_xx = ws.uu_filt - ws.u_filt * ws.u_filt

    Derivatives.ddx!(ws.S_xx, ws.u_filt, grid, dplan)

    mask = FlowGeometries.Grids.mask(grid)
    @inbounds @. Π = ifelse(mask, -(ws.S_xx * ws.τ_xx), zero(T))
    return Π
end

# ---------------------------------------------------------------------------
# Filtering Energy Spectrum E(ℓ)
# ---------------------------------------------------------------------------

"""
    active_area(grid) -> T

Total area of the active cells: the denominator of every spatial average here.
"""
function active_area(grid::FlowGeometries.Grids.AbstractGrid{G,T}) where {G, T<:AbstractFloat}
    total = zero(T)
    for I in CartesianIndices(FlowGeometries.Grids.size_tuple(grid))
        FlowGeometries.Grids.isactive(grid, Tuple(I)...) || continue
        total += FlowGeometries.Grids.area(grid, Tuple(I)...)
    end
    total > zero(T) || throw(ArgumentError("grid has no active cells (all masked out)"))
    return total
end

"""
    _area_mean(field, grid, total_area) -> T

Area-weighted mean of `field` over the ACTIVE cells, sharing `active_area`'s denominator so every
spatial average in this module is normalized the same way.
"""
function _area_mean(
    field::AbstractArray{T}, grid::FlowGeometries.Grids.AbstractGrid, total_area::T,
) where {T<:AbstractFloat}
    acc = zero(T)
    @inbounds for I in CartesianIndices(FlowGeometries.Grids.size_tuple(grid))
        t = Tuple(I)
        FlowGeometries.Grids.isactive(grid, t...) || continue
        acc += field[I] * FlowGeometries.Grids.area(grid, t...)
    end
    return acc / total_area
end

"""
    energy_from_filtered(ws, grid, has_w, total_area) -> E(ℓ)

`E(ℓ) = ½⟨|ū_ℓ|²⟩` read from the filtered velocities already in `ws`. [`compute_Π!`](@ref) leaves
exactly those there, so calling this straight after it filters `u`/`v` once per scale rather than
twice.
"""
function energy_from_filtered(
    ws::ΠWorkspace{T}, grid::FlowGeometries.Grids.AbstractGrid, has_w::Bool, total_area::T,
) where {T<:AbstractFloat}
    return _energy_over(ws.u_filt, ws.v_filt, ws.w_filt, grid, has_w, total_area,
                        FlowGeometries.Grids.size_tuple(grid))
end

"""
    energy_from_filtered!(out, ws, grid, has_w, total_area) -> out

Per-slice `E(ℓ)` from a **batched** workspace: `out` holds one energy per slice, shaped like the
workspace's trailing axes.

`E(ℓ)` is a mean over the domain, so unlike everything else on the batch path this reduces the spatial
axes away while leaving the batch axes intact — it cannot simply broadcast. The mask and cell areas are
spatial-only, so each slice reduces against the same geometry.
"""
function energy_from_filtered!(
    out::AbstractArray{T}, ws::ΠWorkspace{T}, grid::FlowGeometries.Grids.AbstractGrid,
    has_w::Bool, total_area::T,
) where {T<:AbstractFloat}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    colons = ntuple(_ -> Colon(), Val(length(gsz)))
    size(out) == size(ws.u_filt)[(length(gsz)+1):end] || throw(DimensionMismatch(
        "out has size $(size(out)); the workspace's batch axes are $(size(ws.u_filt)[(length(gsz)+1):end])",
    ))
    @inbounds for J in CartesianIndices(size(out))
        out[J] = _energy_over(
            view(ws.u_filt, colons..., Tuple(J)...),
            view(ws.v_filt, colons..., Tuple(J)...),
            view(ws.w_filt, colons..., Tuple(J)...),
            grid, has_w, total_area, gsz,
        )
    end
    return out
end

# One slice's domain mean, indexed over the GRID's extent rather than the array's, so it is unaffected by
# any trailing batch axes the caller has already sliced away.
@inline function _energy_over(uf, vf, wf, grid, has_w::Bool, total_area::T, gsz::Tuple) where {T}
    e = zero(T)
    @inbounds for I in CartesianIndices(gsz)
        FlowGeometries.Grids.isactive(grid, Tuple(I)...) || continue
        v2 = uf[I]^2 + vf[I]^2
        has_w && (v2 += wf[I]^2)
        e += v2 * FlowGeometries.Grids.area(grid, Tuple(I)...)
    end
    return T(0.5) * e / total_area
end

"""
    cumulative_energy!(spectrum, u, v, w, grid, kernel, scales; workspace=nothing, backend=AutoBackend(), mask_strategy=ZeroFill())

In-place [`cumulative_energy`](@ref): writes into the caller-supplied `spectrum` vector and, when
`workspace` (a [`ΠWorkspace`](@ref)) is supplied, reuses its `u_filt`/`v_filt`/`w_filt` scratch arrays
instead of allocating fresh ones — the same buffers `compute_Π!` already fills at each scale, so a
`coarse_grain!` sweep pays for this filtered-velocity scratch space once, not twice.
"""
function cumulative_energy!(
    spectrum::AbstractVector{T},
    u::AbstractArray,
    v::AbstractArray,
    w::Union{Nothing, AbstractArray},
    grid::Union{FlowGeometries.Grids.StructuredGrid{G,T}, FlowGeometries.Grids.CurvilinearGrid{T,G}, FlowGeometries.Grids.UnstructuredGrid{T,G}},
    kernel::Kernels.AbstractFilterKernel,
    scales::AbstractVector;
    workspace::Union{Nothing, ΠWorkspace} = nothing,
    filter_plans::Union{Nothing, AbstractVector} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Filtering.AbstractFilterMethod = Filtering.RealSpace(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    size(u) == gsz || throw(DimensionMismatch("u has size $(size(u)), grid expects $gsz"))
    size(v) == gsz || throw(DimensionMismatch("v has size $(size(v)), grid expects $gsz"))
    w === nothing || size(w) == gsz || throw(DimensionMismatch("w has size $(size(w)), grid expects $gsz"))

    Nscales = length(scales)
    length(spectrum) == Nscales || throw(DimensionMismatch(
        "spectrum has length $(length(spectrum)), expected $Nscales (= length(scales))",
    ))

    ws = workspace === nothing ? ΠWorkspace(grid) : workspace
    u_filt, v_filt, w_filt = ws.u_filt, ws.v_filt, ws.w_filt

    # Dimension-generic active-cell iteration: `Tuple(I)...` splats to (i,) for a 1D UnstructuredGrid
    # or (i,j) for a 2D Structured/CurvilinearGrid, matching each grid's own `isactive`/`area` arity.
    idxs = CartesianIndices(u)

    # Precompute total active-cell area for spatial averaging
    total_area = zero(T)
    for I in idxs
        if FlowGeometries.Grids.isactive(grid, Tuple(I)...)
            total_area += FlowGeometries.Grids.area(grid, Tuple(I)...)
        end
    end
    total_area > zero(T) || throw(ArgumentError("grid has no active cells (all masked out)"))

    # Sweep through scales. When the caller (typically `coarse_grain!`, which already builds one
    # plan per scale for its own `compute_Π!` loop) supplies `filter_plans`, reuse those instead of
    # rebuilding the same footprint a second time — otherwise this becomes the dominant allocation in
    # a `coarse_grain!` sweep, since each footprint build costs far more than the rest of the loop body.
    for s_idx in 1:Nscales
        ℓ = T(scales[s_idx])
        plan = filter_plans === nothing ?
            Filtering.plan_filter(grid, kernel, ℓ; mask_strategy=mask_strategy, backend=backend, method=method) :
            filter_plans[s_idx]

        # Filter velocity fields at this scale — batched (one derivation per point, not one per field).
        if w !== nothing
            Filtering.filter_apply_batch!((u_filt, v_filt, w_filt), (u, v, w), plan)
        else
            Filtering.filter_apply_batch!((u_filt, v_filt), (u, v), plan)
        end

        # Compute spatial average specific energy: E(ℓ) = 0.5 * ∫ |ū_ℓ|² dA / ∫ dA
        integrated_energy = zero(T)
        for I in idxs
            if FlowGeometries.Grids.isactive(grid, Tuple(I)...)
                vel2 = u_filt[I]^2 + v_filt[I]^2
                if w !== nothing
                    vel2 += w_filt[I]^2
                end
                integrated_energy += vel2 * FlowGeometries.Grids.area(grid, Tuple(I)...)
            end
        end

        spectrum[s_idx] = T(0.5) * integrated_energy / total_area
    end

    return spectrum
end

"""
    cumulative_energy(u, v, w, grid, kernel, scales; backend=AutoBackend(), mask_strategy=ZeroFill())

Cumulative coarse-grained kinetic energy `E(ℓ) = 0.5 ⟨|ū_ℓ|²⟩` at each filter scale
(Sadek & Aluie 2018, PRF, Eq. 15). This is the CUMULATIVE quantity; the filtering spectral DENSITY
(comparable to a Fourier energy spectrum) is its derivative w.r.t. filtering wavenumber — see
[`Diagnostics.filtering_spectrum`](@ref). Allocates a fresh `spectrum` vector each call; for a repeated sweep
(e.g. inside `coarse_grain!`), call [`cumulative_energy!`](@ref) directly with a reused buffer.

# Examples
```julia
scales = collect(10000.0:10000.0:100000.0)  # 10-100 km
E = cumulative_energy(u, v, nothing, grid, TopHatKernel(), scales)
# E[i] is the cumulative coarse KE at scale scales[i]
```

# References
- Sadek & Aluie (2018), *Phys. Rev. Fluids* 3, 124610 — extracting the spectrum by filtering.
"""
function cumulative_energy(
    u::AbstractArray,
    v::AbstractArray,
    w::Union{Nothing, AbstractArray},
    grid::Union{FlowGeometries.Grids.StructuredGrid{G,T}, FlowGeometries.Grids.CurvilinearGrid{T,G}, FlowGeometries.Grids.UnstructuredGrid{T,G}},
    kernel::Kernels.AbstractFilterKernel,
    scales::AbstractVector;
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Filtering.AbstractFilterMethod = Filtering.RealSpace(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    spectrum = zeros(T, length(scales))
    return cumulative_energy!(spectrum, u, v, w, grid, kernel, scales; backend=backend, mask_strategy=mask_strategy, method=method)
end

"""
    filtering_spectrum(u, v, w, grid, kernel, scales; L=1, backend=AutoBackend(), mask_strategy=ZeroFill())
        -> (k_ℓ, Ẽ)

Filtering spectral DENSITY (Sadek & Aluie 2018, PRF, Eq. 14): the derivative of the cumulative
coarse-grained KE w.r.t. the filtering wavenumber `k_ℓ = L/ℓ`,

    Ẽ(k_ℓ) = d/dk_ℓ [ ½⟨|ū_ℓ|²⟩ ] = -(ℓ²/L) d/dℓ[ ½⟨|ū_ℓ|²⟩ ].

Unlike [`cumulative_energy`](@ref) (the cumulative quantity, Eq. 15), this is the spectral density
comparable to a Fourier energy spectrum. `scales` need not be uniform. Returns the filtering
wavenumbers `k_ℓ` and the density `Ẽ` per scale.

# The `k_ℓ = C/ℓ` convention, and why it must be stated

`L` is the region length, and `k_ℓ = L/ℓ` is the Sadek–Aluie convention: with their Fourier series
`f(x) = Σ_k f̂(k) e^{i(2π/L)k·x}`, `k` is a dimensionless index, so `L` is the domain size. The default
`L = 1` instead gives `k_ℓ = 1/ℓ`, matching Storer et al. (2022, 2023) and FlowSieve. A third
convention, `k_ℓ = 2π/ℓ` (Rivera, Aluie & Ecke 2014), is `L = 2π`.

**The choice rescales the answer.** Under `k_ℓ = C/ℓ` the density carries a Jacobian `dℓ/dk_ℓ =
-ℓ²/C`, so `Ẽ` scales as `1/C` while `k_ℓ` scales as `C`. Comparing amplitudes — or peak locations —
against a Fourier spectrum or against another code is meaningless unless the conventions match. The
cumulative energy [`cumulative_energy`](@ref) is convention-free; only the density is not.

# Limits

- **Slope ceiling.** Sadek & Aluie eq. (18): a kernel with `p` vanishing moments recovers a true
  `k^{-α}` spectrum only for `α < p + 2`, and otherwise saturates at `k^{-(p+2)}`. Both
  `TopHatKernel` and `GaussianKernel` have `p = 1`, so **the measured slope locks at `k⁻³`**. This
  bites hardest in 2-D and QG work, where the enstrophy-range target slope *is* ≈ `k⁻³`. The flux
  `Π` is unaffected — this is a limitation of the spectrum diagnostic alone.
- **Kernel admissibility.** `Ẽ(k_ℓ) ≥ 0` is guaranteed only when `d|Ĝ(k)|²/dk ≤ 0`. By default this
  function throws for a kernel that fails it; pass `policy = ForceSpectrum()` to compute it anyway.
  See [`AbstractSpectrumPolicy`](@ref) and [`Kernels.transfer_monotone`](@ref).

# References
- Sadek & Aluie (2018), *Phys. Rev. Fluids* 3, 124610.
"""
function filtering_spectrum(
    u::AbstractArray,
    v::AbstractArray,
    w::Union{Nothing, AbstractArray},
    grid::Union{FlowGeometries.Grids.StructuredGrid{G,T}, FlowGeometries.Grids.CurvilinearGrid{T,G}, FlowGeometries.Grids.UnstructuredGrid{T,G}},
    kernel::Kernels.AbstractFilterKernel,
    scales::AbstractVector;
    L::Real = one(T),
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Filtering.AbstractFilterMethod = Filtering.RealSpace(),
    policy::AbstractSpectrumPolicy = StrictSpectrum(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    compute = gate_spectrum(kernel, policy)
    cum = cumulative_energy(u, v, w, grid, kernel, scales; backend=backend, mask_strategy=mask_strategy, method=method)
    kℓ = T(L) ./ T.(scales)
    compute || return kℓ, fill(T(NaN), length(kℓ))
    return kℓ, spectral_density(cum, kℓ)
end

"""
    spectral_density!(g, C, k) -> g

In-place [`spectral_density`](@ref): writes the non-uniform finite-difference derivative of `C`
w.r.t. `k` into the caller-supplied `g` (central in the interior, one-sided at the ends). Fills
zeros for fewer than two points.
"""
function spectral_density!(g::AbstractVector{T}, C::AbstractVector{T}, k::AbstractVector) where {T<:AbstractFloat}
    n = length(C)
    length(g) == n || throw(DimensionMismatch("g has length $(length(g)), expected $n (= length(C))"))
    n < 2 && (fill!(g, zero(T)); return g)
    @inbounds for i in 1:n
        if i == 1
            g[i] = (C[2] - C[1]) / (k[2] - k[1])
        elseif i == n
            g[i] = (C[n] - C[n-1]) / (k[n] - k[n-1])
        else
            g[i] = (C[i+1] - C[i-1]) / (k[i+1] - k[i-1])
        end
    end
    return g
end

"""
    spectral_density(C, k) -> dC/dk

Non-uniform finite-difference derivative of cumulative values `C` w.r.t. `k` (central in the
interior, one-sided at the ends). Returns zeros for fewer than two points.
"""
function spectral_density(C::AbstractVector{T}, k::AbstractVector) where {T<:AbstractFloat}
    return spectral_density!(zeros(T, length(C)), C, k)
end

# ---------------------------------------------------------------------------
# Subfilter-stress decomposition (Germano 1992): τ = Leonard + Cross + Reynolds
# ---------------------------------------------------------------------------

"""
    tau_decomposition(u, v, grid, kernel, scale; backend=AutoBackend(), mask_strategy=ZeroFill())
        -> (; L, C, R)

Split the 2D subfilter-scale stress `τ_ij = ⟨u_i u_j⟩ - ū_i ū_j` into Leonard, Cross, and Reynolds
contributions (Germano 1992, *JFM* 238, using generalized central moments so each piece is
individually Galilean-invariant). With `ū = G * u` the filtered velocity and `u' = u - ū` the
residual, and the generalized second moment `M(f, g) = (fg)‾ - f̄ ḡ`:

- Leonard  `L_ij = M(ū_i, ū_j)`            (resolved–resolved),
- Cross    `C_ij = M(ū_i, u'_j) + M(u'_i, ū_j)`,
- Reynolds `R_ij = M(u'_i, u'_j)`          (subfilter–subfilter; backscatter),

with `L + C + R = τ` exactly. Returns a named tuple of named tuples, each holding the symmetric
2D components `(; xx, xy, yy)` as arrays.
"""
function tau_decomposition(
    u::AbstractMatrix,
    v::AbstractMatrix,
    grid::FlowGeometries.Grids.StructuredGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    return tau_decomposition!(TauWorkspace(grid), u, v, grid, kernel, scale;
        backend = backend, mask_strategy = mask_strategy)
end

"""
    TauWorkspace(grid)

Scratch for [`tau_decomposition!`](@ref): the nine output components, the filtered fields and
residuals, and two product buffers. Allocated once and reused, so a repeated decomposition — over
timesteps, or over scales — costs no allocation after the first.
"""
struct TauWorkspace{T<:AbstractFloat, M<:AbstractMatrix{T}}
    ub::M; vb::M; up::M; vp::M
    ubb::M; vbb::M; upb::M; vpb::M
    prod::M; fprod::M; fprod2::M
    Lxx::M; Lxy::M; Lyy::M
    Cxx::M; Cxy::M; Cyy::M
    Rxx::M; Rxy::M; Ryy::M
end

function TauWorkspace(grid::FlowGeometries.Grids.StructuredGrid{G,T}) where {T<:AbstractFloat, G}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    z() = zeros(T, gsz)
    return TauWorkspace(z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z(),
                        z(), z(), z(), z(), z(), z(), z(), z(), z())
end

# Generalized second moment M(a,b) = (ab)‾ - ā b̄, written into `dst` through the shared product and
# filtered-product buffers. A plain function rather than a closure over the workspace: a closure that
# captured these would box them.
@inline function _second_moment!(dst, a, b, fa, fb, prod, fprod, plan)
    @. prod = a * b
    Filtering.filter_apply!(fprod, prod, plan)
    @. dst = fprod - fa * fb
    return dst
end

"""
    tau_decomposition!(ws::TauWorkspace, u, v, grid, kernel, scale; filter_plan=nothing, ...) -> (; L, C, R)

In-place [`tau_decomposition`](@ref). Writes into `ws` and returns views of its component buffers, so
the result is valid until the next call on the same workspace. Supplying `filter_plan` as well makes a
repeated decomposition allocation-free.
"""
function tau_decomposition!(
    ws::TauWorkspace{T},
    u::AbstractMatrix,
    v::AbstractMatrix,
    grid::FlowGeometries.Grids.StructuredGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    filter_plan::Union{Nothing,Filtering.AbstractFilterPlan} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend) :
        filter_plan

    Filtering.filter_apply_batch!((ws.ub, ws.vb), (u, v), plan)      # ū, v̄
    @. ws.up = u - ws.ub                                             # residuals u', v'
    @. ws.vp = v - ws.vb
    Filtering.filter_apply_batch!(                                   # ū̄, v̄̄, ū', v̄'
        (ws.ubb, ws.vbb, ws.upb, ws.vpb), (ws.ub, ws.vb, ws.up, ws.vp), plan,
    )

    _second_moment!(ws.Lxx, ws.ub, ws.ub, ws.ubb, ws.ubb, ws.prod, ws.fprod, plan)
    _second_moment!(ws.Lxy, ws.ub, ws.vb, ws.ubb, ws.vbb, ws.prod, ws.fprod, plan)
    _second_moment!(ws.Lyy, ws.vb, ws.vb, ws.vbb, ws.vbb, ws.prod, ws.fprod, plan)

    _second_moment!(ws.Cxx, ws.ub, ws.up, ws.ubb, ws.upb, ws.prod, ws.fprod, plan)
    @. ws.Cxx *= T(2)
    # The cross term is the sum of both orderings, so the second lands in `fprod2` before adding.
    _second_moment!(ws.Cxy, ws.ub, ws.vp, ws.ubb, ws.vpb, ws.prod, ws.fprod, plan)
    _second_moment!(ws.fprod2, ws.up, ws.vb, ws.upb, ws.vbb, ws.prod, ws.fprod, plan)
    @. ws.Cxy += ws.fprod2
    _second_moment!(ws.Cyy, ws.vb, ws.vp, ws.vbb, ws.vpb, ws.prod, ws.fprod, plan)
    @. ws.Cyy *= T(2)

    _second_moment!(ws.Rxx, ws.up, ws.up, ws.upb, ws.upb, ws.prod, ws.fprod, plan)
    _second_moment!(ws.Rxy, ws.up, ws.vp, ws.upb, ws.vpb, ws.prod, ws.fprod, plan)
    _second_moment!(ws.Ryy, ws.vp, ws.vp, ws.vpb, ws.vpb, ws.prod, ws.fprod, plan)

    return (
        L = (xx = ws.Lxx, xy = ws.Lxy, yy = ws.Lyy),
        C = (xx = ws.Cxx, xy = ws.Cxy, yy = ws.Cyy),
        R = (xx = ws.Rxx, xy = ws.Rxy, yy = ws.Ryy),
    )
end

# The (east, north) block of the same rotation, for callers with no radial component. Inlined, so the
# three unused contractions are dead code the compiler drops.
@inline function _rotate_sym_to_local_en(
    geo::FlowGeometries.Geometry.AbstractSphericalGeometry,
    txx::T, txy::T, txz::T, tyy::T, tyz::T, tzz::T, λ::T, φ::T,
) where {T<:AbstractFloat}
    τ = FlowGeometries.Geometry.tensor_to_local(geo, txx, tyy, tzz, txy, txz, tyz, λ, φ)
    return τ.λλ, τ.λφ, τ.φφ
end

"""
    tau_decomposition(u, v, grid::StructuredGrid{<:SphericalGeometry}, kernel, scale; ...) -> (; L, C, R)

Spherical counterpart of the Cartesian method above: like [`compute_Π!`](@ref)'s spherical branch,
the Leonard/Cross/Reynolds moments are formed in PLANETARY-CARTESIAN coordinates (so filtering
commutes with the moment/residual operations, Aluie 2019), then each of `L`, `C`, `R`'s resulting 3×3
symmetric tensor is rotated back to the local (east, north) frame at every grid point. `L+C+R = τ`
still holds exactly (the rotation is linear). Returns the same `(; L, C, R)` shape as the Cartesian
method — local `(; xx, xy, yy)` (≡ east-east/east-north/north-north) components.
"""
function tau_decomposition(
    u::AbstractMatrix,
    v::AbstractMatrix,
    grid::FlowGeometries.Grids.StructuredGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.SphericalGeometry{T}}
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend)
    flt(f) = (o = zeros(T, size(f)); Filtering.filter_apply!(o, f, plan); o)
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)

    # Local (u,v) -> planetary Cartesian (ux,uy,uz) at every point.
    ux = zeros(T, Nx, Ny); uy = zeros(T, Nx, Ny); uz = zeros(T, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        if FlowGeometries.Grids.isactive(grid, i, j)
            λ, φ = FlowGeometries.Grids.coords(grid, i, j)
            pc = FlowGeometries.Geometry.vector_to_cartesian(FlowGeometries.Grids.grid_geometry(grid), u[i, j], v[i, j], λ, φ)
            ux[i, j], uy[i, j], uz[i, j] = pc[1], pc[2], pc[3]
        end
    end

    uxb = flt(ux); uyb = flt(uy); uzb = flt(uz)             # ū (planetary Cartesian)
    uxp = ux .- uxb; uyp = uy .- uyb; uzp = uz .- uzb        # residuals u'
    uxbb = flt(uxb); uybb = flt(uyb); uzbb = flt(uzb)        # double-filtered ū̄
    uxpb = flt(uxp); uypb = flt(uyp); uzpb = flt(uzp)        # filtered residuals ū'

    # Generalized second moment M(f,g) = (fg)‾ - f̄ ḡ for each planetary-Cartesian tensor component.
    M(f, g, fb, gb) = flt(f .* g) .- fb .* gb
    Lxx = M(uxb, uxb, uxbb, uxbb); Lxy = M(uxb, uyb, uxbb, uybb); Lxz = M(uxb, uzb, uxbb, uzbb)
    Lyy = M(uyb, uyb, uybb, uybb); Lyz = M(uyb, uzb, uybb, uzbb); Lzz = M(uzb, uzb, uzbb, uzbb)
    Cxx = T(2) .* M(uxb, uxp, uxbb, uxpb)
    Cxy = M(uxb, uyp, uxbb, uypb) .+ M(uxp, uyb, uxpb, uybb)
    Cxz = M(uxb, uzp, uxbb, uzpb) .+ M(uxp, uzb, uxpb, uzbb)
    Cyy = T(2) .* M(uyb, uyp, uybb, uypb)
    Cyz = M(uyb, uzp, uybb, uzpb) .+ M(uyp, uzb, uypb, uzbb)
    Czz = T(2) .* M(uzb, uzp, uzbb, uzpb)
    Rxx = M(uxp, uxp, uxpb, uxpb); Rxy = M(uxp, uyp, uxpb, uypb); Rxz = M(uxp, uzp, uxpb, uzpb)
    Ryy = M(uyp, uyp, uypb, uypb); Ryz = M(uyp, uzp, uypb, uzpb); Rzz = M(uzp, uzp, uzpb, uzpb)

    Lee = zeros(T, Nx, Ny); Len = zeros(T, Nx, Ny); Lnn = zeros(T, Nx, Ny)
    Cee = zeros(T, Nx, Ny); Cen = zeros(T, Nx, Ny); Cnn = zeros(T, Nx, Ny)
    Ree = zeros(T, Nx, Ny); Ren = zeros(T, Nx, Ny); Rnn = zeros(T, Nx, Ny)
    geo = FlowGeometries.Grids.grid_geometry(grid)
    for j in 1:Ny, i in 1:Nx
        if FlowGeometries.Grids.isactive(grid, i, j)
            λ, φ = FlowGeometries.Grids.coords(grid, i, j)
            Lee[i,j], Len[i,j], Lnn[i,j] = _rotate_sym_to_local_en(geo, Lxx[i,j], Lxy[i,j], Lxz[i,j], Lyy[i,j], Lyz[i,j], Lzz[i,j], λ, φ)
            Cee[i,j], Cen[i,j], Cnn[i,j] = _rotate_sym_to_local_en(geo, Cxx[i,j], Cxy[i,j], Cxz[i,j], Cyy[i,j], Cyz[i,j], Czz[i,j], λ, φ)
            Ree[i,j], Ren[i,j], Rnn[i,j] = _rotate_sym_to_local_en(geo, Rxx[i,j], Rxy[i,j], Rxz[i,j], Ryy[i,j], Ryz[i,j], Rzz[i,j], λ, φ)
        end
    end
    return (;
        L = (xx = Lee, xy = Len, yy = Lnn),
        C = (xx = Cee, xy = Cen, yy = Cnn),
        R = (xx = Ree, xy = Ren, yy = Rnn),
    )
end

# ---------------------------------------------------------------------------
# Rotational / divergent (Helmholtz) decomposition of the energy flux
# ---------------------------------------------------------------------------

"""
    compute_Π_decomposed(u, v, u_rot, v_rot, grid, kernel, scale; backend=AutoBackend(), mask_strategy=ZeroFill())
        -> (; total, rotational, cross, divergent)

Split the 2D Cartesian cross-scale KE flux Π = -S̄_ij τ_ij into rotational-rotational (Π_RR),
divergent-divergent (Π_DD), and cross/interaction (Π_X — the "stimulated cascade" channel of
Barkan, Srinivasan & McWilliams 2024, JPO) parts, by decomposing **both sides** of the bilinear
contraction, not just the stress.

The Helmholtz decomposition itself is NOT recomputed here — pass the rotational (solenoidal,
divergence-free) part `(u_rot, v_rot)` from a Helmholtz solver (e.g. `HelmholtzDecomposition.jl`); the
divergent (irrotational) part is taken as the complement `(u, v) - (u_rot, v_rot)`. Writing
`u = uʳ + uᵈ`:

  - The strain S̄ is LINEAR in velocity, so it splits with **no cross term**: `S̄ = S̄ʳ + S̄ᵈ`.
  - The stress τ is BILINEAR (quadratic in velocity), so it splits into three pieces:
    `τ(u,u) = τ(uʳ,uʳ) + τ(uᵈ,uᵈ) + [τ(uʳ,uᵈ) + τ(uᵈ,uʳ)] = τʳʳ + τᵈᵈ + τ_X`.

Substituting both splits into `Π = -S̄:τ = -(S̄ʳ+S̄ᵈ):(τʳʳ+τᵈᵈ+τ_X)` and expanding the six resulting
terms into three physically named channels:

    Π_RR = -S̄ʳ:τʳʳ                                        (pure rotational-to-rotational cascade)
    Π_DD = -S̄ᵈ:τᵈᵈ                                        (pure divergent-to-divergent cascade)
    Π_X  = -(S̄ʳ:τᵈᵈ + S̄ᵈ:τʳʳ + S̄ʳ:τ_X + S̄ᵈ:τ_X)          (all rotational/divergent interaction terms)

so the channels sum **exactly** to the total flux, Π = Π_RR + Π_X + Π_DD — each piece constructed
directly (not as a residual), yet the identity holds by the same bilinearity/linearity argument.
Contracting the split stress against the *full*, undecomposed strain S̄ — a one-sided split — is only
correct when S̄ᵈ ≡ 0, and silently wrong whenever the divergent part carries strain of its own.

Returns a named tuple of flux maps (W m⁻³): `rotational` = Π_RR, `divergent` = Π_DD, `cross` = Π_X.
"""
function compute_Π_decomposed(
    u::AbstractMatrix,
    v::AbstractMatrix,
    u_rot::AbstractMatrix,
    v_rot::AbstractMatrix,
    grid::FlowGeometries.Grids.StructuredGrid{G,T},
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
    size(u_rot) == gsz || throw(DimensionMismatch("u_rot has size $(size(u_rot)), grid expects $gsz"))
    size(v_rot) == gsz || throw(DimensionMismatch("v_rot has size $(size(v_rot)), grid expects $gsz"))
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend)
    return compute_Π_decomposed!(
        PiDecomposedWorkspace(grid), u, v, u_rot, v_rot, grid, kernel, scale;
        filter_plan = plan, deriv_plan = dplan,
    )
end

"""
    PiDecomposedWorkspace(grid)

Scratch for [`compute_Π_decomposed!`](@ref): the divergent components, the four filtered means, the
three stress tensors, the two strain tensors, the four flux fields and three shared temporaries.
"""
struct PiDecomposedWorkspace{T<:AbstractFloat, M<:AbstractMatrix{T}}
    u_div::M; v_div::M
    ūr::M; v̄r::M; ūd::M; v̄d::M
    prod::M; fbuf::M; scratch::M
    τRR_xx::M; τRR_xy::M; τRR_yy::M
    τDD_xx::M; τDD_xy::M; τDD_yy::M
    τX_xx::M;  τX_xy::M;  τX_yy::M
    SR_xx::M;  SR_xy::M;  SR_yy::M
    SD_xx::M;  SD_xy::M;  SD_yy::M
    Πrr::M; Πdd::M; Πx::M; total::M
end

function PiDecomposedWorkspace(grid::FlowGeometries.Grids.StructuredGrid{G,T}) where {T<:AbstractFloat, G}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    z() = zeros(T, gsz)
    return PiDecomposedWorkspace(ntuple(_ -> z(), 28)...)
end

# S̄_xx = ∂ā/∂x, S̄_yy = ∂b̄/∂y, S̄_xy = ½(∂ā/∂y + ∂b̄/∂x), from an already-filtered pair, into caller
# buffers. `tmp` is scratch for the second cross derivative.
@inline function _strain_into!(Sxx, Sxy, Syy, ā, b̄, tmp, grid, dplan, ::Type{T}) where {T}
    Derivatives.ddx!(Sxx, ā, grid, dplan)
    Derivatives.ddy!(Syy, b̄, grid, dplan)
    Derivatives.ddy!(Sxy, ā, grid, dplan)
    Derivatives.ddx!(tmp, b̄, grid, dplan)
    @. Sxy = T(0.5) * (Sxy + tmp)
    return nothing
end

"""
    compute_Π_decomposed!(ws, u, v, u_rot, v_rot, grid, kernel, scale; filter_plan=nothing, deriv_plan=nothing, ...)
        -> (; total, rotational, cross, divergent)

In-place [`compute_Π_decomposed`](@ref). Returns views of `ws`'s buffers, valid until the next call on
the same workspace. With `ws` and both plans supplied, a repeated evaluation allocates nothing.
"""
function compute_Π_decomposed!(
    ws::PiDecomposedWorkspace{T},
    u::AbstractMatrix,
    v::AbstractMatrix,
    u_rot::AbstractMatrix,
    v_rot::AbstractMatrix,
    grid::FlowGeometries.Grids.StructuredGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    filter_plan::Union{Nothing,Filtering.AbstractFilterPlan} = nothing,
    deriv_plan::Union{Nothing,Derivatives.StencilPlan} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend) :
        filter_plan
    dplan = deriv_plan === nothing ? Derivatives.StencilPlan(grid) : deriv_plan

    # Divergent (irrotational) part is the complement of the supplied rotational part.
    @. ws.u_div = u - u_rot
    @. ws.v_div = v - v_rot

    # The four filtered means each feed a self stress, the cross stress AND a strain, so they are
    # filtered once. One batch, so a scattered engine derives each neighbourhood once for all four.
    Filtering.filter_apply_batch!(
        (ws.ūr, ws.v̄r, ws.ūd, ws.v̄d), (u_rot, v_rot, ws.u_div, ws.v_div), plan,
    )

    # Self stresses τ(a,a)_ij = ⟨a_i a_j⟩ - ā_i ā_j.
    _second_moment!(ws.τRR_xx, u_rot, u_rot, ws.ūr, ws.ūr, ws.prod, ws.fbuf, plan)
    _second_moment!(ws.τRR_xy, u_rot, v_rot, ws.ūr, ws.v̄r, ws.prod, ws.fbuf, plan)
    _second_moment!(ws.τRR_yy, v_rot, v_rot, ws.v̄r, ws.v̄r, ws.prod, ws.fbuf, plan)
    _second_moment!(ws.τDD_xx, ws.u_div, ws.u_div, ws.ūd, ws.ūd, ws.prod, ws.fbuf, plan)
    _second_moment!(ws.τDD_xy, ws.u_div, ws.v_div, ws.ūd, ws.v̄d, ws.prod, ws.fbuf, plan)
    _second_moment!(ws.τDD_yy, ws.v_div, ws.v_div, ws.v̄d, ws.v̄d, ws.prod, ws.fbuf, plan)

    # Cross stress τ(rot,div) + τ(div,rot). The diagonals are symmetric under the swap so they just
    # double; the off-diagonal genuinely needs both orderings.
    _second_moment!(ws.τX_xx, u_rot, ws.u_div, ws.ūr, ws.ūd, ws.prod, ws.fbuf, plan)
    @. ws.τX_xx *= T(2)
    _second_moment!(ws.τX_yy, v_rot, ws.v_div, ws.v̄r, ws.v̄d, ws.prod, ws.fbuf, plan)
    @. ws.τX_yy *= T(2)
    _second_moment!(ws.τX_xy, u_rot, ws.v_div, ws.ūr, ws.v̄d, ws.prod, ws.fbuf, plan)
    _second_moment!(ws.scratch, ws.u_div, v_rot, ws.ūd, ws.v̄r, ws.prod, ws.fbuf, plan)
    @. ws.τX_xy += ws.scratch

    _strain_into!(ws.SR_xx, ws.SR_xy, ws.SR_yy, ws.ūr, ws.v̄r, ws.scratch, grid, dplan, T)
    _strain_into!(ws.SD_xx, ws.SD_xy, ws.SD_yy, ws.ūd, ws.v̄d, ws.scratch, grid, dplan, T)

    mask = FlowGeometries.Grids.mask(grid)
    @. ws.Πrr = ifelse(mask,
        -(ws.SR_xx * ws.τRR_xx + T(2) * ws.SR_xy * ws.τRR_xy + ws.SR_yy * ws.τRR_yy), zero(T))
    @. ws.Πdd = ifelse(mask,
        -(ws.SD_xx * ws.τDD_xx + T(2) * ws.SD_xy * ws.τDD_xy + ws.SD_yy * ws.τDD_yy), zero(T))
    # Masking is linear, so the four interaction contractions sum inside one `ifelse`.
    @. ws.Πx = ifelse(mask, -(
        ws.SR_xx * ws.τDD_xx + T(2) * ws.SR_xy * ws.τDD_xy + ws.SR_yy * ws.τDD_yy +
        ws.SD_xx * ws.τRR_xx + T(2) * ws.SD_xy * ws.τRR_xy + ws.SD_yy * ws.τRR_yy +
        ws.SR_xx * ws.τX_xx  + T(2) * ws.SR_xy * ws.τX_xy  + ws.SR_yy * ws.τX_yy  +
        ws.SD_xx * ws.τX_xx  + T(2) * ws.SD_xy * ws.τX_xy  + ws.SD_yy * ws.τX_yy
    ), zero(T))
    @. ws.total = ws.Πrr + ws.Πx + ws.Πdd
    return (; total = ws.total, rotational = ws.Πrr, cross = ws.Πx, divergent = ws.Πdd)
end

# ---------------------------------------------------------------------------
# Strain / convergence decomposition of Π (Srinivasan, Barkan & McWilliams 2023, eq. 10)
# ---------------------------------------------------------------------------

"""
    PiStrainWorkspace(grid)

Scratch for [`compute_Π_strain_convergence!`](@ref): the two filtered velocities, the three stress
components, the four velocity-gradient components, the two rotation invariants and the three flux
fields, plus the two shared product buffers.
"""
struct PiStrainWorkspace{T<:AbstractFloat, M<:AbstractMatrix{T}}
    ū::M; v̄::M
    prod::M; fbuf::M
    τuu::M; τuv::M; τvv::M
    ux::M; uy::M; vx::M; vy::M
    δ::M; α::M
    Πα::M; Πδ::M; total::M
end

function PiStrainWorkspace(grid::FlowGeometries.Grids.StructuredGrid{G,T}) where {T<:AbstractFloat, G}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    return PiStrainWorkspace(ntuple(_ -> zeros(T, gsz), 16)...)
end

"""
    compute_Π_strain_convergence!(ws, u, v, grid, kernel, scale; filter_plan=nothing, deriv_plan=nothing, ...)
        -> (; total, strain, convergence, divergence, strain_magnitude)

In-place [`compute_Π_strain_convergence`](@ref). Returns views of `ws`'s buffers, valid until the next
call on the same workspace. With `ws` and both plans supplied, a repeated evaluation allocates nothing.
"""
function compute_Π_strain_convergence!(
    ws::PiStrainWorkspace{T},
    u::AbstractMatrix,
    v::AbstractMatrix,
    grid::FlowGeometries.Grids.StructuredGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    filter_plan::Union{Nothing,Filtering.AbstractFilterPlan} = nothing,
    deriv_plan::Union{Nothing,Derivatives.StencilPlan} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend) :
        filter_plan
    dplan = deriv_plan === nothing ? Derivatives.StencilPlan(grid) : deriv_plan

    Filtering.filter_apply_batch!((ws.ū, ws.v̄), (u, v), plan)
    _second_moment!(ws.τuu, u, u, ws.ū, ws.ū, ws.prod, ws.fbuf, plan)
    _second_moment!(ws.τuv, u, v, ws.ū, ws.v̄, ws.prod, ws.fbuf, plan)
    _second_moment!(ws.τvv, v, v, ws.v̄, ws.v̄, ws.prod, ws.fbuf, plan)

    Derivatives.ddx!(ws.ux, ws.ū, grid, dplan)
    Derivatives.ddy!(ws.uy, ws.ū, grid, dplan)
    Derivatives.ddx!(ws.vx, ws.v̄, grid, dplan)
    Derivatives.ddy!(ws.vy, ws.v̄, grid, dplan)

    mask = FlowGeometries.Grids.mask(grid)
    # δ̄ = ū_x + v̄_y and ᾱ² = σ̄_n² + σ̄_s² are the two rotation invariants of the filtered gradient;
    # σ̄_n = ū_x − v̄_y (normal strain) and σ̄_s = ū_y + v̄_x (shear strain) are not, so they are
    # consumed inline rather than returned.
    @. ws.δ = ifelse(mask, ws.ux + ws.vy, zero(T))
    @. ws.α = ifelse(mask, sqrt((ws.ux - ws.vy)^2 + (ws.uy + ws.vx)^2), zero(T))
    @. ws.Πα = ifelse(mask,
        (ws.τvv - ws.τuu) * (ws.ux - ws.vy) / T(2) - ws.τuv * (ws.uy + ws.vx), zero(T))
    @. ws.Πδ = ifelse(mask, (ws.τvv + ws.τuu) * (ws.ux + ws.vy) / T(2), zero(T))
    @. ws.total = ws.Πα - ws.Πδ
    return (; total = ws.total, strain = ws.Πα, convergence = ws.Πδ,
            divergence = ws.δ, strain_magnitude = ws.α)
end

"""
    compute_Π_strain_convergence(u, v, grid, kernel, scale; backend=AutoBackend(), mask_strategy=ZeroFill())
        -> (; total, strain, convergence, divergence, strain_magnitude)

Split the 2D cross-scale flux into the two production terms of Srinivasan, Barkan & McWilliams (2023),
eq. (10), by diagonalizing the filtered strain tensor. With

```
δ̄   = ū_x + v̄_y            divergence         (rotation invariant)
σ̄_n = ū_x − v̄_y            normal strain
σ̄_s = ū_y + v̄_x            shear strain
ᾱ   = √(σ̄_n² + σ̄_s²)       strain magnitude   (rotation invariant)
```

the flux separates into

```
Π = Π_α − Π_δ ,   Π_α = (τ_vv − τ_uu) σ̄_n/2 − τ_uv σ̄_s ,   Π_δ = (τ_vv + τ_uu) δ̄/2 ,
```

with `Π_α` the **deformation/shear production** — energy transferred by straining, present even in
non-divergent flow — and `Π_δ` the **convergence production**, which vanishes identically for a
non-divergent field and is the term that paper adds. Setting `δ̄ = 0` recovers Polzin (2010); the
equivalent `Π = E′(γᵖ ᾱ − δ̄)` form with `E′ = (τ_vv + τ_uu)/2` recovers Jing et al. (2017), and
`|γᵖ| ≤ 1` gives the bound `|Π_α| ≤ ᾱ E′`.

`Π_α − Π_δ` is **algebraically identical** to the direct `Π = −S̄:τ̄` that [`compute_Π!`](@ref)
computes — expanding eq. (10) collapses to `−τ_uu ū_x − τ_uv(ū_y + v̄_x) − τ_vv v̄_y`. The two are
therefore a genuine cross-check rather than a tautology: they contract different combinations of the
same four derivatives, so a sign or an ordering error in either shows up as a disagreement. The suite
asserts they match to round-off on masked and unmasked grids.

Returns flux maps in W m⁻³, plus the two rotation invariants, which are the natural axes to bin the
flux against (`divergence` = δ̄, `strain_magnitude` = ᾱ).

# References
- Srinivasan, K., Barkan, R., & McWilliams, J. C. (2023). A forward energy flux at submesoscales
  driven by frontogenesis. *J. Phys. Oceanogr.* 53(1), 287–305. doi:10.1175/JPO-D-22-0001.1
"""
function compute_Π_strain_convergence(
    u::AbstractMatrix,
    v::AbstractMatrix,
    grid::FlowGeometries.Grids.StructuredGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    size(u) == gsz || throw(DimensionMismatch("u has size $(size(u)), grid expects $gsz"))
    size(v) == gsz || throw(DimensionMismatch("v has size $(size(v)), grid expects $gsz"))
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend)
    return compute_Π_strain_convergence!(
        PiStrainWorkspace(grid), u, v, grid, kernel, scale;
        filter_plan = plan, deriv_plan = Derivatives.StencilPlan(grid),
    )
end

"""
    compute_Π_decomposed(u, v, w, u_rot, v_rot, w_rot, grid::StructuredGrid{Cartesian,T,3}, kernel, scale; backend=AutoBackend(), mask_strategy=ZeroFill())
        -> (; total, rotational, cross, divergent)

True three-dimensional analog of the 2D [`compute_Π_decomposed`](@ref) above: the same both-sides
(strain AND stress) rotational/divergent split — see that method's docstring for the derivation —
generalized to all six independent strain/stress tensor components, contracted the same way the
true-3D [`compute_Π!`](@ref) method does (nine-term symmetric contraction).
"""
function compute_Π_decomposed(
    u::AbstractArray{<:Any,3},
    v::AbstractArray{<:Any,3},
    w::AbstractArray{<:Any,3},
    u_rot::AbstractArray{<:Any,3},
    v_rot::AbstractArray{<:Any,3},
    w_rot::AbstractArray{<:Any,3},
    grid::FlowGeometries.Grids.StructuredGrid{G,T,3},
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
    size(u_rot) == gsz || throw(DimensionMismatch("u_rot has size $(size(u_rot)), grid expects $gsz"))
    size(v_rot) == gsz || throw(DimensionMismatch("v_rot has size $(size(v_rot)), grid expects $gsz"))
    size(w_rot) == gsz || throw(DimensionMismatch("w_rot has size $(size(w_rot)), grid expects $gsz"))
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend)
    flt(f) = (o = zeros(T, size(f)); Filtering.filter_apply!(o, f, plan); o)

    u_div = u .- u_rot
    v_div = v .- v_rot
    w_div = w .- w_rot

    # Self subfilter stress τ(a,a) for a velocity triple a=(a1,a2,a3): all six components.
    function self_stress(a1, a2, a3)
        ā1 = flt(a1); ā2 = flt(a2); ā3 = flt(a3)
        return (xx = flt(a1 .* a1) .- ā1 .* ā1, xy = flt(a1 .* a2) .- ā1 .* ā2, xz = flt(a1 .* a3) .- ā1 .* ā3,
                yy = flt(a2 .* a2) .- ā2 .* ā2, yz = flt(a2 .* a3) .- ā2 .* ā3, zz = flt(a3 .* a3) .- ā3 .* ā3)
    end
    # Combined cross stress τ(a,b)+τ(b,a) for two DIFFERENT velocity triples a=(a1,a2,a3), b=(b1,b2,b3).
    function cross_stress(a1, a2, a3, b1, b2, b3)
        ā1 = flt(a1); ā2 = flt(a2); ā3 = flt(a3); b̄1 = flt(b1); b̄2 = flt(b2); b̄3 = flt(b3)
        return (
            xx = T(2) .* (flt(a1 .* b1) .- ā1 .* b̄1),
            yy = T(2) .* (flt(a2 .* b2) .- ā2 .* b̄2),
            zz = T(2) .* (flt(a3 .* b3) .- ā3 .* b̄3),
            xy = (flt(a1 .* b2) .- ā1 .* b̄2) .+ (flt(b1 .* a2) .- b̄1 .* ā2),
            xz = (flt(a1 .* b3) .- ā1 .* b̄3) .+ (flt(b1 .* a3) .- b̄1 .* ā3),
            yz = (flt(a2 .* b3) .- ā2 .* b̄3) .+ (flt(b2 .* a3) .- b̄2 .* ā3),
        )
    end

    τ_RR = self_stress(u_rot, v_rot, w_rot)
    τ_DD = self_stress(u_div, v_div, w_div)
    τ_X  = cross_stress(u_rot, v_rot, w_rot, u_div, v_div, w_div)

    # Strain from a velocity triple (a,b,c): three diagonals + three off-diagonals.
    function strain(a, b, c)
        ā = flt(a); b̄ = flt(b); c̄ = flt(c)
        Sxx = similar(ā); Derivatives.ddx!(Sxx, ā, grid, dplan)
        Syy = similar(ā); Derivatives.ddy!(Syy, b̄, grid, dplan)
        Szz = similar(ā); Derivatives.ddz!(Szz, c̄, grid, dplan)
        p = similar(ā); q = similar(ā)
        Derivatives.ddy!(p, ā, grid, dplan); Derivatives.ddx!(q, b̄, grid, dplan)
        Sxy = T(0.5) .* (p .+ q)
        Derivatives.ddz!(p, ā, grid, dplan); Derivatives.ddx!(q, c̄, grid, dplan)
        Sxz = T(0.5) .* (p .+ q)
        Derivatives.ddz!(p, b̄, grid, dplan); Derivatives.ddy!(q, c̄, grid, dplan)
        Syz = T(0.5) .* (p .+ q)
        return (xx = Sxx, xy = Sxy, xz = Sxz, yy = Syy, yz = Syz, zz = Szz)
    end
    S_R = strain(u_rot, v_rot, w_rot)
    S_D = strain(u_div, v_div, w_div)

    mask = FlowGeometries.Grids.mask(grid)
    contract(S, τ) = ifelse.(
        mask,
        -(S.xx .* τ.xx .+ S.yy .* τ.yy .+ S.zz .* τ.zz .+
          T(2) .* (S.xy .* τ.xy .+ S.xz .* τ.xz .+ S.yz .* τ.yz)),
        zero(T),
    )

    Πrr = contract(S_R, τ_RR)
    Πdd = contract(S_D, τ_DD)
    Πx  = contract(S_R, τ_DD) .+ contract(S_D, τ_RR) .+ contract(S_R, τ_X) .+ contract(S_D, τ_X)
    return (; total = Πrr .+ Πx .+ Πdd, rotational = Πrr, cross = Πx, divergent = Πdd)
end

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
    u::AbstractMatrix,
    v::AbstractMatrix,
    θ::AbstractMatrix,
    grid::FlowGeometries.Grids.StructuredGrid{G,T},
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
    size(θ) == gsz || throw(DimensionMismatch("θ has size $(size(θ)), grid expects $gsz"))
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend)

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
struct TracerFluxWorkspace{T<:AbstractFloat, M<:AbstractMatrix{T}}
    ū::M; v̄::M; θ̄::M
    uθ::M; vθ::M
    τx::M; τy::M
    gx::M; gy::M
end

function TracerFluxWorkspace(grid::FlowGeometries.Grids.StructuredGrid{G,T}) where {T<:AbstractFloat, G}
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
    Πθ::AbstractMatrix{T},
    ws::TracerFluxWorkspace{T},
    u::AbstractMatrix,
    v::AbstractMatrix,
    θ::AbstractMatrix,
    grid::FlowGeometries.Grids.StructuredGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    filter_plan::Union{Nothing,Filtering.AbstractFilterPlan} = nothing,
    deriv_plan::Union{Nothing,Derivatives.StencilPlan} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend) :
        filter_plan
    dplan = deriv_plan === nothing ? Derivatives.StencilPlan(grid) : deriv_plan

    @. ws.uθ = u * θ
    @. ws.vθ = v * θ
    Filtering.filter_apply_batch!(
        (ws.ū, ws.v̄, ws.θ̄, ws.τx, ws.τy), (u, v, θ, ws.uθ, ws.vθ), plan,
    )

    # Subfilter tracer flux τ_j = ⟨u_j θ⟩ - ū_j θ̄.
    @. ws.τx -= ws.ū * ws.θ̄
    @. ws.τy -= ws.v̄ * ws.θ̄

    # Resolved tracer gradient ∂_j θ̄.
    Derivatives.ddx!(ws.gx, ws.θ̄, grid, dplan)
    Derivatives.ddy!(ws.gy, ws.θ̄, grid, dplan)

    mask = FlowGeometries.Grids.mask(grid)
    @. Πθ = ifelse(mask, -(ws.τx * ws.gx + ws.τy * ws.gy), zero(T))
    return Πθ
end

# ---------------------------------------------------------------------------
# Favre (density-weighted) coarse-graining — Aluie 2013
# ---------------------------------------------------------------------------

"""
    FavreWorkspace(grid)

Scratch for [`compressible_flux!`](@ref): the filtered density and pressure, the Favre velocities, the
unweighted velocities, the three Favre stress components, the two unweighted mass-flux components, the
four velocity gradients, the two pressure gradients, and the three output fields.
"""
struct FavreWorkspace{T<:AbstractFloat, M<:AbstractMatrix{T}}
    ρ̄::M; P̄::M
    ũ::M; ṽ::M
    ū::M; v̄::M
    τxx::M; τxy::M; τyy::M
    mx::M; my::M          # τ̄(ρ, u_j): the UNWEIGHTED subscale mass flux
    ux::M; uy::M; vx::M; vy::M
    Px::M; Py::M
    prod::M; fbuf::M
    Π::M; Λ::M; PD::M
end

function FavreWorkspace(grid::FlowGeometries.Grids.StructuredGrid{G,T}) where {T<:AbstractFloat, G}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    return FavreWorkspace(ntuple(_ -> zeros(T, gsz), 22)...)
end

"""
    compressible_flux!(ws, u, v, ρ, P, grid, kernel, scale; filter_plan=nothing, deriv_plan=nothing, ...)
        -> (; Π, Λ, pressure_dilatation, ρ̄, P̄, ũ, ṽ)

In-place [`compressible_flux`](@ref). Returns views of `ws`'s buffers, valid until the next call on the
same workspace. With `ws` and both plans supplied, a repeated evaluation allocates nothing.
"""
function compressible_flux!(
    ws::FavreWorkspace{T},
    u::AbstractMatrix,
    v::AbstractMatrix,
    ρ::AbstractMatrix,
    P::AbstractMatrix,
    grid::FlowGeometries.Grids.StructuredGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    filter_plan::Union{Nothing,Filtering.AbstractFilterPlan} = nothing,
    deriv_plan::Union{Nothing,Derivatives.StencilPlan} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend) :
        filter_plan
    dplan = deriv_plan === nothing ? Derivatives.StencilPlan(grid) : deriv_plan

    # ρ̄, P̄ and the UNWEIGHTED velocities. `ū`/`v̄` are not a convenience: the budget's pressure term is
    # `P̄ ∇·ū`, with the unweighted divergence, and `τ̄(ρ,u_j)` needs `ū_j` too.
    Filtering.filter_apply_batch!((ws.ρ̄, ws.P̄, ws.ū, ws.v̄), (ρ, P, u, v), plan)

    # Favre velocities ũ_i = (ρu_i)‾/ρ̄. `mx`/`my` hold (ρu_i)‾ first, then become the unweighted
    # subscale mass flux τ̄(ρ,u_i) = (ρu_i)‾ − ρ̄ū_i, which is what baropycnal work contracts against.
    @. ws.prod = ρ * u
    Filtering.filter_apply!(ws.mx, ws.prod, plan)
    @. ws.prod = ρ * v
    Filtering.filter_apply!(ws.my, ws.prod, plan)
    @. ws.ũ = ws.mx / ws.ρ̄
    @. ws.ṽ = ws.my / ws.ρ̄
    @. ws.mx -= ws.ρ̄ * ws.ū
    @. ws.my -= ws.ρ̄ * ws.v̄

    # Favre stress τ̃(u_i,u_j) = (ρu_iu_j)‾/ρ̄ − ũ_iũ_j.
    @. ws.prod = ρ * u * u
    Filtering.filter_apply!(ws.fbuf, ws.prod, plan)
    @. ws.τxx = ws.fbuf / ws.ρ̄ - ws.ũ * ws.ũ
    @. ws.prod = ρ * u * v
    Filtering.filter_apply!(ws.fbuf, ws.prod, plan)
    @. ws.τxy = ws.fbuf / ws.ρ̄ - ws.ũ * ws.ṽ
    @. ws.prod = ρ * v * v
    Filtering.filter_apply!(ws.fbuf, ws.prod, plan)
    @. ws.τyy = ws.fbuf / ws.ρ̄ - ws.ṽ * ws.ṽ

    # Deformation work uses the FAVRE velocity gradient; pressure dilatation uses the unweighted one.
    Derivatives.ddx!(ws.ux, ws.ũ, grid, dplan)
    Derivatives.ddy!(ws.uy, ws.ũ, grid, dplan)
    Derivatives.ddx!(ws.vx, ws.ṽ, grid, dplan)
    Derivatives.ddy!(ws.vy, ws.ṽ, grid, dplan)
    Derivatives.ddx!(ws.Px, ws.P̄, grid, dplan)
    Derivatives.ddy!(ws.Py, ws.P̄, grid, dplan)

    mask = FlowGeometries.Grids.mask(grid)
    # Π = −ρ̄ ∂_j ũ_i τ̃(u_i,u_j), summed over i,j; τ̃ is symmetric so the two off-diagonals combine.
    @. ws.Π = ifelse(mask,
        -ws.ρ̄ * (ws.ux * ws.τxx + (ws.uy + ws.vx) * ws.τxy + ws.vy * ws.τyy), zero(T))
    # Λ = (1/ρ̄) ∂_j P̄ · τ̄(ρ,u_j) — baropycnal work.
    @. ws.Λ = ifelse(mask, (ws.Px * ws.mx + ws.Py * ws.my) / ws.ρ̄, zero(T))
    # P̄ ∇·ū, with the UNWEIGHTED divergence. `ws.prod`/`ws.fbuf` are free again here.
    Derivatives.ddx!(ws.prod, ws.ū, grid, dplan)
    Derivatives.ddy!(ws.fbuf, ws.v̄, grid, dplan)
    @. ws.PD = ifelse(mask, ws.P̄ * (ws.prod + ws.fbuf), zero(T))

    return (; Π = ws.Π, Λ = ws.Λ, pressure_dilatation = ws.PD,
            ρ̄ = ws.ρ̄, P̄ = ws.P̄, ũ = ws.ũ, ṽ = ws.ṽ)
end

"""
    compressible_flux(u, v, ρ, P, grid, kernel, scale; backend=AutoBackend(), mask_strategy=ZeroFill())
        -> (; Π, Λ, pressure_dilatation, ρ̄, P̄, ũ, ṽ)

The variable-density (Favre) cross-scale energy budget of Aluie (2013). Returns the three terms of that
budget which act on the large-scale kinetic energy `ρ̄|ũ|²/2`, plus the filtered fields they are built
from.

# Favre filtering

`f̃ ≡ (ρf)‾/ρ̄` is the density-weighted filter. It exists because it is the one that makes the filtered
continuity equation close exactly, `∂_tρ̄ + ∂_i(ρ̄ũ_i) = 0`; the unweighted filter does not. It is linear
but **does not commute with derivatives**, so the two filters are not interchangeable and the budget
genuinely needs both — which is the source of the trap below.

# The three terms

```
Π = −ρ̄ ∂_j ũ_i τ̃(u_i,u_j) ,   τ̃(u_i,u_j) = (ρu_iu_j)‾/ρ̄ − ũ_iũ_j        deformation work
Λ = (1/ρ̄) ∂_j P̄ · τ̄(ρ,u_j) ,  τ̄(ρ,u_j)   = (ρu_j)‾ − ρ̄ū_j              baropycnal work
                                                                        (τ̄ is UNWEIGHTED)
P̄ ∇·ū                                                                   pressure dilatation
```

`Π` and `Λ` both pit a large-scale field against small-scale fluctuations, so **both transfer energy
across scales**. `P̄∇·ū` involves only large scales and cannot — it is a conversion between kinetic and
internal energy at the resolved scale, not a cascade term.

# The trap

`Λ` is frequently absorbed into the pressure term by writing it as `P̄∇·ũ` (plus a transport term) and
then dismissed as "large-scale pressure dilatation that needs no modelling". That is wrong: the budget
term is `P̄∇·ū` with the **unweighted** divergence, and writing `∇·ũ` silently destroys `Λ` — a genuine
cross-scale transfer. This implementation keeps them separate and uses `ū` for the dilatation; the
suite asserts that `Λ` is non-zero for a baroclinic configuration, so it cannot be quietly dropped.

# Asymptotics

For a smooth field, Lees & Aluie (2019) give `Λ ≈ (C₂ℓ²/ρ̄)·c_d·[∇P̄·S̄·∇ρ̄ + ½ ω̄·(∇ρ̄ × ∇P̄)]` with
`C₂` the kernel's second moment — a strain-generation part plus a **baroclinic** part that survives
even in pure solenoidal flow. That `C₂ ≠ 0` requirement is another reason the flux framework wants a
kernel with a NON-vanishing second moment; see [`Kernels.HighOrderKernel`](@ref) for the kernels that
deliberately give it up.

# References
- Aluie, H. (2013). Scale decomposition in compressible turbulence. *Physica D* 247, 54–65.
- Lees, A., & Aluie, H. (2019). Baropycnal work: a mechanism for energy transfer across scales.
  *Fluids* 4, 92. doi:10.3390/fluids4020092
"""
function compressible_flux(
    u::AbstractMatrix,
    v::AbstractMatrix,
    ρ::AbstractMatrix,
    P::AbstractMatrix,
    grid::FlowGeometries.Grids.StructuredGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    for (nm, a) in (("u", u), ("v", v), ("ρ", ρ), ("P", P))
        size(a) == gsz || throw(DimensionMismatch("$nm has size $(size(a)), grid expects $gsz"))
    end
    all(>(0), ρ) || throw(ArgumentError(
        "Favre filtering divides by the filtered density, so ρ must be strictly positive everywhere; " *
        "got a minimum of $(minimum(ρ)).",
    ))
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend)
    return compressible_flux!(
        FavreWorkspace(grid), u, v, ρ, P, grid, kernel, scale;
        filter_plan = plan, deriv_plan = Derivatives.StencilPlan(grid),
    )
end

"""
    favre_filter!(out, tmp, f, ρ, ρ̄, plan) -> out

`f̃ = (ρf)‾/ρ̄`, given an already-filtered `ρ̄` and a scratch array `tmp`. The building block of
[`compressible_flux`](@ref), exposed because a caller filtering their own tracer Favre-style should not
have to reimplement it (and get the weighting backwards).
"""
function favre_filter!(
    out::AbstractArray{T}, tmp::AbstractArray{T}, f::AbstractArray, ρ::AbstractArray,
    ρ̄::AbstractArray{T}, plan::Filtering.AbstractFilterPlan,
) where {T<:AbstractFloat}
    @. tmp = ρ * f
    Filtering.filter_apply!(out, tmp, plan)
    @. out = out / ρ̄
    return out
end

# ---------------------------------------------------------------------------
# Scale-band energy decomposition (Aluie & Eyink 2009 App. 2; Germano 1992 eq. 33)
# ---------------------------------------------------------------------------

"""
    band_energies(u, v, w, grid, kernel, scales; backend=AutoBackend(), mask_strategy=ZeroFill())
        -> (; bands, resolved, total, band_maps, resolved_map)

Split the kinetic energy into contributions from each scale band, using the **repeated-filter**
generalization of the Germano identity rather than by band-passing the velocity.

With `scales` in ASCENDING order `ℓ₁ < ℓ₂ < … < ℓ_N`, define the repeatedly filtered fields

```
f₀ = u ,   f_n = G_{ℓ_n} * f_{n-1} ,
```

so `f_n` has had every scale below `ℓ_n` removed, successively. Band `n` holds the energy the `n`-th
application removed, which is the generalized second moment at that level:

```
k_n = ½ τ_{ℓ_n}(f_{n-1}; f_{n-1}) = ½[ (|f_{n-1}|²)‾_{ℓ_n} − |f_n|² ] ,
```

and the decomposition is exact:

```
½⟨|u|²⟩ = Σ_{n=1}^N ⟨k_n⟩ + ½⟨|f_N|²⟩ .
```

The sum telescopes because each `⟨G * x⟩ = ⟨x⟩` — i.e. **because the filter conserves the domain
mean**. Measured, that holds to round-off on a periodic, unmasked grid (relative error 5e-16 for the
top-hat, 2e-15 for the Gaussian) and the identity is exact there.

Anywhere the footprint is truncated, it is not, and the identity carries a residual of order `ℓ/L`:
measured on the same field, **1.1e-2 relative on a BOUNDED grid** (the footprint runs off the domain
edge) and **1.1e-2 on a masked periodic grid under `ZeroFill`** (energy is smeared onto masked cells,
which report zero). `Deformable` renormalizes that leakage away and does better on a masked domain —
4.8e-4 — at the cost of the commutation property `ZeroFill` is the default for. So: read the bands as
exact on a periodic unmasked domain, and as carrying an `O(ℓ/L)` boundary residual otherwise.

# Why not band-pass the velocity

The obvious alternative, `u = ū₀ + Σ(ū_n − ū_{n-1})`, gives
`½⟨|u|²⟩ = ½⟨|ū₀|²⟩ + ½Σ_{n,m}⟨ū_n · ū_m⟩` — cross terms of indefinite sign, so there is no
well-defined energy at a given scale at all (Aluie & Eyink 2009). The second-moment form above has no
cross terms by construction, and `k_n ≥ 0` pointwise **iff the kernel is non-negative** — so use a
positive kernel here (`TopHatKernel`, `GaussianKernel`, `SmoothHatKernel`, `HyperGaussianKernel`); a
signed one such as [`Kernels.HighOrderKernel`](@ref) can give negative band energies.

Returns the per-band domain-averaged energies `bands` (length `N`), the energy left in `f_N`
(`resolved`), their sum `total`, and the corresponding pointwise maps.

# References
- Germano, M. (1992). *J. Fluid Mech.* 238, eq. (33).
- Aluie, H., & Eyink, G. L. (2009). Localness of energy cascade in hydrodynamic turbulence.
  *Phys. Fluids* 21, 115108, Appendix 2.
"""
function band_energies(
    u::AbstractArray,
    v::AbstractArray,
    w::Union{Nothing, AbstractArray},
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scales::AbstractVector;
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Union{Nothing, Filtering.AbstractFilterMethod} = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    isempty(scales) && throw(ArgumentError("band_energies needs at least one scale"))
    issorted(scales) || throw(ArgumentError(
        "band_energies needs `scales` in ascending order (finest first): band n is the energy the " *
        "n-th, progressively coarser, filter application removes. Got $scales.",
    ))
    gsz = FlowGeometries.Grids.size_tuple(grid)
    has_w = w !== nothing
    total_area = active_area(grid)

    plans = [method === nothing ?
             Filtering.plan_filter(grid, kernel, T(s); mask_strategy = mask_strategy, backend = backend) :
             Filtering.plan_filter(grid, kernel, T(s); mask_strategy = mask_strategy, backend = backend,
                                   method = method)
             for s in scales]

    # `f` is the running repeatedly-filtered field; `nxt` receives each next application. `sq`/`fsq`
    # carry `|f|²` and its filtered image, which is what makes this a second moment and not a
    # band-passed velocity.
    f = (copy(u), copy(v), has_w ? copy(w) : nothing)
    nxt = (zeros(T, gsz), zeros(T, gsz), has_w ? zeros(T, gsz) : nothing)
    sq = zeros(T, gsz); fsq = zeros(T, gsz)

    band_maps = [zeros(T, gsz) for _ in eachindex(scales)]
    bands = zeros(T, length(scales))

    for n in eachindex(scales)
        km = band_maps[n]
        fill!(km, zero(T))
        for c in 1:(has_w ? 3 : 2)
            fc = f[c]
            Filtering.filter_apply!(nxt[c], fc, plans[n])
            @. sq = fc * fc
            Filtering.filter_apply!(fsq, sq, plans[n])
            # τ(f;f) = (f²)‾ − (f̄)², summed over components; the ½ is applied once at the end.
            @. km += fsq - nxt[c] * nxt[c]
        end
        mask = FlowGeometries.Grids.mask(grid)
        @. km = ifelse(mask, T(0.5) * km, zero(T))
        bands[n] = _area_mean(km, grid, total_area)
        for c in 1:(has_w ? 3 : 2)
            copyto!(f[c], nxt[c])
        end
    end

    resolved_map = zeros(T, gsz)
    let mask = FlowGeometries.Grids.mask(grid)
        for c in 1:(has_w ? 3 : 2)
            fc = f[c]
            @. resolved_map += fc * fc
        end
        @. resolved_map = ifelse(mask, T(0.5) * resolved_map, zero(T))
    end
    resolved = _area_mean(resolved_map, grid, total_area)
    return (; bands, resolved, total = sum(bands) + resolved, band_maps, resolved_map)
end

band_energies(u, v, grid::FlowGeometries.Grids.AbstractGrid, kernel, scales; kwargs...) =
    band_energies(u, v, nothing, grid, kernel, scales; kwargs...)

# ---------------------------------------------------------------------------
# Enstrophy flux (Rivera, Aluie & Ecke 2014, eq. 16)
# ---------------------------------------------------------------------------

"""
    vorticity!(ω, u, v, grid[, deriv_plan]) -> ω

Vertical component of the relative vorticity, `ω = ∂v/∂x − ∂u/∂y`, on a 2-D grid. Masked cells are
zeroed, as everywhere else. Uses the same `ddx!`/`ddy!` the flux diagnostics do, so `ω` and the
gradients it is later contracted against are consistent to the last bit.
"""
function vorticity!(
    ω::AbstractMatrix{T},
    u::AbstractMatrix,
    v::AbstractMatrix,
    grid::FlowGeometries.Grids.StructuredGrid{G,T,2},
    deriv_plan::Union{Nothing,Derivatives.StencilPlan} = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    dplan = deriv_plan === nothing ? Derivatives.StencilPlan(grid) : deriv_plan
    tmp = similar(ω)
    Derivatives.ddx!(ω, v, grid, dplan)
    Derivatives.ddy!(tmp, u, grid, dplan)
    mask = FlowGeometries.Grids.mask(grid)
    @. ω = ifelse(mask, ω - tmp, zero(T))
    return ω
end

"""
    vorticity(u, v, grid[, deriv_plan]) -> ω

Allocating [`vorticity!`](@ref).
"""
function vorticity(
    u::AbstractMatrix, v::AbstractMatrix,
    grid::FlowGeometries.Grids.StructuredGrid{G,T,2},
    deriv_plan::Union{Nothing,Derivatives.StencilPlan} = nothing,
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
    u::AbstractMatrix,
    v::AbstractMatrix,
    grid::FlowGeometries.Grids.StructuredGrid{G,T,2},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    dplan = Derivatives.StencilPlan(grid)
    ω = vorticity(u, v, grid, dplan)
    return tracer_variance_flux(u, v, ω, grid, kernel, scale;
                                backend = backend, mask_strategy = mask_strategy)
end

"""
    EnstrophyFluxWorkspace(grid)

Scratch for [`enstrophy_flux!`](@ref): the vorticity plus the tracer-flux scratch it is fed into.
"""
struct EnstrophyFluxWorkspace{T<:AbstractFloat, M<:AbstractMatrix{T}}
    ω::M
    tracer::TracerFluxWorkspace{T,M}
end

EnstrophyFluxWorkspace(grid::FlowGeometries.Grids.StructuredGrid{G,T}) where {T<:AbstractFloat, G} =
    EnstrophyFluxWorkspace(zeros(T, FlowGeometries.Grids.size_tuple(grid)), TracerFluxWorkspace(grid))

"""
    enstrophy_flux!(Z, ws, u, v, grid, kernel, scale; filter_plan=nothing, deriv_plan=nothing, ...) -> Z

In-place [`enstrophy_flux`](@ref). With `ws` and both plans supplied, a repeated evaluation allocates
nothing.
"""
function enstrophy_flux!(
    Z::AbstractMatrix{T},
    ws::EnstrophyFluxWorkspace{T},
    u::AbstractMatrix,
    v::AbstractMatrix,
    grid::FlowGeometries.Grids.StructuredGrid{G,T,2},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    filter_plan::Union{Nothing,Filtering.AbstractFilterPlan} = nothing,
    deriv_plan::Union{Nothing,Derivatives.StencilPlan} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    dplan = deriv_plan === nothing ? Derivatives.StencilPlan(grid) : deriv_plan
    # `ws.tracer.uθ` is free at this point (it is only written inside `tracer_variance_flux!`), so it
    # serves as the `∂u/∂y` scratch the curl needs — no extra buffer in the workspace for it.
    Derivatives.ddx!(ws.ω, v, grid, dplan)
    Derivatives.ddy!(ws.tracer.uθ, u, grid, dplan)
    mask = FlowGeometries.Grids.mask(grid)
    @. ws.ω = ifelse(mask, ws.ω - ws.tracer.uθ, zero(T))
    return tracer_variance_flux!(Z, ws.tracer, u, v, ws.ω, grid, kernel, scale;
                                 filter_plan = filter_plan, deriv_plan = dplan,
                                 backend = backend, mask_strategy = mask_strategy)
end

"""
    tracer_variance_flux(u, v, θ, grid::StructuredGrid{<:SphericalGeometry}, kernel, scale; ...) -> Πθ

Spherical form of the tracer-variance flux. `τ_j = ⟨u_j θ⟩ - ū_j θ̄` is a vector, so — exactly as in
[`compute_Π!`](@ref) and [`tau_decomposition`](@ref) — the velocity is rotated to planetary Cartesian
before filtering (Aluie 2019 commutativity: component-wise filtering of a local east/north pair is not
a filtered vector, since the local basis turns from point to point), and the filtered flux is rotated
back to the local east/north frame to contract against `∂_j θ̄`. The scalar `θ` needs no rotation. The
radial component of `τ` is dropped, matching this 2-D shell's dropping of radial derivatives.
"""
function tracer_variance_flux(
    u::AbstractMatrix,
    v::AbstractMatrix,
    θ::AbstractMatrix,
    grid::FlowGeometries.Grids.StructuredGrid{G,T},
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
    size(θ) == gsz || throw(DimensionMismatch("θ has size $(size(θ)), grid expects $gsz"))
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend)
    geo = FlowGeometries.Grids.grid_geometry(grid)

    ux = zeros(T, gsz); uy = zeros(T, gsz); uz = zeros(T, gsz)
    uxθ = zeros(T, gsz); uyθ = zeros(T, gsz); uzθ = zeros(T, gsz)
    @inbounds for I in CartesianIndices(u)
        i = Tuple(I)
        FlowGeometries.Grids.isactive(grid, i...) || continue
        λ, φ = FlowGeometries.Grids.coords(grid, i...)
        p = FlowGeometries.Geometry.vector_to_cartesian(geo, u[I], v[I], λ, φ)
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

    # Rotate the planetary-Cartesian subfilter flux back to local (east, north, radial); the radial
    # component is not used, as the resolved gradient here has no radial part.
    τe = zeros(T, gsz); τn = zeros(T, gsz)
    @inbounds for I in CartesianIndices(u)
        i = Tuple(I)
        FlowGeometries.Grids.isactive(grid, i...) || continue
        λ, φ = FlowGeometries.Grids.coords(grid, i...)
        l = FlowGeometries.Geometry.vector_from_cartesian(geo, τX[I], τY[I], τZ[I], λ, φ)
        τe[I] = l[1]; τn[I] = l[2]
    end

    gx = similar(θ̄); Derivatives.ddx!(gx, θ̄, grid, dplan)
    gy = similar(θ̄); Derivatives.ddy!(gy, θ̄, grid, dplan)

    mask = FlowGeometries.Grids.mask(grid)
    return ifelse.(mask, .-(τe .* gx .+ τn .* gy), zero(T))
end

"""
    tracer_variance_flux(u, v, w, θ, grid::StructuredGrid{Cartesian,T,3}, kernel, scale; backend=AutoBackend(), mask_strategy=ZeroFill())
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
    grid::FlowGeometries.Grids.StructuredGrid{G,T,3},
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
    tracer_variance_flux(u, v, w, θ, grid::StructuredGrid{<:SphericalGeometry,T,3}, kernel, scale; ...) -> Πθ

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
    grid::FlowGeometries.Grids.StructuredGrid{G,T,3},
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

end # module
