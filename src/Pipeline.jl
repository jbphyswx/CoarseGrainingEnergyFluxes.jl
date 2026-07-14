module Pipeline

using ..Geometry: Geometry
using ..Grids: Grids
using ..Kernels: Kernels
using ..Filtering: Filtering
using ..Derivatives: Derivatives
using ..Diagnostics: Diagnostics
using ..Backends: Backends

export CoarseGrainResult, coarse_grain, coarse_grain!

"""
    CoarseGrainResult{T<:AbstractFloat, N, A<:AbstractArray{T,N}}

Container for results of a complete coarse-graining multiscale analysis.

# Fields
- `scales::AbstractVector{T}`: filter scales ℓ in meters
- `Π::A`: energy-flux maps stacked into ONE contiguous `(spatial dims..., Nscales)` array (W/m³) —
  not a `Vector` of separately-allocated per-scale matrices, so the whole sweep is a single allocation
  and each scale's map is a zero-copy view (`compute_Π!` writes directly into its slice).
- `cumulative_energy::AbstractArray{T}`: cumulative coarse specific KE ½⟨|ū_ℓ|²⟩ per scale (Sadek–Aluie Eq.
  15) — a `Vector` (per scale) for `coarse_grain`, or a `(Nlevels, Nscales)` `Matrix` for
  `coarse_grain_profile` (per depth level AND scale — deliberately not summed across levels, since
  that would need volume/thickness weighting this function doesn't have).
- `wavenumber::AbstractVector{T}`: filtering wavenumber `k_ℓ = L/ℓ` per scale (level-independent)
- `filtering_spectrum::AbstractArray{T}`: filtering spectral density `Ẽ(k_ℓ)` per scale (Eq. 14), same
  shape convention as `cumulative_energy`

# Examples
```julia
res = coarse_grain(u, v, grid; scales=[10e3, 20e3, 30e3], kernel=TopHatKernel())
# Access results:
res.scales[1]              # First scale (10 km)
res.Π[:, :, 1]              # Energy-flux map at 10 km (a view; use `@view` to avoid copying)
res.cumulative_energy[1]   # cumulative coarse KE at 10 km
res.filtering_spectrum[1]  # filtering spectral density at k_ℓ = res.wavenumber[1]
```
"""
struct CoarseGrainResult{T<:AbstractFloat, N, A<:AbstractArray{T,N}}
    scales::AbstractVector{T}
    Π::A
    cumulative_energy::AbstractArray{T}
    wavenumber::AbstractVector{T}
    filtering_spectrum::AbstractArray{T}
end

# ---------------------------------------------------------------------------
# StructuredGrid pipeline
# ---------------------------------------------------------------------------

"""
    coarse_grain(u, v, w, grid; scales, kernel=TopHatKernel(), backend=AutoBackend(), mask_strategy=Deformable(), L=1)
    coarse_grain(u, v, grid; scales, ...)  # 2D convenience wrapper

Perform complete coarse-graining analysis across multiple filter scales, allocating a fresh
[`CoarseGrainResult`](@ref) and workspace. This is a thin wrapper around [`coarse_grain!`](@ref);
for repeated sweeps over the same grid/scales (e.g. successive timesteps), allocate the result once
and call `coarse_grain!` directly to reuse its buffers.

# Arguments
- `u::AbstractMatrix`: Eastward/zonal velocity component (m/s)
- `v::AbstractMatrix`: Northward/meridional velocity component (m/s)
- `w::Union{Nothing,AbstractMatrix}`: Vertical velocity (nothing for 2D)
- `grid::StructuredGrid`: Grid geometry and active-cell mask

# Keyword Arguments
- `scales::AbstractVector`: Vector of filter scales ℓ in meters (e.g., `10e3:10e3:100e3`)
- `kernel::AbstractFilterKernel=TopHatKernel()`: Filter kernel
- `backend::AbstractExecutionBackend=AutoBackend()`: Execution backend
- `mask_strategy::AbstractMaskStrategy=Deformable()`: Land masking strategy (`ZeroFill()` or `Deformable()`)

# Returns
- `CoarseGrainResult`: Container with scales, Π maps, and spectrum

# Examples
```julia
geom = SphericalGeometry(6371000.0)
grid = StructuredGrid(geom, lon_rad, lat_rad, mask)
scales = collect(10e3:10e3:100e3)
res = coarse_grain(u, v, grid; scales=scales, kernel=TopHatKernel())
plot(res.wavenumber, res.filtering_spectrum, xscale=:log10, yscale=:log10)
heatmap(res.Π[:, :, 3])  # 3rd scale is 30 km
```
"""
function coarse_grain(
    u::AbstractMatrix,
    v::AbstractMatrix,
    w::Union{Nothing, AbstractMatrix},
    grid::Grids.StructuredGrid{G,T};
    scales::AbstractVector,
    kernel::Kernels.AbstractFilterKernel = Kernels.TopHatKernel(),
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    L::Real = one(T),
) where {T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}}
    result = _allocate_result(grid, length(scales))
    workspace = Diagnostics.ΠWorkspace(grid)
    return coarse_grain!(
        result, u, v, w, grid;
        scales = scales, kernel = kernel, workspace = workspace,
        backend = backend, mask_strategy = mask_strategy, L = L,
    )
end

"""
    coarse_grain!(result, u, v, w, grid; scales, kernel=TopHatKernel(), workspace=nothing, backend=AutoBackend(), mask_strategy=Deformable(), L=1)
    coarse_grain!(result, u, v, grid; scales, ...)  # 2D convenience wrapper

In-place [`coarse_grain`](@ref): refills an existing [`CoarseGrainResult`](@ref)'s buffers (scales,
the stacked `Π` array, cumulative energy, wavenumber, spectrum) instead of allocating fresh ones.
When `workspace` (a [`Diagnostics.ΠWorkspace`](@ref)) is supplied, its filtered-velocity/strain/stress scratch
arrays are reused too — for a sweep repeated across many timesteps of the same grid/scale set, this
is the zero-(re)allocation entry point (`coarse_grain` is a thin allocating wrapper around it).

`result` must already be sized for `length(scales)` scales over `grid`'s shape (as produced by a
prior `coarse_grain` call) — a mismatch throws `DimensionMismatch`.
"""
function coarse_grain!(
    result::CoarseGrainResult{T},
    u::AbstractMatrix,
    v::AbstractMatrix,
    w::Union{Nothing, AbstractMatrix},
    grid::Grids.StructuredGrid{G,T};
    scales::AbstractVector,
    kernel::Kernels.AbstractFilterKernel = Kernels.TopHatKernel(),
    workspace::Union{Nothing, Diagnostics.ΠWorkspace} = nothing,
    filter_plans::Union{Nothing, AbstractVector} = nothing,
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    L::Real = one(T),
) where {T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}}
    _check_result_shape(result, grid, scales)
    ws = workspace === nothing ? Diagnostics.ΠWorkspace(grid) : workspace

    # Each scale's filter plan is reused for both `compute_Π!` (below) and `cumulative_energy!`
    # (after the loop) — they used to each independently rebuild the same footprint per scale
    # (measured: a 3-scale sweep allocated ~4.6x the raw sum of the three footprint builds, because
    # `cumulative_energy!` redundantly re-filtered u/v/w from scratch instead of reusing the plan
    # `compute_Π!` had just built for that exact same scale). For a sweep repeated across many
    # timesteps of the SAME grid/kernel/scales (this function's own documented zero-allocation use
    # case), pass a prebuilt `filter_plans` (one entry per scale, e.g. from a first `coarse_grain`
    # call) so this doesn't allocate a fresh plan vector — or fresh plans — on every repeat call.
    plans = filter_plans === nothing ? Vector{Filtering.AbstractFilterPlan}(undef, length(scales)) : filter_plans
    for s_idx in eachindex(scales)
        scale = T(scales[s_idx])
        result.scales[s_idx] = scale
        filter_plans === nothing &&
            (plans[s_idx] = Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend))
        Diagnostics.compute_Π!(
            view(result.Π, :, :, s_idx),
            u, v, w, grid, kernel, scale;
            workspace = ws, filter_plan = plans[s_idx], backend = backend, mask_strategy = mask_strategy,
        )
    end

    Diagnostics.cumulative_energy!(
        result.cumulative_energy, u, v, w, grid, kernel, scales;
        workspace = ws, filter_plans = plans, backend = backend, mask_strategy = mask_strategy,
    )
    result.wavenumber .= T(L) ./ result.scales
    Diagnostics.spectral_density!(result.filtering_spectrum, result.cumulative_energy, result.wavenumber)
    return result
end

# 2.5D Cartesian constructor wrapper (2D velocity fields without a vertical component)
function coarse_grain(
    u::AbstractMatrix,
    v::AbstractMatrix,
    grid::Grids.StructuredGrid{G,T};
    scales::AbstractVector,
    kernel::Kernels.AbstractFilterKernel = Kernels.TopHatKernel(),
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    L::Real = one(T),
) where {T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}}
    return coarse_grain(u, v, nothing, grid; scales=scales, kernel=kernel, backend=backend, mask_strategy=mask_strategy, L=L)
end

function coarse_grain!(
    result::CoarseGrainResult{T},
    u::AbstractMatrix,
    v::AbstractMatrix,
    grid::Grids.StructuredGrid{G,T};
    scales::AbstractVector,
    kernel::Kernels.AbstractFilterKernel = Kernels.TopHatKernel(),
    workspace::Union{Nothing, Diagnostics.ΠWorkspace} = nothing,
    filter_plans::Union{Nothing, AbstractVector} = nothing,
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    L::Real = one(T),
) where {T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}}
    return coarse_grain!(result, u, v, nothing, grid; scales=scales, kernel=kernel, workspace=workspace, filter_plans=filter_plans, backend=backend, mask_strategy=mask_strategy, L=L)
end

"""
    coarse_grain_profile(u, v, w, grid, scales; kernel=TopHatKernel(), backend=AutoBackend(), mask_strategy=Deformable(), L=1)

Depth-profile sweep: given 3D `(lon, lat, depth)` velocity arrays, runs [`Diagnostics.compute_Π_profile!`](@ref)
(the literature-standard independent-per-level 2D/2.5D method — see [`Diagnostics.compute_Π!`](@ref)'s
thin-layer/QG regime note) at every scale, returning a [`CoarseGrainResult`](@ref) whose `Π` is one
contiguous `(Nlon, Nlat, Nlevels, Nscales)` array. The workspace is built once and reused across the
whole level × scale sweep.
"""
function coarse_grain_profile(
    u::AbstractArray{T,3},
    v::AbstractArray{T,3},
    w::Union{Nothing, AbstractArray{T,3}},
    grid::Grids.StructuredGrid{G,T};
    scales::AbstractVector,
    kernel::Kernels.AbstractFilterKernel = Kernels.TopHatKernel(),
    workspace::Union{Nothing, Diagnostics.ΠWorkspace} = nothing,
    filter_plans::Union{Nothing, AbstractVector} = nothing,
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    L::Real = one(T),
) where {T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}}
    Nscales = length(scales)
    Nlon, Nlat = Grids.size_tuple(grid)
    Nlevels = size(u, 3)

    Π = zeros(T, Nlon, Nlat, Nlevels, Nscales)
    ws = workspace === nothing ? Diagnostics.ΠWorkspace(grid) : workspace
    scales_vec = zeros(T, Nscales)
    # One filter plan per scale, reused both across all levels of the Π sweep below AND across all
    # levels of the cumulative-energy sweep after it — otherwise the energy sweep (looping per
    # level, each internally looping per scale) would rebuild the same Nscales footprints again for
    # every one of the Nlevels iterations, on top of what the Π sweep already built (confirmed by
    # measurement on the plain 2D `coarse_grain!` case this mirrors). For a sweep repeated across
    # many timesteps of the SAME grid/kernel/scales, pass a prebuilt `filter_plans` (and `workspace`)
    # so this doesn't allocate a fresh plan vector — or fresh plans — on every repeat call.
    plans = filter_plans === nothing ? Vector{Filtering.AbstractFilterPlan}(undef, Nscales) : filter_plans
    for s_idx in 1:Nscales
        scale = T(scales[s_idx])
        scales_vec[s_idx] = scale
        filter_plans === nothing &&
            (plans[s_idx] = Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend))
        Diagnostics.compute_Π_profile!(
            view(Π, :, :, :, s_idx), u, v, w, grid, kernel, scale;
            workspace = ws, filter_plan = plans[s_idx], backend = backend, mask_strategy = mask_strategy,
        )
    end

    # Cumulative energy/spectrum per level, reusing the 2D primitives with a view per level.
    cumE = zeros(T, Nlevels, Nscales)
    for k in 1:Nlevels
        wk = w === nothing ? nothing : view(w, :, :, k)
        Diagnostics.cumulative_energy!(
            view(cumE, k, :), view(u, :, :, k), view(v, :, :, k), wk, grid, kernel, scales;
            workspace = ws, filter_plans = plans, backend = backend, mask_strategy = mask_strategy,
        )
    end
    kℓ = T(L) ./ scales_vec
    spec = zeros(T, Nlevels, Nscales)
    for k in 1:Nlevels
        Diagnostics.spectral_density!(view(spec, k, :), view(cumE, k, :), kℓ)
    end

    # cumE/spec are kept as genuine (Nlevels, Nscales) matrices — not summed across levels, which
    # would need volume/thickness weighting this function isn't given.
    return CoarseGrainResult{T, 4, Array{T,4}}(scales_vec, Π, cumE, kℓ, spec)
end

# 2.5D convenience wrapper (no vertical velocity).
function coarse_grain_profile(
    u::AbstractArray{T,3},
    v::AbstractArray{T,3},
    grid::Grids.StructuredGrid{G,T};
    scales::AbstractVector,
    kernel::Kernels.AbstractFilterKernel = Kernels.TopHatKernel(),
    workspace::Union{Nothing, Diagnostics.ΠWorkspace} = nothing,
    filter_plans::Union{Nothing, AbstractVector} = nothing,
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    L::Real = one(T),
) where {T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}}
    return coarse_grain_profile(u, v, nothing, grid; scales=scales, kernel=kernel, workspace=workspace, filter_plans=filter_plans, backend=backend, mask_strategy=mask_strategy, L=L)
end

# ---------------------------------------------------------------------------
# 1D Cartesian pipeline: a single scalar velocity component `u` along one axis (a genuinely 1D
# `StructuredGrid`, not a 2D grid with a singleton dimension). Cumulative energy is computed directly
# here (0.5⟨ū_ℓ²⟩), not via the shared `cumulative_energy!` — that function's signature genuinely
# needs two velocity components (u,v), which don't exist in a 1D flow.
# ---------------------------------------------------------------------------
function coarse_grain(
    u::AbstractVector,
    grid::Grids.StructuredGrid{G,T,1};
    scales::AbstractVector,
    kernel::Kernels.AbstractFilterKernel = Kernels.TopHatKernel(),
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    L::Real = one(T),
) where {T<:AbstractFloat, G<:Geometry.CartesianGeometry{T}}
    result = _allocate_result(grid, length(scales))
    workspace = Diagnostics.ΠWorkspace(grid)
    return coarse_grain!(
        result, u, grid;
        scales = scales, kernel = kernel, workspace = workspace,
        backend = backend, mask_strategy = mask_strategy, L = L,
    )
end

function coarse_grain!(
    result::CoarseGrainResult{T},
    u::AbstractVector,
    grid::Grids.StructuredGrid{G,T,1};
    scales::AbstractVector,
    kernel::Kernels.AbstractFilterKernel = Kernels.TopHatKernel(),
    workspace::Union{Nothing, Diagnostics.ΠWorkspace} = nothing,
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    L::Real = one(T),
) where {T<:AbstractFloat, G<:Geometry.CartesianGeometry{T}}
    _check_result_shape(result, grid, scales)
    ws = workspace === nothing ? Diagnostics.ΠWorkspace(grid) : workspace
    Nlon = Grids.size_tuple(grid)[1]

    total_area = sum(Grids.area(grid, i) for i in 1:Nlon if Grids.isactive(grid, i))

    for s_idx in eachindex(scales)
        scale = T(scales[s_idx])
        result.scales[s_idx] = scale
        Diagnostics.compute_Π!(
            view(result.Π, :, s_idx),
            u, grid, kernel, scale;
            workspace = ws, backend = backend, mask_strategy = mask_strategy,
        )

        plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend)
        Filtering.filter_apply!(ws.u_filt, u, plan)
        integrated_energy = sum(ws.u_filt[i]^2 * Grids.area(grid, i) for i in 1:Nlon if Grids.isactive(grid, i))
        result.cumulative_energy[s_idx] = T(0.5) * integrated_energy / total_area
    end

    result.wavenumber .= T(L) ./ result.scales
    Diagnostics.spectral_density!(result.filtering_spectrum, result.cumulative_energy, result.wavenumber)
    return result
end

# ---------------------------------------------------------------------------
# True-3D pipeline (Cartesian OR spherical volumetric): genuinely coupled (all nine strain
# components, real vertical/radial derivatives), distinct from `coarse_grain_profile`'s per-level
# 2.5D sweep above — dispatches on a `StructuredGrid{G,T,3}` (3D grid) + 3D velocity arrays, not a 2D
# grid with a depth-stacked array. `Diagnostics.compute_Π!` itself dispatches Cartesian vs. spherical.
# ---------------------------------------------------------------------------
function coarse_grain(
    u::AbstractArray{<:Any,3},
    v::AbstractArray{<:Any,3},
    w::AbstractArray{<:Any,3},
    grid::Grids.StructuredGrid{G,T,3};
    scales::AbstractVector,
    kernel::Kernels.AbstractFilterKernel = Kernels.TopHatKernel(),
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    L::Real = one(T),
) where {T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}}
    result = _allocate_result(grid, length(scales))
    workspace = Diagnostics.ΠWorkspace(grid)
    return coarse_grain!(
        result, u, v, w, grid;
        scales = scales, kernel = kernel, workspace = workspace,
        backend = backend, mask_strategy = mask_strategy, L = L,
    )
end

function coarse_grain!(
    result::CoarseGrainResult{T},
    u::AbstractArray{<:Any,3},
    v::AbstractArray{<:Any,3},
    w::AbstractArray{<:Any,3},
    grid::Grids.StructuredGrid{G,T,3};
    scales::AbstractVector,
    kernel::Kernels.AbstractFilterKernel = Kernels.TopHatKernel(),
    workspace::Union{Nothing, Diagnostics.ΠWorkspace} = nothing,
    filter_plans::Union{Nothing, AbstractVector} = nothing,
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    L::Real = one(T),
) where {T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}}
    _check_result_shape(result, grid, scales)
    ws = workspace === nothing ? Diagnostics.ΠWorkspace(grid) : workspace

    # See the 2D `coarse_grain!` method above for why `plans` is reused with `cumulative_energy!`
    # below (instead of each independently rebuilding the same footprint) and why a caller doing a
    # repeated sweep should pass a prebuilt `filter_plans` rather than let this allocate a fresh one.
    plans = filter_plans === nothing ? Vector{Filtering.AbstractFilterPlan}(undef, length(scales)) : filter_plans
    for s_idx in eachindex(scales)
        scale = T(scales[s_idx])
        result.scales[s_idx] = scale
        filter_plans === nothing &&
            (plans[s_idx] = Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend))
        Diagnostics.compute_Π!(
            view(result.Π, :, :, :, s_idx),
            u, v, w, grid, kernel, scale;
            workspace = ws, filter_plan = plans[s_idx], backend = backend, mask_strategy = mask_strategy,
        )
    end

    Diagnostics.cumulative_energy!(
        result.cumulative_energy, u, v, w, grid, kernel, scales;
        workspace = ws, filter_plans = plans, backend = backend, mask_strategy = mask_strategy,
    )
    result.wavenumber .= T(L) ./ result.scales
    Diagnostics.spectral_density!(result.filtering_spectrum, result.cumulative_energy, result.wavenumber)
    return result
end

# ---------------------------------------------------------------------------
# Curvilinear-grid pipeline: same orchestration as the StructuredGrid path, but the WLSQ derivative
# plan (like the workspace) is built ONCE and reused across the whole scale sweep.
# ---------------------------------------------------------------------------
function coarse_grain(
    u::AbstractMatrix,
    v::AbstractMatrix,
    w::Union{Nothing, AbstractMatrix},
    grid::Grids.CurvilinearGrid{T,G};
    scales::AbstractVector,
    kernel::Kernels.AbstractFilterKernel = Kernels.TopHatKernel(),
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    L::Real = one(T),
) where {T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}}
    result = _allocate_result(grid, length(scales))
    workspace = Diagnostics.ΠWorkspace(grid)
    deriv_plan = Derivatives.WLSQGradientPlan(grid)
    return coarse_grain!(
        result, u, v, w, grid;
        scales = scales, kernel = kernel, workspace = workspace, deriv_plan = deriv_plan,
        backend = backend, mask_strategy = mask_strategy, L = L,
    )
end

function coarse_grain!(
    result::CoarseGrainResult{T},
    u::AbstractMatrix,
    v::AbstractMatrix,
    w::Union{Nothing, AbstractMatrix},
    grid::Grids.CurvilinearGrid{T,G};
    scales::AbstractVector,
    kernel::Kernels.AbstractFilterKernel = Kernels.TopHatKernel(),
    workspace::Union{Nothing, Diagnostics.ΠWorkspace} = nothing,
    deriv_plan::Union{Nothing, Derivatives.WLSQGradientPlan} = nothing,
    filter_plans::Union{Nothing, AbstractVector} = nothing,
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    L::Real = one(T),
) where {T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}}
    _check_result_shape(result, grid, scales)
    ws = workspace === nothing ? Diagnostics.ΠWorkspace(grid) : workspace
    dplan = deriv_plan === nothing ? Derivatives.WLSQGradientPlan(grid) : deriv_plan

    # See the 2D `StructuredGrid` `coarse_grain!` method above for why `plans` is reused with
    # `cumulative_energy!` below, and why a repeated sweep should pass a prebuilt `filter_plans`.
    plans = filter_plans === nothing ? Vector{Filtering.AbstractFilterPlan}(undef, length(scales)) : filter_plans
    for s_idx in eachindex(scales)
        scale = T(scales[s_idx])
        result.scales[s_idx] = scale
        filter_plans === nothing &&
            (plans[s_idx] = Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend))
        Diagnostics.compute_Π!(
            view(result.Π, :, :, s_idx),
            u, v, w, grid, kernel, scale;
            workspace = ws, deriv_plan = dplan, filter_plan = plans[s_idx], backend = backend, mask_strategy = mask_strategy,
        )
    end

    Diagnostics.cumulative_energy!(
        result.cumulative_energy, u, v, w, grid, kernel, scales;
        filter_plans = plans,
        workspace = ws, backend = backend, mask_strategy = mask_strategy,
    )
    result.wavenumber .= T(L) ./ result.scales
    Diagnostics.spectral_density!(result.filtering_spectrum, result.cumulative_energy, result.wavenumber)
    return result
end

# 2D curvilinear convenience wrapper (no vertical velocity).
function coarse_grain(
    u::AbstractMatrix,
    v::AbstractMatrix,
    grid::Grids.CurvilinearGrid{T,G};
    scales::AbstractVector,
    kernel::Kernels.AbstractFilterKernel = Kernels.TopHatKernel(),
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    L::Real = one(T),
) where {T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}}
    return coarse_grain(u, v, nothing, grid; scales=scales, kernel=kernel, backend=backend, mask_strategy=mask_strategy, L=L)
end

function coarse_grain!(
    result::CoarseGrainResult{T},
    u::AbstractMatrix,
    v::AbstractMatrix,
    grid::Grids.CurvilinearGrid{T,G};
    scales::AbstractVector,
    kernel::Kernels.AbstractFilterKernel = Kernels.TopHatKernel(),
    workspace::Union{Nothing, Diagnostics.ΠWorkspace} = nothing,
    deriv_plan::Union{Nothing, Derivatives.WLSQGradientPlan} = nothing,
    filter_plans::Union{Nothing, AbstractVector} = nothing,
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    L::Real = one(T),
) where {T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}}
    return coarse_grain!(result, u, v, nothing, grid; scales=scales, kernel=kernel, workspace=workspace, deriv_plan=deriv_plan, filter_plans=filter_plans, backend=backend, mask_strategy=mask_strategy, L=L)
end

# ---------------------------------------------------------------------------
# UnstructuredGrid pipeline: 1D node-indexed, same orchestration pattern as CurvilinearGrid but using
# `Derivatives.UnstructuredWLSQGradientPlan` (node-adjacency WLSQ, not index-offset WLSQ) and defaulting
# to spectral filtering (no real-space engine exists yet for scattered points).
# ---------------------------------------------------------------------------
function coarse_grain(
    u::AbstractVector,
    v::AbstractVector,
    w::Union{Nothing, AbstractVector},
    grid::Grids.UnstructuredGrid{T};
    scales::AbstractVector,
    kernel::Kernels.AbstractFilterKernel = Kernels.TopHatKernel(),
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    method::Filtering.AbstractFilterMethod = Filtering.Spectral(),
    L::Real = one(T),
) where {T<:AbstractFloat}
    result = _allocate_result(grid, length(scales))
    workspace = Diagnostics.ΠWorkspace(grid)
    deriv_plan = Derivatives.WLSQGradientPlan(grid)
    return coarse_grain!(
        result, u, v, w, grid;
        scales = scales, kernel = kernel, workspace = workspace, deriv_plan = deriv_plan,
        backend = backend, mask_strategy = mask_strategy, method = method, L = L,
    )
end

function coarse_grain!(
    result::CoarseGrainResult{T},
    u::AbstractVector,
    v::AbstractVector,
    w::Union{Nothing, AbstractVector},
    grid::Grids.UnstructuredGrid{T};
    scales::AbstractVector,
    kernel::Kernels.AbstractFilterKernel = Kernels.TopHatKernel(),
    workspace::Union{Nothing, Diagnostics.ΠWorkspace} = nothing,
    deriv_plan::Union{Nothing, Derivatives.UnstructuredWLSQGradientPlan} = nothing,
    filter_plans::Union{Nothing, AbstractVector} = nothing,
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    method::Filtering.AbstractFilterMethod = Filtering.Spectral(),
    L::Real = one(T),
) where {T<:AbstractFloat}
    _check_result_shape(result, grid, scales)
    ws = workspace === nothing ? Diagnostics.ΠWorkspace(grid) : workspace
    dplan = deriv_plan === nothing ? Derivatives.WLSQGradientPlan(grid) : deriv_plan

    # See the 2D `StructuredGrid` `coarse_grain!` method above for why `plans` is reused with
    # `cumulative_energy!` below, and why a repeated sweep should pass a prebuilt `filter_plans`.
    plans = filter_plans === nothing ? Vector{Filtering.AbstractFilterPlan}(undef, length(scales)) : filter_plans
    for s_idx in eachindex(scales)
        scale = T(scales[s_idx])
        result.scales[s_idx] = scale
        filter_plans === nothing &&
            (plans[s_idx] = Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend, method = method))
        Diagnostics.compute_Π!(
            view(result.Π, :, s_idx),
            u, v, w, grid, kernel, scale;
            workspace = ws, deriv_plan = dplan, filter_plan = plans[s_idx], backend = backend, mask_strategy = mask_strategy, method = method,
        )
    end

    Diagnostics.cumulative_energy!(
        result.cumulative_energy, u, v, w, grid, kernel, scales;
        workspace = ws, filter_plans = plans, backend = backend, mask_strategy = mask_strategy, method = method,
    )
    result.wavenumber .= T(L) ./ result.scales
    Diagnostics.spectral_density!(result.filtering_spectrum, result.cumulative_energy, result.wavenumber)
    return result
end

# 2D-velocity convenience wrapper (no vertical component).
function coarse_grain(
    u::AbstractVector,
    v::AbstractVector,
    grid::Grids.UnstructuredGrid{T};
    scales::AbstractVector,
    kernel::Kernels.AbstractFilterKernel = Kernels.TopHatKernel(),
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    method::Filtering.AbstractFilterMethod = Filtering.Spectral(),
    L::Real = one(T),
) where {T<:AbstractFloat}
    return coarse_grain(u, v, nothing, grid; scales=scales, kernel=kernel, backend=backend, mask_strategy=mask_strategy, method=method, L=L)
end

function coarse_grain!(
    result::CoarseGrainResult{T},
    u::AbstractVector,
    v::AbstractVector,
    grid::Grids.UnstructuredGrid{T};
    scales::AbstractVector,
    kernel::Kernels.AbstractFilterKernel = Kernels.TopHatKernel(),
    workspace::Union{Nothing, Diagnostics.ΠWorkspace} = nothing,
    deriv_plan::Union{Nothing, Derivatives.UnstructuredWLSQGradientPlan} = nothing,
    filter_plans::Union{Nothing, AbstractVector} = nothing,
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    method::Filtering.AbstractFilterMethod = Filtering.Spectral(),
    L::Real = one(T),
) where {T<:AbstractFloat}
    return coarse_grain!(result, u, v, nothing, grid; scales=scales, kernel=kernel, workspace=workspace, deriv_plan=deriv_plan, filter_plans=filter_plans, backend=backend, mask_strategy=mask_strategy, method=method, L=L)
end

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Allocates a result sized for `Nscales` scales over `grid`'s current spatial shape — dimension
# generic (a 1-tuple for UnstructuredGrid, 2-tuple for Structured/CurvilinearGrid, 3-tuple for a true
# 3D StructuredGrid), not hardcoded to 2D.
function _allocate_result(grid::Grids.AbstractGrid{G,T}, Nscales::Integer) where {G, T<:AbstractFloat}
    spatial = Grids.size_tuple(grid)
    N = length(spatial) + 1
    return CoarseGrainResult{T, N, Array{T,N}}(
        zeros(T, Nscales),
        zeros(T, spatial..., Nscales),
        zeros(T, Nscales),
        zeros(T, Nscales),
        zeros(T, Nscales),
    )
end

function _check_result_shape(result::CoarseGrainResult, grid::Grids.AbstractGrid, scales::AbstractVector)
    Nscales = length(scales)
    length(result.scales) == Nscales || throw(DimensionMismatch(
        "result holds $(length(result.scales)) scales, got $Nscales scales to sweep",
    ))
    size(result.Π, ndims(result.Π)) == Nscales || throw(DimensionMismatch(
        "result.Π's last dimension holds $(size(result.Π, ndims(result.Π))) scales, got $Nscales",
    ))
    size(result.Π)[1:(end-1)] == Grids.size_tuple(grid) || throw(DimensionMismatch(
        "result.Π's spatial shape $(size(result.Π)[1:(end-1)]) does not match grid shape $(Grids.size_tuple(grid))",
    ))
    return nothing
end

end # module
