module Pipeline

using FlowGeometries: FlowGeometries
using ..Kernels: Kernels
using ..Filtering: Filtering
using ..Derivatives: Derivatives
using ..Diagnostics: Diagnostics
using ComputationalBackends: ComputationalBackends

export CoarseGrainResult, coarse_grain, coarse_grain!

"""
    CoarseGrainResult(scales, Π, cumulative_energy, wavenumber, filtering_spectrum)

Container for results of a complete coarse-graining multiscale analysis. Every field's container type
is a type parameter, inferred by the constructor above, so nothing here is stored behind an abstract
annotation.

# Fields
- `scales::AbstractVector{T}`: filter scales ℓ in meters
- `Π::A`: energy-flux maps stacked into ONE contiguous `(spatial dims..., Nscales)` array (W/m³) —
  not a `Vector` of separately-allocated per-scale matrices, so the whole sweep is a single allocation
  and each scale's map is a zero-copy view (`compute_Π!` writes directly into its slice).
- `cumulative_energy::AbstractArray{T}`: cumulative coarse specific KE ½⟨|ū_ℓ|²⟩ per scale (Sadek–Aluie Eq.
  15) — a `Vector` (per scale) for `coarse_grain`, or a `(Nlevels, Nscales)` `Matrix` for
  `coarse_grain_profile` (per vertical level AND scale — deliberately not summed across levels, since
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
struct CoarseGrainResult{T<:AbstractFloat, N, A<:AbstractArray{T,N}, V<:AbstractVector{T}, C<:AbstractArray{T}}
    scales::V
    Π::A
    cumulative_energy::C
    wavenumber::V
    filtering_spectrum::C
end

# Every field is a type parameter rather than an `AbstractArray{T}` annotation: an abstract field type
# makes each access a dynamic dispatch, so even `wavenumber .= L ./ scales` allocated.
# `cumulative_energy` and `filtering_spectrum` share one parameter because they share a shape by
# construction — a vector per scale here, a `(level, scale)` matrix from `coarse_grain_profile`.
CoarseGrainResult(scales::AbstractVector{T}, Π::AbstractArray{T,N}, cumE::AbstractArray{T},
                  kℓ::AbstractVector{T}, spec::AbstractArray{T}) where {T<:AbstractFloat, N} =
    CoarseGrainResult{T, N, typeof(Π), typeof(scales), typeof(cumE)}(scales, Π, cumE, kℓ, spec)

# ---------------------------------------------------------------------------
# StructuredGrid pipeline
# ---------------------------------------------------------------------------

"""
    coarse_grain(u, v, w, grid; scales, kernel=TopHatKernel(), backend=AutoBackend(), mask_strategy=Deformable(), method=nothing, L=1)
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
- `mask_strategy::AbstractMaskStrategy=Deformable()`: Masking strategy (`ZeroFill()` or `Deformable()`)
- `method::Union{Nothing,AbstractFilterMethod}=nothing`: filtering engine; `nothing` takes
  `plan_filter`'s per-grid default (real space where a grid has that engine)
- `L::Real=1`: reference length setting the wavenumber normalization `k_ℓ = L/ℓ`

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
    grid::FlowGeometries.Grids.StructuredGrid{G,T};
    scales::AbstractVector,
    kernel::Kernels.AbstractFilterKernel = Kernels.TopHatKernel(),
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    method::Union{Nothing, Filtering.AbstractFilterMethod} = nothing,
    L::Real = one(T),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    result = _allocate_result(grid, length(scales))
    workspace = Diagnostics.ΠWorkspace(grid)
    return coarse_grain!(
        result, u, v, w, grid;
        scales = scales, kernel = kernel, workspace = workspace,
        backend = backend, mask_strategy = mask_strategy, method = method, L = L,
    )
end

# The derivative cache is geometry only — a stencil table on a rectilinear grid, a least-squares fit on
# a curvilinear or node grid — so one serves the whole scale sweep rather than being rebuilt per scale.
@inline _deriv_plan(grid::FlowGeometries.Grids.StructuredGrid, dp) =
    dp === nothing ? Derivatives.StencilPlan(grid) : dp
@inline _deriv_plan(grid, dp) = dp === nothing ? FlowGeometries.Connectivity.gradient_plan(grid) : dp

# `method === nothing` OMITS the keyword rather than forwarding it, so `plan_filter`'s per-grid
# default engine still applies.
@inline _plan_filter(grid, kernel, scale, strat, backend, method) =
    method === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy = strat, backend = backend) :
        Filtering.plan_filter(grid, kernel, scale; mask_strategy = strat, backend = backend, method = method)

@inline _flux!(out, u, v, w, grid, kernel, scale, ws, plan, backend, strat, dplan) =
    Diagnostics.compute_Π!(out, u, v, w, grid, kernel, scale;
        workspace = ws, filter_plan = plan, backend = backend, mask_strategy = strat, deriv_plan = dplan)

"""
    coarse_grain!(result, u, v, w, grid; scales, kernel, workspace, filter_plans, deriv_plan, backend, mask_strategy, method, L)
    coarse_grain!(result, u, v, grid; scales, ...)  # 2D convenience wrapper

In-place [`coarse_grain`](@ref): refills an existing [`CoarseGrainResult`](@ref)'s buffers — scales,
the stacked `Π` array, cumulative energy, wavenumber, spectrum — instead of allocating fresh ones.
Supplying `workspace` and `filter_plans` reuses the scratch arrays and the per-scale plans too, which
is the zero-reallocation entry point for a sweep repeated across timesteps.

`result` must already be sized for `length(scales)` scales over `grid`'s shape; a mismatch throws
`DimensionMismatch`.

One driver for every grid architecture: the scale loop, the plan reuse and the spectrum are the same
for all of them. Only the derivative plan and the trailing dimension of `result.Π` vary.
"""
function coarse_grain!(
    result::CoarseGrainResult{T},
    u::AbstractArray,
    v::AbstractArray,
    w::Union{Nothing, AbstractArray},
    grid::FlowGeometries.Grids.AbstractGrid{G,T};
    scales::AbstractVector,
    kernel::Kernels.AbstractFilterKernel = Kernels.TopHatKernel(),
    workspace::Union{Nothing, Diagnostics.ΠWorkspace} = nothing,
    filter_plans::Union{Nothing, AbstractVector} = nothing,
    deriv_plan = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    # `nothing` leaves the choice to `plan_filter`, whose default differs by grid: real space where a
    # grid has that engine, spectral for a node set.
    method::Union{Nothing, Filtering.AbstractFilterMethod} = nothing,
    L::Real = one(T),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    _check_result_shape(result, grid, scales)
    ws = workspace === nothing ? Diagnostics.ΠWorkspace(grid) : workspace
    dplan = _deriv_plan(grid, deriv_plan)
    # Each scale's plan is shared by `compute_Π!` and `cumulative_energy!`. A sweep repeated over many
    # timesteps of the same grid, kernel and scales should pass a prebuilt `filter_plans`, so neither
    # the vector nor the plans are reallocated.
# Built as a comprehension so the element type is the plans' own concrete type. An
# `AbstractFilterPlan` element type makes every `plans[s_idx]` a dynamic dispatch: measured at 3.1% of
# a 256x256 8-scale sweep and 21 kB, against 0 B concrete. Plan types that genuinely differ across
# scales (a cache strategy that flips with window width) still work — the comprehension then infers
# their union or supertype instead, exactly as before.
    plans = filter_plans === nothing ?
        [_plan_filter(grid, kernel, T(s), mask_strategy, backend, method) for s in scales] : filter_plans
    # `E(ℓ)` is read from the filtered velocities `compute_Π!` has just left in the workspace, in the
    # same pass. Running the energy sweep afterwards instead would filter `u` and `v` a second time at
    # every scale — two of the seven applies per scale — because the buffers holding them are
    # overwritten by the next scale.
    total_area = Diagnostics.active_area(grid)
    for s_idx in eachindex(scales)
        scale = T(scales[s_idx])
        result.scales[s_idx] = scale
        _flux!(
            selectdim(result.Π, ndims(result.Π), s_idx), u, v, w, grid, kernel, scale,
            ws, plans[s_idx], backend, mask_strategy, dplan,
        )
        result.cumulative_energy[s_idx] =
            Diagnostics.energy_from_filtered(ws, grid, w !== nothing, total_area)
    end
    result.wavenumber .= T(L) ./ result.scales
    Diagnostics.spectral_density!(result.filtering_spectrum, result.cumulative_energy, result.wavenumber)
    return result
end

# 2.5D Cartesian constructor wrapper (2D velocity fields without a vertical component)
function coarse_grain(
    u::AbstractMatrix,
    v::AbstractMatrix,
    grid::FlowGeometries.Grids.StructuredGrid{G,T};
    scales::AbstractVector,
    kernel::Kernels.AbstractFilterKernel = Kernels.TopHatKernel(),
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    method::Union{Nothing, Filtering.AbstractFilterMethod} = nothing,
    L::Real = one(T),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    return coarse_grain(u, v, nothing, grid; scales=scales, kernel=kernel, backend=backend, mask_strategy=mask_strategy, method=method, L=L)
end

function coarse_grain!(
    result::CoarseGrainResult{T},
    u::AbstractMatrix,
    v::AbstractMatrix,
    grid::FlowGeometries.Grids.StructuredGrid{G,T};
    scales::AbstractVector,
    kernel::Kernels.AbstractFilterKernel = Kernels.TopHatKernel(),
    workspace::Union{Nothing, Diagnostics.ΠWorkspace} = nothing,
    filter_plans::Union{Nothing, AbstractVector} = nothing,
    deriv_plan = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    method::Union{Nothing, Filtering.AbstractFilterMethod} = nothing,
    L::Real = one(T),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    return coarse_grain!(result, u, v, nothing, grid; scales=scales, kernel=kernel, workspace=workspace, filter_plans=filter_plans, deriv_plan=deriv_plan, backend=backend, mask_strategy=mask_strategy, method=method, L=L)
end

"""
    coarse_grain_profile(u, v, w, grid; scales, kernel=TopHatKernel(), backend=AutoBackend(), mask_strategy=Deformable(), method=nothing, L=1)

Vertical-profile sweep: given 3D `(x, y, z)` velocity arrays, runs [`Diagnostics.compute_Π_profile!`](@ref)
(the literature-standard independent-per-level 2D/2.5D method — see [`Diagnostics.compute_Π!`](@ref)'s
thin-layer/QG regime note) at every scale, returning a [`CoarseGrainResult`](@ref) whose `Π` is one
contiguous `(Nx, Ny, Nlevels, Nscales)` array. The workspace is built once and reused across the
whole level × scale sweep.
"""
function coarse_grain_profile(
    u::AbstractArray{T,3},
    v::AbstractArray{T,3},
    w::Union{Nothing, AbstractArray{T,3}},
    grid::FlowGeometries.Grids.StructuredGrid{G,T};
    scales::AbstractVector,
    kernel::Kernels.AbstractFilterKernel = Kernels.TopHatKernel(),
    workspace::Union{Nothing, Diagnostics.ΠWorkspace} = nothing,
    filter_plans::Union{Nothing, AbstractVector} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    method::Union{Nothing, Filtering.AbstractFilterMethod} = nothing,
    deriv_plan::Union{Nothing, Derivatives.StencilPlan} = nothing,
    L::Real = one(T),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    # Grid-only, so one table serves every level of every scale.
    dplan = deriv_plan === nothing ? Derivatives.StencilPlan(grid) : deriv_plan
    Nscales = length(scales)
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    Nlevels = size(u, 3)

    Π = zeros(T, Nx, Ny, Nlevels, Nscales)
    ws = workspace === nothing ? Diagnostics.ΠWorkspace(grid) : workspace
    scales_vec = zeros(T, Nscales)
    # One filter plan per scale, reused both across all levels of the Π sweep below AND across all
    # levels of the cumulative-energy sweep after it — otherwise the energy sweep (looping per
    # level, each internally looping per scale) would rebuild the same Nscales footprints again for
    # every one of the Nlevels iterations, on top of what the Π sweep already built (confirmed by
    # measurement on the plain 2D `coarse_grain!` case this mirrors). For a sweep repeated across
    # many timesteps of the SAME grid/kernel/scales, pass a prebuilt `filter_plans` (and `workspace`)
    # so this doesn't allocate a fresh plan vector — or fresh plans — on every repeat call.
# Built as a comprehension so the element type is the plans' own concrete type. An
# `AbstractFilterPlan` element type makes every `plans[s_idx]` a dynamic dispatch: measured at 3.1% of
# a 256x256 8-scale sweep and 21 kB, against 0 B concrete. Plan types that genuinely differ across
# scales (a cache strategy that flips with window width) still work — the comprehension then infers
# their union or supertype instead, exactly as before.
    plans = filter_plans === nothing ?
        [_plan_filter(grid, kernel, T(s), mask_strategy, backend, method) for s in scales] : filter_plans
    for s_idx in 1:Nscales
        scale = T(scales[s_idx])
        scales_vec[s_idx] = scale
        Diagnostics.compute_Π_profile!(
            view(Π, :, :, :, s_idx), u, v, w, grid, kernel, scale;
            workspace = ws, filter_plan = plans[s_idx], backend = backend, mask_strategy = mask_strategy,
            deriv_plan = dplan,
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
    return CoarseGrainResult(scales_vec, Π, cumE, kℓ, spec)
end

# 2.5D convenience wrapper (no vertical velocity).
function coarse_grain_profile(
    u::AbstractArray{T,3},
    v::AbstractArray{T,3},
    grid::FlowGeometries.Grids.StructuredGrid{G,T};
    scales::AbstractVector,
    kernel::Kernels.AbstractFilterKernel = Kernels.TopHatKernel(),
    workspace::Union{Nothing, Diagnostics.ΠWorkspace} = nothing,
    filter_plans::Union{Nothing, AbstractVector} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    method::Union{Nothing, Filtering.AbstractFilterMethod} = nothing,
    deriv_plan::Union{Nothing, Derivatives.StencilPlan} = nothing,
    L::Real = one(T),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    return coarse_grain_profile(u, v, nothing, grid; scales=scales, kernel=kernel, workspace=workspace, filter_plans=filter_plans, backend=backend, mask_strategy=mask_strategy, method=method, deriv_plan=deriv_plan, L=L)
end

# ---------------------------------------------------------------------------
# 1D Cartesian pipeline: a single scalar velocity component `u` along one axis (a genuinely 1D
# `StructuredGrid`, not a 2D grid with a singleton dimension). Cumulative energy is computed directly
# here (0.5⟨ū_ℓ²⟩), not via the shared `cumulative_energy!` — that function's signature genuinely
# needs two velocity components (u,v), which don't exist in a 1D flow.
# ---------------------------------------------------------------------------
function coarse_grain(
    u::AbstractVector,
    grid::FlowGeometries.Grids.StructuredGrid{G,T,1};
    scales::AbstractVector,
    kernel::Kernels.AbstractFilterKernel = Kernels.TopHatKernel(),
    filter_plans::Union{Nothing, AbstractVector} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    method::Union{Nothing, Filtering.AbstractFilterMethod} = nothing,
    L::Real = one(T),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    result = _allocate_result(grid, length(scales))
    workspace = Diagnostics.ΠWorkspace(grid)
    return coarse_grain!(
        result, u, grid;
        scales = scales, kernel = kernel, workspace = workspace, filter_plans = filter_plans,
        backend = backend, mask_strategy = mask_strategy, method = method, L = L,
    )
end

function coarse_grain!(
    result::CoarseGrainResult{T},
    u::AbstractVector,
    grid::FlowGeometries.Grids.StructuredGrid{G,T,1};
    scales::AbstractVector,
    kernel::Kernels.AbstractFilterKernel = Kernels.TopHatKernel(),
    workspace::Union{Nothing, Diagnostics.ΠWorkspace} = nothing,
    filter_plans::Union{Nothing, AbstractVector} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    method::Union{Nothing, Filtering.AbstractFilterMethod} = nothing,
    L::Real = one(T),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    _check_result_shape(result, grid, scales)
    ws = workspace === nothing ? Diagnostics.ΠWorkspace(grid) : workspace
# Built as a comprehension so the element type is the plans' own concrete type. An
# `AbstractFilterPlan` element type makes every `plans[s_idx]` a dynamic dispatch: measured at 3.1% of
# a 256x256 8-scale sweep and 21 kB, against 0 B concrete. Plan types that genuinely differ across
# scales (a cache strategy that flips with window width) still work — the comprehension then infers
# their union or supertype instead, exactly as before.
    plans = filter_plans === nothing ?
        [_plan_filter(grid, kernel, T(s), mask_strategy, backend, method) for s in scales] : filter_plans
    Nx = FlowGeometries.Grids.size_tuple(grid)[1]

    total_area = sum(FlowGeometries.Grids.area(grid, i) for i in 1:Nx if FlowGeometries.Grids.isactive(grid, i))

    for s_idx in eachindex(scales)
        scale = T(scales[s_idx])
        result.scales[s_idx] = scale
        # One plan per scale, shared by the flux and the energy integral below, and reusable across
        # calls through `filter_plans` — as every other grid type's method already allows.
        plan = plans[s_idx]
        Diagnostics.compute_Π!(
            view(result.Π, :, s_idx),
            u, grid, kernel, scale;
            workspace = ws, filter_plan = plan, backend = backend, mask_strategy = mask_strategy,
        )
        Filtering.filter_apply!(ws.u_filt, u, plan)
        integrated_energy = sum(ws.u_filt[i]^2 * FlowGeometries.Grids.area(grid, i) for i in 1:Nx if FlowGeometries.Grids.isactive(grid, i))
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
# grid with a level-stacked array. `Diagnostics.compute_Π!` itself dispatches Cartesian vs. spherical.
# ---------------------------------------------------------------------------
function coarse_grain(
    u::AbstractArray{<:Any,3},
    v::AbstractArray{<:Any,3},
    w::AbstractArray{<:Any,3},
    grid::FlowGeometries.Grids.StructuredGrid{G,T,3};
    scales::AbstractVector,
    kernel::Kernels.AbstractFilterKernel = Kernels.TopHatKernel(),
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    method::Union{Nothing, Filtering.AbstractFilterMethod} = nothing,
    L::Real = one(T),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    result = _allocate_result(grid, length(scales))
    workspace = Diagnostics.ΠWorkspace(grid)
    return coarse_grain!(
        result, u, v, w, grid;
        scales = scales, kernel = kernel, workspace = workspace,
        backend = backend, mask_strategy = mask_strategy, method = method, L = L,
    )
end


# ---------------------------------------------------------------------------
# Curvilinear-grid pipeline: same orchestration as the StructuredGrid path, but the WLSQ derivative
# plan (like the workspace) is built ONCE and reused across the whole scale sweep.
# ---------------------------------------------------------------------------
function coarse_grain(
    u::AbstractMatrix,
    v::AbstractMatrix,
    w::Union{Nothing, AbstractMatrix},
    grid::FlowGeometries.Grids.CurvilinearGrid{T,G};
    scales::AbstractVector,
    kernel::Kernels.AbstractFilterKernel = Kernels.TopHatKernel(),
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    method::Union{Nothing, Filtering.AbstractFilterMethod} = nothing,
    L::Real = one(T),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    result = _allocate_result(grid, length(scales))
    workspace = Diagnostics.ΠWorkspace(grid)
    deriv_plan = FlowGeometries.Connectivity.gradient_plan(grid)
    return coarse_grain!(
        result, u, v, w, grid;
        scales = scales, kernel = kernel, workspace = workspace, deriv_plan = deriv_plan,
        backend = backend, mask_strategy = mask_strategy, method = method, L = L,
    )
end


# 2D curvilinear convenience wrapper (no vertical velocity).
function coarse_grain(
    u::AbstractMatrix,
    v::AbstractMatrix,
    grid::FlowGeometries.Grids.CurvilinearGrid{T,G};
    scales::AbstractVector,
    kernel::Kernels.AbstractFilterKernel = Kernels.TopHatKernel(),
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    method::Union{Nothing, Filtering.AbstractFilterMethod} = nothing,
    L::Real = one(T),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    return coarse_grain(u, v, nothing, grid; scales=scales, kernel=kernel, backend=backend, mask_strategy=mask_strategy, method=method, L=L)
end

function coarse_grain!(
    result::CoarseGrainResult{T},
    u::AbstractMatrix,
    v::AbstractMatrix,
    grid::FlowGeometries.Grids.CurvilinearGrid{T,G};
    scales::AbstractVector,
    kernel::Kernels.AbstractFilterKernel = Kernels.TopHatKernel(),
    workspace::Union{Nothing, Diagnostics.ΠWorkspace} = nothing,
    deriv_plan::Union{Nothing, FlowGeometries.Discretization.GradientPlan} = nothing,
    filter_plans::Union{Nothing, AbstractVector} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    method::Union{Nothing, Filtering.AbstractFilterMethod} = nothing,
    L::Real = one(T),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    return coarse_grain!(result, u, v, nothing, grid; scales=scales, kernel=kernel, workspace=workspace, deriv_plan=deriv_plan, filter_plans=filter_plans, backend=backend, mask_strategy=mask_strategy, method=method, L=L)
end

# ---------------------------------------------------------------------------
# UnstructuredGrid pipeline: 1D node-indexed, same orchestration pattern as CurvilinearGrid — the same
# `Connectivity.gradient_plan`, built over the stored adjacency rather than an index stencil — and
# defaulting to spectral filtering.
# ---------------------------------------------------------------------------
function coarse_grain(
    u::AbstractVector,
    v::AbstractVector,
    w::Union{Nothing, AbstractVector},
    grid::FlowGeometries.Grids.UnstructuredGrid{T};
    scales::AbstractVector,
    kernel::Kernels.AbstractFilterKernel = Kernels.TopHatKernel(),
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    method::Filtering.AbstractFilterMethod = Filtering.Spectral(),
    L::Real = one(T),
) where {T<:AbstractFloat}
    result = _allocate_result(grid, length(scales))
    workspace = Diagnostics.ΠWorkspace(grid)
    deriv_plan = FlowGeometries.Connectivity.gradient_plan(grid)
    return coarse_grain!(
        result, u, v, w, grid;
        scales = scales, kernel = kernel, workspace = workspace, deriv_plan = deriv_plan,
        backend = backend, mask_strategy = mask_strategy, method = method, L = L,
    )
end


# 2D-velocity convenience wrapper (no vertical component).
function coarse_grain(
    u::AbstractVector,
    v::AbstractVector,
    grid::FlowGeometries.Grids.UnstructuredGrid{T};
    scales::AbstractVector,
    kernel::Kernels.AbstractFilterKernel = Kernels.TopHatKernel(),
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
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
    grid::FlowGeometries.Grids.UnstructuredGrid{T};
    scales::AbstractVector,
    kernel::Kernels.AbstractFilterKernel = Kernels.TopHatKernel(),
    workspace::Union{Nothing, Diagnostics.ΠWorkspace} = nothing,
    deriv_plan::Union{Nothing, FlowGeometries.Discretization.GradientPlan} = nothing,
    filter_plans::Union{Nothing, AbstractVector} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
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
function _allocate_result(grid::FlowGeometries.Grids.AbstractGrid{G,T}, Nscales::Integer) where {G, T<:AbstractFloat}
    spatial = FlowGeometries.Grids.size_tuple(grid)
    return CoarseGrainResult(
        zeros(T, Nscales),
        zeros(T, spatial..., Nscales),
        zeros(T, Nscales),
        zeros(T, Nscales),
        zeros(T, Nscales),
    )
end

function _check_result_shape(result::CoarseGrainResult, grid::FlowGeometries.Grids.AbstractGrid, scales::AbstractVector)
    Nscales = length(scales)
    length(result.scales) == Nscales || throw(DimensionMismatch(
        "result holds $(length(result.scales)) scales, got $Nscales scales to sweep",
    ))
    size(result.Π, ndims(result.Π)) == Nscales || throw(DimensionMismatch(
        "result.Π's last dimension holds $(size(result.Π, ndims(result.Π))) scales, got $Nscales",
    ))
    size(result.Π)[1:(end-1)] == FlowGeometries.Grids.size_tuple(grid) || throw(DimensionMismatch(
        "result.Π's spatial shape $(size(result.Π)[1:(end-1)]) does not match grid shape $(FlowGeometries.Grids.size_tuple(grid))",
    ))
    return nothing
end

end # module
