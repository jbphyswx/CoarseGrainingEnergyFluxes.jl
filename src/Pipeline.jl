module Pipeline

using FlowGeometries: FlowGeometries
using ..Kernels: Kernels
using ..Filtering: Filtering
using ..Derivatives: Derivatives
using ..Diagnostics: Diagnostics
using ComputationalBackends: ComputationalBackends

export CoarseGrainResult, coarse_grain, coarse_grain!
export CoarseGrainBatchResult, coarse_grain_batch!, coarse_grain_slices!
export coarse_grain_profile, coarse_grain_profile!
export check_setup, SetupReport

# ---------------------------------------------------------------------------
# check_setup — what will actually run, and what this (grid, kernel, ℓ) can support
# ---------------------------------------------------------------------------

"""
    SetupReport

What [`check_setup`](@ref) found. Printed as a readable report, and every field is also readable
programmatically so a script can gate on it.

# Fields
- `grid_kind::String`, `grid_size::Tuple`, `spacing::Tuple`: the grid, and the SMALLEST spacing per
  axis (the one that sets stencil widths)
- `scale::Float64`, `cells_per_scale::Tuple`: `ℓ`, and `ℓ / Δx` per axis
- `kernel::String`, `mask_strategy::String`, `method::String`
- `backend_requested::String`, `backend_resolved::String`: what was asked for, and what will run
- `engine::String`: which real-space engine class the filter will take
- `resolvable::Bool`: `ℓ` is wide enough to mean anything on this grid (`> 2Δx` on every axis)
- `supports_flux::Bool`: the kernel is non-negative, so `τ` is realizable and `Π` is a pointwise
  transfer
- `supports_spectrum::Bool`: [`Kernels.transfer_monotone`](@ref) — whether this kernel's spectral
  density is *guaranteed* non-negative, and so whether the default
  [`Diagnostics.StrictSpectrum`](@ref) policy will accept it. `false` does not mean unreachable:
  `Diagnostics.ForceSpectrum()` computes it regardless.
- `supports_spectral_method::Bool`: the kernel has an isotropic transfer function, so
  `method = Spectral()` is available
- `boundary_buffer_cells::Tuple`: how many cells in from a coast or domain edge are contaminated —
  the kernel radius in cells. Results inside this band are not trustworthy under any mask strategy.
- `notes::Vector{String}`: everything above that needs acting on, in words
"""
struct SetupReport
    grid_kind::String
    grid_size::Tuple
    spacing::Tuple
    scale::Float64
    cells_per_scale::Tuple
    kernel::String
    mask_strategy::String
    method::String
    backend_requested::String
    backend_resolved::String
    engine::String
    resolvable::Bool
    supports_flux::Bool
    supports_spectrum::Bool
    supports_spectral_method::Bool
    boundary_buffer_cells::Tuple
    notes::Vector{String}
end

function Base.show(io::IO, ::MIME"text/plain", r::SetupReport)
    yn(b) = b ? "yes" : "NO"
    println(io, "CoarseGrainingEnergyFluxes setup check")
    println(io, "  grid            : ", r.grid_kind, " ", r.grid_size,
                "   min spacing ", map(x -> round(x; sigdigits = 4), r.spacing))
    println(io, "  scale ℓ         : ", r.scale, "  = ",
                map(x -> round(x; digits = 2), r.cells_per_scale), " cells per axis")
    println(io, "  kernel          : ", r.kernel)
    println(io, "  masking         : ", r.mask_strategy)
    println(io, "  method          : ", r.method)
    println(io, "  backend         : ", r.backend_requested, " -> ", r.backend_resolved)
    println(io, "  real-space engine: ", r.engine)
    println(io, "  ℓ resolvable    : ", yn(r.resolvable))
    println(io, "  supports Π      : ", yn(r.supports_flux))
    println(io, "  spectrum ≥ 0    : ", yn(r.supports_spectrum))
    println(io, "  supports Spectral(): ", yn(r.supports_spectral_method))
    println(io, "  boundary buffer : ", r.boundary_buffer_cells, " cells (contaminated)")
    if isempty(r.notes)
        print(io, "  nothing to flag.")
    else
        println(io, "  notes:")
        for (i, n) in enumerate(r.notes)
            print(io, "    ", i, ". ", n, i == length(r.notes) ? "" : "\n")
        end
    end
    return nothing
end

Base.show(io::IO, r::SetupReport) =
    print(io, "SetupReport(", r.grid_kind, " ", r.grid_size, ", ", r.kernel,
          ", ℓ=", r.scale, ", ", length(r.notes), " note(s))")

# Unqualified, with the parameters that actually change the kernel — `show` on the struct prints the
# full module path, which is noise in a report.
_kname(k::Kernels.AbstractFilterKernel) = string(nameof(typeof(k)))
_kname(k::Kernels.GaussianKernel) = "GaussianKernel(α = $(k.α))"
_kname(k::Kernels.HyperGaussianKernel) = "HyperGaussianKernel(α = $(k.α))"
_kname(k::Kernels.SmoothHatKernel) = "SmoothHatKernel(steepness = $(k.steepness))"
_kname(k::Kernels.HighOrderKernel{P}) where {P} = "HighOrderKernel{$P}(b_over_ℓ = $(k.b_over_ℓ))"

_grid_kind(::FlowGeometries.Grids.StructuredGrid{G,T,N}) where {G,T,N} = "StructuredGrid{$(nameof(G)),$N}"
_grid_kind(::FlowGeometries.Grids.CurvilinearGrid{T,G}) where {T,G} = "CurvilinearGrid{$(nameof(G))}"
_grid_kind(::FlowGeometries.Grids.UnstructuredGrid{T,G}) where {T,G} = "UnstructuredGrid{$(nameof(G))}"
_grid_kind(g) = string(nameof(typeof(g)))

# Smallest spacing per axis. A node set has no axes, so its "spacing" is reported as the mean nearest
# neighbour distance would have to be — left empty rather than guessed.
_axis_spacings(grid::FlowGeometries.Grids.StructuredGrid{G,T,N}) where {G,T,N} =
    ntuple(d -> FlowGeometries.Grids.minimum_spacing(grid, d), Val(N))
_axis_spacings(::FlowGeometries.Grids.AbstractGrid) = ()

# Which real-space engine class the filter will take, from the same predicates the builder dispatches
# on. Derived rather than read off a built plan: a scattered plan at a wide radius can be gigabytes.
function _engine_class(grid, kernel)
    Kernels.is_separable(kernel) && grid isa FlowGeometries.Grids.StructuredGrid &&
        FlowGeometries.Grids.measure_factors(grid) !== nothing &&
        return "separable, two-pass O(N·Σw) per axis"
    kernel isa Kernels.TopHatKernel && grid isa FlowGeometries.Grids.StructuredGrid{<:Any,<:Any,2} &&
        FlowGeometries.Grids.measure_factors(grid) !== nothing &&
        return "prefix-sum top-hat, exact O(N)"
    grid isa FlowGeometries.Grids.StructuredGrid{<:Any,<:Any,2} &&
        return "banded footprint, O(N·w²)"
    return "scattered per-point footprint, O(N·w²) with a neighbour cache"
end

"""
    check_setup(grid, kernel, scale; backend, mask_strategy, method) -> SetupReport

Report what a filter/flux call on this `(grid, kernel, scale)` will actually do, and what it can
support, **without running or even planning it**. Cheap: it consults the same dispatch predicates the
builders use rather than constructing a plan, so it is safe on a configuration whose plan would be
enormous.

It answers the questions that otherwise need reading `src/`:

- which real-space engine will run, and which backend the request resolves to;
- whether `ℓ` is meaningful on this grid at all (`ℓ > 2Δx`), and how many cells it spans;
- whether this kernel can carry `Π` (is it non-negative?) and a filtering spectrum
  (is `|Ĝ|²` monotone?), and whether `method = Spectral()` exists for it;
- how wide the contaminated band along a coast or domain edge is.

# Examples
```julia
julia> CGEF.check_setup(grid, CGEF.TopHatKernel(), 5000.0)
```
"""
function check_setup(
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::Real;
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Union{Nothing, Filtering.AbstractFilterMethod} = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    ℓ = T(scale)
    ℓ > 0 || throw(ArgumentError("check_setup needs a positive scale, got $scale"))
    notes = String[]
    sp = _axis_spacings(grid)
    cps = map(s -> (isfinite(s) && s > 0) ? ℓ / s : T(NaN), sp)
    resolvable = isempty(cps) ? true : all(c -> !isnan(c) && c > 2, cps)
    resolvable || push!(notes,
        "ℓ spans $(map(x -> round(x; digits = 2), cps)) cells; below ~2 cells on an axis the filter " *
        "is a no-op on that axis. Increase ℓ or coarsen the grid.")

    rad = Kernels.kernel_radius(kernel, ℓ)
    buf = map(s -> (isfinite(s) && s > 0) ? ceil(Int, rad / s) : 0, sp)
    isempty(buf) || push!(notes,
        "points within $(buf) cells of a coast or domain edge are contaminated by footprint " *
        "truncation, under either mask strategy — exclude them before averaging.")

    resolved = try
        string(nameof(typeof(Filtering._resolve_backend(backend, grid))))
    catch e
        push!(notes, "backend $(nameof(typeof(backend))) is not available for this grid: " *
                     first(sprint(showerror, e), 200))
        "UNAVAILABLE"
    end

    flux_ok = Kernels.is_radial(kernel) ?
        all(r -> Kernels.kernel_weight(kernel, T(r), ℓ) >= 0, range(zero(T), rad; length = 512)) :
        all(>=(0), Kernels.limb_amplitudes(kernel, T))
    flux_ok || push!(notes,
        "$(_kname(kernel)) takes NEGATIVE values, so τ is not realizable and Π is not a pointwise " *
        "energy transfer. Use TopHatKernel or GaussianKernel for a flux.")

    spec_ok = Kernels.transfer_monotone(kernel)
    spec_ok || push!(notes,
        "$(_kname(kernel))'s |Ĝ|² is not monotone, so a spectral density is not guaranteed " *
        "non-negative: `coarse_grain` needs `kernel = GaussianKernel()`, or " *
        "`spectrum = Diagnostics.ForceSpectrum()` to compute it anyway, or " *
        "`spectrum = Diagnostics.NoSpectrum()` to skip it.")

    # Two different reasons `Spectral()` can be unavailable, and they need different actions: a
    # `MethodError` means the transfer function lives in an extension that is not loaded (fixable by
    # loading it), an `ArgumentError` means the kernel genuinely has no isotropic transfer function.
    spectral_ok, spectral_note = try
        Kernels.spectral_transfer(kernel, one(T) / ℓ, ℓ)
        true, nothing
    catch e
        if e isa MethodError
            false, "$(_kname(kernel))'s spectral transfer function is provided by a weak dependency " *
                   "that is not loaded, so `method = Spectral()` is unavailable in this session — run " *
                   "`using SpecialFunctions`. Real-space filtering is unaffected."
        else
            false, "$(_kname(kernel)) has no isotropic transfer function, so `method = Spectral()` " *
                   "does not exist for it — it is a real-space kernel by construction."
        end
    end
    spectral_note === nothing || push!(notes, spectral_note)

    if kernel isa Kernels.HighOrderKernel && !isempty(sp)
        b = kernel.b_over_ℓ * ℓ
        mn = minimum(sp)
        b < mn && push!(notes,
            "each limb is $(round(b / mn; digits = 2)) cells wide; below 1 the vanishing moments do " *
            "not survive discretization. This kernel needs ℓ ≥ $(round(mn / kernel.b_over_ℓ)).")
    end
    # A node set has no axes, so resolvability and the boundary buffer could not be checked at all —
    # say so rather than report a clean bill of health.
    if isempty(sp)
        push!(notes,
            "this grid has no axes, so ℓ-resolvability and the boundary buffer could not be checked " *
            "here; compare ℓ against the typical nearest-neighbour distance yourself " *
            "(the kernel radius is $(round(rad; sigdigits = 4))).")
    end

    return SetupReport(
        _grid_kind(grid), FlowGeometries.Grids.size_tuple(grid), sp, Float64(ℓ), cps,
        _kname(kernel), string(nameof(typeof(mask_strategy))),
        method === nothing ? "default for this grid" : string(nameof(typeof(method))),
        string(nameof(typeof(backend))), resolved, _engine_class(grid, kernel),
        resolvable, flux_ok, spec_ok, spectral_ok, buf, notes,
    )
end

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
    coarse_grain(u, v, w, grid; scales, kernel=TopHatKernel(), backend=AutoBackend(), mask_strategy=ZeroFill(), method=nothing, L=1, spectrum=StrictSpectrum())
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
- `mask_strategy::AbstractMaskStrategy=ZeroFill()`: Masking strategy (`ZeroFill()` or `Deformable()`).
  `ZeroFill` is the default because it keeps the kernel position-independent, so filtering commutes
  with spatial derivatives — the property the flux budget is derived by. See
  [`Filtering.filter_field!`](@ref) for the boundary artifacts of both choices.
- `method::Union{Nothing,AbstractFilterMethod}=nothing`: filtering engine; `nothing` takes
  `plan_filter`'s per-grid default (real space where a grid has that engine)
- `L::Real=1`: reference length setting the wavenumber normalization `k_ℓ = L/ℓ`. The choice rescales
  the spectral density by `1/L` — see [`Diagnostics.filtering_spectrum`](@ref).
- `spectrum::AbstractSpectrumPolicy=Diagnostics.StrictSpectrum()`: how to fill `filtering_spectrum`.
  The spectral density is guaranteed non-negative only for a kernel whose `|Ĝ|²` is monotone decreasing
  ([`Kernels.transfer_monotone`](@ref)), which the default `TopHatKernel` is **not** — so
  `coarse_grain(u, v, grid; scales)` throws under the default policy. The alternatives are
  `Diagnostics.ForceSpectrum()` (compute it anyway, warning once — the condition is sufficient, not
  necessary), `Diagnostics.NoSpectrum()` (fill `NaN`), or `kernel = GaussianKernel()`. `Π` and
  `cumulative_energy` are unaffected under every policy, since neither depends on that condition.

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
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Union{Nothing, Filtering.AbstractFilterMethod} = nothing,
    L::Real = one(T),
    spectrum::Diagnostics.AbstractSpectrumPolicy = Diagnostics.StrictSpectrum(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    result = _allocate_result(grid, length(scales))
    workspace = Diagnostics.ΠWorkspace(grid)
    return coarse_grain!(
        result, u, v, w, grid;
        scales = scales, kernel = kernel, workspace = workspace,
        backend = backend, mask_strategy = mask_strategy, method = method, L = L, spectrum = spectrum,
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

# Only the paths that can share a scale-independent analysis are handed one; the keyword is omitted
# entirely otherwise, so a grid whose flux method has no such half never sees it.
@inline _flux!(out, u, v, w, grid, kernel, scale, ws, plan, backend, strat, dplan, ::Nothing = nothing) =
    Diagnostics.compute_Π!(out, u, v, w, grid, kernel, scale;
        workspace = ws, filter_plan = plan, backend = backend, mask_strategy = strat, deriv_plan = dplan)

@inline _flux!(out, u, v, w, grid, kernel, scale, ws, plan, backend, strat, dplan, analyzed) =
    Diagnostics.compute_Π!(out, u, v, w, grid, kernel, scale;
        workspace = ws, filter_plan = plan, backend = backend, mask_strategy = strat,
        deriv_plan = dplan, analyzed = analyzed)

# The filtering spectral DENSITY is only guaranteed non-negative for a kernel whose |Ĝ|² is monotone
# decreasing, which the default `TopHatKernel` is not — see `Kernels.transfer_monotone`. `Π` and the
# cumulative energy carry no such condition, so which reading applies is the caller's to pick:
# `Diagnostics.StrictSpectrum` (throw), `ForceSpectrum` (compute and warn) or `NoSpectrum` (fill NaN).
@inline function _check_spectrum(kernel, spectrum::Diagnostics.AbstractSpectrumPolicy)
    Diagnostics.gate_spectrum(kernel, spectrum)
    return spectrum
end

@inline _fill_spectrum!(g, C, k, ::Diagnostics.NoSpectrum) = fill!(g, convert(eltype(g), NaN))
@inline _fill_spectrum!(g, C, k, ::Diagnostics.AbstractSpectrumPolicy) =
    Diagnostics.spectral_density!(g, C, k)

"""
    coarse_grain!(result, u, v, w, grid; scales, kernel, workspace, filter_plans, deriv_plan, backend, mask_strategy, method, L, spectrum)
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
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    # `nothing` leaves the choice to `plan_filter`, whose default differs by grid: real space where a
    # grid has that engine, spectral for a node set.
    method::Union{Nothing, Filtering.AbstractFilterMethod} = nothing,
    L::Real = one(T),
    spectrum::Diagnostics.AbstractSpectrumPolicy = Diagnostics.StrictSpectrum(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    _check_result_shape(result, grid, scales)
    _check_spectrum(kernel, spectrum)
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
    # A spectral filter's forward transform depends on the field, not the scale, and every field the flux
    # computation filters is raw — the velocities and their products. Transforming them once here rather
    # than once per scale takes a sweep from 10·S transforms to 5 + 5S. Real-space engines have no such
    # half to share and return `nothing`, leaving the per-scale path exactly as it was.
    analyzed = isempty(scales) ? nothing : Diagnostics.analyze_sweep(u, v, w, grid, ws, first(plans))
    for s_idx in eachindex(scales)
        scale = T(scales[s_idx])
        result.scales[s_idx] = scale
        _flux!(
            selectdim(result.Π, ndims(result.Π), s_idx), u, v, w, grid, kernel, scale,
            ws, plans[s_idx], backend, mask_strategy, dplan, analyzed,
        )
        result.cumulative_energy[s_idx] =
            Diagnostics.energy_from_filtered(ws, grid, w !== nothing, total_area)
    end
    result.wavenumber .= T(L) ./ result.scales
    _fill_spectrum!(result.filtering_spectrum, result.cumulative_energy, result.wavenumber, spectrum)
    return result
end

# ---------------------------------------------------------------------------
# Batch axis: independent sweeps sharing one grid
# ---------------------------------------------------------------------------

# Batch-parallel form: many INDEPENDENT sweeps, one scratch set per worker. A different axis from
# `coarse_grain!`'s own `backend`, which parallelizes inside a single sweep.
function threaded_coarse_grain_batch!(args...; kwargs...)
    throw(ArgumentError("ThreadedBackend is unavailable — run `using OhMyThreads` (or use SerialBackend())."))
end

function distributed_coarse_grain_batch!(args...; kwargs...)
    throw(ArgumentError("DistributedBackend is unavailable — run `using Distributed` (or use SerialBackend())."))
end

function mpi_coarse_grain_batch!(args...; kwargs...)
    throw(ArgumentError("MPIBackend is unavailable — run `using MPI` (or use SerialBackend())."))
end

# Which backend each slice's own sweep runs under. Host-side parallel batching forces the inner sweep
# serial so the two levels never both claim the pool; a GPU request is different in kind — the device
# already parallelizes inside a slice, so it is passed INWARD and slices run one at a time.
@inline _batch_inner_backend(resolved::ComputationalBackends.AbstractExecutionBackend) =
    resolved isa ComputationalBackends.GPUBackend ? resolved : ComputationalBackends.SerialBackend()

# One method per backend, so adding a backend means adding a method rather than editing a branch. The
# generic method throws: a backend must never quietly become a serial loop here.
_batch_driver!(backend::ComputationalBackends.AbstractExecutionBackend, args...) = throw(ArgumentError(
    "coarse_grain_batch! has no batch driver for $(typeof(backend))",
))

# Host-serial slice loop.
function _batch_driver!(::ComputationalBackends.SerialBackend, batch, u, v, w, grid, valR, ctx)
    for t in eachindex(batch.slices)
        coarse_grain_batch_slice!(batch, u, v, w, grid, valR, ctx, t, 1)
    end
    return batch
end

# GPU: batch-native, and deliberately NOT the slice loop the host backends use. A device launch covering
# one slice of a small grid cannot fill the hardware, so each scale is computed for the whole batch in one
# pass — the filter engines fold the batch into a single launch, the tensor algebra broadcasts over the
# trailing axes, and the stencil derivatives carry them through.
#
# The host backends must keep the slice loop instead: slices are independent, so looping them threads at
# near-linear efficiency, whereas this path's parallelism is inside one sweep and is capped by the
# non-filtering fraction of it.
#
# `E(ℓ)` is the one reduction here — it collapses the spatial axes and keeps the batch axes, so it writes
# one energy per slice rather than broadcasting.
function _batch_driver!(
    be::ComputationalBackends.GPUBackend, batch, u, v, w, grid, ::Val{R}, ctx,
) where {R}
    T = eltype(batch.Π)
    ws = ctx.workspaces === nothing ?
        Diagnostics.ΠWorkspace(grid, batch.batch_size) : _pool_get(ctx.workspaces, 1)
    # This driver hands the WHOLE batch to `compute_Π!`, so the workspace must be sized for the batch. A
    # slice-sized one leaves the scratch at the grid's rank while the fields carry a batch axis, and the
    # mismatch surfaces deep inside as a backend hook that has no method for the wider array — an error
    # that names the backend rather than the shape. Checked here, where the shape is still meaningful.
    _batch_ws_shape(ws) == (FlowGeometries.Grids.size_tuple(grid)..., batch.batch_size...) ||
        throw(DimensionMismatch(
            "coarse_grain_batch! on a GPU backend needs a batch-sized workspace: got scratch of size " *
            "$(_batch_ws_shape(ws)), expected $((FlowGeometries.Grids.size_tuple(grid)..., batch.batch_size...)). " *
            "Build it with `ΠWorkspace(grid, batch_size)`.",
        ))
    dplan = _deriv_plan(grid, _pool_get(ctx.deriv_plans, 1))
    plans = ctx.filter_plans === nothing ?
        [_plan_filter(grid, ctx.kernel, T(s), ctx.mask_strategy, be, ctx.method) for s in ctx.scales] :
        _pool_get(ctx.filter_plans, 1)
    total_area = Diagnostics.active_area(grid)
    has_w = w !== nothing
    for s_idx in eachindex(ctx.scales)
        scale = T(ctx.scales[s_idx])
        selectdim(batch.scales, 1, s_idx) .= scale
        Diagnostics.compute_Π!(
            selectdim(batch.Π, R + 1, s_idx), u, v, w, grid, ctx.kernel, scale;
            workspace = ws, filter_plan = plans[s_idx], backend = be,
            mask_strategy = ctx.mask_strategy, deriv_plan = dplan,
        )
        Diagnostics.energy_from_filtered!(
            selectdim(batch.cumulative_energy, 1, s_idx), ws, grid, has_w, total_area,
        )
    end
    batch.wavenumber .= T(ctx.L) ./ batch.scales
    for J in CartesianIndices(batch.batch_size)
        _fill_spectrum!(
            view(batch.filtering_spectrum, :, Tuple(J)...),
            view(batch.cumulative_energy, :, Tuple(J)...),
            view(batch.wavenumber, :, Tuple(J)...),
            ctx.spectrum,
        )
    end
    return batch
end

_batch_driver!(::ComputationalBackends.ThreadedBackend, batch, u, v, w, grid, valR, ctx) =
    threaded_coarse_grain_batch!(batch, u, v, w, grid, valR, ctx)
_batch_driver!(::ComputationalBackends.DistributedBackend, batch, u, v, w, grid, valR, ctx) =
    distributed_coarse_grain_batch!(batch, u, v, w, grid, valR, ctx)
_batch_driver!(::ComputationalBackends.MPIBackend, batch, u, v, w, grid, valR, ctx) =
    mpi_coarse_grain_batch!(batch, u, v, w, grid, valR, ctx)

_slices_driver!(backend::ComputationalBackends.AbstractExecutionBackend, args...) = throw(ArgumentError(
    "coarse_grain_slices! has no batch driver for $(typeof(backend))",
))

function _slices_driver!(
    ::Union{ComputationalBackends.SerialBackend, ComputationalBackends.GPUBackend},
    results, us, vs, ws, grids, ctx,
)
    for t in eachindex(results)
        coarse_grain_slice_serial!(results, us, vs, ws, grids, ctx, t)
    end
    return results
end

_slices_driver!(::ComputationalBackends.ThreadedBackend, results, us, vs, ws, grids, ctx) =
    threaded_coarse_grain_slices!(results, us, vs, ws, grids, ctx)
_slices_driver!(::ComputationalBackends.DistributedBackend, results, us, vs, ws, grids, ctx) =
    distributed_coarse_grain_slices!(results, us, vs, ws, grids, ctx)
_slices_driver!(::ComputationalBackends.MPIBackend, results, us, vs, ws, grids, ctx) =
    mpi_coarse_grain_slices!(results, us, vs, ws, grids, ctx)

"""
    CoarseGrainBatchResult(grid, Nscales, batch_size)

Batched result storage plus the per-slice [`CoarseGrainResult`](@ref) views a batch sweep writes into.

The batch axes are **trailing**, which is what makes each slice's `Π` a contiguous
`(spatial..., Nscales)` view: fixing every trailing index of a column-major array selects a contiguous
slab. So `slices[i]` is a genuine `CoarseGrainResult` aliasing this storage, and `coarse_grain!` fills
it with no copy and no shape special-casing.

`batch_size` may hold any number of axes — `(Nt,)` for a time batch, `(Nlevels,)` for a vertical one,
`(Nlevels, Nt)` for both.
"""
struct CoarseGrainBatchResult{T<:AbstractFloat, NB, A<:AbstractArray{T}, S<:AbstractArray{T}, R<:AbstractVector}
    Π::A
    scales::S
    cumulative_energy::S
    wavenumber::S
    filtering_spectrum::S
    slices::R
    batch_size::NTuple{NB,Int}
end

"""
    batch_alloc_shared(T, dims...) -> AbstractArray{T}

Zero-filled shared-memory storage for a batch result, so worker processes write where the caller can
see them. Needs `using Distributed`; pass as `CoarseGrainBatchResult(...; alloc = batch_alloc_shared)`.
"""
function batch_alloc_shared(args...)
    throw(ArgumentError("Shared batch storage is unavailable — run `using Distributed`."))
end

function CoarseGrainBatchResult(
    grid::FlowGeometries.Grids.AbstractGrid{G,T}, Nscales::Integer, batch_size::NTuple{NB,Integer};
    alloc = zeros,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}, NB}
    spatial = FlowGeometries.Grids.size_tuple(grid)
    bs = Int.(batch_size)
    Π = alloc(T, spatial..., Nscales, bs...)
    scales = alloc(T, Nscales, bs...)
    cumE = alloc(T, Nscales, bs...)
    kℓ = alloc(T, Nscales, bs...)
    spec = alloc(T, Nscales, bs...)
    # `Π`'s slice keeps the spatial axes AND the scale axis; the per-scale vectors keep only the scale
    # axis. Both are built as views so a slice sweep writes straight into the batched storage.
    # `vec` because more than one batch axis makes the comprehension an N-D array; column-major order
    # is exactly the linear slice index the drivers use.
    slices = vec([
        CoarseGrainResult(
            view(scales, :, Tuple(J)...),
            view(Π, ntuple(_ -> Colon(), Val(length(spatial) + 1))..., Tuple(J)...),
            view(cumE, :, Tuple(J)...),
            view(kℓ, :, Tuple(J)...),
            view(spec, :, Tuple(J)...),
        ) for J in CartesianIndices(bs)
    ])
    return CoarseGrainBatchResult(Π, scales, cumE, kℓ, spec, slices, bs)
end

# Trailing batch indices fixed, every leading axis whole — contiguous by column-major layout, so the
# number of batch axes costs nothing in access pattern and the caller never reshapes.
@inline _slice_view(A::AbstractArray, ::Val{R}, J::CartesianIndex) where {R} =
    view(A, ntuple(_ -> Colon(), Val(R))..., Tuple(J)...)
@inline _slice_view(::Nothing, ::Val, ::CartesianIndex) = nothing

"""
    coarse_grain_batch!(batch, u, v, w, grid; scales, kernel, workspaces, filter_plans, deriv_plans, backend, mask_strategy, method, L, spectrum) -> batch
    coarse_grain_batch!(batch, u, v, grid; scales, ...)  # no vertical component

Run a full [`coarse_grain!`](@ref) sweep per slice over a batch that **shares one grid**.

The grid fixes the spatial rank; every trailing dimension of `u`, `v` and `w` beyond it is a batch
axis. So against a 2D grid, `(x, y, t)` is a time batch and `(x, y, z, t)` is a level *and* time
batch — `Nlevels·Nt` independent slices on one parallel axis, with no reshape. Against a true-3D grid
the same `(x, y, z)` array batches nothing, because the grid claims all three axes. Slices are
independent, so this is the outermost race-free axis and the one that turns thread count into
throughput: threading *inside* one sweep is bounded by the fraction of it that is filtering, while
this axis scales with the batch.

Each slice runs **serially inside**, as in [`Filtering.filter_slices!`](@ref): nesting a threaded
sweep under a threaded batch loop would have both levels claim the whole pool.

`workspaces`, `filter_plans` and `deriv_plans` are scratch **pooled per worker, not per slice** — pass
vectors of length ≥ the concurrency actually used (`Threads.nthreads()` is always enough) and a
repeated batch is allocation-free. Leaving them `nothing` builds one set per worker.

For a batch whose slices do NOT share a grid, use [`coarse_grain_slices!`](@ref) instead.

The `spectrum` policy behaves exactly as in [`coarse_grain`](@ref): the default `TopHatKernel` cannot
produce a guaranteed non-negative spectral density, so pass `kernel = GaussianKernel()`, or
`spectrum = Diagnostics.ForceSpectrum()` to compute it anyway, or `spectrum = Diagnostics.NoSpectrum()`.
"""
function coarse_grain_batch!(
    batch::CoarseGrainBatchResult{T,NB},
    u::AbstractArray,
    v::AbstractArray,
    w::Union{Nothing, AbstractArray},
    grid::FlowGeometries.Grids.AbstractGrid{G,T};
    scales::AbstractVector,
    kernel::Kernels.AbstractFilterKernel = Kernels.TopHatKernel(),
    workspaces::Union{Nothing, AbstractVector} = nothing,
    filter_plans::Union{Nothing, AbstractVector} = nothing,
    deriv_plans::Union{Nothing, AbstractVector} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Union{Nothing, Filtering.AbstractFilterMethod} = nothing,
    L::Real = one(T),
    spectrum::Diagnostics.AbstractSpectrumPolicy = Diagnostics.StrictSpectrum(),
) where {T<:AbstractFloat, NB, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    spatial = FlowGeometries.Grids.size_tuple(grid)
    valR = Val(length(spatial))
    _check_batch_shape(u, "u", spatial, batch.batch_size, valR)
    _check_batch_shape(v, "v", spatial, batch.batch_size, valR)
    w === nothing || _check_batch_shape(w, "w", spatial, batch.batch_size, valR)
    length(batch.slices) == prod(batch.batch_size) || throw(DimensionMismatch(
        "batch holds $(length(batch.slices)) slice views for batch size $(batch.batch_size)",
    ))
    _check_pools(workspaces, filter_plans, deriv_plans)
    _check_spectrum(kernel, spectrum)
    resolved = Filtering._resolve_slice_backend(backend)
    inner = _batch_inner_backend(resolved)
    ctx = (; scales, kernel, workspaces, filter_plans, deriv_plans, mask_strategy, method, L, spectrum, inner)
    return _batch_driver!(resolved, batch, u, v, w, grid, valR, ctx)
end

function coarse_grain_batch!(
    batch::CoarseGrainBatchResult, u::AbstractArray, v::AbstractArray,
    grid::FlowGeometries.Grids.AbstractGrid; kwargs...,
)
    return coarse_grain_batch!(batch, u, v, nothing, grid; kwargs...)
end

# `Val`-typed rank so the compared tuples are built by `ntuple` at a known length. Slicing `size(A)`
# with a runtime range instead cannot infer a fixed-size tuple and allocates on every call.
function _check_batch_shape(
    A::AbstractArray, name::AbstractString, spatial::NTuple{R,Int}, bs::NTuple{NB,Int}, ::Val{R},
) where {R, NB}
    ndims(A) == R + NB || throw(DimensionMismatch(
        "$name has $(ndims(A)) dimensions; expected $R spatial + $NB batch",
    ))
    ntuple(i -> size(A, i), Val(R)) == spatial || throw(DimensionMismatch(
        "$name's leading dimensions $(ntuple(i -> size(A, i), Val(R))) do not match grid shape $spatial",
    ))
    ntuple(i -> size(A, R + i), Val(NB)) == bs || throw(DimensionMismatch(
        "$name's batch dimensions $(ntuple(i -> size(A, R + i), Val(NB))) do not match batch size $bs",
    ))
    return nothing
end

"""
    coarse_grain_batch_slice!(batch, u, v, w, grid, Val(R), ctx, t, p) -> batch.slices[t]

Slice `t`'s sweep, drawing scratch from pool entry `p` and forcing the inner backend serial. Shared by
the serial loop and every parallel driver so all of them run identical code.
"""
function coarse_grain_batch_slice!(
    batch::CoarseGrainBatchResult, u, v, w, grid, ::Val{R}, ctx, t::Integer, p::Integer,
) where {R}
    J = CartesianIndices(batch.batch_size)[t]
    return coarse_grain!(
        batch.slices[t],
        _slice_view(u, Val(R), J), _slice_view(v, Val(R), J), _slice_view(w, Val(R), J),
        grid;
        scales = ctx.scales,
        kernel = ctx.kernel,
        workspace = _pool_get(ctx.workspaces, p),
        filter_plans = _pool_get(ctx.filter_plans, p),
        deriv_plan = _pool_get(ctx.deriv_plans, p),
        backend = ctx.inner,
        mask_strategy = ctx.mask_strategy,
        method = ctx.method,
        L = ctx.L,
        spectrum = ctx.spectrum,
    )
end

@inline _batch_ws_shape(ws::Diagnostics.ΠWorkspace) = size(ws.u_filt)

@inline _pool_get(::Nothing, ::Integer) = nothing
@inline _pool_get(pool::AbstractVector, p::Integer) = pool[p]

# Length of the caller's scratch pools, or `nothing` when none were supplied.
function _pool_length(ctx)
    ctx.workspaces !== nothing && return length(ctx.workspaces)
    ctx.filter_plans !== nothing && return length(ctx.filter_plans)
    ctx.deriv_plans !== nothing && return length(ctx.deriv_plans)
    return nothing
end

"""
    batch_concurrency(ctx, nslices) -> Int

How many slices a batch may run at once. A worker indexes its scratch by chunk, so supplying shorter
pools than `nthreads()` deliberately caps concurrency rather than aliasing scratch between workers.
"""
function batch_concurrency(ctx, nslices::Integer)
    np = _pool_length(ctx)
    return min(nslices, np === nothing ? Threads.nthreads() : np)
end

# Pools are indexed by worker, so they must agree in length or a chunk index valid for one is out of
# bounds for another. Written over plain integers — looping a heterogeneous tuple of (name, pool)
# pairs is type-unstable and allocates on every call.
@inline _pool_n(::Nothing) = -1
@inline _pool_n(pool::AbstractVector) = length(pool)

function _check_pools(workspaces, filter_plans, deriv_plans)
    a, b, c = _pool_n(workspaces), _pool_n(filter_plans), _pool_n(deriv_plans)
    (a == 0 || b == 0 || c == 0) &&
        throw(ArgumentError("coarse_grain_batch! got an empty scratch pool"))
    ref = max(a, b, c)
    ref < 0 && return nothing
    ((a < 0 || a == ref) && (b < 0 || b == ref) && (c < 0 || c == ref)) || throw(DimensionMismatch(
        "coarse_grain_batch! scratch pools must agree in length; got workspaces=$a, " *
        "filter_plans=$b, deriv_plans=$c (-1 means not supplied)",
    ))
    return nothing
end

# ---------------------------------------------------------------------------
# Batch axis: independent sweeps whose slices do NOT share a grid
# ---------------------------------------------------------------------------

function threaded_coarse_grain_slices!(args...; kwargs...)
    throw(ArgumentError("ThreadedBackend is unavailable — run `using OhMyThreads` (or use SerialBackend())."))
end

function distributed_coarse_grain_slices!(args...; kwargs...)
    throw(ArgumentError("DistributedBackend is unavailable — run `using Distributed` (or use SerialBackend())."))
end

function mpi_coarse_grain_slices!(args...; kwargs...)
    throw(ArgumentError("MPIBackend is unavailable — run `using MPI` (or use SerialBackend())."))
end

"""
    slice_pipeline_costs(grids, Nscales) -> Vector{Int}

Relative per-slice sweep cost, for longest-first scheduling of a ragged batch. Sweep cost grows at
least linearly in a slice's point count and every scale repeats the sweep, so points × scales orders
the slices correctly even though it is not an absolute time.
"""
slice_pipeline_costs(grids::AbstractVector, Nscales::Integer) =
    [prod(FlowGeometries.Grids.size_tuple(g)) * Nscales for g in grids]

"""
    coarse_grain_slices!(results, us, vs, ws, grids; scales, kernel, workspaces, filter_plans, deriv_plans, backend, mask_strategy, method, L, spectrum) -> results
    coarse_grain_slices!(results, us, vs, grids; scales, ...)  # no vertical component

Run a full [`coarse_grain!`](@ref) sweep per slice over a batch whose slices each carry **their own
grid** — different regions, different point counts, a separate node set per slice.

Use [`coarse_grain_batch!`](@ref) when every slice shares one grid: it stores the batch on trailing
array axes and slices it with contiguous views, which this form cannot do because the spatial shapes
differ. `results` is correspondingly a vector of independent [`CoarseGrainResult`](@ref)s rather than
views into shared storage.

Slices are dispatched **longest-first** under a dynamic scheduler (see
[`slice_pipeline_costs`](@ref)). Unlike `coarse_grain_batch!`, where one shared grid makes every slice
cost the same, ragged counts mean equal chunking would leave one worker holding the largest slice
after the others have drained.

Scratch is **per slice** here, not per worker: a workspace and a filter plan are grid-shaped, so they
cannot be reused across slices of different shape. Where many slices do share a shape, group them and
call this once per group to keep pool memory bounded by thread count instead of batch length.

Each slice runs serially inside, the same non-nesting rule as [`coarse_grain_batch!`](@ref).

The `spectrum` policy behaves exactly as in [`coarse_grain`](@ref): the default `TopHatKernel` cannot
produce a guaranteed non-negative spectral density, so pass `kernel = GaussianKernel()`, or
`spectrum = Diagnostics.ForceSpectrum()` to compute it anyway, or `spectrum = Diagnostics.NoSpectrum()`.
"""
function coarse_grain_slices!(
    results::AbstractVector,
    us::AbstractVector,
    vs::AbstractVector,
    ws::Union{Nothing, AbstractVector},
    grids::AbstractVector;
    scales::AbstractVector,
    kernel::Kernels.AbstractFilterKernel = Kernels.TopHatKernel(),
    workspaces::Union{Nothing, AbstractVector} = nothing,
    filter_plans::Union{Nothing, AbstractVector} = nothing,
    deriv_plans::Union{Nothing, AbstractVector} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Union{Nothing, Filtering.AbstractFilterMethod} = nothing,
    L::Real = 1,
    spectrum::Diagnostics.AbstractSpectrumPolicy = Diagnostics.StrictSpectrum(),
)
    n = length(results)
    (length(us) == n && length(vs) == n && length(grids) == n) || throw(DimensionMismatch(
        "coarse_grain_slices! got $n results, $(length(us)) u, $(length(vs)) v and $(length(grids)) grids",
    ))
    ws === nothing || length(ws) == n || throw(DimensionMismatch(
        "coarse_grain_slices! got $n results and $(length(ws)) w slices",
    ))
    workspaces === nothing || length(workspaces) == n || throw(DimensionMismatch(
        "coarse_grain_slices! got $n results and $(length(workspaces)) workspaces — ragged scratch is per slice",
    ))
    filter_plans === nothing || length(filter_plans) == n || throw(DimensionMismatch(
        "coarse_grain_slices! got $n results and $(length(filter_plans)) filter_plans entries — ragged scratch is per slice",
    ))
    deriv_plans === nothing || length(deriv_plans) == n || throw(DimensionMismatch(
        "coarse_grain_slices! got $n results and $(length(deriv_plans)) deriv_plans entries — ragged scratch is per slice",
    ))
    _check_spectrum(kernel, spectrum)
    resolved = Filtering._resolve_slice_backend(backend)
    inner = _batch_inner_backend(resolved)
    ctx = (; scales, kernel, workspaces, filter_plans, deriv_plans, mask_strategy, method, L, spectrum, inner)
    return _slices_driver!(resolved, results, us, vs, ws, grids, ctx)
end

function coarse_grain_slices!(
    results::AbstractVector, us::AbstractVector, vs::AbstractVector, grids::AbstractVector; kwargs...,
)
    return coarse_grain_slices!(results, us, vs, nothing, grids; kwargs...)
end

"""
    coarse_grain_slice_serial!(results, us, vs, ws, grids, ctx, t) -> results[t]

Slice `t`'s sweep with the inner backend forced serial. Shared by the serial loop and every parallel
driver so all of them run identical code.
"""
function coarse_grain_slice_serial!(results, us, vs, ws, grids, ctx, t::Integer)
    return coarse_grain!(
        results[t], us[t], vs[t], ws === nothing ? nothing : ws[t], grids[t];
        scales = ctx.scales,
        kernel = ctx.kernel,
        workspace = _pool_get(ctx.workspaces, t),
        filter_plans = _pool_get(ctx.filter_plans, t),
        deriv_plan = _pool_get(ctx.deriv_plans, t),
        backend = ctx.inner,
        mask_strategy = ctx.mask_strategy,
        method = ctx.method,
        L = ctx.L,
        spectrum = ctx.spectrum,
    )
end

# 2.5D Cartesian constructor wrapper (2D velocity fields without a vertical component)
function coarse_grain(
    u::AbstractMatrix,
    v::AbstractMatrix,
    grid::FlowGeometries.Grids.StructuredGrid{G,T};
    scales::AbstractVector,
    kernel::Kernels.AbstractFilterKernel = Kernels.TopHatKernel(),
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Union{Nothing, Filtering.AbstractFilterMethod} = nothing,
    L::Real = one(T),
    spectrum::Diagnostics.AbstractSpectrumPolicy = Diagnostics.StrictSpectrum(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    return coarse_grain(u, v, nothing, grid; scales=scales, kernel=kernel, backend=backend, mask_strategy=mask_strategy, method=method, L=L, spectrum=spectrum)
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
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Union{Nothing, Filtering.AbstractFilterMethod} = nothing,
    L::Real = one(T),
    spectrum::Diagnostics.AbstractSpectrumPolicy = Diagnostics.StrictSpectrum(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    return coarse_grain!(result, u, v, nothing, grid; scales=scales, kernel=kernel, workspace=workspace, filter_plans=filter_plans, deriv_plan=deriv_plan, backend=backend, mask_strategy=mask_strategy, method=method, L=L, spectrum=spectrum)
end

"""
    coarse_grain_profile!(batch, u, v, w, grid; scales, kernel, workspaces, filter_plans, deriv_plans, backend, mask_strategy, method, L, spectrum) -> batch
    coarse_grain_profile!(batch, u, v, grid; scales, ...)  # no vertical component

Vertical-profile sweep, in place: the literature-standard independent-per-level 2D/2.5D method (see
[`Diagnostics.compute_Π!`](@ref)'s thin-layer/QG regime note), filtering horizontally and treating each
vertical level as its own 2D problem on the shared horizontal grid.

That is exactly a shared-grid batch whose axis happens to be the vertical one, so this delegates to
[`coarse_grain_batch!`](@ref) rather than reimplementing the level loop. Two consequences worth knowing:
the level axis is **parallel** on every backend the batch entry point supports, and `E(ℓ)` is read out of
the workspace in the same pass as the flux, so `u` and `v` are not filtered a second time per level.

`batch` is a [`CoarseGrainBatchResult`](@ref) with batch size `(Nlevels,)`, so `Π` is
`(Nx, Ny, Nscales, Nlevels)` and `cumulative_energy`/`filtering_spectrum` are `(Nscales, Nlevels)` — the
level axis trailing, which is what makes each level's result a contiguous view. Per-level energies are
deliberately not summed across levels; that needs thickness weighting this function is not given.

The `spectrum` policy behaves exactly as in [`coarse_grain`](@ref): the default `TopHatKernel` cannot
produce a guaranteed non-negative spectral density, so pass `kernel = GaussianKernel()`, or
`spectrum = Diagnostics.ForceSpectrum()` to compute it anyway, or `spectrum = Diagnostics.NoSpectrum()`.
"""
function coarse_grain_profile!(
    batch::CoarseGrainBatchResult,
    u::AbstractArray{T,3},
    v::AbstractArray{T,3},
    w::Union{Nothing, AbstractArray{T,3}},
    grid::FlowGeometries.Grids.StructuredGrid{G,T,2};
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    return coarse_grain_batch!(batch, u, v, w, grid; kwargs...)
end

function coarse_grain_profile!(
    batch::CoarseGrainBatchResult, u::AbstractArray{T,3}, v::AbstractArray{T,3},
    grid::FlowGeometries.Grids.StructuredGrid{G,T,2}; kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    return coarse_grain_batch!(batch, u, v, nothing, grid; kwargs...)
end

"""
    coarse_grain_profile(u, v, w, grid; scales, kernel=TopHatKernel(), backend=AutoBackend(), mask_strategy=ZeroFill(), method=nothing, L=1, spectrum=StrictSpectrum())

Allocating [`coarse_grain_profile!`](@ref): sizes a [`CoarseGrainBatchResult`](@ref) for `size(u, 3)`
levels and fills it. Pass a prebuilt `batch` to `coarse_grain_profile!` to sweep timesteps without
reallocating the result, which for a `(Nx, Ny, Nlevels, Nscales)` flux array is the dominant allocation.
"""
function coarse_grain_profile(
    u::AbstractArray{T,3},
    v::AbstractArray{T,3},
    w::Union{Nothing, AbstractArray{T,3}},
    grid::FlowGeometries.Grids.StructuredGrid{G,T,2};
    scales::AbstractVector,
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    batch = CoarseGrainBatchResult(grid, length(scales), (size(u, 3),))
    return coarse_grain_profile!(batch, u, v, w, grid; scales = scales, kwargs...)
end

# 2.5D convenience wrapper (no vertical velocity).
function coarse_grain_profile(
    u::AbstractArray{T,3}, v::AbstractArray{T,3},
    grid::FlowGeometries.Grids.StructuredGrid{G,T,2}; kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    return coarse_grain_profile(u, v, nothing, grid; kwargs...)
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
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Union{Nothing, Filtering.AbstractFilterMethod} = nothing,
    L::Real = one(T),
    spectrum::Diagnostics.AbstractSpectrumPolicy = Diagnostics.StrictSpectrum(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    result = _allocate_result(grid, length(scales))
    workspace = Diagnostics.ΠWorkspace(grid)
    return coarse_grain!(
        result, u, grid;
        scales = scales, kernel = kernel, workspace = workspace, filter_plans = filter_plans,
        backend = backend, mask_strategy = mask_strategy, method = method, L = L, spectrum = spectrum,
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
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Union{Nothing, Filtering.AbstractFilterMethod} = nothing,
    L::Real = one(T),
    spectrum::Diagnostics.AbstractSpectrumPolicy = Diagnostics.StrictSpectrum(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    _check_result_shape(result, grid, scales)
    _check_spectrum(kernel, spectrum)
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
    _fill_spectrum!(result.filtering_spectrum, result.cumulative_energy, result.wavenumber, spectrum)
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
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Union{Nothing, Filtering.AbstractFilterMethod} = nothing,
    L::Real = one(T),
    spectrum::Diagnostics.AbstractSpectrumPolicy = Diagnostics.StrictSpectrum(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    result = _allocate_result(grid, length(scales))
    workspace = Diagnostics.ΠWorkspace(grid)
    return coarse_grain!(
        result, u, v, w, grid;
        scales = scales, kernel = kernel, workspace = workspace,
        backend = backend, mask_strategy = mask_strategy, method = method, L = L, spectrum = spectrum,
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
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Union{Nothing, Filtering.AbstractFilterMethod} = nothing,
    L::Real = one(T),
    spectrum::Diagnostics.AbstractSpectrumPolicy = Diagnostics.StrictSpectrum(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    result = _allocate_result(grid, length(scales))
    workspace = Diagnostics.ΠWorkspace(grid)
    deriv_plan = FlowGeometries.Connectivity.gradient_plan(grid)
    return coarse_grain!(
        result, u, v, w, grid;
        scales = scales, kernel = kernel, workspace = workspace, deriv_plan = deriv_plan,
        backend = backend, mask_strategy = mask_strategy, method = method, L = L, spectrum = spectrum,
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
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Union{Nothing, Filtering.AbstractFilterMethod} = nothing,
    L::Real = one(T),
    spectrum::Diagnostics.AbstractSpectrumPolicy = Diagnostics.StrictSpectrum(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    return coarse_grain(u, v, nothing, grid; scales=scales, kernel=kernel, backend=backend, mask_strategy=mask_strategy, method=method, L=L, spectrum=spectrum)
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
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Union{Nothing, Filtering.AbstractFilterMethod} = nothing,
    L::Real = one(T),
    spectrum::Diagnostics.AbstractSpectrumPolicy = Diagnostics.StrictSpectrum(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    return coarse_grain!(result, u, v, nothing, grid; scales=scales, kernel=kernel, workspace=workspace, deriv_plan=deriv_plan, filter_plans=filter_plans, backend=backend, mask_strategy=mask_strategy, method=method, L=L, spectrum=spectrum)
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
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Filtering.AbstractFilterMethod = Filtering.Spectral(),
    L::Real = one(T),
    spectrum::Diagnostics.AbstractSpectrumPolicy = Diagnostics.StrictSpectrum(),
) where {T<:AbstractFloat}
    result = _allocate_result(grid, length(scales))
    workspace = Diagnostics.ΠWorkspace(grid)
    deriv_plan = FlowGeometries.Connectivity.gradient_plan(grid)
    return coarse_grain!(
        result, u, v, w, grid;
        scales = scales, kernel = kernel, workspace = workspace, deriv_plan = deriv_plan,
        backend = backend, mask_strategy = mask_strategy, method = method, L = L, spectrum = spectrum,
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
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Filtering.AbstractFilterMethod = Filtering.Spectral(),
    L::Real = one(T),
    spectrum::Diagnostics.AbstractSpectrumPolicy = Diagnostics.StrictSpectrum(),
) where {T<:AbstractFloat}
    return coarse_grain(u, v, nothing, grid; scales=scales, kernel=kernel, backend=backend, mask_strategy=mask_strategy, method=method, L=L, spectrum=spectrum)
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
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Filtering.AbstractFilterMethod = Filtering.Spectral(),
    L::Real = one(T),
    spectrum::Diagnostics.AbstractSpectrumPolicy = Diagnostics.StrictSpectrum(),
) where {T<:AbstractFloat}
    return coarse_grain!(result, u, v, nothing, grid; scales=scales, kernel=kernel, workspace=workspace, deriv_plan=deriv_plan, filter_plans=filter_plans, backend=backend, mask_strategy=mask_strategy, method=method, L=L, spectrum=spectrum)
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
