module Pipeline

using FlowGeometries: FlowGeometries
using ..Kernels: Kernels
using ..Filtering: Filtering
using ..Derivatives: Derivatives
using ..Diagnostics: Diagnostics
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends

export CoarseGrainResult, coarse_grain, coarse_grain!
export CoarseGrainBatchResult, coarse_grain_batch!, coarse_grain_slices!
export coarse_grain_profile, coarse_grain_profile!
export check_setup, SetupReport

# ---------------------------------------------------------------------------
# check_setup — what will actually run, and what this (grid, kernel, ℓ) can support
# ---------------------------------------------------------------------------

"""
    SetupReport

What [`check_setup`](@ref) found; `show` prints it as a report.

# Fields
- `grid_kind::String`, `grid_size::Tuple`: the grid
- `spacing::Tuple`, `coarsest_spacing::Tuple`: the smallest and largest physical distance between
  index-adjacent cells, per axis, in the units of `ℓ`
- `scale::Float64`, `cells_per_scale::Tuple`: `ℓ`, and `ℓ / coarsest_spacing` per axis — the fewest
  cells `ℓ` spans anywhere on the grid
- `kernel::String`, `mask_strategy::String`, `method::String`
- `execution_requested::String`, `execution_resolved::String`: the execution backend asked for, and
  the one that will run
- `engine::String`: which real-space engine class the filter will take
- `resolvable::Bool`: `ℓ` spans more than 2 cells on every axis everywhere on the grid
- `supports_flux::Bool`: the kernel is non-negative, so `τ` is realizable and `Π` is a pointwise
  transfer
- `supports_spectrum::Bool`: [`Kernels.transfer_monotone`](@ref): the spectral density is guaranteed
  non-negative, and the default [`Diagnostics.AutoSpectrum`](@ref) computes it (`NaN` otherwise).
  `Diagnostics.ForceSpectrum()` computes it regardless.
- `supports_spectral_method::Bool`: the kernel has an isotropic transfer function, so
  `method = Spectral()` is available
- `boundary_buffer_cells::Tuple`: the kernel radius over the smallest spacing — how many cells from a
  coast or domain edge the result depends on the mask strategy
- `notes::Vector{String}`: everything above that needs acting on, in words
"""
struct SetupReport{GS<:Tuple, SP<:Tuple, CS<:Tuple, CP<:Tuple, BB<:Tuple, VS <: AbstractVector{<:AbstractString}, FT}
    grid_kind::String
    grid_size::GS
    spacing::SP
    coarsest_spacing::CS
    scale::FT
    cells_per_scale::CP
    kernel::String
    mask_strategy::String
    method::String
    execution_requested::String
    execution_resolved::String
    engine::String
    resolvable::Bool
    supports_flux::Bool
    supports_spectrum::Bool
    supports_spectral_method::Bool
    boundary_buffer_cells::BB
    notes::VS
end

function Base.show(io::IO, ::MIME"text/plain", r::SetupReport)
    yn(b) = b ? "yes" : "NO"
    println(io, "CoarseGrainingEnergyFluxes setup check")
    println(io, "  grid            : ", r.grid_kind, " ", r.grid_size,
                "   spacing ", map(x -> round(x; sigdigits = 4), r.spacing), " to ",
                map(x -> round(x; sigdigits = 4), r.coarsest_spacing))
    println(io, "  scale ℓ         : ", r.scale, "  = ",
                map(x -> round(x; digits = 2), r.cells_per_scale), " cells per axis")
    println(io, "  kernel          : ", r.kernel)
    println(io, "  masking         : ", r.mask_strategy)
    println(io, "  method          : ", r.method)
    println(io, "  execution       : ", r.execution_requested, " -> ", r.execution_resolved)
    println(io, "  real-space engine: ", r.engine)
    println(io, "  ℓ resolvable    : ", yn(r.resolvable))
    println(io, "  supports Π      : ", yn(r.supports_flux))
    println(io, "  spectrum ≥ 0    : ", yn(r.supports_spectrum))
    println(io, "  supports Spectral(): ", yn(r.supports_spectral_method))
    println(io, "  boundary buffer : ", r.boundary_buffer_cells, " cells")
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

_kname(k::Kernels.AbstractFilterKernel) = string(nameof(typeof(k)))
_kname(k::Kernels.GaussianKernel) = "GaussianKernel(α = $(k.α))"
_kname(k::Kernels.HyperGaussianKernel) = "HyperGaussianKernel(α = $(k.α))"
_kname(k::Kernels.SmoothHatKernel) = "SmoothHatKernel(steepness = $(k.steepness))"
_kname(k::Kernels.HighOrderKernel{P}) where {P} = "HighOrderKernel{$P}(b_over_ℓ = $(k.b_over_ℓ))"

_grid_kind(::FlowGeometries.Grids.StructuredGrid{T,G,N}) where {G,T,N} = "StructuredGrid{$(nameof(G)),$N}"
_grid_kind(::FlowGeometries.Grids.CurvilinearGrid{T,G}) where {T,G} = "CurvilinearGrid{$(nameof(G))}"
_grid_kind(::FlowGeometries.Grids.UnstructuredGrid{T,G}) where {T,G} = "UnstructuredGrid{$(nameof(G))}"
_grid_kind(g) = string(nameof(typeof(g)))

_grid_float(::FlowGeometries.Grids.AbstractGrid{G,T}) where {G,T} = T

# One value of the transfer function a spectral apply on this geometry evaluates: per spherical-harmonic
# degree on a sphere's surface, per wavevector elsewhere.
_transfer_probe(geo::FlowGeometries.Geometry.AbstractSphericalGeometry, kernel, ℓ::T, ::Val{2}) where {T} =
    Kernels.spectral_transfer_degree(kernel, 1, ℓ, T(FlowGeometries.Geometry.radius(geo)))
_transfer_probe(_, kernel, ℓ::T, ::Val{D}) where {T,D} =
    Kernels.spectral_transfer(kernel, ntuple(_ -> one(T) / ℓ, Val(D)), ℓ)

# `(smallest, largest)` metric distance between index-adjacent cells per axis, in the units of `ℓ`
# (metres on a sphere). A node set has no axes and returns `((), ())`.
_axis_spacings(grid::Union{FlowGeometries.Grids.StructuredGrid, FlowGeometries.Grids.CurvilinearGrid}) =
    _index_spacings(grid, FlowGeometries.Grids.size_tuple(grid))
_axis_spacings(::FlowGeometries.Grids.AbstractGrid) = ((), ())

function _index_spacings(grid::FlowGeometries.Grids.AbstractGrid{G,T}, sz::NTuple{N,Int}) where {G,T,N}
    geo = FlowGeometries.Grids.grid_geometry(grid)
    fine = fill(T(Inf), N)
    coarse = fill(zero(T), N)
    for I in CartesianIndices(sz), d in 1:N
        I[d] < sz[d] || continue
        J = I + CartesianIndex(ntuple(e -> e == d ? 1 : 0, Val(N)))
        s = T(FlowGeometries.Geometry.distance(
            geo, FlowGeometries.Grids.coords(grid, Tuple(I)...), FlowGeometries.Grids.coords(grid, Tuple(J)...),
        ))
        fine[d] = min(fine[d], s)
        coarse[d] = max(coarse[d], s)
    end
    return ntuple(d -> fine[d], Val(N)), ntuple(d -> coarse[d], Val(N))
end

_engine_description(::Filtering.BinnedTopHatEngine) = "binned prefix-sum top-hat, exact, O(N·r/h)"
_engine_description(::Filtering.NUFFTSumEngine) = "nonuniform-FFT kernel sum, O(N log N)"
_engine_description(::Filtering.PaddedFFTEngine) = "padded-FFT of the sampled kernel, O(N log N)"
_engine_description(::Filtering.ZonalFFTEngine) = "zonal-FFT along longitude, O(N·(log Nλ + w_φ))"

# The 2-D prefix sums take the centre-test top-hat at any spacing but on a spheroid, whose geodesic disk has no
# closed band along a row, and the cell overlap where axis 1 is uniform.
_prefix_sums_take(::Kernels.TopHatKernel{Kernels.CellCentre}, grid) =
    !(FlowGeometries.Grids.grid_geometry(grid) isa FlowGeometries.Geometry.AbstractEllipsoidalGeometry)
_prefix_sums_take(::Kernels.TopHatKernel{Kernels.CellOverlap}, grid) = FlowGeometries.Grids.isuniform(grid, 1)

# The real-space engine the filter will take on the resolved backend and its cost: `Filtering._select_engine`'s
# choice, and for the footprint the predicates `Filtering.build_footprint` dispatches on. `w` is the filter
# half-width in cells on the named axis. A backend that did not resolve (`nothing`) takes the footprint.
function _engine_class(grid, kernel, scale, resolved = ComputationalBackends.SerialBackend())
    is_struct = grid isa FlowGeometries.Grids.StructuredGrid
    is_cart = FlowGeometries.Grids.grid_geometry(grid) isa FlowGeometries.Geometry.CartesianGeometry
    rank = is_struct ? length(FlowGeometries.Grids.size_tuple(grid)) : 0
    sep_measure = is_struct && FlowGeometries.Grids.measure_factors(grid) !== nothing
    ranges = is_struct && FlowGeometries.Grids.isuniform(grid)

    engine = resolved === nothing ? Filtering.FootprintEngine() : Filtering._select_engine(grid, kernel, scale, resolved)
    engine isa Filtering.FootprintEngine || return _engine_description(engine)

    if Filtering._is_flat_cell(grid)
        return Filtering._binned_list(grid) ? "node CSR gather, its list read from the point bins, O(N·⟨neighbours⟩)" :
                                              "node CSR gather over the grid's ball query, O(N·⟨neighbours⟩)"
    end

    # The separable engines are Cartesian-only: great-circle distance does not factor.
    if Kernels.is_separable(kernel) && is_struct && is_cart && sep_measure
        return rank == 2 ? "separable two-pass, O(N·(wx+wy))" : "separable N-pass, O(N·Σ wd)"
    end

    if kernel isa Kernels.TopHatKernel && is_struct && rank == 2 && sep_measure
        _prefix_sums_take(kernel, grid) && return "prefix-sum top-hat, exact, O(N·w_y)"
        return "scattered per-point footprint, O(N·wx·wy) with an optional neighbour cache"
    end
    # 3-D prefix sums need uniform Cartesian axes, where the axis-1 half-width depends on the offsets
    # `(dj, dk)` alone.
    if kernel isa Kernels.TopHatKernel && is_struct && rank == 3 && is_cart && ranges
        return "prefix-sum top-hat, exact, O(N·w_y·w_z)"
    end

    # A `HighOrderKernel` on a sphere is corrected once per latitude row where a row's targets share one
    # correction, once per point otherwise; on a regional box that depends on the scale.
    if kernel isa Kernels.HighOrderKernel && is_struct && rank == 2 && !is_cart
        shares = Filtering._highorder_shares_rows(grid, kernel, convert(_grid_float(grid), scale))
        return shares ? "banded footprint, one band and one correction per latitude, O(N·wλ·wφ)" :
                        "scattered per-point footprint, one correction per point, O(N·wx·wy) with an optional neighbour cache"
    end

    # The banded engine needs a uniform axis 1 alone; the other axis decides one band or one per row.
    if is_struct && rank == 2
        FlowGeometries.Grids.isuniform(grid, 1) ||
            return "scattered per-point footprint, O(N·wx·wy) with an optional neighbour cache"
        is_cart || return "banded footprint, one band per latitude, O(N·wλ·wφ)"
        return ranges ? "banded footprint, one band, O(N·wx·wy)" : "banded footprint, one band per row, O(N·wx·wy)"
    end
    if grid isa FlowGeometries.Grids.CurvilinearGrid
        return "scattered per-point footprint, O(N·wi·wj) with an optional neighbour cache"
    end
    if is_struct && (rank == 1 || rank == 3)
        return ranges && is_cart ?
            "N-dimensional ball footprint, O(N·∏ wd)" :
            "scattered N-dimensional footprint, O(N·∏ wd) with an optional neighbour cache"
    end
    return "scattered per-point footprint, O(N·∏ wd) with an optional neighbour cache"
end

"""
    check_setup(grid, kernel, scale; execution, mask_strategy, method) -> SetupReport

Report what a filter/flux call on this `(grid, kernel, scale)` will do and what it supports, without
building a plan: it evaluates the dispatch predicates the plan builders use. It reports

- which real-space engine will run, and which execution backend the request resolves to;
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
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Union{Nothing, Filtering.AbstractFilterMethod} = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    ℓ = T(scale)
    ℓ > 0 || throw(ArgumentError("check_setup needs a positive scale, got $scale"))
    notes = String[]
    sp, sp_coarse = _axis_spacings(grid)
    cps = map(s -> (isfinite(s) && s > 0) ? ℓ / s : T(NaN), sp_coarse)
    resolvable = isempty(cps) ? true : all(c -> !isnan(c) && c > 2, cps)
    resolvable || push!(notes,
        "where the grid is coarsest, ℓ spans $(map(x -> round(x; digits = 2), cps)) cells; below ~2 " *
        "cells on an axis the filter is a no-op on that axis there. Increase ℓ or coarsen the grid.")

    dim = Filtering._kernel_dim(grid)
    rad = Kernels.kernel_radius(kernel, ℓ, dim)
    buf = map(s -> (isfinite(s) && s > 0) ? ceil(Int, rad / s) : 0, sp)
    isempty(buf) || push!(notes,
        "within $(buf) cells of a coast or domain edge the result depends on the mask strategy: " *
        "ZeroFill treats land and the domain exterior as fluid at rest, Deformable renormalizes the " *
        "kernel over active cells.")

    resolved_execution = try
        Filtering._resolve_backend(execution, grid)
    catch e
        push!(notes, "execution backend $(nameof(typeof(execution))) is not available for this grid: " *
                     first(sprint(showerror, e), 200))
        nothing
    end
    resolved = resolved_execution === nothing ? "UNAVAILABLE" : string(nameof(typeof(resolved_execution)))

    # A weight in the grid's own dimension: `SharpSpectralKernel`'s 2-D weight needs a weak dependency, and
    # the `ArgumentError` it throws without one becomes a note.
    try
        Kernels.is_radial(kernel) && Kernels.kernel_weight(Kernels.point_kernel(kernel), zero(T), ℓ, dim)
    catch e
        e isa ArgumentError || rethrow()
        push!(notes, sprint(showerror, e))
    end
    flux_ok = Kernels.is_nonnegative(kernel)
    flux_ok || push!(notes,
        "$(_kname(kernel)) takes NEGATIVE values, so τ is not realizable and Π is not a pointwise " *
        "energy transfer. Use TopHatKernel or GaussianKernel for a flux.")

    spec_ok = Kernels.transfer_monotone(kernel)
    spec_ok || push!(notes,
        "$(_kname(kernel))'s |Ĝ|² is not monotone, so a spectral density is not guaranteed " *
        "non-negative and `coarse_grain` fills `filtering_spectrum` with NaN: pass " *
        "`kernel = GaussianKernel()`, or `spectrum = Diagnostics.ForceSpectrum()` to compute it anyway.")

    # The transfer function the grid's spectral path evaluates; its `ArgumentError` names a missing weak
    # dependency or a kernel with no transfer function.
    spectral_ok, spectral_note = try
        _transfer_probe(FlowGeometries.Grids.grid_geometry(grid), kernel, ℓ, dim)
        true, nothing
    catch e
        e isa ArgumentError || rethrow()
        false, "`method = Spectral()` is unavailable: " * sprint(showerror, e)
    end
    spectral_note === nothing || push!(notes, spectral_note)

    if isempty(sp)
        push!(notes,
            "this grid has no axes, so ℓ-resolvability and the boundary buffer could not be checked " *
            "here; compare ℓ against the typical nearest-neighbour distance yourself " *
            "(the kernel radius is $(round(rad; sigdigits = 4))).")
    end

    return SetupReport(
        _grid_kind(grid), FlowGeometries.Grids.size_tuple(grid), sp, sp_coarse, Float64(ℓ), cps,
        _kname(kernel), string(nameof(typeof(mask_strategy))),
        method === nothing ? "default for this grid" : string(nameof(typeof(method))),
        string(nameof(typeof(execution))), resolved, _engine_class(grid, kernel, ℓ, resolved_execution),
        resolvable, flux_ok, spec_ok, spectral_ok, buf, notes,
    )
end

"""
    CoarseGrainResult(scales, Π, cumulative_energy, wavenumber, filtering_spectrum[, spectral_slope, spectrum_map,
                      slope_map])

Results of a coarse-graining sweep over `scales`.

# Fields
- `scales::AbstractVector{T}`: filter scales ℓ in meters
- `Π::A`: specific energy-flux maps (m² s⁻³) in one `(spatial dims..., Nscales)` array; scale `s` is
  `selectdim(Π, ndims(Π), s)`
- `cumulative_energy::AbstractArray{T}`: cumulative coarse specific KE ½⟨|ū_ℓ|²⟩ per scale
  (Sadek–Aluie Eq. 15)
- `wavenumber::AbstractVector{T}`: filtering wavenumber `k_ℓ = L/ℓ` per scale
- `filtering_spectrum::AbstractArray{T}`: filtering spectral density `Ẽ(k_ℓ)` per scale (Eq. 14)
- `spectral_slope`: `d ln Ẽ/d ln k_ℓ` per scale under `slopes = Diagnostics.Slopes()`, `nothing` otherwise
- `spectrum_map`: the local spectrum `Ẽ(x, k_ℓ)`, shaped like `Π`, under `spectrum_maps = Diagnostics.SpectrumMaps()`,
  `nothing` otherwise
- `slope_map`: its slope per cell, under both, `nothing` otherwise

# Examples
```julia
res = coarse_grain(u, v, grid; scales=[10e3, 20e3, 30e3], kernel=TopHatKernel())
# Access results:
res.scales[1]              # First scale (10 km)
res.Π[:, :, 1]              # Energy-flux map at 10 km (a copy; `@view res.Π[:, :, 1]` aliases it)
res.cumulative_energy[1]   # cumulative coarse KE at 10 km
res.filtering_spectrum[1]  # filtering spectral density at k_ℓ = res.wavenumber[1]
```
"""
struct CoarseGrainResult{
    T<:AbstractFloat, N, A<:AbstractArray{T,N}, V<:AbstractVector{T}, C<:AbstractArray{T},
    SL<:Union{Nothing,AbstractArray{T}}, SM<:Union{Nothing,AbstractArray{T,N}}, PM<:Union{Nothing,AbstractArray{T,N}},
}
    scales::V
    Π::A
    cumulative_energy::C
    wavenumber::V
    filtering_spectrum::C
    spectral_slope::SL
    spectrum_map::SM
    slope_map::PM
end

CoarseGrainResult(scales::AbstractVector{T}, Π::AbstractArray{T,N}, cumE::AbstractArray{T},
                  kℓ::AbstractVector{T}, spec::AbstractArray{T}, slope = nothing, smap = nothing,
                  pmap = nothing) where {T<:AbstractFloat, N} =
    CoarseGrainResult{T, N, typeof(Π), typeof(scales), typeof(cumE), typeof(slope), typeof(smap), typeof(pmap)}(
        scales, Π, cumE, kℓ, spec, slope, smap, pmap)

# What a sweep writes its rates, slopes and maps into, and the derivative order they read.
@inline _result_outputs(r, L) =
    Diagnostics._SpectrumOutputs(r.filtering_spectrum, r.spectral_slope, r.spectrum_map, r.slope_map, L)
@inline _result_order(r) = Diagnostics._rate_order(r.spectral_slope, r.slope_map)

# ---------------------------------------------------------------------------
# The sweep, on any grid
# ---------------------------------------------------------------------------

"""
    coarse_grain(u, v, w, grid; scales, kernel=TopHatKernel(), execution=AutoBackend(), mask_strategy=ZeroFill(),
                 method=nothing, vector_filter=nothing, transform=AutoSpectralBackend(), L=1, spectrum=AutoSpectrum(),
                 derivative=AutoDerivative(), scale_derivative=AutoScaleDerivative(), slopes=NoSlopes(),
                 spectrum_maps=NoSpectrumMaps())
    coarse_grain(u, v, grid; scales, ...)  # no vertical component

Coarse-graining analysis across `scales`, returning a new [`CoarseGrainResult`](@ref). Allocating
form of [`coarse_grain!`](@ref), which reuses a result and workspace across calls. A one-component
field on a 1-D grid takes `coarse_grain(u, grid; …)`.

# Arguments
- `u`, `v`: the velocity components along the grid's first two directions (m/s), arrays of the
  grid's shape: one value per cell, a vector on a node set
- `w`: the vertical velocity, or `nothing`: on a 2-D grid a thin-layer third component, on a 3-D
  grid the third component
- `grid`: any `FlowGeometries` grid: geometry, cells and active-cell mask

# Keyword Arguments
- `scales::AbstractVector`: Vector of filter scales ℓ in meters (e.g., `10e3:10e3:100e3`)
- `kernel::AbstractFilterKernel=TopHatKernel()`: Filter kernel
- `execution::AbstractExecutionBackend=AutoBackend()`: Execution backend of the filter and of the
  flux's own loops; the fields and the result's `Π` live in its memory
- `mask_strategy::AbstractMaskStrategy=ZeroFill()`: Masking strategy (`ZeroFill()` or `Deformable()`).
  Under `ZeroFill` the kernel is position-independent, so filtering commutes with spatial derivatives.
  See [`Filtering.filter_field!`](@ref) for the boundary artifacts of both choices.
- `method::Union{Nothing,AbstractFilterMethod}=nothing`: filtering engine; `nothing` takes
  `plan_filter`'s per-grid default (real space where a grid has that engine)
- `vector_filter::Union{Nothing,AbstractVectorFilter}=nothing`: how the velocity and the stress are
  filtered on the surface of a sphere ([`Filtering.resolve_vector_filter`](@ref)). `nothing` takes the
  generalized convolution of Aluie (2019) under `ZeroFill`, whose spin transforms are NUFSHT's where it
  is loaded, FastSphericalHarmonics' on its own grids, and an in-core direct sum otherwise, and which
  reads no scalar filter plan, so none is built; under `Deformable` it takes
  `Filtering.CartesianComponents()`.
- `transform::AbstractSpectralBackend=AutoSpectralBackend()`: the transform library of
  `method = Spectral()` and of the generalized convolution
- `L::Real=1`: reference length in the wavenumber `k_ℓ = L/ℓ`; it rescales the spectral density by
  `1/L` (see [`Diagnostics.filtering_spectrum`](@ref)).
- `spectrum::AbstractSpectrumPolicy=Diagnostics.AutoSpectrum()`: how to fill `filtering_spectrum`.
  The density is guaranteed non-negative only for a kernel whose `|Ĝ|²` is monotone decreasing
  ([`Kernels.transfer_monotone`](@ref)). The default computes it for such a kernel and fills `NaN` for
  any other, the default `TopHatKernel` included. `Diagnostics.ForceSpectrum()` computes it anyway
  (warning once), `Diagnostics.StrictSpectrum()` throws for a non-monotone kernel, and
  `Diagnostics.NoSpectrum()` always fills `NaN`. `Π` and `cumulative_energy` do not depend on it.
- `derivative::AbstractDerivative=Derivatives.AutoDerivative()`: how the strain differentiates the filtered
  velocity: the filter's own derivative where its output is a series (`method = Spectral()` and the generalized
  convolution), a 5-node stencil or the least-squares gradient elsewhere ([`Derivatives.AbstractDerivative`](@ref)).
- `scale_derivative::AbstractScaleDerivative=Diagnostics.AutoScaleDerivative()`: how the density differentiates
  `E(ℓ)` in `ℓ`, exactly through the filter where its kernel allows, by differences across the scales otherwise.
- `slopes::AbstractSlopes=Diagnostics.NoSlopes()`: `Diagnostics.Slopes()` fills `spectral_slope`.
- `spectrum_maps::AbstractSpectrumMaps=Diagnostics.NoSpectrumMaps()`: `Diagnostics.SpectrumMaps()` fills
  `spectrum_map`, and with `Slopes()` `slope_map`.

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
    u::AbstractArray,
    v::AbstractArray,
    w::Union{Nothing, AbstractArray},
    grid::FlowGeometries.Grids.AbstractGrid{G,T};
    scales::AbstractVector,
    kernel::Kernels.AbstractFilterKernel = Kernels.TopHatKernel(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Union{Nothing, Filtering.AbstractFilterMethod} = nothing,
    vector_filter::Union{Nothing, Filtering.AbstractVectorFilter} = nothing,
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    L::Real = one(T),
    spectrum::Diagnostics.AbstractSpectrumPolicy = Diagnostics.AutoSpectrum(),
    derivative::Derivatives.AbstractDerivative = Derivatives.AutoDerivative(),
    scale_derivative::Diagnostics.AbstractScaleDerivative = Diagnostics.AutoScaleDerivative(),
    slopes::Diagnostics.AbstractSlopes = Diagnostics.NoSlopes(),
    spectrum_maps::Diagnostics.AbstractSpectrumMaps = Diagnostics.NoSpectrumMaps(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    result = _allocate_result(grid, length(scales), slopes, spectrum_maps; execution = execution)
    return coarse_grain!(
        result, u, v, w, grid;
        scales = scales, kernel = kernel, execution = execution, mask_strategy = mask_strategy,
        method = method, vector_filter = vector_filter, transform = transform, L = L, spectrum = spectrum,
        derivative = derivative, scale_derivative = scale_derivative,
    )
end

coarse_grain(u::AbstractArray, v::AbstractArray, grid::FlowGeometries.Grids.AbstractGrid; kwargs...) =
    coarse_grain(u, v, nothing, grid; kwargs...)

# The derivative route of a sweep on the workspace `ws`, resolved once from its first scale's plan: a sweep's
# plans share their method and form, and a derivative plan's weights depend on the grid alone.
@inline _sweep_route(derivative, deriv_plan, ws, plans, strat, grid) = Diagnostics._derivative_route(
    derivative, deriv_plan, Diagnostics._route_source(ws.spin, _read_plan(_scale_plan(plans, 1), strat)), grid,
)

@inline _plan_filter_sweep(grid, kernel, scales, strat, backend, method, transform) =
    Filtering.plan_filter_sweep(
        grid, kernel, scales; mask_strategy = strat, execution = backend, method = method, transform = transform,
    )

"""
    _sweep_setup(grid, batch_size, w, workspace, filter_plans, kernel, scales, execution, mask_strategy,
                 method, vector_filter, transform) -> (workspace, plans)

The workspace a sweep runs on, `workspace` or one built for the vector filter `vector_filter` resolves to,
and its per-scale plans: `filter_plans`, a family built here, or `nothing` where the workspace's
generalized convolution filters every field and no plans are given.
"""
function _sweep_setup(
    grid, batch_size::Tuple, w, workspace, filter_plans, kernel, scales, execution, strat, method,
    vector_filter, transform,
)
    vf = Filtering.resolve_vector_filter(grid, strat, vector_filter)
    sweep() = _plan_filter_sweep(grid, kernel, scales, strat, execution, method, transform)
    plans = _sweep_plans(filter_plans, workspace, vf, sweep)
    _check_plan_strategy(plans, strat)
    ws = _sweep_workspace(workspace, grid, batch_size, w, vf, transform, plans, execution)
    return ws, plans
end

# The plans a sweep reads: the caller's, or swept unless the vector filter, the given workspace's or the
# one `vf` resolves to, filters through its own transforms.
@inline _sweep_plans(plans, _, _, _) = plans
@inline _sweep_plans(::Nothing, ::Nothing, vf, sweep::F) where {F} = Diagnostics._setup_plan(nothing, vf, sweep)
@inline _sweep_plans(::Nothing, ws, _, sweep::F) where {F} = _sweep_unless_spin(ws.spin, sweep)
@inline _sweep_unless_spin(::Nothing, sweep::F) where {F} = sweep()
@inline _sweep_unless_spin(_, _) = nothing

# The caller's workspace, or one placed where the plans run.
_sweep_workspace(ws, _, _, _, _, _, _, _) = ws
_sweep_workspace(::Nothing, grid, batch_size, w, vf, transform, plans, execution) = Diagnostics.ΠWorkspace(
    grid, batch_size, w; vector_filter = vf, transform = transform,
    execution = Diagnostics._plan_backend(plans, execution),
)

# Scale `i`'s plan, `nothing` for a sweep reading none; and what a diagnostic reading the filtered fields
# takes in its place, the mask strategy.
@inline _scale_plan(::Nothing, ::Int) = nothing
@inline _scale_plan(plans, i::Int) = plans[i]
@inline _read_plan(::Nothing, strat) = strat
@inline _read_plan(plan, _) = plan

# With no analysis the `analyzed` keyword is omitted: not every `compute_Π!` method accepts it.
@inline _flux!(out, u, v, w, grid, kernel, scale, ws, plan, backend, strat, dplan, ::Nothing = nothing) =
    Diagnostics.compute_Π!(out, u, v, w, grid, kernel, scale;
        workspace = ws, filter_plan = plan, execution = backend, mask_strategy = strat, deriv_plan = dplan)

@inline _flux!(out, u, v, w, grid, kernel, scale, ws, plan, backend, strat, dplan, analyzed) =
    Diagnostics.compute_Π!(out, u, v, w, grid, kernel, scale;
        workspace = ws, filter_plan = plan, execution = backend, mask_strategy = strat,
        deriv_plan = dplan, analyzed = analyzed)

# The resolved policy the sweep fills `filtering_spectrum` with; throws or warns per `gate_spectrum`.
@inline function _check_spectrum(kernel, spectrum::Diagnostics.AbstractSpectrumPolicy)
    policy = Diagnostics.resolve_spectrum(kernel, spectrum)
    Diagnostics.gate_spectrum(kernel, policy)
    return policy
end

_check_plan_strategy(::Nothing, ::Filtering.AbstractMaskStrategy) = nothing

# The flux reads the strategy off each plan and the energy off `mask_strategy`, so prebuilt plans must
# have been built for the strategy the keyword names.
function _check_plan_strategy(plans, strategy::Filtering.AbstractMaskStrategy)
    isempty(plans) && return nothing
    s = Filtering.plan_strategy(first(plans))
    typeof(s) === typeof(strategy) || throw(ArgumentError(
        "filter_plans were built for $(nameof(typeof(s))) but mask_strategy is " *
        "$(nameof(typeof(strategy))); pass the mask_strategy the plans were built with",
    ))
    return nothing
end


"""
    coarse_grain!(result, u, v, w, grid; scales, kernel, workspace, filter_plans, deriv_plan, execution, mask_strategy,
                  method, vector_filter, transform, L, spectrum, derivative, scale_derivative)
    coarse_grain!(result, u, v, grid; scales, ...)  # no vertical component

In-place [`coarse_grain`](@ref): refills every buffer of an existing [`CoarseGrainResult`](@ref).
Supplying `workspace`, `filter_plans` and `deriv_plan` reuses the scratch and the per-scale plans as
well, for a sweep repeated across timesteps. The workspace and `result.Π` live in the memory of the
plans' backend, and the sweep's own loops run under the workspace's. A given `workspace` carries its own
vector filter; `vector_filter` and `transform` build one when it is not given.

`result` must already be sized for `length(scales)` scales over `grid`'s shape; a mismatch throws
`DimensionMismatch`.
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
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    # `nothing` takes `Filtering._default_method(grid)`.
    method::Union{Nothing, Filtering.AbstractFilterMethod} = nothing,
    vector_filter::Union{Nothing, Filtering.AbstractVectorFilter} = nothing,
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    L::Real = one(T),
    spectrum::Diagnostics.AbstractSpectrumPolicy = Diagnostics.AutoSpectrum(),
    derivative::Derivatives.AbstractDerivative = Derivatives.AutoDerivative(),
    scale_derivative::Diagnostics.AbstractScaleDerivative = Diagnostics.AutoScaleDerivative(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    _check_result_shape(result, grid, scales)
    policy = _check_spectrum(kernel, spectrum)
    isempty(scales) && return result
    ws, plans = _sweep_setup(grid, (), w, workspace, filter_plans, kernel, scales, execution, mask_strategy,
                             method, vector_filter, transform)
    # The filtered fields are differenced on the grid the strategy defines them on; the weights, built
    # there, are moved once to where the workspace's loops run.
    dplan = Diagnostics._placed(ws.at, _sweep_route(derivative, deriv_plan, ws, plans, mask_strategy, grid))
    rate = _sweep_rate(policy, scale_derivative, kernel, ws, plans, mask_strategy, u, v, w, _result_order(result))
    _coarse_grain_sweep!(result, u, v, w, grid, kernel, scales, ws, plans, dplan, rate, execution, mask_strategy, policy, L)
    return result
end

# The sweep and its density on a set-up workspace, plans, derivative plan and rate route; returns the sweep's
# analysis, whose spectra `held`, an earlier sweep's on fields of the same shape, receives in place.
function _coarse_grain_sweep!(
    result::CoarseGrainResult{T}, u, v, w, grid, kernel, scales, ws, plans, dplan, rate, execution, strat, policy, L,
    held = nothing,
) where {T}
    outputs = _result_outputs(result, L)
    analyzed = _sweep!(result, outputs, u, v, w, grid, kernel, scales, ws, plans, dplan, rate, execution, strat, held)
    result.wavenumber .= T(L) ./ result.scales
    Diagnostics.filtering_density!(result.filtering_spectrum, result.cumulative_energy, result.wavenumber, policy,
                                   rate, result.scales, L)
    Diagnostics.finish_slopes!(outputs, result.filtering_spectrum, result.wavenumber, policy, rate)
    return analyzed
end

# A sweep's scale-independent part into `held`'s spectra where an earlier sweep left them, freshly otherwise.
@inline _sweep_analysis(_held, u, v, w, grid, ws, plan) = Diagnostics.analyze_sweep(u, v, w, grid, ws, plan)
@inline _sweep_analysis(held::NamedTuple{(:velocity, :product)}, u, v, w, grid, ws, plan) =
    Diagnostics.analyze_sweep!(held, u, v, w, grid, ws, plan)

# How a sweep takes `⟨ū·∂_ℓū⟩`, and to `order` its second derivative, resolved once from its plans; none where no
# density is filled.
@inline _sweep_rate(::Diagnostics.NoSpectrum, _sd, _kernel, _ws, _plans, _strat, _u, _v, _w, _order) =
    Diagnostics._DifferenceRate()
@inline _sweep_rate(::Diagnostics.AbstractSpectrumPolicy, sd, kernel, ws, plans, strat, u, v, w, order) =
    Diagnostics._rate_route(sd, kernel, ws, _read_plan(plans, strat), u, v, w, order)

# `Π`, `E(ℓ)` and, on an exact rate route, `⟨ū·∂_ℓū⟩` (into `filtering_spectrum`) with the slopes and maps the outputs
# holds, of every scale into `result`; `plans` is `nothing` for a sweep reading none. Returns the analysis.
function _sweep!(
    result::CoarseGrainResult{T}, outputs, u, v, w, grid, kernel, scales, ws, plans, dplan, rate, execution, strat,
    held = nothing,
) where {T}
    # `energy_from_filtered` reads the filtered velocities `compute_Π!` leaves in `ws`.
    total_area = Diagnostics.active_area(grid)
    # The scale-independent part shared by every scale (spectral forward transforms, the spin analysis of
    # the generalized convolution, or the rotation into ambient Cartesian components), or `nothing`.
    analyzed = _sweep_analysis(held, u, v, w, grid, ws, _read_plan(_scale_plan(plans, 1), strat))
    for s_idx in eachindex(scales)
        scale = T(scales[s_idx])
        result.scales[s_idx] = scale
        plan = _scale_plan(plans, s_idx)
        _flux!(
            selectdim(result.Π, ndims(result.Π), s_idx), u, v, w, grid, kernel, scale,
            ws, plan, execution, strat, dplan, analyzed,
        )
        result.cumulative_energy[s_idx] =
            Diagnostics.energy_from_filtered(ws, _read_plan(plan, strat), w !== nothing, total_area)
        Diagnostics._store_rate!(outputs, s_idx, rate, ws, _read_plan(plan, strat), kernel, scale, u, v, w, total_area,
                                 analyzed)
    end
    return analyzed
end

# ---------------------------------------------------------------------------
# Batch axis: independent sweeps sharing one grid
# ---------------------------------------------------------------------------

# Parallel batch drivers, defined by the backend extensions: independent sweeps, one scratch set per
# worker.
function threaded_coarse_grain_batch!(args...; kwargs...)
    throw(ArgumentError("ThreadedBackend is unavailable — run `using OhMyThreads` (or use SerialBackend())."))
end

function distributed_coarse_grain_batch!(args...; kwargs...)
    throw(ArgumentError("DistributedBackend is unavailable — run `using Distributed` (or use SerialBackend())."))
end

function mpi_coarse_grain_batch!(args...; kwargs...)
    throw(ArgumentError("MPIBackend is unavailable — run `using MPI` (or use SerialBackend())."))
end

# The execution backend of each slice's own sweep: serial under host batching, so only the batch level
# claims the thread pool; a GPU backend runs inside each slice.
@inline _batch_inner_backend(resolved::ComputationalBackends.AbstractExecutionBackend) =
    resolved isa ComputationalBackends.GPUBackend ? resolved : ComputationalBackends.SerialBackend()

_batch_driver!(backend::ComputationalBackends.AbstractExecutionBackend, args...) = throw(ArgumentError(
    "coarse_grain_batch! has no batch driver for $(typeof(backend))",
))

# Host-serial slice loop over one set of scratch.
_batch_driver!(::ComputationalBackends.SerialBackend, batch, u, v, w, grid, valR, ctx) =
    _run_base!(batch, u, v, w, grid, valR, ctx, eachindex(batch.slices))

# GPU: each scale is computed for the whole batch in one pass; the filter engines, the tensor algebra and
# the stencil derivatives carry the trailing batch axes. `E(ℓ)` reduces the spatial axes to one energy
# per slice. `u`, `v`, `w` and `batch.Π` are in the device's memory.
function _batch_driver!(
    be::ComputationalBackends.GPUBackend, batch, u, v, w, grid, ::Val{R}, ctx,
) where {R}
    T = eltype(batch.Π)
    isempty(ctx.scales) && return batch
    ws, plans = _sweep_setup(
        grid, batch.batch_size, w, _pool_get(ctx.workspaces, 1), _pool_get(ctx.filter_plans, 1), ctx.kernel,
        ctx.scales, be, ctx.mask_strategy, ctx.method, ctx.vector_filter, ctx.transform,
    )
    # `compute_Π!` receives the whole batch, so the workspace must be batch-sized.
    _batch_ws_shape(ws) == (FlowGeometries.Grids.size_tuple(grid)..., batch.batch_size...) ||
        throw(DimensionMismatch(
            "coarse_grain_batch! on a GPU backend needs a batch-sized workspace: got scratch of size " *
            "$(_batch_ws_shape(ws)), expected $((FlowGeometries.Grids.size_tuple(grid)..., batch.batch_size...)). " *
            "Build it with `ΠWorkspace(grid, batch_size; execution)`.",
        ))
    dplan = Diagnostics._placed(
        ws.at, _sweep_route(ctx.derivative, _pool_get(ctx.deriv_plans, 1), ws, plans, ctx.mask_strategy, grid),
    )
    rate = _sweep_rate(ctx.spectrum, ctx.scale_derivative, ctx.kernel, ws, plans, ctx.mask_strategy, u, v, w,
                       _result_order(batch))
    _batch_scales!(batch, u, v, w, grid, Val(R), ctx, ws, plans, dplan, rate, be)
    batch.wavenumber .= T(ctx.L) ./ batch.scales
    for result in batch.slices
        Diagnostics.filtering_density!(result.filtering_spectrum, result.cumulative_energy, result.wavenumber,
                                       ctx.spectrum, rate, result.scales, ctx.L)
        Diagnostics.finish_slopes!(_result_outputs(result, ctx.L), result.filtering_spectrum, result.wavenumber,
                                   ctx.spectrum, rate)
    end
    return batch
end

# A batch's entries at scale `s` as a outputs over its slices: the per-scale values along the slices, the maps over the
# spatial axes and the slices.
_scale_outputs(batch, ::Val{R}, s::Int, L) where {R} = Diagnostics._SpectrumOutputs(
    selectdim(batch.filtering_spectrum, 1, s), _select(batch.spectral_slope, 1, s),
    _select(batch.spectrum_map, R + 1, s), _select(batch.slope_map, R + 1, s), L,
)
@inline _select(::Nothing, _d, _s) = nothing
@inline _select(A::AbstractArray, d::Int, s::Int) = selectdim(A, d, s)

# Every scale of a whole batch; `plans` is `nothing` for a sweep reading none.
function _batch_scales!(batch, u, v, w, grid, ::Val{R}, ctx, ws, plans, dplan, rate, be) where {R}
    T = eltype(batch.Π)
    strat = ctx.mask_strategy
    total_area = Diagnostics.active_area(grid)
    has_w = w !== nothing
    analyzed = Diagnostics.analyze_sweep(u, v, w, grid, ws, _read_plan(_scale_plan(plans, 1), strat))
    for s_idx in eachindex(ctx.scales)
        scale = T(ctx.scales[s_idx])
        selectdim(batch.scales, 1, s_idx) .= scale
        plan = _scale_plan(plans, s_idx)
        _flux!(selectdim(batch.Π, R + 1, s_idx), u, v, w, grid, ctx.kernel, scale, ws, plan, be, strat, dplan,
               analyzed)
        Diagnostics.energy_from_filtered!(
            selectdim(batch.cumulative_energy, 1, s_idx), ws, _read_plan(plan, strat), has_w, total_area,
        )
        Diagnostics._store_rates!(_scale_outputs(batch, Val(R), s_idx, ctx.L), s_idx, rate, ws, _read_plan(plan, strat),
                                  ctx.kernel, scale, u, v, w, total_area)
    end
    return batch
end

_batch_driver!(::ComputationalBackends.ThreadedBackend, batch, u, v, w, grid, valR, ctx) =
    threaded_coarse_grain_batch!(batch, u, v, w, grid, valR, ctx)
_batch_driver!(::ComputationalBackends.DistributedBackend, batch, u, v, w, grid, valR, ctx) =
    distributed_coarse_grain_batch!(batch, u, v, w, grid, valR, ctx)
_batch_driver!(be::ComputationalBackends.MPIBackend, batch, u, v, w, grid, valR, ctx) =
    mpi_coarse_grain_batch!(be, batch, u, v, w, grid, valR, ctx)

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
_slices_driver!(be::ComputationalBackends.MPIBackend, results, us, vs, ws, grids, ctx) =
    mpi_coarse_grain_slices!(be, results, us, vs, ws, grids, ctx)

"""
    CoarseGrainBatchResult(grid, Nscales, batch_size; execution = SerialBackend())

Batched result storage plus the per-slice [`CoarseGrainResult`](@ref) views a batch sweep writes into.

The batch axes trail: `Π` is `(spatial..., Nscales, batch...)` and the per-scale arrays are
`(Nscales, batch...)`, so `slices[i]` is a `CoarseGrainResult` of contiguous views into this storage.
`batch_size` may hold any number of axes: `(Nt,)`, `(Nlevels,)`, `(Nlevels, Nt)`. `Π` lives in the
memory of `execution` and the per-scale arrays in host memory; under a `DistributedBackend` every array
is shared with the worker processes (Distributed extension), which write the slices. `slopes` and
`spectrum_maps` add the slopes and maps of [`CoarseGrainResult`](@ref), the maps shaped like `Π`.
"""
struct CoarseGrainBatchResult{
    T<:AbstractFloat, NB, A<:AbstractArray{T}, S<:AbstractArray{T}, R<:AbstractVector,
    SL<:Union{Nothing,AbstractArray{T}}, SM<:Union{Nothing,AbstractArray{T}}, PM<:Union{Nothing,AbstractArray{T}},
}
    Π::A
    scales::S
    cumulative_energy::S
    wavenumber::S
    filtering_spectrum::S
    slices::R
    batch_size::NTuple{NB,Int}
    spectral_slope::SL
    spectrum_map::SM
    slope_map::PM
end

"""
    _result_maps(execution, T, dims) -> AbstractArray{T}
    _result_values(execution, T, dims) -> AbstractArray{T}

Zeroed storage for a result's flux maps, in `execution`'s memory, and for its per-scale values, in host
memory. The Distributed extension gives a `DistributedBackend` memory the worker processes share.
"""
_result_maps(execution::ComputationalBackends.AbstractExecutionBackend, ::Type{T}, dims::Dims) where {T} =
    Filtering._allocate(execution, T, dims)
_result_values(::ComputationalBackends.AbstractExecutionBackend, ::Type{T}, dims::Dims) where {T} = zeros(T, dims)

function CoarseGrainBatchResult(
    grid::FlowGeometries.Grids.AbstractGrid{G,T}, Nscales::Integer, batch_size::NTuple{NB,Integer};
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
    slopes::Diagnostics.AbstractSlopes = Diagnostics.NoSlopes(),
    spectrum_maps::Diagnostics.AbstractSpectrumMaps = Diagnostics.NoSpectrumMaps(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}, NB}
    spatial = FlowGeometries.Grids.size_tuple(grid)
    bs = Int.(batch_size)
    ns = Int(Nscales)
    Π = _result_maps(execution, T, (spatial..., ns, bs...))
    scales = _result_values(execution, T, (ns, bs...))
    cumE = _result_values(execution, T, (ns, bs...))
    kℓ = _result_values(execution, T, (ns, bs...))
    spec = _result_values(execution, T, (ns, bs...))
    slope = _slope_storage(slopes, execution, T, (ns, bs...))
    smap, pmap = _map_storage(spectrum_maps, slopes, execution, T, (spatial..., ns, bs...))
    lead = Val(length(spatial) + 1)
    # Column-major order of `CartesianIndices(bs)` is the linear slice index the drivers use.
    slices = vec([
        CoarseGrainResult(
            view(scales, :, Tuple(J)...),
            view(Π, ntuple(_ -> Colon(), lead)..., Tuple(J)...),
            view(cumE, :, Tuple(J)...),
            view(kℓ, :, Tuple(J)...),
            view(spec, :, Tuple(J)...),
            _slice_view(slope, Val(1), J),
            _slice_view(smap, lead, J),
            _slice_view(pmap, lead, J),
        ) for J in CartesianIndices(bs)
    ])
    return CoarseGrainBatchResult(Π, scales, cumE, kℓ, spec, slices, bs, slope, smap, pmap)
end

# A result's slope vector and maps where asked for, `nothing` otherwise; a slope map needs both.
_slope_storage(::Diagnostics.NoSlopes, _execution, ::Type, _dims) = nothing
_slope_storage(::Diagnostics.Slopes, execution, ::Type{T}, dims::Dims) where {T} = _result_values(execution, T, dims)
_map_storage(::Diagnostics.NoSpectrumMaps, _slopes, _execution, ::Type, _dims) = (nothing, nothing)
_map_storage(::Diagnostics.SpectrumMaps, ::Diagnostics.NoSlopes, execution, ::Type{T}, dims::Dims) where {T} =
    (_result_maps(execution, T, dims), nothing)
_map_storage(::Diagnostics.SpectrumMaps, ::Diagnostics.Slopes, execution, ::Type{T}, dims::Dims) where {T} =
    (_result_maps(execution, T, dims), _result_maps(execution, T, dims))

# Fixes the trailing batch indices with every leading axis whole, a contiguous view.
@inline _slice_view(A::AbstractArray, ::Val{R}, J::CartesianIndex) where {R} =
    view(A, ntuple(_ -> Colon(), Val(R))..., Tuple(J)...)
@inline _slice_view(::Nothing, ::Val, ::CartesianIndex) = nothing

"""
    coarse_grain_batch!(batch, u, v, w, grid; scales, kernel, workspaces, filter_plans, deriv_plans, execution, mask_strategy, method, vector_filter, transform, L, spectrum) -> batch
    coarse_grain_batch!(batch, u, v, grid; scales, ...)  # no vertical component

Run a full [`coarse_grain!`](@ref) sweep per slice over a batch that shares one grid.

The grid fixes the spatial rank; every trailing dimension of `u`, `v` and `w` beyond it is a batch
axis. Against a 2D grid, `(x, y, t)` is a time batch and `(x, y, z, t)` is `Nlevels·Nt` slices;
against a true-3D grid an `(x, y, z)` array is a single slice.

On a host backend `execution` parallelizes over slices and each slice's sweep runs serially, as in
[`Filtering.filter_slices!`](@ref); a GPU backend computes the whole batch per scale, with `u`, `v`, `w`
and `batch` in its memory (`CoarseGrainBatchResult(…; execution)`).

`workspaces`, `filter_plans` and `deriv_plans` are scratch pools indexed by worker: vectors of equal
length, at least the concurrency used (`Threads.nthreads()` suffices), make a repeated batch
allocation-free; [`Filtering.worker_copy`](@ref) of one family makes a pool sharing its tables. A pool left
`nothing` is built once and each further worker takes a worker copy of it.

[`coarse_grain_slices!`](@ref) takes slices on different grids. `vector_filter`, `transform` and the
`spectrum` policy behave as in [`coarse_grain`](@ref).
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
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Union{Nothing, Filtering.AbstractFilterMethod} = nothing,
    vector_filter::Union{Nothing, Filtering.AbstractVectorFilter} = nothing,
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    L::Real = one(T),
    spectrum::Diagnostics.AbstractSpectrumPolicy = Diagnostics.AutoSpectrum(),
    derivative::Derivatives.AbstractDerivative = Derivatives.AutoDerivative(),
    scale_derivative::Diagnostics.AbstractScaleDerivative = Diagnostics.AutoScaleDerivative(),
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
    policy = _check_spectrum(kernel, spectrum)
    resolved = Filtering._resolve_slice_backend(execution)
    inner = _batch_inner_backend(resolved)
    ctx = (; scales, kernel, workspaces, filter_plans, deriv_plans, mask_strategy, method, vector_filter,
             transform, L, spectrum = policy, inner, derivative, scale_derivative)
    return _batch_driver!(resolved, batch, u, v, w, grid, valR, ctx)
end

function coarse_grain_batch!(
    batch::CoarseGrainBatchResult, u::AbstractArray, v::AbstractArray,
    grid::FlowGeometries.Grids.AbstractGrid; kwargs...,
)
    return coarse_grain_batch!(batch, u, v, nothing, grid; kwargs...)
end

# `Val`-typed rank: `size(A)` sliced by a runtime range infers no fixed-size tuple and allocates.
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
    coarse_grain_batch_slice!(batch, u, v, w, grid, Val(R), ctx, t, scratch, rate[, held]) -> analysis

Slice `t`'s sweep into `batch.slices[t]` on one worker's `scratch` (see `_worker_scratch`) and rate route, with the
slice's execution backend `ctx.inner`; `held` is the analysis an earlier slice of the worker returned, whose spectra
this one refills. Every host batch driver runs its slices through this.
"""
function coarse_grain_batch_slice!(
    batch::CoarseGrainBatchResult, u, v, w, grid, ::Val{R}, ctx, t::Integer, scratch::NamedTuple, rate, held = nothing,
) where {R}
    J = CartesianIndices(batch.batch_size)[t]
    result = batch.slices[t]
    _check_result_shape(result, grid, ctx.scales)
    isempty(ctx.scales) && return held
    ws = scratch.workspace
    return _coarse_grain_sweep!(
        result, _slice_view(u, Val(R), J), _slice_view(v, Val(R), J), _slice_view(w, Val(R), J), grid, ctx.kernel,
        ctx.scales, ws, scratch.filter_plans, Diagnostics._placed(ws.at, scratch.deriv_plan), rate, ctx.inner,
        ctx.mask_strategy, ctx.spectrum, ctx.L, held,
    )
end

# The rate route of a worker's slices, from slice `t`'s fields: it holds buffers and the plans' field-independent
# parts, so one serves every slice the worker sweeps.
function _slice_rate(batch, u, v, w, ::Val{R}, ctx, t::Integer, scratch::NamedTuple) where {R}
    J = CartesianIndices(batch.batch_size)[t]
    return _sweep_rate(ctx.spectrum, ctx.scale_derivative, ctx.kernel, scratch.workspace, scratch.filter_plans,
                       ctx.mask_strategy, _slice_view(u, Val(R), J), _slice_view(v, Val(R), J), _slice_view(w, Val(R), J),
                       _result_order(batch))
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

How many slices a batch may run at once: `min(nslices, n)`, with `n` the length of the supplied
scratch pools or `Threads.nthreads()` without any. A worker indexes its scratch by chunk, so no two
workers share scratch.
"""
function batch_concurrency(ctx, nslices::Integer)
    np = _pool_length(ctx)
    return min(nslices, np === nothing ? Threads.nthreads() : np)
end

"""
    _batch_base(batch, u, v, w, grid, Val(R), ctx, t) -> (; scratch, rate)

What a batch's workers share, built once: entry 1 of each pool the caller supplied, and a workspace, per-scale filter
plans and derivative route built here for each pool left `nothing` (`filter_plans` stays `nothing` where the
workspace's generalized convolution reads none, `_sweep_setup`), with the rate route of slice `t`'s fields. A family
whose rate route is its real-space scale rates holds them, so the workers' copies share their tables. Worker 1 sweeps
on `scratch` itself.
"""
function _batch_base(batch, u, v, w, grid, valR, ctx, t::Integer)
    workspace, filter_plans = _sweep_setup(
        grid, (), w, _pool_get(ctx.workspaces, 1), _pool_get(ctx.filter_plans, 1), ctx.kernel, ctx.scales,
        ctx.inner, ctx.mask_strategy, ctx.method, ctx.vector_filter, ctx.transform,
    )
    deriv_plan = _sweep_route(ctx.derivative, _pool_get(ctx.deriv_plans, 1), workspace, filter_plans,
                              ctx.mask_strategy, grid)
    rate = _slice_rate(batch, u, v, w, valR, ctx, t, (; workspace, filter_plans, deriv_plan))
    plans = _holding_rates(filter_plans, rate, _result_order(batch))
    return (; scratch = (; workspace, filter_plans = plans, deriv_plan), rate)
end

# A family holding the scale rates its rate route was built from.
_holding_rates(plans, _rate, _order) = plans
_holding_rates(f::Filtering.FilterPlanFamily{<:Any,<:Any,<:Any,<:Any,Nothing}, rate::Diagnostics._WindowRate, order) =
    Filtering.FilterPlanFamily(f.grid_plan, f.plans, f.scratch, rate.parts, order)

"""
    _worker_scratch(ctx, base, grid, p) -> (; workspace, filter_plans, deriv_plan)

Worker `p`'s scratch for a batch sweep: entry `p` of each pool the caller supplied, and for each pool left `nothing`
a worker copy of the base's (`Filtering.worker_copy`), which shares its tables and holds its own buffers. A derivative
plan is read-only and shared; a series gradient holds buffers and is built from the worker's plans.
"""
function _worker_scratch(ctx, base, grid, p::Integer)
    p == 1 && return base.scratch
    workspace = _pool_or_copy(ctx.workspaces, p, base.scratch.workspace)
    filter_plans = _pool_or_copy(ctx.filter_plans, p, base.scratch.filter_plans)
    deriv_plan = _worker_route(_pool_get(ctx.deriv_plans, p), base.scratch.deriv_plan, ctx, workspace, filter_plans, grid)
    return (; workspace, filter_plans, deriv_plan)
end

_pool_or_copy(::Nothing, _p, x) = _worker_copy(x)
_pool_or_copy(pool::AbstractVector, p::Integer, _) = pool[p]
_worker_copy(::Nothing) = nothing
_worker_copy(x) = Filtering.worker_copy(x)
_worker_copy(f::Filtering.FilterPlanFamily) = Filtering.worker_copy(f)
_worker_copy(v::AbstractVector{<:Filtering.AbstractFilterPlan}) = map(Filtering.worker_copy, v)

_worker_route(given, _route, _ctx, _ws, _plans, _grid) = given
_worker_route(::Nothing, route, _ctx, _ws, _plans, _grid) = route
_worker_route(::Nothing, ::Diagnostics.SeriesStrain, ctx, ws, plans, grid) =
    _sweep_route(ctx.derivative, nothing, ws, plans, ctx.mask_strategy, grid)

"""
    _run_base!(batch, u, v, w, grid, Val(R), ctx, slices) -> batch

Sweep `slices` on one base built from the first of them: the loop of a driver whose workers are separate processes.
"""
function _run_base!(batch, u, v, w, grid, valR, ctx, slices)
    isempty(slices) && return batch
    base = _batch_base(batch, u, v, w, grid, valR, ctx, first(slices))
    return _run_slices!(batch, u, v, w, grid, valR, ctx, slices, base.scratch, base.rate)
end

"""
    _run_slices!(batch, u, v, w, grid, Val(R), ctx, slices, scratch) -> batch

Sweep each slice in `slices` on one worker's `scratch` and one rate route; the loop compiles against their
concrete types.
"""
function _run_slices!(batch, u, v, w, grid, valR, ctx, slices, scratch)
    isempty(slices) && return batch
    return _run_slices!(batch, u, v, w, grid, valR, ctx, slices, scratch,
                        _slice_rate(batch, u, v, w, valR, ctx, first(slices), scratch))
end

function _run_slices!(batch, u, v, w, grid, valR, ctx, slices, scratch, rate)
    isempty(slices) && return batch
    held = coarse_grain_batch_slice!(batch, u, v, w, grid, valR, ctx, first(slices), scratch, rate)
    return _run_rest!(batch, u, v, w, grid, valR, ctx, Iterators.drop(slices, 1), scratch, rate, held)
end

# The worker's later slices, each refilling the analysis spectra the first one left.
function _run_rest!(batch, u, v, w, grid, valR, ctx, slices, scratch, rate, held)
    for t in slices
        coarse_grain_batch_slice!(batch, u, v, w, grid, valR, ctx, t, scratch, rate, held)
    end
    return batch
end

# Pools are indexed by worker, so the supplied ones must agree in length; `-1` marks one not supplied.
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
# Batch axis: independent sweeps, one grid per slice
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

Relative per-slice sweep cost, points × scales, for longest-first scheduling of a ragged batch. It
orders the slices and is not a time.
"""
slice_pipeline_costs(grids::AbstractVector, Nscales::Integer) =
    [prod(FlowGeometries.Grids.size_tuple(g)) * Nscales for g in grids]

"""
    coarse_grain_slices!(results, us, vs, ws, grids; scales, kernel, workspaces, filter_plans, deriv_plans, execution, mask_strategy, method, vector_filter, transform, L, spectrum) -> results
    coarse_grain_slices!(results, us, vs, grids; scales, ...)  # no vertical component

Run a full [`coarse_grain!`](@ref) sweep per slice over a batch whose slices each carry their own
grid: different regions, point counts or node sets. `results` is a vector of independent
[`CoarseGrainResult`](@ref)s; [`coarse_grain_batch!`](@ref) serves slices on one shared grid.

Slices are dispatched longest-first under a dynamic scheduler (see [`slice_pipeline_costs`](@ref)).
Scratch is per slice: `workspaces`, `filter_plans` and `deriv_plans`, when given, hold one entry per
slice, each built for that slice's grid. Each slice's sweep runs serially on a host backend, as in
`coarse_grain_batch!`.

`vector_filter`, `transform` and the `spectrum` policy behave as in [`coarse_grain`](@ref).
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
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Union{Nothing, Filtering.AbstractFilterMethod} = nothing,
    vector_filter::Union{Nothing, Filtering.AbstractVectorFilter} = nothing,
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    L::Real = 1,
    spectrum::Diagnostics.AbstractSpectrumPolicy = Diagnostics.AutoSpectrum(),
    derivative::Derivatives.AbstractDerivative = Derivatives.AutoDerivative(),
    scale_derivative::Diagnostics.AbstractScaleDerivative = Diagnostics.AutoScaleDerivative(),
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
    policy = _check_spectrum(kernel, spectrum)
    resolved = Filtering._resolve_slice_backend(execution)
    inner = _batch_inner_backend(resolved)
    ctx = (; scales, kernel, workspaces, filter_plans, deriv_plans, mask_strategy, method, vector_filter,
             transform, L, spectrum = policy, inner, derivative, scale_derivative)
    return _slices_driver!(resolved, results, us, vs, ws, grids, ctx)
end

function coarse_grain_slices!(
    results::AbstractVector, us::AbstractVector, vs::AbstractVector, grids::AbstractVector; kwargs...,
)
    return coarse_grain_slices!(results, us, vs, nothing, grids; kwargs...)
end

"""
    coarse_grain_slice_serial!(results, us, vs, ws, grids, ctx, t) -> results[t]

Slice `t`'s sweep with the slice's execution backend `ctx.inner`. The serial loop and every parallel
slices driver run each slice through this.
"""
function coarse_grain_slice_serial!(results, us, vs, ws, grids, ctx, t::Integer)
    return coarse_grain!(
        results[t], us[t], vs[t], ws === nothing ? nothing : ws[t], grids[t];
        scales = ctx.scales,
        kernel = ctx.kernel,
        workspace = _pool_get(ctx.workspaces, t),
        filter_plans = _pool_get(ctx.filter_plans, t),
        deriv_plan = _pool_get(ctx.deriv_plans, t),
        execution = ctx.inner,
        mask_strategy = ctx.mask_strategy,
        method = ctx.method,
        vector_filter = ctx.vector_filter,
        transform = ctx.transform,
        L = ctx.L,
        spectrum = ctx.spectrum,
        derivative = ctx.derivative,
        scale_derivative = ctx.scale_derivative,
    )
end

coarse_grain!(result::CoarseGrainResult, u::AbstractArray, v::AbstractArray,
              grid::FlowGeometries.Grids.AbstractGrid; kwargs...) =
    coarse_grain!(result, u, v, nothing, grid; kwargs...)

"""
    coarse_grain_profile!(batch, u, v, w, grid; scales, kernel, workspaces, filter_plans, deriv_plans, execution, mask_strategy, method, vector_filter, transform, L, spectrum) -> batch
    coarse_grain_profile!(batch, u, v, grid; scales, ...)  # no vertical component

Vertical-profile sweep, in place: each vertical level is its own 2D/2.5D problem on the shared
horizontal grid, filtered horizontally (see [`Diagnostics.compute_Π!`](@ref) on the thin-layer
regime). Runs [`coarse_grain_batch!`](@ref) with the level axis as the batch axis.

`batch` is a [`CoarseGrainBatchResult`](@ref) with batch size `(Nlevels,)`: `Π` is
`(Nx, Ny, Nscales, Nlevels)` and `cumulative_energy`/`filtering_spectrum` are `(Nscales, Nlevels)`.
Energies are per level; a vertical sum needs thickness weights, which this function does not take.

The `spectrum` policy behaves as in [`coarse_grain`](@ref).
"""
function coarse_grain_profile!(
    batch::CoarseGrainBatchResult,
    u::AbstractArray{T,3},
    v::AbstractArray{T,3},
    w::Union{Nothing, AbstractArray{T,3}},
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2};
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    return coarse_grain_batch!(batch, u, v, w, grid; kwargs...)
end

function coarse_grain_profile!(
    batch::CoarseGrainBatchResult, u::AbstractArray{T,3}, v::AbstractArray{T,3},
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2}; kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    return coarse_grain_batch!(batch, u, v, nothing, grid; kwargs...)
end

"""
    coarse_grain_profile(u, v, w, grid; scales, kernel=TopHatKernel(), execution=AutoBackend(), mask_strategy=ZeroFill(), method=nothing, vector_filter=nothing, transform=AutoSpectralBackend(), L=1, spectrum=AutoSpectrum())

Allocating [`coarse_grain_profile!`](@ref): sizes a [`CoarseGrainBatchResult`](@ref) for `size(u, 3)`
levels and fills it.
"""
function coarse_grain_profile(
    u::AbstractArray{T,3},
    v::AbstractArray{T,3},
    w::Union{Nothing, AbstractArray{T,3}},
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2};
    scales::AbstractVector,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    slopes::Diagnostics.AbstractSlopes = Diagnostics.NoSlopes(),
    spectrum_maps::Diagnostics.AbstractSpectrumMaps = Diagnostics.NoSpectrumMaps(),
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    batch = CoarseGrainBatchResult(grid, length(scales), (size(u, 3),); execution, slopes, spectrum_maps)
    return coarse_grain_profile!(batch, u, v, w, grid; scales = scales, execution = execution, kwargs...)
end

# 2.5D convenience wrapper (no vertical velocity).
function coarse_grain_profile(
    u::AbstractArray{T,3}, v::AbstractArray{T,3},
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2}; kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    return coarse_grain_profile(u, v, nothing, grid; kwargs...)
end

# ---------------------------------------------------------------------------
# 1D Cartesian pipeline: one velocity component `u` on a 1D `StructuredGrid`, with E(ℓ) = ½⟨ū_ℓ²⟩.
# ---------------------------------------------------------------------------
function coarse_grain(
    u::AbstractVector,
    grid::FlowGeometries.Grids.StructuredGrid{T,G,1};
    scales::AbstractVector,
    kernel::Kernels.AbstractFilterKernel = Kernels.TopHatKernel(),
    filter_plans::Union{Nothing, AbstractVector} = nothing,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Union{Nothing, Filtering.AbstractFilterMethod} = nothing,
    L::Real = one(T),
    spectrum::Diagnostics.AbstractSpectrumPolicy = Diagnostics.AutoSpectrum(),
    derivative::Derivatives.AbstractDerivative = Derivatives.AutoDerivative(),
    scale_derivative::Diagnostics.AbstractScaleDerivative = Diagnostics.AutoScaleDerivative(),
    slopes::Diagnostics.AbstractSlopes = Diagnostics.NoSlopes(),
    spectrum_maps::Diagnostics.AbstractSpectrumMaps = Diagnostics.NoSpectrumMaps(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    result = _allocate_result(grid, length(scales), slopes, spectrum_maps; execution = execution)
    workspace = Diagnostics.ΠWorkspace(grid; execution = execution)   # 1-D: no vertical component
    return coarse_grain!(
        result, u, grid;
        scales = scales, kernel = kernel, workspace = workspace, filter_plans = filter_plans,
        execution = execution, mask_strategy = mask_strategy, method = method, L = L, spectrum = spectrum,
        derivative = derivative, scale_derivative = scale_derivative,
    )
end

function coarse_grain!(
    result::CoarseGrainResult{T},
    u::AbstractVector,
    grid::FlowGeometries.Grids.StructuredGrid{T,G,1};
    scales::AbstractVector,
    kernel::Kernels.AbstractFilterKernel = Kernels.TopHatKernel(),
    workspace::Union{Nothing, Diagnostics.ΠWorkspace} = nothing,
    filter_plans::Union{Nothing, AbstractVector} = nothing,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Union{Nothing, Filtering.AbstractFilterMethod} = nothing,
    L::Real = one(T),
    spectrum::Diagnostics.AbstractSpectrumPolicy = Diagnostics.AutoSpectrum(),
    derivative::Derivatives.AbstractDerivative = Derivatives.AutoDerivative(),
    scale_derivative::Diagnostics.AbstractScaleDerivative = Diagnostics.AutoScaleDerivative(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    _check_result_shape(result, grid, scales)
    policy = _check_spectrum(kernel, spectrum)
    isempty(scales) && return result
    plans = filter_plans === nothing ?
        _plan_filter_sweep(grid, kernel, scales, mask_strategy, execution, method,
                           SpectralBackends.AutoSpectralBackend()) : filter_plans
    _check_plan_strategy(plans, mask_strategy)
    ws = workspace === nothing ?   # 1-D: no vertical component
        Diagnostics.ΠWorkspace(grid; execution = Filtering.plan_execution(first(plans))) : workspace
    dplan = Diagnostics._placed(ws.at, _sweep_route(derivative, nothing, ws, plans, mask_strategy, grid))
    rate = _sweep_rate(policy, scale_derivative, kernel, ws, plans, mask_strategy, u, nothing, nothing,
                       _result_order(result))
    outputs = _result_outputs(result, L)

    total_area = Diagnostics.active_area(grid)

    for s_idx in eachindex(scales)
        scale = T(scales[s_idx])
        result.scales[s_idx] = scale
        plan = plans[s_idx]
        # `compute_Π!` leaves `ū` in `ws.u_filt`.
        Diagnostics.compute_Π!(
            view(result.Π, :, s_idx),
            u, grid, kernel, scale;
            workspace = ws, filter_plan = plan, execution = execution, mask_strategy = mask_strategy,
            deriv_plan = dplan,
        )
        integrated_energy = Diagnostics._kinetic_integral(
            (ws.u_filt,), Diagnostics._output(ws.at, plan), 0, ws.at.execution,
        )
        result.cumulative_energy[s_idx] = T(0.5) * integrated_energy / total_area
        Diagnostics._store_rate!(outputs, s_idx, rate, ws, plan, kernel, scale, u, nothing, nothing, total_area, nothing)
    end

    result.wavenumber .= T(L) ./ result.scales
    Diagnostics.filtering_density!(result.filtering_spectrum, result.cumulative_energy, result.wavenumber, policy,
                                   rate, result.scales, L)
    Diagnostics.finish_slopes!(outputs, result.filtering_spectrum, result.wavenumber, policy, rate)
    return result
end

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# A zeroed result for `Nscales` scales over `grid`'s spatial shape, of any rank: `Π` in `execution`'s
# memory, the per-scale values in host memory, all shared with the worker processes under a
# `DistributedBackend`.
function _allocate_result(
    grid::FlowGeometries.Grids.AbstractGrid{G,T}, Nscales::Integer,
    slopes::Diagnostics.AbstractSlopes = Diagnostics.NoSlopes(),
    spectrum_maps::Diagnostics.AbstractSpectrumMaps = Diagnostics.NoSpectrumMaps();
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
) where {G, T<:AbstractFloat}
    spatial = FlowGeometries.Grids.size_tuple(grid)
    ns = Int(Nscales)
    smap, pmap = _map_storage(spectrum_maps, slopes, execution, T, (spatial..., ns))
    return CoarseGrainResult(
        _result_values(execution, T, (ns,)),
        _result_maps(execution, T, (spatial..., ns)),
        _result_values(execution, T, (ns,)),
        _result_values(execution, T, (ns,)),
        _result_values(execution, T, (ns,)),
        _slope_storage(slopes, execution, T, (ns,)),
        smap,
        pmap,
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
    Base.front(size(result.Π)) == FlowGeometries.Grids.size_tuple(grid) || throw(DimensionMismatch(
        "result.Π's spatial shape $(Base.front(size(result.Π))) does not match grid shape $(FlowGeometries.Grids.size_tuple(grid))",
    ))
    return nothing
end

end # module
