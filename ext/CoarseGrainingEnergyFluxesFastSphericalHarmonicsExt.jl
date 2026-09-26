module CoarseGrainingEnergyFluxesFastSphericalHarmonicsExt

using FastSphericalHarmonics: FastSphericalHarmonics as FSH
using FFTW: FFTW
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF
using FlowGeometries: FlowGeometries
using FlowTransformBindings: FlowTransformBindings as FTB

# Spectral filtering for uniform spherical grids, via the scalar spherical-harmonic transform. The
# wavenumber of degree `l` is the Laplace–Beltrami eigenvalue `k_l = √(l(l+1))/R`, so a degree-`l`
# coefficient is scaled by `Ĝ(k_l, ℓ)`; `Ĝ(0) = 1` preserves the mean.
#
# FSH samples `N` colatitudes `θ_j = π(j−½)/N` by `M = 2N−1` longitudes `φ_k = 2π(k−1)/M`
# (`FastSphericalHarmonics.sph_points(N)`), the nodes of a `ClenshawCurtisSampling` grid at
# `nlon = 2·nlat − 1`, and stores a field as `F[θ, φ]` where this package stores `[lon, lat]` — hence
# the transpose in and out.
#
# Masking follows the same normalized-convolution identity as the other spectral backends, with the
# `Deformable` denominator computed once at plan-build time.
#
# FastSphericalHarmonics transforms host memory. A plan for a backend on a device reads each field
# through a host buffer and writes its output back through it. Under `MPIBackend` every rank holds the
# field and transforms it whole.
#
# Each transform runs through `FTB.with_fasttransforms_threads`, which sets FastTransforms' OpenMP count
# on the calling OS thread to the grid plan's count and restores it after.
_transform!(C, gp) = FTB.with_fasttransforms_threads(() -> FSH.sph_transform!(C; cache = gp.cache), gp.nthreads)
_evaluate!(C, gp) = FTB.with_fasttransforms_threads(() -> FSH.sph_evaluate!(C; cache = gp.cache), gp.nthreads)

"""
    SHTGridPlan

The half of a spherical-harmonic plan the filter scale does not reach: the `SphPlanCache` holding the
transform's internal FFT plans, the mask, the grid's validated mode limits, the OpenMP thread count the
transforms run at, and whether fields arrive in device memory.

Only `mult` — the per-degree transfer multiplier — and the `Deformable` renormalization computed
through it depend on ℓ. Sharing the rest across a sweep matters more here than for a plain FFT: a
fresh `SphPlanCache` per scale means the transform rebuilds its internal plans the first time it sees
each one, and the node validation below re-derives and re-compares the quadrature nodes every time.

FastTransforms builds the cached plans on the libfftw3 FFTW.jl loads, whose planner is process-global,
so the cache is filled when the grid plan is built, under FFTW.jl's planner lock at the plan's thread
count, and a transform only reads it. Executing a cached plan writes its scratch, so a concurrent
driver needs its own grid plan per worker.
"""
struct SHTGridPlan{T<:AbstractFloat, MK} <: CGEF.Filtering.AbstractGridPlan
    cache::FSH.SphPlanCache{T}
    N::Int
    M::Int
    lmax::Int
    mmax::Int
    radius::T
    mask::MK
    nthreads::Int
    staged::Bool
end

"""
    SHTScratch

The transient half of a spherical-harmonic plan: the `N × M` buffer carrying the [lon,lat] ↔ FSH [θ,φ]
transpose and, in place, the coefficients and the evaluated points; the `M × N` `mask · field` array;
and for a device plan the `M × N` host buffer a field and its output pass through. One per concurrent
worker. FSH's `sph_transform!`/`sph_evaluate!` take `Array{T,2}`.
"""
struct SHTScratch{T<:AbstractFloat, S<:AbstractMatrix{T}, MV<:AbstractMatrix{T}, ST<:Union{Nothing,Matrix{T}}} <:
       CGEF.Filtering.AbstractFilterScratch
    scratch::S        # N × M, FSH layout
    masked_input::MV  # M × N [lon,lat]; unused when mask === nothing
    stage::ST         # M × N host buffer, or nothing for a host plan
end

"""
    SHTFilterPlan

Cached spherical-harmonic filter plan: the per-coefficient transfer multiplier `Ĝ(k_l, ℓ)` laid out on
the FSH coefficient array (a function of degree `l` alone), the shared [`SHTGridPlan`](@ref), the
`Deformable` inverse local-mass renormalization (or `nothing`), and the [`SHTScratch`](@ref) the apply
transposes through. Built by
`plan_filter(spherical_structured_grid, kernel, scale; method = Spectral())`.
"""
struct SHTFilterPlan{
    T<:AbstractFloat, A<:AbstractMatrix{T}, GP<:SHTGridPlan{T}, R,
    MS<:CGEF.Filtering.AbstractMaskStrategy, SC<:SHTScratch{T},
} <: CGEF.Filtering.AbstractFilterPlan
    mult::A        # N × M multiplier on the spherical-harmonic coefficients
    grid_plan::GP
    invrenorm::R   # 1/filter(mask) for Deformable, zero at a masked point; or nothing
    strategy::MS
    scratch::SC
end

CGEF.Filtering.plan_strategy(plan::SHTFilterPlan) = plan.strategy

# The grids whose nodes FSH samples, given `nlon == 2·nlat − 1`. FSH transforms `Float64` only.
const _CCGrid = FlowGeometries.Grids.StructuredGrid{
    Float64, <:FlowGeometries.Geometry.SphericalGeometry{Float64}, 2,
    <:FlowGeometries.SphericalSampling.AbstractClenshawCurtisSampling,
}
const _SphericalGrid = FlowGeometries.Grids.StructuredGrid{T,<:FlowGeometries.Geometry.SphericalGeometry{T}} where {T}

_sht_shape(grid::_CCGrid) = ((M, N) = size(FlowGeometries.Grids.mask(grid)); M == 2N - 1)

# FSH transforms one field at a time, so a plan serves a trailing batch of any extent and `batch` plans
# nothing.
function _sht_grid_plan(
    grid::_CCGrid;
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy = CGEF.Filtering.ZeroFill(),
    backend::CGEF.ComputationalBackends.AbstractExecutionBackend = CGEF.ComputationalBackends.AutoBackend(),
    batch::Union{Nothing,Integer} = nothing,
)
    M, N = size(FlowGeometries.Grids.mask(grid))   # CGEF layout is [x, y] = [longitude, latitude] = [M, N]
    _sht_shape(grid) || throw(ArgumentError(
        "FastSphericalHarmonics transforms a ClenshawCurtisSampling grid of 2N-1 longitudes by N " *
        "latitudes; this one has $M by $N. Build it with " *
        "`FlowGeometries.Connectivity.structured_grid(ClenshawCurtisSampling(), N)`, or filter it over its " *
        "cells with `spectral_backend = AutoSpectralBackend()` and `using NUFSHT`.",
    ))
    nth = CGEF.Filtering._library_threads(backend)
    return SHTGridPlan(
        _warmed_cache(N, M, nth), N, M, N - 1, M ÷ 2,
        Float64(FlowGeometries.Geometry.radius(FlowGeometries.Grids.grid_geometry(grid))),
        all(FlowGeometries.Grids.mask(grid)) ? nothing : FlowGeometries.Grids.mask(grid),
        nth, !CGEF.Filtering._host_memory(backend),
    )
end

# Both transforms' plans, built by one transform and one evaluation of a zero field; see `SHTGridPlan`.
function _warmed_cache(N::Int, M::Int, nthreads::Int)
    cache = FSH.SphPlanCache{Float64}()
    scratch = zeros(Float64, N, M)
    FFTW.set_num_threads(nthreads) do
        FTB.with_fasttransforms_threads(nthreads) do
            FSH.sph_transform!(scratch; cache = cache)
            FSH.sph_evaluate!(scratch; cache = cache)
        end
    end
    return cache
end

@noinline _not_sht_grid(grid) = throw(ArgumentError(
    "FastSphericalHarmonics transforms a Float64 ClenshawCurtisSampling grid of 2N-1 longitudes by N " *
    "latitudes, built with `FlowGeometries.Connectivity.structured_grid(ClenshawCurtisSampling(), N)`; " *
    "this grid's sampling is $(nameof(typeof(FlowGeometries.Grids.sampling(grid)))). Filter it over its " *
    "cells with `spectral_backend = AutoSpectralBackend()` and `using NUFSHT`.",
))

# FSH works in [θ,φ] (N×M) and this package stores [lon,lat] (M×N), so the scratch carries one of each.
_sht_scratch(gp::SHTGridPlan{T}) where {T<:AbstractFloat} =
    SHTScratch(zeros(T, gp.N, gp.M), zeros(T, gp.M, gp.N), gp.staged ? zeros(T, gp.M, gp.N) : nothing)

# `FSHTSpectralBackend` is honoured on the grids FSH samples and refused on every other spherical grid;
# `Auto` takes the harmonic transform on a grid of FSH's shape and filters any other over its cells.
CGEF.Filtering.spectral_grid_plan(
    ::CGEF.SpectralBackends.AbstractFSHTSpectralBackend, grid::_CCGrid, ::CGEF.Kernels.AbstractFilterKernel;
    kwargs...,
) = _sht_grid_plan(grid; kwargs...)

CGEF.Filtering.spectral_grid_plan(
    ::CGEF.SpectralBackends.AbstractFSHTSpectralBackend, grid::_SphericalGrid, ::CGEF.Kernels.AbstractFilterKernel;
    kwargs...,
) = _not_sht_grid(grid)

CGEF.Filtering.spectral_grid_plan(
    auto::CGEF.SpectralBackends.AbstractAutoSpectralBackend, grid::_CCGrid,
    kernel::CGEF.Kernels.AbstractFilterKernel; kwargs...,
) = _sht_shape(grid) ? _sht_grid_plan(grid; kwargs...) :
    CGEF.Filtering._node_set_grid_plan(auto, grid, kernel; kwargs...)

CGEF.Filtering.spectral_scratch(gp::SHTGridPlan) = _sht_scratch(gp)

CGEF.Filtering.spectral_filter_plan(
    ::CGEF.SpectralBackends.AbstractFSHTSpectralBackend, grid::_CCGrid, kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::Float64; kwargs...,
) = _sht_filter_plan(grid, kernel, scale; kwargs...)

CGEF.Filtering.spectral_filter_plan(
    ::CGEF.SpectralBackends.AbstractFSHTSpectralBackend, grid::_SphericalGrid,
    ::CGEF.Kernels.AbstractFilterKernel, ::AbstractFloat; kwargs...,
) = _not_sht_grid(grid)

CGEF.Filtering.spectral_filter_plan(
    auto::CGEF.SpectralBackends.AbstractAutoSpectralBackend, grid::_CCGrid,
    kernel::CGEF.Kernels.AbstractFilterKernel, scale::Float64; kwargs...,
) = _sht_shape(grid) ? _sht_filter_plan(grid, kernel, scale; kwargs...) :
    CGEF.Filtering._node_set_filter_plan(auto, grid, kernel, scale; kwargs...)

function _sht_filter_plan(
    grid::_CCGrid,
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::Float64;
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy = CGEF.Filtering.ZeroFill(),
    backend::CGEF.ComputationalBackends.AbstractExecutionBackend = CGEF.ComputationalBackends.AutoBackend(),
    batch::Union{Nothing,Integer} = nothing,
    grid_plan::Union{Nothing,SHTGridPlan} = nothing,
    scratch::Union{Nothing,SHTScratch} = nothing,
)
    T = Float64
    gp = grid_plan === nothing ? _sht_grid_plan(grid; backend = backend) : grid_plan
    sc = scratch === nothing ? _sht_scratch(gp) : scratch
    N, M, R = gp.N, gp.M, gp.radius

    # Mirror FastSphericalHarmonics' own coefficient-iteration (see `sph_laplace!`): the packed layout
    # stores degrees up to lmax + mmax for high |m|. The transfer value depends only on l, not m, so
    # compute it once per degree (not once per (l,m) pair, ~2l+1 times more calls) — negligible for
    # Gaussian/SharpSpectral but genuinely wasteful for TopHatKernel's Legendre-recurrence evaluation.
    lmax, mmax = gp.lmax, gp.mmax
    lmax_full = lmax + mmax
    transfer_by_l = [CGEF.Kernels.spectral_transfer_degree(kernel, l, scale, R) for l in 0:lmax_full]
    mult = ones(T, N, M)
    for l in 0:lmax_full, m in (-l):l
        if l - lmax <= abs(m) <= mmax
            mult[FSH.sph_mode(l, m)] = transfer_by_l[l+1]
        end
    end

    mask = gp.mask
    invrenorm = if mask !== nothing && mask_strategy isa CGEF.Filtering.Deformable
        # Local kernel mass over active points, `filter(mask)`, through the plan's own
        # transform/multiply/inverse pipeline. The mask is fixed for the plan, so this is built once
        # and stored inverted.
        sc.masked_input .= mask                         # [lon,lat] (M×N)
        permutedims!(sc.scratch, sc.masked_input, (2, 1))  # → FSH [θ,φ] (N×M)
        _transform!(sc.scratch, gp)
        sc.scratch .*= mult
        _evaluate!(sc.scratch, gp)
        renorm = zeros(T, M, N)
        permutedims!(renorm, sc.scratch, (2, 1))        # back to [lon,lat]
        ir = similar(renorm)
        # A masked point is zero under `Deformable`, as in the real-space engines.
        @. ir = ifelse(mask, CGEF.Filtering._inv_mass(renorm), zero(T))
        ir
    else
        nothing   # ZeroFill: already exactly `filter(mask .* field)`, no renormalization
    end
    return SHTFilterPlan(mult, gp, invrenorm, mask_strategy, sc)
end

# The forward harmonic transform depends on the field alone, so a sweep runs it once and each scale only
# scales the coefficients by its own `Ĝ(k_l, ℓ)` and evaluates back to points.
CGEF.Filtering.analyze_buffer(plan::SHTFilterPlan, ::AbstractMatrix) = similar(plan.scratch.scratch)

# The field in host memory: itself for a host plan, copied into the stage buffer for a device plan.
@inline _host_field(::Nothing, field) = field
@inline _host_field(stage::Matrix, field) = copyto!(stage, field)
# Where a host routine writes an output bound for `out`, and the copy back into `out`.
@inline _host_out(::Nothing, out) = out
@inline _host_out(stage::Matrix, _) = stage
@inline _store!(::Nothing, out, _) = out
@inline _store!(::Matrix, out, host) = copyto!(out, host)

function _sht_analyze!(Ĉ, field, plan::SHTFilterPlan)
    gp, sc = plan.grid_plan, plan.scratch
    if gp.mask === nothing
        permutedims!(Ĉ, field, (2, 1))
    else
        @. sc.masked_input = gp.mask * field
        permutedims!(Ĉ, sc.masked_input, (2, 1))
    end
    _transform!(Ĉ, gp)
    return Ĉ
end

function _sht_synthesize!(out, Ĉ, plan::SHTFilterPlan)
    # `sph_evaluate!` works in place, and `Ĉ` is reused by every later scale, so evaluate a scaled copy.
    sc = plan.scratch
    sc.scratch .= Ĉ .* plan.mult
    _evaluate!(sc.scratch, plan.grid_plan)
    permutedims!(out, sc.scratch, (2, 1))
    plan.invrenorm === nothing || (out .*= plan.invrenorm)
    return out
end

# Host memory in and out; `out` may be `field`, which is read before `out` is written.
function _sht_apply!(out, field, plan::SHTFilterPlan)
    gp, sc = plan.grid_plan, plan.scratch
    if gp.mask === nothing
        permutedims!(sc.scratch, field, (2, 1))     # [lon, lat] (M×N) → FSH [θ, φ] (N×M), in place
    else
        @. sc.masked_input = gp.mask * field
        permutedims!(sc.scratch, sc.masked_input, (2, 1))
    end
    _transform!(sc.scratch, gp)                     # in place: scratch now holds coefficients
    sc.scratch .*= plan.mult                        # Ĝ(k_l, ℓ) per coefficient
    _evaluate!(sc.scratch, gp)                      # in place: scratch now holds point values
    permutedims!(out, sc.scratch, (2, 1))           # back to [lon, lat]
    plan.invrenorm === nothing || (out .*= plan.invrenorm)
    return out
end

CGEF.Filtering.filter_analyze!(Ĉ::AbstractMatrix{T}, field::AbstractMatrix{T}, plan::SHTFilterPlan{T}) where {T<:AbstractFloat} =
    _sht_analyze!(Ĉ, _host_field(plan.scratch.stage, field), plan)

function CGEF.Filtering.filter_synthesize!(
    out::AbstractMatrix{T}, Ĉ::AbstractMatrix{T}, plan::SHTFilterPlan{T},
) where {T<:AbstractFloat}
    stage = plan.scratch.stage
    return _store!(stage, out, _sht_synthesize!(_host_out(stage, out), Ĉ, plan))
end

function CGEF.Filtering.filter_apply!(
    out::AbstractMatrix{T}, field::AbstractMatrix{T}, plan::SHTFilterPlan{T},
) where {T<:AbstractFloat}
    stage = plan.scratch.stage
    return _store!(stage, out, _sht_apply!(_host_out(stage, out), _host_field(stage, field), plan))
end

# A trailing batch axis: FSH transforms one field at a time, so each field of the batch takes the
# single-field apply, a device field's slice passing through the stage buffer by linear offset.
CGEF.Filtering._batched_fields(outs, ::SHTFilterPlan) = ndims(first(outs)) == 3

function CGEF.Filtering.filter_apply_batched!(
    out::AbstractArray{T,3}, field::AbstractArray{T,3}, plan::SHTFilterPlan{T},
) where {T<:AbstractFloat}
    gp, stage = plan.grid_plan, plan.scratch.stage
    size(out) == size(field) || throw(DimensionMismatch(
        "filter_apply_batched! got out $(size(out)) and field $(size(field))",
    ))
    (size(out, 1), size(out, 2)) == (gp.M, gp.N) || throw(DimensionMismatch(
        "field's leading axes $((size(out, 1), size(out, 2))) do not match the plan's $((gp.M, gp.N))",
    ))
    n = gp.M * gp.N
    for b in axes(out, 3)
        if stage === nothing
            _sht_apply!(view(out, :, :, b), view(field, :, :, b), plan)
        else
            copyto!(stage, 1, field, (b - 1) * n + 1, n)
            _sht_apply!(stage, stage, plan)
            copyto!(out, (b - 1) * n + 1, stage, 1, n)
        end
    end
    return out
end

end # module
