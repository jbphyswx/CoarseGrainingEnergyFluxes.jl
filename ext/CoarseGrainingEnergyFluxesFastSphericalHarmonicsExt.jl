module CoarseGrainingEnergyFluxesFastSphericalHarmonicsExt

using FastSphericalHarmonics: FastSphericalHarmonics as FSH
using FFTW: FFTW
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF
using FlowGeometries: FlowGeometries
using FlowTransformBindings: FlowTransformBindings as FTB

# Spectral filtering for uniform spherical grids by the scalar spherical-harmonic transform: each
# degree-`l` coefficient is scaled by the kernel's `Ĝ(l)`, `Kernels.spectral_transfer_degrees!`.
#
# FSH samples `N` colatitudes `θ_j = π(j−½)/N` by `M = 2N−1` longitudes `φ_k = 2π(k−1)/M`
# (`FastSphericalHarmonics.sph_points(N)`), the nodes of a `ClenshawCurtisSampling` grid at
# `nlon = 2·nlat − 1`. FSH stores a field as `F[θ, φ]` and this package as `[lon, lat]`, so every
# transform is wrapped in a transpose.
#
# FSH transforms host memory: a device plan passes each field and its output through a host buffer.
# Under `MPIBackend` every rank transforms the whole field.
_transform!(C, gp) = FTB.with_fasttransforms_threads(() -> FSH.sph_transform!(C; cache = gp.cache), gp.nthreads)
_evaluate!(C, gp) = FTB.with_fasttransforms_threads(() -> FSH.sph_evaluate!(C; cache = gp.cache), gp.nthreads)

"""
    SHTGridPlan

The half of a spherical-harmonic plan the filter scale does not reach: the `SphPlanCache` holding the
transform's internal FFT plans, the mask, the grid's validated mode limits, the OpenMP thread count the
transforms run at, whether fields arrive in device memory, and the execution backend the plan was built
for. The ℓ-dependent multiplier and `Deformable` renormalization live in the `SHTFilterPlan`.

FastTransforms plans on the process-global libfftw3 of FFTW.jl, so the cache is filled when the grid
plan is built, under FFTW.jl's planner lock at the plan's thread count, and a transform only reads it.
Executing a cached plan writes its scratch: a concurrent driver needs one grid plan per worker.
"""
struct SHTGridPlan{T<:AbstractFloat, MK, B<:CGEF.ComputationalBackends.AbstractExecutionBackend} <:
       CGEF.Filtering.AbstractGridPlan
    cache::FSH.SphPlanCache{T}
    N::Int
    M::Int
    lmax::Int
    mmax::Int
    radius::T
    mask::MK
    nthreads::Int
    staged::Bool
    execution::B
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

Cached spherical-harmonic filter plan: the per-coefficient transfer multiplier `Ĝ(l)` laid out on the
FSH coefficient array, the shared [`SHTGridPlan`](@ref), the
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
CGEF.Filtering.plan_execution(plan::SHTFilterPlan) = plan.grid_plan.execution

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
    execution::CGEF.ComputationalBackends.AbstractExecutionBackend = CGEF.ComputationalBackends.AutoBackend(),
    batch::Union{Nothing,Integer} = nothing,
)
    M, N = size(FlowGeometries.Grids.mask(grid))   # CGEF layout is [x, y] = [longitude, latitude] = [M, N]
    _sht_shape(grid) || throw(ArgumentError(
        "FastSphericalHarmonics transforms a ClenshawCurtisSampling grid of 2N-1 longitudes by N " *
        "latitudes; this one has $M by $N. Build it with " *
        "`FlowGeometries.Connectivity.structured_grid(ClenshawCurtisSampling(), N)`, or filter it over its " *
        "cells with `transform = AutoSpectralBackend()` and `using NUFSHT`.",
    ))
    nth = CGEF.Filtering._library_threads(execution)
    return SHTGridPlan(
        _warmed_cache(N, M, nth), N, M, N - 1, M ÷ 2,
        Float64(FlowGeometries.Geometry.radius(FlowGeometries.Grids.grid_geometry(grid))),
        all(FlowGeometries.Grids.mask(grid)) ? nothing : FlowGeometries.Grids.mask(grid),
        nth, !CGEF.Filtering._host_memory(execution), execution,
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
    "cells with `transform = AutoSpectralBackend()` and `using NUFSHT`.",
))

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
    execution::CGEF.ComputationalBackends.AbstractExecutionBackend = CGEF.ComputationalBackends.AutoBackend(),
    batch::Union{Nothing,Integer} = nothing,
    grid_plan::Union{Nothing,SHTGridPlan} = nothing,
    scratch::Union{Nothing,SHTScratch} = nothing,
)
    T = Float64
    gp = grid_plan === nothing ? _sht_grid_plan(grid; execution = execution) : grid_plan
    sc = scratch === nothing ? _sht_scratch(gp) : scratch
    N, M, R = gp.N, gp.M, gp.radius

    # FSH's packed layout holds degrees up to `lmax + mmax` for high `|m|`, with the mode bounds of
    # `FSH.sph_laplace!`.
    lmax, mmax = gp.lmax, gp.mmax
    lmax_full = lmax + mmax
    transfer_by_l = CGEF.Kernels.spectral_transfer_degrees!(zeros(T, lmax_full + 1), kernel, scale, R)
    mult = ones(T, N, M)
    for l in 0:lmax_full, m in (-l):l
        if l - lmax <= abs(m) <= mmax
            mult[FSH.sph_mode(l, m)] = transfer_by_l[l+1]
        end
    end

    mask = gp.mask
    invrenorm = if mask !== nothing && mask_strategy isa CGEF.Filtering.Deformable
        # Local kernel mass over active points, `filter(mask)`, through the plan's own transforms.
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

# The forward transform depends on the field alone, so a sweep shares one analysis across its scales.
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
    sc.scratch .*= plan.mult                        # Ĝ(l) per coefficient
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

# ---------------------------------------------------------------------------
# Spin-weighted transforms of the generalized convolution on FSH's grids
# ---------------------------------------------------------------------------

"""
    SHTSpinTransforms

The spin-0, 1 and 2 transforms of `Filtering.GeneralizedConvolution` on a grid FSH samples: the
`SpinSphPlanCache` holding every spin's FFT plans, the `N × M` layout, and the OpenMP thread count the
transforms run at. Spin `s` stores degree `r − 1 + max(|s|, ⌊c/2⌋)` in slot `(r, c)`, degrees `0:2N−2`
in all. The cache is filled when the transforms are built, as [`SHTGridPlan`](@ref)'s is, and a transform
writes its scratch: a concurrent driver needs its own.
"""
struct SHTSpinTransforms{C<:FSH.SpinSphPlanCache} <: CGEF.Filtering.AbstractSpinTransforms
    cache::C
    N::Int
    M::Int
    nthreads::Int
end

Base.show(io::IO, tr::SHTSpinTransforms) = print(io, "SHTSpinTransforms(", tr.N, " × ", tr.M, ")")
Base.show(io::IO, ::MIME"text/plain", tr::SHTSpinTransforms) = show(io, tr)

function _sht_spin_transforms(grid::_CCGrid, lmax, nthreads::Integer)
    M, N = size(FlowGeometries.Grids.mask(grid))
    _sht_shape(grid) || _not_sht_grid(grid)
    lmax === nothing || lmax == N - 1 || throw(ArgumentError(
        "FastSphericalHarmonics transforms at its grid's band limit, lmax = $(N - 1); got lmax = $lmax. " *
        "Pass `GeneralizedConvolution()`, or transform over the grid's cells with " *
        "`transform = AutoSpectralBackend()`, which takes NUFSHT where it is loaded and the direct sum " *
        "otherwise.",
    ))
    cache = FSH.SpinSphPlanCache{ComplexF64}()
    F = zeros(ComplexF64, N, M)
    nth = Int(nthreads)
    FFTW.set_num_threads(nth) do
        FTB.with_fasttransforms_threads(nth) do
            for s in 0:2
                FSH.spinsph_transform!(F, s; cache = cache)
                FSH.spinsph_evaluate!(F, s; cache = cache)
            end
        end
    end
    return SHTSpinTransforms(cache, N, M, nth)
end

CGEF.Filtering.spin_transforms(
    ::CGEF.SpectralBackends.AbstractFSHTSpectralBackend, grid::_CCGrid;
    lmax = nothing, nthreads::Integer = 1, nufft = nothing, execution = nothing,
) = _sht_spin_transforms(grid, lmax, nthreads)

CGEF.Filtering.spin_transforms(
    ::CGEF.SpectralBackends.AbstractFSHTSpectralBackend, grid::_SphericalGrid; kwargs...,
) = _not_sht_grid(grid)

CGEF.Filtering.spin_transforms(
    auto::CGEF.SpectralBackends.AbstractAutoSpectralBackend, grid::_CCGrid;
    lmax = nothing, nthreads::Integer = 1,
    nufft::CGEF.SpectralBackends.AbstractSpectralBackend = CGEF.SpectralBackends.AutoSpectralBackend(),
    execution::CGEF.ComputationalBackends.AbstractLocalBackend = CGEF.ComputationalBackends.SerialBackend(),
) = _sht_shape(grid) && lmax === nothing ? _sht_spin_transforms(grid, lmax, nthreads) :
    CGEF.Filtering._scattered_spin_transforms(auto, grid; lmax, nthreads, nufft, execution)

CGEF.Filtering.spin_field(tr::SHTSpinTransforms) = zeros(ComplexF64, tr.N, tr.M)
CGEF.Filtering.spin_field_size(tr::SHTSpinTransforms) = (tr.N, tr.M)
CGEF.Filtering.spin_position(tr::SHTSpinTransforms) = CGEF.Filtering.TransposedCells(tr.M, tr.N)
CGEF.Filtering.spin_coefficients(tr::SHTSpinTransforms) = zeros(ComplexF64, tr.N, tr.M)
CGEF.Filtering.spin_transfer_buffer(tr::SHTSpinTransforms) = zeros(Float64, 2tr.N - 1)

function CGEF.Filtering.spin_analyze!(C, F, tr::SHTSpinTransforms, s::Int)
    copyto!(C, F)
    FTB.with_fasttransforms_threads(() -> FSH.spinsph_transform!(C, s; cache = tr.cache), tr.nthreads)
    return C
end

function CGEF.Filtering.spin_synthesize!(F, C, W, Ĝ, tr::SHTSpinTransforms, s::Int)
    a = abs(s)
    @inbounds for c in axes(C, 2), r in axes(C, 1)
        W[r, c] = C[r, c] * Ĝ[r + max(a, c ÷ 2)]
    end
    FTB.with_fasttransforms_threads(() -> FSH.spinsph_evaluate!(W, s; cache = tr.cache), tr.nthreads)
    copyto!(F, W)
    return F
end

end # module
