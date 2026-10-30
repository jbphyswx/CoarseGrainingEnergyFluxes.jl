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
# (`FastSphericalHarmonics.sph_points(N)`), the nodes of a `Fejer1Sampling` grid at
# `nlon = 2·nlat − 1`. FSH stores a field as `F[θ, φ]` and this package as `[lon, lat]`, so every
# transform is wrapped in a transpose.
#
# FSH transforms host memory: a device plan passes each field and its output through a host buffer.
# Under `MPIBackend` every rank transforms the whole field.
_transform!(C, gp) = FTB.with_fasttransforms_threads(() -> FSH.sph_transform!(C; cache = gp.cache), gp.nthreads)
_evaluate!(C, gp) = FTB.with_fasttransforms_threads(() -> FSH.sph_evaluate!(C; cache = gp.cache), gp.nthreads)

"""
    FSHGridPlan{T}

The half of a spherical-harmonic plan the filter scale does not reach, for fields of element type `T`:
the `SphPlanCache` holding the transform's internal FFT plans, the mask, the grid's validated mode
limits, the OpenMP thread count the transforms run at, and the execution backend the plan was built for,
whose memory the fields arrive in. FSH transforms in `Float64`. The ℓ-dependent multiplier and
`Deformable` renormalization live in the `FSHFilterPlan`.

FastTransforms plans on the process-global libfftw3 of FFTW.jl, so the cache is filled when the grid
plan is built, under FFTW.jl's planner lock at the plan's thread count, and a transform only reads it.
Executing a cached plan writes its scratch: a concurrent driver needs one grid plan per worker.
"""
struct FSHGridPlan{T<:AbstractFloat, MK, B<:CGEF.ComputationalBackends.AbstractExecutionBackend} <:
       CGEF.Filtering.AbstractGridPlan
    cache::FSH.SphPlanCache{Float64}
    N::Int
    M::Int
    lmax::Int
    mmax::Int
    radius::T
    mask::MK
    nthreads::Int
    execution::B
end

"""
    FSHScratch

The transient half of a spherical-harmonic plan, in `Float64`: the `N × M` buffer carrying the [lon,lat] ↔
FSH [θ,φ] transpose and, in place, the coefficients and the evaluated points; the `M × N` `mask · field`
array; and for a device plan or fields of another float type the `M × N` host buffer a field and its
output pass through. One per concurrent worker. FSH's `sph_transform!`/`sph_evaluate!` take
`Array{Float64,2}`.
"""
struct FSHScratch{T<:AbstractFloat, S<:AbstractMatrix{T}, MV<:AbstractMatrix{T}, ST<:Union{Nothing,Matrix{T}}} <:
       CGEF.Filtering.AbstractFilterScratch
    scratch::S        # N × M, FSH layout
    masked_input::MV  # M × N [lon,lat]; unused when mask === nothing
    stage::ST         # M × N host buffer, or nothing for a host plan
end

"""
    FSHFilterPlan{T}

Cached spherical-harmonic filter plan for fields of element type `T`, in `Float64` within: the
per-coefficient transfer multiplier `Ĝ(l)` laid out on the
FSH coefficient array, the shared [`FSHGridPlan`](@ref), the
`Deformable` inverse local-mass renormalization (or `nothing`), and the [`FSHScratch`](@ref) the apply
transposes through. Built by
`plan_filter(spherical_structured_grid, kernel, scale; method = Spectral())`.
"""
struct FSHFilterPlan{
    T<:AbstractFloat, A<:AbstractMatrix{Float64}, GP<:FSHGridPlan{T}, R,
    MS<:CGEF.Filtering.AbstractMaskStrategy, SC<:FSHScratch{Float64},
} <: CGEF.Filtering.AbstractFilterPlan
    mult::A        # N × M multiplier on the spherical-harmonic coefficients
    grid_plan::GP
    invrenorm::R   # 1/filter(mask) for Deformable, zero at a masked point; or nothing
    strategy::MS
    scratch::SC
end

_fsh_plan(mult, gp::FSHGridPlan{T}, invrenorm, strategy, sc) where {T} =
    FSHFilterPlan{T,typeof(mult),typeof(gp),typeof(invrenorm),typeof(strategy),typeof(sc)}(mult, gp, invrenorm, strategy, sc)

CGEF.Filtering.plan_strategy(plan::FSHFilterPlan) = plan.strategy
CGEF.Filtering.plan_execution(plan::FSHFilterPlan) = plan.grid_plan.execution

# The grids whose nodes FSH samples, given `nlon == 2·nlat − 1`, of any float type; FSH transforms in
# `Float64`.
const _CCGrid{T} = FlowGeometries.Grids.StructuredGrid{
    T, <:FlowGeometries.Geometry.SphericalGeometry{T}, 2,
    <:FlowGeometries.SphericalSampling.AbstractFejer1Sampling,
} where {T}
const _SphericalGrid = FlowGeometries.Grids.StructuredGrid{T,<:FlowGeometries.Geometry.SphericalGeometry{T}} where {T}

_fsh_shape(grid::_CCGrid) = ((M, N) = size(FlowGeometries.Grids.mask(grid)); M == 2N - 1)

# FSH transforms one field at a time, so a plan serves a trailing batch of any extent and `batch` plans
# nothing. The sphere has no bounded direction, so `max_scale` pads nothing.
function _fsh_grid_plan(
    grid::_CCGrid{T};
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy = CGEF.Filtering.ZeroFill(),
    execution::CGEF.ComputationalBackends.AbstractExecutionBackend = CGEF.ComputationalBackends.AutoBackend(),
    batch::Union{Nothing,Integer} = nothing,
    max_scale::Real = T(0),
) where {T}
    M, N = size(FlowGeometries.Grids.mask(grid))   # CGEF layout is [x, y] = [longitude, latitude] = [M, N]
    _fsh_shape(grid) || throw(ArgumentError(
        "FastSphericalHarmonics transforms a Fejer1Sampling grid of 2N-1 longitudes by N " *
        "latitudes; this one has $M by $N. Build it with " *
        "`FlowGeometries.Connectivity.structured_grid(Fejer1Sampling(), N)`, or filter it over its " *
        "cells with `transform = AutoSpectralBackend()` and `using NUFSHT`.",
    ))
    nth = CGEF.Filtering._library_threads(execution)
    mask = CGEF.Filtering._plan_mask(FlowGeometries.Grids.mask(grid), identity, CGEF.ComputationalBackends.SerialBackend())
    return FSHGridPlan{T,typeof(mask),typeof(execution)}(
        _warmed_cache(N, M, nth), N, M, N - 1, M ÷ 2,
        Float64(FlowGeometries.Geometry.radius(FlowGeometries.Grids.grid_geometry(grid))), mask, nth, execution,
    )
end

# Both transforms' plans, built by one transform and one evaluation of a zero field; see `FSHGridPlan`.
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

@noinline _not_fsh_grid(grid) = throw(ArgumentError(
    "FastSphericalHarmonics transforms a Fejer1Sampling grid of 2N-1 longitudes by N latitudes, built " *
    "with `FlowGeometries.Connectivity.structured_grid(Fejer1Sampling(), N)`; this grid's sampling is " *
    "$(nameof(typeof(FlowGeometries.Grids.sampling(grid)))). Filter it over its cells with " *
    "`transform = AutoSpectralBackend()` and `using NUFSHT`.",
))

# A field in device memory, or of another float type than FSH's `Float64`, passes through a host
# `Float64` buffer; both are answered from types.
_fsh_scratch(gp::FSHGridPlan{T}) where {T<:AbstractFloat} = FSHScratch(
    zeros(Float64, gp.N, gp.M), zeros(Float64, gp.M, gp.N),
    (T === Float64 && CGEF.Filtering._host_memory(gp.execution)) ? nothing : zeros(Float64, gp.M, gp.N),
)

# `FSHTSpectralBackend` is honoured on the grids FSH samples and refused on every other spherical grid;
# `Auto` takes the harmonic transform on a grid of FSH's shape and filters any other over its cells.
CGEF.Filtering.spectral_grid_plan(
    ::CGEF.SpectralBackends.AbstractFSHTSpectralBackend, grid::_CCGrid, ::CGEF.Kernels.AbstractFilterKernel;
    kwargs...,
) = _fsh_grid_plan(grid; kwargs...)

CGEF.Filtering.spectral_grid_plan(
    ::CGEF.SpectralBackends.AbstractFSHTSpectralBackend, grid::_SphericalGrid, ::CGEF.Kernels.AbstractFilterKernel;
    kwargs...,
) = _not_fsh_grid(grid)

CGEF.Filtering.spectral_grid_plan(
    auto::CGEF.SpectralBackends.AbstractAutoSpectralBackend, grid::_CCGrid,
    kernel::CGEF.Kernels.AbstractFilterKernel; kwargs...,
) = _fsh_shape(grid) ? _fsh_grid_plan(grid; kwargs...) :
    CGEF.Filtering._node_set_grid_plan(auto, grid, kernel; kwargs...)

CGEF.Filtering.spectral_scratch(gp::FSHGridPlan) = _fsh_scratch(gp)

# A worker's copy: the multiplier and renormalization shared, the plan cache and scratch its own.
CGEF.Filtering._scratch(plan::FSHFilterPlan) = plan.scratch
CGEF.Filtering._with_scratch(plan::FSHFilterPlan, sc::FSHScratch) =
    _fsh_plan(plan.mult, plan.grid_plan, plan.invrenorm, plan.strategy, sc)
CGEF.Filtering._grid_plan(plan::FSHFilterPlan) = plan.grid_plan
CGEF.Filtering._with_grid_plan(plan::FSHFilterPlan, gp::FSHGridPlan) =
    _fsh_plan(plan.mult, gp, plan.invrenorm, plan.strategy, plan.scratch)
CGEF.Filtering._fresh(sc::FSHScratch) =
    FSHScratch(zero(sc.scratch), zero(sc.masked_input), CGEF.Filtering._fresh_buffer(sc.stage))
CGEF.Filtering._alike(a::S, b::S) where {S<:FSHScratch} = size(a.scratch) == size(b.scratch)
CGEF.Filtering._fresh_grid_plan(gp::FSHGridPlan{T,MK,B}) where {T,MK,B} = FSHGridPlan{T,MK,B}(
    _warmed_cache(gp.N, gp.M, gp.nthreads), gp.N, gp.M, gp.lmax, gp.mmax, gp.radius, gp.mask, gp.nthreads, gp.execution,
)

CGEF.Filtering.spectral_filter_plan(
    ::CGEF.SpectralBackends.AbstractFSHTSpectralBackend, grid::_CCGrid{T}, kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T; kwargs...,
) where {T<:AbstractFloat} = _fsh_filter_plan(grid, kernel, scale; kwargs...)

CGEF.Filtering.spectral_filter_plan(
    ::CGEF.SpectralBackends.AbstractFSHTSpectralBackend, grid::_SphericalGrid,
    ::CGEF.Kernels.AbstractFilterKernel, ::AbstractFloat; kwargs...,
) = _not_fsh_grid(grid)

CGEF.Filtering.spectral_filter_plan(
    auto::CGEF.SpectralBackends.AbstractAutoSpectralBackend, grid::_CCGrid{T},
    kernel::CGEF.Kernels.AbstractFilterKernel, scale::T; kwargs...,
) where {T<:AbstractFloat} = _fsh_shape(grid) ? _fsh_filter_plan(grid, kernel, scale; kwargs...) :
    CGEF.Filtering._node_set_filter_plan(auto, grid, kernel, scale; kwargs...)

# The multiplier and renormalization are `Float64`, as FSH's transforms are.
function _fsh_filter_plan(
    grid::_CCGrid{S},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::S;
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy = CGEF.Filtering.ZeroFill(),
    execution::CGEF.ComputationalBackends.AbstractExecutionBackend = CGEF.ComputationalBackends.AutoBackend(),
    batch::Union{Nothing,Integer} = nothing,
    grid_plan::Union{Nothing,FSHGridPlan} = nothing,
    scratch::Union{Nothing,FSHScratch} = nothing,
) where {S}
    T = Float64
    gp = grid_plan === nothing ? _fsh_grid_plan(grid; execution = execution) : grid_plan
    sc = scratch === nothing ? _fsh_scratch(gp) : scratch
    N, M = gp.N, gp.M
    mult = _fsh_multiplier(gp, kernel, scale, one(T))

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
    return _fsh_plan(mult, gp, invrenorm, mask_strategy, sc)
end

# `kernel`'s degree transfer on FSH's coefficient array, `unused` in the slots the packed layout leaves unused. The
# layout holds degrees up to `lmax + mmax` for high `|m|`, with the mode bounds of `FSH.sph_laplace!`.
function _fsh_multiplier(gp::FSHGridPlan, kernel::CGEF.Kernels.AbstractFilterKernel, scale::Real, unused::Float64)
    lmax, mmax = gp.lmax, gp.mmax
    lmax_full = lmax + mmax
    transfer_by_l = CGEF.Kernels.spectral_transfer_degrees!(zeros(Float64, lmax_full + 1), kernel, Float64(scale), gp.radius)
    mult = fill!(zeros(Float64, gp.N, gp.M), unused)
    for l in 0:lmax_full, m in (-l):l
        if l - lmax <= abs(m) <= mmax
            mult[FSH.sph_mode(l, m)] = transfer_by_l[l+1]
        end
    end
    return mult
end

CGEF.Filtering.rate_available(::FSHFilterPlan, kernel::CGEF.Kernels.AbstractFilterKernel) =
    CGEF.Kernels.has_degree_rate(kernel)
CGEF.Filtering.rate_plan(plan::FSHFilterPlan, kernel::CGEF.Kernels.AbstractFilterKernel, scale::AbstractFloat) =
    _fsh_plan(_fsh_multiplier(plan.grid_plan, CGEF.Kernels.TransferRate(kernel), scale, 0.0), plan.grid_plan,
              plan.invrenorm, plan.strategy, plan.scratch)

# The forward transform depends on the field alone, so a sweep shares one analysis across its scales.
CGEF.Filtering.analyze_buffer(plan::FSHFilterPlan, ::AbstractMatrix) = similar(plan.scratch.scratch)

# The field in host memory: itself for a host plan, copied into the stage buffer for a device plan.
@inline _host_field(::Nothing, field) = field
@inline _host_field(stage::Matrix, field) = copyto!(stage, field)
# Where a host routine writes an output bound for `out`, and the copy back into `out`.
@inline _host_out(::Nothing, out) = out
@inline _host_out(stage::Matrix, _) = stage
@inline _store!(::Nothing, out, _) = out
@inline _store!(::Matrix, out, host) = copyto!(out, host)

function _fsh_analyze!(Ĉ, field, plan::FSHFilterPlan)
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

function _fsh_synthesize!(out, Ĉ, plan::FSHFilterPlan)
    # `sph_evaluate!` works in place, and `Ĉ` is reused by every later scale, so evaluate a scaled copy.
    sc = plan.scratch
    sc.scratch .= Ĉ .* plan.mult
    _evaluate!(sc.scratch, plan.grid_plan)
    permutedims!(out, sc.scratch, (2, 1))
    plan.invrenorm === nothing || (out .*= plan.invrenorm)
    return out
end

# Host memory in and out; `out` may be `field`, which is read before `out` is written.
function _fsh_apply!(out, field, plan::FSHFilterPlan)
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

CGEF.Filtering.filter_analyze!(Ĉ::AbstractMatrix, field::AbstractMatrix{T}, plan::FSHFilterPlan{T}) where {T<:AbstractFloat} =
    _fsh_analyze!(Ĉ, _host_field(plan.scratch.stage, field), plan)

function CGEF.Filtering.filter_synthesize!(
    out::AbstractMatrix{T}, Ĉ::AbstractMatrix, plan::FSHFilterPlan{T},
) where {T<:AbstractFloat}
    stage = plan.scratch.stage
    return _store!(stage, out, _fsh_synthesize!(_host_out(stage, out), Ĉ, plan))
end

function CGEF.Filtering.filter_apply!(
    out::AbstractMatrix{T}, field::AbstractMatrix{T}, plan::FSHFilterPlan{T},
) where {T<:AbstractFloat}
    stage = plan.scratch.stage
    return _store!(stage, out, _fsh_apply!(_host_out(stage, out), _host_field(stage, field), plan))
end

CGEF.Filtering._batched_fields(outs, ::FSHFilterPlan) = ndims(first(outs)) == 3

function CGEF.Filtering.filter_apply_batched!(
    out::AbstractArray{T,3}, field::AbstractArray{T,3}, plan::FSHFilterPlan{T},
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
            _fsh_apply!(view(out, :, :, b), view(field, :, :, b), plan)
        else
            copyto!(stage, 1, field, (b - 1) * n + 1, n)
            _fsh_apply!(stage, stage, plan)
            copyto!(out, (b - 1) * n + 1, stage, 1, n)
        end
    end
    return out
end

# A trailing batch axis, slice by slice: FSH transforms `Matrix{Float64}` in place, so each slice's coefficients
# pass through the plan's scratch.
CGEF.Filtering.analyze_buffer(plan::FSHFilterPlan, field::AbstractArray{<:Any,3}) =
    zeros(Float64, plan.grid_plan.N, plan.grid_plan.M, size(field, 3))

function CGEF.Filtering.filter_analyze!(
    Ĉ::AbstractArray{Float64,3}, field::AbstractArray{T,3}, plan::FSHFilterPlan{T},
) where {T<:AbstractFloat}
    sc = plan.scratch.scratch
    for b in axes(field, 3)
        _fsh_analyze!(sc, _host_field(plan.scratch.stage, view(field, :, :, b)), plan)
        copyto!(view(Ĉ, :, :, b), sc)
    end
    return Ĉ
end

function CGEF.Filtering.filter_synthesize!(
    out::AbstractArray{T,3}, Ĉ::AbstractArray{Float64,3}, plan::FSHFilterPlan{T},
) where {T<:AbstractFloat}
    for b in axes(out, 3)
        CGEF.Filtering.filter_synthesize!(view(out, :, :, b), view(Ĉ, :, :, b), plan)
    end
    return out
end

# ---------------------------------------------------------------------------
# The surface gradient of a filtered field, from its series
# ---------------------------------------------------------------------------

# `ū = Σ Ĝ Ĉ Y`, and under `Deformable` that over the filtered mask.
CGEF.Filtering.series_form(plan::FSHFilterPlan) = CGEF.Filtering._ratio_form(plan.invrenorm)

"""
    FSHGradient

What the series gradient of an [`FSHFilterPlan`](@ref) reads: the spin-1 transform, the ladder `√(n(n+1))/R`
by degree, the filtered coefficients as real and as spin-0 ones, the host east and north components, and
under `Deformable` the mask's coefficients and the gradient of its filter. Built by `Filtering.series_gradient`.
"""
struct FSHGradient{TR, CM, DM, VF <: AbstractVector{Float64}, MF <: AbstractMatrix{Float64}, MC <: AbstractMatrix{ComplexF64} }
    tr::TR
    ladder::VF
    R̂::MF               # N × M
    C0::MC            # N × M, spin-0 coefficients, then the evaluated spin-1 field
    W::MC
    G::NTuple{2,MF}      # M × N east and north
    Cmask::CM                         # N × M, or nothing
    dM::DM                            # M × N pair, or nothing
end

function CGEF.Filtering.series_gradient(plan::FSHFilterPlan)
    gp = plan.grid_plan
    N, M = gp.N, gp.M
    pair() = (zeros(M, N), zeros(M, N))
    Cmask = _mask_coefficients(plan.invrenorm, plan)
    return FSHGradient(
        _fsh_spin1(N, M, gp.nthreads), [sqrt(Float64(n * (n + 1))) / gp.radius for n in 0:(2N - 2)],
        zeros(N, M), zeros(ComplexF64, N, M), zeros(ComplexF64, N, M), pair(), Cmask,
        Cmask === nothing ? nothing : pair(),
    )
end

# The spin-1 synthesis' cache, filled when built as `FSHGridPlan`'s is.
function _fsh_spin1(N::Int, M::Int, nthreads::Int)
    cache = FSH.SpinSphPlanCache{ComplexF64}()
    F = zeros(ComplexF64, N, M)
    FFTW.set_num_threads(nthreads) do
        FTB.with_fasttransforms_threads(() -> FSH.spinsph_evaluate!(F, 1; cache = cache), nthreads)
    end
    return FSHSpinTransforms(cache, N, M, nthreads)
end

_mask_coefficients(::Nothing, _) = nothing
function _mask_coefficients(::AbstractMatrix, plan::FSHFilterPlan)
    gp, sc = plan.grid_plan, plan.scratch
    C = zeros(gp.N, gp.M)
    sc.masked_input .= gp.mask
    permutedims!(C, sc.masked_input, (2, 1))
    _transform!(C, gp)
    return C
end

# `ð` of the filtered field whose real coefficients are `Ĉ`, east and north into `G`: `Ĉ` times the plan's
# `Ĝ` as spin-0 coefficients by FSH's `coeff_real2complex` rule, `s_{−m} = (C_m + iC_{−m})/√2` and
# `s_m = conj(s_{−m})`, the ladder and a spin-1 synthesis, `ðū = ∂_n ū − i∂_e ū`.
function _fsh_gradient!(G::NTuple{2,Matrix{Float64}}, g::FSHGradient, Ĉ, plan::FSHFilterPlan)
    R̂, C0 = g.R̂, g.C0
    R̂ .= Ĉ .* plan.mult
    r = 1 / sqrt(2.0)
    @inbounds for col in axes(C0, 2), row in axes(C0, 1)
        if col == 1
            C0[row, 1] = R̂[row, 1]
        elseif iseven(col)
            z = r * complex(R̂[row, col + 1], R̂[row, col])
            C0[row, col] = z
            C0[row, col + 1] = conj(z)
        end
    end
    F = CGEF.Filtering.spin_ladder_synthesize!(g.W, C0, g.W, g.ladder, g.tr, 0, 1)
    @inbounds for i in axes(F, 2), j in axes(F, 1)
        G[1][i, j] = -imag(F[j, i])
        G[2][i, j] = real(F[j, i])
    end
    return G
end

# `∇ū = (∇N − ū∇M)/M` under `Deformable`, `1/M` the plan's `invrenorm`.
_ratio_gradient!(G, _g, _ū, _plan, ::Nothing) = G
function _ratio_gradient!(G, g::FSHGradient, ū, plan::FSHFilterPlan, ir::AbstractMatrix)
    dM = _fsh_gradient!(g.dM, g, g.Cmask, plan)
    u = _host_field(plan.scratch.stage, ū)
    @. G[1] = ir * (G[1] - u * dM[1])
    @. G[2] = ir * (G[2] - u * dM[2])
    return G
end

function CGEF.Filtering.filter_synthesize_gradient!(
    g1::AbstractMatrix, g2::AbstractMatrix, Ĉ::AbstractMatrix, ū::AbstractMatrix, plan::FSHFilterPlan,
    g::FSHGradient,
)
    G = _ratio_gradient!(_fsh_gradient!(g.G, g, Ĉ, plan), g, ū, plan, plan.invrenorm)
    copyto!(g1, G[1])
    copyto!(g2, G[2])
    return g1, g2
end

function CGEF.Filtering.filter_synthesize_gradient!(
    g1::AbstractArray{<:Any,3}, g2::AbstractArray{<:Any,3}, Ĉ::AbstractArray{<:Any,3}, ū::AbstractArray{<:Any,3},
    plan::FSHFilterPlan, g::FSHGradient,
)
    for b in axes(g1, 3)
        CGEF.Filtering.filter_synthesize_gradient!(view(g1, :, :, b), view(g2, :, :, b), view(Ĉ, :, :, b),
                                                   view(ū, :, :, b), plan, g)
    end
    return g1, g2
end

# ---------------------------------------------------------------------------
# Spin-weighted transforms of the generalized convolution on FSH's grids
# ---------------------------------------------------------------------------

"""
    FSHSpinTransforms

The spin-0, 1 and 2 transforms of `Filtering.GeneralizedConvolution` on a grid FSH samples: the
`SpinSphPlanCache` holding every spin's FFT plans, the `N × M` layout, and the OpenMP thread count the
transforms run at. Spin `s` stores degree `r − 1 + max(|s|, ⌊c/2⌋)` in slot `(r, c)`, degrees `0:2N−2`
in all. The cache is filled when the transforms are built, as [`FSHGridPlan`](@ref)'s is, and a transform
writes its scratch: a concurrent driver needs its own.
"""
struct FSHSpinTransforms{C<:FSH.SpinSphPlanCache} <: CGEF.Filtering.AbstractSpinTransforms
    cache::C
    N::Int
    M::Int
    nthreads::Int
end

Base.show(io::IO, tr::FSHSpinTransforms) = print(io, "FSHSpinTransforms(", tr.N, " × ", tr.M, ")")
Base.show(io::IO, ::MIME"text/plain", tr::FSHSpinTransforms) = show(io, tr)

function _fsh_spin_transforms(grid::_CCGrid, lmax, nthreads::Integer)
    M, N = size(FlowGeometries.Grids.mask(grid))
    _fsh_shape(grid) || _not_fsh_grid(grid)
    lmax === nothing || lmax == N - 1 || throw(ArgumentError(
        "FastSphericalHarmonics transforms at its grid's band limit, lmax = $(N - 1); got lmax = $lmax. " *
        "Pass `GeneralizedConvolution()`, or transform over the grid's cells with " *
        "`transform = AutoSpectralBackend()`, which takes NUFSHT where it is loaded and the direct sum " *
        "otherwise.",
    ))
    return _warmed_spin(N, M, Int(nthreads))
end

# Every spin's plans, built by one transform and one evaluation each of a zero field.
function _warmed_spin(N::Int, M::Int, nth::Int)
    cache = FSH.SpinSphPlanCache{ComplexF64}()
    F = zeros(ComplexF64, N, M)
    FFTW.set_num_threads(nth) do
        FTB.with_fasttransforms_threads(nth) do
            for s in 0:2
                FSH.spinsph_transform!(F, s; cache = cache)
                FSH.spinsph_evaluate!(F, s; cache = cache)
            end
        end
    end
    return FSHSpinTransforms(cache, N, M, nth)
end

# A worker's transforms: a plan cache of its own, since a transform writes its cache's scratch.
CGEF.Filtering._worker_transforms(tr::FSHSpinTransforms) = _warmed_spin(tr.N, tr.M, tr.nthreads)

CGEF.Filtering.spin_transforms(
    ::CGEF.SpectralBackends.AbstractFSHTSpectralBackend, grid::_CCGrid;
    lmax = nothing, nthreads::Integer = 1, nufft = nothing, execution = nothing,
) = _fsh_spin_transforms(grid, lmax, nthreads)

CGEF.Filtering.spin_transforms(
    ::CGEF.SpectralBackends.AbstractFSHTSpectralBackend, grid::_SphericalGrid; kwargs...,
) = _not_fsh_grid(grid)

CGEF.Filtering.spin_transforms(
    auto::CGEF.SpectralBackends.AbstractAutoSpectralBackend, grid::_CCGrid;
    lmax = nothing, nthreads::Integer = 1,
    nufft::CGEF.SpectralBackends.AbstractSpectralBackend = CGEF.SpectralBackends.AutoSpectralBackend(),
    execution::CGEF.ComputationalBackends.AbstractLocalBackend = CGEF.ComputationalBackends.SerialBackend(),
) = _fsh_shape(grid) && lmax === nothing ? _fsh_spin_transforms(grid, lmax, nthreads) :
    CGEF.Filtering._scattered_spin_transforms(auto, grid; lmax, nthreads, nufft, execution)

CGEF.Filtering.spin_field(tr::FSHSpinTransforms) = zeros(ComplexF64, tr.N, tr.M)
CGEF.Filtering.spin_field_size(tr::FSHSpinTransforms) = (tr.N, tr.M)
CGEF.Filtering.spin_position(tr::FSHSpinTransforms) = CGEF.Filtering.TransposedCells(tr.M, tr.N)
CGEF.Filtering.spin_coefficients(tr::FSHSpinTransforms) = zeros(ComplexF64, tr.N, tr.M)
CGEF.Filtering.spin_transfer_buffer(tr::FSHSpinTransforms) = zeros(Float64, 2tr.N - 1)

function CGEF.Filtering.spin_analyze!(C, F, tr::FSHSpinTransforms, s::Int)
    copyto!(C, F)
    FTB.with_fasttransforms_threads(() -> FSH.spinsph_transform!(C, s; cache = tr.cache), tr.nthreads)
    return C
end

function CGEF.Filtering.spin_synthesize!(F, C, W, Ĝ, tr::FSHSpinTransforms, s::Int)
    a = abs(s)
    @inbounds for c in axes(C, 2), r in axes(C, 1)
        W[r, c] = C[r, c] * Ĝ[r + max(a, c ÷ 2)]
    end
    FTB.with_fasttransforms_threads(() -> FSH.spinsph_evaluate!(W, s; cache = tr.cache), tr.nthreads)
    copyto!(F, W)
    return F
end

# The transform is exact on its grid, so no analysis fits.
CGEF.Filtering.spin_fit(::CGEF.Filtering.AbstractAnalysis, ::FlowGeometries.Grids.AbstractGrid, ::FSHSpinTransforms) =
    nothing

# A coefficient of degree `n` and order `m` sits at row `n − max(|s|, |m|) + 1` of column `2|m| + (m ≥ 0)`, so a
# step from spin `s` to `t` moves it to its row at spin `t`. In these harmonics a step of `ð` or `ð̄` between
# non-negative spins takes the standard ladder factor's negative for `m < −min(s, t)`.
function CGEF.Filtering.spin_ladder_synthesize!(F, C, W, d, tr::FSHSpinTransforms, s::Int, t::Int)
    (s >= 0 && t >= 0) || throw(ArgumentError("the FastSphericalHarmonics ladder takes non-negative spins; got $s → $t"))
    lo = min(s, t)
    fill!(W, zero(eltype(W)))
    @inbounds for col in axes(C, 2)
        am = col ÷ 2
        m = col == 1 ? 0 : iseven(col) ? -am : am
        sg = m < -lo ? -1 : 1
        for rt in axes(C, 1)
            n = rt - 1 + max(t, am)
            rs = n - max(s, am) + 1
            1 <= rs <= size(C, 1) || continue
            W[rt, col] = sg * d[n + 1] * C[rs, col]
        end
    end
    FTB.with_fasttransforms_threads(() -> FSH.spinsph_evaluate!(W, t; cache = tr.cache), tr.nthreads)
    copyto!(F, W)
    return F
end

end # module
