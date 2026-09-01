module CoarseGrainingEnergyFluxesFINUFFTExt

using FINUFFT: FINUFFT
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF
using FlowGeometries: FlowGeometries

# Spectral filtering for SCATTERED / non-uniform Cartesian data (an `UnstructuredGrid{Cartesian}`),
# via the non-uniform FFT. Identical structure to the FFTW backend — forward transform, multiply by
# the shared transfer function `Ĝ(|k|, ℓ)` (`CGEF.Kernels.spectral_transfer`), inverse transform — but the
# transforms are type-1 / type-2 NUFFTs that map between the scattered sample points and a uniform
# Fourier mode grid:
#
#   type-1 (pts→modes, isign −1):  F_k = Σ_j c_j e^{-i k·x_j}
#   multiply:                      F_k ← Ĝ(|k|, ℓ) F_k
#   type-2 (modes→pts, isign +1):  g_j = Σ_k F_k e^{+i k·x_j}
#   normalize:                     ḡ_j = g_j / N_pts
#
# Normalizing by the point count preserves the domain mean (Ĝ(0)=1 ⇒ ḡ ≡ c̄ for a constant field) for
# any quasi-uniform sampling, and reduces exactly to the FFTW result on a uniform periodic lattice.
# Spectral filtering assumes periodicity; the per-axis period is derived below from the sample extent
# and the mode count. Highly non-uniform sampling is an ill-conditioned inverse problem, so results
# there are approximate.
#
# Masking: same normalized-convolution identity as FFTW (Knutsson & Westin 1993), applied over the
# scattered points instead of a dense grid — `ZeroFill` filters `mask·field` directly (no
# renormalization); `Deformable` additionally divides by the LOCAL kernel mass over active points,
# `filter(mask)`, run through the SAME type-1/transfer/type-2 pipeline as any other point-indexed
# field and computed ONCE here at plan-build time (not per `filter_apply!` call).

"""
    FINUFFTGridPlan

The half of a FINUFFT plan the filter scale does not reach: a PERSISTENT pair of guru plans (type-1
points→modes, type-2 modes→points) with `finufft_setpts!` already called, the rescaled point
coordinates, the mode counts and box periods, and the mask.

Only `transfer` — and, for `Deformable`, the renormalization computed through it — depends on ℓ.
Sharing the rest matters most here of all the spectral backends, because `finufft_setpts!` sorts every
point: doing that once per scale means re-sorting the same cloud `S` times.

A guru plan holds the working state of its own execution, so two tasks may not execute one
concurrently; a concurrent driver needs its own grid plan per worker.
"""
struct FINUFFTGridPlan{T<:AbstractFloat, VT<:AbstractVector{T}, P1, P2, MK} <: CGEF.Filtering.AbstractGridPlan
    X::VT      # x points scaled to [0, 2π)
    Y::VT      # y points scaled to [0, 2π)
    M::Int
    N::Int
    npts::Int
    Lx::T
    Ly::T
    plan1::P1
    plan2::P2
    mask::MK   # Vector{T} of 0/1, or nothing when fully active
end

"""
    FINUFFTScratch

The transient half of a FINUFFT plan: the length-`npts` coefficient buffer (type-1 input, then type-2
output), the `M × N` mode buffer, and the `mask · field` staging vector. One per concurrent worker.
"""
struct FINUFFTScratch{
    T<:AbstractFloat, CV<:AbstractVector{Complex{T}}, CM<:AbstractMatrix{Complex{T}},
    MV<:AbstractVector{T},
} <: CGEF.Filtering.AbstractFilterScratch
    c_scratch::CV
    F_scratch::CM
    masked_input::MV   # unused when mask === nothing
end

"""
    FINUFFTFilterPlan

Cached scattered-data spectral filter plan: the shared [`FINUFFTGridPlan`](@ref), the precomputed
transfer-function array on the `M × N` Fourier modes, the `Deformable` inverse local-mass
renormalization (or `nothing`), and the [`FINUFFTScratch`](@ref) the apply writes through. Built by
`plan_filter(unstructured_grid, kernel, scale; method = Spectral())`.
"""
struct FINUFFTFilterPlan{
    T<:AbstractFloat, GP<:FINUFFTGridPlan{T}, A<:AbstractMatrix{T}, R, SC<:FINUFFTScratch{T},
} <: CGEF.Filtering.AbstractFilterPlan
    grid_plan::GP
    transfer::A    # Ĝ(|k|, ℓ) on the M × N CMCL-ordered mode grid
    invrenorm::R   # precomputed 1/filter(mask) for Deformable, or nothing (ZeroFill / fully active)
    scratch::SC
end

function _finufft_grid_plan(
    grid::FlowGeometries.Grids.UnstructuredGrid{T,G}; finufft_nthreads::Integer = 1,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    nth = Int(finufft_nthreads)
    nth >= 0 || throw(ArgumentError(
        "finufft_nthreads must be >= 0 (0 means FINUFFT's own default of all threads), got $nth",
    ))
    x = FlowGeometries.Grids.coordinates(grid, 1)
    y = FlowGeometries.Grids.coordinates(grid, 2)
    npts = length(x)
    npts > 0 || throw(ArgumentError("FINUFFT spectral filtering needs at least one point."))
    xmin, xmax = extrema(x)
    ymin, ymax = extrema(y)
    dxext = xmax - xmin
    dyext = ymax - ymin

    # Mode count from the POINT COUNT: `Mx*My ~ npts`, the information content of the data, split by
    # the raw-extent aspect ratio. Scattered data has no grid spacing to divide an extent by, and a
    # mode count derived from one is unbounded in the problem size.
    aspect0 = dyext > 0 ? dxext / dyext : one(T)
    My_est = sqrt(T(npts) / aspect0)
    Mx_est = T(npts) / My_est
    M = max(2, round(Int, Mx_est)); iseven(M) || (M += 1)
    N = max(2, round(Int, My_est)); iseven(N) || (N += 1)

    # Periodic-box period: the sample extent padded by the spacing implied by the FINAL, rounded mode
    # count, `extent / (M - 1)`. It must be the rounded count, not the estimate, or the assumed period
    # disagrees with the mode grid actually built. On a uniform lattice this recovers the true spacing
    # exactly — an 8-point, 1000 m axis gives M=8, pad=1000 m, Lx=8000 m.
    dx_nom = M > 1 ? dxext / (M - 1) : one(T)
    dy_nom = N > 1 ? dyext / (N - 1) : one(T)
    Lx = dxext + dx_nom
    Ly = dyext + dy_nom
    X = T(2π) .* (x .- xmin) ./ Lx
    Y = T(2π) .* (y .- ymin) ./ Ly
    ϵ = max(T(1e-9), eps(T) * 10)

    # Persistent guru plans: `finufft_setpts!` does the point sort, spreader tables and FFTW planning
    # once here, rather than per `filter_apply!` — `compute_Π!` makes ~9 of those per scale. The
    # one-shot `nufft2d1`/`nufft2d2` wrappers redo all of it on every call.
    #
    # `ntrans = 1`: the batch axis lives ABOVE this, in the slice/batch drivers, not in FINUFFT.
    # `nthreads` is therefore FINUFFT's INTERNAL parallelism (spreader + its own FFTW plan) — see the
    # `finufft_nthreads` note on this method for why it defaults to 1.
    plan1 = FINUFFT.finufft_makeplan(1, [M, N], -1, 1, ϵ; dtype = T, nthreads = nth)
    plan2 = FINUFFT.finufft_makeplan(2, [M, N], 1, 1, ϵ; dtype = T, nthreads = nth)
    FINUFFT.finufft_setpts!(plan1, X, Y)
    FINUFFT.finufft_setpts!(plan2, X, Y)
    finalizer(FINUFFT.finufft_destroy!, plan1)
    finalizer(FINUFFT.finufft_destroy!, plan2)

    mask = all(FlowGeometries.Grids.mask(grid)) ? nothing : T.(FlowGeometries.Grids.mask(grid))

    return FINUFFTGridPlan(X, Y, M, N, npts, Lx, Ly, plan1, plan2, mask)
end

_finufft_scratch(gp::FINUFFTGridPlan{T}) where {T<:AbstractFloat} = FINUFFTScratch(
    zeros(Complex{T}, gp.npts), zeros(Complex{T}, gp.M, gp.N), zeros(T, gp.npts),
)

CGEF.Filtering.spectral_scratch(gp::FINUFFTGridPlan) = _finufft_scratch(gp)

CGEF.Filtering.spectral_grid_plan(
    ::Union{CGEF.SpectralBackends.AbstractAutoSpectralBackend, CGEF.SpectralBackends.AbstractNUFFTSpectralBackend},
    grid::FlowGeometries.Grids.UnstructuredGrid{T,G},
    kernel::CGEF.Kernels.AbstractFilterKernel;
    finufft_nthreads::Integer = 1,
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}} =
    _finufft_grid_plan(grid; finufft_nthreads = finufft_nthreads)

"""
    spectral_filter_plan(...; finufft_nthreads = 1)

`finufft_nthreads` is FINUFFT's INTERNAL thread count — its spreader and its own FFTW plan. Not the
batch axis: that is `ntrans`, which is 1 here, with batching in the slice/batch drivers.

It defaults to 1 because FFTW's thread count is process-global and other packages raise it (loading
FastSphericalHarmonics does, via FastTransforms). An unpinned plan then builds a multi-threaded
internal FFTW plan, and FFTW.jl's threading provider spawns a Julia `Task` per work chunk on every
`finufft_execute` — measured 216 tasks and ~120 kB per execution on an 18×18 mode grid, ~1.2 MB per
`compute_Π!`, which breaks the zero-allocation-on-reuse contract.

Measured, one type-1 transform (2 Julia threads, FFTW at 4):

| points | modes | `nthreads = 1` | FINUFFT default |
|---|---|---|---|
| 300 | 18² | 0.031 ms, 0 B | 3.416 ms, 119 808 B |
| 10 000 | 100² | 1.207 ms, 0 B | 1.061 ms, 4 032 B |
| 200 000 | 448² | 67.5 ms, 0 B | 40.7 ms, 4 032 B |
| 1 000 000 | 1000² | 373.6 ms, 0 B | 329.7 ms, 6 336 B |

Pinning is 110× faster at 300 points and 1.66× slower at 200 000. Pass `finufft_nthreads = 0` (all
threads) for a large single transform where that is worth ~4 kB per apply; do not combine it with a
threaded slice/batch loop, which would nest two levels of threading.
"""
function CGEF.Filtering.spectral_filter_plan(
    ::Union{CGEF.SpectralBackends.AbstractAutoSpectralBackend, CGEF.SpectralBackends.AbstractNUFFTSpectralBackend},
    grid::FlowGeometries.Grids.UnstructuredGrid{T,G},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy = CGEF.Filtering.ZeroFill(),
    backend = CGEF.ComputationalBackends.AutoBackend(),
    finufft_nthreads::Integer = 1,
    grid_plan::Union{Nothing,FINUFFTGridPlan} = nothing,
    scratch::Union{Nothing,FINUFFTScratch} = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    gp = grid_plan === nothing ?
        _finufft_grid_plan(grid; finufft_nthreads = finufft_nthreads) : grid_plan
    sc = scratch === nothing ? _finufft_scratch(gp) : scratch
    M, N, npts = gp.M, gp.N, gp.npts
    Lx, Ly = gp.Lx, gp.Ly

    nx = (-(M ÷ 2)):(M ÷ 2 - 1)
    ny = (-(N ÷ 2)):(N ÷ 2 - 1)
    transfer = T[
        CGEF.Kernels.spectral_transfer(kernel, sqrt((T(2π) * ix / Lx)^2 + (T(2π) * iy / Ly)^2), scale)
        for ix in nx, iy in ny
    ]

    mask = gp.mask
    invrenorm = if mask !== nothing && mask_strategy isa CGEF.Filtering.Deformable
        # Local kernel mass over active points, `filter(mask)`, through the plan's own NUFFT pipeline.
        # The mask is fixed for the plan, so this is built once and stored inverted.
        @. sc.c_scratch = Complex{T}(mask)
        FINUFFT.finufft_exec!(gp.plan1, sc.c_scratch, sc.F_scratch)
        sc.F_scratch .*= transfer
        FINUFFT.finufft_exec!(gp.plan2, sc.F_scratch, sc.c_scratch)
        renorm = real.(sc.c_scratch) ./ npts
        threshold = T(0.01)
        ir = similar(renorm)
        @. ir = ifelse(abs(renorm) >= threshold, one(T) / renorm, zero(T))
        ir
    else
        nothing   # ZeroFill: already exactly `filter(mask .* field)`, no renormalization
    end
    return FINUFFTFilterPlan(gp, transfer, invrenorm, sc)
end

# Analysis (pts → modes) depends on the field alone, so a sweep runs it once and each scale only applies
# its own transfer function and evaluates back to points.
CGEF.Filtering.analyze_buffer(plan::FINUFFTFilterPlan, ::AbstractVector) = similar(plan.scratch.F_scratch)

function CGEF.Filtering.filter_analyze!(
    F̂::AbstractArray, field::AbstractVector, plan::FINUFFTFilterPlan{T},
) where {T<:AbstractFloat}
    gp, sc = plan.grid_plan, plan.scratch
    if gp.mask === nothing
        @. sc.c_scratch = Complex{T}(field)
    else
        @. sc.masked_input = gp.mask * field
        @. sc.c_scratch = Complex{T}(sc.masked_input)
    end
    FINUFFT.finufft_exec!(gp.plan1, sc.c_scratch, F̂)
    return F̂
end

function CGEF.Filtering.filter_synthesize!(
    out::AbstractVector{T}, F̂::AbstractArray, plan::FINUFFTFilterPlan{T},
) where {T<:AbstractFloat}
    # `finufft_exec!` consumes its input, and `F̂` is reused by every later scale, so the scaled copy goes
    # through the plan's own mode scratch.
    gp, sc = plan.grid_plan, plan.scratch
    sc.F_scratch .= F̂ .* plan.transfer
    FINUFFT.finufft_exec!(gp.plan2, sc.F_scratch, sc.c_scratch)
    @. out = real(sc.c_scratch) / gp.npts
    plan.invrenorm === nothing || (out .*= plan.invrenorm)
    return out
end

function CGEF.Filtering.filter_apply!(
    out::AbstractVector{T},
    field::AbstractVector,
    plan::FINUFFTFilterPlan{T},
) where {T<:AbstractFloat}
    gp, sc = plan.grid_plan, plan.scratch
    if gp.mask === nothing
        @. sc.c_scratch = Complex{T}(field)
    else
        @. sc.masked_input = gp.mask * field
        @. sc.c_scratch = Complex{T}(sc.masked_input)
    end
    FINUFFT.finufft_exec!(gp.plan1, sc.c_scratch, sc.F_scratch)   # pts → modes
    sc.F_scratch .*= plan.transfer                                # Ĝ · F̂
    FINUFFT.finufft_exec!(gp.plan2, sc.F_scratch, sc.c_scratch)   # modes → pts (reuses c_scratch)
    @. out = real(sc.c_scratch) / gp.npts
    plan.invrenorm === nothing || (out .*= plan.invrenorm)
    return out
end

end # module
