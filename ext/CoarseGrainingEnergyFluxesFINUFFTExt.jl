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
    FINUFFTFilterPlan

Cached scattered-data spectral filter plan: the sample points rescaled to `[0, 2π)` per axis, the
precomputed transfer-function array on the `M × N` Fourier modes, a PERSISTENT pair of FINUFFT
guru plans (type-1 points→modes, type-2 modes→points) with `finufft_setpts!` already called — the
expensive setup (internal point sort, spreader tables, FFTW planning) happens ONCE here, not on every
`filter_apply!` call — and, for a masked grid, the mask itself, a scratch buffer for `mask · field`,
and (for `Deformable`) the precomputed inverse local-mass renormalization. Built by
`plan_filter(unstructured_grid, kernel, scale; method = Spectral())`.
"""
struct FINUFFTFilterPlan{
    XT <: AbstractVector, YT<: AbstractVector, T<:AbstractFloat,
    A<:AbstractMatrix{T}, FA<:AbstractMatrix{Complex{T}}, CV<:AbstractVector{Complex{T}},
    M, MV<:AbstractVector{T}, R,
} <: CGEF.Filtering.AbstractFilterPlan
    X::XT      # x points scaled to [0, 2π)
    Y::YT   # y points scaled to [0, 2π)
    transfer::A       # Ĝ(|k|, ℓ) on the M × N CMCL-ordered mode grid
    M::Int
    N::Int
    npts::Int
    plan1::FINUFFT.finufft_plan{T}   # type-1 guru plan: points → modes (isign -1), setpts! already done
    plan2::FINUFFT.finufft_plan{T}   # type-2 guru plan: modes → points (isign +1), setpts! already done
    c_scratch::CV                     # length-npts scratch: type-1 input, then reused as type-2 output
    F_scratch::FA                     # M×N scratch: type-1 output / (Ĝ·F̂) / type-2 input
    mask::M            # Vector{T} of 0/1, or nothing when fully active
    masked_input::MV   # scratch for `mask .* field`; unused when mask === nothing
    invrenorm::R        # precomputed 1/filter(mask) for Deformable, or nothing (ZeroFill / fully active)
end

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
    nx = (-(M ÷ 2)):(M ÷ 2 - 1)
    ny = (-(N ÷ 2)):(N ÷ 2 - 1)
    transfer = T[
        CGEF.Kernels.spectral_transfer(kernel, sqrt((T(2π) * ix / Lx)^2 + (T(2π) * iy / Ly)^2), scale)
        for ix in nx, iy in ny
    ]

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

    c_scratch = zeros(Complex{T}, npts)
    F_scratch = zeros(Complex{T}, M, N)
    masked_input = zeros(T, npts)

    fully_active = all(FlowGeometries.Grids.mask(grid))
    if fully_active
        return FINUFFTFilterPlan(
            X, Y, transfer, M, N, npts, plan1, plan2, c_scratch, F_scratch, nothing, masked_input, nothing,
        )
    end

    mask = T.(FlowGeometries.Grids.mask(grid))
    invrenorm = if mask_strategy isa CGEF.Filtering.Deformable
        # Local kernel mass over active points, `filter(mask)` — the SAME NUFFT pipeline, applied to
        # the mask itself once, since the mask never changes across `filter_apply!` calls.
        @. c_scratch = Complex{T}(mask)
        FINUFFT.finufft_exec!(plan1, c_scratch, F_scratch)
        F_scratch .*= transfer
        FINUFFT.finufft_exec!(plan2, F_scratch, c_scratch)
        renorm = real.(c_scratch) ./ npts
        threshold = T(0.01)
        ir = similar(renorm)
        @. ir = ifelse(abs(renorm) >= threshold, one(T) / renorm, zero(T))
        ir
    else
        nothing   # ZeroFill: already exactly `filter(mask .* field)`, no renormalization
    end
    return FINUFFTFilterPlan(
        X, Y, transfer, M, N, npts, plan1, plan2, c_scratch, F_scratch, mask, masked_input, invrenorm,
    )
end

# Analysis (pts → modes) depends on the field alone, so a sweep runs it once and each scale only applies
# its own transfer function and evaluates back to points.
CGEF.Filtering.analyze_buffer(plan::FINUFFTFilterPlan, ::AbstractVector) = similar(plan.F_scratch)

function CGEF.Filtering.filter_analyze!(
    F̂::AbstractArray, field::AbstractVector, plan::FINUFFTFilterPlan{XT, YT, T},
) where {XT, YT, T<:AbstractFloat}
    if plan.mask === nothing
        @. plan.c_scratch = Complex{T}(field)
    else
        @. plan.masked_input = plan.mask * field
        @. plan.c_scratch = Complex{T}(plan.masked_input)
    end
    FINUFFT.finufft_exec!(plan.plan1, plan.c_scratch, F̂)
    return F̂
end

function CGEF.Filtering.filter_synthesize!(
    out::AbstractVector{T}, F̂::AbstractArray, plan::FINUFFTFilterPlan{XT, YT, T},
) where {XT, YT, T<:AbstractFloat}
    # `finufft_exec!` consumes its input, and `F̂` is reused by every later scale, so the scaled copy goes
    # through the plan's own mode scratch.
    plan.F_scratch .= F̂ .* plan.transfer
    FINUFFT.finufft_exec!(plan.plan2, plan.F_scratch, plan.c_scratch)
    @. out = real(plan.c_scratch) / plan.npts
    plan.invrenorm === nothing || (out .*= plan.invrenorm)
    return out
end

function CGEF.Filtering.filter_apply!(
    out::AbstractVector{T},
    field::AbstractVector,
    plan::FINUFFTFilterPlan{XT, YT, T},
) where {XT, YT, T<:AbstractFloat}
    if plan.mask === nothing
        @. plan.c_scratch = Complex{T}(field)
    else
        @. plan.masked_input = plan.mask * field
        @. plan.c_scratch = Complex{T}(plan.masked_input)
    end
    FINUFFT.finufft_exec!(plan.plan1, plan.c_scratch, plan.F_scratch)   # pts → modes
    plan.F_scratch .*= plan.transfer                                     # Ĝ · F̂
    FINUFFT.finufft_exec!(plan.plan2, plan.F_scratch, plan.c_scratch)   # modes → pts (reuses c_scratch)
    @. out = real(plan.c_scratch) / plan.npts
    plan.invrenorm === nothing || (out .*= plan.invrenorm)
    return out
end

end # module
