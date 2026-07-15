module CoarseGrainingEnergyFluxesFINUFFTExt

using FINUFFT: FINUFFT
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF

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
# Normalizing by the point count preserves the domain mean (Ĝ(0)=1 ⇒ ḡ ≡ c̄ for a constant field)
# for any quasi-uniform sampling, and reduces EXACTLY to the FFTW result when the points are a uniform
# periodic lattice. Spectral filtering implicitly assumes periodicity; the per-axis period is taken as
# the sample extent plus one grid spacing `geometry.dx`/`dy` (the periodic-box convention — exact for a
# uniform lattice). Highly non-uniform sampling is an ill-conditioned inverse problem; results there are
# approximate (see NUFSHT's scattered-point conditioning for the spherical analogue). Land masks are
# not applied (like FFTW); use `method = DirectSum()` on a StructuredGrid for masked/regional fields.

"""
    FINUFFTFilterPlan

Cached scattered-data spectral filter plan: the sample points rescaled to `[0, 2π)` per axis, the
precomputed transfer-function array on the `M × N` Fourier modes, and a PERSISTENT pair of FINUFFT
guru plans (type-1 points→modes, type-2 modes→points) with `finufft_setpts!` already called — the
expensive setup (internal point sort, spreader tables, FFTW planning) happens ONCE here, not on every
`filter_apply!` call. Built by `plan_filter(unstructured_grid, kernel, scale; method = Spectral())`.
"""
struct FINUFFTFilterPlan{
    XT <: AbstractVector, YT<: AbstractVector, T<:AbstractFloat,
    A<:AbstractMatrix{T}, FA<:AbstractMatrix{Complex{T}},
} <: CGEF.Filtering.AbstractFilterPlan
    X::XT      # x points scaled to [0, 2π)
    Y::YT   # y points scaled to [0, 2π)
    transfer::A       # Ĝ(|k|, ℓ) on the M × N CMCL-ordered mode grid
    M::Int
    N::Int
    npts::Int
    plan1::FINUFFT.finufft_plan{T}   # type-1 guru plan: points → modes (isign -1), setpts! already done
    plan2::FINUFFT.finufft_plan{T}   # type-2 guru plan: modes → points (isign +1), setpts! already done
    c_scratch::Vector{Complex{T}}    # length-npts scratch: type-1 input, then reused as type-2 output
    F_scratch::FA                     # M×N scratch: type-1 output / (Ĝ·F̂) / type-2 input
end

function CGEF.Filtering.spectral_filter_plan(
    grid::CGEF.UnstructuredGrid{T,G},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy = CGEF.Filtering.Deformable(),
    backend = CGEF.Backends.AutoBackend(),
) where {T<:AbstractFloat, G<:CGEF.CartesianGeometry{T}}
    x = grid.lon
    y = grid.lat
    npts = length(x)
    npts > 0 || throw(ArgumentError("FINUFFT spectral filtering needs at least one point."))
    all(grid.mask) || throw(ArgumentError(
        "FINUFFT spectral filtering does not support a partial mask; use method = DirectSum() on a StructuredGrid.",
    ))
    xmin, xmax = extrema(x)
    ymin, ymax = extrema(y)
    dxext = xmax - xmin
    dyext = ymax - ymin

    # Mode count: for a roughly-uniform scattered point cloud, choose Mx*My ~ npts (matching the
    # actual information content of the data, split by the RAW-extent aspect ratio — the periodic
    # pad below is a small, second-order correction that doesn't materially change this) — NOT via
    # geometry.dx/dy, which is meaningless for scattered data (a genuinely SCATTERED
    # `UnstructuredGrid` has no fixed grid spacing) and can be wrong by orders of magnitude relative
    # to the real point spacing, silently blowing the mode count (and FINUFFT's internal oversampled
    # FFT grid) up to gigabytes for a handful of points.
    aspect0 = dyext > 0 ? dxext / dyext : one(T)
    My_est = sqrt(T(npts) / aspect0)
    Mx_est = T(npts) / My_est
    M = max(2, round(Int, Mx_est)); iseven(M) || (M += 1)
    N = max(2, round(Int, My_est)); iseven(N) || (N += 1)

    # Periodic-box period: sample extent, padded by the per-axis spacing IMPLIED by the FINAL,
    # ROUNDED mode count M/N (extent / (M - 1)) — NOT the unrounded Mx_est/My_est estimate. For a
    # uniform lattice this reduces exactly to the true physical spacing (e.g. an 8-point, 1000 m
    # axis gives M=8, pad=1000 m, Lx=8000 m, the correct periodic length); using the unrounded
    # estimate instead introduces a small but real inconsistency between the mode grid actually built
    # (M points) and the assumed period, breaking bit-level exactness even in the well-resolved case
    # where M/N alone would already be exactly right (caught by the "Spectral FINUFFT filtering"
    # cross-check against the FFTW reference at atol=1e-7 — a coarser rtol wouldn't have caught this).
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

    # Persistent guru plans: `finufft_setpts!` does the expensive one-time setup (internal point
    # sort, spreader tables, FFTW planning) HERE, once, instead of on every `filter_apply!` call —
    # the old code called the one-shot `nufft2d1`/`nufft2d2` convenience wrappers, which silently
    # redo all of this from scratch every time (~9 times per `compute_Π!`, and again per scale in a
    # `coarse_grain` sweep). X/Y pass straight through, whatever concrete type the broadcast above
    # produced — no conversion here.
    plan1 = FINUFFT.finufft_makeplan(1, [M, N], -1, 1, ϵ; dtype = T)
    plan2 = FINUFFT.finufft_makeplan(2, [M, N], 1, 1, ϵ; dtype = T)
    FINUFFT.finufft_setpts!(plan1, X, Y)
    FINUFFT.finufft_setpts!(plan2, X, Y)
    finalizer(FINUFFT.finufft_destroy!, plan1)
    finalizer(FINUFFT.finufft_destroy!, plan2)

    c_scratch = zeros(Complex{T}, npts)
    F_scratch = zeros(Complex{T}, M, N)

    return FINUFFTFilterPlan(X, Y, transfer, M, N, npts, plan1, plan2, c_scratch, F_scratch)
end

function CGEF.Filtering.filter_apply!(
    out::AbstractVector{T},
    field::AbstractVector,
    plan::FINUFFTFilterPlan{XT, YT, T},
) where {XT, YT, T<:AbstractFloat}
    @. plan.c_scratch = Complex{T}(field)
    FINUFFT.finufft_exec!(plan.plan1, plan.c_scratch, plan.F_scratch)   # pts → modes
    plan.F_scratch .*= plan.transfer                                     # Ĝ · F̂
    FINUFFT.finufft_exec!(plan.plan2, plan.F_scratch, plan.c_scratch)   # modes → pts (reuses c_scratch)
    @. out = real(plan.c_scratch) / plan.npts
    return out
end

end # module
