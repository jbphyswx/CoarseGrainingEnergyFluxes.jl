module CoarseGrainingEnergyFluxesFINUFFTExt

using FINUFFT: FINUFFT
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF

# Spectral filtering for SCATTERED / non-uniform Cartesian data (an `UnstructuredGrid{Cartesian}`),
# via the non-uniform FFT. Identical structure to the FFTW backend — forward transform, multiply by
# the shared transfer function `Ĝ(|k|, ℓ)` (`CGEF.spectral_transfer`), inverse transform — but the
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
precomputed transfer-function array on the `M × N` Fourier modes, the NUFFT tolerance, and the point
count used for normalization. Built by `plan_filter(unstructured_grid, kernel, scale; method = Spectral())`.
"""
struct FINUFFTFilterPlan{T<:AbstractFloat, A<:AbstractMatrix{T}} <: CGEF.Filtering.AbstractFilterPlan
    X::Vector{T}      # x points scaled to [0, 2π)
    Y::Vector{T}      # y points scaled to [0, 2π)
    transfer::A       # Ĝ(|k|, ℓ) on the M × N CMCL-ordered mode grid
    M::Int
    N::Int
    ϵ::T              # NUFFT tolerance
    npts::Int
end

function CGEF.Filtering.spectral_filter_plan(
    grid::CGEF.UnstructuredGrid{G,T},
    kernel::CGEF.AbstractFilterKernel,
    scale::T;
    mask_strategy = CGEF.Deformable(),
    backend = CGEF.AutoBackend(),
) where {T<:AbstractFloat, G<:CGEF.CartesianGeometry{T}}
    x = grid.lon
    y = grid.lat
    npts = length(x)
    npts > 0 || throw(ArgumentError("FINUFFT spectral filtering needs at least one point."))
    dx = grid.geometry.dx
    dy = grid.geometry.dy
    xmin, xmax = extrema(x)
    ymin, ymax = extrema(y)
    # Periodic-box period: sample extent + one spacing (exact for a uniform periodic lattice).
    Lx = (xmax - xmin) + dx
    Ly = (ymax - ymin) + dy
    X = T(2π) .* (x .- xmin) ./ Lx
    Y = T(2π) .* (y .- ymin) ./ Ly

    # Mode count = number of grid cells spanning the period (= Nx, Ny for a uniform lattice). Even.
    M = max(2, round(Int, Lx / dx)); iseven(M) || (M += 1)
    N = max(2, round(Int, Ly / dy)); iseven(N) || (N += 1)
    nx = (-(M ÷ 2)):(M ÷ 2 - 1)
    ny = (-(N ÷ 2)):(N ÷ 2 - 1)
    transfer = T[
        CGEF.spectral_transfer(kernel, sqrt((T(2π) * ix / Lx)^2 + (T(2π) * iy / Ly)^2), scale)
        for ix in nx, iy in ny
    ]

    ϵ = max(T(1e-9), eps(T) * 10)
    return FINUFFTFilterPlan{T, typeof(transfer)}(collect(T, X), collect(T, Y), transfer, M, N, ϵ, npts)
end

function CGEF.Filtering.filter_apply!(
    out::AbstractVector{T},
    field::AbstractVector,
    plan::FINUFFTFilterPlan{T},
) where {T<:AbstractFloat}
    c = Complex{T}.(field)
    F = FINUFFT.nufft2d1(plan.X, plan.Y, c, -1, plan.ϵ, plan.M, plan.N)  # pts → modes
    F .*= plan.transfer                                                   # Ĝ · F̂
    g = FINUFFT.nufft2d2(plan.X, plan.Y, +1, plan.ϵ, F)                   # modes → pts
    @. out = real(g) / plan.npts
    return out
end

end # module
