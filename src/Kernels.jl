module Kernels

export AbstractFilterKernel, TopHatKernel, GaussianKernel, SharpSpectralKernel
export kernel_weight, kernel_radius, spectral_transfer

# Convention (Pope 2000, turbulence/LES standard): the filter scale `ℓ` is the FULL filter
# width. The top-hat spans the disk/ball of radius ℓ/2; the Gaussian is variance-matched to that
# box (constant α = 6); the sharp spectral cutoff is at k_c = π/ℓ. Kernel weights here are
# UNNORMALIZED — the filtering routines normalize by the running area/volume-weighted sum, so any
# constant prefactor is irrelevant.

"""
    AbstractFilterKernel

Abstract supertype for all filter kernels (TopHat, Gaussian, SharpSpectral, …).
"""
abstract type AbstractFilterKernel end

"""
    TopHatKernel <: AbstractFilterKernel

Real-space top-hat (box) filter of full width `ℓ`: unit weight for `d ≤ ℓ/2`, zero otherwise.
"""
struct TopHatKernel <: AbstractFilterKernel end

"""
    GaussianKernel(; α = 6.0) <: AbstractFilterKernel

Real-space Gaussian filter `G_ℓ(d) ∝ exp(-α (d/ℓ)²)`, with `ℓ` the full filter width.

- `α = 6` (default) is the Pope/turbulence-literature convention: the Gaussian's second moment
  matches the top-hat box of width `ℓ` (`σ² = ℓ²/12`).
- `α = 4` reproduces FlowSieve's default Gaussian (which also treats `ℓ` as a diameter), so
  `GaussianKernel(; α = 4)` is directly comparable to FlowSieve output.
"""
struct GaussianKernel{T<:Real} <: AbstractFilterKernel
    α::T
end
GaussianKernel(; α::Real = 6.0) = GaussianKernel(α)

"""
    SharpSpectralKernel <: AbstractFilterKernel

Sharp-spectral (brick-wall) filter: `Ĝ_ℓ(k) = 1` for `k ≤ k_c`, else `0`, with `k_c = π/ℓ`. Best
applied in spectral space (FFTW / FINUFFT / spherical-harmonic extensions); the physical-space
form below is a slowly-decaying `sinc` fallback.
"""
struct SharpSpectralKernel <: AbstractFilterKernel end

# Relative weight below which the (rapidly-decaying) Gaussian footprint is truncated.
const GAUSSIAN_TRUNCATION_TOL = 1e-10

# ---------------------------------------------------------------------------
# Kernel evaluation
# ---------------------------------------------------------------------------

"""
    kernel_weight(kernel::AbstractFilterKernel, d::T, ℓ::T) where {T<:AbstractFloat}

Evaluate the unnormalized kernel weight at distance `d` for filter width `ℓ`.
"""
@inline function kernel_weight(::TopHatKernel, d::T, ℓ::T) where {T<:AbstractFloat}
    return d <= ℓ / T(2) ? one(T) : zero(T)
end

@inline function kernel_weight(k::GaussianKernel, d::T, ℓ::T) where {T<:AbstractFloat}
    return exp(-T(k.α) * (d / ℓ)^2)
end

@inline function kernel_weight(::SharpSpectralKernel, d::T, ℓ::T) where {T<:AbstractFloat}
    # Physical-space fallback: sinc(π d / ℓ). Spectral filters are best applied in spectral space.
    if iszero(d)
        return one(T)
    else
        val = T(π) * d / ℓ
        return sin(val) / val
    end
end

# ---------------------------------------------------------------------------
# Kernel support boundary
# ---------------------------------------------------------------------------

"""
    kernel_radius(kernel::AbstractFilterKernel, ℓ::T) where {T<:AbstractFloat}

Distance beyond which the kernel weight is negligible, used to truncate physical-space
convolution footprints.
"""
@inline kernel_radius(::TopHatKernel, ℓ::T) where {T<:AbstractFloat} = ℓ / T(2)

@inline function kernel_radius(k::GaussianKernel, ℓ::T) where {T<:AbstractFloat}
    # Truncate where exp(-α (r/ℓ)²) < GAUSSIAN_TRUNCATION_TOL  ⇒  r = ℓ √(-ln(tol)/α).
    # For α = 6 this is ≈ 1.96 ℓ (vs the previous, ~4× more expensive, 3 ℓ).
    return ℓ * sqrt(-log(T(GAUSSIAN_TRUNCATION_TOL)) / T(k.α))
end

# Sinc decays only as O(1/d), so the physical-space fallback needs a wide footprint.
@inline kernel_radius(::SharpSpectralKernel, ℓ::T) where {T<:AbstractFloat} = T(10) * ℓ

# ---------------------------------------------------------------------------
# Spectral transfer function Ĝ(|k|, ℓ)
# ---------------------------------------------------------------------------

"""
    spectral_transfer(kernel, kmag::T, ℓ::T) where {T<:AbstractFloat}

Isotropic spectral transfer function `Ĝ(|k|, ℓ)`: the factor by which a Fourier (or
spherical-harmonic) mode of physical wavenumber magnitude `kmag` (rad m⁻¹) is multiplied when
filtering at width `ℓ`. Normalized so `Ĝ(0, ℓ) = 1` (preserves the domain mean).

This is the single source of truth shared by every spectral backend (FFTW / FINUFFT /
spherical-harmonic / NUFSHT extensions). For spherical harmonics, pass the wavenumber of degree `n`,
`kmag = √(n(n+1)) / R`.

- `GaussianKernel(α)`: `Ĝ = exp(-k² ℓ² / (4α))` (the exact Fourier transform of `exp(-α(r/ℓ)²)`).
- `SharpSpectralKernel`: `Ĝ = 1` for `k ≤ π/ℓ`, else `0`.
- `TopHatKernel`: errors — its multidimensional transfer function is an oscillatory Airy/sinc pattern
  that rings; use `method = DirectSum()` for the top-hat, or a Gaussian / sharp-spectral kernel.
"""
@inline spectral_transfer(k::GaussianKernel, kmag::T, ℓ::T) where {T<:AbstractFloat} =
    exp(-kmag^2 * ℓ^2 / (T(4) * T(k.α)))
@inline spectral_transfer(::SharpSpectralKernel, kmag::T, ℓ::T) where {T<:AbstractFloat} =
    kmag <= T(π) / ℓ ? one(T) : zero(T)
@inline function spectral_transfer(::TopHatKernel, ::T, ::T) where {T<:AbstractFloat}
    throw(ArgumentError(
        "Spectral filtering with TopHatKernel is not supported (its multidimensional transfer " *
        "function is an oscillatory Airy/sinc pattern that rings). Use `method = DirectSum()` for " *
        "the top-hat, or a GaussianKernel / SharpSpectralKernel for spectral filtering.",
    ))
end

end # module
