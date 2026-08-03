module Kernels

export AbstractFilterKernel, TopHatKernel, GaussianKernel, SharpSpectralKernel
export kernel_weight, kernel_radius, spectral_transfer, spectral_transfer_degree

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
    # Truncate where exp(-α (r/ℓ)²) < GAUSSIAN_TRUNCATION_TOL  ⇒  r = ℓ √(-ln(tol)/α); ≈ 1.96 ℓ at α = 6.
    return ℓ * sqrt(-log(T(GAUSSIAN_TRUNCATION_TOL)) / T(k.α))
end

# Sinc decays only as O(1/d), so the physical-space fallback needs a wide footprint.
@inline kernel_radius(::SharpSpectralKernel, ℓ::T) where {T<:AbstractFloat} = T(10) * ℓ

# ---------------------------------------------------------------------------
# Spectral transfer function Ĝ(|k|, ℓ)
# ---------------------------------------------------------------------------

"""
    spectral_transfer(kernel, kmag::T, ℓ::T) where {T<:AbstractFloat}

Isotropic planar spectral transfer function `Ĝ(|k|, ℓ)`: the factor by which a Fourier mode of
physical wavenumber magnitude `kmag` (rad m⁻¹) is multiplied when filtering at width `ℓ` on a 2D
Cartesian grid. Normalized so `Ĝ(0, ℓ) = 1` (preserves the domain mean). Shared by the FFTW and
FINUFFT backends (both 2D-Cartesian-only today). For the spherical-harmonic-degree analog used by
the FastSphericalHarmonics/NUFSHT backends, see [`spectral_transfer_degree`](@ref).

- `GaussianKernel(α)`: `Ĝ = exp(-k² ℓ² / (4α))` (the exact Fourier transform of `exp(-α(r/ℓ)²)`).
- `SharpSpectralKernel`: `Ĝ = 1` for `k ≤ π/ℓ`, else `0`.
- `TopHatKernel`: `Ĝ = 2 J₁(kR)/(kR)`, `R = ℓ/2` — the exact 2D (disk) Fourier transform of a top-hat
  (the "jinc" function, the circular-aperture analog of `sinc`). This oscillates and goes negative in
  `k`; that is the correct, exact behavior of a disk's Fourier transform, not an approximation error.
  This method is provided entirely by the SpecialFunctions weak dependency
  (`CoarseGrainingEnergyFluxesSpecialFunctionsExt`, for `besselj1`) — core has no method for
  `TopHatKernel` here (Julia disallows two modules defining the identical method signature, so a
  throwing core stub could never be replaced by the extension's real one); without `using
  SpecialFunctions` loaded, calling this is a `MethodError` with a registered hint pointing at the fix.
"""
@inline spectral_transfer(k::GaussianKernel, kmag::T, ℓ::T) where {T<:AbstractFloat} =
    exp(-kmag^2 * ℓ^2 / (T(4) * T(k.α)))
@inline spectral_transfer(::SharpSpectralKernel, kmag::T, ℓ::T) where {T<:AbstractFloat} =
    kmag <= T(π) / ℓ ? one(T) : zero(T)

"""
    spectral_transfer_degree(kernel, l::Integer, ℓ::T, R::T) where {T<:AbstractFloat}

Spherical-harmonic-DEGREE-indexed transfer function `Ĝ_l`, used by the FastSphericalHarmonics/NUFSHT
backends in place of [`spectral_transfer`](@ref)'s continuous wavenumber `kmag` when a kernel's shape
needs the discrete degree `l` itself, not just the Laplace–Beltrami eigenvalue `k_l = √(l(l+1))/R`.
`GaussianKernel`/`SharpSpectralKernel` are smooth isotropic functions of `k_l` alone, so they simply
delegate to `spectral_transfer`; `TopHatKernel`'s spherical-cap window genuinely needs `l`.
"""
@inline function spectral_transfer_degree(
    kernel::Union{GaussianKernel,SharpSpectralKernel}, l::Integer, ℓ::T, R::T,
) where {T<:AbstractFloat}
    k_l = sqrt(T(l) * T(l + 1)) / R
    return spectral_transfer(kernel, k_l, ℓ)
end

"""
    spectral_transfer_degree(::TopHatKernel, l::Integer, ℓ::T, R::T) where {T<:AbstractFloat}

Exact spherical-cap top-hat window function (Jekeli 1981's gravity-field averaging kernel; the
sphere's analog of the planar top-hat's Bessel-`J₁` transfer function): for a cap of angular radius
`θ0 = ℓ/(2R)` (i.e. full physical width `ℓ`),

```
Ĝ_l = [P_{l-1}(x) - P_{l+1}(x)] / [(2l+1)(1 - x)]  ≡  (1 + x) P′_l(x) / (l(l+1)),   x = cosθ0
```

evaluated in the right-hand form, via the Legendre recurrences
`(n+1)P_{n+1}(x) = (2n+1)x Pₙ(x) - n P_{n-1}(x)` and `P′_{n+1}(x) = (2n+1)Pₙ(x) + P′_{n-1}(x)`
— no external dependency needed (unlike the planar case's Bessel `J₁`). Like the planar top-hat, this
oscillates and goes negative in `l`; that is the exact, correct behavior of a spherical cap's harmonic
content, not an artifact.
"""
@inline function spectral_transfer_degree(::TopHatKernel, l::Integer, ℓ::T, R::T) where {T<:AbstractFloat}
    l == 0 && return one(T)
    # The derivative form, not the difference: as the cap shrinks both `P_{l-1} - P_{l+1}` and `1 - x`
    # vanish, costing 8 significant digits at ℓ = 1 km on Earth and 4 at ℓ = 10 m.
    x = one(T) - T(2) * sin(ℓ / (4R))^2   # cos θ0, without the cancellation in `1 - cos θ0`
    P0, P1 = one(T), x                    # P_0(x), P_1(x)
    D0, D1 = zero(T), one(T)              # P′_0(x), P′_1(x)
    for n in 1:(l - 1)
        P0, P1 = P1, ((2n + 1) * x * P1 - n * P0) / (n + 1)
        D0, D1 = D1, (2n + 1) * P0 + D0   # P0 is P_n after the line above, so this is P′_{n+1}
    end
    return (one(T) + x) * D1 / (l * (l + 1))
end

end # module
