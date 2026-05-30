module Kernels

export AbstractFilterKernel, TopHatKernel, GaussianKernel, SharpSpectralKernel
export kernel_weight, kernel_radius

"""
    AbstractFilterKernel

Abstract supertype for all filter kernels (TopHat, Gaussian, SharpSpectral, etc.).
"""
abstract type AbstractFilterKernel end

"""
    TopHatKernel <: AbstractFilterKernel

Standard real-space top-hat (box) filter:
G_ℓ(d) = 1 / (π (ℓ/2)²) for d ≤ ℓ/2, 0 otherwise.
"""
struct TopHatKernel <: AbstractFilterKernel end

"""
    GaussianKernel <: AbstractFilterKernel

Standard real-space Gaussian filter:
G_ℓ(d) = A * exp(-6 * d² / ℓ²).
"""
struct GaussianKernel <: AbstractFilterKernel end

"""
    SharpSpectralKernel <: AbstractFilterKernel

Sharp-spectral filter that acts as a brick-wall cutoff in spectral space.
G_ℓ(k) = 1 for k ≤ k_c, 0 otherwise, where k_c = π / ℓ.
"""
struct SharpSpectralKernel <: AbstractFilterKernel end

# ---------------------------------------------------------------------------
# Kernel evaluation
# ---------------------------------------------------------------------------

"""
    kernel_weight(kernel::AbstractFilterKernel, d::T, ℓ::T) where {T<:AbstractFloat}

Evaluate the unnormalized kernel weight at distance `d` for a filter scale `ℓ`.
"""
@inline function kernel_weight(::TopHatKernel, d::T, ℓ::T) where {T<:AbstractFloat}
    return d <= ℓ / T(2) ? one(T) : zero(T)
end

@inline function kernel_weight(::GaussianKernel, d::T, ℓ::T) where {T<:AbstractFloat}
    # Standard choice in turbulence literature is exp(-6 * (d/ℓ)²) which matches the second moment of the box filter
    return exp(-T(6) * (d / ℓ)^2)
end

@inline function kernel_weight(::SharpSpectralKernel, d::T, ℓ::T) where {T<:AbstractFloat}
    # For physical space filtering: sinc(π d / ℓ)
    # However, spectral filters are best computed directly in spectral space (FFTWExt / FINUFFTExt)
    # We provide physical space fallback here.
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

Return the coordinate integration distance after which the kernel weight is negligible.
Used to optimize physical space convolutions by truncating integration bounds.
"""
@inline kernel_radius(::TopHatKernel, ℓ::T) where {T<:AbstractFloat} = ℓ / T(2)
@inline kernel_radius(::GaussianKernel, ℓ::T) where {T<:AbstractFloat} = T(3) * ℓ # 3-sigma cutoff (e.g. exp(-6*9) ≈ 3.7e-24, very safe)
@inline kernel_radius(::SharpSpectralKernel, ℓ::T) where {T<:AbstractFloat} = T(10) * ℓ # Sinc decay is slow (O(1/x)), require larger footprint in physical space fallback

end # module
