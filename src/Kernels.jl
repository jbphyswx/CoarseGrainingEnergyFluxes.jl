module Kernels

export AbstractFilterKernel, TopHatKernel, GaussianKernel, SharpSpectralKernel
export SmoothHatKernel, HyperGaussianKernel, HighOrderKernel
export kernel_profile, is_separable, is_radial, limb_amplitudes
export profile_integral, profile_cell_average
export kernel_weight, kernel_radius, spectral_transfer, spectral_transfer_degree
export transfer_monotone

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

- `α = 6` (default) is the Pope/turbulence-literature convention: `σ² = ℓ²/(2α) = ℓ²/12`, which is the
  second moment of a top-hat **of width `ℓ` in one dimension**.
- `α = 4` reproduces FlowSieve's default Gaussian (which also treats `ℓ` as a diameter), so
  `GaussianKernel(; α = 4)` is directly comparable to FlowSieve output.

# Variance matching is dimension-dependent

Because the Gaussian is separable, its per-component variance is `ℓ²/(2α)` in every dimension. The
top-hat here is not a separable box but the **disk/ball of radius `ℓ/2`**, whose per-component variance
therefore shrinks with dimension. Measured (quadrature, units of `ℓ²`):

| | 1-D | 2-D (disk) | 3-D (ball) |
|---|---|---|---|
| top-hat `⟨x²⟩` | `ℓ²/12` | `ℓ²/16` | `ℓ²/20` |
| `α` that matches it | 6 | 8 | 10 |

So `α = 6` matches the top-hat exactly on a 1-D grid, and on a 2-D grid it is **15% wider in RMS**
than the top-hat of the same nominal `ℓ`. Use `GaussianKernel(; α = 8)` when the point is a
like-for-like comparison against `TopHatKernel` on a 2-D grid, and `α = 10` in 3-D.

Discretization behaves very differently between the two, also measured: the Gaussian's footprint
reproduces its continuum `⟨x²⟩` to 2e-10 already at `ℓ = 8Δx`, while the top-hat's staircased disk
boundary leaves it 2.0% low at `ℓ = 8Δx`, falling as `O(Δx)` to 0.24% at `ℓ = 64Δx`.
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

"""
    SmoothHatKernel(; steepness = 10.0) <: AbstractFilterKernel

Tapered top-hat, `G_ℓ(d) ∝ ½(1 − tanh(s(2d/ℓ − 1)))`, the kernel used in the Storer et al. papers. It
is a top-hat whose rim is smoothed over a width `≈ ℓ/(2s)`, so it keeps the box's compactness while
removing the discontinuity that makes the box's transfer function ring.

`steepness = 10` is the published value, i.e. `½(1 − tanh((D − 1)/0.1))` with `D = 2d/ℓ`; larger values
approach `TopHatKernel`, smaller ones approach a broad bell.

Real space only — there is no closed-form transfer function, so `method = Spectral()` throws rather
than silently substituting a different kernel. Its `|Ĝ|²` is not monotone (max sidelobe 0.119
measured), so [`transfer_monotone`](@ref) is `false` and it cannot produce a filtering spectrum.
"""
struct SmoothHatKernel{T<:Real} <: AbstractFilterKernel
    steepness::T
end
SmoothHatKernel(; steepness::Real = 10.0) = SmoothHatKernel(steepness)

"""
    HyperGaussianKernel(; α = 1.0) <: AbstractFilterKernel

Super-Gaussian, `G_ℓ(d) ∝ exp(-α (2d/ℓ)⁴)`. Flatter in the core and steeper in the skirt than a
Gaussian of the same nominal width — closer to a box while staying smooth and strictly positive.

Real space only, for the same reason as [`SmoothHatKernel`](@ref): `exp(-r⁴)` has no elementary
Fourier transform. Its `|Ĝ|²` is not monotone (max sidelobe 0.056 measured), so it cannot produce a
filtering spectrum either.
"""
struct HyperGaussianKernel{T<:Real} <: AbstractFilterKernel
    α::T
end
HyperGaussianKernel(; α::Real = 1.0) = HyperGaussianKernel(α)

"""
    HighOrderKernel{P}(; b_over_ℓ = 1/8) <: AbstractFilterKernel

Piecewise-constant kernel with `P` vanishing moments (Sadek & Aluie 2018 §V): a positive body of
half-width `ℓ/2` flanked by alternating-sign limbs of width `b = b_over_ℓ · ℓ`, whose amplitudes are
solved so that `∫xⁿG dx = 0` for `n = 1 … P`.

- `P = 3` is the paper's `M^I`: one negative limb, support `|x| < ℓ/2 + b`.
- `P = 5` is `M^II`: a negative limb then a positive one, support `|x| < ℓ/2 + 2b`.

At the paper's `b = ℓ/8` the amplitudes reduce to `(1, -64/61)` and `(1, -568/257, +200/257)`
respectively, relative to the body (the weights here are unnormalized, as everywhere in this module).

# This is a SEPARABLE kernel, not a radial one

`G(x, y) ≡ G(x) G(y)` — the paper's own generalization, and a genuinely different object from a
radially symmetric kernel: its support is a square, not a disk. [`kernel_weight`](@ref) therefore has
no method for it and throws, and it can only be applied on a grid with separable axes (rectilinear
`StructuredGrid`). A curvilinear or unstructured grid has no axes to factor over, so `plan_filter`
refuses it there rather than silently applying a radial approximation.

# Slope recovery, and the domain it needs

Measured on a synthetic `k^-α` field, `N = 256`, fit over `ℓ = 8-32Δx`: `P = 3` recovers `k⁻⁴` (-4.00
against a true -4) where `TopHatKernel`/`GaussianKernel` saturate at -2.83/-2.73. `P = 5` does not
reach `k⁻⁷` in that box.

That last one is a property of the domain, not of the kernel. With `𝓔(ℓ) = ∫E(k)|Ĝ(kℓ)|²dk` and
`x = kℓ`,

```
Ẽ(k_ℓ) ∝ k_ℓ^{-α} · I ,     I = -∫₀^∞ x^{1-α} (|Ĝ|²)'(x) dx
```

so the slope is `-α` unless `I` diverges; since `1 - |Ĝ|² ∝ x^{p+1}` the integrand is `x^{p+1-α}` and
`I` diverges at **small `x = kℓ`** iff `α ≥ p+2` (Sadek & Aluie eq. 18). The saturation is therefore an
INFRARED effect, set by scales larger than the filter: seeing it needs many decades of spectrum below
`1/ℓ`, and the integrand sharpens with `p`. A `256²` box leaves ~1.5 decades there — enough for
`p = 1`, marginal for `p = 3`, not enough for `p = 5`. A finer grid does not help; it adds modes above
`1/ℓ`, which `I` does not see.

# Two cautions

- **It needs resolution.** Every limb must span at least one cell, `b ≥ Δx`, i.e. `ℓ ≥ 8Δx` at the
  default `b_over_ℓ = 1/8`; `plan_filter` warns below that. Weights are the kernel's integral over each
  cell, not its value at the node ([`profile_cell_average`](@ref)): the discrete `m₂` residual then
  falls as `O(Δx²)` rather than point sampling's `O(Δx)`, 100× smaller at `ℓ = 128Δx`. Still nonzero at
  finite resolution, so the effective order is always below the nominal one.
- **Do not use it for `Π`.** The vanishing moments are bought with negative weights, so `τ` is not
  realizable, and `|Ĝ|²` reaches 0.282 (`P = 3`) and **1.0125** (`P = 5`) past its first zero — the
  latter amplifies. For a flux use `TopHatKernel` or `GaussianKernel`.
"""
struct HighOrderKernel{P, T<:Real} <: AbstractFilterKernel
    b_over_ℓ::T
end

function HighOrderKernel(; order::Integer = 3, b_over_ℓ::Real = 1 / 8)
    order in (3, 5) || throw(ArgumentError(
        "HighOrderKernel is defined for order 3 (the paper's M^I) and order 5 (M^II); got $order. " *
        "M^I cannot be promoted to order 5 by tuning `b_over_ℓ` — the two moment conditions " *
        "coincide only at the degenerate b = 0.",
    ))
    b_over_ℓ > 0 || throw(ArgumentError("HighOrderKernel needs b_over_ℓ > 0, got $b_over_ℓ"))
    return HighOrderKernel{Int(order), typeof(b_over_ℓ)}(b_over_ℓ)
end

"""
    limb_amplitudes(k::HighOrderKernel{P}, ::Type{T}) -> NTuple{P == 3 ? 1 : 2, T}

Limb amplitudes relative to the body, from the vanishing-moment conditions. Dimensionless in
`β = b/ℓ`, so `ℓ` does not enter.
"""
@inline function limb_amplitudes(k::HighOrderKernel{3}, ::Type{T}) where {T<:AbstractFloat}
    β = T(k.b_over_ℓ)
    # ∫x²G = 0 ⇒ a/c = 1/(t³ - 1) with t = 1 + 2β.
    t = one(T) + T(2) * β
    return (-one(T) / (t^3 - one(T)),)
end

@inline function limb_amplitudes(k::HighOrderKernel{5}, ::Type{T}) where {T<:AbstractFloat}
    β = T(k.b_over_ℓ)
    # ∫x²G = ∫x⁴G = 0, eq. (35) divided through by ℓ⁶.
    den = T(4) * β^2 * (T(192) * β^4 + T(400) * β^3 + T(340) * β^2 + T(120) * β + T(15))
    a = (T(124) * β^3 + T(88) * β^2 + T(19) * β + one(T)) / den
    e = (T(4) * β^3 + T(8) * β^2 + T(5) * β + one(T)) / den
    return (-a, e)
end

@inline _n_limbs(::HighOrderKernel{3}) = 1
@inline _n_limbs(::HighOrderKernel{5}) = 2

"""
    kernel_profile(kernel, δ::T, ℓ::T) -> T

One-dimensional profile of a SEPARABLE kernel at signed-or-unsigned axis offset `δ`. The full weight
at a displacement `(δ₁, …, δ_N)` is the product `∏ kernel_profile(kernel, δ_d, ℓ)`.

Defined for [`HighOrderKernel`](@ref), which is separable by construction, and for
[`GaussianKernel`](@ref), which is separable as an identity (`exp(-α r²/ℓ²)` factors), where it
coincides with [`kernel_weight`](@ref). A kernel with no method here is radial-only.
"""
function kernel_profile end

@inline kernel_profile(k::GaussianKernel, δ::T, ℓ::T) where {T<:AbstractFloat} =
    kernel_weight(k, abs(δ), ℓ)

@inline function kernel_profile(k::HighOrderKernel{P}, δ::T, ℓ::T) where {P, T<:AbstractFloat}
    d = abs(δ)
    half = ℓ / T(2)
    d < half && return one(T)
    b = T(k.b_over_ℓ) * ℓ
    amps = limb_amplitudes(k, T)
    @inbounds for n in 1:length(amps)
        d < half + T(n) * b && return amps[n]
    end
    return zero(T)
end

"""
    profile_integral(kernel, t::T, ℓ::T) -> T

`∫₀ᵗ G₁(s) ds` for the 1-D profile, exact. Odd in `t`, since the profile is even.

This exists because [`HighOrderKernel`](@ref) is **discontinuous** with limbs only `b = ℓ/8` wide.
Point-sampling such a kernel onto a grid is a poor quadrature — the sampled mass of a limb depends on
exactly where the nodes fall, which on a non-uniform axis can miss a limb entirely and drive the
normalization denominator negative (measured: the order-5 profile reaches -1.77e3 on a ±3% jittered
axis, and a filtered constant comes back sign-flipped). Integrating the kernel over each cell instead
of sampling it at the node removes that failure completely and is exact for a piecewise-constant
kernel; Sadek & Aluie 2018 §III.F point at the same remedy.
"""
function profile_integral end

@inline function profile_integral(k::HighOrderKernel{P}, t::T, ℓ::T) where {P, T<:AbstractFloat}
    s = sign(t)
    a = abs(t)
    half = ℓ / T(2)
    b = T(k.b_over_ℓ) * ℓ
    amps = limb_amplitudes(k, T)
    # Body, then one term per limb: each contributes its amplitude times the overlap of [0, a] with it.
    acc = min(a, half)
    edge = half
    @inbounds for n in 1:length(amps)
        nxt = half + T(n) * b
        acc += amps[n] * (min(a, nxt) - min(a, edge))
        edge = nxt
    end
    return s * acc
end

"""
    profile_cell_average(kernel, δ::T, Δ::T, ℓ::T) -> T

The 1-D profile averaged over a cell of width `Δ` centred at offset `δ`, i.e.
`(1/Δ) ∫_{δ-Δ/2}^{δ+Δ/2} G₁(s) ds`.

The default is the point sample `kernel_profile(kernel, δ, ℓ)` — for a smooth kernel the two agree to
`O(Δ²)` and the point sample is what every existing engine and stored reference uses, so it is left
exactly as it was. [`HighOrderKernel`](@ref) overrides it with the exact integral, which is what makes
it well-posed on a non-uniform grid; see [`profile_integral`](@ref).
"""
@inline profile_cell_average(k::AbstractFilterKernel, δ::T, ::T, ℓ::T) where {T<:AbstractFloat} =
    kernel_profile(k, δ, ℓ)

@inline function profile_cell_average(k::HighOrderKernel, δ::T, Δ::T, ℓ::T) where {T<:AbstractFloat}
    Δ > 0 || return kernel_profile(k, δ, ℓ)
    h = Δ / T(2)
    return (profile_integral(k, δ + h, ℓ) - profile_integral(k, δ - h, ℓ)) / Δ
end

"""
    is_separable(kernel) -> Bool

Whether the kernel factors as `G(x₁,…,x_N) = ∏ G(x_d)`, so it can be applied as successive 1-D passes
and MUST be, if it has no radial form. See [`kernel_profile`](@ref).
"""
is_separable(::AbstractFilterKernel) = false
is_separable(::GaussianKernel) = true
is_separable(::HighOrderKernel) = true

"""
    is_radial(kernel) -> Bool

Whether the kernel is a function of the scalar distance alone, i.e. whether [`kernel_weight`](@ref)
has a method for it. `HighOrderKernel` is the one kernel here that is separable but NOT radial.
"""
is_radial(::AbstractFilterKernel) = true
is_radial(::HighOrderKernel) = false

@noinline function kernel_weight(k::HighOrderKernel, ::AbstractFloat, ::AbstractFloat)
    throw(ArgumentError(
        "$(nameof(typeof(k))) is a SEPARABLE kernel, G(x,y) = G(x)G(y), not a radial one, so it has " *
        "no weight as a function of distance alone — its support is a square, not a disk. It can only " *
        "be applied on a grid with separable axes (a rectilinear `StructuredGrid`); use " *
        "`Kernels.kernel_profile(kernel, δ, ℓ)` for the per-axis factor.",
    ))
end

# Relative weight below which a rapidly-decaying footprint (Gaussian, hyper-Gaussian, smooth hat) is
# truncated. The same tolerance for all of them, so their radii are comparable.
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

@inline function kernel_weight(k::SmoothHatKernel, d::T, ℓ::T) where {T<:AbstractFloat}
    return (one(T) - tanh(T(k.steepness) * (T(2) * d / ℓ - one(T)))) / T(2)
end

@inline function kernel_weight(k::HyperGaussianKernel, d::T, ℓ::T) where {T<:AbstractFloat}
    return exp(-T(k.α) * (T(2) * d / ℓ)^4)
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

# ½(1 - tanh z) < tol ⟺ z > atanh(1 - 2 tol); the rim sits at 2d/ℓ = 1, so the truncation radius is
# (ℓ/2)(1 + atanh(1 - 2 tol)/s) — ≈ 1.08 ℓ at the default steepness.
@inline function kernel_radius(k::SmoothHatKernel, ℓ::T) where {T<:AbstractFloat}
    z = atanh(one(T) - T(2) * T(GAUSSIAN_TRUNCATION_TOL))
    return ℓ / T(2) * (one(T) + z / T(k.steepness))
end

# exp(-α D⁴) < tol  ⇒  D = (-ln(tol)/α)^(1/4); ≈ 1.09 ℓ at α = 1.
@inline function kernel_radius(k::HyperGaussianKernel, ℓ::T) where {T<:AbstractFloat}
    return ℓ / T(2) * (-log(T(GAUSSIAN_TRUNCATION_TOL)) / T(k.α))^(one(T) / T(4))
end

# Sinc decays only as O(1/d), so the physical-space fallback needs a wide footprint.
@inline kernel_radius(::SharpSpectralKernel, ℓ::T) where {T<:AbstractFloat} = T(10) * ℓ

# Compact by construction: body ℓ/2 plus one limb per vanishing-moment pair. This is the PER-AXIS
# half-width, which is what a separable apply needs; the full support is the square of side 2× this.
@inline kernel_radius(k::HighOrderKernel, ℓ::T) where {T<:AbstractFloat} =
    ℓ / T(2) + T(_n_limbs(k)) * T(k.b_over_ℓ) * ℓ

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

# `½(1 - tanh)` and `exp(-r⁴)` have no elementary Fourier transform, so there is nothing to return
# here. Refusing explicitly beats a `MethodError`: the message says what to do instead, and it makes
# clear these are real-space kernels by construction rather than by omission.
spectral_transfer(k::Union{SmoothHatKernel,HyperGaussianKernel}, ::AbstractFloat, ::AbstractFloat) =
    throw(ArgumentError(
        "$(nameof(typeof(k))) has no closed-form spectral transfer function, so `method = Spectral()` " *
        "cannot be used with it — apply it with `method = RealSpace()`, which is its exact form, or " *
        "switch to `GaussianKernel()`/`SharpSpectralKernel()`/`TopHatKernel()` for a spectral apply.",
    ))
spectral_transfer_degree(k::Union{SmoothHatKernel,HyperGaussianKernel}, ::Integer, ℓ::T, ::T) where {T<:AbstractFloat} =
    spectral_transfer(k, ℓ, ℓ)

# A separable kernel's transfer function is a product over axes, `∏ Ĝ₁(k_d)`, not a function of |k|,
# so the isotropic interface cannot express it. The 1-D factor is a closed form — a sum of sincs, one
# per limb — but the spectral backends here all multiply by an isotropic `Ĝ(|k|)`, so the honest answer
# is to refuse rather than to isotropize it.
@noinline spectral_transfer(k::HighOrderKernel, ::AbstractFloat, ::AbstractFloat) =
    throw(ArgumentError(
        "$(nameof(typeof(k))) is separable, so its transfer function is the product ∏ Ĝ₁(k_d) over " *
        "axes rather than a function of |k| — the isotropic spectral backends here cannot represent " *
        "it. Apply it with `method = RealSpace()`, which is its exact form.",
    ))
@noinline spectral_transfer_degree(k::HighOrderKernel, ::Integer, ℓ::T, ::T) where {T<:AbstractFloat} =
    spectral_transfer(k, ℓ, ℓ)

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

# ---------------------------------------------------------------------------
# Spectrum admissibility
# ---------------------------------------------------------------------------

"""
    transfer_monotone(kernel::AbstractFilterKernel) -> Bool

Whether `d|Ĝ(k)|²/dk ≤ 0` holds on `(0, ∞)`.

Sadek & Aluie (2018) eq. (21): this is the condition under which the filtering spectral density
`Ẽ(k_ℓ)` is guaranteed non-negative. A kernel that fails it can produce a spectrum with negative
values, which is why `Diagnostics.filtering_spectrum` refuses one.

| kernel | monotone | why |
|---|---|---|
| `GaussianKernel` | `true` | `\\|Ĝ\\|² = exp(-k²ℓ²/2α)`, strictly decreasing |
| `SharpSpectralKernel` | `true` | `1` then `0` — non-increasing |
| `TopHatKernel` | `false` | `Ĝ = 2J₁(kR)/(kR)` oscillates; `\\|Ĝ\\|²` rises by `+0.0026` near `kℓ ≈ 8.8` |

The condition is sufficient, not necessary — Sadek & Aluie's own fallback argument concedes it is
"not a rigorous proof" — so `false` means "not guaranteed", not "certainly negative". It is also
narrower than it looks: it says nothing about the `k^{-(p+2)}` slope ceiling, which binds the
Gaussian just as hard as the top-hat (see `Diagnostics.filtering_spectrum`).

There is deliberately **no fallback method**. A new kernel gets a `MethodError` here rather than a
guessed answer, so its author has to establish which way it goes.
"""
function transfer_monotone end

transfer_monotone(::GaussianKernel) = true
transfer_monotone(::SharpSpectralKernel) = true
transfer_monotone(::TopHatKernel) = false
# Both are compact-ish real-space kernels, so both ring: measured max `|Ĝ|` sidelobes 0.119 and 0.056.
transfer_monotone(::SmoothHatKernel) = false
transfer_monotone(::HyperGaussianKernel) = false
# The high-order kernels ring hardest of all — `M^II`'s `|Ĝ|²` reaches 1.0124, i.e. it AMPLIFIES.
transfer_monotone(::HighOrderKernel) = false

"""
    check_spectrum_kernel(kernel::AbstractFilterKernel)

Throw an `ArgumentError` unless `kernel` satisfies [`transfer_monotone`](@ref). Called wherever a
filtering spectral DENSITY is produced; the cumulative energy needs no such check.
"""
function check_spectrum_kernel(kernel::AbstractFilterKernel)
    transfer_monotone(kernel) && return nothing
    throw(ArgumentError(
        "$(nameof(typeof(kernel))) cannot produce a filtering spectral density: its |Ĝ(k)|² is not " *
        "monotone decreasing, so Sadek & Aluie (2018) eq. (21) does not hold and Ẽ(k_ℓ) may come out " *
        "negative. Use a kernel with `transfer_monotone(kernel) == true` (e.g. `GaussianKernel()`), " *
        "or ask for `cumulative_energy`, which is well defined for any kernel. `$(nameof(typeof(kernel)))` " *
        "remains a valid — and the default — choice for the flux Π, which this condition does not bear on.",
    ))
end

end # module
