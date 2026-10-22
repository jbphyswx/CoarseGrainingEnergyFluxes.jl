module Kernels

using FlowGeometries: FlowGeometries

export AbstractFilterKernel, TopHatKernel, GaussianKernel, SharpSpectralKernel
export SmoothHatKernel, HyperGaussianKernel, HighOrderKernel
export kernel_profile, is_separable, is_radial, limb_amplitudes, radial_limb_amplitudes
export spherical_limb_amplitudes
export profile_integral, profile_cell_average
export kernel_weight, kernel_radius, spectral_transfer, spectral_transfer_degree
export spectral_transfer_degrees!, LegendreQuadrature
export transfer_monotone

# Convention (Pope 2000): the filter scale `ℓ` is the full filter width. The top-hat spans the disk/ball
# of radius ℓ/2, the default Gaussian (α = 6) has the 1-D top-hat's variance, and the sharp spectral
# cutoff is at k_c = π/ℓ. Kernel weights are unnormalized: the filtering routines divide by the
# area/volume-weighted sum of the weights.

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

Real-space Gaussian filter `G_ℓ(d) ∝ exp(-α (d/ℓ)²)`, with `ℓ` the full filter width. Its
per-component variance is `σ² = ℓ²/(2α)` in every dimension; the default `α = 6` gives `ℓ²/12`, the
second moment of a top-hat of width `ℓ` in one dimension (Pope 2000).

[`TopHatKernel`](@ref) is the disk/ball of radius `ℓ/2`, whose per-component variance shrinks with
dimension:

| | 1-D | 2-D (disk) | 3-D (ball) |
|---|---|---|---|
| top-hat `⟨x²⟩` | `ℓ²/12` | `ℓ²/16` | `ℓ²/20` |
| `α` that matches it | 6 | 8 | 10 |

For a like-for-like comparison with `TopHatKernel`, use `α = 8` on a 2-D grid and `α = 10` in 3-D.
"""
struct GaussianKernel{T<:Real} <: AbstractFilterKernel
    α::T
end
GaussianKernel(; α::Real = 6.0) = GaussianKernel(α)

"""
    SharpSpectralKernel <: AbstractFilterKernel

Sharp-spectral (brick-wall) filter: `Ĝ_ℓ(k) = 1` for `k ≤ k_c`, else `0`, with `k_c = π/ℓ`.

`method = Spectral()` multiplies by that transfer function. `RealSpace()` convolves with its inverse
transform in the grid's `D` dimensions, `G(r) ∝ J_{D/2}(k_c r)/(k_c r)^{D/2}`, normalized to 1 at `r = 0`:

| `D` | weight at `x = k_c r` |
|---|---|
| 1 | `sin x / x` |
| 2 | `2 J₁(x)/x`, from `SpecialFunctions` (run `using SpecialFunctions`) |
| 3 | `3 (sin x − x cos x)/x³` |

`G` decays as `r^{-(D+1)/2}`, so `∫|G|` diverges and a truncated footprint's transfer function ripples
about the brick wall, most near `k_c`, at any truncation radius. The footprint ends near `10ℓ`, at the
radius where the truncated kernel's mass equals the full kernel's ([`kernel_radius`](@ref)), so the
passband is centred on 1. The weights change sign, so under `Deformable` a window mass beside a coast
is a sum of signed weights and can approach zero.
"""
struct SharpSpectralKernel <: AbstractFilterKernel end

"""
    SmoothHatKernel(; steepness = 10.0) <: AbstractFilterKernel

Tapered top-hat, `G_ℓ(d) ∝ ½(1 − tanh(s(2d/ℓ − 1)))`, the kernel used in the Storer et al. papers: a
top-hat whose rim is smoothed over a width `≈ ℓ/(2s)`.

`steepness = 10` is the published value, i.e. `½(1 − tanh((D − 1)/0.1))` with `D = 2d/ℓ`; larger values
approach `TopHatKernel`, smaller ones approach a broad bell.

It has no closed-form transfer function on the plane, where `method = Spectral()` throws; on a sphere
its degree transfer is its Legendre transform ([`spectral_transfer_degrees!`](@ref)). Its `|Ĝ|²` is
not monotone, so [`transfer_monotone`](@ref) is `false`.
"""
struct SmoothHatKernel{T<:Real} <: AbstractFilterKernel
    steepness::T
end
SmoothHatKernel(; steepness::Real = 10.0) = SmoothHatKernel(steepness)

"""
    HyperGaussianKernel(; α = 1.0) <: AbstractFilterKernel

Super-Gaussian, `G_ℓ(d) ∝ exp(-α (2d/ℓ)⁴)`: flatter in the core and steeper in the skirt than a
Gaussian of the same nominal width, smooth and strictly positive.

`exp(-r⁴)` has no elementary Fourier transform, so on the plane `method = Spectral()` throws; on a sphere
its degree transfer is its Legendre transform ([`spectral_transfer_degrees!`](@ref)). Its `|Ĝ|²` is not
monotone, so [`transfer_monotone`](@ref) is `false`.
"""
struct HyperGaussianKernel{T<:Real} <: AbstractFilterKernel
    α::T
end
HyperGaussianKernel(; α::Real = 1.0) = HyperGaussianKernel(α)

"""
    HighOrderKernel(; order = 3, b_over_ℓ = 1/8) <: AbstractFilterKernel

Piecewise-constant kernel with `P = order` vanishing moments (Sadek & Aluie 2018 §V): a positive body of
half-width `ℓ/2` flanked by alternating-sign limbs of width `b = b_over_ℓ · ℓ`, whose amplitudes are
solved so that `∫xⁿG dx = 0` for `n = 1 … P`.

- `P = 3` is the paper's `M^I`: one negative limb, support `|x| < ℓ/2 + b`.
- `P = 5` is `M^II`: a negative limb then a positive one, support `|x| < ℓ/2 + 2b`.

At the paper's `b = ℓ/8` the amplitudes reduce to `(1, -64/61)` and `(1, -568/257, +200/257)`
respectively, relative to the body.

# Its form on each geometry

On the plane it is the paper's product `G(x, y) = G(x) G(y)` over the plane's axes ([`kernel_profile`](@ref)),
whose support is a square. A sphere has no global axes, so there it is the isotropic form: the same
body and limbs as a function of geodesic distance, with the amplitudes of
[`spherical_limb_amplitudes`](@ref) and the same `P` vanishing moments of total degree.
[`kernel_weight`](@ref) throws for it: the product is not a function of distance.

# Slope recovery, and the domain it needs

`P` vanishing moments let the filtering spectrum follow a `k^-α` spectrum for `α < P + 2` (Sadek &
Aluie 2018, eq. 18), past the `k⁻³` ceiling of `TopHatKernel` and `GaussianKernel`. With
`𝓔(ℓ) = ∫E(k)|Ĝ(kℓ)|²dk` and `x = kℓ`,

```
Ẽ(k_ℓ) ∝ k_ℓ^{-α} · I ,     I = -∫₀^∞ x^{1-α} (|Ĝ|²)'(x) dx
```

so the slope is `-α` unless `I` diverges; since `1 - |Ĝ|² ∝ x^{P+1}` the integrand is `x^{P+1-α}` and
`I` diverges at small `x = kℓ` iff `α ≥ P+2` (Sadek & Aluie eq. 18). The saturation is an infrared
effect, set by scales larger than the filter: seeing it needs many decades of spectrum below `1/ℓ`,
more as `P` grows. A finer grid adds only modes above `1/ℓ`, which `I` does not see.

# On a grid

The filter a grid runs is a sum over cells, so the moments that set its order are the discrete ones,
`Σₖ wₖ eₖᵅ` over the cells' displacements `eₖ`. Its weights are the kernel's mean over each cell times
the cell measure ([`profile_cell_average`](@ref) on a rectilinear grid), corrected by the least change
that makes every moment of degree `1…P` vanish, so the order holds at every `ℓ` and on every grid; as
`ℓ` nears a few cells the weights tend to the identity filter.

# Not for `Π`

The vanishing moments need negative weights, so the filter is not a positive operator and `τ` is not
guaranteed positive (realizable). Past its first zero `|Ĝ|²` rises again, above 1 for `P = 5`, which
amplifies those scales. For a flux use `TopHatKernel` or `GaussianKernel`.
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
    radial_limb_amplitudes(k::HighOrderKernel{P}, ::Type{T}, ::Val{D}) -> NTuple

Limb amplitudes, relative to the body, of the kernel's isotropic form in `D` dimensions: the profile
`G(r)` with the body and limbs of [`HighOrderKernel`](@ref) as a function of distance, whose moments
`∫ r^{2m} G(r) dV`, `m = 1…⌊P/2⌋`, vanish. The odd moments, and every moment `∫ x^α G` of odd degree in
some coordinate, vanish by symmetry, so the form has `P` vanishing moments of total degree, as the
product does. `D = 1` is [`limb_amplitudes`](@ref).

With `ρₙ = ½ + nβ` the body and limb radii in units of `ℓ` and `q = 2m + D`, the amplitudes solve
`Σₙ aₙ (ρₙ^q − ρₙ₋₁^q) = −ρ₀^q`.
"""
radial_limb_amplitudes(k::HighOrderKernel, ::Type{T}, ::Val{D}) where {T<:AbstractFloat, D} =
    _limb_solve(k, T, (a, b, m) -> (b^(2m + D) - a^(2m + D)) / T(2m + D))

"""
    spherical_limb_amplitudes(k::HighOrderKernel{P}, ℓ::T, R::T) -> NTuple

[`radial_limb_amplitudes`](@ref) on a sphere of radius `R`, with the moments taken against the sphere's
area element in geodesic polar coordinates, `∫ r^{2m} G(r) R sin(r/R) dr dθ`, so the form's moments
vanish in the normal coordinates of its centre. They tend to the planar two-dimensional amplitudes as
`ℓ/R → 0`.
"""
spherical_limb_amplitudes(k::HighOrderKernel, ℓ::T, R::T) where {T<:AbstractFloat} =
    _limb_solve(k, T, (a, b, m) -> _sine_moment(a, b, 2m, ℓ / R))

# `∫ₐᵇ s^q sin(x s) ds / x` by the sine series, `Σⱼ (−x²)ʲ/(2j+1)! · (b^p − a^p)/p` with `p = q + 2j + 2`.
function _sine_moment(a::T, b::T, q::Int, x::T) where {T<:AbstractFloat}
    s = zero(T)
    c = one(T)
    for j in 0:200
        p = q + 2j + 2
        term = c * (b^p - a^p) / T(p)
        s += term
        abs(term) <= eps(T) * abs(s) && break
        c *= -x * x / T((2j + 2) * (2j + 3))
    end
    return s
end

# The limb amplitudes zeroing `moment(ρ₋, ρ₊, m)`, the `m`-th moment of a shell, for `m = 1…L`, with
# `ρₙ = ½ + nβ` and the body's amplitude 1.
function _limb_solve(k::HighOrderKernel{3}, ::Type{T}, moment::F) where {T<:AbstractFloat, F}
    β = T(k.b_over_ℓ)
    ρ0, ρ1 = T(1) / 2, T(1) / 2 + β
    return (-moment(zero(T), ρ0, 1) / moment(ρ0, ρ1, 1),)
end

function _limb_solve(k::HighOrderKernel{5}, ::Type{T}, moment::F) where {T<:AbstractFloat, F}
    β = T(k.b_over_ℓ)
    ρ0, ρ1, ρ2 = T(1) / 2, T(1) / 2 + β, T(1) / 2 + 2β
    a11, a12, b1 = moment(ρ0, ρ1, 1), moment(ρ1, ρ2, 1), -moment(zero(T), ρ0, 1)
    a21, a22, b2 = moment(ρ0, ρ1, 2), moment(ρ1, ρ2, 2), -moment(zero(T), ρ0, 2)
    det = a11 * a22 - a12 * a21
    return ((b1 * a22 - a12 * b2) / det, (a11 * b2 - b1 * a21) / det)
end

"""
    kernel_profile(kernel, δ::T, ℓ::T) -> T

One-dimensional profile of a separable kernel at signed-or-unsigned axis offset `δ`. The full weight
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

[`profile_cell_average`](@ref) differences it to average [`HighOrderKernel`](@ref)'s discontinuous
profile over a cell, exact for a piecewise-constant kernel. A point sample of its narrow limbs depends
on where the nodes fall and can miss a limb on a non-uniform axis (Sadek & Aluie 2018 §III.F).
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

The default is the point sample `kernel_profile(kernel, δ, ℓ)`, which agrees with the average to
`O(Δ²)` for a smooth kernel. [`HighOrderKernel`](@ref) takes the exact average through
[`profile_integral`](@ref).
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

Whether the kernel factors as `G(x₁,…,x_N) = ∏ G(x_d)`, so it can be applied as successive 1-D passes,
and must be, if it has no radial form. See [`kernel_profile`](@ref).
"""
is_separable(::AbstractFilterKernel) = false
is_separable(::GaussianKernel) = true
is_separable(::HighOrderKernel) = true

"""
    is_radial(kernel) -> Bool

Whether the kernel is a function of the scalar distance alone, i.e. whether [`kernel_weight`](@ref)
has a method for it. [`HighOrderKernel`](@ref) is the one separable kernel here without a radial form.
"""
is_radial(::AbstractFilterKernel) = true
is_radial(::HighOrderKernel) = false

@noinline function kernel_weight(k::HighOrderKernel, ::AbstractFloat, ::AbstractFloat)
    throw(ArgumentError(
        "$(nameof(typeof(k))) has no weight as a function of distance alone: on the plane it is the " *
        "product G(x)G(y) of `Kernels.kernel_profile(kernel, δ, ℓ)`, and on a sphere its isotropic form " *
        "has the limb amplitudes of `Kernels.radial_limb_amplitudes`. The filtering engines weigh it " *
        "cell by cell.",
    ))
end

# Relative weight below which the Gaussian, hyper-Gaussian and smooth-hat footprints are truncated.
const GAUSSIAN_TRUNCATION_TOL = 1e-10

# ---------------------------------------------------------------------------
# Kernel evaluation
# ---------------------------------------------------------------------------

"""
    kernel_weight(kernel::AbstractFilterKernel, d::T, ℓ::T) where {T<:AbstractFloat}
    kernel_weight(kernel::AbstractFilterKernel, d::T, ℓ::T, ::Val{D}) where {T<:AbstractFloat, D}

Evaluate the unnormalized kernel weight at distance `d` for filter width `ℓ`, in `D` dimensions. A kernel
given by a radial profile has the same weight in every dimension; [`SharpSpectralKernel`](@ref)'s is
the inverse transform of its transfer function, which depends on `D`.
"""
@inline kernel_weight(k::AbstractFilterKernel, d::T, ℓ::T, ::Val) where {T<:AbstractFloat} =
    kernel_weight(k, d, ℓ)

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

@noinline kernel_weight(::SharpSpectralKernel, ::AbstractFloat, ::AbstractFloat) = throw(ArgumentError(
    "SharpSpectralKernel's real-space weight is the inverse transform of its brick wall, which depends " *
    "on the dimension: call `kernel_weight(kernel, d, ℓ, Val(D))`.",
))

@inline kernel_weight(::SharpSpectralKernel, d::T, ℓ::T, dim::Val) where {T<:AbstractFloat} =
    _ball_transform(T(π) * d / ℓ, dim)

"""
    _ball_transform(x, Val(D))

The Fourier transform of the unit ball's indicator in `D` dimensions at `|k| = x`, divided by the
ball's volume: `sin x / x`, `2J₁(x)/x`, `3(sin x − x cos x)/x³`. It is the transfer function of a
top-hat of radius `R` at `x = |k|R`, and the inverse transform of the brick wall `|k| ≤ k_c` at
`x = k_c r`.
"""
@inline _ball_transform(x::T, ::Val{1}) where {T<:AbstractFloat} = iszero(x) ? one(T) : sin(x) / x

# Below `x = 0.2` the Taylor series to `x⁸` is exact to round-off, where the closed form loses
# `log₁₀(1/x²)` digits to the cancellation in `sin x − x cos x`.
@inline function _ball_transform(x::T, ::Val{3}) where {T<:AbstractFloat}
    if abs(x) < T(0.2)
        x2 = x * x
        return one(T) - x2 * (inv(T(10)) - x2 * (inv(T(280)) - x2 * (inv(T(15120)) - x2 / T(1330560))))
    end
    s, c = sincos(x)
    return T(3) * (s - x * c) / x^3
end

# `D = 2` is `2J₁(x)/x`, defined by the SpecialFunctions extension as the more specific `Val{2}` method.
@noinline function _ball_transform(::T, ::Val{D}) where {T<:AbstractFloat, D}
    throw(ArgumentError(D == 2 ?
        "the two-dimensional top-hat transfer and sharp-spectral weight are the jinc `2J₁(x)/x`, provided " *
        "by the SpecialFunctions weak dependency. Run `using SpecialFunctions`." :
        "the top-hat transfer and sharp-spectral weight are defined in 1, 2 and 3 dimensions; got $D."))
end

# ---------------------------------------------------------------------------
# Kernel support boundary
# ---------------------------------------------------------------------------

"""
    kernel_radius(kernel::AbstractFilterKernel, ℓ::T) where {T<:AbstractFloat}
    kernel_radius(kernel::AbstractFilterKernel, ℓ::T, ::Val{D}) where {T<:AbstractFloat, D}

Distance at which a physical-space convolution footprint is truncated, in `D` dimensions: where the
weight is negligible, or, for [`SharpSpectralKernel`](@ref), where the truncated kernel keeps its full
mass.
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

@inline kernel_radius(k::AbstractFilterKernel, ℓ::T, ::Val) where {T<:AbstractFloat} = kernel_radius(k, ℓ)

@noinline kernel_radius(::SharpSpectralKernel, ::AbstractFloat) = throw(ArgumentError(
    "SharpSpectralKernel's truncation radius depends on the dimension: call " *
    "`kernel_radius(kernel, ℓ, Val(D))`.",
))

# `k_c R` where the kernel's mass inside radius `R` equals its mass over all space, nearest `10π`: the
# roots of `(2/π)Si(x) = 1`, `J₀(x) = 0` and `(2/π)(Si(x) − sin x) = 1` in one, two and three dimensions.
# Truncated there, the filter keeps `Ĝ(0) = 1`, and normalizing it leaves the passband at 1.
const _SHARP_TRUNCATION = (29.87844038878343, 30.634606468431976, 31.384170011211985)

@inline function kernel_radius(::SharpSpectralKernel, ℓ::T, ::Val{D}) where {T<:AbstractFloat, D}
    1 <= D <= 3 || throw(ArgumentError(
        "SharpSpectralKernel's real-space weight is defined in 1, 2 and 3 dimensions; got $D."))
    return T(_SHARP_TRUNCATION[D]) * ℓ / T(π)
end

# Compact by construction: body ℓ/2 plus one limb per vanishing-moment pair. The product's half-width
# along each axis, and the isotropic form's radius.
@inline kernel_radius(k::HighOrderKernel, ℓ::T) where {T<:AbstractFloat} =
    ℓ / T(2) + T(_n_limbs(k)) * T(k.b_over_ℓ) * ℓ

# ---------------------------------------------------------------------------
# Spectral transfer function Ĝ(|k|, ℓ)
# ---------------------------------------------------------------------------

"""
    spectral_transfer(kernel, kmag::T, ℓ::T) where {T<:AbstractFloat}
    spectral_transfer(kernel, kmag::T, ℓ::T, ::Val{D}) where {T<:AbstractFloat, D}

Isotropic transfer function `Ĝ(|k|, ℓ)` in `D` dimensions (two without the `Val`): the factor by
which a Fourier mode of wavenumber magnitude `kmag` (rad m⁻¹) is multiplied when filtering at width
`ℓ`, with `Ĝ(0, ℓ) = 1`. For the spherical-harmonic-degree analog see
[`spectral_transfer_degree`](@ref).

- `GaussianKernel(α)`: `Ĝ = exp(-k² ℓ² / (4α))`, the transform of `exp(-α(r/ℓ)²)` in every dimension.
- `SharpSpectralKernel`: `Ĝ = 1` for `k ≤ π/ℓ`, else `0`.
- `TopHatKernel`: the transform of the ball of radius `R = ℓ/2`, `sin(kR)/(kR)`, `2J₁(kR)/(kR)` and
  `3(sin kR − kR cos kR)/(kR)³` in one, two and three dimensions. It oscillates and goes negative in
  `k`. The two-dimensional form needs `using SpecialFunctions`.
"""
@inline spectral_transfer(k::GaussianKernel, kmag::T, ℓ::T) where {T<:AbstractFloat} =
    exp(-kmag^2 * ℓ^2 / (T(4) * T(k.α)))
@inline spectral_transfer(::SharpSpectralKernel, kmag::T, ℓ::T) where {T<:AbstractFloat} =
    kmag <= T(π) / ℓ ? one(T) : zero(T)
@inline spectral_transfer(k::TopHatKernel, kmag::T, ℓ::T) where {T<:AbstractFloat} =
    spectral_transfer(k, kmag, ℓ, Val(2))

@inline spectral_transfer(k::AbstractFilterKernel, kmag::T, ℓ::T, ::Val) where {T<:AbstractFloat} =
    spectral_transfer(k, kmag, ℓ)
@inline spectral_transfer(::TopHatKernel, kmag::T, ℓ::T, dim::Val) where {T<:AbstractFloat} =
    _ball_transform(kmag * ℓ / T(2), dim)

# `½(1 - tanh)` and `exp(-r⁴)` have no elementary Fourier transform.
spectral_transfer(k::Union{SmoothHatKernel,HyperGaussianKernel}, ::AbstractFloat, ::AbstractFloat) =
    throw(ArgumentError(
        "$(nameof(typeof(k))) has no closed-form transfer function on the plane, so `method = Spectral()` " *
        "cannot filter with it there — apply it with `method = RealSpace()`, which is its exact form, or " *
        "switch to `GaussianKernel()`/`SharpSpectralKernel()`/`TopHatKernel()` for a spectral apply. On a " *
        "sphere its degree transfer is its Legendre transform, `spectral_transfer_degrees!`.",
    ))

"""
    spectral_transfer(kernel, k::NTuple{D,T}, ℓ::T) -> T

The transfer function at the wavevector `k`: `spectral_transfer(kernel, |k|, ℓ, Val(D))` for an isotropic
kernel, and `∏ Ĝ₁(k_d)` for [`HighOrderKernel`](@ref)'s product on the plane, `Ĝ₁` the profile's
transform over its mass.
"""
@inline spectral_transfer(kernel::AbstractFilterKernel, k::NTuple{D,T}, ℓ::T) where {D,T<:AbstractFloat} =
    spectral_transfer(kernel, sqrt(sum(abs2, k)), ℓ, Val(D))
@inline spectral_transfer(kernel::HighOrderKernel, k::NTuple{D,T}, ℓ::T) where {D,T<:AbstractFloat} =
    prod(map(kd -> _profile_transfer(kernel, kd, ℓ), k))

@noinline spectral_transfer(k::HighOrderKernel, ::AbstractFloat, ::AbstractFloat) =
    throw(ArgumentError(
        "$(nameof(typeof(k))) on the plane is the product G(x)G(y), whose transfer function depends on " *
        "the wavevector rather than on |k|: call `spectral_transfer(kernel, (k₁, …, k_D), ℓ)`. On a " *
        "sphere its isotropic form has `spectral_transfer_degree`.",
    ))

# The body and limbs as a sum of balls, `G = Σₙ cₙ·[r < ρₙ]` with `ρₙ = ℓ/2 + nb`, `cₙ = aₙ − aₙ₊₁`,
# `a₀ = 1` and `a_{L+1} = 0`: `Σₙ cₙ Vₙ τₙ / Σₙ cₙ Vₙ` for `(Vₙ, τₙ) = f(ρₙ)`, a ball's measure and
# transfer function.
@inline function _shell_sum(f::F, amps::NTuple{L,T}, k::HighOrderKernel, ℓ::T) where {F, L, T<:AbstractFloat}
    b = T(k.b_over_ℓ) * ℓ
    num = zero(T)
    den = zero(T)
    for n in 0:L
        an = n == 0 ? one(T) : amps[n]
        an1 = n == L ? zero(T) : amps[n + 1]
        V, τ = f(ℓ / T(2) + T(n) * b)
        num += (an - an1) * V * τ
        den += (an - an1) * V
    end
    return num / den
end

# One axis of the product: the interval `|x| < ρ` has measure `2ρ` and transform `sin(kρ)/(kρ)`.
@inline _profile_transfer(k::HighOrderKernel, kd::T, ℓ::T) where {T<:AbstractFloat} =
    _shell_sum(ρ -> (T(2) * ρ, _ball_transform(kd * ρ, Val(1))), limb_amplitudes(k, T), k, ℓ)

# ---------------------------------------------------------------------------
# Transfer per spherical-harmonic degree
# ---------------------------------------------------------------------------

"""
    LegendreQuadrature(T = Float64)

Gauss–Legendre nodes, weights and per-degree sums for [`spectral_transfer_degrees!`](@ref), in `T`. The
buffers are growable host vectors that keep the largest size asked of them, so a transfer repeated at any
scale allocates nothing once that size is reached.
"""
mutable struct LegendreQuadrature{T<:AbstractFloat, V<:AbstractVector{T}}
    λ::V     # the longitudes `spherical_quadrature!` writes with its nodes, unread here
    φ::V     # the nodes `x` on `[-1, 1]` as latitudes, `asin x`
    w::V
    acc::V   # one sum per degree
    n::Int   # the order the nodes are for, 0 before the first
end

LegendreQuadrature(::Type{T} = Float64) where {T<:AbstractFloat} =
    LegendreQuadrature(Vector{T}(), Vector{T}(), Vector{T}(), Vector{T}(), 0)

# The `n`-point rule, solved only when the order changes.
function _legendre_nodes!(q::LegendreQuadrature, n::Int)
    q.n == n && return q
    resize!(q.λ, 2n - 1)
    resize!(q.φ, n)
    resize!(q.w, n)
    FlowGeometries.SphericalSampling.spherical_quadrature!(
        q.λ, q.φ, q.w, FlowGeometries.SphericalSampling.GaussLegendreSampling(), n,
    )
    q.n = n
    return q
end

"""
    spectral_transfer_degrees!(Ĝ, kernel, ℓ, R[, quadrature]) -> Ĝ

`Ĝ[n+1]` for every degree `n = 0 … length(Ĝ) − 1`: the factor by which filtering with `kernel` at width `ℓ`
on a sphere of radius `R` multiplies a degree-`n` spherical harmonic. It is the Legendre transform of the
two-dimensional real-space weight `G` in geodesic distance over the support the real-space engines keep,
normalized to `Ĝ(0) = 1` (Aluie 2019, eq. 34),

    Ĝ(n) = ∫₀^γₜ G(Rγ) Pₙ(cos γ) sin γ dγ / ∫₀^γₜ G(Rγ) sin γ dγ ,   γₜ = min(kernel_radius(kernel, ℓ, Val(2))/R, π),

which by the Funk–Hecke theorem (eqs. 36–37) scales every harmonic of degree `n`. A spherical cap
(`TopHatKernel`, and each shell of `HighOrderKernel`'s isotropic form) has the closed form of Jekeli
(1981); every other kernel is integrated by Gauss–Legendre quadrature in `γ`. Both run in the element
type of `quadrature`, by default that of `Ĝ`.
"""
function spectral_transfer_degrees!(
    Ĝ::AbstractVector{<:AbstractFloat}, kernel::AbstractFilterKernel, ℓ::Real, R::Real,
    quadrature::LegendreQuadrature = LegendreQuadrature(eltype(Ĝ)),
)
    isempty(Ĝ) && return Ĝ
    acc = resize!(quadrature.acc, length(Ĝ))
    fill!(acc, zero(eltype(acc)))
    _degree_sums!(acc, kernel, ℓ, R, quadrature)
    m = @inbounds acc[1]
    Ĝ .= acc ./ m
    return Ĝ
end

"""
    spectral_transfer_degree(kernel, l::Integer, ℓ::T, R::T) -> T

The degree-`l` entry of [`spectral_transfer_degrees!`](@ref): the transfer of a degree-`l` spherical
harmonic on a sphere of radius `R`. It oscillates and goes negative in `l` for a kernel with an edge, as the
planar transfer does in `k`.
"""
spectral_transfer_degree(kernel::AbstractFilterKernel, l::Integer, ℓ::T, R::T) where {T<:AbstractFloat} =
    spectral_transfer_degrees!(Vector{T}(undef, Int(l) + 1), kernel, ℓ, R)[Int(l) + 1]

# `acc[l+1] += c Ĝ_l` for every degree of the cap of angular radius `θ₀`, by Jekeli's (1981)
# `Ĝ_l = [P_{l−1}(x) − P_{l+1}(x)] / [(2l+1)(1 − x)] = (1 + x) P′_l(x)/(l(l+1))`, `x = cos θ₀`, in the
# derivative form, which keeps its digits as the cap shrinks; the recurrences are
# `(n+1)P_{n+1} = (2n+1)x Pₙ − n P_{n−1}` and `P′_{n+1} = (2n+1)Pₙ + P′_{n−1}`.
function _add_cap!(acc::AbstractVector{TW}, c::TW, θ0::TW) where {TW<:AbstractFloat}
    x = one(TW) - TW(2) * sin(min(θ0, TW(π)) / 2)^2   # cos θ₀, with no cancellation in `1 − cos θ₀`
    @inbounds acc[1] += c
    P0, P1 = one(TW), x                                # P_{l−1}, P_l
    D0, D1 = zero(TW), one(TW)                         # P′_{l−1}, P′_l
    @inbounds for l in 1:(length(acc) - 1)
        acc[l + 1] += c * (one(TW) + x) * D1 / (TW(l) * TW(l + 1))
        P0, P1 = P1, (TW(2l + 1) * x * P1 - TW(l) * P0) / TW(l + 1)
        D0, D1 = D1, TW(2l + 1) * P0 + D0
    end
    return acc
end

_degree_sums!(acc::AbstractVector{TW}, ::TopHatKernel, ℓ::Real, R::Real, ::LegendreQuadrature) where {TW} =
    _add_cap!(acc, one(TW), TW(ℓ) / (2 * TW(R)))

# The isotropic form is a sum of caps, `G = Σₙ cₙ·[γ < θₙ]` with `θₙ = (ℓ/2 + nb)/R` and `cₙ = aₙ − aₙ₊₁`
# (`a₀ = 1`, `a_{L+1} = 0`, the amplitudes of `spherical_limb_amplitudes`), each weighted by its area over
# `4πR²`, `sin²(θₙ/2)`.
function _degree_sums!(acc::AbstractVector{TW}, k::HighOrderKernel, ℓ::Real, R::Real, ::LegendreQuadrature) where {TW}
    ℓw, Rw = TW(ℓ), TW(R)
    amps = spherical_limb_amplitudes(k, ℓw, Rw)
    L = length(amps)
    b = TW(k.b_over_ℓ) * ℓw
    for n in 0:L
        an = n == 0 ? one(TW) : amps[n]
        an1 = n == L ? zero(TW) : amps[n + 1]
        θ = min((ℓw / 2 + TW(n) * b) / Rw, TW(π))
        _add_cap!(acc, (an - an1) * sin(θ / 2)^2, θ)
    end
    return acc
end

const _QuadratureKernel = Union{GaussianKernel, SharpSpectralKernel, SmoothHatKernel, HyperGaussianKernel}

function _degree_sums!(acc::AbstractVector{TW}, kernel::_QuadratureKernel, ℓ::Real, R::Real,
                       q::LegendreQuadrature{TW}) where {TW<:AbstractFloat}
    ℓw, Rw = TW(ℓ), TW(R)
    γt = min(kernel_radius(kernel, ℓw, Val(2)) / Rw, TW(π))
    L = length(acc) - 1
    n = _legendre_order(kernel, L, γt, ℓw, Rw)
    _legendre_nodes!(q, n)
    @inbounds for i in 1:n
        # `γ = γₜ(1 + xᵢ)/2` from the node's latitude `asin xᵢ`, as `1 + xᵢ = 2sin²(π/4 + φᵢ/2)`.
        γ = γt * sin(TW(π) / 4 + q.φ[i] / 2)^2
        g = q.w[i] * kernel_weight(kernel, Rw * γ, ℓw, Val(2)) * sin(γ)
        x = cos(γ)
        acc[1] += g
        L == 0 && continue
        acc[2] += g * x
        P0, P1 = one(TW), x
        for l in 1:(L - 1)
            P0, P1 = P1, (TW(2l + 1) * x * P1 - TW(l) * P0) / TW(l + 1)
            acc[l + 2] += g * P1
        end
    end
    return acc
end

# Gauss–Legendre points that integrate `G(Rγ) Pₙ(cos γ) sin γ` over `[0, γₜ]` to round-off for every
# `n ≤ L`. In `x ∈ [-1, 1]`, `γ = γₜ(1 + x)/2`, `Pₙ(cos γ) sin γ` is a trigonometric polynomial of frequency
# at most `ω = (L + 1)γₜ/2`, whose Chebyshev series ends near degree `ω + 12∛ω`; the profile adds the
# degree of `_profile_degree`, and `n` points integrate degree `2n − 1`.
function _legendre_order(kernel::AbstractFilterKernel, L::Int, γt::T, ℓ::T, R::T) where {T<:AbstractFloat}
    ω = (L + 1) * γt / 2
    return cld(ceil(Int, ω + 12 * cbrt(ω)) + _profile_degree(kernel, γt, ℓ, R) + 32, 2)
end

# The Chebyshev degree in `x` past which a profile's coefficients are below round-off. The Gaussian is
# `exp(-c(1 + x)²)` and the hyper-Gaussian `exp(-a((1 + x)/2)⁴)`, entire, with `c ≤ -ln(tol)/4` and
# `a ≤ -ln(tol)` at the truncation tolerance; the sharp-spectral `2J₁(z)/z` oscillates at frequency
# `Z = πRγₜ/(2ℓ)` in `x`; the smooth hat's `tanh` has a pole at distance `δ = πℓ/(2sRγₜ)` from the
# interval, and the Bernstein bound halfway to it puts its coefficients below round-off past `80/δ`.
_profile_degree(::GaussianKernel, _, _, _) = 64
_profile_degree(::HyperGaussianKernel, _, _, _) = 128
function _profile_degree(::SharpSpectralKernel, γt::T, ℓ::T, R::T) where {T<:AbstractFloat}
    Z = T(π) * R * γt / (2 * ℓ)
    return ceil(Int, Z + 12 * cbrt(Z)) + 40
end
_profile_degree(k::SmoothHatKernel, γt::T, ℓ::T, R::T) where {T<:AbstractFloat} =
    ceil(Int, 160 * T(k.steepness) * R * γt / (T(π) * ℓ))

# ---------------------------------------------------------------------------
# Spectrum admissibility
# ---------------------------------------------------------------------------

"""
    transfer_monotone(kernel::AbstractFilterKernel) -> Bool

Whether `d|Ĝ(k)|²/dk ≤ 0` holds on `(0, ∞)`.

Sadek & Aluie (2018) eq. (21): this is the condition under which the filtering spectral density
`Ẽ(k_ℓ)` is guaranteed non-negative. A kernel that fails it can produce a spectrum with negative
values: `Diagnostics.StrictSpectrum` refuses one, and the `coarse_grain` default
`Diagnostics.AutoSpectrum` fills `NaN` for it.

| kernel | monotone | why |
|---|---|---|
| `GaussianKernel` | `true` | `\\|Ĝ\\|² = exp(-k²ℓ²/2α)`, strictly decreasing |
| `SharpSpectralKernel` | `true` | `1` then `0` — non-increasing |
| `TopHatKernel` | `false` | `Ĝ = 2J₁(kR)/(kR)` oscillates; `\\|Ĝ\\|²` hits zero at `kℓ ≈ 7.66`, then climbs to `0.0175` at `kℓ ≈ 10.27` |

The condition is sufficient: `false` means a non-negative spectrum is not guaranteed. It does not bear
on the `k⁻³` slope ceiling, which binds the Gaussian as hard as the top-hat (see
`Diagnostics.filtering_spectrum`).

`transfer_monotone` has no fallback method: every kernel defines its own.
"""
function transfer_monotone end

transfer_monotone(::GaussianKernel) = true
transfer_monotone(::SharpSpectralKernel) = true
transfer_monotone(::TopHatKernel) = false
transfer_monotone(::SmoothHatKernel) = false
transfer_monotone(::HyperGaussianKernel) = false
transfer_monotone(::HighOrderKernel) = false

"""
    check_spectrum_kernel(kernel::AbstractFilterKernel)

Throw an `ArgumentError` unless `kernel` satisfies [`transfer_monotone`](@ref). This is the strict
reading of the admissibility condition, selected by `Diagnostics.StrictSpectrum()`; the cumulative
energy needs no such check.
"""
function check_spectrum_kernel(kernel::AbstractFilterKernel)
    transfer_monotone(kernel) && return nothing
    throw(ArgumentError(
        "$(nameof(typeof(kernel))) cannot produce a guaranteed non-negative filtering spectral " *
        "density: its |Ĝ(k)|² is not monotone decreasing, so Sadek & Aluie (2018) eq. (21) does not " *
        "hold and Ẽ(k_ℓ) may come out negative. Four ways forward:\n" *
        "  • `spectrum = CoarseGrainingEnergyFluxes.Diagnostics.ForceSpectrum()` — compute it anyway " *
        "and check its sign yourself. The condition is sufficient, not necessary, and often fails only " *
        "in a narrow band.\n" *
        "  • `spectrum = CoarseGrainingEnergyFluxes.Diagnostics.NoSpectrum()` — skip the density; the " *
        "field comes back `NaN`.\n" *
        "  • Use a kernel with `transfer_monotone(kernel) == true`, e.g. `GaussianKernel()`.\n" *
        "  • Ask for `cumulative_energy`, which is well defined for any kernel.\n" *
        "`$(nameof(typeof(kernel)))` remains a valid — and the default — choice for the flux Π, which " *
        "this condition does not bear on.",
    ))
end

end # module
