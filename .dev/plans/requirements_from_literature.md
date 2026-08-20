# The coarse-graining framework: theory, requirements, and implementation constraints

This is the reference for *why* the package computes what it computes. It carries the derivations, the
equations with their sources, the conditions each result depends on, and the measurements that pin down
the numbers — enough that no claim here needs to be re-derived or re-looked-up.

Notation throughout: `ℓ` is the filter scale, `G_ℓ` the kernel, `ū_ℓ = G_ℓ * u` the filtered field,
`S̄` the filtered strain rate, and

```
τ̄_ℓ(f,g) ≡ (fg)‾_ℓ − f̄_ℓ ḡ_ℓ
```

the **generalized second moment** (Germano 1992 eq. 24). Sources are cited as `[AE-I]` etc. and listed
in full at the end. Measured values are marked **[measured]** and were computed directly, not quoted.

---

# Part I — The filtering operation

## 1. What coarse-graining is, and what it is not

Coarse-graining is convolution with a normalized kernel:

```
ū_ℓ(x) = ∫ d^d r  G_ℓ(r) u(x − r),        G_ℓ(r) = ℓ^{−d} G(r/ℓ)
```

The whole framework rests on this being interpretable as a **local space average**, which is a stronger
requirement than "a linear smoothing operator" and is where the kernel conditions come from.

Two things it is deliberately *not*:

- **It is not a projection.** `ū̄ ≠ ū` in general. This is why the multiscale/band-pass machinery
  (§7) is nontrivial and why the Germano identity is needed to relate filtering at two scales.
- **It is not a Fourier truncation.** The sharp spectral filter is a special (and, per §5, awkward) case
  rather than the definition. `[BSK+23]` demonstrates concretely why the distinction matters for bounded
  domains: spherical-harmonic filtering produces "beams of spectral ringing that extend deep into land
  regions", and at a 1000 km filter it "fills the global ocean with zonal bands", whereas convolutional
  coarse-graining confines nonzero velocity to within `ℓ/2` inland.

## 2. Kernel conditions and where each one is consumed

### 2.1 The stated conditions

`[AE-I]` §II.A, verbatim:

> "Any real-valued function G(**r**) may be chosen as a filter kernel, as long as it is sufficiently
> smooth, decays sufficiently rapidly for large r, and is normalized so that ∫d^d r G(**r**)=1. It is
> assumed furthermore that G satisfies ∫d^d r **r** G(**r**)=**0** and ∫d^d r |**r**|²G(**r**)=𝒪(1),
> so that the main support of G is in the ball of radius 1 about the origin."

> "In order to interpret this as a local space-average, G must also be positive, or G(**r**) ≥ 0 for
> all **r**."

`[A17]` §2.2 restates it: "any kernel function G which is smooth, rapidly decaying, positive and with
integral normalized to unity … We usually assume also that the kernel is centered … and that it has
variance of order unity, ∫d^d r |r|² G(r) ≈ 1", with the hedge "Some of these requirements (e.g.
smoothness, positivity) may be relaxed in some situations".

### 2.2 What depends on what

| condition | required | what consumes it |
|---|---|---|
| `∫G = 1` | yes | `Ĝ(0) = 1`; the domain mean is preserved; everything downstream |
| `∫**r**G = 0` | yes | the increment representation, §2.3 — hence Galilean invariance and the locality bounds |
| `∫\|**r**\|²G = 𝒪(1)`, **non-zero** | yes | this *defines* `ℓ`. Nothing fails if it is non-zero; it is required to be non-zero |
| `G ≥ 0` | yes | (i) `ū` is a genuine local average; (ii) pointwise positivity of subscale KE — an **iff**, §2.4 |
| smoothness | yes | `[A17]` Prop. 1 (inertial-range inviscid dynamics) needs `‖∇²G‖₂ < ∞`; Prop. 2 (vanishing direct viscous dissipation of large-scale energy) needs `‖∇G‖₂ < ∞` |
| compact support | **no** | rapid decay suffices; compact support only extends Prop. 1 from `𝕋³` to `ℝ³` |
| radial symmetry | no | "often convenient", explicitly optional |
| "S-type" (`Ĝ=1` for `k<1`, `0` for `k>ρ`) | no | a *convenience* for the telescoping band decomposition (§7.2) and for the UV-locality bound |

**An unresolved tension inside the literature itself:** `[A17]` requires smoothness — its Props. 1–2 need
`‖∇G‖₂` and `‖∇²G‖₂` finite, both of which *diverge* for a top-hat — while `[AHV18]`'s working kernel is
a discontinuous top-hat. `[A17]`'s hedge ("may be relaxed in some situations") is the only bridge offered.
This is worth knowing before treating either paper's conditions as absolute.

### 2.3 Why the first moment matters: the increment representation

This is the mechanism behind the locality results, and it is the reason `∫rG = 0` is not cosmetic.

With `∫G = 1` and `∫rG = 0`, the filtered gradient, the stress, and the subscale field can each be
written purely in terms of **velocity increments** `δu(x;r) ≡ u(x+r) − u(x)` over `|r| ≲ ℓ`, with no
dependence on the absolute velocity `u(x)` itself. Schematically, because `∫G(r) dr = 1` kills the
constant and `∫r G(r) dr = 0` kills the linear term,

```
ū_ℓ(x) − u(x) = ∫ dr G_ℓ(r) δu(x;−r)
τ̄_ℓ(u_i,u_j)  = ∫ dr G_ℓ(r) δu_i δu_j  −  (∫ dr G_ℓ(r) δu_i)(∫ dr G_ℓ(r) δu_j)
```

Both sides are manifestly invariant under `u → u + U₀`. **The physical content**: the flux at scale `ℓ`
depends only on the velocity *differences* within a ball of radius `~ℓ`, not on the sweeping velocity of
the large-scale flow past that point. Without a vanishing first moment, a uniform translation leaks into
the flux, and "the amount of energy cascade at a point" becomes observer-dependent.

The resulting locality bounds, `[AE-I]` eqs. (25)–(26), for a flow with Hölder exponent `σ₃`:

```
IR contribution to Π_ℓ from scales Δ ≫ ℓ :   𝒪( ε (ℓ/Δ)^{1−σ₃} )
UV contribution to Π_ℓ from scales δ ≪ ℓ :   𝒪( ε (δ/ℓ)^{2σ₃} )
```

Both vanish as the scale separation grows, which is the precise statement that the cascade is
scale-local. For K41 scaling `σ₃ = 1/3` these are `(ℓ/Δ)^{2/3}` and `(δ/ℓ)^{2/3}`. `[AE-II]` measured
the UV exponent in 512³ DNS and found `P^{−0.83}` to `P^{−0.98}` against a predicted `(K/P)^{2/3}`.

### 2.4 Realizability: why positivity is an iff

`[AE-I]` §II.A, verbatim:

> "It is a positive quantity at every point in the flow if and only if the filtering kernel G(**r**) is
> positive for all **r**, as was proved by Vreman et al. (1994)."

The quantity is the **subscale kinetic energy** `k̄_ℓ ≡ ½τ̄_ℓ(u_i,u_i) = ½[(|u|²)‾ − |ū|²]`. The forward
direction is Jensen's inequality: if `G ≥ 0` and `∫G = 1`, then `G` defines a probability measure, so
`(|u|²)‾ ≥ |ū|²` pointwise. The converse — that a kernel taking negative values admits a field with
`k̄ < 0` somewhere — is Vreman, Geurts & Kuerten (1994).

The consequence is concrete: with a sign-indefinite kernel you can compute a **negative local subscale
energy**, and any model or interpretation resting on `k̄ ≥ 0` becomes unsound. `[A13]` makes positivity
an *explicit hypothesis* of its Props. 2, 4, 5, 6, because positivity is what licenses Jensen's
`1/ρ̄_ℓ ≤ (1/ρ)‾_ℓ` in the compressible case.

## 3. Kernel order, and its conflict with positivity

### 3.1 Definition and the Fourier characterization

`[SA18]` eq. (11), verbatim:

> "We shall call a kernel G(x) ``p-th order'' iff ∫dx x^n G(x) = 0 for n=1,…,p, and ∫dx x^{p+1} G(x) ≠ 0."

Note **`n = 0` is excluded** — it equals 1 by normalization. This differs from the wavelet convention,
where `∫x^n ψ = 0` for `n = 0,…,p−1`; `[SA18]` §IV flags the difference explicitly.

The derivation connecting moments to the transfer function, `[SA18]` eq. (10): expanding
`Ĝ(k) = ∫ G(x) e^{−ikx} dx` in `k`,

```
Ĝ^{(n)}(0) = (−i)^n ∫ x^n G(x) dx
```

so vanishing moments `1…p` ⇔ **flatness of `Ĝ` at `k = 0` to order `p`**, i.e. eq. (12):

```
Ĝ(k) = 1 + k^{p+1} φ(k),        φ(0) ≠ 0
⟺  1 − Ĝ(k) ∝ k^{p+1}  as k → 0
```

Because odd moments vanish automatically for an even kernel, "Any even kernel is of an odd integer order
p ≥ 1", so `p` climbs in steps of two: 1 → 3 → 5. **Top-hat and Gaussian are `p = 1`.** A `p`-th order
even kernel therefore has `(p−1)/2` *non-trivial* (even) vanishing moments: `M^I` cancels `m₂`; `M^II`
cancels `m₂` and `m₄`.

`[SA18]` also compares against wavelets: wavelets achieve `α ≤ 2p+1` versus filtering's `α ≤ p+2`,
"However, for even kernels … odd moments automatically vanish and p increases in increments of two,
eliminating the advantage of wavelets in this regard."

### 3.2 The conflict, stated from both sides

`[AE-I]` Appendix 1, verbatim:

> "Note, however, that filter kernels G of ``S-type'' cannot be non-negative, since ∫d^d r |**r**|^{2n}
> G(**r**) = ((−△)^n Ĝ)(**k**)|_{**k**=0} = 0 for all positive integers n. In that respect, they are
> similar to the sharp spectral filter, whose physical-space kernel is proportional to a Bessel function
> that also takes on negative values."

`[SA18]` §V: "any kernel of order higher than unity cannot be positive everywhere in x-space, otherwise
even moments would not vanish."

The argument in one line: a positive normalized kernel is a probability density, and a probability
density has strictly positive variance. Requiring `∫x²G = 0` therefore forces `G` to change sign.

**So the two families of papers want opposite things**, and this is the single most important structural
fact in the whole subject:

| | wants | for |
|---|---|---|
| `[AE-I]`, `[A13]`, `[A17]` — the **flux** framework | `G ≥ 0`, `∫x²G = 𝒪(1)` non-zero | local-average interpretation, realizability, the inviscid criterion |
| `[SA18]` — the **spectrum** framework | `∫x^n G = 0` for `n ≤ p`, hence `G` sign-indefinite | resolving spectral slopes steeper than `k⁻³` |

A positive `p = 1` kernel is therefore the **correct** choice for flux, not a deficiency. `[AE-I]`'s own
DNS uses a Gaussian; `[AHV18]` uses a top-hat.

### 3.3 The resolution: sequential high-pass filtering

`[ZA24]` resolves the tension rather than trading off. Instead of raising the kernel order, apply a
`p = 1` **positive** kernel repeatedly in a high-pass arrangement, reaching `α < 2p+3` while every
individual filtering step remains a genuine local average. This preserves realizability and gets steeper
extractable slopes, and is the route to prefer if steep spectra are needed.

`[ZA24]` §III.B also gives a `p = 3` kernel as a linear combination of Gaussians:

```
G^{p3}_ℓ(x) = c·G_ℓ(x) − c'·G_{ℓ'}(x−x₀) − c'·G_{ℓ'}(x+x₀),     c' = (c−1)/2
```

**The published condition for `x₀` is a typo**: it reads `x₀ = cℓ²/(12(c−1)) − ℓ'²/12`, whose right-hand
side has units of length². The dimensionally correct condition is

```
x₀² = cℓ²/(12(c−1)) − ℓ'²/12
```

**[measured]** verified at the paper's stated `c = 1.1`, `ℓ/ℓ' = 2`: the `x₀²` reading gives
`m₂ = 9.7e−16`; the printed reading gives `m₂ = 9.3e−3`.

## 4. The `M^I` and `M^II` stencil kernels

`[SA18]` §V, "Constructing Simple Stencil Kernels of Higher Order". Both are symmetric, piecewise
constant, compactly supported, with `b` a **free** parameter and `ℓ` the width of the main body.

### 4.1 Definitions

**`M^I`**, order `p = 3`. Body `+c` on `|x| < ℓ/2`; legs `−a` on `ℓ/2 < |x| < ℓ/2 + b`. Support
`|x| < ℓ/2 + b`. Eq. (34):

```
a/c = 1 / ((1 + 2b/ℓ)³ − 1)         from  ∫dx x² M^I(x) = 0
cℓ − 2ab = 1                         normalization
```

**`M^II`**, order `p = 5`. Adds arms `+e` on `ℓ/2 + b < |x| < ℓ/2 + 2b` (the paper sets `d = b`).
Support `|x| < ℓ/2 + 2b`. Eq. (35):

```
a/c = (124b³ℓ³ + 88b²ℓ⁴ + 19bℓ⁵ + ℓ⁶) / (4b²(192b⁴ + 400b³ℓ + 340b²ℓ² + 120bℓ³ + 15ℓ⁴))
e/c = (  4b³ℓ³ +  8b²ℓ⁴ +  5bℓ⁵ + ℓ⁶) / (4b²(192b⁴ + 400b³ℓ + 340b²ℓ² + 120bℓ³ + 15ℓ⁴))
cℓ − 2ab + 2eb = 1
```

from the two constraints `∫dx x² M^II = 0` and `∫dx x⁴ M^II = 0`.

At the paper's choice `b = ℓ/8`, fully reduced:

| kernel | region | value | support |
|---|---|---|---|
| `M^I` | `\|x\| < ℓ/2` | `+61/(45ℓ) ≈ +1.355556/ℓ` | `\|x\| < 5ℓ/8` |
| | `ℓ/2 < \|x\| < 5ℓ/8` | `−64/(45ℓ) ≈ −1.422222/ℓ` | |
| `M^II` | `\|x\| < ℓ/2` | `+257/(165ℓ) ≈ +1.557576/ℓ` | `\|x\| < 3ℓ/4` |
| | `ℓ/2 < \|x\| < 5ℓ/8` | `−568/(165ℓ) ≈ −3.442424/ℓ` | |
| | `5ℓ/8 < \|x\| < 3ℓ/4` | `+40/(33ℓ) ≈ +1.212121/ℓ` | |

matching the paper's reported `a/c = 64/61`, `c = 61/(45ℓ)`; `a/c = 568/257`, `e/c = 200/257`,
`c = 257/(165ℓ)`.

**[measured]**, in exact rational arithmetic: `M^I` moments `n = 1,2,3` vanish and `n = 4` gives
`−5ℓ⁴/256 ≠ 0` ⇒ `p = 3`. `M^II` moments `n = 1…5` vanish and `n = 6` gives `+225ℓ⁶/28672 ≠ 0` ⇒ `p = 5`.
Independently confirmed `1 − Ĝ(k) ∝ k^{p+1}` with `p = 1, 3, 5` for top-hat, `M^I`, `M^II`.

### 4.2 Two implementation details from §V

- **Higher dimensions are a separable product, not radial**: "it is also straightforward to generalize any
  of these kernels to higher dimensions by defining it as a separable product; for example in 2D, we
  define G(x,y) ≡ G(x)G(y). If kernel G(x) is of order p in 1D, then G(x) is of the same order p in
  higher dimensions." This is a different object from a radially symmetric kernel.
- **Grid constraint**: "representing SS kernels on a grid requires at least 1 grid-cell of size Δx for
  each of the limbs. Therefore, the smallest length-scale ℓ that can be probed by such a kernels is
  limited by b ≥ Δx." With `b = ℓ/8`, that is `ℓ ≥ 8Δx`, i.e. `m ≥ 4` in their `ℓ_m = 2mΔx` convention.

`M^I` cannot be promoted to `p = 5` by tuning `b`: "It might be tempting to choose the free parameter b to
satisfy ∫dx x⁴ M^I(x) = 0, however, it is straightforward to check that the solution is not realizable."
The reason: `m₂ = 0` requires `a/c = 1/(t³−1)` and `m₄ = 0` requires `a/c = 1/(t⁵−1)`, with
`t = 1 + 2b/ℓ`; these coincide only at the degenerate `b = 0`.

### 4.3 Two measured cautions against using them for flux

Both verified two independent ways (closed-form segment transform and direct quadrature).

**(1) They are poor low-pass filters at high `k`.** `|Ĝ|²` sidelobe peaks: top-hat 0.047, `M^I` 0.282,
and **`M^II` reaches 1.0124 at `kℓ ≈ 17`** — it *amplifies* energy there, with further lobes at 0.893 and
0.863. They are constructed to make `1 − Ĝ ∝ k^{p+1}` near `k = 0`, which is all the slope argument needs;
nothing constrains their behaviour at high `k`. Using `M^II` to form `ū` and hence `Π = −S̄:τ̄` would admit
small-scale energy essentially unattenuated. This is consistent with `[SA18]`'s own hedge that "compact
spatial kernels are not strictly local in k-space … which can lead to additional smoothing as a function
of scale."

**(2) The vanishing moment does not survive naive discretization.** For a point-sampled stencil, midpoint
quadrature gives

```
m₂^discrete = m₂^exact − (Δx²/12)·m₀
```

so a kernel with exact `m₂ = 0` discretizes to `m₂ = −Δx²/12`, **independent of kernel shape**. At the
paper's own minimum resolution `ℓ = 8Δx`, that residual is `−ℓ²/768` — only ~64× below the top-hat's
`ℓ²/12`, and comparable to the intended leading `m₄k⁴` term at `kℓ ≈ 1`, which is exactly the range the
filtering spectrum probes. `[SA18]` §III.F guards against this ("This is necessary to guarantee that the
discretized kernel (or stencil) moments vanish to within high precision") but gives no quantitative
criterion. To obtain the order actually paid for, either `ℓ/Δx` must be well above 8, or the weights must
be **cell-averaged** rather than point-sampled.

## 5. The sharp spectral filter

Explicitly endorsed for flux, with specific caveats. `[AE-II]` §IV, verbatim:

> "However, we would like to moderate the claim of Eyink that spectral energy flux ``is an inappropriate
> measure of energy transfer.'' This is true for the *unsubtracted* flux Π_K^{uns}, which is pointwise
> non-Galilei-invariant and becomes scale-local only when averaged over space. On the other hand, the SGS
> spectral flux Π_K is scale-local, even in the absolute sense, without cancellations due to
> space-averaging. In particular, the sharp spectral filter has a sound theoretical basis for use in
> large-eddy simulation (LES) modeling of turbulent energy cascade."

The caveats, all stated in `[AE-II]`:

1. It applies to the **subtracted / Galilean-invariant** SGS flux only.
2. **Logarithmic (octave) bands are required** — "The width of wavenumber bands is more important for
   scale-locality than the grading of the filter kernel."
3. The rigorous IR decay exponent is **worse** than for graded filters: `(Q/K)^{1/6}` vs `(Q/K)^{2/3}`.
4. **No pointwise Hölder estimates are obtainable** — "we do not have pointwise estimates of energy flux
   at each space point x in terms of the local Hölder exponent, using spectral filters."
5. It fails the real-space decay conditions of `[AE-I]` — "The discussion in these papers did not apply
   to sharp-spectral filters, which do not satisfy the modest decay conditions in physical space that
   were employed there."
6. A lacunary-series counterexample means "it is true as a general mathematical fact that energy flux
   defined by the sharp spectral filter may be dominated by nonlocal triads (unlike flux defined with
   graded filters)", excluded only by an empirical spectrum condition.
7. Rigour requires the `|k|₁` or `|k|∞` norm, not the Euclidean one (Fefferman's 1971 ball-multiplier
   failure).

Notably **absent** from `[AE-II]`: any discussion of Gibbs oscillations, kernel non-positivity, compact
support, or realizability — it treats the sharp filter purely in wavenumber space and never writes its
real-space kernel. Those caveats come from elsewhere: `[AE-I]` Appendix 1 (Bessel-function kernel taking
negative values, decaying "only as a power-law `r^{−p}`"); `[AHV18]` §4c on the sinc kernel ("has very
poor localization in x-space and is more costly to implement and use"); and for the compressible case,
Zhao & Aluie §IV avoid it because it "violates physical realizability since it can have negative values."
Lees & Aluie Table 2 quantifies the practical damage: correlation with the `Λ` model is box 0.93/0.94,
Gaussian 0.97/0.97, **sharp spectral 0.27/0.28**.

`[ZA24]` gives the clean summary of the limiting case: "the filtering spectrum converges to the Fourier
spectrum when using a kernel with an infinite number of vanishing moments (e.g. the Dirichlet kernel),
which is justified only for homogeneous fields given the highly non-local nature of such kernels."

## 6. Kernel width conventions

Two incompatible conventions are both called "a Gaussian of width ℓ":

- `[RAE14]`: `G_ℓ(x) = (π/ℓ²)^{1/2} e^{−π²x²/ℓ²}`, `Ĝ_ℓ(k) = e^{−k²/k_ℓ²}` → **σ = ℓ/(π√2) ≈ 0.225ℓ**,
  chosen so `Ĝ` e-folds exactly at `k_ℓ = 2π/ℓ`.
- `[ZA24]`, Germano 1992, gcm-filters: `(6/πℓ²)^{n/2} e^{−6|x|²/ℓ²}` → **σ = ℓ/(2√3) ≈ 0.289ℓ**, whose
  second moment `ℓ²/12` **matches a top-hat of width `ℓ` exactly**.

That is a **28% difference in effective filter scale for the same nominal `ℓ`.**

In this package's `exp(−α(d/ℓ)²)` parameterization:

| `α` | corresponds to | second moment |
|---|---|---|
| 4 | FlowSieve's shipped default, `exp(−4r²/ℓ²)` | — |
| **6** | the `[ZA24]` / Germano convention | `ℓ²/12`, matching the top-hat exactly |
| `π² ≈ 9.87` | `[AE-I]`'s own DNS Gaussian, `exp(−π²r²/ℓ²)` | `ℓ²/(2π²) ≈ ℓ²/19.74` |

`α = 6` as the default is therefore well-founded: it is the current convention of the group that
developed the framework, and it makes the Gaussian and top-hat directly comparable at equal `ℓ`.

Beware a name collision: FlowSieve's `kernel_alpha` is a *2-D second moment* `⟨r²⟩/ℓ²`, not an exponent.

## 7. Transfer functions, measured

**[measured]** 2-D Hankel transform, `Ĝ` normalized to `Ĝ(0) = 1`, `k` in units of `1/ℓ`:

| kernel | `Ĝ(1)` | `Ĝ(2)` | `k_half` | first zero | max abs sidelobe | `\|Ĝ\|²` monotone ↓ |
|---|---|---|---|---|---|---|
| TopHat | 0.9691 | 0.8801 | 4.43 | 7.67 | 0.132 | **NO** (+0.0026 at `k≈8.8`) |
| HyperGaussian `exp(−D⁴)` | 0.9652 | 0.8665 | 4.26 | 8.10 | 0.056 | NO |
| Gaussian `exp(−D²)` | 0.9394 | 0.7788 | 3.33 | — | 0.000 | **YES** |
| JohnsonGaussian `exp(−D²/8)` | 0.6065 | 0.1353 | 1.18 | — | 0.000 | **YES** |
| SmoothHat `½(1−tanh((D−1)/0.1))` | 0.9678 | 0.8756 | 4.36 | 7.64 | 0.119 | NO |
| HighOrder (see §16.1) | 0.9991 | 0.9869 | 5.82 | 8.21 | 0.190 | NO |

`HighOrder`'s `1 − Ĝ = 𝒪(k⁴)` — 0.0009 at `k=1` versus 0.031/0.061 for top-hat/Gaussian — is the
vanishing second moment visible in Fourier space. It buys large-scale fidelity at the cost of the
*largest* high-`k` sidelobe in the set.

---

# Part II — The energy cascade

## 8. The flux `Π`

### 8.1 Definition and sign

**`Π_ℓ = −(∂_j ū_i) τ̄_ℓ(u_i,u_j) = −S̄_ij τ̄_ij`, with positive meaning forward / downscale.**

The two forms are *identically* equal, not approximations of each other, because `τ̄` is symmetric so its
contraction with the antisymmetric part of `∂_j ū_i` vanishes. `[RAE14]` §IV.B eq. (13) states this
explicitly: "Since the subgrid stress is a symmetric tensor, `τ̄(u_i,u_j) = τ̄(u_j,u_i)`, the SGS flux
`Π_ℓ(x)` can be rewritten in terms in the large-scale symmetric strain tensor".

Confirmed identical across `[AE-I]` (eq. `flux`), `[AE-II]` (eq. 3), Germano 1992 (eq. 25), Meneveau &
Katz 2000, `[AHV18]` (eq. 7, with `ρ₀`), `[A17]` (eq. 36), `[A13]` (eq. 14, Favre), Storer et al. 2023
(eq. 5), Srinivasan et al. 2023 (eq. 4).

Interpretation, `[AE-I]`: `Π̄` "acts as a sink term in (4), representing the energy transferred from
scales larger than ℓ to the smaller (sub-grid) scales at point x."

**Two printed sign errors in the literature**, which are typos rather than convention splits: `[A17]`
§4.5 and eq. (43) print `Π_ℓ = ∇ū_ℓ : τ^u_ℓ` **without the minus**, contradicting its own eqs. (36) and
(49) and Theorem 1 — eq. (36) is authoritative. Srinivasan et al. eq. (3) prints `ῡ_x` where the strain
matrix in eq. (4) requires `ῡ_y`.

### 8.2 Gauge freedom, and why the advective form is rejected

There is a genuine freedom in how to split the large-scale energy equation between a "flux" and a
"transport divergence". `[AE-I]` defines the unsubtracted alternative and rejects it:

```
Π̄^uns ≡ −∂_j ū_i (u_iu_j)‾ = Π̄ − ∂_j(½ū_j|ū|²)
```

> "this definition is not pointwise Galilean-invariant, so that the amount of 'energy cascade' at any
> point in the fluid according to this definition would differ for observers moving at different uniform
> velocities!"

`[AHV18]` names this a **gauge freedom** and fixes the gauge by requiring Galilean invariance. Crucially:
in a *homogeneous* flow all gauge choices share the same spatial mean, so the choice is invisible; in an
**inhomogeneous** domain — which is the case for any real ocean or masked region — they "differ
qualitatively as well as quantitatively."

**Implication**: use the deformation form. If an enstrophy flux is added, use the deformation form there
too; mixing gauges between `Π` and `Z` is internally inconsistent.

### 8.3 The full large-scale budget

`[AE-I]` eq. (4), incompressible and unforced:

```
∂_t(½|ū|²) + ∂_j[ (½|ū|² + p̄)ū_j + ū_i τ̄_ij − ν∂_j(½|ū|²) ] = −Π̄ − ν|∇ū|²
```

`[AHV18]` eq. (6), rotating Boussinesq, which adds the two terms a hydrodynamic budget lacks:

```
∂_t ρ₀|ū_ℓ|²/2 + ∇·J^transport_ℓ = −Π_ℓ − ρ₀ν|∇ū_ℓ|² + ρ̄_ℓ g·ū_ℓ + ρ₀ F̄^forcing_ℓ·ū_ℓ

J^transport_ℓ = ρ₀|ū_ℓ|²/2 ū_ℓ + P̄_ℓ ū_ℓ − ρ₀ν∇(|ū_ℓ|²/2) + ρ₀ ū_ℓ·τ̄_ℓ(u,u)
```

Term inventory with `[AHV18]`'s own naming: (a) advection of large-scale KE by `ū_ℓ`; (b) pressure
transport `P̄ū`; (c) viscous diffusion; (d) **turbulent diffusion** `ū·τ̄`, which "accounts for the role
of motion at scales <ℓ in transporting KE"; (e) cross-scale flux `−Π_ℓ`; (f) direct viscous dissipation,
which "can be shown mathematically to be negligible at scales ℓ≫ℓ_d"; (g) baroclinic conversion
`ρ̄_ℓ g·ū_ℓ`, "conversion from gravitational potential into kinetic energy"; (h) injection by wind/tides.

### 8.4 The complementary small-scale budget

`[AE-I]` eq. (`u-small`) — this is what makes `Π` a *flux* rather than just a sink:

```
∂_t ½τ̄(u_i,u_i) + ∂_j[ ½τ̄(u_i,u_i)ū_j + τ̄(p,u_j) + ½τ̄(u_i,u_i,u_j) − ν∂_j ½τ̄(u_i,u_i) ]
    = +Π̄ − ν τ̄(∂_iu_j,∂_iu_j)
```

`Π̄` appears with the **opposite sign**: what leaves the large scales enters the small scales. Note the
third-order generalized moment `τ̄(f,g,h)`, Germano 1992 eq. (24):

```
τ̄(f,g,h) = ⟨fgh⟩ − ⟨f⟩τ̄(g,h) − ⟨g⟩τ̄(h,f) − ⟨h⟩τ̄(f,g) − ⟨f⟩⟨g⟩⟨h⟩
```

### 8.5 What a `Π`-only diagnostic omits

Under exact homogeneity/periodicity `⟨∇·J⟩ = 0`, so transport drops out of the mean. **That is the only
term `Π`-only gets for free.**

Baroclinic conversion and injection are **not** divergences and do not vanish under averaging. The same
structure appears in MHD: `[A17]`'s conversion term `−(1/4π)B̄_iB̄_j∂_jū_i` cancels only against a
corresponding term in the magnetic budget — "It does not transfer energy across scale ℓ, which is why we
describe it as large-scale conversion … to be distinguished from SGS fluxes … which transfer energy
across scales." So `⟨Π^u⟩` alone never closes the large-scale KE budget in MHD.

**Quantified for the ocean**: Loose et al. 2023 define *relative backscatter* as the ratio of cross-scale
KE transfer to baroclinic EKE production, and report "35%–40%" in the eddy-permitting regime, "70%–100%"
at some latitudes, and ">100%" locally. **The omitted baroclinic term is typically 2.5–3.5× larger than
`Π`.** A `Π`-only diagnostic therefore cannot attribute an EKE change, and cannot distinguish "mesoscale
energy came from upscale transfer" from "it came from baroclinic instability."

### 8.6 `E(ℓ)` — the correct large-scale energy

`[AHV18]`, immediately after eq. (6):

> "what we dub *large-scale KE* is the KE in the large-scale flow, based on `ū_ℓ`, rather than the
> filtered KE itself, `ρ₀|u|²‾/2`, which *does not* cascade across scales."

So `E(ℓ) = ½⟨|ū_ℓ|²⟩` is right; `½⟨(|u|²)‾_ℓ⟩` is a different object and does not obey the cascade
budget.

## 9. `τ` and the Germano identity

### 9.1 Central moments, not Leonard/cross/Reynolds

Germano 1992's requirements on the averaging operator are only **linearity** (eq. 13), **constant
preservation** (14), and **commutation with space/time derivatives** (15) — explicitly *not* the Reynolds
rules (6)–(7). A convolution filter satisfies the first three and violates the Reynolds rules, which is
precisely why the classical decomposition is the wrong tool.

Germano declines it outright: "In this paper we prefer to compare what happens at different levels, and
**there will be no recourse to any kind of decomposition**."

The decomposition that *is* valid is Germano (1986)'s, in which each of the three pieces is itself a
single-level central moment of the form `⟨ab⟩ − ⟨a⟩⟨b⟩` and is therefore **individually Galilean
invariant** — verified by direct substitution `u → u + U₀`, where the `U₀`-linear and `U₀²` terms cancel
identically. Meneveau & Katz 2000 corroborates the history: "In early papers, `C_sim ≈ 1` has been chosen
based on Galilean-invariance arguments (Speziale 1985). However, with the currently standard definition
of Equation 13 which is already Galilean invariant (see Germano 1986), no such limitation exists."

### 9.2 The Germano identity and its multiscale generalization

Germano 1992 eq. (33), relating filtering at two levels `f` and `g`:

```
τ_fg(u_i,u_j) = ⟨τ_f(u_i,u_j)⟩_g + τ_g(⟨u_i⟩_f, ⟨u_j⟩_f)
```

`[AE-I]` Appendix 2 gives the `N`-level generalization, and it is **not** the naive extension:

```
(fg)‾_{N,0} = f̄_{N,0} ḡ_{N,0}
            + Σ_{n=1}^N ( τ_{n-1}( f̄_{N,n}, ḡ_{N,n} ) )‾_{n-2,0}
            + ( τ_N(f,g) )‾_{N-1,0}
```

> "This identity generalizes the well-known Germano identity, to which it reduces when N=1."

The essential feature is the **repeated** low-pass filters `f̄_{m,n} = (G_n * ⋯ * G_m * f)`. Only for
"S-type" kernels (`Ĝ(k) = 1` for `|k| < 1`, `0` for `|k| > ρ`) does `f̄_{N,n} = f̄_n`, collapsing this to
the clean band decomposition, `[AE-I]` eq. (13):

```
½∫|u|² = ½∫|ū₀|² + ½ Σ_{n=1}^N ∫ τ_{n-1}(ū_n; ū_n) + ½∫ τ_N(u;u)
```

with band energies `k_{[ℓ̄,ℓ̃]} ≡ ½τ̃(ū;ū) = ½(|ū|²)~ − ½|ū~|²` and a band budget whose transfer terms are
`−S̃̄_ij τ̃̄(u_i,u_j) + (τ̄(u_i,u_j) S̄_ij)~`.

**The naive alternative is explicitly warned against.** Band-passing the *velocity* gives
`½∫|u|² = ½∫|ū₀|² + ½Σ_{n,m}∫ū_n·ū_m` with "off-diagonal terms of indefinite sign … It is thus not clear
in this approach how precisely to identify the kinetic energy at a given length-scale."

Positivity is load-bearing again here: the band energy `k̄ ≡ ½τ̄(u;u)` "is a positive quantity at every
point in the flow if and only if the filtering kernel `G(r)` is positive for all `r`."

## 10. The filtering spectrum

### 10.1 Definition

`[SA18]` eqs. (14)–(15):

```
Ē(k_ℓ) ≡ (d/dk_ℓ) ⟨|ū_ℓ(x)|²⟩/2  =  −(ℓ²/L)(d/dℓ) ⟨|ū_ℓ(x)|²⟩/2 ,      k_ℓ = L/ℓ
𝓔(k_ℓ) ≡ ½⟨|ū_ℓ(x)|²⟩                                                   (cumulative spectrum)
```

Energy conservation, eqs. (19)–(20):

```
½⟨|u|²⟩ = ½⟨|ū_{ℓ₀}|²⟩ + ∫_{k_{ℓ₀}}^{∞} dk_ℓ Ē(k_ℓ) ,      lim_{k_ℓ→∞} 𝓔(k_ℓ) = ½⟨|u|²⟩
```

Numerical recipe from §Numerical Implementation: `ℓ_m = 2mΔx` for `m = 1,2,…` — "The factor 2 ensures
that even kernels yield vanishing odd moments" — and

```
Ē(k_{ℓ_m}) = [ 𝓔(k_{ℓ_m}) − 𝓔(k_{ℓ_{m-1}}) ] / [ k_{ℓ_m} − k_{ℓ_{m-1}} ]
```

### 10.2 The slope ceiling

`[SA18]` eq. (18) — the one result in this literature that genuinely requires vanishing moments:

```
Ē(k) ~ k^{−α}        if α < p+2
       k^{−(p+2)}    if α > p+2
```

With `p = 1` (top-hat, Gaussian) the measured slope **locks at `k⁻³`**. This is a limitation on the
*spectrum* diagnostic only; `Π` is unaffected. It bites hardest in 2-D / QG work, where the enstrophy-range
target slope *is* ≈ `k⁻³`. `[SA18]`'s own 2-D DNS figure shows the top-hat locking at `k⁻³` while `M^II`
recovers the true slope.

### 10.3 Positive-definiteness

`[SA18]` eq. (21): `Ē(k_ℓ) ≥ 0` is guaranteed if `d|Ĝ(k)|²/dk ≤ 0` on `(0,∞)`. The paper states "both the
gaussian and sharp spectral filters satisfy this condition, **but not the Top-hat kernel**."

This is sufficient, not necessary; the fallback argument invokes concavity of `G`, and the appendix
concedes "the analysis just presented is not a rigorous proof but an argument which relies on significant
approximations." (The Gaussian is not globally concave either — `d²/dx² e^{−ax²} > 0` for `x² > 1/(2a)` —
so the concavity argument is loose for both kernels it is applied to.) When it fails, "it is possible for
`Ē(k_ℓ)` to have negative values."

The `|Ĝ|²`-monotonicity column in §7 is the practical test.

Favre generalization, eqs. (22)–(23): `𝓔^F(k_ℓ) ≡ ½⟨|ρu‾_ℓ|²/ρ̄_ℓ⟩`, `Ē^F = d𝓔^F/dk_ℓ`.

### 10.4 The `k_ℓ = C/ℓ` convention — a real ambiguity

| source | mapping | note |
|---|---|---|
| `[SA18]` eqs. (14), (32) | `k_ℓ = L/ℓ`, `L` = domain size | Fourier convention `f(x)=Σ_k f̂(k)e^{i(2π/L)k·x}`, so `k` is a dimensionless **index**; with `L = 2π` this is `2π/ℓ` in radian units |
| `[RAE14]` | `k_ℓ ≡ 2π/ℓ` | stated outright |
| Storer et al. 2022/2023 | `k_ℓ = 1/ℓ`, Jacobian `−ℓ² d/dℓ` | |
| Loose et al. 2023 | `k_L = π/L` | "we associate the filter scale with only one-half of a wavelength" |
| `[AHV18]` | `K = 2π/ℓ` in one figure; `K = 10⁴/ℓ` km⁻¹ elsewhere | with the disclaimer "**Here, K is not a wavenumber, just a number proportional to ℓ⁻¹**" |

Each paper is internally consistent, and **any** `C` satisfies `∫Ē dk_ℓ = E_total` by the fundamental
theorem of calculus *provided the same `C` appears in the Jacobian and on the abscissa*. But the
**amplitude scales as `1/C`**, and so does the apparent location of a spectral peak. Comparing against a
Fourier spectrum therefore requires matching that transform's convention explicitly.

Related, for spherical work: the commonly quoted degree↔wavelength relations are `λ = 2πR/n` and the
sharper **Jeans rule** `√(n(n+1)) = 2πR/λ`. These were **not** verified against a primary source and
should be treated as unconfirmed. One relation that *was* confirmed, from Stoll et al. 2023, is
`λ = 2πa cos(φ)/n` for **zonal** wavenumber at latitude `φ` — which is zonal-Fourier, not SH degree.
`[BSK+23]`, the one paper that directly compares coarse-graining scales to SH degrees, never states a
conversion.

## 11. The strain / convergence decomposition of `Π`

Srinivasan, Barkan & McWilliams 2023, eq. (10). Start from the horizontal flux and diagonalize `S̄`:

```
δ̄  = ū_x + v̄_y                                    divergence          (rotation invariant)
ᾱ² = (v̄_y − ū_x)² + (v̄_x + ū_y)²                  strain magnitude    (rotation invariant)
σ̄_n = ū_x − v̄_y      normal strain
σ̄_s = ū_y + v̄_x      shear strain

[S̄]_principal = diag( (δ̄+ᾱ)/2 , (δ̄−ᾱ)/2 )
```

giving

```
Π_h = (τ_vv − τ_uu)·σ̄_n/2  −  τ_uv·σ̄_s  −  (τ_vv + τ_uu)·δ̄/2
      └────────── Π_α : deformation shear production ──────────┘  └─ −Π_δ : convergence production ─┘
```

Equivalently `Π_h = E′ γᵖ ᾱ − E′ δ̄` with `E′ = (τ_vv+τ_uu)/2` and
`γᵖ = (τᵖ_vv − τᵖ_uu)/(τᵖ_vv + τᵖ_uu)`, whence the bound `−ᾱE′ ≤ Π_α ≤ ᾱE′` since `−1 ≤ γᵖ ≤ 1`.

**[verified]** Expanding eq. (10) collapses exactly to `−τ_uu ū_x − τ_uv(ū_y + v̄_x) − τ_vv v̄_y`, i.e. the
direct form. So the two are algebraically identical and **must** agree numerically — which makes this a
strong correctness test rather than a tautology, because the two paths use different combinations of
derivatives and are reported to disagree in at least one existing implementation, "primarily at scales
where Pi(scales) > 0 (forward KE cascade)", with differentiation order suspected.

`Π_δ` (convergence production) is new in that paper; eq. (10) with `δ = 0` recovers Polzin 2010, and the
`γᵖ` form with `δ = 0` recovers Jing et al. 2017.

The buoyancy-gradient analogue, eq. (16), has the same structure:

```
B_h = (b_y² − b_x²)σ_n/2 − b_x b_y σ_s − (b_y² + b_x²)δ/2
```

## 12. Related fluxes

### 12.1 Enstrophy

`[RAE14]` eqs. (10), (14), (16):

```
τ̄_ℓ(u,ω) = (uω)‾_ℓ − ū_ℓ ω̄_ℓ
Z_ℓ(x)    = −∂_j ω̄ · τ̄(ω,u_j) = −∇ω̄_ℓ · τ̄_ℓ(u,ω)
∂_t |ω̄_ℓ|²/2 + ∇·J^Ω_ℓ = −Z_ℓ − ν|∇ω̄_ℓ|² − α|ω̄_ℓ|²
```

Use the **deformation form**, matching `Π`'s gauge (§8.2).

### 12.2 Tracer variance

Aluie & Kurien 2011, eq. (8) — the correct citation for a tracer-variance flux:

```
Π_ℓ(x) = −∂_j ū_i τ̄(u_i,u_j)  −  ∂_j θ̄ τ̄(θ,u_j)
  ⇒  Π^θ_ℓ = −τ̄(θ,u_j) ∂_j θ̄
     J^θ_j  = ū_j|θ̄|²/2 + θ̄ τ̄(θ,u_j) − κ ∂_j|θ̄|²/2
```

and eq. (12) for potential enstrophy: `Π^Q_ℓ(x) = −∂_j q̄ τ̄(q,u_j)`.

---

# Part III — Compressible flow

## 13. Favre filtering

`[A13]` eq. (9): `f̃_ℓ(x) ≡ (ρf)‾_ℓ(x) / ρ̄_ℓ(x)`. "The operator (~·) is linear but **does not commute
with derivatives**."

Stress carries an explicit `ρ̄`, eq. (11): `ρ̄ τ̃(u_i,u_j) ≡ ρ̄(ũ_iu_j − ũ_iũ_j)`. Favre continuity closes
exactly, eq. (12): `∂_tρ̄ + ∂_i(ρ̄ũ_i) = 0` — this is the property that motivates the whole construction.

Budget, eq. (13) with terms (14)–(18):

```
∂_t ρ̄_ℓ|ũ_ℓ|²/2 + ∇·J_ℓ = −Π_ℓ − Λ_ℓ + P̄_ℓ ∇·ū_ℓ − D_ℓ + ε^inj_ℓ

Π_ℓ      = −ρ̄ ∂_j ũ_i τ̃(u_i,u_j)                                deformation work
Λ_ℓ      =  (1/ρ̄) ∂_j P̄ · τ̄(ρ,u_j)                               baropycnal work
D_ℓ      =  ∂_j ũ_i [ 2(μS_ij)‾ − (2/d)(μS_kk)‾ δ_ij ]
J_j      =  ρ̄(|ũ|²/2)ũ_j + P̄ū_j + ũ_i ρ̄ τ̃(u_i,u_j) − ũ_i σ̄_ij
ε^inj_ℓ  =  ũ_i ρ̄ F̃_i
```

Note `τ̄(ρ,u_j) ≡ (ρu_j)‾_ℓ − ρ̄_ℓ ū_{j,ℓ}` is the **unweighted** subscale mass flux: the framework needs
*both* moment types simultaneously.

**Physical distinction**, `[A13]` §4.2: "both deformation work, `Π_ℓ`, and baropycnal work, `Λ_ℓ`, involve
**large-scale** fields acting against **small-scale** fluctuations. This makes them capable of
transferring energy **across** scales. On the other hand, large-scale pressure dilatation, `P̄_ℓ∇·ū_ℓ`,
involves only large-scales and cannot transfer energy directly across scales."

**The main implementation trap**, `[A13]` §4.2: `Λ` "has often been lumped with `P̄_ℓ∇·ū_ℓ` in the form
`P̄_ℓ∇·ũ_ℓ` (plus an additional space-transport term) and treated as a large-scale pressure dilatation
which does not require modeling." Eq. (13) uses `∇·ū_ℓ`, **unweighted**. Writing `P̄∇·ũ` silently absorbs
and destroys `Λ`.

Closed form for `Λ`, Lees & Aluie 2019 eqs. (16)–(18), with `C₂ ≡ ∫d³r G(r)|r|²`:

```
Λ ≈ (1/3) C₂ ℓ² (1/ρ̄) [ ∇P̄·S̄·∇ρ̄ + ½ ω̄·(∇ρ̄ × ∇P̄) ] = Λ_SR + Λ_BC
```

— a strain-generation part and a **baroclinic** part. Note `C₂` is the kernel's second moment, which is
another reason it must be non-zero (§2.2).

## 14. Why an incompressible-style stress is wrong here

Three documented failures of using `τ̄_ij = (u_iu_j)‾ − ū_iū_j` in variable-density flow:

1. **The inviscid criterion fails.** `[A13]` §3.1: with large-scale momentum `ρ̄_ℓū_ℓ`, "a viscous term …
   would have the form `μρ̄_ℓ(ρ^{-1}∇²u)‾_ℓ`. Here, again, the filtering operation would not commute with
   the laplacian and, due to possibly dominant contributions from high wavenumber modes ≫ ℓ⁻¹, we would
   not be able to guarantee a priori that viscous terms are negligible at large ℓ." The escape requires
   "a density spectrum decaying faster than k⁻³, which is physically unrealistic."
2. **Loss of Galilean invariance of the dissipation.** Zhao & Aluie 2018 §III: "`Σ_ℓ^{F,diss}` is Galilean
   invariant for any ℓ, whereas `Σ_ℓ^{C,diss}` and `Σ_ℓ^{K,diss}` are not."
3. **Measured severity is catastrophic, not marginal.** The non-Favre viscous terms "are several orders of
   magnitude larger than `‖Σ_ℓ^{F,diss}‖₁`, precluding inertial dynamics at those 'length-scales'"; in
   high-contrast 3-D Rayleigh–Taylor they "do not decay at large scales, in violation of the inviscid
   criterion."

Uniqueness is **not** claimed: `[A13]` §3.2 — "we did not prove that it is necessary. We only showed that
the Favre decomposition is sufficient."

---

# Part IV — Bounded and curved domains

## 15. Masked domains

This is the least settled area in the subject, and the literature is unanimous against normalized
convolution.

### 15.1 The three policies

| policy | what it does | preserves | breaks |
|---|---|---|---|
| **zero-fill land** (`[AHV18]`, `[BSK+23]`, FlowSieve default) | treat land as zero-velocity fluid; integrate over all cells | commutation with `∇`, hence the whole budget derivation | field magnitude near coasts |
| **renormalize over water** ("deformed kernel") | divide by the filtered mask | constants exactly | commutation; domain-average conservation |
| **modified no-flux operator** (gcm-filters) | zero the edge fluxes in a discrete Laplacian | area integral exactly (to 1e−16) | effective scale near coasts |

### 15.2 What the papers say

`[AHV18]`, §"Difficulties with our approach", verbatim:

> "A practical choice made in this work is to treat land as water with zero velocity." … "coarse-grained
> velocity `ū_ℓ` can be nonzero within a distance ℓ/2 beyond the continental boundary over land.
> Therefore, terms in the large-scale energy budget (6), such as `Π_ℓ` and `J_ℓ^transport`, **are only
> guaranteed to be zero over land a distance ℓ/2 beyond the boundary**." … "**The alternative choice is
> to make the filter kernel change shape as it approaches the boundary … but such a filtering operation
> will no longer commute with spatial derivatives.** … **This would prevent us from deriving the
> large-scale energy budget (6)** … we leave the filter independent of its proximity to the boundary."

`[BSK+23]` names the operation and refutes it with an exact matrix argument:

> "Such a deformed kernel **must be renormalized to yield an average over just ocean points** rather than
> the whole sphere."
>
> "such inhomogeneous kernels (which also include averaging values at adjacent grid-cells or
> block-averaging on the sphere) **do not commute with spatial derivatives**. Consequently, the
> coarse-grained field resulting from a deformed kernel is not guaranteed to satisfy fundamental flow
> properties … such as non-divergence, geostrophy, and the vorticity present at various scales."
>
> "a kernel that is inhomogeneous … **does not conserve domain averages, including the kinetic energy of
> the flow** … it can yield total energy that is either less than 100% … or **greater than 100%**."

**The mechanism**, their Appendix A: for a 5-point 1-D renormalized kernel, `S·G = [11/12, 13/12, 1,
13/12, 11/12] ≠ S`, "and so in general `S·KE‾ ≠ S·KE`. Moreover, there is no guarantee that
`S·KE‾ ≤ S·KE`." Renormalizing the **rows** of the convolution matrix to sum to 1 does not make its
**columns** sum to 1 — and it is column sums that give measure preservation. Their conclusion: "**To
fully conserve energy and maintain commutativity with differentiation, we choose the 'Fixed Kernel w/
Land' option**, which treats land as zero-velocity water and includes land cells in spatial integrals."

The counter-consideration, Grooms et al. 2021 §3.3: "Setting the velocity to zero over land allows the
filter to commute with derivatives, but at the cost of reducing the strength of currents near land. For
example, the Florida Current is much weaker … It is thus clear that both methods have pros and cons near
boundaries."

### 15.3 Measured coast artifacts

**[measured]** reproducing FlowSieve's exact discrete filter — Gaussian, `ℓ = 16` cells, straight coast,
uniform ocean field `u ≡ 1`:

| cells from coast | `ū` zero-fill | `ū` renormalized | kernel centroid offset | `m₂ / m₂(interior)` |
|---|---|---|---|---|
| 1 | **0.535** | 1.000 | +4.21 cells (0.26ℓ) | **0.38** |
| 2 | 0.605 | 1.000 | +3.61 | 0.42 |
| 4 | 0.732 | 1.000 | +2.55 | 0.52 |
| 8 | 0.908 | 1.000 | +1.03 | 0.72 |
| 16 (= ℓ) | 0.997 | 1.000 | +0.05 | 0.97 |
| 24 | 0.99998 | 1.000 | +0.0004 | 1.00 |

Zero-fill suppresses coarse KE by **46% one cell from a coast**, recovering to 0.3% only at `~ℓ`.
Renormalization restores constants exactly, but the effective kernel is still one-sided: centroid
displaced `0.26ℓ` offshore, **effective filter width reduced to ~62% of intended**.

**[measured]** gcm-filters' no-flux operator conserves the area integral to `1.5e−16`, but at the coast
`m₂` falls to **37%** of interior with the centroid displaced +1.365 cells — i.e. the effective scale
shrinks by ~40% there too. Its influence zone is exactly `n_steps` cells.

### 15.4 Other masking facts

- **Area convention with zero-fill**: sum over **all** cells including land, divide by **water** area
  (Storer et al. 2022 Methods). Grooms et al.: "to preserve the integral with this method, the integral
  has to be extended over land."
- **Leakage onto land**, `[BSK+23]`: "energy leakage of the order of 1% at coarse-graining scales
  <100 km, ≈4% for scales ≲500 km, and up to 12% at scales of order 2000 km."
- **A quantified exclusion buffer exists.** `[RAE14]` Appendix A measures rms flux error against distance
  from an artificial boundary and concludes: "we have considered a maximum error rate of 10⁻⁴ as
  acceptable … this stipulates that we restrict our analysis to flow locations **at least a distance
  ≈ ℓ(n−1) from the domain boundary**, where n is the localization order". For a Gaussian `n = 2`, i.e. a
  buffer of `≈ℓ`; less localized kernels need more.
- **Never renormalize separably.** A per-1-D-pass renormalization is **not** equal to a 2-D masked
  normalized convolution unless the mask itself is separable. The correct construction is to run the full
  separable convolution *on the mask* and divide by that.
- **Non-negative kernels are needed for realizability** independent of commutation: a kernel with negative
  lobes can produce a negative local variance `f²‾ − f̄²`.
- **Tracers are harder than velocity.** Zero-fill is boundary-condition-consistent for velocity under
  no-slip, but for tracers "there remains an infinite family of possible extensions that all satisfy the
  boundary conditions." Storer, Kharghani, Adcroft & Aluie (2025) propose solving a Laplace BVP over land.
- **Empirically the policies differ modestly** in the resulting spectra — the argument against
  renormalization is **structural** (no commutation ⇒ no budget; sign-indefinite conservation error), not
  a matter of magnitude.

## 16. The sphere

- **`ℓ` is a great-circle *diameter*.** The convention is `D = γ/(ℓ/2)` with `γ` the great-circle
  distance, so a top-hat is a spherical cap of great-circle *radius* `ℓ/2`. `[AHV18]` eq. (2) gives the
  cap area `A = 2πR²[1 − cos(ℓ/2R)]`.
- **Commutation requires a zonal kernel.** `[A19]` Proposition 1 requires the kernel be an *x-zonal*
  function — a function of geodesic distance from `x` alone — and `G ∈ L^p[−1,1]`. **No moment conditions.**
  Normalization is needed separately for mean preservation, not for commutation. The spherical convolution
  theorem is `[A19]` eq. (37), with Legendre transform `Ĝ(n) = 2πr²∫P_n(t)G(t)dt`.
- **Radial derivatives commute for free.** `[A19]` Corollary 1 extends commutation to `∇f‾=∇f̄`,
  `∇×u‾=∇×ū`, `∇·u‾=∇·ū`, `Δu‾=Δū`, `∇·T‾=∇·T̄`, with the proof: "The proof is a simple consequence of
  our filtering operation not acting in the radial direction. Therefore, it commutes with radial
  derivatives." **This is what justifies the layered / 2.5-D approach** — with respect to commutation it
  is exact, not an approximation.
- **Vectors are not filtered component-wise, strictly.** `[A19]`'s generalized vector convolution uses
  *spectrally shifted* kernels `G^{(1)} = LT⁻¹{Ĝ(n−1)}` and `G^{(2)} = LT⁻¹{Ĝ(n+1)}` (eqs. 59–61,
  122–126), yielding `∇*·ū = ∇*·u‾` etc. (eqs. 148–162). Filtering the 3-D **Cartesian** components
  instead is valid **only for non-divergent vectors**; per `[BSK+23]` summarizing Aluie & Teeraratkul
  (2023), "Significant errors can arise for a general flow field … where the complete coarse-graining
  formalism of Aluie (2019) is necessary." Converting spherical components to planetary Cartesian before
  filtering is strictly better than filtering spherical components directly, but is **not** the full
  formalism.
- **`[A19]` gives no explicit worked kernel** — the concrete forms come from `[AHV18]` (spherical cap) and
  Storer et al. (graded tanh).

### 16.1 What the layered (2.5-D) approach drops

Only the vertical-shear part of `Π`. Srinivasan et al. 2023 eq. (3):

```
Π = −[τ_uu ū_x + τ_uv(ū_y + ῡ_x) + τ_υυ ῡ_y]  −  (τ_uw ū_z + τ_υw ῡ_z)
       └────────────── Π_h ──────────────┘        └────── Π_υ ──────┘
```

Their justification, the only quantified one located: "we neglect the vertical shear terms in both cases,
which is justified in the scaling analysis of Barkan et al. (2019), supported by our model analysis; in
particular Π_z … is on average about 5 times smaller than Π_h". (Their Fig. 14 caption says **4** — an
internal inconsistency in that paper.)

Storer et al. 2023 justify dropping vertical velocity entirely for hydrostatic ocean flow: "We conducted
identical analysis that included radial/vertical velocities diagnosed using flow incompressibility, and
found that including the radial velocity `u_r` has a negligible impact on both diagnostics across all
scales analyzed here."

**Layered ≠ isopycnal.** Loose et al. 2023 filter *along isopycnals* with a spatially varying filter
scale, which genuinely does break horizontal commutation, and they carry an explicit extra term for it.
Their quantification of why that matters: "if we commuted our spatial filter with ∇ in the baroclinic
conversions … we would obtain baroclinic conversions that integrate to **34 GW less** than the values
shown … Accounting for noncommutativity of filter and derivatives … is therefore crucial."

---

# Part V — Practice

## 17. Reference kernel set

The de-facto standard set, all in terms of `D = 2r/ℓ` (so `ℓ` is a diameter):

| name | form | `≥ 0` | notes |
|---|---|---|---|
| TopHat | `1` for `D<1`, else `0` | yes | `[AHV18]`'s kernel; fails the spectrum positivity condition; ill-defined `ℓ`-derivatives |
| HyperGaussian | `exp(−D⁴)` | yes | |
| Gaussian | `exp(−D²)` | yes | `α = 4` in `exp(−α(r/ℓ)²)`; satisfies spectrum positivity |
| JohnsonGaussian | `exp(−D²/8)` | yes | very wide; satisfies spectrum positivity |
| Sinc | `sinc(πD)` | no | sharp-spectral real-space form |
| SmoothHat | `½(1 − tanh((D−1)/0.1))` | yes | **this is the published Storer et al. kernel**, `G_ℓ(γ) = (A/2)[1 − tanh(10(γ/(ℓ/2) − 1))]` |
| HighOrder | `SmoothHat − c₂·exp(−((D−1)/0.5)²)`, `c₂ = 0.21534029041474162` | **no** | tuned so the 2-D second moment vanishes |

**[measured]** 2-D radial moments `∫k(D)D^{n+1}dD`, confirming `HighOrder`'s construction:

| kernel | `M₀` | `M₂` | `α = M₂/M₀` | negative range |
|---|---|---|---|---|
| TopHat | 0.5000 | 0.2500 | 0.5000 | — |
| Gaussian | 0.5000 | 0.5000 | 1.0000 | — |
| SmoothHat | 0.5041 | 0.2624 | 0.5205 | — |
| **HighOrder** | 0.3132 | **1.1e−15** | **≈0** | `D ∈ [1.066, 3.554]`, min −0.166 |

So `HighOrder` achieves `M^I`'s goal (`p = 3`, `M₄ ≠ 0`) by a completely different construction — a tanh
taper minus a Gaussian bump rather than a piecewise stencil. **[measured]** truncating it at `D = 2.5`
(mid-negative-lobe) leaves a residual `α = 1.15e−4` rather than 0; it needs a truncation radius `≥ 3.5D`
to be exact. Every other kernel's truncation error is `≤ 1e−11` of peak.

## 18. Validation ladder

In rough order of what would catch the most, each from a published validation.

1. **Synthetic spectra with prescribed slopes** (`[SA18]`) — this is the sharpest single gate:
   - `k⁻⁵ᐟ³`: top-hat and Gaussian both correct.
   - `k⁻⁴`: top-hat and Gaussian **lock at `k⁻³`**; `M^I`, `M^II` recover `k⁻⁴`.
   - `k⁻⁷`: top-hat locks at `k⁻³`, **`M^I` locks at `k⁻⁵`**, only `M^II` recovers `k⁻⁷`.
   Cumulative spectra must conserve energy for all kernels. A direct gate on eq. (18).
2. **Pure large-scale strain `u = (y, x)`** — the analytic no-small-scales benchmark. A correct
   high-order result gives "almost zero energy at small-scales", and the `ℓ = L/3` vs `L/6` KE fields
   "look almost identical". A mirrored-subdomain Fourier estimate instead produces a spurious `k⁻⁴` law
   "almost 12 orders of magnitude larger", which is the point of the test.
3. **Filter of a constant** returns that constant — partition of unity — including adjacent to a mask
   boundary, under whichever masking policy is chosen.
4. **`filter ∘ div = div ∘ filter`** and `filter ∘ grad = grad ∘ filter` on a divergence-free field. This
   is the `[A19]` property and exactly what a deformed/renormalized kernel breaks.
5. **Solid-body rotation invariance** (`u = cos(lat)`, `v = 0`) — a corollary of conserving angular
   momentum.
6. **Both forms of `Π` agree** — direct `−S̄:τ̄` versus the §11 strain/convergence decomposition.
7. **Enstrophy flux against the third-order relation** `S^W₂(r) = −2ηr` at `r = 2ℓ` (`[RAE14]` Fig. 7),
   which they report agreeing to within ≈3/4. The closest thing in this literature to a ground-truth flux
   check.
8. **Kernel moment gates** — `∫G = 1` exactly; `∫xG = 0`; `∫x²G` finite, non-zero, and equal to the
   documented value for the chosen `α`; and for any high-order kernel, `∫x²G ≈ 0` **after discretization**
   with the truncation radius checked per §4.3.

## 19. Open questions and unverified items

Recorded so they are not silently treated as settled.

- **The `M^I`/`M^II` coefficients were transcribed from the arXiv LaTeX source, not the journal PDF.** One
  transcription error was found elsewhere in the same corpus (the `[ZA24]` `x₀` typo, §3.3), so verify
  against the PDF before implementing.
- **Whether `[SA18]` addresses the top-hat's non-monotone `|Ĝ|²`** against its own eq. (21) condition —
  not found either way in the text read. The measurement in §7 shows the top-hat does violate it.
- **Whether Storer et al. state the graded-tanh kernel's order or max extractable slope** — not found.
  Measurement says it has no vanishing moment beyond the trivial first one, and violates the positivity
  condition.
- **`[AHV18]` figure filenames encode `Tanh10`**, suggesting the production runs used a tanh-graded
  top-hat with sharpness 10, while the text says top-hat throughout. Only one arXiv version exists, so
  there is no revision history to compare. Unresolved.
- **Whether `[AHV18]` evaluated `Π_ℓ` with the full 3-D `τ̄` or only the horizontal 2×2 block** — eq. (7)
  is written with the full contraction, but "vertical velocity", "w", and `τ_iz` appear nowhere in the
  paper.
- **Whether zero-filling interacts badly with `∂/∂z` over sloping topography.** `[A19]` Cor. 1 proves the
  *operator* commutes with radial derivatives, and masking does not touch that proof. But zero-fill
  modifies the *field*, and the water mask changes with depth, so `∂_z` of the zero-filled field need not
  equal the zero-filled `∂_z` near sloping bathymetry. No paper found that addresses this.
- **The degree↔wavelength relation on the sphere** (`λ = 2πR/n`, or the Jeans rule `√(n(n+1)) = 2πR/λ`)
  was not verified against a primary source.
- **Whether the original Leonard (1974) L/C/R terms are individually non-Galilean-invariant** — the
  Germano (1986) form was verified by direct substitution, but the pre-1986 claim is attributed to
  Speziale 1985 via Meneveau & Katz, not read directly.

---

## References

- `[AE-I]` Eyink & Aluie 2009, "Localness of energy cascade in hydrodynamic turbulence, I. Smooth
  coarse-graining", *Phys. Fluids* 21, 115107. arXiv:0909.2386, doi:10.1063/1.3266883
- `[AE-II]` Aluie & Eyink 2009, "… II. Sharp spectral filter", *Phys. Fluids* 21, 115108. arXiv:0909.2451
- `[SA18]` Sadek & Aluie 2018, "Extracting the spectrum of a flow by spatial filtering", *Phys. Rev.
  Fluids* 3, 124610. arXiv:1811.08259
- `[A13]` Aluie 2013, "Scale decomposition in compressible turbulence", *Physica D* 247, 54.
  arXiv:1012.5877
- `[A17]` Aluie 2017, "Coarse-grained incompressible magnetohydrodynamics: analyzing the turbulent
  cascades", *New J. Phys.* 19, 025008. doi:10.1088/1367-2630/aa5d2f, arXiv:1701.08692
- `[AHV18]` Aluie, Hecht & Vallis 2018, "Mapping the energy cascade in the North Atlantic Ocean: The
  coarse-graining approach", *JPO* 48, 225–244. arXiv:1710.07963
- `[A19]` Aluie 2019, "Convolutions on the sphere: commuting with differential operators", *GEM* 10:9.
  arXiv:1808.03323
- `[RAE14]` Rivera, Aluie & Ecke 2014, "The direct enstrophy cascade of two-dimensional soap film flows",
  *Phys. Fluids* 26, 055105. arXiv:1309.4894
- `[BSK+23]` Buzzicotti, Storer, Khatri, Griffies & Aluie 2023, *JAMES*. arXiv:2106.04157
- `[ZA24]` Zhao & Aluie 2024. arXiv:2412.11891
- Zhao & Aluie 2018, *Phys. Rev. Fluids* 3, 054603. arXiv:1804.07715
- Lees & Aluie 2019, "Baropycnal work: A mechanism for energy transfer across scales", *Fluids* 4, 92.
  arXiv:1905.03581
- Aluie & Kurien 2011, arXiv:1107.5006 — tracer variance and potential-enstrophy flux
- Germano 1992, "Turbulence: the filtering approach", *JFM* 238, 325
- Germano 1986 — the Galilean-invariant three-way stress decomposition
- Vreman, Geurts & Kuerten 1994, "Realizability conditions for the turbulent stress tensor in LES",
  *JFM* 278, 351
- Meneveau & Katz 2000, "Scale-invariance and turbulence models for large-eddy simulation", *Annu. Rev.
  Fluid Mech.* 32, 1
- Srinivasan, Barkan & McWilliams 2023, "A forward energy flux at submesoscales driven by
  frontogenesis", *JPO* 53(1), 287–305. doi:10.1175/JPO-D-22-0001.1
- Barkan, Srinivasan & McWilliams 2024, "Eddy–internal wave interactions: stimulated cascades in
  cross-scale kinetic energy and enstrophy fluxes", *JPO* 54(6), 1309–1326. doi:10.1175/JPO-D-23-0191.1
- Storer, Buzzicotti, Khatri, Griffies & Aluie 2022, *Nat. Commun.* arXiv:2208.04859
- Storer et al. 2023, *Sci. Adv.* arXiv:2311.09100
- Storer, Kharghani, Adcroft & Aluie 2025, arXiv:2512.03051 — harmonic extension of fields over land
- Grooms, Loose, Abernathey, Steinberg, Bachman, Marques, Guillaumin & Yankovsky 2021, "Diffusion-based
  smoothers for spatial filtering of gridded geophysical data", *JAMES* 13
- Loose, Bachman, Grooms & Jansen 2023, "Diagnosing scale-dependent energy cycles in a high-resolution
  isopycnal ocean model", *JPO* 53, 157
- Aluie & Teeraratkul 2023 — vector coarse-graining on the sphere for general (divergent) fields
