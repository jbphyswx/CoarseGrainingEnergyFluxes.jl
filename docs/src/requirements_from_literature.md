# Requirements from the literature

What the coarse-graining framework actually demands of an implementation, with the equations and
citations needed to check the code against it. Written so that no claim here has to be re-derived or
re-looked-up: every requirement states its source, and every number states whether it is quoted from a
paper or measured.

Conventions used below: `ℓ` is the filter scale, `G_ℓ` the kernel, `ū_ℓ = G_ℓ * u` the filtered field,
`τ̄_ℓ(f,g) = (fg)‾_ℓ − f̄_ℓ ḡ_ℓ` the generalized second moment, `S̄` the filtered strain rate.

Sources are cited as `[AE-I]` etc. and listed in full at the bottom.

---

## 1. What a kernel must satisfy

### 1.1 The flux framework's requirements

`[AE-I]` §II.A, verbatim:

> "Any real-valued function G(**r**) may be chosen as a filter kernel, as long as it is sufficiently
> smooth, decays sufficiently rapidly for large r, and is normalized so that ∫d^d r G(**r**)=1. It is
> assumed furthermore that G satisfies ∫d^d r **r** G(**r**)=**0** and ∫d^d r |**r**|²G(**r**)=𝒪(1),
> so that the main support of G is in the ball of radius 1 about the origin."

> "In order to interpret this as a local space-average, G must also be positive, or G(**r**) ≥ 0 for
> all **r**."

So the requirements are:

| property | required? | what depends on it |
|---|---|---|
| `∫G = 1` | yes | everything; `Ĝ(0) = 1`, mean preserved |
| `∫**r**G = 0` (first moment) | yes | filter is centered; lets `∇ū`, `τ̄`, `u'` be written via velocity increments `δu(x,r)`, which is what removes sweeping/Galilean contamination and yields the locality bounds |
| `∫\|**r**\|²G = 𝒪(1)` | yes, **non-zero** | this *defines* `ℓ`. Nothing fails if it is non-zero — it is required to be non-zero |
| `G ≥ 0` (positivity) | yes | (i) interpretation of `ū` as a local space average; (ii) **pointwise positivity of subscale KE**, an iff |
| smoothness | yes | `[A17]` Prop. 1 (inertial-range inviscid dynamics) needs `‖∇²G‖₂ < ∞`; Prop. 2 (vanishing direct viscous dissipation at large scale) needs `‖∇G‖₂ < ∞` |
| compact support | **no** | rapid decay suffices; compact support only extends Prop. 1 from `𝕋³` to `ℝ³` |
| radial symmetry | no | "often convenient", explicitly optional |

The positivity result, `[AE-I]` §II.A verbatim — the sharpest statement in this literature on kernel
sign:

> "It is a positive quantity at every point in the flow if and only if the filtering kernel G(**r**) is
> positive for all **r**, as was proved by Vreman et al. (1994)."

`[A17]` §2.2 restates the whole set: "any kernel function G which is smooth, rapidly decaying, positive
and with integral normalized to unity … We usually assume also that the kernel is centered … and that
it has variance of order unity, ∫d^d r |r|² G(r) ≈ 1", with the hedge "Some of these requirements (e.g.
smoothness, positivity) may be relaxed in some situations".

**Note the tension inside the literature**: `[A17]` requires smoothness (its Props. 1–2 need `‖∇G‖₂`
and `‖∇²G‖₂` finite, both of which diverge for a top-hat), while `[AHV18]`'s working kernel is a
discontinuous top-hat. The hedge above is the only bridge offered.

### 1.2 Kernel order, and the conflict with positivity

`[SA18]` eq. (11), verbatim:

> "We shall call a kernel G(x) ``p-th order'' iff ∫dx x^n G(x) = 0 for n=1,…,p, and ∫dx x^{p+1} G(x) ≠ 0."

`n = 0` is **excluded** (it equals 1 by normalization) — this differs from the wavelet convention, where
`∫x^n ψ = 0` for `n = 0,…,p−1`. Because odd moments vanish automatically for an even kernel, "Any even
kernel is of an odd integer order p ≥ 1", so `p` rises 1 → 3 → 5. Equivalently `Ĝ(k) = 1 + k^{p+1}φ(k)`
with `φ(0) ≠ 0`.

**Top-hat and Gaussian are `p = 1`.**

Vanishing moments and positivity are **mutually exclusive**. `[AE-I]` Appendix 1, verbatim:

> "Note, however, that filter kernels G of ``S-type'' cannot be non-negative, since ∫d^d r |**r**|^{2n}
> G(**r**) = ((−△)^n Ĝ)(**k**)|_{**k**=0} = 0 for all positive integers n. In that respect, they are
> similar to the sharp spectral filter, whose physical-space kernel is proportional to a Bessel function
> that also takes on negative values."

`[SA18]` §V says the same from the other side: "any kernel of order higher than unity cannot be positive
everywhere in x-space, otherwise even moments would not vanish."

**Consequence for this package**: raising kernel order forfeits the Vreman realizability iff and the
local-space-average interpretation. A positive `p = 1` kernel is the *correct* choice for flux, not a
deficiency. `[AE-I]`'s own DNS uses a Gaussian; `[AHV18]` uses a top-hat.

### 1.3 Where kernel order *does* matter: the filtering spectrum

`[SA18]` eq. (18) — the only result found in this literature that requires vanishing moments:

```
Ē(k) ~ k^{-α}       if α < p+2
       k^{-(p+2)}   if α > p+2
```

With `p = 1` the measured slope **locks at k⁻³**. This is a limit on the *spectrum* diagnostic, not on
`Π`. It matters specifically for 2-D / QG work, where the enstrophy-range target slope *is* ≈ k⁻³.

`[SA18]`'s own 2-D DNS figure shows the top-hat locking at k⁻³ while `M^II` recovers the true slope.

### 1.4 Positive-definiteness of the filtering spectrum

`[SA18]` eq. (21): `Ē(k_ℓ) ≥ 0` is guaranteed if `d|Ĝ(k)|²/dk ≤ 0` on `(0,∞)`. The paper states "both
the gaussian and sharp spectral filters satisfy this condition, **but not the Top-hat kernel**".

Sufficient, not necessary; the fallback argument invokes concavity, and `[SA18]`'s appendix concedes
"the analysis just presented is not a rigorous proof but an argument which relies on significant
approximations". (The Gaussian is not globally concave either, so the concavity argument is loose for
both kernels it is applied to.)

**Measured** (2-D Hankel transform, `Ĝ` normalized to `Ĝ(0)=1`, `k` in `1/ℓ`), for the FlowSieve kernel
set — `|Ĝ|²` monotone decreasing?

| kernel | `Ĝ(k=1)` | first zero | max abs side lobe | monotone |
|---|---|---|---|---|
| TopHat | 0.9691 | 7.67 | 0.132 | **NO** (+0.0026 at k≈8.8) |
| HyperGaussian `exp(−D⁴)` | 0.9652 | 8.10 | 0.056 | NO |
| Gaussian `exp(−D²)` | 0.9394 | — | 0.000 | **YES** |
| JohnsonGaussian `exp(−D²/8)` | 0.6065 | — | 0.000 | **YES** |
| SmoothHat | 0.9678 | 7.64 | 0.119 | NO |
| HighOrder | 0.9991 | 8.21 | 0.190 | NO |

### 1.5 `M^I` and `M^II` — exact definitions

`[SA18]` §V, "Constructing Simple Stencil Kernels of Higher Order". Both are symmetric, piecewise
constant, compactly supported, with `b` a **free** parameter and `ℓ` the width of the main body.

**`M^I`** — order `p = 3`. Body `c` on `|x| < ℓ/2`; legs `−a` on `ℓ/2 < |x| < ℓ/2 + b`. Support
`|x| < ℓ/2 + b`. Eq. (34):

```
a/c = 1/((1 + 2b/ℓ)³ − 1)          from ∫dx x² M^I(x) = 0
cℓ − 2ab = 1                        normalization
```

**`M^II`** — order `p = 5`. Adds arms `+e` on `ℓ/2 + b < |x| < ℓ/2 + 2b` (paper sets `d = b`). Support
`|x| < ℓ/2 + 2b`. Eq. (35):

```
a/c = (124b³ℓ³ + 88b²ℓ⁴ + 19bℓ⁵ + ℓ⁶) / (4b²(192b⁴ + 400b³ℓ + 340b²ℓ² + 120bℓ³ + 15ℓ⁴))
e/c = (  4b³ℓ³ +  8b²ℓ⁴ +  5bℓ⁵ + ℓ⁶) / (4b²(192b⁴ + 400b³ℓ + 340b²ℓ² + 120bℓ³ + 15ℓ⁴))
cℓ − 2ab + 2eb = 1
```

At the paper's choice `b = ℓ/8`, fully reduced:

| kernel | region | value | support |
|---|---|---|---|
| `M^I` | `\|x\| < ℓ/2` | `+61/(45ℓ)` ≈ `+1.355556/ℓ` | `\|x\| < 5ℓ/8` |
| | `ℓ/2 < \|x\| < 5ℓ/8` | `−64/(45ℓ)` ≈ `−1.422222/ℓ` | |
| `M^II` | `\|x\| < ℓ/2` | `+257/(165ℓ)` ≈ `+1.557576/ℓ` | `\|x\| < 3ℓ/4` |
| | `ℓ/2 < \|x\| < 5ℓ/8` | `−568/(165ℓ)` ≈ `−3.442424/ℓ` | |
| | `5ℓ/8 < \|x\| < 3ℓ/4` | `+40/(33ℓ)` ≈ `+1.212121/ℓ` | |

**Verified in exact rational arithmetic**: `M^I` moments `n = 1,2,3` vanish, `n = 4` gives `−5ℓ⁴/256 ≠ 0`
(⇒ `p = 3`); `M^II` moments `n = 1…5` vanish, `n = 6` gives `+225ℓ⁶/28672 ≠ 0` (⇒ `p = 5`). Also
confirmed `1 − Ĝ(k) ∝ k^{p+1}` with `p = 1, 3, 5` for top-hat, `M^I`, `M^II`.

Two implementation details from §V that are easy to miss:

- **Higher dimensions are a separable product, not radial**: "in 2D, we define G(x,y) ≡ G(x)G(y). If
  kernel G(x) is of order p in 1D, then G(x) is of the same order p in higher dimensions."
- **Grid constraint**: "representing SS kernels on a grid requires at least 1 grid-cell of size Δx for
  each of the limbs. Therefore, the smallest length-scale ℓ that can be probed by such a kernels is
  limited by b ≥ Δx." With `b = ℓ/8` that is `ℓ ≥ 8Δx`.

`M^I` cannot be pushed to `p = 5` by tuning `b`: "It might be tempting to choose the free parameter b to
satisfy ∫dx x⁴ M^I(x) = 0, however, it is straightforward to check that the solution is not realizable."
(`m₂ = 0` needs `a/c = 1/(t³−1)`, `m₄ = 0` needs `1/(t⁵−1)`, with `t = 1 + 2b/ℓ`; equal only at the
degenerate `b = 0`.)

### 1.6 Two measured cautions against using `M^I`/`M^II` for flux

Both are measurements, verified two independent ways (closed-form segment transform and direct
quadrature), not claims from the papers.

1. **They are poor low-pass filters.** `|Ĝ|²` sidelobes: top-hat peaks at 0.047; `M^I` at 0.282;
   **`M^II` reaches 1.0124 at `kℓ ≈ 17`** — it *amplifies* energy there, with further lobes of 0.893 and
   0.863. They are built to make `1 − Ĝ ∝ k^{p+1}` near `k = 0`, which is all the spectrum-slope argument
   needs; nothing constrains them at high `k`. Using `M^II` to form `ū` and hence `Π = −S̄:τ̄` would let
   small-scale energy through essentially unattenuated.
2. **The vanishing second moment does not survive naive discretization.** For any point-sampled stencil,
   midpoint quadrature gives `m₂^discrete = m₂^exact − (Δx²/12)·m₀`. A kernel with exact `m₂ = 0`
   discretizes to `m₂ = −Δx²/12`, **independent of kernel shape**. At the paper's minimum resolution
   `ℓ = 8Δx` that residual is `−ℓ²/768`, only ~64× below the top-hat's `ℓ²/12`, and comparable to the
   intended leading `m₄k⁴` term at `kℓ ≈ 1` — the very range the filtering spectrum probes. `[SA18]`
   §III.F guards against this but gives no quantitative criterion. To get the order you paid for, you
   need `ℓ/Δx` well above 8, or cell-averaged rather than point-sampled weights.

An alternative `p = 3` construction, if wanted: `[ZA24]` §III.B builds one as a linear combination of
Gaussians, `G^{p3}_ℓ(x) = c·G_ℓ(x) − c'·G_{ℓ'}(x−x₀) − c'·G_{ℓ'}(x+x₀)` with `c' = (c−1)/2`.
**The published formula for `x₀` is a typo** — it reads `x₀ = cℓ²/(12(c−1)) − ℓ'²/12`, whose RHS has
units of length². The dimensionally correct condition is `x₀² = cℓ²/(12(c−1)) − ℓ'²/12`; verified
numerically at their stated `c = 1.1, ℓ/ℓ' = 2` (the `x₀²` reading gives `m₂ = 9.7e−16`, the printed
reading gives `9.3e−3`).

### 1.7 The Gaussian width convention

Two incompatible conventions both called "a Gaussian of width ℓ":

- `[RAE14]`: `G_ℓ(x) = (π/ℓ²)^{1/2} e^{−π²x²/ℓ²}`, `Ĝ_ℓ(k) = e^{−k²/k_ℓ²}` → **σ = ℓ/(π√2) ≈ 0.225ℓ**,
  chosen so `Ĝ` e-folds exactly at `k_ℓ = 2π/ℓ`.
- `[ZA24]`, Germano 1992, gcm-filters: `(6/πℓ²)^{n/2} e^{−6|x|²/ℓ²}` → **σ = ℓ/(2√3) ≈ 0.289ℓ**, whose
  second moment `ℓ²/12` matches a top-hat of width `ℓ` exactly.

That is a **28% difference in effective filter scale for the same nominal ℓ**.

In this package's `exp(−α(d/ℓ)²)` parameterization:

| α | corresponds to |
|---|---|
| 4 | FlowSieve's shipped default, `exp(−4r²/ℓ²)` |
| **6** | the `[ZA24]` convention; second moment `ℓ²/12`, matching the top-hat exactly |
| π² ≈ 9.87 | `[AE-I]`'s own DNS Gaussian, `exp(−π²r²/ℓ²)`, second moment `ℓ²/(2π²)` |

`α = 6` as the default is therefore well-founded. Beware a name collision: FlowSieve's `kernel_alpha` is
a *2-D second moment* `⟨r²⟩/ℓ²`, not an exponent.

---

## 2. The flux `Π`

### 2.1 Definition and sign — universal

**`Π = −(∂_j ū_i) τ̄_ij = −S̄_ij τ̄_ij`, positive = forward/downscale.** The two forms are *identically*
equal because `τ̄` is symmetric; `[RAE14]` §IV.B eq. (13) says so explicitly. Confirmed identical across
`[AE-I]`, `[AE-II]`, Germano 1992 eq. (25), Meneveau & Katz 2000, `[AHV18]` eq. (7), `[A17]` eq. (36),
`[A13]` eq. (14), Storer et al. 2023 eq. (5), Srinivasan et al. 2023 eq. (4).

Two printed sign errors exist in the literature and should not be taken as convention splits:
`[A17]` §4.5 and eq. (43) print `Π_ℓ = ∇ū_ℓ : τ^u_ℓ` **without the minus**, contradicting its own
eqs. (36) and (49) — eq. (36) is authoritative. Srinivasan et al. eq. (3) prints `ῡ_x` where eq. (4)
requires `ῡ_y`.

### 2.2 The divergence/advective form is rejected, not an alternative

`[AE-I]`: `Π̄^uns ≡ −∂_j ū_i (u_iu_j)‾ = Π̄ − ∂_j(½ū_j|ū|²)` — "this definition is not pointwise
Galilean-invariant, so that the amount of 'energy cascade' at any point in the fluid according to this
definition would differ for observers moving at different uniform velocities!"

`[AHV18]` calls the freedom to move divergences between `Π` and `J^transport` a **gauge freedom** and
fixes the gauge by Galilean invariance. In a homogeneous flow all choices share the same mean; in an
inhomogeneous domain they "differ qualitatively as well as quantitatively."

**Use the deformation form.** If an enstrophy flux is ever added, use the deformation form there too —
mixing gauges between `Π` and `Z` is inconsistent.

### 2.3 The full budget

`[AE-I]` eq. (4), incompressible and unforced:

```
∂_t(½|ū|²) + ∂_j[(½|ū|² + p̄)ū_j + ū_i τ̄_ij − ν∂_j(½|ū|²)] = −Π̄ − ν|∇ū|²
```

`[AHV18]` eq. (6), rotating Boussinesq, adds the two terms a hydro-only budget lacks:

```
∂_t ρ₀|ū_ℓ|²/2 + ∇·J^transport_ℓ = −Π_ℓ − ρ₀ν|∇ū_ℓ|² + ρ̄_ℓ g·ū_ℓ + ρ₀ F̄^forcing_ℓ·ū_ℓ
J^transport_ℓ = ρ₀|ū_ℓ|²/2 ū_ℓ + P̄_ℓ ū_ℓ − ρ₀ν∇(|ū_ℓ|²/2) + ρ₀ ū_ℓ·τ̄_ℓ(u,u)
```

Terms: (a) advection of large-scale KE; (b) pressure transport; (c) viscous diffusion; (d) turbulent
diffusion `ū·τ̄`; (e) cross-scale flux `−Π_ℓ`; (f) direct viscous dissipation, "negligible at scales
ℓ≫ℓ_d"; (g) baroclinic conversion `ρ̄_ℓ g·ū_ℓ`; (h) injection.

The complementary small-scale budget (`[AE-I]`, eq. `u-small`) has `Π̄` with the opposite sign, which is
what makes it a flux between the two.

**What a `Π`-only diagnostic omits.** Under exact homogeneity `⟨∇·J⟩ = 0`, so transport drops from the
mean — that is the *only* term `Π`-only gets for free. **Baroclinic conversion and injection are not
divergences and do not vanish under averaging.** Quantified for the ocean: Loose et al. 2023 define
relative backscatter as the ratio of cross-scale KE transfer to baroclinic EKE production and report
"35%–40%" in the eddy-permitting regime, "70%–100%" at some latitudes, and ">100%" locally — i.e. **the
omitted baroclinic term is typically 2.5–3.5× larger than `Π`**. A `Π`-only diagnostic cannot attribute
an EKE change.

### 2.4 `E(ℓ)` — the right object

`[AHV18]`, after eq. (6): "what we dub *large-scale KE* is the KE in the large-scale flow, based on
`ū_ℓ`, rather than the filtered KE itself, `ρ₀|u|²‾/2`, which *does not* cascade across scales."

So `E(ℓ) = ½⟨|ū_ℓ|²⟩` is correct and `½⟨(|u|²)‾_ℓ⟩` is not.

### 2.5 `τ` — Germano central moment, no Leonard split needed

Germano 1992 eq. (24): `τ(f,g) = ⟨fg⟩ − ⟨f⟩⟨g⟩`. Its requirements on the averaging operator are only
linearity, constant preservation, and commutation with derivatives — **not** the Reynolds rules. Germano
explicitly declines the Leonard decomposition: "there will be no recourse to any kind of decomposition."

The Germano (1986) three-way split, unlike Leonard (1974), is Galilean invariant term by term because
each piece is itself a single-level central moment `⟨ab⟩−⟨a⟩⟨b⟩`.

**Germano identity**, eq. (33): `τ_fg(u_i,u_j) = ⟨τ_f(u_i,u_j)⟩_g + τ_g(⟨u_i⟩_f, ⟨u_j⟩_f)`.

For a **multiscale / band-pass** extension the naive generalization is wrong. `[AE-I]` Appendix 2 gives
the correct one, involving **repeated** low-pass filters `f̄_{m,n} = (G_n*⋯*G_m*f)`; only for "S-type"
kernels does `f̄_{N,n} = f̄_n`, giving the clean band decomposition (eq. 13). `[AE-I]` warns explicitly
against band-passing the *velocity* instead: that yields off-diagonal terms of indefinite sign, and "It
is thus not clear in this approach how precisely to identify the kinetic energy at a given length-scale."

### 2.6 The filtering spectrum

`[SA18]` eq. (14)–(15):

```
Ē(k_ℓ) ≡ (d/dk_ℓ)⟨|ū_ℓ(x)|²⟩/2 = −(ℓ²/L)(d/dℓ)⟨|ū_ℓ(x)|²⟩/2      where k_ℓ = L/ℓ
𝓔(k_ℓ) ≡ ½⟨|ū_ℓ(x)|²⟩                                            (cumulative)
```

Energy conservation, eqs. (19)–(20): `½⟨|u|²⟩ = ½⟨|ū_{ℓ₀}|²⟩ + ∫_{k_{ℓ₀}}^∞ dk_ℓ Ē(k_ℓ)`.

Numerical recipe: `ℓ_m = 2mΔx` ("The factor 2 ensures that even kernels yield vanishing odd moments"),
`k_{ℓ_m} = L/ℓ_m`, and `Ē(k_{ℓ_m}) = [𝓔(k_{ℓ_m}) − 𝓔(k_{ℓ_{m-1}})]/[k_{ℓ_m} − k_{ℓ_{m-1}}]`.

**The constant `C` in `k_ℓ = C/ℓ` genuinely differs across the literature**, by up to 2π:

| source | mapping |
|---|---|
| `[SA18]` eqs. (14), (32) | `k_ℓ = L/ℓ` with `L` the domain size (a dimensionless index; with `L = 2π` this is `2π/ℓ` in radian units) |
| `[RAE14]` | `k_ℓ ≡ 2π/ℓ`, stated outright |
| Storer et al. 2022/2023 | `k_ℓ = 1/ℓ`, Jacobian `−ℓ² d/dℓ` |
| Loose et al. 2023 | `k_L = π/L` — "we associate the filter scale with only one-half of a wavelength" |
| `[AHV18]` | figures use `K = 2π/ℓ` and also `K = 10⁴/ℓ` km⁻¹ with the disclaimer "**Here, K is not a wavenumber, just a number proportional to ℓ⁻¹**" |

Each paper is internally consistent, and any `C` satisfies `∫Ē dk_ℓ = E_total` provided the same `C`
appears in the Jacobian and on the abscissa. But **the amplitude scales as `1/C`**, so comparing against
a Fourier spectrum requires stating the convention. This package must document which it uses.

---

## 3. Compressible / Favre

`[A13]` eq. (9): `f̃_ℓ ≡ (ρf)‾_ℓ / ρ̄_ℓ`. "The operator (~·) is linear but **does not commute with
derivatives**." Stress carries a `ρ̄`, eq. (11): `ρ̄ τ̃(u_i,u_j) ≡ ρ̄(ũ_iu_j − ũ_iũ_j)`. Favre continuity
closes exactly: `∂_tρ̄ + ∂_i(ρ̄ũ_i) = 0`.

Budget, eq. (13) with terms (14)–(18):

```
∂_t ρ̄_ℓ|ũ_ℓ|²/2 + ∇·J_ℓ = −Π_ℓ − Λ_ℓ + P̄_ℓ ∇·ū_ℓ − D_ℓ + ε^inj_ℓ
Π_ℓ = −ρ̄ ∂_j ũ_i τ̃(u_i,u_j)                                 deformation work
Λ_ℓ =  (1/ρ̄) ∂_j P̄ τ̄(ρ,u_j)                                  baropycnal work
D_ℓ =  ∂_j ũ_i [2(μS_ij)‾ − (2/d)(μS_kk)‾δ_ij]
J_j =  ρ̄(|ũ|²/2)ũ_j + P̄ū_j + ũ_i ρ̄τ̃(u_i,u_j) − ũ_iσ̄_ij
ε^inj_ℓ = ũ_i ρ̄ F̃_i
```

`τ̄(ρ,u_j) ≡ (ρu_j)‾_ℓ − ρ̄_ℓū_{j,ℓ}` is the **unweighted** subscale mass flux — the framework needs both
moment types.

**The main implementation trap**, `[A13]` §4.2: `Λ` "has often been lumped with `P̄_ℓ∇·ū_ℓ` in the form
`P̄_ℓ∇·ũ_ℓ` … and treated as a large-scale pressure dilatation which does not require modeling." Eq. (13)
uses `∇·ū_ℓ` (**unweighted**). Writing `P̄∇·ũ` silently absorbs and destroys `Λ`.

Closed form for `Λ`, Lees & Aluie 2019 eqs. (16)–(18), with `C₂ ≡ ∫d³r G(r)|r|²`:

```
Λ ≈ (1/3)C₂ℓ²(1/ρ̄)[∇P̄·S̄·∇ρ̄ + ½ω̄·(∇ρ̄×∇P̄)] = Λ_SR + Λ_BC
```

**Why an incompressible-style `τ̄` is wrong in variable-density flow** — three documented failures:
the inviscid criterion fails (`[A13]` §3.1); the dissipation loses Galilean invariance (Zhao & Aluie
2018 §III); and measured severity is catastrophic, not marginal — the non-Favre viscous terms "are
several orders of magnitude larger" and in high-contrast 3-D Rayleigh–Taylor they "do not decay at large
scales, in violation of the inviscid criterion."

Uniqueness is **not** proven: `[A13]` §3.2 — "we did not prove that it is necessary. We only showed that
the Favre decomposition is sufficient."

Positivity `G ≥ 0` is an **explicit hypothesis** of `[A13]` Props. 2, 4, 5, 6, because positivity makes
coarse-graining an average and gives Jensen's `1/ρ̄_ℓ ≤ (1/ρ)‾_ℓ`. Zhao & Aluie §IV avoid the
sharp-spectral filter because it "violates physical realizability since it can have negative values."
Lees & Aluie Table 2 quantifies: correlation with the `Λ` model is box 0.93/0.94, Gaussian 0.97/0.97,
sharp spectral **0.27/0.28**.

---

## 4. Masked domains — the three policies

**This is the least settled area, and the literature is unanimous against normalized convolution.**

### 4.1 What the papers say

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
> "such inhomogeneous kernels … **do not commute with spatial derivatives**. Consequently, the
> coarse-grained field resulting from a deformed kernel is not guaranteed to satisfy fundamental flow
> properties … such as non-divergence, geostrophy, and the vorticity present at various scales."
>
> "a kernel that is inhomogeneous … **does not conserve domain averages, including the kinetic energy of
> the flow** … it can yield total energy that is either less than 100% … or **greater than 100%**."

Their Appendix A gives the mechanism as a 5-point 1-D matrix: renormalizing *rows* to sum to 1 does not
make *columns* sum to 1, so measure preservation is lost. Their conclusion: "**To fully conserve energy
and maintain commutativity with differentiation, we choose the 'Fixed Kernel w/ Land' option**, which
treats land as zero-velocity water and includes land cells in spatial integrals."

Grooms et al. 2021 states the tradeoff neutrally, and notes zero-fill's cost: "Setting the velocity to
zero over land allows the filter to commute with derivatives, but at the cost of reducing the strength
of currents near land. For example, the Florida Current is much weaker … It is thus clear that both
methods have pros and cons near boundaries."

### 4.2 Measured coast artifacts

Uniform ocean field `u ≡ 1`, Gaussian, `ℓ = 16` cells, straight coast. **Measured**, reproducing
FlowSieve's exact discrete filter:

| cells from coast | `ū` zero-fill | `ū` renormalized | kernel centroid offset | `m₂/m₂(interior)` |
|---|---|---|---|---|
| 1 | **0.535** | 1.000 | +4.21 cells (0.26ℓ) | **0.38** |
| 2 | 0.605 | 1.000 | +3.61 | 0.42 |
| 4 | 0.732 | 1.000 | +2.55 | 0.52 |
| 8 | 0.908 | 1.000 | +1.03 | 0.72 |
| 16 (= ℓ) | 0.997 | 1.000 | +0.05 | 0.97 |
| 24 | 0.99998 | 1.000 | +0.0004 | 1.00 |

So zero-fill suppresses coarse KE by 46% one cell from a coast, recovering to 0.3% only at ~ℓ.
Renormalization restores constants exactly, but the effective kernel is still one-sided — centroid
displaced 0.26ℓ offshore and **effective filter width reduced to ~62% of intended** at the coast.

gcm-filters' third policy (modified no-flux operator) conserves the integral exactly (measured to
1.5e−16) but also shrinks the effective scale at the coast — measured `m₂` down to **37%** of interior,
with the centroid displaced +1.365 cells.

**Summary of the three:**

| policy | preserves | coast cost |
|---|---|---|
| zero-fill land (`[AHV18]`, `[BSK+23]`, FlowSieve default) | commutation with ∇, hence the budget | `ū` down 46% at 1 cell |
| renormalize over water ("deformed kernel") | constants exactly | breaks commutation; effective ℓ → 62% |
| modified no-flux operator (gcm-filters) | area integral exactly | effective ℓ → ~60% |

### 4.3 Other masking facts worth keeping

- **Area convention with zero-fill**: sum over **all** cells including land, divide by **water** area
  (Storer et al. 2022 Methods). Grooms et al.: "to preserve the integral with this method, the integral
  has to be extended over land."
- **Leakage onto land**, `[BSK+23]`: "energy leakage of the order of 1% at coarse-graining scales
  <100 km, ≈4% for scales ≲500 km, and up to 12% at scales of order 2000 km."
- **A quantified exclusion buffer exists**: `[RAE14]` Appendix A measures rms flux error vs distance
  from an artificial boundary and concludes "we have considered a maximum error rate of 10⁻⁴ as
  acceptable … this stipulates that we restrict our analysis to flow locations **at least a distance
  ≈ ℓ(n−1) from the domain boundary**, where n is the localization order" (n = 2 for the Gaussian, i.e.
  a buffer of ≈ℓ).
- **Never renormalize separably.** A separable per-1-D-pass renormalization is *not* equal to a 2-D
  masked normalized convolution unless the mask itself is separable. (This package builds its denominator
  by running the full separable convolution on the mask, which is correct; the trap is renormalizing
  inside each pass.)
- **Non-negative kernels are needed for realizability** independent of commutation: a kernel with
  negative lobes can produce a negative local variance `f²‾ − f̄²`.
- **Scalars are harder than velocity.** Zero-fill is boundary-condition-consistent for velocity under
  no-slip; for tracers there is "an infinite family of possible extensions." Storer, Kharghani, Adcroft
  & Aluie (arXiv:2512.03051, 2025) propose solving a Laplace BVP over land. No public code.

---

## 5. The sphere

- **`ℓ` is a great-circle *diameter*.** The standard is `D = γ/(ℓ/2)` with `γ` the great-circle distance,
  so a top-hat is a spherical cap of great-circle *radius* `ℓ/2`. `[AHV18]` eq. (2) gives the cap area
  `A = 2πR²[1 − cos(ℓ/2R)]`.
- **Commutation requires a zonal kernel.** `[A19]` requires the kernel be a function of geodesic distance
  alone, and `G ∈ L^p[−1,1]`. Normalization is needed separately for mean preservation, not for
  commutation. **No moment conditions.**
- **Radial derivatives commute for free.** `[A19]` Corollary 1 extends commutation to `∇f‾=∇f̄`,
  `∇×u‾=∇×ū`, `∇·u‾=∇·ū`, `Δu‾=Δū`, `∇·T‾=∇·T̄`, with the proof: "The proof is a simple consequence of
  our filtering operation not acting in the radial direction. Therefore, it commutes with radial
  derivatives." **This is what justifies the layered / 2.5-D approach** — it is not an approximation
  with respect to commutation.
- **Vectors are not filtered component-wise, strictly.** `[A19]`'s generalized vector convolution uses
  *spectrally shifted* kernels `G^{(1)} = LT⁻¹{Ĝ(n−1)}`, `G^{(2)} = LT⁻¹{Ĝ(n+1)}` (eqs. 59–61). Filtering
  the 3-D Cartesian components instead is valid **only for non-divergent vectors**; per `[BSK+23]`
  summarizing Aluie & Teeraratkul (2023), "significant errors can arise for a general flow field."
  Converting spherical components to planetary Cartesian before filtering (what this package and
  FlowSieve do) is strictly better than filtering the spherical components directly, but is **not** the
  full formalism. Document this.
- **Spherical-harmonic filtering is a worse tool for this**, per `[BSK+23]`: too noisy at gyre scales,
  "beams of spectral ringing that extend deep into land regions", and at a 1000 km filter it "fills the
  global ocean with zonal bands." Convolutional coarse-graining confines nonzero velocity to within `ℓ/2`
  inland.

### 5.1 What the layered (2.5-D) approach actually drops

Only the vertical-shear part of `Π`. Srinivasan et al. 2023 eq. (3) splits it:

```
Π = −[τ_uu ū_x + τ_uv(ū_y+ῡ_x) + τ_υυ ῡ_y]  −  (τ_uw ū_z + τ_υw ῡ_z)
       └────────────── Π_h ──────────────┘      └────── Π_υ ──────┘
```

Their stated justification, the only quantified one found: "Π_z … is on average about 5 times smaller
than Π_h". (Their Fig. 14 caption says 4 — an internal inconsistency in that paper.)

Storer et al. 2023 justify dropping vertical velocity entirely for hydrostatic ocean flow: "We conducted
identical analysis that included radial/vertical velocities … and found that including the radial
velocity `u_r` has a negligible impact on both diagnostics across all scales analyzed here."

---

## 6. The strain / convergence decomposition of `Π`

Srinivasan, Barkan & McWilliams 2023, *JPO* 53(1) 287–305, eq. (10). Diagonalizing `S̄`:

```
δ̄  = ū_x + v̄_y                                   divergence (rotation invariant)
ᾱ² = (v̄_y − ū_x)² + (v̄_x + ū_y)²                 strain magnitude (rotation invariant)
σ̄_n = ū_x − v̄_y     normal strain
σ̄_s = ū_y + v̄_x     shear strain

Π_h = (τ_vv − τ_uu)·σ̄_n/2 − τ_uv·σ̄_s  −  (τ_vv + τ_uu)·δ̄/2
      └────── Π_α : deformation shear production ──────┘  └─ −Π_δ : convergence production ─┘
```

Equivalently `Π_h = E′γᵖᾱ − E′δ̄` with `E′ = (τ_vv+τ_uu)/2`, `γᵖ = (τᵖ_vv−τᵖ_uu)/(τᵖ_vv+τᵖ_uu)`, and
bounds `−ᾱE′ ≤ Π_α ≤ ᾱE′` since `−1 ≤ γᵖ ≤ 1`. The algebra was verified: expanding (10) collapses
exactly to `−τ_uu ū_x − τ_uv(ū_y+v̄_x) − τ_vv v̄_y`.

`Π_δ` (convergence production) is new in that paper; eq. (10) with `δ = 0` recovers Polzin 2010, and the
`γᵖ` form with `δ = 0` recovers Jing et al. 2017. Their buoyancy analogue, eq. (16):
`B_h = (b_y²−b_x²)σ_n/2 − b_x b_y σ_s − (b_y²+b_x²)δ/2`.

**This decomposition and the direct form are reported to disagree in practice** — an open, unresolved
report against FlowSieve, where the two computed with the same kernel and constants differ "primarily at
scales where Pi(scales) > 0 (forward KE cascade)", with fourth- vs second-order differentiation
suspected. Computing both forms and checking they agree is therefore a *strong* correctness test.

---

## 7. Validation cases

The ladder below is what would catch real errors; every item is from a published validation.

1. **`[SA18]` synthetic spectra** — doubly-periodic fields with prescribed slopes.
   - `k⁻⁵ᐟ³`: top-hat and Gaussian both correct.
   - `k⁻⁴`: top-hat and Gaussian **lock at k⁻³**; `M^I`, `M^II` recover `k⁻⁴`.
   - `k⁻⁷`: top-hat locks at `k⁻³`, **`M^I` locks at `k⁻⁵`**, only `M^II` recovers `k⁻⁷`.
   This is a direct gate on eq. (18), `Ē ~ k^{−min(α, p+2)}`.
2. **Pure large-scale strain `u = (y, x)`** — the analytic no-small-scales benchmark. A correct
   high-order result gives "almost zero energy at small-scales", and the `ℓ = L/3` vs `L/6` KE fields
   "look almost identical".
3. **Filter of a constant** = that constant (partition of unity), including near a mask boundary under
   whichever masking policy is chosen.
4. **`filter ∘ div = div ∘ filter`** and `filter ∘ grad = grad ∘ filter` on a divergence-free field. This
   is the `[A19]` property, and it is exactly what a deformed/renormalized kernel breaks.
5. **Solid-body rotation invariance** (`u = cos(lat)`, `v = 0`) — a corollary of conserving angular
   momentum.
6. **`[RAE14]` enstrophy-flux vs the third-order relation** `S^W₂(r) = −2ηr` at `r = 2ℓ` — they report
   agreement to within ≈3/4. The closest thing in this literature to a ground-truth flux check.
7. **Kernel moment gates** — `∫G = 1` exactly; `∫xG = 0`; `∫x²G` finite and non-zero (and equal to the
   documented value for the chosen `α`); for any high-order kernel, `∫x²G ≈ 0` *after discretization*,
   with the truncation radius checked against §1.6 item 2.
8. **Both forms of `Π`** — direct `−S̄:τ̄` vs the §6 strain/convergence decomposition — must agree.

---

## 8. Open questions and unverified items

Recorded so they are not silently treated as settled.

- **`M^I`/`M^II` coefficients were transcribed from the arXiv LaTeX source, not the journal PDF.** One
  transcription error was found elsewhere in the same corpus (the `[ZA24]` `x₀` typo, §1.6), so verify
  against the PDF before implementing.
- **Whether `[SA18]` addresses the top-hat's non-monotone `|Ĝ|²` against its own eq. (21) condition** —
  not found either way in the text read. The measurement in §1.4 shows the top-hat does violate it.
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
- **The `k_ℓ = C/ℓ` convention this package should adopt** is a decision, not a fact — see §2.6.

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
- Germano 1992, "Turbulence: the filtering approach", *JFM* 238, 325
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
- Aluie & Kurien 2011, arXiv:1107.5006 — tracer variance flux, `Π^θ_ℓ = −τ̄(θ,u_j) ∂_jθ̄`
- Grooms, Loose, Abernathey, Steinberg, Bachman, Marques, Guillaumin & Yankovsky 2021, "Diffusion-based
  smoothers for spatial filtering of gridded geophysical data", *JAMES* 13
- Loose, Bachman, Grooms & Jansen 2023, "Diagnosing scale-dependent energy cycles in a
  high-resolution isopycnal ocean model", *JPO* 53, 157
- Storer, Kharghani, Adcroft & Aluie 2025, arXiv:2512.03051 — harmonic extension over land
