# Development plan

Goal: a package that computes the right quantities, states its own limits, is fast, and is usable
without reading the source. Written to survive being handed to someone with no prior context.

Every item below has a **Done when** that is checkable by a test or a measurement, not by judgement.
`[REQ §n]` refers to sections of [`requirements_from_literature.md`](requirements_from_literature.md),
which holds the equations, coefficients and citations — this plan does not restate them.

Ordering is by risk: things that make current output wrong or unverifiable come before things that add
capability, which come before speed.

---

## Stage 0 — Convention changes

These change results, so they are listed first and separately. They do **not** block anything: the tests
in Stage 1 are parameterized over every mask policy and every kernel, so they are independent of which
option is the default.

The evidence settles all three. They are written here as conclusions, not as open questions.

### 0.1 Masking: default to `ZeroFill`

`Deformable` is the "deformed kernel". It preserves constants exactly but **breaks commutation with
derivatives**, and the `Π` budget derivation is obtained by commuting the filter through the momentum
equation — so with `Deformable` the budget the package reports is not derivable on a masked domain
`[REQ §15.2]`. `[BSK+23]` further shows it does not conserve the domain average, and can return total
energy either below or **above** 100%, with the row-vs-column normalization argument in their Appendix A.

`ZeroFill` preserves commutation. Its cost is real and bounded: `ū` suppressed 46% one cell from a coast,
recovering to 0.3% at `~ℓ` `[REQ §15.3]`.

A recoverable, quantified error beats an unrecoverable structural one. **Default `ZeroFill`.**

- Keep `Deformable` available and document that budget-level conclusions are not supported with it.
- Document both artifacts with the `[REQ §15.3]` table.
- Add the `[RAE14]` exclusion buffer (`≈ℓ(n−1)` from the boundary, `n = 2` for a Gaussian) as an opt-in
  output mask, so users can drop contaminated points rather than eyeball them.
- Follow the zero-fill area convention: sum over **all** cells including land, divide by **water** area
  `[REQ §15.4]`.

**Done when**: the default is `ZeroFill`; a test asserts `filter ∘ div == div ∘ filter` for it; the same
test is documented and asserted to *fail* for `Deformable`, so the difference is visible rather than
folklore; and both artifacts appear in the `mask_strategy` docstring with numbers.

### 0.2 Kernel: keep `TopHatKernel` for `Π`, reject it for the spectrum

`TopHatKernel` is what `[AHV18]` uses for flux and is a fine default there. But it violates the spectrum
positive-definiteness condition `[REQ §10.3]` — **measured**, `|Ĝ|²` is non-monotone with a `+0.0026`
lobe near `kℓ ≈ 8.8` `[REQ §7]` — and it has ill-defined `ℓ`-derivatives, which the spectrum needs.

**Keep the flux default; make `filtering_spectrum` error** on a kernel whose `|Ĝ|²` is not monotone
decreasing, rather than silently returning a possibly-negative spectrum.

**Done when**: requesting a spectrum with a non-conforming kernel throws, naming the condition and
suggesting `GaussianKernel`; and a test asserts the spectrum is non-negative for every kernel that is
allowed through.

### 0.3 Spectrum wavenumber: keep `k_ℓ = L/ℓ`, state it

Four conventions exist and the amplitude scales as `1/C` `[REQ §10.4]`. The current behaviour matches
`[SA18]`. **Keep it**, and state it wherever the spectrum is defined, together with the fact that
comparing amplitudes or peak locations against a Fourier spectrum requires matching conventions.

**Done when**: the convention is in the `filtering_spectrum` docstring and in the theory docs, with the
`1/C` amplitude caveat.

---

## Stage 1 — Make the current output verifiable

Nothing here adds features. It makes the existing numbers checkable, and it is the stage that would have
caught the problems found so far.

### 1.1 The validation ladder

Implement `[REQ §18]` as tests. In priority order:

1. **Synthetic spectra** `[REQ §18.1]` — prescribed `k⁻⁵ᐟ³`, `k⁻⁴`, `k⁻⁷` fields; assert the recovered
   slope equals `min(α, p+2)`. This directly gates the kernel-order claim and will show the top-hat and
   Gaussian locking at `k⁻³`.
2. **`filter ∘ div == div ∘ filter`** on a divergence-free field `[REQ §18.4]`.
3. **Filter of a constant** returns that constant, including adjacent to a mask boundary `[REQ §18.3]`.
4. **Solid-body rotation invariance** `[REQ §18.5]`.
5. **Pure strain `u = (y,x)`** `[REQ §18.2]` — near-zero small-scale energy.
6. **Kernel moment gates** `[REQ §18.8]` — `∫G = 1` exact, `∫xG = 0`, `∫x²G` equal to the documented value
   for the chosen `α`.

**Done when**: all six are in `runtests.jl`, and each one *fails* if the corresponding property is broken
(verify by temporarily breaking it).

### 1.2 State the limits in the docs

The package currently documents what each kernel *is*, never what the theory *requires* `[REQ §2.1]`.

- Kernel requirements section in `theory.md`, pointing at `requirements_from_literature.md`.
- The `k⁻³` slope ceiling for `p = 1` kernels, stated on `filtering_spectrum` `[REQ §10.2]`.
- The coast artifact numbers on `mask_strategy` `[REQ §15.3]`.
- The Cartesian-component caveat on spherical filtering: valid strictly only for non-divergent fields;
  the full formalism is `[A19]`'s spectrally shifted kernels `[REQ §16]`.
- What a `Π`-only diagnostic omits, with the "baroclinic term is 2.5–3.5× larger than `Π`" number
  `[REQ §8.5]`.
- `ℓ` is a **diameter** `[REQ §16]`.

**Done when**: each bullet is in the rendered docs and the docs build clean.

### 1.3 Both forms of `Π` must agree

Implement the strain/convergence decomposition `[REQ §11]` and assert it equals the direct
`−S̄:τ̄` form. This is reported to disagree in at least one other implementation, so it is a real test,
not a tautology — the two use different derivative combinations.

**Done when**: `Π_α + Π_δ == Π` to round-off on a masked and an unmasked grid, and both are exported.

---

## Stage 2 — Fill in what the framework needs

### 2.1 Kernels

Add, with moment gates from 1.1:

- `SmoothHat` — `½(1 − tanh(10(γ/(ℓ/2) − 1)))`. This is the kernel used in the Storer et al. papers.
- `HyperGaussian` — `exp(−D⁴)`.
- `M^I` and `M^II` `[REQ §4.1]`, with the coefficients verified against the journal PDF first
  `[REQ §19]`, the separable-product rule for ≥2-D, and the `ℓ ≥ 8Δx` guard.

For `M^I`/`M^II`, also implement the two cautions as guards `[REQ §4.3]`: warn when `ℓ/Δx` is small
enough that discretization destroys the vanishing moment, and document the high-`k` sidelobes
(`M^II` reaches `|Ĝ|² = 1.0124`) so they are not used for `Π` unwittingly.

**Done when**: each kernel passes the moment gates, `M^I` recovers `k⁻⁴` and `M^II` recovers `k⁻⁷` in the
synthetic-spectrum test, and the sidelobe caution is in the docstring.

### 2.2 Favre / compressible

The equations are in `[REQ §13]`. Implement `f̃ = (ρf)‾/ρ̄`, `ρ̄τ̃(u_i,u_j)`, deformation work `Π`,
**baropycnal work `Λ`**, and the budget's `P̄∇·ū` term.

The trap to avoid is explicit `[REQ §13]`: the budget uses `∇·ū` (**unweighted**); writing `P̄∇·ũ`
silently absorbs and destroys `Λ`.

**Done when**: the Favre continuity `∂_tρ̄ + ∂_i(ρ̄ũ_i) = 0` closes to round-off on a synthetic
variable-density field, `Λ` matches the Lees & Aluie closed form on a smooth field, and a test asserts
`Λ ≠ 0` for a baroclinic configuration (so it cannot be silently dropped).

### 2.3 Enstrophy flux

`Z_ℓ = −∇ω̄_ℓ · τ̄_ℓ(u,ω)` `[RAE14]` eq. (16). Use the **deformation form**, consistent with `Π` — mixing
gauges between the two is the inconsistency `[REQ §8.2]` warns about.

**Done when**: `Z` is exported, uses the deformation form, and is checked against the third-order relation
`S^W₂(r) = −2ηr` at `r = 2ℓ` `[REQ §18.7]`.

### 2.4 Band-pass / multiscale transfer

`[REQ §9.2]` gives the correct multiscale Germano identity and warns that the naive version — band-passing
the velocity — produces off-diagonal terms of indefinite sign and no well-defined per-scale energy.

**Done when**: band energies sum to the total to round-off, and the implementation uses the repeated-filter
identity rather than velocity band-passing.

---

## Stage 3 — Usability

The package is currently usable only by someone who reads the source.

- **A worked example per entry point** in the docs: single sweep, batch sweep, vertical profile, masked
  spherical grid — each with the output shapes stated.
- **Shape and convention table**: what every result field's axes mean, including the profile layout
  `(spatial…, Nscales, levels…)` and the batch layout.
- **Error messages that name the fix.** Already true for the batch-workspace and spectral-batch errors;
  make it true for kernel/mask/backend mismatches.
- **A `check_setup(grid, kernel, scale; ...)` entry point** that reports, without computing: which engine
  will run, on which backend, whether the kernel supports the requested diagnostics, whether `ℓ` is
  resolvable on this grid (`ℓ ≥ 8Δx` for high-order, `ℓ > 2Δx` at all), and the coast-buffer width.

**Done when**: a new user can go from a grid and a velocity field to a correct `Π` without opening `src/`.

---

## Stage 4 — Performance

Largely complete. Recorded here so it is not redone, and so the remaining items are visible.

**Done and measured** (all bit-identical to a serial per-slice reference):

| | result |
|---|---|
| `coarse_grain!` 512², 5 scales | 982.6 → 116.5 ms serial, 78.8 ms threaded |
| spherical Gaussian ℓ=800 km | 883.1 → 43.7 ms |
| 2-D top-hat, half-width 32 | 92.4 → 11.6 ms |
| 3-D Gaussian 96³ | 131.5 → 20.4 ms |
| three diagnostics | 148.8 MB → 2.9 kB |
| batch axis, 4 threads | 2.49× fixed-grid, 2.26× ragged, 2.59× profile |
| spectral sweep | `10·S` → `5 + 5S` transforms via analyze-once/synthesize-per-scale |

All serial paths allocate 0 B with scratch held; threaded paths carry only a constant task-spawn floor.

**Remaining:**

- **Julia 1.11** fails 9 allocation assertions. Decide: assert no-growth there, or raise the compat floor.
- **The banded engine's bounded-x edge columns** are still per-point — 63% of a row for a
  `radius = 10ℓ` kernel. Needs a per-column denominator scratch.
- **`Π` layout** was measured (strided per-scale target costs 1.002× at 256²) and left as
  `(spatial…, Nscales, batch…)`. Revisit only if the GPU batch path becomes dominant.
- **The 48-thread throughput check** that motivated the original performance work has never been rerun on
  that machine.

**Done when**: the 1.11 decision is made and encoded, the banded edge columns are either fixed or
documented with a measurement, and the 48-thread check is rerun.

---

## Stage 5 — Standing rules for this package

Learned from what went wrong, kept here so they are not relearned.

1. **A fix for a failing test may not shrink what the package does.** If an edit removes, narrows or
   opts something out of a capability, it is a workaround and needs saying out loud, not committing.
2. **"X doesn't support Y" is a claim needing evidence**, not a premise for a narrowing. Read the source
   or the paper first.
3. **Measure before asserting a performance or accuracy claim**, at realistic size, and gate complexity
   claims by *counting* operations rather than timing them.
4. **`@allocated` inside a testset charges the closure's captures to the call.** Measure through a
   top-level, fully-qualified helper.
5. **Tuple-slicing `size(A)` with a runtime range allocates.** Use `ntuple(..., Val(R))`.
6. **A capability that is not reachable from an entry point is not delivered**, and an unreachable kernel
   or constructor is worse than none.
7. **Physics conventions are decisions with citations**, not defaults to inherit silently — sign, `C` in
   `k_ℓ`, kernel `α`, `ℓ` as diameter, masking policy.
