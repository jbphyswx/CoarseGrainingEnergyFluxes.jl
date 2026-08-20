```@meta
CurrentModule = CoarseGrainingEnergyFluxes
```

# Theory

## Coarse-Graining Framework

The coarse-grained (filtered) field at scale ℓ is a convolution with a normalized kernel `G_ℓ`:

```
ū_ℓ(x) = ∫ G_ℓ(x, x') u(x') dA(x')
```

`G_ℓ` is normalized to unit mass, so a constant field filters to itself. On a masked domain the
integral runs over active cells only, and the default `ZeroFill` strategy keeps `G_ℓ` unchanged —
excluded cells simply contribute nothing. That preserves the property the rest of this page depends
on: a position-independent kernel **commutes with spatial derivatives**, which is the step that turns
the pointwise momentum equation into the filtered energy budget below. The alternative `Deformable`
strategy renormalizes over the locally-active area, recovering constants exactly next to a boundary at
the cost of that commutation. Under either strategy, points within `≈ℓ` of a mask boundary are
contaminated; see [`Filtering.filter_field!`](@ref).

### Sub-Scale Stress Tensor

The sub-filter-scale (SFS) stress captures the effect of motions smaller than ℓ:

```
τ_ℓ = (u ⊗ u)̄_ℓ − ū_ℓ ⊗ ū_ℓ
```

It is the local, scale-resolved analogue of the Reynolds stress.

### Cross-Scale Energy Flux

The filtered kinetic-energy budget (Aluie 2011; Aluie, Hecht & Vallis 2018) contains the cross-scale
flux

```
Π_ℓ(x) = −ρ₀ τ_ℓ : S̄_ℓ = −ρ₀ Σᵢⱼ τᵢⱼ S̄ᵢⱼ ,   S̄ = ½(∇ū + (∇ū)ᵀ)
```

- **Π > 0** — forward cascade (energy from large → small scales)
- **Π < 0** — inverse cascade (small → large)
- **⟨Π⟩ > 0** — net forward cascade across the domain at scale ℓ

In 2D the contraction is `S_xx τ_xx + 2 S_xy τ_xy + S_yy τ_yy`; in 3D it gains
`+ S_zz τ_zz + 2 S_xz τ_xz + 2 S_yz τ_yz` (see [`Diagnostics.compute_Π!`](@ref)).

### What a `Π`-only diagnostic does not tell you

`Π` is the cross-scale transfer term of the filtered kinetic-energy budget — not the budget. Under
exact homogeneity the transport term `⟨∇·J⟩` averages to zero, and **that is the only term `Π` alone
gets for free.** Baroclinic conversion and forcing/injection are not divergences and do not vanish
under averaging.

The size of the gap is not marginal. Loose et al. (2023) report cross-scale KE transfer as 35–40% of
baroclinic EKE production in the eddy-permitting regime, 70–100% at some latitudes, and >100% locally
— i.e. **the omitted baroclinic term is typically 2.5–3.5× larger than `Π`**. So `Π` supports
statements like "at this scale and place, energy moves upscale", and does *not* support "the mesoscale
energy came from upscale transfer rather than from baroclinic instability". Attribution needs the
other terms, which this package does not compute.

Relatedly, the correct large-scale energy is `E(ℓ) = ½⟨|ū_ℓ|²⟩` — the energy *of the filtered flow* —
and not the filtered energy `½⟨(|u|²)‾_ℓ⟩`, which is a different quantity that does not obey the
cascade budget. [`Diagnostics.cumulative_energy`](@ref) computes the former.

## Filter Kernels

### What the framework requires of a kernel

The results above are not valid for an arbitrary weighting. A kernel must be:

1. **Normalized**, `∫G dA = 1`, so a constant filters to itself and `ū` is a genuine local mean.
2. **Even** (`G(-x) = G(x)`), so `∫xG = 0` and filtering does not displace the field — a non-zero
   first moment would advect while smoothing, and every linear field would acquire a spurious `τ`.
3. **Position-independent**, so filtering commutes with `∇`. This is the step that turns the
   pointwise momentum equation into the filtered budget, and it is why `ZeroFill` is the default
   masking strategy; see [`Filtering.filter_field!`](@ref).
4. **Non-negative and of non-vanishing second moment**, if `Π` is to be interpreted as a local energy
   transfer. A kernel engineered to have `∫x²G = 0` (a "high-order" kernel) buys spectrum fidelity at
   the cost of taking negative values, which breaks realizability of `τ` — the flux framework and the
   spectrum framework want opposite things here.

Requirements 1 and 2 are gated in `runtests.jl` against closed forms; 3 is gated by the
`filter ∘ ∇ == ∇ ∘ filter` test. `TopHatKernel` and `GaussianKernel` satisfy all four.
`SharpSpectralKernel` satisfies 1–3 but its real-space `sinc` form takes negative values, so it fails
4 — usable for the spectrum, questionable for a pointwise `Π`.

### Scale convention

The filter scale `ℓ` is the **full filter width — a diameter, not a radius** (Pope 2000). The top-hat
spans the disk/ball of radius `ℓ/2`; the sharp-spectral cutoff sits at `k_c = π/ℓ`. FlowSieve uses the
same convention, so an `ℓ` is directly comparable; a source that reports a *radius* or a Gaussian
*standard deviation* is not, and needs converting first. Real-space weights are unnormalized — the
filtering routines divide by the running area/volume-weighted sum.

### Top-Hat (box) — `TopHatKernel`

Unit weight inside the disk/ball of radius `ℓ/2`, zero outside. The literature default
(Aluie et al. 2018, Storer et al. 2022). Spectral filtering needs `using SpecialFunctions` to load
the exact planar transfer function (Bessel `J₁`); see below.

### Gaussian — `GaussianKernel(; α = 6)`

```
G_ℓ(r) ∝ exp(−α (r/ℓ)²)
```

- `α = 6` (default) is the Pope/turbulence convention: per-component variance `σ² = ℓ²/(2α) = ℓ²/12`,
  the second moment of a top-hat of width `ℓ` **in one dimension**.
- `α = 4` reproduces FlowSieve's Gaussian, so `GaussianKernel(; α = 4)` is directly comparable to
  FlowSieve output.

Variance matching is dimension-dependent, and it is easy to get wrong: the Gaussian is separable so its
per-component variance is `ℓ²/(2α)` in every dimension, but the top-hat here is the **disk/ball** of
radius `ℓ/2`, not a separable box, so its per-component variance shrinks with dimension. Measured:

| | 1-D | 2-D (disk) | 3-D (ball) |
|---|---|---|---|
| top-hat `⟨x²⟩` | `ℓ²/12` | `ℓ²/16` | `ℓ²/20` |
| `α` matching it | 6 | 8 | 10 |

So the default `α = 6` is 15% wider in RMS than a 2-D top-hat at the same nominal `ℓ`. For a
like-for-like kernel comparison use `GaussianKernel(; α = 8)` in 2-D and `α = 10` in 3-D.

### Sharp Spectral — `SharpSpectralKernel`

Ideal low-pass: `Ĝ_ℓ(k) = 1` for `k ≤ π/ℓ`, else `0`. Perfect scale separation in spectral space;
the physical-space form is a slowly-decaying sinc.

### Spectral transfer functions

For spectral filtering, each mode of wavenumber magnitude `k` is multiplied by
[`Kernels.spectral_transfer`](@ref)`(kernel, k, ℓ)`, normalized so `Ĝ(0) = 1` (the mean is preserved):

| Kernel | `Ĝ(k, ℓ)` |
|--------|-----------|
| `GaussianKernel(α)`   | `exp(−k² ℓ² / 4α)` |
| `SharpSpectralKernel` | `1` if `k ≤ π/ℓ`, else `0` |
| `TopHatKernel`        | `2 J₁(kR)/(kR)`, `R = ℓ/2` (needs `using SpecialFunctions`) |

On the sphere, `GaussianKernel`/`SharpSpectralKernel` use the wavenumber of harmonic degree `l`,
`k_l = √(l(l+1)) / R`, via the same [`Kernels.spectral_transfer`](@ref). `TopHatKernel`'s spherical-cap
window genuinely needs the degree `l` itself (not just `k_l`), via the separate
[`Kernels.spectral_transfer_degree`](@ref) — the exact Legendre-polynomial cap-averaging function
(Jekeli 1981), no extra dependency needed.

## The Filtering Spectrum (Sadek & Aluie 2018)

Filtering at a continuum of scales yields a spectrum without windowing or periodicity assumptions.
The **cumulative** coarse-grained kinetic energy ([`Diagnostics.cumulative_energy`](@ref), their Eq. 15) is

```
E(ℓ) = ½ ρ₀ ⟨|ū_ℓ|²⟩ ,
```

a *cumulative* quantity. The **filtering spectral density** ([`Diagnostics.filtering_spectrum`](@ref), their
Eq. 14 — comparable to a Fourier energy spectrum) is its derivative with respect to the filtering
wavenumber `k_ℓ = L/ℓ`:

```
Ẽ(k_ℓ) = d/dk_ℓ [ ½ ρ₀ ⟨|ū_ℓ|²⟩ ] .
```

`L` is the region length (`L = 1` gives the FlowSieve convention `k_ℓ = 1/ℓ`).

### The `k_ℓ = C/ℓ` convention is not standardized

Three mappings are all in use, and they are not interchangeable:

| source | mapping | reached here by |
|---|---|---|
| Sadek & Aluie (2018) | `k_ℓ = L/ℓ`, `L` the domain size | `L = domain size` |
| Storer et al. (2022, 2023), FlowSieve | `k_ℓ = 1/ℓ` | `L = 1` (the default) |
| Rivera, Aluie & Ecke (2014) | `k_ℓ = 2π/ℓ` | `L = 2π` |

Under `k_ℓ = C/ℓ` the Jacobian is `dℓ/dk_ℓ = -ℓ²/C`, so **`k_ℓ` scales as `C` and the density `Ẽ`
scales as `1/C`.** Comparing amplitudes — or peak locations — against a Fourier spectrum or against
another code therefore means nothing unless the conventions have been matched first. The cumulative
`E(ℓ)` is convention-free; only the density is not.

### Two limits on the density

- **Slope ceiling.** Sadek & Aluie's Eq. 18: a kernel with `p` vanishing moments recovers a true
  `k^{-α}` spectrum only while `α < p + 2`, and otherwise saturates at `k^{-(p+2)}`. `TopHatKernel`
  and `GaussianKernel` both have `p = 1`, so the recovered slope **locks at `k⁻³`** — which is
  precisely the 2-D enstrophy-range target slope, so this bites hardest in QG work. `Π` is unaffected;
  the ceiling is a property of the spectrum diagnostic alone.
- **Positive-definiteness.** `Ẽ(k_ℓ) ≥ 0` is guaranteed (their Eq. 21) only when `d|Ĝ(k)|²/dk ≤ 0` on
  `(0, ∞)`. The Gaussian and sharp-spectral kernels satisfy it; the top-hat does not — its `|Ĝ|²`
  rises again by `+0.0026` near `kℓ ≈ 8.8`. [`Diagnostics.filtering_spectrum`](@ref) and
  [`coarse_grain`](@ref) therefore refuse a non-conforming kernel rather than return a density that
  may go negative; see [`Kernels.transfer_monotone`](@ref).

## Decompositions

### Leonard / Cross / Reynolds — `tau_decomposition`

Germano's (1992) split of the stress into generalized central moments, each individually Galilean
invariant, with `L + C + R = τ` exactly: resolved–resolved (Leonard), resolved–subfilter (Cross), and
subfilter–subfilter (Reynolds, the backscatter-carrying term). On spherical grids, `tau_decomposition`
takes the moments in planetary-Cartesian coordinates (the same frame `compute_Π!` uses so that
filtering commutes with the moments — Aluie 2019) and rotates the result back to local east/north
components — building it from raw local `(u,v)` components directly, without this rotation, would be
frame-inconsistent on a sphere.

### Rotational / divergent (Helmholtz) — `compute_Π_decomposed`

Given the rotational (solenoidal) velocity `uʳ` (from a Helmholtz solver such as
[HelmholtzDecomposition.jl](https://github.com/jbphyswx/HelmholtzDecomposition.jl); the divergent part
`uᵈ = u - uʳ` is the complement), a **both-sided** split is required for a physically complete
decomposition: not just the stress, but also the strain, since `S̄ = S̄ʳ + S̄ᵈ` in general. Splitting
only the stress against the *full*, undecomposed strain (`Π ≟ -S̄:τʳʳ - S̄:τ_cross - S̄:τᵈᵈ`) silently
drops the `S̄ᵈ:τʳʳ` and `S̄ʳ:τᵈᵈ` cross-terms whenever the divergent strain is non-negligible — an
incomplete decomposition, not merely a naming difference (Wang et al.'s extension of Aluie's
framework; Barkan, Srinivasan & McWilliams 2024). The stress itself is also bilinear in the two
velocity parts, `τ = τʳʳ + τᵈᵈ + τ_X` (`τ_X` the rotational–divergent cross stress), so contracting
both sides in full gives three exact channels:

```
Π_RR = −S̄ʳ : τʳʳ                                          (rotational → rotational)
Π_DD = −S̄ᵈ : τᵈᵈ                                          (divergent  → divergent)
Π_X  = −( S̄ʳ:τᵈᵈ + S̄ᵈ:τʳʳ + S̄ʳ:τ_X + S̄ᵈ:τ_X )              (every interaction term)

Π = Π_RR + Π_X + Π_DD   exactly, to machine precision
```

`compute_Π_decomposed` returns `(; total, rotational, cross, divergent)` with `total = rotational .+
cross .+ divergent`; the three channels hold `Π_RR`, `Π_X`, and `Π_DD` respectively. `Π_X` is the
"stimulated cascade" / interaction channel of Barkan, Srinivasan & McWilliams (2024) — energy
exchanged *between* the rotational and divergent parts of the flow, which the one-sided
(stress-only) split cannot represent at all.

### Strain / convergence — `compute_Π_strain_convergence`

Srinivasan, Barkan & McWilliams (2023) eq. (10) splits the same `Π` a different way, by diagonalizing
the filtered strain tensor rather than by decomposing the velocity. With the rotation invariants
`δ̄ = ū_x + v̄_y` (divergence) and `ᾱ = √(σ̄_n² + σ̄_s²)` (strain magnitude), built from the normal and
shear strains `σ̄_n = ū_x − v̄_y` and `σ̄_s = ū_y + v̄_x`,

```
Π = Π_α − Π_δ ,   Π_α = (τ_vv − τ_uu) σ̄_n/2 − τ_uv σ̄_s ,   Π_δ = (τ_vv + τ_uu) δ̄/2
```

`Π_α` is **deformation/shear production** — present in non-divergent flow, and the only term when
`δ̄ = 0`, which recovers Polzin (2010). `Π_δ` is **convergence production**, the frontogenetic term
that paper adds; it vanishes identically for a non-divergent field.

Unlike the Helmholtz split, this is not new information: expanding eq. (10) collapses exactly to
`−τ_uu ū_x − τ_uv(ū_y + v̄_x) − τ_vv v̄_y`, the direct `−S̄:τ̄`. Its value is (a) the physical reading —
`Π` binned against `δ̄` and `ᾱ`, which the function also returns — and (b) that the two forms contract
different combinations of the same derivatives, so agreeing to round-off is a real check on both. The
suite asserts that agreement on masked and unmasked grids, for both kernels and both mask strategies.

### Tracer / buoyancy variance flux — `tracer_variance_flux`

The scalar analogue of Π for a tracer θ (Aluie & Eyink):

```
Πθ = −∂_j θ̄ · τ_j(u, θ) ,   τ_j = ⟨u_j θ⟩ − ū_j θ̄ .
```

With θ = buoyancy this is the cross-scale buoyancy-variance (APE-related) transfer, needing only
`(u, v, θ)`.

`τ_j` is a vector, so on a spherical grid it follows the same commutativity argument as the momentum
stress: the velocity is rotated to planetary Cartesian before filtering and `τ` rotated back to the
local east/north(/radial) frame before contracting with `∂_j θ̄`. Filtering the local components in
place would average vectors expressed in a basis that turns from point to point.

## Vertical structure: vertical-profile vs. true 3D

In 2.5D, `compute_Π!` deliberately drops the vertical-shear strain terms (`S_xz`, `S_yz`, `S_zz` are
either omitted or zero) whenever only `(u, v)` (optionally `w`) is supplied on a 2D grid. This is not
an oversight: it is the standard thin-layer/quasi-geostrophic scaling (Vallis, *Atmospheric and
Oceanic Fluid Dynamics*; Pedlosky, *Geophysical Fluid Dynamics*), valid when the aspect ratio
`δ = H/L` (vertical/horizontal scale) is small — the normal regime for large-scale ocean/atmosphere
flow, where vertical shear is genuinely subdominant to horizontal gradients. The actual "vertical
structure via coarse-graining" literature (Aluie, Hecht & Vallis 2018; the
Buzzicotti/Storer/Khatri/Griffies/Aluie line of work) does not compute a coupled vertical-derivative
tensor either — it runs this same 2D/2.5D method **independently at each vertical level** of a
multi-level model and compares/stacks the resulting profiles. `coarse_grain_profile` implements exactly
this: given 3D `(x, y, z)` arrays it treats the vertical axis as a batch over the shared horizontal
grid, sweeping the existing 2D/2.5D `coarse_grain!` independently per level and returning the stacked
profile — no new tensor math, and the level axis parallelizes like any other batch axis.

Separately, and not to be conflated with the above, `compute_Π!` also has a genuinely **coupled true
3D** method (`StructuredGrid{...,3}`, Cartesian or spherical-volumetric): all nine strain/stress
components, real vertical derivatives, one filter kernel blending all three directions at once. This
targets a different, narrower-audience physics case — homogeneous/isotropic turbulence (e.g.
Rayleigh–Taylor or boundary-layer studies) — where the thin-layer assumption above does not hold and
levels genuinely interact through the filter.

## Spherical Geometry

On `S²` of radius `R`, convolution uses the great-circle distance

```
d(x, x') = R · arccos(sin φ sin φ' + cos φ cos φ' cos(λ − λ')) ,
```

with area element `dA = R² cos φ dλ dφ`. The true 3D spherical-volumetric `StructuredGrid` (a
`(lon, lat, radius)` axis triple, radius stored directly rather than a depth/height sign convention)
uses the corresponding volume element at each level's *local* radius `r[k]`,

```
dV = r[k]² cos φ dλ dφ dr ,
```

with horizontal arc-length spacing `r[k] cos φ · Δλ` / `r[k] · Δφ` and radial spacing `Δr` feeding the
3D `ddx!`/`ddy!`/`ddz!` stencils, and the full 3×3 planetary-Cartesian tensor rotation in
`compute_Π!` extended to include the radial direction now that real multi-level vertical derivatives
exist (as opposed to the 2.5D flat-layer assumption above).

### Commutativity on the sphere

Aluie (2019) shows that filtering vector components as scalars does **not** commute with `∇` on `S²`.
This package transforms velocities to planetary-Cartesian components before filtering and back
afterward (`Geometry.vector_to_cartesian` / `Geometry.vector_from_cartesian`), which is exact for non-divergent
flow (Storer et al. 2022). For strongly divergent flow, decompose first (HelmholtzDecomposition.jl)
and use [`Diagnostics.compute_Π_decomposed`](@ref).

There is a second, separate loss of commutativity that no choice of vector convention removes: a
great-circle kernel is homogeneous in *distance*, while `∂/∂x` is taken along `λ` whose metric factor
`h_λ = R cos φ` varies across the footprint. So even for a scalar, on an unmasked grid, with a
position-independent kernel, `filter ∘ ∂/∂x ≠ ∂/∂x ∘ filter` on the sphere. Measured on a 1° grid with
a top-hat, as a relative error against `max|∂(ū)/∂x|`:

| `ℓ` | at 0°N | at 40°N | at 65°N |
|---|---|---|---|
| 222 km (footprint ≈ 1 cell) | 1.7e-13 | 8.0e-14 | 6.0e-14 |
| 444 km | 1.2e-4 | 2.2e-3 | 6.3e-3 |
| 888 km | 7.2e-4 | 1.3e-2 | 3.2e-2 |

The error is round-off while the footprint spans a single latitude row and grows with both `ℓ/R` and
`|φ|`, reaching **3% at 65° for `ℓ = 888 km`**. That is a floor on how exactly the `Π` budget can close
on a sphere at large `ℓ` and high latitude, independent of masking or of the flat-metric commutation
that [`Filtering.filter_field!`](@ref)'s `mask_strategy` note is about. The full remedy is Aluie
(2019)'s spectrally shifted kernels, which this package does not implement.

## Curvilinear & unstructured grids: WLSQ gradients

`CurvilinearGrid` and `UnstructuredGrid` have no fixed axis spacing to difference against, so the
gradient (`FG.Connectivity.gradient_plan` + `FG.Discretization.gradient!`, which returns both tangent
components from one neighbour sweep) is reconstructed from a local weighted-least-squares (WLSQ) fit over each point's neighbor
stencil (its 4 index-offset neighbors on a curvilinear mesh; its k-d tree neighbors on a scattered
point cloud), projected into the local tangent plane (`project_to_tangent_plane` — an exact 3D-chord
projection for spherical grids, not a small-angle approximation). This is **not** the same as
inverting a 2×2 local Jacobian built from two independently-differenced index directions: dividing
two independently-differenced quantities does not preserve 2nd-order accuracy unless the specific
combination cancels the leading error term, which a proper WLSQ fit does and a raw Jacobian inverse
does not. WLSQ gradients are conditionally 2nd order — degrading toward 1st order on strongly skewed
local stencils, a known, expected property (not a silent surprise), verified directly against an
adversarial-stencil test.

## References

- Aluie, H. (2011). Compressible turbulence: the cascade and its locality. *Phys. Rev. Lett.* 106(17).
- Aluie, H. (2019). Convolutions on the sphere: commutation with differential operators. *GEM* 10(1). doi:10.1007/s13137-019-0123-9
- Aluie, H., Hecht, M., & Vallis, G. K. (2018). Mapping the energy cascade in the North Atlantic Ocean. *J. Phys. Oceanogr.* 48(8). doi:10.1175/JPO-D-17-0100.1
- Barkan, R., Srinivasan, K., & McWilliams, J. C. (2024). Eddy–internal wave interactions: stimulated cascades in cross-scale kinetic energy and enstrophy fluxes. *J. Phys. Oceanogr.* 54(6), 1309–1326. doi:10.1175/JPO-D-23-0191.1
- Germano, M. (1992). Turbulence: the filtering approach. *J. Fluid Mech.* 238. doi:10.1017/S0022112092001733
- Loose, N., Bachman, S., Grooms, I., & Jansen, M. (2023). Diagnosing scale-dependent energy cycles in a high-resolution isopycnal ocean model. *J. Phys. Oceanogr.* 53, 157.
- Pedlosky, J. (1987). *Geophysical Fluid Dynamics* (2nd ed.). Springer.
- Sadek, M., & Aluie, H. (2018). Extracting the spectrum of a flow by spatial filtering. *Phys. Rev. Fluids* 3, 124610. doi:10.1103/PhysRevFluids.3.124610
- Srinivasan, K., Barkan, R., & McWilliams, J. C. (2023). A forward energy flux at submesoscales driven by frontogenesis. *J. Phys. Oceanogr.* 53(1), 287–305. doi:10.1175/JPO-D-22-0001.1
- Storer, B. A. et al. (2022). Global energy spectrum of the general oceanic circulation. *Nat. Commun.* 13, 5314. doi:10.1038/s41467-022-33031-3
- Vallis, G. K. (2017). *Atmospheric and Oceanic Fluid Dynamics* (2nd ed.). Cambridge University Press.
