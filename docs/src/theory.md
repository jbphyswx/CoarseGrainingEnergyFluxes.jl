```@meta
CurrentModule = CoarseGrainingEnergyFluxes
```

# Theory

## Coarse-Graining Framework

The coarse-grained (filtered) field at scale ℓ is a convolution with a normalized kernel `G_ℓ`:

```
ū_ℓ(x) = ∫ G_ℓ(x, x') u(x') dA(x')
```

`G_ℓ` is normalized to unit mass over all space, so a constant field filters to itself. The default
`ZeroFill` strategy filters the field extended by zero over land and past a bounded edge — land is
fluid at rest (Aluie et al. 2018; Grooms et al. 2021, eq. 7) — with `G_ℓ` unchanged: the denominator
is the kernel's mass over the lattice continued past each bounded edge at its edge spacing, a latitude
to the pole and a regional longitude around the ring, and for the top-hat weighing cells by their overlap
the measure of its support itself. That preserves the property the rest of this
page depends on: a position-independent kernel **commutes with spatial derivatives**, which is the
step that turns the pointwise momentum equation into the filtered energy budget below. `ū_ℓ` and `Π`
are therefore defined over land too, `Π` vanishing beyond the kernel's reach of the coast, and a domain
mean sums every cell and divides by the water area (Storer et al. 2022). The alternative `Deformable`
strategy renormalizes over the locally-active area and zeroes masked cells, recovering constants
exactly next to a boundary at the cost of that commutation; see [`Filtering.filter_field!`](@ref).

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
Π_ℓ(x) = −τ_ℓ : S̄_ℓ = −Σᵢⱼ τᵢⱼ S̄ᵢⱼ ,   S̄ = ½(∇ū + (∇ū)ᵀ)
```

per unit mass, in m² s⁻³; `ρ₀ Π_ℓ` is the flux per unit volume.

- **Π > 0** — forward cascade (energy from large → small scales)
- **Π < 0** — inverse cascade (small → large)
- **⟨Π⟩ > 0** — net forward cascade across the domain at scale ℓ

In 2D the contraction is `S_xx τ_xx + 2 S_xy τ_xy + S_yy τ_yy`; in 3D it gains
`+ S_zz τ_zz + 2 S_xz τ_xz + 2 S_yz τ_yz` (see [`Diagnostics.compute_Π!`](@ref)).

### Derivatives of the filtered field

`S̄` differentiates `ū`, which every filter defines at every point. Every diagnostic that differentiates a
filtered field takes `derivative` ([`Derivatives.AbstractDerivative`](@ref)):

- `PhysicalDerivative(; nodes = 5)` differences `ū`'s values: FlowGeometries' stencils of `nodes` nodes on a
  `StructuredGrid`, a window that reaches an inactive cell shrinking to the cell's active run, and its
  least-squares tangent-plane gradient on every other grid. A centred stencil of `n` nodes is of order `n − 1`,
  so its error on `ū`, smooth on the scale `ℓ`, falls as `(Δ/ℓ)^{n−1}`.
- `SpectralDerivative()` differentiates the filter's own output. A spectral filter's output is the series
  `ū(x) = Σ_k w_k Ĝ(k) F̂(k) e^{ik·x}` (`w_k` splitting each Nyquist pair), and `Σ_k w_k i k_d Ĝ(k) F̂(k) e^{ik·x}`
  is its derivative, exact for that series; a renormalized filter `ū = N/M` takes the quotient rule
  `∂ū = (∂N − ū ∂M)/M`. Under the generalized convolution the strain comes from the filtered spin-1
  coefficients through one spin-0 and one spin-2 synthesis, the spin-lowering and raising operators scaling a
  degree-`ℓ` coefficient by `√(ℓ(ℓ+1))/R` and `√((ℓ−1)(ℓ+2))/R`; under `CartesianComponents` each ambient
  component's surface gradient is a spin-1 synthesis of its scalar coefficients scaled by `√(ℓ(ℓ+1))/R`, rotated
  to the local frame. A plan with no series raises.
- `AutoDerivative()`, the default, takes the spectral derivative where the caller filters spectrally
  (`method = Spectral()`, and the generalized convolution) and `PhysicalDerivative()` elsewhere, whichever
  engine evaluates a `RealSpace()` filter.

`enstrophy_flux` takes the vorticity of the unfiltered velocity by differences under every route, since only a
filtered field is a series; the gradient of the filtered vorticity follows the route.

### What a `Π`-only diagnostic does not tell you

`Π` is the cross-scale transfer term of the filtered kinetic-energy budget, one term of several.
Under exact homogeneity the transport term `⟨∇·J⟩` averages to zero; it is the only other term that
drops out. Baroclinic conversion and forcing/injection are not divergences and do not vanish under
averaging.

Loose et al. (2023) report cross-scale KE transfer as 35–40% of baroclinic EKE production in the
eddy-permitting regime, 70–100% at some latitudes, and >100% locally. So `Π` supports statements like
"at this scale and place, energy moves upscale", and does not by itself attribute the mesoscale energy
to upscale transfer: weighing it against baroclinic instability needs the other terms, which this
package does not compute.

The large-scale energy is `E(ℓ) = ½⟨|ū_ℓ|²⟩`, the energy *of the filtered flow*, which
[`Diagnostics.cumulative_energy`](@ref) computes. The filtered energy `½⟨(|u|²)‾_ℓ⟩` is a different
quantity and does not obey the cascade budget.

## Filter Kernels

### What the framework requires of a kernel

The results above hold for a kernel that is:

1. **Normalized**, `∫G dA = 1`, so a constant filters to itself and `ū` is a local mean.
2. **Even** (`G(-x) = G(x)`), so `∫xG = 0` and filtering does not displace the field. A kernel with a
   non-zero first moment advects while it smooths, and gives every linear field a spurious `τ`.
3. **Position-independent**, so filtering commutes with `∇`. Commutation turns the pointwise momentum
   equation into the filtered budget, and the default masking strategy `ZeroFill` preserves it; see
   [`Filtering.filter_field!`](@ref).
4. **Non-negative and of non-vanishing second moment**, if `Π` is to be interpreted as a local energy
   transfer. A kernel with `∫x²G = 0` (a "high-order" kernel) improves spectrum fidelity and takes
   negative values, which breaks realizability of `τ`: the flux and the spectrum ask opposite things
   of the kernel here.

Requirements 1 and 2 hold by construction: every engine divides by the window mass, and every kernel is a
function of distance or, for `HighOrderKernel` on the plane, a product of even profiles. Requirement 3 is
gated by the `filter ∘ ∇ == ∇ ∘ filter` test (`test_filtering.jl`), and 4 by the sign checks of
`test_kernels.jl`. `TopHatKernel` and `GaussianKernel` satisfy all four.
`SharpSpectralKernel` satisfies 1–3 but its real-space form takes negative values, so it fails 4 —
usable for the spectrum, questionable for a pointwise `Π`.

### Scale convention

The filter scale `ℓ` is the **full filter width, a diameter** (Pope 2000). The top-hat
spans the disk/ball of radius `ℓ/2`; the sharp-spectral cutoff sits at `k_c = π/ℓ`. FlowSieve uses the
same convention, so an `ℓ` is directly comparable; a source that reports a *radius* or a Gaussian
*standard deviation* is not, and needs converting first. Real-space weights are unnormalized — the
filtering routines divide by the running area/volume-weighted sum.

### Top-Hat (box) — `TopHatKernel`

Unit weight inside the disk/ball of radius `ℓ/2`, zero outside. The literature default
(Aluie et al. 2018, Storer et al. 2022). Spectral filtering needs `using SpecialFunctions` to load
the exact planar transfer function (Bessel `J₁`); see below.

On a grid the top-hat's sum over cells weighs each cell as `cells` says ([`Kernels.AbstractCellWeight`](@ref)).
The default, `CellOverlap()`, weighs a cell by its measure inside the support, the continuous top-hat integrated
over the cell, so the filtered field varies continuously with `ℓ` and with the target's position. The support is
the segment, disk or ball of a Cartesian grid, the spherical cap on a sphere, the geodesic disk on a spheroid,
and the ball of chord radius `ℓ/2` in a spherical or spheroidal shell. Each fraction is a closed form (a box or a
polygon against a disk) or a Gauss–Legendre quadrature of one between the exact breakpoints of the integrand.
`CellCentre()` weighs a cell by its whole measure where its centre lies within `ℓ/2` and by nothing otherwise,
so the filter steps as `ℓ` crosses the cell centres. A node set, whose points carry no cells, takes the centre
test under either.

A top-hat's `Π` on a grid converges to the continuous filter's at first order in `Δ/ℓ` under either rule: the
kernel's edge passes grid-scale content into `ū`. The centre test's steps add to that error at small `ℓ/Δ`, and
dominate the filtering spectrum there, which differentiates `E(ℓ)` in `ℓ`. A smooth kernel's `Π` converges
faster than any power of `Δ/ℓ` on a uniform grid.

### Gaussian — `GaussianKernel(; α = 6)`

```
G_ℓ(r) ∝ exp(−α (r/ℓ)²)
```

- `α = 6` (default) is the Pope/turbulence convention: per-component variance `σ² = ℓ²/(2α) = ℓ²/12`,
  the second moment of a top-hat of width `ℓ` **in one dimension**.
- `α = 4` reproduces FlowSieve's Gaussian, so `GaussianKernel(; α = 4)` is directly comparable to
  FlowSieve output.

Variance matching is dimension-dependent. The Gaussian is separable, so its per-component variance is
`ℓ²/(2α)` in every dimension. The top-hat is the **disk/ball** of radius `R = ℓ/2`, whose
per-component variance `R²/(D+2)` shrinks with the dimension `D`:

| | 1-D | 2-D (disk) | 3-D (ball) |
|---|---|---|---|
| top-hat `⟨x²⟩` | `ℓ²/12` | `ℓ²/16` | `ℓ²/20` |
| `α` matching it | 6 | 8 | 10 |

So the default `α = 6` is 15% wider in RMS than a 2-D top-hat at the same nominal `ℓ`. For a
like-for-like kernel comparison use `GaussianKernel(; α = 8)` in 2-D and `α = 10` in 3-D.

### Sharp Spectral — `SharpSpectralKernel`

Ideal low-pass: `Ĝ_ℓ(k) = 1` for `k ≤ k_c = π/ℓ`, else `0`. `method = Spectral()` multiplies by it.
`RealSpace()` convolves with its inverse transform in the grid's `D` dimensions,

```
G(r) ∝ J_{D/2}(k_c r) / (k_c r)^{D/2}:    sin x / x,   2 J₁(x)/x,   3 (sin x − x cos x)/x³,    x = k_c r,
```

for `D = 1, 2, 3` (the two-dimensional form needs `using SpecialFunctions`). `G` decays as
`r^{−(D+1)/2}`, so `∫|G|` diverges and no finite footprint reproduces the brick wall: the truncated
kernel's transfer function ripples about it, most near `k_c`, with an amplitude falling as
`1/(|k_c − k| R)` in the truncation radius `R`. The footprint ends near `10ℓ`, at the root of the
truncated mass, `(2/π) Si(k_c R) = 1`, `J₀(k_c R) = 0` or `(2/π)(Si(k_c R) − sin k_c R) = 1`, so the
normalized passband is centred on 1. Its weights change sign, so under `Deformable` a window mass
beside a coast can approach zero.

### Spectral transfer functions

For spectral filtering, each mode of wavenumber magnitude `k` is multiplied by
[`Kernels.spectral_transfer`](@ref)`(kernel, k, ℓ)`, normalized so `Ĝ(0) = 1` (the mean is preserved):

| Kernel | `Ĝ(k, ℓ)` |
|--------|-----------|
| `GaussianKernel(α)`   | `exp(−k² ℓ² / 4α)` |
| `SharpSpectralKernel` | `1` if `k ≤ π/ℓ`, else `0` |
| `TopHatKernel`        | `2 J₁(kR)/(kR)`, `R = ℓ/2` (needs `using SpecialFunctions`) |
| `HighOrderKernel`     | `∏_d Ĝ₁(k_d)` over the wavevector, `Ĝ₁(k) = Σₙ cₙ 2ρₙ sinc(kρₙ) / Σₙ cₙ 2ρₙ` with `ρₙ = ℓ/2 + nb` and `cₙ = aₙ − aₙ₊₁` |

Along a bounded direction the spectral filter is, as the real-space one is, the convolution of the field
extended by zero. The transform pads the direction until the kernel at the largest scale it serves
(`max_scale` of [`Filtering.spectral_grid_plan`](@ref), the largest of a sweep's scales) stays clear of the
domain's periodic copies, and a plan applied at a larger scale raises. Where a kernel's transfer at the grid's
Nyquist wavenumber is not negligible (a Gaussian below `ℓ ≈ 8Δ`), the lattice cuts it and leaves a slowly
decaying tail in real space, so the result depends on the padding by about `Ĝ(π/Δ)/(π g)` for a gap of `g`
cells.

The sharp kernel reaches without bound, so no padding serves it. On a uniform grid of two directions the FFT
path filters by the lattice brick-wall kernel over the offsets the domain holds, whose transform on the padded
box is exact: with both directions bounded `h(o) = ΔxΔy k_c²/(4π) · 2J₁(k_c r)/(k_c r)`, `r = |o|`, and with one
periodic, per wavenumber `k` of the periodic direction `h(o) = Δ sin(a oΔ)/(π oΔ)` along the bounded one,
`a = √(k_c² − k²)`, zero where `|k| > k_c`. It needs `ℓ ≥ Δ` along a bounded direction. A nonuniform-FFT plan with
a bounded direction raises for the sharp kernel.

On a sphere of radius `R` a radial kernel multiplies every harmonic of degree `l` by its Legendre
transform (Aluie 2019, eq. 34, normalized as eq. 10; the Funk–Hecke theorem, eqs. 36–37), taken over the
support the real-space engines keep:

```
Ĝ(l) = ∫₀^γₜ G(Rγ) P_l(cos γ) sin γ dγ / ∫₀^γₜ G(Rγ) sin γ dγ ,   γₜ = min(kernel_radius(kernel, ℓ, Val(2))/R, π) .
```

[`Kernels.spectral_transfer_degrees!`](@ref) fills `Ĝ(0 … L)` at once. A spherical cap (`TopHatKernel`,
and each shell of `HighOrderKernel`'s isotropic form, weighted by its mass) has the closed form of Jekeli
(1981), `Ĝ(l) = (1 + x)P′_l(x)/(l(l+1))` with `x = cos γₜ`; every other kernel is integrated by
Gauss–Legendre quadrature in `γ`, with enough points to integrate `G(Rγ)P_l(cos γ) sin γ` to round-off
for every `l ≤ L`. Both run in the element type of the transfer, the grid's. The spectral filter on a
sphere is thus the transform of the real-space filter's own kernel: `SharpSpectralKernel` there is the
Legendre transform of its truncated inverse transform, which ripples about the brick wall as on the
plane, and `SmoothHatKernel` and `HyperGaussianKernel`, which have no closed planar transform, have a
degree transfer.

## The Filtering Spectrum (Sadek & Aluie 2018)

Filtering at a continuum of scales yields a spectrum without windowing or periodicity assumptions.
The **cumulative** coarse-grained kinetic energy ([`Diagnostics.cumulative_energy`](@ref), their Eq. 15) is

```
E(ℓ) = ½ ⟨|ū_ℓ|²⟩ ,
```

per unit mass, a *cumulative* quantity. The **filtering spectral density** ([`Diagnostics.filtering_spectrum`](@ref), their
Eq. 14 — comparable to a Fourier energy spectrum) is its derivative with respect to the filtering
wavenumber `k_ℓ = L/ℓ`:

```
Ẽ(k_ℓ) = d/dk_ℓ [ ½ ⟨|ū_ℓ|²⟩ ] .
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
  `(0, ∞)`. The Gaussian and sharp-spectral kernels satisfy it; the top-hat does not — `Ĝ = 2J₁(kR)/(kR)`
  falls to zero at `kℓ ≈ 7.66` and its `|Ĝ|²` then climbs back to `0.0175` at `kℓ ≈ 10.27`, the first
  Airy sidelobe, with later ones at `0.0042` and `0.0016`. The condition is **sufficient but not
  necessary**, and the violation here is under 2% of the DC value and confined to the far sub-filter
  tail. By default [`Diagnostics.filtering_spectrum`](@ref) throws for a non-conforming kernel and
  [`coarse_grain`](@ref) fills its density with `NaN`; `Diagnostics.ForceSpectrum()` computes the
  density anyway and leaves the sign check to the caller. See
  [`Diagnostics.AbstractSpectrumPolicy`](@ref) and [`Kernels.transfer_monotone`](@ref).

### The derivative in `ℓ`

`E(ℓ) = ½⟨|ū_ℓ|²⟩` gives `dE/dℓ = ⟨ū_ℓ · ∂_ℓ ū_ℓ⟩` exactly, and `Ẽ(k_ℓ) = −(ℓ²/L) dE/dℓ`. `scale_derivative`
([`Diagnostics.AbstractScaleDerivative`](@ref)) chooses how `∂_ℓ ū` is taken:

- `ExactScaleDerivative()` through the filter itself. A spectral filter or the generalized convolution
  synthesizes with the transfer `∂_ℓĜ`, and a renormalized one takes the quotient rule over its mask's
  series. A real-space filter is `ū = N_G/M_G`, `N_G = Σ g(r/ℓ) u A` and `M_G` its window mass, so with
  `H(s) = −s g′(s)` ([`Kernels.ScaleRate`](@ref)), `∂_ℓ ū = (M_H/M_G)(ū_H − ū)/ℓ`, `ū_H` the filter of kernel `H`
  over the same taps: the derivative of the discrete filter as it is evaluated.
- `DifferenceScaleDerivative()`, the slope at each scale of the quadratic through the three nearest scales'
  `E`, second order in the scale spacing.
- `AutoScaleDerivative()`, the default: exact where the kernel has the derivative on the filter's route, and
  differences elsewhere. The Gaussian, smooth hat and hyper-Gaussian have it in real space; the Gaussian,
  top-hat and `HighOrderKernel` in a planar spectral filter; the Gaussian, smooth hat, hyper-Gaussian and top-hat
  by degree on a sphere. The top-hat's `H` is a ring at its rim, `HighOrderKernel`'s sits at its edges and the
  sharp kernel's `∂_ℓĜ` is a delta, so their real-space footprints take differences.

With `slopes = Slopes()` a sweep also returns the spectrum's slope `d ln Ẽ/d ln k_ℓ` at each scale. With
`a = ℓ∂_ℓū` and `b = ℓ²∂²_ℓū` it is `−(2⟨ū·a⟩ + ⟨a·a⟩ + ⟨ū·b⟩)/⟨ū·a⟩` on an exact route, the second derivative
taken as the first is, and the slope of the quadratic through the three nearest scales' `ln Ẽ` in `ln k_ℓ`
otherwise. `spectrum_maps = SpectrumMaps()` keeps the local spectrum `Ẽ(x, k_ℓ) = −(ℓ/L) ū·a` at every cell
(shaped like `Π`, its area mean the filtering spectrum) and, with `Slopes()`, its slope per cell.
[`Filtering.plan_filter_sweep`](@ref)`(…; scale_rates = ScaleRates())` builds a real-space family's rate plans
with it, for a sweep repeated over timesteps.

## Decompositions

### Leonard / Cross / Reynolds — `tau_decomposition`

Germano's (1992) split of the stress into generalized central moments, each individually Galilean
invariant, with `L + C + R = τ` exactly: resolved–resolved (Leonard), resolved–subfilter (Cross), and
subfilter–subfilter (Reynolds, the backscatter-carrying term). On a sphere `tau_decomposition` filters
the velocity and every product tensor by the vector filter `compute_Π!` uses
([Filtering vectors and tensors on the sphere](@ref)); each filter is linear, so the three moments sum to
that filter's `τ` and come back in local east/north components.

### Rotational / divergent (Helmholtz) — `compute_Π_decomposed`

Given the rotational (solenoidal) velocity `uʳ` (from a Helmholtz solver such as
[HelmholtzDecomposition.jl](https://github.com/jbphyswx/HelmholtzDecomposition.jl); the divergent part
`uᵈ = u - uʳ` is the complement), a **both-sided** split is required for a physically complete
decomposition: the strain as well as the stress, since `S̄ = S̄ʳ + S̄ᵈ` in general. Splitting
only the stress against the *full*, undecomposed strain (`Π ≟ -S̄:τʳʳ - S̄:τ_cross - S̄:τᵈᵈ`) drops
the `S̄ᵈ:τʳʳ` and `S̄ʳ:τᵈᵈ` cross-terms whenever the divergent strain is non-negligible, leaving the
decomposition incomplete (Wang et al.'s extension of Aluie's framework; Barkan, Srinivasan &
McWilliams 2024). The stress itself is also bilinear in the two
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

Srinivasan, Barkan & McWilliams (2023) eq. (10) splits the same `Π` by diagonalizing the filtered
strain tensor. With the rotation invariants
`δ̄ = ū_x + v̄_y` (divergence) and `ᾱ = √(σ̄_n² + σ̄_s²)` (strain magnitude), built from the normal and
shear strains `σ̄_n = ū_x − v̄_y` and `σ̄_s = ū_y + v̄_x`,

```
Π = Π_α − Π_δ ,   Π_α = (τ_vv − τ_uu) σ̄_n/2 − τ_uv σ̄_s ,   Π_δ = (τ_vv + τ_uu) δ̄/2
```

`Π_α` is **deformation/shear production** — present in non-divergent flow, and the only term when
`δ̄ = 0`, which recovers Polzin (2010). `Π_δ` is **convergence production**, the frontogenetic term
that paper adds; it vanishes identically for a non-divergent field.

Expanding eq. (10) collapses exactly to `−τ_uu ū_x − τ_uv(ū_y + v̄_x) − τ_vv v̄_y`, the direct
`−S̄:τ̄`, so the split adds no new information. Its value is (a) the physical reading —
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

`τ_j` is a vector, so on a spherical grid `u` and `uθ` are filtered by the vector filter of the momentum
stress ([Filtering vectors and tensors on the sphere](@ref)), `θ̄` by the same transforms at spin 0, and
`τ` is contracted with `∂_j θ̄` in the local east/north frame. In a spherical shell `τ` is formed in ambient Cartesian components
and rotated to the local east/north/radial frame.

## Vertical structure: vertical-profile vs. true 3D

In 2.5D, `compute_Π!` omits the vertical-shear strain terms (`S_xz`, `S_yz`, `S_zz` are omitted or
zero) whenever only `(u, v)` (optionally `w`) is supplied on a 2D grid. This is the standard
thin-layer/quasi-geostrophic scaling (Vallis, *Atmospheric and Oceanic Fluid Dynamics*; Pedlosky,
*Geophysical Fluid Dynamics*), valid when the aspect ratio `δ = H/L` (vertical/horizontal scale) is
small — the normal regime for large-scale ocean/atmosphere flow, where vertical shear is subdominant
to horizontal gradients. The "vertical structure via coarse-graining" literature (Aluie, Hecht &
Vallis 2018; the Buzzicotti/Storer/Khatri/Griffies/Aluie line of work) runs this same 2D/2.5D method
**independently at each vertical level** of a multi-level model and compares/stacks the resulting
profiles. `coarse_grain_profile` implements this: given 3D `(x, y, z)` arrays it treats the vertical
axis as a batch over the shared horizontal grid, sweeping the 2D/2.5D `coarse_grain!` independently
per level and returning the stacked profile; the level axis parallelizes like any other batch axis.

`compute_Π!` also has a **coupled true 3D** method (`StructuredGrid{...,3}`, Cartesian or
spherical-volumetric): all nine strain/stress components, vertical derivatives, and one filter kernel
spanning all three directions. It targets homogeneous/isotropic turbulence (e.g. Rayleigh–Taylor or
boundary-layer studies), where the thin-layer assumption does not hold and levels interact through
the filter.

## Spherical Geometry

On `S²` of radius `R`, convolution uses the great-circle distance

```
d(x, x') = R · arccos(sin φ sin φ' + cos φ cos φ' cos(λ − λ')) ,
```

with area element `dA = R² cos φ dλ dφ`. The true 3D spherical-volumetric `StructuredGrid` (a
`(lon, lat, radius)` axis triple whose third axis is the radius itself) uses the corresponding volume element at each level's *local* radius `r[k]`,

```
dV = r[k]² cos φ dλ dφ dr ,
```

with horizontal arc-length spacing `r[k] cos φ · Δλ` / `r[k] · Δφ` and radial spacing `Δr` in the
derivative stencils, and the 3×3 ambient Cartesian tensor rotation of `compute_Π!` keeping the
radial direction.

On a spheroid the distance is the geodesic's (Karney 2013), and a vector is filtered by its ambient Cartesian
components. The top-hat's support on the surface is the geodesic disk, whose edge FlowGeometries' direct
geodesic problem gives at a set of azimuths, its half-width in longitude held as a Chebyshev series in latitude.
A circle of geodesic radius `R` bounds the disk while `R < πb`, the distance to the nearest cut locus; no two
points lie farther apart than half a meridian, and a top-hat that wide covers the whole surface. A top-hat of
radius between the two raises. A `(λ, φ, h)` grid of geodetic heights filters over the ball of chord radius
`R`, each cell's fraction integrated over its heights.

### Filtering vectors and tensors on the sphere

Filtering each local (east, north) component of a vector as a scalar does not commute with the surface
derivatives on `S²`, and filtering its ambient Cartesian components does not either (Aluie 2019, §3 and
eq. 41): the Cartesian components of the gradient `∇*Y_n` of a degree-`n` harmonic are harmonics of
degrees `n − 1` and `n + 1`, which a radial kernel scales by `Ĝ(n − 1)` and `Ĝ(n + 1)`, and the filtered
field gains a radial component. Aluie (2019) builds the convolution that commutes.

**The generalized convolution.** The Edmonds basis `K^{(i)}_{n,j}` (eqs. 42–45, 50–51) splits a vector
field into `u = u^{(1)} + u^{(2)} + u^{(3)}` (eqs. 56–57), and

```
G →* u = G^{(1)} * u^{(1)} + G^{(2)} * u^{(2)} + G^{(3)} * u^{(3)}                        (eq. 58)
```

with `G^{(1)}` and `G^{(2)}` the kernels whose Legendre transforms are `Ĝ(n − 1)` and `Ĝ(n + 1)`, and
`G^{(3)} = G` (eqs. 59–61). Every vector spherical harmonic `Y_{n,j}`, `Ψ_{n,j}`, `Φ_{n,j}` is then an
eigenfunction with eigenvalue `Ĝ(n)` (Lemma 1), and filtering commutes with the surface gradient,
divergence and curl (Proposition 1, eqs. 148–162). A tangent field `u = ∇*Φ + e_r × ∇*Ψ` (eq. 192) filters
to `∇*Φ̄ + e_r × ∇*Ψ̄`, `Φ̄` and `Ψ̄` the scalar convolutions (Proposition 2): the vector is filtered as the
scalars of its Helmholtz decomposition. Rank-2 tensors follow the same construction (eqs. 121–126,
Lemma 2), and a symmetric tangent tensor is filtered as the scalars of its decomposition (eq. 193,
Proposition 3, eq. 194). A radial component is filtered as a scalar (eq. 150), and the tangent vector a
tensor's radial row forms is filtered as a vector (eqs. 157–158).

**Spin-weighted form.** With `ₛY_ℓm(θ, φ) = N_ℓ d^ℓ_{m,−s}(θ) e^{imφ}` the spin-weighted harmonics
(`N_ℓ = √((2ℓ+1)/4π)`, `d^ℓ` the Wigner small-d, `θ` the colatitude) and `λ_ℓ = √(ℓ(ℓ+1))`, the identities
`∂_β d^ℓ_{m,0} = (λ_ℓ/2)(d^ℓ_{m,−1} − d^ℓ_{m,1})` and `(m/sin β) d^ℓ_{m,0} = −(λ_ℓ/2)(d^ℓ_{m,−1} + d^ℓ_{m,1})`
give

```
(∂_θ + i csc θ ∂_φ) Y_ℓm = λ_ℓ ₁Y_ℓm .
```

For `u = ∇*f + e_r × ∇*g` on the unit sphere, then, `u_θ + i u_φ = Σ λ_ℓ (f_ℓm + i g_ℓm) ₁Y_ℓm`: a spin-1
field whose degree-`ℓ` coefficients are `λ_ℓ(f_ℓm + i g_ℓm)`. `f` and `g` are real, so their coefficients
obey `f_ℓ,−m = (−1)^m conj(f_ℓm)`, and the coefficients at `±m` determine `f_ℓm` and `g_ℓm`. Scaling `f` and
`g` by `Ĝ(ℓ)`, Proposition 2, is therefore scaling each degree-`ℓ` spin-1 coefficient by `Ĝ(ℓ)`.

A symmetric tangent tensor `T` is its trace, a scalar, and its traceless part, which
`T(m, m) = T_ee − T_nn + 2i T_en`, `m = e_e + i e_n`, holds whole. The traceless parts of Proposition 3,
`∇*∇*f − ½(∇²f) I` and its rotation by 45°, have `T(m, m)` proportional to the spin-2 field `ð²(f + ig)`,
whose degree-`ℓ` coefficients are `√((ℓ−1)ℓ(ℓ+1)(ℓ+2)) (f_ℓm + i g_ℓm)`; scaling them by `Ĝ(ℓ)` is again
Proposition 3. A symmetric tensor has no antisymmetric part. In the local frame `U = u_e + i u_n` and
`Q = T_ee − T_nn + 2i T_en` differ from the `(θ, φ)` forms by constant phases, `u_θ + i u_φ = iU` and
`T(e_θ + i e_φ, e_θ + i e_φ) = −Q`, which a real `Ĝ(ℓ)` commutes with.

The generalized convolution is, then: analyze `U` at spin 1, `Q` at spin 2 and the trace at spin 0,
multiply every degree-`ℓ` coefficient by `Ĝ(ℓ)`, and synthesize. A vertical velocity `w` is a scalar and
`wU` a spin-1 field. A flux sweep analyzes its fields once and synthesizes them at every scale.

**Transforms.** [`Filtering.GeneralizedConvolution`](@ref) runs FastSphericalHarmonics' spin transforms on
a `Float64` `Fejer1Sampling` grid of `2N − 1` longitudes by `N` latitudes, exact for a field of
degree below `N`; NUFSHT's on every other spherical grid where it and a NUFFT library are loaded; and
otherwise the same sums
evaluated directly, the Wigner `d` from their three-term recurrence in `ℓ`: on a latitude–longitude
`StructuredGrid` by rows ([`Filtering.RingSpinTransforms`](@ref)), whose cells share a colatitude per
row, so the sums along the rows are one product with `e^{imφ_i}` and the recurrence runs once per row, in
`O(M·ℓ_max + n_lat·ℓ_max²)` for `M` cells; on any other grid cell by cell
([`Filtering.DirectSpinTransforms`](@ref)), in `O(M·ℓ_max²)`. Its seed `d^{|m|}_{m,n} ≈ sin^{|m|} θ` falls below
the floating-point range where `ℓ_max · sin θ · log₁₀(1/sin θ)` exceeds the decimal exponent range, about
`ℓ_max ≈ 240` in `Float32` and `1900` in `Float64`, for orders that still reach `O(ℓ^{−1/2})` past the
turning point `ℓ sin θ = |m|`; the recurrence carries such a value as a significand and its own binary
exponent. NUFSHT's and the direct transforms take a field's coefficients as `analysis` says
([`Filtering.AbstractAnalysis`](@ref)). `Quadrature()` is the sum `Σ_k (A_k/R²) F_k conj(ₛY_ℓm(x_k))` over the
grid's cells, exact for a field of degree at most `ℓ_max` where the measure integrates degree `2ℓ_max` exactly
(Gauss–Legendre rows), and over a regional grid the integral of the field extended by zero. `LeastSquares()`
minimizes `Σ_k A_k |F_k − Σ C_ℓm ₛY_ℓm(x_k)|²` by LSMR in the inner product of the cell measures, starting from the
quadrature, and recovers a field of degree at most `ℓ_max` wherever the cells determine its coefficients; a fit
that stops at its iteration limit keeps its last iterate and reports through
[`Filtering.analysis_status`](@ref). `AutoAnalysis()`, the default, keeps the quadrature on Gauss–Legendre and
reduced Gaussian samplings, where it is exact, and on grids that may be regional (no sampling,
latitude–longitude, scattered), and fits on every other spherical sampling, whose quadrature is not exact at its
band limit (Fejér, Driscoll–Healy, McEwen–Wiaux, HEALPix, cubed sphere, icosahedral, Yin–Yang); a
latitude–longitude grid over the whole sphere takes `LeastSquares()` by name. `ℓ_max` is the band limit of a
sampling that ties one to its ring count, and on any other grid the degree whose half wavelength is the side
`√A` of the largest cell, `⌈πR/√A⌉ − 1` ([`Filtering.spin_band_limit`](@ref)); `GeneralizedConvolution(; lmax)`
sets it, and the vector filter cuts sharply there. A `method = Spectral()` plan over a spherical node set takes
`analysis` the same way. The transforms run in the grid's element type; the libraries
run in host memory, through which a device workspace stages its fields, and the direct sum runs in the
workspace's own memory under its execution backend.

**Masks and regional grids.** Under `ZeroFill` the analysis reads zero at an inactive cell and nothing
past a regional grid's edges: the result is the generalized convolution of the field extended by zero over
the whole sphere. `Deformable` renormalizes each window over its active cells, which is no convolution,
and under it a vector is filtered by its ambient Cartesian components, [`Filtering.CartesianComponents`](@ref).
These equal the generalized convolution on a non-divergent tangent field, whose Cartesian components of
degree `n` are degree-`n` harmonics, and filter the degree-`n` content of a divergent part with
`Ĝ(n ± 1)`.

**Choosing.** Every diagnostic that filters a vector or a tensor takes `vector_filter`
([`Filtering.resolve_vector_filter`](@ref)) and `transform`. Under the generalized convolution a scalar
that combines with a filtered vector in the same diagnostic (`θ` in the tracer flux, `ρ` and `P` in the
Favre budget) is filtered at spin 0 by the same transforms and the same `Ĝ(ℓ)`, so the diagnostic reads
one operator: a uniform tracer carries no flux and a constant density gives `ρΠ`, both to round-off. A
scalar-only filter keeps the filter plan's engine.

| grid | `vector_filter = nothing` | `GeneralizedConvolution()` | `CartesianComponents()` |
|---|---|---|---|
| sphere, `ZeroFill` | generalized convolution | generalized convolution | Cartesian components |
| sphere, `Deformable` | Cartesian components | `ArgumentError` | Cartesian components |
| spheroid | Cartesian components | `ArgumentError` | Cartesian components |
| plane, volume, spherical shell | Cartesian components | Cartesian components | Cartesian components |

On a plane and in a volume, whose kernel is a ball of the ambient space, the convolution of each
Cartesian component is the generalized convolution. `transform` names the spin transforms' library:
`AutoSpectralBackend()` takes the fastest loaded one, and `DirectSumSpectralBackend()` the direct sum.

A great-circle kernel is homogeneous in distance, and `∂/∂x` is taken along `λ`, whose metric factor
`h_λ = R cos φ` varies across the footprint, so even for a scalar `filter ∘ ∂/∂x ≠ ∂/∂x ∘ filter` on the
sphere, by an amount that grows with `ℓ/R` and `|φ|` and is round-off while the footprint spans one
latitude row. `∂f/∂x` is a component of the vector `∇*f`, and Proposition 1 is the commuting statement,
`∇*(G * f) = G →* (∇*f)`. The filtered budget reads derivatives of filtered fields (`S̄` from `ū`, `∇θ̄`) and
the generalized convolution of the velocity and the stress, which commutes with the divergence.

## Curvilinear & unstructured grids: WLSQ gradients

`CurvilinearGrid` and `UnstructuredGrid` have no fixed axis spacing to difference against, so the
gradient (`FG.Operators.gradient_plan` + `FG.Operators.gradient!`, which returns both tangent
components from one neighbour sweep) is reconstructed from a local weighted-least-squares (WLSQ) fit over each point's neighbor
stencil (its 4 index-offset neighbors on a curvilinear mesh; its k-d tree neighbors on a scattered
point cloud), projected into the local tangent plane (`project_to_tangent_plane`, an exact 3D-chord
projection on spherical grids). WLSQ gradients are conditionally second order, degrading toward
first order on strongly skewed local stencils.

## References

- Aluie, H. (2011). Compressible turbulence: the cascade and its locality. *Phys. Rev. Lett.* 106(17).
- Aluie, H. (2019). Convolutions on the sphere: commutation with differential operators. *GEM* 10(1). doi:10.1007/s13137-019-0123-9
- Aluie, H., Hecht, M., & Vallis, G. K. (2018). Mapping the energy cascade in the North Atlantic Ocean: the coarse-graining approach. *J. Phys. Oceanogr.* 48(2), 225–244. doi:10.1175/JPO-D-17-0100.1
- Barkan, R., Srinivasan, K., & McWilliams, J. C. (2024). Eddy–internal wave interactions: stimulated cascades in cross-scale kinetic energy and enstrophy fluxes. *J. Phys. Oceanogr.* 54(6), 1309–1326. doi:10.1175/JPO-D-23-0191.1
- Germano, M. (1992). Turbulence: the filtering approach. *J. Fluid Mech.* 238. doi:10.1017/S0022112092001733
- Grooms, I., Loose, N., Abernathey, R., Steinberg, J. M., Bachman, S. D., Marques, G., Guillaumin, A. P., & Yankovsky, E. (2021). Diffusion-based smoothers for spatial filtering of gridded geophysical data. *J. Adv. Model. Earth Syst.* 13, e2021MS002552. doi:10.1029/2021MS002552
- Jekeli, C. (1981). *Alternative methods to smooth the Earth's gravity field.* Rep. 327, Dept. of Geodetic Science and Surveying, Ohio State University.
- Karney, C. F. F. (2013). Algorithms for geodesics. *J. Geodesy* 87(1), 43–55. doi:10.1007/s00190-012-0578-z
- Loose, N., Bachman, S., Grooms, I., & Jansen, M. (2023). Diagnosing scale-dependent energy cycles in a high-resolution isopycnal ocean model. *J. Phys. Oceanogr.* 53, 157.
- Pedlosky, J. (1987). *Geophysical Fluid Dynamics* (2nd ed.). Springer.
- Sadek, M., & Aluie, H. (2018). Extracting the spectrum of a flow by spatial filtering. *Phys. Rev. Fluids* 3, 124610. doi:10.1103/PhysRevFluids.3.124610
- Srinivasan, K., Barkan, R., & McWilliams, J. C. (2023). A forward energy flux at submesoscales driven by frontogenesis. *J. Phys. Oceanogr.* 53(1), 287–305. doi:10.1175/JPO-D-22-0001.1
- Storer, B. A. et al. (2022). Global energy spectrum of the general oceanic circulation. *Nat. Commun.* 13, 5314. doi:10.1038/s41467-022-33031-3
- Vallis, G. K. (2017). *Atmospheric and Oceanic Fluid Dynamics* (2nd ed.). Cambridge University Press.
