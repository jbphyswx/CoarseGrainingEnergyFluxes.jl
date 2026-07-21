```@meta
CurrentModule = CoarseGrainingEnergyFluxes
```

# Theory

## Coarse-Graining Framework

The coarse-grained (filtered) field at scale ℓ is a convolution with a normalized kernel `G_ℓ`:

```
ū_ℓ(x) = ∫ G_ℓ(x, x') u(x') dA(x')
```

Over an unmasked domain the kernel is renormalized by its running mass, so a constant field filters
to itself even next to a mask boundary (the `Deformable` mask strategy).

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

## Filter Kernels

The filter scale `ℓ` is the **full filter width** (Pope 2000 convention). Real-space weights are
unnormalized — the filtering routines divide by the running area/volume-weighted sum.

### Top-Hat (box) — `TopHatKernel`

Unit weight inside the disk/ball of radius `ℓ/2`, zero outside. The literature default
(Aluie et al. 2018, Storer et al. 2022). Not available for spectral filtering (its multidimensional
transfer function is an oscillatory Airy/sinc pattern that rings).

### Gaussian — `GaussianKernel(; α = 6)`

```
G_ℓ(r) ∝ exp(−α (r/ℓ)²)
```

- `α = 6` (default) is the Pope/turbulence convention: the Gaussian's second moment matches the
  top-hat box of width ℓ (`σ² = ℓ²/12`).
- `α = 4` reproduces FlowSieve's Gaussian, so `GaussianKernel(; α = 4)` is directly comparable to
  FlowSieve output.

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
| `TopHatKernel`        | unsupported (rings) |

On the sphere the wavenumber of harmonic degree `l` is `k_l = √(l(l+1)) / R`.

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

### Tracer / buoyancy variance flux — `tracer_variance_flux`

The scalar analogue of Π for a tracer θ (Aluie & Eyink):

```
Πθ = −∂_j θ̄ · τ_j(u, θ) ,   τ_j = ⟨u_j θ⟩ − ū_j θ̄ .
```

With θ = buoyancy this is the cross-scale buoyancy-variance (APE-related) transfer, needing only
`(u, v, θ)`.

## Vertical structure: depth-profile vs. true 3D

In 2.5D, `compute_Π!` deliberately drops the vertical-shear strain terms (`S_xz`, `S_yz`, `S_zz` are
either omitted or zero) whenever only `(u, v)` (optionally `w`) is supplied on a 2D grid. This is not
an oversight: it is the standard thin-layer/quasi-geostrophic scaling (Vallis, *Atmospheric and
Oceanic Fluid Dynamics*; Pedlosky, *Geophysical Fluid Dynamics*), valid when the aspect ratio
`δ = H/L` (vertical/horizontal scale) is small — the normal regime for large-scale ocean/atmosphere
flow, where vertical shear is genuinely subdominant to horizontal gradients. The actual "vertical
structure via coarse-graining" literature (Aluie, Hecht & Vallis 2018; the
Buzzicotti/Storer/Khatri/Griffies/Aluie line of work) does not compute a coupled vertical-derivative
tensor either — it runs this same 2D/2.5D method **independently at each depth level** of a
multi-level model and compares/stacks the resulting profiles. `coarse_grain_profile` /
`compute_Π_profile!` implement exactly this: given 3D `(lon, lat, depth)` arrays, they loop the
existing 2D/2.5D `compute_Π!`/`coarse_grain` independently over each level and return the stacked
profile — no new tensor math, a convenience wrapper over an already-correct 2D method.

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
afterward (`to_planetary_cartesian` / `from_planetary_cartesian`), which is exact for non-divergent
flow (Storer et al. 2022). For strongly divergent flow, decompose first (HelmholtzDecomposition.jl)
and use [`Diagnostics.compute_Π_decomposed`](@ref).

## Curvilinear & unstructured grids: WLSQ gradients

`CurvilinearGrid` and `UnstructuredGrid` have no fixed axis spacing to difference against, so `ddx!`/
`ddy!` are reconstructed from a local weighted-least-squares (WLSQ) fit over each point's neighbor
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
- Pedlosky, J. (1987). *Geophysical Fluid Dynamics* (2nd ed.). Springer.
- Sadek, M., & Aluie, H. (2018). Extracting the spectrum of a flow by spatial filtering. *Phys. Rev. Fluids* 3, 124610. doi:10.1103/PhysRevFluids.3.124610
- Storer, B. A. et al. (2022). Global energy spectrum of the general oceanic circulation. *Nat. Commun.* 13, 5314. doi:10.1038/s41467-022-33031-3
- Vallis, G. K. (2017). *Atmospheric and Oceanic Fluid Dynamics* (2nd ed.). Cambridge University Press.
