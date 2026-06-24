```@meta
CurrentModule = CoarseGrainingEnergyFluxes
```

# Theory

## Coarse-Graining Framework

The coarse-grained (filtered) field at scale ℓ is a convolution with a normalized kernel `G_ℓ`:

```
ū_ℓ(x) = ∫ G_ℓ(x, x') u(x') dA(x')
```

Over a wet (unmasked) domain the kernel is renormalized by its running mass, so a constant field
filters to itself even next to coastlines (the `Deformable` mask strategy).

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
`+ S_zz τ_zz + 2 S_xz τ_xz + 2 S_yz τ_yz` (see [`compute_Π!`](@ref)).

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
[`spectral_transfer`](@ref)`(kernel, k, ℓ)`, normalized so `Ĝ(0) = 1` (the mean is preserved):

| Kernel | `Ĝ(k, ℓ)` |
|--------|-----------|
| `GaussianKernel(α)`   | `exp(−k² ℓ² / 4α)` |
| `SharpSpectralKernel` | `1` if `k ≤ π/ℓ`, else `0` |
| `TopHatKernel`        | unsupported (rings) |

On the sphere the wavenumber of harmonic degree `l` is `k_l = √(l(l+1)) / R`.

## The Filtering Spectrum (Sadek & Aluie 2018)

Filtering at a continuum of scales yields a spectrum without windowing or periodicity assumptions.
The **cumulative** coarse-grained kinetic energy ([`cumulative_energy`](@ref), their Eq. 15) is

```
E(ℓ) = ½ ρ₀ ⟨|ū_ℓ|²⟩ ,
```

a *cumulative* quantity. The **filtering spectral density** ([`filtering_spectrum`](@ref), their
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
subfilter–subfilter (Reynolds, the backscatter-carrying term).

### Rotational / divergent (Helmholtz) — `compute_Π_decomposed`

Given the rotational (solenoidal) velocity `uʳ` (from a Helmholtz solver such as
[HelmholtzDecomposition.jl](https://github.com/jbphyswx/HelmholtzDecomposition.jl); the divergent part
is the complement), the stress splits as `τ = τʳʳ + τ_cross + τᵈᵈ`, giving

```
Π = Π_rotational + Π_cross + Π_divergent
```

(each channel contracted with the full strain S̄, summing to Π to machine precision).

### Tracer / buoyancy variance flux — `tracer_variance_flux`

The scalar analogue of Π for a tracer θ (Aluie & Eyink):

```
Πθ = −∂_j θ̄ · τ_j(u, θ) ,   τ_j = ⟨u_j θ⟩ − ū_j θ̄ .
```

With θ = buoyancy this is the cross-scale buoyancy-variance (APE-related) transfer, needing only
`(u, v, θ)`.

## Spherical Geometry

On `S²` of radius `R`, convolution uses the great-circle distance

```
d(x, x') = R · arccos(sin φ sin φ' + cos φ cos φ' cos(λ − λ')) ,
```

with area element `dA = R² cos φ dλ dφ`.

### Commutativity on the sphere

Aluie (2019) shows that filtering vector components as scalars does **not** commute with `∇` on `S²`.
This package transforms velocities to planetary-Cartesian components before filtering and back
afterward (`to_planetary_cartesian` / `from_planetary_cartesian`), which is exact for non-divergent
flow (Storer et al. 2022). For strongly divergent flow, decompose first (HelmholtzDecomposition.jl)
and use [`compute_Π_decomposed`](@ref).

## References

- Aluie, H. (2011). Compressible turbulence: the cascade and its locality. *Phys. Rev. Lett.* 106(17).
- Aluie, H. (2019). Convolutions on the sphere: commutation with differential operators. *GEM* 10(1). doi:10.1007/s13137-019-0123-9
- Aluie, H., Hecht, M., & Vallis, G. K. (2018). Mapping the energy cascade in the North Atlantic Ocean. *J. Phys. Oceanogr.* 48(8). doi:10.1175/JPO-D-17-0100.1
- Germano, M. (1992). Turbulence: the filtering approach. *J. Fluid Mech.* 238. doi:10.1017/S0022112092001733
- Sadek, M., & Aluie, H. (2018). Extracting the spectrum of a flow by spatial filtering. *Phys. Rev. Fluids* 3, 124610. doi:10.1103/PhysRevFluids.3.124610
- Storer, B. A. et al. (2022). Global energy spectrum of the general oceanic circulation. *Nat. Commun.* 13, 5314. doi:10.1038/s41467-022-33031-3
