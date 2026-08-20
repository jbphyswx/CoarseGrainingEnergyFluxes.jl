```@meta
CurrentModule = CoarseGrainingEnergyFluxes
```

# CoarseGrainingEnergyFluxes.jl

Spatial coarse-graining analysis of energy fluxes in geophysical fluid dynamics.

## Overview

This package implements the coarse-graining (spatial filtering) framework. The core quantity is the
**cross-scale energy flux** Π(x, ℓ), the local rate of kinetic-energy transfer across scale ℓ. Around it:

| | Function |
|---|---|
| Cumulative coarse energy and the filtering spectral density Ẽ(k_ℓ) | [`Diagnostics.cumulative_energy`](@ref), [`Diagnostics.filtering_spectrum`](@ref) |
| Strain/convergence split of Π | [`Diagnostics.compute_Π_strain_convergence`](@ref) |
| Rotational/divergent (Helmholtz) split, with the interaction channel | [`Diagnostics.compute_Π_decomposed`](@ref) |
| Leonard/Cross/Reynolds stress decomposition | [`Diagnostics.tau_decomposition`](@ref) |
| Tracer / buoyancy-variance flux | [`Diagnostics.tracer_variance_flux`](@ref) |
| Enstrophy flux, in the same gauge as Π | [`Diagnostics.enstrophy_flux`](@ref) |
| Energy per scale band | [`Diagnostics.band_energies`](@ref) |
| Variable-density (Favre) budget: Π, baropycnal work Λ, pressure dilatation | [`Diagnostics.compressible_flux`](@ref) |

Six filter kernels are available — top-hat, Gaussian, sharp-spectral, smooth-hat, hyper-Gaussian and
the high-order `M^I`/`M^II` pair — and they are **not** interchangeable: only some have a spectral
transfer function, only some can carry a filtering spectrum, and only some are valid for Π.
[`check_setup`](@ref) reports which, for your grid and scale, without running anything.

The approach follows Aluie (2011, 2019) and Aluie, Hecht, & Vallis (2018), using real-space convolution kernels to separate large-scale (ū) and sub-scale (u') motions at each point in space.

Every diagnostic works across the full grid×dimensionality matrix — `StructuredGrid` (1D, 2D, and
true 3D Cartesian or spherical-volumetric), `CurvilinearGrid` (model-native orthogonal curvilinear meshes), and
`UnstructuredGrid` (scattered points, via k-d tree neighbors, Voronoi cell areas, and non-uniform
spectral transforms) — see [Architecture](architecture.md) for the full capability matrix.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/jbphyswx/CoarseGrainingEnergyFluxes.jl")
```

## Key Concepts

### Coarse-Graining vs Fourier Spectra

Traditional Fourier spectral analysis provides wavenumber spectra E(k) but:
- Requires periodicity or windowing
- Cannot localize energy transfer in physical space
- Poorly suited to irregular domains and boundaries

Coarse-graining provides:
- **Local** energy flux Π(x, ℓ) at every grid point
- Works on arbitrary domains with masked (excluded) regions
- No periodicity assumption
- Direct physical-space interpretation

### The Energy Flux Π

The cross-scale energy flux at position x and scale ℓ is:

```
Π(x, ℓ) = −τ_ℓ : S̄_ℓ
```

where:
- S̄_ℓ = ½(∇ū_ℓ + (∇ū_ℓ)ᵀ) is the filtered strain rate
- τ_ℓ = (u⊗u)̄_ℓ − ū_ℓ⊗ū_ℓ is the sub-scale stress

When Π > 0, energy flows from large to small scales (forward cascade).
When Π < 0, energy flows from small to large scales (inverse cascade).

### The Filtering Spectrum

The **cumulative** coarse-grained kinetic energy (`cumulative_energy`; Sadek & Aluie 2018, Eq. 15)
is the domain average of the filtered KE:

```
E(ℓ) = ½ ⟨|ū_ℓ|²⟩
```

This is a *cumulative* quantity, **not** a spectral density. The **filtering spectral density**
(`filtering_spectrum`; their Eq. 14 — comparable to a Fourier energy spectrum) is its derivative
with respect to the filtering wavenumber `k_ℓ = L/ℓ`:

```
Ẽ(k_ℓ) = d/dk_ℓ [ ½ ⟨|ū_ℓ|²⟩ ]
```

`coarse_grain` returns both (`result.cumulative_energy`, `result.filtering_spectrum`,
`result.wavenumber`).

`Ẽ(k_ℓ) ≥ 0` is guaranteed only for a kernel whose `|Ĝ(k)|²` is monotone decreasing, which the
default `TopHatKernel` is not. `coarse_grain` therefore throws rather than return a density it cannot
vouch for: pass `kernel = GaussianKernel()` for a spectrum, or `spectrum = false` for `Π` and the
cumulative energy alone. Note also that a `p = 1` kernel — top-hat and Gaussian both — saturates the
recovered slope at `k⁻³`; see [`Diagnostics.filtering_spectrum`](@ref).
