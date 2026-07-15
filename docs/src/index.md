# CoarseGrainingEnergyFluxes.jl

Spatial coarse-graining analysis of energy fluxes in geophysical fluid dynamics.

## Overview

This package implements the coarse-graining (spatial filtering) framework for computing:

- **Cross-scale energy flux** Π(x, ℓ) — the local rate of kinetic energy transfer across scale ℓ
- **Cumulative coarse-grained energy** ½⟨|ū_ℓ|²⟩ (`cumulative_energy`) and the **filtering spectral density** Ẽ(k_ℓ) (`filtering_spectrum`) — the spectrum extracted by filtering (Sadek & Aluie 2018)

The approach follows Aluie (2011, 2019) and Aluie, Hecht, & Vallis (2018), using real-space convolution kernels (top-hat, Gaussian) to separate large-scale (ū) and sub-scale (u') motions at each point in space.

Every diagnostic works across the full grid×dimensionality matrix — `StructuredGrid` (1D, 2D, and
true 3D Cartesian or spherical-volumetric), `CurvilinearGrid` (model-native meshes, e.g. ROMS), and
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
- Poorly suited to irregular domains and coastlines

Coarse-graining provides:
- **Local** energy flux Π(x, ℓ) at every grid point
- Works on arbitrary domains with land masks
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
