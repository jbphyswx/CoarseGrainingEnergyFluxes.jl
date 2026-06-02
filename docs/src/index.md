# CoarseGrainingEnergyFluxes.jl

Spatial coarse-graining analysis of energy fluxes in geophysical fluid dynamics.

## Overview

This package implements the coarse-graining (spatial filtering) framework for computing:

- **Cross-scale energy flux** Π(x, ℓ) — the local rate of kinetic energy transfer across scale ℓ
- **Filtering energy spectrum** E(ℓ) — domain-averaged kinetic energy at each scale

The approach follows Aluie (2011, 2019) and Aluie, Hecht, & Vallis (2018), using real-space convolution kernels (top-hat, Gaussian) to separate large-scale (ū) and sub-scale (u') motions at each point in space.

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

### The Filtering Energy Spectrum

```
E(ℓ) = ½ ⟨|ū_ℓ|²⟩
```

where ⟨·⟩ denotes domain average. The spectral slope of E(ℓ) vs ℓ reveals the energy distribution across scales.
