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
the high-order `M^I`/`M^II` pair — and they differ in what they support: only some have a spectral
transfer function, only some can carry a filtering spectrum, and only some are valid for Π.
[`check_setup`](@ref) reports which, for your grid and scale, without running anything.

The approach follows Aluie (2011, 2019) and Aluie, Hecht, & Vallis (2018), using real-space convolution kernels to separate large-scale (ū) and sub-scale (u') motions at each point in space.

`RealSpace()` names that operator: a local space average against the compact kernel. Which engine
evaluates it is chosen from the grid and kernel — exact prefix sums for a top-hat in 2-D and 3-D, separable passes for the factorizing
kernels, a vectorized banded footprint otherwise — and
[`check_setup`](@ref) reports the choice and its cost before anything runs. Where it takes fewer
operations than that engine, the same convolution is evaluated by transform: a padded FFT of the
sampled kernel on a uniform Cartesian lattice in any dimension, and a transform along longitude on a
sphere of uniform longitude, where a great-circle kernel depends on the longitude offset alone. See
[Architecture](architecture.md#Real-space-engines) for the full table.

For a sweep over scales, [`Filtering.plan_filter_sweep`](@ref) builds the parts of a plan the scale
does not change — transform objects, measure prefix scans, point sorts — once for the whole sweep, and
every diagnostic has an in-place form whose workspace holds only the buffers its configuration
reaches.

The top-hat weighs each grid cell by its area inside the disk, cap or geodesic disk about the target
(`Kernels.CellOverlap`), the continuous filter integrated over the cell. The strain differentiates the filtered
velocity by the filter's own series where the filter is spectral, and otherwise by 5-node stencils or the
least-squares gradient ([`Derivatives.AbstractDerivative`](@ref)).

Every diagnostic works across the full grid×dimensionality matrix — `StructuredGrid` (1D, 2D, and
true 3D Cartesian or spherical-volumetric), `CurvilinearGrid` (model-native orthogonal curvilinear meshes),
`UnstructuredGrid` (scattered points, via k-d tree neighbors, Voronoi cell areas, and non-uniform
spectral transforms) and the sphere pixelizations, on a plane, a sphere or a spheroid — see
[Architecture](architecture.md) for the full capability matrix.

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

This is a *cumulative* quantity. The **filtering spectral density**
(`filtering_spectrum`; their Eq. 14 — comparable to a Fourier energy spectrum) is its derivative
with respect to the filtering wavenumber `k_ℓ = L/ℓ`:

```
Ẽ(k_ℓ) = d/dk_ℓ [ ½ ⟨|ū_ℓ|²⟩ ]
```

`coarse_grain` returns both (`result.cumulative_energy`, `result.filtering_spectrum`,
`result.wavenumber`). `dE/dℓ = ⟨ū·∂_ℓū⟩` is taken through the filter itself where its kernel has a
derivative in `ℓ` ([`Diagnostics.AbstractScaleDerivative`](@ref)); `slopes = Diagnostics.Slopes()` adds the
spectrum's slope per scale and `spectrum_maps = Diagnostics.SpectrumMaps()` the local spectrum per cell.

`Ẽ(k_ℓ) ≥ 0` is guaranteed for a kernel whose `|Ĝ(k)|²` is monotone decreasing; the default
`TopHatKernel`'s is not, so by default (`spectrum = AutoSpectrum()`) `coarse_grain` fills its
`filtering_spectrum` with `NaN` for it, and `Diagnostics.filtering_spectrum` throws. The condition is
sufficient but not necessary, and three choices are available: `kernel = GaussianKernel()` for a
guaranteed-non-negative spectrum, `spectrum = CGEF.Diagnostics.ForceSpectrum()` to compute the
top-hat's anyway and check its sign yourself, or `spectrum = CGEF.Diagnostics.NoSpectrum()` for `Π`
and the cumulative energy alone. A `p = 1` kernel — top-hat and Gaussian both — saturates the
recovered slope at `k⁻³`; see [`Diagnostics.AbstractSpectrumPolicy`](@ref) and
[`Diagnostics.filtering_spectrum`](@ref).
