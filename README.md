# CoarseGrainingEnergyFluxes.jl

[![Build Status](https://github.com/jbphyswx/CoarseGrainingEnergyFluxes.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/jbphyswx/CoarseGrainingEnergyFluxes.jl/actions/workflows/CI.yml)
[![Dev Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://jbphyswx.github.io/CoarseGrainingEnergyFluxes.jl/dev/)
[![Coverage](https://codecov.io/gh/jbphyswx/CoarseGrainingEnergyFluxes.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/jbphyswx/CoarseGrainingEnergyFluxes.jl)

Spatial coarse-graining (Aluie/FlowSieve-style) analysis of energy fluxes in geophysical fluid
dynamics: cross-scale kinetic-energy transfer Π(x, ℓ), the filtering spectrum, and related
diagnostics from velocity fields on Cartesian or spherical grids — structured, curvilinear
(model-native), and scattered/unstructured, in 1D, 2D, and true 3D.

![Coarse-graining pipeline](docs/src/assets/hero.png)

## What This Package Does

Coarse-graining (spatial filtering) decomposes a turbulent flow into scale-dependent contributions
and measures the energy transferred between them. Given the filtered velocity ū_ℓ and the sub-scale
stress τ_ℓ = (u⊗u)̄_ℓ − ū_ℓ⊗ū_ℓ, the cross-scale kinetic-energy flux is

```
Π(x, ℓ) = −τ_ℓ : S̄_ℓ
```

per unit mass, in m² s⁻³; `ρ₀ Π` is the flux per unit volume (Π > 0 forward cascade, Π < 0 inverse
cascade). Alongside Π the package computes:

| Diagnostic | Function |
|---|---|
| Filtering spectrum and cumulative coarse energy (Sadek & Aluie 2018) | `filtering_spectrum`, `cumulative_energy` |
| Strain/convergence split of Π (Srinivasan, Barkan & McWilliams 2023) | `compute_Π_strain_convergence` |
| Rotational/divergent (Helmholtz) three-way split, including the interaction "stimulated cascade" channel | `compute_Π_decomposed` |
| Leonard/Cross/Reynolds stress decomposition (Cartesian and spherical) | `tau_decomposition` |
| Tracer / buoyancy-variance flux | `tracer_variance_flux` |
| Enstrophy flux, in the same deformation gauge as Π | `enstrophy_flux` |
| Energy per scale band, via the repeated-filter Germano identity | `band_energies` |
| Variable-density (Favre) budget: Π, baropycnal work Λ, and pressure dilatation | `compressible_flux` |

— on masked, regional, or global domains, with real-space (direct-sum) or spectral (FFTW, nonuniform
FFT through NonuniformFFTs or FINUFFT, spherical-harmonic, NUFSHT) backends and
serial/threaded/GPU/distributed/MPI execution.

Each diagnostic has an in-place form taking a workspace, so a sweep over scales or timesteps allocates
nothing after the first call, and each workspace holds only the buffers its configuration can reach —
a 2-D flux without a vertical component does not carry the vertical-component buffers, and the
energy/spectrum path takes a two-buffer `EnergyWorkspace` rather than a full flux workspace.

The grid×dimensionality matrix spans 1D transects, 2D (Cartesian or spherical, single-level or the
standard literature "vertical structure" profile method), true 3D (Cartesian and spherical-volumetric,
genuinely coupled vertical derivatives), model-native curvilinear grids (orthogonal curvilinear meshes,
via weighted-least-squares gradients), scattered/unstructured point clouds (via k-d tree neighbor
search, Voronoi cell areas, and non-uniform spectral transforms), and the sphere pixelizations.

Every diagnostic takes any grid in that matrix, on either metric. What varies is the rank each one is
defined for:

| diagnostic | rank |
|---|---|
| `compute_Π!`, `coarse_grain`, `cumulative_energy`, `filtering_spectrum`, `band_energies` | 1D, 2D, true 3D |
| `tau_decomposition`, `tracer_variance_flux`, `compressible_flux`, `compute_Π_decomposed` | 2D tangent, and true 3D |
| `vorticity`, `enstrophy_flux`, `compute_Π_strain_convergence` | 2D tangent — the definition of each, the vertical vorticity and the horizontal strain split |

"Any grid" means any layout resolving two tangent directions: `StructuredGrid`, `CurvilinearGrid`,
`UnstructuredGrid` and every sphere pixelization. The gradient each one needs is taken through the
architecture's own operator — a stencil table where there are axes to difference along, a
least-squares tangent-plane fit where there are not — so nothing in these calls names a grid type.

On a sphere every quantity carrying a direction is built in planetary-Cartesian coordinates and
rotated back to the local frame, since a local (east, north) pair filtered component-wise is not a
filtered vector (Aluie 2019); and every spatial operator carries the local frame's `tanφ/R` curvature
terms. The suite gates that by asserting each decomposition's total against `compute_Π!`, which is an
identity on either metric.

## Results

### Spatial filtering across scales
Filtering coarsens a field as ℓ grows — shown for a deterministic fractal pattern and an eddy+noise flow.

![Filtering Scales](docs/src/assets/filtering_scales.png)

### Filter kernels and their spectral transfer
Top-hat vs Gaussian (α = 6 Pope / α = 4 FlowSieve) real-space shapes, and the sharp-spectral vs Gaussian transfer functions.

![Kernels](docs/src/assets/kernels.png)

### The filtering spectrum (recovers the Fourier slope)
Cumulative coarse KE E(ℓ) and the spectral density Ẽ(k_ℓ); the sharp-spectral kernel recovers the k⁻³ slope, while a Gaussian smooths it.

![Filtering spectrum](docs/src/assets/filtering_spectrum.png)

### Rotational / divergent (Helmholtz) decomposition of Π
Π splits exactly into rotational→rotational, divergent→divergent, and interaction ("stimulated cascade") channels.

![Helmholtz decomposition](docs/src/assets/helmholtz_decomposition.png)

### Strain / convergence split of Π
![Strain / convergence split of Π](docs/src/assets/strain_convergence.png)

### Enstrophy flux (the 2-D companion to Π)
![Enstrophy flux (the 2-D companion to Π)](docs/src/assets/enstrophy_flux.png)

### Energy by scale band
![Energy by scale band](docs/src/assets/band_energies.png)

### Variable-density (Favre) budget: Π, baropycnal work Λ
![Variable-density (Favre) budget: Π, baropycnal work Λ](docs/src/assets/compressible_flux.png)

### Cross-scale tracer / buoyancy-variance flux
The scalar analogue of Π (buoyancy ⇒ available-potential-energy transfer).

![Tracer flux](docs/src/assets/tracer_flux.png)

### Masking: zero-fill vs deformable
`ZeroFill` is the default: land and the domain exterior count as fluid at rest (Aluie et al. 2018; Grooms et al. 2021), and the kernel keeps its full mass at every cell, so filtering commutes with spatial derivatives — the step the flux budget is derived by. Every output is defined on every cell: `Π` over land is nonzero within the kernel's reach of the coast and zero beyond it, and domain means sum every cell over the water area. `Deformable` renormalizes over the locally-active area, reproducing constants exactly next to a boundary at the cost of that commutation, and zeroes masked cells. The two differ within the kernel's reach of a coast or bounded edge.

![Masking](docs/src/assets/masking.png)

### Spectral filtering on the sphere
Global spherical-harmonic filtering (FFTW, the nonuniform FFT, FastSphericalHarmonics and NUFSHT cover Cartesian/spherical × uniform/scattered).

![Spherical filtering](docs/src/assets/spherical_filtering.png)

### Validation: rigid-body rotation → Π = 0
Pure rotation has no deformation, so the flux must vanish (to machine precision).

![Rigid Rotation Validation](docs/src/assets/rigid_rotation_validation.png)

### Curvilinear (model-native) grids
A sheared/rotated curvilinear mesh filtered via weighted-least-squares gradients — no rectilinear
assumption anywhere in the pipeline.

![Curvilinear grid](docs/src/assets/curvilinear.png)

### Scattered / unstructured point clouds
k-d tree neighbor search + exact Voronoi cell areas + nonuniform-FFT spectral filtering, taking
`compute_Π!` all the way to a real flux map on genuinely scattered observations.

![Unstructured grid](docs/src/assets/unstructured.png)

### True 3D volumetric flux (Cartesian and spherical shells)
Genuinely coupled 3D strain/stress (all nine components) — homogeneous/isotropic-turbulence-style
filtering that blends all three directions in one kernel, distinct from the 2.5D vertical-profile method.

![True 3D volumetric flux](docs/src/assets/volumetric_3d.png)

### Vertical profile (2.5D per-level) vertical structure
The literature-standard "vertical structure" method (Aluie, Hecht & Vallis 2018): the existing 2D/2.5D
`compute_Π!` run independently at each vertical level and stacked into a profile.

![Vertical profile](docs/src/assets/profile.png)

## Quick Start

```julia
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF
using FlowGeometries: FlowGeometries as FG   # geometries and grid types live here

# Create grid
geom = FG.Geometry.SphericalGeometry(6.371e6)  # Earth radius in meters
grid = FG.Grids.StructuredGrid(geom, lon_rad, lat_rad, mask)

# Run multi-scale analysis
scales = collect(10e3:10e3:300e3)  # 10 km to 300 km
result = CGEF.coarse_grain(u, v, grid; scales = scales, kernel = CGEF.TopHatKernel(),
                           spectrum = CGEF.Diagnostics.NoSpectrum())

# result.Π                 — (Nlon, Nlat, Nscales) stacked flux array; result.Π[:, :, i] at scales[i]
# result.cumulative_energy — ½⟨|ū_ℓ|²⟩ per scale, per unit mass (Sadek–Aluie Eq. 15)
# result.wavenumber        — k_ℓ = L/ℓ

# The top-hat's |Ĝ|² is not monotone, so it cannot carry a filtering spectral density (Sadek & Aluie
# 2018 eq. 21) and `coarse_grain` refuses to produce one. Ask a kernel that can:
spec = CGEF.coarse_grain(u, v, grid; scales = scales, kernel = CGEF.GaussianKernel())
# spec.filtering_spectrum  — Ẽ(k_ℓ) spectral density (Eq. 14)
```

Not sure what a given `(grid, kernel, ℓ)` will actually do? `CGEF.check_setup(grid, kernel, ℓ)` reports
the engine, the resolved backend, whether the kernel can carry `Π` and a spectrum, whether `ℓ` is
resolvable on that grid, and how far from a coast or domain edge the result depends on the mask
strategy — without building a plan.

Only a minimal set of names is exported at the top level: the sweep entry points (`coarse_grain`,
`coarse_grain!`, `coarse_grain_profile`, `coarse_grain_batch!`, `coarse_grain_slices!`), their result
types, `check_setup`, the three headline kernels, and `plot_Π_map`/`plot_spectrum`. Grids and
geometries come from FlowGeometries.jl, and everything else — `filter_field!`, `compute_Π!`,
`compute_Π_strain_convergence`, `compute_Π_decomposed`, `tau_decomposition`, `tracer_variance_flux`,
`enstrophy_flux`, `band_energies`, the remaining kernels, backends, mask strategies,
`ddx!`/`ddy!`/`ddz!`, `plan_filter`, `plan_filter_sweep`, `spectral_transfer`, the workspaces
(`ΠWorkspace`, `EnergyWorkspace`, `TauWorkspace`, `Sym3TauWorkspace`, `FavreWorkspace`,
`EnstrophyFluxWorkspace`, …), … — is reached through the
qualified submodule path shown in [Architecture](#architecture) below, e.g.
`CGEF.Diagnostics.compute_Π!(...)`, `CGEF.Filtering.filter_field!(...)`.

## Architecture

Geometries, grid types and the execution/spectral backend taxonomies are external packages
(`FlowGeometries.jl`, `ComputationalBackends.jl`, `SpectralBackends.jl`); this package is the
coarse-graining engine on top of them.

```
src/
  Kernels.jl      — TopHatKernel, GaussianKernel, SmoothHatKernel, HyperGaussianKernel,
                    HighOrderKernel{P}, SharpSpectralKernel, and the spectrum policies
  Filtering.jl    — filter_field! (real-space footprint engine + spectral plan dispatch)
  Filtering/      — the module's files: Strategies, Lifetimes, CacheStrategy, Hooks, Api, Apply,
                    Plans, Selection, Slices, and engines/ (Exterior, Footprint, PrefixSumTopHat,
                    Separable, NDim, PrefixSumTopHat3D, NodeSpectral, NUFFTSpectral — the nonuniform
                    FFT through FlowTransformBindings)
  Derivatives.jl  — ddx!/ddy!/ddz! + StencilPlan, over FlowGeometries' discretization
                    (least-squares gradients on CurvilinearGrid/UnstructuredGrid come from
                    Operators.gradient_plan there)
  Diagnostics.jl  — compute_Π!, compute_Π_decomposed, compute_Π_strain_convergence,
                    tau_decomposition, tracer_variance_flux, enstrophy_flux, band_energies,
                    compressible_flux, cumulative_energy, filtering_spectrum
  Diagnostics/    — the module's files: SpectrumPolicy, Flux, TensorDriver, Spectrum, Stress,
                    Helmholtz, StrainConvergence, Tracer, Favre, Bands, Enstrophy
  Pipeline.jl     — coarse_grain / coarse_grain! / coarse_grain_profile / coarse_grain_batch!,
                    check_setup (high-level orchestration)
  Visualization.jl — plot_Π_map / plot_spectrum stubs (methods provided by the CairoMakie ext)
ext/
  FFTWExt                       — FFT spectral filtering (uniform Cartesian StructuredGrid; bounded axes zero-padded)
  FastSphericalHarmonicsExt     — spherical-harmonic transform (StructuredGrid on ClenshawCurtisSampling, nlon = 2·nlat − 1)
  NUFSHTExt                     — non-uniform spherical-harmonic transform (scattered spherical UnstructuredGrid)
  OhMyThreadsExt                — ThreadedBackend (2D row-parallel; also 1D/true-3D point-parallel)
  GPUExt                        — GPUBackend via KernelAbstractions
  DistributedExt                — DistributedBackend (Distributed + SharedArrays)
  MPIExt                        — MPIBackend (multi-node domain decomposition)
  SpecialFunctionsExt           — exact top-hat spectral transfer 2·J₁(kR)/(kR)
  CairoMakieExt                 — plot_Π_map / plot_spectrum implementations
```

Backend implementations and the transform libraries are **package extensions** (weak dependencies),
so the core package has no heavy dependencies. The nonuniform FFT runs through FlowTransformBindings,
whose own extensions bind NonuniformFFTs and FINUFFT: `using NonuniformFFTs` or `using FINUFFT` enables
it, and `spectral_backend = FlowTransformBindings.FINUFFTBackend()` (or `NonuniformFFTsBackend()`)
names one.

## Grid Types

| Grid | Dimensionality | Real-space filter | Spectral filter | Derivatives | `compute_Π!` |
|------|-----------------|--------------------|-----------------|--------------|--------------|
| `StructuredGrid` | 1D, 2D, true 3D (Cartesian or spherical-volumetric) | Yes | Yes (FFTW on two uniform Cartesian axes; FastSphericalHarmonics on `structured_grid(ClenshawCurtisSampling(), N)`; any other over its cells by the nonuniform transform) | `ddx!`/`ddy!`/`ddz!` (+ a reusable `Derivatives.StencilPlan`) | Yes, all dimensionalities + a 2.5D vertical-profile wrapper (`compute_Π_profile!`) |
| `CurvilinearGrid` | 2D (model-native, orthogonal curvilinear meshes) | Yes (per-point footprint, no translation invariance assumed) | Yes, over its cells by the nonuniform FFT | `FG.Operators.gradient_plan` + `FG.Operators.gradient!` (least-squares tangent plane) | Yes |
| `UnstructuredGrid` | 1D (scattered points) | Yes, and the default (`RealSpace()` — ball query over the grid's own adjacency) | Yes (nonuniform FFT for 1–3 Cartesian coordinates; NUFSHT spherical) | the same, over the grid's k-d tree adjacency | Yes |
| `RingGrid`, `CubedSphereGrid`, `HEALPixGrid`, `IcosahedralGrid`, `YinYangGrid` | 1D (sphere pixelizations: one index names a cell) | Yes, and the default — the same node CSR gather over each grid's own ball query | Yes, over its cells by NUFSHT | `Derivatives.gradient_plan` + `FG.Operators.gradient!`, through each grid's own adjacency | Yes |

`CurvilinearGrid` and `UnstructuredGrid` are built genuinely from scratch, not thin wrappers: exact
quadrilateral corner-based cell areas (curvilinear) or k-d tree adjacency + real Voronoi tessellation
cell areas (unstructured, `NearestNeighbors`/`DelaunayTriangulation`/`Quickhull`), and the same
`compute_Π!`/`coarse_grain` pipeline as `StructuredGrid`, sharing the per-point tensor-rotation kernel.

```julia
using NearestNeighbors: NearestNeighbors     # enables UnstructuredGrid's k-d tree neighbor search
using DelaunayTriangulation: DelaunayTriangulation  # enables exact Voronoi areas (Cartesian)
# using Quickhull: Quickhull                 # enables exact Voronoi areas (spherical)

ug = FlowGeometries.Grids.UnstructuredGrid(geom, x, y, mask; k = 8)  # k-nearest neighbors, auto Voronoi areas
```

## Filter Kernels

| Kernel | Description | Use case |
|--------|-------------|----------|
| `TopHatKernel()` | Uniform weight within radius ℓ/2 | Standard, most common (spectral transfer needs `using SpecialFunctions`) |
| `GaussianKernel(; α = 6)` | Gaussian, variance-matched to a box of width ℓ (`σ² = ℓ²/12`) | Smooth, differentiable, has an exact spectral transfer |
| `SharpSpectralKernel()` | Ideal low-pass, `Ĝ = 1` for `k ≤ π/ℓ`; in real space its inverse transform in the grid's dimension (sinc, jinc, spherical Bessel), truncated near `10ℓ` | Scale separation with `method = Spectral()`; in real space the 2-D form needs `using SpecialFunctions` |
| `SmoothHatKernel(; steepness = 10)` | Tanh-tapered top-hat (Storer et al.) | A box without the discontinuity; real space only |
| `HyperGaussianKernel(; α = 1)` | Super-Gaussian, `exp(-α(2d/ℓ)⁴)` | Flatter core, steeper skirt than a Gaussian; real space only |
| `HighOrderKernel{P}(; b_over_ℓ = 1/8)` | `P` vanishing moments, `P ∈ (3, 5)` (Sadek & Aluie `M^I`/`M^II`) | Lifts the filtering spectrum's `k⁻³` slope ceiling. **Separable, not radial**, and signed — needs axes, `ℓ ≥ 8Δx`, and is not for Π |

Only `TopHatKernel`, `GaussianKernel` and `SharpSpectralKernel` have an isotropic spectral transfer
function, so only they can be used with `method = Spectral()`. And only `GaussianKernel` and
`SharpSpectralKernel` have a monotone `|Ĝ|²`, which is what `filtering_spectrum` requires — see
`Kernels.transfer_monotone`, or just ask `check_setup`.

Real-space filtering cost is **not** `O(N · window^d)` for every kernel. `RealSpace()` names the
operator — a local space average against the compact kernel — and the engine that evaluates it is
picked from the grid and kernel. Ask `check_setup` which one a given configuration will take.

| Kernel | Grid | Real-space algorithm | Cost |
|--------|------|----------------------|------|
| `TopHatKernel` | any rectilinear 2D `StructuredGrid` (Cartesian or spherical, uniform or nonuniform) | per-row prefix sums + monotone two-pointer interval sweep | `O(N · w_y)`, exact |
| `TopHatKernel` | uniform Cartesian 3D `StructuredGrid` | per-plane prefix sums; the ball's `x`-interval is contiguous at each `(dy, dz)` | `O(N · w_y · w_z)`, exact |
| `GaussianKernel`, `HighOrderKernel` | Cartesian `StructuredGrid`, 1D/2D/3D, uniform or stretched axes | one separable pass per axis | `O(N · Σᵈwᵈ)`, exact up to the square-vs-disk truncation shape |
| any radial | uniform 2D grid (one band on Cartesian, one per latitude on the sphere) | banded footprint, contiguous axpy along `x` | `O(N · wx · wy)` |
| any | everything else (curvilinear meshes, nonuniform axes) | bounded per-point window (optionally cached, see `AbstractCacheStrategy`) | `O(N · window)` |

Two further engines evaluate that **same** convolution by transform, and are selected with
`method = AutoMethod()`:

| Engine | Applies to | Cost |
|--------|-----------|------|
| padded FFT of the sampled kernel | uniform Cartesian, kernels with no factored engine (`SharpSpectralKernel`, `SmoothHatKernel`) | `O(N log N)` |
| zonal FFT along the longitude ring | global rectilinear sphere, radial kernel — for a fixed pair of latitudes the great-circle weight depends on the longitude difference alone, so each band is a circular convolution | `O(N·(log Nλ + w_φ))` |

Both use the same compact kernel and the same weights, so they agree with the direct sum to round-off
rather than exactly — which is why `RealSpace()` stays the default and they are one keyword away.
`check_setup` flags when `AutoMethod()` would pick a faster engine than the method you asked about.
Neither is `Spectral()`: no transfer function is sampled and no spherical-harmonic truncation happens,
so the kernel keeps its compact support.

### Sweeping scales

A plan splits into three parts by how often each changes: what the **grid** fixes (transform objects,
measure prefix scans, point sorts), what the **scale** fixes (tap tables, `Ĝ(ℓ)`, the reciprocal window
mass), and transient **scratch**. `plan_filter` builds one of each for a single scale;
`plan_filter_sweep` builds the grid part and the scratch **once** for a whole sweep:

```julia
# one grid plan, one scratch, N scale plans
family = CGEF.Filtering.plan_filter_sweep(grid, kernel, scales)

result = CGEF.coarse_grain(u, v, grid; scales = scales)   # allocates the result and a workspace
ws = CGEF.Diagnostics.ΠWorkspace(grid)
for t in 2:nt                                            # later timesteps reuse all three
    CGEF.coarse_grain!(result, us[t], vs[t], grid; scales = scales,
                       filter_plans = family, workspace = ws)
end
```

A `FilterPlanFamily` is an `AbstractVector` of the per-scale plans, so it goes anywhere a vector of
plans did. `coarse_grain` builds one internally for a one-shot sweep; `coarse_grain!` takes a prebuilt
one, along with the result and workspace to write through. One family may not be applied from several
tasks at once — the scratch is shared — so give each concurrent worker its own.

## Execution Backends

The backend only changes *how* the real-space (`RealSpace()`) convolution is evaluated —
results are identical to the serial path. Every backend below reuses a single footprint/plan built
once per `(grid, kernel, scale)` (via `plan_filter`) rather than rebuilding it on every call.

| Backend | Extension | Grid shapes supported | Notes |
|---------|-----------|------------------------|-------|
| `SerialBackend()` | — | Everything | The reference every other backend is asserted bit-identical to |
| `ThreadedBackend()` | OhMyThreads | 2D structured/curvilinear (row-parallel); 1D and true-3D structured, and every flat-cell grid (cell-parallel) | |
| `GPUBackend()` | KernelAbstractions | the same three shapes | Device residency established once with the plan; kernels run the grid's own ball query, so device and host results are bit-identical |
| `DistributedBackend()` | Distributed + SharedArrays | the same three shapes | Multi-process on one shared-memory node via `SharedArray` |
| `MPIBackend()` | MPI | the same three shapes | Multi-node, round-robin decomposition + `Allreduce!`; exercised by `test/mpi_runtests.jl` under `mpiexec` |
| `AutoBackend()` | — | — | Picks `ThreadedBackend` when `nthreads() > 1`, else `SerialBackend` |

`DistributedBackend`/`MPIBackend` are parametric over an inner local backend (e.g.
`MPIBackend(ThreadedBackend())`) for hybrid execution.

## Spherical Commutativity Note

On the sphere, filtering velocity Cartesian components does **NOT** commute with differential operators (Aluie 2019). This package currently uses the "planetary Cartesian" approach, which is:
- **Exact** for non-divergent velocity (e.g., SSH-derived geostrophic flow)
- **Approximate** for full velocity with divergent components

For the theoretically correct approach with general velocity fields, use [HelmholtzDecomposition.jl](https://github.com/jbphyswx/HelmholtzDecomposition.jl) to decompose into scalar potentials, filter those as scalars, then reconstruct, and pass the rotational part to `compute_Π_decomposed`. See Buzzicotti et al. (2023) for the workflow.

## References

- **Aluie (2019)**: doi:10.1007/s13137-019-0123-9 — Convolutions on the sphere
- **Aluie, Hecht, Vallis (2018)**: doi:10.1175/JPO-D-17-0100.1 — Mapping the energy cascade
- **Aluie (2011)**: doi:10.1016/j.physd.2011.06.001 — Compressible turbulence coarse-graining
- **Germano (1992)**: doi:10.1017/S0022112092001733 — The filtering approach (Leonard/Cross/Reynolds)
- **Sadek & Aluie (2018)**: doi:10.1103/PhysRevFluids.3.124610 — Extracting the spectrum by spatial filtering
- **Storer et al. (2022)**: doi:10.1038/s41467-022-33031-3 — Global energy spectrum
- **Buzzicotti et al. (2023)**: doi:10.1126/sciadv.adi7420 — Global cascade of kinetic energy
- **Barkan, Srinivasan & McWilliams (2024)**: doi:10.1175/JPO-D-23-0191.1 — Eddy–internal wave interactions: stimulated cascades (the interaction channel in `compute_Π_decomposed`)

## See Also

- [HelmholtzDecomposition.jl](https://github.com/jbphyswx/HelmholtzDecomposition.jl) — Helmholtz decomposition for correct spherical filtering
- [StructureFunctions.jl](https://github.com/jbphyswx/StructureFunctions.jl) — Structure function analysis (complementary to filtering)
- [FlowSieve](https://flowsieve.readthedocs.io/) — C++ coarse-graining toolkit (Storer et al.)
