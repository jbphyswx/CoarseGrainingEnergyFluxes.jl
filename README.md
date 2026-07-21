# CoarseGrainingEnergyFluxes.jl

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
Π(x, ℓ) = −ρ₀ τ_ℓ : S̄_ℓ
```

(Π > 0 forward cascade, Π < 0 inverse cascade). The package also computes the **filtering spectrum**
(Sadek & Aluie 2018), a corrected **rotational/divergent (Helmholtz) three-way split** of Π
(rotational→rotational, divergent→divergent, and the interaction/"stimulated cascade" channel), the
**Leonard/Cross/Reynolds** stress decomposition (Cartesian and spherical), and the **tracer/buoyancy
variance flux** — on masked, regional, or global domains, with real-space (direct-sum) or spectral
(FFTW / FINUFFT / spherical-harmonic / NUFSHT) backends and serial/threaded/GPU/distributed/MPI
execution.

Every diagnostic works across the full grid×dimensionality matrix: 1D transects, 2D (Cartesian or
spherical, single-level or the standard literature "vertical structure" depth-profile method), true
3D (Cartesian and spherical-volumetric, genuinely coupled vertical derivatives), model-native
curvilinear grids (orthogonal curvilinear meshes, via weighted-least-squares gradients), and scattered/unstructured
point clouds (via k-d tree neighbor search, Voronoi cell areas, and non-uniform spectral transforms).

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

### Cross-scale tracer / buoyancy-variance flux
The scalar analogue of Π (buoyancy ⇒ available-potential-energy transfer).

![Tracer flux](docs/src/assets/tracer_flux.png)

### Masking: deformable vs zero-fill
The deformable kernel renormalizes over active cells; the difference between strategies is concentrated at the mask boundary.

![Masking](docs/src/assets/masking.png)

### Spectral filtering on the sphere
Global spherical-harmonic filtering (the FFTW / FINUFFT / FastSphericalHarmonics / NUFSHT backends cover Cartesian/spherical × uniform/scattered).

![Spherical filtering](docs/src/assets/spherical_filtering.png)

### Validation: rigid-body rotation → Π = 0
Pure rotation has no deformation, so the flux must vanish (to machine precision).

![Rigid Rotation Validation](docs/src/assets/rigid_rotation_validation.png)

### Curvilinear (model-native) grids
A sheared/rotated curvilinear mesh filtered via weighted-least-squares gradients — no rectilinear
assumption anywhere in the pipeline.

![Curvilinear grid](docs/src/assets/curvilinear.png)

### Scattered / unstructured point clouds
k-d tree neighbor search + exact Voronoi cell areas + non-uniform spectral filtering (FINUFFT), taking
`compute_Π!` all the way to a real flux map on genuinely scattered observations.

![Unstructured grid](docs/src/assets/unstructured.png)

### True 3D volumetric flux (Cartesian and spherical shells)
Genuinely coupled 3D strain/stress (all nine components) — homogeneous/isotropic-turbulence-style
filtering that blends all three directions in one kernel, distinct from the 2.5D depth-profile method.

![True 3D volumetric flux](docs/src/assets/volumetric_3d.png)

### Depth-profile (2.5D per-level) vertical structure
The literature-standard "vertical structure" method (Aluie, Hecht & Vallis 2018): the existing 2D/2.5D
`compute_Π!` run independently at each depth level and stacked into a profile.

![Depth profile](docs/src/assets/depth_profile.png)

## Quick Start

```julia
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF

# Create grid
geom = CGEF.SphericalGeometry(6.371e6)  # Earth radius in meters
grid = CGEF.StructuredGrid(geom, lon_rad, lat_rad, mask)

# Run multi-scale analysis
scales = collect(10e3:10e3:300e3)  # 10 km to 300 km
result = CGEF.coarse_grain(u, v, grid; scales = scales, kernel = CGEF.TopHatKernel())

# result.Π                 — (Nlon, Nlat, Nscales) stacked flux array; result.Π[:, :, i] at scales[i]
# result.cumulative_energy — ½ρ₀⟨|ū_ℓ|²⟩ per scale (Sadek–Aluie Eq. 15)
# result.wavenumber        — k_ℓ = L/ℓ
# result.filtering_spectrum — Ẽ(k_ℓ) spectral density (Eq. 14)
```

Only a minimal set of names is exported at the top level (`coarse_grain`, `coarse_grain!`,
`coarse_grain_profile`, `CoarseGrainResult`, the three grid types, the two geometries, the three
kernels, `plot_Π_map`/`plot_spectrum`); everything else — `filter_field!`, `compute_Π!`,
`compute_Π_decomposed`, `tau_decomposition`, `tracer_variance_flux`, backends, mask strategies,
`ddx!`/`ddy!`/`ddz!`, `plan_filter`, `ΠWorkspace`, `spectral_transfer`, … — is reached through the
qualified submodule path shown in [Architecture](#architecture) below, e.g.
`CGEF.Diagnostics.compute_Π!(...)`, `CGEF.Filtering.filter_field!(...)`.

## Architecture

```
src/
  Backends.jl     — execution-backend taxonomy (Serial/Threaded/GPU/Distributed/MPI/Auto)
  Geometry.jl     — CartesianGeometry, SphericalGeometry, planetary-Cartesian rotation,
                    tangent-plane projection (curvilinear/unstructured), volume/area elements
  Grids.jl        — StructuredGrid (1D/2D/3D), CurvilinearGrid, UnstructuredGrid
  Kernels.jl      — TopHatKernel, GaussianKernel, SharpSpectralKernel
  Filtering.jl    — filter_field! (real-space footprint engine + spectral plan dispatch)
  Derivatives.jl  — ddx!/ddy!/ddz! (finite-difference on StructuredGrid; weighted-least-squares
                    gradients on CurvilinearGrid/UnstructuredGrid)
  Diagnostics.jl  — compute_Π!, compute_Π_profile!, compute_Π_decomposed, tau_decomposition,
                    tracer_variance_flux, cumulative_energy, filtering_spectrum
  Pipeline.jl     — coarse_grain / coarse_grain! / coarse_grain_profile (high-level orchestration)
  Visualization.jl — plot_Π_map / plot_spectrum stubs (methods provided by the CairoMakie ext)
ext/
  FFTWExt                       — FFT spectral filtering (uniform periodic Cartesian StructuredGrid)
  FINUFFTExt                    — non-uniform FFT filtering (scattered Cartesian UnstructuredGrid)
  FastSphericalHarmonicsExt     — spherical-harmonic transform (uniform spherical StructuredGrid)
  NUFSHTExt                     — non-uniform spherical-harmonic transform (scattered spherical UnstructuredGrid)
  NearestNeighborsExt           — k-d tree neighbor search for UnstructuredGrid construction
  DelaunayTriangulationExt      — exact planar Voronoi cell areas for UnstructuredGrid{Cartesian}
  QuickhullExt                  — exact spherical Voronoi cell areas for UnstructuredGrid{Spherical}
  OhMyThreadsExt                — ThreadedBackend (2D row-parallel; also 1D/true-3D point-parallel)
  GPUExt                        — GPUBackend via KernelAbstractions
  DistributedExt                — DistributedBackend (Distributed + SharedArrays)
  MPIExt                        — MPIBackend (multi-node domain decomposition)
  CairoMakieExt                 — plot_Π_map / plot_spectrum implementations
```

Backend implementations and all spectral/spatial-indexing transforms live in **package extensions**
(weak dependencies), so the core package has no heavy dependencies.

## Grid Types

| Grid | Dimensionality | Real-space filter | Spectral filter | Derivatives | `compute_Π!` |
|------|-----------------|--------------------|-----------------|--------------|--------------|
| `StructuredGrid` | 1D, 2D, true 3D (Cartesian or spherical-volumetric) | Yes | Yes (FFTW 2D Cartesian; FastSphericalHarmonics 2D spherical) | `ddx!`/`ddy!`/`ddz!` | Yes, all dimensionalities + a 2.5D depth-profile wrapper (`compute_Π_profile!`) |
| `CurvilinearGrid` | 2D (model-native, orthogonal curvilinear meshes) | Yes (per-point footprint, no translation invariance assumed) | Not yet (no spectral extension targets it — real-space only) | `ddx!`/`ddy!` via weighted-least-squares (WLSQ) | Yes |
| `UnstructuredGrid` | 1D (scattered points) | Not applicable — spectral only | Yes (FINUFFT Cartesian; NUFSHT spherical) | `ddx!`/`ddy!` via WLSQ over k-d tree adjacency | Yes |

`CurvilinearGrid` and `UnstructuredGrid` are built genuinely from scratch, not thin wrappers: exact
quadrilateral corner-based cell areas (curvilinear) or k-d tree adjacency + real Voronoi tessellation
cell areas (unstructured, `NearestNeighbors`/`DelaunayTriangulation`/`Quickhull`), and the same
`compute_Π!`/`coarse_grain` pipeline as `StructuredGrid`, sharing the per-point tensor-rotation kernel.

```julia
using NearestNeighbors: NearestNeighbors     # enables UnstructuredGrid's k-d tree neighbor search
using DelaunayTriangulation: DelaunayTriangulation  # enables exact Voronoi areas (Cartesian)
# using Quickhull: Quickhull                 # enables exact Voronoi areas (spherical)

ug = CGEF.UnstructuredGrid(geom, lon, lat, mask; k = 8)   # k-nearest neighbors, auto Voronoi areas
```

## Filter Kernels

| Kernel | Description | Use case |
|--------|-------------|----------|
| `TopHatKernel()` | Uniform weight within radius ℓ | Standard, most common (real-space only — its spectral transfer rings) |
| `GaussianKernel(; α = 6)` | Gaussian with std ℓ/2 | Smooth, differentiable, has an exact spectral transfer |
| `SharpSpectralKernel()` | Ideal low-pass in spectral space | Perfect scale separation for spectral filtering |

## Execution Backends

The backend only changes *how* the real-space (`DirectSum()`) footprint convolution is evaluated —
results are identical to the serial path. Every backend below reuses a single footprint/plan built
once per `(grid, kernel, scale)` (via `plan_filter`) rather than rebuilding it on every call.

| Backend | Extension | Grid shapes supported | Notes |
|---------|-----------|------------------------|-------|
| `SerialBackend()` | — | All (1D/2D/3D, Structured/Curvilinear; Unstructured via spectral) | Default for small grids |
| `ThreadedBackend()` | OhMyThreads | 2D (row-parallel) **and** 1D/true-3D (point-parallel) | Only backend with 1D/3D parallel support |
| `GPUBackend()` | KernelAbstractions | 2D (`StructuredGrid`/`CurvilinearGrid`) | Footprint reused; device buffer upload still per-call |
| `DistributedBackend()` | Distributed + SharedArrays | 2D | Multi-process via `SharedArray` |
| `MPIBackend()` | MPI | 2D | Multi-node, round-robin row decomposition + `Allreduce!`; exercised by `test/mpi_runtests.jl` under `mpiexec` |
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
