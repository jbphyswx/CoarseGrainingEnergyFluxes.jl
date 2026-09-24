```@meta
CurrentModule = CoarseGrainingEnergyFluxes
```

# Architecture

## Module Structure

Geometries and grid types come from **FlowGeometries.jl**, and the execution/spectral backend
taxonomies from **ComputationalBackends.jl** / **SpectralBackends.jl**. This package is the
coarse-graining engine built on them:

```
CoarseGrainingEnergyFluxes (main module)
├── Kernels       — filter kernels + spectral transfer functions Ĝ(|k|, ℓ)
├── Filtering     — real-space footprint convolution engine + spectral plan dispatch
├── Derivatives   — `FG.Operators.derivative!` per direction on a StructuredGrid, plus a
│                   reusable `StencilPlan`. A grid with no separable axis takes the least-squares
│                   tangent-plane gradient from FlowGeometries directly
├── Diagnostics   — energy flux Π (2D/2.5D, vertical-profile, and true 3D), filtering spectrum,
│                   stress / Helmholtz / tracer decompositions
├── Pipeline      — high-level `coarse_grain` orchestration over scales
└── Visualization — `plot_Π_map` / `plot_spectrum` stubs (methods provided by the CairoMakie ext)
```

`Filtering` and `Diagnostics` are each one module assembled from per-topic files. `src/Filtering/`
holds the strategy singletons, the plan lifetimes, the cache strategy, the extension hooks, the public
API, one file per real-space engine under `engines/`, the apply drivers, the plan types, engine
selection and the slice-parallel entry. `src/Diagnostics/` holds the spectrum policy, the flux, the
shared per-point tensor driver, the spectrum, and one file per diagnostic. Every name is still reached
through its module, exactly as a single-file module's would be.

Backend implementations and the spectral transforms live in **package extensions** (weak
dependencies), so the core package has no heavy dependencies.

## Data Flow

```
Input: u(x,y), v(x,y) [, w], grid, kernel, scales
                    │
                    ▼
   plan_filter(grid, kernel, ℓ)      build the footprint / transform plan ONCE per scale
                    │
                    ▼
   filter_apply!(…, plan)            filter ū, v̄ and the quadratic products ⟨u_i u_j⟩
                    │
                    ▼
   ddx! / ddy! / ddz!                resolved strain rate S̄_ℓ = ½(∇ū + ∇ūᵀ)
                    │
                    ▼
   compute_Π!                        Π_ℓ = −S̄_ℓ : τ_ℓ, per unit mass   (τ_ℓ = ⟨u⊗u⟩̄ − ū⊗ū)
                    │
                    ▼
Output: Π(x) per scale, cumulative_energy E(ℓ), filtering spectrum Ẽ(k_ℓ)
```

## 2.5D vertical-profile vs. true 3D

Given 3D `(x, y, z)` velocity, there are two distinct, non-interchangeable ways to get a
"vertical structure":

- **`coarse_grain_profile`** — the literature-standard method (Aluie, Hecht &
  Vallis 2018): run the existing 2D/2.5D `compute_Π!` **independently at each vertical level** and stack
  the results. The vertical axis is a batch axis over the shared horizontal grid, so this is
  `coarse_grain_batch!` with the batch named "level". This is the thin-layer/quasi-geostrophic regime (vertical shear subdominant to
  horizontal gradients — the usual large-scale ocean/atmosphere assumption); levels do not interact.
- **True 3D `compute_Π!`** (`StructuredGrid{...,3}`, Cartesian or spherical-volumetric) — a genuinely
  **coupled** 3D filter kernel and all nine strain/stress components, including real vertical
  derivatives. This is the homogeneous/isotropic-turbulence regime (e.g. Rayleigh–Taylor or
  boundary-layer studies), a different and narrower-audience physics case from the vertical-profile
  method above — the two should never be conflated.

## Execution backends vs. filter method

Two orthogonal choices control *how* a filter is evaluated:

1. **Filter method** (`method = RealSpace()` default, or `Spectral()`):
   - `RealSpace()` — real-space footprint convolution. Supports masks and regional/non-periodic
     domains at arbitrary scales.
   - `Spectral()` — transform → multiply by Ĝ(|k|, ℓ) → inverse transform. `O(N log N)`,
     scale-independent cost. A bounded Cartesian direction is zero-padded, so the result is the filter
     of the field extended by zero beyond the domain; the spherical-harmonic transforms need the whole
     sphere. A partial mask is supported by normalized convolution (Knutsson & Westin 1993), with
     `ZeroFill`/`Deformable` defined as for `RealSpace()`.

2. **Execution backend** (for the `RealSpace()` engine): `SerialBackend`, `ThreadedBackend`
   (OhMyThreads), `GPUBackend` (KernelAbstractions), `DistributedBackend` (Distributed +
   SharedArrays), `MPIBackend` (MPI), or `AutoBackend` (picks threaded when `nthreads() > 1`). All
   backends share the *same* footprint engine, so results are identical to the serial path, and every
   backend reuses a single footprint/plan built once per `(grid, kernel, scale)` rather than
   rebuilding it on every `filter_field!` call. Coverage differs by backend:
   - Every parallel backend covers 2D `StructuredGrid`/`CurvilinearGrid`, decomposed by latitude row,
     and 1D/true-3D `StructuredGrid` plus every flat-cell grid, decomposed by output point — threads
     over `CartesianIndices`, Distributed over a `SharedArray`, MPI round-robin with an `Allreduce!`,
     and the device over one linear index. All of them reuse the per-point kernel the serial n-D
     engine uses.
   - An explicit request for a backend on a grid shape it has no hook for raises an `ArgumentError`;
     only `AutoBackend` downgrades to serial.
   - Every flat-cell layout — a node set, and the ring/cubed-sphere/healpix/icosahedral/Yin–Yang
     pixelizations — builds a `NodeFilterPlan` from the grid's own ball query, and takes `RealSpace()`
     by default like every other architecture.

## Real-space engines

`RealSpace()` names the operator — a local space average against the compact kernel — not a summation
method. Which engine evaluates it is chosen from the grid and kernel:

| engine | selected for | cost |
|---|---|---|
| prefix-sum top-hat | `TopHatKernel`, any rectilinear 2-D grid | `O(N·w_y)`, exact |
| prefix-sum top-hat (3-D) | `TopHatKernel`, uniform Cartesian volume | `O(N·w_y·w_z)`, exact |
| separable two-pass | `GaussianKernel`/`HighOrderKernel`, Cartesian | `O(N·(wx+wy))` |
| separable N-pass | the same, 1-D or true 3-D Cartesian | `O(N·Σ w_d)` |
| banded footprint | other radial kernels on a uniform grid; one band on Cartesian, one per latitude on the sphere | `O(N·wx·wy)` |
| scattered footprint | any nonuniform axis, or a curvilinear mesh | `O(N·wx·wy)`, optional neighbour cache |
| node CSR gather | any flat-cell layout: a node set, or a ring/cubed-sphere/healpix/icosahedral/Yin–Yang pixelization | `O(N·⟨neighbours⟩)` |

Every one of these has a device kernel. The prefix-sum engines run as two launches — a scan that is
sequential along axis 1 and parallel across the rows (2-D) or planes (3-D), then a per-point window
difference. On the device that difference locates its own interval, from the window table where the
axis is uniform and by a binary search over the extended axis where it is not, costing an extra `log`
per point that the host's monotone two-pointer walk avoids.

Two further engines evaluate that **same** convolution by transform, and are reached with
`method = AutoMethod()`:

- **FFT of the sampled kernel** — uniform Cartesian, for kernels with no factored engine. Along a
  periodic axis the transform is circular, with the kernel summed over its images; along a bounded one
  it is zero-padded, so it computes the *linear* convolution and holds on bounded and masked domains.
- **zonal FFT along the longitude ring** — a global rectilinear sphere with a radial kernel. For a
  fixed pair of latitudes the great-circle weight depends on the longitude difference alone, so each
  latitude band is a circular convolution. It is worth most near the poles, where the direct engine's
  longitude window widens as `1/cos φ` and a transform's cost does not.

Both use the same compact kernel and the same weights, so they agree with the direct sum to round-off
rather than exactly — which is why neither is the default. `check_setup` names the engine each method
would select, and flags when `AutoMethod` would choose a faster one.

Neither is [`Filtering.Spectral`](@ref): no transfer function is sampled and no spherical-harmonic truncation is
involved, so the kernel keeps its compact support.

## Spectral backend lattice

`Spectral()` filtering dispatches on grid type to a transform adapter (a thin wrapper that forward
transforms, multiplies by the shared `spectral_transfer`, and inverse transforms):

| Grid | Sampling | Extension | Transform |
|------|----------|-----------|-----------|
| `StructuredGrid{Cartesian}`   | uniform           | `FFTW`                   | real FFT, zero-padded along bounded axes |
| `UnstructuredGrid{Cartesian}` | scattered         | `FINUFFT`                | type-1/2 NUFFT |
| `StructuredGrid{Spherical}`   | uniform (FSH grid)| `FastSphericalHarmonics` | scalar SHT |
| `UnstructuredGrid{Spherical}` | scattered         | `NUFSHT`                 | non-uniform SHT |

## Type Hierarchy

```
AbstractGeometry{T}                 AbstractFilterKernel
├── CartesianGeometry{T}            ├── TopHatKernel
└── SphericalGeometry{T}            ├── GaussianKernel{T}       (α: 6 = Pope, 4 = FlowSieve)
                                    ├── SmoothHatKernel{T}      (tanh-tapered top hat)
                                    ├── HyperGaussianKernel{T}  (exp(-D⁴), flatter core)
                                    ├── HighOrderKernel{P,T}    (P = 3, 5 vanishing moments; separable,
                                    │                            sign-indefinite, no radial form)
                                    └── SharpSpectralKernel

AbstractSpectrumPolicy              (what to do when |Ĝ|² is not monotone decreasing)
├── StrictSpectrum                  refuse (default)
├── ForceSpectrum                   compute anyway, warn once
└── NoSpectrum                      skip; fill NaN
AbstractGrid{G,T}
├── StructuredGrid{G,T,N}      N = 1, 2, 3   (rectilinear; N-D cell measure + mask; N=3 spherical
│                              is a genuine volumetric shell — lon,lat,radius axes, r²cosφ volume)
├── CurvilinearGrid{T,G,...}   2D, model-native (orthogonal curvilinear meshes); exact corner-based
│                              quadrilateral cell areas; independent type params for x/y vs.
│                              the derived areas array (no shared-eltype over-constraint)
└── UnstructuredGrid{T,G,...}  1D, scattered points; k-d tree adjacency (CSR) + Voronoi cell areas;
                               same independent-type-param split as CurvilinearGrid

AbstractExecutionBackend            AbstractFilterMethod    AbstractMaskStrategy
├── SerialBackend                   ├── RealSpace           ├── ZeroFill
├── ThreadedBackend                 └── Spectral            └── Deformable
├── GPUBackend{B}
├── DistributedBackend{Inner}
├── MPIBackend{Inner}
└── AutoBackend
```

## Grid construction: neighbor search & cell areas

`UnstructuredGrid`'s adjacency and per-node area are not required at construction time (a
zero-neighbor grid still supports spectral filtering), but the convenience constructor
`UnstructuredGrid(geometry, x, y, mask; k, radius, areas)` builds both for real, dispatched on
geometry, via three additional weak-dependency extensions:

| Need | Extension | Method |
|------|-----------|--------|
| k-d tree neighbor search (both geometries) | `NearestNeighborsExt` | Cartesian: tree on `(x,y)` directly. Spherical: tree on the exact 3D unit-sphere Cartesian embedding, so chord distance ≡ great-circle distance (exact, not an approximation) |
| Voronoi cell area, Cartesian | `DelaunayTriangulationExt` | Planar Delaunay triangulation → clipped Voronoi dual |
| Voronoi cell area, spherical | `QuickhullExt` | 3D convex hull of the unit-sphere embedding (facets ≡ spherical Delaunay) → L'Huilier spherical-triangle fan-area summation |

Not loading the relevant extension (and not supplying `areas`/adjacency explicitly) raises an
`ArgumentError` naming the exact package needed, rather than silently falling back to a brute-force
or approximate method.

## Plan lifetimes

An engine's state changes at three different rates, and each piece is built once per rate rather than
once per plan:

| lifetime | depends on | examples | built |
|---|---|---|---|
| **grid plan** (`AbstractGridPlan`) | grid, kernel family, mask strategy, method | measure prefix scans, the extended axis and its permutation, FFT/SHT transform objects, NUFFT point sorts | once per sweep |
| **scale plan** | grid plan, ℓ | support radius and band limit, per-axis tap tables, the reciprocal window mass `1/den(ℓ)`, the transfer function `Ĝ(ℓ)` | once per scale |
| **scratch** (`AbstractFilterScratch`) | grid, applied array rank, batch shape | the per-field prefix scans, row-pass and masked-input buffers, padded transform buffers, the complex spectrum buffers | once per concurrent worker |

[`Filtering.plan_filter`](@ref) builds one of each for a single scale. A sweep uses `plan_filter_sweep`, which
returns a `FilterPlanFamily`: one grid plan and one scratch, shared by a vector of per-scale plans.
The family indexes and iterates like that vector, so it drops in wherever per-scale plans were passed.

Scratch is the only part mutated during an apply. A driver that runs applies concurrently —
`filter_slices!`, the batch pipeline drivers, Distributed/MPI ranks — must therefore give each worker
its own family; grid and scale plans are freely shared.

Two spectral backends narrow that: `FastSphericalHarmonics`' `SphPlanCache` is a memo table its
transform populates on first use, and a FINUFFT/NUFSHT guru plan carries the working state of its own
execution. Those grid plans are written during an apply, so they too go one per worker.

## Plan reuse & workspace pre-allocation

`plan_filter` builds the convolution footprint (or cached transform plan) once; `filter_apply!`
reuses it across every velocity component, quadratic product, and vertical layer. `compute_Π!` accepts a
pre-allocated `ΠWorkspace` to avoid per-scale allocations when sweeping scales:

```julia
ws = CGEF.Diagnostics.ΠWorkspace(grid; has_w = w !== nothing)   # allocate once
for ℓ in scales
    CGEF.Diagnostics.compute_Π!(Π, u, v, w, grid, kernel, ℓ; workspace = ws)
end
```

`has_w` is a construction argument: the vertical-component buffers have to exist before the first call,
so a workspace built without them refuses a `w` and names the fix.

The high-level [`coarse_grain`](@ref) handles plan reuse and the scale sweep automatically.
