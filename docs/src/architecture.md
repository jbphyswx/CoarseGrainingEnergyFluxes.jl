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
├── Geometry      — a cell's area or volume inside a top-hat's support: box, polygon, cap, geodesic disk, ball
├── Kernels       — filter kernels + spectral transfer functions Ĝ(|k|, ℓ)
├── Filtering     — real-space footprint convolution engine + spectral plan dispatch
├── Derivatives   — `derivative_plan`: FlowGeometries' stencil plans on a StructuredGrid and its
│                   least-squares tangent-plane gradient plan on any other grid, which its strain,
│                   curl and frame rotations take; `ddz!` for a stack of levels
├── Diagnostics   — energy flux Π (2D/2.5D, vertical-profile, and true 3D), filtering spectrum,
│                   stress / Helmholtz / tracer decompositions
├── Pipeline      — high-level `coarse_grain` orchestration over scales
└── Visualization — `plot_Π_map` / `plot_spectrum` stubs (methods provided by the CairoMakie ext)
```

`Filtering` and `Diagnostics` are each one module assembled from per-topic files. `src/Filtering/`
holds the strategy singletons, the plan lifetimes, the cache strategy, the extension hooks, the public
API, one file per real-space engine under `engines/`, the apply drivers, the plan types, engine
selection and the slice-parallel entry. `src/Diagnostics/` holds the spectrum policy, the flux, the
shared per-point tensor driver, the spectrum, and one file per diagnostic. Every name is reached
through its module.

Backend implementations and the spectral transforms live in **package extensions** (weak
dependencies), so the core package has no heavy dependencies.

## Data Flow

```
Input: u(x,y), v(x,y) [, w], grid, kernel, scales
                    │
                    ▼
   plan_filter(grid, kernel, ℓ)      build the footprint / transform plan once per scale
                    │
                    ▼
   filter_apply!(…, plan)            filter ū, v̄ and the quadratic products ⟨u_i u_j⟩
                    │
                    ▼
   FG.Operators.strain_rate!         resolved strain rate S̄_ℓ = ½(∇ū + ∇ūᵀ)
                    │
                    ▼
   compute_Π!                        Π_ℓ = −S̄_ℓ : τ_ℓ, per unit mass   (τ_ℓ = ⟨u⊗u⟩̄ − ū⊗ū)
                    │
                    ▼
Output: Π(x) per scale, cumulative_energy E(ℓ), filtering spectrum Ẽ(k_ℓ)
```

## 2.5D vertical-profile vs. true 3D

Given 3D `(x, y, z)` velocity, there are two distinct ways to get a "vertical structure", answering
different physical questions:

- **`coarse_grain_profile`** — the literature-standard method (Aluie, Hecht &
  Vallis 2018): run the 2D/2.5D `compute_Π!` **independently at each vertical level** and stack
  the results. The vertical axis is a batch axis over the shared horizontal grid, so this is
  `coarse_grain_batch!` with the batch named "level". This is the thin-layer/quasi-geostrophic regime (vertical shear subdominant to
  horizontal gradients — the usual large-scale ocean/atmosphere assumption); levels do not interact.
- **True 3D `compute_Π!`** (`StructuredGrid{...,3}`, Cartesian or spherical-volumetric) — a
  **coupled** 3D filter kernel and all nine strain/stress components, including vertical
  derivatives. This is the homogeneous/isotropic-turbulence regime (e.g. Rayleigh–Taylor or
  boundary-layer studies).

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

2. **Execution backend** (`execution = …`, for the `RealSpace()` engine): `SerialBackend`, `ThreadedBackend`
   (OhMyThreads), `GPUBackend` (KernelAbstractions), `DistributedBackend` (Distributed +
   SharedArrays), `MPIBackend` (MPI), or `AutoBackend` (`ThreadedBackend` where OhMyThreads is loaded and
   `nthreads() > 1`, `SerialBackend` otherwise). All
   backends share the *same* footprint engine, so every map is identical to the serial path's, and
   every backend applies the footprint/plan built once per `(grid, kernel, scale)`. Under `GPUBackend`
   the diagnostics run on the device too: a workspace holds device buffers and device copies of the
   grid, the strain, rotations and contractions run there, and a domain sum is a device reduction,
   equal to the serial one to round-off. Coverage differs by backend:
   - Every parallel backend covers 2D `StructuredGrid`/`CurvilinearGrid`, decomposed by latitude row,
     and 1D/true-3D `StructuredGrid` plus every flat-cell grid, decomposed by output point or line, and
     each runs the row and point bodies the serial engines run.
   - A process backend's plan places each share at plan build. A Distributed worker holds its block, its
     workspace on the inner backend and the shared field and output buffers, in a `Future` it owns; an MPI
     rank holds its contiguous block. An apply runs every block on the inner backend, serial, threaded or a
     device, and assembles the output in the shared buffers or by an in-place `Allreduce!`. A named
     transform or binned engine on a process backend raises.
   - An explicit request for a backend on a grid shape it has no hook for raises an `ArgumentError`;
     only `AutoBackend` downgrades to serial.
   - Every flat-cell layout — a node set, and the ring/cubed-sphere/healpix/icosahedral/Yin–Yang
     pixelizations — builds a `NodeFilterPlan`, a Cartesian node set's read from its point bins and any
     other's from the grid's own ball query, and takes `RealSpace()` by default like every other
     architecture.

## Real-space engines

`RealSpace()` names the operator: a local space average against the compact kernel. Which engine
evaluates it is chosen from the grid and kernel:

| engine | selected for | cost |
|---|---|---|
| prefix-sum top-hat | `TopHatKernel` on a rectilinear 2-D grid of separable measure: any axes under `CellCentre` (but a spheroid's), a uniform axis 1 under `CellOverlap`, each row's whole cells by one interval sum and its rim cells by their overlap weights | `O(N·w_y)` plus the rim cells, exact |
| prefix-sum top-hat (3-D) | `TopHatKernel`, uniform Cartesian volume | `O(N·w_y·w_z)` plus the rim cells, exact |
| separable two-pass | `GaussianKernel`/`HighOrderKernel`, Cartesian | `O(N·(wx+wy))` |
| separable N-pass | the same, 1-D or true 3-D Cartesian | `O(N·Σ w_d)` |
| banded footprint | other radial kernels on a 2-D grid whose axis 1 is uniform, one band on a uniform Cartesian grid and one per row otherwise (a stretched `y` or any latitude axis); `HighOrderKernel` on a sphere of uniform longitude, one correction per latitude | `O(N·wx·wy)` |
| scattered per-point footprint | a nonuniform axis 1, a curvilinear mesh, a spheroid's centre-test top-hat; `HighOrderKernel` corrected per point | `O(N·wx·wy)`, optional neighbour cache |
| N-dimensional footprint | 1-D and true 3-D grids the separable and prefix-sum engines do not take; on a spherical or spheroidal shell of uniform longitude, a ring or a regional span, one weight table per latitude row and level, shifted along longitude | `O(N·∏ w_d)`, optional neighbour cache |
| binned prefix-sum top-hat | `TopHatKernel` over Cartesian points, a node set or a curvilinear mesh: points binned about one spacing wide, per line of bins a scan of the bin totals; under `CellOverlap` a mesh's rim cells by polygon overlap | `O(N·(r/h)^{D−1})`, exact |
| nonuniform-FFT kernel sum | a kernel with a transfer cutoff (`GaussianKernel`) over Cartesian points, where it costs fewer operations than the neighbour list or the list exceeds the cache budget | `O(N log N)`, to the transform's tolerance |
| node CSR gather | any flat-cell layout: a node set, its list read from the point bins where Cartesian and from chord bins on a sphere, or a ring/cubed-sphere/healpix/icosahedral/Yin–Yang pixelization; a list past the cache budget raises before it is built | `O(N·⟨neighbours⟩)` |

`plan_filter(…; engine)` names an engine ([`Filtering.AbstractEngine`](@ref)); the default `AutoEngine()` takes
the one above, or a transform below where it costs fewer operations. A node set's list is built and held in a
space-filling order of its points, so a ball query and an apply read nearby memory.

Every one of these has a device kernel. The prefix-sum engines run as two launches — a scan that is
sequential along axis 1 and parallel across the rows (2-D) or planes (3-D), then a per-point window
difference. Each scan carries the rounding error of its additions alongside the running sum
(`CompensatedScan`), so a window sum is as accurate as summing the window directly in the working
precision, whatever the row length. On the device that difference locates its own interval, from the
window table where the axis is uniform and by a binary search over the extended axis where it is not,
an extra `log` factor per point; the host walks the intervals with a monotone two-pointer sweep.

Two further engines evaluate that **same** convolution by transform. On a CPU backend `RealSpace()`
takes one where it applies and costs fewer operations than the kernel's own engine, counted with a
fused multiply-add as one: one per tap of a direct sum or separable pass, a few per band and scan entry
of the prefix sums, and `2.5 n log₂ n` per real transform of length `n`.

- **FFT of the sampled kernel** — uniform Cartesian, in one, two or three directions, for every kernel,
  over the weights the replaced engine sums: a separable kernel's per-axis tables with their profile
  denominators, a radial kernel's ball. Along a periodic axis the transform is circular, with the kernel
  summed over its images; along a bounded one it is zero-padded, so it computes the *linear*
  convolution and holds on bounded and masked domains. Under `ZeroFill` a radial kernel's denominator
  is its mass over the whole ball of lattice offsets, which the continued lattice gives every cell.
- **zonal FFT along longitude** — a sphere of uniform longitude, a ring or a regional box (zero-padded
  along longitude), with a radial kernel or a `HighOrderKernel` whose latitude rows share one
  correction. For a fixed pair of latitudes the great-circle weight depends on the longitude offset
  alone, so each latitude band is a convolution along longitude, over the banded engine's taps and
  weights. It is worth most near the poles, where the direct engine's longitude window widens as
  `1/cos φ` and a transform's cost does not.

Both agree with the direct sum to round-off. A kernel a few cells wide costs less as a direct sum and
keeps it, and a device, Distributed or MPI plan takes the direct engines. `check_setup` names the engine
a call takes.

Neither is [`Filtering.Spectral`](@ref): no transfer function is sampled and no spherical-harmonic
truncation is involved, so the kernel keeps its compact support.

## Spectral backend lattice

`Spectral()` filtering dispatches on grid type and the `transform` keyword (`AutoSpectralBackend()` by
default) to a transform adapter (a thin wrapper that forward transforms, multiplies by the kernel's
transfer function, `spectral_transfer` per wavevector and `spectral_transfer_degrees!` per spherical
harmonic degree, and inverse transforms):

| Grid | Sampling | Library | Transform |
|------|----------|---------|-----------|
| `StructuredGrid{Cartesian}`   | uniform           | an AbstractFFTs provider (`FFTW` on the host) | real FFT, zero-padded along bounded axes by the kernel's reach |
| `UnstructuredGrid{Cartesian}` | scattered, 1–3 D  | `NonuniformFFTs` or `FINUFFT` | type-1/2 NUFFT through FlowTransformBindings |
| `StructuredGrid{Spherical}`   | `Fejer1Sampling`, `nlon = 2·nlat − 1` | `FastSphericalHarmonics` | scalar SHT |
| `UnstructuredGrid{Spherical}` | scattered         | `NUFSHT`                      | non-uniform SHT, by quadrature or a fit (`analysis`) |

Any other grid is filtered over its cells as a node set, by the nonuniform transform of its geometry;
a regional sphere is completed to the whole sphere by its lattice continued past each bounded edge.

The FFT path runs on the AbstractFFTs interface for host and device arrays alike. `AutoSpectralBackend()` takes
it where an extension declares a provider for the field's memory: FFTW's on the host, and on a device the
`KernelAbstractions` + `AbstractFFTs` extension's; `transform = FFTSpectralBackend()` names it with any
provider loaded. `plan_filter(…; fft_options)` passes planner keywords to the provider's `plan_*` unchanged,
`(; flags = FFTW.MEASURE)` under FFTW, whose extension adds the thread count. Along a bounded axis the
transform pads by the kernel's reach at the largest scale it serves, and the sharp kernel there takes the
lattice brick wall exactly (see [Theory](theory.md)).

A spectral plan runs where its `execution` backend does. The library runs on one thread under `SerialBackend()`
and on `Threads.nthreads()` under `ThreadedBackend()`, `AutoBackend()` and `GPUBackend()`, whose plans
hold the device's memory; FastSphericalHarmonics transforms host memory, so its device plan passes each
field through a host buffer. Under `MPIBackend` every rank holds the field: a nonuniform transform
gives each rank a share of the points and adds their partial analyses, and an FFT or
FastSphericalHarmonics transform runs whole on each rank. Under `DistributedBackend` a nonuniform
transform gives each worker a block of the points and an FFT a block of columns and then of rows;
FastSphericalHarmonics runs in the calling process.

On the surface of a sphere the diagnostics filter vectors and tensors by the generalized convolution,
`Filtering.GeneralizedConvolution`, through spin-weighted transforms of spins 0, 1 and 2 that a
workspace holds (`Filtering.spin_transforms`): FastSphericalHarmonics' on its own grid, NUFSHT's
quadrature over any other spherical grid's cells where a NUFFT library runs it, and otherwise the same
quadrature summed directly: `Filtering.RingSpinTransforms` by rows on a latitude–longitude `StructuredGrid`,
`Filtering.DirectSpinTransforms` cell by cell elsewhere. The libraries run in host memory at the execution
backend's thread count, and a device workspace stages its fields through a host copy; the direct sums
run in the workspace's memory under its loops, one work item per order in the analysis and per cell (per
row, by rows) in the synthesis.

## Type Hierarchy

```
AbstractGeometry{T}                 AbstractFilterKernel
├── CartesianGeometry{T}            ├── TopHatKernel{C}         (C: CellOverlap or CellCentre)
├── SphericalGeometry{T}            ├── GaussianKernel{T}       (α: 6 = Pope, 4 = FlowSieve)
└── SpheroidGeometry{T}             ├── SmoothHatKernel{T}      (tanh-tapered top hat)
                                    ├── HyperGaussianKernel{T}  (exp(-D⁴), flatter core)
                                    ├── HighOrderKernel{P,T}    (P = 3, 5 vanishing moments; product on
                                    │                            the plane, isotropic on a sphere; signed)
                                    └── SharpSpectralKernel

AbstractSpectrumPolicy              (what to do when |Ĝ|² is not monotone decreasing)
├── AutoSpectrum                    density if monotone, else NaN (default of coarse_grain)
├── StrictSpectrum                  refuse (default of filtering_spectrum)
├── ForceSpectrum                   compute anyway, warn once
└── NoSpectrum                      skip; fill NaN
AbstractGrid{G,T}
├── StructuredGrid{T,G,N,…}    N = 1, 2, 3   (rectilinear; N-D cell measure + mask; N=3 spherical
│                              is a volumetric shell — lon,lat,radius axes, r²cosφ volume)
├── CurvilinearGrid{T,G,…}     2D, model-native (orthogonal curvilinear meshes); exact corner-based
│                              quadrilateral cell areas
├── UnstructuredGrid{T,G,…}    1D, scattered points; k-d tree adjacency (CSR) + Voronoi cell areas
└── RingGrid, CubedSphereGrid, HEALPixGrid, IcosahedralGrid, YinYangGrid   1D sphere pixelizations

AbstractExecutionBackend            AbstractFilterMethod    AbstractMaskStrategy
├── SerialBackend                   ├── RealSpace           ├── ZeroFill
├── ThreadedBackend                 └── Spectral            └── Deformable
├── GPUBackend{B}
├── DistributedBackend{Inner}
├── MPIBackend{Inner}
└── AutoBackend

AbstractDerivative                  AbstractScaleDerivative           AbstractEngine
├── AutoDerivative                  ├── AutoScaleDerivative           ├── AutoEngine
├── PhysicalDerivative              ├── ExactScaleDerivative          ├── PaddedFFTEngine
└── SpectralDerivative              └── DifferenceScaleDerivative     ├── ZonalFFTEngine
                                                                      ├── BinnedTopHatEngine
AbstractVectorFilter                AbstractAnalysis                  ├── NUFFTSumEngine
├── GeneralizedConvolution          ├── AutoAnalysis                  └── FootprintEngine
└── CartesianComponents             ├── Quadrature
                                    └── LeastSquares
```

## Grid construction: neighbor search & cell areas

`UnstructuredGrid`'s adjacency and per-node area are not required at construction time (a
zero-neighbor grid still supports spectral filtering), but the convenience constructor
`UnstructuredGrid(geometry, x, y, mask; k, radius, areas)` builds both, dispatched on geometry, via
three of FlowGeometries' weak-dependency extensions:

| Need | Load | Method |
|------|-----------|--------|
| k-d tree neighbor search (both geometries) | `NearestNeighbors` | Cartesian: tree on `(x,y)` directly. Spherical: tree on the exact 3D unit-sphere Cartesian embedding, where chord distance is a monotone function of great-circle distance, so the neighbour sets are exact |
| Voronoi cell area, Cartesian | `DelaunayTriangulation` | Planar Delaunay triangulation → clipped Voronoi dual |
| Voronoi cell area, spherical | `Quickhull` | 3D convex hull of the unit-sphere embedding (facets ≡ spherical Delaunay) → L'Huilier spherical-triangle fan-area summation |

Unless the relevant extension is loaded or `areas`/adjacency are supplied explicitly, the constructor
raises an `ArgumentError` naming the package needed.

## Plan lifetimes

An engine's state changes at three different rates, and each piece is built once at its own rate:

| lifetime | depends on | examples | built |
|---|---|---|---|
| **grid plan** (`AbstractGridPlan`) | grid, kernel family, mask strategy, method | measure prefix scans, the extended axis and its permutation, FFT/SHT transform objects, NUFFT point sorts | once per sweep |
| **scale plan** | grid plan, ℓ | support radius and band limit, per-axis tap tables, the reciprocal window mass `1/den(ℓ)`, the transfer function `Ĝ(ℓ)` | once per scale |
| **scratch** (`AbstractFilterScratch`) | grid, applied array rank, batch shape | the per-field prefix scans, row-pass and masked-input buffers, padded transform buffers, the complex spectrum buffers | once per concurrent worker |

[`Filtering.plan_filter`](@ref) builds one of each for a single scale. A sweep uses `plan_filter_sweep`, which
returns a `FilterPlanFamily`: one grid plan and one scratch, shared by a vector of per-scale plans.
The family indexes and iterates like that vector, so it is accepted wherever per-scale plans are.

Scratch is the only part mutated during an apply. A driver that runs applies concurrently —
`filter_slices!`, the batch pipeline drivers, Distributed/MPI ranks — must therefore give each worker
its own family; grid and scale plans are freely shared.

Two spectral backends narrow that: `FastSphericalHarmonics`' `SphPlanCache` is a memo table its
transform populates on first use, and a NUFFT or NUFSHT plan carries the working state of its own
execution. Those grid plans are written during an apply, so they too go one per worker.

## Plan reuse & workspace pre-allocation

`plan_filter` builds the convolution footprint (or cached transform plan) once; `filter_apply!`
reuses it across every velocity component, quadratic product, and vertical layer. `compute_Π!` accepts a
pre-allocated `ΠWorkspace` to avoid per-scale allocations when sweeping scales:

```julia
ws = CGEF.Diagnostics.ΠWorkspace(grid, (), w)   # allocate once; `w` an array or `nothing`
for ℓ in scales
    CGEF.Diagnostics.compute_Π!(Π, u, v, w, grid, kernel, ℓ; workspace = ws)
end
```

The workspace reads `w`'s type alone: the vertical-component buffers have to exist before the first call,
so a workspace built for `nothing` refuses a `w` and names the fix. The second argument, a tuple of batch
sizes, sizes the buffers for trailing batch axes. `execution` places the workspace where a plan's outputs
live, `ΠWorkspace(grid; execution = CGEF.Filtering.plan_execution(plan))`; a workspace in host memory refuses
a plan whose outputs are in device memory.

The high-level [`coarse_grain`](@ref) handles plan reuse and the scale sweep automatically.
