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
├── Derivatives   — `FG.Discretization.derivative!` per direction on a StructuredGrid, plus a
│                   reusable `StencilPlan`. A grid with no separable axis takes the least-squares
│                   tangent-plane gradient from FlowGeometries directly
├── Diagnostics   — energy flux Π (2D/2.5D, vertical-profile, and true 3D), filtering spectrum,
│                   stress / Helmholtz / tracer decompositions
├── Pipeline      — high-level `coarse_grain` orchestration over scales
└── Visualization — `plot_Π_map` / `plot_spectrum` stubs (methods provided by the CairoMakie ext)
```

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
   compute_Π!                        Π_ℓ = −ρ₀ S̄_ℓ : τ_ℓ   (τ_ℓ = ⟨u⊗u⟩̄ − ū⊗ū)
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
     domains at arbitrary scales. The only method for a bounded (non-periodic) domain.
   - `Spectral()` — transform → multiply by Ĝ(|k|, ℓ) → inverse transform. `O(N log N)`,
     scale-independent cost; requires a homogeneous (periodic / global) domain, but a partial mask is
     supported (normalized convolution, Knutsson & Westin 1993 — `ZeroFill`/`Deformable` work exactly
     as they do for `RealSpace()`, just evaluated in Fourier/spherical-harmonic space instead of real
     space).

2. **Execution backend** (for the `RealSpace()` engine): `SerialBackend`, `ThreadedBackend`
   (OhMyThreads), `GPUBackend` (KernelAbstractions), `DistributedBackend` (Distributed +
   SharedArrays), `MPIBackend` (MPI), or `AutoBackend` (picks threaded when `nthreads() > 1`). All
   backends share the *same* footprint engine, so results are identical to the serial path, and every
   backend reuses a single footprint/plan built once per `(grid, kernel, scale)` rather than
   rebuilding it on every `filter_field!` call. Coverage differs by backend:
   - `ThreadedBackend` is the only parallel backend with 1D/true-3D support: 2D `StructuredGrid`/
     `CurvilinearGrid` are decomposed by latitude row; 1D/true-3D `StructuredGrid` are decomposed by
     output point (`CartesianIndices`), reusing the same per-point kernel the serial n-D engine uses.
   - `GPUBackend`/`DistributedBackend`/`MPIBackend` cover 2D `StructuredGrid`/`CurvilinearGrid` only;
     an explicit request for one of these on an unsupported grid shape raises an `ArgumentError`
     rather than silently falling back (only `AutoBackend` silently downgrades to serial).
   - `UnstructuredGrid` has a real-space engine (`RealSpace()` builds a `NodeFilterPlan` from the
     grid's own ball query), but no parallel decomposition, so it runs serially whichever backend is
     asked for. Its default method remains `Spectral()`.

## Spectral backend lattice

`Spectral()` filtering dispatches on grid type to a transform adapter (a thin wrapper that forward
transforms, multiplies by the shared `spectral_transfer`, and inverse transforms):

| Grid | Sampling | Extension | Transform |
|------|----------|-----------|-----------|
| `StructuredGrid{Cartesian}`   | uniform periodic  | `FFTW`                   | real FFT |
| `UnstructuredGrid{Cartesian}` | scattered         | `FINUFFT`                | type-1/2 NUFFT |
| `StructuredGrid{Spherical}`   | uniform (FSH grid)| `FastSphericalHarmonics` | scalar SHT |
| `UnstructuredGrid{Spherical}` | scattered         | `NUFSHT`                 | non-uniform SHT |

## Type Hierarchy

```
AbstractGeometry{T}                 AbstractFilterKernel
├── CartesianGeometry{T}            ├── TopHatKernel
└── SphericalGeometry{T}            ├── GaussianKernel{T}      (α: 6 = Pope, 4 = FlowSieve)
                                    └── SharpSpectralKernel
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

## Plan reuse & workspace pre-allocation

`plan_filter` builds the convolution footprint (or cached transform plan) once; `filter_apply!`
reuses it across every velocity component, quadratic product, and vertical layer. `compute_Π!` accepts a
pre-allocated `ΠWorkspace` to avoid per-scale allocations when sweeping scales:

```julia
ws = ΠWorkspace(grid)                  # allocate once
for ℓ in scales
    compute_Π!(Π, u, v, w, grid, kernel, ℓ; workspace = ws)
end
```

The high-level [`coarse_grain`](@ref) handles plan reuse and the scale sweep automatically.
