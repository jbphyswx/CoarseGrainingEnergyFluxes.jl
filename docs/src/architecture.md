```@meta
CurrentModule = CoarseGrainingEnergyFluxes
```

# Architecture

## Module Structure

```
CoarseGrainingEnergyFluxes (main module)
├── Backends      — execution-backend taxonomy (Serial/Threaded/GPU/Distributed/MPI/Auto)
├── Geometry      — coordinate systems (Cartesian, Spherical) + planetary-Cartesian transforms
├── Grids         — grid types (Structured 1D/2D/3D, Curvilinear, Unstructured) with land masks
├── Kernels       — filter kernels + spectral transfer functions Ĝ(|k|, ℓ)
├── Filtering     — real-space footprint convolution engine + spectral plan dispatch
├── Derivatives   — finite-difference stencils (2D/3D Cartesian, spherical with curvature)
├── Diagnostics   — energy flux Π, filtering spectrum, stress / Helmholtz / tracer decompositions
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

## Execution backends vs. filter method

Two orthogonal choices control *how* a filter is evaluated:

1. **Filter method** (`method = DirectSum()` default, or `Spectral()`):
   - `DirectSum()` — real-space footprint convolution. Supports land masks, regional/non-periodic
     domains, and arbitrary scales. The only method for masked or bounded fields.
   - `Spectral()` — transform → multiply by Ĝ(|k|, ℓ) → inverse transform. `O(N log N)`,
     scale-independent cost, but assumes a homogeneous (periodic / global) domain with no mask.

2. **Execution backend** (for the `DirectSum()` engine): `SerialBackend`, `ThreadedBackend`
   (OhMyThreads), `GPUBackend` (KernelAbstractions), `DistributedBackend` (Distributed +
   SharedArrays), `MPIBackend` (MPI), or `AutoBackend` (picks threaded when `nthreads() > 1`). All
   backends share the *same* footprint engine, so results are identical to the serial path; the
   parallel backends are latitude-row decomposed (2D structured grids), and non-2D grids fall back to
   the serial n-D engine.

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
├── StructuredGrid{G,T,N}    N = 1, 2, 3   (rectilinear; N-D cell measure + mask)
├── CurvilinearGrid{G,T}                   (model-native curvilinear)
└── UnstructuredGrid{G,T}                  (scattered points)

AbstractExecutionBackend            AbstractFilterMethod    AbstractMaskStrategy
├── SerialBackend                   ├── DirectSum           ├── ZeroFill
├── ThreadedBackend                 └── Spectral            └── Deformable
├── GPUBackend{B}
├── DistributedBackend{Inner}
├── MPIBackend{Inner}
└── AutoBackend
```

## Plan reuse & workspace pre-allocation

`plan_filter` builds the convolution footprint (or cached transform plan) once; `filter_apply!`
reuses it across every velocity component, quadratic product, and depth layer. `compute_Π!` accepts a
pre-allocated `ΠWorkspace` to avoid per-scale allocations when sweeping scales:

```julia
ws = ΠWorkspace(grid)                  # allocate once
for ℓ in scales
    compute_Π!(Π, u, v, w, grid, kernel, ℓ; workspace = ws)
end
```

The high-level [`coarse_grain`](@ref) handles plan reuse and the scale sweep automatically.
