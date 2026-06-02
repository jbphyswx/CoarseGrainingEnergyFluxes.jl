# Architecture

## Module Structure

```
CoarseGrainingEnergyFluxes (main module)
├── Geometry     — Coordinate system abstractions
├── Grids        — Grid types with mask support
├── Kernels      — Filter kernel definitions
├── Filtering    — Core convolution engine
├── Derivatives  — Finite-difference stencils
├── Helmholtz    — Poisson solver for Helmholtz decomposition
├── Diagnostics  — Energy flux and spectrum computation
└── Pipeline     — High-level orchestration
```

## Data Flow

```
Input: u(x,y), v(x,y), grid, kernel, scales
                    │
                    ▼
            ┌───────────────┐
            │  filter_field! │  Filter u, v, and u⊗u products
            └───────────────┘
                    │
                    ▼
            ┌───────────────┐
            │  ddx!, ddy!   │  Compute strain rate S̄_ℓ
            └───────────────┘
                    │
                    ▼
            ┌───────────────┐
            │  compute_Π!   │  τ_ℓ : S̄_ℓ = cross-scale energy flux
            └───────────────┘
                    │
                    ▼
Output: Π(x,y) at each scale ℓ, E(ℓ) spectrum
```

## Extension Mechanism

Extensions are Julia weak dependencies that register execution backends or I/O formats:

```julia
# In ext/CoarseGrainingEnergyFluxesFFTWExt.jl:
function Filtering.filter_field!(out, field, grid, kernel::SharpSpectralKernel, ℓ;
                                  backend::FINUFFTBackend, ...)
    # FFT-based filtering implementation
end
```

The `AutoBackend()` automatically selects the best available backend at runtime.

## Type Hierarchy

```
AbstractGeometry{T}
├── CartesianGeometry{T}
└── SphericalGeometry{T}

AbstractGrid{G, T}
├── StructuredGrid{G, T}       (regular lon/lat or x/y)
├── CurvilinearGrid{G, T}      (model-native curvilinear)
└── UnstructuredGrid{G, T}     (scattered points)

AbstractFilterKernel
├── TopHatKernel
├── GaussianKernel
└── SharpSpectralKernel

AbstractExecutionBackend
├── SerialBackend
├── ThreadedBackend        (ext: OhMyThreads)
├── GPUBackend             (ext: KernelAbstractions)
├── FINUFFTBackend         (ext: FINUFFT)
├── DistributedBackend     (ext: Distributed + SharedArrays)
└── AutoBackend
```

## Workspace Pre-Allocation

`compute_Π!` uses a `ΠWorkspace` struct to avoid repeated allocations when sweeping across scales:

```julia
workspace = ΠWorkspace(grid)  # Allocate once
for ℓ in scales
    compute_Π!(Π, u, v, w, grid, kernel, ℓ; workspace=workspace)
end
```

The high-level `coarse_grain()` function handles this automatically.
