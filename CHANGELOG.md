# Changelog

All notable changes to CoarseGrainingEnergyFluxes.jl are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Diagnostics
- `compute_Π!` on every grid architecture (`StructuredGrid` in 1-D, 2-D and true 3-D, `CurvilinearGrid`,
  `UnstructuredGrid`, the ring, cubed-sphere, HEALPix, icosahedral and Yin–Yang pixelizations) and every
  metric (plane, sphere, spheroid, spherical and spheroidal shell), with the thin-layer `w` form and the
  per-level profile (`coarse_grain_profile[!]`).
- `tau_decomposition` (Germano 1992), `tracer_variance_flux`, `vorticity`, `enstrophy_flux`,
  `compute_Π_strain_convergence` (Srinivasan, Barkan & McWilliams 2023), `compute_Π_decomposed` (the
  both-sided rotational/divergent split with its interaction channel; Barkan, Srinivasan & McWilliams 2024),
  `compressible_flux` (the Favre budget: `Π`, baropycnal work, pressure dilatation) and `band_energies` (the
  repeated-filter Germano identity), on every grid resolving two tangent directions and in true 3-D where
  defined. Each decomposition's total equals `compute_Π!`.
- In-place forms of every diagnostic with workspaces holding only the buffers their configuration reaches
  (`ΠWorkspace(grid, batch_size, w)`, `EnergyWorkspace`, `TauWorkspace`, `Sym3TauWorkspace`,
  `TracerFluxWorkspace`, `PiDecomposedWorkspace`, `FavreWorkspace`, `EnstrophyFluxWorkspace` and their
  spherical forms), allocation-free with held plans, in the memory of their `execution` backend.
- `cumulative_energy` and `filtering_spectrum` (Sadek & Aluie 2018), with the spectrum policies
  `AutoSpectrum`, `StrictSpectrum`, `ForceSpectrum` and `NoSpectrum`.
- `scale_derivative = AutoScaleDerivative() | ExactScaleDerivative() | DifferenceScaleDerivative()`:
  `dE/dℓ = ⟨ū·∂_ℓū⟩` through the filter's own derivative in its width (`Kernels.ScaleRate`,
  `Kernels.TransferRate`, `Kernels.AxisRate`, `Filtering.rate_plans`, `Filtering.window_mass!`).
  `slopes = Slopes()` adds the spectrum's slope per scale and `spectrum_maps = SpectrumMaps()` the local
  spectrum and its slope per cell.
- `derivative = AutoDerivative() | PhysicalDerivative(; nodes = 5) | SpectralDerivative()` (`Derivatives`) on
  every diagnostic that differentiates a filtered field: the derivative of a spectral plan's series and of the
  generalized convolution's spin coefficients, by the quotient rule under a renormalized mask, and stencils of
  any node count or the least-squares gradient elsewhere.

#### Pipeline
- `coarse_grain`, `coarse_grain!`, `coarse_grain_batch!`, `coarse_grain_slices!`, `coarse_grain_profile[!]`
  with `CoarseGrainResult` and `CoarseGrainBatchResult`, batch axes trailing.
- `check_setup(grid, kernel, ℓ)`: the engine, the resolved backend, and whether the kernel carries `Π`, a
  spectrum and `Spectral()`, without building a plan.

#### Filtering
- `plan_filter`, `filter_apply!`, `filter_field!`, `filter_fields!`, `filter_slices!`, and
  `plan_filter_sweep`, whose `FilterPlanFamily` holds one grid plan and one scratch for every scale
  (`scale_rates = ScaleRates() | ScaleCurvatures()` holds a real-space family's rate plans too).
- Kernels `TopHatKernel(; cells = CellOverlap() | CellCentre())`, `GaussianKernel`, `SmoothHatKernel`,
  `HyperGaussianKernel`, `HighOrderKernel{P}` (Sadek & Aluie's `M^I`/`M^II`) and `SharpSpectralKernel`;
  `Kernels.spectral_transfer`, `Kernels.spectral_transfer_degrees!` (the Legendre transform of the real-space
  weight, Jekeli's closed form for a cap), `Kernels.truncation_tol(T)`, `Kernels.is_nonnegative`,
  `Kernels.transfer_monotone`.
- `CellOverlap()`, the top-hat's default: a cell weighs its measure inside the support, the segment, disk or
  ball of a Cartesian grid, a cap on a sphere, the geodesic disk on a spheroid and the chord ball in a shell,
  from the `Geometry` module (closed forms, and Gauss–Legendre quadrature between exact breakpoints).
- Real-space engines chosen by operation count at plan build, or named by `engine`: the prefix-sum top-hat in
  2-D and 3-D with its rim cells, separable passes, the banded footprint, the padded and zonal FFTs of the
  sampled kernel, the scattered and N-dimensional footprints (one weight table per latitude row and level
  on a shell), the binned top-hat over Cartesian points and meshes, the nonuniform-FFT kernel sum, and the
  node CSR gather over every flat-cell layout, held in a space-filling order and refused past the cache
  budget before it is built.
- Mask strategies `ZeroFill` (the default, which commutes with derivatives) and `Deformable` on every engine
  and spectral path.
- `method = Spectral()`: the FFT through any AbstractFFTs provider (`fft_options` reaching its planner), the
  nonuniform FFT (NonuniformFFTs or FINUFFT, through FlowTransformBindings), FastSphericalHarmonics, and
  NUFSHT with `analysis = Quadrature() | LeastSquares()`. A bounded direction is zero-padded by the
  kernel's reach at the largest scale served (`spectral_grid_plan(…; max_scale)`), and the sharp kernel there
  takes the lattice brick wall exactly on the FFT path.
- `vector_filter = GeneralizedConvolution(; lmax, analysis) | CartesianComponents()`: the convolution of
  vectors and tensors on a sphere of Aluie (2019), by spin-weighted transforms (FastSphericalHarmonics,
  NUFSHT, `RingSpinTransforms`, `DirectSpinTransforms`) analyzing by quadrature or a weighted least-squares
  fit, whose outcome `Filtering.analysis_status` reports.
- Spheroids: every kernel on 2-D and `(λ, φ, h)` grids at the geodesic distance (Karney 2013); a top-hat
  whose radius lies between `πb` and half a meridian raises.

#### Execution
- `execution = SerialBackend() | ThreadedBackend() | GPUBackend(device) | DistributedBackend(inner) |
  MPIBackend(inner) | AutoBackend()` on every grid architecture and engine, every map equal to the serial
  one. A process backend's plan places each worker's or rank's share at plan build and runs it on the inner
  backend, a device included. `AutoBackend()` takes `ThreadedBackend()` where OhMyThreads is loaded and
  `nthreads() > 1`.
- Extensions: AbstractFFTs, FFTW, AbstractFFTs with Distributed and SharedArrays, KernelAbstractions (with
  AbstractFFTs), Distributed with SharedArrays, MPI, OhMyThreads, FastSphericalHarmonics, NUFSHT,
  SpecialFunctions and CairoMakie (`plot_Π_map`, `plot_spectrum`).

### Changed
- Grids, geometries, operators and the execution and spectral tags come from FlowGeometries,
  ComputationalBackends and SpectralBackends. Derivatives are FlowGeometries' stencil and least-squares
  gradient plans (`Derivatives.derivative_plan`); grids take FlowGeometries' `topology` tags and
  geometry-neutral coordinate names.
- `Filtering.DirectSum` → `Filtering.RealSpace`, the default method on every grid, node sets included.
- `backend =` → `execution =`, `spectral_backend =` → `transform =`; `filter_field!(…; workspace)` →
  `filter_plan`.
- `ΠWorkspace(grid; has_w)` → `ΠWorkspace(grid, batch_size, w)`; `band_energies(…; maps = true)` →
  `maps = BandMaps()`; `energy_from_filtered[!]` take the plan; `CoarseGrainBatchResult(grid, Nscales,
  batch_size; execution)`.
- Numbers change: the top-hat weighs cells by overlap (`cells = CellCentre()` keeps the centre test); the
  strain of a spectral filter is its series' derivative and differences take 5 nodes
  (`derivative = PhysicalDerivative(; nodes = 3)` gives 3); a vector on a sphere under `ZeroFill` is
  filtered by the generalized convolution; the spectrum takes the exact `ℓ`-derivative where the kernel has
  one.

### Removed
- The IO extensions (CSV, NCDatasets, Zarr); callers read their own data.
- The in-package Helmholtz solver (`helmholtz_decompose!`, `solve_poisson!`): HelmholtzDecomposition.jl
  supplies the rotational velocity `compute_Π_decomposed` takes.
- The FastTransforms extension, `WLSQGradientPlan`, the `Backends` module, `AutoMethod`,
  `serial_filter_field!`, `Pipeline.batch_alloc_shared` and `CoarseGrainBatchResult`'s `alloc`.

### Fixed
- The spherical relative vorticity carries its curvature term `u tanφ/R`, the gauge of `Π`.
- Real-space filtering near the poles of a uniform sphere: the longitude window covers every row the ball
  reaches, and each cell counts once.
- A periodic Cartesian axis tiles, each image within the radius counting, and a spherical longitude
  identifies, each cell counting once; derivatives wrap the stencil index at a periodic seam; a periodic
  range whose period leaves a gap at the seam takes the per-cell engines.
- A decreasing axis has positive cell measure.
- The NUFSHT spectral path honours `mask_strategy`, and every backend's footprint is built for the requested
  strategy.
- The device streaming kernel takes the host's ball traversal.
- The spherical-cap transfer holds its precision at small widths, as `(1 + x)P′_l(x)/(l(l + 1))`.
- `Derivatives.derivative_plan` raises for a node set with no adjacency, through FlowGeometries'
  `gradient_plan`.
