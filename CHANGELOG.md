# Changelog

All notable changes to CoarseGrainingEnergyFluxes.jl are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

A correctness/performance/feature overhaul is in progress (see the project plan). This entry
tracks the work as it lands.

### Added
- **`Derivatives.StencilPlan(grid)`** — the finite-difference weights of every direction, built once.
  They depend only on the axis, its wrap period and the order, never on a field, so a caller taking
  many derivatives on one grid builds this once and passes it; `compute_Π!` builds one internally and
  reuses it across all its derivatives and every scale. `ddx!`/`ddy!`/`ddz!` take it as a trailing
  argument and are then **allocation-free** and bit-identical — 0 B against 2880 B per call at 48²
  without it, and 0 B on a masked grid too, the plan also carrying the scratch the degrade path needs
  to rebuild a window at a mask edge.

  The scratch is written per cell, so a plan is **one per task**, the same contract as
  `Connectivity.ball_scratch`. Sharing one across concurrent tasks races.

  `StencilPlan` also builds over a bare axis, for the level-stack `ddz!` whose vertical spacing is an
  argument rather than a grid axis and so cannot come from a grid-built plan: 1472 B per call at
  `Nz = 20` without one, 0 B with.
- **`CoarseGrainResult` stored four of its five fields behind abstract annotations**
  (`AbstractVector{T}`, `AbstractArray{T}`), so every access was a dynamic dispatch and even
  `wavenumber .= L ./ scales` allocated. They are type parameters now, inferred by a new outer
  constructor. `cumulative_energy!` went 544 B → **0 B** and a 4-scale `coarse_grain!` 656 B → **0 B**.

  With the workspace, filter plans and stencil plan held, the whole multi-scale sweep — and
  `compute_Π_profile!` over every level — is now allocation-free, masked or not.

  Every internal caller now holds one: each structured `compute_Π!`, and `coarse_grain`, which builds
  it once for the whole scale sweep rather than per scale — the weights do not depend on the scale. The
  stress/Helmholtz/tracer decompositions took up to 12 derivatives per call, each rebuilding its own
  table; they now share one.
- **`method` on every pipeline entry point.** `coarse_grain`, `coarse_grain!` and
  `coarse_grain_profile` all take `method::Union{Nothing,AbstractFilterMethod}`; previously only the
  `UnstructuredGrid` methods and the generic `coarse_grain!` did, so `Spectral()` was unreachable from
  the allocating API on structured, curvilinear, 1D and profile sweeps. `nothing` still defers to
  `plan_filter`'s per-grid default.

### Changed
- **Discretization moved to FlowGeometries.** `Derivatives.jl` held no coarse-graining content — every
  external name in it was a FlowGeometries one — so it is now a thin layer over the geometry package:
  - the structured `ddx!`/`ddy!`/`ddz!` are `Discretization.derivative!` per direction, one set of
    methods for every geometry instead of six. The metric division and the pole, where `h_λ = R cos φ`
    vanishes and the derivative does not exist, belong to the geometry and are handled there.
  - `WLSQGradientPlan` is **removed** in favour of `Connectivity.gradient_plan` +
    `Discretization.gradient!` — the same least-squares tangent-plane fit, one implementation covering
    curvilinear grids and node sets. `deriv_plan` arguments now take that type. Since `gradient!`
    returns both components from a single neighbour sweep and the strain tensor needs both of every
    field, this also halves the traversals `compute_Π!` does on those two architectures.
  - the local-frame rotation of the subfilter stress is `Geometry.tensor_to_local`.
  - the top-hat row extent is `Connectivity.metric_band`, and the whole-grid index window is
    `Connectivity.metric_window` — O(1) from cached axis statistics, replacing a per-plan `cos` scan
    over the latitude axis.

  Verified against the implementations they replace: derivatives and the top-hat band bit-identical,
  the tensor rotation to 2e-16, the gradient to 1e-15 and allocation-free.
- **Nothing passes `trues(N, N)` as a "no mask" argument any more** — examples, docs, benchmarks and the
  figure generator. Omitting it makes the grid store `AllActive`, whose `isactive` folds to a constant;
  an all-true `BitArray` is a load and a branch per cell and sends every call down the masked path.
  Measured on the derivative sweep: **3.09×** at 256², 1.33× at 1024², bit-identical output. The
  benchmarks were the worst of these, since they were timing the masked path on unmasked data. The
  precompile workload now covers an unmasked grid too, that being what a caller who never mentions a
  mask gets. `trues` remains where a mask is real, and where `CurvilinearGrid`/`UnstructuredGrid`
  require one positionally.
- **Repository split.** The Julia package now lives in its own repository
  (`CoarseGrainingEnergyFluxes.jl`, package at the repo root); the Python implementation moved to
  a separate repository. Julia git history is preserved.

### Removed
- **IO extensions** (CSV, NCDatasets, Zarr) — the package operates on plain arrays + a grid;
  callers handle their own I/O (ecosystem convention).
- **In-package Helmholtz/SOR solver** (`helmholtz_decompose!`, `solve_poisson!`) — rotational/
  divergent decomposition is a preprocessing concern (`HelmholtzDecomposition.jl`); the
  rotational/divergent cascade split will be provided as `compute_Π_decomposed!`.
- **`FastTransforms` extension** — replaced (in progress) by a `FastSphericalHarmonics` extension
  for uniform lat-lon spherical filtering, matching the rest of the ecosystem.
- `test_periodic.jl` print-debug script.

### Added
- **Masked spectral filtering** (`ZeroFill`/`Deformable`) on all four spectral backends (FFTW,
  FINUFFT, FastSphericalHarmonics, NUFSHT) via normalized convolution (Knutsson & Westin 1993): a
  partial mask no longer forces a fallback to `RealSpace()`. `ZeroFill` filters `mask·field` directly
  (the kernel is already normalized so Ĝ(0)=1, matching `RealSpace`'s convention exactly);
  `Deformable` additionally divides by the local kernel mass over active cells, `filter(mask)`,
  computed once at plan-build time (not per `filter_apply!` call) since the mask never changes for a
  fixed plan.
- **Real spectral filtering with `TopHatKernel`**, previously unsupported. Planar (FFTW/FINUFFT):
  the exact 2D-disk Fourier transform `Ĝ(k) = 2J₁(kR)/(kR)` ("jinc", the circular-aperture analog of
  `sinc`), via a new `SpecialFunctions` weak dependency (`CoarseGrainingEnergyFluxesSpecialFunctionsExt`,
  for `besselj1`). Spherical (FastSphericalHarmonics/NUFSHT): the exact spherical-cap window function
  `Ĝ_l = [P_{l-1}(cosθ0) - P_{l+1}(cosθ0)] / [(2l+1)(1-cosθ0)]` (Jekeli 1981's gravity-field averaging
  kernel), via a pure-Julia Legendre-polynomial recurrence — no extra dependency needed there. Both
  transfer functions genuinely oscillate and go negative (the exact, correct behavior of a top-hat's
  spectral content — the same behavior `RealSpace` has too, by the convolution theorem, just not as
  visibly exposed), verified by convergence to the same real-space `RealSpace` result as resolution
  increases. New `spectral_transfer_degree` function for the degree-indexed (spherical) case,
  alongside the existing continuous-wavenumber `spectral_transfer`.
- Code-quality gates wired into the test suite: Aqua, ExplicitImports, and JET, fully enforced.
- Standard package scaffolding: CI / CompatHelper / TagBot / Docs workflows, `.JuliaFormatter.toml`,
  `examples/`, `benchmark/`, and a `gpu/` test environment.
- **Execution-backend taxonomy** (`Backends.jl`) and four parallel backends, each sharing the same
  footprint engine as the serial path (bit-identical results) and each caching its filter plan once
  per `(grid, kernel, scale)` instead of rebuilding it on every call: `ThreadedBackend` (OhMyThreads,
  row-parallel for 2D grids *and* point-parallel for 1D/true-3D grids — the only backend with 1D/3D
  parallel support), `GPUBackend` (KernelAbstractions), `DistributedBackend` (Distributed +
  SharedArrays), `MPIBackend` (multi-node, round-robin row decomposition + `Allreduce!`).
- **Four spectral backends** completing the {Cartesian, spherical} × {uniform, scattered} matrix,
  all driven by one shared `Kernels.spectral_transfer` so the Gaussian convention is identical
  everywhere: `FFTWExt` (uniform periodic Cartesian), `FINUFFTExt` (scattered Cartesian, persistent
  guru NUFFT plans built once), `FastSphericalHarmonicsExt` (uniform spherical, validated against the
  actual FSH quadrature node values, not just grid shape), `NUFSHTExt` (scattered spherical, with an
  exact Clenshaw–Curtis bandlimit when the point count proves it, else a documented heuristic
  fallback).
- **Nonuniform-axis support** for `StructuredGrid`: per-axis spacing is read from the real axis
  (a `Range` proves uniform spacing at the type level; a plain `Vector` triggers a general
  conservative-search real-space footprint) rather than a single global `dx`/`dy`. `ddx!`/`ddy!`/
  `ddz!` use a proper 2nd-order nonuniform 3-point stencil (`Geometry.nonuniform_first_derivative`)
  on nonuniform axes, and a regional (non-periodic) spherical grid's longitude boundary derivative no
  longer silently wraps.
- **`CurvilinearGrid` built genuinely from scratch**: exact quadrilateral corner-based cell areas
  (planar shoelace for Cartesian, L'Huilier spherical-quad for spherical), a per-point real-space
  footprint (`ScatteredFilterPlan`, no translation-invariance assumed), `ddx!`/`ddy!` via
  weighted-least-squares (WLSQ) tangent-plane gradient reconstruction (`WLSQGradientPlan`), and full
  `compute_Π!`/`coarse_grain` support sharing the same per-point tensor-rotation kernel as
  `StructuredGrid`.
- **`UnstructuredGrid` built genuinely from scratch**: real k-d tree neighbor search
  (`NearestNeighborsExt`; spherical built on the exact 3D unit-sphere embedding so chord distance ≡
  great-circle distance), real per-node Voronoi cell areas (`DelaunayTriangulationExt` for planar
  Cartesian, `QuickhullExt` for spherical via a 3D convex hull + L'Huilier fan-area summation),
  `ddx!`/`ddy!` via WLSQ over the k-d tree adjacency (`UnstructuredWLSQGradientPlan`), and full
  `compute_Π!`/`coarse_grain` support, defaulting to spectral filtering (FINUFFT/NUFSHT); the
  node-set real-space engine is listed separately below.
- **True 3D support**, distinguished from the existing 2.5D method (documented as the standard
  thin-layer/quasi-geostrophic scaling, valid when vertical shear is subdominant to horizontal
  gradients — Vallis; Pedlosky): genuinely coupled 3D Cartesian `compute_Π!` (all nine strain/stress
  components) and, new, true 3D **spherical volumetric** support — a radius/depth axis, a spherical-
  shell volume element `dV = r²cosφ·dλ·dφ·dr` (`Geometry.volume_element`), 3D spherical `ddx!`/
  `ddy!`/`ddz!` using the local radius at each level, and the full 3×3 planetary-Cartesian tensor
  rotation. `coarse_grain`/`cumulative_energy`/`filtering_spectrum`/`compute_Π_decomposed`/
  `tracer_variance_flux` are now dimension-generic and cover true 3D Cartesian.
- **`compute_Π_profile!` / `coarse_grain_profile`**: the literature-standard "vertical structure"
  method (Aluie, Hecht & Vallis 2018) — the existing 2D/2.5D `compute_Π!` run independently at each
  depth level of a 3D `(lon,lat,depth)` array and stacked into a profile, distinct from (and not to be
  confused with) the coupled true-3D method above.
- **1D `StructuredGrid` full diagnostics**: `ddx!`, a 1-term `compute_Π!`, and a 1D `coarse_grain`
  wrapper.
- **Corrected, both-sided rotational/divergent (Helmholtz) flux decomposition**
  (`compute_Π_decomposed`): splits *both* the strain and the stress before contracting (the previous
  one-sided version, which split only the stress against the full strain, was a genuinely incomplete
  decomposition), giving three exact channels — `Π_RR` (rotational→rotational), `Π_DD`
  (divergent→divergent), and `Π_X`, the interaction / "stimulated cascade" channel (Barkan,
  Srinivasan & McWilliams 2024) — summing to the undecomposed Π to machine precision. Extended to
  true 3D Cartesian. Validated against analytically exact rotational/divergent test fields (a
  streamfunction/potential pair that is exactly non-divergent/irrotational by construction, for any
  wavenumber) rather than a synthetic non-divergence-free split.
- **`tau_decomposition` on spherical grids**: the Leonard/Cross/Reynolds (Germano 1992) decomposition
  now rotates through planetary-Cartesian coordinates before taking moments, then rotates the result
  back to local east/north — matching `compute_Π!`'s existing spherical approach — instead of
  silently building the decomposition from frame-inconsistent raw local components.
- **Real MPI test execution**: `MPI` added to `test/Project.toml`; `test/mpi_runtests.jl` (run via
  `mpiexec -n P`) compares multi-rank `MPIBackend` output (`Allreduce!`-recombined) against the serial
  reference on plain, masked, and periodic-spherical grids; a dedicated `mpi` CI job runs it.
- **Defensive input validation** (boundary-only, no hot-loop cost): `ArgumentError`/`DimensionMismatch`
  for non-positive/non-finite filter scales, mismatched field/grid array sizes, empty/fully-masked
  grids, malformed curvilinear corner/area arrays, and backend requests a grid shape can't honor.
- **Test suite hardening**: real convergence-rate tests (refine ≥3×, assert the measured order, not
  just "error is small"); physical-invariant regression tests throughout (a normalized low-pass filter
  can never amplify a field beyond its input range; a single Fourier eigenmode transforms exactly by
  the kernel's spectral transfer function; solid-body rotation ⇒ Π ≈ 0; Voronoi/corner-based areas
  sum to the true domain area); corrected two pre-existing tests whose tolerances didn't match their
  own derivation (the spherical periodic-boundary brute-force reference is now genuinely
  area-weighted, not plain-count-averaged; the strain-rate-tensor test no longer allows 50% error on
  a quantity that is exact for a linear field). No test asserts on wall clock: the NUFFT
  mode-count regression is gated on the mode grid the plan actually builds — an integer it already
  stores — so the assertion is exact rather than a threshold that can trip under load.
- New weak dependencies: `NearestNeighbors` (k-d tree neighbor search), `DelaunayTriangulation` +
  `Quickhull` (exact planar/spherical Voronoi cell areas), each with a matching extension.

### Fixed
- **The spherical-cap top-hat transfer function lost precision at small filter widths.**
  `[P_{l-1}(x) - P_{l+1}(x)] / [(2l+1)(1-x)]` cancels twice as the cap shrinks — both the numerator and
  `1-x` go to zero — so it returned 8 significant digits at `ℓ` = 1 km on Earth, 4 at 10 m, and `NaN`
  once `1-x` underflowed. It is evaluated as the algebraically identical `(1+x)·P′_l(x)/(l(l+1))`,
  which has no subtraction of nearly-equal terms: against a 512-bit reference the relative error is
  ≤2.3e-13 at every width and degree tested, against 2.4e-4 before. One Legendre recurrence pass
  instead of one plus an extra step, and no `cos` call.
- **Real-space filtering was wrong at high latitude on a uniform spherical grid**, for any kernel whose
  footprint is the banded `FilterFootprint` (Gaussian, sharp-spectral; the top-hat takes the prefix-sum
  plan and was unaffected). Two defects in the same builder, both only reachable where a ball reaches
  over a pole:
  - the longitude window was `rad / (R·cos φ_j·dλ)` at the **target** row, while the cells it admits lie
    in rows the ball also reaches — nearer the pole, where the same radius spans more longitude. On a
    180-latitude grid at `rad` = 1500 km, a target row at −76.93° is 1451 km from the pole, so the whole
    pole ring is in range, and that bound admitted 60 of its 180 columns — leaving out up to 12% of the
    in-range cell area around ±82°.
  - the offset range was not capped at one turn, so where the window exceeded the ring, columns were
    visited more than once and weighted accordingly.

  Together these gave the **wrong sign** at 77.9° (−0.0372 against a brute-force 0.00847) and a value
  100× too large at 88°. The window now comes from `Connectivity.metric_window`, which takes the
  smallest `cos φ` over the latitude window, and the ring is traversed exactly once. Verified against an
  all-pairs reference: exact at every latitude tested.
- **`ddx!`/`ddy!`/`ddz!` on a `StructuredGrid` did not wrap the stencil index at a periodic Cartesian
  seam** — they wrapped only the spacing, so the first and last rows differenced against the wrong
  neighbour. Measured on a periodic 64 km box: relative error 0.00641 at the seam against 0.00161 in
  the interior; now 0.00161 throughout. Each structured derivative is one
  `Discretization.apply_stencil!` call plus a division by the direction's scale factor, so periodicity,
  masking and nonuniform spacing all come from the grid.
- **The pole guard in the spherical `ddx!` never fired in `Float32`.** Its `1e-12` threshold sits below
  `eps(Float32)`, so `cos(Float32(π/2)) ≈ -4.4e-8` was treated as a regular latitude and the `1/(R cosφ)`
  factor let a spurious finite value through on the pole rows. The threshold is now `√eps(T)` relative
  to the radius.
- **The NUFSHT spectral backend ignored `mask_strategy`**: the local-mass renormalization ran
  unconditionally, so `ZeroFill()` on a masked scattered-spherical grid returned the `Deformable`
  answer. A masked apply also now reuses a plan-owned scratch buffer instead of allocating
  `mask .* field` per call.
- **A filter plan printed its transform library's entire internal plan tree** — 6 KB for a 16×16 FFTW
  transform, and a call into the C library from wherever the plan was shown, including a worker that
  does not own it. Plans now print their type name.
- **The ND scattered window bounds derived the wrap period as extent + `coords[2] - coords[1]`**, which
  is the first gap rather than the wrap gap, so a nonuniform periodic axis wrapped at the wrong length.
  It now reads the grid's stored `period`.
- **Bounded-domain end points are now 2nd order.** Where the stencil ran out of room the derivative fell
  back to a one-sided two-point difference; `apply_stencil!` keeps three nodes and shifts the window
  inward instead. Interior values are unchanged.
- **A test/example building a uniform axis via `collect(range)` (or `deg2rad.(range)`) instead of a
  bare `Range` silently forced the real-space engine's general/scattered footprint path instead of the
  fast shared-stencil path** — since `build_footprint` dispatches on whether both axes are
  `AbstractRange`, and `collect`/broadcasting a range always returns a `Vector`. The scattered path's
  `sizehint!` reserves `Nx*Ny*(2*di_lim+1)^2` entries across three arrays; for one test (a 320×320
  grid, Gaussian kernel radius ≈47 cells) this reserved ~23 GB for what should have been a
  millisecond, sub-megabyte operation, and several documentation-figure-generation functions had the
  same pattern at up to ~20 GB each. Fixed by using bare `Range`s (or, where a function must be
  applied to the bounds, `range(f(first(r)); step = f(step(r)), length = length(r))`, which preserves
  both the exact original point count and `Range`-ness — `f.(r)` for a non-identity `f` broadcasts to
  a `Vector` even when `r` is already a `Range`) at every affected construction site in the test
  suite, examples, and asset-generation script. Verified: the same test now uses the fast path
  (`FilterFootprint`, not `ScatteredFilterPlan`) and allocates under 1 MB per call.
- **A decreasing coordinate axis produced negative per-cell area/volume, silently zeroing every
  `RealSpace`-filtered point.** `_cell_width` computed a cell's physical width/area/volume directly
  from signed coordinate differences (`x[i+1]-x[i]`) with no `abs`, so any axis stored in descending
  order — `lat = π/2 .- θ`, the natural `FastSphericalHarmonics.sph_points` recipe, or any dataset
  storing latitude/depth/pressure levels top-down — got a negative `measure`. The real-space engine's
  `weight_norm > threshold` gate is built from these area-derived weights, so it never passed and
  every output point silently fell to its "no valid neighbours" zero branch, with no error, on a
  completely ordinary, previously-unexercised input shape. Found via a `RealSpace` cross-check on a
  natural FSH-recipe grid returning exactly zero everywhere. Fixed by taking `abs` of the spacing in
  `_cell_width` (a physical measure must be non-negative regardless of storage order); the signed
  gaps in `_local_spacing` itself are unchanged, since the nonuniform derivative stencils need the
  sign to get `df/dx`'s direction right.
- **`compute_Π!` rebuilt its filter footprint on every single call, even when a `workspace` was
  supplied** — the `workspace` parameter only ever covered the scratch *arrays* (`u_filt`/`v_filt`/
  strain/stress buffers); the actual footprint/plan build (`Filtering.plan_filter`, the dominant cost
  for anything but a trivially small grid) was never reusable, contradicting `ΠWorkspace`'s own
  docstring claim of avoiding "reallocating temporaries on every call." Found by actually implementing
  the allocation-regression tests this section describes below, rather than assuming the workspace
  already covered it. This had two concrete, currently-shipping consequences, not just a theoretical
  cost: `compute_Π_profile!` rebuilt the *same* footprint once per depth level (wasted work scaling
  with `Nlevels`), and `coarse_grain!`/`coarse_grain_profile` each independently rebuilt the same
  per-scale footprint a *second* time inside `cumulative_energy!` right after `compute_Π!` had just
  built (and discarded) one for that exact same scale — a measured 3-scale `coarse_grain!` sweep
  allocated ~4.6× the raw sum of the three footprint builds it actually needed. Fixed by adding an
  optional `filter_plan`/`filter_plans` argument (mirroring the existing `deriv_plan` pattern) to
  `compute_Π!`, `compute_Π_profile!`, `cumulative_energy!`, `coarse_grain!`, and `coarse_grain_profile`,
  and threading a single built-once-per-scale plan through all of them. A repeated sweep over the same
  grid/kernel/scales — `coarse_grain!`'s own documented "many timesteps" zero-allocation use case — now
  allocates on the order of 1 KB total for a multi-scale call when the caller supplies a prebuilt
  `workspace`/`filter_plans`, down from tens of KB. Verified with real regression tests
  (`test/test_allocs.jl`) that assert the fast/scattered footprint paths, derivatives, and spectral
  `filter_apply!` calls are exact-zero-allocation on a warmed-up call, and that the pipeline entry
  points are bounded to a small, non-scaling, documented residual rather than "whatever it happens to
  be" — including a sanity check that the bound is actually discriminating (the same call without a
  prebuilt plan allocates over 10× more). Also surfaced, but out of scope for this repository: NUFSHT's
  spectral `filter_apply!` allocates a substantial, real amount per call — confirmed via a direct,
  isolated measurement of `NUFSHT.nusht_filter!` itself (not this package's adapter code) — living
  entirely inside the separately-maintained `NUFSHT.jl` sibling package.
- **`test/mpi_runtests.jl` had never actually been run under a real MPI runtime, and failed
  immediately (all 3 sub-tests, every rank) the first time it was.** `mpiexec -n P` launches each
  rank as an independent OS process, so an unseeded `rand()` call gave every rank a genuinely
  DIFFERENT random field — silently violating the `MPIBackend`'s documented "field replicated across
  ranks" assumption, so the `Allreduce!`-combined result was meaningless (not a bug in `MPIBackend`
  itself — confirmed by seeding identically on every rank, after which the multi-rank result matches
  the serial reference exactly at 2, 3, and 4 ranks). Fixed by seeding the RNG identically across
  ranks before generating each test field.
- **FINUFFT scattered-Cartesian spectral filtering was catastrophically slow and, transiently while
  fixing it, briefly wrong.** The NUFFT mode count was derived from `geometry.dx`/`dy` — a meaningless
  placeholder field for a genuinely scattered `UnstructuredGrid` — which could be wrong by orders of
  magnitude relative to the true point spacing (a 120 s / 4.16 GiB single call observed on an 8-point
  test case). Fixed by deriving the mode count from the actual point count and aspect ratio, and by
  switching from FINUFFT's one-shot convenience API (which silently rebuilds all internal FFTW/spreader
  state on every call) to its persistent guru-plan API, built once per `spectral_filter_plan` call
  (down to 3 ms / 28 KiB measured on the same case).
- **Periodic Cartesian grids silently lost their periodicity in the general/nonuniform-axis
  footprint path.** `_build_footprint_scattered` (2D) and `_build_footprint_nd_scattered` (1D/true
  3D) — the real-space footprint builders used whenever an axis is a plain `Vector` rather than a
  `Range`, e.g. every axis built via `collect(...)`, uniform-valued or not — wrapped a boundary
  candidate's array INDEX via `mod1` but then measured its physical distance from the wrapped
  index's raw, unshifted coordinate (a full domain-width away for Cartesian geometry), so the
  `d <= rad` gate silently rejected every genuinely-close wrapped neighbor: boundary cells behaved as
  if `periodic = false` regardless of the actual flag, with no error. (Spherical grids were
  unaffected: great-circle distance is built from `cos`/`sin` of the raw longitude, which is already
  exactly 2π-periodic regardless of the literal angle value, so no coordinate shift is needed there.)
  Found via a direct cross-check between the general path and the independently-trusted fast
  (`Range`-axis) path, which an existing "no boundary weight corruption" regression test had missed —
  its eigenmode was too smooth and its tolerance too loose to distinguish a correctly-wrapped
  boundary from a silently-truncated one. Fixed by shifting a wrapped Cartesian neighbor's coordinate
  by one period (`extent + one cell spacing`, the same convention used elsewhere) before the distance
  check; a new regression test cross-checks the general and fast paths directly (2D, 1D, and true 3D)
  and asserts they agree exactly.
- **Periodic Cartesian grids used a spherical periodicity constant.** `StructuredGrid`'s 2D
  constructor applied `lon_period = 2π` unconditionally whenever an axis was marked periodic,
  regardless of geometry — meaningless for a Cartesian (meters) axis, whose true period is
  `extent + one cell spacing`. This silently produced physically-impossible (even negative) boundary
  filter weights and output amplification beyond the input's range on periodic Cartesian grids; fixed
  by conditioning the period on geometry type.
- **`FastSphericalHarmonicsExt`/`FINUFFTExt` silently accepted a masked grid and ignored the mask**
  (unlike `FFTWExt`, which already threw); both now raise the same `ArgumentError` FFTW does, directing
  masked/regional use to `method = RealSpace()`.
- **`FastSphericalHarmonicsExt`'s validation only checked grid shape** (`M = 2N-1`), never that
  `grid.lat`/`grid.lon` actually sit on `sph_points(N)`'s quadrature nodes — a shape-correct but
  wrong-node grid silently produced a meaningless transform. Now validated against the real node
  values.
- **`FastSphericalHarmonicsExt.filter_apply!` allocated fresh transpose buffers on every call**
  (`permutedims(field)`/`permutedims(G)`); now uses a cached scratch buffer and in-place
  `permutedims!`.
- **The 2D real-space footprint engine only ever wrapped axis 1, never axis 2, regardless of
  `isperiodic(grid, 2)`** — a doubly-periodic Cartesian box (the standard homogeneous-turbulence
  setup) silently got a non-periodic axis-2 boundary in both the fast (`Range`-axis) and general
  (scattered) footprint paths, and in the GPU kernel's own duplicated offset-wrap logic. Fixed by
  making all three paths honor both axes symmetrically, matching how the 1D/3D footprint builder
  already treated every axis; a new regression test checks a single Fourier mode against its exact
  analytic transfer-function value at every point of a doubly-periodic grid, including corners.

### Changed
- **`StructuredGrid`/`CurvilinearGrid`/`UnstructuredGrid` field, constructor-argument, and local
  identifier names are geometry-neutral** (`lon`/`lat` → `x`/`y`, `Nlon`/`Nlat` → `Nx`/`Ny`,
  `lon_corner`/`lat_corner` → `x_corner`/`y_corner`, etc.) throughout the public API, `src/`, `ext/`,
  examples, docs, and tests — `lon`/`lat` naming remains only where a function is genuinely dispatched
  on `SphericalGeometry` and the value really is a longitude/latitude. Calling a Cartesian axis "lon"/
  "lat" had no physical meaning and is a breaking rename for any caller using positional or keyword
  grid-constructor arguments by name.

### Added
- **Adaptive real-space footprint caching** (`AbstractCacheStrategy`: `AutoCache`, `AlwaysCache`,
  `NeverCache`, new `cache_strategy`/`cache_byte_budget` keywords on `build_footprint`/`plan_filter`)
  for the nonuniform-axis and `CurvilinearGrid` real-space filtering paths. `AutoCache` (the default)
  builds the full per-point neighbour-list cache only when its estimated byte size fits a budget
  (256 MiB by default); `NeverCache` forces streaming — the per-point neighbour list/weight is
  recomputed at apply time from the same compact scalar window bound the builder already derives,
  instead of being stored — for a known memory ceiling; `AlwaysCache` forces the old always-cached
  behavior. Cached and streaming apply are bit-identical (same candidate-iteration order), verified by
  regression tests in `test/runtests.jl`; `Base.summarysize` regression tests in `test/test_allocs.jl`
  confirm the streaming footprint carries only O(1) scalar metadata and does not grow with grid size at
  fixed relative kernel radius, while the optional cache remains the O(N·M) structure a caller can
  still opt into.
- **`filter_apply_batch!`**: apply one filter plan to several same-shape fields at once, deriving each
  target point's neighbour list/weight exactly once and reusing it across every field in the batch
  instead of once per field. `compute_Π!`'s internal filtering (2D/1D/true-3D, Cartesian/spherical) and
  `cumulative_energy!` now use it for their several same-scale filter calls per point, eliminating
  redundant per-field neighbour re-derivation under a streaming plan. Verified bit-identical to calling
  `filter_apply!` per field, and, separately, that it measurably collapses the redundant-derivation
  cost relative to independent per-field calls under `NeverCache`. `ThreadedBackend` batches as well:
  rows (2-D) or index blocks (1-D/true-3-D) run in parallel while the batch still shares one
  neighbour enumeration per point, measured at 3.0× over a per-field threaded apply on a streaming
  scattered plan. The public `filter_fields!` routes through it too, rather than looping
  `filter_apply!` per field and paying that re-derivation itself.
- **Separable Gaussian fast path** (`SeparableGaussianFootprint`, `SeparableGaussianFootprintND`):
  `GaussianKernel`'s weight factors exactly as `∏ᵈ Gᵈ(Δxᵈ)` on a Cartesian grid, so filtering becomes
  one 1-D pass per axis — `O(N·Σᵈrᵈ)` instead of `O(N·∏ᵈrᵈ)` — on a 1-D, 2-D or true-3-D
  `StructuredGrid`. Separability is a property of the kernel, not of the sampling, so a stretched axis
  takes the same engine: the weight table is a vector where the spacing is constant (a `Range` axis)
  and a `(2r+1) × N` matrix per position where it is not, with the cell measure folded in. A
  non-Cartesian grid (e.g. spherical, where great-circle distance does not factor this way) falls back
  to the general footprint unchanged. Masked-normalization semantics (`ZeroFill`/`Deformable`) are
  preserved exactly via the same separable machinery applied to the mask. Cross-checked against an independent brute-force square-truncated
  reference in `test/runtests.jl`. Supported on every execution backend, not just serial: the
  row-pass-then-column-pass algorithm isn't row-independent the way the disk-truncated footprint
  engines are (the column-pass reads across rows), so each backend gets its own two-phase
  implementation sharing the same per-row/per-column bodies — `ThreadedBackend` (two `tforeach`
  sweeps), `DistributedBackend` (two `SharedArray`-backed distributed sweeps, zero extra
  communication since the array is already shared), `MPIBackend` (a redundant, communication-free
  row-pass on every rank since the input field is already replicated, then a round-robin column-pass
  combined via the existing `Allreduce!`), and `GPUBackend` (two `@kernel` launches with a
  device-resident intermediate buffer).

- **Real-space filtering on a node set** (`NodeFilterPlan`): `plan_filter(::UnstructuredGrid, …;
  method = RealSpace())` builds a per-node neighbour list from the grid's own ball query, so scattered
  data is no longer spectral-only. `Spectral()` remains the default. Agrees with the structured engine
  on identical points to 3.3e-16.
- **Device-resident GPU plans.** `plan_filter(…; backend = GPUBackend(dev))` uploads the footprint
  tables, the mask and the separable scratch planes once, through the new `prepare_workspace` backend
  hook, instead of re-transferring them on every `filter_apply!`. A rectilinear grid's cell measure is
  uploaded as its two axis factors rather than the materialized outer product (16 kB instead of 8 MB
  at 1000²).
- **`tracer_variance_flux` on spherical grids**, 2-D and true-3-D. The subfilter tracer flux is a
  vector, so it follows the same convention as `compute_Π!`: rotate the velocity to planetary
  Cartesian, filter, rotate the flux back to local east/north(/radial), then contract with `∂_j θ̄`.

### Fixed
- **A periodic axis whose search window was wider than the axis itself double-counted cells, making
  near-polar rows of global spherical grids wrong by ~5%.** The real-space candidate enumeration walked
  `2·lim+1` index offsets and wrapped out-of-range ones with `mod1`; when `2·lim+1` exceeded the axis
  length, several offsets mapped onto the same index and each was accumulated as a separate neighbour.
  For a periodic *Cartesian* axis the repeats are at least distinct periodic images (the coordinate is
  shifted by ±one period, so their distances differ); for a periodic *spherical* longitude they are not
  distinct at all — great-circle distance is intrinsically 2π-periodic, no shift is applied, and λ and
  λ+2π are the same physical point, so the duplicate is an exact repeat of a contribution already
  counted. This was not an exotic corner: the longitude window `di_lim` is derived from the smallest
  `cos(φ)` anywhere on the grid, so on any global grid reaching toward the poles it routinely exceeds
  `Nx`. Found by cross-checking against an independent all-pairs great-circle reference while adding the
  prefix-sum engine below: on a 20×20 global grid the polar rows summed 21 contributions over the 20
  genuinely in-radius neighbours, a 4.96% error, while every non-polar row was exact. Fixed by adopting
  the **minimum-image convention** explicitly — each grid cell contributes exactly once, via its
  nearest periodic image — in both the scattered engine and the new prefix-sum engine, which now agree
  with the independent reference to roundoff (5.6e-17 and 2.4e-15 respectively). Note this also fixes
  the analogous axis-2 case, and changes results in the previously-double-counting regime for periodic
  Cartesian grids filtered at more than half the domain width.
- **A periodic Cartesian axis and a periodic spherical longitude need opposite treatments, and one
  flag was serving both.** On a torus `x` and `x + L` are distinct locations, so a cell contributes
  once per periodic image inside the filter radius; on a sphere `λ` and `λ + 2π` are the same point, so
  each of the `Nlon` cells contributes at most once. Cartesian periodic axes now tile (offsets are not
  capped and the displacement carries the full multi-period shift), which makes `RealSpace()` the exact
  periodic convolution and brings it into agreement with `Spectral()`; spherical longitude identifies,
  capped at `Nx` with no shift. On a doubly-periodic box filtered at `ℓ/Lx = 0.375` the error against
  the analytic transfer-function result goes from 1.0e-1 to 1.2e-11. Cache build, streaming apply and
  batched apply now share one candidate iterator, so cached and streaming results are bit-identical
  rather than tolerance-close.
- **`GPUBackend` filtered a spherical grid's polar rows wrong** — 7.6% against the serial result at
  ±88°, exact at the equator. The device streaming kernel carried its own copy of the neighbour
  traversal, and that copy wrapped a periodic longitude with `mod1` and no cap: where the filter window
  exceeds the longitude ring (routine on a global grid, since the window is set by the smallest `cosφ`)
  it visited cells repeatedly and accumulated each visit. Its Cartesian branch had the matching defect,
  displacing every image past the first by a single period regardless of how many turns away it was —
  2.5× too many neighbours admitted at `rad/L = 3.75`. Both are gone with the copy: the kernel now
  calls the same `Connectivity.fold_within` the host does, so the window, the two periodic conventions
  and the distance have one implementation, and device and host results are bit-identical rather than
  merely close.
- **Node-set and curvilinear ball queries never passed a topology,** so the grid's spatial index could
  not be used and every query scanned the whole grid. Building an `UnstructuredGrid` real-space plan
  was `O(n²)` — 6.8 s at n = 8000, now 24 ms — and a `CurvilinearGrid`'s streaming apply was `O(N²)` on
  *every* call: 357 ms at 96², now 22 ms. The topology is built once and stored on the plan. A
  `StructuredGrid` keeps the unindexed one, whose query `metric_window` already bounds.
- **Six real-space fast-path signatures were structurally unmatchable.** They constrained the grid's
  topology type parameter where they meant to constrain its coordinates, so the separable Gaussian, the
  prefix-sum top-hat range variant and both n-D paths were never selected and every grid fell to the
  general `O(N·window²)` engine. With the fast paths reachable, a stretched 2-D Gaussian at 400² runs
  165× faster and a 3-D Gaussian at 24³ 161×.
- **The nonuniform-axis (`StructuredGrid`) and `CurvilinearGrid` real-space `RealSpace` filtering
  paths stored a full per-target-point neighbour list unconditionally** — the search window
  (`di_lim`/`dj_lim`) is already a global scalar bound, so nothing about correctness required storing
  the resulting neighbour list; it can be regenerated identically at apply time. This made a
  `CurvilinearGrid` (which has no other real-space path) or any nonuniform-axis `StructuredGrid` filter
  plan's memory scale as O(N·M) (grid points × kernel window size) regardless of whether that memory
  was ever needed again. Fixed by making the cache optional (see the new `AbstractCacheStrategy` entry
  above); `AutoCache` is the new default, so existing callers on grids small enough to fit the budget
  see no behavior change, while a `CurvilinearGrid`/nonuniform-axis plan on a large grid or tight
  memory budget now streams instead of exhausting memory.
- **The spherical per-latitude-band fast path's `sizehint!` reserved capacity from a single global
  worst-case per-band window (`di_lim`, widest near the poles) applied to every band**, over-reserving
  every band away from the poles relative to its own, tighter window. Fixed with a two-pass build: the
  first pass computes each band's real `di_lim` and sums the exact total entry count; the second reuses
  those precomputed values instead of recomputing them. A regression test in `test/test_allocs.jl`
  measures construction allocation on a near-polar grid and confirms it tracks the actual per-band
  content rather than the old global bound.
- **Every parallel backend hook (`ThreadedBackend`, `DistributedBackend`, `GPUBackend`, `MPIBackend`),
  and the 3D-layered `filter_field!`'s serial branch, silently ignored `mask_strategy` when building a
  footprint from scratch** (`build_footprint(grid, kernel, scale)`, with no `mask_strategy` keyword
  passed at all) — harmless for the disk-truncated footprint engines, whose `ZeroFill`-vs-`Deformable`
  branch is decided entirely at apply time from an explicit argument, but a real correctness bug for
  the new separable-Gaussian footprint, which bakes that choice into the footprint itself at build
  time. A `ZeroFill()` request on a masked Cartesian/uniform-axis Gaussian filter was silently computed
  as `Deformable()` instead on every backend except serial. Fixed by forwarding `mask_strategy` at every
  affected call site; the 3D-layered path's serial branch was also missed and additionally called
  `apply_footprint!` directly (which has no method for the new footprint type — this would have thrown
  `MethodError`, not silently misbehaved), fixed by routing it through the shared `_apply_serial!`
  dispatcher instead. Caught by, and now covered by, the per-backend/per-strategy regression tests
  described above.

### Changed
- **`Filtering.DirectSum` is renamed to `Filtering.RealSpace`** (breaking). "DirectSum" now means
  brute-force spectral transform consistently across this ecosystem, matching
  `SpectralBackends.DirectSumSpectralBackend`; the real-space windowed convolution needed a name that
  says what it is. Update `method = DirectSum()` to `method = RealSpace()`.
- **Execution-backend types now come from `ComputationalBackends.jl`** instead of a package-local
  `Backends` module, so backend selection uses the same types across the ecosystem. Reached as
  `CGEF.ComputationalBackends.SerialBackend()` etc.; `src/Backends.jl` is removed. Behaviour is
  unchanged, including `AutoBackend` resolving to `ThreadedBackend` when the session has threads.
- **The `SharpSpectralKernel` + `RealSpace()` throw/warn guard is removed.** It refused, or nagged
  about, a computation the caller had explicitly requested. `RealSpace()` now computes the
  truncated-sinc real-space form on any grid without comment; the accuracy tradeoff is documented on
  `SharpSpectralKernel` itself. The `Filtering._spectral_alternative_available` hook it relied on is
  gone with it.
- **`filter_field!`/`filter_fields!`'s `workspace=` keyword is renamed to
  `filter_plan::Union{Nothing,AbstractFilterPlan}=nothing`** and is now a real, working reuse point —
  previously it was accepted but never read by any backend's dispatch path (every call rebuilt its
  footprint from scratch regardless). Both functions are now thin wrappers over
  [`plan_filter`](@ref)/[`filter_apply!`](@ref): `filter_plan = filter_plan === nothing ?
  plan_filter(...) : filter_plan`, then `filter_apply!(out, field, filter_plan)` (looped once per
  layer for the 3D case). This also fixes a real bug in the 3D-layered `filter_field!` method: its
  serial branch built one footprint and reused it across every vertical layer, but every *other*
  backend recursed into the 2D method per layer with no plan to reuse, silently rebuilding the
  footprint once per layer — the rewrite fixes this uniformly for every backend. `serial_filter_field!`
  (now dead — nothing calls it once `filter_field!` no longer dispatches to it directly) is removed.
  No caller in this repository ever passed `workspace=` (grep-confirmed), so this rename has no
  effect on any existing call site here; external callers passing `workspace=` positionally as a
  keyword need to rename it to `filter_plan=`.

### Added
- **`plan_filter(...; method = Spectral(), spectral_backend = ...)`** — the spectral transform
  algorithm is now selectable rather than implicit. Previously the choice was whatever extension
  happened to define a `spectral_filter_plan` method for the grid type, with no way to ask for a
  specific one; `spectral_filter_plan` now takes a `SpectralBackends.jl` tag as its first argument
  (`FFTSpectralBackend`, `NUFFTSpectralBackend`, `FSHTSpectralBackend`, `NUFSHTSpectralBackend`), and
  the default `AutoSpectralBackend()` preserves the previous grid-type-driven behaviour exactly.
- **Exact `O(N·dj_lim)` real-space top-hat filtering** (`PrefixSumTopHatPlan`), replacing the
  `O(N·di_lim·dj_lim)` windowed sum for `TopHatKernel` on every rectilinear 2D `StructuredGrid` —
  Cartesian or spherical, uniform or nonuniform axes, masked or not, periodic or not. Two properties
  make it exact rather than approximate: the cell measure is separable (`area[i,j] == wx[i]*wy[j]`;
  Cartesian `Δx_i·Δy_j`, spherical `Δλ_i · R²cos(φ_j)Δφ_j`), so the area-weighted window sum factors
  into per-row one-dimensional interval sums; and for a fixed row offset the in-support set of axis-1
  indices is a single contiguous interval with endpoints monotone in the target index (on the sphere
  because `cos d = sinφ₁sinφ₂ + cosφ₁cosφ₂cosΔλ` is monotone in `|Δλ|`). The inner sum is therefore
  `O(1)` from a per-row prefix sum and its endpoints `O(1)` amortized from a two-pointer sweep, with no
  per-point neighbour list of any kind. `ZeroFill`'s denominator uses the mask-independent 1-D prefix;
  `Deformable`'s uses a `mask·wx` prefix built once at plan-build time (mirroring the existing
  `invrenorm` convention), and a `Deformable` apply against a `ZeroFill`-built plan on a masked grid now
  throws rather than silently computing `ZeroFill`. A periodic axis is handled by a tripled coordinate
  array so a wrapped support interval stays contiguous, with an explicit full-period branch so a support
  wider than the domain counts each cell exactly once.
  Measured, single-threaded, on the previously-catastrophic case (4,000,000 points, 1 km grid, 100 km
  filter, radius 50 cells): **1.8 s and zero bytes allocated per apply, down from ~179 s**. A radius
  sweep at 512² shows no regression at any width — break-even at a 1-cell radius and monotonically
  better beyond it (3× at 4 cells, 15× at 16, 47× at 64, 129× at 128) — so it is used unconditionally,
  with no crossover threshold. Cross-checked against the previous engine to roundoff (max relative
  difference ~1e-15) across Cartesian/spherical × uniform/nonuniform × masked/unmasked ×
  periodic/non-periodic, plus semantic checks that masking changes the result, that `ZeroFill` and
  `Deformable` genuinely differ, that a constant field is preserved exactly, and that a
  wider-than-the-sphere filter returns the exact global area-weighted mean. `ThreadedBackend` runs it as
  two barrier-separated parallel row sweeps (the prefix table must be complete before any output row
  reads other rows' bands); `MPIBackend` builds the table redundantly per rank with zero communication
  and keeps the existing disjoint-rows `Allreduce!`.
