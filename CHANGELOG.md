# Changelog

All notable changes to CoarseGrainingEnergyFluxes.jl are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

A correctness/performance/feature overhaul is in progress (see the project plan). This entry
tracks the work as it lands.

### Changed
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
  footprint (`FilterFootprintScattered`, no translation-invariance assumed), `ddx!`/`ddy!` via
  weighted-least-squares (WLSQ) tangent-plane gradient reconstruction (`WLSQGradientPlan`), and full
  `compute_Π!`/`coarse_grain` support sharing the same per-point tensor-rotation kernel as
  `StructuredGrid`.
- **`UnstructuredGrid` built genuinely from scratch**: real k-d tree neighbor search
  (`NearestNeighborsExt`; spherical built on the exact 3D unit-sphere embedding so chord distance ≡
  great-circle distance), real per-node Voronoi cell areas (`DelaunayTriangulationExt` for planar
  Cartesian, `QuickhullExt` for spherical via a 3D convex hull + L'Huilier fan-area summation),
  `ddx!`/`ddy!` via WLSQ over the k-d tree adjacency (`UnstructuredWLSQGradientPlan`), and full
  `compute_Π!`/`coarse_grain` support (spectral filtering only — FINUFFT/NUFSHT — since a genuinely
  scattered point cloud has no real-space footprint engine).
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
  true 3D Cartesian. `HelmholtzDecomposition.jl` added as a **test-only** dependency so this is
  validated against a genuine Helmholtz solver rather than a synthetic non-divergence-free split.
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
  a quantity that is exact for a linear field).
- New weak dependencies: `NearestNeighbors` (k-d tree neighbor search), `DelaunayTriangulation` +
  `Quickhull` (exact planar/spherical Voronoi cell areas), each with a matching extension.

### Fixed
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
  masked/regional use to `method = DirectSum()`.
- **`FastSphericalHarmonicsExt`'s validation only checked grid shape** (`M = 2N-1`), never that
  `grid.lat`/`grid.lon` actually sit on `sph_points(N)`'s quadrature nodes — a shape-correct but
  wrong-node grid silently produced a meaningless transform. Now validated against the real node
  values.
- **`FastSphericalHarmonicsExt.filter_apply!` allocated fresh transpose buffers on every call**
  (`permutedims(field)`/`permutedims(G)`); now uses a cached scratch buffer and in-place
  `permutedims!`.
