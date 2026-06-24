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
- Code-quality gates wired into the test suite: Aqua, ExplicitImports, and JET (skeletons; fully
  enforced as the relevant phases land).
- Standard package scaffolding: CI / CompatHelper / TagBot / Docs workflows, `.JuliaFormatter.toml`,
  `examples/`, `benchmark/`, and a `gpu/` test environment.
