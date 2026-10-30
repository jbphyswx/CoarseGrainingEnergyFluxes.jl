# API Reference

```@meta
CurrentModule = CoarseGrainingEnergyFluxes
```

One page per submodule, each a complete `@autodocs` listing of what that module defines.

| page | what it covers |
|---|---|
| [Pipeline](reference/pipeline.md) | `coarse_grain` and its relatives, `check_setup`, the result types, and the plotting stubs |
| [Diagnostics](reference/diagnostics.md) | `compute_Π!`, the decompositions, the spectrum, the tracer/enstrophy/Favre budgets, and every workspace |
| [Filtering](reference/filtering.md) | `filter_field!`, the plan family, mask strategies, methods and engine selection |
| [Filtering on the sphere](reference/filtering_sphere.md) | the generalized convolution, its spin-weighted transforms and analysis |
| [Filtering engines](reference/filtering_engines.md) | the real-space engines |
| [Spectral plans](reference/spectral_plans.md) | the FFT, nonuniform-FFT and node-set plans of `method = Spectral()` |
| [Kernels and derivatives](reference/kernels.md) | the filter kernels, their spectral transfer and derivatives in `ℓ`, and the derivative tags and operators |
| [Cell geometry](reference/geometry.md) | a cell's area or volume inside a top-hat's support, which `CellOverlap` weighs cells by |

Only a minimal set of names is exported at the top level; everything else is reached through its
submodule, as in `CGEF.Diagnostics.compute_Π!`. See [Architecture](architecture.md) for the layout.
