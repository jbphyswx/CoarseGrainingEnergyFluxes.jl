# API Reference

```@meta
CurrentModule = CoarseGrainingEnergyFluxes
```

One page per submodule, each a complete `@autodocs` listing of what that module defines.

| page | what it covers |
|---|---|
| [Pipeline](reference/pipeline.md) | `coarse_grain` and its relatives, `check_setup`, the result types, and the plotting stubs |
| [Diagnostics](reference/diagnostics.md) | `compute_Π!`, the decompositions, the spectrum, the tracer/enstrophy/Favre budgets, and every workspace |
| [Filtering](reference/filtering.md) | `filter_field!`, the plan family, the real-space engines, mask strategies and methods |
| [Kernels and derivatives](reference/kernels.md) | the filter kernels, their spectral transfer, and the derivative operators |

Only a minimal set of names is exported at the top level; everything else is reached through its
submodule, as in `CGEF.Diagnostics.compute_Π!`. See [Architecture](architecture.md) for the layout.
