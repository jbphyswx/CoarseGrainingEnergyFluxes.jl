# Filtering

```@meta
CurrentModule = CoarseGrainingEnergyFluxes
```

The filter API, plans, mask strategies, engine selection and the process shares. The real-space engines are on
[Filtering engines](filtering_engines.md), and the vector filter of the sphere on
[Filtering on the sphere](filtering_sphere.md).

```@autodocs
Modules = [Filtering]
Pages = ["Filtering.jl", "Filtering/Api.jl", "Filtering/Apply.jl", "Filtering/CacheStrategy.jl",
         "Filtering/Hooks.jl", "Filtering/Lifetimes.jl", "Filtering/Plans.jl", "Filtering/Rates.jl",
         "Filtering/Selection.jl", "Filtering/Shares.jl", "Filtering/Slices.jl", "Filtering/Strategies.jl"]
```
