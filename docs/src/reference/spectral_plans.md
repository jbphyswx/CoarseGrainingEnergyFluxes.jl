# Spectral plans

```@meta
CurrentModule = CoarseGrainingEnergyFluxes
```

The `method = Spectral()` plans of the core: the FFT path's padding and reach, the nonuniform FFT, and the node-set
plan every other grid takes.

```@autodocs
Modules = [Filtering]
Pages = ["engines/FFTSpectral.jl", "engines/NUFFTSpectral.jl", "engines/NodeSpectral.jl"]
```
