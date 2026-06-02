# Examples

## Visual Results

### Spatial Filtering at Multiple Scales

![Filtering Scales](../assets/filtering_scales.png)

### Cross-Scale Energy Flux Π(x, ℓ)

![Energy Flux](../assets/energy_flux_pi.png)

### Filtering Energy Spectrum E(ℓ) and Mean |Π|

![Spectrum](../assets/energy_spectrum.png)

### Validation: Rigid-Body Rotation → Π = 0

![Rigid Rotation Validation](../assets/rigid_rotation_validation.png)

## Cartesian Periodic Domain

```julia
using CoarseGrainingEnergyFluxes

# Setup a 100 km × 100 km Cartesian grid with 1 km resolution
dx = 1000.0  # meters
N = 100
geom = CartesianGeometry(dx, dx)
xs = collect(0.0:dx:(N-1)*dx)
ys = collect(0.0:dx:(N-1)*dx)
grid = StructuredGrid(geom, xs, ys)

# Generate synthetic turbulent velocity (replace with your data)
u = randn(N, N)
v = randn(N, N)

# Compute Π at a single scale
Π = zeros(N, N)
compute_Π!(Π, u, v, nothing, grid, TopHatKernel(), 10_000.0)  # 10 km

# Multi-scale analysis
scales = collect(5e3:5e3:50e3)  # 5 km to 50 km
result = coarse_grain(u, v, grid; scales=scales)
```

## Spherical Ocean Domain

```julia
using CoarseGrainingEnergyFluxes

# Create spherical grid from lon/lat arrays (in radians)
geom = SphericalGeometry(6.371e6)
lon = deg2rad.(collect(0.0:0.25:359.75))
lat = deg2rad.(collect(-80.0:0.25:80.0))
mask = trues(length(lon), length(lat))  # or load your land mask

grid = StructuredGrid(geom, lon, lat, mask)

# Load ocean velocity data (your I/O)
# u, v = load_velocity(...)

# Run analysis from 10 km to 300 km
scales = collect(10e3:10e3:300e3)
result = coarse_grain(u, v, grid; scales=scales, kernel=TopHatKernel())

# Plot spectrum
using CairoMakie
fig = Figure()
ax = Axis(fig[1,1]; xlabel="ℓ (km)", ylabel="E(ℓ) (m²/s²)",
          xscale=log10, yscale=log10)
lines!(ax, result.scales ./ 1e3, result.spectrum)
```

## Using Different Backends

```julia
# Multi-threaded (requires OhMyThreads extension)
using OhMyThreads
result = coarse_grain(u, v, grid; scales=scales, backend=ThreadedBackend())

# FINUFFT for non-uniform grids (requires FINUFFT extension)
using FINUFFT
result = coarse_grain(u, v, grid; scales=scales, backend=FINUFFTBackend())

# GPU (requires KernelAbstractions extension)
using KernelAbstractions
result = coarse_grain(u, v, grid; scales=scales, backend=GPUBackend())
```

## With Helmholtz Decomposition (Correct Spherical Filtering)

For full velocity fields with both rotational and divergent components:

```julia
using CoarseGrainingEnergyFluxes
using HelmholtzDecomposition: HelmholtzDecomposition

# 1. Decompose into scalar potentials
helm = HelmholtzDecomposition.helmholtz_decompose(u, v, grid_helmholtz)

# 2. Filter the scalar potentials
ψ_bar = zeros(size(u))
χ_bar = zeros(size(u))
filter_field!(ψ_bar, helm.ψ, grid, TopHatKernel(), ℓ)
filter_field!(χ_bar, helm.χ, grid, TopHatKernel(), ℓ)

# 3. Reconstruct filtered velocity from filtered potentials
# (theoretically correct — commutes with ∇ on S²)
```
