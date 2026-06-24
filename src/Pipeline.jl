module Pipeline

using ..Geometry: Geometry
using ..Grids: Grids
using ..Kernels: Kernels
using ..Filtering: Filtering
using ..Derivatives: Derivatives
using ..Diagnostics: Diagnostics
using ..Backends: Backends

export CoarseGrainResult, coarse_grain

"""
    CoarseGrainResult{T<:AbstractFloat, A<:AbstractArray{T}}

Container for results of a complete coarse-graining multiscale analysis.

# Fields
- `scales::Vector{T}`: filter scales ℓ in meters
- `Π::Vector{A}`: energy-flux maps at each scale (W/m³)
- `cumulative_energy::Vector{T}`: cumulative coarse KE ½ρ₀⟨|ū_ℓ|²⟩ per scale (Sadek–Aluie Eq. 15)
- `wavenumber::Vector{T}`: filtering wavenumber `k_ℓ = L/ℓ` per scale
- `filtering_spectrum::Vector{T}`: filtering spectral density `Ẽ(k_ℓ)` per scale (Eq. 14)

# Examples
```julia
res = coarse_grain(u, v, grid; scales=[10e3, 20e3, 30e3], kernel=TopHatKernel())
# Access results:
res.scales[1]              # First scale (10 km)
res.Π[1]                   # Energy-flux map at 10 km
res.cumulative_energy[1]   # cumulative coarse KE at 10 km
res.filtering_spectrum[1]  # filtering spectral density at k_ℓ = res.wavenumber[1]
```
"""
struct CoarseGrainResult{T<:AbstractFloat, A<:AbstractArray{T}}
    scales::Vector{T}
    Π::Vector{A}
    cumulative_energy::Vector{T}
    wavenumber::Vector{T}
    filtering_spectrum::Vector{T}
end

"""
    coarse_grain(u, v, w, grid; scales, kernel=TopHatKernel(), ρ₀=1025.0, backend=AutoBackend(), mask_strategy=Deformable())
    coarse_grain(u, v, grid; scales, ...)  # 2D convenience wrapper

Perform complete coarse-graining analysis across multiple filter scales.

This is the high-level orchestration function that runs the full pipeline:
1. Pre-allocates reusable workspace arrays
2. Sweeps through all filter scales
3. Computes cross-scale energy flux Π at each scale
4. Computes filtering energy spectrum E(ℓ)

# Arguments
- `u::AbstractMatrix`: Eastward/zonal velocity component (m/s)
- `v::AbstractMatrix`: Northward/meridional velocity component (m/s)
- `w::Union{Nothing,AbstractMatrix}`: Vertical velocity (nothing for 2D)
- `grid::StructuredGrid`: Grid geometry and land mask

# Keyword Arguments
- `scales::AbstractVector`: Vector of filter scales ℓ in meters (e.g., `10e3:10e3:100e3`)
- `kernel::AbstractFilterKernel=TopHatKernel()`: Filter kernel
- `ρ₀::T=1025.0`: Reference density (kg/m³)
- `backend::AbstractExecutionBackend=AutoBackend()`: Execution backend
- `mask_strategy::AbstractMaskStrategy=Deformable()`: Land masking strategy (`ZeroFill()` or `Deformable()`)

# Returns
- `CoarseGrainResult`: Container with scales, Π maps, and spectrum

# Examples
```julia
# Load velocity data
u, v = load_velocity_data()  # Your I/O function

# Create grid
geom = SphericalGeometry(6371000.0)
grid = StructuredGrid(geom, lon_rad, lat_rad, mask)

# Run coarse-graining from 10 km to 100 km
scales = collect(10e3:10e3:100e3)
res = coarse_grain(u, v, grid; scales=scales, kernel=TopHatKernel())

# Plot the filtering spectral density vs filtering wavenumber
plot(res.wavenumber, res.filtering_spectrum, xscale=:log10, yscale=:log10)

# Plot flux at 30 km
heatmap(res.Π[3])  # 3rd scale is 30 km
```

# Performance Notes
- Workspace arrays are pre-allocated once and reused across scales
- The function automatically selects appropriate execution backend
- For large grids, consider using ThreadedBackend or GPU extensions
"""
function coarse_grain(
    u::AbstractMatrix,
    v::AbstractMatrix,
    w::Union{Nothing, AbstractMatrix},
    grid::Grids.StructuredGrid{G,T};
    scales::AbstractVector,
    kernel::Kernels.AbstractFilterKernel = Kernels.TopHatKernel(),
    ρ₀::T = T(1025.0),
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    L::Real = one(T),
) where {T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}}
    
    Nscales = length(scales)
    Nlon, Nlat = Grids.size_tuple(grid)
    
    # 1. Pre-allocate results vectors
    Π_maps = [zeros(T, Nlon, Nlat) for _ in 1:Nscales]
    
    # 2. Pre-allocate a single, reusable workspace for the entire scale sweep
    workspace = Diagnostics.ΠWorkspace(grid)
    
    # 3. Sweep through scales and compute the cross-scale energy transfer Π maps
    for s_idx in 1:Nscales
        scale = T(scales[s_idx])
        Diagnostics.compute_Π!(
            Π_maps[s_idx],
            u, v, w,
            grid,
            kernel,
            scale;
            ρ₀ = ρ₀,
            workspace = workspace,
            backend = backend,
            mask_strategy = mask_strategy
        )
    end
    
    # 4. Cumulative coarse KE per scale, and the filtering spectral density Ẽ(k_ℓ) — its derivative
    #    w.r.t. the filtering wavenumber k_ℓ = L/ℓ (Sadek & Aluie 2018).
    cumE = Diagnostics.cumulative_energy(
        u, v, w, grid, kernel, scales;
        ρ₀ = ρ₀, backend = backend, mask_strategy = mask_strategy,
    )
    kℓ = T(L) ./ Vector{T}(scales)
    spec = Diagnostics.spectral_density(cumE, kℓ)

    return CoarseGrainResult{T, Matrix{T}}(
        Vector{T}(scales),
        Π_maps,
        cumE,
        kℓ,
        spec,
    )
end

# 2.5D Cartesian constructor wrapper (standard ocean/atmosphere model outputs without vertical velocity)
function coarse_grain(
    u::AbstractMatrix,
    v::AbstractMatrix,
    grid::Grids.StructuredGrid{G,T};
    scales::AbstractVector,
    kernel::Kernels.AbstractFilterKernel = Kernels.TopHatKernel(),
    ρ₀::T = T(1025.0),
    backend::Backends.AbstractExecutionBackend = Backends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.Deformable(),
    L::Real = one(T),
) where {T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}}
    return coarse_grain(u, v, nothing, grid; scales=scales, kernel=kernel, ρ₀=ρ₀, backend=backend, mask_strategy=mask_strategy, L=L)
end

end # module
