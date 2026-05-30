module CoarseGrainingEnergyFluxes

using LinearAlgebra
using StaticArrays
using PrecompileTools

# Submodules inclusion
include("Geometry.jl")
include("Grids.jl")
include("Kernels.jl")
include("Helmholtz.jl")
include("Filtering.jl")
include("Derivatives.jl")
include("Diagnostics.jl")
include("Pipeline.jl")

# Re-export public components from Geometry
using .Geometry: AbstractGeometry, CartesianGeometry, SphericalGeometry
using .Geometry: distance, area_element, to_planetary_cartesian, from_planetary_cartesian
export AbstractGeometry, CartesianGeometry, SphericalGeometry
export distance, area_element, to_planetary_cartesian, from_planetary_cartesian

# Re-export public components from Grids
using .Grids: AbstractGrid, StructuredGrid, CurvilinearGrid, UnstructuredGrid
using .Grids: coords, area, iswet, grid_geometry, size_tuple
export AbstractGrid, StructuredGrid, CurvilinearGrid, UnstructuredGrid
export coords, area, iswet, grid_geometry, size_tuple

# Re-export public components from Kernels
using .Kernels: AbstractFilterKernel, TopHatKernel, GaussianKernel, SharpSpectralKernel
using .Kernels: kernel_weight, kernel_radius
export AbstractFilterKernel, TopHatKernel, GaussianKernel, SharpSpectralKernel
export kernel_weight, kernel_radius

# Re-export public components from Helmholtz
using .Helmholtz: helmholtz_decompose!, solve_poisson!
export helmholtz_decompose!, solve_poisson!

# Re-export public components from Filtering
using .Filtering: AbstractExecutionBackend, SerialBackend, ThreadedBackend, DistributedBackend, GPUBackend, AutoBackend
using .Filtering: filter_field!
export AbstractExecutionBackend, SerialBackend, ThreadedBackend, DistributedBackend, GPUBackend, AutoBackend
export filter_field!

# Re-export public components from Derivatives
using .Derivatives: AbstractStencilOrder, SecondOrderStencil
using .Derivatives: ddx!, ddy!, ddz!
export AbstractStencilOrder, SecondOrderStencil
export ddx!, ddy!, ddz!

# Re-export public components from Diagnostics
using .Diagnostics: PiWorkspace, compute_Pi!, compute_filtering_spectrum
export PiWorkspace, compute_Pi!, compute_filtering_spectrum

# Re-export public components from Pipeline
using .Pipeline: CoarseGrainResult, coarse_grain
export CoarseGrainResult, coarse_grain

# Precompile workload to minimize Time To First Execution (TTFX)
@setup_workload begin
    geom = CartesianGeometry(2000.0, 2000.0)
    lon = collect(0.0:2000.0:10000.0)
    lat = collect(0.0:2000.0:10000.0)
    mask = trues(6, 6)
    grid = StructuredGrid(geom, lon, lat, mask)
    u = rand(6, 6)
    v = rand(6, 6)
    
    @compile_workload begin
        out = zeros(6, 6)
        filter_field!(out, u, grid, TopHatKernel(), 4000.0)
        filter_field!(out, u, grid, GaussianKernel(), 4000.0)
        
        Π = zeros(6, 6)
        compute_Pi!(Π, u, v, nothing, grid, TopHatKernel(), 4000.0)
    end
end

end # module
