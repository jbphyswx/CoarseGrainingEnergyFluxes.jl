module CoarseGrainingEnergyFluxes

using PrecompileTools: @setup_workload, @compile_workload

# Submodules inclusion
include("Backends.jl")
include("Geometry.jl")
include("Grids.jl")
include("Kernels.jl")
include("Filtering.jl")
include("Derivatives.jl")
include("Diagnostics.jl")
include("Pipeline.jl")
include("Visualization.jl")

# Re-export public components from Geometry
using .Geometry: AbstractGeometry, CartesianGeometry, SphericalGeometry
using .Geometry: distance, area_element, to_planetary_cartesian, from_planetary_cartesian
export AbstractGeometry, CartesianGeometry, SphericalGeometry
export distance, area_element, to_planetary_cartesian, from_planetary_cartesian

# Re-export public components from Grids
using .Grids: AbstractGrid, StructuredGrid, CurvilinearGrid, UnstructuredGrid
using .Grids: coords, area, iswet, grid_geometry, size_tuple, isperiodic
export AbstractGrid, StructuredGrid, CurvilinearGrid, UnstructuredGrid
export coords, area, iswet, grid_geometry, size_tuple, isperiodic

# Re-export public components from Kernels
using .Kernels: AbstractFilterKernel, TopHatKernel, GaussianKernel, SharpSpectralKernel
using .Kernels: kernel_weight, kernel_radius, spectral_transfer
export AbstractFilterKernel, TopHatKernel, GaussianKernel, SharpSpectralKernel
export kernel_weight, kernel_radius, spectral_transfer

# Re-export the execution-backend taxonomy from Backends (matches ScatteringTransforms.jl).
using .Backends: AbstractExecutionBackend, SerialBackend, ThreadedBackend, GPUBackend, AutoBackend,
    DistributedBackend, MPIBackend, local_backend, is_distributed
export AbstractExecutionBackend, SerialBackend, ThreadedBackend, GPUBackend, AutoBackend,
    DistributedBackend, MPIBackend, local_backend, is_distributed

# Re-export public components from Filtering
using .Filtering: filter_field!, filter_fields!, AbstractMaskStrategy, ZeroFill, Deformable
using .Filtering: AbstractFilterPlan, plan_filter, filter_apply!
using .Filtering: AbstractFilterMethod, DirectSum, Spectral
export filter_field!, filter_fields!, AbstractMaskStrategy, ZeroFill, Deformable
export AbstractFilterPlan, plan_filter, filter_apply!
export AbstractFilterMethod, DirectSum, Spectral

# Re-export public components from Derivatives
using .Derivatives: AbstractStencilOrder, SecondOrderStencil
using .Derivatives: ddx!, ddy!, ddz!
export AbstractStencilOrder, SecondOrderStencil
export ddx!, ddy!, ddz!

# Re-export public components from Diagnostics
using .Diagnostics: ΠWorkspace, compute_Π!, cumulative_energy, filtering_spectrum, spectral_density,
    tau_decomposition, compute_Π_decomposed, tracer_variance_flux
export ΠWorkspace, compute_Π!, cumulative_energy, filtering_spectrum, spectral_density,
    tau_decomposition, compute_Π_decomposed, tracer_variance_flux

# Re-export public components from Pipeline
using .Pipeline: CoarseGrainResult, coarse_grain
export CoarseGrainResult, coarse_grain

# Visualization stubs (real methods provided by the CairoMakie package extension)
using .Visualization: plot_Π_map, plot_spectrum
export plot_Π_map, plot_spectrum

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
        compute_Π!(Π, u, v, nothing, grid, TopHatKernel(), 4000.0)
    end
end

end # module
