module CoarseGrainingEnergyFluxes

using PrecompileTools: PrecompileTools

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

# Bind each submodule's own name only (never `using X: specific_function`) — internal code and
# extensions reach everything through `Submodule.func(...)`, never a flattened top-level re-export.
using .Backends: Backends
using .Geometry: Geometry
using .Grids: Grids
using .Kernels: Kernels
using .Filtering: Filtering
using .Derivatives: Derivatives
using .Diagnostics: Diagnostics
using .Pipeline: Pipeline
using .Visualization: Visualization

# Top-level exports are deliberately minimal: only the few headline entry points a new user needs to
# get started. Everything else (backends, mask strategies, ddx!/ddy!/ddz!, coords/area/isactive,
# compute_Π!, plan_filter, ΠWorkspace, tau_decomposition, spectral_transfer, ...) is reachable via
# qualified submodule access (`CoarseGrainingEnergyFluxes.Diagnostics.compute_Π!(...)`, etc.) rather
# than being re-exported — every additional top-level export is a namespace-clash risk against other
# packages a user has loaded, and a flat 60+-name export list actively hurts discoverability instead
# of helping it. Users who want deeper internals can already reach them qualified, or bind their own
# local aliases; that's a better default than exporting everything "because it's there."
using .Pipeline: coarse_grain, coarse_grain!, coarse_grain_profile, CoarseGrainResult
using .Grids: StructuredGrid, CurvilinearGrid, UnstructuredGrid
using .Geometry: CartesianGeometry, SphericalGeometry
using .Kernels: TopHatKernel, GaussianKernel, SharpSpectralKernel
using .Visualization: plot_Π_map, plot_spectrum

export coarse_grain, coarse_grain!, coarse_grain_profile, CoarseGrainResult
export StructuredGrid, CurvilinearGrid, UnstructuredGrid
export CartesianGeometry, SphericalGeometry
export TopHatKernel, GaussianKernel, SharpSpectralKernel
export plot_Π_map, plot_spectrum

# Precompile workload to minimize Time To First Execution (TTFX)
PrecompileTools.@setup_workload begin
    geom = CartesianGeometry(2000.0, 2000.0)
    lon = collect(0.0:2000.0:10000.0)
    lat = collect(0.0:2000.0:10000.0)
    mask = trues(6, 6)
    grid = StructuredGrid(geom, lon, lat, mask)
    u = rand(6, 6)
    v = rand(6, 6)

    PrecompileTools.@compile_workload begin
        out = zeros(6, 6)
        Filtering.filter_field!(out, u, grid, TopHatKernel(), 4000.0)
        Filtering.filter_field!(out, u, grid, GaussianKernel(), 4000.0)

        Π = zeros(6, 6)
        Diagnostics.compute_Π!(Π, u, v, nothing, grid, TopHatKernel(), 4000.0)
    end
end

end # module
