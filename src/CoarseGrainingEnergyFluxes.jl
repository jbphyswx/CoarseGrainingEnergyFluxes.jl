module CoarseGrainingEnergyFluxes

# First, so its `__init__` selects the OpenMP runtime's thread-local mode before FastTransforms loads
# that runtime; see `FlowTransformBindings.with_fasttransforms_threads`.
using FlowTransformBindings: FlowTransformBindings
using PrecompileTools: PrecompileTools
using ComputationalBackends: ComputationalBackends
using FlowGeometries: FlowGeometries
using SpectralBackends: SpectralBackends

include("Kernels.jl")
include("Filtering.jl")
include("Derivatives.jl")
include("Diagnostics.jl")
include("Pipeline.jl")
include("Visualization.jl")

using .Kernels: Kernels
using .Filtering: Filtering
using .Derivatives: Derivatives
using .Diagnostics: Diagnostics
using .Pipeline: Pipeline
using .Visualization: Visualization

using .Pipeline: coarse_grain, coarse_grain!, coarse_grain_profile, coarse_grain_profile!, CoarseGrainResult
using .Pipeline: coarse_grain_batch!, coarse_grain_slices!, CoarseGrainBatchResult
using .Pipeline: check_setup, SetupReport
using .Kernels: TopHatKernel, GaussianKernel, SharpSpectralKernel
using .Visualization: plot_Π_map, plot_spectrum

export coarse_grain, coarse_grain!, coarse_grain_profile, coarse_grain_profile!, CoarseGrainResult
export coarse_grain_batch!, coarse_grain_slices!, CoarseGrainBatchResult
export check_setup, SetupReport
export TopHatKernel, GaussianKernel, SharpSpectralKernel
export plot_Π_map, plot_spectrum

PrecompileTools.@setup_workload begin
    geom = FlowGeometries.Geometry.CartesianGeometry()
    # A Range axis selects the separable footprints and a Vector axis the position-indexed ones.
    xr = 0.0:2000.0:10000.0
    xv = collect(xr)
    # `grid_a` has no mask and stores `AllActive`, a separate specialization from a `BitArray` mask.
    mask = trues(6, 6)
    grid_r = FlowGeometries.Grids.StructuredGrid(geom, xr, xr, mask)
    grid_v = FlowGeometries.Grids.StructuredGrid(geom, xv, xv, mask)
    grid_a = FlowGeometries.Grids.StructuredGrid(geom, xr, xr)
    grid_3d = FlowGeometries.Grids.StructuredGrid(geom, xr, xr, xr, trues(6, 6, 6))
    u = rand(6, 6)
    v = rand(6, 6)
    u3 = rand(6, 6, 6)
    v3 = rand(6, 6, 6)
    w3 = rand(6, 6, 6)
    # The node grid is given its measure and an empty adjacency, so its real-space filter loads no weakdep.
    xc = [Float64(i - 1) * 2000.0 for i in 1:6, j in 1:6]
    yc = [Float64(j - 1) * 2000.0 for i in 1:6, j in 1:6]
    grid_curv = FlowGeometries.Grids.CurvilinearGrid(geom, xc, yc, trues(6, 6))
    npt = 36
    grid_node = FlowGeometries.Grids.UnstructuredGrid(
        geom, vec(xc), vec(yc), fill(4.0e6, npt), trues(npt), Int[], ones(Int, npt + 1),
    )
    un_u = rand(npt)

    PrecompileTools.@compile_workload begin
        out = zeros(6, 6)
        for grid in (grid_r, grid_v, grid_a)
            Filtering.filter_field!(out, u, grid, TopHatKernel(), 4000.0)
            Filtering.filter_field!(out, u, grid, GaussianKernel(), 4000.0)
            Π = zeros(6, 6)
            Diagnostics.compute_Π!(Π, u, v, nothing, grid, TopHatKernel(), 4000.0)
            Filtering.filter_fields!((out,), (u,), grid, GaussianKernel(), 4000.0)
        end
        out3 = zeros(6, 6, 6)
        Filtering.filter_field!(out3, u3, grid_3d, GaussianKernel(), 4000.0)
        Π3 = zeros(6, 6, 6)
        Diagnostics.compute_Π!(Π3, u3, v3, w3, grid_3d, TopHatKernel(), 4000.0)

        Filtering.filter_field!(out, u, grid_curv, GaussianKernel(), 4000.0)
        Diagnostics.compute_Π!(zeros(6, 6), u, v, nothing, grid_curv, GaussianKernel(), 4000.0)

        Filtering.filter_field!(
            zeros(npt), un_u, grid_node, GaussianKernel(), 4000.0; method = Filtering.RealSpace(),
        )
    end
end

end # module
