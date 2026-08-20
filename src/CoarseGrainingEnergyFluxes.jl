module CoarseGrainingEnergyFluxes

using PrecompileTools: PrecompileTools
using ComputationalBackends: ComputationalBackends
using FlowGeometries: FlowGeometries
using SpectralBackends: SpectralBackends

# Submodules inclusion
include("Kernels.jl")
include("Filtering.jl")
include("Derivatives.jl")
include("Diagnostics.jl")
include("Pipeline.jl")
include("Visualization.jl")

# Bind each submodule's own name only (never `using X: specific_function`) — internal code and
# extensions reach everything through `Submodule.func(...)`, never a flattened top-level re-export.
using .Kernels: Kernels
using .Filtering: Filtering
using .Derivatives: Derivatives
using .Diagnostics: Diagnostics
using .Pipeline: Pipeline
using .Visualization: Visualization

# Top-level exports are the headline entry points only. Everything else — backends, mask strategies,
# `ddx!`/`ddy!`/`ddz!`, `compute_Π!`, `plan_filter`, `ΠWorkspace`, `spectral_transfer`, … — is reached
# qualified (`CoarseGrainingEnergyFluxes.Diagnostics.compute_Π!(...)`). Each additional export is a
# name-clash risk in the user's namespace.
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

if isdefined(Base.Experimental, :register_error_hint)
    function __init__()
        Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, kwargs
            if exc.f === Kernels.spectral_transfer && length(argtypes) >= 1 && argtypes[1] === Kernels.TopHatKernel
                print(io, "\nSpectral filtering with TopHatKernel needs its exact planar transfer " *
                    "function 2*J₁(kR)/(kR), provided by the SpecialFunctions weak dependency. Run " *
                    "`using SpecialFunctions` to enable it.")
            end
        end
    end
end

# Precompile workload to minimize Time To First Execution (TTFX)
PrecompileTools.@setup_workload begin
    geom = FlowGeometries.Geometry.CartesianGeometry()
    # Both axis representations: a Range is a compile-time proof of constant spacing and selects the
    # separable footprints, a Vector selects the position-indexed ones. They are different code, so
    # precompiling one does not cover the other.
    xr = 0.0:2000.0:10000.0
    xv = collect(xr)
    # The mask representation is a second axis of specialization: an unmasked grid stores `AllActive`,
    # whose `isactive` folds to a constant, where a `BitArray` is a load and a branch. Different code
    # again, and the unmasked one is what a caller who never mentions a mask gets.
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
    # Curvilinear and unstructured are each their own footprint representation. Neither needs a weakdep
    # here: the node grid is given its measure and (unused) adjacency explicitly, and its real-space
    # engine reads neither a k-d tree nor a tessellation. Only the spectral methods stay uncovered,
    # their backends being weakdeps not loaded while this package precompiles.
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
        # True 3D: a distinct footprint representation (point-indexed, not row-indexed), so it shares
        # almost no specializations with the 2D path above.
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
