module CoarseGrainingEnergyFluxesTests

using Test: Test
using Logging: Logging  # `@test_logs min_level = Logging.Warn`, for asserting the ABSENCE of a warning
using StaticArrays: StaticArrays as SA
using Aqua: Aqua
using ExplicitImports: ExplicitImports as EI
using JET: JET
using FFTW: FFTW  # triggers the spectral-filtering extension
using FINUFFT: FINUFFT  # triggers the scattered-Cartesian spectral extension
using FastSphericalHarmonics: FastSphericalHarmonics as FSH  # triggers the uniform-spherical spectral extension
using NUFSHT: NUFSHT  # triggers the scattered-spherical spectral extension
using SpecialFunctions: SpecialFunctions  # triggers the TopHatKernel spectral-transfer extension
using OhMyThreads: OhMyThreads  # triggers the threaded-backend extension
using Distributed: Distributed  # with SharedArrays, triggers the distributed-backend extension
using SharedArrays: SharedArrays
using MPI: MPI  # triggers the MPI-backend extension; real multi-rank execution is
                # test/mpi_runtests.jl, run via `mpiexec`, not this single-process suite
MPI.Init()  # required before ANY MPI routine runs (only MPI.Initialized/Finalized are safe pre-init) —
            # single-rank here, so MPIBackend degenerates to "this rank owns every row," exercised
            # below in the "Backends" testset the same way DistributedBackend already is.
using KernelAbstractions: KernelAbstractions as KA  # triggers the GPU backend extension (CPU device here)
# These three trigger FlowGeometries' own extensions, which is where grid construction needs them:
# k-d-tree adjacency for an UnstructuredGrid, and the Cartesian/spherical Voronoi cell areas.
using NearestNeighbors: NearestNeighbors
using DelaunayTriangulation: DelaunayTriangulation
using Quickhull: Quickhull
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF
using FlowGeometries: FlowGeometries as FG

# Each file owns one topic and opens its own top-level testset, so a single one can be run on its
# own — `include("test/test_filtering.jl")` into a session that has already loaded the preamble —
# instead of the whole suite for every change.

include("test_quality.jl")
include("test_geometry.jl")
include("test_kernels.jl")
include("test_highorder.jl")
include("test_grids.jl")
include("test_filtering.jl")
include("test_spectral.jl")
include("test_backends.jl")
include("test_periodic.jl")
include("test_derivatives.jl")
include("test_curvilinear.jl")
include("test_unstructured.jl")
include("test_nd.jl")
include("test_diagnostics.jl")
include("test_validation.jl")
include("test_visualization.jl")
include("test_allocs.jl")

end # module
