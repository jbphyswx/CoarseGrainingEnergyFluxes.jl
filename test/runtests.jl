module CoarseGrainingEnergyFluxesTests

using Test: Test
# Before FastSphericalHarmonics: CGEF loads FlowTransformBindings, which selects the OpenMP runtime's
# thread-local mode before FastTransforms loads that runtime.
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF
using FlowTransformBindings: FlowTransformBindings as FTB
using Logging: Logging
using Aqua: Aqua
using ExplicitImports: ExplicitImports as EI
using JET: JET
using FFTW: FFTW
using FINUFFT: FINUFFT
using NonuniformFFTs: NonuniformFFTs
using FastSphericalHarmonics: FastSphericalHarmonics as FSH
using NUFSHT: NUFSHT
using SpecialFunctions: SpecialFunctions
using OhMyThreads: OhMyThreads
using Distributed: Distributed
using SharedArrays: SharedArrays
using MPI: MPI
# One rank here; test/mpi_runtests.jl runs several under `mpiexec`.
MPI.Init()
using KernelAbstractions: KernelAbstractions as KA
using JLArrays: JLArrays
using GPUArraysCore: GPUArraysCore
using Adapt: Adapt
# FlowGeometries' extensions for k-d-tree adjacency and Voronoi cell areas.
using NearestNeighbors: NearestNeighbors
using DelaunayTriangulation: DelaunayTriangulation
using Quickhull: Quickhull
using FlowGeometries: FlowGeometries as FG

include("reference_filters.jl")

include("test_quality.jl")
include("test_kernels.jl")
include("test_filtering.jl")
include("test_spectral.jl")
include("test_diagnostics.jl")
include("test_batch.jl")
include("test_device_genericity.jl")
include("test_allocs.jl")

end # module
