module Filtering

using FlowGeometries: FlowGeometries
using ..Kernels: Kernels
using ..Geometry: Geometry
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends
using FlowTransformBindings: FlowTransformBindings
using LinearAlgebra: LinearAlgebra
using StaticArrays: StaticArrays as SA

export AbstractMaskStrategy, ZeroFill, Deformable
export AbstractFilterMethod, RealSpace, Spectral
export AbstractCacheStrategy, AutoCache, AlwaysCache, NeverCache
export filter_field!, filter_fields!, filter_slices!
export AbstractFilterPlan, plan_filter, filter_apply!, filter_apply_batch!, analysis_status
export AbstractGridPlan, AbstractFilterScratch, FilterPlanFamily, plan_filter_sweep
export AbstractVectorFilter, GeneralizedConvolution, CartesianComponents, resolve_vector_filter
export AbstractAnalysis, AutoAnalysis, Quadrature, LeastSquares

# The kernels with `Kernels.is_separable`, `G(x₁,…,x_N) = ∏ G(x_d)`, filtered by one 1-D pass per axis.
const SeparableKernel = Union{Kernels.GaussianKernel, Kernels.HighOrderKernel, Kernels.AxisRate}

include("Filtering/Strategies.jl")
include("Filtering/VectorFilter.jl")
include("Filtering/SpinDirect.jl")
include("Filtering/SpinRings.jl")
include("Filtering/Lifetimes.jl")
include("Filtering/CacheStrategy.jl")
include("Filtering/Hooks.jl")
include("Filtering/Api.jl")
include("Filtering/engines/Exterior.jl")
include("Filtering/engines/HighOrder.jl")
include("Filtering/engines/Overlap.jl")
include("Filtering/engines/Footprint.jl")
include("Filtering/engines/Banded.jl")
include("Filtering/engines/PrefixSumTopHat.jl")
include("Filtering/engines/Separable.jl")
include("Filtering/engines/NDim.jl")
include("Filtering/engines/PrefixSumTopHat3D.jl")
include("Filtering/engines/ScatteredND.jl")
include("Filtering/engines/BinnedTopHat.jl")
include("Filtering/engines/Nodes.jl")
include("Filtering/Apply.jl")
include("Filtering/Shares.jl")
include("Filtering/Plans.jl")
include("Filtering/SpinFit.jl")
include("Filtering/Selection.jl")
include("Filtering/Rates.jl")
include("Filtering/engines/NodeSpectral.jl")
include("Filtering/engines/NUFFTSpectral.jl")
include("Filtering/engines/FFTSpectral.jl")
include("Filtering/Slices.jl")

end # module
