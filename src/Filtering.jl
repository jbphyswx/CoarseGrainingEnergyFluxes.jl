module Filtering

using FlowGeometries: FlowGeometries
using ..Kernels: Kernels
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends
using FlowTransformBindings: FlowTransformBindings
using StaticArrays: StaticArrays as SA

export AbstractMaskStrategy, ZeroFill, Deformable
export AbstractFilterMethod, RealSpace, Spectral, AutoMethod
export AbstractCacheStrategy, AutoCache, AlwaysCache, NeverCache
export filter_field!, filter_fields!, filter_slices!
export AbstractFilterPlan, plan_filter, filter_apply!, filter_apply_batch!
export AbstractGridPlan, AbstractFilterScratch, FilterPlanFamily, plan_filter_sweep

# Kernels that factor per axis, `G(x₁,…,x_N) = ∏ G(x_d)`, and so take the two-pass `O(N·Σwᵈ)` engine
# instead of the `O(N·∏wᵈ)` footprint one. A `Union` rather than an abstract supertype because
# `GaussianKernel` is separable *incidentally* (its radial form factors) while `HighOrderKernel` is
# separable *by definition* and has no radial form at all — they share no useful supertype, only this
# property. `Kernels.is_separable` is the trait; this is the dispatch handle.
const SeparableKernel = Union{Kernels.GaussianKernel, Kernels.HighOrderKernel}

include("Filtering/Strategies.jl")
include("Filtering/Lifetimes.jl")
include("Filtering/CacheStrategy.jl")
include("Filtering/Hooks.jl")
include("Filtering/Api.jl")
include("Filtering/engines/Exterior.jl")
include("Filtering/engines/Footprint.jl")
include("Filtering/engines/PrefixSumTopHat.jl")
include("Filtering/engines/Separable.jl")
include("Filtering/engines/NDim.jl")
include("Filtering/engines/PrefixSumTopHat3D.jl")
include("Filtering/Apply.jl")
include("Filtering/Plans.jl")
include("Filtering/Selection.jl")
include("Filtering/engines/NodeSpectral.jl")
include("Filtering/engines/NUFFTSpectral.jl")
include("Filtering/engines/FFTSpectral.jl")
include("Filtering/Slices.jl")

end # module
