# ---------------------------------------------------------------------------
# Plan lifetimes
# ---------------------------------------------------------------------------
#
#   grid part  (grid, kernel family, mask strategy, method)  one per sweep
#   scale part (grid part, ℓ)                                one per scale
#   scratch    (grid, array rank, batch shape)               one per concurrent worker
#
# A footprint references its grid part and scratch and stores only its own ℓ-dependent fields.

"""
    AbstractGridPlan

Precomputation that depends on the grid, the kernel family, the mask strategy and the method, and is
independent of the filter scale. One instance serves every scale of a sweep; see
[`plan_filter_sweep`](@ref).
"""
abstract type AbstractGridPlan end

"""
    AbstractFilterScratch

Transient per-apply buffers, sized by the grid and the applied array's rank: the only part of a plan
`filter_apply!` mutates. Concurrent applies need one scratch each; the sequential scales of one sweep
share one.
"""
abstract type AbstractFilterScratch end
