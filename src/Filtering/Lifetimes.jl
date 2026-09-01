# ---------------------------------------------------------------------------
# Plan lifetimes: three things change at three different rates, so they are three objects
# ---------------------------------------------------------------------------
#
# A sweep filters K fields at each of S scales. Splitting an engine's state by WHEN it stops changing
# is what keeps each piece of work paid exactly once:
#
#   grid part  (grid, kernel family, mask strategy, method)  built ONCE per sweep
#   scale part (grid part, ℓ)                                built S times, cheaply
#   scratch    (grid, array rank, batch shape)               ONE per concurrent worker
#
# Engines express this by COMPOSITION rather than by taking three arguments: each footprint holds a
# reference to its shared grid part and scratch and stores only its own ℓ-dependent fields. Every
# apply signature is therefore unchanged, and a sweep shares the grid part and scratch by handing the
# same objects to each scale's footprint.

"""
    AbstractGridPlan

Precomputation determined by the grid, the kernel family, the mask strategy and the method — never by
the filter scale. One instance serves every scale of a sweep; see [`plan_filter_sweep`](@ref).
"""
abstract type AbstractGridPlan end

"""
    AbstractFilterScratch

Transient per-apply buffers, sized by the grid and the applied array's rank. This is the only part of
a plan mutated during `filter_apply!`, so **one scratch may not be shared between concurrent
applies**: a driver that runs applies in parallel (`filter_slices!`, the batch pipeline drivers,
Distributed/MPI ranks) must give each worker its own. Scales within one sweep run sequentially and
therefore share.
"""
abstract type AbstractFilterScratch end
