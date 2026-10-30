# ---------------------------------------------------------------------------
# Cache strategy for the neighbour lists of nonuniform real-space footprints
# ---------------------------------------------------------------------------

"""
    AbstractCacheStrategy

Whether a real-space footprint over a nonuniform axis (`StructuredGrid` with a `Vector` axis,
`CurvilinearGrid`, or ND with a non-`Range` axis) stores its per-point neighbour list for reuse across
`filter_apply!` calls, or recomputes it at every apply. Both give the same neighbours and weights.
"""
abstract type AbstractCacheStrategy end

"""
    AlwaysCache <: AbstractCacheStrategy

Store the per-point neighbour list, the default: built in one pass of the ball query, it makes every apply a pass
over the list. It holds each target's neighbours, an index and a weight per neighbour.
"""
struct AlwaysCache <: AbstractCacheStrategy end

"""
    NeverCache <: AbstractCacheStrategy

Recompute neighbours at every `filter_apply!`/`filter_apply_batch!` call and store no list: each apply runs the
ball query the list is built from, in no memory beyond the plan's.
"""
struct NeverCache <: AbstractCacheStrategy end

"""
    _cache_keep(strategy) -> Bool

Whether a footprint built under `strategy` keeps its neighbour list.
"""
_cache_keep(::AlwaysCache) = true
_cache_keep(::NeverCache) = false
