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
    AutoCache <: AbstractCacheStrategy

Store the per-point neighbour-list cache when it fits in `cache_byte_budget` (default
[`default_cache_byte_budget`](@ref)), and recompute neighbours at apply time otherwise. The size is the
exact entry count, counted by the build's ball query before anything is allocated; counting stops once
it passes the budget.
"""
struct AutoCache <: AbstractCacheStrategy end

"""
    AlwaysCache <: AbstractCacheStrategy

Store the per-point neighbour-list cache whatever its size; `cache_byte_budget` is ignored.
"""
struct AlwaysCache <: AbstractCacheStrategy end

"""
    NeverCache <: AbstractCacheStrategy

Recompute neighbours at every `filter_apply!`/`filter_apply_batch!` call and store no cache, for a
memory ceiling `cache_byte_budget` does not cover, such as a GPU's device memory.
"""
struct NeverCache <: AbstractCacheStrategy end

"""
    default_cache_byte_budget() -> Int

[`AutoCache`](@ref)'s default byte budget: a quarter of the memory available to the process
(`Sys.total_memory()`, which honours a cgroup limit), shared by every scale of a sweep.
"""
default_cache_byte_budget() = Int(Sys.total_memory() ÷ 4)

"""
    _cache_keep(strategy, budget, entry_bytes) -> (n -> Bool)

Whether a neighbour cache of `n` entries is kept under `strategy`: always, never, or while it fits the
byte budget.
"""
_cache_keep(::AlwaysCache, _, _) = Returns(true)
_cache_keep(::NeverCache, _, _) = Returns(false)
_cache_keep(::AutoCache, budget::Integer, entry_bytes::Integer) = n -> n <= Int(budget) ÷ entry_bytes
