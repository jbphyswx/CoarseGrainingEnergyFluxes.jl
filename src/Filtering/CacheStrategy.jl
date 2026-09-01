# ---------------------------------------------------------------------------
# Cache strategy: how much of the nonuniform-axis/curvilinear/ND real-space footprint to precompute
# and store, vs. recompute on the fly at apply time (singleton types — specializable, same idiom as
# AbstractMaskStrategy).
# ---------------------------------------------------------------------------

"""
    AbstractCacheStrategy

Whether a real-space footprint over a genuinely nonuniform axis (`StructuredGrid` with a `Vector`
axis, `CurvilinearGrid`, or ND with a non-`Range` axis) stores its full per-point neighbour list, or
recomputes it on the fly at apply time. The per-point neighbour/weight computation itself is always
the same either way (there is no shared translation-invariant table to exploit on a nonuniform axis,
unlike the `FilterFootprint` fast path) — this only controls whether that computation's RESULT is
kept in memory for reuse across separate future `filter_apply!` calls, or re-derived each time.
"""
abstract type AbstractCacheStrategy end

"""
    AutoCache <: AbstractCacheStrategy

Build and store the full per-point neighbour-list cache only if its estimated size is under
`cache_byte_budget` (default [`DEFAULT_CACHE_BYTE_BUDGET`](@ref)); otherwise fall back to recomputing
neighbours on the fly at apply time. This is the only cache-strategy knob most callers ever need —
it caches whenever doing so is affordable, which is strictly better than not caching whenever a
plan will be reused across more than one `filter_apply!` call.
"""
struct AutoCache <: AbstractCacheStrategy end

"""
    AlwaysCache <: AbstractCacheStrategy

Force building the full per-point neighbour-list cache regardless of `cache_byte_budget` — for a
caller who knows more memory is available than the conservative default budget assumes.
"""
struct AlwaysCache <: AbstractCacheStrategy end

"""
    NeverCache <: AbstractCacheStrategy

Force recomputing neighbours on the fly at every `filter_apply!`/`filter_apply_batch!` call, never
storing the cache — for a genuine, known memory ceiling `AutoCache`'s budget check doesn't already
account for (e.g. a GPU's device memory budget, or deliberately running many large plans
concurrently). Not a speed/memory preference: for any workflow that reuses a plan across more than
one call, caching is strictly better whenever it fits in memory, so this should be reached for only
when a specific external memory constraint is known, not by default.
"""
struct NeverCache <: AbstractCacheStrategy end

"""
    DEFAULT_CACHE_BYTE_BUDGET

Default byte budget for [`AutoCache`](@ref)'s size check on a real-space footprint cache (256 MiB).
"""
const DEFAULT_CACHE_BYTE_BUDGET = 256 * 1024^2
