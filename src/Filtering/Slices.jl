# ---------------------------------------------------------------------------
# Slice-parallel apply: many independent problems, one plan each
# ---------------------------------------------------------------------------

"""
    filter_slices!(outs, fields, plans; backend = AutoBackend()) -> outs

Apply `plans[t]` to `fields[t]`, writing `outs[t]`, over a collection of **independent** slices.

This is a different parallel axis from [`filter_apply_batch!`](@ref), which shares one grid across
several fields: here each slice has its own grid, plan and point count, and slices share nothing, so
there is no synchronization at all. Where a workload has many slices, this is the outermost
race-free axis and the one that converts thread count into throughput — threading *within* one slice
saturates once the slice is small enough that per-task overhead dominates its work.

Each slice runs **serially inside**, whatever backend its own plan carries: nesting a threaded apply
under a threaded slice loop would have both levels claim the whole thread pool.
"""
function filter_slices!(
    outs, fields, plans;
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
)
    length(outs) == length(fields) == length(plans) || throw(DimensionMismatch(
        "filter_slices! got $(length(outs)) outputs, $(length(fields)) fields and $(length(plans)) plans",
    ))
    resolved = _resolve_slice_backend(backend)
    resolved isa ComputationalBackends.ThreadedBackend &&
        return threaded_filter_slices!(outs, fields, plans)
    for t in eachindex(plans)
        apply_slice_serial!(outs[t], fields[t], plans[t])
    end
    return outs
end

# Slices are independent for every grid architecture, so — unlike `_resolve_backend` — this needs no
# grid capability check: the parallelism is over the collection, not inside any one slice.
@inline function _resolve_slice_backend(backend::ComputationalBackends.AbstractExecutionBackend)
    backend isa ComputationalBackends.AbstractAutoBackend ||
        return ComputationalBackends.resolve_backend(backend)
    return (Threads.nthreads() > 1 && _threading_available()) ?
        ComputationalBackends.ThreadedBackend() : ComputationalBackends.SerialBackend()
end

"""
    apply_slice_serial!(out, field, plan) -> out

One slice, forced down the serial engine regardless of the backend recorded in `plan`. The slice
loop owns the parallelism; see [`filter_slices!`](@ref).
"""
@inline apply_slice_serial!(out, field, plan::PhysicalFilterPlan) =
    _apply_serial!(out, field, plan.grid, plan.footprint, plan.strategy)
# A spectral plan has no separate serial engine — its transform is already the whole apply.
@inline apply_slice_serial!(out, field, plan::AbstractFilterPlan) = filter_apply!(out, field, plan)

"""
    slice_costs(plans) -> Vector{Int}

Per-slice work proxy: the number of target points each plan writes. A slice's cost grows at least
linearly in this, so it is what a longest-first schedule should sort on.
"""
slice_costs(plans) = [_plan_npoints(p) for p in plans]

@inline _plan_npoints(p::PhysicalFilterPlan) = prod(FlowGeometries.Grids.size_tuple(p.grid))
@inline _plan_npoints(p::AbstractFilterPlan) = 1
