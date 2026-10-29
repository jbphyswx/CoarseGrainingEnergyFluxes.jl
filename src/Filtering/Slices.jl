# ---------------------------------------------------------------------------
# Slice-parallel apply: many independent problems, one plan each
# ---------------------------------------------------------------------------

"""
    filter_slices!(outs, fields, plans; execution = AutoBackend()) -> outs

Apply `plans[t]` to `fields[t]`, writing `outs[t]`, for every slice `t`. Slices are independent, each
with its own grid and plan; for several fields on one grid, see [`filter_apply_batch!`](@ref).

`execution` parallelizes over the slices. Each slice runs serially inside, whatever execution backend
its own plan carries.
"""
function filter_slices!(
    outs, fields, plans;
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
)
    length(outs) == length(fields) == length(plans) || throw(DimensionMismatch(
        "filter_slices! got $(length(outs)) outputs, $(length(fields)) fields and $(length(plans)) plans",
    ))
    resolved = _resolve_slice_backend(execution)
    resolved isa ComputationalBackends.ThreadedBackend &&
        return threaded_filter_slices!(outs, fields, plans)
    for t in eachindex(plans)
        apply_slice_serial!(outs[t], fields[t], plans[t])
    end
    return outs
end

# Slices are independent on every grid, so `AutoBackend` resolves with no grid capability check.
@inline function _resolve_slice_backend(backend::ComputationalBackends.AbstractExecutionBackend)
    backend isa ComputationalBackends.AbstractAutoBackend ||
        return ComputationalBackends.resolve_backend(backend)
    return (Threads.nthreads() > 1 && _available(_Threads())) ?
        ComputationalBackends.ThreadedBackend() : ComputationalBackends.SerialBackend()
end

"""
    apply_slice_serial!(out, field, plan) -> out

Apply `plan` to one slice on the serial engine, whatever backend `plan` records; see
[`filter_slices!`](@ref).
"""
@inline apply_slice_serial!(out, field, plan::PhysicalFilterPlan) =
    _apply_serial!(out, field, plan.grid, plan.footprint, plan.strategy)
# A spectral plan has one engine, its transform, run at the thread count the plan was built for.
@inline apply_slice_serial!(out, field, plan::AbstractFilterPlan) = filter_apply!(out, field, plan)

"""
    slice_costs(plans) -> Vector{Int}

Per-slice work proxy for a longest-first schedule: the number of target points a physical-space plan
writes, and 1 for any other plan. A slice's cost grows at least linearly in it.
"""
slice_costs(plans) = [_plan_npoints(p) for p in plans]

@inline _plan_npoints(p::PhysicalFilterPlan) = prod(FlowGeometries.Grids.size_tuple(p.grid))
@inline _plan_npoints(p::AbstractFilterPlan) = 1
