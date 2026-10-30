# ---------------------------------------------------------------------------
# Worker copies: one set of tables, one scratch per concurrent worker
# ---------------------------------------------------------------------------
#
# A plan's tables (weights, spectra, window masses, neighbour lists, transfers) are read-only once it is
# built. Two parts are written by every apply: its scratch (`_scratch`), and, for a nonuniform transform,
# the library plan of its grid plan (`_grid_plan`), which holds the working state of its execution. A
# worker copy shares the tables and holds its own of the two.

"""
    worker_copy(plan) -> plan
    worker_copy(family::FilterPlanFamily) -> FilterPlanFamily

A plan that applies as `plan` does, sharing its read-only tables and holding its own scratch
([`AbstractFilterScratch`](@ref)), so that the two may apply concurrently. Each concurrent worker of a batch
takes one: `W` workers hold one set of tables and `W` scratch sets. A family's copy shares its grid plan and
the tables of every scale, gives the plans that shared one scratch one new scratch to share, and copies its
held [`scale_rate`](@ref)s alike. A nonuniform transform's library plan holds the state of its execution and
is planned anew over the same nodes. A plan whose applies run on a device or across processes raises.
"""
function worker_copy(plan::AbstractFilterPlan)
    _check_worker_backend(plan_execution(plan))
    return _worker_plan(plan)
end

function worker_copy(f::FilterPlanFamily)
    isempty(f.plans) || _check_worker_backend(plan_execution(first(f.plans)))
    gp = _fresh_grid_plan(f.grid_plan)
    plans = similar(f.plans)
    for i in eachindex(f.plans)
        p = f.plans[i]
        q = _grid_plan(p) === f.grid_plan ? _with_grid_plan(p, gp) : _with_grid_plan(p, _fresh_grid_plan(_grid_plan(p)))
        j = _first_sharing(f.plans, i)
        plans[i] = _with_scratch(q, j == i ? _fresh(_scratch(p)) : _scratch(plans[j]))
    end
    return FilterPlanFamily(gp, plans, _matching_scratch(f.scratch, f.plans, plans), _worker_rates(f.rates, f.plans, plans),
                            f.rate_order)
end

@noinline _check_worker_backend(b::ComputationalBackends.AbstractExecutionBackend) = _host_backend(b) || throw(ArgumentError(
    "worker_copy serves plans whose applies run in this process's loops (SerialBackend, ThreadedBackend); a " *
    "$(nameof(typeof(b))) plan holds its buffers on its device or its worker processes"))

_worker_plan(plan) = _with_scratch(_with_grid_plan(plan, _fresh_grid_plan(_grid_plan(plan))), _fresh(_scratch(plan)))

"""
    _scratch(plan_or_footprint) -> scratch or nothing
    _with_scratch(plan_or_footprint, scratch) -> plan_or_footprint
    _fresh(scratch) -> scratch
    _alike(a, b) -> Bool

The buffers an apply writes, as one value whose identity (`===`) is the buffers', `nothing` for an apply writing
none; the plan rebound to other buffers; new buffers of the same shapes, zero where an apply reads zero padding;
and whether two such values serve the same plans, one type and the same shapes, so that the sequential scales of
one family may share one.
"""
_scratch(_) = nothing
_with_scratch(x, ::Nothing) = x
_fresh(::Nothing) = nothing
_alike(_, _) = false

"""
    _grid_plan(plan_or_footprint) -> grid plan or nothing
    _with_grid_plan(plan_or_footprint, grid_plan) -> plan_or_footprint
    _fresh_grid_plan(grid_plan) -> grid plan

A plan's grid plan where it holds state an apply writes, `nothing` otherwise; the plan rebound to another; and a grid
plan sharing the tables of `grid_plan` with its own library plans. A grid plan holding no such state is its own.
"""
_grid_plan(_) = nothing
_with_grid_plan(x, ::Nothing) = x
_fresh_grid_plan(gp) = gp

_scratch(p::PhysicalFilterPlan) = _scratch(p.footprint)
_with_scratch(p::PhysicalFilterPlan, sc) = _with_footprint(p, _with_scratch(p.footprint, sc))
_with_scratch(p::PhysicalFilterPlan, ::Nothing) = p
_grid_plan(p::PhysicalFilterPlan) = _grid_plan(p.footprint)
_with_grid_plan(p::PhysicalFilterPlan, gp) = _with_footprint(p, _with_grid_plan(p.footprint, gp))
_with_grid_plan(p::PhysicalFilterPlan, ::Nothing) = p

_with_footprint(p::PhysicalFilterPlan, fp) = PhysicalFilterPlan(fp, p.grid, p.strategy, p.kernel, p.scale, p.execution)

# The first plan up to `i` whose scratch is plan `i`'s or alike.
function _first_sharing(plans::AbstractVector, i::Int)
    s = _scratch(plans[i])
    for j in firstindex(plans):i
        _shares(_scratch(plans[j]), s) && return j
    end
    return i
end

_shares(a, b) = a === b || _alike(a, b)

# The family's scratch in the copy: the one its plans that held the original now share.
function _matching_scratch(sc, old::AbstractVector, new::AbstractVector)
    sc === nothing && return nothing
    for j in eachindex(old)
        _scratch(old[j]) === sc && return _scratch(new[j])
    end
    return _fresh(sc)
end

"""
    _share_scratch(plans) -> plans

`plans` with each one's scratch replaced by the first earlier one's that is [`_alike`](@ref), so the sequential scales
of one family hold one scratch per kind of buffers.
"""
function _share_scratch(plans::AbstractVector)
    out = similar(plans)
    for i in eachindex(plans)
        j = _first_sharing(plans, i)
        out[i] = j == i ? plans[i] : _with_scratch(plans[i], _scratch(out[j]))
    end
    return out
end

# A family's held rates in its copy: each rate plan rebound to the scratch its scale plan's copy holds where it
# shared the scale plan's, a worker copy of its own otherwise.
_worker_rates(::Nothing, _old, _new) = nothing
_worker_rates(rates::AbstractVector, old::AbstractVector, new::AbstractVector) = map(r -> _worker_rate(r, old, new), rates)

_worker_rate(r::WindowRate, old, new) =
    WindowRate(map(p -> _worker_member(p, old, new), r.plans), r.ρ, r.B, _worker_rate(r.second, old, new))
_worker_rate(r::WindowSecond, old, new) = WindowSecond(map(p -> _worker_member(p, old, new), r.plans), r.ρ, r.B)
_worker_rate(::Nothing, _old, _new) = nothing

function _worker_rate(r::SumRate, old, new)
    plan = _matching_plan(r.plan, old, new)
    return SumRate(plan, _on_parts_of(r.rate, plan), r.B, map(similar, r.F), r.scale, _worker_second(r.second, plan))
end
_worker_second(r::SumSecond, plan) = SumSecond(_on_parts_of(r.rate, plan), r.B)
_worker_second(::Nothing, _plan) = nothing

_worker_member(p, old, new) = _worker_member(p, _scratch(p), old, new)
_worker_member(p, ::Nothing, _old, _new) = p
function _worker_member(p, s, old, new)
    for j in eachindex(old)
        _scratch(old[j]) === s && return _with_scratch(p, _scratch(new[j]))
    end
    return _worker_plan(p)
end

function _matching_plan(p, old, new)
    for j in eachindex(old)
        old[j] === p && return new[j]
    end
    return _worker_plan(p)
end

# `r` on the grid plan and scratch `plan` holds, which it shared with `plan`'s original.
_on_parts_of(r, plan) = _with_scratch(_with_grid_plan(r, _grid_plan(plan)), _scratch(plan))

# ---------------------------------------------------------------------------
# Scratch of the engines whose footprints hold their buffers as fields of their own
# ---------------------------------------------------------------------------

"""
    ApplyBuffers(buffers)

The arrays a footprint's apply writes where the footprint holds them as fields of its own, as one scratch value; a
`nothing` entry stands for a buffer the footprint does not hold.
"""
struct ApplyBuffers{B<:Tuple} <: AbstractFilterScratch
    buffers::B
end

_fresh(s::ApplyBuffers) = ApplyBuffers(map(_fresh_buffer, s.buffers))
_alike(a::S, b::S) where {S<:ApplyBuffers} = map(_extent, a.buffers) == map(_extent, b.buffers)

# A new buffer of the same shape, zero wherever an apply reads it before writing; a vector of per-field slots
# starts empty and grows as its plan's batches need.
_fresh_buffer(::Nothing) = nothing
_fresh_buffer(a::AbstractArray{<:Number}) = zero(a)
_fresh_buffer(v::AbstractVector{<:Tuple}) = empty(v)
_extent(::Nothing) = nothing
_extent(a::AbstractArray) = size(a)

_scratch(fp::SeparableFootprint) = SeparableScratch(fp.masked_input, fp.row_pass)
_with_scratch(fp::SeparableFootprint, s::SeparableScratch) = SeparableFootprint(
    fp.gx, fp.gy, fp.di_lim, fp.dj_lim, fp.periodic_x, fp.periodic_y, fp.Nx_profile, fp.Ny_profile, fp.invrenorm,
    fp.strategy, fp.masked, fp.bound, s.masked_input, s.row_pass,
)
_scratch(fp::SeparableFootprintND) = SeparableScratch(fp.masked_input, fp.scratch)
_with_scratch(fp::SeparableFootprintND, s::SeparableScratch) = SeparableFootprintND(
    fp.g, fp.lim, fp.periodic, fp.profiles, fp.invrenorm, fp.strategy, fp.masked, fp.bound, s.masked_input, s.row_pass,
)
_fresh(s::SeparableScratch) = SeparableScratch(zero(s.masked_input), zero(s.row_pass))
_alike(a::S, b::S) where {S<:SeparableScratch} =
    size(a.masked_input) == size(b.masked_input) && size(a.row_pass) == size(b.row_pass)

_scratch(fp::FilterFootprintND) = ApplyBuffers((fp.masked_input,))
_with_scratch(fp::FilterFootprintND, s::ApplyBuffers) = FilterFootprintND(
    fp.offsets, fp.w, fp.runs, fp.periodic, fp.strategy, fp.masked, fp.bound, fp.invden, s.buffers[1],
)

_scratch(fp::NodeFilterPlan) = ApplyBuffers((fp.vals, fp.act))
_with_scratch(fp::NodeFilterPlan, s::ApplyBuffers) = NodeFilterPlan(fp.nbrs, fp.w, fp.ptr, fp.perm, s.buffers...)

_scratch(fp::BinnedTopHatPlan) = ApplyBuffers((fp.buffers,))
_with_scratch(fp::BinnedTopHatPlan, s::ApplyBuffers) = BinnedTopHatPlan(
    fp.points, fp.rad, fp.lines, fp.win, fp.wout, s.buffers[1], fp.invden, fp.strategy, fp.bound, fp.edge,
)

# ---------------------------------------------------------------------------
# Engines holding a scratch of their own type
# ---------------------------------------------------------------------------

_scratch(fp::PrefixSumTopHatPlan) = fp.scratch
_with_scratch(fp::PrefixSumTopHatPlan, sc::PrefixSumScratch) = PrefixSumTopHatPlan(
    fp.grid_plan, sc, fp.rad, fp.dj_lim, fp.uniform_axis, fp.bound, fp.invden, fp.hw, fp.wcell, fp.edge,
)
# Slot 1 of each kind, zero: a scan's first row is the empty prefix, which the fill never writes. A batch grows more.
_fresh(sc::PrefixSumScratch) = PrefixSumScratch([_fresh_scan(first(sc.prefix_nums))], _fresh_slots(sc.cell_values))
_alike(a::S, b::S) where {S<:PrefixSumScratch} = size(first(a.prefix_nums).hi) == size(first(b.prefix_nums).hi)
_fresh_scan(P::CompensatedScan) = CompensatedScan(zero(P.hi), zero(P.lo))
_fresh_slots(::Nothing) = nothing
_fresh_slots(v::AbstractVector) = isempty(v) ? empty(v) : [zero(first(v))]

_scratch(fp::PrefixSumTopHat3DPlan) = fp.scratch
_with_scratch(fp::PrefixSumTopHat3DPlan, sc::PrefixSum3DScratch) = PrefixSumTopHat3DPlan(
    sc, fp.rad, fp.dj_lim, fp.dk_lim, fp.periodic, fp.masked, fp.bound, fp.strategy, fp.wcell, fp.invden, fp.edge,
)
_fresh(sc::PrefixSum3DScratch) = PrefixSum3DScratch(_fresh_scan(sc.prefix), _fresh_buffer(sc.cell_values))
_alike(a::S, b::S) where {S<:PrefixSum3DScratch} = size(a.prefix.hi) == size(b.prefix.hi)

_scratch(fp::FilterFootprint) = fp.scratch
_with_scratch(fp::FilterFootprint, sc::BandedScratch) = FilterFootprint(
    fp.taps, fp.nbands, fp.strategy, fp.masked, fp.bound, fp.periodic_x, fp.periodic_y, fp.invden, sc,
)
_fresh(sc::BandedScratch) = BandedScratch(_fresh_slots(sc.masked_inputs))
_alike(a::S, b::S) where {S<:BandedScratch} = _first_extent(a.masked_inputs) == _first_extent(b.masked_inputs)
_first_extent(v::AbstractVector) = isempty(v) ? nothing : size(first(v))

# A nonuniform-FFT kernel sum: its scratch, and its grid plan's library plans, which hold their execution's state.
_scratch(fp::NUFFTSumFootprint) = fp.scratch
_with_scratch(fp::NUFFTSumFootprint, sc::NUFFTScratch) =
    NUFFTSumFootprint(fp.grid_plan, fp.transfer, fp.invden, fp.dden, fp.strategy, fp.bound, sc)
_grid_plan(fp::NUFFTSumFootprint) = fp.grid_plan
_with_grid_plan(fp::NUFFTSumFootprint, gp::NUFFTGridPlan) =
    NUFFTSumFootprint(gp, fp.transfer, fp.invden, fp.dden, fp.strategy, fp.bound, fp.scratch)

_scratch(plan::NUFFTFilterPlan) = plan.scratch
_with_scratch(plan::NUFFTFilterPlan, sc::NUFFTScratch) =
    NUFFTFilterPlan(plan.grid_plan, plan.transfer, plan.invrenorm, plan.dmass, plan.strategy, sc)
_grid_plan(plan::NUFFTFilterPlan) = plan.grid_plan
_with_grid_plan(plan::NUFFTFilterPlan, gp::NUFFTGridPlan) =
    NUFFTFilterPlan(gp, plan.transfer, plan.invrenorm, plan.dmass, plan.strategy, plan.scratch)

_fresh(sc::NUFFTScratch) = NUFFTScratch(similar(sc.values), similar(sc.modes), _fresh_pair(sc.batched))
_alike(a::S, b::S) where {S<:NUFFTScratch} = size(a.modes) == size(b.modes) && _extent_pair(a.batched) == _extent_pair(b.batched)
_fresh_pair(::Nothing) = nothing
_fresh_pair(b::NamedTuple) = map(similar, b)
_extent_pair(::Nothing) = nothing
_extent_pair(b::NamedTuple) = map(size, b)

function _fresh_grid_plan(gp::NUFFTGridPlan)
    plan = FlowTransformBindings.task_local_plan(gp.plan)::typeof(gp.plan)
    return NUFFTGridPlan(plan, _task_local(gp.batched), gp.nb, gp.period, gp.counts, gp.weights, gp.mask, gp.topology,
                         gp.reach, gp.execution, gp.own, gp.npts, gp.ik)
end
_task_local(::Nothing) = nothing
_task_local(p) = FlowTransformBindings.task_local_plan(p)::typeof(p)

# A node set's plan: the inner plan's parts and the node buffers.
_scratch(plan::NodeSetPlan) = plan.scratch
_with_scratch(plan::NodeSetPlan, sc::NodeSetScratch) = NodeSetPlan(_with_scratch(plan.inner, sc.inner), plan.grid_plan, sc)
_grid_plan(plan::NodeSetPlan) = plan.grid_plan
_with_grid_plan(plan::NodeSetPlan, gp::NodeSetGridPlan) = NodeSetPlan(_with_grid_plan(plan.inner, gp.inner), gp, plan.scratch)
_fresh(sc::NodeSetScratch) =
    NodeSetScratch(zero(sc.field), zero(sc.out), _zero_pair(sc.batched), _fresh(sc.inner))
_zero_pair(::Nothing) = nothing
_zero_pair(b::NamedTuple) = map(zero, b)
_alike(a::S, b::S) where {S<:NodeSetScratch} = size(a.field) == size(b.field) && _alike(a.inner, b.inner)
_fresh_grid_plan(gp::NodeSetGridPlan) =
    NodeSetGridPlan(gp.nodes, _fresh_grid_plan(gp.inner), gp.ncells, gp.dims, gp.batch, gp.execution)

# ---------------------------------------------------------------------------
# Spin transforms of the generalized convolution
# ---------------------------------------------------------------------------

"""
    _worker_transforms(tr) -> AbstractSpinTransforms

The spin transforms a worker holds: `tr`'s node tables and longitude step shared, the arrays a transform writes its
own; a library's transforms, whose plans hold their execution's buffers, planned anew.
"""
function _worker_transforms end

_worker_transforms(tr::DirectSpinTransforms) = DirectSpinTransforms(
    tr.x, tr.xl, tr.u, tr.pos, tr.lc, tr.ls, tr.φ, tr.w, tr.norm, tr.starts, similar(tr.coef), similar(tr.transfer),
    tr.lmax, tr.execution,
)

_worker_transforms(tr::RingSpinTransforms) = RingSpinTransforms(
    tr.x, tr.xl, tr.u, tr.pos, tr.lc, tr.ls, tr.cs, tr.w, zero(tr.ring), zero(tr.wf), tr.lon, tr.norm, tr.starts,
    tr.ratio, similar(tr.coef), similar(tr.y), similar(tr.e), similar(tr.k), similar(tr.ar), similar(tr.ai),
    similar(tr.transfer), tr.lmax, tr.execution,
)

"""
    _worker_fit(fit) -> SpinFit or nothing

A worker's least-squares analysis: the weights and the unit transfer shared, the scratch and LSMR workspace (arrays
alone) its own, its outcome reset.
"""
_worker_fit(::Nothing) = nothing
_worker_fit(f::SpinFit{T}) where {T} = SpinFit(
    f.sqrtw, zero(f.t), zero(f.work), f.unit, deepcopy(f.ws), f.rtol, f.maxiter, Base.RefValue(_exact_status(T)),
)
