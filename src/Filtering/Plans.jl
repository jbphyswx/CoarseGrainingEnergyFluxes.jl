# ---------------------------------------------------------------------------
# Reusable filter plans
# ---------------------------------------------------------------------------

"""
    PhysicalFilterPlan

A physical-space filter plan: the precomputed footprint, with the grid, mask strategy, kernel, scale
and resolved execution backend it was built for. `kernel` and `scale` are passed to the backend hooks,
which take them positionally; the footprint is never rebuilt from them.
"""
struct PhysicalFilterPlan{FP, G<:FlowGeometries.Grids.AbstractGrid, S<:AbstractMaskStrategy, K<:Kernels.AbstractFilterKernel, T<:AbstractFloat, B<:ComputationalBackends.AbstractExecutionBackend} <: AbstractFilterPlan
    footprint::FP   # the engine's footprint, as `prepare_workspace` returns it for `execution`
    grid::G
    strategy::S
    kernel::K
    scale::T
    execution::B
end

"""
    plan_strategy(plan) -> AbstractMaskStrategy

The mask strategy `plan` filters under. Every plan type defines a method.
"""
function plan_strategy end
plan_strategy(plan::PhysicalFilterPlan) = plan.strategy

"""
    plan_method(plan) -> AbstractFilterMethod

The method a plan evaluates: `RealSpace()` for a real-space engine, `AutoMethod()` for one of its
transform evaluators, `Spectral()` for a transfer-function plan.
"""
plan_method(plan::PhysicalFilterPlan) = _transform_footprint(plan.footprint) ? AutoMethod() : RealSpace()
plan_method(::AbstractFilterPlan) = Spectral()

"""
    plan_execution(plan) -> AbstractExecutionBackend

The execution backend `plan` was built for, whose memory its inputs and outputs live in. Every plan
type defines a method.
"""
function plan_execution end
plan_execution(plan::PhysicalFilterPlan) = plan.execution

"""
    prepare_workspace(backend, grid, footprint) -> workspace

Backend hook [`plan_filter`](@ref) calls once; its result is stored as the plan's `footprint`. The
default returns `footprint` unchanged; a backend with its own memory, such as a GPU's device buffers,
returns the resident form its apply step reads.
"""
prepare_workspace(
    ::ComputationalBackends.AbstractExecutionBackend, ::FlowGeometries.Grids.AbstractGrid, fp,
) = fp

@inline function _validate_scale(scale::T) where {T<:AbstractFloat}
    isfinite(scale) && scale > zero(T) || throw(ArgumentError(
        "filter scale must be finite and positive, got $scale",
    ))
    return nothing
end

@noinline _unused_keywords(kwargs, method) = throw(ArgumentError(
    "plan_filter with method = $(nameof(typeof(method)))() takes no keyword " *
    "$(join(map(k -> "`$k`", collect(keys(kwargs))), ", "))",
))

"""
    plan_filter(grid, kernel, scale; mask_strategy = ZeroFill(), execution = AutoBackend(),
                method = RealSpace(), transform = AutoSpectralBackend(),
                cache_strategy = AutoCache(), cache_byte_budget = default_cache_byte_budget(),
                grid_plan = nothing, scratch = nothing) -> AbstractFilterPlan

A reusable filter plan for `execution`, applied by `filter_apply!(out, field, plan)`: the real-space
footprint, or under `method = Spectral()` the transform plan. A spectral plan also takes `batch = nb`
for a trailing batch axis and, over a spherical node set, `nufft`, the NUFFT library NUFSHT runs; a
keyword the method does not take raises an `ArgumentError`. `grid_plan` and `scratch` are the parts a
sweep shares; see [`plan_filter_sweep`](@ref).

A transform runs on 1 thread under `SerialBackend()` and on `Threads.nthreads()` under
`ThreadedBackend()`, `AutoBackend()` and `GPUBackend()`, in the device's memory under the last.
Under `MPIBackend(inner)` every rank holds the field and the output. A nonuniform transform (NUFFT,
NUFSHT) gives each rank a disjoint share of the points: the analysis `F_k = Σⱼ wⱼ cⱼ exp(−i k⋅xⱼ)` is a
sum over points, so the ranks' partial sums add to it exactly, and each rank evaluates the filtered
series at its own points. An FFT or FastSphericalHarmonics transform runs whole on every rank. Under
`DistributedBackend(inner)` the worker processes share the transform, a nonuniform one by blocks of
points and an FFT by blocks of columns and then of rows, each at `inner`'s thread count; a
FastSphericalHarmonics transform runs in the calling process.
"""
function plan_filter(
    grid::FlowGeometries.Grids.StructuredGrid{T,G},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    method::AbstractFilterMethod = RealSpace(),
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    cache_strategy::AbstractCacheStrategy = AutoCache(),
    cache_byte_budget::Integer = default_cache_byte_budget(),
    grid_plan::Union{Nothing,AbstractGridPlan} = nothing,
    scratch::Union{Nothing,AbstractFilterScratch} = nothing,
    kwargs...,
) where {G<:FlowGeometries.Geometry.AbstractGeometry{T}} where {T<:AbstractFloat}
    _validate_scale(scale)
    if _resolve_method(method) isa Spectral
        return spectral_filter_plan(
            transform, grid, kernel, scale;
            mask_strategy = mask_strategy, execution = execution,
            grid_plan = grid_plan, scratch = scratch, kwargs...,
        )
    end
    isempty(kwargs) || _unused_keywords(kwargs, method)
    resolved = _resolve_backend(execution, grid)
    _check_backend_compatible(grid, execution)
    # The transform engines run on the host (see `_transform_footprint`), so `AutoMethod` takes them only
    # there; a device, distributed or MPI plan keeps the direct engine.
    host = _host_backend(resolved)
    fp = if host && _padded_fft_applicable(grid, kernel, method)
        padded_fft_footprint(grid, kernel, scale; mask_strategy = mask_strategy, threads = _library_threads(resolved))
    elseif host && _zonal_fft_applicable(grid, kernel, method)
        zonal_fft_footprint(grid, kernel, scale; mask_strategy = mask_strategy, threads = _library_threads(resolved))
    else
        build_footprint(grid, kernel, scale; mask_strategy = mask_strategy,
            cache_strategy = cache_strategy, cache_byte_budget = cache_byte_budget,
            grid_plan = grid_plan, scratch = scratch)
    end
    return PhysicalFilterPlan(prepare_workspace(resolved, grid, fp), grid, mask_strategy, kernel, scale, resolved)
end
