# ---------------------------------------------------------------------------
# Reusable filter plans: build the footprint ONCE, apply to many fields/scales
# ---------------------------------------------------------------------------

"""
Physical-space plan: a precomputed footprint reused across all longitudes, fields, and layers — for
EVERY backend, not just serial. `kernel`/`scale` are retained only so the cached-footprint path can
still call each backend's row-parallel hook (which takes them positionally); they're not used to
rebuild the footprint once `footprint` is already built.
"""
struct PhysicalFilterPlan{FP, G<:FlowGeometries.Grids.AbstractGrid, S<:AbstractMaskStrategy, K<:Kernels.AbstractFilterKernel, T<:AbstractFloat, B<:ComputationalBackends.AbstractExecutionBackend} <: AbstractFilterPlan
    footprint::FP   # FilterFootprint (2D structured), FilterFootprintND (1D/3D), or ScatteredFilterPlan/NDScatteredFilterPlan (nonuniform/curvilinear)
    grid::G
    strategy::S
    kernel::K
    scale::T
    backend::B
end

"""
    plan_strategy(plan) -> AbstractMaskStrategy

The mask strategy `plan` filters under. Every plan type carries one: a diagnostic reads its filtered
fields on the grid that strategy defines them on.
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
    prepare_workspace(backend, grid, footprint) -> workspace

Backend hook run ONCE by [`plan_filter`](@ref), whose result becomes the plan's stored workspace. The
default returns the footprint unchanged; a backend that needs its own residency — the GPU's device
buffers — returns something its apply step consumes directly, so no transfer is repeated per call.
"""
prepare_workspace(
    ::ComputationalBackends.AbstractExecutionBackend, ::FlowGeometries.Grids.AbstractGrid, fp,
) = fp

# Boundary-only validation (paid once per `plan_filter` call, not per grid point): a non-positive or
# non-finite filter scale is never physically meaningful and would otherwise surface later as a
# confusing NaN/zero-radius footprint deep in the call stack instead of a clear error at the API edge.
@inline function _validate_scale(scale::T) where {T<:AbstractFloat}
    isfinite(scale) && scale > zero(T) || throw(ArgumentError(
        "filter scale must be finite and positive, got $scale",
    ))
    return nothing
end

# A keyword the plan's method does not take is refused.
@noinline _unused_keywords(kwargs, method) = throw(ArgumentError(
    "plan_filter with method = $(nameof(typeof(method)))() takes no keyword " *
    "$(join(map(k -> "`$k`", collect(keys(kwargs))), ", "))",
))

"""
    plan_filter(grid, kernel, scale; mask_strategy = ZeroFill(), backend = AutoBackend(),
                method = RealSpace(), spectral_backend = AutoSpectralBackend(),
                cache_strategy = AutoCache(), cache_byte_budget) -> AbstractFilterPlan

A reusable filter plan for `backend`, applied by `filter_apply!(out, field, plan)`: the real-space
footprint, or under `method = Spectral()` the transform plan, which also takes `batch = nb` for a
trailing batch axis and, over a spherical node set, `nufft`, the NUFFT library NUFSHT runs. A keyword
the method does not take is refused.

A transform runs on 1 thread under `SerialBackend()` and on `Threads.nthreads()` under
`ThreadedBackend()`, `AutoBackend()` and `GPUBackend()`, in the device's memory under the last.
Under `MPIBackend(inner)` every rank holds the field and the output. A nonuniform transform (NUFFT,
NUFSHT) gives each rank a disjoint share of the points: the analysis `F_k = Σⱼ wⱼ cⱼ exp(−i k⋅xⱼ)` is a
sum over points, so the ranks' partial sums add to it exactly, and each rank evaluates the filtered
series at its own points. An FFT or FastSphericalHarmonics transform runs whole on every rank. Under
`DistributedBackend(inner)` the worker processes share the transform: a nonuniform transform gives
each a block of the points, and an FFT a block of the columns and then of the rows; each worker runs its
share at `inner`'s thread count. A FastSphericalHarmonics transform runs in the calling process.
"""
function plan_filter(
    grid::FlowGeometries.Grids.StructuredGrid{T,G},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    method::AbstractFilterMethod = RealSpace(),
    spectral_backend::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    cache_strategy::AbstractCacheStrategy = AutoCache(),
    cache_byte_budget::Integer = DEFAULT_CACHE_BYTE_BUDGET,
    grid_plan::Union{Nothing,AbstractGridPlan} = nothing,
    scratch::Union{Nothing,AbstractFilterScratch} = nothing,
    kwargs...,
) where {G<:FlowGeometries.Geometry.AbstractGeometry{T}} where {T<:AbstractFloat}
    _validate_scale(scale)
    if _resolve_method(method) isa Spectral
        return spectral_filter_plan(
            spectral_backend, grid, kernel, scale;
            mask_strategy = mask_strategy, backend = backend,
            grid_plan = grid_plan, scratch = scratch, kwargs...,
        )
    end
    isempty(kwargs) || _unused_keywords(kwargs, method)
    resolved = _resolve_backend(backend, grid)
    _check_backend_compatible(grid, backend)
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
