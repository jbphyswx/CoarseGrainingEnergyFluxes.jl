# ---------------------------------------------------------------------------
# Sweep families: one grid plan and one scratch shared by every scale
# ---------------------------------------------------------------------------

"""
    FilterPlanFamily(grid_plan, plans, scratch)

The plans for a sweep of scales over one grid, with the scale-independent
[`AbstractGridPlan`](@ref) and the per-apply [`AbstractFilterScratch`](@ref) they share; either is
`nothing` for an engine with nothing to share. A family is an `AbstractVector` of its plans.

The plans of one family share one scratch, so they must not be applied concurrently; give each
concurrent worker its own family.
"""
struct FilterPlanFamily{
    P, GP<:Union{Nothing,AbstractGridPlan}, PV<:AbstractVector{P},
    SC<:Union{Nothing,AbstractFilterScratch},
} <: AbstractVector{P}
    grid_plan::GP
    plans::PV
    scratch::SC
end

FilterPlanFamily(grid_plan::GP, plans::PV, scratch::SC) where {
    P, GP, PV<:AbstractVector{P}, SC,
} = FilterPlanFamily{P,GP,PV,SC}(grid_plan, plans, scratch)

Base.size(f::FilterPlanFamily) = size(f.plans)
Base.@propagate_inbounds Base.getindex(f::FilterPlanFamily, i::Int) = f.plans[i]
Base.IndexStyle(::Type{<:FilterPlanFamily}) = Base.IndexLinear()

Base.show(io::IO, f::FilterPlanFamily) =
    print(io, "FilterPlanFamily(", length(f.plans), " scales, ",
          f.grid_plan === nothing ? "no shared grid plan" : nameof(typeof(f.grid_plan)), ")")
Base.show(io::IO, ::MIME"text/plain", f::FilterPlanFamily) = show(io, f)

# `(grid_plan, scratch)` an engine shares across the scales of a sweep, each `nothing` independently.
# Its methods follow `build_footprint`'s dispatch, so the per-scale builder takes what they return.
_sweep_shared(grid, kernel, mask_strategy, method; kwargs...) =
    _spectral_sweep_shared(grid, kernel, mask_strategy, method; kwargs...)

function _spectral_sweep_shared(
    grid, kernel, mask_strategy, method;
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    kwargs...,
)
    _resolve_method(method) isa Spectral || return (nothing, nothing)
    gp = spectral_grid_plan(transform, grid, kernel; mask_strategy = mask_strategy, kwargs...)
    return (gp, gp === nothing ? nothing : spectral_scratch(gp))
end

function _sweep_shared(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    kernel::Kernels.TopHatKernel,
    mask_strategy::AbstractMaskStrategy,
    method::AbstractFilterMethod;
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    _resolve_method(method) isa Spectral &&
        return _spectral_sweep_shared(grid, kernel, mask_strategy, method; kwargs...)
    FlowGeometries.Grids.measure_factors(grid) === nothing && return (nothing, nothing)
    gp = _build_prefixsum_grid_plan(grid; mask_strategy = mask_strategy)
    return (gp, _prefixsum_scratch(gp, grid))
end

function _sweep_shared(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    kernel::SeparableKernel,
    mask_strategy::AbstractMaskStrategy,
    method::AbstractFilterMethod;
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    _resolve_method(method) isa Spectral &&
        return _spectral_sweep_shared(grid, kernel, mask_strategy, method; kwargs...)
    FlowGeometries.Grids.measure_factors(grid) === nothing && return (nothing, nothing)
    return (nothing, _separable_scratch(grid))
end

"""
    plan_filter_sweep(grid, kernel, scales; mask_strategy = ZeroFill(), method = nothing,
                      cache_byte_budget = default_cache_byte_budget(), kwargs...) -> FilterPlanFamily

One [`plan_filter`](@ref) per scale in `scales`, all sharing one grid-determined half of the engine
and one per-apply scratch. For the prefix-sum top-hat engine the shared half is the measure prefix
scans, the extended axis and the mask scan. `method = nothing` takes the grid's default method.

Forwards every other [`plan_filter`](@ref) keyword. `cache_byte_budget` bounds the neighbour caches of
the whole sweep: each scale's plan is given what the scales before it left.

```julia
family = plan_filter_sweep(grid, TopHatKernel(), scales)
for (s, plan) in zip(scales, family)
    filter_apply!(out, field, plan)
end
```
"""
function plan_filter_sweep(
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scales::AbstractVector;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    method::Union{Nothing,AbstractFilterMethod} = nothing,
    cache_byte_budget::Integer = default_cache_byte_budget(),
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    meth = method === nothing ? _default_method(grid) : method
    gp, sc = _sweep_shared(grid, kernel, mask_strategy, meth; kwargs...)
    # Only the keywords the engine takes are passed; `plan_filter` raises on any other.
    shared = gp === nothing && sc === nothing ? (;) :
             sc === nothing ? (; grid_plan = gp) : (; grid_plan = gp, scratch = sc)
    left = Int(cache_byte_budget)
    plans = map(scales) do s
        budget = _takes_cache_budget(grid) ? (; cache_byte_budget = max(left, 0)) : (;)
        p = plan_filter(grid, kernel, T(s); mask_strategy = mask_strategy, method = meth,
                        shared..., budget..., kwargs...)
        left -= _cache_bytes(p)
        p
    end
    return FilterPlanFamily(gp, plans, sc)
end

# The grids whose real-space plans hold an optional neighbour cache under a byte budget.
_takes_cache_budget(::Union{FlowGeometries.Grids.StructuredGrid, FlowGeometries.Grids.CurvilinearGrid}) = true
_takes_cache_budget(::FlowGeometries.Grids.AbstractGrid) = false

# The bytes a plan's neighbour cache holds, counted as the budget counts them.
_cache_bytes(plan::PhysicalFilterPlan) = _cache_bytes(plan.footprint)
_cache_bytes(fp::ScatteredFilterPlan{T}) where {T} =
    fp.cache === nothing ? 0 : length(fp.cache.w) * _cache_entry_bytes(Val(2), T)
_cache_bytes(fp::NDScatteredFilterPlan{N,T}) where {N,T} =
    fp.cache === nothing ? 0 : length(fp.cache.w) * _cache_entry_bytes(Val(N), T)
_cache_bytes(_) = 0

"""
    _default_method(grid) -> AbstractFilterMethod

The method [`plan_filter`](@ref) and [`plan_filter_sweep`](@ref) take on `grid` when the caller names
none: `RealSpace()` on every grid, the compact-kernel convolution that defines coarse graining (Aluie
2019). The transfer-function filter is reached only by an explicit `method = Spectral()`.
"""
_default_method(::FlowGeometries.Grids.AbstractGrid) = RealSpace()

# Grids whose parallel backends decompose over rows through `apply_footprint_row!`.
_row_parallelizable(::FlowGeometries.Grids.StructuredGrid{T,G,2}) where {G,T} = true
_row_parallelizable(::FlowGeometries.Grids.CurvilinearGrid) = true
_row_parallelizable(::FlowGeometries.Grids.AbstractGrid) = false

"""
    _is_flat_cell(grid) -> Bool

Whether `grid` names a cell by a single integer, `FlowGeometries.Grids.FlatCells()`: a node set, or a
spherical pixelization whose cells carry one id. The node CSR engine reads such a grid only through
`fold_within(grid, t)`, `measure(grid, j)` and `isactive(grid, t)`. A `CartesianCells()` grid names a
cell by an index tuple and takes the rectilinear or curvilinear engines.
"""
@inline _is_flat_cell(grid::FlowGeometries.Grids.AbstractGrid) =
    FlowGeometries.Grids.cell_address(grid) === FlowGeometries.Grids.FlatCells()

# Dispatch handle for the trait: a method that must vary with it takes this as its first argument.
@inline _cell_address(grid::FlowGeometries.Grids.AbstractGrid) =
    FlowGeometries.Grids.cell_address(grid)

# Grids with point-indexed footprints, whose parallel backends split over points: each point writes
# only its own cell.
_nd_parallelizable(::FlowGeometries.Grids.StructuredGrid{T,G,1}) where {G,T} = true
_nd_parallelizable(::FlowGeometries.Grids.StructuredGrid{T,G,3}) where {G,T} = true
_nd_parallelizable(grid::FlowGeometries.Grids.AbstractGrid) = _is_flat_cell(grid)

# Whether `grid` has a parallel path for a concrete backend.
_backend_supported(grid::FlowGeometries.Grids.AbstractGrid, ::ComputationalBackends.SerialBackend) = true
_backend_supported(grid::FlowGeometries.Grids.AbstractGrid, ::ComputationalBackends.ThreadedBackend) = _row_parallelizable(grid) || _nd_parallelizable(grid)
_backend_supported(grid::FlowGeometries.Grids.AbstractGrid, ::ComputationalBackends.DistributedBackend) =
    _row_parallelizable(grid) || _nd_parallelizable(grid)
_backend_supported(grid::FlowGeometries.Grids.AbstractGrid, ::ComputationalBackends.GPUBackend) =
    _row_parallelizable(grid) || _nd_parallelizable(grid)
_backend_supported(grid::FlowGeometries.Grids.AbstractGrid, ::ComputationalBackends.MPIBackend) =
    _row_parallelizable(grid) || _nd_parallelizable(grid)

@inline _fftw_available() =
    Base.get_extension(parentmodule(@__MODULE__), :CoarseGrainingEnergyFluxesFFTWExt) !== nothing

# `AutoMethod` chooses an evaluator of the real-space convolution, so it resolves to `RealSpace`.
@inline _resolve_method(::AutoMethod) = RealSpace()
@inline _resolve_method(method::AbstractFilterMethod) = method

# Kernels with a factored real-space engine.
_has_fast_real_space_engine(::Kernels.TopHatKernel) = true       # O(N) prefix sum
_has_fast_real_space_engine(::SeparableKernel) = true            # O(N·(wx+wy)) separable
_has_fast_real_space_engine(::Kernels.AbstractFilterKernel) = false

# `AutoMethod` evaluates a real-space filter with no factored engine on a uniform Cartesian grid by
# padded transform, the same linear convolution in O(N log N); the transform needs one
# translation-invariant footprint.
_padded_fft_applicable(grid, kernel, method::AbstractFilterMethod) = false
_padded_fft_applicable(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2,S,TP,<:Tuple{AbstractRange,AbstractRange}},
    kernel::Kernels.AbstractFilterKernel, ::AutoMethod,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}, S, TP} =
    !_has_fast_real_space_engine(kernel) && _fftw_available()

# `AutoMethod` evaluates a spherical real-space filter by transform along longitude when the weights
# depend on the longitude difference alone (a radial kernel, or a `HighOrderKernel` corrected per row)
# and longitude is a uniform periodic `Range`; `TopHatKernel` keeps its prefix-sum engine. Latitude may
# be any axis. The weights are the direct sum's, with compact support.
_zonal_fft_applicable(grid, kernel, method::AbstractFilterMethod) = false
_zonal_fft_applicable(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2,S,TP,<:Tuple{AbstractRange,<:AbstractVector}},
    kernel::Kernels.AbstractFilterKernel, ::AutoMethod,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.SphericalGeometry{T}, S, TP} =
    !(kernel isa Kernels.TopHatKernel) && (Kernels.is_radial(kernel) || kernel isa Kernels.HighOrderKernel) &&
    FlowGeometries.Grids.isperiodic(grid, 1) && _fftw_available()

@inline _threading_available() =
    Base.get_extension(parentmodule(@__MODULE__), :CoarseGrainingEnergyFluxesOhMyThreadsExt) !== nothing

# The backends whose applies run in this process's own loops, where the transform engines
# (`_transform_footprint`) are available.
_host_backend(b::ComputationalBackends.AbstractExecutionBackend) =
    b isa Union{ComputationalBackends.AbstractSerialBackend, ComputationalBackends.AbstractThreadedBackend}

# Resolves `AutoBackend` against this package's extensions and `grid`.
# `ComputationalBackends.resolve_backend(::AutoBackend)` is ComputationalBackends' own signature, shared
# by all its consumers.
@inline function _resolve_backend(
    backend::ComputationalBackends.AbstractExecutionBackend, grid::FlowGeometries.Grids.AbstractGrid,
)
    backend isa ComputationalBackends.AbstractAutoBackend ||
        return ComputationalBackends.resolve_backend(backend)
    threaded = ComputationalBackends.ThreadedBackend()
    return (Threads.nthreads() > 1 && _threading_available() && _backend_supported(grid, threaded)) ?
        threaded : ComputationalBackends.SerialBackend()
end

# Raises for an explicit non-serial `backend` that `grid` has no parallel path for; `AutoBackend()`
# resolves to serial there instead.
function _check_backend_compatible(grid::FlowGeometries.Grids.AbstractGrid, backend::ComputationalBackends.AbstractExecutionBackend)
    if !(backend isa ComputationalBackends.AutoBackend) && !(backend isa ComputationalBackends.SerialBackend) && !_backend_supported(grid, _resolve_backend(backend, grid))
        throw(ArgumentError(
            "execution = $(typeof(backend)) was requested explicitly, but $(typeof(grid)) has no " *
            "matching parallel hook for it — there is no way to honor this request. Pass " *
            "`execution = SerialBackend()` explicitly if serial execution is acceptable, or " *
            "`execution = AutoBackend()` to let the library choose.",
        ))
    end
    return nothing
end

"""
    build_footprint(grid, kernel, scale; kwargs...) -> NodeFilterPlan

Real-space footprint for a flat-cell grid (see [`_is_flat_cell`](@ref)) from
`Connectivity.fold_within`, the grid's own metric ball query, so the neighbourhood follows the
geometry's distance and periodic wrap and each weight takes the distance the fold returns. Raises for a
grid whose cells are an index tuple and that no rectilinear or curvilinear engine matches.

The ball queries run on the topology `Connectivity.default_sweep_topology` returns: a spatial index,
`O(n log n)` over the grid, when `NearestNeighbors` is loaded, and a linear scan, `O(n²)`, otherwise.
"""
build_footprint(
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}} =
    _build_footprint_flat(_cell_address(grid), grid, kernel, scale; kwargs...)

# Reached only by a `CartesianCells` grid that no rectilinear or curvilinear `build_footprint` matches.
@noinline _build_footprint_flat(
    ::FlowGeometries.Grids.CartesianCells, grid, kernel, scale; kwargs...,
) = throw(ArgumentError(
    "no real-space engine is implemented for $(nameof(typeof(grid))). Its cells are named by an " *
    "index tuple, so the flat-cell node engine does not apply, and no rectilinear or curvilinear " *
    "engine matches it either. Use `method = Spectral()` if a transform targets this grid.",
))

function _build_footprint_flat(
    ::FlowGeometries.Grids.FlatCells,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    dim = _kernel_dim(grid)
    rad = Kernels.kernel_radius(kernel, scale, dim)
    n = length(FlowGeometries.Grids.mask(grid))
    ptr = Vector{Int}(undef, n + 1)
    nbrs = Int[]
    w = T[]
    ptr[1] = 1
    mt = _query_topology(grid, rad)
    scratch = FlowGeometries.Connectivity.ball_scratch()
    for t in 1:n
        # `active_only = false` keeps the plan mask-independent, the mask entering at apply time;
        # `self = true` includes the centre cell.
        FlowGeometries.Connectivity.fold_within(
            nothing, grid, t; ball = rad, self = true, active_only = false,
            topology = mt, scratch = scratch,
        ) do _, j, d
            push!(nbrs, j)
            push!(w, Kernels.kernel_weight(kernel, T(d), scale, dim) * FlowGeometries.Grids.measure(grid, j))
            return nothing
        end
        ptr[t+1] = length(nbrs) + 1
    end
    return NodeFilterPlan(nbrs, w, ptr)
end

# A `HighOrderKernel`'s weights are corrected about each node ([`_highorder_targets`](@ref)), and a node
# plan stores them all.
function _build_footprint_flat(
    ::FlowGeometries.Grids.FlatCells,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.HighOrderKernel,
    scale::T;
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    rad = _query_radius(grid, kernel, scale)
    mt = _query_topology(grid, rad)
    scratch = FlowGeometries.Connectivity.ball_scratch()
    fold = (f, acc, t) -> FlowGeometries.Connectivity.fold_within(
        f, acc, grid, t; ball = rad, self = true, active_only = false, topology = mt, scratch = scratch,
    )
    _, _, kept = _highorder_targets(grid, kernel, scale, rad, fold, nothing, Returns(true))
    return NodeFilterPlan(kept...)
end

"""
    apply_footprint!(out, field, grid, fp::NodeFilterPlan, strategy) -> out

Weighted mean over each node's stored neighbourhood, with the same two mask conventions the structured
engines use: `ZeroFill` keeps a masked neighbour in the denominator, contributes nothing for it and
filters a masked node too; `Deformable` drops it from both sums and leaves a masked node at zero.
"""
function apply_footprint!(
    out::AbstractVector{T}, field::AbstractVector, grid::FlowGeometries.Grids.AbstractGrid,
    fp::NodeFilterPlan{T}, strategy::AbstractMaskStrategy,
) where {T<:AbstractFloat}
    @inbounds for t in eachindex(out)
        out[t] = _footprint_node_point(field, grid, fp, strategy, t)
    end
    return out
end

# Node `t`'s filtered value, shared by the serial and parallel drivers. A flat-cell grid has no lattice
# past its cells, so no exterior mass enters the normalization.
@inline function _footprint_node_point(
    field::AbstractVector, grid::FlowGeometries.Grids.AbstractGrid,
    fp::NodeFilterPlan{T}, strategy::AbstractMaskStrategy, t::Integer,
) where {T<:AbstractFloat}
    _filters_target(strategy, FlowGeometries.Grids.isactive(grid, t)) || return zero(T)
    ws = zero(T)
    wn = zero(T)
    @inbounds for k in fp.ptr[t]:(fp.ptr[t+1] - 1)
        j = fp.nbrs[k]
        wj = fp.w[k]
        active = FlowGeometries.Grids.isactive(grid, j)
        if strategy isa ZeroFill
            wn += wj
            active && (ws += wj * field[j])
        else
            active || continue
            wn += wj
            ws += wj * field[j]
        end
    end
    return _normalized(ws, wn)
end

"""
    plan_filter(grid, kernel, scale; mask_strategy = ZeroFill(), execution = AutoBackend(),
                method = _default_method(grid), transform = AutoSpectralBackend(), kwargs...)

The plan for a grid with no rectilinear or curvilinear engine of its own; `method` defaults to
[`_default_method`](@ref). In real space a flat-cell grid (see [`_is_flat_cell`](@ref)) has the node
CSR engine over its own ball query, and any other grid raises. Under `method = Spectral()` the grid is
filtered over its cells by the nonuniform transform of its geometry: NUFFT for a Cartesian one, NUFSHT
for a spherical one.

`RealSpace()` applies the kernel with its compact support. A transform is exact for a band-limited
field and its per-apply cost is independent of the filter scale; its support is global.
"""
function plan_filter(
    grid::FlowGeometries.Grids.AbstractGrid,
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    method::AbstractFilterMethod = _default_method(grid),
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    kwargs...,
) where {T<:AbstractFloat}
    _validate_scale(scale)
    method isa Spectral && return spectral_filter_plan(
        transform, grid, kernel, scale; mask_strategy = mask_strategy, execution = execution, kwargs...,
    )
    isempty(kwargs) || _unused_keywords(kwargs, method)
    _check_backend_compatible(grid, execution)
    resolved = _resolve_backend(execution, grid)
    fp = build_footprint(grid, kernel, scale)
    return PhysicalFilterPlan(prepare_workspace(resolved, grid, fp), grid, mask_strategy, kernel, scale, resolved)
end

# A curvilinear grid's real-space engine is the scattered per-point footprint, whose neighbour cache
# `cache_strategy` controls.
function plan_filter(
    grid::FlowGeometries.Grids.CurvilinearGrid{T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    method::AbstractFilterMethod = RealSpace(),
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    cache_strategy::AbstractCacheStrategy = AutoCache(),
    cache_byte_budget::Integer = default_cache_byte_budget(),
    kwargs...,
) where {T<:AbstractFloat}
    _validate_scale(scale)
    if method isa Spectral
        # Over its cells as a node set (engines/NodeSpectral).
        return spectral_filter_plan(transform, grid, kernel, scale; mask_strategy = mask_strategy, execution = execution, kwargs...)
    end
    isempty(kwargs) || _unused_keywords(kwargs, method)
    resolved = _resolve_backend(execution, grid)
    _check_backend_compatible(grid, execution)
    fp = build_footprint(grid, kernel, scale; cache_strategy = cache_strategy, cache_byte_budget = cache_byte_budget)
    return PhysicalFilterPlan(prepare_workspace(resolved, grid, fp), grid, mask_strategy, kernel, scale, resolved)
end

"""
    filter_apply!(out, field, plan) -> out

Apply the prebuilt `plan` (see [`plan_filter`](@ref)) to `field`, writing `out`, on the backend the
plan was built for. Every backend applies the footprint stored in `plan`.
"""
function filter_apply!(out::AbstractArray, field::AbstractArray, plan::PhysicalFilterPlan)
    if plan.execution isa ComputationalBackends.SerialBackend
        return _apply_serial!(out, field, plan.grid, plan.footprint, plan.strategy)
    elseif plan.execution isa ComputationalBackends.ThreadedBackend
        return threaded_filter_field!(out, field, plan.grid, plan.kernel, plan.scale, plan.strategy, plan.footprint)
    elseif plan.execution isa ComputationalBackends.DistributedBackend
        return distributed_filter_field!(out, field, plan.grid, plan.kernel, plan.scale, plan.strategy, plan.footprint)
    elseif plan.execution isa ComputationalBackends.GPUBackend
        return gpu_filter_field!(plan.execution, out, field, plan.grid, plan.kernel, plan.scale, plan.strategy, plan.footprint)
    elseif plan.execution isa ComputationalBackends.MPIBackend
        return mpi_filter_field!(plan.execution, out, field, plan.grid, plan.kernel, plan.scale, plan.strategy, plan.footprint)
    else
        throw(ArgumentError("Unsupported execution backend: $(typeof(plan.execution))"))
    end
end

# Whether the device extension has a fused batch kernel for `plan`'s engine; every other engine takes
# the slice loop.
_gpu_batched_supported(::AbstractFilterPlan) = false

"""
    filter_apply_batched!(out, field, plan) -> out

Apply `plan` to each slice along a trailing batch axis: `out` and `field` are `(spatial..., Nb)` over
the plan's grid. On a device with a fused kernel for the engine this is one launch of
`prod(spatial) * Nb` work items; otherwise it loops over the slices. For several separate arrays on
one grid, see [`filter_apply_batch!`](@ref).
"""
function filter_apply_batched!(out::AbstractArray, field::AbstractArray, plan::AbstractFilterPlan)
    spatial = FlowGeometries.Grids.size_tuple(plan.grid)
    valR = Val(length(spatial))
    _check_batched_shape(out, "out", spatial, valR)
    _check_batched_shape(field, "field", spatial, valR)
    size(out) == size(field) || throw(DimensionMismatch(
        "filter_apply_batched! got out $(size(out)) and field $(size(field))",
    ))
    if plan.execution isa ComputationalBackends.GPUBackend && _gpu_batched_supported(plan)
        gpu_filter_field_batched!(
            plan.execution, out, field, plan.grid, plan.kernel, plan.scale, plan.strategy, plan.footprint,
        )
        return out
    end
    d = ndims(out)
    for b in axes(out, d)
        filter_apply!(selectdim(out, d, b), selectdim(field, d, b), plan)
    end
    return out
end

function _check_batched_shape(
    A::AbstractArray, name::AbstractString, spatial::NTuple{R,Int}, ::Val{R},
) where {R}
    ndims(A) == R + 1 || throw(DimensionMismatch(
        "$name has $(ndims(A)) dimensions; filter_apply_batched! expects $R spatial + 1 batch",
    ))
    ntuple(i -> size(A, i), Val(R)) == spatial || throw(DimensionMismatch(
        "$name's leading dimensions $(ntuple(i -> size(A, i), Val(R))) do not match grid shape $spatial",
    ))
    return nothing
end

"""
    analyze_buffer(plan, field) -> F̂ or nothing
    filter_analyze!(F̂, field, plan) -> F̂
    filter_synthesize!(out, F̂, plan) -> out

A spectral apply in two halves: `filter_analyze!`, the forward transform, and `filter_synthesize!`,
the multiply by `Ĝ(|k|, ℓ)` and the inverse transform. Only the second depends on the scale, so a sweep
analyzes each field once and synthesizes it at every scale: `F̂` analyzed with one plan may be
synthesized with the plan for any other scale over the same grid.

`analyze_buffer` returns a buffer for `F̂`, or `nothing` for an engine with no scale-independent half;
callers then use [`filter_apply!`](@ref).
"""
function analyze_buffer end
function filter_analyze! end
function filter_synthesize! end

# Real-space engines have no scale-independent half to hoist.
analyze_buffer(::AbstractFilterPlan, ::AbstractArray) = nothing

function gpu_filter_field_batched!(args...; kwargs...)
    throw(ArgumentError("GPUBackend is unavailable — run `using KernelAbstractions` (or use SerialBackend())."))
end

# Several separate fields on one device footprint: their launches are enqueued back to back and the
# host waits once, at the end of the batch.
function gpu_filter_fields!(args...; kwargs...)
    throw(ArgumentError("GPUBackend is unavailable — run `using KernelAbstractions` (or use SerialBackend())."))
end

"""
    filter_apply_batch!(outs, fields, plan) -> outs

Apply one prebuilt `plan` to several separate fields on its grid, writing `outs[i]` from `fields[i]`;
the result equals [`filter_apply!`](@ref) per field. The scattered footprints derive each target
point's neighbourhood once for the whole batch.

`outs` and `fields` are equal-length collections of matching arrays: an `NTuple{K,V}` for a batch size
known at compile time (stack accumulators), or an `AbstractVector`. Arrays with a trailing batch axis
go through [`filter_apply_batched!`](@ref). For independent fields on different grids, see
[`filter_slices!`](@ref).
"""
function filter_apply_batch!(outs, fields, plan::PhysicalFilterPlan)
    _batched_fields(outs, plan) && return filter_apply_batch_trailing!(outs, fields, plan)
    if plan.execution isa ComputationalBackends.SerialBackend
        return _apply_serial_batch!(outs, fields, plan.grid, plan.footprint, plan.strategy)
    elseif plan.execution isa ComputationalBackends.ThreadedBackend
        return threaded_filter_fields!(outs, fields, plan.grid, plan.kernel, plan.scale, plan.strategy, plan.footprint)
    elseif plan.execution isa ComputationalBackends.GPUBackend
        return gpu_filter_fields!(
            plan.execution, outs, fields, plan.grid, plan.kernel, plan.scale, plan.strategy, plan.footprint,
        )
    else
        # Distributed and MPI apply field by field.
        for k in eachindex(outs)
            filter_apply!(outs[k], fields[k], plan)
        end
        return outs
    end
end

function filter_apply_batch!(outs, fields, plan::AbstractFilterPlan)
    _batched_fields(outs, plan) && return filter_apply_batch_trailing!(outs, fields, plan)
    for k in eachindex(outs)
        filter_apply!(outs[k], fields[k], plan)
    end
    return outs
end

# Whether the fields carry a trailing batch axis: one dimension more than the plan's grid.
@inline function _batched_fields(outs, plan::PhysicalFilterPlan)
    R = length(FlowGeometries.Grids.size_tuple(plan.grid))
    return ndims(first(outs)) == R + 1
end

# A spectral plan type with no method of its own has no trailing-batch form.
@inline _batched_fields(outs, ::AbstractFilterPlan) = false

function filter_apply_batch_trailing!(outs, fields, plan::AbstractFilterPlan)
    for k in eachindex(outs)
        filter_apply_batched!(outs[k], fields[k], plan)
    end
    return outs
end

"""
    filter_fields!(outs, fields, grid, kernel, scale; mask_strategy = ZeroFill(), filter_plan = nothing,
                   execution = AutoBackend())

Filter several fields on one grid at one kernel and scale with a single plan, `filter_plan` or one
built by [`plan_filter`](@ref), through [`filter_apply_batch!`](@ref). `outs` and `fields` are
indexable collections of matching arrays, such as a tuple of velocity components.
"""
function filter_fields!(
    outs,
    fields,
    grid::FlowGeometries.Grids.StructuredGrid{T,G},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    filter_plan::Union{Nothing,AbstractFilterPlan} = nothing,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
) where {G<:FlowGeometries.Geometry.AbstractGeometry{T}} where {T<:AbstractFloat}
    plan = filter_plan === nothing ? plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, execution = execution) : filter_plan
    return filter_apply_batch!(outs, fields, plan)
end
