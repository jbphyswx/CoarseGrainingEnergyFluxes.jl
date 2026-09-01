# ---------------------------------------------------------------------------
# Sweep families: one grid plan and one scratch shared by every scale
# ---------------------------------------------------------------------------

"""
    FilterPlanFamily(grid_plan, plans, scratch)

The plans for a whole sweep of scales over one grid, together with the two pieces they share: the
scale-independent [`AbstractGridPlan`](@ref) and the per-apply [`AbstractFilterScratch`](@ref).

Indexing and iteration go straight to `plans`, so anywhere a `Vector` of per-scale plans was expected
a family works unchanged.

`grid_plan`/`scratch` are `nothing` for an engine that has no shared state to hoist; the family is
then just the vector of plans and costs nothing extra.

Because the scales share one scratch, **the plans of one family may not be applied concurrently with
each other**. Give each concurrent worker its own family.
"""
struct FilterPlanFamily{
    P, GP<:Union{Nothing,AbstractGridPlan}, PV<:AbstractVector{P},
    SC<:Union{Nothing,AbstractFilterScratch},
} <: AbstractVector{P}
    grid_plan::GP
    plans::PV
    scratch::SC
end

# `P` is read off the plan vector's element type rather than spelled by the caller.
FilterPlanFamily(grid_plan::GP, plans::PV, scratch::SC) where {
    P, GP, PV<:AbstractVector{P}, SC,
} = FilterPlanFamily{P,GP,PV,SC}(grid_plan, plans, scratch)

# A family IS the per-scale plan vector, plus the two things the scales share. Subtyping `AbstractVector`
# rather than forwarding a handful of methods is what lets one be passed anywhere a vector of plans was
# — `filter_plans = …`, `first(plans)`, `plans[s_idx]` — with no signature widened to admit it.
Base.size(f::FilterPlanFamily) = size(f.plans)
Base.@propagate_inbounds Base.getindex(f::FilterPlanFamily, i::Int) = f.plans[i]
Base.IndexStyle(::Type{<:FilterPlanFamily}) = Base.IndexLinear()

# A plan prints as its type name (its transforms would otherwise walk a C library's internal plan
# tree), so a family says what it holds rather than listing them.
Base.show(io::IO, f::FilterPlanFamily) =
    print(io, "FilterPlanFamily(", length(f.plans), " scales, ",
          f.grid_plan === nothing ? "no shared grid plan" : nameof(typeof(f.grid_plan)), ")")
Base.show(io::IO, ::MIME"text/plain", f::FilterPlanFamily) = show(io, f)

# What an engine can share across the scales of one sweep. Returns `(grid_plan, scratch)`, either of
# which may be `nothing` independently: an engine can have scale-independent TABLES, or only transient
# BUFFERS, or neither. The separable engines are the second case — every table they hold (tap weights,
# profiles, `invrenorm`) is a function of ℓ, while their two pass buffers are sized by the grid alone.
#
# Derived from the same predicates `build_footprint` dispatches on, so the family cannot prepare shared
# state the per-scale builder will not use.
_sweep_shared(grid, kernel, mask_strategy, method) =
    _spectral_sweep_shared(grid, kernel, mask_strategy, method)

# A spectral engine's shared half is its transform objects, whichever backend supplies them; its
# scratch is sized from those and shared by the sequential scales of the sweep.
function _spectral_sweep_shared(grid, kernel, mask_strategy, method)
    _resolve_method(grid, kernel, method) isa Spectral || return (nothing, nothing)
    gp = spectral_grid_plan(
        SpectralBackends.AutoSpectralBackend(), grid, kernel; mask_strategy = mask_strategy,
    )
    return (gp, gp === nothing ? nothing : spectral_scratch(gp))
end

function _sweep_shared(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    kernel::Kernels.TopHatKernel,
    mask_strategy::AbstractMaskStrategy,
    method::AbstractFilterMethod,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    _resolve_method(grid, kernel, method) isa Spectral &&
        return _spectral_sweep_shared(grid, kernel, mask_strategy, method)
    FlowGeometries.Grids.measure_factors(grid) === nothing && return (nothing, nothing)
    gp = _build_prefixsum_grid_plan(grid; mask_strategy = mask_strategy)
    return (gp, _prefixsum_scratch(gp, grid))
end

function _sweep_shared(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    kernel::SeparableKernel,
    mask_strategy::AbstractMaskStrategy,
    method::AbstractFilterMethod,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    _resolve_method(grid, kernel, method) isa Spectral &&
        return _spectral_sweep_shared(grid, kernel, mask_strategy, method)
    FlowGeometries.Grids.measure_factors(grid) === nothing && return (nothing, nothing)
    return (nothing, _separable_scratch(grid))
end

"""
    plan_filter_sweep(grid, kernel, scales; kwargs...) -> FilterPlanFamily

Plan a whole sweep of scales at once, building the grid-determined half of the engine and the
per-apply scratch **once** instead of once per scale.

This is what [`plan_filter`](@ref) cannot do on its own: given only one scale it has no way to know
another is coming, so it must own its own copies. For the prefix-sum top-hat engine, the shared half
is the measure prefix scans, the extended axis and the mask scan; a sweep of `S` scales over a
`1024²` grid holds one copy of each rather than `S`.

Accepts and forwards every [`plan_filter`](@ref) keyword.

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
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    meth = method === nothing ? _default_method(grid) : method
    gp, sc = _sweep_shared(grid, kernel, mask_strategy, meth)
    # An engine with nothing to share is planned exactly as before, without the two extra keywords —
    # the spectral and node builders forward `kwargs...` onward, so handing them a keyword they do not
    # know about would surface deep inside a transform plan constructor rather than here.
    (gp === nothing && sc === nothing) && return FilterPlanFamily(
        nothing,
        [plan_filter(grid, kernel, T(s); mask_strategy = mask_strategy, method = meth, kwargs...) for s in scales],
        nothing,
    )
    # Only the keywords the engine actually has are passed: an engine with scale-independent tables but
    # no apply-time buffers gets `grid_plan` alone, and handing it a `scratch` it does not accept would
    # surface as a keyword error from inside its constructor.
    plans = sc === nothing ?
        [plan_filter(grid, kernel, T(s); mask_strategy = mask_strategy, method = meth,
                     grid_plan = gp, kwargs...) for s in scales] :
        [plan_filter(grid, kernel, T(s); mask_strategy = mask_strategy, method = meth,
                     grid_plan = gp, scratch = sc, kwargs...) for s in scales]
    return FilterPlanFamily(gp, plans, sc)
end

"""
    _default_method(grid) -> AbstractFilterMethod

Which method `plan_filter` takes on `grid` when the caller names none. The sweep resolves it the same
way before asking whether the engine has a shared grid plan.

`RealSpace()` wherever an engine for it exists, on every grid architecture alike. Coarse graining is
defined as a convolution against a compact kernel (Aluie 2019), so the filter at a point reads only its
own neighbourhood, which survives a regional domain and a coastline; `ZeroFill` keeps that kernel
position-independent, so filtering commutes with `∇` and the `Π` budget closes.

A transform reproduces the same convolution only on a global, unmasked domain and only up to its
bandlimit, so it is reached by an explicit `method = Spectral()`, or by `AutoMethod()` choosing an
evaluator on capability. The grid architecture never changes which operator ran.

A layout with no real-space engine gets `Spectral()`, whose builder then names the backends that exist.
"""
_default_method(::FlowGeometries.Grids.StructuredGrid) = RealSpace()
_default_method(::FlowGeometries.Grids.CurvilinearGrid) = RealSpace()
_default_method(grid::FlowGeometries.Grids.AbstractGrid) =
    _is_flat_cell(grid) ? RealSpace() : RealSpace()

# The row-based parallel backends (Threaded/Distributed/GPU/MPI) decompose over rows of a 2D grid
# via `apply_footprint_row!`, which already works generically for CurvilinearGrid (a 2D grid using
# the scattered per-point footprint) as well as StructuredGrid.
_row_parallelizable(::FlowGeometries.Grids.StructuredGrid{T,G,2}) where {G,T} = true
_row_parallelizable(::FlowGeometries.Grids.CurvilinearGrid) = true
_row_parallelizable(::FlowGeometries.Grids.AbstractGrid) = false

"""
    _is_flat_cell(grid) -> Bool

Whether `grid` names a cell by a single integer — `FlowGeometries.Grids.FlatCells()`.

The node CSR engine and everything built on it are written against exactly that: `fold_within(grid, t)`
with one index, `measure(grid, j)`, `isactive(grid, t)`. A node set carries the trait, and so does every
spherical pixelization whose cells carry one id: ring, cubed-sphere, healpix, icosahedral and Yin–Yang
layouts. The trait is asked at dispatch, so a layout upstream adds is served the day it declares itself
flat-celled.

The counterpart is `CartesianCells()`, whose cells are an index tuple; those grids take the rectilinear
and curvilinear engines above.
"""
@inline _is_flat_cell(grid::FlowGeometries.Grids.AbstractGrid) =
    FlowGeometries.Grids.cell_address(grid) === FlowGeometries.Grids.FlatCells()

# Dispatch handle for the trait: a method that must vary with it takes this as its first argument.
@inline _cell_address(grid::FlowGeometries.Grids.AbstractGrid) =
    FlowGeometries.Grids.cell_address(grid)

# 1D/true-3D grids use point-indexed footprints, so their parallel hook iterates points rather than
# rows. Each output point reads neighbours and writes only its own cell, so that is equally valid.
# Threaded only; Distributed/GPU/MPI would need a domain decomposition for the ND case.
_nd_parallelizable(::FlowGeometries.Grids.StructuredGrid{T,G,1}) where {G,T} = true
_nd_parallelizable(::FlowGeometries.Grids.StructuredGrid{T,G,3}) where {G,T} = true
# A flat-cell grid is point-indexed with no row structure, so it parallelizes on the same argument as
# the 1D/true-3D grids: `_footprint_node_point` writes only cell `t`'s own value.
_nd_parallelizable(grid::FlowGeometries.Grids.AbstractGrid) = _is_flat_cell(grid)

# Whether `grid` can actually honor a specific concrete backend request.
_backend_supported(grid::FlowGeometries.Grids.AbstractGrid, ::ComputationalBackends.SerialBackend) = true
_backend_supported(grid::FlowGeometries.Grids.AbstractGrid, ::ComputationalBackends.ThreadedBackend) = _row_parallelizable(grid) || _nd_parallelizable(grid)
# Point-indexed grids decompose over linear indices into a SharedArray rather than over rows, so they
# are supported even though `_row_parallelizable` is false for them.
_backend_supported(grid::FlowGeometries.Grids.AbstractGrid, ::ComputationalBackends.DistributedBackend) =
    _row_parallelizable(grid) || _nd_parallelizable(grid)
# The device kernels for point-indexed footprints need no row decomposition, so GPU support follows
# `_nd_parallelizable` as well as `_row_parallelizable` — one kernel over a linear index serves 1-D,
# true-3-D and node sets alike.
_backend_supported(grid::FlowGeometries.Grids.AbstractGrid, ::ComputationalBackends.GPUBackend) =
    _row_parallelizable(grid) || _nd_parallelizable(grid)
# Point-indexed grids partition round-robin over linear indices and recombine with `Allreduce!`, so
# they are supported even though `_row_parallelizable` is false for them.
_backend_supported(grid::FlowGeometries.Grids.AbstractGrid, ::ComputationalBackends.MPIBackend) =
    _row_parallelizable(grid) || _nd_parallelizable(grid)

# `AutoBackend()` landing on serial is auto-selection working; an explicit non-serial request that
# cannot be honoured is an error, since the caller would otherwise run on believing they had the
# parallelism. Checked against the original `backend`, before `AutoBackend()` is resolved away.
@inline _fftw_available() =
    Base.get_extension(parentmodule(@__MODULE__), :CoarseGrainingEnergyFluxesFFTWExt) !== nothing

# A transform filters by PERIODIC convolution, so it reproduces the intended filter only where every
# axis wraps, and it needs constant spacing — which a `Range` axis proves at the type level.
_spectral_exact(grid::FlowGeometries.Grids.StructuredGrid{T,G,N}) where {
    T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}, N,
} =
    all(ntuple(d -> FlowGeometries.Grids.isperiodic(grid, d), N)) &&
    all(ntuple(d -> FlowGeometries.Grids.coordinates(grid, d) isa AbstractRange, N))
_spectral_exact(::FlowGeometries.Grids.AbstractGrid) = false

@inline _besselj1_available() =
    Base.get_extension(parentmodule(@__MODULE__), :CoarseGrainingEnergyFluxesSpecialFunctionsExt) !== nothing

# `AutoMethod` picks on real capability, never on a preference: a transform only where it is available
# AND exact for this grid. Every kernel wins there, including the top-hat — its prefix-sum engine is
# O(N) but with a large enough constant to lose 60x to a transform whose cost does not scale with the
# filter width at all (30.0 ms vs 0.5 ms at half-width 64 on a periodic 256^2 grid).
@inline function _resolve_method(grid, kernel, method::AbstractFilterMethod)
    method isa AutoMethod || return method
    (_fftw_available() && _spectral_exact(grid)) || return RealSpace()
    # The planar top-hat's transfer function is the Bessel-J₁ form, which only exists when the
    # SpecialFunctions extension is loaded; without it there is no spectral top-hat to select.
    kernel isa Kernels.TopHatKernel && !_besselj1_available() && return RealSpace()
    return Spectral()
end

# Which kernels have a factored real-space engine. A `SharpSpectralKernel` has none — its radial `sinc`
# does not separate — so it falls to the banded disk sum at O(N·w²) with a 10ℓ radius.
_has_fast_real_space_engine(::Kernels.TopHatKernel) = true       # O(N) prefix sum
_has_fast_real_space_engine(::SeparableKernel) = true            # O(N·(wx+wy)) separable
_has_fast_real_space_engine(::Kernels.AbstractFilterKernel) = false

# `AutoMethod` may evaluate a real-space filter by padded transform: it is the SAME linear convolution,
# valid on bounded and masked domains, at O(N log N) instead of O(N·w²) — 716.7 → 0.7 ms at half-width
# 40. Only for a kernel with no factored engine, and Cartesian only, since the padded transform assumes
# one translation-invariant footprint and a spherical grid's per-latitude bands are not that.
_padded_fft_applicable(grid, kernel, method::AbstractFilterMethod) = false
_padded_fft_applicable(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2,S,TP,<:Tuple{AbstractRange,AbstractRange}},
    kernel::Kernels.AbstractFilterKernel, ::AutoMethod,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}, S, TP} =
    !_has_fast_real_space_engine(kernel) && _fftw_available()

# `AutoMethod` may also evaluate a SPHERICAL real-space filter by transform along longitude. Conditions,
# all necessary:
#
#   * the kernel is RADIAL — it must depend on the great-circle distance alone, or it is not a function
#     of the longitude difference and there is no convolution to exploit;
#   * longitude is a uniform `Range` AND periodic, so the ring really closes and a circular transform is
#     the exact operator rather than an approximation of a truncated one;
#   * not `TopHatKernel`, which already has the exact prefix-sum engine at O(N·dj_lim) — strictly better
#     than any transform here.
#
# Latitude may be anything: the band structure is a plain sum over source rows either way. This is NOT
# `Spectral()`: no spherical-harmonic truncation is involved, the compact kernel keeps its support, and
# the weights are the same ones the direct sum uses.
_zonal_fft_applicable(grid, kernel, method::AbstractFilterMethod) = false
_zonal_fft_applicable(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2,S,TP,<:Tuple{AbstractRange,<:AbstractVector}},
    kernel::Kernels.AbstractFilterKernel, ::AutoMethod,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.SphericalGeometry{T}, S, TP} =
    !(kernel isa Kernels.TopHatKernel) && Kernels.is_radial(kernel) &&
    FlowGeometries.Grids.isperiodic(grid, 1) && _fftw_available()

@inline _threading_available() =
    Base.get_extension(parentmodule(@__MODULE__), :CoarseGrainingEnergyFluxesOhMyThreadsExt) !== nothing

# Upstream leaves `resolve_backend(::AutoBackend)` to the consumer, since it cannot see whether this
# package's threading extension is loaded. Kept package-local rather than added as a method there:
# that signature is ComputationalBackends' own, so every consumer defining it would overwrite the rest.
@inline function _resolve_backend(
    backend::ComputationalBackends.AbstractExecutionBackend, grid::FlowGeometries.Grids.AbstractGrid,
)
    backend isa ComputationalBackends.AbstractAutoBackend ||
        return ComputationalBackends.resolve_backend(backend)
    threaded = ComputationalBackends.ThreadedBackend()
    # Auto must choose on REAL capability: threads available, the extension that implements them
    # loaded, AND this grid having a parallel path. Choosing a backend the grid cannot honor is what
    # silently produced serial execution under a reported parallel backend.
    return (Threads.nthreads() > 1 && _threading_available() && _backend_supported(grid, threaded)) ?
        threaded : ComputationalBackends.SerialBackend()
end

function _check_backend_compatible(grid::FlowGeometries.Grids.AbstractGrid, backend::ComputationalBackends.AbstractExecutionBackend)
    if !(backend isa ComputationalBackends.AutoBackend) && !(backend isa ComputationalBackends.SerialBackend) && !_backend_supported(grid, _resolve_backend(backend, grid))
        throw(ArgumentError(
            "backend = $(typeof(backend)) was requested explicitly, but $(typeof(grid)) has no " *
            "matching parallel hook for it — there is no way to honor this request. Pass " *
            "`backend = SerialBackend()` explicitly if serial execution is acceptable, or " *
            "`backend = AutoBackend()` to let the library choose.",
        ))
    end
    return nothing
end

"""
    build_footprint(grid, kernel, scale; kwargs...) -> NodeFilterPlan

Real-space footprint for a flat-cell grid — see [`_is_flat_cell`](@ref) — from
`Connectivity.fold_within`, the grid's own metric ball query. The neighbourhood therefore honours the
geometry's distance and any periodic wrap as the grid defines them, and the fold hands back each
neighbour's distance for the weight to use directly.

A node set and a spherical pixelization present the same interface here: one integer per cell, a
measure per cell, and a ball query. So one engine serves both, and nothing in it reads a coordinate
directly.

The sweep visits every cell, which is where a spatial index pays for itself:
`Connectivity.default_sweep_topology` builds one when `NearestNeighbors` is loaded, taking the build
from `O(n²)` to `O(n log n)`, and returns the unindexed topology (same rows, linear scan) when it is
not. Either way it is paid once per plan and reused by every `filter_apply!`, including the six to
nine a single `compute_Π!` makes.
"""
build_footprint(
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}} =
    _build_footprint_flat(_cell_address(grid), grid, kernel, scale; kwargs...)

# The rectilinear and curvilinear engines above are matched on the concrete grid type, so they are more
# specific than the entry point and this is only reached by a layout none of them names.
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
    rad = Kernels.kernel_radius(kernel, scale)
    n = length(FlowGeometries.Grids.mask(grid))
    ptr = Vector{Int}(undef, n + 1)
    nbrs = Int[]
    w = T[]
    ptr[1] = 1
    mt = _query_topology(grid, rad)
    scratch = FlowGeometries.Connectivity.ball_scratch()
    for t in 1:n
        # `active_only = false`: which cells are masked is the mask STRATEGY's business at apply time,
        # exactly as it is for the structured engines, whose caches are likewise mask-independent.
        # `self = true` folds the centre at distance zero, where the kernel carries its largest weight.
        FlowGeometries.Connectivity.fold_within(
            nothing, grid, t; ball = rad, self = true, active_only = false,
            topology = mt, scratch = scratch,
        ) do _, j, d
            push!(nbrs, j)
            push!(w, Kernels.kernel_weight(kernel, T(d), scale) * FlowGeometries.Grids.measure(grid, j))
            return nothing
        end
        ptr[t+1] = length(nbrs) + 1
    end
    return NodeFilterPlan(nbrs, w, ptr)
end

"""
    apply_footprint!(out, field, grid, fp::NodeFilterPlan, strategy) -> out

Weighted mean over each node's stored neighbourhood, with the same two mask conventions the structured
engines use: `ZeroFill` keeps a masked neighbour in the denominator and contributes nothing for it,
`Deformable` drops it from both.
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

# Per-node kernel, factored out of the loop above so a parallel driver reuses the exact same
# arithmetic rather than duplicating it — node `t` reads neighbours and writes only its own cell, so
# the threaded result is bit-identical. Mirrors `_footprint_nd_point`'s role for the ND engines.
@inline function _footprint_node_point(
    field::AbstractVector, grid::FlowGeometries.Grids.AbstractGrid,
    fp::NodeFilterPlan{T}, strategy::AbstractMaskStrategy, t::Integer,
) where {T<:AbstractFloat}
    FlowGeometries.Grids.isactive(grid, t) || return zero(T)
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
    return wn > T(1e-15) ? ws / wn : zero(T)
end

"""
    plan_filter(grid, kernel, scale; method = <the grid's own default>, …)

Entry point for every grid architecture with no rectilinear or curvilinear engine of its own.

A flat-cell grid — see [`_is_flat_cell`](@ref) — has the node CSR real-space engine over its own ball
query, and a spectral one wherever a transform targets it: the nonuniform-FFT backend for a Cartesian
node set, the nonuniform spherical-harmonic one for a spherical node set. [`_default_method`](@ref)
chooses between them per grid, and both plan in `O(n log n)`.

`RealSpace()` applies the kernel as written, with compact support. A transform is exact for a
band-limited field and its per-apply cost does not grow with the filter scale, at the price of global
support. `build_footprint` raises for a layout with neither engine.
"""
function plan_filter(
    grid::FlowGeometries.Grids.AbstractGrid,
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    method::AbstractFilterMethod = _default_method(grid),
    spectral_backend::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    kwargs...,
) where {T<:AbstractFloat}
    _validate_scale(scale)
    method isa Spectral && return spectral_filter_plan(
        spectral_backend, grid, kernel, scale; mask_strategy = mask_strategy, backend = backend, kwargs...,
    )
    _check_backend_compatible(grid, backend)
    return PhysicalFilterPlan(
        build_footprint(grid, kernel, scale), grid, mask_strategy, kernel, scale,
        _resolve_backend(backend, grid),
    )
end

# Curvilinear grids have a genuine real-space direct-sum engine (the scattered per-point footprint),
# so — unlike the unstructured/spectral-only fallback above — they precompute a `PhysicalFilterPlan`.
# More specific than the `AbstractGrid` method, so it is chosen for a `CurvilinearGrid`.
function plan_filter(
    grid::FlowGeometries.Grids.CurvilinearGrid{T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    method::AbstractFilterMethod = RealSpace(),
    spectral_backend::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    cache_strategy::AbstractCacheStrategy = AutoCache(),
    cache_byte_budget::Integer = DEFAULT_CACHE_BYTE_BUDGET,
    kwargs...,
) where {T<:AbstractFloat}
    _validate_scale(scale)
    if method isa Spectral
        # No spectral backend targets a CurvilinearGrid (FINUFFT/NUFSHT are UnstructuredGrid-only),
        # so this raises the standard informative "spectral unavailable" error.
        return spectral_filter_plan(spectral_backend, grid, kernel, scale; mask_strategy = mask_strategy, backend = backend, kwargs...)
    end
    resolved = _resolve_backend(backend, grid)
    _check_backend_compatible(grid, backend)
    fp = build_footprint(grid, kernel, scale; cache_strategy = cache_strategy, cache_byte_budget = cache_byte_budget)
    return PhysicalFilterPlan(prepare_workspace(resolved, grid, fp), grid, mask_strategy, kernel, scale, resolved)
end

"""
    filter_apply!(out, field, plan) -> out

Apply a prebuilt [`plan_filter`](@ref) to a single 2D field, dispatching to whichever backend the
plan was built for — the footprint is ALWAYS the one cached in `plan`, never rebuilt here, for every
backend (serial, threaded, distributed, GPU, MPI).
"""
function filter_apply!(out::AbstractArray, field::AbstractArray, plan::PhysicalFilterPlan)
    if plan.backend isa ComputationalBackends.SerialBackend
        return _apply_serial!(out, field, plan.grid, plan.footprint, plan.strategy)
    elseif plan.backend isa ComputationalBackends.ThreadedBackend
        return threaded_filter_field!(out, field, plan.grid, plan.kernel, plan.scale, plan.strategy, plan.footprint)
    elseif plan.backend isa ComputationalBackends.DistributedBackend
        return distributed_filter_field!(out, field, plan.grid, plan.kernel, plan.scale, plan.strategy, plan.footprint)
    elseif plan.backend isa ComputationalBackends.GPUBackend
        return gpu_filter_field!(plan.backend, out, field, plan.grid, plan.kernel, plan.scale, plan.strategy, plan.footprint)
    elseif plan.backend isa ComputationalBackends.MPIBackend
        return mpi_filter_field!(out, field, plan.grid, plan.kernel, plan.scale, plan.strategy, plan.footprint)
    else
        throw(ArgumentError("Unsupported backend: $(typeof(plan.backend))"))
    end
end

"""
    filter_apply_batch!(outs, fields, plan::AbstractFilterPlan) -> outs

Apply `plan` to every field in `fields`, writing into the matching entry of `outs`, deriving each
target point's neighbour list/weight exactly ONCE and reusing it across the whole batch — not once
per field. `outs`/`fields` must be equal-length, matching-shape collections of arrays: an
`NTuple{K,V}` (single concrete array type `V`) for a compile-time-known batch size (fastest — see
`_batch_zeros`), or an `AbstractVector` for a runtime-determined batch size.
"""
# A fused device batch exists only for engines whose kernel carries a batch index; the extension
# overrides this for those. Never guess — an unfused engine must take the slice loop, not a wrong kernel.
_gpu_batched_supported(::AbstractFilterPlan) = false

"""
    filter_apply_batched!(out, field, plan) -> out

Apply `plan` across a **trailing batch axis**: `out` and `field` are `(spatial..., Nb)` over the plan's
grid. The filter gathers only along spatial axes, so slices are independent and the batch index is
carried through untouched.

On a device this issues ONE launch of `prod(spatial) * Nb` work items instead of `Nb` launches of
`prod(spatial)`, which is what matters when a single slice does not fill the device — a 64² slice is
4k work items. On the host it is the same work as slicing and looping, and exists so callers have one
shape-generic entry point rather than reimplementing the loop.

Differs from [`filter_apply_batch!`](@ref), which takes several *separate* fields sharing a grid; here
the batch is one contiguous array, which is what allows the fused launch.
"""
function filter_apply_batched!(out::AbstractArray, field::AbstractArray, plan::AbstractFilterPlan)
    spatial = FlowGeometries.Grids.size_tuple(plan.grid)
    valR = Val(length(spatial))
    _check_batched_shape(out, "out", spatial, valR)
    _check_batched_shape(field, "field", spatial, valR)
    size(out) == size(field) || throw(DimensionMismatch(
        "filter_apply_batched! got out $(size(out)) and field $(size(field))",
    ))
    if plan.backend isa ComputationalBackends.GPUBackend && _gpu_batched_supported(plan)
        gpu_filter_field_batched!(
            plan.backend, out, field, plan.grid, plan.kernel, plan.scale, plan.strategy, plan.footprint,
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

Split of a spectral apply into its two halves. A spectral filter is forward transform → multiply by
`Ĝ(|k|, ℓ)` → inverse transform, and only the multiply depends on the scale, so a sweep over S scales
needs the forward ONCE per field rather than once per (field, scale): `5 + 5S` transforms instead of
`10S`.

`analyze_buffer` returns `nothing` for an engine with no transform to share — a real-space filter does
all its work per scale — and callers fall back to [`filter_apply!`](@ref).

Plans for different scales over the same grid share a forward transform, so `F̂` produced with any one
of them may be synthesized with any other.
"""
function analyze_buffer end
function filter_analyze! end
function filter_synthesize! end

# Real-space engines have no scale-independent half to hoist.
analyze_buffer(::AbstractFilterPlan, ::AbstractArray) = nothing

function gpu_filter_field_batched!(args...; kwargs...)
    throw(ArgumentError("GPUBackend is unavailable — run `using KernelAbstractions` (or use SerialBackend())."))
end

# Several separate fields sharing one grid and one device footprint. Separate from
# `gpu_filter_field_batched!`, which carries a trailing batch axis inside one array: here the K launches
# are independent, so they are enqueued back to back and the host waits once at the end.
function gpu_filter_fields!(args...; kwargs...)
    throw(ArgumentError("GPUBackend is unavailable — run `using KernelAbstractions` (or use SerialBackend())."))
end

"""
    filter_apply_batch!(outs, fields, plan) -> outs

Apply one prebuilt `plan` to SEVERAL separate fields sharing its grid, writing `outs[i]` from
`fields[i]`. Equivalent to calling [`filter_apply!`](@ref) per field and asserted bit-identical to it,
but the engines that derive per-point geometry — the scattered/node footprints — derive each target
point's neighbourhood once for the whole batch instead of once per field, which is where the saving is.

`compute_Π!` filters five to nine fields per scale, so this is the shape its inner loop uses.

Distinct from [`filter_slices!`](@ref), which takes independent fields on DIFFERENT grids, one plan
each; here there is one grid and one plan. For a single array carrying a trailing batch axis, this
dispatches on to the fused-launch path.
"""
function filter_apply_batch!(outs, fields, plan::PhysicalFilterPlan)
    _batched_fields(outs, plan) && return filter_apply_batch_trailing!(outs, fields, plan)
    if plan.backend isa ComputationalBackends.SerialBackend
        return _apply_serial_batch!(outs, fields, plan.grid, plan.footprint, plan.strategy)
    elseif plan.backend isa ComputationalBackends.ThreadedBackend
        return threaded_filter_fields!(outs, fields, plan.grid, plan.kernel, plan.scale, plan.strategy, plan.footprint)
    elseif plan.backend isa ComputationalBackends.GPUBackend
        # The K applies are independent launches over one device footprint: they enter the queue back to
        # back and the host blocks once, at the end of the batch.
        return gpu_filter_fields!(
            plan.backend, outs, fields, plan.grid, plan.kernel, plan.scale, plan.strategy, plan.footprint,
        )
    else
        # Distributed/MPI own the decomposition of the domain across their workers, and the batch hoist
        # belongs inside that decomposition. They apply per field, each call reusing the plan's footprint.
        for k in eachindex(outs)
            filter_apply!(outs[k], fields[k], plan)
        end
        return outs
    end
end

# Spectral plans have no per-point neighbour derivation to hoist out of a per-field loop — each apply
# is an independent transform pass — so batching has nothing to collapse and they apply sequentially.
function filter_apply_batch!(outs, fields, plan::AbstractFilterPlan)
    _batched_fields(outs, plan) && return filter_apply_batch_trailing!(outs, fields, plan)
    for k in eachindex(outs)
        filter_apply!(outs[k], fields[k], plan)
    end
    return outs
end

# When the fields themselves carry a trailing batch axis, each one goes through the batched entry point
# rather than the per-slice apply: the field axis stays a loop, but every field's whole batch is one pass.
# `_batched_fields` is checked against the plan's grid, so a true-3D field on a 3D grid is not mistaken
# for a batched 2D field.
@inline function _batched_fields(outs, plan::PhysicalFilterPlan)
    R = length(FlowGeometries.Grids.size_tuple(plan.grid))
    return ndims(first(outs)) == R + 1
end

# A spectral plan carries transforms rather than a grid, so it has no spatial rank to compare against and
# no trailing-batch form; it keeps the per-field loop.
@inline _batched_fields(outs, ::AbstractFilterPlan) = false

function filter_apply_batch_trailing!(outs, fields, plan::AbstractFilterPlan)
    for k in eachindex(outs)
        filter_apply_batched!(outs[k], fields[k], plan)
    end
    return outs
end

"""
    filter_fields!(outs, fields, grid, kernel, scale; mask_strategy=ZeroFill(), backend=AutoBackend())

Filter several fields that share the same grid/kernel/scale, building the footprint/plan ONCE and
applying it through [`filter_apply_batch!`](@ref), so each target point's neighbours are enumerated
once for the whole batch rather than once per field. `outs` and `fields` are indexable collections of
matching arrays — a tuple of velocity components, or a vector of them.
"""
function filter_fields!(
    outs,
    fields,
    grid::FlowGeometries.Grids.StructuredGrid{T,G},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    filter_plan::Union{Nothing,AbstractFilterPlan} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
) where {G<:FlowGeometries.Geometry.AbstractGeometry{T}} where {T<:AbstractFloat}
    plan = filter_plan === nothing ? plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend) : filter_plan
    return filter_apply_batch!(outs, fields, plan)
end
