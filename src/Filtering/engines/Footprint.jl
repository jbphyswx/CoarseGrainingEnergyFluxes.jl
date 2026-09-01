# ---------------------------------------------------------------------------
# Serial physical-space convolution: precomputed footprint + single apply loop
# ---------------------------------------------------------------------------

"""
    BandedScratch{T,MT,VM} <: AbstractFilterScratch

Per-apply buffers for the banded engine: the masked input `mask · field`, one per field in flight.

Empty for an unmasked grid — there the engine convolves the caller's array directly, since
`mask · field == field` and the copy would be pure cost. Slots are added on demand for a batched apply
and then reused, so repeat applies allocate nothing.
"""
struct BandedScratch{
    T<:AbstractFloat, MT<:AbstractMatrix{T}, VM<:AbstractVector{MT},
} <: AbstractFilterScratch
    masked_inputs::VM
end

# Grown on demand, `similar` off slot 1 so a new buffer inherits the array type actually in use.
function _banded_inputs!(sc::BandedScratch, K::Integer)
    while length(sc.masked_inputs) < K
        push!(sc.masked_inputs, similar(@inbounds sc.masked_inputs[1]))
    end
    return sc.masked_inputs
end

"""
    FilterFootprint{T,VI,VT,MT,SC,MS}

Precomputed convolution footprint for a structured grid + kernel + scale. The in-support neighbour
offsets `(di, dj)` and their geometric weights `w = kernel_weight(distance) * cell_area` are stored
in a flat CSR-like layout, grouped into axis-2 (`y`) bands (`ptr[b]:ptr[b+1]-1`). For Cartesian grids
the footprint is translation-invariant → a single band; for a spherical grid it is invariant in `x`
(longitude) → one band per `y` (latitude) value.

The offsets and weights are geometry only. The normalization is not: `invden` is the reciprocal window
mass, which depends on the mask, the mask strategy and where the window is truncated by a domain edge.
It is accumulated once at plan build by running the SAME tap loop the numerator uses over a mass field
— `1` everywhere for `ZeroFill`, whose denominator counts every in-support tap regardless of the mask,
and the mask itself for `Deformable`, which drops inactive cells from both sums.

Precomputing it is what lets the apply hold the tap index in the outer loop and convolve by contiguous
axpy along `x`, at every column of every grid. A denominator accumulated per point would force the tap
index inside, turning the inner loop into a floating-point reduction that cannot be reassociated and so
does not vectorize. Masked grids, and the columns within one filter radius of a domain edge, therefore
run the same vectorized path as the unmasked interior.
"""
struct FilterFootprint{
    T<:AbstractFloat, VI<:AbstractVector{Int}, VT<:AbstractVector{T},
    MT<:AbstractMatrix{T}, SC<:BandedScratch{T}, MS<:AbstractMaskStrategy,
}
    di::VI    # axis-1 (x) index offset
    dj::VI    # axis-2 (y) index offset
    w::VT       # kernel_weight(distance) * cell area
    ptr::VI   # band b's entries: ptr[b]:ptr[b+1]-1
    nbands::Int        # 1 (Cartesian) or Ny (spherical)
    strategy::MS       # the strategy `invden` was accumulated under
    masked::Bool       # grid has inactive cells, so the numerator needs `mask · field`
    # Captured from the grid at build, because `invden` was accumulated under them: honouring a
    # different wrap at apply time would pair a numerator with a denominator computed over a different
    # support. The apply signature still accepts the flags, for interface uniformity, and ignores them
    # — the same contract `PrefixSumTopHatPlan` documents.
    periodic_x::Bool
    periodic_y::Bool
    invden::MT         # 1/(window mass) per point; zero where the target is inactive or the window empty
    scratch::SC        # per-apply `mask · field` buffers
end

"""
    _banded_row_accumulate!(oc, src, di, dj, w, lo, hi, periodic_x, periodic_y, Nx, Ny, j) -> oc

Accumulate one output row's weighted tap sum, contiguously along `x`.

This is the whole banded engine. It is used twice with different inputs: on `mask · field` to build
the numerator at apply time, and on the mass field to build `invden` at plan build. Holding the tap
index in the OUTER loop is what makes the inner loop a unit-stride axpy — accumulating one output
point at a time would make it a floating-point reduction, which cannot be reassociated and so never
vectorizes.

On a bounded axis a tap contributes only where its source index is in range. Clamping the RANGE, rather
than branching inside the loop, keeps that run contiguous — which is what lets the columns within one
filter radius of an edge take the same vectorized path as the interior. At a wide kernel those columns
are most of every row, so they are not an edge case worth handling separately.
"""
@inline function _banded_row_accumulate!(
    oc::AbstractVector{T}, src::AbstractMatrix,
    di::AbstractVector{Int}, dj::AbstractVector{Int}, w::AbstractVector{T},
    lo::Int, hi::Int, periodic_x::Bool, periodic_y::Bool, Nx::Int, Ny::Int, j::Integer,
) where {T<:AbstractFloat}
    @inbounds for k in lo:hi
        jj = j + dj[k]
        if jj < 1 || jj > Ny
            periodic_y || continue
            jj = mod1(jj, Ny)
        end
        wk = w[k]
        d = di[k]
        fc = view(src, :, jj)
        if periodic_x && abs(d) < Nx
            # A wrapped tap is two contiguous runs, each at a CONSTANT offset. One `mod1` loop would
            # make the index data-dependent and cost the vectorization.
            if d >= 0
                @simd for i in 1:(Nx - d)
                    oc[i] += wk * fc[i + d]
                end
                @simd for i in (Nx - d + 1):Nx
                    oc[i] += wk * fc[i + d - Nx]
                end
            else
                @simd for i in 1:(-d)
                    oc[i] += wk * fc[i + d + Nx]
                end
                @simd for i in (1 - d):Nx
                    oc[i] += wk * fc[i + d]
                end
            end
        elseif periodic_x
            @simd for i in 1:Nx
                oc[i] += wk * fc[mod1(i + d, Nx)]
            end
        else
            @simd for i in max(1, 1 - d):min(Nx, Nx - d)
                oc[i] += wk * fc[i + d]
            end
        end
    end
    return oc
end

@inline _banded_band(nbands::Int, j::Integer) = nbands == 1 ? 1 : Int(j)

"""
    _banded_build_invden(grid, di, dj, w, ptr, nbands, strategy, periodic_x, periodic_y) -> invden

The reciprocal window mass, accumulated once per plan by running [`_banded_row_accumulate!`](@ref)
over a mass field:

- `ZeroFill` counts every in-support tap whether or not its source cell is active, so its mass field
  is `1` everywhere and the result varies only where a domain edge truncates the window.
- `Deformable` drops inactive cells from both sums, so its mass field is the mask.

The target-activity test and the degeneracy floor are folded in here too, so the apply's epilogue is
one multiply with no branch.
"""
function _banded_build_invden(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    di::AbstractVector{Int}, dj::AbstractVector{Int}, w::AbstractVector{T},
    ptr::AbstractVector{Int}, nbands::Int, strategy::AbstractMaskStrategy,
    periodic_x::Bool, periodic_y::Bool,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    mass = Matrix{T}(undef, Nx, Ny)
    if strategy isa ZeroFill
        fill!(mass, one(T))
    else
        @inbounds for j in 1:Ny, i in 1:Nx
            mass[i, j] = FlowGeometries.Grids.isactive(grid, i, j) ? one(T) : zero(T)
        end
    end
    den = zeros(T, Nx, Ny)
    for j in 1:Ny
        b = _banded_band(nbands, j)
        _banded_row_accumulate!(
            view(den, :, j), mass, di, dj, w, ptr[b], ptr[b + 1] - 1,
            periodic_x, periodic_y, Nx, Ny, j,
        )
    end
    @inbounds for j in 1:Ny, i in 1:Nx
        den[i, j] = (FlowGeometries.Grids.isactive(grid, i, j) && den[i, j] > T(1e-15)) ?
            inv(den[i, j]) : zero(T)
    end
    return den
end

# Assembles the parts every `FilterFootprint` needs beyond its offset/weight lists. Separate from the
# two builders above it so the Cartesian and spherical branches share one definition of what
# normalization and scratch mean.
function _banded_footprint(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    di::AbstractVector{Int}, dj::AbstractVector{Int}, w::AbstractVector{T},
    ptr::AbstractVector{Int}, nbands::Int, strategy::AbstractMaskStrategy,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    periodic_x = FlowGeometries.Grids.isperiodic(grid, 1)
    periodic_y = FlowGeometries.Grids.isperiodic(grid, 2)
    masked = !all(FlowGeometries.Grids.mask(grid))
    invden = _banded_build_invden(grid, di, dj, w, ptr, nbands, strategy, periodic_x, periodic_y)
    # An unmasked grid convolves the caller's array directly, so it needs no buffer at all.
    scratch = BandedScratch(masked ? [zeros(T, Nx, Ny)] : Matrix{T}[])
    return FilterFootprint(di, dj, w, ptr, nbands, strategy, masked, periodic_x, periodic_y, invden, scratch)
end


"""
    ScatteredCache{T}

The full per-target-point neighbour list for a [`ScatteredFilterPlan`](@ref) (absolute neighbour
indices + weights), built only when the plan's [`AbstractCacheStrategy`](@ref) calls for it. Same
CSR-style layout as before: target `t = i + (j-1)*Nx`, entries `ptr[t]:ptr[t+1]-1`.
"""
struct ScatteredCache{T<:AbstractFloat, VI<:AbstractVector{Int}, VT<:AbstractVector{T}}
    ii::VI    # absolute neighbour axis-1 (x) index (periodic wrap already resolved)
    jj::VI    # absolute neighbour axis-2 (y) index
    w::VT       # kernel_weight(distance) * cell area
    ptr::VI   # target t = i + (j-1)*Nx; entries ptr[t]:ptr[t+1]-1
end

"""
    ScatteredFilterPlan{T,K}

Real-space footprint for a genuinely nonuniform 2D `StructuredGrid` axis, or a `CurvilinearGrid`
(exactly the `periodic_x = periodic_y = false` case of the same candidate-window/distance-gate
computation). `FilterFootprint` is a translation-invariant cache — the SAME index offset (and its
weight) is reused for every target point — which is only valid on a uniform axis; here that
assumption is false (offset `+3` means a different physical displacement depending on where you
start), so there is no way to share one offset/weight set across points. That does NOT mean the
result must be stored, though: since the search window (`di_lim`/`dj_lim`) is already a global scalar
bound (not per-point), the exact same candidate enumeration + `distance`/`kernel_weight` gate that
determines a point's neighbours can be re-run identically at apply time from these few scalars alone
— `cache` holds the materialized [`ScatteredCache`](@ref) only when the plan's cache strategy decided
to build it (see [`AbstractCacheStrategy`](@ref)), `nothing` otherwise (apply-time recomputation).
"""
struct ScatteredFilterPlan{T<:AbstractFloat, K<:Kernels.AbstractFilterKernel, C<:Union{Nothing,ScatteredCache{T}}, MT}
    kernel::K
    scale::T
    rad::T
    di_lim::Int
    dj_lim::Int
    periodic_x::Bool
    periodic_y::Bool
    x_period::T
    y_period::T
    is_cartesian::Bool
    cache::C
    topology::MT   # built once; both the cache build and the streaming apply query through it
end

# Whether a ball query on this grid should carry a spatial index is upstream's call: a separable window
# already bounds a `StructuredGrid`, and a curvilinear mesh has none.
#
# `active_only = false`, because every fold below queries for masked cells too — `ZeroFill` keeps a
# masked neighbour in the window mass and contributes zero for it, so an index built over the active
# region alone cannot answer the query.
@inline _query_topology(grid::FlowGeometries.Grids.AbstractGrid, ball) =
    FlowGeometries.Connectivity.default_sweep_topology(grid, ball, false)

# Estimated cache byte size for the `AutoCache` budget check. The builder pushes only the candidates
# inside the metric ball (`d <= rad`), so the estimate is the enclosing-box count divided by the
# box-to-ball volume ratio: 4/π in 2-D, 6/π in 3-D. That ratio is exact in the limit of many cells and
# slightly conservative at small windows, which is the safe direction for a budget check.
const _BOX_TO_BALL_2D = 4 / π
const _BOX_TO_BALL_3D = 6 / π

@inline _scattered_cache_bytes(Nx::Integer, Ny::Integer, di_lim::Integer, dj_lim::Integer, ::Type{T}) where {T} =
    round(Int, Nx * Ny * (2*di_lim + 1) * (2*dj_lim + 1) * (2*sizeof(Int) + sizeof(T)) / _BOX_TO_BALL_2D)

@inline function _should_cache(cache_strategy::AbstractCacheStrategy, est_bytes::Integer, cache_byte_budget::Integer)
    cache_strategy isa AlwaysCache && return true
    cache_strategy isa NeverCache && return false
    return est_bytes <= cache_byte_budget   # AutoCache
end

"""
    _scattered_window_bounds(grid::StructuredGrid{...,2}, rad) -> (di_lim, dj_lim, periodic_x, periodic_y, x_period, y_period, is_cartesian)

The widest window the grid's own ball query will scan, taken from `Connectivity.metric_window` rather
than re-derived here. On a rectilinear grid the window depends on the row, not the column, so one
evaluation per row covers the grid.
"""
function _scattered_window_bounds(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2}, rad::T,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    periodic_x = FlowGeometries.Grids.isperiodic(grid, 1)
    # Axis 2 is a Cartesian y (periodicity meaningful — a doubly-periodic box is standard) or a
    # spherical latitude (periodicity meaningless — wrapping past a pole is not an index wrap, so no
    # spherical grid sets it). Read it generically rather than assuming per geometry.
    periodic_y = FlowGeometries.Grids.isperiodic(grid, 2)

    _, Ny = FlowGeometries.Grids.size_tuple(grid)
    mt = FlowGeometries.Connectivity.MetricTopology(grid)
    di_lim = 0
    dj_lim = 0
    for j in 1:Ny
        w = FlowGeometries.Connectivity.metric_window(grid, (1, j), rad, mt)
        di_lim = max(di_lim, w[1])
        dj_lim = max(dj_lim, w[2])
    end

    # A wrapped candidate's raw stored coordinate sits a full period away from the target on a
    # periodic CARTESIAN axis (e.g. index Nx is `Lx` meters from index 1, not adjacent to it), so
    # the plain Euclidean `distance` below would reject every genuinely-close wrapped neighbor unless
    # shifted back by one period first. A periodic SPHERICAL axis needs no such shift: great-circle
    # distance is built from `cos`/`sin` of the raw longitude, which is already exactly 2π-periodic
    # regardless of the literal angle value. `x_period`/`y_period` mirror the same "extent + one
    # cell spacing" convention `StructuredGrid`'s own constructor uses to derive its periodic cell width.
    is_cartesian = G <: FlowGeometries.Geometry.CartesianGeometry{T}
    # The grid's stored wrap length is authoritative: `n·|Δ|` for a uniform axis, caller-supplied for a
    # stretched one, whose seam gap its samples do not determine.
    x_period = (periodic_x && is_cartesian) ? T(FlowGeometries.Grids.period(grid, 1)) : zero(T)
    y_period = (periodic_y && is_cartesian) ? T(FlowGeometries.Grids.period(grid, 2)) : zero(T)
    return di_lim, dj_lim, periodic_x, periodic_y, x_period, y_period, is_cartesian
end

# Folds `acc = f(acc, iin, jjn, d)` over every cell within `rad` of `(i, j)`, the centre included.
@inline function _scattered_foldl(
    f::F, acc, grid, target, i::Integer, j::Integer, Nx::Integer, Ny::Integer,
    di_lim::Integer, dj_lim::Integer, periodic_x::Bool, periodic_y::Bool,
    x_period::T, y_period::T, is_cartesian::Bool, rad::T, mt, scratch = nothing,
) where {F, T<:AbstractFloat}
    # The window bound, the periodic convention and the distance are all properties of the grid, so the
    # traversal is the grid's. The scalar arguments above are still taken because the plan stores them
    # for its cache-size estimate.
    return _ball_fold(acc, grid, Int(i), Int(j), rad, is_cartesian, mt, scratch) do a, J, d
        f(a, J[1], J[2], d)
    end
end

# `AllImages` is the torus convention a convolution needs: past `rad = L/2` a cell contributes through
# several images, each at its own displacement. A curvilinear grid has no axis to tile along, so its
# query takes no `images` argument.
#
# `scratch` is the candidate buffer an INDEXED query fills; without one it allocates a fresh list per
# call. A separable window has none to reuse, so the structured form ignores it. One buffer per task.
@inline _ball_fold(
    f::F, acc, grid::FlowGeometries.Grids.StructuredGrid, i::Int, j::Int, rad, is_cartesian::Bool, mt,
    scratch = nothing,
) where {F} = FlowGeometries.Connectivity.fold_within(
    f, acc, grid, i, j;
    ball = rad, self = true, active_only = false, topology = mt,
    images = is_cartesian ? FlowGeometries.Connectivity.AllImages() :
                            FlowGeometries.Connectivity.NearestImage(),
)

@inline _ball_fold(f::F, acc, grid, i::Int, j::Int, rad, ::Bool, mt, scratch = nothing) where {F} =
    FlowGeometries.Connectivity.fold_within(
        f, acc, grid, i, j;
        ball = rad, self = true, active_only = false, topology = mt, scratch = scratch,
    )

"""
    _build_footprint_scattered(grid, kernel, scale; cache_strategy=AutoCache(), cache_byte_budget=DEFAULT_CACHE_BYTE_BUDGET) -> ScatteredFilterPlan

Build the compact nonuniform-axis plan (O(1) scalar metadata) and, only if `cache_strategy` calls for
it, the full per-point [`ScatteredCache`](@ref) — correct for any spacing pattern (Cartesian or
spherical, uniform or not), since it never assumes translation invariance. The search window comes
from `_scattered_window_bounds`, i.e. the grid's own `Connectivity.metric_window`; the exact
`d <= rad` check still gates inclusion, so a loose bound only costs iterations, never a missed cell.
"""
function _build_footprint_scattered(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    cache_strategy::AbstractCacheStrategy = AutoCache(),
    cache_byte_budget::Integer = DEFAULT_CACHE_BYTE_BUDGET,
    kwargs...,   # accepts (and ignores) mask_strategy — only the separable-Gaussian path needs it
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    rad = Kernels.kernel_radius(kernel, scale)
    di_lim, dj_lim, periodic_x, periodic_y, x_period, y_period, is_cartesian =
        _scattered_window_bounds(grid, rad)
    mt = _query_topology(grid, rad)

    cache = if _should_cache(cache_strategy, _scattered_cache_bytes(Nx, Ny, di_lim, dj_lim, T), cache_byte_budget)
        ii = Int[]
        jj = Int[]
        w = T[]
        sizehint!(ii, Nx * Ny * (2*di_lim + 1) * (2*dj_lim + 1))
        sizehint!(jj, Nx * Ny * (2*di_lim + 1) * (2*dj_lim + 1))
        sizehint!(w, Nx * Ny * (2*di_lim + 1) * (2*dj_lim + 1))
        ptr = Vector{Int}(undef, Nx * Ny + 1)
        ptr[1] = 1
        sc = FlowGeometries.Connectivity.ball_scratch()
        for j in 1:Ny, i in 1:Nx # column-major target order: t = i + (j-1)*Nx, increasing monotonically
            t = i + (j - 1) * Nx
            target = FlowGeometries.Grids.coords(SA.SVector, grid, i, j)
            _scattered_foldl(
                nothing, grid, target, i, j, Nx, Ny, di_lim, dj_lim, periodic_x, periodic_y, x_period, y_period, is_cartesian, rad, mt, sc,
            ) do _, iin, jjn, d
                push!(ii, iin)
                push!(jj, jjn)
                push!(w, Kernels.kernel_weight(kernel, d, scale) * FlowGeometries.Grids.area(grid, iin, jjn))
                nothing
            end
            ptr[t+1] = length(ii) + 1
        end
        ScatteredCache(ii, jj, w, ptr)
    else
        nothing
    end
    return ScatteredFilterPlan(kernel, scale, rad, di_lim, dj_lim, periodic_x, periodic_y, x_period, y_period, is_cartesian, cache, mt)
end

"""
    _build_footprint_curvilinear(grid, kernel, scale; cache_strategy=AutoCache(), cache_byte_budget=DEFAULT_CACHE_BYTE_BUDGET) -> ScatteredFilterPlan

Compact plan (and, if `cache_strategy` calls for it, the full per-point cache) for a
`FlowGeometries.Grids.CurvilinearGrid`. Enumeration goes through the grid's own ball query, as
it does for a `StructuredGrid`; what differs is the SIZE ESTIMATE the cache budget is checked against.
`Connectivity.metric_window` bounds a window from per-axis spacing, and a curvilinear mesh has no
separable axes to bound with, so the estimate comes from the smallest adjacent-node spacing in each
index direction instead. It feeds no computation — only the `AutoCache` decision — so being loose
costs a cache that would have fit, never a wrong answer.
"""
function _build_footprint_curvilinear(
    grid::FlowGeometries.Grids.CurvilinearGrid{T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    cache_strategy::AbstractCacheStrategy = AutoCache(),
    cache_byte_budget::Integer = DEFAULT_CACHE_BYTE_BUDGET,
    kwargs...,   # accepts (and ignores) mask_strategy — irrelevant here, no separable-Gaussian path
                 # for CurvilinearGrid (non-Cartesian-only concern), but accepted for a uniform call
                 # signature with the StructuredGrid `build_footprint` methods.
) where {T<:AbstractFloat}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    geo = FlowGeometries.Grids.grid_geometry(grid)
    rad = Kernels.kernel_radius(kernel, scale)

    # Smallest adjacent-node spacing in each index direction (walk both directions of the 2D mesh) —
    # an O(N) one-time pass to learn the mesh's spacing, not part of the O(N·M) storage question.
    min_di = T(Inf)
    min_dj = T(Inf)
    for j in 1:Ny, i in 1:Nx
        c = FlowGeometries.Grids.coords(SA.SVector, grid, i, j)
        if i < Nx
            d = FlowGeometries.Geometry.distance(geo, c, FlowGeometries.Grids.coords(SA.SVector, grid, i + 1, j))
            d > 0 && (min_di = min(min_di, d))
        end
        if j < Ny
            d = FlowGeometries.Geometry.distance(geo, c, FlowGeometries.Grids.coords(SA.SVector, grid, i, j + 1))
            d > 0 && (min_dj = min(min_dj, d))
        end
    end
    di_lim = isfinite(min_di) && min_di > 0 ? ceil(Int, rad / min_di) : 0
    dj_lim = isfinite(min_dj) && min_dj > 0 ? ceil(Int, rad / min_dj) : 0
    mt = _query_topology(grid, rad)

    cache = if _should_cache(cache_strategy, _scattered_cache_bytes(Nx, Ny, di_lim, dj_lim, T), cache_byte_budget)
        ii = Int[]
        jj = Int[]
        w = T[]
        sizehint!(ii, Nx * Ny * (2*di_lim + 1) * (2*dj_lim + 1))
        sizehint!(jj, Nx * Ny * (2*di_lim + 1) * (2*dj_lim + 1))
        sizehint!(w, Nx * Ny * (2*di_lim + 1) * (2*dj_lim + 1))
        ptr = Vector{Int}(undef, Nx * Ny + 1)
        ptr[1] = 1
        sc = FlowGeometries.Connectivity.ball_scratch()
        for j in 1:Ny, i in 1:Nx # column-major target order: t = i + (j-1)*Nx
            t = i + (j - 1) * Nx
            target = FlowGeometries.Grids.coords(SA.SVector, grid, i, j)
            _scattered_foldl(
                nothing, grid, target, i, j, Nx, Ny, di_lim, dj_lim, false, false, zero(T), zero(T), false, rad, mt, sc,
            ) do _, iin, jjn, d
                push!(ii, iin)
                push!(jj, jjn)
                push!(w, Kernels.kernel_weight(kernel, d, scale) * FlowGeometries.Grids.area(grid, iin, jjn))
                nothing
            end
            ptr[t+1] = length(ii) + 1
        end
        ScatteredCache(ii, jj, w, ptr)
    else
        nothing
    end
    return ScatteredFilterPlan(kernel, scale, rad, di_lim, dj_lim, false, false, zero(T), zero(T), false, cache, mt)
end

"""
    build_footprint(grid::CurvilinearGrid, kernel, scale; kwargs...) -> ScatteredFilterPlan

Real-space direct-sum footprint for a curvilinear grid (see [`_build_footprint_curvilinear`](@ref)).
"""
build_footprint(
    grid::FlowGeometries.Grids.CurvilinearGrid{T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    kwargs...,
) where {T<:AbstractFloat} = _build_footprint_curvilinear(grid, kernel, scale; kwargs...)

"""
    FilterFootprintND{N, T}

General N-dimensional footprint: in-support neighbour offsets (`NTuple{N,Int}`) and their geometric
weights `w = kernel_weight(distance) · cell_measure`. Used for 1D and 3D (Cartesian,
translation-invariant ⇒ a single offset set); the 2D path uses the optimized per-row
`FilterFootprint`.
"""
struct FilterFootprintND{N, T<:AbstractFloat, VO<:AbstractVector{NTuple{N,Int}}, VT<:AbstractVector{T}}
    offsets::VO
    w::VT
end

"""
    build_footprint(grid, kernel, scale) -> FilterFootprint

Fast path — real multiple dispatch, not a runtime check: both axes are `AbstractRange`, a
compile-time proof of constant spacing, so the footprint is genuinely translation-invariant and can
be shared via a single (Cartesian) or per-latitude-band (spherical) offset/weight cache. Spacing is
read via `step(...)` directly from the axis that's already proven uniform by its type — not from the
geometry's separately-stored `dx`/`dy` scalar, so there's no possibility of the two disagreeing.
"""
function build_footprint(
    # `StructuredGrid{T,G,N,S,TP,C,…}`: `TP` is the per-direction topology and `C` the coordinates, so the
    # axis constraint belongs in slot FIVE. Naming `TP` rather than leaving it implicit is what keeps
    # this from silently constraining the topology instead — a signature that then matches nothing and
    # sends every grid to the general path with no error anywhere.
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2,S,TP,<:Tuple{AbstractRange,AbstractRange}},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}, S, TP<:NTuple{2,FlowGeometries.Grids.AbstractTopology}}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    rad = Kernels.kernel_radius(kernel, scale)

    di = Int[]
    dj = Int[]
    w = T[]
    ptr = Int[1]

    if G <: FlowGeometries.Geometry.CartesianGeometry{T}
        dx = step(FlowGeometries.Grids.coordinates(grid, 1))
        dy = step(FlowGeometries.Grids.coordinates(grid, 2))
        A = FlowGeometries.Grids.area(grid, 1, 1)   # uniform Cartesian cell area
        di_lim = dx > 0 ? ceil(Int, rad / dx) : 0
        dj_lim = dy > 0 ? ceil(Int, rad / dy) : 0
        # Exact window size (a single shared translation-invariant footprint, not per grid point):
        # every candidate offset in this rectangle is visited exactly once below.
        sizehint!(di, (2*di_lim + 1) * (2*dj_lim + 1))
        sizehint!(dj, (2*di_lim + 1) * (2*dj_lim + 1))
        sizehint!(w, (2*di_lim + 1) * (2*dj_lim + 1))
        for ddj in -dj_lim:dj_lim, ddi in -di_lim:di_lim
            d = sqrt((ddi * dx)^2 + (ddj * dy)^2)
            if d <= rad
                push!(di, ddi)
                push!(dj, ddj)
                push!(w, Kernels.kernel_weight(kernel, T(d), scale) * A)
            end
        end
        push!(ptr, length(di) + 1)
        return _banded_footprint(grid, di, dj, w, ptr, 1, mask_strategy)
    else
        R = FlowGeometries.Geometry.radius(FlowGeometries.Grids.grid_geometry(grid))
        dλ = step(FlowGeometries.Grids.coordinates(grid, 1))
        dφ = step(FlowGeometries.Grids.coordinates(grid, 2))
        dj_lim = dφ > 0 ? ceil(Int, rad / (R * dφ)) : 0
        # Per band, not one global bound: the longitude window widens as cosφ→0, so a pole-worst-case
        # bound over-reserves every band away from the poles. Pass 1 records each band's window and its
        # entry count; pass 2 reuses them.
        #
        # The window must hold for every row the ball reaches, not just the target's — a row nearer the
        # pole spans more longitude for the same radius — which is what `metric_window` gives, taking
        # the smallest cosφ over the latitude window. Capped at one turn besides: longitude identifies
        # rather than tiles, so a ring contributes each of its `Nx` cells at most once.
        di_lims = Vector{Int}(undef, Ny)
        total_entries = 0
        for j in 1:Ny
            dl = FlowGeometries.Connectivity.metric_window(grid, (1, j), rad)[1]
            di_lims[j] = min(dl, Nx ÷ 2)
            for ddj in -dj_lim:dj_lim
                jj = j + ddj
                (1 <= jj <= Ny) || continue
                total_entries += min(2 * dl + 1, Nx)
            end
        end
        sizehint!(di, total_entries)
        sizehint!(dj, total_entries)
        sizehint!(w, total_entries)
        for j in 1:Ny
            φ = FlowGeometries.Grids.coordinates(grid, 2)[j]
            di_lim = di_lims[j]
            for ddj in -dj_lim:dj_lim
                jj = j + ddj
                (1 <= jj <= Ny) || continue
                φ2 = FlowGeometries.Grids.coordinates(grid, 2)[jj]
                A = FlowGeometries.Grids.area(grid, 1, jj)   # spherical cell area depends only on latitude
                # Asymmetric by one when the window closes the ring: `-dl:dl` is `2dl+1` offsets, and at
                # `dl = Nx÷2` on an even ring that is `Nx+1` — the antipodal column visited from both
                # sides. Dropping the upper end leaves each of the `Nx` columns exactly once.
                hi_i = (2 * di_lim + 1 > Nx) ? di_lim - 1 : di_lim
                for ddi in -di_lim:hi_i
                    # Great-circle distance with Δλ = ddi·dλ (longitude-translation-invariant).
                    d = FlowGeometries.Geometry.distance(
                        FlowGeometries.Grids.grid_geometry(grid),
                        SA.SVector{2,T}(zero(T), φ),
                        SA.SVector{2,T}(T(ddi) * dλ, φ2),
                    )
                    if d <= rad
                        push!(di, ddi)
                        push!(dj, ddj)
                        push!(w, Kernels.kernel_weight(kernel, d, scale) * A)
                    end
                end
            end
            push!(ptr, length(di) + 1)
        end
        return _banded_footprint(grid, di, dj, w, ptr, Ny, mask_strategy)
    end
end

"""
    build_footprint(grid, kernel, scale; kwargs...) -> ScatteredFilterPlan

General path: at least one axis is a plain (non-`Range`) `AbstractVector`, which makes no type-level
uniformity guarantee — its values might happen to be evenly spaced, but nothing proves it, so no
assumption is made and the always-correct per-point plan is built instead. (Less specific than
the method above, so Julia only reaches this one when the fast method's constraint doesn't match.)
"""
function build_footprint(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    return _build_footprint_scattered(grid, kernel, scale; kwargs...)
end
