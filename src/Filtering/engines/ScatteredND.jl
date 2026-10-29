# ---------------------------------------------------------------------------
# The cell-by-cell real-space plan of a 1-D or 3-D structured grid
# ---------------------------------------------------------------------------

# Cell-by-cell engine for the 1-D and 3-D grids no more specific method takes: an axis that is not
# uniform, or a non-Cartesian geometry.
_structured_footprint(
    ::Tuple, grid::FlowGeometries.Grids.StructuredGrid{T,G,1}, kernel::Kernels.AbstractFilterKernel, scale::T;
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}} =
    _build_footprint_nd_scattered(grid, kernel, scale; kwargs...)
_structured_footprint(
    ::Tuple, grid::FlowGeometries.Grids.StructuredGrid{T,G,3}, kernel::Kernels.AbstractFilterKernel, scale::T;
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}} =
    _build_footprint_nd_scattered(grid, kernel, scale; kwargs...)

"""
    NDScatteredCache{N, T}

The full per-target-point neighbour list for an [`NDScatteredFilterPlan`](@ref) (absolute neighbour
multi-indices + weights), built only when the plan's [`AbstractCacheStrategy`](@ref) calls for it.
"""
struct NDScatteredCache{N, T<:AbstractFloat, VO<:AbstractVector{NTuple{N,Int}}, VT<:AbstractVector{T}, VI<:AbstractVector{Int}}
    nbrs::VO
    w::VT
    ptr::VI   # target t = LinearIndices(dims)[I]; entries ptr[t]:ptr[t+1]-1
end

"""
    NDRingCache{N, T}

The neighbour lists of an [`NDScatteredFilterPlan`](@ref) whose targets along a uniform periodic direction 1
share their cells and weights: one list per index of the other directions, about the target in column 1. Target
`(i, J...)` reads cell `(mod1(i + c₁ − 1, dims[1]), c₂, …)` for each stored cell `c`, `c₁ ∈ 1:dims[1]`.
"""
struct NDRingCache{N, T<:AbstractFloat, VO<:AbstractVector{NTuple{N,Int}}, VT<:AbstractVector{T}, VI<:AbstractVector{Int}}
    nbrs::VO
    w::VT
    ptr::VI   # row r = LinearIndices(dims[2:N])[J]; entries ptr[r]:ptr[r+1]-1
    dims::NTuple{N,Int}
end

# The entries target `I` (linear index `t`) reads, and entry `k`'s cell.
@inline _entries_of(c::NDScatteredCache, t::Int, ::CartesianIndex) = @inbounds c.ptr[t]:(c.ptr[t + 1] - 1)
@inline _cell_of(c::NDScatteredCache, k::Int, ::CartesianIndex) = @inbounds c.nbrs[k]
@inline function _entries_of(c::NDRingCache{N}, ::Int, I::CartesianIndex{N}) where {N}
    r = LinearIndices(Base.tail(c.dims))[Base.tail(Tuple(I))...]
    return @inbounds c.ptr[r]:(c.ptr[r + 1] - 1)
end
@inline function _cell_of(c::NDRingCache, k::Int, I::CartesianIndex)
    J = @inbounds c.nbrs[k]
    i = I[1] + J[1] - 1
    return (ifelse(i > c.dims[1], i - c.dims[1], i), Base.tail(J)...)
end

# The list of every target, as a device reads it.
_expanded(c::NDScatteredCache) = c
function _expanded(c::NDRingCache{N,T}) where {N,T}
    targets = CartesianIndices(c.dims)
    n = sum(I -> length(_entries_of(c, 0, I)), targets)
    nbrs = Vector{NTuple{N,Int}}(undef, n)
    w = Vector{T}(undef, n)
    ptr = Vector{Int}(undef, length(targets) + 1)
    ptr[1] = 1
    m = 0
    for (t, I) in enumerate(targets)
        for k in _entries_of(c, 0, I)
            m += 1
            nbrs[m] = _cell_of(c, k, I)
            w[m] = c.w[k]
        end
        ptr[t + 1] = m + 1
    end
    return NDScatteredCache(nbrs, w, ptr)
end

"""
    NDScatteredFilterPlan{N, T, K, C, MT}

N-D (1D or 3D) analog of [`ScatteredFilterPlan`](@ref), for a grid with no translation-invariant
footprint, such as one with a plain `AbstractVector` axis: the kernel, the support radius, the
per-axis index window `lim` at the worst cell, and the grid's ball-query topology. `cache` holds the
[`NDScatteredCache`](@ref), or the [`NDRingCache`](@ref) of a shell on a uniform ring, when the plan's cache
strategy builds one, `nothing` otherwise (apply-time recomputation). `exterior` is the mass past a bounded edge
that `ZeroFill` adds to each point's denominator, or `nothing`. `weights` is the rule a cell's weight comes from,
[`RadialWeights`](@ref) or [`HighOrderWeights`](@ref).
"""
struct NDScatteredFilterPlan{
    N, T<:AbstractFloat, K<:Kernels.AbstractFilterKernel, C<:Union{Nothing,NDScatteredCache{N,T},NDRingCache{N,T}}, MT,
    EX<:Union{Nothing,AbstractArray{T,N}}, W,
}
    kernel::K
    scale::T
    rad::T
    lim::NTuple{N,Int}
    cache::C
    topology::MT   # built once; the cache build and the streaming apply both query through it
    exterior::EX
    weights::W
end

"""
    _nd_rule_fold(f, acc, rule, grid, t, Ti, rad, mt) -> acc

The N-D [`_rule_fold`](@ref): `acc = f(acc, J, w)` over the cells `rule` reaches about target `Ti` (linear
index `t`), through the grid's ball query or, for an [`OverlapWeightsND`](@ref) rule, the support's own walk.
"""
@inline _nd_rule_fold(f::F, acc, rule, grid, t, Ti, rad, mt) where {F} =
    _nd_foldl(acc, grid, Ti, rad, mt) do a, J, d
        f(a, J, last(_cell_weight(rule, grid, t, Ti, J, d)))
    end

@inline _nd_rule_fold(f::F, acc, rule::OverlapWeightsND, grid, _, Ti, _, _) where {F} =
    _overlap_fold_nd(f, acc, rule, grid, Ti)

# A shell cell's weight: its measure times its fraction inside the ball, over the cells a query of the rule's
# reach returns.
@inline _nd_rule_fold(f::F, acc, rule::OverlapShells, grid, _, Ti, rad, mt) where {F} =
    _nd_foldl(acc, grid, Ti, rad, mt) do a, J, _
        fr = _shell_fraction(rule, Ti, J)
        iszero(fr) && return a
        A = _cell_measure(grid, J)
        f(a, J, A * oftype(A, fr))
    end

# Entries of an N-D cache, counted in target order, stopping once past `limit`.
function _count_nd_scattered(grid, rule, rad, mt, limit::Int)
    n = 0
    for (t, I) in enumerate(CartesianIndices(FlowGeometries.Grids.size_tuple(grid)))
        n = _nd_rule_fold((m, _, _) -> m + 1, n, rule, grid, t, Tuple(I), rad, mt)
        n > limit && break
    end
    return n
end

# The N-D cache, `n` entries from `_count_nd_scattered`, filled in linear target order.
function _fill_nd_scattered(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, rule, rad, mt, n::Int,
) where {T, G, N}
    dims = FlowGeometries.Grids.size_tuple(grid)
    nbrs = Vector{NTuple{N,Int}}(undef, n)
    w = Vector{T}(undef, n)
    ptr = Vector{Int}(undef, prod(dims) + 1)
    ptr[1] = 1
    k = 0
    for (t, I) in enumerate(CartesianIndices(dims))
        k = _nd_rule_fold(k, rule, grid, t, Tuple(I), rad, mt) do m, J, wt
            m += 1
            nbrs[m] = J
            w[m] = wt
            m
        end
        ptr[t + 1] = k + 1
    end
    return NDScatteredCache(nbrs, w, ptr)
end

# An N-D cell-by-cell plan's neighbour cache (or `nothing`), its mass past a bounded edge (or `nothing`)
# and its weight rule, as `_scattered_parts` gives them in two dimensions.
function _nd_scattered_parts(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, kernel::Kernels.AbstractFilterKernel, scale::T,
    rad, mt, strategy, budget,
) where {T, G, N}
    rule = RadialWeights(kernel, scale, Val(N))
    return _nd_cache(grid, rule, rad, mt, strategy, budget), _exterior_mass(grid, kernel, scale), rule
end

function _nd_scattered_parts(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, kernel::Kernels.TopHatKernel{Kernels.CellOverlap}, scale::T,
    rad, mt, strategy, budget,
) where {T, N, G<:FlowGeometries.Geometry.AbstractCartesianGeometry{T}}
    rule = OverlapWeightsND(grid, rad)
    return _nd_cache(grid, rule, rad, mt, strategy, budget), _overlap_exterior(grid, rule, kernel, scale), rule
end

# A sphere's or spheroid's single ring of longitude, the equator.
function _nd_scattered_parts(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,1}, kernel::Kernels.TopHatKernel{Kernels.CellOverlap}, scale::T,
    rad, mt, strategy, budget,
) where {T, G<:FlowGeometries.Geometry.AbstractLonLatGeometry{T}}
    rule = OverlapWeightsND(grid, rad)
    return _nd_cache(grid, rule, rad, mt, strategy, budget), _overlap_exterior(grid, rule, kernel, scale), rule
end

# A sphere with a radius: the ball of the chord distance over shell sectors.
function _nd_scattered_parts(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,3}, kernel::Kernels.TopHatKernel{Kernels.CellOverlap}, scale::T,
    rad, mt, strategy, budget,
) where {T, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}}
    rule = OverlapShells(grid, Kernels.kernel_radius(kernel, scale))
    return _nd_cache(grid, rule, rad, mt, strategy, budget), _overlap_exterior(grid, rule, kernel, scale, rad, mt), rule
end

_nd_cache(grid::FlowGeometries.Grids.StructuredGrid, rule, rad, mt, strategy, budget) =
    _nd_cache_each(grid, rule, rad, mt, strategy, budget)

# The cache folded about every target.
function _nd_cache_each(grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, rule, rad, mt, strategy, budget) where {T,G,N}
    n = _cache_entries(strategy, budget, _cache_entry_bytes(Val(N), T)) do limit
        _count_nd_scattered(grid, rule, rad, mt, limit)
    end
    return n === nothing ? nothing : _fill_nd_scattered(grid, rule, rad, mt, n)
end

# On a uniform periodic ring of longitude a shell target's cells and weights are its row and level's shifted along
# longitude, so each (row, level) folds once.
function _nd_cache(grid::FlowGeometries.Grids.StructuredGrid{T,G,3}, rule::OverlapShells, rad, mt, strategy, budget) where {T,G}
    _shell_ring(grid) || return _nd_cache_each(grid, rule, rad, mt, strategy, budget)
    dims = FlowGeometries.Grids.size_tuple(grid)
    lists = _shell_row_lists(grid, rule, rad, mt)
    n = _cache_entries(strategy, budget, _cache_entry_bytes(Val(3), T)) do _
        sum(length, lists)
    end
    n === nothing && return nothing
    nbrs = Vector{NTuple{3,Int}}(undef, n)
    w = Vector{T}(undef, n)
    ptr = Vector{Int}(undef, length(lists) + 1)
    ptr[1] = 1
    m = 0
    for (r, list) in enumerate(lists)
        for (J, wt) in list
            m += 1
            nbrs[m] = J
            w[m] = wt
        end
        ptr[r + 1] = m + 1
    end
    return NDRingCache(nbrs, w, ptr, dims)
end

@inline _shell_ring(grid) = FlowGeometries.Grids.isperiodic(grid, 1) && FlowGeometries.Grids.isuniform(grid, 1)

# The cells and weights about column 1 of each (row, level).
function _shell_row_lists(grid::FlowGeometries.Grids.StructuredGrid{T,G,3}, rule::OverlapShells, rad, mt) where {T,G}
    _, Ny, Nz = FlowGeometries.Grids.size_tuple(grid)
    return [_nd_rule_fold((acc, J, wt) -> (push!(acc, (J, wt)); acc), Tuple{NTuple{3,Int},T}[], rule, grid, 0, (1, j, k), rad, mt)
            for j in 1:Ny, k in 1:Nz]
end

function _nd_scattered_parts(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, kernel::Kernels.HighOrderKernel, scale::T,
    rad, mt, strategy, budget,
) where {T, G, N}
    fold = (f, acc, I) -> _nd_foldl(f, acc, grid, I, rad, mt)
    keep = _cache_keep(strategy, budget, _cache_entry_bytes(Val(N), T))
    weights, exterior, kept = _highorder_targets(grid, kernel, scale, rad, fold, _ho_lattice(grid, rad), keep)
    cache = kept === nothing ? nothing : NDScatteredCache(kept...)
    return cache, exterior, weights
end

function _build_footprint_nd_scattered(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    cache_strategy::AbstractCacheStrategy = AutoCache(),
    cache_byte_budget::Integer = default_cache_byte_budget(),
    kwargs...,   # absorbs and ignores the other engines' keywords, such as `mask_strategy`
) where {N, T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    rad = _query_radius(grid, kernel, scale)
    # Per-axis index bound at the worst cell; the traversal is the grid's.
    lim = FlowGeometries.Connectivity.metric_window(grid, rad)
    mt = _query_topology(grid, rad)
    cache, exterior, weights = _nd_scattered_parts(grid, kernel, scale, rad, mt, cache_strategy, cache_byte_budget)
    # `cache` and `exterior` are typed `Union{Nothing,…}`, so the plan's type depends on the grid and
    # the kernel alone.
    return NDScatteredFilterPlan{
        N, T, typeof(kernel),
        Union{Nothing,NDScatteredCache{N,T,Vector{NTuple{N,Int}},Vector{T},Vector{Int}},
              NDRingCache{N,T,Vector{NTuple{N,Int}},Vector{T},Vector{Int}}}, typeof(mt),
        Union{Nothing,Array{T,N}}, typeof(weights),
    }(kernel, scale, rad, lim, cache, mt, exterior, weights)
end

# Folds `acc = f(acc, J, d)` over every in-support neighbour of `Ti`, the centre included, through the
# grid's own ball query under its image convention.
@inline _nd_foldl(
    f::F, acc, grid::FlowGeometries.Grids.StructuredGrid, Ti::NTuple{N,Int}, rad, mt,
) where {F, N} = FlowGeometries.Connectivity.fold_within(
    f, acc, grid, Ti...;
    ball = rad, self = true, active_only = false, topology = mt, images = _image_convention(grid),
)

# The two point functions below return zero at a target the strategy skips, so a caller evaluates them
# at every point. The parallel backend extensions call them too.
@inline function _footprint_nd_point_cached(
    field::AbstractArray, cache::Union{NDScatteredCache{N,T},NDRingCache{N,T}}, exterior, strategy::AbstractMaskStrategy,
    mask, lin::LinearIndices{N}, I::CartesianIndex{N},
) where {N, T<:AbstractFloat}
    _filters_target(strategy, @inbounds mask[I]) || return zero(T)
    ws = zero(T)
    wn = zero(T)
    zerofill = strategy isa ZeroFill
    @inbounds for k in _entries_of(cache, lin[I], I)
        J = _cell_of(cache, k, I)
        active = mask[J...]
        wk = cache.w[k]
        if zerofill
            wn += wk
            active && (ws += wk * field[J...])
        elseif active
            wn += wk
            ws += wk * field[J...]
        end
    end
    zerofill && (wn = _add_exterior(wn, exterior, I))
    return _emit(field, ws, wn)
end

# Per-point recompute without a cache, over the enumeration the cache builder uses. The accumulator
# travels through the fold's return value, so the closure mutates no captured variable.
function _footprint_nd_point_streaming(
    field::AbstractArray, grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, fp::NDScatteredFilterPlan{N,T},
    strategy::AbstractMaskStrategy, mask, I::CartesianIndex{N},
) where {N, T<:AbstractFloat, G}
    _filters_target(strategy, @inbounds mask[I]) || return zero(T)
    rule = fp.weights
    Ti = Tuple(I)
    t = LinearIndices(FlowGeometries.Grids.size_tuple(grid))[I]
    zerofill = strategy isa ZeroFill
    ws, wn = _nd_rule_fold((zero(T), zero(T)), rule, grid, t, Ti, fp.rad, fp.topology) do acc, J, wk
        s, n = acc
        active = mask[J...]
        if zerofill
            return (active ? s + wk * field[J...] : s, n + wk)
        else
            active || return acc
            return (s + wk * field[J...], n + wk)
        end
    end
    zerofill && (wn = _add_exterior(wn, fp.exterior, I))
    return _emit(field, ws, wn)
end

function apply_footprint_nd!(
    out::AbstractArray{T,N},
    field::AbstractArray,
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N},
    fp::NDScatteredFilterPlan{N,T},
    strategy::AbstractMaskStrategy,
) where {N, T<:AbstractFloat, G}
    dims = FlowGeometries.Grids.size_tuple(grid)
    mask = FlowGeometries.Grids.mask(grid)
    if fp.cache !== nothing
        lin = LinearIndices(dims)
        cache = fp.cache
        @inbounds for I in CartesianIndices(out)
            out[I] = _footprint_nd_point_cached(field, cache, fp.exterior, strategy, mask, lin, I)
        end
    else
        @inbounds for I in CartesianIndices(out)
            out[I] = _footprint_nd_point_streaming(field, grid, fp, strategy, mask, I)
        end
    end
    return out
end
