# ---------------------------------------------------------------------------
# The cell-by-cell real-space plan, and the weight and normalization rules the real-space engines share
# ---------------------------------------------------------------------------

# The cell measure, under whichever index the grid names a cell by.
@inline _cell_measure(grid, J::Tuple) = FlowGeometries.Grids.area(grid, J...)
@inline _cell_measure(grid, j::Integer) = FlowGeometries.Grids.measure(grid, j)

"""
    _query_radius(grid, kernel, scale) -> T

The radius a cell-by-cell engine queries about each target: the kernel's truncation radius, and for a
`HighOrderKernel` the radius of the ball holding every cell that meets its support.
"""
@inline _query_radius(
    grid::FlowGeometries.Grids.AbstractGrid, kernel::Kernels.AbstractFilterKernel, scale,
) = Kernels.kernel_radius(kernel, scale, _kernel_dim(grid))

"""
    RadialWeights{D,K,T}

A radial kernel's weight on a cell: `kernel_weight(d)` in `D` dimensions times the cell measure.
"""
struct RadialWeights{D, K<:Kernels.AbstractFilterKernel, T<:AbstractFloat}
    kernel::K
    scale::T
end

RadialWeights(kernel::K, scale::T, ::Val{D}) where {D, K<:Kernels.AbstractFilterKernel, T<:AbstractFloat} =
    RadialWeights{D,K,T}(kernel, scale)

# The rule with its correction table replaced by `coeffs` (a device array); a radial rule has none.
_with_coeffs(r::RadialWeights, _) = r

"""
    _cell_weight(rule, grid, t, I, J, d) -> (inside::Bool, w)

Cell `J`'s weight about target `I`, whose linear index is `t` and distance `d`, under a
[`RadialWeights`](@ref) or [`HighOrderWeights`](@ref) rule. `inside` is false, and `w` zero, for a cell
outside a `HighOrderKernel`'s support box.
"""
@inline _cell_weight(r::RadialWeights{D,K,T}, grid, _, _, J, d) where {D,K,T} =
    (true, Kernels.kernel_weight(r.kernel, T(d), r.scale, Val(D)) * _cell_measure(grid, J))

"""
    ScatteredCache{T}

The full per-target neighbour list for a [`ScatteredFilterPlan`](@ref) (absolute neighbour indices and
weights), built only when the plan's [`AbstractCacheStrategy`](@ref) calls for it. CSR layout: target
`t = i + (j-1)*Nx` holds entries `ptr[t]:ptr[t+1]-1`.
"""
struct ScatteredCache{T<:AbstractFloat, VI<:AbstractVector{Int}, VT<:AbstractVector{T}}
    ii::VI    # absolute neighbour axis-1 (x) index (periodic wrap already resolved)
    jj::VI    # absolute neighbour axis-2 (y) index
    w::VT       # the cell's weight under the plan's rule, `_cell_weight`'s `w`
    ptr::VI
end

"""
    ScatteredFilterPlan{T,K}

Cell-by-cell real-space plan for a 2-D `StructuredGrid` the banded [`FilterFootprint`](@ref) does not
cover, or a `CurvilinearGrid`. A target's neighbours are the cells within `rad` that the grid's ball query
returns through `topology`; `di_lim`/`dj_lim` bound that query's index window. `cache` holds them as a
[`ScatteredCache`](@ref) when the plan's [`AbstractCacheStrategy`](@ref) builds one, and is `nothing`
when the apply recomputes them. `weights` is the rule a cell's weight comes from,
[`RadialWeights`](@ref) or [`HighOrderWeights`](@ref).
"""
struct ScatteredFilterPlan{
    T<:AbstractFloat, K<:Kernels.AbstractFilterKernel, C<:Union{Nothing,ScatteredCache{T}}, MT,
    EX<:Union{Nothing,AbstractMatrix{T}}, W,
}
    kernel::K
    scale::T
    rad::T
    di_lim::Int
    dj_lim::Int
    periodic_x::Bool
    periodic_y::Bool
    x_period::T
    y_period::T
    cache::C
    topology::MT   # ball-query topology for both the cache build and the streaming apply
    exterior::EX   # mass past a bounded edge, added to `ZeroFill`'s denominator, or nothing
    weights::W
end

# The topology `FlowGeometries` picks for a ball query of radius `ball`, over every cell
# (`active_only = false`): `ZeroFill` counts a masked neighbour in the window mass.
@inline _query_topology(grid::FlowGeometries.Grids.AbstractGrid, ball) =
    FlowGeometries.Connectivity.default_sweep_topology(grid, ball, false)

# Bytes one cache entry holds: its neighbour index, one `Int` per axis, and its weight.
@inline _cache_entry_bytes(::Val{N}, ::Type{T}) where {N,T} = N * sizeof(Int) + sizeof(T)

"""
    _cache_entries(count, strategy, budget, entry_bytes) -> Union{Int,Nothing}

The number of entries a neighbour cache holds, or `nothing` when the plan streams. `count(limit)` returns
the number of entries the cache build's ball query yields, and may stop at any count above `limit`.
`AutoCache` sets `limit` to the budget in entries.
"""
function _cache_entries(
    count::F, strategy::AbstractCacheStrategy, budget::Integer, entry_bytes::Integer,
) where {F}
    strategy isa NeverCache && return nothing
    limit = strategy isa AlwaysCache ? typemax(Int) : Int(budget) ÷ entry_bytes
    n = count(limit)
    return n > limit ? nothing : n
end

"""
    _rule_fold(f, acc, rule, grid, t, I, rad, mt, sc) -> acc

Fold `acc = f(acc, J, w)` over the cells the plan's weight `rule` reaches about target `I` (linear index `t`),
`w` cell `J`'s weight: through the grid's ball query of radius `rad` for a [`RadialWeights`](@ref) or
[`HighOrderWeights`](@ref) rule, by the support's own walk for an [`OverlapWeights`](@ref) one.
"""
@inline _rule_fold(f::F, acc, rule, grid, t, I, rad, mt, sc) where {F} =
    _ball_fold(acc, grid, I[1], I[2], rad, mt, sc) do a, J, d
        f(a, J, last(_cell_weight(rule, grid, t, I, J, d)))
    end

@inline _rule_fold(f::F, acc, rule::OverlapWeights, grid, _, I, _, _, _) where {F} =
    _overlap_fold(f, acc, rule, grid, I[1], I[2])

# A curvilinear cell's weight: its measure times its polygon's fraction inside the support, over the cells a
# query of the rule's reach returns.
@inline function _rule_fold(f::F, acc, rule::OverlapPolygons, grid, _, I, rad, mt, sc) where {F}
    p = FlowGeometries.Grids.coords(SA.SVector, grid, I[1], I[2])
    return _ball_fold(acc, grid, I[1], I[2], rad, mt, sc) do a, J, _
        fr = _polygon_fraction(rule, J, p, FlowGeometries.Grids.coords(SA.SVector, grid, J[1], J[2]))
        iszero(fr) && return a
        A = _cell_measure(grid, J)
        f(a, J, A * oftype(A, fr))
    end
end

# A `CellOverlap` top-hat's query on a curvilinear grid or a sphere with a radius reaches every cell its support
# touches.
_query_radius(grid::FlowGeometries.Grids.CurvilinearGrid{T}, kernel::Kernels.TopHatKernel{Kernels.CellOverlap}, scale) where {T} =
    T(OverlapPolygons(grid, Kernels.kernel_radius(kernel, scale)).reach)
function _query_radius(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,3}, kernel::Kernels.TopHatKernel{Kernels.CellOverlap}, scale,
) where {T, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}}
    F = _overlap_float(T)
    faces = ntuple(d -> _cell_faces(grid, d, F), Val(3))
    x = ntuple(d -> F.(collect(FlowGeometries.Grids.coordinates(grid, d))), Val(3))
    return T(F(Kernels.kernel_radius(kernel, scale)) + _shell_extent(grid, faces, x))
end

# Entries of a 2-D cache, counted in target order, stopping once past `limit`.
function _count_scattered(grid, rule, rad, mt, sc, limit::Int)
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    n = 0
    for j in 1:Ny, i in 1:Nx
        n = _rule_fold((m, _, _) -> m + 1, n, rule, grid, i + (j - 1) * Nx, (i, j), rad, mt, sc)
        n > limit && break
    end
    return n
end

# The 2-D cache, `n` entries from `_count_scattered`, filled in target order `t = i + (j-1)·Nx`.
function _fill_scattered(grid::FlowGeometries.Grids.AbstractGrid{G,T}, rule, rad, mt, sc, n::Int) where {G,T}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    ii = Vector{Int}(undef, n)
    jj = Vector{Int}(undef, n)
    w = Vector{T}(undef, n)
    ptr = Vector{Int}(undef, Nx * Ny + 1)
    ptr[1] = 1
    k = 0
    for j in 1:Ny, i in 1:Nx
        t = i + (j - 1) * Nx
        k = _rule_fold(k, rule, grid, t, (i, j), rad, mt, sc) do m, J, wt
            m += 1
            ii[m] = J[1]
            jj[m] = J[2]
            w[m] = wt
            m
        end
        ptr[t + 1] = k + 1
    end
    return ScatteredCache(ii, jj, w, ptr)
end

# A 2-D cell-by-cell plan's neighbour cache (or `nothing`), its mass past a bounded edge (or `nothing`)
# and its weight rule.
function _scattered_parts(
    grid, kernel::Kernels.AbstractFilterKernel, scale::T, rad, mt, strategy, budget,
) where {T}
    rule = RadialWeights(kernel, scale, _kernel_dim(grid))
    cache = _scattered_cache(grid, rule, rad, mt, strategy, budget)
    return cache, _plan_exterior(grid, kernel, scale), rule
end

function _scattered_parts(
    grid::FlowGeometries.Grids.StructuredGrid, kernel::Kernels.TopHatKernel{Kernels.CellOverlap}, scale::T,
    rad, mt, strategy, budget,
) where {T}
    rule = OverlapWeights(grid, rad)
    cache = _scattered_cache(grid, rule, rad, mt, strategy, budget)
    return cache, _overlap_exterior(grid, rule, kernel, scale), rule
end

function _scattered_parts(
    grid::FlowGeometries.Grids.CurvilinearGrid{T,G,2}, kernel::Kernels.TopHatKernel{Kernels.CellOverlap}, scale::T,
    rad, mt, strategy, budget,
) where {T, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    rule = OverlapPolygons(grid, Kernels.kernel_radius(kernel, scale))
    cache = _scattered_cache(grid, rule, rad, mt, strategy, budget)
    return cache, _polygon_exterior(grid, rule, rad, mt), rule
end

# `ZeroFill`'s mass past the edges of a curvilinear grid: the support's measure less the weights a target's
# window sums over the grid's own cells.
function _polygon_exterior(grid::FlowGeometries.Grids.CurvilinearGrid{T}, rule::OverlapPolygons, rad, mt) where {T}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    M = T(_support_mass(rule.ov, 2))
    sc = FlowGeometries.Connectivity.ball_scratch()
    E = Matrix{T}(undef, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        E[i, j] = M - _rule_fold((a, _, w) -> a + w, zero(T), rule, grid, 0, (i, j), rad, mt, sc)
    end
    return E
end

function _scattered_cache(grid::FlowGeometries.Grids.AbstractGrid{G,T}, rule, rad, mt, strategy, budget) where {G,T}
    sc = FlowGeometries.Connectivity.ball_scratch()
    n = _cache_entries(strategy, budget, _cache_entry_bytes(Val(2), T)) do limit
        _count_scattered(grid, rule, rad, mt, sc, limit)
    end
    return n === nothing ? nothing : _fill_scattered(grid, rule, rad, mt, sc, n)
end

function _scattered_parts(
    grid, kernel::Kernels.HighOrderKernel, scale::T, rad, mt, strategy, budget,
) where {T}
    sc = FlowGeometries.Connectivity.ball_scratch()
    fold = (f, acc, I) -> _ball_fold(f, acc, grid, I[1], I[2], rad, mt, sc)
    keep = _cache_keep(strategy, budget, _cache_entry_bytes(Val(2), T))
    weights, exterior, kept = _highorder_targets(grid, kernel, scale, rad, fold, _ho_lattice(grid, rad), keep)
    cache = kept === nothing ? nothing :
        ScatteredCache(map(first, kept[1]), map(last, kept[1]), kept[2], kept[3])
    return cache, exterior, weights
end

# A rectilinear grid continues its lattice past a bounded edge; a curvilinear mesh carries none.
_plan_exterior(grid::FlowGeometries.Grids.StructuredGrid, kernel, scale) = _exterior_mass(grid, kernel, scale)
_plan_exterior(_, _, _) = nothing

"""
    _scattered_window_bounds(grid::StructuredGrid{...,2}, rad) -> (di_lim, dj_lim, periodic_x, periodic_y, x_period, y_period)

The widest index window the grid's ball query scans, from `Connectivity.metric_window`, with the
periodic flags and the Cartesian wrap lengths. On a rectilinear grid the window depends on the row alone,
so one evaluation per row covers the grid.
"""
function _scattered_window_bounds(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2}, rad::T,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    periodic_x = FlowGeometries.Grids.isperiodic(grid, 1)
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

    is_cartesian = G <: FlowGeometries.Geometry.CartesianGeometry{T}
    # The grid's stored wrap length on a periodic Cartesian axis (`n·|Δ|` when uniform, caller-supplied
    # when stretched); zero on a sphere, whose great-circle distance is 2π-periodic in longitude.
    x_period = (periodic_x && is_cartesian) ? T(FlowGeometries.Grids.period(grid, 1)) : zero(T)
    y_period = (periodic_y && is_cartesian) ? T(FlowGeometries.Grids.period(grid, 2)) : zero(T)
    return di_lim, dj_lim, periodic_x, periodic_y, x_period, y_period
end

# Folds `acc = f(acc, J, d)` over every cell within `rad` of `(i, j)`, the centre included, under the
# grid's image convention. `scratch` is an indexed query's candidate buffer, one per task; the structured
# form ignores it.
@inline _ball_fold(
    f::F, acc, grid::FlowGeometries.Grids.StructuredGrid, i::Int, j::Int, rad, mt, scratch = nothing,
) where {F} = FlowGeometries.Connectivity.fold_within(
    f, acc, grid, i, j;
    ball = rad, self = true, active_only = false, topology = mt, images = _image_convention(grid),
)

@inline _ball_fold(f::F, acc, grid, i::Int, j::Int, rad, mt, scratch = nothing) where {F} =
    FlowGeometries.Connectivity.fold_within(
        f, acc, grid, i, j;
        ball = rad, self = true, active_only = false, topology = mt, scratch = scratch,
    )

# A periodic Cartesian direction tiles, so each image of a cell inside the support comes at its own
# displacement; a periodic angle identifies, so its cells come once.
@inline _image_convention(grid::FlowGeometries.Grids.StructuredGrid) =
    _image_convention(FlowGeometries.Grids.grid_geometry(grid))
@inline _image_convention(::FlowGeometries.Geometry.AbstractCartesianGeometry) =
    FlowGeometries.Connectivity.AllImages()
@inline _image_convention(::FlowGeometries.Geometry.AbstractGeometry) =
    FlowGeometries.Connectivity.NearestImage()

"""
    _build_footprint_scattered(grid, kernel, scale; cache_strategy=AutoCache(), cache_byte_budget=default_cache_byte_budget()) -> ScatteredFilterPlan

The [`ScatteredFilterPlan`](@ref) of a 2-D `StructuredGrid` of any spacing and geometry, with a
[`ScatteredCache`](@ref) when `cache_strategy` calls for one. Other keywords, `mask_strategy` among them,
are ignored: the plan takes its mask strategy at apply time.
"""
function _build_footprint_scattered(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    cache_strategy::AbstractCacheStrategy = AutoCache(),
    cache_byte_budget::Integer = default_cache_byte_budget(),
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    rad = _query_radius(grid, kernel, scale)
    di_lim, dj_lim, periodic_x, periodic_y, x_period, y_period = _scattered_window_bounds(grid, rad)
    mt = _query_topology(grid, rad)
    cache, exterior, weights = _scattered_parts(grid, kernel, scale, rad, mt, cache_strategy, cache_byte_budget)
    return _scattered_plan(kernel, scale, rad, di_lim, dj_lim, periodic_x, periodic_y, x_period, y_period,
                           cache, mt, exterior, weights)
end

# The plan with its cache and its exterior mass each held as `nothing` or an array under one field type,
# so the plan's type is fixed by the grid and the kernel.
@inline _scattered_plan(
    kernel::K, scale::T, rad::T, di_lim, dj_lim, periodic_x, periodic_y, x_period, y_period,
    cache, mt::MT, exterior, weights::W,
) where {K,T,MT,W} = ScatteredFilterPlan{
    T, K, Union{Nothing,ScatteredCache{T,Vector{Int},Vector{T}}}, MT, Union{Nothing,Matrix{T}}, W,
}(kernel, scale, rad, di_lim, dj_lim, periodic_x, periodic_y, x_period, y_period, cache, mt, exterior, weights)

"""
    _build_footprint_curvilinear(grid, kernel, scale; cache_strategy=AutoCache(), cache_byte_budget=default_cache_byte_budget()) -> ScatteredFilterPlan

The [`ScatteredFilterPlan`](@ref) of a `FlowGeometries.Grids.CurvilinearGrid`, with its cache when
`cache_strategy` calls for one. `di_lim`/`dj_lim` are the index window the smallest adjacent-node
spacing in each direction implies. Other keywords are ignored, as in
[`_build_footprint_scattered`](@ref).
"""
function _build_footprint_curvilinear(
    grid::FlowGeometries.Grids.CurvilinearGrid{T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    cache_strategy::AbstractCacheStrategy = AutoCache(),
    cache_byte_budget::Integer = default_cache_byte_budget(),
    kwargs...,
) where {T<:AbstractFloat}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    geo = FlowGeometries.Grids.grid_geometry(grid)
    rad = _query_radius(grid, kernel, scale)

    # Smallest adjacent-node spacing in each index direction.
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
    cache, exterior, weights = _scattered_parts(grid, kernel, scale, rad, mt, cache_strategy, cache_byte_budget)
    return _scattered_plan(kernel, scale, rad, di_lim, dj_lim, false, false, zero(T), zero(T),
                           cache, mt, exterior, weights)
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
    build_footprint(grid::StructuredGrid, kernel, scale; kwargs...)

The real-space footprint of a structured grid, chosen by the spacing of its directions
(`FlowGeometries.Grids.spacing_trait`, a periodic direction's wrap included), its geometry and the kernel:
a uniform direction 1 takes the engines shared along it, and the cell-by-cell plan takes what no more
specific engine covers.
"""
build_footprint(
    grid::FlowGeometries.Grids.StructuredGrid{T}, kernel::Kernels.AbstractFilterKernel, scale::T; kwargs...,
) where {T<:AbstractFloat} = _with_spacings(sp -> _structured_footprint(sp, grid, kernel, scale; kwargs...), grid)

# `f(sp)`, `sp` each direction's `UniformSpacing()` or `NonuniformSpacing()` with a periodic direction's
# wrap included, the tuple the engines dispatch on; `f(nothing)` on a grid without structured directions. A
# periodic range other than a `PeriodicUniformAxis` shares its type with ranges of other periods, so
# `Grids.isuniform` compares its period with its `n·|Δ|`, and each answer calls `f` from its own branch;
# every other direction's types answer.
@inline _with_spacings(f::F, ::FlowGeometries.Grids.AbstractGrid) where {F} = f(nothing)
@inline _with_spacings(f::F, grid::FlowGeometries.Grids.StructuredGrid) where {F} =
    _spaced(f, map(_spacing, FlowGeometries.Grids.topology(grid), FlowGeometries.Grids.coordinates(grid)), grid)

# A direction's spacing where its types answer, `_WrapCheck()` where the period decides.
struct _WrapCheck end
@inline _spacing(::FlowGeometries.Grids.Bounded, x) = FlowGeometries.Axes.spacing_trait(x)
@inline _spacing(::FlowGeometries.Grids.Periodic, ::FlowGeometries.Axes.PeriodicUniformAxis) =
    FlowGeometries.Axes.UniformSpacing()
@inline _spacing(::FlowGeometries.Grids.Periodic, ::AbstractRange) = _WrapCheck()
@inline _spacing(::FlowGeometries.Grids.Periodic, ::AbstractArray) = FlowGeometries.Axes.NonuniformSpacing()

# `g` of direction `d`'s spacing: the answer itself, or each answer of the wrap check from its own branch.
@inline _pick(g::F, s, _, ::Int) where {F} = g(s)
@inline _pick(g::F, ::_WrapCheck, grid, d::Int) where {F} =
    FlowGeometries.Grids.isuniform(grid, d) ? g(FlowGeometries.Axes.UniformSpacing()) :
                                              g(FlowGeometries.Axes.NonuniformSpacing())

@inline _spaced(f::F, k::Tuple{Any}, grid) where {F} = _pick(a -> f((a,)), k[1], grid, 1)
@inline _spaced(f::F, k::Tuple{Any,Any}, grid) where {F} =
    _pick(a -> _pick(b -> f((a, b)), k[2], grid, 2), k[1], grid, 1)
@inline _spaced(f::F, k::Tuple{Any,Any,Any}, grid) where {F} =
    _pick(a -> _pick(b -> _pick(c -> f((a, b, c)), k[3], grid, 3), k[2], grid, 2), k[1], grid, 1)


# The cell-by-cell plan ([`_build_footprint_scattered`](@ref)) of a 2-D grid no more specific engine
# covers, such as one whose axis 1 is not uniform.
_structured_footprint(
    ::Tuple, grid::FlowGeometries.Grids.StructuredGrid{T,G,2}, kernel::Kernels.AbstractFilterKernel, scale::T;
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}} =
    _build_footprint_scattered(grid, kernel, scale; kwargs...)

# `wn` plus the exterior mass at a point, for `ZeroFill`'s full-mass denominator.
@inline _add_exterior(wn, ::Nothing, I...) = wn
@inline _add_exterior(wn, E::AbstractArray, I...) = wn + @inbounds(E[I...])

# The dimension a kernel's real-space weight is taken in: the grid's coordinate count.
@inline _kernel_dim(grid::FlowGeometries.Grids.AbstractGrid) = Val(FlowGeometries.Grids.ncoordinates(grid))

# The weighted mean `ws / wn`, zero where no weight was summed. A signed kernel's window mass keeps its
# sign.
@inline _normalized(ws, wn) = iszero(wn) ? zero(ws) : ws / wn
@inline _inv_mass(m) = iszero(m) ? zero(m) : inv(m)

"""
    _MassOnly{T}(dims)

The field a cell-by-cell engine applies over to write each target's window mass: its loops pass their sums to
`_emit`, which returns the summed weight for it and the weighted mean for any other field. Its values are
never read into the output.
"""
struct _MassOnly{T,N} <: AbstractArray{T,N}
    dims::NTuple{N,Int}
end
_MassOnly{T}(dims::NTuple{N,Int}) where {T,N} = _MassOnly{T,N}(dims)
Base.size(m::_MassOnly) = m.dims
Base.@propagate_inbounds Base.getindex(::_MassOnly{T,N}, ::Vararg{Int,N}) where {T,N} = zero(T)

@inline _emit(::AbstractArray, ws, wn) = _normalized(ws, wn)
@inline _emit(::_MassOnly, _, wn) = wn

# Whether a target is filtered: every one under `ZeroFill`, the active ones under `Deformable`.
@inline _filters_target(::ZeroFill, ::Bool) = true
@inline _filters_target(::AbstractMaskStrategy, active::Bool) = active

# Whether `strategy`'s window mass is the kernel's mass over every in-domain offset whatever the mask holds:
# `ZeroFill` counts every in-support cell, and `Deformable` counts the active ones, which a mask of type
# `AllActive` alone fixes. Answered from the types.
@inline _mask_free_mass(::ZeroFill, ::AbstractArray{Bool}) = true
@inline _mask_free_mass(::AbstractMaskStrategy, mask::AbstractArray{Bool}) = mask isa FlowGeometries.Grids.AllActive

"""
    apply_footprint!(out, field, grid, fp::ScatteredFilterPlan, strategy, periodic_x, periodic_y)

Whole-grid convolve using a [`ScatteredFilterPlan`](@ref), the engine for nonuniform-axis and
curvilinear grids. Periodicity comes from `fp`; `periodic_x`/`periodic_y` are unused.
"""
function apply_footprint!(
    out::AbstractMatrix{T},
    field::AbstractMatrix,
    grid::FlowGeometries.Grids.AbstractGrid,
    fp::ScatteredFilterPlan{T},
    strategy::AbstractMaskStrategy,
    periodic_x::Bool,
    periodic_y::Bool,
) where {T<:AbstractFloat}
    _, Ny = FlowGeometries.Grids.size_tuple(grid)
    fill!(out, zero(T))
    sc = row_scratch(fp)
    for j in 1:Ny
        apply_footprint_row!(out, field, grid, fp, strategy, periodic_x, periodic_y, j, sc)
    end
    return out
end

"""
    row_scratch(fp) -> scratch or nothing

The candidate buffer a row apply's ball queries reuse, the last argument of `apply_footprint_row!` and
`apply_footprint_row_batch!`: one per concurrent worker. `nothing` for an engine whose rows query nothing.
"""
row_scratch(_) = nothing
row_scratch(fp::ScatteredFilterPlan) = FlowGeometries.Connectivity.ball_scratch(fp.topology)

"""
    apply_footprint_row!(out, field, grid, fp::ScatteredFilterPlan, strategy, periodic_x, periodic_y, j, sc = row_scratch(fp))

Fill output row `j` from a [`ScatteredFilterPlan`](@ref): if `fp.cache !== nothing`, read the
precomputed per-point neighbour list (absolute `(ii,jj)` indices, periodic wrap resolved at build
time); otherwise recompute each point's neighbours and weights through the grid's ball query, the
enumeration the cache was built from, reusing `sc`. A target the strategy skips is left unwritten.
"""
function apply_footprint_row!(
    out::AbstractMatrix{T},
    field::AbstractMatrix,
    grid::FlowGeometries.Grids.AbstractGrid,
    fp::ScatteredFilterPlan{T},
    strategy::AbstractMaskStrategy,
    periodic_x::Bool,
    periodic_y::Bool,
    j::Integer,
    sc = row_scratch(fp),
) where {T<:AbstractFloat}
    Nx = FlowGeometries.Grids.size_tuple(grid)[1]
    cache = fp.cache
    zerofill = strategy isa ZeroFill
    if cache !== nothing
        for i in 1:Nx
            _filters_target(strategy, FlowGeometries.Grids.isactive(grid, i, j)) || continue
            t = i + (j - 1) * Nx
            lo = cache.ptr[t]
            hi = cache.ptr[t+1] - 1
            weighted_sum = zero(T)
            weight_norm = zero(T)
            @inbounds for k in lo:hi
                ii = cache.ii[k]
                jj = cache.jj[k]
                active = FlowGeometries.Grids.isactive(grid, ii, jj)
                w = cache.w[k]
                if zerofill
                    weight_norm += w
                    active && (weighted_sum += w * field[ii, jj])
                else
                    active || continue
                    weight_norm += w
                    weighted_sum += w * field[ii, jj]
                end
            end
            zerofill && (weight_norm = _add_exterior(weight_norm, fp.exterior, i, j))
            out[i, j] = _emit(field, weighted_sum, weight_norm)
        end
    else
        rule = fp.weights
        for i in 1:Nx
            _filters_target(strategy, FlowGeometries.Grids.isactive(grid, i, j)) || continue
            t = i + (j - 1) * Nx
            I = (Int(i), Int(j))
            weighted_sum, weight_norm = _rule_fold(
                (zero(T), zero(T)), rule, grid, t, I, fp.rad, fp.topology, sc,
            ) do acc, J, w
                ws, wn = acc
                active = FlowGeometries.Grids.isactive(grid, J[1], J[2])
                if zerofill
                    return (active ? ws + w * field[J[1], J[2]] : ws, wn + w)
                else
                    active || return acc
                    return (ws + w * field[J[1], J[2]], wn + w)
                end
            end
            zerofill && (weight_norm = _add_exterior(weight_norm, fp.exterior, i, j))
            out[i, j] = _emit(field, weighted_sum, weight_norm)
        end
    end
    return out
end
