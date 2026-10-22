# ---------------------------------------------------------------------------
# Real-space convolution footprints: the banded engine and the cell-by-cell plan
# ---------------------------------------------------------------------------

"""
    BandedScratch{T,MT,VM} <: AbstractFilterScratch

Per-apply buffers for the banded engine: the masked input `mask · field`, one per field in flight.

Empty on an unmasked grid, where the engine convolves the caller's array directly. Slots are added on
demand for a batched apply and reused by later applies.
"""
struct BandedScratch{
    T<:AbstractFloat, MT<:AbstractMatrix{T}, VM<:AbstractVector{MT},
} <: AbstractFilterScratch
    masked_inputs::VM
end

# Grows to `K` slots, each `similar` to slot 1, which must exist: callers hold a masked footprint.
function _banded_inputs!(sc::BandedScratch, K::Integer)
    while length(sc.masked_inputs) < K
        push!(sc.masked_inputs, similar(@inbounds sc.masked_inputs[1]))
    end
    return sc.masked_inputs
end

"""
    BandTable{VI,VT}

The banded engine's taps by band. Band `b` holds runs `ptr[b]:ptr[b+1]-1`, and run `s` covers the
consecutive axis-1 offsets `di0[s] + t − 1` at axis-2 offset `dj[s]`, with weights `w[start[s] + t − 1]`
for `t = 1:(start[s+1] − start[s])`.
"""
struct BandTable{VI<:AbstractVector{Int}, VT<:AbstractVector}
    dj::VI
    di0::VI
    start::VI
    w::VT
    ptr::VI
end

"""
    _band_table(di, dj, w, ptr) -> BandTable

The flat per-tap lists a builder produces (band `b`'s taps `ptr[b]:ptr[b+1]-1`) grouped into runs: each
band's taps ordered by axis-2 then axis-1 offset, a run broken wherever the axis-2 offset changes or the
axis-1 offset skips.
"""
function _band_table(
    di::AbstractVector{Int}, dj::AbstractVector{Int}, w::AbstractVector{T}, ptr::AbstractVector{Int},
) where {T}
    nb = length(ptr) - 1
    rdj, rdi, rstart = Int[], Int[], Int[]
    rw = sizehint!(T[], length(w))
    rptr = Vector{Int}(undef, nb + 1)
    rptr[1] = 1
    for b in 1:nb
        prev = 0
        for k in sort(ptr[b]:(ptr[b + 1] - 1); by = k -> (dj[k], di[k]))
            if prev == 0 || dj[k] != dj[prev] || di[k] != di[prev] + 1
                push!(rdj, dj[k])
                push!(rdi, di[k])
                push!(rstart, length(rw) + 1)
            end
            push!(rw, w[k])
            prev = k
        end
        rptr[b + 1] = length(rdj) + 1
    end
    push!(rstart, length(rw) + 1)
    return BandTable(rdj, rdi, rstart, rw, rptr)
end

"""
    FilterFootprint{T,BT,MT,SC,MS}

Precomputed convolution footprint for a structured grid + kernel + scale whose axis 1 is uniform. The
in-support neighbour offsets and their geometric weights `w = kernel_weight(distance) * cell_area` are
held in a [`BandTable`](@ref), grouped into axis-2 (`y`) bands. A Cartesian grid uniform in both
directions is translation-invariant, a single band; otherwise the footprint is invariant along axis 1
only, one band per row.

The offsets and weights depend on the geometry alone. `invden`, the reciprocal window mass, also depends
on the mask, the mask strategy and where a domain edge cuts the window; [`_banded_build_invden`](@ref)
accumulates it at plan build.
"""
struct FilterFootprint{
    T<:AbstractFloat, BT<:BandTable, MT<:AbstractMatrix{T}, SC<:BandedScratch{T}, MS<:AbstractMaskStrategy,
}
    taps::BT           # offsets and weights by band
    nbands::Int        # 1 (uniform Cartesian) or Ny
    strategy::MS       # the strategy `invden` was accumulated under
    masked::Bool       # grid has inactive cells, so the numerator needs `mask · field`
    bound::Bool        # the strategies' denominators differ: a cell is inactive or a window leaves the grid
    # The grid's wrap at build, which `invden` was accumulated under. The apply uses these and ignores
    # the periodic flags it is passed.
    periodic_x::Bool
    periodic_y::Bool
    invden::MT         # 1/(window mass) per point; zero where the window is empty or, under Deformable, the target inactive
    scratch::SC        # per-apply `mask · field` buffers
end

"""
    _banded_row_accumulate!(oc, src, taps::BandTable, b, periodic_x, periodic_y, Nx, Ny, j) -> oc

Add output row `j`'s weighted tap sum over band `b` into `oc`, one contiguous axpy along `x` per tap.
`src` is `mask · field` for the numerator at apply time and the mass field for `invden` at plan build.
On a bounded axis 1 a tap covers the columns whose source index stays in range.
"""
@inline function _banded_row_accumulate!(
    oc::AbstractVector{T}, src::AbstractMatrix, taps::BandTable, b::Integer,
    periodic_x::Bool, periodic_y::Bool, Nx::Int, Ny::Int, j::Integer,
) where {T<:AbstractFloat}
    w = taps.w
    @inbounds for s in taps.ptr[b]:(taps.ptr[b + 1] - 1)
        jj = j + taps.dj[s]
        if jj < 1 || jj > Ny
            periodic_y || continue
            jj = mod1(jj, Ny)
        end
        fc = view(src, :, jj)
        k0 = taps.start[s]
        d0 = taps.di0[s] - k0
        for k in k0:(taps.start[s + 1] - 1)
            wk = T(w[k])
            d = d0 + k
            if periodic_x && abs(d) < Nx
                # With `|d| < Nx` the wrapped tap is two contiguous runs, each at a constant offset.
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
    end
    return oc
end

@inline _banded_band(nbands::Int, j::Integer) = nbands == 1 ? 1 : Int(j)

"""
    _banded_unit_mass!(den, taps::BandTable, nbands, periodic_x, periodic_y) -> den

[`_banded_row_accumulate!`](@ref) over a mass field of ones, into a zeroed `den`, from the runs' weights
alone. On a periodic axis 1 every target of a row takes every tap, so the row is one sum, added in the
order the accumulation adds it. On a bounded axis 1 a target takes the part of each run whose sources stay
on the row, a difference of the run's prefix sums. Cost `O(taps)` per row on a periodic axis 1 and
`O(runs·Nx)` on a bounded one.
"""
function _banded_unit_mass!(
    den::AbstractMatrix{T}, taps::BandTable, nbands::Int, periodic_x::Bool, periodic_y::Bool,
) where {T<:AbstractFloat}
    Nx, Ny = size(den)
    w = taps.w
    cum = T[]
    @inbounds for j in 1:Ny
        b = _banded_band(nbands, j)
        acc = zero(T)
        for s in taps.ptr[b]:(taps.ptr[b + 1] - 1)
            jj = j + taps.dj[s]
            (1 <= jj <= Ny || periodic_y) || continue
            k0, k1 = taps.start[s], taps.start[s + 1] - 1
            if periodic_x
                for k in k0:k1
                    acc += T(w[k])
                end
            else
                L = k1 - k0 + 1
                resize!(cum, L + 1)
                cum[1] = zero(T)
                for t in 1:L
                    cum[t + 1] = cum[t] + T(w[k0 + t - 1])
                end
                # Tap `t` has offset `di0 + t − 1`, so target `i` takes `t` with `1 ≤ i + di0 + t − 1 ≤ Nx`.
                di0 = taps.di0[s]
                for i in 1:Nx
                    lo = max(1, 2 - i - di0)
                    hi = min(L, Nx - i - di0 + 1)
                    hi >= lo && (den[i, j] += cum[hi + 1] - cum[lo])
                end
            end
        end
        periodic_x && fill!(view(den, :, j), acc)
    end
    return den
end

"""
    _banded_build_invden(grid, taps, nbands, strategy, periodic_x, periodic_y, exterior) -> invden

The reciprocal window mass, accumulated once per plan by running [`_banded_row_accumulate!`](@ref)
over a mass field:

- `ZeroFill` counts every in-support tap whether or not its source cell is active, so its mass field
  is `1` everywhere, and adds `exterior`, the mass past a bounded edge: the kernel's full mass. An
  `exterior` vector is that full mass per row, and is the denominator itself.
- `Deformable` drops inactive cells from both sums, so its mass field is the mask, and an inactive
  target gets zero.
"""
function _banded_build_invden(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2}, taps::BandTable, nbands::Int,
    strategy::AbstractMaskStrategy, periodic_x::Bool, periodic_y::Bool,
    exterior::Union{Nothing,AbstractArray{T}},
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    zerofill = strategy isa ZeroFill
    den = zeros(T, Nx, Ny)
    if zerofill && exterior isa AbstractVector
        @inbounds for j in 1:Ny, i in 1:Nx
            den[i, j] = exterior[j]
        end
    elseif zerofill || all(FlowGeometries.Grids.mask(grid))
        _banded_unit_mass!(den, taps, nbands, periodic_x, periodic_y)
        zerofill && exterior !== nothing && (den .+= exterior)
    else
        mass = Matrix{T}(undef, Nx, Ny)
        @inbounds for j in 1:Ny, i in 1:Nx
            mass[i, j] = FlowGeometries.Grids.isactive(grid, i, j) ? one(T) : zero(T)
        end
        for j in 1:Ny
            _banded_row_accumulate!(
                view(den, :, j), mass, taps, _banded_band(nbands, j), periodic_x, periodic_y, Nx, Ny, j,
            )
        end
    end
    @inbounds for j in 1:Ny, i in 1:Nx
        keep = zerofill || FlowGeometries.Grids.isactive(grid, i, j)
        den[i, j] = keep ? _inv_mass(den[i, j]) : zero(T)
    end
    return den
end

# The `FilterFootprint` of a builder's per-tap lists `di, dj, w, ptr`. `exterior` is
# [`_banded_build_invden`](@ref)'s.
function _banded_footprint(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    di::AbstractVector{Int}, dj::AbstractVector{Int}, w::AbstractVector{T},
    ptr::AbstractVector{Int}, nbands::Int, strategy::AbstractMaskStrategy,
    kernel::Kernels.AbstractFilterKernel, scale::T;
    exterior::Union{Nothing,AbstractArray{T}} =
        strategy isa ZeroFill ? _exterior_mass(grid, kernel, scale) : nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    periodic_x = FlowGeometries.Grids.isperiodic(grid, 1)
    periodic_y = FlowGeometries.Grids.isperiodic(grid, 2)
    masked = !all(FlowGeometries.Grids.mask(grid))
    taps = _band_table(di, dj, w, ptr)
    invden = _banded_build_invden(grid, taps, nbands, strategy, periodic_x, periodic_y, exterior)
    scratch = BandedScratch(masked ? [zeros(T, Nx, Ny)] : Matrix{T}[])
    return FilterFootprint(
        taps, nbands, strategy, masked, masked || _reaches_exterior(grid, kernel, scale),
        periodic_x, periodic_y, invden, scratch,
    )
end


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
    is_cartesian::Bool
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

# Entries of a 2-D cache, counted in target order, stopping once past `limit`.
function _count_scattered(grid, rad, is_cartesian::Bool, mt, sc, limit::Int)
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    n = 0
    for j in 1:Ny, i in 1:Nx
        n = _ball_fold((m, _, _) -> m + 1, n, grid, i, j, rad, is_cartesian, mt, sc)
        n > limit && break
    end
    return n
end

# The 2-D cache, `n` entries from `_count_scattered`, filled in target order `t = i + (j-1)·Nx`.
function _fill_scattered(grid, rule::RadialWeights{D,K,T}, rad, is_cartesian::Bool, mt, sc, n::Int) where {D,K,T}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    ii = Vector{Int}(undef, n)
    jj = Vector{Int}(undef, n)
    w = Vector{T}(undef, n)
    ptr = Vector{Int}(undef, Nx * Ny + 1)
    ptr[1] = 1
    k = 0
    for j in 1:Ny, i in 1:Nx
        t = i + (j - 1) * Nx
        k = _ball_fold(k, grid, i, j, rad, is_cartesian, mt, sc) do m, J, d
            m += 1
            ii[m] = J[1]
            jj[m] = J[2]
            w[m] = last(_cell_weight(rule, grid, t, (i, j), J, d))
            m
        end
        ptr[t + 1] = k + 1
    end
    return ScatteredCache(ii, jj, w, ptr)
end

# A 2-D cell-by-cell plan's neighbour cache (or `nothing`), its mass past a bounded edge (or `nothing`)
# and its weight rule.
function _scattered_parts(
    grid, kernel::Kernels.AbstractFilterKernel, scale::T, rad, is_cartesian::Bool, mt, strategy, budget,
) where {T}
    rule = RadialWeights(kernel, scale, _kernel_dim(grid))
    sc = FlowGeometries.Connectivity.ball_scratch()
    n = _cache_entries(strategy, budget, _cache_entry_bytes(Val(2), T)) do limit
        _count_scattered(grid, rad, is_cartesian, mt, sc, limit)
    end
    cache = n === nothing ? nothing : _fill_scattered(grid, rule, rad, is_cartesian, mt, sc, n)
    return cache, _plan_exterior(grid, kernel, scale), rule
end

function _scattered_parts(
    grid, kernel::Kernels.HighOrderKernel, scale::T, rad, is_cartesian::Bool, mt, strategy, budget,
) where {T}
    sc = FlowGeometries.Connectivity.ball_scratch()
    fold = (f, acc, I) -> _ball_fold(f, acc, grid, I[1], I[2], rad, is_cartesian, mt, sc)
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
    _scattered_window_bounds(grid::StructuredGrid{...,2}, rad) -> (di_lim, dj_lim, periodic_x, periodic_y, x_period, y_period, is_cartesian)

The widest index window the grid's ball query scans, from `Connectivity.metric_window`, with the
periodic flags, the Cartesian wrap lengths and whether the geometry is Cartesian. On a rectilinear grid
the window depends on the row alone, so one evaluation per row covers the grid.
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
    return di_lim, dj_lim, periodic_x, periodic_y, x_period, y_period, is_cartesian
end

# Folds `acc = f(acc, J, d)` over every cell within `rad` of `(i, j)`, the centre included. On a
# Cartesian grid a cell within `rad` through several periodic images contributes once per image.
# `scratch` is an indexed query's candidate buffer, one per task; the structured form ignores it.
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
    di_lim, dj_lim, periodic_x, periodic_y, x_period, y_period, is_cartesian =
        _scattered_window_bounds(grid, rad)
    mt = _query_topology(grid, rad)
    cache, exterior, weights =
        _scattered_parts(grid, kernel, scale, rad, is_cartesian, mt, cache_strategy, cache_byte_budget)
    return _scattered_plan(kernel, scale, rad, di_lim, dj_lim, periodic_x, periodic_y, x_period, y_period,
                           is_cartesian, cache, mt, exterior, weights)
end

# The plan with its cache and its exterior mass each held as `nothing` or an array under one field type,
# so the plan's type is fixed by the grid and the kernel.
@inline _scattered_plan(
    kernel::K, scale::T, rad::T, di_lim, dj_lim, periodic_x, periodic_y, x_period, y_period, is_cartesian,
    cache, mt::MT, exterior, weights::W,
) where {K,T,MT,W} = ScatteredFilterPlan{
    T, K, Union{Nothing,ScatteredCache{T,Vector{Int},Vector{T}}}, MT, Union{Nothing,Matrix{T}}, W,
}(kernel, scale, rad, di_lim, dj_lim, periodic_x, periodic_y, x_period, y_period, is_cartesian,
  cache, mt, exterior, weights)

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
    cache, exterior, weights =
        _scattered_parts(grid, kernel, scale, rad, false, mt, cache_strategy, cache_byte_budget)
    return _scattered_plan(kernel, scale, rad, di_lim, dj_lim, false, false, zero(T), zero(T), false,
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
    FilterFootprintND{N, T}

Translation-invariant footprint of a 1-D or 3-D Cartesian grid with uniform axes: one set of in-support
neighbour offsets (`NTuple{N,Int}`) and their weights `w = kernel_weight(distance) · cell_measure`.
`exterior` is the mass past a bounded edge that `ZeroFill` adds to each point's denominator
([`_exterior_mass`](@ref)), or `nothing`.
"""
struct FilterFootprintND{
    N, T<:AbstractFloat, VO<:AbstractVector{NTuple{N,Int}}, VT<:AbstractVector{T},
    EX<:Union{Nothing,AbstractArray{T,N}},
}
    offsets::VO
    w::VT
    exterior::EX
end

# `wn` plus the exterior mass at a point, for `ZeroFill`'s full-mass denominator.
@inline _add_exterior(wn, ::Nothing, I...) = wn
@inline _add_exterior(wn, E::AbstractArray, I...) = wn + @inbounds(E[I...])

# The dimension a kernel's real-space weight is taken in: the grid's coordinate count.
@inline _kernel_dim(grid::FlowGeometries.Grids.AbstractGrid) = Val(FlowGeometries.Grids.ncoordinates(grid))

# The weighted mean `ws / wn`, zero where no weight was summed. A signed kernel's window mass keeps its
# sign.
@inline _normalized(ws, wn) = iszero(wn) ? zero(ws) : ws / wn
@inline _inv_mass(m) = iszero(m) ? zero(m) : inv(m)

# Whether a target is filtered: every one under `ZeroFill`, the active ones under `Deformable`.
@inline _filters_target(::ZeroFill, ::Bool) = true
@inline _filters_target(::AbstractMaskStrategy, active::Bool) = active

"""
    _banded_cartesian_rows!(di, dj, w, ptr, grid, kernel, scale, rad) -> nothing

One offset/weight band per row of a Cartesian grid uniform along `x` and stretched along `y`, appended to
`di, dj, w, ptr`. A periodic `y` tiles: a band reaches every row image inside the support, each at its
own displacement, and the apply maps a raw row offset back onto the grid with `mod1`.
"""
function _banded_cartesian_rows!(
    di::AbstractVector{Int}, dj::AbstractVector{Int}, w::AbstractVector{T}, ptr::AbstractVector{Int},
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2}, kernel::Kernels.AbstractFilterKernel, scale::T, rad::T,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    _, Ny = FlowGeometries.Grids.size_tuple(grid)
    dx = step(FlowGeometries.Grids.coordinates(grid, 1))
    y = FlowGeometries.Grids.coordinates(grid, 2)
    periodic_y = FlowGeometries.Grids.isperiodic(grid, 2) && Ny > 1
    Py = periodic_y ? T(FlowGeometries.Grids.period(grid, 2)) : zero(T)
    sy = (Ny > 1 && y[Ny] < y[1]) ? -one(T) : one(T)   # coordinate direction of increasing index
    min_dy = T(FlowGeometries.Grids.minimum_spacing(grid, 2))
    di_lim = dx > 0 ? ceil(Int, rad / dx) : 0
    dj_lim = min_dy > 0 ? ceil(Int, rad / min_dy) : 0
    for j in 1:Ny
        for ddj in -dj_lim:dj_lim
            jraw = j + ddj
            (periodic_y || 1 <= jraw <= Ny) || continue
            m = fld(jraw - 1, Ny)
            jj = jraw - m * Ny
            Δy = T(y[jj]) + sy * T(m) * Py - T(y[j])
            abs(Δy) <= rad || continue
            A = FlowGeometries.Grids.area(grid, 1, jj)
            for ddi in -di_lim:di_lim
                d = sqrt((ddi * dx)^2 + Δy^2)
                d <= rad || continue
                push!(di, ddi)
                push!(dj, ddj)
                push!(w, Kernels.kernel_weight(kernel, T(d), scale, Val(2)) * A)
            end
        end
        push!(ptr, length(di) + 1)
    end
    return nothing
end

"""
    build_footprint(grid, kernel, scale; mask_strategy = ZeroFill()) -> FilterFootprint

The banded engine, for a 2-D grid whose axis 1 is a `Range`: constant spacing along axis 1 makes the
footprint invariant under a shift along it, so each row of targets shares one offset/weight list. A
Cartesian grid uniform in both directions has one list for every row; otherwise each row has its own
(`nbands = Ny`), on a stretched Cartesian `y` and on any latitude axis of a sphere.
"""
function build_footprint(
    # `StructuredGrid{T,G,N,S,TP,C,…}`: `C`, the axes' types, is the sixth parameter.
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2,S,TP,<:Tuple{AbstractRange{T},AbstractVector{T}}},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}, S, TP<:NTuple{2,FlowGeometries.Grids.AbstractTopology}}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    rad = Kernels.kernel_radius(kernel, scale, Val(2))

    di = Int[]
    dj = Int[]
    w = T[]
    ptr = Int[1]

    if G <: FlowGeometries.Geometry.CartesianGeometry{T} && FlowGeometries.Grids.coordinates(grid, 2) isa AbstractRange
        dx = step(FlowGeometries.Grids.coordinates(grid, 1))
        dy = step(FlowGeometries.Grids.coordinates(grid, 2))
        A = FlowGeometries.Grids.area(grid, 1, 1)   # uniform Cartesian cell area
        di_lim = dx > 0 ? ceil(Int, rad / dx) : 0
        dj_lim = dy > 0 ? ceil(Int, rad / dy) : 0
        sizehint!(di, (2*di_lim + 1) * (2*dj_lim + 1))
        sizehint!(dj, (2*di_lim + 1) * (2*dj_lim + 1))
        sizehint!(w, (2*di_lim + 1) * (2*dj_lim + 1))
        for ddj in -dj_lim:dj_lim, ddi in -di_lim:di_lim
            d = sqrt((ddi * dx)^2 + (ddj * dy)^2)
            if d <= rad
                push!(di, ddi)
                push!(dj, ddj)
                push!(w, Kernels.kernel_weight(kernel, T(d), scale, Val(2)) * A)
            end
        end
        push!(ptr, length(di) + 1)
        return _banded_footprint(grid, di, dj, w, ptr, 1, mask_strategy, kernel, scale)
    elseif G <: FlowGeometries.Geometry.CartesianGeometry{T}
        _banded_cartesian_rows!(di, dj, w, ptr, grid, kernel, scale, rad)
        return _banded_footprint(grid, di, dj, w, ptr, Ny, mask_strategy, kernel, scale)
    else
        # Band `j`'s window, from `metric_window`, holds for every row its ball reaches: a row nearer a
        # pole spans more longitude at the same radius. `_banded_lon_offsets` turns it into offsets.
        dλ = step(FlowGeometries.Grids.coordinates(grid, 1))
        periodic_λ = FlowGeometries.Grids.isperiodic(grid, 1)
        offsets = Vector{NTuple{3,UnitRange{Int}}}(undef, Ny)
        dj_lims = Vector{Int}(undef, Ny)
        total_entries = 0
        for j in 1:Ny
            dl, djl = FlowGeometries.Connectivity.metric_window(grid, (1, j), rad)
            offsets[j] = _banded_lon_offsets(dl, Nx, dλ, periodic_λ)
            dj_lims[j] = djl
            for ddj in -djl:djl
                jj = j + ddj
                (1 <= jj <= Ny) || continue
                total_entries += sum(length, offsets[j])
            end
        end
        sizehint!(di, total_entries)
        sizehint!(dj, total_entries)
        sizehint!(w, total_entries)
        for j in 1:Ny
            φ = FlowGeometries.Grids.coordinates(grid, 2)[j]
            for ddj in -dj_lims[j]:dj_lims[j]
                jj = j + ddj
                (1 <= jj <= Ny) || continue
                φ2 = FlowGeometries.Grids.coordinates(grid, 2)[jj]
                A = FlowGeometries.Grids.area(grid, 1, jj)   # spherical cell area depends only on latitude
                for range in offsets[j], ddi in range
                    # Great-circle distance with Δλ = ddi·dλ (longitude-translation-invariant).
                    d = FlowGeometries.Geometry.distance(
                        FlowGeometries.Grids.grid_geometry(grid),
                        SA.SVector{2,T}(zero(T), φ),
                        SA.SVector{2,T}(T(ddi) * dλ, φ2),
                    )
                    if d <= rad
                        push!(di, ddi)
                        push!(dj, ddj)
                        push!(w, Kernels.kernel_weight(kernel, d, scale, Val(2)) * A)
                    end
                end
            end
            push!(ptr, length(di) + 1)
        end
        return _banded_footprint(grid, di, dj, w, ptr, Ny, mask_strategy, kernel, scale)
    end
end

"""
    _banded_lon_offsets(dl, Nx, dλ, periodic) -> NTuple{3,UnitRange{Int}}

The longitude offsets a latitude row's taps are drawn from, for a window of `dl` steps of `dλ`. A ring
contributes each column once: `-dl:dl` capped at one turn, the antipodal column of an even ring taken from
one side. A regional box takes the near offsets `|Δi| ≤ dl` and the far ones that come back within the
window across its seam, `|Δi|·|dλ| ≥ 2π − dl·|dλ|`, up to `Nx − 1`; longitude identifies, so the
distance between two columns is taken the short way round.
"""
@inline function _banded_lon_offsets(dl::Int, Nx::Int, dλ::T, periodic::Bool) where {T<:AbstractFloat}
    if periodic
        lim = min(dl, Nx ÷ 2)
        return (1:0, -lim:(2lim + 1 > Nx ? lim - 1 : lim), 1:0)
    end
    near = min(dl, Nx - 1)
    s = abs(dλ)
    far = dl * s >= T(2π) ? near + 1 : max(near + 1, floor(Int, (T(2π) - dl * s) / s))
    return (-(Nx - 1):-far, -near:near, far:(Nx - 1))
end

"""
    build_footprint(grid::StructuredGrid{T,<:AbstractSphericalGeometry,2}, kernel::HighOrderKernel, scale; kwargs...)

A `HighOrderKernel` on a latitude–longitude grid of uniform longitude, any latitude axis: the banded
engine with one correction per row ([`_highorder_rows`](@ref)), or the per-cell plan where the targets of
a row do not share one.
"""
function build_footprint(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2,S,TP,<:Tuple{AbstractRange{T},AbstractVector{T}}},
    kernel::Kernels.HighOrderKernel, scale::T; kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}, S, TP<:NTuple{2,FlowGeometries.Grids.AbstractTopology}}
    return _highorder_spherical_footprint(grid, kernel, scale; kwargs...)
end

function _highorder_spherical_footprint(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2}, kernel::Kernels.HighOrderKernel, scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(), kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}}
    rows = _highorder_rows(grid, kernel, scale)
    rows === nothing && return _build_footprint_scattered(grid, kernel, scale; kwargs...)
    di, dj, w, ptr, mass = rows
    _, Ny = FlowGeometries.Grids.size_tuple(grid)
    return _banded_footprint(grid, di, dj, w, ptr, Ny, mask_strategy, kernel, scale; exterior = mass)
end

"""
    build_footprint(grid, kernel, scale; kwargs...) -> ScatteredFilterPlan

The cell-by-cell plan ([`_build_footprint_scattered`](@ref)) of a 2-D `StructuredGrid` no more specific
method covers, such as one whose axis 1 is a plain `AbstractVector`, which carries no guarantee of
uniform spacing.
"""
function build_footprint(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    return _build_footprint_scattered(grid, kernel, scale; kwargs...)
end
