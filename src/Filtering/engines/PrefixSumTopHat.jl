# ---------------------------------------------------------------------------
# Exact top-hat filtering on any rectilinear 2D `StructuredGrid`, in O(N·dj_lim). Requires
# `TopHatKernel`, whose weight is constant inside its support, so the weighted window sum is a plain
# interval sum. Two properties of a rectilinear grid make that sum O(1):
#
#  1. The cell measure is separable, `area[i,j] == wx[i]*wy[j]`:
#       Cartesian  Δx_i · Δy_j                    → wx[i]=Δx_i,  wy[j]=Δy_j
#       Spherical  R²·cos(φ_j)·Δλ_i·Δφ_j          → wx[i]=Δλ_i,  wy[j]=R²cos(φ_j)Δφ_j
#     so the window sum factors as `Σ_jj wy[jj] · Σ_ii field[ii,jj]·wx[ii]`, the inner term being an
#     interval sum along one row — O(1) from a per-row prefix sum.
#
#  2. At a fixed row offset the in-support axis-1 indices form one contiguous interval whose endpoints
#     are monotone in the target index, so a two-pointer sweep finds them in O(1) amortized.
#     Cartesian: `|Δx| ≤ √(rad²−Δy²)`. Spherical: `cos d = sinφ₁sinφ₂ + cosφ₁cosφ₂·cosΔλ` decreases
#     monotonically in `|Δλ|` on [0,π], so `d ≤ rad` ⟺ `|Δλ| ≤ acos((cos rad − sinφ₁sinφ₂)/(cosφ₁cosφ₂))`.
#
# A periodic Cartesian axis tiles, and a cell contributes once per image inside the support. Along axis
# 1, with the interval half-width `h` and period `P`, write `2h = qP + r` (`0 ≤ r < P`): the interval is
# `q` whole periods and a closed remainder of length `r`, which a shift by whole periods places at
# `[x − ρ, x − ρ + r]`, `ρ = mod(h, P)`. So each row contributes `q` row totals plus one interval sum, and
# both ends of that interval stay monotone in `x`. Along axis 2 every row image is its own band, at its
# own displacement. A spherical longitude identifies instead: each cell counts once.
# ---------------------------------------------------------------------------

"""
    CompensatedScan{T,A}

A prefix scan held as two tables: `hi`, the floating-point running sum, and `lo`, the running sum of
the exact rounding error of each addition, so entry `k` is `hi[k] + lo[k]` to about twice `T`'s
precision (Ogita, Rump & Oishi, *SIAM J. Sci. Comput.* 26 (2005), Algorithm 4.4, `Sum2`). A difference
of two entries, `_scan_diff`, is then as accurate as summing the interval directly, whatever the row
length or the mean of the data.
"""
struct CompensatedScan{T<:AbstractFloat, A<:AbstractArray{T}}
    hi::A
    lo::A
end

CompensatedScan(::Type{T}, dims::Integer...) where {T<:AbstractFloat} =
    CompensatedScan(zeros(T, dims...), zeros(T, dims...))

# Knuth's TwoSum: `s + e == a + b` exactly, with `s = fl(a + b)`.
@inline function _two_sum(a::T, b::T) where {T<:AbstractFloat}
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)
end

# One step of a compensated scan: the running sum and running error after adding `x`.
@inline function _scan_step(h::T, l::T, x::T) where {T<:AbstractFloat}
    s, e = _two_sum(h, x)
    return s, l + e
end

# `Σ_{a ≤ k < b}` of the scanned values, at trailing indices `J`. Each table is differenced on its own:
# rounding `hi[k] + lo[k]` to `T` first discards what `lo` carries.
@inline _scan_diff(hi::AbstractArray, lo::AbstractArray, b::Integer, a::Integer, J::Integer...) =
    @inbounds (hi[b, J...] - hi[a, J...]) + (lo[b, J...] - lo[a, J...])
@inline _scan_diff(P::CompensatedScan, b::Integer, a::Integer, J::Integer...) =
    _scan_diff(P.hi, P.lo, b, a, J...)

"""
    PrefixSumGridPlan{T,VT,PW,DT,BT,WX,WY,MS} <: AbstractGridPlan

The scale-independent part of the prefix-sum top-hat engine: everything fixed by the grid, its mask
and the mask strategy. One instance serves every scale of a sweep.

Holds the separable measure factors (`wx`,`wy`), the axis-1 coordinates in ascending order, and the two
measure scans the window mass is formed from: `prefix_wx` of `wx` along axis 1 (`ZeroFill`), and
`prefix_den` of `mask·wx` per row (`Deformable` on a masked grid, `nothing` otherwise). Every scan holds one
copy of the axis. Where axis 1 wraps (`tiles_x || rings_x`), the two-pointer sweep walks an extended axis of
three copies shifted by `−P, 0, +P`, so a wrapped support interval is one contiguous run: entry `c·Nx + t`
is `xe[t] + (c − 1)·P`, and its scan entry `c` row totals plus the scan's entry `t` ([`_walk_band`](@ref)).
"""
struct PrefixSumGridPlan{
    T<:AbstractFloat,
    VT<:AbstractVector{T},
    PW<:CompensatedScan{T},
    DT<:Union{Nothing,CompensatedScan{T}},
    BT<:AbstractVector{Int},
    WX<:AbstractVector{T},
    WY<:AbstractVector{T},
    MS<:AbstractMaskStrategy,
} <: AbstractGridPlan
    periodic_x::Bool
    periodic_y::Bool
    masked::Bool        # grid has an inactive cell
    x_period::T
    y_period::T
    tiles_x::Bool       # axis 1 is a periodic Cartesian direction: its cells count once per image
    rings_x::Bool       # axis 1 is a periodic longitude: its cells count once
    tiles_y::Bool       # axis 2 is a periodic Cartesian direction
    strategy::MS        # the strategy `prefix_den` and every scale's `invden` are built for
    wx::WX              # axis-1 cell width;  measure[i,j] == wx[i]*wy[j]
    wy::WY              # axis-2 measure factor
    xe::VT              # axis-1 coordinates (Nx), strictly increasing
    src::BT             # xe[t] belongs to axis-1 index src[t]
    prefix_wx::PW       # scan of wx in xe's order (Nx+1) — ZeroFill denominator
    prefix_den::DT      # per-row scan of mask·wx (Nx+1 × Ny), or nothing (Deformable only)
end

"""
    PrefixSumScratch{T,S,VS,VV} <: AbstractFilterScratch

The prefix-sum engine's per-apply buffers: one [`CompensatedScan`](@ref) of `mask·field·wx` per field
in flight, and under [`Kernels.CellOverlap`](@ref) the cells' values `mask·field·wx` themselves in the scan's
order (`cell_values`, `nothing` under `CellCentre`), which the cells the disk's edge cuts read. Every apply
refills them, so one set serves every scale of a sweep, one apply at a time. Each row is written
independently, so the numerator fill may run concurrently by row.

A single-field apply uses slot 1, a batched apply of `K` fields slots `1:K`; `_prefixsum_numerators!`
adds missing slots.
"""
struct PrefixSumScratch{
    T<:AbstractFloat, S<:CompensatedScan{T}, VS<:AbstractVector{S}, VV<:Union{Nothing,AbstractVector},
} <: AbstractFilterScratch
    prefix_nums::VS     # slot k: per-row scan of mask·fieldₖ·wx (nrep*Nx+1 × Ny)
    cell_values::VV     # slot k: mask·fieldₖ·wx by cell in the scan's order (Nx × Ny), or nothing
end

PrefixSumScratch(prefix_nums::AbstractVector) = PrefixSumScratch(prefix_nums, nothing)

# Slot 1 always exists, so the single-field path never checks.
@inline _prefixsum_num(sc::PrefixSumScratch) = @inbounds sc.prefix_nums[1]

# Slot `k`'s cell values, or `nothing`.
@inline _prefixsum_values(sc::PrefixSumScratch, k::Int) = _slot(sc.cell_values, k)
@inline _slot(::Nothing, ::Int) = nothing
@inline _slot(v::AbstractVector, k::Int) = @inbounds v[k]

"""
    _prefixsum_numerators!(sc, K) -> sc.prefix_nums

Grow `sc` to at least `K` numerator slots (and cell-value slots, where held) and return them. A new slot is
a zeroed `similar` of slot 1, since the fill never writes row 1, the empty prefix. Call it before any parallel
region that uses the slots: growing `push!`es.
"""
function _prefixsum_numerators!(sc::PrefixSumScratch{T}, K::Integer) where {T<:AbstractFloat}
    while length(sc.prefix_nums) < K
        P1 = @inbounds sc.prefix_nums[1]
        push!(sc.prefix_nums, CompensatedScan(fill!(similar(P1.hi), zero(T)), fill!(similar(P1.lo), zero(T))))
    end
    _grow_values!(sc.cell_values, K)
    return sc.prefix_nums
end

_grow_values!(::Nothing, ::Integer) = nothing
function _grow_values!(v::AbstractVector, K::Integer)
    while length(v) < K
        push!(v, similar(@inbounds v[1]))
    end
    return v
end

"""
    PrefixEdgeCells{VI,VT,MI}

The cells the support of a [`Kernels.CellOverlap`](@ref) top-hat cuts, for a plan whose axis 1 is uniform. Row
`j`'s taps `ptr[j]:ptr[j+1]-1` hold an axis-1 offset `di`, a raw axis-2 offset `dj` and `w = f·wy`, `f` the
cell's fraction inside the support, which multiplies the cell's scanned value `mask·field·wx`. `far[b, j]` is
the least axis-1 offset from which every cell across a regional longitude box's seam lies wholly inside, `0`
where none does. Axis-1 offsets count along the scan's ascending order (`rev`: the coordinates descend), and
modulo the axis's length where it closes on itself (`wrap`).
"""
struct PrefixEdgeCells{VI<:AbstractVector{Int}, VT<:AbstractVector, MI<:AbstractMatrix{Int}}
    wrap::Bool
    rev::Bool
    far::MI
    ptr::VI
    di::VI
    dj::VI
    w::VT
end

"""
    PrefixSumTopHatPlan{T,GP,SC,MT,WT,RM}

Exact `O(N·dj_lim)` top-hat footprint for a rectilinear 2D `StructuredGrid`, holding the state that
depends on the filter scale ℓ. `grid_plan` and `scratch` may be shared with the other scales of a
sweep.

- `invden`: the reciprocal window mass per point, a function of the grid, the mask, the strategy and ℓ.
- `hw`/`wcell`: under [`Kernels.CellCentre`](@ref), the support half-width per `(band, row)` and, on a
  uniform ascending axis, the constant cell half-width it implies; under [`Kernels.CellOverlap`](@ref),
  `hw ≥ 0` marks a band holding cells wholly inside the support and `wcell` the half-width, in cells, of
  those about the target.
- `edge`: `nothing` under `CellCentre`, the [`PrefixEdgeCells`](@ref) under `CellOverlap`.

`invden` is built for one mask strategy: where the two strategies normalize differently (`bound`), the
apply throws unless its `strategy` has the type of `grid_plan.strategy`.
"""
struct PrefixSumTopHatPlan{
    T<:AbstractFloat,
    GP<:PrefixSumGridPlan{T},
    SC<:PrefixSumScratch{T},
    MT<:AbstractMatrix{T},
    WT<:AbstractMatrix{Int},
    RM<:Union{Nothing,PrefixEdgeCells},
}
    grid_plan::GP
    scratch::SC
    rad::T
    dj_lim::Int
    uniform_axis::Bool  # every nonempty band is constant-width (`wcell >= 0`): no two-pointer walk
    bound::Bool         # the strategies' denominators differ: a cell is inactive or a window leaves the grid
    invden::MT          # 1/(window mass) per point (Nx × Ny); zero where empty or, under Deformable, inactive
    hw::MT              # support half-width per (band, row); negative ⇒ band empty
    wcell::WT           # uniform-axis constant cell half-width per (band, row), else -1
    edge::RM
end

"""
    _rectilinear_measure_factors(grid) -> (wx, wy)

The grid's stored per-axis measure factors, with `measure[i, j] == wx[i] * wy[j]` exactly, including
on a degenerate direction: a single-latitude grid measures arc length `R·Δλ` and a single-longitude
one `R·Δφ`. Throws an `ArgumentError` on a dense measure.
"""
function _rectilinear_measure_factors(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    mf = FlowGeometries.Grids.measure_factors(grid)
    mf === nothing && throw(ArgumentError(
        "the prefix-sum top-hat path needs a separable cell measure, but this grid's measure is dense",
    ))
    return mf[1], mf[2]
end

"""
    _build_prefixsum_grid_plan(grid; mask_strategy = ZeroFill()) -> PrefixSumGridPlan

Build the scale-independent half of the prefix-sum top-hat engine. Nothing here reads the filter
scale, so one call serves an entire sweep — see [`plan_filter_sweep`](@ref).
"""
function _build_prefixsum_grid_plan(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2};
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    periodic_x = FlowGeometries.Grids.isperiodic(grid, 1)
    periodic_y = FlowGeometries.Grids.isperiodic(grid, 2)

    is_cartesian = G <: FlowGeometries.Geometry.CartesianGeometry{T}
    # A longitude support wraps at 2π even where the grid's cells do not close the ring.
    rings = !is_cartesian && Nx > 1

    # Wrap length from the grid's stored period, which on a stretched axis the samples do not determine.
    x_period = (periodic_x && Nx > 1) ? T(FlowGeometries.Grids.period(grid, 1)) : rings ? T(2π) : zero(T)
    y_period = (periodic_y && Ny > 1) ? T(FlowGeometries.Grids.period(grid, 2)) : zero(T)

    wx, wy = _rectilinear_measure_factors(grid)

    # Axis-1 coordinates, ascending; `src` maps each entry to its axis index, which a descending axis permutes.
    x = FlowGeometries.Grids.coordinates(grid, 1)
    src = issorted(x) ? collect(1:Nx) : sortperm(x)
    xe = T[x[i] for i in src]

    prefix_wx = CompensatedScan(T, Nx + 1)
    let h = zero(T), l = zero(T)
        @inbounds for k in 1:Nx
            h, l = _scan_step(h, l, T(wx[src[k]]))
            prefix_wx.hi[k + 1] = h
            prefix_wx.lo[k + 1] = l
        end
    end

    mask = FlowGeometries.Grids.mask(grid)
    masked = !all(mask)

    prefix_den = if !_mask_free_mass(mask_strategy, mask)
        P = CompensatedScan(T, Nx + 1, Ny)
        @inbounds for j in 1:Ny
            h = l = zero(T)
            for k in 1:Nx
                i = src[k]
                h, l = _scan_step(h, l, mask[i, j] ? T(wx[i]) : zero(T))
                P.hi[k + 1, j] = h
                P.lo[k + 1, j] = l
            end
        end
        P
    else
        nothing
    end

    # A support interval wraps on a periodic axis or a longitude (P exceeds the axis extent).
    tiles_x = periodic_x && Nx > 1 && is_cartesian
    rings_x = rings
    tiles_y = periodic_y && Ny > 1 && is_cartesian && y_period > zero(T)
    return PrefixSumGridPlan(
        periodic_x, periodic_y, masked, x_period, y_period, tiles_x, rings_x, tiles_y, mask_strategy,
        wx, wy, xe, src, prefix_wx, prefix_den,
    )
end

"""
    _prefixsum_scratch(gp::PrefixSumGridPlan, grid, kernel) -> PrefixSumScratch

Allocate the engine's per-apply numerator scan, and under [`Kernels.CellOverlap`](@ref) the cell values. Sized
by the grid alone, so one scratch serves every scale — but only one apply at a time; see
[`AbstractFilterScratch`](@ref).
"""
function _prefixsum_scratch(
    gp::PrefixSumGridPlan{T}, grid::FlowGeometries.Grids.StructuredGrid{T,G,2}, ::Kernels.TopHatKernel{Kernels.CellCentre},
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    _, Ny = FlowGeometries.Grids.size_tuple(grid)
    return PrefixSumScratch([CompensatedScan(T, length(gp.xe) + 1, Ny)])
end

function _prefixsum_scratch(
    gp::PrefixSumGridPlan{T}, grid::FlowGeometries.Grids.StructuredGrid{T,G,2}, ::Kernels.TopHatKernel{Kernels.CellOverlap},
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    return PrefixSumScratch([CompensatedScan(T, length(gp.xe) + 1, Ny)], [zeros(T, Nx, Ny)])
end

# An ascending axis 1 uniform around its wrap where it wraps, on which a band shorter than a period is a
# constant ±w cells.
@inline _prefixsum_uniform_axis(grid, x) = FlowGeometries.Grids.isuniform(grid, 1) && step(x) > zero(eltype(x))

# Whether a constant-width band wraps modulo the axis length: axis 1 tiles or is a closed longitude ring.
@inline _prefixsum_band_wraps(gp::PrefixSumGridPlan) = gp.tiles_x || (gp.rings_x && gp.periodic_x)

# The scanned sum over cells `i − w : i + w`: clamped to the axis, or modulo its length where it wraps, for
# `2w + 1 ≤ Nx`. A run across the wrap is one row total less the scan before its start, plus the scan to its end.
@inline function _band_diff(Ph, Pl, wrap::Bool, i::Int, w::Int, Nx::Int, J::Integer...)
    wrap || return _scan_diff(Ph, Pl, min(Nx, i + w) + 1, max(1, i - w), J...)
    i - w < 1 && return _wrap_sum(Ph, Pl, 1, i + w + 1, Nx + i - w, Nx, J...)
    i + w > Nx && return _wrap_sum(Ph, Pl, 1, i + w - Nx + 1, i - w, Nx, J...)
    return _scan_diff(Ph, Pl, i + w + 1, i - w, J...)
end

# The shift of copy `c` of the extended axis: `(c − 1)·P` where axis 1 wraps, none otherwise.
@inline _copy_shift(wraps::Bool, c::Int, P::T) where {T} = wraps ? T(c - 1) * P : zero(T)

# The extended-axis sum from scan entry `a` of copy `cl` to entry `b` of copy `cl + n`: `n` row totals and the
# difference of the two entries, the totals less the lower entry first.
@inline function _wrap_sum(Ph, Pl, n::Int, b::Int, a::Int, Nx::Int, J::Integer...)
    iszero(n) && return _scan_diff(Ph, Pl, b, a, J...)
    @inbounds return ((n * Ph[Nx + 1, J...] - Ph[a, J...]) + Ph[b, J...]) +
                     ((n * Pl[Nx + 1, J...] - Pl[a, J...]) + Pl[b, J...])
end

"""
    _walk_band(f, xe, src, x, wraps, P, a, c, Nx)

The two-pointer sweep of one band: for each target `i = src[t]` in ascending coordinate order whose interval
`[x[i] − a, x[i] + c]` holds an entry of the extended axis, `f(i, n, hi, lo)` with the run's sum
`_wrap_sum(Ph, Pl, n, hi, lo, Nx, …)`. Each pointer holds its copy and entry, so the extended axis is never
stored; entry `t` of copy `k` lies at `xe[t] + _copy_shift(wraps, k, P)`.
"""
@inline function _walk_band(f::F, xe, src, x, wraps::Bool, P::T, a::T, c::T, Nx::Int) where {F, T}
    nc = wraps ? 3 : 1
    lc, lt = 0, 1   # first entry at or past the interval's start
    hc, ht = 0, 0   # last entry at or before its end (none yet)
    @inbounds for t in 1:Nx
        i = src[t]
        xc = x[i]
        xlo = xc - a
        xhi = xc + c
        while lc < nc && xe[lt] + _copy_shift(wraps, lc, P) < xlo
            lt += 1
            lt > Nx && (lt = 1; lc += 1)
        end
        while true
            nt, ncp = ht + 1, hc
            nt > Nx && (nt = 1; ncp += 1)
            (ncp < nc && xe[nt] + _copy_shift(wraps, ncp, P) <= xhi) || break
            ht, hc = nt, ncp
        end
        hc * Nx + ht >= lc * Nx + lt && f(i, hc - lc, ht + 1, lt)
    end
    return nothing
end

# The source row of band 0 for target row `j`. On a tiling axis 2 every raw offset `-dj_lim:dj_lim` is
# one row image; on a bounded one the bands start at the first row in reach, so at most `Ny` of them.
@inline _prefixsum_row0(tiles_y::Bool, j::Integer, dj_lim::Int) = tiles_y ? j - dj_lim : max(1, j - dj_lim)
@inline _prefixsum_nbands(tiles_y::Bool, dj_lim::Int, Ny::Int) =
    tiles_y ? 2 * dj_lim + 1 : min(2 * dj_lim + 1, Ny)

# For a band of half-width `h`, `(q, a, c)`: the support holds `q` whole row totals plus the interval
# `[x - a, x + c]` (empty where `c < -a`). Scalar arguments only: the device sweep calls it too.
@inline function _prefixsum_window(tiles_x::Bool, rings_x::Bool, P::T, h::T) where {T<:AbstractFloat}
    if tiles_x
        q = floor(Int, 2h / P)
        ρ = mod(h, P)
        return q, ρ, (2h - q * P) - ρ
    elseif rings_x && 2h >= P
        return 1, zero(T), -one(T)
    end
    return 0, h, h
end

"""
    _build_prefixsum_tophat(grid, kernel::TopHatKernel, scale; mask_strategy=ZeroFill(), grid_plan=nothing, scratch=nothing, kwargs...) -> PrefixSumTopHatPlan

Build the exact `O(N·dj_lim)` prefix-sum top-hat plan for one scale. `grid_plan`/`scratch` let a sweep
hand in the shared pieces; omitted, they are built here. A given `grid_plan` fixes the mask strategy,
and `mask_strategy` is then unused. `kwargs...` absorbs and ignores `cache_strategy`.
"""
function _build_prefixsum_tophat(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    kernel::Kernels.TopHatKernel{Kernels.CellCentre},
    scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    grid_plan::Union{Nothing,PrefixSumGridPlan} = nothing,
    scratch::Union{Nothing,PrefixSumScratch} = nothing,
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    gp = grid_plan === nothing ?
        _build_prefixsum_grid_plan(grid; mask_strategy = mask_strategy) : grid_plan
    sc = scratch === nothing ? _prefixsum_scratch(gp, grid, kernel) : scratch

    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    rad = Kernels.kernel_radius(kernel, scale)
    is_cartesian = G <: FlowGeometries.Geometry.CartesianGeometry{T}

    # Axis-2 band bound from the axis's own minimum gap, converted to a physical distance. A tiling
    # axis-2 reaches rows through their images, so its bound is not capped at the axis.
    min_dy = FlowGeometries.Grids.minimum_spacing(grid, 2)
    dy_phys = is_cartesian ? min_dy : FlowGeometries.Geometry.radius(FlowGeometries.Grids.grid_geometry(grid)) * min_dy
    dj_lim = !(isfinite(dy_phys) && dy_phys > 0) ? 0 :
             gp.tiles_y ? ceil(Int, rad / dy_phys) : min(Ny - 1, ceil(Int, rad / dy_phys))

    x = FlowGeometries.Grids.coordinates(grid, 1)
    y = FlowGeometries.Grids.coordinates(grid, 2)
    nb = _prefixsum_nbands(gp.tiles_y, dj_lim, Ny)
    uniform = _prefixsum_uniform_axis(grid, x)
    # A band shorter than half a period is constant-width where it wraps modulo the axis; on a regional
    # longitude box, where it is also shorter than the gap across the box's seam, which keeps it to the grid's
    # own columns.
    wrap = _prefixsum_band_wraps(gp)
    seam = gp.rings_x ? gp.x_period - abs(T(x[Nx]) - T(x[1])) : T(Inf)
    sy = (Ny > 1 && y[Ny] < y[1]) ? -one(T) : one(T)   # coordinate direction of increasing index

    hw = fill(-one(T), nb, Ny)
    wcell = fill(-1, nb, Ny)
    @inbounds for j in 1:Ny
        y_t = y[j]
        row0 = _prefixsum_row0(gp.tiles_y, j, dj_lim)
        for b in 0:(nb - 1)
            jj_raw = row0 + b
            (gp.tiles_y || jj_raw <= Ny) || continue
            m = fld(jj_raw - 1, Ny)
            dy = y[jj_raw - m * Ny] + sy * T(m) * gp.y_period - y_t
            h = FlowGeometries.Connectivity.metric_band(grid, 1, y_t, y_t + dy, rad)
            h < zero(T) && continue
            hw[b + 1, j] = h
            if uniform && (wrap || h < seam) && !((gp.rings_x || gp.tiles_x) && 2h >= gp.x_period)
                # Same `≤` the pointer loop uses, so it makes the same boundary decision.
                w = 0
                while (w + 1) * step(x) <= h
                    w += 1
                end
                wcell[b + 1, j] = w
            end
        end
    end

    zerofill = gp.strategy isa ZeroFill
    exterior = zerofill ? _exterior_mass(grid, kernel, scale) : nothing
    invden = _prefixsum_build_invden(grid, gp, dj_lim, hw, wcell, exterior)
    bound = gp.masked || _reaches_exterior(grid, kernel, scale)
    # Every band constant-width: nothing for a batch to share.
    every_band = uniform && all(k -> hw[k] < zero(T) || wcell[k] >= 0, eachindex(hw))

    return PrefixSumTopHatPlan(gp, sc, rad, dj_lim, every_band, bound, invden, hw, wcell, nothing)
end

# `(near, far)`: a band's cells about the target are scanned at offsets `0:near`, and a regional longitude box's
# cells across its seam at offsets `far:(Nx − 1)` (none where `far > Nx − 1`). A tiling axis has every image
# (the scan stops at the first cell outside the support), a closed ring each cell once, a bounded axis its
# own cells; a regional longitude box keeps the near offsets within half a turn.
function _overlap_offsets(gp::PrefixSumGridPlan, Nx::Int, Δ::F) where {F}
    gp.tiles_x && return typemax(Int) - 1, typemax(Int)
    gp.rings_x && gp.periodic_x && return Nx ÷ 2, typemax(Int)
    gp.rings_x || return Nx - 1, typemax(Int)
    half = floor(Int, F(π) / Δ)
    return min(Nx - 1, half), half + 1
end

# A band's fractions on a uniform axis 1 of step `Δ`: at offsets `0, 1, …, near_max` into `near`, stopping at
# the first cell wholly outside (one cell where axis 1 is a single column), and across a regional box's seam
# at offsets `Nx − 1, Nx − 2, …, far_min` into `far` (entry `k` at offset `Nx − k`), stopping likewise. Both
# offsets' signs give one fraction.
function _band_fractions!(near, far, ov, band, Δ::F, near_max::Int, far_min::Int, Nx::Int, line_x::Bool) where {F}
    empty!(near)
    empty!(far)
    for d in 0:near_max
        f = _cell_fraction(ov, band, (d - F(1) / 2) * Δ, (d + F(1) / 2) * Δ, line_x)
        iszero(f) && break
        push!(near, f)
        line_x && break
    end
    for d in (Nx - 1):-1:far_min
        f = _cell_fraction(ov, band, (d - F(1) / 2) * Δ, (d + F(1) / 2) * Δ, line_x)
        iszero(f) && break
        push!(far, f)
    end
    return near, far
end

"""
    _overlap_band_rows(grid, kernel::TopHatKernel{CellOverlap}, scale) -> (di, dj, w, ptr)

Every tap of a `CellOverlap` top-hat on a 2-D grid of uniform axis 1, by target row: row `j`'s entries
`ptr[j]:ptr[j+1]-1` hold the axis-1 and axis-2 offsets of each cell the support about a target in row `j`
reaches and its weight `measure · f`. The same for every target of a row, as [`_sphere_band_rows`](@ref)'s.
"""
function _overlap_band_rows(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2}, kernel::Kernels.TopHatKernel{Kernels.CellOverlap}, scale::T,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    gp = _build_prefixsum_grid_plan(grid)
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    F = _overlap_float(T)
    rad = Kernels.kernel_radius(kernel, scale)
    ov = _overlap_rule(FlowGeometries.Grids.grid_geometry(grid), rad)
    line_x, line_y = Nx == 1, Ny == 1
    y = FlowGeometries.Grids.coordinates(grid, 2)
    Δ = line_x ? zero(F) : abs(F(step(FlowGeometries.Grids.coordinates(grid, 1))))
    fy = line_y ? F[] : _cell_faces(grid, 2, F)
    near_max, far_min = _overlap_offsets(gp, Nx, Δ)
    closed_ring = gp.rings_x && gp.periodic_x
    sy = (Ny > 1 && y[Ny] < y[1]) ? -one(F) : one(F)
    di, dj, w = Int[], Int[], T[]
    ptr = Int[1]
    fr, ff = F[], F[]
    for j in 1:Ny
        tgt = _overlap_target(ov, F(y[j]))
        # The target's row, then rows outward each way until one lies wholly outside.
        for dir in (0, 1, -1)
            (line_y && dir != 0) && continue
            k = abs(dir)
            while true
                jr = j + dir * k
                (gp.tiles_y || 1 <= jr <= Ny) || break
                m = fld(jr - 1, Ny)
                jj = jr - m * Ny
                band = _overlap_band(ov, fy, jj, sy * F(m) * F(gp.y_period), tgt, line_y)
                band === nothing && break
                A = T(gp.wx[1]) * T(gp.wy[jj])
                _band_fractions!(fr, ff, ov, band, Δ, near_max, far_min, Nx, line_x)
                for d in 0:(length(fr) - 1)
                    _push_edge!(di, dj, w, d, jr - j, A * T(fr[d + 1]), d > 0 && !(closed_ring && 2d == Nx))
                end
                for q in eachindex(ff)
                    _push_edge!(di, dj, w, Nx - q, jr - j, A * T(ff[q]), true)
                end
                dir == 0 && break
                k += 1
            end
        end
        push!(ptr, length(di) + 1)
    end
    return di, dj, w, ptr
end

# Appends the tap of offset `d` and, unless `d` names one cell on both sides, of `-d`.
function _push_edge!(di, dj, w, d::Int, djr::Int, wt, both::Bool)
    push!(di, d); push!(dj, djr); push!(w, wt)
    if both
        push!(di, -d); push!(dj, djr); push!(w, wt)
    end
    return nothing
end

"""
    _build_prefixsum_tophat(grid, kernel::TopHatKernel{CellOverlap}, scale; ...) -> PrefixSumTopHatPlan

The prefix-sum top-hat with each cell weighted by its fraction inside the support, on a grid whose axis 1 is
uniform: per `(band, row)` the integer half-width of the cells wholly inside about the target (by its scan
interval) and, across a regional longitude box's seam, the offset from which they are; per target row, the
cells the support's boundary cuts as [`PrefixEdgeCells`](@ref) taps. A band's fractions are symmetric in the axis-1
offset, and fall from one to zero as it grows. `ZeroFill` divides by the support's measure
([`_support_mass`](@ref)) where a window leaves the grid, and by the summed weights elsewhere.
"""
function _build_prefixsum_tophat(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    kernel::Kernels.TopHatKernel{Kernels.CellOverlap},
    scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    grid_plan::Union{Nothing,PrefixSumGridPlan} = nothing,
    scratch::Union{Nothing,PrefixSumScratch} = nothing,
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    gp = grid_plan === nothing ?
        _build_prefixsum_grid_plan(grid; mask_strategy = mask_strategy) : grid_plan
    sc = scratch === nothing ? _prefixsum_scratch(gp, grid, kernel) : scratch
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    F = _overlap_float(T)
    rad = Kernels.kernel_radius(kernel, scale)
    geo = FlowGeometries.Grids.grid_geometry(grid)
    ov = _overlap_rule(geo, rad)
    line_x, line_y = Nx == 1, Ny == 1
    x = FlowGeometries.Grids.coordinates(grid, 1)
    y = FlowGeometries.Grids.coordinates(grid, 2)
    Δ = line_x ? zero(F) : abs(F(step(x)))
    fy = line_y ? F[] : _cell_faces(grid, 2, F)
    wrap = gp.tiles_x || (gp.rings_x && gp.periodic_x)
    rev = !line_x && step(x) < zero(eltype(x))

    # Rows whose band can reach the support, from the axis's least gap; a tiling axis 2 reaches rows through
    # their images.
    min_dy = FlowGeometries.Grids.minimum_spacing(grid, 2)
    dy_phys = G <: FlowGeometries.Geometry.CartesianGeometry{T} ? min_dy :
        _least_meridian_radius(geo) * min_dy
    dj_lim = !(isfinite(dy_phys) && dy_phys > 0) ? 0 :
             gp.tiles_y ? ceil(Int, rad / dy_phys) + 1 : min(Ny - 1, ceil(Int, rad / dy_phys) + 1)
    nb = _prefixsum_nbands(gp.tiles_y, dj_lim, Ny)
    sy = (Ny > 1 && y[Ny] < y[1]) ? -one(F) : one(F)
    near_max, far_min = _overlap_offsets(gp, Nx, Δ)
    closed_ring = gp.rings_x && gp.periodic_x

    hw = fill(-one(T), nb, Ny)
    wcell = fill(-1, nb, Ny)
    far = zeros(Int, nb, Ny)
    ptr = Vector{Int}(undef, Ny + 1)
    ptr[1] = 1
    rdi, rdj, rw = Int[], Int[], T[]
    fr, ff = F[], F[]
    tgts = map(t -> _overlap_target(ov, F(t)), y)
    for j in 1:Ny
        tgt = tgts[j]
        row0 = _prefixsum_row0(gp.tiles_y, j, dj_lim)
        for b in 0:(nb - 1)
            jr = row0 + b
            (gp.tiles_y || jr <= Ny) || continue
            m = fld(jr - 1, Ny)
            jj = jr - m * Ny
            band = _overlap_band(ov, fy, jj, sy * F(m) * F(gp.y_period), tgt, line_y)
            band === nothing && continue
            wy = T(gp.wy[jj])
            _band_fractions!(fr, ff, ov, band, Δ, near_max, far_min, Nx, line_x)
            wf = 0
            while wf < length(fr) && isone(fr[wf + 1])
                wf += 1
            end
            wf -= 1
            closed_ring && (wf = min(wf, (Nx - 1) ÷ 2))
            for d in (wf + 1):(length(fr) - 1)
                _push_edge!(rdi, rdj, rw, d, jr - j, T(fr[d + 1]) * wy, d > 0 && !(closed_ring && 2d == Nx))
            end
            fa = 0
            whole = true
            for k in eachindex(ff)
                d = Nx - k
                if whole && isone(ff[k])
                    fa = d
                else
                    whole = false
                    _push_edge!(rdi, rdj, rw, d, jr - j, T(ff[k]) * wy, true)
                end
            end
            if wf >= 0 || fa > 0
                hw[b + 1, j] = zero(T)
                wcell[b + 1, j] = wf
                far[b + 1, j] = fa
            end
        end
        ptr[j + 1] = length(rdi) + 1
    end

    edge = PrefixEdgeCells(wrap, rev, far, ptr, rdi, rdj, rw)
    bound = gp.masked || _reaches_exterior(grid, kernel, scale)
    invden = zeros(T, Nx, Ny)
    fp = PrefixSumTopHatPlan(gp, sc, rad, dj_lim, true, bound, invden, hw, wcell, edge)
    # The window mass: the row body over the scan of the mass field, `wx` everywhere (`ZeroFill`, or every
    # cell active) or `mask·wx`.
    Pm = gp.prefix_den === nothing ? _replicated_scan(gp.prefix_wx, Ny) : gp.prefix_den
    mask = FlowGeometries.Grids.mask(grid)
    Vm = [(gp.prefix_den === nothing || mask[gp.src[k], j]) ? T(gp.wx[gp.src[k]]) : zero(T) for k in 1:Nx, j in 1:Ny]
    for j in 1:Ny
        _overlap_row_sum!(_entry_row(invden, j, rev, Nx), Pm, Vm, fp, edge, j, Nx, Ny)
    end
    zerofill = gp.strategy isa ZeroFill
    if zerofill && _reaches_exterior(grid, kernel, scale)
        for j in 1:Ny
            fill!(view(invden, :, j), T(_support_mass(ov, line_x, line_y, tgts[j])))
        end
    end
    @inbounds for j in 1:Ny, i in 1:Nx
        keep = zerofill || FlowGeometries.Grids.isactive(grid, i, j)
        invden[i, j] = keep ? _inv_mass(invden[i, j]) : zero(T)
    end
    return fp
end

# The scan `P` of one row, held for each of `Ny` rows.
function _replicated_scan(P::CompensatedScan{T}, Ny::Int) where {T}
    R = CompensatedScan(T, length(P.hi), Ny)
    for j in 1:Ny
        copyto!(view(R.hi, :, j), P.hi)
        copyto!(view(R.lo, :, j), P.lo)
    end
    return R
end

"""
    _prefixsum_build_invden(grid, gp, dj_lim, hw, wcell, exterior) -> invden

The reciprocal window mass per point, zero where the mass is zero. `ZeroFill` adds `exterior`, the mass
past a bounded edge, and keeps every target; `Deformable` counts active cells only and gives an
inactive target zero, so the apply's epilogue is one multiply.
"""
function _prefixsum_build_invden(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    gp::PrefixSumGridPlan{T},
    dj_lim::Int,
    hw::AbstractMatrix{T},
    wcell::AbstractMatrix{Int},
    exterior::Union{Nothing,AbstractMatrix{T}},
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    x = FlowGeometries.Grids.coordinates(grid, 1)
    Pd, Pwx = gp.prefix_den, gp.prefix_wx
    use_mask_den = !(gp.strategy isa ZeroFill) && Pd !== nothing
    nb = size(hw, 1)
    wrap = _prefixsum_band_wraps(gp)
    wraps = gp.tiles_x || gp.rings_x

    den = zeros(T, Nx, Ny)
    @inbounds for j in 1:Ny
        row0 = _prefixsum_row0(gp.tiles_y, j, dj_lim)
        for b in 0:(nb - 1)
            h = hw[b + 1, j]
            h < zero(T) && continue
            jj = mod1(row0 + b, Ny)
            wyj = gp.wy[jj]

            q, a, c = _prefixsum_window(gp.tiles_x, gp.rings_x, gp.x_period, h)
            if q > 0
                den_all = q * (use_mask_den ? _scan_diff(Pd, Nx + 1, 1, jj) : _scan_diff(Pwx, Nx + 1, 1))
                for i in 1:Nx
                    den[i, j] += wyj * den_all
                end
            end
            if wcell[b + 1, j] >= 0
                w = wcell[b + 1, j]
                for i in 1:Nx
                    den[i, j] += wyj * (use_mask_den ? _band_diff(Pd.hi, Pd.lo, wrap, i, w, Nx, jj) :
                        _band_diff(Pwx.hi, Pwx.lo, wrap, i, w, Nx))
                end
            elseif -a <= c
                _walk_band(gp.xe, gp.src, x, wraps, gp.x_period, a, c, Nx) do i, n, hi, lo
                    den[i, j] += wyj * (use_mask_den ? _wrap_sum(Pd.hi, Pd.lo, n, hi, lo, Nx, jj) :
                        _wrap_sum(Pwx.hi, Pwx.lo, n, hi, lo, Nx))
                end
            end
        end
    end

    zerofill = gp.strategy isa ZeroFill
    zerofill && exterior !== nothing && (den .+= exterior)
    @inbounds for j in 1:Ny, i in 1:Nx
        keep = zerofill || FlowGeometries.Grids.isactive(grid, i, j)
        den[i, j] = keep ? _inv_mass(den[i, j]) : zero(T)
    end
    return den
end

"""
    prefixsum_fill_numerator!(fp, field, grid)

Refill the scratch numerator scan (cumulative `mask·field·wx` along each row) from `field`, in `O(N)`.
Must run before any `apply_prefixsum_tophat_row!` call for that field.
"""
function prefixsum_fill_numerator!(
    fp::PrefixSumTopHatPlan{T}, field::AbstractMatrix, grid::FlowGeometries.Grids.StructuredGrid,
) where {T<:AbstractFloat}
    _, Ny = FlowGeometries.Grids.size_tuple(grid)
    for j in 1:Ny
        prefixsum_fill_numerator_row!(fp, field, grid, j)
    end
    return nothing
end

"""
    prefixsum_fill_numerator_row!(fp, field, grid, j)

Fill row `j`'s numerator prefix scan. Writes only column `j` of the scratch, so it may run
concurrently across `j`.
"""
function prefixsum_fill_numerator_row!(
    fp::PrefixSumTopHatPlan{T}, field::AbstractMatrix, grid::FlowGeometries.Grids.StructuredGrid, j::Integer,
) where {T<:AbstractFloat}
    _prefixsum_scan_row!(_prefixsum_num(fp.scratch), _prefixsum_values(fp.scratch, 1), fp.grid_plan, field, grid, j)
    return nothing
end

# The scan itself, over one destination table, shared by the single-field and batched fills, and the cells'
# values in the scan's order where `V` holds them. Row 1 is the empty prefix, zero from construction, and is
# not written here.
@inline function _prefixsum_scan_row!(
    P::CompensatedScan{T}, V, gp::PrefixSumGridPlan{T}, field::AbstractMatrix,
    grid::FlowGeometries.Grids.StructuredGrid, j::Integer,
) where {T<:AbstractFloat}
    ne = length(gp.xe)
    src, mask, wx = gp.src, FlowGeometries.Grids.mask(grid), gp.wx
    Ph, Pl = P.hi, P.lo
    h = l = zero(T)
    @inbounds for k in 1:ne
        i = src[k]
        h, l = _scan_step(h, l, mask[i, j] ? T(field[i, j]) * wx[i] : zero(T))
        Ph[k + 1, j] = h
        Pl[k + 1, j] = l
    end
    _store_cell_values!(V, gp, field, mask, j)
    return nothing
end

@inline _store_cell_values!(::Nothing, gp, field, mask, j) = nothing
@inline function _store_cell_values!(V::AbstractMatrix{T}, gp::PrefixSumGridPlan, field, mask, j) where {T}
    src, wx = gp.src, gp.wx
    @inbounds for k in axes(V, 1)
        i = src[k]
        V[k, j] = mask[i, j] ? T(field[i, j]) * wx[i] : zero(T)
    end
    return nothing
end

"""
    prefixsum_fill_numerator_batch_row!(fp, fields, grid, j)

Fill row `j`'s numerator scan for every field of a batch, one slot each. Writes only column `j` of
each slot, so it may run concurrently across `j` once `_prefixsum_numerators!` has grown the slots.
"""
function prefixsum_fill_numerator_batch_row!(
    fp::PrefixSumTopHatPlan{T}, fields, grid::FlowGeometries.Grids.StructuredGrid, j::Integer,
) where {T<:AbstractFloat}
    Ps = _prefixsum_numerators!(fp.scratch, length(fields))
    gp = fp.grid_plan
    for m in eachindex(fields)
        _prefixsum_scan_row!(@inbounds(Ps[m]), _prefixsum_values(fp.scratch, m), gp, @inbounds(fields[m]), grid, j)
    end
    return nothing
end

function prefixsum_fill_numerator_batch!(
    fp::PrefixSumTopHatPlan{T}, fields, grid::FlowGeometries.Grids.StructuredGrid,
) where {T<:AbstractFloat}
    _prefixsum_numerators!(fp.scratch, length(fields))   # grow once, outside any parallel region
    _, Ny = FlowGeometries.Grids.size_tuple(grid)
    for j in 1:Ny
        prefixsum_fill_numerator_batch_row!(fp, fields, grid, j)
    end
    return nothing
end

# `ZeroFill` divides by the kernel's full mass and `Deformable` by the mass of the active in-domain
# cells. The two agree when every cell is active and no window leaves the grid, which is when `fp.bound`
# is false. The message interpolates nothing: `show(::DataType)` is a runtime dispatch that
# `JET.@test_opt` reports in every caller.
@noinline function _prefixsum_strategy_mismatch()
    throw(ArgumentError(
        "PrefixSumTopHatPlan is being applied with a different mask strategy than it was built for, on " *
        "a grid where the two normalize differently: a cell is inactive, or a window reaches past a " *
        "bounded edge. Its normalization is precomputed per scale from the grid, the mask and the " *
        "strategy, so one plan cannot serve both. Rebuild the plan with the `mask_strategy` you intend " *
        "to apply with.",
    ))
end

@inline function _prefixsum_check_strategy(fp::PrefixSumTopHatPlan, strategy::AbstractMaskStrategy)
    (!fp.bound || typeof(strategy) === typeof(fp.grid_plan.strategy)) || _prefixsum_strategy_mismatch()
    return nothing
end

"""
    apply_prefixsum_tophat_row!(out, grid, fp, strategy, j) -> out

Fill output row `j` from the prefix scans, which `prefixsum_fill_numerator!` must have filled for the
current field. For each band, two monotone pointers sweep axis 1 in `O(1)` amortized per point, so the
row costs `O(Nx·dj_lim)`. Writes only row `j` of `out`, so rows may run concurrently.

This sums the numerator only and scales it by the per-scale reciprocal window mass `fp.invden`.
"""
apply_prefixsum_tophat_row!(
    out::AbstractMatrix{T}, grid::FlowGeometries.Grids.StructuredGrid{T,G,2}, fp::PrefixSumTopHatPlan{T},
    strategy::AbstractMaskStrategy, j::Integer,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}} =
    _prefixsum_row!(out, _prefixsum_num(fp.scratch), _prefixsum_values(fp.scratch, 1), grid, fp, j)

# The row body, taking its numerator table and cell values explicitly so a batched apply can drive it once per
# field.
@inline _prefixsum_row!(
    out::AbstractMatrix{T}, Pn::CompensatedScan{T}, V,
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2}, fp::PrefixSumTopHatPlan{T}, j::Integer,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}} =
    _prefixsum_row!(out, Pn, V, grid, fp, fp.edge, j)

function _prefixsum_row!(
    out::AbstractMatrix{T}, Pn::CompensatedScan{T}, _,
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2}, fp::PrefixSumTopHatPlan{T}, ::Nothing, j::Integer,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    gp = fp.grid_plan
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    x = FlowGeometries.Grids.coordinates(grid, 1)
    invden, hwj, wcellj = fp.invden, fp.hw, fp.wcell
    Ph, Pl = Pn.hi, Pn.lo
    wrap = _prefixsum_band_wraps(gp)
    wraps = gp.tiles_x || gp.rings_x

    @inbounds for i in 1:Nx
        out[i, j] = zero(T)
    end

    # On a tiling axis-2 two bands naming one row are two of its images.
    nb = size(hwj, 1)
    row0 = _prefixsum_row0(gp.tiles_y, j, fp.dj_lim)
    @inbounds for b in 0:(nb - 1)
        hw = hwj[b + 1, j]
        hw < zero(T) && continue        # band empty, or off a non-periodic axis-2 edge
        jj = mod1(row0 + b, Ny)
        wyj = gp.wy[jj]

        q, a, c = _prefixsum_window(gp.tiles_x, gp.rings_x, gp.x_period, hw)
        if q > 0
            # Whole periods in support: `q` row totals.
            num_all = q * _scan_diff(Ph, Pl, Nx + 1, 1, jj)
            for i in 1:Nx
                out[i, j] += wyj * num_all
            end
        end
        if wcellj[b + 1, j] >= 0
            _band_sum!(view(out, :, j), Ph, Pl, wrap, wcellj[b + 1, j], wyj, Nx, jj)
        elseif -a <= c
            _walk_band(gp.xe, gp.src, x, wraps, gp.x_period, a, c, Nx) do i, n, hi, lo
                out[i, j] += wyj * _wrap_sum(Ph, Pl, n, hi, lo, Nx, jj)
            end
        end
    end

    # `invden` is zero at a target the strategy skips.
    @inbounds @simd for i in 1:Nx
        out[i, j] *= invden[i, j]
    end
    return out
end

function _prefixsum_row!(
    out::AbstractMatrix{T}, Pn::CompensatedScan{T}, V::AbstractMatrix,
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2}, fp::PrefixSumTopHatPlan{T}, edge::PrefixEdgeCells, j::Integer,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    @inbounds for i in 1:Nx
        out[i, j] = zero(T)
    end
    _overlap_row_sum!(_entry_row(out, j, edge.rev, Nx), Pn, V, fp, edge, j, Nx, Ny)
    invden = fp.invden
    @inbounds @simd for i in 1:Nx
        out[i, j] *= invden[i, j]
    end
    return out
end

# Row `j` of `A` in the scan's ascending axis-1 order.
@inline _entry_row(A::AbstractMatrix, j::Integer, rev::Bool, Nx::Int) = rev ? view(A, Nx:-1:1, j) : view(A, :, j)

"""
    _overlap_row_sum!(oc, Pn, V, fp, edge, j, Nx, Ny) -> oc

Add target row `j`'s weighted sums under a [`PrefixEdgeCells`](@ref) plan into `oc`, the row in the scan's ascending
order: per band, the cells wholly inside about the target and across a regional seam from the scan `Pn`, then the
cells the disk's edge cuts from their values `V`. The device sweep adds in this order.
"""
function _overlap_row_sum!(
    oc::AbstractVector{T}, Pn::CompensatedScan, V::AbstractMatrix, fp::PrefixSumTopHatPlan, edge::PrefixEdgeCells,
    j::Integer, Nx::Int, Ny::Int,
) where {T}
    gp = fp.grid_plan
    Ph, Pl = Pn.hi, Pn.lo
    hwj, wcellj = fp.hw, fp.wcell
    nb = size(hwj, 1)
    row0 = _prefixsum_row0(gp.tiles_y, j, fp.dj_lim)
    @inbounds for b in 0:(nb - 1)
        hwj[b + 1, j] < zero(eltype(hwj)) && continue
        jj = mod1(row0 + b, Ny)
        wyj = T(gp.wy[jj])
        w = wcellj[b + 1, j]
        w >= 0 && _band_sum!(oc, Ph, Pl, edge.wrap, w, wyj, Nx, jj)
        fa = edge.far[b + 1, j]
        fa > 0 && _overlap_far!(oc, Ph, Pl, fa, wyj, Nx, jj)
    end
    @inbounds for s in edge.ptr[j]:(edge.ptr[j + 1] - 1)
        jr = j + edge.dj[s]
        if jr < 1 || jr > Ny
            gp.tiles_y || continue
            jr = mod1(jr, Ny)
        end
        _edge_cell!(oc, V, T(edge.w[s]), edge.di[s], edge.wrap, Nx, jr)
    end
    return oc
end

# Target `e` sums the cells `e − w : e + w`: clamped to the axis, or modulo its length where it closes, as `q`
# whole turns and a remainder where the span exceeds one. The targets whose run stays inside the axis take one
# scan difference.
@inline function _band_sum!(oc::AbstractVector{T}, Ph, Pl, wrap::Bool, w::Int, wyj::T, Nx::Int, jj::Int) where {T}
    if !wrap
        ilo = min(w + 1, Nx + 1)
        ihi = max(Nx - w, 0)
        @inbounds for e in 1:min(w, Nx)
            oc[e] += wyj * _scan_diff(Ph, Pl, min(Nx, e + w) + 1, 1, jj)
        end
        @inbounds @simd for e in ilo:ihi
            oc[e] += wyj * _scan_diff(Ph, Pl, e + w + 1, e - w, jj)
        end
        @inbounds for e in max(ihi + 1, w + 1):Nx
            oc[e] += wyj * _scan_diff(Ph, Pl, Nx + 1, max(1, e - w), jj)
        end
    elseif 2w + 1 <= Nx
        @inbounds @simd for e in 1:w
            oc[e] += wyj * _wrap_sum(Ph, Pl, 1, e + w + 1, Nx + e - w, Nx, jj)
        end
        @inbounds @simd for e in (w + 1):(Nx - w)
            oc[e] += wyj * _scan_diff(Ph, Pl, e + w + 1, e - w, jj)
        end
        @inbounds @simd for e in (Nx - w + 1):Nx
            oc[e] += wyj * _wrap_sum(Ph, Pl, 1, e + w - Nx + 1, e - w, Nx, jj)
        end
    else
        q, rc = divrem(2w + 1, Nx)
        tot = q * _scan_diff(Ph, Pl, Nx + 1, 1, jj)
        @inbounds for e in 1:Nx
            lo = mod1(e - w, Nx)
            hi = lo + rc - 1
            part = rc == 0 ? zero(T) : hi <= Nx ? _scan_diff(Ph, Pl, hi + 1, lo, jj) :
                _scan_diff(Ph, Pl, Nx + 1, lo, jj) + _scan_diff(Ph, Pl, hi - Nx + 1, 1, jj)
            oc[e] += wyj * (tot + part)
        end
    end
    return oc
end

# Target `e` sums the cells at offsets `fa` and more away along a regional box, across its seam.
@inline function _overlap_far!(oc::AbstractVector{T}, Ph, Pl, fa::Int, wyj::T, Nx::Int, jj::Int) where {T}
    @inbounds for e in 1:Nx
        s = zero(T)
        e + fa <= Nx && (s += _scan_diff(Ph, Pl, Nx + 1, e + fa, jj))
        e - fa >= 1 && (s += _scan_diff(Ph, Pl, e - fa + 1, 1, jj))
        oc[e] += wyj * s
    end
    return oc
end

# Target `e` takes `w` times the value of the cell at offset `d` in row `jj` of `V`, absent off a bounded axis and
# modulo the length of one that closes.
@inline function _edge_cell!(oc::AbstractVector{T}, V::AbstractMatrix, w::T, d::Int, wrap::Bool, Nx::Int, jj::Int) where {T}
    if !wrap
        @inbounds @simd for e in max(1, 1 - d):min(Nx, Nx - d)
            oc[e] += w * V[e + d, jj]
        end
    else
        r = mod(d, Nx)
        @inbounds @simd for e in 1:(Nx - r)
            oc[e] += w * V[e + r, jj]
        end
        @inbounds @simd for e in (Nx - r + 1):Nx
            oc[e] += w * V[e + r - Nx, jj]
        end
    end
    return oc
end

"""
    apply_prefixsum_tophat!(out, field, grid, fp, strategy) -> out

Whole-grid exact top-hat convolve: one `O(N)` prefix-sum pass over the field, then one `O(N·dj_lim)`
two-pointer sweep.
"""
function apply_prefixsum_tophat!(
    out::AbstractMatrix{T}, field::AbstractMatrix, grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    fp::PrefixSumTopHatPlan{T}, strategy::AbstractMaskStrategy,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    _, Ny = FlowGeometries.Grids.size_tuple(grid)
    _prefixsum_check_strategy(fp, strategy)
    prefixsum_fill_numerator!(fp, field, grid)
    for j in 1:Ny
        apply_prefixsum_tophat_row!(out, grid, fp, strategy, j)
    end
    return out
end

"""
    apply_prefixsum_tophat_batch_row!(outs, grid, fp, strategy, j) -> outs

Fill row `j` of every output in a batch from the per-field prefix scans, which
`prefixsum_fill_numerator_batch!` must have filled. Where `fp.uniform_axis`, each field runs the
single-field row body; otherwise each band's two-pointer walk runs once and feeds every field. Writes
only row `j` of each output.
"""
function apply_prefixsum_tophat_batch_row!(
    outs, grid::FlowGeometries.Grids.StructuredGrid{T,G,2}, fp::PrefixSumTopHatPlan{T},
    strategy::AbstractMaskStrategy, j::Integer,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    Ps = fp.scratch.prefix_nums
    if fp.uniform_axis
        for m in eachindex(outs)
            _prefixsum_row!(@inbounds(outs[m]), @inbounds(Ps[m]), _prefixsum_values(fp.scratch, m), grid, fp, j)
        end
        return outs
    end

    gp = fp.grid_plan
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    x = FlowGeometries.Grids.coordinates(grid, 1)
    invden, hwj, wcellj = fp.invden, fp.hw, fp.wcell
    wrap = _prefixsum_band_wraps(gp)
    wraps = gp.tiles_x || gp.rings_x

    for m in eachindex(outs)
        o = @inbounds outs[m]
        @inbounds for i in 1:Nx
            o[i, j] = zero(T)
        end
    end

    nb = size(hwj, 1)
    row0 = _prefixsum_row0(gp.tiles_y, j, fp.dj_lim)
    @inbounds for b in 0:(nb - 1)
        hw = hwj[b + 1, j]
        hw < zero(T) && continue
        jj = mod1(row0 + b, Ny)
        wyj = gp.wy[jj]

        q, a, c = _prefixsum_window(gp.tiles_x, gp.rings_x, gp.x_period, hw)
        if q > 0
            for m in eachindex(outs)
                o = outs[m]
                num_all = q * _scan_diff(Ps[m], Nx + 1, 1, jj)
                for i in 1:Nx
                    o[i, j] += wyj * num_all
                end
            end
        end
        if wcellj[b + 1, j] >= 0
            # The bands the single-field row takes at constant width, taken the same way.
            for m in eachindex(outs)
                _band_sum!(view(outs[m], :, j), Ps[m].hi, Ps[m].lo, wrap, wcellj[b + 1, j], wyj, Nx, jj)
            end
        elseif -a <= c
            _walk_band(gp.xe, gp.src, x, wraps, gp.x_period, a, c, Nx) do i, n, hi, lo
                for m in eachindex(outs)
                    outs[m][i, j] += wyj * _wrap_sum(Ps[m].hi, Ps[m].lo, n, hi, lo, Nx, jj)
                end
            end
        end
    end

    for m in eachindex(outs)
        o = @inbounds outs[m]
        @inbounds @simd for i in 1:Nx
            o[i, j] *= invden[i, j]
        end
    end
    return outs
end

"""
    apply_prefixsum_tophat_batch!(outs, fields, grid, fp, strategy) -> outs

Whole-grid batched top-hat convolve: one prefix scan per field, then a single sweep that walks each
support interval once and feeds every field from it. Where [`_prefixsum_batch_fuses`](@ref) is false,
each field is applied on its own.
"""
function apply_prefixsum_tophat_batch!(
    outs, fields, grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    fp::PrefixSumTopHatPlan{T}, strategy::AbstractMaskStrategy,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    if !_prefixsum_batch_fuses(fp)
        for k in eachindex(outs)
            apply_prefixsum_tophat!(@inbounds(outs[k]), @inbounds(fields[k]), grid, fp, strategy)
        end
        return outs
    end
    _, Ny = FlowGeometries.Grids.size_tuple(grid)
    _prefixsum_check_strategy(fp, strategy)
    prefixsum_fill_numerator_batch!(fp, fields, grid)
    for j in 1:Ny
        apply_prefixsum_tophat_batch_row!(outs, grid, fp, strategy, j)
    end
    return outs
end

"""
    _prefixsum_batch_fuses(fp) -> Bool

Whether a batched apply shares one support walk across its fields: true unless every band is
constant-width (`fp.uniform_axis`), where no band has a walk to share.
"""
@inline _prefixsum_batch_fuses(fp::PrefixSumTopHatPlan) = !fp.uniform_axis

"""
    apply_footprint!(out, field, grid, fp::PrefixSumTopHatPlan, strategy, periodic_x, periodic_y) -> out

Whole-grid convolve with a prefix-sum plan, as `apply_prefixsum_tophat!`. `periodic_x`/`periodic_y`
are unused: the plan fixes periodicity, and the matching extended coordinates, from the grid at build
time.
"""
apply_footprint!(
    out::AbstractMatrix{T}, field::AbstractMatrix, grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    fp::PrefixSumTopHatPlan{T}, strategy::AbstractMaskStrategy, periodic_x::Bool, periodic_y::Bool,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}} =
    apply_prefixsum_tophat!(out, field, grid, fp, strategy)

"""
    _structured_footprint(::Tuple, grid::StructuredGrid{T,<:AbstractGeometry,2}, kernel::TopHatKernel, scale; kwargs...)

The top-hat on a rectilinear 2D `StructuredGrid`; keywords as for `_build_prefixsum_tophat`. Under
[`Kernels.CellCentre`](@ref), the prefix-sum engine at any spacing. Under [`Kernels.CellOverlap`](@ref), the
prefix-sum engine with its edge where axis 1 is uniform, so a row's cut cells are the same about every target,
and the cell-by-cell plan with the overlap rule otherwise.
"""
_structured_footprint(
    ::Tuple, grid::FlowGeometries.Grids.StructuredGrid{T,G,2}, kernel::Kernels.TopHatKernel{Kernels.CellCentre},
    scale::T; kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}} =
    _build_prefixsum_tophat(grid, kernel, scale; kwargs...)

_structured_footprint(
    ::Tuple, grid::FlowGeometries.Grids.StructuredGrid{T,G,2}, kernel::Kernels.TopHatKernel{Kernels.CellOverlap},
    scale::T; kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}} =
    _build_footprint_scattered(grid, kernel, scale; kwargs...)

# A spheroid: the centre test by the cell-by-cell plan at the geodesic distance, since the geodesic disk has no
# closed band along a row; the cell overlap by the prefix sums where longitude is uniform, each row's fractions shared
# round it, and cell by cell otherwise.
_structured_footprint(
    ::Tuple, grid::FlowGeometries.Grids.StructuredGrid{T,G,2}, kernel::Kernels.TopHatKernel{Kernels.CellCentre}, scale::T;
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractEllipsoidalGeometry{T}} =
    _build_footprint_scattered(grid, kernel, scale; kwargs...)

_structured_footprint(
    ::Tuple{FlowGeometries.Axes.UniformSpacing,Any}, grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    kernel::Kernels.TopHatKernel{Kernels.CellCentre}, scale::T; kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractEllipsoidalGeometry{T}} =
    _build_footprint_scattered(grid, kernel, scale; kwargs...)

_structured_footprint(
    ::Tuple, grid::FlowGeometries.Grids.StructuredGrid{T,G,2}, kernel::Kernels.TopHatKernel{Kernels.CellOverlap}, scale::T;
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractEllipsoidalGeometry{T}} =
    _build_footprint_scattered(grid, kernel, scale; kwargs...)

_structured_footprint(
    ::Tuple{FlowGeometries.Axes.UniformSpacing,Any}, grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    kernel::Kernels.TopHatKernel{Kernels.CellOverlap}, scale::T; kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractEllipsoidalGeometry{T}} =
    _build_prefixsum_tophat(grid, kernel, scale; kwargs...)

# A uniform axis 1 is also what the banded engine takes for any kernel, so a top-hat there matches both;
# these methods are more specific than either and keep it on the prefix-sum path.
_structured_footprint(
    ::Tuple{FlowGeometries.Axes.UniformSpacing,Any}, grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    kernel::Kernels.TopHatKernel{Kernels.CellCentre}, scale::T; kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}} =
    _build_prefixsum_tophat(grid, kernel, scale; kwargs...)

_structured_footprint(
    ::Tuple{FlowGeometries.Axes.UniformSpacing,Any}, grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    kernel::Kernels.TopHatKernel{Kernels.CellOverlap}, scale::T; kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}} =
    _build_prefixsum_tophat(grid, kernel, scale; kwargs...)
