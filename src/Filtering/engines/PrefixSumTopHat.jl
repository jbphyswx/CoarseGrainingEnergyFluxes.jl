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

Holds the separable measure factors (`wx`,`wy`), the extended axis-1 coordinate array the two-pointer
sweep walks (tripled when axis 1 is periodic, so a wrapped support interval is still one contiguous
run), and the two measure scans the window mass is formed from: `prefix_wx` of `wx` along the extended
axis (`ZeroFill`), and `prefix_den` of `mask·wx` per row (`Deformable` on a masked grid, `nothing`
otherwise).
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
    xe::VT              # extended axis-1 coordinates (nrep*Nx), strictly increasing
    src::BT             # xe[k] belongs to real axis-1 index src[k]
    prefix_wx::PW       # scan of wx along the extended axis (nrep*Nx+1) — ZeroFill denominator
    prefix_den::DT      # per-row scan of mask·wx (nrep*Nx+1 × Ny), or nothing (Deformable only)
end

"""
    PrefixSumScratch{T,S,VS} <: AbstractFilterScratch

The prefix-sum engine's per-apply buffers: one [`CompensatedScan`](@ref) of `mask·field·wx` per field
in flight. Every apply refills them, so one set serves every scale of a sweep, one apply at a time.
Each row of a scan is written independently, so the numerator fill may run concurrently by row.

A single-field apply uses slot 1, a batched apply of `K` fields slots `1:K`; `_prefixsum_numerators!`
adds missing slots.
"""
struct PrefixSumScratch{
    T<:AbstractFloat, S<:CompensatedScan{T}, VS<:AbstractVector{S},
} <: AbstractFilterScratch
    prefix_nums::VS     # slot k: per-row scan of mask·fieldₖ·wx (nrep*Nx+1 × Ny)
end

# Slot 1 always exists, so the single-field path never checks.
@inline _prefixsum_num(sc::PrefixSumScratch) = @inbounds sc.prefix_nums[1]

"""
    _prefixsum_numerators!(sc, K) -> sc.prefix_nums

Grow `sc` to at least `K` numerator slots and return them. A new slot is a zeroed `similar` of slot
1, since the fill never writes row 1, the empty prefix. Call it before any parallel region that uses
the slots: growing `push!`es.
"""
function _prefixsum_numerators!(sc::PrefixSumScratch{T}, K::Integer) where {T<:AbstractFloat}
    while length(sc.prefix_nums) < K
        P1 = @inbounds sc.prefix_nums[1]
        push!(sc.prefix_nums, CompensatedScan(fill!(similar(P1.hi), zero(T)), fill!(similar(P1.lo), zero(T))))
    end
    return sc.prefix_nums
end

"""
    PrefixSumTopHatPlan{T,GP,SC,MT,WT}

Exact `O(N·dj_lim)` top-hat footprint for a rectilinear 2D `StructuredGrid`, holding the state that
depends on the filter scale ℓ. `grid_plan` and `scratch` may be shared with the other scales of a
sweep.

- `invden`: the reciprocal window mass per point, a function of the grid, the mask, the strategy and ℓ.
- `hw`/`wcell`: the support half-width per `(band, row)` and, on a uniform ascending axis, the
  constant cell half-width it implies.

`invden` is built for one mask strategy: where the two strategies normalize differently (`bound`), the
apply throws unless its `strategy` has the type of `grid_plan.strategy`.
"""
struct PrefixSumTopHatPlan{
    T<:AbstractFloat,
    GP<:PrefixSumGridPlan{T},
    SC<:PrefixSumScratch{T},
    MT<:AbstractMatrix{T},
    WT<:AbstractMatrix{Int},
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

    # Extended axis-1 coordinates, ascending: one copy, or three shifted by −P, 0, +P on a periodic axis
    # or a longitude (P exceeds the axis extent), so a wrapped support interval is one contiguous run.
    # `src` maps each entry to its real axis index, which a descending axis permutes.
    nrep = ((periodic_x && Nx > 1) || rings) ? 3 : 1
    ne = nrep * Nx
    ord = issorted(FlowGeometries.Grids.coordinates(grid, 1)) ? collect(1:Nx) : sortperm(FlowGeometries.Grids.coordinates(grid, 1))
    xe = Vector{T}(undef, ne)
    src = Vector{Int}(undef, ne)
    @inbounds for r in 0:(nrep - 1), t in 1:Nx
        k = r * Nx + t
        i = ord[t]
        xe[k] = FlowGeometries.Grids.coordinates(grid, 1)[i] + (nrep == 3 ? (r - 1) * x_period : zero(T))
        src[k] = i
    end

    prefix_wx = CompensatedScan(T, ne + 1)
    let h = zero(T), l = zero(T)
        @inbounds for k in 1:ne
            h, l = _scan_step(h, l, T(wx[src[k]]))
            prefix_wx.hi[k + 1] = h
            prefix_wx.lo[k + 1] = l
        end
    end

    masked = !all(FlowGeometries.Grids.mask(grid))

    prefix_den = if mask_strategy isa Deformable && masked
        P = CompensatedScan(T, ne + 1, Ny)
        mask = FlowGeometries.Grids.mask(grid)
        @inbounds for j in 1:Ny
            h = l = zero(T)
            for k in 1:ne
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

    tiles_x = nrep == 3 && is_cartesian
    rings_x = rings
    tiles_y = periodic_y && Ny > 1 && is_cartesian && y_period > zero(T)
    return PrefixSumGridPlan(
        periodic_x, periodic_y, masked, x_period, y_period, tiles_x, rings_x, tiles_y, mask_strategy,
        wx, wy, xe, src, prefix_wx, prefix_den,
    )
end

"""
    _prefixsum_scratch(gp::PrefixSumGridPlan, grid) -> PrefixSumScratch

Allocate the engine's per-apply numerator scan. Sized by the grid alone, so one scratch serves every
scale — but only one apply at a time; see [`AbstractFilterScratch`](@ref).
"""
function _prefixsum_scratch(
    gp::PrefixSumGridPlan{T}, grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    _, Ny = FlowGeometries.Grids.size_tuple(grid)
    return PrefixSumScratch([CompensatedScan(T, length(gp.xe) + 1, Ny)])
end

# An ascending `Range` axis 1, where a band shorter than a period is a constant ±w cells.
@inline _prefixsum_uniform_axis(::PrefixSumGridPlan, x) = x isa AbstractRange && step(x) > zero(eltype(x))

# The extended-axis entries `lo:hi` of a constant-width band at target `i`: `i ± w` clamped to the axis,
# or, where axis 1 tiles, `±w` about the target's entry in the middle copy.
@inline _prefixsum_wcell_range(tiles_x::Bool, i::Int, w::Int, Nx::Int) =
    tiles_x ? (Nx + i - w, Nx + i + w) : (max(1, i - w), min(Nx, i + w))

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
and `mask_strategy` is then unused. `kwargs...` absorbs and ignores `cache_strategy`/`cache_byte_budget`.
"""
function _build_prefixsum_tophat(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    kernel::Kernels.TopHatKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    grid_plan::Union{Nothing,PrefixSumGridPlan} = nothing,
    scratch::Union{Nothing,PrefixSumScratch} = nothing,
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    gp = grid_plan === nothing ?
        _build_prefixsum_grid_plan(grid; mask_strategy = mask_strategy) : grid_plan
    sc = scratch === nothing ? _prefixsum_scratch(gp, grid) : scratch

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
    uniform = _prefixsum_uniform_axis(gp, x)
    # A longitude band shorter than the gap across the grid's seam, and than half the ring, keeps to the
    # grid's own columns.
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
            if uniform && h < seam && !((gp.rings_x || gp.tiles_x) && 2h >= gp.x_period)
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

    return PrefixSumTopHatPlan(gp, sc, rad, dj_lim, every_band, bound, invden, hw, wcell)
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
    Pd, Pwx, src, xe = gp.prefix_den, gp.prefix_wx, gp.src, gp.xe
    ne = length(xe)
    use_mask_den = !(gp.strategy isa ZeroFill) && Pd !== nothing
    nb = size(hw, 1)

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
                    lo, hi = _prefixsum_wcell_range(gp.tiles_x, i, w, Nx)
                    den[i, j] += wyj * (use_mask_den ? _scan_diff(Pd, hi + 1, lo, jj) : _scan_diff(Pwx, hi + 1, lo))
                end
            elseif -a <= c
                lo = 1
                hi = 0
                for t in 1:Nx
                    i = src[t]
                    xc = x[i]
                    xlo = xc - a
                    xhi = xc + c
                    while lo <= ne && xe[lo] < xlo
                        lo += 1
                    end
                    while hi < ne && xe[hi + 1] <= xhi
                        hi += 1
                    end
                    if hi >= lo
                        den[i, j] += wyj * (use_mask_den ? _scan_diff(Pd, hi + 1, lo, jj) : _scan_diff(Pwx, hi + 1, lo))
                    end
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
    _prefixsum_scan_row!(_prefixsum_num(fp.scratch), fp.grid_plan, field, grid, j)
    return nothing
end

# The scan itself, over one destination table, shared by the single-field and batched fills. Row 1 is
# the empty prefix, zero from construction, and is not written here.
@inline function _prefixsum_scan_row!(
    P::CompensatedScan{T}, gp::PrefixSumGridPlan{T}, field::AbstractMatrix,
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
        _prefixsum_scan_row!(@inbounds(Ps[m]), gp, @inbounds(fields[m]), grid, j)
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
    _prefixsum_row!(out, _prefixsum_num(fp.scratch), grid, fp, j)

# The row body, taking its numerator table explicitly so a batched apply can drive it once per field.
function _prefixsum_row!(
    out::AbstractMatrix{T}, Pn::CompensatedScan{T},
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2}, fp::PrefixSumTopHatPlan{T}, j::Integer,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    gp = fp.grid_plan
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    ne = length(gp.xe)
    x, xe, src = FlowGeometries.Grids.coordinates(grid, 1), gp.xe, gp.src
    invden, hwj, wcellj = fp.invden, fp.hw, fp.wcell
    Ph, Pl = Pn.hi, Pn.lo

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
            # Whole periods in support: `q` totals of one replica.
            num_all = q * _scan_diff(Ph, Pl, Nx + 1, 1, jj)
            for i in 1:Nx
                out[i, j] += wyj * num_all
            end
        end
        if wcellj[b + 1, j] >= 0
            _prefixsum_band!(out, Ph, Pl, gp.tiles_x, wcellj[b + 1, j], wyj, Nx, jj, j)
        elseif -a <= c
            lo = 1
            hi = 0
            # Targets in ascending coordinate order, `src[1:Nx]`: the pointers only advance, so the target
            # coordinate must increase.
            for t in 1:Nx
                i = src[t]
                xc = x[i]
                xlo = xc - a
                xhi = xc + c
                while lo <= ne && xe[lo] < xlo
                    lo += 1
                end
                while hi < ne && xe[hi + 1] <= xhi
                    hi += 1
                end
                if hi >= lo
                    out[i, j] += wyj * _scan_diff(Ph, Pl, hi + 1, lo, jj)
                end
            end
        end
    end

    # `invden` is zero at a target the strategy skips.
    @inbounds @simd for i in 1:Nx
        out[i, j] *= invden[i, j]
    end
    return out
end

@inline _prefixsum_band!(out, Ph, Pl, tiles_x::Bool, w::Int, wyj, Nx::Int, jj::Int, j::Integer) =
    tiles_x ? _prefixsum_tiled_band!(out, Ph, Pl, w, wyj, Nx, jj, j) :
              _prefixsum_wcell_band!(out, Ph, Pl, w, wyj, Nx, jj, j)

# A constant-width band on a tiling axis: target `i` sums the extended entries `Nx+i-w:Nx+i+w`.
@inline function _prefixsum_tiled_band!(
    out::AbstractMatrix{T}, Ph, Pl, w::Int, wyj::T, Nx::Int, jj::Int, j::Integer,
) where {T<:AbstractFloat}
    @inbounds @simd for i in 1:Nx
        out[i, j] += wyj * _scan_diff(Ph, Pl, Nx + i + w + 1, Nx + i - w, jj)
    end
    return nothing
end

# A constant-width band on a uniform ascending axis: target `i` sums cells `i-w:i+w`, clamped to the axis.
@inline function _prefixsum_wcell_band!(
    out::AbstractMatrix{T}, Ph, Pl, w::Int, wyj::T, Nx::Int, jj::Int, j::Integer,
) where {T<:AbstractFloat}
    ilo = min(w + 1, Nx + 1)
    ihi = max(Nx - w, 0)
    @inbounds for i in 1:min(w, Nx)                     # left edge: lo clamps to 1
        hi = min(Nx, i + w)
        out[i, j] += wyj * _scan_diff(Ph, Pl, hi + 1, 1, jj)
    end
    @inbounds @simd for i in ilo:ihi                    # interior: no clamping at all
        out[i, j] += wyj * _scan_diff(Ph, Pl, i + w + 1, i - w, jj)
    end
    @inbounds for i in max(ihi + 1, w + 1):Nx           # right edge: hi clamps to Nx
        lo = max(1, i - w)
        out[i, j] += wyj * _scan_diff(Ph, Pl, Nx + 1, lo, jj)
    end
    return nothing
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
            _prefixsum_row!(@inbounds(outs[m]), @inbounds(Ps[m]), grid, fp, j)
        end
        return outs
    end

    gp = fp.grid_plan
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    ne = length(gp.xe)
    x, xe, src = FlowGeometries.Grids.coordinates(grid, 1), gp.xe, gp.src
    invden, hwj, wcellj = fp.invden, fp.hw, fp.wcell

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
                _prefixsum_band!(outs[m], Ps[m].hi, Ps[m].lo, gp.tiles_x, wcellj[b + 1, j], wyj, Nx, jj, j)
            end
        elseif -a <= c
            lo = 1
            hi = 0
            for t in 1:Nx
                i = src[t]
                xc = x[i]
                xlo = xc - a
                xhi = xc + c
                while lo <= ne && xe[lo] < xlo
                    lo += 1
                end
                while hi < ne && xe[hi + 1] <= xhi
                    hi += 1
                end
                if hi >= lo
                    for m in eachindex(outs)
                        outs[m][i, j] += wyj * _scan_diff(Ps[m], hi + 1, lo, jj)
                    end
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
    build_footprint(grid::StructuredGrid{T,<:AbstractGeometry,2}, kernel::TopHatKernel, scale; kwargs...) -> PrefixSumTopHatPlan

The exact prefix-sum top-hat engine on any rectilinear 2D `StructuredGrid`; keywords as for
`_build_prefixsum_tophat`. More specific in the kernel than the generic 2D methods, so every
`TopHatKernel` on such a grid dispatches here.
"""
function build_footprint(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    kernel::Kernels.TopHatKernel,
    scale::T;
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    return _build_prefixsum_tophat(grid, kernel, scale; kwargs...)
end

# A `Range` axis 1 is also what the banded method takes for any kernel, so a top-hat there matches both;
# this method is more specific than either and keeps it on the prefix-sum path.
function build_footprint(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2,S,TP,<:Tuple{AbstractRange{T},AbstractVector{T}}},
    kernel::Kernels.TopHatKernel,
    scale::T;
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}, S, TP<:NTuple{2,FlowGeometries.Grids.AbstractTopology}}
    return _build_prefixsum_tophat(grid, kernel, scale; kwargs...)
end
