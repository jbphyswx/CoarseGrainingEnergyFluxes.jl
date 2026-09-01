# ---------------------------------------------------------------------------
# Exact top-hat filtering on any rectilinear 2D `StructuredGrid`, in O(N·dj_lim) rather than
# O(N·di_lim·dj_lim). Requires `TopHatKernel`, whose weight is constant inside its support, so the
# weighted window sum is a plain interval sum. Two properties of a rectilinear grid make that sum O(1):
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
# ---------------------------------------------------------------------------

"""
    PrefixSumGridPlan{T,VT,DT,BT,WX,WY,MS} <: AbstractGridPlan

The half of the prefix-sum top-hat engine that the filter scale does not reach: everything fixed by
the grid, its mask and the mask strategy. One instance serves every scale of a sweep.

Holds the separable measure factors (`wx`,`wy`), the extended axis-1 coordinate array the two-pointer
sweep walks (tripled when axis 1 is periodic, so a wrapped support interval is still one contiguous
run), and the two mask/measure prefix scans.

`prefix_den` (Deformable masking) is the prefix sum of `mask·wx`; `ZeroFill`'s denominator is
mask-independent and needs only the 1-D `prefix_wx`. Neither depends on ℓ, which is why an S-scale
sweep builds them once rather than S times.
"""
struct PrefixSumGridPlan{
    T<:AbstractFloat,
    VT<:AbstractVector{T},
    DT<:Union{Nothing,AbstractMatrix{T}},
    BT<:AbstractVector{Int},
    WX<:AbstractVector{T},
    WY<:AbstractVector{T},
    MS<:AbstractMaskStrategy,
} <: AbstractGridPlan
    periodic_x::Bool
    periodic_y::Bool
    masked::Bool        # grid has inactive cells, so Deformable genuinely needs `prefix_den`
    x_period::T
    y_period::T
    strategy::MS        # the strategy `prefix_den` — and hence every scale's `invden` — was built for
    # The grid's own measure factors, whatever it stores them as: a uniform axis carries one number
    # and a length, so these are not pinned to a dense vector.
    wx::WX              # axis-1 cell width;  measure[i,j] == wx[i]*wy[j]
    wy::WY              # axis-2 measure factor
    xe::VT              # extended axis-1 coordinates (nrep*Nx), strictly increasing
    src::BT             # xe[k] belongs to real axis-1 index src[k]
    prefix_wx::VT       # cumulative Σ wx along the extended axis (nrep*Nx+1) — ZeroFill denominator
    prefix_den::DT      # per-row cumulative Σ mask·wx (nrep*Nx+1 × Ny), or nothing (Deformable only)
end

"""
    PrefixSumScratch{T,MT,VM} <: AbstractFilterScratch

The prefix-sum engine's per-apply buffers: one cumulative `mask·field·wx` scan per field in flight.
They are refilled at the start of every apply, so their contents never outlive one call and one set
serves the whole sweep. Row-disjoint, so the threaded numerator fill writes them concurrently by row.

A single-field apply uses slot 1. A batched apply needs all `K` live at once so that one walk of the
support interval can feed every field — see [`apply_prefixsum_tophat_batch_row!`](@ref) — so slots are
added on demand and then reused, which keeps repeat applies allocation-free.
"""
struct PrefixSumScratch{
    T<:AbstractFloat, MT<:AbstractMatrix{T}, VM<:AbstractVector{MT},
} <: AbstractFilterScratch
    prefix_nums::VM     # slot k: per-row cumulative Σ mask·fieldₖ·wx (nrep*Nx+1 × Ny)
end

# Slot 1 always exists, so the single-field path never checks.
@inline _prefixsum_num(sc::PrefixSumScratch) = @inbounds sc.prefix_nums[1]

"""
    _prefixsum_numerators!(sc, K) -> the first `K` numerator slots

Allocates any slot that does not exist yet, by `similar` on slot 1 so a new buffer inherits whatever
array type the engine is actually running on rather than assuming a host `Matrix`. Row 1 of each scan
is the empty-prefix zero and is never written by the fill, so a fresh slot must be zeroed.

Growth happens once per (scratch, batch width) — `compute_Π!` asks for 2 and then 3 — so it is off the
repeated-apply path and the allocation gates still see zero.
"""
function _prefixsum_numerators!(sc::PrefixSumScratch{T}, K::Integer) where {T<:AbstractFloat}
    while length(sc.prefix_nums) < K
        push!(sc.prefix_nums, fill!(similar(@inbounds sc.prefix_nums[1]), zero(T)))
    end
    return sc.prefix_nums
end

"""
    PrefixSumTopHatPlan{T,GP,SC,MT,WT}

Exact `O(N·dj_lim)` top-hat footprint for a rectilinear 2D `StructuredGrid` — see the section comment
above for the derivation.

Only the ℓ-dependent state lives here. The grid half is reached through `grid_plan` and the per-apply
buffer through `scratch`, both shared with every other scale of the same sweep, so S scales hold one
copy of each rather than S.

Two field-independent quantities are built here, once per scale, so that the apply is numerator-only:

- `invden` — the reciprocal window mass, a function of the grid, the mask, the strategy and ℓ alone.
  Accumulating it alongside the numerator would double the arithmetic of the apply's inner loop and
  repeat that for each of the five to nine fields a flux calculation filters.
- `hw`/`wcell` — the per-`(band, row)` support half-width and, on a uniform ascending axis, the
  constant cell half-width it implies. `hw` calls the geometry's `metric_band`, a transcendental on
  the sphere, and `wcell` is found by a linear walk, so both are worth tabulating.

`invden` is built for one mask strategy, so the apply asserts the strategy it is handed matches
`grid_plan.strategy`. Build the plan with the strategy you intend to apply with.
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
    uniform_axis::Bool  # axis 1 is a bounded ascending Range: every interval is O(1), no walk anywhere
    invden::MT          # 1/(window mass) per point (Nx × Ny), zero where the point is inactive/empty
    hw::MT              # support half-width per (band, row) ((2dj_lim+1) × Ny); negative ⇒ band empty
    wcell::WT           # uniform-axis constant cell half-width per (band, row), else -1
end

"""
    _rectilinear_measure_factors(grid) -> (wx, wy)

The grid's own per-axis measure factors, so that `measure[i, j] == wx[i] * wy[j]` exactly. Read from
the stored `SeparableMeasure` rather than rebuilt here: rebuilding has to reproduce every convention
the measure was constructed under, and a degenerate direction is where that fails — a single-latitude
grid measures arc length `R·Δλ` and a single-longitude one `R·Δφ`, neither of which is the `R²cosφ`
area form.
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

    # The grid's own stored wrap length per direction — never re-derived from the samples, or the two
    # real-space engines wrap at different lengths on a stretched axis.
    x_period = (periodic_x && Nx > 1) ? T(FlowGeometries.Grids.period(grid, 1)) : zero(T)
    y_period = (periodic_y && Ny > 1) ? T(FlowGeometries.Grids.period(grid, 2)) : zero(T)

    wx, wy = _rectilinear_measure_factors(grid)

    # Extended axis-1 coordinates: one copy when non-periodic; three (shifted −P, 0, +P) when periodic,
    # so a wrapped support interval stays a single contiguous run and the two-pointer stays monotone.
    # The sweep searches `xe` by value (`x[i] ± hw`), so no index offset into the replicas is needed.
    # `xe` must be ascending — both the prefix sums and the sweep depend on it — while the axis itself
    # may be stored descending, so order the extension by coordinate and carry the real axis index in
    # `src`. Each replica is internally sorted and the period exceeds the axis extent, so concatenating
    # in shift order keeps `xe` globally sorted.
    nrep = (periodic_x && Nx > 1) ? 3 : 1
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

    prefix_wx = Vector{T}(undef, ne + 1)
    prefix_wx[1] = zero(T)
    @inbounds for k in 1:ne
        prefix_wx[k + 1] = prefix_wx[k] + wx[src[k]]
    end

    masked = !all(FlowGeometries.Grids.mask(grid))

    # Deformable's denominator depends only on the mask, so build it once here, never per apply.
    prefix_den = if mask_strategy isa Deformable && masked
        P = zeros(T, ne + 1, Ny)
        @inbounds for j in 1:Ny
            acc = zero(T)
            for k in 1:ne
                i = src[k]
                acc += FlowGeometries.Grids.mask(grid)[i, j] ? wx[i] : zero(T)
                P[k + 1, j] = acc
            end
        end
        P
    else
        nothing
    end

    return PrefixSumGridPlan(
        periodic_x, periodic_y, masked, x_period, y_period, mask_strategy,
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
    return PrefixSumScratch([zeros(T, length(gp.xe) + 1, Ny)])
end

# Which of the three interval branches a `(band, row)` takes. Written once and called from both the
# plan build (to accumulate the denominator) and the apply (to sum the numerator), so the two cannot
# drift apart and disagree about where a window starts.
@inline _prefixsum_spans_period(gp::PrefixSumGridPlan{T}, ne::Int, Nx::Int, hw::T) where {T} =
    gp.periodic_x && ne > Nx && T(2) * hw >= gp.x_period

@inline _prefixsum_uniform_axis(gp::PrefixSumGridPlan, x, ne::Int, Nx::Int) =
    !gp.periodic_x && ne == Nx && x isa AbstractRange && step(x) > zero(eltype(x))

"""
    _build_prefixsum_tophat(grid, kernel::TopHatKernel, scale; mask_strategy=ZeroFill(), kwargs...)

Build the exact `O(N·dj_lim)` prefix-sum top-hat plan for one scale. `grid_plan`/`scratch` let a sweep
hand in the shared pieces; omitted, they are built here for a standalone single-scale plan.
`kwargs...` absorbs (and ignores) `cache_strategy`/`cache_byte_budget` — there is no per-point
neighbour list here to cache at all.
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

    # Axis-2 band bound from the axis's own minimum gap, converted to a physical distance.
    min_dy = FlowGeometries.Grids.minimum_spacing(grid, 2)
    dy_phys = is_cartesian ? min_dy : FlowGeometries.Geometry.radius(FlowGeometries.Grids.grid_geometry(grid)) * min_dy
    dj_lim = (isfinite(dy_phys) && dy_phys > 0) ? min(Ny - 1, ceil(Int, rad / dy_phys)) : 0

    x = FlowGeometries.Grids.coordinates(grid, 1)
    y = FlowGeometries.Grids.coordinates(grid, 2)
    ne = length(gp.xe)
    nb = min(2 * dj_lim + 1, Ny)
    uniform = _prefixsum_uniform_axis(gp, x, ne, Nx)

    # `hw` needs `metric_band`, which is a transcendental on the sphere, and `wcell` was a linear walk
    # up from zero. Both are functions of the grid and `rad`, so they are tabulated per (band, row)
    # once here instead of being recomputed on every apply of every field.
    hw = fill(-one(T), nb, Ny)
    wcell = fill(-1, nb, Ny)
    @inbounds for j in 1:Ny
        y_t = y[j]
        for b in 0:(nb - 1)
            jj_raw = j - dj_lim + b
            (gp.periodic_y || (1 <= jj_raw <= Ny)) || continue
            jj = mod1(jj_raw, Ny)
            dy = y[jj] - y_t
            if gp.periodic_y && gp.y_period > zero(T)
                dy -= gp.y_period * round(dy / gp.y_period)
            end
            h = FlowGeometries.Connectivity.metric_band(grid, 1, y_t, y_t + dy, rad)
            h < zero(T) && continue
            hw[b + 1, j] = h
            if uniform
                # Same `≤` the pointer loop uses, so it makes the same boundary decision.
                w = 0
                while (w + 1) * step(x) <= h
                    w += 1
                end
                wcell[b + 1, j] = w
            end
        end
    end

    invden = _prefixsum_build_invden(grid, gp, dj_lim, hw, wcell)

    return PrefixSumTopHatPlan(gp, sc, rad, dj_lim, uniform, invden, hw, wcell)
end

"""
    _prefixsum_build_invden(grid, gp, dj_lim, hw, wcell) -> invden

Accumulate the window mass once per scale and store its reciprocal.

The mass depends on the grid, the mask, the mask strategy and ℓ — never on the field — so it belongs
here rather than in the apply, where accumulating it would double the inner loop's arithmetic and be
repeated for every field. Folding `isactive` and the degeneracy floor into the same table turns the
apply's epilogue from a branch and a division into one multiply.
"""
function _prefixsum_build_invden(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    gp::PrefixSumGridPlan{T},
    dj_lim::Int,
    hw::AbstractMatrix{T},
    wcell::AbstractMatrix{Int},
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    x = FlowGeometries.Grids.coordinates(grid, 1)
    Pd, Pwx, src, xe = gp.prefix_den, gp.prefix_wx, gp.src, gp.xe
    ne = length(xe)
    use_mask_den = !(gp.strategy isa ZeroFill) && Pd !== nothing
    nb = size(hw, 1)

    den = zeros(T, Nx, Ny)
    @inbounds for j in 1:Ny
        for b in 0:(nb - 1)
            h = hw[b + 1, j]
            h < zero(T) && continue
            jj = mod1(j - dj_lim + b, Ny)
            wyj = gp.wy[jj]

            if _prefixsum_spans_period(gp, ne, Nx, h)
                den_all = use_mask_den ? (Pd[Nx + 1, jj] - Pd[1, jj]) : (Pwx[Nx + 1] - Pwx[1])
                for i in 1:Nx
                    den[i, j] += wyj * den_all
                end
            elseif wcell[b + 1, j] >= 0
                w = wcell[b + 1, j]
                for i in 1:Nx
                    lo = max(1, i - w)
                    hi = min(Nx, i + w)
                    den[i, j] += wyj * (use_mask_den ? (Pd[hi + 1, jj] - Pd[lo, jj]) : (Pwx[hi + 1] - Pwx[lo]))
                end
            else
                lo = 1
                hi = 0
                for t in 1:Nx
                    i = src[t]
                    xc = x[i]
                    xlo = xc - h
                    xhi = xc + h
                    while lo <= ne && xe[lo] < xlo
                        lo += 1
                    end
                    while hi < ne && xe[hi + 1] <= xhi
                        hi += 1
                    end
                    if hi >= lo
                        den[i, j] += wyj * (use_mask_den ? (Pd[hi + 1, jj] - Pd[lo, jj]) : (Pwx[hi + 1] - Pwx[lo]))
                    end
                end
            end
        end
    end

    @inbounds for j in 1:Ny, i in 1:Nx
        den[i, j] = (FlowGeometries.Grids.isactive(grid, i, j) && den[i, j] > T(1e-15)) ?
            inv(den[i, j]) : zero(T)
    end
    return den
end

"""
    prefixsum_fill_numerator!(fp, field, grid)

The single O(N) per-apply pass: refill the scratch numerator scan (cumulative `mask·field·wx` per
row) from the current field. Must run before any `apply_prefixsum_tophat_row!` call for that field —
both the serial and threaded drivers guarantee this.
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

Fill row `j`'s numerator prefix scan. Writes only column `j` of the scratch, so rows are mutually
independent and this may be run concurrently across `j` (the threaded backend does exactly that).
"""
function prefixsum_fill_numerator_row!(
    fp::PrefixSumTopHatPlan{T}, field::AbstractMatrix, grid::FlowGeometries.Grids.StructuredGrid, j::Integer,
) where {T<:AbstractFloat}
    _prefixsum_scan_row!(_prefixsum_num(fp.scratch), fp.grid_plan, field, grid, j)
    return nothing
end

# The scan itself, over one destination table. Shared by the single-field and batched fills so the two
# cannot drift; `P[1, j]` is the empty-prefix zero and is deliberately not written here.
@inline function _prefixsum_scan_row!(
    P::AbstractMatrix{T}, gp::PrefixSumGridPlan{T}, field::AbstractMatrix,
    grid::FlowGeometries.Grids.StructuredGrid, j::Integer,
) where {T<:AbstractFloat}
    ne = length(gp.xe)
    src, mask, wx = gp.src, FlowGeometries.Grids.mask(grid), gp.wx
    acc = zero(T)
    @inbounds for k in 1:ne
        i = src[k]
        acc += mask[i, j] ? T(field[i, j]) * wx[i] : zero(T)
        P[k + 1, j] = acc
    end
    return nothing
end

"""
    prefixsum_fill_numerator_batch_row!(fp, fields, grid, j)

Fill row `j`'s numerator scan for every field of a batch, one table each. This is the whole of the
per-field work in a batched apply — measured at ~1% of it — which is why the batch shares the support
walk rather than the scan.
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

# The plan's reciprocal window mass `invden` is accumulated at build time under ONE mask strategy, so
# applying it under another would silently normalize by the wrong denominator. Checked once per apply,
# not per row.
#
# On an UNMASKED grid the strategies coincide exactly — `ZeroFill` divides by the total window mass and
# `Deformable` by the mass of the active cells, which is the same number when every cell is active — so
# one table serves both and any strategy is accepted. The restriction is therefore exactly as narrow as
# the mathematics requires: it bites only where the two denominators genuinely differ.
#
# `@noinline`, and the message interpolates nothing: interpolating `typeof(strategy)` pulls Base's
# dynamically-dispatched `show(::DataType)` into the call graph, so `JET.@test_opt` on any caller
# reports runtime dispatch even though the throw never executes.
@noinline function _prefixsum_strategy_mismatch()
    throw(ArgumentError(
        "PrefixSumTopHatPlan is being applied to a MASKED grid with a different mask strategy than it " *
        "was built for. Its normalization is precomputed per scale from the grid, the mask and the " *
        "strategy, and the two strategies divide by different masses wherever a cell is inactive, so " *
        "one plan cannot serve both. Rebuild the plan with the `mask_strategy` you intend to apply with.",
    ))
end

@inline function _prefixsum_check_strategy(fp::PrefixSumTopHatPlan, strategy::AbstractMaskStrategy)
    gp = fp.grid_plan
    (!gp.masked || typeof(strategy) === typeof(gp.strategy)) || _prefixsum_strategy_mismatch()
    return nothing
end

"""
    apply_prefixsum_tophat_row!(out, grid, fp, strategy, j) -> out

Fill output row `j` from the (already current) prefix sums. For each row offset in the band, sweeps
axis 1 with two MONOTONE pointers — `O(1)` amortized per point, not a per-point binary search — so the
whole row costs `O(Nx·dj_lim)`. Touches only row `j` of `out`, so rows may run concurrently.

This sums the NUMERATOR only. The window mass and its reciprocal are tabulated per scale in
`fp.invden` (they depend on the grid, mask, strategy and ℓ, never on the field), which halves the
inner loop and turns the epilogue into a multiply — and the saving repeats for every one of the five
to nine fields a flux calculation filters at each scale.
"""
apply_prefixsum_tophat_row!(
    out::AbstractMatrix{T}, grid::FlowGeometries.Grids.StructuredGrid{T,G,2}, fp::PrefixSumTopHatPlan{T},
    strategy::AbstractMaskStrategy, j::Integer,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}} =
    _prefixsum_row!(out, _prefixsum_num(fp.scratch), grid, fp, j)

# The row body, taking its numerator table explicitly so a batched apply can drive it once per field.
function _prefixsum_row!(
    out::AbstractMatrix{T}, Pn::AbstractMatrix{T},
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2}, fp::PrefixSumTopHatPlan{T}, j::Integer,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    gp = fp.grid_plan
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    ne = length(gp.xe)
    x, xe, src = FlowGeometries.Grids.coordinates(grid, 1), gp.xe, gp.src
    invden, hwj, wcellj = fp.invden, fp.hw, fp.wcell

    @inbounds for i in 1:Nx
        out[i, j] = zero(T)
    end

    # Visit each contributing row EXACTLY ONCE. `nb` is capped at `Ny` because a periodic axis-2 with a
    # band wider than the grid would otherwise map two different offsets onto the same row via `mod1`
    # and count it twice; capping at `Ny` consecutive raw offsets keeps `mod1` a bijection onto `1:Ny`.
    nb = min(2 * fp.dj_lim + 1, Ny)
    @inbounds for b in 0:(nb - 1)
        hw = hwj[b + 1, j]
        hw < zero(T) && continue        # band empty, or off a non-periodic axis-2 edge
        jj = mod1(j - fp.dj_lim + b, Ny)
        wyj = gp.wy[jj]

        if _prefixsum_spans_period(gp, ne, Nx, hw)
            # Support spans at least a full period: every cell of the row is included exactly once.
            # Summing over the tripled axis would triple-count, so use one replica's total directly.
            num_all = Pn[Nx + 1, jj] - Pn[1, jj]
            for i in 1:Nx
                out[i, j] += wyj * num_all
            end
        elseif wcellj[b + 1, j] >= 0
            # Uniform ascending axis: the two pointers advance by exactly one per target, so the window
            # is a constant ±w and the walk collapses. Peeling the interior leaves it branch-free.
            w = wcellj[b + 1, j]
            ilo = min(w + 1, Nx + 1)
            ihi = max(Nx - w, 0)
            for i in 1:min(w, Nx)                     # left edge: lo clamps to 1
                hi = min(Nx, i + w)
                out[i, j] += wyj * (Pn[hi + 1, jj] - Pn[1, jj])
            end
            @simd for i in ilo:ihi                    # interior: no clamping at all
                out[i, j] += wyj * (Pn[i + w + 1, jj] - Pn[i - w, jj])
            end
            for i in max(ihi + 1, w + 1):Nx           # right edge: hi clamps to Nx
                lo = max(1, i - w)
                out[i, j] += wyj * (Pn[Nx + 1, jj] - Pn[lo, jj])
            end
        else
            lo = 1
            hi = 0
            # Walk the TARGET index in ascending-coordinate order (`src[1:Nx]` is exactly that
            # permutation), not raw index order: the two pointers only ever advance forward, which
            # requires the target coordinate to increase monotonically. On a descending axis raw order
            # would decrease it and silently corrupt every interval.
            for t in 1:Nx
                i = src[t]
                xc = x[i]
                xlo = xc - hw
                xhi = xc + hw
                while lo <= ne && xe[lo] < xlo
                    lo += 1
                end
                while hi < ne && xe[hi + 1] <= xhi
                    hi += 1
                end
                if hi >= lo
                    out[i, j] += wyj * (Pn[hi + 1, jj] - Pn[lo, jj])
                end
            end
        end
    end

    # `invden` already carries the `isactive` test and the degeneracy floor, so an inactive or empty
    # point holds exactly zero there and needs no branch here.
    @inbounds @simd for i in 1:Nx
        out[i, j] *= invden[i, j]
    end
    return out
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

Fill row `j` of every output in a batch from the (already current) per-field prefix scans, walking the
support interval **once** for the whole batch.

That walk is the only thing a batch here can share, and it is shared only where it exists — see
[`_prefixsum_batch_fuses`](@ref):

- **Uniform axis**: every interval comes from `wcell` in `O(1)`, so there is nothing to amortize, and
  hoisting the band loop above the field loop would re-stream each output row once per band instead
  of keeping it resident across all of them. Those fields run the ordinary per-field row body.
- **Nonuniform axis**: the two-pointer walk dominates, so the band loop goes outermost and every field
  is fed from one walk. `lo`/`hi` stay in registers — no interval table is materialized.
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
    invden, hwj = fp.invden, fp.hw

    for m in eachindex(outs)
        o = @inbounds outs[m]
        @inbounds for i in 1:Nx
            o[i, j] = zero(T)
        end
    end

    nb = min(2 * fp.dj_lim + 1, Ny)
    @inbounds for b in 0:(nb - 1)
        hw = hwj[b + 1, j]
        hw < zero(T) && continue
        jj = mod1(j - fp.dj_lim + b, Ny)
        wyj = gp.wy[jj]

        if _prefixsum_spans_period(gp, ne, Nx, hw)
            # Whole period in support: one total per field, no walk.
            for m in eachindex(outs)
                o = outs[m]
                num_all = Ps[m][Nx + 1, jj] - Ps[m][1, jj]
                for i in 1:Nx
                    o[i, j] += wyj * num_all
                end
            end
        else
            lo = 1
            hi = 0
            for t in 1:Nx
                i = src[t]
                xc = x[i]
                xlo = xc - hw
                xhi = xc + hw
                while lo <= ne && xe[lo] < xlo
                    lo += 1
                end
                while hi < ne && xe[hi + 1] <= xhi
                    hi += 1
                end
                if hi >= lo
                    for m in eachindex(outs)
                        outs[m][i, j] += wyj * (Ps[m][hi + 1, jj] - Ps[m][lo, jj])
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
support interval once and feeds every field from it.
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

Whether a batched apply should share one support walk across the fields, or simply run them one at a
time.

On a **uniform** axis the support interval comes from `wcell` in `O(1)`, so there is no walk to
amortize and fusing is pure cost: the batch must hold `K` numerator scans live where the per-field
form works through one at a time, and the larger working set is what decides it.

On a **nonuniform** axis the two-pointer walk dominates the apply and is identical for every field, so
one walk feeding `K` accumulators is strictly less work.
"""
@inline _prefixsum_batch_fuses(fp::PrefixSumTopHatPlan) = !fp.uniform_axis

"""
    apply_footprint!(out, field, grid, fp::PrefixSumTopHatPlan, strategy, periodic_x, periodic_y) -> out

`apply_footprint!`-shaped entry point for the prefix-sum plan, so the generic whole-grid convolve name
works uniformly across every footprint type. `periodic_x`/`periodic_y` are accepted for interface
compatibility but IGNORED: unlike the offset-based footprints, this plan captured its periodicity (and
the matching extended-coordinate layout) from the grid at build time, and cannot honour a different
choice at apply time.
"""
apply_footprint!(
    out::AbstractMatrix{T}, field::AbstractMatrix, grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    fp::PrefixSumTopHatPlan{T}, strategy::AbstractMaskStrategy, periodic_x::Bool, periodic_y::Bool,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}} =
    apply_prefixsum_tophat!(out, field, grid, fp, strategy)

"""
    build_footprint(grid::StructuredGrid{T,<:AbstractGeometry,2}, kernel::TopHatKernel, scale; kwargs...) -> PrefixSumTopHatPlan

Exact prefix-sum top-hat path for any rectilinear 2D `StructuredGrid` (see the section comment above).
More specific than the generic 2D methods (constrained on kernel type), so Julia selects it whenever
`kernel isa TopHatKernel`.
"""
function build_footprint(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    kernel::Kernels.TopHatKernel,
    scale::T;
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    return _build_prefixsum_tophat(grid, kernel, scale; kwargs...)
end

# Range axes are a strict special case of "rectilinear", so they take the same exact prefix-sum path.
# This method exists to resolve what would otherwise be a genuine dispatch AMBIGUITY between the
# generic-kernel Range-axis method and the generic-axis `TopHatKernel` method above (neither is more
# specific than the other for the Range+TopHat combination) — it is strictly more specific than both.
function build_footprint(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2,S,TP,<:Tuple{AbstractRange,AbstractRange}},
    kernel::Kernels.TopHatKernel,
    scale::T;
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}, S, TP<:NTuple{2,FlowGeometries.Grids.AbstractTopology}}
    return _build_prefixsum_tophat(grid, kernel, scale; kwargs...)
end
