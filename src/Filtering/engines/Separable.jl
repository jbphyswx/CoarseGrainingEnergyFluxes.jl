# ---------------------------------------------------------------------------
# Separable Gaussian fast path. `exp(-α(Δx²+Δy²)/ℓ²)` factors as `Gx(Δx)·Gy(Δy)`, so a 2D convolution
# becomes a row pass then a column pass: O(N·r) against O(N·r²), up to the truncation-shape difference
# noted below. The factorization needs a rectilinear grid, not a uniform one — a stretched axis makes
# `Gx` depend on position as well as offset, which is a wider weight table (see
# `_separable_axis_weights`) rather than a different algorithm.
#
# Cartesian only: great-circle distance does not factor, so a spherical grid takes the per-latitude-band
# path instead.
# ---------------------------------------------------------------------------

"""
    SeparableScratch{T,MT} <: AbstractFilterScratch

The separable engine's two pass buffers: the masked input, and the intermediate the row pass writes
and the column pass reads.

Every table the engine holds — the per-axis tap weights, the denominator profiles, `invrenorm` — is a
function of the filter scale, so these buffers are the whole of what the scales of a sweep can share.
They are sized by the grid alone, hence one set per sweep rather than one per scale.

One set serves one apply at a time, so a driver running applies concurrently needs one per worker.
"""
struct SeparableScratch{T<:AbstractFloat, MT<:AbstractMatrix{T}} <: AbstractFilterScratch
    masked_input::MT
    row_pass::MT
end

function _separable_scratch(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    return SeparableScratch(zeros(T, Nx, Ny), zeros(T, Nx, Ny))
end

"""
    SeparableFootprint{T}

Precomputed 1D Gaussian weight vectors (`gx`,`gy`) plus preallocated scratch buffers for the
row-pass/column-pass separable convolution. `invrenorm` (Deformable masking only) is the precomputed
reciprocal local kernel mass over active cells — the SAME separable machinery run once, at plan-build
time, on `Float.(mask)` instead of `field`, mirroring `FFTWFilterPlan`/`SHTFilterPlan`'s established
`invrenorm` pattern. `Nx_profile`/`Ny_profile` (ZeroFill masking) are the mask-INDEPENDENT denominator
profiles: `Σ w` over valid (in-bounds/periodic) offsets is itself separable into one Nx-length and
one Ny-length vector, since which offsets are valid depends only on `i` (resp. `j`) and periodicity,
never on the mask.

Note: unlike the disk-truncated (`d <= rad`) non-separable footprint, this truncates each axis
independently at the SAME per-axis `rad` (the Gaussian's own 1D marginal decays at the identical rate
`kernel_radius` was derived from) — a square window, not a disk. The Gaussian has no true hard
support (only a numerical truncation tolerance), so this is an equally valid truncation, just a
different shape — matched against `RealSpace` within a measured tolerance, not asserted bit-identical.
"""
struct SeparableFootprint{
    T<:AbstractFloat,
    GX<:AbstractVecOrMat{T},
    # One type parameter per axis: a uniform axis gives a length-`2·lim+1` offset vector and a stretched
    # one a `(n, 2·lim+1)` position-major table, so a grid uniform in x and stretched in y carries one
    # of each.
    GY<:AbstractVecOrMat{T},
    PVT<:Union{Nothing,AbstractVector{T}},
    MT<:AbstractMatrix{T},
    IMT<:Union{Nothing,AbstractMatrix{T}},
}
    gx::GX
    gy::GY
    di_lim::Int
    dj_lim::Int
    periodic_x::Bool
    periodic_y::Bool
    Nx_profile::PVT
    Ny_profile::PVT
    invrenorm::IMT
    masked::Bool          # whether the grid had any inactive cell when this was built
    masked_input::MT
    row_pass::MT
end

# The per-row and per-column bodies, factored out so the ThreadedBackend extension parallelizes over
# `j` through these same functions. The row pass at row `j` reads only `src[:, j]`, so it parallelizes
# over `j`; the column pass reads `row_pass[:, jj]` across rows, so it must run as a separate pass
# after every row's `row_pass` is written, and cannot be fused with it.
"""
    _sepw(g, i, k) -> T

Weight of stencil slot `k` at axis position `i`.

The Gaussian factorizes on ANY rectilinear grid — `exp(-α(Δx²+Δy²)/ℓ²) = Gx(Δx)·Gy(Δy)` needs no
constant spacing — but on a uniform axis `Gx` depends on the OFFSET alone, while on a stretched one it
depends on the position too. Both are the same convolution with a different weight table, so the two
are one code path distinguished by the table's rank: a vector is shared across positions and a matrix
is `(2·lim+1) × N`, column-major so each position's stencil is contiguous.
"""
@inline _sepw(g::AbstractVector, ::Int, k::Int) = @inbounds g[k]
@inline _sepw(g::AbstractMatrix, i::Int, k::Int) = @inbounds g[i, k]

@inline function _separable_row_pass_at!(
    row_pass::AbstractMatrix{T}, src::AbstractMatrix{T}, gx::AbstractVecOrMat{T},
    di_lim::Int, periodic_x::Bool, Nx::Int, j::Int,
) where {T<:AbstractFloat}
    # Taps outermost, position innermost. Accumulating one output point at a time makes the inner loop
    # an FP reduction, which cannot be reassociated and so never vectorizes; this way the inner loop is
    # a unit-stride axpy over `i` and does. Measured 1.29 → 0.30 ns per weighted add at Nx=512, w=32.
    @inbounds begin
        for i in 1:Nx
            row_pass[i, j] = zero(T)
        end
        for ddi in (-di_lim):di_lim
            k = ddi + di_lim + 1
            if periodic_x && abs(ddi) < Nx
                # A wrapped tap is two contiguous runs, each at a CONSTANT offset. Writing it as one
                # loop over `mod1` costs the vectorization, since the index is then data-dependent.
                if ddi >= 0
                    @simd for i in 1:(Nx - ddi)
                        row_pass[i, j] += _sepw(gx, i, k) * src[i + ddi, j]
                    end
                    @simd for i in (Nx - ddi + 1):Nx
                        row_pass[i, j] += _sepw(gx, i, k) * src[i + ddi - Nx, j]
                    end
                else
                    @simd for i in 1:(-ddi)
                        row_pass[i, j] += _sepw(gx, i, k) * src[i + ddi + Nx, j]
                    end
                    @simd for i in (-ddi + 1):Nx
                        row_pass[i, j] += _sepw(gx, i, k) * src[i + ddi, j]
                    end
                end
            elseif periodic_x
                # A tap wider than the axis wraps more than once, so it needs the general index.
                @simd for i in 1:Nx
                    row_pass[i, j] += _sepw(gx, i, k) * src[mod1(i + ddi, Nx), j]
                end
            else
                @simd for i in max(1, 1 - ddi):min(Nx, Nx - ddi)
                    row_pass[i, j] += _sepw(gx, i, k) * src[i + ddi, j]
                end
            end
        end
    end
    return nothing
end

@inline function _separable_column_pass_at!(
    dst::AbstractMatrix{T}, row_pass::AbstractMatrix{T}, gy::AbstractVecOrMat{T},
    dj_lim::Int, periodic_y::Bool, Nx::Int, Ny::Int, j::Int,
) where {T<:AbstractFloat}
    # Same inversion as the row pass, and here the weight is constant across `i` — it is indexed by the
    # output column `j` — so it hoists out of the inner loop entirely.
    @inbounds begin
        for i in 1:Nx
            dst[i, j] = zero(T)
        end
        for ddj in (-dj_lim):dj_lim
            jj = j + ddj
            if jj < 1 || jj > Ny
                periodic_y || continue
                jj = mod1(jj, Ny)
            end
            wt = _sepw(gy, j, ddj + dj_lim + 1)
            @simd for i in 1:Nx
                dst[i, j] += wt * row_pass[i, jj]
            end
        end
    end
    return nothing
end

# Row-pass (over axis 1) then column-pass (over axis 2) of `src` into `dst`, using `fp`'s scratch
# `row_pass` buffer — the single shared primitive both the plan-build-time `invrenorm` computation
# and every serial `apply_separable!` call use, so they can never drift out of sync with
# each other. The ThreadedBackend extension calls `_separable_row_pass_at!`/`_separable_column_pass_at!`
# directly (parallelized over `j`) instead of this serial driver.
function _separable_convolve!(dst::AbstractMatrix{T}, src::AbstractMatrix{T}, fp::SeparableFootprint{T}, Nx::Int, Ny::Int) where {T<:AbstractFloat}
    gx, gy = fp.gx, fp.gy
    di_lim, dj_lim = fp.di_lim, fp.dj_lim
    periodic_x, periodic_y = fp.periodic_x, fp.periodic_y
    row_pass = fp.row_pass
    for j in 1:Ny
        _separable_row_pass_at!(row_pass, src, gx, di_lim, periodic_x, Nx, j)
    end
    for j in 1:Ny
        _separable_column_pass_at!(dst, row_pass, gy, dj_lim, periodic_y, Nx, Ny, j)
    end
    return dst
end

"""
    _separable_axis_weights(x, lim, periodic, period, kernel, scale, wfac) -> AbstractVecOrMat

A separable kernel's per-axis weight table: `Kernels.kernel_profile(kernel, Δx, ℓ)` over the stencil,
in the layout [`_sepw`](@ref) reads. Any kernel with a 1-D profile takes this path — the Gaussian,
whose radial form happens to factor, and [`Kernels.HighOrderKernel`](@ref), which is separable by
definition and has no radial form at all.

Uniform axis: the displacement is `ddi·Δ` wherever the stencil sits, so one vector serves every
position. Stretched axis: the displacement depends on the position too, so the table gains a position
axis — `(2·lim+1) × N`, which is `O(N·lim)` against the `O(N·lim²)` of a per-point neighbour cache, and
leaves the apply at `O(N·lim)` instead of `O(N·lim²)`.

`lim` comes from the SMALLEST gap on the axis, so on a stretched axis a coarse region's stencil is
wider than it needs to be; those slots hold exact zeros rather than being trimmed, which keeps the
inner loop's bounds static. A periodic displacement carries the image offset, matching the tiling
convention the scattered engine uses.
"""
# Cell-averaged weights keep a discontinuous kernel's MASS exact on any grid, but they cannot recover
# the vanishing MOMENTS if a limb is thinner than a cell — there is simply no resolution there to
# distinguish it from a box. That is a warning rather than an error: the filter is still a valid
# normalized low-pass, it just is not the high-order one that was asked for.
_warn_unresolved_limbs(::Kernels.AbstractFilterKernel, _, ::Int, _) = nothing

function _warn_unresolved_limbs(
    kernel::Kernels.HighOrderKernel, x::AbstractVector, d::Int, scale::Real,
)
    length(x) < 2 && return nothing
    Δ = minimum(abs(x[i] - x[i - 1]) for i in (firstindex(x) + 1):lastindex(x))
    b = kernel.b_over_ℓ * scale
    (isfinite(Δ) && Δ > 0 && b < Δ) && @warn(
        "$(nameof(typeof(kernel))) at ℓ = $scale has limb width b = $b, below axis $d's spacing " *
        "$Δ — under one cell per limb, so the vanishing moments it is built for do not survive " *
        "discretization and the result is effectively a box. At b_over_ℓ = $(kernel.b_over_ℓ) this " *
        "needs ℓ ≥ $(Δ / kernel.b_over_ℓ).",
        maxlog = 1,
    )
    return nothing
end

function _separable_axis_weights(
    x::AbstractRange{T}, lim::Int, ::Bool, ::T, kernel::Kernels.AbstractFilterKernel, scale::T,
    ::AbstractVector{T},
) where {T<:AbstractFloat}
    Δ = T(step(x))
    # `profile_cell_average` is the point sample for every smooth kernel and the exact cell integral
    # for a discontinuous one — see `Kernels.profile_cell_average`.
    return [Kernels.profile_cell_average(kernel, T(ddi) * Δ, Δ, scale) for ddi in -lim:lim]
end

function _separable_axis_weights(
    x::AbstractVector{T}, lim::Int, periodic::Bool, period::T,
    kernel::Kernels.AbstractFilterKernel, scale::T, wfac::AbstractVector{T},
) where {T<:AbstractFloat}
    n = length(x)
    # POSITION-major, `(n, 2·lim+1)`: the passes hold a tap fixed and sweep position, so position must
    # be the contiguous axis or every inner loop gathers with stride `2·lim+1`.
    # An untouched slot is an exact zero: outside the domain.
    g = zeros(T, n, 2 * lim + 1)
    @inbounds for i in 1:n, ddi in -lim:lim
        ii = i + ddi
        shift = zero(T)
        if ii < 1 || ii > n
            periodic || continue
            shift = T(fld(ii - 1, n)) * period
            ii = mod1(ii, n)
        end
        # The NEIGHBOUR's measure factor, so the two passes together weight by `kernel · cell area` —
        # the same quantity the scattered engine forms as `kernel_weight(d) * area(grid, ii, jj)`.
        # A rectilinear measure is itself a product of per-axis factors, so it splits over the passes
        # exactly as the kernel does. On a uniform axis it is a constant that cancels in the
        # normalization, which is why the vector method above can leave it out.
        # `wfac[ii]` is the neighbour cell's width, so it doubles as the averaging window: the weight
        # is the kernel's INTEGRAL over that cell, not its value at the node. Identical to the point
        # sample for a smooth kernel (`profile_cell_average`'s default), and the difference between
        # working and not for a discontinuous one on a stretched axis.
        g[i, ddi + lim + 1] =
            Kernels.profile_cell_average(kernel, x[ii] + shift - x[i], wfac[ii], scale) * wfac[ii]
    end
    return g
end

"""
    _build_separable_footprint(grid, kernel::GaussianKernel, scale; mask_strategy=ZeroFill(), kwargs...) -> SeparableFootprint

Build the per-axis weight tables, the mask-dependent normalization data for `mask_strategy`, and the
preallocated scratch buffers for the separable path. `kwargs...` absorbs (and ignores)
`cache_strategy`/`cache_byte_budget` — there is no per-point neighbour list to cache here at all, so
those knobs (which only govern the scattered path) don't apply.
"""
function _build_separable_footprint(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    kernel::SeparableKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    scratch::Union{Nothing,SeparableScratch} = nothing,
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    _warn_unresolved_limbs(kernel, FlowGeometries.Grids.coordinates(grid, 1), 1, scale)
    _warn_unresolved_limbs(kernel, FlowGeometries.Grids.coordinates(grid, 2), 2, scale)
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    rad = Kernels.kernel_radius(kernel, scale)
    # The smallest gap on each axis, so the stencil never under-covers: a coarser region's slots then
    # hold exact zeros rather than missing neighbours.
    dx = FlowGeometries.Grids.minimum_spacing(grid, 1)
    dy = FlowGeometries.Grids.minimum_spacing(grid, 2)
    di_lim = (isfinite(dx) && dx > 0) ? ceil(Int, rad / dx) : 0
    dj_lim = (isfinite(dy) && dy > 0) ? ceil(Int, rad / dy) : 0
    periodic_x = FlowGeometries.Grids.isperiodic(grid, 1)
    periodic_y = FlowGeometries.Grids.isperiodic(grid, 2)
    # The cell measure enters the weights, so the two passes together form `kernel · area` — the same
    # product the scattered engine builds per candidate. A rectilinear measure is separable by
    # construction, so it splits over the passes; anything else cannot take this path at all.
    mf = FlowGeometries.Grids.measure_factors(grid)
    mf === nothing && throw(ArgumentError(
        "the separable path needs a separable cell measure, but this grid's measure is dense",
    ))
    gx = _separable_axis_weights(
        FlowGeometries.Grids.coordinates(grid, 1), di_lim, periodic_x,
        T(FlowGeometries.Grids.period(grid, 1)), kernel, scale, convert(AbstractVector{T}, mf[1]),
    )
    gy = _separable_axis_weights(
        FlowGeometries.Grids.coordinates(grid, 2), dj_lim, periodic_y,
        T(FlowGeometries.Grids.period(grid, 2)), kernel, scale, convert(AbstractVector{T}, mf[2]),
    )

    # Both pass buffers are sized by the grid alone, so a sweep hands in one set for every scale
    # instead of each scale owning its own pair.
    sc = scratch === nothing ? _separable_scratch(grid) : scratch
    masked_input = sc.masked_input
    row_pass = sc.row_pass
    fully_active = all(FlowGeometries.Grids.mask(grid))
    fp_partial = SeparableFootprint(gx, gy, di_lim, dj_lim, periodic_x, periodic_y, nothing, nothing, nothing, !fully_active, masked_input, row_pass)

    if fully_active || mask_strategy isa ZeroFill
        # `ZeroFill`'s denominator is the mask-independent geometric profile. A fully-active grid takes
        # the same branch whatever its strategy: with nothing excluded, `Deformable` coincides with it.
        Nx_profile = _separable_profile(di_lim, gx, Nx, periodic_x)
        Ny_profile = _separable_profile(dj_lim, gy, Ny, periodic_y)
        return SeparableFootprint(gx, gy, di_lim, dj_lim, periodic_x, periodic_y, Nx_profile, Ny_profile, nothing, !fully_active, masked_input, row_pass)
    else
        # Deformable: precompute invrenorm = 1/separable_convolve(Float.(mask)) ONCE — the mask never
        # changes across repeated `filter_apply!` calls on a fixed plan.
        maskf = T.(FlowGeometries.Grids.mask(grid))
        denom = zeros(T, Nx, Ny)
        _separable_convolve!(denom, maskf, fp_partial, Nx, Ny)
        invrenorm = similar(denom)
        @. invrenorm = ifelse(denom > T(1e-15), one(T) / denom, zero(T))
        return SeparableFootprint(gx, gy, di_lim, dj_lim, periodic_x, periodic_y, nothing, nothing, invrenorm, !fully_active, masked_input, row_pass)
    end
end

# ZeroFill's mask-independent denominator profile: Σ w over geometrically-valid offsets at each
# index — separable since validity depends only on the index/periodicity, never on the mask.
function _separable_profile(lim::Int, g::AbstractVecOrMat{T}, N::Int, periodic::Bool) where {T<:AbstractFloat}
    profile = zeros(T, N)
    @inbounds for i in 1:N
        s = zero(T)
        for dd in -lim:lim
            ii = i + dd
            valid = (ii >= 1 && ii <= N) || periodic
            valid && (s += _sepw(g, i, dd + lim + 1))
        end
        profile[i] = s
    end
    return profile
end

"""
    apply_separable!(out, field, grid, fp::SeparableFootprint, strategy) -> out

Apply the separable fast path: `masked_input = mask .* field` (the SAME numerator input for
both mask strategies — see the struct docstring), one shared row-pass/column-pass convolution, then
divide by whichever mask-strategy-specific denominator `fp` holds.
"""
function apply_separable!(
    out::AbstractMatrix{T}, field::AbstractMatrix, grid::FlowGeometries.Grids.StructuredGrid, fp::SeparableFootprint{T}, strategy::AbstractMaskStrategy,
) where {T<:AbstractFloat}
    _separable_check_strategy(fp, strategy)
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    mask = FlowGeometries.Grids.mask(grid)
    @. fp.masked_input = T(mask) * field
    _separable_convolve!(out, fp.masked_input, fp, Nx, Ny)
    _separable_normalize_and_mask!(out, fp, mask, Nx, Ny)
    return out
end

@noinline function _separable_strategy_mismatch()
    throw(ArgumentError(
        "SeparableFootprint was built for ZeroFill masking, so it holds only the rank-1 " *
        "profiles and no renormalization field, but is being applied with a renormalizing " *
        "(non-ZeroFill) mask strategy on a masked grid. Rebuild the plan with the same " *
        "`mask_strategy` you intend to apply with.",
    ))
end

# The denominator is fixed at build time — rank-1 profiles for ZeroFill, a dense `invrenorm` for
# Deformable — so the apply cannot honour a strategy the plan was not built for.
#
# `invrenorm === nothing` does not by itself mean ZeroFill: on a fully-active grid both strategies
# coincide and take the profiles. Hence the stored `masked` flag, rather than an `all(mask)` scan.
@inline function _separable_check_strategy(fp::SeparableFootprint, strategy::AbstractMaskStrategy)
    if !(strategy isa ZeroFill) && fp.masked && fp.invrenorm === nothing
        _separable_strategy_mismatch()
    end
    return nothing
end

# Shared denominator-normalize + mask-zero epilogue, used identically by the serial path above and
# the ThreadedBackend extension's parallel-row-pass/column-pass path — kept in one place so they can
# never drift out of sync (mirrors `_separable_convolve!`'s own "single shared primitive" role).
function _separable_normalize_and_mask!(
    out::AbstractMatrix{T}, fp::SeparableFootprint{T}, mask::AbstractMatrix{Bool}, Nx::Int, Ny::Int,
) where {T<:AbstractFloat}
    if fp.invrenorm !== nothing
        @. out *= fp.invrenorm
    else
        Nx_profile, Ny_profile = fp.Nx_profile, fp.Ny_profile
        @inbounds for j in 1:Ny, i in 1:Nx
            denom = Nx_profile[i] * Ny_profile[j]
            out[i, j] = denom > T(1e-15) ? out[i, j] / denom : zero(T)
        end
    end
    @inbounds for j in 1:Ny, i in 1:Nx
        mask[i, j] || (out[i, j] = zero(T))
    end
    return out
end

"""
    build_footprint(grid::StructuredGrid{T,Cartesian,2,S,TP,<:Tuple{AbstractRange,AbstractRange}}, kernel::GaussianKernel, scale; kwargs...) -> SeparableFootprint

Fast path for a `GaussianKernel` on a uniform (`Range`-axis) Cartesian grid — see the "Separable
Gaussian fast path" section above. More specific than the generic Range-axis method (constrained on
kernel type too), so Julia picks this one whenever `kernel isa GaussianKernel`.
"""
function build_footprint(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2,S,TP,<:Tuple{AbstractRange,AbstractRange}},
    kernel::SeparableKernel,
    scale::T;
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}, S, TP<:NTuple{2,FlowGeometries.Grids.AbstractTopology}}
    return _build_separable_footprint(grid, kernel, scale; kwargs...)
end

"""
    build_footprint(grid::StructuredGrid{T,Cartesian,2}, kernel::GaussianKernel, scale; kwargs...) -> SeparableFootprint

Separability does not require constant spacing: `exp(-α(Δx²+Δy²)/ℓ²)` factorizes on any rectilinear
grid, and a stretched axis only makes the per-axis weight depend on position as well as offset — see
`_separable_axis_weights`. So a stretched Cartesian grid gets the same two-pass `O(N·(wx+wy))`
convolution rather than falling to the `O(N·wx·wy)` scattered engine, which for a Gaussian at `w = 20`
is a factor `(2w+1)/2` in operations and a much larger one in per-operation cost.

The Range-axis method above is strictly more specific and resolves what would otherwise be an
ambiguity with the generic Range-axis method; both build the same footprint.
"""
function build_footprint(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    kernel::SeparableKernel,
    scale::T;
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    return _build_separable_footprint(grid, kernel, scale; kwargs...)
end

"""
    apply_footprint!(out, field, grid, fp, strategy, periodic_x, periodic_y)

Convolve `field` with a precomputed `fp` into `out`, applying the mask `strategy`. `out` and
`field` are 2D (a single layer). The masking branch specializes on the strategy type.
"""
function apply_footprint!(
    out::AbstractMatrix{T},
    field::AbstractMatrix,
    grid::FlowGeometries.Grids.StructuredGrid,
    fp::FilterFootprint{T},
    strategy::AbstractMaskStrategy,
    periodic_x::Bool,
    periodic_y::Bool,
) where {T<:AbstractFloat}
    _, Ny = FlowGeometries.Grids.size_tuple(grid)
    _banded_check_strategy(fp, strategy)
    _banded_fill_source!(fp, field, grid)
    for j in 1:Ny
        apply_footprint_row!(out, field, grid, fp, strategy, periodic_x, periodic_y, j)
    end
    return out
end

"""
    apply_footprint_row!(out, field, grid, fp, strategy, periodic_x, periodic_y, j)

Fill output row `j` (`out[:, j]`) from a precomputed footprint, as one contiguous axpy per tap
normalized by the plan's `invden`. Rows are independent (each writes a disjoint column of the
column-major output), so this is the unit of parallelism for the threaded / distributed backends.

The whole row is written, including the columns within one filter radius of an axis-1 edge and every
column of a masked grid: a tap that would read out of bounds simply contributes over a shorter range,
and the matching shortfall is already in `invden`.
"""
# A plan's `invden` was accumulated under one mask strategy, so it cannot serve another. On an
# UNMASKED grid the two coincide exactly — `ZeroFill` divides by the total in-support mass and
# `Deformable` by the mass of the active taps, the same number when every cell is active — so the
# restriction bites only where the denominators genuinely differ.
@noinline function _banded_strategy_mismatch()
    throw(ArgumentError(
        "FilterFootprint is being applied to a MASKED grid with a different mask strategy than it was " *
        "built for. Its normalization is precomputed per scale from the grid, the mask and the " *
        "strategy, and the two strategies divide by different masses wherever a cell is inactive, so " *
        "one plan cannot serve both. Rebuild the plan with the `mask_strategy` you intend to apply with.",
    ))
end

@inline function _banded_check_strategy(fp::FilterFootprint, strategy::AbstractMaskStrategy)
    (!fp.masked || typeof(strategy) === typeof(fp.strategy)) || _banded_strategy_mismatch()
    return nothing
end

"""
    _banded_source(fp, field) -> src

The array the tap loop actually convolves: `mask · field` on a masked grid, and the caller's array
itself when there is nothing masked out, since the copy would then be pure cost.
"""
@inline _banded_source(fp::FilterFootprint, field::AbstractMatrix, slot::Int = 1) =
    fp.masked ? @inbounds(fp.scratch.masked_inputs[slot]) : field

function _banded_fill_source!(
    fp::FilterFootprint{T}, field::AbstractMatrix, grid::FlowGeometries.Grids.StructuredGrid,
    slot::Int = 1,
) where {T<:AbstractFloat}
    fp.masked || return nothing
    buf = @inbounds fp.scratch.masked_inputs[slot]
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    @inbounds for j in 1:Ny, i in 1:Nx
        buf[i, j] = FlowGeometries.Grids.isactive(grid, i, j) ? T(field[i, j]) : zero(T)
    end
    return nothing
end

"""
    prepare_row_apply!(fp, field, grid) -> nothing
    prepare_row_apply!(fp, fields, grid) -> nothing

Whole-grid work an engine needs done ONCE before any of its per-row applies run.

Every parallel backend decomposes over rows and calls `apply_footprint_row!` /
`apply_footprint_row_batch!` directly rather than going through the whole-grid entry point, so
anything that is not per-row has to be hoisted here or it is silently skipped on those paths. The
banded engine needs its `mask · field` source materialized; the other engines need nothing, and get
the no-op fallback.

Call it before opening a parallel region, never inside one: it writes the whole buffer.
"""
prepare_row_apply!(fp, field, grid) = nothing

prepare_row_apply!(
    fp::FilterFootprint, field::AbstractMatrix, grid::FlowGeometries.Grids.StructuredGrid,
) = _banded_fill_source!(fp, field, grid)

function prepare_row_apply_batch!(fp, fields, grid)
    return nothing
end

function prepare_row_apply_batch!(
    fp::FilterFootprint, fields, grid::FlowGeometries.Grids.StructuredGrid,
)
    fp.masked || return nothing
    _banded_inputs!(fp.scratch, length(fields))
    for m in eachindex(fields)
        _banded_fill_source!(fp, @inbounds(fields[m]), grid, m)
    end
    return nothing
end

function apply_footprint_row!(
    out::AbstractMatrix{T},
    field::AbstractMatrix,
    grid::FlowGeometries.Grids.StructuredGrid,
    fp::FilterFootprint{T},
    strategy::AbstractMaskStrategy,
    periodic_x::Bool,
    periodic_y::Bool,
    j::Integer,
) where {T<:AbstractFloat}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    b = _banded_band(fp.nbands, j)
    oc = view(out, :, j)
    @inbounds @simd for i in 1:Nx
        oc[i] = zero(T)
    end
    _banded_row_accumulate!(
        oc, _banded_source(fp, field), fp.di, fp.dj, fp.w, fp.ptr[b], fp.ptr[b + 1] - 1,
        fp.periodic_x, fp.periodic_y, Nx, Ny, j,
    )
    # `invden` carries the target-activity test and the degeneracy floor, so no branch is needed here.
    invden = fp.invden
    @inbounds @simd for i in 1:Nx
        oc[i] *= invden[i, j]
    end
    return out
end

"""
    apply_footprint!(out, field, grid, fp::ScatteredFilterPlan, strategy, periodic_x, periodic_y)

Whole-grid convolve using a [`ScatteredFilterPlan`](@ref) (the nonuniform-axis/curvilinear fallback).
`periodic_x`/`periodic_y` are accepted only for a uniform call signature with the `FilterFootprint`
method above — periodicity for this footprint kind lives in `fp` itself, not these arguments.
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
    for j in 1:Ny
        apply_footprint_row!(out, field, grid, fp, strategy, periodic_x, periodic_y, j)
    end
    return out
end

"""
    apply_footprint_row!(out, field, grid, fp::ScatteredFilterPlan, strategy, periodic_x, periodic_y, j)

Fill output row `j` from a [`ScatteredFilterPlan`](@ref): if `fp.cache !== nothing`, read the
precomputed per-point neighbour list (absolute `(ii,jj)` indices, periodic wrap already resolved at
build time); otherwise recompute each point's neighbours/weights on the fly from `fp`'s compact
scalar metadata. Both branches enumerate candidates through `_scattered_foldl`, so they are
bit-identical by construction rather than by convention. The accumulator is threaded through the
fold's return value rather than captured and mutated, which is what keeps the streaming branch free
of per-iteration allocation (verified by `@allocated` tests) without a second copy of the loop.
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
) where {T<:AbstractFloat}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    cache = fp.cache
    if cache !== nothing
        for i in 1:Nx
            FlowGeometries.Grids.isactive(grid, i, j) || continue
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
                if strategy isa ZeroFill
                    weight_norm += w
                    active && (weighted_sum += w * field[ii, jj])
                else
                    active || continue
                    weight_norm += w
                    weighted_sum += w * field[ii, jj]
                end
            end
            out[i, j] = weight_norm > T(1e-15) ? weighted_sum / weight_norm : zero(T)
        end
    else
        kernel = fp.kernel
        scale = fp.scale
        di_lim, dj_lim = fp.di_lim, fp.dj_lim
        fp_periodic_x, fp_periodic_y = fp.periodic_x, fp.periodic_y
        x_period, y_period = fp.x_period, fp.y_period
        is_cartesian = fp.is_cartesian
        rad = fp.rad
        # One candidate buffer for the row, not one per point: an indexed query fills it, and each row
        # is its own task under a row-parallel backend, so nothing is shared across tasks.
        sc = FlowGeometries.Connectivity.ball_scratch()
        for i in 1:Nx
            FlowGeometries.Grids.isactive(grid, i, j) || continue
            target = FlowGeometries.Grids.coords(SA.SVector, grid, i, j)
            weighted_sum, weight_norm = _scattered_foldl(
                (zero(T), zero(T)), grid, target, i, j, Nx, Ny, di_lim, dj_lim,
                fp_periodic_x, fp_periodic_y, x_period, y_period, is_cartesian, rad, fp.topology, sc,
            ) do acc, iin, jjn, d
                ws, wn = acc
                active = FlowGeometries.Grids.isactive(grid, iin, jjn)
                w = Kernels.kernel_weight(kernel, d, scale) * FlowGeometries.Grids.area(grid, iin, jjn)
                if strategy isa ZeroFill
                    return (active ? ws + w * field[iin, jjn] : ws, wn + w)
                else
                    active || return acc
                    return (ws + w * field[iin, jjn], wn + w)
                end
            end
            out[i, j] = weight_norm > T(1e-15) ? weighted_sum / weight_norm : zero(T)
        end
    end
    return out
end
