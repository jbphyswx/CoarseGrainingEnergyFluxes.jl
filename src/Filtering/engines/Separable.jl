# ---------------------------------------------------------------------------
# Separable-kernel engine. A `SeparableKernel` factors as `Gx(Δx)·Gy(Δy)` (the Gaussian
# `exp(-α(Δx²+Δy²)/ℓ²)` among them), so the 2D convolution is a row pass then a column pass, O(N·r).
# On a stretched axis `Gx` depends on position as well as offset; see `_separable_axis_weights`.
# Cartesian grids only: great-circle distance does not factor.
# ---------------------------------------------------------------------------

"""
    SeparableScratch{T,MT} <: AbstractFilterScratch

The separable engine's two pass buffers: the masked input, and the intermediate the row pass writes
and the column pass reads.

Sized by the grid alone, so one set serves every scale of a sweep. One set serves one apply at a time,
so a driver running applies concurrently needs one per worker.
"""
struct SeparableScratch{T<:AbstractFloat, MT<:AbstractArray{T}} <: AbstractFilterScratch
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

Per-axis kernel weight tables (`gx`, `gy`, read through `_sepw`), the denominator, and the scratch
buffers of the two-pass separable convolution.

The denominator is `invrenorm` for `Deformable` on a masked grid: the reciprocal kernel mass over
active cells, the same two-pass convolution applied to the mask. Otherwise it is the outer product of
`Nx_profile` and `Ny_profile`: `ZeroFill` counts the offsets in the domain and past a bounded edge,
`Deformable` on a fully active grid only those in the domain. Whether an offset lies in the domain
depends on `i` and `j` separately, so the profiles are rank-1.

Each axis is truncated at `rad` on its own, so the window is a square. For the Gaussian it differs from
the disk-truncated engines by weights below the truncation tolerance of `kernel_radius`.
"""
struct SeparableFootprint{
    T<:AbstractFloat,
    GX<:AbstractVecOrMat{T},
    # A uniform axis gives a length-`2·lim+1` vector, a stretched one an `(n, 2·lim+1)` table.
    GY<:AbstractVecOrMat{T},
    PVT<:Union{Nothing,AbstractVector{T}},
    MT<:AbstractMatrix{T},
    IMT<:Union{Nothing,AbstractMatrix{T}},
    MS<:AbstractMaskStrategy,
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
    strategy::MS          # the strategy the denominator was built for
    masked::Bool          # whether the grid had any inactive cell when this was built
    bound::Bool           # the strategies' denominators differ: a cell is inactive or a window leaves the grid
    masked_input::MT
    row_pass::MT
end

"""
    _sepw(g, i, k) -> T

Weight of stencil slot `k` at axis position `i`: `g[k]` for a uniform axis, whose weight depends on
the offset alone, and `g[i, k]` for a stretched axis's `n × (2·lim+1)` table.
"""
@inline _sepw(g::AbstractVector, ::Int, k::Int) = @inbounds g[k]
@inline _sepw(g::AbstractMatrix, i::Int, k::Int) = @inbounds g[i, k]

# Row `j` of the row pass reads only `src[:, j]`. The column pass reads `row_pass` across rows, so it
# starts after every row is written. The parallel backend extensions call both per `j`.
@inline function _separable_row_pass_at!(
    row_pass::AbstractMatrix{T}, src::AbstractMatrix{T}, gx::AbstractVecOrMat{T},
    di_lim::Int, periodic_x::Bool, Nx::Int, j::Int,
) where {T<:AbstractFloat}
    # Taps outermost, so each inner loop is a unit-stride axpy over `i` and vectorizes.
    @inbounds begin
        for i in 1:Nx
            row_pass[i, j] = zero(T)
        end
        for ddi in (-di_lim):di_lim
            k = ddi + di_lim + 1
            if periodic_x && abs(ddi) < Nx
                # A wrapped tap is two contiguous runs, each at a constant offset.
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

# Row pass over axis 1, then column pass over axis 2, of `src` into `dst` through `fp.row_pass`, which
# neither may alias. The build of `invrenorm` and the serial apply both run it.
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
    _separable_axis_weights(spacing, x, lim, periodic, period, kernel, scale, wfac) -> AbstractVecOrMat

A separable kernel's per-axis weight table, `Kernels.profile_cell_average` over the stencil, in the
layout [`_sepw`](@ref) reads, for a direction of spacing trait `spacing` (`Grids.spacing_trait`, a
periodic direction's wrap included). Serves the Gaussian and [`Kernels.HighOrderKernel`](@ref).

Uniform direction: the displacement is `ddi·Δ` at every position, so one vector serves them all.
Otherwise an `n × (2·lim+1)` table, one stencil per position, each weight times the neighbour's width.

`lim` comes from the smallest gap on the axis, so on a stretched axis a coarse region's stencil has
slots past `rad`, holding the kernel's negligible or zero tail. A periodic displacement includes the
image shift, so each image of a cell counts at its own distance.
"""
function _separable_axis_weights(
    ::FlowGeometries.Axes.UniformSpacing, x::AbstractVector{T}, lim::Int, ::Bool, ::T,
    kernel::Kernels.AbstractFilterKernel, scale::T, ::AbstractVector{T},
) where {T<:AbstractFloat}
    Δ = abs(T(step(x)))
    # The constant cell width cancels in the normalization, so it is left out of the weight.
    return [Kernels.profile_cell_average(kernel, T(ddi) * Δ, Δ, scale) for ddi in -lim:lim]
end

function _separable_axis_weights(
    ::FlowGeometries.Axes.NonuniformSpacing, x::AbstractVector{T}, lim::Int, periodic::Bool, period::T,
    kernel::Kernels.AbstractFilterKernel, scale::T, wfac::AbstractVector{T},
) where {T<:AbstractFloat}
    n = length(x)
    # Position-major, since the passes sweep position at a fixed tap. A slot left zero lies past a
    # bounded edge.
    g = zeros(T, n, 2 * lim + 1)
    s = (n > 1 && x[n] < x[1]) ? -one(T) : one(T)   # coordinate direction of increasing index
    @inbounds for i in 1:n, ddi in -lim:lim
        ii = i + ddi
        shift = zero(T)
        if ii < 1 || ii > n
            periodic || continue
            shift = s * T(fld(ii - 1, n)) * period
            ii = mod1(ii, n)
        end
        # The kernel averaged over neighbour cell `ii`, times its width `wfac[ii]`: the two passes
        # together weight by `kernel · cell area`, a rectilinear measure being a product of widths.
        g[i, ddi + lim + 1] =
            Kernels.profile_cell_average(kernel, x[ii] + shift - x[i], wfac[ii], scale) * wfac[ii]
    end
    return g
end

"""
    _separable_tables(sp, grid, kernel::SeparableKernel, scale) -> (; g, lim, periodic, mf)

The separable engines' per-axis weight tables `g` ([`_separable_axis_weights`](@ref)) by the spacing
tuple `sp` (`_with_spacings`) and stencil half-widths `lim`, from the smallest gap on each axis so
the stencil covers the kernel's radius everywhere, with each axis's periodicity and the grid's measure
factors `mf`. Axis `d` takes the profile of `Kernels._axis_kernel(kernel, Val(d))`.
"""
function _separable_tables(
    sp::NTuple{N,Any}, grid::FlowGeometries.Grids.StructuredGrid{T,G,N},
    kernel::SeparableKernel, scale::T,
) where {T<:AbstractFloat, N, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    lim = _separable_halfwidths(grid, kernel, scale)
    periodic = FlowGeometries.Grids.periodic_flags(grid)
    mf = FlowGeometries.Grids.measure_factors(grid)
    mf === nothing && throw(ArgumentError(
        "the separable path needs a separable cell measure, but this grid's measure is dense",
    ))
    periods = ntuple(d -> T(FlowGeometries.Grids.period(grid, d)), Val(N))
    axes_d = ntuple(Val, Val(N))
    g = map(sp, FlowGeometries.Grids.coordinates(grid), lim, periodic, periods, mf, axes_d) do s, x, l, p, P, w, d
        _separable_axis_weights(s, x, l, p, P, Kernels._axis_kernel(kernel, d), scale, convert(AbstractVector{T}, w))
    end
    return (; g, lim, periodic, mf)
end

# The stencil half-widths in cells along each axis, from its smallest gap.
function _separable_halfwidths(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, kernel::SeparableKernel, scale::T,
) where {T<:AbstractFloat, N, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    rad = Kernels.kernel_radius(kernel, scale)
    return ntuple(Val(N)) do d
        s = FlowGeometries.Grids.minimum_spacing(grid, d)
        (isfinite(s) && s > 0) ? ceil(Int, rad / s) : 0
    end
end

"""
    _build_separable_footprint(sp, grid, kernel::SeparableKernel, scale; mask_strategy=ZeroFill(), scratch=nothing, kwargs...) -> SeparableFootprint

Build the per-axis weight tables for the spacing tuple `sp`, the denominator for `mask_strategy`, and the
pass buffers of the separable engine. A given `scratch`, a `SeparableScratch`, supplies the buffers.
`kwargs...` absorbs and ignores `cache_strategy`.
"""
function _build_separable_footprint(
    sp::Tuple,
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    kernel::SeparableKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    scratch::Union{Nothing,SeparableScratch} = nothing,
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    (; g, lim, periodic, mf) = _separable_tables(sp, grid, kernel, scale)
    gx, gy = g
    di_lim, dj_lim = lim
    periodic_x, periodic_y = periodic

    sc = scratch === nothing ? _separable_scratch(grid) : scratch
    masked_input = sc.masked_input
    row_pass = sc.row_pass
    fully_active = all(FlowGeometries.Grids.mask(grid))
    zerofill = mask_strategy isa ZeroFill
    bound = !fully_active || _reaches_exterior(grid, kernel, scale)
    fp_partial = SeparableFootprint(
        gx, gy, di_lim, dj_lim, periodic_x, periodic_y, nothing, nothing, nothing,
        mask_strategy, !fully_active, bound, masked_input, row_pass,
    )

    if _mask_free_mass(mask_strategy, FlowGeometries.Grids.mask(grid))
        # Rank-1: `ZeroFill` counts the offsets past a bounded edge, `Deformable` on an `AllActive` grid
        # only the in-domain ones.
        Nx_profile = _separable_profile(FlowGeometries.Grids.coordinates(grid, 1), di_lim, gx, periodic_x,
                                        Kernels._axis_kernel(kernel, Val(1)), scale, convert(AbstractVector{T}, mf[1]),
                                        zerofill)
        Ny_profile = _separable_profile(FlowGeometries.Grids.coordinates(grid, 2), dj_lim, gy, periodic_y,
                                        Kernels._axis_kernel(kernel, Val(2)), scale, convert(AbstractVector{T}, mf[2]),
                                        zerofill)
        return SeparableFootprint(
            gx, gy, di_lim, dj_lim, periodic_x, periodic_y, Nx_profile, Ny_profile, nothing,
            mask_strategy, !fully_active, bound, masked_input, row_pass,
        )
    else
        # Deformable on a masked grid: `invrenorm = 1 / (two-pass convolution of the mask)`.
        maskf = T.(FlowGeometries.Grids.mask(grid))
        denom = zeros(T, Nx, Ny)
        _separable_convolve!(denom, maskf, fp_partial, Nx, Ny)
        invrenorm = similar(denom)
        @. invrenorm = _inv_mass(denom)
        return SeparableFootprint(
            gx, gy, di_lim, dj_lim, periodic_x, periodic_y, nothing, nothing, invrenorm,
            mask_strategy, !fully_active, bound, masked_input, row_pass,
        )
    end
end

"""
    _separable_profile(x, lim, g, periodic, kernel, scale, wfac, exterior) -> profile

The denominator profile along one axis: `Σ w` over the stencil's in-domain offsets at each index and,
with `exterior`, over the offsets past a bounded edge too, on the lattice continued at its edge gap.
`wfac` is the axis's cell widths, the measure factor `_separable_axis_weights` was given.
"""
function _separable_profile(
    x::AbstractVector, lim::Int, g::AbstractVecOrMat{T}, periodic::Bool,
    kernel::Kernels.AbstractFilterKernel, scale::T, ::AbstractVector{T}, exterior::Bool,
) where {T<:AbstractFloat}
    N = length(x)
    profile = zeros(T, N)
    @inbounds for i in 1:N
        s = zero(T)
        for dd in -lim:lim
            ii = i + dd
            if (ii >= 1 && ii <= N) || periodic
                s += _sepw(g, i, dd + lim + 1)
            elseif exterior
                s += _separable_exterior_weight(g, x, i, ii, dd + lim + 1, kernel, scale)
            end
        end
        profile[i] = s
    end
    return profile
end

# A uniform axis's weight depends on the offset alone. A stretched axis continues at its edge gap, and
# the cell there carries the weight `_separable_axis_weights` gives a cell of that width.
@inline _separable_exterior_weight(g::AbstractVector, _, _, _, k::Int, _, _) = @inbounds g[k]

@inline function _separable_exterior_weight(
    ::AbstractMatrix{T}, x::AbstractVector, i::Int, ii::Int, ::Int,
    kernel::Kernels.AbstractFilterKernel, scale::T,
) where {T<:AbstractFloat}
    n = length(x)
    s = ii < 1 ? T(x[2] - x[1]) : T(x[n] - x[n - 1])
    xe = ii < 1 ? T(x[1]) + (ii - 1) * s : T(x[n]) + (ii - n) * s
    return Kernels.profile_cell_average(kernel, xe - T(x[i]), abs(s), scale) * abs(s)
end

"""
    apply_separable!(out, field, grid, fp::SeparableFootprint, strategy) -> out

Filter `field` into `out`: the two-pass convolution of `mask .* field`, normalized by the denominator
`fp` holds. Where the strategies normalize differently (`fp.bound`), `strategy` must be the one `fp`
was built for, or this throws an `ArgumentError`.
"""
function apply_separable!(
    out::AbstractMatrix{T}, field::AbstractMatrix, grid::FlowGeometries.Grids.StructuredGrid, fp::SeparableFootprint{T}, strategy::AbstractMaskStrategy,
) where {T<:AbstractFloat}
    _separable_check_strategy(fp, strategy)
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    mask = FlowGeometries.Grids.mask(grid)
    # A `Bool` is a strong zero: an inactive cell contributes nothing, whatever it holds.
    @. fp.masked_input = mask * field
    _separable_convolve!(out, fp.masked_input, fp, Nx, Ny)
    _separable_normalize_and_mask!(out, fp, mask, Nx, Ny)
    return out
end

@noinline function _separable_strategy_mismatch()
    throw(ArgumentError(
        "A separable footprint is being applied with a different mask strategy than it was built for, " *
        "on a grid where the two normalize differently: a cell is inactive, or a window reaches past a " *
        "bounded edge. Its denominator is fixed at build time, so one plan cannot serve both. Rebuild " *
        "the plan with the `mask_strategy` you intend to apply with.",
    ))
end

# The denominator is fixed at build time, so where the strategies differ (`fp.bound`) only the one `fp`
# was built for may apply.
@inline function _separable_check_strategy(fp::SeparableFootprint, strategy::AbstractMaskStrategy)
    (!fp.bound || typeof(strategy) === typeof(fp.strategy)) || _separable_strategy_mismatch()
    return nothing
end

# `Deformable` on a masked grid zeroes an inactive target; `ZeroFill` keeps its value, the filter of the
# zero-extended field there.
function _separable_normalize_and_mask!(
    out::AbstractMatrix{T}, fp::SeparableFootprint{T}, mask::AbstractMatrix{Bool}, Nx::Int, Ny::Int,
) where {T<:AbstractFloat}
    for j in 1:Ny
        _separable_normalize_row!(out, fp, mask, Nx, j)
    end
    return out
end

# Row `j` of the normalization; it reads and writes that row alone.
@inline function _separable_normalize_row!(
    out::AbstractMatrix{T}, fp::SeparableFootprint{T}, mask::AbstractMatrix{Bool}, Nx::Int, j::Int,
) where {T<:AbstractFloat}
    inv = fp.invrenorm
    @inbounds if inv !== nothing
        for i in 1:Nx
            out[i, j] *= inv[i, j]
            mask[i, j] || (out[i, j] = zero(T))
        end
    else
        Nx_profile, Ny_profile = fp.Nx_profile, fp.Ny_profile
        for i in 1:Nx
            out[i, j] = _normalized(out[i, j], Nx_profile[i] * Ny_profile[j])
        end
    end
    return nothing
end

# A uniform axis 1 is also what the banded engine takes for any kernel, so a separable kernel there
# matches both; this method is more specific than either and keeps it on the two-pass engine.
_structured_footprint(
    sp::Tuple{FlowGeometries.Axes.UniformSpacing,Any}, grid::FlowGeometries.Grids.StructuredGrid{T,G,2}, kernel::SeparableKernel, scale::T;
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}} =
    _build_separable_footprint(sp, grid, kernel, scale; kwargs...)

"""
    _structured_footprint(::Tuple, grid::StructuredGrid{T,<:CartesianGeometry,2}, kernel::SeparableKernel, scale; kwargs...) -> SeparableFootprint

The separable two-pass engine, `O(N·(wx+wy))`, on any rectilinear Cartesian grid: a stretched axis
gets a per-position weight table (see `_separable_axis_weights`). Keywords as for
`_build_separable_footprint`.
"""
_structured_footprint(
    sp::Tuple, grid::FlowGeometries.Grids.StructuredGrid{T,G,2}, kernel::SeparableKernel, scale::T; kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}} =
    _build_separable_footprint(sp, grid, kernel, scale; kwargs...)

