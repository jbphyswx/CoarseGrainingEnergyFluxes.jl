# ---------------------------------------------------------------------------
# General N-dimensional engine (1D + 3D Cartesian); the 2D path uses the per-row engine above.
# ---------------------------------------------------------------------------

# One shared offset set for the whole grid, so it needs both uniform spacing (`AbstractRange` axes) and
# a position-independent metric. A spherical metric is position-dependent — arc length varies with
# latitude, and in 3D with radius — so a spherical 1D/3D grid takes the general path below.
"""
    SeparableFootprintND{N,T}

A separable kernel in `N` dimensions: one weight table per axis, applied as `N` successive 1-D
passes.

This is the same factorization the 2-D path uses, and the reason it matters grows with `N`. A
`FilterFootprintND` enumerates the whole `∏(2wᵈ+1)` box per point; `N` passes cost `∑(2wᵈ+1)`. At
`w = 20` in 3-D that is 68,921 multiply-adds per point against 123.

Weight tables follow [`_sepw`](@ref): a vector where the axis is uniform, an `Nᵈ × (2wᵈ+1)` matrix where
it is stretched.
"""
struct SeparableFootprintND{
    N, T<:AbstractFloat,
    GT<:NTuple{N,AbstractVecOrMat{T}},
    PVT<:Union{Nothing,NTuple{N,AbstractVector{T}}},
    AT<:AbstractArray{T,N},
    IAT<:Union{Nothing,AbstractArray{T,N}},
    MS<:AbstractMaskStrategy,
}
    g::GT
    lim::NTuple{N,Int}
    periodic::NTuple{N,Bool}
    profiles::PVT      # rank-1 denominator, one factor per axis
    invrenorm::IAT     # dense Deformable denominator on a masked grid, or nothing
    strategy::MS       # the strategy the denominator was built for
    masked::Bool
    bound::Bool        # the strategies' denominators differ: a cell is inactive or a window leaves the grid
    masked_input::AT
    scratch::AT
end

"""
    _sep_serial(f, indices)

The default pass driver: apply `f` to every index in order. Every output point of a pass is
independent, so a backend can substitute a parallel driver of the same shape and get a bit-identical
result — only the barrier BETWEEN passes is required.
"""
@inline function _sep_serial(f::F, indices) where {F}
    for I in indices
        f(I)
    end
    return nothing
end

@inline function _separable_pass!(
    dst::AbstractArray{T,N}, src::AbstractArray{T,N}, g::AbstractVecOrMat{T},
    lim::Int, periodic::Bool, dims::NTuple{N,Int}, ::Val{d}, driver::D,
) where {T<:AbstractFloat, N, d, D}
    n = dims[d]
    # Driven over COLUMNS, not points: taps move to the outer loop so the inner loop always walks
    # dimension 1 with unit stride, whichever axis the pass is along. A per-point form cannot do this —
    # for `d ≥ 2` it strides by `∏dims[1:d-1]` on every tap. Each column is written by exactly one task,
    # so this is as race-free as a per-point form and every backend gets the same answer.
    #
    # Rank 1 needs no separate branch: `Base.tail((n,))` is `()` and `CartesianIndices(())` holds one
    # 0-dimensional index, so the column pass runs once over the whole vector.
    driver(CartesianIndices(Base.tail(dims))) do J
        if d == 1
            _separable_col_pass!(dst, src, g, lim, periodic, dims[1], Tuple(J))
        else
            _separable_slab_pass!(dst, src, g, lim, periodic, n, dims[1], Tuple(J), Val(d))
        end
    end
    return dst
end

# One line along dimension `d > 1`. The weight is indexed by the OUTPUT position along `d`, which is
# fixed for this line, so it hoists out of the inner loop entirely and each tap is a plain axpy.
@inline function _separable_slab_pass!(
    dst::AbstractArray{T}, src::AbstractArray{T}, g::AbstractVecOrMat{T},
    lim::Int, periodic::Bool, n::Int, n1::Int, J::Tuple, ::Val{d},
) where {T<:AbstractFloat, d}
    jd = J[d - 1]
    dv = view(dst, :, J...)
    @inbounds begin
        for i in 1:n1
            dv[i] = zero(T)
        end
        for dd in (-lim):lim
            jj = jd + dd
            if jj < 1 || jj > n
                periodic || continue
                jj = mod1(jj, n)
            end
            wt = _sepw(g, jd, dd + lim + 1)
            sv = view(src, :, Base.setindex(J, jj, d - 1)...)
            @simd for i in 1:n1
                dv[i] += wt * sv[i]
            end
        end
    end
    return nothing
end

# One line along dimension 1, tap-outer. `dv`/`sv` are contiguous views, so each tap is an axpy.
@inline function _separable_col_pass!(
    dst::AbstractArray{T}, src::AbstractArray{T}, g::AbstractVecOrMat{T},
    lim::Int, periodic::Bool, n1::Int, J::Tuple,
) where {T<:AbstractFloat}
    dv = view(dst, :, J...)
    sv = view(src, :, J...)
    @inbounds begin
        for i in 1:n1
            dv[i] = zero(T)
        end
        for dd in (-lim):lim
            k = dd + lim + 1
            if periodic && abs(dd) < n1
                if dd >= 0
                    @simd for i in 1:(n1 - dd)
                        dv[i] += _sepw(g, i, k) * sv[i + dd]
                    end
                    @simd for i in (n1 - dd + 1):n1
                        dv[i] += _sepw(g, i, k) * sv[i + dd - n1]
                    end
                else
                    @simd for i in 1:(-dd)
                        dv[i] += _sepw(g, i, k) * sv[i + dd + n1]
                    end
                    @simd for i in (-dd + 1):n1
                        dv[i] += _sepw(g, i, k) * sv[i + dd]
                    end
                end
            elseif periodic
                @simd for i in 1:n1
                    dv[i] += _sepw(g, i, k) * sv[mod1(i + dd, n1)]
                end
            else
                @simd for i in max(1, 1 - dd):min(n1, n1 - dd)
                    dv[i] += _sepw(g, i, k) * sv[i + dd]
                end
            end
        end
    end
    return nothing
end

# The passes, unrolled by recursion on `Val(d)`. Buffers alternate and the last pass writes `dst`, so
# nothing aliases: pass 1 reads the caller's masked input and writes `scratch`, and every later pass
# reads what its predecessor just wrote. `masked_input` is free to be reused from pass 2 on.
# The two alternating intermediates are passed IN rather than taken from the footprint, because a batched
# apply needs batch-sized ones and the footprint's are sized for a single slice. `dims` is the driven
# array's shape, so trailing batch axes ride along; `Val(N)` is the number of PASSES, which stays at the
# grid's rank so no pass ever differences along a batch axis.
@inline _sep_pass_chain!(dst, src, fp, dims, ::Val{N}, ::Val{N}, driver::D, b1, b2) where {N,D} =
    _separable_pass!(dst, src, fp.g[N], fp.lim[N], fp.periodic[N], dims, Val(N), driver)

@inline function _sep_pass_chain!(dst, src, fp, dims, ::Val{N}, ::Val{d}, driver::D, b1, b2) where {N,d,D}
    buf = isodd(d) ? b1 : b2
    _separable_pass!(buf, src, fp.g[d], fp.lim[d], fp.periodic[d], dims, Val(d), driver)
    return _sep_pass_chain!(dst, buf, fp, dims, Val(N), Val(d + 1), driver, b1, b2)
end

# Unbatched callers keep the footprint's own buffers.
@inline _sep_pass_chain!(dst, src, fp, dims, vn::Val, vd::Val, driver::D = _sep_serial) where {D} =
    _sep_pass_chain!(dst, src, fp, dims, vn, vd, driver, fp.scratch, fp.masked_input)

function _build_separable_nd(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N},
    kernel::SeparableKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    kwargs...,
) where {T<:AbstractFloat, N, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    dims = FlowGeometries.Grids.size_tuple(grid)
    rad = Kernels.kernel_radius(kernel, scale)
    periodic = FlowGeometries.Grids.periodic_flags(grid)
    lim = ntuple(Val(N)) do d
        s = FlowGeometries.Grids.minimum_spacing(grid, d)
        (isfinite(s) && s > 0) ? ceil(Int, rad / s) : 0
    end
    mf = FlowGeometries.Grids.measure_factors(grid)
    mf === nothing && throw(ArgumentError(
        "the separable path needs a separable cell measure, but this grid's measure is dense",
    ))
    g = ntuple(Val(N)) do d
        _separable_axis_weights(
            FlowGeometries.Grids.coordinates(grid, d), lim[d], periodic[d],
            T(FlowGeometries.Grids.period(grid, d)), kernel, scale, convert(AbstractVector{T}, mf[d]),
        )
    end

    masked_input = zeros(T, dims)
    scratch = zeros(T, dims)
    fully_active = all(FlowGeometries.Grids.mask(grid))
    zerofill = mask_strategy isa ZeroFill
    bound = !fully_active || _reaches_exterior(grid, kernel, scale)
    fp_partial = SeparableFootprintND(
        g, lim, periodic, nothing, nothing, mask_strategy, !fully_active, bound, masked_input, scratch,
    )
    if zerofill || fully_active
        # Mask-independent denominator, a product of one factor per axis: `ZeroFill` counts the offsets
        # past a bounded edge, `Deformable` on a fully active grid only the in-domain ones.
        profiles = ntuple(Val(N)) do d
            _separable_profile(
                FlowGeometries.Grids.coordinates(grid, d), lim[d], g[d], periodic[d], kernel, scale, zerofill,
            )
        end
        return SeparableFootprintND(
            g, lim, periodic, profiles, nothing, mask_strategy, !fully_active, bound, masked_input, scratch,
        )
    end
    maskf = T.(FlowGeometries.Grids.mask(grid))
    denom = zeros(T, dims)
    copyto!(masked_input, maskf)
    _sep_pass_chain!(denom, masked_input, fp_partial, dims, Val(N), Val(1))
    invrenorm = similar(denom)
    @. invrenorm = _inv_mass(denom)
    return SeparableFootprintND(
        g, lim, periodic, nothing, invrenorm, mask_strategy, !fully_active, bound, masked_input, scratch,
    )
end

@inline function _separable_check_strategy(
    fp::SeparableFootprintND, strategy::AbstractMaskStrategy,
)
    (!fp.bound || typeof(strategy) === typeof(fp.strategy)) || _separable_strategy_mismatch()
    return nothing
end

"""
    apply_separable_nd!(out, field, grid, fp, strategy, driver = _sep_serial)

Run the `N` separable passes and the pointwise normalization. `driver` supplies the per-pass index
sweep — see [`_sep_serial`](@ref); a threaded backend passes its own and gets the same answer, since
every point within a pass is independent and the passes themselves stay ordered.

`out` and `field` may carry trailing batch axes beyond the grid's rank `R`: each pass is driven over the
array's shape, so a whole batch is one pass (one launch on a device), and the pass count stays `R`. The
profile tables, the renormalization array and the mask are spatial, indexed by the leading `R`
components of the driven index.
"""
function apply_separable_nd!(
    out::AbstractArray{T}, field::AbstractArray, grid::FlowGeometries.Grids.StructuredGrid,
    fp::SeparableFootprintND{R,T}, strategy::AbstractMaskStrategy, driver::D = _sep_serial,
) where {T<:AbstractFloat, R, D}
    _separable_check_strategy(fp, strategy)
    dims = size(out)
    mask = FlowGeometries.Grids.mask(grid)
    b1, b2 = _sep_nd_buffers(fp, out, Val(R))
    @. b2 = mask * field   # a `Bool` strong zero: an inactive cell contributes nothing, whatever it holds
    _sep_pass_chain!(out, b2, fp, dims, Val(R), Val(1), driver, b1, b2)
    prof = fp.profiles
    inv = fp.invrenorm
    driver(CartesianIndices(dims)) do I
        @inbounds begin
            Is = CartesianIndex(ntuple(d -> I[d], Val(R)))
            if inv === nothing
                out[I] = _normalized(out[I], prod(ntuple(d -> prof[d][I[d]], Val(R))))
            else
                # Deformable on a masked grid: an inactive target is zero.
                out[I] = mask[Is] ? out[I] * inv[Is] : zero(T)
            end
        end
    end
    return out
end

# The footprint's own pass buffers when the apply is unbatched; batch-sized ones otherwise, since the
# chain's intermediates must match the driven shape.
@inline function _sep_nd_buffers(fp, out::AbstractArray{T}, ::Val{R}) where {T,R}
    ndims(out) == R && return (fp.scratch, fp.masked_input)
    return (similar(out), similar(out))
end

# A Gaussian on a 1-D or true-3-D Cartesian grid is separable exactly as it is in 2-D, so it takes the
# `N`-pass engine rather than `FilterFootprintND`'s full-box enumeration. More specific than the
# generic-kernel methods below (constrained on the kernel too), and unconstrained in the axis types
# because separability does not require uniform spacing — see `_separable_axis_weights`.
build_footprint(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,1},
    kernel::SeparableKernel, scale::T; kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}} =
    _build_separable_nd(grid, kernel, scale; kwargs...)

build_footprint(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,3},
    kernel::SeparableKernel, scale::T; kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}} =
    _build_separable_nd(grid, kernel, scale; kwargs...)

# Range axes AND a Gaussian is more specific than either of the two methods that would otherwise both
# apply (kernel-specific with free axes, axis-specific with a free kernel), so these resolve that pair.
# They route to the separable engine as well: uniform spacing makes the weight tables vectors instead
# of matrices, not a different algorithm.
build_footprint(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,1,S,TP,<:Tuple{AbstractRange}},
    kernel::SeparableKernel, scale::T; kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}, S, TP<:NTuple{1,FlowGeometries.Grids.AbstractTopology}} =
    _build_separable_nd(grid, kernel, scale; kwargs...)

build_footprint(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,3,S,TP,<:Tuple{AbstractRange,AbstractRange,AbstractRange}},
    kernel::SeparableKernel, scale::T; kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}, S, TP<:NTuple{3,FlowGeometries.Grids.AbstractTopology}} =
    _build_separable_nd(grid, kernel, scale; kwargs...)

build_footprint(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,1,S,TP,<:Tuple{AbstractRange}},
    kernel::Kernels.AbstractFilterKernel, scale::T; kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}, S, TP<:NTuple{1,FlowGeometries.Grids.AbstractTopology}} = _build_footprint_nd(grid, kernel, scale)
build_footprint(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,3,S,TP,<:Tuple{AbstractRange,AbstractRange,AbstractRange}},
    kernel::Kernels.AbstractFilterKernel, scale::T; kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}, S, TP<:NTuple{3,FlowGeometries.Grids.AbstractTopology}} = _build_footprint_nd(grid, kernel, scale)
