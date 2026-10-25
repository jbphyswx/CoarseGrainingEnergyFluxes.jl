# ---------------------------------------------------------------------------
# N-dimensional engines for 1-D and 3-D Cartesian grids
# ---------------------------------------------------------------------------

"""
    SeparableFootprintND{N,T}

A separable kernel in `N` dimensions: one weight table per axis, applied as `N` successive 1-D passes,
`∑(2wᵈ+1)` multiply-adds per point.

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
    invrenorm::IAT     # dense reciprocal Deformable denominator on a masked grid, or nothing
    strategy::MS       # the strategy the denominator was built for
    masked::Bool
    bound::Bool        # the strategies' denominators differ: a cell is inactive or a window leaves the grid
    masked_input::AT
    scratch::AT
end

"""
    _sep_serial(f, indices)

The default pass driver: apply `f` to every index in order. A substitute driver may call `f` on the
indices in any order or concurrently, and must return once every call has completed; each call writes
its own outputs, so the result is bit-identical.
"""
@inline function _sep_serial(f::F, indices) where {F}
    for I in indices
        f(I)
    end
    return nothing
end

"""
    _drive_pass!(f, driver, indices, dst)
    _drive_pointwise!(f, driver, indices)

A separable pass's sweep under `driver`, `f(I)` writing line `I` of `dst`, and the normalization's
pointwise sweep, which reads and writes its output in place. Both call `driver(f, indices)`; a driver
that divides a pass among processes defines its own methods, assembling `dst` on each.
"""
@inline _drive_pass!(f::F, driver::D, indices, _dst) where {F,D} = driver(f, indices)
@inline _drive_pointwise!(f::F, driver::D, indices) where {F,D} = driver(f, indices)

@inline function _separable_pass!(
    dst::AbstractArray{T,N}, src::AbstractArray{T,N}, g::AbstractVecOrMat{T},
    lim::Int, periodic::Bool, dims::NTuple{N,Int}, ::Val{d}, driver::D,
) where {T<:AbstractFloat, N, d, D}
    n = dims[d]
    # One driver call per dimension-1 line, which that call alone writes. On rank 1 `CartesianIndices(())`
    # holds one index, so the column pass runs once over the whole vector.
    _drive_pass!(driver, CartesianIndices(Base.tail(dims)), dst) do J
        if d == 1
            _separable_col_pass!(dst, src, g, lim, periodic, dims[1], Tuple(J))
        else
            _separable_slab_pass!(dst, src, g, lim, periodic, n, dims[1], Tuple(J), Val(d))
        end
    end
    return dst
end

# The pass along dimension `d > 1` into the dimension-1 line at `J`. The weight depends on the output
# position `J[d-1]` along `d`, constant over the line, so each tap is an axpy.
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

# Passes `d…N`, unrolled by recursion on `Val(d)`: pass `d < N` writes `b1` for odd `d` and `b2` for even
# `d`, and pass `N` writes `dst`. `src` may alias `b2`; `dst` aliases neither buffer. `dims` is the driven
# shape, trailing batch axes included, and `N` is the grid's rank, the number of passes.
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
    (; g, lim, periodic, mf) = _separable_tables(grid, kernel, scale)

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
                FlowGeometries.Grids.coordinates(grid, d), lim[d], g[d], periodic[d], kernel, scale,
                convert(AbstractVector{T}, mf[d]), zerofill,
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

Filter `field` into `out`: the `R` separable passes, then the pointwise normalization. `driver` runs
each pass's index sweep under [`_sep_serial`](@ref)'s contract. `strategy` must match the footprint's
wherever the two strategies' denominators differ (`fp.bound`).

`out` and `field` may carry trailing batch axes beyond the grid's rank `R`; each pass covers the whole
batch. The profiles, the renormalization array and the mask are indexed by the leading `R` components
of the driven index.
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
    _drive_pointwise!(driver, CartesianIndices(dims)) do I
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

# The footprint's own pass buffers for an unbatched apply, fresh batch-sized ones otherwise.
@inline function _sep_nd_buffers(fp, out::AbstractArray{T}, ::Val{R}) where {T,R}
    ndims(out) == R && return (fp.scratch, fp.masked_input)
    return (similar(out), similar(out))
end

# A `SeparableKernel` on a 1-D or 3-D Cartesian grid of any spacing takes the `N`-pass engine.
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

# `Range` axes with a `SeparableKernel` match both the methods above and those below; these resolve the
# ambiguity, to the separable engine.
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
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}, S, TP<:NTuple{1,FlowGeometries.Grids.AbstractTopology}} =
    _build_footprint_nd(grid, kernel, scale; kwargs...)
build_footprint(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,3,S,TP,<:Tuple{AbstractRange,AbstractRange,AbstractRange}},
    kernel::Kernels.AbstractFilterKernel, scale::T; kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}, S, TP<:NTuple{3,FlowGeometries.Grids.AbstractTopology}} =
    _build_footprint_nd(grid, kernel, scale; kwargs...)

"""
    LineRuns{M,VO,VI,VT}

The taps of a translation-invariant footprint in `M + 1` dimensions by line: run `s` covers the
consecutive axis-1 offsets `di0[s] + t − 1` at offsets `tail[s]` along axes `2…M+1`, with weights
`w[start[s] + t − 1]` for `t = 1:(start[s+1] − start[s])`.
"""
struct LineRuns{M, VO<:AbstractVector{NTuple{M,Int}}, VI<:AbstractVector{Int}, VT<:AbstractVector}
    tail::VO
    di0::VI
    start::VI
    w::VT
end

# The runs of taps listed with the axis-1 offset varying fastest, a run broken wherever the offsets along
# the other axes change or the axis-1 offset skips.
function _line_runs(offsets::AbstractVector{NTuple{N,Int}}, w::AbstractVector) where {N}
    tail = NTuple{N - 1,Int}[]
    di0 = Int[]
    start = Int[]
    for k in eachindex(offsets)
        o = offsets[k]
        if k == firstindex(offsets) || Base.tail(o) != Base.tail(offsets[k - 1]) || o[1] != offsets[k - 1][1] + 1
            push!(tail, Base.tail(o))
            push!(di0, o[1])
            push!(start, k)
        end
    end
    push!(start, lastindex(offsets) + 1)
    return LineRuns(tail, di0, start, w)
end

"""
    FilterFootprintND{N,T}

Translation-invariant footprint of a 1-D or 3-D Cartesian grid with uniform axes: the in-support
neighbour offsets (`NTuple{N,Int}`) and their weights `w = kernel_weight(distance) · cell_measure`, which
a device applies point by point, and the same taps as [`LineRuns`](@ref), which the host applies as one
contiguous axpy per tap along axis 1; both sum a point's taps in the order `offsets` lists them.
`invden` is the reciprocal window mass under `strategy`, `ZeroFill`'s including the mass past a bounded
edge ([`_exterior_mass`](@ref)); it is zero where the window is empty or, under `Deformable`, the target
inactive.
"""
struct FilterFootprintND{
    N, T<:AbstractFloat, VO<:AbstractVector{NTuple{N,Int}}, VT<:AbstractVector{T},
    RT<:LineRuns, A<:AbstractArray{T,N}, MS<:AbstractMaskStrategy,
}
    offsets::VO
    w::VT
    runs::RT
    periodic::NTuple{N,Bool}
    strategy::MS       # the strategy `invden` was accumulated under
    masked::Bool       # the grid has an inactive cell, so the sweep reads `mask · field`
    bound::Bool        # the strategies' denominators differ: a cell is inactive or a window leaves the grid
    invden::A
    masked_input::A    # `mask · field` per apply on a masked grid; empty otherwise
end

# Every tap's contribution to output line `J`, the indices along axes `2…N`, added into `oc` from `src`.
@inline function _nd_line_accumulate!(
    oc::AbstractVector{T}, src::AbstractArray, runs::LineRuns{M}, J::NTuple{M,Int},
    dims::NTuple{N,Int}, periodic::NTuple{N,Bool},
) where {T, M, N}
    tdims, tperiodic = Base.tail(dims), Base.tail(periodic)
    w = runs.w
    @inbounds for s in eachindex(runs.tail)
        JJ = map((j, o, n, p) -> (jj = j + o; 1 <= jj <= n ? jj : p ? mod1(jj, n) : 0), J, runs.tail[s], tdims, tperiodic)
        all(>(0), JJ) || continue
        fc = view(src, :, JJ...)
        k0 = runs.start[s]
        _run_accumulate!(oc, fc, w, k0, runs.start[s + 1] - 1, runs.di0[s] - k0, periodic[1], dims[1])
    end
    return oc
end

"""
    _nd_out_line!(out, src, fp::FilterFootprintND, J) -> nothing

Output line `J` (`out[:, J...]`) from the source [`_nd_source!`](@ref) returned, scaled by the reciprocal
window mass. Writes that line alone, so lines may run concurrently.
"""
@inline function _nd_out_line!(
    out::AbstractArray{T,N}, src::AbstractArray, fp::FilterFootprintND{N,T}, J::NTuple{M,Int},
) where {T, N, M}
    oc = view(out, :, J...)
    @inbounds @simd for i in eachindex(oc)
        oc[i] = zero(T)
    end
    _nd_line_accumulate!(oc, src, fp.runs, J, size(out), fp.periodic)
    inv = view(fp.invden, :, J...)
    @inbounds @simd for i in eachindex(oc)
        oc[i] *= inv[i]
    end
    return nothing
end

"""
    _nd_source!(fp::FilterFootprintND, field, grid) -> src

The array the line sweep reads: on a masked grid the plan's buffer, filled with `mask · field`;
otherwise `field` itself. Call it before the lines run.
"""
function _nd_source!(fp::FilterFootprintND, field::AbstractArray, grid::FlowGeometries.Grids.StructuredGrid)
    fp.masked || return field
    mask = FlowGeometries.Grids.mask(grid)
    @. fp.masked_input = mask * field   # a `Bool` strong zero: an inactive cell contributes nothing
    return fp.masked_input
end

@noinline _nd_strategy_mismatch() = throw(ArgumentError(
    "FilterFootprintND is being applied with a different mask strategy than it was built for, on a grid " *
    "where the two normalize differently: a cell is inactive, or a window reaches past a bounded edge. " *
    "Rebuild the plan with the `mask_strategy` you intend to apply with.",
))

@inline function _nd_check_strategy(fp::FilterFootprintND, strategy::AbstractMaskStrategy)
    (!fp.bound || typeof(strategy) === typeof(fp.strategy)) || _nd_strategy_mismatch()
    return nothing
end

function apply_footprint_nd!(
    out::AbstractArray{T,N}, field::AbstractArray, grid::FlowGeometries.Grids.StructuredGrid{T,G,N},
    fp::FilterFootprintND{N,T}, strategy::AbstractMaskStrategy,
) where {N, T<:AbstractFloat, G}
    _nd_check_strategy(fp, strategy)
    src = _nd_source!(fp, field, grid)
    for J in CartesianIndices(Base.tail(size(out)))
        _nd_out_line!(out, src, fp, Tuple(J))
    end
    return out
end

function _build_footprint_nd(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    kwargs...,
) where {N, T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    rad = Kernels.kernel_radius(kernel, scale, Val(N))
    # The caller's dispatch guarantees every coordinate axis is a `Range`.
    spacing = ntuple(d -> step(FlowGeometries.Grids.coordinates(grid, d)), N)
    A = FlowGeometries.Grids.measure(grid)[ntuple(_ -> 1, N)...]   # uniform Cartesian cell measure
    lim = ntuple(d -> spacing[d] > 0 ? ceil(Int, rad / spacing[d]) : 0, N)
    offsets = NTuple{N,Int}[]
    w = T[]
    sizehint!(offsets, prod(2 .* lim .+ 1))
    sizehint!(w, prod(2 .* lim .+ 1))
    for off in CartesianIndices(ntuple(d -> (-lim[d]):lim[d], N))
        o = Tuple(off)
        d2 = zero(T)
        for d in 1:N
            d2 += (T(o[d]) * spacing[d])^2
        end
        dist = sqrt(d2)
        if dist <= rad
            push!(offsets, o)
            push!(w, Kernels.kernel_weight(kernel, dist, scale, Val(N)) * A)
        end
    end
    runs = _line_runs(offsets, w)
    dims = FlowGeometries.Grids.size_tuple(grid)
    periodic = FlowGeometries.Grids.periodic_flags(grid)
    mask = FlowGeometries.Grids.mask(grid)
    masked = !all(mask)
    zerofill = mask_strategy isa ZeroFill
    exterior = _exterior_mass(grid, kernel, scale)
    # Window mass: `ZeroFill` counts every in-support cell and those past a bounded edge, `Deformable`
    # only the active in-domain ones.
    massf = zerofill ? fill(one(T), dims) : T.(mask)
    den = zeros(T, dims)
    for J in CartesianIndices(Base.tail(dims))
        _nd_line_accumulate!(view(den, :, Tuple(J)...), massf, runs, Tuple(J), dims, periodic)
    end
    zerofill && exterior !== nothing && (den .+= exterior)
    @inbounds for I in CartesianIndices(dims)
        den[I] = (zerofill || mask[I]) ? _inv_mass(den[I]) : zero(T)
    end
    buf = masked ? zeros(T, dims) : zeros(T, ntuple(_ -> 0, Val(N)))
    bound = masked || _reaches_exterior(grid, kernel, scale)
    return FilterFootprintND(offsets, w, runs, periodic, mask_strategy, masked, bound, den, buf)
end
