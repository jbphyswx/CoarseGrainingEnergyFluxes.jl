# ---------------------------------------------------------------------------
# Exact 3-D top-hat by per-plane prefix sums: O(N·w_y·w_z) rather than O(N·w³).
#
# The generic N-dimensional engine walks the whole inscribed ball at every point. For a top-hat that is
# unnecessary: the weight is constant inside the support and the cell volume is constant on a uniform
# grid, so the window sum is a plain COUNT-weighted sum, and for each `(dj, dk)` offset the admissible
# `di` form one contiguous interval
#
#     |di · dx| ≤ sqrt(rad² − (dj·dy)² − (dk·dz)²)
#
# whose endpoints do not depend on position. One prefix scan along axis 1 therefore reduces each of
# those intervals to a subtraction, leaving `(2·dj_lim+1)(2·dk_lim+1)` of them per point instead of
# `(4/3)π w³` neighbour visits.
#
# The numerator and the window mass are the same operator on different inputs, exactly as in the banded
# engine: `mask · field` for one, and `1` (`ZeroFill`, which counts every in-support cell) or the mask
# (`Deformable`) for the other. So the mass is precomputed per scale and the apply is a sum and a
# multiply.
# ---------------------------------------------------------------------------

"""
    PrefixSum3DScratch{T,A} <: AbstractFilterScratch

The 3-D top-hat engine's per-apply buffer: the axis-1 cumulative scan of `mask · field`, sized
`(Nx+1, Ny, Nz)`. Refilled at the start of every apply, so one copy serves a whole sweep.
"""
struct PrefixSum3DScratch{T<:AbstractFloat, A<:AbstractArray{T,3}} <: AbstractFilterScratch
    prefix::A
end

"""
    PrefixSumTopHat3DPlan{T,SC,A,WT,MS}

Exact `O(N·w_y·w_z)` top-hat footprint for a uniform 3-D Cartesian `StructuredGrid`.

`wcell[dj, dk]` is the axis-1 half-width of the ball's slice at that offset, or `-1` where the slice is
empty. It is a function of the offsets alone — a uniform grid makes the ball translation-invariant — so
it is a `(2·dj_lim+1) × (2·dk_lim+1)` table, not a per-point one.

`invden` is the reciprocal window mass, built once per scale under the plan's mask strategy; see the
section comment above.
"""
struct PrefixSumTopHat3DPlan{
    T<:AbstractFloat, SC<:PrefixSum3DScratch{T}, A<:AbstractArray{T,3},
    WT<:AbstractMatrix{Int}, MS<:AbstractMaskStrategy,
}
    scratch::SC
    rad::T
    dj_lim::Int
    dk_lim::Int
    periodic::NTuple{3,Bool}
    masked::Bool
    strategy::MS
    wcell::WT      # axis-1 half-width per (dj, dk); -1 where that slice of the ball is empty
    invden::A      # 1/(window mass); zero where the target is inactive or the window empty
end

_prefixsum3d_scratch(grid::FlowGeometries.Grids.StructuredGrid{T,G,3}) where {T,G} =
    PrefixSum3DScratch(zeros(T, FlowGeometries.Grids.size_tuple(grid)[1] + 1,
                        FlowGeometries.Grids.size_tuple(grid)[2],
                        FlowGeometries.Grids.size_tuple(grid)[3]))

"""
    _prefixsum3d_fill_plane!(P, src, grid, masked, j, k) -> nothing

Cumulative sum along axis 1 of `src` for one `(j, k)` plane, with `P[1, j, k] = 0` as the empty prefix.
Writes only that plane's column, so planes are mutually independent and the threaded driver fills them
concurrently.
"""
@inline function _prefixsum3d_fill_plane!(
    P::AbstractArray{T,3}, src::AbstractArray, grid::FlowGeometries.Grids.StructuredGrid{T,G,3},
    masked::Bool, j::Integer, k::Integer,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    Nx = FlowGeometries.Grids.size_tuple(grid)[1]
    acc = zero(T)
    @inbounds begin
        P[1, j, k] = acc
        for i in 1:Nx
            acc += (masked && !FlowGeometries.Grids.isactive(grid, i, j, k)) ?
                zero(T) : T(src[i, j, k])
            P[i + 1, j, k] = acc
        end
    end
    return nothing
end

function _prefixsum3d_fill!(
    P::AbstractArray{T,3}, src::AbstractArray, grid::FlowGeometries.Grids.StructuredGrid{T,G,3},
    masked::Bool,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    _, Ny, Nz = FlowGeometries.Grids.size_tuple(grid)
    for k in 1:Nz, j in 1:Ny
        _prefixsum3d_fill_plane!(P, src, grid, masked, j, k)
    end
    return nothing
end

# Add every in-ball interval's contribution for output plane `(j, k)`. Shared by the apply (over the
# field's scan) and the plan build (over the mass field's scan), so the two cannot disagree about which
# cells are inside the ball.
@inline function _prefixsum3d_plane!(
    oc::AbstractVector{T}, P::AbstractArray{T,3}, fp_wcell::AbstractMatrix{Int},
    dj_lim::Int, dk_lim::Int, periodic::NTuple{3,Bool}, Nx::Int, Ny::Int, Nz::Int,
    j::Int, k::Int,
) where {T<:AbstractFloat}
    @inbounds for dk in (-dk_lim):dk_lim
        kk = k + dk
        if kk < 1 || kk > Nz
            periodic[3] || continue
            kk = mod1(kk, Nz)
        end
        for dj in (-dj_lim):dj_lim
            w = fp_wcell[dj + dj_lim + 1, dk + dk_lim + 1]
            w < 0 && continue
            jj = j + dj
            if jj < 1 || jj > Ny
                periodic[2] || continue
                jj = mod1(jj, Ny)
            end
            if periodic[1]
                # The `2w+1` raw offsets are `q` whole turns of the axis and a remainder of `rc` cells,
                # each cell counted once per image the offsets land on.
                q, rc = divrem(2 * w + 1, Nx)
                if q > 0
                    tot = q * (P[Nx + 1, jj, kk] - P[1, jj, kk])
                    @simd for i in 1:Nx
                        oc[i] += tot
                    end
                end
                if rc > 0
                    for i in 1:Nx
                        lo = mod1(i - w, Nx)
                        hi = lo + rc - 1
                        oc[i] += hi <= Nx ? P[hi + 1, jj, kk] - P[lo, jj, kk] :
                                 (P[Nx + 1, jj, kk] - P[lo, jj, kk]) + (P[hi - Nx + 1, jj, kk] - P[1, jj, kk])
                    end
                end
            else
                # Bounded: peel the clamped edges so the interior run is branch-free.
                ilo = min(w + 1, Nx + 1)
                ihi = max(Nx - w, 0)
                for i in 1:min(w, Nx)
                    oc[i] += P[min(Nx, i + w) + 1, jj, kk] - P[1, jj, kk]
                end
                @simd for i in ilo:ihi
                    oc[i] += P[i + w + 1, jj, kk] - P[i - w, jj, kk]
                end
                for i in max(ihi + 1, w + 1):Nx
                    oc[i] += P[Nx + 1, jj, kk] - P[max(1, i - w), jj, kk]
                end
            end
        end
    end
    return oc
end

function _build_prefixsum_tophat_3d(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,3},
    kernel::Kernels.TopHatKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    scratch::Union{Nothing,PrefixSum3DScratch} = nothing,
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    Nx, Ny, Nz = FlowGeometries.Grids.size_tuple(grid)
    rad = Kernels.kernel_radius(kernel, scale)
    dx = step(FlowGeometries.Grids.coordinates(grid, 1))
    dy = step(FlowGeometries.Grids.coordinates(grid, 2))
    dz = step(FlowGeometries.Grids.coordinates(grid, 3))
    periodic = ntuple(d -> FlowGeometries.Grids.isperiodic(grid, d), 3)
    # A periodic direction tiles, so the ball reaches a plane through each of its images and the band
    # count is not capped at the axis.
    dj_lim = dy > 0 ? (periodic[2] ? ceil(Int, rad / dy) : min(Ny - 1, ceil(Int, rad / dy))) : 0
    dk_lim = dz > 0 ? (periodic[3] ? ceil(Int, rad / dz) : min(Nz - 1, ceil(Int, rad / dz))) : 0

    # The ball's axis-1 half-width at each (dj, dk). `-1` marks a slice the ball never reaches, matching
    # the `dist <= rad` gate the general engine applies per offset.
    wcell = fill(-1, 2 * dj_lim + 1, 2 * dk_lim + 1)
    for dk in (-dk_lim):dk_lim, dj in (-dj_lim):dj_lim
        rem2 = rad^2 - (T(dj) * dy)^2 - (T(dk) * dz)^2
        rem2 < 0 && continue
        wcell[dj + dj_lim + 1, dk + dk_lim + 1] = dx > 0 ? floor(Int, sqrt(rem2) / dx) : 0
    end

    sc = scratch === nothing ? _prefixsum3d_scratch(grid) : scratch
    masked = !all(FlowGeometries.Grids.mask(grid))

    # Window mass: `ZeroFill` counts every in-support cell, `Deformable` only the active ones.
    mass = Array{T,3}(undef, Nx, Ny, Nz)
    if mask_strategy isa ZeroFill
        fill!(mass, one(T))
    else
        @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
            mass[i, j, k] = FlowGeometries.Grids.isactive(grid, i, j, k) ? one(T) : zero(T)
        end
    end
    P = sc.prefix
    _prefixsum3d_fill!(P, mass, grid, false)
    den = zeros(T, Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny
        oc = view(den, :, j, k)
        _prefixsum3d_plane!(oc, P, wcell, dj_lim, dk_lim, periodic, Nx, Ny, Nz, j, k)
    end
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        den[i, j, k] = (FlowGeometries.Grids.isactive(grid, i, j, k) && den[i, j, k] > T(1e-15)) ?
            inv(den[i, j, k]) : zero(T)
    end

    return PrefixSumTopHat3DPlan(sc, rad, dj_lim, dk_lim, periodic, masked, mask_strategy, wcell, den)
end

@noinline function _prefixsum3d_strategy_mismatch()
    throw(ArgumentError(
        "PrefixSumTopHat3DPlan is being applied to a MASKED grid with a different mask strategy than " *
        "it was built for. Its normalization is precomputed per scale from the grid, the mask and the " *
        "strategy, so one plan cannot serve both. Rebuild it with the `mask_strategy` you will apply with.",
    ))
end

@inline function _prefixsum3d_check_strategy(fp::PrefixSumTopHat3DPlan, strategy::AbstractMaskStrategy)
    (!fp.masked || typeof(strategy) === typeof(fp.strategy)) || _prefixsum3d_strategy_mismatch()
    return nothing
end

function apply_prefixsum_tophat_3d!(
    out::AbstractArray{T,3}, field::AbstractArray,
    grid::FlowGeometries.Grids.StructuredGrid{T,G,3}, fp::PrefixSumTopHat3DPlan{T},
    strategy::AbstractMaskStrategy,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    _prefixsum3d_check_strategy(fp, strategy)
    Nx, Ny, Nz = FlowGeometries.Grids.size_tuple(grid)
    P = fp.scratch.prefix
    _prefixsum3d_fill!(P, field, grid, fp.masked)
    invden = fp.invden
    for k in 1:Nz, j in 1:Ny
        oc = view(out, :, j, k)
        @inbounds @simd for i in 1:Nx
            oc[i] = zero(T)
        end
        _prefixsum3d_plane!(oc, P, fp.wcell, fp.dj_lim, fp.dk_lim, fp.periodic, Nx, Ny, Nz, j, k)
        @inbounds @simd for i in 1:Nx
            oc[i] *= invden[i, j, k]
        end
    end
    return out
end

apply_footprint!(
    out::AbstractArray{T,3}, field::AbstractArray, grid::FlowGeometries.Grids.StructuredGrid{T,G,3},
    fp::PrefixSumTopHat3DPlan{T}, strategy::AbstractMaskStrategy,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}} =
    apply_prefixsum_tophat_3d!(out, field, grid, fp, strategy)

# More specific than the generic 3-D Range-axis method above (constrained on the kernel), so a top-hat
# on a uniform Cartesian volume selects this instead of the O(w³) ball walk.
build_footprint(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,3,S,TP,<:Tuple{AbstractRange,AbstractRange,AbstractRange}},
    kernel::Kernels.TopHatKernel, scale::T; kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}, S, TP<:NTuple{3,FlowGeometries.Grids.AbstractTopology}} =
    _build_prefixsum_tophat_3d(grid, kernel, scale; kwargs...)

# General path: at least one axis is a plain AbstractVector (no uniformity guarantee), OR the
# geometry is non-Cartesian (no translation-invariant fast path exists, see above) — less specific
# than the two Cartesian-only methods above, reached whenever they don't match.
build_footprint(grid::FlowGeometries.Grids.StructuredGrid{T,G,1}, kernel::Kernels.AbstractFilterKernel, scale::T; kwargs...) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}} =
    _build_footprint_nd_scattered(grid, kernel, scale; kwargs...)
build_footprint(grid::FlowGeometries.Grids.StructuredGrid{T,G,3}, kernel::Kernels.AbstractFilterKernel, scale::T; kwargs...) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}} =
    _build_footprint_nd_scattered(grid, kernel, scale; kwargs...)

function _build_footprint_nd(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N},
    kernel::Kernels.AbstractFilterKernel,
    scale::T,
) where {N, T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    rad = Kernels.kernel_radius(kernel, scale)
    # Real per-axis step, read from the axis itself (already proven uniform by its Range type via
    # the calling method's dispatch constraint) — not the geometry's separately-stored dx/dy/dz,
    # so there's no possibility of the two disagreeing.
    spacing = ntuple(d -> step(FlowGeometries.Grids.coordinates(grid, d)), N)
    A = FlowGeometries.Grids.measure(grid)[ntuple(_ -> 1, N)...]   # uniform Cartesian cell measure
    lim = ntuple(d -> spacing[d] > 0 ? ceil(Int, rad / spacing[d]) : 0, N)
    offsets = NTuple{N,Int}[]
    w = T[]
    # Exact window size (single shared translation-invariant footprint): every candidate offset in
    # this hyperrectangle is visited exactly once below.
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
            push!(w, Kernels.kernel_weight(kernel, dist, scale) * A)
        end
    end
    return FilterFootprintND(offsets, w)
end

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
    NDScatteredFilterPlan{N, T, K, C, MT}

N-D (1D or 3D) analog of [`ScatteredFilterPlan`](@ref), for when at least one of the N axes is a plain
`AbstractVector` (no type-level uniformity proof): the kernel, the support radius, the per-axis window
the cache size is estimated from, and the grid's ball-query topology. No translation invariance is
assumed. `cache` holds the materialized [`NDScatteredCache`](@ref) only when the plan's cache strategy
decided to build it, `nothing` otherwise (apply-time recomputation).
"""
struct NDScatteredFilterPlan{
    N, T<:AbstractFloat, K<:Kernels.AbstractFilterKernel, C<:Union{Nothing,NDScatteredCache{N,T}}, MT,
}
    kernel::K
    scale::T
    rad::T
    lim::NTuple{N,Int}
    cache::C
    topology::MT   # built once; the cache build and the streaming apply both query through it
end

# Ball-gated, as in 2-D. The 1-D case needs no correction (a 1-D ball IS the interval).
@inline _nd_box_to_ball(::Val{N}) where {N} = N == 3 ? _BOX_TO_BALL_3D : (N == 2 ? _BOX_TO_BALL_2D : 1.0)
@inline _nd_scattered_cache_bytes(dims::NTuple{N,Int}, lim::NTuple{N,Int}, ::Type{T}) where {N,T} =
    round(Int, prod(dims) * prod(2 .* lim .+ 1) * (sizeof(NTuple{N,Int}) + sizeof(T)) /
               _nd_box_to_ball(Val(N)))

function _build_footprint_nd_scattered(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    cache_strategy::AbstractCacheStrategy = AutoCache(),
    cache_byte_budget::Integer = DEFAULT_CACHE_BYTE_BUDGET,
    kwargs...,   # accepts (and ignores) mask_strategy — only the 2D separable-Gaussian path needs it
) where {N, T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    dims = FlowGeometries.Grids.size_tuple(grid)
    rad = Kernels.kernel_radius(kernel, scale)
    # Per-axis index bound at the worst cell, for the cache estimate only; the traversal is the grid's.
    lim = FlowGeometries.Connectivity.metric_window(grid, rad)
    mt = _query_topology(grid, rad)

    cache = if _should_cache(cache_strategy, _nd_scattered_cache_bytes(dims, lim, T), cache_byte_budget)
        nbrs = NTuple{N,Int}[]
        w = T[]
        sizehint!(nbrs, prod(dims) * prod(2 .* lim .+ 1))
        sizehint!(w, prod(dims) * prod(2 .* lim .+ 1))
        lin = LinearIndices(dims)
        ptr = Vector{Int}(undef, prod(dims) + 1)
        ptr[1] = 1
        for I in CartesianIndices(dims)
            t = lin[I]
            _nd_foldl(nothing, grid, Tuple(I), rad, mt) do _, J, d
                push!(nbrs, J)
                push!(w, Kernels.kernel_weight(kernel, d, scale) * FlowGeometries.Grids.area(grid, J...))
                nothing
            end
            ptr[t+1] = length(nbrs) + 1
        end
        NDScatteredCache(nbrs, w, ptr)
    else
        nothing
    end
    return NDScatteredFilterPlan(kernel, scale, rad, lim, cache, mt)
end

# Folds `acc = f(acc, J, d)` over every in-support neighbour of `Ti`, the centre included, through the
# grid's own ball query. A periodic Cartesian direction tiles, so each image of a cell inside the
# support comes at its own displacement; a periodic angle identifies, so its cells come once. The cache
# builder and the streaming apply both enumerate through it.
@inline _nd_foldl(
    f::F, acc, grid::FlowGeometries.Grids.StructuredGrid, Ti::NTuple{N,Int}, rad, mt,
) where {F, N} = FlowGeometries.Connectivity.fold_within(
    f, acc, grid, Ti...;
    ball = rad, self = true, active_only = false, topology = mt, images = _image_convention(grid),
)

@inline _image_convention(grid::FlowGeometries.Grids.StructuredGrid) =
    _image_convention(FlowGeometries.Grids.grid_geometry(grid))
@inline _image_convention(::FlowGeometries.Geometry.AbstractCartesianGeometry) =
    FlowGeometries.Connectivity.AllImages()
@inline _image_convention(::FlowGeometries.Geometry.AbstractGeometry) =
    FlowGeometries.Connectivity.NearestImage()

# Shifted neighbour multi-index with per-axis periodic wrap; returns (index, in-bounds?).
@inline function _shift_index(I::NTuple{N,Int}, o::NTuple{N,Int}, dims::NTuple{N,Int}, periodic::NTuple{N,Bool}) where {N}
    J = ntuple(N) do d
        jj = I[d] + o[d]
        (jj < 1 || jj > dims[d]) ? (periodic[d] ? mod1(jj, dims[d]) : 0) : jj
    end
    return J, !any(==(0), J)
end

# Per-point kernel factored out of `apply_footprint_nd!` so a parallel (per-point-independent) loop
# can reuse the EXACT same arithmetic instead of duplicating it — see
# `CoarseGrainingEnergyFluxesOhMyThreadsExt`'s ND threaded hook.
@inline function _footprint_nd_point(
    field::AbstractArray, fp::FilterFootprintND{N,T}, strategy::AbstractMaskStrategy,
    dims::NTuple{N,Int}, periodic::NTuple{N,Bool}, mask, I::CartesianIndex{N},
) where {N, T<:AbstractFloat}
    Ti = Tuple(I)
    ws = zero(T)
    wn = zero(T)
    @inbounds for k in eachindex(fp.offsets)
        J, valid = _shift_index(Ti, fp.offsets[k], dims, periodic)
        valid || continue
        active = mask[J...]
        wk = fp.w[k]
        if strategy isa ZeroFill
            wn += wk
            active && (ws += wk * field[J...])
        elseif active
            wn += wk
            ws += wk * field[J...]
        end
    end
    return wn > T(1e-15) ? ws / wn : zero(T)
end

function apply_footprint_nd!(
    out::AbstractArray{T,N},
    field::AbstractArray,
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N},
    fp::FilterFootprintND{N,T},
    strategy::AbstractMaskStrategy,
) where {N, T<:AbstractFloat, G}
    dims = FlowGeometries.Grids.size_tuple(grid)
    periodic = FlowGeometries.Grids.periodic_flags(grid)
    mask = FlowGeometries.Grids.mask(grid)
    fill!(out, zero(T))
    @inbounds for I in CartesianIndices(out)
        mask[I] || continue
        out[I] = _footprint_nd_point(field, fp, strategy, dims, periodic, mask, I)
    end
    return out
end

@inline function _footprint_nd_point_cached(
    field::AbstractArray, cache::NDScatteredCache{N,T}, strategy::AbstractMaskStrategy,
    mask, lin::LinearIndices{N}, I::CartesianIndex{N},
) where {N, T<:AbstractFloat}
    t = lin[I]
    lo = cache.ptr[t]
    hi = cache.ptr[t+1] - 1
    ws = zero(T)
    wn = zero(T)
    @inbounds for k in lo:hi
        J = cache.nbrs[k]
        active = mask[J...]
        wk = cache.w[k]
        if strategy isa ZeroFill
            wn += wk
            active && (ws += wk * field[J...])
        elseif active
            wn += wk
            ws += wk * field[J...]
        end
    end
    return wn > T(1e-15) ? ws / wn : zero(T)
end

# Streaming (no cache) per-point recompute, over the same enumeration the cache builder uses. The
# accumulator is threaded through the fold's return value rather than captured and mutated, which is
# what keeps this allocation-free.
function _footprint_nd_point_streaming(
    field::AbstractArray, grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, fp::NDScatteredFilterPlan{N,T},
    strategy::AbstractMaskStrategy, mask, I::CartesianIndex{N},
) where {N, T<:AbstractFloat, G}
    kernel, scale = fp.kernel, fp.scale
    ws, wn = _nd_foldl((zero(T), zero(T)), grid, Tuple(I), fp.rad, fp.topology) do acc, J, d
        s, n = acc
        active = mask[J...]
        wk = Kernels.kernel_weight(kernel, d, scale) * FlowGeometries.Grids.area(grid, J...)
        if strategy isa ZeroFill
            return (active ? s + wk * field[J...] : s, n + wk)
        else
            active || return acc
            return (s + wk * field[J...], n + wk)
        end
    end
    return wn > T(1e-15) ? ws / wn : zero(T)
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
    fill!(out, zero(T))
    if fp.cache !== nothing
        lin = LinearIndices(dims)
        cache = fp.cache
        @inbounds for I in CartesianIndices(out)
            mask[I] || continue
            out[I] = _footprint_nd_point_cached(field, cache, strategy, mask, lin, I)
        end
    else
        @inbounds for I in CartesianIndices(out)
            mask[I] || continue
            out[I] = _footprint_nd_point_streaming(field, grid, fp, strategy, mask, I)
        end
    end
    return out
end

# Dispatch the apply on the footprint kind.
"""
    NodeFilterPlan{T}

Real-space filter footprint for a node set: per-node CSR neighbour blocks with their geometric weights
`w = kernel_weight(d) · control volume`.

A node set has no axes, so there is no index window to bound a search with and the neighbourhood
cannot be re-derived per apply the way a structured grid's can. It is found once, at plan time, and
stored — which also means the search is paid once per plan rather than once per field, and a single
`compute_Π!` makes six to nine applies against one plan.

The node itself is included. `Connectivity.neighbors_within` excludes it, matching stencil semantics
where a cell is not its own neighbour, but a filter's zero offset is a genuine contribution.
"""
struct NodeFilterPlan{T<:AbstractFloat, VI<:AbstractVector{Int}, VT<:AbstractVector{T}}
    nbrs::VI
    w::VT
    ptr::VI
end

_apply_serial!(out, field, grid, fp::FilterFootprint, strategy) =
    apply_footprint!(out, field, grid, fp, strategy, FlowGeometries.Grids.isperiodic(grid, 1), FlowGeometries.Grids.isperiodic(grid, 2))
_apply_serial!(out, field, grid, fp::ScatteredFilterPlan, strategy) =
    apply_footprint!(out, field, grid, fp, strategy, FlowGeometries.Grids.isperiodic(grid, 1), FlowGeometries.Grids.isperiodic(grid, 2))
_apply_serial!(out, field, grid, fp::FilterFootprintND, strategy) =
    apply_footprint_nd!(out, field, grid, fp, strategy)
_apply_serial!(out, field, grid, fp::NDScatteredFilterPlan, strategy) =
    apply_footprint_nd!(out, field, grid, fp, strategy)
_apply_serial!(out, field, grid, fp::PrefixSumTopHatPlan, strategy) =
    apply_prefixsum_tophat!(out, field, grid, fp, strategy)
_apply_serial!(out, field, grid, fp::PrefixSumTopHat3DPlan, strategy) =
    apply_prefixsum_tophat_3d!(out, field, grid, fp, strategy)
_apply_serial!(out, field, grid, fp::SeparableFootprintND, strategy) =
    apply_separable_nd!(out, field, grid, fp, strategy)
_apply_serial!(out, field, grid, fp::NodeFilterPlan, strategy) =
    apply_footprint!(out, field, grid, fp, strategy)
_apply_serial!(out, field, grid, fp::SeparableFootprint, strategy) =
    apply_separable!(out, field, grid, fp, strategy)
