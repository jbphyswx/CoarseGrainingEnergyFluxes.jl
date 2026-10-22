# ---------------------------------------------------------------------------
# Exact 3-D top-hat on a uniform Cartesian grid by per-plane prefix sums, O(N·w_y·w_z).
#
# The top-hat weight and the cell volume are constant, so the window sum is an unweighted sum over the
# in-support cells, and for each `(dj, dk)` offset the admissible `di` form one contiguous interval
#
#     |di · dx| ≤ sqrt(rad² − (dj·dy)² − (dk·dz)²)
#
# whose endpoints do not depend on position. One prefix scan along axis 1 reduces each interval to a
# subtraction, `(2·dj_lim+1)(2·dk_lim+1)` of them per point.
#
# The numerator and the window mass are the same operator on different inputs: `mask · field` for one,
# and `1` (`ZeroFill`, which counts every in-support cell) or the mask (`Deformable`) for the other. The
# mass is precomputed per scale, so the apply is a sum and a multiply.
# ---------------------------------------------------------------------------

"""
    PrefixSum3DScratch{T,A} <: AbstractFilterScratch

The 3-D top-hat engine's per-apply buffer: the axis-1 [`CompensatedScan`](@ref) of `mask · field`,
sized `(Nx+1, Ny, Nz)`. Refilled at the start of every apply, so one copy serves a whole sweep.
"""
struct PrefixSum3DScratch{T<:AbstractFloat, A<:AbstractArray{T,3}} <: AbstractFilterScratch
    prefix::CompensatedScan{T,A}
end

"""
    PrefixSumTopHat3DPlan{T,SC,A,WT,MS}

Exact `O(N·w_y·w_z)` top-hat footprint for a uniform 3-D Cartesian `StructuredGrid`.

`wcell` is the `(2·dj_lim+1) × (2·dk_lim+1)` table of the ball's axis-1 half-width at each offset
`(dj, dk)`, `-1` where that slice is empty; on a uniform grid it is the same at every point.

`invden` is the reciprocal window mass, built once per scale under `strategy`.
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
    bound::Bool    # the strategies' denominators differ: a cell is inactive or a window leaves the grid
    strategy::MS
    wcell::WT      # axis-1 half-width per (dj, dk); -1 where that slice of the ball is empty
    invden::A      # 1/(window mass); zero where empty or, under Deformable, at an inactive target
end

function _prefixsum3d_scratch(grid::FlowGeometries.Grids.StructuredGrid{T,G,3}) where {T,G}
    Nx, Ny, Nz = FlowGeometries.Grids.size_tuple(grid)
    return PrefixSum3DScratch(CompensatedScan(T, Nx + 1, Ny, Nz))
end

"""
    _prefixsum3d_fill_plane!(P, src, grid, masked, j, k) -> nothing

Compensated scan along axis 1 of `src` for one `(j, k)` plane, with entry 1 the empty prefix. Writes
only that plane's column, so planes may be filled concurrently.
"""
@inline function _prefixsum3d_fill_plane!(
    P::CompensatedScan{T}, src::AbstractArray, grid::FlowGeometries.Grids.StructuredGrid{T,G,3},
    masked::Bool, j::Integer, k::Integer,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    Nx = FlowGeometries.Grids.size_tuple(grid)[1]
    Ph, Pl = P.hi, P.lo
    h = l = zero(T)
    @inbounds begin
        Ph[1, j, k] = h
        Pl[1, j, k] = l
        for i in 1:Nx
            x = (masked && !FlowGeometries.Grids.isactive(grid, i, j, k)) ? zero(T) : T(src[i, j, k])
            h, l = _scan_step(h, l, x)
            Ph[i + 1, j, k] = h
            Pl[i + 1, j, k] = l
        end
    end
    return nothing
end

function _prefixsum3d_fill!(
    P::CompensatedScan{T}, src::AbstractArray, grid::FlowGeometries.Grids.StructuredGrid{T,G,3},
    masked::Bool,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    _, Ny, Nz = FlowGeometries.Grids.size_tuple(grid)
    for k in 1:Nz, j in 1:Ny
        _prefixsum3d_fill_plane!(P, src, grid, masked, j, k)
    end
    return nothing
end

# Add every in-ball interval's sum from `P` into `oc`, output plane `(j, k)`. The apply runs it over the
# field's scan and the plan build over the mass's.
@inline function _prefixsum3d_plane!(
    oc::AbstractVector{T}, P::CompensatedScan{T}, fp_wcell::AbstractMatrix{Int},
    dj_lim::Int, dk_lim::Int, periodic::NTuple{3,Bool}, Nx::Int, Ny::Int, Nz::Int,
    j::Int, k::Int,
) where {T<:AbstractFloat}
    Ph, Pl = P.hi, P.lo
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
                    tot = q * _scan_diff(Ph, Pl, Nx + 1, 1, jj, kk)
                    @simd for i in 1:Nx
                        oc[i] += tot
                    end
                end
                if rc > 0
                    for i in 1:Nx
                        lo = mod1(i - w, Nx)
                        hi = lo + rc - 1
                        oc[i] += hi <= Nx ? _scan_diff(Ph, Pl, hi + 1, lo, jj, kk) :
                                 _scan_diff(Ph, Pl, Nx + 1, lo, jj, kk) + _scan_diff(Ph, Pl, hi - Nx + 1, 1, jj, kk)
                    end
                end
            else
                # Bounded: peel the clamped edges so the interior run is branch-free.
                ilo = min(w + 1, Nx + 1)
                ihi = max(Nx - w, 0)
                for i in 1:min(w, Nx)
                    oc[i] += _scan_diff(Ph, Pl, min(Nx, i + w) + 1, 1, jj, kk)
                end
                @simd for i in ilo:ihi
                    oc[i] += _scan_diff(Ph, Pl, i + w + 1, i - w, jj, kk)
                end
                for i in max(ihi + 1, w + 1):Nx
                    oc[i] += _scan_diff(Ph, Pl, Nx + 1, max(1, i - w), jj, kk)
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

    # The ball's axis-1 half-width at each (dj, dk); `-1` marks a slice outside the ball.
    wcell = fill(-1, 2 * dj_lim + 1, 2 * dk_lim + 1)
    for dk in (-dk_lim):dk_lim, dj in (-dj_lim):dj_lim
        rem2 = rad^2 - (T(dj) * dy)^2 - (T(dk) * dz)^2
        rem2 < 0 && continue
        wcell[dj + dj_lim + 1, dk + dk_lim + 1] = dx > 0 ? floor(Int, sqrt(rem2) / dx) : 0
    end

    sc = scratch === nothing ? _prefixsum3d_scratch(grid) : scratch
    masked = !all(FlowGeometries.Grids.mask(grid))

    # Window mass, in cells: `ZeroFill` counts every in-support cell and the exterior ones past a bounded
    # edge, `Deformable` only the active in-domain ones.
    zerofill = mask_strategy isa ZeroFill
    mass = Array{T,3}(undef, Nx, Ny, Nz)
    if zerofill
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
    exterior = zerofill ? _exterior_mass(grid, kernel, scale) : nothing
    # The scan sums cells, and the exterior mass is a measure, so it enters divided by the cell volume.
    exterior === nothing || (den .+= exterior ./ FlowGeometries.Grids.area(grid, 1, 1, 1))
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        keep = zerofill || FlowGeometries.Grids.isactive(grid, i, j, k)
        den[i, j, k] = keep ? _inv_mass(den[i, j, k]) : zero(T)
    end

    bound = masked || _reaches_exterior(grid, kernel, scale)
    return PrefixSumTopHat3DPlan(sc, rad, dj_lim, dk_lim, periodic, masked, bound, mask_strategy, wcell, den)
end

@noinline function _prefixsum3d_strategy_mismatch()
    throw(ArgumentError(
        "PrefixSumTopHat3DPlan is being applied with a different mask strategy than it was built for, " *
        "on a grid where the two normalize differently: a cell is inactive, or a window reaches past a " *
        "bounded edge. Its normalization is precomputed per scale from the grid, the mask and the " *
        "strategy, so one plan cannot serve both. Rebuild it with the `mask_strategy` you will apply with.",
    ))
end

@inline function _prefixsum3d_check_strategy(fp::PrefixSumTopHat3DPlan, strategy::AbstractMaskStrategy)
    (!fp.bound || typeof(strategy) === typeof(fp.strategy)) || _prefixsum3d_strategy_mismatch()
    return nothing
end

"""
    _prefixsum3d_out_plane!(out, fp, j, k) -> nothing

Output plane `(j, k)` from the scan `fp.scratch.prefix` already holds, scaled by the reciprocal window
mass. Writes only that plane, so planes may run concurrently.
"""
@inline function _prefixsum3d_out_plane!(
    out::AbstractArray{T,3}, fp::PrefixSumTopHat3DPlan{T}, j::Int, k::Int,
) where {T<:AbstractFloat}
    Nx, Ny, Nz = size(out)
    oc = view(out, :, j, k)
    @inbounds @simd for i in 1:Nx
        oc[i] = zero(T)
    end
    _prefixsum3d_plane!(oc, fp.scratch.prefix, fp.wcell, fp.dj_lim, fp.dk_lim, fp.periodic, Nx, Ny, Nz, j, k)
    invden = fp.invden
    @inbounds @simd for i in 1:Nx
        oc[i] *= invden[i, j, k]
    end
    return nothing
end

function apply_prefixsum_tophat_3d!(
    out::AbstractArray{T,3}, field::AbstractArray,
    grid::FlowGeometries.Grids.StructuredGrid{T,G,3}, fp::PrefixSumTopHat3DPlan{T},
    strategy::AbstractMaskStrategy,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    _prefixsum3d_check_strategy(fp, strategy)
    _, Ny, Nz = FlowGeometries.Grids.size_tuple(grid)
    _prefixsum3d_fill!(fp.scratch.prefix, field, grid, fp.masked)
    for k in 1:Nz, j in 1:Ny
        _prefixsum3d_out_plane!(out, fp, j, k)
    end
    return out
end

apply_footprint!(
    out::AbstractArray{T,3}, field::AbstractArray, grid::FlowGeometries.Grids.StructuredGrid{T,G,3},
    fp::PrefixSumTopHat3DPlan{T}, strategy::AbstractMaskStrategy,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}} =
    apply_prefixsum_tophat_3d!(out, field, grid, fp, strategy)

# More specific in the kernel than the generic 3-D `Range`-axis method, so a top-hat on a uniform
# Cartesian volume dispatches here.
build_footprint(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,3,S,TP,<:Tuple{AbstractRange,AbstractRange,AbstractRange}},
    kernel::Kernels.TopHatKernel, scale::T; kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}, S, TP<:NTuple{3,FlowGeometries.Grids.AbstractTopology}} =
    _build_prefixsum_tophat_3d(grid, kernel, scale; kwargs...)

# Cell-by-cell engine for the 1-D and 3-D grids no more specific method takes: a plain
# `AbstractVector` axis, or a non-Cartesian geometry.
build_footprint(grid::FlowGeometries.Grids.StructuredGrid{T,G,1}, kernel::Kernels.AbstractFilterKernel, scale::T; kwargs...) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}} =
    _build_footprint_nd_scattered(grid, kernel, scale; kwargs...)
build_footprint(grid::FlowGeometries.Grids.StructuredGrid{T,G,3}, kernel::Kernels.AbstractFilterKernel, scale::T; kwargs...) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}} =
    _build_footprint_nd_scattered(grid, kernel, scale; kwargs...)

function _build_footprint_nd(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N},
    kernel::Kernels.AbstractFilterKernel,
    scale::T,
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
    return FilterFootprintND(offsets, w, _exterior_mass(grid, kernel, scale))
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

N-D (1D or 3D) analog of [`ScatteredFilterPlan`](@ref), for a grid with no translation-invariant
footprint, such as one with a plain `AbstractVector` axis: the kernel, the support radius, the
per-axis index window `lim` at the worst cell, and the grid's ball-query topology. `cache` holds the
[`NDScatteredCache`](@ref) when the plan's cache strategy builds one, `nothing` otherwise (apply-time
recomputation). `exterior` is the mass past a bounded edge that `ZeroFill` adds to each point's
denominator, or `nothing`. `weights` is the rule a cell's weight comes from, [`RadialWeights`](@ref)
or [`HighOrderWeights`](@ref).
"""
struct NDScatteredFilterPlan{
    N, T<:AbstractFloat, K<:Kernels.AbstractFilterKernel, C<:Union{Nothing,NDScatteredCache{N,T}}, MT,
    EX<:Union{Nothing,AbstractArray{T,N}}, W,
}
    kernel::K
    scale::T
    rad::T
    lim::NTuple{N,Int}
    cache::C
    topology::MT   # built once; the cache build and the streaming apply both query through it
    exterior::EX
    weights::W
end

# Entries of an N-D cache, counted in target order, stopping once past `limit`.
function _count_nd_scattered(grid, rad, mt, limit::Int)
    n = 0
    for I in CartesianIndices(FlowGeometries.Grids.size_tuple(grid))
        n = _nd_foldl((m, _, _) -> m + 1, n, grid, Tuple(I), rad, mt)
        n > limit && break
    end
    return n
end

# The N-D cache, `n` entries from `_count_nd_scattered`, filled in linear target order.
function _fill_nd_scattered(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, rule::RadialWeights, rad, mt, n::Int,
) where {T, G, N}
    dims = FlowGeometries.Grids.size_tuple(grid)
    nbrs = Vector{NTuple{N,Int}}(undef, n)
    w = Vector{T}(undef, n)
    ptr = Vector{Int}(undef, prod(dims) + 1)
    ptr[1] = 1
    k = 0
    for (t, I) in enumerate(CartesianIndices(dims))
        Ti = Tuple(I)
        k = _nd_foldl(k, grid, Ti, rad, mt) do m, J, d
            m += 1
            nbrs[m] = J
            w[m] = last(_cell_weight(rule, grid, t, Ti, J, d))
            m
        end
        ptr[t + 1] = k + 1
    end
    return NDScatteredCache(nbrs, w, ptr)
end

# An N-D cell-by-cell plan's neighbour cache (or `nothing`), its mass past a bounded edge (or `nothing`)
# and its weight rule, as `_scattered_parts` gives them in two dimensions.
function _nd_scattered_parts(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, kernel::Kernels.AbstractFilterKernel, scale::T,
    rad, mt, strategy, budget,
) where {T, G, N}
    rule = RadialWeights(kernel, scale, Val(N))
    n = _cache_entries(strategy, budget, _cache_entry_bytes(Val(N), T)) do limit
        _count_nd_scattered(grid, rad, mt, limit)
    end
    cache = n === nothing ? nothing : _fill_nd_scattered(grid, rule, rad, mt, n)
    return cache, _exterior_mass(grid, kernel, scale), rule
end

function _nd_scattered_parts(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, kernel::Kernels.HighOrderKernel, scale::T,
    rad, mt, strategy, budget,
) where {T, G, N}
    fold = (f, acc, I) -> _nd_foldl(f, acc, grid, I, rad, mt)
    keep = _cache_keep(strategy, budget, _cache_entry_bytes(Val(N), T))
    weights, exterior, kept = _highorder_targets(grid, kernel, scale, rad, fold, _ho_lattice(grid, rad), keep)
    cache = kept === nothing ? nothing : NDScatteredCache(kept...)
    return cache, exterior, weights
end

function _build_footprint_nd_scattered(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    cache_strategy::AbstractCacheStrategy = AutoCache(),
    cache_byte_budget::Integer = default_cache_byte_budget(),
    kwargs...,   # absorbs and ignores the other engines' keywords, such as `mask_strategy`
) where {N, T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    rad = _query_radius(grid, kernel, scale)
    # Per-axis index bound at the worst cell; the traversal is the grid's.
    lim = FlowGeometries.Connectivity.metric_window(grid, rad)
    mt = _query_topology(grid, rad)
    cache, exterior, weights = _nd_scattered_parts(grid, kernel, scale, rad, mt, cache_strategy, cache_byte_budget)
    # `cache` and `exterior` are typed `Union{Nothing,…}`, so the plan's type depends on the grid and
    # the kernel alone.
    return NDScatteredFilterPlan{
        N, T, typeof(kernel),
        Union{Nothing,NDScatteredCache{N,T,Vector{NTuple{N,Int}},Vector{T},Vector{Int}}}, typeof(mt),
        Union{Nothing,Array{T,N}}, typeof(weights),
    }(kernel, scale, rad, lim, cache, mt, exterior, weights)
end

# Folds `acc = f(acc, J, d)` over every in-support neighbour of `Ti`, the centre included, through the
# grid's own ball query. A periodic Cartesian direction tiles, so each image of a cell inside the
# support comes at its own displacement; a periodic angle identifies, so its cells come once.
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

# The three point functions below return zero at a target the strategy skips, so a caller evaluates
# them at every point. The parallel backend extensions call them too.
@inline function _footprint_nd_point(
    field::AbstractArray, fp::FilterFootprintND{N,T}, strategy::AbstractMaskStrategy,
    dims::NTuple{N,Int}, periodic::NTuple{N,Bool}, mask, I::CartesianIndex{N},
) where {N, T<:AbstractFloat}
    _filters_target(strategy, @inbounds mask[I]) || return zero(T)
    Ti = Tuple(I)
    ws = zero(T)
    wn = zero(T)
    zerofill = strategy isa ZeroFill
    @inbounds for k in eachindex(fp.offsets)
        J, valid = _shift_index(Ti, fp.offsets[k], dims, periodic)
        valid || continue
        active = mask[J...]
        wk = fp.w[k]
        if zerofill
            wn += wk
            active && (ws += wk * field[J...])
        elseif active
            wn += wk
            ws += wk * field[J...]
        end
    end
    zerofill && (wn = _add_exterior(wn, fp.exterior, I))
    return _normalized(ws, wn)
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
    @inbounds for I in CartesianIndices(out)
        out[I] = _footprint_nd_point(field, fp, strategy, dims, periodic, mask, I)
    end
    return out
end

@inline function _footprint_nd_point_cached(
    field::AbstractArray, cache::NDScatteredCache{N,T}, exterior, strategy::AbstractMaskStrategy,
    mask, lin::LinearIndices{N}, I::CartesianIndex{N},
) where {N, T<:AbstractFloat}
    _filters_target(strategy, @inbounds mask[I]) || return zero(T)
    t = lin[I]
    lo = cache.ptr[t]
    hi = cache.ptr[t+1] - 1
    ws = zero(T)
    wn = zero(T)
    zerofill = strategy isa ZeroFill
    @inbounds for k in lo:hi
        J = cache.nbrs[k]
        active = mask[J...]
        wk = cache.w[k]
        if zerofill
            wn += wk
            active && (ws += wk * field[J...])
        elseif active
            wn += wk
            ws += wk * field[J...]
        end
    end
    zerofill && (wn = _add_exterior(wn, exterior, I))
    return _normalized(ws, wn)
end

# Per-point recompute without a cache, over the enumeration the cache builder uses. The accumulator
# travels through the fold's return value, so the closure mutates no captured variable.
function _footprint_nd_point_streaming(
    field::AbstractArray, grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, fp::NDScatteredFilterPlan{N,T},
    strategy::AbstractMaskStrategy, mask, I::CartesianIndex{N},
) where {N, T<:AbstractFloat, G}
    _filters_target(strategy, @inbounds mask[I]) || return zero(T)
    rule = fp.weights
    Ti = Tuple(I)
    t = LinearIndices(FlowGeometries.Grids.size_tuple(grid))[I]
    zerofill = strategy isa ZeroFill
    ws, wn = _nd_foldl((zero(T), zero(T)), grid, Ti, fp.rad, fp.topology) do acc, J, d
        s, n = acc
        active = mask[J...]
        wk = last(_cell_weight(rule, grid, t, Ti, J, d))
        if zerofill
            return (active ? s + wk * field[J...] : s, n + wk)
        else
            active || return acc
            return (s + wk * field[J...], n + wk)
        end
    end
    zerofill && (wn = _add_exterior(wn, fp.exterior, I))
    return _normalized(ws, wn)
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
    if fp.cache !== nothing
        lin = LinearIndices(dims)
        cache = fp.cache
        @inbounds for I in CartesianIndices(out)
            out[I] = _footprint_nd_point_cached(field, cache, fp.exterior, strategy, mask, lin, I)
        end
    else
        @inbounds for I in CartesianIndices(out)
            out[I] = _footprint_nd_point_streaming(field, grid, fp, strategy, mask, I)
        end
    end
    return out
end

"""
    NodeFilterPlan{T}

Real-space filter footprint for a node set: per-node CSR neighbour blocks (`nbrs`, `w`, offsets `ptr`)
with their geometric weights `w = kernel_weight(d) · control volume`, found once at plan time.

Each node's block includes the node itself, which `Connectivity.neighbors_within` leaves out.
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
