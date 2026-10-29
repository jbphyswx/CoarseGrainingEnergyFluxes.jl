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
    PrefixSum3DScratch{T,A,VV} <: AbstractFilterScratch

The 3-D top-hat engine's per-apply buffers: the axis-1 [`CompensatedScan`](@ref) of `mask · field`,
sized `(Nx+1, Ny, Nz)`, and the cells' values `mask · field` (`cell_values`, `Nx × Ny × Nz`), which the
cells the ball's boundary cuts read; `nothing` under [`Kernels.CellCentre`](@ref). Refilled at the start
of every apply.
"""
struct PrefixSum3DScratch{
    T<:AbstractFloat, A<:AbstractArray{T,3}, VV<:Union{Nothing,AbstractArray{T,3}},
} <: AbstractFilterScratch
    prefix::CompensatedScan{T,A}
    cell_values::VV
end

"""
    PrefixEdgeCells3D{VO,VT}

The cells the support of a [`Kernels.CellOverlap`](@ref) top-hat cuts on a uniform 3-D grid: tap `s` at offset
`offsets[s]` with weight `w[s]`, the cell's fraction inside the support, applied to the cell's scanned value.
"""
struct PrefixEdgeCells3D{VO<:AbstractVector{NTuple{3,Int}}, VT<:AbstractVector}
    offsets::VO
    w::VT
end

"""
    PrefixSumTopHat3DPlan{T,SC,A,WT,MS,RM}

Exact `O(N·w_y·w_z)` top-hat footprint for a uniform 3-D Cartesian `StructuredGrid`.

`wcell` is the `(2·dj_lim+1) × (2·dk_lim+1)` table of the ball's axis-1 half-width at each offset
`(dj, dk)`, `-1` where that slice is empty; on a uniform grid it is the same at every point. Under
[`Kernels.CellOverlap`](@ref) it is the half-width of the cells wholly inside, and `edge` the
[`PrefixEdgeCells3D`](@ref) of those the boundary cuts; `nothing` under [`Kernels.CellCentre`](@ref).

`invden` is the reciprocal window mass, built once per scale under `strategy`.
"""
struct PrefixSumTopHat3DPlan{
    T<:AbstractFloat, SC<:PrefixSum3DScratch{T}, A<:AbstractArray{T,3},
    WT<:AbstractMatrix{Int}, MS<:AbstractMaskStrategy, RM<:Union{Nothing,PrefixEdgeCells3D},
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
    edge::RM
end

function _prefixsum3d_scratch(grid::FlowGeometries.Grids.StructuredGrid{T,G,3}, kernel::Kernels.TopHatKernel) where {T,G}
    Nx, Ny, Nz = FlowGeometries.Grids.size_tuple(grid)
    return PrefixSum3DScratch(CompensatedScan(T, Nx + 1, Ny, Nz), _cell_values_3d(T, kernel, Nx, Ny, Nz))
end

_cell_values_3d(::Type, ::Kernels.TopHatKernel{Kernels.CellCentre}, _, _, _) = nothing
_cell_values_3d(::Type{T}, ::Kernels.TopHatKernel{Kernels.CellOverlap}, Nx, Ny, Nz) where {T} = Array{T,3}(undef, Nx, Ny, Nz)

"""
    _prefixsum3d_fill_plane!(sc, src, grid, masked, j, k) -> nothing

Compensated scan along axis 1 of `src` for one `(j, k)` plane into `sc.prefix`, with entry 1 the empty
prefix, and the plane's values into `sc.cell_values` where it holds them. Writes only that plane, so
planes may be filled concurrently.
"""
@inline function _prefixsum3d_fill_plane!(
    sc::PrefixSum3DScratch{T}, src::AbstractArray, grid::FlowGeometries.Grids.StructuredGrid{T,G,3},
    masked::Bool, j::Integer, k::Integer,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    Nx = FlowGeometries.Grids.size_tuple(grid)[1]
    Ph, Pl = sc.prefix.hi, sc.prefix.lo
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
    _store_plane_values!(sc.cell_values, src, grid, masked, j, k)
    return nothing
end

@inline _store_plane_values!(::Nothing, src, grid, masked, j, k) = nothing
@inline function _store_plane_values!(V::AbstractArray{T,3}, src, grid, masked::Bool, j, k) where {T}
    @inbounds for i in axes(V, 1)
        V[i, j, k] = (masked && !FlowGeometries.Grids.isactive(grid, i, j, k)) ? zero(T) : T(src[i, j, k])
    end
    return nothing
end

function _prefixsum3d_fill!(
    sc::PrefixSum3DScratch{T}, src::AbstractArray, grid::FlowGeometries.Grids.StructuredGrid{T,G,3},
    masked::Bool,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    _, Ny, Nz = FlowGeometries.Grids.size_tuple(grid)
    for k in 1:Nz, j in 1:Ny
        _prefixsum3d_fill_plane!(sc, src, grid, masked, j, k)
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
            if periodic[1] && 2 * w + 1 < Nx
                # A window shorter than the period wraps only for the `w` cells at either end.
                for i in 1:w
                    oc[i] += _scan_diff(Ph, Pl, Nx + 1, Nx + i - w, jj, kk) + _scan_diff(Ph, Pl, i + w + 1, 1, jj, kk)
                end
                @simd for i in (w + 1):(Nx - w)
                    oc[i] += _scan_diff(Ph, Pl, i + w + 1, i - w, jj, kk)
                end
                for i in (Nx - w + 1):Nx
                    oc[i] += _scan_diff(Ph, Pl, Nx + 1, i - w, jj, kk) + _scan_diff(Ph, Pl, i + w - Nx + 1, 1, jj, kk)
                end
            elseif periodic[1]
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
    kernel::Kernels.TopHatKernel{Kernels.CellCentre},
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
    periodic = ntuple(d -> FlowGeometries.Grids.isperiodic(grid, d), Val(3))
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

    sc = scratch === nothing ? _prefixsum3d_scratch(grid, kernel) : scratch
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
    _prefixsum3d_fill!(sc, mass, grid, false)
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
    return PrefixSumTopHat3DPlan(sc, rad, dj_lim, dk_lim, periodic, masked, bound, mask_strategy, wcell, den, nothing)
end

"""
    _build_prefixsum_tophat_3d(grid, kernel::TopHatKernel{CellOverlap}, scale; ...) -> PrefixSumTopHat3DPlan

The 3-D prefix-sum top-hat with each cell weighted by its fraction inside the ball: per `(dj, dk)` the
half-width of the cells wholly inside about the target, and a tap for every cell the ball's boundary cuts.
`ZeroFill` divides by the ball's measure where a window leaves the grid, and by the summed weights elsewhere.
"""
function _build_prefixsum_tophat_3d(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,3},
    kernel::Kernels.TopHatKernel{Kernels.CellOverlap},
    scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    scratch::Union{Nothing,PrefixSum3DScratch{T,<:Any,<:AbstractArray}} = nothing,
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    Nx, Ny, Nz = FlowGeometries.Grids.size_tuple(grid)
    rad = Kernels.kernel_radius(kernel, scale)
    Δ = ntuple(d -> abs(T(step(FlowGeometries.Grids.coordinates(grid, d)))), Val(3))
    ov = _overlap_rule(FlowGeometries.Grids.grid_geometry(grid), rad)
    periodic = ntuple(d -> FlowGeometries.Grids.isperiodic(grid, d), Val(3))
    reach(d, n) = Δ[d] > 0 ? (periodic[d] ? ceil(Int, rad / Δ[d] + T(1) / 2) : min(n - 1, ceil(Int, rad / Δ[d] + T(1) / 2))) : 0
    dj_lim, dk_lim = reach(2, Ny), reach(3, Nz)
    di_max = periodic[1] ? typemax(Int) - 1 : Nx - 1

    wcell = fill(-1, 2 * dj_lim + 1, 2 * dk_lim + 1)
    offsets = NTuple{3,Int}[]
    w = T[]
    fr = _overlap_float(T)[]
    for dk in (-dk_lim):dk_lim, dj in (-dj_lim):dj_lim
        empty!(fr)
        for di in 0:di_max
            f = _lattice_fraction(ov, CartesianIndex(di, dj, dk), Δ)
            iszero(f) && break
            push!(fr, f)
        end
        wf = 0
        while wf < length(fr) && isone(fr[wf + 1])
            wf += 1
        end
        wcell[dj + dj_lim + 1, dk + dk_lim + 1] = wf - 1
        for di in wf:(length(fr) - 1)
            push!(offsets, (di, dj, dk)); push!(w, T(fr[di + 1]))
            di > 0 && (push!(offsets, (-di, dj, dk)); push!(w, T(fr[di + 1])))
        end
    end
    edge = PrefixEdgeCells3D(offsets, w)

    sc = scratch === nothing ? _prefixsum3d_scratch(grid, kernel) : scratch
    masked = !all(FlowGeometries.Grids.mask(grid))
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
    _prefixsum3d_fill!(sc, mass, grid, false)
    den = zeros(T, Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny
        oc = view(den, :, j, k)
        _prefixsum3d_plane!(oc, P, wcell, dj_lim, dk_lim, periodic, Nx, Ny, Nz, j, k)
        _prefixsum3d_edge!(oc, sc.cell_values, edge, periodic, Nx, Ny, Nz, j, k)
    end
    # The scan sums cells, so the support's measure enters in cells.
    zerofill && _full_support(grid, kernel, scale) &&
        fill!(den, _support_measure(grid, kernel, scale, 1) / FlowGeometries.Grids.area(grid, 1, 1, 1))
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        keep = zerofill || FlowGeometries.Grids.isactive(grid, i, j, k)
        den[i, j, k] = keep ? _inv_mass(den[i, j, k]) : zero(T)
    end
    bound = masked || _reaches_exterior(grid, kernel, scale)
    return PrefixSumTopHat3DPlan(sc, rad, dj_lim, dk_lim, periodic, masked, bound, mask_strategy, wcell, den, edge)
end

# Add the edge taps' contributions to output plane `(j, k)` from the cells' values `V`. Nothing under
# `CellCentre`.
@inline _prefixsum3d_edge!(oc, V, ::Nothing, periodic, Nx, Ny, Nz, j, k) = oc

@inline function _prefixsum3d_edge!(
    oc::AbstractVector{T}, V::AbstractArray{T,3}, edge::PrefixEdgeCells3D, periodic::NTuple{3,Bool},
    Nx::Int, Ny::Int, Nz::Int, j::Int, k::Int,
) where {T<:AbstractFloat}
    @inbounds for s in eachindex(edge.offsets)
        di, dj, dk = edge.offsets[s]
        jj, kk = j + dj, k + dk
        if jj < 1 || jj > Ny
            periodic[2] || continue
            jj = mod1(jj, Ny)
        end
        if kk < 1 || kk > Nz
            periodic[3] || continue
            kk = mod1(kk, Nz)
        end
        ws = T(edge.w[s])
        if !periodic[1]
            @simd for i in max(1, 1 - di):min(Nx, Nx - di)
                oc[i] += ws * V[i + di, jj, kk]
            end
        else
            r = mod(di, Nx)
            @simd for i in 1:(Nx - r)
                oc[i] += ws * V[i + r, jj, kk]
            end
            @simd for i in (Nx - r + 1):Nx
                oc[i] += ws * V[i + r - Nx, jj, kk]
            end
        end
    end
    return oc
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
    _prefixsum3d_edge!(oc, fp.scratch.cell_values, fp.edge, fp.periodic, Nx, Ny, Nz, j, k)
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
    _prefixsum3d_fill!(fp.scratch, field, grid, fp.masked)
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

# More specific in the kernel than the generic 3-D uniform-axis method, so a top-hat on a uniform
# Cartesian volume dispatches here.
_structured_footprint(
    ::NTuple{3,FlowGeometries.Axes.UniformSpacing}, grid::FlowGeometries.Grids.StructuredGrid{T,G,3},
    kernel::Kernels.TopHatKernel, scale::T; kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}} =
    _build_prefixsum_tophat_3d(grid, kernel, scale; kwargs...)


