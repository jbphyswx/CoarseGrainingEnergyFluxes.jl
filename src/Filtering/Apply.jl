# ---------------------------------------------------------------------------
# Batched apply: derive each target point's neighbours and weights once and apply them to every field
# in the batch. `compute_Π!` makes 6-9 `filter_apply!` calls per scale against one plan, so this is
# where the redundancy is. Orthogonal to caching, which changes only how the weights are derived.
# ---------------------------------------------------------------------------

# Homogeneous-`NTuple` batches (compile-time-known K): `MVector` accumulators are stack-allocated,
# not heap — genuinely zero-allocation. `Vector` batches (runtime-known K, e.g. a variable number of
# quadratic-product terms): a small `Vector{T}` accumulator, allocated ONCE per row/point-loop (not
# per candidate or per target point), so its cost is O(Ny) or O(N) total, not part of the O(N·M) hot
# path. Both share `eachindex`/`fill!`/indexing, so the rest of the per-row/per-point logic below is
# written once, generic over which container `outs`/`fields` actually is.
@inline _batch_zeros(::NTuple{K,<:Any}, ::Type{T}) where {K,T<:AbstractFloat} = SA.MVector{K,T}(ntuple(_ -> zero(T), K))
@inline _batch_zeros(v::AbstractVector, ::Type{T}) where {T<:AbstractFloat} = zeros(T, length(v))

_apply_serial_batch!(outs, fields, grid, fp::FilterFootprint, strategy) =
    apply_footprint_batch!(outs, fields, grid, fp, strategy, FlowGeometries.Grids.isperiodic(grid, 1), FlowGeometries.Grids.isperiodic(grid, 2))
_apply_serial_batch!(outs, fields, grid, fp::ScatteredFilterPlan, strategy) =
    apply_footprint_batch!(outs, fields, grid, fp, strategy, FlowGeometries.Grids.isperiodic(grid, 1), FlowGeometries.Grids.isperiodic(grid, 2))
_apply_serial_batch!(outs, fields, grid, fp::FilterFootprintND, strategy) =
    apply_footprint_nd_batch!(outs, fields, grid, fp, strategy)
_apply_serial_batch!(outs, fields, grid, fp::NDScatteredFilterPlan, strategy) =
    apply_footprint_nd_batch!(outs, fields, grid, fp, strategy)
_apply_serial_batch!(outs, fields, grid, fp::PrefixSumTopHatPlan, strategy) =
    apply_prefixsum_tophat_batch!(outs, fields, grid, fp, strategy)

# Every other footprint applies field by field. None carries a per-point neighbour derivation for a
# batch to share: the separable Gaussians hold precomputed 1-D weight tables, `NodeFilterPlan` stores
# its adjacency outright, and the FFTW extension's padded- and zonal-FFT engines filter a whole field
# per transform.
#
# `PrefixSumTopHatPlan` is not among them. Its support intervals are O(1) amortized per point per band,
# but the sweep TOTAL is O(N·dj_lim) — the engine's dominant cost — while its genuinely per-field part,
# the numerator scan, is a small fraction of that. Whether sharing the sweep pays depends on the axis;
# see `_prefixsum_batch_fuses`.
function _apply_serial_batch!(outs, fields, grid, fp, strategy)
    for k in eachindex(outs)
        _apply_serial!(outs[k], fields[k], grid, fp, strategy)
    end
    return outs
end

function apply_footprint_batch!(
    outs, fields, grid::FlowGeometries.Grids.AbstractGrid, fp::FilterFootprint{T}, strategy::AbstractMaskStrategy,
    periodic_x::Bool, periodic_y::Bool,
) where {T<:AbstractFloat}
    _, Ny = FlowGeometries.Grids.size_tuple(grid)
    _banded_check_strategy(fp, strategy)
    if fp.masked
        _banded_inputs!(fp.scratch, length(fields))
        for m in eachindex(fields)
            _banded_fill_source!(fp, @inbounds(fields[m]), grid, m)
        end
    end
    for j in 1:Ny
        apply_footprint_row_batch!(outs, fields, grid, fp, strategy, periodic_x, periodic_y, j)
    end
    return outs
end

"""
    apply_footprint_row_batch!(outs, fields, grid, fp, strategy, periodic_x, periodic_y, j) -> outs

Batched banded row apply: one contiguous axpy per field per tap, every field sharing the plan's
`invden`.

The split is exactly along what depends on the field. The numerator does, so it is computed once per
field. The window mass does not — it is a function of the geometry, the mask and the strategy — so it
is precomputed per scale and read here, which is also what keeps each field's inner loop a vectorizable
axpy rather than a per-point reduction.
"""
function apply_footprint_row_batch!(
    outs, fields, grid::FlowGeometries.Grids.StructuredGrid, fp::FilterFootprint{T}, strategy::AbstractMaskStrategy,
    periodic_x::Bool, periodic_y::Bool, j::Integer,
) where {T<:AbstractFloat}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    b = _banded_band(fp.nbands, j)
    lo = fp.ptr[b]
    hi = fp.ptr[b + 1] - 1
    invden = fp.invden
    for m in eachindex(outs)
        oc = view(@inbounds(outs[m]), :, j)
        @inbounds @simd for i in 1:Nx
            oc[i] = zero(T)
        end
        _banded_row_accumulate!(
            oc, _banded_source(fp, @inbounds(fields[m]), m), fp.di, fp.dj, fp.w, lo, hi,
            fp.periodic_x, fp.periodic_y, Nx, Ny, j,
        )
        @inbounds @simd for i in 1:Nx
            oc[i] *= invden[i, j]
        end
    end
    return outs
end

function apply_footprint_batch!(
    outs, fields, grid::FlowGeometries.Grids.AbstractGrid, fp::ScatteredFilterPlan{T}, strategy::AbstractMaskStrategy,
    periodic_x::Bool, periodic_y::Bool,
) where {T<:AbstractFloat}
    _, Ny = FlowGeometries.Grids.size_tuple(grid)
    for out in outs
        fill!(out, zero(T))
    end
    for j in 1:Ny
        apply_footprint_row_batch!(outs, fields, grid, fp, strategy, periodic_x, periodic_y, j)
    end
    return outs
end

function apply_footprint_row_batch!(
    outs, fields, grid::FlowGeometries.Grids.AbstractGrid, fp::ScatteredFilterPlan{T}, strategy::AbstractMaskStrategy,
    periodic_x::Bool, periodic_y::Bool, j::Integer,
) where {T<:AbstractFloat}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    acc_ws = _batch_zeros(outs, T)
    acc_wn = _batch_zeros(outs, T)
    cache = fp.cache
    if cache !== nothing
        for i in 1:Nx
            FlowGeometries.Grids.isactive(grid, i, j) || continue
            t = i + (j - 1) * Nx
            lo = cache.ptr[t]
            hi = cache.ptr[t+1] - 1
            fill!(acc_ws, zero(T))
            fill!(acc_wn, zero(T))
            @inbounds for k in lo:hi
                ii = cache.ii[k]
                jj = cache.jj[k]
                active = FlowGeometries.Grids.isactive(grid, ii, jj)
                w = cache.w[k]
                if strategy isa ZeroFill
                    for m in eachindex(fields)
                        acc_wn[m] += w
                    end
                    if active
                        for m in eachindex(fields)
                            acc_ws[m] += w * fields[m][ii, jj]
                        end
                    end
                elseif active
                    for m in eachindex(fields)
                        acc_wn[m] += w
                        acc_ws[m] += w * fields[m][ii, jj]
                    end
                end
            end
            for m in eachindex(outs)
                outs[m][i, j] = acc_wn[m] > T(1e-15) ? acc_ws[m] / acc_wn[m] : zero(T)
            end
        end
    else
        kernel = fp.kernel
        scale = fp.scale
        di_lim, dj_lim = fp.di_lim, fp.dj_lim
        fp_periodic_x, fp_periodic_y = fp.periodic_x, fp.periodic_y
        x_period, y_period = fp.x_period, fp.y_period
        is_cartesian = fp.is_cartesian
        rad = fp.rad
        sc = FlowGeometries.Connectivity.ball_scratch()   # per row; see the single-field row apply
        for i in 1:Nx
            FlowGeometries.Grids.isactive(grid, i, j) || continue
            target = FlowGeometries.Grids.coords(SA.SVector, grid, i, j)
            fill!(acc_ws, zero(T))
            fill!(acc_wn, zero(T))
            _scattered_foldl(
                nothing, grid, target, i, j, Nx, Ny, di_lim, dj_lim,
                fp_periodic_x, fp_periodic_y, x_period, y_period, is_cartesian, rad, fp.topology, sc,
            ) do _, iin, jjn, d
                active = FlowGeometries.Grids.isactive(grid, iin, jjn)
                w = Kernels.kernel_weight(kernel, d, scale) * FlowGeometries.Grids.area(grid, iin, jjn)
                if strategy isa ZeroFill
                    for m in eachindex(fields)
                        acc_wn[m] += w
                    end
                    if active
                        for m in eachindex(fields)
                            acc_ws[m] += w * fields[m][iin, jjn]
                        end
                    end
                elseif active
                    for m in eachindex(fields)
                        acc_wn[m] += w
                        acc_ws[m] += w * fields[m][iin, jjn]
                    end
                end
                nothing
            end
            for m in eachindex(outs)
                outs[m][i, j] = acc_wn[m] > T(1e-15) ? acc_ws[m] / acc_wn[m] : zero(T)
            end
        end
    end
    return outs
end

"""
    apply_footprint_nd_batch!(outs, fields, grid, fp, strategy) -> outs

Batched point-indexed apply over the whole grid. The batch shares each point's neighbour enumeration
across all `K` fields, so the enumeration is paid once rather than `K` times.
"""
function apply_footprint_nd_batch!(
    outs, fields, grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, fp::Union{FilterFootprintND{N,T}, NDScatteredFilterPlan{N,T}}, strategy::AbstractMaskStrategy,
) where {N, T<:AbstractFloat, G}
    for out in outs
        fill!(out, zero(T))
    end
    return apply_footprint_nd_batch_over!(
        outs, fields, grid, fp, strategy, CartesianIndices(FlowGeometries.Grids.size_tuple(grid)),
    )
end

"""
    apply_footprint_nd_batch_over!(outs, fields, grid, fp, strategy, indices) -> outs

The batched apply restricted to `indices`. Each output point depends only on its own neighbourhood,
so a parallel backend can hand disjoint index blocks to different tasks and get the serial answer.
`outs` must already be zeroed — the caller owns that, since a block only writes its own points.
"""
function apply_footprint_nd_batch_over!(
    outs, fields, grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, fp::FilterFootprintND{N,T}, strategy::AbstractMaskStrategy,
    indices,
) where {N, T<:AbstractFloat, G}
    dims = FlowGeometries.Grids.size_tuple(grid)
    periodic = FlowGeometries.Grids.periodic_flags(grid)
    mask = FlowGeometries.Grids.mask(grid)
    acc_ws = _batch_zeros(outs, T)
    acc_wn = _batch_zeros(outs, T)
    @inbounds for I in indices
        mask[I] || continue
        Ti = Tuple(I)
        fill!(acc_ws, zero(T))
        fill!(acc_wn, zero(T))
        for k in eachindex(fp.offsets)
            J, valid = _shift_index(Ti, fp.offsets[k], dims, periodic)
            valid || continue
            active = mask[J...]
            wk = fp.w[k]
            if strategy isa ZeroFill
                for m in eachindex(fields)
                    acc_wn[m] += wk
                end
                if active
                    for m in eachindex(fields)
                        acc_ws[m] += wk * fields[m][J...]
                    end
                end
            elseif active
                for m in eachindex(fields)
                    acc_wn[m] += wk
                    acc_ws[m] += wk * fields[m][J...]
                end
            end
        end
        for m in eachindex(outs)
            outs[m][I] = acc_wn[m] > T(1e-15) ? acc_ws[m] / acc_wn[m] : zero(T)
        end
    end
    return outs
end

# One streamed point of the ND batch. An `NTuple` batch has its width in the type, so it folds
# immutable accumulators through `_nd_foldl`'s `acc` and allocates nothing; unknown width keeps the
# mutable buffers. Both enumerate through `_nd_foldl`, so both match the cache builder.
@inline function _nd_stream_point!(
    outs::NTuple{K,<:AbstractArray}, fields::NTuple{K,<:AbstractArray},
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, fp::NDScatteredFilterPlan{N,T},
    strategy::AbstractMaskStrategy, mask, I::CartesianIndex{N}, kernel, scale,
    _acc_ws, _acc_wn,
) where {K, N, T<:AbstractFloat, G}
    z = zero(SA.SVector{K,T})
    ws, wn = _nd_foldl((z, z), grid, Tuple(I), fp.rad, fp.topology) do a, J, d
        aws, awn = a
        active = mask[J...]
        wk = Kernels.kernel_weight(kernel, d, scale) * FlowGeometries.Grids.area(grid, J...)
        if strategy isa ZeroFill
            awn = awn .+ wk
            active && (aws = aws .+ wk .* SA.SVector{K,T}(ntuple(m -> fields[m][J...], Val(K))))
        elseif active
            awn = awn .+ wk
            aws = aws .+ wk .* SA.SVector{K,T}(ntuple(m -> fields[m][J...], Val(K)))
        end
        (aws, awn)
    end
    @inbounds for m in 1:K
        outs[m][I] = wn[m] > T(1e-15) ? ws[m] / wn[m] : zero(T)
    end
    return nothing
end

@inline function _nd_stream_point!(
    outs, fields, grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, fp::NDScatteredFilterPlan{N,T},
    strategy::AbstractMaskStrategy, mask, I::CartesianIndex{N}, kernel, scale,
    acc_ws, acc_wn,
) where {N, T<:AbstractFloat, G}
    fill!(acc_ws, zero(T))
    fill!(acc_wn, zero(T))
    _nd_foldl(nothing, grid, Tuple(I), fp.rad, fp.topology) do _, J, d
        active = mask[J...]
        wk = Kernels.kernel_weight(kernel, d, scale) * FlowGeometries.Grids.area(grid, J...)
        if strategy isa ZeroFill
            for m in eachindex(fields)
                acc_wn[m] += wk
            end
            if active
                for m in eachindex(fields)
                    acc_ws[m] += wk * fields[m][J...]
                end
            end
        elseif active
            for m in eachindex(fields)
                acc_wn[m] += wk
                acc_ws[m] += wk * fields[m][J...]
            end
        end
        nothing
    end
    @inbounds for m in eachindex(outs)
        outs[m][I] = acc_wn[m] > T(1e-15) ? acc_ws[m] / acc_wn[m] : zero(T)
    end
    return nothing
end

function apply_footprint_nd_batch_over!(
    outs, fields, grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, fp::NDScatteredFilterPlan{N,T}, strategy::AbstractMaskStrategy,
    indices,
) where {N, T<:AbstractFloat, G}
    dims = FlowGeometries.Grids.size_tuple(grid)
    mask = FlowGeometries.Grids.mask(grid)
    acc_ws = _batch_zeros(outs, T)
    acc_wn = _batch_zeros(outs, T)
    if fp.cache !== nothing
        cache = fp.cache
        lin = LinearIndices(dims)
        @inbounds for I in indices
            mask[I] || continue
            t = lin[I]
            lo = cache.ptr[t]
            hi = cache.ptr[t+1] - 1
            fill!(acc_ws, zero(T))
            fill!(acc_wn, zero(T))
            for k in lo:hi
                J = cache.nbrs[k]
                active = mask[J...]
                wk = cache.w[k]
                if strategy isa ZeroFill
                    for m in eachindex(fields)
                        acc_wn[m] += wk
                    end
                    if active
                        for m in eachindex(fields)
                            acc_ws[m] += wk * fields[m][J...]
                        end
                    end
                elseif active
                    for m in eachindex(fields)
                        acc_wn[m] += wk
                        acc_ws[m] += wk * fields[m][J...]
                    end
                end
            end
            for m in eachindex(outs)
                outs[m][I] = acc_wn[m] > T(1e-15) ? acc_ws[m] / acc_wn[m] : zero(T)
            end
        end
    else
        kernel, scale = fp.kernel, fp.scale
        @inbounds for I in indices
            mask[I] || continue
            _nd_stream_point!(outs, fields, grid, fp, strategy, mask, I, kernel, scale, acc_ws, acc_wn)
        end
    end
    return outs
end
