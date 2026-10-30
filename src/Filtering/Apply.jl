# ---------------------------------------------------------------------------
# Batched apply: each target point's neighbours and weights serve every field of the batch
# ---------------------------------------------------------------------------

# One field through each engine's own apply.
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

# A footprint whose denominator is fixed at build time accepts only the strategy it was built for where
# the two differ; one that accumulates its denominator per point takes either. Every backend's entry
# point calls this before its first row or point.
_check_strategy(fp::FilterFootprint, s::AbstractMaskStrategy) = _banded_check_strategy(fp, s)
_check_strategy(fp::Union{SeparableFootprint,SeparableFootprintND}, s::AbstractMaskStrategy) =
    _separable_check_strategy(fp, s)
_check_strategy(fp::PrefixSumTopHatPlan, s::AbstractMaskStrategy) = _prefixsum_check_strategy(fp, s)
_check_strategy(fp::PrefixSumTopHat3DPlan, s::AbstractMaskStrategy) = _prefixsum3d_check_strategy(fp, s)
_check_strategy(fp::FilterFootprintND, s::AbstractMaskStrategy) = _nd_check_strategy(fp, s)
_check_strategy(_, ::AbstractMaskStrategy) = nothing

# Per-field accumulators: a stack `MVector` for an `NTuple` batch, a `Vector` for an `AbstractVector`
# batch. Called once per row or index block, outside the per-point loop.
@inline _batch_zeros(::NTuple{K,<:Any}, ::Type{T}) where {K,T<:AbstractFloat} = SA.MVector{K,T}(ntuple(_ -> zero(T), Val(K)))
@inline _batch_zeros(v::AbstractVector, ::Type{T}) where {T<:AbstractFloat} = zeros(T, length(v))

_apply_serial_batch!(outs, fields, grid, fp::FilterFootprint, strategy) =
    apply_footprint_batch!(outs, fields, grid, fp, strategy, FlowGeometries.Grids.isperiodic(grid, 1), FlowGeometries.Grids.isperiodic(grid, 2))
_apply_serial_batch!(outs, fields, grid, fp::ScatteredFilterPlan, strategy) =
    apply_footprint_batch!(outs, fields, grid, fp, strategy, FlowGeometries.Grids.isperiodic(grid, 1), FlowGeometries.Grids.isperiodic(grid, 2))
_apply_serial_batch!(outs, fields, grid, fp::NDScatteredFilterPlan, strategy) =
    apply_footprint_nd_batch!(outs, fields, grid, fp, strategy)
_apply_serial_batch!(outs, fields, grid, fp::PrefixSumTopHatPlan, strategy) =
    apply_prefixsum_tophat_batch!(outs, fields, grid, fp, strategy)

# Every other footprint applies field by field: none derives a per-point neighbourhood for a batch to
# share.
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

Row `j` of the batched banded apply: each field's numerator by one contiguous axpy per tap, scaled by
the plan's precomputed inverse window mass `invden`, which depends on the geometry, the mask and the
strategy alone.
"""
function apply_footprint_row_batch!(
    outs, fields, grid::FlowGeometries.Grids.StructuredGrid, fp::FilterFootprint{T}, strategy::AbstractMaskStrategy,
    periodic_x::Bool, periodic_y::Bool, j::Integer, _sc = nothing,
) where {T<:AbstractFloat}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    b = _banded_band(fp.nbands, j)
    invden = fp.invden
    for m in eachindex(outs)
        oc = view(@inbounds(outs[m]), :, j)
        @inbounds @simd for i in 1:Nx
            oc[i] = zero(T)
        end
        _banded_row_accumulate!(
            oc, _banded_source(fp, @inbounds(fields[m]), m), fp.taps, b,
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
    sc = row_scratch(fp)
    for j in 1:Ny
        apply_footprint_row_batch!(outs, fields, grid, fp, strategy, periodic_x, periodic_y, j, sc)
    end
    return outs
end

# The fields share each target's window mass, summed once.
function apply_footprint_row_batch!(
    outs, fields, grid::FlowGeometries.Grids.AbstractGrid, fp::ScatteredFilterPlan{T}, strategy::AbstractMaskStrategy,
    periodic_x::Bool, periodic_y::Bool, j::Integer, sc = row_scratch(fp),
) where {T<:AbstractFloat}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    cache = fp.cache
    zerofill = strategy isa ZeroFill
    if cache !== nothing
        acc_ws = _batch_zeros(outs, T)
        mask = FlowGeometries.Grids.mask(grid)
        for i in 1:Nx
            _filters_target(strategy, FlowGeometries.Grids.isactive(grid, i, j)) || continue
            t = i + (j - 1) * Nx
            fill!(acc_ws, zero(T))
            wn = zero(T)
            k = cache.wptr[t]
            @inbounds for r in cache.rptr[t]:(cache.rptr[t + 1] - 1)
                c0 = Int(cache.start[r])
                for c in c0:(c0 + Int(cache.len[r]) - 1)
                    active = _cell_active(mask, c)
                    w = cache.w[k]
                    k += 1
                    (zerofill || active) || continue
                    wn += w
                    if active
                        for m in eachindex(fields)
                            acc_ws[m] += w * fields[m][c]
                        end
                    end
                end
            end
            zerofill && (wn = _add_exterior(wn, fp.exterior, i, j))
            for m in eachindex(outs)
                outs[m][i, j] = _normalized(acc_ws[m], wn)
            end
        end
    else
        _scattered_stream_row_batch!(outs, fields, grid, fp, strategy, Int(j), sc)
    end
    return outs
end

# One row of the streamed batch; the fields share one window mass. An `NTuple` batch folds its sums
# through the query as values and allocates nothing; a batch of unknown width sums into one buffer per row.
function _scattered_stream_row_batch!(
    outs::NTuple{K,<:AbstractArray}, fields::NTuple{K,<:AbstractArray}, grid, fp::ScatteredFilterPlan{T},
    strategy::AbstractMaskStrategy, j::Int, sc,
) where {K, T<:AbstractFloat}
    Nx = FlowGeometries.Grids.size_tuple(grid)[1]
    rule = fp.weights
    zerofill = strategy isa ZeroFill
    for i in 1:Nx
        _filters_target(strategy, FlowGeometries.Grids.isactive(grid, i, j)) || continue
        t = i + (j - 1) * Nx
        ws, wn = _rule_fold((zero(SA.SVector{K,T}), zero(T)), rule, grid, t, (i, j), fp.rad, fp.topology, sc) do acc, J, w
            aws, awn = acc
            active = FlowGeometries.Grids.isactive(grid, J[1], J[2])
            (zerofill || active) || return acc
            active &&(aws = aws .+ w .* SA.SVector{K,T}(ntuple(m -> @inbounds(fields[m][J[1], J[2]]), Val(K))))
            return (aws, awn + w)
        end
        zerofill && (wn = _add_exterior(wn, fp.exterior, i, j))
        @inbounds for m in 1:K
            outs[m][i, j] = _normalized(ws[m], wn)
        end
    end
    return outs
end

function _scattered_stream_row_batch!(
    outs, fields, grid, fp::ScatteredFilterPlan{T}, strategy::AbstractMaskStrategy, j::Int, sc,
) where {T<:AbstractFloat}
    Nx = FlowGeometries.Grids.size_tuple(grid)[1]
    rule = fp.weights
    zerofill = strategy isa ZeroFill
    acc_ws = zeros(T, length(fields))
    for i in 1:Nx
        _filters_target(strategy, FlowGeometries.Grids.isactive(grid, i, j)) || continue
        t = i + (j - 1) * Nx
        fill!(acc_ws, zero(T))
        wn = _rule_fold(zero(T), rule, grid, t, (i, j), fp.rad, fp.topology, sc) do awn, J, w
            active = FlowGeometries.Grids.isactive(grid, J[1], J[2])
            (zerofill || active) || return awn
            if active
                for m in eachindex(fields)
                    @inbounds acc_ws[m] += w * fields[m][J[1], J[2]]
                end
            end
            return awn + w
        end
        zerofill && (wn = _add_exterior(wn, fp.exterior, i, j))
        @inbounds for m in eachindex(outs)
            outs[m][i, j] = _normalized(acc_ws[m], wn)
        end
    end
    return outs
end

"""
    apply_footprint_nd_batch!(outs, fields, grid, fp, strategy) -> outs

Batched point-indexed apply over the whole grid: each point's neighbours are enumerated once for all
fields.
"""
function apply_footprint_nd_batch!(
    outs, fields, grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, fp::NDScatteredFilterPlan{N,T}, strategy::AbstractMaskStrategy,
) where {N, T<:AbstractFloat, G}
    for out in outs
        fill!(out, zero(T))
    end
    return apply_footprint_nd_batch_over!(
        outs, fields, grid, fp, strategy, CartesianIndices(FlowGeometries.Grids.size_tuple(grid)),
    )
end

# One streamed point of the ND batch; the fields share one window mass. An `NTuple` batch folds immutable
# accumulators and allocates nothing; a batch of unknown width sums into the mutable buffer. Both
# enumerate through `_nd_rule_fold`, as the cache builder does.
@inline function _nd_stream_point!(
    outs::NTuple{K,<:AbstractArray}, fields::NTuple{K,<:AbstractArray},
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, fp::NDScatteredFilterPlan{N,T},
    strategy::AbstractMaskStrategy, mask, I::CartesianIndex{N}, t::Int, _acc_ws,
) where {K, N, T<:AbstractFloat, G}
    rule = fp.weights
    Ti = Tuple(I)
    zerofill = strategy isa ZeroFill
    ws, wn = _nd_rule_fold((zero(SA.SVector{K,T}), zero(T)), rule, grid, t, Ti, fp.rad, fp.topology) do a, J, wk
        aws, awn = a
        active = mask[J...]
        (zerofill || active) || return a
        active &&(aws = aws .+ wk .* SA.SVector{K,T}(ntuple(m -> fields[m][J...], Val(K))))
        (aws, awn + wk)
    end
    zerofill && (wn = _add_exterior(wn, fp.exterior, I))
    @inbounds for m in 1:K
        outs[m][I] = _normalized(ws[m], wn)
    end
    return nothing
end

@inline function _nd_stream_point!(
    outs, fields, grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, fp::NDScatteredFilterPlan{N,T},
    strategy::AbstractMaskStrategy, mask, I::CartesianIndex{N}, t::Int, acc_ws,
) where {N, T<:AbstractFloat, G}
    fill!(acc_ws, zero(T))
    rule = fp.weights
    Ti = Tuple(I)
    zerofill = strategy isa ZeroFill
    wn = _nd_rule_fold(zero(T), rule, grid, t, Ti, fp.rad, fp.topology) do awn, J, wk
        active = mask[J...]
        (zerofill || active) || return awn
        if active
            for m in eachindex(fields)
                acc_ws[m] += wk * fields[m][J...]
            end
        end
        awn + wk
    end
    zerofill && (wn = _add_exterior(wn, fp.exterior, I))
    @inbounds for m in eachindex(outs)
        outs[m][I] = _normalized(acc_ws[m], wn)
    end
    return nothing
end

# The per-field sums a batch of unknown width accumulates into; an `NTuple` batch folds its sums as values.
_nd_batch_buffer(::NTuple, ::Type) = nothing
_nd_batch_buffer(v::AbstractVector, ::Type{T}) where {T<:AbstractFloat} = zeros(T, length(v))

# One cached point of the ND batch, over the cells and weights the cache lists for it; the fields share one window
# mass.
@inline function _nd_cached_point!(
    outs::NTuple{K,<:AbstractArray}, fields::NTuple{K,<:AbstractArray},
    cache::Union{NDScatteredCache{N,T},NDRingCache{N,T}}, exterior, strategy::AbstractMaskStrategy, mask,
    I::CartesianIndex{N}, t::Int, _acc_ws,
) where {K, N, T<:AbstractFloat}
    zerofill = strategy isa ZeroFill
    ws, wn = _fold_cells((zero(SA.SVector{K,T}), zero(T)), cache, t, I) do a, J, wk
        aws, awn = a
        active = _cell_active(mask, J)
        (zerofill || active) || return a
        active && (aws = aws .+ wk .* SA.SVector{K,T}(ntuple(m -> _at(fields[m], J), Val(K))))
        (aws, awn + wk)
    end
    zerofill && (wn = _add_exterior(wn, exterior, I))
    @inbounds for m in 1:K
        outs[m][I] = _normalized(ws[m], wn)
    end
    return nothing
end

@inline function _nd_cached_point!(
    outs, fields, cache::Union{NDScatteredCache{N,T},NDRingCache{N,T}}, exterior, strategy::AbstractMaskStrategy, mask,
    I::CartesianIndex{N}, t::Int, acc_ws,
) where {N, T<:AbstractFloat}
    fill!(acc_ws, zero(T))
    zerofill = strategy isa ZeroFill
    wn = _fold_cells(zero(T), cache, t, I) do n, J, wk
        active = _cell_active(mask, J)
        (zerofill || active) || return n
        if active
            for m in eachindex(fields)
                acc_ws[m] += wk * _at(fields[m], J)
            end
        end
        n + wk
    end
    zerofill && (wn = _add_exterior(wn, exterior, I))
    @inbounds for m in eachindex(outs)
        outs[m][I] = _normalized(acc_ws[m], wn)
    end
    return nothing
end

"""
    apply_footprint_nd_batch_over!(outs, fields, grid, fp, strategy, indices) -> outs

The batched apply restricted to `indices`. Each point writes only its own output, so disjoint index
blocks may run concurrently. The caller zeroes `outs`.
"""
function apply_footprint_nd_batch_over!(
    outs, fields, grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, fp::NDScatteredFilterPlan{N,T}, strategy::AbstractMaskStrategy,
    indices,
) where {N, T<:AbstractFloat, G}
    dims = FlowGeometries.Grids.size_tuple(grid)
    mask = FlowGeometries.Grids.mask(grid)
    acc_ws = _nd_batch_buffer(outs, T)
    if fp.cache !== nothing
        cache = fp.cache
        lin = LinearIndices(dims)
        @inbounds for I in indices
            _filters_target(strategy, mask[I]) || continue
            _nd_cached_point!(outs, fields, cache, fp.exterior, strategy, mask, I, lin[I], acc_ws)
        end
    else
        lin = LinearIndices(dims)
        @inbounds for I in indices
            _filters_target(strategy, mask[I]) || continue
            _nd_stream_point!(outs, fields, grid, fp, strategy, mask, I, lin[I], acc_ws)
        end
    end
    return outs
end
