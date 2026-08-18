module CoarseGrainingEnergyFluxesDistributedExt

using Distributed: Distributed
using SharedArrays: SharedArrays
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF
using FlowGeometries: FlowGeometries

# DistributedBackend: build the footprint once, then fill output rows across worker processes into a
# SharedArray — a single shared-memory node, not a multi-node decomposition. Rows write disjoint
# output columns, so the result is identical to serial. With no extra workers the `@distributed` loop
# runs on the caller. Works for any row-decomposable 2D grid, structured or curvilinear.
function CGEF.Filtering.distributed_filter_field!(
    out::AbstractMatrix{T},
    field::AbstractMatrix,
    grid::Union{FlowGeometries.Grids.StructuredGrid{G,T,2}, FlowGeometries.Grids.CurvilinearGrid{T,G}},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
    workspace,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    fp = workspace === nothing ? CGEF.Filtering.build_footprint(grid, kernel, scale; mask_strategy = mask_strategy) : workspace
    if fp isa CGEF.Filtering.SeparableGaussianFootprint
        CGEF.Filtering._separable_check_strategy(fp, mask_strategy)
        return _distributed_apply_separable_gaussian!(out, field, grid, fp)
    elseif fp isa CGEF.Filtering.PrefixSumTopHatPlan
        # The prefix table lives in the plan (ordinary process-local arrays, not a SharedArray), so a
        # `@distributed` loop over rows would build it in the workers' own address spaces and the
        # caller would see nothing. The prefix pass is O(N) — negligible against the O(N·dj_lim) apply
        # — so run this path locally in full rather than pretending to distribute it.
        return CGEF.Filtering.apply_prefixsum_tophat!(out, field, grid, fp, mask_strategy)
    end
    periodic_x = FlowGeometries.Grids.isperiodic(grid, 1)
    periodic_y = FlowGeometries.Grids.isperiodic(grid, 2)
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    s_out = SharedArrays.SharedArray{T}(Nx, Ny)
    fill!(s_out, zero(T))
    @sync Distributed.@distributed for j in 1:Ny
        CGEF.Filtering.apply_footprint_row!(s_out, field, grid, fp, mask_strategy, periodic_x, periodic_y, j)
    end
    copyto!(out, s_out)
    return out
end

# Separable Gaussian. `masked_input` is copied into a `SharedArray` before either loop starts, so the
# row pass needs no communication — each worker's rows read only their own column. The column pass is
# then row-parallel against the completed `row_pass` SharedArray. Both call the same per-row/per-column
# bodies as serial, so results are bit-identical.
function _distributed_apply_separable_gaussian!(
    out::AbstractMatrix{T}, field::AbstractMatrix, grid::FlowGeometries.Grids.StructuredGrid, fp::CGEF.Filtering.SeparableGaussianFootprint{T},
) where {T<:AbstractFloat}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    mask = FlowGeometries.Grids.mask(grid)
    gx, gy = fp.gx, fp.gy
    di_lim, dj_lim = fp.di_lim, fp.dj_lim
    periodic_x, periodic_y = fp.periodic_x, fp.periodic_y

    masked_input = SharedArrays.SharedArray{T}(Nx, Ny)
    masked_input .= mask .* field   # fused into the SharedArray; `copyto!` would materialize a temporary
    row_pass = SharedArrays.SharedArray{T}(Nx, Ny)
    @sync Distributed.@distributed for j in 1:Ny
        CGEF.Filtering._separable_row_pass_at!(row_pass, masked_input, gx, di_lim, periodic_x, Nx, j)
    end

    s_out = SharedArrays.SharedArray{T}(Nx, Ny)
    @sync Distributed.@distributed for j in 1:Ny
        CGEF.Filtering._separable_column_pass_at!(s_out, row_pass, gy, dj_lim, periodic_y, Nx, Ny, j)
    end
    CGEF.Filtering._separable_normalize_and_mask!(s_out, fp, mask, Nx, Ny)
    copyto!(out, s_out)
    return out
end


# Distributed analogue of the driver the serial and threaded backends pass to
# `apply_separable_gaussian_nd!`, so the `N`-pass engine has one implementation across all three.
@inline function _dist_driver(f::F, indices) where {F}
    @sync Distributed.@distributed for i in eachindex(indices)
        f(indices[i])
    end
    return nothing
end

_shared_like(a::AbstractArray{T}) where {T} =
    (sh = SharedArrays.SharedArray{T}(size(a)); copyto!(sh, a); sh)

# 1-D and true-3-D grids. `N` is unconstrained, so the 2-D method above is more specific and still wins
# for N=2. Point-indexed footprints decompose over linear indices; the separable Gaussian instead has
# its plan-local pass buffers REBUILT as SharedArrays, because a `@distributed` loop writing the plan's
# own arrays would write them in the workers' address spaces and the caller would see nothing.
function CGEF.Filtering.distributed_filter_field!(
    out::AbstractArray{T,N},
    field::AbstractArray,
    grid::FlowGeometries.Grids.StructuredGrid{G,T,N},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
    workspace,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}, N}
    fp = workspace === nothing ? CGEF.Filtering.build_footprint(grid, kernel, scale; mask_strategy = mask_strategy) : workspace
    dims = FlowGeometries.Grids.size_tuple(grid)
    mask = FlowGeometries.Grids.mask(grid)

    if fp isa CGEF.Filtering.SeparableGaussianFootprintND
        CGEF.Filtering._separable_check_strategy(fp, mask_strategy)
        sfp = CGEF.Filtering.SeparableGaussianFootprintND(
            fp.g, fp.lim, fp.periodic, fp.profiles, fp.invrenorm, fp.masked,
            _shared_like(fp.masked_input), _shared_like(fp.scratch),
        )
        s_out = SharedArrays.SharedArray{T}(dims)
        CGEF.Filtering.apply_separable_gaussian_nd!(s_out, field, grid, sfp, mask_strategy, _dist_driver)
        copyto!(out, s_out)
        return out
    end

    s_out = SharedArrays.SharedArray{T}(dims)
    fill!(s_out, zero(T))
    cart = CartesianIndices(dims)
    if fp isa CGEF.Filtering.FilterFootprintND
        periodic = FlowGeometries.Grids.periodic_flags(grid)
        @sync Distributed.@distributed for lin in 1:length(s_out)
            I = cart[lin]
            if mask[I]
                s_out[I] = CGEF.Filtering._footprint_nd_point(field, fp, mask_strategy, dims, periodic, mask, I)
            end
        end
    elseif fp.cache !== nothing
        lin_idx = LinearIndices(dims)
        cache = fp.cache
        @sync Distributed.@distributed for lin in 1:length(s_out)
            I = cart[lin]
            if mask[I]
                s_out[I] = CGEF.Filtering._footprint_nd_point_cached(field, cache, mask_strategy, mask, lin_idx, I)
            end
        end
    else
        @sync Distributed.@distributed for lin in 1:length(s_out)
            I = cart[lin]
            if mask[I]
                s_out[I] = CGEF.Filtering._footprint_nd_point_streaming(field, grid, fp, mask_strategy, mask, dims, I)
            end
        end
    end
    copyto!(out, s_out)
    return out
end

# Node sets. A node grid's real-space footprint is always a `NodeFilterPlan`, so unlike the 2-D method
# there is no separable or prefix-sum variant to branch on. Nodes carry no row structure, so the
# decomposition is over `eachindex(out)` directly; node `t` writes only `out[t]`, so the SharedArray
# needs no reduction and the result is identical to serial.
function CGEF.Filtering.distributed_filter_field!(
    out::AbstractVector{T},
    field::AbstractVector,
    grid::FlowGeometries.Grids.UnstructuredGrid{T},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
    workspace,
) where {T<:AbstractFloat}
    fp = workspace === nothing ? CGEF.Filtering.build_footprint(grid, kernel, scale; mask_strategy = mask_strategy) : workspace
    n = length(out)
    s_out = SharedArrays.SharedArray{T}(n)
    fill!(s_out, zero(T))
    @sync Distributed.@distributed for t in 1:n
        s_out[t] = CGEF.Filtering._footprint_node_point(field, grid, fp, mask_strategy, t)
    end
    copyto!(out, s_out)
    return out
end

CGEF.Pipeline.batch_alloc_shared(::Type{T}, dims::Integer...) where {T} =
    (sh = SharedArrays.SharedArray{T}(dims); fill!(sh, zero(T)); sh)

# Batch-parallel PIPELINE sweeps across worker processes — one shared-memory node, not a multi-node
# decomposition. Slices write disjoint views of the batched result, so no synchronization is needed and
# the assembled result is identical to serial.
#
# The storage must be shared: a `@distributed` loop over plain arrays would write them in the workers'
# own address spaces and the caller would see zeros. That is a hard error rather than a silent copy.
function CGEF.Pipeline.distributed_coarse_grain_batch!(batch, u, v, w, grid, valR, ctx)
    _require_shared(batch.Π, "coarse_grain_batch!")
    n = length(batch.slices)
    @sync Distributed.@distributed for t in 1:n
        CGEF.Pipeline.coarse_grain_batch_slice!(batch, u, v, w, grid, valR, ctx, t, 1)
    end
    return batch
end

# Ragged batch across workers. Shapes differ per slice, so each slice owns its own result and there is
# no shared batched tensor to write into — every result's storage must be shared instead.
function CGEF.Pipeline.distributed_coarse_grain_slices!(results, us, vs, ws, grids, ctx)
    for r in results
        _require_shared(r.Π, "coarse_grain_slices!")
    end
    n = length(results)
    @sync Distributed.@distributed for t in 1:n
        CGEF.Pipeline.coarse_grain_slice_serial!(results, us, vs, ws, grids, ctx, t)
    end
    return results
end

function _require_shared(A, fname::AbstractString)
    parent(A) isa SharedArrays.SharedArray || throw(ArgumentError(
        "DistributedBackend needs shared result storage for $fname; allocate it with " *
        "`alloc = CoarseGrainingEnergyFluxes.Pipeline.batch_alloc_shared`, or use SerialBackend()/ThreadedBackend()",
    ))
    return nothing
end

end # module
