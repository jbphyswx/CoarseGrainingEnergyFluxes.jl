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

end # module
