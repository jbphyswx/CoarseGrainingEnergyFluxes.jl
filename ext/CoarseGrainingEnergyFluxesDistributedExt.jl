module CoarseGrainingEnergyFluxesDistributedExt

using Distributed: Distributed
using SharedArrays: SharedArrays
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF

# DistributedBackend: build the footprint once, then fill output rows across worker processes into a
# SharedArray (single shared-memory node — a dask-like multiprocess fill). Rows are independent
# (disjoint output columns), so results are IDENTICAL to the serial backend (same shared footprint
# + per-row kernel), including masking and periodic wrapping. With no extra worker processes the
# `@distributed` loop runs serially on the caller — still correct. `workspace`, when supplied by a
# cached `PhysicalFilterPlan`, IS the already-built footprint — reused instead of rebuilding it.
# Grid-generic (see the OhMyThreadsExt comment): `apply_footprint_row!` already works for any
# row-decomposable 2D grid, StructuredGrid or CurvilinearGrid.
function CGEF.Filtering.distributed_filter_field!(
    out::AbstractMatrix{T},
    field::AbstractMatrix,
    grid::Union{CGEF.StructuredGrid{G,T,2}, CGEF.CurvilinearGrid{T,G}},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
    workspace,
) where {T<:AbstractFloat, G<:CGEF.Geometry.AbstractGeometry{T}}
    fp = workspace === nothing ? CGEF.Filtering.build_footprint(grid, kernel, scale) : workspace
    periodic = CGEF.Grids.isperiodic(grid, 1)
    Nlon, Nlat = CGEF.Grids.size_tuple(grid)
    s_out = SharedArrays.SharedArray{T}(Nlon, Nlat)
    fill!(s_out, zero(T))
    @sync Distributed.@distributed for j in 1:Nlat
        CGEF.Filtering.apply_footprint_row!(s_out, field, grid, fp, mask_strategy, periodic, j)
    end
    copyto!(out, s_out)
    return out
end

end # module
