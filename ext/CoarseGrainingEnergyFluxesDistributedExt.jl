module CoarseGrainingEnergyFluxesDistributedExt

using Distributed: Distributed
using SharedArrays: SharedArrays
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF

# DistributedBackend: build the footprint once, then fill output rows across worker processes into a
# SharedArray (single shared-memory node — a dask-like multiprocess fill). Rows are independent
# (disjoint output columns), so results are IDENTICAL to the serial backend (same shared footprint
# + per-row kernel), including masking and periodic wrapping. With no extra worker processes the
# `@distributed` loop runs serially on the caller — still correct.
function CGEF.Filtering.distributed_filter_field!(
    out::AbstractMatrix{T},
    field::AbstractMatrix,
    grid::CGEF.StructuredGrid{G,T},
    kernel::CGEF.AbstractFilterKernel,
    scale::T,
    mask_strategy::CGEF.AbstractMaskStrategy,
    workspace,
) where {T<:AbstractFloat, G<:CGEF.AbstractGeometry{T}}
    fp = CGEF.Filtering.build_footprint(grid, kernel, scale)
    periodic = CGEF.isperiodic(grid, 1)
    Nlon, Nlat = CGEF.size_tuple(grid)
    s_out = SharedArrays.SharedArray{T}(Nlon, Nlat)
    fill!(s_out, zero(T))
    @sync Distributed.@distributed for j in 1:Nlat
        CGEF.Filtering.apply_footprint_row!(s_out, field, grid, fp, mask_strategy, periodic, j)
    end
    copyto!(out, s_out)
    return out
end

end # module
