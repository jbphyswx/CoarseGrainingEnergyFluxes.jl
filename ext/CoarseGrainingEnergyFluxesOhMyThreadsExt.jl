module CoarseGrainingEnergyFluxesOhMyThreadsExt

using OhMyThreads: OhMyThreads
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF

# ThreadedBackend: build the footprint once, then fill output rows in parallel. Rows are
# independent (each writes a disjoint column of the column-major output), so this is data-race-free
# and produces results IDENTICAL to the serial backend (same shared footprint + per-row kernel),
# including periodic wrapping and masking. Dynamic scheduling balances the uneven per-row cost from
# land masks.
function CGEF.Filtering.threaded_filter_field!(
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
    _, Nlat = CGEF.size_tuple(grid)
    fill!(out, zero(T))
    OhMyThreads.tforeach(1:Nlat; scheduler = :dynamic) do j
        CGEF.Filtering.apply_footprint_row!(out, field, grid, fp, mask_strategy, periodic, j)
    end
    return out
end

end # module
