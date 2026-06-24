module CoarseGrainingEnergyFluxesMPIExt

using MPI: MPI
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF

# MPIBackend: multi-node (distributed-memory) execution. Each rank fills a disjoint stride of output
# latitude rows from the shared footprint (using the SAME per-row kernel as the serial backend), then
# the partial outputs are combined with an in-place Allreduce — since ranks own disjoint rows, the
# sum reconstructs the full field on every rank. The full input `field` is assumed replicated across
# ranks (each rank reads neighbour rows within the footprint); scatter/halo-only layouts are a future
# refinement. The caller is responsible for `MPI.Init()`.
#
# NOTE: built but not exercised in CI (no MPI runtime here); validate under `mpiexec -n P`.
function CGEF.Filtering.mpi_filter_field!(
    out::AbstractMatrix{T},
    field::AbstractMatrix,
    grid::CGEF.StructuredGrid{G,T},
    kernel::CGEF.AbstractFilterKernel,
    scale::T,
    mask_strategy::CGEF.AbstractMaskStrategy,
    workspace,
) where {T<:AbstractFloat, G<:CGEF.AbstractGeometry{T}}
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nproc = MPI.Comm_size(comm)

    fp = CGEF.Filtering.build_footprint(grid, kernel, scale)
    periodic = CGEF.isperiodic(grid, 1)
    _, Nlat = CGEF.size_tuple(grid)

    fill!(out, zero(T))
    # Round-robin row partition across ranks (balances land-mask cost like dynamic scheduling).
    for j in (rank + 1):nproc:Nlat
        CGEF.Filtering.apply_footprint_row!(out, field, grid, fp, mask_strategy, periodic, j)
    end
    MPI.Allreduce!(out, +, comm)   # disjoint rows ⇒ sum == full assembled field on every rank
    return out
end

end # module
