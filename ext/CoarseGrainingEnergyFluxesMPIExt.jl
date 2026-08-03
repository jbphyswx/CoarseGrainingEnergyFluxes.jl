module CoarseGrainingEnergyFluxesMPIExt

using MPI: MPI
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF
using FlowGeometries: FlowGeometries

# MPIBackend: multi-node (distributed-memory) execution. Each rank fills a disjoint stride of output
# latitude rows from the shared footprint (using the SAME per-row kernel as the serial backend), then
# the partial outputs are combined with an in-place Allreduce — since ranks own disjoint rows, the
# sum reconstructs the full field on every rank. The full input `field` is assumed replicated across
# ranks (each rank reads neighbour rows within the footprint); scatter/halo-only layouts are a future
# refinement. The caller is responsible for `MPI.Init()`.
#
# Not exercised in CI (no MPI runtime); validate under `mpiexec -n P`. `apply_footprint_row!` works
# for any row-decomposable 2D grid, structured or curvilinear.
function CGEF.Filtering.mpi_filter_field!(
    out::AbstractMatrix{T},
    field::AbstractMatrix,
    grid::Union{FlowGeometries.Grids.StructuredGrid{G,T,2}, FlowGeometries.Grids.CurvilinearGrid{T,G}},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
    workspace,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nproc = MPI.Comm_size(comm)

    # `workspace`, when supplied by a cached `PhysicalFilterPlan`, IS the already-built footprint —
    # reused instead of rebuilding it on every call (and every rank, in this backend's case).
    fp = workspace === nothing ? CGEF.Filtering.build_footprint(grid, kernel, scale; mask_strategy = mask_strategy) : workspace
    if fp isa CGEF.Filtering.SeparableGaussianFootprint
        CGEF.Filtering._separable_check_strategy(fp, mask_strategy)
        return _mpi_apply_separable_gaussian!(out, field, grid, fp, rank, nproc, comm)
    elseif fp isa CGEF.Filtering.PrefixSumTopHatPlan
        return _mpi_apply_prefixsum_tophat!(out, field, grid, fp, mask_strategy, rank, nproc, comm)
    end
    periodic_x = FlowGeometries.Grids.isperiodic(grid, 1)
    periodic_y = FlowGeometries.Grids.isperiodic(grid, 2)
    _, Ny = FlowGeometries.Grids.size_tuple(grid)

    fill!(out, zero(T))
    # Round-robin row partition across ranks (balances per-row masking cost like dynamic scheduling).
    for j in (rank + 1):nproc:Ny
        CGEF.Filtering.apply_footprint_row!(out, field, grid, fp, mask_strategy, periodic_x, periodic_y, j)
    end
    MPI.Allreduce!(out, +, comm)   # disjoint rows ⇒ sum == full assembled field on every rank
    return out
end

# Separable Gaussian. `field` is replicated, so every rank computes the whole `row_pass` locally with
# no communication — redundant work, zero messages. Each rank then column-passes only its own stride,
# and the `Allreduce!` assembles the un-normalized result. Normalization is pointwise in the assembled
# `out`, so it runs on every rank afterwards.
function _mpi_apply_separable_gaussian!(
    out::AbstractMatrix{T}, field::AbstractMatrix, grid::FlowGeometries.Grids.StructuredGrid, fp::CGEF.Filtering.SeparableGaussianFootprint{T},
    rank::Integer, nproc::Integer, comm,
) where {T<:AbstractFloat}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    mask = FlowGeometries.Grids.mask(grid)
    gx, gy = fp.gx, fp.gy
    di_lim, dj_lim = fp.di_lim, fp.dj_lim
    periodic_x, periodic_y = fp.periodic_x, fp.periodic_y

    masked_input = T.(mask) .* field
    row_pass = similar(masked_input)
    for j in 1:Ny   # redundant across ranks -- zero communication, `field` already replicated
        CGEF.Filtering._separable_row_pass_at!(row_pass, masked_input, gx, di_lim, periodic_x, Nx, j)
    end

    fill!(out, zero(T))
    for j in (rank + 1):nproc:Ny   # only this rank's assigned output rows
        CGEF.Filtering._separable_column_pass_at!(out, row_pass, gy, dj_lim, periodic_y, Nx, Ny, j)
    end
    MPI.Allreduce!(out, +, comm)   # disjoint rows ⇒ sum == full assembled raw convolution on every rank
    CGEF.Filtering._separable_normalize_and_mask!(out, fp, mask, Nx, Ny)
    return out
end

# Prefix-sum top-hat. Each rank builds the full prefix table locally, applies its own stride, and the
# `Allreduce!` assembles the result. No deferred normalization here: the row kernel already divides by
# its own per-point denominator, and disjoint strides mean each point is written by exactly one rank.
function _mpi_apply_prefixsum_tophat!(
    out::AbstractMatrix{T}, field::AbstractMatrix, grid::FlowGeometries.Grids.StructuredGrid,
    fp::CGEF.Filtering.PrefixSumTopHatPlan{T}, strategy::CGEF.Filtering.AbstractMaskStrategy,
    rank::Integer, nproc::Integer, comm,
) where {T<:AbstractFloat}
    _, Ny = FlowGeometries.Grids.size_tuple(grid)
    CGEF.Filtering._prefixsum_check_strategy(fp, strategy)
    CGEF.Filtering.prefixsum_fill_numerator!(fp, field, grid)   # redundant across ranks, no messages
    fill!(out, zero(T))
    for j in (rank + 1):nproc:Ny
        CGEF.Filtering.apply_prefixsum_tophat_row!(out, grid, fp, strategy, j)
    end
    MPI.Allreduce!(out, +, comm)   # disjoint rows ⇒ sum == full assembled field on every rank
    return out
end

end # module
