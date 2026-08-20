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
    if fp isa CGEF.Filtering.SeparableFootprint
        CGEF.Filtering._separable_check_strategy(fp, mask_strategy)
        return _mpi_apply_separable!(out, field, grid, fp, rank, nproc, comm)
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
function _mpi_apply_separable!(
    out::AbstractMatrix{T}, field::AbstractMatrix, grid::FlowGeometries.Grids.StructuredGrid, fp::CGEF.Filtering.SeparableFootprint{T},
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


# 1-D and true-3-D grids. `N` unconstrained, so the 2-D method above stays more specific for N=2. Ranks
# take a round-robin stride of linear indices and recombine with `Allreduce!`; strides are disjoint, so
# the sum reassembles the field exactly as the row partition does.
#
# The separable engine is run WHOLE on every rank rather than partitioned: its `N` passes each read
# across the previous pass's entire output, so a strided partition would need an Allreduce between
# passes, and the passes write buffers held inside the plan. Replicating it costs one rank's work and
# keeps the result exact; partitioning it needs a halo exchange per pass, which is a real design, not a
# loop change.
function CGEF.Filtering.mpi_filter_field!(
    out::AbstractArray{T,N},
    field::AbstractArray,
    grid::FlowGeometries.Grids.StructuredGrid{G,T,N},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
    workspace,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}, N}
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nproc = MPI.Comm_size(comm)
    fp = workspace === nothing ? CGEF.Filtering.build_footprint(grid, kernel, scale; mask_strategy = mask_strategy) : workspace
    dims = FlowGeometries.Grids.size_tuple(grid)
    mask = FlowGeometries.Grids.mask(grid)

    if fp isa CGEF.Filtering.SeparableFootprintND
        return CGEF.Filtering.apply_separable_nd!(out, field, grid, fp, mask_strategy)
    end

    fill!(out, zero(T))
    cart = CartesianIndices(dims)
    if fp isa CGEF.Filtering.FilterFootprintND
        periodic = FlowGeometries.Grids.periodic_flags(grid)
        for lin in (rank + 1):nproc:length(out)
            I = cart[lin]
            mask[I] || continue
            out[I] = CGEF.Filtering._footprint_nd_point(field, fp, mask_strategy, dims, periodic, mask, I)
        end
    elseif fp.cache !== nothing
        lin_idx = LinearIndices(dims)
        for lin in (rank + 1):nproc:length(out)
            I = cart[lin]
            mask[I] || continue
            out[I] = CGEF.Filtering._footprint_nd_point_cached(field, fp.cache, mask_strategy, mask, lin_idx, I)
        end
    else
        for lin in (rank + 1):nproc:length(out)
            I = cart[lin]
            mask[I] || continue
            out[I] = CGEF.Filtering._footprint_nd_point_streaming(field, grid, fp, mask_strategy, mask, dims, I)
        end
    end
    MPI.Allreduce!(out, +, comm)
    return out
end

# Node sets. A node grid's real-space footprint is always a `NodeFilterPlan`, so there is no separable
# or prefix-sum variant to branch on, and nodes have no row structure — the partition is round-robin
# over nodes instead. Ranks own disjoint nodes, so the `Allreduce!` sum reassembles the field exactly
# as the row version does, and the result is identical to serial.
function CGEF.Filtering.mpi_filter_field!(
    out::AbstractVector{T},
    field::AbstractVector,
    grid::FlowGeometries.Grids.UnstructuredGrid{T},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
    workspace,
) where {T<:AbstractFloat}
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nproc = MPI.Comm_size(comm)
    fp = workspace === nothing ? CGEF.Filtering.build_footprint(grid, kernel, scale; mask_strategy = mask_strategy) : workspace

    fill!(out, zero(T))
    for t in (rank + 1):nproc:length(out)
        out[t] = CGEF.Filtering._footprint_node_point(field, grid, fp, mask_strategy, t)
    end
    MPI.Allreduce!(out, +, comm)
    return out
end

# Batch-parallel PIPELINE sweeps across ranks: each rank takes a round-robin stride of slices, and the
# five batched arrays are assembled with an in-place Allreduce. Every slice's columns are written by
# exactly one rank, so summing zero-initialized partials reproduces the full result — including
# `scales` and `wavenumber`, which are per-slice views rather than shared vectors.
function CGEF.Pipeline.mpi_coarse_grain_batch!(batch, u, v, w, grid, valR, ctx)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nproc = MPI.Comm_size(comm)
    _zero_batch!(batch)
    for t in (rank + 1):nproc:length(batch.slices)
        CGEF.Pipeline.coarse_grain_batch_slice!(batch, u, v, w, grid, valR, ctx, t, 1)
    end
    MPI.Allreduce!(batch.Π, +, comm)
    MPI.Allreduce!(batch.scales, +, comm)
    MPI.Allreduce!(batch.cumulative_energy, +, comm)
    MPI.Allreduce!(batch.wavenumber, +, comm)
    MPI.Allreduce!(batch.filtering_spectrum, +, comm)
    return batch
end

# Ragged batch across ranks. Same round-robin split; each slice's own result is assembled separately
# because the shapes differ.
function CGEF.Pipeline.mpi_coarse_grain_slices!(results, us, vs, ws, grids, ctx)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nproc = MPI.Comm_size(comm)
    for r in results
        _zero_result!(r)
    end
    for t in (rank + 1):nproc:length(results)
        CGEF.Pipeline.coarse_grain_slice_serial!(results, us, vs, ws, grids, ctx, t)
    end
    for r in results
        MPI.Allreduce!(r.Π, +, comm)
        MPI.Allreduce!(r.scales, +, comm)
        MPI.Allreduce!(r.cumulative_energy, +, comm)
        MPI.Allreduce!(r.wavenumber, +, comm)
        MPI.Allreduce!(r.filtering_spectrum, +, comm)
    end
    return results
end

function _zero_batch!(batch)
    fill!(batch.Π, false)
    fill!(batch.scales, false)
    fill!(batch.cumulative_energy, false)
    fill!(batch.wavenumber, false)
    fill!(batch.filtering_spectrum, false)
    return nothing
end

function _zero_result!(r)
    fill!(r.Π, false)
    fill!(r.scales, false)
    fill!(r.cumulative_energy, false)
    fill!(r.wavenumber, false)
    fill!(r.filtering_spectrum, false)
    return nothing
end

end # module
