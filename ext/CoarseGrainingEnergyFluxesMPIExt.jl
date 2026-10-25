module CoarseGrainingEnergyFluxesMPIExt

using MPI: MPI
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF
using FlowGeometries: FlowGeometries

# The backend's communicator; `nothing` is the world.
_comm(b::CGEF.ComputationalBackends.AbstractMPIBackend) = something(b.comm, MPI.COMM_WORLD)

# A spectral transform under MPI: each rank transforms its round-robin share of the points of a field
# every rank holds, and the partial spectra and outputs are summed across ranks.
function CGEF.Filtering._owned(b::CGEF.ComputationalBackends.AbstractMPIBackend, n::Integer)
    comm = _comm(b)
    return collect((MPI.Comm_rank(comm) + 1):MPI.Comm_size(comm):Int(n))
end
CGEF.Filtering._sum_across!(b::CGEF.ComputationalBackends.AbstractMPIBackend, A::AbstractArray) =
    MPI.Allreduce!(A, +, _comm(b))

# Each rank fills a round-robin stride of rows, and an in-place `Allreduce!` sums the disjoint partials
# into the full field on every rank. The caller replicates `field` on every rank and calls `MPI.Init()`.
function CGEF.Filtering.mpi_filter_field!(
    b::CGEF.ComputationalBackends.AbstractMPIBackend,
    out::AbstractMatrix{T},
    field::AbstractMatrix,
    grid::Union{FlowGeometries.Grids.StructuredGrid{T,G,2}, FlowGeometries.Grids.CurvilinearGrid{T,G}},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
    workspace,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    comm = _comm(b)
    rank = MPI.Comm_rank(comm)
    nproc = MPI.Comm_size(comm)

    # A `workspace` from a `PhysicalFilterPlan` is its built footprint.
    fp = workspace === nothing ? CGEF.Filtering.build_footprint(grid, kernel, scale; mask_strategy = mask_strategy) : workspace
    CGEF.Filtering._check_strategy(fp, mask_strategy)
    if fp isa CGEF.Filtering.SeparableFootprint
        return _mpi_apply_separable!(out, field, grid, fp, rank, nproc, comm)
    elseif fp isa CGEF.Filtering.PrefixSumTopHatPlan
        return _mpi_apply_prefixsum_tophat!(out, field, grid, fp, mask_strategy, rank, nproc, comm)
    end
    periodic_x = FlowGeometries.Grids.isperiodic(grid, 1)
    periodic_y = FlowGeometries.Grids.isperiodic(grid, 2)
    _, Ny = FlowGeometries.Grids.size_tuple(grid)

    fill!(out, zero(T))
    CGEF.Filtering.prepare_row_apply!(fp, field, grid)
    # Round-robin rows balance the per-row masking cost across ranks.
    for j in (rank + 1):nproc:Ny
        CGEF.Filtering.apply_footprint_row!(out, field, grid, fp, mask_strategy, periodic_x, periodic_y, j)
    end
    MPI.Allreduce!(out, +, comm)   # disjoint rows ⇒ sum == full assembled field on every rank
    return out
end

# Separable Gaussian. Each rank runs both passes over its own stride of rows, and an `Allreduce!` after
# each assembles the pass on every rank: the column pass reads every row of `row_pass`. Normalization is
# pointwise, so it runs whole on each rank after the second.
function _mpi_apply_separable!(
    out::AbstractMatrix{T}, field::AbstractMatrix, grid::FlowGeometries.Grids.StructuredGrid, fp::CGEF.Filtering.SeparableFootprint{T},
    rank::Integer, nproc::Integer, comm,
) where {T<:AbstractFloat}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    mask = FlowGeometries.Grids.mask(grid)
    gx, gy = fp.gx, fp.gy
    di_lim, dj_lim = fp.di_lim, fp.dj_lim
    periodic_x, periodic_y = fp.periodic_x, fp.periodic_y

    masked_input = fp.masked_input
    row_pass = fp.row_pass
    @. masked_input = mask * field
    fill!(row_pass, zero(T))
    for j in (rank + 1):nproc:Ny
        CGEF.Filtering._separable_row_pass_at!(row_pass, masked_input, gx, di_lim, periodic_x, Nx, j)
    end
    MPI.Allreduce!(row_pass, +, comm)

    fill!(out, zero(T))
    for j in (rank + 1):nproc:Ny
        CGEF.Filtering._separable_column_pass_at!(out, row_pass, gy, dj_lim, periodic_y, Nx, Ny, j)
    end
    MPI.Allreduce!(out, +, comm)   # disjoint rows ⇒ sum == full assembled raw convolution on every rank
    CGEF.Filtering._separable_normalize_and_mask!(out, fp, mask, Nx, Ny)
    return out
end

# The N-pass separable engine's driver under MPI: each rank runs its stride of a pass's lines into the
# zeroed destination, and an `Allreduce!` assembles the pass on every rank before the next reads it. The
# pointwise normalization runs whole on each rank.
struct _RankPasses{C}
    comm::C
    rank::Int
    nproc::Int
end

function CGEF.Filtering._drive_pass!(f::F, d::_RankPasses, indices, dst) where {F}
    fill!(dst, zero(eltype(dst)))
    for q in (d.rank + 1):d.nproc:length(indices)
        f(@inbounds indices[q])
    end
    MPI.Allreduce!(dst, +, d.comm)
    return nothing
end

CGEF.Filtering._drive_pointwise!(f::F, ::_RankPasses, indices) where {F} = CGEF.Filtering._sep_serial(f, indices)

# 3-D prefix-sum top-hat: each rank scans the whole field, which it holds, and sweeps its stride of
# `(j, k)` planes; each plane is written by one rank, so the `Allreduce!` sum is final.
function _mpi_apply_prefixsum_tophat_3d!(
    out::AbstractArray{T,3}, field::AbstractArray, grid::FlowGeometries.Grids.StructuredGrid,
    fp::CGEF.Filtering.PrefixSumTopHat3DPlan{T}, strategy::CGEF.Filtering.AbstractMaskStrategy,
    rank::Integer, nproc::Integer, comm,
) where {T<:AbstractFloat}
    CGEF.Filtering._prefixsum3d_check_strategy(fp, strategy)
    _, Ny, Nz = FlowGeometries.Grids.size_tuple(grid)
    CGEF.Filtering._prefixsum3d_fill!(fp.scratch.prefix, field, grid, fp.masked)
    fill!(out, zero(T))
    planes = CartesianIndices((Ny, Nz))
    for q in (rank + 1):nproc:length(planes)
        JK = planes[q]
        CGEF.Filtering._prefixsum3d_out_plane!(out, fp, JK[1], JK[2])
    end
    MPI.Allreduce!(out, +, comm)
    return out
end

# Prefix-sum top-hat. Every rank builds the whole prefix table and sweeps its own stride of rows; the
# row kernel normalizes each point, so the `Allreduce!` sum is final.
function _mpi_apply_prefixsum_tophat!(
    out::AbstractMatrix{T}, field::AbstractMatrix, grid::FlowGeometries.Grids.StructuredGrid,
    fp::CGEF.Filtering.PrefixSumTopHatPlan{T}, strategy::CGEF.Filtering.AbstractMaskStrategy,
    rank::Integer, nproc::Integer, comm,
) where {T<:AbstractFloat}
    _, Ny = FlowGeometries.Grids.size_tuple(grid)
    CGEF.Filtering._prefixsum_check_strategy(fp, strategy)
    CGEF.Filtering.prefixsum_fill_numerator!(fp, field, grid)
    fill!(out, zero(T))
    for j in (rank + 1):nproc:Ny
        CGEF.Filtering.apply_prefixsum_tophat_row!(out, grid, fp, strategy, j)
    end
    MPI.Allreduce!(out, +, comm)   # disjoint rows ⇒ sum == full assembled field on every rank
    return out
end


# 1-D and 3-D grids; the 2-D method above takes `N == 2`. Ranks take disjoint round-robin strides of
# linear indices, of each separable pass's lines, or of the 3-D prefix sum's planes, summed by
# `Allreduce!`.
function CGEF.Filtering.mpi_filter_field!(
    b::CGEF.ComputationalBackends.AbstractMPIBackend,
    out::AbstractArray{T,N},
    field::AbstractArray,
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
    workspace,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}, N}
    comm = _comm(b)
    rank = MPI.Comm_rank(comm)
    nproc = MPI.Comm_size(comm)
    fp = workspace === nothing ? CGEF.Filtering.build_footprint(grid, kernel, scale; mask_strategy = mask_strategy) : workspace
    CGEF.Filtering._check_strategy(fp, mask_strategy)
    dims = FlowGeometries.Grids.size_tuple(grid)
    mask = FlowGeometries.Grids.mask(grid)

    if fp isa CGEF.Filtering.PrefixSumTopHat3DPlan
        return _mpi_apply_prefixsum_tophat_3d!(out, field, grid, fp, mask_strategy, rank, nproc, comm)
    end

    if fp isa CGEF.Filtering.SeparableFootprintND
        return CGEF.Filtering.apply_separable_nd!(out, field, grid, fp, mask_strategy, _RankPasses(comm, rank, nproc))
    end

    fill!(out, zero(T))
    cart = CartesianIndices(dims)
    if fp isa CGEF.Filtering.FilterFootprintND
        # Ranks take a round-robin stride of the lines along axis 1.
        src = CGEF.Filtering._nd_source!(fp, field, grid)
        lines = CartesianIndices(Base.tail(dims))
        for q in (rank + 1):nproc:length(lines)
            CGEF.Filtering._nd_out_line!(out, src, fp, Tuple(lines[q]))
        end
    elseif fp.cache !== nothing
        # The point functions decide which targets the strategy filters.
        lin_idx = LinearIndices(dims)
        for lin in (rank + 1):nproc:length(out)
            I = cart[lin]
            out[I] = CGEF.Filtering._footprint_nd_point_cached(field, fp.cache, fp.exterior, mask_strategy, mask, lin_idx, I)
        end
    else
        for lin in (rank + 1):nproc:length(out)
            I = cart[lin]
            out[I] = CGEF.Filtering._footprint_nd_point_streaming(field, grid, fp, mask_strategy, mask, I)
        end
    end
    MPI.Allreduce!(out, +, comm)
    return out
end

# Node sets: ranks own disjoint round-robin strides of nodes, summed by `Allreduce!`.
function CGEF.Filtering.mpi_filter_field!(
    b::CGEF.ComputationalBackends.AbstractMPIBackend,
    out::AbstractVector{T},
    field::AbstractVector,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
    fp::CGEF.Filtering.NodeFilterPlan,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    comm = _comm(b)
    rank = MPI.Comm_rank(comm)
    nproc = MPI.Comm_size(comm)

    fill!(out, zero(T))
    for t in (rank + 1):nproc:length(out)
        out[t] = CGEF.Filtering._footprint_node_point(field, grid, fp, mask_strategy, t)
    end
    MPI.Allreduce!(out, +, comm)
    return out
end

# Each rank sweeps a round-robin stride of slices into the zeroed batch arrays. A slice's columns are
# written by one rank, so the `Allreduce!` sums reproduce the full result, `scales` and `wavenumber`
# included.
function CGEF.Pipeline.mpi_coarse_grain_batch!(
    b::CGEF.ComputationalBackends.AbstractMPIBackend, batch, u, v, w, grid, valR, ctx,
)
    comm = _comm(b)
    rank = MPI.Comm_rank(comm)
    nproc = MPI.Comm_size(comm)
    _zero_batch!(batch)
    mine = (rank + 1):nproc:length(batch.slices)
    isempty(mine) || CGEF.Pipeline._run_slices!(batch, u, v, w, grid, valR, ctx, mine,
                                                CGEF.Pipeline._worker_scratch(ctx, grid, w, 1))
    MPI.Allreduce!(batch.Π, +, comm)
    MPI.Allreduce!(batch.scales, +, comm)
    MPI.Allreduce!(batch.cumulative_energy, +, comm)
    MPI.Allreduce!(batch.wavenumber, +, comm)
    MPI.Allreduce!(batch.filtering_spectrum, +, comm)
    return batch
end

# Ragged batch across ranks. Same round-robin split; each slice's own result is assembled separately
# because the shapes differ.
function CGEF.Pipeline.mpi_coarse_grain_slices!(
    b::CGEF.ComputationalBackends.AbstractMPIBackend, results, us, vs, ws, grids, ctx,
)
    comm = _comm(b)
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
