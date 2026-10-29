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

# A real-space apply divided among the ranks, each holding the whole field: every pass runs the rank's
# block of its units on the rank's inner backend into the zeroed arrays the pass writes, and an
# `Allreduce!` sums the disjoint blocks into the whole pass on every rank, through a host buffer where the
# share holds one. A pass that costs less to run whole than to assemble runs whole on every rank. The
# caller replicates `field` on every rank and calls `MPI.Init()`.
CGEF.Filtering.prepare_workspace(
    b::CGEF.ComputationalBackends.MPIBackend, grid::FlowGeometries.Grids.AbstractGrid, fp::CGEF.Filtering._SharedFootprint,
) = CGEF.Filtering._share_workspace(CGEF.ComputationalBackends.local_backend(b), grid, fp)

function CGEF.Filtering.mpi_filter_field!(
    b::CGEF.ComputationalBackends.AbstractMPIBackend, out::AbstractArray, field::AbstractArray,
    grid::FlowGeometries.Grids.AbstractGrid, kernel::CGEF.Kernels.AbstractFilterKernel, scale::AbstractFloat,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy, workspace,
)
    fp = _footprint(workspace, grid, kernel, scale, mask_strategy)
    CGEF.Filtering._check_strategy(fp, mask_strategy)
    comm = _comm(b)
    k, nk = MPI.Comm_rank(comm) + 1, MPI.Comm_size(comm)
    inner = CGEF.ComputationalBackends.local_backend(b)
    _passes!(out, field, grid, fp, mask_strategy, Val(1), CGEF.Filtering._share_passes(fp), k, nk, inner, comm)
    return out
end

_footprint(::Nothing, grid, kernel, scale, strategy) =
    CGEF.Filtering.build_footprint(grid, kernel, scale; mask_strategy = strategy)
_footprint(fp::Union{CGEF.Filtering._SharedFootprint, CGEF.Filtering.AbstractProcessShare}, _, _, _, _) = fp
_footprint(fp, _, _, _, _) = throw(ArgumentError(
    "MPIBackend divides the real-space footprints among ranks; the $(nameof(typeof(fp))) engine runs in " *
    "one process. Name a footprint engine (`engine = FootprintEngine()`) or plan with a local backend."))

_passes!(out, field, grid, fp, strategy, ::Val{n}, ::Val{n}, k, nk, inner, comm) where {n} =
    _pass!(out, field, grid, fp, strategy, Val(n), k, nk, inner, comm)

function _passes!(out, field, grid, fp, strategy, ::Val{p}, ::Val{n}, k, nk, inner, comm) where {p,n}
    _pass!(out, field, grid, fp, strategy, Val(p), k, nk, inner, comm)
    return _passes!(out, field, grid, fp, strategy, Val(p + 1), Val(n), k, nk, inner, comm)
end

function _pass!(out, field, grid, fp, strategy, pass::Val, k::Int, nk::Int, inner, comm)
    if CGEF.Filtering._share_local(fp, pass)
        CGEF.Filtering._share_pass!(out, field, grid, fp, strategy, pass, 1, 1, inner)
        return nothing
    end
    dst = CGEF.Filtering._share_dst(out, fp, pass)
    foreach(a -> fill!(a, zero(eltype(a))), dst)
    CGEF.Filtering._share_pass!(out, field, grid, fp, strategy, pass, k, nk, inner)
    staging = CGEF.Filtering._share_staging(fp)
    foreach(a -> _sum_ranks!(a, staging, comm), dst)
    return nothing
end

_sum_ranks!(a, ::Nothing, comm) = MPI.Allreduce!(a, +, comm)
_sum_ranks!(a::Array, ::Array, comm) = MPI.Allreduce!(a, +, comm)
function _sum_ranks!(a, host::Array, comm)
    copyto!(host, a)
    MPI.Allreduce!(host, +, comm)
    copyto!(a, host)
    return a
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
