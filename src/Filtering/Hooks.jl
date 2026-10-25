# ---------------------------------------------------------------------------
# Extension hook points
# ---------------------------------------------------------------------------
# Fallbacks that raise until the backend's extension, which overrides them, is loaded.

function threaded_filter_field!(args...; kwargs...)
    throw(ArgumentError("ThreadedBackend is unavailable — run `using OhMyThreads` (or use SerialBackend())."))
end

# Several fields sharing one plan; a scattered footprint derives each point's neighbour list once for
# the whole batch.
function threaded_filter_fields!(args...; kwargs...)
    throw(ArgumentError("ThreadedBackend is unavailable — run `using OhMyThreads` (or use SerialBackend())."))
end

# Independent slices, one plan each; see `filter_slices!`.
function threaded_filter_slices!(args...; kwargs...)
    throw(ArgumentError("ThreadedBackend is unavailable — run `using OhMyThreads` (or use SerialBackend())."))
end

# Padded-FFT real-space engine (FFTW extension). Zero-padding makes the transform the linear
# convolution, valid on bounded and masked domains.
function padded_fft_footprint(args...; kwargs...)
    throw(ArgumentError("The padded-FFT real-space engine needs FFTW — run `using FFTW`."))
end

# Zonal-FFT real-space engine (FFTW extension). On a sphere of uniform longitude, for a fixed pair of
# latitudes a great-circle kernel depends on the longitude difference alone, so each latitude band is a
# convolution along longitude, circular on a ring and zero-padded on a regional box, with the direct
# sum's weights.
function zonal_fft_footprint(args...; kwargs...)
    throw(ArgumentError("The zonal-FFT spherical engine needs FFTW — run `using FFTW`."))
end

# Whether `fp` is a host engine applied through `apply_footprint!(out, field, grid, fp, strategy, driver)`,
# serially or under `ThreadedBackend` with a driver of `_sep_serial`'s shape: the padded and zonal FFTs, the
# node-set NUFFT sum and the binned top-hat.
_driven_footprint(::Any) = false

# A batch through a driven engine, field by field; an engine that shares its work across the fields of a
# batch has its own method.
function _driven_batch!(outs, fields, grid, fp, strategy, driver = _sep_serial)
    for k in eachindex(outs)
        apply_footprint!(outs[k], fields[k], grid, fp, strategy, driver)
    end
    return outs
end

const _DistributedBackends =
    Union{ComputationalBackends.AbstractDistributedBackend, ComputationalBackends.AbstractMPIBackend}

"""
    _library_threads(backend) -> Int

The thread count a transform library runs at in a plan built for `backend`: 1 under `SerialBackend`,
the backend a threaded slice or batch loop builds its plans with; `Threads.nthreads()` under
`ThreadedBackend`, `AutoBackend` and `GPUBackend`. A distributed backend runs each process's share at
the count of its inner backend.
"""
_library_threads(::ComputationalBackends.AbstractSerialBackend) = 1
_library_threads(
    ::Union{ComputationalBackends.AbstractThreadedBackend, ComputationalBackends.AbstractAutoBackend,
            ComputationalBackends.AbstractGPUBackend},
) = Threads.nthreads()
_library_threads(b::_DistributedBackends) = _library_threads(ComputationalBackends.local_backend(b))

"""
    _allocate(backend, T, dims) -> array
    _on_backend(backend, x) -> array
    _host_memory(backend) -> Bool

Zeroed storage of element type `T`, and a copy of the host array `x`, in the memory `backend` runs in,
and whether that memory is the host's. It is host memory for every backend but a `GPUBackend` on a
device other than KernelAbstractions' CPU, whose memory the KernelAbstractions extension allocates. An
MPI rank runs in its inner backend's memory; a `DistributedBackend` plan's caller holds host memory,
and its workers their inner backend's.
"""
_allocate(::ComputationalBackends.AbstractExecutionBackend, ::Type{T}, dims::Dims) where {T} = zeros(T, dims)
_allocate(b::ComputationalBackends.AbstractMPIBackend, ::Type{T}, dims::Dims) where {T} =
    _allocate(ComputationalBackends.local_backend(b), T, dims)
_on_backend(backend::ComputationalBackends.AbstractExecutionBackend, x::AbstractArray) =
    copyto!(_allocate(backend, eltype(x), size(x)), Array(x))
_host_memory(::ComputationalBackends.AbstractExecutionBackend) = true
_host_memory(b::ComputationalBackends.AbstractMPIBackend) = _host_memory(ComputationalBackends.local_backend(b))

"""
    _owned(backend, n) -> Vector{Int} or nothing
    _sum_across!(backend, A) -> A

The points of `n` this process transforms, `nothing` for all of them, and the sum of `A` over the
processes that share the transform, in place. Under `MPIBackend` (MPI extension) each rank owns the
points `rank + 1, rank + 1 + nproc, …` of a field every rank holds, and the sum is an `Allreduce`.
"""
_owned(::ComputationalBackends.AbstractExecutionBackend, ::Integer) = nothing
_sum_across!(::ComputationalBackends.AbstractExecutionBackend, A::AbstractArray) = A

function distributed_filter_field!(args...; kwargs...)
    throw(ArgumentError("DistributedBackend is unavailable — run `using Distributed` (or use SerialBackend())."))
end

function gpu_filter_field!(args...; kwargs...)
    throw(ArgumentError("GPUBackend is unavailable — run `using KernelAbstractions` + a GPU backend (or use SerialBackend())."))
end

function mpi_filter_field!(args...; kwargs...)
    throw(ArgumentError("MPIBackend is unavailable — run `using MPI` (or use SerialBackend())."))
end

# Build a spectral filter plan: forward transform → multiply by `spectral_transfer` → inverse
# transform, one method per grid type:
#   FFTW extension        StructuredGrid{…,Cartesian}   (uniform Cartesian)
#   engines/NUFFTSpectral UnstructuredGrid{Cartesian}   (scattered Cartesian, through FlowTransformBindings)
#   SHT extension         StructuredGrid{…,Spherical}   (ClenshawCurtisSampling at nlon = 2·nlat − 1)
#   NUFSHT extension      UnstructuredGrid{Spherical}   (scattered spherical)
# Every other grid is filtered over its cells as a node set (engines/NodeSpectral).
function spectral_filter_plan(spectral_backend, grid, kernel, scale; kwargs...)
    throw(ArgumentError(
        "Spectral filtering with $(typeof(spectral_backend)) is unavailable for $(typeof(grid)) — " *
        "load a spectral backend (`using FFTW` uniform Cartesian, `using NonuniformFFTs` or " *
        "`using FINUFFT` scattered Cartesian, `using FastSphericalHarmonics` uniform spherical, " *
        "`using NUFSHT` scattered spherical).",
    ))
end

"""
    spectral_grid_plan(spectral_backend, grid, kernel; mask_strategy, execution, batch) -> AbstractGridPlan or nothing

The scale-independent half of a spectral plan: the transform objects themselves, the wavenumber grids,
and the mask. Only the transfer function `Ĝ(|k|, ℓ)` and the `Deformable` renormalization depend on
the filter scale, so a sweep builds this once and each scale keeps only those two. The transforms run on
`_library_threads(execution)` threads.

Returns `nothing` for a backend with no grid plan; a sweep then builds a full plan per scale.
"""
spectral_grid_plan(spectral_backend, grid, kernel; kwargs...) = nothing

"""
    spectral_scratch(grid_plan) -> AbstractFilterScratch or nothing

The transient half of a spectral plan — the complex spectrum buffer, the `mask · field` staging array,
and whatever else the apply overwrites — sized from `grid_plan` (see [`spectral_grid_plan`](@ref)). A
sweep hands one to every scale; a driver that applies concurrently builds one per worker.

Returns `nothing` for a backend that keeps no apply-time buffers.
"""
spectral_scratch(grid_plan) = nothing
