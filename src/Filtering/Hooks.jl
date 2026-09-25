# ---------------------------------------------------------------------------
# Extension hook points
# ---------------------------------------------------------------------------
# Fallbacks that error until the relevant backend extension is loaded. Each execution-backend
# extension overrides its hook; the public `filter_field!` dispatches here based on the resolved
# backend. (Backend TYPES live in `Backends`; these are the per-backend filtering implementations.)

function threaded_filter_field!(args...; kwargs...)
    throw(ArgumentError("ThreadedBackend is unavailable — run `using OhMyThreads` (or use SerialBackend())."))
end

# Batched form: several fields sharing one grid/kernel/scale. Separate from the single-field hook
# because the scattered footprints derive each point's neighbour list on the fly, and that derivation
# is shared across the batch — a per-field loop would repeat it once per field.
function threaded_filter_fields!(args...; kwargs...)
    throw(ArgumentError("ThreadedBackend is unavailable — run `using OhMyThreads` (or use SerialBackend())."))
end

# Slice-parallel form: many INDEPENDENT problems, one plan each. A different axis from
# `threaded_filter_fields!`, which shares one grid across several fields.
function threaded_filter_slices!(args...; kwargs...)
    throw(ArgumentError("ThreadedBackend is unavailable — run `using OhMyThreads` (or use SerialBackend())."))
end

# Padded-FFT real-space engine (FFTW extension). Zero-padding makes the transform compute the LINEAR
# convolution, so unlike a periodic transform it holds on bounded and masked domains.
function padded_fft_footprint(args...; kwargs...)
    throw(ArgumentError("The padded-FFT real-space engine needs FFTW — run `using FFTW`."))
end

# Zonal-FFT real-space engine (FFTW extension). Longitude on a global rectilinear sphere IS a ring, and
# for a fixed pair of latitudes a great-circle kernel depends on the longitude DIFFERENCE alone, so each
# latitude band is a circular convolution along that ring. Same compact kernel, same weights, evaluated
# by transform instead of by a tap loop.
function zonal_fft_footprint(args...; kwargs...)
    throw(ArgumentError("The zonal-FFT spherical engine needs FFTW — run `using FFTW`."))
end

# Whether `fp` is one of those two transform engines. They run on the host: serially, or under
# `ThreadedBackend` through `apply_footprint!(out, field, grid, fp, strategy, driver)` with a row
# driver of `_sep_serial`'s shape.
_transform_footprint(::Any) = false

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
    spectral_grid_plan(spectral_backend, grid, kernel; mask_strategy, backend, batch) -> AbstractGridPlan or nothing

The scale-independent half of a spectral plan: the transform objects themselves, the wavenumber grids,
and the mask. Only the transfer function `Ĝ(|k|, ℓ)` and the `Deformable` renormalization depend on
the filter scale, so a sweep builds this once and each scale keeps only those two. The transforms run on
`_library_threads(backend)` threads.

Planning a transform is not cheap — FFTW measures, and a nonuniform transform additionally sorts its
points — so paying it once per grid rather than once per scale is the whole reason this hook exists.

Returns `nothing` for a backend that has not been given one; the sweep then falls back to a full plan
per scale, which is correct, just not shared.
"""
spectral_grid_plan(spectral_backend, grid, kernel; kwargs...) = nothing

"""
    spectral_scratch(grid_plan) -> AbstractFilterScratch or nothing

The transient half of a spectral plan — the complex spectrum buffer, the `mask · field` staging array,
and whatever else the apply overwrites — sized from the grid plan the transforms were built against.

Splitting it out is what makes [`spectral_grid_plan`](@ref)'s result shareable: transforms and
wavenumber grids are read-only during an apply, these buffers are not. A sweep builds one and hands it
to every scale; a driver that applies concurrently builds one per worker.

Returns `nothing` for a backend that keeps no apply-time buffers.
"""
spectral_scratch(grid_plan) = nothing
