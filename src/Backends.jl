module Backends

# Execution-backend taxonomy. Two orthogonal concerns:
#   LOCAL compute backend  — what one process/rank computes on: SerialBackend, ThreadedBackend
#                            (OhMyThreads ext), GPUBackend{B} (KernelAbstractions ext).
#   DISTRIBUTION wrapper    — how work is split across processes, PARAMETRIC over the inner local
#                            backend: DistributedBackend{Inner} (Distributed ext), MPIBackend{Inner}
#                            (MPI ext). Composition expresses real HPC layouts:
#                            MPIBackend{ThreadedBackend} = hybrid MPI+threads,
#                            MPIBackend{GPUBackend{...}} = multi-GPU cluster,
#                            DistributedBackend{ThreadedBackend} = multithreaded workers.
# AutoBackend resolves to the best available local backend. Heavy implementations live in
# extensions; this module only defines the dispatch types + helpers, so the core has no
# parallel/GPU dependencies.
#
# NOTE on naming: backend TYPES use the `…Backend` suffix specifically so they never collide with
# the packages loaded in the extensions (the stdlib `Distributed`, `MPI.jl`, etc.).

export AbstractExecutionBackend, SerialBackend, ThreadedBackend, GPUBackend, AutoBackend
export DistributedBackend, MPIBackend, local_backend, is_distributed, resolve_backend

"""
    AbstractExecutionBackend

Supertype for all execution backends — local compute backends (`SerialBackend`, `ThreadedBackend`,
`GPUBackend`) and distribution wrappers (`DistributedBackend`, `MPIBackend`).
"""
abstract type AbstractExecutionBackend end

"Serial single-threaded CPU compute (always available, no extension needed)."
struct SerialBackend <: AbstractExecutionBackend end

"Multithreaded CPU compute (requires `using OhMyThreads`)."
struct ThreadedBackend <: AbstractExecutionBackend end

"""
    GPUBackend{B}

GPU compute on KernelAbstractions backend object `B` (e.g. `CUDABackend()`). Requires the
KernelAbstractions extension (`using KernelAbstractions` + a GPU backend).
"""
struct GPUBackend{B} <: AbstractExecutionBackend
    backend::B
end

"Resolve to the best available local backend (`ThreadedBackend` if `Threads.nthreads() > 1`, else `SerialBackend`)."
struct AutoBackend <: AbstractExecutionBackend end

"""
    DistributedBackend(inner = SerialBackend())

Single-node multi-process execution (workers + `SharedArray`), each worker running `inner`
locally. Requires `using Distributed`. Parametric over the inner local backend, e.g.
`DistributedBackend(ThreadedBackend())` for multithreaded workers.
"""
struct DistributedBackend{Inner<:AbstractExecutionBackend} <: AbstractExecutionBackend
    inner::Inner
end
DistributedBackend() = DistributedBackend(SerialBackend())

"""
    MPIBackend(inner = SerialBackend())

Multi-node (distributed-memory) execution via domain decomposition + halo exchange, each rank
running `inner` locally. Requires `using MPI`. Not CPU-only: `MPIBackend(GPUBackend(...))` targets
multi-GPU clusters and `MPIBackend(ThreadedBackend())` is hybrid MPI+threads.
"""
struct MPIBackend{Inner<:AbstractExecutionBackend} <: AbstractExecutionBackend
    inner::Inner
end
MPIBackend() = MPIBackend(SerialBackend())

"""
    local_backend(backend) -> AbstractExecutionBackend

The per-process compute backend: the wrapped `inner` for a distribution wrapper, else the backend
itself.
"""
local_backend(b::AbstractExecutionBackend) = b
local_backend(b::DistributedBackend) = b.inner
local_backend(b::MPIBackend) = b.inner

"`true` if `backend` distributes work across processes."
is_distributed(::AbstractExecutionBackend) = false
is_distributed(::DistributedBackend) = true
is_distributed(::MPIBackend) = true

"""
    resolve_backend(backend) -> AbstractExecutionBackend

Resolve `AutoBackend` to a concrete local backend instance; all other backends are returned as-is.
"""
resolve_backend(backend::AbstractExecutionBackend) = backend
resolve_backend(::AutoBackend) = Threads.nthreads() > 1 ? ThreadedBackend() : SerialBackend()

end # module
