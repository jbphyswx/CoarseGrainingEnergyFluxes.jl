# ---------------------------------------------------------------------------
# Spectrum admissibility policy
# ---------------------------------------------------------------------------

"""
    AbstractSpectrumPolicy

What to do when a filtering spectral density is asked for with a kernel whose `|Ĝ(k)|²` is not monotone
decreasing — [`AutoSpectrum`](@ref), [`StrictSpectrum`](@ref), [`ForceSpectrum`](@ref) or
[`NoSpectrum`](@ref).

Sadek & Aluie (2018) eq. (21) guarantees `Ẽ(k_ℓ) ≥ 0` when `d|Ĝ(k)|²/dk ≤ 0`, a sufficient
condition. `TopHatKernel`'s `|Ĝ|²` falls to zero at `kℓ ≈ 7.66` and rises to `0.0175` at
`kℓ ≈ 10.27`: its violation lies in the sub-filter tail, below 2% of the DC value.

`Π` and the cumulative energy carry no such condition and are unaffected by this choice.
"""
abstract type AbstractSpectrumPolicy end

"""
    AutoSpectrum <: AbstractSpectrumPolicy

The default of the `coarse_grain` family: the density for a kernel that passes
[`Kernels.transfer_monotone`](@ref), `NaN` for one that does not, as [`NoSpectrum`](@ref) gives.
"""
struct AutoSpectrum <: AbstractSpectrumPolicy end

"""
    StrictSpectrum <: AbstractSpectrumPolicy

Refuse to produce a spectral density for a kernel that fails [`Kernels.transfer_monotone`](@ref):
either a density guaranteed non-negative, or an error naming the alternatives. The default of
[`filtering_spectrum`](@ref), whose result is that density.
"""
struct StrictSpectrum <: AbstractSpectrumPolicy end

"""
    ForceSpectrum <: AbstractSpectrumPolicy

Compute the density regardless, warning once per session; the caller checks its sign. Suited to a
kernel whose non-monotone band lies outside the scales being interpreted.
"""
struct ForceSpectrum <: AbstractSpectrumPolicy end

"""
    NoSpectrum <: AbstractSpectrumPolicy

Skip the density and fill the field with `NaN`. `Π` and the cumulative energy are still produced.
"""
struct NoSpectrum <: AbstractSpectrumPolicy end

"""
    gate_spectrum(kernel, policy) -> Bool

Apply `policy` to `kernel`, returning whether a spectral density should be computed. Throws under
[`StrictSpectrum`](@ref) for a non-monotone kernel; warns once under [`ForceSpectrum`](@ref).
"""
function gate_spectrum end

gate_spectrum(kernel, ::StrictSpectrum) = (Kernels.check_spectrum_kernel(kernel); true)
gate_spectrum(::Any, ::NoSpectrum) = false
gate_spectrum(kernel, ::AutoSpectrum) = Kernels.transfer_monotone(kernel)

"""
    resolve_spectrum(kernel, policy) -> AbstractSpectrumPolicy

`policy` with [`AutoSpectrum`](@ref) replaced by the policy it means for `kernel`:
[`StrictSpectrum`](@ref) for a kernel that passes [`Kernels.transfer_monotone`](@ref), [`NoSpectrum`](@ref)
otherwise.
"""
resolve_spectrum(::Any, policy::AbstractSpectrumPolicy) = policy
resolve_spectrum(kernel, ::AutoSpectrum) =
    Kernels.transfer_monotone(kernel) ? StrictSpectrum() : NoSpectrum()

function gate_spectrum(kernel, ::ForceSpectrum)
    Kernels.transfer_monotone(kernel) || @warn(
        "ForceSpectrum with $(nameof(typeof(kernel))): its |Ĝ(k)|² is not monotone decreasing, so " *
        "Sadek & Aluie (2018) eq. (21) does not apply and the filtering spectral density is not " *
        "guaranteed non-negative. Check its sign before interpreting it.",
        maxlog = 1,
    )
    return true
end

"""
    Placement(grid; execution = SerialBackend())

Where a diagnostic workspace lives: the local backend its loops run under, and copies of `grid`, as
given and with every cell active, held in that backend's memory. Under `MPIBackend` each rank runs its
inner backend; a `DistributedBackend`'s caller holds host memory, and runs a device inner backend's
loops serially.
"""
struct Placement{
    B<:ComputationalBackends.AbstractLocalBackend, G<:FlowGeometries.Grids.AbstractGrid,
    U<:FlowGeometries.Grids.AbstractGrid,
}
    execution::B
    grid::G
    unmasked::U
end

function Placement(
    grid::FlowGeometries.Grids.AbstractGrid;
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
)
    b = _loop_backend(Filtering._resolve_backend(execution, grid))
    g = FlowGeometries.Execution.on_backend(b, grid)
    return Placement(b, g, _all_active(g))
end

# `grid` with every cell active: itself when it has no inactive cell.
@inline _all_active(grid::FlowGeometries.Grids.AbstractGrid) =
    FlowGeometries.Grids.mask(grid) isa FlowGeometries.Grids.AllActive ? grid : _unmasked(grid)

@inline _loop_backend(b::ComputationalBackends.AbstractLocalBackend) = b
@inline _loop_backend(b::ComputationalBackends.AbstractMPIBackend) = ComputationalBackends.local_backend(b)
@inline _loop_backend(b::ComputationalBackends.AbstractDistributedBackend) =
    _caller_loops(ComputationalBackends.local_backend(b))
@inline _caller_loops(::ComputationalBackends.AbstractGPUBackend) = ComputationalBackends.SerialBackend()
@inline _caller_loops(b::ComputationalBackends.AbstractLocalBackend) = b

# A zeroed buffer in the placement's memory.
@inline _zeros(at::Placement, ::Type{T}, dims::Tuple) where {T} =
    Filtering._allocate(at.execution, T, map(Int, dims))

# The grid a plan's outputs are read on, as held by the placement: see `output_grid`.
@inline _output(at::Placement, plan::Filtering.AbstractFilterPlan) = _output(at, Filtering.plan_strategy(plan))
@inline _output(at::Placement, ::Filtering.ZeroFill) = at.unmasked
@inline _output(at::Placement, ::Filtering.AbstractMaskStrategy) = at.grid

# `x`, a derivative plan or another value holding arrays, where the placement's loops read it.
@inline _placed(at::Placement, x) = FlowGeometries.Execution.on_backend(at.execution, x)

# A workspace's buffers and a plan's outputs are written by the same loops, so both are host memory or
# both are one device's.
@inline function _check_placement(at::Placement, plan::Filtering.AbstractFilterPlan)
    Filtering._host_memory(at.execution) == Filtering._host_memory(Filtering.plan_execution(plan)) ||
        _placement_mismatch()
    return nothing
end

@noinline _placement_mismatch() = throw(ArgumentError(
    "the workspace and the filter plan live in different memory, one on the host and the other on a " *
    "device; build the workspace with `execution = Filtering.plan_execution(plan)`",
))

"""
    ΠWorkspace{T, A}

Pre-allocated arrays for computing cross-scale energy flux Π to avoid heap allocations in scale loops.
"""
struct ΠWorkspace{
    T<:AbstractFloat, A<:AbstractArray{T},
    AZ<:Union{Nothing,AbstractArray{T}},   # vertical/third-component buffers
    AS<:Union{Nothing,AbstractArray{T}},   # extra buffers of the ambient Cartesian-component filter
    SF,                                    # the generalized convolution, a `SpinFilter`, or nothing
    P<:Placement,
}
    # Filtered velocity components (local coordinates)
    u_filt::A
    v_filt::A
    w_filt::AZ

    # Ambient Cartesian velocities under the Cartesian-component filter on a non-Cartesian geometry
    ux::A
    uy::A
    uz::A
    ux_filt::AZ
    uy_filt::AZ
    uz_filt::AZ

    # Filtered quadratic velocity products (ambient Cartesian on a non-Cartesian geometry)
    uu_filt::A
    uv_filt::A
    uw_filt::AZ
    vv_filt::A
    vw_filt::AZ
    ww_filt::AZ

    # Velocity derivatives / strain rate components (local coordinates)
    S_xx::A
    S_xy::A
    S_xz::AZ
    S_yy::A
    S_yz::AZ
    S_zz::AZ

    # Subfilter-scale stress components (local coordinates)
    τ_xx::A
    τ_xy::A
    τ_xz::AZ
    τ_yy::A
    τ_yz::AZ
    τ_zz::AZ

    # Under the Cartesian-component filter on a non-Cartesian geometry six products are filtered in one
    # batched apply: their inputs occupy these three and `u_filt`/`v_filt`/`w_filt`, before the rotation to
    # the local frame overwrites those.
    scratch::A
    scratch2::AS
    scratch3::AS

    spin::SF

    at::P
end

"""
    ΠWorkspace(grid, batch_size = (); has_w = false, execution = SerialBackend(), vector_filter = nothing,
               transform = AutoSpectralBackend(), nufft = AutoSpectralBackend())

Scratch for a flux computation, in the memory of `execution`, whose loops the flux runs under (see
[`Placement`](@ref)). `batch_size` sizes every buffer as `(spatial..., batch...)`, holding a whole batch
of slices at once.

On the surface of a sphere the workspace holds the transforms of the vector filter `vector_filter` names
([`Filtering.resolve_vector_filter`](@ref) under `ZeroFill`): `GeneralizedConvolution`'s spin transforms,
chosen by `transform` and `nufft`, by default. A plan filtering under `Deformable` needs
`vector_filter = Filtering.CartesianComponents()`.

Only the buffers the configuration reaches are allocated, fixed by the grid, the vector filter and
`has_w`:

| configuration | full-size buffers |
|---|---|
| 2-D Cartesian, or a sphere under `GeneralizedConvolution`, no `w` | 15 |
| either with `w` (the 2.5-D path), or true 3-D Cartesian | 28 |
| `CartesianComponents` on another geometry (a sphere, a spheroid, a shell) | 30 |

`has_w` must be set at construction; a workspace built without the vertical buffers throws when given
a `w`. The last row and true-3-D grids always carry the vertical set.
"""
@inline ΠWorkspace(
    grid::FlowGeometries.Grids.AbstractGrid, batch_size::Tuple = (); has_w::Bool = false,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
    vector_filter::Union{Nothing,Filtering.AbstractVectorFilter} = nothing,
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    nufft::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
) = ΠWorkspace(grid, batch_size, Val(has_w); execution, vector_filter, transform, nufft)

# `Val(has_w)` puts the choice in the type, so a caller that knows `w`'s type gets a concrete workspace.
function ΠWorkspace(
    grid::FlowGeometries.Grids.AbstractGrid{G,T}, batch_size::Tuple, ::Val{W};
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
    vector_filter::Union{Nothing,Filtering.AbstractVectorFilter} = nothing,
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    nufft::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
) where {G, T<:AbstractFloat, W}
    at = Placement(grid; execution = execution)
    spin = _workspace_spin(grid, at, vector_filter, transform, nufft)
    sz = (FlowGeometries.Grids.size_tuple(grid)..., batch_size...)
    ambient = !(G <: FlowGeometries.Geometry.CartesianGeometry) && spin === nothing
    rank = length(FlowGeometries.Grids.size_tuple(grid))
    vertical = W || ambient || rank == 3

    z() = _zeros(at, T, sz)
    zv() = vertical ? z() : nothing
    zs() = ambient ? z() : nothing

    return ΠWorkspace(
        z(), z(), zv(),
        z(), z(), z(), zv(), zv(), zv(),
        z(), z(), zv(), z(), zv(), zv(),
        z(), z(), zv(), z(), zv(), zv(),
        z(), z(), zv(), z(), zv(), zv(),
        z(), zs(), zs(), spin, at,
    )
end

"""
    EnergyWorkspace(grid, batch_size = (); has_w = false, execution = SerialBackend(), vector_filter = nothing,
                    transform = AutoSpectralBackend(), nufft = AutoSpectralBackend())

Scratch for the diagnostics that read only the filtered velocity, `cumulative_energy!` and through it
`filtering_spectrum`, in the memory of `execution` (see [`Placement`](@ref)): two buffers, three with
a vertical component. Interchangeable with [`ΠWorkspace`](@ref) wherever only the velocity buffers are
used.

On the surface of a sphere it holds the vector filter's transforms as [`ΠWorkspace`](@ref) does. Under
`CartesianComponents` on a grid of non-Cartesian geometry it also holds the ambient Cartesian
components, `ux/uy/uz`, and all three filtered buffers.
"""
struct EnergyWorkspace{
    T<:AbstractFloat, A<:AbstractArray{T}, AW<:Union{Nothing,AbstractArray{T}},
    AP<:Union{Nothing,AbstractArray{T}}, SF, P<:Placement,
}
    u_filt::A
    v_filt::A
    w_filt::AW
    ux::AP
    uy::AP
    uz::AP
    spin::SF     # the generalized convolution, a `SpinFilter`, or nothing
    at::P
end

function EnergyWorkspace(
    grid::FlowGeometries.Grids.AbstractGrid{G,T}, batch_size::Tuple = ();
    has_w::Bool = false,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
    vector_filter::Union{Nothing,Filtering.AbstractVectorFilter} = nothing,
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    nufft::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
) where {G, T<:AbstractFloat}
    at = Placement(grid; execution = execution)
    spin = _workspace_spin(grid, at, vector_filter, transform, nufft)
    sz = (FlowGeometries.Grids.size_tuple(grid)..., batch_size...)
    ambient = !(G <: FlowGeometries.Geometry.CartesianGeometry) && spin === nothing
    z() = _zeros(at, T, sz)
    za() = ambient ? z() : nothing
    return EnergyWorkspace(z(), z(), (has_w || ambient) ? z() : nothing, za(), za(), za(), spin, at)
end

# The ambient Cartesian input buffers and the three filtered ones they are filtered into.
@inline _ambient_buffers(ws::Union{ΠWorkspace, EnergyWorkspace}) =
    ((ws.ux, ws.uy, ws.uz), (ws.u_filt, ws.v_filt, ws.w_filt))

# Whether a workspace carries the third-component buffers, read off its type.
@inline _has_vertical(::ΠWorkspace{T,A,AZ}) where {T,A,AZ} = AZ !== Nothing
@inline _has_vertical(::EnergyWorkspace{T,A,AW}) where {T,A,AW} = AW !== Nothing

@noinline function _workspace_missing_vertical()
    throw(ArgumentError(
        "this ΠWorkspace was built without vertical-component buffers, but a vertical velocity `w` was " *
        "supplied. Which buffers exist is fixed when the workspace is built, so rebuild it as " *
        "`ΠWorkspace(grid; has_w = true)` (or `ΠWorkspace(grid, batch_size; has_w = true)`).",
    ))
end

@inline function _check_workspace_w(ws::Union{ΠWorkspace, EnergyWorkspace}, w)
    (w === nothing || _has_vertical(ws)) || _workspace_missing_vertical()
    return nothing
end
