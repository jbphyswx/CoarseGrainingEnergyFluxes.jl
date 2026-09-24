# ---------------------------------------------------------------------------
# Spectrum admissibility policy (singleton types — specializable, same idiom as AbstractMaskStrategy)
# ---------------------------------------------------------------------------

"""
    AbstractSpectrumPolicy

What to do when a filtering spectral density is asked for with a kernel whose `|Ĝ(k)|²` is not monotone
decreasing — [`AutoSpectrum`](@ref), [`StrictSpectrum`](@ref), [`ForceSpectrum`](@ref) or
[`NoSpectrum`](@ref).

Sadek & Aluie (2018) eq. (21) guarantees `Ẽ(k_ℓ) ≥ 0` only when `d|Ĝ(k)|²/dk ≤ 0`. The condition is
sufficient, and a kernel that violates it usually does so narrowly: `TopHatKernel`'s `|Ĝ|²` falls to
zero at `kℓ ≈ 7.66` and climbs back to only `0.0175` at `kℓ ≈ 10.27`, a violation in the far sub-filter
tail at under 2% of the DC value.

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

Compute the density regardless, warning once per session. The caller owns checking its sign — the right
setting when the kernel's non-monotone band sits outside the range of scales being interpreted.
"""
struct ForceSpectrum <: AbstractSpectrumPolicy end

"""
    NoSpectrum <: AbstractSpectrumPolicy

Skip the density entirely; the field is filled with `NaN` rather than a number that would read as
computed. `Π` and the cumulative energy are still produced.
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
    ΠWorkspace{T, A}

Pre-allocated arrays for computing cross-scale energy flux Π to avoid heap allocations in scale loops.
"""
struct ΠWorkspace{
    T<:AbstractFloat, A<:AbstractArray{T},
    AZ<:Union{Nothing,AbstractArray{T}},   # vertical/third-component buffers
    AS<:Union{Nothing,AbstractArray{T}},   # extra pre-filter buffers the spherical branch needs
}
    # Filtered velocity components (local coordinates)
    u_filt::A
    v_filt::A
    w_filt::AZ

    # Planetary Cartesian velocities (if Spherical geometry is used)
    ux::A
    uy::A
    uz::A
    ux_filt::AZ
    uy_filt::AZ
    uz_filt::AZ

    # Filtered quadratic velocity products (planetary Cartesian if Spherical, else Cartesian)
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

    # Three scratch arrays, so `filter_apply_batch!` can filter all 6 quadratic velocity products in
    # one pass. The spherical branch needs 6 simultaneous pre-filter buffers and only 3 are idle at
    # that point (`u_filt`/`v_filt`/`w_filt`, before the planetary→local transform overwrites them).
    scratch::A
    scratch2::AS
    scratch3::AS
end

# Workspace constructor based on grid structure and float type. `A` is inferred from what
# `zeros(T, sz...)` actually produces (Vector for a 1D grid, Matrix for 2D, Array{T,3} for 3D) —
# NOT hardcoded, since a 1D/3D grid's `sz` is a 1- or 3-tuple, not always 2D.
"""
    ΠWorkspace(grid; has_w = false)
    ΠWorkspace(grid, batch_size; has_w = false)

Scratch for a flux computation. Passing `batch_size` sizes every buffer as `(spatial..., batch...)` so a
whole batch of slices is held at once; the elementwise algebra then broadcasts over the trailing axes
unchanged, and a filter apply can cover the batch in one pass instead of one per slice.

Only the buffers the requested configuration can reach are allocated, because which ones those are is
fixed by the grid and by whether a vertical component is supplied — not discovered at run time:

| configuration | full-size buffers |
|---|---|
| 2-D Cartesian, no `w` | 15 |
| 2-D Cartesian with `w` (the 2.5-D path) | 28 |
| spherical | 30 |
| true 3-D | 30 |

`has_w` is what selects between the first two, and it must be given at construction because the buffers
have to exist before the first call. A workspace built without them refuses a `w` rather than
returning a wrong answer, with a message naming the fix.

Spherical and true-3-D grids always carry the vertical set: the spherical branch works in planetary
Cartesian components, so it has three velocity components and six products whether or not the caller
supplied a vertical velocity.
"""
function ΠWorkspace(
    grid::FlowGeometries.Grids.AbstractGrid{G,T}, batch_size::Tuple = ();
    has_w::Bool = false,
) where {G, T<:AbstractFloat}
    sz = (FlowGeometries.Grids.size_tuple(grid)..., batch_size...)
    spherical = G <: FlowGeometries.Geometry.SphericalGeometry
    rank = length(FlowGeometries.Grids.size_tuple(grid))
    vertical = has_w || spherical || rank == 3

    z() = zeros(T, sz...)
    zv() = vertical ? zeros(T, sz...) : nothing
    zs() = spherical ? zeros(T, sz...) : nothing

    return ΠWorkspace(
        z(), z(), zv(),
        z(), z(), z(), zv(), zv(), zv(),
        z(), z(), zv(), z(), zv(), zv(),
        z(), z(), zv(), z(), zv(), zv(),
        z(), z(), zv(), z(), zv(), zv(),
        z(), zs(), zs(),
    )
end

"""
    EnergyWorkspace(grid; has_w = false)
    EnergyWorkspace(grid, batch_size; has_w = false)

Scratch for the diagnostics that only need FILTERED VELOCITY — `cumulative_energy!` and, through it,
`filtering_spectrum`. Those compute `E(ℓ) = ½⟨|ū_ℓ|²⟩`, which reads no stress, no strain and no
quadratic product, so two buffers serve (three with a vertical component).

Interchangeable with [`ΠWorkspace`](@ref) wherever only the velocity buffers are used, so a sweep that
already holds a flux workspace passes it straight through instead of allocating a second one.

On a grid of non-Cartesian geometry the velocity is filtered as its planetary Cartesian components
(Aluie 2019), so the workspace also holds those, `ux/uy/uz`, and all three filtered buffers.
"""
struct EnergyWorkspace{
    T<:AbstractFloat, A<:AbstractArray{T}, AW<:Union{Nothing,AbstractArray{T}},
    AP<:Union{Nothing,AbstractArray{T}},
}
    u_filt::A
    v_filt::A
    w_filt::AW
    ux::AP
    uy::AP
    uz::AP
end

function EnergyWorkspace(
    grid::FlowGeometries.Grids.AbstractGrid{G,T}, batch_size::Tuple = ();
    has_w::Bool = false,
) where {G, T<:AbstractFloat}
    sz = (FlowGeometries.Grids.size_tuple(grid)..., batch_size...)
    planetary = !(G <: FlowGeometries.Geometry.CartesianGeometry)
    z() = zeros(T, sz...)
    zp() = planetary ? z() : nothing
    return EnergyWorkspace(z(), z(), (has_w || planetary) ? z() : nothing, zp(), zp(), zp())
end

# The planetary input buffers and the three filtered ones they are filtered into.
@inline _planetary_buffers(ws::Union{ΠWorkspace, EnergyWorkspace}) =
    ((ws.ux, ws.uy, ws.uz), (ws.u_filt, ws.v_filt, ws.w_filt))

# Whether a workspace carries the third-component buffers. Read off the type, so the check costs
# nothing and cannot disagree with what was allocated.
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
