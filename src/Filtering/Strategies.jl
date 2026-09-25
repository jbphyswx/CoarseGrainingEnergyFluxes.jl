# ---------------------------------------------------------------------------
# Masking strategy (singleton types — specializable, unlike Symbol dispatch)
# ---------------------------------------------------------------------------

"""
    AbstractMaskStrategy

How masked (inactive) cells enter the filter normalization.
"""
abstract type AbstractMaskStrategy end

"""
    ZeroFill <: AbstractMaskStrategy

Masked cells and the domain's exterior are zero-velocity water (Aluie et al. 2018; Storer et al.
2022): they contribute nothing to the numerator, and the kernel keeps its full mass, the cells past a
bounded edge included. The kernel is then the same everywhere, so filtering commutes with derivatives
and conserves the integral over all space, which on a periodic grid is the grid (Aluie 2019; Grooms
et al. 2021, eq. 7).

The filtered field is defined at every cell, a masked one too: within the kernel's reach of an active
cell it is nonzero over land.
"""
struct ZeroFill <: AbstractMaskStrategy end

"""
    Deformable <: AbstractMaskStrategy

Masked cells and the exterior are excluded from both numerator and denominator, so the kernel is
renormalized over the active cells in its window ("deformable kernel"), and a masked cell is zero. A
constant is reproduced exactly next to a boundary, but the kernel changes shape there, so filtering
neither commutes with derivatives nor conserves the domain integral.
"""
struct Deformable <: AbstractMaskStrategy end

# ---------------------------------------------------------------------------
# Filtering method: physical direct-sum (default) vs spectral (FFT/SHT/NUFFT via extensions)
# ---------------------------------------------------------------------------

"""
    AbstractFilterMethod

How the convolution is evaluated: [`RealSpace`](@ref) (physical-space footprint, any grid/mask) or
[`Spectral`](@ref) (transform-space multiply — FFT for uniform periodic Cartesian, spherical
harmonics for the uniform sphere, NUFFT/NUFSHT for scattered points; provided by extensions).
"""
abstract type AbstractFilterMethod end

"Physical-space direct-sum convolution (works on any grid, mask, and geometry)."
struct RealSpace <: AbstractFilterMethod end

"""
    Spectral <: AbstractFilterMethod

Transform-space filtering (kernel applied as a multiply on the transformed field). Requires a
spectral extension and a compatible grid (e.g. `using FFTW` for a uniform, periodic Cartesian grid).
"""
struct Spectral <: AbstractFilterMethod end

"""
    AutoMethod <: AbstractFilterMethod

The [`RealSpace`](@ref) convolution, evaluated by the fastest engine that computes its sum: the direct
engines, or a transform of the same sampled kernel where the grid allows one — the FFT engine on a
uniform Cartesian grid (circular along periodic axes, zero-padded along bounded ones) and the
transform along the longitude ring of a global rectilinear sphere. These agree with the direct sum to
round-off. [`Spectral`](@ref), which multiplies by the kernel's transfer function, is a different
discretization and is never selected.
"""
struct AutoMethod <: AbstractFilterMethod end

"""
    AbstractFilterPlan

A prebuilt filter (grid + kernel + scale + mask strategy + backend) that can be applied to many
fields without redoing setup. Physical-space backends precompute a `FilterFootprint`; the spectral
engines hold cached transform plans. Declared ahead of `filter_field!`, whose
`filter_plan::Union{Nothing,AbstractFilterPlan}` keyword annotation, just below, names it.
"""
abstract type AbstractFilterPlan end

# A plan owns transform-library plan objects, whose own `show` can walk the library's internal plan
# tree and call into a C library from wherever the plan happens to be printed, including a worker
# that does not own it. The type name is what a plan usefully prints.
Base.show(io::IO, p::AbstractFilterPlan) = print(io, nameof(typeof(p)), "(…)")
Base.show(io::IO, ::MIME"text/plain", p::AbstractFilterPlan) = show(io, p)
