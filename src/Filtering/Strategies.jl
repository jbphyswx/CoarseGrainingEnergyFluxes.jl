# ---------------------------------------------------------------------------
# Mask strategies
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
# Filtering methods
# ---------------------------------------------------------------------------

"""
    AbstractFilterMethod

How the convolution is evaluated: [`RealSpace`](@ref) (physical-space footprint, any grid/mask) or
[`Spectral`](@ref) (transform-space multiply, provided by extensions: an FFT on a uniform Cartesian
grid, zero-padded along a bounded axis; spherical harmonics on a Fejér first-rule sphere; a NUFFT or
NUFSHT on scattered points).
"""
abstract type AbstractFilterMethod end

"""
    RealSpace <: AbstractFilterMethod

The physical-space convolution with the compact kernel, on any grid, mask and geometry. The grid and
kernel decide the engine that evaluates its sum. On a host backend that includes a transform of the same
sampled kernel where the grid allows one: the FFT on a uniform Cartesian grid (circular along periodic
axes, zero-padded along bounded ones) and the transform along longitude on a sphere of uniform
longitude, which agree with the direct sum to round-off. [`Spectral`](@ref), which multiplies by the
kernel's transfer function, is a different discretization.
"""
struct RealSpace <: AbstractFilterMethod end

"""
    Spectral <: AbstractFilterMethod

Transform-space filtering (kernel applied as a multiply on the transformed field). Requires a
spectral extension and a compatible grid (e.g. `using FFTW` for a uniform Cartesian grid, periodic or
bounded).
"""
struct Spectral <: AbstractFilterMethod end

"""
    AbstractFilterPlan

A prebuilt filter (grid + kernel + scale + mask strategy + execution backend), applied to many fields
with no repeated setup. A physical-space plan holds a precomputed footprint; a spectral plan holds its
transform plans.
"""
abstract type AbstractFilterPlan end

# A plan prints its type name only: the `show` of a transform-library plan it owns can call into a C
# library, on a worker that does not own the plan too.
Base.show(io::IO, p::AbstractFilterPlan) = print(io, nameof(typeof(p)), "(…)")
Base.show(io::IO, ::MIME"text/plain", p::AbstractFilterPlan) = show(io, p)
