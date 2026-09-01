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

Excluded cells are treated as zero-valued: they contribute to the denominator (kernel weight) but
zero to the numerator. The kernel is homogeneous (same shape everywhere), which preserves domain
averages and commutation with derivatives (the Storer 2022 / Aluie 2019 "fixed kernel" mode).
"""
struct ZeroFill <: AbstractMaskStrategy end

"""
    Deformable <: AbstractMaskStrategy

Masked cells are excluded from BOTH numerator and denominator, so the kernel is renormalized over the
locally-included area only ("deformable kernel"). Excluded cells are genuinely dropped, but the kernel
becomes inhomogeneous near a mask boundary (breaks the strict commutation theorems).
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

Pick the engine from real capability, the same contract [`AutoCache`](@ref) and `AutoBackend` follow:
[`Spectral`](@ref) only where a transform is available AND exact for this grid, otherwise
[`RealSpace`](@ref).

Selects [`Spectral`](@ref) only where every axis is periodic and uniform and a transform backend is
loaded — the case where a periodic transform is the exact filter. Otherwise [`RealSpace`](@ref).
"""
struct AutoMethod <: AbstractFilterMethod end

"""
    AbstractFilterPlan

A prebuilt filter (grid + kernel + scale + mask strategy + backend) that can be applied to many
fields without redoing setup. Physical-space backends precompute a `FilterFootprint`; the spectral
extensions (FFTW/FINUFFT/SHT) hold cached transform plans. Declared here rather than alongside
`PhysicalFilterPlan` further down so that `filter_field!`'s `filter_plan::Union{Nothing,
AbstractFilterPlan}` keyword annotation, just below, can name it.
"""
abstract type AbstractFilterPlan end

# A plan owns FFTW/FINUFFT/SHT plan objects, whose own `show` walks the library's internal plan tree —
# 6 KB of output for a 16×16 transform, and a call into the C library from wherever the plan happens to
# be printed, including a worker that does not own it. The type name is what a plan usefully prints.
Base.show(io::IO, p::AbstractFilterPlan) = print(io, nameof(typeof(p)), "(…)")
Base.show(io::IO, ::MIME"text/plain", p::AbstractFilterPlan) = show(io, p)
