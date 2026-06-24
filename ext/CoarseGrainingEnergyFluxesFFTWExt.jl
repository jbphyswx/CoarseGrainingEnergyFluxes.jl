module CoarseGrainingEnergyFluxesFFTWExt

using FFTW: FFTW
using LinearAlgebra: LinearAlgebra as LA
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF

# Spectral (FFT) filtering for uniform, doubly-periodic Cartesian grids: ĝ·f̂ is a pointwise
# multiply in Fourier space, so cost is O(N log N) and INDEPENDENT of the filter scale (vs the
# direct sum's O(N · footprint)). Plans + the transfer-function array are built once and reused.
#
# The filter is applied as a multiply by the kernel's spectral transfer function Ĝ(|k|, ℓ),
# normalized to 1 at k = 0 (so the domain mean is preserved). Masking is NOT applied (FFT assumes a
# homogeneous periodic domain); use the direct-sum method for masked/regional grids.

# Isotropic transfer functions Ĝ(|k|, ℓ).
@inline _transfer(k::CGEF.GaussianKernel, kmag::T, ℓ::T) where {T} = exp(-kmag^2 * ℓ^2 / (T(4) * T(k.α)))
@inline _transfer(::CGEF.SharpSpectralKernel, kmag::T, ℓ::T) where {T} = kmag <= T(π) / ℓ ? one(T) : zero(T)
@inline function _transfer(::CGEF.TopHatKernel, kmag::T, ℓ::T) where {T}
    throw(ArgumentError(
        "Spectral filtering with TopHatKernel is not supported (its 2D transfer function is an " *
        "oscillatory Airy pattern that rings). Use `method = DirectSum()` for the top-hat, or a " *
        "GaussianKernel / SharpSpectralKernel for spectral filtering.",
    ))
end

"""
    FFTWFilterPlan

Cached FFT filter plan: forward/inverse real-FFT plans, the precomputed transfer-function array,
and a reusable complex spectrum buffer. Built by `plan_filter(...; method = Spectral())`.
"""
struct FFTWFilterPlan{
    T<:AbstractFloat,
    FP,
    IP,
    A<:AbstractMatrix{T},
    CA<:AbstractMatrix{Complex{T}},
} <: CGEF.Filtering.AbstractFilterPlan
    fwd::FP        # plan_rfft
    inv::IP        # plan_irfft
    transfer::A    # Ĝ(|k|, ℓ) on the rfft grid  (Nlon÷2+1, Nlat)
    cbuf::CA       # reusable complex spectrum buffer
end

function CGEF.Filtering.spectral_filter_plan(
    grid::CGEF.StructuredGrid{G,T},
    kernel::CGEF.AbstractFilterKernel,
    scale::T;
    mask_strategy = CGEF.Deformable(),
    backend = CGEF.AutoBackend(),
) where {T<:AbstractFloat, G<:CGEF.CartesianGeometry{T}}
    (CGEF.isperiodic(grid, 1) && CGEF.isperiodic(grid, 2)) || throw(ArgumentError(
        "Spectral FFT filtering requires a doubly-periodic Cartesian grid; build it with " *
        "`StructuredGrid(geom, x, y, mask; periodic = (true, true))`.",
    ))
    all(grid.mask) || throw(ArgumentError("Spectral FFT filtering does not support land masks; use method = DirectSum()."))

    Nlon, Nlat = size(grid.mask)
    dx = grid.geometry.dx
    dy = grid.geometry.dy
    # Angular wavenumbers (rfft halves the first axis).
    kx = T(2π) .* FFTW.rfftfreq(Nlon, one(T) / dx)
    ky = T(2π) .* FFTW.fftfreq(Nlat, one(T) / dy)
    transfer = T[_transfer(kernel, sqrt(kx[i]^2 + ky[j]^2), scale) for i in eachindex(kx), j in eachindex(ky)]

    sample = zeros(T, Nlon, Nlat)
    fwd = FFTW.plan_rfft(sample)
    cbuf = fwd * sample                 # complex spectrum (Nlon÷2+1, Nlat)
    inv = FFTW.plan_irfft(cbuf, Nlon)
    return FFTWFilterPlan(fwd, inv, transfer, cbuf)
end

function CGEF.Filtering.filter_apply!(
    out::AbstractMatrix{T},
    field::AbstractMatrix{T},
    plan::FFTWFilterPlan{T},
) where {T<:AbstractFloat}
    LA.mul!(plan.cbuf, plan.fwd, field)   # f̂ = rfft(field)
    plan.cbuf .*= plan.transfer           # ĝ · f̂
    LA.mul!(out, plan.inv, plan.cbuf)     # irfft  (consumes cbuf, rebuilt next call)
    return out
end

end # module
