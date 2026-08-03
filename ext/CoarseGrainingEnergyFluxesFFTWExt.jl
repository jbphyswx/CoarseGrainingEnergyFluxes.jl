module CoarseGrainingEnergyFluxesFFTWExt

using FFTW: FFTW
using LinearAlgebra: LinearAlgebra as LA
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF
using FlowGeometries: FlowGeometries

# Spectral filtering for uniform, doubly-periodic Cartesian grids. Convolution is a pointwise multiply
# by the kernel's transfer function `Ĝ(|k|, ℓ)` (`CGEF.Kernels.spectral_transfer`, shared with the
# other spectral backends), so the cost is O(N log N) and independent of the filter scale. Plans and
# the transfer array are built once per plan. `Ĝ(0) = 1`, so the domain mean is preserved.
#
# Masking uses the normalized-convolution identity (Knutsson & Westin 1993) that `RealSpace`'s
# strategies implement pointwise: `ZeroFill` filters `mask·field` directly, and `Deformable`
# additionally divides by the local kernel mass over active cells, `filter(mask)`. The mask is fixed
# for a plan, so that denominator is computed at build time and stored inverted — one multiply per
# apply rather than a divide.

"""
    FFTWFilterPlan

Cached FFT filter plan: forward/inverse real-FFT plans, the precomputed transfer-function array,
a reusable complex spectrum buffer, and — for a masked grid — the mask itself, a scratch buffer for
`mask · field`, and (for `Deformable`) the precomputed inverse local-mass renormalization. Built by
`plan_filter(...; method = Spectral())`.
"""
struct FFTWFilterPlan{
    T<:AbstractFloat,
    FP,
    IP,
    A<:AbstractMatrix{T},
    CA<:AbstractMatrix{Complex{T}},
    M,
    R,
} <: CGEF.Filtering.AbstractFilterPlan
    fwd::FP        # plan_rfft
    inv::IP        # plan_irfft
    transfer::A    # Ĝ(|k|, ℓ) on the rfft grid  (Nx÷2+1, Ny)
    cbuf::CA       # reusable complex spectrum buffer
    mask::M        # BitMatrix, or nothing when fully active (no masking overhead at all)
    masked_input::A   # scratch buffer for `mask .* field`; unused (and unallocated-cost) when mask === nothing
    invrenorm::R   # precomputed 1/filter(mask) for Deformable, or nothing (ZeroFill / fully active)
end

function CGEF.Filtering.spectral_filter_plan(
    ::Union{CGEF.SpectralBackends.AbstractAutoSpectralBackend, CGEF.SpectralBackends.AbstractFFTSpectralBackend},
    grid::FlowGeometries.Grids.StructuredGrid{G,T},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy = CGEF.Filtering.Deformable(),
    backend = CGEF.ComputationalBackends.AutoBackend(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    (FlowGeometries.Grids.isperiodic(grid, 1) && FlowGeometries.Grids.isperiodic(grid, 2)) || throw(ArgumentError(
        "Spectral FFT filtering requires a doubly-periodic Cartesian grid; build it with " *
        "`StructuredGrid(geom, x, y, mask; periodic = (true, true))`.",
    ))

    Nx, Ny = size(FlowGeometries.Grids.mask(grid))
    # The transform's wavenumbers are `k = 2π·m/(N·dx) = 2π·m/L`, so the only spacing the FFT actually
    # needs is the domain PERIOD — which this plan already requires to exist, and which is well defined
    # for any axis representation. On a uniform axis `L/N` is exactly `step`.
    dx = FlowGeometries.Grids.period(grid, 1) / Nx
    dy = FlowGeometries.Grids.period(grid, 2) / Ny
    # Angular wavenumbers (rfft halves the first axis).
    kx = T(2π) .* FFTW.rfftfreq(Nx, one(T) / dx)
    ky = T(2π) .* FFTW.fftfreq(Ny, one(T) / dy)
    transfer = T[CGEF.Kernels.spectral_transfer(kernel, sqrt(kx[i]^2 + ky[j]^2), scale) for i in eachindex(kx), j in eachindex(ky)]

    sample = zeros(T, Nx, Ny)
    fwd = FFTW.plan_rfft(sample)
    cbuf = fwd * sample                 # complex spectrum (Nx÷2+1, Ny)
    inv = FFTW.plan_irfft(cbuf, Nx)

    fully_active = all(FlowGeometries.Grids.mask(grid))
    masked_input = zeros(T, Nx, Ny)   # allocated once regardless; only touched when mask !== nothing
    if fully_active
        return FFTWFilterPlan(fwd, inv, transfer, cbuf, nothing, masked_input, nothing)
    end

    mask = FlowGeometries.Grids.mask(grid)
    invrenorm = if mask_strategy isa CGEF.Filtering.Deformable
        # Local kernel mass over active cells, `filter(mask)` — the SAME transform machinery,
        # applied to the mask itself once, since the mask never changes across `filter_apply!` calls.
        masked_input .= mask            # reuse the apply-path scratch rather than a fresh Nx×Ny temporary
        mcbuf = fwd * masked_input
        mcbuf .*= transfer
        renorm = similar(sample)
        LA.mul!(renorm, inv, mcbuf)
        threshold = T(0.01)
        ir = similar(renorm)
        @. ir = ifelse(abs(renorm) >= threshold, one(T) / renorm, zero(T))
        ir
    else
        nothing   # ZeroFill: already exactly `filter(mask .* field)`, no renormalization
    end
    return FFTWFilterPlan(fwd, inv, transfer, cbuf, mask, masked_input, invrenorm)
end

function CGEF.Filtering.filter_apply!(
    out::AbstractMatrix{T},
    field::AbstractMatrix{T},
    plan::FFTWFilterPlan{T},
) where {T<:AbstractFloat}
    if plan.mask === nothing
        LA.mul!(plan.cbuf, plan.fwd, field)       # f̂ = rfft(field)
    else
        @. plan.masked_input = plan.mask * field
        LA.mul!(plan.cbuf, plan.fwd, plan.masked_input)
    end
    plan.cbuf .*= plan.transfer           # ĝ · f̂
    LA.mul!(out, plan.inv, plan.cbuf)     # irfft  (consumes cbuf, rebuilt next call)
    plan.invrenorm === nothing || (out .*= plan.invrenorm)
    return out
end

end # module
