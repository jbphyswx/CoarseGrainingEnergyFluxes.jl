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

# ---------------------------------------------------------------------------
# Padded-FFT real-space engine
# ---------------------------------------------------------------------------
#
# A non-separable kernel has no factored form, so the direct engine enumerates a whole disk per point:
# O(N·w²), and `SharpSpectralKernel`'s radius is 10ℓ. Zero-padding to `N + 2w` makes the circular
# convolution equal the LINEAR one, so this computes exactly what the direct sum computes — including
# on bounded and masked domains, where a periodic transform would be wrong.
#
# `Deformable`/`ZeroFill` are normalized convolution: num = conv(mask·f, g), den = conv(mask, g). `den`
# depends only on grid/kernel/scale/mask, so it is built once here rather than per apply.
struct PaddedFFTFootprint{T<:AbstractFloat, A<:AbstractMatrix{T}, C<:AbstractMatrix{Complex{T}}, FP, IP, S}
    Ĝ::C            # kernel spectrum on the padded grid
    den::A          # the normalization, cropped and precomputed — it depends on the mask STRATEGY
    pad::A          # padded scratch for mask·field
    num::A          # padded scratch for the inverse transform
    spec::C         # spectrum scratch
    fwd::FP
    inv::IP
    strategy::S     # the strategy `den` was built for; applying another one would be wrong
    N::NTuple{2,Int}
    P::Int
end

function CGEF.Filtering.padded_fft_footprint(
    grid::FlowGeometries.Grids.StructuredGrid{G,T,2},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy = CGEF.Filtering.Deformable(),
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    dx = step(FlowGeometries.Grids.coordinates(grid, 1))
    dy = step(FlowGeometries.Grids.coordinates(grid, 2))
    rad = CGEF.Kernels.kernel_radius(kernel, scale)
    wx = ceil(Int, rad / dx)
    wy = ceil(Int, rad / dy)
    P = max(Nx + 2wx, Ny + 2wy)
    A = FlowGeometries.Grids.area(grid, 1, 1)

    # Kernel field, wrapped so index (1,1) is the zero offset — otherwise the result is shifted. Gated
    # on `d <= rad`: the footprint is a DISK, and including the enclosing box's corners changes a
    # slowly-decaying kernel's answer outright.
    gpad = zeros(T, P, P)
    for dj in (-wy):wy, di in (-wx):wx
        d = hypot(di * dx, dj * dy)
        d <= rad || continue
        wt = CGEF.Kernels.kernel_weight(kernel, d, scale) * A
        iszero(wt) && continue
        gpad[mod1(1 + di, P), mod1(1 + dj, P)] += wt
    end

    fwd = FFTW.plan_rfft(gpad)
    Ĝ = fwd * gpad
    spec = similar(Ĝ)
    inv = FFTW.plan_irfft(spec, P)

    # The denominator differs by strategy, exactly as it does in the direct engine: `Deformable`
    # renormalizes over ACTIVE cells, `ZeroFill` keeps a masked neighbour in the denominator and
    # contributes nothing for it, so its denominator is the in-domain kernel mass.
    maskv = FlowGeometries.Grids.mask(grid)
    dpad = zeros(T, P, P)
    @inbounds for j in 1:Ny, i in 1:Nx
        dpad[i, j] = (mask_strategy isa CGEF.Filtering.ZeroFill || maskv[i, j]) ? one(T) : zero(T)
    end
    spec = fwd * dpad
    spec .*= Ĝ
    denfull = inv * spec
    den = Array{T}(undef, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        den[i, j] = denfull[i, j]
    end

    return PaddedFFTFootprint(
        Ĝ, den, zeros(T, P, P), zeros(T, P, P), similar(Ĝ), fwd, inv, mask_strategy, (Nx, Ny), P,
    )
end

function CGEF.Filtering.apply_footprint!(
    out::AbstractMatrix{T}, field::AbstractMatrix,
    grid::FlowGeometries.Grids.StructuredGrid{G,T,2},
    fp::PaddedFFTFootprint{T}, strategy::CGEF.Filtering.AbstractMaskStrategy,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    strategy === fp.strategy || throw(ArgumentError(
        "this padded-FFT footprint was built for $(typeof(fp.strategy)); its denominator is not the " *
        "one $(typeof(strategy)) needs. Rebuild the plan with the strategy you intend to apply.",
    ))
    Nx, Ny = fp.N
    maskv = FlowGeometries.Grids.mask(grid)
    fill!(fp.pad, zero(T))
    @inbounds for j in 1:Ny, i in 1:Nx
        fp.pad[i, j] = maskv[i, j] ? T(field[i, j]) : zero(T)
    end
    # In place through the held buffers: `plan * array` allocates a fresh result on every apply.
    LA.mul!(fp.spec, fp.fwd, fp.pad)
    fp.spec .*= fp.Ĝ
    LA.mul!(fp.num, fp.inv, fp.spec)
    @inbounds for j in 1:Ny, i in 1:Nx
        d = fp.den[i, j]
        out[i, j] = (maskv[i, j] && d > T(1e-15)) ? fp.num[i, j] / d : zero(T)
    end
    return out
end

CGEF.Filtering._apply_serial!(out, field, grid, fp::PaddedFFTFootprint, strategy) =
    CGEF.Filtering.apply_footprint!(out, field, grid, fp, strategy)

end # module
