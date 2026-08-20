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
    BT,
} <: CGEF.Filtering.AbstractFilterPlan
    fwd::FP        # plan_rfft
    inv::IP        # plan_irfft
    transfer::A    # Ĝ(|k|, ℓ) on the rfft grid  (Nx÷2+1, Ny)
    cbuf::CA       # reusable complex spectrum buffer
    mask::M        # BitMatrix, or nothing when fully active (no masking overhead at all)
    masked_input::A   # scratch buffer for `mask .* field`; unused (and unallocated-cost) when mask === nothing
    invrenorm::R   # precomputed 1/filter(mask) for Deformable, or nothing (ZeroFill / fully active)
    # Transforms for a batch of fields, or nothing when the plan was not built for one. A transform is
    # bound to one field shape, so the batch extent is a construction parameter rather than something
    # discovered at apply time: a plan is shared across tasks by the batch drivers, so it must not
    # acquire state while being applied.
    batched::BT
end

function _make_batched_parts(masked_input::AbstractMatrix{T}, Nb::Integer) where {T<:AbstractFloat}
    buf = similar(masked_input, size(masked_input)..., Int(Nb))
    fwd = FFTW.plan_rfft(buf, (1, 2))
    cbuf = fwd * buf
    inv = FFTW.plan_irfft(cbuf, size(buf, 1), (1, 2))
    return (fwd = fwd, inv = inv, cbuf = cbuf, masked_input = buf)
end

CGEF.Filtering._batched_fields(outs, plan::FFTWFilterPlan) =
    plan.batched !== nothing && ndims(first(outs)) == ndims(plan.masked_input) + 1

# The transform runs over the spatial region, so trailing axes ride along; the transfer function, the mask
# and the renormalization are spatial-only and broadcast across the batch unchanged.
function CGEF.Filtering.filter_apply_batched!(
    out::AbstractArray{T,3}, field::AbstractArray{T,3}, plan::FFTWFilterPlan{T},
) where {T<:AbstractFloat}
    size(out) == size(field) || throw(DimensionMismatch(
        "filter_apply_batched! got out $(size(out)) and field $(size(field))",
    ))
    (size(out, 1), size(out, 2)) == size(plan.masked_input) || throw(DimensionMismatch(
        "field's leading axes $((size(out, 1), size(out, 2))) do not match the plan's $(size(plan.masked_input))",
    ))
    p = plan.batched
    p === nothing && throw(ArgumentError(
        "this spectral plan was not built for a batch; pass `batch = Nb` to `plan_filter`",
    ))
    size(p.masked_input, 3) == size(out, 3) || throw(DimensionMismatch(
        "plan was built for a batch of $(size(p.masked_input, 3)), got $(size(out, 3))",
    ))
    if plan.mask === nothing
        LA.mul!(p.cbuf, p.fwd, field)
    else
        @. p.masked_input = plan.mask * field
        LA.mul!(p.cbuf, p.fwd, p.masked_input)
    end
    p.cbuf .*= plan.transfer
    LA.mul!(out, p.inv, p.cbuf)
    plan.invrenorm === nothing || (out .*= plan.invrenorm)
    return out
end

# Analysis is scale-independent, so a sweep transforms each field once and then only multiplies by each
# scale's transfer function and inverts. `masked_input` is the plan's own scratch and is not touched
# after analysis, so the mask is applied here rather than being redone per scale.
CGEF.Filtering.analyze_buffer(plan::FFTWFilterPlan, field::AbstractMatrix) = similar(plan.cbuf)
CGEF.Filtering.analyze_buffer(plan::FFTWFilterPlan, field::AbstractArray{<:Any,3}) =
    similar(plan.cbuf, size(plan.cbuf)..., size(field, 3))

function CGEF.Filtering.filter_analyze!(
    F̂::AbstractMatrix{Complex{T}}, field::AbstractMatrix{T}, plan::FFTWFilterPlan{T},
) where {T<:AbstractFloat}
    if plan.mask === nothing
        LA.mul!(F̂, plan.fwd, field)
    else
        @. plan.masked_input = plan.mask * field
        LA.mul!(F̂, plan.fwd, plan.masked_input)
    end
    return F̂
end

function CGEF.Filtering.filter_synthesize!(
    out::AbstractMatrix{T}, F̂::AbstractMatrix{Complex{T}}, plan::FFTWFilterPlan{T},
) where {T<:AbstractFloat}
    # `plan.inv` consumes its input, so synthesize from a copy — `F̂` is reused by every later scale.
    plan.cbuf .= F̂ .* plan.transfer
    LA.mul!(out, plan.inv, plan.cbuf)
    plan.invrenorm === nothing || (out .*= plan.invrenorm)
    return out
end

function CGEF.Filtering.spectral_filter_plan(
    ::Union{CGEF.SpectralBackends.AbstractAutoSpectralBackend, CGEF.SpectralBackends.AbstractFFTSpectralBackend},
    grid::FlowGeometries.Grids.StructuredGrid{G,T},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy = CGEF.Filtering.ZeroFill(),
    backend = CGEF.ComputationalBackends.AutoBackend(),
    # Extent of the trailing batch axis this plan will be applied over, or `nothing` for single fields.
    # A transform is bound to one field shape, so it is fixed here rather than discovered at apply time.
    batch::Union{Nothing,Integer} = nothing,
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
        return FFTWFilterPlan(
            fwd, inv, transfer, cbuf, nothing, masked_input, nothing,
            batch === nothing ? nothing : _make_batched_parts(masked_input, batch),
        )
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
    return FFTWFilterPlan(
        fwd, inv, transfer, cbuf, mask, masked_input, invrenorm,
        batch === nothing ? nothing : _make_batched_parts(masked_input, batch),
    )
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
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy = CGEF.Filtering.ZeroFill(),
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
