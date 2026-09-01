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
    FFTWGridPlan

The half of an FFTW spectral plan the filter scale does not reach: the forward and inverse transforms,
the mask, the angular wavenumber grids, and — when the plan was built for a trailing batch axis — the
transforms bound to that shape. One instance serves every scale of a sweep, and it is immutable during
an apply, so concurrent workers may share it.

Only `transfer` — and, for `Deformable`, the renormalization computed through it — depends on ℓ; every
buffer written during an apply lives in [`FFTWScratch`](@ref). Planning an FFT means measuring, so it
is paid once per grid.
"""
struct FFTWGridPlan{T<:AbstractFloat, FP, IP, M, VX<:AbstractVector{T}, VY<:AbstractVector{T}, BT} <:
       CGEF.Filtering.AbstractGridPlan
    fwd::FP        # plan_rfft
    inv::IP        # plan_irfft
    mask::M        # BitMatrix, or nothing when fully active (no masking overhead at all)
    kx::VX
    ky::VY
    dims::NTuple{2,Int}
    batched::BT    # (; fwd, inv, nb) for a trailing batch axis, or nothing
end

"""
    FFTWScratch

The transient half of an FFTW spectral plan: the complex spectrum buffer and the `mask · field` staging
array, plus the same pair sized for a trailing batch axis. Every scale of a sweep shares one, since
scales run sequentially; a driver that applies concurrently gives each worker its own.
"""
struct FFTWScratch{T<:AbstractFloat, CA<:AbstractMatrix{Complex{T}}, A<:AbstractMatrix{T}, BT} <:
       CGEF.Filtering.AbstractFilterScratch
    cbuf::CA
    masked_input::A
    batched::BT    # (; cbuf, masked_input) sized for the batch axis, or nothing
end

"""
    FFTWFilterPlan

Cached FFT filter plan: the shared [`FFTWGridPlan`](@ref), the precomputed transfer-function array, the
`Deformable` local-mass renormalization (or `nothing`), and the [`FFTWScratch`](@ref) the apply writes
through. Built by `plan_filter(...; method = Spectral())`.
"""
struct FFTWFilterPlan{
    T<:AbstractFloat, GP<:FFTWGridPlan{T}, A<:AbstractMatrix{T}, R, SC<:FFTWScratch{T},
} <: CGEF.Filtering.AbstractFilterPlan
    grid_plan::GP
    transfer::A    # Ĝ(|k|, ℓ) on the rfft grid  (Nx÷2+1, Ny)
    invrenorm::R   # precomputed 1/filter(mask) for Deformable, or nothing (ZeroFill / fully active)
    scratch::SC
end

CGEF.Filtering._batched_fields(outs, plan::FFTWFilterPlan) =
    plan.grid_plan.batched !== nothing && ndims(first(outs)) == 3

# The transform runs over the spatial region, so trailing axes ride along; the transfer function, the mask
# and the renormalization are spatial-only and broadcast across the batch unchanged.
function CGEF.Filtering.filter_apply_batched!(
    out::AbstractArray{T,3}, field::AbstractArray{T,3}, plan::FFTWFilterPlan{T},
) where {T<:AbstractFloat}
    gp, sc = plan.grid_plan, plan.scratch
    size(out) == size(field) || throw(DimensionMismatch(
        "filter_apply_batched! got out $(size(out)) and field $(size(field))",
    ))
    (size(out, 1), size(out, 2)) == gp.dims || throw(DimensionMismatch(
        "field's leading axes $((size(out, 1), size(out, 2))) do not match the plan's $(gp.dims)",
    ))
    b = gp.batched
    (b === nothing || sc.batched === nothing) && throw(ArgumentError(
        "this spectral plan was not built for a batch; pass `batch = Nb` to `plan_filter`",
    ))
    b.nb == size(out, 3) || throw(DimensionMismatch(
        "plan was built for a batch of $(b.nb), got $(size(out, 3))",
    ))
    p = sc.batched
    if gp.mask === nothing
        LA.mul!(p.cbuf, b.fwd, field)
    else
        @. p.masked_input = gp.mask * field
        LA.mul!(p.cbuf, b.fwd, p.masked_input)
    end
    p.cbuf .*= plan.transfer
    LA.mul!(out, b.inv, p.cbuf)
    plan.invrenorm === nothing || (out .*= plan.invrenorm)
    return out
end

# Analysis is scale-independent, so a sweep transforms each field once and then only multiplies by each
# scale's transfer function and inverts. `masked_input` is the plan's own scratch and is not read after
# analysis, so the mask is applied here, once for the whole sweep.
CGEF.Filtering.analyze_buffer(plan::FFTWFilterPlan, field::AbstractMatrix) = similar(plan.scratch.cbuf)

# A trailing batch axis needs the transforms bound to that shape; without them there is no shareable
# analysis and the caller takes the per-scale apply.
CGEF.Filtering.analyze_buffer(plan::FFTWFilterPlan, field::AbstractArray{<:Any,3}) =
    (plan.grid_plan.batched === nothing || plan.grid_plan.batched.nb != size(field, 3)) ? nothing :
        similar(plan.scratch.cbuf, size(plan.scratch.cbuf)..., size(field, 3))

function CGEF.Filtering.filter_analyze!(
    F̂::AbstractMatrix{Complex{T}}, field::AbstractMatrix{T}, plan::FFTWFilterPlan{T},
) where {T<:AbstractFloat}
    gp = plan.grid_plan
    if gp.mask === nothing
        LA.mul!(F̂, gp.fwd, field)
    else
        @. plan.scratch.masked_input = gp.mask * field
        LA.mul!(F̂, gp.fwd, plan.scratch.masked_input)
    end
    return F̂
end

# Rank-3 analyze/synthesize over the batched transforms, so a batch driver that analyzes once per sweep
# has the same two halves available as the single-field path.
function CGEF.Filtering.filter_analyze!(
    F̂::AbstractArray{Complex{T},3}, field::AbstractArray{T,3}, plan::FFTWFilterPlan{T},
) where {T<:AbstractFloat}
    gp, sc = plan.grid_plan, plan.scratch
    b = gp.batched
    (b === nothing || sc.batched === nothing) && throw(ArgumentError(
        "this spectral plan was not built for a batch; pass `batch = Nb` to `plan_filter`",
    ))
    b.nb == size(field, 3) || throw(DimensionMismatch(
        "plan was built for a batch of $(b.nb), got $(size(field, 3))",
    ))
    if gp.mask === nothing
        LA.mul!(F̂, b.fwd, field)
    else
        @. sc.batched.masked_input = gp.mask * field
        LA.mul!(F̂, b.fwd, sc.batched.masked_input)
    end
    return F̂
end

function CGEF.Filtering.filter_synthesize!(
    out::AbstractMatrix{T}, F̂::AbstractMatrix{Complex{T}}, plan::FFTWFilterPlan{T},
) where {T<:AbstractFloat}
    # `inv` consumes its input, so synthesize from a copy — `F̂` is reused by every later scale.
    plan.scratch.cbuf .= F̂ .* plan.transfer
    LA.mul!(out, plan.grid_plan.inv, plan.scratch.cbuf)
    plan.invrenorm === nothing || (out .*= plan.invrenorm)
    return out
end

function CGEF.Filtering.filter_synthesize!(
    out::AbstractArray{T,3}, F̂::AbstractArray{Complex{T},3}, plan::FFTWFilterPlan{T},
) where {T<:AbstractFloat}
    gp, sc = plan.grid_plan, plan.scratch
    b = gp.batched
    (b === nothing || sc.batched === nothing) && throw(ArgumentError(
        "this spectral plan was not built for a batch; pass `batch = Nb` to `plan_filter`",
    ))
    sc.batched.cbuf .= F̂ .* plan.transfer
    LA.mul!(out, b.inv, sc.batched.cbuf)
    plan.invrenorm === nothing || (out .*= plan.invrenorm)
    return out
end

function _fftw_grid_plan(
    grid::FlowGeometries.Grids.StructuredGrid{T,G}; batch::Union{Nothing,Integer} = nothing,
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

    sample = zeros(T, Nx, Ny)
    fwd = FFTW.plan_rfft(sample)
    cbuf = fwd * sample                 # complex spectrum (Nx÷2+1, Ny)
    inv = FFTW.plan_irfft(cbuf, Nx)
    mask = all(FlowGeometries.Grids.mask(grid)) ? nothing : FlowGeometries.Grids.mask(grid)
    batched = if batch === nothing
        nothing
    else
        nb = Int(batch)
        bbuf = zeros(T, Nx, Ny, nb)
        bfwd = FFTW.plan_rfft(bbuf, (1, 2))
        (fwd = bfwd, inv = FFTW.plan_irfft(bfwd * bbuf, Nx, (1, 2)), nb = nb)
    end
    return FFTWGridPlan(fwd, inv, mask, kx, ky, (Nx, Ny), batched)
end

# The buffers an apply writes through, sized from the grid plan's own shape.
function _fftw_scratch(gp::FFTWGridPlan{T}) where {T<:AbstractFloat}
    Nx, Ny = gp.dims
    b = gp.batched
    return FFTWScratch(
        zeros(Complex{T}, Nx ÷ 2 + 1, Ny),
        zeros(T, Nx, Ny),   # allocated regardless; only touched when mask !== nothing
        b === nothing ? nothing :
            (cbuf = zeros(Complex{T}, Nx ÷ 2 + 1, Ny, b.nb), masked_input = zeros(T, Nx, Ny, b.nb)),
    )
end

CGEF.Filtering.spectral_grid_plan(
    ::Union{CGEF.SpectralBackends.AbstractAutoSpectralBackend, CGEF.SpectralBackends.AbstractFFTSpectralBackend},
    grid::FlowGeometries.Grids.StructuredGrid{T,G},
    kernel::CGEF.Kernels.AbstractFilterKernel;
    batch::Union{Nothing,Integer} = nothing,
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}} =
    _fftw_grid_plan(grid; batch = batch)

CGEF.Filtering.spectral_scratch(gp::FFTWGridPlan) = _fftw_scratch(gp)

function CGEF.Filtering.spectral_filter_plan(
    ::Union{CGEF.SpectralBackends.AbstractAutoSpectralBackend, CGEF.SpectralBackends.AbstractFFTSpectralBackend},
    grid::FlowGeometries.Grids.StructuredGrid{T,G},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy = CGEF.Filtering.ZeroFill(),
    backend = CGEF.ComputationalBackends.AutoBackend(),
    # Extent of the trailing batch axis this plan will be applied over, or `nothing` for single fields.
    # A transform is bound to one field shape, so it is fixed here rather than discovered at apply time.
    batch::Union{Nothing,Integer} = nothing,
    grid_plan::Union{Nothing,FFTWGridPlan} = nothing,
    scratch::Union{Nothing,FFTWScratch} = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    gp = grid_plan === nothing ? _fftw_grid_plan(grid; batch = batch) : grid_plan
    sc = scratch === nothing ? _fftw_scratch(gp) : scratch
    kx, ky = gp.kx, gp.ky

    transfer = T[CGEF.Kernels.spectral_transfer(kernel, sqrt(kx[i]^2 + ky[j]^2), scale) for i in eachindex(kx), j in eachindex(ky)]

    mask = gp.mask
    invrenorm = if mask !== nothing && mask_strategy isa CGEF.Filtering.Deformable
        # Local kernel mass over active cells, `filter(mask)`, through the plan's own transforms. The
        # mask is fixed for the plan, so this is built once and stored inverted.
        sc.masked_input .= mask
        LA.mul!(sc.cbuf, gp.fwd, sc.masked_input)
        sc.cbuf .*= transfer
        renorm = similar(sc.masked_input)
        LA.mul!(renorm, gp.inv, sc.cbuf)
        threshold = T(0.01)
        ir = similar(renorm)
        @. ir = ifelse(abs(renorm) >= threshold, one(T) / renorm, zero(T))
        ir
    else
        nothing   # ZeroFill: already exactly `filter(mask .* field)`, no renormalization
    end
    return FFTWFilterPlan(gp, transfer, invrenorm, sc)
end

function CGEF.Filtering.filter_apply!(
    out::AbstractMatrix{T},
    field::AbstractMatrix{T},
    plan::FFTWFilterPlan{T},
) where {T<:AbstractFloat}
    gp, sc = plan.grid_plan, plan.scratch
    if gp.mask === nothing
        LA.mul!(sc.cbuf, gp.fwd, field)       # f̂ = rfft(field)
    else
        @. sc.masked_input = gp.mask * field
        LA.mul!(sc.cbuf, gp.fwd, sc.masked_input)
    end
    sc.cbuf .*= plan.transfer             # ĝ · f̂
    LA.mul!(out, gp.inv, sc.cbuf)         # irfft  (consumes cbuf, rebuilt next call)
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
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
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

# ---------------------------------------------------------------------------
# Zonal-FFT engine: the same compact great-circle kernel, evaluated by transform along longitude.
#
# For a target at latitude row `j` and a source at row `jj`, the great-circle distance
#
#     cos d = sin φ_j sin φ_jj + cos φ_j cos φ_jj cos(Δλ)
#
# depends on the longitude DIFFERENCE alone, and a rectilinear spherical cell's area `R² cos φ_jj Δλ Δφ`
# does not depend on longitude at all. So the weight is a function `w_{j,jj}(di)` of the index offset,
# and each band contributes a circular cross-correlation along the longitude ring:
#
#     out[i, j] = Σ_jj Σ_di w_{j,jj}[di] · src[i+di, jj]
#
# That turns the O(di_lim) tap loop per band into one pointwise multiply in the transform domain, which
# matters most exactly where the direct engine is worst: `di_lim` grows as 1/cos φ, so the polar rows
# dominate the direct cost while costing a transform no more than the equatorial ones.
#
# `w_{j,jj}` is EVEN in `di` (the distance depends on cos Δλ), so its spectrum is real — the accumulate
# is a real-times-complex multiply, and the spectra are stored as reals.
#
# The window mass is accumulated through the SAME transform, not by a direct sum. That is deliberate:
# numerator and denominator are then the same linear operator applied to `mask·field` and to the mass
# field, so a constant field filters back to exactly that constant rather than to within round-off.
# ---------------------------------------------------------------------------

struct ZonalFFTFootprint{
    T<:AbstractFloat, A<:AbstractMatrix{T}, R3<:AbstractArray{T,3},
    C<:AbstractMatrix{Complex{T}}, FP, IP, MS<:CGEF.Filtering.AbstractMaskStrategy,
}
    Ŵ::R3           # (Nx÷2+1, 2·dj_lim+1, Ny) real kernel spectra, per (band offset, target row)
    invden::A       # (Nx, Ny) reciprocal window mass, built through this same transform
    dj_lim::Int
    masked::Bool
    strategy::MS    # the strategy `invden` was built for
    src::A          # (Nx, Ny) scratch for `mask · field`
    F::C            # (Nx÷2+1, Ny) scratch: every source row's spectrum
    acc::C          # (Nx÷2+1, Ny) scratch: the accumulated spectrum per target row
    fwd::FP         # batched rfft along longitude
    inv::IP
end

# The band sum, in the transform domain. Shared by the apply and by the plan-time mass accumulation so
# the two cannot disagree about which bands contribute.
function _zonal_accumulate!(
    out::AbstractMatrix{T}, src::AbstractMatrix{T}, fp::ZonalFFTFootprint{T}, Ny::Int,
) where {T<:AbstractFloat}
    LA.mul!(fp.F, fp.fwd, src)
    fill!(fp.acc, zero(Complex{T}))
    nk = size(fp.F, 1)
    Ŵ = fp.Ŵ
    @inbounds for j in 1:Ny
        for dj in (-fp.dj_lim):(fp.dj_lim)
            jj = j + dj
            (1 <= jj <= Ny) || continue
            b = dj + fp.dj_lim + 1
            @simd for n in 1:nk
                fp.acc[n, j] += Ŵ[n, b, j] * fp.F[n, jj]
            end
        end
    end
    LA.mul!(out, fp.inv, fp.acc)
    return out
end

function CGEF.Filtering.zonal_fft_footprint(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy = CGEF.Filtering.ZeroFill(),
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.SphericalGeometry{T}}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    geo = FlowGeometries.Grids.grid_geometry(grid)
    R = FlowGeometries.Geometry.radius(geo)
    lat = FlowGeometries.Grids.coordinates(grid, 2)
    dλ = step(FlowGeometries.Grids.coordinates(grid, 1))
    rad = CGEF.Kernels.kernel_radius(kernel, scale)

    # Latitude band bound, from the axis's own minimum gap converted to a physical distance — the same
    # rule the banded builder uses, so the two engines cover the same set of source rows.
    min_dφ = FlowGeometries.Grids.minimum_spacing(grid, 2)
    dφ_phys = R * min_dφ
    dj_lim = (isfinite(dφ_phys) && dφ_phys > 0) ? min(Ny - 1, ceil(Int, rad / dφ_phys)) : 0
    nb = 2 * dj_lim + 1

    # `cos d ≥ cos rad` is the support test, and monotone, so the gate costs no inverse trig; only the
    # in-support offsets pay for `acos`. `cosΔλ` is shared by every (j, jj) pair.
    cosΔλ = T[cos(T(di) * dλ) for di in 0:(Nx - 1)]
    cos_rad = cos(min(rad / R, T(π)))

    sample = zeros(T, Nx, Ny)
    fwd = FFTW.plan_rfft(sample, 1)
    Fbuf = fwd * sample
    iplan = FFTW.plan_irfft(Fbuf, Nx, 1)
    nk = size(Fbuf, 1)

    Ŵ = zeros(T, nk, nb, Ny)
    gcol = zeros(T, Nx, nb)
    gplan = FFTW.plan_rfft(gcol, 1)
    for j in 1:Ny
        φj = lat[j]
        sj, cj = sin(φj), cos(φj)
        fill!(gcol, zero(T))
        for dj in (-dj_lim):dj_lim
            jj = j + dj
            (1 <= jj <= Ny) || continue
            b = dj + dj_lim + 1
            φk = lat[jj]
            A0 = sj * sin(φk)
            B0 = cj * cos(φk)
            area = FlowGeometries.Grids.area(grid, 1, jj)   # longitude-independent on this grid
            for di in 0:(Nx - 1)
                cd = A0 + B0 * cosΔλ[di + 1]
                cd >= cos_rad || continue
                d = R * acos(clamp(cd, -one(T), one(T)))
                gcol[di + 1, b] = CGEF.Kernels.kernel_weight(kernel, d, scale) * area
            end
        end
        Ĝ = gplan * gcol
        @inbounds for b in 1:nb, n in 1:nk
            Ŵ[n, b, j] = real(Ĝ[n, b])
        end
    end

    masked = !all(FlowGeometries.Grids.mask(grid))
    fp = ZonalFFTFootprint(
        Ŵ, zeros(T, Nx, Ny), dj_lim, masked, mask_strategy,
        zeros(T, Nx, Ny), Fbuf, similar(Fbuf), fwd, iplan,
    )

    # Window mass through the same operator: `ZeroFill` counts every in-support cell, `Deformable` only
    # the active ones.
    maskv = FlowGeometries.Grids.mask(grid)
    @inbounds for j in 1:Ny, i in 1:Nx
        fp.src[i, j] = (mask_strategy isa CGEF.Filtering.ZeroFill || maskv[i, j]) ? one(T) : zero(T)
    end
    den = zeros(T, Nx, Ny)
    _zonal_accumulate!(den, fp.src, fp, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        fp.invden[i, j] = (maskv[i, j] && den[i, j] > T(1e-15)) ? one(T) / den[i, j] : zero(T)
    end
    return fp
end

function CGEF.Filtering.apply_footprint!(
    out::AbstractMatrix{T}, field::AbstractMatrix,
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    fp::ZonalFFTFootprint{T}, strategy::CGEF.Filtering.AbstractMaskStrategy,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.SphericalGeometry{T}}
    (!fp.masked || typeof(strategy) === typeof(fp.strategy)) || throw(ArgumentError(
        "this zonal-FFT footprint was built for $(nameof(typeof(fp.strategy))) on a masked grid; its " *
        "normalization is not the one $(nameof(typeof(strategy))) needs. Rebuild the plan with the " *
        "strategy you intend to apply with.",
    ))
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    maskv = FlowGeometries.Grids.mask(grid)
    @inbounds for j in 1:Ny, i in 1:Nx
        fp.src[i, j] = maskv[i, j] ? T(field[i, j]) : zero(T)
    end
    _zonal_accumulate!(out, fp.src, fp, Ny)
    invden = fp.invden
    @inbounds for j in 1:Ny
        @simd for i in 1:Nx
            out[i, j] *= invden[i, j]
        end
    end
    return out
end

CGEF.Filtering._apply_serial!(
    out, field, grid::FlowGeometries.Grids.StructuredGrid{T,G,2}, fp::ZonalFFTFootprint{T}, strategy,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.SphericalGeometry{T}} =
    CGEF.Filtering.apply_footprint!(out, field, grid, fp, strategy)

function CGEF.Filtering.apply_footprint!(
    out::AbstractMatrix{T}, field::AbstractMatrix,
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
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
