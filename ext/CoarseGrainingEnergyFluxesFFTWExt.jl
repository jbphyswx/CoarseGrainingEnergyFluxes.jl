module CoarseGrainingEnergyFluxesFFTWExt

using FFTW: FFTW
using LinearAlgebra: LinearAlgebra as LA
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF
using FlowGeometries: FlowGeometries

# Spectral filtering for uniform Cartesian grids: a pointwise multiply by the kernel's transfer function
# `Ĝ(|k|, ℓ)` (`CGEF.Kernels.spectral_transfer`, shared with the other spectral backends), O(N log N)
# whatever the filter scale. `Ĝ(0) = 1`.
#
# A periodic axis is transformed at its own length. A bounded one is zero-padded to
# `2·nextprod((2,3,5), N)`, the least even 2·3·5-smooth length of at least `2N`, and the result cropped
# back to the grid: the field is extended by zero beyond the domain, the kernel's full mass is in the
# normalization, and the wrap-around path between two cells of the grid is at least the domain length.
#
# Masking uses the normalized-convolution identity (Knutsson & Westin 1993) that `RealSpace`'s
# strategies implement pointwise: `ZeroFill` filters `mask·field` directly, and `Deformable`
# additionally divides by the local kernel mass over active cells, `filter(mask)`. The mask is fixed
# for a plan, so that denominator is computed at build time and stored inverted.

"""
    FFTWGridPlan

The half of an FFTW spectral plan the filter scale does not reach: the forward and inverse transforms
over the transform grid, the mask, the angular wavenumber grids, and — when the plan was built for a
trailing batch axis — the transforms bound to that shape. One instance serves every scale of a sweep,
and it is immutable during an apply, so concurrent workers may share it.

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
    dims::NTuple{2,Int}   # the grid
    P::NTuple{2,Int}      # the transform grid: `dims` along periodic axes, padded along bounded ones
    padded::Bool
    batched::BT    # (; fwd, inv, nb) for a trailing batch axis, or nothing
end

"""
    FFTWScratch

The transient half of an FFTW spectral plan: the complex spectrum buffer, the transform-grid input
(`mask · field`, zero beyond the grid), the transform-grid output a padded plan crops from, and the same
set sized for a trailing batch axis. Every scale of a sweep shares one, since scales run sequentially; a
driver that applies concurrently gives each worker its own.
"""
struct FFTWScratch{T<:AbstractFloat, CA<:AbstractMatrix{Complex{T}}, A<:AbstractMatrix{T}, PO, BT} <:
       CGEF.Filtering.AbstractFilterScratch
    cbuf::CA
    masked_input::A
    pad_out::PO    # transform-grid output, or nothing when unpadded
    batched::BT    # (; cbuf, masked_input, pad_out) sized for the batch axis, or nothing
end

# The grid's cells within a transform-grid array.
@inline _fftw_region(A::AbstractMatrix, (Nx, Ny)) = view(A, 1:Nx, 1:Ny)
@inline _fftw_region(A::AbstractArray{<:Any,3}, (Nx, Ny)) = view(A, 1:Nx, 1:Ny, :)

# The forward transform of `field` into `dst`: directly where nothing is masked or padded, otherwise
# of `mask · field` written into the grid's region of `buf`, whose padding stays zero.
function _fftw_forward!(dst, fwd, field, buf, gp::FFTWGridPlan)
    if gp.padded
        _fftw_stage!(_fftw_region(buf, gp.dims), field, gp.mask)
        LA.mul!(dst, fwd, buf)
    elseif gp.mask === nothing
        LA.mul!(dst, fwd, field)
    else
        _fftw_stage!(buf, field, gp.mask)
        LA.mul!(dst, fwd, buf)
    end
    return dst
end

@inline _fftw_stage!(data, field, ::Nothing) = (data .= field)
@inline _fftw_stage!(data, field, mask) = (@. data = mask * field)

# The inverse transform into `out`; a padded plan holds a transform-grid `pout` and crops it.
@inline _fftw_inverse!(out, inv, spec, ::Nothing, ::FFTWGridPlan) = LA.mul!(out, inv, spec)
function _fftw_inverse!(out, inv, spec, pout::AbstractArray, gp::FFTWGridPlan)
    LA.mul!(pout, inv, spec)
    out .= _fftw_region(pout, gp.dims)
    return out
end

"""
    FFTWFilterPlan

Cached FFT filter plan: the shared [`FFTWGridPlan`](@ref), the precomputed transfer-function array, the
`Deformable` local-mass renormalization (or `nothing`), and the [`FFTWScratch`](@ref) the apply writes
through. Built by `plan_filter(...; method = Spectral())`.
"""
struct FFTWFilterPlan{
    T<:AbstractFloat, GP<:FFTWGridPlan{T}, A<:AbstractMatrix{T}, R,
    MS<:CGEF.Filtering.AbstractMaskStrategy, SC<:FFTWScratch{T},
} <: CGEF.Filtering.AbstractFilterPlan
    grid_plan::GP
    transfer::A    # Ĝ(|k|, ℓ) on the rfft transform grid  (Px÷2+1, Py)
    invrenorm::R   # 1/filter(mask) for Deformable, zero at a masked cell; or nothing
    strategy::MS
    scratch::SC
end

CGEF.Filtering.plan_strategy(plan::FFTWFilterPlan) = plan.strategy

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
    _fftw_forward!(p.cbuf, b.fwd, field, p.masked_input, gp)
    p.cbuf .*= plan.transfer
    _fftw_inverse!(out, b.inv, p.cbuf, p.pad_out, gp)
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
    return _fftw_forward!(F̂, gp.fwd, field, plan.scratch.masked_input, gp)
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
    return _fftw_forward!(F̂, b.fwd, field, sc.batched.masked_input, gp)
end

function CGEF.Filtering.filter_synthesize!(
    out::AbstractMatrix{T}, F̂::AbstractMatrix{Complex{T}}, plan::FFTWFilterPlan{T},
) where {T<:AbstractFloat}
    # `inv` consumes its input, so synthesize from a copy — `F̂` is reused by every later scale.
    gp, sc = plan.grid_plan, plan.scratch
    sc.cbuf .= F̂ .* plan.transfer
    _fftw_inverse!(out, gp.inv, sc.cbuf, sc.pad_out, gp)
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
    _fftw_inverse!(out, b.inv, sc.batched.cbuf, sc.batched.pad_out, gp)
    plan.invrenorm === nothing || (out .*= plan.invrenorm)
    return out
end

function _fftw_grid_plan(
    grid::FlowGeometries.Grids.StructuredGrid{T,G}; batch::Union{Nothing,Integer} = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    (FlowGeometries.Grids.isuniform(grid, 1) && FlowGeometries.Grids.isuniform(grid, 2)) || throw(ArgumentError(
        "Spectral FFT filtering needs uniformly spaced axes (`AbstractRange` coordinates); this grid has " *
        "a stretched one. Use `RealSpace()`, or a node grid with `FINUFFT`.",
    ))
    Nx, Ny = size(FlowGeometries.Grids.mask(grid))
    px = FlowGeometries.Grids.isperiodic(grid, 1)
    py = FlowGeometries.Grids.isperiodic(grid, 2)
    # A periodic axis's spacing is its period over its length; a bounded one's is its step.
    dx = px ? T(FlowGeometries.Grids.period(grid, 1)) / Nx : abs(T(FlowGeometries.Grids.spacing(grid, 1)))
    dy = py ? T(FlowGeometries.Grids.period(grid, 2)) / Ny : abs(T(FlowGeometries.Grids.spacing(grid, 2)))
    Px = px ? Nx : 2 * nextprod((2, 3, 5), Nx)
    Py = py ? Ny : 2 * nextprod((2, 3, 5), Ny)
    padded = (Px, Py) != (Nx, Ny)
    # Angular wavenumbers on the transform grid (rfft halves the first axis).
    kx = T(2π) .* FFTW.rfftfreq(Px, one(T) / dx)
    ky = T(2π) .* FFTW.fftfreq(Py, one(T) / dy)

    sample = zeros(T, Px, Py)
    fwd = FFTW.plan_rfft(sample)
    cbuf = fwd * sample                 # complex spectrum (Px÷2+1, Py)
    inv = FFTW.plan_irfft(cbuf, Px)
    mask = all(FlowGeometries.Grids.mask(grid)) ? nothing : FlowGeometries.Grids.mask(grid)
    batched = if batch === nothing
        nothing
    else
        nb = Int(batch)
        bbuf = zeros(T, Px, Py, nb)
        bfwd = FFTW.plan_rfft(bbuf, (1, 2))
        (fwd = bfwd, inv = FFTW.plan_irfft(bfwd * bbuf, Px, (1, 2)), nb = nb)
    end
    return FFTWGridPlan(fwd, inv, mask, kx, ky, (Nx, Ny), (Px, Py), padded, batched)
end

# The buffers an apply writes through, sized from the grid plan's own shape.
function _fftw_scratch(gp::FFTWGridPlan{T}) where {T<:AbstractFloat}
    Px, Py = gp.P
    b = gp.batched
    return FFTWScratch(
        zeros(Complex{T}, Px ÷ 2 + 1, Py),
        zeros(T, Px, Py),   # touched when masked or padded
        gp.padded ? zeros(T, Px, Py) : nothing,
        b === nothing ? nothing :
            (cbuf = zeros(Complex{T}, Px ÷ 2 + 1, Py, b.nb), masked_input = zeros(T, Px, Py, b.nb),
             pad_out = gp.padded ? zeros(T, Px, Py, b.nb) : nothing),
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
    invrenorm = if mask_strategy isa CGEF.Filtering.Deformable && (mask !== nothing || gp.padded)
        # Local kernel mass over active cells, `filter(mask)`, through the plan's own transforms; the
        # padding beyond a bounded axis is inactive. The mask is fixed for the plan, so this is built
        # once and stored inverted.
        region = _fftw_region(sc.masked_input, gp.dims)
        mask === nothing ? fill!(region, one(T)) : (region .= mask)
        LA.mul!(sc.cbuf, gp.fwd, sc.masked_input)
        sc.cbuf .*= transfer
        renorm = zeros(T, gp.dims)
        _fftw_inverse!(renorm, gp.inv, sc.cbuf, sc.pad_out, gp)
        threshold = T(0.01)
        ir = similar(renorm)
        # A masked cell is zero under `Deformable`, as in the real-space engines.
        active = mask === nothing ? trues(gp.dims) : mask
        @. ir = ifelse(active & (abs(renorm) >= threshold), one(T) / renorm, zero(T))
        ir
    else
        nothing   # ZeroFill: already exactly `filter(mask .* field)`, no renormalization
    end
    return FFTWFilterPlan(gp, transfer, invrenorm, mask_strategy, sc)
end

function CGEF.Filtering.filter_apply!(
    out::AbstractMatrix{T},
    field::AbstractMatrix{T},
    plan::FFTWFilterPlan{T},
) where {T<:AbstractFloat}
    gp, sc = plan.grid_plan, plan.scratch
    _fftw_forward!(sc.cbuf, gp.fwd, field, sc.masked_input, gp)   # f̂ = rfft(mask · field)
    sc.cbuf .*= plan.transfer                                     # ĝ · f̂
    _fftw_inverse!(out, gp.inv, sc.cbuf, sc.pad_out, gp)          # irfft (consumes cbuf)
    plan.invrenorm === nothing || (out .*= plan.invrenorm)
    return out
end

# ---------------------------------------------------------------------------
# Padded-FFT real-space engine
# ---------------------------------------------------------------------------
#
# A non-separable kernel has no factored form, so the direct engine enumerates a whole disk per point:
# O(N·w²), and `SharpSpectralKernel`'s radius is 10ℓ. This evaluates the same sum by transform. A
# periodic axis keeps its length, and the sampled kernel is accumulated modulo it, which is the sum over
# every image. A bounded axis is zero-padded to `N + m`, `m = min(w, N - 1)` the widest offset between
# two cells, the least length at which the circular convolution equals the linear one.
#
# `Deformable`/`ZeroFill` are normalized convolution: num = conv(mask·f, g), den = conv(mask, g). `den`
# depends only on grid/kernel/scale/mask, so it is built once here rather than per apply.
struct PaddedFFTFootprint{T<:AbstractFloat, A<:AbstractMatrix{T}, C<:AbstractMatrix{Complex{T}}, FP, IP, S}
    Ĝ::C            # kernel spectrum on the transform grid
    den::A          # the normalization, cropped and precomputed — it depends on the mask STRATEGY
    pad::A          # transform-grid scratch for mask·field
    num::A          # transform-grid scratch for the inverse transform
    spec::C         # spectrum scratch
    fwd::FP
    inv::IP
    strategy::S     # the strategy `den` was built for; applying another one would be wrong
    N::NTuple{2,Int}
    P::NTuple{2,Int}   # transform length per axis
end

# Transform length along one axis: the axis itself where it wraps, else the least 2·3·5-smooth length
# holding the data and the widest offset between two of its cells.
_padded_length(N::Int, w::Int, periodic::Bool) = periodic ? N : nextprod((2, 3, 5), N + min(w, N - 1))

function CGEF.Filtering.padded_fft_footprint(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy = CGEF.Filtering.ZeroFill(),
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    dx = abs(step(FlowGeometries.Grids.coordinates(grid, 1)))
    dy = abs(step(FlowGeometries.Grids.coordinates(grid, 2)))
    px = FlowGeometries.Grids.isperiodic(grid, 1)
    py = FlowGeometries.Grids.isperiodic(grid, 2)
    rad = CGEF.Kernels.kernel_radius(kernel, scale, Val(2))
    # A periodic axis reaches every image inside the support; a bounded one no farther than its extent.
    wx = px ? ceil(Int, rad / dx) : min(ceil(Int, rad / dx), Nx - 1)
    wy = py ? ceil(Int, rad / dy) : min(ceil(Int, rad / dy), Ny - 1)
    Px = _padded_length(Nx, wx, px)
    Py = _padded_length(Ny, wy, py)
    A = FlowGeometries.Grids.area(grid, 1, 1)

    # Kernel field, wrapped so index (1,1) is the zero offset — otherwise the result is shifted. Gated
    # on `d <= rad`: the footprint is a DISK, and including the enclosing box's corners changes a
    # slowly-decaying kernel's answer outright.
    gpad = zeros(T, Px, Py)
    for dj in (-wy):wy, di in (-wx):wx
        d = hypot(di * dx, dj * dy)
        d <= rad || continue
        wt = CGEF.Kernels.kernel_weight(kernel, d, scale, Val(2)) * A
        iszero(wt) && continue
        gpad[mod1(1 + di, Px), mod1(1 + dj, Py)] += wt
    end

    fwd = FFTW.plan_rfft(gpad)
    Ĝ = fwd * gpad
    spec = similar(Ĝ)
    inv = FFTW.plan_irfft(spec, Px)

    # The denominator differs by strategy, as it does in the direct engine: `Deformable` renormalizes
    # over the active cells, and `ZeroFill` divides by the kernel's full mass, its in-domain part through
    # the transform and the part past a bounded edge from `_exterior_mass`.
    zerofill = mask_strategy isa CGEF.Filtering.ZeroFill
    maskv = FlowGeometries.Grids.mask(grid)
    dpad = zeros(T, Px, Py)
    @inbounds for j in 1:Ny, i in 1:Nx
        dpad[i, j] = (zerofill || maskv[i, j]) ? one(T) : zero(T)
    end
    spec = fwd * dpad
    spec .*= Ĝ
    denfull = inv * spec
    den = Array{T}(undef, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        den[i, j] = denfull[i, j]
    end
    exterior = zerofill ? CGEF.Filtering._exterior_mass(grid, kernel, scale) : nothing
    exterior === nothing || (den .+= exterior)

    return PaddedFFTFootprint(
        Ĝ, den, zeros(T, Px, Py), zeros(T, Px, Py), similar(Ĝ), fwd, inv, mask_strategy, (Nx, Ny),
        (Px, Py),
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
    bound::Bool     # the strategies' denominators differ: a cell is inactive or a window leaves the grid
    strategy::MS    # the strategy `invden` was built for
    src::A          # (Nx, Ny) scratch for `mask · field`
    F::C            # (Nx÷2+1, Ny) scratch: every source row's spectrum
    acc::C          # (Nx÷2+1, Ny) scratch: the accumulated spectrum per target row
    fwd::FP         # batched rfft along longitude
    inv::IP
end

# The band sum, in the transform domain. Shared by the apply and by the plan-time mass accumulation so
# the two cannot disagree about which bands contribute. Target row `j` writes only column `j` of
# `acc`, so `rows` may visit the rows in any order or in parallel.
function _zonal_accumulate!(
    out::AbstractMatrix{T}, src::AbstractMatrix{T}, fp::ZonalFFTFootprint{T}, Ny::Int,
    rows::D = CGEF.Filtering._sep_serial,
) where {T<:AbstractFloat, D}
    LA.mul!(fp.F, fp.fwd, src)
    nk = size(fp.F, 1)
    Ŵ, F, acc, dj_lim = fp.Ŵ, fp.F, fp.acc, fp.dj_lim
    rows(1:Ny) do j
        @inbounds begin
            @simd for n in 1:nk
                acc[n, j] = zero(Complex{T})
            end
            for dj in (-dj_lim):dj_lim
                jj = j + dj
                (1 <= jj <= Ny) || continue
                b = dj + dj_lim + 1
                @simd for n in 1:nk
                    acc[n, j] += Ŵ[n, b, j] * F[n, jj]
                end
            end
        end
        return nothing
    end
    LA.mul!(out, fp.inv, acc)
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
    rad = CGEF.Kernels.kernel_radius(kernel, scale, Val(2))

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
                gcol[di + 1, b] = CGEF.Kernels.kernel_weight(kernel, d, scale, Val(2)) * area
            end
        end
        Ĝ = gplan * gcol
        @inbounds for b in 1:nb, n in 1:nk
            Ŵ[n, b, j] = real(Ĝ[n, b])
        end
    end

    bound = !all(FlowGeometries.Grids.mask(grid)) || CGEF.Filtering._reaches_exterior(grid, kernel, scale)
    fp = ZonalFFTFootprint(
        Ŵ, zeros(T, Nx, Ny), dj_lim, bound, mask_strategy,
        zeros(T, Nx, Ny), Fbuf, similar(Fbuf), fwd, iplan,
    )

    # Window mass through the same operator: `ZeroFill` counts every in-support cell and the rows past a
    # bounded latitude, as far as the poles; `Deformable` only the active in-domain cells, and gives an
    # inactive target zero.
    zerofill = mask_strategy isa CGEF.Filtering.ZeroFill
    maskv = FlowGeometries.Grids.mask(grid)
    @inbounds for j in 1:Ny, i in 1:Nx
        fp.src[i, j] = (zerofill || maskv[i, j]) ? one(T) : zero(T)
    end
    den = zeros(T, Nx, Ny)
    _zonal_accumulate!(den, fp.src, fp, Ny)
    exterior = zerofill ? CGEF.Filtering._exterior_mass(grid, kernel, scale) : nothing
    exterior === nothing || (den .+= exterior)
    @inbounds for j in 1:Ny, i in 1:Nx
        fp.invden[i, j] = (zerofill || maskv[i, j]) ? CGEF.Filtering._inv_mass(den[i, j]) : zero(T)
    end
    return fp
end

CGEF.Filtering._transform_footprint(::Union{ZonalFFTFootprint, PaddedFFTFootprint}) = true

# `rows(f, 1:Ny)` visits every latitude row; each row writes only its own column of every buffer.
function CGEF.Filtering.apply_footprint!(
    out::AbstractMatrix{T}, field::AbstractMatrix,
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    fp::ZonalFFTFootprint{T}, strategy::CGEF.Filtering.AbstractMaskStrategy,
    rows::D = CGEF.Filtering._sep_serial,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.SphericalGeometry{T}, D}
    (!fp.bound || typeof(strategy) === typeof(fp.strategy)) || throw(ArgumentError(
        "this zonal-FFT footprint was built for $(nameof(typeof(fp.strategy))) on a grid where the " *
        "strategies normalize differently (a masked cell, or a window past a bounded latitude); its " *
        "normalization is not the one $(nameof(typeof(strategy))) needs. Rebuild the plan with the " *
        "strategy you intend to apply with.",
    ))
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    maskv = FlowGeometries.Grids.mask(grid)
    src, invden = fp.src, fp.invden
    rows(1:Ny) do j
        @inbounds @simd for i in 1:Nx
            src[i, j] = maskv[i, j] ? T(field[i, j]) : zero(T)
        end
        return nothing
    end
    _zonal_accumulate!(out, src, fp, Ny, rows)
    rows(1:Ny) do j
        @inbounds @simd for i in 1:Nx
            out[i, j] *= invden[i, j]
        end
        return nothing
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
    rows::D = CGEF.Filtering._sep_serial,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}, D}
    strategy === fp.strategy || throw(ArgumentError(
        "this padded-FFT footprint was built for $(typeof(fp.strategy)); its denominator is not the " *
        "one $(typeof(strategy)) needs. Rebuild the plan with the strategy you intend to apply.",
    ))
    Nx, Ny = fp.N
    maskv = FlowGeometries.Grids.mask(grid)
    pad, num, den = fp.pad, fp.num, fp.den
    zerofill = strategy isa CGEF.Filtering.ZeroFill
    fill!(pad, zero(T))
    rows(1:Ny) do j
        @inbounds @simd for i in 1:Nx
            pad[i, j] = maskv[i, j] ? T(field[i, j]) : zero(T)
        end
        return nothing
    end
    # In place through the held buffers: `plan * array` allocates a fresh result on every apply.
    LA.mul!(fp.spec, fp.fwd, pad)
    fp.spec .*= fp.Ĝ
    LA.mul!(num, fp.inv, fp.spec)
    rows(1:Ny) do j
        @inbounds @simd for i in 1:Nx
            out[i, j] = (zerofill || maskv[i, j]) ? CGEF.Filtering._normalized(num[i, j], den[i, j]) : zero(T)
        end
        return nothing
    end
    return out
end

CGEF.Filtering._apply_serial!(out, field, grid, fp::PaddedFFTFootprint, strategy) =
    CGEF.Filtering.apply_footprint!(out, field, grid, fp, strategy)

end # module
