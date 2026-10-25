module CoarseGrainingEnergyFluxesFFTWExt

using FFTW: FFTW
using LinearAlgebra: LinearAlgebra as LA
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF
using FlowGeometries: FlowGeometries

# Spectral filtering for uniform Cartesian grids: a pointwise multiply by the kernel's transfer function
# `CGEF.Kernels.spectral_transfer`, with `Ĝ(0) = 1`.
#
# A periodic axis is transformed at its own length. A bounded one is zero-padded to
# `2·nextprod((2,3,5), N)`, the least even 2·3·5-smooth length of at least `2N`, and the result cropped
# back to the grid: the field is extended by zero beyond the domain, the kernel's full mass is in the
# normalization, and the wrap-around path between two cells of the grid is at least the domain length.
#
# Masking is normalized convolution (Knutsson & Westin 1993): `ZeroFill` filters `mask·field`, and
# `Deformable` also divides by `filter(mask)`, the local kernel mass over active cells, computed at plan
# build and stored inverted.

"""
    FFTWGridPlan

The half of an FFT spectral plan the filter scale does not reach: the forward and inverse transforms
over the transform grid, the mask, the angular wavenumber grids, the execution backend whose memory the
transforms run in, and — when the plan was built for a trailing batch axis — the transforms bound to that
shape. One instance serves every scale of a sweep, and it is immutable during an apply, so concurrent
workers may share it.

Only `transfer` — and, for `Deformable`, the renormalization computed through it — depends on ℓ; every
buffer written during an apply lives in [`FFTWScratch`](@ref).
"""
struct FFTWGridPlan{
    T<:AbstractFloat, FP, IP, M, VX<:AbstractVector{T}, VY<:AbstractVector{T}, BT,
    B<:CGEF.ComputationalBackends.AbstractExecutionBackend,
} <: CGEF.Filtering.AbstractGridPlan
    fwd::FP        # plan_rfft
    inv::IP        # plan_irfft
    mask::M        # the mask in the backend's memory, or nothing when fully active
    kx::VX
    ky::VY
    dims::NTuple{2,Int}   # the grid
    P::NTuple{2,Int}      # the transform grid: `dims` along periodic axes, padded along bounded ones
    padded::Bool
    batched::BT    # (; fwd, inv, nb) for a trailing batch axis, or nothing
    execution::B
end

# FFTW plans a host array at a planner thread count; any other array plans through the AbstractFFTs
# provider of its own type (CUFFT for a `CuArray`).
_plan_rfft(A::Array, nt::Int, region...) = FFTW.plan_rfft(A, region...; num_threads = nt)
_plan_rfft(A::AbstractArray, ::Int, region...) = FFTW.plan_rfft(A, region...)
_plan_irfft(A::Array, n::Int, nt::Int, region...) = FFTW.plan_irfft(A, n, region...; num_threads = nt)
_plan_irfft(A::AbstractArray, n::Int, ::Int, region...) = FFTW.plan_irfft(A, n, region...)

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
CGEF.Filtering.plan_execution(plan::FFTWFilterPlan) = plan.grid_plan.execution

CGEF.Filtering._batched_fields(outs, plan::FFTWFilterPlan) =
    plan.grid_plan.batched !== nothing && ndims(first(outs)) == 3

# The transfer function, mask and renormalization are spatial and broadcast across the batch axis.
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

# Analysis is scale-independent, so a sweep transforms each field once and each scale multiplies by its
# transfer function and inverts.
CGEF.Filtering.analyze_buffer(plan::FFTWFilterPlan, field::AbstractMatrix) = similar(plan.scratch.cbuf)

# `nothing` unless the plan holds transforms bound to this batch extent; the caller then applies per scale.
CGEF.Filtering.analyze_buffer(plan::FFTWFilterPlan, field::AbstractArray{<:Any,3}) =
    (plan.grid_plan.batched === nothing || plan.grid_plan.batched.nb != size(field, 3)) ? nothing :
        similar(plan.scratch.cbuf, size(plan.scratch.cbuf)..., size(field, 3))

function CGEF.Filtering.filter_analyze!(
    F̂::AbstractMatrix{Complex{T}}, field::AbstractMatrix{T}, plan::FFTWFilterPlan{T},
) where {T<:AbstractFloat}
    gp = plan.grid_plan
    return _fftw_forward!(F̂, gp.fwd, field, plan.scratch.masked_input, gp)
end

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

# Two `UniformSpacing` axes (FlowGeometries' `spacing_trait`) take FFTW; any other structured grid, with
# a stretched axis or a rank other than two, is filtered over its cells as a node set.
const _FFTAxes = Tuple{FlowGeometries.Axes.UniformSpacing, FlowGeometries.Axes.UniformSpacing}
@inline _fft_axes(grid::FlowGeometries.Grids.StructuredGrid) =
    map(FlowGeometries.Axes.spacing_trait, FlowGeometries.Grids.coordinates(grid))

function _fftw_grid_plan(
    grid::FlowGeometries.Grids.StructuredGrid{T,G};
    execution::CGEF.ComputationalBackends.AbstractExecutionBackend = CGEF.ComputationalBackends.AutoBackend(),
    batch::Union{Nothing,Integer} = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    nt = CGEF.Filtering._library_threads(execution)
    (; dims, P, padded, kx, ky) = CGEF.Filtering._fft_layout(grid)
    Nx, Ny = dims
    Px, Py = P

    sample = CGEF.Filtering._allocate(execution, T, (Px, Py))
    fwd = _plan_rfft(sample, nt)
    inv = _plan_irfft(CGEF.Filtering._allocate(execution, Complex{T}, (Px ÷ 2 + 1, Py)), Px, nt)
    m = FlowGeometries.Grids.mask(grid)
    mask = all(m) ? nothing : CGEF.Filtering._on_backend(execution, m)
    batched = if batch === nothing
        nothing
    else
        nb = Int(batch)
        bfwd = _plan_rfft(CGEF.Filtering._allocate(execution, T, (Px, Py, nb)), nt, (1, 2))
        binv = _plan_irfft(CGEF.Filtering._allocate(execution, Complex{T}, (Px ÷ 2 + 1, Py, nb)), Px, nt, (1, 2))
        (fwd = bfwd, inv = binv, nb = nb)
    end
    return FFTWGridPlan(fwd, inv, mask, kx, ky, (Nx, Ny), (Px, Py), padded, batched, execution)
end

function _fftw_scratch(gp::FFTWGridPlan{T}) where {T<:AbstractFloat}
    Px, Py = gp.P
    b = gp.batched
    zs(S, dims...) = CGEF.Filtering._allocate(gp.execution, S, dims)
    return FFTWScratch(
        zs(Complex{T}, Px ÷ 2 + 1, Py),
        zs(T, Px, Py),   # touched when masked or padded
        gp.padded ? zs(T, Px, Py) : nothing,
        b === nothing ? nothing :
            (cbuf = zs(Complex{T}, Px ÷ 2 + 1, Py, b.nb), masked_input = zs(T, Px, Py, b.nb),
             pad_out = gp.padded ? zs(T, Px, Py, b.nb) : nothing),
    )
end

CGEF.Filtering.spectral_grid_plan(
    sb::Union{CGEF.SpectralBackends.AbstractAutoSpectralBackend, CGEF.SpectralBackends.AbstractFFTSpectralBackend},
    grid::FlowGeometries.Grids.StructuredGrid{T,G},
    kernel::CGEF.Kernels.AbstractFilterKernel;
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}} =
    _fftw_grid_plan_for(_fft_axes(grid), sb, grid, kernel; kwargs...)

_fftw_grid_plan_for(
    ::_FFTAxes, _sb, grid, _kernel;
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy = CGEF.Filtering.ZeroFill(),
    execution::CGEF.ComputationalBackends.AbstractExecutionBackend = CGEF.ComputationalBackends.AutoBackend(),
    batch::Union{Nothing,Integer} = nothing,
) = _grid_plan_for(grid, execution, batch)
_fftw_grid_plan_for(::Tuple, sb, grid, kernel; kwargs...) =
    CGEF.Filtering._node_set_grid_plan(sb, grid, kernel; kwargs...)

# In this process, or divided among the worker processes.
_grid_plan_for(grid, backend::CGEF.ComputationalBackends.AbstractDistributedBackend, batch) =
    CGEF.Filtering.distributed_fft_grid_plan(grid; execution = backend, batch = batch)
_grid_plan_for(grid, backend::CGEF.ComputationalBackends.AbstractExecutionBackend, batch) =
    _fftw_grid_plan(grid; execution = backend, batch = batch)
CGEF.Filtering.spectral_scratch(gp::FFTWGridPlan) = _fftw_scratch(gp)

CGEF.Filtering.spectral_filter_plan(
    sb::Union{CGEF.SpectralBackends.AbstractAutoSpectralBackend, CGEF.SpectralBackends.AbstractFFTSpectralBackend},
    grid::FlowGeometries.Grids.StructuredGrid{T,G},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T;
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}} =
    _fftw_filter_plan(_fft_axes(grid), sb, grid, kernel, scale; kwargs...)

_fftw_filter_plan(::Tuple, sb, grid, kernel, scale; kwargs...) =
    CGEF.Filtering._node_set_filter_plan(sb, grid, kernel, scale; kwargs...)

function _fftw_filter_plan(
    ::_FFTAxes, _,
    grid::FlowGeometries.Grids.StructuredGrid{T,G},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy = CGEF.Filtering.ZeroFill(),
    execution::CGEF.ComputationalBackends.AbstractExecutionBackend = CGEF.ComputationalBackends.AutoBackend(),
    # Extent of the trailing batch axis this plan will be applied over, or `nothing` for single fields.
    # A transform is bound to one field shape, so it is fixed at planning.
    batch::Union{Nothing,Integer} = nothing,
    grid_plan::Union{Nothing,CGEF.Filtering.AbstractGridPlan} = nothing,
    scratch::Union{Nothing,FFTWScratch} = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    gp = grid_plan === nothing ? _grid_plan_for(grid, execution, batch) : grid_plan
    return _filter_plan(gp, scratch, kernel, scale, mask_strategy)
end

_filter_plan(gp::CGEF.Filtering.AbstractGridPlan, _, kernel, scale, mask_strategy) =
    CGEF.Filtering.distributed_fft_filter_plan(gp, kernel, scale, mask_strategy)

function _filter_plan(
    gp::FFTWGridPlan{T}, scratch, kernel::CGEF.Kernels.AbstractFilterKernel, scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
) where {T<:AbstractFloat}
    sc = scratch === nothing ? _fftw_scratch(gp) : scratch
    kx, ky = gp.kx, gp.ky

    transfer = CGEF.Filtering._on_backend(gp.execution,
        T[CGEF.Kernels.spectral_transfer(kernel, (kx[i], ky[j]), scale) for i in eachindex(kx), j in eachindex(ky)])

    mask = gp.mask
    invrenorm = if mask_strategy isa CGEF.Filtering.Deformable && (mask !== nothing || gp.padded)
        # Local kernel mass over active cells, `filter(mask)`, through the plan's own transforms; the
        # padding beyond a bounded axis is inactive.
        region = _fftw_region(sc.masked_input, gp.dims)
        mask === nothing ? fill!(region, one(T)) : (region .= mask)
        LA.mul!(sc.cbuf, gp.fwd, sc.masked_input)
        sc.cbuf .*= transfer
        renorm = CGEF.Filtering._allocate(gp.execution, T, gp.dims)
        _fftw_inverse!(renorm, gp.inv, sc.cbuf, sc.pad_out, gp)
        # A masked cell is zero under `Deformable`, as in the real-space engines.
        mask === nothing ? CGEF.Filtering._inv_mass.(renorm) :
            ifelse.(mask, CGEF.Filtering._inv_mass.(renorm), zero(T))
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
# A direct footprint sum, evaluated by transform: a radial kernel's weights over the disk of its radius,
# a separable kernel's the separable engine's per-axis tables multiplied. A periodic axis keeps its
# length, and the sampled kernel is accumulated modulo it, the sum over every image. A bounded axis is
# zero-padded to at least `N + min(w, N - 1)`, `w` the support's half-width in cells, the least length
# at which the circular convolution equals the linear one.
#
# The numerator is `conv(mask·f, g)`; the denominator depends on grid, kernel, scale and mask strategy
# alone and is built with the plan, as the real-space engine for the kernel builds it.
struct PaddedFFTFootprint{T<:AbstractFloat, A<:AbstractArray{T}, C<:AbstractArray{Complex{T}}, FP, IP, S, D}
    Ĝ::C            # kernel spectrum on the transform grid
    den::A          # the normalization over the grid, built for `strategy`
    pad::A          # transform-grid scratch for mask·field
    num::A          # transform-grid scratch for the inverse transform
    spec::C         # spectrum scratch
    fwd::FP
    inv::IP
    strategy::S     # the strategy `den` was built for; `apply_footprint!` refuses any other
    dims::NTuple{D,Int}
    P::NTuple{D,Int}   # transform length per axis
end

function CGEF.Filtering.padded_fft_footprint(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy = CGEF.Filtering.ZeroFill(),
    threads::Integer = 1,
) where {T<:AbstractFloat, N, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    dims = FlowGeometries.Grids.size_tuple(grid)
    # A periodic axis reaches every image inside the support; a bounded one no farther than its extent.
    w = CGEF.Filtering._padded_halfwidths(grid, kernel, scale)
    P = ntuple(d -> CGEF.Filtering._padded_length(dims[d], w[d], FlowGeometries.Grids.isperiodic(grid, d)), Val(N))

    # The weights wrapped so index (1,…,1) is the zero offset.
    gpad = zeros(T, P)
    _sample_kernel!(gpad, grid, kernel, scale, w)
    fwd = FFTW.plan_rfft(gpad; num_threads = threads)
    Ĝ = fwd * gpad
    spec = similar(Ĝ)
    inv = FFTW.plan_irfft(spec, P[1]; num_threads = threads)
    den = _padded_denominator(grid, kernel, scale, mask_strategy, fwd, inv, Ĝ, P)
    return PaddedFFTFootprint(Ĝ, den, zeros(T, P), zeros(T, P), similar(Ĝ), fwd, inv, mask_strategy, dims, P)
end

# The offset `o` wrapped onto a transform grid of lengths `P`, the zero offset at index (1,…,1).
@inline _wrapped(o::CartesianIndex{N}, P::NTuple{N,Int}) where {N} = CartesianIndex(ntuple(d -> mod1(1 + o[d], P[d]), Val(N)))

# A radial kernel over the ball `d <= rad` of the direct footprint within half-widths `w`, each weight
# times the cell measure.
function _sample_kernel!(gpad::AbstractArray{T,N}, grid, kernel, scale::T, w::NTuple{N,Int}) where {T,N}
    Δ = ntuple(d -> CGEF.Filtering._axis_step(grid, d), Val(N))
    rad = CGEF.Kernels.kernel_radius(kernel, scale, Val(N))
    A = T(FlowGeometries.Grids.measure(grid, ntuple(_ -> 1, Val(N))...))
    for o in CartesianIndices(map(wd -> (-wd):wd, w))
        d = hypot(ntuple(k -> o[k] * Δ[k], Val(N))...)
        d <= rad || continue
        wt = CGEF.Kernels.kernel_weight(kernel, d, scale, Val(N)) * A
        iszero(wt) && continue
        gpad[_wrapped(o, size(gpad))] += wt
    end
    return gpad
end

# A separable kernel: the separable engine's per-axis tables over its box stencil.
function _sample_kernel!(gpad::AbstractArray{T,N}, grid, kernel::CGEF.Filtering.SeparableKernel, scale::T,
                         w::NTuple{N,Int}) where {T,N}
    (; g, lim) = CGEF.Filtering._separable_tables(grid, kernel, scale)
    for o in CartesianIndices(map(wd -> (-wd):wd, w))
        gpad[_wrapped(o, size(gpad))] += prod(ntuple(k -> g[k][o[k] + lim[k] + 1], Val(N)))
    end
    return gpad
end

# The kernel mass over the active cells at each cell, through the plan's transform.
function _transform_mass(grid, fwd, inv, Ĝ, P)
    dims = FlowGeometries.Grids.size_tuple(grid)
    T = real(eltype(Ĝ))
    maskv = FlowGeometries.Grids.mask(grid)
    dpad = zeros(T, P)
    @inbounds for I in CartesianIndices(dims)
        dpad[I] = maskv[I] ? one(T) : zero(T)
    end
    spec = fwd * dpad
    spec .*= Ĝ
    return (inv * spec)[CartesianIndices(dims)]
end

# A radial kernel under `ZeroFill`: the grid's lattice continues uniformly past a bounded edge, so a cell's
# in-domain and exterior mass add to the kernel's mass over the whole ball of lattice offsets, the same at
# every cell. Under `Deformable`, the mass over the active cells.
function _padded_denominator(grid, kernel, scale, strategy, fwd, inv, Ĝ, P)
    strategy isa CGEF.Filtering.ZeroFill || return _transform_mass(grid, fwd, inv, Ĝ, P)
    T = real(eltype(Ĝ))
    N = length(P)
    Δ = ntuple(d -> CGEF.Filtering._axis_step(grid, d), N)
    rad = CGEF.Kernels.kernel_radius(kernel, scale, Val(N))
    A = T(FlowGeometries.Grids.measure(grid, ntuple(_ -> 1, N)...))
    mass = zero(T)
    for o in CartesianIndices(ntuple(d -> (-floor(Int, rad / Δ[d])):floor(Int, rad / Δ[d]), N))
        d = hypot(ntuple(k -> o[k] * Δ[k], N)...)
        d <= rad && (mass += CGEF.Kernels.kernel_weight(kernel, d, scale, Val(N)) * A)
    end
    return fill(mass, FlowGeometries.Grids.size_tuple(grid))
end

# A separable kernel's is the separable engine's: the product of the per-axis profiles, every offset
# counted under `ZeroFill` and the in-domain ones under `Deformable` on an unmasked grid, and the mass
# over the active cells under `Deformable` on a masked one.
function _padded_denominator(grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, kernel::CGEF.Filtering.SeparableKernel,
                             scale, strategy, fwd, inv, Ĝ, P) where {T,G,N}
    zerofill = strategy isa CGEF.Filtering.ZeroFill
    zerofill || all(FlowGeometries.Grids.mask(grid)) || return _transform_mass(grid, fwd, inv, Ĝ, P)
    (; g, lim, periodic, mf) = CGEF.Filtering._separable_tables(grid, kernel, scale)
    prof = ntuple(Val(N)) do d
        CGEF.Filtering._separable_profile(FlowGeometries.Grids.coordinates(grid, d), lim[d], g[d], periodic[d], kernel,
                                          scale, convert(AbstractVector{T}, mf[d]), zerofill)
    end
    return [prod(ntuple(d -> prof[d][I[d]], Val(N))) for I in CartesianIndices(FlowGeometries.Grids.size_tuple(grid))]
end

# ---------------------------------------------------------------------------
# Zonal-FFT engine: the compact great-circle kernel, evaluated by transform along longitude.
#
# On uniform longitude the great-circle distance between a target in latitude row `j` and a source in row
# `jj` depends on the longitude offset alone, and a cell's area on its latitude alone, so the direct sum's
# weight is a function `w_{j,jj}(di)` of the index offset and each band is a cross-correlation along
# longitude:
#
#     out[i, j] = Σ_jj Σ_di w_{j,jj}[di] · src[i+di, jj]
#
# The taps are the banded engine's (`_sphere_band_rows`, or `_highorder_rows`), the same offsets and
# weights. `w_{j,jj}` is even in `di`, so its spectrum is real and is stored as reals. On a ring of `Nx`
# columns the correlation is circular of length `Nx`. On a regional box longitude is zero-padded to
# `Px ≥ Nx + a`, `a` the widest offset among the taps, where the circular correlation is the linear one.
#
# A radial kernel's weight is `G(d)·A_jj`, with `d` symmetric in the two rows and `A_jj` the source row's
# cell area, so one spectrum of `G` serves both orders of a pair of rows: `Ŵ[:, |dj| + 1, min(j, jj)]`,
# and each source row's spectrum is multiplied by its area. A `HighOrderKernel`'s weights are corrected
# per target row, so it keeps one spectrum per ordered pair.
# ---------------------------------------------------------------------------

struct ZonalFFTFootprint{
    T<:AbstractFloat, A<:AbstractMatrix{T}, R3<:AbstractArray{T,3}, AR,
    C<:AbstractMatrix{Complex{T}}, FP, IP, MS<:CGEF.Filtering.AbstractMaskStrategy,
}
    Ŵ::R3           # real kernel spectra: (Px÷2+1, dj_lim+1, Ny) per unordered row pair when `area` is a
                    # vector, (Px÷2+1, 2·dj_lim+1, Ny) per (band offset, target row) when it is `nothing`
    invden::A       # (Nx, Ny) reciprocal window mass, built through this same transform
    dj_lim::Int
    area::AR        # (Ny) cell area per latitude row, or `nothing`
    bound::Bool     # the strategies' denominators differ: a cell is inactive or a window leaves the grid
    strategy::MS    # the strategy `invden` was built for
    src::A          # (Px, Ny) scratch for `mask · field`, zero past column `Nx`
    wide::A         # (Px, Ny) scratch: the correlation on the padded axis; empty on a ring, `Px == Nx`
    F::C            # (Px÷2+1, Ny) scratch: every source row's spectrum
    acc::C          # (Px÷2+1, Ny) scratch: the accumulated spectrum per target row
    fwd::FP         # batched rfft along longitude
    inv::IP
end

# The band sum, in the transform domain. Target row `j` writes only column `j` of `acc`, so `rows` may
# visit the rows in any order or in parallel.
function _zonal_accumulate!(
    out::AbstractMatrix{T}, src::AbstractMatrix{T}, fp::ZonalFFTFootprint{T}, Ny::Int,
    rows::D = CGEF.Filtering._sep_serial,
) where {T<:AbstractFloat, D}
    LA.mul!(fp.F, fp.fwd, src)
    _scale_rows!(fp.F, fp.area)
    _band_sum!(fp.acc, fp.Ŵ, fp.F, fp.dj_lim, fp.area, Ny, rows)
    acc = fp.acc
    Nx = size(out, 1)
    if size(fp.src, 1) == Nx
        LA.mul!(out, fp.inv, acc)
    else
        LA.mul!(fp.wide, fp.inv, acc)
        out .= view(fp.wide, 1:Nx, :)
    end
    return out
end

_scale_rows!(F::AbstractMatrix, ::Nothing) = F
function _scale_rows!(F::AbstractMatrix, area::AbstractVector)
    @inbounds for j in axes(F, 2)
        a = area[j]
        @simd for n in axes(F, 1)
            F[n, j] *= a
        end
    end
    return F
end

# `acc[:, j] = Σ_dj Ŵ_{j, j+dj} F[:, j+dj]` over the rows in the grid, from either layout of `Ŵ`.
function _band_sum!(acc::AbstractMatrix{Complex{T}}, Ŵ::AbstractArray{T,3}, F::AbstractMatrix{Complex{T}},
                    dj_lim::Int, area, Ny::Int, rows::D) where {T<:AbstractFloat, D}
    nk = size(F, 1)
    sym = area !== nothing
    rows(1:Ny) do j
        @inbounds begin
            @simd for n in 1:nk
                acc[n, j] = zero(Complex{T})
            end
            for dj in (-dj_lim):dj_lim
                jj = j + dj
                (1 <= jj <= Ny) || continue
                b, r = sym ? (abs(dj) + 1, min(j, jj)) : (dj + dj_lim + 1, j)
                @simd for n in 1:nk
                    acc[n, j] += Ŵ[n, b, r] * F[n, jj]
                end
            end
        end
        return nothing
    end
    return acc
end

function CGEF.Filtering.zonal_fft_footprint(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy = CGEF.Filtering.ZeroFill(),
    threads::Integer = 1,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.SphericalGeometry{T}}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    rows = kernel isa CGEF.Kernels.HighOrderKernel ? CGEF.Filtering._highorder_rows(grid, kernel, scale) :
        CGEF.Filtering._sphere_band_rows(grid, kernel, scale, CGEF.Filtering._query_radius(grid, kernel, scale))
    di, dj, w, ptr = rows
    dj_lim = maximum(abs, dj; init = 0)
    area = kernel isa CGEF.Kernels.HighOrderKernel ? nothing :
        T[FlowGeometries.Grids.area(grid, 1, j) for j in 1:Ny]
    nb = area === nothing ? 2 * dj_lim + 1 : dj_lim + 1
    Px = FlowGeometries.Grids.isperiodic(grid, 1) ? Nx : nextprod((2, 3, 5), Nx + maximum(abs, di; init = 0))

    sample = zeros(T, Px, Ny)
    fwd = FFTW.plan_rfft(sample, 1; num_threads = threads)
    Fbuf = fwd * sample
    iplan = FFTW.plan_irfft(Fbuf, Px, 1; num_threads = threads)
    nk = size(Fbuf, 1)

    Ŵ = zeros(T, nk, nb, Ny)
    gcol = zeros(T, Px, nb)
    gplan = FFTW.plan_rfft(gcol, 1; num_threads = threads)
    Ĝ = gplan * gcol
    for j in 1:Ny
        fill!(gcol, zero(T))
        for q in ptr[j]:(ptr[j + 1] - 1)
            if area === nothing
                gcol[mod(di[q], Px) + 1, dj[q] + dj_lim + 1] += w[q]
            elseif dj[q] >= 0
                gcol[mod(di[q], Px) + 1, dj[q] + 1] += w[q] / area[j + dj[q]]
            end
        end
        LA.mul!(Ĝ, gplan, gcol)
        @inbounds for b in 1:nb, n in 1:nk
            Ŵ[n, b, j] = real(Ĝ[n, b])
        end
    end

    bound = !all(FlowGeometries.Grids.mask(grid)) || CGEF.Filtering._reaches_exterior(grid, kernel, scale)
    fp = ZonalFFTFootprint(
        Ŵ, zeros(T, Nx, Ny), dj_lim, area, bound, mask_strategy,
        zeros(T, Px, Ny), Px == Nx ? zeros(T, 0, 0) : zeros(T, Px, Ny), Fbuf, similar(Fbuf), fwd, iplan,
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
    if zerofill && kernel isa CGEF.Kernels.HighOrderKernel
        # Each row's whole support mass, the continued lattice included.
        mass = rows[5]
        @inbounds for j in 1:Ny, i in 1:Nx
            den[i, j] = mass[j]
        end
    else
        _zonal_accumulate!(den, fp.src, fp, Ny)
        exterior = zerofill ? CGEF.Filtering._exterior_mass(grid, kernel, scale) : nothing
        exterior === nothing || (den .+= exterior)
    end
    @inbounds for j in 1:Ny, i in 1:Nx
        fp.invden[i, j] = (zerofill || maskv[i, j]) ? CGEF.Filtering._inv_mass(den[i, j]) : zero(T)
    end
    return fp
end

CGEF.Filtering._driven_footprint(::Union{ZonalFFTFootprint, PaddedFFTFootprint}) = true

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

# `rows(f, 1:n)` visits every index along the last direction; each writes only its own slab. The padding
# past the grid's cells is zero from construction, and the forward transform leaves its input as it is.
function CGEF.Filtering.apply_footprint!(
    out::AbstractArray{T,N}, field::AbstractArray,
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N},
    fp::PaddedFFTFootprint{T}, strategy::CGEF.Filtering.AbstractMaskStrategy,
    rows::D = CGEF.Filtering._sep_serial,
) where {T<:AbstractFloat, N, G<:FlowGeometries.Geometry.CartesianGeometry{T}, D}
    strategy === fp.strategy || throw(ArgumentError(
        "this padded-FFT footprint was built for $(typeof(fp.strategy)); its denominator is not the " *
        "one $(typeof(strategy)) needs. Rebuild the plan with the strategy you intend to apply.",
    ))
    dims = fp.dims
    slab = CartesianIndices(Base.front(dims))
    maskv = FlowGeometries.Grids.mask(grid)
    pad, num, den = fp.pad, fp.num, fp.den
    zerofill = strategy isa CGEF.Filtering.ZeroFill
    rows(1:dims[N]) do k
        @inbounds for I in slab
            J = CartesianIndex(Tuple(I)..., k)
            pad[J] = maskv[J] ? T(field[J]) : zero(T)
        end
        return nothing
    end
    LA.mul!(fp.spec, fp.fwd, pad)
    fp.spec .*= fp.Ĝ
    LA.mul!(num, fp.inv, fp.spec)
    rows(1:dims[N]) do k
        @inbounds for I in slab
            J = CartesianIndex(Tuple(I)..., k)
            out[J] = (zerofill || maskv[J]) ? CGEF.Filtering._normalized(num[J], den[J]) : zero(T)
        end
        return nothing
    end
    return out
end

CGEF.Filtering._apply_serial!(out, field, grid, fp::PaddedFFTFootprint, strategy) =
    CGEF.Filtering.apply_footprint!(out, field, grid, fp, strategy)

# ---------------------------------------------------------------------------
# The longitude step of `RingSpinTransforms` on a whole ring of uniform longitude
# ---------------------------------------------------------------------------
#
# With `φ_i = φ₁ + σ(i − 1)·2π/n`, `σ = ±1` the axis's direction, the row sum of order `m` is
# `e^{−imφ₁}` times bin `mod(σm, n)` of the row's discrete Fourier transform, and the synthesis puts each
# order's term `e^{imφ₁}·ring[m]` in that bin and transforms back unnormalized. Orders `m` and `m + n`
# share a bin, as `e^{imφ_i}` does on the ring.

"""
    FFTLongitudes

The longitude step of `Filtering.RingSpinTransforms` by one FFT per row, on a grid whose longitude is a
`FlowGeometries.Axes.PeriodicAxis`: the in-place transforms of the `(nlon, nlat)` scratch they are
planned on, the phases `e^{−imφ₁}` and each order's bin.
"""
struct FFTLongitudes{P,Q,V,I,B}
    fwd::P
    bwd::Q
    phase::V
    bin::I
    execution::B
end

_plan_fft!(A::Array, nt::Int) = FFTW.plan_fft!(A, 1; num_threads = nt)
_plan_fft!(A::AbstractArray, ::Int) = FFTW.plan_fft!(A, 1)
_plan_bfft!(A::Array, nt::Int) = FFTW.plan_bfft!(A, 1; num_threads = nt)
_plan_bfft!(A::AbstractArray, ::Int) = FFTW.plan_bfft!(A, 1)

function CGEF.Filtering._ring_longitudes(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2,S,TP,<:Tuple{FlowGeometries.Axes.PeriodicAxis,AbstractVector}},
    φ::AbstractVector{T}, L::Int, wf, execution,
) where {T<:AbstractFloat, G, S, TP}
    n = length(φ)
    σ = FlowGeometries.Axes.wrap_sign(FlowGeometries.Grids.coordinates(grid, 1)) > 0 ? 1 : -1
    nt = CGEF.Filtering._library_threads(execution)
    put(a) = FlowGeometries.Execution.on_backend(execution, a)
    return FFTLongitudes(_plan_fft!(wf, nt), _plan_bfft!(wf, nt), put([cis(-T(m) * φ[1]) for m in -L:L]),
                         put([mod(σ * m, n) + 1 for m in -L:L]), execution)
end

function CGEF.Filtering._longitude_sums!(ring, wf, lon::FFTLongitudes)
    lon.fwd * wf
    phase, bin = lon.phase, lon.bin
    FlowGeometries.Execution.run_indices(size(ring, 2), lon.execution) do j
        @inbounds for col in axes(ring, 1)
            ring[col, j] = phase[col] * wf[bin[col], j]
        end
        return nothing
    end
    return ring
end

function CGEF.Filtering._longitude_synthesis!(F, ring, wf, lon::FFTLongitudes)
    phase, bin = lon.phase, lon.bin
    FlowGeometries.Execution.run_indices(size(ring, 2), lon.execution) do j
        @inbounds begin
            for i in axes(wf, 1)
                wf[i, j] = zero(eltype(wf))
            end
            for col in axes(ring, 1)
                wf[bin[col], j] += conj(phase[col]) * ring[col, j]
            end
        end
        return nothing
    end
    lon.bwd * wf
    copyto!(F, wf)
    return F
end

end # module
