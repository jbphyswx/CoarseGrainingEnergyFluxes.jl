module CoarseGrainingEnergyFluxesAbstractFFTsExt

using AbstractFFTs: AbstractFFTs
using LinearAlgebra: LinearAlgebra as LA
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF
using FlowGeometries: FlowGeometries

# The FFT engines, through the AbstractFFTs implementation of each array's type: FFTW's on the host, a
# device library's (CUFFT for a `CuArray`) on a device. A plan takes the planner options the provider's
# extension declares (`Filtering._fft_options`).

# Spectral filtering for uniform Cartesian grids: a pointwise multiply by the kernel's transfer function
# `CGEF.Kernels.spectral_transfer`, with `Ĝ(0) = 1`.
#
# A periodic axis is transformed at its own length. A bounded one is zero-padded to
# `Filtering._bounded_length`, at least `2N` and far enough that the kernel at the grid plan's largest
# scale stays clear of the domain's periodic copies, and the result cropped back to the grid: the field is
# extended by zero beyond the domain, and the kernel's full mass is in the normalization. The sharp kernel
# on a bounded axis takes the transfer of its lattice kernel (`Filtering._lattice_brick_wall`).
#
# Masking is normalized convolution (Knutsson & Westin 1993): `ZeroFill` filters `mask·field`, and
# `Deformable` also divides by `filter(mask)`, the local kernel mass over active cells, computed at plan
# build and stored inverted.

"""
    FFTGridPlan

The half of an FFT spectral plan a sweep shares: the forward and inverse transforms over the transform
grid, the mask, the angular wavenumber grids, the spacing, the distance every image of a cell lies
beyond the grid's cells, the execution backend whose memory the transforms run in, and — when the plan
was built for a trailing batch axis — the transforms bound to that shape. One instance serves every
scale of a sweep up to the one it was padded for, and it is immutable during an apply, so concurrent
workers may share it.

Only `transfer` — and, for `Deformable`, the renormalization computed through it — depends on ℓ; every
buffer written during an apply lives in [`FFTScratch`](@ref).
"""
struct FFTGridPlan{
    T<:AbstractFloat, FP, IP, M, VX<:AbstractVector{T}, VY<:AbstractVector{T}, IK<:NTuple{2,AbstractMatrix{Complex{T}}},
    TP<:NTuple{2,FlowGeometries.Grids.AbstractTopology}, BT, B<:CGEF.ComputationalBackends.AbstractExecutionBackend,
} <: CGEF.Filtering.AbstractGridPlan
    fwd::FP        # plan_rfft, or `_ColumnForward` where axis 2 is padded
    inv::IP        # plan_irfft, or `_ColumnInverse` where axis 2 is padded
    mask::M        # the mask in the backend's memory, or nothing on a grid whose mask is `AllActive`
    kx::VX
    ky::VY
    ik::IK         # `i k_x` along the first axis and `i k_y` along the second, zero at an even axis's ±P/2
    dims::NTuple{2,Int}   # the grid
    P::NTuple{2,Int}      # the transform grid: `dims` along periodic axes, padded along bounded ones
    Δ::NTuple{2,T}
    topology::TP          # the grid's
    reach::T
    batched::BT    # (; fwd, inv, nb) for a trailing batch axis, or nothing
    execution::B
end

# Plans at `nt` threads where the array's provider takes a thread count, with the caller's planner keywords.
_plan_rfft(A::AbstractArray, nt::Int, options, region...) =
    AbstractFFTs.plan_rfft(A, region...; CGEF.Filtering._fft_options(A, nt, options)...)
_plan_irfft(A::AbstractArray, n::Int, nt::Int, options, region...) =
    AbstractFFTs.plan_irfft(A, n, region...; CGEF.Filtering._fft_options(A, nt, options)...)
_plan_fft!(A::AbstractArray, nt::Int, options, region...) =
    AbstractFFTs.plan_fft!(A, region...; CGEF.Filtering._fft_options(A, nt, options)...)
_plan_bfft!(A::AbstractArray, nt::Int, options, region...) =
    AbstractFFTs.plan_bfft!(A, region...; CGEF.Filtering._fft_options(A, nt, options)...)
_plan_brfft(A::AbstractArray, n::Int, nt::Int, options, region...) =
    AbstractFFTs.plan_brfft(A, n, region...; CGEF.Filtering._fft_options(A, nt, options)...)

# The transforms of a transform grid zero-padded along axis 2, whose data fill its first `Ny` columns. Forward: the
# axis-1 real transform of those columns, the rest of the spectrum zeroed, then the axis-2 transform of every row
# in place. Inverse: the axis-2 transform back in place, which consumes the spectrum, then the axis-1 real transform
# of the first `Ny` columns, unnormalized; `scale` is applied as the grid's region is cropped (`_fft_inverse!`). The
# axis-1 transforms are planned on one field's columns and run field by field along a trailing batch axis, whose
# columns are not evenly strided.
struct _ColumnForward{F1, F2}
    f1::F1
    f2::F2
    Ny::Int
end

struct _ColumnInverse{B2, B1, T<:AbstractFloat}
    b2::B2
    b1::B1
    Ny::Int
    scale::T
end

# The first `n` columns of field `b`, and the columns past them of every field.
@inline _columns(A::AbstractArray, n::Int, b::Int) = view(A, :, 1:n, b)
@inline _beyond(A::AbstractMatrix, n::Int) = view(A, :, (n + 1):size(A, 2))
@inline _beyond(A::AbstractArray{<:Any,3}, n::Int) = view(A, :, (n + 1):size(A, 2), :)

function LA.mul!(dst::AbstractArray, p::_ColumnForward, src::AbstractArray)
    for b in axes(dst, 3)
        LA.mul!(_columns(dst, p.Ny, b), p.f1, _columns(src, p.Ny, b))
    end
    fill!(_beyond(dst, p.Ny), zero(eltype(dst)))
    p.f2 * dst
    return dst
end

function _unscaled_inverse!(pout::AbstractArray, p::_ColumnInverse, spec::AbstractArray)
    p.b2 * spec
    for b in axes(spec, 3)
        LA.mul!(_columns(pout, p.Ny, b), p.b1, _columns(spec, p.Ny, b))
    end
    return pout
end

# The forward and inverse transforms over a transform grid of `sample`'s shape, `region` its two spatial axes: by
# columns where axis 2 is padded and the arrays are host arrays, whole otherwise.
_fft_pair(sample::AbstractArray, csample::AbstractArray, Px::Int, nt::Int, options, ::Int, _topology) =
    (_plan_rfft(sample, nt, options, (1, 2)), _plan_irfft(csample, Px, nt, options, (1, 2)))

function _fft_pair(
    sample::Array{T}, csample::Array, Px::Int, nt::Int, options, Ny::Int,
    ::Tuple{FlowGeometries.Grids.AbstractTopology, FlowGeometries.Grids.Bounded},
) where {T<:AbstractFloat}
    Py = size(sample, 2)
    fwd = _ColumnForward(_plan_rfft(_columns(sample, Ny, 1), nt, options, 1), _plan_fft!(csample, nt, options, 2), Ny)
    bwd = _ColumnInverse(_plan_bfft!(csample, nt, options, 2), _plan_brfft(_columns(csample, Ny, 1), Px, nt, options, 1), Ny,
                         one(T) / (T(Px) * T(Py)))
    return fwd, bwd
end

"""
    FFTScratch

The transient half of an FFT spectral plan: the complex spectrum buffer, the transform-grid input
(`mask · field`, zero beyond the grid), the transform-grid output a padded plan crops from, and the same
set sized for a trailing batch axis. Every scale of a sweep shares one, since scales run sequentially; a
driver that applies concurrently gives each worker its own.
"""
struct FFTScratch{T<:AbstractFloat, CA<:AbstractMatrix{Complex{T}}, A<:AbstractMatrix{T}, PO, BT} <:
       CGEF.Filtering.AbstractFilterScratch
    cbuf::CA
    masked_input::A
    pad_out::PO    # transform-grid output, or nothing when unpadded
    batched::BT    # (; cbuf, masked_input, pad_out) sized for the batch axis, or nothing
end

# The grid's cells within a transform-grid array.
@inline _fft_region(A::AbstractMatrix, (Nx, Ny)) = view(A, 1:Nx, 1:Ny)
@inline _fft_region(A::AbstractArray{<:Any,3}, (Nx, Ny)) = view(A, 1:Nx, 1:Ny, :)

# The forward transform of `field` into `dst`: directly where nothing is masked or padded, otherwise
# of `mask · field` written into the grid's region of `buf`, whose padding stays zero.
function _fft_forward!(dst, fwd, field, buf, gp::FFTGridPlan)
    if CGEF.Filtering._padded(gp.topology)
        _fft_stage!(_fft_region(buf, gp.dims), field, gp.mask)
        LA.mul!(dst, fwd, buf)
    elseif gp.mask === nothing
        LA.mul!(dst, fwd, field)
    else
        _fft_stage!(buf, field, gp.mask)
        LA.mul!(dst, fwd, buf)
    end
    return dst
end

@inline _fft_stage!(data, field, ::Nothing) = (data .= field)
@inline _fft_stage!(data, field, mask) = (@. data = mask * field)

# The inverse transform into `out`; a padded plan holds a transform-grid `pout` and crops it.
@inline _fft_inverse!(out, inv, spec, ::Nothing, ::FFTGridPlan) = LA.mul!(out, inv, spec)
function _fft_inverse!(out, inv, spec, pout::AbstractArray, gp::FFTGridPlan)
    LA.mul!(pout, inv, spec)
    out .= _fft_region(pout, gp.dims)
    return out
end
function _fft_inverse!(out, inv::_ColumnInverse, spec, pout::AbstractArray, gp::FFTGridPlan)
    _unscaled_inverse!(pout, inv, spec)
    out .= _fft_region(pout, gp.dims) .* inv.scale
    return out
end

"""
    FFTFilterPlan

Cached FFT filter plan: the shared [`FFTGridPlan`](@ref), the precomputed transfer-function array, the
`Deformable` local-mass renormalization (or `nothing`), and the [`FFTScratch`](@ref) the apply writes
through. Built by `plan_filter(...; method = Spectral())`.
"""
struct FFTFilterPlan{
    T<:AbstractFloat, GP<:FFTGridPlan{T}, A<:AbstractMatrix{T}, R, DM,
    MS<:CGEF.Filtering.AbstractMaskStrategy, SC<:FFTScratch{T},
} <: CGEF.Filtering.AbstractFilterPlan
    grid_plan::GP
    transfer::A    # Ĝ(|k|, ℓ) on the rfft transform grid  (Px÷2+1, Py)
    invrenorm::R   # 1/filter(mask) for Deformable, zero at a masked cell; or nothing
    dmass::DM      # ∂filter(mask)/∂x and ∂/∂y beside `invrenorm`; or nothing
    strategy::MS
    scratch::SC
end

CGEF.Filtering.plan_strategy(plan::FFTFilterPlan) = plan.strategy
CGEF.Filtering.plan_execution(plan::FFTFilterPlan) = plan.grid_plan.execution
CGEF.Filtering.series_form(plan::FFTFilterPlan) = CGEF.Filtering._ratio_form(plan.invrenorm)

CGEF.Filtering.rate_available(::FFTFilterPlan, kernel::CGEF.Kernels.AbstractFilterKernel) =
    CGEF.Kernels.has_transfer_rate(kernel)
function CGEF.Filtering.rate_plan(plan::FFTFilterPlan{T}, kernel::CGEF.Kernels.AbstractFilterKernel, scale::T) where {T}
    gp = plan.grid_plan
    rate = CGEF.Filtering._fft_transfer(CGEF.Kernels.TransferRate(kernel), gp.topology, gp, scale)
    return FFTFilterPlan(gp, CGEF.Filtering._on_backend(gp.execution, rate), plan.invrenorm, plan.dmass, plan.strategy,
                         plan.scratch)
end

CGEF.Filtering._batched_fields(outs, plan::FFTFilterPlan) =
    plan.grid_plan.batched !== nothing && ndims(first(outs)) == 3

# The grid plan's transforms are shared by every worker; each holds its own scratch.
CGEF.Filtering._scratch(plan::FFTFilterPlan) = plan.scratch
CGEF.Filtering._with_scratch(plan::FFTFilterPlan, sc::FFTScratch) =
    FFTFilterPlan(plan.grid_plan, plan.transfer, plan.invrenorm, plan.dmass, plan.strategy, sc)
CGEF.Filtering._fresh(sc::FFTScratch) = FFTScratch(
    similar(sc.cbuf), zero(sc.masked_input), CGEF.Filtering._fresh_buffer(sc.pad_out), _fresh_set(sc.batched),
)
CGEF.Filtering._alike(a::S, b::S) where {S<:FFTScratch} =
    size(a.cbuf) == size(b.cbuf) && _set_extent(a.batched) == _set_extent(b.batched)
_fresh_set(::Nothing) = nothing
_fresh_set(b::NamedTuple) = map(CGEF.Filtering._fresh_buffer, b)
_set_extent(::Nothing) = nothing
_set_extent(b::NamedTuple) = size(b.cbuf)

# The transfer function, mask and renormalization are spatial and broadcast across the batch axis.
function CGEF.Filtering.filter_apply_batched!(
    out::AbstractArray{T,3}, field::AbstractArray{T,3}, plan::FFTFilterPlan{T},
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
    _fft_forward!(p.cbuf, b.fwd, field, p.masked_input, gp)
    p.cbuf .*= plan.transfer
    _fft_inverse!(out, b.inv, p.cbuf, p.pad_out, gp)
    plan.invrenorm === nothing || (out .*= plan.invrenorm)
    return out
end

# Analysis is scale-independent, so a sweep transforms each field once and each scale multiplies by its
# transfer function and inverts.
CGEF.Filtering.analyze_buffer(plan::FFTFilterPlan, field::AbstractMatrix) = similar(plan.scratch.cbuf)

# `nothing` for a plan holding no transforms bound to a batch; the caller then applies per scale.
CGEF.Filtering.analyze_buffer(plan::FFTFilterPlan, field::AbstractArray{<:Any,3}) = CGEF.Filtering._batch_analysis(
    plan.grid_plan.batched, size(field, 3), b -> b.nb,
    () -> similar(plan.scratch.cbuf, size(plan.scratch.cbuf)..., size(field, 3)),
)

function CGEF.Filtering.filter_analyze!(
    F̂::AbstractMatrix{Complex{T}}, field::AbstractMatrix{T}, plan::FFTFilterPlan{T},
) where {T<:AbstractFloat}
    gp = plan.grid_plan
    return _fft_forward!(F̂, gp.fwd, field, plan.scratch.masked_input, gp)
end

function CGEF.Filtering.filter_analyze!(
    F̂::AbstractArray{Complex{T},3}, field::AbstractArray{T,3}, plan::FFTFilterPlan{T},
) where {T<:AbstractFloat}
    gp, sc = plan.grid_plan, plan.scratch
    b = gp.batched
    (b === nothing || sc.batched === nothing) && throw(ArgumentError(
        "this spectral plan was not built for a batch; pass `batch = Nb` to `plan_filter`",
    ))
    b.nb == size(field, 3) || throw(DimensionMismatch(
        "plan was built for a batch of $(b.nb), got $(size(field, 3))",
    ))
    return _fft_forward!(F̂, b.fwd, field, sc.batched.masked_input, gp)
end

function CGEF.Filtering.filter_synthesize!(
    out::AbstractMatrix{T}, F̂::AbstractMatrix{Complex{T}}, plan::FFTFilterPlan{T},
) where {T<:AbstractFloat}
    # `inv` consumes its input, so synthesize from a copy — `F̂` is reused by every later scale.
    gp, sc = plan.grid_plan, plan.scratch
    sc.cbuf .= F̂ .* plan.transfer
    _fft_inverse!(out, gp.inv, sc.cbuf, sc.pad_out, gp)
    plan.invrenorm === nothing || (out .*= plan.invrenorm)
    return out
end

function CGEF.Filtering.filter_synthesize!(
    out::AbstractArray{T,3}, F̂::AbstractArray{Complex{T},3}, plan::FFTFilterPlan{T},
) where {T<:AbstractFloat}
    gp, sc = plan.grid_plan, plan.scratch
    b = gp.batched
    (b === nothing || sc.batched === nothing) && throw(ArgumentError(
        "this spectral plan was not built for a batch; pass `batch = Nb` to `plan_filter`",
    ))
    sc.batched.cbuf .= F̂ .* plan.transfer
    _fft_inverse!(out, b.inv, sc.batched.cbuf, sc.batched.pad_out, gp)
    plan.invrenorm === nothing || (out .*= plan.invrenorm)
    return out
end

# Two `UniformSpacing` axes (FlowGeometries' `spacing_trait`) take the FFT; any other structured grid,
# with a stretched axis or a rank other than two, is filtered over its cells as a node set, as is every
# grid under Auto when no FFT provider is loaded for the execution backend's memory. `f` is called on the
# spacing tuple, or on `()` for the node set.
const _FFTAxes = Tuple{FlowGeometries.Axes.UniformSpacing, FlowGeometries.Axes.UniformSpacing}
@inline _with_fft_axes(f::F, ::CGEF.SpectralBackends.AbstractFFTSpectralBackend, grid::FlowGeometries.Grids.StructuredGrid,
                       _) where {F} =
    CGEF.Filtering._with_spacings(f, grid)
@inline _with_fft_axes(f::F, ::CGEF.SpectralBackends.AbstractAutoSpectralBackend, grid::FlowGeometries.Grids.StructuredGrid,
                       execution) where {F} =
    CGEF.Filtering._fft_provided(execution) ? CGEF.Filtering._with_spacings(f, grid) : f(())

function _check_provided(execution)
    CGEF.Filtering._fft_provided(execution) || throw(ArgumentError(
        "no FFT library is loaded for the arrays of $(typeof(execution)): run `using FFTW` for host arrays, " *
        "or load the device's FFT package"))
    return nothing
end

function _fft_grid_plan(
    grid::FlowGeometries.Grids.StructuredGrid{T,G}, reach::T;
    execution::CGEF.ComputationalBackends.AbstractExecutionBackend = CGEF.ComputationalBackends.AutoBackend(),
    batch::Union{Nothing,Integer} = nothing,
    fft_options::NamedTuple = (;),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    nt = CGEF.Filtering._library_threads(execution)
    layout = CGEF.Filtering._fft_layout(grid, reach)
    (; dims, P, topology, kx, ky) = layout
    Nx, Ny = dims
    Px, Py = P

    fwd, inv = _fft_pair(CGEF.Filtering._allocate(execution, T, (Px, Py)),
                         CGEF.Filtering._allocate(execution, Complex{T}, (Px ÷ 2 + 1, Py)), Px, nt, fft_options, Ny, topology)
    mask = CGEF.Filtering._plan_mask(FlowGeometries.Grids.mask(grid), identity, execution)
    batched = if batch === nothing
        nothing
    else
        nb = Int(batch)
        bfwd, binv = _fft_pair(CGEF.Filtering._allocate(execution, T, (Px, Py, nb)),
                               CGEF.Filtering._allocate(execution, Complex{T}, (Px ÷ 2 + 1, Py, nb)), Px, nt, fft_options,
                               Ny, topology)
        (fwd = bfwd, inv = binv, nb = nb)
    end
    ik = map(a -> CGEF.Filtering._on_backend(execution, a), CGEF.Filtering._fft_ik((; kx, ky, P = (Px, Py))))
    return FFTGridPlan(fwd, inv, mask, kx, ky, ik, (Nx, Ny), (Px, Py), layout.Δ, topology, layout.reach, batched,
                       execution)
end

function _fft_scratch(gp::FFTGridPlan{T}) where {T<:AbstractFloat}
    Px, Py = gp.P
    b = gp.batched
    padded = CGEF.Filtering._padded(gp.topology)
    zs(S, dims...) = CGEF.Filtering._allocate(gp.execution, S, dims)
    return FFTScratch(
        zs(Complex{T}, Px ÷ 2 + 1, Py),
        zs(T, Px, Py),   # touched when masked or padded
        padded ? zs(T, Px, Py) : nothing,
        b === nothing ? nothing :
            (cbuf = zs(Complex{T}, Px ÷ 2 + 1, Py, b.nb), masked_input = zs(T, Px, Py, b.nb),
             pad_out = padded ? zs(T, Px, Py, b.nb) : nothing),
    )
end

CGEF.Filtering.spectral_grid_plan(
    sb::Union{CGEF.SpectralBackends.AbstractAutoSpectralBackend, CGEF.SpectralBackends.AbstractFFTSpectralBackend},
    grid::FlowGeometries.Grids.StructuredGrid{T,G},
    kernel::CGEF.Kernels.AbstractFilterKernel;
    execution::CGEF.ComputationalBackends.AbstractExecutionBackend = CGEF.ComputationalBackends.AutoBackend(),
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}} =
    _with_fft_axes(ax -> _fft_grid_plan_for(ax, sb, grid, kernel; execution, kwargs...), sb, grid, execution)

function _fft_grid_plan_for(
    ::_FFTAxes, _sb, grid::FlowGeometries.Grids.StructuredGrid{T}, kernel;
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy = CGEF.Filtering.ZeroFill(),
    execution::CGEF.ComputationalBackends.AbstractExecutionBackend = CGEF.ComputationalBackends.AutoBackend(),
    batch::Union{Nothing,Integer} = nothing,
    max_scale::Real = 0,
    fft_options::NamedTuple = (;),
) where {T}
    _check_provided(execution)
    return _grid_plan_for(grid, execution, batch, CGEF.Filtering._kernel_reach(kernel, max_scale, T, Val(2)), fft_options)
end
_fft_grid_plan_for(::Tuple, sb, grid, kernel; kwargs...) =
    CGEF.Filtering._node_set_grid_plan(sb, grid, kernel; kwargs...)

# In this process, or divided among the worker processes, padded for `reach`.
_grid_plan_for(grid, backend::CGEF.ComputationalBackends.AbstractDistributedBackend, batch, reach, options) =
    CGEF.Filtering.distributed_fft_grid_plan(grid, reach; execution = backend, batch = batch, fft_options = options)
_grid_plan_for(grid, backend::CGEF.ComputationalBackends.AbstractExecutionBackend, batch, reach, options) =
    _fft_grid_plan(grid, reach; execution = backend, batch = batch, fft_options = options)
CGEF.Filtering.spectral_scratch(gp::FFTGridPlan) = _fft_scratch(gp)

CGEF.Filtering.spectral_filter_plan(
    sb::Union{CGEF.SpectralBackends.AbstractAutoSpectralBackend, CGEF.SpectralBackends.AbstractFFTSpectralBackend},
    grid::FlowGeometries.Grids.StructuredGrid{T,G},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T;
    execution::CGEF.ComputationalBackends.AbstractExecutionBackend = CGEF.ComputationalBackends.AutoBackend(),
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}} =
    _with_fft_axes(ax -> _fft_filter_plan(ax, sb, grid, kernel, scale; execution, kwargs...), sb, grid, execution)

_fft_filter_plan(::Tuple, sb, grid, kernel, scale; kwargs...) =
    CGEF.Filtering._node_set_filter_plan(sb, grid, kernel, scale; kwargs...)

function _fft_filter_plan(
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
    scratch::Union{Nothing,FFTScratch} = nothing,
    fft_options::NamedTuple = (;),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    gp = grid_plan === nothing ?
        (_check_provided(execution);
         _grid_plan_for(grid, execution, batch, CGEF.Filtering._kernel_reach(kernel, scale, T, Val(2)), fft_options)) :
        grid_plan
    return _filter_plan(gp, scratch, kernel, scale, mask_strategy)
end

_filter_plan(gp::CGEF.Filtering.AbstractGridPlan, _, kernel, scale, mask_strategy) =
    CGEF.Filtering.distributed_fft_filter_plan(gp, kernel, scale, mask_strategy)

function _filter_plan(
    gp::FFTGridPlan{T}, scratch, kernel::CGEF.Kernels.AbstractFilterKernel, scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
) where {T<:AbstractFloat}
    CGEF.Filtering._check_reach(gp.reach, kernel, scale, Val(2))
    sc = scratch === nothing ? _fft_scratch(gp) : scratch
    transfer = CGEF.Filtering._on_backend(gp.execution, CGEF.Filtering._fft_transfer(kernel, gp.topology, gp, scale))

    # ZeroFill filters `mask .* field` with no renormalization.
    invrenorm, dmass = CGEF.Filtering._divides_by_mass(mask_strategy, gp.mask, gp.topology) ?
        _window_mass(gp, sc, transfer) : (nothing, nothing)
    return FFTFilterPlan(gp, transfer, invrenorm, dmass, mask_strategy, sc)
end

# The sharp kernel's lattice transfer on a box with a bounded direction (`Filtering._lattice_brick_wall`),
# formed by host transforms.
function CGEF.Filtering._lattice_brick_wall(
    tp::Tuple{FlowGeometries.Grids.Bounded,FlowGeometries.Grids.Bounded}, layout, scale::T,
) where {T<:AbstractFloat}
    kc = _lattice_cutoff(tp, layout.Δ, scale)
    (Nx, Ny), (Px, Py), (dx, dy) = layout.dims, layout.P, layout.Δ
    sharp = CGEF.Kernels.SharpSpectralKernel()
    c = dx * dy * kc^2 / (4 * T(π))
    h = zeros(T, Px, Py)
    for oy in 0:(Ny - 1), ox in 0:(Nx - 1)
        v = c * CGEF.Kernels.kernel_weight(sharp, hypot(ox * dx, oy * dy), scale, Val(2))
        for sy in (oy, -oy), sx in (ox, -ox)
            h[mod(sx, Px) + 1, mod(sy, Py) + 1] = v
        end
    end
    return real.(_host_rfft(h, 1:2))
end

function CGEF.Filtering._lattice_brick_wall(
    tp::Tuple{FlowGeometries.Grids.Periodic,FlowGeometries.Grids.Bounded}, layout, scale::T,
) where {T<:AbstractFloat}
    kc = _lattice_cutoff(tp, layout.Δ, scale)
    h = zeros(Complex{T}, length(layout.kx), layout.P[2])
    for (i, k) in enumerate(layout.kx)
        _brick_wall_line!(view(h, i, :), kc, k, layout.dims[2], layout.Δ[2])
    end
    return real.(_host_fft!(h, 2))
end

function CGEF.Filtering._lattice_brick_wall(
    tp::Tuple{FlowGeometries.Grids.Bounded,FlowGeometries.Grids.Periodic}, layout, scale::T,
) where {T<:AbstractFloat}
    kc = _lattice_cutoff(tp, layout.Δ, scale)
    h = zeros(T, layout.P[1], length(layout.ky))
    for (j, k) in enumerate(layout.ky)
        _brick_wall_line!(view(h, :, j), kc, k, layout.dims[1], layout.Δ[1])
    end
    return real.(_host_rfft(h, 1))
end

# The one-dimensional lattice brick wall `|k′| ≤ a = √(k_c² − k²)` over the offsets of `N` cells `Δ` apart,
# wrapped onto `line`: `Δ sin(a o Δ)/(π o Δ)`, `Δ a/π` at `o = 0`; zero where `|k| > k_c`.
function _brick_wall_line!(line::AbstractVector, kc::T, k::T, N::Int, Δ::T) where {T<:AbstractFloat}
    abs(k) > kc && return line
    a = sqrt(kc^2 - k^2)
    P = length(line)
    line[1] = Δ * a / T(π)
    for o in 1:(N - 1)
        v = sin(a * o * Δ) / (T(π) * o)
        line[o + 1] = v
        line[P - o + 1] = v
    end
    return line
end

# `π/ℓ`, raising where it passes the Nyquist wavenumber of a bounded direction.
function _lattice_cutoff(tp::Tuple, Δ::NTuple{2,T}, scale::T) where {T<:AbstractFloat}
    map((t, δ) -> _check_spacing(t, δ, scale), tp, Δ)
    return T(π) / scale
end

_check_spacing(::FlowGeometries.Grids.Periodic, _, _) = nothing
_check_spacing(::FlowGeometries.Grids.Bounded, Δ, scale) = scale < Δ ? _below_spacing(scale, Δ) : nothing

@noinline _below_spacing(scale, Δ) = throw(ArgumentError(
    "SharpSpectralKernel at scale $scale is below the spacing $Δ of a bounded direction, so its cutoff π/ℓ " *
    "passes that direction's Nyquist wavenumber; its spectral form on a bounded direction takes ℓ ≥ $Δ"))

_host_rfft(h::Array, region) = (_require_host_fft(); _plan_rfft(h, 1, (;), region) * h)
_host_fft!(h::Array, region) = (_require_host_fft(); _plan_fft!(h, 1, (;), region) * h)

_require_host_fft() = CGEF.Filtering._available(CGEF.Filtering._HostFFT()) || throw(ArgumentError(
    "the sharp kernel's transfer on a bounded direction is formed by an FFT of host arrays; load an FFT " *
    "library for them (`using FFTW`)"))

# `1/M` and `∂M/∂x`, `∂M/∂y` for the local kernel mass over active cells, `M = filter(mask)`, through the
# plan's own transforms; the padding beyond a bounded axis is inactive, and a masked cell's `1/M` is zero,
# as in the real-space engines.
function _window_mass(gp::FFTGridPlan{T}, sc::FFTScratch{T}, transfer) where {T<:AbstractFloat}
    region = _fft_region(sc.masked_input, gp.dims)
    gp.mask === nothing ? fill!(region, one(T)) : (region .= gp.mask)
    M̂ = LA.mul!(similar(sc.cbuf), gp.fwd, sc.masked_input)
    # The inverse transform consumes its input, so each synthesis starts from `M̂`.
    synth(ik) = (sc.cbuf .= M̂ .* transfer .* ik;
                 _fft_inverse!(CGEF.Filtering._allocate(gp.execution, T, gp.dims), gp.inv, sc.cbuf, sc.pad_out, gp))
    M = synth(one(T))
    invmass = gp.mask === nothing ? CGEF.Filtering._inv_mass.(M) : ifelse.(gp.mask, CGEF.Filtering._inv_mass.(M), zero(T))
    return invmass, map(synth, gp.ik)
end

function CGEF.Filtering.filter_synthesize_derivative!(
    out::AbstractMatrix{T}, F̂::AbstractMatrix{Complex{T}}, ū::AbstractMatrix, plan::FFTFilterPlan{T}, d::Int,
) where {T<:AbstractFloat}
    gp, sc = plan.grid_plan, plan.scratch
    sc.cbuf .= F̂ .* plan.transfer .* gp.ik[d]
    _fft_inverse!(out, gp.inv, sc.cbuf, sc.pad_out, gp)
    return CGEF.Filtering._quotient_rule!(out, ū, plan.invrenorm, plan.dmass, d)
end

function CGEF.Filtering.filter_synthesize_derivative!(
    out::AbstractArray{T,3}, F̂::AbstractArray{Complex{T},3}, ū::AbstractArray{<:Any,3}, plan::FFTFilterPlan{T}, d::Int,
) where {T<:AbstractFloat}
    gp, sc = plan.grid_plan, plan.scratch
    b = gp.batched
    (b === nothing || sc.batched === nothing) && throw(ArgumentError(
        "this spectral plan was not built for a batch; pass `batch = Nb` to `plan_filter`",
    ))
    sc.batched.cbuf .= F̂ .* plan.transfer .* gp.ik[d]
    _fft_inverse!(out, b.inv, sc.batched.cbuf, sc.batched.pad_out, gp)
    return CGEF.Filtering._quotient_rule!(out, ū, plan.invrenorm, plan.dmass, d)
end

function CGEF.Filtering.filter_apply!(
    out::AbstractMatrix{T},
    field::AbstractMatrix{T},
    plan::FFTFilterPlan{T},
) where {T<:AbstractFloat}
    gp, sc = plan.grid_plan, plan.scratch
    _fft_forward!(sc.cbuf, gp.fwd, field, sc.masked_input, gp)   # f̂ = rfft(mask · field)
    sc.cbuf .*= plan.transfer                                     # ĝ · f̂
    _fft_inverse!(out, gp.inv, sc.cbuf, sc.pad_out, gp)          # irfft (consumes cbuf)
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
# alone and is built with the plan, as the real-space engine for the kernel builds it. Every array lives
# in the execution backend's memory: the kernel is sampled on the host and moved, and the transforms run
# where the arrays are.
struct PaddedFFTFootprint{T<:AbstractFloat, A<:AbstractArray{T}, C<:AbstractArray{Complex{T}}, FP, IP, MK, S, D}
    Ĝ::C            # kernel spectrum on the transform grid
    den::A          # the normalization over the grid, built for `strategy`
    pad::A          # transform-grid scratch for mask·field
    num::A          # transform-grid scratch for the inverse transform
    spec::C         # spectrum scratch
    fwd::FP
    inv::IP
    mask::MK        # the grid's mask in the arrays' memory, `AllActive` as it is
    strategy::S     # the strategy `den` was built for
    bound::Bool     # the strategies' denominators differ: a cell is inactive or a window leaves the grid
    dims::NTuple{D,Int}
    P::NTuple{D,Int}   # transform length per axis
end

# The grid's mask in `execution`'s memory; `AllActive` holds only its size.
_resident_mask(_, m::FlowGeometries.Grids.AllActive) = m
_resident_mask(execution, m::AbstractArray{Bool}) = CGEF.Filtering._on_backend(execution, m)

CGEF.Filtering.padded_fft_footprint(
    sp::Tuple, grid::FlowGeometries.Grids.StructuredGrid{T,G}, kernel::CGEF.Kernels.AbstractFilterKernel, scale::T;
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}} =
    _padded_fft_footprint(sp, grid, kernel, scale; kwargs...)

_padded_fft_footprint(sp::Tuple, grid, kernel, scale; kwargs...) = throw(ArgumentError(
    "the padded-FFT engine needs a uniform axis in every direction; this grid's direction " *
    "$(join(_stretched(sp), ", ")) is stretched. The default engine selection serves this grid"))

# The directions whose axis is not uniform.
_stretched(sp::Tuple) = [d for (d, s) in enumerate(sp) if !(s isa FlowGeometries.Axes.UniformSpacing)]

function _padded_fft_footprint(
    sp::NTuple{N,FlowGeometries.Axes.UniformSpacing},
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy = CGEF.Filtering.ZeroFill(),
    execution::CGEF.ComputationalBackends.AbstractExecutionBackend = CGEF.ComputationalBackends.SerialBackend(),
    fft_options::NamedTuple = (;),
) where {T<:AbstractFloat, N, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    dims = FlowGeometries.Grids.size_tuple(grid)
    # A periodic axis reaches every image inside the support; a bounded one no farther than its extent.
    w = CGEF.Filtering._padded_halfwidths(grid, kernel, scale)
    P = ntuple(d -> CGEF.Filtering._padded_length(dims[d], w[d], FlowGeometries.Grids.isperiodic(grid, d)), Val(N))
    nt = CGEF.Filtering._library_threads(execution)

    # The weights wrapped so index (1,…,1) is the zero offset. A planner that times its candidates writes the array
    # it plans on, so the kernel is copied in after.
    gpad = zeros(T, P)
    _sample_kernel!(gpad, sp, grid, kernel, scale, w)
    pad = CGEF.Filtering._allocate(execution, T, P)
    fwd = _plan_rfft(pad, nt, fft_options)
    copyto!(pad, gpad)
    Ĝ = fwd * pad
    spec = similar(Ĝ)
    inv = _plan_irfft(spec, P[1], nt, fft_options)
    mask = _resident_mask(execution, FlowGeometries.Grids.mask(grid))
    den = _padded_denominator(sp, grid, kernel, scale, mask_strategy, fwd, inv, Ĝ, P, mask, execution)
    fill!(pad, zero(T))
    bound = !all(FlowGeometries.Grids.mask(grid)) || CGEF.Filtering._reaches_exterior(grid, kernel, scale)
    return PaddedFFTFootprint(Ĝ, den, pad, CGEF.Filtering._allocate(execution, T, P), spec, fwd, inv, mask,
                              mask_strategy, bound, dims, P)
end

# The offset `o` wrapped onto a transform grid of lengths `P`, the zero offset at index (1,…,1).
@inline _wrapped(o::CartesianIndex{N}, P::NTuple{N,Int}) where {N} = CartesianIndex(ntuple(d -> mod1(1 + o[d], P[d]), Val(N)))

# A radial kernel over the ball `d <= rad` of the direct footprint within half-widths `w`, each weight
# times the cell measure.
function _sample_kernel!(gpad::AbstractArray{T,N}, _sp, grid, kernel, scale::T, w::NTuple{N,Int}) where {T,N}
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

# A `CellOverlap` top-hat: each lattice cell's measure times its fraction inside the support.
function _sample_kernel!(gpad::AbstractArray{T,N}, _sp, grid,
                         kernel::CGEF.Kernels.TopHatKernel{CGEF.Kernels.CellOverlap}, scale::T,
                         w::NTuple{N,Int}) where {T,N}
    Δ = ntuple(d -> CGEF.Filtering._axis_step(grid, d), Val(N))
    ov = CGEF.Filtering._overlap_rule(FlowGeometries.Grids.grid_geometry(grid),
                                      CGEF.Kernels.kernel_radius(kernel, scale, Val(N)))
    A = T(FlowGeometries.Grids.measure(grid, ntuple(_ -> 1, Val(N))...))
    for o in CartesianIndices(map(wd -> (-wd):wd, w))
        f = CGEF.Filtering._lattice_fraction(ov, o, Δ)
        iszero(f) && continue
        gpad[_wrapped(o, size(gpad))] += A * T(f)
    end
    return gpad
end

# A separable kernel: the separable engine's per-axis tables over its box stencil.
function _sample_kernel!(gpad::AbstractArray{T,N}, sp, grid, kernel::CGEF.Filtering.SeparableKernel, scale::T,
                         w::NTuple{N,Int}) where {T,N}
    (; g, lim) = CGEF.Filtering._separable_tables(sp, grid, kernel, scale)
    for o in CartesianIndices(map(wd -> (-wd):wd, w))
        gpad[_wrapped(o, size(gpad))] += prod(ntuple(k -> g[k][o[k] + lim[k] + 1], Val(N)))
    end
    return gpad
end

# `region .= mask` as reals: one on every active cell.
_unit_on!(region, ::FlowGeometries.Grids.AllActive) = fill!(region, one(eltype(region)))
_unit_on!(region, mask) = (region .= mask)

# The kernel mass over the active cells at each cell, through the plan's transform.
function _transform_mass(grid, fwd, inv, Ĝ, P, mask, execution)
    dims = FlowGeometries.Grids.size_tuple(grid)
    T = real(eltype(Ĝ))
    dpad = CGEF.Filtering._allocate(execution, T, P)
    _unit_on!(view(dpad, map(Base.OneTo, dims)...), mask)
    spec = fwd * dpad
    spec .*= Ĝ
    return (inv * spec)[map(Base.OneTo, dims)...]
end

# A radial kernel under `ZeroFill`: the grid's lattice continues uniformly past a bounded edge, so a cell's
# in-domain and exterior mass add to the kernel's mass over the whole ball of lattice offsets, the same at
# every cell. Under `Deformable`, the mass over the active cells.
function _padded_denominator(_sp, grid, kernel, scale, strategy, fwd, inv, Ĝ, P::NTuple{N,Int}, mask, execution) where {N}
    strategy isa CGEF.Filtering.ZeroFill || return _transform_mass(grid, fwd, inv, Ĝ, P, mask, execution)
    T = real(eltype(Ĝ))
    Δ = ntuple(d -> CGEF.Filtering._axis_step(grid, d), Val(N))
    rad = CGEF.Kernels.kernel_radius(kernel, scale, Val(N))
    A = T(FlowGeometries.Grids.measure(grid, ntuple(_ -> 1, Val(N))...))
    mass = zero(T)
    for o in CartesianIndices(map(δ -> (-floor(Int, rad / δ)):floor(Int, rad / δ), Δ))
        d = hypot(ntuple(k -> o[k] * Δ[k], Val(N))...)
        d <= rad && (mass += CGEF.Kernels.kernel_weight(kernel, d, scale, Val(N)) * A)
    end
    return fill!(CGEF.Filtering._allocate(execution, T, FlowGeometries.Grids.size_tuple(grid)), mass)
end

# A `CellOverlap` top-hat under `ZeroFill`: the lattice's cells, wrapped or continued, tile the support, so a
# cell's mass is the support's measure. Under `Deformable`, the mass over the active cells.
function _padded_denominator(_sp, grid, kernel::CGEF.Kernels.TopHatKernel{CGEF.Kernels.CellOverlap}, scale,
                             strategy, fwd, inv, Ĝ, P::NTuple{N,Int}, mask, execution) where {N}
    strategy isa CGEF.Filtering.ZeroFill || return _transform_mass(grid, fwd, inv, Ĝ, P, mask, execution)
    T = real(eltype(Ĝ))
    den = CGEF.Filtering._allocate(execution, T, FlowGeometries.Grids.size_tuple(grid))
    return fill!(den, CGEF.Filtering._support_measure(grid, kernel, scale, 1))
end

# A separable kernel's is the separable engine's: the product of the per-axis profiles, every offset
# counted under `ZeroFill` and the in-domain ones under `Deformable` on an unmasked grid, and the mass
# over the active cells under `Deformable` on a masked one.
function _padded_denominator(sp, grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, kernel::CGEF.Filtering.SeparableKernel,
                             scale, strategy, fwd, inv, Ĝ, P::NTuple{N,Int}, mask, execution) where {T,G,N}
    CGEF.Filtering._mask_free_mass(strategy, mask) || return _transform_mass(grid, fwd, inv, Ĝ, P, mask, execution)
    zerofill = strategy isa CGEF.Filtering.ZeroFill
    (; g, lim, periodic, mf) = CGEF.Filtering._separable_tables(sp, grid, kernel, scale)
    prof = ntuple(Val(N)) do d
        CGEF.Filtering._separable_profile(FlowGeometries.Grids.coordinates(grid, d), lim[d], g[d], periodic[d],
                                          CGEF.Kernels._axis_kernel(kernel, Val(d)), scale,
                                          convert(AbstractVector{T}, mf[d]), zerofill)
    end
    den = [prod(ntuple(d -> prof[d][I[d]], Val(N))) for I in CartesianIndices(FlowGeometries.Grids.size_tuple(grid))]
    return CGEF.Filtering._on_backend(execution, den)
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
#
# Every array lives in the execution backend's memory. The taps are found on the host and each row pair's
# spectrum is transformed where the arrays are.
# ---------------------------------------------------------------------------

struct ZonalFFTFootprint{
    T<:AbstractFloat, A<:AbstractMatrix{T}, R3<:AbstractArray{T,3}, AR,
    C<:AbstractMatrix{Complex{T}}, FP, IP, MK, MS<:CGEF.Filtering.AbstractMaskStrategy,
}
    Ŵ::R3           # real kernel spectra, the first modes of `Px÷2+1` (all, or those `_zonal_modes` keeps):
                    # (modes, dj_lim+1, Ny) per unordered row pair when `area` is a vector, (modes, 2·dj_lim+1, Ny)
                    # per (band offset, target row) when it is `nothing`
    invden::A       # (Nx, Ny) reciprocal window mass, built through this same transform
    dj_lim::Int
    area::AR        # (Ny) cell area per latitude row, or `nothing`
    bound::Bool     # the strategies' denominators differ: a cell is inactive or a window leaves the grid
    mask::MK        # the grid's mask in the arrays' memory, `AllActive` as it is
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
_scale_rows!(F::AbstractMatrix, area::AbstractVector) = (F .*= transpose(area))

# `acc[:, j] = Σ_dj Ŵ_{j, j+dj} F[:, j+dj]` over the rows in the grid, from either layout of `Ŵ`: on host
# arrays through `rows`, elsewhere one broadcast per band offset `dj` over the target rows it reaches. Every target
# adds its offsets in ascending order on both, so they agree bitwise. A spectrum stored per unordered pair is read
# once for both of its rows on the host, by blocks of target rows. `Ŵ` holds the first `size(Ŵ, 1)` modes; `acc`'s
# modes past them are zero.
function _band_sum!(acc::Matrix{Complex{T}}, Ŵ::Array{T,3}, F::Matrix{Complex{T}},
                    dj_lim::Int, ::Nothing, Ny::Int, rows::D) where {T<:AbstractFloat, D}
    nk, nw = size(F, 1), size(Ŵ, 1)
    rows(1:Ny) do j
        @inbounds begin
            @simd for n in 1:nk
                acc[n, j] = zero(Complex{T})
            end
            for dj in (-dj_lim):dj_lim
                jj = j + dj
                (1 <= jj <= Ny) || continue
                @simd for n in 1:nw
                    acc[n, j] += Ŵ[n, dj + dj_lim + 1, j] * F[n, jj]
                end
            end
        end
        return nothing
    end
    return acc
end

function _band_sum!(acc::Matrix{Complex{T}}, Ŵ::Array{T,3}, F::Matrix{Complex{T}},
                    dj_lim::Int, ::AbstractVector, Ny::Int, rows::D) where {T<:AbstractFloat, D}
    nb = clamp(Threads.nthreads(), 1, Ny)
    B = cld(Ny, nb)
    rows(1:cld(Ny, B)) do k
        _pair_band_sum!(acc, Ŵ, F, dj_lim, Ny, ((k - 1) * B + 1):min(k * B, Ny))
        return nothing
    end
    return acc
end

# One block of target rows `a:b`: first the pairs whose lower row precedes the block, lower row ascending; then
# each lower row `r` of the block, the spectrum of offset `dj ≥ 0` feeding target `r` and, inside the block, target
# `r + dj`. A target's offsets arrive in ascending order.
@inline function _pair_band_sum!(acc::Matrix{Complex{T}}, Ŵ::Array{T,3}, F::Matrix{Complex{T}}, dj_lim::Int, Ny::Int,
                                 block::UnitRange{Int}) where {T<:AbstractFloat}
    nk, nw = size(F, 1), size(Ŵ, 1)
    a, b = first(block), last(block)
    @inbounds begin
        for j in block
            @simd for n in 1:nk
                acc[n, j] = zero(Complex{T})
            end
        end
        for r in max(1, a - dj_lim):(a - 1), jj in a:min(b, r + dj_lim)
            dj = jj - r
            @simd for n in 1:nw
                acc[n, jj] += Ŵ[n, dj + 1, r] * F[n, r]
            end
        end
        for r in block, dj in 0:min(dj_lim, Ny - r)
            jj = r + dj
            if dj == 0 || jj > b
                @simd for n in 1:nw
                    acc[n, r] += Ŵ[n, dj + 1, r] * F[n, jj]
                end
            else
                @simd for n in 1:nw
                    w = Ŵ[n, dj + 1, r]
                    acc[n, r] += w * F[n, jj]
                    acc[n, jj] += w * F[n, r]
                end
            end
        end
    end
    return acc
end

function _band_sum!(acc::AbstractMatrix{Complex{T}}, Ŵ::AbstractArray{T,3}, F::AbstractMatrix{Complex{T}},
                    dj_lim::Int, area, Ny::Int, _) where {T<:AbstractFloat}
    fill!(acc, zero(Complex{T}))
    modes = axes(Ŵ, 1)
    for dj in (-dj_lim):dj_lim
        js = max(1, 1 - dj):min(Ny, Ny - dj)
        isempty(js) && continue
        view(acc, modes, js) .+= _band_weights(Ŵ, dj, dj_lim, js, area) .* view(F, modes, js .+ dj)
    end
    return acc
end

# The spectra of offset `dj` for target rows `js`: stored per unordered pair under the lower row when the
# areas are factored out, per target row otherwise.
_band_weights(Ŵ, dj, dj_lim, js, ::Nothing) = view(Ŵ, :, dj + dj_lim + 1, js)
_band_weights(Ŵ, dj, dj_lim, js, ::AbstractVector) = view(Ŵ, :, abs(dj) + 1, dj >= 0 ? js : js .+ dj)

CGEF.Filtering.zonal_fft_footprint(
    sp::Tuple, grid::FlowGeometries.Grids.StructuredGrid{T,G,2}, kernel::CGEF.Kernels.AbstractFilterKernel, scale::T;
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.SphericalGeometry{T}} =
    _zonal_fft_footprint(sp, grid, kernel, scale; kwargs...)

_zonal_fft_footprint(::Tuple, grid, kernel, scale; kwargs...) = throw(ArgumentError(
    "the zonal-FFT engine needs a uniform longitude axis; this grid's longitude axis (direction 1) is " *
    "stretched. The default engine selection serves this grid"))

function _zonal_fft_footprint(
    ::Tuple{FlowGeometries.Axes.UniformSpacing,Any},
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy = CGEF.Filtering.ZeroFill(),
    execution::CGEF.ComputationalBackends.AbstractExecutionBackend = CGEF.ComputationalBackends.SerialBackend(),
    fft_options::NamedTuple = (;),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.SphericalGeometry{T}}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    rows = _zonal_rows(grid, kernel, scale)
    dj_lim = CGEF.Filtering._row_dj_lim(rows)
    area_h = _zonal_area(grid, kernel)
    nb = area_h === nothing ? 2 * dj_lim + 1 : dj_lim + 1
    Px = FlowGeometries.Grids.isperiodic(grid, 1) ? Nx : CGEF.Filtering._even_smooth_length(Nx + CGEF.Filtering._row_reach(rows))
    nt = CGEF.Filtering._library_threads(execution)
    zs(S, dims...) = CGEF.Filtering._allocate(execution, S, dims)

    # A planner that times its candidates writes the arrays it plans on: `src`'s padding is zeroed after.
    src = zs(T, Px, Ny)
    fwd = _plan_rfft(src, nt, fft_options, 1)
    Fbuf = fwd * src
    iplan = _plan_irfft(Fbuf, Px, nt, fft_options, 1)
    fill!(src, zero(T))
    nk = size(Fbuf, 1)

    # Each target row's taps, gathered on the host one row at a time and transformed where the spectra live, the rows
    # divided among a threaded plan's threads. A kernel whose spectrum along longitude falls off keeps the modes above
    # `truncation_tol` of its row's mass (`_zonal_modes`).
    gplan = _plan_rfft(zs(T, Px, nb), 1, (;), 1)
    tol = CGEF.Kernels.truncation_tol(T)
    cut = _zonal_truncates(kernel)
    spectra = Vector{Matrix{T}}(undef, cut ? Ny : 0)
    full = zeros(T, cut ? 0 : nk, nb, cut ? 0 : Ny)
    mass = CGEF.Filtering._has_row_mass(rows) ? Vector{T}(undef, Ny) : nothing
    CGEF.Filtering._share_run(CGEF.Filtering._build_share(execution), 1:Ny) do js
        r = CGEF.Filtering._row_worker(rows)
        gcol_h = zeros(T, Px, nb)
        gcol = zs(T, Px, nb)
        Ĝ = gplan * gcol
        s = zeros(T, cut ? nk : 0, nb)
        for j in js
            fill!(gcol_h, zero(T))
            m = CGEF.Filtering._row_taps!(r, j) do ddi, ddj, wt
                if area_h === nothing
                    gcol_h[mod(ddi, Px) + 1, ddj + dj_lim + 1] += wt
                elseif ddj >= 0
                    gcol_h[mod(ddi, Px) + 1, ddj + 1] += wt / area_h[j + ddj]
                end
            end
            mass === nothing || (mass[j] = m)
            copyto!(gcol, gcol_h)
            LA.mul!(Ĝ, gplan, gcol)
            if cut
                _real_part!(s, Ĝ)
                spectra[j] = s[1:_zonal_modes(s, tol), :]
            else
                _real_part!(view(full, :, :, j), Ĝ)
            end
        end
    end
    Ŵ = CGEF.Filtering._on_backend(execution, cut ? _stack_modes(spectra, nb) : full)

    mask = _resident_mask(execution, FlowGeometries.Grids.mask(grid))
    area = area_h === nothing ? nothing : CGEF.Filtering._on_backend(execution, area_h)
    bound = !all(FlowGeometries.Grids.mask(grid)) || CGEF.Filtering._reaches_exterior(grid, kernel, scale)
    fp = ZonalFFTFootprint(
        Ŵ, zs(T, Nx, Ny), dj_lim, area, bound, mask, mask_strategy,
        src, Px == Nx ? zs(T, 0, 0) : zs(T, Px, Ny), Fbuf, similar(Fbuf), fwd, iplan,
    )

    # Window mass through the same operator: `ZeroFill` counts every in-support cell and the rows past a
    # bounded latitude, as far as the poles; `Deformable` only the active in-domain cells, and gives an
    # inactive target zero.
    zerofill = mask_strategy isa CGEF.Filtering.ZeroFill
    unit = view(src, 1:Nx, :)
    zerofill ? fill!(unit, one(T)) : _unit_on!(unit, mask)
    den = zs(T, Nx, Ny)
    if zerofill
        _zonal_zerofill_mass!(den, src, fp, grid, kernel, scale, mass, execution)
    else
        _zonal_accumulate!(den, src, fp, Ny)
    end
    fill!(unit, zero(T))
    _masked_inverse!(fp.invden, den, mask_strategy, mask)
    return fp
end

# `dst .= real.(Ĝ)` into host memory, through one copy from a device's.
_real_part!(dst::AbstractArray, Ĝ::Array) = (dst .= real.(Ĝ))
_real_part!(dst::AbstractArray, Ĝ::AbstractArray) = copyto!(dst, Array(real.(Ĝ)))

# The taps of the target rows, one row at a time (`Filtering._row_taps!`): the banded engine's for a radial
# kernel, a `HighOrderKernel`'s corrected rows, a `CellOverlap` top-hat's overlap weights.
_zonal_rows(grid, kernel, scale) =
    CGEF.Filtering.SphereBandRows(grid, kernel, scale, CGEF.Filtering._query_radius(grid, kernel, scale))
_zonal_rows(grid, kernel::CGEF.Kernels.HighOrderKernel, scale) =
    _rows_or_raise(CGEF.Filtering._highorder_band_rows(grid, kernel, scale))
_zonal_rows(grid, kernel::CGEF.Kernels.TopHatKernel{CGEF.Kernels.CellOverlap}, scale) =
    CGEF.Filtering.ListedBandRows(CGEF.Filtering._overlap_band_rows(grid, kernel, scale)...)

# Whether a kernel's row spectra fall off fast enough to drop the modes below `truncation_tol`: the Gaussian, whose
# spectrum falls as a Gaussian.
_zonal_truncates(::CGEF.Kernels.AbstractFilterKernel) = false
_zonal_truncates(::CGEF.Kernels.GaussianKernel) = true

# The modes of one row's spectra `s` (mode by row pair) to keep: through the last whose weights, summed over the pairs,
# exceed `tol` times the row's mass, mode 0 summed.
function _zonal_modes(s::AbstractMatrix{T}, tol::T) where {T}
    m0 = sum(abs, view(s, 1, :))
    n = findlast(i -> sum(abs, view(s, i, :)) > tol * m0, axes(s, 1))
    return n === nothing ? 1 : n
end

# The rows' kept spectra as one table of the most modes any row keeps, zero past a row's own.
function _stack_modes(spectra::Vector{Matrix{T}}, nb::Int) where {T}
    W = zeros(T, maximum(s -> size(s, 1), spectra), nb, length(spectra))
    for j in eachindex(spectra)
        view(W, axes(spectra[j], 1), :, j) .= spectra[j]
    end
    return W
end

_rows_or_raise(rows) = rows
@noinline _rows_or_raise(::Nothing) = throw(ArgumentError(
    "the zonal-FFT engine needs one HighOrderKernel correction per latitude row, and this grid's rows do not share " *
    "one: their supports reach the seam where the continued longitude closes. The default engine selection serves " *
    "this grid"))

# The source rows' cell areas where a weight factors as `G(d)·A_jj`, symmetric in the two rows; `nothing` for
# weights kept per ordered pair of rows.
_zonal_area(grid::FlowGeometries.Grids.StructuredGrid{T}, ::CGEF.Kernels.AbstractFilterKernel) where {T} =
    T[FlowGeometries.Grids.area(grid, 1, j) for j in 1:FlowGeometries.Grids.size_tuple(grid)[2]]
_zonal_area(::FlowGeometries.Grids.StructuredGrid, ::CGEF.Kernels.HighOrderKernel) = nothing
_zonal_area(::FlowGeometries.Grids.StructuredGrid, ::CGEF.Kernels.TopHatKernel{CGEF.Kernels.CellOverlap}) = nothing

# `ZeroFill`'s window mass: every in-support cell through the transform, and the mass past a bounded latitude
# as far as the poles.
function _zonal_zerofill_mass!(den, src, fp::ZonalFFTFootprint{T}, grid, kernel, scale, _, execution) where {T}
    _zonal_accumulate!(den, src, fp, size(den, 2))
    exterior = CGEF.Filtering._exterior_mass(grid, kernel, scale, execution)
    exterior === nothing || (den .+= CGEF.Filtering._on_backend(execution, exterior))
    return den
end

# Each row's whole support mass, the continued lattice included.
_zonal_zerofill_mass!(den, _, ::ZonalFFTFootprint{T}, _, ::CGEF.Kernels.HighOrderKernel, _, mass, execution) where {T} =
    den .= reshape(CGEF.Filtering._on_backend(execution, mass), 1, size(den, 2))

# The cap's measure where a window leaves the grid, which the overlap weights past the edge complete.
function _zonal_zerofill_mass!(den, src, fp::ZonalFFTFootprint{T}, grid,
                               kernel::CGEF.Kernels.TopHatKernel{CGEF.Kernels.CellOverlap}, scale, _, execution) where {T}
    Ny = size(den, 2)
    CGEF.Filtering._full_support(grid, kernel, scale) || return _zonal_accumulate!(den, src, fp, Ny)
    m = T[CGEF.Filtering._support_measure(grid, kernel, scale, j) for j in 1:Ny]
    return den .= reshape(CGEF.Filtering._on_backend(execution, m), 1, Ny)
end

# `1/den` where a target is filtered: everywhere under `ZeroFill` or without a mask, on the active cells
# otherwise, and zero elsewhere.
_masked_inverse!(inv, den, strategy, mask) = CGEF.Filtering._mask_free_mass(strategy, mask) ?
    (inv .= CGEF.Filtering._inv_mass.(den)) :
    (inv .= ifelse.(mask, CGEF.Filtering._inv_mass.(den), zero(eltype(inv))))

CGEF.Filtering._driven_footprint(::Union{ZonalFFTFootprint, PaddedFFTFootprint}) = true

# Each footprint holds its window mass; a separable kernel's padded footprint weighs by the separable tables.
CGEF.Filtering._window_mass(fp::PaddedFFTFootprint, _) = fp.den
CGEF.Filtering._window_mass(fp::ZonalFFTFootprint, _) = CGEF.Filtering._inv_mass.(fp.invden)
CGEF.Filtering._engine_of(::PaddedFFTFootprint) = CGEF.Filtering.PaddedFFTEngine()
CGEF.Filtering._engine_of(::ZonalFFTFootprint) = CGEF.Filtering.ZonalFFTEngine()
CGEF.Filtering._weight_family(::PaddedFFTFootprint, ::CGEF.Filtering.SeparableKernel) = CGEF.Filtering._AxisRates()

# The buffers an apply writes; the spectra, masses and transforms are shared by every worker. A transform
# executes on any arrays of the shapes it was planned on.
CGEF.Filtering._scratch(fp::ZonalFFTFootprint) = CGEF.Filtering.ApplyBuffers((fp.src, fp.wide, fp.F, fp.acc))
CGEF.Filtering._with_scratch(fp::ZonalFFTFootprint, s::CGEF.Filtering.ApplyBuffers) =
    ZonalFFTFootprint(fp.Ŵ, fp.invden, fp.dj_lim, fp.area, fp.bound, fp.mask, fp.strategy, s.buffers..., fp.fwd, fp.inv)
CGEF.Filtering._scratch(fp::PaddedFFTFootprint) = CGEF.Filtering.ApplyBuffers((fp.pad, fp.num, fp.spec))
CGEF.Filtering._with_scratch(fp::PaddedFFTFootprint, s::CGEF.Filtering.ApplyBuffers) =
    PaddedFFTFootprint(fp.Ĝ, fp.den, s.buffers..., fp.fwd, fp.inv, fp.mask, fp.strategy, fp.bound, fp.dims, fp.P)

function CGEF.Filtering._check_strategy(fp::ZonalFFTFootprint, strategy::CGEF.Filtering.AbstractMaskStrategy)
    (!fp.bound || typeof(strategy) === typeof(fp.strategy)) || throw(ArgumentError(
        "this zonal-FFT footprint was built for $(nameof(typeof(fp.strategy))) on a grid where the " *
        "strategies normalize differently (a masked cell, or a window past a bounded latitude); its " *
        "normalization is not the one $(nameof(typeof(strategy))) needs. Rebuild the plan with the " *
        "strategy you intend to apply with.",
    ))
    return nothing
end

function CGEF.Filtering._check_strategy(fp::PaddedFFTFootprint, strategy::CGEF.Filtering.AbstractMaskStrategy)
    (!fp.bound || typeof(strategy) === typeof(fp.strategy)) || throw(ArgumentError(
        "this padded-FFT footprint was built for $(nameof(typeof(fp.strategy))) on a grid where the " *
        "strategies normalize differently (a masked cell, or a window past a bounded edge); its " *
        "denominator is not the one $(nameof(typeof(strategy))) needs. Rebuild the plan with the " *
        "strategy you intend to apply with.",
    ))
    return nothing
end

# `mask · field`, in the footprint's memory.
_stage!(dst, field, ::FlowGeometries.Grids.AllActive) = (dst .= field)
_stage!(dst, field, mask) = (dst .= ifelse.(mask, field, zero(eltype(dst))))

# `rows(f, 1:Ny)` visits every latitude row; each row writes only its own column of every buffer.
function CGEF.Filtering.apply_footprint!(
    out::AbstractMatrix{T}, field::AbstractMatrix,
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    fp::ZonalFFTFootprint{T}, strategy::CGEF.Filtering.AbstractMaskStrategy,
    rows::D = CGEF.Filtering._sep_serial,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.SphericalGeometry{T}, D}
    CGEF.Filtering._check_strategy(fp, strategy)
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

function CGEF.Filtering._resident_apply!(
    out::AbstractMatrix{T}, field::AbstractMatrix, _, fp::ZonalFFTFootprint{T}, ::CGEF.Filtering.AbstractMaskStrategy,
) where {T<:AbstractFloat}
    Nx, Ny = size(out)
    _stage!(view(fp.src, 1:Nx, :), field, fp.mask)
    _zonal_accumulate!(out, fp.src, fp, Ny)
    out .*= fp.invden
    return out
end

# `rows(f, 1:n)` visits every index along the last direction; each writes only its own slab. The padding
# past the grid's cells is zero from construction, and the forward transform leaves its input as it is.
function CGEF.Filtering.apply_footprint!(
    out::AbstractArray{T,N}, field::AbstractArray,
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N},
    fp::PaddedFFTFootprint{T}, strategy::CGEF.Filtering.AbstractMaskStrategy,
    rows::D = CGEF.Filtering._sep_serial,
) where {T<:AbstractFloat, N, G<:FlowGeometries.Geometry.CartesianGeometry{T}, D}
    CGEF.Filtering._check_strategy(fp, strategy)
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

function CGEF.Filtering._resident_apply!(
    out::AbstractArray{T}, field::AbstractArray, _, fp::PaddedFFTFootprint{T}, strategy::CGEF.Filtering.AbstractMaskStrategy,
) where {T<:AbstractFloat}
    region = map(Base.OneTo, fp.dims)
    _stage!(view(fp.pad, region...), field, fp.mask)
    LA.mul!(fp.spec, fp.fwd, fp.pad)
    fp.spec .*= fp.Ĝ
    LA.mul!(fp.num, fp.inv, fp.spec)
    num = view(fp.num, region...)
    if CGEF.Filtering._mask_free_mass(strategy, fp.mask)
        out .= CGEF.Filtering._normalized.(num, fp.den)
    else
        out .= ifelse.(fp.mask, CGEF.Filtering._normalized.(num, fp.den), zero(T))
    end
    return out
end

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
`FlowGeometries.Axes.PeriodicUniformAxis`: the in-place transforms of the `(nlon, nlat)` scratch they are
planned on, the phases `e^{−imφ₁}` and each order's bin.
"""
struct FFTLongitudes{P,Q,V,I,B}
    fwd::P
    bwd::Q
    phase::V
    bin::I
    execution::B
end

function CGEF.Filtering._ring_longitudes(
    λ::FlowGeometries.Axes.PeriodicUniformAxis, φ::AbstractVector{T}, L::Int, wf, execution,
) where {T<:AbstractFloat}
    CGEF.Filtering._fft_provided(execution) || return CGEF.Filtering._dense_longitudes(φ, L, execution)
    n = length(φ)
    σ = FlowGeometries.Axes.wrap_sign(λ) > 0 ? 1 : -1
    nt = CGEF.Filtering._library_threads(execution)
    put(a) = FlowGeometries.Execution.on_backend(execution, a)
    return FFTLongitudes(_plan_fft!(wf, nt, (;), 1), _plan_bfft!(wf, nt, (;), 1), put([cis(-T(m) * φ[1]) for m in -L:L]),
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
