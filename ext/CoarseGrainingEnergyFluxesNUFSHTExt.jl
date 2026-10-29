module CoarseGrainingEnergyFluxesNUFSHTExt

using NUFSHT: NUFSHT
using FlowTransformBindings: FlowTransformBindings as FTB
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF
using FlowGeometries: FlowGeometries

# Spectral filtering for scattered spherical data through NUFSHT.jl: take the field's harmonic coefficients by the
# plan's analysis, multiply each degree by the kernel's transfer function, synthesize at the points
# (`nusht_synthesize!`). The filter is `A H B`, with `A` the synthesis at the points, `H` the per-degree transfer and
# `B` the quadrature `Aᴴ W` over the points' measures (`Filtering.Quadrature`) or the least-squares fit
# (`Filtering.LeastSquares`, `nusht_solve!`).

# Adapter exposing CGEF's transfer function to NUFSHT's per-degree `kernel_transfer`: `Ĝ` holds
# `Kernels.spectral_transfer_degrees!` at every degree a plan's coefficient array carries, `0:2lmax`.
struct _CGEFTransfer{K<:CGEF.Kernels.AbstractFilterKernel, T<:AbstractFloat, V<:AbstractVector{T}} <:
       NUFSHT.AbstractSpectralTransfer
    kernel::K
    scale::T
    R::T
    Ĝ::V
end

_CGEFTransfer(kernel, scale::T, R::T, lmax::Int) where {T<:AbstractFloat} = _CGEFTransfer(
    kernel, scale, R, CGEF.Kernels.spectral_transfer_degrees!(zeros(T, 2lmax + 1), kernel, scale, R),
)

@inline NUFSHT.kernel_transfer(t::_CGEFTransfer, l) = @inbounds t.Ĝ[Int(l) + 1]

"""
    NUFSHTGridPlan

The half of a scattered-spherical plan the filter scale does not reach: the NUSHT plan over the grid's
nodes this process transforms (under `MPIBackend` its share of them, under `DistributedBackend` a plan
NUFSHT divides among the worker processes), the sphere radius, the mask over those nodes, the mask's
fitted coefficients, the execution backend whose memory holds them, and the nodes owned (`nothing` for
all). A plan built for a trailing batch axis of `nb` fields also holds a
`NUSHTplan` with `ntrans = nb` over the same nodes, whose fit and synthesis carry every field of the
batch at once.

The band limit is `lmax`, by default `Filtering.spin_band_limit(grid)`, the degree the grid's largest cell
resolves, as the spin transforms of the generalized convolution take it. `analysis` is the resolved
`Filtering.Quadrature()` or `Filtering.LeastSquares()`. Both weigh each node by its measure over `R²`: the
quadrature sums `Σⱼ wⱼ fⱼ conj(Y(xⱼ))`, and the fit minimizes `Σⱼ wⱼ |(A c − f)ⱼ|²`, the discrete form of the
squared error integrated over the sphere; each plan's weighted operator is built once (under
`DistributedBackend` NUFSHT weighs each worker's block per fit).

A `NUSHTplan` holds buffers every transform overwrites, so two tasks may not execute one grid plan
concurrently; a concurrent driver needs its own grid plan per worker. The colatitudes, longitudes and NUFFT
library `plan` was built on serve the spin-1 plan of a series gradient (`Filtering.series_gradient`).
"""
struct NUFSHTGridPlan{P, T<:AbstractFloat, M, CM, BP, OP, BOP, W, B<:CGEF.ComputationalBackends.AbstractExecutionBackend, O, ND, NF,
                      AN<:Union{CGEF.Filtering.Quadrature,CGEF.Filtering.LeastSquares}} <:
       CGEF.Filtering.AbstractGridPlan
    plan::P
    radius::T
    mask::M          # the grid's mask over this process's nodes, or nothing on a grid whose mask is `AllActive`
    C_mask::CM       # the mask's coefficients, or nothing when unmasked
    # The outcome of the analysis of the mask's 0/1 values, which `Deformable`'s `filter(mask)` is synthesized
    # from; exact on an unmasked grid and under the quadrature.
    mask_fit::CGEF.Filtering.AnalysisStatus{T}
    npts::Int        # the grid's node count
    batched::BP      # a `NUSHTplan` with `ntrans = nb`, or nothing
    op::OP           # `plan` weighted by `weights`, or nothing under `DistributedBackend`
    batched_op::BOP  # `batched` weighted by `weights`, or nothing
    weights::W       # this process's nodes' measures over R²
    execution::B
    own::O           # this process's nodes, or nothing for all of them
    nodes::ND        # (θ, φ) as `plan` was built on them
    nufft::NF
    analysis::AN
end

"""
    NUFSHTScratch

The transient half of a scattered-spherical plan: the staging array for `mask .* field`, the fitted
coefficients, the least-squares workspace, the synthesis of this process's nodes where it holds only some
of them, and the last fit's outcome; and the same set for the batched plan, so an apply allocates nothing.
One per concurrent worker.
"""
struct NUFSHTScratch{T<:AbstractFloat, SV<:AbstractVector{T}, CA<:AbstractArray, W, LO, BT} <:
       CGEF.Filtering.AbstractFilterScratch
    masked_input::SV   # unused when mask === nothing
    coeffs::CA
    ws::W              # FTB.LSMRWorkspace, or nothing where NUFSHT's workers hold theirs
    local_out::LO      # this process's synthesized values, or nothing when it holds every node
    batched::BT        # (; masked_input, coeffs, ws, local_out) for the batched plan, or nothing
    status::Base.RefValue{CGEF.Filtering.AnalysisStatus{T}}
end

"""
    NUFSHTFilterPlan

Cached scattered-spherical filter plan: the shared [`NUFSHTGridPlan`](@ref), the CGEF transfer adapter,
the `Deformable` reciprocal of the filtered mask (or `nothing`), and the [`NUFSHTScratch`](@ref) an apply
works in. Built by `plan_filter(scattered_spherical_grid, kernel, scale; method = Spectral())`.
"""
struct NUFSHTFilterPlan{
    T<:AbstractFloat, GP<:NUFSHTGridPlan, F, R, MS<:CGEF.Filtering.AbstractMaskStrategy, SC<:NUFSHTScratch{T},
} <: CGEF.Filtering.AbstractFilterPlan
    grid_plan::GP
    filter::F
    invrenorm::R     # 1/filter(mask) under `Deformable` on a masked grid, zero at a masked point; or nothing
    strategy::MS
    scratch::SC
end

CGEF.Filtering.plan_strategy(plan::NUFSHTFilterPlan) = plan.strategy
CGEF.Filtering.plan_execution(plan::NUFSHTFilterPlan) = plan.grid_plan.execution
CGEF.Filtering.analysis_status(plan::NUFSHTFilterPlan) =
    CGEF.Filtering._combine_status(plan.grid_plan.mask_fit, plan.scratch.status[])

# The fit cannot resolve a relative residual finer than the transform it is built on.
_fit_rtol(tol, ::Type{T}) where {T<:AbstractFloat} = max(T(10 * tol), sqrt(eps(T)))

# The weighted fit through `op`, `plan` with the weights `w` on its range. Under `MPIBackend` each rank
# holds a plan over its own nodes, and the fit is one least-squares problem over every rank's nodes, its
# coefficients replicated on every rank. A plan divided among `DistributedBackend` workers carries its
# own solve and weighs each worker's block.
_solve!(C, f, plan, op, w, ws, rtol, maxiter, ::CGEF.ComputationalBackends.AbstractExecutionBackend) =
    NUFSHT.nusht_solve!(C, f, op; ws = ws, rtol = rtol, maxiter = maxiter)
_solve!(C, f, plan, op, w, ws, rtol, maxiter, b::CGEF.ComputationalBackends.AbstractMPIBackend) =
    NUFSHT.nusht_solve!(C, f, op, b; ws = ws, rtol = rtol, maxiter = maxiter)
_solve!(C, f, plan, op, w, ws, rtol, maxiter, ::CGEF.ComputationalBackends.AbstractDistributedBackend) =
    NUFSHT.nusht_solve!(C, f, plan; weights = w, rtol = rtol, maxiter = maxiter)

# The quadrature over the points of every process `backend` names.
_quadrature!(C, f, plan, w, ::CGEF.ComputationalBackends.AbstractExecutionBackend) =
    NUFSHT.nusht_analyze!(C, f, plan; weights = w)
_quadrature!(C, f, plan, w, b::CGEF.ComputationalBackends.AbstractMPIBackend) =
    NUFSHT.nusht_analyze!(C, f, plan, b; weights = w)

# The coefficients of `f` into `C` by `analysis`, and the outcome; a fit that stops short reports with the
# coefficients it reached.
function _analysis!(C, f, plan, op, w, ws, backend, ::CGEF.Filtering.Quadrature)
    _quadrature!(C, f, plan, w, backend)
    return C, CGEF.Filtering._exact_status(real(eltype(C)))
end

function _analysis!(C, f, plan, op, w, ws, backend, a::CGEF.Filtering.LeastSquares)
    T = real(eltype(C))
    rtol = a.rtol === nothing ? _fit_rtol(plan.tol, T) : T(a.rtol)
    _, iters, rel, converged = _solve!(C, f, plan, op, w, ws, rtol, a.maxiter, backend)
    return C, CGEF.Filtering.AnalysisStatus{T}((converged, iters, T(rel)))
end

# This process's plan at `_library_threads(backend)` threads, or under `DistributedBackend` one whose
# points NUFSHT divides among the workers, each at the inner backend's count.
_make_plan(T, θ, φ, lmax, backend::CGEF.ComputationalBackends.AbstractDistributedBackend; kwargs...) =
    NUFSHT.make_plan(T, θ, φ, lmax; execution = backend, kwargs...)
_make_plan(T, θ, φ, lmax, backend::CGEF.ComputationalBackends.AbstractExecutionBackend; kwargs...) =
    NUFSHT.make_plan(T, θ, φ, lmax; nthreads = CGEF.Filtering._library_threads(backend), kwargs...)

# The fit's workspace for a plan `_make_plan` built; the workers of a divided plan hold their own, and the
# quadrature takes none.
_workspace(plan, ::CGEF.ComputationalBackends.AbstractDistributedBackend, _) = nothing
_workspace(plan, ::CGEF.ComputationalBackends.AbstractExecutionBackend, ::CGEF.Filtering.LeastSquares) = FTB.LSMRWorkspace(plan)
_workspace(plan, ::CGEF.ComputationalBackends.AbstractExecutionBackend, ::CGEF.Filtering.Quadrature) = nothing

# `plan` with the weights `w` on its range, built once; a divided plan's workers weigh their blocks.
_weighted_op(plan, w, ::CGEF.ComputationalBackends.AbstractDistributedBackend) = nothing
_weighted_op(plan, w, ::CGEF.ComputationalBackends.AbstractExecutionBackend) = FTB.RangeWeighted(plan, w)

# `nufft` is the NUFFT library NUFSHT runs: a FlowTransformBindings tag, or `AutoSpectralBackend()` for
# NUFSHT's own choice.
function _nufsht_grid_plan(
    grid::FlowGeometries.Grids.UnstructuredGrid{T,G,2};
    execution::CGEF.ComputationalBackends.AbstractExecutionBackend = CGEF.ComputationalBackends.AutoBackend(),
    batch::Union{Nothing,Integer} = nothing,
    nufft::CGEF.SpectralBackends.AbstractSpectralBackend = CGEF.SpectralBackends.AutoSpectralBackend(),
    lmax::Union{Nothing,Integer} = nothing,
    analysis::CGEF.Filtering.AbstractAnalysis = CGEF.Filtering.AutoAnalysis(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.SphericalGeometry{T}}
    an = CGEF.Filtering.resolved_analysis(analysis, grid)
    npts = length(FlowGeometries.Grids.coordinates(grid, 1))
    npts > 0 || throw(ArgumentError("NUFSHT spectral filtering needs at least one point."))
    L = lmax === nothing ? CGEF.Filtering.spin_band_limit(grid) : Int(lmax)
    R = T(FlowGeometries.Geometry.radius(FlowGeometries.Grids.grid_geometry(grid)))
    # This process's nodes go where `execution` runs, and NUFSHT plans in the memory they are in.
    own = CGEF.Filtering._owned(execution, npts)
    part(v) = own === nothing ? v : v[own]
    θ = CGEF.Filtering._on_backend(execution, T(π) / 2 .- T.(part(FlowGeometries.Grids.coordinates(grid, 2))))  # colatitude
    φ = CGEF.Filtering._on_backend(execution, T.(part(FlowGeometries.Grids.coordinates(grid, 1))))
    area = FlowGeometries.Grids.measure(grid)
    w = CGEF.Filtering._on_backend(execution, part([T(area[k]) / R^2 for k in 1:npts]))
    nplan = _make_plan(T, θ, φ, L, execution; nufft = nufft)
    bplan = batch === nothing ? nothing : _make_plan(T, θ, φ, L, execution; ntrans = Int(batch), nufft = nufft)
    op = _weighted_op(nplan, w, execution)
    bop = bplan === nothing ? nothing : _weighted_op(bplan, w, execution)
    mask = CGEF.Filtering._plan_mask(FlowGeometries.Grids.mask(grid), part, execution)
    C_mask, mask_fit = _mask_analysis(mask, nplan, op, w, execution, an)
    return NUFSHTGridPlan(
        nplan, R, mask, C_mask, mask_fit, npts, bplan, op, bop, w, execution,
        own === nothing ? nothing : CGEF.Filtering._on_backend(execution, own), (θ, φ), nufft, an,
    )
end

# The mask's coefficients and their analysis's outcome; none on an unmasked grid.
_mask_analysis(::Nothing, nplan, op, w, execution, _) = (nothing, CGEF.Filtering._exact_status(eltype(w)))
_mask_analysis(mask, nplan, op, w, execution, an) =
    _analysis!(NUFSHT.allocate_coefficients(nplan), eltype(w).(mask), nplan, op, w, _workspace(nplan, execution, an),
               execution, an)

function _nufsht_scratch(gp::NUFSHTGridPlan{P,T}) where {P, T<:AbstractFloat}
    bp = gp.batched
    n = gp.own === nothing ? gp.npts : length(gp.own)
    zs(dims...) = CGEF.Filtering._allocate(gp.execution, T, dims)
    return NUFSHTScratch(
        zs(n), NUFSHT.allocate_coefficients(gp.plan), _workspace(gp.plan, gp.execution, gp.analysis),
        gp.own === nothing ? nothing : zs(n),
        bp === nothing ? nothing :
            (masked_input = zs(n, bp.B), coeffs = NUFSHT.allocate_coefficients(bp),
             ws = _workspace(bp, gp.execution, gp.analysis), local_out = gp.own === nothing ? nothing : zs(n, bp.B)),
        Base.RefValue(CGEF.Filtering._exact_status(T)),
    )
end

# The synthesis of this process's nodes into `out`, which every process then holds whole.
function _synthesize_into!(out, _, Ĉ, plan::NUFSHTFilterPlan, nplan, ::Nothing)
    NUFSHT.nusht_synthesize!(out, Ĉ, plan.filter, nplan)
    plan.invrenorm === nothing || (out .*= plan.invrenorm)
    return out
end
function _synthesize_into!(out, local_out, Ĉ, plan::NUFSHTFilterPlan, nplan, own)
    _synthesize_into!(local_out, nothing, Ĉ, plan, nplan, nothing)
    fill!(out, zero(eltype(out)))
    CGEF.Filtering._local(out, own) .= local_out
    return CGEF.Filtering._sum_across!(plan.grid_plan.execution, out)
end

# The sphere has no bounded direction, so `max_scale` pads nothing.
CGEF.Filtering.spectral_grid_plan(
    ::Union{CGEF.SpectralBackends.AbstractAutoSpectralBackend, CGEF.SpectralBackends.AbstractNUFSHTSpectralBackend},
    grid::FlowGeometries.Grids.UnstructuredGrid{T,G,2},
    kernel::CGEF.Kernels.AbstractFilterKernel;
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy = CGEF.Filtering.ZeroFill(),
    execution::CGEF.ComputationalBackends.AbstractExecutionBackend = CGEF.ComputationalBackends.AutoBackend(),
    batch::Union{Nothing,Integer} = nothing,
    nufft::CGEF.SpectralBackends.AbstractSpectralBackend = CGEF.SpectralBackends.AutoSpectralBackend(),
    lmax::Union{Nothing,Integer} = nothing,
    max_scale::Real = T(0),
    analysis::CGEF.Filtering.AbstractAnalysis = CGEF.Filtering.AutoAnalysis(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.SphericalGeometry{T}} =
    _nufsht_grid_plan(grid; execution = execution, batch = batch, nufft = nufft, lmax = lmax, analysis = analysis)

CGEF.Filtering.spectral_scratch(gp::NUFSHTGridPlan) = _nufsht_scratch(gp)

function CGEF.Filtering.spectral_filter_plan(
    ::Union{CGEF.SpectralBackends.AbstractAutoSpectralBackend, CGEF.SpectralBackends.AbstractNUFSHTSpectralBackend},
    grid::FlowGeometries.Grids.UnstructuredGrid{T,G,2},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy = CGEF.Filtering.ZeroFill(),
    execution::CGEF.ComputationalBackends.AbstractExecutionBackend = CGEF.ComputationalBackends.AutoBackend(),
    # Extent of the trailing batch axis this plan will be applied over, or `nothing` for single fields.
    batch::Union{Nothing,Integer} = nothing,
    nufft::CGEF.SpectralBackends.AbstractSpectralBackend = CGEF.SpectralBackends.AutoSpectralBackend(),
    lmax::Union{Nothing,Integer} = nothing,
    grid_plan::Union{Nothing,NUFSHTGridPlan} = nothing,
    scratch::Union{Nothing,NUFSHTScratch} = nothing,
    analysis::CGEF.Filtering.AbstractAnalysis = CGEF.Filtering.AutoAnalysis(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.SphericalGeometry{T}}
    gp = grid_plan === nothing ?
        _nufsht_grid_plan(grid; execution = execution, batch = batch, nufft = nufft, lmax = lmax, analysis = analysis) :
        grid_plan
    sc = scratch === nothing ? _nufsht_scratch(gp) : scratch
    filter = _CGEFTransfer(kernel, scale, gp.radius, gp.plan.lmax)
    # `Deformable` divides by the local mass, the mask's coefficients synthesized through this scale's transfer;
    # `ZeroFill` is `filter(mask .* field)` as it stands.
    invrenorm = if gp.mask !== nothing && mask_strategy isa CGEF.Filtering.Deformable
        mf = CGEF.Filtering._allocate(gp.execution, T, (gp.own === nothing ? gp.npts : length(gp.own),))
        NUFSHT.nusht_synthesize!(mf, gp.C_mask, filter, gp.plan)
        @. ifelse(gp.mask, CGEF.Filtering._inv_mass(mf), zero(T))
    else
        nothing
    end
    return NUFSHTFilterPlan(gp, filter, invrenorm, mask_strategy, sc)
end

# A `Bool` mask is a strong zero: a masked point is fitted as 0 even where the field is NaN.
@inline _masked(buf, field, ::Nothing) = field
@inline _masked(buf, field, mask) = (buf .= mask .* field)

# The fit depends on the field alone, so a sweep shares one analysis across its scales;
# `nusht_synthesize!` leaves `C` intact.
CGEF.Filtering.analyze_buffer(plan::NUFSHTFilterPlan, ::AbstractVector) =
    NUFSHT.allocate_coefficients(plan.grid_plan.plan)

function CGEF.Filtering.filter_analyze!(
    Ĉ::AbstractArray, field::AbstractVector, plan::NUFSHTFilterPlan{T},
) where {T<:AbstractFloat}
    gp, sc = plan.grid_plan, plan.scratch
    f = CGEF.Filtering._local(field, gp.own)
    _, sc.status[] = _analysis!(Ĉ, _masked(sc.masked_input, f, gp.mask), gp.plan, gp.op, gp.weights, sc.ws, gp.execution,
                                gp.analysis)
    return Ĉ
end

function CGEF.Filtering.filter_synthesize!(
    out::AbstractVector{T}, Ĉ::AbstractArray, plan::NUFSHTFilterPlan{T},
) where {T<:AbstractFloat}
    gp = plan.grid_plan
    return _synthesize_into!(out, plan.scratch.local_out, Ĉ, plan, gp.plan, gp.own)
end

function CGEF.Filtering.filter_apply!(
    out::AbstractVector{T},
    field::AbstractVector,
    plan::NUFSHTFilterPlan{T},
) where {T<:AbstractFloat}
    C = plan.scratch.coeffs
    CGEF.Filtering.filter_analyze!(C, field, plan)
    return CGEF.Filtering.filter_synthesize!(out, C, plan)
end

# ---------------------------------------------------------------------------
# A trailing batch axis: `nb` fields on the same points, fitted and synthesized together
# ---------------------------------------------------------------------------

CGEF.Filtering._batched_fields(outs, plan::NUFSHTFilterPlan) =
    plan.grid_plan.batched !== nothing && ndims(first(outs)) == 2

function _batch_parts(plan::NUFSHTFilterPlan, field::AbstractMatrix)
    gp, sc = plan.grid_plan, plan.scratch
    bp = gp.batched
    (bp === nothing || sc.batched === nothing) && throw(ArgumentError(
        "this spectral plan was not built for a batch; pass `batch = nb` to `plan_filter`",
    ))
    size(field) == (gp.npts, bp.B) || throw(DimensionMismatch(
        "the plan was built for $(gp.npts) points × a batch of $(bp.B); got $(size(field))",
    ))
    return bp, sc.batched
end

CGEF.Filtering.analyze_buffer(plan::NUFSHTFilterPlan, field::AbstractMatrix) = CGEF.Filtering._batch_analysis(
    plan.grid_plan.batched, size(field, 2), bp -> bp.B, () -> NUFSHT.allocate_coefficients(plan.grid_plan.batched),
)

function CGEF.Filtering.filter_analyze!(
    Ĉ::AbstractArray, field::AbstractMatrix, plan::NUFSHTFilterPlan{T},
) where {T<:AbstractFloat}
    gp = plan.grid_plan
    bp, p = _batch_parts(plan, field)
    f = CGEF.Filtering._local(field, gp.own)
    _, plan.scratch.status[] = _analysis!(Ĉ, _masked(p.masked_input, f, gp.mask), bp, gp.batched_op, gp.weights, p.ws,
                                          gp.execution, gp.analysis)
    return Ĉ
end

function CGEF.Filtering.filter_synthesize!(
    out::AbstractMatrix{T}, Ĉ::AbstractArray, plan::NUFSHTFilterPlan{T},
) where {T<:AbstractFloat}
    bp, p = _batch_parts(plan, out)
    return _synthesize_into!(out, p.local_out, Ĉ, plan, bp, plan.grid_plan.own)
end

function CGEF.Filtering.filter_apply_batched!(
    out::AbstractMatrix{T}, field::AbstractMatrix, plan::NUFSHTFilterPlan{T},
) where {T<:AbstractFloat}
    size(out) == size(field) || throw(DimensionMismatch(
        "filter_apply_batched! got out $(size(out)) and field $(size(field))",
    ))
    _, p = _batch_parts(plan, field)
    CGEF.Filtering.filter_analyze!(p.coeffs, field, plan)
    return CGEF.Filtering.filter_synthesize!(out, p.coeffs, plan)
end

# ---------------------------------------------------------------------------
# The surface gradient of a filtered field, from its series
# ---------------------------------------------------------------------------

# `ū = Σ Ĝ Ĉ Y`, and under `Deformable` that over the filtered mask.
CGEF.Filtering.series_form(plan::NUFSHTFilterPlan) = CGEF.Filtering._ratio_form(plan.invrenorm)

CGEF.Filtering.rate_available(::NUFSHTFilterPlan, kernel::CGEF.Kernels.AbstractFilterKernel) =
    CGEF.Kernels.has_degree_rate(kernel)
function CGEF.Filtering.rate_plan(plan::NUFSHTFilterPlan{T}, kernel::CGEF.Kernels.AbstractFilterKernel, scale::T) where {T}
    gp = plan.grid_plan
    rate = _CGEFTransfer(CGEF.Kernels.TransferRate(kernel), scale, gp.radius, gp.plan.lmax)
    return NUFSHTFilterPlan(gp, rate, plan.invrenorm, plan.strategy, plan.scratch)
end

"""
    NUFSHTGradient

What the series gradient of a [`NUFSHTFilterPlan`](@ref) reads: a spin-1 plan over the grid plan's nodes, built
as the grid plan's own is, the spin-0 coefficients, the ladder by degree and its product with a scale's `Ĝ`
(host and where the coefficients are), `ðū` at the process's nodes, and under `Deformable` the gradient of the
filtered mask there. Built by `Filtering.series_gradient`.
"""
struct NUFSHTGradient{P, S, L<:AbstractVector, WD<:AbstractVector, Z, DM}
    plan::P
    s0::S
    ladder::L     # −√(ℓ(ℓ+1))/R: NUFSHT's harmonics take the standard ladder factor's negative
    wh::L         # the ladder times `Ĝ`, on the host
    w::WD         # the same where `s0` is
    z::Z
    dM::DM        # (∂_e M, ∂_n M) at the process's nodes, or nothing
end

_make_spin1(T, θ, φ, L, backend::CGEF.ComputationalBackends.AbstractDistributedBackend; kwargs...) =
    NUFSHT.make_spin_plan(Complex{T}, θ, φ, L, 1; execution = backend, kwargs...)
_make_spin1(T, θ, φ, L, backend::CGEF.ComputationalBackends.AbstractExecutionBackend; kwargs...) =
    NUFSHT.make_spin_plan(Complex{T}, θ, φ, L, 1; nthreads = CGEF.Filtering._library_threads(backend), kwargs...)

function CGEF.Filtering.series_gradient(plan::NUFSHTFilterPlan{T}) where {T<:AbstractFloat}
    gp = plan.grid_plan
    θ, φ = gp.nodes
    L = gp.plan.lmax
    p1 = _make_spin1(T, θ, φ, L, gp.execution; nufft = gp.nufft)
    ladder = T[-sqrt(T(ℓ * (ℓ + 1))) / gp.radius for ℓ in 0:L]
    dM = plan.invrenorm === nothing ? nothing : (similar(gp.weights), similar(gp.weights))
    return NUFSHTGradient(p1, NUFSHT.allocate_coefficients(p1), ladder, similar(ladder),
                          CGEF.Filtering._on_backend(gp.execution, copy(ladder)), similar(gp.weights, Complex{T}), dM)
end

# `ð` of the field whose scalar coefficients are `Ĉ`, filtered by `Ĝ`, at the process's nodes: `Ĉ` as spin-0
# coefficients times `Ĝ` and the ladder, a spin-1 synthesis, `ðū = ∂_n ū − i∂_e ū`.
function _nufsht_eth!(g::NUFSHTGradient, Ĉ, Ĝ::AbstractVector)
    NUFSHT.scalar_to_spin0!(g.s0, Ĉ)
    n = length(g.ladder)
    @views g.wh .= g.ladder .* Ĝ[1:n]
    copyto!(g.w, g.wh)
    g.s0 .*= g.w
    NUFSHT.nusht_type2_spin!(g.z, g.s0, g.plan)
    return g.z
end

# `(∂_e M, ∂_n M)` of the filtered mask under `Deformable`.
_mask_gradient!(_g, ::Nothing, _plan) = nothing
function _mask_gradient!(g::NUFSHTGradient, ::AbstractVector, plan::NUFSHTFilterPlan)
    z = _nufsht_eth!(g, plan.grid_plan.C_mask, plan.filter.Ĝ)
    @. g.dM[1] = -imag(z)
    @. g.dM[2] = real(z)
    return nothing
end

# The east and north components at the process's nodes; under `Deformable` `(∇N − ū∇M)/M`.
_put_gradient!(l1, l2, z, ::Nothing, _u, ::Nothing) = (@. l1 = -imag(z); @. l2 = real(z); nothing)
_put_gradient!(l1, l2, z, dM, u, ir) =
    (@. l1 = ir * (-imag(z) - u * dM[1]); @. l2 = ir * (real(z) - u * dM[2]); nothing)

function _nufsht_gradient!(g1, g2, Ĉ, ū, plan::NUFSHTFilterPlan, g::NUFSHTGradient)
    gp = plan.grid_plan
    own = gp.own
    _mask_gradient!(g, plan.invrenorm, plan)
    z = _nufsht_eth!(g, Ĉ, plan.filter.Ĝ)
    if own !== nothing
        fill!(g1, zero(eltype(g1)))
        fill!(g2, zero(eltype(g2)))
    end
    _put_gradient!(CGEF.Filtering._local(g1, own), CGEF.Filtering._local(g2, own), z, g.dM,
                   CGEF.Filtering._local(ū, own), plan.invrenorm)
    CGEF.Filtering._sum_across!(gp.execution, g1)
    CGEF.Filtering._sum_across!(gp.execution, g2)
    return nothing
end

function CGEF.Filtering.filter_synthesize_gradient!(
    g1::AbstractVector, g2::AbstractVector, Ĉ::AbstractArray, ū::AbstractVector, plan::NUFSHTFilterPlan,
    g::NUFSHTGradient,
)
    _nufsht_gradient!(g1, g2, Ĉ, ū, plan, g)
    return g1, g2
end

# A batch's coefficients `(lmax+1, 2lmax+1, nb)`, field by field.
function CGEF.Filtering.filter_synthesize_gradient!(
    g1::AbstractMatrix, g2::AbstractMatrix, Ĉ::AbstractArray{<:Any,3}, ū::AbstractMatrix, plan::NUFSHTFilterPlan,
    g::NUFSHTGradient,
)
    for b in axes(g1, 2)
        _nufsht_gradient!(view(g1, :, b), view(g2, :, b), view(Ĉ, :, :, b), view(ū, :, b), plan, g)
    end
    return g1, g2
end

# ---------------------------------------------------------------------------
# Spin-weighted transforms of the generalized convolution, over any spherical grid's cells
# ---------------------------------------------------------------------------

"""
    NUFSHTSpinTransforms

The spin-0, 1 and 2 transforms of `Filtering.GeneralizedConvolution` over every cell of a spherical grid,
in the grid's linear cell order: complex-field `SpinNUSHTplan`s to degree `lmax` at the cell centres, and
the quadrature weights, each cell's measure over `R²`, in the memory of the execution backend whose
loops read them. The analysis is the quadrature `Σ_k w_k F_k conj(ₛY_ℓm(x_k))`, exact for a field of
degree at most `lmax` on a grid whose measure is a quadrature exact to degree `2lmax`; a cell holding zero
contributes nothing.
"""
struct NUFSHTSpinTransforms{P, W<:AbstractVector{<:AbstractFloat}, B} <: CGEF.Filtering.AbstractSpinTransforms
    plans::NTuple{3,P}  # spin 0, 1, 2
    weights::W
    transfer::W         # Ĝ(ℓ) where the plans read it
    lmax::Int
    execution::B
end

Base.show(io::IO, tr::NUFSHTSpinTransforms) =
    print(io, "NUFSHTSpinTransforms(", length(tr.weights), " cells, lmax = ", tr.lmax, ")")
Base.show(io::IO, ::MIME"text/plain", tr::NUFSHTSpinTransforms) = show(io, tr)

# NUFSHT is the library `Auto` takes on a grid FastSphericalHarmonics does not transform, where a NUFFT
# library runs it. With none loaded and `nufft` left to `Auto`, NUFSHT's NUFFT is a sum over every point
# and mode, `O(M·lmax²)` per transform with no recurrence to share, and the in-core sums take its place.
function CGEF.Filtering._scattered_spin_transforms(
    auto::CGEF.SpectralBackends.AbstractAutoSpectralBackend, grid::FlowGeometries.Grids.AbstractGrid{G,T};
    nufft::CGEF.SpectralBackends.AbstractSpectralBackend = CGEF.SpectralBackends.AutoSpectralBackend(), kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}}
    _nufsht_serves(nufft) || return CGEF.Filtering._direct_spin_transforms(grid; nufft, kwargs...)
    return _nufsht_spin_transforms(grid; nufft, kwargs...)
end

# A NUFFT the caller names is taken as given.
_nufsht_serves(::CGEF.SpectralBackends.AbstractAutoSpectralBackend) =
    FTB.is_available(FTB.NonuniformFFTsBackend()) || FTB.is_available(FTB.FINUFFTBackend())
_nufsht_serves(::CGEF.SpectralBackends.AbstractSpectralBackend) = true

CGEF.Filtering._scattered_spin_transforms(
    ::CGEF.SpectralBackends.AbstractNUFSHTSpectralBackend,
    grid::FlowGeometries.Grids.AbstractGrid{G,T}; kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}} = _nufsht_spin_transforms(grid; kwargs...)

# The plans are built on the nodes in the memory of `execution`, whose loops read them: device plans for a
# device backend.
function _nufsht_spin_transforms(
    grid::FlowGeometries.Grids.AbstractGrid{G,T};
    lmax::Union{Nothing,Integer} = nothing,
    nthreads::Integer = 1,
    nufft::CGEF.SpectralBackends.AbstractSpectralBackend = CGEF.SpectralBackends.AutoSpectralBackend(),
    execution::CGEF.ComputationalBackends.AbstractLocalBackend = CGEF.ComputationalBackends.SerialBackend(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}}
    L = lmax === nothing ? CGEF.Filtering.spin_band_limit(grid) : Int(lmax)
    L >= 2 || throw(ArgumentError("the generalized convolution's spin-2 transform needs lmax ≥ 2; got $L"))
    put(a) = FlowGeometries.Execution.on_backend(execution, a)
    θ, φ, w = map(put, CGEF.Filtering._spin_nodes(grid))
    plan(s) = NUFSHT.make_spin_plan(Complex{T}, θ, φ, L, s; nthreads = Int(nthreads), nufft = nufft)
    return NUFSHTSpinTransforms((plan(0), plan(1), plan(2)), w, put(zeros(T, L + 1)), L, execution)
end

CGEF.Filtering.spin_field(tr::NUFSHTSpinTransforms) =
    fill!(similar(tr.weights, complex(eltype(tr.weights))), zero(complex(eltype(tr.weights))))
CGEF.Filtering.spin_field_size(tr::NUFSHTSpinTransforms) = (length(tr.weights),)
CGEF.Filtering.spin_position(::NUFSHTSpinTransforms) = CGEF.Filtering.CellOrder()
CGEF.Filtering.spin_coefficients(tr::NUFSHTSpinTransforms) = NUFSHT.allocate_coefficients(first(tr.plans))
CGEF.Filtering.spin_transfer_buffer(tr::NUFSHTSpinTransforms) = zeros(eltype(tr.weights), tr.lmax + 1)
CGEF.Filtering.spin_host(tr::NUFSHTSpinTransforms) = CGEF.Filtering._host_memory(tr.execution)

function CGEF.Filtering.spin_analyze!(C, F, tr::NUFSHTSpinTransforms, s::Int)
    NUFSHT.nusht_analyze!(C, F, tr.plans[s + 1]; weights = tr.weights)
    return C
end

CGEF.Filtering.spin_weights(tr::NUFSHTSpinTransforms) = tr.weights

# The coefficients are `C[ℓ+1, m+lmax+1]`, so row `r` is degree `r − 1`. `Ĝ` is in host memory.
function CGEF.Filtering.spin_synthesize!(F, C, W, Ĝ, tr::NUFSHTSpinTransforms, s::Int)
    copyto!(tr.transfer, Ĝ)
    W .= C .* tr.transfer
    NUFSHT.nusht_type2_spin!(F, W, tr.plans[s + 1])
    return F
end

# NUFSHT's harmonics are the ring transforms': a step of `ð` or `ð̄` is the standard ladder factor's negative.
function CGEF.Filtering.spin_ladder_synthesize!(F, C, W, d, tr::NUFSHTSpinTransforms, s::Int, t::Int)
    CGEF.Filtering.spin_synthesize!(F, C, W, d, tr, t)
    F .= .-F
    return F
end

end # module
