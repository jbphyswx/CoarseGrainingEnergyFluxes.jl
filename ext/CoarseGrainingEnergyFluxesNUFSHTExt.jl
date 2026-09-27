module CoarseGrainingEnergyFluxesNUFSHTExt

using NUFSHT: NUFSHT
using FlowTransformBindings: FlowTransformBindings as FTB
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF
using FlowGeometries: FlowGeometries

# Spectral filtering for scattered spherical data through NUFSHT.jl: fit the field's harmonic
# coefficients by least squares (`nusht_solve!`), multiply each degree by the kernel's transfer
# function, synthesize at the points (`nusht_synthesize!`). The fit makes this the filter `A H A⁺`;
# the adjoint in its place gives the smoothing `A H A†`. A single apply and the analyze/synthesize pair
# of a sweep run the same fit.
#
# The transfer is the shared `CGEF.Kernels.spectral_transfer_degree(kernel, l, ℓ, R)`, so the kernel
# convention matches the other spectral backends.

# Adapter exposing CGEF's shared transfer function to NUFSHT's per-degree `kernel_transfer`.
struct _CGEFTransfer{K<:CGEF.Kernels.AbstractFilterKernel, T<:AbstractFloat} <: NUFSHT.AbstractSpectralTransfer
    kernel::K
    scale::T
    R::T
end
@inline NUFSHT.kernel_transfer(t::_CGEFTransfer, l) =
    CGEF.Kernels.spectral_transfer_degree(t.kernel, l, t.scale, t.R)

"""
    NUFSHTGridPlan

The half of a scattered-spherical plan the filter scale does not reach: the NUSHT plan over the grid's
nodes this process transforms (under `MPIBackend` its share of them, under `DistributedBackend` a plan
NUFSHT divides among the worker processes), the sphere radius, the mask over those nodes, the mask's
fitted coefficients, the execution backend whose memory holds them, and the nodes owned (`nothing` for
all). A plan built for a trailing batch axis of `nb` fields also holds a
`NUSHTplan` with `ntrans = nb` over the same nodes, whose fit and synthesis carry every field of the
batch at once.

The band limit is the largest `L` with `(L+1)(2L+1) ≤ npts`: the `(L+1)²` coefficients are fitted from
about twice as many points, and a Clenshaw–Curtis grid of degree `L` gets exactly `L`.

A `NUSHTplan` holds buffers every transform overwrites, so two tasks may not execute one grid plan
concurrently; a concurrent driver needs its own grid plan per worker.
"""
struct NUFSHTGridPlan{P, T<:AbstractFloat, M, CM, BP, B<:CGEF.ComputationalBackends.AbstractExecutionBackend, O} <:
       CGEF.Filtering.AbstractGridPlan
    plan::P
    radius::T
    mask::M          # the grid's mask over this process's nodes, or nothing when fully active
    C_mask::CM       # the mask's fitted coefficients, or nothing when unmasked
    npts::Int        # the grid's node count
    batched::BP      # a `NUSHTplan` with `ntrans = nb`, or nothing
    backend::B
    own::O           # this process's nodes, or nothing for all of them
end

"""
    NUFSHTScratch

The transient half of a scattered-spherical plan: the staging array for `mask .* field`, the fitted
coefficients, the least-squares workspace, and the synthesis of this process's nodes where it holds
only some of them; and the same set for the batched plan, so an apply allocates nothing. One per
concurrent worker.
"""
struct NUFSHTScratch{T<:AbstractFloat, SV<:AbstractVector{T}, CA<:AbstractArray, W, LO, BT} <:
       CGEF.Filtering.AbstractFilterScratch
    masked_input::SV   # unused when mask === nothing
    coeffs::CA
    ws::W              # FTB.LSMRWorkspace, or nothing where NUFSHT's workers hold theirs
    local_out::LO      # this process's synthesized values, or nothing when it holds every node
    batched::BT        # (; masked_input, coeffs, ws, local_out) for the batched plan, or nothing
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

# The fit cannot resolve a relative residual finer than the transform it is built on.
_fit_rtol(tol, ::Type{T}) where {T<:AbstractFloat} = max(T(10 * tol), sqrt(eps(T)))

# Under `MPIBackend` each rank holds a plan over its own nodes, and the fit is NUFSHT's MPI solve: one
# least-squares problem over every rank's nodes, with the coefficients replicated on every rank. A plan
# divided among `DistributedBackend` workers carries its own solve.
_solve!(C, f, plan, ws, rtol, ::CGEF.ComputationalBackends.AbstractExecutionBackend) =
    NUFSHT.nusht_solve!(C, f, plan; ws = ws, rtol = rtol)
_solve!(C, f, plan, ws, rtol, b::CGEF.ComputationalBackends.AbstractMPIBackend) =
    NUFSHT.nusht_solve!(C, f, plan, b; ws = ws, rtol = rtol)

function _fit!(C, f, plan, ws, backend)
    rtol = _fit_rtol(plan.tol, real(eltype(C)))
    _, iters, rel, converged = _solve!(C, f, plan, ws, rtol, backend)
    converged || throw(ErrorException(
        "NUFSHT spectral filtering: the least-squares fit of the degree-$(plan.lmax) harmonic " *
        "coefficients stopped at relative residual $rel after $iters iterations, above the tolerance " *
        "$rtol. The nodes do not determine those coefficients.",
    ))
    return C
end

# This process's plan at `_library_threads(backend)` threads, or under `DistributedBackend` one whose
# points NUFSHT divides among the workers, each at the inner backend's count.
_make_plan(T, θ, φ, lmax, backend::CGEF.ComputationalBackends.AbstractDistributedBackend; kwargs...) =
    NUFSHT.make_plan(T, θ, φ, lmax, backend; kwargs...)
_make_plan(T, θ, φ, lmax, backend::CGEF.ComputationalBackends.AbstractExecutionBackend; kwargs...) =
    NUFSHT.make_plan(T, θ, φ, lmax; nthreads = CGEF.Filtering._library_threads(backend), kwargs...)

# The fit's workspace for a plan `_make_plan` built; the workers of a divided plan hold their own.
_workspace(plan, ::CGEF.ComputationalBackends.AbstractDistributedBackend) = nothing
_workspace(plan, ::CGEF.ComputationalBackends.AbstractExecutionBackend) = FTB.LSMRWorkspace(plan)

# `nufft` is the NUFFT library NUFSHT runs: a FlowTransformBindings tag, or `AutoSpectralBackend()` for
# NUFSHT's own choice.
function _nufsht_grid_plan(
    grid::FlowGeometries.Grids.UnstructuredGrid{T,G,2};
    backend::CGEF.ComputationalBackends.AbstractExecutionBackend = CGEF.ComputationalBackends.AutoBackend(),
    batch::Union{Nothing,Integer} = nothing,
    nufft::CGEF.SpectralBackends.AbstractSpectralBackend = CGEF.SpectralBackends.AutoSpectralBackend(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.SphericalGeometry{T}}
    npts = length(FlowGeometries.Grids.coordinates(grid, 1))
    npts > 0 || throw(ArgumentError("NUFSHT spectral filtering needs at least one point."))
    lmax = max(1, floor(Int, (sqrt(1 + 8 * npts) - 3) / 4))
    # This process's nodes go where `backend` runs, and NUFSHT plans in the memory they are in.
    own = CGEF.Filtering._owned(backend, npts)
    part(v) = own === nothing ? v : v[own]
    θ = CGEF.Filtering._on_backend(backend, T(π) / 2 .- T.(part(FlowGeometries.Grids.coordinates(grid, 2))))  # colatitude
    φ = CGEF.Filtering._on_backend(backend, T.(part(FlowGeometries.Grids.coordinates(grid, 1))))
    nplan = _make_plan(T, θ, φ, lmax, backend; nufft = nufft)
    bplan = batch === nothing ? nothing : _make_plan(T, θ, φ, lmax, backend; ntrans = Int(batch), nufft = nufft)
    m = FlowGeometries.Grids.mask(grid)
    mask = all(m) ? nothing : CGEF.Filtering._on_backend(backend, part(m))
    C_mask = mask === nothing ? nothing :
             _fit!(NUFSHT.allocate_coefficients(nplan), CGEF.Filtering._on_backend(backend, T.(part(m))), nplan,
                   _workspace(nplan, backend), backend)
    return NUFSHTGridPlan(
        nplan, T(FlowGeometries.Geometry.radius(FlowGeometries.Grids.grid_geometry(grid))), mask,
        C_mask, npts, bplan, backend, own === nothing ? nothing : CGEF.Filtering._on_backend(backend, own),
    )
end

function _nufsht_scratch(gp::NUFSHTGridPlan{P,T}) where {P, T<:AbstractFloat}
    bp = gp.batched
    n = gp.own === nothing ? gp.npts : length(gp.own)
    zs(dims...) = CGEF.Filtering._allocate(gp.backend, T, dims)
    return NUFSHTScratch(
        zs(n), NUFSHT.allocate_coefficients(gp.plan), _workspace(gp.plan, gp.backend),
        gp.own === nothing ? nothing : zs(n),
        bp === nothing ? nothing :
            (masked_input = zs(n, bp.B), coeffs = NUFSHT.allocate_coefficients(bp),
             ws = _workspace(bp, gp.backend), local_out = gp.own === nothing ? nothing : zs(n, bp.B)),
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
    return CGEF.Filtering._sum_across!(plan.grid_plan.backend, out)
end

CGEF.Filtering.spectral_grid_plan(
    ::Union{CGEF.SpectralBackends.AbstractAutoSpectralBackend, CGEF.SpectralBackends.AbstractNUFSHTSpectralBackend},
    grid::FlowGeometries.Grids.UnstructuredGrid{T,G,2},
    kernel::CGEF.Kernels.AbstractFilterKernel;
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy = CGEF.Filtering.ZeroFill(),
    backend::CGEF.ComputationalBackends.AbstractExecutionBackend = CGEF.ComputationalBackends.AutoBackend(),
    batch::Union{Nothing,Integer} = nothing,
    nufft::CGEF.SpectralBackends.AbstractSpectralBackend = CGEF.SpectralBackends.AutoSpectralBackend(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.SphericalGeometry{T}} =
    _nufsht_grid_plan(grid; backend = backend, batch = batch, nufft = nufft)

CGEF.Filtering.spectral_scratch(gp::NUFSHTGridPlan) = _nufsht_scratch(gp)

function CGEF.Filtering.spectral_filter_plan(
    ::Union{CGEF.SpectralBackends.AbstractAutoSpectralBackend, CGEF.SpectralBackends.AbstractNUFSHTSpectralBackend},
    grid::FlowGeometries.Grids.UnstructuredGrid{T,G,2},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy = CGEF.Filtering.ZeroFill(),
    backend::CGEF.ComputationalBackends.AbstractExecutionBackend = CGEF.ComputationalBackends.AutoBackend(),
    # Extent of the trailing batch axis this plan will be applied over, or `nothing` for single fields.
    batch::Union{Nothing,Integer} = nothing,
    nufft::CGEF.SpectralBackends.AbstractSpectralBackend = CGEF.SpectralBackends.AutoSpectralBackend(),
    grid_plan::Union{Nothing,NUFSHTGridPlan} = nothing,
    scratch::Union{Nothing,NUFSHTScratch} = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.SphericalGeometry{T}}
    gp = grid_plan === nothing ? _nufsht_grid_plan(grid; backend = backend, batch = batch, nufft = nufft) : grid_plan
    sc = scratch === nothing ? _nufsht_scratch(gp) : scratch
    filter = _CGEFTransfer(kernel, scale, gp.radius)
    # `ZeroFill` is already exactly `filter(mask .* field)`; only `Deformable` divides by the local mass,
    # the mask's fit synthesized through this scale's transfer. The mask is fixed for the plan, so it is
    # formed once here and stored inverted.
    invrenorm = if gp.mask !== nothing && mask_strategy isa CGEF.Filtering.Deformable
        mf = CGEF.Filtering._allocate(gp.backend, T, (gp.own === nothing ? gp.npts : length(gp.own),))
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

# The fit (points → harmonic coefficients) depends on the field alone, so a sweep runs it once and each
# scale only applies its own transfer function and evaluates back to points. `nusht_synthesize!` leaves
# `C` intact, so the same coefficients serve every scale.
CGEF.Filtering.analyze_buffer(plan::NUFSHTFilterPlan, ::AbstractVector) =
    NUFSHT.allocate_coefficients(plan.grid_plan.plan)

function CGEF.Filtering.filter_analyze!(
    Ĉ::AbstractArray, field::AbstractVector, plan::NUFSHTFilterPlan{T},
) where {T<:AbstractFloat}
    gp, sc = plan.grid_plan, plan.scratch
    f = CGEF.Filtering._local(field, gp.own)
    return _fit!(Ĉ, _masked(sc.masked_input, f, gp.mask), gp.plan, sc.ws, gp.backend)
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

CGEF.Filtering.analyze_buffer(plan::NUFSHTFilterPlan, field::AbstractMatrix) =
    (plan.grid_plan.batched === nothing || size(field, 2) != plan.grid_plan.batched.B) ? nothing :
        NUFSHT.allocate_coefficients(plan.grid_plan.batched)

function CGEF.Filtering.filter_analyze!(
    Ĉ::AbstractArray, field::AbstractMatrix, plan::NUFSHTFilterPlan{T},
) where {T<:AbstractFloat}
    gp = plan.grid_plan
    bp, p = _batch_parts(plan, field)
    f = CGEF.Filtering._local(field, gp.own)
    return _fit!(Ĉ, _masked(p.masked_input, f, gp.mask), bp, p.ws, gp.backend)
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

end # module
