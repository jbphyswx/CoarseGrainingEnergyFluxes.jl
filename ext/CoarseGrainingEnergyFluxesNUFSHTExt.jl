module CoarseGrainingEnergyFluxesNUFSHTExt

using NUFSHT: NUFSHT
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

The half of a scattered-spherical plan the filter scale does not reach: the `NUFSHT.NUSHTplan` over the
grid's nodes, the sphere radius, the mask, and the mask's fitted coefficients. A plan built for a
trailing batch axis of `nb` fields also holds a `NUSHTplan` with `ntrans = nb` over the same nodes, whose
fit and synthesis carry every field of the batch at once.

The band limit is the largest `L` with `(L+1)(2L+1) ≤ npts`: the `(L+1)²` coefficients are fitted from
about twice as many points, and a Clenshaw–Curtis grid of degree `L` gets exactly `L`.

A `NUSHTplan` holds buffers every transform overwrites, so two tasks may not execute one grid plan
concurrently; a concurrent driver needs its own grid plan per worker.
"""
struct NUFSHTGridPlan{P, T<:AbstractFloat, M, CM, BP} <: CGEF.Filtering.AbstractGridPlan
    plan::P
    radius::T
    mask::M          # the grid's mask, or nothing when fully active
    C_mask::CM       # the mask's fitted coefficients, or nothing when unmasked
    npts::Int
    batched::BP      # a `NUSHTplan` with `ntrans = nb`, or nothing
end

"""
    NUFSHTScratch

The transient half of a scattered-spherical plan: the staging array for `mask .* field`, the fitted
coefficients and the least-squares workspace, and the same set for the batched plan, so an apply
allocates nothing. One per concurrent worker.
"""
struct NUFSHTScratch{T<:AbstractFloat, SV<:AbstractVector{T}, CA<:AbstractArray, W, BT} <:
       CGEF.Filtering.AbstractFilterScratch
    masked_input::SV   # unused when mask === nothing
    coeffs::CA
    ws::W              # NUFSHT.LSMRWorkspace
    batched::BT        # (; masked_input, coeffs, ws) for the batched plan, or nothing
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
_fit_rtol(plan::NUFSHT.NUSHTplan{T}) where {T} = max(T(10 * plan.tol), sqrt(eps(T)))

function _fit!(C, f, plan::NUFSHT.NUSHTplan, ws)
    rtol = _fit_rtol(plan)
    _, iters, rel, converged = NUFSHT.nusht_solve!(C, f, plan; ws = ws, rtol = rtol)
    converged || throw(ErrorException(
        "NUFSHT spectral filtering: the least-squares fit of the degree-$(plan.lmax) harmonic " *
        "coefficients stopped at relative residual $rel after $iters iterations, above the tolerance " *
        "$rtol. The $(size(f, 1)) points do not determine those coefficients.",
    ))
    return C
end

function _nufsht_grid_plan(
    grid::FlowGeometries.Grids.UnstructuredGrid{T,G,2}; batch::Union{Nothing,Integer} = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.SphericalGeometry{T}}
    npts = length(FlowGeometries.Grids.coordinates(grid, 1))
    npts > 0 || throw(ArgumentError("NUFSHT spectral filtering needs at least one point."))
    lmax = max(1, floor(Int, (sqrt(1 + 8 * npts) - 3) / 4))
    θ = collect(T, T(π) / 2 .- FlowGeometries.Grids.coordinates(grid, 2))   # colatitude from latitude
    φ = collect(T, FlowGeometries.Grids.coordinates(grid, 1))
    nplan = NUFSHT.make_plan(T, θ, φ, lmax)
    bplan = batch === nothing ? nothing : NUFSHT.make_plan(T, θ, φ, lmax; ntrans = Int(batch))
    mask = all(FlowGeometries.Grids.mask(grid)) ? nothing : FlowGeometries.Grids.mask(grid)
    C_mask = mask === nothing ? nothing :
             _fit!(NUFSHT.allocate_coefficients(nplan), mask, nplan, NUFSHT.LSMRWorkspace(nplan))
    return NUFSHTGridPlan(
        nplan, T(FlowGeometries.Geometry.radius(FlowGeometries.Grids.grid_geometry(grid))), mask,
        C_mask, npts, bplan,
    )
end

function _nufsht_scratch(gp::NUFSHTGridPlan{P,T}) where {P, T<:AbstractFloat}
    bp = gp.batched
    return NUFSHTScratch(
        zeros(T, gp.npts), NUFSHT.allocate_coefficients(gp.plan), NUFSHT.LSMRWorkspace(gp.plan),
        bp === nothing ? nothing :
            (masked_input = zeros(T, gp.npts, bp.B), coeffs = NUFSHT.allocate_coefficients(bp),
             ws = NUFSHT.LSMRWorkspace(bp)),
    )
end

CGEF.Filtering.spectral_grid_plan(
    ::Union{CGEF.SpectralBackends.AbstractAutoSpectralBackend, CGEF.SpectralBackends.AbstractNUFSHTSpectralBackend},
    grid::FlowGeometries.Grids.UnstructuredGrid{T,G,2},
    kernel::CGEF.Kernels.AbstractFilterKernel;
    batch::Union{Nothing,Integer} = nothing,
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.SphericalGeometry{T}} = _nufsht_grid_plan(grid; batch = batch)

CGEF.Filtering.spectral_scratch(gp::NUFSHTGridPlan) = _nufsht_scratch(gp)

function CGEF.Filtering.spectral_filter_plan(
    ::Union{CGEF.SpectralBackends.AbstractAutoSpectralBackend, CGEF.SpectralBackends.AbstractNUFSHTSpectralBackend},
    grid::FlowGeometries.Grids.UnstructuredGrid{T,G,2},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy = CGEF.Filtering.ZeroFill(),
    backend = CGEF.ComputationalBackends.AutoBackend(),
    # Extent of the trailing batch axis this plan will be applied over, or `nothing` for single fields.
    batch::Union{Nothing,Integer} = nothing,
    grid_plan::Union{Nothing,NUFSHTGridPlan} = nothing,
    scratch::Union{Nothing,NUFSHTScratch} = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.SphericalGeometry{T}}
    gp = grid_plan === nothing ? _nufsht_grid_plan(grid; batch = batch) : grid_plan
    sc = scratch === nothing ? _nufsht_scratch(gp) : scratch
    filter = _CGEFTransfer(kernel, scale, gp.radius)
    # `ZeroFill` is already exactly `filter(mask .* field)`; only `Deformable` divides by the local mass,
    # the mask's fit synthesized through this scale's transfer. The mask is fixed for the plan, so it is
    # formed once here and stored inverted.
    invrenorm = if gp.mask !== nothing && mask_strategy isa CGEF.Filtering.Deformable
        mf = zeros(T, gp.npts)
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
    return _fit!(Ĉ, _masked(sc.masked_input, field, gp.mask), gp.plan, sc.ws)
end

function CGEF.Filtering.filter_synthesize!(
    out::AbstractVector{T}, Ĉ::AbstractArray, plan::NUFSHTFilterPlan{T},
) where {T<:AbstractFloat}
    NUFSHT.nusht_synthesize!(out, Ĉ, plan.filter, plan.grid_plan.plan)
    plan.invrenorm === nothing || (out .*= plan.invrenorm)
    return out
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
    bp, p = _batch_parts(plan, field)
    return _fit!(Ĉ, _masked(p.masked_input, field, plan.grid_plan.mask), bp, p.ws)
end

function CGEF.Filtering.filter_synthesize!(
    out::AbstractMatrix{T}, Ĉ::AbstractArray, plan::NUFSHTFilterPlan{T},
) where {T<:AbstractFloat}
    bp, _ = _batch_parts(plan, out)
    NUFSHT.nusht_synthesize!(out, Ĉ, plan.filter, bp)
    plan.invrenorm === nothing || (out .*= plan.invrenorm)
    return out
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
