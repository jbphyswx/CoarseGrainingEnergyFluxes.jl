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
grid's nodes, the sphere radius, the mask, and the mask's fitted coefficients.

The band limit is the largest `L` with `(L+1)(2L+1) ≤ npts`: the `(L+1)²` coefficients are fitted from
about twice as many points, and a Clenshaw–Curtis grid of degree `L` gets exactly `L`.

The `NUSHTplan` holds buffers every transform overwrites, so two tasks may not execute one grid plan
concurrently; a concurrent driver needs its own grid plan per worker.
"""
struct NUFSHTGridPlan{P, T<:AbstractFloat, M, CM} <: CGEF.Filtering.AbstractGridPlan
    plan::P
    radius::T
    mask::M          # the grid's mask, or nothing when fully active
    C_mask::CM       # the mask's fitted coefficients, or nothing when unmasked
    npts::Int
end

"""
    NUFSHTScratch

The transient half of a scattered-spherical plan: the staging vector for `mask .* field`, the
filtered mask `Deformable` divides by, the fitted coefficients, and the least-squares workspace, so an
apply allocates nothing. One per concurrent worker.
"""
struct NUFSHTScratch{T<:AbstractFloat, SV<:AbstractVector{T}, CA<:AbstractArray, W} <:
       CGEF.Filtering.AbstractFilterScratch
    masked_input::SV   # unused when mask === nothing
    mask_filt::SV      # unused unless `Deformable`
    coeffs::CA
    ws::W              # NUFSHT.LSMRWorkspace
end

"""
    NUFSHTFilterPlan

Cached scattered-spherical filter plan: the shared [`NUFSHTGridPlan`](@ref), the CGEF transfer adapter,
whether `Deformable` renormalization applies, and the [`NUFSHTScratch`](@ref) an apply works in. Built
by `plan_filter(scattered_spherical_grid, kernel, scale; method = Spectral())`.
"""
struct NUFSHTFilterPlan{T<:AbstractFloat, GP<:NUFSHTGridPlan, F, SC<:NUFSHTScratch{T}} <:
       CGEF.Filtering.AbstractFilterPlan
    grid_plan::GP
    filter::F
    renorm::Bool     # divide by the filtered mask mass: `Deformable` only, never `ZeroFill`
    scratch::SC
end

# The fit cannot resolve a relative residual finer than the transform it is built on.
_fit_rtol(plan::NUFSHT.NUSHTplan{T}) where {T} = max(T(10 * plan.tol), sqrt(eps(T)))

function _fit!(C, f, plan::NUFSHT.NUSHTplan, ws)
    rtol = _fit_rtol(plan)
    _, iters, rel, converged = NUFSHT.nusht_solve!(C, f, plan; ws = ws, rtol = rtol)
    converged || throw(ErrorException(
        "NUFSHT spectral filtering: the least-squares fit of the degree-$(plan.lmax) harmonic " *
        "coefficients stopped at relative residual $rel after $iters iterations, above the tolerance " *
        "$rtol. The $(length(f)) points do not determine those coefficients.",
    ))
    return C
end

function _nufsht_grid_plan(
    grid::FlowGeometries.Grids.UnstructuredGrid{T,G},
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.SphericalGeometry{T}}
    npts = length(FlowGeometries.Grids.coordinates(grid, 1))
    npts > 0 || throw(ArgumentError("NUFSHT spectral filtering needs at least one point."))
    lmax = max(1, floor(Int, (sqrt(1 + 8 * npts) - 3) / 4))
    θ = T(π) / 2 .- FlowGeometries.Grids.coordinates(grid, 2)        # colatitude from latitude
    φ = FlowGeometries.Grids.coordinates(grid, 1)
    nplan = NUFSHT.make_plan(T, collect(T, θ), collect(T, φ), lmax)
    mask = all(FlowGeometries.Grids.mask(grid)) ? nothing : FlowGeometries.Grids.mask(grid)
    C_mask = mask === nothing ? nothing :
             _fit!(NUFSHT.allocate_coefficients(nplan), mask, nplan, NUFSHT.LSMRWorkspace(nplan))
    return NUFSHTGridPlan(
        nplan, T(FlowGeometries.Geometry.radius(FlowGeometries.Grids.grid_geometry(grid))), mask,
        C_mask, npts,
    )
end

_nufsht_scratch(gp::NUFSHTGridPlan{P,T}) where {P, T<:AbstractFloat} = NUFSHTScratch(
    zeros(T, gp.npts), zeros(T, gp.npts), NUFSHT.allocate_coefficients(gp.plan),
    NUFSHT.LSMRWorkspace(gp.plan),
)

CGEF.Filtering.spectral_grid_plan(
    ::Union{CGEF.SpectralBackends.AbstractAutoSpectralBackend, CGEF.SpectralBackends.AbstractNUFSHTSpectralBackend},
    grid::FlowGeometries.Grids.UnstructuredGrid{T,G},
    kernel::CGEF.Kernels.AbstractFilterKernel;
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.SphericalGeometry{T}} = _nufsht_grid_plan(grid)

CGEF.Filtering.spectral_scratch(gp::NUFSHTGridPlan) = _nufsht_scratch(gp)

function CGEF.Filtering.spectral_filter_plan(
    ::Union{CGEF.SpectralBackends.AbstractAutoSpectralBackend, CGEF.SpectralBackends.AbstractNUFSHTSpectralBackend},
    grid::FlowGeometries.Grids.UnstructuredGrid{T,G},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy = CGEF.Filtering.ZeroFill(),
    backend = CGEF.ComputationalBackends.AutoBackend(),
    grid_plan::Union{Nothing,NUFSHTGridPlan} = nothing,
    scratch::Union{Nothing,NUFSHTScratch} = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.SphericalGeometry{T}}
    gp = grid_plan === nothing ? _nufsht_grid_plan(grid) : grid_plan
    sc = scratch === nothing ? _nufsht_scratch(gp) : scratch
    filter = _CGEFTransfer(kernel, scale, gp.radius)
    # `ZeroFill` is already exactly `filter(mask .* field)`; only `Deformable` divides by the local mass.
    renorm = gp.mask !== nothing && mask_strategy isa CGEF.Filtering.Deformable
    return NUFSHTFilterPlan(gp, filter, renorm, sc)
end

# The fit (points → harmonic coefficients) depends on the field alone, so a sweep runs it once and each
# scale only applies its own transfer function and evaluates back to points. `nusht_synthesize!` leaves
# `C` intact, so the same coefficients serve every scale.
CGEF.Filtering.analyze_buffer(plan::NUFSHTFilterPlan, ::AbstractVector) =
    NUFSHT.allocate_coefficients(plan.grid_plan.plan)

function CGEF.Filtering.filter_analyze!(
    Ĉ::AbstractArray, field::AbstractVector, plan::NUFSHTFilterPlan{T},
) where {T<:AbstractFloat}
    # A `Bool` mask is a strong zero: a masked point is fitted as 0 even where the field is NaN.
    gp, sc = plan.grid_plan, plan.scratch
    src = gp.mask === nothing ? field : (sc.masked_input .= gp.mask .* field)
    return _fit!(Ĉ, src, gp.plan, sc.ws)
end

function CGEF.Filtering.filter_synthesize!(
    out::AbstractVector{T}, Ĉ::AbstractArray, plan::NUFSHTFilterPlan{T},
) where {T<:AbstractFloat}
    gp, sc = plan.grid_plan, plan.scratch
    NUFSHT.nusht_synthesize!(out, Ĉ, plan.filter, gp.plan)
    plan.renorm && NUFSHT.nusht_filter_renorm!(out, gp.mask, plan.filter, gp.plan;
                                               mask_filt = sc.mask_filt, ws = sc.ws, C_mask = gp.C_mask)
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

end # module
