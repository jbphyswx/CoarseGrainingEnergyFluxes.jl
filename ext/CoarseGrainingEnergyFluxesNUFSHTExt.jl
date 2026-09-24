module CoarseGrainingEnergyFluxesNUFSHTExt

using NUFSHT: NUFSHT
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF
using FlowGeometries: FlowGeometries

# Spectral filtering for scattered spherical data, delegated to NUFSHT.jl, which already implements
# the whole pipeline — analysis, per-degree transfer multiply, synthesis, optional mask
# renormalization. This extension builds the plan from the grid's (colatitude, longitude) nodes and
# drives `nusht_filter!`.
#
# NUFSHT's own `GaussianTransfer` is deliberately bypassed: an adapter feeds it the shared
# `CGEF.Kernels.spectral_transfer(kernel, k_l, ℓ)` per degree, `k_l = √(l(l+1))/R`, so the Gaussian
# convention matches the other spectral backends exactly.
#
# `nusht_filter!` uses the adjoint analysis — exact on a Clenshaw–Curtis grid, well-behaved for
# quasi-uniform scattered sampling, ill-conditioned for very irregular sampling, where NUFSHT's
# `nusht_solve!` offers CG inversion instead.

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
nodes, the sphere radius, and the mask.

Only the transfer adapter depends on ℓ, and it is three fields. Everything expensive — the bandlimit
choice, the Clenshaw–Curtis detection, and building the transform over every node — belongs here, so a
sweep pays for it once.

The NUSHT plan carries the coefficient workspace a transform overwrites, so two tasks may not execute
one concurrently; a concurrent driver needs its own grid plan per worker.
"""
struct NUFSHTGridPlan{P, T<:AbstractFloat, M} <: CGEF.Filtering.AbstractGridPlan
    plan::P
    radius::T
    mask::M          # Vector{T} of 0/1, or nothing when fully active (unmasked)
    npts::Int
end

"""
    NUFSHTScratch

The transient half of a scattered-spherical plan: the length-`npts` staging vector for `mask .* field`,
so a masked apply allocates nothing. One per concurrent worker.
"""
struct NUFSHTScratch{T<:AbstractFloat, SV<:AbstractVector{T}} <: CGEF.Filtering.AbstractFilterScratch
    masked_input::SV   # unused when mask === nothing
end

"""
    NUFSHTFilterPlan

Cached scattered-spherical filter plan: the shared [`NUFSHTGridPlan`](@ref), the CGEF transfer adapter,
whether `Deformable` renormalization applies, and the [`NUFSHTScratch`](@ref) a masked apply stages
through. Built by `plan_filter(scattered_spherical_grid, kernel, scale; method = Spectral())`.
"""
struct NUFSHTFilterPlan{T<:AbstractFloat, GP<:NUFSHTGridPlan, F, SC<:NUFSHTScratch{T}} <:
       CGEF.Filtering.AbstractFilterPlan
    grid_plan::GP
    filter::F
    renorm::Bool     # divide by the filtered mask mass: `Deformable` only, never `ZeroFill`
    scratch::SC
end

function _nufsht_grid_plan(
    grid::FlowGeometries.Grids.UnstructuredGrid{T,G},
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.SphericalGeometry{T}}
    npts = length(FlowGeometries.Grids.coordinates(grid, 1))
    npts > 0 || throw(ArgumentError("NUFSHT spectral filtering needs at least one point."))
    # Bandlimit. A Clenshaw–Curtis grid has npts = (L+1)(2L+1); detect it and use that exact L so the
    # adjoint analysis is an EXACT round-trip. For genuinely irregular sampling fall back to the
    # solvability bound lmax ≈ √npts − 1 (the adjoint filter is then approximate — use NUFSHT's
    # `nusht_solve!` directly for ill-conditioned point sets).
    Lcc = (-3 + sqrt(1 + 8 * npts)) / 4
    Lr = round(Int, Lcc)
    is_clenshaw_curtis = Lr >= 1 && (Lr + 1) * (2Lr + 1) == npts
    if !is_clenshaw_curtis
        @warn "NUFSHT spectral filtering: the point set is not an exact Clenshaw–Curtis grid, so the " *
              "adjoint analysis is only approximate (falling back to the heuristic bandlimit " *
              "lmax ≈ √npts − 1). For genuinely irregular/ill-conditioned point sets, use NUFSHT's " *
              "`nusht_solve!` directly for an exact (iteratively-solved) inversion instead." maxlog=1
    end
    lmax = is_clenshaw_curtis ? Lr : max(1, floor(Int, sqrt(npts)) - 1)
    θ = T(π) / 2 .- FlowGeometries.Grids.coordinates(grid, 2)        # colatitude from latitude
    φ = FlowGeometries.Grids.coordinates(grid, 1)
    # Element type is the leading positional argument, not a keyword.
    nplan = NUFSHT.make_plan(T, collect(T, θ), collect(T, φ), lmax)
    mask = all(FlowGeometries.Grids.mask(grid)) ? nothing : T.(FlowGeometries.Grids.mask(grid))
    return NUFSHTGridPlan(
        nplan, T(FlowGeometries.Geometry.radius(FlowGeometries.Grids.grid_geometry(grid))), mask, npts,
    )
end

_nufsht_scratch(gp::NUFSHTGridPlan{P,T}) where {P, T<:AbstractFloat} =
    NUFSHTScratch(zeros(T, gp.npts))

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

# The forward transform (points → harmonic coefficients) depends on the field alone, so a sweep runs it
# once and each scale only applies its own transfer function and evaluates back to points.
# `nusht_synthesize!` leaves `C` intact, so the same coefficients serve every scale.
CGEF.Filtering.analyze_buffer(plan::NUFSHTFilterPlan, ::AbstractVector) =
    NUFSHT.allocate_coefficients(plan.grid_plan.plan)

function CGEF.Filtering.filter_analyze!(
    Ĉ::AbstractArray, field::AbstractVector, plan::NUFSHTFilterPlan{T},
) where {T<:AbstractFloat}
    gp, sc = plan.grid_plan, plan.scratch
    if gp.mask === nothing
        NUFSHT.nusht_type1!(Ĉ, convert(Vector{T}, field), gp.plan)
    else
        sc.masked_input .= field .* gp.mask
        NUFSHT.nusht_type1!(Ĉ, sc.masked_input, gp.plan)
    end
    return Ĉ
end

function CGEF.Filtering.filter_synthesize!(
    out::AbstractVector{T}, Ĉ::AbstractArray, plan::NUFSHTFilterPlan{T},
) where {T<:AbstractFloat}
    gp = plan.grid_plan
    NUFSHT.nusht_synthesize!(out, Ĉ, plan.filter, gp.plan)
    plan.renorm && NUFSHT.nusht_filter_renorm!(out, gp.mask, plan.filter, gp.plan)
    return out
end

function CGEF.Filtering.filter_apply!(
    out::AbstractVector{T},
    field::AbstractVector,
    plan::NUFSHTFilterPlan{T},
) where {T<:AbstractFloat}
    gp, sc = plan.grid_plan, plan.scratch
    if gp.mask === nothing
        NUFSHT.nusht_filter!(out, convert(Vector{T}, field), plan.filter, gp.plan)
        return out
    end
    sc.masked_input .= field .* gp.mask
    NUFSHT.nusht_filter!(out, sc.masked_input, plan.filter, gp.plan)
    plan.renorm && NUFSHT.nusht_filter_renorm!(out, gp.mask, plan.filter, gp.plan)
    return out
end

end # module
