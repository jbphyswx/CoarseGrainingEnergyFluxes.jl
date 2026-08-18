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
    NUFSHTFilterPlan

Cached scattered-spherical filter plan: the NUSHT plan over the grid's nodes, the CGEF transfer
adapter, the (optional) mask, and a scratch buffer for `mask · field` so a masked apply allocates
nothing. Built by `plan_filter(scattered_spherical_grid, kernel, scale; method = Spectral())`.
"""
struct NUFSHTFilterPlan{P, F, T<:AbstractFloat, M, SV<:AbstractVector{T}} <: CGEF.Filtering.AbstractFilterPlan
    plan::P
    filter::F
    mask::M          # Vector{T} of 0/1, or nothing when fully active (unmasked)
    renorm::Bool     # divide by the filtered mask mass: `Deformable` only, never `ZeroFill`
    scratch::SV      # length-npts scratch for `mask .* field`; unused when mask === nothing
end

function CGEF.Filtering.spectral_filter_plan(
    ::Union{CGEF.SpectralBackends.AbstractAutoSpectralBackend, CGEF.SpectralBackends.AbstractNUFSHTSpectralBackend},
    grid::FlowGeometries.Grids.UnstructuredGrid{T,G},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy = CGEF.Filtering.Deformable(),
    backend = CGEF.ComputationalBackends.AutoBackend(),
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
    filter = _CGEFTransfer(kernel, scale, FlowGeometries.Geometry.radius(FlowGeometries.Grids.grid_geometry(grid)))
    mask = all(FlowGeometries.Grids.mask(grid)) ? nothing : T.(FlowGeometries.Grids.mask(grid))
    # `ZeroFill` is already exactly `filter(mask · field)`; only `Deformable` divides by the local mass.
    renorm = mask !== nothing && mask_strategy isa CGEF.Filtering.Deformable
    scratch = zeros(T, npts)
    return NUFSHTFilterPlan{typeof(nplan), typeof(filter), T, typeof(mask), typeof(scratch)}(
        nplan, filter, mask, renorm, scratch,
    )
end

# The forward transform (points → harmonic coefficients) depends on the field alone, so a sweep runs it
# once and each scale only applies its own transfer function and evaluates back to points.
# `nusht_synthesize!` does not consume `C`, which is what lets the same coefficients serve every scale.
CGEF.Filtering.analyze_buffer(plan::NUFSHTFilterPlan, ::AbstractVector) = similar(plan.plan.C)

function CGEF.Filtering.filter_analyze!(
    Ĉ::AbstractArray, field::AbstractVector, plan::NUFSHTFilterPlan{P, F, T},
) where {P, F, T<:AbstractFloat}
    if plan.mask === nothing
        NUFSHT.nusht_type1!(Ĉ, convert(Vector{T}, field), plan.plan)
    else
        plan.scratch .= field .* plan.mask
        NUFSHT.nusht_type1!(Ĉ, plan.scratch, plan.plan)
    end
    return Ĉ
end

function CGEF.Filtering.filter_synthesize!(
    out::AbstractVector{T}, Ĉ::AbstractArray, plan::NUFSHTFilterPlan{P, F, T},
) where {P, F, T<:AbstractFloat}
    NUFSHT.nusht_synthesize!(out, Ĉ, plan.filter, plan.plan)
    plan.renorm && NUFSHT.nusht_filter_renorm!(out, plan.mask, plan.filter, plan.plan)
    return out
end

function CGEF.Filtering.filter_apply!(
    out::AbstractVector{T},
    field::AbstractVector,
    plan::NUFSHTFilterPlan{P, F, T},
) where {P, F, T<:AbstractFloat}
    if plan.mask === nothing
        NUFSHT.nusht_filter!(out, convert(Vector{T}, field), plan.filter, plan.plan)
        return out
    end
    plan.scratch .= field .* plan.mask
    NUFSHT.nusht_filter!(out, plan.scratch, plan.filter, plan.plan)
    plan.renorm && NUFSHT.nusht_filter_renorm!(out, plan.mask, plan.filter, plan.plan)
    return out
end

end # module
