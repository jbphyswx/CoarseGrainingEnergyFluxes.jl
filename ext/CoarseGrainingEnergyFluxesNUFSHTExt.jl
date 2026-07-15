module CoarseGrainingEnergyFluxesNUFSHTExt

using NUFSHT: NUFSHT
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF

# Spectral filtering for SCATTERED spherical data (an `UnstructuredGrid{Spherical}`), delegated to
# NUFSHT.jl (Non-Uniform Fast Spherical Harmonic Transforms). NUFSHT already implements the full
# scattered-point filter — type-1 (analysis) → per-degree transfer multiply → type-2 (synthesis), with
# optional mask renormalization — so this extension is a thin adapter: it builds the NUSHT plan
# from the grid's scattered (colatitude, longitude) nodes and drives `nusht_filter!`.
#
# To keep the Gaussian convention IDENTICAL to the other three spectral backends (FFTW / FINUFFT /
# FastSphericalHarmonics), we do NOT use NUFSHT's own GaussianTransfer; instead a tiny adapter feeds
# NUFSHT the shared `CGEF.Kernels.spectral_transfer(kernel, k_l, ℓ)` per degree, with k_l = √(l(l+1))/R.
#
# Conditioning note: `nusht_filter!` uses the adjoint analysis, which is exact on a Clenshaw–Curtis
# grid and well-behaved for quasi-uniform scattered sampling; very irregular sampling is an
# ill-conditioned inverse (see NUFSHT's `nusht_solve!` for CG inversion).

# Adapter exposing CGEF's shared transfer function to NUFSHT's per-degree `kernel_transfer`.
struct _CGEFTransfer{K<:CGEF.Kernels.AbstractFilterKernel, T<:AbstractFloat} <: NUFSHT.AbstractSpectralTransfer
    kernel::K
    scale::T
    R::T
end
@inline NUFSHT.kernel_transfer(t::_CGEFTransfer, ℓ) =
    CGEF.Kernels.spectral_transfer(t.kernel, sqrt(oftype(t.R, ℓ * (ℓ + 1))) / t.R, t.scale)

"""
    NUFSHTFilterPlan

Cached scattered-spherical filter plan: the NUSHT plan over the grid's nodes, the CGEF transfer
adapter, and the (optional) mask for renormalization. Built by
`plan_filter(scattered_spherical_grid, kernel, scale; method = Spectral())`.
"""
struct NUFSHTFilterPlan{P, F, T<:AbstractFloat, M} <: CGEF.Filtering.AbstractFilterPlan
    plan::P
    filter::F
    mask::M        # Vector{T} of 0/1, or nothing when fully wet
end

function CGEF.Filtering.spectral_filter_plan(
    grid::CGEF.UnstructuredGrid{T,G},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy = CGEF.Filtering.Deformable(),
    backend = CGEF.Backends.AutoBackend(),
) where {T<:AbstractFloat, G<:CGEF.SphericalGeometry{T}}
    npts = length(grid.lon)
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
    θ = T(π) / 2 .- grid.lat        # colatitude from latitude
    φ = grid.lon
    nplan = NUFSHT.make_plan(collect(T, θ), collect(T, φ), lmax; T = T)
    filter = _CGEFTransfer(kernel, scale, grid.geometry.R)
    mask = all(grid.mask) ? nothing : T.(grid.mask)
    return NUFSHTFilterPlan{typeof(nplan), typeof(filter), T, typeof(mask)}(nplan, filter, mask)
end

function CGEF.Filtering.filter_apply!(
    out::AbstractVector{T},
    field::AbstractVector,
    plan::NUFSHTFilterPlan{P, F, T},
) where {P, F, T<:AbstractFloat}
    f = convert(Vector{T}, field)
    if plan.mask === nothing
        NUFSHT.nusht_filter!(out, f, plan.filter, plan.plan)
    else
        # Zero land, filter, then divide by the filtered mass over wet points (deformable masking).
        NUFSHT.nusht_filter!(out, f .* plan.mask, plan.filter, plan.plan)
        NUFSHT.nusht_filter_renorm!(out, plan.mask, plan.filter, plan.plan)
    end
    return out
end

end # module
