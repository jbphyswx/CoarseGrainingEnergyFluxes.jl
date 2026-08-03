module CoarseGrainingEnergyFluxesFastSphericalHarmonicsExt

using FastSphericalHarmonics: FastSphericalHarmonics as FSH
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF
using FlowGeometries: FlowGeometries

# Spectral filtering for uniform spherical grids, via the scalar spherical-harmonic transform. The
# wavenumber of degree `l` is the Laplace–Beltrami eigenvalue `k_l = √(l(l+1))/R`, so a degree-`l`
# coefficient is scaled by `Ĝ(k_l, ℓ)`; `Ĝ(0) = 1` preserves the mean.
#
# The FSH grid is `N` colatitudes × `M = 2N−1` longitudes sampled as `F[θ, φ]`, where this package
# stores fields as `[lon, lat]` — hence the transpose in and out. The grid must be built on the FSH
# points (`FastSphericalHarmonics.sph_points(N)`): `θ_j = π(j−½)/N`, `φ_k = 2π(k−1)/M`.
#
# Masking follows the same normalized-convolution identity as the other spectral backends, with the
# `Deformable` denominator computed once at plan-build time.

"""
    SHTFilterPlan

Cached spherical-harmonic filter plan: the per-coefficient transfer multiplier `Ĝ(k_l, ℓ)` laid out on
the FSH coefficient array (depends only on degree l), a reusable `N × M` scratch buffer for the
[lon,lat] <-> FSH [θ,φ] transpose (filled via `permutedims!`, not a fresh `permutedims` allocation
every call), a `FastSphericalHarmonics.SphPlanCache` — WITHOUT an explicit cache, `sph_transform`/
`sph_evaluate` each build a fresh FFT plan internally on every call (its own internal `Dict`s only get
populated, and thus reused, when the SAME cache object is passed repeatedly) — and, for a masked grid,
the mask itself, a scratch buffer for `mask · field`, and (for `Deformable`) the precomputed inverse
local-mass renormalization. Built by
`plan_filter(spherical_structured_grid, kernel, scale; method = Spectral())`.
"""
struct SHTFilterPlan{
    T<:AbstractFloat, A<:AbstractMatrix{T}, S<:AbstractMatrix{T}, MK, MV<:AbstractMatrix{T}, R,
} <: CGEF.Filtering.AbstractFilterPlan
    mult::A   # N × M multiplier on the spherical-harmonic coefficients
    scratch::S                  # N × M transpose/transform scratch buffer (always a concrete Array in
                                 # practice, from `zeros(T,N,M)` below — FSH's sph_transform!/sph_evaluate!
                                 # require `Array{T,2}` — but the field itself isn't hardcoded to it)
    cache::FSH.SphPlanCache{T}  # cached FFT plans, reused across calls
    N::Int
    M::Int
    mask::MK            # M × N [lon,lat] mask, or nothing when fully active
    masked_input::MV    # M × N scratch for `mask .* field`; unused when mask === nothing
    invrenorm::R         # precomputed 1/filter(mask) for Deformable, or nothing (ZeroFill / fully active)
end

function CGEF.Filtering.spectral_filter_plan(
    ::Union{CGEF.SpectralBackends.AbstractAutoSpectralBackend, CGEF.SpectralBackends.AbstractFSHTSpectralBackend},
    grid::FlowGeometries.Grids.StructuredGrid{G,T},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy = CGEF.Filtering.Deformable(),
    backend = CGEF.ComputationalBackends.AutoBackend(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.SphericalGeometry{T}}
    M, N = size(FlowGeometries.Grids.mask(grid))   # CGEF layout is [x, y] = [longitude, latitude] = [M, N]
    M == 2N - 1 || throw(ArgumentError(
        "Spherical-harmonic filtering needs a FastSphericalHarmonics grid with M = 2N-1 longitudes " *
        "per N latitudes (got N=$N lat, M=$M lon); build the grid on `sph_points(N)`.",
    ))
    # Shape alone doesn't prove the grid sits on the actual FSH quadrature nodes — a shape-correct but
    # wrong-node grid would silently produce a meaningless transform. Check the node values themselves.
    Θ, Φ = FSH.sph_points(N)
    isapprox(FlowGeometries.Grids.coordinates(grid, 2), T(π) / 2 .- Θ; atol = 10 * eps(T)) || throw(ArgumentError(
        "The grid's latitude coordinates do not match the FastSphericalHarmonics quadrature nodes for N=$N " *
        "(expected θ = π(j-½)/N via `sph_points(N)`, lat = π/2 - θ); build the grid on `sph_points(N)`.",
    ))
    isapprox(FlowGeometries.Grids.coordinates(grid, 1), T.(Φ); atol = 10 * eps(T)) || throw(ArgumentError(
        "The grid's longitude coordinates do not match the FastSphericalHarmonics quadrature nodes for M=$M " *
        "(expected φ = 2π(k-1)/M via `sph_points(N)`); build the grid on `sph_points(N)`.",
    ))
    R = FlowGeometries.Geometry.radius(FlowGeometries.Grids.grid_geometry(grid))

    # Mirror FastSphericalHarmonics' own coefficient-iteration (see `sph_laplace!`): the packed layout
    # stores degrees up to lmax + mmax for high |m|. The transfer value depends only on l, not m, so
    # compute it once per degree (not once per (l,m) pair, ~2l+1 times more calls) — negligible for
    # Gaussian/SharpSpectral but genuinely wasteful for TopHatKernel's Legendre-recurrence evaluation.
    lmax = N - 1
    mmax = M ÷ 2
    lmax_full = lmax + mmax
    transfer_by_l = [CGEF.Kernels.spectral_transfer_degree(kernel, l, scale, R) for l in 0:lmax_full]
    mult = ones(T, N, M)
    for l in 0:lmax_full, m in (-l):l
        if l - lmax <= abs(m) <= mmax
            mult[FSH.sph_mode(l, m)] = transfer_by_l[l+1]
        end
    end
    scratch = zeros(T, N, M)
    cache = FSH.SphPlanCache{T}()
    masked_input = zeros(T, M, N)   # [lon,lat] layout, matching `field`

    fully_active = all(FlowGeometries.Grids.mask(grid))
    if fully_active
        return SHTFilterPlan(mult, scratch, cache, N, M, nothing, masked_input, nothing)
    end

    mask = FlowGeometries.Grids.mask(grid)
    invrenorm = if mask_strategy isa CGEF.Filtering.Deformable
        # Local kernel mass over active points, `filter(mask)` — the SAME transform/multiply/inverse
        # pipeline, applied to the mask itself once, since the mask never changes across
        # `filter_apply!` calls.
        masked_input .= mask                         # [lon,lat] (M×N), reusing the apply-path scratch
        permutedims!(scratch, masked_input, (2, 1))  # → FSH [θ,φ] (N×M)
        FSH.sph_transform!(scratch; cache = cache)
        scratch .*= mult
        FSH.sph_evaluate!(scratch; cache = cache)
        renorm = zeros(T, M, N)
        permutedims!(renorm, scratch, (2, 1))        # back to [lon,lat]
        threshold = T(0.01)
        ir = similar(renorm)
        @. ir = ifelse(abs(renorm) >= threshold, one(T) / renorm, zero(T))
        ir
    else
        nothing   # ZeroFill: already exactly `filter(mask .* field)`, no renormalization
    end
    return SHTFilterPlan(mult, scratch, cache, N, M, mask, masked_input, invrenorm)
end

function CGEF.Filtering.filter_apply!(
    out::AbstractMatrix{T},
    field::AbstractMatrix{T},
    plan::SHTFilterPlan{T},
) where {T<:AbstractFloat}
    if plan.mask === nothing
        permutedims!(plan.scratch, field, (2, 1))       # [lon, lat] (M×N) → FSH [θ, φ] (N×M), in place
    else
        @. plan.masked_input = plan.mask * field
        permutedims!(plan.scratch, plan.masked_input, (2, 1))
    end
    FSH.sph_transform!(plan.scratch; cache = plan.cache)   # in place: scratch now holds coefficients
    plan.scratch .*= plan.mult                          # Ĝ(k_l, ℓ) per coefficient
    FSH.sph_evaluate!(plan.scratch; cache = plan.cache)    # in place: scratch now holds point values
    permutedims!(out, plan.scratch, (2, 1))             # back to [lon, lat]
    plan.invrenorm === nothing || (out .*= plan.invrenorm)
    return out
end

end # module
