module CoarseGrainingEnergyFluxesFastSphericalHarmonicsExt

using FastSphericalHarmonics: FastSphericalHarmonics as FSH
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF

# Spectral filtering for UNIFORM spherical grids (a `StructuredGrid{Spherical}` whose latitudes /
# longitudes are the FastSphericalHarmonics grid). Same adapter pattern as FFTW/FINUFFT — forward
# transform, multiply by the shared transfer function, inverse transform — but the transform is the
# scalar spherical-harmonic transform and the "wavenumber" of degree l is k_l = √(l(l+1)) / R (the
# Laplace–Beltrami eigenvalue), so a coefficient of degree l is multiplied by Ĝ(k_l, ℓ). The l = 0
# coefficient (the mean) is preserved since Ĝ(0) = 1.
#
# The FSH grid has N points in colatitude θ and M = 2N−1 in longitude φ, with the field sampled as
# F[θ, φ] (an N×M array). CGEF stores fields as [lon, lat] (M×N), so we transpose in and out. The grid
# must be built on the FSH points (`FastSphericalHarmonics.sph_points(N)`): θ_j = π(j−½)/N,
# φ_k = 2π(k−1)/M, i.e. lat = π/2 − θ. Land masks are not applied (a global transform), like FFTW.

"""
    SHTFilterPlan

Cached spherical-harmonic filter plan: the per-coefficient transfer multiplier `Ĝ(k_l, ℓ)` laid out on
the FSH coefficient array (depends only on degree l), a reusable `N × M` scratch buffer for the
[lon,lat] <-> FSH [θ,φ] transpose (filled via `permutedims!`, not a fresh `permutedims` allocation
every call), and a `FastSphericalHarmonics.SphPlanCache` — WITHOUT an explicit cache, `sph_transform`/
`sph_evaluate` each build a fresh FFT plan internally on every call (its own internal `Dict`s only get
populated, and thus reused, when the SAME cache object is passed repeatedly). Built by
`plan_filter(spherical_structured_grid, kernel, scale; method = Spectral())`.
"""
struct SHTFilterPlan{T<:AbstractFloat, A<:AbstractMatrix{T}, S<:AbstractMatrix{T}} <: CGEF.Filtering.AbstractFilterPlan
    mult::A   # N × M multiplier on the spherical-harmonic coefficients
    scratch::S                  # N × M transpose/transform scratch buffer (always a concrete Array in
                                 # practice, from `zeros(T,N,M)` below — FSH's sph_transform!/sph_evaluate!
                                 # require `Array{T,2}` — but the field itself isn't hardcoded to it)
    cache::FSH.SphPlanCache{T}  # cached FFT plans, reused across calls
    N::Int
    M::Int
end

function CGEF.Filtering.spectral_filter_plan(
    grid::CGEF.StructuredGrid{G,T},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy = CGEF.Filtering.Deformable(),
    backend = CGEF.Backends.AutoBackend(),
) where {T<:AbstractFloat, G<:CGEF.SphericalGeometry{T}}
    M, N = size(grid.mask)   # CGEF layout is [lon, lat] = [M, N]
    M == 2N - 1 || throw(ArgumentError(
        "Spherical-harmonic filtering needs a FastSphericalHarmonics grid with M = 2N-1 longitudes " *
        "per N latitudes (got N=$N lat, M=$M lon); build the grid on `sph_points(N)`.",
    ))
    # Shape alone doesn't prove the grid sits on the actual FSH quadrature nodes — a shape-correct but
    # wrong-node grid would silently produce a meaningless transform. Check the node values themselves.
    Θ, Φ = FSH.sph_points(N)
    isapprox(grid.lat, T(π) / 2 .- Θ; atol = 10 * eps(T)) || throw(ArgumentError(
        "grid.lat does not match the FastSphericalHarmonics quadrature nodes for N=$N " *
        "(expected θ = π(j-½)/N via `sph_points(N)`, lat = π/2 - θ); build the grid on `sph_points(N)`.",
    ))
    isapprox(grid.lon, T.(Φ); atol = 10 * eps(T)) || throw(ArgumentError(
        "grid.lon does not match the FastSphericalHarmonics quadrature nodes for M=$M " *
        "(expected φ = 2π(k-1)/M via `sph_points(N)`); build the grid on `sph_points(N)`.",
    ))
    all(grid.mask) || throw(ArgumentError(
        "Spherical-harmonic filtering does not support a partial mask (a global transform); use method = DirectSum().",
    ))
    R = grid.geometry.R

    # Mirror FastSphericalHarmonics' own coefficient-iteration (see `sph_laplace!`): the packed layout
    # stores degrees up to lmax + mmax for high |m|.
    lmax = N - 1
    mmax = M ÷ 2
    mult = ones(T, N, M)
    for l in 0:(lmax + mmax), m in (-l):l
        if l - lmax <= abs(m) <= mmax
            k_l = sqrt(T(l * (l + 1))) / R
            mult[FSH.sph_mode(l, m)] = CGEF.Kernels.spectral_transfer(kernel, k_l, scale)
        end
    end
    scratch = zeros(T, N, M)
    cache = FSH.SphPlanCache{T}()
    return SHTFilterPlan(mult, scratch, cache, N, M)
end

function CGEF.Filtering.filter_apply!(
    out::AbstractMatrix{T},
    field::AbstractMatrix{T},
    plan::SHTFilterPlan{T},
) where {T<:AbstractFloat}
    permutedims!(plan.scratch, field, (2, 1))           # [lon, lat] (M×N) → FSH [θ, φ] (N×M), in place
    FSH.sph_transform!(plan.scratch; cache = plan.cache)   # in place: scratch now holds coefficients
    plan.scratch .*= plan.mult                          # Ĝ(k_l, ℓ) per coefficient
    FSH.sph_evaluate!(plan.scratch; cache = plan.cache)    # in place: scratch now holds point values
    permutedims!(out, plan.scratch, (2, 1))             # back to [lon, lat]
    return out
end

end # module
