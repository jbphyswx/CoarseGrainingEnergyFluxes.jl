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
the FSH coefficient array (depends only on degree l). Built by
`plan_filter(spherical_structured_grid, kernel, scale; method = Spectral())`.
"""
struct SHTFilterPlan{T<:AbstractFloat, A<:AbstractMatrix{T}} <: CGEF.Filtering.AbstractFilterPlan
    mult::A   # N × M multiplier on the spherical-harmonic coefficients
    N::Int
    M::Int
end

function CGEF.Filtering.spectral_filter_plan(
    grid::CGEF.StructuredGrid{G,T},
    kernel::CGEF.AbstractFilterKernel,
    scale::T;
    mask_strategy = CGEF.Deformable(),
    backend = CGEF.AutoBackend(),
) where {T<:AbstractFloat, G<:CGEF.SphericalGeometry{T}}
    M, N = size(grid.mask)   # CGEF layout is [lon, lat] = [M, N]
    M == 2N - 1 || throw(ArgumentError(
        "Spherical-harmonic filtering needs a FastSphericalHarmonics grid with M = 2N-1 longitudes " *
        "per N latitudes (got N=$N lat, M=$M lon); build the grid on `sph_points(N)`.",
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
            mult[FSH.sph_mode(l, m)] = CGEF.spectral_transfer(kernel, k_l, scale)
        end
    end
    return SHTFilterPlan{T, typeof(mult)}(mult, N, M)
end

function CGEF.Filtering.filter_apply!(
    out::AbstractMatrix{T},
    field::AbstractMatrix{T},
    plan::SHTFilterPlan{T},
) where {T<:AbstractFloat}
    F = permutedims(field)              # [lon, lat] (M×N) → FSH [θ, φ] (N×M)
    C = FSH.sph_transform(F)
    C .*= plan.mult                     # Ĝ(k_l, ℓ) per coefficient
    G = FSH.sph_evaluate(C)
    out .= permutedims(G)               # back to [lon, lat]
    return out
end

end # module
