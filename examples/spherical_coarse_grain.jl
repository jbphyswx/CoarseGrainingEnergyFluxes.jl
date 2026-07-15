# Spherical regional coarse-graining demo: cross-scale kinetic-energy flux Π(x, ℓ) and the filtering
# spectrum on a synthetic non-divergent velocity field over a lon/lat patch with a land mask.
#
# The velocity is built as u = ∇⊥ψ (a streamfunction), so it is non-divergent — the regime in which
# the planetary-Cartesian filtering of Aluie (2019) / Storer et al. (2022) is exact. Land cells are
# masked and handled by the default deformable kernel renormalization.

using Random: Random
using Statistics: Statistics
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF

Random.seed!(2024)

# Regional spherical patch: 20°×20° box near the Gulf Stream latitude, 0.1° resolution.
R = 6.371e6
geom = CGEF.SphericalGeometry(R)
lon = deg2rad.(collect(-70.0:0.1:-50.0))
lat = deg2rad.(collect(30.0:0.1:50.0))
Nlon, Nlat = length(lon), length(lat)

# A simple round "island" land mask in the middle of the domain.
mask = trues(Nlon, Nlat)
ci, cj = Nlon ÷ 2, Nlat ÷ 2
for j in 1:Nlat, i in 1:Nlon
    if (i - ci)^2 + (j - cj)^2 <= 8^2
        mask[i, j] = false
    end
end
grid = CGEF.StructuredGrid(geom, lon, lat, mask)

# Non-divergent velocity from a two-scale streamfunction ψ: u = (1/R) ∂ψ/∂φ, v = -(1/(R cosφ)) ∂ψ/∂λ
# (here we just sample the analytic derivatives of ψ = sin(kλ)cos(kφ) for two wavenumbers).
ψamp(k, λ, φ) = sin(k * λ) * cos(k * φ)
u = zeros(Nlon, Nlat); v = zeros(Nlon, Nlat)
for j in 1:Nlat, i in 1:Nlon
    λ, φ = lon[i], lat[j]
    cφ = cos(φ)
    for (k, a) in ((6.0, 1.0), (40.0, 0.25))   # large eddy + small-scale wiggle
        u[i, j] += a * (-k * sin(k * λ) * sin(k * φ)) / R
        v[i, j] += a * (-k * cos(k * λ) * cos(k * φ)) / (R * cφ)
    end
end

scales = collect(20e3:20e3:200e3)              # 20–200 km
result = CGEF.coarse_grain(u, v, grid; scales = scales, kernel = CGEF.TopHatKernel())

println("scale [km]   cumulative-KE     mean|Π| [W/m³]")
for (i, ℓ) in enumerate(scales)
    println(
        rpad(round(ℓ / 1e3; digits = 1), 13),
        rpad(round(result.cumulative_energy[i]; sigdigits = 4), 18),
        round(Statistics.mean(abs, @view result.Π[:, :, i]); sigdigits = 4),
    )
end

# Filtering spectral density Ẽ(k_ℓ) (Sadek & Aluie 2018, Eq. 14), with k_ℓ = L/ℓ (here L = 1).
println("\nfiltering wavenumber k_ℓ [1/m] vs density Ẽ:")
for (kℓ, Ẽ) in zip(result.wavenumber, result.filtering_spectrum)
    println("  ", rpad(round(kℓ; sigdigits = 3), 12), round(Ẽ; sigdigits = 4))
end
