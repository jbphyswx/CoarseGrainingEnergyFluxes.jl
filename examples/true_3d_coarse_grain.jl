# True 3D volumetric flux demo (Cartesian and spherical). Distinct from the depth-profile method
# (examples/depth_profile.jl): here the filter kernel genuinely blends all three directions at
# once, and Π is the full nine-component (six independent) strain/stress contraction with real
# vertical derivatives — the homogeneous/isotropic-turbulence regime (Rayleigh-Taylor, boundary
# layers), not the thin-layer/quasi-geostrophic depth-stacking regime.

using Random: Random
using Statistics: Statistics
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF

Random.seed!(3)

# ── Cartesian: uniform (x, y, z) box ──────────────────────────────────────────
N = 24
dx = 500.0
geom = CGEF.CartesianGeometry(dx, dx, dx)
x = collect(0.0:dx:(N - 1) * dx)
mask = trues(N, N, N)
grid = CGEF.StructuredGrid(geom, x, x, x, mask)

u = randn(N, N, N); v = randn(N, N, N); w = randn(N, N, N)
scales = collect(2_000.0:2_000.0:8_000.0)
result = CGEF.coarse_grain(u, v, w, grid; scales = scales, kernel = CGEF.TopHatKernel())

println("True 3D Cartesian volumetric flux")
println("scale [m]   coarse-KE         mean|Π|")
for (k, ℓ) in enumerate(scales)
    println(
        rpad(round(ℓ; digits = 1), 11),
        rpad(round(result.cumulative_energy[k]; sigdigits = 4), 18),
        round(Statistics.mean(abs, @view result.Π[:, :, :, k]); sigdigits = 4),
    )
end

# ── Spherical volumetric shell: (lon, lat, radius) ────────────────────────────
# A small regional patch at mid-latitude with roughly ISOTROPIC grid spacing in all three
# directions (~30 km horizontal and vertical) — the true-3D method assumes one filter scale ℓ
# applies equally in all three directions, so (unlike the real ocean, where horizontal scales are
# orders of magnitude larger than the depth range) the demo needs comparable spacing to be
# meaningful; a fully global, ocean-depth-scale patch would put every scale below inside a single
# horizontal grid cell and show no scale dependence at all.
R = 6.371e6
sgeom = CGEF.SphericalGeometry(R)
lon = deg2rad.(collect(0.0:0.3:3.3))                # ~28 km spacing at this latitude
lat = deg2rad.(collect(30.0:0.3:33.0))               # ~33 km spacing
r = collect((R - 200e3):20e3:R)                     # 11 levels, 20 km spacing, 200 km total depth
smask = trues(length(lon), length(lat), length(r))
sgrid = CGEF.StructuredGrid(sgeom, lon, lat, r, smask)

Nlon, Nlat, Nz = length(lon), length(lat), length(r)
su = randn(Nlon, Nlat, Nz); sv = randn(Nlon, Nlat, Nz); sw = randn(Nlon, Nlat, Nz)
sscales = collect(40e3:40e3:160e3)
sresult = CGEF.coarse_grain(su, sv, sw, sgrid; scales = sscales, kernel = CGEF.TopHatKernel())

println("\nTrue 3D spherical-volumetric flux")
println("scale [km]   coarse-KE         mean|Π|")
for (k, ℓ) in enumerate(sscales)
    println(
        rpad(round(ℓ / 1e3; digits = 1), 13),
        rpad(round(sresult.cumulative_energy[k]; sigdigits = 4), 18),
        round(Statistics.mean(abs, @view sresult.Π[:, :, :, k]); sigdigits = 4),
    )
end
