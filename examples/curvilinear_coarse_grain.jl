# CurvilinearGrid demo: a sheared/rotated model-native mesh (the kind a ROMS-style ocean model
# stores at its rho-points) — no rectilinear axis anywhere. Filtering, derivatives, and Π all work
# directly off the 2D (lon, lat) coordinate arrays via a per-point footprint and weighted-least-
# squares (WLSQ) gradients, not a fast-path that assumes uniform spacing.

using Random: Random
using Statistics: Statistics
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF

Random.seed!(42)

# Build a sheared + rotated index grid: physical (lon,lat) = a linear (non-orthogonal) map of (i,j).
N = 60
dx = 2_000.0
geom = CGEF.CartesianGeometry(dx, dx)
i = collect(0.0:(N - 1))
j = collect(0.0:(N - 1))
θ = deg2rad(15.0)
shear = 0.3
lon = [dx * (ii * cos(θ) - jj * shear * sin(θ)) for ii in i, jj in j]
lat = [dx * (ii * sin(θ) + jj * (1 + shear * cos(θ))) for ii in i, jj in j]
mask = trues(N, N)
grid = CGEF.CurvilinearGrid(geom, lon, lat, mask)   # exact corner-based quad areas, auto-reconstructed

# Exact check: the (i,j) -> (lon,lat) map is linear with Jacobian determinant det = cos(θ) + shear
# (per unit dx^2), so N grid cells per axis (each of nominal width dx, including the two half-cells
# at each boundary) must tile a total area of exactly N^2 * dx^2 * (cos(θ) + shear).
true_area = N^2 * dx^2 * (cos(θ) + shear)
println("total curvilinear cell area / true area: ", round(sum(grid.areas) / true_area; sigdigits = 10))

# Synthetic non-divergent flow so Π should be small in the interior for a coarse scale.
u = randn(N, N); v = randn(N, N)

scales = collect(10e3:10e3:60e3)
result = CGEF.coarse_grain(u, v, grid; scales = scales, kernel = CGEF.TopHatKernel())

println("\nscale [km]   coarse-KE         mean|Π|")
for (k, ℓ) in enumerate(scales)
    println(
        rpad(round(ℓ / 1e3; digits = 1), 13),
        rpad(round(result.cumulative_energy[k]; sigdigits = 4), 18),
        round(Statistics.mean(abs, @view result.Π[:, :, k]); sigdigits = 4),
    )
end
