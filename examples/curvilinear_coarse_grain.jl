# CurvilinearGrid demo: a sheared/rotated model-native mesh (the kind a structured-grid ocean or
# atmosphere model stores at its cell centers) — no rectilinear axis anywhere. Filtering,
# derivatives, and Π all work directly off the 2D (x, y) coordinate arrays via a per-point
# footprint and weighted-least-squares (WLSQ) gradients, not a fast-path that assumes uniform spacing.

using Random: Random
using Statistics: Statistics
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF
using FlowGeometries: FlowGeometries as FG

Random.seed!(42)

# Build a sheared + rotated index grid: physical (x,y) = a linear (non-orthogonal) map of (i,j).
N = 60
dx = 2_000.0
geom = FG.Geometry.CartesianGeometry()
i = collect(0.0:(N - 1))
j = collect(0.0:(N - 1))
θ = deg2rad(15.0)
shear = 0.3
x = [dx * (ii * cos(θ) - jj * shear * sin(θ)) for ii in i, jj in j]
y = [dx * (ii * sin(θ) + jj * (1 + shear * cos(θ))) for ii in i, jj in j]
grid = FG.Grids.CurvilinearGrid(geom, x, y)   # exact corner-based quad areas, auto-reconstructed

# Exact check: the (i,j) -> (x,y) map is linear with Jacobian determinant det = cos(θ) + shear
# (per unit dx^2), so N grid cells per axis (each of nominal width dx, including the two half-cells
# at each boundary) must tile a total area of exactly N^2 * dx^2 * (cos(θ) + shear).
true_area = N^2 * dx^2 * (cos(θ) + shear)
println("total curvilinear cell area / true area: ", round(sum(FG.Grids.measure(grid)) / true_area; sigdigits = 10))

# Synthetic non-divergent flow so Π should be small in the interior for a coarse scale.
u = randn(N, N); v = randn(N, N)

scales = collect(10e3:10e3:60e3)
result = CGEF.coarse_grain(u, v, grid; scales = scales, kernel = CGEF.TopHatKernel(),
                           spectrum = false)

println("\nscale [km]   coarse-KE         mean|Π|")
for (k, ℓ) in enumerate(scales)
    println(
        rpad(round(ℓ / 1e3; digits = 1), 13),
        rpad(round(result.cumulative_energy[k]; sigdigits = 4), 18),
        round(Statistics.mean(abs, @view result.Π[:, :, k]); sigdigits = 4),
    )
end
