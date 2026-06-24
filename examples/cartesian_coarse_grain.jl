# Minimal Cartesian coarse-graining demo: cross-scale kinetic-energy flux Π(x, ℓ) and the
# coarse-grained kinetic energy across scales on a synthetic two-scale 2D velocity field.
#
# NOTE: this is a smoke/placeholder example to keep the package runnable end-to-end. A fuller set
# of scientifically-motivated examples (forced-2D-turbulence forward/inverse cascade, 3D isotropic
# turbulence, spherical ocean SSH submesoscale flux with land masks, the corrected filtering
# spectrum recovering a known slope, batched/threaded/GPU/MPI demos) lands in the docs/examples
# phase of the overhaul.

using Random: Random
using Statistics: Statistics
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF

Random.seed!(1234)

# 128 x 128 Cartesian patch, 1 km spacing
N = 128
dx = 1_000.0
geom = CGEF.CartesianGeometry(dx, dx)
x = collect(0.0:dx:dx*(N - 1))
y = collect(0.0:dx:dx*(N - 1))
mask = trues(N, N)
grid = CGEF.StructuredGrid(geom, x, y, mask)

# Synthetic field: a large coherent eddy plus small-scale fluctuations
L = dx * N
u = [sin(2π * 2 * xi / L) * cos(2π * 2 * yi / L) for xi in x, yi in y] .+ 0.2 .* randn(N, N)
v = [-cos(2π * 2 * xi / L) * sin(2π * 2 * yi / L) for xi in x, yi in y] .+ 0.2 .* randn(N, N)

scales = collect(5_000.0:5_000.0:40_000.0)
result = CGEF.coarse_grain(u, v, grid; scales = scales, kernel = CGEF.TopHatKernel())

println("scale [km]   coarse-KE         mean|Π|")
for (i, ℓ) in enumerate(scales)
    println(
        rpad(round(ℓ / 1e3; digits = 1), 13),
        rpad(round(result.spectrum[i]; sigdigits = 4), 18),
        round(Statistics.mean(abs, result.Π[i]); sigdigits = 4),
    )
end
