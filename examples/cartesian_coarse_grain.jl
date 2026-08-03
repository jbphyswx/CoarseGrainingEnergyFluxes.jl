# Minimal Cartesian coarse-graining demo: cross-scale kinetic-energy flux Π(x, ℓ) and the
# coarse-grained kinetic energy across scales on a synthetic two-scale 2D velocity field.
#
# See also in this directory: spherical_coarse_grain.jl (spherical + mask),
# curvilinear_coarse_grain.jl (model-native curvilinear mesh), unstructured_coarse_grain.jl
# (scattered points via k-d tree + Voronoi + spectral filtering), true_3d_coarse_grain.jl (coupled
# 3D Cartesian + spherical-volumetric flux), profile_coarse_grain.jl (2.5D-per-level vertical structure).

using Random: Random
using Statistics: Statistics
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF
using FlowGeometries: FlowGeometries as FG

Random.seed!(1234)

# 128 x 128 Cartesian patch, 1 km spacing
N = 128
dx = 1_000.0
geom = FG.Geometry.CartesianGeometry()
x = 0.0:dx:dx*(N - 1)   # a Range, not `collect`ed — keeps the fast translation-invariant footprint path
y = 0.0:dx:dx*(N - 1)
# No mask argument: the grid then stores `AllActive`, whose `isactive` folds to a constant. Passing an
# all-true `BitArray` instead means a load and a branch per cell — measured 3x on the derivative sweep.
grid = FG.Grids.StructuredGrid(geom, x, y)

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
        rpad(round(result.cumulative_energy[i]; sigdigits = 4), 18),
        round(Statistics.mean(abs, @view result.Π[:, :, i]); sigdigits = 4),
    )
end
