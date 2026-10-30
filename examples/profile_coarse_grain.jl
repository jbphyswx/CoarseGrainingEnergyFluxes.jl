# Vertical-profile (2.5D-per-level) demo — the literature-standard method (Aluie, Hecht & Vallis
# 2018): the existing 2D/2.5D compute_Π! is run independently at each vertical level of a 3D
# (x, y, z) array and the results are stacked into a profile, levels not interacting. The coupled
# true-3D method, which filters across levels, is examples/true_3d_coarse_grain.jl; the two answer
# different physical questions.

using Random: Random
using Statistics: Statistics
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF
using FlowGeometries: FlowGeometries as FG

Random.seed!(11)

N = 80
Nz = 6                                              # e.g. 6 levels of a multi-level model
dx = 1_000.0
geom = FG.Geometry.CartesianGeometry()
xs = 0.0:dx:(N - 1) * dx   # a range: a uniform axis, which the translation-invariant engines take
# Periodic, as the synthetic field below is; a 2D grid, z being a separate array axis.
grid = FG.Grids.StructuredGrid(geom, xs, xs; topology = (FG.Grids.Periodic(), FG.Grids.Periodic()))

# A flow whose eddy amplitude decays with z (surface-intensified, as in a typical boundary-forced
# geophysical flow).
L = N * dx
decay = [exp(-3.0 * (k - 1) / (Nz - 1)) for k in 1:Nz]
u = zeros(N, N, Nz); v = zeros(N, N, Nz)
for k in 1:Nz, (xi, x) in enumerate(xs), (yi, y) in enumerate(xs)
    u[xi, yi, k] = decay[k] * sin(2π * 3 * x / L) * cos(2π * 3 * y / L) + 0.1 * randn()
    v[xi, yi, k] = -decay[k] * cos(2π * 3 * x / L) * sin(2π * 3 * y / L) + 0.1 * randn()
end

scales = collect(5e3:5e3:30e3)
result = CGEF.coarse_grain_profile(u, v, grid; scales = scales, kernel = CGEF.TopHatKernel(),
                                   spectrum = CGEF.Diagnostics.NoSpectrum())

# The level is a trailing batch axis: Π is (x, y, scale, level), cumulative_energy (scale, level).
println("mean|Π| by level and scale (rows = level 1..$Nz, cols = scales):")
for k in 1:Nz
    row = [round(Statistics.mean(abs, @view result.Π[:, :, s, k]); sigdigits = 3) for s in eachindex(scales)]
    println("level $k: ", row)
end
println("\ncumulative_energy by level and scale:")
for k in 1:Nz
    println("level $k: ", round.(result.cumulative_energy[:, k]; sigdigits = 4))
end
