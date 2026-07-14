# Depth-profile (2.5D-per-level) vertical-structure demo — the literature-standard method (Aluie,
# Hecht & Vallis 2018): the existing 2D/2.5D compute_Π! is run independently at each depth level of
# a 3D (lon, lat, depth) array and the results are stacked into a profile. This is a convenience
# wrapper over an already-correct 2D method, NOT the coupled true-3D method
# (examples/true_3d_coarse_grain.jl) — the two answer different physical questions and should never
# be conflated.

using Random: Random
using Statistics: Statistics
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF

Random.seed!(11)

N = 80
Nz = 6                                              # e.g. 6 depth levels of a multi-level ocean model
dx = 1_000.0
geom = CGEF.CartesianGeometry(dx, dx)
xs = collect(0.0:dx:(N - 1) * dx)
# periodic: the synthetic field below is exactly periodic, so this avoids domain-edge
# footprint-truncation artifacts from dominating the flux (a 2D grid — depth is a separate array axis)
grid = CGEF.StructuredGrid(geom, xs, xs, trues(N, N); periodic = (true, true))

# A flow whose eddy amplitude decays with depth (surface-intensified, as in the real ocean).
L = N * dx
decay = [exp(-3.0 * (k - 1) / (Nz - 1)) for k in 1:Nz]
u = zeros(N, N, Nz); v = zeros(N, N, Nz)
for k in 1:Nz, (xi, x) in enumerate(xs), (yi, y) in enumerate(xs)
    u[xi, yi, k] = decay[k] * sin(2π * 3 * x / L) * cos(2π * 3 * y / L) + 0.1 * randn()
    v[xi, yi, k] = -decay[k] * cos(2π * 3 * x / L) * sin(2π * 3 * y / L) + 0.1 * randn()
end

scales = collect(5e3:5e3:30e3)
result = CGEF.coarse_grain_profile(u, v, grid; scales = scales, kernel = CGEF.TopHatKernel())

println("mean|Π| by depth level and scale (rows = level 1..$Nz, cols = scales):")
for k in 1:Nz
    row = [round(Statistics.mean(abs, @view result.Π[:, :, k, s]); sigdigits = 3) for s in eachindex(scales)]
    println("level $k: ", row)
end
println("\ncumulative_energy by depth level and scale:")
for k in 1:Nz
    println("level $k: ", round.(result.cumulative_energy[k, :]; sigdigits = 4))
end
