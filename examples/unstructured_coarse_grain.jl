# UnstructuredGrid demo: genuinely scattered observations (e.g. moorings, drifters, along-track
# altimetry), where no rectilinear or curvilinear grid structure exists at all. k-d tree neighbor
# search (NearestNeighbors) and exact Voronoi cell areas (DelaunayTriangulation)
# are built at construction time; ddx!/ddy! use weighted-least-squares gradients over that
# adjacency; filtering is necessarily spectral (FINUFFT) since a scattered point cloud has no
# translation-invariant real-space footprint.

using Random: Random
using Statistics: Statistics
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF
using NearestNeighbors: NearestNeighbors            # enables k-d tree neighbor search
using DelaunayTriangulation: DelaunayTriangulation   # enables exact planar Voronoi cell areas
using FINUFFT: FINUFFT                              # enables scattered-Cartesian spectral filtering

Random.seed!(7)

npts = 2_000
L = 100_000.0                                       # 100 km x 100 km domain
geom = CGEF.CartesianGeometry(1.0, 1.0)             # placeholder: UnstructuredGrid has no fixed spacing
lon = L .* rand(npts)
lat = L .* rand(npts)
mask = trues(npts)
grid = CGEF.UnstructuredGrid(geom, lon, lat, mask; k = 8)   # k-nearest adjacency + auto Voronoi areas

# The Voronoi tessellation is clipped to the CONVEX HULL of the sample, not the full [0,L]^2 square —
# for 2000 uniform random points that hull is a few tenths of a percent smaller than L^2 (it can't
# quite reach the corners), so the areas should sum to the hull area exactly, not L^2 exactly.
println("total Voronoi cell area / L^2 (expect slightly < 1 — hull misses the domain corners): ",
    round(sum(grid.areas) / L^2; sigdigits = 6))

# Synthetic two-scale non-divergent flow sampled at the scattered points.
ψamp(k, x, y) = sin(2π * k * x / L) * cos(2π * k * y / L)
u = zeros(npts); v = zeros(npts)
for (k, a) in ((2.0, 1.0), (14.0, 0.3))
    for q in 1:npts
        x, y = lon[q], lat[q]
        u[q] += a * (-2π * k / L) * sin(2π * k * x / L) * sin(2π * k * y / L)
        v[q] += a * (-2π * k / L) * cos(2π * k * x / L) * cos(2π * k * y / L)
    end
end

scales = collect(5e3:5e3:30e3)
result = CGEF.coarse_grain(u, v, grid; scales = scales, kernel = CGEF.GaussianKernel())

println("\nscale [km]   coarse-KE         mean|Π|")
for (k, ℓ) in enumerate(scales)
    println(
        rpad(round(ℓ / 1e3; digits = 1), 13),
        rpad(round(result.cumulative_energy[k]; sigdigits = 4), 18),
        round(Statistics.mean(abs, @view result.Π[:, k]); sigdigits = 4),
    )
end
