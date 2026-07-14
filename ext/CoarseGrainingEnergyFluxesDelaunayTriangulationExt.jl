module CoarseGrainingEnergyFluxesDelaunayTriangulationExt

using DelaunayTriangulation: DelaunayTriangulation as DT
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF

# Exact per-node Cartesian Voronoi-cell area for `UnstructuredGrid` (overrides the throwing fallback
# in `Grids._voronoi_areas` for `CartesianGeometry`) — the planar Voronoi tessellation dual to the
# Delaunay triangulation of the node coordinates, clipped to the point set's own convex hull (so
# boundary-node cells are the true, finite clipped polygons rather than unbounded regions). On a
# regular Cartesian lattice this reduces EXACTLY to the true grid-cell area at every INTERIOR node
# (verified in the test suite); the domain-total sum is then exactly the convex-hull area of the
# point set (a genuinely independently-computable invariant for a lattice: `(Nx-1)(Ny-1)*dx*dy`).
function CGEF.Grids._voronoi_areas(
    ::CGEF.CartesianGeometry{T}, lon::AbstractVector{T}, lat::AbstractVector{T},
) where {T<:AbstractFloat}
    N = length(lon)
    pts = [(Float64(lon[i]), Float64(lat[i])) for i in 1:N]
    tri = DT.triangulate(pts)
    vorn = DT.voronoi(tri; clip = true)
    areas = Vector{T}(undef, N)
    for i in DT.each_polygon_index(vorn)
        areas[i] = T(DT.get_area(vorn, i))
    end
    return areas
end

end # module
