module CoarseGrainingEnergyFluxesQuickhullExt

using Quickhull: Quickhull as QH
using LinearAlgebra: LinearAlgebra as LA
using StaticArrays: StaticArrays as SA
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF

# Exact per-node spherical Voronoi-cell area for `UnstructuredGrid` (overrides the throwing fallback
# in `Grids._voronoi_areas` for `SphericalGeometry`): "project to unit sphere -> 3D convex hull ->
# Voronoi cell areas from the dual."
#
# Points already lying on a sphere have a special property: their plain (unlifted) 3D convex hull IS
# the spherical Delaunay triangulation (no paraboloid-lift trick is needed, unlike planar Delaunay —
# the sphere itself is the lifting surface), so `Quickhull.quickhull` on the unit-sphere embedding
# gives exactly the triangulation whose dual is the spherical Voronoi diagram.
#
# For each triangle facet (a, b, c), its dual Voronoi vertex is the triangle's SPHERICAL circumcenter
# — the point on the sphere equidistant (along great circles) from a, b, c. This is exactly the unit
# normal of the plane through a, b, c (oriented outward): every point of that plane is equidistant
# from the plane itself, but that's only relevant here because the plane's normal direction, dotted
# with each of a, b, c, gives the SAME value (the plane's signed distance from the origin) for all
# three — so the great-circle distance from the normal direction to each vertex depends only on that
# common value, i.e. is identical for a, b, and c.
#
# Each node's Voronoi cell is the (convex, star-shaped around the node) spherical polygon whose
# vertices are the circumcenters of every facet incident to that node, in cyclic order; its area is
# computed by fan-triangulating from the node itself and summing exact spherical-triangle areas
# (L'Huilier's theorem, the SAME formula `Grids._quad_area`'s spherical method already uses for
# `CurvilinearGrid` cell areas — reused here, not reimplemented). Since this is a full closed-sphere
# tessellation (no boundary to clip, unlike the planar Cartesian case), the areas sum to EXACTLY
# `4πR²` for any point set in general position — an exact invariant, not merely quasi-uniform-only.

# Outward-oriented spherical circumcenter (unit normal of the plane through a, b, c).
@inline function _circumcenter_direction(a::SA.SVector{3,T}, b::SA.SVector{3,T}, c::SA.SVector{3,T}) where {T}
    n = LA.cross(b - a, c - a)
    n = n / LA.norm(n)
    return LA.dot(n, a + b + c) < 0 ? -n : n
end

@inline _dir_to_lonlat(v::SA.SVector{3,T}) where {T} =
    SA.SVector{2,T}(atan(v[2], v[1]), asin(clamp(v[3], -one(T), one(T))))

function CGEF.Grids._voronoi_areas(
    geo::CGEF.SphericalGeometry{T}, lon::AbstractVector{T}, lat::AbstractVector{T},
) where {T<:AbstractFloat}
    N = length(lon)
    pts = [
        SA.SVector{3,T}(cos(lat[i]) * cos(lon[i]), cos(lat[i]) * sin(lon[i]), sin(lat[i]))
        for i in 1:N
    ]
    hull = QH.quickhull(pts)
    fs = QH.facets(hull)
    nf = length(fs)

    centers = Vector{SA.SVector{3,T}}(undef, nf)
    @inbounds for (fi, f) in enumerate(fs)
        centers[fi] = _circumcenter_direction(pts[f[1]], pts[f[2]], pts[f[3]])
    end

    # Vertex -> incident-facet CSR adjacency, exact two-pass preallocation (degree counted first, no
    # push!-growth): every facet contributes exactly 3 (vertex, facet) incidences.
    deg = zeros(Int, N)
    @inbounds for f in fs, v in f
        deg[v] += 1
    end
    ptr = Vector{Int}(undef, N + 1)
    ptr[1] = 1
    @inbounds for i in 1:N
        ptr[i+1] = ptr[i] + deg[i]
    end
    adj = Vector{Int}(undef, ptr[end] - 1)
    cursor = copy(ptr[1:end-1])
    @inbounds for (fi, f) in enumerate(fs), v in f
        adj[cursor[v]] = fi
        cursor[v] += 1
    end

    areas = Vector{T}(undef, N)
    angs = Vector{T}(undef, 0)
    cidxs = Vector{Int}(undef, 0)
    @inbounds for i in 1:N
        lo, hi = ptr[i], ptr[i+1] - 1
        m = hi - lo + 1
        if m < 3
            areas[i] = zero(T)   # degenerate (should not occur for a genuine triangulation)
            continue
        end
        center_i = SA.SVector{2,T}(lon[i], lat[i])
        resize!(angs, m); resize!(cidxs, m)
        for (k, idx) in enumerate(lo:hi)
            fi = adj[idx]
            nb = _dir_to_lonlat(centers[fi])
            d = CGEF.Geometry.project_to_tangent_plane(geo, center_i, nb)
            angs[k] = atan(d[2], d[1])
            cidxs[k] = fi
        end
        order = sortperm(angs)
        A = zero(T)
        for k in 1:m
            v1 = _dir_to_lonlat(centers[cidxs[order[k]]])
            v2 = _dir_to_lonlat(centers[cidxs[order[mod1(k + 1, m)]]])
            A += CGEF.Grids._sph_triangle_area(geo, center_i, v1, v2)
        end
        areas[i] = A
    end
    return areas
end

end # module
