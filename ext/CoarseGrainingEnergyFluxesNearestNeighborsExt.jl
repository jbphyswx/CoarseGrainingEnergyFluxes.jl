module CoarseGrainingEnergyFluxesNearestNeighborsExt

using NearestNeighbors: NearestNeighbors
using StaticArrays: StaticArrays as SA
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF

# Real k-d-tree neighbor construction for `UnstructuredGrid` (overrides the throwing fallback in
# `Grids._build_kdtree_neighbors`) — brute-force O(N²) doesn't scale, so this is the only supported
# adjacency-construction path. Two embeddings, dispatched on geometry type:
#   CartesianGeometry: the tree is built directly on (lon, lat) — nearest in the tree IS nearest in
#     the geometry's own Euclidean metric.
#   SphericalGeometry: the tree is built on the 3D Cartesian embedding of each node (unit-sphere
#     direction) — nearest-by-3D-chord is EXACTLY nearest-by-great-circle-distance (monotonic in the
#     chord length), so this is exact, not an approximation.
# Either a `k`-nearest-neighbor query or a `radius` (physical distance, converted to the embedding's
# own metric) query populates the CSR `(neighbor_nbrs, neighbor_ptr)` pair with an EXACT
# preallocation — the row-length upper bound is `k` (or the true return length of `inrange`), known
# before allocating, so no `push!`-growth is needed anywhere in this file.

# Build the CSR adjacency from a set of embedded points (2D Cartesian or 3D spherical-direction),
# self-excluded, via k-nearest-neighbor query.
function _csr_from_knn(pts::AbstractVector, k::Integer)
    N = length(pts)
    if N < 2
        return Int[], ones(Int, N + 1)
    end
    kq = min(k, N - 1)
    tree = NearestNeighbors.KDTree(pts)
    idxs, _ = NearestNeighbors.knn(tree, pts, kq + 1, true)
    # Exact row length (not a conservative bound): count of returned neighbours excluding self.
    rowlen = Vector{Int}(undef, N)
    @inbounds for i in 1:N
        rowlen[i] = min(count(!=(i), idxs[i]), kq)
    end
    ptr = Vector{Int}(undef, N + 1)
    ptr[1] = 1
    @inbounds for i in 1:N
        ptr[i+1] = ptr[i] + rowlen[i]
    end
    nbrs = Vector{Int}(undef, ptr[end] - 1)
    @inbounds for i in 1:N
        cursor = ptr[i]
        stop = ptr[i+1] - 1
        for j in idxs[i]
            j == i && continue
            cursor > stop && break
            nbrs[cursor] = j
            cursor += 1
        end
    end
    return nbrs, ptr
end

# Build the CSR adjacency via an all-neighbors-within-`r` query (self excluded); `r` is already in
# the embedding's own distance units (chord distance for the spherical embedding).
function _csr_from_radius(pts::AbstractVector, r::Real)
    N = length(pts)
    if N < 2
        return Int[], ones(Int, N + 1)
    end
    tree = NearestNeighbors.KDTree(pts)
    lists = NearestNeighbors.inrange(tree, pts, r, false)   # self always included (distance 0 <= r)
    ptr = Vector{Int}(undef, N + 1)
    ptr[1] = 1
    @inbounds for i in 1:N
        ptr[i+1] = ptr[i] + (length(lists[i]) - 1)
    end
    nbrs = Vector{Int}(undef, ptr[end] - 1)
    @inbounds for i in 1:N
        cursor = ptr[i]
        for j in lists[i]
            j == i && continue
            nbrs[cursor] = j
            cursor += 1
        end
    end
    return nbrs, ptr
end

function CGEF.Grids._build_kdtree_neighbors(
    geo::CGEF.CartesianGeometry{T}, lon::AbstractVector{T}, lat::AbstractVector{T};
    k::Integer = 6, radius::Union{Nothing,Real} = nothing,
) where {T<:AbstractFloat}
    pts = [SA.SVector{2,T}(lon[i], lat[i]) for i in eachindex(lon)]
    return radius === nothing ? _csr_from_knn(pts, k) : _csr_from_radius(pts, T(radius))
end

function CGEF.Grids._build_kdtree_neighbors(
    geo::CGEF.SphericalGeometry{T}, lon::AbstractVector{T}, lat::AbstractVector{T};
    k::Integer = 6, radius::Union{Nothing,Real} = nothing,
) where {T<:AbstractFloat}
    pts = [
        SA.SVector{3,T}(cos(lat[i]) * cos(lon[i]), cos(lat[i]) * sin(lon[i]), sin(lat[i]))
        for i in eachindex(lon)
    ]
    if radius === nothing
        return _csr_from_knn(pts, k)
    else
        # Physical arc radius -> unit-sphere chord radius: chord = 2 sin(arc / 2), arc = radius / R.
        arc = T(radius) / geo.R
        chord_radius = T(2) * sin(arc / T(2))
        return _csr_from_radius(pts, chord_radius)
    end
end

end # module
