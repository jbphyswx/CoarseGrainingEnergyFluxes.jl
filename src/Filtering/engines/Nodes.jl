# ---------------------------------------------------------------------------
# The real-space plan of a flat-cell grid: a node set or a sphere pixelization
# ---------------------------------------------------------------------------

"""
    NodeFilterPlan{T}

Real-space filter footprint for a node set: per-node CSR neighbour blocks (`nbrs`, `w`, offsets `ptr`)
with their geometric weights `w = kernel_weight(d) · control volume`, found once at plan time.

Each node's block includes the node itself, which `Connectivity.neighbors_within` leaves out.
"""
struct NodeFilterPlan{T<:AbstractFloat, VI<:AbstractVector{Int}, VT<:AbstractVector{T}}
    nbrs::VI
    w::VT
    ptr::VI
end

"""
    build_footprint(grid, kernel, scale; kwargs...) -> NodeFilterPlan

Real-space footprint for a flat-cell grid (see [`_is_flat_cell`](@ref)): every cell's neighbours within
`kernel_radius` and their weights. A Cartesian node set's are read from its bins
([`_binned_csr`](@ref)), each image of a cell across a periodic direction at its own distance; any other
grid's come from `Connectivity.fold_within`, the grid's own metric ball query on the cell list
`Connectivity.default_sweep_topology` returns, so the neighbourhood follows the geometry's distance and
periodic wrap. Raises for a grid whose cells are an index tuple and that no rectilinear or curvilinear
engine matches.
"""
build_footprint(
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}} =
    _build_footprint_flat(_cell_address(grid), grid, kernel, scale; kwargs...)

# Reached only by a `CartesianCells` grid that no rectilinear or curvilinear `build_footprint` matches.
@noinline _build_footprint_flat(
    ::FlowGeometries.Grids.CartesianCells, grid, kernel, scale; kwargs...,
) = throw(ArgumentError(
    "no real-space engine is implemented for $(nameof(typeof(grid))). Its cells are named by an " *
    "index tuple, so the flat-cell node engine does not apply, and no rectilinear or curvilinear " *
    "engine matches it either. Use `method = Spectral()` if a transform targets this grid.",
))

function _build_footprint_flat(
    ::FlowGeometries.Grids.FlatCells,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    return NodeFilterPlan(_node_list(grid, kernel, scale, _build_backend(execution))...)
end

# Whether a flat-cell grid's neighbour list is read from its bins, every image of a cell across a periodic
# direction at its own distance: a Cartesian node set. Any other is built over the grid's own ball query.
_binned_list(::FlowGeometries.Grids.UnstructuredGrid{T,G}) where {T, G<:FlowGeometries.Geometry.AbstractCartesianGeometry{T}} = true
_binned_list(_) = false

function _node_list(grid, kernel, scale, backend)
    _binned_list(grid) && return _binned_csr(grid, kernel, scale, backend)
    built, perm = _build_order(grid)
    return _renumbered(_node_csr(built, kernel, scale, backend), perm)
end

# Points on a sphere's surface are searched by chord in three dimensions; a shell's, with a radius, by the grid's
# own ball query.
function _node_list(grid::FlowGeometries.Grids.AbstractGrid{G,T}, kernel, scale, backend) where {T, G<:FlowGeometries.Geometry.SphericalGeometry{T}}
    FlowGeometries.Grids.ncoordinates(grid) == 2 && return _sphere_csr(grid, kernel, scale, backend)
    built, perm = _build_order(grid)
    return _renumbered(_node_csr(built, kernel, scale, backend), perm)
end

"""
    _sphere_csr(grid, kernel, scale, backend) -> (nbrs, w, ptr)

A spherical flat-cell grid's list, read from bins of its points embedded in three dimensions
([`_binned_csr`](@ref)): the ball of great-circle radius `r` on the sphere of radius `R` is the chord ball of radius
`2R sin(r/2R)`, and a neighbour a chord `c` away lies `2R asin(c/2R)` along the sphere. A bin's side is the largest
of the mean spacing, half the chord radius and `2R/128`, so the lattice over the sphere's bounding cube holds at
most `128³` bins.
"""
function _sphere_csr(grid::FlowGeometries.Grids.AbstractGrid{G,T}, kernel, scale::T, backend) where {T, G}
    R = T(FlowGeometries.Geometry.radius(FlowGeometries.Grids.grid_geometry(grid)))
    λ, φ = FlowGeometries.Grids.materialize(grid)
    x = @. R * cos(φ) * cos(λ)
    y = @. R * cos(φ) * sin(λ)
    z = @. R * sin(φ)
    dim = _kernel_dim(grid)
    rad = T(Kernels.kernel_radius(kernel, scale, dim))
    rc = rad >= T(π) * R ? 2R : 2R * sin(rad / (2R))
    n = length(x)
    width = max(sqrt(T(4π) * R^2 / n), rc / 2, 2R / 128)
    cart = FlowGeometries.Grids.UnstructuredGrid(FlowGeometries.Geometry.CartesianGeometry{T}(), (vec(x), vec(y), vec(z)),
                                                 collect(T, vec(FlowGeometries.Grids.measure(grid))),
                                                 _flat_mask(FlowGeometries.Grids.mask(grid), n))
    bp = BinnedPoints(cart, Val(3); width = width)
    return _binned_csr(bp, kernel, scale, dim, rc, c -> 2R * asin(min(c / (2R), one(T))), backend)
end

_flat_mask(::FlowGeometries.Grids.AllActive, n::Int) = FlowGeometries.Grids.AllActive((n,))
_flat_mask(m::AbstractArray{Bool}, ::Int) = BitVector(vec(m))

# A neighbour list of `n` entries over `npts` points holds an `Int` and a weight per entry; one past the cache
# budget is refused before it is allocated.
function _check_list_bytes(n::Integer, npts::Integer, ::Type{T}) where {T}
    bytes = n * (sizeof(Int) + sizeof(T))
    bytes <= default_cache_byte_budget() || _list_too_large(n, npts, bytes)
    return nothing
end

@noinline _list_too_large(n, npts, bytes) = throw(ArgumentError(
    "this filter's neighbour list would hold $n entries, $(round(Int, n / max(npts, 1))) per point " *
    "($(round(bytes / 2^30; digits = 1)) GiB), past the cache budget of a quarter of this machine's memory. Filter at " *
    "a smaller scale or on fewer points, or with `method = Spectral()` where a transform serves the grid."))

# The backend a node plan's build runs on: the host's threads where the plan's apply is threaded, each
# thread taking rows as it finishes since their neighbour counts differ; the calling thread otherwise.
_build_backend(::ComputationalBackends.AbstractThreadedBackend) = FlowGeometries.Execution.BalancedThreads()
_build_backend(::ComputationalBackends.AbstractExecutionBackend) = ComputationalBackends.SerialBackend()

# The order a node plan is built in. A point cloud's ids need not follow its points through space, and a
# ball query then reads its candidates from all over memory, so the plan is built over the nodes in a
# space-filling order (`spatial_order`) and renumbered back. A pixelization's ids already do.
_build_order(grid::FlowGeometries.Grids.UnstructuredGrid) =
    (perm = FlowGeometries.Grids.spatial_order(grid); (FlowGeometries.Grids.reorder(grid, perm), perm))
_build_order(grid) = (grid, nothing)

# A list built over the nodes in the order `perm` lists them, with its rows and neighbours named by the
# grid's own ids: built node `s` is node `perm[s]`.
_renumbered(csr::Tuple, ::Nothing) = csr
function _renumbered((bnbrs, bw, bptr)::Tuple, perm::AbstractVector{<:Integer})
    n = length(perm)
    ptr = Vector{Int}(undef, n + 1)
    ptr[1] = 1
    len = Vector{Int}(undef, n)
    @inbounds for s in 1:n
        len[perm[s]] = bptr[s + 1] - bptr[s]
    end
    @inbounds for t in 1:n
        ptr[t + 1] = ptr[t] + len[t]
    end
    nbrs, w = similar(bnbrs), similar(bw)
    @inbounds for s in 1:n
        o = ptr[perm[s]] - bptr[s]
        for k in bptr[s]:(bptr[s + 1] - 1)
            nbrs[k + o] = perm[bnbrs[k]]
            w[k + o] = bw[k]
        end
    end
    return nbrs, w, ptr
end

# Every node's neighbours within the kernel's radius and their weights, as CSR in the grid's node order:
# one pass counts each row, the next fills it, so every row is written where the counts put it and the
# rows run on `backend` independently.
function _node_csr(
    grid::FlowGeometries.Grids.AbstractGrid{G,T}, kernel::Kernels.AbstractFilterKernel, scale::T, backend,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    dim = _kernel_dim(grid)
    rad = Kernels.kernel_radius(kernel, scale, dim)
    n = length(FlowGeometries.Grids.mask(grid))
    mt = _query_topology(grid, rad)
    # `active_only = false` keeps the plan mask-independent, the mask entering at apply time;
    # `self = true` includes the centre cell.
    query(f::F, acc, t, sc) where {F} = FlowGeometries.Connectivity.fold_within(
        f, acc, grid, t; ball = rad, self = true, active_only = false, topology = mt, scratch = sc,
    )
    ptr = Vector{Int}(undef, n + 1)
    ptr[1] = 1
    FlowGeometries.Execution.run_chunks(n, backend) do rng
        sc = FlowGeometries.Connectivity.ball_scratch(mt)
        @inbounds for t in rng
            ptr[t + 1] = query((m, _, _) -> m + 1, 0, t, sc)
        end
    end
    @inbounds for t in 1:n
        ptr[t + 1] += ptr[t]
    end
    _check_list_bytes(ptr[n + 1] - 1, n, T)
    nbrs = Vector{Int}(undef, ptr[n + 1] - 1)
    w = Vector{T}(undef, ptr[n + 1] - 1)
    FlowGeometries.Execution.run_chunks(n, backend) do rng
        sc = FlowGeometries.Connectivity.ball_scratch(mt)
        @inbounds for t in rng
            query(ptr[t], t, sc) do k, j, d
                nbrs[k] = j
                w[k] = Kernels.kernel_weight(Kernels.point_kernel(kernel), T(d), scale, dim) *
                       FlowGeometries.Grids.measure(grid, j)
                return k + 1
            end
        end
    end
    return nbrs, w, ptr
end

# A `HighOrderKernel`'s weights are corrected about each node ([`_highorder_targets`](@ref)), and a node
# plan stores them all.
function _build_footprint_flat(
    ::FlowGeometries.Grids.FlatCells,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.HighOrderKernel,
    scale::T;
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    rad = _query_radius(grid, kernel, scale)
    mt = _query_topology(grid, rad)
    scratch = FlowGeometries.Connectivity.ball_scratch()
    fold = (f, acc, t) -> FlowGeometries.Connectivity.fold_within(
        f, acc, grid, t; ball = rad, self = true, active_only = false, topology = mt, scratch = scratch,
    )
    _, _, kept = _highorder_targets(grid, kernel, scale, rad, fold, nothing, Returns(true))
    return NodeFilterPlan(kept...)
end

"""
    apply_footprint!(out, field, grid, fp::NodeFilterPlan, strategy) -> out

Weighted mean over each node's stored neighbourhood, with the same two mask conventions the structured
engines use: `ZeroFill` keeps a masked neighbour in the denominator, contributes nothing for it and
filters a masked node too; `Deformable` drops it from both sums and leaves a masked node at zero.
"""
function apply_footprint!(
    out::AbstractVector{T}, field::AbstractVector, grid::FlowGeometries.Grids.AbstractGrid,
    fp::NodeFilterPlan{T}, strategy::AbstractMaskStrategy,
) where {T<:AbstractFloat}
    @inbounds for t in eachindex(out)
        out[t] = _footprint_node_point(field, grid, fp, strategy, t)
    end
    return out
end

# Node `t`'s filtered value, shared by the serial and parallel drivers. A flat-cell grid has no lattice
# past its cells, so no exterior mass enters the normalization.
@inline function _footprint_node_point(
    field::AbstractVector, grid::FlowGeometries.Grids.AbstractGrid,
    fp::NodeFilterPlan{T}, strategy::AbstractMaskStrategy, t::Integer,
) where {T<:AbstractFloat}
    _filters_target(strategy, FlowGeometries.Grids.isactive(grid, t)) || return zero(T)
    ws = zero(T)
    wn = zero(T)
    @inbounds for k in fp.ptr[t]:(fp.ptr[t+1] - 1)
        j = fp.nbrs[k]
        wj = fp.w[k]
        active = FlowGeometries.Grids.isactive(grid, j)
        if strategy isa ZeroFill
            wn += wj
            active && (ws += wj * field[j])
        else
            active || continue
            wn += wj
            ws += wj * field[j]
        end
    end
    return _emit(field, ws, wn)
end
