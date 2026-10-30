# ---------------------------------------------------------------------------
# The real-space plan of a flat-cell grid: a node set or a sphere pixelization
# ---------------------------------------------------------------------------

"""
    NodeFilterPlan{T}

Real-space filter footprint for a node set: per-node CSR neighbour blocks (`nbrs`, `w`, offsets `ptr`)
with their geometric weights `w = kernel_weight(d) · control volume`, found once at plan time, in the plan's node
order: a point cloud's space-filling order, its node `s` the grid's node `perm[s]`, so a row's neighbours lie near
it in memory; the grid's own (`perm` `nothing`) where its ids follow space, as a pixelization's do. A host apply
reads the field and the mask into that order once (`vals`, `act`) and writes each node's result to its own id.

Each node's block includes the node itself, which `Connectivity.neighbors_within` leaves out. A neighbour is an
`Int32` node index; a node set of more nodes raises at build.
"""
struct NodeFilterPlan{T<:AbstractFloat, VN<:AbstractVector{Int32}, VT<:AbstractVector{T}, VI<:AbstractVector{Int}, P, V, B}
    nbrs::VN
    w::VT
    ptr::VI
    perm::P
    vals::V
    act::B
end

NodeFilterPlan(nbrs::AbstractVector{Int32}, w::AbstractVector{T}, ptr::AbstractVector{Int}, ::Nothing) where {T} =
    NodeFilterPlan(nbrs, w, ptr, nothing, nothing, nothing)
NodeFilterPlan(nbrs::AbstractVector{Int32}, w::AbstractVector{T}, ptr::AbstractVector{Int}, perm::AbstractVector{Int}) where {T} =
    NodeFilterPlan(nbrs, w, ptr, perm, Vector{T}(undef, length(perm)), Vector{Bool}(undef, length(perm)))

# The grid's id of the plan's node `s`.
@inline _node_id(fp::NodeFilterPlan, s::Int) = _node_id(fp.perm, s)
@inline _node_id(::Nothing, s::Int) = s
@inline _node_id(perm::AbstractVector{Int}, s::Int) = @inbounds perm[s]

# Node `s`'s value and activity read into the plan's order.
@inline _node_stage!(fp::NodeFilterPlan, field, grid, s::Int) = _node_stage!(fp.perm, fp, field, grid, s)
@inline _node_stage!(::Nothing, _, _, _, ::Int) = nothing
@inline function _node_stage!(perm::AbstractVector{Int}, fp::NodeFilterPlan{T}, field, grid, s::Int) where {T}
    @inbounds begin
        t = perm[s]
        fp.vals[s] = T(field[t])
        fp.act[s] = FlowGeometries.Grids.isactive(grid, t)
    end
    return nothing
end

# The values and activities the plan's rows read, by the plan's node index.
@inline _node_inputs(fp::NodeFilterPlan, field, grid) = _node_inputs(fp.perm, fp, field, grid)
@inline _node_inputs(::Nothing, _, field, grid) = (field, _NodeActivity(grid))
@inline _node_inputs(::AbstractVector{Int}, fp, _, _) = (fp.vals, fp.act)

struct _NodeActivity{G}
    grid::G
end
@inline Base.getindex(a::_NodeActivity, j::Integer) = FlowGeometries.Grids.isactive(a.grid, j)

# A list in grid order: the plan's own, or its rows and neighbours renamed by `perm`.
_grid_order_list(fp::NodeFilterPlan) = _grid_order_list(fp.perm, fp)
_grid_order_list(::Nothing, fp) = (fp.nbrs, fp.w, fp.ptr)
_grid_order_list(perm::AbstractVector{Int}, fp) = _renumbered((fp.nbrs, fp.w, fp.ptr), perm)

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
    built, perm = _build_order(grid)
    return NodeFilterPlan(_node_list(built, kernel, scale, _build_backend(execution))..., perm)
end

# Whether a flat-cell grid's neighbour list is read from its bins, every image of a cell across a periodic
# direction at its own distance: a Cartesian node set. Any other is built over the grid's own ball query.
_binned_list(::FlowGeometries.Grids.UnstructuredGrid{T,G}) where {T, G<:FlowGeometries.Geometry.AbstractCartesianGeometry{T}} = true
_binned_list(_) = false

function _node_list(grid, kernel, scale, backend)
    _binned_list(grid) && return _binned_csr(grid, kernel, scale, backend)
    return _node_csr(grid, kernel, scale, backend)
end

# Points on a sphere's surface are searched by chord in three dimensions; a shell's, with a radius, by the grid's
# own ball query.
function _node_list(grid::FlowGeometries.Grids.AbstractGrid{G,T}, kernel, scale, backend) where {T, G<:FlowGeometries.Geometry.SphericalGeometry{T}}
    FlowGeometries.Grids.ncoordinates(grid) == 2 && return _sphere_csr(grid, kernel, scale, backend)
    return _node_csr(grid, kernel, scale, backend)
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

# The backend a node plan's build runs on: the host's threads where the plan's apply is threaded, each
# thread taking rows as it finishes since their neighbour counts differ; the calling thread otherwise.
_build_backend(::ComputationalBackends.AbstractThreadedBackend) = FlowGeometries.Execution.BalancedThreads()
_build_backend(::ComputationalBackends.AbstractExecutionBackend) = ComputationalBackends.SerialBackend()

# The order a node plan is built and held in. A point cloud's ids need not follow its points through space, and a
# ball query and an apply then read from all over memory, so the plan is built over the nodes in a space-filling
# order (`spatial_order`) and keeps it. A pixelization's ids already do.
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
    nbrs = Vector{Int32}(undef, ptr[n + 1] - 1)
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
    fold = (f, acc, t, sc) -> FlowGeometries.Connectivity.fold_within(
        f, acc, grid, t; ball = rad, self = true, active_only = false, topology = mt, scratch = sc,
    )
    _, _, kept = _highorder_targets(grid, kernel, scale, rad, fold, FlowGeometries.Connectivity.ball_scratch, nothing,
                                    true, false, _build_share(get(kwargs, :execution, ComputationalBackends.SerialBackend())))
    return NodeFilterPlan(Vector{Int32}(kept[1]), kept[2], kept[3], nothing)
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
    for s in eachindex(out)
        _node_stage!(fp, field, grid, s)
    end
    vals, act = _node_inputs(fp, field, grid)
    @inbounds for s in eachindex(out)
        out[_node_id(fp, s)] = _footprint_node_point(field, vals, act, fp, strategy, s)
    end
    return out
end

# The plan's node `s`'s filtered value from the values and activities in the plan's order, shared by the serial
# and parallel drivers; `field` decides what is emitted. A flat-cell grid has no lattice past its cells, so no
# exterior mass enters the normalization.
@inline function _footprint_node_point(
    field::AbstractVector, vals, act, fp::NodeFilterPlan{T}, strategy::AbstractMaskStrategy, s::Integer,
) where {T<:AbstractFloat}
    _filters_target(strategy, act[s]) || return zero(T)
    ws = zero(T)
    wn = zero(T)
    @inbounds for k in fp.ptr[s]:(fp.ptr[s+1] - 1)
        j = fp.nbrs[k]
        wj = fp.w[k]
        active = act[j]
        if strategy isa ZeroFill
            wn += wj
            active && (ws += wj * vals[j])
        else
            active || continue
            wn += wj
            ws += wj * vals[j]
        end
    end
    return _emit(field, ws, wn)
end
