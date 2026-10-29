# ---------------------------------------------------------------------------
# Spectral filtering over a grid's cells as a node set
# ---------------------------------------------------------------------------
#
# A grid no transform targets directly (a stretched or curvilinear Cartesian grid, a regional or
# pixelized sphere) is filtered over its cells as a node set: the nonuniform transform of its geometry
# (NUFFT on the plane, NUFSHT on the sphere) runs over the cell centres, and fields are read and written
# in the grid's linear cell order.
#
# A regional spherical grid is completed to the whole sphere by its lattice continued past each bounded
# edge, as inactive nodes, so the harmonic fit sees the field extended by zero.

# Which spectral backends may be served through the node set: the nonuniform transforms, and `Auto`.
@inline _node_route(::SpectralBackends.AbstractAutoSpectralBackend) = true
@inline _node_route(::SpectralBackends.AbstractNUFFTSpectralBackend) = true
@inline _node_route(::SpectralBackends.AbstractNUFSHTSpectralBackend) = true
@inline _node_route(::SpectralBackends.AbstractSpectralBackend) = false

"""
    NodeSetGridPlan

The scale-independent half of a node-set spectral plan: the node set built from the grid's cells, its
completion to the whole sphere where the grid is regional, the nonuniform transform's own grid plan
over it, and the execution backend whose memory the transform runs in. The grid's cells come first, in
linear order. `batch` is the trailing batch extent the transform was planned for, or `nothing`.
"""
struct NodeSetGridPlan{
    NG<:FlowGeometries.Grids.UnstructuredGrid, GP<:AbstractGridPlan, N, B<:Union{Nothing,Int},
    BE<:ComputationalBackends.AbstractExecutionBackend,
} <: AbstractGridPlan
    nodes::NG
    inner::GP
    ncells::Int
    dims::NTuple{N,Int}
    batch::B
    execution::BE
end

"""
    NodeSetScratch

The node-set plan's transient buffers: a field and an output over the node set, the same pair with a
trailing batch axis for a batched plan, and the nonuniform transform's own scratch. One per concurrent
worker.
"""
struct NodeSetScratch{T<:AbstractFloat, V<:AbstractVector{T}, BT, S} <: AbstractFilterScratch
    field::V
    out::V
    batched::BT   # (; field, out) of size nnodes × batch, or nothing
    inner::S
end

"""
    NodeSetPlan

A spectral filter plan over a grid's cells: the nonuniform transform's plan over the node set, with
the [`NodeSetGridPlan`](@ref) and [`NodeSetScratch`](@ref) it was built from.
"""
struct NodeSetPlan{P<:AbstractFilterPlan, GP<:NodeSetGridPlan, SC<:NodeSetScratch} <: AbstractFilterPlan
    inner::P
    grid_plan::GP
    scratch::SC
end

plan_strategy(plan::NodeSetPlan) = plan_strategy(plan.inner)
plan_execution(plan::NodeSetPlan) = plan.grid_plan.execution
analysis_status(plan::NodeSetPlan) = analysis_status(plan.inner)

Base.show(io::IO, plan::NodeSetPlan) =
    print(io, "NodeSetPlan(", plan.grid_plan.ncells, " cells, ", length(plan.scratch.field), " nodes, ",
          nameof(typeof(plan.inner)), ")")

"""
    _node_set(grid) -> UnstructuredGrid

`grid`'s cells as a node set — centres, measures and mask, in linear order — followed on a regional
sphere by the rest of its lattice, inactive.
"""
function _node_set(grid::FlowGeometries.Grids.AbstractGrid{G,T}) where {G,T}
    geo = FlowGeometries.Grids.grid_geometry(grid)
    x = FlowGeometries.Grids.materialize(grid)
    A = collect(T, vec(FlowGeometries.Grids.measure(grid)))
    m = _node_mask(geo, FlowGeometries.Grids.mask(grid))
    xc, Ac, mc = _complete_sphere(geo, grid, x, A, m)
    return FlowGeometries.Grids.UnstructuredGrid(geo, xc, Ac, mc; _node_closure(geo, grid)...)
end

_node_set(grid::FlowGeometries.Grids.UnstructuredGrid) = grid

# The node set's mask in linear order. A Cartesian set adds no node, so an `AllActive` grid stays
# `AllActive`; a sphere's completion appends inactive nodes, so its mask is always an array.
_node_mask(::FlowGeometries.Geometry.AbstractCartesianGeometry, m::FlowGeometries.Grids.AllActive) =
    FlowGeometries.Grids.AllActive((length(m),))
_node_mask(::FlowGeometries.Geometry.AbstractGeometry, m::AbstractArray{Bool}) = BitVector(vec(m))

# A Cartesian node set carries the grid's own wrap, which fixes the transform's box. The sphere closes
# by itself.
_node_closure(::FlowGeometries.Geometry.AbstractSphericalGeometry, _) = (;)
function _node_closure(::FlowGeometries.Geometry.AbstractCartesianGeometry{T}, grid) where {T}
    tp = FlowGeometries.Grids.topology(grid)
    prd = ntuple(d -> FlowGeometries.Grids.isperiodic(grid, d) ? T(FlowGeometries.Grids.period(grid, d)) : zero(T),
                 Val(length(tp)))
    return (topology = tp, period = prd)
end

_complete_sphere(::FlowGeometries.Geometry.AbstractGeometry, _, x, A, m) = (x, A, m)

function _complete_sphere(
    geo::FlowGeometries.Geometry.AbstractSphericalGeometry{T},
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2}, x, A, m,
) where {T<:AbstractFloat, G}
    lattice = _exterior_lattice(grid, T(π) * T(FlowGeometries.Geometry.radius(geo)); cap = false)
    lattice === nothing && return (x, A, m)
    ext, lo, _ = lattice
    return _complete_from(ext, lo, FlowGeometries.Grids.size_tuple(grid), x, A, m)
end

# The lattice's cells outside the grid, appended inactive.
function _complete_from(ext, lo::NTuple{N,Int}, dims::NTuple{N,Int}, x, A, m) where {N}
    outside = vec([!_inside(Tuple(J), lo, dims) for J in CartesianIndices(FlowGeometries.Grids.size_tuple(ext))])
    xe = FlowGeometries.Grids.materialize(ext)
    Ae = vec(collect(FlowGeometries.Grids.measure(ext)))
    return (map((xd, xed) -> vcat(xd, xed[outside]), x, xe), vcat(A, Ae[outside]),
            vcat(m, falses(count(outside))))
end

"""
    _node_set_grid_plan(spectral_backend, grid, kernel; execution = AutoBackend(), batch = nothing,
                        kwargs...) -> NodeSetGridPlan or nothing

The node set of `grid` and the nonuniform transform's grid plan over it, planned for a trailing batch of
`batch` fields, or `nothing` where the backend does not take the node set or no loaded extension
transforms it.
"""
function _node_set_grid_plan(
    spectral_backend::SpectralBackends.AbstractSpectralBackend, grid::FlowGeometries.Grids.AbstractGrid,
    kernel::Kernels.AbstractFilterKernel;
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    batch::Union{Nothing,Integer} = nothing, analysis::AbstractAnalysis = AutoAnalysis(), kwargs...,
)
    _node_route(spectral_backend) || return nothing
    nodes = _node_set(grid)
    inner = spectral_grid_plan(spectral_backend, nodes, kernel; execution = execution, batch = batch,
                               _node_analysis(FlowGeometries.Grids.grid_geometry(grid), analysis, grid)..., kwargs...)
    inner === nothing && return nothing
    return NodeSetGridPlan(nodes, inner, length(FlowGeometries.Grids.mask(grid)),
                           FlowGeometries.Grids.size_tuple(grid), batch === nothing ? nothing : Int(batch),
                           execution)
end

# How a spherical node set's harmonic coefficients are taken, resolved from the grid it was made from, whose sampling
# the node set does not keep ([`AutoAnalysis`](@ref)). A Cartesian set's transform takes no analysis.
_node_analysis(::FlowGeometries.Geometry.AbstractSphericalGeometry, a::AbstractAnalysis, grid) =
    (; analysis = resolved_analysis(a, grid))
_node_analysis(::FlowGeometries.Geometry.AbstractCartesianGeometry, ::AutoAnalysis, _) = (;)
_node_analysis(::FlowGeometries.Geometry.AbstractCartesianGeometry, a::AbstractAnalysis, _) = (; analysis = a)

function _node_set_scratch(gp::NodeSetGridPlan, inner_scratch)
    T = eltype(FlowGeometries.Grids.measure(gp.nodes))
    n = length(FlowGeometries.Grids.mask(gp.nodes))
    b = gp.batch
    zs(dims...) = _allocate(gp.execution, T, dims)
    return NodeSetScratch(zs(n), zs(n), b === nothing ? nothing : (field = zs(n, b), out = zs(n, b)),
                          inner_scratch)
end

spectral_scratch(gp::NodeSetGridPlan) = _node_set_scratch(gp, spectral_scratch(gp.inner))

"""
    _node_set_filter_plan(spectral_backend, grid, kernel, scale; execution = AutoBackend(),
                          grid_plan = nothing, scratch = nothing, batch = nothing, kwargs...) -> NodeSetPlan

The spectral plan for `grid` over its node set. Raises where the backend does not take a node set or no
loaded extension transforms it.
"""
function _node_set_filter_plan(
    spectral_backend::SpectralBackends.AbstractSpectralBackend, grid::FlowGeometries.Grids.AbstractGrid,
    kernel::Kernels.AbstractFilterKernel, scale::AbstractFloat;
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    grid_plan::Union{Nothing,NodeSetGridPlan} = nothing,
    scratch::Union{Nothing,NodeSetScratch} = nothing,
    batch::Union{Nothing,Integer} = nothing,
    analysis::AbstractAnalysis = AutoAnalysis(),
    kwargs...,
)
    _node_route(spectral_backend) || _no_spectral_backend(spectral_backend, grid)
    nodes = grid_plan === nothing ? _node_set(grid) : grid_plan.nodes
    nb = grid_plan === nothing ? (batch === nothing ? nothing : Int(batch)) : grid_plan.batch
    inner = spectral_filter_plan(
        spectral_backend, nodes, kernel, scale;
        execution = execution, grid_plan = grid_plan === nothing ? nothing : grid_plan.inner,
        scratch = scratch === nothing ? nothing : scratch.inner, batch = nb,
        _node_analysis(FlowGeometries.Grids.grid_geometry(grid), analysis, grid)..., kwargs...,
    )
    gp = grid_plan === nothing ?
        NodeSetGridPlan(nodes, inner.grid_plan, length(FlowGeometries.Grids.mask(grid)),
                        FlowGeometries.Grids.size_tuple(grid), nb, execution) : grid_plan
    sc = scratch === nothing ? _node_set_scratch(gp, inner.scratch) : scratch
    return NodeSetPlan(inner, gp, sc)
end

# Every grid an extension does not claim takes its node set; a node set no extension claims has no
# transform loaded. The extensions' methods are narrower in both arguments, so they take precedence.
spectral_filter_plan(
    spectral_backend::SpectralBackends.AbstractSpectralBackend, grid::FlowGeometries.Grids.AbstractGrid,
    kernel::Kernels.AbstractFilterKernel, scale::AbstractFloat; kwargs...,
) = _node_set_filter_plan(spectral_backend, grid, kernel, scale; kwargs...)

spectral_filter_plan(
    spectral_backend::SpectralBackends.AbstractSpectralBackend, grid::FlowGeometries.Grids.UnstructuredGrid,
    ::Kernels.AbstractFilterKernel, ::AbstractFloat; kwargs...,
) = _no_spectral_backend(spectral_backend, grid)

spectral_grid_plan(
    spectral_backend::SpectralBackends.AbstractSpectralBackend, grid::FlowGeometries.Grids.AbstractGrid,
    kernel::Kernels.AbstractFilterKernel; kwargs...,
) = _node_set_grid_plan(spectral_backend, grid, kernel; kwargs...)

spectral_grid_plan(
    ::SpectralBackends.AbstractSpectralBackend, ::FlowGeometries.Grids.UnstructuredGrid,
    ::Kernels.AbstractFilterKernel; kwargs...,
) = nothing

@noinline _no_spectral_backend(::SpectralBackends.AbstractFFTSpectralBackend, grid) = throw(ArgumentError(
    "FFT spectral filtering samples each direction at equal steps, which a grid states in its axis " *
    "types (`FlowGeometries.Grids.spacing_trait`); this $(nameof(typeof(grid))) is not two uniform " *
    "directions. Filter it over its cells with `transform = AutoSpectralBackend()` and a " *
    "nonuniform FFT: $(_NUFFT_LIBRARIES_HINT).",
))

@noinline _no_spectral_backend(spectral_backend, grid) = throw(ArgumentError(
    "Spectral filtering with $(nameof(typeof(spectral_backend))) is unavailable for " *
    "$(nameof(typeof(grid))). A grid whose own layout has no transform is filtered over its cells by " *
    "the nonuniform transform of its geometry: pass `transform = AutoSpectralBackend()` and load " *
    "a nonuniform FFT for a Cartesian grid ($(_NUFFT_LIBRARIES_HINT)) or `NUFSHT` for a spherical one.",
))

const _NUFFT_LIBRARIES_HINT =
    "`using NonuniformFFTs` or `using FINUFFT` (tags `FlowTransformBindings.NonuniformFFTsBackend()`, " *
    "`FlowTransformBindings.FINUFFTBackend()`)"

# The grid's cells are the node set's first `ncells` nodes, in linear order, so a field of the grid's
# shape copies in and out linearly. The completion's nodes are inactive and never written.
@inline _single(plan::NodeSetPlan, A::AbstractArray) = ndims(A) == length(plan.grid_plan.dims)

function _load_nodes!(buf::AbstractArray, field::AbstractArray, nc::Int)
    nn = size(buf, 1)
    for b in 1:size(buf, 2)
        copyto!(buf, (b - 1) * nn + 1, field, (b - 1) * nc + 1, nc)
    end
    return buf
end

function _store_cells!(out::AbstractArray, buf::AbstractArray, nc::Int)
    nn = size(buf, 1)
    for b in 1:size(buf, 2)
        copyto!(out, (b - 1) * nc + 1, buf, (b - 1) * nn + 1, nc)
    end
    return out
end

function _batch_buffers(plan::NodeSetPlan, A::AbstractArray)
    p, nb = plan.scratch.batched, plan.grid_plan.batch
    (p === nothing || nb != size(A, ndims(A))) && throw(ArgumentError(
        "this spectral plan was not built for a batch of $(size(A, ndims(A))); pass `batch = nb` to " *
        "`plan_filter`",
    ))
    return p
end

function filter_apply!(out::AbstractArray, field::AbstractArray, plan::NodeSetPlan)
    sc, n = plan.scratch, plan.grid_plan.ncells
    _load_nodes!(sc.field, field, n)
    filter_apply!(sc.out, sc.field, plan.inner)
    return _store_cells!(out, sc.out, n)
end

_batched_fields(outs, plan::NodeSetPlan) =
    plan.grid_plan.batch !== nothing && ndims(first(outs)) == length(plan.grid_plan.dims) + 1

function filter_apply_batched!(out::AbstractArray, field::AbstractArray, plan::NodeSetPlan)
    size(out) == size(field) || throw(DimensionMismatch(
        "filter_apply_batched! got out $(size(out)) and field $(size(field))",
    ))
    p, n = _batch_buffers(plan, field), plan.grid_plan.ncells
    _load_nodes!(p.field, field, n)
    filter_apply_batched!(p.out, p.field, plan.inner)
    return _store_cells!(out, p.out, n)
end

analyze_buffer(plan::NodeSetPlan, field::AbstractArray) =
    _single(plan, field) ? analyze_buffer(plan.inner, plan.scratch.field) :
    _batch_analysis(plan.scratch.batched, size(field, ndims(field)), _ -> plan.grid_plan.batch,
                    () -> analyze_buffer(plan.inner, plan.scratch.batched.field))

function filter_analyze!(F̂, field::AbstractArray, plan::NodeSetPlan)
    buf = _single(plan, field) ? plan.scratch.field : _batch_buffers(plan, field).field
    _load_nodes!(buf, field, plan.grid_plan.ncells)
    return filter_analyze!(F̂, buf, plan.inner)
end

function filter_synthesize!(out::AbstractArray, F̂, plan::NodeSetPlan)
    buf = _single(plan, out) ? plan.scratch.out : _batch_buffers(plan, out).out
    filter_synthesize!(buf, F̂, plan.inner)
    return _store_cells!(out, buf, plan.grid_plan.ncells)
end

series_form(plan::NodeSetPlan) = series_form(plan.inner)

rate_available(plan::NodeSetPlan, kernel::Kernels.AbstractFilterKernel) = rate_available(plan.inner, kernel)
rate_plan(plan::NodeSetPlan, kernel::Kernels.AbstractFilterKernel, scale::AbstractFloat) =
    NodeSetPlan(rate_plan(plan.inner, kernel, scale), plan.grid_plan, plan.scratch)

# The node set's field buffer is free once analyzed, so it holds `ū` for the inner plan's quotient rule.
function filter_synthesize_derivative!(out::AbstractArray, F̂, ū::AbstractArray, plan::NodeSetPlan, d::Int)
    bufs = _single(plan, out) ? plan.scratch : _batch_buffers(plan, out)
    n = plan.grid_plan.ncells
    _load_nodes!(bufs.field, ū, n)
    filter_synthesize_derivative!(bufs.out, F̂, bufs.field, plan.inner, d)
    return _store_cells!(out, bufs.out, n)
end

"""
    NodeSetGradient

The inner plan's [`series_gradient`](@ref) and the node-set pair a gradient lands in before its cells are
copied out, with the same pair over the batch for a batched plan.
"""
struct NodeSetGradient{G, V<:AbstractArray, BT}
    inner::G
    out::NTuple{2,V}
    batched::BT   # a pair of nnodes × batch arrays, or nothing
end

# An inner plan needing no gradient state takes `filter_synthesize_derivative!` per direction.
series_gradient(plan::NodeSetPlan) = _node_gradient_state(series_gradient(plan.inner), plan)
_node_gradient_state(::Nothing, ::NodeSetPlan) = nothing
function _node_gradient_state(inner, plan::NodeSetPlan)
    sc = plan.scratch
    pair(a) = (similar(a), similar(a))
    return NodeSetGradient(inner, pair(sc.out), sc.batched === nothing ? nothing : pair(sc.batched.out))
end

function filter_synthesize_gradient!(g1::AbstractArray, g2::AbstractArray, F̂, ū::AbstractArray, plan::NodeSetPlan,
                                     g::NodeSetGradient)
    if _single(plan, g1)
        _node_gradient!(g1, g2, F̂, ū, plan, plan.scratch.field, g.out, g.inner)
    else
        _node_gradient!(g1, g2, F̂, ū, plan, _batch_buffers(plan, g1).field, g.batched, g.inner)
    end
    return g1, g2
end

function _node_gradient!(g1, g2, F̂, ū, plan::NodeSetPlan, field, out::NTuple{2,Any}, inner)
    n = plan.grid_plan.ncells
    _load_nodes!(field, ū, n)
    filter_synthesize_gradient!(out[1], out[2], F̂, field, plan.inner, inner)
    _store_cells!(g1, out[1], n)
    _store_cells!(g2, out[2], n)
    return nothing
end
