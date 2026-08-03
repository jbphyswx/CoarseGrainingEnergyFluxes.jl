module Derivatives

using FlowGeometries: FlowGeometries

export ddx!, ddy!, ddz!
export StencilPlan

# A grid with no separable axis to difference along — curvilinear, or a node set — takes the
# least-squares tangent-plane gradient instead: `Connectivity.gradient_plan(grid)` once, then
# `Discretization.gradient!(g1, g2, field, plan)`, which returns both components from one traversal.
# Nothing of that belongs here; it is geometry and connectivity.

# ---------------------------------------------------------------------------
# Structured derivatives: `Discretization.derivative!` per direction.
#
# One set of methods for every geometry — the metric division and the pole, where `h_λ = R cos φ → 0`
# and the derivative does not exist, are the geometry's own and are handled there. On a Cartesian
# metric the division is the identity and costs nothing.
# ---------------------------------------------------------------------------
"""
    _dd!(∂f, f, grid, d) -> ∂f

Derivative of `f` with respect to distance along direction `d`. `nodes = 3` for 2nd order on a
stretched axis; `ReduceInRun` keeps the one-sided value at a mask edge, where the default writes zero.
"""
_dd!(∂f::AbstractArray{T}, f::AbstractArray{T}, grid, d::Int) where {T<:AbstractFloat} =
    FlowGeometries.Discretization.derivative!(
        ∂f, f, grid, d;
        order = 1, nodes = 3, masked = zero(T),
        policy = FlowGeometries.Discretization.ReduceInRun(),
    )

"""
    ddx!(∂f∂x, f, grid[, plan]) -> ∂f∂x

Derivative of `f` with respect to distance along the Eastward/λ direction.

Pass a [`StencilPlan`](@ref) to reuse the weights across calls; without one they are rebuilt each time.
The dimensionality is pinned per direction, so asking a grid for a derivative it has no axis for is a
`MethodError` at the call rather than a bounds error inside the kernel.
"""
ddx!(∂f, f, grid::FlowGeometries.Grids.StructuredGrid{G,T,N}) where {T<:AbstractFloat,N,G<:FlowGeometries.Geometry.AbstractGeometry{T}} =
    _dd!(∂f, f, grid, 1)

"""
    ddy!(∂f∂y, f, grid[, plan]) -> ∂f∂y

Derivative of `f` with respect to distance along the Northward/φ direction — see [`ddx!`](@ref).
"""
ddy!(∂f, f, grid::FlowGeometries.Grids.StructuredGrid{G,T,N}) where {T<:AbstractFloat,N,G<:FlowGeometries.Geometry.AbstractGeometry{T}} =
    _dd!(∂f, f, grid, 2)

"""
    ddz!(∂f∂z, f, grid[, plan]) -> ∂f∂z

Derivative of `f` with respect to distance along the third direction of a 3D grid, which supplies the
axis — see [`ddx!`](@ref). For a 3D field over a *2D* grid, see the `dz` method below.
"""
ddz!(∂f, f, grid::FlowGeometries.Grids.StructuredGrid{G,T,3}) where {T<:AbstractFloat,G<:FlowGeometries.Geometry.AbstractGeometry{T}} =
    _dd!(∂f, f, grid, 3)

"""
    StencilPlan(grid; order = 1, nodes = 3)

The finite-difference weights of every direction of `grid`, built once. `Discretization.axis_stencils`
per axis; the derivative is then `Discretization.derivative!` reading a table it does not have to
rebuild.

The weights depend only on the axis, its wrap period and the requested order — never on a field — so a
caller taking many derivatives on one grid should build this once and pass it. Without it each call
rebuilds an `order`-by-`nodes` table per axis sample, which is `O(n)` work and `O(n·nodes)` garbage
against an `O(nᴺ)` apply: negligible on a large grid, several times the whole cost on a small one.

It also carries the degrade path's scratch, so a masked grid allocates nothing per call either. That
scratch is written per cell, so a plan is **one per task** — the same contract as
`Connectivity.ball_scratch`. Sharing one across concurrent tasks races; build one per task instead. (A
threaded `backend` passed to `apply_stencil!` allocates its own set per chunk and ignores this one.)

`compute_Π!` builds one internally when its `deriv_plan` is `nothing`.
"""
struct StencilPlan{N, TT<:Tuple, S}
    tables::TT      # per direction: (indices, weights), or `nothing` where the axis is too short
    order::Int
    scratch::S      # Fornberg buffers for a window rebuilt at a mask edge; written per cell
end

Base.show(io::IO, p::StencilPlan{N}) where {N} =
    print(io, "StencilPlan{", N, "}(order ", p.order, ", ",
          count(!isnothing, p.tables), " of ", N, " directions differentiable)")

function StencilPlan(
    grid::FlowGeometries.Grids.StructuredGrid{G,T,N}; order::Integer = 1, nodes::Integer = 3,
) where {T<:AbstractFloat, N, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    ord = Int(order)
    tables = ntuple(Val(N)) do d
        x = FlowGeometries.Grids.coordinates(grid, d)
        # An axis with fewer than `order + 1` samples carries no derivative of that order at all, so
        # there is no table to build — `ReduceInRun` writes `masked` across it, which is what the
        # apply below does directly.
        length(x) < ord + 1 && return nothing
        return FlowGeometries.Discretization.axis_stencils(
            x, ord, min(Int(nodes), length(x));
            period = FlowGeometries.Grids.isperiodic(grid, d) ?
                     FlowGeometries.Grids.period(grid, d) : nothing,
        )
    end
    scratch = FlowGeometries.Discretization.stencil_scratch(T, ord, Int(nodes))
    return StencilPlan{N, typeof(tables), typeof(scratch)}(tables, ord, scratch)
end

"""
    StencilPlan(axis::AbstractVector; order = 1, nodes = 3, period = nothing)

A one-direction plan over a bare axis, for the level-stack [`ddz!`](@ref) — there the vertical spacing
is an argument rather than a grid axis, so there is no grid to take it from.
"""
function StencilPlan(
    x::AbstractVector{T}; order::Integer = 1, nodes::Integer = 3, period = nothing,
) where {T<:AbstractFloat}
    ord = Int(order)
    tables = (length(x) < ord + 1 ? nothing :
              FlowGeometries.Discretization.axis_stencils(x, ord, min(Int(nodes), length(x));
                                                          period = period),)
    scratch = FlowGeometries.Discretization.stencil_scratch(T, ord, Int(nodes))
    return StencilPlan{1, typeof(tables), typeof(scratch)}(tables, ord, scratch)
end

@inline function _dd!(
    ∂f::AbstractArray{T}, f::AbstractArray{T}, grid, d::Int, plan::StencilPlan,
) where {T<:AbstractFloat}
    tab = plan.tables[d]
    tab === nothing && return fill!(∂f, zero(T))
    return FlowGeometries.Discretization.derivative!(
        ∂f, f, grid, tab[1], tab[2], d;
        order = plan.order, masked = zero(T),
        policy = FlowGeometries.Discretization.ReduceInRun(),
        scratch = plan.scratch,
    )
end

ddx!(∂f, f, grid::FlowGeometries.Grids.StructuredGrid, plan::StencilPlan) = _dd!(∂f, f, grid, 1, plan)
ddy!(∂f, f, grid::FlowGeometries.Grids.StructuredGrid, plan::StencilPlan) = _dd!(∂f, f, grid, 2, plan)
ddz!(∂f, f, grid::FlowGeometries.Grids.StructuredGrid{G,T,3}, plan::StencilPlan) where {T<:AbstractFloat,G} =
    _dd!(∂f, f, grid, 3, plan)

# ---------------------------------------------------------------------------
# Z-derivative (ddz!) - Supports 3D structures
# ---------------------------------------------------------------------------

"""
    ddz!(∂f∂z, f, grid, dz[, plan]) -> ∂f∂z

Calculate spatial derivative of `f` in the vertical coordinate z, writing to `∂f∂z`.

This is the level-stack case: a 3D field over a **2D** grid (the same shape `filter_field!` accepts for
a stack of levels). The grid describes only the horizontal, so it cannot supply the vertical spacing —
`dz` is therefore an explicit argument rather than something read off the geometry. For a genuine 3D
grid use the `StructuredGrid{…,3}` method, which takes its spacing from the grid's own third axis and
handles nonuniform levels.

Since the vertical axis is not on the grid, its weights cannot come from a grid-built
[`StencilPlan`](@ref); build one over the axis instead — `StencilPlan(range(0; step = dz, length = Nz))`
— and pass it, or the table is rebuilt on every call at `O(Nz)`.
"""
function ddz!(
    ∂f∂z::AbstractArray{T,3},
    f::AbstractArray{T,3},
    grid::FlowGeometries.Grids.StructuredGrid{FlowGeometries.Geometry.CartesianGeometry{T},T,2},
    dz::T,
    plan::Union{Nothing,StencilPlan} = nothing,
) where {T<:AbstractFloat}
    Nx, Ny, Nz = size(f)
    tab = plan === nothing ? nothing : plan.tables[1]
    if tab === nothing
        FlowGeometries.Discretization.apply_stencil!(
            ∂f∂z, f, range(zero(T); step = dz, length = Nz), 3;
            order = 1, nodes = 3, masked = zero(T),
        )
    else
        FlowGeometries.Discretization.apply_stencil!(
            ∂f∂z, f, tab[1], tab[2], 3; masked = zero(T),
        )
    end
    # The grid's mask is horizontal, so it blanks whole columns; the vertical itself is unmasked and
    # is the one direction `apply_stencil!` cannot take the mask for, the shapes differing by a rank.
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        FlowGeometries.Grids.isactive(grid, i, j) || (∂f∂z[i, j, k] = zero(T))
    end
    return ∂f∂z
end

# A true 3D spherical grid needs no methods of its own: the third axis is the radius, so `h_λ = r cos φ`
# and `h_φ = r` vary per level rather than using a fixed reference radius, and `scale_factors` is
# evaluated at each point's own coordinates — which is what the generic methods above already do.

end # module
