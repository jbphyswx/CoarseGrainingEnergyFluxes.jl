module Derivatives

using FlowGeometries: FlowGeometries
using ComputationalBackends: ComputationalBackends

"""
    AbstractDerivative

How a diagnostic differentiates a filtered field `ū`: [`AutoDerivative`](@ref), [`PhysicalDerivative`](@ref)
or [`SpectralDerivative`](@ref).
"""
abstract type AbstractDerivative end

"""
    AutoDerivative()

The exact derivative where the filter evaluates one, [`SpectralDerivative`](@ref) on a plan whose output is
a series or a ratio of two, and [`PhysicalDerivative`](@ref)`()` elsewhere.
"""
struct AutoDerivative <: AbstractDerivative end

"""
    PhysicalDerivative(; nodes = 5)

Differences of `ū`'s values: on a `StructuredGrid` FlowGeometries' stencils of `nodes` nodes, shrinking to a
cell's active run beside a mask; on any other grid FlowGeometries' least-squares gradient, which takes no
node count.
"""
struct PhysicalDerivative <: AbstractDerivative
    nodes::Int
end
PhysicalDerivative(; nodes::Integer = 5) = PhysicalDerivative(Int(nodes))

"""
    SpectralDerivative()

The derivative of the filter's own output: `∂_d ū = Σ_k i k_d Ĝ(k) F̂(k) e^{ik·x}` for a plan whose output
is that series, and the quotient rule for one whose output is a ratio of two series (a renormalized window
mass). A plan with neither raises.
"""
struct SpectralDerivative <: AbstractDerivative end

"""
    derivative_plan(grid; nodes = 5, execution = SerialBackend()) -> FG.Operators.StencilPlans | FG.Operators.GradientPlan

The derivative weights the diagnostics difference filtered fields with on `grid`: on a
`StructuredGrid`, `FlowGeometries.Operators.stencil_plans` with `nodes` nodes, where a window reaching an
inactive cell shrinks to the cell's active run (`ReduceInRun`); on any other grid, the least-squares
`FlowGeometries.Operators.gradient_plan`, which needs the grid's adjacency. The weights are built from
the host `grid` and held where `execution`'s loops read them.

The weights depend on the grid alone: build one per grid and pass it as `deriv_plan` to every diagnostic
taken there. Tasks may share it.
"""
derivative_plan(
    grid::FlowGeometries.Grids.AbstractGrid;
    nodes::Integer = 5,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
) = FlowGeometries.Execution.on_backend(execution, _host_derivative_plan(grid, Int(nodes)))

_host_derivative_plan(grid::FlowGeometries.Grids.StructuredGrid, nodes::Int) =
    FlowGeometries.Operators.stencil_plans(grid; nodes = nodes, policy = FlowGeometries.Operators.ReduceInRun())
_host_derivative_plan(grid::FlowGeometries.Grids.AbstractGrid, ::Int) = FlowGeometries.Operators.gradient_plan(grid)

"""
    ddz!(∂f∂z, f, grid, dz[, plan]; execution = SerialBackend()) -> ∂f∂z

The vertical derivative of a stack of levels, a 3-D field over a 2-D `grid`, the shape `filter_field!`
accepts. The grid describes the horizontal alone, so the vertical spacing `dz` is an argument. `plan` is
`FlowGeometries.Discretization.stencil_plan(range(0; step = dz, length = Nz), 1, 3)`, built once for
repeated calls. A masked horizontal cell is zero at every level. The fields live in the memory of
`execution`, whose loops difference them; a grid already in that memory is read there, and a host grid's
mask is copied there on each call.
"""
function ddz!(
    ∂f∂z::AbstractArray{T,3},
    f::AbstractArray{T,3},
    grid::FlowGeometries.Grids.StructuredGrid{T,FlowGeometries.Geometry.CartesianGeometry{T},2},
    dz::T,
    plan::FlowGeometries.Discretization.AbstractStencilPlan =
        FlowGeometries.Discretization.stencil_plan(range(zero(T); step = dz, length = size(f, 3)), 1,
                                                   min(3, size(f, 3)));
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
) where {T<:AbstractFloat}
    FlowGeometries.Operators.apply_stencil!(∂f∂z, f, plan, 3; masked = zero(T), execution = execution)
    return _blank_columns!(∂f∂z, FlowGeometries.Grids.mask(grid), execution)
end

# The mask is horizontal, so it blanks whole columns; an `AllActive` mask blanks none.
_blank_columns!(∂f∂z::AbstractArray{<:Any,3}, ::FlowGeometries.Grids.AllActive, _) = ∂f∂z
function _blank_columns!(∂f∂z::AbstractArray{T,3}, mask::AbstractMatrix{Bool}, execution) where {T}
    m = FlowGeometries.Execution.on_backend(execution, mask)
    @. ∂f∂z = ifelse(m, ∂f∂z, zero(T))
    return ∂f∂z
end

end # module
