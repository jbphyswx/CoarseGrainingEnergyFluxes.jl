module Derivatives

using FlowGeometries: FlowGeometries
using ComputationalBackends: ComputationalBackends

"""
    derivative_plan(grid; execution = SerialBackend()) -> FG.Operators.StencilPlans | FG.Operators.GradientPlan

The derivative weights the diagnostics difference filtered fields with on `grid`: on a
`StructuredGrid`, `FlowGeometries.Operators.stencil_plans` with 3 nodes, where a window reaching an
inactive cell shrinks to the cell's active run (`ReduceInRun`); on any other grid, the least-squares
`FlowGeometries.Operators.gradient_plan`, which needs the grid's adjacency. The weights are built from
the host `grid` and held where `execution`'s loops read them.

The weights depend on the grid alone: build one per grid and pass it as `deriv_plan` to every diagnostic
taken there. Tasks may share it.
"""
derivative_plan(
    grid::FlowGeometries.Grids.AbstractGrid;
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
) = FlowGeometries.Execution.on_backend(execution, _host_derivative_plan(grid))

_host_derivative_plan(grid::FlowGeometries.Grids.StructuredGrid) =
    FlowGeometries.Operators.stencil_plans(grid; nodes = 3, policy = FlowGeometries.Operators.ReduceInRun())
_host_derivative_plan(grid::FlowGeometries.Grids.AbstractGrid) = FlowGeometries.Operators.gradient_plan(grid)

"""
    ddz!(∂f∂z, f, grid, dz[, plan]; execution = SerialBackend()) -> ∂f∂z

The vertical derivative of a stack of levels, a 3-D field over a 2-D `grid`, the shape `filter_field!`
accepts. The grid describes the horizontal alone, so the vertical spacing `dz` is an argument. `plan` is
`FlowGeometries.Discretization.stencil_plan(range(0; step = dz, length = Nz), 1, 3)`, built once for
repeated calls. A masked horizontal cell is zero at every level. The fields live in the memory of
`execution`, whose loops difference them.
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
    # The mask is horizontal, so it blanks whole columns.
    mask = FlowGeometries.Execution.on_backend(execution, FlowGeometries.Grids.mask(grid))
    @. ∂f∂z = ifelse(mask, ∂f∂z, zero(T))
    return ∂f∂z
end

end # module
