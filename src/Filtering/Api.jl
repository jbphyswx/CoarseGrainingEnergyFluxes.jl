# ---------------------------------------------------------------------------
# Public Filtering API
# ---------------------------------------------------------------------------

"""
    filter_field!(out, field, grid, kernel, scale; mask_strategy = ZeroFill(), filter_plan = nothing,
                  execution = AutoBackend(), method = RealSpace(), transform = AutoSpectralBackend())

Filter a field on a grid using `kernel` at characteristic full width `scale` (ℓ), writing the
result to `out` (returned).

# Keyword Arguments
- `mask_strategy::AbstractMaskStrategy=ZeroFill()`: how masked cells and a bounded edge enter the
  filter. [`ZeroFill`](@ref) filters the field extended by zero over masked cells and past each bounded
  edge, normalized by the kernel's full mass over the lattice continued at its edge spacing, so
  filtering commutes with spatial derivatives, conserves the integral over all space, and defines the
  output at masked cells too; a uniform field falls toward a coast. [`Deformable`](@ref) normalizes
  over the active in-domain cells of each window and zeroes masked cells, so a constant is reproduced
  next to a boundary and the kernel changes shape there.
- `filter_plan::Union{Nothing,AbstractFilterPlan}=nothing`: a prebuilt [`plan_filter`](@ref) result,
  reused across calls on the same grid, kernel and scale. When supplied, `mask_strategy`, `execution`,
  `method` and `transform` are ignored.
- `execution::AbstractExecutionBackend=AutoBackend()`: execution backend (SerialBackend,
  ThreadedBackend, GPUBackend, …).
- `method`, `transform`: as for [`plan_filter`](@ref).

For spherical grids the longitude footprint wraps only when the grid is periodic (`isperiodic`);
distances use the great-circle (Haversine) metric.

# Examples
```julia
geom = CartesianGeometry()
grid = StructuredGrid(geom, 0.0:1000.0:99_000.0, 0.0:1000.0:99_000.0, mask)
out = zeros(100, 100)
filter_field!(out, field, grid, TopHatKernel(), 5000.0; mask_strategy = Deformable())
```
"""
function filter_field!(
    out::AbstractArray{T},
    field::AbstractArray,
    grid::FlowGeometries.Grids.AbstractGrid,
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    filter_plan::Union{Nothing,AbstractFilterPlan} = nothing,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    method::AbstractFilterMethod = RealSpace(),
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
) where {T<:AbstractFloat}
    plan = filter_plan === nothing ?
        plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, execution = execution, method = method,
                    transform = transform) : filter_plan
    return filter_apply!(out, field, plan)
end

# A 3D field on a 2D grid, filtered layer by layer along its third axis with one plan.
function filter_field!(
    out::AbstractArray{T,3},
    field::AbstractArray{<:Any,3},
    grid::FlowGeometries.Grids.StructuredGrid{T,<:FlowGeometries.Geometry.AbstractGeometry,2},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    filter_plan::Union{Nothing,AbstractFilterPlan} = nothing,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    method::AbstractFilterMethod = RealSpace(),
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
) where {T<:AbstractFloat}
    plan = filter_plan === nothing ?
        plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, execution = execution, method = method,
                    transform = transform) : filter_plan
    for k in axes(field, 3)
        filter_apply!(view(out, :, :, k), view(field, :, :, k), plan)
    end
    return out
end
