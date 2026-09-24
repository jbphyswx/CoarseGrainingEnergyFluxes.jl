# ---------------------------------------------------------------------------
# Public Filtering API
# ---------------------------------------------------------------------------

"""
    filter_field!(out, field, grid, kernel, scale; mask_strategy=ZeroFill(), filter_plan=nothing, backend=AutoBackend())

Filter a field on a grid using `kernel` at characteristic full width `scale` (ℓ), writing the
result to `out` (returned).

# Keyword Arguments
- `mask_strategy::AbstractMaskStrategy=ZeroFill()`: how masked cells and a bounded edge enter the
  filter. [`ZeroFill`](@ref) filters the field extended by zero over masked cells and past each bounded
  edge, normalized by the kernel's full mass over the lattice continued at its edge spacing; the kernel
  is position-independent, so filtering **commutes with spatial derivatives** — the step the flux
  budget is derived by — and conserves the integral over all space (over the grid itself on a periodic
  grid, masked or not), and the output is defined at masked cells too. [`Deformable`](@ref) sums and normalizes over the active in-domain cells of each window
  and zeroes masked cells, so a constant is reproduced next to a boundary and the kernel changes shape
  there.

  Near a coast the two see the same footprint shape, displaced offshore and narrowed within `≈ℓ`, and
  differ only in its mass: `ZeroFill` keeps the active part of the full mass, so a uniform field falls
  toward the coast, and `Deformable` divides it out. `ZeroFill` is the default because the flux budget
  is a statement about commuting operators; prefer `Deformable` when a locally unbiased amplitude near
  a coast matters more than a budget that closes.
- `filter_plan::Union{Nothing,AbstractFilterPlan}=nothing`: a prebuilt [`plan_filter`](@ref) result to
  reuse instead of building one from scratch — the zero-(re)allocation entry point for a repeated
  sweep (many timesteps/fields over the same grid/kernel/scale). When supplied, `mask_strategy`/
  `backend`/`method` are ignored (already baked into the plan); build it once with `plan_filter` and
  pass it here on every subsequent call.
- `backend::AbstractExecutionBackend=AutoBackend()`: execution backend (SerialBackend,
  ThreadedBackend, GPUBackend, …). Ignored when `filter_plan` is supplied.

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
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    method::AbstractFilterMethod = RealSpace(),
) where {T<:AbstractFloat}
    plan = filter_plan === nothing ?
        plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend, method = method) : filter_plan
    return filter_apply!(out, field, plan)
end

# 3D volume filtering: horizontal filtering layer by layer. The plan is built once and reused across
# every layer on every backend, which is why this dispatches here rather than recursing into the 2D
# method per layer with no plan to pass.
function filter_field!(
    out::AbstractArray{T,3},
    field::AbstractArray{<:Any,3},
    grid::FlowGeometries.Grids.StructuredGrid{T,<:FlowGeometries.Geometry.AbstractGeometry,2},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    filter_plan::Union{Nothing,AbstractFilterPlan} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    method::AbstractFilterMethod = RealSpace(),
) where {T<:AbstractFloat}
    plan = filter_plan === nothing ?
        plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend, method = method) : filter_plan
    for k in axes(field, 3)
        filter_apply!(view(out, :, :, k), view(field, :, :, k), plan)
    end
    return out
end
