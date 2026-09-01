# ---------------------------------------------------------------------------
# Public Filtering API
# ---------------------------------------------------------------------------

"""
    filter_field!(out, field, grid, kernel, scale; mask_strategy=ZeroFill(), filter_plan=nothing, backend=AutoBackend())

Filter a field on a grid using `kernel` at characteristic full width `scale` (ℓ), writing the
result to `out` (returned).

# Keyword Arguments
- `mask_strategy::AbstractMaskStrategy=ZeroFill()`: masking strategy — `ZeroFill()` (excluded cells count
  in the denominator as zero; the kernel stays homogeneous) or `Deformable()` (excluded cells dropped
  from numerator and denominator; renormalized over the locally-included area).

  Near a boundary the footprint is truncated, and both strategies inherit the same shape distortion
  from that: measured on a straight coast, Gaussian at `ℓ = 16` cells, one cell inshore the footprint's
  centroid sits 0.21ℓ offshore and its width is 62% of the interior value (75% at `ℓ/4` inshore, 90% at
  `ℓ/2`, 100% at `ℓ`). Points within `≈ℓ` of a boundary are contaminated either way.

  They differ on the footprint's **mass**:

  - `ZeroFill` leaves it at the truncated value, so the kernel is position-independent and filtering
    **commutes with spatial derivatives** — the step the flux budget is derived by. A uniform field
    ≡ 1 then filters to 0.543 one cell from the coast, 0.948 at `ℓ/2`, 0.9996 at `ℓ`.
  - `Deformable` divides it out, so a constant filters to 1.000 everywhere, at the cost of a
    position-dependent kernel, which does not commute with derivatives.

  `ZeroFill` is the default because the flux budget is a statement about commuting operators; prefer
  `Deformable` when a locally unbiased amplitude near a coast matters more than a budget that closes.

  Neither conserves the ACTIVE-cell integral on a masked domain, and they fail differently. `ZeroFill`
  conserves it over the whole domain exactly (7e-18 relative, unmasked periodic) but smears part of it
  onto masked cells, which report zero; `Deformable` renormalizes that away and tracks the active-cell
  integral better — 4.8e-4 relative drift against `ZeroFill`'s 1.1e-2 on a masked periodic grid. On a
  bounded grid the domain edge costs both about 1e-2 at `ℓ = 6Δx`. A closed energy budget wants a
  periodic unmasked domain; otherwise expect an `O(ℓ/L)` boundary residual.
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
