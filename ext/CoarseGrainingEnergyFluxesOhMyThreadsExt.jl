module CoarseGrainingEnergyFluxesOhMyThreadsExt

using OhMyThreads: OhMyThreads
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF
using FlowGeometries: FlowGeometries

# Dynamic scheduling balances the uneven per-row cost that masking creates. Constructed as an object
# rather than passed as the `:dynamic` symbol: the symbol form resolves the scheduler type at runtime,
# which shows up as a dynamic dispatch on every parallel call in this file.
@inline _sched() = OhMyThreads.DynamicScheduler()

# Pass driver for the separable ND engine: same shape as `CGEF.Filtering._sep_serial`, so the passes
# and the normalization sweep parallelize without duplicating either kernel.
@inline _omt_driver(f::F, indices) where {F} =
    OhMyThreads.tforeach(f, indices; scheduler = _sched())

# ThreadedBackend: build the footprint once, then fill output rows in parallel. Rows write disjoint
# columns of the column-major output, so this is race-free and bit-identical to serial. Works for any
# row-decomposable 2D grid, structured or curvilinear, via the shared per-row kernel.
function CGEF.Filtering.threaded_filter_field!(
    out::AbstractMatrix{T},
    field::AbstractMatrix,
    grid::Union{FlowGeometries.Grids.StructuredGrid{G,T,2}, FlowGeometries.Grids.CurvilinearGrid{T,G}},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
    workspace,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    fp = workspace === nothing ? CGEF.Filtering.build_footprint(grid, kernel, scale; mask_strategy = mask_strategy) : workspace
    if fp isa CGEF.Filtering.SeparableGaussianFootprint
        return _threaded_apply_separable_gaussian!(out, field, grid, fp, mask_strategy)
    elseif fp isa CGEF.Filtering.PrefixSumTopHatPlan
        return _threaded_apply_prefixsum_tophat!(out, field, grid, fp, mask_strategy)
    end
    periodic_x = FlowGeometries.Grids.isperiodic(grid, 1)
    periodic_y = FlowGeometries.Grids.isperiodic(grid, 2)
    _, Ny = FlowGeometries.Grids.size_tuple(grid)
    fill!(out, zero(T))
    OhMyThreads.tforeach(1:Ny; scheduler = _sched()) do j
        CGEF.Filtering.apply_footprint_row!(out, field, grid, fp, mask_strategy, periodic_x, periodic_y, j)
    end
    return out
end

# Separable Gaussian. The column pass at row `j` reads `row_pass` across rows, so this cannot use the
# single fused row-parallel loop: two `tforeach` sweeps instead, one per pass, relying on `tforeach`'s
# implicit barrier to complete the whole `row_pass` before the column pass reads it. Calls the same
# per-row/per-column bodies and epilogue as serial, so results are bit-identical.
function _threaded_apply_separable_gaussian!(
    out::AbstractMatrix{T}, field::AbstractMatrix, grid::FlowGeometries.Grids.StructuredGrid, fp::CGEF.Filtering.SeparableGaussianFootprint{T}, strategy::CGEF.Filtering.AbstractMaskStrategy,
) where {T<:AbstractFloat}
    # Same contract as the serial path: the footprint's denominator is fixed at build time, so a
    # strategy it was not built for cannot be honoured and must not be silently ignored.
    CGEF.Filtering._separable_check_strategy(fp, strategy)
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    mask = FlowGeometries.Grids.mask(grid)
    @. fp.masked_input = T(mask) * field
    gx, gy = fp.gx, fp.gy
    di_lim, dj_lim = fp.di_lim, fp.dj_lim
    periodic_x, periodic_y = fp.periodic_x, fp.periodic_y
    row_pass = fp.row_pass
    masked_input = fp.masked_input
    OhMyThreads.tforeach(1:Ny; scheduler = _sched()) do j
        CGEF.Filtering._separable_row_pass_at!(row_pass, masked_input, gx, di_lim, periodic_x, Nx, j)
    end
    OhMyThreads.tforeach(1:Ny; scheduler = _sched()) do j
        CGEF.Filtering._separable_column_pass_at!(out, row_pass, gy, dj_lim, periodic_y, Nx, Ny, j)
    end
    CGEF.Filtering._separable_normalize_and_mask!(out, fp, mask, Nx, Ny)
    return out
end

# Prefix-sum top-hat. Every output row reads prefix sums of other rows in its band, so the table must
# be complete before phase 2 starts; `tforeach`'s barrier provides that. Within each phase row `j`
# writes only its own column of `prefix_num` / row of `out`, so both parallelize cleanly.
function _threaded_apply_prefixsum_tophat!(
    out::AbstractMatrix{T}, field::AbstractMatrix, grid::FlowGeometries.Grids.StructuredGrid,
    fp::CGEF.Filtering.PrefixSumTopHatPlan{T}, strategy::CGEF.Filtering.AbstractMaskStrategy,
) where {T<:AbstractFloat}
    _, Ny = FlowGeometries.Grids.size_tuple(grid)
    CGEF.Filtering._prefixsum_check_strategy(fp, strategy)
    OhMyThreads.tforeach(1:Ny; scheduler = _sched()) do j
        CGEF.Filtering.prefixsum_fill_numerator_row!(fp, field, grid, j)
    end
    OhMyThreads.tforeach(1:Ny; scheduler = _sched()) do j
        CGEF.Filtering.apply_prefixsum_tophat_row!(out, grid, fp, strategy, j)
    end
    return out
end

# Batched 2D apply: rows in parallel, and within a row the whole batch shares one enumeration of each
# point's neighbours. A per-field loop over the single-field hook would thread just as well but pay
# that enumeration `K` times, which for a streaming scattered footprint is the dominant cost.
# `SeparableGaussianFootprint` and `PrefixSumTopHatPlan` have no such shared derivation — their weight
# tables and support intervals are already computed once at plan-build time — so for them a per-field
# threaded apply is the whole of what batching would buy, and is what runs.
function CGEF.Filtering.threaded_filter_fields!(
    outs,
    fields,
    grid::Union{FlowGeometries.Grids.StructuredGrid{G,T,2}, FlowGeometries.Grids.CurvilinearGrid{T,G}},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
    workspace,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    fp = workspace === nothing ? CGEF.Filtering.build_footprint(grid, kernel, scale; mask_strategy = mask_strategy) : workspace
    if fp isa CGEF.Filtering.SeparableGaussianFootprint || fp isa CGEF.Filtering.PrefixSumTopHatPlan
        for k in eachindex(outs)
            CGEF.Filtering.threaded_filter_field!(outs[k], fields[k], grid, kernel, scale, mask_strategy, fp)
        end
        return outs
    end
    periodic_x = FlowGeometries.Grids.isperiodic(grid, 1)
    periodic_y = FlowGeometries.Grids.isperiodic(grid, 2)
    _, Ny = FlowGeometries.Grids.size_tuple(grid)
    for out in outs
        fill!(out, zero(T))
    end
    OhMyThreads.tforeach(1:Ny; scheduler = _sched()) do j
        CGEF.Filtering.apply_footprint_row_batch!(outs, fields, grid, fp, mask_strategy, periodic_x, periodic_y, j)
    end
    return outs
end

# Batched 1D/true-3D apply: point-indexed, so the parallel unit is a block of the index space rather
# than a row. Blocks are disjoint and each point writes only its own cell, so the result matches
# serial exactly.
function CGEF.Filtering.threaded_filter_fields!(
    outs,
    fields,
    grid::FlowGeometries.Grids.StructuredGrid{G,T,N},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
    workspace,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}, N}
    fp = workspace === nothing ? CGEF.Filtering.build_footprint(grid, kernel, scale; mask_strategy = mask_strategy) : workspace
    if fp isa CGEF.Filtering.SeparableGaussianFootprintND
        for k in eachindex(outs)
            CGEF.Filtering.threaded_filter_field!(outs[k], fields[k], grid, kernel, scale, mask_strategy, fp)
        end
        return outs
    end
    for out in outs
        fill!(out, zero(T))
    end
    blocks = OhMyThreads.chunks(CartesianIndices(FlowGeometries.Grids.size_tuple(grid)); n = Threads.nthreads())
    OhMyThreads.tforeach(blocks; scheduler = _sched()) do block
        CGEF.Filtering.apply_footprint_nd_batch_over!(outs, fields, grid, fp, mask_strategy, block)
    end
    return outs
end

# 1D/true-3D grids have point-indexed footprints and no row structure, so this parallelizes over
# `CartesianIndices(out)`. Each point writes only its own cell, and it reuses the serial per-point
# kernels, so the result is identical. `N` is unconstrained here; the 2D method above is more specific
# and wins for N=2.
function CGEF.Filtering.threaded_filter_field!(
    out::AbstractArray{T,N},
    field::AbstractArray,
    grid::FlowGeometries.Grids.StructuredGrid{G,T,N},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
    workspace,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}, N}
    fp = workspace === nothing ? CGEF.Filtering.build_footprint(grid, kernel, scale; mask_strategy = mask_strategy) : workspace
    mask = FlowGeometries.Grids.mask(grid)
    # Bound once, outside the branches: a name assigned in more than one branch of this function and
    # then captured by a closure is boxed, which OhMyThreads rejects outright.
    dims = FlowGeometries.Grids.size_tuple(grid)
    fill!(out, zero(T))
    if fp isa CGEF.Filtering.SeparableGaussianFootprintND
        # The passes are ordered in `d` — each reads across its predecessor's whole output — but every
        # point WITHIN a pass is independent, so the sweep is threaded and `tforeach`'s implicit
        # barrier separates the passes.
        return CGEF.Filtering.apply_separable_gaussian_nd!(out, field, grid, fp, mask_strategy, _omt_driver)
    elseif fp isa CGEF.Filtering.FilterFootprintND
        periodic = FlowGeometries.Grids.periodic_flags(grid)
        OhMyThreads.tforeach(CartesianIndices(out); scheduler = _sched()) do I
            mask[I] || return
            out[I] = CGEF.Filtering._footprint_nd_point(field, fp, mask_strategy, dims, periodic, mask, I)
        end
    elseif fp.cache !== nothing
        lin = LinearIndices(dims)
        cache = fp.cache
        OhMyThreads.tforeach(CartesianIndices(out); scheduler = _sched()) do I
            mask[I] || return
            out[I] = CGEF.Filtering._footprint_nd_point_cached(field, cache, mask_strategy, mask, lin, I)
        end
    else
        OhMyThreads.tforeach(CartesianIndices(out); scheduler = _sched()) do I
            mask[I] || return
            out[I] = CGEF.Filtering._footprint_nd_point_streaming(field, grid, fp, mask_strategy, mask, dims, I)
        end
    end
    return out
end


# Node sets are point-indexed with no row structure, so this parallelizes over `eachindex(out)` on
# the same argument as the 1D/true-3D method above: node `t` writes only `out[t]`. It reuses the
# serial per-node kernel, so the result is bit-identical to serial.
function CGEF.Filtering.threaded_filter_field!(
    out::AbstractVector{T},
    field::AbstractVector,
    grid::FlowGeometries.Grids.UnstructuredGrid{T},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
    workspace,
) where {T<:AbstractFloat}
    fp = workspace === nothing ? CGEF.Filtering.build_footprint(grid, kernel, scale; mask_strategy = mask_strategy) : workspace
    OhMyThreads.tforeach(eachindex(out); scheduler = _sched()) do t
        out[t] = CGEF.Filtering._footprint_node_point(field, grid, fp, mask_strategy, t)
    end
    return out
end

# Batched form: one pass over the nodes applying the shared neighbourhood to every field, so the
# adjacency walk is done once per node rather than once per node per field.
function CGEF.Filtering.threaded_filter_fields!(
    outs,
    fields,
    grid::FlowGeometries.Grids.UnstructuredGrid{T},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
    workspace,
) where {T<:AbstractFloat}
    fp = workspace === nothing ? CGEF.Filtering.build_footprint(grid, kernel, scale; mask_strategy = mask_strategy) : workspace
    OhMyThreads.tforeach(eachindex(first(outs)); scheduler = _sched()) do t
        for k in eachindex(outs)
            outs[k][t] = CGEF.Filtering._footprint_node_point(fields[k], grid, fp, mask_strategy, t)
        end
    end
    return outs
end


# Slice-parallel apply. Slices are independent and each writes only its own output, so this needs no
# synchronization; the inner apply is forced serial so the two levels of threading never nest.
#
# Slices are dispatched LONGEST-FIRST under a dynamic scheduler. Per-slice cost grows at least
# linearly in the point count and the counts are ragged in practice, so equal-count chunking leaves
# one worker holding the largest slice after the others have drained. Longest-processing-time-first
# is the standard remedy and bounds the makespan at 4/3 of optimal.
function CGEF.Filtering.threaded_filter_slices!(outs, fields, plans)
    order = sortperm(CGEF.Filtering.slice_costs(plans); rev = true)
    OhMyThreads.tforeach(order; scheduler = OhMyThreads.DynamicScheduler()) do t
        CGEF.Filtering.apply_slice_serial!(outs[t], fields[t], plans[t])
    end
    return outs
end

# Batch-parallel PIPELINE sweeps over one shared grid — the whole sweep per slice, not just the filter
# step. Slices write disjoint views of the batched result, so this needs no synchronization, and the
# inner sweep is forced serial so the two levels of threading never nest.
#
# Scratch is pooled per WORKER: chunk the flat slice list and let each chunk own pool entry `ci`, so
# pool memory is bounded by thread count rather than by batch length. No longest-first ordering here —
# one shared grid makes every slice the same cost, unlike `threaded_coarse_grain_slices!`.
function CGEF.Pipeline.threaded_coarse_grain_batch!(batch, u, v, w, grid, valR, ctx)
    n = length(batch.slices)
    groups = OhMyThreads.chunks(1:n; n = CGEF.Pipeline.batch_concurrency(ctx, n))
    OhMyThreads.tforeach(enumerate(groups); scheduler = _sched()) do (ci, group)
        for t in group
            CGEF.Pipeline.coarse_grain_batch_slice!(batch, u, v, w, grid, valR, ctx, t, ci)
        end
    end
    return batch
end

# Batch-parallel PIPELINE sweeps whose slices carry their own grids. Scratch is per slice here — a
# workspace is grid-shaped and cannot be reused across differing shapes — so slices index it directly
# and no chunk mapping is needed.
#
# Dispatched LONGEST-FIRST under a dynamic scheduler: ragged point counts mean equal chunking leaves
# one worker holding the largest sweep after the others have drained.
function CGEF.Pipeline.threaded_coarse_grain_slices!(results, us, vs, ws, grids, ctx)
    order = sortperm(CGEF.Pipeline.slice_pipeline_costs(grids, length(ctx.scales)); rev = true)
    OhMyThreads.tforeach(order; scheduler = _sched()) do t
        CGEF.Pipeline.coarse_grain_slice_serial!(results, us, vs, ws, grids, ctx, t)
    end
    return results
end

end # module
