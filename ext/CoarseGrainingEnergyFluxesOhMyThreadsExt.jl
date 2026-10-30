module CoarseGrainingEnergyFluxesOhMyThreadsExt

using OhMyThreads: OhMyThreads
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF
using FlowGeometries: FlowGeometries

CGEF.Filtering._available(::CGEF.Filtering._Threads) = true
using FlowTransformBindings: FlowTransformBindings as FTB
using LinearAlgebra: LinearAlgebra

# A scheduler object of concrete type, so every `tforeach` call below resolves its scheduler statically.
const _SCHED = OhMyThreads.DynamicScheduler()
@inline _sched() = _SCHED
# One task per slice, spawned in the order given: slices sorted by descending cost are taken largest
# first as threads come free.
const _SLICE_SCHED = OhMyThreads.DynamicScheduler(; chunking = false)

# Driver for the separable ND engine and the transform engines, with the signature of
# `CGEF.Filtering._sep_serial`.
@inline _omt_driver(f::F, indices) where {F} =
    OhMyThreads.tforeach(f, indices; scheduler = _sched())

# A process's share of a real-space apply on a threaded inner backend: its units in one chunk per thread.
CGEF.Filtering._share_run(f::F, ::CGEF.ComputationalBackends.AbstractThreadedBackend, units) where {F} =
    OhMyThreads.tforeach(f, OhMyThreads.chunks(units; n = Threads.nthreads()); scheduler = _sched())

# Row `j` writes only `out[:, j]`, so the parallel row loop is race-free and bit-identical to serial.
function CGEF.Filtering.threaded_filter_field!(
    out::AbstractMatrix{T},
    field::AbstractMatrix,
    grid::Union{FlowGeometries.Grids.StructuredGrid{T,G,2}, FlowGeometries.Grids.CurvilinearGrid{T,G}},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
    workspace,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    fp = workspace === nothing ? CGEF.Filtering.build_footprint(grid, kernel, scale; mask_strategy = mask_strategy) : workspace
    CGEF.Filtering._check_strategy(fp, mask_strategy)
    if fp isa CGEF.Filtering.SeparableFootprint
        return _threaded_apply_separable!(out, field, grid, fp, mask_strategy)
    elseif fp isa CGEF.Filtering.PrefixSumTopHatPlan
        return _threaded_apply_prefixsum_tophat!(out, field, grid, fp, mask_strategy)
    elseif CGEF.Filtering._driven_footprint(fp)
        return CGEF.Filtering.apply_footprint!(out, field, grid, fp, mask_strategy, _omt_driver)
    end
    periodic_x = FlowGeometries.Grids.isperiodic(grid, 1)
    periodic_y = FlowGeometries.Grids.isperiodic(grid, 2)
    _, Ny = FlowGeometries.Grids.size_tuple(grid)
    fill!(out, zero(T))
    CGEF.Filtering.prepare_row_apply!(fp, field, grid)
    # One query buffer per task, over that task's block of rows.
    OhMyThreads.tforeach(OhMyThreads.chunks(1:Ny; n = Threads.nthreads()); scheduler = _sched()) do rows
        sc = CGEF.Filtering.row_scratch(fp)
        for j in rows
            CGEF.Filtering.apply_footprint_row!(out, field, grid, fp, mask_strategy, periodic_x, periodic_y, j, sc)
        end
    end
    return out
end

# Separable Gaussian. The column pass reads `row_pass` across rows; the first `tforeach` returns only
# once every row of it is written.
function _threaded_apply_separable!(
    out::AbstractMatrix{T}, field::AbstractMatrix, grid::FlowGeometries.Grids.StructuredGrid, fp::CGEF.Filtering.SeparableFootprint{T}, strategy::CGEF.Filtering.AbstractMaskStrategy,
) where {T<:AbstractFloat}
    CGEF.Filtering._separable_check_strategy(fp, strategy)
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    mask = FlowGeometries.Grids.mask(grid)
    @. fp.masked_input = mask * field
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

# Prefix-sum top-hat. An output row reads the prefix sums of other rows in its band, so the first
# `tforeach` completes the table before the second starts. Within a phase, row `j` writes only its own.
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

# Batched form: phase 1 fills every field's numerator scan, phase 2 walks each support interval once for
# all fields. The numerator slots are grown before the parallel region, since growing them mutates the
# shared scratch.
function _threaded_apply_prefixsum_tophat_batch!(
    outs, fields, grid::FlowGeometries.Grids.StructuredGrid,
    fp::CGEF.Filtering.PrefixSumTopHatPlan{T}, strategy::CGEF.Filtering.AbstractMaskStrategy,
) where {T<:AbstractFloat}
    if !CGEF.Filtering._prefixsum_batch_fuses(fp)
        for k in eachindex(outs)
            _threaded_apply_prefixsum_tophat!(outs[k], fields[k], grid, fp, strategy)
        end
        return outs
    end
    _, Ny = FlowGeometries.Grids.size_tuple(grid)
    CGEF.Filtering._prefixsum_check_strategy(fp, strategy)
    CGEF.Filtering._prefixsum_numerators!(fp.scratch, length(fields))
    OhMyThreads.tforeach(1:Ny; scheduler = _sched()) do j
        CGEF.Filtering.prefixsum_fill_numerator_batch_row!(fp, fields, grid, j)
    end
    OhMyThreads.tforeach(1:Ny; scheduler = _sched()) do j
        CGEF.Filtering.apply_prefixsum_tophat_batch_row!(outs, grid, fp, strategy, j)
    end
    return outs
end

# Batched 2-D apply: rows in parallel; within a row, each point's neighbours are enumerated once for the
# whole batch.
function CGEF.Filtering.threaded_filter_fields!(
    outs,
    fields,
    grid::Union{FlowGeometries.Grids.StructuredGrid{T,G,2}, FlowGeometries.Grids.CurvilinearGrid{T,G}},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
    workspace,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    fp = workspace === nothing ? CGEF.Filtering.build_footprint(grid, kernel, scale; mask_strategy = mask_strategy) : workspace
    CGEF.Filtering._check_strategy(fp, mask_strategy)
    if fp isa CGEF.Filtering.PrefixSumTopHatPlan
        return _threaded_apply_prefixsum_tophat_batch!(outs, fields, grid, fp, mask_strategy)
    end
    CGEF.Filtering._driven_footprint(fp) &&
        return CGEF.Filtering._driven_batch!(outs, fields, grid, fp, mask_strategy, _omt_driver)
    if fp isa CGEF.Filtering.SeparableFootprint
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
    CGEF.Filtering.prepare_row_apply_batch!(fp, fields, grid)
    OhMyThreads.tforeach(OhMyThreads.chunks(1:Ny; n = Threads.nthreads()); scheduler = _sched()) do rows
        sc = CGEF.Filtering.row_scratch(fp)
        for j in rows
            CGEF.Filtering.apply_footprint_row_batch!(outs, fields, grid, fp, mask_strategy, periodic_x, periodic_y, j, sc)
        end
    end
    return outs
end

# 3-D prefix-sum top-hat, two phases as in 2-D: an output `(j, k)` plane reads the axis-1 scans of other
# planes in its band, and within a phase each plane writes only its own.
function _threaded_apply_prefixsum_tophat_3d!(
    out::AbstractArray{T,3}, field::AbstractArray, grid::FlowGeometries.Grids.StructuredGrid,
    fp::CGEF.Filtering.PrefixSumTopHat3DPlan{T}, strategy::CGEF.Filtering.AbstractMaskStrategy,
) where {T<:AbstractFloat}
    _, Ny, Nz = FlowGeometries.Grids.size_tuple(grid)
    sc = fp.scratch
    planes = CartesianIndices((Ny, Nz))
    OhMyThreads.tforeach(planes; scheduler = _sched()) do JK
        CGEF.Filtering._prefixsum3d_fill_plane!(sc, field, grid, fp.masked, JK[1], JK[2])
    end
    OhMyThreads.tforeach(planes; scheduler = _sched()) do JK
        CGEF.Filtering._prefixsum3d_out_plane!(out, fp, JK[1], JK[2])
    end
    return out
end

# Batched 1-D / 3-D apply over disjoint blocks of the index space; each point writes only its own cell,
# so the result matches serial.
function CGEF.Filtering.threaded_filter_fields!(
    outs,
    fields,
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
    workspace,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}, N}
    fp = workspace === nothing ? CGEF.Filtering.build_footprint(grid, kernel, scale; mask_strategy = mask_strategy) : workspace
    CGEF.Filtering._check_strategy(fp, mask_strategy)
    CGEF.Filtering._driven_footprint(fp) &&
        return CGEF.Filtering._driven_batch!(outs, fields, grid, fp, mask_strategy, _omt_driver)
    # These engines pass every field through the plan's one set of pass/scan/source buffers, so they apply
    # per field.
    if fp isa Union{CGEF.Filtering.SeparableFootprintND, CGEF.Filtering.PrefixSumTopHat3DPlan,
                    CGEF.Filtering.FilterFootprintND}
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

# 1-D / 3-D grids, in parallel over points; each point writes only its own cell. The 2-D method above is
# more specific and takes `N == 2`.
function CGEF.Filtering.threaded_filter_field!(
    out::AbstractArray{T,N},
    field::AbstractArray,
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
    workspace,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}, N}
    fp = workspace === nothing ? CGEF.Filtering.build_footprint(grid, kernel, scale; mask_strategy = mask_strategy) : workspace
    CGEF.Filtering._check_strategy(fp, mask_strategy)
    mask = FlowGeometries.Grids.mask(grid)
    # Bound once, outside the branches: a name assigned in more than one branch of this function and
    # then captured by a closure is boxed, which OhMyThreads rejects outright.
    dims = FlowGeometries.Grids.size_tuple(grid)
    fill!(out, zero(T))
    if fp isa CGEF.Filtering.SeparableFootprintND
        # Pass `d` reads the whole output of pass `d - 1`; the points of one pass are independent, and
        # each `tforeach` returns before the next pass starts.
        return CGEF.Filtering.apply_separable_nd!(out, field, grid, fp, mask_strategy, _omt_driver)
    elseif fp isa CGEF.Filtering.PrefixSumTopHat3DPlan
        return _threaded_apply_prefixsum_tophat_3d!(out, field, grid, fp, mask_strategy)
    elseif CGEF.Filtering._driven_footprint(fp)
        return CGEF.Filtering.apply_footprint!(out, field, grid, fp, mask_strategy, _omt_driver)
    elseif fp isa CGEF.Filtering.FilterFootprintND
        # Each line along axis 1 writes only itself.
        src = CGEF.Filtering._nd_source!(fp, field, grid)
        OhMyThreads.tforeach(CartesianIndices(Base.tail(dims)); scheduler = _sched()) do J
            CGEF.Filtering._nd_out_line!(out, src, fp, Tuple(J))
        end
    elseif fp.cache !== nothing
        lin = LinearIndices(dims)
        cache = fp.cache
        exterior = fp.exterior
        OhMyThreads.tforeach(CartesianIndices(out); scheduler = _sched()) do I
            out[I] = CGEF.Filtering._footprint_nd_point_cached(field, cache, exterior, mask_strategy, mask, lin, I)
        end
    else
        OhMyThreads.tforeach(CartesianIndices(out); scheduler = _sched()) do I
            out[I] = CGEF.Filtering._footprint_nd_point_streaming(field, grid, fp, mask_strategy, mask, I)
        end
    end
    return out
end


# The node CSR engine on any flat-cell grid, selected by the `NodeFilterPlan` slot; cell `t` writes only
# `out[t]`. A 1-D `StructuredGrid` also has a vector output; its ND footprint takes the method above.
function CGEF.Filtering.threaded_filter_field!(
    out::AbstractVector{T},
    field::AbstractVector,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
    fp::CGEF.Filtering.NodeFilterPlan,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    OhMyThreads.tforeach(eachindex(out); scheduler = _sched()) do s
        CGEF.Filtering._node_stage!(fp, field, grid, s)
    end
    vals, act = CGEF.Filtering._node_inputs(fp, field, grid)
    OhMyThreads.tforeach(eachindex(out); scheduler = _sched()) do s
        out[CGEF.Filtering._node_id(fp, s)] = CGEF.Filtering._footprint_node_point(field, vals, act, fp, mask_strategy, s)
    end
    return out
end

function CGEF.Filtering.threaded_filter_fields!(
    outs,
    fields,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
    fp::CGEF.Filtering.NodeFilterPlan,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    for k in eachindex(outs)
        CGEF.Filtering.threaded_filter_field!(outs[k], fields[k], grid, kernel, scale, mask_strategy, fp)
    end
    return outs
end

# The host engines of a node set, through their drivers.
const _NodeDriven = Union{CGEF.Filtering.BinnedTopHatPlan, CGEF.Filtering.NUFFTSumFootprint}

CGEF.Filtering.threaded_filter_field!(
    out::AbstractVector{T}, field::AbstractVector, grid::FlowGeometries.Grids.UnstructuredGrid{T},
    kernel::CGEF.Kernels.AbstractFilterKernel, scale::T, mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
    fp::_NodeDriven,
) where {T<:AbstractFloat} = CGEF.Filtering.apply_footprint!(out, field, grid, fp, mask_strategy, _omt_driver)

CGEF.Filtering.threaded_filter_fields!(
    outs, fields, grid::FlowGeometries.Grids.UnstructuredGrid{T},
    kernel::CGEF.Kernels.AbstractFilterKernel, scale::T, mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
    fp::_NodeDriven,
) where {T<:AbstractFloat} = CGEF.Filtering._driven_batch!(outs, fields, grid, fp, mask_strategy, _omt_driver)


# Runs `f` with FastTransforms on one OpenMP thread and BLAS on one thread, for tasks that each run a whole
# serial apply or sweep. BLAS's count is process-global, restored after.
function _farm(f)
    prev = LinearAlgebra.BLAS.get_num_threads()
    try
        LinearAlgebra.BLAS.set_num_threads(1)
        return Base.ScopedValues.with(f, FTB.FASTTRANSFORMS_THREADS => 1)
    finally
        LinearAlgebra.BLAS.set_num_threads(prev)
    end
end

# Slice-parallel apply. Each slice runs serially and writes only its own output.
function CGEF.Filtering.threaded_filter_slices!(outs, fields, plans)
    order = sortperm(CGEF.Filtering.slice_costs(plans); rev = true)
    _farm() do
        OhMyThreads.tforeach(order; scheduler = _SLICE_SCHED) do t
            CGEF.Filtering.apply_slice_serial!(outs[t], fields[t], plans[t])
        end
    end
    return outs
end

# Batch-parallel pipeline sweeps over one shared grid; slices write disjoint views of the batched result.
# The tables are built once, their rows divided among the threads; chunk `ci` owns scratch pool entry `ci`, a worker
# copy of the base built in its task when the caller supplied none, so its buffers are first touched by the thread
# that uses them.
function CGEF.Pipeline.threaded_coarse_grain_batch!(batch, u, v, w, grid, valR, ctx)
    n = length(batch.slices)
    n == 0 && return batch
    groups = OhMyThreads.chunks(1:n; n = CGEF.Pipeline.batch_concurrency(ctx, n))
    _farm() do
        base = Base.ScopedValues.with(CGEF.Filtering._BATCH_BUILD => CGEF.ComputationalBackends.ThreadedBackend()) do
            CGEF.Pipeline._batch_base(batch, u, v, w, grid, valR, ctx, 1)
        end
        OhMyThreads.tforeach(enumerate(groups); scheduler = _sched()) do (ci, group)
            _run_worker!(batch, u, v, w, grid, valR, ctx, group, base, ci)
        end
    end
    return batch
end

# Worker 1 sweeps on the base and its rate route; any other on its copy, its rate route built from its first slice.
_run_worker!(batch, u, v, w, grid, valR, ctx, group, base, ci::Int) = ci == 1 ?
    CGEF.Pipeline._run_slices!(batch, u, v, w, grid, valR, ctx, group, base.scratch, base.rate) :
    CGEF.Pipeline._run_slices!(batch, u, v, w, grid, valR, ctx, group, CGEF.Pipeline._worker_scratch(ctx, base, grid, ci))

# Pipeline sweeps whose slices carry their own grids, each with its own scratch.
function CGEF.Pipeline.threaded_coarse_grain_slices!(results, us, vs, ws, grids, ctx)
    order = sortperm(CGEF.Pipeline.slice_pipeline_costs(grids, length(ctx.scales)); rev = true)
    _farm() do
        OhMyThreads.tforeach(order; scheduler = _SLICE_SCHED) do t
            CGEF.Pipeline.coarse_grain_slice_serial!(results, us, vs, ws, grids, ctx, t)
        end
    end
    return results
end

end # module
