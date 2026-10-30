# ---------------------------------------------------------------------------
# Shares of an apply, for the backends that divide one field among processes
# ---------------------------------------------------------------------------
#
# A share is block `k` of `nk` of one pass's output units: rows of a 2-D grid, lines along axis 1,
# `(j, k)` planes or points of an N-D one, nodes of a node set. Every unit writes its own outputs alone. A
# pass after the first reads the whole output of the one before, so a backend completes every share of a
# pass before any share of the next. A share runs its units on the process's inner backend through
# `_share_run`.

"""
    _share_passes(fp) -> Val(n)
    _share_pass!(out, field, grid, fp, strategy, ::Val{p}, k, nk, inner) -> nothing
    _share_dst(out, fp, ::Val{p}) -> Tuple of arrays
    _share_local(fp, ::Val{p}) -> Bool

The `n` passes of `fp`'s apply; share `k` of `nk` of pass `p`, run on `inner`, reading the whole `field`
(and the whole output of pass `p − 1`) and writing its units of `out` or of `fp`'s pass buffers; the arrays
pass `p` writes, which a backend holding a separate memory per process assembles after the pass; and
whether pass `p` costs less to run whole on every process than to assemble, `O(N)` work with nothing to
exchange before it.
"""
function _share_pass! end

_share_local(_, ::Val) = false

# Block `k` of `nk` of `1:n`, contiguous, the blocks' lengths differing by at most one.
@inline _share_block(k::Int, nk::Int, n::Int) = (div((k - 1) * n, nk) + 1):div(k * n, nk)

"""
    AbstractProcessShare
    _share_workspace(inner, grid, fp) -> footprint or share
    _process_passes(inner, fp) -> Val(n)
    _share_staging(share) -> host array or nothing

What a process of a process backend holds to run its shares on `inner`, and the passes it runs: the host
footprint itself with its passes on a host backend; on a device backend the share the device's extension
builds (an `AbstractProcessShare`), whose one pass runs the engine on the device. `_share_staging` is the
host buffer a share's output passes through when it is reduced across processes.
"""
abstract type AbstractProcessShare end

_share_workspace(::ComputationalBackends.AbstractExecutionBackend, _, fp) = fp
_process_passes(::ComputationalBackends.AbstractExecutionBackend, fp) = _share_passes(fp)
_share_staging(_) = nothing

"""
    _share_run(f, inner, units)

Calls `f(chunk)` for chunks covering `units` on the inner backend, each chunk one task with its own
scratch: one chunk under a serial backend; the OhMyThreads extension splits them among threads.
"""
_share_run(f::F, ::ComputationalBackends.AbstractSerialBackend, units) where {F} = (f(units); nothing)
_share_run(_, inner::ComputationalBackends.AbstractExecutionBackend, _) = throw(ArgumentError(
    "a process's share of a real-space filter runs on $(nameof(typeof(inner))) through that backend's " *
    "extension: `using OhMyThreads` for a threaded inner backend"))

# The backend a plan's build divides its independent rows among, through `_share_run`: a threaded plan's threads,
# one chunk for any other plan.
_build_share(b::ComputationalBackends.AbstractThreadedBackend) = b
_build_share(::ComputationalBackends.AbstractExecutionBackend) = ComputationalBackends.SerialBackend()

# ---- 2-D rows: the banded and cell-by-cell footprints, one pass -----------------------------------

const _RowFootprint = Union{FilterFootprint, ScatteredFilterPlan}

_share_passes(::_RowFootprint) = Val(1)
_share_dst(out, ::_RowFootprint, ::Val{1}) = (out,)

function _share_pass!(
    out, field, grid, fp::_RowFootprint, strategy::AbstractMaskStrategy, ::Val{1}, k::Int, nk::Int, inner,
)
    prepare_row_apply!(fp, field, grid)
    px, py = FlowGeometries.Grids.isperiodic(grid, 1), FlowGeometries.Grids.isperiodic(grid, 2)
    _share_run(inner, _share_block(k, nk, size(out, 2))) do rows
        sc = row_scratch(fp)
        for j in rows
            apply_footprint_row!(out, field, grid, fp, strategy, px, py, j, sc)
        end
        return nothing
    end
    return nothing
end

# ---- 2-D separable: row pass into `fp.row_pass`, then column pass and normalization into `out` -------

_share_passes(::SeparableFootprint) = Val(2)
_share_dst(_, fp::SeparableFootprint, ::Val{1}) = (fp.row_pass,)
_share_dst(out, ::SeparableFootprint, ::Val{2}) = (out,)

function _share_pass!(
    out, field, grid, fp::SeparableFootprint{T}, ::AbstractMaskStrategy, ::Val{1}, k::Int, nk::Int, inner,
) where {T}
    Nx, Ny = size(out)
    mask = FlowGeometries.Grids.mask(grid)
    src, row_pass = fp.masked_input, fp.row_pass
    _share_run(inner, _share_block(k, nk, Ny)) do rows
        @inbounds for j in rows
            for i in 1:Nx
                src[i, j] = mask[i, j] * field[i, j]   # a `Bool` strong zero
            end
            _separable_row_pass_at!(row_pass, src, fp.gx, fp.di_lim, fp.periodic_x, Nx, j)
        end
        return nothing
    end
    return nothing
end

function _share_pass!(
    out, _, grid, fp::SeparableFootprint{T}, ::AbstractMaskStrategy, ::Val{2}, k::Int, nk::Int, inner,
) where {T}
    Nx, Ny = size(out)
    mask = FlowGeometries.Grids.mask(grid)
    _share_run(inner, _share_block(k, nk, Ny)) do rows
        for j in rows
            _separable_column_pass_at!(out, fp.row_pass, fp.gy, fp.dj_lim, fp.periodic_y, Nx, Ny, j)
            _separable_normalize_row!(out, fp, mask, Nx, j)
        end
        return nothing
    end
    return nothing
end

# ---- 2-D prefix-sum top-hat: the numerator scan by rows, then the output rows -----------------------

_share_passes(::PrefixSumTopHatPlan) = Val(2)
_share_local(::PrefixSumTopHatPlan, ::Val{1}) = true
_share_dst(_, fp::PrefixSumTopHatPlan, ::Val{1}) =
    _and_values((_prefixsum_num(fp.scratch).hi, _prefixsum_num(fp.scratch).lo), _prefixsum_values(fp.scratch, 1))
_share_dst(out, ::PrefixSumTopHatPlan, ::Val{2}) = (out,)

# A prefix-sum fill's arrays: the scan's two tables, and the cells' values where the scratch holds them.
@inline _and_values(scan::Tuple, ::Nothing) = scan
@inline _and_values(scan::Tuple, V::AbstractArray) = (scan..., V)

function _share_pass!(
    out, field, grid, fp::PrefixSumTopHatPlan, ::AbstractMaskStrategy, ::Val{1}, k::Int, nk::Int, inner,
)
    _share_run(inner, _share_block(k, nk, size(out, 2))) do rows
        for j in rows
            prefixsum_fill_numerator_row!(fp, field, grid, j)
        end
        return nothing
    end
    return nothing
end

function _share_pass!(
    out, _, grid, fp::PrefixSumTopHatPlan, strategy::AbstractMaskStrategy, ::Val{2}, k::Int, nk::Int, inner,
)
    _share_run(inner, _share_block(k, nk, size(out, 2))) do rows
        for j in rows
            apply_prefixsum_tophat_row!(out, grid, fp, strategy, j)
        end
        return nothing
    end
    return nothing
end

# ---- N-D lines of the ball footprint, one pass --------------------------------------------------------

_share_passes(::FilterFootprintND) = Val(1)
_share_dst(out, ::FilterFootprintND, ::Val{1}) = (out,)

function _share_pass!(
    out, field, grid, fp::FilterFootprintND, ::AbstractMaskStrategy, ::Val{1}, k::Int, nk::Int, inner,
)
    src = _nd_source!(fp, field, grid)
    lines = CartesianIndices(Base.tail(size(out)))
    _share_run(inner, _share_block(k, nk, length(lines))) do qs
        for q in qs
            _nd_out_line!(out, src, fp, Tuple(@inbounds lines[q]))
        end
        return nothing
    end
    return nothing
end

# ---- N-D points of the cell-by-cell footprint, one pass ------------------------------------------------

_share_passes(::NDScatteredFilterPlan) = Val(1)
_share_dst(out, ::NDScatteredFilterPlan, ::Val{1}) = (out,)

function _share_pass!(
    out, field, grid, fp::NDScatteredFilterPlan, strategy::AbstractMaskStrategy, ::Val{1}, k::Int, nk::Int, inner,
)
    mask = FlowGeometries.Grids.mask(grid)
    cart = CartesianIndices(size(out))
    lin = LinearIndices(size(out))
    cache, exterior = fp.cache, fp.exterior
    _share_run(inner, _share_block(k, nk, length(cart))) do qs
        for q in qs
            I = @inbounds cart[q]
            out[I] = cache === nothing ? _footprint_nd_point_streaming(field, grid, fp, strategy, mask, I) :
                _footprint_nd_point_cached(field, cache, exterior, strategy, mask, lin, I)
        end
        return nothing
    end
    return nothing
end

# ---- N-D separable: `mask · field`, then one pass per axis, the last normalizing its lines ------------
#
# Pass 1 writes `fp.masked_input`; axis pass `d` (share pass `d + 1`) reads the buffer the pass before
# wrote and writes `fp.scratch` for odd `d`, `fp.masked_input` for even `d`, and `out` for `d = R`, as
# `_sep_pass_chain!` does.

_share_passes(::SeparableFootprintND{R}) where {R} = Val(R + 1)
_share_local(::SeparableFootprintND, ::Val{1}) = true
_share_dst(_, fp::SeparableFootprintND, ::Val{1}) = (fp.masked_input,)
_share_dst(out, fp::SeparableFootprintND{R}, ::Val{p}) where {R,p} = (_sep_share_buffer(out, fp, Val(R), Val(p - 1)),)

@inline _sep_share_buffer(out, _, ::Val{R}, ::Val{R}) where {R} = out
@inline _sep_share_buffer(_, fp, ::Val{R}, ::Val{d}) where {R,d} = isodd(d) ? fp.scratch : fp.masked_input
@inline _sep_share_source(fp, ::Val{1}) = fp.masked_input
@inline _sep_share_source(fp, ::Val{d}) where {d} = isodd(d - 1) ? fp.scratch : fp.masked_input

function _share_pass!(
    out, field, grid, fp::SeparableFootprintND, ::AbstractMaskStrategy, ::Val{1}, k::Int, nk::Int, inner,
)
    mask = FlowGeometries.Grids.mask(grid)
    b = fp.masked_input
    _share_run(inner, _share_block(k, nk, length(b))) do qs
        @inbounds for q in qs
            b[q] = mask[q] * field[q]   # a `Bool` strong zero
        end
        return nothing
    end
    return nothing
end

function _share_pass!(
    out, _, grid, fp::SeparableFootprintND{R,T}, ::AbstractMaskStrategy, ::Val{p}, k::Int, nk::Int, inner,
) where {R, T, p}
    d = p - 1
    dims = size(out)
    dst = _sep_share_buffer(out, fp, Val(R), Val(d))
    src = _sep_share_source(fp, Val(d))
    lines = CartesianIndices(Base.tail(dims))
    mask = FlowGeometries.Grids.mask(grid)
    _share_run(inner, _share_block(k, nk, length(lines))) do qs
        for q in qs
            J = Tuple(@inbounds lines[q])
            if d == 1
                _separable_col_pass!(dst, src, fp.g[1], fp.lim[1], fp.periodic[1], dims[1], J)
            else
                _separable_slab_pass!(dst, src, fp.g[d], fp.lim[d], fp.periodic[d], dims[d], dims[1], J, Val(d))
            end
            d == R && _separable_nd_normalize_line!(out, fp, mask, J)
        end
        return nothing
    end
    return nothing
end

# ---- 3-D prefix-sum top-hat: the axis-1 scan by planes, then the output planes ----------------------

_share_passes(::PrefixSumTopHat3DPlan) = Val(2)
_share_local(::PrefixSumTopHat3DPlan, ::Val{1}) = true
_share_dst(_, fp::PrefixSumTopHat3DPlan, ::Val{1}) =
    _and_values((fp.scratch.prefix.hi, fp.scratch.prefix.lo), fp.scratch.cell_values)
_share_dst(out, ::PrefixSumTopHat3DPlan, ::Val{2}) = (out,)

function _share_pass!(
    out, field, grid, fp::PrefixSumTopHat3DPlan, ::AbstractMaskStrategy, ::Val{1}, k::Int, nk::Int, inner,
)
    planes = CartesianIndices(Base.tail(size(out)))
    _share_run(inner, _share_block(k, nk, length(planes))) do qs
        for q in qs
            JK = @inbounds planes[q]
            _prefixsum3d_fill_plane!(fp.scratch, field, grid, fp.masked, JK[1], JK[2])
        end
        return nothing
    end
    return nothing
end

function _share_pass!(
    out, _, _, fp::PrefixSumTopHat3DPlan, ::AbstractMaskStrategy, ::Val{2}, k::Int, nk::Int, inner,
)
    planes = CartesianIndices(Base.tail(size(out)))
    _share_run(inner, _share_block(k, nk, length(planes))) do qs
        for q in qs
            JK = @inbounds planes[q]
            _prefixsum3d_out_plane!(out, fp, JK[1], JK[2])
        end
        return nothing
    end
    return nothing
end

# ---- Nodes of a flat-cell grid, one pass ----------------------------------------------------------------

_share_passes(::NodeFilterPlan) = Val(1)
_share_dst(out, ::NodeFilterPlan, ::Val{1}) = (out,)

function _share_pass!(
    out, field, grid, fp::NodeFilterPlan, strategy::AbstractMaskStrategy, ::Val{1}, k::Int, nk::Int, inner,
)
    for s in eachindex(out)
        _node_stage!(fp, field, grid, s)
    end
    vals, act = _node_inputs(fp, field, grid)
    _share_run(inner, _share_block(k, nk, length(out))) do ss
        for s in ss
            out[_node_id(fp, s)] = _footprint_node_point(field, vals, act, fp, strategy, s)
        end
        return nothing
    end
    return nothing
end

# The footprints a process backend divides; the transform and binned engines run in one process's loops.
const _SharedFootprint = Union{
    _RowFootprint, SeparableFootprint, PrefixSumTopHatPlan, FilterFootprintND, NDScatteredFilterPlan,
    SeparableFootprintND, PrefixSumTopHat3DPlan, NodeFilterPlan,
}
