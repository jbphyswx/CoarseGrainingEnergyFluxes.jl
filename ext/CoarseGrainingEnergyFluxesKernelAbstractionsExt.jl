module CoarseGrainingEnergyFluxesKernelAbstractionsExt

# `@kernel` pattern-matches the unqualified `@Const`/`@index` syntax in its body, so these macros are
# imported bare; qualified forms fail to precompile.
using KernelAbstractions: KernelAbstractions as KA, @kernel, @index, @Const
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF
using FlowGeometries: FlowGeometries

# Every kernel launches with a batch extent; `size(A, 3)` is 1 for a 2-D array, so an unbatched apply is
# the `Nb == 1` case. Footprint tables, weights and mask are spatial-only: only field and output take `b`.
@inline _at(A::AbstractArray{<:Any,2}, i::Int, j::Int, ::Int) = @inbounds A[i, j]
@inline _at(A::AbstractArray{<:Any,3}, i::Int, j::Int, b::Int) = @inbounds A[i, j, b]
@inline _set!(A::AbstractArray{<:Any,2}, i::Int, j::Int, ::Int, v) = @inbounds A[i, j] = v
@inline _set!(A::AbstractArray{<:Any,3}, i::Int, j::Int, b::Int, v) = @inbounds A[i, j, b] = v

# An array whose rank matches the index's is unbatched; a higher rank means the trailing axis is the batch.
@inline _atI(A::AbstractArray{<:Any,N}, I::CartesianIndex{N}, ::Int) where {N} = @inbounds A[I]
@inline _atI(A::AbstractArray{<:Any,M}, I::CartesianIndex{N}, b::Int) where {M,N} = @inbounds A[I, b]
@inline _setI!(A::AbstractArray{<:Any,N}, I::CartesianIndex{N}, ::Int, v) where {N} = @inbounds A[I] = v
@inline _setI!(A::AbstractArray{<:Any,M}, I::CartesianIndex{N}, b::Int, v) where {M,N} = @inbounds A[I, b] = v

# Every kernel below writes the outputs of a block of its units, rows `j0 + 1 …` of a 2-D grid or linear
# cells `lin0 + 1 …`, launched over the block; a whole-grid apply passes `0` and the whole range.

# Sums the taps in the host's `k` order and scales by the host-built reciprocal `invden`, so the result
# is bit-identical to serial. An inactive source contributes `w[k] * 0`, as in the host's `mask · field`;
# `invden` is zero at a target the strategy does not filter.
@kernel function _cgef_filter_kernel!(
    out, @Const(field), @Const(mask), @Const(rdj), @Const(rdi), @Const(rstart), @Const(w), @Const(ptr),
    @Const(invden), nbands::Int, periodic_x::Bool, periodic_y::Bool, j0::Int,
)
    i, jl, b = @index(Global, NTuple)
    j = jl + j0
    T = eltype(out)
    Nx, Ny, Nb = size(out, 1), size(out, 2), size(out, 3)
    if i <= Nx && j <= Ny && b <= Nb
        band = nbands == 1 ? 1 : j
        ws = zero(T)
        # The runs of a `Filtering.BandTable`, in the host's tap order.
        for s in ptr[band]:(ptr[band+1] - 1)
            jj = j + rdj[s]
            inrow = true
            if jj < 1 || jj > Ny
                if periodic_y
                    jj = mod1(jj, Ny)
                else
                    inrow = false
                end
            end
            if inrow
                k0 = rstart[s]
                for k in k0:(rstart[s+1] - 1)
                    ii = i + rdi[s] + (k - k0)
                    inbounds = true
                    if ii < 1 || ii > Nx
                        if periodic_x
                            ii = mod1(ii, Nx)
                        else
                            inbounds = false
                        end
                    end
                    if inbounds
                        ws += T(w[k]) * (mask[ii, jj] ? _at(field, ii, jj, b) : zero(T))
                    end
                end
            end
        end
        _set!(out, i, j, b, ws * invden[i, j])
    end
end

# 1-D / 3-D grids: the translation-invariant offset table per point, summed in the host's tap order and
# scaled by the host-built reciprocal `invden`, as `_cgef_filter_kernel!` does in 2-D.
@kernel function _cgef_filter_kernel_nd!(
    out, @Const(field), @Const(mask), @Const(offsets), @Const(w), @Const(invden), dims, periodic, lin0::Int,
)
    ll, b = @index(Global, NTuple)
    lin = ll + lin0
    T = eltype(out)
    nspatial = prod(dims)
    Nb = size(out, length(dims) + 1)
    if lin <= nspatial && b <= Nb
        I = CartesianIndices(dims)[lin]
        Ti = Tuple(I)
        ws = zero(T)
        for k in eachindex(offsets)
            off = offsets[k]
            # The `ntuple` closure below only reads captured variables: Julia boxes a captured
            # variable that is assigned, and a device kernel cannot hold a box.
            valid = true
            for d in eachindex(dims)
                jj = Ti[d] + off[d]
                if (jj < 1 || jj > dims[d]) && !periodic[d]
                    valid = false
                end
            end
            if valid
                J = ntuple(Val(length(dims))) do d
                    jj = Ti[d] + off[d]
                    (jj < 1 || jj > dims[d]) ? mod1(jj, dims[d]) : jj
                end
                JI = CartesianIndex(J)
                ws += T(w[k]) * (mask[JI] ? _atI(field, JI, b) : zero(T))
            end
        end
        _setI!(out, I, b, ws * invden[I])
    end
end

# Driver for `Filtering.apply_separable_nd!`: applies `f` once per index on the device. Launches on one
# backend run in order, so each pass completes before the next starts; the caller synchronizes before
# reading on the host.
@kernel function _cgef_driver_kernel!(f, indices)
    i = @index(Global, Linear)
    if i <= length(indices)
        f(indices[i])
    end
end

struct GPUDriver{D}
    dev::D
end
@inline function (drv::GPUDriver)(f::F, indices) where {F}
    _cgef_driver_kernel!(drv.dev)(f, indices; ndrange = length(indices))
    return nothing
end

# Cached ND scattered footprint: `nbrs` holds neighbour multi-indices with the periodic wrap applied at
# build time.
@kernel function _cgef_filter_kernel_nd_cached!(
    out, @Const(field), @Const(mask), @Const(nbrs), @Const(w), @Const(ptr), exterior, dims, strategy, lin0::Int,
)
    ll, b = @index(Global, NTuple)
    lin = ll + lin0
    T = eltype(out)
    nspatial = prod(dims)
    Nb = size(out, length(dims) + 1)
    is_zerofill = strategy isa CGEF.Filtering.ZeroFill
    if lin <= nspatial && b <= Nb
        I = CartesianIndices(dims)[lin]
        if is_zerofill || mask[I]
            ws = zero(T)
            wn = zero(T)
            for k in ptr[lin]:(ptr[lin+1] - 1)
                JI = CartesianIndex(nbrs[k])
                active = mask[JI]
                wk = w[k]
                if is_zerofill
                    wn += wk
                    if active
                        ws += wk * _atI(field, JI, b)
                    end
                elseif active
                    wn += wk
                    ws += wk * _atI(field, JI, b)
                end
            end
            is_zerofill && (wn = CGEF.Filtering._add_exterior(wn, exterior, I))
            _setI!(out, I, b, CGEF.Filtering._normalized(ws, wn))
        else
            _setI!(out, I, b, zero(T))
        end
    end
end

# Node sets: a gather over each node's CSR neighbour block, in any embedding dimension. `size(A, 2)` is
# 1 for a node vector, so an unbatched apply is the `Nb == 1` case.
@inline _at1(A::AbstractVector, t::Int, ::Int) = @inbounds A[t]
@inline _at1(A::AbstractMatrix, t::Int, b::Int) = @inbounds A[t, b]
@inline _set1!(A::AbstractVector, t::Int, ::Int, v) = @inbounds A[t] = v
@inline _set1!(A::AbstractMatrix, t::Int, b::Int, v) = @inbounds A[t, b] = v

@kernel function _cgef_filter_kernel_node!(
    out, @Const(field), @Const(mask), @Const(nbrs), @Const(w), @Const(ptr), strategy, t0::Int,
)
    tl, b = @index(Global, NTuple)
    t = tl + t0
    T = eltype(out)
    Nn, Nb = size(out, 1), size(out, 2)
    is_zerofill = strategy isa CGEF.Filtering.ZeroFill
    if t <= Nn && b <= Nb
        if is_zerofill || mask[t]
            ws = zero(T)
            wn = zero(T)
            for k in ptr[t]:(ptr[t+1] - 1)
                j = nbrs[k]
                wk = w[k]
                active = mask[j]
                if is_zerofill
                    wn += wk
                    if active
                        ws += wk * _at1(field, j, b)
                    end
                elseif active
                    wn += wk
                    ws += wk * _at1(field, j, b)
                end
            end
            _set1!(out, t, b, CGEF.Filtering._normalized(ws, wn))
        else
            _set1!(out, t, b, zero(T))
        end
    end
end

# Cached scattered footprint: `ii_arr`/`jj_arr` hold absolute neighbour indices with the periodic wrap
# resolved at build time (`ScatteredCache`'s convention).
@kernel function _cgef_filter_kernel_scattered!(
    out, @Const(field), @Const(mask), @Const(ii_arr), @Const(jj_arr), @Const(w), @Const(ptr),
    exterior, strategy, j0::Int,
)
    i, jl, b = @index(Global, NTuple)
    j = jl + j0
    T = eltype(out)
    Nx, Ny, Nb = size(out, 1), size(out, 2), size(out, 3)
    is_zerofill = strategy isa CGEF.Filtering.ZeroFill
    if i <= Nx && j <= Ny && b <= Nb
        if is_zerofill || mask[i, j]
            t = i + (j - 1) * Nx
            lo = ptr[t]
            hi = ptr[t+1] - 1
            ws = zero(T)
            wn = zero(T)
            for k in lo:hi
                ii = ii_arr[k]
                jj = jj_arr[k]
                active = mask[ii, jj]
                wk = w[k]
                if is_zerofill
                    wn += wk
                    if active
                        ws += wk * _at(field, ii, jj, b)
                    end
                elseif active
                    wn += wk
                    ws += wk * _at(field, ii, jj, b)
                end
            end
            is_zerofill && (wn = CGEF.Filtering._add_exterior(wn, exterior, i, j))
            _set!(out, i, j, b, CGEF.Filtering._normalized(ws, wn))
        else
            _set!(out, i, j, b, zero(T))
        end
    end
end

# Runs the host's traversal, `FlowGeometries.Connectivity.fold_within`, inside a `@kernel`. This needs
# it to thread its accumulator by value and, on a `StructuredGrid`, to query with no scratch.
@kernel function _cgef_filter_kernel_traversal!(
    out, @Const(field), @Const(mask), grid, mt, weights, rad, exterior, strategy, j0::Int,
)
    i, jl, bi = @index(Global, NTuple)
    j = jl + j0
    T = eltype(out)
    Nx, Ny, Nb = size(out, 1), size(out, 2), size(out, 3)
    is_zerofill = strategy isa CGEF.Filtering.ZeroFill
    if i <= Nx && j <= Ny && bi <= Nb
        if is_zerofill || mask[i, j]
            # The fold closure assigns no captured variable: Julia boxes one that is assigned, and a
            # device kernel cannot hold a box.
            t = Int(i) + (Int(j) - 1) * Nx
            ws, wn = CGEF.Filtering._ball_fold(
                (zero(T), zero(T)), grid, Int(i), Int(j), rad, mt,
            ) do acc, J, d
                a, b = acc
                wk = last(CGEF.Filtering._cell_weight(weights, grid, t, (Int(i), Int(j)), J, T(d)))
                active = mask[J[1], J[2]]
                # No `return` anywhere: `@kernel` rejects one even inside a closure.
                if is_zerofill
                    (active ? a + wk * _at(field, J[1], J[2], bi) : a, b + wk)
                elseif active
                    (a + wk * _at(field, J[1], J[2], bi), b + wk)
                else
                    acc
                end
            end
            n = is_zerofill ? CGEF.Filtering._add_exterior(wn, exterior, i, j) : wn
            _set!(out, i, j, bi, CGEF.Filtering._normalized(ws, n))
        else
            _set!(out, i, j, bi, zero(T))
        end
    end
end

# The N-D form of `_cgef_filter_kernel_traversal!`, over `Filtering._nd_foldl`, the host's streamed
# enumeration.
@kernel function _cgef_filter_kernel_nd_traversal!(
    out, @Const(field), @Const(mask), grid, mt, weights, rad, exterior, dims, strategy, lin0::Int,
)
    ll, b = @index(Global, NTuple)
    lin = ll + lin0
    T = eltype(out)
    Nb = size(out, length(dims) + 1)
    is_zerofill = strategy isa CGEF.Filtering.ZeroFill
    if lin <= prod(dims) && b <= Nb
        I = CartesianIndices(dims)[lin]
        if is_zerofill || mask[I]
            Ti = Tuple(I)
            t = Int(lin)
            ws, wn = CGEF.Filtering._nd_foldl((zero(T), zero(T)), grid, Ti, rad, mt) do acc, J, d
                a, c = acc
                wk = last(CGEF.Filtering._cell_weight(weights, grid, t, Ti, J, T(d)))
                JI = CartesianIndex(J)
                active = mask[JI]
                if is_zerofill
                    (active ? a + wk * _atI(field, JI, b) : a, c + wk)
                elseif active
                    (a + wk * _atI(field, JI, b), c + wk)
                else
                    acc
                end
            end
            n = is_zerofill ? CGEF.Filtering._add_exterior(wn, exterior, I) : wn
            _setI!(out, I, b, CGEF.Filtering._normalized(ws, n))
        else
            _setI!(out, I, b, zero(T))
        end
    end
end

# Separable weight table, as `Filtering._sepw`: a vector on a uniform axis, otherwise a matrix indexed
# `[position, tap]`.
@inline _gpu_sepw(g::AbstractVector, ::Int, k::Int) = @inbounds g[k]
@inline _gpu_sepw(g::AbstractMatrix, i::Int, k::Int) = @inbounds g[i, k]

@kernel function _cgef_separable_row_pass_kernel!(row_pass, @Const(masked_input), @Const(gx), di_lim::Int, periodic_x::Bool)
    i, j, b = @index(Global, NTuple)
    T = eltype(row_pass)
    Nx, Ny, Nb = size(row_pass, 1), size(row_pass, 2), size(row_pass, 3)
    if i <= Nx && j <= Ny && b <= Nb
        s = zero(T)
        for ddi in (-di_lim):di_lim
            ii = i + ddi
            inbounds = true
            if ii < 1 || ii > Nx
                if periodic_x
                    ii = mod1(ii, Nx)
                else
                    inbounds = false
                end
            end
            if inbounds
                s += _gpu_sepw(gx, i, ddi + di_lim + 1) * _at(masked_input, ii, j, b)
            end
        end
        _set!(row_pass, i, j, b, s)
    end
end

@kernel function _cgef_separable_column_pass_kernel!(out, @Const(row_pass), @Const(gy), dj_lim::Int, periodic_y::Bool, j0::Int)
    i, jl, b = @index(Global, NTuple)
    j = jl + j0
    T = eltype(out)
    Nx, Ny, Nb = size(out, 1), size(out, 2), size(out, 3)
    if i <= Nx && j <= Ny && b <= Nb
        s = zero(T)
        for ddj in (-dj_lim):dj_lim
            jj = j + ddj
            inbounds = true
            if jj < 1 || jj > Ny
                if periodic_y
                    jj = mod1(jj, Ny)
                else
                    inbounds = false
                end
            end
            if inbounds
                s += _gpu_sepw(gy, j, ddj + dj_lim + 1) * _at(row_pass, i, jj, b)
            end
        end
        _set!(out, i, j, b, s)
    end
end

move(dev, x) = (y = KA.allocate(dev, eltype(x), size(x)); copyto!(y, x); y)

CGEF.Filtering._allocate(b::CGEF.ComputationalBackends.GPUBackend, ::Type{T}, dims::Dims) where {T} =
    KA.zeros(b.backend, T, dims...)
CGEF.Filtering._host_memory(b::CGEF.ComputationalBackends.GPUBackend) = b.backend isa KA.CPU

# ---------------------------------------------------------------------------
# Prefix-sum top-hat, on the device
# ---------------------------------------------------------------------------
#
# Scan: one work item per row (2-D) or `(j, k)` plane (3-D), summing along axis 1 in the host's order.
# Sweep: each output point takes its interval from `wcell` on a uniform axis and by binary search over
# `xe` otherwise; both give the interval the host's two-pointer walk reaches.

# The extended axis is `nc` copies of the ascending coordinates `xe`, copy `k` shifted by
# `Filtering._copy_shift(wraps, k, P)`; an entry is `(copy, index)`, compared as the host walk compares it.

# First entry at or past `v`, or `(nc, 1)`.
@inline function _xe_lower(xe, Nx::Int, wraps::Bool, P, v)
    nc = wraps ? 3 : 1
    for k in 0:(nc - 1)
        s = CGEF.Filtering._copy_shift(wraps, k, P)
        if @inbounds(xe[Nx]) + s >= v
            lo = 1
            hi = Nx
            while lo < hi
                mid = (lo + hi) >>> 1
                if @inbounds(xe[mid]) + s < v
                    lo = mid + 1
                else
                    hi = mid
                end
            end
            return k, lo
        end
    end
    return nc, 1
end

# Last entry at or before `v`, or `(0, 0)`.
@inline function _xe_upper(xe, Nx::Int, wraps::Bool, P, v)
    nc = wraps ? 3 : 1
    for k in (nc - 1):-1:0
        s = CGEF.Filtering._copy_shift(wraps, k, P)
        if @inbounds(xe[1]) + s <= v
            lo = 1
            hi = Nx
            while lo < hi
                mid = (lo + hi + 1) >>> 1
                if @inbounds(xe[mid]) + s <= v
                    lo = mid
                else
                    hi = mid - 1
                end
            end
            return k, lo
        end
    end
    return 0, 0
end

# `Ph`/`Pl` are the two tables of a `Filtering.CompensatedScan`; their trailing axis is the batch.
@kernel function _cgef_prefixsum_scan_kernel!(
    Ph, Pl, @Const(field), @Const(mask), @Const(src), @Const(wx), ne::Int,
)
    j, b = @index(Global, NTuple)
    T = eltype(Ph)
    Ny, Nb = size(Ph, 2), size(Ph, 3)
    if j <= Ny && b <= Nb
        h = zero(T)
        l = zero(T)
        @inbounds Ph[1, j, b] = h
        @inbounds Pl[1, j, b] = l
        for k in 1:ne
            i = @inbounds src[k]
            x = @inbounds(mask[i, j]) ? T(_at(field, i, j, b)) * @inbounds(wx[i]) : zero(T)
            h, l = CGEF.Filtering._scan_step(h, l, x)
            @inbounds Ph[k+1, j, b] = h
            @inbounds Pl[k+1, j, b] = l
        end
    end
end

# Summation order matches the host row body, `wyj * (q * window)` included, so the two agree bitwise.
@kernel function _cgef_prefixsum_sweep_kernel!(
    out, @Const(Ph), @Const(Pl), @Const(x), @Const(xe), @Const(wy), @Const(invden), @Const(hwt),
    @Const(wcell), dj_lim::Int, Nx::Int, Ny::Int, tiles_x::Bool, rings_x::Bool, wrap::Bool, tiles_y::Bool,
    x_period, j0::Int,
)
    i, jl, b = @index(Global, NTuple)
    j = jl + j0
    T = eltype(out)
    Nb = size(Ph, 3)
    if i <= Nx && j <= Ny && b <= Nb
        acc = zero(T)
        nband = size(hwt, 1)
        xc = @inbounds x[i]
        row0 = CGEF.Filtering._prefixsum_row0(tiles_y, j, dj_lim)
        for t in 0:(nband - 1)
            h = @inbounds hwt[t+1, j]
            if h >= zero(T)
                jj = mod1(row0 + t, Ny)
                wyj = @inbounds wy[jj]
                q, a, c = CGEF.Filtering._prefixsum_window(tiles_x, rings_x, x_period, h)
                if q > 0
                    acc += wyj * (q * CGEF.Filtering._scan_diff(Ph, Pl, Nx + 1, 1, jj, b))
                end
                w = @inbounds wcell[t+1, j]
                if w >= 0
                    acc += wyj * CGEF.Filtering._band_diff(Ph, Pl, wrap, i, w, Nx, jj, b)
                elseif -a <= c
                    wraps = tiles_x || rings_x
                    lc, lt = _xe_lower(xe, Nx, wraps, x_period, xc - a)
                    hc, ht = _xe_upper(xe, Nx, wraps, x_period, xc + c)
                    if hc * Nx + ht >= lc * Nx + lt
                        acc += wyj * CGEF.Filtering._wrap_sum(Ph, Pl, hc - lc, ht + 1, lt, Nx, jj, b)
                    end
                end
            end
        end
        _set!(out, i, j, b, acc * @inbounds(invden[i, j]))
    end
end

# A `CellOverlap` plan's sweep, summing per target in the host row body's order (`_overlap_row_sum!`): per band
# the cells wholly inside about the target and across a regional seam, then the edge taps; `e` is the target's
# position in the scan's ascending order.
@kernel function _cgef_prefixsum_overlap_kernel!(
    out, @Const(Ph), @Const(Pl), @Const(V), @Const(wy), @Const(invden), @Const(hwt), @Const(wcell), @Const(far),
    @Const(ptr), @Const(rdi), @Const(rdj), @Const(rw), dj_lim::Int, Nx::Int, Ny::Int,
    wrap::Bool, rev::Bool, tiles_y::Bool, j0::Int,
)
    i, jl, b = @index(Global, NTuple)
    j = jl + j0
    T = eltype(out)
    Nb = size(Ph, 3)
    if i <= Nx && j <= Ny && b <= Nb
        e = rev ? Nx + 1 - i : i
        acc = zero(T)
        row0 = CGEF.Filtering._prefixsum_row0(tiles_y, j, dj_lim)
        for t in 0:(size(hwt, 1) - 1)
            if @inbounds(hwt[t+1, j]) >= zero(eltype(hwt))
                jj = mod1(row0 + t, Ny)
                wyj = T(@inbounds wy[jj])
                w = @inbounds wcell[t+1, j]
                if w >= 0
                    if !wrap
                        acc += wyj * CGEF.Filtering._scan_diff(Ph, Pl, min(Nx, e + w) + 1, max(1, e - w), jj, b)
                    elseif 2w + 1 <= Nx
                        acc += wyj * CGEF.Filtering._band_diff(Ph, Pl, true, e, w, Nx, jj, b)
                    else
                        q, rc = divrem(2w + 1, Nx)
                        tot = q * CGEF.Filtering._scan_diff(Ph, Pl, Nx + 1, 1, jj, b)
                        lo = mod1(e - w, Nx)
                        hi = lo + rc - 1
                        part = rc == 0 ? zero(T) : hi <= Nx ? CGEF.Filtering._scan_diff(Ph, Pl, hi + 1, lo, jj, b) :
                            CGEF.Filtering._scan_diff(Ph, Pl, Nx + 1, lo, jj, b) +
                            CGEF.Filtering._scan_diff(Ph, Pl, hi - Nx + 1, 1, jj, b)
                        acc += wyj * (tot + part)
                    end
                end
                fa = @inbounds far[t+1, j]
                if fa > 0
                    s = zero(T)
                    e + fa <= Nx && (s += CGEF.Filtering._scan_diff(Ph, Pl, Nx + 1, e + fa, jj, b))
                    e - fa >= 1 && (s += CGEF.Filtering._scan_diff(Ph, Pl, e - fa + 1, 1, jj, b))
                    acc += wyj * s
                end
            end
        end
        for s in @inbounds(ptr[j]):(@inbounds(ptr[j+1]) - 1)
            jr = j + @inbounds(rdj[s])
            ok = true
            if jr < 1 || jr > Ny
                tiles_y ? (jr = mod1(jr, Ny)) : (ok = false)
            end
            ee = e + @inbounds(rdi[s])
            if ok && wrap
                ee = mod1(ee, Nx)
            elseif ok
                ok = 1 <= ee <= Nx
            end
            if ok
                acc += T(@inbounds rw[s]) * @inbounds(V[ee, jr, b])
            end
        end
        _set!(out, i, j, b, acc * @inbounds(invden[i, j]))
    end
end

@kernel function _cgef_prefixsum3d_scan_kernel!(
    Ph, Pl, @Const(field), @Const(mask), masked::Bool, Nx::Int,
)
    j, k, b = @index(Global, NTuple)
    T = eltype(Ph)
    Ny, Nz, Nb = size(Ph, 2), size(Ph, 3), size(Ph, 4)
    if j <= Ny && k <= Nz && b <= Nb
        h = zero(T)
        l = zero(T)
        @inbounds Ph[1, j, k, b] = h
        @inbounds Pl[1, j, k, b] = l
        for i in 1:Nx
            x = (masked && !@inbounds(mask[i, j, k])) ? zero(T) : T(_atI(field, CartesianIndex(i, j, k), b))
            h, l = CGEF.Filtering._scan_step(h, l, x)
            @inbounds Ph[i+1, j, k, b] = h
            @inbounds Pl[i+1, j, k, b] = l
        end
    end
end

@kernel function _cgef_prefixsum3d_sweep_kernel!(
    out, @Const(Ph), @Const(Pl), @Const(V), @Const(wcell), @Const(invden), @Const(edge_off), @Const(edge_w), dims,
    dj_lim::Int, dk_lim::Int, px::Bool, py::Bool, pz::Bool, lin0::Int,
)
    ll, b = @index(Global, NTuple)
    lin = ll + lin0
    T = eltype(out)
    Nx, Ny, Nz = dims[1], dims[2], dims[3]
    Nb = size(Ph, 4)
    if lin <= Nx * Ny * Nz && b <= Nb
        I = CartesianIndices(dims)[lin]
        i, j, k = Tuple(I)
        acc = zero(T)
        for dk in (-dk_lim):dk_lim
            kk = k + dk
            inz = true
            if kk < 1 || kk > Nz
                pz ? (kk = mod1(kk, Nz)) : (inz = false)
            end
            if inz
                for dj in (-dj_lim):dj_lim
                    w = @inbounds wcell[dj+dj_lim+1, dk+dk_lim+1]
                    if w >= 0
                        jj = j + dj
                        iny = true
                        if jj < 1 || jj > Ny
                            py ? (jj = mod1(jj, Ny)) : (iny = false)
                        end
                        if iny
                            if px
                                q, rc = divrem(2 * w + 1, Nx)
                                if q > 0
                                    acc += q * CGEF.Filtering._scan_diff(Ph, Pl, Nx + 1, 1, jj, kk, b)
                                end
                                if rc > 0
                                    lo = mod1(i - w, Nx)
                                    hi = lo + rc - 1
                                    if hi <= Nx
                                        acc += CGEF.Filtering._scan_diff(Ph, Pl, hi + 1, lo, jj, kk, b)
                                    else
                                        acc += CGEF.Filtering._scan_diff(Ph, Pl, Nx + 1, lo, jj, kk, b) +
                                               CGEF.Filtering._scan_diff(Ph, Pl, hi - Nx + 1, 1, jj, kk, b)
                                    end
                                end
                            else
                                lo = max(1, i - w)
                                hi = min(Nx, i + w)
                                acc += CGEF.Filtering._scan_diff(Ph, Pl, hi + 1, lo, jj, kk, b)
                            end
                        end
                    end
                end
            end
        end
        # The edge taps after the planes, as the host's `_prefixsum3d_edge!` adds them.
        for s in 1:length(edge_w)
            di, dj, dk = @inbounds edge_off[s]
            ii, jj, kk = i + di, j + dj, k + dk
            ok = true
            if jj < 1 || jj > Ny
                py ? (jj = mod1(jj, Ny)) : (ok = false)
            end
            if kk < 1 || kk > Nz
                pz ? (kk = mod1(kk, Nz)) : (ok = false)
            end
            if ii < 1 || ii > Nx
                px ? (ii = mod1(ii, Nx)) : (ok = false)
            end
            if ok
                acc += T(@inbounds edge_w[s]) * @inbounds(V[ii, jj, kk, b])
            end
        end
        _setI!(out, I, b, acc * @inbounds(invden[I]))
    end
end

"""
    GPUResident

A footprint whose tables, mask and intermediate buffers live on the device. `plan_filter` builds one
through [`CGEF.Filtering.prepare_workspace`](@ref), and every later `filter_apply!` reuses it.
"""
abstract type GPUResident end

# Banded (uniform-axis) footprint: the band table's arrays.
struct GPUBandedFootprint{F, VI, VW, MA, IN} <: GPUResident
    fp::F
    dj::VI
    di0::VI
    start::VI
    w::VW
    ptr::VI
    maskd::MA
    invden::IN   # reciprocal window mass, built once on the host with the plan
end

# Scattered footprint with a materialized neighbour cache: absolute neighbour indices per point.
struct GPUScatteredCached{F, II, JJ, VW, VP, MA, EX} <: GPUResident
    fp::F
    ii::II
    jj::JJ
    w::VW
    ptr::VP
    maskd::MA
    exterior::EX   # `ZeroFill`'s mass past a bounded edge, or nothing
end

# Streaming footprint: the kernel derives each point's neighbourhood from the device copy of the grid.
struct GPUStreaming{F, G, MT, MA, EX, W} <: GPUResident
    fp::F
    grid::G
    topology::MT
    maskd::MA
    exterior::EX
    weights::W
end

# An N-D cell-by-cell plan with no neighbour list: the device copies of the grid and topology it queries.
struct GPUStreamingND{F, G, MT, MA, EX, W} <: GPUResident
    fp::F
    grid::G
    topology::MT
    maskd::MA
    exterior::EX
    weights::W
end

_device_weights(_, r::CGEF.Filtering.RadialWeights) = r
_device_weights(dev, r::CGEF.Filtering.HighOrderWeights) = CGEF.Filtering._with_coeffs(r, move(dev, r.coeffs))

# The neighbour list a device plan reads. A streaming kernel evaluates radial and high-order weights; a
# `CellOverlap` rule's walk runs on the host, so a plan of one without a list has it built there.
_device_cache(_, fp, _) = fp.cache
_device_cache(
    grid, fp::CGEF.Filtering.ScatteredFilterPlan,
    w::Union{CGEF.Filtering.OverlapWeights, CGEF.Filtering.OverlapPolygons},
) = fp.cache === nothing ?
    CGEF.Filtering._scattered_cache(grid, w, fp.rad, fp.topology, CGEF.Filtering.AlwaysCache(), 0) : fp.cache
_device_cache(
    grid, fp::CGEF.Filtering.NDScatteredFilterPlan, w::Union{CGEF.Filtering.OverlapWeightsND, CGEF.Filtering.OverlapShells},
) = CGEF.Filtering._expanded(
    fp.cache === nothing ? CGEF.Filtering._nd_cache(grid, w, fp.rad, fp.topology, CGEF.Filtering.AlwaysCache(), 0) :
    fp.cache)

# Separable Gaussian: the two weight tables, the two pass buffers, and the footprint's denominator.
struct GPUSeparable{F, GX, GY, AT, IR, DE, MA, BB<:Base.RefValue} <: GPUResident
    fp::F
    gx::GX
    gy::GY
    masked_input::AT
    row_pass::AT
    invrenorm::IR   # dense Deformable denominator, or nothing
    denom::DE       # rank-1 ZeroFill denominator, or nothing
    maskd::MA
    # Batch-shaped pass buffers, allocated on the first batched apply and reused. Typed
    # `Union{Nothing,Tuple{AT3,AT3}}` with `AT3` the rank-3 device array type, so a read is type-stable.
    batch_buffers::BB
end

# Separable Gaussian on a 1-D / 3-D grid: the host footprint rebuilt with device arrays, and the
# batch-shaped pass buffers of rank `N + 1`, allocated on the first batched apply and reused as
# `GPUSeparable`'s are.
struct GPUSeparableND{F, BB<:Base.RefValue} <: GPUResident
    fp::F
    batch_buffers::BB
end

# Cached ND scattered footprint, device-resident.
struct GPUScatteredNDCached{F, VN, VW, VP, MA, EX} <: GPUResident
    fp::F
    nbrs::VN
    w::VW
    ptr::VP
    maskd::MA
    exterior::EX
end

# 1-D / true-3-D translation-invariant offset table.
struct GPUFootprintND{F, VO, VW, MA, AT} <: GPUResident
    fp::F
    offsets::VO
    w::VW
    maskd::MA
    invden::AT   # reciprocal window mass, built once on the host with the plan
end

# Node set: the CSR adjacency and its weights.
struct GPUNodeFootprint{F, VN, VW, VP, MA} <: GPUResident
    fp::F
    nbrs::VN
    w::VW
    ptr::VP
    maskd::MA
end

# Prefix-sum top-hat. `P` is the device numerator scan (a `Filtering.CompensatedScan`) with a trailing
# batch axis of extent 1; a batched apply uses a wider one from `batch_scan`, allocated on first use.
struct GPUPrefixSum{F, VT, VI, MT, WT, MA, PB} <: GPUResident
    fp::F
    x::VT
    xe::VT
    src::VI
    wx::VT
    wy::VT
    invden::MT
    hw::MT
    wcell::WT
    maskd::MA
    P::PB
    # A batched scan table has type `PB`; only its trailing extent differs.
    batch_scan::Base.RefValue{Union{Nothing,PB}}
    ne::Int
end

# A `CellOverlap` prefix-sum plan: the bands' whole-cell widths and far offsets, its edge cells by target row,
# and the cells' values the edge cells read (`V`, with a trailing batch axis; a batched apply's wider table from
# `batch_values`).
struct GPUPrefixOverlap{F, VT, VI, MT, WT, MA, PB, VA} <: GPUResident
    fp::F
    src::VI
    wx::VT
    wy::VT
    invden::MT
    hw::MT
    wcell::WT
    far::WT
    ptr::VI
    rdi::VI
    rdj::VI
    rw::VT
    maskd::MA
    P::PB
    batch_scan::Base.RefValue{Union{Nothing,PB}}
    ne::Int
    V::VA
    batch_values::Base.RefValue{Union{Nothing,VA}}
end

# `base` when it has size `sz`; otherwise the array held in `cache`, reallocated when its size differs.
@inline function _held_table(dev, base::A, cache::Base.RefValue{Union{Nothing,A}}, sz::NTuple{N,Int}) where {A,N}
    size(base) == sz && return base
    c = cache[]
    (c !== nothing && size(c) == sz) && return c
    b = KA.allocate(dev, eltype(base), sz...)
    cache[] = b
    return b
end

# The cells' values `mask·field·wx` in the scan's order.
@kernel function _cgef_cell_values_kernel!(V, @Const(field), @Const(mask), @Const(src), @Const(wx))
    k, j, b = @index(Global, NTuple)
    T = eltype(V)
    if k <= size(V, 1) && j <= size(V, 2) && b <= size(V, 3)
        i = @inbounds src[k]
        @inbounds V[k, j, b] = @inbounds(mask[i, j]) ? T(_at(field, i, j, b)) * @inbounds(wx[i]) : zero(T)
    end
end

# `edge_off`/`edge_w` hold a `CellOverlap` plan's edge taps and `V` the cells' values they read, with a trailing
# batch axis (a batched apply's wider table from `batch_values`); under `CellCentre` the taps are empty and `V`
# has no entries.
struct GPUPrefixSum3D{F, WT, AT, MA, PB, VO, VW, VA} <: GPUResident
    fp::F
    wcell::WT
    invden::AT
    maskd::MA
    P::PB
    batch_scan::Base.RefValue{Union{Nothing,PB}}
    edge_off::VO
    edge_w::VW
    V::VA
    batch_values::Base.RefValue{Union{Nothing,VA}}
end

# The cells' values `mask·field`, cell `lin` of the grid in column-major order.
@kernel function _cgef_cell_values3d_kernel!(V, @Const(field), @Const(mask), masked::Bool, dims)
    lin, b = @index(Global, NTuple)
    T = eltype(V)
    if lin <= dims[1] * dims[2] * dims[3] && b <= size(V, 4)
        I = CartesianIndices(dims)[lin]
        i, j, k = Tuple(I)
        @inbounds V[i, j, k, b] = (masked && !@inbounds(mask[i, j, k])) ? zero(T) : T(_atI(field, I, b))
    end
end

# The values table an apply over `Nb` fields reads, filled from `field`; under `CellCentre` nothing reads it.
_cell_values_3d!(_, ws::GPUPrefixSum3D, ::Nothing, _, _, _) = ws.V
function _cell_values_3d!(dev, ws::GPUPrefixSum3D, ::CGEF.Filtering.PrefixEdgeCells3D, field, dims, Nb)
    V = _held_table(dev, ws.V, ws.batch_values, (dims..., Nb))
    _cgef_cell_values3d_kernel!(dev)(V, field, ws.maskd, ws.fp.masked, dims; ndrange = (prod(dims), Nb))
    return V
end

@inline _batch_scan_ref(P) = Base.RefValue{Union{Nothing,typeof(P)}}(nothing)

# A device scan table. Every entry the sweep reads is written by the scan, so it starts uninitialized.
_device_scan(dev, ::Type{T}, dims::Integer...) where {T} =
    CGEF.Filtering.CompensatedScan(KA.allocate(dev, T, dims...), KA.allocate(dev, T, dims...))

# `base` when it has size `sz`; otherwise the table held in `cache`, reallocated when its size differs.
@inline function _scan_table(dev, base::PB, cache::Base.RefValue{Union{Nothing,PB}}, sz::NTuple{N,Int}) where {PB,N}
    size(base.hi) == sz && return base
    c = cache[]
    (c !== nothing && size(c.hi) == sz) && return c
    b = _device_scan(dev, eltype(base.hi), sz...)
    cache[] = b
    return b
end

# The grid's mask in device memory. `AllActive` holds only its size and goes to a kernel as it is.
_maskd(dev, grid) = _device_mask(dev, FlowGeometries.Grids.mask(grid))
_device_mask(_, m::FlowGeometries.Grids.AllActive) = m
_device_mask(dev, m::AbstractArray{Bool}) = move(dev, Array{Bool}(m))
_moved(_, ::Nothing) = nothing
_moved(dev, x::AbstractArray) = move(dev, x)

CGEF.Filtering.prepare_workspace(
    b::CGEF.ComputationalBackends.GPUBackend, grid::FlowGeometries.Grids.AbstractGrid,
    fp::CGEF.Filtering.PrefixSumTopHatPlan,
) = _prefix_workspace(b.backend, grid, fp, fp.edge)

function _prefix_workspace(dev, grid, fp::CGEF.Filtering.PrefixSumTopHatPlan{T}, ::Nothing) where {T}
    gp = fp.grid_plan
    _, Ny = FlowGeometries.Grids.size_tuple(grid)
    ne = length(gp.xe)
    P = _device_scan(dev, T, ne + 1, Ny, 1)
    return GPUPrefixSum(
        fp,
        move(dev, collect(T, FlowGeometries.Grids.coordinates(grid, 1))),
        move(dev, collect(T, gp.xe)), move(dev, collect(Int, gp.src)),
        move(dev, collect(T, gp.wx)), move(dev, collect(T, gp.wy)),
        move(dev, fp.invden), move(dev, fp.hw), move(dev, fp.wcell),
        _maskd(dev, grid), P, _batch_scan_ref(P), ne,
    )
end

function _prefix_workspace(dev, grid, fp::CGEF.Filtering.PrefixSumTopHatPlan{T}, edge::CGEF.Filtering.PrefixEdgeCells) where {T}
    gp = fp.grid_plan
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    ne = length(gp.xe)
    P = _device_scan(dev, T, ne + 1, Ny, 1)
    V = KA.allocate(dev, T, Nx, Ny, 1)
    return GPUPrefixOverlap(
        fp, move(dev, collect(Int, gp.src)), move(dev, collect(T, gp.wx)), move(dev, collect(T, gp.wy)),
        move(dev, fp.invden), move(dev, fp.hw), move(dev, fp.wcell), move(dev, edge.far),
        move(dev, edge.ptr), move(dev, edge.di), move(dev, edge.dj), move(dev, collect(T, edge.w)),
        _maskd(dev, grid), P, _batch_scan_ref(P), ne, V, Base.RefValue{Union{Nothing,typeof(V)}}(nothing),
    )
end

function CGEF.Filtering.prepare_workspace(
    b::CGEF.ComputationalBackends.GPUBackend, grid::FlowGeometries.Grids.AbstractGrid,
    fp::CGEF.Filtering.PrefixSumTopHat3DPlan{T},
) where {T<:AbstractFloat}
    dev = b.backend
    Nx, Ny, Nz = FlowGeometries.Grids.size_tuple(grid)
    P = _device_scan(dev, T, Nx + 1, Ny, Nz, 1)
    off, w = _edge_cells(fp.edge, T)
    V = KA.allocate(dev, T, _values_size(fp.edge, Nx, Ny, Nz)...)
    return GPUPrefixSum3D(
        fp, move(dev, fp.wcell), move(dev, fp.invden), _maskd(dev, grid), P, _batch_scan_ref(P),
        move(dev, off), move(dev, w), V, Base.RefValue{Union{Nothing,typeof(V)}}(nothing),
    )
end

_edge_cells(::Nothing, ::Type{T}) where {T} = (NTuple{3,Int}[], T[])
_edge_cells(edge::CGEF.Filtering.PrefixEdgeCells3D, ::Type{T}) where {T} = (edge.offsets, collect(T, edge.w))
_values_size(::Nothing, _, _, _) = (0, 0, 0, 1)
_values_size(::CGEF.Filtering.PrefixEdgeCells3D, Nx, Ny, Nz) = (Nx, Ny, Nz, 1)

function CGEF.Filtering.prepare_workspace(
    b::CGEF.ComputationalBackends.GPUBackend, grid::FlowGeometries.Grids.AbstractGrid,
    fp::CGEF.Filtering.FilterFootprint,
)
    dev = b.backend
    t = fp.taps
    return GPUBandedFootprint(
        fp, move(dev, t.dj), move(dev, t.di0), move(dev, t.start), move(dev, t.w), move(dev, t.ptr),
        _maskd(dev, grid), move(dev, fp.invden),
    )
end

# A plan with a neighbour list reads it; one without streams, querying device copies of the grid and
# topology in the host's order.
function CGEF.Filtering.prepare_workspace(
    b::CGEF.ComputationalBackends.GPUBackend, grid::FlowGeometries.Grids.AbstractGrid,
    fp::CGEF.Filtering.NDScatteredFilterPlan,
)
    dev = b.backend
    maskd = _maskd(dev, grid)
    exterior = _moved(dev, fp.exterior)
    cache = _device_cache(grid, fp, fp.weights)
    cache === nothing || return GPUScatteredNDCached(
        fp, move(dev, cache.nbrs), move(dev, cache.w), move(dev, cache.ptr), maskd, exterior,
    )
    return GPUStreamingND(
        fp, KA.adapt(dev, grid), KA.adapt(dev, fp.topology), maskd, exterior, _device_weights(dev, fp.weights),
    )
end

function CGEF.Filtering.prepare_workspace(
    b::CGEF.ComputationalBackends.GPUBackend, grid::FlowGeometries.Grids.AbstractGrid,
    fp::CGEF.Filtering.SeparableFootprintND{N,T},
) where {N, T<:AbstractFloat}
    dev = b.backend
    # The batched device array type, of rank `N + 1`.
    AB = typeof(KA.allocate(dev, T, ntuple(_ -> 0, Val(N + 1))...))
    return GPUSeparableND(
        CGEF.Filtering.SeparableFootprintND(
            map(g -> move(dev, g), fp.g), fp.lim, fp.periodic,
            fp.profiles === nothing ? nothing : map(pv -> move(dev, pv), fp.profiles),
            _moved(dev, fp.invrenorm), fp.strategy, fp.masked, fp.bound,
            move(dev, fp.masked_input), move(dev, fp.scratch),
        ),
        Base.RefValue{Union{Nothing,Tuple{AB,AB}}}(nothing),
    )
end

function CGEF.Filtering.prepare_workspace(
    b::CGEF.ComputationalBackends.GPUBackend, grid::FlowGeometries.Grids.AbstractGrid,
    fp::CGEF.Filtering.FilterFootprintND,
)
    dev = b.backend
    return GPUFootprintND(fp, move(dev, fp.offsets), move(dev, fp.w), _maskd(dev, grid), move(dev, fp.invden))
end

function CGEF.Filtering.prepare_workspace(
    b::CGEF.ComputationalBackends.GPUBackend, grid::FlowGeometries.Grids.AbstractGrid,
    fp::CGEF.Filtering.NodeFilterPlan,
)
    dev = b.backend
    return GPUNodeFootprint(fp, move(dev, fp.nbrs), move(dev, fp.w), move(dev, fp.ptr), _maskd(dev, grid))
end

function CGEF.Filtering.prepare_workspace(
    b::CGEF.ComputationalBackends.GPUBackend, grid::FlowGeometries.Grids.AbstractGrid,
    fp::CGEF.Filtering.ScatteredFilterPlan,
)
    dev = b.backend
    maskd = _maskd(dev, grid)
    exterior = _moved(dev, fp.exterior)
    cache = _device_cache(grid, fp, fp.weights)
    cache === nothing || return GPUScatteredCached(
        fp, move(dev, cache.ii), move(dev, cache.jj), move(dev, cache.w), move(dev, cache.ptr), maskd,
        exterior,
    )
    # The streaming kernel queries device copies of the grid and topology (moved by FlowGeometries'
    # `Adapt` methods), visiting each point's candidates in the host's order.
    return GPUStreaming(
        fp, KA.adapt(dev, grid), KA.adapt(dev, fp.topology), maskd, exterior,
        _device_weights(dev, fp.weights),
    )
end

function CGEF.Filtering.prepare_workspace(
    b::CGEF.ComputationalBackends.GPUBackend, grid::FlowGeometries.Grids.AbstractGrid,
    fp::CGEF.Filtering.SeparableFootprint{T},
) where {T<:AbstractFloat}
    dev = b.backend
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    invrenorm = fp.invrenorm === nothing ? nothing : move(dev, fp.invrenorm)
    denom = if fp.invrenorm === nothing
        xp = move(dev, fp.Nx_profile)
        yp = move(dev, fp.Ny_profile)
        d = KA.allocate(dev, T, Nx, Ny)
        d .= reshape(xp, Nx, 1) .* reshape(yp, 1, Ny)
        d
    else
        nothing
    end
    # The rank-3 device array type of the batch buffers.
    AT3 = typeof(KA.allocate(dev, T, 0, 0, 0))
    return GPUSeparable(
        fp, move(dev, fp.gx), move(dev, fp.gy),
        KA.allocate(dev, T, Nx, Ny), KA.allocate(dev, T, Nx, Ny),
        invrenorm, denom, _maskd(dev, grid),
        Base.RefValue{Union{Nothing,Tuple{AT3,AT3}}}(nothing),
    )
end

# The units a resident footprint's final kernel runs over, and the linear cells of the output a block of
# them writes: rows of a 2-D grid, cells of an N-D grid, lines along axis 1 for the separable N-D passes,
# nodes of a node set. A process's share runs one block of them.
const _RowResident = Union{GPUBandedFootprint, GPUScatteredCached, GPUStreaming, GPUSeparable, GPUPrefixSum, GPUPrefixOverlap}
const _CellResident = Union{GPUFootprintND, GPUScatteredNDCached, GPUStreamingND, GPUPrefixSum3D}

_gpu_units(::_RowResident, grid) = FlowGeometries.Grids.size_tuple(grid)[2]
_gpu_units(::_CellResident, grid) = prod(FlowGeometries.Grids.size_tuple(grid))
_gpu_units(::GPUSeparableND, grid) = prod(Base.tail(FlowGeometries.Grids.size_tuple(grid)))
_gpu_units(::GPUNodeFootprint, grid) = length(FlowGeometries.Grids.mask(grid))

_gpu_cells(::Union{_RowResident, GPUSeparableND}, grid, units::UnitRange{Int}) =
    (n1 = FlowGeometries.Grids.size_tuple(grid)[1]; ((first(units) - 1) * n1 + 1):(last(units) * n1))
_gpu_cells(::Union{_CellResident, GPUNodeFootprint}, _, units::UnitRange{Int}) = units

# Every `_run_gpu_kernel!` takes the mask strategy, which an engine whose denominator holds it ignores, and
# the block of units its final kernel writes, every unit for a whole apply.
_run_gpu_kernel!(dev, out, field, ws::GPUResident, grid, strategy::CGEF.Filtering.AbstractMaskStrategy) =
    _run_gpu_kernel!(dev, out, field, ws, grid, strategy, 1:_gpu_units(ws, grid))

# The banded engine serves only 2-D structured grids, so a 3-D `out` carries a batch axis.
function _run_gpu_kernel!(
    dev, out, field, ws::GPUBandedFootprint, grid, ::CGEF.Filtering.AbstractMaskStrategy, rows::UnitRange{Int},
)
    _cgef_filter_kernel!(dev)(
        out, field, ws.maskd, ws.dj, ws.di0, ws.start, ws.w, ws.ptr, ws.invden, ws.fp.nbands,
        ws.fp.periodic_x, ws.fp.periodic_y, first(rows) - 1;
        ndrange = (size(out, 1), length(rows), size(out, 3)),
    )
end

# `plan.footprint` on a GPU plan is the device-resident footprint `prepare_workspace` built.
CGEF.Filtering._gpu_batched_supported(plan::CGEF.Filtering.PhysicalFilterPlan) =
    plan.footprint isa Union{GPUBandedFootprint, GPUSeparable, GPUNodeFootprint, GPUFootprintND, GPUSeparableND,
                             GPUScatteredCached, GPUScatteredNDCached, GPUStreaming, GPUStreamingND,
                             GPUPrefixSum, GPUPrefixOverlap, GPUPrefixSum3D}

# Node grids: a batch is an `(Nnodes, Nb)` matrix.
function CGEF.Filtering.gpu_filter_field_batched!(
    gpu_backend::CGEF.ComputationalBackends.GPUBackend,
    out::AbstractArray{T,2},
    field::AbstractArray{T,2},
    grid::FlowGeometries.Grids.AbstractGrid,
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
    workspace,
) where {T<:AbstractFloat}
    dev = gpu_backend.backend
    ws = _resident(gpu_backend, grid, workspace, kernel, scale, mask_strategy)
    _run_gpu_kernel!(dev, out, field, ws, grid, mask_strategy)
    KA.synchronize(dev)
    return out
end

# Any structured rank. The caller, `filter_apply_batched!`, has checked that the leading axes are the
# grid's and the trailing one is the batch.
function CGEF.Filtering.gpu_filter_field_batched!(
    gpu_backend::CGEF.ComputationalBackends.GPUBackend,
    out::AbstractArray{T},
    field::AbstractArray{T},
    grid::Union{FlowGeometries.Grids.StructuredGrid{T,G}, FlowGeometries.Grids.CurvilinearGrid{T,G}},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
    workspace,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    dev = gpu_backend.backend
    ws = _resident(gpu_backend, grid, workspace, kernel, scale, mask_strategy)
    _check_resident_strategy(ws, mask_strategy)
    _run_gpu_kernel!(dev, out, field, ws, grid, mask_strategy)
    KA.synchronize(dev)
    return out
end

# Two launches on one backend, which runs them in order: the sweep reads every row of the scan, so the
# scan covers the whole grid whatever rows the sweep writes.
function _run_gpu_kernel!(
    dev, out, field, ws::GPUPrefixSum, grid, ::CGEF.Filtering.AbstractMaskStrategy, rows::UnitRange{Int},
)
    fp = ws.fp
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    Nb = ndims(out) == 2 ? 1 : size(out, 3)
    P = _scan_table(dev, ws.P, ws.batch_scan, (ws.ne + 1, Ny, Nb))
    _cgef_prefixsum_scan_kernel!(dev)(P.hi, P.lo, field, ws.maskd, ws.src, ws.wx, ws.ne; ndrange = (Ny, Nb))
    gp = fp.grid_plan
    _cgef_prefixsum_sweep_kernel!(dev)(
        out, P.hi, P.lo, ws.x, ws.xe, ws.wy, ws.invden, ws.hw, ws.wcell, fp.dj_lim, Nx, Ny,
        gp.tiles_x, gp.rings_x, CGEF.Filtering._prefixsum_band_wraps(gp), gp.tiles_y, gp.x_period, first(rows) - 1;
        ndrange = (Nx, length(rows), Nb),
    )
    return out
end

function _run_gpu_kernel!(
    dev, out, field, ws::GPUPrefixOverlap, grid, ::CGEF.Filtering.AbstractMaskStrategy, rows::UnitRange{Int},
)
    fp = ws.fp
    edge = fp.edge
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    Nb = ndims(out) == 2 ? 1 : size(out, 3)
    P = _scan_table(dev, ws.P, ws.batch_scan, (ws.ne + 1, Ny, Nb))
    V = _held_table(dev, ws.V, ws.batch_values, (Nx, Ny, Nb))
    _cgef_prefixsum_scan_kernel!(dev)(P.hi, P.lo, field, ws.maskd, ws.src, ws.wx, ws.ne; ndrange = (Ny, Nb))
    _cgef_cell_values_kernel!(dev)(V, field, ws.maskd, ws.src, ws.wx; ndrange = (Nx, Ny, Nb))
    _cgef_prefixsum_overlap_kernel!(dev)(
        out, P.hi, P.lo, V, ws.wy, ws.invden, ws.hw, ws.wcell, ws.far, ws.ptr, ws.rdi, ws.rdj, ws.rw,
        fp.dj_lim, Nx, Ny, edge.wrap, edge.rev, fp.grid_plan.tiles_y, first(rows) - 1;
        ndrange = (Nx, length(rows), Nb),
    )
    return out
end

function _run_gpu_kernel!(
    dev, out, field, ws::GPUPrefixSum3D, grid, ::CGEF.Filtering.AbstractMaskStrategy, cells::UnitRange{Int},
)
    fp = ws.fp
    dims = FlowGeometries.Grids.size_tuple(grid)
    Nx, Ny, Nz = dims
    Nb = ndims(out) == 3 ? 1 : size(out, 4)
    P = _scan_table(dev, ws.P, ws.batch_scan, (Nx + 1, Ny, Nz, Nb))
    _cgef_prefixsum3d_scan_kernel!(dev)(P.hi, P.lo, field, ws.maskd, fp.masked, Nx; ndrange = (Ny, Nz, Nb))
    V = _cell_values_3d!(dev, ws, fp.edge, field, dims, Nb)
    _cgef_prefixsum3d_sweep_kernel!(dev)(
        out, P.hi, P.lo, V, ws.wcell, ws.invden, ws.edge_off, ws.edge_w, dims, fp.dj_lim, fp.dk_lim,
        fp.periodic[1], fp.periodic[2], fp.periodic[3], first(cells) - 1; ndrange = (length(cells), Nb),
    )
    return out
end

function _run_gpu_kernel!(
    dev, out, field, ws::GPUScatteredNDCached, grid, strategy::CGEF.Filtering.AbstractMaskStrategy,
    cells::UnitRange{Int},
)
    dims = FlowGeometries.Grids.size_tuple(grid)
    _cgef_filter_kernel_nd_cached!(dev)(
        out, field, ws.maskd, ws.nbrs, ws.w, ws.ptr, ws.exterior, dims, strategy, first(cells) - 1;
        ndrange = (length(cells), size(out, length(dims) + 1)),
    )
end

# The passes run in order on the backend; the next apply's staging broadcast, which on a host-memory
# backend runs on the host, writes the buffer the passes read, so the apply waits for them. Every pass but
# the last covers the whole grid; the last writes `lines`.
function _run_gpu_kernel!(
    dev, out, field, ws::GPUSeparableND, grid, strategy::CGEF.Filtering.AbstractMaskStrategy, lines::UnitRange{Int},
)
    CGEF.Filtering.apply_separable_nd!(out, field, grid, ws.fp, strategy, GPUDriver(dev), _sep_nd_buffers(dev, ws, out),
                                       lines)
    KA.synchronize(dev)
    return out
end

# Pass buffers for one apply: the footprint's own for a single field, a batch-shaped pair held on the
# workspace for a batch.
@inline _sep_nd_buffers(dev, ws::GPUSeparableND, out::AbstractArray) =
    ndims(out) == ndims(ws.fp.scratch) ? (ws.fp.scratch, ws.fp.masked_input) : _held_pair(dev, ws.batch_buffers, out)

function _run_gpu_kernel!(
    dev, out, field, ws::GPUFootprintND, grid, ::CGEF.Filtering.AbstractMaskStrategy, cells::UnitRange{Int},
)
    dims = FlowGeometries.Grids.size_tuple(grid)
    _cgef_filter_kernel_nd!(dev)(
        out, field, ws.maskd, ws.offsets, ws.w, ws.invden, dims, FlowGeometries.Grids.periodic_flags(grid),
        first(cells) - 1; ndrange = (length(cells), size(out, length(dims) + 1)),
    )
end

function _run_gpu_kernel!(
    dev, out, field, ws::GPUNodeFootprint, grid, strategy::CGEF.Filtering.AbstractMaskStrategy, nodes::UnitRange{Int},
)
    _cgef_filter_kernel_node!(dev)(
        out, field, ws.maskd, ws.nbrs, ws.w, ws.ptr, strategy, first(nodes) - 1;
        ndrange = (length(nodes), size(out, 2)),
    )
end

function _run_gpu_kernel!(
    dev, out, field, ws::GPUScatteredCached, grid, strategy::CGEF.Filtering.AbstractMaskStrategy, rows::UnitRange{Int},
)
    _cgef_filter_kernel_scattered!(dev)(
        out, field, ws.maskd, ws.ii, ws.jj, ws.w, ws.ptr, ws.exterior, strategy, first(rows) - 1;
        ndrange = (size(out, 1), length(rows), size(out, 3)),
    )
end

function _run_gpu_kernel!(
    dev, out, field, ws::GPUStreamingND, grid, strategy::CGEF.Filtering.AbstractMaskStrategy, cells::UnitRange{Int},
)
    dims = FlowGeometries.Grids.size_tuple(grid)
    _cgef_filter_kernel_nd_traversal!(dev)(
        out, field, ws.maskd, ws.grid, ws.topology, ws.weights, ws.fp.rad, ws.exterior, dims, strategy,
        first(cells) - 1; ndrange = (length(cells), size(out, length(dims) + 1)),
    )
end

function _run_gpu_kernel!(
    dev, out, field, ws::GPUStreaming, grid, strategy::CGEF.Filtering.AbstractMaskStrategy, rows::UnitRange{Int},
)
    fp = ws.fp
    _cgef_filter_kernel_traversal!(dev)(
        out, field, ws.maskd, ws.grid, ws.topology, ws.weights, fp.rad, ws.exterior,
        strategy, first(rows) - 1; ndrange = (size(out, 1), length(rows), size(out, 3)),
    )
end

# Pass buffers for one apply: the resident pair for a single slice, and for a batch a batch-shaped pair
# held on the workspace.
@inline _sep_buffers(dev, ws::GPUSeparable, ::AbstractArray{<:Any,2}) = (ws.masked_input, ws.row_pass)

@inline _sep_buffers(dev, ws::GPUSeparable, out::AbstractArray{<:Any,3}) = _held_pair(dev, ws.batch_buffers, out)

# A pair of `out`-shaped device buffers held in `ref`, allocated again when `out`'s shape changes.
function _held_pair(dev, ref::Base.RefValue, out::AbstractArray{T}) where {T}
    cached = ref[]
    (cached !== nothing && size(cached[1]) == size(out)) && return cached
    bufs = (KA.allocate(dev, T, size(out)...), KA.allocate(dev, T, size(out)...))
    ref[] = bufs
    return bufs
end

# A footprint with no kernel here holds its arrays in the device's memory and applies by broadcasts and
# library transforms.
_run_gpu_kernel!(dev, out, field, fp, grid, strategy::CGEF.Filtering.AbstractMaskStrategy) =
    CGEF.Filtering._resident_apply!(out, field, grid, fp, strategy)

# The column pass reads other rows of the row pass, so the row pass covers every row whatever rows the
# column pass writes; one backend runs the two launches in order. The normalization is a broadcast, which
# on a host-memory backend runs on the host, so it waits for both.
function _run_gpu_kernel!(
    dev, out, field, ws::GPUSeparable, grid, ::CGEF.Filtering.AbstractMaskStrategy, rows::UnitRange{Int},
)
    T = eltype(out)
    nd = (size(out, 1), size(out, 2), size(out, 3))
    fp = ws.fp
    masked, rowp = _sep_buffers(dev, ws, out)
    masked .= ws.maskd .* field
    _cgef_separable_row_pass_kernel!(dev)(rowp, masked, ws.gx, fp.di_lim, fp.periodic_x; ndrange = nd)
    _cgef_separable_column_pass_kernel!(dev)(out, rowp, ws.gy, fp.dj_lim, fp.periodic_y, first(rows) - 1;
                                             ndrange = (nd[1], length(rows), nd[3]))
    KA.synchronize(dev)
    o = _rows_view(out, rows)
    if ws.invrenorm === nothing
        o .= CGEF.Filtering._normalized.(o, view(ws.denom, :, rows))
    else
        # Deformable on a masked grid: an inactive target is zero.
        o .= ifelse.(view(ws.maskd, :, rows), o .* view(ws.invrenorm, :, rows), zero(T))
    end
    return out
end

# Rows `rows` of an output, its batch axis whole.
@inline _rows_view(A::AbstractMatrix, rows) = view(A, :, rows)
@inline _rows_view(A::AbstractArray{<:Any,3}, rows) = view(A, :, rows, :)

# The host footprint a resident one wraps carries the strategy its denominator was built for, and its window mass.
CGEF.Filtering._host_footprint(ws::GPUResident) = ws.fp
_check_resident_strategy(ws::GPUResident, s) = CGEF.Filtering._check_strategy(ws.fp, s)
_check_resident_strategy(fp, s) = CGEF.Filtering._check_strategy(fp, s)

# A device-resident footprint for `ws`: `ws` itself when resident, otherwise uploaded here (and first
# built when `ws === nothing`).
@inline _resident(::CGEF.ComputationalBackends.GPUBackend, grid, ws::GPUResident, kernel, scale, mask_strategy) = ws
@inline _resident(b::CGEF.ComputationalBackends.GPUBackend, grid, ws, kernel, scale, mask_strategy) =
    CGEF.Filtering.prepare_workspace(b, grid, ws)
@inline _resident(b::CGEF.ComputationalBackends.GPUBackend, grid, ::Nothing, kernel, scale, mask_strategy) =
    CGEF.Filtering.prepare_workspace(b, grid, CGEF.Filtering.build_footprint(grid, kernel, scale; mask_strategy = mask_strategy))

# On a `CurvilinearGrid`, `isperiodic` is `AbstractGrid`'s default, `false`.
function CGEF.Filtering.gpu_filter_field!(
    gpu_backend::CGEF.ComputationalBackends.GPUBackend,
    out::AbstractMatrix{T},
    field::AbstractMatrix{T},
    grid::Union{FlowGeometries.Grids.StructuredGrid{T,G,2}, FlowGeometries.Grids.CurvilinearGrid{T,G}},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
    workspace,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    dev = gpu_backend.backend
    ws = _resident(gpu_backend, grid, workspace, kernel, scale, mask_strategy)
    _check_resident_strategy(ws, mask_strategy)
    _run_gpu_kernel!(dev, out, field, ws, grid, mask_strategy)
    KA.synchronize(dev)
    return out
end

# 1-D and 3-D structured grids; the 2-D method above is more specific and takes `N == 2`.
function CGEF.Filtering.gpu_filter_field!(
    gpu_backend::CGEF.ComputationalBackends.GPUBackend,
    out::AbstractArray{T,N},
    field::AbstractArray{T,N},
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
    workspace,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}, N}
    dev = gpu_backend.backend
    ws = _resident(gpu_backend, grid, workspace, kernel, scale, mask_strategy)
    _check_resident_strategy(ws, mask_strategy)
    _run_gpu_kernel!(dev, out, field, ws, grid, mask_strategy)
    KA.synchronize(dev)
    return out
end

# The node CSR engine on any flat-cell grid, selected by the `workspace` type. A 1-D `StructuredGrid`
# also has a vector output; its ND footprint takes the rank-generic method above.
function CGEF.Filtering.gpu_filter_field!(
    gpu_backend::CGEF.ComputationalBackends.GPUBackend,
    out::AbstractVector{T},
    field::AbstractVector{T},
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
    workspace::Union{CGEF.Filtering.NodeFilterPlan, GPUNodeFootprint},
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    dev = gpu_backend.backend
    ws = _resident(gpu_backend, grid, workspace, kernel, scale, mask_strategy)
    _run_gpu_kernel!(dev, out, field, ws, grid, mask_strategy)
    KA.synchronize(dev)
    return out
end

# ---------------------------------------------------------------------------
# Several separate fields, one device footprint, one host wait
# ---------------------------------------------------------------------------
#
# Each field is an independent launch over the same resident footprint; the host synchronizes once.
function CGEF.Filtering.gpu_filter_fields!(
    gpu_backend::CGEF.ComputationalBackends.GPUBackend,
    outs, fields,
    grid::FlowGeometries.Grids.AbstractGrid,
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
    workspace,
) where {T<:AbstractFloat}
    dev = gpu_backend.backend
    ws = _resident(gpu_backend, grid, workspace, kernel, scale, mask_strategy)
    _check_resident_strategy(ws, mask_strategy)
    S = Val(ndims(FlowGeometries.Grids.mask(grid)))
    # The separable and prefix-sum engines pass every field through the same workspace buffers; one
    # backend runs the launches in order, so each field's writes follow the last field's reads.
    for k in eachindex(outs)
        _run_gpu_kernel!(dev, _one_batch_axis(outs[k], S), _one_batch_axis(fields[k], S), ws, grid, mask_strategy)
    end
    KA.synchronize(dev)
    return outs
end

# ---------------------------------------------------------------------------
# A process's share of a real-space apply on its device
# ---------------------------------------------------------------------------
#
# Under a process backend whose inner backend is a `GPUBackend`, each process holds the device-resident
# footprint and device copies of the field and the output. A share is one pass: the field copied in whole,
# every pass of the engine but the last over the whole grid, the last over the process's block of units,
# and the block's cells copied into the process's output. The host buffer carries the output through an
# MPI reduction.

struct GPUShare{W, D, A<:AbstractArray, H<:Array} <: CGEF.Filtering.AbstractProcessShare
    ws::W       # the device-resident footprint
    dev::D
    field::A
    out::A
    host::H
end

function CGEF.Filtering._share_workspace(
    b::CGEF.ComputationalBackends.GPUBackend, grid::FlowGeometries.Grids.AbstractGrid{G,T}, fp,
) where {G<:FlowGeometries.Geometry.AbstractGeometry, T<:AbstractFloat}
    dev = b.backend
    dims = size(FlowGeometries.Grids.mask(grid))
    return GPUShare(CGEF.Filtering.prepare_workspace(b, grid, fp), dev, KA.zeros(dev, T, dims...),
                    KA.zeros(dev, T, dims...), zeros(T, dims))
end

CGEF.Filtering._process_passes(::CGEF.ComputationalBackends.GPUBackend, _) = Val(1)
CGEF.Filtering._share_passes(::GPUShare) = Val(1)
CGEF.Filtering._share_dst(out, ::GPUShare, ::Val{1}) = (out,)
CGEF.Filtering._share_staging(s::GPUShare) = s.host
CGEF.Filtering._check_strategy(s::GPUShare, strategy::CGEF.Filtering.AbstractMaskStrategy) =
    _check_resident_strategy(s.ws, strategy)
CGEF.Filtering._host_footprint(s::GPUShare) = CGEF.Filtering._host_footprint(s.ws)

function CGEF.Filtering._share_pass!(
    out, field, grid, s::GPUShare, strategy::CGEF.Filtering.AbstractMaskStrategy, ::Val{1}, k::Int, nk::Int, _,
)
    copyto!(s.field, field)
    units = CGEF.Filtering._share_block(k, nk, _gpu_units(s.ws, grid))
    isempty(units) && return nothing
    _run_gpu_kernel!(s.dev, s.out, s.field, s.ws, grid, strategy, units)
    KA.synchronize(s.dev)
    cells = _gpu_cells(s.ws, grid, units)
    copyto!(out, first(cells), s.out, first(cells), length(cells))
    return nothing
end

# Folds trailing batch axes beyond the first into one, since a kernel launches over the spatial axes and
# one batch axis. The result shares `a`'s memory.
@inline _one_batch_axis(a::AbstractArray, ::Val{S}) where {S} =
    ndims(a) <= S + 1 ? a : reshape(a, ntuple(d -> size(a, d), Val(S))..., :)

end # module
