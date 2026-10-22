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

# Sums the taps in the host's `k` order and scales by the host-built reciprocal `invden`, so the result
# is bit-identical to serial. An inactive source contributes `w[k] * 0`, as in the host's `mask · field`;
# `invden` is zero at a target the strategy does not filter.
@kernel function _cgef_filter_kernel!(
    out, @Const(field), @Const(mask), @Const(rdj), @Const(rdi), @Const(rstart), @Const(w), @Const(ptr),
    @Const(invden), nbands::Int, periodic_x::Bool, periodic_y::Bool,
)
    i, j, b = @index(Global, NTuple)
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

# 1-D / 3-D grids: walks the translation-invariant offset table per point, as the serial
# `_footprint_nd_point` does.
@kernel function _cgef_filter_kernel_nd!(
    out, @Const(field), @Const(mask), @Const(offsets), @Const(w), exterior,
    dims, periodic, is_zerofill::Bool,
)
    lin, b = @index(Global, NTuple)
    T = eltype(out)
    nspatial = prod(dims)
    Nb = size(out, length(dims) + 1)
    if lin <= nspatial && b <= Nb
        I = CartesianIndices(dims)[lin]
        if is_zerofill || mask[I]
            Ti = Tuple(I)
            ws = zero(T)
            wn = zero(T)
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
            end
            is_zerofill && (wn = CGEF.Filtering._add_exterior(wn, exterior, I))
            _setI!(out, I, b, CGEF.Filtering._normalized(ws, wn))
        else
            _setI!(out, I, b, zero(T))
        end
    end
end

# Driver for `Filtering.apply_separable_nd!`: applies `f` once per index on the device and synchronizes
# before returning, so each pass completes before the next starts.
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
    KA.synchronize(drv.dev)
    return nothing
end

# Cached ND scattered footprint: `nbrs` holds neighbour multi-indices with the periodic wrap applied at
# build time.
@kernel function _cgef_filter_kernel_nd_cached!(
    out, @Const(field), @Const(mask), @Const(nbrs), @Const(w), @Const(ptr), exterior, dims,
    is_zerofill::Bool,
)
    lin, b = @index(Global, NTuple)
    T = eltype(out)
    nspatial = prod(dims)
    Nb = size(out, length(dims) + 1)
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
    out, @Const(field), @Const(mask), @Const(nbrs), @Const(w), @Const(ptr), is_zerofill::Bool,
)
    t, b = @index(Global, NTuple)
    T = eltype(out)
    Nn, Nb = size(out, 1), size(out, 2)
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
    exterior, is_zerofill::Bool,
)
    i, j, b = @index(Global, NTuple)
    T = eltype(out)
    Nx, Ny, Nb = size(out, 1), size(out, 2), size(out, 3)
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
    out, @Const(field), @Const(mask), grid, mt, weights, rad, exterior, is_cartesian::Bool,
    is_zerofill::Bool,
)
    i, j, bi = @index(Global, NTuple)
    T = eltype(out)
    Nx, Ny, Nb = size(out, 1), size(out, 2), size(out, 3)
    if i <= Nx && j <= Ny && bi <= Nb
        if is_zerofill || mask[i, j]
            # The fold closure assigns no captured variable: Julia boxes one that is assigned, and a
            # device kernel cannot hold a box.
            t = Int(i) + (Int(j) - 1) * Nx
            ws, wn = CGEF.Filtering._ball_fold(
                (zero(T), zero(T)), grid, Int(i), Int(j), rad, is_cartesian, mt,
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

@kernel function _cgef_separable_column_pass_kernel!(out, @Const(row_pass), @Const(gy), dj_lim::Int, periodic_y::Bool)
    i, j, b = @index(Global, NTuple)
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

# First index with `xe[idx] >= v`, or `n + 1`.
@inline function _xe_lower(xe, n::Int, v)
    lo = 1
    hi = n + 1
    while lo < hi
        mid = (lo + hi) >>> 1
        if @inbounds(xe[mid]) < v
            lo = mid + 1
        else
            hi = mid
        end
    end
    return lo
end

# Last index with `xe[idx] <= v`, or `0`.
@inline function _xe_upper(xe, n::Int, v)
    lo = 1
    hi = n + 1
    while lo < hi
        mid = (lo + hi) >>> 1
        if @inbounds(xe[mid]) <= v
            lo = mid + 1
        else
            hi = mid
        end
    end
    return lo - 1
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
    @Const(wcell), dj_lim::Int, ne::Int, Nx::Int, Ny::Int, tiles_x::Bool, rings_x::Bool, tiles_y::Bool,
    x_period,
)
    i, j, b = @index(Global, NTuple)
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
                    lo = max(1, i - w)
                    hi = min(Nx, i + w)
                    acc += wyj * CGEF.Filtering._scan_diff(Ph, Pl, hi + 1, lo, jj, b)
                elseif -a <= c
                    lo = _xe_lower(xe, ne, xc - a)
                    hi = _xe_upper(xe, ne, xc + c)
                    if hi >= lo
                        acc += wyj * CGEF.Filtering._scan_diff(Ph, Pl, hi + 1, lo, jj, b)
                    end
                end
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
    out, @Const(Ph), @Const(Pl), @Const(wcell), @Const(invden), dims,
    dj_lim::Int, dk_lim::Int, px::Bool, py::Bool, pz::Bool,
)
    lin, b = @index(Global, NTuple)
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

_device_weights(_, r::CGEF.Filtering.RadialWeights) = r
_device_weights(dev, r::CGEF.Filtering.HighOrderWeights) = CGEF.Filtering._with_coeffs(r, move(dev, r.coeffs))

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

# Separable Gaussian on a 1-D / 3-D grid: the host footprint rebuilt with device arrays.
struct GPUSeparableND{F} <: GPUResident
    fp::F
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
struct GPUFootprintND{F, VO, VW, MA, EX} <: GPUResident
    fp::F
    offsets::VO
    w::VW
    maskd::MA
    exterior::EX
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

struct GPUPrefixSum3D{F, WT, AT, MA, PB} <: GPUResident
    fp::F
    wcell::WT
    invden::AT
    maskd::MA
    P::PB
    batch_scan::Base.RefValue{Union{Nothing,PB}}
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

_maskd(dev, grid) = move(dev, Array{Bool}(FlowGeometries.Grids.mask(grid)))
_moved(_, ::Nothing) = nothing
_moved(dev, x::AbstractArray) = move(dev, x)

function CGEF.Filtering.prepare_workspace(
    b::CGEF.ComputationalBackends.GPUBackend, grid::FlowGeometries.Grids.AbstractGrid,
    fp::CGEF.Filtering.PrefixSumTopHatPlan{T},
) where {T<:AbstractFloat}
    dev = b.backend
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

function CGEF.Filtering.prepare_workspace(
    b::CGEF.ComputationalBackends.GPUBackend, grid::FlowGeometries.Grids.AbstractGrid,
    fp::CGEF.Filtering.PrefixSumTopHat3DPlan{T},
) where {T<:AbstractFloat}
    dev = b.backend
    Nx, Ny, Nz = FlowGeometries.Grids.size_tuple(grid)
    P = _device_scan(dev, T, Nx + 1, Ny, Nz, 1)
    return GPUPrefixSum3D(
        fp, move(dev, fp.wcell), move(dev, fp.invden), _maskd(dev, grid), P, _batch_scan_ref(P),
    )
end

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

# The ND device kernel reads a materialized neighbour list, so a plan built with `NeverCache` is rebuilt
# here with `AlwaysCache`, at the cache's memory cost; the filtered values are unchanged.
function CGEF.Filtering.prepare_workspace(
    b::CGEF.ComputationalBackends.GPUBackend, grid::FlowGeometries.Grids.AbstractGrid,
    fp::CGEF.Filtering.NDScatteredFilterPlan,
)
    dev = b.backend
    cached = fp.cache === nothing ?
        CGEF.Filtering.build_footprint(grid, fp.kernel, fp.scale;
            cache_strategy = CGEF.Filtering.AlwaysCache()) : fp
    cache = cached.cache
    return GPUScatteredNDCached(
        cached, move(dev, cache.nbrs), move(dev, cache.w), move(dev, cache.ptr), _maskd(dev, grid),
        _moved(dev, cached.exterior),
    )
end

function CGEF.Filtering.prepare_workspace(
    b::CGEF.ComputationalBackends.GPUBackend, grid::FlowGeometries.Grids.AbstractGrid,
    fp::CGEF.Filtering.SeparableFootprintND{N,T},
) where {N, T<:AbstractFloat}
    dev = b.backend
    return GPUSeparableND(CGEF.Filtering.SeparableFootprintND(
        map(g -> move(dev, g), fp.g), fp.lim, fp.periodic,
        fp.profiles === nothing ? nothing : map(pv -> move(dev, pv), fp.profiles),
        _moved(dev, fp.invrenorm), fp.strategy, fp.masked, fp.bound,
        move(dev, fp.masked_input), move(dev, fp.scratch),
    ))
end

function CGEF.Filtering.prepare_workspace(
    b::CGEF.ComputationalBackends.GPUBackend, grid::FlowGeometries.Grids.AbstractGrid,
    fp::CGEF.Filtering.FilterFootprintND,
)
    dev = b.backend
    return GPUFootprintND(
        fp, move(dev, fp.offsets), move(dev, fp.w), _maskd(dev, grid), _moved(dev, fp.exterior),
    )
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
    cache = fp.cache
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

# The banded engine serves only 2-D structured grids, so a 3-D `out` carries a batch axis.
function _run_gpu_kernel!(dev, out, field, ws::GPUBandedFootprint, grid, periodic_x::Bool, periodic_y::Bool, is_zerofill::Bool)
    # `is_zerofill` is unused: the strategy is built into `invden`.
    _cgef_filter_kernel!(dev)(
        out, field, ws.maskd, ws.dj, ws.di0, ws.start, ws.w, ws.ptr, ws.invden, ws.fp.nbands,
        ws.fp.periodic_x, ws.fp.periodic_y;
        ndrange = (size(out, 1), size(out, 2), size(out, 3)),
    )
end

# `plan.footprint` on a GPU plan is the device-resident footprint `prepare_workspace` built.
CGEF.Filtering._gpu_batched_supported(plan::CGEF.Filtering.PhysicalFilterPlan) =
    plan.footprint isa Union{GPUBandedFootprint, GPUSeparable, GPUNodeFootprint, GPUFootprintND, GPUSeparableND,
                             GPUScatteredCached, GPUScatteredNDCached, GPUStreaming,
                             GPUPrefixSum, GPUPrefixSum3D}

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
    _run_gpu_kernel!(dev, out, field, ws, grid, false, false, mask_strategy isa CGEF.Filtering.ZeroFill)
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
    # A 1-D grid has no direction 2. The ND engines read `periodic_flags` and ignore `px`/`py`.
    nspatial = length(FlowGeometries.Grids.size_tuple(grid))
    px = FlowGeometries.Grids.isperiodic(grid, 1)
    py = nspatial >= 2 ? FlowGeometries.Grids.isperiodic(grid, 2) : false
    _run_gpu_kernel!(dev, out, field, ws, grid, px, py, mask_strategy isa CGEF.Filtering.ZeroFill)
    KA.synchronize(dev)
    return out
end

# Two launches, ordered: the sweep reads every row of the scan, so it waits on the whole scan.
function _run_gpu_kernel!(dev, out, field, ws::GPUPrefixSum, grid, periodic_x::Bool, periodic_y::Bool, is_zerofill::Bool)
    fp = ws.fp
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    Nb = ndims(out) == 2 ? 1 : size(out, 3)
    P = _scan_table(dev, ws.P, ws.batch_scan, (ws.ne + 1, Ny, Nb))
    _cgef_prefixsum_scan_kernel!(dev)(P.hi, P.lo, field, ws.maskd, ws.src, ws.wx, ws.ne; ndrange = (Ny, Nb))
    KA.synchronize(dev)
    gp = fp.grid_plan
    _cgef_prefixsum_sweep_kernel!(dev)(
        out, P.hi, P.lo, ws.x, ws.xe, ws.wy, ws.invden, ws.hw, ws.wcell, fp.dj_lim, ws.ne, Nx, Ny,
        gp.tiles_x, gp.rings_x, gp.tiles_y, gp.x_period; ndrange = (Nx, Ny, Nb),
    )
    return out
end

function _run_gpu_kernel!(dev, out, field, ws::GPUPrefixSum3D, grid, periodic_x::Bool, periodic_y::Bool, is_zerofill::Bool)
    fp = ws.fp
    dims = FlowGeometries.Grids.size_tuple(grid)
    Nx, Ny, Nz = dims
    Nb = ndims(out) == 3 ? 1 : size(out, 4)
    P = _scan_table(dev, ws.P, ws.batch_scan, (Nx + 1, Ny, Nz, Nb))
    _cgef_prefixsum3d_scan_kernel!(dev)(P.hi, P.lo, field, ws.maskd, fp.masked, Nx; ndrange = (Ny, Nz, Nb))
    KA.synchronize(dev)
    _cgef_prefixsum3d_sweep_kernel!(dev)(
        out, P.hi, P.lo, ws.wcell, ws.invden, dims, fp.dj_lim, fp.dk_lim,
        fp.periodic[1], fp.periodic[2], fp.periodic[3]; ndrange = (prod(dims), Nb),
    )
    return out
end

function _run_gpu_kernel!(dev, out, field, ws::GPUScatteredNDCached, grid, periodic_x::Bool, periodic_y::Bool, is_zerofill::Bool)
    dims = FlowGeometries.Grids.size_tuple(grid)
    _cgef_filter_kernel_nd_cached!(dev)(
        out, field, ws.maskd, ws.nbrs, ws.w, ws.ptr, ws.exterior, dims, is_zerofill;
        ndrange = (prod(dims), size(out, length(dims) + 1)),
    )
end

function _run_gpu_kernel!(dev, out, field, ws::GPUSeparableND, grid, periodic_x::Bool, periodic_y::Bool, is_zerofill::Bool)
    strategy = is_zerofill ? CGEF.Filtering.ZeroFill() : CGEF.Filtering.Deformable()
    CGEF.Filtering.apply_separable_nd!(out, field, grid, ws.fp, strategy, GPUDriver(dev))
end

function _run_gpu_kernel!(dev, out, field, ws::GPUFootprintND, grid, periodic_x::Bool, periodic_y::Bool, is_zerofill::Bool)
    dims = FlowGeometries.Grids.size_tuple(grid)
    _cgef_filter_kernel_nd!(dev)(
        out, field, ws.maskd, ws.offsets, ws.w, ws.exterior,
        dims, FlowGeometries.Grids.periodic_flags(grid), is_zerofill;
        ndrange = (prod(dims), size(out, length(dims) + 1)),
    )
end

function _run_gpu_kernel!(dev, out, field, ws::GPUNodeFootprint, grid, periodic_x::Bool, periodic_y::Bool, is_zerofill::Bool)
    _cgef_filter_kernel_node!(dev)(
        out, field, ws.maskd, ws.nbrs, ws.w, ws.ptr, is_zerofill;
        ndrange = (size(out, 1), size(out, 2)),
    )
end

function _run_gpu_kernel!(dev, out, field, ws::GPUScatteredCached, grid, periodic_x::Bool, periodic_y::Bool, is_zerofill::Bool)
    _cgef_filter_kernel_scattered!(dev)(
        out, field, ws.maskd, ws.ii, ws.jj, ws.w, ws.ptr, ws.exterior, is_zerofill;
        ndrange = (size(out, 1), size(out, 2), size(out, 3)),
    )
end

function _run_gpu_kernel!(dev, out, field, ws::GPUStreaming, grid, periodic_x::Bool, periodic_y::Bool, is_zerofill::Bool)
    fp = ws.fp
    _cgef_filter_kernel_traversal!(dev)(
        out, field, ws.maskd, ws.grid, ws.topology, ws.weights, fp.rad, ws.exterior,
        fp.is_cartesian, is_zerofill; ndrange = (size(out, 1), size(out, 2), size(out, 3)),
    )
end

# Pass buffers for one apply: the resident pair for a single slice, and for a batch a batch-shaped pair
# held on the workspace.
@inline _sep_buffers(dev, ws::GPUSeparable, ::AbstractArray{<:Any,2}) = (ws.masked_input, ws.row_pass)

function _sep_buffers(dev, ws::GPUSeparable, out::AbstractArray{T,3}) where {T}
    cached = ws.batch_buffers[]
    if cached !== nothing && size(cached[1]) == size(out)
        return cached
    end
    bufs = (KA.allocate(dev, T, size(out)...), KA.allocate(dev, T, size(out)...))
    ws.batch_buffers[] = bufs
    return bufs
end

# The column pass reads other rows of the row pass, so the two launches are synchronized in between.
function _run_gpu_kernel!(dev, out, field, ws::GPUSeparable, grid, periodic_x::Bool, periodic_y::Bool, is_zerofill::Bool)
    T = eltype(out)
    nd = (size(out, 1), size(out, 2), size(out, 3))
    fp = ws.fp
    masked, rowp = _sep_buffers(dev, ws, out)
    masked .= ws.maskd .* field
    _cgef_separable_row_pass_kernel!(dev)(rowp, masked, ws.gx, fp.di_lim, fp.periodic_x; ndrange = nd)
    KA.synchronize(dev)
    _cgef_separable_column_pass_kernel!(dev)(out, rowp, ws.gy, fp.dj_lim, fp.periodic_y; ndrange = nd)
    KA.synchronize(dev)
    if ws.invrenorm === nothing
        out .= CGEF.Filtering._normalized.(out, ws.denom)
    else
        # Deformable on a masked grid: an inactive target is zero.
        out .= ifelse.(ws.maskd, out .* ws.invrenorm, zero(T))
    end
    return out
end

# The host footprint a resident one wraps carries the strategy its denominator was built for.
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
    is_zerofill = mask_strategy isa CGEF.Filtering.ZeroFill
    _run_gpu_kernel!(
        dev, out, field, ws, grid,
        FlowGeometries.Grids.isperiodic(grid, 1), FlowGeometries.Grids.isperiodic(grid, 2), is_zerofill,
    )
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
    _run_gpu_kernel!(
        dev, out, field, ws, grid, false, false, mask_strategy isa CGEF.Filtering.ZeroFill,
    )
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
    _run_gpu_kernel!(
        dev, out, field, ws, grid, false, false, mask_strategy isa CGEF.Filtering.ZeroFill,
    )
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
    nspatial = length(FlowGeometries.Grids.size_tuple(grid))
    px = FlowGeometries.Grids.isperiodic(grid, 1)
    py = nspatial >= 2 ? FlowGeometries.Grids.isperiodic(grid, 2) : false
    is_zerofill = mask_strategy isa CGEF.Filtering.ZeroFill
    # The separable and prefix-sum engines pass their fields through workspace buffers every launch
    # overwrites, so their applies stay serialized behind one another.
    serial = ws isa Union{GPUSeparable, GPUPrefixSum, GPUPrefixSum3D}
    for k in eachindex(outs)
        _run_gpu_kernel!(dev, _one_batch_axis(outs[k], S), _one_batch_axis(fields[k], S), ws, grid, px, py,
                         is_zerofill)
        serial && KA.synchronize(dev)
    end
    serial || KA.synchronize(dev)
    return outs
end

# Folds trailing batch axes beyond the first into one, since a kernel launches over the spatial axes and
# one batch axis. The result shares `a`'s memory.
@inline _one_batch_axis(a::AbstractArray, ::Val{S}) where {S} =
    ndims(a) <= S + 1 ? a : reshape(a, ntuple(d -> size(a, d), Val(S))..., :)

end # module
