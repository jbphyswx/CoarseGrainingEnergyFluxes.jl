# ---------------------------------------------------------------------------
# Exact top-hat over a Cartesian point set by binned prefix sums
# ---------------------------------------------------------------------------
#
# The points (a node set's nodes, a curvilinear mesh's cell centres) are binned on a lattice of boxes about
# one mean spacing wide. For a target in bin `(i₀, J₀)` and a line of bins `J₀ + o` along axis 1, the bins
# `|i − i₀| ≤ w_in(o)` lie wholly inside the ball for any target in the bin, and those with
# `w_in(o) < |i − i₀| ≤ w_out(o)` may hold points inside it; no farther bin does. The inner run adds the
# totals of its bins, an interval of a prefix scan along the line, and only the bins between are summed
# point by point. The offsets and their widths depend on the scale alone, so a target costs one interval
# and a few partial bins per line within reach: `O((r/h)^{D−1})`, where a neighbour list holds
# `O((r/h)^D)` points.

"""
    BinnedPoints{D,T}

A Cartesian point set binned on a lattice of `nb` boxes of widths `h` from `lo`, axis 1 fastest: bin
`b`'s points are `order[ptr[b]:ptr[b+1]-1]`, and `x`, `measure` and `mask` hold the points in that order.
Along a periodic direction the lattice tiles the period, and `x` is reduced into `[lo, lo + period)`.
"""
struct BinnedPoints{D, T<:AbstractFloat, VI<:AbstractVector{Int}, VT<:AbstractVector{T}, MK}
    lo::NTuple{D,T}
    h::NTuple{D,T}
    nb::NTuple{D,Int}
    periodic::NTuple{D,Bool}
    period::NTuple{D,T}
    order::VI
    ptr::VI
    x::NTuple{D,VT}
    measure::VT
    mask::MK       # `Vector{Bool}` in bin order, or `nothing` when every point is active
end

"""
    BinnedTopHatPlan{D,T}

The top-hat of radius `rad` over a [`BinnedPoints`](@ref) set: per offset `lines[q]` of a line of bins
across axes `2…D`, the half-widths `win[q]` of its run of bins inside the ball for any target in the bin
(`-1` for none) and `wout[q]` of the bins that may reach it; the per-apply buffers, one
`(values, scan, sums)` triple per field in flight (the weighted values in bin order, the compensated scan
of the bin totals along axis 1, and each target's window sum in bin order), added on demand by a batched
apply; and `invden`, the reciprocal window mass under
`strategy` per point in the grid's order, zero where the window is empty or, under `Deformable`, the
target inactive. `bound` is whether the two strategies' denominators differ, which they do on a masked
grid.
"""
struct BinnedTopHatPlan{
    D, T<:AbstractFloat, BP<:BinnedPoints{D,T}, VL<:AbstractVector, VI<:AbstractVector{Int},
    VB<:AbstractVector, VT<:AbstractVector{T}, MS<:AbstractMaskStrategy,
}
    points::BP
    rad::T
    lines::VL
    win::VI
    wout::VI
    buffers::VB
    invden::VT
    strategy::MS
    bound::Bool
end

Base.show(io::IO, fp::BinnedTopHatPlan) =
    print(io, "BinnedTopHatPlan(", length(fp.points.order), " points, ", join(fp.points.nb, "×"), " bins)")

_driven_footprint(::BinnedTopHatPlan) = true

"""
    _binned_lattice(grid, Val(D)) -> (lo, h, nb, periodic, period)

The lattice a point set is binned on: origin `lo`, widths `h` and counts `nb`, the width the mean cell
measure, about one point per bin, along a bounded direction, and the period divided into a whole number
of bins along a periodic one.
"""
function _binned_lattice(grid::FlowGeometries.Grids.AbstractGrid{G,T}, ::Val{D}) where {G, T<:AbstractFloat, D}
    n = length(FlowGeometries.Grids.mask(grid))
    periodic = ntuple(d -> FlowGeometries.Grids.isperiodic(grid, d), Val(D))
    period = ntuple(d -> periodic[d] ? T(FlowGeometries.Grids.period(grid, d)) : zero(T), Val(D))
    lo = ntuple(d -> T(FlowGeometries.Grids.bounds(grid, d)[1]), Val(D))
    extent = ntuple(d -> T(FlowGeometries.Grids.bounds(grid, d)[2]) - lo[d], Val(D))
    h0 = (T(sum(FlowGeometries.Grids.measure(grid))) / n)^(one(T) / D)
    nb = ntuple(d -> periodic[d] ? max(1, floor(Int, period[d] / h0)) : floor(Int, extent[d] / h0) + 1, Val(D))
    h = ntuple(d -> periodic[d] ? period[d] / nb[d] : h0, Val(D))
    return lo, h, nb, periodic, period
end

function BinnedPoints(grid::FlowGeometries.Grids.AbstractGrid{G,T}, ::Val{D}) where {G, T<:AbstractFloat, D}
    xs = ntuple(d -> vec(FlowGeometries.Grids.coordinates(grid, d)), Val(D))
    n = length(first(xs))
    measure = T.(vec(FlowGeometries.Grids.measure(grid)))
    mask = FlowGeometries.Grids.mask(grid)
    lo, h, nb, periodic, period = _binned_lattice(grid, Val(D))
    # Each point's coordinates reduced into the lattice's period, and its bin.
    xr = ntuple(Val(D)) do d
        periodic[d] ? [lo[d] + mod(x - lo[d], period[d]) for x in xs[d]] : collect(T, xs[d])
    end
    cart = ntuple(d -> [clamp(floor(Int, (v - lo[d]) / h[d]), 0, nb[d] - 1) for v in xr[d]], Val(D))
    strides = accumulate(*, (1, Base.front(nb)...))
    bin = [1 + sum(ntuple(d -> cart[d][j] * strides[d], Val(D))) for j in 1:n]
    # Counting sort by bin.
    nbins = prod(nb)
    ptr = zeros(Int, nbins + 1)
    for b in bin
        ptr[b + 1] += 1
    end
    ptr[1] = 1
    for b in 1:nbins
        ptr[b + 1] += ptr[b]
    end
    order = Vector{Int}(undef, n)
    next = ptr[1:nbins]
    for j in 1:n
        order[next[bin[j]]] = j
        next[bin[j]] += 1
    end
    active = all(mask) ? nothing : collect(Bool, vec(mask))[order]
    return BinnedPoints(lo, h, nb, periodic, period, order, ptr, ntuple(d -> xr[d][order], Val(D)),
                        measure[order], active)
end

"""
    _binned_lines(h, rin, rout) -> (lines, win, wout)

The offsets `o` across axes `2…D` of the lines of bins a ball of radius `rout` about a point of bin
`(i₀, J₀)` reaches, with `win`, the largest `|i − i₀|` of a bin of line `J₀ + o` wholly inside radius `rin`
from any point of bin `(i₀, J₀)` (`-1` when none is), and `wout`, the largest of a bin with a point within
`rout` of one. A box offset by `k` bins of width `h` along a direction is between `max(|k| − 1, 0)·h` and
`(|k| + 1)·h` from another.
"""
function _binned_lines(h::NTuple{D,T}, rin::T, rout::T) where {D,T}
    reach = ntuple(d -> ceil(Int, rout / h[d + 1]) + 1, Val(D - 1))
    lines = NTuple{D - 1,Int}[]
    win = Int[]
    wout = Int[]
    for o in CartesianIndices(map(q -> (-q):q, reach))
        t = Tuple(o)
        near = sum(ntuple(d -> (max(abs(t[d]) - 1, 0) * h[d + 1])^2, Val(D - 1)); init = zero(T))
        far = sum(ntuple(d -> ((abs(t[d]) + 1) * h[d + 1])^2, Val(D - 1)); init = zero(T))
        near <= rout^2 || continue
        push!(lines, t)
        push!(wout, floor(Int, sqrt(rout^2 - near) / h[1]) + 1)
        push!(win, far <= rin^2 ? max(floor(Int, sqrt(rin^2 - far) / h[1]) - 1, -1) : -1)
    end
    return lines, win, wout
end

"""
    binned_tophat_plan(grid, kernel::TopHatKernel, scale; mask_strategy = ZeroFill()) -> BinnedTopHatPlan

The binned top-hat over the cells of a Cartesian grid with no lattice of its own, a node set or a
curvilinear mesh: the same sum as its neighbour-list engine, over the cells whose centres lie within
`kernel_radius` of the target's, each image of a cell across a periodic direction at its own distance.
"""
function binned_tophat_plan(
    grid::FlowGeometries.Grids.AbstractGrid{G,T}, kernel::Kernels.TopHatKernel, scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractCartesianGeometry{T}}
    D = FlowGeometries.Grids.ncoordinates(grid)
    return _binned_tophat_plan(grid, kernel, scale, mask_strategy, Val(D))
end

function _binned_tophat_plan(grid, kernel, scale::T, mask_strategy, ::Val{D}) where {T, D}
    bp = BinnedPoints(grid, Val(D))
    rad = T(Kernels.kernel_radius(kernel, scale, Val(D)))
    # An offset formed from coordinates of magnitude up to `xmax` is off by a few ulps of `xmax`, so a bin's
    # points lie in its box to that much; the run and the reach are widened and narrowed by it.
    xmax = maximum(d -> maximum(abs, bp.x[d]) + abs(bp.period[d]), 1:D)
    δ = 16 * eps(T) * (xmax + rad)
    lines, win, wout = _binned_lines(bp.h, rad - δ, rad + δ)
    n = length(bp.order)
    buf = _binned_buffer(bp)
    values, prefix, sums = buf
    deformable = mask_strategy isa Deformable && bp.mask !== nothing
    # The window mass: the same sum over the measures of the points the strategy counts.
    @inbounds for s in 1:n
        values[s] = (deformable && !bp.mask[s]) ? zero(T) : bp.measure[s]
    end
    _binned_scan!(prefix, values, bp)
    den = Vector{T}(undef, n)
    _binned_sweep!((den,), Returns(true), bp, (values,), (prefix,), (sums,), rad, lines, win, wout,
                   1:prod(Base.tail(bp.nb)))
    invden = similar(den)
    @inbounds for s in 1:n
        t = bp.order[s]
        invden[t] = (deformable && !bp.mask[s]) ? zero(T) : _inv_mass(den[t])
    end
    return BinnedTopHatPlan(bp, rad, lines, win, wout, [buf], invden, mask_strategy, bp.mask !== nothing)
end

# One field's `(values, scan, sums)` buffers.
_binned_buffer(bp::BinnedPoints{D,T}) where {D,T} =
    (Vector{T}(undef, length(bp.order)), CompensatedScan(T, bp.nb[1] + 1, Base.tail(bp.nb)...),
     Vector{T}(undef, length(bp.order)))

# The first `K` buffer triples, as many added as a batch of `K` fields needs. Run it before any driver
# starts: it grows the plan's vector.
function _binned_buffers!(fp::BinnedTopHatPlan, ::Val{K}) where {K}
    while length(fp.buffers) < K
        push!(fp.buffers, _binned_buffer(fp.points))
    end
    return ntuple(k -> @inbounds(fp.buffers[k]), Val(K))
end

# The compensated scan of the bin totals along axis 1, one line per bin index along axes `2…D`.
function _binned_scan!(P::CompensatedScan{T}, values::AbstractVector{T}, bp::BinnedPoints{D}) where {T, D}
    nb1 = bp.nb[1]
    ptr = bp.ptr
    @inbounds for (l, J) in enumerate(CartesianIndices(Base.tail(bp.nb)))
        h = zero(T)
        e = zero(T)
        P.hi[1, J] = h
        P.lo[1, J] = e
        b0 = (l - 1) * nb1
        for i in 1:nb1
            b = b0 + i
            tot = zero(T)
            for s in ptr[b]:(ptr[b + 1] - 1)
                tot += values[s]
            end
            h, e = _scan_step(h, e, tot)
            P.hi[i + 1, J] = h
            P.lo[i + 1, J] = e
        end
    end
    return P
end

# Bin `i` of a direction as its bin on the lattice and the shift of its image, `(-1, 0)` past a bounded
# edge. A bin one period out wraps by a comparison; farther ones by a division.
@inline function _binned_wrap(i::Int, nb::Int, periodic::Bool, period::T) where {T}
    0 <= i < nb && return i, zero(T)
    periodic || return -1, zero(T)
    -nb <= i < 0 && return i + nb, -period
    nb <= i < 2nb && return i - nb, period
    return mod(i, nb), T(fld(i, nb)) * period
end

# Sum of the bins `a…b` of the line at 1-based tail index `J` (raw indices along axis 1, any integers on a
# periodic axis, where each run of `nb₁` is one image of the line), one per field's scan.
@inline _binned_run(P::Tuple, a::Int, b::Int, nb::Int, periodic::Bool, J) =
    map(Pk -> _binned_run(Pk, a, b, nb, periodic, J), P)

@inline function _binned_run(P::CompensatedScan{T}, a::Int, b::Int, nb::Int, periodic::Bool, J) where {T}
    b < a && return zero(T)
    if !periodic
        a, b = max(a, 0), min(b, nb - 1)
        return b < a ? zero(T) : _scan_diff(P, b + 2, a + 1, J...)
    end
    q, r = divrem(b - a + 1, nb)
    acc = q > 0 ? q * _scan_diff(P, nb + 1, 1, J...) : zero(T)
    if r > 0
        a0 = mod(a, nb)
        e = a0 + r - 1
        acc += e < nb ? _scan_diff(P, e + 2, a0 + 1, J...) :
                        _scan_diff(P, nb + 1, a0 + 1, J...) + _scan_diff(P, e - nb + 2, 1, J...)
    end
    return acc
end

# The line `J₀ + o` of bins across axes `2…D` (0-based bin indices): its bin indices `J`, the linear offset
# `lin` of its first bin, and the shift `tshift` of the image it is; `nothing` past a bounded edge.
@inline function _binned_line(bp::BinnedPoints{D}, J0, o, strides) where {D}
    wrapped = map((j, oo, n, pd, len) -> _binned_wrap(j + oo, n, pd, len), J0, o, Base.tail(bp.nb),
                  Base.tail(bp.periodic), Base.tail(bp.period))
    any(w -> first(w) < 0, wrapped) && return nothing
    J = map(first, wrapped)
    return J, sum(ntuple(d -> J[d] * strides[d + 1], Val(D - 1)); init = 0), map(last, wrapped)
end

# Fold `acc = g(acc, s₀, s₁, shift)` over the bins `a…b` (raw indices along axis 1) of the line at linear
# offset `lin`, `tshift` its image across axes `2…D`: each image of the line's bins is one run `s₀:s₁` of
# points, since a line's bins are consecutive in bin order, and `shift` is that image's.
@inline function _binned_runs(g::G, acc, bp::BinnedPoints{D,T}, a::Int, b::Int, lin::Int,
                              tshift::NTuple{M,T}) where {G,D,T,M}
    nb1 = bp.nb[1]
    # A bounded axis holds one image of the line: the bins `a…b` on the lattice are one run.
    if !bp.periodic[1]
        a, b = max(a, 0), min(b, nb1 - 1)
        a <= b || return acc
        return @inbounds g(acc, bp.ptr[1 + lin + a], bp.ptr[2 + lin + b] - 1, (zero(T), tshift...))
    end
    i = a
    @inbounds while i <= b
        iw, sx = _binned_wrap(i, nb1, true, bp.period[1])
        e = min(b, i + nb1 - 1 - iw)
        acc = g(acc, bp.ptr[1 + lin + iw], bp.ptr[2 + lin + iw + e - i] - 1, (sx, tshift...))
        i = e + 1
    end
    return acc
end

# The values of the points within `rad` of `p` in bins `a…b` of the line at linear offset `lin`, one sum per
# field.
@inline _binned_partial(bp::BinnedPoints{D,T}, values::NTuple{K}, a::Int, b::Int, lin::Int, p::NTuple{D,T},
                        tshift::NTuple{M,T}, r2::T) where {D,T,K,M} =
    _binned_runs(ntuple(_ -> zero(T), Val(K)), bp, a, b, lin, tshift) do acc, s0, s1, shift
        map(+, acc, _binned_points(bp, values, s0, s1, p, shift, r2))
    end

# A point inside the ball adds its value and one outside adds zero, with no branch on which.
@inline function _binned_points(bp::BinnedPoints{D,T}, values::NTuple{K}, s0::Int, s1::Int, p::NTuple{D,T},
                                shift::NTuple{D,T}, r2::T) where {D,T,K}
    acc = ntuple(_ -> zero(T), Val(K))
    @inbounds for s in s0:s1
        d2 = zero(T)
        for d in 1:D
            δ = bp.x[d][s] + shift[d] - p[d]
            d2 += δ * δ
        end
        inside = d2 <= r2
        acc = map((a, v) -> a + ifelse(inside, v[s], zero(T)), acc, values)
    end
    return acc
end

# `outs[k][t] = scale(t) · window sum of field k` for the points of the rows of bins `rows`, linear indices
# over axes `2…D`; each writes its own entries of every output, indexed in the grid's order, and of `sums`,
# indexed in bin order. Each line of bins within reach of a row is found once for the row.
function _binned_sweep!(outs::NTuple{K}, scale::F, bp::BinnedPoints{D,T}, values::NTuple{K}, P::NTuple{K},
                        sums::NTuple{K}, rad::T, lines, win, wout, rows) where {K, F, D, T}
    r2 = rad * rad
    nb1 = bp.nb[1]
    strides = accumulate(*, (1, Base.front(bp.nb)...))
    tails = CartesianIndices(Base.tail(bp.nb))
    @inbounds for row in rows
        base = (row - 1) * nb1
        s0, s1 = bp.ptr[1 + base], bp.ptr[1 + base + nb1] - 1
        s0 > s1 && continue
        for acc in sums, s in s0:s1
            acc[s] = zero(T)
        end
        J0 = map(j -> j - 1, Tuple(tails[row]))
        for q in eachindex(lines)
            line = _binned_line(bp, J0, lines[q], strides)
            line === nothing && continue
            J, lin, tshift = line
            _binned_line_sums!(sums, bp, values, P, r2, win[q], wout[q], map(j -> j + 1, J), lin, tshift, base)
        end
        for s in s0:s1
            t = bp.order[s]
            c = scale(t)
            foreach((out, acc) -> (out[t] = iszero(c) ? zero(T) : c * acc[s]), outs, sums)
        end
    end
    return outs
end

# For every point of the row of bins at linear offset `base`, its window sum over the line of bins at `lin`
# (tail index `Jp`, 1-based, image shift `tshift`), added to `sums` in bin order: the run `i₀ ± wi` by its
# scan, once per bin, and the bins out to `i₀ ± wo` point by point. A bin whose reach stays on the lattice
# along axis 1 reads its runs of points directly.
@inline function _binned_line_sums!(sums::NTuple{K}, bp::BinnedPoints{D,T}, values::NTuple{K}, P::NTuple{K}, r2::T,
                                    wi::Int, wo::Int, Jp, lin::Int, tshift::NTuple{M,T}, base::Int) where {K, D, T, M}
    nb1, ptr = bp.nb[1], bp.ptr
    shift = (zero(T), tshift...)
    @inbounds for i0 in 0:(nb1 - 1)
        t0, t1 = ptr[1 + base + i0], ptr[2 + base + i0] - 1
        t0 > t1 && continue
        if i0 - wo >= 0 && i0 + wo < nb1
            run = wi >= 0 ? map(Pk -> _scan_diff(Pk, i0 + wi + 2, i0 - wi + 1, Jp...), P) : ntuple(_ -> zero(T), Val(K))
            # The partial bins before the run, `i₀ − wo … i₀ − wi − 1` (the whole reach where there is no run),
            # and after it, `i₀ + wi + 1 … i₀ + wo`.
            l0 = ptr[1 + lin + i0 - wo]
            l1 = wi >= 0 ? ptr[1 + lin + i0 - wi] - 1 : ptr[2 + lin + i0 + wo] - 1
            r0 = wi >= 0 ? ptr[2 + lin + i0 + wi] : 1
            r1 = wi >= 0 ? ptr[2 + lin + i0 + wo] - 1 : 0
            for s in t0:t1
                p = ntuple(d -> bp.x[d][s], Val(D))
                a = map(+, run, _binned_points(bp, values, l0, l1, p, shift, r2), _binned_points(bp, values, r0, r1, p, shift, r2))
                foreach((acc, ak) -> (acc[s] += ak), sums, a)
            end
        else
            for s in t0:t1
                p = ntuple(d -> bp.x[d][s], Val(D))
                a = if wi >= 0
                    map(+, _binned_run(P, i0 - wi, i0 + wi, nb1, bp.periodic[1], Jp),
                        _binned_partial(bp, values, i0 - wo, i0 - wi - 1, lin, p, tshift, r2),
                        _binned_partial(bp, values, i0 + wi + 1, i0 + wo, lin, p, tshift, r2))
                else
                    _binned_partial(bp, values, i0 - wo, i0 + wo, lin, p, tshift, r2)
                end
                foreach((acc, ak) -> (acc[s] += ak), sums, a)
            end
        end
    end
    return nothing
end

# Bins per block a driver hands out, as whole rows.
const _BINNED_BLOCK = 256

@noinline _binned_strategy_mismatch() = throw(ArgumentError(
    "BinnedTopHatPlan is being applied with a different mask strategy than it was built for, on a masked " *
    "grid, where the two normalize differently. Rebuild the plan with the `mask_strategy` you intend to " *
    "apply with.",
))

function _check_strategy(fp::BinnedTopHatPlan, strategy::AbstractMaskStrategy)
    (!fp.bound || typeof(strategy) === typeof(fp.strategy)) || _binned_strategy_mismatch()
    return nothing
end

"""
    apply_footprint!(out, field, grid, fp::BinnedTopHatPlan, strategy, driver = _sep_serial) -> out
    _driven_batch!(outs, fields, grid, fp::BinnedTopHatPlan, strategy, driver = _sep_serial) -> outs

The top-hat of `field` at every cell, or of each field of a batch in one sweep that finds every target's
lines and partial points once for all of them: the weighted values and their scans, then each target's
window sums over the reciprocal mass. `driver` runs blocks of bins under [`_sep_serial`](@ref)'s contract.
"""
apply_footprint!(
    out::AbstractArray, field::AbstractArray, grid, fp::BinnedTopHatPlan, strategy::AbstractMaskStrategy,
    driver::DR = _sep_serial,
) where {DR} = (_binned_apply!((out,), (field,), fp, strategy, driver); out)

_driven_batch!(
    outs::NTuple{K,AbstractArray}, fields::NTuple{K,AbstractArray}, grid, fp::BinnedTopHatPlan,
    strategy::AbstractMaskStrategy, driver::DR = _sep_serial,
) where {K, DR} = _binned_apply!(outs, fields, fp, strategy, driver)

function _binned_apply!(outs::NTuple{K}, fields::NTuple{K}, fp::BinnedTopHatPlan{D,T}, strategy, driver::DR) where {K, D, T, DR}
    _check_strategy(fp, strategy)
    bp = fp.points
    bufs = _binned_buffers!(fp, Val(K))
    values, P, sums = map(b -> b[1], bufs), map(b -> b[2], bufs), map(b -> b[3], bufs)
    foreach(values, fields, P) do v, field, Pk
        @inbounds for s in eachindex(v)
            active = bp.mask === nothing || bp.mask[s]
            v[s] = active ? bp.measure[s] * T(field[bp.order[s]]) : zero(T)
        end
        _binned_scan!(Pk, v, bp)
    end
    invden, rad, lines, win, wout = fp.invden, fp.rad, fp.lines, fp.win, fp.wout
    nrows = prod(Base.tail(bp.nb))
    per = max(1, _BINNED_BLOCK ÷ bp.nb[1])
    driver(1:cld(nrows, per)) do k
        rows = ((k - 1) * per + 1):min(k * per, nrows)
        _binned_sweep!(outs, t -> @inbounds(invden[t]), bp, values, P, sums, rad, lines, win, wout, rows)
    end
    return outs
end

_apply_serial!(out, field, grid, fp::BinnedTopHatPlan, strategy) = apply_footprint!(out, field, grid, fp, strategy)
_apply_serial_batch!(outs, fields, grid, fp::BinnedTopHatPlan, strategy) = _driven_batch!(outs, fields, grid, fp, strategy)

# ---------------------------------------------------------------------------
# A neighbour list read from the bins
# ---------------------------------------------------------------------------

# `row(t, p, i₀, J₀)` for every point of the bins `bins`: `t` its index in the grid's order, `p` its
# position, `(i₀, J₀)` its bin (0-based).
@inline function _binned_rows(row::R, bp::BinnedPoints{D}, bins) where {R, D}
    cart = CartesianIndices(bp.nb)
    @inbounds for b in bins
        I0 = Tuple(cart[b])
        J0 = map(j -> j - 1, Base.tail(I0))
        for s in bp.ptr[b]:(bp.ptr[b + 1] - 1)
            row(bp.order[s], ntuple(d -> bp.x[d][s], Val(D)), I0[1] - 1, J0)
        end
    end
    return nothing
end

# Fold `acc = f(acc, s, d²)` over the points `s` (in bin order) within `√r2` of `p`, a point of bin
# `(i₀, J₀)`, `d²` the squared distance from `p` to the image of `s` a run names: per line `lines[q]`, the
# bins `i₀ ± wout[q]`, each image of a point once.
@inline function _binned_within(f::F, acc, bp::BinnedPoints{D,T}, p::NTuple{D,T}, i0::Int, J0, r2::T,
                                lines, wout, strides) where {F, D, T}
    @inbounds for q in eachindex(lines)
        line = _binned_line(bp, J0, lines[q], strides)
        line === nothing && continue
        _, lin, tshift = line
        acc = _binned_runs(acc, bp, i0 - wout[q], i0 + wout[q], lin, tshift) do a, s0, s1, shift
            for s in s0:s1
                d2 = zero(T)
                for d in 1:D
                    δ = bp.x[d][s] + shift[d] - p[d]
                    d2 += δ * δ
                end
                d2 <= r2 && (a = f(a, s, d2))
            end
            a
        end
    end
    return acc
end

"""
    _binned_csr(grid, kernel, scale, backend) -> (nbrs, w, ptr)

Every cell's neighbours within `kernel_radius` of a Cartesian point set and their weights,
`kernel_weight · measure`, as CSR in the grid's order, read from its [`BinnedPoints`](@ref): a target's
candidates are the points of the bins `i₀ ± w_out` of each line of bins within reach, one run per image of
the line. Every image of a cell across a periodic direction within the radius is its own entry at its own
distance. One pass counts each row and the next fills it, blocks of bins running on `backend`.
"""
_binned_csr(grid::FlowGeometries.Grids.AbstractGrid{G,T}, kernel::Kernels.AbstractFilterKernel, scale::T,
            backend) where {G, T<:AbstractFloat} =
    _binned_csr(grid, kernel, scale, backend, Val(FlowGeometries.Grids.ncoordinates(grid)))

function _binned_csr(grid, kernel, scale::T, backend, dim::Val{D}) where {T, D}
    bp = BinnedPoints(grid, dim)
    rad = T(Kernels.kernel_radius(kernel, scale, dim))
    xmax = maximum(d -> maximum(abs, bp.x[d]) + abs(bp.period[d]), 1:D)
    lines, _, wout = _binned_lines(bp.h, rad, rad + 16 * eps(T) * (xmax + rad))
    r2 = rad * rad
    strides = accumulate(*, (1, Base.front(bp.nb)...))
    n = length(bp.order)
    nbins = prod(bp.nb)
    block(k) = ((k - 1) * _BINNED_BLOCK + 1):min(k * _BINNED_BLOCK, nbins)
    ptr = Vector{Int}(undef, n + 1)
    ptr[1] = 1
    FlowGeometries.Execution.run_chunks(cld(nbins, _BINNED_BLOCK), backend) do ks
        for k in ks
            _binned_rows(bp, block(k)) do t, p, i0, J0
                @inbounds ptr[t + 1] = _binned_within((m, _, _) -> m + 1, 0, bp, p, i0, J0, r2, lines, wout, strides)
            end
        end
    end
    @inbounds for t in 1:n
        ptr[t + 1] += ptr[t]
    end
    nbrs = Vector{Int}(undef, ptr[n + 1] - 1)
    w = Vector{T}(undef, ptr[n + 1] - 1)
    FlowGeometries.Execution.run_chunks(cld(nbins, _BINNED_BLOCK), backend) do ks
        for k in ks
            _binned_rows(bp, block(k)) do t, p, i0, J0
                _binned_within(@inbounds(ptr[t]), bp, p, i0, J0, r2, lines, wout, strides) do m, s, d2
                    @inbounds nbrs[m] = bp.order[s]
                    @inbounds w[m] = Kernels.kernel_weight(kernel, sqrt(d2), scale, dim) * bp.measure[s]
                    m + 1
                end
            end
        end
    end
    return nbrs, w, ptr
end
