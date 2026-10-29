# ---------------------------------------------------------------------------
# Spin-weighted transforms by rows of a latitude–longitude grid
# ---------------------------------------------------------------------------

"""
    RingSpinTransforms(grid; lmax = nothing, execution = SerialBackend())

The sums of [`DirectSpinTransforms`](@ref) on a two-dimensional latitude–longitude grid, whose rows share a
colatitude. The analysis is `C_ℓm = Σ_j N_ℓ d^ℓ_{m,−s}(θ_j) Σ_i w_ij F_ij e^{−imφ_i}`: the sums along every
row are taken at once by the grid's [`DenseLongitudes`](@ref), or with an FFT library loaded on a whole
ring of uniform longitude by one FFT per row, and the Wigner recurrence runs once per row and order, one
work item per order, two degrees a pass over the rows. The synthesis is its transpose with `Ĝ(ℓ)`. A transform
costs `O(nlat·lmax²)` for the recurrence and `O(M·lmax)` for the dense longitude sums, `O(M log nlon)` by
FFT, over `M` cells. `lmax = nothing` takes [`spin_band_limit`](@ref).
"""
struct RingSpinTransforms{
    T<:AbstractFloat, S<:AbstractFloat, V<:AbstractVector{T}, VB<:AbstractVector{Bool}, VS<:AbstractVector{S},
    ST<:AbstractMatrix{FlowTransformBindings.WignerDStart{S}}, RS<:AbstractMatrix{S}, Q<:AbstractMatrix{T},
    K<:AbstractMatrix{Int32}, W<:AbstractMatrix{T}, C<:AbstractMatrix{Complex{T}}, LN, B,
} <: AbstractSpinTransforms
    x::V          # cos θ_j per row …
    xl::V         # … and its rest, to twice the precision
    u::V          # Reinsch's 1 ∓ cos θ_j
    pos::VB       # θ_j ≤ π/2
    lc::VS        # log₂ cos(θ_j/2)
    ls::VS        # log₂ sin(θ_j/2)
    cs::VS        # cos(θ_j/2) sin(θ_j/2)
    w::W          # (nlon, nlat) measure over R²
    ring::C       # (2L+1, nlat): the sums along each row
    wf::C         # (nlon, nlat): w·F, and the longitude step's scratch
    lon::LN       # the longitude step
    norm::V       # N_ℓ, ℓ = 0 … lmax
    starts::ST    # starts[m + lmax + 1, n + 3]
    ratio::RS     # ratio[m + lmax + 1, n + 3], |m| > |n|: the step of the starting value across orders
    coef::Q       # (_wigner_ncoef(T)·lmax, 2lmax+1): each order's step coefficients at the spin transformed
    y::Q          # (nlat, 2lmax+1): per row and order the recurrence's value …
    e::Q          # … its second state value …
    k::K          # … and its scale
    ar::Q         # (nlat, 2lmax+1): per row and order the row sum (analysis) or the synthesis, real part …
    ai::Q         # … and imaginary part
    transfer::V
    lmax::Int
    execution::B
end

"""
    DenseLongitudes(E)

The longitude step of [`RingSpinTransforms`](@ref) as products with `E[i, m+L+1] = e^{imφ_i}`: the row
sums `ring = Eᴴ·(w F)` and the synthesis `F = E·ring`, for any longitudes the rows share.
"""
struct DenseLongitudes{E<:AbstractMatrix}
    E::E
end

_longitude_sums!(ring, wf, lon::DenseLongitudes) = LinearAlgebra.mul!(ring, adjoint(lon.E), wf)
_longitude_synthesis!(F, ring, wf, lon::DenseLongitudes) = copyto!(F, LinearAlgebra.mul!(wf, lon.E, ring))

# The longitude step for the longitude axis `λ` sampled at `φ`: dense products; the AbstractFFTs extension
# takes a `PeriodicUniformAxis`, whose `n` equal cells close the circle, where an FFT library serves `execution`.
_ring_longitudes(λ::AbstractVector, φ::AbstractVector, L::Int, wf, execution) = _dense_longitudes(φ, L, execution)
_dense_longitudes(φ::AbstractVector{T}, L::Int, execution) where {T} =
    DenseLongitudes(FlowGeometries.Execution.on_backend(execution, [cis(T(m) * φ[i]) for i in eachindex(φ), m in -L:L]))

function RingSpinTransforms(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2}; lmax::Union{Nothing,Integer} = nothing,
    execution::ComputationalBackends.AbstractLocalBackend = ComputationalBackends.SerialBackend(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}}
    L = lmax === nothing ? spin_band_limit(grid) : Int(lmax)
    L >= 2 || throw(ArgumentError("the generalized convolution's spin-2 transform needs lmax ≥ 2; got $L"))
    S = promote_type(T, Float64)
    nlon, nlat = FlowGeometries.Grids.size_tuple(grid)
    θ, φ, w = _spin_nodes(grid)
    nd = _wigner_nodes(T, S, [θ[1 + (j - 1) * nlon] for j in 1:nlat])
    nrm = [sqrt(T(2ℓ + 1) / (4 * T(π))) for ℓ in 0:L]
    starts = _wigner_starts(S, L)
    put(a) = FlowGeometries.Execution.on_backend(execution, a)
    wf = put(zeros(Complex{T}, nlon, nlat))
    rows() = put(zeros(T, nlat, 2L + 1))
    return RingSpinTransforms(
        put(nd.x), put(nd.xl), put(nd.u), put(nd.pos), put(nd.lc), put(nd.ls), put(nd.cs),
        put(reshape(w, nlon, nlat)), put(zeros(Complex{T}, 2L + 1, nlat)), wf,
        _ring_longitudes(FlowGeometries.Grids.coordinates(grid, 1), φ[1:nlon], L, wf, execution),
        put(nrm), put(starts), put(_wigner_ratios(starts, L)), put(zeros(T, _wigner_ncoef(T) * L, 2L + 1)),
        rows(), rows(), put(zeros(Int32, nlat, 2L + 1)), rows(), rows(), put(zeros(T, L + 1)), L, execution,
    )
end

# `ratio[m + L + 1, n + 3] = d^{|m|}_{m,n} / (d^{|m|−1}_{m∓1,n} cos(θ/2) sin(θ/2))` for `|m| > |n|`: an order
# one further from zero raises both powers of the closed form by one, so the ratio is that of the start
# constants' signs and `2^{lC}`.
function _wigner_ratios(starts::AbstractMatrix{FlowTransformBindings.WignerDStart{S}}, L::Int) where {S}
    r = zeros(S, 2L + 1, 5)
    for n in -2:2, m in -L:L
        abs(m) <= abs(n) && continue
        st, sp = starts[m + L + 1, n + 3], starts[m - sign(m) + L + 1, n + 3]
        r[m + L + 1, n + 3] = st.sg / sp.sg * exp2(st.lC - sp.lC)
    end
    return r
end

Base.show(io::IO, tr::RingSpinTransforms) =
    print(io, "RingSpinTransforms(", join(size(tr.w), "×"), " cells, lmax = ", tr.lmax, ")")
Base.show(io::IO, ::MIME"text/plain", tr::RingSpinTransforms) = show(io, tr)

spin_field(tr::RingSpinTransforms{T}) where {T} = _allocate(tr.execution, Complex{T}, (length(tr.w),))
spin_field_size(tr::RingSpinTransforms) = (length(tr.w),)
spin_position(::RingSpinTransforms) = CellOrder()
spin_coefficients(tr::RingSpinTransforms{T}) where {T} =
    _allocate(tr.execution, Complex{T}, (tr.lmax + 1, 2tr.lmax + 1))
spin_transfer_buffer(tr::RingSpinTransforms{T}) where {T} = zeros(T, tr.lmax + 1)
spin_host(tr::RingSpinTransforms) = _host_memory(tr.execution)

function spin_analyze!(C, F, tr::RingSpinTransforms{T,S}, s::Int) where {T,S}
    abs(s) <= 2 || _spin_out_of_range(s)
    L, n = tr.lmax, -s
    nlat = size(tr.w, 2)
    copyto!(tr.wf, F)
    tr.wf .*= tr.w
    _longitude_sums!(tr.ring, tr.wf, tr.lon)
    _rows_by_order!(tr.ar, tr.ai, tr.ring, tr.execution)
    _ring_seeds!(tr, n)
    t = _ring_tables(tr)
    FlowGeometries.Execution.run_indices(2L + 1, _order_backend(tr.execution)) do col
        _ring_analysis!(C, t, L, n, col, nlat)
    end
    return C
end

# What the order loops read, which their device kernels capture in place of the transforms: the tables and
# buffers, and a zero of the type `S` the step coefficients are formed in.
_ring_tables(tr::RingSpinTransforms{T,S}) where {T,S} =
    (; tr.x, tr.xl, tr.u, tr.pos, tr.coef, tr.y, tr.e, tr.k, tr.ar, tr.ai, tr.norm, tr.transfer, s = zero(S))

# Order `m` costs `lmax − |m|` recurrence steps per row, so threads take the orders a few at a time as
# they finish.
_order_backend(b) = b
_order_backend(::ComputationalBackends.ThreadedBackend) = FlowGeometries.Execution.BalancedThreads()

# In the harmonics `N_ℓ d^ℓ_{m,−s}(θ) e^{imφ}` of the direct and ring transforms, a step of `ð` or `ð̄` is the
# standard ladder factor's negative.
function spin_ladder_synthesize!(F, C, W, d, tr::Union{DirectSpinTransforms,RingSpinTransforms}, s::Int, t::Int)
    spin_synthesize!(F, C, W, d, tr, t)
    F .= .-F
    return F
end

function spin_synthesize!(F, C, _W, Ĝ, tr::RingSpinTransforms{T,S}, s::Int) where {T,S}
    abs(s) <= 2 || _spin_out_of_range(s)
    copyto!(tr.transfer, Ĝ)
    L, n = tr.lmax, -s
    nlat = size(tr.w, 2)
    _ring_seeds!(tr, n)
    t = _ring_tables(tr)
    FlowGeometries.Execution.run_indices(2L + 1, _order_backend(tr.execution)) do col
        _ring_synthesis!(C, t, L, n, col, nlat)
    end
    _orders_by_row!(tr.ring, tr.ar, tr.ai, tr.execution)
    _longitude_synthesis!(F, tr.ring, tr.wf, tr.lon)
    return F
end

# `ring[col, j]` into `ar[j, col]` and `ai[j, col]`, and back.
function _rows_by_order!(ar, ai, ring, execution)
    FlowGeometries.Execution.run_indices(size(ring, 2), execution) do j
        @inbounds for col in axes(ring, 1)
            z = ring[col, j]
            ar[j, col] = real(z)
            ai[j, col] = imag(z)
        end
        return nothing
    end
    return nothing
end

function _orders_by_row!(ring, ar, ai, execution)
    FlowGeometries.Execution.run_indices(size(ring, 2), execution) do j
        @inbounds for col in axes(ring, 1)
            ring[col, j] = complex(ar[j, col], ai[j, col])
        end
        return nothing
    end
    return nothing
end

# Every row's `d^{ℓ₀}_{m,n}(θ_j)` at spin `n` for every order, with its scale and second state value.
function _ring_seeds!(tr::RingSpinTransforms, n::Int)
    y, e, k, starts, ratio, cs, lc, ls, L = tr.y, tr.e, tr.k, tr.starts, tr.ratio, tr.cs, tr.lc, tr.ls, tr.lmax
    FlowGeometries.Execution.run_indices(size(y, 1), tr.execution) do j
        _ring_row_seeds!(y, e, k, starts, ratio, cs, lc, ls, L, n, j)
    end
    return nothing
end

# Row `j`: the closed form for `|m| ≤ |n|`; each order further from zero the one before times
# `ratio · cos(θ_j/2) sin(θ_j/2)`, carried in `S` and moved down a scale where it falls to `2^{−K/2}`.
@inline function _ring_row_seeds!(y, e, k, starts, ratio, cs, lc, ls, L::Int, n::Int, j::Int)
    T, S = eltype(y), eltype(cs)
    K = FlowTransformBindings.wigner_d_span(T)
    lo, upK = ldexp(one(S), -(K ÷ 2)), ldexp(one(S), K)
    a0 = abs(n)
    @inbounds begin
        for m in (-a0):a0
            col = m + L + 1
            yj, kj = FlowTransformBindings.wigner_d_start_value(T, starts[col, n + 3], lc[j], ls[j])
            y[j, col] = yj
            e[j, col] = _wigner_e0(yj)
            k[j, col] = kj
        end
        c = cs[j]
        for sgn in (1, -1)
            v, kk = S(y[j, sgn * a0 + L + 1]), k[j, sgn * a0 + L + 1]
            for a in (a0 + 1):L
                col = sgn * a + L + 1
                v *= ratio[col, n + 3] * c
                small = (abs(v) <= lo) & !iszero(v)
                v = ifelse(small, v * upK, v)
                kk = ifelse(small, kk - Int32(1), kk)
                yv = T(v)
                y[j, col] = yv
                e[j, col] = _wigner_e0(yv)
                k[j, col] = kk
            end
        end
    end
    return nothing
end

# Rows `lo:up` of column `col` form an interval whose values are held at scale zero, the first such run;
# empty (`up < lo`) where no row is.
@inline function _clean_rows(k, col::Int, nlat::Int)
    lo = 1
    @inbounds while lo <= nlat && !iszero(k[lo, col])
        lo += 1
    end
    lo > nlat && return 1, 0
    up = lo
    @inbounds while up < nlat && iszero(k[up + 1, col])
        up += 1
    end
    return lo, up
end

# The interval widened by the rows beside it whose scales have reached zero, which they keep.
@inline function _grow_clean(k, col::Int, lo::Int, up::Int, nlat::Int)
    up < lo && return _clean_rows(k, col, nlat)
    @inbounds while lo > 1 && iszero(k[lo - 1, col])
        lo -= 1
    end
    @inbounds while up < nlat && iszero(k[up + 1, col])
        up += 1
    end
    return lo, up
end

# The per-row tables a step reads: `cos θ` as `x + xl` for the plain form, `u` and `pos` for Reinsch's.
_ring_nodes(t::NamedTuple) = (t.x, t.xl, t.u, t.pos)

# The state after one single step: the plain form's `(d^{ℓ+1}, d^ℓ)` sit in `(q, p)`, Reinsch's in place.
@inline _next_state(::Type{T}, p, q) where {T} = _wigner_reinsch(T) ? (p, q) : (q, p)

# Two degrees up at row `j`, held at scale zero, with the steps `cf1` from `ℓ` and `cf2` from `ℓ + 1`: the state
# `(p, q)` of degree `ℓ` becomes that of `ℓ + 2` in place (the plain form's `p = d^{ℓ+2}`, `q = d^{ℓ+1}`).
# Returns `(d^{ℓ+1}, d^{ℓ+2})`.
@inline function _ring_step2!(p, q, nodes, cf1, cf2, j::Int, col::Int)
    T = eltype(p)
    x, xl, u, pos = nodes
    @inbounds if _wigner_reinsch(T)
        σα1, c1, σγ1, σ = FlowTransformBindings.wigner_d_signed(pos[j], cf1[1], cf1[4], cf1[5], cf1[3])
        σα2, c2, σγ2, _ = FlowTransformBindings.wigner_d_signed(pos[j], cf2[1], cf2[4], cf2[5], cf2[3])
        y1, e1 = FlowTransformBindings.wigner_d_step(σα1, c1, σγ1, σ, u[j], p[j, col], q[j, col])
        y2, e2 = FlowTransformBindings.wigner_d_step(σα2, c2, σγ2, σ, u[j], y1, e1)
        p[j, col] = y2
        q[j, col] = e2
    else
        xj, xlj = x[j], xl[j]
        y1, y0 = FlowTransformBindings.wigner_d_step_plain(cf1[1], cf1[2], cf1[3], xj, xlj, p[j, col], q[j, col])
        y2, _ = FlowTransformBindings.wigner_d_step_plain(cf2[1], cf2[2], cf2[3], xj, xlj, y1, y0)
        p[j, col] = y2
        q[j, col] = y1
    end
    return y1, y2
end

# One degree up at row `j`, column `col`, from the state `(p, q)`: `p = d^ℓ` and `q = d^{ℓ−1}` in the plain
# form, which writes `d^{ℓ+1}` over `q` (the caller then swaps the two); `p = y`, `q = e` in Reinsch's,
# which writes both. Returns `d^{ℓ+1}` at its scale. `rescale` moves a value up a scale as it grows.
@inline function _ring_step!(p, q, k, nodes, cf, j::Int, col::Int, rescale::Bool, up, dn)
    T = eltype(p)
    x, xl, u, pos = nodes
    α, β, γ, c₊, c₋ = cf
    @inbounds if _wigner_reinsch(T)
        σα, c, σγ, σ = FlowTransformBindings.wigner_d_signed(pos[j], α, c₊, c₋, γ)
        yn, en = FlowTransformBindings.wigner_d_step(σα, c, σγ, σ, u[j], p[j, col], q[j, col])
    else
        yn, en = FlowTransformBindings.wigner_d_step_plain(α, β, γ, x[j], xl[j], p[j, col], q[j, col])
    end
    kn = Int32(0)
    @inbounds if rescale
        yn, en, kn = FlowTransformBindings.wigner_d_rescale(yn, en, k[j, col], up, dn)
        k[j, col] = kn
    end
    @inbounds if _wigner_reinsch(T)
        p[j, col] = yn
        q[j, col] = en
    else
        q[j, col] = yn
        rescale && (p[j, col] = en)
    end
    return yn, kn
end

# One degree on `rows`: returns `Σ ar·d^{ℓ+1}` and `Σ ai·d^{ℓ+1}`. On rows held at scale zero, which keep
# it, `rescale = false`; on rows below it the values are rescaled as they grow, and one at scale `−1` enters
# weighted by `2^{−K}`, one below it not at all.
@inline function _analysis_rows!(p, q, k, nodes, ar, ai, cf, col::Int, rows::UnitRange{Int}, rescale::Bool)
    T = eltype(p)
    up, dn = FlowTransformBindings.wigner_d_bounds(T)
    sr, si = zero(T), zero(T)
    @inbounds @simd for j in rows
        yn, kn = _ring_step!(p, q, k, nodes, cf, j, col, rescale, up, dn)
        v = rescale ? yn * _wigner_weight(kn, dn) : yn
        sr = muladd(ar[j, col], v, sr)
        si = muladd(ai[j, col], v, si)
    end
    return sr, si
end

# Two degrees on rows `rows` held at scale zero: returns the sums `Σ ar·d`, `Σ ai·d` at `ℓ + 1` and `ℓ + 2`.
@inline function _analysis_rows2!(p, q, nodes, ar, ai, cf1, cf2, col::Int, rows::UnitRange{Int})
    T = eltype(p)
    r1, i1, r2, i2 = zero(T), zero(T), zero(T), zero(T)
    @inbounds @simd for j in rows
        y1, y2 = _ring_step2!(p, q, nodes, cf1, cf2, j, col)
        a, b = ar[j, col], ai[j, col]
        r1 = muladd(a, y1, r1)
        i1 = muladd(b, y1, i1)
        r2 = muladd(a, y2, r2)
        i2 = muladd(b, y2, i2)
    end
    return r1, i1, r2, i2
end

# The rows outside `lo:up` one degree on from `(p, q)`, rescaled as they grow: the sums of
# [`_analysis_rows!`](@ref) over both ranges.
@inline function _analysis_outside!(p, q, k, nodes, ar, ai, cf, col::Int, lo::Int, up::Int, nlat::Int)
    sr, si = _analysis_rows!(p, q, k, nodes, ar, ai, cf, col, 1:(lo - 1), true)
    tr_, ti = _analysis_rows!(p, q, k, nodes, ar, ai, cf, col, (up + 1):nlat, true)
    return sr + tr_, si + ti
end

# Column `col`, order `m = col − L − 1`: `C[ℓ+1, col] = Σ_j N_ℓ d^ℓ_{m,n}(θ_j) ring[col, j]`, the row sums in
# `ar`, `ai`, the recurrence starting from every row's seed, two degrees per pass over the rows. The order's
# step coefficients are formed first, into its column of `tr.coef`.
@inline function _ring_analysis!(C, t::NamedTuple, L::Int, n::Int, col::Int, nlat::Int)
    T = eltype(t.y)
    m = col - L - 1
    l0 = max(abs(m), abs(n))
    _, dn = FlowTransformBindings.wigner_d_bounds(T)
    k, ar, ai, nrm, nodes, coef = t.k, t.ar, t.ai, t.norm, _ring_nodes(t), t.coef
    _wigner_coefficients!(coef, typeof(t.s), L, n, col)
    p, q = t.y, t.e
    z = zero(eltype(C))
    @inbounds for row in 1:(L + 1)
        C[row, col] = z
    end
    sr, si = zero(T), zero(T)
    @inbounds @simd for j in 1:nlat
        v = p[j, col] * _wigner_weight(k[j, col], dn)
        sr = muladd(ar[j, col], v, sr)
        si = muladd(ai[j, col], v, si)
    end
    @inbounds C[l0 + 1, col] = nrm[l0 + 1] * complex(sr, si)
    lo, up = _clean_rows(k, col, nlat)
    ℓ = l0
    @inbounds while ℓ + 1 <= L - 1
        cf1 = _table_coeffs(coef, ℓ, col)
        cf2 = _table_coeffs(coef, ℓ + 1, col)
        r1, i1, r2, i2 = _analysis_rows2!(p, q, nodes, ar, ai, cf1, cf2, col, lo:up)
        o1r, o1i = _analysis_outside!(p, q, k, nodes, ar, ai, cf1, col, lo, up, nlat)
        p2, q2 = _next_state(T, p, q)
        o2r, o2i = _analysis_outside!(p2, q2, k, nodes, ar, ai, cf2, col, lo, up, nlat)
        C[ℓ + 2, col] = nrm[ℓ + 2] * complex(r1 + o1r, i1 + o1i)
        C[ℓ + 3, col] = nrm[ℓ + 3] * complex(r2 + o2r, i2 + o2i)
        lo, up = _grow_clean(k, col, lo, up, nlat)
        ℓ += 2
    end
    @inbounds if ℓ <= L - 1
        cf = _table_coeffs(coef, ℓ, col)
        pr, pi_ = _analysis_rows!(p, q, k, nodes, ar, ai, cf, col, lo:up, false)
        or, oi = _analysis_outside!(p, q, k, nodes, ar, ai, cf, col, lo, up, nlat)
        C[ℓ + 2, col] = nrm[ℓ + 2] * complex(pr + or, pi_ + oi)
    end
    return nothing
end

# The synthesis's degree on `rows`: `h·d^{ℓ+1}` added to `ar`, `ai`, `h = hr + i·hi`, the values rescaled as
# in [`_analysis_rows!`](@ref) where `rescale`.
@inline function _synthesis_rows!(p, q, k, nodes, ar, ai, cf, hr::T, hi::T, col::Int, rows::UnitRange{Int},
                                  rescale::Bool) where {T}
    up, dn = FlowTransformBindings.wigner_d_bounds(T)
    @inbounds @simd for j in rows
        yn, kn = _ring_step!(p, q, k, nodes, cf, j, col, rescale, up, dn)
        v = rescale ? yn * _wigner_weight(kn, dn) : yn
        ar[j, col] = muladd(hr, v, ar[j, col])
        ai[j, col] = muladd(hi, v, ai[j, col])
    end
    return nothing
end

# Two degrees on rows `rows` held at scale zero: `h₁·d^{ℓ+1} + h₂·d^{ℓ+2}` added to `ar`, `ai`, in that order.
@inline function _synthesis_rows2!(p, q, nodes, ar, ai, cf1, cf2, h1r::T, h1i::T, h2r::T, h2i::T, col::Int,
                                   rows::UnitRange{Int}) where {T}
    @inbounds @simd for j in rows
        y1, y2 = _ring_step2!(p, q, nodes, cf1, cf2, j, col)
        ar[j, col] = muladd(h2r, y2, muladd(h1r, y1, ar[j, col]))
        ai[j, col] = muladd(h2i, y2, muladd(h1i, y1, ai[j, col]))
    end
    return nothing
end

# The rows outside `lo:up` one degree on from `(p, q)`, as [`_synthesis_rows!`](@ref) with rescaling.
@inline function _synthesis_outside!(p, q, k, nodes, ar, ai, cf, hr, hi, col::Int, lo::Int, up::Int, nlat::Int)
    _synthesis_rows!(p, q, k, nodes, ar, ai, cf, hr, hi, col, 1:(lo - 1), true)
    _synthesis_rows!(p, q, k, nodes, ar, ai, cf, hr, hi, col, (up + 1):nlat, true)
    return nothing
end

# Column `col`, order `m = col − L − 1`: `ar[j, col] + i·ai[j, col] = Σ_ℓ Ĝ(ℓ) C[ℓ+1, col] N_ℓ d^ℓ_{m,n}(θ_j)` for
# every row, the recurrence starting from every row's seed, two degrees per pass over the rows, from the
# order's step coefficients formed first into its column of `tr.coef`.
@inline function _ring_synthesis!(C, t::NamedTuple, L::Int, n::Int, col::Int, nlat::Int)
    T = eltype(t.y)
    m = col - L - 1
    l0 = max(abs(m), abs(n))
    _, dn = FlowTransformBindings.wigner_d_bounds(T)
    k, ar, ai, g, nrm, nodes, coef = t.k, t.ar, t.ai, t.transfer, t.norm, _ring_nodes(t), t.coef
    _wigner_coefficients!(coef, typeof(t.s), L, n, col)
    p, q = t.y, t.e
    @inbounds h = C[l0 + 1, col] * (g[l0 + 1] * nrm[l0 + 1])
    hr, hi = real(h), imag(h)
    @inbounds @simd for j in 1:nlat
        v = p[j, col] * _wigner_weight(k[j, col], dn)
        ar[j, col] = hr * v
        ai[j, col] = hi * v
    end
    lo, up = _clean_rows(k, col, nlat)
    ℓ = l0
    @inbounds while ℓ + 1 <= L - 1
        cf1 = _table_coeffs(coef, ℓ, col)
        cf2 = _table_coeffs(coef, ℓ + 1, col)
        h1 = C[ℓ + 2, col] * (g[ℓ + 2] * nrm[ℓ + 2])
        h2 = C[ℓ + 3, col] * (g[ℓ + 3] * nrm[ℓ + 3])
        _synthesis_rows2!(p, q, nodes, ar, ai, cf1, cf2, real(h1), imag(h1), real(h2), imag(h2), col, lo:up)
        _synthesis_outside!(p, q, k, nodes, ar, ai, cf1, real(h1), imag(h1), col, lo, up, nlat)
        p2, q2 = _next_state(T, p, q)
        _synthesis_outside!(p2, q2, k, nodes, ar, ai, cf2, real(h2), imag(h2), col, lo, up, nlat)
        lo, up = _grow_clean(k, col, lo, up, nlat)
        ℓ += 2
    end
    @inbounds if ℓ <= L - 1
        cf = _table_coeffs(coef, ℓ, col)
        h = C[ℓ + 2, col] * (g[ℓ + 2] * nrm[ℓ + 2])
        _synthesis_rows!(p, q, k, nodes, ar, ai, cf, real(h), imag(h), col, lo:up, false)
        _synthesis_outside!(p, q, k, nodes, ar, ai, cf, real(h), imag(h), col, lo, up, nlat)
    end
    return nothing
end

# A latitude–longitude grid's direct transforms run by rows.
_direct_spin_transforms(grid::FlowGeometries.Grids.StructuredGrid{T,G,2}; lmax = nothing,
                        execution = ComputationalBackends.SerialBackend(), nthreads = 1,
                        nufft = nothing) where {T, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}} =
    RingSpinTransforms(grid; lmax, execution)
