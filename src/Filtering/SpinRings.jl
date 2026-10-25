# ---------------------------------------------------------------------------
# Spin-weighted transforms by rows of a latitude–longitude grid
# ---------------------------------------------------------------------------

"""
    RingSpinTransforms(grid; lmax = nothing, execution = SerialBackend())

The sums of [`DirectSpinTransforms`](@ref) on a two-dimensional latitude–longitude grid, whose rows share a
colatitude. The analysis is `C_ℓm = Σ_j N_ℓ d^ℓ_{m,−s}(θ_j) Σ_i w_ij F_ij e^{−imφ_i}`: the sums along every
row are taken at once by the grid's [`DenseLongitudes`](@ref), or with FFTW loaded on a whole ring of
uniform longitude by one FFT per row, and the Wigner recurrence runs once per row and order, one work item
per order, each degree a pass over the rows. The synthesis is its transpose with `Ĝ(ℓ)`. A transform
costs `O(nlat·lmax²)` for the recurrence and `O(M·lmax)` for the dense longitude sums, `O(M log nlon)` by
FFT, over `M` cells. `lmax = nothing` takes [`spin_band_limit`](@ref).
"""
struct RingSpinTransforms{T<:AbstractFloat, V<:AbstractVector{T}, Q<:AbstractMatrix{T}, W<:AbstractMatrix{T},
                          C<:AbstractMatrix{Complex{T}}, LN, B} <: AbstractSpinTransforms
    c::V          # cos(θ_j/2), per row
    s::V          # sin(θ_j/2)
    x::V          # cos θ_j
    w::W          # (nlon, nlat) measure over R²
    ring::C       # (2L+1, nlat): the sums along each row
    wf::C         # (nlon, nlat): w·F, and the longitude step's scratch
    lon::LN       # the longitude step
    norm::V       # N_ℓ, ℓ = 0 … lmax
    q::Q
    r::Q
    rc::Q         # (3·lmax, 2lmax+1): the recurrence's coefficients per degree and order, at one spin
    d::Q          # (nlat, 2lmax+1): per row and order the recurrence at two consecutive degrees, alternating
    dq::Q         # as the degree advances, from each row's starting value at one spin …
    ex::Q         # … with the exponent both carry
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

# The longitude step for `grid`'s longitudes `φ`: dense products; the FFTW extension takes a whole ring.
_ring_longitudes(grid, φ::AbstractVector{T}, L::Int, wf, execution) where {T} =
    DenseLongitudes(FlowGeometries.Execution.on_backend(execution, [cis(T(m) * φ[i]) for i in eachindex(φ), m in -L:L]))

function RingSpinTransforms(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2}; lmax::Union{Nothing,Integer} = nothing,
    execution::ComputationalBackends.AbstractLocalBackend = ComputationalBackends.SerialBackend(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}}
    L = lmax === nothing ? spin_band_limit(grid) : Int(lmax)
    L >= 2 || throw(ArgumentError("the generalized convolution's spin-2 transform needs lmax ≥ 2; got $L"))
    nlon, nlat = FlowGeometries.Grids.size_tuple(grid)
    θ, φ, w = _spin_nodes(grid)
    θr = [θ[1 + (j - 1) * nlon] for j in 1:nlat]
    q = zeros(T, L + 2, L + 1)
    for a in 0:L, j in a:(L + 1)
        q[j + 1, a + 1] = sqrt(T(j^2 - a^2))
    end
    r = zeros(T, L + 1, 5)
    for n in -2:2, j in abs(n):(L - 1)
        r[j + 1, n + 3] = sqrt(T((2j + 2) * (2j + 1)) / T((j + 1 + n) * (j + 1 - n)))
    end
    nrm = [sqrt(T(2ℓ + 1) / (4 * T(π))) for ℓ in 0:L]
    put(a) = FlowGeometries.Execution.on_backend(execution, a)
    wf = put(zeros(Complex{T}, nlon, nlat))
    rows() = put(zeros(T, nlat, 2L + 1))
    return RingSpinTransforms(
        put(cos.(θr ./ 2)), put(sin.(θr ./ 2)), put(cos.(θr)), put(reshape(w, nlon, nlat)),
        put(zeros(Complex{T}, 2L + 1, nlat)), wf, _ring_longitudes(grid, φ[1:nlon], L, wf, execution),
        put(nrm), put(q), put(r), put(zeros(T, 3L, 2L + 1)), rows(), rows(), rows(), rows(), rows(),
        put(zeros(T, L + 1)), L, execution,
    )
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

function spin_analyze!(C, F, tr::RingSpinTransforms{T}, s::Int) where {T}
    abs(s) <= 2 || _spin_out_of_range(s)
    L, n = tr.lmax, -s
    nlat = size(tr.w, 2)
    copyto!(tr.wf, F)
    tr.wf .*= tr.w
    _longitude_sums!(tr.ring, tr.wf, tr.lon)
    _rows_by_order!(tr.ar, tr.ai, tr.ring, tr.execution)
    ar, ai, x, nrm, q, rc, d, dq, ex = tr.ar, tr.ai, tr.x, tr.norm, tr.q, tr.rc, tr.d, tr.dq, tr.ex
    _ring_seeds!(tr, n)
    FlowGeometries.Execution.run_indices(2L + 1, _order_backend(tr.execution)) do col
        _ring_coefficients!(rc, q, L, n, col)
        _ring_analysis!(C, ar, ai, x, nrm, rc, d, dq, ex, L, n, col, nlat)
    end
    return C
end

# Order `m` costs `lmax − |m|` recurrence steps per row, so threads take the orders a few at a time as
# they finish.
_order_backend(b) = b
_order_backend(::ComputationalBackends.ThreadedBackend) = FlowGeometries.Execution.BalancedThreads()

function spin_synthesize!(F, C, _W, Ĝ, tr::RingSpinTransforms{T}, s::Int) where {T}
    abs(s) <= 2 || _spin_out_of_range(s)
    copyto!(tr.transfer, Ĝ)
    L, n = tr.lmax, -s
    nlat = size(tr.w, 2)
    ar, ai, g, x, nrm, q, rc, d, dq, ex = tr.ar, tr.ai, tr.transfer, tr.x, tr.norm, tr.q, tr.rc, tr.d, tr.dq, tr.ex
    _ring_seeds!(tr, n)
    FlowGeometries.Execution.run_indices(2L + 1, _order_backend(tr.execution)) do col
        _ring_coefficients!(rc, q, L, n, col)
        _ring_synthesis!(ar, ai, C, g, x, nrm, rc, d, dq, ex, L, n, col, nlat)
    end
    _orders_by_row!(tr.ring, ar, ai, tr.execution)
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

# Every row's starting values at spin `n`, into `tr.d` and `tr.ex`.
function _ring_seeds!(tr::RingSpinTransforms, n::Int)
    d, ex, c, sh, r, L = tr.d, tr.ex, tr.c, tr.s, tr.r, tr.lmax
    FlowGeometries.Execution.run_indices(size(d, 1), tr.execution) do j
        _ring_row_seeds!(d, ex, c, sh, r, L, n, j)
    end
    return nothing
end

# Row `j`'s `d^{ℓ₀}_{m,n}(θ_j)` at `ℓ₀ = max(|m|, |n|)` for every order, as [`_wigner_seed`](@ref) forms
# each: for `|m| ≥ |n|` the product at order `|m|` is the one at `|m| − 1` times `r cos(θ/2) sin(θ/2)`, so
# the row costs `O(lmax)`.
@inline function _ring_row_seeds!(dv, ex, cs_, ss_, r, L::Int, n::Int, j::Int)
    @inbounds begin
        T = eltype(dv)
        B = _exponent_step(T)
        lo = ldexp(one(T), -B)
        c, s = cs_[j], ss_[j]
        cs = c * s
        j0 = abs(n)
        for m in (1 - j0):(j0 - 1)
            d, e = _wigner_seed(m, n, c, s, r)
            dv[j, m + L + 1] = d
            ex[j, m + L + 1] = T(e)
        end
        for sgn in (1, -1)
            v = sgn > 0 ? c^(j0 + n) * s^(j0 - n) : c^(j0 - n) * s^(j0 + n)
            e = 0
            for a in j0:L
                if a > j0
                    v *= r[a, n + 3] * cs
                    if zero(T) < abs(v) < lo
                        v = ldexp(v, B)
                        e -= B
                    end
                end
                m = sgn * a
                (sgn < 0 && a == 0) && continue
                dv[j, m + L + 1] = (m >= 0 && isodd(m - n)) ? -v : v
                ex[j, m + L + 1] = T(e)
            end
        end
    end
    return nothing
end

# Column `col`, order `m = col − L − 1`, of `rc`: the coefficients of [`_wigner_next`](@ref) at spin `n`,
# `d^{ℓ+1} = (a_ℓ x − b_ℓ) d^ℓ − c_ℓ d^{ℓ−1}`, in rows `3ℓ+1 : 3ℓ+3`, for `ℓ` from `max(|m|, |n|)`.
@inline function _ring_coefficients!(rc, q, L::Int, n::Int, col::Int)
    T = eltype(rc)
    m = col - L - 1
    am, an = abs(m) + 1, abs(n) + 1
    @inbounds for ℓ in max(abs(m), abs(n)):(L - 1)
        k = 3ℓ
        if ℓ == 0
            rc[k + 1, col], rc[k + 2, col], rc[k + 3, col] = one(T), zero(T), zero(T)
        else
            D = q[ℓ + 2, am] * q[ℓ + 2, an]
            P = q[ℓ + 1, am] * q[ℓ + 1, an]
            rc[k + 1, col] = T(2ℓ + 1) * T(ℓ + 1) / D
            rc[k + 2, col] = T(2ℓ + 1) * T(m * n) / (T(ℓ) * D)
            rc[k + 3, col] = T(ℓ + 1) * P / (T(ℓ) * D)
        end
    end
    return nothing
end

# The value `d·2^e` of a pair the recurrence carries: `2^e` is exact at `e ∈ {0, −B, −2B}`, and a value carried
# at `e ≤ −3B` lies below the floating-point range, where `ldexp` returns zero.
@inline function _carried_value(d::T, e::T, B::Int) where {T}
    sc = ifelse(iszero(e), one(T),
                ifelse(e == -T(B), ldexp(one(T), -B), ifelse(e == -T(2B), ldexp(one(T), -2B), zero(T))))
    return d * sc
end

# Rows `lo:up` of column `col` form an interval whose values carry no exponent, the first such run; empty
# (`up < lo`) where every row carries one.
@inline function _clean_rows(ex, col::Int, nlat::Int)
    lo = 1
    @inbounds while lo <= nlat && !iszero(ex[lo, col])
        lo += 1
    end
    lo > nlat && return 1, 0
    up = lo
    @inbounds while up < nlat && iszero(ex[up + 1, col])
        up += 1
    end
    return lo, up
end

# The interval widened by the rows beside it whose exponents have reached zero, which they keep.
@inline function _grow_clean(ex, col::Int, lo::Int, up::Int, nlat::Int)
    up < lo && return _clean_rows(ex, col, nlat)
    @inbounds while lo > 1 && iszero(ex[lo - 1, col])
        lo -= 1
    end
    @inbounds while up < nlat && iszero(ex[up + 1, col])
        up += 1
    end
    return lo, up
end

# One degree on rows whose values carry no exponent, `|d| ≤ 1` keeping them there: `d^{ℓ+1}` into `prv`, over
# `d^{ℓ−1}`, with `nb = −b` and `nc = −c`; returns `Σ ar·d^{ℓ+1}` and `Σ ai·d^{ℓ+1}`.
@inline function _analysis_plain!(prv, cur, ar, ai, x, a::T, nb::T, nc::T, col::Int, rows::UnitRange{Int}) where {T}
    sr, si = zero(T), zero(T)
    @inbounds @simd for j in rows
        dn = muladd(muladd(a, x[j], nb), cur[j, col], nc * prv[j, col])
        prv[j, col] = dn
        sr = muladd(ar[j, col], dn, sr)
        si = muladd(ai[j, col], dn, si)
    end
    return sr, si
end

# One degree on rows whose values may carry an exponent: where `e < 0` and `|d^{ℓ+1}|` passes 1, both values
# are shifted down by `2^B` and `e` rises by `B`, which leaves the values they carry unchanged.
@inline function _analysis_careful!(prv, cur, ex, ar, ai, x, a::T, nb::T, nc::T, col::Int, rows::UnitRange{Int},
                                    B::Int) where {T}
    sr, si = zero(T), zero(T)
    down = ldexp(one(T), -B)
    @inbounds @simd for j in rows
        dj, e = cur[j, col], ex[j, col]
        dn = muladd(muladd(a, x[j], nb), dj, nc * prv[j, col])
        need = (e < zero(T)) & (abs(dn) > one(T))
        f = ifelse(need, down, one(T))
        e = ifelse(need, e + T(B), e)
        dn *= f
        cur[j, col] = dj * f
        prv[j, col] = dn
        ex[j, col] = e
        v = _carried_value(dn, e, B)
        sr = muladd(ar[j, col], v, sr)
        si = muladd(ai[j, col], v, si)
    end
    return sr, si
end

# Column `col`, order `m = col − L − 1`: `C[ℓ+1, col] = Σ_j N_ℓ d^ℓ_{m,n}(θ_j) ring[col, j]`, the row sums in
# `ar`, `ai`. `d` holds every row's value at `ℓ₀` and `ex` its exponent; `d` and `dq` then alternate.
@inline function _ring_analysis!(C, ar, ai, x, nrm, rc, d, dq, ex, L::Int, n::Int, col::Int, nlat::Int)
    T = eltype(x)
    m = col - L - 1
    l0 = max(abs(m), abs(n))
    B = _exponent_step(T)
    z = zero(eltype(C))
    @inbounds for row in 1:(L + 1)
        C[row, col] = z
    end
    sr, si = zero(T), zero(T)
    @inbounds @simd for j in 1:nlat
        v = _carried_value(d[j, col], ex[j, col], B)
        dq[j, col] = zero(T)
        sr = muladd(ar[j, col], v, sr)
        si = muladd(ai[j, col], v, si)
    end
    @inbounds C[l0 + 1, col] = nrm[l0 + 1] * complex(sr, si)
    lo, up = _clean_rows(ex, col, nlat)
    cur, prv = d, dq
    @inbounds for ℓ in l0:(L - 1)
        a, nb, nc = rc[3ℓ + 1, col], -rc[3ℓ + 2, col], -rc[3ℓ + 3, col]
        pr, pi_ = _analysis_plain!(prv, cur, ar, ai, x, a, nb, nc, col, lo:up)
        sr, si = _analysis_careful!(prv, cur, ex, ar, ai, x, a, nb, nc, col, 1:(lo - 1), B)
        tr_, ti = _analysis_careful!(prv, cur, ex, ar, ai, x, a, nb, nc, col, (up + 1):nlat, B)
        C[ℓ + 2, col] = nrm[ℓ + 2] * complex(pr + sr + tr_, pi_ + si + ti)
        lo, up = _grow_clean(ex, col, lo, up, nlat)
        cur, prv = prv, cur
    end
    return nothing
end

# The synthesis's degree on rows whose values carry no exponent: `d^{ℓ+1}` into `prv` and `h·d^{ℓ+1}` added to
# `ar`, `ai`, `h = hr + i·hi`.
@inline function _synthesis_plain!(ar, ai, prv, cur, x, a::T, nb::T, nc::T, hr::T, hi::T, col::Int,
                                   rows::UnitRange{Int}) where {T}
    @inbounds @simd for j in rows
        dn = muladd(muladd(a, x[j], nb), cur[j, col], nc * prv[j, col])
        prv[j, col] = dn
        ar[j, col] = muladd(hr, dn, ar[j, col])
        ai[j, col] = muladd(hi, dn, ai[j, col])
    end
    return nothing
end

# The synthesis's degree on rows whose values may carry an exponent, shifted as in [`_analysis_careful!`](@ref).
@inline function _synthesis_careful!(ar, ai, prv, cur, ex, x, a::T, nb::T, nc::T, hr::T, hi::T, col::Int,
                                     rows::UnitRange{Int}, B::Int) where {T}
    down = ldexp(one(T), -B)
    @inbounds @simd for j in rows
        dj, e = cur[j, col], ex[j, col]
        dn = muladd(muladd(a, x[j], nb), dj, nc * prv[j, col])
        need = (e < zero(T)) & (abs(dn) > one(T))
        f = ifelse(need, down, one(T))
        e = ifelse(need, e + T(B), e)
        dn *= f
        cur[j, col] = dj * f
        prv[j, col] = dn
        ex[j, col] = e
        v = _carried_value(dn, e, B)
        ar[j, col] = muladd(hr, v, ar[j, col])
        ai[j, col] = muladd(hi, v, ai[j, col])
    end
    return nothing
end

# Column `col`, order `m = col − L − 1`: `ar[j, col] + i·ai[j, col] = Σ_ℓ Ĝ(ℓ) C[ℓ+1, col] N_ℓ d^ℓ_{m,n}(θ_j)` for
# every row. `d` holds every row's value at `ℓ₀` and `ex` its exponent; `d` and `dq` then alternate.
@inline function _ring_synthesis!(ar, ai, C, g, x, nrm, rc, d, dq, ex, L::Int, n::Int, col::Int, nlat::Int)
    T = eltype(x)
    m = col - L - 1
    l0 = max(abs(m), abs(n))
    B = _exponent_step(T)
    @inbounds h = C[l0 + 1, col] * (g[l0 + 1] * nrm[l0 + 1])
    hr, hi = real(h), imag(h)
    @inbounds @simd for j in 1:nlat
        v = _carried_value(d[j, col], ex[j, col], B)
        dq[j, col] = zero(T)
        ar[j, col] = hr * v
        ai[j, col] = hi * v
    end
    lo, up = _clean_rows(ex, col, nlat)
    cur, prv = d, dq
    @inbounds for ℓ in l0:(L - 1)
        a, nb, nc = rc[3ℓ + 1, col], -rc[3ℓ + 2, col], -rc[3ℓ + 3, col]
        h = C[ℓ + 2, col] * (g[ℓ + 2] * nrm[ℓ + 2])
        hr, hi = real(h), imag(h)
        _synthesis_plain!(ar, ai, prv, cur, x, a, nb, nc, hr, hi, col, lo:up)
        _synthesis_careful!(ar, ai, prv, cur, ex, x, a, nb, nc, hr, hi, col, 1:(lo - 1), B)
        _synthesis_careful!(ar, ai, prv, cur, ex, x, a, nb, nc, hr, hi, col, (up + 1):nlat, B)
        lo, up = _grow_clean(ex, col, lo, up, nlat)
        cur, prv = prv, cur
    end
    return nothing
end

# A latitude–longitude grid's direct transforms run by rows.
_direct_spin_transforms(grid::FlowGeometries.Grids.StructuredGrid{T,G,2}; lmax = nothing,
                        execution = ComputationalBackends.SerialBackend(), nthreads = 1,
                        nufft = nothing) where {T, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}} =
    RingSpinTransforms(grid; lmax, execution)
