# ---------------------------------------------------------------------------
# Spin-weighted transforms by direct summation over a grid's cells
# ---------------------------------------------------------------------------

"""
    DirectSpinTransforms(grid; lmax = nothing, execution = SerialBackend())

The spin-weighted transforms of [`GeneralizedConvolution`](@ref) by direct summation over the cells of a
spherical `grid`, for any spin `|s| ≤ 2`, in the grid's element type and in the memory of the local
backend `execution`, whose loops run them. With `ₛY_ℓm(θ, φ) = N_ℓ d^ℓ_{m,−s}(θ) e^{imφ}`,
`N_ℓ = √((2ℓ+1)/4π)` and `θ` the colatitude, the analysis is the quadrature
`C_ℓm = Σ_k w_k F_k conj(ₛY_ℓm(x_k))` with `w_k` the cell's measure over `R²`, one work item per order `m`,
and the synthesis is `F_k = Σ_ℓm Ĝ(ℓ) C_ℓm ₛY_ℓm(x_k)`, one work item per cell. The coefficients are
`C[ℓ+1, m+lmax+1]`.

The Wigner `d^ℓ_{m,−s}` run up the three-term recurrence in `ℓ` from `ℓ₀ = max(|m|, |s|)`, where `d` is a
product of powers of `cos(θ/2)` and `sin(θ/2)` built order by order, of size about `sin^|m| θ`. A value
below `2^−B`, `B` half the element type's largest binary exponent, is carried as a significand and its
own exponent, so an order whose seed lies below the floating-point range keeps the values it reaches at
higher degree. A transform costs `O(M·lmax²)` over `M` cells and allocates nothing. `lmax = nothing`
takes [`spin_band_limit`](@ref).
"""
struct DirectSpinTransforms{T<:AbstractFloat, V<:AbstractVector{T}, Q<:AbstractMatrix{T}, B} <: AbstractSpinTransforms
    c::V          # cos(θ_k/2)
    s::V          # sin(θ_k/2)
    x::V          # cos θ_k
    φ::V          # longitude
    w::V          # measure over R²
    norm::V       # N_ℓ, ℓ = 0 … lmax
    q::Q          # q[j+1, a+1] = √(j² − a²) for a ≤ j, zero above
    r::Q          # r[j+1, n+3] = √((2j+2)(2j+1)/((j+1+n)(j+1−n))), the seed's step from order j to j + 1
    transfer::V   # Ĝ(ℓ) where the loops read it
    lmax::Int
    execution::B
end

function DirectSpinTransforms(
    grid::FlowGeometries.Grids.AbstractGrid{G,T}; lmax::Union{Nothing,Integer} = nothing,
    execution::ComputationalBackends.AbstractLocalBackend = ComputationalBackends.SerialBackend(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}}
    L = lmax === nothing ? spin_band_limit(grid) : Int(lmax)
    L >= 2 || throw(ArgumentError("the generalized convolution's spin-2 transform needs lmax ≥ 2; got $L"))
    θ, φ, w = _spin_nodes(grid)
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
    return DirectSpinTransforms(
        put(cos.(θ ./ 2)), put(sin.(θ ./ 2)), put(cos.(θ)), put(φ), put(w), put(nrm), put(q), put(r),
        put(zeros(T, L + 1)), L, execution,
    )
end

Base.show(io::IO, tr::DirectSpinTransforms) =
    print(io, "DirectSpinTransforms(", length(tr.x), " cells, lmax = ", tr.lmax, ")")
Base.show(io::IO, ::MIME"text/plain", tr::DirectSpinTransforms) = show(io, tr)

spin_field(tr::DirectSpinTransforms{T}) where {T} = _allocate(tr.execution, Complex{T}, (length(tr.x),))
spin_field_size(tr::DirectSpinTransforms) = (length(tr.x),)
spin_position(::DirectSpinTransforms) = CellOrder()
spin_coefficients(tr::DirectSpinTransforms{T}) where {T} =
    _allocate(tr.execution, Complex{T}, (tr.lmax + 1, 2tr.lmax + 1))
spin_transfer_buffer(tr::DirectSpinTransforms{T}) where {T} = zeros(T, tr.lmax + 1)
spin_host(tr::DirectSpinTransforms) = _host_memory(tr.execution)

function spin_analyze!(C, F, tr::DirectSpinTransforms, s::Int)
    abs(s) <= 2 || _spin_out_of_range(s)
    L, n = tr.lmax, -s
    c, sh, x, φ, w, nrm, q, r = tr.c, tr.s, tr.x, tr.φ, tr.w, tr.norm, tr.q, tr.r
    FlowGeometries.Execution.run_indices(2L + 1, tr.execution) do col
        _direct_analysis!(C, F, c, sh, x, φ, w, nrm, q, r, L, n, col)
    end
    return C
end

function spin_synthesize!(F, C, _W, Ĝ, tr::DirectSpinTransforms, s::Int)
    abs(s) <= 2 || _spin_out_of_range(s)
    copyto!(tr.transfer, Ĝ)
    L, n = tr.lmax, -s
    g, c, sh, x, φ, nrm, q, r = tr.transfer, tr.c, tr.s, tr.x, tr.φ, tr.norm, tr.q, tr.r
    FlowGeometries.Execution.run_indices(length(x), tr.execution) do k
        @inbounds F[k] = _direct_synthesis(C, g, c[k], sh[k], x[k], φ[k], nrm, q, r, L, n)
    end
    return F
end

@noinline _spin_out_of_range(s) = throw(ArgumentError("the direct spin transforms carry |s| ≤ 2; got s = $s"))

# The exponent step `B`: a recurrence value is carried as `(d, e)`, worth `d·2^e`, with `e ≤ 0` a multiple
# of `−B` and `|d| ≥ 2^−B` wherever `e < 0`.
@inline _exponent_step(::Type{T}) where {T<:AbstractFloat} = exponent(floatmax(T)) ÷ 2

# `d^ℓ₀_{m,n}(θ)` at `ℓ₀ = max(|m|, |n|)` as `(d, e)`, from `c = cos(θ/2)`, `s = sin(θ/2)`: for `|m| ≥ |n|`,
# `(−1)^{m−n} √C(2m, m+n) c^{m+n} s^{m−n}` at `m ≥ 0` and `√C(2|m|, |m|−n) c^{|m|−n} s^{|m|+n}` at `m < 0`,
# each built up from order `|n|` by the steps `r`; for `|m| < |n|`, `√C(2|n|, |n|+m) c^{|n|+m} s^{|n|−m}` at
# `n > 0` and `(−1)^{m+|n|} √C(2|n|, |n|+m) c^{|n|−m} s^{|n|+m}` at `n < 0`.
@inline function _wigner_seed(m::Int, n::Int, c::T, s::T, r) where {T}
    if abs(m) >= abs(n)
        B = _exponent_step(T)
        lo = ldexp(one(T), -B)
        j0 = abs(n)
        v = m >= 0 ? c^(j0 + n) * s^(j0 - n) : c^(j0 - n) * s^(j0 + n)
        e = 0
        cs = c * s
        @inbounds for j in j0:(abs(m) - 1)
            v *= r[j + 1, n + 3] * cs
            if zero(T) < abs(v) < lo
                v = ldexp(v, B)
                e -= B
            end
        end
        return ((m >= 0 && isodd(m - n)) ? -v : v), e
    end
    l0 = abs(n)
    b = sqrt(T(_small_binomial(2l0, l0 + m)))
    n > 0 && return b * c^(l0 + m) * s^(l0 - m), 0
    return (isodd(m + l0) ? -b : b) * c^(l0 - m) * s^(l0 + m), 0
end

# `(d, dp, e)` with both significands shifted down by `2^B` once `|d|` passes 1 at `e < 0`; the values they
# carry are unchanged. `|d^ℓ_{mn}| ≤ 1` and the recurrence grows `d` by a bounded factor per degree, so `d`
# stays within `[2^−B, 1]` up to that factor while `e < 0`, and `e` returns to 0 once the value passes `2^−B`.
@inline function _wigner_rescale(d::T, dp::T, e::Int) where {T}
    (e < 0 && abs(d) > one(T)) || return d, dp, e
    B = _exponent_step(T)
    return ldexp(d, -B), ldexp(dp, -B), e + B
end

# The value `d·2^e`.
@inline _wigner_value(d, e::Int) = iszero(e) ? d : ldexp(d, e)

@inline function _small_binomial(a::Int, b::Int)
    v = 1
    for i in 1:b
        v = v * (a - b + i) ÷ i
    end
    return v
end

# `d^{ℓ+1}_{mn}` from `d = d^ℓ_{mn}` and `dp = d^{ℓ−1}_{mn}` at `x = cos θ`, `ℓ ≥ 1`:
# `d^{ℓ+1} = [(2ℓ+1)((ℓ+1)x − mn/ℓ) d^ℓ − ((ℓ+1)/ℓ) √((ℓ²−m²)(ℓ²−n²)) d^{ℓ−1}] / √(((ℓ+1)²−m²)((ℓ+1)²−n²))`.
@inline function _wigner_next(d::T, dp::T, x::T, ℓ::Int, m::Int, n::Int, q) where {T}
    am, an = abs(m) + 1, abs(n) + 1
    @inbounds D = q[ℓ + 2, am] * q[ℓ + 2, an]
    @inbounds P = q[ℓ + 1, am] * q[ℓ + 1, an]
    return (T(2ℓ + 1) * (T(ℓ + 1) * x - T(m * n) / T(ℓ)) * d - T(ℓ + 1) / T(ℓ) * P * dp) / D
end

# Column `col`, order `m = col − L − 1`, of the analysis of `F`.
@inline function _direct_analysis!(C, F, c, s, x, φ, w, nrm, q, r, L::Int, n::Int, col::Int)
    T = eltype(x)
    m = col - L - 1
    l0 = max(abs(m), abs(n))
    z = zero(eltype(C))
    @inbounds for row in 1:(L + 1)
        C[row, col] = z
    end
    @inbounds for k in 1:length(x)
        f = F[k] * w[k]
        iszero(f) && continue
        sn, cn = sincos(T(m) * φ[k])
        a = f * complex(cn, -sn)
        xk = x[k]
        d, e = _wigner_seed(m, n, c[k], s[k], r)
        dp = zero(T)
        C[l0 + 1, col] += a * (nrm[l0 + 1] * _wigner_value(d, e))
        for ℓ in l0:(L - 1)
            dn = ℓ == 0 ? xk * d : _wigner_next(d, dp, xk, ℓ, m, n, q)
            d, dp, e = _wigner_rescale(dn, d, e)
            C[ℓ + 2, col] += a * (nrm[ℓ + 2] * _wigner_value(d, e))
        end
    end
    return nothing
end

# The synthesis at one cell, its `cos(θ/2)`, `sin(θ/2)`, `cos θ` and longitude given.
@inline function _direct_synthesis(C, g, c::T, s::T, x::T, φ::T, nrm, q, r, L::Int, n::Int) where {T}
    acc = zero(Complex{T})
    @inbounds for m in -L:L
        l0 = max(abs(m), abs(n))
        col = m + L + 1
        d, e = _wigner_seed(m, n, c, s, r)
        dp = zero(T)
        col_sum = C[l0 + 1, col] * (g[l0 + 1] * nrm[l0 + 1] * _wigner_value(d, e))
        for ℓ in l0:(L - 1)
            dn = ℓ == 0 ? x * d : _wigner_next(d, dp, x, ℓ, m, n, q)
            d, dp, e = _wigner_rescale(dn, d, e)
            col_sum += C[ℓ + 2, col] * (g[ℓ + 2] * nrm[ℓ + 2] * _wigner_value(d, e))
        end
        sn, cn = sincos(T(m) * φ)
        acc += col_sum * complex(cn, sn)
    end
    return acc
end
