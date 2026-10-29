# ---------------------------------------------------------------------------
# Spin-weighted transforms by direct summation over a grid's cells
# ---------------------------------------------------------------------------

# The Wigner `d^ℓ_{m,n}(θ)` run up FlowTransformBindings' three-term recurrence in `ℓ` from
# `ℓ₀ = max(|m|, |n|)`, as NUFSHT's ring engine runs it: the closed form at `ℓ₀`, Reinsch's step in `Float32`
# and the plain one with `cos θ` carried to twice the working precision otherwise, and a value held as
# `y·2^{Kk}` with an integer scale `k ≤ 0`, rescaled as it grows and summed once `k ≥ −1`.

# Whether transforms in `T` step in Reinsch's form.
@inline _wigner_reinsch(::Type{Float32}) = true
@inline _wigner_reinsch(::Type{<:AbstractFloat}) = false

"""
    _wigner_nodes(T, S, θ) -> (; x, xl, u, pos, lc, ls, cs)

Per colatitude `θ_k`: `cos θ_k` as `x + xl` to twice `T`'s precision (the plain step's), `pos = θ_k ≤ π/2`
and `u = 2sin²(θ_k/2)` where `pos`, `2cos²(θ_k/2)` elsewhere (Reinsch's), and `log₂ cos(θ_k/2)`,
`log₂ sin(θ_k/2)` and `cos(θ_k/2) sin(θ_k/2)` in `S`, which the starting values read.
"""
function _wigner_nodes(::Type{T}, ::Type{S}, θ::AbstractVector) where {T<:AbstractFloat, S<:AbstractFloat}
    n = length(θ)
    x = zeros(T, n); xl = zeros(T, n); u = zeros(T, n); pos = Vector{Bool}(undef, n)
    lc = zeros(S, n); ls = zeros(S, n); cs = zeros(S, n)
    for k in 1:n
        h = S(θ[k]) / 2
        sh, ch = sincos(h)
        pos[k] = 2h <= S(π) / 2
        u[k] = T(2 * (pos[k] ? sh : ch)^2)
        lc[k] = log2(ch)
        ls[k] = log2(sh)
        cs[k] = ch * sh
    end
    _wigner_reinsch(T) || setprecision(BigFloat, 2 * precision(T) + 8) do
        for k in 1:n
            c = cos(BigFloat(θ[k]))
            x[k] = T(c)
            xl[k] = T(c - x[k])
        end
    end
    return (; x, xl, u, pos, lc, ls, cs)
end

# `starts[m + L + 1, n + 3]`: the constants of `d^{ℓ₀}_{m,n}`, `|m| ≤ L`, `|n| ≤ 2`.
_wigner_starts(::Type{S}, L::Int) where {S} =
    [FlowTransformBindings.wigner_d_start(S, m, n) for m in -L:L, n in -2:2]

# The step coefficients per degree a table holds: `(α, β, γ, c₊, c₋)` for Reinsch's form, `(α, β, γ)` for the
# plain one.
@inline _wigner_ncoef(::Type{T}) where {T} = _wigner_reinsch(T) ? 5 : 3

# Column `col`, order `m = col − L − 1`, of the step coefficients at spin `n`: rows `cℓ+1 … cℓ+c`,
# `c = _wigner_ncoef(T)`, hold degree `ℓ`'s, formed in `S` and rounded to the table's type `T`.
@inline function _wigner_coefficients!(coef, ::Type{S}, L::Int, n::Int, col::Int) where {S}
    T = eltype(coef)
    m = col - L - 1
    @inbounds for ℓ in max(abs(m), abs(n)):(L - 1)
        α, β, γ, c₊, c₋ = FlowTransformBindings.wigner_d_coeffs(S, ℓ, m, n)
        i = _wigner_ncoef(T) * ℓ
        coef[i + 1, col] = T(α)
        coef[i + 2, col] = T(β)
        coef[i + 3, col] = T(γ)
        if _wigner_reinsch(T)
            coef[i + 4, col] = T(c₊)
            coef[i + 5, col] = T(c₋)
        end
    end
    return nothing
end

# Degree `ℓ`'s `(α, β, γ, c₊, c₋)` from column `col` of the table; zero `c±` in the plain form.
@inline function _table_coeffs(coef, ℓ::Int, col::Int)
    T = eltype(coef)
    i = _wigner_ncoef(T) * ℓ
    @inbounds if _wigner_reinsch(T)
        return (coef[i + 1, col], coef[i + 2, col], coef[i + 3, col], coef[i + 4, col], coef[i + 5, col])
    else
        return (coef[i + 1, col], coef[i + 2, col], coef[i + 3, col], zero(T), zero(T))
    end
end

function _wigner_coefficients!(coef, ::Type{S}, L::Int, n::Int, execution) where {S}
    FlowGeometries.Execution.run_indices(2L + 1, execution) do col
        _wigner_coefficients!(coef, S, L, n, col)
    end
    return coef
end

# `d^{ℓ+1}` from degree `ℓ` at one node, `(y, e)` the state: `e = y_{ℓ−1}` in the plain form, Reinsch's
# companion otherwise.
@inline function _wigner_step(coef, ℓ::Int, col::Int, pos::Bool, u::T, x::T, xl::T, y::T, e::T) where {T}
    α, β, γ, c₊, c₋ = _table_coeffs(coef, ℓ, col)
    if _wigner_reinsch(T)
        σα, c, σγ, σ = FlowTransformBindings.wigner_d_signed(pos, α, c₊, c₋, γ)
        return FlowTransformBindings.wigner_d_step(σα, c, σγ, σ, u, y, e)
    else
        return FlowTransformBindings.wigner_d_step_plain(α, β, γ, x, xl, y, e)
    end
end

# The state's second value at `ℓ₀`: `y` in Reinsch's form, `y_{ℓ₀−1} = 0` in the plain one.
@inline _wigner_e0(y::T) where {T} = _wigner_reinsch(T) ? y : zero(T)

# The weight of a value at scale `k`: whole at `0`, `dn = 2^{−K}` at `−1`, left out below.
@inline _wigner_weight(k::Int32, dn::T) where {T} =
    ifelse(k == Int32(0), one(T), ifelse(k == Int32(-1), dn, zero(T)))

"""
    DirectSpinTransforms(grid; lmax = nothing, execution = SerialBackend())

The spin-weighted transforms of [`GeneralizedConvolution`](@ref) by direct summation over the cells of a
spherical `grid`, for any spin `|s| ≤ 2`, in the grid's element type and in the memory of the local
backend `execution`, whose loops run them. With `ₛY_ℓm(θ, φ) = N_ℓ d^ℓ_{m,−s}(θ) e^{imφ}`,
`N_ℓ = √((2ℓ+1)/4π)` and `θ` the colatitude, the analysis is the quadrature
`C_ℓm = Σ_k w_k F_k conj(ₛY_ℓm(x_k))` with `w_k` the cell's measure over `R²`, one work item per order `m`,
and the synthesis is `F_k = Σ_ℓm Ĝ(ℓ) C_ℓm ₛY_ℓm(x_k)`, one work item per cell. The coefficients are
`C[ℓ+1, m+lmax+1]`.

The Wigner `d^ℓ_{m,−s}` run up FlowTransformBindings' recurrence in `ℓ` from `ℓ₀ = max(|m|, |s|)`, a value
below the floating-point range carried with its own scale, so an order whose start lies below the range
keeps the values it reaches at higher degree. A transform costs `O(M·lmax²)` over `M` cells and fills its
spin's step coefficients, `O(lmax²)`. `lmax = nothing` takes [`spin_band_limit`](@ref).
"""
struct DirectSpinTransforms{
    T<:AbstractFloat, S<:AbstractFloat, V<:AbstractVector{T}, VB<:AbstractVector{Bool}, VS<:AbstractVector{S},
    ST<:AbstractMatrix{FlowTransformBindings.WignerDStart{S}}, Q<:AbstractMatrix{T}, B,
} <: AbstractSpinTransforms
    x::V          # cos θ_k …
    xl::V         # … and its rest, to twice the precision
    u::V          # Reinsch's 1 ∓ cos θ_k
    pos::VB       # θ_k ≤ π/2
    lc::VS        # log₂ cos(θ_k/2)
    ls::VS        # log₂ sin(θ_k/2)
    φ::V          # longitude
    w::V          # measure over R²
    norm::V       # N_ℓ, ℓ = 0 … lmax
    starts::ST    # starts[m + lmax + 1, n + 3]
    coef::Q       # (_wigner_ncoef(T)·lmax, 2lmax+1): the step coefficients at the spin last transformed
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
    S = promote_type(T, Float64)
    θ, φ, w = _spin_nodes(grid)
    nd = _wigner_nodes(T, S, θ)
    nrm = [sqrt(T(2ℓ + 1) / (4 * T(π))) for ℓ in 0:L]
    put(a) = FlowGeometries.Execution.on_backend(execution, a)
    return DirectSpinTransforms(
        put(nd.x), put(nd.xl), put(nd.u), put(nd.pos), put(nd.lc), put(nd.ls), put(φ), put(w), put(nrm),
        put(_wigner_starts(S, L)), put(zeros(T, _wigner_ncoef(T) * L, 2L + 1)), put(zeros(T, L + 1)), L, execution,
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

function spin_analyze!(C, F, tr::DirectSpinTransforms{T,S}, s::Int) where {T,S}
    abs(s) <= 2 || _spin_out_of_range(s)
    L, n = tr.lmax, -s
    _wigner_coefficients!(tr.coef, S, L, n, tr.execution)
    t = _direct_tables(tr)
    FlowGeometries.Execution.run_indices(2L + 1, tr.execution) do col
        _direct_analysis!(C, F, t, L, n, col)
    end
    return C
end

function spin_synthesize!(F, C, _W, Ĝ, tr::DirectSpinTransforms{T,S}, s::Int) where {T,S}
    abs(s) <= 2 || _spin_out_of_range(s)
    copyto!(tr.transfer, Ĝ)
    L, n = tr.lmax, -s
    _wigner_coefficients!(tr.coef, S, L, n, tr.execution)
    t = _direct_tables(tr)
    FlowGeometries.Execution.run_indices(length(tr.x), tr.execution) do k
        @inbounds F[k] = _direct_synthesis(C, t, L, n, k)
    end
    return F
end

@noinline _spin_out_of_range(s) = throw(ArgumentError("the direct spin transforms carry |s| ≤ 2; got s = $s"))

# Column `col`, order `m = col − L − 1`, of the analysis of `F`.
# What the loops read, which their device kernels capture in place of the transforms.
_direct_tables(tr::DirectSpinTransforms) =
    (; tr.x, tr.xl, tr.u, tr.pos, tr.lc, tr.ls, tr.φ, tr.w, tr.norm, tr.starts, tr.coef, tr.transfer)

@inline function _direct_analysis!(C, F, tr::NamedTuple, L::Int, n::Int, col::Int)
    T = eltype(tr.x)
    m = col - L - 1
    l0 = max(abs(m), abs(n))
    up, dn = FlowTransformBindings.wigner_d_bounds(T)
    st = @inbounds tr.starts[col, n + 3]
    coef, nrm = tr.coef, tr.norm
    z = zero(eltype(C))
    @inbounds for row in 1:(L + 1)
        C[row, col] = z
    end
    @inbounds for k in eachindex(tr.x)
        f = F[k] * tr.w[k]
        iszero(f) && continue
        sn, cn = sincos(T(m) * tr.φ[k])
        a = f * complex(cn, -sn)
        pos, u, x, xl = tr.pos[k], tr.u[k], tr.x[k], tr.xl[k]
        y, sk = FlowTransformBindings.wigner_d_start_value(T, st, tr.lc[k], tr.ls[k])
        e = _wigner_e0(y)
        C[l0 + 1, col] += a * (nrm[l0 + 1] * y * _wigner_weight(sk, dn))
        for ℓ in l0:(L - 1)
            y, e = _wigner_step(coef, ℓ, col, pos, u, x, xl, y, e)
            y, e, sk = FlowTransformBindings.wigner_d_rescale(y, e, sk, up, dn)
            C[ℓ + 2, col] += a * (nrm[ℓ + 2] * y * _wigner_weight(sk, dn))
        end
    end
    return nothing
end

# The synthesis at cell `k`.
@inline function _direct_synthesis(C, tr::NamedTuple, L::Int, n::Int, k::Int)
    T = eltype(tr.x)
    up, dn = FlowTransformBindings.wigner_d_bounds(T)
    coef, nrm, g = tr.coef, tr.norm, tr.transfer
    @inbounds pos, u, x, xl, lc, ls, φ = tr.pos[k], tr.u[k], tr.x[k], tr.xl[k], tr.lc[k], tr.ls[k], tr.φ[k]
    acc = zero(Complex{T})
    @inbounds for m in -L:L
        l0 = max(abs(m), abs(n))
        col = m + L + 1
        y, sk = FlowTransformBindings.wigner_d_start_value(T, tr.starts[col, n + 3], lc, ls)
        e = _wigner_e0(y)
        col_sum = C[l0 + 1, col] * (g[l0 + 1] * nrm[l0 + 1] * y * _wigner_weight(sk, dn))
        for ℓ in l0:(L - 1)
            y, e = _wigner_step(coef, ℓ, col, pos, u, x, xl, y, e)
            y, e, sk = FlowTransformBindings.wigner_d_rescale(y, e, sk, up, dn)
            col_sum += C[ℓ + 2, col] * (g[ℓ + 2] * nrm[ℓ + 2] * y * _wigner_weight(sk, dn))
        end
        sn, cn = sincos(T(m) * φ)
        acc += col_sum * complex(cn, sn)
    end
    return acc
end
