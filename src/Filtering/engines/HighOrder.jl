# ---------------------------------------------------------------------------
# HighOrderKernel on a grid
# ---------------------------------------------------------------------------
#
# Sadek & Aluie (2018) define the kernel on the plane as the product `G(x)G(y)` of a one-dimensional
# profile whose moments `∫ sⁿ G(s) ds`, `n = 1…P`, vanish. A sphere has no global axes, so there the kernel
# is its isotropic form: the same body and limbs as a function of geodesic distance, amplitudes from
# [`Kernels.radial_limb_amplitudes`](@ref), which has the same `P` vanishing moments of total degree, or
# on a sphere's surface from `Kernels.spherical_limb_amplitudes`, with the sphere's own moments.
#
# A grid sums over cells, so the filter's order is set by the moments of its weights, `Σₖ wₖ eₖᵅ` over the
# cells' displacements `eₖ` (`FlowGeometries.Geometry.normal_coordinates` about the target), which the
# kernel's cell integrals meet only as the cells shrink. With `Aₖ` the cell measure and `w̃ₖ` the kernel's
# mean over the cell times `Aₖ`, the weights are the `w` nearest `w̃` in `Σ (wₖ − w̃ₖ)²/Aₖ` that keep the
# total and zero every moment with `0 < |α| ≤ P`:
#
#     wₖ = w̃ₖ + Aₖ p(eₖ/h),   p = Σ cᵢ φᵢ,   (Σₖ Aₖ φ(eₖ/h) φ(eₖ/h)ᵀ) c = φ(0) Σₖ w̃ₖ − Σₖ w̃ₖ φ(eₖ/h),
#
# `φᵢ` the tensor Legendre polynomials of total degree ≤ P and `h` the support's half-width. The sums run
# over the cells that meet the support, those of the lattice continued past a bounded edge included. The
# moment conditions span every polynomial of degree ≤ P, a set no rotation of the axes changes, so the
# corrected weights of the isotropic form depend on no choice of axes.
#
# A rectilinear Cartesian grid takes the product one axis at a time, with the one-dimensional case of the
# correction. The identity filter meets these conditions, so the system is consistent even where too few
# cells make its matrix singular, and there every solution `c` gives the same correction `Aₖ p(eₖ/h)`. The
# sums and the solve run in `Float64` at least.

"""
    _ho_exponents(Val(D), Val(P)) -> NTuple

The exponents `(a₁, …, a_D)` with `Σ aᵢ ≤ P`, the constant first: one per tensor Legendre polynomial of
total degree at most `P`.
"""
@generated function _ho_exponents(::Val{D}, ::Val{P}) where {D,P}
    exps = NTuple{D,Int}[]
    for deg in 0:P, a in Iterators.product(ntuple(_ -> 0:deg, D)...)
        sum(a) == deg && push!(exps, a)
    end
    return Tuple(exps)
end

@inline _ho_nbasis(::Val{D}, ::Val{P}) where {D,P} = length(_ho_exponents(Val(D), Val(P)))

# The polynomials a correction solves for, as indices into `_ho_exponents`: every one, or those even in the first
# coordinate, which alone carry a correction over cells set symmetrically about the target along it.
struct _AllPolynomials end
struct _EvenInFirst end

@generated _ho_indices(::Val{D}, ::Val{P}, ::_AllPolynomials) where {D,P} = Tuple(1:_ho_nbasis(Val(D), Val(P)))
@generated _ho_indices(::Val{D}, ::Val{P}, ::_EvenInFirst) where {D,P} =
    Tuple(i for (i, a) in enumerate(_ho_exponents(Val(D), Val(P))) if iseven(a[1]))

# `(P₀(x), …, P_P(x))`, by `(n + 1) P_{n+1} = (2n + 1) x Pₙ − n P_{n−1}`.
@generated function _legendre(x::T, ::Val{P}) where {T,P}
    body = [:(p0 = one(T)), :(p1 = x)]
    for n in 1:(P - 1)
        push!(body, :($(Symbol(:p, n + 1)) = (T($(2n + 1)) * x * $(Symbol(:p, n)) - T($n) * $(Symbol(:p, n - 1))) / T($(n + 1))))
    end
    return Expr(:block, body..., Expr(:tuple, (Symbol(:p, n) for n in 0:P)...))
end

# The basis at `u`, in the order of `_ho_exponents`.
@inline function _ho_basis(u::NTuple{D,Real}, ::Val{P}) where {D,P}
    L = map(x -> _legendre(x, Val(P)), u)
    return map(a -> _ho_term(L, a), _ho_exponents(Val(D), Val(P)))
end

@inline _ho_term(L, a::NTuple{D,Int}) where {D} = prod(ntuple(d -> @inbounds(L[d][a[d] + 1]), Val(D)))

"""
    ProductSupport

The form of Sadek & Aluie (2018), `G(e₁)⋯G(e_D)` over the plane's own axes. A cell is taken as a
box of its extents; off a rectilinear grid, the cube of its measure.
"""
struct ProductSupport end

"""
    RadialSupport{L,T}

The isotropic form, `G(|e|)` with limb amplitudes `amps`. A cell is taken as the ball of its measure.
"""
struct RadialSupport{L, T<:AbstractFloat}
    amps::NTuple{L,T}
end

# The form a `HighOrderKernel` takes on a grid's geometry: the product where the plane gives axes, the
# isotropic form elsewhere, with the sphere's own moments on a sphere's surface.
_ho_support(grid::FlowGeometries.Grids.AbstractGrid{G,T}, k::Kernels.HighOrderKernel, ℓ::T) where {G,T} =
    _ho_support(FlowGeometries.Grids.grid_geometry(grid), k, ℓ, _kernel_dim(grid))
_ho_support(::FlowGeometries.Geometry.AbstractCartesianGeometry, ::Kernels.HighOrderKernel, ::Real, ::Val) =
    ProductSupport()
_ho_support(::FlowGeometries.Geometry.AbstractGeometry, k::Kernels.HighOrderKernel, ::T, ::Val{D}) where {T,D} =
    RadialSupport(Kernels.radial_limb_amplitudes(k, T, Val(D)))
_ho_support(geo::FlowGeometries.Geometry.AbstractSphericalGeometry, k::Kernels.HighOrderKernel, ℓ::T, ::Val{2}) where {T} =
    RadialSupport(Kernels.spherical_limb_amplitudes(k, ℓ, T(FlowGeometries.Geometry.radius(geo))))

# A cell's size on the local frame: the cube's side under the product, the ball's radius under the
# isotropic form, once per direction.
@inline _cell_size(::ProductSupport, A::T, ::Val{D}) where {T,D} = ntuple(_ -> _cube_side(A, Val(D)), Val(D))
@inline _cell_size(::RadialSupport, A::T, ::Val{D}) where {T,D} = ntuple(_ -> _ball_radius(A, Val(D)), Val(D))

@inline _cube_side(A, ::Val{1}) = A
@inline _cube_side(A, ::Val{2}) = sqrt(A)
@inline _cube_side(A, ::Val{3}) = cbrt(A)
@inline _ball_radius(A::T, ::Val{1}) where {T} = A / T(2)
@inline _ball_radius(A::T, ::Val{2}) where {T} = sqrt(A / T(π))
@inline _ball_radius(A::T, ::Val{3}) where {T} = cbrt(T(3) * A / (T(4) * T(π)))
@inline _ball_volume(ρ::T, ::Val{1}) where {T} = T(2) * ρ
@inline _ball_volume(ρ::T, ::Val{2}) where {T} = T(π) * ρ * ρ
@inline _ball_volume(ρ::T, ::Val{3}) where {T} = T(4) * T(π) * ρ^3 / T(3)

# Whether a cell of size `x` at `e` meets the support of half-width `h`.
@inline _ho_meets(::ProductSupport, e::NTuple{D,T}, x::NTuple{D,T}, h::T) where {D,T} =
    all(ntuple(d -> @inbounds(abs(e[d]) - x[d] / T(2) < h), Val(D)))
@inline _ho_meets(::RadialSupport, e::NTuple{D,T}, x::NTuple{D,T}, h::T) where {D,T} =
    sqrt(sum(abs2, e)) - @inbounds(x[1]) < h

# The kernel's mean over a cell of size `x` at `e`.
@inline _ho_mean(::ProductSupport, k::Kernels.HighOrderKernel, ℓ::T, e::NTuple{D,T}, x::NTuple{D,T}) where {D,T} =
    prod(ntuple(d -> @inbounds(Kernels.profile_cell_average(k, e[d], x[d], ℓ)), Val(D)))

# `G = Σₙ (aₙ − aₙ₊₁)·[r < ρₙ]` over `n = 0…L`, `a₀ = 1` the body and `a_{L+1} = 0`, so its mean over a
# ball is a sum of ball intersections.
@inline function _ho_mean(
    s::RadialSupport{L,T}, k::Kernels.HighOrderKernel, ℓ::T, e::NTuple{D,T}, x::NTuple{D,T},
) where {L,T,D}
    r = sqrt(sum(abs2, e))
    ρc = @inbounds x[1]
    b = T(k.b_over_ℓ) * ℓ
    acc = zero(T)
    for n in 0:L
        an = n == 0 ? one(T) : @inbounds(s.amps[n])
        an1 = n == L ? zero(T) : @inbounds(s.amps[n + 1])
        acc += (an - an1) * _ball_overlap(ℓ / T(2) + T(n) * b, ρc, r, Val(D))
    end
    return acc / _ball_volume(ρc, Val(D))
end

# The measure of the intersection of the balls of radii `R` and `ρ` whose centres are `d` apart.
@inline _ball_overlap(R::T, ρ::T, d::T, ::Val{1}) where {T} = max(zero(T), min(d + ρ, R) - max(d - ρ, -R))

@inline function _ball_overlap(R::T, ρ::T, d::T, ::Val{2}) where {T}
    d >= R + ρ && return zero(T)
    d <= abs(R - ρ) && return T(π) * min(R, ρ)^2
    a = acos(clamp((d * d + R * R - ρ * ρ) / (T(2) * d * R), -one(T), one(T)))
    c = acos(clamp((d * d + ρ * ρ - R * R) / (T(2) * d * ρ), -one(T), one(T)))
    return R * R * (a - sin(a) * cos(a)) + ρ * ρ * (c - sin(c) * cos(c))
end

@inline function _ball_overlap(R::T, ρ::T, d::T, ::Val{3}) where {T}
    d >= R + ρ && return zero(T)
    d <= abs(R - ρ) && return T(4) * T(π) * min(R, ρ)^3 / T(3)
    return T(π) * (R + ρ - d)^2 * (d * d + T(2) * d * (R + ρ) - T(3) * (R - ρ)^2) / (T(12) * d)
end

# The ball holding every cell that meets the support.
@inline _ho_reach(::ProductSupport, h::T, x::NTuple{D,T}) where {D,T} = sqrt(T(D)) * (h + @inbounds(x[1]) / T(2))
@inline _ho_reach(::RadialSupport, h::T, x::NTuple{D,T}) where {D,T} = h + @inbounds(x[1])

# The cells of `grid`, as the index each traversal names one by.
_grid_cells(grid::FlowGeometries.Grids.AbstractGrid) = _grid_cells(_cell_address(grid), grid)
_grid_cells(::FlowGeometries.Grids.FlatCells, grid) = 1:length(FlowGeometries.Grids.mask(grid))
_grid_cells(::FlowGeometries.Grids.CartesianCells, grid) =
    Iterators.map(Tuple, CartesianIndices(FlowGeometries.Grids.size_tuple(grid)))

# The type a traversal names a cell by: an id, or a tuple of as many indices as the grid has axes.
_cell_type(grid::FlowGeometries.Grids.AbstractGrid) = _cell_type(_cell_address(grid), grid)
_cell_type(::FlowGeometries.Grids.FlatCells, _) = Int
_cell_type(::FlowGeometries.Grids.CartesianCells, grid) = _index_tuple(FlowGeometries.Grids.size_tuple(grid))
_index_tuple(::NTuple{N,Int}) where {N} = NTuple{N,Int}

# Cell `J`'s linear index among `grid`'s cells, as a neighbour list stores it; past `typemax(Int32)` it raises.
@inline _cell_id(grid::FlowGeometries.Grids.AbstractGrid, J) = _cell_id(_cell_address(grid), grid, J)
@inline _cell_id(::FlowGeometries.Grids.FlatCells, _, t::Int) = Int32(t)
@inline _cell_id(::FlowGeometries.Grids.CartesianCells, grid, J::Tuple) =
    Int32(LinearIndices(FlowGeometries.Grids.size_tuple(grid))[J...])

function _query_radius(
    grid::FlowGeometries.Grids.AbstractGrid, kernel::Kernels.HighOrderKernel, scale::T,
) where {T<:AbstractFloat}
    sup = _ho_support(grid, kernel, scale)
    Amax = zero(T)
    for J in _grid_cells(grid)
        Amax = max(Amax, T(_cell_measure(grid, J)))
    end
    return _ho_reach(sup, Kernels.kernel_radius(kernel, scale), _cell_size(sup, Amax, _kernel_dim(grid)))
end

# Cell `J`'s normal coordinates about target `I` in `W`, from the pair's coordinates (the nearest image across a
# periodic seam) converted to `W`.
@inline function _ho_normal(::Type{W}, grid, I, J) where {W}
    a, b = FlowGeometries.Grids._cell_image_pair(grid, I, J)
    geo = FlowGeometries.Geometry.similar_geometry(W, FlowGeometries.Grids.grid_geometry(grid))
    return map(W, Tuple(FlowGeometries.Geometry.normal_coordinates(geo, map(W, a), map(W, b))))
end

# Whether a cell of measure `A` at normal coordinates `e` (in `W`) meets the support of half-width `h`, and its size.
@inline function _ho_cell(sup, e::NTuple{D,W}, A::T, h::T) where {D,W,T}
    x = _cell_size(sup, A, Val(D))
    return _ho_meets(sup, e, map(W, x), W(h)), x
end

"""
    HighOrderWeights{P,D,T,K,C,S}

A `HighOrderKernel{P}`'s weight on a cell about target `t`, in `D` normal coordinates under the support
`S`: the kernel's mean over the cell plus the target's correction `Σᵢ coeffs[i, t] φᵢ(e/half)`, times the
cell measure.
"""
struct HighOrderWeights{
    P, D, T<:AbstractFloat, K<:Kernels.HighOrderKernel{P}, C<:AbstractMatrix{T},
    S<:Union{ProductSupport,RadialSupport},
}
    kernel::K
    scale::T
    half::T
    coeffs::C
    support::S
end

HighOrderWeights(kernel::K, scale::T, half::T, coeffs::C, support::S, ::Val{D}) where {
    P, D, T<:AbstractFloat, K<:Kernels.HighOrderKernel{P}, C<:AbstractMatrix{T}, S,
} = HighOrderWeights{P,D,T,K,C,S}(kernel, scale, half, coeffs, support)

_with_coeffs(r::HighOrderWeights{P,D}, coeffs) where {P,D} =
    HighOrderWeights(r.kernel, r.scale, r.half, coeffs, r.support, Val(D))

@inline function _cell_weight(r::HighOrderWeights{P,D,T}, grid, t, I, J, _) where {P,D,T}
    W = promote_type(T, Float64)
    e = _ho_normal(W, grid, I, J)
    A = T(_cell_measure(grid, J))
    meets, x = _ho_cell(r.support, e, A, r.half)
    meets || return (false, zero(T))
    hw = W(r.half)
    φ = _ho_basis(map(v -> v / hw, e), Val(P))
    p = zero(W)
    @inbounds for i in eachindex(φ)
        p += W(r.coeffs[i, t]) * φ[i]
    end
    # `w̃ + A p`: the cached weight of `_ho_correct!`, to rounding.
    return (true, T(W(A * _ho_mean(r.support, r.kernel, r.scale, map(T, e), x)) + W(A) * p))
end

"""
    HighOrderScratch

One target's support cells — index, normal coordinates (in `W = promote_type(T, Float64)`), measure, weight, whether
the cell lies past a bounded edge, and whether it stands for its mirror image too — with the correction's moment
matrix `G = Σ A φφᵀ`, its right-hand side `r`
(then its solution), the solve's work vector `y` and pivot order `piv`, and a block of cells' basis values `φ`, `Aφ`
and weights `w̃` the sums run over (`_HO_BLOCK` rows), all in `W`.
"""
struct HighOrderScratch{
    T<:AbstractFloat, D, C, W<:AbstractFloat,
    VC<:AbstractVector{C}, VE<:AbstractVector{NTuple{D,W}}, VT<:AbstractVector{T}, VB<:AbstractVector{Bool},
    MW<:AbstractMatrix{W}, VW<:AbstractVector{W}, VI<:AbstractVector{Int},
}
    cell::VC
    e::VE
    A::VT
    w::VT
    outside::VB
    pair::VB
    G::MW
    r::VW
    y::VW
    piv::VI
    φ::MW
    Aφ::MW
    w̃::VW
end

# The cells per block of the correction's sums, whose `φ` and `Aφ` stay in a core's first-level cache.
const _HO_BLOCK = 64

function HighOrderScratch(::Type{T}, ::Val{D}, nb::Int, ::Type{C}) where {T,D,C}
    W = promote_type(T, Float64)
    return HighOrderScratch(C[], NTuple{D,W}[], T[], T[], Bool[], Bool[], zeros(W, nb, nb), zeros(W, nb), zeros(W, nb),
                            zeros(Int, nb), zeros(W, _HO_BLOCK, nb), zeros(W, _HO_BLOCK, nb), zeros(W, _HO_BLOCK))
end

function _ho_reset!(s::HighOrderScratch)
    empty!(s.cell); empty!(s.e); empty!(s.A); empty!(s.w); empty!(s.outside); empty!(s.pair)
    return s
end

@inline function _ho_record!(
    s::HighOrderScratch{T,D,C,W}, J, e::NTuple{D,W}, A::T, w::T, outside::Bool, pair::Bool = false,
) where {T,D,C,W}
    push!(s.cell, J); push!(s.e, e); push!(s.A, A); push!(s.w, w); push!(s.outside, outside); push!(s.pair, pair)
    return s
end

# Records cell `J` about target `I` when it meets the support, with `w̃` as its weight.
@inline function _ho_push!(
    s::HighOrderScratch{T,D,C,W}, sup, grid, I, J, k::Kernels.HighOrderKernel, ℓ::T, h::T, outside::Bool,
) where {T,D,C,W}
    e = _ho_normal(W, grid, I, J)
    A = T(_cell_measure(grid, J))
    meets, x = _ho_cell(sup, e, A, h)
    meets || return s
    return _ho_record!(s, J, e, A, A * _ho_mean(sup, k, ℓ, map(T, e), x), outside)
end

# `G c = r` for the moment matrix `G` (its lower triangle) of `s`, into `s.r`: the Cholesky factorization with diagonal
# pivoting `PᵀGP = LLᵀ` (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., §10.3), stopped at the
# first pivot at or below `nb·eps` of the largest diagonal entry, `L₁₁` its leading `k` columns, and
# `c = P[L₁₁⁻ᵀL₁₁⁻¹(Pᵀr)₁:ₖ; 0]`. A target with too few cells for every polynomial has `k < nb`; its system is
# consistent, since the identity filter meets it, and the correction `Aₖ p(eₖ/h)` is the same for every solution.
_ho_solve!(s::HighOrderScratch) = _ho_solve!(s.G, s.r, s.y, s.piv, length(s.r))

# The same on the leading `nb` rows and columns of the buffers.
function _ho_solve!(G::AbstractMatrix{W}, r::AbstractVector{W}, y::AbstractVector{W}, piv::AbstractVector{Int}, nb::Int) where {W}
    dmax = zero(W)
    @inbounds for j in 1:nb
        piv[j] = j
        dmax = max(dmax, G[j, j])
    end
    tol = nb * eps(W) * dmax
    k = 0
    @inbounds for i in 1:nb
        p = i
        for j in (i + 1):nb
            G[j, j] > G[p, p] && (p = j)
        end
        G[p, p] > tol || break
        p == i || _ho_swap!(G, i, p, nb)
        piv[i], piv[p] = piv[p], piv[i]
        L = sqrt(G[i, i])
        G[i, i] = L
        for q in (i + 1):nb
            G[q, i] /= L
        end
        for j in (i + 1):nb
            Lj = G[j, i]
            for q in j:nb
                G[q, j] -= G[q, i] * Lj
            end
        end
        k = i
    end
    @inbounds for i in 1:k
        acc = r[piv[i]]
        for j in 1:(i - 1)
            acc -= G[i, j] * y[j]
        end
        y[i] = acc / G[i, i]
    end
    @inbounds for i in k:-1:1
        acc = y[i]
        for j in (i + 1):k
            acc -= G[j, i] * y[j]
        end
        y[i] = acc / G[i, i]
    end
    fill!(r, zero(W))
    @inbounds for i in 1:k
        r[piv[i]] = y[i]
    end
    return r
end

# Rows and columns `i < p` of the symmetric matrix whose lower triangle `G` holds, exchanged in place.
@inline function _ho_swap!(G::AbstractMatrix, i::Int, p::Int, nb::Int)
    @inbounds begin
        G[i, i], G[p, p] = G[p, p], G[i, i]
        for q in 1:(i - 1)
            G[i, q], G[p, q] = G[p, q], G[i, q]
        end
        for q in (i + 1):(p - 1)
            G[q, i], G[p, q] = G[p, q], G[q, i]
        end
        for q in (p + 1):nb
            G[q, i], G[q, p] = G[q, p], G[q, i]
        end
    end
    return G
end

"""
    _ho_correct!(s, kernel, ℓ, h) -> c

Correct the recorded cells' weights `w̃` in place to `wₖ = w̃ₖ + Aₖ p(eₖ/h)`, with `p = Σ cᵢ φᵢ` the
polynomial that keeps the total and zeroes every moment of order `1…P`, and return `c`: one pass sums the moment
matrix and right-hand side block by block, a second adds the correction.
"""
_ho_correct!(s::HighOrderScratch{T,D}, k::Kernels.HighOrderKernel{P}, ℓ::T, h::T) where {T,D,P} =
    _ho_correct!(s, k, ℓ, h, _ho_indices(Val(D), Val(P), _AllPolynomials()))

"""
    _ho_correct!(s, kernel, ℓ, h, ix) -> c

The correction over the polynomials `ix` (indices into `_ho_exponents`), each recorded cell counted
[`_ho_multiplicity`](@ref) times; `c[i]` is polynomial `ix[i]`'s coefficient.
"""
function _ho_correct!(
    s::HighOrderScratch{T,D}, ::Kernels.HighOrderKernel{P}, ::T, h::T, ix::NTuple{nb,Int},
) where {T,D,P,nb}
    G, r, Φ, AΦ, w̃ = s.G, s.r, s.φ, s.Aφ, s.w̃
    W = eltype(r)
    hw = W(h)
    fill!(G, zero(W))
    fill!(r, zero(W))
    m₀ = zero(W)
    n = length(s.w)
    for c0 in 1:_HO_BLOCK:n
        L = min(_HO_BLOCK, n - c0 + 1)
        @inbounds for q in 1:L
            c = c0 + q - 1
            φ = _ho_basis(map(v -> v / hw, s.e[c]), Val(P))
            m = _ho_multiplicity(s, c, W)
            A = W(s.A[c]) * m
            w̃[q] = W(s.w[c]) * m
            m₀ += w̃[q]
            for j in 1:nb
                Φ[q, j] = φ[ix[j]]
                AΦ[q, j] = A * φ[ix[j]]
            end
        end
        @inbounds for j in 1:nb
            acc = zero(W)
            @simd for q in 1:L
                acc += w̃[q] * Φ[q, j]
            end
            r[j] -= acc
            for i in j:nb
                acc = zero(W)
                @simd for q in 1:L
                    acc += AΦ[q, j] * Φ[q, i]
                end
                G[i, j] += acc
            end
        end
    end
    φ0 = _ho_basis(ntuple(_ -> zero(W), Val(D)), Val(P))
    @inbounds for i in 1:nb
        r[i] += m₀ * φ0[ix[i]]
    end
    c = _ho_solve!(G, r, s.y, s.piv, nb)
    @inbounds for q in eachindex(s.w)
        φ = _ho_basis(map(v -> v / hw, s.e[q]), Val(P))
        p = zero(W)
        for i in 1:nb
            p += c[i] * φ[ix[i]]
        end
        s.w[q] = T(W(s.w[q]) + W(s.A[q]) * p)
    end
    return c
end

"""
    _ho_multiplicity(s, c, W) -> W

How many cells recorded cell `c` stands for: 2 where it is marked as standing for its mirror image too, else 1.
"""
@inline _ho_multiplicity(s::HighOrderScratch, c::Int, ::Type{W}) where {W} = @inbounds(s.pair[c]) ? W(2) : one(W)

# ---- one axis: the separable engine -----------------------------------------------------------------

# Node displacements and widths of the stencil's cells about position `i`, into `ss` and `wcs`: wrapped
# across a periodic seam, continued at the edge gap past a bounded edge, as `ZeroFill` continues the
# lattice.
function _stencil_cells!(
    ss::AbstractVector{T}, wcs::AbstractVector{T}, x::AbstractVector{T}, i::Int, lim::Int,
    periodic::Bool, period::T, wfac::AbstractVector{T},
) where {T<:AbstractFloat}
    n = length(x)
    sgn = (n > 1 && x[n] < x[1]) ? -one(T) : one(T)
    for ddi in -lim:lim
        k = ddi + lim + 1
        ii = i + ddi
        if 1 <= ii <= n
            ss[k], wcs[k] = T(x[ii]) - T(x[i]), wfac[ii]
        elseif periodic
            jj = mod1(ii, n)
            ss[k], wcs[k] = T(x[jj]) + sgn * T(fld(ii - 1, n)) * period - T(x[i]), wfac[jj]
        else
            gap = ii < 1 ? T(x[2] - x[1]) : T(x[n] - x[n - 1])
            xe = ii < 1 ? T(x[1]) + (ii - 1) * gap : T(x[n]) + (ii - n) * gap
            ss[k], wcs[k] = xe - T(x[i]), abs(gap)
        end
    end
    return ss, wcs
end

_axis_scratch(::Type{T}, ::Kernels.HighOrderKernel{P}) where {T,P} =
    HighOrderScratch(T, Val(1), _ho_nbasis(Val(1), Val(P)), Int)

# Position `i`'s weights over its stencil cells, those continued past a bounded edge included.
function _highorder_cell_weights!(
    w::AbstractVector{T}, s::HighOrderScratch{T,1}, ss::AbstractVector{T}, wcs::AbstractVector{T},
    x::AbstractVector{T}, i::Int, lim::Int, periodic::Bool, period::T, k::Kernels.HighOrderKernel,
    ℓ::T, wfac::AbstractVector{T},
) where {T<:AbstractFloat}
    _stencil_cells!(ss, wcs, x, i, lim, periodic, period, wfac)
    return _highorder_axis_correct!(w, s, k, ss, wcs, ℓ)
end

# The weights over the cells `(ss, wcs)`, into `w`: zero for a cell outside the support.
function _highorder_axis_correct!(
    w::AbstractVector{T}, s::HighOrderScratch{T,1,C,W}, k::Kernels.HighOrderKernel, ss::AbstractVector{T},
    wcs::AbstractVector{T}, ℓ::T,
) where {T<:AbstractFloat, C, W}
    h = Kernels.kernel_radius(k, ℓ)
    sup = ProductSupport()
    _ho_reset!(s)
    fill!(w, zero(T))
    for t in eachindex(w)
        e = (W(ss[t]),)
        meets, x = _ho_cell(sup, e, wcs[t], h)
        meets || continue
        _ho_record!(s, t, e, wcs[t], wcs[t] * _ho_mean(sup, k, ℓ, (ss[t],), x), false)
    end
    _ho_correct!(s, k, ℓ, h)
    for q in eachindex(s.w)
        w[s.cell[q]] = s.w[q]
    end
    return w
end

# A uniform axis: one stencil for every position. Its constant cell width cancels in the normalization,
# so the table is in units of it, as the other kernels' uniform tables are.
function _separable_axis_weights(
    ::FlowGeometries.Axes.UniformSpacing, x::AbstractVector{T}, lim::Int, ::Bool, ::T,
    kernel::Kernels.HighOrderKernel, scale::T, ::AbstractVector{T},
) where {T<:AbstractFloat}
    Δ = abs(T(step(x)))
    ss = [T(d) * Δ for d in -lim:lim]
    w = similar(ss)
    _highorder_axis_correct!(w, _axis_scratch(T, kernel), kernel, ss, fill(Δ, 2lim + 1), scale)
    return w ./ Δ
end

function _separable_axis_weights(
    x::AbstractRange{T}, lim::Int, periodic::Bool, period::T,
    kernel::Kernels.HighOrderKernel, scale::T, wfac::AbstractVector{T},
) where {T<:AbstractFloat}
    return _separable_axis_weights(FlowGeometries.Axes.UniformSpacing(), x, lim, periodic, period, kernel, scale, wfac)
end

function _separable_axis_weights(
    ::FlowGeometries.Axes.NonuniformSpacing, x::AbstractVector{T}, lim::Int, periodic::Bool, period::T,
    kernel::Kernels.HighOrderKernel, scale::T, wfac::AbstractVector{T},
) where {T<:AbstractFloat}
    n = length(x)
    g = zeros(T, n, 2lim + 1)
    w = Vector{T}(undef, 2lim + 1)
    ss = Vector{T}(undef, 2lim + 1)
    wcs = Vector{T}(undef, 2lim + 1)
    s = _axis_scratch(T, kernel)
    for i in 1:n
        _highorder_cell_weights!(w, s, ss, wcs, x, i, lim, periodic, period, kernel, scale, wfac)
        for ddi in -lim:lim
            (1 <= i + ddi <= n || periodic) || continue   # past a bounded edge: an exact zero, as for every kernel
            g[i, ddi + lim + 1] = w[ddi + lim + 1]
        end
    end
    return g
end

function _separable_axis_weights(
    x::AbstractVector{T}, lim::Int, periodic::Bool, period::T,
    kernel::Kernels.HighOrderKernel, scale::T, wfac::AbstractVector{T},
) where {T<:AbstractFloat}
    return _separable_axis_weights(FlowGeometries.Axes.NonuniformSpacing(), x, lim, periodic, period, kernel, scale, wfac)
end

# On a stretched axis a position's weights, continued cells included, come from one correction, so a
# position whose stencil leaves a bounded axis sums those directly; elsewhere its row of the table is
# the whole stencil, with exact zeros past a bounded edge.
function _separable_profile(
    x::AbstractVector, lim::Int, g::AbstractMatrix{T}, periodic::Bool,
    kernel::Kernels.HighOrderKernel, scale::T, wfac::AbstractVector{T}, exterior::Bool,
) where {T<:AbstractFloat}
    N = length(x)
    profile = zeros(T, N)
    w = Vector{T}(undef, 2lim + 1)
    ss = Vector{T}(undef, 2lim + 1)
    wcs = Vector{T}(undef, 2lim + 1)
    s = _axis_scratch(T, kernel)
    @inbounds for i in 1:N
        if exterior && !periodic && (i - lim < 1 || i + lim > N)
            _highorder_cell_weights!(w, s, ss, wcs, x, i, lim, false, zero(T), kernel, scale, wfac)
            profile[i] = sum(w)
        else
            profile[i] = sum(view(g, i, :))
        end
    end
    return profile
end

# ---- cell by cell: grids without separable axes -----------------------------------------------------

"""
    _ho_lattice(grid, rad) -> (ext, lo, extended, topology) or nothing

`grid`'s lattice continued past each bounded edge ([`_exterior_lattice`](@ref)), with the ball-query
topology its folds use.
"""
function _ho_lattice(grid::FlowGeometries.Grids.StructuredGrid, rad)
    lattice = _exterior_lattice(grid, rad)
    lattice === nothing && return nothing
    ext, lo, extended = lattice
    return (ext, lo, extended, _query_topology(ext, rad))
end

_ho_lattice(::FlowGeometries.Grids.AbstractGrid, _) = nothing

# Records the cells past a bounded edge about grid cell `I`, from the continued lattice.
function _ho_push_exterior!(
    s::HighOrderScratch, sup, lattice, I::NTuple{N,Int}, dims::NTuple{N,Int}, rad,
    k::Kernels.HighOrderKernel, ℓ, h,
) where {N}
    ext, lo, extended, mt = lattice
    Ie = ntuple(d -> I[d] + lo[d], Val(N))
    _window_inside(Ie, FlowGeometries.Connectivity.metric_window(ext, Ie, rad, mt), lo, dims, extended) &&
        return s
    FlowGeometries.Connectivity.fold_within(
        nothing, ext, Ie...;
        ball = rad, self = true, active_only = false, topology = mt, images = _image_convention(ext),
    ) do _, Je, _
        _inside(Je, lo, dims) || _ho_push!(s, sup, ext, Ie, Je, k, ℓ, h, true)
        nothing
    end
    return s
end

"""
    _highorder_targets(grid, kernel, scale, rad, fold, scratch, lattice, keep, stream, execution) -> (weights, exterior, kept)

The correction about every cell of `grid`. `fold(f, acc, I, sc)` folds `f(acc, J, d)` over the grid's cells
within `rad` of cell `I` with the query buffer `sc`, one per task from `scratch()`; `lattice` is
[`_ho_lattice`](@ref)'s. Returns the [`HighOrderWeights`](@ref), holding each cell's coefficients where `stream` (a
plan that forms its weights at every apply) and none otherwise, the mass past a bounded edge per cell (or
`nothing`), and where `keep` every cell's in-grid neighbours and weights as `(ids, w, ptr)` in fold order, `ids` the
neighbours' [`_cell_id`](@ref)s, else `nothing`. The targets are divided among `execution`'s threads: a first pass
counts each target's in-grid cells, the second corrects each target and writes its entries from its offset.
"""
function _highorder_targets(
    grid::FlowGeometries.Grids.AbstractGrid, kernel::Kernels.HighOrderKernel{P}, scale::T, rad::T,
    fold::F, scratch::S, lattice, keep::Bool, stream::Bool, execution,
) where {P, T<:AbstractFloat, F, S}
    D = _kernel_dim(grid)
    sup = _ho_support(grid, kernel, scale)
    h = Kernels.kernel_radius(kernel, scale)
    C = _cell_type(grid)
    nb = _ho_nbasis(D, Val(P))
    ntarget = length(FlowGeometries.Grids.mask(grid))
    coeffs = Matrix{T}(undef, nb, stream ? ntarget : 0)
    dims = lattice === nothing ? nothing : FlowGeometries.Grids.size_tuple(grid)
    exterior = lattice === nothing ? nothing : zeros(T, dims)
    ptr = keep ? Vector{Int}(undef, ntarget + 1) : Int[]
    if keep
        ptr[1] = 1
        _share_run(execution, 1:ntarget) do ts
            sc = scratch()
            for t in ts
                I = _grid_cell(grid, t)
                ptr[t + 1] = fold(0, I, sc) do n, J, _
                    _ho_in_support(sup, grid, I, J, h) ? n + 1 : n
                end
            end
        end
        for t in 1:ntarget
            ptr[t + 1] += ptr[t]
        end
    end
    ids = Vector{Int32}(undef, keep ? ptr[end] - 1 : 0)
    ws = Vector{T}(undef, keep ? ptr[end] - 1 : 0)
    _share_run(execution, 1:ntarget) do ts
        s = HighOrderScratch(T, D, nb, C)
        sc = scratch()
        for t in ts
            I = _grid_cell(grid, t)
            _ho_reset!(s)
            fold(nothing, I, sc) do _, J, _
                _ho_push!(s, sup, grid, I, J, kernel, scale, h, false)
                nothing
            end
            lattice === nothing || _ho_push_exterior!(s, sup, lattice, I, dims, rad, kernel, scale, h)
            c = _ho_correct!(s, kernel, scale, h)
            if stream
                @inbounds for i in 1:nb
                    coeffs[i, t] = c[i]
                end
            end
            if exterior !== nothing
                past = zero(T)
                @inbounds for q in eachindex(s.w)
                    s.outside[q] && (past += s.w[q])
                end
                @inbounds exterior[t] = past
            end
            if keep
                k = ptr[t] - 1
                @inbounds for q in eachindex(s.w)
                    s.outside[q] && continue
                    k += 1
                    ids[k] = _cell_id(grid, s.cell[q])
                    ws[k] = s.w[q]
                end
            end
        end
    end
    weights = HighOrderWeights(kernel, scale, h, coeffs, sup, D)
    return weights, exterior, keep ? (ids, ws, ptr) : nothing
end

# Target `t` of `grid`, as the index each traversal names a cell by (`_grid_cells`' `t`-th).
_grid_cell(grid::FlowGeometries.Grids.AbstractGrid, t::Int) = _grid_cell(_cell_address(grid), grid, t)
_grid_cell(::FlowGeometries.Grids.FlatCells, _, t::Int) = t
_grid_cell(::FlowGeometries.Grids.CartesianCells, grid, t::Int) =
    Tuple(CartesianIndices(FlowGeometries.Grids.size_tuple(grid))[t])

# Whether cell `J` meets the support about target `I`: the test `_ho_push!` records a cell by.
@inline _ho_in_support(sup, grid, I, J, h::T) where {T} =
    first(_ho_cell(sup, _ho_normal(promote_type(T, Float64), grid, I, J), T(_cell_measure(grid, J)), h))

"""
    _highorder_shares_rows(grid, kernel, scale) -> Bool

Whether every target of a latitude row sees the same cells at the same offsets, so one correction serves
the row: on a ring of uniform longitude, and on a regional box of uniform longitude, continued around
the ring, while no support reaches the seam where the continuation closes.
"""
function _highorder_shares_rows(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2}, kernel::Kernels.HighOrderKernel, scale::T,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}}
    rad = _query_radius(grid, kernel, scale)
    return _rows_shared(grid, rad, _ho_lattice(grid, rad))
end

_highorder_shares_rows(_, _, _) = false

# Whether every target of a latitude row sees the same cells at the same offsets, given the query radius
# and the continued lattice ([`_ho_lattice`](@ref)).
_rows_shared(_, _, _) = false

function _rows_shared(grid::FlowGeometries.Grids.StructuredGrid{T,G,2}, rad::T, lattice) where {T,G}
    FlowGeometries.Grids.isuniform(grid, 1) || return false
    FlowGeometries.Grids.isperiodic(grid, 1) && return true
    (lattice !== nothing && lattice[3][1]) || return false
    lat, lo, _, mt = lattice
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    # The continuation's seam columns are the lattice's first and last.
    reach = min(lo[1], FlowGeometries.Grids.size_tuple(lat)[1] - lo[1] - Nx) - 1
    return all(j -> FlowGeometries.Connectivity.metric_window(lat, (1 + lo[1], j + lo[2]), rad, mt)[1] <= reach, 1:Ny)
end

"""
    _highorder_rows(grid, kernel, scale) -> (di, dj, w, ptr, mass) or nothing

A latitude–longitude grid's weights by row, in the banded layout: row `j`'s entries `ptr[j]:ptr[j+1]-1`
hold the longitude and latitude offsets and the weights of the in-grid cells, and `mass[j]` the whole
support's, the continued lattice included. `nothing` where the targets of a row do not share one
correction ([`_highorder_shares_rows`](@ref)).
"""
function _highorder_rows(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2}, kernel::Kernels.HighOrderKernel, scale::T,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}}
    rows = _highorder_band_rows(grid, kernel, scale)
    return rows === nothing ? nothing : _collect_rows(rows, T)
end

"""
    _highorder_band_rows(grid, kernel, scale) -> HighOrderBandRows or nothing

[`_highorder_rows`](@ref)' corrections one latitude row at a time ([`_row_taps!`](@ref)); `nothing` where the
targets of a row do not share one.
"""
function _highorder_band_rows(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2}, kernel::Kernels.HighOrderKernel{P}, scale::T,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}, P}
    rad = _query_radius(grid, kernel, scale)
    lattice = _ho_lattice(grid, rad)
    _rows_shared(grid, rad, lattice) || return nothing
    lat, lo, mt = lattice === nothing ? (grid, (0, 0), _query_topology(grid, rad)) : (lattice[1], lattice[2], lattice[4])
    Ny = FlowGeometries.Grids.size_tuple(grid)[2]
    windows = [FlowGeometries.Connectivity.metric_window(lat, (1 + lo[1], j + lo[2]), rad, mt) for j in 1:Ny]
    s = HighOrderScratch(T, Val(2), _ho_nbasis(Val(2), Val(P)), NTuple{2,Int})
    dls = first.(windows)
    lim = _row_lon_offsets(FlowGeometries.Grids.isperiodic(grid, 1), maximum(dls; init = 0),
                           FlowGeometries.Grids.size_tuple(lat)[1])
    W = promote_type(T, Float64)
    dλ = W(step(FlowGeometries.Grids.coordinates(grid, 1)))
    sinλ = [sin(W(d) * dλ) for d in 0:(-first(lim))]
    havλ = [sin(W(d) * dλ / W(2))^2 for d in 0:(-first(lim))]
    return HighOrderBandRows(lat, lo, mt, grid, kernel, scale, rad, _ho_support(grid, kernel, scale),
                             Kernels.kernel_radius(kernel, scale), s, dls, last.(windows), sinλ, havλ)
end

# The longitude offsets of a row's window of half-width `dl`: each column of a ring once (the antipodal column of an
# even ring from the west), any offset on a box.
_row_lon_offsets(periodic::Bool, dl::Int, nλ::Int) =
    periodic ? (lim = min(dl, nλ ÷ 2); (-lim):(2lim + 1 > nλ ? lim - 1 : lim)) : ((-dl):dl)

"""
    HighOrderBandRows

A latitude–longitude grid's corrected weights by row over `lat`, the grid itself or its continued lattice, whose
cell `I + lo` is grid cell `I`; `mt` is the topology its folds use, `dls` and `djls` each row's index half-widths,
and `sinλ`, `havλ` the `sin Δλ` and `sin²(Δλ/2)` of each longitude offset `Δi ≥ 0`, at `Δi + 1`, in the
correction's `promote_type(T, Float64)`.
"""
struct HighOrderBandRows{
    L, M, GR, K, T<:AbstractFloat, S, HS<:HighOrderScratch{T}, VI<:AbstractVector{Int}, VT<:AbstractVector{<:AbstractFloat},
}
    lat::L
    lo::NTuple{2,Int}
    mt::M
    grid::GR
    kernel::K
    scale::T
    rad::T
    support::S
    half::T
    scratch::HS
    dls::VI
    djls::VI
    sinλ::VT
    havλ::VT
end

_nrows(r::HighOrderBandRows) = length(r.djls)
_row_dj_lim(r::HighOrderBandRows) = min(maximum(r.djls; init = 0), _nrows(r) - 1)
_row_reach(r::HighOrderBandRows) = maximum(r.dls; init = 0)
_row_entries(r::HighOrderBandRows) = sum(j -> (2 * r.dls[j] + 1) * (2 * r.djls[j] + 1), 1:_nrows(r); init = 0)
_has_row_mass(::HighOrderBandRows) = true

_row_worker(r::HighOrderBandRows{L,M,GR,K,T}) where {L,M,GR,K,T} = HighOrderBandRows(
    r.lat, r.lo, r.mt, r.grid, r.kernel, r.scale, r.rad, r.support, r.half,
    HighOrderScratch(T, Val(2), length(r.scratch.r), NTuple{2,Int}), r.dls, r.djls, r.sinλ, r.havλ,
)

# Row `j`'s cells over its window, each lattice row's factors once and each longitude offset's from the tables, in the
# correction's `W`: a cell's normal coordinates by the half-angle formulas of `FlowGeometries.Geometry.normal_coordinates`
# on a sphere. The window is symmetric about the target's meridian and a cell's mirror image across it has the mirrored
# coordinates `(−e₁, e₂)`, so the cells on and east of it are recorded, each east one standing for its mirror too (the
# antipodal column of an even ring is its own), and corrected over the polynomials even in `e₁`, the others' moments
# vanishing over the mirrored set.
function _row_taps!(f::F, r::HighOrderBandRows{L,M,GR,K,T}, j::Int) where {F, L, M, GR, K, T}
    Ny = _nrows(r)
    s, lo, lat = r.scratch, r.lo, r.lat
    W = eltype(r.sinλ)
    periodic = FlowGeometries.Grids.isperiodic(r.grid, 1)
    nλ, nφ = FlowGeometries.Grids.size_tuple(lat)
    φs = FlowGeometries.Grids.coordinates(lat, 2)
    R = W(FlowGeometries.Geometry.radius(FlowGeometries.Grids.grid_geometry(lat)))
    Ie = (1 + lo[1], j + lo[2])
    φ1 = W(φs[Ie[2]])
    sinφ1, cosφ1 = sincos(φ1)
    east = 0:(-first(_row_lon_offsets(periodic, r.dls[j], nλ)))
    djl = r.djls[j]
    hw = W(r.half)
    _ho_reset!(s)
    @inbounds for Jy in max(1, Ie[2] - djl):min(nφ, Ie[2] + djl)
        φ2 = W(φs[Jy])
        Δφ = φ2 - φ1
        cosφ2 = cos(φ2)
        sinΔφ = sin(Δφ)
        havφ = sin(Δφ / W(2))^2
        A = T(_cell_measure(lat, (Ie[1], Jy)))
        x = _cell_size(r.support, A, Val(2))
        xw = map(W, x)
        for Δi in east
            s² = r.havλ[Δi + 1]
            e = _sphere_frame(R, cosφ2 * r.sinλ[Δi + 1], sinΔφ + W(2) * sinφ1 * cosφ2 * s², havφ + cosφ1 * cosφ2 * s²)
            _ho_meets(r.support, e, xw, hw) || continue
            Jx = periodic ? mod1(Ie[1] + Δi, nλ) : Ie[1] + Δi
            pair = Δi > 0 && !(periodic && 2Δi == nλ)
            _ho_record!(s, (Jx, Jy), e, A, A * _ho_mean(r.support, r.kernel, r.scale, map(T, e), x), false, pair)
        end
    end
    _ho_correct!(s, r.kernel, r.scale, r.half, _ho_indices(Val(2), _ho_order(r.kernel), _EvenInFirst()))
    mass = zero(T)
    for q in eachindex(s.w)
        mass += s.pair[q] ? T(2) * s.w[q] : s.w[q]
        Je = s.cell[q]
        1 <= Je[2] - lo[2] <= Ny || continue
        Δi = Je[1] - Ie[1]
        periodic && (Δi = mod(Δi + nλ ÷ 2, nλ) - nλ ÷ 2)
        f(Δi, Je[2] - Ie[2], s.w[q])
        s.pair[q] && f(-Δi, Je[2] - Ie[2], s.w[q])
    end
    return mass
end

@inline _ho_order(::Kernels.HighOrderKernel{P}) where {P} = Val(P)

# The normal coordinates `R·σ·(sin α, cos α)` of the point whose unit vector on the frame has east and north components
# `pe`, `pn` and whose central angle has haversine `hav`.
@inline function _sphere_frame(R::T, pe::T, pn::T, hav::T) where {T}
    σ = T(2) * atan(sqrt(hav), sqrt(max(zero(T), one(T) - hav)))
    sσ = hypot(pe, pn)
    iszero(sσ) && return (zero(T), R * σ)
    k = R * σ / sσ
    return (k * pe, k * pn)
end
