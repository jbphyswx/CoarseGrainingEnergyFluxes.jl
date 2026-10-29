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
# cells make its matrix singular, and its minimum-norm solution is taken there.

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

# Cell `J`'s normal coordinates about target `I`, in `T`.
@inline _ho_normal(::Type{T}, grid, I, J) where {T} =
    map(T, Tuple(FlowGeometries.Geometry.normal_coordinates(grid, I, J)))

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
    e = _ho_normal(T, grid, I, J)
    A = T(_cell_measure(grid, J))
    x = _cell_size(r.support, A, Val(D))
    _ho_meets(r.support, e, x, r.half) || return (false, zero(T))
    φ = _ho_basis(map(v -> v / r.half, e), Val(P))
    p = zero(T)
    @inbounds for i in eachindex(φ)
        p += r.coeffs[i, t] * φ[i]
    end
    # The arithmetic of `_ho_correct!`'s `w̃ + A p`, so a streamed weight equals the cached one bit for bit.
    return (true, A * _ho_mean(r.support, r.kernel, r.scale, e, x) + A * p)
end

"""
    HighOrderScratch

One target's support cells — index, normal coordinates, size, measure, weight, and whether the cell lies
past a bounded edge — with the matrix, right-hand side and moment buffer its correction uses.
"""
struct HighOrderScratch{
    T<:AbstractFloat, D, C,
    VC<:AbstractVector{C}, VE<:AbstractVector{NTuple{D,T}}, VT<:AbstractVector{T},
    VB<:AbstractVector{Bool}, MT<:AbstractMatrix{T},
}
    cell::VC
    e::VE
    x::VE
    A::VT
    w::VT
    outside::VB
    G::MT
    Gc::MT
    r::VT
    m::VT
end

function HighOrderScratch(::Type{T}, ::Val{D}, nb::Int, ::Type{C}) where {T,D,C}
    return HighOrderScratch(
        C[], NTuple{D,T}[], NTuple{D,T}[], T[], T[], Bool[],
        zeros(T, nb, nb), zeros(T, nb, nb), zeros(T, nb), zeros(T, nb),
    )
end

function _ho_reset!(s::HighOrderScratch)
    empty!(s.cell); empty!(s.e); empty!(s.x); empty!(s.A); empty!(s.w); empty!(s.outside)
    return s
end

@inline function _ho_record!(
    s::HighOrderScratch{T,D}, J, e::NTuple{D,T}, x::NTuple{D,T}, A::T, w::T, outside::Bool,
) where {T,D}
    push!(s.cell, J); push!(s.e, e); push!(s.x, x); push!(s.A, A); push!(s.w, w); push!(s.outside, outside)
    return s
end

# Records cell `J` about target `I` when it meets the support, with `w̃` as its weight.
@inline function _ho_push!(
    s::HighOrderScratch{T,D}, sup, grid, I, J, k::Kernels.HighOrderKernel, ℓ::T, h::T, outside::Bool,
) where {T,D}
    e = _ho_normal(T, grid, I, J)
    A = T(_cell_measure(grid, J))
    x = _cell_size(sup, A, Val(D))
    _ho_meets(sup, e, x, h) || return s
    return _ho_record!(s, J, e, x, A, A * _ho_mean(sup, k, ℓ, e, x), outside)
end

# `(Σ A φφᵀ) c = r` for `c`, in place of `r`.
function _ho_solve!(s::HighOrderScratch)
    copyto!(s.Gc, s.G)
    F = LinearAlgebra.cholesky!(LinearAlgebra.Hermitian(s.G, :U); check = false)
    if LinearAlgebra.issuccess(F)
        LinearAlgebra.ldiv!(F, s.r)
    else
        copyto!(s.r, LinearAlgebra.pinv(LinearAlgebra.Symmetric(s.Gc, :U)) * s.r)
    end
    return s.r
end

"""
    _ho_correct!(s, kernel, ℓ, h) -> c

Correct the recorded cells' weights `w̃` in place to `wₖ = w̃ₖ + Aₖ p(eₖ/h)`, with `p = Σ cᵢ φᵢ` the
polynomial that keeps the total and zeroes every moment of order `1…P`, and return `c`. Raises when
the corrected moments do not vanish to `√eps` of `Σ|w|`.
"""
function _ho_correct!(
    s::HighOrderScratch{T,D}, k::Kernels.HighOrderKernel{P}, ℓ::T, h::T,
) where {T,D,P}
    G, r, m = s.G, s.r, s.m
    nb = length(r)
    fill!(G, zero(T))
    fill!(r, zero(T))
    m₀ = zero(T)
    @inbounds for c in eachindex(s.w)
        φ = _ho_basis(map(v -> v / h, s.e[c]), Val(P))
        A = s.A[c]
        w̃ = s.w[c]
        m₀ += w̃
        for j in 1:nb
            Aφ = A * φ[j]
            for i in 1:j
                G[i, j] += Aφ * φ[i]
            end
            r[j] -= w̃ * φ[j]
        end
    end
    φ0 = _ho_basis(ntuple(_ -> zero(T), Val(D)), Val(P))
    @inbounds for i in 1:nb
        r[i] += m₀ * φ0[i]
    end
    c = _ho_solve!(s)
    fill!(m, zero(T))
    total = zero(T)
    @inbounds for q in eachindex(s.w)
        φ = _ho_basis(map(v -> v / h, s.e[q]), Val(P))
        p = zero(T)
        for i in 1:nb
            p += c[i] * φ[i]
        end
        w = s.w[q] + s.A[q] * p
        s.w[q] = w
        total += abs(w)
        for i in 1:nb
            m[i] += w * φ[i]
        end
    end
    worst = zero(T)
    @inbounds for i in 1:nb
        worst = max(worst, abs(m[i] - m₀ * φ0[i]))
    end
    worst <= sqrt(eps(T)) * total || _no_highorder_weights(k, ℓ)
    return c
end

@noinline _no_highorder_weights(::Kernels.HighOrderKernel{P}, ℓ) where {P} = throw(ArgumentError(
    "HighOrderKernel{$P} at ℓ = $ℓ: the corrected weights about a target did not reach $P vanishing " *
    "moments in this precision",
))

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
    w::AbstractVector{T}, s::HighOrderScratch{T,1}, k::Kernels.HighOrderKernel, ss::AbstractVector{T},
    wcs::AbstractVector{T}, ℓ::T,
) where {T<:AbstractFloat}
    h = Kernels.kernel_radius(k, ℓ)
    sup = ProductSupport()
    _ho_reset!(s)
    fill!(w, zero(T))
    for t in eachindex(w)
        e, x = (ss[t],), (wcs[t],)
        _ho_meets(sup, e, x, h) || continue
        _ho_record!(s, t, e, x, wcs[t], wcs[t] * _ho_mean(sup, k, ℓ, e, x), false)
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
    _highorder_targets(grid, kernel, scale, rad, fold, lattice, keep) -> (weights, exterior, kept)

The correction about every cell of `grid`. `fold(f, acc, I)` folds `f(acc, J, d)` over the grid's cells
within `rad` of cell `I`; `lattice` is [`_ho_lattice`](@ref)'s. Returns the [`HighOrderWeights`](@ref),
the mass past a bounded edge per cell (or `nothing`), and, while `keep(n)` holds for the `n` entries so
far, every cell's in-grid neighbours and weights as `(cells, w, ptr)` in fold order, else `nothing`.
"""
function _highorder_targets(
    grid::FlowGeometries.Grids.AbstractGrid, kernel::Kernels.HighOrderKernel{P}, scale::T, rad::T,
    fold::F, lattice, keep::K,
) where {P, T<:AbstractFloat, F, K}
    D = _kernel_dim(grid)
    sup = _ho_support(grid, kernel, scale)
    h = Kernels.kernel_radius(kernel, scale)
    targets = _grid_cells(grid)
    C = eltype(targets)
    nb = _ho_nbasis(D, Val(P))
    s = HighOrderScratch(T, D, nb, C)
    ntarget = length(FlowGeometries.Grids.mask(grid))
    coeffs = Matrix{T}(undef, nb, ntarget)
    dims = lattice === nothing ? nothing : FlowGeometries.Grids.size_tuple(grid)
    exterior = lattice === nothing ? nothing : zeros(T, dims)
    keeping = keep(0)
    cells = C[]
    ws = T[]
    ptr = keeping ? Vector{Int}(undef, ntarget + 1) : Int[]
    keeping && (ptr[1] = 1)
    for (t, I) in enumerate(targets)
        _ho_reset!(s)
        fold(nothing, I) do _, J, _
            _ho_push!(s, sup, grid, I, J, kernel, scale, h, false)
            nothing
        end
        lattice === nothing || _ho_push_exterior!(s, sup, lattice, I, dims, rad, kernel, scale, h)
        c = _ho_correct!(s, kernel, scale, h)
        @inbounds for i in 1:nb
            coeffs[i, t] = c[i]
        end
        if exterior !== nothing
            past = zero(T)
            @inbounds for q in eachindex(s.w)
                s.outside[q] && (past += s.w[q])
            end
            @inbounds exterior[t] = past
        end
        if keeping
            @inbounds for q in eachindex(s.w)
                s.outside[q] && continue
                push!(cells, s.cell[q])
                push!(ws, s.w[q])
            end
            @inbounds ptr[t + 1] = length(ws) + 1
            keeping = keep(length(ws))
            keeping || (cells = C[]; ws = T[]; ptr = Int[])
        end
    end
    weights = HighOrderWeights(kernel, scale, h, coeffs, sup, D)
    return weights, exterior, keeping ? (cells, ws, ptr) : nothing
end

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
    return _highorder_shares_rows(grid, rad, _ho_lattice(grid, rad))
end

_highorder_shares_rows(_, _, _) = false

# The same, given the query radius and the continued lattice ([`_ho_lattice`](@ref)).
function _highorder_shares_rows(grid::FlowGeometries.Grids.StructuredGrid{T,G,2}, rad::T, lattice) where {T,G}
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
    rad = _query_radius(grid, kernel, scale)
    lattice = _ho_lattice(grid, rad)
    _highorder_shares_rows(grid, rad, lattice) || return nothing
    lattice === nothing && return _highorder_rows_on(grid, (0, 0), _query_topology(grid, rad), grid, kernel, scale, rad)
    return _highorder_rows_on(lattice[1], lattice[2], lattice[4], grid, kernel, scale, rad)
end

# The rows over `lat`, the grid itself or its continued lattice, whose cell `I + lo` is grid cell `I`; `mt`
# is the topology its folds use.
function _highorder_rows_on(
    lat::FlowGeometries.Grids.StructuredGrid, lo::NTuple{2,Int}, mt,
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2}, kernel::Kernels.HighOrderKernel{P}, scale::T, rad::T,
) where {T<:AbstractFloat, G, P}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    periodic = FlowGeometries.Grids.isperiodic(grid, 1)
    nλ = FlowGeometries.Grids.size_tuple(lat)[1]
    sup = _ho_support(grid, kernel, scale)
    h = Kernels.kernel_radius(kernel, scale)
    s = HighOrderScratch(T, Val(2), _ho_nbasis(Val(2), Val(P)), NTuple{2,Int})
    di, dj, w = Int[], Int[], T[]
    ptr = Vector{Int}(undef, Ny + 1)
    ptr[1] = 1
    mass = Vector{T}(undef, Ny)
    for j in 1:Ny
        Ie = (1 + lo[1], j + lo[2])
        _ho_reset!(s)
        FlowGeometries.Connectivity.fold_within(
            nothing, lat, Ie...;
            ball = rad, self = true, active_only = false, topology = mt,
            images = FlowGeometries.Connectivity.NearestImage(),
        ) do _, Je, _
            _ho_push!(s, sup, lat, Ie, Je, kernel, scale, h, false)
            nothing
        end
        _ho_correct!(s, kernel, scale, h)
        mass[j] = sum(s.w)
        for q in eachindex(s.w)
            Je = s.cell[q]
            1 <= Je[2] - lo[2] <= Ny || continue
            Δi = Je[1] - Ie[1]
            periodic && (Δi = mod(Δi + nλ ÷ 2, nλ) - nλ ÷ 2)
            push!(di, Δi)
            push!(dj, Je[2] - Ie[2])
            push!(w, s.w[q])
        end
        ptr[j + 1] = length(w) + 1
    end
    return di, dj, w, ptr, mass
end
