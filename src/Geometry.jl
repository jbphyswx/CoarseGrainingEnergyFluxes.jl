module Geometry

export disk_fraction, quadrant_disk_area2, signed_quadrant_disk_area2, edge_disk_area2, sector_area2

#=
    Blazing fast polygon-distance overlap calculations. These should be useful for kernels that intersect with the grid.

    TODO :: This presumably is right up a GPU's alley and kernels can probably be OOMs faster there.

    If used well these should only be truly necessary for the sliver of cells on the margin (though on GPU you probably apply to all)


    TODO :: Iterative solvers accidentally were placed here, but these are orders of magnitude too slow to be used; also theyre not good for GPU
            We need to remove them and replace them with proper solutions.
=#

# ============================================================================ #
# Rectangular Prism / Axis-Aligned Box Functions (Analytic Fast Paths)
# ============================================================================ #

"""
    quadrant_disk_area2(X::T, Y::T, r::T, r²::T) where {T<:AbstractFloat} -> T

Compute twice the exact area of the intersection between the first-quadrant
rectangle `[0, X] × [0, Y]` (with `X >= 0, Y >= 0`) and the origin-centered
disk of radius `r` (`r² = r^2`).
"""
@inline function quadrant_disk_area2(X::T, Y::T, r::T, r²::T) where {T<:AbstractFloat}
    (iszero(X) | iszero(Y)) && return zero(T)

    X² = X^2
    Y² = Y^2

    # Outer corner is inside the disk -> full rectangle
    X² + Y² <= r² && return 2 * X * Y

    # Endpoints on the circle: p = (x_hi, y_lo) and q = (x_lo, y_hi)
    x_lo = Y < r ? sqrt(r² - Y²) : zero(T)
    y_hi = min(Y, r)

    x_hi = min(X, r)
    y_lo = X < r ? sqrt(r² - X²) : zero(T)

    tri2 = muladd(x_lo, Y, X * y_lo)
    sec2 = sector_area2(x_hi, y_lo, x_lo, y_hi, r²)

    return tri2 + sec2
end

"""
    signed_quadrant_disk_area2(x::T, y::T, r::T, r²::T) where {T<:AbstractFloat} -> T

Signed version of `quadrant_disk_area2` for arbitrary coordinates `(x, y)` in any quadrant:
returns `sign(x) * sign(y) * quadrant_disk_area2(abs(x), abs(y), r, r²)`.
"""
@inline function signed_quadrant_disk_area2(x::T, y::T, r::T, r²::T) where {T<:AbstractFloat}
    area2 = quadrant_disk_area2(abs(x), abs(y), r, r²)
    return flipsign(flipsign(area2, x), y)
end

"""
    disk_fraction(Lx::Real, Ly::Real, r::Real) -> AbstractFloat

Analytic fast path for the fraction of an origin-centered rectangle (or rectangular
prism with horizontal footprint `[-Lx/2, Lx/2] × [-Ly/2, Ly/2]`) that lies within
a cylinder/disk of radius `r` centered at `(0, 0)`.
"""
@inline function disk_fraction(Lx::Real, Ly::Real, r::Real)
    T = float(promote_type(typeof(Lx), typeof(Ly), typeof(r)))
    r_T = T(r)
    iszero(r_T) && return zero(T)

    X = T(Lx) / 2
    Y = T(Ly) / 2
    r² = r_T^2

    # Outer corner inside disk
    muladd(X, X, Y^2) <= r² && return one(T)

    # Disk completely inside box
    if X >= r_T && Y >= r_T
        return clamp((T(π) * r²) / (4 * X * Y), zero(T), one(T))
    end

    area2 = quadrant_disk_area2(X, Y, r_T, r²)
    return clamp(area2 / (2 * X * Y), zero(T), one(T))
end

"""
    disk_fraction(x_min::Real, x_max::Real, y_min::Real, y_max::Real, r::Real) -> AbstractFloat

Analytic fast path for the fraction of an axis-aligned rectangle `[x_min, x_max] × [y_min, y_max]`
(or 3D rectangular prism `[x_min, x_max] × [y_min, y_max] × [z_min, z_max]`) that overlaps
with an origin-centered disk/cylinder of radius `r`.
"""
@inline function disk_fraction(x_min::Real, x_max::Real, y_min::Real, y_max::Real, r::Real)
    T = float(promote_type(typeof(x_min), typeof(x_max), typeof(y_min), typeof(y_max), typeof(r)))
    x1, x2 = T(x_min), T(x_max)
    y1, y2 = T(y_min), T(y_max)
    r_T = T(r)
    iszero(r_T) && return zero(T)

    # Early exit 1: Bounding box strictly outside disk's [-r, r]² box
    if x1 >= r_T || x2 <= -r_T || y1 >= r_T || y2 <= -r_T
        return zero(T)
    end

    r² = r_T^2
    max_x² = max(x1^2, x2^2)
    max_y² = max(y1^2, y2^2)

    # Early exit 2: Farthest corner is inside the disk -> entire box is inside
    max_x² + max_y² <= r² && return one(T)

    # Early exit 3: Nearest point on the box to the origin is outside the disk
    near_x = clamp(zero(T), x1, x2)
    near_y = clamp(zero(T), y1, y2)
    muladd(near_x, near_x, near_y^2) >= r² && return zero(T)

    # Early exit 4: Box completely contains the disk
    box_area = (x2 - x1) * (y2 - y1)
    if x1 <= -r_T && x2 >= r_T && y1 <= -r_T && y2 >= r_T
        return clamp((T(π) * r²) / box_area, zero(T), one(T))
    end

    # General analytic decomposition across the 4 signed quadrants
    inter_area2 = (
        signed_quadrant_disk_area2(x2, y2, r_T, r²) -
        signed_quadrant_disk_area2(x1, y2, r_T, r²) -
        signed_quadrant_disk_area2(x2, y1, r_T, r²) +
        signed_quadrant_disk_area2(x1, y1, r_T, r²)
    )

    return clamp(inter_area2 / (2 * box_area), zero(T), one(T))
end


# ============================================================================ #
# Arbitrary Polygon Functions
# ============================================================================ #

"""
    sector_area2(px, py, qx, qy, r²)

Compute twice the signed area of the circular sector of radius `r` (given as
squared radius `r²`) swept from ray `p = (px, py)` to ray `q = (qx, qy)`
around the origin `(0, 0)`.
"""
@inline sector_area2(px, py, qx, qy, r²) = r² * atan(px * qy - py * qx, px * qx + py * qy)

"""
    edge_disk_area2(ax, ay, bx, by, cross_ab, r²)

Compute twice the signed area swept from the origin `(0, 0)` to the directed
line segment `a -> b` clipped to a disk of squared radius `r²`.
"""
@inline function edge_disk_area2(ax, ay, bx, by, cross_ab, r²)
    ca = muladd(ax, ax, ay^2) - r²
    cb = muladd(bx, bx, by^2) - r²

    # Fast path 1: Both endpoints inside the disk -> pure triangle (0 sqrt, 0 atan)
    if (ca <= 0) & (cb <= 0)
        return cross_ab
    end

    dx = bx - ax
    dy = by - ay
    a = muladd(dx, dx, dy^2)
    iszero(a) && return zero(r²)

    h = muladd(ax, dx, ay * dy)
    Δ4 = muladd(h, h, -a * ca)

    # Fast path 2: Entire segment is outside the disk -> 1 circular sector (0 sqrt, 1 atan)
    if Δ4 <= 0 || (ca > 0 && cb > 0 && (h >= 0 || -h >= a))
        return sector_area2(ax, ay, bx, by, r²)
    end

    # General path: Edge intersects the circle boundary
    s = sqrt(Δ4)
    inv_a = inv(a)
    t1 = (-h - s) * inv_a
    t2 = (-h + s) * inv_a

    t_lo = clamp(t1, zero(r²), one(r²))
    t_hi = clamp(t2, zero(r²), one(r²))

    # Interior segment [t_lo, t_hi] triangle area2 = (t_hi - t_lo) * cross_ab
    area2 = (t_hi - t_lo) * cross_ab

    if t1 > 0
        p1x = muladd(t_lo, dx, ax)
        p1y = muladd(t_lo, dy, ay)
        area2 += sector_area2(ax, ay, p1x, p1y, r²)
    end

    if t2 < 1
        p2x = muladd(t_hi, dx, ax)
        p2y = muladd(t_hi, dy, ay)
        area2 += sector_area2(p2x, p2y, bx, by, r²)
    end

    return area2
end

# The squared distance from the origin to the segment `a -> b`.
@inline function _segment_origin_distance2(ax::T, ay::T, bx::T, by::T) where {T<:AbstractFloat}
    dx, dy = bx - ax, by - ay
    a = muladd(dx, dx, dy^2)
    t = iszero(a) ? zero(T) : clamp(-muladd(ax, dx, ay * dy) / a, zero(T), one(T))
    px, py = muladd(t, dx, ax), muladd(t, dy, ay)
    return muladd(px, px, py^2)
end

"""
    disk_fraction(poly::AbstractVector{NTuple{2,T}}, r::Real) where {T<:AbstractFloat} -> T

Compute the fraction in `[0, 1]` of a simple 2D polygon's area that lies within
a disk of radius `r` centered at the origin `(0, 0)`.
"""
function disk_fraction(poly::AbstractVector{NTuple{2,T}}, r::Real) where {T<:AbstractFloat}
    n = length(poly)
    r_T = T(r)
    iszero(r_T) && return zero(T)
    r² = r_T^2

    # Single-pass bounding-box, max-radius, nearest-edge, crossing-number and Shoelace area pre-check
    ax, ay = @inbounds poly[n]
    min_x = max_x = ax
    min_y = max_y = ay
    max_r² = muladd(ax, ax, ay^2)
    near_r² = max_r²
    inside = false
    polygon_area2 = zero(T)

    @inbounds for i in 1:n
        bx, by = poly[i]
        min_x = min(min_x, bx)
        max_x = max(max_x, bx)
        min_y = min(min_y, by)
        max_y = max(max_y, by)
        max_r² = max(max_r², muladd(bx, bx, by^2))
        near_r² = min(near_r², _segment_origin_distance2(ax, ay, bx, by))
        # The edge crosses the ray from the origin along +x.
        ((ay > 0) != (by > 0)) && (ax + (bx - ax) * (-ay / (by - ay)) > 0) && (inside = !inside)
        polygon_area2 += ax * by - ay * bx
        ax, ay = bx, by
    end

    # All vertices inside the disk -> entire polygon is inside the disk
    max_r² <= r² && return one(T)

    # Polygon bounding box is strictly outside the disk's [-r, r]^2 bounding box
    if min_x >= r_T || max_x <= -r_T || min_y >= r_T || max_y <= -r_T
        return zero(T)
    end

    # The boundary's nearest point lies on or beyond the circle -> the disk is wholly outside the polygon, or
    # wholly inside it when the polygon encloses the origin
    if near_r² >= r²
        inside || return zero(T)
        return clamp(2 * T(π) * r² / abs(polygon_area2), zero(T), one(T))
    end

    # Full boundary integration for partial overlap
    ax, ay = @inbounds poly[n]
    intersection_area2 = zero(T)

    @inbounds for i in 1:n
        bx, by = poly[i]
        cross_ab = ax * by - ay * bx
        intersection_area2 += edge_disk_area2(ax, ay, bx, by, cross_ab, r²)
        ax, ay = bx, by
    end

    return clamp(abs(intersection_area2) / abs(polygon_area2), zero(T), one(T))
end

"""
    interval_fraction(a, b, r) -> T

The fraction of the segment `[a, b]` (`a < b`) inside `[-r, r]`.
"""
@inline interval_fraction(a::T, b::T, r::T) where {T<:AbstractFloat} =
    clamp((min(b, r) - max(a, -r)) / (b - a), zero(T), one(T))


# ============================================================================ #
# Piecewise integrals with square-root ends
# ============================================================================ #

# `∫ f` over `[p, q]` by Gauss–Legendre `nodes = (x, w)` on `[−1, 1]` under `t = p + (q − p) g(u)`,
# `g(u) = 3u² − 2u³`, whose vanishing slope at both ends turns a `√` end into an analytic one. `f` takes the
# offset `t − p`.
function _smooth_piece(f::F, p::T, q::T, nodes) where {F, T<:AbstractFloat}
    x, wt = nodes
    h = q - p
    s = zero(T)
    @inbounds for k in eachindex(x)
        u = (1 + T(x[k])) / 2
        s += T(wt[k]) * 6u * (1 - u) * f(h * (u * u * (3 - 2u)))
    end
    return s * h / 2
end

# `_smooth_piece` over `[p, q]`, cut toward each square-root branch point in `branch_points` lying outside the
# piece within its length: the sub-piece next to the branch point is as long as its distance from it, and the
# rest is cut the same way, so every sub-piece lies at least its own length from the singularity. A branch point
# within a few rounding units of an end counts as at that end, which the substitution makes analytic. `f` takes
# the absolute coordinate split as `(p, t − p)`.
function _graded_piece(f::F, p::T, q::T, nodes, branch_points) where {F, T<:AbstractFloat}
    h = q - p
    δp = minimum(s -> s < p ? p - s : T(Inf), branch_points; init = T(Inf))
    δq = minimum(s -> s > q ? s - q : T(Inf), branch_points; init = T(Inf))
    at_end = 64 * eps(T) * max(abs(p), abs(q), h)
    near_p = at_end < δp < h
    near_q = at_end < δq < h
    if near_p && near_q
        m = (p + q) / 2
        return _graded_piece(f, p, m, nodes, branch_points) + _graded_piece(f, m, q, nodes, branch_points)
    elseif near_p
        return _smooth_piece(t -> f(p, t), p, p + δp, nodes) + _graded_piece(f, p + δp, q, nodes, branch_points)
    elseif near_q
        return _graded_piece(f, p, q - δq, nodes, branch_points) +
               _smooth_piece(t -> f(q - δq, t), q - δq, q, nodes)
    end
    return _smooth_piece(t -> f(p, t), p, q, nodes)
end

# `∫ f` over `[lo, hi]`, piece by piece between the `cuts` inside it (a tuple in any order; `NaN` entries are
# skipped), each piece by `_graded_piece`.
function _piecewise(f::F, lo::T, hi::T, nodes, cuts, branch_points) where {F, T<:AbstractFloat}
    s = zero(T)
    p = lo
    while p < hi
        q = hi
        for c in cuts
            p < c < q && (q = c)
        end
        s += _graded_piece(f, p, q, nodes, branch_points)
        p = q
    end
    return s
end

# `∫ f` over `[lo, hi]` for an outer integral whose integrand runs `_piecewise`: the same pieces between the `cuts`,
# each graded toward the `branch_points` outside it within its length as `_graded_piece` grades, by methods of its
# own, since inference follows a nested integral only through methods it is not already inside. `f` takes
# `(p, t − p)`.
function _outer_piecewise(f::F, lo::T, hi::T, nodes, cuts, branch_points) where {F, T<:AbstractFloat}
    s = zero(T)
    p = lo
    while p < hi
        q = hi
        for c in cuts
            p < c < q && (q = c)
        end
        if _near_cut(_gap_below(branch_points, p), p, q) && _near_cut(_gap_above(branch_points, q), p, q)
            m = (p + q) / 2
            s += _outer_side(f, p, m, nodes, branch_points) + _outer_side(f, m, q, nodes, branch_points)
        else
            s += _outer_side(f, p, q, nodes, branch_points)
        end
        p = q
    end
    return s
end

@inline _gap_below(cuts, x::T) where {T} = minimum(c -> c < x ? x - c : T(Inf), cuts; init = T(Inf))
@inline _gap_above(cuts, x::T) where {T} = minimum(c -> c > x ? c - x : T(Inf), cuts; init = T(Inf))
@inline _near_cut(δ::T, p::T, q::T) where {T} = 64 * eps(T) * max(abs(p), abs(q), q - p) < δ < q - p

# A piece with a cut within its length on at most one side, its sub-pieces as long as their distance from that cut.
function _outer_side(f::F, p::T, q::T, nodes, cuts) where {F, T<:AbstractFloat}
    s = zero(T)
    while true
        δp, δq = _gap_below(cuts, p), _gap_above(cuts, q)
        if _near_cut(δp, p, q)
            s += _outer_smooth(f, p, p + δp, nodes)
            p += δp
        elseif _near_cut(δq, p, q)
            s += _outer_smooth(f, q - δq, q, nodes)
            q -= δq
        else
            return s + _outer_smooth(f, p, q, nodes)
        end
    end
end

# `∫ f` over `[p, q]` as `_smooth_piece` takes it.
function _outer_smooth(f::F, p::T, q::T, nodes) where {F, T<:AbstractFloat}
    x, wt = nodes
    h = q - p
    s = zero(T)
    @inbounds for k in eachindex(x)
        u = (1 + T(x[k])) / 2
        s += T(wt[k]) * 6u * (1 - u) * f(p, h * (u * u * (3 - 2u)))
    end
    return s * h / 2
end


# ============================================================================ #
# Ball ∩ box
# ============================================================================ #

"""
    ball_box_fraction(x1, x2, y1, y2, z1, z2, r, nodes) -> T

The fraction of the box `[x1, x2] × [y1, y2] × [z1, z2]` inside the origin-centred ball of radius `r`: the
integral over `z` of each slice's rectangle ∩ disk of radius `√(r² − z²)` ([`disk_fraction`](@ref)), by the
Gauss–Legendre `nodes = (x, w)` on the pieces between the heights where that radius passes an edge line's
distance (its foot on the edge) or a corner's, each under a smoothstep substitution. `1` and `0` exactly for a
box wholly inside or outside.
"""
function ball_box_fraction(x1::T, x2::T, y1::T, y2::T, z1::T, z2::T, r::T, nodes) where {T<:AbstractFloat}
    r² = r^2
    max(x1^2, x2^2) + max(y1^2, y2^2) + max(z1^2, z2^2) <= r² && return one(T)
    clamp(zero(T), x1, x2)^2 + clamp(zero(T), y1, y2)^2 + clamp(zero(T), z1, z2)^2 >= r² && return zero(T)
    # The heights where the slice radius equals an edge line's distance (its foot on the edge) or a corner's;
    # `NaN` where the ball does not reach it.
    height(c) = c < r ? sqrt((r - c) * (r + c)) : T(NaN)
    fx, fy = y1 <= 0 <= y2, x1 <= 0 <= x2
    zs = (height(fx ? abs(x1) : T(Inf)), height(fx ? abs(x2) : T(Inf)),
          height(fy ? abs(y1) : T(Inf)), height(fy ? abs(y2) : T(Inf)),
          height(hypot(x1, y1)), height(hypot(x1, y2)), height(hypot(x2, y1)), height(hypot(x2, y2)))
    branch_points = (-r, r, zs..., map(-, zs)...)
    area = (x2 - x1) * (y2 - y1)
    slice(p, t) = (z = p + t; ρ² = (r - z) * (r + z); ρ² <= 0 ? zero(T) : disk_fraction(x1, x2, y1, y2, sqrt(ρ²)) * area)
    v = _piecewise(slice, z1, z2, nodes, branch_points, branch_points)
    return clamp(v / (area * (z2 - z1)), zero(T), one(T))
end


# ============================================================================ #
# Spherical cap ∩ latitude–longitude cell
# ============================================================================ #

"""
    cap_half_width(φ, β, θ) -> T

The half-width in longitude of the spherical cap of angular radius `θ` about latitude `β`, longitude `0`, on
the circle of latitude `φ`: the cap holds `(λ, φ)` for `|λ| ≤ w`, `w = 0` where the circle misses it and
`w = π` where it holds the whole circle. From `hav d = hav(φ − β) + cos φ cos β hav λ`, `sin²(w/2)` and
`cos²(w/2)` are `sin((θ − Δ)/2) sin((θ + Δ)/2)` and `sin((π − Σ − θ)/2) sin((π − Σ + θ)/2)` over their common
factor `cos φ cos β`, `Δ = φ − β`, `Σ = φ + β`.
"""
@inline cap_half_width(φ::T, β::T, θ::T) where {T<:AbstractFloat} = _cap_half_width(φ - β, φ + β, θ)

# The half-width from `Δ = φ − β` and `Σ = φ + β`, which a caller may form more accurately than from `φ`.
@inline function _cap_half_width(Δ::T, Σ::T, θ::T) where {T<:AbstractFloat}
    s = sin((θ - Δ) / 2) * sin((θ + Δ) / 2)
    c = sin((T(π) - Σ - θ) / 2) * sin((T(π) - Σ + θ) / 2)
    s <= 0 && return zero(T)
    c <= 0 && return T(π)
    return 2 * atan(sqrt(s), sqrt(c))
end

"""
    meridian_crossings(λ, β, θ) -> (s₁, s₂)

Where the boundary of the cap of angular radius `θ` about latitude `β`, longitude `0`, meets the great circle
through the meridian of longitude `λ`, as angles along that circle from its equator crossing at `λ`, north
positive; an angle in `[−π/2, π/2]` is a latitude on the meridian itself. `NaN`s where they do not meet. The
circle passes `δ` from the centre, `sin δ = cos β |sin λ|`, nearest it at `s* = atan(sin β, cos β cos λ)`, and
meets the boundary at `s* ± η`, `tan²(η/2) = sin((θ−δ)/2) sin((θ+δ)/2) / (cos((θ−δ)/2) cos((θ+δ)/2))`.
"""
@inline function meridian_crossings(λ::T, β::T, θ::T) where {T<:AbstractFloat}
    sβ, cβ = sincos(β)
    sλ, cλ = sincos(λ)
    δ = atan(cβ * abs(sλ), sqrt(sβ^2 + (cβ * cλ)^2))
    (δ >= θ || θ + δ >= T(π)) && return (T(NaN), T(NaN))
    η = 2 * atan(sqrt(sin((θ - δ) / 2) * sin((θ + δ) / 2)), sqrt(cos((θ - δ) / 2) * cos((θ + δ) / 2)))
    s = atan(sβ, cβ * cλ)
    P = 2 * T(π)
    return (rem(s - η, P, RoundNearest), rem(s + η, P, RoundNearest))
end

# `|[a, b] ∩ ⋃ₖ [2πk − w, 2πk + w]|`, the longitudes of `[a, b]` inside a half-width `w`, for a cell of width
# at most `2π` centred in `(−π, π]`.
@inline function _lon_overlap(w::T, a::T, b::T) where {T<:AbstractFloat}
    P = 2 * T(π)
    o = zero(T)
    for k in -1:1
        o += max(zero(T), min(b, w + k * P) - max(a, -w + k * P))
    end
    return o
end

# `sin²(d/2)` of the angular distance between `(λ, φ)` and the cap's centre `(0, β)`.
@inline _hav(λ::T, φ::T, β::T) where {T<:AbstractFloat} = sin((φ - β) / 2)^2 + cos(φ) * cos(β) * sin(λ / 2)^2

# The latitudes of `[φa, φb]` where `sin²(d/2)` along a meridian can take its extremes: the ends, and the
# point of the meridian's great circle nearest the centre and its antipode, where they fall inside.
@inline function _hav_range(λ::T, φa::T, φb::T, β::T) where {T<:AbstractFloat}
    lo = min(_hav(λ, φa, β), _hav(λ, φb, β))
    hi = max(_hav(λ, φa, β), _hav(λ, φb, β))
    s = atan(sin(β), cos(β) * cos(λ))
    for c in (s, s - T(π), s + T(π))
        if φa < c < φb
            h = _hav(λ, c, β)
            lo = min(lo, h)
            hi = max(hi, h)
        end
    end
    return lo, hi
end

# `cos φ · L(φ)` at `φ = p + t`, `L` the cell's longitudes `[a, b]` inside the cap's half-width. The offset
# `φ − β` is carried as `(p − β) + t`, so a small cell keeps its digits.
@inline function _cap_slice(p::T, t::T, a::T, b::T, β::T, θ::T) where {T<:AbstractFloat}
    Δ = (p - β) + t
    return cos(β + Δ) * _lon_overlap(_cap_half_width(Δ, 2β + Δ, θ), a, b)
end

"""
    cap_cell_fraction(λa, λb, φa, φb, β, θ, nodes) -> T

The fraction of the latitude–longitude cell `[λa, λb] × [φa, φb]` inside the spherical cap of angular radius
`θ` about latitude `β`, longitude `0`: longitudes relative to the cap's centre, `0 < λb − λa ≤ 2π`, latitudes in
`[−π/2, π/2]`, `φa < φb`. `1` or `0` exactly when every point of the cell is inside or outside. Otherwise the
overlap `∫ cos φ L(φ) dφ`, `L` the cell's longitudes within [`cap_half_width`](@ref), summed between the
latitudes where `L` changes form (the cap's extreme latitudes, where it holds a whole circle, and where its
boundary crosses a face meridian), each piece by the Gauss–Legendre `nodes = (x, w)` on `[−1, 1]` under a
smoothstep substitution.
"""
function cap_cell_fraction(λa::T, λb::T, φa::T, φb::T, β::T, θ::T, nodes) where {T<:AbstractFloat}
    θ >= T(π) && return one(T)
    P = 2 * T(π)
    k = round((λa + λb) / (2P))
    a = λa - k * P
    b = λb - k * P
    hθ = sin(θ / 2)^2
    # `sin²(λ/2)` is least at the cell's longitude nearest `0` and greatest at the one nearest `±π`; for
    # either, the latitude extremes of `_hav_range` finish the search.
    far = (a <= -T(π) || b >= T(π)) ? T(π) : (sin(a / 2)^2 >= sin(b / 2)^2 ? a : b)
    last(_hav_range(far, φa, φb, β)) <= hθ && return one(T)
    near = (a <= 0 <= b) ? zero(T) : (sin(a / 2)^2 <= sin(b / 2)^2 ? a : b)
    first(_hav_range(near, φa, φb, β)) >= hθ && return zero(T)

    # The half-width's square-root branch points: the cap's extreme latitudes, and where it holds a whole circle.
    branch_points = (β - θ, β + θ, T(π) - θ - β, θ - T(π) - β)
    # Where the boundary crosses a face meridian, `NaN` where it does not.
    crossing(f) = (fw = abs(f - round(f / P) * P); zero(T) < fw < T(π) ? meridian_crossings(fw, β, θ) : (T(NaN), T(NaN)))
    cuts = (branch_points..., crossing(a)..., crossing(b)...)
    area = _piecewise((p0, t) -> _cap_slice(p0, t, a, b, β, θ), φa, φb, nodes, cuts, branch_points)
    cell = (b - a) * 2 * cos((φa + φb) / 2) * sin((φb - φa) / 2)
    return clamp(area / cell, zero(T), one(T))
end

# ============================================================================ #
# Ball ∩ spherical shell sector
# ============================================================================ #

# The angular distance between `(λ, φ)` and `(0, β)`.
@inline _angle_to(λ::T, φ::T, β::T) where {T<:AbstractFloat} = 2 * asin(sqrt(clamp(_hav(λ, φ, β), zero(T), one(T))))

# The angular distances from `(0, β)` at which the cell `[a, b] × [φa, φb]` can meet a cap's boundary first or
# last, `NaN` where absent: its corners; along each meridian face the points of its great circle nearest and
# farthest from the centre; along each latitude face the longitudes `0` and `π`; `0` and `π` where the centre or
# its antipode lies inside. The nearest and farthest of them bound the cell's distances.
function _cell_critical_angles(a::T, b::T, φa::T, φb::T, β::T) where {T<:AbstractFloat}
    P = 2 * T(π)
    meridian(λ) = (s = atan(sin(β), cos(β) * cos(λ));
                   map(c -> φa < c < φb ? _angle_to(λ, c, β) : T(NaN), (s, s - T(π), s + T(π))))
    parallel(λ0) = any(k -> a < λ0 + k * P < b, -1:1)
    lat(λ0) = parallel(λ0) ? (_angle_to(λ0, φa, β), _angle_to(λ0, φb, β)) : (T(NaN), T(NaN))
    centre = (parallel(zero(T)) || a == 0 || b == 0) && φa <= β <= φb ? zero(T) : T(NaN)
    antipode = parallel(T(π)) && φa <= -β <= φb ? T(π) : T(NaN)
    return (_angle_to(a, φa, β), _angle_to(a, φb, β), _angle_to(b, φa, β), _angle_to(b, φb, β),
            meridian(a)..., meridian(b)..., lat(zero(T))..., lat(T(π))..., centre, antipode)
end

# The squared chord from `(0, β, r0)` to `(·, ·, r)` at angular separation `θ`, from `r − r0` and `sin²(θ/2)`.
@inline _chord2(r::T, r0::T, θ::T) where {T<:AbstractFloat} = (r - r0 + 2 * r0 * sin(θ / 2)^2)^2 + (r0 * sin(θ))^2

"""
    shell_cell_fraction(λa, λb, φa, φb, ra, rb, β, r0, R, nodes) -> T

The fraction of the spherical shell sector `[λa, λb] × [φa, φb] × [ra, rb]` inside the ball of radius `R` about the
point at latitude `β`, longitude `0`, radius `r0`: longitudes relative to that point's, latitudes in
`[−π/2, π/2]`, radii `0 ≤ ra < rb`. On the sphere of radius `r` the ball is the cap of angular radius `θ(r)`,
`sin²(θ/2) = (R − (r − r0))(R + (r − r0)) / (4 r r0)`, so the fraction is `∫ r² f(θ(r)) dr / ((rb³ − ra³)/3)` with `f`
the cell's fraction in that cap ([`cap_cell_fraction`](@ref)), by the Gauss–Legendre `nodes` on the pieces between
the radii where `θ(r)` reaches `0` or `π` or passes an angle where `f` changes form, graded toward each. `1` and `0`
exactly for a sector wholly inside or outside.
"""
function shell_cell_fraction(λa::T, λb::T, φa::T, φb::T, ra::T, rb::T, β::T, r0::T, R::T, nodes) where {T<:AbstractFloat}
    P = 2 * T(π)
    k = round((λa + λb) / (2P))
    a = λa - k * P
    b = λb - k * P
    crit = _cell_critical_angles(a, b, φa, φb, β)
    θn = minimum(c -> isnan(c) ? T(Inf) : c, crit)
    θf = maximum(c -> isnan(c) ? -T(Inf) : c, crit)
    R2 = R^2
    max(_chord2(ra, r0, θf), _chord2(rb, r0, θf)) <= R2 && return one(T)
    _chord2(clamp(r0 * cos(θn), ra, rb), r0, θn) >= R2 && return zero(T)
    lo, hi = max(ra, r0 - R), min(rb, r0 + R)
    lo < hi || return zero(T)
    # Where the sphere of radius `r` meets the cone at angle `c` about the point's direction; complex, at
    # `r0 cos c ± i√(−d)`, where the cone misses the ball.
    disc(c) = (R - r0 * sin(c)) * (R + r0 * sin(c))
    root(c, s) = (d = disc(c); d < 0 ? T(NaN) : r0 * cos(c) + s * sqrt(d))
    cuts = (r0 - R, r0 + R, R - r0, map(c -> root(c, 1), crit)..., map(c -> root(c, -1), crit)...)
    function slice(p, t)
        r = p + t
        Δ = (p - r0) + t
        s = (R - Δ) * (R + Δ) / (4 * r * r0)
        θ = s <= 0 ? zero(T) : s >= 1 ? T(π) : 2 * asin(sqrt(s))
        return r^2 * cap_cell_fraction(a, b, φa, φb, β, θ, nodes)
    end
    # The integrand is analytic but at the breakpoints and the complex pairs: four nodes resolve a range whose
    # nearest one lies eight lengths away to `ρ⁻⁸`, `ρ > 30` the Bernstein ellipse's parameter.
    out(x) = x < lo ? lo - x : x > hi ? x - hi : zero(T)
    near(c) = isnan(c) ? T(Inf) : (d = disc(c); d >= 0 ? min(out(r0 * cos(c) + sqrt(d)), out(r0 * cos(c) - sqrt(d))) :
                                              hypot(out(r0 * cos(c)), sqrt(-d)))
    gap = min(out(r0 - R), out(r0 + R), out(R - r0), minimum(near, crit))
    v = gap >= 8 * (hi - lo) ? _gauss4(r -> slice(lo, r - lo), lo, hi) : _outer_piecewise(slice, lo, hi, nodes, cuts, cuts)
    return clamp(3 * v / ((rb - ra) * (ra^2 + ra * rb + rb^2)), zero(T), one(T))
end

# `∫ f` over `[p, q]` by four-point Gauss–Legendre.
@inline function _gauss4(f::F, p::T, q::T) where {F, T<:AbstractFloat}
    s = sqrt(T(6) / 5)
    x1, x2 = sqrt(T(3) / 7 - 2s / 7), sqrt(T(3) / 7 + 2s / 7)
    w1, w2 = (18 + sqrt(T(30))) / 36, (18 - sqrt(T(30))) / 36
    m, h = (p + q) / 2, (q - p) / 2
    return h * (w1 * (f(m - h * x1) + f(m + h * x1)) + w2 * (f(m - h * x2) + f(m + h * x2)))
end

# ============================================================================ #
# Spherical cap ∩ spherical polygon with great-circle edges
# ============================================================================ #
#
# In polar coordinates `(ρ, α)` about the cap's centre `C = (0, 0, 1)` the area element is `sin ρ dρ dα`, so a
# region's area is `∮ (1 − cos ρ) dα` over its boundary, the spherical form of the triangle fan of
# `edge_disk_area2`. Along the great circle whose nearest point to `C` lies `ρ₀` away at azimuth `α₀`,
# `tan ρ = tan ρ₀ / cos ψ` with `ψ = α − α₀`, and `1 − cos ρ` integrates to
# `F(ψ) = ψ − asin(sin ψ / √(1 + tan² ρ₀))`; the parts of an edge beyond the cap contribute `(1 − cos θ) Δα`.

# `F(ψ) = ψ − atan(y, x₂)` for a great circle of pole `n`, `h = √(n₁² + n₂²)`, `v = |n₃|` (`tan ρ₀ = v / h`),
# `y = h sin ψ`, `x₁ = h cos ψ`, `x₂ = √(x₁² + v²)`. Where `x₁ ≥ 0`, `ψ = atan(y, x₁)` and the difference is the
# one angle `atan(y (x₂ − x₁), x₁ x₂ + y²)`, `x₂ − x₁ = v² / (x₂ + x₁)`, so a value of order `ρ₀²` keeps its
# digits; elsewhere `F` is continuous through `ψ = ±π` as written.
@inline function _gc_area(ψ::T, h::T, v::T) where {T<:AbstractFloat}
    s, c = sincos(ψ)
    y, x1 = h * s, h * c
    x2 = sqrt(x1^2 + v^2)
    x1 >= 0 || return ψ - atan(y, x2)
    return atan(y * (v^2 / (x2 + x1)), x1 * x2 + y^2)
end

# `∮ (1 − cos min(ρ, θ)) dα` along the great-circle edge from unit vector `a` to `b`.
function _edge_cap_area(a::NTuple{3,T}, b::NTuple{3,T}, θ::T) where {T<:AbstractFloat}
    n = (a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1])
    h = sqrt(n[1]^2 + n[2]^2)
    v = abs(n[3])
    # A great circle through the centre sweeps no azimuth off it, and `1 − cos ρ` vanishes on it.
    iszero(v) && return zero(T)
    Δα = atan(a[1] * b[2] - a[2] * b[1], a[1] * b[1] + a[2] * b[2])
    α0 = atan(-n[3] * n[2], -n[3] * n[1])
    P = 2 * T(π)
    ψa = rem(atan(a[2], a[1]) - α0, P, RoundNearest)
    ψb = ψa + Δα
    lo, hi = minmax(ψa, ψb)
    # The edge's points inside the cap are those with `cos ψ ≥ tan ρ₀ cot θ`, `|ψ| ≤ ψθ` (mod 2π).
    sθ, cθ = sincos(θ)
    vers = 2 * sin(θ / 2)^2   # 1 − cos θ
    d = (h * sθ - v * cθ) * (h * sθ + v * cθ)
    ψθ = d >= 0 ? atan(sqrt(d), v * cθ) : (cθ > 0 ? zero(T) : T(π))
    total = vers * (hi - lo)
    for k in -1:1
        c = k * P
        l = max(lo, c - ψθ)
        u = min(hi, c + ψθ)
        u > l || continue
        total += (_gc_area(u - c, h, v) - _gc_area(l - c, h, v)) - vers * (u - l)
    end
    return ψb >= ψa ? total : -total
end

# The polygon's area within the cap about `C`, oriented as its vertices run.
@inline function _polygon_cap_area(V, θ::T) where {T<:AbstractFloat}
    s = zero(T)
    n = length(V)
    for i in 1:n
        s += _edge_cap_area(V[i], V[mod1(i + 1, n)], θ)
    end
    return s
end

"""
    cap_polygon_fraction(V, θ) -> T

The fraction of the spherical polygon whose great-circle edges join the unit vectors `V` in turn inside the cap
of angular radius `θ` about `(0, 0, 1)`; the polygon spans less than a hemisphere. `1` exactly when every
vertex is inside. A polygon centred on the far hemisphere is one less its fraction inside the complementary cap
about `(0, 0, −1)`, so the point the formula is singular at never lies inside the polygon.
"""
function cap_polygon_fraction(V, θ::T) where {T<:AbstractFloat}
    θ >= T(π) && return one(T)
    hθ = sin(θ / 2)^2
    # `sin²(ρ/2) = (1 − z)/2 = (x² + y²) / (2(1 + z))`.
    all(v -> v[3] > 0 && (v[1]^2 + v[2]^2) / (2 * (1 + v[3])) <= hθ, V) && return one(T)
    cz = sum(v -> v[3], V)
    if cz < 0
        W = map(v -> (v[1], -v[2], -v[3]), V)
        return one(T) - cap_polygon_fraction(W, T(π) - θ)
    end
    whole = _polygon_cap_area(V, T(π))
    return clamp(_polygon_cap_area(V, θ) / whole, zero(T), one(T))
end

"""
    cap_arc_fraction(λa, λb, φ, β, θ) -> T

The fraction of the arc `[λa, λb]` of the circle of latitude `φ` inside the cap of angular radius `θ` about
latitude `β`, longitude `0`: a cell of a grid with one latitude.
"""
@inline function cap_arc_fraction(λa::T, λb::T, φ::T, β::T, θ::T) where {T<:AbstractFloat}
    θ >= T(π) && return one(T)
    P = 2 * T(π)
    k = round((λa + λb) / (2P))
    return clamp(_lon_overlap(cap_half_width(φ, β, θ), λa - k * P, λb - k * P) / (λb - λa), zero(T), one(T))
end

# ============================================================================ #
# Geodesic disk ∩ latitude–longitude cell on a surface of revolution
# ============================================================================ #
#
# On a surface of revolution a geodesic off a meridian moves monotonically in longitude (Clairaut's relation), so
# the distance from a point grows with the longitude offset along each parallel, and the disk of geodesic radius
# `R` about latitude `t`, longitude `0` holds `(λ, φ)` for `|λ| ≤ w(φ)`. Its edge is the point at distance `R`
# along each azimuth `α`, the direct geodesic problem; the east half, `α` from `0` to `π`, falls in latitude
# from `B` to `A`, past a covered pole onto the far meridian. On a sphere of angular radius `θ`,
# `h = sin²(w/2) cos φ = (cos(φ − t) − cos θ)/(2 cos t)`, a shifted cosine; a spheroid perturbs it by its
# flattening, so one Chebyshev series in `φ` on `[A, B]` holds `h`, interpolated through edge points whose
# azimuths place them at the Chebyshev latitudes of the sphere sharing the disk's extreme latitudes.

const _DISK_TERMS = 25

"""
    GeodesicDisk{T,M,N}

The disk of geodesic radius `R` about latitude `t`, longitude `0`, on a surface of revolution whose lengths per
unit latitude and longitude are `metric(φ) = (M, N cos φ)`: its latitude `span` where `w > 0`, the latitudes
`open = (A, B)` between which `0 < w < π` and `w` at them (`ends`: `0`, or `π` across a covered pole), the latitude
`turn` of `w`'s one extreme between them (an end where `w` is monotone), `whole` where it holds every point, and
the `N`-term Chebyshev series on `[A, B]` of `h = sin²(w/2) cos φ` and of its first two derivatives in the
series' variable.
"""
struct GeodesicDisk{T<:AbstractFloat, M, N}
    metric::M
    R::T
    t::T
    span::NTuple{2,T}
    open::NTuple{2,T}
    ends::NTuple{2,T}
    turn::T
    whole::Bool
    coef::NTuple{N,T}
    dcoef::NTuple{N,T}
    ddcoef::NTuple{N,T}
end

# The root of `g` in `[lo, hi]`, `g` returning its value and derivative and `slo` the sign bit of its value at
# `lo` (the opposite at `hi`): Newton's method from `x` on a polynomial series, a step leaving the bracket
# bisecting it, at most `steps` steps.
function _polish_root(g::G, lo::T, hi::T, slo::Bool, x::T, steps::Int) where {G, T<:AbstractFloat}
    (lo < x < hi) || (x = (lo + hi) / 2)
    for _ in 1:steps
        v, dv = g(x)
        iszero(v) && return x
        signbit(v) == slo ? (lo = x) : (hi = x)
        xn = x - v / dv
        abs(xn - x) <= 4 * eps(T) * max(abs(x), floatmin(T)) && return xn
        (lo < xn < hi) || (xn = (lo + hi) / 2)
        x = xn
    end
    return x
end

# The `n` Chebyshev coefficients of the samples `v[k+1] = f(cos(πk/(n − 1)))`, `k = 0, …, n − 1`.
function _cheb_coefficients(v::Tuple{T,Vararg{T,M}}) where {T<:AbstractFloat, M}
    N = M + 1
    m = M
    return ntuple(Val(N)) do i
        j = i - 1
        s = (v[1] + (iseven(j) ? v[N] : -v[N])) / 2
        for k in 1:(m - 1)
            s += v[k + 1] * cospi(T(j * k) / m)
        end
        (j == 0 || j == m ? s : 2s) / m
    end
end

# The coefficients of the derivative of the Chebyshev series `c`: `c′ₖ₋₁ = c′ₖ₊₁ + 2k cₖ` downward, `c′₀` halved.
function _cheb_derivative(c::Tuple{T,Vararg{T,M}}) where {T<:AbstractFloat, M}
    N = M + 1
    d = zeros(T, N)
    for k in (N - 1):-1:1
        d[k] = (k + 1 < N ? d[k + 2] : zero(T)) + 2k * c[k + 1]
    end
    d[1] /= 2
    return ntuple(i -> d[i], Val(N))
end

@inline function _clenshaw(coef::NTuple{N,T}, x::T) where {N, T<:AbstractFloat}
    b1 = b2 = zero(T)
    @inbounds for j in N:-1:2
        b1, b2 = 2x * b1 - b2 + coef[j], b1
    end
    return x * b1 - b2 + @inbounds(coef[1])
end

# `h` at `φ` and its first `k` derivatives in `φ`.
@inline _disk_x(d::GeodesicDisk{T}, φ::T) where {T} = (2φ - d.open[1] - d.open[2]) / (d.open[2] - d.open[1])
@inline _disk_h(d::GeodesicDisk{T}, φ::T) where {T} = _clenshaw(d.coef, _disk_x(d, φ))
@inline function _disk_h1(d::GeodesicDisk{T}, φ::T) where {T}
    x, s = _disk_x(d, φ), 2 / (d.open[2] - d.open[1])
    return _clenshaw(d.coef, x), s * _clenshaw(d.dcoef, x)
end
@inline function _disk_h2(d::GeodesicDisk{T}, φ::T) where {T}
    x, s = _disk_x(d, φ), 2 / (d.open[2] - d.open[1])
    return _clenshaw(d.coef, x), s * _clenshaw(d.dcoef, x), s^2 * _clenshaw(d.ddcoef, x)
end

"""
    geodesic_disk(direct, metric, R, t, whole) -> GeodesicDisk

The disk of geodesic radius `R` about latitude `t`, longitude `0`: `direct(α)` returns the point `(λ, φ)` at
distance `R` along the azimuth `α` (east of north), its longitude the one swept, so `±π` past a pole, and `R` lies
short of every cut locus; `metric(φ)` the lengths `(M, N cos φ)` per unit latitude and longitude; `whole` where the
disk holds every point. `h` is interpolated through `N` edge points, barycentrically, and re-expanded at the
Chebyshev points of `[A, B]`; the latitude of `w`'s extreme is the root of `(h/cos φ)′`, from the sphere's.
"""
function geodesic_disk(direct::D, metric::M, R::T, t::T, whole::Bool, ::Val{N} = Val(_DISK_TERMS)) where {D, M, T<:AbstractFloat, N}
    hp = T(π) / 2
    λB, B = direct(zero(T))
    λA, A = direct(T(π))
    north, south = abs(λB) > hp, abs(λA) > hp
    span = (south ? -hp : A, north ? hp : B)
    ends = (south ? T(π) : zero(T), north ? T(π) : zero(T))
    zeroes = ntuple(_ -> zero(T), Val(N))
    (whole || B - A <= 4 * eps(T) * hp) &&
        return GeodesicDisk(metric, R, t, span, (A, max(A, B)), ends, A, whole, zeroes, zeroes, zeroes)
    # The sphere whose edge reaches the same extreme latitudes: centre `β`, angular radius `θ`, its edge at latitude
    # `y` along the azimuth with `cos α = (sin y − sin β cos θ)/(cos β sin θ)`.
    P = north ? T(π) - B : B
    Q = south ? -T(π) - A : A
    β, θ = (P + Q) / 2, (P - Q) / 2
    sβ, cβ = sincos(β)
    sθ, cθ = sincos(θ)
    m, c = (A + B) / 2, (B - A) / 2
    edge = ntuple(Val(N)) do k
        y = m + c * cospi(T(k - 1) / (N - 1))
        α = k == 1 ? zero(T) : k == N ? T(π) : acos(clamp((sin(y) - sβ * cθ) / (cβ * sθ), -one(T), one(T)))
        λ, φ = k == 1 ? (λB, B) : k == N ? (λA, A) : direct(α)
        (k == 1 ? one(T) : k == N ? -one(T) : clamp((φ - m) / c, -one(T), one(T)), sin(min(abs(λ), T(π)) / 2)^2 * cos(φ))
    end
    x = map(first, edge)
    hv = map(last, edge)
    wt = ntuple(k -> inv(prod(j -> j == k ? one(T) : x[k] - x[j], 1:N)), Val(N))
    samples = ntuple(Val(N)) do k
        y = cospi(T(k - 1) / (N - 1))
        i = findfirst(==(y), x)
        i === nothing || return hv[i]
        num = den = zero(T)
        for j in 1:N
            r = wt[j] / (y - x[j])
            num += r * hv[j]
            den += r
        end
        num / den
    end
    coef = _cheb_coefficients(samples)
    dcoef = _cheb_derivative(coef)
    ddcoef = _cheb_derivative(dcoef)
    d = GeodesicDisk(metric, R, t, span, (A, B), ends, A, false, coef, dcoef, ddcoef)
    # `w`'s extreme where `(h/cos φ)′ = (h′ cos φ + h sin φ)/cos²φ` vanishes; with both ends `0` (or both `π`) it lies
    # between them, near the sphere's, where `sin φ = sin β / cos θ`.
    ends[1] == ends[2] || return GeodesicDisk(metric, R, t, span, (A, B), ends, ends[1] > ends[2] ? A : B, false,
                                              coef, dcoef, ddcoef)
    g(φ) = ((v, v1, v2) = _disk_h2(d, φ); (s, co) = sincos(φ); (v1 * co + v * s, (v2 + v) * co))
    turn = _polish_root(g, A, B, signbit(first(g(A))), asin(clamp(sβ / cθ, -one(T), one(T))), 16)
    return GeodesicDisk(metric, R, t, span, (A, B), ends, turn, false, coef, dcoef, ddcoef)
end

"""
    geodesic_half_width(d, φ) -> T

The half-width `w` in longitude of the disk `d` on the circle of latitude `φ`: `0` where it misses the circle,
`π` where it holds it.
"""
function geodesic_half_width(d::GeodesicDisk{T}, φ::T) where {T<:AbstractFloat}
    d.whole && return T(π)
    (φ <= d.span[1] || φ >= d.span[2]) && return zero(T)
    φ <= d.open[1] && return d.ends[1]
    φ >= d.open[2] && return d.ends[2]
    return 2 * asin(sqrt(clamp(_disk_h(d, φ) / cos(φ), zero(T), one(T))))
end

# The latitudes in `[lo, hi]` where the disk's edge meets the meridians `±c`, `0 < c < π`: one on each side of
# `turn` where `w` passes `c` there, `NaN` where it does not; the roots of `h − sin²(c/2) cos φ`, from the secant
# across the side.
function _disk_crossings(d::GeodesicDisk{T}, c::T, lo::T, hi::T) where {T<:AbstractFloat}
    uc = sin(c / 2)^2
    g(φ) = ((v, v1) = _disk_h1(d, φ); (s, co) = sincos(φ); (v - uc * co, v1 + uc * s))
    function root(p, q)
        p, q = max(p, lo), min(q, hi)
        p < q || return T(NaN)
        gp, gq = first(g(p)), first(g(q))
        signbit(gp) == signbit(gq) && return T(NaN)
        return _polish_root(g, p, q, signbit(gp), p - gp * (q - p) / (gq - gp), 16)
    end
    return (root(d.open[1], d.turn), root(d.turn, d.open[2]))
end

@inline _disk_density(d::GeodesicDisk, φ) = (m = d.metric(φ); m[1] * m[2])

# Where `w` has square-root ends: the span's and where it reaches `π`; and the poles, past which `u = h/cos φ` is
# singular, beside a disk whose edge passes near one.
@inline _region_branch_points(d) = (d.span..., d.open..., -oftype(d.t, π) / 2, oftype(d.t, π) / 2)

# A region of latitude–longitude holding `(λ, φ)` for `|λ| ≤ w(φ)`, `w` monotone on each side of `turn` between
# `open`: the geodesic disk, and a geodetic ball's layer of constant height.
@inline _region_w(d::GeodesicDisk, φ) = geodesic_half_width(d, φ)
@inline _region_crossings(d::GeodesicDisk, c, lo, hi) = _disk_crossings(d, c, lo, hi)
@inline _region_density(d::GeodesicDisk, φ) = _disk_density(d, φ)

"""
    _region_overlap(r, λa, λb, φa, φb, nodes) -> (inside, whole)

`∫ density·L dφ` over `[φa, φb]`, `L` the longitudes of `[λa, λb]` (relative to the region's centre, at most `2π`
wide) inside the region `r`; `whole` where every point of the cell is inside, the overlap then the cell's own, and
`0` exactly where every point is outside. Summed between the latitudes where `L` changes form (the region's span,
where it holds a whole circle, where its edge crosses a face meridian), each piece by the Gauss–Legendre
`nodes = (x, w)` on `[−1, 1]` under a smoothstep substitution.
"""
function _region_overlap(r, λa::T, λb::T, φa::T, φb::T, nodes) where {T<:AbstractFloat}
    r.whole && return zero(T), true
    P = 2 * T(π)
    k = round((λa + λb) / (2P))
    a, b = λa - k * P, λb - k * P
    lo, hi = max(φa, r.span[1]), min(φb, r.span[2])
    lo < hi || return zero(T), false
    fold(x) = abs(rem(x, P, RoundNearest))
    near = a <= 0 <= b ? zero(T) : min(fold(a), fold(b))
    far = (a <= -T(π) || b >= T(π)) ? T(π) : max(fold(a), fold(b))
    # `w` is monotone on each side of `turn`, so its extremes over `[lo, hi]` lie at the ends and the turn.
    wl, wh = _region_w(r, lo), _region_w(r, hi)
    wt = _region_w(r, clamp(r.turn, lo, hi))
    wmin, wmax = min(wl, wh, wt), max(wl, wh, wt)
    wmax <= near && return zero(T), false
    (lo == φa && hi == φb && wmin >= far) && return zero(T), true
    crossing(f) = (c = fold(f); wmin < c < wmax ? _region_crossings(r, c, lo, hi) : (T(NaN), T(NaN)))
    branch_points = _region_branch_points(r)
    cuts = (branch_points..., crossing(a)..., crossing(b)...)
    slice(p, s) = (φ = p + s; _region_density(r, φ) * _lon_overlap(_region_w(r, φ), a, b))
    return _piecewise(slice, lo, hi, nodes, cuts, branch_points), false
end

# The cell's own `(λb − λa) ∫ density dφ`.
@inline _region_cell(r, λa::T, λb::T, φa::T, φb::T, nodes) where {T} =
    (λb - λa) * _smooth_piece(s -> _region_density(r, φa + s), φa, φb, nodes)

"""
    geodesic_cell_fraction(d, λa, λb, φa, φb, nodes) -> T

The fraction of the latitude–longitude cell `[λa, λb] × [φa, φb]` inside the geodesic disk `d`: longitudes relative
to its centre, `0 < λb − λa ≤ 2π`, latitudes in `[−π/2, π/2]`, `φa < φb`. `1` or `0` exactly when every point of
the cell is inside or outside; otherwise `∫ M N cos φ L(φ) dφ` over the cell's area ([`_region_overlap`](@ref)).
"""
function geodesic_cell_fraction(d::GeodesicDisk{T}, λa::T, λb::T, φa::T, φb::T, nodes) where {T<:AbstractFloat}
    inside, whole = _region_overlap(d, λa, λb, φa, φb, nodes)
    whole && return one(T)
    iszero(inside) && return zero(T)
    return clamp(inside / _region_cell(d, λa, λb, φa, φb, nodes), zero(T), one(T))
end

"""
    geodesic_arc_fraction(d, λa, λb) -> T

The fraction of the arc `[λa, λb]` of the circle of latitude through the centre of `d` inside it: a cell of a grid
with one latitude.
"""
function geodesic_arc_fraction(d::GeodesicDisk{T}, λa::T, λb::T) where {T<:AbstractFloat}
    P = 2 * T(π)
    k = round((λa + λb) / (2P))
    return clamp(_lon_overlap(geodesic_half_width(d, d.t), λa - k * P, λb - k * P) / (λb - λa), zero(T), one(T))
end

"""
    geodesic_meridian_fraction(d, φa, φb, nodes) -> T

The fraction of the meridian segment `[φa, φb]` through the centre of `d` inside it, by length: a cell of a grid
with one longitude.
"""
function geodesic_meridian_fraction(d::GeodesicDisk{T}, φa::T, φb::T, nodes) where {T<:AbstractFloat}
    lo, hi = max(φa, d.span[1]), min(φb, d.span[2])
    lo < hi || return zero(T)
    (lo == φa && hi == φb) && return one(T)
    len(p, q) = _smooth_piece(s -> first(d.metric(p + s)), p, q, nodes)
    return clamp(len(lo, hi) / len(φa, φb), zero(T), one(T))
end

"""
    geodesic_disk_area(d, nodes) -> T

The area of the disk `d`, `2 ∫ M N cos φ w(φ) dφ` over its span.
"""
function geodesic_disk_area(d::GeodesicDisk{T}, nodes) where {T<:AbstractFloat}
    lo, hi = d.whole ? (-T(π) / 2, T(π) / 2) : d.span
    branch_points = _region_branch_points(d)
    slice(p, s) = (φ = p + s; _disk_density(d, φ) * geodesic_half_width(d, φ))
    return 2 * _piecewise(slice, lo, hi, nodes, branch_points, branch_points)
end

"""
    geodesic_meridian_length(d, nodes) -> T

The length of the meridian through the centre of `d` inside it, `∫ M dφ` over its span.
"""
geodesic_meridian_length(d::GeodesicDisk{T}, nodes) where {T<:AbstractFloat} =
    _smooth_piece(s -> first(d.metric(d.span[1] + s)), d.span[1], d.span[2], nodes)

# ============================================================================ #
# Ball ∩ geodetic cell on a spheroid
# ============================================================================ #
#
# Points at geodetic `(λ, φ, h)` on a spheroid of equatorial radius `a` and squared eccentricity `e²` lie at
# `((N + h) cos φ cos λ, (N + h) cos φ sin λ, (N(1 − e²) + h) sin φ)`, and the ball of radius `R` about `(0, t, h₀)`, at
# `(X_c, 0, Z_c)`, holds a point where `sin²(λ/2) ≤ u = (R² − D₀²)/(4ρX_c)`, `ρ = (N + h) cos φ` and `D₀` the distance from
# the centre to `(0, φ, h)`. At each height that is a region of the form `|λ| ≤ w(φ)` ([`_region_overlap`](@ref)), in
# closed form; its breakpoints are roots in `φ`, by bounded Newton steps on the closed forms.

"""
    GeodeticBall{T}

The ball of radius `R` about the point at latitude `t`, longitude `0`, height `h0` above a spheroid of equatorial
radius `a` and squared eccentricity `e2`, at `(Xc, 0, Zc)`.
"""
struct GeodeticBall{T<:AbstractFloat}
    a::T
    e2::T
    R::T
    t::T
    h0::T
    Xc::T
    Zc::T
end

function geodetic_ball(a::T, e2::T, R::T, t::T, h0::T) where {T<:AbstractFloat}
    s, c = sincos(t)
    N = a / sqrt(1 - e2 * s * s)
    return GeodeticBall(a, e2, R, t, h0, (N + h0) * c, (N * (1 - e2) + h0) * s)
end

# The meridian section at height `h`: `(ρ, Z)` at latitude `φ` and their first two derivatives in `φ`.
@inline function _meridian_point(b::GeodeticBall{T}, φ::T, h::T) where {T}
    s, c = sincos(φ)
    w = 1 - b.e2 * s * s
    N = b.a / sqrt(w)
    M = b.a * (1 - b.e2) / (w * sqrt(w))
    dM = 3 * b.a * (1 - b.e2) * b.e2 * s * c / (w * w * sqrt(w))
    ρ, Z = (N + h) * c, (N * (1 - b.e2) + h) * s
    ρ1, Z1 = -(M + h) * s, (M + h) * c
    ρ2, Z2 = -dM * s - (M + h) * c, dM * c - (M + h) * s
    return ρ, Z, ρ1, Z1, ρ2, Z2
end

# The squared distance from the centre to `(λ₀, φ, h)` for `λ₀ = 0` (`sgn = 1`) or `π` (`sgn = −1`), and its first
# two derivatives in `φ`.
@inline function _meridian_d2(b::GeodeticBall{T}, φ::T, h::T, sgn::T) where {T}
    ρ, Z, ρ1, Z1, ρ2, Z2 = _meridian_point(b, φ, h)
    x, z = sgn * ρ - b.Xc, Z - b.Zc
    return x * x + z * z, 2 * (sgn * x * ρ1 + z * Z1), 2 * (ρ1 * ρ1 + Z1 * Z1 + sgn * x * ρ2 + z * Z2)
end

"""
    GeodeticLayer{T}

The ball `ball` at height `h`: the region `|λ| ≤ w(φ)` with its latitude `span`, `open`, `ends`, `turn` and `whole`
as a [`GeodesicDisk`](@ref)'s.
"""
struct GeodeticLayer{T<:AbstractFloat}
    ball::GeodeticBall{T}
    h::T
    t::T
    span::NTuple{2,T}
    open::NTuple{2,T}
    ends::NTuple{2,T}
    turn::T
    whole::Bool
end

# `u` at latitude `φ`, and `R² − D₀²` and `ρ`.
@inline function _layer_u(L::GeodeticLayer{T}, φ::T) where {T}
    b = L.ball
    ρ, Z = _meridian_point(b, φ, L.h)
    D0 = hypot(ρ - b.Xc, Z - b.Zc)
    Q = (b.R - D0) * (b.R + D0)
    return Q / (4 * ρ * b.Xc), Q, ρ
end

function _region_w(L::GeodeticLayer{T}, φ::T) where {T}
    L.whole && return T(π)
    (φ <= L.span[1] || φ >= L.span[2]) && return zero(T)
    φ <= L.open[1] && return L.ends[1]
    φ >= L.open[2] && return L.ends[2]
    return 2 * asin(sqrt(clamp(first(_layer_u(L, φ)), zero(T), one(T))))
end

@inline function _region_density(L::GeodeticLayer{T}, φ::T) where {T}
    b = L.ball
    s, c = sincos(φ)
    w = 1 - b.e2 * s * s
    return (b.a * (1 - b.e2) / (w * sqrt(w)) + L.h) * (b.a / sqrt(w) + L.h) * c
end

# The latitudes in `[lo, hi]` where the layer's edge meets the meridians `±c`: the roots of `R² − D₀² − 4ρX_c sin²(c/2)`.
function _region_crossings(L::GeodeticLayer{T}, c::T, lo::T, hi::T) where {T}
    b = L.ball
    k = 4 * b.Xc * sin(c / 2)^2
    function g(φ)
        ρ, _, ρ1 = _meridian_point(b, φ, L.h)
        d2, d2p = _meridian_d2(b, φ, L.h, one(T))
        return b.R^2 - d2 - k * ρ, -d2p - k * ρ1
    end
    function root(p, q)
        p, q = max(p, lo), min(q, hi)
        p < q || return T(NaN)
        gp, gq = first(g(p)), first(g(q))
        signbit(gp) == signbit(gq) && return T(NaN)
        return _polish_root(g, p, q, signbit(gp), p - gp * (q - p) / (gq - gp), 32)
    end
    return (root(L.open[1], L.turn), root(L.turn, L.open[2]))
end

"""
    geodetic_layer(ball, h) -> GeodeticLayer

The ball at height `h`. Along the centre's meridian `D₀` is least, `|h − h₀|`, at `t` (the constant-height surfaces
share their normals, the centre on the one at `t`), so the span's ends are the roots of `D₀ = R` on each side; a pole
inside the ball is covered, and the circle there ends where the opposite meridian leaves the ball. `w`'s extreme is
the root of `u′ ∝ −(D₀²)′ρ − (R² − D₀²)ρ′`.
"""
function geodetic_layer(b::GeodeticBall{T}, h::T) where {T<:AbstractFloat}
    hp = T(π) / 2
    t = b.t
    empty = GeodeticLayer(b, h, t, (t, t), (t, t), (zero(T), zero(T)), t, false)
    abs(h - b.h0) >= b.R && return empty
    R2 = b.R^2
    near(φ) = ((d2, d2p) = _meridian_d2(b, φ, h, one(T)); (d2 - R2, d2p))
    farm(φ) = ((d2, d2p) = _meridian_d2(b, φ, h, -one(T)); (d2 - R2, d2p))
    north, south = first(near(hp)) <= 0, first(near(-hp)) <= 0
    # Each root from the sphere through the ball's centre and this height's point on its normal.
    ρt, Zt = _meridian_point(b, t, h)
    rh, rc = hypot(ρt, Zt), hypot(b.Xc, b.Zc)
    ψ = acos(clamp((rh^2 + rc^2 - R2) / (2 * rh * rc), -one(T), one(T)))
    φs = south ? -hp : _polish_root(near, -hp, t, false, t - ψ, 32)
    φn = north ? hp : _polish_root(near, t, hp, true, t + ψ, 32)
    A, B, whole = φs, φn, false
    if north && south
        dfar(φ) = ((_, d2p, d2pp) = _meridian_d2(b, φ, h, -one(T)); (d2p, d2pp))
        gs, gn = first(dfar(-hp)), first(dfar(hp))
        φm = signbit(gs) == signbit(gn) ? (first(farm(-hp)) >= first(farm(hp)) ? -hp : hp) :
             _polish_root(dfar, -hp, hp, signbit(gs), -t, 32)
        whole = first(farm(φm)) <= 0
        if !whole
            A = φm > -hp ? _polish_root(farm, -hp, φm, true, (φm - hp) / 2, 32) : -hp
            B = φm < hp ? _polish_root(farm, φm, hp, false, (φm + hp) / 2, 32) : hp
        end
    elseif south
        A = _polish_root(farm, -hp, hp, true, ψ - t - T(π), 32)
    elseif north
        B = _polish_root(farm, -hp, hp, false, T(π) - t - ψ, 32)
    end
    B = max(A, B)
    ends = (south ? T(π) : zero(T), north ? T(π) : zero(T))
    span = (south ? -hp : A, north ? hp : B)
    (whole || ends[1] != ends[2] || B <= A) &&
        return GeodeticLayer(b, h, t, span, (A, B), ends, ends[1] > ends[2] ? A : B, whole)
    function g(φ)
        ρ, _, ρ1, _, ρ2 = _meridian_point(b, φ, h)
        d2, d2p, d2pp = _meridian_d2(b, φ, h, one(T))
        Q = R2 - d2
        return -d2p * ρ - Q * ρ1, -d2pp * ρ - Q * ρ2
    end
    turn = _polish_root(g, A, B, signbit(first(g(A))), asin(clamp(sin(t) / cos(ψ), -one(T), one(T))), 32)
    return GeodeticLayer(b, h, t, span, (A, B), ends, turn, false)
end

# The geodetic latitude of the point `(ρ, Z)`, `ρ > 0`, of a meridian plane: the root of
# `ρ sin φ − Z cos φ − e² N sin φ cos φ`, the condition that the normal at `φ` passes through it.
function _geodetic_latitude(b::GeodeticBall{T}, ρ::T, Z::T) where {T}
    function g(φ)
        s, c = sincos(φ)
        w = 1 - b.e2 * s * s
        N = b.a / sqrt(w)
        dN = b.a * b.e2 * s * c / (w * sqrt(w))
        return ρ * s - Z * c - b.e2 * N * s * c, ρ * c + Z * s - b.e2 * (dN * s * c + N * (c * c - s * s))
    end
    hp = T(π) / 2
    return _polish_root(g, -hp, hp, true, atan(Z, ρ * (1 - b.e2)), 32)
end

# The heights where the ball's surface crosses the normal line through `(λ, φ)`: the roots of
# `|P₀ + h n − C|² = R²`, `n` the unit normal and `P₀` the surface point.
function _normal_crossings(b::GeodeticBall{T}, λ::T, φ::T) where {T}
    s, c = sincos(φ)
    sλ, cλ = sincos(λ)
    N = b.a / sqrt(1 - b.e2 * s * s)
    q = (N * c * cλ - b.Xc, N * c * sλ, N * (1 - b.e2) * s - b.Zc)
    n = (c * cλ, c * sλ, s)
    B = n[1] * q[1] + n[2] * q[2] + n[3] * q[3]
    C = q[1]^2 + q[2]^2 + q[3]^2 - b.R^2
    disc = B * B - C
    disc < 0 && return (T(NaN), T(NaN))
    r = -B - copysign(sqrt(disc), B)
    return iszero(r) ? (zero(T), zero(T)) : (r, C / r)
end

"""
    geodetic_cell_fraction(ball, λa, λb, φa, φb, ha, hb, nodes) -> T

The fraction of the geodetic cell `[λa, λb] × [φa, φb] × [ha, hb]` inside `ball`: longitudes relative to its centre's,
`0 < λb − λa ≤ 2π`. `1` exactly where every corner and every point of its latitude faces at longitude `π` lies
inside; otherwise `∫ A(h) dh` over the cell's volume, `A(h)` the overlap of its layer at height `h`
([`_region_overlap`](@ref), weighted by `(M + h)(N + h) cos φ`), summed between the heights where the ball's surface
crosses the normal through a corner, a latitude face's point at longitude `0` or `π`, a pole, or a meridian face's
point nearest the centre, and `h₀ ± R`, by the Gauss–Legendre `nodes` under a smoothstep substitution.
"""
function geodetic_cell_fraction(b::GeodeticBall{T}, λa::T, λb::T, φa::T, φb::T, ha::T, hb::T, nodes) where {T<:AbstractFloat}
    lo, hi = max(ha, b.h0 - b.R), min(hb, b.h0 + b.R)
    lo < hi || return zero(T)
    P = 2 * T(π)
    k = round((λa + λb) / (2P))
    a, c = λa - k * P, λb - k * P
    has(x) = any(m -> a <= x + m * P <= c, -1:1)
    d2(λ, φ, h) = (p = _meridian_point(b, φ, h); (s, co) = sincos(λ);
                   (p[1] * co - b.Xc)^2 + (p[1] * s)^2 + (p[2] - b.Zc)^2)
    R2 = b.R^2
    λs = has(T(π)) ? (a, c, T(π)) : (a, c, a)
    all(x -> d2(x[1], x[2], x[3]) <= R2, Iterators.product(λs, (φa, φb), (ha, hb))) && return one(T)
    # The meridian faces' points nearest the centre, where their own half-plane faces it.
    nearest(λ) = cos(λ) > 0 ? _geodetic_latitude(b, b.Xc * cos(λ), b.Zc) : T(NaN)
    lines = ((a, φa), (a, φb), (c, φa), (c, φb), (zero(T), φa), (zero(T), φb), (T(π), φa), (T(π), φb),
             (zero(T), T(π) / 2), (zero(T), -T(π) / 2), (a, nearest(a)), (c, nearest(c)))
    keep(λ, φ) = !isnan(φ) && φa <= φ <= φb && (λ == a || λ == c || has(λ) || abs(φ) == T(π) / 2)
    hs(λ, φ) = keep(λ, φ) ? _normal_crossings(b, λ, φ) : (T(NaN), T(NaN))
    cuts = (b.h0 - b.R, b.h0 + b.R, Iterators.flatten(map(x -> hs(x...), lines))...)
    function layer(p, s)
        h = p + s
        L = geodetic_layer(b, h)
        inside, whole = _region_overlap(L, a, c, φa, φb, nodes)
        return whole ? _region_cell(L, a, c, φa, φb, nodes) : inside
    end
    v = _outer_piecewise(layer, lo, hi, nodes, cuts, (b.h0 - b.R, b.h0 + b.R))
    cell = _geodetic_cell_volume(b, c - a, φa, φb, ha, hb, nodes)
    return clamp(v / cell, zero(T), one(T))
end

# `Δλ ∫∫ (M + h)(N + h) cos φ dh dφ` over `[φa, φb] × [ha, hb]`, exact in height.
function _geodetic_cell_volume(b::GeodeticBall{T}, Δλ::T, φa::T, φb::T, ha::T, hb::T, nodes) where {T}
    dh = hb - ha
    h2, h3 = dh * (ha + hb) / 2, dh * (ha * ha + ha * hb + hb * hb) / 3
    function f(s)
        φ = φa + s
        si, co = sincos(φ)
        w = 1 - b.e2 * si * si
        N, M = b.a / sqrt(w), b.a * (1 - b.e2) / (w * sqrt(w))
        return co * (N * M * dh + (N + M) * h2 + h3)
    end
    return Δλ * _smooth_piece(f, φa, φb, nodes)
end

end # module Geometry