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
    v = gap >= 8 * (hi - lo) ? _gauss4(r -> slice(lo, r - lo), lo, hi) : _piecewise(slice, lo, hi, nodes, cuts, cuts)
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
# `R` about latitude `t`, longitude `0` holds `(λ, φ)` for `|λ| ≤ w(φ)`. `u = sin²(w/2)` solves
# `s(2 asin √u, φ) = R`, whose left side is analytic in `u` at both ends, so `u` is analytic in `φ` where
# `0 < w < π` and is held there as piecewise Chebyshev series; `w` is `0` off the disk's latitude span and `π`
# across a pole the disk covers.

"""
    GeodesicDisk{T,M,V}

The disk of geodesic radius `R` about latitude `t`, longitude `0`, on a surface of revolution whose lengths per
unit latitude and longitude are `metric(φ) = (M, N cos φ)`: its latitude `span` where `w > 0`, the latitudes
`open` between which `0 < w < π` and `w` at them (`ends`: `0`, or `π` across a covered pole), the latitude `turn`
of `w`'s one extreme between them (an end where `w` is monotone), `whole` where it holds every point, and the
`n`-term Chebyshev series of `u = sin²(w/2)` and of its derivative in the piece's variable on each piece between
consecutive `knots`.
"""
struct GeodesicDisk{T<:AbstractFloat, M, V<:AbstractVector{T}}
    metric::M
    R::T
    t::T
    span::NTuple{2,T}
    open::NTuple{2,T}
    ends::NTuple{2,T}
    turn::T
    whole::Bool
    knots::V
    coef::V
    dcoef::V
    n::Int
end

# The root of `g` in `[lo, hi]`, `g` returning its value and derivative and `slo` the sign bit of its value at
# `lo` (the opposite at `hi`), by Newton's method from `x`; a step leaving the bracket bisects it.
function _newton_root(g::G, lo::T, hi::T, slo::Bool, x::T) where {G, T<:AbstractFloat}
    (lo < x < hi) || (x = (lo + hi) / 2)
    for _ in 1:100
        v, dv = g(x)
        iszero(v) && return x
        signbit(v) == slo ? (lo = x) : (hi = x)
        xn = x - v / dv
        (lo < xn < hi) || (xn = (lo + hi) / 2)
        abs(xn - x) <= 4 * eps(T) * max(abs(xn), floatmin(T)) && return xn
        x = xn
    end
    return x
end

# The maximizer of the unimodal `f` on `[a, b]` by golden-section search.
function _golden_max(f::G, a::T, b::T) where {G, T<:AbstractFloat}
    r = (sqrt(T(5)) - 1) / 2
    c, d = b - r * (b - a), a + r * (b - a)
    fc, fd = f(c), f(d)
    for _ in 1:90
        if fc >= fd
            b, d, fd = d, c, fc
            c = b - r * (b - a)
            fc = f(c)
        else
            a, c, fc = c, d, fd
            d = a + r * (b - a)
            fd = f(d)
        end
    end
    return (a + b) / 2
end

# `u = sin²(w/2)` on the circle of latitude `φ`: `s(λ, φ) = R` by Newton's method in `u` on `q(s) − q(R)`,
# `q(s) = sin²(s/2ρ)` (linear in `u` on a sphere of radius `ρ`), from the secant through `u = 0` and `u = 1`;
# `0` where the disk misses the circle and `1` where it holds it. `∂s/∂λ = −N cos φ sin α`.
function _disk_u(inverse::I, metric::M, R::T, ρ::T, φ::T) where {I, M, T<:AbstractFloat}
    q(s) = sin(s / (2ρ))^2
    qR = q(R)
    h0 = q(first(inverse(zero(T), φ))) - qR
    h1 = q(first(inverse(T(π), φ))) - qR
    h0 >= 0 && return zero(T)
    h1 <= 0 && return one(T)
    Nc = last(metric(φ))
    lo, hi = zero(T), one(T)
    u = -h0 / (h1 - h0)
    for _ in 1:100
        λ = 2 * asin(sqrt(u))
        s, sα, _ = inverse(λ, φ)
        H = q(s) - qR
        iszero(H) && return u
        H < 0 ? (lo = u) : (hi = u)
        un = u + H * ρ * sin(λ) / (sin(s / ρ) * Nc * sα)
        (lo < un < hi) || (un = (lo + hi) / 2)
        abs(un - u) <= 4 * eps(T) * un && return un
        u = un
    end
    return u
end

# The `n` Chebyshev coefficients of the samples `v[k+1] = f(cos(πk/(n − 1)))`, `k = 0, …, n − 1`.
function _cheb_coefficients(v::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(v)
    m = n - 1
    c = Vector{T}(undef, n)
    for j in 0:m
        s = (v[1] + (iseven(j) ? v[n] : -v[n])) / 2
        for k in 1:(m - 1)
            s += v[k + 1] * cospi(T(j * k) / m)
        end
        c[j + 1] = 2 * s / m
    end
    c[1] /= 2
    c[n] /= 2
    return c
end

# The coefficients of the derivative of the Chebyshev series `c`: `c′ₖ₋₁ = c′ₖ₊₁ + 2k cₖ` downward, `c′₀` halved.
function _cheb_derivative(c::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(c)
    d = zeros(T, n)
    for k in (n - 1):-1:1
        d[k] = (k + 1 < n ? d[k + 2] : zero(T)) + 2k * c[k + 1]
    end
    d[1] /= 2
    return d
end

# Fits `f` on `[p, q]` by `n`-term Chebyshev series, halving a piece (at most `depth` times) until its last three
# coefficients fall below `1e-11` of its largest sample, and appends each piece's right end to `knots`, its
# coefficients to `coef` and its derivative's to `dcoef`. The samples of `u` carry the geodesic distance's rounding,
# about `1e-13` of `u` on short lines at high latitude.
function _fit_pieces!(knots, coef, dcoef, f::G, p::T, q::T, n::Int, depth::Int) where {G, T<:AbstractFloat}
    v = [f((p + q) / 2 + (q - p) / 2 * cospi(T(k) / (n - 1))) for k in 0:(n - 1)]
    c = _cheb_coefficients(v)
    if depth == 0 || max(abs(c[n - 2]), abs(c[n - 1]), abs(c[n])) <= T(1e-11) * maximum(abs, v)
        push!(knots, q)
        append!(coef, c)
        append!(dcoef, _cheb_derivative(c))
        return nothing
    end
    m = (p + q) / 2
    _fit_pieces!(knots, coef, dcoef, f, p, m, n, depth - 1)
    _fit_pieces!(knots, coef, dcoef, f, m, q, n, depth - 1)
    return nothing
end

# The piecewise series at `φ`, by Clenshaw's recurrence on the piece holding it; with `slope`, also its derivative
# in `φ` from the derivative's series.
@inline function _clenshaw(coef::AbstractVector{T}, off::Int, n::Int, x::T) where {T<:AbstractFloat}
    b1 = b2 = zero(T)
    @inbounds for j in n:-1:2
        b1, b2 = 2x * b1 - b2 + coef[off + j], b1
    end
    return x * b1 - b2 + @inbounds(coef[off + 1])
end

@inline function _series_value(knots::AbstractVector{T}, coef::AbstractVector{T}, n::Int, φ::T) where {T<:AbstractFloat}
    k = clamp(searchsortedlast(knots, φ), 1, length(knots) - 1)
    p, q = knots[k], knots[k + 1]
    return _clenshaw(coef, (k - 1) * n, n, (2φ - p - q) / (q - p))
end

@inline function _series_slope(d, φ::T) where {T<:AbstractFloat}
    k = clamp(searchsortedlast(d.knots, φ), 1, length(d.knots) - 1)
    p, q = d.knots[k], d.knots[k + 1]
    x = (2φ - p - q) / (q - p)
    return _clenshaw(d.coef, (k - 1) * d.n, d.n, x), 2 * _clenshaw(d.dcoef, (k - 1) * d.n, d.n, x) / (q - p)
end

"""
    geodesic_disk(inverse, metric, R, t, ρ) -> GeodesicDisk

The disk of geodesic radius `R` about latitude `t`, longitude `0`. `inverse(λ, φ)` returns the distance from
`(λ, φ)` to the centre and the sine and cosine of the azimuth there toward it, east of north; `metric(φ)` the
lengths `(M, N cos φ)` per unit latitude and longitude; `ρ` a length with every distance at most `πρ` (an oblate
spheroid's equatorial radius). The span's ends are the roots of the distance along the centre's meridian, the
covered poles' plateaus end at the roots along the opposite meridian, both by Newton's method with
`∂s/∂φ = −M cos α`.
"""
function geodesic_disk(inverse::I, metric::M, R::T, t::T, ρ::T) where {I, M, T<:AbstractFloat}
    h = T(π) / 2
    dist(λ, φ) = first(inverse(λ, φ))
    along(λ) = φ -> ((s, _, cα) = inverse(λ, φ); (s - R, -first(metric(φ)) * cα))
    north = dist(zero(T), h) <= R
    south = dist(zero(T), -h) <= R
    φs = south ? -h : _newton_root(along(zero(T)), -h, t, false, t - R / first(metric(t)))
    φn = north ? h : _newton_root(along(zero(T)), t, h, true, t + R / first(metric(t)))
    A, B, whole = φs, φn, false
    far = along(T(π))
    if north && south
        φm = _golden_max(φ -> dist(T(π), φ), -h, h)
        whole = dist(T(π), φm) <= R
        if !whole
            A = _newton_root(far, -h, φm, true, φm)
            B = _newton_root(far, φm, h, false, φm)
        end
    elseif south
        A = _newton_root(far, -h, h, true, (R - dist(T(π), -h)) / first(metric(-h)) - h)
    elseif north
        B = _newton_root(far, -h, h, false, h - (R - dist(T(π), h)) / first(metric(h)))
    end
    B = max(A, B)
    ends = (south ? T(π) : zero(T), north ? T(π) : zero(T))
    knots, coef, dcoef, n = T[A], T[], T[], 33
    if !whole && B > A
        _fit_pieces!(knots, coef, dcoef, φ -> _disk_u(inverse, metric, R, ρ, φ), A, B, n, 10)
    else
        push!(knots, B)
        append!(coef, zeros(T, n))
        append!(dcoef, zeros(T, n))
    end
    sgn = iszero(ends[1]) ? one(T) : -one(T)
    turn = (ends[1] != ends[2] || B <= A) ? (ends[1] > ends[2] ? A : B) :
           _golden_max(φ -> sgn * _series_value(knots, coef, n, φ), A, B)
    return GeodesicDisk(metric, R, t, (φs, φn), (A, B), ends, turn, whole, knots, coef, dcoef, n)
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
    return 2 * asin(sqrt(clamp(_series_value(d.knots, d.coef, d.n, φ), zero(T), one(T))))
end

# The latitudes where the disk's boundary meets the meridians `±c`, `0 < c < π`: one on each side of `turn` where
# `w` passes `c` there, `NaN` where it does not; by Newton's method on the series from the secant across the side.
function _disk_crossings(d::GeodesicDisk{T}, c::T) where {T<:AbstractFloat}
    uc = sin(c / 2)^2
    g(φ) = ((u, du) = _series_slope(d, φ); (u - uc, du))
    function root(p, q)
        p < q || return T(NaN)
        gp, gq = first(g(p)), first(g(q))
        signbit(gp) == signbit(gq) && return T(NaN)
        return _newton_root(g, p, q, signbit(gp), p - gp * (q - p) / (gq - gp))
    end
    return (root(d.open[1], d.turn), root(d.turn, d.open[2]))
end

@inline _disk_density(d::GeodesicDisk, φ) = (m = d.metric(φ); m[1] * m[2])

"""
    geodesic_cell_fraction(d, λa, λb, φa, φb, nodes) -> T

The fraction of the latitude–longitude cell `[λa, λb] × [φa, φb]` inside the geodesic disk `d`: longitudes relative
to its centre, `0 < λb − λa ≤ 2π`, latitudes in `[−π/2, π/2]`, `φa < φb`. `1` or `0` exactly when every point of
the cell is inside or outside. Otherwise the overlap `∫ M N cos φ L(φ) dφ`, `L` the cell's longitudes within
[`geodesic_half_width`](@ref), over the cell's area, summed between the latitudes where `L` changes form (the
disk's span, where it holds a whole circle, where its boundary crosses a face meridian), each piece by the
Gauss–Legendre `nodes = (x, w)` on `[−1, 1]` under a smoothstep substitution.
"""
function geodesic_cell_fraction(d::GeodesicDisk{T}, λa::T, λb::T, φa::T, φb::T, nodes) where {T<:AbstractFloat}
    d.whole && return one(T)
    P = 2 * T(π)
    k = round((λa + λb) / (2P))
    a, b = λa - k * P, λb - k * P
    lo, hi = max(φa, d.span[1]), min(φb, d.span[2])
    lo < hi || return zero(T)
    fold(x) = abs(rem(x, P, RoundNearest))
    near = a <= 0 <= b ? zero(T) : min(fold(a), fold(b))
    far = (a <= -T(π) || b >= T(π)) ? T(π) : max(fold(a), fold(b))
    # `w` is monotone on each side of `turn`, so its extremes over `[lo, hi]` lie at the ends and the turn.
    wl, wh = geodesic_half_width(d, lo), geodesic_half_width(d, hi)
    wt = geodesic_half_width(d, clamp(d.turn, lo, hi))
    max(wl, wh, wt) <= near && return zero(T)
    (lo == φa && hi == φb && min(wl, wh, wt) >= far) && return one(T)
    crossing(f) = (c = fold(f); zero(T) < c < T(π) ? _disk_crossings(d, c) : (T(NaN), T(NaN)))
    branch_points = (d.span..., d.open...)
    cuts = (branch_points..., crossing(a)..., crossing(b)...)
    slice(p, s) = (φ = p + s; _disk_density(d, φ) * _lon_overlap(geodesic_half_width(d, φ), a, b))
    inside = _piecewise(slice, lo, hi, nodes, cuts, branch_points)
    cell = (b - a) * _smooth_piece(s -> _disk_density(d, φa + s), φa, φb, nodes)
    return clamp(inside / cell, zero(T), one(T))
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
    branch_points = (d.span..., d.open...)
    slice(p, s) = (φ = p + s; _disk_density(d, φ) * geodesic_half_width(d, φ))
    return 2 * _piecewise(slice, lo, hi, nodes, branch_points, branch_points)
end

"""
    geodesic_meridian_length(d, nodes) -> T

The length of the meridian through the centre of `d` inside it, `∫ M dφ` over its span.
"""
geodesic_meridian_length(d::GeodesicDisk{T}, nodes) where {T<:AbstractFloat} =
    _smooth_piece(s -> first(d.metric(d.span[1] + s)), d.span[1], d.span[2], nodes)

end # module Geometry