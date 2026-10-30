# ---------------------------------------------------------------------------
# A `CellOverlap` top-hat's weight on a gridded cell: its measure times the fraction of it inside the support
# ---------------------------------------------------------------------------
#
# A cell spans the faces FlowGeometries places between its centres (`Discretization.faces`, a latitude face
# clamped at a pole), which are the spans its measure integrates. The fraction is evaluated at no less than
# `Float64` and the weight stored in the grid's type.

_overlap_float(::Type{T}) where {T<:AbstractFloat} = promote_type(T, Float64)

"""
    PlanarOverlap{F,N}

The support of a top-hat on a Cartesian grid: the segment, disk or ball of radius `R`, with the Gauss–Legendre
`nodes` a ball's cell fractions integrate by ([`Geometry.ball_box_fraction`](@ref)).
"""
struct PlanarOverlap{F<:AbstractFloat, N}
    R::F
    nodes::N
end

"""
    CapOverlap{F,N}

The support of a top-hat on a sphere: the cap of angular radius `θ`, with the Gauss–Legendre `nodes` its cell
fractions integrate by ([`Geometry.cap_cell_fraction`](@ref)).
"""
struct CapOverlap{F<:AbstractFloat, N}
    θ::F
    radius::F
    nodes::N
end

# The Gauss–Legendre nodes of the overlap integrals, applied to each graded piece: ten hold a cell's fraction within
# about 1e-9, far below the filter's own error against the continuous one.
_overlap_nodes(::Type{F}, n::Integer = 10) where {F} = (r = FlowTransformBindings.gauss_legendre(F, n); (r.μ, r.w))

_overlap_rule(::FlowGeometries.Geometry.AbstractCartesianGeometry{T}, R::Real) where {T} =
    PlanarOverlap(_overlap_float(T)(R), _overlap_nodes(_overlap_float(T)))

function _overlap_rule(geo::FlowGeometries.Geometry.AbstractSphericalGeometry{T}, R::Real) where {T}
    F = _overlap_float(T)
    a = F(FlowGeometries.Geometry.radius(geo))
    return CapOverlap(min(F(R) / a, F(π)), a, _overlap_nodes(F))
end

"""
    GeodesicOverlap{F,N}

The support of a top-hat on a spheroid of equatorial radius `a` and flattening `f`: the geodesic disk of radius
`R` (the whole surface where `whole`), with the Gauss–Legendre `nodes` its cell fractions integrate by
([`Geometry.geodesic_cell_fraction`](@ref)).
"""
struct GeodesicOverlap{F<:AbstractFloat, N}
    a::F
    f::F
    R::F
    whole::Bool
    nodes::N
end

# Every geodesic of length below `πb` is the shortest between its ends (the nearest cut locus, an equatorial point's,
# lies `πb` away), and no two points lie farther apart than half a meridian.
function _overlap_rule(geo::FlowGeometries.Geometry.AbstractEllipsoidalGeometry{T}, R::Real) where {T}
    F = _overlap_float(T)
    a, f = F(FlowGeometries.Geometry.semimajor_axis(geo)), F(FlowGeometries.Geometry.flattening(geo))
    half = 2 * F(FlowGeometries.Geometry.meridian_distance(geo, F(π) / 2))
    F(π) * a * (1 - f) <= R < half && throw(ArgumentError(
        "a top-hat of radius $R reaches past some points' cut locus, where the circle of geodesic radius $R stops " *
        "bounding the disk; this spheroid's top-hats take radii below $(F(π) * a * (1 - f)), or from $half (the whole " *
        "surface)"))
    return GeodesicOverlap(a, f, F(R), R >= half, _overlap_nodes(F))
end

"""
    _overlap_target(ov, t)

What the support's fractions about a target at axis-2 coordinate `t` read: `t` itself, or on a spheroid the
geodesic disk about latitude `t` ([`Geometry.geodesic_disk`](@ref)), built once per target row from the edge points
of FlowGeometries' direct geodesic problem.
"""
_overlap_target(::Union{PlanarOverlap,CapOverlap}, t) = t

function _overlap_target(ov::GeodesicOverlap{F}, t::F) where {F}
    g = FlowGeometries.Geometry._GeodesicConstants(ov.a, ov.f)
    R = ov.R
    direct = α -> FlowGeometries.Geometry._geodesic_direct(g, (zero(F), t), α, R)
    a, e² = ov.a, ov.f * (2 - ov.f)
    metric = φ -> (s = sin(φ); w = 1 - e² * s * s; (a * (1 - e²) / (w * sqrt(w)), a * cos(φ) / sqrt(w)))
    return Geometry.geodesic_disk(direct, metric, R, t, ov.whole)
end

"""
    _cell_faces(grid, d, F) -> Vector{F}

The `N + 1` faces of direction `d`'s cells, as FlowGeometries places them; a sphere's latitude faces clamped
at the poles.
"""
function _cell_faces(grid::FlowGeometries.Grids.StructuredGrid, d::Int, ::Type{F}) where {F}
    x = FlowGeometries.Grids.coordinates(grid, d)
    p = FlowGeometries.Grids.isperiodic(grid, d) ? FlowGeometries.Grids.period(grid, d) : nothing
    f = F.(collect(FlowGeometries.Discretization.faces(x, p)))
    return _clamp_faces!(f, FlowGeometries.Grids.grid_geometry(grid), d)
end

_clamp_faces!(f, ::FlowGeometries.Geometry.AbstractGeometry, ::Int) = f
function _clamp_faces!(f::AbstractVector{F}, ::FlowGeometries.Geometry.AbstractEllipsoidalGeometry, d::Int) where {F}
    d == 2 && clamp!(f, -F(π) / 2, F(π) / 2)
    return f
end
function _clamp_faces!(f::AbstractVector{F}, ::FlowGeometries.Geometry.AbstractSphericalGeometry, d::Int) where {F}
    d == 2 && clamp!(f, -F(π) / 2, F(π) / 2)
    d == 3 && clamp!(f, zero(F), F(Inf))
    return f
end

"""
    OverlapBand{F}

A source row seen from a target row: its faces `lo < hi` along axis 2 (relative to the target on a plane,
latitudes on a sphere) and the target's axis-2 coordinate `t` (zero on a plane, its latitude on a sphere).
`line` marks a grid of one row, whose cells are segments of it.
"""
struct OverlapBand{F<:AbstractFloat}
    lo::F
    hi::F
    t::F
    line::Bool
end

# The band of source row `jj`, shifted by `shift` along axis 2, seen from target coordinate `yt`, or `nothing`
# where none of it lies within the support.
function _overlap_band(ov::PlanarOverlap{F}, fy, jj::Int, shift::F, yt::F, line::Bool) where {F}
    line && return OverlapBand(zero(F), zero(F), zero(F), true)
    lo, hi = minmax(fy[jj] + shift - yt, fy[jj + 1] + shift - yt)
    near = lo <= 0 <= hi ? zero(F) : min(abs(lo), abs(hi))
    near >= ov.R && return nothing
    return OverlapBand(lo, hi, zero(F), false)
end

function _overlap_band(ov::CapOverlap{F}, fy, jj::Int, ::F, yt::F, line::Bool) where {F}
    line && return OverlapBand(yt, yt, yt, true)
    lo, hi = minmax(fy[jj], fy[jj + 1])
    # The nearest latitude of the band to the cap's, compared along the meridian.
    near = lo <= yt <= hi ? zero(F) : min(abs(lo - yt), abs(hi - yt))
    near >= ov.θ && return nothing
    return OverlapBand(lo, hi, yt, false)
end

"""
    GeodesicBand{F,D}

A source row seen from a target row on a spheroid: its latitude faces `lo < hi` and the target's geodesic `disk`.
`line` marks a grid of one row.
"""
struct GeodesicBand{F<:AbstractFloat, D}
    lo::F
    hi::F
    disk::D
    line::Bool
end

function _overlap_band(::GeodesicOverlap{F}, fy, jj::Int, ::F, d::Geometry.GeodesicDisk{F}, line::Bool) where {F}
    line && return GeodesicBand(d.t, d.t, d, true)
    lo, hi = minmax(fy[jj], fy[jj + 1])
    (hi <= d.span[1] || lo >= d.span[2]) && return nothing
    return GeodesicBand(lo, hi, d, false)
end

"""
    _cell_fraction(ov, band, xa, xb, line_x) -> F

The fraction inside the support of the cell spanning `[xa, xb]` along axis 1 (relative to the target) within
`band`. `line_x` marks a grid of one column, whose cells are segments along axis 2 at the target's own axis-1
position.
"""
@inline function _cell_fraction(ov::PlanarOverlap{F}, band::OverlapBand{F}, xa::F, xb::F, line_x::Bool) where {F}
    line_x && band.line && return one(F)
    line_x && return Geometry.interval_fraction(band.lo, band.hi, ov.R)
    band.line && return Geometry.interval_fraction(xa, xb, ov.R)
    return Geometry.disk_fraction(xa, xb, band.lo, band.hi, ov.R)
end

@inline function _cell_fraction(ov::CapOverlap{F}, band::OverlapBand{F}, xa::F, xb::F, line_x::Bool) where {F}
    line_x && band.line && return one(F)
    line_x && return Geometry.interval_fraction(band.lo - band.t, band.hi - band.t, ov.θ)
    band.line && return Geometry.cap_arc_fraction(xa, xb, band.t, band.t, ov.θ)
    return Geometry.cap_cell_fraction(xa, xb, band.lo, band.hi, band.t, ov.θ, ov.nodes)
end

@inline function _cell_fraction(ov::GeodesicOverlap{F}, band::GeodesicBand{F}, xa::F, xb::F, line_x::Bool) where {F}
    d = band.disk
    line_x && band.line && return one(F)
    line_x && return Geometry.geodesic_meridian_fraction(d, band.lo, band.hi, ov.nodes)
    band.line && return Geometry.geodesic_arc_fraction(d, xa, xb)
    return Geometry.geodesic_cell_fraction(d, xa, xb, band.lo, band.hi, ov.nodes)
end

"""
    _support_mass(ov, line_x, line_y, target) -> F

The support's measure in the grid's units, the mass `ZeroFill` divides by where a window leaves the grid: the
disk `πR²`, or `2R` along a single resolved direction (a missing direction's cells have unit width); the cap
`4πa² sin²(θ/2)`, the arc `2w·a cos t` of a circle of latitude `t`, or the meridian's span within the cap; a
spheroid's geodesic disk, its arc along the target's parallel, or its meridian's span within it.
"""
function _support_mass(ov::PlanarOverlap{F}, line_x::Bool, line_y::Bool, ::F) where {F}
    (line_x && line_y) && return one(F)
    (line_x || line_y) && return 2 * ov.R
    return F(π) * ov.R^2
end

function _support_mass(ov::CapOverlap{F}, line_x::Bool, line_y::Bool, t::F) where {F}
    a, θ = ov.radius, ov.θ
    (line_x && line_y) && return one(F)
    line_y && return 2 * Geometry.cap_half_width(t, t, θ) * a * cos(t)
    line_x && return a * (min(F(π) / 2, t + θ) - max(-F(π) / 2, t - θ))
    return 4 * F(π) * a^2 * sin(θ / 2)^2
end

function _support_mass(ov::GeodesicOverlap{F}, line_x::Bool, line_y::Bool, d::Geometry.GeodesicDisk{F}) where {F}
    (line_x && line_y) && return one(F)
    line_y && return 2 * Geometry.geodesic_half_width(d, d.t) * last(d.metric(d.t))
    line_x && return Geometry.geodesic_meridian_length(d, ov.nodes)
    return Geometry.geodesic_disk_area(d, ov.nodes)
end

"""
    _lattice_fraction(ov::PlanarOverlap, o, Δ) -> F

The fraction inside the support of the cell at lattice offset `o` of a uniform Cartesian grid of steps `Δ`.
"""
@inline _lattice_fraction(ov::PlanarOverlap{F}, o::CartesianIndex{1}, Δ::NTuple{1}) where {F} =
    Geometry.interval_fraction((o[1] - F(1) / 2) * F(Δ[1]), (o[1] + F(1) / 2) * F(Δ[1]), ov.R)
@inline _lattice_fraction(ov::PlanarOverlap{F}, o::CartesianIndex{2}, Δ::NTuple{2}) where {F} =
    Geometry.disk_fraction((o[1] - F(1) / 2) * F(Δ[1]), (o[1] + F(1) / 2) * F(Δ[1]),
                           (o[2] - F(1) / 2) * F(Δ[2]), (o[2] + F(1) / 2) * F(Δ[2]), ov.R)
@inline _lattice_fraction(ov::PlanarOverlap{F}, o::CartesianIndex{3}, Δ::NTuple{3}) where {F} =
    Geometry.ball_box_fraction((o[1] - F(1) / 2) * F(Δ[1]), (o[1] + F(1) / 2) * F(Δ[1]),
                               (o[2] - F(1) / 2) * F(Δ[2]), (o[2] + F(1) / 2) * F(Δ[2]),
                               (o[3] - F(1) / 2) * F(Δ[3]), (o[3] + F(1) / 2) * F(Δ[3]), ov.R, ov.nodes)

"""
    _full_support(grid, kernel, scale) -> Bool

Whether `ZeroFill` divides by the support's measure: a [`Kernels.CellOverlap`](@ref) top-hat on a grid where a
window leaves it, whose cells past the edge, weighted by their overlap, complete the support exactly.
"""
_full_support(grid, kernel, scale) = false
_full_support(
    grid::FlowGeometries.Grids.StructuredGrid, kernel::Kernels.TopHatKernel{Kernels.CellOverlap}, scale,
) = _reaches_exterior(grid, kernel, scale)

# The support's measure about each target of row `j` of a 2-D grid, or of any target of an N-D one.
function _support_measure(grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, kernel, scale, j::Int) where {T,G,N}
    ov = _overlap_rule(FlowGeometries.Grids.grid_geometry(grid), Kernels.kernel_radius(kernel, scale))
    dims = FlowGeometries.Grids.size_tuple(grid)
    N == 2 || return T(_support_mass(ov, count(>(1), dims)))
    t = _overlap_float(T)(FlowGeometries.Grids.coordinates(grid, 2)[j])
    return T(_support_mass(ov, dims[1] == 1, dims[2] == 1, _overlap_target(ov, t)))
end

# The support's measure in `n` resolved Cartesian directions (a direction of one cell has unit width).
function _support_mass(ov::PlanarOverlap{F}, n::Int) where {F}
    n == 0 && return one(F)
    n == 1 && return 2 * ov.R
    n == 2 && return F(π) * ov.R^2
    return 4 * F(π) * ov.R^3 / 3
end

# The support on a sphere: the arc `2θa` of the equator a single ring of longitude lies on, the cap's area on
# the surface. On a spheroid's equator, a circle of radius `a`, the arc `2θa`.
_support_mass(ov::CapOverlap{F}, n::Int) where {F} =
    n == 1 ? 2 * ov.θ * ov.radius : 4 * F(π) * ov.radius^2 * sin(ov.θ / 2)^2
_support_mass(ov::GeodesicOverlap, n::Int) =
    n == 1 ? 2 * _reach(ov) * ov.a : throw(ArgumentError("a spheroid's geodesic disk is the support of a 2-D grid"))

"""
    OverlapWeights{O,F,VF}

A [`Kernels.CellOverlap`](@ref) top-hat's weights on a 2-D `StructuredGrid`, cell by cell: the support `ov`, each
row's target ([`_overlap_target`](@ref)), each direction's cell faces and centres, and its closure. A periodic
Cartesian direction tiles (`tiles_x`, `tiles_y`, its images each counted, shifted by `s·m·p`, `s` the coordinate's
sign along the index); a periodic longitude is a ring (`ring_x`), each cell once; a regional longitude box
(`box_x`) reaches the cells across its seam the short way round.
"""
struct OverlapWeights{O, F<:AbstractFloat, VF<:AbstractVector{F}, VT<:AbstractVector}
    ov::O
    targets::VT
    fx::VF
    fy::VF
    x::VF
    y::VF
    px::F
    py::F
    sx::F
    sy::F
    tiles_x::Bool
    tiles_y::Bool
    ring_x::Bool
    box_x::Bool
end

function OverlapWeights(grid::FlowGeometries.Grids.StructuredGrid{T,G,2}, R::Real) where {T,G}
    F = _overlap_float(T)
    geo = FlowGeometries.Grids.grid_geometry(grid)
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    x = F.(collect(FlowGeometries.Grids.coordinates(grid, 1)))
    y = F.(collect(FlowGeometries.Grids.coordinates(grid, 2)))
    fx = Nx == 1 ? F[] : _cell_faces(grid, 1, F)
    fy = Ny == 1 ? F[] : _cell_faces(grid, 2, F)
    cart = geo isa FlowGeometries.Geometry.AbstractCartesianGeometry
    per = (FlowGeometries.Grids.isperiodic(grid, 1), FlowGeometries.Grids.isperiodic(grid, 2))
    tiles_x = cart && per[1] && Nx > 1
    tiles_y = cart && per[2] && Ny > 1
    px = tiles_x ? F(FlowGeometries.Grids.period(grid, 1)) : zero(F)
    py = tiles_y ? F(FlowGeometries.Grids.period(grid, 2)) : zero(F)
    sx = (Nx > 1 && x[Nx] < x[1]) ? -one(F) : one(F)
    sy = (Ny > 1 && y[Ny] < y[1]) ? -one(F) : one(F)
    ov = _overlap_rule(geo, R)
    return OverlapWeights(
        ov, map(t -> _overlap_target(ov, t), y), fx, fy, x, y, px, py, sx, sy, tiles_x, tiles_y,
        !cart && per[1] && Nx > 1, !cart && !per[1] && Nx > 1,
    )
end

"""
    _overlap_fold(f, acc, r::OverlapWeights, grid, i, j) -> acc

Fold `acc = f(acc, J, w)` over every cell `J` the support about target `(i, j)` reaches, `w` its measure times
its fraction inside. Rows are taken outward from the target's until one lies wholly outside, and within each
the cells outward from the target's column until one does: on a convex support symmetric about the target,
a cell's fraction falls as it moves away.
"""
function _overlap_fold(f::Fn, acc, r::OverlapWeights, grid, i::Int, j::Int) where {Fn}
    Ny = length(r.y)
    acc = _overlap_columns(f, acc, r, grid, i, j, _row_band(r, j, j))
    Ny == 1 && return acc
    for dir in (1, -1)
        d = 1
        while true
            jr = j + dir * d
            (r.tiles_y || 1 <= jr <= Ny) || break
            band = _row_band(r, jr, j)
            band === nothing && break
            acc = _overlap_columns(f, acc, r, grid, i, mod1(jr, Ny), band)
            d += 1
        end
    end
    return acc
end

# Source row `jr` (an image index on a tiling axis) seen from target row `j`.
@inline function _row_band(r::OverlapWeights{O,F}, jr::Int, j::Int) where {O,F}
    Ny = length(r.y)
    m = fld(jr - 1, Ny)
    return _overlap_band(r.ov, r.fy, jr - m * Ny, r.sy * F(m) * r.py, r.targets[j], Ny == 1)
end

function _overlap_columns(f::Fn, acc, r::OverlapWeights{O,F}, grid, i::Int, jj::Int, band) where {Fn,O,F}
    Nx = length(r.x)
    xt = r.x[i]
    acc, _ = _overlap_cell(f, acc, r, grid, i, jj, band, xt)
    Nx == 1 && return acc
    for dir in (1, -1)
        d = 1
        while true
            ir = i + dir * d
            if r.ring_x
                (2d < Nx || (2d == Nx && dir == 1)) || break
            elseif !r.tiles_x
                (1 <= ir <= Nx) || break
                (r.box_x && abs(r.x[ir] - xt) > F(π)) && break
            end
            acc, fr = _overlap_cell(f, acc, r, grid, ir, jj, band, xt)
            iszero(fr) && break
            d += 1
        end
    end
    r.box_x || return acc
    # Across a regional box's seam, inward from each far end while the long way round exceeds half a turn.
    for dir in (1, -1)
        ir = dir > 0 ? Nx : 1
        while (dir > 0 ? ir > i : ir < i) && abs(r.x[ir] - xt) > F(π)
            acc, fr = _overlap_cell(f, acc, r, grid, ir, jj, band, xt)
            iszero(fr) && break
            ir -= dir
        end
    end
    return acc
end

@inline function _overlap_cell(f::Fn, acc, r::OverlapWeights{O,F}, grid, ir::Int, jj::Int, band, xt::F) where {Fn,O,F}
    Nx = length(r.x)
    m = fld(ir - 1, Nx)
    ii = ir - m * Nx
    if Nx == 1
        fr = _cell_fraction(r.ov, band, zero(F), zero(F), true)
    else
        shift = r.tiles_x ? r.sx * F(m) * r.px : zero(F)
        xa, xb = minmax(r.fx[ii] + shift - xt, r.fx[ii + 1] + shift - xt)
        fr = _cell_fraction(r.ov, band, xa, xb, false)
    end
    iszero(fr) && return acc, fr
    A = _cell_measure(grid, (ii, jj))
    return f(acc, (ii, jj), A * oftype(A, fr)), fr
end

"""
    OverlapPolygons{O,F,M}

A [`Kernels.CellOverlap`](@ref) top-hat's weights on a 2-D `CurvilinearGrid`, cell by cell: the support `ov`, the
cell-vertex coordinates `xc`, `yc` (one larger than the grid in each direction; the grid's own, or
FlowGeometries' reconstruction from the centres), the ball radius `reach` a query needs to find every cell the
support touches (`R` plus the farthest any cell's vertex lies from its centre), and each coordinate's period
(zero where it does not wrap), by which a cell is taken at its image nearest the target.
"""
struct OverlapPolygons{O, F<:AbstractFloat, M<:AbstractMatrix{F}}
    ov::O
    xc::M
    yc::M
    reach::F
    period::NTuple{2,F}
end

function OverlapPolygons(grid::FlowGeometries.Grids.CurvilinearGrid{T,G,2}, R::Real) where {T,G}
    F = _overlap_float(T)
    C = FlowGeometries.Grids.coordinates(grid)
    K = FlowGeometries.Grids.has_corners(grid) ? FlowGeometries.Grids.corners(grid) :
        map(FlowGeometries.Grids._centers_to_corners, C)
    xc, yc = F.(K[1]), F.(K[2])
    geo = FlowGeometries.Grids.grid_geometry(grid)
    far = zero(F)
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    for j in 1:Ny, i in 1:Nx
        c = FlowGeometries.Grids.coords(SA.SVector, grid, i, j)
        for (a, b) in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))
            v = SA.SVector{2,T}(T(xc[a, b]), T(yc[a, b]))
            far = max(far, F(FlowGeometries.Geometry.distance(geo, c, v)))
        end
    end
    per = map(d -> FlowGeometries.Grids.isperiodic(grid, d) ? F(FlowGeometries.Grids.period(grid, d)) : zero(F), (1, 2))
    return OverlapPolygons(_overlap_rule(geo, R), xc, yc, F(R) + far, per)
end

# The fraction of curvilinear cell `J`, centred at `c`, inside the support about the point `p`, from its four
# vertices at the image nearest `p`.
@inline function _polygon_fraction(r::OverlapPolygons{<:PlanarOverlap{F}}, J, p, c) where {F}
    i, j = J
    shift(d) = iszero(r.period[d]) ? zero(F) : round((F(c[d]) - F(p[d])) / r.period[d]) * r.period[d]
    px, py = F(p[1]) + shift(1), F(p[2]) + shift(2)
    poly = SA.SVector{4,NTuple{2,F}}(
        (r.xc[i, j] - px, r.yc[i, j] - py), (r.xc[i + 1, j] - px, r.yc[i + 1, j] - py),
        (r.xc[i + 1, j + 1] - px, r.yc[i + 1, j + 1] - py), (r.xc[i, j + 1] - px, r.yc[i, j + 1] - py),
    )
    return Geometry.disk_fraction(poly, r.ov.R)
end

# On a sphere the cell's edges are great circles, and its vertices go into the frame whose pole is the target.
@inline function _polygon_fraction(r::OverlapPolygons{<:CapOverlap{F}}, J, p, _) where {F}
    i, j = J
    λt, φt = F(p[1]), F(p[2])
    sφt, cφt = sincos(φt)
    v(a, b) = _cap_frame(r.xc[a, b] - λt, r.yc[a, b], φt, sφt, cφt)
    return Geometry.cap_polygon_fraction((v(i, j), v(i + 1, j), v(i + 1, j + 1), v(i, j + 1)), r.ov.θ)
end

# The unit vector of `(λt + Δλ, φ)` in the frame whose pole is `(λt, φt)` and whose second axis points north
# there; the target's own neighbourhood keeps its digits through `sin(φ − φt)` and `1 − cos Δλ = 2 sin²(Δλ/2)`.
@inline function _cap_frame(Δλ::F, φ::F, φt::F, sφt::F, cφt::F) where {F}
    cφ = cos(φ)
    h = 2 * sin(Δλ / 2)^2
    return (cφ * sin(Δλ), sin(φ - φt) + sφt * cφ * h, cos(φ - φt) - cφt * cφ * h)
end

"""
    OverlapWeightsND{N,O,F,VF}

A [`Kernels.CellOverlap`](@ref) top-hat's weights on a 1-D or 3-D Cartesian `StructuredGrid`, or a sphere's single
ring of longitude, cell by cell: the support `ov`, each direction's cell faces and centres, and its closure. A
periodic Cartesian direction tiles (`tiles`, its images each counted, shifted by `sgn·m·period`); a periodic
longitude is a ring (`ring`), each cell once; a regional longitude (`box`) reaches the cells across its seam the
short way round.
"""
struct OverlapWeightsND{N, O, F<:AbstractFloat, VF<:AbstractVector{F}}
    ov::O
    faces::NTuple{N,VF}
    x::NTuple{N,VF}
    period::NTuple{N,F}
    sgn::NTuple{N,F}
    tiles::NTuple{N,Bool}
    ring::Bool
    box::Bool
end

function OverlapWeightsND(grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, R::Real) where {T,G,N}
    F = _overlap_float(T)
    dirs = ntuple(identity, Val(N))
    cart = G <: FlowGeometries.Geometry.AbstractCartesianGeometry
    x = map(d -> F.(collect(FlowGeometries.Grids.coordinates(grid, d))), dirs)
    per = map(d -> FlowGeometries.Grids.isperiodic(grid, d) && length(x[d]) > 1, dirs)
    tiles = map(p -> cart && p, per)
    return OverlapWeightsND(
        _overlap_rule(FlowGeometries.Grids.grid_geometry(grid), R),
        map(d -> _cell_faces(grid, d, F), dirs), x,
        map(d -> tiles[d] ? F(FlowGeometries.Grids.period(grid, d)) : zero(F), dirs),
        map(v -> (length(v) > 1 && v[end] < v[1]) ? -one(F) : one(F), x), tiles,
        !cart && per[1], !cart && !per[1] && length(x[1]) > 1,
    )
end

# The support's radius in the walk's units: a length on a plane, an angle on a sphere and along a spheroid's
# equator.
@inline _reach(ov::PlanarOverlap) = ov.R
@inline _reach(ov::CapOverlap) = ov.θ
@inline _reach(ov::GeodesicOverlap{F}) where {F} = min(ov.R / ov.a, F(π))

# A cell's fraction from its faces relative to the target: a segment, a box in a ball, or an arc of the equator.
@inline _box_fraction(ov::PlanarOverlap, box::NTuple{1}) = Geometry.interval_fraction(box[1]..., ov.R)
@inline _box_fraction(ov::PlanarOverlap, box::NTuple{3}) =
    Geometry.ball_box_fraction(box[1]..., box[2]..., box[3]..., ov.R, ov.nodes)
@inline _box_fraction(ov::CapOverlap{F}, box::NTuple{1}) where {F} =
    Geometry.cap_arc_fraction(box[1]..., zero(F), zero(F), ov.θ)
@inline _box_fraction(ov::GeodesicOverlap{F}, box::NTuple{1}) where {F} =
    Geometry.cap_arc_fraction(box[1]..., zero(F), zero(F), _reach(ov))

"""
    _overlap_fold_nd(f, acc, r::OverlapWeightsND, grid, I) -> acc

Fold `acc = f(acc, J, w)` over every cell `J` the support about target `I` reaches, `w` its measure times its
fraction inside. Each direction, from the last, is taken outward from the target until a slab's nearest point
lies at the support's radius, counting the directions already fixed; on a ball symmetric about the target
the slabs only grow farther.
"""
@inline _overlap_fold_nd(f::Fn, acc, r::OverlapWeightsND{1,O,F}, grid, I::NTuple{1,Int}) where {Fn,O,F} =
    _overlap_walk(acc, r, I, Val(1), zero(F)) do a1, i, s1, _
        _overlap_box(f, a1, r, grid, (i,), (s1,))
    end

@inline _overlap_fold_nd(f::Fn, acc, r::OverlapWeightsND{3,O,F}, grid, I::NTuple{3,Int}) where {Fn,O,F} =
    _overlap_walk(acc, r, I, Val(3), zero(F)) do a3, k, s3, n3
        _overlap_walk(a3, r, I, Val(2), n3) do a2, j, s2, n2
            _overlap_walk(a2, r, I, Val(1), n2) do a1, i, s1, _
                _overlap_box(f, a1, r, grid, (i, j, k), (s1, s2, s3))
            end
        end
    end

@inline function _overlap_box(f::Fn, acc, r::OverlapWeightsND, grid, J, box) where {Fn}
    fr = _box_fraction(r.ov, box)
    iszero(fr) && return acc
    A = FlowGeometries.Grids.area(grid, J...)
    return f(acc, J, A * oftype(A, fr))
end

"""
    _overlap_walk(g, acc, r::OverlapWeightsND, I, Val(d), r2) -> acc

Direction `d`'s slabs about target `I`, outward from the target's own, `acc = g(acc, ii, (a, b), r2′)` for each
whose nearest point lies within the support given the squared distance `r2` the directions already fixed
contribute: cell `ii`, its faces `a < b` relative to the target, and `r2′` that distance with this slab's added.
"""
function _overlap_walk(g::G, acc, r::OverlapWeightsND{N,O,F}, I::NTuple{N,Int}, ::Val{d}, r2::F) where {G,N,O,F,d}
    x = r.x[d]
    n = length(x)
    xt = x[I[d]]
    R2 = _reach(r.ov)^2
    ring, box_x = d == 1 && r.ring, d == 1 && r.box
    for dir in (0, 1, -1)
        k = abs(dir)
        while true
            ir = I[d] + dir * k
            if ring
                (k == 0 || 2k < n || (2k == n && dir == 1)) || break
            elseif !r.tiles[d]
                (1 <= ir <= n) || break
                (box_x && abs(x[ir] - xt) > F(π)) && break
            end
            m = fld(ir - 1, n)
            ii = ir - m * n
            a, b, near = _overlap_slab(r, Val(d), ii, m, xt)
            r2 + near^2 >= R2 && break
            acc = g(acc, ii, (a, b), r2 + near^2)
            dir == 0 && break
            k += 1
        end
    end
    box_x || return acc
    # Across a regional longitude's seam, inward from each far end while the long way round exceeds half a turn.
    for dir in (1, -1)
        ir = dir > 0 ? n : 1
        while (dir > 0 ? ir > I[d] : ir < I[d]) && abs(x[ir] - xt) > F(π)
            a, b, near = _overlap_slab(r, Val(d), ir, 0, xt)
            r2 + near^2 >= R2 && break
            acc = g(acc, ir, (a, b), r2 + near^2)
            ir -= dir
        end
    end
    return acc
end

# The faces relative to the target of direction `d`'s cell `ii` (image `m`), and their nearest point's distance.
# A longitude's faces go modulo `2π` to the side nearest the target.
@inline function _overlap_slab(r::OverlapWeightsND{N,O,F}, ::Val{d}, ii::Int, m::Int, xt::F) where {N,O,F,d}
    fc = r.faces[d]
    shift = r.tiles[d] ? r.sgn[d] * F(m) * r.period[d] : zero(F)
    a, b = minmax(fc[ii] + shift - xt, fc[ii + 1] + shift - xt)
    if d == 1 && (r.ring || r.box)
        P = 2 * F(π)
        k = round((a + b) / (2P))
        a, b = a - k * P, b - k * P
    end
    return a, b, (a <= 0 <= b ? zero(F) : min(abs(a), abs(b)))
end

"""
    OverlapShells{F,VF,N,B}

A [`Kernels.CellOverlap`](@ref) top-hat's weights on a 3-D spherical or spheroidal `StructuredGrid`, whose distance
is the chord: the ball of radius `R` about the target, and each cell the region between its longitude, latitude and
third-direction faces (latitude faces clamped at the poles), inside it with the Gauss–Legendre `nodes`: a sphere's
radius sector by [`Geometry.shell_cell_fraction`](@ref) (`body` `nothing`, radius faces clamped at the origin), a
spheroid's geodetic cell of heights by [`Geometry.geodetic_cell_fraction`](@ref) (`body` its equatorial radius and
squared eccentricity). `reach`, `R` plus the farthest any cell's corner lies from its centre, is the ball a query needs
to find every cell the support touches.
"""
struct OverlapShells{F<:AbstractFloat, VF<:AbstractVector{F}, N, B}
    R::F
    nodes::N
    faces::NTuple{3,VF}
    x::NTuple{3,VF}
    reach::F
    body::B
end

function OverlapShells(grid::FlowGeometries.Grids.StructuredGrid{T,G,3}, R::Real) where {T,G}
    F = _overlap_float(T)
    faces = ntuple(d -> _cell_faces(grid, d, F), Val(3))
    x = ntuple(d -> F.(collect(FlowGeometries.Grids.coordinates(grid, d))), Val(3))
    body = _shell_body(FlowGeometries.Grids.grid_geometry(grid), F)
    return OverlapShells(F(R), _overlap_nodes(F), faces, x, F(R) + _shell_extent(grid, faces, x), body)
end

_shell_body(::FlowGeometries.Geometry.AbstractSphericalGeometry, ::Type{F}) where {F} = nothing
_shell_body(geo::FlowGeometries.Geometry.AbstractEllipsoidalGeometry, ::Type{F}) where {F} =
    (F(FlowGeometries.Geometry.semimajor_axis(geo)), (f = F(FlowGeometries.Geometry.flattening(geo)); f * (2 - f)))

# The farthest any cell's corner lies from its centre along the chord.
function _shell_extent(grid::FlowGeometries.Grids.StructuredGrid{T}, faces, x) where {T}
    geo = FlowGeometries.Grids.grid_geometry(grid)
    far = zero(eltype(first(x)))
    for I in CartesianIndices(map(length, x))
        c = ntuple(d -> x[d][I[d]], Val(3))
        for o in CartesianIndices((0:1, 0:1, 0:1))
            v = ntuple(d -> faces[d][I[d] + o[d]], Val(3))
            far = max(far, FlowGeometries.Geometry.distance(geo, c, v))
        end
    end
    return far
end

# Cell `J`'s fraction inside the ball about target `I`.
@inline function _shell_fraction(r::OverlapShells{F,VF,N,Nothing}, I::NTuple{3,Int}, J::NTuple{3,Int}) where {F,VF,N}
    fλ, fφ, fr = r.faces
    λt = r.x[1][I[1]]
    return Geometry.shell_cell_fraction(fλ[J[1]] - λt, fλ[J[1] + 1] - λt, fφ[J[2]], fφ[J[2] + 1], fr[J[3]], fr[J[3] + 1],
                                        r.x[2][I[2]], r.x[3][I[3]], r.R, r.nodes)
end

@inline function _shell_fraction(r::OverlapShells{F,VF,N,NTuple{2,F}}, I::NTuple{3,Int}, J::NTuple{3,Int}) where {F,VF,N}
    fλ, fφ, fh = r.faces
    λt = r.x[1][I[1]]
    ball = Geometry.geodetic_ball(r.body[1], r.body[2], r.R, r.x[2][I[2]], r.x[3][I[3]])
    return Geometry.geodetic_cell_fraction(ball, fλ[J[1]] - λt, fλ[J[1] + 1] - λt, fφ[J[2]], fφ[J[2] + 1], fh[J[3]],
                                           fh[J[3] + 1], r.nodes)
end

"""
    _overlap_exterior(grid, r::OverlapWeights, kernel, scale) -> Matrix or nothing

The mass past the grid's edges under `ZeroFill`: the support's measure less the weights a target's window sums
over the grid's own cells, where a window leaves the grid; `nothing` where none does.
"""
function _overlap_exterior(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2}, r::OverlapWeights, kernel, scale,
) where {T,G}
    _reaches_exterior(grid, kernel, scale) || return nothing
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    E = Matrix{T}(undef, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        s = _overlap_fold((a, _, w) -> a + w, zero(T), r, grid, i, j)
        E[i, j] = T(_support_mass(r.ov, Nx == 1, Ny == 1, r.targets[j])) - s
    end
    return E
end

function _overlap_exterior(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, r::OverlapWeightsND, kernel, scale,
) where {T,G,N}
    _reaches_exterior(grid, kernel, scale) || return nothing
    dims = FlowGeometries.Grids.size_tuple(grid)
    M = T(_support_mass(r.ov, count(>(1), dims)))
    E = Array{T,N}(undef, dims)
    for I in CartesianIndices(dims)
        E[I] = M - _overlap_fold_nd((a, _, w) -> a + w, zero(T), r, grid, Tuple(I))
    end
    return E
end
