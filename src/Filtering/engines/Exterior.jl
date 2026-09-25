# ---------------------------------------------------------------------------
# The kernel's mass beyond a bounded edge
# ---------------------------------------------------------------------------
#
# Coarse graining filters the field extended by zero beyond the domain, with the kernel normalized over
# all space (Aluie et al. 2018; Grooms et al. 2021, eq. 7). `ZeroFill` therefore divides by the mass of
# the in-domain taps an engine sums plus the mass of the cells beyond each bounded edge. Those cells are
# the grid's lattice continued at its edge spacing: a spherical latitude as far as the poles, a regional
# longitude as far as one turn, a radius as far as the origin.

"""
    _exterior_mass(grid, kernel, scale) -> Array or nothing

`Σ kernel_weight(d) · measure` over the exterior cells within the kernel's radius of each cell of
`grid`, or `nothing` when no cell reaches past a bounded edge.
"""
function _exterior_mass(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, kernel::Kernels.AbstractFilterKernel, scale::T,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}, N}
    dim = _kernel_dim(grid)
    rad = Kernels.kernel_radius(kernel, scale, dim)
    lattice = _exterior_lattice(grid, rad)
    lattice === nothing && return nothing
    ext, lo, extended = lattice
    return _exterior_mass(ext, lo, extended, FlowGeometries.Grids.size_tuple(grid), kernel, scale, rad, dim)
end

# Each direction continues as a `Range` or as a `Vector` depending on how far it reaches, so `ext`'s type
# is known only here, past the call.
function _exterior_mass(
    ext::FlowGeometries.Grids.StructuredGrid{T,G,N}, lo::NTuple{N,Int}, extended::NTuple{N,Bool},
    dims::NTuple{N,Int}, kernel::Kernels.AbstractFilterKernel, scale::T, rad::T, dim::Val,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}, N}
    mt = _query_topology(ext, rad)
    images = _image_convention(ext)
    E = zeros(T, dims)
    reached = false
    for I in CartesianIndices(dims)
        Ie = ntuple(d -> I[d] + lo[d], Val(N))
        w = FlowGeometries.Connectivity.metric_window(ext, Ie, rad, mt)
        _window_inside(Ie, w, lo, dims, extended) && continue
        m = FlowGeometries.Connectivity.fold_within(
            zero(T), ext, Ie...;
            ball = rad, self = true, active_only = false, topology = mt, images = images,
        ) do acc, J, d
            _inside(J, lo, dims) ? acc :
                acc + Kernels.kernel_weight(kernel, T(d), scale, dim) * FlowGeometries.Grids.area(ext, J...)
        end
        @inbounds E[I] = m
        reached |= !iszero(m)
    end
    return reached ? E : nothing
end

@inline _inside(J::NTuple{N,Int}, lo::NTuple{N,Int}, dims::NTuple{N,Int}) where {N} =
    all(ntuple(d -> lo[d] < J[d] <= lo[d] + dims[d], Val(N)))

# Whether the index window around `I` stays among the grid's own cells along every extended direction.
@inline _window_inside(I::NTuple{N,Int}, w, lo::NTuple{N,Int}, dims::NTuple{N,Int}, extended::NTuple{N,Bool}) where {N} =
    all(ntuple(d -> !extended[d] || (I[d] - w[d] > lo[d] && I[d] + w[d] <= lo[d] + dims[d]), Val(N)))

"""
    _reaches_exterior(grid, kernel, scale) -> Bool

Whether the lattice continues past a bounded edge of `grid` within the kernel's reach, in which case
`ZeroFill` and `Deformable` divide by different masses even on an unmasked grid. `true` whenever a
window can leave the grid, so a strategy check built on it never accepts a mismatch.
"""
function _reaches_exterior(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, kernel::Kernels.AbstractFilterKernel, scale::T,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}, N}
    rad = Kernels.kernel_radius(kernel, scale, _kernel_dim(grid))
    geo = FlowGeometries.Grids.grid_geometry(grid)
    return any(ntuple(d -> _axis_counts(geo, grid, d, rad, true) != (0, 0), Val(N)))
end

"""
    _exterior_lattice(grid, rad; cap = true) -> (ext, lo, extended) or nothing

`grid`'s lattice continued past each bounded edge far enough to hold every exterior cell within `rad`
of a cell of `grid`, as a grid of the same geometry and closure; `lo[d]` exterior cells precede the
grid's own along direction `d`, and `extended[d]` says whether direction `d` gained any.

`cap = false` continues each direction by the rows centred on its lattice only, leaving out the cap row
between an end face and the pole. That is the node set's completion: its harmonic fit weights every
node alike, and a cap row can cover a sliver of the sphere with as many nodes as a lattice row.
"""
function _exterior_lattice(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, rad::T; cap::Bool = true,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}, N}
    geo = FlowGeometries.Grids.grid_geometry(grid)
    parts = ntuple(d -> _axis_extension(geo, grid, d, rad, cap), Val(N))
    extended = map(p -> p[1] > 0 || p[2] > 0, parts)
    any(extended) || return nothing
    closure = ntuple(d -> _ext_closure(geo, grid, d, extended[d]), Val(N))
    ext = FlowGeometries.Grids.StructuredGrid(
        geo, map(p -> p[3], parts)...; topology = map(first, closure), period = map(last, closure),
    )
    return _exact_continued_measure(ext, geo, parts), map(p -> p[1], parts), extended
end

# The lattice's measure, with each continued cell's factor the exact integral between its own faces. The
# lattice places faces midway between centres, which a cell cut at a pole or at the origin, centred on
# the part that remains, does not satisfy.
_exact_continued_measure(ext, ::FlowGeometries.Geometry.AbstractGeometry, parts) = ext

function _exact_continued_measure(
    ext::FlowGeometries.Grids.StructuredGrid{T,G,N}, geo::FlowGeometries.Geometry.AbstractSphericalGeometry{T},
    parts,
) where {T<:AbstractFloat, G, N}
    N >= 2 || return ext
    factors = FlowGeometries.Grids.measure_factors(ext)
    exact = ntuple(Val(N)) do d
        n_lo, n_hi, axis, faces = parts[d]
        (d == 1 || d > 3 || faces === nothing) && return factors[d]
        f = collect(T, factors[d])
        lo_a, lo_b, hi_a, hi_b = faces
        nλ = length(FlowGeometries.Grids.coordinates(ext, 1))
        for k in 1:n_lo
            f[n_lo + 1 - k] = _cell_factor(geo, d, N, nλ, lo_a[k], lo_b[k])
        end
        n = length(axis) - n_lo - n_hi
        for k in 1:n_hi
            f[n_lo + n + k] = _cell_factor(geo, d, N, nλ, hi_a[k], hi_b[k])
        end
        f
    end
    return FlowGeometries.Grids.rebuild(ext, (measure = FlowGeometries.Grids.SeparableMeasure(exact),))
end

# The sphere's measure factor of one cell between faces `a < b` along direction `d`, as the lattice
# forms it: latitude `R²(sin b − sin a)` in two dimensions (`R(b − a)` along a single meridian) and
# `sin b − sin a` with a radius, radius `(b³ − a³)/3`. `sin b − sin a` is taken as
# `2cos(φ̄)sin(w/2)`, exact near a pole.
@inline function _cell_factor(
    geo::FlowGeometries.Geometry.AbstractSphericalGeometry{T}, d::Int, N::Int, nλ::Int, a::T, b::T,
) where {T<:AbstractFloat}
    d == 3 && return (b - a) * (a * a + a * b + b * b) / T(3)
    R = T(FlowGeometries.Geometry.radius(geo))
    (N == 2 && nλ == 1) && return R * (b - a)
    s = T(2) * cos((a + b) / T(2)) * sin((b - a) / T(2))
    return N == 2 ? R * R * s : s
end

# `(topology, period)` of direction `d` of the continued lattice: the grid's own, except a regional
# longitude continued around the ring, which closes with period 2π.
function _ext_closure(geo::FlowGeometries.Geometry.AbstractGeometry{T}, grid, d::Int, extended::Bool) where {T}
    if geo isa FlowGeometries.Geometry.AbstractSphericalGeometry && d == 1 && extended
        return (FlowGeometries.Grids.Periodic(), T(2π))
    end
    periodic = FlowGeometries.Grids.isperiodic(grid, d)
    return (FlowGeometries.Grids.topology(grid, d), periodic ? FlowGeometries.Grids.period(grid, d) : nothing)
end

# `(n_lo, n_hi)`: how many cells direction `d` continues by below its first and above its last.
function _axis_counts(
    geo::FlowGeometries.Geometry.AbstractGeometry{T}, grid, d::Int, rad::T, cap::Bool,
) where {T<:AbstractFloat}
    x = FlowGeometries.Grids.coordinates(grid, d)
    n = length(x)
    (FlowGeometries.Grids.isperiodic(grid, d) || n < 2 || _spans_sphere(geo, grid, d)) && return (0, 0)
    return _extension_counts(geo, d, x, T(x[2] - x[1]), T(x[n] - x[n - 1]), rad, cap)
end

# A spectral-quadrature sampling spans the sphere in latitude: its cells' measures total the sphere's
# area, so nothing lies past either end.
_spans_sphere(::FlowGeometries.Geometry.AbstractGeometry, _, ::Int) = false
_spans_sphere(::FlowGeometries.Geometry.AbstractSphericalGeometry, grid, d::Int) =
    d == 2 && FlowGeometries.Grids.sampling(grid) isa FlowGeometries.SphericalSampling.AbstractSpectralQuadratureSampling

# `(n_lo, n_hi, axis, faces)`: direction `d` continued by `n_lo` cells below its first and `n_hi` above
# its last, each at the spacing of the gap it continues. `faces = (lo_a, lo_b, hi_a, hi_b)` bounds each
# continued cell, the `k`-th away from its end at index `k`.
function _axis_extension(
    geo::FlowGeometries.Geometry.AbstractGeometry{T}, grid, d::Int, rad::T, cap::Bool,
) where {T<:AbstractFloat}
    x = FlowGeometries.Grids.coordinates(grid, d)
    n = length(x)
    n_lo, n_hi = _axis_counts(geo, grid, d, rad, cap)
    (n_lo == 0 && n_hi == 0) && return (0, 0, x, nothing)
    lo_c, lo_a, lo_b = _continued_cells(geo, d, T(x[1]), -T(x[2] - x[1]), n_lo)
    hi_c, hi_a, hi_b = _continued_cells(geo, d, T(x[n]), T(x[n] - x[n - 1]), n_hi)
    axis = vcat(reverse(lo_c), T.(collect(x)), hi_c)
    return (n_lo, n_hi, axis, (lo_a, lo_b, hi_a, hi_b))
end

# `(centres, a, b)` of the `K` cells stepping `h` away from the end at `x0`: centres `x0 + k h` with faces
# `x0 + (k ∓ 1/2) h`. Where no lattice centre lies past the last one, short of a pole or the origin, the
# last cell reaches that limit and is centred on its span.
function _continued_cells(geo::FlowGeometries.Geometry.AbstractGeometry{T}, d::Int, x0::T, h::T, K::Int) where {T}
    c = [x0 + k * h for k in 1:K]
    inner = [x0 + (k - T(1) / 2) * h for k in 1:K]
    outer = [x0 + (k + T(1) / 2) * h for k in 1:K]
    lim = _cut_limit(geo, d, h)
    if K > 0 && lim !== nothing && !_short_of(x0 + (K + 1) * h, lim, h)
        outer[K] = lim
        c[K] = (inner[K] + outer[K]) / 2
    end
    return (c, min.(inner, outer), max.(inner, outer))
end

# Whether `v` lies short of `lim` stepping `h`.
@inline _short_of(v, lim, h) = h > 0 ? v < lim : v > lim

# Where direction `d` ends when stepping `h`: a pole in latitude, the origin in radius toward the centre.
_cut_limit(::FlowGeometries.Geometry.AbstractGeometry, ::Int, h) = nothing
function _cut_limit(::FlowGeometries.Geometry.AbstractSphericalGeometry, d::Int, h::T) where {T}
    d == 2 && return h > 0 ? T(π) / 2 : -T(π) / 2
    (d == 3 && h < 0) && return zero(T)
    return nothing
end

# A Cartesian direction reaches as far as the kernel does.
_extension_counts(
    ::FlowGeometries.Geometry.AbstractCartesianGeometry{T}, ::Int, ::AbstractVector, s_lo::T, s_hi::T, rad::T,
    ::Bool,
) where {T<:AbstractFloat} = (ceil(Int, rad / abs(s_lo)), ceil(Int, rad / abs(s_hi)))

# Longitude fills the rest of the ring, half past each end, with a seam gap between a half and one and
# a half cells, which the periodic continued axis absorbs into the seam cells. Latitude continues toward
# the pole it heads for, and the radius toward the origin, as far as the kernel reaches.
function _extension_counts(
    geo::FlowGeometries.Geometry.AbstractSphericalGeometry{T}, d::Int, x, s_lo::T, s_hi::T, rad::T,
    cap::Bool,
) where {T<:AbstractFloat}
    n = length(x)
    if d == 1
        s = (abs(s_lo) + abs(s_hi)) / 2
        m = max(0, floor(Int, (T(2π) - abs(T(x[n] - x[1]))) / s - T(1) / 2))
        return (m ÷ 2, m - m ÷ 2)
    end
    step = d == 2 ? T(FlowGeometries.Geometry.radius(geo)) : one(T)
    reach(s) = ceil(Int, rad / (step * abs(s)))
    return (_continuation(d, T(x[1]), -s_lo, reach(s_lo), cap), _continuation(d, T(x[n]), s_hi, reach(s_hi), cap))
end

# Cells stepping `h` away from the end at `x0`, at most `kmax`: one per lattice centre short of the pole
# it heads for, or of the origin toward the centre, and, given `cap`, one when the end cell's own face
# stops short of that limit with no centre left before it. A radius stepping outward continues as far
# as the kernel reaches.
function _continuation(d::Int, x0::T, h::T, kmax::Int, cap::Bool) where {T<:AbstractFloat}
    (d == 3 && h > 0) && return kmax
    lim = d == 2 ? (h > 0 ? T(π) / 2 : -T(π) / 2) : zero(T)
    k = 0
    while k < kmax && _short_of(x0 + (k + 1) * h, lim, h)
        k += 1
    end
    (cap && k == 0 && kmax > 0 && _short_of(x0 + h / 2, lim, h)) && return 1
    return k
end
