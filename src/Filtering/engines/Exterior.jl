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
    _exterior_mass(grid, kernel, scale, execution = SerialBackend()) -> Array or nothing

`Σ kernel_weight(d) · measure` over the exterior cells within the kernel's radius of each cell of
`grid`, or `nothing` when no cell reaches past a bounded edge; rows are divided among `execution`'s threads
where it has them.
"""
function _exterior_mass(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, kernel::Kernels.AbstractFilterKernel, scale::T,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}, N}
    dim = _kernel_dim(grid)
    rad = _query_radius(grid, kernel, scale)
    lattice = _exterior_lattice(grid, rad)
    lattice === nothing && return nothing
    ext, lo, extended = lattice
    dims = FlowGeometries.Grids.size_tuple(grid)
    # A periodic direction 1 uniform around its wrap, which the continuation leaves alone, maps the
    # continued lattice onto itself under a shift along it, so every target of a row has the row's first
    # target's mass.
    ring = !first(extended) && FlowGeometries.Grids.isperiodic(grid, 1) && FlowGeometries.Grids.isuniform(grid, 1)
    (G <: FlowGeometries.Geometry.AbstractSphericalGeometry && first(extended) &&
     _rows_shared(grid, rad, (ext, lo, extended, _query_topology(ext, rad)))) &&
        return _exterior_mass_rows(ext, lo, dims, abs(T(step(FlowGeometries.Grids.coordinates(grid, 1)))),
                                   kernel, scale, rad, dim, _build_share(execution))
    return _exterior_mass(ext, lo, extended, dims, kernel, scale, rad, dim, ring)
end

# A `HighOrderKernel`'s weights past the edge are those of each cell's own correction.
function _exterior_mass(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, kernel::Kernels.HighOrderKernel, scale::T,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}, N}
    rad = _query_radius(grid, kernel, scale)
    lattice = _ho_lattice(grid, rad)
    lattice === nothing && return nothing
    mt = _query_topology(grid, rad)
    fold = (f, acc, I, _) -> FlowGeometries.Connectivity.fold_within(
        f, acc, grid, I...;
        ball = rad, self = true, active_only = false, topology = mt,
        images = FlowGeometries.Connectivity.NearestImage(),
    )
    _, exterior, _ = _highorder_targets(grid, kernel, scale, rad, fold, Returns(nothing), lattice, false, false,
                                        _build_share(execution))
    return exterior
end

function _exterior_mass(
    ext::FlowGeometries.Grids.StructuredGrid{T,G,N}, lo::NTuple{N,Int}, extended::NTuple{N,Bool},
    dims::NTuple{N,Int}, kernel::Kernels.AbstractFilterKernel, scale::T, rad::T, dim::Val, ring::Bool,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}, N}
    mt = _query_topology(ext, rad)
    images = _image_convention(ext)
    E = zeros(T, dims)
    reached = false
    targets = ring ? CartesianIndices((1:1, Base.tail(map(n -> 1:n, dims))...)) : CartesianIndices(dims)
    for I in targets
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
        if ring
            @inbounds for i in 1:dims[1]
                E[i, Base.tail(Tuple(I))...] = m
            end
        else
            @inbounds E[I] = m
        end
        reached |= !iszero(m)
    end
    return reached ? E : nothing
end

# A regional box of longitude step `dλ` whose continued ring no support reaches round to its seam
# (`_rows_shared`): every target of row `j` sees a source row's cells at the same offsets `±k`, of one weight
# `g(k)` each, so each pair of rows forms `g` once, out to the support's edge, and its tails
# `t(k) = Σ_{k′ ≥ k} g(k′)` summed from the far end inward. A grid row's cells past the box lie at the offsets
# `≤ −i` and `≥ Nx − i + 1` of target column `i`; a continued latitude row is exterior whole. Each row writes its
# own column of the result, so the rows are divided among `execution`'s threads.
function _exterior_mass_rows(
    ext::FlowGeometries.Grids.StructuredGrid{T,G,2}, lo::NTuple{2,Int}, dims::NTuple{2,Int}, dλ::T,
    kernel::Kernels.AbstractFilterKernel, scale::T, rad::T, dim::Val, execution,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    E = zeros(T, dims)
    mt = _query_topology(ext, rad)
    _share_run(execution, 1:dims[2]) do js
        tail = T[]
        for j in js
            _exterior_row!(view(E, :, j), tail, ext, lo, dims, dλ, kernel, scale, rad, dim, mt, j)
        end
    end
    return any(!iszero, E) ? E : nothing
end

function _exterior_row!(
    Ej::AbstractVector{T}, tail::Vector{T}, ext, lo::NTuple{2,Int}, dims::NTuple{2,Int}, dλ::T, kernel, scale::T,
    rad::T, dim::Val, mt, j::Int,
) where {T}
    Nx, Ny = dims
    nφ = FlowGeometries.Grids.size_tuple(ext)[2]
    geo = FlowGeometries.Grids.grid_geometry(ext)
    φ = FlowGeometries.Grids.coordinates(ext, 2)
    ie = 1 + lo[1]
    je = j + lo[2]
    dl, djl = FlowGeometries.Connectivity.metric_window(ext, (ie, je), rad, mt)
    for jje in max(1, je - djl):min(nφ, je + djl)
        A = T(FlowGeometries.Grids.area(ext, ie, jje))
        empty!(tail)
        for k in 1:dl
            inside, w = _pair_weight(geo, kernel, scale, rad, dim, T(φ[je]), T(φ[jje]), T(k) * dλ)
            inside || break
            push!(tail, w * A)
        end
        n = length(tail)
        @inbounds for k in (n - 1):-1:1
            tail[k] += tail[k + 1]
        end
        if lo[2] < jje <= lo[2] + Ny
            @inbounds for i in 1:min(Nx, n)
                Ej[i] += tail[i]
            end
            @inbounds for i in max(1, Nx - n + 1):Nx
                Ej[i] += tail[Nx - i + 1]
            end
        else
            g0 = last(_pair_weight(geo, kernel, scale, rad, dim, T(φ[je]), T(φ[jje]), zero(T))) * A
            Ej .+= n > 0 ? g0 + T(2) * @inbounds(tail[1]) : g0
        end
    end
    return Ej
end

# `(inside, w)` of a source cell `Δλ` east of a target, at latitudes `φt` and `φs`: whether it lies within `rad`,
# and the kernel's weight at their distance, the cell's measure left out. Distance grows with `|Δλ|` up to `π`.
@inline function _pair_weight(geo, kernel, scale::T, rad::T, dim::Val, φt::T, φs::T, Δλ::T) where {T}
    d = T(FlowGeometries.Geometry.distance(geo, SA.SVector{2,T}(zero(T), φt), SA.SVector{2,T}(Δλ, φs)))
    d <= rad || return false, zero(T)
    return true, T(Kernels.kernel_weight(kernel, d, scale, dim))
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
    rad = _query_radius(grid, kernel, scale)
    geo = FlowGeometries.Grids.grid_geometry(grid)
    return any(ntuple(d -> _axis_counts(geo, grid, d, rad, true) != (0, 0), Val(N)))
end

"""
    _exterior_lattice(grid, rad; cap = true) -> (ext, lo, extended) or nothing

`grid`'s lattice continued past each bounded edge far enough to hold every exterior cell within `rad`
of a cell of `grid`, as a grid of the same geometry and closure; `lo[d]` exterior cells precede the
grid's own along direction `d`, and `extended[d]` says whether direction `d` gained any.

`cap = false` continues each direction by the rows centred on its lattice only, leaving out the cap row
between an end face and the pole.
"""
function _exterior_lattice(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, rad::T; cap::Bool = true,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}, N}
    geo = FlowGeometries.Grids.grid_geometry(grid)
    parts = ntuple(d -> _axis_extension(geo, grid, d, rad, cap), Val(N))
    extended = map(p -> p[1] > 0 || p[2] > 0, parts)
    any(extended) || return nothing
    closure = map(v -> _ext_closure(geo, grid, v), ntuple(Val, Val(N)))
    ext = FlowGeometries.Grids.StructuredGrid(
        geo, map(p -> p[3], parts)...; topology = map(first, closure), period = map(last, closure),
    )
    return _exact_continued_measure(ext, geo, grid, parts), map(p -> p[1], parts), extended
end

# `ext` with each continued cell's measure factor the exact integral between its own faces; a cell cut at
# a pole or the origin has faces off the midpoints between centres. The grid's own columns keep the
# grid's longitude factors, so the columns beside a regional box's seam leave its gap uncovered.
_exact_continued_measure(ext, ::FlowGeometries.Geometry.AbstractGeometry, _, _) = ext

function _exact_continued_measure(
    ext::FlowGeometries.Grids.StructuredGrid{T,G,N}, geo::FlowGeometries.Geometry.AbstractLonLatGeometry{T},
    grid::FlowGeometries.Grids.StructuredGrid, parts,
) where {T<:AbstractFloat, G, N}
    N >= 2 || return ext
    factors = FlowGeometries.Grids.measure_factors(ext)
    own = FlowGeometries.Grids.measure_factors(grid)
    exact = ntuple(Val(N)) do d
        n_lo, n_hi, axis, faces = parts[d]
        f = collect(T, factors[d])
        if d == 1
            @inbounds for i in 1:(length(axis) - n_lo - n_hi)
                f[n_lo + i] = T(own[1][i])
            end
        end
        (d == 1 || d > 3) && return f
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

# A spheroid with heights: FlowGeometries' slab of `∫∫ (N + h)(M + h) cos φ dφ dh` over latitude and height, coupled,
# the grid's own entries kept and each continued row's or level's taken between its own faces.
function _exact_continued_measure(
    ext::FlowGeometries.Grids.StructuredGrid{T,G,3}, geo::FlowGeometries.Geometry.AbstractEllipsoidalGeometry{T},
    grid::FlowGeometries.Grids.StructuredGrid, parts,
) where {T<:AbstractFloat, G}
    own = FlowGeometries.Grids.measure(grid)
    lead = collect(T, FlowGeometries.Grids.measure(ext).lead)
    for i in eachindex(own.lead)
        lead[parts[1][1] + i] = T(own.lead[i])
    end
    fφ, fh = _continued_faces(grid, parts[2], 2, T), _continued_faces(grid, parts[3], 3, T)
    jo, ko = parts[2][1], parts[3][1]
    nφ, nh = size(own.slab)
    slab = [(jo < j <= jo + nφ && ko < k <= ko + nh) ? T(own.slab[j - jo, k - ko]) : _slab_factor(geo, fφ[j]..., fh[k]...)
            for j in eachindex(fφ), k in eachindex(fh)]
    return FlowGeometries.Grids.rebuild(ext, (measure = FlowGeometries.Grids.SlabMeasure(lead, slab, ()),))
end

# The faces `(a, b)` of each cell of direction `d` of the continued lattice: the grid's own, and the continued ones'
# from `_axis_extension`, the `k`-th away from an end at index `k`.
function _continued_faces(grid, part, d::Int, ::Type{T}) where {T}
    n_lo, n_hi, _, (lo_a, lo_b, hi_a, hi_b) = part
    f = _cell_faces(grid, d, T)
    n = length(f) - 1
    return vcat([(lo_a[k], lo_b[k]) for k in n_lo:-1:1], [(f[i], f[i + 1]) for i in 1:n], [(hi_a[k], hi_b[k]) for k in 1:n_hi])
end

# `∫ cos φ (N M Δh + (N + M) Δ(h²)/2 + Δ(h³)/3) dφ` over `[φa, φb] × [ha, hb]`, as FlowGeometries measures a geodetic
# cell: exact in height, 8-point Gauss–Legendre in latitude.
function _slab_factor(geo::FlowGeometries.Geometry.AbstractEllipsoidalGeometry{T}, φa::T, φb::T, ha::T, hb::T) where {T}
    μ, w = _gl8(T)
    dh = hb - ha
    h2, h3 = dh * (ha + hb) / 2, dh * (ha * ha + ha * hb + hb * hb) / 3
    hφ, cφ = (φb - φa) / 2, (φa + φb) / 2
    s = zero(T)
    for k in 1:8
        φ = cφ + hφ * μ[k]
        N, M = T(FlowGeometries.Geometry.prime_vertical_radius(geo, φ)), T(FlowGeometries.Geometry.meridional_radius(geo, φ))
        s += w[k] * cos(φ) * (N * M * dh + (N + M) * h2 + h3)
    end
    return hφ * s
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

# A spheroid's, as FlowGeometries measures its cells: latitude `∫ M(φ)N(φ)cos φ dφ`, along a single meridian its arc
# `∫ M(φ) dφ`, by 8-point Gauss–Legendre.
@inline function _cell_factor(
    geo::FlowGeometries.Geometry.AbstractEllipsoidalGeometry{T}, ::Int, ::Int, nλ::Int, a::T, b::T,
) where {T<:AbstractFloat}
    μ, w = _gl8(T)
    h, c = (b - a) / 2, (a + b) / 2
    s = zero(T)
    for k in 1:8
        φ = c + h * μ[k]
        M = T(FlowGeometries.Geometry.meridional_radius(geo, φ))
        s += w[k] * (nλ == 1 ? M : M * T(FlowGeometries.Geometry.prime_vertical_radius(geo, φ)) * cos(φ))
    end
    return h * s
end

_gl8(::Type{T}) where {T} = (r = FlowTransformBindings.gauss_legendre(T, 8); (r.μ, r.w))

# `(topology, period)` of direction `d` of the continued lattice: the grid's own, except a spherical
# longitude, which identifies, so the lattice closes it with period 2π whether or not the grid's cells
# reach round. A single meridian keeps its arc-length measure: `_exact_continued_measure` takes the
# grid's own longitude factors.
_ext_closure(::FlowGeometries.Geometry.AbstractLonLatGeometry{T}, grid, ::Val{1}) where {T} =
    (FlowGeometries.Grids.Periodic(), T(2π))
_ext_closure(::FlowGeometries.Geometry.AbstractGeometry, grid, ::Val{d}) where {d} =
    _closure(FlowGeometries.Grids.topology(grid, d), grid, d)

_closure(t::FlowGeometries.Grids.Periodic, grid, d::Int) = (t, FlowGeometries.Grids.period(grid, d))
_closure(t::FlowGeometries.Grids.Bounded, grid, d::Int) = (t, nothing)

# `(n_lo, n_hi)`: how many cells direction `d` continues by below its first and above its last.
function _axis_counts(
    geo::FlowGeometries.Geometry.AbstractGeometry{T}, grid, d::Int, rad::T, cap::Bool,
) where {T<:AbstractFloat}
    x = FlowGeometries.Grids.coordinates(grid, d)
    n = length(x)
    (FlowGeometries.Grids.isperiodic(grid, d) || n < 2 || _spans_sphere(geo, grid, d)) && return (0, 0)
    return _extension_counts(geo, d, x, T(x[2] - x[1]), T(x[n] - x[n - 1]), rad, cap)
end

# A grid that covers the sphere spans it in latitude, so nothing lies past either end.
_spans_sphere(::FlowGeometries.Geometry.AbstractGeometry, _, ::Int) = false
_spans_sphere(::FlowGeometries.Geometry.AbstractLonLatGeometry, grid, d::Int) =
    d == 2 && FlowGeometries.Grids.covers_sphere(grid)

# `(n_lo, n_hi, axis, faces)`: direction `d` continued by `n_lo` cells below its first and `n_hi` above
# its last, each at the spacing of the gap it continues. `faces = (lo_a, lo_b, hi_a, hi_b)` bounds each
# continued cell, the `k`-th away from its end at index `k`. Every direction is a `Vector`, so the
# lattice's type follows from the grid's.
function _axis_extension(
    geo::FlowGeometries.Geometry.AbstractGeometry{T}, grid, d::Int, rad::T, cap::Bool,
) where {T<:AbstractFloat}
    x = FlowGeometries.Grids.coordinates(grid, d)
    n = length(x)
    n_lo, n_hi = _axis_counts(geo, grid, d, rad, cap)
    (n_lo == 0 && n_hi == 0) && return (0, 0, collect(T, x), (T[], T[], T[], T[]))
    lo_c, lo_a, lo_b = _continued_cells(geo, d, T(x[1]), -T(x[2] - x[1]), n_lo)
    hi_c, hi_a, hi_b = _continued_cells(geo, d, T(x[n]), T(x[n] - x[n - 1]), n_hi)
    axis = vcat(reverse(lo_c), collect(T, x), hi_c)
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

# The least ground distance per radian of latitude: the sphere's radius, a spheroid's meridional radius at the
# equator, so a latitude continuation by it reaches at least as far as the kernel.
_least_meridian_radius(geo::FlowGeometries.Geometry.AbstractSphericalGeometry{T}) where {T} =
    T(FlowGeometries.Geometry.radius(geo))
_least_meridian_radius(geo::FlowGeometries.Geometry.AbstractEllipsoidalGeometry{T}) where {T} =
    T(FlowGeometries.Geometry.meridional_radius(geo, zero(T)))

# Where direction `d` ends when stepping `h`: a pole in latitude, the origin in radius toward the centre. A
# spheroid's third direction is a height, which ends nowhere.
_cut_limit(::FlowGeometries.Geometry.AbstractGeometry, ::Int, h) = nothing
function _cut_limit(::FlowGeometries.Geometry.AbstractLonLatGeometry, d::Int, h::T) where {T}
    d == 2 && return h > 0 ? T(π) / 2 : -T(π) / 2
    (d == 3 && h < 0) && return zero(T)
    return nothing
end
_cut_limit(::FlowGeometries.Geometry.AbstractEllipsoidalGeometry, d::Int, h::T) where {T} =
    d == 2 ? (h > 0 ? T(π) / 2 : -T(π) / 2) : nothing

# A Cartesian direction reaches as far as the kernel does.
_extension_counts(
    ::FlowGeometries.Geometry.AbstractCartesianGeometry{T}, ::Int, ::AbstractVector, s_lo::T, s_hi::T, rad::T,
    ::Bool,
) where {T<:AbstractFloat} = (ceil(Int, rad / abs(s_lo)), ceil(Int, rad / abs(s_hi)))

# Longitude fills the rest of the ring, half past each end, with a seam gap between a half and one and
# a half cells, which the periodic continued axis absorbs into the seam cells. Latitude continues toward
# the pole it heads for, and the radius toward the origin, as far as the kernel reaches.
function _extension_counts(
    geo::FlowGeometries.Geometry.AbstractLonLatGeometry{T}, d::Int, x, s_lo::T, s_hi::T, rad::T,
    cap::Bool,
) where {T<:AbstractFloat}
    n = length(x)
    if d == 1
        s = (abs(s_lo) + abs(s_hi)) / 2
        m = max(0, floor(Int, (T(2π) - abs(T(x[n] - x[1]))) / s - T(1) / 2))
        return (m ÷ 2, m - m ÷ 2)
    end
    step = d == 2 ? _least_meridian_radius(geo) : one(T)
    reach(s) = ceil(Int, rad / (step * abs(s)))
    return (_continuation(_cut_limit(geo, d, -s_lo), T(x[1]), -s_lo, reach(s_lo), cap),
            _continuation(_cut_limit(geo, d, s_hi), T(x[n]), s_hi, reach(s_hi), cap))
end

# Cells stepping `h` away from the end at `x0`, at most `kmax`: one per lattice centre short of the limit `lim` it
# heads for ([`_cut_limit`](@ref)), and, given `cap`, one when the end cell's own face stops short of that limit
# with no centre left before it; as far as the kernel reaches where nothing limits the direction.
function _continuation(lim, x0::T, h::T, kmax::Int, cap::Bool) where {T<:AbstractFloat}
    lim === nothing && return kmax
    k = 0
    while k < kmax && _short_of(x0 + (k + 1) * h, lim, h)
        k += 1
    end
    (cap && k == 0 && kmax > 0 && _short_of(x0 + h / 2, lim, h)) && return 1
    return k
end
