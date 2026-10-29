# Independent references for the filter, shared by the topic files. Each is written from the definition
# with FlowGeometries' distances and measures, and none calls an engine of the package.

const _CGEF_SERIAL = CGEF.ComputationalBackends.SerialBackend()
# TODO :: These don't seem to be refernece filters, move them to where they belong
const _CGEF_EXTENSIONS = (
    :CoarseGrainingEnergyFluxesAbstractFFTsExt, :CoarseGrainingEnergyFluxesFFTWExt,
    :CoarseGrainingEnergyFluxesAbstractFFTsDistributedExt, :CoarseGrainingEnergyFluxesFastSphericalHarmonicsExt,
    :CoarseGrainingEnergyFluxesNUFSHTExt, :CoarseGrainingEnergyFluxesOhMyThreadsExt,
    :CoarseGrainingEnergyFluxesDistributedExt, :CoarseGrainingEnergyFluxesKernelAbstractionsExt,
    :CoarseGrainingEnergyFluxesKernelAbstractionsAbstractFFTsExt, :CoarseGrainingEnergyFluxesMPIExt,
    :CoarseGrainingEnergyFluxesSpecialFunctionsExt,
)

# A device reduces each energy in its own order. The summands `½|ū|²·A` are non-negative, so two orders
# of a sum of `n` of them lie within `2n·eps` of each other, relative. The spectrum sums `ū·∂_ℓū·A`, the same
# terms on either side but of either sign, so the two orders agree to `2n·eps` of `Σ|ū·∂_ℓū|·A`, held here by
# `1e-10` of the sum.
_cgef_device_energies(a, b, n) =
    isapprox(collect(a.cumulative_energy), collect(b.cumulative_energy); rtol = 2n * eps()) &&
    isapprox(collect(a.filtering_spectrum), collect(b.filtering_spectrum); rtol = 1e-10)

# The exterior of the domain: the lattice continued past each bounded end at the edge gap, `K` cells
# along a Cartesian direction. A regional longitude continues around the rest of the ring, half past
# each end, until the seam gap lies in `[s/2, 3s/2)` of the mean edge gap `s`, and closes with period
# 2π however few cells it gains. A latitude continues toward the pole it heads for with one row per lattice centre strictly short
# of the pole, the last row reaching the pole and centred on its span, or with one cap row from the end
# cell's face to the pole when no centre is short of it; a spectral-quadrature sampling spans the sphere
# and continues in neither. Built as a grid of its own for the distances.
# Returns it, the offset of the original cells in it, and the faces `(a, b)` of each continued latitude
# row (`nothing` for the grid's own rows, and for every row when latitude is not continued).
function _cgef_extended_lattice(grid, K)
    geo = FG.Grids.grid_geometry(grid)
    sph = geo isa FG.Geometry.AbstractLonLatGeometry
    N = length(FG.Grids.size_tuple(grid))
    sph && N != 2 && error("the reference lattice covers two-dimensional spherical grids only")
    lo = zeros(Int, N)
    ring = falses(N)
    rows = nothing
    axes = map(1:N) do d
        x = collect(FG.Grids.coordinates(grid, d))
        n = length(x)
        (FG.Grids.isperiodic(grid, d) || n < 2) && return x
        sl, sh = x[2] - x[1], x[n] - x[n-1]
        if sph && d == 1
            s = (abs(sl) + abs(sh)) / 2
            G = 2π - abs(x[n] - x[1])
            m = 0
            while G - (m + 1) * s >= s / 2
                m += 1
            end
            ring[d] = true
            lo[d] = m ÷ 2
            return vcat([x[1] - k * sl for k in (m ÷ 2):-1:1], x, [x[n] + k * sh for k in 1:(m - m ÷ 2)])
        elseif sph && d == 2
            FG.Grids.covers_sphere(grid) && return x
            below = _cgef_polar_rows(x[1], -sl, K)
            above = _cgef_polar_rows(x[n], sh, K)
            lo[d] = length(below)
            if !isempty(below) || !isempty(above)
                rows = vcat(reverse(last.(below)), fill(nothing, n), last.(above))
            end
            return vcat(reverse(first.(below)), x, first.(above))
        end
        lo[d] = K
        return vcat([x[1] - k * sl for k in K:-1:1], x, [x[n] + k * sh for k in 1:K])
    end
    top = ntuple(d -> ring[d] ? FG.Grids.Periodic() : FG.Grids.topology(grid, d), N)
    per = ntuple(d -> ring[d] ? 2π : FG.Grids.isperiodic(grid, d) ? FG.Grids.period(grid, d) : nothing, N)
    ext = FG.Grids.StructuredGrid(geo, axes...; topology = top, period = per)
    return ext, Tuple(lo), rows
end

# `(centre, (a, b))` of the latitude rows continued from the end at `x0` in steps `h` toward the pole, at
# most `K`.
function _cgef_polar_rows(x0, h, K)
    pole = h > 0 ? π / 2 : -π / 2
    short(v) = h > 0 ? v < pole : v > pole
    rows = Tuple{Float64,Tuple{Float64,Float64}}[]
    k = 0
    while k < K && short(x0 + (k + 1) * h)
        k += 1
        push!(rows, (x0 + k * h, minmax(x0 + (k - 0.5) * h, x0 + (k + 0.5) * h)))
    end
    if k == 0
        inner = x0 + h / 2
        return short(inner) ? [((inner + pole) / 2, minmax(inner, pole))] : rows
    end
    if !short(x0 + (k + 1) * h)
        inner = x0 + (k - 0.5) * h
        rows[k] = ((inner + pole) / 2, minmax(inner, pole))
    end
    return rows
end

# The measure of cell `J` of the continued lattice: the grid's own inside it; outside, the longitude
# width (the grid's own for its columns, the lattice's past them) times the latitude integral over the
# row, the grid's own factor for its rows and the zone's ([`_cgef_zone`](@ref); the meridian's length along a
# single meridian) for a continued row between faces `a < b`.
function _cgef_lattice_measure(grid, ext, lo, rows, J)
    sz = FG.Grids.size_tuple(grid)
    Jd = ntuple(d -> J[d] - lo[d], length(sz))
    all(d -> 1 <= Jd[d] <= sz[d], 1:length(sz)) && return FG.Grids.measure(grid, Jd...)
    rows === nothing && return FG.Grids.measure(ext, J...)
    λw = 1 <= Jd[1] <= sz[1] ? FG.Grids.measure_factors(grid)[1][Jd[1]] : FG.Grids.measure_factors(ext)[1][J[1]]
    rows[J[2]] === nothing && return λw * FG.Grids.measure_factors(grid)[2][Jd[2]]
    a, b = rows[J[2]]
    geo = FG.Grids.grid_geometry(grid)
    return length(FG.Grids.coordinates(ext, 1)) == 1 ? λw * _cgef_meridian(geo, a, b) : λw * _cgef_zone(geo, a, b)
end

# The area per unit longitude between latitudes `a < b`, `∫ M N cos φ dφ`: `R²(sin b − sin a)` on a sphere, and on a
# spheroid `a²(1 − e²)[s/(2(1 − e²s²)) + atanh(e s)/(2e)]` between `s = sin a` and `sin b`; and the meridian's length.
_cgef_zone(geo::FG.Geometry.AbstractSphericalGeometry, a, b) = FG.Geometry.radius(geo)^2 * (sin(b) - sin(a))
function _cgef_zone(geo::FG.Geometry.AbstractEllipsoidalGeometry, a, b)
    A, e = FG.Geometry.semimajor_axis(geo), sqrt(FG.Geometry.eccentricity²(geo))
    F(φ) = (s = sin(φ); s / (2 * (1 - e^2 * s^2)) + atanh(e * s) / (2e))
    return A^2 * (1 - e^2) * (F(b) - F(a))
end
_cgef_meridian(geo::FG.Geometry.AbstractSphericalGeometry, a, b) = FG.Geometry.radius(geo) * (b - a)
_cgef_meridian(geo::FG.Geometry.AbstractEllipsoidalGeometry, a, b) =
    FG.Geometry.meridian_distance(geo, b) - FG.Geometry.meridian_distance(geo, a)

# The geodesic disk of radius `R` about latitude `t` on a spheroid, through FlowGeometries' geodesic inverse.
function _cgef_geodesic_disk(geo, R, t)
    inverse = (λ, φ) -> FG.Geometry._geodesic_inverse(geo, (λ, φ), (0.0, t))
    metric = φ -> (FG.Geometry.meridional_radius(geo, φ), FG.Geometry.prime_vertical_radius(geo, φ) * cos(φ))
    return CGEF.Geometry.geodesic_disk(inverse, metric, Float64(R), Float64(t), Float64(FG.Geometry.semimajor_axis(geo)))
end

# The filter's definition summed directly (Grooms et al. 2021 eq. 7; Aluie et al. 2018). `ZeroFill`
# filters the field extended by zero over masked cells and past each bounded edge, weighted `G(d)·A`
# and normalized by the kernel's mass over that continued lattice; `Deformable` sums and normalizes over
# the active in-domain cells and zeroes an inactive target. On a periodic Cartesian direction every image
# inside the support counts; a spherical longitude identifies, so its cells come once.
function _cgef_direct_filter(grid, kernel, ℓ, f, strategy = CGEF.Filtering.ZeroFill())
    geo = FG.Grids.grid_geometry(grid)
    cart = geo isa FG.Geometry.CartesianGeometry
    sz = FG.Grids.size_tuple(grid)
    N = length(sz)
    rad = CGEF.Kernels.kernel_radius(kernel, ℓ, Val(N))
    hmin = minimum(d -> FG.Grids.minimum_spacing(grid, d), 1:N) * (cart ? 1.0 : CGEF.Filtering._least_meridian_radius(geo))
    ext, lo, rows = _cgef_extended_lattice(grid, ceil(Int, rad / hmin) + 2)
    tiles = ntuple(d -> cart && FG.Grids.isperiodic(grid, d), N)
    P = ntuple(d -> tiles[d] ? Float64(FG.Grids.period(grid, d)) : 0.0, N)
    M = ntuple(d -> tiles[d] ? ceil(Int, rad / P[d]) + 1 : 0, N)
    mask = BitArray(FG.Grids.mask(grid))
    cells = CartesianIndices(FG.Grids.size_tuple(ext))
    xe = [NTuple{N,Float64}(Tuple(FG.Grids.coords(ext, Tuple(J)...))) for J in cells]
    me = [Float64(_cgef_lattice_measure(grid, ext, lo, rows, Tuple(J))) for J in cells]
    # The field on the continued lattice, and whether each of its cells is an active grid cell.
    fe = zeros(size(cells))
    ae = falses(size(cells))
    for J in cells
        Jd = ntuple(d -> J[d] - lo[d], N)
        if all(d -> 1 <= Jd[d] <= sz[d], 1:N) && mask[Jd...]
            ae[J] = true
            fe[J] = f[Jd...]
        end
    end
    xs = [NTuple{N,Float64}(Tuple(FG.Grids.coords(grid, Tuple(I)...))) for I in CartesianIndices(sz)]
    images = [ntuple(d -> m[d] * P[d], N) for m in CartesianIndices(ntuple(d -> (-M[d]):M[d], N))]
    return _cgef_direct_sum(geo, kernel, Float64(ℓ), Float64(rad), Val(N), xs, mask, xe, me, fe, ae, vec(images),
                            strategy isa CGEF.Filtering.ZeroFill)
end

# A `CellOverlap` top-hat: every cell of the grid, and every periodic image of a Cartesian one, weighted by its
# measure times the fraction of it inside the support (the disk, segment or ball of radius `ℓ/2`; on a sphere
# the cap, for a single ring of longitude its arc of the equator, and with a radius the ball of the chord; on a
# spheroid the geodesic disk), the cells spanning FlowGeometries' faces, latitude faces clamped at the poles and
# radius faces at the origin. `ZeroFill` divides by the support's measure where a window leaves the grid, the
# kernel normalized over all space; elsewhere by the summed weights.
function _cgef_direct_filter(grid, kernel::CGEF.TopHatKernel{CGEF.Kernels.CellOverlap}, ℓ, f,
                             strategy = CGEF.Filtering.ZeroFill(), n::Integer = 10)
    geo = FG.Grids.grid_geometry(grid)
    geo isa FG.Geometry.AbstractEllipsoidalGeometry && return _cgef_geodesic_filter(grid, ℓ, f, strategy, n)
    cart = geo isa FG.Geometry.CartesianGeometry
    dims = FG.Grids.size_tuple(grid)
    N = length(dims)
    R = ℓ / 2
    a = cart ? 1.0 : FG.Geometry.radius(geo)
    θ = min(R / a, π)
    T = eltype(FG.Grids.coordinates(grid, 1))
    nodes = CGEF.Filtering._overlap_nodes(T, n)
    xs = map(d -> collect(T, FG.Grids.coordinates(grid, d)), 1:N)
    fs = map(1:N) do d
        p = FG.Grids.isperiodic(grid, d) ? FG.Grids.period(grid, d) : nothing
        v = collect(T, FG.Discretization.faces(xs[d], p))
        (!cart && d == 2) ? clamp.(v, -π / 2, π / 2) : (!cart && d == 3) ? max.(v, 0.0) : v
    end
    pd = map(d -> cart && FG.Grids.isperiodic(grid, d) ? T(FG.Grids.period(grid, d)) : 0.0, 1:N)
    sg = map(d -> length(xs[d]) > 1 && xs[d][end] < xs[d][1] ? -1 : 1, 1:N)
    ms = map(d -> pd[d] > 0 ? ceil(Int, 2R / pd[d]) + 1 : 0, 1:N)
    mask = BitArray(FG.Grids.mask(grid))
    zf = strategy isa CGEF.Filtering.ZeroFill
    # A sphere's grid of one column or one row: its cells are arcs of the meridian or the parallel through the
    # target, and the support is the arc of that circle inside the cap.
    meridian = !cart && N == 2 && dims[1] == 1
    parallel = !cart && N == 2 && dims[2] == 1
    ext = cart ? !all(d -> FG.Grids.isperiodic(grid, d), 1:N) :
          N == 1 ? !FG.Grids.isperiodic(grid, 1) : N == 3 || meridian || parallel ||
          !(FG.Grids.isperiodic(grid, 1) && FG.Grids.covers_sphere(grid))
    support(I) = cart || N == 3 ? (N == 1 ? 2R : N == 2 ? π * R^2 : 4π * R^3 / 3) :
                 N == 1 ? 2θ * a :
                 meridian ? a * (min(π / 2, xs[2][I[2]] + θ) - max(-π / 2, xs[2][I[2]] - θ)) :
                 parallel ? 2 * CGEF.Geometry.cap_half_width(xs[2][1], xs[2][1], θ) * a * cos(xs[2][1]) :
                 4π * a^2 * sin(θ / 2)^2
    function fraction(I, J, M)
        box = ntuple(d -> minmax(fs[d][J[d]] + sg[d] * M[d] * pd[d] - (cart || d == 1 ? xs[d][I[d]] : 0.0),
                                 fs[d][J[d] + 1] + sg[d] * M[d] * pd[d] - (cart || d == 1 ? xs[d][I[d]] : 0.0)), N)
        meridian && return CGEF.Geometry.interval_fraction(box[2][1] - xs[2][I[2]], box[2][2] - xs[2][I[2]], θ)
        parallel && return CGEF.Geometry.cap_arc_fraction(box[1]..., xs[2][1], xs[2][1], θ)
        cart || return N == 1 ? CGEF.Geometry.cap_arc_fraction(box[1]..., 0.0, 0.0, θ) :
            N == 2 ? CGEF.Geometry.cap_cell_fraction(box[1]..., box[2]..., xs[2][I[2]], θ, nodes) :
            CGEF.Geometry.shell_cell_fraction(box[1]..., box[2]..., box[3]..., xs[2][I[2]], xs[3][I[3]], R, nodes)
        N == 1 && return CGEF.Geometry.interval_fraction(box[1]..., R)
        N == 2 && return CGEF.Geometry.disk_fraction(box[1]..., box[2]..., R)
        return CGEF.Geometry.ball_box_fraction(box[1]..., box[2]..., box[3]..., R, nodes)
    end
    out = zeros(dims)
    for I in CartesianIndices(dims)
        (zf || mask[I]) || continue
        num = den = 0.0
        for J in CartesianIndices(dims), M in CartesianIndices(Tuple(-m:m for m in ms))
            w = FG.Grids.measure(grid, Tuple(J)...) * fraction(I, J, M)
            iszero(w) && continue
            mask[J] && (num += w * f[J])
            (zf || mask[J]) && (den += w)
        end
        out[I] = num / (zf && ext ? support(I) : den)
    end
    return out
end

# The `CellOverlap` top-hat on a 2-D spheroid grid: each cell's measure times its fraction inside the geodesic disk
# about the target, a grid of one row or one column taking the disk's arc of the parallel or meridian through it.
function _cgef_geodesic_filter(grid, ℓ, f, strategy, n)
    geo = FG.Grids.grid_geometry(grid)
    dims = FG.Grids.size_tuple(grid)
    length(dims) == 2 || error("the reference geodesic overlap covers two-dimensional spheroid grids only")
    nodes = CGEF.Filtering._overlap_nodes(Float64, n)
    xs = map(d -> collect(Float64, FG.Grids.coordinates(grid, d)), 1:2)
    fs = map(1:2) do d
        p = FG.Grids.isperiodic(grid, d) ? FG.Grids.period(grid, d) : nothing
        v = collect(Float64, FG.Discretization.faces(xs[d], p))
        d == 2 ? clamp.(v, -π / 2, π / 2) : v
    end
    disks = [_cgef_geodesic_disk(geo, ℓ / 2, t) for t in xs[2]]
    meridian, parallel = dims[1] == 1, dims[2] == 1
    ext = meridian || parallel || !(FG.Grids.isperiodic(grid, 1) && FG.Grids.covers_sphere(grid))
    function support(d)
        meridian && return CGEF.Geometry.geodesic_meridian_length(d, nodes)
        parallel && return 2 * CGEF.Geometry.geodesic_half_width(d, d.t) * last(d.metric(d.t))
        return CGEF.Geometry.geodesic_disk_area(d, nodes)
    end
    function fraction(I, J)
        d = disks[I[2]]
        meridian && return CGEF.Geometry.geodesic_meridian_fraction(d, minmax(fs[2][J[2]], fs[2][J[2] + 1])..., nodes)
        λa, λb = minmax(fs[1][J[1]] - xs[1][I[1]], fs[1][J[1] + 1] - xs[1][I[1]])
        parallel && return CGEF.Geometry.geodesic_arc_fraction(d, λa, λb)
        return CGEF.Geometry.geodesic_cell_fraction(d, λa, λb, minmax(fs[2][J[2]], fs[2][J[2] + 1])..., nodes)
    end
    mask = BitArray(FG.Grids.mask(grid))
    zf = strategy isa CGEF.Filtering.ZeroFill
    out = zeros(dims)
    for I in CartesianIndices(dims)
        (zf || mask[I]) || continue
        num = den = 0.0
        for J in CartesianIndices(dims)
            w = FG.Grids.measure(grid, Tuple(J)...) * fraction(I, J)
            iszero(w) && continue
            mask[J] && (num += w * f[J])
            (zf || mask[J]) && (den += w)
        end
        out[I] = num / (zf && ext ? support(disks[I[2]]) : den)
    end
    return out
end

function _cgef_direct_sum(geo, kernel, ℓ, rad, dim::Val{N}, xs, mask, xe, me, fe, ae, images, zf) where {N}
    out = zeros(size(xs))
    for I in eachindex(xs)
        (zf || mask[I]) || continue
        num = 0.0
        den = 0.0
        for J in eachindex(xe), s in images
            (zf || ae[J]) || continue
            dist = FG.Geometry.distance(geo, xs[I], ntuple(d -> xe[J][d] + s[d], N))
            dist <= rad || continue
            w = CGEF.Kernels.kernel_weight(CGEF.Kernels.point_kernel(kernel), dist, ℓ, dim) * me[J]
            den += w
            ae[J] && (num += w * fe[J])
        end
        out[I] = num / den
    end
    return out
end

# The definition over the grid's own cells, for a grid with no lattice to continue: a curvilinear mesh, a
# node set or a sphere pixelization. `ZeroFill` keeps every cell in the denominator and filters an
# inactive target; `Deformable` normalizes over the active cells and zeroes an inactive target. On a
# periodic Cartesian direction every image inside the support counts.
function _cgef_cellset_filter(grid, kernel, ℓ, f, strategy = CGEF.Filtering.ZeroFill())
    geo = FG.Grids.grid_geometry(grid)
    dim = Val(FG.Grids.ncoordinates(grid))
    rad = CGEF.Kernels.kernel_radius(kernel, ℓ, dim)
    cells = CartesianIndices(size(FG.Grids.mask(grid)))
    N = FG.Grids.ncoordinates(grid)
    pts = [NTuple{N,Float64}(Tuple(FG.Grids.coords(grid, Tuple(I)...))) for I in cells]
    act = BitArray([FG.Grids.isactive(grid, Tuple(I)...) for I in cells])
    meas = [Float64(FG.Grids.measure(grid, Tuple(I)...)) for I in cells]
    tiles = ntuple(d -> geo isa FG.Geometry.CartesianGeometry && FG.Grids.isperiodic(grid, d), N)
    P = ntuple(d -> tiles[d] ? Float64(FG.Grids.period(grid, d)) : 0.0, N)
    M = ntuple(d -> tiles[d] ? ceil(Int, rad / P[d]) + 1 : 0, N)
    images = [ntuple(d -> m[d] * P[d], N) for m in CartesianIndices(ntuple(d -> (-M[d]):M[d], N))]
    return _cgef_direct_sum(geo, kernel, Float64(ℓ), Float64(rad), dim, pts, act, pts, meas, Float64.(f), act,
                            vec(images), strategy isa CGEF.Filtering.ZeroFill)
end

# A `CellOverlap` top-hat on a curvilinear mesh: each cell's quadrilateral, from the grid's corners or
# FlowGeometries' reconstruction of them, weighted by its measure times its fraction inside the disk (straight
# edges) or the cap (great-circle edges); `ZeroFill` divides by the support's measure, the mesh having no
# lattice past its edge.
function _cgef_cellset_filter(grid::FG.Grids.CurvilinearGrid, kernel::CGEF.TopHatKernel{CGEF.Kernels.CellOverlap}, ℓ, f,
                              strategy = CGEF.Filtering.ZeroFill())
    geo = FG.Grids.grid_geometry(grid)
    cart = geo isa FG.Geometry.CartesianGeometry
    Nx, Ny = FG.Grids.size_tuple(grid)
    C = FG.Grids.coordinates(grid)
    xc, yc = FG.Grids.has_corners(grid) ? FG.Grids.corners(grid) : map(FG.Grids._centers_to_corners, C)
    a = cart ? 1.0 : FG.Geometry.radius(geo)
    R = ℓ / 2
    θ = min(R / a, π)
    mask = BitArray(FG.Grids.mask(grid))
    zf = strategy isa CGEF.Filtering.ZeroFill
    support = cart ? π * R^2 : 4π * a^2 * sin(θ / 2)^2
    out = zeros(Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        (zf || mask[i, j]) || continue
        px, py = C[1][i, j], C[2][i, j]
        num = den = 0.0
        for jj in 1:Ny, ii in 1:Nx
            cs = ((ii, jj), (ii + 1, jj), (ii + 1, jj + 1), (ii, jj + 1))
            fr = if cart
                CGEF.Geometry.disk_fraction([(xc[c...] - px, yc[c...] - py) for c in cs], R)
            else
                # Each vertex in the frame whose pole is the target.
                v(c) = (Δ = xc[c...] - px; φ = yc[c...];
                        (cos(φ) * sin(Δ), cos(py) * sin(φ) - sin(py) * cos(φ) * cos(Δ), sin(py) * sin(φ) + cos(py) * cos(φ) * cos(Δ)))
                CGEF.Geometry.cap_polygon_fraction(map(v, cs), θ)
            end
            w = FG.Grids.measure(grid, ii, jj) * fr
            iszero(w) && continue
            mask[ii, jj] && (num += w * f[ii, jj])
            (zf || mask[ii, jj]) && (den += w)
        end
        out[i, j] = num / (zf ? support : den)
    end
    return out
end

_cgef_relerr(a, b) = maximum(abs, a .- b) / maximum(abs, b)

# Composite Simpson over `[a, b]`.
function _cgef_simpson(f, a, b; n = 2000)
    h = (b - a) / n
    return h / 3 * (f(a) + f(b) + 4 * sum(f(a + (2i - 1) * h) for i in 1:(n ÷ 2)) +
                    2 * sum(f(a + 2i * h) for i in 1:(n ÷ 2 - 1)))
end

function _cgef_legendre(l, x)
    p0, p1 = 1.0, x
    l == 0 && return p0
    for m in 1:(l - 1)
        p0, p1 = p1, ((2m + 1) * x * p1 - m * p0) / (m + 1)
    end
    return p1
end

# The factor a radial kernel scales a degree-`l` harmonic by on a sphere of radius `R`: its Legendre
# transform over its support, normalized to 1 at `l = 0` (Aluie 2019, eq. 34), by Simpson's rule in the
# angle `γ`. The top-hat is the indicator of the cap of angular radius `ℓ/(2R)`.
function _cgef_legendre_transfer(kernel, l, ℓ, R; n = 20_000)
    γt = min(CGEF.Kernels.kernel_radius(kernel, ℓ, Val(2)) / R, π)
    G(γ) = kernel isa CGEF.TopHatKernel ? sin(γ) : CGEF.Kernels.kernel_weight(kernel, R * γ, ℓ, Val(2)) * sin(γ)
    return _cgef_simpson(γ -> G(γ) * _cgef_legendre(l, cos(γ)), 0.0, γt; n) / _cgef_simpson(G, 0.0, γt; n)
end

# The derivative of `f` along direction `d` of a structured grid, per unit distance, with the weights of `plan`, by
# default those the diagnostics take; zero where it does not exist.
_cgef_dd(f, grid, d, plan = CGEF.Derivatives.derivative_plan(grid)) =
    FG.Operators.derivative!(zeros(size(f)), f, grid, plan, d; masked = 0.0)

# Inactive cells of `grid` farther than `r` from every active cell.
function _cgef_beyond_reach(grid, r)
    geo = FG.Grids.grid_geometry(grid)
    m = BitArray(FG.Grids.mask(grid))
    xs = [Tuple(FG.Grids.coords(grid, Tuple(I)...)) for I in CartesianIndices(m)]
    act = xs[m]
    return [!m[I] && all(x -> FG.Geometry.distance(geo, xs[I], x) > r, act) for I in CartesianIndices(m)]
end

# One grid, kernel, scale and plan keywords per real-space engine a process backend divides, masked where
# the engine takes a mask.
function _cgef_share_cases()
    cart = FG.Geometry.CartesianGeometry()
    sph = FG.Geometry.SphericalGeometry(6.371e6)
    P = FG.Grids.Periodic()
    hole(sz) = (m = trues(sz); m[3:5, 2:3] .= false; m)
    xs = cumsum([0.0, 1.1, 0.8, 1.3, 0.9, 1.2, 0.7, 1.0, 1.1, 0.95, 1.05, 1.15, 0.85])
    th, ga, sh = CGEF.TopHatKernel(), CGEF.GaussianKernel(), CGEF.Kernels.SmoothHatKernel()
    n = 120
    px = [mod(0.618 * k * 11, 11.0) for k in 1:n]
    py = [mod(0.7548 * k * 9, 9.0) for k in 1:n]
    return (
        (FG.Grids.StructuredGrid(cart, 0.0:1.0:15.0, 0.0:0.7:7.7; topology = (P, P)), sh, 1.5, (;)),
        (FG.Grids.StructuredGrid(cart, 0.0:1.0:15.0, 0.0:0.7:4.9, hole((16, 8))), sh, 1.5, (;)),
        (FG.Grids.StructuredGrid(cart, xs, collect(0.0:0.7:4.9), hole((13, 8))), sh, 2.9,
         (; cache_strategy = CGEF.Filtering.AlwaysCache())),
        (FG.Grids.StructuredGrid(cart, xs, collect(0.0:0.7:4.9), hole((13, 8))), sh, 2.9,
         (; cache_strategy = CGEF.Filtering.NeverCache())),
        (FG.Grids.StructuredGrid(cart, 0.0:1.0:15.0, 0.0:0.7:4.9, hole((16, 8))), ga, 1.5, (;)),
        (FG.Grids.StructuredGrid(cart, xs, collect(0.0:0.7:4.9), hole((13, 8))), th, 2.3, (;)),
        (FG.Grids.StructuredGrid(sph, FG.Axes.PeriodicUniformAxis(0.0, 2π, 36), collect(range(-1.3, 1.3; length = 17))),
         th, 2.0e6, (;)),
        (FG.Grids.StructuredGrid(cart, 0.0:1.0:23.0), th, 4.8, (;)),
        (FG.Grids.StructuredGrid(cart, xs), sh, 2.9, (; cache_strategy = CGEF.Filtering.AlwaysCache())),
        (FG.Grids.StructuredGrid(cart, 0.0:1.0:5.0, 0.0:0.7:2.8, 0.0:0.9:2.7; topology = (P, P, P)), ga, 1.5, (;)),
        (FG.Grids.StructuredGrid(cart, 0.0:1.0:5.0, 0.0:0.7:2.8, 0.0:0.9:2.7, trues(6, 5, 4)), th, 2.9, (;)),
        (FG.Grids.UnstructuredGrid(cart, px, py, fill(0.9, n), [mod(k, 7) != 0 for k in 1:n]), ga, 2.0,
         (; method = CGEF.Filtering.RealSpace())),
    )
end

_cgef_share_field(sz) =
    [sin(1.3 * I[1] + 0.7 * get(Tuple(I), 2, 0) + 0.4 * get(Tuple(I), 3, 0)) + 0.2 for I in CartesianIndices(sz)]
