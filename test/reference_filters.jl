# Independent references for the filter, shared by the topic files. Each is written from the definition
# with FlowGeometries' distances and measures, and none calls an engine of the package.

# The exterior of the domain: the lattice continued past each bounded end at the edge gap, `K` cells
# along a Cartesian direction. A regional longitude continues around the rest of the ring, half past
# each end, until the seam gap lies in `[s/2, 3s/2)` of the mean edge gap `s`, and closes with period
# 2π. A latitude continues toward the pole it heads for with one row per lattice centre strictly short
# of the pole, the last row reaching the pole and centred on its span, or with one cap row from the end
# cell's face to the pole when no centre is short of it; a spectral-quadrature sampling spans the sphere
# and continues in neither. Built as a grid of its own for the distances.
# Returns it, the offset of the original cells in it, and the faces `(a, b)` of each continued latitude
# row (`nothing` for the grid's own rows, and for every row when latitude is not continued).
function _cgef_extended_lattice(grid, K)
    geo = FG.Grids.grid_geometry(grid)
    sph = geo isa FG.Geometry.SphericalGeometry
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
            ring[d] = m > 0
            lo[d] = m ÷ 2
            return vcat([x[1] - k * sl for k in (m ÷ 2):-1:1], x, [x[n] + k * sh for k in 1:(m - m ÷ 2)])
        elseif sph && d == 2
            FG.Grids.sampling(grid) isa FG.SphericalSampling.AbstractSpectralQuadratureSampling && return x
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

# The measure of cell `J` of the continued lattice: the grid's own inside it; outside, the lattice's
# longitude width times the latitude integral over the row, the grid's own factor for its rows and
# `R²(sin b − sin a)` (`R(b − a)` along a single meridian) for a continued row between faces `a < b`.
function _cgef_lattice_measure(grid, ext, lo, rows, J)
    sz = FG.Grids.size_tuple(grid)
    Jd = ntuple(d -> J[d] - lo[d], length(sz))
    all(d -> 1 <= Jd[d] <= sz[d], 1:length(sz)) && return FG.Grids.measure(grid, Jd...)
    rows === nothing && return FG.Grids.measure(ext, J...)
    λw = FG.Grids.measure_factors(ext)[1][J[1]]
    rows[J[2]] === nothing && return λw * FG.Grids.measure_factors(grid)[2][Jd[2]]
    a, b = rows[J[2]]
    R = FG.Geometry.radius(FG.Grids.grid_geometry(grid))
    return length(FG.Grids.coordinates(ext, 1)) == 1 ? λw * R * (b - a) : λw * R^2 * (sin(b) - sin(a))
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
    rad = CGEF.Kernels.kernel_radius(kernel, ℓ)
    hmin = minimum(d -> FG.Grids.minimum_spacing(grid, d), 1:N) * (cart ? 1.0 : FG.Geometry.radius(geo))
    ext, lo, rows = _cgef_extended_lattice(grid, ceil(Int, rad / hmin) + 2)
    tiles = ntuple(d -> cart && FG.Grids.isperiodic(grid, d), N)
    P = ntuple(d -> tiles[d] ? FG.Grids.period(grid, d) : 0.0, N)
    M = ntuple(d -> tiles[d] ? ceil(Int, rad / P[d]) + 1 : 0, N)
    mask = FG.Grids.mask(grid)
    zf = strategy isa CGEF.Filtering.ZeroFill
    out = zeros(sz)
    for I in CartesianIndices(sz)
        (zf || mask[I]) || continue
        xi = FG.Grids.coords(grid, Tuple(I)...)
        num = 0.0
        den = 0.0
        for J in CartesianIndices(FG.Grids.size_tuple(ext)), m in CartesianIndices(ntuple(d -> (-M[d]):M[d], N))
            Jd = ntuple(d -> J[d] - lo[d], N)
            active = all(d -> 1 <= Jd[d] <= sz[d], 1:N) && mask[Jd...]
            (zf || active) || continue
            xj = FG.Grids.coords(ext, Tuple(J)...)
            xm = ntuple(d -> xj[d] + m[d] * P[d], N)
            dist = FG.Geometry.distance(geo, xi, xm)
            dist <= rad || continue
            w = CGEF.Kernels.kernel_weight(kernel, dist, ℓ, Val(N)) * _cgef_lattice_measure(grid, ext, lo, rows, Tuple(J))
            den += w
            active && (num += w * f[Jd...])
        end
        out[I] = num / den
    end
    return out
end

_cgef_relerr(a, b) = maximum(abs, a .- b) / maximum(abs, b)

# Inactive cells of `grid` farther than `r` from every active cell.
function _cgef_beyond_reach(grid, r)
    geo = FG.Grids.grid_geometry(grid)
    m = FG.Grids.mask(grid)
    act = [FG.Grids.coords(grid, Tuple(J)...) for J in CartesianIndices(m) if m[J]]
    return [!m[I] && all(x -> FG.Geometry.distance(geo, FG.Grids.coords(grid, Tuple(I)...), x) > r, act)
            for I in CartesianIndices(m)]
end
