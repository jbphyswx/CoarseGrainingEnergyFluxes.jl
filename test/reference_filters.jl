# Independent references for the filter, shared by the topic files. Each is written from the definition
# with FlowGeometries' distances and measures, and none calls an engine of the package.

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
            ring[d] = true
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

# The measure of cell `J` of the continued lattice: the grid's own inside it; outside, the longitude
# width (the grid's own for its columns, the lattice's past them) times the latitude integral over the
# row, the grid's own factor for its rows and `R²(sin b − sin a)` (`R(b − a)` along a single meridian)
# for a continued row between faces `a < b`.
function _cgef_lattice_measure(grid, ext, lo, rows, J)
    sz = FG.Grids.size_tuple(grid)
    Jd = ntuple(d -> J[d] - lo[d], length(sz))
    all(d -> 1 <= Jd[d] <= sz[d], 1:length(sz)) && return FG.Grids.measure(grid, Jd...)
    rows === nothing && return FG.Grids.measure(ext, J...)
    λw = 1 <= Jd[1] <= sz[1] ? FG.Grids.measure_factors(grid)[1][Jd[1]] : FG.Grids.measure_factors(ext)[1][J[1]]
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
    rad = CGEF.Kernels.kernel_radius(kernel, ℓ, Val(N))
    hmin = minimum(d -> FG.Grids.minimum_spacing(grid, d), 1:N) * (cart ? 1.0 : FG.Geometry.radius(geo))
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
            w = CGEF.Kernels.kernel_weight(kernel, dist, ℓ, dim) * me[J]
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

# The derivative of `f` along direction `d` of a structured grid, per unit distance, with the weights the
# diagnostics take; zero where it does not exist.
_cgef_dd(f, grid, d) =
    FG.Operators.derivative!(zeros(size(f)), f, grid, CGEF.Derivatives.derivative_plan(grid), d; masked = 0.0)

# Inactive cells of `grid` farther than `r` from every active cell.
function _cgef_beyond_reach(grid, r)
    geo = FG.Grids.grid_geometry(grid)
    m = BitArray(FG.Grids.mask(grid))
    xs = [Tuple(FG.Grids.coords(grid, Tuple(I)...)) for I in CartesianIndices(m)]
    act = xs[m]
    return [!m[I] && all(x -> FG.Geometry.distance(geo, xs[I], x) > r, act) for I in CartesianIndices(m)]
end
