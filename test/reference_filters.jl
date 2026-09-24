# Independent references for the filter, shared by the topic files. Each is written from the definition
# with FlowGeometries' distances and measures, and none calls an engine of the package.

# The lattice continued `K` cells past each bounded end at the edge gap — latitude to the poles, a
# regional longitude over the rest of the ring — built as a grid of its own, whose measures and
# distances FlowGeometries supplies. Returns it and the offset of the original cells in it.
function _cgef_extended_lattice(grid, K)
    geo = FG.Grids.grid_geometry(grid)
    sph = geo isa FG.Geometry.SphericalGeometry
    N = length(FG.Grids.size_tuple(grid))
    lo = zeros(Int, N)
    axes = map(1:N) do d
        x = collect(FG.Grids.coordinates(grid, d))
        n = length(x)
        (FG.Grids.isperiodic(grid, d) || n < 2) && return x
        sl, sh = x[2] - x[1], x[n] - x[n-1]
        kl, kh = K, K
        if sph && d == 1
            m = floor(Int, (2π - (abs(x[n] - x[1]) + abs(sl))) / abs(sl) + 1e-9)
            kl, kh = m ÷ 2, m - m ÷ 2
        elseif sph && d == 2
            kl = min(K, floor(Int, abs((sl > 0 ? -π / 2 : π / 2) - x[1]) / abs(sl) + 1e-9))
            kh = min(K, floor(Int, abs((sh > 0 ? π / 2 : -π / 2) - x[n]) / abs(sh) + 1e-9))
        end
        lo[d] = kl
        vcat([x[1] - k * sl for k in kl:-1:1], x, [x[n] + k * sh for k in 1:kh])
    end
    per = ntuple(d -> FG.Grids.isperiodic(grid, d) ? FG.Grids.period(grid, d) : nothing, N)
    ext = FG.Grids.StructuredGrid(geo, axes...; topology = FG.Grids.topology(grid), period = per)
    return ext, Tuple(lo)
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
    ext, lo = _cgef_extended_lattice(grid, ceil(Int, rad / hmin) + 2)
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
            w = CGEF.Kernels.kernel_weight(kernel, dist, ℓ, Val(N)) * FG.Grids.measure(ext, Tuple(J)...)
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
