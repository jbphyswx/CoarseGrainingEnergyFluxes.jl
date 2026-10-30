# The engine table: each row a grid, a kernel and a scale, the footprint type its serial plan builds, and the
# reference and options it is checked with. `test_filtering.jl` checks each row against its reference, and
# `test_jet.jl` runs its plan and apply under JET.

const _CGEF_TH = CGEF.TopHatKernel()
const _CGEF_THC = CGEF.TopHatKernel(; cells = CGEF.Kernels.CellCentre())
const _CGEF_GA = CGEF.GaussianKernel()
const _CGEF_SH = CGEF.Kernels.SmoothHatKernel()
const _CGEF_SS = CGEF.SharpSpectralKernel()
const _CGEF_CACHES = (CGEF.Filtering.AlwaysCache(), CGEF.Filtering.NeverCache())
const _CGEF_FE = CGEF.Filtering.FootprintEngine()
_cgef_hole(sz) = (m = trues(sz); m[5:8, 3:5] .= false; m)

_cgef_field(sz) = [sin(1.3 * I[1] + 0.7 * get(Tuple(I), 2, 0) + 0.4 * get(Tuple(I), 3, 0)) + 0.2
                   for I in CartesianIndices(sz)]

# A box-truncated separable engine differs from the disk-truncated reference by the kernel's weight past its
# radius, below 1e-9 of its peak for the Gaussian: those rows take `rtol = 1e-8`.
_cgef_row(grid, kernel, ℓ, engine; reference = _cgef_direct_filter, rtol = 1e-11, kw...) =
    (; grid, kernel, ℓ, engine, reference, rtol, kw = (; kw...))

# The cell overlap's reference in three dimensions at 40 Gauss–Legendre nodes: its 10-node ball fractions fall short
# of the rows' `rtol`. The engines' own 10-node fractions are within 1e-10 of the ball's, those rows' `rtol`.
_cgef_ball40(grid, kernel, ℓ, f, strategy) = _cgef_direct_filter(grid, kernel, ℓ, f, strategy, 40)

# Top-hat radii here sit off every distance between cells, so no cell lies on a support boundary.
function _cgef_cartesian_rows()
    cart = FG.Geometry.CartesianGeometry()
    xs = cumsum([0.0, 1.1, 0.8, 1.3, 0.9, 1.2, 0.7, 1.0, 1.1, 0.95, 1.05, 1.15, 0.85])
    rows = Any[]
    row(args...; kw...) = push!(rows, _cgef_row(args...; kw...))

    # Periods 16 and 8.4; the supports reach one, two and five periods, and a periodic Cartesian direction
    # counts a cell once per image. Each kernel takes its own engine (prefix sums, separable passes, the
    # banded sum) while that costs fewer operations than the padded transform, and the transform past it.
    gp = FG.Grids.StructuredGrid(cart, 0.0:1.0:15.0, 0.0:0.7:7.7; topology = (FG.Grids.Periodic(), FG.Grids.Periodic()))
    row(gp, _CGEF_THC, 1.5, :PrefixSumTopHatPlan)
    row(gp, _CGEF_TH, 1.5, :PrefixSumTopHatPlan)
    row(gp, _CGEF_SH, 1.5, :FilterFootprint)
    row(gp, _CGEF_GA, 1.5, :SeparableFootprint; rtol = 1e-8)
    for ℓ in (4.8, 19.2, 40.8)
        row(gp, _CGEF_TH, ℓ, :PaddedFFTFootprint)
        row(gp, _CGEF_SH, ℓ, :PaddedFFTFootprint)
    end
    row(gp, _CGEF_GA, 4.8, :PaddedFFTFootprint; rtol = 1e-8)

    # Three rows along a period of 64, the supports within one period and past one, two and five: the centre
    # test's prefix sums cost less than the transform at any support, the cell overlap's edge cells grow with
    # it.
    gt = FG.Grids.StructuredGrid(cart, 0.0:1.0:63.0, 0.0:0.7:1.4; topology = (FG.Grids.Periodic(), FG.Grids.Bounded()))
    for ℓ in (20.3, 70.3, 140.7, 330.1)
        row(gt, _CGEF_THC, ℓ, :PrefixSumTopHatPlan)
        row(gt, _CGEF_TH, ℓ, :PaddedFFTFootprint)
        row(gt, _CGEF_TH, ℓ, :PrefixSumTopHatPlan; engine = _CGEF_FE)
    end

    # Bounded, with the support past more than half the rows, unmasked and masked.
    gb = FG.Grids.StructuredGrid(cart, 0.0:1.0:15.0, 0.0:0.7:2.8)
    gbm = FG.Grids.StructuredGrid(cart, 0.0:1.0:15.0, 0.0:0.7:4.9, _cgef_hole((16, 8)))
    for g in (gb, gbm)
        for ℓ in (4.8, 9.6)
            row(g, _CGEF_TH, ℓ, :PrefixSumTopHatPlan)
            row(g, _CGEF_GA, ℓ, :SeparableFootprint; rtol = 1e-8)
        end
        row(g, _CGEF_SH, 1.5, :FilterFootprint)
        row(g, _CGEF_SH, 19.2, :PaddedFFTFootprint)
    end

    # Stretched axes, bounded and periodic, and the same cells stored in descending order. The centre test
    # takes the prefix sums at any spacing, the cell overlap the cell-by-cell plan on a stretched axis 1.
    gs = FG.Grids.StructuredGrid(cart, xs, collect(0.0:0.7:4.9), _cgef_hole((13, 8)))
    gd = FG.Grids.StructuredGrid(cart, reverse(xs), collect(4.9:-0.7:0.0), reverse(_cgef_hole((13, 8))))
    for g in (gs, gd), ℓ in (2.9, 6.13)
        row(g, _CGEF_THC, ℓ, :PrefixSumTopHatPlan)
        row(g, _CGEF_GA, ℓ, :SeparableFootprint; rtol = 1e-8)
        for cs in _CGEF_CACHES
            row(g, _CGEF_SH, ℓ, :ScatteredFilterPlan; cache_strategy = cs)
            row(g, _CGEF_TH, ℓ, :ScatteredFilterPlan; cache_strategy = cs)
        end
    end
    gsp = FG.Grids.StructuredGrid(cart, xs, 0.0:0.7:7.7; topology = (FG.Grids.Periodic(), FG.Grids.Periodic()), period = (13.0, 8.4))
    for ℓ in (5.1462, 29.9226)
        row(gsp, _CGEF_THC, ℓ, :PrefixSumTopHatPlan)
        row(gsp, _CGEF_TH, ℓ, :ScatteredFilterPlan)
    end

    # A uniform axis 1 with a stretched axis 2: one band of offsets per row.
    ys = cumsum([0.6 + 0.8 * abs(sin(0.3j)) for j in 1:12])
    for g in (FG.Grids.StructuredGrid(cart, 0.0:1.0:15.0, ys),
              FG.Grids.StructuredGrid(cart, 0.0:1.0:15.0, ys; topology = (FG.Grids.Periodic(), FG.Grids.Periodic()), period = (16.0, ys[end] - ys[1] + 1.0)))
        row(g, _CGEF_SS, 2.0, :FilterFootprint)
        row(g, _CGEF_SH, 5.0, :FilterFootprint)
    end

    # One dimension.
    g1 = FG.Grids.StructuredGrid(cart, 0.0:1.0:23.0)
    row(g1, _CGEF_TH, 4.8, :FilterFootprintND)
    row(FG.Grids.StructuredGrid(cart, 0.0:1.0:7.0; topology = FG.Grids.Periodic()), _CGEF_TH, 13.0, :FilterFootprintND)
    row(FG.Grids.StructuredGrid(cart, 0.0:1.0:23.0; topology = FG.Grids.Periodic()), _CGEF_TH, 30.5, :PaddedFFTFootprint)
    row(g1, _CGEF_GA, 4.8, :SeparableFootprintND; rtol = 1e-8)
    for cs in _CGEF_CACHES
        row(FG.Grids.StructuredGrid(cart, xs), _CGEF_SH, 2.9, :NDScatteredFilterPlan; cache_strategy = cs)
    end

    # Three dimensions: whole turns of axis 1 and images along axes 2 and 3; stretched; bounded and masked.
    g3 = FG.Grids.StructuredGrid(cart, 0.0:1.0:5.0, 0.0:0.7:2.8, 0.0:0.9:2.7; topology = (FG.Grids.Periodic(), FG.Grids.Periodic(), FG.Grids.Periodic()))
    row(g3, _CGEF_TH, 1.5, :PrefixSumTopHat3DPlan; reference = _cgef_ball40, rtol = 1e-10)
    row(g3, _CGEF_GA, 1.5, :SeparableFootprintND; rtol = 1e-8)
    for ℓ in (4.8, 13.3)
        row(g3, _CGEF_TH, ℓ, :PaddedFFTFootprint; reference = _cgef_ball40, rtol = 1e-10)
        row(g3, _CGEF_SH, ℓ, :PaddedFFTFootprint)
    end
    row(g3, _CGEF_GA, 2.9, :PaddedFFTFootprint; rtol = 1e-8)
    g3s = FG.Grids.StructuredGrid(cart, cumsum([0.0, 1.1, 0.8, 1.3, 0.9, 1.2]), 0.0:0.7:2.8, 0.0:0.9:2.7;
                                  topology = (FG.Grids.Periodic(), FG.Grids.Periodic(), FG.Grids.Periodic()), period = (6.2, 3.5, 3.6))
    for cs in _CGEF_CACHES
        row(g3s, _CGEF_SH, 4.8, :NDScatteredFilterPlan; cache_strategy = cs)
    end
    m3 = trues(6, 5, 4)
    m3[2:3, 2:3, 2] .= false
    g3b = FG.Grids.StructuredGrid(cart, 0.0:1.0:5.0, 0.0:0.7:2.8, 0.0:0.9:2.7, m3)
    row(g3b, _CGEF_TH, 2.9, :PrefixSumTopHat3DPlan; reference = _cgef_ball40, rtol = 1e-10)
    row(g3b, _CGEF_TH, 8.3, :PaddedFFTFootprint; reference = _cgef_ball40, rtol = 1e-10)
    row(g3b, _CGEF_TH, 8.3, :PrefixSumTopHat3DPlan; engine = _CGEF_FE, reference = _cgef_ball40, rtol = 1e-10)
    row(g3b, _CGEF_SH, 2.9, :FilterFootprintND)
    row(g3b, _CGEF_SH, 5.3, :PaddedFFTFootprint)
    return rows
end

# A spherical longitude identifies, so each cell counts once however far the support reaches. A uniform
# longitude takes the kernel's own engine or, where it costs fewer operations, the transform along
# longitude; the top-hat takes the prefix sums on every one of these grids, the box spanning its gap included.
function _cgef_spherical_rows()
    sph = FG.Geometry.SphericalGeometry(6.371e6)
    rows = Any[]
    row(args...; kw...) = push!(rows, _cgef_row(args...; kw...))
    tophat(g, ℓ) = row(g, _CGEF_TH, ℓ, :PrefixSumTopHatPlan)

    # Global, the support reaching over a pole and round the ring.
    ring = range(0.0; step = deg2rad(10.0), length = 36)
    gw = FG.Grids.StructuredGrid(sph, ring, range(deg2rad(-80.0); step = deg2rad(10.0), length = 17); topology = (FG.Grids.Periodic(), FG.Grids.Bounded()))
    row(gw, _CGEF_THC, 2.0e6, :PrefixSumTopHatPlan)
    tophat(gw, 2.0e6)
    tophat(gw, 2 * 12_000e3)
    row(gw, _CGEF_GA, 0.5e6, :FilterFootprint)
    row(gw, _CGEF_GA, 3.0e6, :ZonalFFTFootprint)

    # Stretched latitude, with a hole.
    φs = [deg2rad(-75 + 150 * (s + 0.12 * sin(π * s))) for s in range(0, 1; length = 14)]
    gls = FG.Grids.StructuredGrid(sph, ring, φs, (m = trues(36, 14); m[10:14, 5:8] .= false; m))
    row(gls, _CGEF_GA, 1.5e6, :FilterFootprint)
    row(gls, _CGEF_GA, 4.0e6, :ZonalFFTFootprint)

    # A regional box whose longitude support is wider than half its columns, with and without a hole; vector
    # axes take the per-point engine.
    λr = range(0.0, deg2rad(38.0); length = 20)
    φr = range(deg2rad(60.0), deg2rad(80.0); length = 11)
    gr = FG.Grids.StructuredGrid(sph, λr, φr)
    row(gr, _CGEF_SH, 1000e3, :ZonalFFTFootprint)
    tophat(gr, 1000e3)
    for cs in _CGEF_CACHES
        row(FG.Grids.StructuredGrid(sph, collect(λr), collect(φr)), _CGEF_SH, 1000e3, :ScatteredFilterPlan;
            cache_strategy = cs)
    end
    grm = FG.Grids.StructuredGrid(sph, λr, φr, (m = trues(20, 11); m[4:6, 3:4] .= false; m))
    row(grm, _CGEF_SH, 1500e3, :ZonalFFTFootprint)
    tophat(grm, 600e3)

    # A box over 320° of longitude, whose 40° gap the support spans: the columns at its two ends are
    # neighbours across the gap.
    λw = range(0.0; step = deg2rad(10.0), length = 33)
    φw = range(deg2rad(50.0), deg2rad(70.0); length = 5)
    tophat(FG.Grids.StructuredGrid(sph, λw, φw), 3.1e6)
    tophat(FG.Grids.StructuredGrid(sph, λw, φw), 5.0e6)
    row(FG.Grids.StructuredGrid(sph, λw, φw), _CGEF_SH, 3.5e6, :ZonalFFTFootprint)
    for cs in _CGEF_CACHES
        row(FG.Grids.StructuredGrid(sph, collect(λw), collect(φw)), _CGEF_SH, 2.5e6, :ScatteredFilterPlan;
            cache_strategy = cs)
    end

    # A meridian and a parallel: the measure is an arc length. Their longitudes are vectors, which the centre test
    # takes to the prefix sums and the cell overlap cell by cell.
    lat = deg2rad.(collect(-60.0:5.0:60.0))
    for g in (FG.Grids.StructuredGrid(sph, [0.0], lat), FG.Grids.StructuredGrid(sph, deg2rad.(collect(0.0:5.0:355.0)), [0.3]))
        row(g, _CGEF_THC, 2.0e6, :PrefixSumTopHatPlan)
        row(g, _CGEF_TH, 2.0e6, :ScatteredFilterPlan)
    end

    # Latitude stored north to south, as `π/2 − θ` gives it.
    gn = FG.Grids.StructuredGrid(sph, range(0.0; step = deg2rad(15.0), length = 24), π / 2 .- range(0.3, π - 0.3; length = 12))
    tophat(gn, 1.7e6)

    # A shell, whose distance is the chord: the top-hat's ball over sectors of latitude, longitude and radius, round
    # the ring and past the radial faces; by one list per target on a longitude vector, and by one list per (row,
    # level) shifted round a uniform ring.
    R = FG.Geometry.radius(sph)
    for λs in (deg2rad.(collect(0.0:30.0:330.0)), range(0.0; step = 2π / 12, length = 12))
        gsh = FG.Grids.StructuredGrid(sph, λs, deg2rad.(collect(-60.0:20.0:60.0)), collect(R:100e3:(R + 300e3));
                                      topology = (FG.Grids.Periodic(), FG.Grids.Bounded(), FG.Grids.Bounded()))
        row(gsh, _CGEF_TH, 8.0e5, :NDScatteredFilterPlan)
    end
    row(gn, _CGEF_GA, 1.2e6, :FilterFootprint)
    row(gn, _CGEF_GA, 4.0e6, :ZonalFFTFootprint)
    return rows
end

# WGS 84, at the geodesic distance. A uniform longitude takes the banded sum for smooth kernels and the prefix sums
# for the cell overlap; the centre test, and every kernel on a longitude vector, the per-point plan. The cell
# overlap's 10-point quadrature leaves its weights within 1e-8 on 10° cells, which a `ZeroFill` window reaching
# both poles divides by the disk's area in the reference and by the summed weights in the engines.
function _cgef_spheroid_rows()
    E = FG.Geometry.SpheroidGeometry(6.378137e6, 1 / 298.257223563)
    rows = Any[]
    row(args...; kw...) = push!(rows, _cgef_row(args...; kw...))
    lat = deg2rad.(collect(-85.0:10.0:85.0))
    gw = FG.Grids.StructuredGrid(E, FG.Axes.PeriodicUniformAxis(0.0, 2π, 36), lat)
    gm = FG.Grids.StructuredGrid(E, FG.Axes.PeriodicUniformAxis(0.0, 2π, 36), lat, (m = trues(36, 18); m[5:9, 4:7] .= false; m))
    gr = FG.Grids.StructuredGrid(E, range(0.0; step = deg2rad(5.0), length = 19), deg2rad.(collect(-40.0:5.0:40.0)))
    gs = FG.Grids.StructuredGrid(E, sort(vcat(deg2rad.(0.0:12.0:348.0), deg2rad.(3.0:12.0:351.0))), lat;
                                 topology = (FG.Grids.Periodic(), FG.Grids.Bounded()), period = (2π, nothing))
    for g in (gw, gm, gr), ℓ in (1.5e6, 4.0e6)
        row(g, _CGEF_GA, ℓ, :FilterFootprint)
        row(g, _CGEF_SH, ℓ, :FilterFootprint)
        row(g, _CGEF_THC, ℓ, :ScatteredFilterPlan)
        row(g, _CGEF_TH, ℓ, :PrefixSumTopHatPlan; rtol = g === gr ? 1e-11 : 1e-8)
    end
    for k in (_CGEF_GA, _CGEF_THC, _CGEF_TH)
        row(gs, k, 4.0e6, :ScatteredFilterPlan; rtol = k === _CGEF_TH ? 1e-8 : 1e-11)
    end
    # The equator, a circle of radius `a`.
    g1 = FG.Grids.StructuredGrid(E, range(0.0; step = 2π / 24, length = 24); topology = FG.Grids.Periodic())
    row(g1, _CGEF_TH, 4.0e6, :NDScatteredFilterPlan)
    return rows
end

# Meshes, node sets and sphere pixelizations, checked against the definition over the grid's own cells.
function _cgef_mesh_rows()
    cart = FG.Geometry.CartesianGeometry()
    rows = Any[]
    row(args...; kw...) = push!(rows, _cgef_row(args...; reference = _cgef_cellset_filter, kw...))

    θ = deg2rad(15.0)
    cx = [i * cos(θ) - 0.3j * sin(θ) for i in 0:11, j in 0:9]
    cy = [i * sin(θ) + j * (1 + 0.3cos(θ)) for i in 0:11, j in 0:9]
    gc = FG.Grids.CurvilinearGrid(cart, cx, cy, _cgef_hole((12, 10)))
    for (k, ℓ) in ((_CGEF_GA, 2.0), (_CGEF_SS, 1.0)), cs in _CGEF_CACHES
        row(gc, k, ℓ, :ScatteredFilterPlan; cache_strategy = cs)
    end
    # A top-hat over the mesh takes bin runs, the support from a few cells to past the whole mesh: the centre
    # test, and the cell overlap with the cells near the disk's edge held per target or found per apply.
    for ℓ in (3.1, 8.3, 29.7)
        row(gc, _CGEF_THC, ℓ, :BinnedTopHatPlan)
        for cs in _CGEF_CACHES
            row(gc, _CGEF_TH, ℓ, :BinnedTopHatPlan; cache_strategy = cs)
        end
    end

    mn = vec(_cgef_hole((12, 10)))
    gu = FG.Grids.UnstructuredGrid(cart, vec(cx) .+ 0.1 .* sin.(1:120), vec(cy), fill(1.1, 120), mn)
    row(gu, _CGEF_GA, 2.0, :NodeFilterPlan; method = CGEF.Filtering.RealSpace())
    for ℓ in (3.1, 8.3, 29.7)
        row(gu, _CGEF_TH, ℓ, :BinnedTopHatPlan)
    end
    # A periodic node set, whose support reaches across both faces, and a node set in three dimensions.
    xp, yp = mod.(vec(cx) .+ 0.1 .* sin.(1:120), 12.5), mod.(vec(cy), 13.0)
    gup = FG.Grids.UnstructuredGrid(cart, (xp, yp), fill(1.35, 120), mn; topology = (FG.Grids.Periodic(), FG.Grids.Periodic()), period = (12.5, 13.0))
    for ℓ in (3.1, 8.3, 12.1)
        row(gup, _CGEF_TH, ℓ, :BinnedTopHatPlan)
    end
    # Other kernels take the neighbour list; at ℓ = 4 the Gaussian's support passes half of each period.
    for ℓ in (2.0, 4.0)
        row(gup, _CGEF_GA, ℓ, :NodeFilterPlan)
    end
    zs = [mod(7.3k, 5.0) for k in 1:120]
    gu3 = FG.Grids.UnstructuredGrid(cart, (vec(cx), vec(cy), zs), fill(6.0, 120), mn)
    for ℓ in (4.1, 11.3)
        row(gu3, _CGEF_TH, ℓ, :BinnedTopHatPlan)
    end
    row(gu3, _CGEF_GA, 2.0, :NodeFilterPlan)

    # Each pixelization with a hole of half the kernel radius about cell 1, at a width of about 1.3 cells.
    sgeo = FG.Geometry.SphericalGeometry()
    builders = (m -> FG.Grids.RingGrid(sgeo, FG.SphericalSampling.ReducedGaussianSampling([16, 24, 32, 32, 24, 16]); mask = m),
                m -> FG.Grids.CubedSphereGrid(sgeo, 6; mask = m), m -> FG.Grids.HEALPixGrid(sgeo, 4; mask = m),
                m -> FG.Grids.IcosahedralGrid(sgeo, 4; mask = m), m -> FG.Grids.YinYangGrid(sgeo, 20, 10; mask = m))
    for build in builders
        g0 = build(nothing)
        n = length(FG.Grids.mask(g0))
        ℓ = 1.3 * FG.Geometry.radius(sgeo) * sqrt(4π / n)
        p1 = Tuple(FG.Grids.coords(g0, 1))
        cut = CGEF.Kernels.kernel_radius(_CGEF_GA, ℓ) / 2
        hole = [FG.Geometry.distance(sgeo, p1, Tuple(FG.Grids.coords(g0, i))) > cut for i in 1:n]
        row(build(hole), _CGEF_GA, ℓ, :NodeFilterPlan)
    end
    return rows
end
