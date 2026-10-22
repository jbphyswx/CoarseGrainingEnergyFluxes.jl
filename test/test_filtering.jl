
# The real-space filter against its definition. Each engine, on grids that reach it, under both mask
# strategies and on every execution backend, equals the direct sum over the lattice continued past the
# grid's bounded edges (`_cgef_direct_filter`) or over the grid's own cells (`_cgef_cellset_filter`).

const _CGEF_SERIAL = CGEF.ComputationalBackends.SerialBackend()
const _CGEF_PARALLEL = (CGEF.ComputationalBackends.ThreadedBackend(), CGEF.ComputationalBackends.GPUBackend(KA.CPU()),
                        CGEF.ComputationalBackends.DistributedBackend(), CGEF.ComputationalBackends.MPIBackend())

# What `check_setup` reports for each engine.
const _CGEF_ENGINE_REPORT = Dict(
    :PrefixSumTopHatPlan => "prefix-sum top-hat, exact, O(N·w_y)",
    :PrefixSumTopHat3DPlan => "prefix-sum top-hat, exact, O(N·w_y·w_z)",
    :SeparableFootprint => "separable two-pass",
    :SeparableFootprintND => "separable N-pass",
    :FilterFootprint => "banded footprint",
    :ScatteredFilterPlan => "scattered per-point",
    :FilterFootprintND => "N-dimensional ball",
    :NDScatteredFilterPlan => "scattered N-dimensional",
    :NodeFilterPlan => "node CSR",
    :PaddedFFTFootprint => "padded-FFT",
    :ZonalFFTFootprint => "zonal-FFT",
)

_cgef_alloc_apply(out, f, plan) = @allocated CGEF.Filtering.filter_apply!(out, f, plan)
_cgef_alloc_batch(outs, fs, plan) = @allocated CGEF.Filtering.filter_apply_batch!(outs, fs, plan)

_cgef_field(sz) = [sin(1.3 * I[1] + 0.7 * get(Tuple(I), 2, 0) + 0.4 * get(Tuple(I), 3, 0)) + 0.2
                   for I in CartesianIndices(sz)]

# `engine` is the footprint type the serial plan must build. A box-truncated separable engine differs
# from the disk-truncated reference by the kernel's weight past its radius, below 1e-9 of its peak for
# the Gaussian.
function _cgef_engine_matches(grid, kernel, ℓ, engine; reference = _cgef_direct_filter, rtol = 1e-11,
                              method = CGEF.Filtering.RealSpace(), kw...)
    f = _cgef_field(size(FG.Grids.mask(grid)))
    fs = (f, 2 .* f)
    for strategy in (CGEF.Filtering.ZeroFill(), CGEF.Filtering.Deformable())
        ref = reference(grid, kernel, ℓ, f, strategy)
        plan(b) = CGEF.Filtering.plan_filter(grid, kernel, ℓ; method, execution = b, mask_strategy = strategy, kw...)
        p = plan(_CGEF_SERIAL)
        built = nameof(typeof(p.footprint))
        built === engine || error("$(nameof(typeof(grid))), $(nameof(typeof(kernel))), ℓ = $ℓ builds $built, not $engine")
        Test.@test occursin(_CGEF_ENGINE_REPORT[engine],
                            CGEF.check_setup(grid, kernel, ℓ; method, mask_strategy = strategy, execution = _CGEF_SERIAL).engine)
        out = CGEF.Filtering.filter_apply!(zero(f), f, p)
        Test.@test _cgef_relerr(out, ref) < rtol
        Test.@test minimum(_ -> _cgef_alloc_apply(out, f, p), 1:2) == 0
        outs = (zero(f), zero(f))
        CGEF.Filtering.filter_apply_batch!(outs, fs, p)
        Test.@test outs[1] == out && outs[2] == CGEF.Filtering.filter_apply!(zero(f), fs[2], p)
        Test.@test minimum(_ -> _cgef_alloc_batch(outs, fs, p), 1:2) == 0
        for b in _CGEF_PARALLEL
            Test.@test _cgef_relerr(CGEF.Filtering.filter_apply!(zero(f), f, plan(b)), ref) < rtol
        end
    end
end

const _CGEF_TH = CGEF.TopHatKernel()
const _CGEF_GA = CGEF.GaussianKernel()
const _CGEF_SH = CGEF.Kernels.SmoothHatKernel()
const _CGEF_SS = CGEF.SharpSpectralKernel()
const _CGEF_AUTO = CGEF.Filtering.AutoMethod()
const _CGEF_CACHES = (CGEF.Filtering.AlwaysCache(), CGEF.Filtering.NeverCache())
_cgef_hole(sz) = (m = trues(sz); m[5:8, 3:5] .= false; m)

# Top-hat radii here sit off every distance between cells, so no cell lies on a support boundary.
Test.@testset "Real-space engines on Cartesian grids" begin
    cart = FG.Geometry.CartesianGeometry()
    xs = cumsum([0.0, 1.1, 0.8, 1.3, 0.9, 1.2, 0.7, 1.0, 1.1, 0.95, 1.05, 1.15, 0.85])

    # Periods 16 and 8.4; the supports reach one, two and five periods, and a periodic Cartesian direction
    # counts a cell once per image.
    gp = FG.Grids.StructuredGrid(cart, 0.0:1.0:15.0, 0.0:0.7:7.7; periodic = (true, true))
    for ℓ in (4.8, 19.2, 40.8)
        _cgef_engine_matches(gp, _CGEF_TH, ℓ, :PrefixSumTopHatPlan)
        _cgef_engine_matches(gp, _CGEF_SH, ℓ, :FilterFootprint)
        _cgef_engine_matches(gp, _CGEF_SH, ℓ, :PaddedFFTFootprint; method = _CGEF_AUTO)
    end
    _cgef_engine_matches(gp, _CGEF_GA, 4.8, :SeparableFootprint; rtol = 1e-8)

    # Bounded, with the support past more than half the rows, unmasked and masked.
    gb = FG.Grids.StructuredGrid(cart, 0.0:1.0:15.0, 0.0:0.7:2.8)
    gbm = FG.Grids.StructuredGrid(cart, 0.0:1.0:15.0, 0.0:0.7:4.9, _cgef_hole((16, 8)))
    for g in (gb, gbm), ℓ in (4.8, 9.6)
        _cgef_engine_matches(g, _CGEF_TH, ℓ, :PrefixSumTopHatPlan)
        _cgef_engine_matches(g, _CGEF_SH, ℓ, :FilterFootprint)
        _cgef_engine_matches(g, _CGEF_SH, ℓ, :PaddedFFTFootprint; method = _CGEF_AUTO)
        _cgef_engine_matches(g, _CGEF_GA, ℓ, :SeparableFootprint; rtol = 1e-8)
    end

    # Stretched axes, bounded and periodic, and the same cells stored in descending order.
    gs = FG.Grids.StructuredGrid(cart, xs, collect(0.0:0.7:4.9), _cgef_hole((13, 8)))
    gd = FG.Grids.StructuredGrid(cart, reverse(xs), collect(4.9:-0.7:0.0), reverse(_cgef_hole((13, 8))))
    for g in (gs, gd), ℓ in (2.9, 6.13)
        _cgef_engine_matches(g, _CGEF_TH, ℓ, :PrefixSumTopHatPlan)
        _cgef_engine_matches(g, _CGEF_GA, ℓ, :SeparableFootprint; rtol = 1e-8)
        for cs in _CGEF_CACHES
            _cgef_engine_matches(g, _CGEF_SH, ℓ, :ScatteredFilterPlan; cache_strategy = cs)
        end
    end
    gsp = FG.Grids.StructuredGrid(cart, xs, 0.0:0.7:7.7; periodic = (true, true), period = (13.0, 8.4))
    for ℓ in (5.1462, 29.9226)
        _cgef_engine_matches(gsp, _CGEF_TH, ℓ, :PrefixSumTopHatPlan)
    end

    # A uniform axis 1 with a stretched axis 2: one band of offsets per row.
    ys = cumsum([0.6 + 0.8 * abs(sin(0.3j)) for j in 1:12])
    for g in (FG.Grids.StructuredGrid(cart, 0.0:1.0:15.0, ys),
              FG.Grids.StructuredGrid(cart, 0.0:1.0:15.0, ys; periodic = (true, true), period = (16.0, ys[end] - ys[1] + 1.0)))
        _cgef_engine_matches(g, _CGEF_SS, 2.0, :FilterFootprint)
        _cgef_engine_matches(g, _CGEF_SH, 5.0, :FilterFootprint)
    end

    # One dimension.
    g1 = FG.Grids.StructuredGrid(cart, 0.0:1.0:23.0)
    _cgef_engine_matches(g1, _CGEF_TH, 4.8, :FilterFootprintND)
    _cgef_engine_matches(FG.Grids.StructuredGrid(cart, 0.0:1.0:23.0; periodic = true), _CGEF_TH, 30.5, :FilterFootprintND)
    _cgef_engine_matches(g1, _CGEF_GA, 4.8, :SeparableFootprintND; rtol = 1e-8)
    for cs in _CGEF_CACHES
        _cgef_engine_matches(FG.Grids.StructuredGrid(cart, xs), _CGEF_SH, 2.9, :NDScatteredFilterPlan; cache_strategy = cs)
    end

    # Three dimensions: whole turns of axis 1 and images along axes 2 and 3; stretched; bounded and masked.
    g3 = FG.Grids.StructuredGrid(cart, 0.0:1.0:5.0, 0.0:0.7:2.8, 0.0:0.9:2.7; periodic = (true, true, true))
    for ℓ in (4.8, 13.3)
        _cgef_engine_matches(g3, _CGEF_TH, ℓ, :PrefixSumTopHat3DPlan)
        _cgef_engine_matches(g3, _CGEF_SH, ℓ, :FilterFootprintND)
    end
    _cgef_engine_matches(g3, _CGEF_GA, 2.9, :SeparableFootprintND; rtol = 1e-8)
    g3s = FG.Grids.StructuredGrid(cart, cumsum([0.0, 1.1, 0.8, 1.3, 0.9, 1.2]), 0.0:0.7:2.8, 0.0:0.9:2.7;
                                  periodic = (true, true, true), period = (6.2, 3.5, 3.6))
    for cs in _CGEF_CACHES
        _cgef_engine_matches(g3s, _CGEF_SH, 4.8, :NDScatteredFilterPlan; cache_strategy = cs)
    end
    m3 = trues(6, 5, 4)
    m3[2:3, 2:3, 2] .= false
    g3b = FG.Grids.StructuredGrid(cart, 0.0:1.0:5.0, 0.0:0.7:2.8, 0.0:0.9:2.7, m3)
    for ℓ in (2.9, 5.3)
        _cgef_engine_matches(g3b, _CGEF_TH, ℓ, :PrefixSumTopHat3DPlan)
        _cgef_engine_matches(g3b, _CGEF_SH, ℓ, :FilterFootprintND)
    end
end

# A spherical longitude identifies, so each cell counts once however far the support reaches.
Test.@testset "Real-space engines on spherical grids" begin
    sph = FG.Geometry.SphericalGeometry(6.371e6)
    # Global, the support reaching over a pole and round the ring.
    ring = range(0.0; step = deg2rad(10.0), length = 36)
    gw = FG.Grids.StructuredGrid(sph, ring, range(deg2rad(-80.0); step = deg2rad(10.0), length = 17); periodic = (true, false))
    _cgef_engine_matches(gw, _CGEF_TH, 2 * 12_000e3, :PrefixSumTopHatPlan)
    _cgef_engine_matches(gw, _CGEF_GA, 3.0e6, :FilterFootprint)
    _cgef_engine_matches(gw, _CGEF_GA, 3.0e6, :ZonalFFTFootprint; method = _CGEF_AUTO)

    # Stretched latitude, with a hole.
    φs = [deg2rad(-75 + 150 * (s + 0.12 * sin(π * s))) for s in range(0, 1; length = 14)]
    gls = FG.Grids.StructuredGrid(sph, ring, φs, (m = trues(36, 14); m[10:14, 5:8] .= false; m))
    _cgef_engine_matches(gls, _CGEF_GA, 1.5e6, :FilterFootprint)

    # A regional box whose longitude support is wider than half its columns, with and without a hole;
    # range axes take the banded engine and vector axes the per-point one.
    λr = range(0.0, deg2rad(38.0); length = 20)
    φr = range(deg2rad(60.0), deg2rad(80.0); length = 11)
    gr = FG.Grids.StructuredGrid(sph, λr, φr)
    _cgef_engine_matches(gr, _CGEF_SH, 1000e3, :FilterFootprint)
    _cgef_engine_matches(gr, _CGEF_TH, 1000e3, :PrefixSumTopHatPlan)
    for cs in _CGEF_CACHES
        _cgef_engine_matches(FG.Grids.StructuredGrid(sph, collect(λr), collect(φr)), _CGEF_SH, 1000e3,
                             :ScatteredFilterPlan; cache_strategy = cs)
    end
    grm = FG.Grids.StructuredGrid(sph, λr, φr, (m = trues(20, 11); m[4:6, 3:4] .= false; m))
    _cgef_engine_matches(grm, _CGEF_SH, 1500e3, :FilterFootprint)
    _cgef_engine_matches(grm, _CGEF_TH, 600e3, :PrefixSumTopHatPlan)

    # A box over 320° of longitude, whose 40° gap the support spans: the columns at its two ends are
    # neighbours across the gap.
    λw = range(0.0; step = deg2rad(10.0), length = 33)
    φw = range(deg2rad(50.0), deg2rad(70.0); length = 5)
    _cgef_engine_matches(FG.Grids.StructuredGrid(sph, λw, φw), _CGEF_TH, 5.0e6, :PrefixSumTopHatPlan)
    _cgef_engine_matches(FG.Grids.StructuredGrid(sph, λw, φw), _CGEF_SH, 2.5e6, :FilterFootprint)
    for cs in _CGEF_CACHES
        _cgef_engine_matches(FG.Grids.StructuredGrid(sph, collect(λw), collect(φw)), _CGEF_SH, 2.5e6,
                             :ScatteredFilterPlan; cache_strategy = cs)
    end

    # A meridian and a parallel: the measure is an arc length.
    lat = deg2rad.(collect(-60.0:5.0:60.0))
    _cgef_engine_matches(FG.Grids.StructuredGrid(sph, [0.0], lat), _CGEF_TH, 2.0e6, :PrefixSumTopHatPlan)
    _cgef_engine_matches(FG.Grids.StructuredGrid(sph, deg2rad.(collect(0.0:5.0:355.0)), [0.3]), _CGEF_TH, 2.0e6,
                         :PrefixSumTopHatPlan)

    # Latitude stored north to south, as `π/2 − θ` gives it.
    gn = FG.Grids.StructuredGrid(sph, range(0.0; step = deg2rad(15.0), length = 24), π / 2 .- range(0.3, π - 0.3; length = 12))
    _cgef_engine_matches(gn, _CGEF_TH, 1.7e6, :PrefixSumTopHatPlan)
    _cgef_engine_matches(gn, _CGEF_GA, 1.2e6, :FilterFootprint)
end

Test.@testset "Real-space engines on meshes, node sets and sphere pixelizations" begin
    cart = FG.Geometry.CartesianGeometry()
    θ = deg2rad(15.0)
    cx = [i * cos(θ) - 0.3j * sin(θ) for i in 0:11, j in 0:9]
    cy = [i * sin(θ) + j * (1 + 0.3cos(θ)) for i in 0:11, j in 0:9]
    gc = FG.Grids.CurvilinearGrid(cart, cx, cy, _cgef_hole((12, 10)))
    for (k, ℓ) in ((_CGEF_TH, 3.1), (_CGEF_GA, 2.0), (_CGEF_SS, 1.0)), cs in _CGEF_CACHES
        _cgef_engine_matches(gc, k, ℓ, :ScatteredFilterPlan; reference = _cgef_cellset_filter, cache_strategy = cs)
    end

    mn = vec(_cgef_hole((12, 10)))
    gu = FG.Grids.UnstructuredGrid(cart, vec(cx) .+ 0.1 .* sin.(1:120), vec(cy), fill(1.1, 120), mn)
    _cgef_engine_matches(gu, _CGEF_GA, 2.0, :NodeFilterPlan; reference = _cgef_cellset_filter, method = CGEF.Filtering.RealSpace())

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
        _cgef_engine_matches(build(hole), _CGEF_GA, ℓ, :NodeFilterPlan; reference = _cgef_cellset_filter)
    end
end

# A window sum read off a running sum is a difference of two numbers `row length / window` times larger
# than itself, and a periodic row is scanned over three copies of it. The rows are 10 apart and the
# support radius 3.5, so each target averages its own row; the reference is that mean summed in Float64
# from the same Float32 values.
Test.@testset "Prefix sums in Float32 on long periodic rows" begin
    Nx = 4096
    x = range(0.0f0; step = 1.0f0, length = Nx)
    window_mean(row, i) = sum(Float64(row[mod1(i + d, Nx)]) for d in -3:3) / 7
    geo = FG.Geometry.CartesianGeometry(Float32)
    y = range(0.0f0; step = 10.0f0, length = 2)
    f2 = Float32[1000 + sin(0.37f0 * i) + 0.5f0 * cos(1.3f0 * i + j) for i in 1:Nx, j in 1:2]
    f3 = Float32[1000 + sin(0.37f0 * i) + 0.5f0 * cos(1.3f0 * i + j - k) for i in 1:Nx, j in 1:2, k in 1:2]
    for (g, f) in ((FG.Grids.StructuredGrid(geo, x, y; periodic = (true, false)), f2),
                   (FG.Grids.StructuredGrid(geo, x, y, y; periodic = (true, false, false)), f3))
        ref = [window_mean(view(f, :, Tuple(I)[2:end]...), I[1]) for I in CartesianIndices(f)]
        for b in (_CGEF_SERIAL, CGEF.ComputationalBackends.GPUBackend(KA.CPU()))
            out = CGEF.Filtering.filter_apply!(similar(f), f, CGEF.Filtering.plan_filter(g, _CGEF_TH, 7.0f0; execution = b))
            Test.@test maximum(abs, out .- ref) <= 8 * eps(1000.0f0)
        end
    end
end

# On a sphere cut from a cell-centred lattice whose faces fall on the poles, the lattice continued past the
# cut is the removed rows, so `ZeroFill` divides by the kernel's mass over the whole lattice.
Test.@testset "ZeroFill on a sphere cut from a whole lattice" begin
    geo = FG.Geometry.SphericalGeometry(6.371e6)
    nlat, nlon = 18, 36
    h = π / nlat
    lat = range(-π / 2 + h / 2; step = h, length = nlat)
    lon = range(0, 2π; length = nlon + 1)[1:nlon]
    whole = FG.Grids.StructuredGrid(geo, lon, lat)
    rows = 4:(nlat - 3)
    f = [cos(2a) * sin(b) + 0.3 for a in lon, b in lat]
    ℓ = 3.0e6
    out = CGEF.Filtering.filter_field!(zeros(nlon, length(rows)), f[:, rows], FG.Grids.StructuredGrid(geo, lon, lat[rows]),
                                       _CGEF_GA, ℓ; execution = _CGEF_SERIAL)
    fz = [j in rows ? f[i, j] : 0.0 for i in 1:nlon, j in 1:nlat]
    ref = _cgef_cellset_filter(whole, _CGEF_GA, ℓ, fz)
    Test.@test _cgef_relerr(out, ref[:, rows]) < 1e-12
end

# A support wider than the sphere averages over all of it: the area-weighted mean of the grid under
# `Deformable`, and under `ZeroFill`, where the lattice continues to both poles and round the ring and
# tiles the rest of the sphere, the grid's integral over `4πR²`.
Test.@testset "A filter wider than the sphere" begin
    R = 6.371e6
    sph = FG.Geometry.SphericalGeometry(R)
    for (lon, lat, periodic) in (
        (range(0.0; step = deg2rad(10.0), length = 36), range(deg2rad(-80.0); step = deg2rad(5.0), length = 33), true),
        (range(0.0; step = deg2rad(10.0), length = 36), range(deg2rad(-88.5); step = deg2rad(2.0), length = 89), true),
        (range(0.0; step = deg2rad(7.0), length = 21), range(deg2rad(-60.0); step = deg2rad(5.0), length = 25), false),
    )
        g = FG.Grids.StructuredGrid(sph, lon, lat; periodic = (periodic, false))
        f = [sin(2a) * cos(b) + 2.0 for a in lon, b in lat]
        A = FG.Grids.measure(g)
        for (strategy, want) in ((CGEF.Filtering.Deformable(), sum(f .* A) / sum(A)),
                                 (CGEF.Filtering.ZeroFill(), sum(f .* A) / (4π * R^2)))
            out = CGEF.Filtering.filter_field!(zero(f), f, g, _CGEF_TH, 4π * R; mask_strategy = strategy)
            Test.@test maximum(abs, out .- want) < 1e-10 * want
        end
    end
end

# The flux budget commutes the filter with derivatives. `ZeroFill` keeps the whole kernel mass in its
# denominator, so it is one convolution of the zero-extended field everywhere, land targets included: it
# commutes with a difference taken over the whole lattice and conserves the domain integral. `Deformable`
# divides by the mass over water, which changes within ℓ of a coast.
Test.@testset "ZeroFill commutes with ∂/∂x and conserves the integral; Deformable does neither" begin
    N = 48
    xs = 0.0:1000.0:((N - 1) * 1000.0)
    coast = 31
    mask = trues(N, N)
    mask[coast:end, :] .= false
    cart = FG.Geometry.CartesianGeometry()
    grid = FG.Grids.StructuredGrid(cart, xs, xs, mask)
    lattice = FG.Grids.StructuredGrid(cart, xs, xs)
    ddx(a) = _cgef_dd(a, lattice, 1)
    # A C³ bump, zero from column 30 on, so it and its derivative vanish on land.
    f = [((r2 = ((i - 24)^2 + (j - 24)^2) / 36) < 1 ? (1 - r2)^4 : 0.0) for i in 1:N, j in 1:N]
    df = ddx(f)
    Test.@test all(iszero, f[coast:end, :]) && all(iszero, df[coast:end, :])
    gap = Dict{Any,Float64}()
    drift = Dict{Any,Float64}()
    for s in (CGEF.Filtering.ZeroFill(), CGEF.Filtering.Deformable())
        filt(a) = CGEF.Filtering.filter_field!(zeros(N, N), a, grid, _CGEF_TH, 6200.0; mask_strategy = s)
        rhs = ddx(filt(f))
        gap[s] = maximum(abs, filt(df) .- rhs) / maximum(abs, rhs)
        drift[s] = abs(sum(filt(f)) - sum(f)) / sum(f)
    end
    Test.@test gap[CGEF.Filtering.ZeroFill()] < 1e-13
    Test.@test drift[CGEF.Filtering.ZeroFill()] < 1e-13
    Test.@test gap[CGEF.Filtering.Deformable()] > 1e-3
    Test.@test drift[CGEF.Filtering.Deformable()] > 1e-6
end

# `AutoCache` keeps a scattered plan's neighbour cache exactly when its entries fit the budget, and one
# budget bounds a whole sweep: each scale gets what the scales before it left.
Test.@testset "The cache budget" begin
    g = FG.Grids.StructuredGrid(FG.Geometry.SphericalGeometry(6.371e6), deg2rad.(collect(0.0:7.5:352.5)),
                                deg2rad.(collect(range(-80.0, 80.0; length = 33))))
    entries(ℓ) = length(CGEF.Filtering.build_footprint(g, _CGEF_GA, ℓ; cache_strategy = CGEF.Filtering.AlwaysCache()).cache.w)
    bytes(ℓ) = entries(ℓ) * (2 * sizeof(Int) + sizeof(Float64))
    cached(fp) = fp.cache !== nothing
    full = bytes(500e3)
    for b in (full, full - 1)
        Test.@test cached(CGEF.Filtering.build_footprint(g, _CGEF_GA, 500e3; cache_strategy = CGEF.Filtering.AutoCache(),
                                                         cache_byte_budget = b)) == (b == full)
    end
    scales = [400e3, 500e3, 600e3]
    budget = bytes(scales[1]) + bytes(scales[2])
    Test.@test bytes(scales[3]) <= budget
    plans = CGEF.Filtering.plan_filter_sweep(g, _CGEF_GA, scales; cache_byte_budget = budget, execution = _CGEF_SERIAL)
    Test.@test [cached(p.footprint) for p in plans] == [true, true, false]
end

# A `HighOrderKernel{P}`'s weights on a grid are corrected so their moments of degree 1…P vanish, so the
# filter reproduces every polynomial of degree ≤ P wherever its support stays on the grid; off separable
# axes the correction runs over the cells each point has, so there every point reproduces them.
Test.@testset "HighOrderKernel on a grid reproduces polynomials of degree ≤ P" begin
    cart = FG.Geometry.CartesianGeometry()
    N = 32
    xs = 0.0:1.0:(N - 1)
    xv = collect(xs) .+ [0.03 * sin(2.3i) for i in 1:N]
    for order in (3, 5), ℓ in (4.0, 11.3), ax in (xs, xv)
        k = CGEF.Kernels.HighOrderKernel(; order)
        g = FG.Grids.StructuredGrid(cart, ax, ax)
        rad = CGEF.Kernels.kernel_radius(k, ℓ)
        inside = [min(ax[i] - ax[1], ax[end] - ax[i], ax[j] - ax[1], ax[end] - ax[j]) > rad + 2 for i in 1:N, j in 1:N]
        Test.@test count(inside) > 50
        plan = CGEF.Filtering.plan_filter(g, k, ℓ; execution = _CGEF_SERIAL)
        for (a, b) in vcat([(n, 0) for n in 0:order], [(0, n) for n in 1:order], [(order, order)])
            f = [(ax[i] / N)^a * (ax[j] / N)^b for i in 1:N, j in 1:N]
            Test.@test maximum(abs, (CGEF.Filtering.filter_apply!(zeros(N, N), f, plan) .- f)[inside]) < 1e-12
        end
    end

    n = 24
    xg = [i + 0.25j + 0.03i^1.2 for i in 1:n, j in 1:n]
    yg = [j + 0.02i + 0.01j^1.3 for i in 1:n, j in 1:n]
    cg = FG.Grids.CurvilinearGrid(cart, xg, yg, trues(n, n))
    ug = FG.Grids.UnstructuredGrid(cart, vec(xg), vec(yg), ones(n * n), trues(n * n))
    for order in (3, 5)
        k = CGEF.Kernels.HighOrderKernel(; order)
        pc = CGEF.Filtering.plan_filter(cg, k, 7.0; execution = _CGEF_SERIAL)
        pu = CGEF.Filtering.plan_filter(ug, k, 7.0; execution = _CGEF_SERIAL)
        for a in 0:order, b in 0:(order - a)
            f = ((xg .- 12) ./ 12) .^ a .* ((yg .- 12) ./ 12) .^ b
            Test.@test maximum(abs, CGEF.Filtering.filter_apply!(zeros(n, n), f, pc) .- f) < 1e-10
            Test.@test maximum(abs, CGEF.Filtering.filter_apply!(zeros(n * n), vec(f), pu) .- vec(f)) < 1e-10
        end
    end
end

# On a sphere one correction serves a latitude row where its targets share their cells, so the banded
# engine and the per-point engine, cached, streamed or on the device, are one filter; and a smooth field's
# error grows as `ℓ^(P+1)` against the Gaussian's `ℓ²`.
Test.@testset "HighOrderKernel on a sphere: one filter whichever engine runs it, order P + 1" begin
    sph = FG.Geometry.SphericalGeometry(6.371e6)
    for (λ, φ, ℓ) in ((range(0, 2π; length = 73)[1:72], range(deg2rad(-70), deg2rad(70); length = 29), 2.0e6),
                      (range(deg2rad(10), deg2rad(70); length = 31), range(deg2rad(-20), deg2rad(20); length = 21), 1.0e6))
        gv = FG.Grids.StructuredGrid(sph, collect(λ), collect(φ))
        f = [cos(p) * cos(2l) + 0.3sin(p)^2 for l in λ, p in φ]
        for order in (3, 5)
            k = CGEF.Kernels.HighOrderKernel(; order)
            pb = CGEF.Filtering.plan_filter(FG.Grids.StructuredGrid(sph, λ, φ), k, ℓ; execution = _CGEF_SERIAL)
            pb.footprint isa CGEF.Filtering.FilterFootprint || error("a uniform-longitude grid builds $(typeof(pb.footprint))")
            ob = CGEF.Filtering.filter_apply!(similar(f), f, pb)
            oc = CGEF.Filtering.filter_apply!(similar(f), f,
                                              CGEF.Filtering.plan_filter(gv, k, ℓ; execution = _CGEF_SERIAL,
                                                                         cache_strategy = CGEF.Filtering.AlwaysCache()))
            Test.@test maximum(abs, ob .- oc) < 1e-13 * maximum(abs, f)
            for b in (_CGEF_SERIAL, CGEF.ComputationalBackends.GPUBackend(KA.CPU()))
                p = CGEF.Filtering.plan_filter(gv, k, ℓ; execution = b, cache_strategy = CGEF.Filtering.NeverCache())
                Test.@test CGEF.Filtering.filter_apply!(similar(f), f, p) == oc
            end
        end
    end

    λ = range(0, 2π; length = 361)[1:360]
    φ = range(deg2rad(-75), deg2rad(75); length = 151)
    g = FG.Grids.StructuredGrid(sph, λ, φ)
    f = [cos(p) * cos(l) + sin(p) for l in λ, p in φ]
    interior = [abs(p) < deg2rad(40) for l in λ, p in φ]
    ℓs = (8.0e5, 1.2e6, 1.8e6)
    for (k, order) in ((_CGEF_GA, 2), (CGEF.Kernels.HighOrderKernel(; order = 3), 4), (CGEF.Kernels.HighOrderKernel(; order = 5), 6))
        errs = map(ℓ -> maximum(abs, (CGEF.Filtering.filter_apply!(similar(f), f,
                       CGEF.Filtering.plan_filter(g, k, ℓ; execution = _CGEF_SERIAL)) .- f)[interior]), ℓs)
        for i in 1:2
            Test.@test log(errs[i + 1] / errs[i]) / log(ℓs[i + 1] / ℓs[i]) ≈ order atol = 0.2
        end
    end
end

# On a periodic lattice with `k_c ≤ π/Δx` the real-space brick wall transforms to the transfer function of
# its truncated footprint, whose ripple the dropped Bessel tail bounds: in two dimensions
# `√(k_c/k)/π · (1/(|k_c − k|R) + 1/((k_c + k)R))`, 0.039 at `k_c/2` and 0.010 at `2k_c`; in one
# `(1/π)(1/(|k_c − k|R) + 1/((k_c + k)R))`, 0.028 and 0.014.
Test.@testset "SharpSpectralKernel in real space passes and stops Fourier modes" begin
    N = 64
    xs = 0.0:1.0:(N - 1)
    geo = FG.Geometry.CartesianGeometry()
    g1 = FG.Grids.StructuredGrid(geo, xs; periodic = true)
    g2 = FG.Grids.StructuredGrid(geo, xs, xs; periodic = (true, true))
    gain(o, f) = sum(o .* f) / sum(f .* f)
    for (m, target, tol) in ((4, 1.0, 0.05), (16, 0.0, 0.02))
        k = 2π * m / N
        f1 = cos.(k .* xs)
        f2 = [cos(k * a) for a in xs, _ in xs]
        o1 = CGEF.Filtering.filter_field!(zeros(N), f1, g1, _CGEF_SS, 4.0)
        o2 = CGEF.Filtering.filter_field!(zeros(N, N), f2, g2, _CGEF_SS, 4.0)
        Test.@test abs(gain(o1, f1) - target) < tol
        Test.@test abs(gain(o2, f2) - target) < tol
        Test.@test o2 ≈ gain(o2, f2) .* f2 atol = 1e-12
    end
end

# `check_setup` measures a sphere's spacing in metres: along longitude the chord-angle distance
# `2R·asin(cos φ·sin(Δλ/2))`, finest at the most poleward row and coarsest at the equator, which decides
# whether ℓ is resolvable.
Test.@testset "check_setup on a sphere" begin
    R = 6.371e6
    d = deg2rad(1.0)
    g = FG.Grids.StructuredGrid(FG.Geometry.SphericalGeometry(R), range(0.0; step = d, length = 360),
                                range(deg2rad(-60.0), deg2rad(60.0); step = d / 4); periodic = (true, false))
    r = CGEF.check_setup(g, _CGEF_GA, 500e3)
    Test.@test r.coarsest_spacing[1] ≈ R * d rtol = 1e-12
    Test.@test r.spacing[1] ≈ 2R * asin(cos(deg2rad(60.0)) * sin(d / 2)) rtol = 1e-12
    Test.@test r.spacing[2] ≈ R * d / 4 rtol = 1e-12
    Test.@test r.cells_per_scale[1] ≈ 500e3 / (R * d) rtol = 1e-12
    Test.@test r.resolvable
    Test.@test !CGEF.check_setup(g, _CGEF_GA, 150e3).resolvable
end
