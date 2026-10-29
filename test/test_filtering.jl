
# The real-space filter against its definition. Each engine, on grids that reach it, under both mask
# strategies and on every execution backend, equals the direct sum over the lattice continued past the
# grid's bounded edges (`_cgef_direct_filter`) or over the grid's own cells (`_cgef_cellset_filter`).

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
    :BinnedTopHatPlan => "binned prefix-sum top-hat",
    :NUFFTSumFootprint => "nonuniform-FFT kernel sum",
)

_cgef_alloc_apply(out, f, plan) = @allocated CGEF.Filtering.filter_apply!(out, f, plan)
_cgef_alloc_batch(outs, fs, plan) = @allocated CGEF.Filtering.filter_apply_batch!(outs, fs, plan)

# `engine` is the footprint type the serial plan must build. `check_setup` reports the engine Auto takes, so a
# row that names its engine skips that check.
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
        haskey(kw, :engine) ||
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

_cgef_engine_matches(r) =
    _cgef_engine_matches(r.grid, r.kernel, r.ℓ, r.engine; reference = r.reference, rtol = r.rtol, r.kw...)

# The device, Distributed and MPI plans of every grid take the kernel's own engine.
Test.@testset "Real-space engines on Cartesian grids" begin
    foreach(_cgef_engine_matches, _cgef_cartesian_rows())
end

Test.@testset "Real-space engines on spherical grids" begin
    foreach(_cgef_engine_matches, _cgef_spherical_rows())
end

Test.@testset "Real-space engines on meshes, node sets and sphere pixelizations" begin
    foreach(_cgef_engine_matches, _cgef_mesh_rows())
end

# A Gaussian over points at a scale near the domain's takes the nonuniform-FFT kernel sum. It continues the
# kernel past its truncation radius, where it is below `Kernels.truncation_tol(Float64)` (1e-10) of its peak,
# and runs its transforms at that tolerance, so it agrees with the truncated sum to a few times 1e-10.
Test.@testset "The nonuniform-FFT kernel sum over points" begin
    cart = FG.Geometry.CartesianGeometry()
    M = 1000
    xy = (20 .* mod.(0.6180339887 .* (1:M), 1.0), 20 .* mod.(0.7548776662 .* (1:M) .+ 0.31, 1.0))
    m = [mod(k, 13) != 0 for k in 1:M]
    f = sin.(0.3 .* xy[1]) .+ cos.(0.2 .* xy[2]) .+ 0.2
    for g in (FG.Grids.UnstructuredGrid(cart, xy, fill(400 / M, M), m),
              FG.Grids.UnstructuredGrid(cart, xy, fill(400 / M, M), m; topology = (FG.Grids.Periodic(), FG.Grids.Periodic()), period = (20.0, 20.0))),
        ℓ in (6.0, 12.0), s in (CGEF.Filtering.ZeroFill(), CGEF.Filtering.Deformable())
        ref = _cgef_cellset_filter(g, _CGEF_GA, ℓ, f, s)
        plan(b) = CGEF.Filtering.plan_filter(g, _CGEF_GA, ℓ; mask_strategy = s, execution = b)
        Test.@test plan(_CGEF_SERIAL).footprint isa CGEF.Filtering.NUFFTSumFootprint
        for b in (_CGEF_SERIAL, _CGEF_PARALLEL...)
            Test.@test _cgef_relerr(CGEF.Filtering.filter_apply!(zero(f), f, plan(b)), ref) < 1e-9
        end
    end
end

# A window sum read off a running sum is a difference of two numbers `row length / window` times larger
# than itself, and a periodic row is scanned over three copies of it. The rows are 10 apart, their cells 10
# wide, and the support radius 3.5, so each target averages its own row: the centre test the 7 cells within
# 3.5, the cell overlap each cell by its part inside the disk or ball. The reference is that mean summed in
# Float64 from the same Float32 values.
Test.@testset "Prefix sums in Float32 on long periodic rows" begin
    Nx = 4096
    x = range(0.0f0; step = 1.0f0, length = Nx)
    nodes = CGEF.Filtering._overlap_nodes(Float64)
    centre(d, _) = abs(d) <= 3 ? 1.0 : 0.0
    overlap(d, N) = N == 2 ? CGEF.Geometry.disk_fraction(d - 0.5, d + 0.5, -5.0, 5.0, 3.5) :
                    CGEF.Geometry.ball_box_fraction(d - 0.5, d + 0.5, -5.0, 5.0, -5.0, 5.0, 3.5, nodes)
    window_mean(row, i, w, N) = sum(w(d, N) * Float64(row[mod1(i + d, Nx)]) for d in -4:4) / sum(d -> w(d, N), -4:4)
    geo = FG.Geometry.CartesianGeometry(Float32)
    y = range(0.0f0; step = 10.0f0, length = 2)
    f2 = Float32[1000 + sin(0.37f0 * i) + 0.5f0 * cos(1.3f0 * i + j) for i in 1:Nx, j in 1:2]
    f3 = Float32[1000 + sin(0.37f0 * i) + 0.5f0 * cos(1.3f0 * i + j - k) for i in 1:Nx, j in 1:2, k in 1:2]
    for (g, f) in ((FG.Grids.StructuredGrid(geo, x, y; topology = (FG.Grids.Periodic(), FG.Grids.Bounded())), f2),
                   (FG.Grids.StructuredGrid(geo, x, y, y; topology = (FG.Grids.Periodic(), FG.Grids.Bounded(), FG.Grids.Bounded())), f3)),
        (k, w) in ((_CGEF_THC, centre), (_CGEF_TH, overlap))
        ref = [window_mean(view(f, :, Tuple(I)[2:end]...), I[1], w, ndims(f)) for I in CartesianIndices(f)]
        for b in (_CGEF_SERIAL, CGEF.ComputationalBackends.GPUBackend(KA.CPU()))
            p = CGEF.Filtering.plan_filter(g, k, 7.0f0; execution = b, engine = CGEF.Filtering.FootprintEngine())
            Test.@test maximum(abs, CGEF.Filtering.filter_apply!(similar(f), f, p) .- ref) <= 8 * eps(1000.0f0)
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
    for (lon, lat, tλ) in (
        (range(0.0; step = deg2rad(10.0), length = 36), range(deg2rad(-80.0); step = deg2rad(5.0), length = 33),
         FG.Grids.Periodic()),
        (range(0.0; step = deg2rad(10.0), length = 36), range(deg2rad(-88.5); step = deg2rad(2.0), length = 89),
         FG.Grids.Periodic()),
        (range(0.0; step = deg2rad(7.0), length = 21), range(deg2rad(-60.0); step = deg2rad(5.0), length = 25),
         FG.Grids.Bounded()),
    )
        g = FG.Grids.StructuredGrid(sph, lon, lat; topology = (tλ, FG.Grids.Bounded()))
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
    # A C³ bump, zero from column 29 on, so it and its derivative on the default 5-node stencil vanish on land.
    f = [((r2 = ((i - 24)^2 + (j - 24)^2) / 25) < 1 ? (1 - r2)^4 : 0.0) for i in 1:N, j in 1:N]
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
    # A node set's list has no streaming form: one past the budget is refused before it is allocated.
    Test.@test_throws ArgumentError CGEF.Filtering._check_list_bytes(10^10, 10^6, Float64)
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

# On a sphere one correction serves a latitude row where its targets share their cells, so the row
# engines (the banded sum, and the transform along longitude where that costs fewer operations) and the
# per-point engine, cached, streamed or on the device, are one filter; and a smooth field's error grows
# as `ℓ^(P+1)` against the Gaussian's `ℓ²`.
Test.@testset "HighOrderKernel on a sphere: one filter whichever engine runs it, order P + 1" begin
    sph = FG.Geometry.SphericalGeometry(6.371e6)
    GPU = CGEF.ComputationalBackends.GPUBackend(KA.CPU())
    # A ring and a regional box, at a scale whose support holds more cells than the correction has
    # conditions, so each target's correction is determined.
    for (λ, φ, tλ, ℓ) in (
        (FG.Axes.PeriodicUniformAxis(0.0, 2π, 72), range(deg2rad(-70), deg2rad(70); length = 29), FG.Grids.Periodic(),
         3.0e6),
        (range(deg2rad(10), deg2rad(70); length = 31), range(deg2rad(-20), deg2rad(20); length = 21), FG.Grids.Bounded(),
         1.5e6),
    )
        gr = FG.Grids.StructuredGrid(sph, λ, φ; topology = (tλ, FG.Grids.Bounded()))
        gv = FG.Grids.StructuredGrid(sph, collect(λ), collect(φ); topology = (tλ, FG.Grids.Bounded()))
        f = [cos(p) * cos(2l) + 0.3sin(p)^2 for l in λ, p in φ]
        for order in (3, 5)
            k = CGEF.Kernels.HighOrderKernel(; order)
            Test.@test nameof(typeof(CGEF.Filtering.plan_filter(gr, k, ℓ; execution = _CGEF_SERIAL).footprint)) ===
                       :ZonalFFTFootprint
            oc = CGEF.Filtering.filter_apply!(similar(f), f,
                                              CGEF.Filtering.plan_filter(gv, k, ℓ; execution = _CGEF_SERIAL,
                                                                         cache_strategy = CGEF.Filtering.AlwaysCache()))
            # The uniform-longitude grid's plans take the transform on the host and, FFTW serving host memory, on
            # `KA.CPU()`.
            for b in (_CGEF_SERIAL, GPU)
                o = CGEF.Filtering.filter_apply!(similar(f), f, CGEF.Filtering.plan_filter(gr, k, ℓ; execution = b))
                Test.@test maximum(abs, o .- oc) < 1e-13 * maximum(abs, f)
            end
            for b in (_CGEF_SERIAL, GPU)
                p = CGEF.Filtering.plan_filter(gv, k, ℓ; execution = b, cache_strategy = CGEF.Filtering.NeverCache())
                Test.@test CGEF.Filtering.filter_apply!(similar(f), f, p) == oc
            end
        end
    end

    λ = range(0, 2π; length = 361)[1:360]
    φ = range(deg2rad(-75), deg2rad(75); length = 151)
    g = FG.Grids.StructuredGrid(sph, λ, φ; topology = (FG.Grids.Periodic(), FG.Grids.Bounded()))
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
    g1 = FG.Grids.StructuredGrid(geo, xs; topology = FG.Grids.Periodic())
    g2 = FG.Grids.StructuredGrid(geo, xs, xs; topology = (FG.Grids.Periodic(), FG.Grids.Periodic()))
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
                                range(deg2rad(-60.0), deg2rad(60.0); step = d / 4); topology = (FG.Grids.Periodic(), FG.Grids.Bounded()))
    r = CGEF.check_setup(g, _CGEF_GA, 500e3)
    Test.@test r.coarsest_spacing[1] ≈ R * d rtol = 1e-12
    Test.@test r.spacing[1] ≈ 2R * asin(cos(deg2rad(60.0)) * sin(d / 2)) rtol = 1e-12
    Test.@test r.spacing[2] ≈ R * d / 4 rtol = 1e-12
    Test.@test r.cells_per_scale[1] ≈ 500e3 / (R * d) rtol = 1e-12
    Test.@test r.resolvable
    Test.@test !CGEF.check_setup(g, _CGEF_GA, 150e3).resolvable
end

# Periodic ranges whose cells leave a gap at the wrap (a period other than `n·Δ`, a longitude range short of
# the circle) and ones they close evenly: every engine, chosen or named, filters as the cell-by-cell plan does,
# which finds neighbours by the grid's own period.
Test.@testset "Periodic ranges with and without a gap at the wrap" begin
    P, B = FG.Grids.Periodic(), FG.Grids.Bounded()
    cart = FG.Geometry.CartesianGeometry()
    sph = FG.Geometry.SphericalGeometry(1.0)
    F = CGEF.Filtering
    for (g, ℓ) in (
        (FG.Grids.StructuredGrid(cart, 0.0:1.0:15.0, 0.0:0.7:7.7; topology = (P, B), period = (19.0, nothing)), 2.5),
        (FG.Grids.StructuredGrid(cart, 0.0:1.0:15.0, 0.0:0.7:7.7; topology = (P, B)), 2.5),
        (FG.Grids.StructuredGrid(sph, range(0.0; step = 2π / 70, length = 64), range(-1.2, 1.2; length = 32);
                                 topology = (P, B)), 0.35),
        (FG.Grids.StructuredGrid(sph, range(0.0; step = 2π / 64, length = 64), range(-1.2, 1.2; length = 32);
                                 topology = (P, B)), 0.35),
    ), k in (_CGEF_GA, _CGEF_TH)
        sz = size(FG.Grids.mask(g))
        f = [sin(3i / sz[1]) * cos(2j / sz[2]) + 0.1i for i in 1:sz[1], j in 1:sz[2]]
        ref = F.apply_footprint!(zeros(sz), f, g, F._build_footprint_scattered(g, k, ℓ), F.ZeroFill(), false, false)
        for e in (F.AutoEngine(), F.FootprintEngine())
            p = F.plan_filter(g, k, ℓ; execution = CGEF.ComputationalBackends.SerialBackend(), engine = e)
            Test.@test maximum(abs, F.filter_apply!(zeros(sz), f, p) .- ref) <= 1e-8 * maximum(abs, ref)
        end
    end
end

# The padded- and zonal-FFT engines planned for `GPUBackend(KA.CPU())`, whose memory FFTW serves, apply by
# broadcasts: within round-off of the serial plan's row loops, both within round-off of the cell-by-cell
# footprint. A named FFT engine on a stretched axis raises.
Test.@testset "FFT engines on a device backend" begin
    F = CGEF.Filtering
    CB = CGEF.ComputationalBackends
    cart = FG.Geometry.CartesianGeometry()
    sph = FG.Geometry.SphericalGeometry(1.0)
    P = FG.Grids.Periodic()
    gpu = CB.GPUBackend(KA.CPU())
    ringλ = FG.Axes.PeriodicUniformAxis(0.0, 2π, 72)
    ringφ = collect(range(-1.4, 1.4; length = 36))
    hole(sz) = (m = trues(sz); m[5:9, 4:8] .= false; m)
    box = FG.Grids.StructuredGrid(sph, range(0.2, 1.6; length = 40), collect(range(-0.6, 0.6; length = 30)))
    for (g, k, ℓ, e) in (
        (FG.Grids.StructuredGrid(cart, 0.0:1.0:23.0, 0.0:0.7:11.9; topology = (P, P)), _CGEF_GA, 4.0, F.PaddedFFTEngine()),
        (FG.Grids.StructuredGrid(cart, 0.0:1.0:23.0, 0.0:0.7:11.9, hole((24, 18))), CGEF.Kernels.SmoothHatKernel(), 3.5,
         F.PaddedFFTEngine()),
        (FG.Grids.StructuredGrid(cart, 0.0:1.0:11.0, 0.0:1.0:9.0, 0.0:1.0:7.0), _CGEF_TH, 2.5, F.PaddedFFTEngine()),
        (FG.Grids.StructuredGrid(cart, 0.0:1.0:63.0), _CGEF_TH, 5.0, F.PaddedFFTEngine()),
        (FG.Grids.StructuredGrid(sph, ringλ, ringφ), _CGEF_GA, 0.3, F.ZonalFFTEngine()),
        (FG.Grids.StructuredGrid(sph, ringλ, ringφ, hole((72, 36))), _CGEF_TH, 0.3, F.ZonalFFTEngine()),
        (box, _CGEF_GA, 0.15, F.ZonalFFTEngine()),
        (FG.Grids.StructuredGrid(sph, ringλ, ringφ), CGEF.Kernels.HighOrderKernel(), 0.3, F.ZonalFFTEngine()),
    ), s in (F.ZeroFill(), F.Deformable())
        sz = size(FG.Grids.mask(g))
        f = _cgef_share_field(sz)
        apply(b, e) = F.filter_apply!(zeros(sz), f, F.plan_filter(g, k, ℓ; execution = b, engine = e, mask_strategy = s))
        os = apply(CB.SerialBackend(), e)
        Test.@test maximum(abs, apply(gpu, e) .- os) <= 1e-13 * maximum(abs, os)
        or = apply(CB.SerialBackend(), F.FootprintEngine())
        Test.@test maximum(abs, os .- or) <= 1e-13 * maximum(abs, or)
        # A planner that times its candidates writes the arrays it plans on, kernel samples and zero padding.
        om = F.filter_apply!(zeros(sz), f, F.plan_filter(g, k, ℓ; execution = CB.SerialBackend(), engine = e, mask_strategy = s,
                                                         fft_options = (; flags = FFTW.MEASURE)))
        Test.@test maximum(abs, om .- os) <= 1e-13 * maximum(abs, os)
    end
    for (g, e) in ((FG.Grids.StructuredGrid(sph, collect(range(0.2, 1.6; length = 40)), collect(range(-0.6, 0.6; length = 30))),
                    F.ZonalFFTEngine()),
                   (FG.Grids.StructuredGrid(cart, cumsum(0.5 .+ (1:20) ./ 20), 0.0:1.0:9.0), F.PaddedFFTEngine()))
        Test.@test_throws ArgumentError F.plan_filter(g, _CGEF_GA, 0.3; execution = CB.SerialBackend(), engine = e)
    end
end
