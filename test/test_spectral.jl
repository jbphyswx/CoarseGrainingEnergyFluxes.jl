
# `Spectral()`: the transform of the field times the kernel's transfer function, against closed forms, the
# zero-extended continuum convolution, and one transform against another on the same points.

const _CGEF_SPECTRAL = CGEF.Filtering.Spectral()
const _CGEF_NUFFTS = (FTB.FINUFFTBackend(), FTB.NonuniformFFTsBackend())

# A plan built for a batch filters each slice as the single-field apply does, and a sweep that analyzes
# once equals applying each scale's plan.
function _cgef_batch_and_sweep(grid, f, ℓs; rtol = 1e-12, kw...)
    k = CGEF.GaussianKernel()
    pb = CGEF.Filtering.plan_filter(grid, k, first(ℓs); method = _CGEF_SPECTRAL, batch = 2, kw...)
    d = ndims(f) + 1
    F = cat(f, 2 .* f .+ 1; dims = d)
    ob = CGEF.Filtering.filter_apply_batched!(zero(F), F, pb)
    for b in 1:2
        Test.@test _cgef_relerr(selectdim(ob, d, b), CGEF.Filtering.filter_apply!(zero(f), selectdim(F, d, b), pb)) < rtol
    end
    plans = CGEF.Filtering.plan_filter_sweep(grid, k, ℓs; method = _CGEF_SPECTRAL, kw...)
    F̂ = CGEF.Filtering.filter_analyze!(CGEF.Filtering.analyze_buffer(plans[1], f), f, plans[1])
    for p in plans
        Test.@test _cgef_relerr(CGEF.Filtering.filter_synthesize!(zero(f), F̂, p), CGEF.Filtering.filter_apply!(zero(f), f, p)) < rtol
    end
end

Test.@testset "FFT: a Fourier mode is scaled by the transfer function" begin
    N = 32
    xs = 0.0:1.0:(N - 1)
    g = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), xs, xs; periodic = (true, true))
    kv = (2π * 3 / N, 2π * 5 / N)
    k = hypot(kv...)
    mode = [cos(kv[1] * x + kv[2] * y) for x in xs, y in xs]
    x = k * 2.0
    cases = ((CGEF.GaussianKernel(), 4.0, exp(-k^2 * 16 / 24)),
             (CGEF.TopHatKernel(), 4.0, 2 * SpecialFunctions.besselj1(x) / x),
             (CGEF.SharpSpectralKernel(), 2.0, 1.0),            # k_c = π/2 above |k|
             (CGEF.SharpSpectralKernel(), 4.0, 0.0),            # k_c = π/4 below it
             (CGEF.Kernels.HighOrderKernel(; order = 3), 8.0, CGEF.Kernels.spectral_transfer(CGEF.Kernels.HighOrderKernel(; order = 3), kv, 8.0)),
             (CGEF.Kernels.HighOrderKernel(; order = 5), 8.0, CGEF.Kernels.spectral_transfer(CGEF.Kernels.HighOrderKernel(; order = 5), kv, 8.0)))
    for (kern, ℓ, Ĝ) in cases
        out = CGEF.Filtering.filter_field!(zeros(N, N), mode, g, kern, ℓ; method = _CGEF_SPECTRAL)
        Test.@test maximum(abs, out .- Ĝ .* mode) < 1e-12
    end
end

# A bounded axis is zero-padded and a masked cell holds zero, so `Spectral()` is the sum of the continuum
# kernel `G(r) = α/(πℓ²) exp(-α r²/ℓ²)` against the field extended by zero, over the grid and its images
# along a periodic axis. At ℓ = 8Δx the transfer at Nyquist is below 1e-11, so transform and sum agree to
# round-off.
Test.@testset "FFT on bounded, periodic and masked axes: the zero-extended continuum sum" begin
    Nx, Ny = 32, 24
    dx, dy = 1.0, 0.7
    x = 0.0:dx:(dx * (Nx - 1))
    y = 0.0:dy:(dy * (Ny - 1))
    α = 6.0
    ℓ = 8.0
    G(r) = α / (π * ℓ^2) * exp(-α * r^2 / ℓ^2)
    f = [1 + 0.02xi - 0.05yj + sin(0.4xi) * cos(0.3yj) for xi in x, yj in y]
    hole = trues(Nx, Ny)
    hole[12:16, 9:12] .= false
    function direct(mask, periodic)
        num = zeros(Nx, Ny)
        den = zeros(Nx, Ny)
        qx = periodic[1] ? (-2:2) : (0:0)
        qy = periodic[2] ? (-3:3) : (0:0)
        for py in qy, px in qx, jj in 1:Ny, ii in 1:Nx, j in 1:Ny, i in 1:Nx
            w = G(hypot(x[i] - x[ii] - px * Nx * dx, y[j] - y[jj] - py * Ny * dy)) * dx * dy * mask[ii, jj]
            num[i, j] += w * f[ii, jj]
            den[i, j] += w
        end
        return num, den
    end
    for (periodic, mask) in (((false, false), trues(Nx, Ny)), ((false, false), hole), ((true, false), hole),
                             ((true, true), hole))
        grid = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), x, y, mask; periodic)
        num, den = direct(mask, periodic)
        spectral(s) = CGEF.Filtering.filter_field!(zeros(Nx, Ny), f, grid, CGEF.GaussianKernel(), ℓ;
                                                   method = _CGEF_SPECTRAL, mask_strategy = s)
        Test.@test _cgef_relerr(spectral(CGEF.Filtering.ZeroFill()), num) < 1e-12
        Test.@test _cgef_relerr(spectral(CGEF.Filtering.Deformable()), ifelse.(mask, num ./ den, 0.0)) < 1e-12
    end
    _cgef_batch_and_sweep(FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), x, y, hole), f, [4.0, 8.0])
end

# On a lattice the nonuniform transforms are the FFT: declared periodic, the box is the grid's period;
# otherwise it is padded as FFTW pads a bounded axis. A masked point contributes nothing, whatever it holds.
Test.@testset "NUFFT on a lattice equals FFT: $(nameof(typeof(lib)))" for lib in _CGEF_NUFFTS
    Nx, Ny = 32, 24
    geo = FG.Geometry.CartesianGeometry()
    x = 0.0:1.0:(Nx - 1)
    y = 0.0:1.0:(Ny - 1)
    u = [sin(2π * xi / Nx) + 0.5cos(4π * yj / Ny) + 0.3sin(6π * xi / Nx) for xi in x, yj in y]
    mask = trues(Nx, Ny)
    mask[8:11, 8:11] .= false
    un = copy(vec(u))
    un[.!vec(mask)] .= NaN
    px = vec([xi for xi in x, _ in y])
    py = vec([yj for _ in x, yj in y])
    nodes(m; kw...) = FG.Grids.UnstructuredGrid(geo, px, py, ones(Nx * Ny), vec(m); kw...)
    for (structured, scattered, field) in (
        (FG.Grids.StructuredGrid(geo, x, y; periodic = (true, true)),
         nodes(trues(Nx, Ny); periodic = (true, true), period = (Float64(Nx), Float64(Ny))), vec(u)),
        (FG.Grids.StructuredGrid(geo, x, y), nodes(trues(Nx, Ny)), vec(u)),
        (FG.Grids.StructuredGrid(geo, x, y, mask), nodes(mask), un),
    ), kern in (CGEF.GaussianKernel(), CGEF.TopHatKernel()), s in (CGEF.Filtering.ZeroFill(), CGEF.Filtering.Deformable())
        ref = CGEF.Filtering.filter_field!(zeros(Nx, Ny), u, structured, kern, 4.0; method = _CGEF_SPECTRAL, mask_strategy = s)
        got = CGEF.Filtering.filter_field!(zeros(Nx * Ny), field, scattered, kern, 4.0; method = _CGEF_SPECTRAL,
                                           transform = lib, mask_strategy = s)
        Test.@test maximum(abs, got .- vec(ref)) < 1e-6
    end
    _cgef_batch_and_sweep(nodes(mask), un, [2.0, 4.0]; transform = lib)
end

# In one and three dimensions the top-hat of radius R = ℓ/2 transforms to the ball's transfer function,
# `sin x / x` and `3(sin x − x cos x)/x³` at x = |k|R.
Test.@testset "NUFFT in one and three dimensions: $(nameof(typeof(lib)))" for lib in _CGEF_NUFFTS
    geo = FG.Geometry.CartesianGeometry()
    th = CGEF.TopHatKernel()
    ℓ = 3.0
    R = ℓ / 2
    nu = (method = _CGEF_SPECTRAL, transform = lib)
    n = 40
    x = collect(0.0:1.0:(n - 1))
    g1 = FG.Grids.UnstructuredGrid(geo, (x,), ones(n), trues(n); periodic = (true,), period = (40.0,))
    k = 2π * 3 / 40
    Test.@test CGEF.Filtering.filter_field!(zeros(n), cos.(k .* x), g1, th, ℓ; nu...) ≈ sin(k * R) / (k * R) .* cos.(k .* x) atol = 1e-8
    ns = (12, 10, 8)
    ax = ntuple(d -> collect(0.0:1.0:(ns[d] - 1)), 3)
    pts = ntuple(d -> vec([(a, b, c)[d] for a in ax[1], b in ax[2], c in ax[3]]), 3)
    g3 = FG.Grids.UnstructuredGrid(geo, pts, ones(prod(ns)), trues(prod(ns)); periodic = (true, true, true), period = Float64.(ns))
    k3 = (2π / ns[1], 4π / ns[2], 2π / ns[3])
    f3 = cos.(k3[1] .* pts[1]) .* cos.(k3[2] .* pts[2]) .* cos.(k3[3] .* pts[3])
    x3 = hypot(k3...) * R
    Test.@test CGEF.Filtering.filter_field!(zeros(prod(ns)), f3, g3, th, ℓ; nu...) ≈ 3 * (sin(x3) - x3 * cos(x3)) / x3^3 .* f3 atol = 1e-8
end

# On scattered points each point's share of the box `A_j/(Lx·Ly)` is the quadrature rule for the Fourier
# coefficient. On a lattice twice as dense in its left half, with its periodic cell areas, that rule is
# the trapezoid rule across a spacing jump, second order in `h`; equal areas weigh the dense half twice at
# any `h`.
Test.@testset "NUFFT weighs each point by its cell area: $(nameof(typeof(lib)))" for lib in _CGEF_NUFFTS
    geo = FG.Geometry.CartesianGeometry()
    k0 = (2π, 4π)
    ℓ = 0.2
    function errors(h)
        xs = vcat(collect(0.0:(h / 2):(0.5 - h / 2)), collect(0.5:h:(1 - h)))
        ys = collect(0.0:h:(1 - h))
        nx = length(xs)
        wx = [(xs[mod1(i + 1, nx)] - xs[mod1(i - 1, nx)] + (i == 1) + (i == nx)) / 2 for i in 1:nx]
        px = vec([a for a in xs, _ in ys])
        py = vec([b for _ in xs, b in ys])
        n = length(px)
        f = @. cos(k0[1] * px) * cos(k0[2] * py)
        ref = CGEF.Kernels.spectral_transfer(CGEF.GaussianKernel(), hypot(k0...), ℓ) .* f
        err(A) = maximum(abs, CGEF.Filtering.filter_field!(zeros(n), f,
                     FG.Grids.UnstructuredGrid(geo, px, py, A, trues(n); periodic = (true, true), period = (1.0, 1.0)),
                     CGEF.GaussianKernel(), ℓ; method = _CGEF_SPECTRAL, transform = lib) .- ref)
        return err(vec([w * h for w in wx, _ in ys])), err(fill(1 / n, n))
    end
    weighted1, equal1 = errors(1 / 32)
    weighted2, equal2 = errors(1 / 64)
    Test.@test weighted1 / weighted2 > 3
    Test.@test equal1 / equal2 < 1.5
    Test.@test equal2 > 10 * weighted2
end

# A spherical harmonic of degree l is scaled by the kernel's Legendre transform at l, through
# FastSphericalHarmonics on its own lattice and through NUFSHT's fit on the same nodes taken as scattered
# points.
Test.@testset "Spherical harmonics are scaled by Ĝ_l" begin
    N = 13
    M = 2N - 1
    R = 6.371e6
    geo = FG.Geometry.SphericalGeometry(R)
    ylm(l, m) = (C = zeros(N, M); C[FSH.sph_mode(l, m)] = 1.0; FSH.sph_evaluate(C))
    Gl(k, l, ℓ) = _cgef_legendre_transfer(k, l, ℓ, R)
    kernels = (CGEF.GaussianKernel(), CGEF.TopHatKernel())

    cc = FG.Connectivity.structured_grid(FG.SphericalSampling.ClenshawCurtisSampling(), N; geometry = geo)
    for k in kernels, (l, m) in ((5, 2), (9, -4))
        f = permutedims(ylm(l, m))
        out = CGEF.Filtering.filter_field!(zeros(M, N), f, cc, k, 2e6; method = _CGEF_SPECTRAL)
        Test.@test maximum(abs, out .- Gl(k, l, 2e6) .* f) < 1e-12
    end
    _cgef_batch_and_sweep(cc, permutedims(ylm(5, 2) .+ 0.3 .* ylm(2, 1)), [1e6, 2e6])

    Θ, Φ = FSH.sph_points(N)
    lat = vec([π / 2 - θ for θ in Θ, φ in Φ])
    lon = vec([φ for θ in Θ, φ in Φ])
    n = length(lat)
    area = vec(permutedims(collect(FG.Grids.measure(cc))))
    ug = FG.Grids.UnstructuredGrid(geo, lon, lat, area, trues(n))
    modes = ((2, 0, 1.0), (5, -3, 0.6), (9, 4, -0.8))
    f = sum(a .* vec(ylm(l, m)) for (l, m, a) in modes)
    for k in kernels
        scales = [5e5, 1e6, 2e6]
        plans = CGEF.Filtering.plan_filter_sweep(ug, k, scales; method = _CGEF_SPECTRAL)
        Ĉ = CGEF.Filtering.filter_analyze!(CGEF.Filtering.analyze_buffer(plans[1], f), f, plans[1])
        for (p, ℓ) in zip(plans, scales)
            ref = sum(a .* Gl(k, l, ℓ) .* vec(ylm(l, m)) for (l, m, a) in modes)
            Test.@test CGEF.Filtering.filter_synthesize!(zeros(n), Ĉ, p) ≈ ref rtol = 1e-7
            Test.@test CGEF.Filtering.filter_apply!(zeros(n), f, p) ≈ ref rtol = 1e-7
        end
    end
    # `Deformable` divides the filtered `c·mask` by the filtered mask: a constant comes back at every active
    # point and a masked point reads zero.
    mask = lat .< deg2rad(40)
    pd = CGEF.Filtering.plan_filter(FG.Grids.UnstructuredGrid(geo, lon, lat, area, mask), CGEF.GaussianKernel(), 1e6;
                                    method = _CGEF_SPECTRAL, mask_strategy = CGEF.Filtering.Deformable())
    cd = CGEF.Filtering.filter_apply!(zeros(n), fill(1.7, n), pd)
    Test.@test cd[mask] ≈ fill(1.7, count(mask)) rtol = 1e-10
    Test.@test all(iszero, cd[.!mask])
end

# A grid no transform targets is filtered over its cells as a node set: centres, measures and mask in
# linear order, a regional sphere completed by its lattice continued past each bounded edge as inactive
# cells. Each equals that node set given explicitly.
Test.@testset "Spectral() over a grid's cells as a node set" begin
    explicit(grid; kw...) = FG.Grids.UnstructuredGrid(FG.Grids.grid_geometry(grid), FG.Grids.materialize(grid),
                                                      collect(vec(FG.Grids.measure(grid))), BitVector(vec(FG.Grids.mask(grid))); kw...)
    g = CGEF.GaussianKernel()
    cart = FG.Geometry.CartesianGeometry()
    for lib in _CGEF_NUFFTS
        sp = (method = _CGEF_SPECTRAL, transform = lib)
        x = cumsum([1.0 + 0.4 * sin(0.3i) for i in 1:30])
        y = collect(0.0:1.0:23.0)
        mask = trues(30, 24)
        mask[10:14, 8:12] .= false
        gs = FG.Grids.StructuredGrid(cart, x, y, mask)
        f = [sin(0.3xi) * cos(0.25yj) + 0.1xi for xi in x, yj in y]
        for s in (CGEF.Filtering.ZeroFill(), CGEF.Filtering.Deformable())
            Test.@test _cgef_relerr(vec(CGEF.Filtering.filter_field!(zeros(30, 24), f, gs, g, 6.0; sp..., mask_strategy = s)),
                                    CGEF.Filtering.filter_field!(zeros(720), vec(f), explicit(gs), g, 6.0; sp..., mask_strategy = s)) < 1e-13
        end
        _cgef_batch_and_sweep(gs, f, [3.0, 6.0]; transform = lib)

        # A uniform lattice held as vectors, against FFTW on the same points held as ranges.
        xr, yr = 0.0:1.0:29.0, 0.0:1.0:23.0
        fu = [sin(0.3a) * cos(0.25b) for a in xr, b in yr]
        Test.@test CGEF.Filtering.filter_field!(zeros(30, 24), fu, FG.Grids.StructuredGrid(cart, collect(xr), collect(yr), mask), g, 6.0; sp...) ≈
                   CGEF.Filtering.filter_field!(zeros(30, 24), fu, FG.Grids.StructuredGrid(cart, xr, yr, mask), g, 6.0; method = _CGEF_SPECTRAL) atol = 1e-7

        θ = deg2rad(15.0)
        cx = [i * cos(θ) - 0.3j * sin(θ) for i in 0:19, j in 0:15]
        cy = [i * sin(θ) + j * (1 + 0.3cos(θ)) for i in 0:19, j in 0:15]
        gc = FG.Grids.CurvilinearGrid(cart, cx, cy, trues(20, 16))
        fc = sin.(0.3 .* cx) .* cos.(0.2 .* cy)
        Test.@test _cgef_relerr(vec(CGEF.Filtering.filter_field!(zeros(20, 16), fc, gc, g, 4.0; sp...)),
                                CGEF.Filtering.filter_field!(zeros(320), vec(fc), explicit(gc), g, 4.0; sp...)) < 1e-13

        z = cumsum([1.0 + 0.3cos(0.5i) for i in 1:8])
        g3 = FG.Grids.StructuredGrid(cart, x[1:10], y[1:9], z)
        f3 = [sin(0.3a) * cos(0.25b) * sin(0.4c) for a in x[1:10], b in y[1:9], c in z]
        Test.@test _cgef_relerr(vec(CGEF.Filtering.filter_field!(zeros(10, 9, 8), f3, g3, g, 3.0; sp...)),
                                CGEF.Filtering.filter_field!(zeros(720), vec(f3), explicit(g3), g, 3.0; sp...)) < 1e-13
    end

    # On a sphere the node set's transform is NUFSHT's fit, whose iterates do not depend on node order.
    R = 6.371e6
    sph = FG.Geometry.SphericalGeometry(R)
    sp = (method = _CGEF_SPECTRAL,)
    field(gg) = (c = FG.Grids.materialize(gg); sin.(c[1]) .* cos.(c[2]) .+ 0.5 .* sin.(c[2]))
    hp = FG.Grids.HEALPixGrid(sph, 4)
    # FastSphericalHarmonics' nodes at a longitude count it does not transform, and a lat–lon grid whose end
    # faces lie on the poles: both cover the sphere, so neither is continued.
    wide = FG.Connectivity.structured_grid(FG.SphericalSampling.ClenshawCurtisSampling(), 12; geometry = sph, nlon = 24)
    poles = FG.Grids.StructuredGrid(sph, range(0.0; step = deg2rad(15.0), length = 24), range(deg2rad(-82.5); step = deg2rad(15.0), length = 12))
    for gg in (hp, wide, poles)
        fg = reshape(field(gg), size(FG.Grids.mask(gg)))
        Test.@test vec(CGEF.Filtering.filter_field!(zero(fg), fg, gg, g, 2e6; sp...)) ≈
                   CGEF.Filtering.filter_field!(zeros(length(fg)), vec(fg), explicit(gg), g, 2e6; sp...) rtol = 1e-10
    end
    # A regional block, completed by its lattice, is the global lattice with only the block active.
    lon = range(0.0; step = deg2rad(10.0), length = 11)
    lat = range(deg2rad(-35.0); step = deg2rad(10.0), length = 8)
    region = FG.Grids.StructuredGrid(sph, lon, lat; periodic = (false, false))
    block = falses(36, 18)
    block[1:11, 6:13] .= true
    whole = FG.Grids.StructuredGrid(sph, range(0.0; step = deg2rad(10.0), length = 36),
                                    range(deg2rad(-85.0); step = deg2rad(10.0), length = 18), block; periodic = (true, false))
    fr = [sin(2a) * cos(b) + 1.0 for a in lon, b in lat]
    fw = zeros(36, 18)
    fw[block] .= vec(fr)
    for s in (CGEF.Filtering.ZeroFill(), CGEF.Filtering.Deformable())
        got = CGEF.Filtering.filter_field!(zeros(11, 8), fr, region, g, 2e6; sp..., mask_strategy = s)
        ref = CGEF.Filtering.filter_field!(zeros(36 * 18), vec(fw), explicit(whole), g, 2e6; sp..., mask_strategy = s)
        Test.@test vec(got) ≈ reshape(ref, 36, 18)[block] rtol = 1e-10
    end
end

# A spectral plan runs its transform library on one thread under `SerialBackend()` and on every Julia
# thread under `ThreadedBackend()`; on KernelAbstractions' CPU device it holds the threaded plan's arrays,
# so it returns the threaded result to the rounding of the library's threaded accumulation.
Test.@testset "Spectral plans run at the execution backend's thread count and on the device" begin
    CB = CGEF.ComputationalBackends
    gpu = CB.GPUBackend(KA.CPU())
    g = CGEF.GaussianKernel()
    function device_matches_threaded(grid, f, ℓ; kw...)
        pt = CGEF.Filtering.plan_filter(grid, g, ℓ; method = _CGEF_SPECTRAL, execution = CB.ThreadedBackend(), batch = 2, kw...)
        pg = CGEF.Filtering.plan_filter(grid, g, ℓ; method = _CGEF_SPECTRAL, execution = gpu, batch = 2, kw...)
        F = cat(f, 2 .* f; dims = ndims(f) + 1)
        Test.@test _cgef_relerr(CGEF.Filtering.filter_apply!(similar(f), f, pg), CGEF.Filtering.filter_apply!(similar(f), f, pt)) < 1e-12
        Test.@test _cgef_relerr(CGEF.Filtering.filter_apply_batched!(similar(F), F, pg),
                                CGEF.Filtering.filter_apply_batched!(similar(F), F, pt)) < 1e-12
    end
    hole(sz) = (m = trues(sz); m[2:3] .= false; m)
    geo = FG.Geometry.CartesianGeometry()
    px = vec([Float64(i) for i in 0:7, _ in 0:5])
    py = vec([Float64(j) for _ in 0:7, j in 0:5])
    ug = FG.Grids.UnstructuredGrid(geo, px, py, ones(48), hole(48))
    for lib in _CGEF_NUFFTS
        for (b, n) in ((CB.SerialBackend(), 1), (CB.ThreadedBackend(), Threads.nthreads()))
            p = CGEF.Filtering.plan_filter(ug, g, 2.0; method = _CGEF_SPECTRAL, transform = lib, execution = b)
            Test.@test FTB.nthreads(p.grid_plan.plan) == n
        end
        for s in (CGEF.Filtering.ZeroFill(), CGEF.Filtering.Deformable())
            device_matches_threaded(ug, sin.(px) .+ cos.(py), 2.0; transform = lib, mask_strategy = s)
        end
    end
    N = 9
    sgeo = FG.Geometry.SphericalGeometry(1.0)
    Θ, Φ = FSH.sph_points(N)
    lat = vec([π / 2 - θ for θ in Θ, φ in Φ])
    lon = vec([φ for θ in Θ, φ in Φ])
    sg = FG.Grids.UnstructuredGrid(sgeo, lon, lat, ones(length(lat)), hole(length(lat)))
    for lib in _CGEF_NUFFTS
        device_matches_threaded(sg, cos.(lat) .* sin.(lon), 0.5; nufft = lib, mask_strategy = CGEF.Filtering.Deformable())
    end
    cc = FG.Connectivity.structured_grid(FG.SphericalSampling.ClenshawCurtisSampling(), N; geometry = sgeo)
    device_matches_threaded(cc, [sin(i) * cos(j) for i in 1:(2N - 1), j in 1:N], 0.5)
    grid = FG.Grids.StructuredGrid(geo, 0.0:1.0:7.0, 0.0:1.0:5.0, hole((8, 6)))
    for s in (CGEF.Filtering.ZeroFill(), CGEF.Filtering.Deformable())
        device_matches_threaded(grid, [sin(i) * cos(j) for i in 1:8, j in 1:6], 2.0; mask_strategy = s)
    end
end

# The filtering spectral density `dC/dk` is second order on any spacing of `k`, so it is exact for a
# quadratic `C(k)`, ends included; with two scales it is their difference quotient and below two, NaN.
Test.@testset "Spectral density" begin
    kk = 1.0 ./ exp10.(range(3.0, 5.0; length = 9))
    C = @. 0.3 - 2.0e4 * kk + 7.0e8 * kk^2
    Test.@test CGEF.Diagnostics.spectral_density(C, kk) ≈ (@. -2.0e4 + 1.4e9 * kk) rtol = 1e-10
    Test.@test CGEF.Diagnostics.spectral_density(C[1:2], kk[1:2]) ≈ fill((C[2] - C[1]) / (kk[2] - kk[1]), 2)
    Test.@test all(isnan, CGEF.Diagnostics.spectral_density(C[1:1], kk[1:1]))
end
