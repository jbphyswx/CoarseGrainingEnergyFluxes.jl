# `Spectral()` on a grid no transform targets directly runs over its cells as a node set: the cells'
# centres, measures and mask, in linear order, a regional sphere completed by its lattice continued past
# each bounded edge as inactive cells. Each case is checked against that node set given explicitly as an
# `UnstructuredGrid`, and a uniform lattice against FFTW on the same points.

const NODE_LIBRARIES = (FTB.FINUFFTBackend(), FTB.NonuniformFFTsBackend())

# The node set of `grid`, given explicitly: its cells' centres, measures and mask, in linear order.
explicit_nodes(grid; kw...) = FG.Grids.UnstructuredGrid(
    FG.Grids.grid_geometry(grid), FG.Grids.materialize(grid), collect(vec(FG.Grids.measure(grid))),
    BitVector(vec(FG.Grids.mask(grid))); kw...)

Test.@testset "Spectral filtering over a grid's cells: $(nameof(typeof(lib)))" for lib in NODE_LIBRARIES
    geom = FG.Geometry.CartesianGeometry()
    g = CGEF.GaussianKernel(); ℓ = 6.0
    sp = (method = CGEF.Filtering.Spectral(), spectral_backend = lib)
    strategies = (CGEF.Filtering.ZeroFill(), CGEF.Filtering.Deformable())

    # A stretched axis: the grid's axis types say it is not uniform, so FFTW does not take it.
    x = cumsum([1.0 + 0.4 * sin(0.3i) for i in 1:30]); y = collect(0.0:1.0:23.0)
    mask = trues(30, 24); mask[10:14, 8:12] .= false
    gs = FG.Grids.StructuredGrid(geom, x, y, mask)
    f = [sin(0.3xi) * cos(0.25yj) + 0.1xi for xi in x, yj in y]
    for st in strategies
        got = CGEF.Filtering.filter_field!(zeros(30, 24), f, gs, g, ℓ; sp..., mask_strategy = st)
        ref = CGEF.Filtering.filter_field!(zeros(30 * 24), vec(f), explicit_nodes(gs), g, ℓ; sp..., mask_strategy = st)
        Test.@test vec(got) ≈ ref rtol = 1e-13
    end
    Test.@test_throws ArgumentError CGEF.Filtering.filter_field!(
        zeros(30, 24), f, gs, g, ℓ; method = CGEF.Filtering.Spectral(),
        spectral_backend = CGEF.SpectralBackends.FFTSpectralBackend())

    # A uniform lattice held as plain vectors is filtered over its cells, and equals FFTW on the same
    # points held as ranges.
    xr = 0.0:1.0:29.0; yr = 0.0:1.0:23.0
    fu = [sin(0.3xi) * cos(0.25yj) for xi in xr, yj in yr]
    by_fft = CGEF.Filtering.filter_field!(zeros(30, 24), fu, FG.Grids.StructuredGrid(geom, xr, yr, mask), g, ℓ;
                                          method = CGEF.Filtering.Spectral())
    by_nodes = CGEF.Filtering.filter_field!(zeros(30, 24), fu, FG.Grids.StructuredGrid(geom, collect(xr), collect(yr), mask),
                                            g, ℓ; sp...)
    Test.@test by_nodes ≈ by_fft atol = 1e-7

    # A sheared curvilinear mesh.
    θ = deg2rad(15.0)
    cx = [ii * cos(θ) - 0.3jj * sin(θ) for ii in 0:19, jj in 0:15]
    cy = [ii * sin(θ) + jj * (1 + 0.3cos(θ)) for ii in 0:19, jj in 0:15]
    gc = FG.Grids.CurvilinearGrid(geom, cx, cy, trues(20, 16))
    fc = sin.(0.3 .* cx) .* cos.(0.2 .* cy)
    Test.@test vec(CGEF.Filtering.filter_field!(zeros(20, 16), fc, gc, g, 4.0; sp...)) ≈
               CGEF.Filtering.filter_field!(zeros(20 * 16), vec(fc), explicit_nodes(gc), g, 4.0; sp...) rtol = 1e-13

    # Three stretched directions.
    z = cumsum([1.0 + 0.3cos(0.5i) for i in 1:8])
    g3 = FG.Grids.StructuredGrid(geom, x[1:10], y[1:9], z, trues(10, 9, 8))
    f3 = [sin(0.3a) * cos(0.25b) * sin(0.4c) for a in x[1:10], b in y[1:9], c in z]
    Test.@test vec(CGEF.Filtering.filter_field!(zeros(10, 9, 8), f3, g3, g, 3.0; sp...)) ≈
               CGEF.Filtering.filter_field!(zeros(720), vec(f3), explicit_nodes(g3), g, 3.0; sp...) rtol = 1e-13

    # A trailing batch axis in one execution per direction, and a sweep's shared analysis.
    pb = CGEF.Filtering.plan_filter(gs, g, ℓ; sp..., batch = 3)
    F = cat(f, 2 .* f, f .^ 2; dims = 3)
    ob = CGEF.Filtering.filter_apply_batched!(zeros(30, 24, 3), F, pb)
    for b in 1:3
        Test.@test ob[:, :, b] ≈ CGEF.Filtering.filter_apply!(zeros(30, 24), F[:, :, b], pb) rtol = 1e-12
    end
    plans = CGEF.Filtering.plan_filter_sweep(gs, g, [3.0, 6.0]; sp...)
    F̂ = CGEF.Filtering.filter_analyze!(CGEF.Filtering.analyze_buffer(plans[1], f), f, plans[1])
    for p in plans
        Test.@test CGEF.Filtering.filter_synthesize!(zeros(30, 24), F̂, p) ≈
                   CGEF.Filtering.filter_apply!(zeros(30, 24), f, p) rtol = 1e-12
    end
end

# On the sphere the node set's transform is NUFSHT's least-squares fit, whose iterates do not depend on
# the order of the nodes, so a permuted node set agrees to round-off.
Test.@testset "Spherical spectral filtering over a grid's cells" begin
    R = 6.371e6
    sph = FG.Geometry.SphericalGeometry(R)
    g = CGEF.GaussianKernel(); ℓ = 2e6
    sp = (method = CGEF.Filtering.Spectral(),)

    hp = FG.Grids.HEALPixGrid(sph, 4)
    λ, φ = FG.Grids.materialize(hp)
    fh = sin.(λ) .* cos.(φ) .+ 0.5 .* sin.(φ)
    Test.@test CGEF.Filtering.filter_field!(zeros(length(fh)), fh, hp, g, ℓ; sp...) ≈
               CGEF.Filtering.filter_field!(zeros(length(fh)), fh, explicit_nodes(hp), g, ℓ; sp...) rtol = 1e-10

    # Two grids whose cells cover the whole sphere, so `Auto` filters them over their own cells alone:
    # FastSphericalHarmonics' nodes at a longitude count it does not transform, and a lat–lon grid whose
    # end faces lie on the poles.
    wide = FG.Connectivity.structured_grid(FG.SphericalSampling.ClenshawCurtisSampling(), 12; geometry = sph, nlon = 24)
    halfoff = FG.Grids.StructuredGrid(sph, range(0.0; step = deg2rad(15.0), length = 24),
                                      range(deg2rad(-82.5); step = deg2rad(15.0), length = 12))
    for gg in (wide, halfoff)
        λg, φg = FG.Grids.materialize(gg)
        fg = reshape(sin.(λg) .* cos.(φg) .+ 0.5 .* sin.(φg), 24, 12)
        Test.@test CGEF.Filtering._exterior_lattice(gg, π * R; cap = false) === nothing
        Test.@test vec(CGEF.Filtering.filter_field!(zeros(24, 12), fg, gg, g, ℓ; sp...)) ≈
                   CGEF.Filtering.filter_field!(zeros(24 * 12), vec(fg), explicit_nodes(gg), g, ℓ; sp...) rtol = 1e-10
    end
    # A quadrature sampling's cells total the sphere, so the real-space exterior is empty too.
    for s in (FG.SphericalSampling.ClenshawCurtisSampling(), FG.SphericalSampling.GaussLegendreSampling(),
              FG.SphericalSampling.DriscollHealySampling())
        Test.@test CGEF.Filtering._exterior_lattice(FG.Connectivity.structured_grid(s, 12; geometry = sph), π * R) === nothing
    end

    # A regional lat–lon block, completed by its lattice, is the global lattice with only the block active:
    # the ring closes in whole steps and the rows continue to faces on the poles.
    lon = range(0.0; step = deg2rad(10.0), length = 11); lat = range(deg2rad(-35.0); step = deg2rad(10.0), length = 8)
    region = FG.Grids.StructuredGrid(sph, lon, lat, trues(11, 8); periodic = (false, false))
    glon = range(0.0; step = deg2rad(10.0), length = 36); glat = range(deg2rad(-85.0); step = deg2rad(10.0), length = 18)
    block = falses(36, 18); block[1:11, 6:13] .= true
    whole = FG.Grids.StructuredGrid(sph, glon, glat, block; periodic = (true, false))
    fr = [sin(2a) * cos(b) + 1.0 for a in lon, b in lat]
    fw = zeros(36, 18); fw[block] .= vec(fr)
    for st in (CGEF.Filtering.ZeroFill(), CGEF.Filtering.Deformable())
        got = CGEF.Filtering.filter_field!(zeros(11, 8), fr, region, g, ℓ; sp..., mask_strategy = st)
        ref = CGEF.Filtering.filter_field!(zeros(36 * 18), vec(fw), explicit_nodes(whole), g, ℓ; sp..., mask_strategy = st)
        Test.@test vec(got) ≈ reshape(ref, 36, 18)[block] rtol = 1e-10
    end
end
