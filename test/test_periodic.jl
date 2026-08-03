
# Test periodic boundary handling for spherical grids
Test.@testset "Spherical Periodic Boundaries" begin
    geom = FG.Geometry.SphericalGeometry(6371000.0)
    # Create a grid that spans nearly 360 degrees in longitude
    lon_deg = collect(0.0:5.0:355.0)  # 72 points, 5-degree spacing
    lat_deg = collect(-45.0:5.0:45.0)  # 19 points
    lon_rad = deg2rad.(lon_deg)
    lat_rad = deg2rad.(lat_deg)
    mask = trues(length(lon_deg), length(lat_deg))
    grid = FG.Grids.StructuredGrid(geom, lon_rad, lat_rad, mask)

    # Create a field with a sharp gradient at the periodic boundary
    field = zeros(length(lon_deg), length(lat_deg))
    for j in 1:length(lat_deg)
        for i in 1:length(lon_deg)
            # Sharp transition near 0/360 boundary
            if lon_deg[i] > 350.0 || lon_deg[i] < 10.0
                field[i, j] = 1.0
            else
                field[i, j] = 0.0
            end
        end
    end

    # Filter at 20° scale (larger than the 15° band, so wrapping matters)
    scale = deg2rad(20.0) * 6371000.0
    out = zeros(length(lon_deg), length(lat_deg))
    CGEF.Filtering.filter_field!(out, field, grid, CGEF.TopHatKernel(), scale)

    # Brute-force reference: an ALL-PAIRS TopHat average built directly from the raw
    # great-circle-distance primitive (NOT the package's footprint-building/seam-wrap index
    # logic under test here). This is naturally, automatically periodic in longitude — the
    # Haversine `sin²(Δλ/2)` term is exactly invariant to whether Δλ is measured "the short way"
    # or "the long way around" the 360° circle, so no explicit modular wrap is needed to get the
    # true periodic answer. This replaces a hand-derived comment that didn't actually match the
    # asserted bounds with a genuinely computed expected value.
    rad = CGEF.Kernels.kernel_radius(CGEF.TopHatKernel(), scale)  # TopHat radius is ℓ/2, not ℓ
    # AREA-weighted (not plain-count) average: grid cells at different latitudes have different
    # physical area on a sphere (∝ cosφ), and a spatial average — which is what a TopHat filter
    # computes (weight = kernel_weight·area, kernel_weight≡1 within radius) — must weight by that
    # area, not just count included points. An unweighted average is a different, wrong quantity
    # whenever the neighbourhood spans more than one latitude row (it always does here).
    function brute_force_avg(i0, j0)
        target = SA.SVector{2,Float64}(lon_rad[i0], lat_rad[j0])
        acc = 0.0; wsum = 0.0
        for j in eachindex(lat_deg), i in eachindex(lon_deg)
            d = FG.Geometry.distance(geom, target, SA.SVector{2,Float64}(lon_rad[i], lat_rad[j]))
            if d <= rad
                a = FG.Grids.area(grid, i, j)
                acc += field[i, j] * a
                wsum += a
            end
        end
        return acc / wsum
    end

    j10 = 10  # a mid-latitude row, away from the poles
    Test.@test out[1, j10] ≈ brute_force_avg(1, j10) rtol=1e-10
    Test.@test out[end, j10] ≈ brute_force_avg(length(lon_deg), j10) rtol=1e-10

    # The seam genuinely wraps: on a NON-periodic version of the same grid, 355° loses the
    # wrapped 0-10° "hot" band entirely, so its average must be strictly LESS than the correctly-
    # wrapped periodic result — removing area-weighted contributions from the hot region can only
    # decrease (never increase) a weighted average. This is the exact, derivable bound (a specific
    # numeric factor like "less than half" isn't independently justified without computing it, and
    # a hand-picked one is exactly the kind of loosened-to-pass tolerance Tier 3.15 replaced
    # elsewhere in this file).
    nonperiodic_grid = FG.Grids.StructuredGrid(geom, lon_rad, lat_rad, mask; periodic = false)
    out_np = zeros(size(field))
    CGEF.Filtering.filter_field!(out_np, field, nonperiodic_grid, CGEF.TopHatKernel(), scale)
    Test.@test out_np[end, j10] < out[end, j10]

    # A point at 30° (well outside the ±20° band around the seam) sees no wrapping benefit —
    # its own brute-force reference already reflects that, at the same tight tolerance.
    i30 = findfirst(==(30.0), lon_deg)
    Test.@test out[i30, j10] ≈ brute_force_avg(i30, j10) rtol=1e-10
    Test.@test out[i30, j10] < out[1, j10]
end


# Regional domains must NOT wrap in longitude (the previous code wrapped every spherical grid,
# double-counting near boundaries when the footprint exceeded a regional domain).
Test.@testset "Regional vs periodic longitude" begin
    geom = FG.Geometry.SphericalGeometry(6371000.0)
    lat = deg2rad.(collect(-4.0:2.0:4.0))

    # Regional lon span -> auto-detected NON-periodic
    lon_reg = deg2rad.(collect(0.0:2.0:20.0))   # 11 points, 20° span
    mask_reg = trues(length(lon_reg), length(lat))
    grid_reg = FG.Grids.StructuredGrid(geom, lon_reg, lat, mask_reg)
    Test.@test FG.Grids.isperiodic(grid_reg, 1) == false

    # Full-circle lon span -> auto-detected periodic
    lon_glob = deg2rad.(collect(0.0:5.0:355.0))
    mask_glob = trues(length(lon_glob), length(lat))
    grid_glob = FG.Grids.StructuredGrid(geom, lon_glob, lat, mask_glob)
    Test.@test FG.Grids.isperiodic(grid_glob, 1) == true

    # Explicit override in both directions
    Test.@test FG.Grids.isperiodic(FG.Grids.StructuredGrid(geom, lon_reg, lat, mask_reg; periodic = true), 1) == true
    Test.@test FG.Grids.isperiodic(FG.Grids.StructuredGrid(geom, lon_glob, lat, mask_glob; periodic = false), 1) == false

    # The periodicity flag must actually change filtering: with a footprint wider than the
    # regional domain, wrapping double-counts and yields a different (incorrect) field.
    grid_forced = FG.Grids.StructuredGrid(geom, lon_reg, lat, mask_reg; periodic = true)
    field = Float64[i for i in 1:length(lon_reg), _ in 1:length(lat)]  # ramp in lon
    scale = deg2rad(30.0) * 6371000.0   # footprint wider than the 20° domain
    out_nowrap = zeros(size(field))
    out_wrap = zeros(size(field))
    CGEF.Filtering.filter_field!(out_nowrap, field, grid_reg, CGEF.TopHatKernel(), scale)
    CGEF.Filtering.filter_field!(out_wrap, field, grid_forced, CGEF.TopHatKernel(), scale)
    Test.@test !any(isnan, out_nowrap)
    Test.@test !(out_nowrap ≈ out_wrap)
end


# Regression test: a periodic CARTESIAN axis was computing its wrapped boundary cell width using
# the SPHERICAL periodicity constant (2π radians) unconditionally, regardless of geometry — for a
# Cartesian domain measured in meters, this produced a wildly wrong (even negative) boundary cell
# width, and hence a nonsensical filter weight there. This is a physically impossible result for a
# correctly-normalized low-pass filter (output can amplify beyond the input's range), not just an
# accuracy issue — caught via a direct comparison against the analytic filtered value of a single
# Fourier mode (an exact eigenfunction relation, Ĝ(k)·field, independent of discretization).
Test.@testset "Periodic Cartesian grid: no boundary weight corruption" begin
    dx = 62.5
    Nx = 320
    geom = FG.Geometry.CartesianGeometry()
    xs = 0.0:dx:(dx*(Nx-1))   # a Range, not `collect`ed — keeps this on the cheap fast-path stencil;
                              # the scattered/general path is exercised (at a deliberately small N) by
                              # the next testset instead
    Lx = dx * Nx
    grid = FG.Grids.StructuredGrid(geom, xs, xs, trues(Nx, Nx); periodic = true)
    Test.@test FG.Grids.isperiodic(grid, 1) == true

    kx0, ky0 = 2, 3
    field = [sin(2π * kx0 * x / Lx) * cos(2π * ky0 * y / Lx) for x in xs, y in xs]
    scale = 1500.0
    kx = 2π * kx0 / Lx; ky = 2π * ky0 / Lx
    Ghat = CGEF.Kernels.spectral_transfer(CGEF.GaussianKernel(), sqrt(kx^2 + ky^2), scale)
    analytic = Ghat .* field

    out = zeros(Nx, Nx)
    CGEF.Filtering.filter_field!(out, field, grid, CGEF.GaussianKernel(), scale)

    # A low-pass filter is a normalized weighted average: it can never amplify beyond the input's
    # range. This is the invariant the bug violated (output reached ~4.8x the input's peak).
    Test.@test maximum(abs, out) <= maximum(abs, field) + 1e-9
    # Constant field ⇒ Ĝ(0)=1 ⇒ preserved exactly, independent of the boundary-wrap bug.
    out_const = zeros(Nx, Nx)
    CGEF.Filtering.filter_field!(out_const, fill(3.7, Nx, Nx), grid, CGEF.GaussianKernel(), scale)
    Test.@test all(x -> isapprox(x, 3.7; atol = 1e-6), out_const)
    # Away from the (genuinely non-periodic) y-boundary, the periodic-in-x wrap must reproduce
    # the analytic single-mode eigenfunction relation closely.
    rad = CGEF.Kernels.kernel_radius(CGEF.GaussianKernel(), scale)
    interior = [(rad < xs[j] < xs[end] - rad) for _ in 1:Nx, j in 1:Nx]
    reldiff = abs.(out[interior] .- analytic[interior]) ./ maximum(abs, analytic)
    Test.@test maximum(reldiff) < 0.3
    Test.@test sum(reldiff) / length(reldiff) < 0.05
end


Test.@testset "Periodic Cartesian grid: general/scattered footprint path matches the fast Range path exactly" begin
    # A genuinely uniform axis passed as a plain Vector (not a Range) forces the general/
    # scattered footprint builder (StructuredGrid's type-is-the-dispatch convention — no runtime
    # uniformity check). That path must honor periodicity exactly like the fast Range-axis path:
    # a wrapped neighbor's distance must be computed from its coordinate SHIFTED by one period,
    # not its raw stored coordinate (which sits a full domain-width away) — otherwise every
    # periodic wrap is silently rejected by the `d <= rad` gate and boundary cells behave as if
    # non-periodic, with no error. The existing "no boundary weight corruption" test above does
    # not catch this on its own — its eigenmode is too smooth and its tolerance too loose to
    # distinguish a correctly-wrapped boundary from a silently truncated one — so this test
    # cross-checks the general/scattered path directly against the independently-trusted fast
    # path instead.
    dx = 1_000.0
    # A `Vector` axis proves no spacing, so its wrap length is not inferable and must be given;
    # each of these is a collected uniform range, so the domain it wraps over is N·dx.
    N = 40
    geom = FG.Geometry.CartesianGeometry()
    xsR = 0.0:dx:(dx * (N - 1))       # Range -> fast uniform path
    xsV = collect(xsR)                # identical values as a plain Vector -> scattered/general path
    field = Float64[i + 2j for i in 1:N, j in 1:N]   # boundary-discriminating, deliberately asymmetric

    for scale in (2_500.0, 6_000.0)   # spans 1 and >2 wrapped-neighbor bands
        gridR = FG.Grids.StructuredGrid(geom, xsR, xsR, trues(N, N); periodic = (true, true))
        gridV = FG.Grids.StructuredGrid(geom, xsV, xsV, trues(N, N); periodic = (true, true),
                                    period = (N * dx, N * dx))
        # Top-hat now takes the exact prefix-sum path on BOTH axis kinds, so comparing them to
        # each other no longer cross-validates two DIFFERENT engines. Keep that cross-validation
        # by comparing the prefix-sum result against the general scattered engine built directly
        # on the same periodic grid — the independent implementation this test exists to check.
        Test.@test CGEF.Filtering.build_footprint(gridR, CGEF.TopHatKernel(), scale) isa CGEF.Filtering.PrefixSumTopHatPlan
        Test.@test CGEF.Filtering.build_footprint(gridV, CGEF.TopHatKernel(), scale) isa CGEF.Filtering.PrefixSumTopHatPlan

        outR = zeros(N, N); outV = zeros(N, N)
        CGEF.Filtering.filter_field!(outR, field, gridR, CGEF.TopHatKernel(), scale)
        CGEF.Filtering.filter_field!(outV, field, gridV, CGEF.TopHatKernel(), scale)
        Test.@test isapprox(outR, outV; atol = 1e-9)

        fp_sca = CGEF.Filtering._build_footprint_scattered(
            gridV, CGEF.TopHatKernel(), scale;
            mask_strategy = CGEF.Filtering.Deformable(), cache_strategy = CGEF.Filtering.AlwaysCache(),
        )
        outSca = zeros(N, N)
        CGEF.Filtering.apply_footprint!(outSca, field, gridV, fp_sca, CGEF.Filtering.Deformable(), true, true)
        Test.@test isapprox(outV, outSca; rtol = 1e-11)

        # Sanity: the match above isn't trivially true regardless of periodicity — a genuinely
        # non-periodic grid must disagree with the periodic one at the boundary.
        gridVn = FG.Grids.StructuredGrid(geom, xsV, xsV, trues(N, N); periodic = (false, false))
        outVn = zeros(N, N)
        CGEF.Filtering.filter_field!(outVn, field, gridVn, CGEF.TopHatKernel(), scale)
        Test.@test maximum(abs, outV[1, :] .- outVn[1, :]) > 1e-6
    end

    # Same check for the N-D scattered path (`_build_footprint_nd_scattered`): true-3D and 1D.
    geom3 = FG.Geometry.CartesianGeometry()
    N3 = 12
    xs3R = 0.0:dx:(dx * (N3 - 1)); xs3V = collect(xs3R)
    field3 = Float64[i + 2j + 3k for i in 1:N3, j in 1:N3, k in 1:N3]
    grid3R = FG.Grids.StructuredGrid(geom3, xs3R, xs3R, xs3R, trues(N3, N3, N3); periodic = (true, true, true))
    grid3V = FG.Grids.StructuredGrid(geom3, xs3V, xs3V, xs3V, trues(N3, N3, N3); periodic = (true, true, true),
                                 period = (N3 * dx, N3 * dx, N3 * dx))
    out3R = zeros(N3, N3, N3); out3V = zeros(N3, N3, N3)
    CGEF.Filtering.filter_field!(out3R, field3, grid3R, CGEF.TopHatKernel(), 1_200.0)
    CGEF.Filtering.filter_field!(out3V, field3, grid3V, CGEF.TopHatKernel(), 1_200.0)
    Test.@test isapprox(out3R, out3V; atol = 1e-9)

    N1 = 30
    xs1R = 0.0:dx:(dx * (N1 - 1)); xs1V = collect(xs1R)
    field1 = Float64.(1:N1)
    grid1R = FG.Grids.StructuredGrid(geom, xs1R, trues(N1); periodic = true)
    grid1V = FG.Grids.StructuredGrid(geom, xs1V, trues(N1); periodic = true, period = N1 * dx)
    out1R = zero(field1); out1V = zero(field1)
    CGEF.Filtering.filter_field!(out1R, field1, grid1R, CGEF.TopHatKernel(), 1_200.0)
    CGEF.Filtering.filter_field!(out1V, field1, grid1V, CGEF.TopHatKernel(), 1_200.0)
    Test.@test isapprox(out1R, out1V; atol = 1e-9)
end


Test.@testset "Doubly-periodic Cartesian grid: axis-2 (y) wrap matches axis-1 (x) exactly" begin
    # A doubly-periodic Cartesian box is the standard homogeneous-turbulence setup, and both axes
    # must wrap. The single-mode eigenfunction relation `filter(mode) = Ĝ(|k|)·mode` holds at EVERY
    # point only when the domain is genuinely periodic in both directions, so this checks rows and
    # columns adjacent to j=1 and j=Nx as well — which an x-only periodicity test cannot.
    dx = 62.5
    Nx = 32   # small on purpose: the scattered/general footprint path's per-point search window is
              # (2*di_lim+1)^2 regardless of Nx, so its total memory is O(Nx^2 * window) — a much
              # bigger Nx here (e.g. the 320 once used by the "no boundary weight corruption" test
              # above, before that was fixed to use a Range) would reserve tens of GB for no benefit,
              # since only exercising both code paths at all is what this test needs, not resolution.
    geom = FG.Geometry.CartesianGeometry()
    xsR = 0.0:dx:(dx*(Nx-1))   # Range -> uniform separable path
    xsV = collect(xsR)         # identical values as a plain Vector -> forces the scattered/general path
    Lx = dx * Nx

    kx0, ky0 = 2, 3
    field = [sin(2π * kx0 * x / Lx) * cos(2π * ky0 * y / Lx) for x in xsR, y in xsR]
    scale = 750.0
    kx = 2π * kx0 / Lx; ky = 2π * ky0 / Lx
    Ghat = CGEF.Kernels.spectral_transfer(CGEF.GaussianKernel(), sqrt(kx^2 + ky^2), scale)
    analytic = Ghat .* field

    # A Range axis proves its spacing in its type and infers its own wrap length; a Vector proves
    # nothing, so its period must be given. Both describe the same domain, to the same values.
    gridR = FG.Grids.StructuredGrid(geom, xsR, xsR, trues(Nx, Nx); periodic = (true, true))
    gridV = FG.Grids.StructuredGrid(geom, xsV, xsV, trues(Nx, Nx); periodic = (true, true), period = (Lx, Lx))
    Test.@test FG.Grids.isuniform(gridR) && !FG.Grids.isuniform(gridV)

    # The axis type is what selects the engine, so this pair is the only place the two are compared
    # on identical coordinates.
    Test.@test CGEF.Filtering.plan_filter(gridR, CGEF.GaussianKernel(), scale).footprint isa
               CGEF.Filtering.SeparableGaussianFootprint

    outs = map((gridR, gridV)) do grid
        Test.@test FG.Grids.isperiodic(grid, 1) && FG.Grids.isperiodic(grid, 2)
        out = zeros(Nx, Nx)
        CGEF.Filtering.filter_field!(out, field, grid, CGEF.GaussianKernel(), scale)
        # Every point, including rows/cols touching both boundaries — no "interior" carve-out.
        # `rad = 1469 > Lx/2 = 1000`, so this only holds if a cell contributes once per periodic
        # image rather than once in total: it is the periodic convolution, and it is exact.
        Test.@test maximum(abs.(out .- analytic)) / maximum(abs, analytic) < 1e-8
        out
    end
    Test.@test maximum(abs.(outs[1] .- outs[2])) / maximum(abs, analytic) < 1e-8

    # Cross-backend agreement on the same doubly-periodic grid (each parallel backend threads the
    # same `periodic_lon`/`periodic_lat` pair through independently; a backend that dropped the
    # axis-2 flag would silently disagree with the serial reference at the y-boundary).
    grid = FG.Grids.StructuredGrid(geom, xsR, xsR, trues(Nx, Nx); periodic = (true, true))
    out_serial = zeros(Nx, Nx)
    CGEF.Filtering.filter_field!(out_serial, field, grid, CGEF.GaussianKernel(), scale; backend = CGEF.ComputationalBackends.SerialBackend())
    out_threaded = zeros(Nx, Nx)
    CGEF.Filtering.filter_field!(out_threaded, field, grid, CGEF.GaussianKernel(), scale; backend = CGEF.ComputationalBackends.ThreadedBackend())
    Test.@test isapprox(out_serial, out_threaded; atol = 1e-9)
    out_gpu = zeros(Nx, Nx)
    CGEF.Filtering.filter_field!(out_gpu, field, grid, CGEF.GaussianKernel(), scale; backend = CGEF.ComputationalBackends.GPUBackend(KA.CPU()))
    Test.@test isapprox(out_serial, out_gpu; atol = 1e-9)
    out_distributed = zeros(Nx, Nx)
    CGEF.Filtering.filter_field!(out_distributed, field, grid, CGEF.GaussianKernel(), scale; backend = CGEF.ComputationalBackends.DistributedBackend())
    Test.@test isapprox(out_serial, out_distributed; atol = 1e-9)
    out_mpi = zeros(Nx, Nx)
    CGEF.Filtering.filter_field!(out_mpi, field, grid, CGEF.GaussianKernel(), scale; backend = CGEF.ComputationalBackends.MPIBackend())
    Test.@test isapprox(out_serial, out_mpi; atol = 1e-9)
end


# -----------------------------------------------------------------------
# The two periodic conventions, each checked against an all-pairs reference written here rather than
# against another of this package's engines. Both cases need a filter window WIDER than the wrapped
# direction, which is where the conventions stop coinciding.
# -----------------------------------------------------------------------
Test.@testset "Periodic conventions past one turn: angular identifies, translational tiles" begin
    # Angular (spherical longitude): λ and λ+2π are the same point, so each of the Nlon cells counts at
    # most once however far the window reaches. Counting the repeats put the polar rows 7.6% out.
    function ref_spherical(grid, f, ker, scale)
        geo = FG.Grids.grid_geometry(grid)
        rad = CGEF.Kernels.kernel_radius(ker, scale)
        Nx, Ny = FG.Grids.size_tuple(grid)
        out = zeros(Nx, Ny)
        for j in 1:Ny, i in 1:Nx
            p = FG.Grids.coords(grid, i, j)
            ws = 0.0; wn = 0.0
            for jn in 1:Ny, in_ in 1:Nx        # every cell once; great-circle distance is 2π-periodic
                d = FG.Geometry.distance(geo, p, FG.Grids.coords(grid, in_, jn))
                d <= rad || continue
                w = CGEF.Kernels.kernel_weight(ker, d, scale) * FG.Grids.area(grid, in_, jn)
                wn += w; ws += w * f[in_, jn]
            end
            out[i, j] = wn > 1e-15 ? ws / wn : 0.0
        end
        return out
    end

    sgeom = FG.Geometry.SphericalGeometry(6.371e6)
    plon = deg2rad.(collect(0.0:6.0:354.0))
    plat = deg2rad.(collect(-88.0:4.0:88.0))
    pgrid = FG.Grids.StructuredGrid(sgeom, plon, plat, trues(length(plon), length(plat)); periodic = (true, false))
    pu = [sin(3λ) * cos(2φ) + 0.3cos(5λ) for λ in plon, φ in plat]
    pker = CGEF.GaussianKernel(); pscale = 8.0e5
    pfp = CGEF.Filtering.build_footprint(pgrid, pker, pscale; cache_strategy = CGEF.Filtering.NeverCache())
    Test.@test 2 * pfp.di_lim + 1 > length(plon)   # the offset span really does exceed the ring
    o_ps = zeros(size(pu))
    CGEF.Filtering.filter_field!(o_ps, pu, pgrid, pker, pscale;
        backend = CGEF.ComputationalBackends.SerialBackend())
    Test.@test maximum(abs, o_ps .- ref_spherical(pgrid, pu, pker, pscale)) < 1e-13

    # Translational (periodic Cartesian): x and x+L are distinct places, so a cell contributes once per
    # image inside the radius, each at its own displacement.
    function ref_tiled(grid, f, ker, scale, L)
        geo = FG.Grids.grid_geometry(grid)
        rad = CGEF.Kernels.kernel_radius(ker, scale)
        Nx, Ny = FG.Grids.size_tuple(grid)
        kmax = ceil(Int, rad / L) + 1
        out = zeros(Nx, Ny)
        for j in 1:Ny, i in 1:Nx
            p = FG.Grids.coords(SA.SVector, grid, i, j)
            ws = 0.0; wn = 0.0
            for jn in 1:Ny, in_ in 1:Nx, kj in (-kmax):kmax, ki in (-kmax):kmax
                q = FG.Grids.coords(SA.SVector, grid, in_, jn) + SA.SVector(ki * L, kj * L)
                d = FG.Geometry.distance(geo, p, q)
                d <= rad || continue
                w = CGEF.Kernels.kernel_weight(ker, d, scale) * FG.Grids.area(grid, in_, jn)
                wn += w; ws += w * f[in_, jn]
            end
            out[i, j] = wn > 1e-15 ? ws / wn : 0.0
        end
        return out
    end

    cgeom = FG.Geometry.CartesianGeometry()
    tN = 16; tdx = 62.5
    tx = collect(0.0:tdx:(tN - 1) * tdx) .+ [0.15tdx * sin(2.9i) for i in 1:tN]
    tL = tN * tdx
    tgrid = FG.Grids.StructuredGrid(cgeom, tx, copy(tx), trues(tN, tN); periodic = (true, true), period = (tL, tL))
    tu = [sin(2π * 3 * xi / tL) * cos(2π * 2 * yj / tL) for xi in tx, yj in tx]
    tker = CGEF.SharpSpectralKernel(); tscale = 500.0
    tfp = CGEF.Filtering.build_footprint(tgrid, tker, tscale; cache_strategy = CGEF.Filtering.NeverCache())
    Test.@test tfp.rad > tL                        # several images deep, not just one wrap
    o_ts = zeros(tN, tN)
    CGEF.Filtering.filter_field!(o_ts, tu, tgrid, tker, tscale;
        backend = CGEF.ComputationalBackends.SerialBackend())
    # A sinc kernel over this many images sums with heavy cancellation, so the reference's summation
    # order and the engine's cannot agree to the last bit.
    Test.@test maximum(abs, o_ts .- ref_tiled(tgrid, tu, tker, tscale, tL)) < 1e-6
end
