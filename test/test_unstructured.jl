
Test.@testset "UnstructuredGrid (k-d tree / Voronoi / WLSQ / pipeline)" begin
    # --- k-d tree neighbor query correctness on a small, hand-verifiable regular lattice: for an
    # interior point of a uniform grid, the k=4 nearest neighbors are exactly its N/S/E/W neighbors. ---
    cart = FG.Geometry.CartesianGeometry()
    gx = collect(0.0:1.0:4.0); gy = collect(0.0:1.0:4.0)
    plon = vec([x for x in gx, y in gy]); plat = vec([y for x in gx, y in gy])
    ugrid_lattice = FG.Grids.UnstructuredGrid(cart, plon, plat, trues(length(plon)); k = 4)
    center_idx = findfirst(i -> plon[i] == 2.0 && plat[i] == 2.0, eachindex(plon))
    nbr_coords = Set((plon[j], plat[j]) for j in FG.Grids.neighbors(ugrid_lattice, center_idx))
    Test.@test nbr_coords == Set([(1.0,2.0), (3.0,2.0), (2.0,1.0), (2.0,3.0)])

    # --- Voronoi-area sum invariant, Cartesian: a regular lattice's Voronoi cells reduce to the
    # exact grid-cell area (1.0 here), so interior nodes' areas must be ≈1 and the total must equal
    # the lattice's bounding area (up to boundary-clipping conventions at the edge). ---
    interior = [i for i in eachindex(plon) if 1.0 <= plon[i] <= 3.0 && 1.0 <= plat[i] <= 3.0]
    Test.@test all(i -> isapprox(FG.Grids.measure(ugrid_lattice)[i], 1.0; rtol = 1e-6), interior)

    # --- Voronoi-area sum invariant, spherical: a quasi-uniform point set covering the WHOLE
    # sphere must have Voronoi areas summing to EXACTLY 4πR² (a full closed tessellation, no
    # boundary to clip — an exact invariant, not merely quasi-uniform-only). ---
    sph_geo = FG.Geometry.SphericalGeometry(6371000.0)
    Nsph = 400
    # Fibonacci sphere point set: deterministic, quasi-uniform coverage of the whole sphere.
    golden = (1 + sqrt(5)) / 2
    sidx = 0:(Nsph-1)
    sphi = acos.(1 .- 2 .* (sidx .+ 0.5) ./ Nsph) .- π/2   # latitude in [-π/2, π/2]
    stheta = 2π .* (sidx ./ golden .% 1)                   # longitude in [0, 2π)
    sgrid_full = FG.Grids.UnstructuredGrid(sph_geo, stheta, sphi, trues(Nsph); k = 8)
    Test.@test sum(FG.Grids.measure(sgrid_full)) ≈ 4π * sph_geo.R^2 rtol = 1e-10

    # --- WLSQ gradient is exact for a LINEAR field (algebraically guaranteed on any stencil,
    # same as the CurvilinearGrid case) on a genuinely irregular (random) point scatter. ---
    Random_N = 150
    rlon = rand(Random_N) .* 10000.0
    rlat = rand(Random_N) .* 10000.0
    rgrid = FG.Grids.UnstructuredGrid(cart, rlon, rlat, trues(Random_N); k = 8)
    flin = 2.0 .* rlon .+ 3.0 .* rlat
    gx_lin = zeros(Random_N); gy_lin = zeros(Random_N)
    FG.Discretization.gradient!(gx_lin, gy_lin, flin, FG.Connectivity.gradient_plan(rgrid))
    # Only nodes with a full (non-rank-deficient) stencil are guaranteed exact; boundary/corner
    # nodes with a degenerate one-sided stencil are excluded — an honest test, not a silent one.
    interior_r = [i for i in 1:Random_N if length(FG.Grids.neighbors(rgrid, i)) >= 4]
    Test.@test maximum(abs.(gx_lin[interior_r] .- 2.0)) < 1e-8
    Test.@test maximum(abs.(gy_lin[interior_r] .- 3.0)) < 1e-8

    # --- Full compute_Π!/coarse_grain pipeline, cross-checked against the equivalent
    # StructuredGrid result on the SAME underlying lattice (points placed exactly on grid nodes,
    # k chosen so neighbors are exactly the 4 structured neighbors). ---
    Nx2, Ny2 = 8, 8
    xs2 = collect(0.0:1000.0:(1000.0*(Nx2-1)))
    ys2 = collect(0.0:1000.0:(1000.0*(Ny2-1)))
    sgrid2 = FG.Grids.StructuredGrid(cart, xs2, ys2, trues(Nx2, Ny2))
    ulon = vec([x for x in xs2, y in ys2]); ulat = vec([y for x in xs2, y in ys2])
    uu2 = vec([sin(x/700) * cos(y/900) for x in xs2, y in ys2])
    vv2 = vec([cos(x/500) * sin(y/1100) for x in xs2, y in ys2])
    ugrid2 = FG.Grids.UnstructuredGrid(cart, ulon, ulat, trues(length(ulon)); k = 4)

    Πu = zeros(length(ulon))
    CGEF.Diagnostics.compute_Π!(Πu, uu2, vv2, nothing, ugrid2, CGEF.GaussianKernel(), 3000.0)
    Test.@test all(isfinite, Πu)
    # Catches a regression to the fixed FINUFFT mode-count-from-geometry.dx bug (which made this
    # 64-point call cost ~4 GiB instead of KB) even if some future change kept the numerics finite.
    # Gated on the mode grid the plan builds and on allocations — both exact, neither load-dependent.
    plan_u = CGEF.Filtering.plan_filter(ugrid2, CGEF.GaussianKernel(), 3000.0; method = CGEF.Filtering.Spectral())
    Test.@test prod(size(plan_u.transfer)) <= 4 * length(ulon)
    b_pi = @allocated CGEF.Diagnostics.compute_Π!(Πu, uu2, vv2, nothing, ugrid2, CGEF.GaussianKernel(), 3000.0)
    Test.@test b_pi < 10_000_000

    res_u = CGEF.coarse_grain(uu2, vv2, ugrid2; scales = [2000.0, 3000.0], kernel = CGEF.GaussianKernel())
    Test.@test size(res_u.Π) == (length(ulon), 2)
    Test.@test !any(isnan, res_u.cumulative_energy)
    Test.@test !any(isnan, res_u.filtering_spectrum)

    # --- Solid-body rotation on a spherical UnstructuredGrid gives Π ≈ 0 (same physical invariant
    # as the StructuredGrid/CurvilinearGrid cases). ---
    Usb = 5.0
    usb = Usb .* cos.(sphi)
    vsb = zeros(Nsph)
    Πsb_u = zeros(Nsph)
    CGEF.Diagnostics.compute_Π!(Πsb_u, usb, vsb, nothing, sgrid_full, CGEF.GaussianKernel(), 5e5)
    Test.@test all(isfinite, Πsb_u)
    Test.@test maximum(abs, Πsb_u) < 1e-2 * Usb^2
end


# A node set has no axes to bound a search window with, so its real-space engine finds each node's
# neighbourhood once at plan time through the grid's own metric-ball query.
Test.@testset "Real-space filtering on an UnstructuredGrid" begin
    geom = FG.Geometry.CartesianGeometry()
    n = 16
    dx = 1000.0
    xr = range(0.0, dx * (n - 1); length = n)
    X = [x for x in xr, y in xr]
    Y = [y for x in xr, y in xr]
    npt = n * n
    K = CGEF.GaussianKernel()
    ℓ = 4000.0
    # The same points and the same cell areas as a structured grid, so the two must agree.
    ug = FG.Grids.UnstructuredGrid(geom, vec(X), vec(Y), fill(dx^2, npt), trues(npt))
    sg = FG.Grids.StructuredGrid(geom, xr, xr, trues(n, n))

    p = CGEF.Filtering.plan_filter(ug, K, ℓ; method = CGEF.Filtering.RealSpace())
    Test.@test p.footprint isa CGEF.Filtering.NodeFilterPlan
    # Each node's block includes the node itself, which the metric-ball query excludes.
    Test.@test all(t -> p.footprint.nbrs[p.footprint.ptr[t]] == t, 1:npt)

    out = zeros(npt)
    CGEF.Filtering.filter_field!(out, fill(3.7, npt), ug, K, ℓ; method = CGEF.Filtering.RealSpace())
    Test.@test all(x -> isapprox(x, 3.7; atol = 1e-12), out)

    f = [sin(x / 5000) * cos(y / 6000) for x in xr, y in xr]
    node = zeros(npt)
    CGEF.Filtering.filter_field!(node, vec(f), ug, K, ℓ; method = CGEF.Filtering.RealSpace())
    structured = zeros(n, n)
    CGEF.Filtering.filter_field!(
        structured, f, sg, K, ℓ;
        filter_plan = CGEF.Filtering.PhysicalFilterPlan(
            CGEF.Filtering._build_footprint_scattered(sg, K, ℓ), sg,
            CGEF.Filtering.Deformable(), K, ℓ, CGEF.ComputationalBackends.SerialBackend(),
        ),
    )
    Test.@test maximum(abs.(node .- vec(structured))) < 1e-12

    # Spectral remains the default for a node set, so the real-space plan is opt-in.
    Test.@test !(CGEF.Filtering.plan_filter(ug, K, ℓ) isa CGEF.Filtering.PhysicalFilterPlan)
end
