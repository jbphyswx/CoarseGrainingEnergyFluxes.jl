
# physical-space filtering algorithms
Test.@testset "Filtering" begin
    geom = FG.Geometry.CartesianGeometry()
    x = collect(0.0:100.0:1000.0) # 11 points
    y = collect(0.0:100.0:1000.0) # 11 points
    mask = trues(11, 11)
    grid = FG.Grids.StructuredGrid(geom, x, y, mask)

    # Constant field filtering must return exactly the same constant
    field = fill(42.0, 11, 11)
    out = zeros(11, 11)
    CGEF.Filtering.filter_field!(out, field, grid, CGEF.TopHatKernel(), 300.0)

    # Active (unmasked) cells must have the exact filtered value (42.0)
    Test.@test out[5, 5] ≈ 42.0

    # Test division by zero protection with single-latitude grid
    # This catches the InexactError: Int64(Inf) bug
    geom_sph = FG.Geometry.SphericalGeometry(6371000.0)
    lon_sph = collect(0.0:5.0:355.0)
    lat_sph = [0.0]  # Single latitude
    mask_sph = trues(length(lon_sph), 1)
    grid_sph = FG.Grids.StructuredGrid(geom_sph, deg2rad.(lon_sph), deg2rad.(lat_sph), mask_sph)

    field_sph = rand(length(lon_sph), 1)
    out_sph = zeros(length(lon_sph), 1)

    # This should not throw InexactError
    Test.@test_nowarn CGEF.Filtering.filter_field!(out_sph, field_sph, grid_sph, CGEF.TopHatKernel(), 1e6)
end


# filter_field!/filter_fields! reimplemented as thin wrappers over plan_filter+filter_apply!
# (`workspace=` renamed to `filter_plan=`, a real prebuilt-plan reuse point, not a dead parameter).
Test.@testset "filter_field!/filter_fields!: filter_plan= reuse is correctness-preserving" begin
    geom = FG.Geometry.CartesianGeometry()
    N = 20
    xsR = 0.0:1000.0:(N - 1) * 1000.0
    mask = trues(N, N); mask[8:11, 8:11] .= false
    grid = FG.Grids.StructuredGrid(geom, xsR, xsR, mask)
    ker = CGEF.TopHatKernel(); scale = 4000.0
    u = randn(N, N); v = randn(N, N)

    out_noplan = zeros(N, N)
    CGEF.Filtering.filter_field!(out_noplan, u, grid, ker, scale)

    plan = CGEF.Filtering.plan_filter(grid, ker, scale)
    out_plan = zeros(N, N)
    CGEF.Filtering.filter_field!(out_plan, u, grid, ker, scale; filter_plan = plan)
    Test.@test out_plan == out_noplan

    outs_noplan = (zeros(N, N), zeros(N, N))
    CGEF.Filtering.filter_fields!(outs_noplan, (u, v), grid, ker, scale)
    outs_plan = (zeros(N, N), zeros(N, N))
    CGEF.Filtering.filter_fields!(outs_plan, (u, v), grid, ker, scale; filter_plan = plan)
    Test.@test outs_plan[1] == outs_noplan[1]
    Test.@test outs_plan[2] == outs_noplan[2]

    # 3D-layered filter_field! on a non-serial backend must build the plan once and reuse it across
    # every layer — checked against per-layer manual calls sharing one explicit plan. A per-layer
    # rebuild gives the same numbers, just wastefully, so this comparison alone does not prove
    # reuse; the real proof is the allocation bound in test_allocs.jl. This checks correctness
    # end-to-end for the 3D entry point on a non-serial backend at all, which had no coverage.
    Nz = 3
    u3 = randn(N, N, Nz)
    out3_threaded = zeros(N, N, Nz)
    CGEF.Filtering.filter_field!(out3_threaded, u3, grid, ker, scale; backend = CGEF.ComputationalBackends.ThreadedBackend())
    out3_manual = zeros(N, N, Nz)
    for k in 1:Nz
        CGEF.Filtering.filter_apply!(view(out3_manual, :, :, k), view(u3, :, :, k), plan)
    end
    Test.@test out3_threaded == out3_manual
end


# Reusable filter plans + batched multi-field filtering
Test.@testset "Filter plans & batching" begin
    geom = FG.Geometry.CartesianGeometry()
    x = collect(0.0:1000.0:30e3)
    y = collect(0.0:1000.0:30e3)
    grid = FG.Grids.StructuredGrid(geom, x, y, trues(length(x), length(y)))
    u = rand(length(x), length(y))
    v = rand(length(x), length(y))
    kern = CGEF.TopHatKernel()
    scale = 5000.0

    # A prebuilt plan applied to a field matches a direct filter_field! call. Pin SerialBackend so
    # the footprint-based PhysicalFilterPlan (whose reuse/allocation we assert below) is built
    # regardless of how many threads the suite happens to run with.
    plan = CGEF.Filtering.plan_filter(grid, kern, scale; backend = CGEF.ComputationalBackends.SerialBackend())
    out_plan = zeros(size(u))
    out_direct = zeros(size(u))
    CGEF.Filtering.filter_apply!(out_plan, u, plan)
    CGEF.Filtering.filter_field!(out_direct, u, grid, kern, scale)
    Test.@test out_plan ≈ out_direct

    # Batched filter_fields! matches per-field filtering.
    ou = zeros(size(u)); ov = zeros(size(v))
    CGEF.Filtering.filter_fields!((ou, ov), (u, v), grid, kern, scale)
    ru = zeros(size(u)); rv = zeros(size(v))
    CGEF.Filtering.filter_field!(ru, u, grid, kern, scale)
    CGEF.Filtering.filter_field!(rv, v, grid, kern, scale)
    Test.@test ou ≈ ru
    Test.@test ov ≈ rv

    # Reapplying a prebuilt plan must NOT rebuild the footprint (a rebuild would allocate the
    # offset/weight vectors, ~kBs); a reused plan allocates essentially nothing.
    CGEF.Filtering.filter_apply!(out_plan, u, plan)  # warm up
    Test.@test (@allocated CGEF.Filtering.filter_apply!(out_plan, u, plan)) < 256
end



# -----------------------------------------------------------------------
# Cache-strategy control, cached/streaming bit-identical equivalence, `filter_apply_batch!`
# correctness, and the separable-Gaussian fast path. Allocation SIZE assertions for these same
# features live in test_allocs.jl; everything here is value correctness.
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# Exact prefix-sum top-hat engine (`PrefixSumTopHatPlan`): reduces the real-space top-hat
# convolution from O(N·di_lim·dj_lim) to O(N·dj_lim) with NO approximation, on any rectilinear 2D
# StructuredGrid. Cross-checked against the general scattered engine (the pre-existing,
# independently-tested implementation) to roundoff, then separately checked for the semantic
# distinctions a fast-vs-reference comparison alone cannot catch — if BOTH paths ignored the mask,
# or if ZeroFill and Deformable silently agreed, a pure agreement test would still pass.
# -----------------------------------------------------------------------
Test.@testset "Prefix-sum top-hat engine: exact agreement with the general scattered engine" begin
    function reference_scattered(field, grid, scale, strategy)
        fp = CGEF.Filtering._build_footprint_scattered(
            grid, CGEF.TopHatKernel(), scale;
            mask_strategy = strategy, cache_strategy = CGEF.Filtering.AlwaysCache(),
        )
        out = zeros(FG.Grids.size_tuple(grid))
        return CGEF.Filtering.apply_footprint!(
            out, field, grid, fp, strategy,
            FG.Grids.isperiodic(grid, 1), FG.Grids.isperiodic(grid, 2),
        )
    end

    function check_agreement(grid, scale, strategy)
        Nx, Ny = FG.Grids.size_tuple(grid)
        field = [sin(i / 6.3) * cos(j / 4.1) + 0.3 for i in 1:Nx, j in 1:Ny]
        fp = CGEF.Filtering.build_footprint(grid, CGEF.TopHatKernel(), scale; mask_strategy = strategy)
        Test.@test fp isa CGEF.Filtering.PrefixSumTopHatPlan
        fast = zeros(Nx, Ny)
        CGEF.Filtering._apply_serial!(fast, field, grid, fp, strategy)
        ref = reference_scattered(field, grid, scale, strategy)
        # Exact in exact arithmetic; differs only by floating-point reassociation (prefix sums vs.
        # a direct per-point accumulation), hence a roundoff-level rtol rather than `==`.
        Test.@test maximum(abs, fast .- ref) / max(maximum(abs, ref), eps()) < 1e-11
    end

    geom = FG.Geometry.CartesianGeometry()
    N = 40
    xR = 0.0:1000.0:(N - 1) * 1000.0
    xV = collect(xR) .+ [0.3 * 1000.0 * sin(2.7i) for i in 1:N]
    yV = collect(xR) .+ [0.25 * 1000.0 * cos(1.9i) for i in 1:N]
    R = 6.371e6
    sgeom = FG.Geometry.SphericalGeometry(R)
    lonR = range(0.0; step = deg2rad(360 / 48), length = 48)
    latR = range(deg2rad(-60.0); step = deg2rad(4.0), length = 31)
    lonV = collect(lonR) .+ [0.2 * deg2rad(360 / 48) * sin(3.1i) for i in 1:48]
    latV = collect(latR) .+ [0.2 * deg2rad(4.0) * cos(2.3j) for j in 1:31]

    for strat in (CGEF.Filtering.Deformable(), CGEF.Filtering.ZeroFill())
        # Cartesian: uniform Range, genuinely nonuniform Vector, masked, and both periodicities.
        check_agreement(FG.Grids.StructuredGrid(geom, xR, xR, trues(N, N)), 8000.0, strat)
        check_agreement(FG.Grids.StructuredGrid(geom, xV, yV, trues(N, N)), 8000.0, strat)
        mask = trues(N, N); mask[12:18, 10:16] .= false; mask[1, :] .= false
        check_agreement(FG.Grids.StructuredGrid(geom, xV, yV, mask), 8000.0, strat)
        check_agreement(FG.Grids.StructuredGrid(geom, xR, xR, trues(N, N); periodic = (true, true)), 6000.0, strat)
        # A nonuniform periodic axis has no seam gap derivable from its samples, so the wrap length is
        # explicit. `xV` is a perturbed uniform axis, so the domain it wraps over is still N·Δ.
        check_agreement(
            FG.Grids.StructuredGrid(
                geom, xV, yV, trues(N, N); periodic = (true, false), period = (N * 1000.0, 0.0),
            ),
            6000.0, strat,
        )

        # Spherical: global (periodic longitude), regional nonuniform, and masked global.
        check_agreement(FG.Grids.StructuredGrid(sgeom, lonR, latR, trues(48, 31)), 700e3, strat)
        check_agreement(FG.Grids.StructuredGrid(sgeom, lonV, latV, trues(48, 31); periodic = false), 700e3, strat)
        smask = trues(48, 31); smask[10:20, 8:14] .= false
        check_agreement(FG.Grids.StructuredGrid(sgeom, lonR, latR, smask), 700e3, strat)
    end
end


Test.@testset "Prefix-sum top-hat engine: masking/strategy semantics and exact physical limits" begin
    geom = FG.Geometry.CartesianGeometry()
    N = 40
    x = collect(0.0:1000.0:(N - 1) * 1000.0) .+ [0.3 * 1000.0 * sin(2.7i) for i in 1:N]
    y = collect(0.0:1000.0:(N - 1) * 1000.0) .+ [0.25 * 1000.0 * cos(1.9i) for i in 1:N]
    field = [sin(i / 6.3) * cos(j / 4.1) + 0.3 for i in 1:N, j in 1:N]
    scale = 8000.0
    m = trues(N, N); m[12:18, 10:16] .= false; m[1, :] .= false
    g_un = FG.Grids.StructuredGrid(geom, x, y, trues(N, N))
    g_m = FG.Grids.StructuredGrid(geom, x, y, m)

    function run_filter(grid, strat, f = field)
        fp = CGEF.Filtering.build_footprint(grid, CGEF.TopHatKernel(), scale; mask_strategy = strat)
        out = zeros(N, N)
        CGEF.Filtering._apply_serial!(out, f, grid, fp, strat)
        return out
    end

    o_un = run_filter(g_un, CGEF.Filtering.Deformable())
    o_m_def = run_filter(g_m, CGEF.Filtering.Deformable())
    o_m_zf = run_filter(g_m, CGEF.Filtering.ZeroFill())

    # Masked-out points are exactly zero, and masking genuinely changes active points (so a
    # mask-ignoring implementation could not pass the agreement test above by accident).
    Test.@test all(o_m_def[.!m] .== 0.0)
    Test.@test maximum(abs, (o_m_def .- o_un)[m]) > 1e-3
    # The two strategies use genuinely different denominators near a mask boundary.
    Test.@test maximum(abs, (o_m_zf .- o_m_def)[m]) > 1e-3

    # A constant field must come back exactly (numerator and denominator share every weight).
    oc = run_filter(g_m, CGEF.Filtering.Deformable(), fill(7.25, N, N))
    Test.@test all(abs.(oc[m] .- 7.25) .< 1e-12)

    # A filter wider than the whole sphere must return the exact global area-weighted mean.
    R = 6.371e6
    gs = FG.Grids.StructuredGrid(
        FG.Geometry.SphericalGeometry(R),
        range(0.0; step = deg2rad(10.0), length = 36),
        range(deg2rad(-80.0); step = deg2rad(5.0), length = 33),
        trues(36, 33),
    )
    sf = [sin(2 * (i - 1) * deg2rad(10.0)) * cos((j - 17) * deg2rad(5.0)) + 2.0 for i in 1:36, j in 1:33]
    fps = CGEF.Filtering.build_footprint(gs, CGEF.TopHatKernel(), 4 * π * R)
    os = zeros(36, 33)
    CGEF.Filtering._apply_serial!(os, sf, gs, fps, CGEF.Filtering.Deformable())
    wsum = sum(FG.Grids.area(gs, i, j) for i in 1:36, j in 1:33)
    gmean = sum(sf[i, j] * FG.Grids.area(gs, i, j) for i in 1:36, j in 1:33) / wsum
    Test.@test maximum(abs, os .- gmean) < 1e-10 * abs(gmean)

    # A Deformable apply against a ZeroFill-built plan on a masked grid has no renormalization data
    # and must THROW rather than silently computing ZeroFill (the silent-wrong-answer shape of a
    # mask_strategy that isn't threaded through to build_footprint).
    fp_zf = CGEF.Filtering.build_footprint(g_m, CGEF.TopHatKernel(), scale; mask_strategy = CGEF.Filtering.ZeroFill())
    Test.@test_throws ArgumentError CGEF.Filtering._apply_serial!(
        zeros(N, N), field, g_m, fp_zf, CGEF.Filtering.Deformable(),
    )
end


Test.@testset "cache-strategy control (AutoCache budget, AlwaysCache/NeverCache force)" begin
    # The cache is the scattered engine's per-point neighbour list, so this needs a kernel with no
    # closed form: top-hat takes the exact prefix-sum path and a Gaussian is separable on any
    # rectilinear grid, uniform or not — neither has a per-point list to cache. `SharpSpectralKernel`
    # has neither shortcut, so it is what actually reaches the machinery under test.
    #
    # The scale is chosen to reproduce the search window the Gaussian at 20 km used, so every
    # cache-size figure asserted below is unchanged. Both kernels' radii are linear in the scale.
    ker = CGEF.SharpSpectralKernel()
    geom = FG.Geometry.CartesianGeometry()
    N = 60
    x_nu = collect(0.0:1000.0:(N - 1) * 1000.0) .+ [iseven(i) ? 3.0 : -2.0 for i in 1:N]
    grid = FG.Grids.StructuredGrid(geom, x_nu, x_nu, trues(N, N))
    scale = 20_000.0 * CGEF.Kernels.kernel_radius(CGEF.GaussianKernel(), 20_000.0) /
            CGEF.Kernels.kernel_radius(ker, 20_000.0)

    fp_always = CGEF.Filtering.build_footprint(grid, ker, scale; cache_strategy = CGEF.Filtering.AlwaysCache())
    fp_never = CGEF.Filtering.build_footprint(grid, ker, scale; cache_strategy = CGEF.Filtering.NeverCache())
    Test.@test fp_always isa CGEF.Filtering.ScatteredFilterPlan
    Test.@test fp_always.cache !== nothing
    Test.@test fp_never.cache === nothing

    fp_auto_generous = CGEF.Filtering.build_footprint(
        grid, ker, scale; cache_strategy = CGEF.Filtering.AutoCache(), cache_byte_budget = 10 * 1024^3,
    )
    Test.@test fp_auto_generous.cache !== nothing
    fp_auto_tiny = CGEF.Filtering.build_footprint(
        grid, ker, scale; cache_strategy = CGEF.Filtering.AutoCache(), cache_byte_budget = 1,
    )
    Test.@test fp_auto_tiny.cache === nothing

    # 1D nonuniform (NDScatteredFilterPlan) — same cache-strategy contract.
    grid1 = FG.Grids.StructuredGrid(geom, x_nu, trues(N))
    fp1_always = CGEF.Filtering.build_footprint(grid1, ker, scale; cache_strategy = CGEF.Filtering.AlwaysCache())
    fp1_never = CGEF.Filtering.build_footprint(grid1, ker, scale; cache_strategy = CGEF.Filtering.NeverCache())
    Test.@test fp1_always isa CGEF.Filtering.NDScatteredFilterPlan
    Test.@test fp1_always.cache !== nothing
    Test.@test fp1_never.cache === nothing

    # CurvilinearGrid — same cache-strategy contract (its only real-space engine).
    Nc = 30
    ic = collect(0.0:(Nc - 1)); jc = collect(0.0:(Nc - 1))
    clon = [1000.0 * ii for ii in ic, jj in jc]
    clat = [1000.0 * jj for ii in ic, jj in jc]
    cgrid = FG.Grids.CurvilinearGrid(geom, clon, clat, trues(Nc, Nc))
    cfp_always = CGEF.Filtering.build_footprint(cgrid, ker, scale; cache_strategy = CGEF.Filtering.AlwaysCache())
    cfp_never = CGEF.Filtering.build_footprint(cgrid, ker, scale; cache_strategy = CGEF.Filtering.NeverCache())
    Test.@test cfp_always.cache !== nothing
    Test.@test cfp_never.cache === nothing

    # plan_filter itself must thread cache_strategy through (not just the internal builder).
    plan_never = CGEF.Filtering.plan_filter(grid, ker, scale; cache_strategy = CGEF.Filtering.NeverCache())
    Test.@test plan_never.footprint.cache === nothing
end


Test.@testset "cached vs streaming apply are bit-identical (same candidate-iteration order)" begin
    # Cached and streaming are two branches of the SCATTERED engine, so this needs the one kernel
    # that reaches it — see the cache-strategy testset above.
    ker = CGEF.SharpSpectralKernel()
    geom = FG.Geometry.CartesianGeometry()
    N = 45
    x_nu = collect(0.0:900.0:(N - 1) * 900.0) .+ [iseven(i) ? 3.0 : -2.0 for i in 1:N]
    y_nu = collect(0.0:1100.0:(N - 1) * 1100.0) .+ [isodd(i) ? -4.0 : 1.5 for i in 1:N]
    mask = trues(N, N); mask[10:13, 10:13] .= false
    grid = FG.Grids.StructuredGrid(geom, x_nu, y_nu, mask)
    # Same window the Gaussian at 12 km spanned, so the cache stays a sane size.
    scale = 12_000.0 * CGEF.Kernels.kernel_radius(CGEF.GaussianKernel(), 12_000.0) /
            CGEF.Kernels.kernel_radius(ker, 12_000.0)
    field = randn(N, N)

    for strategy in (CGEF.Filtering.ZeroFill(), CGEF.Filtering.Deformable())
        plan_cached = CGEF.Filtering.plan_filter(grid, ker, scale; mask_strategy = strategy, cache_strategy = CGEF.Filtering.AlwaysCache())
        plan_stream = CGEF.Filtering.plan_filter(grid, ker, scale; mask_strategy = strategy, cache_strategy = CGEF.Filtering.NeverCache())
        Test.@test plan_cached.footprint.cache !== nothing
        Test.@test plan_stream.footprint.cache === nothing

        out_cached = zeros(N, N); out_stream = zeros(N, N)
        CGEF.Filtering.filter_apply!(out_cached, field, plan_cached)
        CGEF.Filtering.filter_apply!(out_stream, field, plan_stream)
        Test.@test out_cached == out_stream
    end

    # 1D nonuniform (NDScatteredFilterPlan).
    grid1 = FG.Grids.StructuredGrid(geom, x_nu, trues(N))
    plan1_cached = CGEF.Filtering.plan_filter(grid1, ker, scale; cache_strategy = CGEF.Filtering.AlwaysCache())
    plan1_stream = CGEF.Filtering.plan_filter(grid1, ker, scale; cache_strategy = CGEF.Filtering.NeverCache())
    u1 = randn(N)
    o1c = zeros(N); o1s = zeros(N)
    CGEF.Filtering.filter_apply!(o1c, u1, plan1_cached)
    CGEF.Filtering.filter_apply!(o1s, u1, plan1_stream)
    Test.@test o1c == o1s

    # True-3D nonuniform (NDScatteredFilterPlan).
    geom3 = FG.Geometry.CartesianGeometry()
    N3 = 14
    x3_nu = collect(0.0:900.0:(N3 - 1) * 900.0) .+ [iseven(i) ? 2.0 : -1.5 for i in 1:N3]
    grid3 = FG.Grids.StructuredGrid(geom3, x3_nu, x3_nu, x3_nu, trues(N3, N3, N3))
    plan3_cached = CGEF.Filtering.plan_filter(grid3, ker, 2500.0; cache_strategy = CGEF.Filtering.AlwaysCache())
    plan3_stream = CGEF.Filtering.plan_filter(grid3, ker, 2500.0; cache_strategy = CGEF.Filtering.NeverCache())
    u3 = randn(N3, N3, N3)
    o3c = zeros(N3, N3, N3); o3s = zeros(N3, N3, N3)
    CGEF.Filtering.filter_apply!(o3c, u3, plan3_cached)
    CGEF.Filtering.filter_apply!(o3s, u3, plan3_stream)
    Test.@test o3c == o3s

    # CurvilinearGrid (ScatteredFilterPlan, non-Cartesian `is_cartesian=false` branch).
    Nc = 26
    ic = collect(0.0:(Nc - 1)); jc = collect(0.0:(Nc - 1))
    θ = deg2rad(20.0)
    clon = [1000.0 * (ii * cos(θ) - jj * 0.2 * sin(θ)) for ii in ic, jj in jc]
    clat = [1000.0 * (ii * sin(θ) + jj * (1 + 0.2 * cos(θ))) for ii in ic, jj in jc]
    cmask = trues(Nc, Nc); cmask[5:7, 5:7] .= false
    cgrid = FG.Grids.CurvilinearGrid(geom, clon, clat, cmask)
    cplan_cached = CGEF.Filtering.plan_filter(cgrid, ker, 6000.0; cache_strategy = CGEF.Filtering.AlwaysCache())
    cplan_stream = CGEF.Filtering.plan_filter(cgrid, ker, 6000.0; cache_strategy = CGEF.Filtering.NeverCache())
    uc = randn(Nc, Nc)
    occ = zeros(Nc, Nc); ocs = zeros(Nc, Nc)
    CGEF.Filtering.filter_apply!(occ, uc, cplan_cached)
    CGEF.Filtering.filter_apply!(ocs, uc, cplan_stream)
    Test.@test occ == ocs
end


# Every footprint/plan type reaches the batch path through its own method, so a type with no batch
# method is a MethodError the moment a caller filters more than one field on that grid. Each is
# checked on every backend it has a hook for, since the backends route batching separately too.
Test.@testset "filter_apply_batch! is bit-identical to per-field filter_apply! (all footprint/plan types)" begin
    function _check_batch(grid, kernel, scale, dims::Integer...; K::Integer = 4, kwargs...)
        fields = ntuple(_ -> randn(dims...), K)
        for backend in (CGEF.ComputationalBackends.SerialBackend(), CGEF.ComputationalBackends.ThreadedBackend())
            # Ask only for a backend this grid has a hook for; `plan_filter` rejects the rest by
            # design, and that rejection is covered elsewhere.
            CGEF.Filtering._backend_supported(grid, backend) || continue
            plan = CGEF.Filtering.plan_filter(grid, kernel, scale; backend = backend, kwargs...)
            outs_ref = ntuple(_ -> zeros(dims...), K)
            for k in 1:K
                CGEF.Filtering.filter_apply!(outs_ref[k], fields[k], plan)
            end
            outs_batch = ntuple(_ -> zeros(dims...), K)
            CGEF.Filtering.filter_apply_batch!(outs_batch, fields, plan)
            for k in 1:K
                Test.@test outs_batch[k] == outs_ref[k]
            end
            # Vector (runtime-determined batch size) form must agree too.
            fields_v = [fields...]
            outs_v = [zeros(dims...) for _ in 1:K]
            CGEF.Filtering.filter_apply_batch!(outs_v, fields_v, plan)
            for k in 1:K
                Test.@test outs_v[k] == outs_ref[k]
            end
        end
    end

    geom = FG.Geometry.CartesianGeometry()
    N = 32
    xsR = 0.0:1000.0:(N - 1) * 1000.0
    x_nu = collect(xsR) .+ [iseven(i) ? 4.0 : -3.0 for i in 1:N]
    mask = trues(N, N); mask[6:9, 6:9] .= false

    # Fast Range-axis TopHat (FilterFootprint).
    grid_fast = FG.Grids.StructuredGrid(geom, xsR, xsR, mask)
    _check_batch(grid_fast, CGEF.TopHatKernel(), 5000.0, N, N)

    # Separable Gaussian fast path (SeparableFootprint).
    _check_batch(grid_fast, CGEF.GaussianKernel(), 5000.0, N, N)

    # Nonuniform-axis (ScatteredFilterPlan) — cached and streaming.
    grid_nu = FG.Grids.StructuredGrid(geom, x_nu, x_nu, mask)
    _check_batch(grid_nu, CGEF.TopHatKernel(), 5000.0, N, N; cache_strategy = CGEF.Filtering.AlwaysCache())
    _check_batch(grid_nu, CGEF.TopHatKernel(), 5000.0, N, N; cache_strategy = CGEF.Filtering.NeverCache())

    # Spherical per-latitude-band fast path (never separable/uniform-Cartesian).
    R = 6.371e6
    lonR = range(0.0; step = deg2rad(4.0), length = N)
    latR = range(deg2rad(-40.0); step = deg2rad(4.0), length = N)
    sgrid = FG.Grids.StructuredGrid(FG.Geometry.SphericalGeometry(R), lonR, latR, mask)
    _check_batch(sgrid, CGEF.TopHatKernel(), 400e3, N, N)

    # 1D nonuniform (NDScatteredFilterPlan) — cached and streaming.
    grid1 = FG.Grids.StructuredGrid(geom, x_nu, trues(N))
    _check_batch(grid1, CGEF.TopHatKernel(), 5000.0, N; cache_strategy = CGEF.Filtering.AlwaysCache())
    _check_batch(grid1, CGEF.TopHatKernel(), 5000.0, N; cache_strategy = CGEF.Filtering.NeverCache())

    # 1D uniform Gaussian (SeparableFootprintND).
    _check_batch(FG.Grids.StructuredGrid(geom, xsR, trues(N)), CGEF.GaussianKernel(), 5000.0, N)

    # True-3D nonuniform (NDScatteredFilterPlan) and true-3D uniform (FilterFootprintND,
    # SeparableFootprintND).
    N3 = 12
    x3R = 0.0:1000.0:(N3 - 1) * 1000.0
    x3_nu = collect(x3R) .+ [iseven(i) ? 3.0 : -2.0 for i in 1:N3]
    grid3 = FG.Grids.StructuredGrid(geom, x3_nu, x3_nu, x3_nu, trues(N3, N3, N3))
    _check_batch(grid3, CGEF.TopHatKernel(), 2500.0, N3, N3, N3)
    grid3u = FG.Grids.StructuredGrid(geom, x3R, x3R, x3R, trues(N3, N3, N3))
    _check_batch(grid3u, CGEF.TopHatKernel(), 2500.0, N3, N3, N3)
    _check_batch(grid3u, CGEF.GaussianKernel(), 2500.0, N3, N3, N3)

    # CurvilinearGrid.
    Nc = 20
    ic = collect(0.0:(Nc - 1)); jc = collect(0.0:(Nc - 1))
    clon = [1000.0 * ii for ii in ic, jj in jc]
    clat = [1000.0 * jj for ii in ic, jj in jc]
    cgrid = FG.Grids.CurvilinearGrid(geom, clon, clat, trues(Nc, Nc))
    _check_batch(cgrid, CGEF.TopHatKernel(), 3000.0, Nc, Nc)

    # Node set, real-space engine (NodeFilterPlan).
    npts = 150
    nodes_lon = [Float64(mod(2.399963 * t, 2π)) for t in 1:npts]
    nodes_lat = Float64(π) / 2 .- [Float64(π * (t - 0.5) / npts) for t in 1:npts]
    node_grid = FG.Grids.UnstructuredGrid(
        FG.Geometry.SphericalGeometry(R), nodes_lon, nodes_lat, fill(4π * R^2 / npts, npts), trues(npts),
    )
    _check_batch(node_grid, CGEF.TopHatKernel(), 4.0e5, npts; method = CGEF.Filtering.RealSpace())

    # Spectral (FFTW) plan — the generic `AbstractFilterPlan` fallback (no shared per-point
    # derivation to hoist out, but must still be correct).
    gridp = FG.Grids.StructuredGrid(geom, xsR, xsR, trues(N, N); periodic = (true, true))
    _check_batch(gridp, CGEF.GaussianKernel(), 5000.0, N, N; method = CGEF.Filtering.Spectral())
end


Test.@testset "separable fast path — dispatch and correctness against an independent brute-force square-truncated reference" begin
    geom = FG.Geometry.CartesianGeometry()  # deliberately anisotropic dx/dy
    N = 24
    xsR = 0.0:1000.0:(N - 1) * 1000.0
    ysR = 0.0:1500.0:(N - 1) * 1500.0
    ker = CGEF.GaussianKernel()
    scale = 4000.0

    # Dispatch: Range axes -> SeparableFootprint.
    grid_range = FG.Grids.StructuredGrid(geom, xsR, ysR, trues(N, N))
    fp_range = CGEF.Filtering.build_footprint(grid_range, ker, scale)
    Test.@test fp_range isa CGEF.Filtering.SeparableFootprint

    # Dispatch: separability is a property of the KERNEL, so a `Vector` axis takes the same engine.
    # What the axis type decides is the weight table's rank — a vector when the spacing is constant
    # and known from the type, a row per position otherwise — not which algorithm runs.
    grid_uniform_vec = FG.Grids.StructuredGrid(geom, collect(xsR), collect(ysR), trues(N, N))
    fp_uniform_vec = CGEF.Filtering.build_footprint(grid_uniform_vec, ker, scale)
    Test.@test fp_uniform_vec isa CGEF.Filtering.SeparableFootprint
    Test.@test fp_uniform_vec.gx isa AbstractMatrix
    Test.@test fp_range.gx isa AbstractVector

    x_nu = collect(xsR) .+ [iseven(i) ? 5.0 : -5.0 for i in 1:N]
    grid_nonuniform = FG.Grids.StructuredGrid(geom, x_nu, collect(ysR), trues(N, N))
    fp_nonuniform = CGEF.Filtering.build_footprint(grid_nonuniform, ker, scale)
    Test.@test fp_nonuniform isa CGEF.Filtering.SeparableFootprint
    Test.@test size(fp_nonuniform.gx, 1) == N

    # Dispatch: Spherical + Gaussian never takes the Cartesian-only separable path, even with
    # Range axes (great-circle distance does not factor into Gx(Δx)·Gy(Δy)).
    sgeom = FG.Geometry.SphericalGeometry(6.371e6)
    slonR = range(0.0; step = deg2rad(4.0), length = N)
    slatR = range(deg2rad(-40.0); step = deg2rad(4.0), length = N)
    sgrid = FG.Grids.StructuredGrid(sgeom, slonR, slatR, trues(N, N))
    fp_sph = CGEF.Filtering.build_footprint(sgrid, ker, deg2rad(6.0) * 6.371e6)
    Test.@test !(fp_sph isa CGEF.Filtering.SeparableFootprint)

    # Correctness: cross-check against an INDEPENDENT brute-force square-truncated reference
    # (written fresh here, not reusing any of the package's own engines) — this validates the
    # actual mathematical claim (row-pass-then-column-pass == the full 2D square-truncated sum),
    # not just "close to the disk-truncated RealSpace engine" (a genuinely different truncation
    # shape — see `SeparableFootprint`'s own docstring).
    for strategy in (CGEF.Filtering.ZeroFill(), CGEF.Filtering.Deformable())
        mask = trues(N, N); mask[8:11, 8:11] .= false; mask[1, :] .= false
        grid_m = FG.Grids.StructuredGrid(geom, xsR, ysR, mask)
        fp_m = CGEF.Filtering.build_footprint(grid_m, ker, scale; mask_strategy = strategy)
        Test.@test fp_m isa CGEF.Filtering.SeparableFootprint

        field = [sin(i / 3.0) * cos(j / 4.0) for i in 1:N, j in 1:N]
        out_fast = zeros(N, N)
        CGEF.Filtering.apply_separable!(out_fast, field, grid_m, fp_m, strategy)

        di_lim, dj_lim = fp_m.di_lim, fp_m.dj_lim
        gx, gy = fp_m.gx, fp_m.gy
        out_ref = zeros(N, N)
        for j in 1:N, i in 1:N
            mask[i, j] || continue
            wsum = 0.0; wnorm = 0.0
            for ddj in -dj_lim:dj_lim, ddi in -di_lim:di_lim
                ii, jj = i + ddi, j + ddj
                (1 <= ii <= N && 1 <= jj <= N) || continue
                w = gx[ddi + di_lim + 1] * gy[ddj + dj_lim + 1]
                if strategy isa CGEF.Filtering.ZeroFill
                    wnorm += w
                    mask[ii, jj] && (wsum += w * field[ii, jj])
                elseif mask[ii, jj]
                    wnorm += w
                    wsum += w * field[ii, jj]
                end
            end
            out_ref[i, j] = wnorm > 1e-15 ? wsum / wnorm : 0.0
        end
        Test.@test out_fast ≈ out_ref rtol = 1e-10
    end
end


# Separability is a property of the KERNEL, not of the spacing: `exp(-α(Δx²+Δy²)/ℓ²)` factorizes on
# any rectilinear grid. A stretched axis only makes the per-axis weight depend on position as well
# as offset, so it takes the same two-pass engine rather than falling to the scattered one.
Test.@testset "Separable Gaussian on a stretched Cartesian grid" begin
    geom = FG.Geometry.CartesianGeometry()
    n = 41
    xr = range(0.0, 40e3; length = n)
    xs = [20e3 * (1 - cos(π * (i - 1) / (n - 1))) for i in 1:n]   # same endpoints, clustered ends
    ℓ = 5000.0
    K = CGEF.GaussianKernel()
    f = [sin(x / 9000) * cos(y / 7000) for x in xr, y in xr]

    gr = FG.Grids.StructuredGrid(geom, xr, xr, trues(n, n))
    gs = FG.Grids.StructuredGrid(geom, xs, xs, trues(n, n))
    pr = CGEF.Filtering.plan_filter(gr, K, ℓ)
    ps = CGEF.Filtering.plan_filter(gs, K, ℓ)

    # Both take the separable engine; only the weight table's shape differs.
    Test.@test pr.footprint isa CGEF.Filtering.SeparableFootprint
    Test.@test ps.footprint isa CGEF.Filtering.SeparableFootprint
    Test.@test pr.footprint.gx isa AbstractVector          # shared across positions
    Test.@test ps.footprint.gx isa AbstractMatrix          # one row per position
    Test.@test size(ps.footprint.gx, 1) == n

    # The scattered engine is the independent reference: it forms `kernel_weight(d)·area` per
    # candidate with no separability assumption at all. Agreement to ~1e-10 is the square-vs-disk
    # truncation difference — the Gaussian is 1e-10 at its own truncation radius — not an error.
    # This is also what catches the cell measure being dropped from the separable weights, which is
    # invisible on a uniform grid (constant area cancels in the normalization) and ~16% here.
    for (grid, ax) in ((gr, xr), (gs, xs))
        sep = zeros(n, n)
        CGEF.Filtering.filter_field!(sep, f, grid, K, ℓ)
        sca = zeros(n, n)
        CGEF.Filtering.filter_field!(
            sca, f, grid, K, ℓ;
            filter_plan = CGEF.Filtering.PhysicalFilterPlan(
                CGEF.Filtering._build_footprint_scattered(grid, K, ℓ), grid,
                CGEF.Filtering.Deformable(), K, ℓ, CGEF.ComputationalBackends.SerialBackend(),
            ),
        )
        Test.@test maximum(abs.(sep .- sca)) / maximum(abs, sca) < 1e-8
    end

    # A filter reproduces a constant exactly, whatever the spacing does.
    out = zeros(n, n)
    CGEF.Filtering.filter_field!(out, fill(2.5, n, n), gs, K, ℓ)
    Test.@test all(x -> isapprox(x, 2.5; atol = 1e-12), out)
end


# The Gaussian factorizes in any number of dimensions, and the saving grows with it: a full-box
# footprint enumerates ∏(2wᵈ+1) offsets per point where N passes cost ∑(2wᵈ+1). The reference here
# is that full-box engine, which assumes no separability at all.
Test.@testset "Separable Gaussian in 1D and 3D" begin
    geom = FG.Geometry.CartesianGeometry()
    K = CGEF.GaussianKernel()
    dx = 1000.0
    ℓ = 4000.0
    serial = CGEF.ComputationalBackends.SerialBackend()

    Test.@testset "3D" begin
        n = 20
        x = range(0.0, dx * (n - 1); length = n)
        g3 = FG.Grids.StructuredGrid(geom, x, x, x, trues(n, n, n))
        f = [sin(a / 5000) * cos(b / 6000) * sin(c / 7000) for a in x, b in x, c in x]
        p = CGEF.Filtering.plan_filter(g3, K, ℓ)
        Test.@test p.footprint isa CGEF.Filtering.SeparableFootprintND
        Test.@test length(p.footprint.lim) == 3

        sep = zeros(n, n, n)
        CGEF.Filtering.filter_field!(sep, f, g3, K, ℓ)
        ref = zeros(n, n, n)
        CGEF.Filtering.filter_field!(
            ref, f, g3, K, ℓ;
            filter_plan = CGEF.Filtering.PhysicalFilterPlan(
                CGEF.Filtering._build_footprint_nd(g3, K, ℓ), g3,
                CGEF.Filtering.Deformable(), K, ℓ, serial,
            ),
        )
        Test.@test maximum(abs.(sep .- ref)) / maximum(abs, ref) < 1e-8

        c = zeros(n, n, n)
        CGEF.Filtering.filter_field!(c, fill(2.5, n, n, n), g3, K, ℓ)
        Test.@test all(v -> isapprox(v, 2.5; atol = 1e-12), c)

        # A masked grid under Deformable takes the dense denominator; masked cells read zero and
        # the rest are still a weighted mean, so a constant survives there too.
        m = trues(n, n, n)
        m[3:5, 3:5, 3:5] .= false
        gm = FG.Grids.StructuredGrid(geom, x, x, x, m)
        cm = zeros(n, n, n)
        CGEF.Filtering.filter_field!(cm, fill(2.5, n, n, n), gm, K, ℓ;
                                     mask_strategy = CGEF.Filtering.Deformable())
        Test.@test all(I -> m[I] ? isapprox(cm[I], 2.5; atol = 1e-12) : cm[I] == 0.0,
                       CartesianIndices(cm))

        # The default, ZeroFill, keeps the excluded cells in the denominator, so the same constant is
        # DAMPED wherever the footprint overlaps the hole and exact everywhere else. The separable ND
        # engine's footprint is a per-axis BOX of half-width `lim`, not a ball, so the split between
        # the two halves is taken from the plan's own limits rather than from the kernel radius.
        cz = zeros(n, n, n)
        CGEF.Filtering.filter_field!(cz, fill(2.5, n, n, n), gm, K, ℓ)
        lim = CGEF.Filtering.plan_filter(gm, K, ℓ).footprint.lim
        holes = filter(J -> !m[J], collect(CartesianIndices(m)))
        overlaps(I) = any(J -> all(d -> abs(I[d] - J[d]) <= lim[d], 1:3), holes)
        Test.@test all(iszero, @view cz[.!m])
        Test.@test all(I -> !m[I] || overlaps(I) || isapprox(cz[I], 2.5; atol = 1e-12),
                       CartesianIndices(cz))
        Test.@test any(I -> m[I] && cz[I] < 2.5 - 1e-6, CartesianIndices(cz))
    end

    Test.@testset "1D" begin
        n = 64
        x = range(0.0, dx * (n - 1); length = n)
        g1 = FG.Grids.StructuredGrid(geom, x, trues(n))
        f = [sin(a / 5000) for a in x]
        p = CGEF.Filtering.plan_filter(g1, K, ℓ)
        Test.@test p.footprint isa CGEF.Filtering.SeparableFootprintND

        sep = zeros(n)
        CGEF.Filtering.filter_field!(sep, f, g1, K, ℓ)
        ref = zeros(n)
        CGEF.Filtering.filter_field!(
            ref, f, g1, K, ℓ;
            filter_plan = CGEF.Filtering.PhysicalFilterPlan(
                CGEF.Filtering._build_footprint_nd(g1, K, ℓ), g1,
                CGEF.Filtering.Deformable(), K, ℓ, serial,
            ),
        )
        Test.@test maximum(abs.(sep .- ref)) / maximum(abs, ref) < 1e-8
    end

    # A stretched direction only changes the weight table's shape, exactly as in 2D.
    Test.@testset "stretched third direction" begin
        n = 16
        x = range(0.0, dx * (n - 1); length = n)
        z = [dx * (n - 1) * (i - 1)^2 / (n - 1)^2 for i in 1:n]
        gz = FG.Grids.StructuredGrid(geom, x, x, z, trues(n, n, n))
        p = CGEF.Filtering.plan_filter(gz, K, ℓ)
        Test.@test p.footprint.g[1] isa AbstractVector
        Test.@test p.footprint.g[3] isa AbstractMatrix
        c = zeros(n, n, n)
        CGEF.Filtering.filter_field!(c, fill(1.25, n, n, n), gz, K, ℓ)
        Test.@test all(v -> isapprox(v, 1.25; atol = 1e-12), c)
    end
end


# Mathematical correctness: Filtered field of constant = constant
Test.@testset "Filter Normalization - Constant Field" begin
    # Filtering a constant field must return exactly the same constant
    # This tests that kernel weights are properly normalized

    geom = FG.Geometry.CartesianGeometry()
    x = collect(0.0:1000.0:50e3)
    y = collect(0.0:1000.0:50e3)
    mask = trues(length(x), length(y))
    grid = FG.Grids.StructuredGrid(geom, x, y, mask)

    C = 42.0  # Constant value
    field = fill(C, length(x), length(y))

    # Test both masking strategies
    for kernel in [CGEF.TopHatKernel(), CGEF.GaussianKernel()]
        for scale in [5000.0, 10000.0, 20000.0]
            out_zero = zeros(length(x), length(y))
            out_renorm = zeros(length(x), length(y))

            CGEF.Filtering.filter_field!(out_zero, field, grid, kernel, scale; mask_strategy=CGEF.Filtering.ZeroFill())
            CGEF.Filtering.filter_field!(out_renorm, field, grid, kernel, scale; mask_strategy=CGEF.Filtering.Deformable())

            # Interior points should be exactly C
            for j in 20:length(y)-20, i in 20:length(x)-20
                Test.@test out_zero[i,j] ≈ C rtol=1e-10
                Test.@test out_renorm[i,j] ≈ C rtol=1e-10
            end
        end
    end
end

# The uniform-spherical banded footprint is a single translation-invariant offset list per latitude
# band, so its longitude window has to hold for every row the ball reaches — not just the target's —
# and has to stop after one turn around the ring. Both go wrong only where a ball reaches over a pole,
# which is why this checks against brute force at high latitude specifically.
Test.@testset "Uniform spherical footprint: window covers the pole and counts each column once" begin
    R = 6.371e6
    geo = FG.Geometry.SphericalGeometry(R)
    nlat, nlon = 60, 120
    lat = range(-π / 2 + 1e-6; step = (π - 2e-6) / (nlat - 1), length = nlat)
    lon = range(0.0; step = 2π / nlon, length = nlon)
    grid = FG.Grids.StructuredGrid(geo, lon, lat; periodic = (true, false))
    fld = [cos(3a) * exp(-((abs(b) - 1.40) / 0.12)^2) for a in lon, b in lat]

    ker = CGEF.GaussianKernel()
    scale = 3.0e6
    rad = CGEF.Kernels.kernel_radius(ker, scale)
    # A ball this wide at |φ| ≳ 76° reaches across the pole, which is the configuration that exposes
    # both a short window and a ring counted more than once.
    Test.@test rad > R * (π / 2 - 1.33)

    out = zeros(nlon, nlat)
    CGEF.Filtering.filter_field!(out, fld, grid, ker, scale;
        backend = CGEF.ComputationalBackends.SerialBackend())
    Test.@test CGEF.Filtering.build_footprint(grid, ker, scale) isa CGEF.Filtering.FilterFootprint

    for j in (2, 4, nlat - 6, nlat - 3, nlat - 1), i in (1, 37)
        num = zero(Float64)
        den = zero(Float64)
        for jj in 1:nlat, ii in 1:nlon
            d = FG.Geometry.distance(geo, SA.SVector(lon[i], lat[j]), SA.SVector(lon[ii], lat[jj]))
            d <= rad || continue
            wgt = CGEF.Kernels.kernel_weight(ker, d, scale) * FG.Grids.area(grid, ii, jj)
            num += wgt * fld[ii, jj]
            den += wgt
        end
        Test.@test out[i, j] ≈ (den > 0 ? num / den : 0.0) rtol = 1e-12
    end
end

# The engine restructures below all have a same-answer reference available, so each is asserted
# against it rather than against a recorded number.
Test.@testset "Real-space engine fast paths" begin
    geom = FG.Geometry.CartesianGeometry()
    N = 96
    dx = 1000.0
    xr = 0.0:dx:dx*(N - 1)
    xv = collect(xr)              # same coordinates, but a Vector axis takes the GENERAL path
    u = randn(N, N)

    # Top-hat: the uniform-axis window collapses the two-pointer walk. Same coordinates through both
    # paths must agree EXACTLY — the fast path makes the same boundary decision, it does not
    # approximate it.
    for m in (nothing, (mm = trues(N, N); mm[20:40, 20:40] .= false; mm)),
        strat in (CGEF.Filtering.Deformable(), CGEF.Filtering.ZeroFill()),
        sc in (4000.0, 16000.0)

        gR = m === nothing ? FG.Grids.StructuredGrid(geom, xr, xr) : FG.Grids.StructuredGrid(geom, xr, xr, m)
        gV = m === nothing ? FG.Grids.StructuredGrid(geom, xv, xv) : FG.Grids.StructuredGrid(geom, xv, xv, m)
        oR = zeros(N, N); oV = zeros(N, N)
        CGEF.Filtering.filter_apply!(oR, u, CGEF.Filtering.plan_filter(gR, CGEF.TopHatKernel(), sc;
            backend = CGEF.ComputationalBackends.SerialBackend(), mask_strategy = strat))
        CGEF.Filtering.filter_apply!(oV, u, CGEF.Filtering.plan_filter(gV, CGEF.TopHatKernel(), sc;
            backend = CGEF.ComputationalBackends.SerialBackend(), mask_strategy = strat))
        Test.@test oR == oV
    end

    # Separable Gaussian: the position-major weight table is the matrix path, reached by a Vector axis;
    # the vector-table path is reached by a Range. Same geometry, so they agree to round-off.
    let gR = FG.Grids.StructuredGrid(geom, xr, xr),
        gV = FG.Grids.StructuredGrid(geom, xv, xv),
        oR = zeros(N, N), oV = zeros(N, N)
        CGEF.Filtering.filter_apply!(oR, u, CGEF.Filtering.plan_filter(gR, CGEF.GaussianKernel(), 8000.0;
            backend = CGEF.ComputationalBackends.SerialBackend()))
        CGEF.Filtering.filter_apply!(oV, u, CGEF.Filtering.plan_filter(gV, CGEF.GaussianKernel(), 8000.0;
            backend = CGEF.ComputationalBackends.SerialBackend()))
        Test.@test oR ≈ oV rtol = 1e-12
        # A normalized low-pass returns a constant unchanged, on both tables.
        cR = zeros(N, N); cV = zeros(N, N)
        CGEF.Filtering.filter_apply!(cR, fill(2.75, N, N), CGEF.Filtering.plan_filter(gR, CGEF.GaussianKernel(), 8000.0;
            backend = CGEF.ComputationalBackends.SerialBackend()))
        CGEF.Filtering.filter_apply!(cV, fill(2.75, N, N), CGEF.Filtering.plan_filter(gV, CGEF.GaussianKernel(), 8000.0;
            backend = CGEF.ComputationalBackends.SerialBackend()))
        Test.@test all(≈(2.75; rtol = 1e-12), cR)
        Test.@test all(≈(2.75; rtol = 1e-12), cV)
    end

    # Periodic wrap is split into two constant-offset runs rather than indexed through `mod1`; a
    # nonuniform periodic axis still takes the general path, so it is the reference.
    let xp = range(0.0, dx * N; length = N + 1)[1:N],
        gP = FG.Grids.StructuredGrid(geom, xp, xp; periodic = (true, true)),
        gN = FG.Grids.StructuredGrid(geom, collect(xp), collect(xp); periodic = (true, true), period = (dx * N, dx * N)),
        oP = zeros(N, N), oN = zeros(N, N)
        CGEF.Filtering.filter_apply!(oP, u, CGEF.Filtering.plan_filter(gP, CGEF.GaussianKernel(), 8000.0;
            backend = CGEF.ComputationalBackends.SerialBackend()))
        CGEF.Filtering.filter_apply!(oN, u, CGEF.Filtering.plan_filter(gN, CGEF.GaussianKernel(), 8000.0;
            backend = CGEF.ComputationalBackends.SerialBackend()))
        Test.@test oP ≈ oN rtol = 1e-12
    end
end

# ---------------------------------------------------------------------------
# Why `ZeroFill` is the default mask strategy.
#
# The cross-scale flux budget is derived by commuting the filter with the spatial derivative, so the
# kernel must not depend on position. `ZeroFill` keeps the full kernel mass in the denominator and so
# stays position-independent; `Deformable` divides by the locally-included mass, which varies within ℓ
# of a mask boundary. These testsets pin that difference numerically instead of leaving it as prose.
# ---------------------------------------------------------------------------

# A compactly-supported C³ bump, EXACTLY zero outside `rad` — so `mask ⊙ f == f` and
# `mask ⊙ ∂f == ∂f` hold to the bit, and the identity under test is not contaminated by a tail
# leaking onto the mask.
function _cgef_test_bump(N, ic, jc, rad)
    f = zeros(N, N)
    for j in 1:N, i in 1:N
        r2 = ((i - ic)^2 + (j - jc)^2) / rad^2
        r2 < 1 && (f[i, j] = (1 - r2)^4)
    end
    return f
end

Test.@testset "mask_strategy: ZeroFill commutes with ∂/∂x, Deformable does not" begin
    geom = FG.Geometry.CartesianGeometry()
    N = 96
    dx = 1000.0
    xs = 0.0:dx:(N - 1) * dx
    icoast = 61                                   # first LAND column
    mask = trues(N, N); mask[icoast:end, :] .= false
    grid = FG.Grids.StructuredGrid(geom, xs, xs, mask)
    ker = CGEF.TopHatKernel()
    ℓ = 8000.0                                    # footprint radius ℓ/2 = 4 cells

    f = _cgef_test_bump(N, 49, 48, 8)
    df = zeros(N, N)
    CGEF.Derivatives.ddx!(df, f, grid)

    # Preconditions. Without these the comparison measures the mask cutting into the field rather
    # than the filter operator, and neither strategy would commute.
    Test.@test all(iszero, @view f[icoast:end, :])
    Test.@test all(iszero, @view df[icoast:end, :])

    lhs = zeros(N, N); rhs = zeros(N, N); ff = zeros(N, N)
    rel = Dict{Any,Float64}(); edge = Dict{Any,Float64}(); drift = Dict{Any,Float64}()
    for strat in (CGEF.Filtering.ZeroFill(), CGEF.Filtering.Deformable())
        CGEF.Filtering.filter_field!(lhs, df, grid, ker, ℓ; mask_strategy = strat)
        CGEF.Filtering.filter_field!(ff, f, grid, ker, ℓ; mask_strategy = strat)
        CGEF.Derivatives.ddx!(rhs, ff, grid)
        # `ddx!` runs the `ReduceInRun` policy, which degrades to a one-sided stencil at the domain
        # edge and at the land boundary. Those two columns test the DERIVATIVE's edge treatment, not
        # the filter, so they are excluded here and asserted separately below.
        ii = 2:(icoast - 2)
        s = maximum(abs, @view rhs[ii, :])
        rel[strat] = maximum(abs, @view(lhs[ii, :]) .- @view(rhs[ii, :])) / s
        edge[strat] = maximum(abs, @view(lhs[icoast - 1, :]) .- @view(rhs[icoast - 1, :])) / s
        drift[strat] = (sum(ff) - sum(f)) / sum(f)
    end

    # `ZeroFill`: the identity holds to round-off. `Deformable`: it fails by ~13 orders of magnitude
    # more, and the failure is the whole point of the default — it is asserted, not tolerated.
    Test.@test rel[CGEF.Filtering.ZeroFill()] < 1e-13
    Test.@test rel[CGEF.Filtering.Deformable()] > 1e-3
    Test.@test rel[CGEF.Filtering.Deformable()] > 1e9 * rel[CGEF.Filtering.ZeroFill()]

    # The excluded column is excluded for a reason that is itself checked: there `ZeroFill` fails too,
    # at the same order as `Deformable`, because the one-sided stencil is not the operator the identity
    # is stated for.
    Test.@test edge[CGEF.Filtering.ZeroFill()] > 1e-3

    # A position-independent kernel with the full mass in the denominator conserves the domain
    # integral; a renormalized one does not.
    Test.@test abs(drift[CGEF.Filtering.ZeroFill()]) < 1e-13
    Test.@test abs(drift[CGEF.Filtering.Deformable()]) > 1e-6

    # Unmasked reference: with nothing excluded the two strategies coincide and both commute, for a
    # compact footprint (top-hat) and a truncated-tail one (Gaussian) alike.
    let gfull = FG.Grids.StructuredGrid(geom, xs, xs, trues(N, N)),
        fu = _cgef_test_bump(N, 48, 48, 10), dfu = zeros(N, N)
        CGEF.Derivatives.ddx!(dfu, fu, gfull)
        for k in (CGEF.TopHatKernel(), CGEF.GaussianKernel())
            CGEF.Filtering.filter_field!(lhs, dfu, gfull, k, ℓ)
            CGEF.Filtering.filter_field!(ff, fu, gfull, k, ℓ)
            CGEF.Derivatives.ddx!(rhs, ff, gfull)
            ii = 2:(N - 1)
            Test.@test maximum(abs, @view(lhs[ii, :]) .- @view(rhs[ii, :])) <
                       1e-13 * maximum(abs, @view rhs[ii, :])
        end
    end
end

Test.@testset "mask_strategy: the coast artifacts quoted in the filter_field! docstring" begin
    geom = FG.Geometry.CartesianGeometry()
    N = 200
    xs = 0.0:1.0:(N - 1)                          # unit cells, so ℓ is in cells
    icoast = 101
    mask = trues(N, N); mask[icoast:end, :] .= false
    grid = FG.Grids.StructuredGrid(geom, xs, xs, mask)
    ker = CGEF.GaussianKernel(); ℓ = 16.0
    jm = N ÷ 2

    # Footprint moments AT a point without building an impulse response: filtering is linear, so
    # filter(x)/filter(1) is the footprint-weighted mean of x there, and filter(1) is its mass. Both
    # ratios divide the strategy's normalization out, which is exactly why they come out equal below.
    xc = [xs[i] - xs[icoast] for i in 1:N, _ in 1:N]
    f1 = zeros(N, N); fx = zeros(N, N); fx2 = zeros(N, N)
    ds = (1, 4, 8, 16)
    mass = Dict{Any,Vector{Float64}}(); cent = Dict{Any,Vector{Float64}}(); wid = Dict{Any,Vector{Float64}}()
    for strat in (CGEF.Filtering.ZeroFill(), CGEF.Filtering.Deformable())
        plan = CGEF.Filtering.plan_filter(grid, ker, ℓ; mask_strategy = strat)
        CGEF.Filtering.filter_apply!(f1, ones(N, N), plan)
        CGEF.Filtering.filter_apply!(fx, xc, plan)
        CGEF.Filtering.filter_apply!(fx2, xc .^ 2, plan)
        σint = sqrt(fx2[50, jm] / f1[50, jm] - (fx[50, jm] / f1[50, jm])^2)
        mass[strat] = [f1[icoast - d, jm] for d in ds]
        cent[strat] = [fx[icoast - d, jm] / f1[icoast - d, jm] - xc[icoast - d, jm] for d in ds]
        wid[strat] = [sqrt(fx2[icoast - d, jm] / f1[icoast - d, jm] -
                           (fx[icoast - d, jm] / f1[icoast - d, jm])^2) / σint for d in ds]
    end

    # The shape distortion is the truncated footprint's, so it is IDENTICAL under both strategies —
    # it is not something `Deformable` buys its way out of.
    Test.@test cent[CGEF.Filtering.ZeroFill()] ≈ cent[CGEF.Filtering.Deformable()] rtol = 1e-12
    Test.@test wid[CGEF.Filtering.ZeroFill()] ≈ wid[CGEF.Filtering.Deformable()] rtol = 1e-12
    Test.@test cent[CGEF.Filtering.ZeroFill()] ./ ℓ ≈ [-0.211, -0.111, -0.032, 0.0] atol = 5e-3
    Test.@test wid[CGEF.Filtering.ZeroFill()] ≈ [0.621, 0.747, 0.897, 0.998] atol = 5e-3

    # Only the mass differs, and it is exactly the value a uniform field filters to.
    Test.@test mass[CGEF.Filtering.ZeroFill()] ≈ [0.543, 0.776, 0.948, 0.9996] atol = 5e-3
    Test.@test all(≈(1.0; atol = 1e-12), mass[CGEF.Filtering.Deformable()])
    let uni = zeros(N, N)
        CGEF.Filtering.filter_field!(uni, ones(N, N), grid, ker, ℓ)   # default ⇒ ZeroFill
        Test.@test [uni[icoast - d, jm] for d in ds] ≈ mass[CGEF.Filtering.ZeroFill()] rtol = 1e-12
    end
end
