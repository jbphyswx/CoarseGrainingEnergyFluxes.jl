
# Execution backend lattice (ComputationalBackends.jl)
Test.@testset "Backends" begin
    # Backend resolution returns INSTANCES; AutoBackend picks a concrete local backend. Auto is
    # resolved by this package, not upstream — upstream leaves that method to the consumer — and it
    # takes the grid, since a backend is only selectable if that grid has a path for it.
    Test.@test CGEF.ComputationalBackends.resolve_backend(CGEF.ComputationalBackends.SerialBackend()) === CGEF.ComputationalBackends.SerialBackend()
    let g2 = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), 0.0:1000.0:23e3, 0.0:1000.0:23e3)
        Test.@test CGEF.Filtering._resolve_backend(CGEF.ComputationalBackends.SerialBackend(), g2) === CGEF.ComputationalBackends.SerialBackend()
        Test.@test CGEF.Filtering._resolve_backend(CGEF.ComputationalBackends.AutoBackend(), g2) isa Union{CGEF.ComputationalBackends.SerialBackend, CGEF.ComputationalBackends.ThreadedBackend}
    end

    # distribution wrappers are parametric over an inner local backend
    Test.@test CGEF.ComputationalBackends.DistributedBackend(CGEF.ComputationalBackends.SerialBackend()) isa CGEF.ComputationalBackends.DistributedBackend
    Test.@test CGEF.ComputationalBackends.MPIBackend() isa CGEF.ComputationalBackends.MPIBackend
    Test.@test CGEF.ComputationalBackends.DistributedBackend().inner === CGEF.ComputationalBackends.SerialBackend()
    Test.@test CGEF.ComputationalBackends.local_backend(CGEF.ComputationalBackends.DistributedBackend(CGEF.ComputationalBackends.ThreadedBackend())) === CGEF.ComputationalBackends.ThreadedBackend()
    Test.@test CGEF.ComputationalBackends.is_distributed(CGEF.ComputationalBackends.MPIBackend()) && !CGEF.ComputationalBackends.is_distributed(CGEF.ComputationalBackends.SerialBackend())

    # Backend honesty, over every (grid architecture x backend) pair. A backend request is either
    # honored exactly or refused; `AutoBackend` may choose, but only among backends the grid can
    # actually run. Asserted on the resolution itself rather than on timing, so it is deterministic.
    let cart = FG.Geometry.CartesianGeometry(),
        xr = 0.0:1000.0:23e3,
        nn = 24
        xg = [Float64(i - 1) * 1000.0 for i in 1:nn, j in 1:nn]
        yg = [Float64(j - 1) * 1000.0 for i in 1:nn, j in 1:nn]
        honesty_grids = (
            FG.Grids.StructuredGrid(cart, xr),                       # 1D
            FG.Grids.StructuredGrid(cart, xr, xr),                   # 2D
            FG.Grids.StructuredGrid(cart, 0.0:1e3:7e3, 0.0:1e3:7e3, 0.0:1e3:7e3),  # true 3D
            FG.Grids.CurvilinearGrid(cart, xg, yg, trues(nn, nn)),
            FG.Grids.UnstructuredGrid(cart, vec(xg), vec(yg), trues(nn * nn);
                                      k = 4, areas = fill(1.0e6, nn * nn)),
        )
        backends = (
            CGEF.ComputationalBackends.SerialBackend(),
            CGEF.ComputationalBackends.ThreadedBackend(),
            CGEF.ComputationalBackends.DistributedBackend(),
            CGEF.ComputationalBackends.MPIBackend(),
        )
        for g in honesty_grids
            # Auto never selects a backend this grid cannot honor.
            picked = CGEF.Filtering._resolve_backend(CGEF.ComputationalBackends.AutoBackend(), g)
            Test.@test CGEF.Filtering._backend_supported(g, picked)
            # An explicit unsupported request errors rather than silently running serial.
            for b in backends
                CGEF.Filtering._backend_supported(g, b) && continue
                Test.@test_throws ArgumentError CGEF.Filtering.plan_filter(
                    g, CGEF.TopHatKernel(), 5000.0;
                    backend = b, method = CGEF.Filtering.RealSpace(),
                )
            end
        end
        # Slice-parallel apply: many independent problems, one plan each. Ragged point counts, since
        # that is what a real batch looks like and it is what the longest-first schedule exists for.
        let counts = [37, 121, 64, 199, 88, 150, 45, 176],
            sl_plans = map(counts) do n
                xs = collect(range(0.0, 1.0e4; length = n))
                sg = FG.Grids.StructuredGrid(cart, xs)
                CGEF.Filtering.plan_filter(sg, CGEF.GaussianKernel(), 2000.0;
                    backend = CGEF.ComputationalBackends.SerialBackend(),
                    method = CGEF.Filtering.RealSpace())
            end
            sl_fields = [randn(n) for n in counts]
            o_ser = [zeros(n) for n in counts]
            o_thr = [zeros(n) for n in counts]
            o_ref = [zeros(n) for n in counts]

            CGEF.Filtering.filter_slices!(o_ser, sl_fields, sl_plans;
                backend = CGEF.ComputationalBackends.SerialBackend())
            CGEF.Filtering.filter_slices!(o_thr, sl_fields, sl_plans;
                backend = CGEF.ComputationalBackends.ThreadedBackend())
            # Reference: each slice applied on its own through the ordinary single-plan entry point.
            for t in eachindex(sl_plans)
                CGEF.Filtering.filter_apply!(o_ref[t], sl_fields[t], sl_plans[t])
            end

            # Threading the slice axis reorders nothing within a slice, so this is exact, not ≈.
            Test.@test all(o_ser[t] == o_ref[t] for t in eachindex(sl_plans))
            Test.@test all(o_thr[t] == o_ref[t] for t in eachindex(sl_plans))
            # Cost proxy drives the longest-first schedule, so it must track the point counts.
            Test.@test CGEF.Filtering.slice_costs(sl_plans) == counts
            # Mismatched collection lengths are a caller error, not something to silently truncate.
            Test.@test_throws DimensionMismatch CGEF.Filtering.filter_slices!(
                o_ser[1:3], sl_fields, sl_plans)
        end

        # A node set is point-parallel, so threading it is supported and bit-identical to serial.
        ug = honesty_grids[end]
        Test.@test CGEF.Filtering._backend_supported(ug, CGEF.ComputationalBackends.ThreadedBackend())
        uf = randn(nn * nn)
        os = zeros(nn * nn); ot = zeros(nn * nn)
        CGEF.Filtering.filter_apply!(os, uf, CGEF.Filtering.plan_filter(
            ug, CGEF.GaussianKernel(), 5000.0;
            backend = CGEF.ComputationalBackends.SerialBackend(), method = CGEF.Filtering.RealSpace()))
        CGEF.Filtering.filter_apply!(ot, uf, CGEF.Filtering.plan_filter(
            ug, CGEF.GaussianKernel(), 5000.0;
            backend = CGEF.ComputationalBackends.ThreadedBackend(), method = CGEF.Filtering.RealSpace()))
        Test.@test os == ot

        # Every point-indexed grid (1D, true-3D, node) against every parallel backend, both kernels and
        # both mask strategies, asserted BIT-identical to serial. These paths decompose over linear
        # indices rather than rows; the assertion is `==` because none of them reorders a reduction.
        let dx = 1000.0,
            x1 = 0.0:dx:dx*39,
            x3 = 0.0:dx:dx*9,
            m3 = (mm = trues(10, 10, 10); mm[3:4, 3:4, 3:4] .= false; mm),
            pt_grids = (
                (FG.Grids.StructuredGrid(cart, x1), (40,)),
                (FG.Grids.StructuredGrid(cart, collect(x1) .+ [iseven(i) ? 3.0 : -2.0 for i in 1:40]), (40,)),
                (FG.Grids.StructuredGrid(cart, x3, x3, x3), (10, 10, 10)),
                (FG.Grids.StructuredGrid(cart, x3, x3, x3, m3), (10, 10, 10)),
            ),
            par_backends = (
                CGEF.ComputationalBackends.ThreadedBackend(),
                CGEF.ComputationalBackends.DistributedBackend(),
                CGEF.ComputationalBackends.MPIBackend(),
                CGEF.ComputationalBackends.GPUBackend(KA.CPU()),
            )
            for (pg, sz) in pt_grids
                pu = randn(sz...)
                for kern in (CGEF.TopHatKernel(), CGEF.GaussianKernel()),
                    strat in (CGEF.Filtering.Deformable(), CGEF.Filtering.ZeroFill())
                    ref = zeros(sz...)
                    CGEF.Filtering.filter_apply!(ref, pu, CGEF.Filtering.plan_filter(
                        pg, kern, 4000.0;
                        backend = CGEF.ComputationalBackends.SerialBackend(), mask_strategy = strat))
                    for b in par_backends
                        Test.@test CGEF.Filtering._backend_supported(pg, b)
                        got = zeros(sz...)
                        CGEF.Filtering.filter_apply!(got, pu, CGEF.Filtering.plan_filter(
                            pg, kern, 4000.0; backend = b, mask_strategy = strat))
                        Test.@test got == ref
                    end
                end
            end
        end

        # Node sets also have a device path: the footprint is a CSR adjacency, so the kernel is a flat
        # per-node gather with no row decomposition. Verified on `KA.CPU()`, which needs no hardware.
        Test.@test CGEF.Filtering._backend_supported(ug, CGEF.ComputationalBackends.GPUBackend(KA.CPU()))
        for strat in (CGEF.Filtering.Deformable(), CGEF.Filtering.ZeroFill())
            p_ser = CGEF.Filtering.plan_filter(ug, CGEF.GaussianKernel(), 5000.0;
                backend = CGEF.ComputationalBackends.SerialBackend(),
                method = CGEF.Filtering.RealSpace(), mask_strategy = strat)
            p_gpu = CGEF.Filtering.plan_filter(ug, CGEF.GaussianKernel(), 5000.0;
                backend = CGEF.ComputationalBackends.GPUBackend(KA.CPU()),
                method = CGEF.Filtering.RealSpace(), mask_strategy = strat)
            g_ser = zeros(nn * nn); g_gpu = zeros(nn * nn)
            CGEF.Filtering.filter_apply!(g_ser, uf, p_ser)
            CGEF.Filtering.filter_apply!(g_gpu, uf, p_gpu)
            Test.@test g_ser == g_gpu
        end
    end

    geom = FG.Geometry.CartesianGeometry()
    x = collect(0.0:1000.0:20e3)
    y = collect(0.0:1000.0:20e3)
    grid = FG.Grids.StructuredGrid(geom, x, y, trues(length(x), length(y)))
    field = rand(length(x), length(y))
    out_serial = zeros(size(field))
    out_default = zeros(size(field))
    CGEF.Filtering.filter_field!(out_serial, field, grid, CGEF.TopHatKernel(), 5000.0; backend = CGEF.ComputationalBackends.SerialBackend())
    CGEF.Filtering.filter_field!(out_default, field, grid, CGEF.TopHatKernel(), 5000.0)  # AutoBackend
    Test.@test out_serial ≈ out_default

    # MPI extension IS loaded here (see the top-level `MPI.Init()`) — with a single rank, every
    # row is owned by rank 0, and `Allreduce!` over one rank is a no-op sum, so this must match
    # the serial reference exactly, the same way the "Distributed backend" testset below already
    # exercises the real `DistributedBackend` rather than expecting an error.
    out_mpi = zeros(size(field))
    CGEF.Filtering.filter_field!(out_mpi, field, grid, CGEF.TopHatKernel(), 5000.0; backend = CGEF.ComputationalBackends.MPIBackend())
    Test.@test out_mpi ≈ out_serial

    # SeparableGaussianFootprint's own native two-phase MPI implementation (redundant
    # zero-communication row-pass + round-robin column-pass + Allreduce) — a single rank still
    # exercises the real code path (not a serial fallback), unlike the plain TopHat case above
    # which never reaches the separable-Gaussian branch at all.
    grid_range = FG.Grids.StructuredGrid(geom, 0.0:1000.0:20e3, 0.0:1000.0:20e3, trues(length(x), length(y)))
    gplan_check = CGEF.Filtering.build_footprint(grid_range, CGEF.GaussianKernel(), 5000.0)
    Test.@test gplan_check isa CGEF.Filtering.SeparableGaussianFootprint
    mask_g = trues(length(x), length(y)); mask_g[5:8, 5:8] .= false
    grid_g = FG.Grids.StructuredGrid(geom, 0.0:1000.0:20e3, 0.0:1000.0:20e3, mask_g)
    field_g = rand(length(x), length(y))
    for strat in (CGEF.Filtering.Deformable(), CGEF.Filtering.ZeroFill())
        out_g_serial = zeros(size(field_g)); out_g_mpi = zeros(size(field_g))
        CGEF.Filtering.filter_field!(out_g_serial, field_g, grid_g, CGEF.GaussianKernel(), 5000.0; backend = CGEF.ComputationalBackends.SerialBackend(), mask_strategy = strat)
        CGEF.Filtering.filter_field!(out_g_mpi, field_g, grid_g, CGEF.GaussianKernel(), 5000.0; backend = CGEF.ComputationalBackends.MPIBackend(), mask_strategy = strat)
        Test.@test out_g_mpi ≈ out_g_serial
    end
end


# Threaded must agree with serial exactly, including masking and periodic wrapping — it shares the
# footprint engine, so any disagreement means a hand-rolled parallel path has diverged from it.
Test.@testset "Threaded backend (OhMyThreads)" begin
    geom = FG.Geometry.CartesianGeometry()
    x = collect(0.0:1000.0:30e3)
    y = collect(0.0:1000.0:30e3)
    u = rand(length(x), length(y))

    grid = FG.Grids.StructuredGrid(geom, x, y, trues(length(x), length(y)))
    for strat in (CGEF.Filtering.Deformable(), CGEF.Filtering.ZeroFill())
        os = zeros(size(u)); ot = zeros(size(u))
        CGEF.Filtering.filter_field!(os, u, grid, CGEF.TopHatKernel(), 5000.0; backend = CGEF.ComputationalBackends.SerialBackend(), mask_strategy = strat)
        CGEF.Filtering.filter_field!(ot, u, grid, CGEF.TopHatKernel(), 5000.0; backend = CGEF.ComputationalBackends.ThreadedBackend(), mask_strategy = strat)
        Test.@test ot ≈ os
    end

    # Masked Cartesian
    mask = trues(length(x), length(y)); mask[5:8, 5:8] .= false
    mgrid = FG.Grids.StructuredGrid(geom, x, y, mask)
    os = zeros(size(u)); ot = zeros(size(u))
    CGEF.Filtering.filter_field!(os, u, mgrid, CGEF.GaussianKernel(), 4000.0; backend = CGEF.ComputationalBackends.SerialBackend())
    CGEF.Filtering.filter_field!(ot, u, mgrid, CGEF.GaussianKernel(), 4000.0; backend = CGEF.ComputationalBackends.ThreadedBackend())
    Test.@test ot ≈ os

    # Periodic global spherical grid (threaded must wrap exactly like serial)
    sgeom = FG.Geometry.SphericalGeometry(6371000.0)
    slon = deg2rad.(collect(0.0:5.0:355.0))
    slat = deg2rad.(collect(-40.0:5.0:40.0))
    sgrid = FG.Grids.StructuredGrid(sgeom, slon, slat, trues(length(slon), length(slat)))
    su = rand(length(slon), length(slat))
    oss = zeros(size(su)); ost = zeros(size(su))
    CGEF.Filtering.filter_field!(oss, su, sgrid, CGEF.TopHatKernel(), deg2rad(15.0) * 6371000.0; backend = CGEF.ComputationalBackends.SerialBackend())
    CGEF.Filtering.filter_field!(ost, su, sgrid, CGEF.TopHatKernel(), deg2rad(15.0) * 6371000.0; backend = CGEF.ComputationalBackends.ThreadedBackend())
    Test.@test ost ≈ oss
end


# ThreadedBackend for 1D/true-3D StructuredGrid: these use the point-indexed FilterFootprintND/
# NDScatteredFilterPlan representation (no row structure), but the per-point kernel is
# data-race-free (reads neighbours, writes only its own cell) — verify it matches serial exactly,
# covering both the fast (Range-axis, translation-invariant) and general (nonuniform/spherical
# scattered) footprint paths. Distributed/GPU/MPI remain unsupported here (still row-only) and
# must raise a clear error when requested explicitly, per `_check_backend_compatible`.
Test.@testset "Threaded backend: 1D/true-3D StructuredGrid (ND footprint)" begin
    # 1D Cartesian, uniform (Range) axis -> fast FilterFootprintND path.
    geom1 = FG.Geometry.CartesianGeometry()
    x1 = collect(0.0:1000.0:30e3)
    grid1 = FG.Grids.StructuredGrid(geom1, x1, trues(length(x1)))
    u1 = rand(length(x1))
    os1 = zeros(size(u1)); ot1 = zeros(size(u1))
    CGEF.Filtering.filter_field!(os1, u1, grid1, CGEF.TopHatKernel(), 5000.0; backend = CGEF.ComputationalBackends.SerialBackend())
    CGEF.Filtering.filter_field!(ot1, u1, grid1, CGEF.TopHatKernel(), 5000.0; backend = CGEF.ComputationalBackends.ThreadedBackend())
    Test.@test ot1 ≈ os1

    # True-3D Cartesian, uniform axes -> fast FilterFootprintND path, with a mask.
    geom3 = FG.Geometry.CartesianGeometry()
    x3 = collect(0.0:1000.0:10e3); y3 = collect(0.0:1000.0:10e3); z3 = collect(0.0:500.0:4e3)
    mask3 = trues(length(x3), length(y3), length(z3)); mask3[3:5, 3:5, 2] .= false
    grid3 = FG.Grids.StructuredGrid(geom3, x3, y3, z3, mask3)
    u3 = rand(length(x3), length(y3), length(z3))
    os3 = zeros(size(u3)); ot3 = zeros(size(u3))
    CGEF.Filtering.filter_field!(os3, u3, grid3, CGEF.GaussianKernel(), 2000.0; backend = CGEF.ComputationalBackends.SerialBackend())
    CGEF.Filtering.filter_field!(ot3, u3, grid3, CGEF.GaussianKernel(), 2000.0; backend = CGEF.ComputationalBackends.ThreadedBackend())
    Test.@test ot3 ≈ os3

    # True-3D SPHERICAL (general/scattered footprint path — spherical never uses the translation-
    # invariant fast path, see `build_footprint`'s Cartesian-only fast dispatch).
    R = 6.371e6
    sgeom = FG.Geometry.SphericalGeometry(R)
    slon = deg2rad.(collect(0.0:20.0:340.0))
    slat = deg2rad.(collect(-40.0:20.0:40.0))
    sr = collect(R:100e3:(R + 300e3))
    sgrid = FG.Grids.StructuredGrid(sgeom, slon, slat, sr, trues(length(slon), length(slat), length(sr)))
    su3 = rand(length(slon), length(slat), length(sr))
    oss3 = zeros(size(su3)); ost3 = zeros(size(su3))
    CGEF.Filtering.filter_field!(oss3, su3, sgrid, CGEF.TopHatKernel(), 150e3; backend = CGEF.ComputationalBackends.SerialBackend())
    CGEF.Filtering.filter_field!(ost3, su3, sgrid, CGEF.TopHatKernel(), 150e3; backend = CGEF.ComputationalBackends.ThreadedBackend())
    Test.@test ost3 ≈ oss3

    # Distributed now has an ND hook: it decomposes over linear indices into a SharedArray instead of
    # over rows, so a 1D grid is honored rather than refused, and agrees with serial exactly.
    osd1 = zeros(size(u1))
    CGEF.Filtering.filter_field!(osd1, u1, grid1, CGEF.TopHatKernel(), 5000.0;
        backend = CGEF.ComputationalBackends.DistributedBackend())
    Test.@test osd1 == os1
end


# Distributed backend must also agree with serial (footprint engine + SharedArray fill). With no
# extra worker processes the @distributed loop runs serially on the caller — still exact.
Test.@testset "Distributed backend" begin
    geom = FG.Geometry.CartesianGeometry()
    x = collect(0.0:1000.0:30e3)
    y = collect(0.0:1000.0:30e3)
    mask = trues(length(x), length(y)); mask[6:9, 6:9] .= false
    grid = FG.Grids.StructuredGrid(geom, x, y, mask)
    u = rand(length(x), length(y))
    os = zeros(size(u)); od = zeros(size(u))
    CGEF.Filtering.filter_field!(os, u, grid, CGEF.TopHatKernel(), 5000.0; backend = CGEF.ComputationalBackends.SerialBackend())
    CGEF.Filtering.filter_field!(od, u, grid, CGEF.TopHatKernel(), 5000.0; backend = CGEF.ComputationalBackends.DistributedBackend())
    Test.@test od ≈ os

    # SeparableGaussianFootprint's own native two-phase Distributed implementation
    # (SharedArray-backed row-pass then column-pass, zero extra communication).
    gplan_check = CGEF.Filtering.build_footprint(FG.Grids.StructuredGrid(geom, 0.0:1000.0:30e3, 0.0:1000.0:30e3, trues(length(x), length(y))), CGEF.GaussianKernel(), 5000.0)
    Test.@test gplan_check isa CGEF.Filtering.SeparableGaussianFootprint
    for strat in (CGEF.Filtering.Deformable(), CGEF.Filtering.ZeroFill())
        os_g = zeros(size(u)); od_g = zeros(size(u))
        CGEF.Filtering.filter_field!(os_g, u, grid, CGEF.GaussianKernel(), 5000.0; backend = CGEF.ComputationalBackends.SerialBackend(), mask_strategy = strat)
        CGEF.Filtering.filter_field!(od_g, u, grid, CGEF.GaussianKernel(), 5000.0; backend = CGEF.ComputationalBackends.DistributedBackend(), mask_strategy = strat)
        Test.@test od_g ≈ os_g
    end
end


# GPU backend on the KernelAbstractions CPU device must match serial (validates the GPU kernel
# logic here; actual GPU hardware is exercised separately). Same engine ⇒ masking + periodicity
# consistent.
Test.@testset "GPU backend (KernelAbstractions CPU)" begin
    gpu = CGEF.ComputationalBackends.GPUBackend(KA.CPU())
    geom = FG.Geometry.CartesianGeometry()
    x = collect(0.0:1000.0:20e3)
    y = collect(0.0:1000.0:20e3)
    mask = trues(length(x), length(y)); mask[5:7, 5:7] .= false
    grid = FG.Grids.StructuredGrid(geom, x, y, mask)
    u = rand(length(x), length(y))
    for strat in (CGEF.Filtering.Deformable(), CGEF.Filtering.ZeroFill())
        os = zeros(size(u)); og = zeros(size(u))
        CGEF.Filtering.filter_field!(os, u, grid, CGEF.TopHatKernel(), 5000.0; backend = CGEF.ComputationalBackends.SerialBackend(), mask_strategy = strat)
        CGEF.Filtering.filter_field!(og, u, grid, CGEF.TopHatKernel(), 5000.0; backend = gpu, mask_strategy = strat)
        Test.@test og ≈ os
    end

    # SeparableGaussianFootprint's own native two-phase GPU implementation (row-pass kernel,
    # then column-pass kernel, with a device-resident intermediate buffer between them).
    gplan_check = CGEF.Filtering.build_footprint(FG.Grids.StructuredGrid(geom, 0.0:1000.0:20e3, 0.0:1000.0:20e3, trues(length(x), length(y))), CGEF.GaussianKernel(), 5000.0)
    Test.@test gplan_check isa CGEF.Filtering.SeparableGaussianFootprint
    for strat in (CGEF.Filtering.Deformable(), CGEF.Filtering.ZeroFill())
        os_g = zeros(size(u)); og_g = zeros(size(u))
        CGEF.Filtering.filter_field!(os_g, u, grid, CGEF.GaussianKernel(), 5000.0; backend = CGEF.ComputationalBackends.SerialBackend(), mask_strategy = strat)
        CGEF.Filtering.filter_field!(og_g, u, grid, CGEF.GaussianKernel(), 5000.0; backend = gpu, mask_strategy = strat)
        Test.@test og_g ≈ os_g
    end

    # Periodic global spherical grid
    sgeom = FG.Geometry.SphericalGeometry(6371000.0)
    slon = deg2rad.(collect(0.0:5.0:355.0))
    slat = deg2rad.(collect(-30.0:5.0:30.0))
    sgrid = FG.Grids.StructuredGrid(sgeom, slon, slat, trues(length(slon), length(slat)))
    su = rand(length(slon), length(slat))
    oss = zeros(size(su)); osg = zeros(size(su))
    CGEF.Filtering.filter_field!(oss, su, sgrid, CGEF.TopHatKernel(), deg2rad(15.0) * 6371000.0; backend = CGEF.ComputationalBackends.SerialBackend())
    CGEF.Filtering.filter_field!(osg, su, sgrid, CGEF.TopHatKernel(), deg2rad(15.0) * 6371000.0; backend = gpu)
    Test.@test osg ≈ oss

    # ScatteredFilterPlan, both cached and streaming, on each grid architecture that reaches it. The
    # streaming kernel recomputes weights from the grid's cell measure, which is uploaded factored on
    # a rectilinear grid and dense on a curvilinear one — two different device representations of the
    # same numbers, so both need checking against serial.
    x_nu = collect(0.0:1000.0:20e3) .+ [iseven(i) ? 40.0 : -30.0 for i in 1:21]
    grid_nu = FG.Grids.StructuredGrid(geom, x_nu, x_nu, mask)
    Nc = 16
    clon = [1000.0 * (ii - 1) for ii in 1:Nc, jj in 1:Nc]
    clat = [1000.0 * (jj - 1) for ii in 1:Nc, jj in 1:Nc]
    cgrid = FG.Grids.CurvilinearGrid(geom, clon, clat, trues(Nc, Nc))
    scattered_cases = (
        (grid_nu, CGEF.SharpSpectralKernel(), 5000.0, rand(size(grid_nu.mask)...)),
        (sgrid, CGEF.GaussianKernel(), deg2rad(15.0) * 6371000.0, su),
        (cgrid, CGEF.TopHatKernel(), 3000.0, rand(Nc, Nc)),
    )
    for (g, ker, scale, f) in scattered_cases, cs in (CGEF.Filtering.AlwaysCache(), CGEF.Filtering.NeverCache())
        fp = CGEF.Filtering.build_footprint(g, ker, scale; cache_strategy = cs)
        Test.@test fp isa CGEF.Filtering.ScatteredFilterPlan
        o_s = zeros(size(f)); o_g = zeros(size(f))
        ps = CGEF.Filtering.plan_filter(g, ker, scale; backend = CGEF.ComputationalBackends.SerialBackend(), cache_strategy = cs)
        pg = CGEF.Filtering.plan_filter(g, ker, scale; backend = gpu, cache_strategy = cs)
        CGEF.Filtering.filter_apply!(o_s, f, ps)
        CGEF.Filtering.filter_apply!(o_g, f, pg)
        Test.@test o_g ≈ o_s
        # A plan uploads its device buffers once; applying it again must still give the same answer.
        fill!(o_g, 0.0)
        CGEF.Filtering.filter_apply!(o_g, f, pg)
        Test.@test o_g ≈ o_s
    end

end
