
# Execution backend lattice (ComputationalBackends.jl)
Test.@testset "Backends" begin
    # resolve_backend returns INSTANCES; AutoBackend picks a concrete local backend
    Test.@test CGEF.ComputationalBackends.resolve_backend(CGEF.ComputationalBackends.SerialBackend()) === CGEF.ComputationalBackends.SerialBackend()
    Test.@test CGEF.ComputationalBackends.resolve_backend(CGEF.ComputationalBackends.AutoBackend()) isa Union{CGEF.ComputationalBackends.SerialBackend, CGEF.ComputationalBackends.ThreadedBackend}

    # distribution wrappers are parametric over an inner local backend
    Test.@test CGEF.ComputationalBackends.DistributedBackend(CGEF.ComputationalBackends.SerialBackend()) isa CGEF.ComputationalBackends.DistributedBackend
    Test.@test CGEF.ComputationalBackends.MPIBackend() isa CGEF.ComputationalBackends.MPIBackend
    Test.@test CGEF.ComputationalBackends.DistributedBackend().inner === CGEF.ComputationalBackends.SerialBackend()
    Test.@test CGEF.ComputationalBackends.local_backend(CGEF.ComputationalBackends.DistributedBackend(CGEF.ComputationalBackends.ThreadedBackend())) === CGEF.ComputationalBackends.ThreadedBackend()
    Test.@test CGEF.ComputationalBackends.is_distributed(CGEF.ComputationalBackends.MPIBackend()) && !CGEF.ComputationalBackends.is_distributed(CGEF.ComputationalBackends.SerialBackend())

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

    # Distributed/GPU/MPI still have no ND hook -- an explicit request must error, not silently
    # downgrade to serial.
    Test.@test_throws ArgumentError CGEF.Filtering.filter_field!(os1, u1, grid1, CGEF.TopHatKernel(), 5000.0; backend = CGEF.ComputationalBackends.DistributedBackend())
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
