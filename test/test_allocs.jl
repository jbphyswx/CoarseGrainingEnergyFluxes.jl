#=
Zero-/bounded-allocation regression tests for the package's core hot-path methods.

Every method below is measured on a WARMED-UP call (at least two calls before the `@allocated` one,
so the number reflects real runtime behavior, not one-time JIT compilation — a cold-call measurement
is a different, much larger number and is not what a caller cares about at all). No cherry-picking:
this covers every core in-place (`!`) numerical kernel, every grid-type/dimensionality dispatch of
`compute_Π!`, every spectral backend, every parallel backend that has a `filter_apply!` hook, and the
`coarse_grain!`/`cumulative_energy!` pipeline entry points — not a handful of methods picked because
they were already known to be allocation-free.

Two genuinely different classes of assertion appear here, and the comment on each testset says which:

1. **Exact zero** — the true numerical kernels (`filter_apply!` on a cached plan, `ddx!`/`ddy!`/`ddz!`
   with a cached derivative plan). These have no legitimate reason to allocate at all once warmed up,
   and are asserted with `Test.@test a == 0`.
2. **Bounded, documented, non-zero** — a small, understood, non-scaling cost from one of two sources:
   (a) an outer wrapper accepting an abstract-typed optional keyword (`workspace::Union{Nothing,
   ΠWorkspace}`, `filter_plan::Union{Nothing, Filtering.AbstractFilterPlan}`) pays a fixed dynamic-
   dispatch cost per call that doesn't scale with problem size — this is NOT a rebuild, confirmed by
   measuring the underlying kernel directly at 0 bytes; or (b) a backend with a real, already-documented
   per-call cost (`GPUBackend`'s device buffer upload — see docs/src/architecture.md; the `OhMyThreads`
   scheduler's own task-spawn bookkeeping) or an upstream dependency's own allocation behavior (NUFSHT.jl's
   `nusht_filter!`, confirmed via a direct, isolated measurement of that exact call — not something this
   package's own extension code does; not something fixable from this repository). Bounds use a
   generous-but-real safety margin over the measured value so minor Julia-version/CPU noise doesn't
   make this flaky, while still catching an actual regression (e.g. a reintroduced full footprint
   rebuild, which is orders of magnitude larger than any bound here).

`filter_plan`/`filter_plans`/`deriv_plan`/`workspace` reuse is exactly what makes a caller's REPEATED
sweep (many timesteps over the same grid/scales) genuinely zero-(re)allocation — the whole point of
these tests is to catch a regression back to a per-call footprint/plan rebuild, which is real,
previously-shipped, silently-wasteful behavior this session found and fixed (see CHANGELOG.md).
=#

using Test: Test

Test.@testset "Zero-/bounded-allocation hot paths" begin

    # -----------------------------------------------------------------------
    # Real-space (DirectSum) filter_apply! on a cached plan — exact zero
    # -----------------------------------------------------------------------
    Test.@testset "filter_apply! (real-space, cached plan): exact zero" begin
        ker = CGEF.TopHatKernel()

        # StructuredGrid 2D Cartesian (uniform Range axes -> fast translation-invariant path)
        N = 48; dx = 1000.0
        geom = CGEF.CartesianGeometry(dx, dx)
        xsR = 0.0:dx:(N - 1) * dx
        grid2d = CGEF.StructuredGrid(geom, xsR, xsR, trues(N, N))
        plan2d = CGEF.Filtering.plan_filter(grid2d, ker, 5000.0)
        u2d = randn(N, N); out2d = zeros(N, N)
        CGEF.Filtering.filter_apply!(out2d, u2d, plan2d)
        CGEF.Filtering.filter_apply!(out2d, u2d, plan2d)
        Test.@test (@allocated CGEF.Filtering.filter_apply!(out2d, u2d, plan2d)) == 0

        # StructuredGrid 2D Spherical (per-latitude-band fast path)
        R = 6.371e6
        lonR = deg2rad.(0.0:4.0:356.0); latR = deg2rad.(-80.0:4.0:80.0)
        grids2d = CGEF.StructuredGrid(CGEF.SphericalGeometry(R), lonR, latR, trues(length(lonR), length(latR)))
        plans2d = CGEF.Filtering.plan_filter(grids2d, ker, 400e3)
        us2d = randn(length(lonR), length(latR)); outs2d = zeros(length(lonR), length(latR))
        CGEF.Filtering.filter_apply!(outs2d, us2d, plans2d)
        CGEF.Filtering.filter_apply!(outs2d, us2d, plans2d)
        Test.@test (@allocated CGEF.Filtering.filter_apply!(outs2d, us2d, plans2d)) == 0

        # StructuredGrid 1D Cartesian
        grid1d = CGEF.StructuredGrid(geom, xsR, trues(N))
        plan1d = CGEF.Filtering.plan_filter(grid1d, ker, 5000.0)
        u1d = randn(N); out1d = zeros(N)
        CGEF.Filtering.filter_apply!(out1d, u1d, plan1d)
        CGEF.Filtering.filter_apply!(out1d, u1d, plan1d)
        Test.@test (@allocated CGEF.Filtering.filter_apply!(out1d, u1d, plan1d)) == 0

        # StructuredGrid true-3D Cartesian
        N3 = 16
        geom3 = CGEF.CartesianGeometry(dx, dx, dx)
        xs3R = 0.0:dx:(N3 - 1) * dx
        grid3d = CGEF.StructuredGrid(geom3, xs3R, xs3R, xs3R, trues(N3, N3, N3))
        plan3d = CGEF.Filtering.plan_filter(grid3d, ker, 2500.0)
        u3d = randn(N3, N3, N3); out3d = zeros(N3, N3, N3)
        CGEF.Filtering.filter_apply!(out3d, u3d, plan3d)
        CGEF.Filtering.filter_apply!(out3d, u3d, plan3d)
        Test.@test (@allocated CGEF.Filtering.filter_apply!(out3d, u3d, plan3d)) == 0

        # CurvilinearGrid (per-point scattered footprint, no translation invariance)
        Nc = 40
        i = collect(0.0:(Nc - 1)); j = collect(0.0:(Nc - 1))
        θ = deg2rad(15.0); shear = 0.3
        clon = [dx * (ii * cos(θ) - jj * shear * sin(θ)) for ii in i, jj in j]
        clat = [dx * (ii * sin(θ) + jj * (1 + shear * cos(θ))) for ii in i, jj in j]
        cgrid = CGEF.CurvilinearGrid(geom, clon, clat, trues(Nc, Nc))
        cplan = CGEF.Filtering.plan_filter(cgrid, ker, 8000.0)
        uc = randn(Nc, Nc); outc = zeros(Nc, Nc)
        CGEF.Filtering.filter_apply!(outc, uc, cplan)
        CGEF.Filtering.filter_apply!(outc, uc, cplan)
        Test.@test (@allocated CGEF.Filtering.filter_apply!(outc, uc, cplan)) == 0
    end

    # -----------------------------------------------------------------------
    # Derivatives ddx!/ddy!/ddz! — exact zero
    # -----------------------------------------------------------------------
    Test.@testset "ddx!/ddy!/ddz!: exact zero" begin
        N = 48; dx = 1000.0
        geom = CGEF.CartesianGeometry(dx, dx)
        xsR = 0.0:dx:(N - 1) * dx
        grid2d = CGEF.StructuredGrid(geom, xsR, xsR, trues(N, N))
        u2d = randn(N, N); out2d = zeros(N, N)
        CGEF.Derivatives.ddx!(out2d, u2d, grid2d); CGEF.Derivatives.ddx!(out2d, u2d, grid2d)
        Test.@test (@allocated CGEF.Derivatives.ddx!(out2d, u2d, grid2d)) == 0
        CGEF.Derivatives.ddy!(out2d, u2d, grid2d); CGEF.Derivatives.ddy!(out2d, u2d, grid2d)
        Test.@test (@allocated CGEF.Derivatives.ddy!(out2d, u2d, grid2d)) == 0

        R = 6.371e6
        lonR = deg2rad.(0.0:4.0:356.0); latR = deg2rad.(-80.0:4.0:80.0)
        grids2d = CGEF.StructuredGrid(CGEF.SphericalGeometry(R), lonR, latR, trues(length(lonR), length(latR)))
        us2d = randn(length(lonR), length(latR)); outs2d = zeros(length(lonR), length(latR))
        CGEF.Derivatives.ddx!(outs2d, us2d, grids2d); CGEF.Derivatives.ddx!(outs2d, us2d, grids2d)
        Test.@test (@allocated CGEF.Derivatives.ddx!(outs2d, us2d, grids2d)) == 0
        CGEF.Derivatives.ddy!(outs2d, us2d, grids2d); CGEF.Derivatives.ddy!(outs2d, us2d, grids2d)
        Test.@test (@allocated CGEF.Derivatives.ddy!(outs2d, us2d, grids2d)) == 0

        grid1d = CGEF.StructuredGrid(geom, xsR, trues(N))
        u1d = randn(N); out1d = zeros(N)
        CGEF.Derivatives.ddx!(out1d, u1d, grid1d); CGEF.Derivatives.ddx!(out1d, u1d, grid1d)
        Test.@test (@allocated CGEF.Derivatives.ddx!(out1d, u1d, grid1d)) == 0

        N3 = 16
        geom3 = CGEF.CartesianGeometry(dx, dx, dx)
        xs3R = 0.0:dx:(N3 - 1) * dx
        grid3d = CGEF.StructuredGrid(geom3, xs3R, xs3R, xs3R, trues(N3, N3, N3))
        u3d = randn(N3, N3, N3); out3d = zeros(N3, N3, N3)
        CGEF.Derivatives.ddx!(out3d, u3d, grid3d); CGEF.Derivatives.ddx!(out3d, u3d, grid3d)
        Test.@test (@allocated CGEF.Derivatives.ddx!(out3d, u3d, grid3d)) == 0
        CGEF.Derivatives.ddz!(out3d, u3d, grid3d); CGEF.Derivatives.ddz!(out3d, u3d, grid3d)
        Test.@test (@allocated CGEF.Derivatives.ddz!(out3d, u3d, grid3d)) == 0

        # CurvilinearGrid: cached WLSQGradientPlan
        Nc = 40
        i = collect(0.0:(Nc - 1)); j = collect(0.0:(Nc - 1))
        θ = deg2rad(15.0); shear = 0.3
        clon = [dx * (ii * cos(θ) - jj * shear * sin(θ)) for ii in i, jj in j]
        clat = [dx * (ii * sin(θ) + jj * (1 + shear * cos(θ))) for ii in i, jj in j]
        cgrid = CGEF.CurvilinearGrid(geom, clon, clat, trues(Nc, Nc))
        cdplan = CGEF.Derivatives.WLSQGradientPlan(cgrid)
        uc = randn(Nc, Nc); outc = zeros(Nc, Nc)
        CGEF.Derivatives.ddx!(outc, uc, cgrid, cdplan); CGEF.Derivatives.ddx!(outc, uc, cgrid, cdplan)
        Test.@test (@allocated CGEF.Derivatives.ddx!(outc, uc, cgrid, cdplan)) == 0

        # UnstructuredGrid: cached UnstructuredWLSQGradientPlan (k-d tree adjacency)
        npts = 300
        ugeom = CGEF.CartesianGeometry(1.0, 1.0)
        ulon = 60e3 .* rand(npts); ulat = 60e3 .* rand(npts)
        ugrid = CGEF.UnstructuredGrid(ugeom, ulon, ulat, trues(npts); k = 8)
        udplan = CGEF.Derivatives.WLSQGradientPlan(ugrid)
        uu = randn(npts); outu = zeros(npts)
        CGEF.Derivatives.ddx!(outu, uu, ugrid, udplan); CGEF.Derivatives.ddx!(outu, uu, ugrid, udplan)
        Test.@test (@allocated CGEF.Derivatives.ddx!(outu, uu, ugrid, udplan)) == 0
    end

    # -----------------------------------------------------------------------
    # Spectral filter_apply! — exact zero for FFTW/FINUFFT/FastSphericalHarmonics; NUFSHT is an
    # upstream (NUFSHT.jl) allocation, not something this package's extension code does.
    # -----------------------------------------------------------------------
    Test.@testset "filter_apply! (spectral, cached plan)" begin
        N = 48; dx = 1000.0
        geom = CGEF.CartesianGeometry(dx, dx)
        xsR = 0.0:dx:(N - 1) * dx
        gridp = CGEF.StructuredGrid(geom, xsR, xsR, trues(N, N); periodic = (true, true))
        u = randn(N, N); out = zeros(N, N)
        fftplan = CGEF.Filtering.plan_filter(gridp, CGEF.GaussianKernel(), 5000.0; method = CGEF.Filtering.Spectral())
        CGEF.Filtering.filter_apply!(out, u, fftplan); CGEF.Filtering.filter_apply!(out, u, fftplan)
        Test.@test (@allocated CGEF.Filtering.filter_apply!(out, u, fftplan)) == 0

        npts = 400
        ugeom = CGEF.CartesianGeometry(1.0, 1.0)
        ulon = 60e3 .* rand(npts); ulat = 60e3 .* rand(npts)
        ugrid = CGEF.UnstructuredGrid(ugeom, ulon, ulat, ones(npts), trues(npts))
        uf = randn(npts); outu = zeros(npts)
        finufftplan = CGEF.Filtering.plan_filter(ugrid, CGEF.GaussianKernel(), 5000.0; method = CGEF.Filtering.Spectral())
        CGEF.Filtering.filter_apply!(outu, uf, finufftplan); CGEF.Filtering.filter_apply!(outu, uf, finufftplan)
        Test.@test (@allocated CGEF.Filtering.filter_apply!(outu, uf, finufftplan)) == 0

        Ndeg = 16; Nsh = Ndeg + 1; Msh = 2Nsh - 1
        Θ, Φ = FSH.sph_points(Nsh)
        R = 6.371e6
        sgrid = CGEF.StructuredGrid(CGEF.SphericalGeometry(R), collect(Φ), π / 2 .- collect(Θ), trues(Msh, Nsh))
        field = randn(Msh, Nsh); outsh = zeros(Msh, Nsh)
        shtplan = CGEF.Filtering.plan_filter(sgrid, CGEF.GaussianKernel(), π * R / 8; method = CGEF.Filtering.Spectral())
        CGEF.Filtering.filter_apply!(outsh, field, shtplan); CGEF.Filtering.filter_apply!(outsh, field, shtplan)
        Test.@test (@allocated CGEF.Filtering.filter_apply!(outsh, field, shtplan)) == 0

        # NUFSHT: confirmed via a direct, isolated measurement of NUFSHT.nusht_filter! itself (not
        # through this package's extension code) that the allocation lives entirely inside NUFSHT.jl
        # — an unregistered, separately-maintained sibling package (github.com/jbphyswx/NUFSHT.jl),
        # not something fixable from this repository. Bounded (not zero) so a regression here — e.g.
        # this extension accidentally adding ITS OWN allocation on top — is still caught.
        Mpts = 200
        sφ = 2π .* rand(Mpts); sθ = acos.(clamp.(1 .- 2 .* rand(Mpts), -1, 1))
        slat = π / 2 .- sθ
        nugrid = CGEF.UnstructuredGrid(CGEF.SphericalGeometry(R), sφ, slat, ones(Mpts), trues(Mpts))
        nuf = randn(Mpts); outnu = zeros(Mpts)
        nushtplan = CGEF.Filtering.plan_filter(nugrid, CGEF.GaussianKernel(), π * R / 8; method = CGEF.Filtering.Spectral())
        CGEF.Filtering.filter_apply!(outnu, nuf, nushtplan); CGEF.Filtering.filter_apply!(outnu, nuf, nushtplan)
        a_nufsht = @allocated CGEF.Filtering.filter_apply!(outnu, nuf, nushtplan)
        Test.@test a_nufsht < 300_000
    end

    # -----------------------------------------------------------------------
    # compute_Π! — bounded, small, non-scaling: a fixed per-call dynamic-dispatch cost from the
    # abstract-typed optional `workspace`/`filter_plan`/`deriv_plan` keywords (confirmed NOT a
    # footprint/buffer rebuild: the underlying `filter_apply!`/`ddx!`/`ddy!`/`ddz!` calls it makes are
    # independently verified at exact zero above; `_compute_Π_2d!` called directly with concretely-
    # typed arguments is also exact zero — the residual lives entirely in the outer generic wrapper).
    # -----------------------------------------------------------------------
    Test.@testset "compute_Π!: bounded, non-scaling residual (all grid types/dimensions)" begin
        ker = CGEF.TopHatKernel()
        N = 48; dx = 1000.0
        geom = CGEF.CartesianGeometry(dx, dx)
        xsR = 0.0:dx:(N - 1) * dx

        # 2D Cartesian
        grid2d = CGEF.StructuredGrid(geom, xsR, xsR, trues(N, N))
        plan2d = CGEF.Filtering.plan_filter(grid2d, ker, 5000.0)
        ws2d = CGEF.Diagnostics.ΠWorkspace(grid2d)
        u2d = randn(N, N); v2d = randn(N, N); Π2d = zeros(N, N)
        CGEF.Diagnostics.compute_Π!(Π2d, u2d, v2d, nothing, grid2d, ker, 5000.0; workspace = ws2d, filter_plan = plan2d)
        CGEF.Diagnostics.compute_Π!(Π2d, u2d, v2d, nothing, grid2d, ker, 5000.0; workspace = ws2d, filter_plan = plan2d)
        Test.@test (@allocated CGEF.Diagnostics.compute_Π!(Π2d, u2d, v2d, nothing, grid2d, ker, 5000.0; workspace = ws2d, filter_plan = plan2d)) < 2048

        # 2D Spherical
        R = 6.371e6
        lonR = deg2rad.(0.0:4.0:356.0); latR = deg2rad.(-80.0:4.0:80.0)
        grids2d = CGEF.StructuredGrid(CGEF.SphericalGeometry(R), lonR, latR, trues(length(lonR), length(latR)))
        plans2d = CGEF.Filtering.plan_filter(grids2d, ker, 400e3)
        wss2d = CGEF.Diagnostics.ΠWorkspace(grids2d)
        us2d = randn(length(lonR), length(latR)); vs2d = randn(length(lonR), length(latR)); Πs2d = zeros(length(lonR), length(latR))
        CGEF.Diagnostics.compute_Π!(Πs2d, us2d, vs2d, nothing, grids2d, ker, 400e3; workspace = wss2d, filter_plan = plans2d)
        CGEF.Diagnostics.compute_Π!(Πs2d, us2d, vs2d, nothing, grids2d, ker, 400e3; workspace = wss2d, filter_plan = plans2d)
        Test.@test (@allocated CGEF.Diagnostics.compute_Π!(Πs2d, us2d, vs2d, nothing, grids2d, ker, 400e3; workspace = wss2d, filter_plan = plans2d)) < 2048

        # 1D Cartesian
        grid1d = CGEF.StructuredGrid(geom, xsR, trues(N))
        plan1d = CGEF.Filtering.plan_filter(grid1d, ker, 5000.0)
        ws1d = CGEF.Diagnostics.ΠWorkspace(grid1d)
        u1d = randn(N); Π1d = zeros(N)
        CGEF.Diagnostics.compute_Π!(Π1d, u1d, grid1d, ker, 5000.0; workspace = ws1d, filter_plan = plan1d)
        CGEF.Diagnostics.compute_Π!(Π1d, u1d, grid1d, ker, 5000.0; workspace = ws1d, filter_plan = plan1d)
        Test.@test (@allocated CGEF.Diagnostics.compute_Π!(Π1d, u1d, grid1d, ker, 5000.0; workspace = ws1d, filter_plan = plan1d)) < 2048

        # true-3D Cartesian
        N3 = 16
        geom3 = CGEF.CartesianGeometry(dx, dx, dx)
        xs3R = 0.0:dx:(N3 - 1) * dx
        grid3d = CGEF.StructuredGrid(geom3, xs3R, xs3R, xs3R, trues(N3, N3, N3))
        plan3d = CGEF.Filtering.plan_filter(grid3d, ker, 2500.0)
        ws3d = CGEF.Diagnostics.ΠWorkspace(grid3d)
        u3d = randn(N3, N3, N3); v3d = randn(N3, N3, N3); w3d = randn(N3, N3, N3); Π3d = zeros(N3, N3, N3)
        CGEF.Diagnostics.compute_Π!(Π3d, u3d, v3d, w3d, grid3d, ker, 2500.0; workspace = ws3d, filter_plan = plan3d)
        CGEF.Diagnostics.compute_Π!(Π3d, u3d, v3d, w3d, grid3d, ker, 2500.0; workspace = ws3d, filter_plan = plan3d)
        Test.@test (@allocated CGEF.Diagnostics.compute_Π!(Π3d, u3d, v3d, w3d, grid3d, ker, 2500.0; workspace = ws3d, filter_plan = plan3d)) < 2048

        # CurvilinearGrid
        Nc = 40
        i = collect(0.0:(Nc - 1)); j = collect(0.0:(Nc - 1))
        θ = deg2rad(15.0); shear = 0.3
        clon = [dx * (ii * cos(θ) - jj * shear * sin(θ)) for ii in i, jj in j]
        clat = [dx * (ii * sin(θ) + jj * (1 + shear * cos(θ))) for ii in i, jj in j]
        cgrid = CGEF.CurvilinearGrid(geom, clon, clat, trues(Nc, Nc))
        cplan = CGEF.Filtering.plan_filter(cgrid, ker, 8000.0)
        cdplan = CGEF.Derivatives.WLSQGradientPlan(cgrid)
        wsc = CGEF.Diagnostics.ΠWorkspace(cgrid)
        uc = randn(Nc, Nc); vc = randn(Nc, Nc); Πc = zeros(Nc, Nc)
        CGEF.Diagnostics.compute_Π!(Πc, uc, vc, nothing, cgrid, ker, 8000.0; workspace = wsc, filter_plan = cplan, deriv_plan = cdplan)
        CGEF.Diagnostics.compute_Π!(Πc, uc, vc, nothing, cgrid, ker, 8000.0; workspace = wsc, filter_plan = cplan, deriv_plan = cdplan)
        Test.@test (@allocated CGEF.Diagnostics.compute_Π!(Πc, uc, vc, nothing, cgrid, ker, 8000.0; workspace = wsc, filter_plan = cplan, deriv_plan = cdplan)) < 2048

        # UnstructuredGrid
        npts = 300
        ugeom = CGEF.CartesianGeometry(1.0, 1.0)
        ulon = 60e3 .* rand(npts); ulat = 60e3 .* rand(npts)
        ugrid = CGEF.UnstructuredGrid(ugeom, ulon, ulat, trues(npts); k = 8)
        uplan = CGEF.Derivatives.WLSQGradientPlan(ugrid)
        ufplan = CGEF.Filtering.plan_filter(ugrid, CGEF.GaussianKernel(), 5000.0; method = CGEF.Filtering.Spectral())
        wsu = CGEF.Diagnostics.ΠWorkspace(ugrid)
        uu = randn(npts); vu = randn(npts); Πu = zeros(npts)
        CGEF.Diagnostics.compute_Π!(Πu, uu, vu, nothing, ugrid, CGEF.GaussianKernel(), 5000.0; workspace = wsu, filter_plan = ufplan, deriv_plan = uplan)
        CGEF.Diagnostics.compute_Π!(Πu, uu, vu, nothing, ugrid, CGEF.GaussianKernel(), 5000.0; workspace = wsu, filter_plan = ufplan, deriv_plan = uplan)
        Test.@test (@allocated CGEF.Diagnostics.compute_Π!(Πu, uu, vu, nothing, ugrid, CGEF.GaussianKernel(), 5000.0; workspace = wsu, filter_plan = ufplan, deriv_plan = uplan)) < 2048
    end

    # -----------------------------------------------------------------------
    # compute_Π_profile! / coarse_grain! / coarse_grain_profile / cumulative_energy! — a repeated
    # sweep over the SAME grid/kernel/scales, with workspace + prebuilt per-scale filter_plan(s)
    # supplied, must not rebuild the footprint (the real, previously-shipped bug this session found:
    # `compute_Π_profile!` rebuilt the same footprint once per depth level; `coarse_grain!` and
    # `cumulative_energy!` each independently rebuilt the same per-scale footprint a second time).
    # -----------------------------------------------------------------------
    Test.@testset "Repeated-sweep pipeline entry points: no redundant footprint rebuild" begin
        ker = CGEF.TopHatKernel()
        N = 48; dx = 1000.0
        geom = CGEF.CartesianGeometry(dx, dx)
        xsR = 0.0:dx:(N - 1) * dx
        grid = CGEF.StructuredGrid(geom, xsR, xsR, trues(N, N))
        scales = collect(5e3:5e3:15e3)
        u = randn(N, N); v = randn(N, N)

        # compute_Π_profile!: one plan per scale, reused across every depth level.
        Nz = 4
        u3 = randn(N, N, Nz); v3 = randn(N, N, Nz); Π3 = zeros(N, N, Nz)
        ws = CGEF.Diagnostics.ΠWorkspace(grid)
        plan = CGEF.Filtering.plan_filter(grid, ker, 10e3)
        CGEF.Diagnostics.compute_Π_profile!(Π3, u3, v3, nothing, grid, ker, 10e3; workspace = ws, filter_plan = plan)
        CGEF.Diagnostics.compute_Π_profile!(Π3, u3, v3, nothing, grid, ker, 10e3; workspace = ws, filter_plan = plan)
        a_profile = @allocated CGEF.Diagnostics.compute_Π_profile!(Π3, u3, v3, nothing, grid, ker, 10e3; workspace = ws, filter_plan = plan)
        Test.@test a_profile < 2048 * Nz  # one compute_Π! residual per level, no footprint rebuild

        # coarse_grain!: workspace + filter_plans prebuilt and reused (the documented "repeated
        # timestep sweep" zero-allocation entry point).
        result = CGEF.coarse_grain(u, v, grid; scales = scales, kernel = ker)
        plans = [CGEF.Filtering.plan_filter(grid, ker, Float64(s)) for s in scales]
        CGEF.Pipeline.coarse_grain!(result, u, v, grid; scales = scales, kernel = ker, workspace = ws, filter_plans = plans)
        CGEF.Pipeline.coarse_grain!(result, u, v, grid; scales = scales, kernel = ker, workspace = ws, filter_plans = plans)
        a_cg = @allocated CGEF.Pipeline.coarse_grain!(result, u, v, grid; scales = scales, kernel = ker, workspace = ws, filter_plans = plans)
        Test.@test a_cg < 2048 * length(scales)

        # Sanity: this bound is genuinely discriminating, not vacuous — WITHOUT prebuilt filter_plans,
        # `coarse_grain!` must allocate substantially more (it rebuilds `Nscales` footprints).
        a_cg_noplans = @allocated CGEF.Pipeline.coarse_grain!(result, u, v, grid; scales = scales, kernel = ker, workspace = ws)
        Test.@test a_cg_noplans > 10 * a_cg

        # coarse_grain_profile: same "prebuilt workspace + filter_plans" zero-(re)allocation contract.
        CGEF.Pipeline.coarse_grain_profile(u3, v3, grid; scales = scales, workspace = ws, filter_plans = plans)
        CGEF.Pipeline.coarse_grain_profile(u3, v3, grid; scales = scales, workspace = ws, filter_plans = plans)
        a_cgp = @allocated CGEF.Pipeline.coarse_grain_profile(u3, v3, grid; scales = scales, workspace = ws, filter_plans = plans)
        # This allocates the `Π`/`cumE`/`spec`/`CoarseGrainResult` output arrays every call (it has no
        # in-place `!` counterpart, unlike `coarse_grain!`) — bounded relative to the OUTPUT size, not
        # asserted small in absolute terms.
        Test.@test a_cgp < sizeof(Float64) * (N * N * Nz * length(scales)) * 2

        # cumulative_energy!/spectral_density! with prebuilt plans.
        spectrum = zeros(length(scales))
        CGEF.Diagnostics.cumulative_energy!(spectrum, u, v, nothing, grid, ker, scales; workspace = ws, filter_plans = plans)
        CGEF.Diagnostics.cumulative_energy!(spectrum, u, v, nothing, grid, ker, scales; workspace = ws, filter_plans = plans)
        a_ce = @allocated CGEF.Diagnostics.cumulative_energy!(spectrum, u, v, nothing, grid, ker, scales; workspace = ws, filter_plans = plans)
        Test.@test a_ce < 2048 * length(scales)

        wavenumber = 1.0 ./ scales
        density = zeros(length(scales))
        CGEF.Diagnostics.spectral_density!(density, spectrum, wavenumber)
        CGEF.Diagnostics.spectral_density!(density, spectrum, wavenumber)
        Test.@test (@allocated CGEF.Diagnostics.spectral_density!(density, spectrum, wavenumber)) < 512
    end

    # -----------------------------------------------------------------------
    # Parallel backends — bounded per real, already-documented behavior (see
    # docs/src/architecture.md): ThreadedBackend pays OhMyThreads' own small task-spawn bookkeeping;
    # GPUBackend re-uploads the mask/footprint device buffers every call (a known, not-yet-cached
    # inefficiency, distinct from a footprint REBUILD — the footprint itself is still reused).
    # Distributed/MPI backends involve real inter-process communication and are exercised for
    # correctness (not allocation) by the "Distributed backend"/"MPI"/`mpi_runtests.jl` tests instead.
    # -----------------------------------------------------------------------
    Test.@testset "Parallel backends: bounded, documented per-call cost" begin
        ker = CGEF.TopHatKernel()
        N = 48; dx = 1000.0
        geom = CGEF.CartesianGeometry(dx, dx)
        xsR = 0.0:dx:(N - 1) * dx
        grid2d = CGEF.StructuredGrid(geom, xsR, xsR, trues(N, N))
        u2d = randn(N, N); out2d = zeros(N, N)

        tplan = CGEF.Filtering.plan_filter(grid2d, ker, 5000.0; backend = CGEF.Backends.ThreadedBackend())
        CGEF.Filtering.filter_apply!(out2d, u2d, tplan); CGEF.Filtering.filter_apply!(out2d, u2d, tplan)
        Test.@test (@allocated CGEF.Filtering.filter_apply!(out2d, u2d, tplan)) < 4096

        N1 = 150
        grid1d = CGEF.StructuredGrid(geom, 0.0:dx:(N1 - 1) * dx, trues(N1))
        tplan1 = CGEF.Filtering.plan_filter(grid1d, ker, 5000.0; backend = CGEF.Backends.ThreadedBackend())
        u1d = randn(N1); out1d = zeros(N1)
        CGEF.Filtering.filter_apply!(out1d, u1d, tplan1); CGEF.Filtering.filter_apply!(out1d, u1d, tplan1)
        Test.@test (@allocated CGEF.Filtering.filter_apply!(out1d, u1d, tplan1)) < 4096

        gpuplan = CGEF.Filtering.plan_filter(grid2d, ker, 5000.0; backend = CGEF.Backends.GPUBackend(KA.CPU()))
        CGEF.Filtering.filter_apply!(out2d, u2d, gpuplan); CGEF.Filtering.filter_apply!(out2d, u2d, gpuplan)
        Test.@test (@allocated CGEF.Filtering.filter_apply!(out2d, u2d, gpuplan)) < 32_768
    end

end
