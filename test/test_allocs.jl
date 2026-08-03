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
   measuring the underlying kernel directly at 0 bytes; or (b) a backend's own per-call bookkeeping
   (`GPUBackend`'s kernel launches, the `OhMyThreads`
   scheduler's task spawning) or an upstream dependency's own allocation behavior (NUFSHT.jl's
   `nusht_filter!`, confirmed via a direct, isolated measurement of that exact call — not something this
   package's own extension code does; not something fixable from this repository). Bounds use a
   generous-but-real safety margin over the measured value so minor Julia-version/CPU noise doesn't
   make this flaky, while still catching an actual regression (e.g. a reintroduced full footprint
   rebuild, which is orders of magnitude larger than any bound here).

`filter_plan`/`filter_plans`/`deriv_plan`/`workspace` reuse is exactly what makes a caller's REPEATED
sweep (many timesteps over the same grid/scales) genuinely zero-(re)allocation — the whole point of
these tests is to catch a regression back to a per-call footprint/plan rebuild (see CHANGELOG.md for
the concrete cost of that regression).
=#

using Test: Test

Test.@testset "Zero-/bounded-allocation hot paths" begin

    # These bounds are about the numerical kernels, so the backend is pinned rather than left to
    # `AutoBackend`: on a multi-threaded session Auto resolves to `ThreadedBackend`, whose task
    # spawning allocates by design. That cost has its own bounded testset further down.
    SERIAL = CGEF.ComputationalBackends.SerialBackend()

    # The transform libraries (FFTW, FastSphericalHarmonics, NUFSHT, FINUFFT) and OhMyThreads spawn
    # Julia tasks once the session has more than one thread, and task creation allocates. That cost
    # belongs to the dependency, scales with `nthreads()` rather than with problem size, and is
    # exactly zero at one thread — so the strict bounds below hold verbatim single-threaded and get
    # this allowance otherwise. It stays far below a rebuild, which is what these tests exist to catch.
    TASK_SLACK = (Threads.nthreads() - 1) * 512 * 1024

    # -----------------------------------------------------------------------
    # Real-space (RealSpace) filter_apply! on a cached plan — exact zero
    # -----------------------------------------------------------------------
    Test.@testset "filter_apply! (real-space, cached plan): exact zero" begin
        ker = CGEF.TopHatKernel()

        # StructuredGrid 2D Cartesian (uniform Range axes -> fast translation-invariant path)
        N = 48; dx = 1000.0
        geom = FG.Geometry.CartesianGeometry()
        xsR = 0.0:dx:(N - 1) * dx
        grid2d = FG.Grids.StructuredGrid(geom, xsR, xsR, trues(N, N))
        plan2d = CGEF.Filtering.plan_filter(grid2d, ker, 5000.0; backend = SERIAL)
        u2d = randn(N, N); out2d = zeros(N, N)
        CGEF.Filtering.filter_apply!(out2d, u2d, plan2d)
        CGEF.Filtering.filter_apply!(out2d, u2d, plan2d)
        Test.@test (@allocated CGEF.Filtering.filter_apply!(out2d, u2d, plan2d)) == 0

        # StructuredGrid 2D Spherical (per-latitude-band fast path)
        R = 6.371e6
        lon_deg = 0.0:4.0:356.0; lat_deg = -80.0:4.0:80.0
        lonR = range(deg2rad(first(lon_deg)); step = deg2rad(step(lon_deg)), length = length(lon_deg))
        latR = range(deg2rad(first(lat_deg)); step = deg2rad(step(lat_deg)), length = length(lat_deg))
        grids2d = FG.Grids.StructuredGrid(FG.Geometry.SphericalGeometry(R), lonR, latR, trues(length(lonR), length(latR)))
        plans2d = CGEF.Filtering.plan_filter(grids2d, ker, 400e3; backend = SERIAL)
        us2d = randn(length(lonR), length(latR)); outs2d = zeros(length(lonR), length(latR))
        CGEF.Filtering.filter_apply!(outs2d, us2d, plans2d)
        CGEF.Filtering.filter_apply!(outs2d, us2d, plans2d)
        Test.@test (@allocated CGEF.Filtering.filter_apply!(outs2d, us2d, plans2d)) == 0

        # StructuredGrid 1D Cartesian
        grid1d = FG.Grids.StructuredGrid(geom, xsR, trues(N))
        plan1d = CGEF.Filtering.plan_filter(grid1d, ker, 5000.0; backend = SERIAL)
        u1d = randn(N); out1d = zeros(N)
        CGEF.Filtering.filter_apply!(out1d, u1d, plan1d)
        CGEF.Filtering.filter_apply!(out1d, u1d, plan1d)
        Test.@test (@allocated CGEF.Filtering.filter_apply!(out1d, u1d, plan1d)) == 0

        # StructuredGrid true-3D Cartesian
        # The level-stack ddz! takes its spacing as an argument, so its axis is not on the grid and its
        # weights cannot come from a grid-built plan. Given one over the axis it is free too; without
        # one the table is O(Nz) per call.
        let Nz = 12, u25 = randn(N, N, Nz), o25 = zeros(N, N, Nz), r25 = zeros(N, N, Nz)
            pz = CGEF.Derivatives.StencilPlan(range(0.0; step = dx, length = Nz))
            CGEF.Derivatives.ddz!(r25, u25, grid2d, dx)
            CGEF.Derivatives.ddz!(o25, u25, grid2d, dx, pz)
            CGEF.Derivatives.ddz!(o25, u25, grid2d, dx, pz)
            Test.@test (@allocated CGEF.Derivatives.ddz!(o25, u25, grid2d, dx, pz)) == 0
            Test.@test o25 == r25
        end

        N3 = 16
        geom3 = FG.Geometry.CartesianGeometry()
        xs3R = 0.0:dx:(N3 - 1) * dx
        grid3d = FG.Grids.StructuredGrid(geom3, xs3R, xs3R, xs3R, trues(N3, N3, N3))
        plan3d = CGEF.Filtering.plan_filter(grid3d, ker, 2500.0; backend = SERIAL)
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
        cgrid = FG.Grids.CurvilinearGrid(geom, clon, clat, trues(Nc, Nc))
        cplan = CGEF.Filtering.plan_filter(cgrid, ker, 8000.0; backend = SERIAL)
        uc = randn(Nc, Nc); outc = zeros(Nc, Nc)
        CGEF.Filtering.filter_apply!(outc, uc, cplan)
        CGEF.Filtering.filter_apply!(outc, uc, cplan)
        Test.@test (@allocated CGEF.Filtering.filter_apply!(outc, uc, cplan)) == 0
    end

    # -----------------------------------------------------------------------
    # Derivatives ddx!/ddy!/ddz! — exact zero with a held stencil table
    # -----------------------------------------------------------------------
    # The weights are a function of the grid alone, so the reusable form is the one that can be free:
    # given a `StencilPlan` a derivative touches no new memory. Without one every call rebuilds an
    # `n × nodes` table, which is the honest cost of asking for a derivative and nothing else.
    Test.@testset "ddx!/ddy!/ddz!: exact zero" begin
        N = 48; dx = 1000.0
        geom = FG.Geometry.CartesianGeometry()
        xsR = 0.0:dx:(N - 1) * dx
        grid2d = FG.Grids.StructuredGrid(geom, xsR, xsR)
        p2d = CGEF.Derivatives.StencilPlan(grid2d)
        u2d = randn(N, N); out2d = zeros(N, N)
        CGEF.Derivatives.ddx!(out2d, u2d, grid2d, p2d); CGEF.Derivatives.ddx!(out2d, u2d, grid2d, p2d)
        Test.@test (@allocated CGEF.Derivatives.ddx!(out2d, u2d, grid2d, p2d)) == 0
        CGEF.Derivatives.ddy!(out2d, u2d, grid2d, p2d); CGEF.Derivatives.ddy!(out2d, u2d, grid2d, p2d)
        Test.@test (@allocated CGEF.Derivatives.ddy!(out2d, u2d, grid2d, p2d)) == 0

        # The held table has to give the same answer as rebuilding it, or the reuse is not reuse.
        ref2d = zeros(N, N)
        CGEF.Derivatives.ddx!(ref2d, u2d, grid2d)
        CGEF.Derivatives.ddx!(out2d, u2d, grid2d, p2d)
        Test.@test out2d == ref2d

        # A mask edge puts the degrade path in play, rebuilding a window per cell. The plan carries that
        # scratch too, so it is free as well — and a mask with real holes is what actually exercises it,
        # an all-true mask never reaching the rebuild at all.
        let mask = trues(N, N)
            mask[3:6, 3:6] .= false
            gmask = FG.Grids.StructuredGrid(geom, xsR, xsR, mask)
            pm = CGEF.Derivatives.StencilPlan(gmask)
            om = zeros(N, N)
            CGEF.Derivatives.ddx!(om, u2d, gmask, pm); CGEF.Derivatives.ddx!(om, u2d, gmask, pm)
            Test.@test (@allocated CGEF.Derivatives.ddx!(om, u2d, gmask, pm)) == 0
        end

        R = 6.371e6
        lon_deg = 0.0:4.0:356.0; lat_deg = -80.0:4.0:80.0
        lonR = range(deg2rad(first(lon_deg)); step = deg2rad(step(lon_deg)), length = length(lon_deg))
        latR = range(deg2rad(first(lat_deg)); step = deg2rad(step(lat_deg)), length = length(lat_deg))
        grids2d = FG.Grids.StructuredGrid(FG.Geometry.SphericalGeometry(R), lonR, latR)
        ps2d = CGEF.Derivatives.StencilPlan(grids2d)
        us2d = randn(length(lonR), length(latR)); outs2d = zeros(length(lonR), length(latR))
        CGEF.Derivatives.ddx!(outs2d, us2d, grids2d, ps2d); CGEF.Derivatives.ddx!(outs2d, us2d, grids2d, ps2d)
        Test.@test (@allocated CGEF.Derivatives.ddx!(outs2d, us2d, grids2d, ps2d)) == 0
        CGEF.Derivatives.ddy!(outs2d, us2d, grids2d, ps2d); CGEF.Derivatives.ddy!(outs2d, us2d, grids2d, ps2d)
        Test.@test (@allocated CGEF.Derivatives.ddy!(outs2d, us2d, grids2d, ps2d)) == 0

        grid1d = FG.Grids.StructuredGrid(geom, xsR)
        p1d = CGEF.Derivatives.StencilPlan(grid1d)
        u1d = randn(N); out1d = zeros(N)
        CGEF.Derivatives.ddx!(out1d, u1d, grid1d, p1d); CGEF.Derivatives.ddx!(out1d, u1d, grid1d, p1d)
        Test.@test (@allocated CGEF.Derivatives.ddx!(out1d, u1d, grid1d, p1d)) == 0

        N3 = 16
        geom3 = FG.Geometry.CartesianGeometry()
        xs3R = 0.0:dx:(N3 - 1) * dx
        grid3d = FG.Grids.StructuredGrid(geom3, xs3R, xs3R, xs3R)
        p3d = CGEF.Derivatives.StencilPlan(grid3d)
        u3d = randn(N3, N3, N3); out3d = zeros(N3, N3, N3)
        CGEF.Derivatives.ddx!(out3d, u3d, grid3d, p3d); CGEF.Derivatives.ddx!(out3d, u3d, grid3d, p3d)
        Test.@test (@allocated CGEF.Derivatives.ddx!(out3d, u3d, grid3d, p3d)) == 0
        CGEF.Derivatives.ddz!(out3d, u3d, grid3d, p3d); CGEF.Derivatives.ddz!(out3d, u3d, grid3d, p3d)
        Test.@test (@allocated CGEF.Derivatives.ddz!(out3d, u3d, grid3d, p3d)) == 0

        # CurvilinearGrid: cached least-squares gradient plan
        Nc = 40
        i = collect(0.0:(Nc - 1)); j = collect(0.0:(Nc - 1))
        θ = deg2rad(15.0); shear = 0.3
        clon = [dx * (ii * cos(θ) - jj * shear * sin(θ)) for ii in i, jj in j]
        clat = [dx * (ii * sin(θ) + jj * (1 + shear * cos(θ))) for ii in i, jj in j]
        cgrid = FG.Grids.CurvilinearGrid(geom, clon, clat, trues(Nc, Nc))
        cdplan = FG.Connectivity.gradient_plan(cgrid)
        uc = randn(Nc, Nc); outc = zeros(Nc, Nc)
        outc2 = similar(outc)
        FG.Discretization.gradient!(outc, outc2, uc, cdplan); FG.Discretization.gradient!(outc, outc2, uc, cdplan)
        Test.@test (@allocated FG.Discretization.gradient!(outc, outc2, uc, cdplan)) == 0

        # UnstructuredGrid: the same plan over k-d tree adjacency
        npts = 300
        ugeom = FG.Geometry.CartesianGeometry()
        ulon = 60e3 .* rand(npts); ulat = 60e3 .* rand(npts)
        ugrid = FG.Grids.UnstructuredGrid(ugeom, ulon, ulat, trues(npts); k = 8)
        udplan = FG.Connectivity.gradient_plan(ugrid)
        uu = randn(npts); outu = zeros(npts)
        outu2 = similar(outu)
        FG.Discretization.gradient!(outu, outu2, uu, udplan); FG.Discretization.gradient!(outu, outu2, uu, udplan)
        Test.@test (@allocated FG.Discretization.gradient!(outu, outu2, uu, udplan)) == 0
    end

    # -----------------------------------------------------------------------
    # Spectral filter_apply! — exact zero for FFTW/FINUFFT/FastSphericalHarmonics; NUFSHT is an
    # upstream (NUFSHT.jl) allocation, not something this package's extension code does.
    # -----------------------------------------------------------------------
    Test.@testset "filter_apply! (spectral, cached plan)" begin
        N = 48; dx = 1000.0
        geom = FG.Geometry.CartesianGeometry()
        xsR = 0.0:dx:(N - 1) * dx
        gridp = FG.Grids.StructuredGrid(geom, xsR, xsR, trues(N, N); periodic = (true, true))
        u = randn(N, N); out = zeros(N, N)
        fftplan = CGEF.Filtering.plan_filter(gridp, CGEF.GaussianKernel(), 5000.0; backend = SERIAL, method = CGEF.Filtering.Spectral())
        CGEF.Filtering.filter_apply!(out, u, fftplan); CGEF.Filtering.filter_apply!(out, u, fftplan)
        Test.@test (@allocated CGEF.Filtering.filter_apply!(out, u, fftplan)) <= TASK_SLACK

        npts = 400
        ugeom = FG.Geometry.CartesianGeometry()
        ulon = 60e3 .* rand(npts); ulat = 60e3 .* rand(npts)
        ugrid = FG.Grids.UnstructuredGrid(ugeom, ulon, ulat, ones(npts), trues(npts))
        uf = randn(npts); outu = zeros(npts)
        finufftplan = CGEF.Filtering.plan_filter(ugrid, CGEF.GaussianKernel(), 5000.0; backend = SERIAL, method = CGEF.Filtering.Spectral())
        CGEF.Filtering.filter_apply!(outu, uf, finufftplan); CGEF.Filtering.filter_apply!(outu, uf, finufftplan)
        Test.@test (@allocated CGEF.Filtering.filter_apply!(outu, uf, finufftplan)) <= TASK_SLACK

        Ndeg = 16; Nsh = Ndeg + 1; Msh = 2Nsh - 1
        Θ, Φ = FSH.sph_points(Nsh)
        R = 6.371e6
        sgrid = FG.Grids.StructuredGrid(FG.Geometry.SphericalGeometry(R), collect(Φ), π / 2 .- collect(Θ), trues(Msh, Nsh))
        field = randn(Msh, Nsh); outsh = zeros(Msh, Nsh)
        shtplan = CGEF.Filtering.plan_filter(sgrid, CGEF.GaussianKernel(), π * R / 8; backend = SERIAL, method = CGEF.Filtering.Spectral())
        CGEF.Filtering.filter_apply!(outsh, field, shtplan); CGEF.Filtering.filter_apply!(outsh, field, shtplan)
        Test.@test (@allocated CGEF.Filtering.filter_apply!(outsh, field, shtplan)) <= TASK_SLACK

        # NUFSHT: confirmed via a direct, isolated measurement of NUFSHT.nusht_filter! itself (not
        # through this package's extension code) that the allocation lives entirely inside NUFSHT.jl
        # — an unregistered, separately-maintained sibling package (github.com/jbphyswx/NUFSHT.jl),
        # not something fixable from this repository. Bounded (not zero) so a regression here — e.g.
        # this extension accidentally adding ITS OWN allocation on top — is still caught.
        Mpts = 200
        sφ = 2π .* rand(Mpts); sθ = acos.(clamp.(1 .- 2 .* rand(Mpts), -1, 1))
        slat = π / 2 .- sθ
        nugrid = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(R), sφ, slat, ones(Mpts), trues(Mpts))
        nuf = randn(Mpts); outnu = zeros(Mpts)
        nushtplan = CGEF.Filtering.plan_filter(nugrid, CGEF.GaussianKernel(), π * R / 8; backend = SERIAL, method = CGEF.Filtering.Spectral())
        CGEF.Filtering.filter_apply!(outnu, nuf, nushtplan); CGEF.Filtering.filter_apply!(outnu, nuf, nushtplan)
        a_nufsht = @allocated CGEF.Filtering.filter_apply!(outnu, nuf, nushtplan)
        Test.@test a_nufsht < 300_000 + TASK_SLACK
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
        geom = FG.Geometry.CartesianGeometry()
        xsR = 0.0:dx:(N - 1) * dx

        # 2D Cartesian
        grid2d = FG.Grids.StructuredGrid(geom, xsR, xsR, trues(N, N))
        plan2d = CGEF.Filtering.plan_filter(grid2d, ker, 5000.0; backend = SERIAL)
        ws2d = CGEF.Diagnostics.ΠWorkspace(grid2d)
        dp2d = CGEF.Derivatives.StencilPlan(grid2d)
        u2d = randn(N, N); v2d = randn(N, N); Π2d = zeros(N, N)
        CGEF.Diagnostics.compute_Π!(Π2d, u2d, v2d, nothing, grid2d, ker, 5000.0; workspace = ws2d, filter_plan = plan2d, deriv_plan = dp2d)
        CGEF.Diagnostics.compute_Π!(Π2d, u2d, v2d, nothing, grid2d, ker, 5000.0; workspace = ws2d, filter_plan = plan2d, deriv_plan = dp2d)
        Test.@test (@allocated CGEF.Diagnostics.compute_Π!(Π2d, u2d, v2d, nothing, grid2d, ker, 5000.0; workspace = ws2d, filter_plan = plan2d, deriv_plan = dp2d)) < 2048

        # 2D Spherical
        R = 6.371e6
        lon_deg = 0.0:4.0:356.0; lat_deg = -80.0:4.0:80.0
        lonR = range(deg2rad(first(lon_deg)); step = deg2rad(step(lon_deg)), length = length(lon_deg))
        latR = range(deg2rad(first(lat_deg)); step = deg2rad(step(lat_deg)), length = length(lat_deg))
        grids2d = FG.Grids.StructuredGrid(FG.Geometry.SphericalGeometry(R), lonR, latR, trues(length(lonR), length(latR)))
        plans2d = CGEF.Filtering.plan_filter(grids2d, ker, 400e3; backend = SERIAL)
        wss2d = CGEF.Diagnostics.ΠWorkspace(grids2d)
        dps2d = CGEF.Derivatives.StencilPlan(grids2d)
        us2d = randn(length(lonR), length(latR)); vs2d = randn(length(lonR), length(latR)); Πs2d = zeros(length(lonR), length(latR))
        CGEF.Diagnostics.compute_Π!(Πs2d, us2d, vs2d, nothing, grids2d, ker, 400e3; workspace = wss2d, filter_plan = plans2d, deriv_plan = dps2d)
        CGEF.Diagnostics.compute_Π!(Πs2d, us2d, vs2d, nothing, grids2d, ker, 400e3; workspace = wss2d, filter_plan = plans2d, deriv_plan = dps2d)
        Test.@test (@allocated CGEF.Diagnostics.compute_Π!(Πs2d, us2d, vs2d, nothing, grids2d, ker, 400e3; workspace = wss2d, filter_plan = plans2d, deriv_plan = dps2d)) < 2048

        # 1D Cartesian
        grid1d = FG.Grids.StructuredGrid(geom, xsR, trues(N))
        plan1d = CGEF.Filtering.plan_filter(grid1d, ker, 5000.0; backend = SERIAL)
        ws1d = CGEF.Diagnostics.ΠWorkspace(grid1d)
        dp1d = CGEF.Derivatives.StencilPlan(grid1d)
        u1d = randn(N); Π1d = zeros(N)
        CGEF.Diagnostics.compute_Π!(Π1d, u1d, grid1d, ker, 5000.0; workspace = ws1d, filter_plan = plan1d, deriv_plan = dp1d)
        CGEF.Diagnostics.compute_Π!(Π1d, u1d, grid1d, ker, 5000.0; workspace = ws1d, filter_plan = plan1d, deriv_plan = dp1d)
        Test.@test (@allocated CGEF.Diagnostics.compute_Π!(Π1d, u1d, grid1d, ker, 5000.0; workspace = ws1d, filter_plan = plan1d, deriv_plan = dp1d)) < 2048

        # true-3D Cartesian. With the workspace, filter plan and stencil table all held, an unmasked
        # grid reaches exactly zero. A dense mask puts each derivative on the degrade path, whose
        # per-call scratch is upstream's and O(1) — so it is asserted not to grow with the grid rather
        # than against a magic number.
        N3 = 16
        geom3 = FG.Geometry.CartesianGeometry()
        xs3R = 0.0:dx:(N3 - 1) * dx
        grid3d = FG.Grids.StructuredGrid(geom3, xs3R, xs3R, xs3R)
        plan3d = CGEF.Filtering.plan_filter(grid3d, ker, 2500.0; backend = SERIAL)
        ws3d = CGEF.Diagnostics.ΠWorkspace(grid3d)
        dp3d = CGEF.Derivatives.StencilPlan(grid3d)
        u3d = randn(N3, N3, N3); v3d = randn(N3, N3, N3); w3d = randn(N3, N3, N3); Π3d = zeros(N3, N3, N3)
        CGEF.Diagnostics.compute_Π!(Π3d, u3d, v3d, w3d, grid3d, ker, 2500.0; workspace = ws3d, filter_plan = plan3d, deriv_plan = dp3d)
        CGEF.Diagnostics.compute_Π!(Π3d, u3d, v3d, w3d, grid3d, ker, 2500.0; workspace = ws3d, filter_plan = plan3d, deriv_plan = dp3d)
        Test.@test (@allocated CGEF.Diagnostics.compute_Π!(Π3d, u3d, v3d, w3d, grid3d, ker, 2500.0; workspace = ws3d, filter_plan = plan3d, deriv_plan = dp3d)) == 0

        # …and with a genuinely masked 3D grid, where all nine derivatives take the degrade path.
        let m3 = trues(N3, N3, N3)
            m3[3:5, 3:5, 3:5] .= false
            g = FG.Grids.StructuredGrid(geom3, xs3R, xs3R, xs3R, m3)
            pl = CGEF.Filtering.plan_filter(g, ker, 2500.0; backend = SERIAL)
            wsm = CGEF.Diagnostics.ΠWorkspace(g); dpm = CGEF.Derivatives.StencilPlan(g)
            o = zeros(N3, N3, N3)
            for _ in 1:2
                CGEF.Diagnostics.compute_Π!(o, u3d, v3d, w3d, g, ker, 2500.0; workspace = wsm, filter_plan = pl, deriv_plan = dpm)
            end
            Test.@test (@allocated CGEF.Diagnostics.compute_Π!(o, u3d, v3d, w3d, g, ker, 2500.0; workspace = wsm, filter_plan = pl, deriv_plan = dpm)) == 0
        end

        # CurvilinearGrid
        Nc = 40
        i = collect(0.0:(Nc - 1)); j = collect(0.0:(Nc - 1))
        θ = deg2rad(15.0); shear = 0.3
        clon = [dx * (ii * cos(θ) - jj * shear * sin(θ)) for ii in i, jj in j]
        clat = [dx * (ii * sin(θ) + jj * (1 + shear * cos(θ))) for ii in i, jj in j]
        cgrid = FG.Grids.CurvilinearGrid(geom, clon, clat, trues(Nc, Nc))
        cplan = CGEF.Filtering.plan_filter(cgrid, ker, 8000.0; backend = SERIAL)
        cdplan = FG.Connectivity.gradient_plan(cgrid)
        wsc = CGEF.Diagnostics.ΠWorkspace(cgrid)
        uc = randn(Nc, Nc); vc = randn(Nc, Nc); Πc = zeros(Nc, Nc)
        CGEF.Diagnostics.compute_Π!(Πc, uc, vc, nothing, cgrid, ker, 8000.0; workspace = wsc, filter_plan = cplan, deriv_plan = cdplan)
        CGEF.Diagnostics.compute_Π!(Πc, uc, vc, nothing, cgrid, ker, 8000.0; workspace = wsc, filter_plan = cplan, deriv_plan = cdplan)
        Test.@test (@allocated CGEF.Diagnostics.compute_Π!(Πc, uc, vc, nothing, cgrid, ker, 8000.0; workspace = wsc, filter_plan = cplan, deriv_plan = cdplan)) < 2048

        # UnstructuredGrid
        npts = 300
        ugeom = FG.Geometry.CartesianGeometry()
        ulon = 60e3 .* rand(npts); ulat = 60e3 .* rand(npts)
        ugrid = FG.Grids.UnstructuredGrid(ugeom, ulon, ulat, trues(npts); k = 8)
        uplan = FG.Connectivity.gradient_plan(ugrid)
        ufplan = CGEF.Filtering.plan_filter(ugrid, CGEF.GaussianKernel(), 5000.0; backend = SERIAL, method = CGEF.Filtering.Spectral())
        wsu = CGEF.Diagnostics.ΠWorkspace(ugrid)
        uu = randn(npts); vu = randn(npts); Πu = zeros(npts)
        CGEF.Diagnostics.compute_Π!(Πu, uu, vu, nothing, ugrid, CGEF.GaussianKernel(), 5000.0; workspace = wsu, filter_plan = ufplan, deriv_plan = uplan)
        CGEF.Diagnostics.compute_Π!(Πu, uu, vu, nothing, ugrid, CGEF.GaussianKernel(), 5000.0; workspace = wsu, filter_plan = ufplan, deriv_plan = uplan)
        Test.@test (@allocated CGEF.Diagnostics.compute_Π!(Πu, uu, vu, nothing, ugrid, CGEF.GaussianKernel(), 5000.0; workspace = wsu, filter_plan = ufplan, deriv_plan = uplan)) < 2048 + TASK_SLACK
    end

    # -----------------------------------------------------------------------
    # compute_Π_profile! / coarse_grain! / coarse_grain_profile / cumulative_energy! — a repeated
    # sweep over the SAME grid/kernel/scales, with workspace + prebuilt per-scale filter_plan(s)
    # supplied, must not rebuild the footprint: `compute_Π_profile!` must not rebuild it once per
    # depth level, and `coarse_grain!`/`cumulative_energy!` must not each independently rebuild the
    # same per-scale footprint a second time.
    # -----------------------------------------------------------------------
    Test.@testset "Repeated-sweep pipeline entry points: no redundant footprint rebuild" begin
        ker = CGEF.TopHatKernel()
        N = 48; dx = 1000.0
        geom = FG.Geometry.CartesianGeometry()
        xsR = 0.0:dx:(N - 1) * dx
        grid = FG.Grids.StructuredGrid(geom, xsR, xsR, trues(N, N))
        scales = collect(5e3:5e3:15e3)
        u = randn(N, N); v = randn(N, N)

        # compute_Π_profile!: one plan per scale, reused across every depth level.
        Nz = 4
        u3 = randn(N, N, Nz); v3 = randn(N, N, Nz); Π3 = zeros(N, N, Nz)
        ws = CGEF.Diagnostics.ΠWorkspace(grid)
        dpg = CGEF.Derivatives.StencilPlan(grid)
        plan = CGEF.Filtering.plan_filter(grid, ker, 10e3; backend = SERIAL)
        CGEF.Diagnostics.compute_Π_profile!(Π3, u3, v3, nothing, grid, ker, 10e3; backend = SERIAL, workspace = ws, filter_plan = plan, deriv_plan = dpg)
        CGEF.Diagnostics.compute_Π_profile!(Π3, u3, v3, nothing, grid, ker, 10e3; backend = SERIAL, workspace = ws, filter_plan = plan, deriv_plan = dpg)
        a_profile = @allocated CGEF.Diagnostics.compute_Π_profile!(Π3, u3, v3, nothing, grid, ker, 10e3; backend = SERIAL, workspace = ws, filter_plan = plan, deriv_plan = dpg)
        Test.@test a_profile == 0  # every level shares the one workspace, filter plan and stencil table

        # coarse_grain!: workspace + filter_plans prebuilt and reused (the documented "repeated
        # timestep sweep" zero-allocation entry point).
        result = CGEF.coarse_grain(u, v, grid; backend = SERIAL, scales = scales, kernel = ker)
        plans = [CGEF.Filtering.plan_filter(grid, ker, Float64(s); backend = SERIAL) for s in scales]
        CGEF.Pipeline.coarse_grain!(result, u, v, grid; backend = SERIAL, scales = scales, kernel = ker, workspace = ws, filter_plans = plans, deriv_plan = dpg)
        CGEF.Pipeline.coarse_grain!(result, u, v, grid; backend = SERIAL, scales = scales, kernel = ker, workspace = ws, filter_plans = plans, deriv_plan = dpg)
        a_cg = @allocated CGEF.Pipeline.coarse_grain!(result, u, v, grid; backend = SERIAL, scales = scales, kernel = ker, workspace = ws, filter_plans = plans, deriv_plan = dpg)
        Test.@test a_cg == 0

        # Sanity: zero is a real result, not a vacuous one — WITHOUT prebuilt filter_plans the same
        # call rebuilds `Nscales` footprints and allocates. (A ratio against `a_cg` would say nothing
        # now that it is zero.)
        a_cg_noplans = @allocated CGEF.Pipeline.coarse_grain!(result, u, v, grid; backend = SERIAL, scales = scales, kernel = ker, workspace = ws, deriv_plan = dpg)
        Test.@test a_cg_noplans > 10_000

        # coarse_grain_profile: same "prebuilt workspace + filter_plans" zero-(re)allocation contract.
        CGEF.Pipeline.coarse_grain_profile(u3, v3, grid; backend = SERIAL, scales = scales, workspace = ws, filter_plans = plans, deriv_plan = dpg)
        CGEF.Pipeline.coarse_grain_profile(u3, v3, grid; backend = SERIAL, scales = scales, workspace = ws, filter_plans = plans, deriv_plan = dpg)
        a_cgp = @allocated CGEF.Pipeline.coarse_grain_profile(u3, v3, grid; backend = SERIAL, scales = scales, workspace = ws, filter_plans = plans, deriv_plan = dpg)
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
    # Parallel backends — bounded by each one's real per-call bookkeeping: ThreadedBackend pays
    # OhMyThreads' task spawning, GPUBackend the kernel launches (its device tables, mask and scratch
    # planes are uploaded once by `prepare_workspace`, not per call). Distributed/MPI involve real
    # inter-process communication and are exercised for correctness (not allocation) by the
    # "Distributed backend"/"MPI"/`mpi_runtests.jl` tests instead.
    # -----------------------------------------------------------------------
    Test.@testset "Parallel backends: bounded, documented per-call cost" begin
        ker = CGEF.TopHatKernel()
        N = 48; dx = 1000.0
        geom = FG.Geometry.CartesianGeometry()
        xsR = 0.0:dx:(N - 1) * dx
        grid2d = FG.Grids.StructuredGrid(geom, xsR, xsR, trues(N, N))
        u2d = randn(N, N); out2d = zeros(N, N)

        tplan = CGEF.Filtering.plan_filter(grid2d, ker, 5000.0; backend = CGEF.ComputationalBackends.ThreadedBackend())
        CGEF.Filtering.filter_apply!(out2d, u2d, tplan); CGEF.Filtering.filter_apply!(out2d, u2d, tplan)
        Test.@test (@allocated CGEF.Filtering.filter_apply!(out2d, u2d, tplan)) < 4096 + TASK_SLACK

        N1 = 150
        grid1d = FG.Grids.StructuredGrid(geom, 0.0:dx:(N1 - 1) * dx, trues(N1))
        tplan1 = CGEF.Filtering.plan_filter(grid1d, ker, 5000.0; backend = CGEF.ComputationalBackends.ThreadedBackend())
        u1d = randn(N1); out1d = zeros(N1)
        CGEF.Filtering.filter_apply!(out1d, u1d, tplan1); CGEF.Filtering.filter_apply!(out1d, u1d, tplan1)
        Test.@test (@allocated CGEF.Filtering.filter_apply!(out1d, u1d, tplan1)) < 4096 + TASK_SLACK

        # A top-hat on uniform axes plans to the prefix-sum engine, which the GPU backend runs on the
        # host; the Gaussian is what actually launches device kernels, so it is the one that would
        # expose a per-call upload.
        gpuplan = CGEF.Filtering.plan_filter(grid2d, ker, 5000.0; backend = CGEF.ComputationalBackends.GPUBackend(KA.CPU()))
        CGEF.Filtering.filter_apply!(out2d, u2d, gpuplan); CGEF.Filtering.filter_apply!(out2d, u2d, gpuplan)
        Test.@test (@allocated CGEF.Filtering.filter_apply!(out2d, u2d, gpuplan)) < 4096

        gpugauss = CGEF.Filtering.plan_filter(grid2d, CGEF.GaussianKernel(), 5000.0; backend = CGEF.ComputationalBackends.GPUBackend(KA.CPU()))
        CGEF.Filtering.filter_apply!(out2d, u2d, gpugauss); CGEF.Filtering.filter_apply!(out2d, u2d, gpugauss)
        Test.@test (@allocated CGEF.Filtering.filter_apply!(out2d, u2d, gpugauss)) < 4096
    end

    # -----------------------------------------------------------------------
    # filter_field!/filter_fields! are thin wrappers over plan_filter+filter_apply! — bare calls
    # (no `filter_plan=`) rebuild the footprint EVERY call (the historical, still-correct default
    # behavior for a one-shot call), while supplying a prebuilt `filter_plan=` makes repeated calls
    # over the same grid/kernel/scale genuinely zero-(re)allocation. The bound below is a "genuinely
    # discriminating" sanity check (mirroring the `coarse_grain!` one above), not just "some improvement".
    # -----------------------------------------------------------------------
    Test.@testset "filter_field!/filter_fields!: filter_plan reuse eliminates the per-call footprint rebuild" begin
        ker = CGEF.TopHatKernel()
        N = 48; dx = 1000.0
        geom = FG.Geometry.CartesianGeometry()
        xsR = 0.0:dx:(N - 1) * dx
        grid = FG.Grids.StructuredGrid(geom, xsR, xsR, trues(N, N))
        u = randn(N, N); out = zeros(N, N)

        CGEF.Filtering.filter_field!(out, u, grid, ker, 5000.0; backend = SERIAL)  # warm up
        a_noplan = @allocated CGEF.Filtering.filter_field!(out, u, grid, ker, 5000.0; backend = SERIAL)

        plan = CGEF.Filtering.plan_filter(grid, ker, 5000.0; backend = SERIAL)
        CGEF.Filtering.filter_field!(out, u, grid, ker, 5000.0; backend = SERIAL, filter_plan = plan)  # warm up
        a_plan = @allocated CGEF.Filtering.filter_field!(out, u, grid, ker, 5000.0; backend = SERIAL, filter_plan = plan)
        Test.@test a_plan == 0
        Test.@test a_noplan > 10 * max(a_plan, 1)

        # filter_fields!: same contract, several fields sharing one plan.
        v = randn(N, N); out2 = zeros(N, N)
        CGEF.Filtering.filter_fields!((out, out2), (u, v), grid, ker, 5000.0; backend = SERIAL, filter_plan = plan)
        a_fields_plan = @allocated CGEF.Filtering.filter_fields!((out, out2), (u, v), grid, ker, 5000.0; backend = SERIAL, filter_plan = plan)
        Test.@test a_fields_plan == 0
    end

    # -----------------------------------------------------------------------
    # Prefix-sum top-hat engine: the apply must be exact-zero-allocation (all scratch, including the
    # prefix table, lives in the plan), and the plan must scale with the GRID, not with the grid times
    # the window — the whole point of the O(N·dj_lim) reformulation. Contrast this with the scattered
    # engine's optional cache, whose size does grow with the window (measured in the testset below).
    # -----------------------------------------------------------------------
    Test.@testset "Prefix-sum top-hat: zero-allocation apply, plan size independent of filter width" begin
        N = 64; dx = 1000.0
        geom = FG.Geometry.CartesianGeometry()
        xsV = collect(0.0:dx:(N - 1) * dx) .+ [0.3 * dx * sin(2.7i) for i in 1:N]   # genuinely nonuniform
        grid = FG.Grids.StructuredGrid(geom, xsV, xsV, trues(N, N))
        u = randn(N, N); out = zeros(N, N)

        for strat in (CGEF.Filtering.Deformable(), CGEF.Filtering.ZeroFill())
            plan = CGEF.Filtering.plan_filter(grid, CGEF.TopHatKernel(), 20_000.0; backend = SERIAL, mask_strategy = strat)
            Test.@test plan.footprint isa CGEF.Filtering.PrefixSumTopHatPlan
            CGEF.Filtering.filter_apply!(out, u, plan); CGEF.Filtering.filter_apply!(out, u, plan)
            Test.@test (@allocated CGEF.Filtering.filter_apply!(out, u, plan)) == 0
        end

        # Plan size must be essentially independent of the filter width: a 16x wider filter touches 16x
        # more neighbours per point, which an O(N·M) neighbour list would reflect directly in its size.
        fp_narrow = CGEF.Filtering.build_footprint(grid, CGEF.TopHatKernel(), 4_000.0)
        fp_wide = CGEF.Filtering.build_footprint(grid, CGEF.TopHatKernel(), 64_000.0)
        Test.@test Base.summarysize(fp_wide) < 1.05 * Base.summarysize(fp_narrow)

        # ...and it grows only linearly in the number of grid points (prefix tables are O(N)).
        N2 = 2 * N
        xsV2 = collect(0.0:dx:(N2 - 1) * dx) .+ [0.3 * dx * sin(2.7i) for i in 1:N2]
        grid2 = FG.Grids.StructuredGrid(geom, xsV2, xsV2, trues(N2, N2))
        fp2 = CGEF.Filtering.build_footprint(grid2, CGEF.TopHatKernel(), 20_000.0)
        fp1 = CGEF.Filtering.build_footprint(grid, CGEF.TopHatKernel(), 20_000.0)
        # 4x the points (2x per axis) => ~4x the storage, not 16x or worse.
        Test.@test Base.summarysize(fp2) < 6 * Base.summarysize(fp1)
    end

    # -----------------------------------------------------------------------
    # A streaming footprint (`NeverCache`) is O(1) scalar metadata and must not grow with N at fixed
    # relative kernel radius. `AlwaysCache`/`AutoCache` are exercised for contrast: an O(N·M) cache is
    # expected to be large, which is why it is optional.
    # -----------------------------------------------------------------------
    Test.@testset "ScatteredFilterPlan/NDScatteredFilterPlan size is O(1) (streaming) vs O(N·M) (cached)" begin
        # `GaussianKernel`, not `TopHatKernel`: top-hat now takes the exact prefix-sum path, which has
        # no per-point neighbour list at all (and hence no cache to size). A Gaussian on a nonuniform
        # axis still builds the scattered plan whose O(1)-vs-O(N·M) sizing is what this testset measures.
        # The cache is the SCATTERED engine's per-point neighbour list, so this needs the one kernel that
        # reaches it: top-hat takes the exact prefix-sum path and a Gaussian is separable on any
        # rectilinear grid, uniform or not, so neither builds a per-point list at all. The scale is
        # rescaled to the window the Gaussian spanned, since both radii are linear in it.
        ker = CGEF.SharpSpectralKernel()
        _rescale(s0) = s0 * CGEF.Kernels.kernel_radius(CGEF.GaussianKernel(), s0) /
                       CGEF.Kernels.kernel_radius(ker, s0)
        geom = FG.Geometry.CartesianGeometry()
        dx = 1000.0
        N = 60
        # The perturbation is a fixed function of the index, so di_lim/dj_lim stay exactly
        # reproducible across the different N compared further down.
        xsV = collect(0.0:dx:(N - 1) * dx) .+ [0.3 * dx * sin(2.7i) for i in 1:N]
        grid = FG.Grids.StructuredGrid(geom, xsV, xsV, trues(N, N))
        scale = _rescale(15_000.0)

        fp_always = CGEF.Filtering.build_footprint(grid, ker, scale; cache_strategy = CGEF.Filtering.AlwaysCache())
        fp_never = CGEF.Filtering.build_footprint(grid, ker, scale; cache_strategy = CGEF.Filtering.NeverCache())
        Test.@test fp_always isa CGEF.Filtering.ScatteredFilterPlan

        bytes_cached = Base.summarysize(fp_always)
        bytes_streaming = Base.summarysize(fp_never)
        # Streaming carries only scalar metadata (kernel, scale, rad, window limits, periodicity/
        # period, geometry flag) — a handful of Float64/Int/Bool fields, not O(N·M) data.
        Test.@test bytes_streaming < 512
        # The cached plan holds a per-point neighbour list, so at this N/M it must be orders of
        # magnitude larger — that size is exactly why caching is opt-in.
        Test.@test bytes_cached > 1000 * bytes_streaming

        # Size INVARIANCE: the streaming footprint's size must not grow with N at fixed relative
        # kernel radius/spacing — this, not "smaller than caching," is the actual O(1) proof.
        N2 = 4 * N
        xsV2 = collect(0.0:dx:(N2 - 1) * dx) .+ [0.3 * dx * sin(2.7i) for i in 1:N2]
        grid2 = FG.Grids.StructuredGrid(geom, xsV2, xsV2, trues(N2, N2))
        fp_never2 = CGEF.Filtering.build_footprint(grid2, ker, scale; cache_strategy = CGEF.Filtering.NeverCache())
        Test.@test Base.summarysize(fp_never2) == Base.summarysize(fp_never)

        # NDScatteredFilterPlan (1D): the same O(1)-vs-O(N) contrast.
        grid1 = FG.Grids.StructuredGrid(geom, xsV, trues(N))
        fp1_always = CGEF.Filtering.build_footprint(grid1, ker, scale; cache_strategy = CGEF.Filtering.AlwaysCache())
        fp1_never = CGEF.Filtering.build_footprint(grid1, ker, scale; cache_strategy = CGEF.Filtering.NeverCache())
        Test.@test fp1_always isa CGEF.Filtering.NDScatteredFilterPlan
        Test.@test Base.summarysize(fp1_never) < 512
        Test.@test Base.summarysize(fp1_always) > 100 * Base.summarysize(fp1_never)

        # Construction itself (not just the finished result) must not transiently allocate O(N·M) —
        # `NeverCache` must skip the fill loop entirely, not build-then-discard the neighbour list.
        CGEF.Filtering.build_footprint(grid, ker, scale; cache_strategy = CGEF.Filtering.NeverCache())  # warm up
        bytes_build_never = @allocated CGEF.Filtering.build_footprint(grid, ker, scale; cache_strategy = CGEF.Filtering.NeverCache())
        CGEF.Filtering.build_footprint(grid, ker, scale; cache_strategy = CGEF.Filtering.AlwaysCache())  # warm up
        bytes_build_always = @allocated CGEF.Filtering.build_footprint(grid, ker, scale; cache_strategy = CGEF.Filtering.AlwaysCache())
        Test.@test bytes_build_never < 1024
        Test.@test bytes_build_always > 100 * bytes_build_never

        CGEF.Filtering.build_footprint(grid1, ker, scale; cache_strategy = CGEF.Filtering.NeverCache())  # warm up
        bytes_build1_never = @allocated CGEF.Filtering.build_footprint(grid1, ker, scale; cache_strategy = CGEF.Filtering.NeverCache())
        Test.@test bytes_build1_never < 1024
    end

    # -----------------------------------------------------------------------
    # The spherical per-latitude-band fast path's two-pass sizehint must reserve
    # capacity proportional to each band's OWN (cosφ-dependent) di_lim, not a single global
    # pole-worst-case bound applied to every band. Measured directly (construction `@allocated`), not
    # asserted from source reading — a near-polar grid makes the equatorial-vs-polar di_lim ratio
    # large enough that a global bound would be a real, order-of-magnitude over-reservation.
    # -----------------------------------------------------------------------
    Test.@testset "spherical per-band footprint construction allocation tracks actual content, not a global pole-worst-case bound" begin
        R = 6.371e6
        sgeom = FG.Geometry.SphericalGeometry(R)
        lat_deg = -88.0:2.0:88.0
        lon_deg = 0.0:2.0:358.0
        lon = range(deg2rad(first(lon_deg)); step = deg2rad(step(lon_deg)), length = length(lon_deg))
        lat = range(deg2rad(first(lat_deg)); step = deg2rad(step(lat_deg)), length = length(lat_deg))
        grid = FG.Grids.StructuredGrid(sgeom, lon, lat, trues(length(lon), length(lat)))
        # `GaussianKernel` keeps the spherical per-latitude-band `FilterFootprint` path under test:
        # the separable fast path is Cartesian-only, and top-hat now takes the prefix-sum path.
        ker = CGEF.GaussianKernel(); scale = 300e3
        Ny = length(lat); dλ = step(lon); dφ = step(lat)
        rad = CGEF.Kernels.kernel_radius(ker, scale)
        dj_lim = ceil(Int, rad / (R * dφ))

        # The global worst-case di_lim across every band, recomputed here rather than calling the
        # package's own helper. A single shared sizehint would reserve this for every band regardless
        # of that band's own need.
        di_lim_max = 0
        for φ in lat
            cosφ = cos(φ)
            dl = abs(cosφ) > 1e-12 ? ceil(Int, rad / (R * cosφ * dλ)) : 0
            di_lim_max = max(di_lim_max, dl)
        end
        naive_total = 0
        for j in 1:Ny, ddj in -dj_lim:dj_lim
            jj = j + ddj
            (1 <= jj <= Ny) || continue
            naive_total += 2 * di_lim_max + 1
        end

        fp = CGEF.Filtering.build_footprint(grid, ker, scale)
        actual_total = length(fp.w)  # the true, final entry count (content is the same either way;
                                     # only reserved CAPACITY during construction differs)

        # This grid genuinely exercises the fix: the global-worst-case bound is a large multiple of
        # the real content (near-polar di_lim >> equatorial di_lim).
        Test.@test naive_total > 5 * actual_total

        CGEF.Filtering.build_footprint(grid, ker, scale)  # warm up (compile)
        bytes_actual = @allocated CGEF.Filtering.build_footprint(grid, ker, scale)
        bytes_per_entry = 2 * sizeof(Int) + sizeof(Float64)
        # Generous margin over the tight content-only estimate (three separate Vectors' own object
        # overhead, the `di_lims` pass-1 scratch buffer, GC bookkeeping — measured ~5x on this grid,
        # not the ~1x a perfectly packed single allocation would give) — still far below what
        # sizing every band to the global worst case, `naive_total`, would cost.
        Test.@test bytes_actual < 8 * actual_total * bytes_per_entry
        Test.@test bytes_actual < 0.5 * naive_total * bytes_per_entry
    end

    # -----------------------------------------------------------------------
    # filter_apply_batch! — zero-allocation across every footprint/plan type, and (separately) a
    # timing check proving it collapses compute_Π!'s dominant real redundancy: K independent
    # `filter_apply!` calls on a STREAMING (NeverCache) plan each re-derive the same per-point
    # neighbour list/weight from scratch, while one batched call derives it exactly once and reuses it
    # across all K fields.
    # -----------------------------------------------------------------------
    Test.@testset "filter_apply_batch! (NTuple batch): exact zero" begin
        ker = CGEF.TopHatKernel()
        N = 40; dx = 1000.0
        geom = FG.Geometry.CartesianGeometry()
        xsR = 0.0:dx:(N - 1) * dx
        grid = FG.Grids.StructuredGrid(geom, xsR, xsR, trues(N, N))
        plan = CGEF.Filtering.plan_filter(grid, ker, 5000.0; backend = SERIAL)
        fields = (randn(N, N), randn(N, N), randn(N, N))
        outs = (zeros(N, N), zeros(N, N), zeros(N, N))
        CGEF.Filtering.filter_apply_batch!(outs, fields, plan)
        CGEF.Filtering.filter_apply_batch!(outs, fields, plan)
        Test.@test (@allocated CGEF.Filtering.filter_apply_batch!(outs, fields, plan)) == 0

        # Nonuniform-axis (ScatteredFilterPlan), cached and streaming.
        x_nu = collect(xsR) .+ [iseven(i) ? 3.0 : -2.0 for i in 1:N]
        grid_nu = FG.Grids.StructuredGrid(geom, x_nu, x_nu, trues(N, N))
        plan_cached = CGEF.Filtering.plan_filter(grid_nu, ker, 5000.0; backend = SERIAL, cache_strategy = CGEF.Filtering.AlwaysCache())
        plan_stream = CGEF.Filtering.plan_filter(grid_nu, ker, 5000.0; backend = SERIAL, cache_strategy = CGEF.Filtering.NeverCache())
        CGEF.Filtering.filter_apply_batch!(outs, fields, plan_cached); CGEF.Filtering.filter_apply_batch!(outs, fields, plan_cached)
        Test.@test (@allocated CGEF.Filtering.filter_apply_batch!(outs, fields, plan_cached)) == 0
        CGEF.Filtering.filter_apply_batch!(outs, fields, plan_stream); CGEF.Filtering.filter_apply_batch!(outs, fields, plan_stream)
        Test.@test (@allocated CGEF.Filtering.filter_apply_batch!(outs, fields, plan_stream)) == 0

        # Separable Gaussian fast path (per-field loop internally — no derivation to hoist out, but
        # still zero-allocation).
        gplan = CGEF.Filtering.plan_filter(grid, CGEF.GaussianKernel(), 5000.0; backend = SERIAL)
        CGEF.Filtering.filter_apply_batch!(outs, fields, gplan); CGEF.Filtering.filter_apply_batch!(outs, fields, gplan)
        Test.@test (@allocated CGEF.Filtering.filter_apply_batch!(outs, fields, gplan)) == 0

        # 1D nonuniform (NDScatteredFilterPlan), streaming.
        grid1 = FG.Grids.StructuredGrid(geom, x_nu, trues(N))
        plan1_stream = CGEF.Filtering.plan_filter(grid1, ker, 5000.0; backend = SERIAL, cache_strategy = CGEF.Filtering.NeverCache())
        fields1 = (randn(N), randn(N), randn(N))
        outs1 = (zeros(N), zeros(N), zeros(N))
        CGEF.Filtering.filter_apply_batch!(outs1, fields1, plan1_stream); CGEF.Filtering.filter_apply_batch!(outs1, fields1, plan1_stream)
        Test.@test (@allocated CGEF.Filtering.filter_apply_batch!(outs1, fields1, plan1_stream)) == 0
    end

    Test.@testset "filter_apply_batch! collapses compute_Π!'s redundant per-field neighbour-derivation cost (streaming, NeverCache)" begin
        _min_elapsed(f::Function, n::Integer) = minimum(@elapsed(f()) for _ in 1:n)

        # Batching collapses the per-point neighbour re-derivation, which only the scattered engine
        # does — see the cache-size testset above for why a Gaussian no longer reaches it.
        ker = CGEF.SharpSpectralKernel()
        _rescale(s0) = s0 * CGEF.Kernels.kernel_radius(CGEF.GaussianKernel(), s0) /
                       CGEF.Kernels.kernel_radius(ker, s0)
        geom = FG.Geometry.CartesianGeometry()
        N = 90
        x_nu = collect(0.0:800.0:(N - 1) * 800.0) .+ [iseven(i) ? 4.0 : -3.5 for i in 1:N]
        grid = FG.Grids.StructuredGrid(geom, x_nu, x_nu, trues(N, N))
        plan = CGEF.Filtering.plan_filter(grid, ker, _rescale(15_000.0); backend = SERIAL, cache_strategy = CGEF.Filtering.NeverCache())

        K = 9  # mirrors compute_Π!'s 9 quadratic-product filter_apply! calls per scale
        fields = ntuple(_ -> randn(N, N), K)
        outs_batch = ntuple(_ -> zeros(N, N), K)
        outs_naive = ntuple(_ -> zeros(N, N), K)

        CGEF.Filtering.filter_apply_batch!(outs_batch, fields, plan)  # warm up (compile)
        for k in 1:K
            CGEF.Filtering.filter_apply!(outs_naive[k], fields[k], plan)
        end
        for k in 1:K
            Test.@test outs_batch[k] == outs_naive[k]  # bit-identical: batching is a pure performance change
        end

        t_batch = _min_elapsed(() -> CGEF.Filtering.filter_apply_batch!(outs_batch, fields, plan), 5)
        t_naive = _min_elapsed(5) do
            for k in 1:K
                CGEF.Filtering.filter_apply!(outs_naive[k], fields[k], plan)
            end
        end

        # The batched call derives each point's neighbour list/weight ONCE and reuses it across all K
        # fields; K independent filter_apply! calls under a streaming plan redundantly re-derive it K
        # times. A generous (not tight) margin still firmly distinguishes "collapsed toward 1x" from
        # "no real improvement".
        Test.@test t_batch < t_naive * 0.7
    end

    Test.@testset "compute_Π! under NeverCache is not catastrophically slower than AlwaysCache (batching closes the gap)" begin
        _min_elapsed(f::Function, n::Integer) = minimum(@elapsed(f()) for _ in 1:n)

        ker = CGEF.TopHatKernel()
        geom = FG.Geometry.CartesianGeometry()
        N = 70
        x_nu = collect(0.0:900.0:(N - 1) * 900.0) .+ [iseven(i) ? 3.0 : -2.5 for i in 1:N]
        grid = FG.Grids.StructuredGrid(geom, x_nu, x_nu, trues(N, N))
        scale = 12_000.0
        u = randn(N, N); v = randn(N, N)
        Π = zeros(N, N)

        plan_always = CGEF.Filtering.plan_filter(grid, ker, scale; backend = SERIAL, cache_strategy = CGEF.Filtering.AlwaysCache())
        plan_never = CGEF.Filtering.plan_filter(grid, ker, scale; backend = SERIAL, cache_strategy = CGEF.Filtering.NeverCache())
        ws = CGEF.Diagnostics.ΠWorkspace(grid)

        CGEF.Diagnostics.compute_Π!(Π, u, v, nothing, grid, ker, scale; workspace = ws, filter_plan = plan_always)
        CGEF.Diagnostics.compute_Π!(Π, u, v, nothing, grid, ker, scale; workspace = ws, filter_plan = plan_never)

        t_always = _min_elapsed(() -> CGEF.Diagnostics.compute_Π!(Π, u, v, nothing, grid, ker, scale; workspace = ws, filter_plan = plan_always), 5)
        t_never = _min_elapsed(() -> CGEF.Diagnostics.compute_Π!(Π, u, v, nothing, grid, ker, scale; workspace = ws, filter_plan = plan_never), 5)

        # Before batching, EVERY one of compute_Π!'s ~9 internal filter_apply! calls under a streaming
        # plan independently re-derived the same per-point neighbour list/weight; a generous (not
        # tight) multiplier still catches a regression back to that unbatched behavior, where the
        # streaming case would be close to 9x slower rather than a modest constant-factor away.
        Test.@test t_never < 4 * t_always
    end

    # -----------------------------------------------------------------------
    # separable Gaussian fast path — empirical speedup, not just analytical (O(N·r) vs O(N·r²)).
    # Compared against a `CurvilinearGrid` built from the SAME physical points: `CurvilinearGrid`
    # always uses the general (disk-truncated) `ScatteredFilterPlan` real-space engine, so it is a
    # genuine same-kernel/same-scale reference for "the non-separable direct-sum cost on this grid,"
    # not a strawman — a `StructuredGrid` with the same axes would just take the separable path itself.
    # -----------------------------------------------------------------------
    Test.@testset "Separable Gaussian fast path is empirically faster than the general scattered engine at a real kernel radius" begin
        _min_elapsed(f::Function, n::Integer) = minimum(@elapsed(f()) for _ in 1:n)

        geom = FG.Geometry.CartesianGeometry()
        N = 80
        xsR = 0.0:1000.0:(N - 1) * 1000.0
        ker = CGEF.GaussianKernel()
        scale = 15_000.0  # a real, non-trivial kernel radius (rad ~ a few grid cells wide)

        grid_fast = FG.Grids.StructuredGrid(geom, xsR, xsR, trues(N, N))
        plan_fast = CGEF.Filtering.plan_filter(grid_fast, ker, scale; backend = SERIAL)
        Test.@test plan_fast.footprint isa CGEF.Filtering.SeparableGaussianFootprint

        ic = collect(0.0:(N - 1)); jc = collect(0.0:(N - 1))
        clon = [1000.0 * ii for ii in ic, jj in jc]
        clat = [1000.0 * jj for ii in ic, jj in jc]
        cgrid = FG.Grids.CurvilinearGrid(geom, clon, clat, trues(N, N))
        plan_general = CGEF.Filtering.plan_filter(cgrid, ker, scale; backend = SERIAL)
        Test.@test plan_general.footprint isa CGEF.Filtering.ScatteredFilterPlan

        field = randn(N, N)
        out_fast = zeros(N, N); out_general = zeros(N, N)
        CGEF.Filtering.filter_apply!(out_fast, field, plan_fast)
        CGEF.Filtering.filter_apply!(out_general, field, plan_general)

        t_fast = _min_elapsed(() -> CGEF.Filtering.filter_apply!(out_fast, field, plan_fast), 5)
        t_general = _min_elapsed(() -> CGEF.Filtering.filter_apply!(out_general, field, plan_general), 5)

        # O(N·r) vs O(N·r²): a generous (not tight) margin still firmly distinguishes a real speedup
        # from "no benefit" — the exact ratio depends on r and isn't asserted as a specific number.
        Test.@test t_fast < t_general * 0.5
    end

end
