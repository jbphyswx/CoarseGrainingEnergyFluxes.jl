
# Allocations of the `!` entry points on their second call, with plans and workspaces held and an explicit
# `SerialBackend()`, each measured through a top-level function so the testset's captured locals are not
# counted. The filter applies are gated per engine in test_filtering.jl.

_cgef_a_ddz(o, f, g, dz, p) = @allocated CGEF.Derivatives.ddz!(o, f, g, dz, p)
_cgef_a_flux2(Π, u, v, g, k, ℓ, ws, fp, dp) =
    @allocated CGEF.Diagnostics.compute_Π!(Π, u, v, nothing, g, k, ℓ; workspace = ws, filter_plan = fp, deriv_plan = dp,
                                           execution = CGEF.ComputationalBackends.SerialBackend())
_cgef_a_flux3(Π, u, v, w, g, k, ℓ, ws, fp, dp) =
    @allocated CGEF.Diagnostics.compute_Π!(Π, u, v, w, g, k, ℓ; workspace = ws, filter_plan = fp, deriv_plan = dp,
                                           execution = CGEF.ComputationalBackends.SerialBackend())
_cgef_a_flux1(Π, u, g, k, ℓ, ws, fp, dp) =
    @allocated CGEF.Diagnostics.compute_Π!(Π, u, g, k, ℓ; workspace = ws, filter_plan = fp, deriv_plan = dp)
_cgef_a_sweep(r, u, v, g, kw) = @allocated CGEF.Pipeline.coarse_grain!(r, u, v, g; kw...)
_cgef_a_profile(b, u, v, g, kw) = @allocated CGEF.Pipeline.coarse_grain_profile!(b, u, v, g; kw...)
_cgef_a_batch(b, u, v, g, kw) = @allocated CGEF.Pipeline.coarse_grain_batch!(b, u, v, g; kw...)
_cgef_a_cumulative(E, u, v, g, k, s, ws, ps) =
    @allocated CGEF.Diagnostics.cumulative_energy!(E, u, v, nothing, g, k, s; workspace = ws, filter_plans = ps)
_cgef_a_density(d, C, k) = @allocated CGEF.Diagnostics.spectral_density!(d, C, k)
_cgef_a_energy(E, ws, p, a) = @allocated CGEF.Diagnostics.energy_from_filtered!(E, ws, p, false, a)
_cgef_a_field(o, f, g, k, ℓ, p) =
    @allocated CGEF.Filtering.filter_field!(o, f, g, k, ℓ; execution = CGEF.ComputationalBackends.SerialBackend(), filter_plan = p)
_cgef_a_apply(o, f, p) = @allocated CGEF.Filtering.filter_apply!(o, f, p)
_cgef_a_nufft(gp, sc) = @allocated begin
    FTB.nufft_type1!(sc.modes, gp.plan, sc.values)
    FTB.nufft_type2!(sc.values, gp.plan, sc.modes)
end
_cgef_a_nusht(o, f, xfer, plan, ws, rtol) = @allocated NUFSHT.nusht_filter!(o, f, xfer, plan; ws, rtol)
_cgef_a_tau(ws, u, v, g, k, ℓ, p, dp) =
    @allocated CGEF.Diagnostics.tau_decomposition!(ws, u, v, g, k, ℓ; filter_plan = p, derived_plan = dp)
_cgef_a_sc(ws, u, v, g, k, ℓ, p, d) =
    @allocated CGEF.Diagnostics.compute_Π_strain_convergence!(ws, u, v, g, k, ℓ; filter_plan = p, deriv_plan = d)
_cgef_a_dec(ws, u, v, g, k, ℓ, p, d) =
    @allocated CGEF.Diagnostics.compute_Π_decomposed!(ws, u, v, u, v, g, k, ℓ; filter_plan = p, deriv_plan = d)
_cgef_a_fav(ws, u, v, r, P, g, k, ℓ, p, d) =
    @allocated CGEF.Diagnostics.compressible_flux!(ws, u, v, r, P, g, k, ℓ; filter_plan = p, deriv_plan = d)
_cgef_a_trc(o, ws, u, v, t, g, k, ℓ, p, d) =
    @allocated CGEF.Diagnostics.tracer_variance_flux!(o, ws, u, v, t, g, k, ℓ; filter_plan = p, deriv_plan = d)
_cgef_a_ens(o, ws, u, v, g, k, ℓ, p, d, c) =
    @allocated CGEF.Diagnostics.enstrophy_flux!(o, ws, u, v, g, k, ℓ; filter_plan = p, deriv_plan = d, curl_plan = c)

# The second of two calls.
_cgef_second(f) = (f(); f())

Test.@testset "Allocations" begin
    SER = CGEF.ComputationalBackends.SerialBackend()
    # The transform libraries and OhMyThreads spawn tasks once the session has more than one thread; that
    # cost is the dependency's, zero at one thread, and far below a plan rebuild.
    TASK_SLACK = (Threads.nthreads() - 1) * 512 * 1024
    cart = FG.Geometry.CartesianGeometry()
    dx = 1000.0
    N = 32
    xs = 0.0:dx:(N - 1) * dx
    g2 = FG.Grids.StructuredGrid(cart, xs, xs)
    m2 = trues(N, N)
    m2[3:6, 3:6] .= false
    g2m = FG.Grids.StructuredGrid(cart, xs, xs, m2)
    R = 6.371e6
    gs = FG.Grids.StructuredGrid(FG.Geometry.SphericalGeometry(R), range(0.0; step = deg2rad(8.0), length = 45),
                                 range(deg2rad(-80.0); step = deg2rad(8.0), length = 21))
    g1 = FG.Grids.StructuredGrid(cart, xs)
    x3 = 0.0:dx:15dx
    g3 = FG.Grids.StructuredGrid(cart, x3, x3, x3)
    m3 = trues(16, 16, 16)
    m3[3:5, 3:5, 3:5] .= false
    g3m = FG.Grids.StructuredGrid(cart, x3, x3, x3, m3)
    θ = deg2rad(15.0)
    cg = FG.Grids.CurvilinearGrid(cart, [dx * (i * cos(θ) - 0.3j * sin(θ)) for i in 0:(N - 1), j in 0:(N - 1)],
                                  [dx * (i * sin(θ) + j * (1 + 0.3cos(θ))) for i in 0:(N - 1), j in 0:(N - 1)], trues(N, N))
    field(sz) = [sin(0.3 * I[1] + 0.7 * get(Tuple(I), 2, 0) + 0.2 * get(Tuple(I), 3, 0)) for I in CartesianIndices(sz)]
    th = CGEF.TopHatKernel()
    ga = CGEF.GaussianKernel()

    Test.@testset "a level stack's vertical derivative with a held plan" begin
        f = field((N, N, 12))
        pz = FG.Discretization.stencil_plan(range(0.0; step = dx, length = 12), 1, 3)
        Test.@test _cgef_second(() -> _cgef_a_ddz(zeros(N, N, 12), f, g2, dx, pz)) == 0
    end

    # A diagnostic's derivative plan is built on the grid its filtered fields are read on.
    dplan(g, fp) = CGEF.Derivatives.derivative_plan(CGEF.Diagnostics.output_grid(g, fp))

    # A sphere's workspace holds the generalized convolution's spin transforms; FINUFFT, which NUFSHT runs
    # them on here, allocates nothing per execution.
    FIN = (; nufft = FTB.FINUFFTBackend())

    Test.@testset "compute_Π! with its workspace and plans held" begin
        for (g, k, ℓ) in ((g2, th, 5dx), (g2m, ga, 5dx), (gs, ga, 1.5e6), (gs, th, 1.5e6), (cg, th, 8dx))
            sz = FG.Grids.size_tuple(g)
            ws = CGEF.Diagnostics.ΠWorkspace(g; FIN...)
            fp = CGEF.Filtering.plan_filter(g, k, ℓ; execution = SER)
            u, v = field(sz), field(sz) .^ 2
            Test.@test _cgef_second(() -> _cgef_a_flux2(zeros(sz), u, v, g, k, ℓ, ws, fp, dplan(g, fp))) == 0
        end
        # The in-core direct spin transforms.
        let sz = FG.Grids.size_tuple(gs), fp = CGEF.Filtering.plan_filter(gs, ga, 1.5e6; execution = SER),
            ws = CGEF.Diagnostics.ΠWorkspace(gs; transform = CGEF.SpectralBackends.DirectSumSpectralBackend())
            Test.@test _cgef_second(() -> _cgef_a_flux2(zeros(sz), field(sz), field(sz) .^ 2, gs, ga, 1.5e6, ws, fp,
                                                         dplan(gs, fp))) == 0
        end
        for g in (g3, g3m)
            ws = CGEF.Diagnostics.ΠWorkspace(g; has_w = true)
            fp = CGEF.Filtering.plan_filter(g, th, 2.5dx; execution = SER)
            f = field((16, 16, 16))
            Test.@test _cgef_second(() -> _cgef_a_flux3(zeros(16, 16, 16), f, f .^ 2, f .^ 3, g, th, 2.5dx, ws, fp,
                                                         dplan(g, fp))) == 0
        end
        let ws = CGEF.Diagnostics.ΠWorkspace(g1), fp = CGEF.Filtering.plan_filter(g1, th, 5dx; execution = SER)
            Test.@test _cgef_second(() -> _cgef_a_flux1(zeros(N), field((N,)), g1, th, 5dx, ws, fp, dplan(g1, fp))) == 0
        end
        # A node set's nonuniform transform: FINUFFT allocates nothing per execution.
        let n = 300, px = 6e4 .* (0.5 .+ 0.5 .* sin.(1:n)), py = 6e4 .* (0.5 .+ 0.5 .* cos.(1.7 .* (1:n))),
            ug = FG.Grids.UnstructuredGrid(cart, px, py, trues(n); k = 8)
            fp = CGEF.Filtering.plan_filter(ug, ga, 5dx; execution = SER, method = CGEF.Filtering.Spectral(),
                                            transform = FTB.FINUFFTBackend())
            Test.@test _cgef_second(() -> _cgef_a_flux2(zeros(n), sin.(px ./ 7e3), cos.(py ./ 9e3), ug, ga, 5dx,
                                                         CGEF.Diagnostics.ΠWorkspace(ug), fp, dplan(ug, fp))) <=
                       TASK_SLACK
        end
    end

    Test.@testset "sweeps and batches" begin
        scales = [5dx, 10dx]
        ws = CGEF.Diagnostics.ΠWorkspace(g2)
        dp = CGEF.Derivatives.derivative_plan(g2)
        plans = [CGEF.Filtering.plan_filter(g2, th, s; execution = SER) for s in scales]
        u, v = field((N, N)), field((N, N)) .^ 2
        r = CGEF.coarse_grain(u, v, g2; scales, kernel = th, execution = SER, spectrum = CGEF.Diagnostics.NoSpectrum())
        kw = (; scales, kernel = th, workspace = ws, filter_plans = plans, deriv_plan = dp, execution = SER,
              spectrum = CGEF.Diagnostics.NoSpectrum())
        Test.@test _cgef_second(() -> _cgef_a_sweep(r, u, v, g2, kw)) == 0
        u3, v3 = field((N, N, 3)), field((N, N, 3)) .^ 2
        pkw = (; scales, kernel = th, workspaces = [ws], filter_plans = [plans], deriv_plans = [dp], execution = SER,
               spectrum = CGEF.Diagnostics.NoSpectrum())
        b = CGEF.Pipeline.CoarseGrainBatchResult(g2, 2, (3,))
        Test.@test _cgef_second(() -> _cgef_a_profile(b, u3, v3, g2, pkw)) == 0
        Test.@test _cgef_second(() -> _cgef_a_batch(b, u3, v3, g2, pkw)) == 0
        # Scratch the caller leaves out is built once per worker, so a batch twice as long costs the same.
        bkw = (; scales, kernel = th, execution = SER, spectrum = CGEF.Diagnostics.NoSpectrum())
        b6 = CGEF.Pipeline.CoarseGrainBatchResult(g2, 2, (6,))
        u6, v6 = cat(u3, u3; dims = 3), cat(v3, v3; dims = 3)
        Test.@test _cgef_second(() -> _cgef_a_batch(b6, u6, v6, g2, bkw)) == _cgef_second(() -> _cgef_a_batch(b, u3, v3, g2, bkw))
        E = zeros(2)
        Test.@test _cgef_second(() -> _cgef_a_cumulative(E, u, v, g2, th, scales, ws, plans)) == 0
        Test.@test _cgef_second(() -> _cgef_a_density(zeros(2), E, 1.0 ./ scales)) == 0

        # On a sphere the generalized convolution reads no filter plan, and a sweep builds none.
        szs = FG.Grids.size_tuple(gs)
        us, vs = field(szs), field(szs) .^ 2
        sscales = [1.5e6, 3e6]
        wss = CGEF.Diagnostics.ΠWorkspace(gs; FIN...)
        rs = CGEF.coarse_grain(us, vs, gs; scales = sscales, kernel = ga, execution = SER,
                               spectrum = CGEF.Diagnostics.NoSpectrum())
        skw = (; scales = sscales, kernel = ga, workspace = wss, execution = SER, spectrum = CGEF.Diagnostics.NoSpectrum(),
               deriv_plan = CGEF.Derivatives.derivative_plan(CGEF.Diagnostics.output_grid(gs, CGEF.Filtering.ZeroFill())))
        Test.@test _cgef_second(() -> _cgef_a_sweep(rs, us, vs, gs, skw)) == 0
        ews = CGEF.Diagnostics.EnergyWorkspace(gs; FIN...)
        Test.@test _cgef_second(() -> _cgef_a_cumulative(zeros(2), us, vs, gs, ga, sscales, ews, nothing)) == 0
        wsb = CGEF.Diagnostics.ΠWorkspace(g2, (3,))
        CGEF.Diagnostics.compute_Π!(zeros(N, N, 3), u3, v3, nothing, g2, th, 5dx; workspace = wsb, filter_plan = plans[1], execution = SER)
        Test.@test _cgef_second(() -> _cgef_a_energy(zeros(3), wsb, plans[1], CGEF.Diagnostics.active_area(g2))) == 0
        Test.@test _cgef_second(() -> _cgef_a_field(zeros(N, N), u, g2, th, 5dx, plans[1])) == 0
    end

    # The returned NamedTuple and nothing field-sized: one field here is well over the bound.
    Test.@testset "diagnostic workspace forms" begin
        BOUND = 4096
        D = CGEF.Diagnostics
        for g in (g2m, gs)
            sz = FG.Grids.size_tuple(g)
            ℓ = g === gs ? 1.5e6 : 5dx
            u, v, t = field(sz), field(sz) .^ 2, field(sz) .^ 3
            ρ, P = fill(2.5, sz), fill(1.0e5, sz)
            pl = CGEF.Filtering.plan_filter(g, ga, ℓ; execution = SER)
            # The second filter of a Germano moment reads the first's output over every cell.
            pd = CGEF.Filtering.plan_filter(D.output_grid(g, pl), ga, ℓ; execution = SER)
            dp = dplan(g, pl)
            sph = g === gs
            Test.@test _cgef_second(() -> _cgef_a_tau(sph ? D.Sym3TauWorkspace(g; FIN...) : D.TauWorkspace(g), u, v, g, ga, ℓ, pl, pd)) < BOUND
            Test.@test _cgef_second(() -> _cgef_a_sc(sph ? D.ΠWorkspace(g; FIN...) : D.PiStrainWorkspace(g), u, v, g, ga, ℓ, pl, dp)) < BOUND
            Test.@test _cgef_second(() -> _cgef_a_dec(sph ? D.SphericalPiDecomposedWorkspace(g; FIN...) : D.PiDecomposedWorkspace(g),
                                                      u, v, g, ga, ℓ, pl, dp)) < BOUND
            Test.@test _cgef_second(() -> _cgef_a_fav(sph ? D.SphericalFavreWorkspace(g; FIN...) : D.FavreWorkspace(g),
                                                      u, v, ρ, P, g, ga, ℓ, pl, dp)) < BOUND
            Test.@test _cgef_second(() -> _cgef_a_trc(zeros(sz), sph ? D.SphericalTracerFluxWorkspace(g; FIN...) : D.TracerFluxWorkspace(g),
                                                      u, v, t, g, ga, ℓ, pl, dp)) < BOUND
            # The vorticity differences the raw velocity over the grid's own cells.
            cp = CGEF.Derivatives.derivative_plan(g)
            Test.@test _cgef_second(() -> _cgef_a_ens(zeros(sz), D.EnstrophyFluxWorkspace(g; FIN...), u, v, g, ga, ℓ, pl, dp, cp)) < BOUND
        end
    end

    Test.@testset "parallel and device applies" begin
        CB = CGEF.ComputationalBackends
        u = field((N, N))
        for (b, bound) in ((CB.ThreadedBackend(), 4096 + TASK_SLACK), (CB.GPUBackend(KA.CPU()), 4096))
            for k in (th, ga)
                p = CGEF.Filtering.plan_filter(g2, k, 5dx; execution = b)
                Test.@test _cgef_second(() -> _cgef_a_apply(zeros(N, N), u, p)) < bound
            end
        end
        # A device node plan holds its adjacency on the device, so an apply costs the same however many
        # neighbours each node has.
        n = 24 * 24
        ug = FG.Grids.UnstructuredGrid(cart, vec([Float64(i) for i in 1:24, j in 1:24]), vec([Float64(j) for i in 1:24, j in 1:24]),
                                       ones(n), trues(n))
        pn(ℓ) = CGEF.Filtering.plan_filter(ug, ga, ℓ; execution = CB.GPUBackend(KA.CPU()), method = CGEF.Filtering.RealSpace())
        un = sin.(1:n)
        Test.@test _cgef_second(() -> _cgef_a_apply(zeros(n), un, pn(2.0))) == _cgef_second(() -> _cgef_a_apply(zeros(n), un, pn(8.0)))
    end

    Test.@testset "spectral applies add nothing to their library's transforms" begin
        u = field((N, N))
        gp = FG.Grids.StructuredGrid(cart, xs, xs; periodic = (true, true))
        p = CGEF.Filtering.plan_filter(gp, ga, 5dx; execution = SER, method = CGEF.Filtering.Spectral())
        Test.@test _cgef_second(() -> _cgef_a_apply(zeros(N, N), u, p)) <= TASK_SLACK
        n = 400
        ug = FG.Grids.UnstructuredGrid(cart, 6e4 .* (0.5 .+ 0.5 .* sin.(1:n)), 6e4 .* (0.5 .+ 0.5 .* cos.(1.7 .* (1:n))), ones(n), trues(n))
        un = sin.(1:n)
        for lib in (FTB.FINUFFTBackend(), FTB.NonuniformFFTsBackend())
            pu = CGEF.Filtering.plan_filter(ug, ga, 5dx; execution = SER, method = CGEF.Filtering.Spectral(), transform = lib)
            CGEF.Filtering.filter_apply!(zeros(n), un, pu)
            pair = minimum(_ -> _cgef_a_nufft(pu.grid_plan, pu.scratch), 1:3)
            Test.@test minimum(_ -> _cgef_a_apply(zeros(n), un, pu), 1:3) <= pair + TASK_SLACK
            lib isa FTB.FINUFFTBackend && Test.@test pair <= TASK_SLACK
        end
        sg = FG.Connectivity.structured_grid(FG.SphericalSampling.ClenshawCurtisSampling(), 17; geometry = FG.Geometry.SphericalGeometry(R))
        ps = CGEF.Filtering.plan_filter(sg, ga, π * R / 8; execution = SER, method = CGEF.Filtering.Spectral())
        Test.@test _cgef_second(() -> _cgef_a_apply(zeros(33, 17), field((33, 17)), ps)) <= TASK_SLACK
        # NUFSHT allocates inside its own fit; the extension adds nothing over the same upstream call.
        M = 200
        lat = asin.(clamp.(2 .* mod.(0.618 .* (1:M), 1.0) .- 1, -1, 1))
        lon = 2π .* mod.(0.4142 .* (1:M), 1.0)
        nug = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(R), lon, lat, ones(M), trues(M))
        pq = CGEF.Filtering.plan_filter(nug, ga, π * R / 8; execution = SER, method = CGEF.Filtering.Spectral())
        f = sin.(lon) .* cos.(lat)
        CGEF.Filtering.filter_apply!(zeros(M), f, pq)
        rtol = Base.get_extension(CGEF, :CoarseGrainingEnergyFluxesNUFSHTExt)._fit_rtol(pq.grid_plan.plan.tol, Float64)
        upstream = _cgef_second(() -> _cgef_a_nusht(zeros(M), f, pq.filter, pq.grid_plan.plan, pq.scratch.ws, rtol))
        Test.@test _cgef_second(() -> _cgef_a_apply(zeros(M), f, pq)) <= upstream + TASK_SLACK
    end

    # Plan memory: the prefix-sum engine keeps no neighbour list, so a sixteen-fold wider filter barely
    # grows its plan; a streaming scattered plan derives each neighbourhood per apply, so its build does
    # not grow with the window.
    Test.@testset "plan memory against the filter width" begin
        xv = collect(0.0:dx:63dx) .+ [0.3dx * sin(2.7i) for i in 1:64]
        g = FG.Grids.StructuredGrid(cart, xv, xv)
        Test.@test Base.summarysize(CGEF.Filtering.build_footprint(g, th, 64dx)) < 3 * Base.summarysize(CGEF.Filtering.build_footprint(g, th, 4dx))
        build(ℓ) = @allocated CGEF.Filtering.build_footprint(g, CGEF.SharpSpectralKernel(), ℓ; cache_strategy = CGEF.Filtering.NeverCache())
        build(2dx)
        build(4dx)
        Test.@test build(4dx) < 1.5 * build(2dx)
    end
end
