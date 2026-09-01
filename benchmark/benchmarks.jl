# Benchmark suite (PkgBenchmark-compatible: defines a global `SUITE::BenchmarkGroup`).

using BenchmarkTools: BenchmarkTools
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF
using FlowGeometries: FlowGeometries as FG

const SUITE = BenchmarkTools.BenchmarkGroup()

let
    N = 128
    dx = 1_000.0
    geom = FG.Geometry.CartesianGeometry()
    x = collect(0.0:dx:dx*(N - 1))
    y = collect(0.0:dx:dx*(N - 1))
    grid = FG.Grids.StructuredGrid(geom, x, y)
    u = rand(N, N)
    v = rand(N, N)
    out = zeros(N, N)
    Π = zeros(N, N)
    scale = 10_000.0

    # Cold: nothing prebuilt, so each call also pays for a footprint and (for `compute_Π!`) a whole
    # workspace. This is what a one-shot caller sees.
    SUITE["filter_field!/tophat/128x128"] =
        BenchmarkTools.@benchmarkable CGEF.Filtering.filter_field!($out, $u, $grid, CGEF.TopHatKernel(), $scale)
    SUITE["compute_Pi!/tophat/128x128"] =
        BenchmarkTools.@benchmarkable CGEF.Diagnostics.compute_Π!($Π, $u, $v, nothing, $grid, CGEF.TopHatKernel(), $scale)

    # Held: the workspace, filter plan and stencil table prebuilt — the documented repeated-sweep path,
    # and the one that is allocation-free. Without these entries the suite only ever tracked the cold
    # cost, where a fresh `ΠWorkspace` dominates both the time and the memory.
    ker = CGEF.TopHatKernel()
    ws = CGEF.Diagnostics.ΠWorkspace(grid)
    dplan = CGEF.Derivatives.StencilPlan(grid)
    fplan = CGEF.Filtering.plan_filter(grid, ker, scale)
    SUITE["compute_Pi!/tophat/128x128/plans-held"] =
        BenchmarkTools.@benchmarkable CGEF.Diagnostics.compute_Π!(
            $Π, $u, $v, nothing, $grid, $ker, $scale;
            workspace = $ws, filter_plan = $fplan, deriv_plan = $dplan,
        )

    scales = collect(6_000.0:2_000.0:12_000.0)
    plans = [CGEF.Filtering.plan_filter(grid, ker, s) for s in scales]
    result = CGEF.coarse_grain(u, v, grid; scales = scales, kernel = ker, spectrum = CGEF.Diagnostics.NoSpectrum())
    SUITE["coarse_grain!/tophat/128x128/4-scales/plans-held"] =
        BenchmarkTools.@benchmarkable CGEF.Pipeline.coarse_grain!(
            $result, $u, $v, $grid; scales = $scales, kernel = $ker,
            workspace = $ws, filter_plans = $plans, deriv_plan = $dplan,
        )
end

# The three diagnostics that have workspace forms, each measured BOTH ways. The allocating form is
# what a one-shot caller gets; the held form is the repeated-sweep path and is the one whose
# allocation must stay flat — these entries exist so a regression to per-call allocation is visible
# in the baseline rather than discovered later.
let
    N = 192
    dx = 1_000.0
    geom = FG.Geometry.CartesianGeometry()
    x = 0.0:dx:dx*(N - 1)
    grid = FG.Grids.StructuredGrid(geom, x, x)
    u = rand(N, N); v = rand(N, N); θ = rand(N, N)
    u_rot = rand(N, N); v_rot = rand(N, N)
    ker = CGEF.TopHatKernel()
    scale = 12_000.0
    plan = CGEF.Filtering.plan_filter(grid, ker, scale)
    dplan = CGEF.Derivatives.StencilPlan(grid)

    SUITE["tau_decomposition/192x192/allocating"] =
        BenchmarkTools.@benchmarkable CGEF.Diagnostics.tau_decomposition($u, $v, $grid, $ker, $scale)
    let ws = CGEF.Diagnostics.TauWorkspace(grid)
        SUITE["tau_decomposition!/192x192/workspace-held"] =
            BenchmarkTools.@benchmarkable CGEF.Diagnostics.tau_decomposition!(
                $ws, $u, $v, $grid, $ker, $scale; filter_plan = $plan)
    end

    SUITE["tracer_variance_flux/192x192/allocating"] =
        BenchmarkTools.@benchmarkable CGEF.Diagnostics.tracer_variance_flux(
            $u, $v, $θ, $grid, $ker, $scale)
    let ws = CGEF.Diagnostics.TracerFluxWorkspace(grid), out = zeros(N, N)
        SUITE["tracer_variance_flux!/192x192/workspace-held"] =
            BenchmarkTools.@benchmarkable CGEF.Diagnostics.tracer_variance_flux!(
                $out, $ws, $u, $v, $θ, $grid, $ker, $scale;
                filter_plan = $plan, deriv_plan = $dplan)
    end

    SUITE["compute_Pi_decomposed/192x192/allocating"] =
        BenchmarkTools.@benchmarkable CGEF.Diagnostics.compute_Π_decomposed(
            $u, $v, $u_rot, $v_rot, $grid, $ker, $scale)
    let ws = CGEF.Diagnostics.PiDecomposedWorkspace(grid)
        SUITE["compute_Pi_decomposed!/192x192/workspace-held"] =
            BenchmarkTools.@benchmarkable CGEF.Diagnostics.compute_Π_decomposed!(
                $ws, $u, $v, $u_rot, $v_rot, $grid, $ker, $scale;
                filter_plan = $plan, deriv_plan = $dplan)
    end
end

# Slice axis: many independent problems, one plan each. Ragged point counts, since that is the shape
# the longest-first schedule exists for and an equal-count baseline would hide the imbalance.
let
    nslices = 64
    dx = 1_000.0
    geom = FG.Geometry.CartesianGeometry()
    counts = [1_232 + 47 * i % 2_350 for i in 1:nslices]
    plans = map(counts) do n
        xs = collect(range(0.0, dx * 60; length = n))
        g = FG.Grids.StructuredGrid(geom, xs)
        CGEF.Filtering.plan_filter(g, CGEF.GaussianKernel(), 8_000.0)
    end
    fields = [rand(n) for n in counts]
    outs = [zeros(n) for n in counts]
    SUITE["filter_slices!/64-ragged-slices"] =
        BenchmarkTools.@benchmarkable CGEF.Filtering.filter_slices!($outs, $fields, $plans)
end

# Realistic-scale real-space filtering: a 100 km filter on 1 km data (top-hat radius = 50 grid cells).
# This is the regime where cost is dominated by the filter WIDTH rather than the point count, and where
# an O(N·di_lim·dj_lim) windowed sum stops being usable. Run on a genuinely NONUNIFORM axis — both the
# harder case and the one real swath/observational products actually have.
let
    N = 1_024
    dx = 1_000.0
    geom = FG.Geometry.CartesianGeometry()
    x = collect(0.0:dx:dx*(N - 1)) .+ [0.3 * dx * sin(2.7i) for i in 1:N]
    grid = FG.Grids.StructuredGrid(geom, x, copy(x))
    u = rand(N, N)
    out = zeros(N, N)
    scale = 100_000.0

    plan = CGEF.Filtering.plan_filter(grid, CGEF.TopHatKernel(), scale)
    SUITE["filter_apply!/tophat/1024x1024/100km-on-1km"] =
        BenchmarkTools.@benchmarkable CGEF.Filtering.filter_apply!($out, $u, $plan)

    # The general scattered engine on the same grid/scale, as the reference the prefix-sum path is
    # measured against. Streaming (`NeverCache`), because a materialized neighbour list for a window
    # this wide does not fit any sane memory budget.
    fp_scattered = CGEF.Filtering._build_footprint_scattered(
        grid, CGEF.TopHatKernel(), scale;
        mask_strategy = CGEF.Filtering.Deformable(),
        cache_strategy = CGEF.Filtering.NeverCache(),
    )
    SUITE["filter_apply!/tophat/1024x1024/100km-on-1km/scattered-reference"] =
        BenchmarkTools.@benchmarkable CGEF.Filtering.apply_footprint!(
            $out, $u, $grid, $fp_scattered, CGEF.Filtering.Deformable(), false, false,
        )
end

# Prefix-sum top-hat across the axes that select different branches of its inner loop, at several
# filter widths. A uniform axis takes the constant-window branch; a nonuniform one takes the
# two-pointer walk, which is the branch a K-field batch can share. Both are benchmarked single-field
# and as a 5-field batch, which is the shape `compute_Π!` actually issues.
let
    N = 256
    dx = 1_000.0
    geom = FG.Geometry.CartesianGeometry()
    xr = 0.0:dx:dx*(N - 1)
    xv = collect(xr) .+ [0.3 * dx * sin(2.7i) for i in 1:N]
    ker = CGEF.TopHatKernel()
    K = 5
    fields = ntuple(k -> rand(N, N), K)
    outs = ntuple(_ -> zeros(N, N), K)

    for (axname, ax) in (("uniform", xr), ("nonuniform", xv))
        grid = FG.Grids.StructuredGrid(geom, ax, ax, trues(N, N))
        for scale in (20_000.0, 40_000.0)
            w = round(Int, scale / (2 * dx))
            plan = CGEF.Filtering.plan_filter(
                grid, ker, scale; backend = CGEF.ComputationalBackends.SerialBackend(),
            )
            SUITE["filter_apply!/tophat/$(N)x$(N)/$axname/w$w"] =
                BenchmarkTools.@benchmarkable CGEF.Filtering.filter_apply!($(outs[1]), $(fields[1]), $plan)
            # The batch entry is the one that moves when the support walk is shared across fields.
            SUITE["filter_apply_batch!/tophat/$(N)x$(N)/$axname/w$w/K$K"] =
                BenchmarkTools.@benchmarkable CGEF.Filtering.filter_apply_batch!($outs, $fields, $plan)
        end
    end
end

# Sweep planning: one family against independent per-scale plans. What this measures is the
# grid-determined work (measure prefix scans, extended axis, mask scan) being paid once instead of
# once per scale, and the per-apply scratch being held once instead of `S` times.
let
    N = 512
    dx = 1_000.0
    geom = FG.Geometry.CartesianGeometry()
    x = collect(0.0:dx:dx*(N - 1)) .+ [0.3 * dx * sin(2.7i) for i in 1:N]
    grid = FG.Grids.StructuredGrid(geom, x, x, trues(N, N))
    ker = CGEF.TopHatKernel()
    scales = collect(8_000.0:8_000.0:64_000.0)   # 8 scales
    SER = CGEF.ComputationalBackends.SerialBackend()

    SUITE["plan_filter_sweep/tophat/$(N)x$(N)/8-scales"] =
        BenchmarkTools.@benchmarkable CGEF.Filtering.plan_filter_sweep($grid, $ker, $scales; backend = $SER)
    SUITE["plan_filter/tophat/$(N)x$(N)/8-independent-plans"] =
        BenchmarkTools.@benchmarkable [
            CGEF.Filtering.plan_filter($grid, $ker, s; backend = $SER) for s in $scales
        ]
end

# Spherical Gaussian: the package's primary use case and, at a wide filter, its slowest engine. The
# per-latitude band footprint is O(N·dj_lim·di_lim) and `di_lim` grows as 1/cos(latitude), so the cost
# is set by the polar rows. Recorded at two filter widths so a change to this engine has a baseline.
let
    Nlon, Nlat = 256, 128
    R = 6.371e6
    geom = FG.Geometry.SphericalGeometry(R)
    lon = range(0.0; step = 2π / Nlon, length = Nlon)
    lat = range(deg2rad(-88.0); stop = deg2rad(88.0), length = Nlat)
    grid = FG.Grids.StructuredGrid(geom, lon, lat, trues(Nlon, Nlat))
    u = rand(Nlon, Nlat)
    out = zeros(Nlon, Nlat)
    SER = CGEF.ComputationalBackends.SerialBackend()

    for (name, scale) in (("200km", 200e3), ("800km", 800e3))
        plan = CGEF.Filtering.plan_filter(grid, CGEF.GaussianKernel(), scale; backend = SER)
        SUITE["filter_apply!/gaussian-spherical/$(Nlon)x$(Nlat)/$name"] =
            BenchmarkTools.@benchmarkable CGEF.Filtering.filter_apply!($out, $u, $plan)
    end

    # Top-hat on the same grid takes the prefix-sum engine instead, so the two are directly comparable
    # and show what a fast engine is worth at the same scale.
    plan_th = CGEF.Filtering.plan_filter(grid, CGEF.TopHatKernel(), 800e3; backend = SER)
    SUITE["filter_apply!/tophat-spherical/$(Nlon)x$(Nlat)/800km"] =
        BenchmarkTools.@benchmarkable CGEF.Filtering.filter_apply!($out, $u, $plan_th)
end

# True 3-D. A 3-D top-hat currently walks the inscribed ball at O(w³) because the prefix-sum engine is
# 2-D only; the separable Gaussian is O(N·Σw). Both recorded, since the gap between them is the size
# of the opportunity.
let
    N = 64
    dx = 1_000.0
    geom = FG.Geometry.CartesianGeometry()
    ax = 0.0:dx:dx*(N - 1)
    grid = FG.Grids.StructuredGrid(geom, ax, ax, ax, trues(N, N, N))
    u = rand(N, N, N)
    out = zeros(N, N, N)
    SER = CGEF.ComputationalBackends.SerialBackend()

    for (kname, ker) in (("tophat", CGEF.TopHatKernel()), ("gaussian", CGEF.GaussianKernel()))
        plan = CGEF.Filtering.plan_filter(grid, ker, 12_000.0; backend = SER)
        SUITE["filter_apply!/$kname-3d/$(N)cubed/w6"] =
            BenchmarkTools.@benchmarkable CGEF.Filtering.filter_apply!($out, $u, $plan)
    end
    # The engine the 3-D top-hat prefix sums replace, at the same kernel and scale — the honest
    # before/after pair for that change.
    ball = CGEF.Filtering._build_footprint_nd(grid, CGEF.TopHatKernel(), 12_000.0)
    SUITE["filter_apply!/tophat-3d/$(N)cubed/w6/ball-walk-reference"] =
        BenchmarkTools.@benchmarkable CGEF.Filtering._apply_serial!(
            $out, $u, $grid, $ball, CGEF.Filtering.ZeroFill())
end

# The three comparisons that used to be wall-clock assertions inside `runtests.jl`. They are real
# performance claims and belong here, where a slow machine makes a number worse rather than a test red.
let
    SER = CGEF.ComputationalBackends.SerialBackend()
    geom = FG.Geometry.CartesianGeometry()

    # 1. Batched vs per-field apply on a STREAMING plan, where the per-point neighbour derivation is
    #    what the batch shares. K = 9 mirrors `compute_Π!`'s per-scale product count.
    let N = 96, ker = CGEF.TopHatKernel()
        x_nu = collect(0.0:800.0:(N - 1) * 800.0) .+ [iseven(i) ? 4.0 : -3.5 for i in 1:N]
        grid = FG.Grids.StructuredGrid(geom, x_nu, x_nu, trues(N, N))
        plan = CGEF.Filtering.plan_filter(grid, ker, 15_000.0;
                                          backend = SER, cache_strategy = CGEF.Filtering.NeverCache())
        K = 9
        fields = ntuple(_ -> rand(N, N), K)
        outs = ntuple(_ -> zeros(N, N), K)
        SUITE["filter_apply_batch!/streaming/$(N)x$(N)/K$K"] =
            BenchmarkTools.@benchmarkable CGEF.Filtering.filter_apply_batch!($outs, $fields, $plan)
        SUITE["filter_apply!/streaming/$(N)x$(N)/K$K-separate-calls"] =
            BenchmarkTools.@benchmarkable for k in 1:$K
                CGEF.Filtering.filter_apply!($outs[k], $fields[k], $plan)
            end
    end

    # 2. Separable Gaussian against the general scattered engine on the same physical points.
    let N = 80, ker = CGEF.GaussianKernel(), scale = 15_000.0
        xsR = 0.0:1000.0:(N - 1) * 1000.0
        gf = FG.Grids.StructuredGrid(geom, xsR, xsR, trues(N, N))
        cg = FG.Grids.CurvilinearGrid(geom, [1000.0 * i for i in 0:(N-1), j in 0:(N-1)],
                                      [1000.0 * j for i in 0:(N-1), j in 0:(N-1)], trues(N, N))
        pf = CGEF.Filtering.plan_filter(gf, ker, scale; backend = SER)
        pg = CGEF.Filtering.plan_filter(cg, ker, scale; backend = SER)
        f = rand(N, N); o = zeros(N, N)
        SUITE["filter_apply!/gaussian/$(N)x$(N)/separable"] =
            BenchmarkTools.@benchmarkable CGEF.Filtering.filter_apply!($o, $f, $pf)
        SUITE["filter_apply!/gaussian/$(N)x$(N)/scattered-reference"] =
            BenchmarkTools.@benchmarkable CGEF.Filtering.filter_apply!($o, $f, $pg)
    end
end

# The decompositions against the flux they decompose. Each is dominated by its filter applies, so the
# ratio to `compute_Π!` is the count of applies its algebra needs: the strain/convergence split reads
# the same stress and strain and costs the same, the Favre budget adds the mass-weighted products, and
# the Germano and Helmholtz splits each form three symmetric tensors. A ratio that drifts off that
# count is the signal — an extra pass crept in.
let sgeo = FG.Geometry.SphericalGeometry(), ker = CGEF.GaussianKernel(), ℓ = 200e3
    SER = CGEF.ComputationalBackends.SerialBackend()
    nλ, nφ = 256, 128
    lon = range(0.0; step = 2π / nλ, length = nλ)
    lat = range(deg2rad(-80.0); stop = deg2rad(80.0), length = nφ)
    grid = FG.Grids.StructuredGrid(sgeo, lon, lat, trues(nλ, nφ))
    u = [cos(p) * sin(l) for l in lon, p in lat]
    v = [cos(p) * cos(l) for l in lon, p in lat]
    ρ = fill(2.5, nλ, nφ); P = fill(1.0e5, nλ, nφ)
    pl = CGEF.Filtering.plan_filter(grid, ker, ℓ; backend = SER)
    dp = CGEF.Derivatives.StencilPlan(grid)
    Π = zeros(nλ, nφ)
    wsΠ = CGEF.Diagnostics.ΠWorkspace(grid)
    wsc = CGEF.Diagnostics.ΠWorkspace(grid)
    wd = CGEF.Diagnostics.SphericalPiDecomposedWorkspace(grid)
    wf = CGEF.Diagnostics.SphericalFavreWorkspace(grid)
    wt = CGEF.Diagnostics.Sym3TauWorkspace(grid)

    SUITE["compute_Pi!/spherical/$(nλ)x$(nφ)/plans-held"] =
        BenchmarkTools.@benchmarkable CGEF.Diagnostics.compute_Π!(
            $Π, $u, $v, nothing, $grid, $ker, $ℓ;
            workspace = $wsΠ, filter_plan = $pl, deriv_plan = $dp, backend = $SER)
    SUITE["strain_convergence!/spherical/$(nλ)x$(nφ)"] =
        BenchmarkTools.@benchmarkable CGEF.Diagnostics.compute_Π_strain_convergence!(
            $wsc, $u, $v, $grid, $ker, $ℓ; filter_plan = $pl, deriv_plan = $dp)
    SUITE["Pi_decomposed!/spherical/$(nλ)x$(nφ)"] =
        BenchmarkTools.@benchmarkable CGEF.Diagnostics.compute_Π_decomposed!(
            $wd, $u, $v, $u, $v, $grid, $ker, $ℓ; filter_plan = $pl, deriv_plan = $dp)
    SUITE["compressible_flux!/spherical/$(nλ)x$(nφ)"] =
        BenchmarkTools.@benchmarkable CGEF.Diagnostics.compressible_flux!(
            $wf, $u, $v, $ρ, $P, $grid, $ker, $ℓ; filter_plan = $pl, deriv_plan = $dp)
    SUITE["tau_decomposition!/spherical/$(nλ)x$(nφ)"] =
        BenchmarkTools.@benchmarkable CGEF.Diagnostics.tau_decomposition!(
            $wt, $u, $v, $grid, $ker, $ℓ; filter_plan = $pl)
end

# The same three, over a true-3-D volume, where every tensor carries six components.
let geom = FG.Geometry.CartesianGeometry(), ker = CGEF.GaussianKernel(), ℓ = 4000.0
    SER = CGEF.ComputationalBackends.SerialBackend()
    n = 48
    x = range(0.0, 1000.0 * (n - 1); length = n)
    grid = FG.Grids.StructuredGrid(geom, x, x, x, trues(n, n, n))
    u = [sin(i / 4) * cos(j / 5) for i in 1:n, j in 1:n, k in 1:n]
    v = [cos(i / 3) * sin(k / 6) for i in 1:n, j in 1:n, k in 1:n]
    w = [sin(j / 5) * cos(k / 4) for i in 1:n, j in 1:n, k in 1:n]
    ρ = fill(2.5, n, n, n); P = fill(1.0e5, n, n, n)
    pl = CGEF.Filtering.plan_filter(grid, ker, ℓ; backend = SER)
    dp = CGEF.Derivatives.StencilPlan(grid)
    Π = zeros(n, n, n)
    wsΠ = CGEF.Diagnostics.ΠWorkspace(grid; has_w = true)
    wt = CGEF.Diagnostics.Sym3TauWorkspace(grid)
    wf = CGEF.Diagnostics.Favre3DWorkspace(grid)
    wd = CGEF.Diagnostics.PiDecomposed3DWorkspace(grid)

    SUITE["compute_Pi!/volume-$(n)^3/plans-held"] =
        BenchmarkTools.@benchmarkable CGEF.Diagnostics.compute_Π!(
            $Π, $u, $v, $w, $grid, $ker, $ℓ;
            workspace = $wsΠ, filter_plan = $pl, deriv_plan = $dp, backend = $SER)
    SUITE["tau_decomposition!/volume-$(n)^3"] =
        BenchmarkTools.@benchmarkable CGEF.Diagnostics.tau_decomposition!(
            $wt, $u, $v, $w, $grid, $ker, $ℓ; filter_plan = $pl)
    SUITE["compressible_flux!/volume-$(n)^3"] =
        BenchmarkTools.@benchmarkable CGEF.Diagnostics.compressible_flux!(
            $wf, $u, $v, $w, $ρ, $P, $grid, $ker, $ℓ; filter_plan = $pl, deriv_plan = $dp)
    SUITE["Pi_decomposed!/volume-$(n)^3"] =
        BenchmarkTools.@benchmarkable CGEF.Diagnostics.compute_Π_decomposed!(
            $wd, $u, $v, $w, $u, $v, $w, $grid, $ker, $ℓ; filter_plan = $pl, deriv_plan = $dp)
end

# The node CSR engine, at a pixelization size a global run actually uses. Every flat-cell architecture
# reaches this one engine, so the entries that matter are the plan build (one ball query per cell,
# amortized over the sweep) and the apply (one gather over the stored blocks). A `HEALPixGrid` at
# nside 64 is 49,152 cells; the entry at nside 32 alongside it makes the growth in the plan build
# visible, since that is the part that scales with the neighbour count.
let sgeo = FG.Geometry.SphericalGeometry(6.371e6), ker = CGEF.GaussianKernel()
    SER = CGEF.ComputationalBackends.SerialBackend()
    for ns in (32, 64)
        grid = FG.Grids.HEALPixGrid(sgeo, ns)
        n = length(FG.Grids.mask(grid))
        scale = 4 * 6.371e6 * sqrt(4π / n)          # ~4 cell widths
        plan = CGEF.Filtering.plan_filter(grid, ker, scale; backend = SER)
        f = rand(n); o = zeros(n)
        SUITE["plan_filter/node-csr/healpix-nside$(ns)"] =
            BenchmarkTools.@benchmarkable CGEF.Filtering.plan_filter($grid, $ker, $scale; backend = $SER)
        SUITE["filter_apply!/node-csr/healpix-nside$(ns)"] =
            BenchmarkTools.@benchmarkable CGEF.Filtering.filter_apply!($o, $f, $plan)
    end
end

# Threaded batch barrier cost. A K-field separable batch is K threaded applies, each two `tforeach`
# sweeps, so it crosses `2K` barriers per scale. Fusing to two would need K live full-grid pass buffers,
# so what decides it is the barrier cost against the batch. These two entries measure exactly that: the
# batch, and the same count of empty parallel regions over the same index range.
#
# Requires `using OhMyThreads` for the threaded backend; skipped where the extension is absent.
if Base.get_extension(CGEF, :CoarseGrainingEnergyFluxesOhMyThreadsExt) !== nothing
    let geom = FG.Geometry.CartesianGeometry(), K = 9
        OMT = Base.require(Base.PkgId(
            Base.UUID("67456a42-1dca-4109-a031-0a68de7e3ad5"), "OhMyThreads"))
        for N in (512, 1024)
            ax = range(0.0, 1.0e6; length = N + 1)[1:N]
            grid = FG.Grids.StructuredGrid(geom, ax, ax; periodic = (true, true))
            plan = CGEF.Filtering.plan_filter(grid, CGEF.GaussianKernel(), 6.0e4;
                                              backend = CGEF.ComputationalBackends.ThreadedBackend())
            fields = ntuple(_ -> rand(N, N), K)
            outs = ntuple(_ -> zeros(N, N), K)
            sched = OMT.DynamicScheduler()
            SUITE["filter_apply_batch!/separable-threaded/$(N)x$(N)/K$K"] =
                BenchmarkTools.@benchmarkable CGEF.Filtering.filter_apply_batch!($outs, $fields, $plan)
            SUITE["tforeach/empty-regions/$(N)rows/x$(2K)"] =
                BenchmarkTools.@benchmarkable for _ in 1:$(2K)
                    $OMT.tforeach(_ -> nothing, 1:$N; scheduler = $sched)
                end
        end
    end
end
