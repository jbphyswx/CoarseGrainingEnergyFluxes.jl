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
    result = CGEF.coarse_grain(u, v, grid; scales = scales, kernel = ker)
    SUITE["coarse_grain!/tophat/128x128/4-scales/plans-held"] =
        BenchmarkTools.@benchmarkable CGEF.Pipeline.coarse_grain!(
            $result, $u, $v, $grid; scales = $scales, kernel = $ker,
            workspace = $ws, filter_plans = $plans, deriv_plan = $dplan,
        )
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
