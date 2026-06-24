# Benchmark suite (PkgBenchmark-compatible: defines a global `SUITE::BenchmarkGroup`).
# Minimal for now; expanded alongside the performance work (plan reuse, batching, backends).

using BenchmarkTools: BenchmarkTools
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF

const SUITE = BenchmarkTools.BenchmarkGroup()

let
    N = 128
    dx = 1_000.0
    geom = CGEF.CartesianGeometry(dx, dx)
    x = collect(0.0:dx:dx*(N - 1))
    y = collect(0.0:dx:dx*(N - 1))
    grid = CGEF.StructuredGrid(geom, x, y, trues(N, N))
    u = rand(N, N)
    v = rand(N, N)
    out = zeros(N, N)
    Π = zeros(N, N)
    scale = 10_000.0

    SUITE["filter_field!/tophat/128x128"] =
        BenchmarkTools.@benchmarkable CGEF.filter_field!($out, $u, $grid, CGEF.TopHatKernel(), $scale)
    SUITE["compute_Pi!/tophat/128x128"] =
        BenchmarkTools.@benchmarkable CGEF.compute_Π!($Π, $u, $v, nothing, $grid, CGEF.TopHatKernel(), $scale)
end
