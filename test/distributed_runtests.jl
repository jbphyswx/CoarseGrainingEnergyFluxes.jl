# DistributedBackend test suite — not part of `Pkg.test()`, which runs in one process. Run with:
#
#     julia --project=test test/distributed_runtests.jl
#
# Adds two worker processes on this machine and compares each `DistributedBackend` plan with the
# `SerialBackend` plan on the same grid and field. Dividing a transform among the workers changes the
# order of its sums and nothing else, so the tolerance is the transform's own accuracy: round-off for an
# FFT, 1e-8 for a nonuniform FFT, and 1e-6 for a NUSHT, whose least-squares fit stops at 1e-7.

using Distributed: Distributed
Distributed.addprocs(2; exeflags = "--project=$(Base.active_project())")
Distributed.@everywhere begin
    using SharedArrays: SharedArrays
    using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF
    using FlowGeometries: FlowGeometries as FG
    using FlowTransformBindings: FlowTransformBindings as FTB
    using FFTW: FFTW
    using FINUFFT: FINUFFT
    using NonuniformFFTs: NonuniformFFTs
    using NUFSHT: NUFSHT
end
using Test: Test
using Random: Random

const CB = CGEF.ComputationalBackends
const STRATEGIES = (CGEF.Filtering.ZeroFill(), CGEF.Filtering.Deformable())
apply(p, f) = CGEF.Filtering.filter_apply!(similar(f), f, p)
batched(p, F) = CGEF.Filtering.filter_apply_batched!(similar(F), F, p)

# The distributed plan against the serial one: one field, a batch, and a sweep that analyzes once and
# synthesizes each scale.
function distributed_matches_serial(grid, f, F, scales, rtol; kw...)
    g = CGEF.GaussianKernel()
    kw = (; method = CGEF.Filtering.Spectral(), batch = size(F)[end], kw...)
    serial(ℓ) = CGEF.Filtering.plan_filter(grid, g, ℓ; backend = CB.SerialBackend(), kw...)
    pd = CGEF.Filtering.plan_filter(grid, g, first(scales); backend = CB.DistributedBackend(), kw...)
    Test.@test isapprox(apply(pd, f), apply(serial(first(scales)), f); rtol)
    Test.@test isapprox(batched(pd, F), batched(serial(first(scales)), F); rtol)
    plans = CGEF.Filtering.plan_filter_sweep(grid, g, scales; backend = CB.DistributedBackend(), kw...)
    F̂ = CGEF.Filtering.filter_analyze!(CGEF.Filtering.analyze_buffer(plans[1], f), f, plans[1])
    for (p, ℓ) in zip(plans, scales)
        Test.@test isapprox(CGEF.Filtering.filter_synthesize!(similar(f), F̂, p), apply(serial(ℓ), f); rtol)
    end
    return pd
end

Test.@testset "Distributed FFT spectral filtering on $(Distributed.nworkers()) workers" begin
    geom = FG.Geometry.CartesianGeometry()
    x = 0.0:1.0:31.0
    y = 0.0:0.7:(0.7 * 23)
    hole = trues(32, 24); hole[12:16, 9:12] .= false
    f = [sin(0.4xi) * cos(0.3yj) + 0.02xi for xi in x, yj in y]
    F = cat(f, f .^ 2; dims = 3)
    for (periodic, mask) in (((true, true), trues(32, 24)), ((false, false), hole), ((true, false), hole)),
        st in STRATEGIES
        grid = FG.Grids.StructuredGrid(geom, x, y, mask; periodic = periodic)
        pd = distributed_matches_serial(grid, f, F, [4.0, 6.0], 1e-12; mask_strategy = st)
        Test.@test pd.grid_plan.workers == Distributed.workers()
    end
end

Test.@testset "Distributed nonuniform-FFT spectral filtering on $(Distributed.nworkers()) workers" begin
    Random.seed!(11)
    n = 240
    px = 10 .* rand(n); py = 8 .* rand(n)
    mask = trues(n); mask[1:15] .= false
    ug = FG.Grids.UnstructuredGrid(FG.Geometry.CartesianGeometry(), px, py, fill(80 / n, n), mask)
    f = sin.(px) .* cos.(py)
    for lib in (FTB.FINUFFTBackend(), FTB.NonuniformFFTsBackend()), st in STRATEGIES
        pd = distributed_matches_serial(ug, f, hcat(f, f .^ 2), [1.5, 2.0], 1e-8; spectral_backend = lib,
                                        mask_strategy = st)
        Test.@test pd.grid_plan.workers == Distributed.workers()
        Test.@test vcat(pd.grid_plan.blocks...) == 1:n
    end
end

Test.@testset "Distributed NUSHT spectral filtering on $(Distributed.nworkers()) workers" begin
    M = 400
    θ = [acos(1 - 2 * (k - 0.5) / M) for k in 1:M]
    φ = [2π * mod(k * (sqrt(5) - 1) / 2, 1) for k in 1:M]
    mask = trues(M); mask[1:20] .= false
    sg = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(1.0), φ, π / 2 .- θ, fill(4π / M, M), mask)
    f = cos.(θ) .+ sin.(θ) .* cos.(φ)
    for lib in (FTB.FINUFFTBackend(), FTB.NonuniformFFTsBackend()), st in STRATEGIES
        pd = distributed_matches_serial(sg, f, hcat(f, f .^ 2), [0.4, 0.6], 1e-6; nufft = lib, mask_strategy = st)
        dp = pd.grid_plan.plan
        Test.@test dp.workers == Distributed.workers()
        Test.@test vcat(dp.blocks...) == 1:M
        Test.@test pd.grid_plan.batched.workers == Distributed.workers()
    end
end
