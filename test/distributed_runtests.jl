# DistributedBackend test suite — not part of `Pkg.test()`, which runs in one process. Run with:
#
#     julia --project=test test/distributed_runtests.jl
#
# Adds two worker processes on this machine and compares each `DistributedBackend` plan with the
# `SerialBackend` plan on the same grid and field. Dividing a transform among the workers changes the
# order of its sums and nothing else, so the tolerance is the transform's own accuracy: round-off for an
# FFT, 1e-8 for a nonuniform FFT, and 1e-6 for a NUSHT, whose least-squares fit stops at 1e-7. A
# real-space engine writes each output cell on one worker in the serial order, so it equals the serial
# result exactly.

using Distributed: Distributed
Distributed.addprocs(2; exeflags = ["--project=$(Base.active_project())", "--threads=2"])
Distributed.@everywhere begin
    using SharedArrays: SharedArrays
    using OhMyThreads: OhMyThreads
    using KernelAbstractions: KernelAbstractions as KA
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

include("reference_filters.jl")

const CB = CGEF.ComputationalBackends
const STRATEGIES = (CGEF.Filtering.ZeroFill(), CGEF.Filtering.Deformable())
apply(p, f) = CGEF.Filtering.filter_apply!(similar(f), f, p)
batched(p, F) = CGEF.Filtering.filter_apply_batched!(similar(F), F, p)

# The distributed plan against the serial one: one field, a batch, and a sweep that analyzes once and
# synthesizes each scale.
function distributed_matches_serial(grid, f, F, scales, rtol; kw...)
    g = CGEF.GaussianKernel()
    kw = (; method = CGEF.Filtering.Spectral(), batch = size(F)[end], kw...)
    serial(ℓ) = CGEF.Filtering.plan_filter(grid, g, ℓ; execution = CB.SerialBackend(), kw...)
    pd = CGEF.Filtering.plan_filter(grid, g, first(scales); execution = CB.DistributedBackend(), kw...)
    Test.@test isapprox(apply(pd, f), apply(serial(first(scales)), f); rtol)
    Test.@test isapprox(batched(pd, F), batched(serial(first(scales)), F); rtol)
    plans = CGEF.Filtering.plan_filter_sweep(grid, g, scales; execution = CB.DistributedBackend(), kw...)
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
    P, B = FG.Grids.Periodic(), FG.Grids.Bounded()
    for (topology, mask) in (((P, P), trues(32, 24)), ((B, B), hole), ((P, B), hole)), st in STRATEGIES
        grid = FG.Grids.StructuredGrid(geom, x, y, mask; topology)
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
        pd = distributed_matches_serial(ug, f, hcat(f, f .^ 2), [1.5, 2.0], 1e-8; transform = lib,
                                        mask_strategy = st)
        Test.@test pd.grid_plan.workers == Distributed.workers()
    end
end

# Each worker sweeps a contiguous block of slices from one scratch set into the shared result, so every
# slice equals the serial sweep of that slice.
Test.@testset "Distributed coarse_grain_batch! on $(Distributed.nworkers()) workers" begin
    P = CGEF.Pipeline
    xs = 0.0:1000.0:23e3
    mask = trues(24, 24); mask[8:12, 6:9] .= false
    grid = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), xs, xs, mask)
    scales = [4e3, 8e3]
    kern = CGEF.GaussianKernel()
    Random.seed!(3)
    Nt = 5
    u = randn(24, 24, Nt); v = randn(24, 24, Nt)
    b = P.CoarseGrainBatchResult(grid, length(scales), (Nt,); execution = CB.DistributedBackend())
    P.coarse_grain_batch!(b, u, v, grid; scales, kernel = kern, execution = CB.DistributedBackend())
    for t in 1:Nt
        r = CGEF.coarse_grain(view(u, :, :, t), view(v, :, :, t), grid; scales, kernel = kern,
                              execution = CB.SerialBackend())
        Test.@test b.slices[t].Π == r.Π
        Test.@test b.slices[t].cumulative_energy == r.cumulative_energy
        Test.@test b.slices[t].filtering_spectrum == r.filtering_spectrum
    end
end

# One field's sweep on the workers: the exact filtering spectrum reads each scale's window masses from the footprint
# the caller holds and applies its rate plans across the workers, each output cell written in the serial order.
Test.@testset "Distributed coarse_grain's exact filtering spectrum on $(Distributed.nworkers()) workers" begin
    xs = 0.0:1000.0:23e3
    mask = trues(24, 24); mask[8:12, 6:9] .= false
    grid = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), xs, xs, mask)
    Random.seed!(7)
    u = randn(24, 24); v = randn(24, 24)
    for kern in (CGEF.GaussianKernel(), CGEF.Kernels.SmoothHatKernel()), st in STRATEGIES
        kw = (; scales = [3e3, 6e3], kernel = kern, mask_strategy = st, spectrum = CGEF.Diagnostics.ForceSpectrum(),
              scale_derivative = CGEF.Diagnostics.ExactScaleDerivative())
        Test.@test CGEF.coarse_grain(u, v, grid; kw..., execution = CB.DistributedBackend()).filtering_spectrum ==
                   CGEF.coarse_grain(u, v, grid; kw..., execution = CB.SerialBackend()).filtering_spectrum
    end
end

# The real-space engines divide their rows, `(j, k)` planes or pass lines among the workers. Each is the
# direct sum to the engine's accuracy: round-off for a top-hat, and for the Gaussian's separable passes
# the square window's truncation, below 1e-9 of the kernel's peak.
Test.@testset "Distributed real-space engines on $(Distributed.nworkers()) workers" begin
    cart = FG.Geometry.CartesianGeometry()
    m2 = trues(16, 8); m2[5:8, 3:5] .= false
    g2 = FG.Grids.StructuredGrid(cart, 0.0:1.0:15.0, 0.0:0.7:4.9, m2)
    m3 = trues(6, 5, 4); m3[2:3, 2:3, 2] .= false
    g3 = FG.Grids.StructuredGrid(cart, 0.0:1.0:5.0, 0.0:0.7:2.8, 0.0:0.9:2.7, m3)
    g1 = FG.Grids.StructuredGrid(cart, 0.0:1.0:23.0)
    th, ga = CGEF.TopHatKernel(), CGEF.GaussianKernel()
    cases = ((g2, th, 4.8, :PrefixSumTopHatPlan, 1e-11), (g2, ga, 4.8, :SeparableFootprint, 1e-8),
             (g3, th, 2.9, :PrefixSumTopHat3DPlan, 1e-11), (g3, ga, 2.9, :SeparableFootprintND, 1e-8),
             (g1, ga, 4.8, :SeparableFootprintND, 1e-8))
    for (grid, kern, ℓ, engine, rtol) in cases, st in STRATEGIES
        f = [sin(1.3 * I[1] + 0.7 * get(Tuple(I), 2, 0) + 0.4 * get(Tuple(I), 3, 0)) + 0.2
             for I in CartesianIndices(FG.Grids.size_tuple(grid))]
        plan(b) = CGEF.Filtering.plan_filter(grid, kern, ℓ; execution = b, mask_strategy = st,
                                             method = CGEF.Filtering.RealSpace())
        ps = plan(CB.SerialBackend())
        Test.@test nameof(typeof(ps.footprint)) === engine
        out = apply(plan(CB.DistributedBackend()), f)
        Test.@test out == apply(ps, f)
        Test.@test _cgef_relerr(out, _cgef_direct_filter(grid, kern, ℓ, f, st)) < rtol
    end
end

# Every real-space engine divided among the workers on a serial, a threaded and a device inner backend, the
# plan applied twice: equal to the serial plan, and on the device to the whole-grid device plan.
Test.@testset "Distributed shares on every engine and inner backend" begin
    dev = CB.GPUBackend(KA.CPU())
    for (grid, kern, ℓ, kw) in _cgef_share_cases(), st in STRATEGIES
        f = _cgef_share_field(size(FG.Grids.mask(grid)))
        plan(b) = CGEF.Filtering.plan_filter(grid, kern, ℓ; execution = b, mask_strategy = st, kw...)
        ref, devref = apply(plan(CB.SerialBackend()), f), apply(plan(dev), f)
        for (inner, want) in ((CB.SerialBackend(), ref), (CB.ThreadedBackend(), ref), (dev, devref))
            p = plan(CB.DistributedBackend(inner))
            Test.@test apply(p, f) == want
            Test.@test apply(p, f) == want
        end
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
        Test.@test pd.grid_plan.plan.workers == Distributed.workers()
        Test.@test pd.grid_plan.batched.workers == Distributed.workers()
    end
end
