# MPIBackend test suite — not part of `Pkg.test()`, which runs in one process. Run with:
#
#     mpiexec -n 4 julia --project=test test/mpi_runtests.jl
#
# Compares the `MPIBackend` result, summed across ranks by `Allreduce!`, with the `SerialBackend` result
# on the same grid and field, on every rank. Every rank holds the whole field; each owns a disjoint set of
# rows or points.

using MPI: MPI
using Test: Test
using Random: Random
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF
using FlowGeometries: FlowGeometries as FG
using FlowTransformBindings: FlowTransformBindings as FTB
using FINUFFT: FINUFFT
using NonuniformFFTs: NonuniformFFTs
using NUFSHT: NUFSHT

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nproc = MPI.Comm_size(comm)

include("reference_filters.jl")

function _serial_vs_mpi(grid, field, kernel, scale; mask_strategy = CGEF.Filtering.Deformable())
    serial = zeros(size(field))
    CGEF.Filtering.filter_field!(
        serial, field, grid, kernel, scale;
        execution = CGEF.ComputationalBackends.SerialBackend(), mask_strategy = mask_strategy,
    )
    mpi_out = zeros(size(field))
    CGEF.Filtering.filter_field!(
        mpi_out, field, grid, kernel, scale;
        execution = CGEF.ComputationalBackends.MPIBackend(), mask_strategy = mask_strategy,
    )
    return serial, mpi_out
end

Test.@testset "MPI backend (rank $rank of $nproc)" begin
    # Cartesian.
    geom = FG.Geometry.CartesianGeometry()
    x = collect(0.0:1000.0:30e3)
    y = collect(0.0:1000.0:30e3)
    grid = FG.Grids.StructuredGrid(geom, x, y, trues(length(x), length(y)))
    # The same seed on every rank gives every rank the same field.
    Random.seed!(1234)
    field = rand(length(x), length(y))
    serial, mpi_out = _serial_vs_mpi(grid, field, CGEF.TopHatKernel(), 5000.0)
    Test.@test mpi_out ≈ serial

    # Masked Cartesian: `Deformable` renormalizes over neighbours another rank owns.
    mask = trues(length(x), length(y)); mask[5:8, 5:8] .= false
    mgrid = FG.Grids.StructuredGrid(geom, x, y, mask)
    serial_m, mpi_m = _serial_vs_mpi(mgrid, field, CGEF.GaussianKernel(), 4000.0)
    Test.@test mpi_m ≈ serial_m

    # Periodic spherical: windows wrap the longitude seam across rank boundaries.
    sgeom = FG.Geometry.SphericalGeometry(6371000.0)
    slon = deg2rad.(collect(0.0:5.0:355.0))
    slat = deg2rad.(collect(-40.0:5.0:40.0))
    sgrid = FG.Grids.StructuredGrid(sgeom, slon, slat, trues(length(slon), length(slat)))
    Random.seed!(5678)
    sfield = rand(length(slon), length(slat))
    serial_s, mpi_s = _serial_vs_mpi(sgrid, sfield, CGEF.TopHatKernel(), 300e3)
    Test.@test mpi_s ≈ serial_s
end

# Each rank runs its stride of the rows, `(j, k)` planes or separable pass lines, and every pass is
# assembled on every rank before the next reads it. A cell is written by one rank and the other ranks add
# zeros, so the result is the serial one; both are the direct sum to the engine's accuracy, round-off for
# a top-hat and, for the Gaussian's separable passes, the square window's truncation, below 1e-9 of the
# kernel's peak.
Test.@testset "MPI real-space engines (rank $rank of $nproc)" begin
    CB = CGEF.ComputationalBackends
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
    for (grid, kern, ℓ, engine, rtol) in cases, st in (CGEF.Filtering.ZeroFill(), CGEF.Filtering.Deformable())
        f = [sin(1.3 * I[1] + 0.7 * get(Tuple(I), 2, 0) + 0.4 * get(Tuple(I), 3, 0)) + 0.2
             for I in CartesianIndices(FG.Grids.size_tuple(grid))]
        plan(b) = CGEF.Filtering.plan_filter(grid, kern, ℓ; execution = b, mask_strategy = st,
                                             method = CGEF.Filtering.RealSpace())
        ps = plan(CB.SerialBackend())
        Test.@test nameof(typeof(ps.footprint)) === engine
        out = CGEF.Filtering.filter_apply!(zero(f), f, plan(CB.MPIBackend()))
        Test.@test out == CGEF.Filtering.filter_apply!(zero(f), f, ps)
        Test.@test _cgef_relerr(out, _cgef_direct_filter(grid, kern, ℓ, f, st)) < rtol
    end
end

# Each rank sweeps its stride of slices from one scratch set, and the Allreduce assembles the whole batch
# on every rank. With one slice more than ranks, a rank runs two slices on the same scratch.
Test.@testset "MPI coarse_grain_batch! (rank $rank of $nproc)" begin
    P = CGEF.Pipeline
    CB = CGEF.ComputationalBackends
    xs = 0.0:1000.0:23e3
    mask = trues(24, 24); mask[8:12, 6:9] .= false
    grid = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), xs, xs, mask)
    scales = [4e3, 8e3]
    kern = CGEF.GaussianKernel()
    Random.seed!(3)   # identical on every rank
    Nt = nproc + 1
    u = randn(24, 24, Nt); v = randn(24, 24, Nt)
    b = P.CoarseGrainBatchResult(grid, length(scales), (Nt,))
    P.coarse_grain_batch!(b, u, v, grid; scales, kernel = kern, execution = CB.MPIBackend())
    for t in 1:Nt
        r = CGEF.coarse_grain(view(u, :, :, t), view(v, :, :, t), grid; scales, kernel = kern,
                              execution = CB.SerialBackend())
        Test.@test b.slices[t].Π == r.Π
        Test.@test b.slices[t].cumulative_energy == r.cumulative_energy
        Test.@test b.slices[t].filtering_spectrum == r.filtering_spectrum
    end
end

# A nonuniform spectral plan under MPI gives each rank a disjoint share of the points: the partial
# analyses add to the whole spectrum and each rank evaluates its own points. Every rank builds the serial
# plan over all the points and holds the same field, so both outputs are the whole filtered field. The
# tolerance is the transform's own accuracy: 1e-8 for a NUFFT, 1e-6 for a NUSHT, whose least-squares fit
# stops at 1e-7.
Test.@testset "MPI spectral filtering (rank $rank of $nproc)" begin
    CB = CGEF.ComputationalBackends
    S = CGEF.Filtering.Spectral()
    g = CGEF.GaussianKernel()
    apply(p, f) = CGEF.Filtering.filter_apply!(similar(f), f, p)
    batched(p, F) = CGEF.Filtering.filter_apply_batched!(similar(F), F, p)
    function mpi_matches_serial(grid, f, ℓ, rtol; kw...)
        ps = CGEF.Filtering.plan_filter(grid, g, ℓ; method = S, execution = CB.SerialBackend(), batch = 2, kw...)
        pm = CGEF.Filtering.plan_filter(grid, g, ℓ; method = S, execution = CB.MPIBackend(), batch = 2, kw...)
        F = hcat(f, f .^ 2)
        Test.@test length(pm.grid_plan.own) == length((rank + 1):nproc:length(f))
        Test.@test isapprox(apply(pm, f), apply(ps, f); rtol)
        Test.@test isapprox(batched(pm, F), batched(ps, F); rtol)
    end
    libs = (FTB.FINUFFTBackend(), FTB.NonuniformFFTsBackend())
    strategies = (CGEF.Filtering.ZeroFill(), CGEF.Filtering.Deformable())

    Random.seed!(11)   # identical on every rank
    n = 240
    px = 10 .* rand(n); py = 8 .* rand(n)
    mask = trues(n); mask[1:15] .= false
    ug = FG.Grids.UnstructuredGrid(FG.Geometry.CartesianGeometry(), px, py, fill(80 / n, n), mask)
    for lib in libs, st in strategies
        mpi_matches_serial(ug, sin.(px) .* cos.(py), 1.5, 1e-8; transform = lib, mask_strategy = st)
    end

    M = 400
    θ = [acos(1 - 2 * (k - 0.5) / M) for k in 1:M]
    φ = [2π * mod(k * (sqrt(5) - 1) / 2, 1) for k in 1:M]
    smask = trues(M); smask[1:20] .= false
    sg = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(1.0), φ, π / 2 .- θ, fill(4π / M, M), smask)
    for lib in libs, st in strategies
        mpi_matches_serial(sg, cos.(θ) .+ sin.(θ) .* cos.(φ), 0.4, 1e-6; nufft = lib, mask_strategy = st)
    end
end

# Two groups of ranks, each on its own communicator and holding a different field: every result is the
# serial one for the group's own field.
Test.@testset "MPI backend on a split communicator (rank $rank of $nproc)" begin
    CB = CGEF.ComputationalBackends
    Test.@test nproc >= 2
    color = rank % 2
    sub = MPI.Comm_split(comm, color, rank)
    be = CB.MPIBackend(; comm = sub)
    geom = FG.Geometry.CartesianGeometry()
    xs = collect(0.0:1000.0:23e3)
    mask = trues(24, 24); mask[8:12, 6:9] .= false
    grid = FG.Grids.StructuredGrid(geom, xs, xs, mask)
    Random.seed!(100 + color)   # identical within a group, different between groups
    field = randn(24, 24)

    serial = zeros(24, 24)
    CGEF.Filtering.filter_field!(serial, field, grid, CGEF.GaussianKernel(), 4000.0; execution = CB.SerialBackend())
    out = zeros(24, 24)
    CGEF.Filtering.filter_field!(out, field, grid, CGEF.GaussianKernel(), 4000.0; execution = be)
    Test.@test out ≈ serial

    ugrid = FG.Grids.UnstructuredGrid(geom, vec([x for x in xs, _ in xs]), vec([y for _ in xs, y in xs]),
                                      fill(1e6, 24 * 24), vec(mask))
    serial_u = zeros(24 * 24)
    CGEF.Filtering.filter_field!(serial_u, vec(field), ugrid, CGEF.GaussianKernel(), 4000.0; execution = CB.SerialBackend())
    out_u = zeros(24 * 24)
    CGEF.Filtering.filter_field!(out_u, vec(field), ugrid, CGEF.GaussianKernel(), 4000.0; execution = be)
    Test.@test out_u ≈ serial_u

    P = CGEF.Pipeline
    scales = [4e3, 8e3]
    Nt = 3
    u = randn(24, 24, Nt); v = randn(24, 24, Nt)
    b = P.CoarseGrainBatchResult(grid, length(scales), (Nt,))
    P.coarse_grain_batch!(b, u, v, grid; scales, kernel = CGEF.GaussianKernel(), execution = be)
    results = [P.CoarseGrainResult(zeros(2), zeros(24, 24, 2), zeros(2), zeros(2), zeros(2)) for _ in 1:Nt]
    P.coarse_grain_slices!(results, [u[:, :, t] for t in 1:Nt], [v[:, :, t] for t in 1:Nt], fill(grid, Nt);
                           scales, kernel = CGEF.GaussianKernel(), execution = be)
    for t in 1:Nt
        r = CGEF.coarse_grain(view(u, :, :, t), view(v, :, :, t), grid; scales, kernel = CGEF.GaussianKernel(),
                              execution = CB.SerialBackend())
        Test.@test b.slices[t].Π == r.Π
        Test.@test b.slices[t].cumulative_energy == r.cumulative_energy
        Test.@test results[t].Π == r.Π
        Test.@test results[t].cumulative_energy == r.cumulative_energy
    end
    MPI.free(sub)
end
