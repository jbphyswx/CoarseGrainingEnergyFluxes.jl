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

function _serial_vs_mpi(grid, field, kernel, scale; mask_strategy = CGEF.Filtering.Deformable())
    serial = zeros(size(field))
    CGEF.Filtering.filter_field!(
        serial, field, grid, kernel, scale;
        backend = CGEF.ComputationalBackends.SerialBackend(), mask_strategy = mask_strategy,
    )
    mpi_out = zeros(size(field))
    CGEF.Filtering.filter_field!(
        mpi_out, field, grid, kernel, scale;
        backend = CGEF.ComputationalBackends.MPIBackend(), mask_strategy = mask_strategy,
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
        ps = CGEF.Filtering.plan_filter(grid, g, ℓ; method = S, backend = CB.SerialBackend(), batch = 2, kw...)
        pm = CGEF.Filtering.plan_filter(grid, g, ℓ; method = S, backend = CB.MPIBackend(), batch = 2, kw...)
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
        mpi_matches_serial(ug, sin.(px) .* cos.(py), 1.5, 1e-8; spectral_backend = lib, mask_strategy = st)
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
