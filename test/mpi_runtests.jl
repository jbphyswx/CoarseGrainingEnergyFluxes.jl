# MPI backend test suite — NOT part of `Pkg.test()` (that runs single-process). Run explicitly via:
#
#     mpiexec -n 4 julia --project=test test/mpi_runtests.jl
#
# Compares the `MPIBackend`'s multi-rank `Allreduce!`-recombined result against the `SerialBackend`
# reference on the SAME grid/field, on every rank — validating the "field replicated across ranks,
# disjoint row ownership, sum-reduce" assumption `CoarseGrainingEnergyFluxesMPIExt` documents but
# (until this file existed) had never actually been run under a real MPI runtime.

using MPI: MPI
using Test: Test
using Random: Random
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nproc = MPI.Comm_size(comm)

function _serial_vs_mpi(grid, field, kernel, scale; mask_strategy = CGEF.Filtering.Deformable())
    serial = zeros(size(field))
    CGEF.Filtering.filter_field!(
        serial, field, grid, kernel, scale;
        backend = CGEF.Backends.SerialBackend(), mask_strategy = mask_strategy,
    )
    mpi_out = zeros(size(field))
    CGEF.Filtering.filter_field!(
        mpi_out, field, grid, kernel, scale;
        backend = CGEF.Backends.MPIBackend(), mask_strategy = mask_strategy,
    )
    return serial, mpi_out
end

Test.@testset "MPI backend (rank $rank of $nproc)" begin
    # Cartesian.
    geom = CGEF.CartesianGeometry(1000.0, 1000.0)
    lon = collect(0.0:1000.0:30e3)
    lat = collect(0.0:1000.0:30e3)
    grid = CGEF.StructuredGrid(geom, lon, lat, trues(length(lon), length(lat)))
    # Each rank is a SEPARATE OS process (mpiexec launches independent `julia` instances), so an
    # unseeded `rand()` gives every rank a DIFFERENT field — silently violating the MPIBackend's
    # documented "field replicated across ranks" assumption and making the Allreduce-combined result
    # meaningless. Seed identically on every rank so the field truly is replicated, matching the
    # assumption under test.
    Random.seed!(1234)
    field = rand(length(lon), length(lat))
    serial, mpi_out = _serial_vs_mpi(grid, field, CGEF.TopHatKernel(), 5000.0)
    Test.@test mpi_out ≈ serial

    # Masked Cartesian — exercises Deformable-strategy renormalization across rank boundaries (a
    # masked cell's neighbours may be owned by a different rank than the cell being filtered).
    mask = trues(length(lon), length(lat)); mask[5:8, 5:8] .= false
    mgrid = CGEF.StructuredGrid(geom, lon, lat, mask)
    serial_m, mpi_m = _serial_vs_mpi(mgrid, field, CGEF.GaussianKernel(), 4000.0)
    Test.@test mpi_m ≈ serial_m

    # Periodic spherical — exercises longitude-seam wrapping across rank boundaries.
    sgeom = CGEF.SphericalGeometry(6371000.0)
    slon = deg2rad.(collect(0.0:5.0:355.0))
    slat = deg2rad.(collect(-40.0:5.0:40.0))
    sgrid = CGEF.StructuredGrid(sgeom, slon, slat, trues(length(slon), length(slat)))
    Random.seed!(5678)   # identical across ranks — see the Cartesian case above for why
    sfield = rand(length(slon), length(slat))
    serial_s, mpi_s = _serial_vs_mpi(sgrid, sfield, CGEF.TopHatKernel(), 300e3)
    Test.@test mpi_s ≈ serial_s
end
