using Test: Test
using StaticArrays: StaticArrays as SA
using Aqua: Aqua
using ExplicitImports: ExplicitImports as EI
using JET: JET
using FFTW: FFTW  # triggers the spectral-filtering extension
using FINUFFT: FINUFFT  # triggers the scattered-Cartesian spectral extension
using OhMyThreads: OhMyThreads  # triggers the threaded-backend extension
using Distributed: Distributed  # with SharedArrays, triggers the distributed-backend extension
using SharedArrays: SharedArrays
using KernelAbstractions: KernelAbstractions as KA  # triggers the GPU backend extension (CPU device here)
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF

Test.@testset "CoarseGrainingEnergyFluxes.jl" begin

    # -----------------------------------------------------------------------
    # Code quality / hygiene gates (house style: Aqua + ExplicitImports + JET)
    # -----------------------------------------------------------------------
    Test.@testset "Aqua" begin
        Aqua.test_all(CGEF; ambiguities = false, unbound_args = (VERSION >= v"1.12"))
    end

    Test.@testset "Explicit imports (core)" begin
        # Core module + submodules: no bare `using` (no implicit imports) and no stale explicit
        # imports — the strict qualified-import policy.
        # TODO(Phase 5+): add per-extension checks (EI.check_no_implicit_imports(ext)) as each
        # backend extension is rewritten and pulled into the test environment.
        Test.@test (EI.check_no_implicit_imports(CGEF); true)
        Test.@test (EI.check_no_stale_explicit_imports(CGEF); true)
        # Per-extension checks (each loaded backend extension must also be import-clean).
        for extname in (
            :CoarseGrainingEnergyFluxesFFTWExt,
            :CoarseGrainingEnergyFluxesFINUFFTExt,
            :CoarseGrainingEnergyFluxesOhMyThreadsExt,
            :CoarseGrainingEnergyFluxesDistributedExt,
            :CoarseGrainingEnergyFluxesGPUExt,
        )
            ext = Base.get_extension(CGEF, extname)
            ext === nothing && continue
            Test.@test (EI.check_no_implicit_imports(ext); true)
            Test.@test (EI.check_no_stale_explicit_imports(ext); true)
        end
    end

    Test.@testset "JET type stability (hot path)" begin
        geom = CGEF.CartesianGeometry(1000.0, 1000.0)
        lon = collect(0.0:1000.0:20e3)
        lat = collect(0.0:1000.0:20e3)
        grid = CGEF.StructuredGrid(geom, lon, lat, trues(length(lon), length(lat)))
        field = rand(length(lon), length(lat))
        out = zeros(size(field))
        kern = CGEF.TopHatKernel()
        scale = 5000.0

        # Footprint build + the convolution apply must be type-stable.
        JET.@test_opt CGEF.Filtering.build_footprint(grid, kern, scale)
        fp = CGEF.Filtering.build_footprint(grid, kern, scale)
        JET.@test_opt CGEF.Filtering.apply_footprint!(out, field, grid, fp, CGEF.Deformable(), false)
        JET.@test_opt CGEF.Filtering.apply_footprint!(out, field, grid, fp, CGEF.ZeroFill(), false)

        # The serial public entry point is type-stable too.
        JET.@test_opt CGEF.filter_field!(out, field, grid, kern, scale; backend = CGEF.SerialBackend())
    end

    # Coordinate system and distance tests
    Test.@testset "Geometry" begin
        # 2D Cartesian
        geom_cart = CGEF.CartesianGeometry(1000.0, 1000.0)
        p1 = SA.SVector{2,Float64}(0.0, 0.0)
        p2 = SA.SVector{2,Float64}(3000.0, 4000.0)
        Test.@test CGEF.distance(geom_cart, p1, p2) ≈ 5000.0
        Test.@test CGEF.area_element(geom_cart) ≈ 1000.0 * 1000.0

        # Spherical
        geom_sph = CGEF.SphericalGeometry(6371000.0)
        # London (0.1278 W, 51.5074 N) to Paris (2.3522 E, 48.8566 N)
        # Coordinates in radians
        london = SA.SVector{2,Float64}(deg2rad(-0.1278), deg2rad(51.5074))
        paris  = SA.SVector{2,Float64}(deg2rad(2.3522), deg2rad(48.8566))
        d_km = CGEF.distance(geom_sph, london, paris) / 1000.0
        Test.@test 340.0 < d_km < 350.0 # Paris-London ≈ 344 km

        # Coordinate projection conversions: to and from planetary Cartesian
        u_east, u_north = 10.0, -5.0
        λ, φ = deg2rad(-122.0), deg2rad(38.0) # San Francisco coords
        p_vel = CGEF.to_planetary_cartesian(geom_sph, u_east, u_north, λ, φ)
        Test.@test length(p_vel) == 3

        l_vel = CGEF.from_planetary_cartesian(geom_sph, p_vel[1], p_vel[2], p_vel[3], λ, φ)
        Test.@test l_vel[1] ≈ u_east
        Test.@test l_vel[2] ≈ u_north
        Test.@test abs(l_vel[3]) < 1e-12
    end

    # Kernel shape evaluation and support range tests
    Test.@testset "Kernels" begin
        th = CGEF.TopHatKernel()
        g  = CGEF.GaussianKernel()            # default Pope convention, α = 6
        g4 = CGEF.GaussianKernel(; α = 4.0)   # FlowSieve convention
        ss = CGEF.SharpSpectralKernel()

        # Width 100 km
        ℓ = 100000.0
        Test.@test CGEF.kernel_weight(th, 10000.0, ℓ) == 1.0
        Test.@test CGEF.kernel_weight(th, 60000.0, ℓ) == 0.0
        Test.@test CGEF.kernel_radius(th, ℓ) == ℓ / 2

        # Gaussian: exponent is configurable; default 6 (Pope), 4 reproduces FlowSieve
        Test.@test CGEF.kernel_weight(g, 0.0, ℓ) == 1.0
        Test.@test CGEF.kernel_weight(g, ℓ, ℓ) ≈ exp(-6.0)
        Test.@test CGEF.kernel_weight(g4, ℓ, ℓ) ≈ exp(-4.0)

        # Gaussian footprint truncates where the weight is negligible (~2ℓ for α=6, not the old 3ℓ)
        r = CGEF.kernel_radius(g, ℓ)
        Test.@test 1.5ℓ < r < 2.5ℓ
        Test.@test CGEF.kernel_weight(g, r, ℓ) < 1e-9
    end

    # Grids constructor and area calculations
    Test.@testset "Grids" begin
        geom = CGEF.CartesianGeometry(2000.0, 2000.0)
        lon = collect(0.0:2000.0:20000.0) # 11 points
        lat = collect(0.0:2000.0:10000.0) # 6 points
        mask = trues(11, 6) # active water grid

        grid = CGEF.StructuredGrid(geom, lon, lat, mask)
        Test.@test CGEF.size_tuple(grid) == (11, 6)
        Test.@test CGEF.area(grid, 2, 2) == 2000.0 * 2000.0
        Test.@test CGEF.coords(grid, 2, 3) == SA.SVector{2,Float64}(2000.0, 4000.0)

        # CurvilinearGrid coords bug test: verify i,j indices are used correctly
        # Create non-square grid to catch index swapping bugs
        lon_m = [Float64(i) for i in 1:10, j in 1:5]  # 10x5
        lat_m = [Float64(j*10) for i in 1:10, j in 1:5]  # lat varies with j
        areas_m = ones(10, 5)
        mask_m = trues(10, 5)
        cgrid = CGEF.CurvilinearGrid(geom, lon_m, lat_m, areas_m, mask_m)

        # coords(i,j) should return (lon[i,j], lat[i,j])
        pt = CGEF.coords(cgrid, 5, 3)
        Test.@test pt[1] == 5.0  # lon[5,3] = 5
        Test.@test pt[2] == 30.0 # lat[5,3] = 30

        # This test catches the bug where lat[j,j] was used instead of lat[i,j]
        pt_corner = CGEF.coords(cgrid, 10, 5)
        Test.@test pt_corner[1] == 10.0  # lon[10,5]
        Test.@test pt_corner[2] == 50.0 # lat[10,5], not lat[5,5]=50 vs lat[10,10] error
    end

    # physical-space filtering algorithms
    Test.@testset "Filtering" begin
        geom = CGEF.CartesianGeometry(100.0, 100.0)
        lon = collect(0.0:100.0:1000.0) # 11 points
        lat = collect(0.0:100.0:1000.0) # 11 points
        mask = trues(11, 11)
        grid = CGEF.StructuredGrid(geom, lon, lat, mask)

        # Constant field filtering must return exactly the same constant
        field = fill(42.0, 11, 11)
        out = zeros(11, 11)
        CGEF.filter_field!(out, field, grid, CGEF.TopHatKernel(), 300.0)

        # Wet cells must have the exact filtered value (42.0)
        Test.@test out[5, 5] ≈ 42.0

        # Test division by zero protection with single-latitude grid
        # This catches the InexactError: Int64(Inf) bug
        geom_sph = CGEF.SphericalGeometry(6371000.0)
        lon_sph = collect(0.0:5.0:355.0)
        lat_sph = [0.0]  # Single latitude
        mask_sph = trues(length(lon_sph), 1)
        grid_sph = CGEF.StructuredGrid(geom_sph, deg2rad.(lon_sph), deg2rad.(lat_sph), mask_sph)

        field_sph = rand(length(lon_sph), 1)
        out_sph = zeros(length(lon_sph), 1)

        # This should not throw InexactError
        Test.@test_nowarn CGEF.filter_field!(out_sph, field_sph, grid_sph, CGEF.TopHatKernel(), 1e6)
    end

    # Execution backend lattice (Backends.jl)
    Test.@testset "Backends" begin
        # resolve_backend returns INSTANCES; AutoBackend picks a concrete local backend
        Test.@test CGEF.Backends.resolve_backend(CGEF.SerialBackend()) === CGEF.SerialBackend()
        Test.@test CGEF.Backends.resolve_backend(CGEF.AutoBackend()) isa Union{CGEF.SerialBackend, CGEF.ThreadedBackend}

        # distribution wrappers are parametric over an inner local backend
        Test.@test CGEF.DistributedBackend(CGEF.SerialBackend()) isa CGEF.DistributedBackend
        Test.@test CGEF.MPIBackend() isa CGEF.MPIBackend
        Test.@test CGEF.DistributedBackend().inner === CGEF.SerialBackend()
        Test.@test CGEF.local_backend(CGEF.DistributedBackend(CGEF.ThreadedBackend())) === CGEF.ThreadedBackend()
        Test.@test CGEF.is_distributed(CGEF.MPIBackend()) && !CGEF.is_distributed(CGEF.SerialBackend())

        geom = CGEF.CartesianGeometry(1000.0, 1000.0)
        lon = collect(0.0:1000.0:20e3)
        lat = collect(0.0:1000.0:20e3)
        grid = CGEF.StructuredGrid(geom, lon, lat, trues(length(lon), length(lat)))
        field = rand(length(lon), length(lat))
        out_serial = zeros(size(field))
        out_default = zeros(size(field))
        CGEF.filter_field!(out_serial, field, grid, CGEF.TopHatKernel(), 5000.0; backend = CGEF.SerialBackend())
        CGEF.filter_field!(out_default, field, grid, CGEF.TopHatKernel(), 5000.0)  # AutoBackend
        Test.@test out_serial ≈ out_default

        # A backend whose extension is not loaded errors helpfully (MPI is intentionally not loaded
        # in the test env — it isn't runnable here — so its hook is still the informative stub).
        Test.@test_throws ArgumentError CGEF.filter_field!(
            out_serial, field, grid, CGEF.TopHatKernel(), 5000.0; backend = CGEF.MPIBackend(),
        )
    end

    # Reusable filter plans + batched multi-field filtering
    Test.@testset "Filter plans & batching" begin
        geom = CGEF.CartesianGeometry(1000.0, 1000.0)
        lon = collect(0.0:1000.0:30e3)
        lat = collect(0.0:1000.0:30e3)
        grid = CGEF.StructuredGrid(geom, lon, lat, trues(length(lon), length(lat)))
        u = rand(length(lon), length(lat))
        v = rand(length(lon), length(lat))
        kern = CGEF.TopHatKernel()
        scale = 5000.0

        # A prebuilt plan applied to a field matches a direct filter_field! call. Pin SerialBackend so
        # the footprint-based PhysicalFilterPlan (whose reuse/allocation we assert below) is built
        # regardless of how many threads the suite happens to run with.
        plan = CGEF.plan_filter(grid, kern, scale; backend = CGEF.SerialBackend())
        out_plan = zeros(size(u))
        out_direct = zeros(size(u))
        CGEF.filter_apply!(out_plan, u, plan)
        CGEF.filter_field!(out_direct, u, grid, kern, scale)
        Test.@test out_plan ≈ out_direct

        # Batched filter_fields! matches per-field filtering.
        ou = zeros(size(u)); ov = zeros(size(v))
        CGEF.filter_fields!((ou, ov), (u, v), grid, kern, scale)
        ru = zeros(size(u)); rv = zeros(size(v))
        CGEF.filter_field!(ru, u, grid, kern, scale)
        CGEF.filter_field!(rv, v, grid, kern, scale)
        Test.@test ou ≈ ru
        Test.@test ov ≈ rv

        # Reapplying a prebuilt plan must NOT rebuild the footprint (a rebuild would allocate the
        # offset/weight vectors, ~kBs); a reused plan allocates essentially nothing.
        CGEF.filter_apply!(out_plan, u, plan)  # warm up
        Test.@test (@allocated CGEF.filter_apply!(out_plan, u, plan)) < 256
    end

    # Spectral (FFT) filtering on a uniform doubly-periodic Cartesian grid
    Test.@testset "Spectral FFTW filtering" begin
        N = 32
        dx = 1.0
        geom = CGEF.CartesianGeometry(dx, dx)
        x = collect(0.0:dx:dx*(N - 1))
        y = collect(0.0:dx:dx*(N - 1))
        grid = CGEF.StructuredGrid(geom, x, y, trues(N, N); periodic = (true, true))
        L = N * dx
        g = CGEF.GaussianKernel()  # α = 6
        ℓ = 4.0

        # A pure Fourier mode is an eigenfunction of the filter: out = Ĝ(k)·field, exactly.
        m = 3
        kx0 = 2π * m / L
        field = Float64[cos(kx0 * xi) for xi in x, _ in y]
        out = zeros(N, N)
        CGEF.filter_field!(out, field, grid, g, ℓ; method = CGEF.Spectral())
        Test.@test out ≈ exp(-kx0^2 * ℓ^2 / 24) .* field rtol = 1e-10  # Gaussian α=6 transfer

        # DC (constant field) is preserved, Ĝ(0) = 1.
        cfield = fill(2.5, N, N)
        cout = zeros(N, N)
        CGEF.filter_field!(cout, cfield, grid, g, ℓ; method = CGEF.Spectral())
        Test.@test cout ≈ cfield

        # Sharp spectral cutoff: a mode below k_c passes, above k_c is removed.
        ss = CGEF.SharpSpectralKernel()
        sout = zeros(N, N)
        CGEF.filter_field!(sout, field, grid, ss, L / 8; method = CGEF.Spectral())  # k_c = 8π/L > 6π/L
        Test.@test sout ≈ field rtol = 1e-10
        CGEF.filter_field!(sout, field, grid, ss, L; method = CGEF.Spectral())       # k_c = π/L < 6π/L
        Test.@test maximum(abs, sout) < 1e-10

        # TopHat spectral is unsupported (Airy ringing) and errors with guidance.
        Test.@test_throws ArgumentError CGEF.filter_field!(out, field, grid, CGEF.TopHatKernel(), ℓ; method = CGEF.Spectral())

        # Non-periodic grid: spectral FFT must refuse.
        npgrid = CGEF.StructuredGrid(geom, x, y, trues(N, N))  # periodic = (false, false)
        Test.@test_throws ArgumentError CGEF.filter_field!(out, field, npgrid, g, ℓ; method = CGEF.Spectral())
    end

    # Scattered-Cartesian spectral filtering (FINUFFT): on a uniform periodic lattice it must
    # reproduce the FFTW result, and it must preserve the mean of a constant field.
    Test.@testset "Spectral FINUFFT filtering" begin
        Nx, Ny = 32, 24
        dx = dy = 1.0
        geom = CGEF.CartesianGeometry(dx, dy)
        x = collect(0.0:dx:dx*(Nx - 1)); y = collect(0.0:dy:dy*(Ny - 1))
        u = [sin(2π*xi/(Nx*dx)) + 0.5cos(4π*yj/(Ny*dy)) + 0.3sin(6π*xi/(Nx*dx)) for xi in x, yj in y]
        g = CGEF.GaussianKernel(); ℓ = 4.0

        # FFTW reference on the structured grid.
        sg = CGEF.StructuredGrid(geom, x, y, trues(Nx, Ny); periodic = (true, true))
        outf = zeros(Nx, Ny)
        CGEF.filter_field!(outf, u, sg, g, ℓ; method = CGEF.Spectral())

        # The same points as a scattered (unstructured) grid; FINUFFT spectral filter.
        ptsx = vec([xi for xi in x, _ in y]); ptsy = vec([yj for _ in x, yj in y])
        ug = CGEF.UnstructuredGrid(geom, ptsx, ptsy, fill(dx*dy, Nx*Ny), trues(Nx*Ny), Vector{Vector{Int}}())
        outu = zeros(Nx*Ny)
        CGEF.filter_field!(outu, vec(u), ug, g, ℓ; method = CGEF.Spectral())
        Test.@test reshape(outu, Nx, Ny) ≈ outf atol = 1e-7

        # Constant field ⇒ mean preserved (Ĝ(0)=1) for the scattered transform.
        cout = zeros(Nx*Ny)
        CGEF.filter_field!(cout, fill(3.7, Nx*Ny), ug, g, ℓ; method = CGEF.Spectral())
        Test.@test all(≈(3.7; atol = 1e-6), cout)

        # TopHat spectral still errors (shared transfer function).
        Test.@test_throws ArgumentError CGEF.filter_field!(outu, vec(u), ug, CGEF.TopHatKernel(), ℓ; method = CGEF.Spectral())
    end

    # Threaded backend must agree with serial EXACTLY (shared footprint engine), including masking
    # and periodic wrapping — the old hand-rolled threaded path silently disagreed on periodicity.
    Test.@testset "Threaded backend (OhMyThreads)" begin
        geom = CGEF.CartesianGeometry(1000.0, 1000.0)
        lon = collect(0.0:1000.0:30e3)
        lat = collect(0.0:1000.0:30e3)
        u = rand(length(lon), length(lat))

        grid = CGEF.StructuredGrid(geom, lon, lat, trues(length(lon), length(lat)))
        for strat in (CGEF.Deformable(), CGEF.ZeroFill())
            os = zeros(size(u)); ot = zeros(size(u))
            CGEF.filter_field!(os, u, grid, CGEF.TopHatKernel(), 5000.0; backend = CGEF.SerialBackend(), mask_strategy = strat)
            CGEF.filter_field!(ot, u, grid, CGEF.TopHatKernel(), 5000.0; backend = CGEF.ThreadedBackend(), mask_strategy = strat)
            Test.@test ot ≈ os
        end

        # Masked Cartesian
        mask = trues(length(lon), length(lat)); mask[5:8, 5:8] .= false
        mgrid = CGEF.StructuredGrid(geom, lon, lat, mask)
        os = zeros(size(u)); ot = zeros(size(u))
        CGEF.filter_field!(os, u, mgrid, CGEF.GaussianKernel(), 4000.0; backend = CGEF.SerialBackend())
        CGEF.filter_field!(ot, u, mgrid, CGEF.GaussianKernel(), 4000.0; backend = CGEF.ThreadedBackend())
        Test.@test ot ≈ os

        # Periodic global spherical grid (threaded must wrap exactly like serial)
        sgeom = CGEF.SphericalGeometry(6371000.0)
        slon = deg2rad.(collect(0.0:5.0:355.0))
        slat = deg2rad.(collect(-40.0:5.0:40.0))
        sgrid = CGEF.StructuredGrid(sgeom, slon, slat, trues(length(slon), length(slat)))
        su = rand(length(slon), length(slat))
        oss = zeros(size(su)); ost = zeros(size(su))
        CGEF.filter_field!(oss, su, sgrid, CGEF.TopHatKernel(), deg2rad(15.0) * 6371000.0; backend = CGEF.SerialBackend())
        CGEF.filter_field!(ost, su, sgrid, CGEF.TopHatKernel(), deg2rad(15.0) * 6371000.0; backend = CGEF.ThreadedBackend())
        Test.@test ost ≈ oss
    end

    # Distributed backend must also agree with serial (footprint engine + SharedArray fill). With no
    # extra worker processes the @distributed loop runs serially on the caller — still exact.
    Test.@testset "Distributed backend" begin
        geom = CGEF.CartesianGeometry(1000.0, 1000.0)
        lon = collect(0.0:1000.0:30e3)
        lat = collect(0.0:1000.0:30e3)
        mask = trues(length(lon), length(lat)); mask[6:9, 6:9] .= false
        grid = CGEF.StructuredGrid(geom, lon, lat, mask)
        u = rand(length(lon), length(lat))
        os = zeros(size(u)); od = zeros(size(u))
        CGEF.filter_field!(os, u, grid, CGEF.TopHatKernel(), 5000.0; backend = CGEF.SerialBackend())
        CGEF.filter_field!(od, u, grid, CGEF.TopHatKernel(), 5000.0; backend = CGEF.DistributedBackend())
        Test.@test od ≈ os
    end

    # GPU backend on the KernelAbstractions CPU device must match serial (validates the GPU kernel
    # logic here; actual GPU hardware is exercised separately). Same engine ⇒ masking + periodicity
    # consistent.
    Test.@testset "GPU backend (KernelAbstractions CPU)" begin
        gpu = CGEF.GPUBackend(KA.CPU())
        geom = CGEF.CartesianGeometry(1000.0, 1000.0)
        lon = collect(0.0:1000.0:20e3)
        lat = collect(0.0:1000.0:20e3)
        mask = trues(length(lon), length(lat)); mask[5:7, 5:7] .= false
        grid = CGEF.StructuredGrid(geom, lon, lat, mask)
        u = rand(length(lon), length(lat))
        for strat in (CGEF.Deformable(), CGEF.ZeroFill())
            os = zeros(size(u)); og = zeros(size(u))
            CGEF.filter_field!(os, u, grid, CGEF.TopHatKernel(), 5000.0; backend = CGEF.SerialBackend(), mask_strategy = strat)
            CGEF.filter_field!(og, u, grid, CGEF.TopHatKernel(), 5000.0; backend = gpu, mask_strategy = strat)
            Test.@test og ≈ os
        end

        # Periodic global spherical grid
        sgeom = CGEF.SphericalGeometry(6371000.0)
        slon = deg2rad.(collect(0.0:5.0:355.0))
        slat = deg2rad.(collect(-30.0:5.0:30.0))
        sgrid = CGEF.StructuredGrid(sgeom, slon, slat, trues(length(slon), length(slat)))
        su = rand(length(slon), length(slat))
        oss = zeros(size(su)); osg = zeros(size(su))
        CGEF.filter_field!(oss, su, sgrid, CGEF.TopHatKernel(), deg2rad(15.0) * 6371000.0; backend = CGEF.SerialBackend())
        CGEF.filter_field!(osg, su, sgrid, CGEF.TopHatKernel(), deg2rad(15.0) * 6371000.0; backend = gpu)
        Test.@test osg ≈ oss
    end

    # True n-D Cartesian filtering (1D + 3D) via the general footprint engine.
    Test.@testset "n-D filtering (1D + true 3D Cartesian)" begin
        # --- 1D ---
        geom1 = CGEF.CartesianGeometry(1.0, 1.0)
        x = collect(0.0:1.0:50.0)
        grid1 = CGEF.StructuredGrid(geom1, x, trues(length(x)))
        Test.@test CGEF.size_tuple(grid1) == (length(x),)
        # constant -> constant (normalization)
        o1 = zeros(length(x))
        CGEF.filter_field!(o1, fill(7.0, length(x)), grid1, CGEF.TopHatKernel(), 6.0)
        Test.@test all(≈(7.0), o1[10:40])
        # sub-grid scale -> identity (only the self cell is in support)
        g1 = rand(length(x)); oi = zeros(length(x))
        CGEF.filter_field!(oi, g1, grid1, CGEF.TopHatKernel(), 0.5)
        Test.@test oi ≈ g1

        # --- 3D (dz ≫ footprint, so only the in-plane disk contributes) ---
        geom3 = CGEF.CartesianGeometry(1.0, 1.0, 100.0)
        x3 = collect(0.0:1.0:20.0); y3 = collect(0.0:1.0:20.0); z3 = collect(0.0:100.0:300.0)
        nx, ny, nz = length(x3), length(y3), length(z3)
        grid3 = CGEF.StructuredGrid(geom3, x3, y3, z3, trues(nx, ny, nz))
        Test.@test CGEF.size_tuple(grid3) == (nx, ny, nz)
        # constant -> constant
        o3 = zeros(nx, ny, nz)
        CGEF.filter_field!(o3, fill(3.5, nx, ny, nz), grid3, CGEF.TopHatKernel(), 6.0)
        Test.@test all(≈(3.5), o3)

        # A z-invariant 3D field must reduce EXACTLY to the 2D filter of its slice (dz ≫ rad ⇒ no
        # vertical neighbours), validating the n-D engine against the 2D engine.
        f2d = rand(nx, ny)
        f3z = repeat(reshape(f2d, nx, ny, 1), 1, 1, nz)
        o3z = zeros(nx, ny, nz)
        CGEF.filter_field!(o3z, f3z, grid3, CGEF.TopHatKernel(), 6.0)
        grid2 = CGEF.StructuredGrid(CGEF.CartesianGeometry(1.0, 1.0), x3, y3, trues(nx, ny))
        o2 = zeros(nx, ny)
        CGEF.filter_field!(o2, f2d, grid2, CGEF.TopHatKernel(), 6.0)
        for k in 1:nz
            Test.@test o3z[:, :, k] ≈ o2
        end
    end

    # True 3D Cartesian energy flux Π = -ρ₀ S̄_ij τ_ij (all nine strain components).
    Test.@testset "3D Cartesian energy flux" begin
        geom3 = CGEF.CartesianGeometry(1.0, 1.0, 50.0)
        x = collect(0.0:1.0:24.0); y = collect(0.0:1.0:24.0); z = collect(0.0:50.0:150.0)
        nx, ny, nz = length(x), length(y), length(z)
        grid3 = CGEF.StructuredGrid(geom3, x, y, z, trues(nx, ny, nz))
        ker = CGEF.TopHatKernel(); ℓ = 5.0

        # (1) Constant velocity ⇒ zero strain ⇒ Π ≡ 0.
        Πc = zeros(nx, ny, nz)
        CGEF.compute_Π!(Πc, fill(2.0, nx, ny, nz), fill(-3.0, nx, ny, nz),
                        fill(0.5, nx, ny, nz), grid3, ker, ℓ)
        Test.@test maximum(abs, Πc) < 1e-9

        # (2) z-invariant (u, v) with w = 0: the 3D six-term contraction must collapse EXACTLY to the
        # 2D three-term flux on every layer (Szz = Sxz = Syz = τxz = τyz = τzz = 0), validating the 3D
        # assembly + 3D derivatives against the established 2D path.
        u2 = rand(nx, ny) .- 0.5; v2 = rand(nx, ny) .- 0.5
        u3 = repeat(reshape(u2, nx, ny, 1), 1, 1, nz)
        v3 = repeat(reshape(v2, nx, ny, 1), 1, 1, nz)
        w3 = zeros(nx, ny, nz)
        Π3 = zeros(nx, ny, nz)
        CGEF.compute_Π!(Π3, u3, v3, w3, grid3, ker, ℓ)

        grid2 = CGEF.StructuredGrid(CGEF.CartesianGeometry(1.0, 1.0), x, y, trues(nx, ny))
        Π2 = zeros(nx, ny)
        CGEF.compute_Π!(Π2, u2, v2, nothing, grid2, ker, ℓ)
        for k in 1:nz
            Test.@test Π3[:, :, k] ≈ Π2
        end
    end

    # spatial finite differences and boundary stencil fallbacks
    Test.@testset "Derivatives" begin
        geom = CGEF.CartesianGeometry(2.0, 2.0)
        lon = collect(0.0:2.0:10.0) # 6 points
        lat = collect(0.0:2.0:10.0) # 6 points
        mask = trues(6, 6)
        grid = CGEF.StructuredGrid(geom, lon, lat, mask)

        # Test horizontal derivatives of f(x) = 3x + 1
        # ∂f/∂x should be exactly 3.0 at all wet cells
        f = zeros(6, 6)
        for j in 1:6, i in 1:6
            f[i, j] = 3.0 * grid.lon[i] + 1.0
        end

        ∂f∂x = zeros(6, 6)
        CGEF.ddx!(∂f∂x, f, grid)

        Test.@test ∂f∂x[2, 3] ≈ 3.0
        Test.@test ∂f∂x[1, 3] ≈ 3.0 # forward difference at boundary
        Test.@test ∂f∂x[6, 3] ≈ 3.0 # backward difference at boundary
    end

    # NOTE: the in-package Helmholtz/SOR solver was removed in the overhaul. Rotational/divergent
    # decomposition is now a preprocessing step via HelmholtzDecomposition.jl, and the
    # rot/div/cross cascade split will be `compute_Π_decomposed!` (Phase 4). See the overhaul plan.

    # SFS Stresses and cross-scale energy transfer (Π) calculations
    Test.@testset "Diagnostics & Pipeline" begin
        # 2D rigid-body rotation u = -Ωy, v = Ωx has zero kinetic energy transfer (Π = 0)
        geom = CGEF.CartesianGeometry(2000.0, 2000.0)
        lon = collect(-20000.0:2000.0:20000.0) # 21 points
        lat = collect(-20000.0:2000.0:20000.0) # 21 points
        mask = trues(21, 21)
        grid = CGEF.StructuredGrid(geom, lon, lat, mask)

        Ω = 1e-4 # Coriolis frequency-like rotation rate
        u = zeros(21, 21)
        v = zeros(21, 21)
        for j in 1:21, i in 1:21
            u[i, j] = -Ω * grid.lat[j]
            v[i, j] = Ω * grid.lon[i]
        end

        Π = zeros(21, 21)
        CGEF.compute_Π!(Π, u, v, nothing, grid, CGEF.TopHatKernel(), 10000.0)

        # Kinetic energy transfer must be zero (rigid body rotation is pure laminar cascade-free flow)
        Test.@test Π[11, 11] ≈ 0.0 atol=1e-12

        # Test Pipeline integration with unicode Π
        res = CGEF.coarse_grain(u, v, grid; scales=[10000.0], kernel=CGEF.TopHatKernel())
        Test.@test res.Π[1] ≈ Π

        # Test Spherical projections and coarse graining with mixed types
        sgeom = CGEF.SphericalGeometry(6371000.0)
        slon = collect(0.0:2.0:10.0)
        slat = collect(0.0:2.0:10.0)
        smask = trues(length(slon), length(slat))
        sgrid = CGEF.StructuredGrid(sgeom, deg2rad.(slon), deg2rad.(slat), smask)

        # Test to_planetary_cartesian and from_planetary_cartesian mixed type support
        proj = CGEF.to_planetary_cartesian(sgeom, Float32(1.0), Float32(2.0), 0.1, 0.2, 0.3)
        Test.@test proj isa SA.SVector{3, Float64}

        inv_proj = CGEF.from_planetary_cartesian(sgeom, Float32(1.0), 2.0, 3.0, 0.1, 0.2)
        Test.@test inv_proj isa SA.SVector{3, Float64}

        # Test coarse_grain on sphere with Float32 inputs (matching PythonCall runtime environment)
        su = fill(Float32(1.0), length(slon), length(slat))
        sv = fill(Float32(0.5), length(slon), length(slat))
        sres = CGEF.coarse_grain(su, sv, sgrid; scales=[50000.0], kernel=CGEF.TopHatKernel())
        Test.@test !any(isnan, sres.Π[1])
        Test.@test !any(isnan, sres.cumulative_energy)
        Test.@test !any(isnan, sres.filtering_spectrum)
    end

    # Test periodic boundary handling for spherical grids
    Test.@testset "Spherical Periodic Boundaries" begin
        geom = CGEF.SphericalGeometry(6371000.0)
        # Create a grid that spans nearly 360 degrees in longitude
        lon_deg = collect(0.0:5.0:355.0)  # 72 points, 5-degree spacing
        lat_deg = collect(-45.0:5.0:45.0)  # 19 points
        lon_rad = deg2rad.(lon_deg)
        lat_rad = deg2rad.(lat_deg)
        mask = trues(length(lon_deg), length(lat_deg))
        grid = CGEF.StructuredGrid(geom, lon_rad, lat_rad, mask)

        # Create a field with a sharp gradient at the periodic boundary
        field = zeros(length(lon_deg), length(lat_deg))
        for j in 1:length(lat_deg)
            for i in 1:length(lon_deg)
                # Sharp transition near 0/360 boundary
                if lon_deg[i] > 350.0 || lon_deg[i] < 10.0
                    field[i, j] = 1.0
                else
                    field[i, j] = 0.0
                end
            end
        end

        # Filter at 20° scale (larger than the 15° band, so wrapping matters)
        out = zeros(length(lon_deg), length(lat_deg))
        CGEF.filter_field!(out, field, grid, CGEF.TopHatKernel(), deg2rad(20.0) * 6371000.0)

        # Key test: points at 0° and 355° (5° apart) should have similar values
        # because they're physically close and the filter wraps around
        val_0 = out[1, 10]      # 0°
        val_355 = out[end, 10]  # 355°

        # Without periodic wrapping, val_355 would see only 335-355° (mostly 0.0)
        # With wrapping, it sees wrapped 0-15° plus 335-355°, giving higher values
        # The values won't be identical due to asymmetric neighborhoods:
        # - 0° sees 0-20° (indices 1-5): 2 ones, 3 zeros = ~0.4 average
        # - 355° sees 335-355° + wrapped 0-15° (9 indices): 3 ones, 6 zeros = ~0.33 average
        # But both should be in the 0.6-0.9 range (weighted by distance)
        Test.@test val_0 > 0.6 && val_0 < 0.95
        Test.@test val_355 > 0.6 && val_355 < 0.95

        # The ratio should be reasonable (within 30% of each other)
        Test.@test abs(val_0 - val_355) / max(val_0, val_355) < 0.3

        # A point at 30° (6 indices in) should have much lower value since it's outside the band
        val_30 = out[7, 10]  # 30°
        Test.@test val_30 < val_0 * 0.5  # Should be significantly lower than at 0°
    end

    # Regional domains must NOT wrap in longitude (the previous code wrapped every spherical grid,
    # double-counting near boundaries when the footprint exceeded a regional domain).
    Test.@testset "Regional vs periodic longitude" begin
        geom = CGEF.SphericalGeometry(6371000.0)
        lat = deg2rad.(collect(-4.0:2.0:4.0))

        # Regional lon span -> auto-detected NON-periodic
        lon_reg = deg2rad.(collect(0.0:2.0:20.0))   # 11 points, 20° span
        mask_reg = trues(length(lon_reg), length(lat))
        grid_reg = CGEF.StructuredGrid(geom, lon_reg, lat, mask_reg)
        Test.@test CGEF.isperiodic(grid_reg, 1) == false

        # Full-circle lon span -> auto-detected periodic
        lon_glob = deg2rad.(collect(0.0:5.0:355.0))
        mask_glob = trues(length(lon_glob), length(lat))
        grid_glob = CGEF.StructuredGrid(geom, lon_glob, lat, mask_glob)
        Test.@test CGEF.isperiodic(grid_glob, 1) == true

        # Explicit override in both directions
        Test.@test CGEF.isperiodic(CGEF.StructuredGrid(geom, lon_reg, lat, mask_reg; periodic = true), 1) == true
        Test.@test CGEF.isperiodic(CGEF.StructuredGrid(geom, lon_glob, lat, mask_glob; periodic = false), 1) == false

        # The periodicity flag must actually change filtering: with a footprint wider than the
        # regional domain, wrapping double-counts and yields a different (incorrect) field.
        grid_forced = CGEF.StructuredGrid(geom, lon_reg, lat, mask_reg; periodic = true)
        field = Float64[i for i in 1:length(lon_reg), _ in 1:length(lat)]  # ramp in lon
        scale = deg2rad(30.0) * 6371000.0   # footprint wider than the 20° domain
        out_nowrap = zeros(size(field))
        out_wrap = zeros(size(field))
        CGEF.filter_field!(out_nowrap, field, grid_reg, CGEF.TopHatKernel(), scale)
        CGEF.filter_field!(out_wrap, field, grid_forced, CGEF.TopHatKernel(), scale)
        Test.@test !any(isnan, out_nowrap)
        Test.@test !(out_nowrap ≈ out_wrap)
    end

    # Test kernel normalization (weights must sum to 1.0 for uniform field)
    Test.@testset "Kernel Normalization" begin
        geom = CGEF.SphericalGeometry(6371000.0)
        lon_deg = collect(0.0:2.0:10.0)
        lat_deg = collect(0.0:2.0:10.0)
        lon_rad = deg2rad.(lon_deg)
        lat_rad = deg2rad.(lat_deg)
        mask = trues(length(lon_deg), length(lat_deg))
        grid = CGEF.StructuredGrid(geom, lon_rad, lat_rad, mask)

        # Constant field
        field = ones(length(lon_deg), length(lat_deg))
        out_zero = zeros(length(lon_deg), length(lat_deg))
        out_renorm = zeros(length(lon_deg), length(lat_deg))

        # Filter with both masking strategies
        CGEF.filter_field!(out_zero, field, grid, CGEF.TopHatKernel(), 100000.0; mask_strategy=CGEF.ZeroFill())
        CGEF.filter_field!(out_renorm, field, grid, CGEF.TopHatKernel(), 100000.0; mask_strategy=CGEF.Deformable())

        # For a constant field of ones, output should be exactly 1.0 everywhere
        # (or very close, allowing for small numerical errors)
        for j in 2:length(lat_deg)-1, i in 2:length(lon_deg)-1
            Test.@test out_zero[i, j] ≈ 1.0 atol=1e-10
            Test.@test out_renorm[i, j] ≈ 1.0 atol=1e-10
        end
    end

    # Test great-circle distance accuracy
    Test.@testset "Great-Circle Distance Accuracy" begin
        geom = CGEF.SphericalGeometry(6371000.0)

        # Test: distance from (0, 0) to (0, 90) should be ~1/4 Earth circumference
        p1 = SA.SVector{2,Float64}(0.0, 0.0)  # (lon, lat) = (0, 0) on equator
        p2 = SA.SVector{2,Float64}(0.0, deg2rad(90.0))  # North pole
        d = CGEF.distance(geom, p1, p2)

        # Should be approximately quarter circumference
        quarter_circumference = π * geom.R / 2
        Test.@test d ≈ quarter_circumference rtol=1e-6

        # Test: distance along equator for 1 degree
        p3 = SA.SVector{2,Float64}(0.0, 0.0)
        p4 = SA.SVector{2,Float64}(deg2rad(1.0), 0.0)
        d_equator = CGEF.distance(geom, p3, p4)

        # Should be approximately 111.195 km per degree at equator (2πR/360)
        Test.@test d_equator ≈ π * geom.R / 180 rtol=1e-6
    end

    # Test Taylor-Green vortex for strain rate verification
    Test.@testset "Taylor-Green Vortex" begin
        # Taylor-Green vortex has known analytical solutions
        # u = sin(x)cos(y), v = -cos(x)sin(y)
        # Strain rates and vorticity have exact analytical forms

        geom = CGEF.CartesianGeometry(0.1, 0.1)  # 0.1 unit grid spacing
        lon = collect(0.0:0.1:2π)
        lat = collect(0.0:0.1:2π)
        mask = trues(length(lon), length(lat))
        grid = CGEF.StructuredGrid(geom, lon, lat, mask)

        u = [sin(x) * cos(y) for x in lon, y in lat]
        v = [-cos(x) * sin(y) for x in lon, y in lat]

        # Compute derivatives
        dudx = zeros(length(lon), length(lat))
        dudy = zeros(length(lon), length(lat))
        dvdx = zeros(length(lon), length(lat))
        dvdy = zeros(length(lon), length(lat))

        CGEF.ddx!(dudx, u, grid)
        CGEF.ddy!(dudy, u, grid)
        CGEF.ddx!(dvdx, v, grid)
        CGEF.ddy!(dvdy, v, grid)

        # Check a point away from boundaries
        i, j = 10, 10
        x, y = lon[i], lat[j]

        # Analytical: ∂u/∂x = cos(x)cos(y)
        Test.@test dudx[i, j] ≈ cos(x) * cos(y) rtol=0.01

        # Analytical: ∂u/∂y = -sin(x)sin(y)
        Test.@test dudy[i, j] ≈ -sin(x) * sin(y) rtol=0.01

        # Analytical: ∂v/∂x = sin(x)sin(y)
        Test.@test dvdx[i, j] ≈ sin(x) * sin(y) rtol=0.01

        # Analytical: ∂v/∂y = -cos(x)cos(y)
        Test.@test dvdy[i, j] ≈ -cos(x) * cos(y) rtol=0.01
    end

    # Mathematical correctness: Rigid body rotation must have exactly Π = 0
    Test.@testset "Rigid Body Rotation - Zero Energy Flux" begin
        # Rigid body rotation has no deformation, so no energy cascade
        # u = -Ωy, v = Ωx should give Π = 0 everywhere

        geom = CGEF.CartesianGeometry(1000.0, 1000.0)
        lon = collect(-50e3:1000.0:50e3)  # 101 points
        lat = collect(-50e3:1000.0:50e3)  # 101 points
        mask = trues(length(lon), length(lat))
        grid = CGEF.StructuredGrid(geom, lon, lat, mask)

        Ω = 1e-4  # rotation rate
        u = [-Ω * y for x in lon, y in lat]
        v = [Ω * x for x in lon, y in lat]

        # Test at multiple scales
        for scale in [5000.0, 10000.0, 20000.0]
            Π = zeros(length(lon), length(lat))
            CGEF.compute_Π!(Π, u, v, nothing, grid, CGEF.TopHatKernel(), scale)

            # Check interior points (away from boundaries)
            for j in 40:60, i in 40:60
                Test.@test abs(Π[i, j]) < 1e-10  # Should be exactly zero
            end
        end
    end

    # Mathematical correctness: Strain rate properties
    Test.@testset "Strain Rate Tensor Properties" begin
        # Strain rate tensor S_ij must be symmetric: S_ij = S_ji
        # For 2D incompressible flow: S_xx + S_yy = 0 (trace = divergence)

        geom = CGEF.CartesianGeometry(1000.0, 1000.0)
        lon = collect(0.0:1000.0:50e3)
        lat = collect(0.0:1000.0:50e3)
        mask = trues(length(lon), length(lat))
        grid = CGEF.StructuredGrid(geom, lon, lat, mask)

        # Create a divergent flow field
        u = [0.01 * x for x in lon, y in lat]  # Linear in x
        v = [0.01 * y for x in lon, y in lat]  # Linear in y

        # Filter the field
        u_filt = zeros(length(lon), length(lat))
        v_filt = zeros(length(lon), length(lat))
        CGEF.filter_field!(u_filt, u, grid, CGEF.TopHatKernel(), 10000.0)
        CGEF.filter_field!(v_filt, v, grid, CGEF.TopHatKernel(), 10000.0)

        # Compute strain rates
        S_xx = zeros(length(lon), length(lat))
        S_yy = zeros(length(lon), length(lat))
        S_xy = zeros(length(lon), length(lat))
        scratch = zeros(length(lon), length(lat))

        CGEF.ddx!(S_xx, u_filt, grid)
        CGEF.ddy!(S_yy, v_filt, grid)
        CGEF.ddy!(S_xy, u_filt, grid)
        CGEF.ddx!(scratch, v_filt, grid)
        @. S_xy = 0.5 * (S_xy + scratch)

        # Test symmetry: S_xy should equal S_yx (we only computed S_xy)
        # Test trace = divergence for incompressible flow
        for j in 20:length(lat)-20, i in 20:length(lon)-20
            # For filtered divergent flow, S_xx + S_yy should equal divergence
            divergence = S_xx[i,j] + S_yy[i,j]
            # Divergence should be approximately constant (0.02 for this field)
            Test.@test abs(divergence - 0.02) < 0.01
        end
    end

    # Mathematical correctness: SFS stress properties
    Test.@testset "SFS Stress Tensor Properties" begin
        # τ_ij = [u_i*u_j]̄ - ū_i*ū_j must be symmetric: τ_ij = τ_ji
        # For isotropic turbulence, trace of τ should be positive (energy in SFS)

        geom = CGEF.CartesianGeometry(1000.0, 1000.0)
        lon = collect(0.0:1000.0:30e3)
        lat = collect(0.0:1000.0:30e3)
        mask = trues(length(lon), length(lat))
        grid = CGEF.StructuredGrid(geom, lon, lat, mask)

        # Create random velocity field
        u = rand(length(lon), length(lat))
        v = rand(length(lon), length(lat))

        # Filter fields
        u_filt = zeros(length(lon), length(lat))
        v_filt = zeros(length(lon), length(lat))
        CGEF.filter_field!(u_filt, u, grid, CGEF.TopHatKernel(), 5000.0)
        CGEF.filter_field!(v_filt, v, grid, CGEF.TopHatKernel(), 5000.0)

        # Filter products
        uu = zeros(length(lon), length(lat))
        uv = zeros(length(lon), length(lat))
        vv = zeros(length(lon), length(lat))
        scratch = zeros(length(lon), length(lat))

        @. scratch = u * u
        CGEF.filter_field!(uu, scratch, grid, CGEF.TopHatKernel(), 5000.0)
        @. scratch = u * v
        CGEF.filter_field!(uv, scratch, grid, CGEF.TopHatKernel(), 5000.0)
        @. scratch = v * v
        CGEF.filter_field!(vv, scratch, grid, CGEF.TopHatKernel(), 5000.0)

        # Compute SFS stress
        τ_xx = zeros(length(lon), length(lat))
        τ_xy = zeros(length(lon), length(lat))
        τ_yy = zeros(length(lon), length(lat))

        @. τ_xx = uu - u_filt * u_filt
        @. τ_xy = uv - u_filt * v_filt
        @. τ_yy = vv - v_filt * v_filt

        # Test that trace of τ is positive (physical constraint for filtering)
        # τ_xx + τ_yy = [u²+v²]̄ - (ū² + v̄²) ≥ 0 by Jensen's inequality
        for j in 10:length(lat)-10, i in 10:length(lon)-10
            trace_τ = τ_xx[i,j] + τ_yy[i,j]
            Test.@test trace_τ >= -1e-10  # Should be non-negative
        end

        # Test symmetry: compute τ_yx and verify equals τ_xy
        scratch2 = zeros(length(lon), length(lat))
        @. scratch2 = v * u
        CGEF.filter_field!(scratch, scratch2, grid, CGEF.TopHatKernel(), 5000.0)
        @. scratch2 = scratch - v_filt * u_filt  # τ_yx

        for j in 10:length(lat)-10, i in 10:length(lon)-10
            Test.@test τ_xy[i,j] ≈ scratch2[i,j] rtol=1e-10
        end
    end

    # Mathematical correctness: Π sign consistency with SFS stress and strain
    Test.@testset "Energy Flux Sign Consistency" begin
        # Π = -ρ₀ * S̄_ij * τ_ij should have consistent sign based on S and τ
        # For a convergent strain with positive SFS stress, Π should be negative
        # (energy goes from resolved to sub-grid = forward cascade)

        geom = CGEF.CartesianGeometry(1000.0, 1000.0)
        lon = collect(0.0:1000.0:50e3)
        lat = collect(0.0:1000.0:50e3)
        mask = trues(length(lon), length(lat))
        grid = CGEF.StructuredGrid(geom, lon, lat, mask)

        # Create a simple deformation field
        # u = a*x, v = -a*y gives pure strain (convergence in x, divergence in y)
        a = 0.001
        u = [a * x for x in lon, y in lat]
        v = [-a * y for x in lon, y in lat]

        # The strain rate tensor for this field:
        # S_xx = a, S_yy = -a, S_xy = 0

        # At scale where filtering matters, we can verify:
        # - SFS stress τ should be computed correctly
        # - The sign of Π should match the sign of -S:τ

        Π = zeros(length(lon), length(lat))
        CGEF.compute_Π!(Π, u, v, nothing, grid, CGEF.TopHatKernel(), 10000.0)

        # For this pure linear deformation, filtering doesn't change the field
        # (linear fields are invariant under top-hat filtering)
        # So τ should be ~0 and Π should be ~0
        for j in 20:length(lat)-20, i in 20:length(lon)-20
            Test.@test abs(Π[i,j]) < 1e-8
        end
    end

    # Mathematical correctness: Energy budget closure
    Test.@testset "Energy Budget - Filtered vs Unfiltered" begin
        # Test that: 0.5*ρ₀*|u|² = 0.5*ρ₀*|ū|² + 0.5*ρ₀*trace(τ) + (boundary terms)
        # For periodic domains, the resolved + SFS energies should relate to total energy

        geom = CGEF.CartesianGeometry(1000.0, 1000.0)
        lon = collect(0.0:1000.0:30e3)
        lat = collect(0.0:1000.0:30e3)
        mask = trues(length(lon), length(lat))
        grid = CGEF.StructuredGrid(geom, lon, lat, mask)

        # Random velocity field
        u = rand(length(lon), length(lat))
        v = rand(length(lon), length(lat))

        ρ₀ = 1025.0

        # Compute total energy
        total_energy = 0.5 * ρ₀ * (u.^2 + v.^2)

        # Filter at some scale
        scale = 5000.0
        u_filt = zeros(length(lon), length(lat))
        v_filt = zeros(length(lon), length(lat))
        CGEF.filter_field!(u_filt, u, grid, CGEF.TopHatKernel(), scale)
        CGEF.filter_field!(v_filt, v, grid, CGEF.TopHatKernel(), scale)

        # Filter products for SFS stress trace
        uu_filt = zeros(length(lon), length(lat))
        vv_filt = zeros(length(lon), length(lat))
        scratch = zeros(length(lon), length(lat))

        @. scratch = u * u
        CGEF.filter_field!(uu_filt, scratch, grid, CGEF.TopHatKernel(), scale)
        @. scratch = v * v
        CGEF.filter_field!(vv_filt, scratch, grid, CGEF.TopHatKernel(), scale)

        # SFS energy = 0.5*ρ₀*trace(τ) = 0.5*ρ₀*([u²]̄ + [v²]̄ - ū² - v̄²)
        sfs_energy = zeros(length(lon), length(lat))
        @. sfs_energy = 0.5 * ρ₀ * (uu_filt + vv_filt - u_filt^2 - v_filt^2)

        # Resolved energy
        resolved_energy = zeros(length(lon), length(lat))
        @. resolved_energy = 0.5 * ρ₀ * (u_filt^2 + v_filt^2)

        # Verify: sfs_energy ≥ 0 (Jensen's inequality)
        for j in 10:length(lat)-10, i in 10:length(lon)-10
            Test.@test sfs_energy[i,j] >= -1e-12  # Should be non-negative
        end

        # Verify: sfs_energy + resolved_energy ≈ filtered total energy
        # ([u²]̄ + [v²]̄)/2 = ([u²+v²]̄)/2
        filtered_total = zeros(length(lon), length(lat))
        @. scratch = u.^2 + v.^2
        CGEF.filter_field!(filtered_total, scratch, grid, CGEF.TopHatKernel(), scale)
        @. filtered_total = 0.5 * ρ₀ * filtered_total

        for j in 10:length(lat)-10, i in 10:length(lon)-10
            energy_sum = sfs_energy[i,j] + resolved_energy[i,j]
            Test.@test energy_sum ≈ filtered_total[i,j] rtol=1e-10
        end
    end

    # Mathematical correctness: Filtered field of constant = constant
    Test.@testset "Filter Normalization - Constant Field" begin
        # Filtering a constant field must return exactly the same constant
        # This tests that kernel weights are properly normalized

        geom = CGEF.CartesianGeometry(1000.0, 1000.0)
        lon = collect(0.0:1000.0:50e3)
        lat = collect(0.0:1000.0:50e3)
        mask = trues(length(lon), length(lat))
        grid = CGEF.StructuredGrid(geom, lon, lat, mask)

        C = 42.0  # Constant value
        field = fill(C, length(lon), length(lat))

        # Test both masking strategies
        for kernel in [CGEF.TopHatKernel(), CGEF.GaussianKernel()]
            for scale in [5000.0, 10000.0, 20000.0]
                out_zero = zeros(length(lon), length(lat))
                out_renorm = zeros(length(lon), length(lat))

                CGEF.filter_field!(out_zero, field, grid, kernel, scale; mask_strategy=CGEF.ZeroFill())
                CGEF.filter_field!(out_renorm, field, grid, kernel, scale; mask_strategy=CGEF.Deformable())

                # Interior points should be exactly C
                for j in 20:length(lat)-20, i in 20:length(lon)-20
                    Test.@test out_zero[i,j] ≈ C rtol=1e-10
                    Test.@test out_renorm[i,j] ≈ C rtol=1e-10
                end
            end
        end
    end

    # Cumulative coarse KE (Sadek-Aluie Eq.15) vs the filtering spectral density (Eq.14)
    Test.@testset "Filtering spectrum" begin
        geom = CGEF.CartesianGeometry(2000.0, 2000.0)
        lon = collect(0.0:2000.0:100e3)
        lat = collect(0.0:2000.0:100e3)
        grid = CGEF.StructuredGrid(geom, lon, lat, trues(length(lon), length(lat)))

        U = 0.5  # m/s
        V = 0.3  # m/s
        u = fill(U, length(lon), length(lat))
        v = fill(V, length(lon), length(lat))
        ρ₀ = 1025.0
        expected_energy = 0.5 * ρ₀ * (U^2 + V^2)
        scales = [5000.0, 10000.0, 20000.0, 40000.0]

        # A uniform field is unchanged by filtering, so the CUMULATIVE coarse KE equals the kinetic
        # energy at every scale (Eq. 15).
        cumE = CGEF.cumulative_energy(u, v, nothing, grid, CGEF.TopHatKernel(), scales; ρ₀=ρ₀)
        for E in cumE
            Test.@test E ≈ expected_energy rtol=1e-6
        end

        # Since the cumulative energy is constant in ℓ, the filtering spectral DENSITY (its
        # k_ℓ-derivative, Eq. 14) must be ≈ 0 everywhere — NOT equal to the energy.
        kℓ, Ẽ = CGEF.filtering_spectrum(u, v, nothing, grid, CGEF.TopHatKernel(), scales; ρ₀=ρ₀, L=1.0)
        Test.@test length(kℓ) == length(scales)
        Test.@test all(abs.(Ẽ) .< 1e-6 * expected_energy)

        # spectral_density reproduces a known derivative: C(k)=k² ⇒ dC/dk = 2k (central differences
        # are exact for a quadratic on a uniform grid).
        kk = collect(1.0:1.0:5.0)
        Test.@test CGEF.spectral_density(kk .^ 2, kk)[3] ≈ 2 * kk[3]
    end

    # Germano subfilter-stress decomposition: L + C + R == τ (exact closure)
    Test.@testset "Stress decomposition (Leonard/Cross/Reynolds)" begin
        geom = CGEF.CartesianGeometry(1000.0, 1000.0)
        lon = collect(0.0:1000.0:30e3)
        lat = collect(0.0:1000.0:30e3)
        grid = CGEF.StructuredGrid(geom, lon, lat, trues(length(lon), length(lat)))
        u = rand(length(lon), length(lat))
        v = rand(length(lon), length(lat))
        kern = CGEF.TopHatKernel()
        scale = 5000.0

        d = CGEF.tau_decomposition(u, v, grid, kern, scale)

        # Reference τ_ij = filter(u_i u_j) - ū_i ū_j with the same filter.
        ub = zeros(size(u)); vb = zeros(size(v))
        CGEF.filter_field!(ub, u, grid, kern, scale)
        CGEF.filter_field!(vb, v, grid, kern, scale)
        uu = zeros(size(u)); uv = zeros(size(u)); vv = zeros(size(u))
        CGEF.filter_field!(uu, u .* u, grid, kern, scale)
        CGEF.filter_field!(uv, u .* v, grid, kern, scale)
        CGEF.filter_field!(vv, v .* v, grid, kern, scale)
        τxx = uu .- ub .^ 2
        τxy = uv .- ub .* vb
        τyy = vv .- vb .^ 2

        Test.@test d.L.xx .+ d.C.xx .+ d.R.xx ≈ τxx
        Test.@test d.L.xy .+ d.C.xy .+ d.R.xy ≈ τxy
        Test.@test d.L.yy .+ d.C.yy .+ d.R.yy ≈ τyy
        # Reynolds (subfilter–subfilter) stress trace is non-negative (Jensen).
        Test.@test all(d.R.xx .+ d.R.yy .>= -1e-10)
    end

    # Rotational/divergent (Helmholtz) decomposition of the energy flux.
    Test.@testset "Helmholtz flux decomposition" begin
        geom = CGEF.CartesianGeometry(1000.0, 1000.0)
        x = collect(0.0:1000.0:30e3); y = collect(0.0:1000.0:30e3)
        grid = CGEF.StructuredGrid(geom, x, y, trues(length(x), length(y)))
        u = rand(length(x), length(y)) .- 0.5
        v = rand(length(x), length(y)) .- 0.5
        kern = CGEF.TopHatKernel(); scale = 5000.0

        # An arbitrary split exercises the bilinear identity (algebra holds for any rotational part).
        ur = 0.4 .* u; vr = 0.4 .* v
        dec = CGEF.compute_Π_decomposed(u, v, ur, vr, grid, kern, scale)

        # (1) channels sum EXACTLY to the total.
        Test.@test dec.total ≈ dec.rotational .+ dec.cross .+ dec.divergent
        # (2) the total equals the standard full-flux computation.
        Πfull = zeros(size(u)); CGEF.compute_Π!(Πfull, u, v, nothing, grid, kern, scale)
        Test.@test dec.total ≈ Πfull
        # (3) pure-rotational input ⇒ divergent and cross channels vanish, total = rotational.
        dec_r = CGEF.compute_Π_decomposed(u, v, u, v, grid, kern, scale)
        Test.@test maximum(abs, dec_r.divergent) < 1e-10
        Test.@test maximum(abs, dec_r.cross) < 1e-10
        Test.@test dec_r.rotational ≈ Πfull
    end

    # Cross-scale tracer-variance flux (scalar analog of Π).
    Test.@testset "Tracer variance flux" begin
        geom = CGEF.CartesianGeometry(1000.0, 1000.0)
        x = collect(0.0:1000.0:30e3); y = collect(0.0:1000.0:30e3)
        grid = CGEF.StructuredGrid(geom, x, y, trues(length(x), length(y)))
        u = rand(length(x), length(y)) .- 0.5
        v = rand(length(x), length(y)) .- 0.5
        θ = rand(length(x), length(y))
        kern = CGEF.TopHatKernel(); scale = 5000.0

        # (1) constant tracer ⇒ zero gradient ⇒ zero flux.
        Πc = CGEF.tracer_variance_flux(u, v, fill(2.5, size(θ)), grid, kern, scale)
        Test.@test maximum(abs, Πc) < 1e-9

        # (2) matches the explicit definition Πθ = -(τx ∂x θ̄ + τy ∂y θ̄) built from primitives.
        Πθ = CGEF.tracer_variance_flux(u, v, θ, grid, kern, scale)
        ub = zeros(size(u)); vb = zeros(size(v)); θb = zeros(size(θ))
        CGEF.filter_field!(ub, u, grid, kern, scale)
        CGEF.filter_field!(vb, v, grid, kern, scale)
        CGEF.filter_field!(θb, θ, grid, kern, scale)
        uθ = zeros(size(u)); vθ = zeros(size(u))
        CGEF.filter_field!(uθ, u .* θ, grid, kern, scale)
        CGEF.filter_field!(vθ, v .* θ, grid, kern, scale)
        τx = uθ .- ub .* θb; τy = vθ .- vb .* θb
        gx = zeros(size(θ)); gy = zeros(size(θ))
        CGEF.ddx!(gx, θb, grid); CGEF.ddy!(gy, θb, grid)
        ref = .-(τx .* gx .+ τy .* gy)
        Test.@test Πθ ≈ ref
    end

    # The CairoMakie viz functions are parent-owned stubs; without the extension loaded they must
    # raise a helpful error (the real methods are exercised when `using CairoMakie`).
    Test.@testset "Visualization stubs" begin
        Test.@test isdefined(CGEF, :plot_Π_map)
        Test.@test isdefined(CGEF, :plot_spectrum)
        Test.@test_throws ArgumentError CGEF.plot_Π_map(nothing, 1, nothing)
        Test.@test_throws ArgumentError CGEF.plot_spectrum(nothing)
    end
end
