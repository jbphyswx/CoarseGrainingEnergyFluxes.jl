using Test: Test
using StaticArrays: StaticArrays as SA
using Aqua: Aqua
using ExplicitImports: ExplicitImports as EI
using JET: JET
using FFTW: FFTW  # triggers the spectral-filtering extension
using FINUFFT: FINUFFT  # triggers the scattered-Cartesian spectral extension
using FastSphericalHarmonics: FastSphericalHarmonics as FSH  # triggers the uniform-spherical spectral extension
using NUFSHT: NUFSHT  # triggers the scattered-spherical spectral extension
using OhMyThreads: OhMyThreads  # triggers the threaded-backend extension
using Distributed: Distributed  # with SharedArrays, triggers the distributed-backend extension
using SharedArrays: SharedArrays
using MPI: MPI  # triggers the MPI-backend extension; real multi-rank execution is
                # test/mpi_runtests.jl, run via `mpiexec`, not this single-process suite
MPI.Init()  # required before ANY MPI routine runs (only MPI.Initialized/Finalized are safe pre-init) —
            # single-rank here, so MPIBackend degenerates to "this rank owns every row," exercised
            # below in the "Backends" testset the same way DistributedBackend already is.
using KernelAbstractions: KernelAbstractions as KA  # triggers the GPU backend extension (CPU device here)
using NearestNeighbors: NearestNeighbors  # triggers the UnstructuredGrid k-d-tree adjacency extension
using DelaunayTriangulation: DelaunayTriangulation  # triggers the Cartesian Voronoi-area extension
using Quickhull: Quickhull  # triggers the spherical Voronoi-area extension
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
            :CoarseGrainingEnergyFluxesFastSphericalHarmonicsExt,
            :CoarseGrainingEnergyFluxesNUFSHTExt,
            :CoarseGrainingEnergyFluxesOhMyThreadsExt,
            :CoarseGrainingEnergyFluxesDistributedExt,
            :CoarseGrainingEnergyFluxesGPUExt,
            :CoarseGrainingEnergyFluxesMPIExt,
        )
            ext = Base.get_extension(CGEF, extname)
            ext === nothing && continue
            Test.@test (EI.check_no_implicit_imports(ext); true)
            Test.@test (EI.check_no_stale_explicit_imports(ext); true)
        end
    end

    Test.@testset "JET type stability (hot path)" begin
        # JET tracks compiler internals and explicitly refuses to run on pre-release Julia
        # (`@test_opt` throws on nightly/rc — it loads as a no-op stub). Skip there; the type-stability
        # gate runs on every released version in CI.
        if !isempty(VERSION.prerelease)
            @info "Skipping JET type-stability checks on pre-release Julia $(VERSION)"
            Test.@test_skip true
        else
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
            JET.@test_opt CGEF.Filtering.apply_footprint!(out, field, grid, fp, CGEF.Filtering.Deformable(), false)
            JET.@test_opt CGEF.Filtering.apply_footprint!(out, field, grid, fp, CGEF.Filtering.ZeroFill(), false)

            # The serial public entry point is type-stable too.
            JET.@test_opt CGEF.Filtering.filter_field!(out, field, grid, kern, scale; backend = CGEF.Backends.SerialBackend())
        end
    end

    # Coordinate system and distance tests
    Test.@testset "Geometry" begin
        # 2D Cartesian
        geom_cart = CGEF.CartesianGeometry(1000.0, 1000.0)
        p1 = SA.SVector{2,Float64}(0.0, 0.0)
        p2 = SA.SVector{2,Float64}(3000.0, 4000.0)
        Test.@test CGEF.Geometry.distance(geom_cart, p1, p2) ≈ 5000.0
        Test.@test CGEF.Geometry.area_element(geom_cart) ≈ 1000.0 * 1000.0

        # Spherical
        geom_sph = CGEF.SphericalGeometry(6371000.0)
        # London (0.1278 W, 51.5074 N) to Paris (2.3522 E, 48.8566 N)
        # Coordinates in radians
        london = SA.SVector{2,Float64}(deg2rad(-0.1278), deg2rad(51.5074))
        paris  = SA.SVector{2,Float64}(deg2rad(2.3522), deg2rad(48.8566))
        d_km = CGEF.Geometry.distance(geom_sph, london, paris) / 1000.0
        Test.@test 340.0 < d_km < 350.0 # Paris-London ≈ 344 km

        # Coordinate projection conversions: to and from planetary Cartesian
        u_east, u_north = 10.0, -5.0
        λ, φ = deg2rad(-122.0), deg2rad(38.0) # San Francisco coords
        p_vel = CGEF.Geometry.to_planetary_cartesian(geom_sph, u_east, u_north, λ, φ)
        Test.@test length(p_vel) == 3

        l_vel = CGEF.Geometry.from_planetary_cartesian(geom_sph, p_vel[1], p_vel[2], p_vel[3], λ, φ)
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
        Test.@test CGEF.Kernels.kernel_weight(th, 10000.0, ℓ) == 1.0
        Test.@test CGEF.Kernels.kernel_weight(th, 60000.0, ℓ) == 0.0
        Test.@test CGEF.Kernels.kernel_radius(th, ℓ) == ℓ / 2

        # Gaussian: exponent is configurable; default 6 (Pope), 4 reproduces FlowSieve
        Test.@test CGEF.Kernels.kernel_weight(g, 0.0, ℓ) == 1.0
        Test.@test CGEF.Kernels.kernel_weight(g, ℓ, ℓ) ≈ exp(-6.0)
        Test.@test CGEF.Kernels.kernel_weight(g4, ℓ, ℓ) ≈ exp(-4.0)

        # Gaussian footprint truncates where the weight is negligible (~2ℓ for α=6, not the old 3ℓ)
        r = CGEF.Kernels.kernel_radius(g, ℓ)
        Test.@test 1.5ℓ < r < 2.5ℓ
        Test.@test CGEF.Kernels.kernel_weight(g, r, ℓ) < 1e-9
    end

    # Grids constructor and area calculations
    Test.@testset "Grids" begin
        geom = CGEF.CartesianGeometry(2000.0, 2000.0)
        lon = collect(0.0:2000.0:20000.0) # 11 points
        lat = collect(0.0:2000.0:10000.0) # 6 points
        mask = trues(11, 6) # active water grid

        grid = CGEF.StructuredGrid(geom, lon, lat, mask)
        Test.@test CGEF.Grids.size_tuple(grid) == (11, 6)
        Test.@test CGEF.Grids.area(grid, 2, 2) == 2000.0 * 2000.0
        Test.@test CGEF.Grids.coords(grid, 2, 3) == SA.SVector{2,Float64}(2000.0, 4000.0)

        # CurvilinearGrid coords bug test: verify i,j indices are used correctly
        # Create non-square grid to catch index swapping bugs
        lon_m = [Float64(i) for i in 1:10, j in 1:5]  # 10x5
        lat_m = [Float64(j*10) for i in 1:10, j in 1:5]  # lat varies with j
        areas_m = ones(10, 5)
        mask_m = trues(10, 5)
        cgrid = CGEF.CurvilinearGrid(geom, lon_m, lat_m, areas_m, mask_m)

        # coords(i,j) should return (lon[i,j], lat[i,j])
        pt = CGEF.Grids.coords(cgrid, 5, 3)
        Test.@test pt[1] == 5.0  # lon[5,3] = 5
        Test.@test pt[2] == 30.0 # lat[5,3] = 30

        # This test catches the bug where lat[j,j] was used instead of lat[i,j]
        pt_corner = CGEF.Grids.coords(cgrid, 10, 5)
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
        CGEF.Filtering.filter_field!(out, field, grid, CGEF.TopHatKernel(), 300.0)

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
        Test.@test_nowarn CGEF.Filtering.filter_field!(out_sph, field_sph, grid_sph, CGEF.TopHatKernel(), 1e6)
    end

    # Execution backend lattice (Backends.jl)
    Test.@testset "Backends" begin
        # resolve_backend returns INSTANCES; AutoBackend picks a concrete local backend
        Test.@test CGEF.Backends.resolve_backend(CGEF.Backends.SerialBackend()) === CGEF.Backends.SerialBackend()
        Test.@test CGEF.Backends.resolve_backend(CGEF.Backends.AutoBackend()) isa Union{CGEF.Backends.SerialBackend, CGEF.Backends.ThreadedBackend}

        # distribution wrappers are parametric over an inner local backend
        Test.@test CGEF.Backends.DistributedBackend(CGEF.Backends.SerialBackend()) isa CGEF.Backends.DistributedBackend
        Test.@test CGEF.Backends.MPIBackend() isa CGEF.Backends.MPIBackend
        Test.@test CGEF.Backends.DistributedBackend().inner === CGEF.Backends.SerialBackend()
        Test.@test CGEF.Backends.local_backend(CGEF.Backends.DistributedBackend(CGEF.Backends.ThreadedBackend())) === CGEF.Backends.ThreadedBackend()
        Test.@test CGEF.Backends.is_distributed(CGEF.Backends.MPIBackend()) && !CGEF.Backends.is_distributed(CGEF.Backends.SerialBackend())

        geom = CGEF.CartesianGeometry(1000.0, 1000.0)
        lon = collect(0.0:1000.0:20e3)
        lat = collect(0.0:1000.0:20e3)
        grid = CGEF.StructuredGrid(geom, lon, lat, trues(length(lon), length(lat)))
        field = rand(length(lon), length(lat))
        out_serial = zeros(size(field))
        out_default = zeros(size(field))
        CGEF.Filtering.filter_field!(out_serial, field, grid, CGEF.TopHatKernel(), 5000.0; backend = CGEF.Backends.SerialBackend())
        CGEF.Filtering.filter_field!(out_default, field, grid, CGEF.TopHatKernel(), 5000.0)  # AutoBackend
        Test.@test out_serial ≈ out_default

        # MPI extension IS loaded here (see the top-level `MPI.Init()`) — with a single rank, every
        # row is owned by rank 0, and `Allreduce!` over one rank is a no-op sum, so this must match
        # the serial reference exactly, the same way the "Distributed backend" testset below already
        # exercises the real `DistributedBackend` rather than expecting an error.
        out_mpi = zeros(size(field))
        CGEF.Filtering.filter_field!(out_mpi, field, grid, CGEF.TopHatKernel(), 5000.0; backend = CGEF.Backends.MPIBackend())
        Test.@test out_mpi ≈ out_serial
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
        plan = CGEF.Filtering.plan_filter(grid, kern, scale; backend = CGEF.Backends.SerialBackend())
        out_plan = zeros(size(u))
        out_direct = zeros(size(u))
        CGEF.Filtering.filter_apply!(out_plan, u, plan)
        CGEF.Filtering.filter_field!(out_direct, u, grid, kern, scale)
        Test.@test out_plan ≈ out_direct

        # Batched filter_fields! matches per-field filtering.
        ou = zeros(size(u)); ov = zeros(size(v))
        CGEF.Filtering.filter_fields!((ou, ov), (u, v), grid, kern, scale)
        ru = zeros(size(u)); rv = zeros(size(v))
        CGEF.Filtering.filter_field!(ru, u, grid, kern, scale)
        CGEF.Filtering.filter_field!(rv, v, grid, kern, scale)
        Test.@test ou ≈ ru
        Test.@test ov ≈ rv

        # Reapplying a prebuilt plan must NOT rebuild the footprint (a rebuild would allocate the
        # offset/weight vectors, ~kBs); a reused plan allocates essentially nothing.
        CGEF.Filtering.filter_apply!(out_plan, u, plan)  # warm up
        Test.@test (@allocated CGEF.Filtering.filter_apply!(out_plan, u, plan)) < 256
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
        CGEF.Filtering.filter_field!(out, field, grid, g, ℓ; method = CGEF.Filtering.Spectral())
        Test.@test out ≈ exp(-kx0^2 * ℓ^2 / 24) .* field rtol = 1e-10  # Gaussian α=6 transfer

        # DC (constant field) is preserved, Ĝ(0) = 1.
        cfield = fill(2.5, N, N)
        cout = zeros(N, N)
        CGEF.Filtering.filter_field!(cout, cfield, grid, g, ℓ; method = CGEF.Filtering.Spectral())
        Test.@test cout ≈ cfield

        # Sharp spectral cutoff: a mode below k_c passes, above k_c is removed.
        ss = CGEF.SharpSpectralKernel()
        sout = zeros(N, N)
        CGEF.Filtering.filter_field!(sout, field, grid, ss, L / 8; method = CGEF.Filtering.Spectral())  # k_c = 8π/L > 6π/L
        Test.@test sout ≈ field rtol = 1e-10
        CGEF.Filtering.filter_field!(sout, field, grid, ss, L; method = CGEF.Filtering.Spectral())       # k_c = π/L < 6π/L
        Test.@test maximum(abs, sout) < 1e-10

        # TopHat spectral is unsupported (Airy ringing) and errors with guidance.
        Test.@test_throws ArgumentError CGEF.Filtering.filter_field!(out, field, grid, CGEF.TopHatKernel(), ℓ; method = CGEF.Filtering.Spectral())

        # Non-periodic grid: spectral FFT must refuse.
        npgrid = CGEF.StructuredGrid(geom, x, y, trues(N, N))  # periodic = (false, false)
        Test.@test_throws ArgumentError CGEF.Filtering.filter_field!(out, field, npgrid, g, ℓ; method = CGEF.Filtering.Spectral())
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
        CGEF.Filtering.filter_field!(outf, u, sg, g, ℓ; method = CGEF.Filtering.Spectral())

        # The same points as a scattered (unstructured) grid; FINUFFT spectral filter.
        ptsx = vec([xi for xi in x, _ in y]); ptsy = vec([yj for _ in x, yj in y])
        ug = CGEF.UnstructuredGrid(geom, ptsx, ptsy, fill(dx*dy, Nx*Ny), trues(Nx*Ny))
        outu = zeros(Nx*Ny)
        CGEF.Filtering.filter_field!(outu, vec(u), ug, g, ℓ; method = CGEF.Filtering.Spectral())
        Test.@test reshape(outu, Nx, Ny) ≈ outf atol = 1e-7

        # Constant field ⇒ mean preserved (Ĝ(0)=1) for the scattered transform.
        cout = zeros(Nx*Ny)
        CGEF.Filtering.filter_field!(cout, fill(3.7, Nx*Ny), ug, g, ℓ; method = CGEF.Filtering.Spectral())
        Test.@test all(≈(3.7; atol = 1e-6), cout)

        # TopHat spectral still errors (shared transfer function).
        Test.@test_throws ArgumentError CGEF.Filtering.filter_field!(outu, vec(u), ug, CGEF.TopHatKernel(), ℓ; method = CGEF.Filtering.Spectral())
    end

    # Regression test: `spectral_filter_plan` used to derive the NUFFT mode count from
    # `grid.geometry.dx`/`dy` — a value that is only meaningful when it happens to match the actual
    # point spacing (as the test above does by construction: `dx=1.0` matches the real spacing
    # exactly, which is why it never caught this). For an `UnstructuredGrid`, `geometry` only carries
    # the geometry TYPE for dispatch; its `dx`/`dy` fields are not otherwise meaningful, so a caller
    # is free to build the grid with any placeholder `CartesianGeometry`, including one whose
    # dx/dy don't match the data's real spacing at all — exactly this case, `geometry.dx=1.0` on data
    # spaced 1000 m apart, silently produced a ~49-million-mode NUFFT (dividing a ~7000 m extent by a
    # placeholder dx of 1.0) and took ~120s/4 GiB for a 64-point `compute_Π!` call before being fixed
    # to derive the mode count from `npts` instead.
    Test.@testset "Spectral FINUFFT filtering: mode count independent of placeholder geometry.dx" begin
        placeholder_geom = CGEF.CartesianGeometry(1.0, 1.0)  # dx=1.0, deliberately NOT the real spacing
        Nx, Ny = 8, 8
        dx_real = 1000.0
        x = collect(0.0:dx_real:dx_real*(Nx-1)); y = collect(0.0:dx_real:dx_real*(Ny-1))
        ptsx = vec([xi for xi in x, _ in y]); ptsy = vec([yj for _ in x, yj in y])
        ug = CGEF.UnstructuredGrid(placeholder_geom, ptsx, ptsy, trues(Nx*Ny); k = 4)
        g = CGEF.GaussianKernel(); ℓ = 3000.0

        # A tight, generous performance budget: the pre-fix version took ~120s/4 GiB for this exact
        # scenario (compute_Π! on this grid); a correct implementation is milliseconds/KB.
        out = zeros(Nx*Ny)
        CGEF.Filtering.filter_field!(out, fill(3.7, Nx*Ny), ug, g, ℓ; method = CGEF.Filtering.Spectral())  # warm up
        t = @elapsed CGEF.Filtering.filter_field!(out, fill(3.7, Nx*Ny), ug, g, ℓ; method = CGEF.Filtering.Spectral())
        b = @allocated CGEF.Filtering.filter_field!(out, fill(3.7, Nx*Ny), ug, g, ℓ; method = CGEF.Filtering.Spectral())
        Test.@test t < 1.0
        Test.@test b < 10_000_000
        Test.@test all(x -> isapprox(x, 3.7; atol = 1e-6), out)

        Π = zeros(Nx*Ny)
        u = vec([sin(xi/700)*cos(yj/900) for xi in x, yj in y]); v = vec([cos(xi/500)*sin(yj/1100) for xi in x, yj in y])
        CGEF.Diagnostics.compute_Π!(Π, u, v, nothing, ug, g, ℓ)  # warm up (compile) before timing
        t_pi = @elapsed CGEF.Diagnostics.compute_Π!(Π, u, v, nothing, ug, g, ℓ)
        b_pi = @allocated CGEF.Diagnostics.compute_Π!(Π, u, v, nothing, ug, g, ℓ)
        Test.@test t_pi < 1.0
        Test.@test b_pi < 10_000_000
        Test.@test all(isfinite, Π)
    end

    # Uniform-spherical spectral filtering (spherical harmonics): a single degree-l harmonic is an
    # exact eigenfunction, scaled by Ĝ(k_l) with k_l = √(l(l+1))/R.
    Test.@testset "Spectral spherical-harmonic filtering" begin
        N = 24; M = 2N - 1
        Θ, Φ = FSH.sph_points(N)
        R = 1.0
        geom = CGEF.SphericalGeometry(R)
        grid = CGEF.StructuredGrid(geom, collect(Φ), π/2 .- collect(Θ), trues(M, N))
        ker = CGEF.GaussianKernel(); ℓ = 1.0

        l, m = 5, 2
        C0 = zeros(N, M); C0[FSH.sph_mode(l, m)] = 1.0
        field = permutedims(FSH.sph_evaluate(C0))    # CGEF [lon, lat]
        out = zeros(M, N)
        CGEF.Filtering.filter_field!(out, field, grid, ker, ℓ; method = CGEF.Filtering.Spectral())
        Ghat = exp(-(l*(l+1)/R^2) * ℓ^2 / 24)        # Gaussian α=6
        Test.@test out ≈ Ghat .* field atol = 1e-12

        # l = 0 (mean) preserved.
        cout = zeros(M, N)
        CGEF.Filtering.filter_field!(cout, fill(2.3, M, N), grid, ker, ℓ; method = CGEF.Filtering.Spectral())
        Test.@test all(≈(2.3; atol = 1e-10), cout)

        # Grid that is not an FSH grid (M ≠ 2N-1) is rejected.
        badgrid = CGEF.StructuredGrid(geom, collect(0.0:0.1:1.0), collect(0.0:0.1:1.0), trues(11, 11))
        Test.@test_throws ArgumentError CGEF.Filtering.filter_field!(zeros(11, 11), rand(11, 11), badgrid, ker, ℓ; method = CGEF.Filtering.Spectral())

        # Shape-correct (M = 2N-1) but NOT on the actual FSH quadrature nodes: must still be rejected,
        # not silently accepted and given a meaningless transform.
        wronggrid = CGEF.StructuredGrid(
            geom, collect(range(0.0, 2π; length = M + 1)[1:M]), collect(range(π/2, -π/2; length = N)), trues(M, N),
        )
        Test.@test_throws ArgumentError CGEF.Filtering.filter_field!(zeros(M, N), rand(M, N), wronggrid, ker, ℓ; method = CGEF.Filtering.Spectral())
    end

    # Scattered-spherical spectral filtering (NUFSHT): on a Clenshaw–Curtis grid passed as scattered
    # points the adjoint analysis is exact, so a single degree-l harmonic is scaled by exactly Ĝ(k_l).
    Test.@testset "Spectral NUFSHT filtering" begin
        L = 12; N = L + 1; M = 2N - 1
        Θ, Φ = FSH.sph_points(N)
        R = 6.371e6
        geom = CGEF.SphericalGeometry(R)
        lat = vec([π/2 - θ for θ in Θ, φ in Φ])
        lon = vec([φ for θ in Θ, φ in Φ])
        npts = length(lat)
        ug = CGEF.UnstructuredGrid(geom, lon, lat, ones(npts), trues(npts))

        l, m = 4, 1
        C0 = zeros(N, M); C0[FSH.sph_mode(l, m)] = 1.0
        Fgrid = FSH.sph_evaluate(C0)
        # Must flatten in the SAME (column-major) order as `lat`/`lon` above (`vec` of a `(θ,φ)`
        # matrix comprehension, θ fastest) — a `for it in 1:N for ip in 1:M` double-for flattens
        # with ip (φ) fastest instead, silently pairing each `field` value with the WRONG (lat,lon)
        # point. That mismatch is what was actually failing here, not the spectral filter itself:
        # confirmed by comparing the two flattenings directly (32% error vs 1.5e-9 with this fix).
        field = vec([Fgrid[it, ip] for it in 1:N, ip in 1:M])

        scale = 2e6; ker = CGEF.GaussianKernel()
        out = zeros(npts)
        CGEF.Filtering.filter_field!(out, field, ug, ker, scale; method = CGEF.Filtering.Spectral())
        kl = sqrt(l*(l+1)) / R
        Ghat = exp(-kl^2 * scale^2 / 24)
        Test.@test out ≈ Ghat .* field rtol = 1e-7
    end

    # Threaded backend must agree with serial EXACTLY (shared footprint engine), including masking
    # and periodic wrapping — the old hand-rolled threaded path silently disagreed on periodicity.
    Test.@testset "Threaded backend (OhMyThreads)" begin
        geom = CGEF.CartesianGeometry(1000.0, 1000.0)
        lon = collect(0.0:1000.0:30e3)
        lat = collect(0.0:1000.0:30e3)
        u = rand(length(lon), length(lat))

        grid = CGEF.StructuredGrid(geom, lon, lat, trues(length(lon), length(lat)))
        for strat in (CGEF.Filtering.Deformable(), CGEF.Filtering.ZeroFill())
            os = zeros(size(u)); ot = zeros(size(u))
            CGEF.Filtering.filter_field!(os, u, grid, CGEF.TopHatKernel(), 5000.0; backend = CGEF.Backends.SerialBackend(), mask_strategy = strat)
            CGEF.Filtering.filter_field!(ot, u, grid, CGEF.TopHatKernel(), 5000.0; backend = CGEF.Backends.ThreadedBackend(), mask_strategy = strat)
            Test.@test ot ≈ os
        end

        # Masked Cartesian
        mask = trues(length(lon), length(lat)); mask[5:8, 5:8] .= false
        mgrid = CGEF.StructuredGrid(geom, lon, lat, mask)
        os = zeros(size(u)); ot = zeros(size(u))
        CGEF.Filtering.filter_field!(os, u, mgrid, CGEF.GaussianKernel(), 4000.0; backend = CGEF.Backends.SerialBackend())
        CGEF.Filtering.filter_field!(ot, u, mgrid, CGEF.GaussianKernel(), 4000.0; backend = CGEF.Backends.ThreadedBackend())
        Test.@test ot ≈ os

        # Periodic global spherical grid (threaded must wrap exactly like serial)
        sgeom = CGEF.SphericalGeometry(6371000.0)
        slon = deg2rad.(collect(0.0:5.0:355.0))
        slat = deg2rad.(collect(-40.0:5.0:40.0))
        sgrid = CGEF.StructuredGrid(sgeom, slon, slat, trues(length(slon), length(slat)))
        su = rand(length(slon), length(slat))
        oss = zeros(size(su)); ost = zeros(size(su))
        CGEF.Filtering.filter_field!(oss, su, sgrid, CGEF.TopHatKernel(), deg2rad(15.0) * 6371000.0; backend = CGEF.Backends.SerialBackend())
        CGEF.Filtering.filter_field!(ost, su, sgrid, CGEF.TopHatKernel(), deg2rad(15.0) * 6371000.0; backend = CGEF.Backends.ThreadedBackend())
        Test.@test ost ≈ oss
    end

    # ThreadedBackend for 1D/true-3D StructuredGrid: these use the point-indexed FilterFootprintND/
    # FilterFootprintNDScattered representation (no row structure), but the per-point kernel is
    # data-race-free (reads neighbours, writes only its own cell) — verify it matches serial exactly,
    # covering both the fast (Range-axis, translation-invariant) and general (nonuniform/spherical
    # scattered) footprint paths. Distributed/GPU/MPI remain unsupported here (still row-only) and
    # must raise a clear error when requested explicitly, per `_check_backend_compatible`.
    Test.@testset "Threaded backend: 1D/true-3D StructuredGrid (ND footprint)" begin
        # 1D Cartesian, uniform (Range) axis -> fast FilterFootprintND path.
        geom1 = CGEF.CartesianGeometry(1000.0, 1000.0)
        x1 = collect(0.0:1000.0:30e3)
        grid1 = CGEF.StructuredGrid(geom1, x1, trues(length(x1)))
        u1 = rand(length(x1))
        os1 = zeros(size(u1)); ot1 = zeros(size(u1))
        CGEF.Filtering.filter_field!(os1, u1, grid1, CGEF.TopHatKernel(), 5000.0; backend = CGEF.Backends.SerialBackend())
        CGEF.Filtering.filter_field!(ot1, u1, grid1, CGEF.TopHatKernel(), 5000.0; backend = CGEF.Backends.ThreadedBackend())
        Test.@test ot1 ≈ os1

        # True-3D Cartesian, uniform axes -> fast FilterFootprintND path, with a mask.
        geom3 = CGEF.CartesianGeometry(1000.0, 1000.0, 500.0)
        x3 = collect(0.0:1000.0:10e3); y3 = collect(0.0:1000.0:10e3); z3 = collect(0.0:500.0:4e3)
        mask3 = trues(length(x3), length(y3), length(z3)); mask3[3:5, 3:5, 2] .= false
        grid3 = CGEF.StructuredGrid(geom3, x3, y3, z3, mask3)
        u3 = rand(length(x3), length(y3), length(z3))
        os3 = zeros(size(u3)); ot3 = zeros(size(u3))
        CGEF.Filtering.filter_field!(os3, u3, grid3, CGEF.GaussianKernel(), 2000.0; backend = CGEF.Backends.SerialBackend())
        CGEF.Filtering.filter_field!(ot3, u3, grid3, CGEF.GaussianKernel(), 2000.0; backend = CGEF.Backends.ThreadedBackend())
        Test.@test ot3 ≈ os3

        # True-3D SPHERICAL (general/scattered footprint path — spherical never uses the translation-
        # invariant fast path, see `build_footprint`'s Cartesian-only fast dispatch).
        R = 6.371e6
        sgeom = CGEF.SphericalGeometry(R)
        slon = deg2rad.(collect(0.0:20.0:340.0))
        slat = deg2rad.(collect(-40.0:20.0:40.0))
        sr = collect(R:100e3:(R + 300e3))
        sgrid = CGEF.StructuredGrid(sgeom, slon, slat, sr, trues(length(slon), length(slat), length(sr)))
        su3 = rand(length(slon), length(slat), length(sr))
        oss3 = zeros(size(su3)); ost3 = zeros(size(su3))
        CGEF.Filtering.filter_field!(oss3, su3, sgrid, CGEF.TopHatKernel(), 150e3; backend = CGEF.Backends.SerialBackend())
        CGEF.Filtering.filter_field!(ost3, su3, sgrid, CGEF.TopHatKernel(), 150e3; backend = CGEF.Backends.ThreadedBackend())
        Test.@test ost3 ≈ oss3

        # Distributed/GPU/MPI still have no ND hook -- an explicit request must error, not silently
        # downgrade to serial.
        Test.@test_throws ArgumentError CGEF.Filtering.filter_field!(os1, u1, grid1, CGEF.TopHatKernel(), 5000.0; backend = CGEF.Backends.DistributedBackend())
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
        CGEF.Filtering.filter_field!(os, u, grid, CGEF.TopHatKernel(), 5000.0; backend = CGEF.Backends.SerialBackend())
        CGEF.Filtering.filter_field!(od, u, grid, CGEF.TopHatKernel(), 5000.0; backend = CGEF.Backends.DistributedBackend())
        Test.@test od ≈ os
    end

    # GPU backend on the KernelAbstractions CPU device must match serial (validates the GPU kernel
    # logic here; actual GPU hardware is exercised separately). Same engine ⇒ masking + periodicity
    # consistent.
    Test.@testset "GPU backend (KernelAbstractions CPU)" begin
        gpu = CGEF.Backends.GPUBackend(KA.CPU())
        geom = CGEF.CartesianGeometry(1000.0, 1000.0)
        lon = collect(0.0:1000.0:20e3)
        lat = collect(0.0:1000.0:20e3)
        mask = trues(length(lon), length(lat)); mask[5:7, 5:7] .= false
        grid = CGEF.StructuredGrid(geom, lon, lat, mask)
        u = rand(length(lon), length(lat))
        for strat in (CGEF.Filtering.Deformable(), CGEF.Filtering.ZeroFill())
            os = zeros(size(u)); og = zeros(size(u))
            CGEF.Filtering.filter_field!(os, u, grid, CGEF.TopHatKernel(), 5000.0; backend = CGEF.Backends.SerialBackend(), mask_strategy = strat)
            CGEF.Filtering.filter_field!(og, u, grid, CGEF.TopHatKernel(), 5000.0; backend = gpu, mask_strategy = strat)
            Test.@test og ≈ os
        end

        # Periodic global spherical grid
        sgeom = CGEF.SphericalGeometry(6371000.0)
        slon = deg2rad.(collect(0.0:5.0:355.0))
        slat = deg2rad.(collect(-30.0:5.0:30.0))
        sgrid = CGEF.StructuredGrid(sgeom, slon, slat, trues(length(slon), length(slat)))
        su = rand(length(slon), length(slat))
        oss = zeros(size(su)); osg = zeros(size(su))
        CGEF.Filtering.filter_field!(oss, su, sgrid, CGEF.TopHatKernel(), deg2rad(15.0) * 6371000.0; backend = CGEF.Backends.SerialBackend())
        CGEF.Filtering.filter_field!(osg, su, sgrid, CGEF.TopHatKernel(), deg2rad(15.0) * 6371000.0; backend = gpu)
        Test.@test osg ≈ oss
    end

    # True n-D Cartesian filtering (1D + 3D) via the general footprint engine.
    Test.@testset "n-D filtering (1D + true 3D Cartesian)" begin
        # --- 1D ---
        geom1 = CGEF.CartesianGeometry(1.0, 1.0)
        x = collect(0.0:1.0:50.0)
        grid1 = CGEF.StructuredGrid(geom1, x, trues(length(x)))
        Test.@test CGEF.Grids.size_tuple(grid1) == (length(x),)
        # constant -> constant (normalization)
        o1 = zeros(length(x))
        CGEF.Filtering.filter_field!(o1, fill(7.0, length(x)), grid1, CGEF.TopHatKernel(), 6.0)
        Test.@test all(≈(7.0), o1[10:40])
        # sub-grid scale -> identity (only the self cell is in support)
        g1 = rand(length(x)); oi = zeros(length(x))
        CGEF.Filtering.filter_field!(oi, g1, grid1, CGEF.TopHatKernel(), 0.5)
        Test.@test oi ≈ g1

        # --- 3D (dz ≫ footprint, so only the in-plane disk contributes) ---
        geom3 = CGEF.CartesianGeometry(1.0, 1.0, 100.0)
        x3 = collect(0.0:1.0:20.0); y3 = collect(0.0:1.0:20.0); z3 = collect(0.0:100.0:300.0)
        nx, ny, nz = length(x3), length(y3), length(z3)
        grid3 = CGEF.StructuredGrid(geom3, x3, y3, z3, trues(nx, ny, nz))
        Test.@test CGEF.Grids.size_tuple(grid3) == (nx, ny, nz)
        # constant -> constant
        o3 = zeros(nx, ny, nz)
        CGEF.Filtering.filter_field!(o3, fill(3.5, nx, ny, nz), grid3, CGEF.TopHatKernel(), 6.0)
        Test.@test all(≈(3.5), o3)

        # A z-invariant 3D field must reduce EXACTLY to the 2D filter of its slice (dz ≫ rad ⇒ no
        # vertical neighbours), validating the n-D engine against the 2D engine.
        f2d = rand(nx, ny)
        f3z = repeat(reshape(f2d, nx, ny, 1), 1, 1, nz)
        o3z = zeros(nx, ny, nz)
        CGEF.Filtering.filter_field!(o3z, f3z, grid3, CGEF.TopHatKernel(), 6.0)
        grid2 = CGEF.StructuredGrid(CGEF.CartesianGeometry(1.0, 1.0), x3, y3, trues(nx, ny))
        o2 = zeros(nx, ny)
        CGEF.Filtering.filter_field!(o2, f2d, grid2, CGEF.TopHatKernel(), 6.0)
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
        CGEF.Diagnostics.compute_Π!(Πc, fill(2.0, nx, ny, nz), fill(-3.0, nx, ny, nz),
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
        CGEF.Diagnostics.compute_Π!(Π3, u3, v3, w3, grid3, ker, ℓ)

        grid2 = CGEF.StructuredGrid(CGEF.CartesianGeometry(1.0, 1.0), x, y, trues(nx, ny))
        Π2 = zeros(nx, ny)
        CGEF.Diagnostics.compute_Π!(Π2, u2, v2, nothing, grid2, ker, ℓ)
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
        CGEF.Derivatives.ddx!(∂f∂x, f, grid)

        Test.@test ∂f∂x[2, 3] ≈ 3.0
        Test.@test ∂f∂x[1, 3] ≈ 3.0 # forward difference at boundary
        Test.@test ∂f∂x[6, 3] ≈ 3.0 # backward difference at boundary
    end

    # Nonuniform-axis correctness (not just "doesn't regress on a uniform grid"): the standard
    # 3-point nonuniform stencil is EXACT (to floating point) for any quadratic function of its
    # input coordinate, on ANY spacing pattern -- a much stronger and simpler proof than a
    # convergence sweep, and one that would fail outright under the old bug (global Δ read once
    # from the first two samples and reused everywhere).
    Test.@testset "Nonuniform axes" begin
        # --- Cartesian: geometrically-stretched axis (genuinely nonuniform, no constant step) ---
        geom = CGEF.CartesianGeometry(1.0, 1.0) # dx/dy unused on this path; area comes from the axis
        lon_nu = [0.0, 1.0, 2.5, 4.0, 7.0, 12.0, 19.0] # strictly increasing, non-constant gaps
        lat_nu = [0.0, 0.5, 1.5, 3.5, 7.5]
        Nlon_nu, Nlat_nu = length(lon_nu), length(lat_nu)
        mask_nu = trues(Nlon_nu, Nlat_nu)
        grid_nu = CGEF.StructuredGrid(geom, lon_nu, lat_nu, mask_nu)

        # `lon_nu` is a plain Vector, not a Range -- confirm this actually takes the general
        # (nonuniform-safe) footprint path, not the fast Range-only path.
        fp_nu = CGEF.Filtering.build_footprint(grid_nu, CGEF.TopHatKernel(), 2.0)
        Test.@test fp_nu isa CGEF.Filtering.FilterFootprintScattered

        # f(x,y) = x^2 + y^2: interior centered stencil is exact for a quadratic on any spacing.
        f_quad = [lon_nu[i]^2 + lat_nu[j]^2 for i in 1:Nlon_nu, j in 1:Nlat_nu]
        ∂f∂x_nu = zeros(Nlon_nu, Nlat_nu)
        ∂f∂y_nu = zeros(Nlon_nu, Nlat_nu)
        CGEF.Derivatives.ddx!(∂f∂x_nu, f_quad, grid_nu)
        CGEF.Derivatives.ddy!(∂f∂y_nu, f_quad, grid_nu)
        for i in 2:(Nlon_nu-1), j in 1:Nlat_nu
            Test.@test ∂f∂x_nu[i, j] ≈ 2 * lon_nu[i] atol=1e-10
        end
        for i in 1:Nlon_nu, j in 2:(Nlat_nu-1)
            Test.@test ∂f∂y_nu[i, j] ≈ 2 * lat_nu[j] atol=1e-10
        end

        # Filtering a constant field must return the constant unchanged even on a nonuniform grid
        # (this exercises the scattered/per-point footprint path end to end).
        field_const = fill(7.5, Nlon_nu, Nlat_nu)
        out_const = zeros(Nlon_nu, Nlat_nu)
        CGEF.Filtering.filter_field!(out_const, field_const, grid_nu, CGEF.TopHatKernel(), 3.0)
        Test.@test all(x -> isapprox(x, 7.5; atol=1e-10), out_const)

        # --- Spherical: nonuniform lon/lat. f(λ,φ)=λ has EXACT physical x-derivative 1/(R cosφ)
        # and f(λ,φ)=φ has EXACT physical y-derivative 1/R, on ANY spacing pattern (verified
        # algebraically: the nonuniform stencil applied to a coordinate-linear field returns the
        # exact analytic slope regardless of h_m/h_p) -- this is what the old bug (global Δ from
        # `lon[2]-lon[1]`) would have gotten wrong away from the first grid cell.
        sgeom = CGEF.SphericalGeometry(6371000.0)
        lon_s = deg2rad.([0.0, 3.0, 8.0, 20.0, 45.0, 46.0, 90.0])
        lat_s = deg2rad.([-40.0, -35.0, -20.0, 0.0, 10.0, 30.0])
        Nlon_s, Nlat_s = length(lon_s), length(lat_s)
        mask_s = trues(Nlon_s, Nlat_s)
        grid_s = CGEF.StructuredGrid(sgeom, lon_s, lat_s, mask_s; periodic = false)
        R = sgeom.R

        f_lon = [lon_s[i] for i in 1:Nlon_s, j in 1:Nlat_s]
        f_lat = [lat_s[j] for i in 1:Nlon_s, j in 1:Nlat_s]
        ∂flon∂x = zeros(Nlon_s, Nlat_s)
        ∂flat∂y = zeros(Nlon_s, Nlat_s)
        CGEF.Derivatives.ddx!(∂flon∂x, f_lon, grid_s)
        CGEF.Derivatives.ddy!(∂flat∂y, f_lat, grid_s)
        for i in 2:(Nlon_s-1), j in 1:Nlat_s
            Test.@test ∂flon∂x[i, j] ≈ 1 / (R * cos(lat_s[j])) rtol=1e-8
        end
        for i in 1:Nlon_s, j in 2:(Nlat_s-1)
            Test.@test ∂flat∂y[i, j] ≈ 1 / R rtol=1e-8
        end

        # Regional (non-periodic) grid: the boundary derivative must NOT silently wrap to the far
        # edge (an adjacent bug fixed alongside the nonuniform-spacing one -- the old spherical
        # `ddx!` wrapped unconditionally regardless of `grid.periodic`). One-sided differences at
        # the true domain edge should use only the interior neighbour, not the opposite edge.
        Test.@test !CGEF.Grids.isperiodic(grid_s, 1)
        # Perturbing the far-edge value must NOT change the near-edge (i=1) derivative on a
        # non-periodic grid -- if it wrapped, it would.
        f_lon2 = copy(f_lon)
        f_lon2[end, 3] += 1000.0 # perturb the far edge only
        ∂flon∂x2 = zeros(Nlon_s, Nlat_s)
        CGEF.Derivatives.ddx!(∂flon∂x2, f_lon2, grid_s)
        Test.@test ∂flon∂x2[1, 3] ≈ ∂flon∂x[1, 3]

        # Solid-body rotation on a NONUNIFORM spherical grid: u = U*cos(φ), v = 0. Filtering is
        # performed in planetary-Cartesian coordinates (Aluie 2019 commutativity formulation)
        # specifically so that this rigid-rotation field commutes with filtering; Π should be ≈ 0
        # regardless of axis nonuniformity.
        U = 5.0
        u_rot = [U * cos(lat_s[j]) for i in 1:Nlon_s, j in 1:Nlat_s]
        v_rot = zeros(Nlon_s, Nlat_s)
        Π_s = zeros(Nlon_s, Nlat_s)
        CGEF.Diagnostics.compute_Π!(Π_s, u_rot, v_rot, nothing, grid_s, CGEF.TopHatKernel(), 5e5)
        for i in 2:(Nlon_s-1), j in 2:(Nlat_s-1)
            Test.@test Π_s[i, j] ≈ 0.0 atol=1e-6
        end
    end

    # Real convergence-RATE tests: refine resolution 3x (not just "error is small at one
    # resolution") and assert the observed order matches the claimed 2nd-order accuracy of the
    # centered nonuniform stencil (`nonuniform_first_derivative`), for a field whose third
    # derivative is genuinely nonzero (sin, unlike the quadratic-exactness checks above, which are
    # exact at ANY resolution and so can't demonstrate a convergence RATE at all).
    Test.@testset "Convergence rate: ddx! is genuinely 2nd order" begin
        geom = CGEF.CartesianGeometry(1.0, 1.0)  # dx/dy unused; area/spacing come from the axis
        L = 100.0
        # A single wavelength keeps k·h small even at the coarsest resolution tested (k·h ≈ 0.16 at
        # N=40), safely inside the asymptotic h² regime where the NEXT Taylor term (O((k·h)²) smaller
        # than the leading one) is negligible — a few wavelengths would make the coarsest doubling's
        # observed rate noticeably pulled away from 2 by that pre-asymptotic correction.
        k = 2π / L
        Ns = (40, 80, 160, 320)  # 3 successive doublings
        errs = Float64[]
        for N in Ns
            x = collect(range(0.0, L; length = N))
            y = collect(0.0:1.0:2.0)
            grid = CGEF.StructuredGrid(geom, x, y, trues(N, length(y)))
            f = [sin(k * xi) for xi in x, _ in y]
            df = zeros(N, length(y))
            CGEF.Derivatives.ddx!(df, f, grid)
            exact = [k * cos(k * xi) for xi in x, _ in y]
            # Interior only: the one-sided boundary stencil is 1st order by construction (a
            # separate, correct behavior — not what this test is checking).
            interior = 3:(N - 2)
            push!(errs, maximum(abs, df[interior, :] .- exact[interior, :]))
        end
        # Halving h should reduce error by ~4x for a genuinely 2nd-order stencil: assert the
        # observed rate log2(err_N / err_2N) is close to 2 for each successive doubling, not just
        # that some fixed-resolution error is "small enough."
        for k_idx in 1:(length(Ns) - 1)
            rate = log2(errs[k_idx] / errs[k_idx + 1])
            Test.@test 1.8 < rate < 2.2
        end
    end

    # CurvilinearGrid support built from scratch (Stage 3): WLSQ tangent-plane gradients, exact
    # corner-based quadrilateral areas, real-space scattered-footprint filtering, and the full
    # compute_Π!/coarse_grain pipeline, cross-checked against the StructuredGrid engine.
    Test.@testset "CurvilinearGrid (WLSQ / areas / pipeline)" begin
        cart = CGEF.CartesianGeometry(1.0, 1.0)

        # --- HARD GATE: a "fake curvilinear" separable grid with UNIFORM (equally spaced) axes must
        # reproduce the StructuredGrid derivatives EXACTLY (to floating point). Distinct dx≠dy and a
        # linear+quadratic field make this sensitive to any axis-swap / transpose / sign error. On a
        # uniform separable stencil the WLSQ normal matrix is diagonal and the fit decouples per axis,
        # reducing exactly to the centered/one-sided difference the structured engine uses. ---
        xs = collect(0.0:2.0:20.0)   # dx = 2
        ys = collect(0.0:5.0:30.0)   # dy = 5 (distinct spacing)
        Nx, Ny = length(xs), length(ys)
        sgrid = CGEF.StructuredGrid(cart, xs, ys, trues(Nx, Ny))
        lon = [xs[i] for i in 1:Nx, j in 1:Ny]
        lat = [ys[j] for i in 1:Nx, j in 1:Ny]
        cgrid = CGEF.CurvilinearGrid(cart, lon, lat, trues(Nx, Ny))
        dplan = CGEF.Derivatives.WLSQGradientPlan(cgrid)

        f = [3.0*xs[i] - 2.0*ys[j] + 0.5*xs[i]^2 + 0.25*ys[j]^2 for i in 1:Nx, j in 1:Ny]
        sx = zeros(Nx, Ny); sy = zeros(Nx, Ny); cx = zeros(Nx, Ny); cy = zeros(Nx, Ny)
        CGEF.Derivatives.ddx!(sx, f, sgrid); CGEF.Derivatives.ddy!(sy, f, sgrid)
        CGEF.Derivatives.ddx!(cx, f, cgrid, dplan); CGEF.Derivatives.ddy!(cy, f, cgrid, dplan)
        Test.@test maximum(abs.(cx .- sx)) < 1e-10   # hard gate (ddx exact vs structured)
        Test.@test maximum(abs.(cy .- sy)) < 1e-10   # hard gate (ddy exact vs structured)

        # --- Non-orthogonal (sheared + rotated) curvilinear grid: WLSQ reconstructs a LINEAR field's
        # gradient exactly on ANY stencil (it cancels the leading truncation term), so a known
        # analytic gradient is recovered to floating point at every node — the specific case a
        # transposed 2×2 solve would fail, since here the normal matrix is genuinely non-diagonal
        # (Axy ≠ 0). ---
        a, b, c, d = 2.0, 0.7, -0.4, 3.0   # non-orthogonal affine index→physical map (|det| = 6.28)
        Ni, Nj = 9, 7
        slon = [a*i + b*j for i in 1:Ni, j in 1:Nj]
        slat = [c*i + d*j for i in 1:Ni, j in 1:Nj]
        shear = CGEF.CurvilinearGrid(cart, slon, slat, trues(Ni, Nj))
        splan = CGEF.Derivatives.WLSQGradientPlan(shear)
        p, q = 1.3, -2.1
        g = [p*slon[i,j] + q*slat[i,j] for i in 1:Ni, j in 1:Nj]
        gx = zeros(Ni, Nj); gy = zeros(Ni, Nj)
        CGEF.Derivatives.ddx!(gx, g, shear, splan); CGEF.Derivatives.ddy!(gy, g, shear, splan)
        Test.@test maximum(abs.(gx .- p)) < 1e-10
        Test.@test maximum(abs.(gy .- q)) < 1e-10

        # --- Nonuniform separable grid: WLSQ is a LINEAR reconstruction, so (per the plan) it agrees
        # with the structured nonuniform result only to within its truncation bound, NOT to floating
        # point for a quadratic. It is, however, exact for a linear field on any spacing — assert
        # that clean, honest property against both engines. ---
        xnu = [0.0, 1.0, 2.5, 4.0, 7.0, 12.0]
        ynu = [0.0, 0.5, 1.5, 3.5, 7.5]
        Nxn, Nyn = length(xnu), length(ynu)
        sgnu = CGEF.StructuredGrid(cart, xnu, ynu, trues(Nxn, Nyn))
        lonn = [xnu[i] for i in 1:Nxn, j in 1:Nyn]
        latn = [ynu[j] for i in 1:Nxn, j in 1:Nyn]
        cgnu = CGEF.CurvilinearGrid(cart, lonn, latn, trues(Nxn, Nyn))
        flin = [2.0*xnu[i] + 3.0*ynu[j] for i in 1:Nxn, j in 1:Nyn]
        gxs = zeros(Nxn, Nyn); gxc = zeros(Nxn, Nyn); gyc = zeros(Nxn, Nyn)
        CGEF.Derivatives.ddx!(gxs, flin, sgnu)
        CGEF.Derivatives.ddx!(gxc, flin, cgnu); CGEF.Derivatives.ddy!(gyc, flin, cgnu)
        Test.@test maximum(abs.(gxc .- 2.0)) < 1e-10   # WLSQ exact for linear on nonuniform stencil
        Test.@test maximum(abs.(gyc .- 3.0)) < 1e-10
        Test.@test maximum(abs.(gxs .- 2.0)) < 1e-10   # structured also exact for linear

        # --- Exact corner-based quadrilateral areas sum to the true domain area. A sheared
        # parallelogram mesh: each cell is a parallelogram of area |ad-bc|, so the total is
        # |ad-bc|·Ncell_i·Ncell_j (the area of the enclosing parallelogram). ---
        ax, bx, cx2, dx2 = 2.0, 0.5, 0.0, 3.0   # |det| = 6
        Nci, Ncj = 5, 4
        lonc = [ax*(i-1) + bx*(j-1) for i in 1:(Nci+1), j in 1:(Ncj+1)]
        latc = [cx2*(i-1) + dx2*(j-1) for i in 1:(Nci+1), j in 1:(Ncj+1)]
        cen_lon = [(lonc[i,j]+lonc[i+1,j]+lonc[i+1,j+1]+lonc[i,j+1])/4 for i in 1:Nci, j in 1:Ncj]
        cen_lat = [(latc[i,j]+latc[i+1,j]+latc[i+1,j+1]+latc[i,j+1])/4 for i in 1:Nci, j in 1:Ncj]
        agrid = CGEF.CurvilinearGrid(cart, cen_lon, cen_lat, trues(Nci, Ncj);
                                     lon_corner=lonc, lat_corner=latc)
        Test.@test sum(agrid.areas) ≈ abs(ax*dx2 - bx*cx2) * Nci * Ncj
        Test.@test all(≈(6.0), agrid.areas)

        # --- Spherical corner-based quadrilateral area is diagonal-invariant: splitting the SAME
        # quadrilateral along its other diagonal must give the identical area (both decompositions
        # describe the identical enclosed region) — an exact identity (zero tolerance, no limit or
        # approximation involved), unlike comparing to a lon/lat "zonal band" formula, which is a
        # DIFFERENT shape (a graticule cell's east/west edges are parallels — small circles — not the
        # great-circle arcs `_quad_area` uses), so it only coincides with the geodesic-quadrilateral
        # area in the Δφ→0 limit and was the wrong thing to compare against here. ---
        sph = CGEF.SphericalGeometry(6371000.0)
        λc = deg2rad.(collect(0.0:2.0:10.0))
        φc = deg2rad.(collect(10.0:2.0:20.0))
        slonc = [λc[i] for i in 1:length(λc), j in 1:length(φc)]
        slatc = [φc[j] for i in 1:length(λc), j in 1:length(φc)]
        Ncλ, Ncφ = length(λc)-1, length(φc)-1
        scen_lon = [(slonc[i,j]+slonc[i+1,j])/2 for i in 1:Ncλ, j in 1:Ncφ]
        scen_lat = [(slatc[i,j]+slatc[i,j+1])/2 for i in 1:Ncλ, j in 1:Ncφ]
        sagrid = CGEF.CurvilinearGrid(sph, scen_lon, scen_lat, trues(Ncλ, Ncφ);
                                      lon_corner=slonc, lat_corner=slatc)
        for j in 1:Ncφ, i in 1:Ncλ
            λ1, φ1 = slonc[i,j],     slatc[i,j]
            λ2, φ2 = slonc[i+1,j],   slatc[i+1,j]
            λ3, φ3 = slonc[i+1,j+1], slatc[i+1,j+1]
            λ4, φ4 = slonc[i,j+1],   slatc[i,j+1]
            other_diag = CGEF.Grids._quad_area(sph, λ2, φ2, λ3, φ3, λ4, φ4, λ1, φ1)
            Test.@test sagrid.areas[i,j] ≈ other_diag rtol=1e-12
        end

        # --- Real-space filtering on a curvilinear grid: a constant field is returned unchanged. ---
        cfld = fill(7.5, Nx, Ny)
        cfo = zeros(Nx, Ny)
        CGEF.Filtering.filter_field!(cfo, cfld, cgrid, CGEF.TopHatKernel(), 6.0)
        Test.@test all(x -> isapprox(x, 7.5; atol=1e-10), cfo)

        # --- Full compute_Π!/coarse_grain pipeline: on the uniform Cartesian "fake curvilinear" grid
        # the result must match the StructuredGrid pipeline on the identical coordinates (derivatives
        # are exact and the footprints carry identical weights/neighbours). ---
        uu = [sin(xs[i]/7) * cos(ys[j]/9) for i in 1:Nx, j in 1:Ny]
        vv = [cos(xs[i]/5) * sin(ys[j]/11) for i in 1:Nx, j in 1:Ny]
        Πs = zeros(Nx, Ny); Πc = zeros(Nx, Ny)
        CGEF.Diagnostics.compute_Π!(Πs, uu, vv, nothing, sgrid, CGEF.TopHatKernel(), 8.0)
        CGEF.Diagnostics.compute_Π!(Πc, uu, vv, nothing, cgrid, CGEF.TopHatKernel(), 8.0)
        Test.@test maximum(abs.(Πc .- Πs)) < 1e-9 * maximum(abs.(Πs)) + 1e-12

        res = CGEF.coarse_grain(uu, vv, cgrid; scales=[8.0, 12.0], kernel=CGEF.TopHatKernel())
        Test.@test size(res.Π, 3) == 2
        Test.@test res.Π[:, :, 1] ≈ Πc
        Test.@test !any(isnan, res.cumulative_energy)
        Test.@test !any(isnan, res.filtering_spectrum)

        # --- Spherical curvilinear sanity: solid-body rotation u = U cos φ, v = 0 gives Π ≈ 0
        # (filtering is done in planetary-Cartesian coordinates, so rigid rotation commutes with the
        # filter). The residual is set by WLSQ/chord discretization, so the interior tolerance is
        # looser than the arc-length-exact StructuredGrid case. ---
        lon_s = deg2rad.([0.0, 4.0, 9.0, 15.0, 22.0, 30.0])   # moderate nonuniformity (no huge jumps)
        lat_s = deg2rad.([-20.0, -12.0, -5.0, 3.0, 12.0, 20.0])
        Nls, Nas = length(lon_s), length(lat_s)
        clon = [lon_s[i] for i in 1:Nls, j in 1:Nas]
        clat = [lat_s[j] for i in 1:Nls, j in 1:Nas]
        scurv = CGEF.CurvilinearGrid(sph, clon, clat, trues(Nls, Nas))
        U = 5.0
        urot = [U*cos(lat_s[j]) for i in 1:Nls, j in 1:Nas]
        vrot = zeros(Nls, Nas)
        Πsb = zeros(Nls, Nas)
        CGEF.Diagnostics.compute_Π!(Πsb, urot, vrot, nothing, scurv, CGEF.TopHatKernel(), 5e5)
        Test.@test all(isfinite, Πsb)
        for i in 2:(Nls-1), j in 2:(Nas-1)
            Test.@test abs(Πsb[i, j]) < 1e-3
        end
    end

    Test.@testset "UnstructuredGrid (k-d tree / Voronoi / WLSQ / pipeline)" begin
        # --- k-d tree neighbor query correctness on a small, hand-verifiable regular lattice: for an
        # interior point of a uniform grid, the k=4 nearest neighbors are exactly its N/S/E/W neighbors. ---
        cart = CGEF.CartesianGeometry(1.0, 1.0)
        gx = collect(0.0:1.0:4.0); gy = collect(0.0:1.0:4.0)
        plon = vec([x for x in gx, y in gy]); plat = vec([y for x in gx, y in gy])
        ugrid_lattice = CGEF.UnstructuredGrid(cart, plon, plat, trues(length(plon)); k = 4)
        center_idx = findfirst(i -> plon[i] == 2.0 && plat[i] == 2.0, eachindex(plon))
        nbr_coords = Set((plon[j], plat[j]) for j in CGEF.Grids.neighbors(ugrid_lattice, center_idx))
        Test.@test nbr_coords == Set([(1.0,2.0), (3.0,2.0), (2.0,1.0), (2.0,3.0)])

        # --- Voronoi-area sum invariant, Cartesian: a regular lattice's Voronoi cells reduce to the
        # exact grid-cell area (1.0 here), so interior nodes' areas must be ≈1 and the total must equal
        # the lattice's bounding area (up to boundary-clipping conventions at the edge). ---
        interior = [i for i in eachindex(plon) if 1.0 <= plon[i] <= 3.0 && 1.0 <= plat[i] <= 3.0]
        Test.@test all(i -> isapprox(ugrid_lattice.areas[i], 1.0; rtol = 1e-6), interior)

        # --- Voronoi-area sum invariant, spherical: a quasi-uniform point set covering the WHOLE
        # sphere must have Voronoi areas summing to EXACTLY 4πR² (a full closed tessellation, no
        # boundary to clip — an exact invariant, not merely quasi-uniform-only). ---
        sph_geo = CGEF.SphericalGeometry(6371000.0)
        Nsph = 400
        # Fibonacci sphere point set: deterministic, quasi-uniform coverage of the whole sphere.
        golden = (1 + sqrt(5)) / 2
        sidx = 0:(Nsph-1)
        sphi = acos.(1 .- 2 .* (sidx .+ 0.5) ./ Nsph) .- π/2   # latitude in [-π/2, π/2]
        stheta = 2π .* (sidx ./ golden .% 1)                   # longitude in [0, 2π)
        sgrid_full = CGEF.UnstructuredGrid(sph_geo, stheta, sphi, trues(Nsph); k = 8)
        Test.@test sum(sgrid_full.areas) ≈ 4π * sph_geo.R^2 rtol = 1e-10

        # --- WLSQ gradient is exact for a LINEAR field (algebraically guaranteed on any stencil,
        # same as the CurvilinearGrid case) on a genuinely irregular (random) point scatter. ---
        Random_N = 150
        rlon = rand(Random_N) .* 10000.0
        rlat = rand(Random_N) .* 10000.0
        rgrid = CGEF.UnstructuredGrid(cart, rlon, rlat, trues(Random_N); k = 8)
        flin = 2.0 .* rlon .+ 3.0 .* rlat
        gx_lin = zeros(Random_N); gy_lin = zeros(Random_N)
        CGEF.Derivatives.ddx!(gx_lin, flin, rgrid)
        CGEF.Derivatives.ddy!(gy_lin, flin, rgrid)
        # Only nodes with a full (non-rank-deficient) stencil are guaranteed exact; boundary/corner
        # nodes with a degenerate one-sided stencil are excluded — an honest test, not a silent one.
        interior_r = [i for i in 1:Random_N if length(CGEF.Grids.neighbors(rgrid, i)) >= 4]
        Test.@test maximum(abs.(gx_lin[interior_r] .- 2.0)) < 1e-8
        Test.@test maximum(abs.(gy_lin[interior_r] .- 3.0)) < 1e-8

        # --- Full compute_Π!/coarse_grain pipeline, cross-checked against the equivalent
        # StructuredGrid result on the SAME underlying lattice (points placed exactly on grid nodes,
        # k chosen so neighbors are exactly the 4 structured neighbors). ---
        Nx2, Ny2 = 8, 8
        xs2 = collect(0.0:1000.0:(1000.0*(Nx2-1)))
        ys2 = collect(0.0:1000.0:(1000.0*(Ny2-1)))
        sgrid2 = CGEF.StructuredGrid(cart, xs2, ys2, trues(Nx2, Ny2))
        ulon = vec([x for x in xs2, y in ys2]); ulat = vec([y for x in xs2, y in ys2])
        uu2 = vec([sin(x/700) * cos(y/900) for x in xs2, y in ys2])
        vv2 = vec([cos(x/500) * sin(y/1100) for x in xs2, y in ys2])
        ugrid2 = CGEF.UnstructuredGrid(cart, ulon, ulat, trues(length(ulon)); k = 4)

        Πu = zeros(length(ulon))
        CGEF.Diagnostics.compute_Π!(Πu, uu2, vv2, nothing, ugrid2, CGEF.GaussianKernel(), 3000.0)
        Test.@test all(isfinite, Πu)
        # Performance budget: catches a regression to the fixed FINUFFT mode-count-from-geometry.dx
        # bug (which made this 64-point call take ~120s/4 GiB instead of milliseconds/KB) even if some
        # future change kept the numerics finite-but-slow.
        t_pi = @elapsed CGEF.Diagnostics.compute_Π!(Πu, uu2, vv2, nothing, ugrid2, CGEF.GaussianKernel(), 3000.0)
        b_pi = @allocated CGEF.Diagnostics.compute_Π!(Πu, uu2, vv2, nothing, ugrid2, CGEF.GaussianKernel(), 3000.0)
        Test.@test t_pi < 1.0
        Test.@test b_pi < 10_000_000

        res_u = CGEF.coarse_grain(uu2, vv2, ugrid2; scales = [2000.0, 3000.0], kernel = CGEF.GaussianKernel())
        Test.@test size(res_u.Π) == (length(ulon), 2)
        Test.@test !any(isnan, res_u.cumulative_energy)
        Test.@test !any(isnan, res_u.filtering_spectrum)
        t_cg = @elapsed CGEF.coarse_grain(uu2, vv2, ugrid2; scales = [2000.0, 3000.0], kernel = CGEF.GaussianKernel())
        Test.@test t_cg < 2.0

        # --- Solid-body rotation on a spherical UnstructuredGrid gives Π ≈ 0 (same physical invariant
        # as the StructuredGrid/CurvilinearGrid cases). ---
        Usb = 5.0
        usb = Usb .* cos.(sphi)
        vsb = zeros(Nsph)
        Πsb_u = zeros(Nsph)
        CGEF.Diagnostics.compute_Π!(Πsb_u, usb, vsb, nothing, sgrid_full, CGEF.GaussianKernel(), 5e5)
        Test.@test all(isfinite, Πsb_u)
        Test.@test maximum(abs, Πsb_u) < 1e-2 * Usb^2
    end

    Test.@testset "1D and singleton-dimension StructuredGrid" begin
        # --- Genuinely 1D Cartesian StructuredGrid: ddx! exact for a linear field, compute_Π!/
        # coarse_grain finite. ---
        cgeom = CGEF.CartesianGeometry(1000.0, 1000.0)
        x1 = collect(0.0:1000.0:10000.0)
        grid1d = CGEF.StructuredGrid(cgeom, x1, trues(length(x1)))
        flin = 3.0 .* x1
        dflin = zeros(length(x1))
        CGEF.Derivatives.ddx!(dflin, flin, grid1d)
        Test.@test all(x -> isapprox(x, 3.0; atol = 1e-8), dflin[2:end-1])

        u1 = rand(length(x1))
        Π1 = zeros(length(x1))
        CGEF.Diagnostics.compute_Π!(Π1, u1, grid1d, CGEF.TopHatKernel(), 3000.0)
        Test.@test all(isfinite, Π1)

        res1 = CGEF.coarse_grain(u1, grid1d; scales = [2000.0, 3000.0], kernel = CGEF.TopHatKernel())
        Test.@test size(res1.Π) == (length(x1), 2)
        Test.@test !any(isnan, res1.cumulative_energy)

        # --- Singleton-dimension StructuredGrid measure: a Cartesian grid with one axis of length 1
        # degenerates from area to the surviving axis's plain width (not zero). ---
        cgrid_singleton = CGEF.StructuredGrid(cgeom, x1, [500.0], trues(length(x1), 1))
        Test.@test all(i -> CGEF.Grids.area(cgrid_singleton, i, 1) ≈ 1000.0, 2:(length(x1)-1))

        # --- Spherical singleton-latitude (zonal transect): the measure is the EXACT arc length
        # along that circle of latitude (R cosφ Δλ), not zero and not `area_element` with a
        # placeholder Δφ (which would leave a spurious extra factor of R). ---
        sgeom2 = CGEF.SphericalGeometry(6371000.0)
        λc = deg2rad.(collect(0.0:5.0:355.0))
        φ0 = deg2rad(10.0)
        sgrid_zonal = CGEF.StructuredGrid(sgeom2, λc, [φ0], trues(length(λc), 1); periodic = (true, false))
        exact_arclen = sgeom2.R * cos(φ0) * deg2rad(5.0)
        Test.@test all(i -> isapprox(CGEF.Grids.area(sgrid_zonal, i, 1), exact_arclen; rtol = 1e-12), eachindex(λc))

        # Before the fix this silently produced NaN (0/0 from a zero total area) — now finite.
        uz = rand(length(λc), 1); vz = rand(length(λc), 1)
        res_zonal = CGEF.coarse_grain(uz, vz, sgrid_zonal; scales = [5e5, 1e6], kernel = CGEF.TopHatKernel())
        Test.@test !any(isnan, res_zonal.cumulative_energy)
        Test.@test !any(isnan, res_zonal.filtering_spectrum)

        # --- Spherical singleton-longitude (meridional transect): arc length R Δφ, no cosφ factor. ---
        φc = deg2rad.(collect(-40.0:5.0:40.0))
        sgrid_merid = CGEF.StructuredGrid(sgeom2, [0.0], φc, trues(1, length(φc)); periodic = (false, false))
        exact_merid = sgeom2.R * deg2rad(5.0)
        Test.@test all(j -> isapprox(CGEF.Grids.area(sgrid_merid, 1, j), exact_merid; rtol = 1e-12), eachindex(φc))
    end

    # --- True-3D Cartesian pipeline: coarse_grain reuses the existing genuinely-coupled compute_Π!
    # (all nine strain components), just wired through the workspace-reusing scale sweep. ---
    Test.@testset "True-3D Cartesian coarse_grain pipeline" begin
        geom3 = CGEF.CartesianGeometry(1000.0, 1000.0, 500.0)
        grid3 = CGEF.StructuredGrid(
            geom3, collect(0.0:1000.0:5000.0), collect(0.0:1000.0:5000.0), collect(0.0:500.0:2000.0),
            trues(6, 6, 5),
        )
        u3 = rand(6, 6, 5); v3 = rand(6, 6, 5); w3 = rand(6, 6, 5)
        res3 = CGEF.coarse_grain(u3, v3, w3, grid3; scales = [2000.0, 3000.0], kernel = CGEF.TopHatKernel())
        Test.@test size(res3.Π) == (6, 6, 5, 2)
        Test.@test !any(isnan, res3.cumulative_energy)
        Test.@test !any(isnan, res3.filtering_spectrum)

        # Cross-check: coarse_grain! (in-place, reusing a workspace) matches the fresh allocation.
        ws3 = CGEF.Diagnostics.ΠWorkspace(grid3)
        res3b = CGEF.coarse_grain(u3, v3, w3, grid3; scales = [2000.0, 3000.0], kernel = CGEF.TopHatKernel())
        CGEF.coarse_grain!(res3b, u3, v3, w3, grid3; scales = [2000.0, 3000.0], kernel = CGEF.TopHatKernel(), workspace = ws3)
        Test.@test res3b.Π ≈ res3.Π
    end

    # True-3D SPHERICAL volumetric support: a genuine radius axis (r = absolute distance from planet
    # center, not a depth/height offset), real ∂/∂r derivatives, and the full 3x3 planetary-Cartesian
    # tensor rotation (as opposed to the 2.5D layer-by-layer path, which has no radial axis at all).
    Test.@testset "True-3D spherical volumetric grid + Π" begin
        R = 6.371e6
        geo = CGEF.SphericalGeometry(R)
        lon = deg2rad.(collect(0.0:10.0:350.0))
        lat = deg2rad.(collect(-60.0:10.0:60.0))
        r = collect(R:5000.0:(R + 20000.0))  # 5 levels, 20 km shell
        mask = trues(length(lon), length(lat), length(r))
        grid = CGEF.StructuredGrid(geo, lon, lat, r, mask)
        Test.@test CGEF.Grids.size_tuple(grid) == (length(lon), length(lat), length(r))

        # Volume element is the genuine spherical-shell r²cosφΔλΔφΔr at each level's OWN local
        # radius, not the fixed reference R — so cell volume must grow with height at fixed (i,j).
        Test.@test CGEF.Grids.area(grid, 1, 7, 5) > CGEF.Grids.area(grid, 1, 7, 1)
        # Exact ratio check at the equator-ish band (φ index 7 is closest to 0): volumes at two
        # levels and the same (i,j) scale as (r[k]/r[k'])² (cosφ, Δλ, Δφ, Δr all cancel exactly
        # away from the domain's radial boundary, where Δr is uniform anyway on this axis).
        Test.@test CGEF.Grids.area(grid, 1, 7, 3) / CGEF.Grids.area(grid, 1, 7, 2) ≈ (r[3] / r[2])^2 rtol=1e-12

        # A single radius level is the 2D/2.5D case, not true-3D — must be rejected, not silently
        # given a wrong (area-not-volume) measure.
        Test.@test_throws ArgumentError CGEF.StructuredGrid(geo, lon, lat, [R], trues(length(lon), length(lat), 1))

        # Rigid-body rotation of the whole 3D shell (u_e = Ω·r·cosφ, v_n = w_r = 0) is a pure
        # rotation — zero strain rate everywhere, hence Π ≡ 0 — regardless of the genuine radial
        # shear ∂u_e/∂r = Ω·cosφ ≠ 0 this induces (unlike the 2D invariant, which has no radial
        # shear to get wrong in the first place; this is the check that actually exercises the new
        # S_er curvature-correction term).
        Ω = 7.292e-5
        u = [Ω * r[k] * cos(lat[j]) for _ in lon, j in eachindex(lat), k in eachindex(r)]
        v = zeros(length(lon), length(lat), length(r))
        w = zeros(length(lon), length(lat), length(r))
        Π = zeros(size(u))
        CGEF.Diagnostics.compute_Π!(Π, u, v, w, grid, CGEF.TopHatKernel(), 500e3)
        Test.@test maximum(abs, Π) < 1e-9 * maximum(abs, u)

        # Full pipeline: shape + finiteness.
        res = CGEF.coarse_grain(u, v, w, grid; scales = [300e3, 500e3], kernel = CGEF.TopHatKernel())
        Test.@test size(res.Π) == (length(lon), length(lat), length(r), 2)
        Test.@test !any(isnan, res.cumulative_energy)
        Test.@test !any(isnan, res.filtering_spectrum)
    end

    # True-3D Helmholtz flux decomposition and 3D tracer-variance flux.
    Test.@testset "True-3D Helmholtz flux decomposition & tracer flux" begin
        geom3 = CGEF.CartesianGeometry(1000.0, 1000.0, 500.0)
        x = collect(0.0:1000.0:10e3); y = collect(0.0:1000.0:10e3); z = collect(0.0:500.0:4e3)
        grid3 = CGEF.StructuredGrid(geom3, x, y, z, trues(length(x), length(y), length(z)))
        kern = CGEF.TopHatKernel(); scale = 2500.0

        Lx = x[end] - x[1]; Ly = y[end] - y[1]; Lz = z[end] - z[1]
        kx = 2π * 3 / Lx; ky = 2π * 2 / Ly; kz = 2π * 2 / Lz

        # Non-divergent field via vector potential A=(0,0,ψ): u=∂ψ/∂y, v=-∂ψ/∂x, w=0 is exactly
        # non-divergent for ANY ψ(x,y) (2D streamfunction embedded in 3D, no z-dependence).
        u_rot = [ky * cos(kx * xi) * sin(ky * yj) for xi in x, yj in y, _ in z]
        v_rot = [-kx * sin(kx * xi) * cos(ky * yj) for xi in x, yj in y, _ in z]
        w_rot = zeros(length(x), length(y), length(z))

        # Irrotational field as the gradient of a scalar potential χ: curl(∇χ) ≡ 0 by construction.
        u_div = [kx * cos(kx * xi) * sin(ky * yj) * sin(kz * zk) for xi in x, yj in y, zk in z]
        v_div = [ky * sin(kx * xi) * cos(ky * yj) * sin(kz * zk) for xi in x, yj in y, zk in z]
        w_div = [kz * sin(kx * xi) * sin(ky * yj) * cos(kz * zk) for xi in x, yj in y, zk in z]

        u = u_rot .+ u_div; v = v_rot .+ v_div; w = w_rot .+ w_div

        dec = CGEF.Diagnostics.compute_Π_decomposed(u, v, w, u_rot, v_rot, w_rot, grid3, kern, scale)
        Πfull = zeros(size(u)); CGEF.Diagnostics.compute_Π!(Πfull, u, v, w, grid3, kern, scale)
        Test.@test dec.total ≈ dec.rotational .+ dec.cross .+ dec.divergent
        Test.@test dec.total ≈ Πfull

        Πr_full = zeros(size(u_rot)); CGEF.Diagnostics.compute_Π!(Πr_full, u_rot, v_rot, w_rot, grid3, kern, scale)
        dec_r = CGEF.Diagnostics.compute_Π_decomposed(u_rot, v_rot, w_rot, u_rot, v_rot, w_rot, grid3, kern, scale)
        Test.@test maximum(abs, dec_r.divergent) < 1e-10
        Test.@test maximum(abs, dec_r.cross) < 1e-10
        Test.@test dec_r.rotational ≈ Πr_full

        # 3D tracer-variance flux: constant tracer ⇒ zero gradient ⇒ zero flux.
        θ = rand(length(x), length(y), length(z))
        Πθ = CGEF.Diagnostics.tracer_variance_flux(u, v, w, θ, grid3, kern, scale)
        Test.@test all(isfinite, Πθ)
        Πθ0 = CGEF.Diagnostics.tracer_variance_flux(u, v, w, fill(2.5, size(θ)), grid3, kern, scale)
        Test.@test maximum(abs, Πθ0) < 1e-9
    end

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
        CGEF.Diagnostics.compute_Π!(Π, u, v, nothing, grid, CGEF.TopHatKernel(), 10000.0)

        # Kinetic energy transfer must be zero (rigid body rotation is pure laminar cascade-free flow)
        Test.@test Π[11, 11] ≈ 0.0 atol=1e-12

        # Test Pipeline integration with unicode Π
        res = CGEF.coarse_grain(u, v, grid; scales=[10000.0], kernel=CGEF.TopHatKernel())
        Test.@test res.Π[:, :, 1] ≈ Π

        # Test Spherical projections and coarse graining with mixed types
        sgeom = CGEF.SphericalGeometry(6371000.0)
        slon = collect(0.0:2.0:10.0)
        slat = collect(0.0:2.0:10.0)
        smask = trues(length(slon), length(slat))
        sgrid = CGEF.StructuredGrid(sgeom, deg2rad.(slon), deg2rad.(slat), smask)

        # Test to_planetary_cartesian and from_planetary_cartesian mixed type support
        proj = CGEF.Geometry.to_planetary_cartesian(sgeom, Float32(1.0), Float32(2.0), 0.1, 0.2, 0.3)
        Test.@test proj isa SA.SVector{3, Float64}

        inv_proj = CGEF.Geometry.from_planetary_cartesian(sgeom, Float32(1.0), 2.0, 3.0, 0.1, 0.2)
        Test.@test inv_proj isa SA.SVector{3, Float64}

        # Test coarse_grain on sphere with Float32 inputs (matching PythonCall runtime environment)
        su = fill(Float32(1.0), length(slon), length(slat))
        sv = fill(Float32(0.5), length(slon), length(slat))
        sres = CGEF.coarse_grain(su, sv, sgrid; scales=[50000.0], kernel=CGEF.TopHatKernel())
        Test.@test !any(isnan, @view sres.Π[:, :, 1])
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
        scale = deg2rad(20.0) * 6371000.0
        out = zeros(length(lon_deg), length(lat_deg))
        CGEF.Filtering.filter_field!(out, field, grid, CGEF.TopHatKernel(), scale)

        # Brute-force reference: an ALL-PAIRS TopHat average built directly from the raw
        # great-circle-distance primitive (NOT the package's footprint-building/seam-wrap index
        # logic under test here). This is naturally, automatically periodic in longitude — the
        # Haversine `sin²(Δλ/2)` term is exactly invariant to whether Δλ is measured "the short way"
        # or "the long way around" the 360° circle, so no explicit modular wrap is needed to get the
        # true periodic answer. This replaces a hand-derived comment that didn't actually match the
        # asserted bounds with a genuinely computed expected value.
        rad = CGEF.Kernels.kernel_radius(CGEF.TopHatKernel(), scale)  # TopHat radius is ℓ/2, not ℓ
        # AREA-weighted (not plain-count) average: grid cells at different latitudes have different
        # physical area on a sphere (∝ cosφ), and a spatial average — which is what a TopHat filter
        # computes (weight = kernel_weight·area, kernel_weight≡1 within radius) — must weight by that
        # area, not just count included points. An unweighted average is a different, wrong quantity
        # whenever the neighbourhood spans more than one latitude row (it always does here).
        function brute_force_avg(i0, j0)
            target = SA.SVector{2,Float64}(lon_rad[i0], lat_rad[j0])
            acc = 0.0; wsum = 0.0
            for j in eachindex(lat_deg), i in eachindex(lon_deg)
                d = CGEF.Geometry.distance(geom, target, SA.SVector{2,Float64}(lon_rad[i], lat_rad[j]))
                if d <= rad
                    a = CGEF.Grids.area(grid, i, j)
                    acc += field[i, j] * a
                    wsum += a
                end
            end
            return acc / wsum
        end

        j10 = 10  # a mid-latitude row, away from the poles
        Test.@test out[1, j10] ≈ brute_force_avg(1, j10) rtol=1e-10
        Test.@test out[end, j10] ≈ brute_force_avg(length(lon_deg), j10) rtol=1e-10

        # The seam genuinely wraps: on a NON-periodic version of the same grid, 355° loses the
        # wrapped 0-10° "hot" band entirely, so its average must be strictly LESS than the correctly-
        # wrapped periodic result — removing area-weighted contributions from the hot region can only
        # decrease (never increase) a weighted average. This is the exact, derivable bound (a specific
        # numeric factor like "less than half" isn't independently justified without computing it, and
        # a hand-picked one is exactly the kind of loosened-to-pass tolerance Tier 3.15 replaced
        # elsewhere in this file).
        nonperiodic_grid = CGEF.StructuredGrid(geom, lon_rad, lat_rad, mask; periodic = false)
        out_np = zeros(size(field))
        CGEF.Filtering.filter_field!(out_np, field, nonperiodic_grid, CGEF.TopHatKernel(), scale)
        Test.@test out_np[end, j10] < out[end, j10]

        # A point at 30° (well outside the ±20° band around the seam) sees no wrapping benefit —
        # its own brute-force reference already reflects that, at the same tight tolerance.
        i30 = findfirst(==(30.0), lon_deg)
        Test.@test out[i30, j10] ≈ brute_force_avg(i30, j10) rtol=1e-10
        Test.@test out[i30, j10] < out[1, j10]
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
        Test.@test CGEF.Grids.isperiodic(grid_reg, 1) == false

        # Full-circle lon span -> auto-detected periodic
        lon_glob = deg2rad.(collect(0.0:5.0:355.0))
        mask_glob = trues(length(lon_glob), length(lat))
        grid_glob = CGEF.StructuredGrid(geom, lon_glob, lat, mask_glob)
        Test.@test CGEF.Grids.isperiodic(grid_glob, 1) == true

        # Explicit override in both directions
        Test.@test CGEF.Grids.isperiodic(CGEF.StructuredGrid(geom, lon_reg, lat, mask_reg; periodic = true), 1) == true
        Test.@test CGEF.Grids.isperiodic(CGEF.StructuredGrid(geom, lon_glob, lat, mask_glob; periodic = false), 1) == false

        # The periodicity flag must actually change filtering: with a footprint wider than the
        # regional domain, wrapping double-counts and yields a different (incorrect) field.
        grid_forced = CGEF.StructuredGrid(geom, lon_reg, lat, mask_reg; periodic = true)
        field = Float64[i for i in 1:length(lon_reg), _ in 1:length(lat)]  # ramp in lon
        scale = deg2rad(30.0) * 6371000.0   # footprint wider than the 20° domain
        out_nowrap = zeros(size(field))
        out_wrap = zeros(size(field))
        CGEF.Filtering.filter_field!(out_nowrap, field, grid_reg, CGEF.TopHatKernel(), scale)
        CGEF.Filtering.filter_field!(out_wrap, field, grid_forced, CGEF.TopHatKernel(), scale)
        Test.@test !any(isnan, out_nowrap)
        Test.@test !(out_nowrap ≈ out_wrap)
    end

    # Regression test: a periodic CARTESIAN axis was computing its wrapped boundary cell width using
    # the SPHERICAL periodicity constant (2π radians) unconditionally, regardless of geometry — for a
    # Cartesian domain measured in meters, this produced a wildly wrong (even negative) boundary cell
    # width, and hence a nonsensical filter weight there. This is a physically impossible result for a
    # correctly-normalized low-pass filter (output can amplify beyond the input's range), not just an
    # accuracy issue — caught via a direct comparison against the analytic filtered value of a single
    # Fourier mode (an exact eigenfunction relation, Ĝ(k)·field, independent of discretization).
    Test.@testset "Periodic Cartesian grid: no boundary weight corruption" begin
        dx = 62.5
        Nx = 320
        geom = CGEF.CartesianGeometry(dx, dx)
        xs = collect(0.0:dx:(dx*(Nx-1)))
        Lx = dx * Nx
        grid = CGEF.StructuredGrid(geom, xs, xs, trues(Nx, Nx); periodic = true)
        Test.@test CGEF.Grids.isperiodic(grid, 1) == true

        kx0, ky0 = 2, 3
        field = [sin(2π * kx0 * x / Lx) * cos(2π * ky0 * y / Lx) for x in xs, y in xs]
        scale = 1500.0
        kx = 2π * kx0 / Lx; ky = 2π * ky0 / Lx
        Ghat = CGEF.Kernels.spectral_transfer(CGEF.GaussianKernel(), sqrt(kx^2 + ky^2), scale)
        analytic = Ghat .* field

        out = zeros(Nx, Nx)
        CGEF.Filtering.filter_field!(out, field, grid, CGEF.GaussianKernel(), scale)

        # A low-pass filter is a normalized weighted average: it can never amplify beyond the input's
        # range. This is the invariant the bug violated (output reached ~4.8x the input's peak).
        Test.@test maximum(abs, out) <= maximum(abs, field) + 1e-9
        # Constant field ⇒ Ĝ(0)=1 ⇒ preserved exactly, independent of the boundary-wrap bug.
        out_const = zeros(Nx, Nx)
        CGEF.Filtering.filter_field!(out_const, fill(3.7, Nx, Nx), grid, CGEF.GaussianKernel(), scale)
        Test.@test all(x -> isapprox(x, 3.7; atol = 1e-6), out_const)
        # Away from the (genuinely non-periodic) y-boundary, the periodic-in-x wrap must reproduce
        # the analytic single-mode eigenfunction relation closely.
        rad = CGEF.Kernels.kernel_radius(CGEF.GaussianKernel(), scale)
        interior = [(rad < xs[j] < xs[end] - rad) for _ in 1:Nx, j in 1:Nx]
        reldiff = abs.(out[interior] .- analytic[interior]) ./ maximum(abs, analytic)
        Test.@test maximum(reldiff) < 0.3
        Test.@test sum(reldiff) / length(reldiff) < 0.05
    end

    Test.@testset "Periodic Cartesian grid: general/scattered footprint path matches the fast Range path exactly" begin
        # A genuinely uniform axis passed as a plain Vector (not a Range) forces the general/
        # scattered footprint builder (StructuredGrid's type-is-the-dispatch convention — no runtime
        # uniformity check). That path must honor periodicity exactly like the fast Range-axis path:
        # a wrapped neighbor's distance must be computed from its coordinate SHIFTED by one period,
        # not its raw stored coordinate (which sits a full domain-width away) — otherwise every
        # periodic wrap is silently rejected by the `d <= rad` gate and boundary cells behave as if
        # non-periodic, with no error. The existing "no boundary weight corruption" test above does
        # not catch this on its own — its eigenmode is too smooth and its tolerance too loose to
        # distinguish a correctly-wrapped boundary from a silently truncated one — so this test
        # cross-checks the general/scattered path directly against the independently-trusted fast
        # path instead.
        dx = 1_000.0
        N = 40
        geom = CGEF.CartesianGeometry(dx, dx)
        xsR = 0.0:dx:(dx * (N - 1))       # Range -> fast uniform path
        xsV = collect(xsR)                # identical values as a plain Vector -> scattered/general path
        field = Float64[i + 2j for i in 1:N, j in 1:N]   # boundary-discriminating, deliberately asymmetric

        for scale in (2_500.0, 6_000.0)   # spans 1 and >2 wrapped-neighbor bands
            gridR = CGEF.StructuredGrid(geom, xsR, xsR, trues(N, N); periodic = (true, true))
            gridV = CGEF.StructuredGrid(geom, xsV, xsV, trues(N, N); periodic = (true, true))
            Test.@test CGEF.Filtering.build_footprint(gridR, CGEF.TopHatKernel(), scale) isa CGEF.Filtering.FilterFootprint
            Test.@test CGEF.Filtering.build_footprint(gridV, CGEF.TopHatKernel(), scale) isa CGEF.Filtering.FilterFootprintScattered

            outR = zeros(N, N); outV = zeros(N, N)
            CGEF.Filtering.filter_field!(outR, field, gridR, CGEF.TopHatKernel(), scale)
            CGEF.Filtering.filter_field!(outV, field, gridV, CGEF.TopHatKernel(), scale)
            Test.@test isapprox(outR, outV; atol = 1e-9)

            # Sanity: the match above isn't trivially true regardless of periodicity — a genuinely
            # non-periodic grid must disagree with the periodic one at the boundary.
            gridVn = CGEF.StructuredGrid(geom, xsV, xsV, trues(N, N); periodic = (false, false))
            outVn = zeros(N, N)
            CGEF.Filtering.filter_field!(outVn, field, gridVn, CGEF.TopHatKernel(), scale)
            Test.@test maximum(abs, outV[1, :] .- outVn[1, :]) > 1e-6
        end

        # Same check for the N-D scattered path (`_build_footprint_nd_scattered`): true-3D and 1D.
        geom3 = CGEF.CartesianGeometry(dx, dx, dx)
        N3 = 12
        xs3R = 0.0:dx:(dx * (N3 - 1)); xs3V = collect(xs3R)
        field3 = Float64[i + 2j + 3k for i in 1:N3, j in 1:N3, k in 1:N3]
        grid3R = CGEF.StructuredGrid(geom3, xs3R, xs3R, xs3R, trues(N3, N3, N3); periodic = (true, true, true))
        grid3V = CGEF.StructuredGrid(geom3, xs3V, xs3V, xs3V, trues(N3, N3, N3); periodic = (true, true, true))
        out3R = zeros(N3, N3, N3); out3V = zeros(N3, N3, N3)
        CGEF.Filtering.filter_field!(out3R, field3, grid3R, CGEF.TopHatKernel(), 1_200.0)
        CGEF.Filtering.filter_field!(out3V, field3, grid3V, CGEF.TopHatKernel(), 1_200.0)
        Test.@test isapprox(out3R, out3V; atol = 1e-9)

        N1 = 30
        xs1R = 0.0:dx:(dx * (N1 - 1)); xs1V = collect(xs1R)
        field1 = Float64.(1:N1)
        grid1R = CGEF.StructuredGrid(geom, xs1R, trues(N1); periodic = true)
        grid1V = CGEF.StructuredGrid(geom, xs1V, trues(N1); periodic = true)
        out1R = zero(field1); out1V = zero(field1)
        CGEF.Filtering.filter_field!(out1R, field1, grid1R, CGEF.TopHatKernel(), 1_200.0)
        CGEF.Filtering.filter_field!(out1V, field1, grid1V, CGEF.TopHatKernel(), 1_200.0)
        Test.@test isapprox(out1R, out1V; atol = 1e-9)
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
        CGEF.Filtering.filter_field!(out_zero, field, grid, CGEF.TopHatKernel(), 100000.0; mask_strategy=CGEF.Filtering.ZeroFill())
        CGEF.Filtering.filter_field!(out_renorm, field, grid, CGEF.TopHatKernel(), 100000.0; mask_strategy=CGEF.Filtering.Deformable())

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
        d = CGEF.Geometry.distance(geom, p1, p2)

        # Should be approximately quarter circumference
        quarter_circumference = π * geom.R / 2
        Test.@test d ≈ quarter_circumference rtol=1e-6

        # Test: distance along equator for 1 degree
        p3 = SA.SVector{2,Float64}(0.0, 0.0)
        p4 = SA.SVector{2,Float64}(deg2rad(1.0), 0.0)
        d_equator = CGEF.Geometry.distance(geom, p3, p4)

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

        CGEF.Derivatives.ddx!(dudx, u, grid)
        CGEF.Derivatives.ddy!(dudy, u, grid)
        CGEF.Derivatives.ddx!(dvdx, v, grid)
        CGEF.Derivatives.ddy!(dvdy, v, grid)

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
            CGEF.Diagnostics.compute_Π!(Π, u, v, nothing, grid, CGEF.TopHatKernel(), scale)

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
        CGEF.Filtering.filter_field!(u_filt, u, grid, CGEF.TopHatKernel(), 10000.0)
        CGEF.Filtering.filter_field!(v_filt, v, grid, CGEF.TopHatKernel(), 10000.0)

        # Compute strain rates
        S_xx = zeros(length(lon), length(lat))
        S_yy = zeros(length(lon), length(lat))
        S_xy = zeros(length(lon), length(lat))
        scratch = zeros(length(lon), length(lat))

        CGEF.Derivatives.ddx!(S_xx, u_filt, grid)
        CGEF.Derivatives.ddy!(S_yy, v_filt, grid)
        CGEF.Derivatives.ddy!(S_xy, u_filt, grid)
        CGEF.Derivatives.ddx!(scratch, v_filt, grid)
        @. S_xy = 0.5 * (S_xy + scratch)

        # Test symmetry: S_xy should equal S_yx (we only computed S_xy)
        # Test trace = divergence for incompressible flow
        #
        # This should be essentially EXACT, not just "approximately 0.02": a TopHat filter is a
        # symmetric (uniform-disk) average, which reproduces a linear field exactly at any interior
        # point with a full, untruncated footprint (odd-order terms cancel by symmetry) — and the
        # centered finite difference (`nonuniform_first_derivative`) has zero truncation error for a
        # linear function too (its leading error term involves the third derivative, which is zero
        # here). `i,j in 20:end-20` keeps a comfortable margin inside the domain relative to the
        # 5000 m footprint radius (10000/2) on a 1000 m grid (5 cells), so every point checked has a
        # full, untruncated footprint. A 50%-of-expected-value tolerance would hide any real bug in
        # either the filter or the derivative; float64 roundoff alone is ~1e-12 relative here.
        for j in 20:length(lat)-20, i in 20:length(lon)-20
            divergence = S_xx[i,j] + S_yy[i,j]
            Test.@test divergence ≈ 0.02 atol=1e-9
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
        CGEF.Filtering.filter_field!(u_filt, u, grid, CGEF.TopHatKernel(), 5000.0)
        CGEF.Filtering.filter_field!(v_filt, v, grid, CGEF.TopHatKernel(), 5000.0)

        # Filter products
        uu = zeros(length(lon), length(lat))
        uv = zeros(length(lon), length(lat))
        vv = zeros(length(lon), length(lat))
        scratch = zeros(length(lon), length(lat))

        @. scratch = u * u
        CGEF.Filtering.filter_field!(uu, scratch, grid, CGEF.TopHatKernel(), 5000.0)
        @. scratch = u * v
        CGEF.Filtering.filter_field!(uv, scratch, grid, CGEF.TopHatKernel(), 5000.0)
        @. scratch = v * v
        CGEF.Filtering.filter_field!(vv, scratch, grid, CGEF.TopHatKernel(), 5000.0)

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
        CGEF.Filtering.filter_field!(scratch, scratch2, grid, CGEF.TopHatKernel(), 5000.0)
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
        CGEF.Diagnostics.compute_Π!(Π, u, v, nothing, grid, CGEF.TopHatKernel(), 10000.0)

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
        CGEF.Filtering.filter_field!(u_filt, u, grid, CGEF.TopHatKernel(), scale)
        CGEF.Filtering.filter_field!(v_filt, v, grid, CGEF.TopHatKernel(), scale)

        # Filter products for SFS stress trace
        uu_filt = zeros(length(lon), length(lat))
        vv_filt = zeros(length(lon), length(lat))
        scratch = zeros(length(lon), length(lat))

        @. scratch = u * u
        CGEF.Filtering.filter_field!(uu_filt, scratch, grid, CGEF.TopHatKernel(), scale)
        @. scratch = v * v
        CGEF.Filtering.filter_field!(vv_filt, scratch, grid, CGEF.TopHatKernel(), scale)

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
        CGEF.Filtering.filter_field!(filtered_total, scratch, grid, CGEF.TopHatKernel(), scale)
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

                CGEF.Filtering.filter_field!(out_zero, field, grid, kernel, scale; mask_strategy=CGEF.Filtering.ZeroFill())
                CGEF.Filtering.filter_field!(out_renorm, field, grid, kernel, scale; mask_strategy=CGEF.Filtering.Deformable())

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
        # cumulative_energy/filtering_spectrum return the DENSITY-FREE specific energy (0.5|ū|²,
        # not 0.5*ρ₀*|ū|²) — ρ₀ is a pure trailing multiplicative scale factor with no bearing on the
        # tensor structure, so it's the caller's job to multiply by their own reference density
        # afterward if they want an absolute volumetric quantity, not this package's to assume one.
        expected_energy = 0.5 * (U^2 + V^2)
        scales = [5000.0, 10000.0, 20000.0, 40000.0]

        # A uniform field is unchanged by filtering, so the CUMULATIVE coarse KE equals the kinetic
        # energy at every scale (Eq. 15).
        cumE = CGEF.Diagnostics.cumulative_energy(u, v, nothing, grid, CGEF.TopHatKernel(), scales)
        for E in cumE
            Test.@test E ≈ expected_energy rtol=1e-6
        end

        # Since the cumulative energy is constant in ℓ, the filtering spectral DENSITY (its
        # k_ℓ-derivative, Eq. 14) must be ≈ 0 everywhere — NOT equal to the energy.
        kℓ, Ẽ = CGEF.Diagnostics.filtering_spectrum(u, v, nothing, grid, CGEF.TopHatKernel(), scales; L=1.0)
        Test.@test length(kℓ) == length(scales)
        Test.@test all(abs.(Ẽ) .< 1e-6 * expected_energy)

        # spectral_density reproduces a known derivative: C(k)=k² ⇒ dC/dk = 2k (central differences
        # are exact for a quadratic on a uniform grid).
        kk = collect(1.0:1.0:5.0)
        Test.@test CGEF.Diagnostics.spectral_density(kk .^ 2, kk)[3] ≈ 2 * kk[3]
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

        d = CGEF.Diagnostics.tau_decomposition(u, v, grid, kern, scale)

        # Reference τ_ij = filter(u_i u_j) - ū_i ū_j with the same filter.
        ub = zeros(size(u)); vb = zeros(size(v))
        CGEF.Filtering.filter_field!(ub, u, grid, kern, scale)
        CGEF.Filtering.filter_field!(vb, v, grid, kern, scale)
        uu = zeros(size(u)); uv = zeros(size(u)); vv = zeros(size(u))
        CGEF.Filtering.filter_field!(uu, u .* u, grid, kern, scale)
        CGEF.Filtering.filter_field!(uv, u .* v, grid, kern, scale)
        CGEF.Filtering.filter_field!(vv, v .* v, grid, kern, scale)
        τxx = uu .- ub .^ 2
        τxy = uv .- ub .* vb
        τyy = vv .- vb .^ 2

        Test.@test d.L.xx .+ d.C.xx .+ d.R.xx ≈ τxx
        Test.@test d.L.xy .+ d.C.xy .+ d.R.xy ≈ τxy
        Test.@test d.L.yy .+ d.C.yy .+ d.R.yy ≈ τyy
        # Reynolds (subfilter–subfilter) stress trace is non-negative (Jensen).
        Test.@test all(d.R.xx .+ d.R.yy .>= -1e-10)
    end

    # Rotational/divergent (Helmholtz) decomposition of the energy flux. Uses genuine
    # streamfunction/potential test fields (exactly non-divergent / exactly irrotational by
    # elementary vector calculus, for ANY kx, ky) rather than an arbitrary scalar split of one
    # field — that older test passed even under the (buggy) one-sided-strain implementation,
    # since it degenerated to S̄ᵈ ≡ 0 and never exercised the cross-strain terms.
    Test.@testset "Helmholtz flux decomposition" begin
        geom = CGEF.CartesianGeometry(1000.0, 1000.0)
        x = collect(0.0:1000.0:30e3); y = collect(0.0:1000.0:30e3)
        grid = CGEF.StructuredGrid(geom, x, y, trues(length(x), length(y)))
        kern = CGEF.TopHatKernel(); scale = 5000.0

        Lx = x[end] - x[1]; Ly = y[end] - y[1]
        kx = 2π / Lx; ky = 2π / Ly

        # Taylor-Green vortex from streamfunction ψ = cos(kx x) cos(ky y): u = -∂ψ/∂y, v = ∂ψ/∂x
        # is exactly non-divergent (∂u/∂x + ∂v/∂y ≡ 0) by construction.
        u_rot = [ky * cos(kx * xi) * sin(ky * yj) for xi in x, yj in y]
        v_rot = [-kx * sin(kx * xi) * cos(ky * yj) for xi in x, yj in y]

        # Sine-sine potential χ = sin(kx x) sin(ky y): u = ∂χ/∂x, v = ∂χ/∂y is exactly
        # irrotational (∂v/∂x - ∂u/∂y ≡ 0) by construction.
        u_div = [kx * cos(kx * xi) * sin(ky * yj) for xi in x, yj in y]
        v_div = [ky * sin(kx * xi) * cos(ky * yj) for xi in x, yj in y]

        u = u_rot .+ u_div
        v = v_rot .+ v_div

        dec = CGEF.Diagnostics.compute_Π_decomposed(u, v, u_rot, v_rot, grid, kern, scale)
        Πfull = zeros(size(u)); CGEF.Diagnostics.compute_Π!(Πfull, u, v, nothing, grid, kern, scale)

        # (1) channels sum EXACTLY to the total, matching the standard full-flux computation.
        Test.@test dec.total ≈ dec.rotational .+ dec.cross .+ dec.divergent
        Test.@test dec.total ≈ Πfull

        # (2) purely-rotational field ⇒ divergent/cross channels vanish, rotational = full flux
        # computed on that field alone.
        Πr_full = zeros(size(u_rot)); CGEF.Diagnostics.compute_Π!(Πr_full, u_rot, v_rot, nothing, grid, kern, scale)
        dec_r = CGEF.Diagnostics.compute_Π_decomposed(u_rot, v_rot, u_rot, v_rot, grid, kern, scale)
        Test.@test maximum(abs, dec_r.divergent) < 1e-10
        Test.@test maximum(abs, dec_r.cross) < 1e-10
        Test.@test dec_r.rotational ≈ Πr_full

        # (3) purely-divergent field ⇒ rotational/cross channels vanish, divergent = full flux
        # computed on that field alone.
        Πd_full = zeros(size(u_div)); CGEF.Diagnostics.compute_Π!(Πd_full, u_div, v_div, nothing, grid, kern, scale)
        dec_d = CGEF.Diagnostics.compute_Π_decomposed(u_div, v_div, zeros(size(u_div)), zeros(size(v_div)), grid, kern, scale)
        Test.@test maximum(abs, dec_d.rotational) < 1e-10
        Test.@test maximum(abs, dec_d.cross) < 1e-10
        Test.@test dec_d.divergent ≈ Πd_full

        # (4) regression test for the fixed one-sided-strain bug: the old implementation contracted
        # the split stress against the FULL (undecomposed) strain S̄ for every channel — only correct
        # when S̄ᵈ ≡ 0. Build that old formula from primitives; the corrected Π_RR must differ here,
        # since S̄ᵈ is genuinely nonzero for this mixed field.
        ub = zeros(size(u)); vb = zeros(size(v))
        CGEF.Filtering.filter_field!(ub, u, grid, kern, scale)
        CGEF.Filtering.filter_field!(vb, v, grid, kern, scale)
        Sxx_full = zeros(size(u)); CGEF.Derivatives.ddx!(Sxx_full, ub, grid)
        Syy_full = zeros(size(u)); CGEF.Derivatives.ddy!(Syy_full, vb, grid)
        p = zeros(size(u)); q = zeros(size(u))
        CGEF.Derivatives.ddy!(p, ub, grid); CGEF.Derivatives.ddx!(q, vb, grid)
        Sxy_full = 0.5 .* (p .+ q)

        urb = zeros(size(u_rot)); vrb = zeros(size(v_rot))
        CGEF.Filtering.filter_field!(urb, u_rot, grid, kern, scale)
        CGEF.Filtering.filter_field!(vrb, v_rot, grid, kern, scale)
        uu = zeros(size(u)); uv = zeros(size(u)); vv = zeros(size(u))
        CGEF.Filtering.filter_field!(uu, u_rot .* u_rot, grid, kern, scale)
        CGEF.Filtering.filter_field!(uv, u_rot .* v_rot, grid, kern, scale)
        CGEF.Filtering.filter_field!(vv, v_rot .* v_rot, grid, kern, scale)
        τRRxx = uu .- urb .^ 2; τRRxy = uv .- urb .* vrb; τRRyy = vv .- vrb .^ 2

        old_Πrr = -(Sxx_full .* τRRxx .+ 2 .* Sxy_full .* τRRxy .+ Syy_full .* τRRyy)
        Test.@test !isapprox(dec.rotational, old_Πrr)
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
        Πc = CGEF.Diagnostics.tracer_variance_flux(u, v, fill(2.5, size(θ)), grid, kern, scale)
        Test.@test maximum(abs, Πc) < 1e-9

        # (2) matches the explicit definition Πθ = -(τx ∂x θ̄ + τy ∂y θ̄) built from primitives.
        Πθ = CGEF.Diagnostics.tracer_variance_flux(u, v, θ, grid, kern, scale)
        ub = zeros(size(u)); vb = zeros(size(v)); θb = zeros(size(θ))
        CGEF.Filtering.filter_field!(ub, u, grid, kern, scale)
        CGEF.Filtering.filter_field!(vb, v, grid, kern, scale)
        CGEF.Filtering.filter_field!(θb, θ, grid, kern, scale)
        uθ = zeros(size(u)); vθ = zeros(size(u))
        CGEF.Filtering.filter_field!(uθ, u .* θ, grid, kern, scale)
        CGEF.Filtering.filter_field!(vθ, v .* θ, grid, kern, scale)
        τx = uθ .- ub .* θb; τy = vθ .- vb .* θb
        gx = zeros(size(θ)); gy = zeros(size(θ))
        CGEF.Derivatives.ddx!(gx, θb, grid); CGEF.Derivatives.ddy!(gy, θb, grid)
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


    # Allocation regression tests for every core hot-path method (real-space + spectral
    # filter_apply!, ddx!/ddy!/ddz!, compute_Π! per grid type, coarse_grain!/cumulative_energy!'s
    # repeated-sweep entry points, parallel backends) — see test_allocs.jl's own header for exactly
    # what's asserted as zero vs. bounded-and-why.
    include("test_allocs.jl")
end
