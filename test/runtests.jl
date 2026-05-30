using CoarseGrainingEnergyFluxes
using Test
using LinearAlgebra
using StaticArrays

@testset "CoarseGrainingEnergyFluxes.jl" begin
    # Coordinate system and distance tests
    @testset "Geometry" begin
        # 2D Cartesian
        geom_cart = CartesianGeometry(1000.0, 1000.0)
        p1 = SVector{2,Float64}(0.0, 0.0)
        p2 = SVector{2,Float64}(3000.0, 4000.0)
        @test distance(geom_cart, p1, p2) ≈ 5000.0
        @test area_element(geom_cart) ≈ 1000.0 * 1000.0
        
        # Spherical
        geom_sph = SphericalGeometry(6371000.0)
        # London (0.1278 W, 51.5074 N) to Paris (2.3522 E, 48.8566 N)
        # Coordinates in radians
        london = SVector{2,Float64}(deg2rad(-0.1278), deg2rad(51.5074))
        paris  = SVector{2,Float64}(deg2rad(2.3522), deg2rad(48.8566))
        d_km = distance(geom_sph, london, paris) / 1000.0
        @test 340.0 < d_km < 350.0 # Paris-London ≈ 344 km
        
        # Coordinate projection conversions: to and from planetary Cartesian
        u_east, u_north = 10.0, -5.0
        λ, φ = deg2rad(-122.0), deg2rad(38.0) # San Francisco coords
        p_vel = to_planetary_cartesian(geom_sph, u_east, u_north, λ, φ)
        @test length(p_vel) == 3
        
        l_vel = from_planetary_cartesian(geom_sph, p_vel[1], p_vel[2], p_vel[3], λ, φ)
        @test l_vel[1] ≈ u_east
        @test l_vel[2] ≈ u_north
        @test abs(l_vel[3]) < 1e-12
    end
    
    # Kernel shape evaluation and support range tests
    @testset "Kernels" begin
        th = TopHatKernel()
        g  = GaussianKernel()
        ss = SharpSpectralKernel()
        
        # Width 100 km
        ℓ = 100000.0
        @test kernel_weight(th, 10000.0, ℓ) == 1.0
        @test kernel_weight(th, 60000.0, ℓ) == 0.0
        
        @test kernel_radius(th, ℓ) == ℓ / 2
        @test kernel_radius(g, ℓ) == 3 * ℓ
        
        # Test generic evaluation limits
        @test kernel_weight(g, 0.0, ℓ) == 1.0
        @test kernel_weight(g, ℓ, ℓ) ≈ exp(-6.0)
    end
    
    # Grids constructor and area calculations
    @testset "Grids" begin
        geom = CartesianGeometry(2000.0, 2000.0)
        lon = collect(0.0:2000.0:20000.0) # 11 points
        lat = collect(0.0:2000.0:10000.0) # 6 points
        mask = trues(11, 6) # active water grid
        
        grid = StructuredGrid(geom, lon, lat, mask)
        @test size_tuple(grid) == (11, 6)
        @test area(grid, 2, 2) == 2000.0 * 2000.0
        @test coords(grid, 2, 3) == SVector{2,Float64}(2000.0, 4000.0)
    end
    
    # physical-space filtering algorithms
    @testset "Filtering" begin
        geom = CartesianGeometry(100.0, 100.0)
        lon = collect(0.0:100.0:1000.0) # 11 points
        lat = collect(0.0:100.0:1000.0) # 11 points
        mask = trues(11, 11)
        grid = StructuredGrid(geom, lon, lat, mask)
        
        # Constant field filtering must return exactly the same constant
        field = fill(42.0, 11, 11)
        out = zeros(11, 11)
        filter_field!(out, field, grid, TopHatKernel(), 300.0)
        
        # Wet cells must have the exact filtered value (42.0)
        @test out[5, 5] ≈ 42.0
    end
    
    # spatial finite differences and boundary stencil fallbacks
    @testset "Derivatives" begin
        geom = CartesianGeometry(2.0, 2.0)
        lon = collect(0.0:2.0:10.0) # 6 points
        lat = collect(0.0:2.0:10.0) # 6 points
        mask = trues(6, 6)
        grid = StructuredGrid(geom, lon, lat, mask)
        
        # Test horizontal derivatives of f(x) = 3x + 1
        # ∂f/∂x should be exactly 3.0 at all wet cells
        f = zeros(6, 6)
        for j in 1:6, i in 1:6
            f[i, j] = 3.0 * grid.lon[i] + 1.0
        end
        
        ∂f∂x = zeros(6, 6)
        ddx!(∂f∂x, f, grid)
        
        @test ∂f∂x[2, 3] ≈ 3.0
        @test ∂f∂x[1, 3] ≈ 3.0 # forward difference at boundary
        @test ∂f∂x[6, 3] ≈ 3.0 # backward difference at boundary
    end
    
    # Helmholtz decomposition (rotational vs divergent velocities)
    @testset "Helmholtz Decomposition" begin
        geom = CartesianGeometry(2000.0, 2000.0)
        lon = collect(0.0:2000.0:40000.0) # 21 points
        lat = collect(0.0:2000.0:40000.0) # 21 points
        mask = trues(21, 21)
        grid = StructuredGrid(geom, lon, lat, mask)
        
        # Create a purely divergent velocity field with zero boundary flux and net-zero divergence:
        # u = sin(π * x/L) * cos(π * y/L)
        # v = cos(π * x/L) * sin(π * y/L)
        # matching Neumann solvability constraints perfectly.
        u_orig = zeros(21, 21)
        v_orig = zeros(21, 21)
        L = 40000.0
        for j in 1:21, i in 1:21
            x = grid.lon[i]
            y = grid.lat[j]
            u_orig[i, j] = sin(π * x / L) * cos(π * y / L)
            v_orig[i, j] = cos(π * x / L) * sin(π * y / L)
        end
        
        u_rot = zeros(21, 21)
        v_rot = zeros(21, 21)
        u_div = zeros(21, 21)
        v_div = zeros(21, 21)
        
        # Decompose with tight convergence tolerances
        helmholtz_decompose!(u_rot, v_rot, u_div, v_div, u_orig, v_orig, grid; max_iter=2000, tol=1e-6)
        
        # Divergent velocity should be extremely close to original velocity
        # Rotational velocity should be extremely close to zero
        @test u_rot[10, 10] ≈ 0.0 atol=1e-2
        @test v_rot[10, 10] ≈ 0.0 atol=1e-2
        @test u_div[10, 10] ≈ u_orig[10, 10] rtol=1e-1
    end
    
    # SFS Stresses and cross-scale energy transfer (Π) calculations
    @testset "Diagnostics & Pipeline" begin
        # 2D rigid-body rotation u = -Ωy, v = Ωx has zero kinetic energy transfer (Π = 0)
        geom = CartesianGeometry(2000.0, 2000.0)
        lon = collect(-20000.0:2000.0:20000.0) # 21 points
        lat = collect(-20000.0:2000.0:20000.0) # 21 points
        mask = trues(21, 21)
        grid = StructuredGrid(geom, lon, lat, mask)
        
        Ω = 1e-4 # Coriolis frequency-like rotation rate
        u = zeros(21, 21)
        v = zeros(21, 21)
        for j in 1:21, i in 1:21
            u[i, j] = -Ω * grid.lat[j]
            v[i, j] = Ω * grid.lon[i]
        end
        
        Π = zeros(21, 21)
        compute_Pi!(Π, u, v, nothing, grid, TopHatKernel(), 10000.0)
        
        # Kinetic energy transfer must be zero (rigid body rotation is pure laminar cascade-free flow)
        @test Π[11, 11] ≈ 0.0 atol=1e-12
    end
end
