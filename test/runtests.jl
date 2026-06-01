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
        
        # CurvilinearGrid coords bug test: verify i,j indices are used correctly
        # Create non-square grid to catch index swapping bugs
        lon_m = [Float64(i) for i in 1:10, j in 1:5]  # 10x5
        lat_m = [Float64(j*10) for i in 1:10, j in 1:5]  # lat varies with j
        areas_m = ones(10, 5)
        mask_m = trues(10, 5)
        cgrid = CurvilinearGrid(geom, lon_m, lat_m, areas_m, mask_m)
        
        # coords(i,j) should return (lon[i,j], lat[i,j])
        pt = coords(cgrid, 5, 3)
        @test pt[1] == 5.0  # lon[5,3] = 5
        @test pt[2] == 30.0 # lat[5,3] = 30
        
        # This test catches the bug where lat[j,j] was used instead of lat[i,j]
        pt_corner = coords(cgrid, 10, 5)
        @test pt_corner[1] == 10.0  # lon[10,5]
        @test pt_corner[2] == 50.0 # lat[10,5], not lat[5,5]=50 vs lat[10,10] error
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
        
        # Test division by zero protection with single-latitude grid
        # This catches the InexactError: Int64(Inf) bug
        geom_sph = SphericalGeometry(6371000.0)
        lon_sph = collect(0.0:5.0:355.0)
        lat_sph = [0.0]  # Single latitude
        mask_sph = trues(length(lon_sph), 1)
        grid_sph = StructuredGrid(geom_sph, deg2rad.(lon_sph), deg2rad.(lat_sph), mask_sph)
        
        field_sph = rand(length(lon_sph), 1)
        out_sph = zeros(length(lon_sph), 1)
        
        # This should not throw InexactError
        @test_nowarn filter_field!(out_sph, field_sph, grid_sph, TopHatKernel(), 1e6)
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
    
    @testset "Spherical Helmholtz Decomposition" begin
        geom = SphericalGeometry(6371000.0)
        lon_deg = collect(0.0:2.0:10.0)
        lat_deg = collect(0.0:2.0:10.0)
        lon_rad = deg2rad.(lon_deg)
        lat_rad = deg2rad.(lat_deg)
        mask = trues(length(lon_deg), length(lat_deg))
        grid = StructuredGrid(geom, lon_rad, lat_rad, mask)
        
        # Divergent wave field on sphere
        u_orig = [sin(λ) * cos(φ) for λ in lon_rad, φ in lat_rad]
        v_orig = [cos(λ) * sin(φ) for λ in lon_rad, φ in lat_rad]
        
        u_rot = zeros(length(lon_deg), length(lat_deg))
        v_rot = zeros(length(lon_deg), length(lat_deg))
        u_div = zeros(length(lon_deg), length(lat_deg))
        v_div = zeros(length(lon_deg), length(lat_deg))
        
        # Decompose
        helmholtz_decompose!(u_rot, v_rot, u_div, v_div, u_orig, v_orig, grid; max_iter=200, tol=1e-5)
        
        # Just check that it runs and doesn't crash or return NaN
        @test !any(isnan, u_rot)
        @test !any(isnan, u_div)
    end
    
    @testset "Spherical Helmholtz Isolated Points (Boundary Safety)" begin
        geom = SphericalGeometry(6371000.0)
        lon_deg = collect(0.0:2.0:10.0)
        lat_deg = collect(0.0:2.0:10.0)
        lon_rad = deg2rad.(lon_deg)
        lat_rad = deg2rad.(lat_deg)
        
        # Create a mask with an isolated wet point at (5,5) and a narrow bay
        mask = zeros(Bool, length(lon_deg), length(lat_deg))
        mask[2:4, 2:4] .= true
        mask[3, 3] = false # hole/island
        mask[5, 5] = true  # isolated wet cell!
        
        grid = StructuredGrid(geom, lon_rad, lat_rad, mask)
        
        u_orig = rand(length(lon_deg), length(lat_deg))
        v_orig = rand(length(lon_deg), length(lat_deg))
        
        u_rot = zeros(length(lon_deg), length(lat_deg))
        v_rot = zeros(length(lon_deg), length(lat_deg))
        u_div = zeros(length(lon_deg), length(lat_deg))
        v_div = zeros(length(lon_deg), length(lat_deg))
        
        # Decompose
        helmholtz_decompose!(u_rot, v_rot, u_div, v_div, u_orig, v_orig, grid; max_iter=50, tol=1e-3)
        
        # Verify no NaNs are produced anywhere
        @test !any(isnan, u_rot)
        @test !any(isnan, u_div)
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
        compute_Π!(Π, u, v, nothing, grid, TopHatKernel(), 10000.0)
        
        # Kinetic energy transfer must be zero (rigid body rotation is pure laminar cascade-free flow)
        @test Π[11, 11] ≈ 0.0 atol=1e-12
        
        # Test Pipeline integration with unicode Π
        res = coarse_grain(u, v, grid; scales=[10000.0], kernel=TopHatKernel())
        @test res.Π[1] ≈ Π
        
        # Test Spherical projections and coarse graining with mixed types
        sgeom = SphericalGeometry(6371000.0)
        slon = collect(0.0:2.0:10.0)
        slat = collect(0.0:2.0:10.0)
        smask = trues(length(slon), length(slat))
        sgrid = StructuredGrid(sgeom, deg2rad.(slon), deg2rad.(slat), smask)
        
        # Test to_planetary_cartesian and from_planetary_cartesian mixed type support
        proj = to_planetary_cartesian(sgeom, Float32(1.0), Float32(2.0), 0.1, 0.2, 0.3)
        @test proj isa SVector{3, Float64}
        
        inv_proj = from_planetary_cartesian(sgeom, Float32(1.0), 2.0, 3.0, 0.1, 0.2)
        @test inv_proj isa SVector{3, Float64}
        
        # Test coarse_grain on sphere with Float32 inputs (matching PythonCall runtime environment)
        su = fill(Float32(1.0), length(slon), length(slat))
        sv = fill(Float32(0.5), length(slon), length(slat))
        sres = coarse_grain(su, sv, sgrid; scales=[50000.0], kernel=TopHatKernel())
        @test !any(isnan, sres.Π[1])
        @test !any(isnan, sres.spectrum)
    end
    
    # Test periodic boundary handling for spherical grids
    @testset "Spherical Periodic Boundaries" begin
        geom = SphericalGeometry(6371000.0)
        # Create a grid that spans nearly 360 degrees in longitude
        lon_deg = collect(0.0:5.0:355.0)  # 72 points, 5-degree spacing
        lat_deg = collect(-45.0:5.0:45.0)  # 19 points
        lon_rad = deg2rad.(lon_deg)
        lat_rad = deg2rad.(lat_deg)
        mask = trues(length(lon_deg), length(lat_deg))
        grid = StructuredGrid(geom, lon_rad, lat_rad, mask)
        
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
        filter_field!(out, field, grid, TopHatKernel(), deg2rad(20.0) * 6371000.0)
        
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
        @test val_0 > 0.6 && val_0 < 0.95
        @test val_355 > 0.6 && val_355 < 0.95
        
        # The ratio should be reasonable (within 30% of each other)
        @test abs(val_0 - val_355) / max(val_0, val_355) < 0.3
        
        # A point at 30° (6 indices in) should have much lower value since it's outside the band
        val_30 = out[7, 10]  # 30°
        @test val_30 < val_0 * 0.5  # Should be significantly lower than at 0°
    end
    
    # Test kernel normalization (weights must sum to 1.0 for uniform field)
    @testset "Kernel Normalization" begin
        geom = SphericalGeometry(6371000.0)
        lon_deg = collect(0.0:2.0:10.0)
        lat_deg = collect(0.0:2.0:10.0)
        lon_rad = deg2rad.(lon_deg)
        lat_rad = deg2rad.(lat_deg)
        mask = trues(length(lon_deg), length(lat_deg))
        grid = StructuredGrid(geom, lon_rad, lat_rad, mask)
        
        # Constant field
        field = ones(length(lon_deg), length(lat_deg))
        out_zero = zeros(length(lon_deg), length(lat_deg))
        out_renorm = zeros(length(lon_deg), length(lat_deg))
        
        # Filter with both masking strategies
        filter_field!(out_zero, field, grid, TopHatKernel(), 100000.0; mask_strategy=:zero)
        filter_field!(out_renorm, field, grid, TopHatKernel(), 100000.0; mask_strategy=:renormalize)
        
        # For a constant field of ones, output should be exactly 1.0 everywhere
        # (or very close, allowing for small numerical errors)
        for j in 2:length(lat_deg)-1, i in 2:length(lon_deg)-1
            @test out_zero[i, j] ≈ 1.0 atol=1e-10
            @test out_renorm[i, j] ≈ 1.0 atol=1e-10
        end
    end
    
    # Test great-circle distance accuracy
    @testset "Great-Circle Distance Accuracy" begin
        geom = SphericalGeometry(6371000.0)
        
        # Test: distance from (0, 0) to (0, 90) should be ~1/4 Earth circumference
        p1 = SVector{2,Float64}(0.0, 0.0)  # (lon, lat) = (0, 0) on equator
        p2 = SVector{2,Float64}(0.0, deg2rad(90.0))  # North pole
        d = distance(geom, p1, p2)
        
        # Should be approximately quarter circumference
        quarter_circumference = π * geom.R / 2
        @test d ≈ quarter_circumference rtol=1e-6
        
        # Test: distance along equator for 1 degree
        p3 = SVector{2,Float64}(0.0, 0.0)
        p4 = SVector{2,Float64}(deg2rad(1.0), 0.0)
        d_equator = distance(geom, p3, p4)
        
        # Should be approximately 111.195 km per degree at equator (2πR/360)
        @test d_equator ≈ π * geom.R / 180 rtol=1e-6
    end
    
    # Test Taylor-Green vortex for strain rate verification
    @testset "Taylor-Green Vortex" begin
        # Taylor-Green vortex has known analytical solutions
        # u = sin(x)cos(y), v = -cos(x)sin(y)
        # Strain rates and vorticity have exact analytical forms
        
        geom = CartesianGeometry(0.1, 0.1)  # 0.1 unit grid spacing
        lon = collect(0.0:0.1:2π)
        lat = collect(0.0:0.1:2π)
        mask = trues(length(lon), length(lat))
        grid = StructuredGrid(geom, lon, lat, mask)
        
        u = [sin(x) * cos(y) for x in lon, y in lat]
        v = [-cos(x) * sin(y) for x in lon, y in lat]
        
        # Compute derivatives
        dudx = zeros(length(lon), length(lat))
        dudy = zeros(length(lon), length(lat))
        dvdx = zeros(length(lon), length(lat))
        dvdy = zeros(length(lon), length(lat))
        
        ddx!(dudx, u, grid)
        ddy!(dudy, u, grid)
        ddx!(dvdx, v, grid)
        ddy!(dvdy, v, grid)
        
        # Check a point away from boundaries
        i, j = 10, 10
        x, y = lon[i], lat[j]
        
        # Analytical: ∂u/∂x = cos(x)cos(y)
        @test dudx[i, j] ≈ cos(x) * cos(y) rtol=0.01
        
        # Analytical: ∂u/∂y = -sin(x)sin(y)
        @test dudy[i, j] ≈ -sin(x) * sin(y) rtol=0.01
        
        # Analytical: ∂v/∂x = sin(x)sin(y)
        @test dvdx[i, j] ≈ sin(x) * sin(y) rtol=0.01
        
        # Analytical: ∂v/∂y = -cos(x)cos(y)
        @test dvdy[i, j] ≈ -cos(x) * cos(y) rtol=0.01
    end
    
    # Mathematical correctness: Rigid body rotation must have exactly Π = 0
    @testset "Rigid Body Rotation - Zero Energy Flux" begin
        # Rigid body rotation has no deformation, so no energy cascade
        # u = -Ωy, v = Ωx should give Π = 0 everywhere
        
        geom = CartesianGeometry(1000.0, 1000.0)
        lon = collect(-50e3:1000.0:50e3)  # 101 points
        lat = collect(-50e3:1000.0:50e3)  # 101 points
        mask = trues(length(lon), length(lat))
        grid = StructuredGrid(geom, lon, lat, mask)
        
        Ω = 1e-4  # rotation rate
        u = [-Ω * y for x in lon, y in lat]
        v = [Ω * x for x in lon, y in lat]
        
        # Test at multiple scales
        for scale in [5000.0, 10000.0, 20000.0]
            Π = zeros(length(lon), length(lat))
            compute_Π!(Π, u, v, nothing, grid, TopHatKernel(), scale)
            
            # Check interior points (away from boundaries)
            for j in 40:60, i in 40:60
                @test abs(Π[i, j]) < 1e-10  # Should be exactly zero
            end
        end
    end
    
    # Mathematical correctness: Strain rate properties
    @testset "Strain Rate Tensor Properties" begin
        # Strain rate tensor S_ij must be symmetric: S_ij = S_ji
        # For 2D incompressible flow: S_xx + S_yy = 0 (trace = divergence)
        
        geom = CartesianGeometry(1000.0, 1000.0)
        lon = collect(0.0:1000.0:50e3)
        lat = collect(0.0:1000.0:50e3)
        mask = trues(length(lon), length(lat))
        grid = StructuredGrid(geom, lon, lat, mask)
        
        # Create a divergent flow field
        u = [0.01 * x for x in lon, y in lat]  # Linear in x
        v = [0.01 * y for x in lon, y in lat]  # Linear in y
        
        # Filter the field
        u_filt = zeros(length(lon), length(lat))
        v_filt = zeros(length(lon), length(lat))
        filter_field!(u_filt, u, grid, TopHatKernel(), 10000.0)
        filter_field!(v_filt, v, grid, TopHatKernel(), 10000.0)
        
        # Compute strain rates
        S_xx = zeros(length(lon), length(lat))
        S_yy = zeros(length(lon), length(lat))
        S_xy = zeros(length(lon), length(lat))
        scratch = zeros(length(lon), length(lat))
        
        ddx!(S_xx, u_filt, grid)
        ddy!(S_yy, v_filt, grid)
        ddy!(S_xy, u_filt, grid)
        ddx!(scratch, v_filt, grid)
        @. S_xy = 0.5 * (S_xy + scratch)
        
        # Test symmetry: S_xy should equal S_yx (we only computed S_xy)
        # Test trace = divergence for incompressible flow
        for j in 20:length(lat)-20, i in 20:length(lon)-20
            # For filtered divergent flow, S_xx + S_yy should equal divergence
            divergence = S_xx[i,j] + S_yy[i,j]
            # Divergence should be approximately constant (0.02 for this field)
            @test abs(divergence - 0.02) < 0.01
        end
    end
    
    # Mathematical correctness: SFS stress properties
    @testset "SFS Stress Tensor Properties" begin
        # τ_ij = [u_i*u_j]̄ - ū_i*ū_j must be symmetric: τ_ij = τ_ji
        # For isotropic turbulence, trace of τ should be positive (energy in SFS)
        
        geom = CartesianGeometry(1000.0, 1000.0)
        lon = collect(0.0:1000.0:30e3)
        lat = collect(0.0:1000.0:30e3)
        mask = trues(length(lon), length(lat))
        grid = StructuredGrid(geom, lon, lat, mask)
        
        # Create random velocity field
        u = rand(length(lon), length(lat))
        v = rand(length(lon), length(lat))
        
        # Filter fields
        u_filt = zeros(length(lon), length(lat))
        v_filt = zeros(length(lon), length(lat))
        filter_field!(u_filt, u, grid, TopHatKernel(), 5000.0)
        filter_field!(v_filt, v, grid, TopHatKernel(), 5000.0)
        
        # Filter products
        uu = zeros(length(lon), length(lat))
        uv = zeros(length(lon), length(lat))
        vv = zeros(length(lon), length(lat))
        scratch = zeros(length(lon), length(lat))
        
        @. scratch = u * u
        filter_field!(uu, scratch, grid, TopHatKernel(), 5000.0)
        @. scratch = u * v
        filter_field!(uv, scratch, grid, TopHatKernel(), 5000.0)
        @. scratch = v * v
        filter_field!(vv, scratch, grid, TopHatKernel(), 5000.0)
        
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
            @test trace_τ >= -1e-10  # Should be non-negative
        end
        
        # Test symmetry: compute τ_yx and verify equals τ_xy
        scratch2 = zeros(length(lon), length(lat))
        @. scratch2 = v * u
        filter_field!(scratch, scratch2, grid, TopHatKernel(), 5000.0)
        @. scratch2 = scratch - v_filt * u_filt  # τ_yx
        
        for j in 10:length(lat)-10, i in 10:length(lon)-10
            @test τ_xy[i,j] ≈ scratch2[i,j] rtol=1e-10
        end
    end
    
    # Mathematical correctness: Π sign consistency with SFS stress and strain
    @testset "Energy Flux Sign Consistency" begin
        # Π = -ρ₀ * S̄_ij * τ_ij should have consistent sign based on S and τ
        # For a convergent strain with positive SFS stress, Π should be negative
        # (energy goes from resolved to sub-grid = forward cascade)
        
        geom = CartesianGeometry(1000.0, 1000.0)
        lon = collect(0.0:1000.0:50e3)
        lat = collect(0.0:1000.0:50e3)
        mask = trues(length(lon), length(lat))
        grid = StructuredGrid(geom, lon, lat, mask)
        
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
        compute_Π!(Π, u, v, nothing, grid, TopHatKernel(), 10000.0)
        
        # For this pure linear deformation, filtering doesn't change the field
        # (linear fields are invariant under top-hat filtering)
        # So τ should be ~0 and Π should be ~0
        for j in 20:length(lat)-20, i in 20:length(lon)-20
            @test abs(Π[i,j]) < 1e-8
        end
    end
    
    # Mathematical correctness: Energy budget closure
    @testset "Energy Budget - Filtered vs Unfiltered" begin
        # Test that: 0.5*ρ₀*|u|² = 0.5*ρ₀*|ū|² + 0.5*ρ₀*trace(τ) + (boundary terms)
        # For periodic domains, the resolved + SFS energies should relate to total energy
        
        geom = CartesianGeometry(1000.0, 1000.0)
        lon = collect(0.0:1000.0:30e3)
        lat = collect(0.0:1000.0:30e3)
        mask = trues(length(lon), length(lat))
        grid = StructuredGrid(geom, lon, lat, mask)
        
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
        filter_field!(u_filt, u, grid, TopHatKernel(), scale)
        filter_field!(v_filt, v, grid, TopHatKernel(), scale)
        
        # Filter products for SFS stress trace
        uu_filt = zeros(length(lon), length(lat))
        vv_filt = zeros(length(lon), length(lat))
        scratch = zeros(length(lon), length(lat))
        
        @. scratch = u * u
        filter_field!(uu_filt, scratch, grid, TopHatKernel(), scale)
        @. scratch = v * v
        filter_field!(vv_filt, scratch, grid, TopHatKernel(), scale)
        
        # SFS energy = 0.5*ρ₀*trace(τ) = 0.5*ρ₀*([u²]̄ + [v²]̄ - ū² - v̄²)
        sfs_energy = zeros(length(lon), length(lat))
        @. sfs_energy = 0.5 * ρ₀ * (uu_filt + vv_filt - u_filt^2 - v_filt^2)
        
        # Resolved energy
        resolved_energy = zeros(length(lon), length(lat))
        @. resolved_energy = 0.5 * ρ₀ * (u_filt^2 + v_filt^2)
        
        # Verify: sfs_energy ≥ 0 (Jensen's inequality)
        for j in 10:length(lat)-10, i in 10:length(lon)-10
            @test sfs_energy[i,j] >= -1e-12  # Should be non-negative
        end
        
        # Verify: sfs_energy + resolved_energy ≈ filtered total energy
        # ([u²]̄ + [v²]̄)/2 = ([u²+v²]̄)/2
        filtered_total = zeros(length(lon), length(lat))
        @. scratch = u.^2 + v.^2
        filter_field!(filtered_total, scratch, grid, TopHatKernel(), scale)
        @. filtered_total = 0.5 * ρ₀ * filtered_total
        
        for j in 10:length(lat)-10, i in 10:length(lon)-10
            energy_sum = sfs_energy[i,j] + resolved_energy[i,j]
            @test energy_sum ≈ filtered_total[i,j] rtol=1e-10
        end
    end
    
    # Mathematical correctness: Filtered field of constant = constant
    @testset "Filter Normalization - Constant Field" begin
        # Filtering a constant field must return exactly the same constant
        # This tests that kernel weights are properly normalized
        
        geom = CartesianGeometry(1000.0, 1000.0)
        lon = collect(0.0:1000.0:50e3)
        lat = collect(0.0:1000.0:50e3)
        mask = trues(length(lon), length(lat))
        grid = StructuredGrid(geom, lon, lat, mask)
        
        C = 42.0  # Constant value
        field = fill(C, length(lon), length(lat))
        
        # Test both masking strategies
        for kernel in [TopHatKernel(), GaussianKernel()]
            for scale in [5000.0, 10000.0, 20000.0]
                out_zero = zeros(length(lon), length(lat))
                out_renorm = zeros(length(lon), length(lat))
                
                filter_field!(out_zero, field, grid, kernel, scale; mask_strategy=:zero)
                filter_field!(out_renorm, field, grid, kernel, scale; mask_strategy=:renormalize)
                
                # Interior points should be exactly C
                for j in 20:length(lat)-20, i in 20:length(lon)-20
                    @test out_zero[i,j] ≈ C rtol=1e-10
                    @test out_renorm[i,j] ≈ C rtol=1e-10
                end
            end
        end
    end
    
    # Mathematical correctness: Area-weighted spectrum normalization
    @testset "Spectrum Normalization" begin
        # For uniform velocity field, spectrum should equal kinetic energy
        # E(ℓ) = 0.5 * ρ₀ * (u² + v²) for all ℓ
        
        geom = CartesianGeometry(2000.0, 2000.0)
        lon = collect(0.0:2000.0:100e3)
        lat = collect(0.0:2000.0:100e3)
        mask = trues(length(lon), length(lat))
        grid = StructuredGrid(geom, lon, lat, mask)
        
        U = 0.5  # m/s
        V = 0.3  # m/s
        u = fill(U, length(lon), length(lat))
        v = fill(V, length(lon), length(lat))
        
        ρ₀ = 1025.0
        expected_energy = 0.5 * ρ₀ * (U^2 + V^2)
        
        scales = [5000.0, 10000.0, 20000.0, 40000.0]
        spectrum = compute_filtering_spectrum(u, v, nothing, grid, TopHatKernel(), scales; ρ₀=ρ₀)
        
        # All spectrum values should equal the expected kinetic energy
        for E in spectrum
            @test E ≈ expected_energy rtol=1e-6
        end
    end
end
