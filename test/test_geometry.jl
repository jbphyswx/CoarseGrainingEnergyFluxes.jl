
# Coordinate system and distance tests
Test.@testset "Geometry" begin
    # 2D Cartesian
    geom_cart = FG.Geometry.CartesianGeometry()
    p1 = SA.SVector{2,Float64}(0.0, 0.0)
    p2 = SA.SVector{2,Float64}(3000.0, 4000.0)
    Test.@test FG.Geometry.distance(geom_cart, p1, p2) ≈ 5000.0
    # The geometry carries no spacing — a cell's extent belongs to the grid's axes — so the extents
    # are the caller's to supply.
    Test.@test FG.Geometry.area_element(geom_cart, 1000.0, 1000.0) ≈ 1000.0 * 1000.0

    # Spherical
    geom_sph = FG.Geometry.SphericalGeometry(6371000.0)
    # London (0.1278 W, 51.5074 N) to Paris (2.3522 E, 48.8566 N)
    # Coordinates in radians
    london = SA.SVector{2,Float64}(deg2rad(-0.1278), deg2rad(51.5074))
    paris  = SA.SVector{2,Float64}(deg2rad(2.3522), deg2rad(48.8566))
    d_km = FG.Geometry.distance(geom_sph, london, paris) / 1000.0
    Test.@test 340.0 < d_km < 350.0 # Paris-London ≈ 344 km

    # Coordinate projection conversions: to and from planetary Cartesian
    u_east, u_north = 10.0, -5.0
    λ, φ = deg2rad(-122.0), deg2rad(38.0) # San Francisco coords
    p_vel = FG.Geometry.vector_to_cartesian(geom_sph, u_east, u_north, λ, φ)
    Test.@test length(p_vel) == 3

    l_vel = FG.Geometry.vector_from_cartesian(geom_sph, p_vel[1], p_vel[2], p_vel[3], λ, φ)
    Test.@test l_vel[1] ≈ u_east
    Test.@test l_vel[2] ≈ u_north
    Test.@test abs(l_vel[3]) < 1e-12
end


# Test great-circle distance accuracy
Test.@testset "Great-Circle Distance Accuracy" begin
    geom = FG.Geometry.SphericalGeometry(6371000.0)

    # Test: distance from (0, 0) to (0, 90) should be ~1/4 Earth circumference
    p1 = SA.SVector{2,Float64}(0.0, 0.0)  # (lon, lat) = (0, 0) on equator
    p2 = SA.SVector{2,Float64}(0.0, deg2rad(90.0))  # North pole
    d = FG.Geometry.distance(geom, p1, p2)

    # Should be approximately quarter circumference
    quarter_circumference = π * geom.R / 2
    Test.@test d ≈ quarter_circumference rtol=1e-6

    # Test: distance along equator for 1 degree
    p3 = SA.SVector{2,Float64}(0.0, 0.0)
    p4 = SA.SVector{2,Float64}(deg2rad(1.0), 0.0)
    d_equator = FG.Geometry.distance(geom, p3, p4)

    # Should be approximately 111.195 km per degree at equator (2πR/360)
    Test.@test d_equator ≈ π * geom.R / 180 rtol=1e-6
end
