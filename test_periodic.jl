using CoarseGrainingEnergyFluxes

# Test periodic boundary with proper 2D spherical grid
geom = SphericalGeometry(6371000.0)

# Grid spanning full 360 longitude, multiple latitudes
lon_deg = collect(0.0:5.0:355.0)  # 72 points, 5° spacing
lat_deg = collect(-20.0:5.0:20.0)  # 9 points
lon_rad = deg2rad.(lon_deg)
lat_rad = deg2rad.(lat_deg)

mask = trues(length(lon_deg), length(lat_deg))
grid = StructuredGrid(geom, lon_rad, lat_rad, mask)

# Create field with sharp transition at prime meridian
# Field = 1.0 for 0-20°, 0.0 for 20-340°, 1.0 for 340-360°
# This creates a 20° wide band around 0°
field = zeros(length(lon_deg), length(lat_deg))
for j in 1:length(lat_deg), i in 1:length(lon_deg)
    lon_i = lon_deg[i]
    if lon_i < 20.0 || lon_i > 340.0  # Within 20° of 0°
        field[i,j] = 1.0
    else
        field[i,j] = 0.0
    end
end

println("Field: 1.0 near 0°, 0.0 elsewhere")
println("Values around boundary:")
println("  Index 1 (0°): ", field[1,5])
println("  Index 2 (5°): ", field[2,5])
println("  Index end (355°): ", field[end,5])
println("  Index end-1 (350°): ", field[end-1,5])

# Filter at 20 degree scale
scale = deg2rad(20.0) * 6371000.0
println("\nFilter scale: ", scale/1000, " km (20° at equator)")

# Check di_lim calculation
rad = CoarseGrainingEnergyFluxes.Filtering.kernel_radius(CoarseGrainingEnergyFluxes.TopHatKernel(), scale)
R = 6371000.0
dλ = deg2rad(5.0)
cosφ = 1.0  # at equator
di_lim = ceil(Int, rad / (R * cosφ * dλ))
println("di_lim = ", di_lim)
println("rad = ", rad/1000, " km")
println("R * dλ = ", R * dλ / 1000, " km")

out = zeros(length(lon_deg), length(lat_deg))
filter_field!(out, field, grid, TopHatKernel(), scale)

println("\nFiltered (20° scale):")
println("  Index 1 (0°): ", out[1,5])
println("  Index 2 (5°): ", out[2,5])
println("  Index end (355°): ", out[end,5])
println("  Index end-1 (350°): ", out[end-1,5])

# The key test: points at 0° and 355° should have similar filtered values
# because they're close in physical space
val_0 = out[1,5]
val_355 = out[end,5]
println("\nPeriodic boundary test:")
println("  Value at 0°: ", val_0)
println("  Value at 355°: ", val_355)
println("  Difference: ", abs(val_0 - val_355))

if abs(val_0 - val_355) < 0.1
    println("  ✓ Periodic boundary handling appears correct")
else
    println("  ✗ WARNING: Large difference across boundary!")
end
