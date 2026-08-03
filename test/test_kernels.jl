
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

    # Gaussian footprint truncates where the weight is negligible: ~2ℓ for α=6
    r = CGEF.Kernels.kernel_radius(g, ℓ)
    Test.@test 1.5ℓ < r < 2.5ℓ
    Test.@test CGEF.Kernels.kernel_weight(g, r, ℓ) < 1e-9

    # SharpSpectralKernel: exact spherical-cap-window transfer function, sinc real-space fallback
    # with a 10x-scale window (20x TopHatKernel's ℓ/2 — sinc decays only as O(1/d)).
    Test.@test CGEF.Kernels.kernel_radius(ss, ℓ) == 10 * ℓ
    Test.@test CGEF.Kernels.spectral_transfer(ss, π / ℓ - 1e-6, ℓ) == 1.0
    Test.@test CGEF.Kernels.spectral_transfer(ss, π / ℓ + 1e-6, ℓ) == 0.0
end


# `SharpSpectralKernel` with the default `RealSpace()` method computes exactly what was asked for:
# the truncated-sinc real-space form. No interception — the accuracy tradeoff is documented on the
# kernel, and the caller's explicit method choice is honoured on every grid, spectral-capable or not.
Test.@testset "SharpSpectralKernel + RealSpace(): computes what was asked, on any grid" begin
    ss = CGEF.SharpSpectralKernel()
    scale = 5000.0
    geom = FG.Geometry.CartesianGeometry()
    N = 16
    xsR = 0.0:1000.0:(N - 1) * 1000.0
    field = randn(N, N)

    for grid in (
        FG.Grids.StructuredGrid(geom, xsR, xsR, trues(N, N); periodic = (true, true)),  # FFTW-capable
        FG.Grids.StructuredGrid(geom, xsR, xsR, trues(N, N)),                            # not FFTW-capable
    )
        out = zeros(N, N)
        CGEF.Filtering.filter_field!(out, field, grid, ss, scale)
        Test.@test all(isfinite, out)
    end

    # A CurvilinearGrid has no spectral backend at all; real-space must still just work.
    ic = collect(0.0:(N - 1)); jc = collect(0.0:(N - 1))
    clon = [1000.0 * ii for ii in ic, jj in jc]
    clat = [1000.0 * jj for ii in ic, jj in jc]
    cgrid = FG.Grids.CurvilinearGrid(geom, clon, clat, trues(N, N))
    Test.@test CGEF.Filtering.plan_filter(cgrid, ss, scale) isa CGEF.Filtering.AbstractFilterPlan
end


# Test kernel normalization (weights must sum to 1.0 for uniform field)
Test.@testset "Kernel Normalization" begin
    geom = FG.Geometry.SphericalGeometry(6371000.0)
    lon_deg = collect(0.0:2.0:10.0)
    lat_deg = collect(0.0:2.0:10.0)
    lon_rad = deg2rad.(lon_deg)
    lat_rad = deg2rad.(lat_deg)
    mask = trues(length(lon_deg), length(lat_deg))
    grid = FG.Grids.StructuredGrid(geom, lon_rad, lat_rad, mask)

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
