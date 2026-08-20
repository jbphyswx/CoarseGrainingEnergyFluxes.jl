
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

# ---------------------------------------------------------------------------
# Kernel moment gates. The coarse-graining framework's error terms are stated in terms of the kernel's
# moments — `∫G = 1` (a constant filters to itself), `∫xG = 0` (no centroid shift, hence no spurious
# advection), and `∫x²G` (the leading truncation term, and what sets the spectrum's slope ceiling). All
# three are checked against closed forms rather than against a stored reference.
# ---------------------------------------------------------------------------

# Radial quadrature of the continuum moments over the kernel's own support, `ℓ = 1`. Midpoints, so the
# ∫ is second-order and comfortably converged at this resolution.
function _cgef_continuum_moments(kern, D::Int; n = 400_000)
    ℓ = 1.0
    R = CGEF.Kernels.kernel_radius(kern, ℓ)
    dr = R / n
    m0 = 0.0
    m2 = 0.0
    for i in 1:n
        r = (i - 0.5) * dr
        w = CGEF.Kernels.kernel_weight(kern, r, ℓ)
        # Radial measure: 2 (both signs) in 1-D, 2πr in 2-D, 4πr² in 3-D.
        μ = D == 1 ? 2.0 : (D == 2 ? 2π * r : 4π * r^2)
        m0 += w * μ * dr
        m2 += w * μ * r^2 * dr
    end
    return m0, m2 / m0                      # ⟨r²⟩; per component this is ⟨r²⟩/D by isotropy
end

# Moments of the DISCRETE 2-D footprint the package actually builds, at the centre of a domain wide
# enough that nothing is truncated.
function _cgef_discrete_moments_2d(kern, ℓ, dx)
    lim = ceil(Int, CGEF.Kernels.kernel_radius(kern, ℓ) / dx)
    m0 = 0.0; mx = 0.0; mxx = 0.0
    for j in (-lim):lim, i in (-lim):lim
        w = CGEF.Kernels.kernel_weight(kern, dx * sqrt(i^2 + j^2), ℓ)
        iszero(w) && continue
        m0 += w
        mx += w * (dx * i)
        mxx += w * (dx * i)^2
    end
    return mx / m0, mxx / m0
end

Test.@testset "Kernel moments: normalization, vanishing first moment, second moment" begin
    # The Gaussian is separable, so its per-component variance is ℓ²/(2α) in EVERY dimension. The
    # top-hat is the disk/ball of radius ℓ/2, not a separable box, so its per-component variance
    # shrinks with dimension: ℓ²/12, ℓ²/16, ℓ²/20. These are the values `GaussianKernel`'s docstring
    # quotes, and the reason `α = 6` matches the top-hat only in 1-D.
    expected = Dict(
        (CGEF.TopHatKernel(), 1) => 1 / 12,
        (CGEF.TopHatKernel(), 2) => 1 / 16,
        (CGEF.TopHatKernel(), 3) => 1 / 20,
        (CGEF.GaussianKernel(), 1) => 1 / 12,
        (CGEF.GaussianKernel(), 2) => 1 / 12,
        (CGEF.GaussianKernel(), 3) => 1 / 12,
        (CGEF.GaussianKernel(; α = 4), 1) => 1 / 8,
        (CGEF.GaussianKernel(; α = 4), 2) => 1 / 8,
        (CGEF.GaussianKernel(; α = 4), 3) => 1 / 8,
        (CGEF.GaussianKernel(; α = 8), 2) => 1 / 16,   # variance-matches the 2-D disk top-hat
        (CGEF.GaussianKernel(; α = 10), 3) => 1 / 20,  # ... and the 3-D ball
    )
    for ((kern, D), x2) in expected
        m0, r2 = _cgef_continuum_moments(kern, D)
        Test.@test m0 > 0                       # normalizable: the filter divides by this
        Test.@test r2 / D ≈ x2 rtol = 1e-4      # per-component second moment
    end

    # `α = 6` is 15% wider in RMS than the 2-D disk top-hat of the same nominal ℓ — the number in the
    # docstring.
    let (_, r2g) = _cgef_continuum_moments(CGEF.GaussianKernel(), 2),
        (_, r2t) = _cgef_continuum_moments(CGEF.TopHatKernel(), 2)
        Test.@test sqrt(r2g / r2t) ≈ 1.155 rtol = 1e-3
    end

    # Every kernel here is even in `d`, so the discrete first moment must vanish to round-off — a
    # non-zero one would advect the field while filtering it.
    for kern in (CGEF.TopHatKernel(), CGEF.GaussianKernel(), CGEF.GaussianKernel(; α = 4),
                 CGEF.Kernels.SmoothHatKernel(), CGEF.Kernels.HyperGaussianKernel())
        x, _ = _cgef_discrete_moments_2d(kern, 1.0, 1 / 16)
        Test.@test abs(x) < 1e-15
    end

    # `HyperGaussianKernel` has closed-form moments too: with `a = 16α/ℓ⁴`,
    # `∫₀^∞ rⁿ e^{-ar⁴} dr = Γ((n+1)/4) / (4 a^((n+1)/4))`, and `a^(-1/2) = ℓ²/(4√α)`.
    for α in (1.0, 2.5)
        s = 1 / (4 * sqrt(α))                            # a^(-1/2), in units of ℓ²
        Γ = SpecialFunctions.gamma
        for (D, x2) in ((1, (Γ(3 / 4) / Γ(1 / 4)) * s),
                        (2, s / sqrt(π) / 2),
                        (3, (Γ(5 / 4) / Γ(3 / 4)) * s / 3))
            _, r2 = _cgef_continuum_moments(CGEF.Kernels.HyperGaussianKernel(; α = α), D)
            Test.@test r2 / D ≈ x2 rtol = 1e-4
        end
    end

    # `SmoothHatKernel` has no closed-form moment (the tanh integral does not reduce), so it is pinned
    # to quadrature values instead — and to the structural fact that a smoothed rim sits just OUTSIDE
    # the box it tapers, so its variance exceeds the top-hat's in every dimension.
    let sh = CGEF.Kernels.SmoothHatKernel()
        for (D, x2) in ((1, 0.0853895), (2, 0.0650669), (3, 0.0528786))
            _, r2 = _cgef_continuum_moments(sh, D)
            Test.@test r2 / D ≈ x2 rtol = 1e-4
            th = _cgef_continuum_moments(CGEF.TopHatKernel(), D)[2] / D
            Test.@test r2 / D > th
        end
        # Steeper ⇒ closer to the box it is a smoothing of.
        d10 = _cgef_continuum_moments(sh, 2)[2] / 2
        d40 = _cgef_continuum_moments(CGEF.Kernels.SmoothHatKernel(; steepness = 40), 2)[2] / 2
        th2 = _cgef_continuum_moments(CGEF.TopHatKernel(), 2)[2] / 2
        Test.@test abs(d40 - th2) < abs(d10 - th2)
    end

    # Realizability: `Π` is only a meaningful pointwise transfer for a NON-NEGATIVE kernel. The two new
    # real-space kernels are strictly positive on their support; `SharpSpectralKernel`'s sinc is not,
    # which is why it is documented as spectrum-only.
    for kern in (CGEF.TopHatKernel(), CGEF.GaussianKernel(),
                 CGEF.Kernels.SmoothHatKernel(), CGEF.Kernels.HyperGaussianKernel())
        R = CGEF.Kernels.kernel_radius(kern, 1.0)
        Test.@test all(r -> CGEF.Kernels.kernel_weight(kern, r, 1.0) >= 0,
                       range(0.0, R; length = 4001))
    end
    Test.@test any(r -> CGEF.Kernels.kernel_weight(CGEF.SharpSpectralKernel(), r, 1.0) < 0,
                   range(0.0, 10.0; length = 4001))

    # Discretization: the Gaussian's footprint reproduces its continuum second moment essentially
    # exactly even at ℓ = 8Δx, while the top-hat's staircased disk boundary leaves it low and converges
    # only as O(Δx). Both numbers are quoted in `GaussianKernel`'s docstring.
    let cont(kern) = _cgef_continuum_moments(kern, 2)[2] / 2
        for kern in (CGEF.GaussianKernel(), CGEF.GaussianKernel(; α = 4))
            _, x2 = _cgef_discrete_moments_2d(kern, 1.0, 1 / 8)
            Test.@test abs(x2 - cont(kern)) < 1e-6 * cont(kern)
        end
        th = CGEF.TopHatKernel(); c = cont(th)
        _, x2_8 = _cgef_discrete_moments_2d(th, 1.0, 1 / 8)
        _, x2_64 = _cgef_discrete_moments_2d(th, 1.0, 1 / 64)
        Test.@test x2_8 < c && x2_64 < c                       # staircasing loses mass at the rim
        Test.@test 0.010 < (c - x2_8) / c < 0.030              # ~1.5% at ℓ = 8Δx
        Test.@test (c - x2_64) / c < 0.005                     # ~0.24% at ℓ = 64Δx
        Test.@test (c - x2_64) < 0.25 * (c - x2_8)             # converges, not a fixed offset
    end

    # `∫G = 1` where it actually matters: the normalized filter reproduces a constant to round-off on a
    # grid wide enough that the footprint is never truncated. Every dimensionality the package supports.
    let geom = FG.Geometry.CartesianGeometry(), dx = 1000.0, ℓ = 4000.0, C = 3.25
        for kern in (CGEF.TopHatKernel(), CGEF.GaussianKernel(), CGEF.GaussianKernel(; α = 4),
                     CGEF.Kernels.SmoothHatKernel(), CGEF.Kernels.HyperGaussianKernel())
            n1 = 64
            x1 = 0.0:dx:(n1 - 1) * dx
            g1 = FG.Grids.StructuredGrid(geom, x1, trues(n1))
            o1 = zeros(n1)
            CGEF.Filtering.filter_field!(o1, fill(C, n1), g1, kern, ℓ)
            Test.@test all(≈(C; rtol = 1e-12), o1)

            n2 = 40
            x2 = 0.0:dx:(n2 - 1) * dx
            g2 = FG.Grids.StructuredGrid(geom, x2, x2, trues(n2, n2))
            o2 = zeros(n2, n2)
            CGEF.Filtering.filter_field!(o2, fill(C, n2, n2), g2, kern, ℓ)
            Test.@test all(≈(C; rtol = 1e-12), o2)

            n3 = 20
            x3 = 0.0:dx:(n3 - 1) * dx
            g3 = FG.Grids.StructuredGrid(geom, x3, x3, x3, trues(n3, n3, n3))
            o3 = zeros(n3, n3, n3)
            CGEF.Filtering.filter_field!(o3, fill(C, n3, n3, n3), g3, kern, ℓ)
            Test.@test all(≈(C; rtol = 1e-12), o3)
        end
    end
end

# `SmoothHatKernel` and `HyperGaussianKernel` are real-space-only by construction: neither has an
# elementary Fourier transform, and both ring, so neither can carry a filtering spectrum. What they buy
# is a smooth, strictly-positive alternative to the box.
Test.@testset "SmoothHat and HyperGaussian: real-space only, and what they sit between" begin
    sh = CGEF.Kernels.SmoothHatKernel()
    hg = CGEF.Kernels.HyperGaussianKernel()
    Test.@test sh.steepness == 10.0                        # the published value
    Test.@test hg.α == 1.0
    Test.@test CGEF.Kernels.SmoothHatKernel(; steepness = 4).steepness == 4
    Test.@test CGEF.Kernels.HyperGaussianKernel(; α = 2).α == 2

    # Shape: at the rim (d = ℓ/2) a smooth hat is exactly ½, and it is monotone decreasing in `d`.
    Test.@test CGEF.Kernels.kernel_weight(sh, 0.5, 1.0) ≈ 0.5
    Test.@test CGEF.Kernels.kernel_weight(sh, 0.0, 1.0) > 0.999
    Test.@test CGEF.Kernels.kernel_weight(hg, 0.0, 1.0) == 1.0
    Test.@test CGEF.Kernels.kernel_weight(hg, 0.5, 1.0) ≈ exp(-1)
    for kern in (sh, hg)
        ws = [CGEF.Kernels.kernel_weight(kern, r, 1.0) for r in range(0, 1.2; length = 500)]
        Test.@test all(<=(0), diff(ws))
    end

    # Truncation radius is where the weight reaches the shared tolerance, so the footprint is neither
    # clipped early nor needlessly wide.
    for kern in (sh, hg)
        R = CGEF.Kernels.kernel_radius(kern, 1.0)
        Test.@test CGEF.Kernels.kernel_weight(kern, R, 1.0) ≈ 1e-10 rtol = 1e-6
        Test.@test 1.0 < R < 1.2
    end

    # No closed-form transfer ⇒ an explicit refusal naming the fix, not a MethodError and not a
    # silently substituted kernel.
    for kern in (sh, hg)
        err = try
            CGEF.Kernels.spectral_transfer(kern, 1.0, 1.0); nothing
        catch e; e end
        Test.@test err isa ArgumentError
        Test.@test occursin("RealSpace", err.msg)
        Test.@test_throws ArgumentError CGEF.Kernels.spectral_transfer_degree(kern, 3, 1.0, 6.371e6)
        # ... and the same refusal reaches the user through `plan_filter`, not just the raw transfer.
        let g = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(),
                                        0.0:1000.0:31_000.0, 0.0:1000.0:31_000.0;
                                        periodic = (true, true))
            Test.@test_throws ArgumentError CGEF.Filtering.plan_filter(
                g, kern, 4000.0; method = CGEF.Filtering.Spectral())
        end
        # Both ring, so neither can produce a filtering spectral density.
        Test.@test !CGEF.Kernels.transfer_monotone(kern)
    end

    # Both are real-space filters that work end to end, and both land between the box and the Gaussian
    # in effective width — the reason to reach for them at all.
    let geom = FG.Geometry.CartesianGeometry(), N = 40, dx = 1000.0,
        xs = 0.0:dx:(N - 1) * dx,
        grid = FG.Grids.StructuredGrid(geom, xs, xs, trues(N, N)),
        u = [sin(i / 5) * cos(j / 7) for i in 1:N, j in 1:N],
        v = [cos(i / 6) * sin(j / 4) for i in 1:N, j in 1:N],
        ℓ = 4000.0
        m2(kern) = _cgef_continuum_moments(kern, 2)[2] / 2
        Test.@test m2(CGEF.TopHatKernel()) < m2(sh) < m2(hg) < m2(CGEF.GaussianKernel())
        for kern in (sh, hg)
            Π = zeros(N, N)
            CGEF.Diagnostics.compute_Π!(Π, u, v, nothing, grid, kern, ℓ)
            Test.@test all(isfinite, Π)
            Test.@test maximum(abs, Π) > 0
            # The generic banded engine is what a kernel with no special-cased algebra should reach.
            Test.@test CGEF.Filtering.plan_filter(grid, kern, ℓ).footprint isa
                       CGEF.Filtering.FilterFootprint
        end
    end
end
