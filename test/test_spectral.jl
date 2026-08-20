
# Spectral (FFT) filtering on a uniform doubly-periodic Cartesian grid
Test.@testset "Spectral FFTW filtering" begin
    N = 32
    dx = 1.0
    geom = FG.Geometry.CartesianGeometry()
    x = 0.0:dx:dx*(N - 1)
    y = 0.0:dx:dx*(N - 1)
    grid = FG.Grids.StructuredGrid(geom, x, y, trues(N, N); periodic = (true, true))
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

    # TopHat spectral filtering: exact planar transfer Ĝ(k) = 2J₁(kR)/(kR), R = ℓ/2 (SpecialFunctions
    # extension). A single Fourier mode must transform by exactly that factor. RealSpace's real-space
    # engine only approximates the same continuous top-hat convolution (its footprint quadrature can't
    # perfectly resolve the kernel's sharp circular edge against the underlying Cartesian grid cells),
    # so it agrees with the exact spectral result up to that quadrature error, not to machine precision
    # — measured ~1.1% relative (2-norm) here, hence the loose-but-real rtol below.
    th = CGEF.TopHatKernel()
    thR = ℓ / 2
    Ghat_th = 2 * SpecialFunctions.besselj1(kx0 * thR) / (kx0 * thR)
    thout = zeros(N, N)
    CGEF.Filtering.filter_field!(thout, field, grid, th, ℓ; method = CGEF.Filtering.Spectral())
    Test.@test thout ≈ Ghat_th .* field rtol = 1e-8
    thout_ds = zeros(N, N)
    CGEF.Filtering.filter_field!(thout_ds, field, grid, th, ℓ; method = CGEF.Filtering.RealSpace())
    Test.@test thout ≈ thout_ds rtol = 0.02

    # Non-periodic grid: spectral FFT must refuse.
    npgrid = FG.Grids.StructuredGrid(geom, x, y, trues(N, N))  # periodic = (false, false)
    Test.@test_throws ArgumentError CGEF.Filtering.filter_field!(out, field, npgrid, g, ℓ; method = CGEF.Filtering.Spectral())

    # Masked spectral filtering (normalized convolution, Knutsson & Westin 1993): both
    # mask_strategy branches must reproduce the corresponding RealSpace result on the SAME
    # doubly-periodic grid — the two are the same normalized-convolution identity, evaluated in
    # Fourier space instead of real space. Compare ACTIVE cells only: RealSpace leaves a masked
    # cell untouched (its `fill!(out,0)` initial value, by convention — see `apply_footprint_row!`'s
    # docstring), while a spectral method is a global transform that computes a real, meaningful
    # normalized-convolution value EVERYWHERE, including at masked points — the two are expected to
    # differ there, by design, not by bug. A smooth Gaussian kernel's real-space quadrature agrees with
    # the exact spectral result far more tightly than TopHat's (no sharp-edge discretization error),
    # measured ~2e-5 relative (2-norm) here — `rtol`, not `atol`, since `≈` on arrays compares the
    # aggregate 2-norm of the difference against the array, and this same active-cell selection is
    # ~1000 elements wide (an `atol` sized by eye against a single point's expected error silently
    # fails once aggregated over that many elements).
    mask = trues(N, N)
    mask[10:14, 10:14] .= false
    mgrid = FG.Grids.StructuredGrid(geom, x, y, mask; periodic = (true, true))
    rfield = [sin(2π*3*xi/L)*cos(2π*2*yi/L) + 0.3*cos(2π*5*xi/L) for xi in x, yi in y]
    for strat in (CGEF.Filtering.Deformable(), CGEF.Filtering.ZeroFill())
        out_spec = zeros(N, N)
        CGEF.Filtering.filter_field!(out_spec, rfield, mgrid, g, ℓ; method = CGEF.Filtering.Spectral(), mask_strategy = strat)
        out_ds = zeros(N, N)
        CGEF.Filtering.filter_field!(out_ds, rfield, mgrid, g, ℓ; method = CGEF.Filtering.RealSpace(), mask_strategy = strat)
        Test.@test out_spec[mask] ≈ out_ds[mask] rtol = 1e-4
    end
    # Constant field over active cells stays constant under Deformable (mask mass cancels exactly).
    cfield_m = fill(1.7, N, N)
    cout_m = zeros(N, N)
    CGEF.Filtering.filter_field!(cout_m, cfield_m, mgrid, g, ℓ; method = CGEF.Filtering.Spectral(),
                                 mask_strategy = CGEF.Filtering.Deformable())
    Test.@test all(x -> isapprox(x, 1.7; atol = 1e-6), cout_m)

    # Under the default, ZeroFill, the excluded cells stay in the denominator, so the same constant
    # comes back scaled by the locally-included mass — identically `1.7 · filter(mask)`, which is the
    # normalized-convolution numerator with the renormalization left off.
    cz = zeros(N, N); mz = zeros(N, N)
    CGEF.Filtering.filter_field!(cz, cfield_m, mgrid, g, ℓ; method = CGEF.Filtering.Spectral())
    CGEF.Filtering.filter_field!(mz, Float64.(mask), mgrid, g, ℓ; method = CGEF.Filtering.Spectral())
    Test.@test cz ≈ 1.7 .* mz rtol = 1e-12
    Test.@test minimum(cz) < 1.7 - 1e-3

    # The filtered mask slightly EXCEEDS 1 next to the hole (measured +3.3e-5 here). That is the
    # spectral engine, not the strategy: `Ĝ` is sampled on the discrete mode grid, so the real-space
    # kernel it implies is not exactly non-negative and a step overshoots. The real-space engine sums
    # non-negative weights directly and cannot. Both are pinned so a change to either is visible.
    mr = zeros(N, N)
    CGEF.Filtering.filter_field!(mr, Float64.(mask), mgrid, g, ℓ; method = CGEF.Filtering.RealSpace())
    Test.@test 0 < maximum(mz) - 1 < 1e-4
    Test.@test maximum(mr) <= 1 + 1e-12
end


# Scattered-Cartesian spectral filtering (FINUFFT): on a uniform periodic lattice it must
# reproduce the FFTW result, and it must preserve the mean of a constant field.
Test.@testset "Spectral FINUFFT filtering" begin
    Nx, Ny = 32, 24
    dx = dy = 1.0
    geom = FG.Geometry.CartesianGeometry()
    x = 0.0:dx:dx*(Nx - 1); y = 0.0:dy:dy*(Ny - 1)
    u = [sin(2π*xi/(Nx*dx)) + 0.5cos(4π*yj/(Ny*dy)) + 0.3sin(6π*xi/(Nx*dx)) for xi in x, yj in y]
    g = CGEF.GaussianKernel(); ℓ = 4.0

    # FFTW reference on the structured grid.
    sg = FG.Grids.StructuredGrid(geom, x, y, trues(Nx, Ny); periodic = (true, true))
    outf = zeros(Nx, Ny)
    CGEF.Filtering.filter_field!(outf, u, sg, g, ℓ; method = CGEF.Filtering.Spectral())

    # The same points as a scattered (unstructured) grid; FINUFFT spectral filter.
    ptsx = vec([xi for xi in x, _ in y]); ptsy = vec([yj for _ in x, yj in y])
    ug = FG.Grids.UnstructuredGrid(geom, ptsx, ptsy, fill(dx*dy, Nx*Ny), trues(Nx*Ny))
    outu = zeros(Nx*Ny)
    CGEF.Filtering.filter_field!(outu, vec(u), ug, g, ℓ; method = CGEF.Filtering.Spectral())
    Test.@test reshape(outu, Nx, Ny) ≈ outf atol = 1e-7

    # Constant field ⇒ mean preserved (Ĝ(0)=1) for the scattered transform.
    cout = zeros(Nx*Ny)
    CGEF.Filtering.filter_field!(cout, fill(3.7, Nx*Ny), ug, g, ℓ; method = CGEF.Filtering.Spectral())
    Test.@test all(≈(3.7; atol = 1e-6), cout)

    # TopHat spectral filtering (shared transfer function with FFTW): must reproduce the FFTW
    # result on the same points, exactly like the Gaussian case above.
    th = CGEF.TopHatKernel()
    outf_th = zeros(Nx, Ny)
    CGEF.Filtering.filter_field!(outf_th, u, sg, th, ℓ; method = CGEF.Filtering.Spectral())
    outu_th = zeros(Nx*Ny)
    CGEF.Filtering.filter_field!(outu_th, vec(u), ug, th, ℓ; method = CGEF.Filtering.Spectral())
    Test.@test reshape(outu_th, Nx, Ny) ≈ outf_th atol = 1e-7

    # Masked spectral filtering on a node set is cross-checked against the FFTW-on-StructuredGrid
    # reference (already verified against RealSpace above) on the SAME points/mask: the two backends
    # share the same normalized-convolution identity over different point layouts, which is a
    # sharper comparison here than the node set's own real-space engine, whose truncation shape
    # differs from a transform's global support.
    mask2d = trues(Nx, Ny)
    mask2d[8:11, 8:11] .= false
    smgrid = FG.Grids.StructuredGrid(geom, x, y, mask2d; periodic = (true, true))
    umgrid = FG.Grids.UnstructuredGrid(geom, ptsx, ptsy, fill(dx*dy, Nx*Ny), vec(mask2d))
    for strat in (CGEF.Filtering.Deformable(), CGEF.Filtering.ZeroFill())
        outf_m = zeros(Nx, Ny)
        CGEF.Filtering.filter_field!(outf_m, u, smgrid, g, ℓ; method = CGEF.Filtering.Spectral(), mask_strategy = strat)
        outu_m = zeros(Nx*Ny)
        CGEF.Filtering.filter_field!(outu_m, vec(u), umgrid, g, ℓ; method = CGEF.Filtering.Spectral(), mask_strategy = strat)
        Test.@test reshape(outu_m, Nx, Ny) ≈ outf_m atol = 1e-6
    end
end


# The NUFFT mode count must come from the POINT COUNT, not from an extent divided by a spacing.
# Scattered data has no spacing to divide by, and a plan that reaches for one produces a mode count
# unrelated to the problem size: deriving ~7000 m of extent from a nominal 1 m gave a
# ~49-million-mode transform, 4 GiB for 64 points. The gate is the mode grid the plan actually
# builds — an integer it already stores, so the assertion is exact and load-independent.
Test.@testset "Spectral FINUFFT filtering: mode count follows the point count" begin
    geom = FG.Geometry.CartesianGeometry()
    Nx, Ny = 8, 8
    dx_real = 1000.0
    x = collect(0.0:dx_real:dx_real*(Nx-1)); y = collect(0.0:dx_real:dx_real*(Ny-1))
    ptsx = vec([xi for xi in x, _ in y]); ptsy = vec([yj for _ in x, yj in y])
    ug = FG.Grids.UnstructuredGrid(geom, ptsx, ptsy, trues(Nx*Ny); k = 4)
    g = CGEF.GaussianKernel(); ℓ = 3000.0

    # `Mx*My ~ npts` by construction, up to rounding each axis up to even; the factor of 4 covers that
    # rounding at any aspect ratio while still being three orders of magnitude below a regression.
    plan = CGEF.Filtering.plan_filter(ug, g, ℓ; method = CGEF.Filtering.Spectral())
    Test.@test prod(size(plan.transfer)) <= 4 * Nx * Ny

    out = zeros(Nx*Ny)
    CGEF.Filtering.filter_field!(out, fill(3.7, Nx*Ny), ug, g, ℓ; method = CGEF.Filtering.Spectral())  # warm up
    b = @allocated CGEF.Filtering.filter_field!(out, fill(3.7, Nx*Ny), ug, g, ℓ; method = CGEF.Filtering.Spectral())
    Test.@test b < 10_000_000
    Test.@test all(x -> isapprox(x, 3.7; atol = 1e-6), out)

    Π = zeros(Nx*Ny)
    u = vec([sin(xi/700)*cos(yj/900) for xi in x, yj in y]); v = vec([cos(xi/500)*sin(yj/1100) for xi in x, yj in y])
    CGEF.Diagnostics.compute_Π!(Π, u, v, nothing, ug, g, ℓ)  # warm up (compile) before measuring
    b_pi = @allocated CGEF.Diagnostics.compute_Π!(Π, u, v, nothing, ug, g, ℓ)
    Test.@test b_pi < 10_000_000
    Test.@test all(isfinite, Π)
end


# Uniform-spherical spectral filtering (spherical harmonics): a single degree-l harmonic is an
# exact eigenfunction, scaled by Ĝ(k_l) with k_l = √(l(l+1))/R.
Test.@testset "Spectral spherical-harmonic filtering" begin
    N = 24; M = 2N - 1
    Θ, Φ = FSH.sph_points(N)
    R = 1.0
    geom = FG.Geometry.SphericalGeometry(R)
    grid = FG.Grids.StructuredGrid(geom, collect(Φ), π/2 .- collect(Θ), trues(M, N))
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

    # Masked spectral filtering (normalized convolution, Knutsson & Westin 1993): both
    # mask_strategy branches must reproduce RealSpace on the SAME masked grid, over ACTIVE cells
    # only (RealSpace leaves a masked cell at its untouched `fill!(out,0)` initial value, by
    # convention, while a global spectral transform computes a real value everywhere — the two are
    # expected to differ at masked points, by design). A modest ℓ (well inside the grid's resolved
    # range, unlike the ℓ = R used for the eigenfunction checks above) keeps this focused on
    # validating the masking logic itself; the ~3% active-cell relative error matches the same N=24
    # RealSpace discretization error already measured (and shown convergent under refinement) for
    # the unmasked case elsewhere in this file, not a masking-specific slack. `rtol`, not `atol`: `≈`
    # on arrays compares the aggregate 2-norm of the difference, and this active-cell selection is
    # ~1100 elements wide, so an `atol` sized against a single point's expected error silently fails
    # once aggregated over that many elements.
    mask = trues(M, N)
    mask[5:8, 5:8] .= false
    mgrid = FG.Grids.StructuredGrid(geom, collect(Φ), π/2 .- collect(Θ), mask)
    ℓ_m = 0.2
    Cr = zeros(N, M); Cr[FSH.sph_mode(3, 1)] = 1.0; Cr[FSH.sph_mode(6, -2)] = 0.5
    rfield = permutedims(FSH.sph_evaluate(Cr))
    for strat in (CGEF.Filtering.Deformable(), CGEF.Filtering.ZeroFill())
        out_spec = zeros(M, N)
        CGEF.Filtering.filter_field!(out_spec, rfield, mgrid, ker, ℓ_m; method = CGEF.Filtering.Spectral(), mask_strategy = strat)
        out_ds = zeros(M, N)
        CGEF.Filtering.filter_field!(out_ds, rfield, mgrid, ker, ℓ_m; method = CGEF.Filtering.RealSpace(), mask_strategy = strat)
        Test.@test out_spec[mask] ≈ out_ds[mask] rtol = 0.06
    end

    # TopHatKernel spherical-cap filtering: exact eigenfunction scaling by the Legendre-based
    # Ĝ_l (spectral_transfer_degree) at the SAME ℓ=1 used above (a pure self-consistency check,
    # independent of any real-space discretization).
    th = CGEF.TopHatKernel()
    thout = zeros(M, N)
    CGEF.Filtering.filter_field!(thout, field, grid, th, ℓ; method = CGEF.Filtering.Spectral())
    Ghat_th_l = CGEF.Kernels.spectral_transfer_degree(th, l, ℓ, R)
    Test.@test thout ≈ Ghat_th_l .* field atol = 1e-10

    # Cross-check against an INDEPENDENT RealSpace spherical-cap average — the real validation:
    # an independently-derived formula agreeing with real-space averaging, not just
    # self-consistency with itself. A smaller ℓ than the eigenfunction check above (a huge ℓ=R
    # cap makes the top-hat's sharp edge a large fraction of the sphere, and RealSpace's
    # discretization error on this N=24 grid genuinely does NOT shrink to below a few % at that
    # scale — confirmed convergent, not a formula bug, by rerunning at N=12/24/48/96 with this
    # ℓ: relative error shrinks ~9%→7%→2%→0.7%, the expected O(1/N) rate for a discontinuous
    # kernel). At this more realistic filter-scale/grid-spacing ratio the N=24 discretization
    # error is small enough for a tight-but-real tolerance — `rtol` (measured ~4.6% relative,
    # 2-norm), not `atol`, for the same aggregate-vs-single-point reason noted above.
    ℓ_th = 0.3
    thout2 = zeros(M, N)
    CGEF.Filtering.filter_field!(thout2, field, grid, th, ℓ_th; method = CGEF.Filtering.Spectral())
    thout2_ds = zeros(M, N)
    CGEF.Filtering.filter_field!(thout2_ds, field, grid, th, ℓ_th; method = CGEF.Filtering.RealSpace())
    Test.@test thout2 ≈ thout2_ds rtol = 0.07

    # Grid that is not an FSH grid (M ≠ 2N-1) is rejected.
    badgrid = FG.Grids.StructuredGrid(geom, collect(0.0:0.1:1.0), collect(0.0:0.1:1.0), trues(11, 11))
    Test.@test_throws ArgumentError CGEF.Filtering.filter_field!(zeros(11, 11), rand(11, 11), badgrid, ker, ℓ; method = CGEF.Filtering.Spectral())

    # Shape-correct (M = 2N-1) but NOT on the actual FSH quadrature nodes: must still be rejected,
    # not silently accepted and given a meaningless transform.
    wronggrid = FG.Grids.StructuredGrid(
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
    geom = FG.Geometry.SphericalGeometry(R)
    lat = vec([π/2 - θ for θ in Θ, φ in Φ])
    lon = vec([φ for θ in Θ, φ in Φ])
    npts = length(lat)
    ug = FG.Grids.UnstructuredGrid(geom, lon, lat, ones(npts), trues(npts))

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

    # TopHatKernel spherical-cap filtering: same Legendre-based Ĝ_l as FastSphericalHarmonics
    # (shared `spectral_transfer_degree`), on the SAME Clenshaw–Curtis point set treated as scattered.
    th = CGEF.TopHatKernel()
    thout = zeros(npts)
    CGEF.Filtering.filter_field!(thout, field, ug, th, scale; method = CGEF.Filtering.Spectral())
    Ghat_th_l = CGEF.Kernels.spectral_transfer_degree(th, l, scale, R)
    Test.@test thout ≈ Ghat_th_l .* field atol = 1e-6
end


# Cumulative coarse KE (Sadek-Aluie Eq.15) vs the filtering spectral density (Eq.14)
Test.@testset "Filtering spectrum" begin
    geom = FG.Geometry.CartesianGeometry()
    x = 0.0:2000.0:100e3
    y = 0.0:2000.0:100e3
    grid = FG.Grids.StructuredGrid(geom, x, y, trues(length(x), length(y)))

    U = 0.5  # m/s
    V = 0.3  # m/s
    u = fill(U, length(x), length(y))
    v = fill(V, length(x), length(y))
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
    # k_ℓ-derivative, Eq. 14) must be ≈ 0 everywhere — NOT equal to the energy. The cumulative
    # quantity above is defined for any kernel; the density needs one with a monotone `|Ĝ|²`, and a
    # constant field is reproduced by either, so the assertion is unchanged by the switch.
    kℓ, Ẽ = CGEF.Diagnostics.filtering_spectrum(u, v, nothing, grid, CGEF.GaussianKernel(), scales; L=1.0)
    Test.@test length(kℓ) == length(scales)
    Test.@test all(abs.(Ẽ) .< 1e-6 * expected_energy)
    Test.@test all(E -> isapprox(E, expected_energy; rtol = 1e-6),
                   CGEF.Diagnostics.cumulative_energy(u, v, nothing, grid, CGEF.GaussianKernel(), scales))

    # spectral_density reproduces a known derivative: C(k)=k² ⇒ dC/dk = 2k (central differences
    # are exact for a quadratic on a uniform grid).
    kk = collect(1.0:1.0:5.0)
    Test.@test CGEF.Diagnostics.spectral_density(kk .^ 2, kk)[3] ≈ 2 * kk[3]
end

# The padded-FFT engine computes the LINEAR convolution, so it is the real-space answer on a bounded,
# masked grid — the configuration a periodic transform cannot serve. Reference is a direct disk sum.
Test.@testset "Padded-FFT real-space engine" begin
    geom = FG.Geometry.CartesianGeometry()
    N = 64
    dx = 1000.0
    x = 0.0:dx:dx*(N - 1)                      # bounded
    m = trues(N, N); m[15:25, 15:25] .= false
    grid = FG.Grids.StructuredGrid(geom, x, x, m)
    u = randn(N, N)
    ker = CGEF.SharpSpectralKernel()           # non-separable: no factored engine exists for it
    sc = 2000.0
    rad = CGEF.Kernels.kernel_radius(ker, sc)
    w = ceil(Int, rad / dx)

    # `RealSpace` must stay the direct sum, whatever extensions are loaded.
    p_rs = CGEF.Filtering.plan_filter(grid, ker, sc;
        backend = CGEF.ComputationalBackends.SerialBackend(), method = CGEF.Filtering.RealSpace())
    Test.@test p_rs.footprint isa CGEF.Filtering.FilterFootprint

    for (strat, zerofill) in ((CGEF.Filtering.Deformable(), false), (CGEF.Filtering.ZeroFill(), true))
        ref = zeros(N, N)
        for j in 1:N, i in 1:N
            m[i, j] || continue
            ws = 0.0; wn = 0.0
            for dj in -w:w, di in -w:w
                ii = i + di; jj = j + dj
                (1 <= ii <= N && 1 <= jj <= N) || continue
                d = hypot(di * dx, dj * dx)
                d <= rad || continue
                wt = CGEF.Kernels.kernel_weight(ker, d, sc)
                if zerofill
                    wn += wt; m[ii, jj] && (ws += wt * u[ii, jj])
                else
                    m[ii, jj] || continue
                    wn += wt; ws += wt * u[ii, jj]
                end
            end
            ref[i, j] = wn > 1e-15 ? ws / wn : 0.0
        end
        p = CGEF.Filtering.plan_filter(grid, ker, sc;
            backend = CGEF.ComputationalBackends.SerialBackend(),
            method = CGEF.Filtering.AutoMethod(), mask_strategy = strat)
        got = zeros(N, N)
        CGEF.Filtering.filter_apply!(got, u, p)
        Test.@test maximum(abs, got .- ref) / maximum(abs, ref) < 1e-9
        # Applying a strategy the denominator was not built for must error, not renormalize wrongly.
        other = zerofill ? CGEF.Filtering.Deformable() : CGEF.Filtering.ZeroFill()
        Test.@test_throws ArgumentError CGEF.Filtering.apply_footprint!(
            zeros(N, N), u, grid, p.footprint, other)
    end

    # AutoMethod picks on real capability: a transform only where it is exact for the grid.
    let xp = range(0.0, dx * N; length = N + 1)[1:N],
        gper = FG.Grids.StructuredGrid(geom, xp, xp; periodic = (true, true)),
        gbnd = FG.Grids.StructuredGrid(geom, x, x)
        Test.@test CGEF.Filtering._resolve_method(gper, CGEF.GaussianKernel(), CGEF.Filtering.AutoMethod()) isa CGEF.Filtering.Spectral
        Test.@test CGEF.Filtering._resolve_method(gbnd, CGEF.GaussianKernel(), CGEF.Filtering.AutoMethod()) isa CGEF.Filtering.RealSpace
        Test.@test CGEF.Filtering._resolve_method(gbnd, CGEF.GaussianKernel(), CGEF.Filtering.RealSpace()) isa CGEF.Filtering.RealSpace
    end
end
