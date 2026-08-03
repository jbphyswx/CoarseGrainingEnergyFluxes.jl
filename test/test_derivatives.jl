
# spatial finite differences and boundary stencil fallbacks
Test.@testset "Derivatives" begin
    geom = FG.Geometry.CartesianGeometry()
    x = collect(0.0:2.0:10.0) # 6 points
    y = collect(0.0:2.0:10.0) # 6 points
    mask = trues(6, 6)
    grid = FG.Grids.StructuredGrid(geom, x, y, mask)

    # Test horizontal derivatives of f(x) = 3x + 1
    # ∂f/∂x should be exactly 3.0 at all active (unmasked) cells
    f = zeros(6, 6)
    for j in 1:6, i in 1:6
        f[i, j] = 3.0 * grid.x[i] + 1.0
    end

    ∂f∂x = zeros(6, 6)
    CGEF.Derivatives.ddx!(∂f∂x, f, grid)

    Test.@test ∂f∂x[2, 3] ≈ 3.0
    Test.@test ∂f∂x[1, 3] ≈ 3.0 # forward difference at boundary
    Test.@test ∂f∂x[6, 3] ≈ 3.0 # backward difference at boundary
end


# Real convergence-RATE tests: refine resolution 3x (not just "error is small at one
# resolution") and assert the observed order matches the claimed 2nd-order accuracy of the
# centered nonuniform stencil (`nonuniform_first_derivative`), for a field whose third
# derivative is genuinely nonzero (sin, unlike the quadratic-exactness checks above, which are
# exact at ANY resolution and so can't demonstrate a convergence RATE at all).
Test.@testset "Convergence rate: ddx! is genuinely 2nd order" begin
    geom = FG.Geometry.CartesianGeometry()  # dx/dy unused; area/spacing come from the axis
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
        grid = FG.Grids.StructuredGrid(geom, x, y, trues(N, length(y)))
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
