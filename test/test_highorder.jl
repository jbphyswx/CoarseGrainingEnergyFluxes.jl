# `HighOrderKernel` — the Sadek & Aluie (2018) §V `M^I` (p = 3) and `M^II` (p = 5) kernels. Three things
# make it structurally different from every other kernel here and are what these tests gate: it is
# SEPARABLE rather than radial, its weights are SIGNED, and it is DISCONTINUOUS with limbs only `b`
# wide, so it has to be integrated over each cell rather than sampled at the node.

# Exact 1-D moments of a piecewise-constant even kernel, in rational arithmetic:
#   ∫ xⁿ G dx = 2 Σ_k v_k (r_k^(n+1) - r_(k-1)^(n+1)) / (n+1)   for even n; odd n vanish by symmetry.
function _cgef_exact_moment(edges::Vector{Rational{BigInt}}, vals::Vector{Rational{BigInt}}, n::Int)
    isodd(n) && return zero(Rational{BigInt})
    m = zero(Rational{BigInt}); prev = zero(Rational{BigInt})
    for (r, v) in zip(edges, vals)
        m += v * (r^(n + 1) - prev^(n + 1)) // (n + 1)
        prev = r
    end
    return 2m
end

# The kernel's pieces re-derived in exact rationals, independent of the float implementation.
# `BigInt` throughout: the order-5 coefficients have sixth powers of `β` in them, which overflow an
# `Int64` rational at even modest denominators.
function _cgef_exact_pieces(order::Int, β_in::Rational)
    β = Rational{BigInt}(β_in)
    half = Rational{BigInt}(1, 2)
    if order == 3
        t = 1 + 2β
        return (Rational{BigInt}[half, half + β], Rational{BigInt}[1, -1 // (t^3 - 1)])
    else
        den = 4β^2 * (192β^4 + 400β^3 + 340β^2 + 120β + 15)
        return (Rational{BigInt}[half, half + β, half + 2β],
                Rational{BigInt}[1,
                                 -(124β^3 + 88β^2 + 19β + 1) // den,
                                 (4β^3 + 8β^2 + 5β + 1) // den])
    end
end

Test.@testset "HighOrderKernel: exact coefficients and vanishing moments" begin
    # The published reduced values at the paper's b = ℓ/8, reproduced exactly — not to a tolerance.
    for (order, want) in ((3, (-64 // 61,)), (5, (-568 // 257, 200 // 257)))
        _, vals = _cgef_exact_pieces(order, 1 // 8)
        Test.@test Tuple(vals[2:end]) == want
        # ... and the Float64 implementation lands on those rationals to the last bit.
        got = CGEF.Kernels.limb_amplitudes(
            CGEF.Kernels.HighOrderKernel(; order = order, b_over_ℓ = 1 / 8), Float64)
        Test.@test all(got .== Float64.(want))
    end

    # `p` vanishing moments, exactly, and the FIRST surviving moment matches the published value —
    # so the order is pinned from both sides rather than just "the low ones are small".
    for (order, m_next) in ((3, -5 // 256), (5, 225 // 28672))
        edges, vals = _cgef_exact_pieces(order, 1 // 8)
        m0 = _cgef_exact_moment(edges, vals, 0)
        Test.@test m0 > 0
        for n in 1:order
            Test.@test _cgef_exact_moment(edges, vals, n) == 0
        end
        Test.@test _cgef_exact_moment(edges, vals, order + 1) // m0 == m_next
    end

    # The moment conditions are solved for GENERAL `b`, not just the paper's 1/8 — so a user changing
    # `b_over_ℓ` still gets a high-order kernel rather than a silently broken one.
    for order in (3, 5), β in (1 // 12, 1 // 6, 1 // 4)
        edges, vals = _cgef_exact_pieces(order, β)
        Test.@test _cgef_exact_moment(edges, vals, 0) > 0
        for n in 1:order
            Test.@test _cgef_exact_moment(edges, vals, n) == 0
        end
        # The float implementation agrees with the exact solve at this `b` too.
        got = CGEF.Kernels.limb_amplitudes(
            CGEF.Kernels.HighOrderKernel(; order = order, b_over_ℓ = float(β)), Float64)
        Test.@test all(isapprox.(got, Float64.(vals[2:end]); rtol = 1e-14))
    end

    # Constructor guards: only the two orders the construction exists for, and a positive limb width.
    for bad in (1, 2, 4, 6, 7)
        Test.@test_throws ArgumentError CGEF.Kernels.HighOrderKernel(; order = bad)
    end
    Test.@test_throws ArgumentError CGEF.Kernels.HighOrderKernel(; b_over_ℓ = 0.0)
    Test.@test_throws ArgumentError CGEF.Kernels.HighOrderKernel(; b_over_ℓ = -0.1)
end

Test.@testset "HighOrderKernel: separable, not radial" begin
    for order in (3, 5)
        k = CGEF.Kernels.HighOrderKernel(; order = order)
        Test.@test CGEF.Kernels.is_separable(k)
        Test.@test !CGEF.Kernels.is_radial(k)
        Test.@test !CGEF.Kernels.transfer_monotone(k)
        # There is no radial form, and asking for one says so rather than returning a wrong number.
        err = try
            CGEF.Kernels.kernel_weight(k, 0.1, 1.0); nothing
        catch e; e end
        Test.@test err isa ArgumentError
        Test.@test occursin("SEPARABLE", err.msg)
        # Nor an isotropic transfer function.
        Test.@test_throws ArgumentError CGEF.Kernels.spectral_transfer(k, 1.0, 1.0)
        Test.@test_throws ArgumentError CGEF.Kernels.spectral_transfer_degree(k, 3, 1.0, 6.371e6)
        # Support is body + one limb per moment pair, per axis.
        nlimb = order == 3 ? 1 : 2
        Test.@test CGEF.Kernels.kernel_radius(k, 1.0) ≈ 0.5 + nlimb * k.b_over_ℓ
        # The profile is exactly the piecewise-constant shape, sampled away from the discontinuities.
        amps = CGEF.Kernels.limb_amplitudes(k, Float64)
        Test.@test CGEF.Kernels.kernel_profile(k, 0.1, 1.0) == 1.0
        Test.@test CGEF.Kernels.kernel_profile(k, -0.1, 1.0) == 1.0          # even
        Test.@test CGEF.Kernels.kernel_profile(k, 0.5625, 1.0) == amps[1]    # mid first limb
        Test.@test CGEF.Kernels.kernel_profile(k, 0.99, 1.0) == 0.0          # past support
        if order == 5
            Test.@test CGEF.Kernels.kernel_profile(k, 0.6875, 1.0) == amps[2]
        end
    end

    # The other kernels keep their traits: the Gaussian is both, the rest are radial only.
    Test.@test CGEF.Kernels.is_separable(CGEF.GaussianKernel())
    Test.@test CGEF.Kernels.is_radial(CGEF.GaussianKernel())
    for k in (CGEF.TopHatKernel(), CGEF.SharpSpectralKernel(),
              CGEF.Kernels.SmoothHatKernel(), CGEF.Kernels.HyperGaussianKernel())
        Test.@test !CGEF.Kernels.is_separable(k)
        Test.@test CGEF.Kernels.is_radial(k)
    end
end

Test.@testset "HighOrderKernel: cell-averaged weights are what make it work off a uniform grid" begin
    # `profile_integral` is the exact antiderivative, so a cell average is exact for this kernel.
    for order in (3, 5)
        k = CGEF.Kernels.HighOrderKernel(; order = order)
        Test.@test CGEF.Kernels.profile_integral(k, 0.0, 1.0) == 0.0
        Test.@test CGEF.Kernels.profile_integral(k, -0.3, 1.0) == -CGEF.Kernels.profile_integral(k, 0.3, 1.0)
        # Tied EXACTLY to the rational moments computed above, which come from a different derivation:
        # twice the integral over the whole support is `m0 = ∫G dx`.
        edges, vals = _cgef_exact_pieces(order, 1 // 8)
        m0 = Float64(_cgef_exact_moment(edges, vals, 0))
        R = CGEF.Kernels.kernel_radius(k, 1.0)
        Test.@test 2 * CGEF.Kernels.profile_integral(k, R, 1.0) ≈ m0 rtol = 1e-14
        Test.@test CGEF.Kernels.profile_integral(k, 2R, 1.0) ==
                   CGEF.Kernels.profile_integral(k, R, 1.0)      # flat past the support

        # And against an independent numerical integral. The integrand is DISCONTINUOUS, so midpoint
        # quadrature is only accurate to `O(dr)` times the jump it straddles, and that error does not
        # fall monotonically with `n` — it depends on where each breakpoint lands inside a cell. So the
        # comparison is against that bound, not against a convergence rate.
        jumps = sum(abs, diff([Float64.(vals); 0.0]))
        for t in (0.2, 0.49, 0.55, 0.7, 1.0)
            n = 200_000; dr = t / n
            q = sum(CGEF.Kernels.kernel_profile(k, (i - 0.5) * dr, 1.0) for i in 1:n) * dr
            Test.@test abs(CGEF.Kernels.profile_integral(k, t, 1.0) - q) <= jumps * dr
        end
        # A cell average IS the integral over the cell divided by its width.
        for (δ, Δ) in ((0.0, 0.1), (0.48, 0.1), (0.6, 0.05), (0.9, 0.2))
            want = (CGEF.Kernels.profile_integral(k, δ + Δ / 2, 1.0) -
                    CGEF.Kernels.profile_integral(k, δ - Δ / 2, 1.0)) / Δ
            Test.@test CGEF.Kernels.profile_cell_average(k, δ, Δ, 1.0) == want
        end
    end

    # For every SMOOTH kernel the cell average is deliberately left as the point sample, so nothing
    # else in the package changes behaviour.
    for k in (CGEF.GaussianKernel(), CGEF.GaussianKernel(; α = 4))
        for δ in (-0.7, 0.0, 0.3), Δ in (0.05, 0.2)
            Test.@test CGEF.Kernels.profile_cell_average(k, δ, Δ, 1.0) ==
                       CGEF.Kernels.kernel_profile(k, δ, 1.0)
        end
    end

    # The payoff. Point-sampling a 1-cell-wide signed limb on a NON-UNIFORM axis can miss it, driving
    # the ZeroFill denominator negative and returning a sign-flipped field. Cell-averaging cannot:
    # the denominator is a true integral of the kernel over the covered region. Both are asserted, so
    # a regression to point sampling fails here rather than silently.
    let geom = FG.Geometry.CartesianGeometry(), N = 48, dx = 1000.0, ℓ = 8 * dx,
        xs = 0.0:dx:(N - 1) * dx,
        xv = collect(xs) .+ [30.0 * sin(2.3i) for i in 1:N]        # ±3% jitter
        for order in (3, 5), (nm, ax) in (("uniform", xs), ("stretched", xv))
            g = FG.Grids.StructuredGrid(geom, ax, ax, trues(N, N))
            fp = CGEF.Filtering.plan_filter(g, CGEF.Kernels.HighOrderKernel(; order = order), ℓ).footprint
            Test.@test fp isa CGEF.Filtering.SeparableFootprint
            # The normalization denominator is strictly positive everywhere — the property that fails
            # under point sampling.
            Test.@test all(>(0), fp.Nx_profile)
            Test.@test all(>(0), fp.Ny_profile)
            # ... so a constant comes back exactly, on a stretched axis as much as a uniform one.
            out = zeros(N, N)
            CGEF.Filtering.filter_field!(out, fill(2.5, N, N), g,
                                         CGEF.Kernels.HighOrderKernel(; order = order), ℓ)
            Test.@test all(≈(2.5; rtol = 1e-12), out)
        end
    end
end

Test.@testset "HighOrderKernel: routing, product structure, and the resolution warning" begin
    geom = FG.Geometry.CartesianGeometry()
    N = 48; dx = 1000.0; ℓ = 8 * dx
    xs = 0.0:dx:(N - 1) * dx
    grid = FG.Grids.StructuredGrid(geom, xs, xs, trues(N, N))

    # Every rank with axes takes the separable engine — there is no other engine it could take.
    for order in (3, 5)
        k = CGEF.Kernels.HighOrderKernel(; order = order)
        Test.@test CGEF.Filtering.plan_filter(grid, k, ℓ).footprint isa CGEF.Filtering.SeparableFootprint
        g1 = FG.Grids.StructuredGrid(geom, xs, trues(N))
        Test.@test CGEF.Filtering.plan_filter(g1, k, ℓ).footprint isa CGEF.Filtering.SeparableFootprintND
        g3 = FG.Grids.StructuredGrid(geom, 0.0:dx:15dx, 0.0:dx:15dx, 0.0:dx:15dx, trues(16, 16, 16))
        Test.@test CGEF.Filtering.plan_filter(g3, k, ℓ).footprint isa CGEF.Filtering.SeparableFootprintND
    end

    # The 2-D response really is the PRODUCT of 1-D profiles — a square footprint, not a disk. Checked
    # against the impulse response, so it tests the engine and not just the kernel definition.
    for order in (3, 5)
        k = CGEF.Kernels.HighOrderKernel(; order = order)
        plan = CGEF.Filtering.plan_filter(grid, k, ℓ)
        δ = zeros(N, N); δ[24, 24] = 1.0
        r = zeros(N, N)
        CGEF.Filtering.filter_apply!(r, δ, plan)
        p1(t) = CGEF.Kernels.profile_cell_average(k, t, dx, ℓ)
        scale = r[24, 24] / p1(0.0)^2
        worst = 0.0
        for dj in -10:10, di in -10:10
            worst = max(worst, abs(r[24 + di, 24 + dj] - scale * p1(di * dx) * p1(dj * dx)))
        end
        Test.@test worst < 1e-15 * abs(r[24, 24])
        # The corners of the square carry real weight — a radial kernel would have zero there.
        Test.@test abs(r[24 + 3, 24 + 3]) > 0.1 * abs(r[24, 24])
    end

    # A grid with no axes to factor over is refused, not approximated by a radial stand-in.
    let nn = 16,
        xg = [Float64(i - 1) * dx for i in 1:nn, j in 1:nn],
        yg = [Float64(j - 1) * dx for i in 1:nn, j in 1:nn]
        cg = FG.Grids.CurvilinearGrid(geom, xg, yg, trues(nn, nn))
        Test.@test_throws ArgumentError CGEF.Filtering.plan_filter(
            cg, CGEF.Kernels.HighOrderKernel(), ℓ; method = CGEF.Filtering.RealSpace())
    end

    # Under-resolved limbs warn and name the ℓ that would fix it: cell-averaging keeps the MASS right
    # but cannot invent the resolution the vanishing moments need.
    Test.@test_logs (:warn, r"vanishing moments") match_mode = :any CGEF.Filtering.plan_filter(
        grid, CGEF.Kernels.HighOrderKernel(), 4 * dx)
    # At the documented threshold there is no warning.
    Test.@test_logs min_level = Logging.Warn CGEF.Filtering.plan_filter(
        grid, CGEF.Kernels.HighOrderKernel(), 8 * dx)

    # It cannot produce a filtering spectrum either, for the same reason as the other ringing kernels.
    u = [sin(i / 5) * cos(j / 7) for i in 1:N, j in 1:N]
    v = [cos(i / 6) * sin(j / 4) for i in 1:N, j in 1:N]
    Test.@test_throws ArgumentError CGEF.Diagnostics.filtering_spectrum(
        u, v, nothing, grid, CGEF.Kernels.HighOrderKernel(), [ℓ, 2ℓ])
    # ... but it is a perfectly good real-space filter, and Π runs.
    Π = zeros(N, N)
    CGEF.Diagnostics.compute_Π!(Π, u, v, nothing, grid, CGEF.Kernels.HighOrderKernel(), ℓ)
    Test.@test all(isfinite, Π)
    Test.@test maximum(abs, Π) > 0
end
