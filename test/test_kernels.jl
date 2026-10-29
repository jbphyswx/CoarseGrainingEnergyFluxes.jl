
# Kernel facts against closed forms: weights and radii, continuum moments, transfer functions, and the
# `HighOrderKernel`'s exact construction.

# `∫G` and `⟨r²⟩` over the support at `ℓ = 1`, by midpoint quadrature in `r`.
function _cgef_continuum_moments(kern, D::Int; n = 200_000)
    R = CGEF.Kernels.kernel_radius(kern, 1.0, Val(D))
    dr = R / n
    m0 = 0.0
    m2 = 0.0
    for i in 1:n
        r = (i - 0.5) * dr
        w = CGEF.Kernels.kernel_weight(kern, r, 1.0, Val(D)) * (D == 1 ? 2.0 : D == 2 ? 2π * r : 4π * r^2)
        m0 += w * dr
        m2 += w * r^2 * dr
    end
    return m0, m2 / m0
end

Test.@testset "Kernel weights and per-component second moments" begin
    # The top-hat's profile is `CellCentre`'s weight; `CellOverlap`'s depends on a cell's shape, and a node
    # set's points take the centre test.
    th = CGEF.TopHatKernel(; cells = CGEF.Kernels.CellCentre())
    Test.@test CGEF.Kernels.point_kernel(CGEF.TopHatKernel()) === th
    Test.@test_throws MethodError CGEF.Kernels.kernel_weight(CGEF.TopHatKernel(), 0.1, 1.0)
    ℓ = 1.0
    Test.@test CGEF.Kernels.kernel_weight(th, 0.49ℓ, ℓ) == 1.0
    Test.@test CGEF.Kernels.kernel_weight(th, 0.51ℓ, ℓ) == 0.0
    Test.@test CGEF.Kernels.kernel_radius(th, ℓ) == ℓ / 2
    for α in (4.0, 6.0)
        Test.@test CGEF.Kernels.kernel_weight(CGEF.GaussianKernel(; α), ℓ, ℓ) ≈ exp(-α)
    end

    # `⟨x²⟩ = ⟨r²⟩/D`: the disk/ball top-hat of radius ℓ/2 gives ℓ²/12, ℓ²/16, ℓ²/20 in 1, 2, 3 dimensions,
    # the Gaussian ℓ²/(2α) in every dimension, and the hyper-Gaussian `exp(−16α r⁴/ℓ⁴)` the ratios of
    # `∫ rⁿ e^{−ar⁴} dr = Γ((n+1)/4) / (4 a^((n+1)/4))`, with `a^(−1/2) = ℓ²/(4√α)`.
    Γ = SpecialFunctions.gamma
    hyper(α, D) = (s = 1 / (4 * sqrt(α));
                   D == 1 ? Γ(3 / 4) / Γ(1 / 4) * s : D == 2 ? s / sqrt(π) / 2 : Γ(5 / 4) / Γ(3 / 4) * s / 3)
    cases = vcat(
        [(th, D, 1 / (4 * (D + 2))) for D in 1:3],
        [(CGEF.GaussianKernel(; α), D, 1 / (2α)) for α in (4.0, 6.0, 10.0) for D in 1:3],
        [(CGEF.Kernels.HyperGaussianKernel(; α), D, hyper(α, D)) for α in (1.0, 2.5) for D in 1:3],
    )
    for (kern, D, x2) in cases
        _, r2 = _cgef_continuum_moments(kern, D)
        Test.@test r2 / D ≈ x2 rtol = 1e-4
    end

    # A smooth kernel's footprint ends where its weight falls to `truncation_tol(T)` of its peak, `1e-10` or
    # `8 eps(T)` where the element type resolves less, to the rounding of the radius's inverse function there.
    Test.@test CGEF.Kernels.truncation_tol(Float64) == 1e-10 && CGEF.Kernels.truncation_tol(Float32) == 8eps(Float32)
    for T in (Float32, Float64), kern in (CGEF.GaussianKernel(), CGEF.Kernels.SmoothHatKernel(), CGEF.Kernels.HyperGaussianKernel())
        tol = CGEF.Kernels.truncation_tol(T)
        R = CGEF.Kernels.kernel_radius(kern, one(T))
        Test.@test R isa T && isfinite(R)
        Test.@test tol / 2 < CGEF.Kernels.kernel_weight(kern, R, one(T)) / CGEF.Kernels.kernel_weight(kern, zero(T), one(T)) < 2tol
    end

    # Π is a pointwise transfer only for a non-negative kernel; the brick wall's inverse transform is
    # signed in every dimension.
    for kern in (th, CGEF.GaussianKernel(), CGEF.Kernels.SmoothHatKernel(), CGEF.Kernels.HyperGaussianKernel())
        R = CGEF.Kernels.kernel_radius(kern, 1.0)
        Test.@test all(r -> CGEF.Kernels.kernel_weight(kern, r, 1.0) >= 0, range(0.0, R; length = 2001))
        Test.@test CGEF.Kernels.is_nonnegative(kern)
    end
    for D in 1:3
        Test.@test any(r -> CGEF.Kernels.kernel_weight(CGEF.SharpSpectralKernel(), r, 1.0, Val(D)) < 0,
                       range(0.0, 10.0; length = 2001))
    end
    Test.@test !CGEF.Kernels.is_nonnegative(CGEF.SharpSpectralKernel())
end

# `SharpSpectralKernel` in real space is the inverse transform of its brick wall in the grid's own
# dimension, truncated where its mass inside the radius equals its mass over all space.
Test.@testset "SharpSpectralKernel: the brick wall's inverse transform in D dimensions" begin
    ss = CGEF.SharpSpectralKernel()
    ℓ = 3.0
    kc = π / ℓ
    forms = (x -> sin(x) / x, x -> 2 * SpecialFunctions.besselj1(x) / x,
             x -> 3 * SpecialFunctions.sphericalbesselj(1, x) / x)
    # `(2/π)Si(x)`, `1 − J₀(x)` and `(2/π)(Si(x) − sin x)` at `x = k_c R`.
    Si = SpecialFunctions.sinint
    mass = (x -> (2 / π) * Si(x), x -> 1 - SpecialFunctions.besselj0(x), x -> (2 / π) * (Si(x) - sin(x)))
    for D in 1:3
        Test.@test CGEF.Kernels.kernel_weight(ss, 0.0, ℓ, Val(D)) == 1.0
        for x in (1e-6, 0.05, 0.19, 0.21, 1.0, 4.3, 17.2, 30.0)
            Test.@test CGEF.Kernels.kernel_weight(ss, x / kc, ℓ, Val(D)) ≈ forms[D](x) rtol = 1e-13
        end
        x = kc * CGEF.Kernels.kernel_radius(ss, ℓ, Val(D))
        Test.@test mass[D](x) ≈ 1 atol = 1e-14
    end
end

Test.@testset "Transfer functions and the monotonicity a filtering spectrum needs" begin
    ℓ = 1.0
    th = CGEF.TopHatKernel()
    for k in (0.3, 2.0, 7.0)
        for α in (4.0, 6.0)
            Test.@test CGEF.Kernels.spectral_transfer(CGEF.GaussianKernel(; α), k, ℓ) ≈ exp(-k^2 * ℓ^2 / (4α))
        end
        x = k * ℓ / 2
        Test.@test CGEF.Kernels.spectral_transfer(th, k, ℓ) ≈ 2 * SpecialFunctions.besselj1(x) / x
    end
    ss = CGEF.SharpSpectralKernel()
    Test.@test CGEF.Kernels.spectral_transfer(ss, π / ℓ * (1 - 1e-9), ℓ) == 1.0
    Test.@test CGEF.Kernels.spectral_transfer(ss, π / ℓ * (1 + 1e-9), ℓ) == 0.0

    # On a sphere of radius R the top-hat is the cap of angular radius ℓ/(2R), and degree l is scaled by
    # its mean of `P_l(cos θ)`.
    R = 4.0
    θ₀ = ℓ / (2R)
    for l in (0, 1, 3, 8, 20)
        Test.@test CGEF.Kernels.spectral_transfer_degree(th, l, ℓ, R) ≈
                   _cgef_simpson(θ -> _cgef_legendre(l, cos(θ)) * sin(θ), 0.0, θ₀) / (1 - cos(θ₀)) rtol = 1e-9 atol = 1e-12
    end

    # A filtering spectral density is non-negative where `|Ĝ|²` never rises (Sadek & Aluie 2018 eq. 21);
    # `transfer_monotone` states whether it does.
    ks = range(0.05, 12.0; length = 4000)
    for kern in (th, CGEF.GaussianKernel(), CGEF.GaussianKernel(; α = 4.0), ss)
        g2 = [CGEF.Kernels.spectral_transfer(kern, k, ℓ)^2 for k in ks]
        Test.@test CGEF.Kernels.transfer_monotone(kern) == all(<=(0), diff(g2))
    end
end

# Every radial kernel's degree transfer is its Legendre transform over the support the real-space engines
# keep (Aluie 2019, eq. 34), to high degree. In single precision the Gauss–Legendre nodes are rounded once
# from double and the Legendre recurrences run in `Float32`, whose error grows with the degree.
Test.@testset "Degree transfer on a sphere: the kernel's Legendre transform" begin
    R = 4.0
    ℓ = 1.0
    ls = (0, 1, 2, 7, 20, 40)
    L = last(ls)
    for kern in (CGEF.GaussianKernel(), CGEF.GaussianKernel(; α = 4.0), CGEF.Kernels.SmoothHatKernel(),
                 CGEF.Kernels.HyperGaussianKernel(), CGEF.SharpSpectralKernel(), CGEF.TopHatKernel())
        Ĝ = CGEF.Kernels.spectral_transfer_degrees!(zeros(L + 1), kern, ℓ, R)
        for l in ls
            Test.@test Ĝ[l + 1] ≈ _cgef_legendre_transfer(kern, l, ℓ, R) rtol = 1e-9 atol = 1e-12
        end
        Ĝ32 = CGEF.Kernels.spectral_transfer_degrees!(zeros(Float32, L + 1), kern, 1.0f0, 4.0f0)
        Test.@test maximum(abs, Ĝ32 .- Ĝ) < 10 * (L + 1) * eps(Float32)
    end
end

# ---- HighOrderKernel (Sadek & Aluie 2018 §V) --------------------------------------------------------

# `∫ xⁿ G dx = 2 Σ_k v_k (r_kⁿ⁺¹ − r_{k−1}ⁿ⁺¹)/(n+1)` for even `n` of a piecewise-constant even profile;
# odd moments vanish by symmetry.
function _cgef_exact_moment(edges::Vector{Rational{BigInt}}, vals::Vector{Rational{BigInt}}, n::Int)
    isodd(n) && return zero(Rational{BigInt})
    m = zero(Rational{BigInt})
    prev = zero(Rational{BigInt})
    for (r, v) in zip(edges, vals)
        m += v * (r^(n + 1) - prev^(n + 1)) // (n + 1)
        prev = r
    end
    return 2m
end

# The profile's edges and values at limb width `β = b/ℓ`, from the paper's moment conditions solved in
# rationals.
function _cgef_exact_pieces(order::Int, β_in::Rational)
    β = Rational{BigInt}(β_in)
    half = Rational{BigInt}(1, 2)
    if order == 3
        t = 1 + 2β
        return (Rational{BigInt}[half, half + β], Rational{BigInt}[1, -1 // (t^3 - 1)])
    end
    den = 4β^2 * (192β^4 + 400β^3 + 340β^2 + 120β + 15)
    return (Rational{BigInt}[half, half + β, half + 2β],
            Rational{BigInt}[1, -(124β^3 + 88β^2 + 19β + 1) // den, (4β^3 + 8β^2 + 5β + 1) // den])
end

# The body and limb radii and amplitudes, in units of `ℓ`.
_cgef_shells(k, amps) = ([0.5 + n * k.b_over_ℓ for n in 0:length(amps)], (1.0, amps...))

Test.@testset "HighOrderKernel: P vanishing moments, exactly" begin
    # The paper's values at b = ℓ/8, and the first surviving moment, in rationals.
    for (order, want, m_next) in ((3, (-64 // 61,), -5 // 256), (5, (-568 // 257, 200 // 257), 225 // 28672))
        edges, vals = _cgef_exact_pieces(order, 1 // 8)
        Test.@test Tuple(vals[2:end]) == want
        Test.@test all(CGEF.Kernels.limb_amplitudes(CGEF.Kernels.HighOrderKernel(; order), Float64) .== Float64.(want))
        m0 = _cgef_exact_moment(edges, vals, 0)
        Test.@test _cgef_exact_moment(edges, vals, order + 1) // m0 == m_next
        k = CGEF.Kernels.HighOrderKernel(; order)
        Test.@test 2 * CGEF.Kernels.profile_integral(k, CGEF.Kernels.kernel_radius(k, 1.0), 1.0) ≈ Float64(m0) rtol = 1e-14
    end
    # Any limb width.
    for order in (3, 5), β in (1 // 12, 1 // 8, 1 // 4)
        edges, vals = _cgef_exact_pieces(order, β)
        Test.@test _cgef_exact_moment(edges, vals, 0) > 0
        Test.@test all(n -> _cgef_exact_moment(edges, vals, n) == 0, 1:order)
        got = CGEF.Kernels.limb_amplitudes(CGEF.Kernels.HighOrderKernel(; order, b_over_ℓ = float(β)), Float64)
        Test.@test all(isapprox.(got, Float64.(vals[2:end]); rtol = 1e-14))
    end
end

# The isotropic form on the plane in D dimensions and on a sphere of radius R: `∫ r^(2m) G(r) w(r) dr`
# vanishes for `m = 1…L` with the radial measure `w`, `r` and `r²` in two and three dimensions and
# `sin(r/R)` on the sphere.
Test.@testset "HighOrderKernel: the isotropic form's moments vanish" begin
    for order in (3, 5), β in (1 / 12, 1 / 8, 1 / 4)
        k = CGEF.Kernels.HighOrderKernel(; order, b_over_ℓ = β)
        L = order == 3 ? 1 : 2
        Test.@test all(isapprox.(CGEF.Kernels.radial_limb_amplitudes(k, Float64, Val(1)),
                                 CGEF.Kernels.limb_amplitudes(k, Float64); rtol = 1e-13))
        for (amps, w) in ((CGEF.Kernels.radial_limb_amplitudes(k, Float64, Val(2)), r -> r),
                          (CGEF.Kernels.radial_limb_amplitudes(k, Float64, Val(3)), r -> r^2),
                          (CGEF.Kernels.spherical_limb_amplitudes(k, 1.0, 3.0), r -> sin(r / 3.0)))
            ρ, a = _cgef_shells(k, amps)
            mom(m) = sum(a[n + 1] * _cgef_simpson(r -> r^(2m) * w(r), n == 0 ? 0.0 : ρ[n], ρ[n + 1]) for n in 0:L)
            Test.@test mom(0) > 0
            Test.@test all(m -> abs(mom(m)) < 1e-10 * mom(0), 1:L)
        end
    end
end

Test.@testset "HighOrderKernel: transfer functions" begin
    for order in (3, 5)
        k = CGEF.Kernels.HighOrderKernel(; order)
        # On the plane, the product of each axis's cosine transform of the profile over its mass.
        ρ, a = _cgef_shells(k, CGEF.Kernels.limb_amplitudes(k, Float64))
        lo(n) = n == 0 ? 0.0 : ρ[n]
        Ĝ1(kd) = sum(a[n + 1] * _cgef_simpson(x -> cos(kd * x), lo(n), ρ[n + 1]) for n in eachindex(a) .- 1) /
                 sum(a[n + 1] * (ρ[n + 1] - lo(n)) for n in eachindex(a) .- 1)
        for kv in ((3.1, 0.0), (2.0, 5.5), (7.3, -4.2))
            Test.@test CGEF.Kernels.spectral_transfer(k, kv, 1.0) ≈ Ĝ1(kv[1]) * Ĝ1(kv[2]) rtol = 1e-9 atol = 1e-12
        end
        # On a sphere, the isotropic form against `P_l(cos θ)` over each shell.
        R = 4.0
        ρs, as = _cgef_shells(k, CGEF.Kernels.spherical_limb_amplitudes(k, 1.0, R))
        shell(f) = sum(as[n + 1] * _cgef_simpson(θ -> f(θ) * sin(θ), (n == 0 ? 0.0 : ρs[n]) / R, ρs[n + 1] / R)
                       for n in eachindex(as) .- 1)
        for l in (0, 1, 2, 5, 13)
            Test.@test CGEF.Kernels.spectral_transfer_degree(k, l, 1.0, R) ≈
                       shell(θ -> _cgef_legendre(l, cos(θ))) / shell(θ -> 1.0) rtol = 1e-9 atol = 1e-12
        end
    end
end
