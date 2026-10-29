
# The filtering spectrum `Ẽ(k_ℓ) = −(ℓ²/L) dĒ/dℓ` with `dĒ/dℓ = ⟨ū·∂_ℓū⟩` (Sadek & Aluie 2018 eq. 14): each kernel's
# rate in the filter width against a central difference of the kernel, each real-space plan's `ℓ∂_ℓū` against a
# central difference of its own output, and the density against closed forms.

_cgef_cd(f, ℓ, h) = (f(ℓ + h) .- f(ℓ - h)) ./ (2h)

Test.@testset "Kernel rates in the filter width against central differences" begin
    K = CGEF.Kernels
    HG = K.HyperGaussianKernel()
    ℓ = 0.37
    for k in (_CGEF_GA, _CGEF_SH, HG), d in (0.0, 0.1, 0.3, 0.5, 0.7, 1.2)
        Test.@test K.kernel_weight(K.ScaleRate(k), d, ℓ) ≈ ℓ * _cgef_cd(l -> K.kernel_weight(k, d, l), ℓ, 1e-7) atol = 1e-8
    end
    Test.@test all(k -> K.kernel_radius(K.ScaleRate(k), ℓ) == K.kernel_radius(k, ℓ), (_CGEF_GA, _CGEF_SH, HG))
    for D in 1:3, k in (_CGEF_GA, _CGEF_TH), km in (0.3, 4.0, 17.0, 40.0)
        Test.@test K.scale_rate_transfer(k, km, ℓ, Val(D)) ≈
                   _cgef_cd(l -> K.spectral_transfer(k, km, l, Val(D)), ℓ, 1e-7) atol = 1e-7
    end
    for P in (3, 5), kv in ((3.0, 1.0), (17.0, -4.0), (0.2, 25.0))
        k = K.HighOrderKernel{P}()
        Test.@test K.scale_rate_transfer(k, kv, ℓ) ≈ _cgef_cd(l -> K.spectral_transfer(k, kv, l), ℓ, 1e-7) atol = 1e-7
    end
    R, L = 6.371e6, 200
    for k in (_CGEF_GA, _CGEF_SH, HG, _CGEF_TH), ℓs in (3e5, 2e6)
        dG = K.spectral_rate_degrees!(zeros(L + 1), k, ℓs, R)
        Test.@test _cgef_relerr(dG, _cgef_cd(l -> K.spectral_transfer_degrees!(zeros(L + 1), k, l, R), ℓs, 1e-6ℓs)) < 1e-6
    end
    # The Gaussian's separable tables: its rate is the sum over axes of the product with that axis's rate profile.
    for δ in ((0.2, -0.1), (0.4, 0.3, -0.25))
        N = length(δ)
        axis(d) = prod(ntuple(e -> K.kernel_profile(K._axis_kernel(K.AxisRate(_CGEF_GA, Val(d)), Val(e)), δ[e], ℓ), Val(N)))
        Test.@test sum(ntuple(axis, Val(N))) ≈ K.kernel_weight(K.ScaleRate(_CGEF_GA), hypot(δ...), ℓ) rtol = 1e-13
    end

    # The second derivatives: `ScaleRate` of a `ScaleRate`, `TransferRate` of a `TransferRate` on the plane and by degree,
    # each against the central difference of the first; and the separable terms `J_d` and twice `H_d H_e` summing to the
    # radial `J`.
    for k in (_CGEF_GA, _CGEF_SH, HG), d in (0.0, 0.1, 0.3, 0.5, 0.7, 1.2)
        Test.@test K.kernel_weight(K.ScaleRate(K.ScaleRate(k)), d, ℓ) ≈
                   ℓ * _cgef_cd(l -> K.kernel_weight(K.ScaleRate(k), d, l), ℓ, 1e-7) atol = 1e-7
    end
    for D in 1:3, k in (_CGEF_GA, _CGEF_TH), km in (0.3, 4.0, 17.0, 40.0)
        Test.@test K.spectral_transfer(K.TransferRate(K.TransferRate(k)), km, ℓ, Val(D)) ≈
                   _cgef_cd(l -> K.scale_rate_transfer(k, km, l, Val(D)), ℓ, 1e-7) rtol = 1e-6 atol = 1e-5
    end
    for P in (3, 5), kv in ((3.0, 1.0), (17.0, -4.0), (0.2, 25.0))
        k = K.HighOrderKernel{P}()
        Test.@test K.spectral_transfer(K.TransferRate(K.TransferRate(k)), kv, ℓ) ≈
                   _cgef_cd(l -> K.scale_rate_transfer(k, kv, l), ℓ, 1e-7) rtol = 1e-6 atol = 1e-5
    end
    for k in (_CGEF_GA, _CGEF_SH, HG, _CGEF_TH), ℓs in (3e5, 2e6)
        d2G = K.spectral_transfer_degrees!(zeros(L + 1), K.TransferRate(K.TransferRate(k)), ℓs, R)
        Test.@test _cgef_relerr(d2G, _cgef_cd(l -> K.spectral_rate_degrees!(zeros(L + 1), k, l, R), ℓs, 1e-6ℓs)) < 1e-5
    end
    for δ in ((0.2, -0.1), (0.4, 0.3, -0.25))
        N = length(δ)
        pair(d, e) = prod(ntuple(f -> K.kernel_profile(K._axis_kernel(K.AxisRate(K.AxisRate(_CGEF_GA, Val(d)), Val(e)), Val(f)),
                                                       δ[f], ℓ), Val(N)))
        terms = sum(d == e ? pair(d, e) : 2pair(d, e) for d in 1:N for e in d:N)
        Test.@test terms ≈ K.kernel_weight(K.ScaleRate(K.ScaleRate(_CGEF_GA)), hypot(δ...), ℓ) rtol = 1e-12
    end
end

# `ℓ∂_ℓū = A − ū·B` from a plan's `scale_rate`: `A = Σ_p ρ_p ū_p` over its rate plans, or a kernel sum's rate series.
function _cgef_plan_rate(f, p)
    F = CGEF.Filtering
    ū = F.filter_apply!(similar(f), f, p)
    return _cgef_rate_terms(F.scale_rate(p), f, ū, p)
end

_cgef_rate_terms(r::CGEF.Filtering.WindowRate, f, ū, _) =
    sum(map((q, ρ) -> ρ .* CGEF.Filtering.filter_apply!(similar(f), f, q), r.plans, r.ρ)) .- ū .* r.B

function _cgef_rate_terms(r::CGEF.Filtering.SumRate, f, ū, p)
    F = CGEF.Filtering
    F̂ = F.filter_analyze!(F.analyze_buffer(p, f), f, p)
    return r.scale .* F.filter_synthesize!(similar(f), F̂, r.rate) .- ū .* r.B
end

# The plan's own output at `ℓ ± 1e-4ℓ` on its engine. The difference's truncation is ~1e-8 of the field, and a cell
# entering the footprint at its rim carries ~1e-10 of the peak weight, ~1e-6 of the field over the step.
function _cgef_rate_matches(grid, kernel, ℓ, s; kw...)
    F = CGEF.Filtering
    f = _cgef_field(size(FG.Grids.mask(grid)))
    p = F.plan_filter(grid, kernel, ℓ; execution = _CGEF_SERIAL, mask_strategy = s, kw...)
    e = F._engine_of(F._host_footprint(p.footprint))
    out(l) = F.filter_apply!(similar(f), f, F.plan_filter(grid, kernel, l; execution = _CGEF_SERIAL, mask_strategy = s,
                                                         engine = e, kw...))
    fd = ℓ .* _cgef_cd(out, ℓ, 1e-4ℓ)
    Test.@test maximum(abs, _cgef_plan_rate(f, p) .- fd) <= 1e-5 * maximum(abs, f)
end

Test.@testset "A real-space plan's ℓ∂_ℓū against the central difference of its own output" begin
    for r in (_cgef_cartesian_rows()..., _cgef_spherical_rows()..., _cgef_mesh_rows()...),
        s in (CGEF.Filtering.ZeroFill(), CGEF.Filtering.Deformable())
        CGEF.Kernels.has_profile_rate(r.kernel) && _cgef_rate_matches(r.grid, r.kernel, r.ℓ, s; r.kw...)
    end
    # The nonuniform-FFT kernel sum over points, its rate a series in the width.
    cart = FG.Geometry.CartesianGeometry()
    M = 1000
    xy = (20 .* mod.(0.6180339887 .* (1:M), 1.0), 20 .* mod.(0.7548776662 .* (1:M) .+ 0.31, 1.0))
    g = FG.Grids.UnstructuredGrid(cart, xy, fill(400 / M, M), [mod(k, 13) != 0 for k in 1:M])
    for s in (CGEF.Filtering.ZeroFill(), CGEF.Filtering.Deformable())
        Test.@test CGEF.Filtering.plan_filter(g, _CGEF_GA, 8.0; execution = _CGEF_SERIAL, mask_strategy = s).footprint isa
                   CGEF.Filtering.NUFFTSumFootprint
        _cgef_rate_matches(g, _CGEF_GA, 8.0, s)
    end
end

# A Fourier mode on a periodic lattice is filtered by the lattice transfer `Ĝ = Σ_o G(|o|/ℓ) cos(k·o) / Σ_o G(|o|/ℓ)`
# over the offsets within the kernel's radius, and `ℓ ∂_ℓĜ = Σ H cos / Σ G − Ĝ Σ H / Σ G` with `H` the rate weight;
# for `u = a cos(k·x)`, `v = 0`, `Ē = a²Ĝ²/4` and `dĒ/dℓ = a²Ĝ ∂_ℓĜ/2`.
Test.@testset "The exact filtering density against closed forms" begin
    D = CGEF.Diagnostics
    K = CGEF.Kernels
    EX = D.ExactScaleDerivative()
    P = FG.Grids.Periodic()
    n = 32
    x = range(0.0, 2π; length = n + 1)[1:n]
    box = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), x, x; topology = (P, P))
    kv, a = (3.0, 2.0), 0.8
    u = [a * cos(kv[1] * p + kv[2] * q) for p in x, q in x]
    scales = collect(range(0.3, 1.2; length = 5))
    Δ = step(x)
    for k in (_CGEF_SH, K.HyperGaussianKernel())
        rate = map(scales) do ℓ
            rad = K.kernel_radius(k, ℓ)
            w = ceil(Int, rad / Δ)
            o = [(i * Δ, j * Δ) for i in -w:w, j in -w:w if hypot(i * Δ, j * Δ) <= rad]
            G = [K.kernel_weight(k, hypot(t...), ℓ) for t in o]
            H = [K.kernel_weight(K.ScaleRate(k), hypot(t...), ℓ) for t in o]
            c = [cos(kv[1] * t[1] + kv[2] * t[2]) for t in o]
            Ĝ = sum(G .* c) / sum(G)
            dĜ = (sum(H .* c) / sum(G) - Ĝ * sum(H) / sum(G)) / ℓ
            a^2 * Ĝ * dĜ / 2
        end
        got = CGEF.coarse_grain(u, zero(u), box; scales, kernel = k, execution = _CGEF_SERIAL, spectrum = D.ForceSpectrum(),
                                scale_derivative = EX).filtering_spectrum
        Test.@test got ≈ -(scales .^ 2) .* rate rtol = 1e-10
    end

    # A spectral plan on a sum of Fourier modes: `Ē = Σ a²Ĝ²/4`, `dĒ/dℓ = Σ a²Ĝ ∂_ℓĜ/2`.
    modes = ((1, 0, 1.0), (3, 2, 0.6), (5, -4, 0.3), (9, 1, 0.2))
    um = [sum(m[3] * cos(m[1] * p + m[2] * q) for m in modes) for p in x, q in x]
    for k in (_CGEF_GA, _CGEF_TH)
        dĒ = [sum(m[3]^2 * K.spectral_transfer(k, hypot(m[1], m[2]), ℓ, Val(2)) *
                  K.scale_rate_transfer(k, hypot(m[1], m[2]), ℓ, Val(2)) for m in modes) / 2 for ℓ in scales]
        got = CGEF.coarse_grain(um, zero(um), box; scales, kernel = k, method = CGEF.Filtering.Spectral(), execution = _CGEF_SERIAL,
                                spectrum = D.ForceSpectrum(), scale_derivative = EX).filtering_spectrum
        Test.@test got ≈ -(scales .^ 2) .* dĒ rtol = 1e-12
    end

    # A band-limited spin-1 field under the generalized convolution: `Ē = Σ_n Ĝ(n)² p_n /(8π)` and
    # `dĒ/dℓ = Σ_n Ĝ(n) ∂_ℓĜ(n) p_n /(4π)`, `p_n = Σ_m |c_nm|²`, FastSphericalHarmonics' spin harmonics orthonormal on the
    # unit sphere and the area mean over `4πR²`.
    Nl = 32
    sg = FG.Connectivity.structured_grid(FG.SphericalSampling.Fejer1Sampling(), Nl)
    nlon, nlat = FG.Grids.size_tuple(sg)
    R = FG.Geometry.radius(FG.Grids.grid_geometry(sg))
    Lb = 10
    C = zeros(ComplexF64, nlat, nlon)
    power = zeros(Lb + 1)
    for l in 1:Lb, m in -l:l
        c = complex(cos(l + 2m), sin(3l - m)) / (1 + l)
        C[FSH.spinsph_mode(1, l, m)] = c
        power[l + 1] += abs2(c)
    end
    U = permutedims(FSH.spinsph_evaluate(C, 1))
    sscales = collect(range(4e5, 4e6; length = 5))
    for k in (_CGEF_GA, _CGEF_TH)
        dĒ = map(sscales) do ℓ
            G = K.spectral_transfer_degrees!(zeros(Lb + 1), k, ℓ, R)
            sum(G .* K.spectral_rate_degrees!(zeros(Lb + 1), k, ℓ, R) .* power) / (4π)
        end
        got = CGEF.coarse_grain(real.(U), imag.(U), sg; scales = sscales, kernel = k, execution = _CGEF_SERIAL,
                                transform = CGEF.SpectralBackends.FSHTSpectralBackend(), spectrum = D.ForceSpectrum(),
                                scale_derivative = EX).filtering_spectrum
        Test.@test got ≈ -(sscales .^ 2) .* dĒ rtol = 1e-12
    end
end

# With `D = ℓ∂_ℓ`, a Fourier mode `u = a cos(k·x)` filtered to `Ĝu` has `ū·Dū = ĜDĜ u²` and `ū·ℓ²∂²_ℓū = Ĝ(D²Ĝ − DĜ)u²`,
# so with `L = 1` the local spectrum is `−ℓĜDĜ u²`, whose area mean is the spectrum, and the slope `d ln Ẽ/d ln k_ℓ`
# is `−(2ĜDĜ + (DĜ)² + Ĝ(D²Ĝ − DĜ))/(ĜDĜ)` at every cell. On a lattice `Ĝ = N/M` with `N = Σ_o g cos(k·o)` and
# `M = Σ_o g` over the footprint's offsets, and `D` takes `g` to `H` and `H` to `J`.
Test.@testset "Spectral slopes and maps against closed forms" begin
    D = CGEF.Diagnostics
    K = CGEF.Kernels
    P = FG.Grids.Periodic()
    n = 32
    x = range(0.0, 2π; length = n + 1)[1:n]
    box = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), x, x; topology = (P, P))
    kv, a = (3.0, 2.0), 0.8
    u = [a * cos(kv[1] * p + kv[2] * q) for p in x, q in x]
    u2 = u .^ 2
    on = u2 .> 0.01a^2
    scales = collect(range(0.3, 1.2; length = 5))
    Δ = step(x)
    slope(G, G1, G2) = -(2G * G1 + G1^2 + G * (G2 - G1)) / (G * G1)
    tags = (; slopes = D.Slopes(), spectrum_maps = D.SpectrumMaps(), execution = _CGEF_SERIAL, spectrum = D.ForceSpectrum(),
            scale_derivative = D.ExactScaleDerivative())
    function check(r, want, rtol)
        Test.@test r.spectral_slope ≈ last.(want) rtol = rtol
        for (s, (e, sl)) in enumerate(want)
            m = selectdim(r.spectrum_map, 3, s)
            Test.@test m ≈ e .* u2 rtol = rtol
            Test.@test sum(m) / n^2 ≈ r.filtering_spectrum[s] rtol = 1e-12
            Test.@test selectdim(r.slope_map, 3, s)[on] ≈ fill(sl, count(on)) rtol = rtol
        end
    end
    for k in (_CGEF_SH, K.HyperGaussianKernel())
        want = map(scales) do ℓ
            rad = K.kernel_radius(k, ℓ)
            w = ceil(Int, rad / Δ)
            o = [(i * Δ, j * Δ) for i in -w:w, j in -w:w if hypot(i * Δ, j * Δ) <= rad]
            c = [cos(kv[1] * t[1] + kv[2] * t[2]) for t in o]
            g, h, j = (map(t -> K.kernel_weight(q, hypot(t...), ℓ), o) for q in (k, K.ScaleRate(k), K.ScaleRate(K.ScaleRate(k))))
            M0, M1, M2 = sum(g), sum(h), sum(j)
            G = sum(g .* c) / M0
            G1 = (sum(h .* c) - G * M1) / M0
            G2 = (sum(j .* c) - G * M2) / M0 - 2G1 * M1 / M0
            (-ℓ * G * G1, slope(G, G1, G2))
        end
        Logging.with_logger(Logging.NullLogger()) do
            check(CGEF.coarse_grain(u, zero(u), box; scales, kernel = k, tags...), want, 1e-9)
        end
    end
    for k in (_CGEF_GA, _CGEF_TH)
        want = map(scales) do ℓ
            km = hypot(kv...)
            G = K.spectral_transfer(k, km, ℓ, Val(2))
            G1 = ℓ * K.spectral_transfer(K.TransferRate(k), km, ℓ, Val(2))
            G2 = ℓ^2 * K.spectral_transfer(K.TransferRate(K.TransferRate(k)), km, ℓ, Val(2)) + G1
            (-ℓ * G * G1, slope(G, G1, G2))
        end
        Logging.with_logger(Logging.NullLogger()) do
            check(CGEF.coarse_grain(u, zero(u), box; scales, kernel = k, method = CGEF.Filtering.Spectral(), tags...), want, 1e-11)
        end
    end

    # Without the tags a result holds neither; the device whole-batch sweep writes each slice's as its own sweep does.
    r0 = CGEF.coarse_grain(u, zero(u), box; scales, kernel = _CGEF_GA, execution = _CGEF_SERIAL)
    Test.@test r0.spectral_slope === nothing && r0.spectrum_map === nothing && r0.slope_map === nothing
    ub = cat(u, 0.5 .* u .+ 0.1; dims = 3)
    vb = cat(zero(u), 0.3 .* u; dims = 3)
    dev = CGEF.ComputationalBackends.GPUBackend(KA.CPU())
    b = CGEF.Pipeline.CoarseGrainBatchResult(box, length(scales), (2,); execution = dev, slopes = D.Slopes(),
                                             spectrum_maps = D.SpectrumMaps())
    CGEF.Pipeline.coarse_grain_batch!(b, ub, vb, box; scales, kernel = _CGEF_GA, execution = dev,
                                      scale_derivative = D.ExactScaleDerivative())
    for t in 1:2
        rt = CGEF.coarse_grain(ub[:, :, t], vb[:, :, t], box; scales, kernel = _CGEF_GA, tags...)
        Test.@test b.slices[t].spectral_slope ≈ rt.spectral_slope rtol = 1e-12
        Test.@test b.slices[t].spectrum_map ≈ rt.spectrum_map rtol = 1e-12
        Test.@test b.slices[t].slope_map ≈ rt.slope_map rtol = 1e-10
    end
end

# Which route a sweep takes: the exact one wherever the kernel has a rate on the plan's route, the difference
# otherwise; asked for exactly where it has none, it raises. A family built with its scale rates gives the spectrum
# the per-call route does, and the exact spectrum is the same on every execution backend.
Test.@testset "Scale-derivative routes, held rates and backends" begin
    D = CGEF.Diagnostics
    F = CGEF.Filtering
    CB = CGEF.ComputationalBackends
    xb = range(0.0, 1.0; length = 40)
    m = [hypot(a - 0.4, b - 0.6) >= 0.12 for a in xb, b in xb]
    g = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), xb, xb, m)
    u = [sin(5a + 2b) + 0.3cos(7b) for a in xb, b in xb]
    v = [cos(3a - 4b) for a in xb, b in xb]
    scales = collect(range(0.05, 0.25; length = 6))
    spec(k, sd; kw...) = CGEF.coarse_grain(u, v, g; scales, kernel = k, execution = _CGEF_SERIAL, spectrum = D.ForceSpectrum(),
                                           scale_derivative = sd, kw...).filtering_spectrum
    Logging.with_logger(Logging.NullLogger()) do
        Test.@test spec(_CGEF_TH, D.AutoScaleDerivative()) == spec(_CGEF_TH, D.DifferenceScaleDerivative())
        Test.@test_throws ArgumentError spec(_CGEF_TH, D.ExactScaleDerivative())
        Test.@test spec(_CGEF_TH, D.ExactScaleDerivative(); method = F.Spectral()) isa Vector{Float64}
    end
    Test.@test spec(_CGEF_GA, D.AutoScaleDerivative()) == spec(_CGEF_GA, D.ExactScaleDerivative())
    Test.@test spec(_CGEF_GA, D.ExactScaleDerivative()) != spec(_CGEF_GA, D.DifferenceScaleDerivative())

    for (k, s) in ((_CGEF_SH, F.Deformable()), (_CGEF_GA, F.ZeroFill()))
        held = F.plan_filter_sweep(g, k, scales; mask_strategy = s, execution = _CGEF_SERIAL, scale_rates = F.ScaleRates())
        Test.@test held.rates isa AbstractVector
        r = CGEF.Pipeline._allocate_result(g, length(scales))
        kw = (; scales, kernel = k, mask_strategy = s, execution = _CGEF_SERIAL, spectrum = D.ForceSpectrum(),
              scale_derivative = D.ExactScaleDerivative())
        Logging.with_logger(Logging.NullLogger()) do
            Test.@test CGEF.coarse_grain!(r, u, v, g; filter_plans = held, kw...).filtering_spectrum ==
                       CGEF.coarse_grain(u, v, g; kw...).filtering_spectrum
        end
    end
    Test.@test_throws ArgumentError F.plan_filter_sweep(g, _CGEF_GA, scales; method = F.Spectral(), execution = _CGEF_SERIAL,
                                                        scale_rates = F.ScaleRates())
    Test.@test_throws ArgumentError F.plan_filter_sweep(g, _CGEF_TH, scales; execution = _CGEF_SERIAL,
                                                        scale_rates = F.ScaleRates())

    ref = spec(_CGEF_GA, D.ExactScaleDerivative())
    for b in (CB.ThreadedBackend(), CB.GPUBackend(KA.CPU()))
        got = CGEF.coarse_grain(u, v, g; scales, kernel = _CGEF_GA, execution = b, spectrum = D.ForceSpectrum(),
                                scale_derivative = D.ExactScaleDerivative()).filtering_spectrum
        Test.@test got ≈ ref rtol = 1e-12
    end
end
