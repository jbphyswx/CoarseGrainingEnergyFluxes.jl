# Physics validation ladder: properties the coarse-graining framework REQUIRES, each checked against a
# closed form or an independently-constructed reference rather than against stored output. These are the
# tests that fail when a convention drifts, as opposed to the per-module tests that fail when an
# implementation detail does.

# ---------------------------------------------------------------------------
# Synthetic fields with a prescribed kinetic-energy spectrum.
#
# Built from a streamfunction, `u = -∂ψ/∂y`, `v = ∂ψ/∂x`, so the field is divergence-free by
# construction. Which derivative that holds for is selected by `symbol`:
#
#   :exact   — `û = -i k_y ψ̂`, divergence-free for the SPECTRAL derivative. `|û|² = k²|ψ̂|²`, so
#              `E(k) ∝ k^(-α)` exactly, which is what the slope test needs.
#   :stencil — `û = -i (sin(k_y Δx)/Δx) ψ̂`, divergence-free for the package's own 3-point CENTERED
#              derivative, so `ddx!(u) + ddy!(v)` vanishes to round-off. The spectrum then rolls off
#              near the grid scale, which the commutation test does not care about.
#
# Phases come from an irrational rotation rather than an RNG: deterministic, no seed, and identical on
# every Julia version.
# ---------------------------------------------------------------------------
_cgef_kaxis(N, dx) = [2π / (N * dx) * (i - 1 <= N ÷ 2 ? i - 1 : i - 1 - N) for i in 1:N]
_cgef_phase(i, j) = 2π * mod(0.7548776662466927 * i + 0.5698402909980532 * j, 1.0)

function _cgef_spectral_field(N, dx, α; symbol = :exact)
    k = _cgef_kaxis(N, dx)
    ψ̂ = zeros(ComplexF64, N, N)
    for j in 1:N, i in 1:N
        kmag = sqrt(k[i]^2 + k[j]^2)
        iszero(kmag) && continue
        ψ̂[i, j] = kmag^(-(α + 3) / 2) * cis(_cgef_phase(i, j))
    end
    # Round-trip through a real field so the coefficients are Hermitian-symmetric.
    ψ̂ = FFTW.fft(real(FFTW.ifft(ψ̂)))
    sym(κ) = symbol === :exact ? κ : sin(κ * dx) / dx
    û = similar(ψ̂); v̂ = similar(ψ̂)
    for j in 1:N, i in 1:N
        û[i, j] = -im * sym(k[j]) * ψ̂[i, j]
        v̂[i, j] = im * sym(k[i]) * ψ̂[i, j]
    end
    return real(FFTW.ifft(û)), real(FFTW.ifft(v̂))
end

# Least-squares slope of log10(y) on log10(x).
function _cgef_logfit(x, y)
    lx = log10.(x); ly = log10.(y); n = length(lx)
    x̄ = sum(lx) / n; ȳ = sum(ly) / n
    return sum((lx .- x̄) .* (ly .- ȳ)) / sum((lx .- x̄) .^ 2)
end

# Shell-averaged Fourier KE spectrum slope — an INDEPENDENT check that the input really has the slope
# it was asked for, so a failure below cannot be blamed on the synthetic field.
function _cgef_fourier_slope(u, v, dx, nlo, nhi)
    N = size(u, 1); kx = _cgef_kaxis(N, dx); dk = 2π / (N * dx)
    û = FFTW.fft(u) ./ (N^2); v̂ = FFTW.fft(v) ./ (N^2)
    E = zeros(N ÷ 2)
    for j in 1:N, i in 1:N
        n = round(Int, sqrt(kx[i]^2 + kx[j]^2) / dk)
        1 <= n <= N ÷ 2 || continue
        E[n] += 0.5 * (abs2(û[i, j]) + abs2(v̂[i, j]))
    end
    ns = [n for n in nlo:nhi if E[n] > 0]
    return _cgef_logfit(Float64.(ns), E[ns])
end

# Sadek & Aluie (2018) Eq. 18: a kernel with `p` vanishing moments recovers a true `k^-α` spectrum only
# while `α < p + 2`, and saturates at `k^-(p+2)` above it. `TopHatKernel` and `GaussianKernel` both have
# `p = 1`, so the ceiling is `k⁻³` for both — the claim on `Diagnostics.filtering_spectrum`, measured.
Test.@testset "Filtering spectrum: recovered slope is min(α, p+2), p = 1" begin
    N = 256; dx = 1000.0
    geom = FG.Geometry.CartesianGeometry()
    xs = range(0.0, dx * N; length = N + 1)[1:N]
    grid = FG.Grids.StructuredGrid(geom, xs, xs; periodic = (true, true))
    # Sweep wider than the fit window: `spectral_density` is one-sided at the ends, so the endpoints
    # carry a slope bias that a fit over the interior does not.
    scales = collect(exp10.(range(log10(5 * dx), log10(64 * dx); length = 17)))
    kℓ = 1.0 ./ scales
    w = findall(i -> 8 * dx <= scales[i] <= 40 * dx, eachindex(scales))

    got = Dict{Tuple{String,Float64},Float64}()
    for (kname, kern) in (("Gaussian", CGEF.GaussianKernel()), ("TopHat", CGEF.TopHatKernel()))
        for α in (5 / 3, 4.0, 7.0)
            u, v = _cgef_spectral_field(N, dx, α)
            # The input is what it claims to be.
            Test.@test _cgef_fourier_slope(u, v, dx, 6, 32) ≈ -α atol = 0.05
            # `cumulative_energy` + `spectral_density`, not `filtering_spectrum`, so the same code path
            # serves both kernels — the gate on the density refuses the top-hat, and the point here is
            # that the ceiling is a property of `p`, not of which kernel is admissible.
            cum = CGEF.Diagnostics.cumulative_energy(u, v, nothing, grid, kern, scales)
            Ẽ = CGEF.Diagnostics.spectral_density(cum, kℓ)
            got[(kname, α)] = _cgef_logfit(kℓ[w], Ẽ[w])
        end
    end

    for kname in ("Gaussian", "TopHat")
        # Shallow spectrum, α < p+2: the true slope IS recovered.
        Test.@test got[(kname, 5 / 3)] ≈ -5 / 3 atol = 0.20
        # Steep spectra, α > p+2: the slope saturates near -3, nowhere near the truth.
        Test.@test -3.15 < got[(kname, 7.0)] < -2.75
        Test.@test got[(kname, 7.0)] > -4.0            # ... i.e. off from the true -7 by > 3
        Test.@test -3.10 < got[(kname, 4.0)] < -2.60
        Test.@test got[(kname, 4.0)] > -3.5            # ... off from the true -4
        # The ceiling is approached from below over a finite fit range, so the ordering is strict.
        Test.@test got[(kname, 7.0)] < got[(kname, 4.0)] < got[(kname, 5 / 3)]
    end
    # Same `p`, so the two kernels must agree on all three slopes.
    for α in (5 / 3, 4.0, 7.0)
        Test.@test got[("Gaussian", α)] ≈ got[("TopHat", α)] atol = 0.05
    end

    # A HIGHER-order kernel lifts the ceiling, which is the only reason to want one. `HighOrderKernel{3}`
    # has `p = 3`, so `min(α, p+2) = 4` at `α = 4` — a slope the `p = 1` kernels above saturate short of.
    # The sweep stays at `ℓ ≥ 8Δx` throughout so its limbs are resolved everywhere.
    let ho = CGEF.Kernels.HighOrderKernel(; order = 3),
        hs = collect(exp10.(range(log10(8 * dx), log10(72 * dx); length = 17))),
        hk = 1.0 ./ hs,
        hw = findall(i -> 12 * dx <= hs[i] <= 48 * dx, eachindex(hs))
        u4, v4 = _cgef_spectral_field(N, dx, 4.0)
        cum = CGEF.Diagnostics.cumulative_energy(u4, v4, nothing, grid, ho, hs)
        Ẽh = CGEF.Diagnostics.spectral_density(cum, hk)
        # A signed kernel's density is NOT guaranteed non-negative — that is exactly what
        # `transfer_monotone(ho) == false` says, and why `filtering_spectrum` refuses this kernel. Fit
        # the positive part, and require enough of it that the fit means something.
        Test.@test !CGEF.Kernels.transfer_monotone(ho)
        pos = [i for i in hw if Ẽh[i] > 0]
        Test.@test length(pos) >= 6
        s4 = _cgef_logfit(hk[pos], Ẽh[pos])
        # Recovers the true -4 …
        Test.@test s4 ≈ -4.0 atol = 0.25
        # … and is decisively steeper than what p = 1 manages on the SAME field and window, which is
        # the comparison that makes this about `p` and not about the sweep.
        for kname in ("Gaussian", "TopHat")
            kk = kname == "Gaussian" ? CGEF.GaussianKernel() : CGEF.TopHatKernel()
            c1 = CGEF.Diagnostics.cumulative_energy(u4, v4, nothing, grid, kk, hs)
            s1 = _cgef_logfit(hk[hw], CGEF.Diagnostics.spectral_density(c1, hk)[hw])
            Test.@test s4 < s1 - 0.5
        end
        # The shallow case is recovered by every `p`, so a higher order costs nothing there.
        u1, v1 = _cgef_spectral_field(N, dx, 5 / 3)
        c5 = CGEF.Diagnostics.cumulative_energy(u1, v1, nothing, grid, ho, hs)
        Ẽ5 = CGEF.Diagnostics.spectral_density(c5, hk)
        p5 = [i for i in hw if Ẽ5[i] > 0]
        Test.@test _cgef_logfit(hk[p5], Ẽ5[p5]) ≈ -5 / 3 atol = 0.20
    end
end

# The flux budget is derived by commuting the filter with a spatial derivative. On a uniform periodic
# grid the kernel is position-independent, so that commutation is exact — and a divergence-free field
# stays divergence-free under filtering, which is the form the budget actually uses.
Test.@testset "filter ∘ div == div ∘ filter, and ∇·ū = 0 for a divergence-free field" begin
    dx = 1000.0
    geom = FG.Geometry.CartesianGeometry()
    for N in (64, 96)
        xs = range(0.0, dx * N; length = N + 1)[1:N]
        grid = FG.Grids.StructuredGrid(geom, xs, xs; periodic = (true, true))
        u, v = _cgef_spectral_field(N, dx, 3.0; symbol = :stencil)
        du = zeros(N, N); dv = zeros(N, N)
        CGEF.Derivatives.ddx!(du, u, grid)
        CGEF.Derivatives.ddy!(dv, v, grid)
        divu = du .+ dv
        scl = max(maximum(abs, du), maximum(abs, dv))
        # Precondition: the field really is divergence-free for the derivative the package uses.
        Test.@test maximum(abs, divu) < 1e-12 * scl

        for kern in (CGEF.GaussianKernel(), CGEF.TopHatKernel()), ℓ in (8 * dx, 16 * dx)
            plan = CGEF.Filtering.plan_filter(grid, kern, ℓ)
            ū = zeros(N, N); v̄ = zeros(N, N)
            CGEF.Filtering.filter_apply!(ū, u, plan)
            CGEF.Filtering.filter_apply!(v̄, v, plan)
            dū = zeros(N, N); dv̄ = zeros(N, N)
            CGEF.Derivatives.ddx!(dū, ū, grid)
            CGEF.Derivatives.ddy!(dv̄, v̄, grid)
            div_filt = dū .+ dv̄
            s2 = max(maximum(abs, dū), maximum(abs, dv̄))
            # ∇·ū = 0: filtering does not create divergence.
            Test.@test maximum(abs, div_filt) < 1e-11 * s2
            # The identity itself, which needs no divergence-free assumption: filter(∇·u) = ∇·(filter u).
            filt_div = zeros(N, N)
            CGEF.Filtering.filter_apply!(filt_div, divu, plan)
            Test.@test maximum(abs, filt_div .- div_filt) < 1e-11 * s2
        end
    end
end

# Measured through a top-level, fully-qualified helper rather than inline in the testset: inside a
# testset the arguments and the `CGEF` alias are captured locals, and `@allocated` then charges that
# capture to the call instead of to the function under test.
_cgef_alloc_strain(ws, u, v, grid, kern, ℓ, plan, dpl) =
    @allocated CGEF.Diagnostics.compute_Π_strain_convergence!(
        ws, u, v, grid, kern, ℓ; filter_plan = plan, deriv_plan = dpl,
        backend = CGEF.ComputationalBackends.SerialBackend())

# A normalized, even kernel has `∫G = 1` and `∫xG = 0`, so it reproduces any LINEAR field exactly. Two
# consequences the framework leans on, both checked away from the domain edge where the footprint is
# truncated: solid-body rotation and pure strain both filter to themselves, so neither generates
# small-scale energy, and pure strain gives `Π = 0` even though its subfilter stress is NOT zero.
Test.@testset "Linear fields: rotation and pure strain reproduce exactly, Π = 0" begin
    geom = FG.Geometry.CartesianGeometry()
    N = 64; dx = 1000.0
    xs = collect(range(-(N ÷ 2) * dx, ((N ÷ 2) - 1) * dx; length = N))
    grid = FG.Grids.StructuredGrid(geom, xs, xs, trues(N, N))
    Ω = 1e-4
    ℓ = 6 * dx
    ones_f = ones(N, N)
    Yf = [xs[j] for i in 1:N, j in 1:N]              # y as a field (y is the second axis)

    for kern in (CGEF.TopHatKernel(), CGEF.GaussianKernel())
        # Interior = farther than the kernel radius from every edge, so the footprint is whole and the
        # linear-reproduction property is the only thing under test.
        pad = ceil(Int, CGEF.Kernels.kernel_radius(kern, ℓ) / dx) + 1
        ii = (pad + 1):(N - pad)
        plan = CGEF.Filtering.plan_filter(grid, kern, ℓ)

        # The kernel's own DISCRETE second moment, read straight out of the filter by linearity, so the
        # τ identity below carries no hardcoded constant and no discretization slack.
        f1 = zeros(N, N); fy = zeros(N, N); fy2 = zeros(N, N)
        CGEF.Filtering.filter_apply!(f1, ones_f, plan)
        CGEF.Filtering.filter_apply!(fy, Yf, plan)
        CGEF.Filtering.filter_apply!(fy2, Yf .^ 2, plan)
        σ² = fy2 ./ f1 .- (fy ./ f1) .^ 2
        σ²ᵢ = σ²[N ÷ 2, N ÷ 2]
        # Translation-invariant in the interior: one number, not a field.
        Test.@test maximum(abs, @view(σ²[ii, ii]) .- σ²ᵢ) < 1e-12 * σ²ᵢ
        # ... and it is the continuum moment the kernel docstring quotes: ℓ²/12 for the Gaussian
        # (separable, so exact on a lattice) and ℓ²/16 for the 2-D disk top-hat (staircased rim, so a
        # few percent off at this coarse ℓ = 6Δx).
        if kern isa CGEF.Kernels.GaussianKernel
            Test.@test σ²ᵢ ≈ ℓ^2 / 12 rtol = 1e-9
        else
            Test.@test σ²ᵢ ≈ ℓ^2 / 16 rtol = 0.06
        end

        for (name, uf, vf) in (
            ("solid-body rotation", (x, y) -> -Ω * y, (x, y) -> Ω * x),
            ("pure strain",         (x, y) ->  Ω * y, (x, y) -> Ω * x),
        )
            u = [uf(xs[i], xs[j]) for i in 1:N, j in 1:N]
            v = [vf(xs[i], xs[j]) for i in 1:N, j in 1:N]
            ū = zeros(N, N); v̄ = zeros(N, N)
            CGEF.Filtering.filter_apply!(ū, u, plan)
            CGEF.Filtering.filter_apply!(v̄, v, plan)
            uscl = maximum(abs, u)
            # A linear field filters to itself: `∫xG = 0` in action, hence no small-scale energy at any
            # scale — `E(ℓ)` is flat and `Ẽ` vanishes.
            Test.@test maximum(abs, @view(ū[ii, ii]) .- @view(u[ii, ii])) < 1e-14 * uscl
            Test.@test maximum(abs, @view(v̄[ii, ii]) .- @view(v[ii, ii])) < 1e-14 * uscl

            Π = zeros(N, N)
            CGEF.Diagnostics.compute_Π!(Π, u, v, nothing, grid, kern, ℓ)
            # `Π ~ S̄·τ ~ Ω · Ω²σ²`, so that product is the scale to measure "zero" against.
            Test.@test maximum(abs, @view Π[ii, ii]) < 1e-11 * (Ω^3 * σ²ᵢ)
        end

        # The pure-strain `Π = 0` is a cancellation, not a vacuum. For `u = (Ωy, Ωx)`:
        # `S̄ = [[0, Ω], [Ω, 0]]`, so `Π = -2Ω τ₁₂`. The stress is NOT zero — `τ₁₁ = τ₂₂ = Ω²σ²`, the
        # kernel's second moment — and `Π` vanishes only because the kernel's CROSS moment does.
        let u = [Ω * xs[j] for i in 1:N, j in 1:N], v = [Ω * xs[i] for i in 1:N, j in 1:N]
            L, C, R = CGEF.Diagnostics.tau_decomposition(u, v, grid, kern, ℓ)
            τ11 = L.xx .+ C.xx .+ R.xx
            τ12 = L.xy .+ C.xy .+ R.xy
            τ22 = L.yy .+ C.yy .+ R.yy
            Test.@test all(>(0), @view τ11[ii, ii])                   # a variance: strictly positive
            Test.@test all(≈(Ω^2 * σ²ᵢ; rtol = 1e-12), @view τ11[ii, ii])
            Test.@test all(≈(Ω^2 * σ²ᵢ; rtol = 1e-12), @view τ22[ii, ii])
            Test.@test maximum(abs, @view τ12[ii, ii]) < 1e-11 * (Ω^2 * σ²ᵢ)
        end
    end
end

# Srinivasan, Barkan & McWilliams (2023) eq. (10) rewrites Π by diagonalizing the filtered strain
# tensor. Expanding it collapses exactly to the direct `-S̄:τ̄`, so the two MUST agree — but they
# contract different combinations of the same four derivatives, which is why the agreement is a real
# test rather than a tautology.
Test.@testset "Π strain/convergence decomposition equals the direct −S̄:τ̄ form" begin
    geom = FG.Geometry.CartesianGeometry()
    N = 48; dx = 1000.0
    xs = 0.0:dx:(N - 1) * dx
    # Both strain and divergence present, so neither term of the split is trivially zero.
    u = [sin(2π * 3 * i / N) * cos(2π * 2 * j / N) + 0.4cos(2π * 5 * i / N) for i in 1:N, j in 1:N]
    v = [cos(2π * 2 * i / N) * sin(2π * 4 * j / N) + 0.3sin(2π * 6 * j / N) for i in 1:N, j in 1:N]

    masks = ((trues(N, N), "unmasked"),
             ((m = trues(N, N); m[12:18, 10:16] .= false; m[1, :] .= false; m), "masked"))
    for (mask, mname) in masks, kern in (CGEF.TopHatKernel(), CGEF.GaussianKernel()),
        strat in (CGEF.Filtering.ZeroFill(), CGEF.Filtering.Deformable())
        grid = FG.Grids.StructuredGrid(geom, xs, xs, mask)
        for ℓ in (4 * dx, 10 * dx)
            d = CGEF.Diagnostics.compute_Π_strain_convergence(
                u, v, grid, kern, ℓ; mask_strategy = strat)
            Π = zeros(N, N)
            CGEF.Diagnostics.compute_Π!(Π, u, v, nothing, grid, kern, ℓ; mask_strategy = strat)
            scl = maximum(abs, Π)
            Test.@test scl > 0                                   # not a vacuous comparison
            Test.@test maximum(abs, d.total .- Π) < 1e-12 * scl
            # Both terms carry real signal, so `total` is not one of them plus noise.
            Test.@test maximum(abs, d.strain) > 0.05 * scl
            Test.@test maximum(abs, d.convergence) > 0.05 * scl
            # Masked cells are zero in every returned field, as everywhere else in the package.
            for f in (d.total, d.strain, d.convergence, d.divergence, d.strain_magnitude)
                Test.@test all(iszero, f[.!mask])
            end
            # ᾱ ≥ 0 by construction, and the paper's bound |Π_α| ≤ ᾱ E′ with E′ = (τ_uu + τ_vv)/2.
            Test.@test all(>=(0), d.strain_magnitude)
            L, C, R = CGEF.Diagnostics.tau_decomposition(u, v, grid, kern, ℓ; mask_strategy = strat)
            E′ = ((L.xx .+ C.xx .+ R.xx) .+ (L.yy .+ C.yy .+ R.yy)) ./ 2
            Test.@test all(I -> !mask[I] ||
                                abs(d.strain[I]) <= abs(d.strain_magnitude[I] * E′[I]) + 1e-12 * scl,
                           CartesianIndices(d.strain))
        end
    end

    # A non-divergent field has δ̄ = 0, so the convergence term vanishes identically and the whole flux
    # is deformation production — the limit in which eq. (10) reduces to Polzin (2010).
    let Np = 64,
        xp = range(0.0, dx * Np; length = Np + 1)[1:Np],
        gp = FG.Grids.StructuredGrid(geom, xp, xp; periodic = (true, true))
        us, vs = _cgef_spectral_field(Np, dx, 3.0; symbol = :stencil)
        d = CGEF.Diagnostics.compute_Π_strain_convergence(us, vs, gp, CGEF.GaussianKernel(), 8 * dx)
        Test.@test maximum(abs, d.divergence) < 1e-11 * maximum(abs, d.strain_magnitude)
        Test.@test maximum(abs, d.convergence) < 1e-11 * maximum(abs, d.strain)
        Test.@test maximum(abs, d.total .- d.strain) < 1e-11 * maximum(abs, d.strain)
    end

    # The in-place form reuses its buffers: a repeated evaluation with the workspace and both plans
    # held allocates nothing, and it agrees bit-for-bit with the allocating form it shares code with.
    let grid = FG.Grids.StructuredGrid(geom, xs, xs, trues(N, N)),
        kern = CGEF.GaussianKernel(), ℓ = 6 * dx,
        SER = CGEF.ComputationalBackends.SerialBackend(),
        ws = CGEF.Diagnostics.PiStrainWorkspace(grid),
        plan = CGEF.Filtering.plan_filter(grid, kern, ℓ; backend = SER),
        dpl = CGEF.Derivatives.StencilPlan(grid)
        CGEF.Diagnostics.compute_Π_strain_convergence!(
            ws, u, v, grid, kern, ℓ; filter_plan = plan, deriv_plan = dpl, backend = SER)
        Test.@test _cgef_alloc_strain(ws, u, v, grid, kern, ℓ, plan, dpl) == 0
        r = CGEF.Diagnostics.compute_Π_strain_convergence!(
            ws, u, v, grid, kern, ℓ; filter_plan = plan, deriv_plan = dpl, backend = SER)
        a = CGEF.Diagnostics.compute_Π_strain_convergence(u, v, grid, kern, ℓ; backend = SER)
        Test.@test r.total == a.total
        Test.@test r.strain == a.strain
        Test.@test r.convergence == a.convergence
    end
end

_cgef_alloc_enstrophy(Z, ws, u, v, grid, kern, ℓ, plan, dpl) =
    @allocated CGEF.Diagnostics.enstrophy_flux!(
        Z, ws, u, v, grid, kern, ℓ; filter_plan = plan, deriv_plan = dpl,
        backend = CGEF.ComputationalBackends.SerialBackend())

# The enstrophy flux is the enstrophy analogue of Π, in the SAME (deformation) gauge. In 2-D turbulence
# it is the quantity expected to cascade forward while Π cascades inverse, so the two are read together
# and must not disagree about which gauge they are in.
Test.@testset "Enstrophy flux: vorticity, deformation gauge, and Π's dual" begin
    geom = FG.Geometry.CartesianGeometry()
    N = 48; dx = 1000.0
    xs = 0.0:dx:(N - 1) * dx
    grid = FG.Grids.StructuredGrid(geom, xs, xs, trues(N, N))
    u = [sin(2π * 3 * i / N) * cos(2π * 2 * j / N) + 0.4cos(2π * 5 * i / N) for i in 1:N, j in 1:N]
    v = [cos(2π * 2 * i / N) * sin(2π * 4 * j / N) + 0.3sin(2π * 6 * j / N) for i in 1:N, j in 1:N]
    ℓ = 6 * dx

    # `ω = ∂v/∂x − ∂u/∂y`, built with the package's own derivative so it matches everything downstream.
    ω = CGEF.Diagnostics.vorticity(u, v, grid)
    let dvdx = zeros(N, N), dudy = zeros(N, N)
        CGEF.Derivatives.ddx!(dvdx, v, grid)
        CGEF.Derivatives.ddy!(dudy, u, grid)
        Test.@test ω == dvdx .- dudy
    end
    # A solid-body rotation has uniform vorticity 2Ω, and pure strain has none — both exact away from
    # the edge, where the one-sided stencil takes over.
    let Ω = 1e-4, ii = 2:(N - 1),
        xc = collect(range(-(N ÷ 2) * dx, ((N ÷ 2) - 1) * dx; length = N)),
        gc = FG.Grids.StructuredGrid(geom, xc, xc, trues(N, N))
        ur = [-Ω * xc[j] for i in 1:N, j in 1:N]; vr = [Ω * xc[i] for i in 1:N, j in 1:N]
        Test.@test all(≈(2Ω; rtol = 1e-12), CGEF.Diagnostics.vorticity(ur, vr, gc)[ii, ii])
        us = [Ω * xc[j] for i in 1:N, j in 1:N]
        Test.@test all(<(1e-12 * Ω), abs.(CGEF.Diagnostics.vorticity(us, vr, gc)[ii, ii]))
    end

    Z = CGEF.Diagnostics.enstrophy_flux(u, v, grid, CGEF.GaussianKernel(), ℓ)
    Test.@test size(Z) == (N, N)
    Test.@test all(isfinite, Z)
    Test.@test maximum(abs, Z) > 0

    # It IS the tracer-variance flux with θ = ω — the enstrophy is the vorticity's variance. Asserted
    # bit-for-bit, so the two can never drift into different conventions.
    Test.@test Z == CGEF.Diagnostics.tracer_variance_flux(u, v, ω, grid, CGEF.GaussianKernel(), ℓ)

    # The deformation gauge, spelled out: `Z = −∂_jω̄ · (⟨u_jω⟩ − ū_jω̄)`, assembled here from
    # primitives. The UNSUBTRACTED gauge `−∂_jω̄ ⟨u_jω⟩` is a genuinely different field — asserted, so
    # a silent switch of gauge cannot pass.
    let plan = CGEF.Filtering.plan_filter(grid, CGEF.GaussianKernel(), ℓ),
        f(a) = (o = zeros(N, N); CGEF.Filtering.filter_apply!(o, a, plan); o)
        ū = f(u); v̄ = f(v); ω̄ = f(ω)
        τx = f(u .* ω) .- ū .* ω̄
        τy = f(v .* ω) .- v̄ .* ω̄
        gx = zeros(N, N); gy = zeros(N, N)
        CGEF.Derivatives.ddx!(gx, ω̄, grid); CGEF.Derivatives.ddy!(gy, ω̄, grid)
        Test.@test maximum(abs, Z .- (-(τx .* gx .+ τy .* gy))) < 1e-12 * maximum(abs, Z)
        uns = -(f(u .* ω) .* gx .+ f(v .* ω) .* gy)
        Test.@test maximum(abs, Z .- uns) > 0.05 * maximum(abs, Z)
    end

    # Masked cells are zero, and the mask is honoured through both stages.
    let m = trues(N, N); m[12:18, 10:16] .= false
        gm = FG.Grids.StructuredGrid(geom, xs, xs, m)
        Zm = CGEF.Diagnostics.enstrophy_flux(u, v, gm, CGEF.GaussianKernel(), ℓ)
        Test.@test all(iszero, Zm[.!m])
        Test.@test all(iszero, CGEF.Diagnostics.vorticity(u, v, gm)[.!m])
    end

    # The in-place form matches the allocating one and reuses its buffers.
    let SER = CGEF.ComputationalBackends.SerialBackend(),
        kern = CGEF.GaussianKernel(),
        ws = CGEF.Diagnostics.EnstrophyFluxWorkspace(grid),
        plan = CGEF.Filtering.plan_filter(grid, kern, ℓ; backend = SER),
        dpl = CGEF.Derivatives.StencilPlan(grid),
        Zi = zeros(N, N)
        CGEF.Diagnostics.enstrophy_flux!(Zi, ws, u, v, grid, kern, ℓ;
                                         filter_plan = plan, deriv_plan = dpl, backend = SER)
        Test.@test Zi == CGEF.Diagnostics.enstrophy_flux(u, v, grid, kern, ℓ; backend = SER)
        Test.@test _cgef_alloc_enstrophy(Zi, ws, u, v, grid, kern, ℓ, plan, dpl) == 0
    end
end

# The scale-band decomposition. Its correctness rests on the SAME property `ZeroFill` was made the
# default for — each filter conserving the domain mean — which holds exactly only where the footprint
# is not truncated. So the identity is asserted EXACT on a periodic unmasked grid, and bounded by the
# measured `O(ℓ/L)` boundary residual elsewhere.
Test.@testset "Band energies: repeated-filter Germano identity sums to the total" begin
    geom = FG.Geometry.CartesianGeometry()
    N = 48; dx = 1000.0
    u = [sin(2π * 3 * i / N) * cos(2π * 2 * j / N) + 0.4cos(2π * 7 * i / N) for i in 1:N, j in 1:N]
    v = [cos(2π * 2 * i / N) * sin(2π * 5 * j / N) + 0.3sin(2π * 6 * j / N) for i in 1:N, j in 1:N]
    scales = [3 * dx, 6 * dx, 12 * dx]
    xp = range(0.0, dx * N; length = N + 1)[1:N]
    grid = FG.Grids.StructuredGrid(geom, xp, xp; periodic = (true, true))
    Etot = 0.5 * sum(u .^ 2 .+ v .^ 2) / (N * N)

    for kern in (CGEF.TopHatKernel(), CGEF.GaussianKernel(), CGEF.Kernels.SmoothHatKernel())
        r = CGEF.Diagnostics.band_energies(u, v, grid, kern, scales; maps = true)
        Test.@test length(r.bands) == length(scales)
        # Maps are opt-in: the default call returns the scalars alone.
        Test.@test CGEF.Diagnostics.band_energies(u, v, grid, kern, scales).band_maps === nothing
        # The identity, exact: bands + resolved == the unfiltered energy.
        Test.@test r.total ≈ Etot rtol = 1e-12
        Test.@test sum(r.bands) + r.resolved == r.total
        # Every band carries real signal — not one band plus noise.
        Test.@test all(>(0), r.bands)
        Test.@test r.resolved > 0
        # A positive kernel gives POINTWISE non-negative band energies: they are variances.
        for km in r.band_maps
            Test.@test all(>=(-1e-14 * Etot), km)
        end
        # Coarser filtering leaves less, and the identity still closes with fewer bands.
        r2 = CGEF.Diagnostics.band_energies(u, v, grid, kern, scales[1:2])
        Test.@test r2.resolved > r.resolved
        Test.@test r2.total ≈ Etot rtol = 1e-12
    end

    # It is NOT the naive velocity band-pass, whose cross terms have indefinite sign.
    let kern = CGEF.GaussianKernel(),
        r = CGEF.Diagnostics.band_energies(u, v, grid, kern, scales),
        f(a, s) = (o = zeros(N, N);
                   CGEF.Filtering.filter_apply!(o, a, CGEF.Filtering.plan_filter(grid, kern, s)); o)
        ū1 = f(u, scales[1]); v̄1 = f(v, scales[1])
        naive1 = 0.5 * sum((u .- ū1) .^ 2 .+ (v .- v̄1) .^ 2) / (N * N)
        Test.@test abs(r.bands[1] - naive1) > 1e-3 * r.bands[1]
    end

    # Where the footprint IS truncated the identity carries a boundary residual, and the residual
    # tracks the filter's own failure to conserve the mean — which is the mechanism, not a fudge.
    let gb = FG.Grids.StructuredGrid(geom, 0.0:dx:(N - 1) * dx, 0.0:dx:(N - 1) * dx, trues(N, N)),
        kern = CGEF.GaussianKernel()
        rb = CGEF.Diagnostics.band_energies(u, v, gb, kern, scales)
        o = zeros(N, N)
        CGEF.Filtering.filter_apply!(o, u, CGEF.Filtering.plan_filter(gb, kern, 6 * dx))
        drift = abs(sum(o) - sum(u)) / sum(abs, u)
        Test.@test 1e-3 < abs(rb.total - Etot) / Etot < 3e-2      # O(ℓ/L), not round-off
        Test.@test abs(rb.total - Etot) / Etot < 5 * drift        # and it is that drift, not more
    end

    # On a MASKED domain both strategies leak, in opposite ways: `ZeroFill` smears energy onto masked
    # cells that then report zero, `Deformable` renormalizes it away and tracks the active-cell
    # integral better. Measured 1.1e-2 against 4.8e-4 — the trade the default makes, asserted.
    let m = trues(N, N); m[12:20, 10:18] .= false
        gm = FG.Grids.StructuredGrid(geom, xp, xp, m; periodic = (true, true))
        area = CGEF.Diagnostics.active_area(gm)
        Em = 0.5 * sum((u[I]^2 + v[I]^2) * FG.Grids.area(gm, Tuple(I)...)
                       for I in CartesianIndices(m) if m[I]) / area
        rz = CGEF.Diagnostics.band_energies(u, v, gm, CGEF.GaussianKernel(), scales;
                                            mask_strategy = CGEF.Filtering.ZeroFill())
        rd = CGEF.Diagnostics.band_energies(u, v, gm, CGEF.GaussianKernel(), scales;
                                            mask_strategy = CGEF.Filtering.Deformable())
        Test.@test abs(rz.total - Em) / Em > 1e-3
        Test.@test abs(rd.total - Em) / Em < 0.2 * (abs(rz.total - Em) / Em)
        Test.@test all(iszero, rz.resolved_map[.!m])
    end

    # A signed kernel breaks the pointwise positivity, hence the non-negative-kernel requirement.
    # The integrated identity still closes: conserving the mean does not need a positive kernel.
    let ho = CGEF.Kernels.HighOrderKernel(; order = 3),
        rh = CGEF.Diagnostics.band_energies(u, v, grid, ho, [8 * dx, 16 * dx]; maps = true)
        Test.@test any(km -> any(<(0), km), rh.band_maps)
        Test.@test rh.total ≈ Etot rtol = 1e-12
    end

    # Argument guards.
    Test.@test_throws ArgumentError CGEF.Diagnostics.band_energies(u, v, grid, CGEF.GaussianKernel(), Float64[])
    Test.@test_throws ArgumentError CGEF.Diagnostics.band_energies(
        u, v, grid, CGEF.GaussianKernel(), [12 * dx, 3 * dx])
end

# `check_setup` must report what will actually run and must not overstate capability. It also must
# not build a plan: a scattered plan at a wide radius can be gigabytes.
Test.@testset "check_setup reports what will run, and every capability it claims" begin
    geom = FG.Geometry.CartesianGeometry()
    N = 48; dx = 1000.0
    xs = 0.0:dx:(N - 1) * dx
    grid = FG.Grids.StructuredGrid(geom, xs, xs, trues(N, N))

    r = CGEF.check_setup(grid, CGEF.TopHatKernel(), 6 * dx)
    Test.@test r isa CGEF.SetupReport
    Test.@test r.grid_size == (N, N)
    Test.@test r.spacing == (dx, dx)
    Test.@test r.cells_per_scale == (6.0, 6.0)
    Test.@test r.resolvable
    Test.@test occursin("prefix-sum", r.engine)
    Test.@test r.boundary_buffer_cells == (3, 3)              # kernel_radius = ℓ/2 = 3 cells
    Test.@test !isempty(sprint(show, MIME"text/plain"(), r))  # the report renders

    # Every capability flag must agree with the function that actually enforces it — a report that
    # drifts from the behaviour is worse than no report.
    for kern in (CGEF.TopHatKernel(), CGEF.GaussianKernel(), CGEF.SharpSpectralKernel(),
                 CGEF.Kernels.SmoothHatKernel(), CGEF.Kernels.HyperGaussianKernel(),
                 CGEF.Kernels.HighOrderKernel(; order = 3))
        rr = CGEF.check_setup(grid, kern, 8 * dx)
        # supports_spectrum ⇔ the default StrictSpectrum policy accepts it. `false` reports "not
        # guaranteed non-negative", not "unreachable".
        Test.@test rr.supports_spectrum == CGEF.Kernels.transfer_monotone(kern)
        u = zeros(N, N); v = zeros(N, N)
        if rr.supports_spectrum
            Test.@test CGEF.Kernels.check_spectrum_kernel(kern) === nothing
            Test.@test CGEF.Diagnostics.gate_spectrum(kern, CGEF.Diagnostics.StrictSpectrum())
        else
            Test.@test_throws ArgumentError CGEF.Kernels.check_spectrum_kernel(kern)
            Test.@test_throws ArgumentError CGEF.Diagnostics.gate_spectrum(
                kern, CGEF.Diagnostics.StrictSpectrum())
        end
        # ForceSpectrum admits every kernel and NoSpectrum refuses every kernel, both unconditionally —
        # that is what makes `supports_spectrum` a statement about the guarantee, not about reachability.
        Test.@test Logging.with_logger(Logging.NullLogger()) do
            CGEF.Diagnostics.gate_spectrum(kern, CGEF.Diagnostics.ForceSpectrum())
        end
        Test.@test !CGEF.Diagnostics.gate_spectrum(kern, CGEF.Diagnostics.NoSpectrum())
        # supports_spectral_method ⇔ spectral_transfer returns rather than throws
        got_spectral = try
            CGEF.Kernels.spectral_transfer(kern, 1 / (8 * dx), 8 * dx); true
        catch; false end
        Test.@test rr.supports_spectral_method == got_spectral
        # supports_flux ⇔ the kernel is non-negative on its support
        if CGEF.Kernels.is_radial(kern)
            rad = CGEF.Kernels.kernel_radius(kern, 8 * dx)
            nonneg = all(t -> CGEF.Kernels.kernel_weight(kern, t, 8 * dx) >= 0,
                         range(0.0, rad; length = 2000))
            Test.@test rr.supports_flux == nonneg
        else
            Test.@test !rr.supports_flux                      # signed limbs
        end
        # A flagged capability always comes with a note explaining it.
        rr.supports_flux || Test.@test any(n -> occursin("NEGATIVE", n), rr.notes)
        rr.supports_spectrum || Test.@test any(n -> occursin("monotone", n), rr.notes)
    end

    # An unresolvable ℓ is flagged rather than silently accepted.
    let bad = CGEF.check_setup(grid, CGEF.TopHatKernel(), 1.5 * dx)
        Test.@test !bad.resolvable
        Test.@test any(n -> occursin("no-op", n), bad.notes)
    end

    # The high-order resolution guard is reported here too, so it is visible BEFORE planning warns.
    let ho = CGEF.check_setup(grid, CGEF.Kernels.HighOrderKernel(), 4 * dx)
        Test.@test any(n -> occursin("limb", n), ho.notes)
        Test.@test occursin("separable", ho.engine)
    end
    let ok = CGEF.check_setup(grid, CGEF.Kernels.HighOrderKernel(), 8 * dx)
        Test.@test !any(n -> occursin("limb", n), ok.notes)
    end

    # The resolved backend is the one that will run, and an impossible request is reported not thrown.
    Test.@test CGEF.check_setup(grid, CGEF.GaussianKernel(), 6 * dx;
                                backend = CGEF.ComputationalBackends.SerialBackend()
                                ).backend_resolved == "SerialBackend"
    let cg = FG.Grids.CurvilinearGrid(geom, [Float64(i) * dx for i in 1:20, j in 1:20],
                                      [Float64(j) * dx for i in 1:20, j in 1:20], trues(20, 20)),
        rc = CGEF.check_setup(cg, CGEF.GaussianKernel(), 6 * dx;
                              backend = CGEF.ComputationalBackends.GPUBackend(KA.CPU()))
        Test.@test rc isa CGEF.SetupReport      # reports, does not throw
    end

    # On a sphere the coordinates are angles and ℓ is in metres. Longitude spacing is the chord-angle
    # distance `2R·asin(cos φ·sin(Δλ/2))`: coarsest at the equator, finest at the most poleward row.
    # Resolvability is judged where the grid is coarsest.
    let R = 6.371e6, d = deg2rad(1.0),
        lon = range(0.0; step = d, length = 360),
        lat = range(deg2rad(-60.0), deg2rad(60.0); step = d / 4),
        sg = FG.Grids.StructuredGrid(FG.Geometry.SphericalGeometry(R), lon, lat; periodic = (true, false)),
        rs = CGEF.check_setup(sg, CGEF.GaussianKernel(), 500e3)
        Test.@test rs.coarsest_spacing[1] ≈ R * d rtol = 1e-12
        Test.@test rs.spacing[1] ≈ 2R * asin(cos(deg2rad(60.0)) * sin(d / 2)) rtol = 1e-12
        Test.@test rs.spacing[2] ≈ R * d / 4 rtol = 1e-12
        Test.@test rs.coarsest_spacing[2] ≈ R * d / 4 rtol = 1e-12
        Test.@test rs.cells_per_scale[1] ≈ 500e3 / (R * d) rtol = 1e-12
        Test.@test rs.resolvable
        # 150 km is 2.7 of the 60° row's longitude cells but 1.35 of the equator's.
        Test.@test !CGEF.check_setup(sg, CGEF.GaussianKernel(), 150e3).resolvable
    end

    # A node set has no axes, so it must SAY it could not check rather than report all-clear.
    let nn = 16,
        xg = [Float64(i - 1) * dx for i in 1:nn, j in 1:nn],
        yg = [Float64(j - 1) * dx for i in 1:nn, j in 1:nn],
        ug = FG.Grids.UnstructuredGrid(geom, vec(xg), vec(yg), trues(nn * nn);
                                       k = 6, areas = fill(dx^2, nn * nn)),
        ru = CGEF.check_setup(ug, CGEF.GaussianKernel(), 6 * dx)
        Test.@test ru.spacing == ()
        Test.@test any(n -> occursin("no axes", n), ru.notes)
        # A flat-cell grid's real-space engine is the CSR gather over its own ball query. A structured
        # grid with a nonuniform axis takes the scattered per-point footprint instead.
        Test.@test occursin("node CSR", ru.engine)
    end

    Test.@test_throws ArgumentError CGEF.check_setup(grid, CGEF.TopHatKernel(), 0.0)
    Test.@test_throws ArgumentError CGEF.check_setup(grid, CGEF.TopHatKernel(), -1.0)
end


# `check_setup` exists so a user does not have to read `src/` to learn which engine runs. That makes
# its engine string a claim about dispatch, and a claim about dispatch has to be checked against
# dispatch — otherwise it drifts silently, which is how it came to describe a spherical Gaussian (the
# package's most expensive engine) as "separable".
Test.@testset "check_setup names the engine that is actually constructed" begin
    dx = 1_000.0
    ℓ = 8 * dx
    geom = FG.Geometry.CartesianGeometry()
    sgeom = FG.Geometry.SphericalGeometry(6.371e6)
    R2 = 0.0:dx:19*dx                                   # Range axis  => uniform-axis engines
    V2 = collect(R2) .+ [0.2 * dx * sin(3.3i) for i in 1:20]   # Vector axis => scattered engines
    R3 = 0.0:dx:9*dx
    lon = range(0.0; step = 2π / 24, length = 24)
    lat = range(deg2rad(-60.0); stop = deg2rad(60.0), length = 16)

    # Every real-space footprint type, and the phrase the report must use for it.
    expected = [
        (CGEF.Filtering.PrefixSumTopHatPlan,   "O(N·w_y)"),
        (CGEF.Filtering.PrefixSumTopHat3DPlan, "O(N·w_y·w_z)"),
        (CGEF.Filtering.SeparableFootprint,    "separable two-pass"),
        (CGEF.Filtering.SeparableFootprintND,  "separable N-pass"),
        (CGEF.Filtering.FilterFootprint,       "banded footprint"),
        (CGEF.Filtering.ScatteredFilterPlan,   "scattered per-point"),
        (CGEF.Filtering.FilterFootprintND,     "N-dimensional ball"),
        (CGEF.Filtering.NDScatteredFilterPlan, "scattered N-dimensional"),
        (CGEF.Filtering.NodeFilterPlan,        "node CSR"),
    ]
    phrase_for(fp) = begin
        hits = [p for (Tfp, p) in expected if fp isa Tfp]
        length(hits) == 1 || error("footprint $(typeof(fp)) matches $(length(hits)) engine phrases")
        only(hits)
    end

    RS = CGEF.Filtering.RealSpace()
    cases = Any[
        # (name, grid, kernel, method)
        ("2D Cartesian Range   / tophat",   FG.Grids.StructuredGrid(geom, R2, R2, trues(20, 20)),  CGEF.TopHatKernel(),   RS),
        ("2D Cartesian Vector  / tophat",   FG.Grids.StructuredGrid(geom, V2, V2, trues(20, 20)),  CGEF.TopHatKernel(),   RS),
        ("2D spherical        / tophat",    FG.Grids.StructuredGrid(sgeom, lon, lat, trues(24, 16)), CGEF.TopHatKernel(), RS),
        ("2D Cartesian Range   / gaussian", FG.Grids.StructuredGrid(geom, R2, R2, trues(20, 20)),  CGEF.GaussianKernel(), RS),
        ("2D Cartesian Vector  / gaussian", FG.Grids.StructuredGrid(geom, V2, V2, trues(20, 20)),  CGEF.GaussianKernel(), RS),
        # The case that motivated this testset: separable KERNEL, non-separable METRIC.
        ("2D spherical        / gaussian",  FG.Grids.StructuredGrid(sgeom, lon, lat, trues(24, 16)), CGEF.GaussianKernel(), RS),
        ("2D Cartesian Range   / sharp",    FG.Grids.StructuredGrid(geom, R2, R2, trues(20, 20)),  CGEF.SharpSpectralKernel(), RS),
        ("3D Cartesian Range   / gaussian", FG.Grids.StructuredGrid(geom, R3, R3, R3, trues(10, 10, 10)), CGEF.GaussianKernel(), RS),
        ("3D Cartesian Range   / tophat",   FG.Grids.StructuredGrid(geom, R3, R3, R3, trues(10, 10, 10)), CGEF.TopHatKernel(), RS),
        ("1D Cartesian Range   / gaussian", FG.Grids.StructuredGrid(geom, R2, trues(20)),          CGEF.GaussianKernel(), RS),
        ("curvilinear         / gaussian",  FG.Grids.CurvilinearGrid(geom,
                                              [Float64(i - 1) * dx for i in 1:20, j in 1:20],
                                              [Float64(j - 1) * dx for i in 1:20, j in 1:20],
                                              trues(20, 20)),                                      CGEF.GaussianKernel(), RS),
    ]

    for (name, g, k, m) in cases
        plan = CGEF.Filtering.plan_filter(g, k, ℓ; method = m,
                                          backend = CGEF.ComputationalBackends.SerialBackend())
        reported = CGEF.Pipeline._engine_class(g, k, m)
        want = phrase_for(plan.footprint)
        Test.@test occursin(want, reported) ||
            error("$name: built $(nameof(typeof(plan.footprint))) but reported \"$reported\"")
    end

    # Flat-cell grids. One CSR engine over the grid's own ball query serves every architecture that
    # names a cell by a single index, so the report must name that engine for each of them.
    let nn = 12,
        xg = [Float64(i - 1) * dx for i in 1:nn, j in 1:nn],
        yg = [Float64(j - 1) * dx for i in 1:nn, j in 1:nn],
        R = 6.371e6,
        flat = Any[
            (FG.Grids.UnstructuredGrid(geom, vec(xg), vec(yg), trues(nn * nn);
                                       k = 6, areas = fill(dx^2, nn * nn)), ℓ),
            (FG.Grids.RingGrid(FG.Geometry.SphericalGeometry(R),
                               FG.SphericalSampling.ReducedGaussianSampling([8, 12, 12, 8])), 0.3R),
            (FG.Grids.CubedSphereGrid(4), 0.3R),
            (FG.Grids.HEALPixGrid(2), 0.3R),
            (FG.Grids.IcosahedralGrid(1), 0.3R),
            (FG.Grids.YinYangGrid(12, 6), 0.3R),
        ]
        for (g, s) in flat
            plan = CGEF.Filtering.plan_filter(g, CGEF.GaussianKernel(), s; method = RS,
                                              backend = CGEF.ComputationalBackends.SerialBackend())
            Test.@test occursin(phrase_for(plan.footprint),
                                CGEF.Pipeline._engine_class(g, CGEF.GaussianKernel(), RS))
        end
    end

    # The complexity the report states must name the width, not claim linearity in N alone: the
    # prefix-sum engine is O(N·w_y), and reporting it as "O(N)" told users the filter width was free.
    let g = FG.Grids.StructuredGrid(geom, R2, R2, trues(20, 20))
        e = CGEF.Pipeline._engine_class(g, CGEF.TopHatKernel(), RS)
        Test.@test occursin("O(N·w", e)
        Test.@test !occursin("exact O(N)", e)
    end

    # `AutoMethod` reaches two transform-based evaluators of the same real-space convolution. They are
    # engines, not `Spectral()`, so the report must name them rather than the direct sum they replace.
    AM = CGEF.Filtering.AutoMethod()
    let g = FG.Grids.StructuredGrid(geom, R2, R2, trues(20, 20))   # bounded => padded, not Spectral
        plan = CGEF.Filtering.plan_filter(g, CGEF.SharpSpectralKernel(), ℓ; method = AM,
                                          backend = CGEF.ComputationalBackends.SerialBackend())
        rep = CGEF.Pipeline._engine_class(g, CGEF.SharpSpectralKernel(), AM)
        Test.@test occursin("padded-FFT", rep) ==
                   (nameof(typeof(plan.footprint)) === :PaddedFFTFootprint)
    end
    let g = FG.Grids.StructuredGrid(sgeom, lon, lat, trues(24, 16))
        plan = CGEF.Filtering.plan_filter(g, CGEF.GaussianKernel(), ℓ; method = AM,
                                          backend = CGEF.ComputationalBackends.SerialBackend())
        rep = CGEF.Pipeline._engine_class(g, CGEF.GaussianKernel(), AM)
        Test.@test occursin("zonal-FFT", rep) ==
                   (nameof(typeof(plan.footprint)) === :ZonalFFTFootprint)
        # A top-hat must NOT be diverted there: its prefix-sum engine is exact and cheaper.
        Test.@test !occursin("zonal-FFT", CGEF.Pipeline._engine_class(g, CGEF.TopHatKernel(), AM))
        # A device plan keeps the direct engine, and the report says so.
        gpu = CGEF.ComputationalBackends.GPUBackend(KA.CPU())
        pg = CGEF.Filtering.plan_filter(g, CGEF.GaussianKernel(), ℓ; method = AM, backend = gpu)
        Test.@test nameof(typeof(pg.footprint)) !== :ZonalFFTFootprint
        Test.@test !occursin("zonal-FFT",
                             CGEF.check_setup(g, CGEF.GaussianKernel(), ℓ; method = AM, backend = gpu).engine)
    end
end

Test.@testset "AutoMethod's transform engines equal the direct engine on every host path" begin
    # The padded FFT of the sampled kernel and the transform along the longitude ring evaluate the
    # direct engine's convolution, so every host apply — one field, a batch, threaded — and every
    # diagnostic built on them agrees with it to round-off.
    AM, RS = CGEF.Filtering.AutoMethod(), CGEF.Filtering.RealSpace()
    SER = CGEF.ComputationalBackends.SerialBackend()
    THR = CGEF.ComputationalBackends.ThreadedBackend()
    dx = 1000.0
    xs = 0.0:dx:(39 * dx)
    cmask = trues(40, 40); cmask[5:9, 12:20] .= false
    gc = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), xs, xs, cmask)       # bounded
    # Periodic in both axes, and in one: the transform wraps where the grid does and pads elsewhere.
    gpp = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), xs, xs, cmask; periodic = (true, true))
    gpx = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), xs, xs, cmask; periodic = (true, false))
    R = 6.371e6
    lon = range(0.0; step = 2π / 48, length = 48)
    lat = range(deg2rad(-70.0); stop = deg2rad(70.0), length = 24)
    smask = trues(48, 24); smask[10:16, 8:12] .= false
    gs = FG.Grids.StructuredGrid(FG.Geometry.SphericalGeometry(R), lon, lat, smask;
                                 periodic = (true, false))
    fieldset(n1, n2) = ntuple(q -> [sin(2π * q * i / n1) * cos(2π * (q + 1) * j / n2) + 0.3q
                                    for i in 1:n1, j in 1:n2], 3)
    cases = ((gc, CGEF.Kernels.HyperGaussianKernel(), 6dx, :PaddedFFTFootprint),
             (gpp, CGEF.Kernels.HyperGaussianKernel(), 30dx, :PaddedFFTFootprint),
             (gpx, CGEF.Kernels.SmoothHatKernel(), 30dx, :PaddedFFTFootprint),
             (gs, CGEF.GaussianKernel(), 900e3, :ZonalFFTFootprint))
    for (g, k, ℓ, engine) in cases, strat in (CGEF.Filtering.ZeroFill(), CGEF.Filtering.Deformable())
        fs = fieldset(size(FG.Grids.mask(g))...)
        pr = CGEF.Filtering.plan_filter(g, k, ℓ; method = RS, backend = SER, mask_strategy = strat)
        ref = ntuple(i -> CGEF.Filtering.filter_apply!(zeros(size(fs[i])), fs[i], pr), 3)
        batches = map((SER, THR)) do b
            p = CGEF.Filtering.plan_filter(g, k, ℓ; method = AM, backend = b, mask_strategy = strat)
            Test.@test nameof(typeof(p.footprint)) === engine
            outs = ntuple(_ -> zeros(size(fs[1])), 3)
            CGEF.Filtering.filter_apply_batch!(outs, fs, p)
            Test.@test CGEF.Filtering.filter_apply!(zeros(size(fs[1])), fs[1], p) == outs[1]
            for i in 1:3
                Test.@test outs[i] ≈ ref[i] rtol = 1e-10
            end
            outs
        end
        Test.@test batches[1] == batches[2]
    end

    # Through a diagnostic: `coarse_grain` filters every field of `Π` through `filter_apply_batch!`.
    for (g, k, scales) in ((gc, CGEF.Kernels.HyperGaussianKernel(), [4dx, 7dx]),
                           (gs, CGEF.GaussianKernel(), [600e3, 1200e3])), b in (SER, THR)
        u, v = fieldset(size(FG.Grids.mask(g))...)
        sweep(m) = CGEF.coarse_grain(u, v, g; scales, kernel = k, method = m, backend = b,
                                     spectrum = CGEF.Diagnostics.NoSpectrum())
        ra, rr = sweep(AM), sweep(RS)
        Test.@test ra.Π ≈ rr.Π rtol = 1e-9
        Test.@test ra.cumulative_energy ≈ rr.cumulative_energy rtol = 1e-12
    end
end

# The filter's definition summed directly: every source cell and, on a periodic Cartesian direction,
# every image of it inside the support, at the geometry's own distance, weighted `G(d)·A` and normalized
# by the weight that falls on the grid. A spherical longitude identifies, so its cells come once.
function _cgef_direct_filter(grid, kernel, ℓ, f)
    geo = FG.Grids.grid_geometry(grid)
    cart = geo isa FG.Geometry.CartesianGeometry
    sz = FG.Grids.size_tuple(grid)
    N = length(sz)
    rad = CGEF.Kernels.kernel_radius(kernel, ℓ)
    tiles = ntuple(d -> cart && FG.Grids.isperiodic(grid, d), N)
    P = ntuple(d -> tiles[d] ? FG.Grids.period(grid, d) : 0.0, N)
    M = ntuple(d -> tiles[d] ? ceil(Int, rad / P[d]) + 1 : 0, N)
    out = zeros(sz)
    for I in CartesianIndices(sz)
        xi = FG.Grids.coords(grid, Tuple(I)...)
        num = 0.0
        den = 0.0
        for J in CartesianIndices(sz), m in CartesianIndices(ntuple(d -> (-M[d]):M[d], N))
            xj = FG.Grids.coords(grid, Tuple(J)...)
            xm = ntuple(d -> xj[d] + m[d] * P[d], N)
            dist = FG.Geometry.distance(geo, xi, xm)
            dist <= rad || continue
            w = CGEF.Kernels.kernel_weight(kernel, dist, ℓ) * FG.Grids.measure(grid, Tuple(J)...)
            num += w * f[J]
            den += w
        end
        out[I] = num / den
    end
    return out
end

_cgef_relerr(a, b) = maximum(abs, a .- b) / maximum(abs, b)

# A support wider than half the domain reaches cells through several images on a periodic Cartesian
# direction, and more than half the rows on a bounded one. Every real-space engine must still equal the
# definition, on every host and device path.
Test.@testset "Real-space engines equal the direct sum at supports past half the domain" begin
    SER = CGEF.ComputationalBackends.SerialBackend()
    THR = CGEF.ComputationalBackends.ThreadedBackend()
    DEV = CGEF.ComputationalBackends.GPUBackend(KA.CPU())
    cart = FG.Geometry.CartesianGeometry()
    TH, SH = CGEF.TopHatKernel(), CGEF.Kernels.SmoothHatKernel()
    field(sz) = [sin(1.3 * I[1] + 0.7 * I[2] + (length(sz) > 2 ? 0.4 * I[3] : 0.0)) + 0.2
                 for I in CartesianIndices(sz)]
    function check(g, k, ℓ, engine; backends = (SER, THR, DEV), method = CGEF.Filtering.RealSpace(), kw...)
        f = field(FG.Grids.size_tuple(g))
        ref = _cgef_direct_filter(g, k, ℓ, f)
        for b in backends
            p = CGEF.Filtering.plan_filter(g, k, ℓ; method = method, backend = b, kw...)
            # A device plan holds its footprint's resident copy, which wraps the host plan.
            fp = b isa CGEF.ComputationalBackends.GPUBackend ? p.footprint.fp : p.footprint
            Test.@test nameof(typeof(fp)) === engine
            Test.@test _cgef_relerr(CGEF.Filtering.filter_apply!(zero(f), f, p), ref) < 1e-11
        end
    end

    # Periodic in both directions, periods 16 and 8.4; ℓ/Lx = 0.3, 0.6, 1.2, 2.55. No cell sits on the
    # support boundary: `i² + 0.49 j²` never equals `rad²` for these radii.
    gp = FG.Grids.StructuredGrid(cart, 0.0:1.0:15.0, 0.0:0.7:7.7; periodic = (true, true))
    for ℓ in (4.8, 9.6, 19.2, 40.8)
        check(gp, TH, ℓ, :PrefixSumTopHatPlan)
        check(gp, SH, ℓ, :FilterFootprint; backends = (SER, THR))
        check(gp, SH, ℓ, :PaddedFFTFootprint; backends = (SER, THR), method = CGEF.Filtering.AutoMethod())
    end
    # A stretched periodic axis takes the two-pointer walk over the tripled coordinates.
    xs = cumsum([0.0, 1.1, 0.8, 1.3, 0.9, 1.2, 0.7, 1.0, 1.1, 0.95, 1.05, 1.15, 0.85])
    gs = FG.Grids.StructuredGrid(cart, xs, 0.0:0.7:7.7; periodic = (true, true), period = (13.0, 8.4))
    # Radii off every sum of consecutive gaps, so no cell sits on the support boundary.
    for ℓ in (5.1462, 14.3874, 29.9226)
        check(gs, TH, ℓ, :PrefixSumTopHatPlan)
    end
    # Bounded, with the support reaching more than half the rows.
    gb = FG.Grids.StructuredGrid(cart, 0.0:1.0:15.0, 0.0:0.7:2.8)
    for ℓ in (4.8, 9.6)
        check(gb, TH, ℓ, :PrefixSumTopHatPlan)
        check(gb, SH, ℓ, :FilterFootprint; backends = (SER, THR))
        check(gb, SH, ℓ, :PaddedFFTFootprint; backends = (SER, THR), method = CGEF.Filtering.AutoMethod())
    end

    # Periodic volume: whole turns of axis 1 and images along axes 2 and 3.
    g3 = FG.Grids.StructuredGrid(cart, 0.0:1.0:5.0, 0.0:0.7:2.8, 0.0:0.9:2.7; periodic = (true, true, true))
    for ℓ in (4.8, 13.3)
        check(g3, TH, ℓ, :PrefixSumTopHat3DPlan)
        check(g3, SH, ℓ, :FilterFootprintND; backends = (SER, THR))
    end
    # A stretched volume streams its neighbourhoods through the grid's ball query, or caches them.
    g3s = FG.Grids.StructuredGrid(cart, cumsum([0.0, 1.1, 0.8, 1.3, 0.9, 1.2]), 0.0:0.7:2.8, 0.0:0.9:2.7;
                                  periodic = (true, true, true), period = (6.2, 3.5, 3.6))
    for ℓ in (4.8, 13.3), cs in (CGEF.Filtering.NeverCache(), CGEF.Filtering.AlwaysCache())
        check(g3s, SH, ℓ, :NDScatteredFilterPlan; backends = (SER, THR), cache_strategy = cs)
    end

    # Sphere: latitude reaching more than half the rows, and a regional box whose longitude support is
    # wider than half its columns (2° at 80° is 39 km, so 500 km spans 13 of 20 columns).
    R = 6.371e6
    sph = FG.Geometry.SphericalGeometry(R)
    gw = FG.Grids.StructuredGrid(sph, deg2rad.(0.0:10.0:350.0), deg2rad.(-80.0:10.0:80.0);
                                 periodic = (true, false))
    check(gw, TH, 2 * 12_000e3, :PrefixSumTopHatPlan)
    # Range axes take the banded footprint; the same box as vectors takes the scattered one.
    gr = FG.Grids.StructuredGrid(sph, range(0.0, deg2rad(38.0); length = 20),
                                 range(deg2rad(60.0), deg2rad(80.0); length = 11))
    grv = FG.Grids.StructuredGrid(sph, deg2rad.(0.0:2.0:38.0), deg2rad.(60.0:2.0:80.0))
    Test.@test !FG.Grids.isperiodic(gr, 1) && !FG.Grids.isperiodic(grv, 1)
    check(gr, SH, 1000e3, :FilterFootprint; backends = (SER, THR))
    check(grv, SH, 1000e3, :ScatteredFilterPlan; backends = (SER, THR))
    check(gr, TH, 1000e3, :PrefixSumTopHatPlan)
end

Test.@testset "Sphere: the energy diagnostics filter the planetary velocity, as the flux does" begin
    SER = CGEF.ComputationalBackends.SerialBackend()
    R = 6.371e6
    lon = range(0.0; step = 2π / 144, length = 144)
    lat = range(deg2rad(-75.0); stop = deg2rad(75.0), length = 61)
    g = FG.Grids.StructuredGrid(FG.Geometry.SphericalGeometry(R), lon, lat, trues(144, 61);
                                periodic = (true, false))
    k = CGEF.GaussianKernel()
    scales = [600e3, 1200e3]
    u = [cos(φ) + 0.3sin(2λ) * cos(φ)^2 for λ in lon, φ in lat]
    v = [0.2cos(3λ) * cos(φ) for λ in lon, φ in lat]

    r = CGEF.coarse_grain(u, v, g; scales, kernel = k, backend = SER)
    Test.@test CGEF.Diagnostics.cumulative_energy(u, v, nothing, g, k, scales; backend = SER) ≈
               r.cumulative_energy rtol = 1e-13
    Test.@test CGEF.Diagnostics.filtering_spectrum(u, v, nothing, g, k, scales; backend = SER)[2] ≈
               r.filtering_spectrum rtol = 1e-12

    # Solid-body rotation about the polar axis: its planetary velocity Ω×r is a degree-1 field, which a
    # radial kernel maps to a multiple of itself (Funk–Hecke), so the filtered field stays `∝ cos φ` and
    # `resolved_map / (½cos²φ)` is one constant across latitude. Rows within |φ| ≤ 50° sit more than one
    # kernel radius inside the ±75° boundary. Filtering `u = cos φ` as a scalar spreads that ratio by
    # 4e-3 here.
    ub = [cos(φ) for λ in lon, φ in lat]
    c2 = [0.5cos(φ)^2 for λ in lon, φ in lat]
    rows = findall(φ -> abs(φ) <= deg2rad(50.0) + 1e-9, lat)
    ratio = CGEF.Diagnostics.band_energies(ub, zero(ub), g, k, [1200e3]; backend = SER).resolved_map ./ c2
    Test.@test (maximum(ratio[:, rows]) - minimum(ratio[:, rows])) < 1e-8 * ratio[1, rows[1]]
end

_cgef_alloc_favre(ws, u, v, ρ, P, g, k, ℓ, fp, dp) =
    @allocated CGEF.Diagnostics.compressible_flux!(ws, u, v, ρ, P, g, k, ℓ;
        filter_plan = fp, deriv_plan = dp, backend = CGEF.ComputationalBackends.SerialBackend())

# The variable-density (Favre) budget. The property that matters is that baropycnal work `Λ` is a
# SEPARATE cross-scale term and cannot be absorbed into the pressure dilatation — the standard trap is
# to write `P̄∇·ũ` instead of `P̄∇·ū`, which silently swallows it.
Test.@testset "Favre budget: Π, Λ and P̄∇·ū are three distinct terms" begin
    geom = FG.Geometry.CartesianGeometry()
    N = 64; dx = 1000.0
    xp = range(0.0, dx * N; length = N + 1)[1:N]
    grid = FG.Grids.StructuredGrid(geom, xp, xp; periodic = (true, true))
    ker = CGEF.GaussianKernel(); ℓ = 8 * dx
    SER = CGEF.ComputationalBackends.SerialBackend()
    u = [sin(2π * 3 * i / N) * cos(2π * 2 * j / N) + 0.4cos(2π * 5 * i / N) for i in 1:N, j in 1:N]
    v = [cos(2π * 2 * i / N) * sin(2π * 4 * j / N) + 0.3sin(2π * 6 * j / N) for i in 1:N, j in 1:N]
    ρ = [1.0 + 0.3sin(2π * 2 * i / N) * cos(2π * 3 * j / N) for i in 1:N, j in 1:N]
    P = [10.0 + 2.0cos(2π * 3 * i / N) * sin(2π * 2 * j / N) for i in 1:N, j in 1:N]

    r = CGEF.Diagnostics.compressible_flux(u, v, ρ, P, grid, ker, ℓ; backend = SER)
    for f in (r.Π, r.Λ, r.pressure_dilatation, r.ρ̄, r.P̄, r.ũ, r.ṽ)
        Test.@test all(isfinite, f)
    end
    Test.@test maximum(abs, r.Π) > 0 && maximum(abs, r.Λ) > 0

    # Each term reassembled independently from primitives — every definition checked, not just the sum.
    let pl = CGEF.Filtering.plan_filter(grid, ker, ℓ; backend = SER),
        f(a) = (o = zeros(N, N); CGEF.Filtering.filter_apply!(o, a, pl); o),
        d(a, dir) = (o = zeros(N, N);
                     dir == 1 ? CGEF.Derivatives.ddx!(o, a, grid) : CGEF.Derivatives.ddy!(o, a, grid); o)
        ρ̄ = f(ρ); P̄ = f(P); ū = f(u); v̄ = f(v)
        ũ = f(ρ .* u) ./ ρ̄; ṽ = f(ρ .* v) ./ ρ̄
        Test.@test r.ρ̄ == ρ̄ && r.P̄ == P̄ && r.ũ == ũ && r.ṽ == ṽ
        τxx = f(ρ .* u .* u) ./ ρ̄ .- ũ .^ 2
        τxy = f(ρ .* u .* v) ./ ρ̄ .- ũ .* ṽ
        τyy = f(ρ .* v .* v) ./ ρ̄ .- ṽ .^ 2
        # τ̄(ρ,u_j) is the UNWEIGHTED subscale mass flux — the framework needs both moment types.
        mx = f(ρ .* u) .- ρ̄ .* ū; my = f(ρ .* v) .- ρ̄ .* v̄
        Πr = -ρ̄ .* (d(ũ, 1) .* τxx .+ (d(ũ, 2) .+ d(ṽ, 1)) .* τxy .+ d(ṽ, 2) .* τyy)
        Λr = (d(P̄, 1) .* mx .+ d(P̄, 2) .* my) ./ ρ̄
        PDr = P̄ .* (d(ū, 1) .+ d(v̄, 2))
        Test.@test r.Π == Πr
        Test.@test r.Λ == Λr
        Test.@test r.pressure_dilatation == PDr

        # THE TRAP: the dilatation must use the unweighted divergence. `P̄∇·ũ` is a different field, and
        # the difference is exactly what would swallow Λ.
        PDw = P̄ .* (d(ũ, 1) .+ d(ṽ, 2))
        Test.@test maximum(abs, PDw .- PDr) > 0.01 * maximum(abs, PDr)
    end

    # Constant density collapses the whole construction: Favre == unweighted, so Λ ≡ 0 and Π reduces to
    # ρ times the incompressible flux. This is the limit that says the weighting is applied correctly.
    let ρc = fill(2.5, N, N),
        rc = CGEF.Diagnostics.compressible_flux(u, v, ρc, P, grid, ker, ℓ; backend = SER),
        Πi = zeros(N, N)
        CGEF.Diagnostics.compute_Π!(Πi, u, v, nothing, grid, ker, ℓ; backend = SER)
        Test.@test maximum(abs, rc.Λ) < 1e-12 * maximum(abs, rc.Π)
        Test.@test maximum(abs, rc.Π ./ 2.5 .- Πi) < 1e-12 * maximum(abs, Πi)
    end

    # Λ is driven by the density gradient, and survives in BOTH gradient configurations — its leading
    # form has a strain-generation part as well as a baroclinic one, so aligning ∇ρ with ∇P does not
    # kill it (measured: the aligned case is in fact the larger of the two here). What kills it is a
    # uniform density, checked above. This is the assertion that stops Λ being quietly dropped.
    let ρg = [1.0 + 0.4 * (i / N) for i in 1:N, j in 1:N],              # ∇ρ along x
        Py = [10.0 + 3.0 * (j / N) for i in 1:N, j in 1:N],             # ∇P along y  (crossed)
        Px = [10.0 + 3.0 * (i / N) for i in 1:N, j in 1:N],             # ∇P along x  (aligned)
        rc1 = CGEF.Diagnostics.compressible_flux(u, v, ρg, Py, grid, ker, ℓ; backend = SER),
        rc2 = CGEF.Diagnostics.compressible_flux(u, v, ρg, Px, grid, ker, ℓ; backend = SER)
        Test.@test maximum(abs, rc1.Λ) > 0
        Test.@test maximum(abs, rc2.Λ) > 0
        # ... and Λ is not negligible against the term it is often folded into.
        Test.@test maximum(abs, rc2.Λ) > 1e-3 * maximum(abs, rc2.pressure_dilatation)
    end

    # `favre_filter!` is the exposed building block and must agree with the definition.
    let pl = CGEF.Filtering.plan_filter(grid, ker, ℓ; backend = SER),
        ρ̄ = zeros(N, N), tmp = zeros(N, N), out = zeros(N, N)
        CGEF.Filtering.filter_apply!(ρ̄, ρ, pl)
        CGEF.Diagnostics.favre_filter!(out, tmp, u, ρ, ρ̄, pl)
        ref = zeros(N, N); CGEF.Filtering.filter_apply!(ref, ρ .* u, pl)
        Test.@test out == ref ./ ρ̄
    end

    # Zero-allocation reuse, and the in-place form agrees bit-for-bit with the allocating one.
    let ws = CGEF.Diagnostics.FavreWorkspace(grid),
        pl = CGEF.Filtering.plan_filter(grid, ker, ℓ; backend = SER),
        dp = CGEF.Derivatives.StencilPlan(grid)
        CGEF.Diagnostics.compressible_flux!(ws, u, v, ρ, P, grid, ker, ℓ;
                                            filter_plan = pl, deriv_plan = dp, backend = SER)
        Test.@test ws.Π == r.Π && ws.Λ == r.Λ && ws.PD == r.pressure_dilatation
        Test.@test _cgef_alloc_favre(ws, u, v, ρ, P, grid, ker, ℓ, pl, dp) == 0
    end

    # ρ must be strictly positive — Favre filtering divides by ρ̄.
    Test.@test_throws ArgumentError CGEF.Diagnostics.compressible_flux(u, v, fill(-1.0, N, N), P, grid, ker, ℓ)
    Test.@test_throws ArgumentError CGEF.Diagnostics.compressible_flux(u, v, zeros(N, N), P, grid, ker, ℓ)
    Test.@test_throws DimensionMismatch CGEF.Diagnostics.compressible_flux(u, v, ρ, zeros(N + 1, N), grid, ker, ℓ)
end
