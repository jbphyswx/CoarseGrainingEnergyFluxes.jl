"""
Generate static figure assets for CoarseGrainingEnergyFluxes.jl docs + README.

Run from this directory:
    julia --project=. generate_assets.jl

Outputs PNGs to ../src/assets/ (served by Documenter and embedded in README.md). Every figure is
built from the package's public API under the qualified-import policy.
"""

using CairoMakie: CairoMakie as MK
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF
using DelaunayTriangulation: DelaunayTriangulation   # triggers CGEF's planar Voronoi-area extension
using FFTW: FFTW
using FINUFFT: FINUFFT        # triggers CGEF's scattered-Cartesian (FINUFFT) extension
using FastSphericalHarmonics: FastSphericalHarmonics as FSH
using NUFSHT: NUFSHT          # triggers CGEF's scattered-spherical (NUFSHT) extension
using NearestNeighbors: NearestNeighbors   # triggers CGEF's k-d tree neighbor-search extension
using Random: Random
using Statistics: Statistics

const ASSETS = joinpath(@__DIR__, "..", "src", "assets")
mkpath(ASSETS)

MK.set_theme!(
    MK.Theme(;
        fontsize = 15,
        figure_padding = (12, 34, 12, 12),   # extra right margin so colorbar ticklabels aren't clipped
        Axis = (; titlesize = 16, titlefont = :bold, xgridvisible = false, ygridvisible = false),
        Colorbar = (; ticklabelsize = 11, labelsize = 13),
    ),
)

# Diverging map for signed fields/flux: white at 0, warm = forward cascade (Π>0), cool = inverse.
const CASCADE = :balance
const FIELDMAP = :balance
# Sequential map for non-negative magnitudes (speed): WHITE at 0, deepening with magnitude.
const SPEEDMAP = MK.cgrad([:white, :skyblue3, :midnightblue])

save_fig(name, fig; px = 2) = (p = joinpath(ASSETS, name); MK.save(p, fig; px_per_unit = px); println("saved ", name))

# Symmetric color limit (robust: ignore the noisy domain boundary, clip to a quantile).
function symclim(A; interior = 0, q = 1.0)
    v = interior > 0 ? vec(@view A[interior+1:end-interior, interior+1:end-interior]) : vec(A)
    m = q < 1 ? Statistics.quantile(abs.(v), q) : maximum(abs, v)
    return m == 0 ? 1.0 : m
end

# ─────────────────────────────────────────────────────────────────────────────
# Canonical fields (fixed seed) reused across figures.
#
# (1) eddy_noise_flow: a non-divergent flow whose VORTICITY has a spectral bump at a chosen eddy scale
#     (~18 km) plus a small-scale noise tail — coherent eddies you can watch decompose, not "noise→0".
# (2) fractal_field:   a deterministic sum of sinusoids at large/medium/small scales — the clearest
#     possible "filtering coarsens the field" demonstration.
# ─────────────────────────────────────────────────────────────────────────────

step_of(xs) = xs[2] - xs[1]
# The synthetic fields are periodic, so filter on a periodic grid — the footprint wraps and there are
# no domain-edge artifacts in the filtered fields or in Π.
cartgrid(xs) = CGEF.StructuredGrid(CGEF.CartesianGeometry(step_of(xs), step_of(xs)), xs, xs,
    trues(length(xs), length(xs)); periodic = (true, true))

function eddy_noise_flow(; N = 192, dx = 1000.0, seed = 7, λ_eddy_km = 18.0, λ_large_km = 55.0,
        width = 2.6, amp_large = 0.8, noise = 0.14, n_noise_max = 60.0)
    Random.seed!(seed)
    L = N * dx
    n_eddy = (L / 1e3) / λ_eddy_km                  # eddy wavelength → cycles per domain
    n_large = (L / 1e3) / λ_large_km                # large-scale organization
    kfreq = 2π .* FFTW.fftfreq(N, 1 / dx)           # angular wavenumbers (rad/m)
    KX = [kfreq[i] for i in 1:N, _ in 1:N]
    KY = [kfreq[j] for _ in 1:N, j in 1:N]
    K2 = KX .^ 2 .+ KY .^ 2
    # Prescribe the vorticity spectrum: a large-scale bump + an eddy bump + a weak broadband tail, so
    # there is genuine energy at large AND small scales (filtering reveals structure at every ℓ).
    ωh = zeros(ComplexF64, N, N)
    for j in 1:N, i in 1:N
        n = sqrt(kfreq[i]^2 + kfreq[j]^2) * L / (2π)
        bump = exp(-((n - n_eddy) / width)^2) + amp_large * exp(-((n - n_large) / 1.5)^2)
        tail = (0 < n <= n_noise_max) ? noise * n^(-1.0) : 0.0
        ωh[i, j] = (bump + tail) * cis(2π * rand())
    end
    ωh[1, 1] = 0
    ψh = ωh ./ K2          # invert the Laplacian: ψ̂ = ω̂ / k²
    ψh[1, 1] = 0.0im       # kill the (undefined) DC mode
    u = real(FFTW.ifft(im .* KY .* ψh))
    v = real(FFTW.ifft(-im .* KX .* ψh))
    ω = real(FFTW.ifft(ωh))
    s = 1.0 / Statistics.std(u)
    return (; xs = collect(0.0:dx:(N - 1) * dx), u = u .* s, v = v .* s, ω = ω .* s)
end

function fractal_field(; N = 192, dx = 1000.0)
    xs = collect(0.0:dx:(N - 1) * dx)
    L = N * dx          # period = N·dx so the integer-cycle pattern is exactly periodic on the grid
    f = [1.0 * sin(2π * 2 * x / L) * cos(2π * 2 * y / L) +
         0.6 * sin(2π * 6 * x / L) * cos(2π * 6 * y / L) +
         0.35 * sin(2π * 14 * x / L) * cos(2π * 14 * y / L) for x in xs, y in xs]
    # a matching divergence-free velocity (so the same pattern can drive a flux figure)
    u = [ 0.6 * sin(2π * 2 * x / L) * cos(2π * 6 * y / L) + 0.3 * sin(2π * 10 * x / L) * cos(2π * 4 * y / L) for x in xs, y in xs]
    v = [-0.6 * cos(2π * 6 * x / L) * sin(2π * 2 * y / L) - 0.3 * cos(2π * 4 * x / L) * sin(2π * 10 * y / L) for x in xs, y in xs]
    return (; xs, f, u, v)
end

# Radial (shell-averaged) kinetic-energy spectrum E(n), n = integer cycles per domain.
function radial_KE_spectrum(u, v)
    N = size(u, 1)
    P = (abs2.(FFTW.fft(u)) .+ abs2.(FFTW.fft(v))) ./ (N^4)
    freqs = FFTW.fftfreq(N, N)                 # integer cycles per domain
    nmax = N ÷ 2
    E = zeros(nmax)
    for j in 1:N, i in 1:N
        nb = round(Int, sqrt(freqs[i]^2 + freqs[j]^2))
        (1 <= nb <= nmax) && (E[nb] += 0.5 * P[i, j])
    end
    return collect(1:nmax), E
end

# Power-law flow: streamfunction ψ̂ ∝ k⁻³ over an inertial band ⇒ kinetic-energy spectrum E(k) ∝ k⁻³
# (the 2D enstrophy-cascade slope). Used to show the filtering spectrum RECOVERS the Fourier slope.
function powerlaw_flow(; N = 320, dx = 1000.0, seed = 11, slope = -3.0, nmin = 3.0, nmax = 130.0)
    Random.seed!(seed)
    L = N * dx
    kfreq = 2π .* FFTW.fftfreq(N, 1 / dx)
    KX = [kfreq[i] for i in 1:N, _ in 1:N]
    KY = [kfreq[j] for _ in 1:N, j in 1:N]
    ψh = zeros(ComplexF64, N, N)
    for j in 1:N, i in 1:N
        n = sqrt(kfreq[i]^2 + kfreq[j]^2) * L / (2π)
        if nmin <= n <= nmax
            ψh[i, j] = n^slope * cis(2π * rand())
        end
    end
    u = real(FFTW.ifft(im .* KY .* ψh))
    v = real(FFTW.ifft(-im .* KX .* ψh))
    ω = real(FFTW.ifft((KX .^ 2 .+ KY .^ 2) .* ψh))
    s = 1.0 / Statistics.std(u)
    return (; xs = collect(0.0:dx:(N - 1) * dx), u = u .* s, v = v .* s, ω = ω .* s)
end

blankaxis!(ax) = (MK.hidedecorations!(ax); MK.hidespines!(ax))

# A heatmap panel with no ticks (clean image tiles).
function tile!(fig, r, c, km, A, cmap, cl; title = "")
    ax = MK.Axis(fig[r, c]; title = title, aspect = MK.DataAspect(),
        xticksvisible = false, yticksvisible = false,
        xticklabelsvisible = false, yticklabelsvisible = false)
    hm = MK.heatmap!(ax, km, km, A; colormap = cmap, colorrange = (-cl, cl))
    return hm
end

# ─── Hero banner: decompose the flow at several scales and measure the flux ──
function fig_hero()
    t = eddy_noise_flow()
    xs = t.xs; km = xs ./ 1e3
    grid = cartgrid(xs)
    ker = CGEF.GaussianKernel()
    scales_km = [6, 18, 40]
    ω̄ = [(o = zero(t.ω); CGEF.Filtering.filter_field!(o, t.ω, grid, ker, ℓ * 1e3); o) for ℓ in scales_km]
    Π = [(o = zero(t.u); CGEF.Diagnostics.compute_Π!(o, t.u, t.v, nothing, grid, ker, ℓ * 1e3); o) for ℓ in scales_km]
    clω = symclim(t.ω; q = 0.995)
    clΠ = maximum(symclim(P; interior = 14, q = 0.99) for P in Π)
    speed = sqrt.(t.u .^ 2 .+ t.v .^ 2)

    fig = MK.Figure(; size = (1480, 760))
    MK.Label(fig[0, 1:4], "Coarse-graining a flow: decompose into large scales, measure the cross-scale energy flux Π";
        fontsize = 20, font = :bold)
    # Top row: full vorticity, then large-scale ω̄_ℓ at each scale.
    tile!(fig, 1, 1, km, t.ω, FIELDMAP, clω; title = "vorticity ω")
    for (k, ℓ) in enumerate(scales_km)
        tile!(fig, 1, k + 1, km, ω̄[k], FIELDMAP, clω; title = "large scale ω̄  (ℓ = $ℓ km)")
    end
    MK.Colorbar(fig[1, 5], colormap = FIELDMAP, colorrange = (-clω, clω), width = 11, label = "ω (1/s)")
    # Bottom row: speed, then the energy flux Π at each scale (shared scale).
    let ax = MK.Axis(fig[2, 1]; title = "speed |u|", aspect = MK.DataAspect(),
            xticksvisible = false, yticksvisible = false, xticklabelsvisible = false, yticklabelsvisible = false)
        MK.heatmap!(ax, km, km, speed; colormap = SPEEDMAP)
    end
    local hmΠ
    for (k, ℓ) in enumerate(scales_km)
        hmΠ = tile!(fig, 2, k + 1, km, Π[k], CASCADE, clΠ; title = "flux Π  (ℓ = $ℓ km)")
    end
    MK.Colorbar(fig[2, 5], hmΠ; width = 11, label = "Π (W/m³)")
    MK.colgap!(fig.layout, 8); MK.rowgap!(fig.layout, 10)
    save_fig("hero.png", fig)
end

# ─── Filtering at multiple scales: two fields (fractal + eddy/noise) ─────────
function fig_filtering_scales()
    scales_km = [6, 18, 40]
    fr = fractal_field()
    ed = eddy_noise_flow()
    km = fr.xs ./ 1e3
    gridf = cartgrid(fr.xs); gride = cartgrid(ed.xs)
    ker = CGEF.GaussianKernel()

    fig = MK.Figure(; size = (1480, 770))
    MK.Label(fig[0, 1:4], "Spatial filtering coarsens a field as the scale ℓ grows"; fontsize = 19, font = :bold)

    # Row 1: deterministic fractal pattern.
    clf = symclim(fr.f)
    tile!(fig, 1, 1, km, fr.f, FIELDMAP, clf; title = "fractal field — original")
    for (k, ℓ) in enumerate(scales_km)
        o = zero(fr.f); CGEF.Filtering.filter_field!(o, fr.f, gridf, ker, ℓ * 1e3)
        tile!(fig, 1, k + 1, km, o, FIELDMAP, clf; title = "ℓ = $ℓ km")
    end
    MK.Colorbar(fig[1, 5], colormap = FIELDMAP, colorrange = (-clf, clf), width = 11)

    # Row 2: eddy + noise vorticity.
    cle = symclim(ed.ω; q = 0.995)
    tile!(fig, 2, 1, km, ed.ω, FIELDMAP, cle; title = "eddy+noise vorticity — original")
    for (k, ℓ) in enumerate(scales_km)
        o = zero(ed.ω); CGEF.Filtering.filter_field!(o, ed.ω, gride, ker, ℓ * 1e3)
        tile!(fig, 2, k + 1, km, o, FIELDMAP, cle; title = "ℓ = $ℓ km")
    end
    MK.Colorbar(fig[2, 5], colormap = FIELDMAP, colorrange = (-cle, cle), width = 11)
    MK.colgap!(fig.layout, 8); MK.rowgap!(fig.layout, 12)
    save_fig("filtering_scales.png", fig)
end

# ─── Filtering spectrum: recover the Fourier k⁻³ slope; kernel choice matters ─
function fig_filtering_spectrum()
    t = powerlaw_flow()
    xs = t.xs; km = xs ./ 1e3
    grid = cartgrid(xs)
    L = length(xs) * step_of(xs)                       # full period
    scales = 1e3 .* exp.(range(log(3.5), log(140); length = 44))
    kℓ = L ./ scales
    # Periodic field ⇒ fast FFT spectral filter (exact, O(N log N)). Build the spectrum with two kernels.
    function spectra(ker)
        Ec = map(scales) do ℓ
            ux = zero(t.u); uy = zero(t.v)
            CGEF.Filtering.filter_field!(ux, t.u, grid, ker, ℓ; method = CGEF.Filtering.Spectral())
            CGEF.Filtering.filter_field!(uy, t.v, grid, ker, ℓ; method = CGEF.Filtering.Spectral())
            0.5 * Statistics.mean(ux .^ 2 .+ uy .^ 2)
        end
        return Ec, CGEF.Diagnostics.spectral_density(Ec, kℓ)
    end
    Ecum, dens_s = spectra(CGEF.SharpSpectralKernel())
    _, dens_g = spectra(CGEF.GaussianKernel())

    # Power-law slope fit over the inertial band; returns (slope, anchor_k, anchor_E).
    function fitslope(k, E)
        m = (k .>= 12) .& (k .<= 80) .& (E .> 0)
        X = log.(k[m]); Y = log.(E[m]); x̄ = Statistics.mean(X); ȳ = Statistics.mean(Y)
        return sum((X .- x̄) .* (Y .- ȳ)) / sum((X .- x̄) .^ 2), exp(x̄), exp(ȳ)
    end
    ss, ka, Ea = fitslope(kℓ, dens_s)
    sg, _, _ = fitslope(kℓ, dens_g)
    ns, Ef = radial_KE_spectrum(t.u, t.v)
    fs, _, _ = fitslope(Float64.(ns), Ef)
    println("Fourier slope ≈ ", round(fs; digits = 2), "   sharp ≈ ", round(ss; digits = 2),
        "   gaussian ≈ ", round(sg; digits = 2))

    fig = MK.Figure(; size = (1580, 480))
    MK.Label(fig[0, 1:3], "Recover the energy spectrum by filtering — no FFT, no periodicity needed (Sadek & Aluie 2018)";
        fontsize = 18, font = :bold)

    clω = symclim(t.ω; q = 0.995)
    axf = MK.Axis(fig[1, 1]; title = "input flow:  E(k) ∝ k⁻³ turbulence", aspect = MK.DataAspect(),
        xticksvisible = false, yticksvisible = false, xticklabelsvisible = false, yticklabelsvisible = false)
    MK.heatmap!(axf, km, km, t.ω; colormap = FIELDMAP, colorrange = (-clω, clω))

    ax1 = MK.Axis(fig[1, 2]; title = "cumulative coarse KE  E(ℓ)   [Eq. 15]",
        xlabel = "filter scale  ℓ  (km)", ylabel = "E(ℓ)  (J/m³)",
        xscale = log10, yscale = log10, xgridvisible = true, ygridvisible = true)
    MK.scatterlines!(ax1, scales ./ 1e3, Ecum; linewidth = 3, markersize = 5, color = :steelblue)
    MK.ylims!(ax1, maximum(Ecum) / 3e3, maximum(Ecum) * 2)

    ax2 = MK.Axis(fig[1, 3]; title = "filtering spectral density  Ẽ(k_ℓ)   [Eq. 14] — kernel matters",
        xlabel = "filtering wavenumber  k_ℓ = L/ℓ", ylabel = "Ẽ(k_ℓ)",
        xscale = log10, yscale = log10, xgridvisible = true, ygridvisible = true)
    ps = dens_s .> 0; pg = dens_g .> 0
    MK.scatterlines!(ax2, kℓ[ps], dens_s[ps]; linewidth = 2.5, markersize = 5, color = :firebrick,
        label = "sharp-spectral  (slope ≈ $(round(ss; digits = 1)))")
    MK.scatterlines!(ax2, kℓ[pg], dens_g[pg]; linewidth = 2.5, markersize = 5, color = :darkorange,
        label = "Gaussian  (slope ≈ $(round(sg; digits = 1)))")
    kline = range(12, 80; length = 40)
    MK.lines!(ax2, kline, Ea .* (kline ./ ka) .^ (-3.0); linestyle = :dash, linewidth = 2, color = :gray25,
        label = "k⁻³ (true Fourier slope)")
    MK.ylims!(ax2, Ea / 1e3, maximum(dens_s[ps]) * 3)
    MK.axislegend(ax2; position = :lb, framevisible = false, labelsize = 11)
    MK.colgap!(fig.layout, 26)
    save_fig("filtering_spectrum.png", fig)
end

# ─── Kernels: real-space profiles + spectral transfer functions ──────────────
function fig_kernels()
    r = collect(0.0:0.005:1.3)                          # distance in units of ℓ
    kk = collect(0.0:0.02:13.0)                         # wavenumber in units of 1/ℓ
    ℓ = 1.0
    fig = MK.Figure(; size = (1180, 470))
    MK.Label(fig[0, 1:2], "Filter kernels: real-space shape and spectral transfer Ĝ(k)"; fontsize = 18, font = :bold)

    ax1 = MK.Axis(fig[1, 1]; title = "real-space kernel  G(r)", xlabel = "distance  r / ℓ",
        ylabel = "weight (normalized to 1 at r=0)", xgridvisible = true, ygridvisible = true)
    MK.lines!(ax1, r, [CGEF.Kernels.kernel_weight(CGEF.TopHatKernel(), d, ℓ) for d in r]; linewidth = 3, color = :seagreen, label = "top-hat")
    MK.lines!(ax1, r, [CGEF.Kernels.kernel_weight(CGEF.GaussianKernel(; α = 6), d, ℓ) for d in r]; linewidth = 3, color = :firebrick, label = "Gaussian α=6 (Pope)")
    MK.lines!(ax1, r, [CGEF.Kernels.kernel_weight(CGEF.GaussianKernel(; α = 4), d, ℓ) for d in r]; linewidth = 3, color = :darkorange, linestyle = :dash, label = "Gaussian α=4 (FlowSieve)")
    MK.axislegend(ax1; position = :rt, framevisible = false, labelsize = 12)

    ax2 = MK.Axis(fig[1, 2]; title = "spectral transfer  Ĝ(k, ℓ)", xlabel = "wavenumber  k·ℓ",
        ylabel = "Ĝ(k)", xgridvisible = true, ygridvisible = true)
    MK.lines!(ax2, kk, [CGEF.Kernels.spectral_transfer(CGEF.SharpSpectralKernel(), k, ℓ) for k in kk]; linewidth = 3, color = :steelblue, label = "sharp-spectral (brick wall)")
    MK.lines!(ax2, kk, [CGEF.Kernels.spectral_transfer(CGEF.GaussianKernel(; α = 6), k, ℓ) for k in kk]; linewidth = 3, color = :firebrick, label = "Gaussian α=6")
    MK.lines!(ax2, kk, [CGEF.Kernels.spectral_transfer(CGEF.GaussianKernel(; α = 4), k, ℓ) for k in kk]; linewidth = 3, color = :darkorange, linestyle = :dash, label = "Gaussian α=4")
    MK.vlines!(ax2, [π]; color = :gray60, linestyle = :dot)
    MK.text!(ax2, π, 0.95; text = " k = π/ℓ", align = (:left, :top), fontsize = 12, color = :gray40)
    MK.axislegend(ax2; position = :rt, framevisible = false, labelsize = 12)
    MK.colgap!(fig.layout, 28)
    save_fig("kernels.png", fig)
end

# ─── Rigid-body rotation: Π must vanish — square vs circular domain ──────────
function fig_rigid_rotation()
    N = 161; dx = 1000.0
    geom = CGEF.CartesianGeometry(dx, dx)
    xs = collect(-80e3:dx:80e3); km = xs ./ 1e3
    Ω = 1e-4
    u = [-Ω * y for x in xs, y in xs]
    v = [Ω * x for x in xs, y in xs]
    R0 = maximum(xs)                                    # inscribed-circle radius (half the box)
    cmask = [hypot(x, y) <= R0 for x in xs, y in xs]    # circular domain via a Bool mask

    gsq = CGEF.StructuredGrid(geom, xs, xs, trues(N, N))
    gci = CGEF.StructuredGrid(geom, xs, xs, cmask)
    ker = CGEF.GaussianKernel()
    Πsq = zero(u); CGEF.Diagnostics.compute_Π!(Πsq, u, v, nothing, gsq, ker, 20e3)
    Πci = zero(u); CGEF.Diagnostics.compute_Π!(Πci, u, v, nothing, gci, ker, 20e3)
    Πci_disp = [cmask[i, j] ? Πci[i, j] : NaN for i in 1:N, j in 1:N]
    mxsq = maximum(abs, Πsq); mxci = maximum(abs, filter(!isnan, Πci_disp))

    fig = MK.Figure(; size = (1340, 470))
    MK.Label(fig[0, 1:3], "Validation — rigid-body rotation is pure rotation, so Π must vanish (edge effects only)";
        fontsize = 18, font = :bold)

    ax1 = MK.Axis(fig[1, 1]; title = "speed |u|  (+ inscribed circular domain)", aspect = MK.DataAspect(),
        xlabel = "x (km)", ylabel = "y (km)")
    hm1 = MK.heatmap!(ax1, km, km, sqrt.(u .^ 2 .+ v .^ 2); colormap = SPEEDMAP)
    MK.arc!(ax1, MK.Point2f(0, 0), R0 / 1e3, -π, π; color = :black, linewidth = 2, linestyle = :dash)
    MK.Colorbar(fig[1, 1, MK.Right()], hm1; width = 11, label = "|u| (m/s)")

    ax2 = MK.Axis(fig[1, 2]; title = "Π — square domain  (edge effect at corners)", aspect = MK.DataAspect(),
        xlabel = "x (km)", ylabel = "y (km)")
    MK.heatmap!(ax2, km, km, Πsq; colormap = CASCADE, colorrange = (-1e-12, 1e-12))

    ax3 = MK.Axis(fig[1, 3]; title = "Π — circular domain  (edge effect at the rim)", aspect = MK.DataAspect(),
        xlabel = "x (km)", ylabel = "y (km)")
    hm3 = MK.heatmap!(ax3, km, km, Πci_disp; colormap = CASCADE, colorrange = (-1e-12, 1e-12))
    MK.Colorbar(fig[1, 3, MK.Right()], hm3; width = 11, label = "Π (W/m³)")

    MK.Label(fig[2, 1:3],
        "Interior Π ≈ machine zero in both domains; nonzero values are finite-domain filtering artifacts that " *
        "live only at the boundary — the square's corners and the circular mask's rim (max ≈ $(round(mxsq; sigdigits = 2)) and " *
        "$(round(mxci; sigdigits = 2)) W/m³). Cropping a boundary band (or a larger domain) removes them.";
        fontsize = 11.5, color = :gray35)
    MK.colgap!(fig.layout, 26)
    save_fig("rigid_rotation_validation.png", fig; px = 1.4)   # keep < 2000 px wide
end

# Rotational (streamfunction) + divergent (velocity-potential) flow, band-limited random phases.
function rot_div_flow(; N = 192, dx = 1000.0, seed = 3)
    Random.seed!(seed)
    L = N * dx; kf = 2π .* FFTW.fftfreq(N, 1 / dx)
    KX = [kf[i] for i in 1:N, _ in 1:N]; KY = [kf[j] for _ in 1:N, j in 1:N]
    ψh = zeros(ComplexF64, N, N); φh = zeros(ComplexF64, N, N)
    for j in 1:N, i in 1:N
        n = sqrt(kf[i]^2 + kf[j]^2) * L / (2π)
        if 4 <= n <= 22
            ψh[i, j] = n^(-2.0) * cis(2π * rand())
            φh[i, j] = n^(-2.0) * cis(2π * rand())
        end
    end
    ur = real(FFTW.ifft(im .* KY .* ψh)); vr = real(FFTW.ifft(-im .* KX .* ψh))   # rotational
    ud = real(FFTW.ifft(im .* KX .* φh)); vd = real(FFTW.ifft(im .* KY .* φh))     # divergent
    s = 1.0 / Statistics.std(ur .+ ud)
    return (; xs = collect(0.0:dx:(N - 1) * dx), ur = ur .* s, vr = vr .* s, ud = ud .* s, vd = vd .* s)
end

# ─── Rotational / divergent (Helmholtz) decomposition of Π ───────────────────
function fig_helmholtz()
    f = rot_div_flow(); xs = f.xs; km = xs ./ 1e3
    grid = cartgrid(xs)
    u = f.ur .+ f.ud; v = f.vr .+ f.vd
    dec = CGEF.Diagnostics.compute_Π_decomposed(u, v, f.ur, f.vr, grid, CGEF.GaussianKernel(), 16e3)
    cl = maximum(symclim(getfield(dec, s); interior = 10, q = 0.99) for s in (:total, :rotational, :cross, :divergent))

    fig = MK.Figure(; size = (1500, 430))
    MK.Label(fig[0, 1:4], "Helmholtz decomposition of the flux:  Π = Π_rotational + Π_cross + Π_divergent  (exact)";
        fontsize = 19, font = :bold)
    local hm
    for (k, (ttl, A)) in enumerate((("total Π", dec.total), ("rotational", dec.rotational),
            ("cross", dec.cross), ("divergent", dec.divergent)))
        hm = tile!(fig, 1, k, km, A, CASCADE, cl; title = ttl)
    end
    MK.Colorbar(fig[1, 5], hm; width = 11, label = "Π (W/m³)")
    MK.colgap!(fig.layout, 8)
    save_fig("helmholtz_decomposition.png", fig)
end

# ─── Tracer-variance (buoyancy / APE) flux ───────────────────────────────────
function fig_tracer_flux()
    ed = eddy_noise_flow(seed = 4)
    xs = ed.xs; km = xs ./ 1e3; L = length(xs) * step_of(xs)
    grid = cartgrid(xs)
    # tracer: a large-scale north–south gradient stirred by the eddies (front-like), + fine structure
    θ = [sin(2π * 1 * y / L) + 0.25 * sin(2π * 7 * x / L) * cos(2π * 5 * y / L) for x in xs, y in xs]
    ℓ = 18e3
    θ̄ = zero(θ); CGEF.Filtering.filter_field!(θ̄, θ, grid, CGEF.GaussianKernel(), ℓ)
    Πθ = CGEF.Diagnostics.tracer_variance_flux(ed.u, ed.v, θ, grid, CGEF.GaussianKernel(), ℓ)

    fig = MK.Figure(; size = (1500, 440))
    MK.Label(fig[0, 1:3], "Cross-scale tracer-variance flux  Πθ  (buoyancy ⇒ available-potential-energy transfer)";
        fontsize = 18, font = :bold)
    clθ = symclim(θ)
    h1 = tile!(fig, 1, 1, km, θ, FIELDMAP, clθ; title = "tracer θ (stirred front)")
    MK.Colorbar(fig[1, 1, MK.Right()], h1; width = 10)
    h2 = tile!(fig, 1, 2, km, θ̄, FIELDMAP, clθ; title = "coarse tracer θ̄  (ℓ = 18 km)")
    MK.Colorbar(fig[1, 2, MK.Right()], h2; width = 10)
    clΠ = symclim(Πθ; interior = 12, q = 0.99)
    h3 = tile!(fig, 1, 3, km, Πθ, CASCADE, clΠ; title = "tracer-variance flux Πθ")
    MK.Colorbar(fig[1, 3, MK.Right()], h3; width = 10, label = "Πθ")
    MK.colgap!(fig.layout, 22)
    save_fig("tracer_flux.png", fig)
end

# ─── Masking: deformable vs zero-fill near a mask boundary ───────────────────
function fig_masking()
    fr = fractal_field(); xs = fr.xs; km = xs ./ 1e3; N = length(xs)
    # an irregular mask boundary: exclude a disk + a corner region
    cx, cy = N ÷ 2, Int(round(0.62N))
    mask = trues(N, N)
    for j in 1:N, i in 1:N
        ((i - cx)^2 + (j - cy)^2 <= (0.16N)^2) && (mask[i, j] = false)
        (i + j <= Int(round(0.5N))) && (mask[i, j] = false)
    end
    geom = CGEF.CartesianGeometry(step_of(xs), step_of(xs))
    grid = CGEF.StructuredGrid(geom, xs, xs, mask)
    ℓ = 16e3; ker = CGEF.GaussianKernel()
    od = zero(fr.f); CGEF.Filtering.filter_field!(od, fr.f, grid, ker, ℓ; mask_strategy = CGEF.Filtering.Deformable())
    oz = zero(fr.f); CGEF.Filtering.filter_field!(oz, fr.f, grid, ker, ℓ; mask_strategy = CGEF.Filtering.ZeroFill())
    # show masked cells as NaN (rendered transparent/blank)
    showmasked(A) = [mask[i, j] ? A[i, j] : NaN for i in 1:N, j in 1:N]
    maskoverlay = [mask[i, j] ? NaN : 1.0 for i in 1:N, j in 1:N]
    cl = symclim(fr.f)
    diff = showmasked(od .- oz)
    cld = symclim(filter(!isnan, diff) |> collect)

    fig = MK.Figure(; size = (1760, 430))
    MK.Label(fig[0, 1:4], "Masking: the deformable kernel renormalizes over active cells (no bleed) — the difference is concentrated at the mask boundary";
        fontsize = 18, font = :bold)
    for (k, (ttl, A)) in enumerate((("field + mask", fr.f), ("Deformable (renormalized)", od),
            ("ZeroFill (masked = 0)", oz)))
        ax = MK.Axis(fig[1, k]; title = ttl, aspect = MK.DataAspect(),
            xticksvisible = false, yticksvisible = false, xticklabelsvisible = false, yticklabelsvisible = false)
        hm = MK.heatmap!(ax, km, km, showmasked(A); colormap = FIELDMAP, colorrange = (-cl, cl))
        MK.heatmap!(ax, km, km, maskoverlay; colormap = [:gray75, :gray75])
        k == 3 && MK.Colorbar(fig[1, k, MK.Right()], hm; width = 10)
    end
    ax4 = MK.Axis(fig[1, 4]; title = "Deformable − ZeroFill", aspect = MK.DataAspect(),
        xticksvisible = false, yticksvisible = false, xticklabelsvisible = false, yticklabelsvisible = false)
    hmd = MK.heatmap!(ax4, km, km, diff; colormap = :PuOr, colorrange = (-cld, cld))
    MK.heatmap!(ax4, km, km, maskoverlay; colormap = [:gray75, :gray75])
    MK.Colorbar(fig[1, 4, MK.Right()], hmd; width = 10)
    MK.colgap!(fig.layout, 12)
    save_fig("masking.png", fig)
end

# ─── Spherical spectral filtering: uniform (FSH) + scattered (NUFSHT) ────────
function fig_spherical()
    Ndeg = 48; N = Ndeg + 1; M = 2N - 1
    Θ, Φ = FSH.sph_points(N)
    R = 6.371e6
    sgrid = CGEF.StructuredGrid(CGEF.SphericalGeometry(R), collect(Φ), π / 2 .- collect(Θ), trues(M, N))
    Random.seed!(9)
    C = zeros(N, M)
    for l in 1:30, m in (-l):l
        C[FSH.sph_mode(l, m)] = (l^(-1.3)) * (2 * rand() - 1)
    end
    field = permutedims(FSH.sph_evaluate(C))            # [lon, lat] = M×N
    ℓ = π * R / 10                                       # keep ~degree ≲ 10
    out = zero(field); CGEF.Filtering.filter_field!(out, field, sgrid, CGEF.GaussianKernel(), ℓ; method = CGEF.Filtering.Spectral())

    Φv = collect(Φ); latgrid = π / 2 .- collect(Θ)
    londeg = rad2deg.(Φv); latdeg = rad2deg.(latgrid)
    p = sortperm(latdeg); lats = latdeg[p]
    cl = symclim(field; q = 0.999)

    # Scattered observations on a well-distributed Fibonacci sphere → filtered with NUFSHT.
    Mpts = 1600; ga = π * (3 - sqrt(5))
    sθ = [acos(clamp(1 - 2 * (k + 0.5) / Mpts, -1, 1)) for k in 0:(Mpts - 1)]
    sφ = [mod(ga * k, 2π) for k in 0:(Mpts - 1)]
    slat = π / 2 .- sθ
    obs = [field[argmin(abs.(Φv .- sφ[q])), argmin(abs.(latgrid .- slat[q]))] for q in 1:Mpts]
    ug = CGEF.UnstructuredGrid(CGEF.SphericalGeometry(R), sφ, slat, ones(Mpts), trues(Mpts))   # no adjacency needed for spectral-only filtering
    fobs = zero(obs); CGEF.Filtering.filter_field!(fobs, obs, ug, CGEF.GaussianKernel(), ℓ; method = CGEF.Filtering.Spectral())

    fig = MK.Figure(; size = (1660, 440))
    MK.Label(fig[0, 1:3], "Spectral filtering on the sphere — uniform grid (FastSphericalHarmonics) and scattered points (NUFSHT)";
        fontsize = 18, font = :bold)
    ax1 = MK.Axis(fig[1, 1]; title = "global field (uniform grid)", xlabel = "longitude (°)", ylabel = "latitude (°)")
    MK.heatmap!(ax1, londeg, lats, field[:, p]; colormap = FIELDMAP, colorrange = (-cl, cl))
    ax2 = MK.Axis(fig[1, 2]; title = "FSH-filtered (degree ≲ 10)", xlabel = "longitude (°)", ylabel = "latitude (°)")
    MK.heatmap!(ax2, londeg, lats, out[:, p]; colormap = FIELDMAP, colorrange = (-cl, cl))
    ax3 = MK.Axis(fig[1, 3]; title = "NUFSHT-filtered (1600 scattered points)", xlabel = "longitude (°)", ylabel = "latitude (°)")
    MK.scatter!(ax3, rad2deg.(sφ), rad2deg.(slat); color = fobs, colormap = FIELDMAP, colorrange = (-cl, cl), markersize = 7)
    for a in (ax1, ax2, ax3); MK.xlims!(a, 0, 360); MK.ylims!(a, -90, 90); end
    MK.Colorbar(fig[1, 4], colormap = FIELDMAP, colorrange = (-cl, cl), width = 11)
    MK.Label(fig[2, 1:3],
        "FFTW (uniform Cartesian) and FINUFFT (scattered Cartesian) complete the {Cartesian, spherical} × {uniform, scattered} set.";
        fontsize = 11.5, color = :gray35)
    MK.colgap!(fig.layout, 26)
    save_fig("spherical_filtering.png", fig)
end

# ─── CurvilinearGrid: a sheared/rotated model-native mesh ────────────────────
function fig_curvilinear()
    N = 55; dx = 2_000.0
    geom = CGEF.CartesianGeometry(dx, dx)
    i = collect(0.0:(N - 1)); j = collect(0.0:(N - 1))
    θ = deg2rad(18.0); shear = 0.35
    lon = [dx * (ii * cos(θ) - jj * shear * sin(θ)) for ii in i, jj in j]
    lat = [dx * (ii * sin(θ) + jj * (1 + shear * cos(θ))) for ii in i, jj in j]
    mask = trues(N, N)
    grid = CGEF.CurvilinearGrid(geom, lon, lat, mask)
    km_lon = lon ./ 1e3; km_lat = lat ./ 1e3

    L = N * dx
    f = [1.0 * sin(2π * 2 * lon[ii, jj] / L) * cos(2π * 2 * lat[ii, jj] / L) +
         0.5 * sin(2π * 6 * lon[ii, jj] / L) * cos(2π * 6 * lat[ii, jj] / L) for ii in 1:N, jj in 1:N]
    u = [ 0.6 * sin(2π * 2 * lon[ii, jj] / L) * cos(2π * 5 * lat[ii, jj] / L) for ii in 1:N, jj in 1:N]
    v = [-0.6 * cos(2π * 5 * lon[ii, jj] / L) * sin(2π * 2 * lat[ii, jj] / L) for ii in 1:N, jj in 1:N]

    ℓ = 18e3; ker = CGEF.GaussianKernel()
    f̄ = zero(f); CGEF.Filtering.filter_field!(f̄, f, grid, ker, ℓ)
    Π = zero(f); CGEF.Diagnostics.compute_Π!(Π, u, v, nothing, grid, ker, ℓ)

    fig = MK.Figure(; size = (1560, 420))
    MK.Label(fig[0, 1:4], "CurvilinearGrid: model-native (sheared/rotated) mesh — no rectilinear axis anywhere";
        fontsize = 18, font = :bold)

    ax0 = MK.Axis(fig[1, 1]; title = "mesh (every 4th grid line)", aspect = MK.DataAspect(),
        xlabel = "x (km)", ylabel = "y (km)")
    for ii in 1:4:N; MK.lines!(ax0, km_lon[ii, :], km_lat[ii, :]; color = :gray55, linewidth = 0.8); end
    for jj in 1:4:N; MK.lines!(ax0, km_lon[:, jj], km_lat[:, jj]; color = :gray55, linewidth = 0.8); end

    clf = maximum(abs, f)
    ax1 = MK.Axis(fig[1, 2]; title = "field f", aspect = MK.DataAspect(), xlabel = "x (km)")
    MK.scatter!(ax1, vec(km_lon), vec(km_lat); color = vec(f), colormap = FIELDMAP, colorrange = (-clf, clf), markersize = 5)

    ax2 = MK.Axis(fig[1, 3]; title = "filtered f̄  (ℓ = 18 km)", aspect = MK.DataAspect(), xlabel = "x (km)")
    MK.scatter!(ax2, vec(km_lon), vec(km_lat); color = vec(f̄), colormap = FIELDMAP, colorrange = (-clf, clf), markersize = 5)
    MK.Colorbar(fig[1, 3, MK.Right()], colormap = FIELDMAP, colorrange = (-clf, clf), width = 10)

    clΠ = symclim(Π; interior = 6, q = 0.98)
    ax3 = MK.Axis(fig[1, 4]; title = "flux Π", aspect = MK.DataAspect(), xlabel = "x (km)")
    hm3 = MK.scatter!(ax3, vec(km_lon), vec(km_lat); color = vec(Π), colormap = CASCADE, colorrange = (-clΠ, clΠ), markersize = 5)
    MK.Colorbar(fig[1, 4, MK.Right()], hm3; width = 10)
    MK.colgap!(fig.layout, 20)
    save_fig("curvilinear.png", fig)
end

# ─── UnstructuredGrid: scattered points, k-d tree + Voronoi + FINUFFT ────────
function fig_unstructured()
    Random.seed!(21)
    npts = 1_500; L = 100e3
    geom = CGEF.CartesianGeometry(1.0, 1.0)   # placeholder — no fixed spacing for scattered data
    lon = L .* rand(npts); lat = L .* rand(npts)
    mask = trues(npts)
    grid = CGEF.UnstructuredGrid(geom, lon, lat, mask; k = 8)   # k-d tree adjacency + Voronoi areas

    f = [1.0 * sin(2π * 2 * lon[q] / L) * cos(2π * 2 * lat[q] / L) +
         0.5 * sin(2π * 7 * lon[q] / L) * cos(2π * 7 * lat[q] / L) for q in 1:npts]
    u = [ 0.6 * sin(2π * 2 * lon[q] / L) * cos(2π * 5 * lat[q] / L) for q in 1:npts]
    v = [-0.6 * cos(2π * 5 * lon[q] / L) * sin(2π * 2 * lat[q] / L) for q in 1:npts]

    ℓ = 12e3; ker = CGEF.GaussianKernel()
    f̄ = zero(f); CGEF.Filtering.filter_field!(f̄, f, grid, ker, ℓ; method = CGEF.Filtering.Spectral())
    Π = zero(f); CGEF.Diagnostics.compute_Π!(Π, u, v, nothing, grid, ker, ℓ)

    km_lon = lon ./ 1e3; km_lat = lat ./ 1e3
    fig = MK.Figure(; size = (1260, 430))
    MK.Label(fig[0, 1:3], "UnstructuredGrid: scattered points — k-d tree neighbors, Voronoi cell areas, FINUFFT spectral filtering";
        fontsize = 17, font = :bold)

    clf = maximum(abs, f)
    ax1 = MK.Axis(fig[1, 1]; title = "field f (scattered samples)", aspect = MK.DataAspect(), xlabel = "x (km)", ylabel = "y (km)")
    MK.scatter!(ax1, km_lon, km_lat; color = f, colormap = FIELDMAP, colorrange = (-clf, clf), markersize = 6)

    ax2 = MK.Axis(fig[1, 2]; title = "spectrally filtered f̄  (ℓ = 12 km)", aspect = MK.DataAspect(), xlabel = "x (km)")
    MK.scatter!(ax2, km_lon, km_lat; color = f̄, colormap = FIELDMAP, colorrange = (-clf, clf), markersize = 6)
    MK.Colorbar(fig[1, 2, MK.Right()], colormap = FIELDMAP, colorrange = (-clf, clf), width = 10)

    clΠ = symclim(Π; q = 0.98)
    ax3 = MK.Axis(fig[1, 3]; title = "flux Π", aspect = MK.DataAspect(), xlabel = "x (km)")
    hm3 = MK.scatter!(ax3, km_lon, km_lat; color = Π, colormap = CASCADE, colorrange = (-clΠ, clΠ), markersize = 6)
    MK.Colorbar(fig[1, 3, MK.Right()], hm3; width = 10)
    MK.colgap!(fig.layout, 20)
    save_fig("unstructured.png", fig)
end

# ─── True 3D volumetric flux: genuinely coupled 3D Cartesian filtering ───────
function fig_volumetric_3d()
    Random.seed!(31)
    N = 22; dx = 500.0
    geom = CGEF.CartesianGeometry(dx, dx, dx)
    x = collect(0.0:dx:(N - 1) * dx); km = x ./ 1e3
    mask = trues(N, N, N)
    grid = CGEF.StructuredGrid(geom, x, x, x, mask)

    L = N * dx
    u = [sin(2π * 2 * x[a] / L) * cos(2π * 2 * x[b] / L) * cos(2π * 2 * x[c] / L) for a in 1:N, b in 1:N, c in 1:N]
    v = [cos(2π * 2 * x[a] / L) * sin(2π * 2 * x[b] / L) * cos(2π * 2 * x[c] / L) for a in 1:N, b in 1:N, c in 1:N]
    w = [-2 .* cos(2π * 2 * x[a] / L) * cos(2π * 2 * x[b] / L) * sin(2π * 2 * x[c] / L) for a in 1:N, b in 1:N, c in 1:N]

    ℓ = 3_000.0
    Π = zeros(N, N, N); CGEF.Diagnostics.compute_Π!(Π, u, v, w, grid, CGEF.TopHatKernel(), ℓ)
    speed = sqrt.(u .^ 2 .+ v .^ 2 .+ w .^ 2)

    zlevels = (6, 11, 16)
    clΠ = maximum(symclim(Π[:, :, k]; interior = 3, q = 0.98) for k in zlevels)
    clsp = maximum(speed)

    fig = MK.Figure(; size = (1420, 560))
    MK.Label(fig[0, 1:4], "True 3D volumetric flux — genuinely coupled 3D Cartesian filtering (all nine strain/stress components)";
        fontsize = 18, font = :bold)
    local hmsp, hmΠ
    for (col, k) in enumerate(zlevels)
        hmsp = tile!(fig, 1, col, km, speed[:, :, k], SPEEDMAP, clsp; title = "speed |u|  (z = $(round(x[k]/1e3; digits=1)) km)")
        hmΠ = tile!(fig, 2, col, km, Π[:, :, k], CASCADE, clΠ; title = "flux Π  (z = $(round(x[k]/1e3; digits=1)) km)")
    end
    MK.Colorbar(fig[1, 4], hmsp; width = 11, label = "|u|")
    MK.Colorbar(fig[2, 4], hmΠ; width = 11, label = "Π")
    MK.colgap!(fig.layout, 10); MK.rowgap!(fig.layout, 10)
    save_fig("volumetric_3d.png", fig)
end

# ─── Depth-profile (2.5D-per-level) vertical structure ───────────────────────
function fig_depth_profile()
    Random.seed!(41)
    N = 70; Nz = 8; dx = 1_000.0
    xs = collect(0.0:dx:(N - 1) * dx); km = xs ./ 1e3
    grid = cartgrid(xs)   # periodic — the synthetic field below is exactly periodic, so this avoids
                          # domain-edge footprint-truncation artifacts swamping the color scale

    L = N * dx
    decay = [exp(-2.5 * (k - 1) / (Nz - 1)) for k in 1:Nz]     # surface-intensified eddy field
    u = zeros(N, N, Nz); v = zeros(N, N, Nz)
    for k in 1:Nz, (a, x) in enumerate(xs), (b, y) in enumerate(xs)
        u[a, b, k] = decay[k] * sin(2π * 3 * x / L) * cos(2π * 3 * y / L)
        v[a, b, k] = -decay[k] * cos(2π * 3 * x / L) * sin(2π * 3 * y / L)
    end

    scales = collect(6e3:6e3:30e3)
    result = CGEF.coarse_grain_profile(u, v, grid; scales = scales, kernel = CGEF.GaussianKernel())
    depth_km = collect(0:(Nz - 1)) .* 0.5      # nominal 0.5 km level spacing, cosmetic only

    j0 = N ÷ 2   # a fixed y-row for the x–depth cross-section
    Πxz = [result.Π[a, j0, k, 2] for a in 1:N, k in 1:Nz]   # scale index 2 = 12 km
    clΠ = symclim(Πxz; q = 0.98)   # no `interior` crop here — Nz is small, an 8-deep symmetric crop leaves nothing

    fig = MK.Figure(; size = (1360, 440))
    MK.Label(fig[0, 1:2], "Depth-profile (2.5D-per-level) vertical structure — the literature-standard method (Aluie, Hecht & Vallis 2018)";
        fontsize = 16, font = :bold)

    ax1 = MK.Axis(fig[1, 1]; title = "x–depth cross-section of Π  (ℓ = 12 km, y = mid-domain)",
        xlabel = "x (km)", ylabel = "depth (km, nominal)", yreversed = true)
    hm1 = MK.heatmap!(ax1, km, depth_km, Πxz; colormap = CASCADE, colorrange = (-clΠ, clΠ))
    MK.Colorbar(fig[1, 1, MK.Right()], hm1; width = 10, label = "Π")

    ax2 = MK.Axis(fig[1, 2]; title = "mean|Π| vs. depth, by scale", xlabel = "mean|Π|", ylabel = "depth (km, nominal)", yreversed = true)
    for (sidx, ℓ) in enumerate(scales)
        sidx % 2 == 1 || continue
        prof = [Statistics.mean(abs, @view result.Π[:, :, k, sidx]) for k in 1:Nz]
        MK.lines!(ax2, prof, depth_km; linewidth = 2.5, label = "ℓ = $(round(Int, ℓ/1e3)) km")
        MK.scatter!(ax2, prof, depth_km; markersize = 8)
    end
    MK.axislegend(ax2; position = :rb, framevisible = false, labelsize = 11)
    MK.colgap!(fig.layout, 26)
    save_fig("depth_profile.png", fig)
end

println("Generating CoarseGrainingEnergyFluxes.jl documentation assets …")
fig_hero()
fig_filtering_scales()
fig_filtering_spectrum()
fig_kernels()
fig_rigid_rotation()
fig_helmholtz()
fig_tracer_flux()
fig_masking()
fig_spherical()
fig_curvilinear()
fig_unstructured()
fig_volumetric_3d()
fig_depth_profile()
println("done.")
