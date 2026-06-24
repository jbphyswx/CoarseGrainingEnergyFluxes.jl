"""
Generate static figure assets for CoarseGrainingEnergyFluxes.jl docs.

Run from this directory:
    julia --project=. generate_assets.jl

Outputs PNG files to ../assets/ which are checked into the repo
and referenced from README.md and docs/ markdown.
"""

using CairoMakie: CairoMakie
using CoarseGrainingEnergyFluxes
using Statistics: Statistics

const ASSETS_DIR = joinpath(@__DIR__, "..", "src", "assets")  # Documenter serves docs/src/assets
mkpath(ASSETS_DIR)

# ─── Figure 1: Filtering at multiple scales ──────────────────────────────

function figure_filtering_scales()
    # Create a turbulent-like velocity field (multi-scale sinusoidal)
    N = 101
    dx = 1000.0
    geom = CartesianGeometry(dx, dx)
    xs = collect(0.0:dx:(N-1)*dx)
    ys = collect(0.0:dx:(N-1)*dx)
    mask = trues(N, N)
    grid = StructuredGrid(geom, xs, ys, mask)

    # Multi-scale field: sum of sinusoids at different wavenumbers
    u = zeros(N, N)
    for j in 1:N, i in 1:N
        x, y = xs[i], ys[j]
        L = (N-1) * dx
        u[i,j] = (1.0 * sin(2π * x / L) * cos(2π * y / L) +     # large scale
                   0.5 * sin(6π * x / L) * cos(6π * y / L) +     # medium scale
                   0.3 * sin(14π * x / L) * cos(14π * y / L))    # small scale
    end

    # Filter at 3 scales
    scales_km = [5, 15, 30]
    scales_m = scales_km .* 1000.0

    fig = CairoMakie.Figure(; size=(1400, 400), fontsize=14)
    CairoMakie.Label(fig[0, 1:8], "Spatial Filtering at Multiple Scales";
                      fontsize=18, font=:bold)

    # Original
    ax0 = CairoMakie.Axis(fig[1, 1]; title="Original field", xlabel="x (km)", ylabel="y (km)",
                           aspect=CairoMakie.DataAspect())
    clim = maximum(abs.(u))
    hm = CairoMakie.heatmap!(ax0, xs ./ 1e3, ys ./ 1e3, u; colormap=:RdBu, colorrange=(-clim, clim))
    CairoMakie.Colorbar(fig[1, 2], hm; width=12)

    for (k, (ℓ_km, ℓ_m)) in enumerate(zip(scales_km, scales_m))
        out = zeros(N, N)
        filter_field!(out, u, grid, TopHatKernel(), ℓ_m)
        ax = CairoMakie.Axis(fig[1, 2*k+1]; title="ℓ = $(ℓ_km) km", xlabel="x (km)", ylabel="y (km)",
                              aspect=CairoMakie.DataAspect())
        hm_k = CairoMakie.heatmap!(ax, xs ./ 1e3, ys ./ 1e3, out; colormap=:RdBu, colorrange=(-clim, clim))
        CairoMakie.Colorbar(fig[1, 2*k+2], hm_k; width=12)
    end

    outpath = joinpath(ASSETS_DIR, "filtering_scales.png")
    CairoMakie.save(outpath, fig; px_per_unit=2)
    println("Saved: $outpath")
end

# ─── Figure 2: Energy flux Π for vortex + strain ─────────────────────────

function figure_energy_flux()
    N = 101
    dx = 1000.0
    geom = CartesianGeometry(dx, dx)
    xs = collect(-50e3:dx:50e3)
    ys = collect(-50e3:dx:50e3)
    mask = trues(N, N)
    grid = StructuredGrid(geom, xs, ys, mask)

    # Superposition of vortex + small-scale perturbation
    u = zeros(N, N)
    v = zeros(N, N)
    Ω = 1e-4
    L = 100e3
    for j in 1:N, i in 1:N
        x, y = xs[i], ys[j]
        # Large-scale solid body rotation
        u[i,j] = -Ω * y
        v[i,j] =  Ω * x
        # Small-scale turbulent perturbation
        u[i,j] += 0.3 * sin(12π * x / L) * cos(10π * y / L)
        v[i,j] += 0.3 * cos(8π * x / L) * sin(14π * y / L)
    end

    # Compute Π at two scales
    Π_small = zeros(N, N)
    Π_large = zeros(N, N)
    compute_Π!(Π_small, u, v, nothing, grid, TopHatKernel(), 10e3)
    compute_Π!(Π_large, u, v, nothing, grid, TopHatKernel(), 30e3)

    fig = CairoMakie.Figure(; size=(1400, 400), fontsize=14)
    CairoMakie.Label(fig[0, 1:8], "Cross-Scale Energy Flux Π(x, ℓ)";
                      fontsize=18, font=:bold)

    # Speed
    speed = sqrt.(u.^2 .+ v.^2)
    ax1 = CairoMakie.Axis(fig[1, 1]; title="|u| (velocity)", xlabel="x (km)", ylabel="y (km)",
                           aspect=CairoMakie.DataAspect())
    hm1 = CairoMakie.heatmap!(ax1, xs ./ 1e3, ys ./ 1e3, speed; colormap=:viridis)
    CairoMakie.Colorbar(fig[1, 2], hm1; width=12)

    # Π at 10 km
    clim_s = maximum(abs.(Π_small[20:N-20, 20:N-20])) * 0.8
    ax2 = CairoMakie.Axis(fig[1, 3]; title="Π at ℓ = 10 km", xlabel="x (km)", ylabel="y (km)",
                           aspect=CairoMakie.DataAspect())
    hm2 = CairoMakie.heatmap!(ax2, xs ./ 1e3, ys ./ 1e3, Π_small; colormap=:RdBu,
                               colorrange=(-clim_s, clim_s))
    CairoMakie.Colorbar(fig[1, 4], hm2; width=12, label="Π (W/m³)")

    # Π at 30 km
    clim_l = maximum(abs.(Π_large[20:N-20, 20:N-20])) * 0.8
    ax3 = CairoMakie.Axis(fig[1, 5]; title="Π at ℓ = 30 km", xlabel="x (km)", ylabel="y (km)",
                           aspect=CairoMakie.DataAspect())
    hm3 = CairoMakie.heatmap!(ax3, xs ./ 1e3, ys ./ 1e3, Π_large; colormap=:RdBu,
                               colorrange=(-clim_l, clim_l))
    CairoMakie.Colorbar(fig[1, 6], hm3; width=12, label="Π (W/m³)")

    outpath = joinpath(ASSETS_DIR, "energy_flux_pi.png")
    CairoMakie.save(outpath, fig; px_per_unit=2)
    println("Saved: $outpath")
end

# ─── Figure 3: Filtering energy spectrum E(ℓ) ────────────────────────────

function figure_spectrum()
    N = 101
    dx = 1000.0
    geom = CartesianGeometry(dx, dx)
    xs = collect(0.0:dx:(N-1)*dx)
    ys = collect(0.0:dx:(N-1)*dx)
    mask = trues(N, N)
    grid = StructuredGrid(geom, xs, ys, mask)

    L = (N-1) * dx

    # Multi-scale field
    u = zeros(N, N)
    v = zeros(N, N)
    for j in 1:N, i in 1:N
        x, y = xs[i], ys[j]
        u[i,j] = (1.0 * sin(2π * x / L) * cos(2π * y / L) +
                   0.5 * sin(6π * x / L) * cos(6π * y / L) +
                   0.3 * sin(14π * x / L) * cos(14π * y / L))
        v[i,j] = (-1.0 * cos(2π * x / L) * sin(2π * y / L) -
                    0.5 * cos(6π * x / L) * sin(6π * y / L) -
                    0.3 * cos(14π * x / L) * sin(14π * y / L))
    end

    scales = collect(3e3:2e3:45e3)
    result = coarse_grain(u, v, grid; scales=scales, kernel=TopHatKernel())

    fig = CairoMakie.Figure(; size=(800, 500), fontsize=14)
    CairoMakie.Label(fig[0, 1:2], "Filtering Energy Spectrum E(ℓ)";
                      fontsize=18, font=:bold)

    ax = CairoMakie.Axis(fig[1, 1];
                          xlabel="Filter scale ℓ (km)",
                          ylabel="E(ℓ) (J/m³)",
                          xscale=CairoMakie.log10,
                          yscale=CairoMakie.log10)
    CairoMakie.lines!(ax, result.scales ./ 1e3, result.spectrum;
                       linewidth=2.5, color=:steelblue, label="E(ℓ)")
    CairoMakie.scatter!(ax, result.scales ./ 1e3, result.spectrum;
                        markersize=6, color=:steelblue)

    # Mean Π
    mean_Π = [Statistics.mean(abs.(Π)) for Π in result.Π]
    ax2 = CairoMakie.Axis(fig[1, 2];
                           xlabel="Filter scale ℓ (km)",
                           ylabel="⟨|Π|⟩ (W/m³)",
                           xscale=CairoMakie.log10,
                           yscale=CairoMakie.log10)
    CairoMakie.lines!(ax2, result.scales ./ 1e3, mean_Π;
                       linewidth=2.5, color=:firebrick, label="⟨|Π|⟩")
    CairoMakie.scatter!(ax2, result.scales ./ 1e3, mean_Π;
                        markersize=6, color=:firebrick)

    outpath = joinpath(ASSETS_DIR, "energy_spectrum.png")
    CairoMakie.save(outpath, fig; px_per_unit=2)
    println("Saved: $outpath")
end

# ─── Figure 4: Rigid body rotation → Π = 0 ───────────────────────────────

function figure_rigid_rotation()
    N = 101
    dx = 1000.0
    geom = CartesianGeometry(dx, dx)
    xs = collect(-50e3:dx:50e3)
    ys = collect(-50e3:dx:50e3)
    mask = trues(N, N)
    grid = StructuredGrid(geom, xs, ys, mask)

    Ω = 1e-4
    u = [-Ω * y for x in xs, y in ys]
    v = [ Ω * x for x in xs, y in ys]

    Π = zeros(N, N)
    compute_Π!(Π, u, v, nothing, grid, TopHatKernel(), 15e3)

    fig = CairoMakie.Figure(; size=(1000, 400), fontsize=14)
    CairoMakie.Label(fig[0, 1:6], "Rigid-Body Rotation: Π Must Be Zero (Validation)";
                      fontsize=18, font=:bold)

    speed = sqrt.(u.^2 .+ v.^2)
    ax1 = CairoMakie.Axis(fig[1, 1]; title="|u| (rigid rotation)", xlabel="x (km)", ylabel="y (km)",
                           aspect=CairoMakie.DataAspect())
    hm1 = CairoMakie.heatmap!(ax1, xs ./ 1e3, ys ./ 1e3, speed; colormap=:viridis)
    CairoMakie.Colorbar(fig[1, 2], hm1; width=12)

    ax2 = CairoMakie.Axis(fig[1, 3]; title="Π at ℓ = 15 km (should be ≈ 0)", xlabel="x (km)", ylabel="y (km)",
                           aspect=CairoMakie.DataAspect())
    hm2 = CairoMakie.heatmap!(ax2, xs ./ 1e3, ys ./ 1e3, Π; colormap=:RdBu,
                               colorrange=(-1e-12, 1e-12))
    CairoMakie.Colorbar(fig[1, 4], hm2; width=12, label="Π (W/m³)")

    # Histogram of Π values (interior)
    ax3 = CairoMakie.Axis(fig[1, 5]; title="Histogram of Π (interior)", xlabel="Π (W/m³)", ylabel="Count")
    interior_Π = vec(Π[20:N-20, 20:N-20])
    CairoMakie.hist!(ax3, interior_Π; bins=30, color=:steelblue)

    outpath = joinpath(ASSETS_DIR, "rigid_rotation_validation.png")
    CairoMakie.save(outpath, fig; px_per_unit=2)
    println("Saved: $outpath")
end

# ─── Generate all ─────────────────────────────────────────────────────────

println("Generating documentation assets for CoarseGrainingEnergyFluxes.jl...")
println()
figure_filtering_scales()
figure_energy_flux()
figure_spectrum()
figure_rigid_rotation()
println()
println("Done! Assets saved to: $ASSETS_DIR")
