module CoarseGrainingEnergyFluxesCairoMakieExt

using CairoMakie: CairoMakie
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF

# Methods for the parent-owned visualization stubs (`CGEF.plot_Π_map`, `CGEF.plot_spectrum`). Every
# Makie call stays qualified (`CairoMakie.Figure`, …) per the package's explicit-import policy.

"""
    plot_Π_map(res, scale_idx, grid; colormap=:balance, title=nothing) -> Figure

Heatmap of the energy-flux map `res.Π[scale_idx]` with a symmetric divergent color range so forward
(Π>0) and inverse (Π<0) cascade read at a glance.
"""
function CGEF.plot_Π_map(
    res::CGEF.CoarseGrainResult{T},
    scale_idx::Integer,
    grid::CGEF.StructuredGrid{G,T};
    colormap = :balance,
    title = nothing,
) where {T<:AbstractFloat, G}
    1 <= scale_idx <= length(res.scales) || throw(BoundsError(res.scales, scale_idx))

    Π_map = res.Π[scale_idx]
    scale = res.scales[scale_idx]
    spherical = G <: CGEF.SphericalGeometry

    fig = CairoMakie.Figure(size = (800, 650))
    ax = CairoMakie.Axis(
        fig[1, 1];
        xlabel = spherical ? "Longitude (rad)" : "X (m)",
        ylabel = spherical ? "Latitude (rad)" : "Y (m)",
        title = title === nothing ?
            "Energy flux Π at ℓ = $(round(scale / 1000; digits = 1)) km" : title,
    )

    # Symmetric limits centered on zero for the divergent cascade colormap.
    max_val = maximum(abs, Π_map)
    colorrange = max_val > 0 ? (-max_val, max_val) : (-one(T), one(T))

    hm = CairoMakie.heatmap!(ax, grid.lon, grid.lat, Π_map; colormap = colormap, colorrange = colorrange)
    CairoMakie.Colorbar(fig[1, 2], hm; label = "Π (W m⁻³)")
    return fig
end

"""
    plot_spectrum(res; which=:density) -> Figure

`which = :density` plots the filtering spectral density `Ẽ(k_ℓ)` vs filtering wavenumber `k_ℓ`
(log x, linear y — the density may dip negative near the endpoints); `which = :cumulative` plots the
cumulative coarse KE vs scale `ℓ` on log–log axes.
"""
function CGEF.plot_spectrum(res::CGEF.CoarseGrainResult{T}; which::Symbol = :density) where {T<:AbstractFloat}
    fig = CairoMakie.Figure(size = (700, 500))
    if which === :cumulative
        ax = CairoMakie.Axis(
            fig[1, 1];
            xscale = log10, yscale = log10,
            xlabel = "Scale ℓ (m)", ylabel = "Cumulative coarse KE  ½ρ₀⟨|ū_ℓ|²⟩ (J m⁻³)",
            title = "Cumulative coarse-grained kinetic energy",
        )
        CairoMakie.lines!(ax, res.scales, res.cumulative_energy; color = :royalblue, linewidth = 2.5)
        CairoMakie.scatter!(ax, res.scales, res.cumulative_energy; color = :royalblue, markersize = 8)
    elseif which === :density
        ax = CairoMakie.Axis(
            fig[1, 1];
            xscale = log10,
            xlabel = "Filtering wavenumber k_ℓ", ylabel = "Filtering spectral density Ẽ(k_ℓ)",
            title = "Filtering spectrum (Sadek & Aluie 2018)",
        )
        CairoMakie.lines!(ax, res.wavenumber, res.filtering_spectrum; color = :firebrick, linewidth = 2.5)
        CairoMakie.scatter!(ax, res.wavenumber, res.filtering_spectrum; color = :firebrick, markersize = 8)
    else
        throw(ArgumentError("`which` must be :density or :cumulative, got :$which"))
    end
    return fig
end

end # module
