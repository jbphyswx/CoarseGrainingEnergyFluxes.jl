module CoarseGrainingEnergyFluxesCairoMakieExt

using CoarseGrainingEnergyFluxes
using CairoMakie

export plot_Pi_map, plot_spectrum

"""
    plot_Pi_map(res, scale_idx, grid; kwargs...)

Create and save a high-quality heatmap visualization of the 2D energy transfer Π map at the specified scale index.
"""
function plot_Pi_map(
    res::CoarseGrainResult{T},
    scale_idx::Integer,
    grid::StructuredGrid{G,T};
    colormap = :balance,
    title = nothing
) where {T<:AbstractFloat, G}
    
    1 <= scale_idx <= length(res.scales) || throw(BoundsError(res.scales, scale_idx))
    
    Pi_map = res.Pi[scale_idx]
    scale = res.scales[scale_idx]
    
    # Pre-allocate layout and figure
    fig = Figure(resolution = (800, 650))
    ax = Axis(
        fig[1, 1],
        xlabel = G <: SphericalGeometry ? "Longitude (rad)" : "X (meters)",
        ylabel = G <: SphericalGeometry ? "Latitude (rad)" : "Y (meters)",
        title = title === nothing ? "Kinetic Energy Flux (Π) at Scale ℓ = $(round(scale/1000, digits=1)) km" : title
    )
    
    # Symmetric limits for divergant/cascade colormap (:balance)
    max_val = maximum(abs.(Pi_map))
    limits = (-max_val, max_val)
    
    hm = heatmap!(ax, grid.lon, grid.lat, Pi_map, colormap = colormap, colorrange = limits)
    Colorbar(fig[1, 2], hm, label = "Π (W / m³)")
    
    return fig
end

"""
    plot_spectrum(res; kwargs...)

Plot the filtering energy spectrum E(ℓ) vs scale ℓ on log-log axes.
"""
function plot_spectrum(res::CoarseGrainResult{T}) where {T<:AbstractFloat}
    fig = Figure(resolution = (700, 500))
    ax = Axis(
        fig[1, 1],
        xscale = log10,
        yscale = log10,
        xlabel = "Scale ℓ (meters)",
        ylabel = "Filtering Energy Spectrum E(ℓ) (m²/s²)",
        title = "Filtering Kinetic Energy Spectrum"
    )
    
    lines!(ax, res.scales, res.spectrum, color = :royalblue, linewidth = 2.5)
    scatter!(ax, res.scales, res.spectrum, color = :royalblue, markersize = 8)
    
    return fig
end

end # module
