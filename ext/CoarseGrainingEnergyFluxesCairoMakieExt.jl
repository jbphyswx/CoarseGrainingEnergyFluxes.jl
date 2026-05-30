module CoarseGrainingEnergyFluxesCairoMakieExt

using CoarseGrainingEnergyFluxes
using CairoMakie

export plot_Π_map, plot_spectrum

"""
    plot_Π_map(res, scale_idx, grid; kwargs...)

Create and save a high-quality heatmap visualization of the 2D energy transfer Π map at the specified scale index.
"""
function plot_Π_map(
    res::CoarseGrainResult{T},
    scale_idx::Integer,
    grid::StructuredGrid{G,T};
    colormap = :balance,
    title = nothing
) where {T<:AbstractFloat, G}
    
    1 <= scale_idx <= length(res.scales) || throw(BoundsError(res.scales, scale_idx))
    
    Π_map = res.Π[scale_idx]
    scale = res.scales[scale_idx]
    
    # Pre-allocate layout and figure
    fig = Figure(resolution = (800, 650))
    ax = Axis(
        fig[1, 1],
        xlabel = G <: SphericalGeometry ? "Longitude (rad)" : "X (meters)",
        ylabel = G <: SphericalGeometry ? "Latitude (rad)" : "Y (meters)",
        title = title === nothing ? "Kinetic Energy Flux (Π) at Scale ℓ = $(round(scale/1000, digits=1)) km" : title
    )
    
    # Symmetric limits for divergent/cascade colormap (:balance)
    max_val = maximum(abs.(Π_map))
    limits = (-max_val, max_val)
    
    hm = heatmap!(ax, grid.lon, grid.lat, Π_map, colormap = colormap, colorrange = limits)
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
