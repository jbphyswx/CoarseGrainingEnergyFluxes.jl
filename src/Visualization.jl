module Visualization

export plot_Π_map, plot_spectrum

"""
    plot_Π_map(res, scale_idx, grid; colormap=:balance, title=nothing) -> Figure

Heatmap of the cross-scale energy-flux map `Π` from a `CoarseGrainResult` at scale index
`scale_idx`. Provided by the **CairoMakie** package extension — run `using CairoMakie` to enable it.
"""
function plot_Π_map end

"""
    plot_spectrum(res; which=:density) -> Figure

Plot the filtering spectrum from a `CoarseGrainResult`. `which = :density` plots the filtering
spectral density `Ẽ(k_ℓ)` against filtering wavenumber `k_ℓ` (log x); `which = :cumulative` plots the
cumulative coarse KE against scale `ℓ` (log–log). Provided by the **CairoMakie** package extension —
run `using CairoMakie` to enable it.
"""
function plot_spectrum end

# Whether the CairoMakie extension, which defines the plotting methods, is loaded: `true` from it.
_plotting_loaded() = false

function _no_plot_method(name::Symbol, args)
    throw(ArgumentError(_plotting_loaded() ?
        "$name has no method for $(map(typeof, args)); the CairoMakie extension plots a " *
        "`CoarseGrainResult` over a `StructuredGrid`." :
        "$name requires the CairoMakie extension — run `using CairoMakie`."))
end

plot_Π_map(args...; kwargs...) = _no_plot_method(:plot_Π_map, args)
plot_spectrum(args...; kwargs...) = _no_plot_method(:plot_spectrum, args)

end # module
