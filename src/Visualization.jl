module Visualization

export plot_Π_map, plot_spectrum

# Parent-owned stub functions. The CairoMakie package extension adds the real methods; without
# CairoMakie loaded these fallbacks raise a helpful error (mirroring the backend-hook stubs in
# `Filtering`). Keeping the generic functions here lets the extension extend rather than shadow them.

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

plot_Π_map(args...; kwargs...) =
    throw(ArgumentError("plot_Π_map requires the CairoMakie extension — run `using CairoMakie`."))
plot_spectrum(args...; kwargs...) =
    throw(ArgumentError("plot_spectrum requires the CairoMakie extension — run `using CairoMakie`."))

end # module
