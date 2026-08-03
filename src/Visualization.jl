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

# Catch-alls reporting why a call did not land on an extension method: either the extension is absent,
# or it is loaded and these argument types have no method in it. Reporting only the first would send a
# caller who already has CairoMakie loaded chasing an import they have.
function _no_plot_method(name::Symbol, args)
    loaded = Base.get_extension(parentmodule(@__MODULE__), :CoarseGrainingEnergyFluxesCairoMakieExt) !== nothing
    throw(ArgumentError(loaded ?
        "$name has no method for $(map(typeof, args)); the CairoMakie extension plots a " *
        "`CoarseGrainResult` over a `StructuredGrid`." :
        "$name requires the CairoMakie extension — run `using CairoMakie`."))
end

plot_Π_map(args...; kwargs...) = _no_plot_method(:plot_Π_map, args)
plot_spectrum(args...; kwargs...) = _no_plot_method(:plot_spectrum, args)

end # module
