using Documenter: Documenter
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes

const CGEF = CoarseGrainingEnergyFluxes

Documenter.makedocs(;
    modules = [
        CGEF,
        CGEF.Backends, CGEF.Geometry, CGEF.Grids, CGEF.Kernels, CGEF.Filtering,
        CGEF.Derivatives, CGEF.Diagnostics, CGEF.Pipeline, CGEF.Visualization,
    ],
    sitename = "CoarseGrainingEnergyFluxes.jl",
    checkdocs = :exports,
    pages = [
        "Home" => "index.md",
        "Theory" => "theory.md",
        "Architecture" => "architecture.md",
        "Examples" => "examples.md",
        "API Reference" => "reference.md",
    ],
)

Documenter.deploydocs(;
    repo = "github.com/jbphyswx/CoarseGrainingEnergyFluxes.git",
    devbranch = "main",
    push_preview = true,
)
