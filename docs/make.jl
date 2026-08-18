using Documenter: Documenter
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes

const CGEF = CoarseGrainingEnergyFluxes

Documenter.makedocs(;
    modules = [
        CGEF,
        CGEF.Kernels, CGEF.Filtering, CGEF.Derivatives, CGEF.Diagnostics,
        CGEF.Pipeline, CGEF.Visualization,
    ],
    sitename = "CoarseGrainingEnergyFluxes.jl",
    # The API reference is one `@autodocs` block per submodule and comes out around 220 KiB, over
    # Documenter's 200 KiB default. That default guards against a page bloated by accident; a complete
    # reference for this many exported symbols is not that.
    format = Documenter.HTML(; size_threshold = 400 * 1024, size_threshold_warn = 250 * 1024),
    checkdocs = :exports,
    pages = [
        "Home" => "index.md",
        "Theory" => "theory.md",
        "Requirements from the literature" => "requirements_from_literature.md",
        "Architecture" => "architecture.md",
        "Examples" => "examples.md",
        "API Reference" => "reference.md",
        "Development plan" => "development_plan.md",
    ],
)

Documenter.deploydocs(;
    repo = "github.com/jbphyswx/CoarseGrainingEnergyFluxes.jl.git",
    devbranch = "main",
    push_preview = true,
)
