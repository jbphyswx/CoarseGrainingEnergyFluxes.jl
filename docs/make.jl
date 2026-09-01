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
    # The reference is one `@autodocs` block per submodule, one submodule per page. As a single page
    # it passed Documenter's hard size threshold, which is a page nobody scrolls as much as a build
    # failure.
    format = Documenter.HTML(; size_threshold = 400 * 1024, size_threshold_warn = 250 * 1024),
    checkdocs = :exports,
    pages = [
        "Home" => "index.md",
        "Theory" => "theory.md",
        "Architecture" => "architecture.md",
        "Examples" => "examples.md",
        "API Reference" => [
            "Overview" => "reference.md",
            "Pipeline" => "reference/pipeline.md",
            "Diagnostics" => "reference/diagnostics.md",
            "Filtering" => "reference/filtering.md",
            "Kernels and derivatives" => "reference/kernels.md",
        ],
    ],
)

Documenter.deploydocs(;
    repo = "github.com/jbphyswx/CoarseGrainingEnergyFluxes.jl.git",
    devbranch = "main",
    push_preview = true,
)
