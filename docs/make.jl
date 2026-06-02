using Documenter: Documenter
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes

Documenter.makedocs(;
    modules=[CoarseGrainingEnergyFluxes],
    sitename="CoarseGrainingEnergyFluxes.jl",
    pages=[
        "Home" => "index.md",
        "Theory" => "theory.md",
        "Architecture" => "architecture.md",
        "Examples" => "examples.md",
    ],
)
