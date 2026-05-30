module CoarseGrainingEnergyFluxesCSVExt

using CoarseGrainingEnergyFluxes
using CSV

export save_spectrum_csv

"""
    save_spectrum_csv(filepath, res)

Save the 1D multiscale filtering energy spectrum `E(ℓ)` to a CSV file.
"""
function save_spectrum_csv(
    filepath::AbstractString,
    res::CoarseGrainResult{T}
) where {T<:AbstractFloat}
    
    # Construct rows matching standard CSV requirements
    rows = []
    for s in eachindex(res.scales)
        push!(rows, (scale = res.scales[s], energy_spectrum = res.spectrum[s]))
    end
    
    CSV.write(filepath, rows)
    return filepath
end

end # module
