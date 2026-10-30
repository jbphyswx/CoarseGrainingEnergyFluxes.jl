module Diagnostics

using FlowGeometries: FlowGeometries
using ..Kernels: Kernels
using ..Filtering: Filtering
using ..Derivatives: Derivatives
using ComputationalBackends: ComputationalBackends
using SpectralBackends: SpectralBackends

export ΠWorkspace, EnergyWorkspace, compute_Π!, cumulative_energy, cumulative_energy!, filtering_spectrum, spectral_density, spectral_density!
export tau_decomposition, tau_decomposition!, TauWorkspace, Sym3TauWorkspace
export compute_Π_decomposed, compute_Π_decomposed!, PiDecomposedWorkspace, PiDecomposed3DWorkspace
export SphericalPiDecomposedWorkspace, SphericalPiDecomposed3DWorkspace
export tracer_variance_flux
export compute_Π_strain_convergence, compute_Π_strain_convergence!, PiStrainWorkspace
export vorticity, vorticity!, enstrophy_flux, enstrophy_flux!, EnstrophyFluxWorkspace
export TracerFluxWorkspace, SphericalTracerFluxWorkspace, TracerFlux3DWorkspace
export tracer_variance_flux!
export band_energies, band_energies!, BandWorkspace, BandMaps
export compressible_flux, compressible_flux!, favre_filter!
export FavreWorkspace, SphericalFavreWorkspace, Favre3DWorkspace
export AbstractSpectrumPolicy, AutoSpectrum, StrictSpectrum, ForceSpectrum, NoSpectrum
export AbstractScaleDerivative, AutoScaleDerivative, ExactScaleDerivative, DifferenceScaleDerivative
export AbstractSlopes, NoSlopes, Slopes, AbstractSpectrumMaps, NoSpectrumMaps, SpectrumMaps
export AbstractFilterPass, LowPass, HighPass

include("Diagnostics/SpectrumPolicy.jl")
include("Diagnostics/SpinFilter.jl")
include("Diagnostics/Flux.jl")
include("Diagnostics/TensorDriver.jl")
include("Diagnostics/Spectrum.jl")
include("Diagnostics/Stress.jl")
include("Diagnostics/Helmholtz.jl")
include("Diagnostics/StrainConvergence.jl")
include("Diagnostics/Tracer.jl")
include("Diagnostics/Favre.jl")
include("Diagnostics/Bands.jl")
include("Diagnostics/Enstrophy.jl")

# A workspace holding a generalized convolution reports its fits; one without, `(true, 0, 0)`.
Filtering.analysis_status(ws::Union{ΠWorkspace, EnergyWorkspace, BandWorkspace, SphericalFavreWorkspace,
                                    SphericalPiDecomposedWorkspace, Sym3TauWorkspace, SphericalTracerFluxWorkspace}) =
    _spin_status(ws.spin)
_spin_status(sf::SpinFilter) = Filtering.analysis_status(sf)
_spin_status(::Nothing) = Filtering._exact_status(Float64)

end # module
