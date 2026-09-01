module Diagnostics

using FlowGeometries: FlowGeometries
using ..Kernels: Kernels
using ..Filtering: Filtering
using ..Derivatives: Derivatives
using ComputationalBackends: ComputationalBackends

export ΠWorkspace, EnergyWorkspace, compute_Π!, cumulative_energy, cumulative_energy!, filtering_spectrum, spectral_density, spectral_density!
export tau_decomposition, tau_decomposition!, TauWorkspace, Sym3TauWorkspace
export compute_Π_decomposed, compute_Π_decomposed!, PiDecomposedWorkspace, PiDecomposed3DWorkspace
export SphericalPiDecomposedWorkspace, SphericalPiDecomposed3DWorkspace
export tracer_variance_flux
export compute_Π_strain_convergence, compute_Π_strain_convergence!, PiStrainWorkspace
export vorticity, vorticity!, enstrophy_flux, enstrophy_flux!, EnstrophyFluxWorkspace
export TracerFluxWorkspace, SphericalTracerFluxWorkspace, TracerFlux3DWorkspace
export tracer_variance_flux!
export band_energies
export compressible_flux, compressible_flux!, favre_filter!
export FavreWorkspace, SphericalFavreWorkspace, Favre3DWorkspace
export AbstractSpectrumPolicy, StrictSpectrum, ForceSpectrum, NoSpectrum

include("Diagnostics/SpectrumPolicy.jl")
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

end # module
