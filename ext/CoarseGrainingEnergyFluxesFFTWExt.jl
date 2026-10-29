module CoarseGrainingEnergyFluxesFFTWExt

using FFTW: FFTW
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF

# FFTW implements AbstractFFTs for host arrays, so CGEF's FFT engines (`CoarseGrainingEnergyFluxesAbstractFFTsExt`)
# run through it there, each plan at the execution backend's thread count.

CGEF.Filtering._available(::CGEF.Filtering._HostFFT) = true

CGEF.Filtering._fft_options(::StridedArray{<:FFTW.fftwNumber}, nthreads::Int, options::NamedTuple) =
    (; num_threads = nthreads, options...)

end # module CoarseGrainingEnergyFluxesFFTWExt
