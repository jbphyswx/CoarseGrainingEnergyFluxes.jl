module CoarseGrainingEnergyFluxesKernelAbstractionsAbstractFFTsExt

using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF

# Device arrays take CGEF's FFT engines (`CoarseGrainingEnergyFluxesAbstractFFTsExt`) through the
# AbstractFFTs implementation of their array type (CUDA's CUFFT for a `CuArray`; `KA.CPU()` allocates host
# arrays, which take the host provider).
CGEF.Filtering._available(::CGEF.Filtering._DeviceFFT) = true

end # module CoarseGrainingEnergyFluxesKernelAbstractionsAbstractFFTsExt
