module CoarseGrainingEnergyFluxesSpecialFunctionsExt

using SpecialFunctions: SpecialFunctions
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF

# Exact 2D (disk) Fourier transform of a top-hat kernel of radius R = ℓ/2: the "jinc" function
# 2*J₁(kR)/(kR), normalized to 1 at k = 0 (lim_{x->0} J₁(x)/x = 1/2). Adds the one method the FFTW/
# FINUFFT extensions are missing — neither of them needs to know SpecialFunctions exists; Julia's
# global method dispatch means they pick this up automatically once this extension is loaded.
@inline function CGEF.Kernels.spectral_transfer(
    ::CGEF.Kernels.TopHatKernel, kmag::T, ℓ::T,
) where {T<:AbstractFloat}
    R = ℓ / T(2)
    x = kmag * R
    iszero(x) && return one(T)
    return T(2) * SpecialFunctions.besselj1(x) / x
end

end # module
