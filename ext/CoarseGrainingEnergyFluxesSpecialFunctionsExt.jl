module CoarseGrainingEnergyFluxesSpecialFunctionsExt

using SpecialFunctions: SpecialFunctions
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF

# The disk's transform `2J₁(x)/x`, 1 at `x = 0` (J₁(x)/x → 1/2).
@inline function CGEF.Kernels._ball_transform(x::T, ::Val{2}) where {T<:AbstractFloat}
    iszero(x) && return one(T)
    return T(2) * SpecialFunctions.besselj1(x) / x
end

end # module
