module CoarseGrainingEnergyFluxesSpecialFunctionsExt

using SpecialFunctions: SpecialFunctions
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF

# The disk's transform `2J₁(x)/x`, 1 at `x = 0` (J₁(x)/x → 1/2).
@inline function CGEF.Kernels._ball_transform(x::T, ::Val{2}) where {T<:AbstractFloat}
    iszero(x) && return one(T)
    return T(2) * SpecialFunctions.besselj1(x) / x
end

# The four-dimensional ball's, `8J₂(x)/x²`, 1 at `x = 0` (J₂(x)/x² → 1/8): the disk's `ℓ`-derivative reads it.
@inline function CGEF.Kernels._ball_transform(x::T, ::Val{4}) where {T<:AbstractFloat}
    iszero(x) && return one(T)
    return T(8) * SpecialFunctions.besselj(2, x) / x^2
end

# The six-dimensional ball's, `48J₃(x)/x³`, 1 at `x = 0`: the disk's second `ℓ`-derivative reads it.
@inline function CGEF.Kernels._ball_transform(x::T, ::Val{6}) where {T<:AbstractFloat}
    iszero(x) && return one(T)
    return T(48) * SpecialFunctions.besselj(3, x) / x^3
end

end # module
