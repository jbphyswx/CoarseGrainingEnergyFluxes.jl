module CoarseGrainingEnergyFluxesFastTransformsExt

using CoarseGrainingEnergyFluxes
using FastTransforms
using StaticArrays

import CoarseGrainingEnergyFluxes.Filtering: fasttransforms_filter_field!

"""
    fasttransforms_filter_field!(out, field, grid, kernel, scale; mask_strategy, workspace)

FastTransforms-based spherical harmonics filtering for StructuredGrid with SphericalGeometry.

This uses spherical harmonic transforms (Driscoll-Healy algorithm) to perform fast O(N log N)
spectral filtering on the sphere, following Aluie (2019).

The algorithm:
1. Transform field to spherical harmonic coefficients (sph2fourier)
2. Apply filter transfer function in spectral space
3. Transform back to physical space (fourier2sph)

# References
- Aluie (2019): Convolutions on the sphere: commutation with differential operators
- Driscoll & Healy (1994): Computing Fourier Transforms and Convolutions on the 2-Sphere
"""
function fasttransforms_filter_field!(
    out::AbstractMatrix{T},
    field::AbstractMatrix,
    grid::StructuredGrid{G,T},
    kernel::AbstractFilterKernel,
    scale::T;
    mask_strategy::Symbol = :renormalize,
    workspace = nothing
) where {T<:AbstractFloat, G<:SphericalGeometry{T}}
    
    Nlon, Nlat = size_tuple(grid)
    
    # FastTransforms requires N×N grid for spherical harmonics
    # For non-square grids, we need to pad or use alternative approach
    N = max(Nlon, Nlat)
    
    # Create padded field for spherical harmonics transform
    field_padded = zeros(T, N, N)
    
    # Copy original field data (center it in the padded array)
    lon_start = (N - Nlon) ÷ 2 + 1
    lat_start = (N - Nlat) ÷ 2 + 1
    
    for j in 1:Nlat
        for i in 1:Nlon
            if iswet(grid, i, j)
                field_padded[lat_start + j - 1, lon_start + i - 1] = field[i, j]
            end
        end
    end
    
    # Step 1: Transform to Fourier (bivariate Fourier series on sphere)
    F = sph2fourier(field_padded)
    
    # Step 2: Apply filter transfer function in spectral space
    # For each mode (m, n), multiply by filter transfer function
    for n in 1:N  # latitude modes
        for m in 1:N  # longitude modes
            # Compute wavenumber magnitude
            # On sphere: k^2 = l(l+1)/R^2 where l is spherical harmonic degree
            l = min(m, n) - 1  # spherical harmonic degree approximation
            k_mag = sqrt(l * (l + 1)) / grid.geometry.R
            
            # Apply filter transfer function
            if kernel isa TopHatKernel
                kR = k_mag * scale / 2
                G_hat = 2 * besselj1(kR * 2) / (kR * 2)  # 2*J1(kR)/(kR)
                if kR == 0
                    G_hat = one(T)
                end
            elseif kernel isa GaussianKernel
                G_hat = exp(-k_mag^2 * scale^2 / 24)
            elseif kernel isa SharpSpectralKernel
                k_cutoff = T(π) / scale
                G_hat = k_mag <= k_cutoff ? one(T) : zero(T)
            else
                G_hat = exp(-k_mag^2 * scale^2 / 24)  # Default to Gaussian
            end
            
            F[m, n] *= G_hat
        end
    end
    
    # Step 3: Transform back to physical space
    field_filtered = fourier2sph(F)
    
    # Extract filtered field (from padded array back to original size)
    fill!(out, zero(T))
    for j in 1:Nlat
        for i in 1:Nlon
            if iswet(grid, i, j)
                out[i, j] = field_filtered[lat_start + j - 1, lon_start + i - 1]
            end
        end
    end
    
    return out
end

# Fallback for Cartesian geometry - use FINUFFT or serial instead
function fasttransforms_filter_field!(
    out::AbstractMatrix{T},
    field::AbstractMatrix,
    grid::StructuredGrid{G,T},
    kernel::AbstractFilterKernel,
    scale::T;
    mask_strategy::Symbol = :renormalize,
    workspace = nothing
) where {T<:AbstractFloat, G<:CartesianGeometry{T}}
    error("FastTransforms extension only supports SphericalGeometry. Use FINUFFTBackend or SerialBackend for Cartesian grids.")
end

end # module
