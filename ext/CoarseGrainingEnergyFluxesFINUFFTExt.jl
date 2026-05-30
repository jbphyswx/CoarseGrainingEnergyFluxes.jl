module CoarseGrainingEnergyFluxesFINUFFTExt

using CoarseGrainingEnergyFluxes
using FINUFFT
using StaticArrays

# Extend filtering module
import CoarseGrainingEnergyFluxes.Filtering: filter_field!

# Bessel function approximation for TopHat transfer function: 2 * J1(x)/x
@inline function bessel_j1_over_x(x::T) where {T<:AbstractFloat}
    if abs(x) < T(1e-5)
        return T(0.5) - x^2 / T(16)
    else
        ax = abs(x)
        if ax < T(8.0)
            y = x * x
            ans1 = x * (T(0.5) + y * (T(-0.0625) + y * (T(0.0026041666666666665) + y * (T(-5.425347222222222e-5) + y * (T(6.781684027777778e-7))))))
            return ans1 / x
        else
            z = T(0.8) / ax
            y = z * z
            xx = ax - T(2.35619449)
            ans2 = sqrt(T(0.636619772) / ax) * (cos(xx) + z * sin(xx))
            return ans2 / x
        end
    end
end

"""
    filter_field!(out, field, grid, kernel, scale; mask_strategy, workspace, backend)

FINUFFT-accelerated non-uniform spectral filtering for Curvilinear or Unstructured grids.
Transforms non-uniform coordinates to Fourier space, filters analytically, and transforms back.
"""
function filter_field!(
    out::AbstractMatrix{T},
    field::AbstractMatrix{T},
    grid::CurvilinearGrid{G,T},
    kernel::AbstractFilterKernel,
    scale::T;
    mask_strategy::Symbol = :renormalize,
    workspace = nothing,
    backend::AbstractExecutionBackend = SerialBackend()
) where {T<:AbstractFloat, G<:AbstractGeometry{T}}
    
    Nlon, Nlat = size_tuple(grid)
    
    # 1. Extract wet point coordinates and values
    wet_coords_x = T[]
    wet_coords_y = T[]
    wet_vals = Complex{T}[]
    
    # Track mapping back to grid indices
    mapping = Tuple{Int,Int}[]
    
    # Rescale coordinates to [-π, π] for FINUFFT compatibility
    # Fetch coordinates
    min_x, max_x = T(Inf), T(-Inf)
    min_y, max_y = T(Inf), T(-Inf)
    
    for j in 1:Nlat
        for i in 1:Nlon
            if iswet(grid, i, j)
                x = grid.lon[i, j]
                y = grid.lat[i, j]
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
            end
        end
    end
    
    span_x = max_x - min_x
    span_y = max_y - min_y
    
    for j in 1:Nlat
        for i in 1:Nlon
            if iswet(grid, i, j)
                # Normalize coords to [-3.0, 3.0] to stay safely inside [-π, π]
                norm_x = (grid.lon[i, j] - min_x) / span_x * T(6.0) - T(3.0)
                norm_y = (grid.lat[i, j] - min_y) / span_y * T(6.0) - T(3.0)
                
                push!(wet_coords_x, norm_x)
                push!(wet_coords_y, norm_y)
                push!(wet_vals, Complex{T}(field[i, j]))
                push!(mapping, (i, j))
            end
        end
    end
    
    Npoints = length(wet_vals)
    if Npoints == 0
        fill!(out, zero(T))
        return out
    end
    
    # 2. Define target Fourier grid resolution
    M = nextprod([2, 3, 5], Nlon)
    N = nextprod([2, 3, 5], Nlat)
    
    # 3. Type 1 FINUFFT (Non-uniform points to uniform Fourier grid coefficients)
    # nufft2d1: non-uniform points (x, y) with strengths c -> uniform Fourier coefficients F
    F = nufft2d1(wet_coords_x, wet_coords_y, wet_vals, 1, T(1e-6), M, N)
    
    # 4. Multiply by filter transfer function in Fourier space
    # Wave number scaling factor (since we mapped physical coordinate span to 6.0 in normalized space)
    # The physical wave numbers matching Fourier grid coordinates (k1, k2) are:
    # k_x = k1 * (6.0 / span_x), k_y = k2 * (6.0 / span_y)
    
    # Fourier coefficients are arranged in FFT shift order
    kx = collect(-M/2 : M/2 - 1)
    ky = collect(-N/2 : N/2 - 1)
    
    for nj in 1:N
        k_y = ky[nj] * (T(6.0) / span_y)
        for mi in 1:M
            k_x = kx[mi] * (T(6.0) / span_x)
            
            k_mag = sqrt(k_x^2 + k_y^2)
            
            # Compute kernel transfer function G_hat(k)
            if kernel isa TopHatKernel
                kR = k_mag * scale / T(2)
                G_hat = T(2) * bessel_j1_over_x(kR)
            elseif kernel isa GaussianKernel
                G_hat = exp(-k_mag^2 * scale^2 / T(24))
            elseif kernel isa SharpSpectralKernel
                k_cutoff = T(π) / scale
                G_hat = k_mag <= k_cutoff ? one(T) : zero(T)
            else
                G_hat = exp(-k_mag^2 * scale^2 / T(24))
            end
            
            F[mi, nj] *= G_hat
        end
    end
    
    # 5. Type 2 FINUFFT (Uniform Fourier coefficients back to original non-uniform coordinates)
    filtered_vals = nufft2d2(wet_coords_x, wet_coords_y, -1, T(1e-6), F)
    
    # 6. Extract real part and write back to out
    fill!(out, zero(T))
    for idx in 1:Npoints
        i, j = mapping[idx]
        out[i, j] = real(filtered_vals[idx]) / (M * N)
    end
    
    return out
end

end # module
