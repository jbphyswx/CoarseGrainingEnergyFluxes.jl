module CoarseGrainingEnergyFluxesFFTWExt

using CoarseGrainingEnergyFluxes
using FFTW
using StaticArrays

# Extend filtering module
import CoarseGrainingEnergyFluxes.Filtering: filter_field!

# Bessel function approximation for TopHat transfer function: 2 * J1(x)/x
@inline function bessel_j1_over_x(x::T) where {T<:AbstractFloat}
    if abs(x) < T(1e-5)
        return T(0.5) - x^2 / T(16)
    else
        # Standard polynomial approximation or simple Bessel computation
        # In FFTWExt, we can approximate J1(x) for efficiency
        # J1(x) ≈ x/2 - x^3/16 + x^5/384 ...
        # Standard Taylor expansion or simple recurrence:
        ax = abs(x)
        if ax < T(8.0)
            y = x * x
            ans1 = x * (T(0.5) + y * (T(-0.0625) + y * (T(0.0026041666666666665) + y * (T(-5.425347222222222e-5) + y * (T(6.781684027777778e-7))))))
            return ans1 / x
        else
            # Asymptotic expansion for large x
            z = T(0.8) / ax
            y = z * z
            xx = ax - T(2.35619449) # ax - 3π/4
            ans2 = sqrt(T(0.636619772) / ax) * (cos(xx) + z * sin(xx))
            return ans2 / x
        end
    end
end

"""
    filter_field!(out, field, grid, kernel, scale; mask_strategy, workspace, backend)

FFT-accelerated spectral convolution for structured grids under `CartesianGeometry`.
Extremely fast for large scales and large grids.
"""
function filter_field!(
    out::AbstractMatrix{T},
    field::AbstractMatrix{T},
    grid::StructuredGrid{CartesianGeometry{T},T},
    kernel::AbstractFilterKernel,
    scale::T;
    mask_strategy::Symbol = :renormalize,
    workspace = nothing,
    backend::AbstractExecutionBackend = SerialBackend()
) where {T<:AbstractFloat}
    
    Nlon, Nlat = size_tuple(grid)
    dx = grid.geometry.dx
    dy = grid.geometry.dy
    
    # 1. Take FFT of input field
    # (Note: FFTW operates on Complex fields, so we convert input to Complex)
    field_c = Complex{T}.(field)
    
    # Pre-plan FFT for maximum speed
    plan = plan_fft!(field_c; flags=FFTW.ESTIMATE)
    plan * field_c # In-place forward FFT
    
    # 2. Get wave numbers kx, ky
    # k_i = 2π * i / L
    Lx = Nlon * dx
    Ly = Nlat * dy
    
    kx = FFTW.fftfreq(Nlon, T(2π) / dx)
    ky = FFTW.fftfreq(Nlat, T(2π) / dy)
    
    # 3. Apply transfer function in spectral space
    for j in 1:Nlat
        ky_val = ky[j]
        for i in 1:Nlon
            kx_val = kx[i]
            k_mag = sqrt(kx_val^2 + ky_val^2)
            
            # Compute kernel transfer function G_hat(k)
            if kernel isa TopHatKernel
                # TopHat in 2D Fourier space is 2 * J1(k*R) / (k*R) where R = ℓ / 2
                kR = k_mag * scale / T(2)
                G_hat = T(2) * bessel_j1_over_x(kR)
            elseif kernel isa GaussianKernel
                # Gaussian transfer function: exp(-k² ℓ² / 24)
                G_hat = exp(-k_mag^2 * scale^2 / T(24))
            elseif kernel isa SharpSpectralKernel
                # Sharp cut-off: 1 if k ≤ π/ℓ, 0 otherwise
                k_cutoff = T(π) / scale
                G_hat = k_mag <= k_cutoff ? one(T) : zero(T)
            else
                # Default fallback (Gaussian-like)
                G_hat = exp(-k_mag^2 * scale^2 / T(24))
            end
            
            field_c[i, j] *= G_hat
        end
    end
    
    # 4. In-place Inverse FFT
    inv_plan = plan_ifft!(field_c; flags=FFTW.ESTIMATE)
    inv_plan * field_c
    
    # 5. Extract real part and write back to out (respecting wet points)
    for j in 1:Nlat
        for i in 1:Nlon
            if iswet(grid, i, j)
                out[i, j] = real(field_c[i, j])
            else
                out[i, j] = zero(T)
            end
        end
    end
    
    return out
end

end # module
