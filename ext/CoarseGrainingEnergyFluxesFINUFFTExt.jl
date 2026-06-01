module CoarseGrainingEnergyFluxesFINUFFTExt

using CoarseGrainingEnergyFluxes
using FINUFFT
using StaticArrays

# Extend filtering module
import CoarseGrainingEnergyFluxes.Filtering: finufft_filter_field!

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
    finufft_filter_field!(out, field, grid, kernel, scale; mask_strategy, workspace)

FINUFFT-accelerated non-uniform spectral filtering for StructuredGrid.
"""
function finufft_filter_field!(
    out::AbstractMatrix{T},
    field::AbstractMatrix,
    grid::StructuredGrid{G,T},
    kernel::AbstractFilterKernel,
    scale::T;
    mask_strategy::Symbol = :renormalize,
    workspace = nothing
) where {T<:AbstractFloat, G<:AbstractGeometry{T}}
    
    Nlon, Nlat = size_tuple(grid)
    
    # 1. Extract wet point coordinates and values
    wet_coords_x = T[]
    wet_coords_y = T[]
    wet_vals = Complex{T}[]
    
    # Track mapping back to grid indices
    mapping = Tuple{Int,Int}[]
    
    min_x, max_x = T(Inf), T(-Inf)
    min_y, max_y = T(Inf), T(-Inf)
    
    for j in 1:Nlat
        for i in 1:Nlon
            if iswet(grid, i, j)
                x = grid.lon[i]
                y = grid.lat[j]
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
            end
        end
    end
    
    # For spherical geometry, use area-preserving coordinates:
    # x = λ (longitude), y = sin(φ) (sin of latitude)
    # This makes dA = R² dλ d(sin φ) uniform in transformed space
    if G <: SphericalGeometry{T}
        min_y_sin, max_y_sin = sin(min_y), sin(max_y)
        span_y_eff = max_y_sin - min_y_sin
    else
        span_y_eff = max_y - min_y
    end
    span_x = max_x - min_x
    
    for j in 1:Nlat
        for i in 1:Nlon
            if iswet(grid, i, j)
                # Normalize coords to [-3.0, 3.0] to stay safely inside [-π, π]
                norm_x = (grid.lon[i] - min_x) / span_x * T(6.0) - T(3.0)
                
                # For spherical: use sin(lat) to preserve area
                if G <: SphericalGeometry{T}
                    norm_y = (sin(grid.lat[j]) - min_y_sin) / span_y_eff * T(6.0) - T(3.0)
                else
                    norm_y = (grid.lat[j] - min_y) / span_y_eff * T(6.0) - T(3.0)
                end
                
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
    
    # 2. Define uniform Fourier grid resolution
    M = nextprod([2, 3, 5], Nlon)
    N = nextprod([2, 3, 5], Nlat)
    
    # 3. Type 1 FINUFFT (Non-uniform points to uniform Fourier grid coefficients)
    F = nufft2d1(wet_coords_x, wet_coords_y, wet_vals, 1, T(1e-6), M, N)
    
    # 4. Multiply by filter transfer function in Fourier space
    kx = collect(-M/2 : M/2 - 1)
    ky = collect(-N/2 : N/2 - 1)
    
    # Physical extent in transformed coordinates
    L_x = span_x * T(6.0)  # Normalized domain extent
    L_y = span_y_eff * T(6.0)
    
    for nj in 1:N
        k_y = ky[nj] * (2π / L_y)
        for mi in 1:M
            k_x = kx[mi] * (2π / L_x)
            
            k_mag = sqrt(k_x^2 + k_y^2)
            
            # Apply filter transfer function
            if kernel isa TopHatKernel
                kR = k_mag * scale / T(2)
                G_hat = kR < T(1e-10) ? one(T) : T(2) * bessel_j1_over_x(kR)
            elseif kernel isa GaussianKernel
                σ = scale / sqrt(T(12))
                G_hat = exp(-k_mag^2 * σ^2 / T(2))
            else
                G_hat = one(T)
            end
            
            F[mi, nj] *= G_hat
        end
    end
    
    # 5. Type 2 FINUFFT (back to original non-uniform coordinates)
    filtered_vals = nufft2d2(wet_coords_x, wet_coords_y, -1, T(1e-6), F)
    
    # 6. Extract real part and normalize
    fill!(out, zero(T))
    norm_factor = M * N
    for idx in 1:Npoints
        i, j = mapping[idx]
        out[i, j] = real(filtered_vals[idx]) / norm_factor
    end
    
    return out
end

"""
    finufft_filter_field!(out, field, grid, kernel, scale; mask_strategy, workspace)

FINUFFT-accelerated non-uniform spectral filtering for CurvilinearGrid.
"""
function finufft_filter_field!(
    out::AbstractMatrix{T},
    field::AbstractMatrix,
    grid::CurvilinearGrid{G,T},
    kernel::AbstractFilterKernel,
    scale::T;
    mask_strategy::Symbol = :renormalize,
    workspace = nothing
) where {T<:AbstractFloat, G<:AbstractGeometry{T}}
    
    Nlon, Nlat = size_tuple(grid)
    
    # 1. Extract wet point coordinates and values
    wet_coords_x = T[]
    wet_coords_y = T[]
    wet_vals = Complex{T}[]
    
    # Track mapping back to grid indices
    mapping = Tuple{Int,Int}[]
    
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
    
    # For spherical geometry, use area-preserving coordinates
    if G <: SphericalGeometry{T}
        min_y_sin, max_y_sin = sin(min_y), sin(max_y)
        span_y_eff = max_y_sin - min_y_sin
    else
        span_y_eff = max_y - min_y
    end
    span_x = max_x - min_x
    
    for j in 1:Nlat
        for i in 1:Nlon
            if iswet(grid, i, j)
                norm_x = (grid.lon[i, j] - min_x) / span_x * T(6.0) - T(3.0)
                
                # For spherical: use sin(lat) to preserve area
                if G <: SphericalGeometry{T}
                    norm_y = (sin(grid.lat[i, j]) - min_y_sin) / span_y_eff * T(6.0) - T(3.0)
                else
                    norm_y = (grid.lat[i, j] - min_y) / span_y_eff * T(6.0) - T(3.0)
                end
                
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
    F = nufft2d1(wet_coords_x, wet_coords_y, wet_vals, 1, T(1e-6), M, N)
    
    # 4. Multiply by filter transfer function in Fourier space
    kx = collect(-M/2 : M/2 - 1)
    ky = collect(-N/2 : N/2 - 1)
    
    # Physical extent of the domain
    L_x_phys = span_x * grid.geometry.R
    L_y_phys = span_y_eff * grid.geometry.R
    
    for nj in 1:N
        # Physical wavenumber in y-direction (rad/m)
        k_y = ky[nj] * (2π / L_y_phys)
        for mi in 1:M
            # Physical wavenumber in x-direction (rad/m)
            k_x = kx[mi] * (2π / L_x_phys)
            
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
    # isign=-1 gives adjoint (not inverse), so we need to normalize by (M*N)
    filtered_vals = nufft2d2(wet_coords_x, wet_coords_y, -1, T(1e-6), F)
    
    # 6. Extract real part and write back to out
    # Normalization: Type 2 NUFFT with isign=-1 computes sum_k F_k exp(-i*k*x)
    # which equals M*N times the inverse transform. So divide by (M*N).
    fill!(out, zero(T))
    norm_factor = M * N
    for idx in 1:Npoints
        i, j = mapping[idx]
        out[i, j] = real(filtered_vals[idx]) / norm_factor
    end
    
    return out
end

end # module
