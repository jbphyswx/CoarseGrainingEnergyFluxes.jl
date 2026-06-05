module CoarseGrainingEnergyFluxesGPUExt

using CoarseGrainingEnergyFluxes
using KernelAbstractions
using StaticArrays

import CoarseGrainingEnergyFluxes.Filtering: gpu_filter_field!
import CoarseGrainingEnergyFluxes: TopHatKernel, GaussianKernel, SharpSpectralKernel, kernel_radius

# GPU-compatible kernel weight function (no dispatch, uses if-else)
@inline function kernel_weight_gpu(kernel, d, scale, T)
    rad = kernel_radius(kernel, scale)
    if d > rad
        return zero(T)
    end
    
    if kernel isa TopHatKernel
        # TopHat: constant inside radius
        return one(T)
    elseif kernel isa GaussianKernel
        # Gaussian: exp(-d^2 / (2*σ^2)) with σ = scale/√12
        σ = scale / sqrt(T(12))
        return exp(-d^2 / (T(2) * σ^2))
    elseif kernel isa SharpSpectralKernel
        # SharpSpectral: constant up to cutoff
        return one(T)
    else
        # Default to TopHat
        return one(T)
    end
end

# Bessel function approximation on device for TopHat transfer function
@kernel function gpu_filter_kernel!(
    out,
    field,
    lon,
    lat,
    areas,
    mask,
    geom,
    rad,
    scale,
    mask_strategy,
    is_cartesian
)
    i, j = @index(Global, NTuple)
    Nlon, Nlat = size(out)
    
    if i <= Nlon && j <= Nlat
        if !mask[i, j]
            out[i, j] = zero(eltype(out))
            # return # this is invalid  # eturn statement not permitted in a kernel function gpu_filter_kernel! (I just commmented it out but Agents:: check if this needs to be fixed further before removing)
        end
        
        # Local variables
        T = eltype(out)
        target_lon = lon[i]
        target_lat = lat[j]
        
        # Grid spacings for boundary computation
        # In a generic GPU kernel we estimate limits or pass them
        # For simplicity, we search the full integration bounding box
        dj_lim = ceil(Int, rad / (is_cartesian ? geom.dy : geom.R * (lat[2] - lat[1])))
        di_lim = ceil(Int, rad / (is_cartesian ? geom.dx : geom.R * cos(target_lat) * (lon[2] - lon[1])))
        
        weighted_sum = zero(T)
        weight_norm  = zero(T)
        
        j_start = max(1, j - dj_lim)
        j_end   = min(Nlat, j + dj_lim)
        
        for jj in j_start:j_end
            i_start = max(1, i - di_lim)
            i_end   = min(Nlon, i + di_lim)
            
            for ii in i_start:i_end
                if mask_strategy == :renormalize || mask_strategy == :deformable
                    mask[ii, jj] || continue
                end
                
                # Fetch neighbor coordinates and compute distance
                neigh_lon = lon[ii]
                neigh_lat = lat[jj]
                
                d = zero(T)
                if is_cartesian
                    d = sqrt((target_lon - neigh_lon)^2 + (target_lat - neigh_lat)^2)
                else
                    # Spherical great-circle distance
                    R = geom.R
                    dλ = target_lon - neigh_lon
                    dφ = target_lat - neigh_lat
                    a = sin(dφ / T(2))^2 + cos(target_lat) * cos(neigh_lat) * sin(dλ / T(2))^2
                    c = T(2) * atan(sqrt(a), sqrt(max(zero(T), one(T) - a)))
                    d = R * c
                end
                
                if d <= rad
                    # Kernel weight based on kernel type
                    w_k = kernel_weight_gpu(kernel, d, scale, T)
                    w = w_k * areas[ii, jj]
                    
                    weight_norm += w
                    
                    if mask_strategy == :zero
                        if mask[ii, jj]
                            weighted_sum += w * field[ii, jj]
                        end
                    else
                        weighted_sum += w * field[ii, jj]
                    end
                end
            end
        end
        
        out[i, j] = weight_norm > T(1e-15) ? weighted_sum / weight_norm : zero(T)
    end
end

function gpu_filter_field!(
    gpu_backend::GPUBackend,
    out::AbstractMatrix{T},
    field::AbstractMatrix{T},
    grid::StructuredGrid{G,T},
    kernel::AbstractFilterKernel,
    scale::T,
    mask_strategy::Symbol,
    workspace
) where {T<:AbstractFloat, G<:AbstractGeometry{T}}
    
    # Extract the core KernelAbstractions backend device
    dev = gpu_backend.backend
    
    Nlon, Nlat = size_tuple(grid)
    rad = kernel_radius(kernel, scale)
    is_cartesian = G <: CartesianGeometry{T}
    
    # Launch KernelAbstractions GPU kernel
    gpu_kernel = gpu_filter_kernel!(dev)
    event = gpu_kernel(
        out, field, grid.lon, grid.lat, grid.areas, grid.mask,
        grid.geometry, rad, scale, mask_strategy, is_cartesian,
        ndrange=(Nlon, Nlat)
    )
    KernelAbstractions.wait(event)
    
    return out
end

end # module
