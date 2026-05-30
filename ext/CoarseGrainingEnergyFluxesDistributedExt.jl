module CoarseGrainingEnergyFluxesDistributedExt

using CoarseGrainingEnergyFluxes
using Distributed
using SharedArrays
using StaticArrays

import CoarseGrainingEnergyFluxes.Filtering: distributed_filter_field!

function distributed_filter_field!(
    out::AbstractMatrix{T},
    field::AbstractMatrix{T},
    grid::StructuredGrid{G,T},
    kernel::AbstractFilterKernel,
    scale::T,
    mask_strategy::Symbol,
    workspace
) where {T<:AbstractFloat, G<:AbstractGeometry{T}}
    
    Nlon, Nlat = size_tuple(grid)
    rad = kernel_radius(kernel, scale)
    
    if G <: CartesianGeometry{T}
        dx = grid.geometry.dx
        dy = grid.geometry.dy
        di_lim = ceil(Int, rad / dx)
        dj_lim = ceil(Int, rad / dy)
    else
        R = grid.geometry.R
        dλ = Nlon > 1 ? grid.lon[2] - grid.lon[1] : T(0)
        dφ = Nlat > 1 ? grid.lat[2] - grid.lat[1] : T(0)
    end
    
    # We use a SharedArray to share the output buffer across local distributed workers without copies
    s_out = SharedArray{T}(Nlon, Nlat)
    fill!(s_out, zero(T))
    
    # Parallel map/loop over latitude rows using workers
    @sync @distributed for j in 1:Nlat
        φ_target = G <: SphericalGeometry{T} ? grid.lat[j] : zero(T)
        dj_lim_loc = G <: SphericalGeometry{T} ? ceil(Int, rad / (R * dφ)) : dj_lim
        
        for i in 1:Nlon
            iswet(grid, i, j) || continue
            
            target_pt = coords(grid, i, j)
            
            if G <: SphericalGeometry{T}
                cosφ = cos(φ_target)
                di_lim_loc = abs(cosφ) > T(1e-12) ? ceil(Int, rad / (R * cosφ * dλ)) : 0
            else
                di_lim_loc = di_lim
            end
            
            weighted_sum = zero(T)
            weight_norm  = zero(T)
            
            j_start = max(1, j - dj_lim_loc)
            j_end   = min(Nlat, j + dj_lim_loc)
            
            for jj in j_start:j_end
                i_start = max(1, i - di_lim_loc)
                i_end   = min(Nlon, i + di_lim_loc)
                
                for ii in i_start:i_end
                    if mask_strategy == :renormalize || mask_strategy == :deformable
                        iswet(grid, ii, jj) || continue
                    end
                    
                    neigh_pt = coords(grid, ii, jj)
                    d = distance(grid.geometry, target_pt, neigh_pt)
                    
                    if d <= rad
                        w = kernel_weight(kernel, d, scale) * area(grid, ii, jj)
                        weight_norm += w
                        
                        if mask_strategy == :zero
                            if iswet(grid, ii, jj)
                                weighted_sum += w * field[ii, jj]
                            end
                        else
                            weighted_sum += w * field[ii, jj]
                        end
                    end
                end
            end
            
            s_out[i, j] = weight_norm > T(1e-15) ? weighted_sum / weight_norm : zero(T)
        end
    end
    
    # Copy shared array back to original output array
    copyto!(out, s_out)
    return out
end

end # module
