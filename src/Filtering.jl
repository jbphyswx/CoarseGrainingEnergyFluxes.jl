module Filtering

using ..Geometry
using ..Grids
using ..Kernels
using StaticArrays

export AbstractExecutionBackend, SerialBackend, ThreadedBackend, DistributedBackend, GPUBackend, AutoBackend
export filter_field!, filter_field_zero!, filter_field_renorm!

# ---------------------------------------------------------------------------
# Execution Backends
# ---------------------------------------------------------------------------

abstract type AbstractExecutionBackend end

struct SerialBackend      <: AbstractExecutionBackend end
struct ThreadedBackend    <: AbstractExecutionBackend end
struct DistributedBackend <: AbstractExecutionBackend end
struct GPUBackend{B}      <: AbstractExecutionBackend
    backend::B
end
struct AutoBackend        <: AbstractExecutionBackend end

# Default fallback functions for thread/distributed backends in case their extensions are not loaded
function threaded_filter_field!(args...; kwargs...)
    throw(ArgumentError("Threaded backend is unavailable. Load the OhMyThreads package or use SerialBackend()."))
end

function distributed_filter_field!(args...; kwargs...)
    throw(ArgumentError("Distributed backend is unavailable. Load the Distributed package or use SerialBackend()."))
end

function gpu_filter_field!(args...; kwargs...)
    throw(ArgumentError("GPU backend is unavailable. Load the KernelAbstractions package or use SerialBackend()."))
end

# ---------------------------------------------------------------------------
# Public Filtering API
# ---------------------------------------------------------------------------

"""
    filter_field!(out, field, grid, kernel, scale; mask_strategy, workspace, backend)

Filter a 2D `field` on a `grid` using `kernel` at characteristic width `scale` (ℓ), writing the result to `out`.
Supported land-masking strategies:
- `:zero`: treats dry cells as zero velocity and renormalizes the kernel.
- `:renormalize` / `:deformable`: deforms/renormalizes the kernel dynamically to exclude land points entirely from both the numerator and denominator.
"""
function filter_field!(
    out::AbstractArray{T},
    field::AbstractArray{T},
    grid::AbstractGrid,
    kernel::AbstractFilterKernel,
    scale::T;
    mask_strategy::Symbol = :renormalize,
    workspace = nothing,
    backend::AbstractExecutionBackend = AutoBackend()
) where {T<:AbstractFloat}
    
    # 1. Resolve AutoBackend based on context
    resolved_backend = resolve_backend(backend)
    
    # 2. Dispatch to the appropriate execution backend
    if resolved_backend <: SerialBackend
        if mask_strategy == :zero
            filter_field_zero!(out, field, grid, kernel, scale, workspace)
        elseif mask_strategy == :renormalize || mask_strategy == :deformable
            filter_field_renorm!(out, field, grid, kernel, scale, workspace)
        else
            throw(ArgumentError("Unknown masking strategy: $mask_strategy"))
        end
    elseif resolved_backend <: ThreadedBackend
        threaded_filter_field!(out, field, grid, kernel, scale, mask_strategy, workspace)
    elseif resolved_backend <: DistributedBackend
        distributed_filter_field!(out, field, grid, kernel, scale, mask_strategy, workspace)
    elseif resolved_backend <: GPUBackend
        gpu_filter_field!(resolved_backend, out, field, grid, kernel, scale, mask_strategy, workspace)
    end
    
    return out
end

# 3D volume filtering method (simple layer-by-layer fallback, can be specialized)
function filter_field!(
    out::AbstractArray{T,3},
    field::AbstractArray{T,3},
    grid::AbstractGrid,
    kernel::AbstractFilterKernel,
    scale::T;
    mask_strategy::Symbol = :renormalize,
    workspace = nothing,
    backend::AbstractExecutionBackend = AutoBackend()
) where {T<:AbstractFloat}
    Ndepth = size(field, 3)
    for k in 1:Ndepth
        out_layer = view(out, :, :, k)
        field_layer = view(field, :, :, k)
        filter_field!(out_layer, field_layer, grid, kernel, scale; mask_strategy=mask_strategy, workspace=workspace, backend=backend)
    end
    return out
end

# Helper to automatically resolve backend
function resolve_backend(backend::AbstractExecutionBackend)
    if backend isa AutoBackend
        # Check threads count
        if Threads.nthreads() > 1
            return ThreadedBackend
        else
            return SerialBackend
        end
    else
        return typeof(backend)
    end
end

# ---------------------------------------------------------------------------
# Serial Physical-Space Convolution Algorithms (StructuredGrid)
# ---------------------------------------------------------------------------

"""
    filter_field_zero!(out, field, grid, kernel, scale, workspace)

Serial implementation of the `:zero` masking strategy (land cells have zero values).
"""
function filter_field_zero!(
    out::AbstractMatrix{T},
    field::AbstractMatrix{T},
    grid::StructuredGrid{G,T},
    kernel::AbstractFilterKernel,
    scale::T,
    workspace
) where {T<:AbstractFloat, G<:AbstractGeometry{T}}
    Nlon, Nlat = size_tuple(grid)
    rad = kernel_radius(kernel, scale)
    
    # Pre-calculated geometry parameters
    if G <: CartesianGeometry{T}
        dx = grid.geometry.dx
        dy = grid.geometry.dy
        # Index bounds for Cartesian coordinates are constant
        di_lim = ceil(Int, rad / dx)
        dj_lim = ceil(Int, rad / dy)
    else
        R = grid.geometry.R
        dλ = Nlon > 1 ? grid.lon[2] - grid.lon[1] : T(0)
        dφ = Nlat > 1 ? grid.lat[2] - grid.lat[1] : T(0)
    end
    
    fill!(out, zero(T))
    
    # Column-major nested loop
    for j in 1:Nlat
        φ_target = grid.lat[j]
        
        # Calculate latitude bounds for Spherical geometry
        if G <: SphericalGeometry{T}
            dj_lim = ceil(Int, rad / (R * dφ))
        end
        
        for i in 1:Nlon
            iswet(grid, i, j) || continue
            
            target_pt = coords(grid, i, j)
            
            # Calculate longitude bounds for Spherical geometry (depends on latitude)
            if G <: SphericalGeometry{T}
                cosφ = cos(φ_target)
                di_lim = abs(cosφ) > T(1e-12) ? ceil(Int, rad / (R * cosφ * dλ)) : 0
            end
            
            weighted_sum = zero(T)
            weight_norm  = zero(T)
            
            # Sub-loop over integration footprint index ranges
            j_start = max(1, j - dj_lim)
            j_end   = min(Nlat, j + dj_lim)
            
            for jj in j_start:j_end
                i_start = max(1, i - di_lim)
                i_end   = min(Nlon, i + di_lim)
                
                for ii in i_start:i_end
                    # Fetch neighbor coords and distance
                    neigh_pt = coords(grid, ii, jj)
                    d = distance(grid.geometry, target_pt, neigh_pt)
                    
                    if d <= rad
                        w = kernel_weight(kernel, d, scale) * area(grid, ii, jj)
                        weight_norm += w
                        
                        # Only integrate wet cells (dry cells treated as zero value)
                        if iswet(grid, ii, jj)
                            weighted_sum += w * field[ii, jj]
                        end
                    end
                end
            end
            
            out[i, j] = weight_norm > T(1e-15) ? weighted_sum / weight_norm : zero(T)
        end
    end
    return out
end

"""
    filter_field_renorm!(out, field, grid, kernel, scale, workspace)

Serial implementation of the `:renormalize` masking strategy (land points completely excluded).
"""
function filter_field_renorm!(
    out::AbstractMatrix{T},
    field::AbstractMatrix{T},
    grid::StructuredGrid{G,T},
    kernel::AbstractFilterKernel,
    scale::T,
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
    
    fill!(out, zero(T))
    
    for j in 1:Nlat
        φ_target = grid.lat[j]
        
        if G <: SphericalGeometry{T}
            dj_lim = ceil(Int, rad / (R * dφ))
        end
        
        for i in 1:Nlon
            iswet(grid, i, j) || continue
            
            target_pt = coords(grid, i, j)
            
            if G <: SphericalGeometry{T}
                cosφ = cos(φ_target)
                di_lim = abs(cosφ) > T(1e-12) ? ceil(Int, rad / (R * cosφ * dλ)) : 0
            end
            
            weighted_sum = zero(T)
            weight_norm  = zero(T)
            
            j_start = max(1, j - dj_lim)
            j_end   = min(Nlat, j + dj_lim)
            
            for jj in j_start:j_end
                i_start = max(1, i - di_lim)
                i_end   = min(Nlon, i + di_lim)
                
                for ii in i_start:i_end
                    # In `:renormalize` strategy, land points are completely ignored
                    iswet(grid, ii, jj) || continue
                    
                    neigh_pt = coords(grid, ii, jj)
                    d = distance(grid.geometry, target_pt, neigh_pt)
                    
                    if d <= rad
                        w = kernel_weight(kernel, d, scale) * area(grid, ii, jj)
                        weight_norm += w
                        weighted_sum += w * field[ii, jj]
                    end
                end
            end
            
            out[i, j] = weight_norm > T(1e-15) ? weighted_sum / weight_norm : zero(T)
        end
    end
    return out
end

end # module
