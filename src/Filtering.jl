module Filtering

using ..Geometry: Geometry
using ..Grids: Grids
using ..Kernels: Kernels

export AbstractExecutionBackend, SerialBackend, ThreadedBackend, DistributedBackend, GPUBackend, FINUFFTBackend, AutoBackend
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
struct FINUFFTBackend     <: AbstractExecutionBackend end
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

function finufft_filter_field!(args...; kwargs...)
    throw(ArgumentError("FINUFFT backend is unavailable. Load the FINUFFT package first."))
end

# ---------------------------------------------------------------------------
# Public Filtering API
# ---------------------------------------------------------------------------

"""
    filter_field!(out, field, grid, kernel, scale; mask_strategy=:renormalize, workspace=nothing, backend=AutoBackend())

Filter a 2D field on a grid using a kernel at characteristic width scale (ℓ), writing the result to out.

# Arguments
- `out::AbstractArray{T}`: Output array for filtered field (modified in-place)
- `field::AbstractArray`: Input field to filter (2D or 3D)
- `grid::AbstractGrid`: Grid geometry and coordinates
- `kernel::AbstractFilterKernel`: Filter kernel (TopHatKernel, GaussianKernel, etc.)
- `scale::T`: Filter scale ℓ in meters

# Keyword Arguments
- `mask_strategy::Symbol=:renormalize`: Land masking strategy
  - `:zero`: Treats dry cells as zero velocity, renormalizes over wet+land
  - `:renormalize` / `:deformable`: Excludes land points entirely from numerator and denominator
- `workspace=nothing`: Pre-allocated workspace for intermediate calculations
- `backend::AbstractExecutionBackend=AutoBackend()`: Execution backend (SerialBackend, ThreadedBackend, etc.)

# Returns
- `out`: The filtered field (same array as input)

# Notes
- For spherical grids, automatically handles periodic longitude wrapping (0° ↔ 360°)
- Uses proper great-circle distance (Haversine) for spherical geometry
- Thread-safe and supports multiple execution backends via dispatch

# Examples
```julia
geom = CartesianGeometry(1000.0, 1000.0)
grid = StructuredGrid(geom, lon, lat, mask)
field = rand(100, 100)
out = zeros(100, 100)
filter_field!(out, field, grid, TopHatKernel(), 5000.0; mask_strategy=:renormalize)
```
"""
function filter_field!(
    out::AbstractArray{T},
    field::AbstractArray,
    grid::Grids.AbstractGrid,
    kernel::Kernels.AbstractFilterKernel,
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
    elseif resolved_backend <: FINUFFTBackend
        finufft_filter_field!(out, field, grid, kernel, scale; mask_strategy=mask_strategy, workspace=workspace)
    end
    
    return out
end

# 3D volume filtering method (simple layer-by-layer fallback, can be specialized)
function filter_field!(
    out::AbstractArray{T,3},
    field::AbstractArray{<:Any,3},
    grid::Grids.AbstractGrid,
    kernel::Kernels.AbstractFilterKernel,
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
    field::AbstractMatrix,
    grid::Grids.StructuredGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T,
    workspace
) where {T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}}
    Nlon, Nlat = Grids.size_tuple(grid)
    rad = Kernels.kernel_radius(kernel, scale)
    
    # Pre-calculated geometry parameters
    if G <: Geometry.CartesianGeometry{T}
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
    
    # Pre-compute kernel weights for spherical geometry
    # For each target latitude j, pre-compute kernel weights for all possible (di, dj) offsets
    kernel_weights = nothing
    if G <: Geometry.SphericalGeometry{T}
        max_dj = ceil(Int, rad / (R * dφ))
        max_di = ceil(Int, rad / (R * dλ * cos(min(abs(grid.lat[1]), abs(grid.lat[end])))))
        
        # kernel_weights[j_target][dj_idx, di_idx] = (weight, area) for offset (di, dj)
        # dj_idx ranges from 1 to 2*max_dj+1, di_idx from 1 to 2*max_di+1
        kernel_weights = Vector{Matrix{Tuple{T,T}}}(undef, Nlat)
        
        for j in 1:Nlat
            target_pt = Grids.coords(grid, 1, j)  # use first longitude as reference
            kw = Matrix{Tuple{T,T}}(undef, 2*max_dj+1, 2*max_di+1)
            
            for dj_idx in 1:(2*max_dj+1)
                dj = dj_idx - max_dj - 1
                jj = clamp(j + dj, 1, Nlat)
                for di_idx in 1:(2*max_di+1)
                    di = di_idx - max_di - 1
                    ii = clamp(1 + di, 1, Nlon)
                    
                    neigh_pt = Grids.coords(grid, ii, jj)
                    d = Geometry.distance(grid.geometry, target_pt, neigh_pt)
                    
                    if d <= rad
                        kw[dj_idx, di_idx] = (Kernels.kernel_weight(kernel, d, scale), Grids.area(grid, ii, jj))
                    else
                        kw[dj_idx, di_idx] = (zero(T), zero(T))
                    end
                end
            end
            kernel_weights[j] = kw
        end
    end
    
    # Column-major nested loop
    for j in 1:Nlat
        φ_target = grid.lat[j]
        
        # Calculate latitude bounds for Spherical geometry
        if G <: Geometry.SphericalGeometry{T}
            dj_lim = (dφ > 0) ? ceil(Int, rad / (R * dφ)) : 0
        end
        
        for i in 1:Nlon
            Grids.iswet(grid, i, j) || continue
            
            target_pt = Grids.coords(grid, i, j)
            
            # Calculate longitude bounds for Spherical geometry (depends on latitude)
            if G <: Geometry.SphericalGeometry{T}
                cosφ = cos(φ_target)
                if dλ > 0 && abs(cosφ) > T(1e-12)
                    di_lim = ceil(Int, rad / (R * cosφ * dλ))
                else
                    di_lim = 0
                end
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
                    if G <: Geometry.CartesianGeometry{T}
                        # Cartesian: compute on-the-fly (fast, uniform grid)
                        neigh_pt = Grids.coords(grid, ii, jj)
                        d = Geometry.distance(grid.geometry, target_pt, neigh_pt)
                        if d <= rad
                            w = Kernels.kernel_weight(kernel, d, scale) * Grids.area(grid, ii, jj)
                            weight_norm += w
                            if Grids.iswet(grid, ii, jj)
                                weighted_sum += w * field[ii, jj]
                            end
                        end
                    else
                        # Spherical: use pre-computed kernel weights
                        dj = jj - j
                        di = ii - i
                        max_dj = (size(kernel_weights[j], 1) - 1) ÷ 2
                        max_di = (size(kernel_weights[j], 2) - 1) ÷ 2
                        dj_idx = dj + max_dj + 1
                        di_idx = di + max_di + 1
                        
                        if 1 <= dj_idx <= 2*max_dj+1 && 1 <= di_idx <= 2*max_di+1
                            w_kern, w_area = kernel_weights[j][dj_idx, di_idx]
                            if w_kern > zero(T)
                                w = w_kern * w_area
                                weight_norm += w
                                if Grids.iswet(grid, ii, jj)
                                    weighted_sum += w * field[ii, jj]
                                end
                            end
                        end
                    end
                end
                
                # Handle periodic longitude wrapping for spherical grids
                # If near the boundary, also check points on the opposite side
                if G <: Geometry.SphericalGeometry{T}
                    # Check if we need to wrap around (near longitude boundaries)
                    wrap_left = i - di_lim < 1
                    wrap_right = i + di_lim > Nlon
                    
                    if wrap_left
                        # Check points on the right edge (wrapped around)
                        for ii_wrap in (Nlon + i - di_lim):Nlon
                            neigh_pt = Grids.coords(grid, ii_wrap, jj)
                            d = Geometry.distance(grid.geometry, target_pt, neigh_pt)
                            
                            if d <= rad
                                w = Kernels.kernel_weight(kernel, d, scale) * Grids.area(grid, ii_wrap, jj)
                                weight_norm += w
                                if Grids.iswet(grid, ii_wrap, jj)
                                    weighted_sum += w * field[ii_wrap, jj]
                                end
                            end
                        end
                    end
                    
                    if wrap_right
                        # Check points on the left edge (wrapped around)
                        for ii_wrap in 1:(i + di_lim - Nlon)
                            neigh_pt = Grids.coords(grid, ii_wrap, jj)
                            d = Geometry.distance(grid.geometry, target_pt, neigh_pt)
                            
                            if d <= rad
                                w = Kernels.kernel_weight(kernel, d, scale) * Grids.area(grid, ii_wrap, jj)
                                weight_norm += w
                                if Grids.iswet(grid, ii_wrap, jj)
                                    weighted_sum += w * field[ii_wrap, jj]
                                end
                            end
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
    field::AbstractMatrix,
    grid::Grids.StructuredGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T,
    workspace
) where {T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}}
    Nlon, Nlat = Grids.size_tuple(grid)
    rad = Kernels.kernel_radius(kernel, scale)
    
    if G <: Geometry.CartesianGeometry{T}
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
        
        if G <: Geometry.SphericalGeometry{T}
            dj_lim = (dφ > 0) ? ceil(Int, rad / (R * dφ)) : 0
        end
        
        for i in 1:Nlon
            Grids.iswet(grid, i, j) || continue
            
            target_pt = Grids.coords(grid, i, j)
            
            if G <: Geometry.SphericalGeometry{T}
                cosφ = cos(φ_target)
                if dλ > 0 && abs(cosφ) > T(1e-12)
                    di_lim = ceil(Int, rad / (R * cosφ * dλ))
                else
                    di_lim = 0
                end
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
                    Grids.iswet(grid, ii, jj) || continue
                    
                    # Fetch neighbor coords and distance (compile-time dispatch)
                    if G <: Geometry.CartesianGeometry{T}
                        neigh_pt = Grids.coords(grid, ii, jj)
                        d = Geometry.distance(grid.geometry, target_pt, neigh_pt)
                    else
                        # Use proper great-circle distance for spherical geometry
                        neigh_pt = Grids.coords(grid, ii, jj)
                        d = Geometry.distance(grid.geometry, target_pt, neigh_pt)
                    end
                    
                    if d <= rad
                        w = Kernels.kernel_weight(kernel, d, scale) * Grids.area(grid, ii, jj)
                        weight_norm += w
                        weighted_sum += w * field[ii, jj]
                    end
                end
                
                # Handle periodic longitude wrapping for spherical grids
                if G <: Geometry.SphericalGeometry{T}
                    wrap_left = i - di_lim < 1
                    wrap_right = i + di_lim > Nlon
                    
                    if wrap_left
                        for ii_wrap in (Nlon + i - di_lim):Nlon
                            Grids.iswet(grid, ii_wrap, jj) || continue
                            
                            neigh_pt = Grids.coords(grid, ii_wrap, jj)
                            d = Geometry.distance(grid.geometry, target_pt, neigh_pt)
                            
                            if d <= rad
                                w = Kernels.kernel_weight(kernel, d, scale) * Grids.area(grid, ii_wrap, jj)
                                weight_norm += w
                                weighted_sum += w * field[ii_wrap, jj]
                            end
                        end
                    end
                    
                    if wrap_right
                        for ii_wrap in 1:(i + di_lim - Nlon)
                            Grids.iswet(grid, ii_wrap, jj) || continue
                            
                            neigh_pt = Grids.coords(grid, ii_wrap, jj)
                            d = Geometry.distance(grid.geometry, target_pt, neigh_pt)
                            
                            if d <= rad
                                w = Kernels.kernel_weight(kernel, d, scale) * Grids.area(grid, ii_wrap, jj)
                                weight_norm += w
                                weighted_sum += w * field[ii_wrap, jj]
                            end
                        end
                    end
                end
            end
            
            out[i, j] = weight_norm > T(1e-15) ? weighted_sum / weight_norm : zero(T)
        end
    end
    return out
end

end # module
