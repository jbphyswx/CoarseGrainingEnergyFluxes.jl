module Pipeline

using ..Geometry
using ..Grids
using ..Kernels
using ..Filtering
using ..Derivatives
using ..Diagnostics
using StaticArrays

export CoarseGrainResult, coarse_grain

"""
    CoarseGrainResult{T, A}

Hold results of a complete coarse-graining analysis.
"""
struct CoarseGrainResult{T<:AbstractFloat, A<:AbstractArray{T}}
    scales::Vector{T}
    Π::Vector{A}
    spectrum::Vector{T}
end

"""
    coarse_grain(u, v, w, grid; scales, kernel, backend, mask_strategy)

Orchestrate the entire coarse-graining analysis by sweeping across multiple filter scales `scales`.
Pre-allocates work arrays once and runs the multiscale sweep loop with maximum efficiency.
"""
function coarse_grain(
    u::AbstractMatrix{T},
    v::AbstractMatrix{T},
    w::Union{Nothing, AbstractMatrix{T}},
    grid::StructuredGrid{G,T};
    scales::AbstractVector{T},
    kernel::AbstractFilterKernel = TopHatKernel(),
    ρ₀::T = T(1025.0),
    backend::AbstractExecutionBackend = AutoBackend(),
    mask_strategy::Symbol = :renormalize
) where {T<:AbstractFloat, G<:AbstractGeometry{T}}
    
    Nscales = length(scales)
    Nlon, Nlat = size_tuple(grid)
    
    # 1. Pre-allocate results vectors
    Π_maps = [zeros(T, Nlon, Nlat) for _ in 1:Nscales]
    
    # 2. Pre-allocate a single, reusable workspace for the entire scale sweep
    workspace = ΠWorkspace(grid)
    
    # 3. Sweep through scales and compute the cross-scale energy transfer Π maps
    for s_idx in 1:Nscales
        scale = scales[s_idx]
        compute_Π!(
            Π_maps[s_idx],
            u, v, w,
            grid,
            kernel,
            scale;
            ρ₀ = ρ₀,
            workspace = workspace,
            backend = backend,
            mask_strategy = mask_strategy
        )
    end
    
    # 4. Sweep through scales to compute the spatial filtering energy spectrum E(ℓ)
    spectrum = compute_filtering_spectrum(
        u, v, w,
        grid,
        kernel,
        scales;
        ρ₀ = ρ₀,
        backend = backend,
        mask_strategy = mask_strategy
    )
    
    return CoarseGrainResult{T, Matrix{T}}(
        Vector{T}(scales),
        Π_maps,
        spectrum
    )
end

# 2.5D Cartesian constructor wrapper (standard ocean/atmosphere model outputs without vertical velocity)
function coarse_grain(
    u::AbstractMatrix{T},
    v::AbstractMatrix{T},
    grid::StructuredGrid{G,T};
    scales::AbstractVector{T},
    kernel::AbstractFilterKernel = TopHatKernel(),
    ρ₀::T = T(1025.0),
    backend::AbstractExecutionBackend = AutoBackend(),
    mask_strategy::Symbol = :renormalize
) where {T<:AbstractFloat, G<:AbstractGeometry{T}}
    return coarse_grain(u, v, nothing, grid; scales=scales, kernel=kernel, ρ₀=ρ₀, backend=backend, mask_strategy=mask_strategy)
end

end # module
