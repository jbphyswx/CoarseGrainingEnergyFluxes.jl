module CoarseGrainingEnergyFluxesOhMyThreadsExt

using OhMyThreads: OhMyThreads
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF

# ThreadedBackend: build the footprint once, then fill output rows in parallel. Rows are
# independent (each writes a disjoint column of the column-major output), so this is data-race-free
# and produces results IDENTICAL to the serial backend (same shared footprint + per-row kernel),
# including periodic wrapping and masking. Dynamic scheduling balances the uneven per-row cost from
# masking. `workspace`, when supplied by a cached `PhysicalFilterPlan`, IS the already-built
# footprint — reused as-is instead of rebuilding it on every call.
# Grid-generic: `apply_footprint_row!`/`size_tuple`/`isperiodic` already work for any row-decomposable
# 2D grid (StructuredGrid OR CurvilinearGrid — both use the same shared per-row engine), so this
# function isn't hardcoded to StructuredGrid; nothing about the threading itself is StructuredGrid-specific.
function CGEF.Filtering.threaded_filter_field!(
    out::AbstractMatrix{T},
    field::AbstractMatrix,
    grid::Union{CGEF.StructuredGrid{G,T,2}, CGEF.CurvilinearGrid{T,G}},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
    workspace,
) where {T<:AbstractFloat, G<:CGEF.Geometry.AbstractGeometry{T}}
    fp = workspace === nothing ? CGEF.Filtering.build_footprint(grid, kernel, scale) : workspace
    periodic = CGEF.Grids.isperiodic(grid, 1)
    _, Nlat = CGEF.Grids.size_tuple(grid)
    fill!(out, zero(T))
    OhMyThreads.tforeach(1:Nlat; scheduler = :dynamic) do j
        CGEF.Filtering.apply_footprint_row!(out, field, grid, fp, mask_strategy, periodic, j)
    end
    return out
end

# ThreadedBackend for 1D/true-3D StructuredGrid: these use the point-indexed `FilterFootprintND`/
# `FilterFootprintNDScattered` representation (no row structure), but every output point is still
# fully independent (reads neighbours, writes only its own cell) — so parallelizing over
# `CartesianIndices(out)` instead of rows is equally data-race-free and identical to the serial
# result. Reuses the exact same per-point kernel (`Filtering._footprint_nd_point`) the serial
# `apply_footprint_nd!` calls, just distributed across threads instead of a plain `for` loop. More
# specific than the 2D method above only in that `N` here is unconstrained; Julia dispatch picks the
# 2D method for N=2 and this one for N=1/3 (the only other cases `_nd_parallelizable` allows).
function CGEF.Filtering.threaded_filter_field!(
    out::AbstractArray{T,N},
    field::AbstractArray,
    grid::CGEF.StructuredGrid{G,T,N},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
    workspace,
) where {T<:AbstractFloat, G<:CGEF.Geometry.AbstractGeometry{T}, N}
    fp = workspace === nothing ? CGEF.Filtering.build_footprint(grid, kernel, scale) : workspace
    mask = grid.mask
    fill!(out, zero(T))
    if fp isa CGEF.Filtering.FilterFootprintND
        dims = CGEF.Grids.size_tuple(grid)
        periodic = grid.periodic
        OhMyThreads.tforeach(CartesianIndices(out); scheduler = :dynamic) do I
            mask[I] || return
            out[I] = CGEF.Filtering._footprint_nd_point(field, fp, mask_strategy, dims, periodic, mask, I)
        end
    else
        lin = LinearIndices(CGEF.Grids.size_tuple(grid))
        OhMyThreads.tforeach(CartesianIndices(out); scheduler = :dynamic) do I
            mask[I] || return
            out[I] = CGEF.Filtering._footprint_nd_point(field, fp, mask_strategy, mask, lin, I)
        end
    end
    return out
end

end # module
