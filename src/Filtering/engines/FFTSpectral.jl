# ---------------------------------------------------------------------------
# The transform grid of an FFT spectral filter on a uniform Cartesian grid
# ---------------------------------------------------------------------------

"""
    _fft_layout(grid) -> (; dims, P, padded, kx, ky)

The transform grid of a two-direction uniform Cartesian grid: `dims` the grid's shape and `P` the
transform's, a periodic axis at its own length and a bounded one zero-padded to `2·nextprod((2,3,5), N)`,
the least even 2·3·5-smooth length of at least `2N`; and the angular wavenumbers of the real-to-complex
half spectrum, `kx` over `k₁ ≥ 0` and `ky` in FFT order.
"""
function _fft_layout(grid::FlowGeometries.Grids.StructuredGrid{T}) where {T<:AbstractFloat}
    Nx, Ny = size(FlowGeometries.Grids.mask(grid))
    px = FlowGeometries.Grids.isperiodic(grid, 1)
    py = FlowGeometries.Grids.isperiodic(grid, 2)
    # A periodic axis's spacing is its period over its length; a bounded one's is its step.
    dx = px ? T(FlowGeometries.Grids.period(grid, 1)) / Nx : abs(T(FlowGeometries.Grids.spacing(grid, 1)))
    dy = py ? T(FlowGeometries.Grids.period(grid, 2)) / Ny : abs(T(FlowGeometries.Grids.spacing(grid, 2)))
    Px = px ? Nx : 2 * nextprod((2, 3, 5), Nx)
    Py = py ? Ny : 2 * nextprod((2, 3, 5), Ny)
    kx = [T(2π) * i / (Px * dx) for i in 0:(Px ÷ 2)]
    ky = [T(2π) * (j < cld(Py, 2) ? j : j - Py) / (Py * dy) for j in 0:(Py - 1)]
    return (; dims = (Nx, Ny), P = (Px, Py), padded = (Px, Py) != (Nx, Ny), kx, ky)
end

"""
    distributed_fft_grid_plan(grid; backend::DistributedBackend, batch) -> AbstractGridPlan
    distributed_fft_filter_plan(grid_plan, kernel, scale, mask_strategy) -> AbstractFilterPlan

An FFT spectral plan whose two-direction transform is divided among the worker processes: each worker
transforms a block of columns along the first direction and a block of rows along the second. Methods in
the extension FFTW, Distributed and SharedArrays load together.
"""
function distributed_fft_grid_plan end
function distributed_fft_filter_plan end

distributed_fft_grid_plan(args...; kwargs...) = throw(ArgumentError(
    "DistributedBackend is unavailable — run `using Distributed, SharedArrays` (or use SerialBackend())."))
