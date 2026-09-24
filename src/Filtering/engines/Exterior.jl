# ---------------------------------------------------------------------------
# The kernel's mass beyond a bounded edge
# ---------------------------------------------------------------------------
#
# Coarse graining filters the field extended by zero beyond the domain, with the kernel normalized over
# all space (Aluie et al. 2018; Grooms et al. 2021, eq. 7). `ZeroFill` therefore divides by the mass of
# the in-domain taps an engine sums plus the mass of the cells beyond each bounded edge. Those cells are
# the grid's lattice continued at its edge spacing: a spherical latitude as far as the poles, a regional
# longitude as far as one turn, a radius as far as the origin.

"""
    _exterior_mass(grid, kernel, scale) -> Array or nothing

`Σ kernel_weight(d) · measure` over the exterior cells within the kernel's radius of each cell of
`grid`, or `nothing` when no cell reaches past a bounded edge.
"""
function _exterior_mass(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, kernel::Kernels.AbstractFilterKernel, scale::T,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}, N}
    dim = _kernel_dim(grid)
    rad = Kernels.kernel_radius(kernel, scale, dim)
    lattice = _exterior_lattice(grid, rad)
    lattice === nothing && return nothing
    ext, lo, extended = lattice
    return _exterior_mass(ext, lo, extended, FlowGeometries.Grids.size_tuple(grid), kernel, scale, rad, dim)
end

# Each direction continues as a `Range` or as a `Vector` depending on how far it reaches, so `ext`'s type
# is known only here, past the call.
function _exterior_mass(
    ext::FlowGeometries.Grids.StructuredGrid{T,G,N}, lo::NTuple{N,Int}, extended::NTuple{N,Bool},
    dims::NTuple{N,Int}, kernel::Kernels.AbstractFilterKernel, scale::T, rad::T, dim::Val,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}, N}
    mt = _query_topology(ext, rad)
    images = _image_convention(ext)
    E = zeros(T, dims)
    reached = false
    for I in CartesianIndices(dims)
        Ie = ntuple(d -> I[d] + lo[d], Val(N))
        w = FlowGeometries.Connectivity.metric_window(ext, Ie, rad, mt)
        _window_inside(Ie, w, lo, dims, extended) && continue
        m = FlowGeometries.Connectivity.fold_within(
            zero(T), ext, Ie...;
            ball = rad, self = true, active_only = false, topology = mt, images = images,
        ) do acc, J, d
            _inside(J, lo, dims) ? acc :
                acc + Kernels.kernel_weight(kernel, T(d), scale, dim) * FlowGeometries.Grids.area(ext, J...)
        end
        @inbounds E[I] = m
        reached |= !iszero(m)
    end
    return reached ? E : nothing
end

@inline _inside(J::NTuple{N,Int}, lo::NTuple{N,Int}, dims::NTuple{N,Int}) where {N} =
    all(ntuple(d -> lo[d] < J[d] <= lo[d] + dims[d], Val(N)))

# Whether the index window around `I` stays among the grid's own cells along every extended direction.
@inline _window_inside(I::NTuple{N,Int}, w, lo::NTuple{N,Int}, dims::NTuple{N,Int}, extended::NTuple{N,Bool}) where {N} =
    all(ntuple(d -> !extended[d] || (I[d] - w[d] > lo[d] && I[d] + w[d] <= lo[d] + dims[d]), Val(N)))

"""
    _reaches_exterior(grid, kernel, scale) -> Bool

Whether the lattice continues past a bounded edge of `grid` within the kernel's reach, in which case
`ZeroFill` and `Deformable` divide by different masses even on an unmasked grid. `true` whenever a
window can leave the grid, so a strategy check built on it never accepts a mismatch.
"""
function _reaches_exterior(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, kernel::Kernels.AbstractFilterKernel, scale::T,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}, N}
    rad = Kernels.kernel_radius(kernel, scale, _kernel_dim(grid))
    geo = FlowGeometries.Grids.grid_geometry(grid)
    return any(ntuple(d -> _axis_counts(geo, grid, d, rad) != (0, 0), Val(N)))
end

"""
    _exterior_lattice(grid, rad) -> (ext, lo, extended) or nothing

`grid`'s lattice continued past each bounded edge far enough to hold every exterior cell within `rad`
of a cell of `grid`, as a grid of the same geometry and closure; `lo[d]` exterior cells precede the
grid's own along direction `d`, and `extended[d]` says whether direction `d` gained any.
"""
function _exterior_lattice(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,N}, rad::T,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}, N}
    geo = FlowGeometries.Grids.grid_geometry(grid)
    parts = ntuple(d -> _axis_extension(geo, grid, d, rad), Val(N))
    extended = map(p -> p[1] > 0 || p[2] > 0, parts)
    any(extended) || return nothing
    period = ntuple(
        d -> FlowGeometries.Grids.isperiodic(grid, d) ? FlowGeometries.Grids.period(grid, d) : nothing, Val(N),
    )
    ext = FlowGeometries.Grids.StructuredGrid(
        geo, map(p -> p[3], parts)...; topology = FlowGeometries.Grids.topology(grid), period = period,
    )
    return ext, map(p -> p[1], parts), extended
end

# `(n_lo, n_hi)`: how many cells direction `d` continues by below its first and above its last.
function _axis_counts(
    geo::FlowGeometries.Geometry.AbstractGeometry{T}, grid, d::Int, rad::T,
) where {T<:AbstractFloat}
    x = FlowGeometries.Grids.coordinates(grid, d)
    n = length(x)
    (FlowGeometries.Grids.isperiodic(grid, d) || n < 2) && return (0, 0)
    return _extension_counts(geo, d, x, T(x[2] - x[1]), T(x[n] - x[n - 1]), rad)
end

# `(n_lo, n_hi, axis)`: direction `d` continued by `n_lo` cells below its first and `n_hi` above its
# last, each at the spacing of the gap it continues.
function _axis_extension(
    geo::FlowGeometries.Geometry.AbstractGeometry{T}, grid, d::Int, rad::T,
) where {T<:AbstractFloat}
    x = FlowGeometries.Grids.coordinates(grid, d)
    n = length(x)
    n_lo, n_hi = _axis_counts(geo, grid, d, rad)
    (n_lo == 0 && n_hi == 0) && return (0, 0, x)
    s_lo = T(x[2] - x[1])
    s_hi = T(x[n] - x[n - 1])
    axis = Vector{T}(undef, n + n_lo + n_hi)
    @inbounds for k in 1:n_lo
        axis[n_lo + 1 - k] = x[1] - k * s_lo
    end
    @inbounds for i in 1:n
        axis[n_lo + i] = x[i]
    end
    @inbounds for k in 1:n_hi
        axis[n_lo + n + k] = x[n] + k * s_hi
    end
    return (n_lo, n_hi, axis)
end

# A Cartesian direction reaches as far as the kernel does.
_extension_counts(
    ::FlowGeometries.Geometry.AbstractCartesianGeometry{T}, ::Int, ::AbstractVector, s_lo::T, s_hi::T, rad::T,
) where {T<:AbstractFloat} = (ceil(Int, rad / abs(s_lo)), ceil(Int, rad / abs(s_hi)))

# Longitude fills the rest of the ring, half past each end; latitude continues to the pole it heads
# for; the radius continues toward the origin while positive.
function _extension_counts(
    geo::FlowGeometries.Geometry.AbstractSphericalGeometry{T}, d::Int, x, s_lo::T, s_hi::T, rad::T,
) where {T<:AbstractFloat}
    n = length(x)
    tol = sqrt(eps(T))
    if d == 1
        s = (abs(s_lo) + abs(s_hi)) / 2
        span = abs(T(x[n] - x[1])) + s
        m = max(0, floor(Int, (T(2π) - span) / s + tol))
        return (m ÷ 2, m - m ÷ 2)
    elseif d == 2
        R = FlowGeometries.Geometry.radius(geo)
        reach(s) = ceil(Int, rad / (R * abs(s)))
        pole_lo = s_lo > 0 ? -T(π) / 2 : T(π) / 2
        pole_hi = s_hi > 0 ? T(π) / 2 : -T(π) / 2
        to_lo = floor(Int, abs(pole_lo - T(x[1])) / abs(s_lo) + tol)
        to_hi = floor(Int, abs(pole_hi - T(x[n])) / abs(s_hi) + tol)
        return (min(reach(s_lo), to_lo), min(reach(s_hi), to_hi))
    else
        # Positive radii only: the cell `k` steps out is at `x[1] - k·s_lo` or `x[n] + k·s_hi`.
        k_lo = s_lo > 0 ? max(0, ceil(Int, T(x[1]) / s_lo) - 1) : typemax(Int)
        k_hi = s_hi < 0 ? max(0, ceil(Int, T(x[n]) / -s_hi) - 1) : typemax(Int)
        return (min(ceil(Int, rad / abs(s_lo)), k_lo), min(ceil(Int, rad / abs(s_hi)), k_hi))
    end
end
