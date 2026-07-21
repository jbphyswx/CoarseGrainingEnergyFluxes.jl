module Derivatives

using ..Geometry: Geometry
using ..Grids: Grids

export AbstractStencilOrder, SecondOrderStencil
export ddx!, ddy!, ddz!
export WLSQGradientPlan

"""
    AbstractStencilOrder

Abstract supertype for spatial difference stencil orders.
"""
abstract type AbstractStencilOrder end

"""
    SecondOrderStencil <: AbstractStencilOrder

Standard 2nd-order centered difference stencil. Near boundaries and masked cells,
it falls back dynamically to 1st-order one-sided differences to avoid contamination.
"""
struct SecondOrderStencil <: AbstractStencilOrder end

# ---------------------------------------------------------------------------
# 1D Cartesian X-derivative — a genuinely 1D `StructuredGrid` (N=1), not a 2D grid with a singleton
# dimension (that case reuses the existing 2D methods directly, no new dispatch needed). Same
# nonuniform-aware stencil (`Grids._local_spacing`/`Geometry.nonuniform_first_derivative`) as the 2D
# Cartesian `ddx!`, just single-indexed.
# ---------------------------------------------------------------------------
function ddx!(
    ∂f∂x::AbstractVector{T},
    f::AbstractVector{T},
    grid::Grids.StructuredGrid{Geometry.CartesianGeometry{T},T,1},
) where {T<:AbstractFloat}
    (Nlon,) = Grids.size_tuple(grid)
    lon = grid.axes[1]

    for i in 1:Nlon
        if !Grids.isactive(grid, i)
            ∂f∂x[i] = zero(T)
            continue
        end

        has_p = i < Nlon && Grids.isactive(grid, i+1)
        has_m = i > 1    && Grids.isactive(grid, i-1)
        h_m, h_p = Grids._local_spacing(lon, i)

        if has_p && has_m
            ∂f∂x[i] = Geometry.nonuniform_first_derivative(f[i-1], f[i], f[i+1], h_m, h_p)
        elseif has_p
            ∂f∂x[i] = (f[i+1] - f[i]) / h_p
        elseif has_m
            ∂f∂x[i] = (f[i] - f[i-1]) / h_m
        else
            ∂f∂x[i] = zero(T)
        end
    end
    return ∂f∂x
end

# ---------------------------------------------------------------------------
# X-derivative (ddx!)
# ---------------------------------------------------------------------------

"""
    ddx!(∂f∂x, f, grid)

Calculate spatial derivative of `f` in the horizontal x (Eastward/λ) direction, writing to `∂f∂x`.
"""
function ddx!(
    ∂f∂x::AbstractMatrix{T},
    f::AbstractMatrix{T},
    grid::Grids.StructuredGrid{Geometry.CartesianGeometry{T},T}
) where {T<:AbstractFloat}
    Nlon, Nlat = Grids.size_tuple(grid)
    lon = grid.lon

    for j in 1:Nlat
        for i in 1:Nlon
            if !Grids.isactive(grid, i, j)
                ∂f∂x[i, j] = zero(T)
                continue
            end

            # Check neighbors
            has_p = i < Nlon && Grids.isactive(grid, i+1, j)
            has_m = i > 1    && Grids.isactive(grid, i-1, j)
            # Real per-point gaps (zero-allocation: two array reads and a subtraction), not a
            # single global Δ read once from the first two samples — correct for nonuniform axes,
            # bit-for-bit identical to the old uniform-Δ formula when the axis IS uniform.
            h_m, h_p = Grids._local_spacing(lon, i)

            if has_p && has_m
                # Nonuniform-aware 2nd-order centered difference
                ∂f∂x[i, j] = Geometry.nonuniform_first_derivative(f[i-1, j], f[i, j], f[i+1, j], h_m, h_p)
            elseif has_p
                # Forward difference (at boundary or near a masked cell)
                ∂f∂x[i, j] = (f[i+1, j] - f[i, j]) / h_p
            elseif has_m
                # Backward difference (at boundary or near a masked cell)
                ∂f∂x[i, j] = (f[i, j] - f[i-1, j]) / h_m
            else
                # Completely isolated point
                ∂f∂x[i, j] = zero(T)
            end
        end
    end
    return ∂f∂x
end

function ddx!(
    ∂f∂x::AbstractMatrix{T},
    f::AbstractMatrix{T},
    grid::Grids.StructuredGrid{Geometry.SphericalGeometry{T},T}
) where {T<:AbstractFloat}
    Nlon, Nlat = Grids.size_tuple(grid)
    R = grid.geometry.R
    lon = grid.lon
    # Only wrap longitude neighbours/spacing across the seam when the grid actually IS periodic in
    # longitude (`grid.periodic`/auto-detected at construction) — the old code wrapped
    # unconditionally, which silently corrupted the boundary derivative on a REGIONAL (non-360°)
    # domain by treating its two unrelated edges as adjacent.
    periodic_lon = Grids.isperiodic(grid, 1)
    period = periodic_lon ? T(2π) : nothing

    for j in 1:Nlat
        φ = grid.lat[j]
        cosφ = cos(φ)
        pole = abs(cosφ) <= T(1e-12) # avoid the 1/cosφ blowup at the poles

        for i in 1:Nlon
            if !Grids.isactive(grid, i, j) || pole
                ∂f∂x[i, j] = zero(T)
                continue
            end

            if i < Nlon
                i_p = i + 1; has_p = Grids.isactive(grid, i_p, j)
            elseif periodic_lon
                i_p = 1; has_p = Grids.isactive(grid, i_p, j)
            else
                i_p = i; has_p = false
            end

            if i > 1
                i_m = i - 1; has_m = Grids.isactive(grid, i_m, j)
            elseif periodic_lon
                i_m = Nlon; has_m = Grids.isactive(grid, i_m, j)
            else
                i_m = i; has_m = false
            end

            # Real per-point angular gaps (wrapped at the seam iff periodic), converted to physical
            # arc-length spacing at this latitude — zero-allocation.
            h_m, h_p = Grids._local_spacing(lon, i, period)
            h_m_phys = R * cosφ * h_m
            h_p_phys = R * cosφ * h_p

            if has_p && has_m
                ∂f∂x[i, j] = Geometry.nonuniform_first_derivative(f[i_m, j], f[i, j], f[i_p, j], h_m_phys, h_p_phys)
            elseif has_p
                ∂f∂x[i, j] = (f[i_p, j] - f[i, j]) / h_p_phys
            elseif has_m
                ∂f∂x[i, j] = (f[i, j] - f[i_m, j]) / h_m_phys
            else
                ∂f∂x[i, j] = zero(T)
            end
        end
    end
    return ∂f∂x
end

# ---------------------------------------------------------------------------
# Y-derivative (ddy!)
# ---------------------------------------------------------------------------

"""
    ddy!(∂f∂y, f, grid)

Calculate spatial derivative of `f` in the vertical y (Northward/φ) direction, writing to `∂f∂y`.
"""
function ddy!(
    ∂f∂y::AbstractMatrix{T},
    f::AbstractMatrix{T},
    grid::Grids.StructuredGrid{Geometry.CartesianGeometry{T},T}
) where {T<:AbstractFloat}
    Nlon, Nlat = Grids.size_tuple(grid)
    lat = grid.lat

    for j in 1:Nlat
        for i in 1:Nlon
            if !Grids.isactive(grid, i, j)
                ∂f∂y[i, j] = zero(T)
                continue
            end

            # Check neighbors
            has_p = j < Nlat && Grids.isactive(grid, i, j+1)
            has_m = j > 1    && Grids.isactive(grid, i, j-1)
            h_m, h_p = Grids._local_spacing(lat, j)

            if has_p && has_m
                # Nonuniform-aware 2nd-order centered difference
                ∂f∂y[i, j] = Geometry.nonuniform_first_derivative(f[i, j-1], f[i, j], f[i, j+1], h_m, h_p)
            elseif has_p
                # Forward difference (at boundary or near a masked cell)
                ∂f∂y[i, j] = (f[i, j+1] - f[i, j]) / h_p
            elseif has_m
                # Backward difference (at boundary or near a masked cell)
                ∂f∂y[i, j] = (f[i, j] - f[i, j-1]) / h_m
            else
                # Completely isolated point
                ∂f∂y[i, j] = zero(T)
            end
        end
    end
    return ∂f∂y
end

function ddy!(
    ∂f∂y::AbstractMatrix{T},
    f::AbstractMatrix{T},
    grid::Grids.StructuredGrid{Geometry.SphericalGeometry{T},T}
) where {T<:AbstractFloat}
    Nlon, Nlat = Grids.size_tuple(grid)
    R = grid.geometry.R
    lat = grid.lat
    # Latitude never wraps (it's bounded, not periodic), so no periodicity handling is needed here
    # — unlike `ddx!`'s longitude direction.

    for j in 1:Nlat
        for i in 1:Nlon
            if !Grids.isactive(grid, i, j)
                ∂f∂y[i, j] = zero(T)
                continue
            end

            has_p = j < Nlat && Grids.isactive(grid, i, j+1)
            has_m = j > 1    && Grids.isactive(grid, i, j-1)
            h_m, h_p = Grids._local_spacing(lat, j)
            h_m_phys = R * h_m
            h_p_phys = R * h_p

            if has_p && has_m
                ∂f∂y[i, j] = Geometry.nonuniform_first_derivative(f[i, j-1], f[i, j], f[i, j+1], h_m_phys, h_p_phys)
            elseif has_p
                ∂f∂y[i, j] = (f[i, j+1] - f[i, j]) / h_p_phys
            elseif has_m
                ∂f∂y[i, j] = (f[i, j] - f[i, j-1]) / h_m_phys
            else
                ∂f∂y[i, j] = zero(T)
            end
        end
    end
    return ∂f∂y
end

# ---------------------------------------------------------------------------
# Z-derivative (ddz!) - Supports 3D structures
# ---------------------------------------------------------------------------

"""
    ddz!(∂f∂z, f, grid)

Calculate spatial derivative of `f` in the vertical coordinate z, writing to `∂f∂z`.
"""
function ddz!(
    ∂f∂z::AbstractArray{T,3},
    f::AbstractArray{T,3},
    grid::Grids.StructuredGrid{Geometry.CartesianGeometry{T},T}
) where {T<:AbstractFloat}
    Nlon, Nlat, Ndepth = size(f)
    dz = grid.geometry.dz
    
    # Boundary/mask-avoiding finite differences in vertical
    for k in 1:Ndepth
        for j in 1:Nlat
            for i in 1:Nlon
                if !Grids.isactive(grid, i, j)
                    ∂f∂z[i, j, k] = zero(T)
                    continue
                end
                
                has_p = k < Ndepth
                has_m = k > 1
                
                if has_p && has_m
                    ∂f∂z[i, j, k] = (f[i, j, k+1] - f[i, j, k-1]) / (T(2) * dz)
                elseif has_p
                    ∂f∂z[i, j, k] = (f[i, j, k+1] - f[i, j, k]) / dz
                elseif has_m
                    ∂f∂z[i, j, k] = (f[i, j, k] - f[i, j, k-1]) / dz
                else
                    ∂f∂z[i, j, k] = zero(T)
                end
            end
        end
    end
    return ∂f∂z
end

# ---------------------------------------------------------------------------
# True 3D Cartesian derivatives (3D grid + 3D mask)
#
# These dispatch on `StructuredGrid{Cartesian,T,3}` (N = 3), which is strictly more specific than the
# N-free 2.5D methods above, so a genuine 3D grid routes here while a 2D grid carrying a 3D field
# (layer-by-layer) keeps using the 2.5D methods. Each direction uses a 2nd-order centered difference
# that falls back to a one-sided difference at the domain edge or against a masked cell, mirroring
# the 2D engine. Masked cells are written as exactly zero.
# ---------------------------------------------------------------------------

for (fn, dim) in ((:ddx!, 1), (:ddy!, 2), (:ddz!, 3))
    @eval function $fn(
        ∂f::AbstractArray{T,3},
        f::AbstractArray{T,3},
        grid::Grids.StructuredGrid{Geometry.CartesianGeometry{T},T,3},
    ) where {T<:AbstractFloat}
        Nx, Ny, Nz = Grids.size_tuple(grid)
        mask = grid.mask
        d = $dim
        ax = grid.axes[d] # real per-axis coordinate vector (x, y, or z) — not a scalar dx/dy/dz
        @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
            if !mask[i, j, k]
                ∂f[i, j, k] = zero(T)
                continue
            end
            ip = d == 1 ? i + 1 : i; jp = d == 2 ? j + 1 : j; kp = d == 3 ? k + 1 : k
            im = d == 1 ? i - 1 : i; jm = d == 2 ? j - 1 : j; km = d == 3 ? k - 1 : k
            lim = d == 1 ? Nx : d == 2 ? Ny : Nz
            idx = d == 1 ? i : d == 2 ? j : k
            has_p = idx < lim && mask[ip, jp, kp]
            has_m = idx > 1 && mask[im, jm, km]
            # Real per-point gaps along this axis, zero-allocation — correct for nonuniform axes,
            # bit-for-bit identical to the old uniform-Δ formula when the axis IS uniform.
            h_m, h_p = Grids._local_spacing(ax, idx)
            if has_p && has_m
                ∂f[i, j, k] = Geometry.nonuniform_first_derivative(f[im, jm, km], f[i, j, k], f[ip, jp, kp], h_m, h_p)
            elseif has_p
                ∂f[i, j, k] = (f[ip, jp, kp] - f[i, j, k]) / h_p
            elseif has_m
                ∂f[i, j, k] = (f[i, j, k] - f[im, jm, km]) / h_m
            else
                ∂f[i, j, k] = zero(T)
            end
        end
        return ∂f
    end
end

# ---------------------------------------------------------------------------
# True 3D spherical derivatives (3D grid + 3D mask): the third axis is the RADIUS r[k] (absolute
# distance from the planet center, not a depth/height offset). Unlike the Cartesian case above, the
# three axes are NOT metrically symmetric — lon/lat need the LOCAL r[k] in their arc-length metric
# factors (1/(r cosφ), 1/r), scaled per level rather than the fixed reference `geo.R` the 2D/2.5D
# methods use — so this is three separate methods, not a metaprogrammed loop. The radial derivative
# needs no metric factor at all: r is already physical distance.
# ---------------------------------------------------------------------------

function ddx!(
    ∂f∂x::AbstractArray{T,3},
    f::AbstractArray{T,3},
    grid::Grids.StructuredGrid{Geometry.SphericalGeometry{T},T,3},
) where {T<:AbstractFloat}
    Nlon, Nlat, Nr = Grids.size_tuple(grid)
    lon, lat, r = grid.axes
    periodic_lon = Grids.isperiodic(grid, 1)
    period = periodic_lon ? T(2π) : nothing

    @inbounds for k in 1:Nr
        rk = r[k]
        for j in 1:Nlat
            φ = lat[j]
            cosφ = cos(φ)
            pole = abs(cosφ) <= T(1e-12)
            for i in 1:Nlon
                if !Grids.isactive(grid, i, j, k) || pole
                    ∂f∂x[i, j, k] = zero(T)
                    continue
                end

                if i < Nlon
                    i_p = i + 1; has_p = Grids.isactive(grid, i_p, j, k)
                elseif periodic_lon
                    i_p = 1; has_p = Grids.isactive(grid, i_p, j, k)
                else
                    i_p = i; has_p = false
                end

                if i > 1
                    i_m = i - 1; has_m = Grids.isactive(grid, i_m, j, k)
                elseif periodic_lon
                    i_m = Nlon; has_m = Grids.isactive(grid, i_m, j, k)
                else
                    i_m = i; has_m = false
                end

                h_m, h_p = Grids._local_spacing(lon, i, period)
                h_m_phys = rk * cosφ * h_m
                h_p_phys = rk * cosφ * h_p

                if has_p && has_m
                    ∂f∂x[i, j, k] = Geometry.nonuniform_first_derivative(
                        f[i_m, j, k], f[i, j, k], f[i_p, j, k], h_m_phys, h_p_phys,
                    )
                elseif has_p
                    ∂f∂x[i, j, k] = (f[i_p, j, k] - f[i, j, k]) / h_p_phys
                elseif has_m
                    ∂f∂x[i, j, k] = (f[i, j, k] - f[i_m, j, k]) / h_m_phys
                else
                    ∂f∂x[i, j, k] = zero(T)
                end
            end
        end
    end
    return ∂f∂x
end

function ddy!(
    ∂f∂y::AbstractArray{T,3},
    f::AbstractArray{T,3},
    grid::Grids.StructuredGrid{Geometry.SphericalGeometry{T},T,3},
) where {T<:AbstractFloat}
    Nlon, Nlat, Nr = Grids.size_tuple(grid)
    _, lat, r = grid.axes

    @inbounds for k in 1:Nr
        rk = r[k]
        for j in 1:Nlat
            has_p = j < Nlat
            has_m = j > 1
            h_m, h_p = Grids._local_spacing(lat, j)
            h_m_phys = rk * h_m
            h_p_phys = rk * h_p
            for i in 1:Nlon
                if !Grids.isactive(grid, i, j, k)
                    ∂f∂y[i, j, k] = zero(T)
                    continue
                end
                hp = has_p && Grids.isactive(grid, i, j+1, k)
                hm = has_m && Grids.isactive(grid, i, j-1, k)
                if hp && hm
                    ∂f∂y[i, j, k] = Geometry.nonuniform_first_derivative(
                        f[i, j-1, k], f[i, j, k], f[i, j+1, k], h_m_phys, h_p_phys,
                    )
                elseif hp
                    ∂f∂y[i, j, k] = (f[i, j+1, k] - f[i, j, k]) / h_p_phys
                elseif hm
                    ∂f∂y[i, j, k] = (f[i, j, k] - f[i, j-1, k]) / h_m_phys
                else
                    ∂f∂y[i, j, k] = zero(T)
                end
            end
        end
    end
    return ∂f∂y
end

function ddz!(
    ∂f∂z::AbstractArray{T,3},
    f::AbstractArray{T,3},
    grid::Grids.StructuredGrid{Geometry.SphericalGeometry{T},T,3},
) where {T<:AbstractFloat}
    Nlon, Nlat, Nr = Grids.size_tuple(grid)
    r = grid.axes[3]

    @inbounds for k in 1:Nr
        has_p = k < Nr
        has_m = k > 1
        for j in 1:Nlat, i in 1:Nlon
            if !Grids.isactive(grid, i, j, k)
                ∂f∂z[i, j, k] = zero(T)
                continue
            end
            hp = has_p && Grids.isactive(grid, i, j, k+1)
            hm = has_m && Grids.isactive(grid, i, j, k-1)
            h_m, h_p = Grids._local_spacing(r, k)
            if hp && hm
                ∂f∂z[i, j, k] = Geometry.nonuniform_first_derivative(f[i, j, k-1], f[i, j, k], f[i, j, k+1], h_m, h_p)
            elseif hp
                ∂f∂z[i, j, k] = (f[i, j, k+1] - f[i, j, k]) / h_p
            elseif hm
                ∂f∂z[i, j, k] = (f[i, j, k] - f[i, j, k-1]) / h_m
            else
                ∂f∂z[i, j, k] = zero(T)
            end
        end
    end
    return ∂f∂z
end

# ---------------------------------------------------------------------------
# Curvilinear-grid gradients: weighted least-squares (WLSQ) reconstruction
# ---------------------------------------------------------------------------
#
# A curvilinear grid has no separable coordinate axis, so there is no `_local_spacing`-style scalar
# stencil to feed `nonuniform_first_derivative`. Instead we reconstruct the local physical gradient
# ∇f = (∂f/∂East, ∂f/∂North) at each node from the up-to-four immediate index neighbours
# (i±1,j)/(i,j±1) by a weighted least-squares fit in the local tangent plane.
#
# For a node with tangent-plane-projected neighbour displacements Δr_k = (rx_k, ry_k) (via
# `Geometry.project_to_tangent_plane`) and value differences Δf_k = f_k - f_0, minimising
# Σ_k w_k (∇f·Δr_k - Δf_k)² with weights w_k = 1/|Δr_k|² gives the 2×2 normal equations
#
#     A ∇f = b,   A = Σ_k w_k (Δr_k ⊗ Δr_k),   b = Σ_k w_k Δr_k Δf_k.
#
# `A` (the weighted Gram matrix) is symmetric positive-semidefinite and depends only on grid
# geometry, NOT on the field, so it is built ONCE per grid into a `WLSQGradientPlan`. Since
# ∇f = A⁻¹ b = Σ_k (A⁻¹ w_k Δr_k) Δf_k, we precompute per-neighbour coefficient vectors
# c_k = A⁻¹ (w_k Δr_k); then each `ddx!`/`ddy!` is just `∂x = Σ_k cx_k Δf_k` / `∂y = Σ_k cy_k Δf_k`
# — one cached dot product per node, zero allocation. This is deliberately NOT a raw 2×2
# index-space Jacobian inverse: the WLSQ combination cancels the leading truncation term (linear
# fields are reconstructed exactly on ANY stencil), whereas dividing two independently-differenced
# quantities does not.
#
# WLSQ over the four-point stencil is a LINEAR reconstruction, hence conditionally 2nd order:
# exact for linear fields on any stencil, 2nd order on locally-symmetric (e.g. uniform separable)
# stencils, and degrading toward 1st order on strongly skewed cells. On a separable orthogonal
# stencil `A` is diagonal and the fit decouples per axis, reducing to the ordinary centered/one-sided
# difference the `StructuredGrid` engine uses.
#
# The stencil reuses the `isactive`-based one-sided fallback of the `StructuredGrid` methods: a
# neighbour enters only if in-bounds and active, so boundary/masked-adjacent nodes silently use fewer
# points. If a whole tangent direction has no data (`A` rank-deficient), that undetermined gradient
# component is set to zero (matching the `StructuredGrid` "isolated point ⇒ 0" convention); a node
# with no active neighbours has a zero gradient. `CurvilinearGrid` is treated as non-periodic (no
# seam wrap).

"""
    WLSQGradientPlan{T, VI, VT}

Cached per-node weighted-least-squares gradient system for a [`Grids.CurvilinearGrid`](@ref) (built
once via `WLSQGradientPlan(grid)`, reused across every `ddx!`/`ddy!` call — the same "build once,
apply many" discipline as `Diagnostics.ΠWorkspace`). CSR-like layout: node `t = i + (j-1)·Nlon`
owns entries `ptr[t]:ptr[t+1]-1`, each a neighbour offset `(di, dj)` and its precomputed gradient
coefficients `(cx, cy) = A⁻¹ (w Δr)`, so `∂f/∂East = Σ cx·Δf` and `∂f/∂North = Σ cy·Δf`.

The storage container types are type parameters, not hardcoded `Vector`s: `VI<:AbstractVector{Int}`
is shared by the integer index arrays (`di`/`dj`/`ptr` — same role, legitimately the same concrete
type) and `VT<:AbstractVector{T}` by the coefficient arrays (`cx`/`cy`), so the plan is not
over-constrained to CPU `Vector` storage and infers precisely from however it was built.
"""
struct WLSQGradientPlan{T<:AbstractFloat, VI<:AbstractVector{Int}, VT<:AbstractVector{T}}
    di::VI              # neighbour i-offset (∈ {+1,-1,0})
    dj::VI              # neighbour j-offset (∈ {0,+1,-1})
    cx::VT              # coefficient of Δf in ∂f/∂East
    cy::VT              # coefficient of Δf in ∂f/∂North
    ptr::VI             # node t = i + (j-1)·Nlon owns entries ptr[t]:ptr[t+1]-1
    dims::Tuple{Int,Int}
end

const _WLSQ_OFFSETS = ((1, 0), (-1, 0), (0, 1), (0, -1))

"""
    WLSQGradientPlan(grid::Grids.CurvilinearGrid)

Build the per-node WLSQ gradient system (geometry only, field-independent) — see the module comment.
One-time O(N) setup; cache and reuse across repeated `ddx!`/`ddy!` calls.
"""
function WLSQGradientPlan(grid::Grids.CurvilinearGrid{T,G}) where {T<:AbstractFloat, G}
    Nlon, Nlat = Grids.size_tuple(grid)
    geo = grid.geometry
    # Hard upper bound: the stencil is exactly the 4 fixed `_WLSQ_OFFSETS`, so at most 4 entries per
    # node — preallocate once (no push!/amortized-growth reallocation) and trim to the true count `n`.
    maxentries = 4 * Nlon * Nlat
    di = Vector{Int}(undef, maxentries)
    dj = Vector{Int}(undef, maxentries)
    cx = Vector{T}(undef, maxentries)
    cy = Vector{T}(undef, maxentries)
    ptr = Vector{Int}(undef, Nlon * Nlat + 1)
    ptr[1] = 1
    n = 0
    for j in 1:Nlat, i in 1:Nlon
        t = i + (j - 1) * Nlon
        if !Grids.isactive(grid, i, j)
            ptr[t+1] = n + 1
            continue
        end
        c0 = Grids.coords(grid, i, j)
        # Pass 1: assemble the weighted normal (Gram) matrix A over the valid active neighbours.
        Axx = zero(T); Ayy = zero(T); Axy = zero(T)
        for (ddi, ddj) in _WLSQ_OFFSETS
            in_ = i + ddi; jn = j + ddj
            (1 <= in_ <= Nlon && 1 <= jn <= Nlat) || continue
            Grids.isactive(grid, in_, jn) || continue
            Δr = Geometry.project_to_tangent_plane(geo, c0, Grids.coords(grid, in_, jn))
            r2 = Δr[1]^2 + Δr[2]^2
            r2 > zero(T) || continue
            w = one(T) / r2
            Axx += w * Δr[1] * Δr[1]
            Ayy += w * Δr[2] * Δr[2]
            Axy += w * Δr[1] * Δr[2]
        end
        det = Axx * Ayy - Axy * Axy
        tr = Axx + Ayy
        full_rank = det > eps(T) * max(tr * tr, one(T))
        # Pass 2: per-neighbour coefficient c_k = A⁻¹ (w_k Δr_k).
        for (ddi, ddj) in _WLSQ_OFFSETS
            in_ = i + ddi; jn = j + ddj
            (1 <= in_ <= Nlon && 1 <= jn <= Nlat) || continue
            Grids.isactive(grid, in_, jn) || continue
            Δr = Geometry.project_to_tangent_plane(geo, c0, Grids.coords(grid, in_, jn))
            r2 = Δr[1]^2 + Δr[2]^2
            r2 > zero(T) || continue
            w = one(T) / r2
            gx = w * Δr[1]; gy = w * Δr[2]
            if full_rank
                ckx = (Ayy * gx - Axy * gy) / det
                cky = (Axx * gy - Axy * gx) / det
            else
                # Rank-deficient stencil (a whole tangent direction carries no data): recover only
                # the well-determined axis-aligned components; the undetermined direction ⇒ 0.
                ckx = Axx > zero(T) ? gx / Axx : zero(T)
                cky = Ayy > zero(T) ? gy / Ayy : zero(T)
            end
            n += 1
            di[n] = ddi; dj[n] = ddj; cx[n] = ckx; cy[n] = cky
        end
        ptr[t+1] = n + 1
    end
    resize!(di, n); resize!(dj, n); resize!(cx, n); resize!(cy, n)
    return WLSQGradientPlan{T, typeof(di), typeof(cx)}(di, dj, cx, cy, ptr, (Nlon, Nlat))
end

"""
    ddx!(∂f∂x, f, grid::CurvilinearGrid[, plan::WLSQGradientPlan])

Eastward physical gradient of `f` on a curvilinear grid via cached WLSQ reconstruction. Pass a
prebuilt `plan` (from `WLSQGradientPlan(grid)`) to avoid rebuilding the per-node system on every
call — callers doing repeated derivatives (e.g. `compute_Π!`) should build it once and reuse it.
"""
function ddx!(
    ∂f∂x::AbstractMatrix{T}, f::AbstractMatrix{T},
    grid::Grids.CurvilinearGrid{T}, plan::WLSQGradientPlan{T},
) where {T<:AbstractFloat}
    Nlon, Nlat = Grids.size_tuple(grid)
    @inbounds for j in 1:Nlat, i in 1:Nlon
        if !Grids.isactive(grid, i, j)
            ∂f∂x[i, j] = zero(T)
            continue
        end
        t = i + (j - 1) * Nlon
        f0 = f[i, j]
        s = zero(T)
        for k in plan.ptr[t]:(plan.ptr[t+1] - 1)
            s += plan.cx[k] * (f[i + plan.di[k], j + plan.dj[k]] - f0)
        end
        ∂f∂x[i, j] = s
    end
    return ∂f∂x
end

"""
    ddy!(∂f∂y, f, grid::CurvilinearGrid[, plan::WLSQGradientPlan])

Northward physical gradient of `f` on a curvilinear grid via cached WLSQ reconstruction (see [`ddx!`](@ref)).
"""
function ddy!(
    ∂f∂y::AbstractMatrix{T}, f::AbstractMatrix{T},
    grid::Grids.CurvilinearGrid{T}, plan::WLSQGradientPlan{T},
) where {T<:AbstractFloat}
    Nlon, Nlat = Grids.size_tuple(grid)
    @inbounds for j in 1:Nlat, i in 1:Nlon
        if !Grids.isactive(grid, i, j)
            ∂f∂y[i, j] = zero(T)
            continue
        end
        t = i + (j - 1) * Nlon
        f0 = f[i, j]
        s = zero(T)
        for k in plan.ptr[t]:(plan.ptr[t+1] - 1)
            s += plan.cy[k] * (f[i + plan.di[k], j + plan.dj[k]] - f0)
        end
        ∂f∂y[i, j] = s
    end
    return ∂f∂y
end

# Convenience: build the plan internally (rebuilt each call — prefer passing a cached plan in hot loops).
ddx!(∂f∂x::AbstractMatrix{T}, f::AbstractMatrix{T}, grid::Grids.CurvilinearGrid{T}) where {T<:AbstractFloat} =
    ddx!(∂f∂x, f, grid, WLSQGradientPlan(grid))
ddy!(∂f∂y::AbstractMatrix{T}, f::AbstractMatrix{T}, grid::Grids.CurvilinearGrid{T}) where {T<:AbstractFloat} =
    ddy!(∂f∂y, f, grid, WLSQGradientPlan(grid))

# ---------------------------------------------------------------------------
# Unstructured-grid gradients: the SAME weighted least-squares (WLSQ) reconstruction as
# `CurvilinearGrid` above, but over each node's REAL adjacency list (`Grids.neighbors(grid, idx)`,
# built by a k-d tree — see `Grids._build_kdtree_neighbors`) rather than a fixed 4-point index-offset
# stencil. There is no `(i,j)` index space at all here (nodes are scattered), so the plan stores
# neighbour entries as absolute NODE indices instead of `(di, dj)` offsets — otherwise identical math
# (same normal-equations solve, same weights `w_k = 1/|Δr_k|²`, same rank-deficient-stencil fallback).
# ---------------------------------------------------------------------------

"""
    UnstructuredWLSQGradientPlan{T, VI, VT}

Cached per-node weighted-least-squares gradient system for a [`Grids.UnstructuredGrid`](@ref) (built
once via `WLSQGradientPlan(grid)`, reused across every `ddx!`/`ddy!` call — same discipline as
[`WLSQGradientPlan`](@ref)/`Diagnostics.ΠWorkspace`). CSR layout: node `i` owns entries
`ptr[i]:ptr[i+1]-1`, each a neighbour NODE INDEX (absolute, not an index-space offset — there is no
index space to offset from on a scattered mesh) and its precomputed gradient coefficients
`(cx, cy) = A⁻¹ (w Δr)`.

Since each node's neighbour COUNT is already known exactly in advance from the grid's own
`neighbor_ptr` (no fixed stencil-size upper bound needed, unlike `CurvilinearGrid`'s fixed 4-point
stencil), storage is preallocated to that EXACT total, then trimmed only for skipped masked
nodes/neighbours or degenerate (zero-displacement) pairs.
"""
struct UnstructuredWLSQGradientPlan{T<:AbstractFloat, VI<:AbstractVector{Int}, VT<:AbstractVector{T}}
    nbr::VI             # neighbour NODE index (absolute)
    cx::VT              # coefficient of Δf in ∂f/∂East
    cy::VT              # coefficient of Δf in ∂f/∂North
    ptr::VI             # node i owns entries ptr[i]:ptr[i+1]-1
end

"""
    WLSQGradientPlan(grid::Grids.UnstructuredGrid)

Build the per-node WLSQ gradient system over `grid`'s real k-d-tree adjacency (geometry only,
field-independent) — see the module comment above. One-time O(N·k̄) setup; cache and reuse across
repeated `ddx!`/`ddy!` calls. Returns an [`UnstructuredWLSQGradientPlan`](@ref).
"""
function WLSQGradientPlan(grid::Grids.UnstructuredGrid{T}) where {T<:AbstractFloat}
    N = length(grid.mask)
    geo = grid.geometry
    # Exact upper bound: every retained entry corresponds to one flat adjacency-array slot, so the
    # true count over the whole grid can never exceed `length(grid.neighbor_nbrs)` — preallocate to
    # that (not a conservative sizehint!) and trim only for skipped masked/degenerate pairs.
    maxentries = length(grid.neighbor_nbrs)
    nbr = Vector{Int}(undef, maxentries)
    cx = Vector{T}(undef, maxentries)
    cy = Vector{T}(undef, maxentries)
    ptr = Vector{Int}(undef, N + 1)
    ptr[1] = 1
    n = 0
    for i in 1:N
        if !Grids.isactive(grid, i)
            ptr[i+1] = n + 1
            continue
        end
        c0 = Grids.coords(grid, i)
        # Pass 1: assemble the weighted normal (Gram) matrix A over the valid active neighbours.
        Axx = zero(T); Ayy = zero(T); Axy = zero(T)
        for j in Grids.neighbors(grid, i)
            Grids.isactive(grid, j) || continue
            Δr = Geometry.project_to_tangent_plane(geo, c0, Grids.coords(grid, j))
            r2 = Δr[1]^2 + Δr[2]^2
            r2 > zero(T) || continue
            w = one(T) / r2
            Axx += w * Δr[1] * Δr[1]
            Ayy += w * Δr[2] * Δr[2]
            Axy += w * Δr[1] * Δr[2]
        end
        det = Axx * Ayy - Axy * Axy
        tr = Axx + Ayy
        full_rank = det > eps(T) * max(tr * tr, one(T))
        # Pass 2: per-neighbour coefficient c_k = A⁻¹ (w_k Δr_k).
        for j in Grids.neighbors(grid, i)
            Grids.isactive(grid, j) || continue
            Δr = Geometry.project_to_tangent_plane(geo, c0, Grids.coords(grid, j))
            r2 = Δr[1]^2 + Δr[2]^2
            r2 > zero(T) || continue
            w = one(T) / r2
            gx = w * Δr[1]; gy = w * Δr[2]
            if full_rank
                ckx = (Ayy * gx - Axy * gy) / det
                cky = (Axx * gy - Axy * gx) / det
            else
                # Rank-deficient stencil (a whole tangent direction carries no data): recover only
                # the well-determined axis-aligned component; the undetermined direction ⇒ 0.
                ckx = Axx > zero(T) ? gx / Axx : zero(T)
                cky = Ayy > zero(T) ? gy / Ayy : zero(T)
            end
            n += 1
            nbr[n] = j; cx[n] = ckx; cy[n] = cky
        end
        ptr[i+1] = n + 1
    end
    resize!(nbr, n); resize!(cx, n); resize!(cy, n)
    return UnstructuredWLSQGradientPlan{T, typeof(nbr), typeof(cx)}(nbr, cx, cy, ptr)
end

"""
    ddx!(∂f∂x, f, grid::UnstructuredGrid[, plan::UnstructuredWLSQGradientPlan])

Eastward physical gradient of `f` on an unstructured (scattered) grid via cached WLSQ reconstruction
over the grid's real k-d-tree adjacency. Pass a prebuilt `plan` (from `WLSQGradientPlan(grid)`) to
avoid rebuilding the per-node system on every call.
"""
function ddx!(
    ∂f∂x::AbstractVector{T}, f::AbstractVector{T},
    grid::Grids.UnstructuredGrid{T}, plan::UnstructuredWLSQGradientPlan{T},
) where {T<:AbstractFloat}
    N = length(grid.mask)
    @inbounds for i in 1:N
        if !Grids.isactive(grid, i)
            ∂f∂x[i] = zero(T)
            continue
        end
        f0 = f[i]
        s = zero(T)
        for k in plan.ptr[i]:(plan.ptr[i+1] - 1)
            s += plan.cx[k] * (f[plan.nbr[k]] - f0)
        end
        ∂f∂x[i] = s
    end
    return ∂f∂x
end

"""
    ddy!(∂f∂y, f, grid::UnstructuredGrid[, plan::UnstructuredWLSQGradientPlan])

Northward physical gradient of `f` on an unstructured (scattered) grid via cached WLSQ
reconstruction (see [`ddx!`](@ref)).
"""
function ddy!(
    ∂f∂y::AbstractVector{T}, f::AbstractVector{T},
    grid::Grids.UnstructuredGrid{T}, plan::UnstructuredWLSQGradientPlan{T},
) where {T<:AbstractFloat}
    N = length(grid.mask)
    @inbounds for i in 1:N
        if !Grids.isactive(grid, i)
            ∂f∂y[i] = zero(T)
            continue
        end
        f0 = f[i]
        s = zero(T)
        for k in plan.ptr[i]:(plan.ptr[i+1] - 1)
            s += plan.cy[k] * (f[plan.nbr[k]] - f0)
        end
        ∂f∂y[i] = s
    end
    return ∂f∂y
end

# Convenience: build the plan internally (rebuilt each call — prefer passing a cached plan in hot loops).
ddx!(∂f∂x::AbstractVector{T}, f::AbstractVector{T}, grid::Grids.UnstructuredGrid{T}) where {T<:AbstractFloat} =
    ddx!(∂f∂x, f, grid, WLSQGradientPlan(grid))
ddy!(∂f∂y::AbstractVector{T}, f::AbstractVector{T}, grid::Grids.UnstructuredGrid{T}) where {T<:AbstractFloat} =
    ddy!(∂f∂y, f, grid, WLSQGradientPlan(grid))

end # module
