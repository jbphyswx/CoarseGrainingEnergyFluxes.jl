# ---------------------------------------------------------------------------
# The banded engine: one offset/weight list per row of targets, for a 2-D grid uniform along axis 1
# ---------------------------------------------------------------------------

"""
    BandedScratch{T,MT,VM} <: AbstractFilterScratch

Per-apply buffers for the banded engine: the masked input `mask · field`, one per field in flight.

Empty on an unmasked grid, where the engine convolves the caller's array directly. Slots are added on
demand for a batched apply and reused by later applies.
"""
struct BandedScratch{
    T<:AbstractFloat, MT<:AbstractMatrix{T}, VM<:AbstractVector{MT},
} <: AbstractFilterScratch
    masked_inputs::VM
end

# Grows to `K` slots, each `similar` to slot 1, which must exist: callers hold a masked footprint.
function _banded_inputs!(sc::BandedScratch, K::Integer)
    while length(sc.masked_inputs) < K
        push!(sc.masked_inputs, similar(@inbounds sc.masked_inputs[1]))
    end
    return sc.masked_inputs
end

"""
    BandTable{VI,VT}

The banded engine's taps by band. Band `b` holds runs `ptr[b]:ptr[b+1]-1`, and run `s` covers the
consecutive axis-1 offsets `di0[s] + t − 1` at axis-2 offset `dj[s]`, with weights `w[start[s] + t − 1]`
for `t = 1:(start[s+1] − start[s])`.
"""
struct BandTable{VI<:AbstractVector{Int}, VT<:AbstractVector}
    dj::VI
    di0::VI
    start::VI
    w::VT
    ptr::VI
end

"""
    _band_table(di, dj, w, ptr) -> BandTable

The flat per-tap lists a builder produces (band `b`'s taps `ptr[b]:ptr[b+1]-1`) grouped into runs: each
band's taps ordered by axis-2 then axis-1 offset, a run broken wherever the axis-2 offset changes or the
axis-1 offset skips.
"""
function _band_table(
    di::AbstractVector{Int}, dj::AbstractVector{Int}, w::AbstractVector{T}, ptr::AbstractVector{Int},
) where {T}
    nb = length(ptr) - 1
    rdj, rdi, rstart = Int[], Int[], Int[]
    rw = sizehint!(T[], length(w))
    rptr = Vector{Int}(undef, nb + 1)
    rptr[1] = 1
    for b in 1:nb
        prev = 0
        for k in sort(ptr[b]:(ptr[b + 1] - 1); by = k -> (dj[k], di[k]))
            if prev == 0 || dj[k] != dj[prev] || di[k] != di[prev] + 1
                push!(rdj, dj[k])
                push!(rdi, di[k])
                push!(rstart, length(rw) + 1)
            end
            push!(rw, w[k])
            prev = k
        end
        rptr[b + 1] = length(rdj) + 1
    end
    push!(rstart, length(rw) + 1)
    return BandTable(rdj, rdi, rstart, rw, rptr)
end

"""
    FilterFootprint{T,BT,MT,SC,MS}

Precomputed convolution footprint for a structured grid + kernel + scale whose axis 1 is uniform. The
in-support neighbour offsets and their geometric weights `w = kernel_weight(distance) * cell_area` are
held in a [`BandTable`](@ref), grouped into axis-2 (`y`) bands. A Cartesian grid uniform in both
directions is translation-invariant, a single band; otherwise the footprint is invariant along axis 1
only, one band per row.

The offsets and weights depend on the geometry alone. `invden`, the reciprocal window mass, also depends
on the mask, the mask strategy and where a domain edge cuts the window; [`_banded_build_invden`](@ref)
accumulates it at plan build.
"""
struct FilterFootprint{
    T<:AbstractFloat, BT<:BandTable, MT<:AbstractMatrix{T}, SC<:BandedScratch{T}, MS<:AbstractMaskStrategy,
}
    taps::BT           # offsets and weights by band
    nbands::Int        # 1 (uniform Cartesian) or Ny
    strategy::MS       # the strategy `invden` was accumulated under
    masked::Bool       # grid has inactive cells, so the numerator needs `mask · field`
    bound::Bool        # the strategies' denominators differ: a cell is inactive or a window leaves the grid
    # The grid's wrap at build, which `invden` was accumulated under. The apply uses these and ignores
    # the periodic flags it is passed.
    periodic_x::Bool
    periodic_y::Bool
    invden::MT         # 1/(window mass) per point; zero where the window is empty or, under Deformable, the target inactive
    scratch::SC        # per-apply `mask · field` buffers
end

"""
    _banded_row_accumulate!(oc, src, taps::BandTable, b, periodic_x, periodic_y, Nx, Ny, j) -> oc

Add output row `j`'s weighted tap sum over band `b` into `oc`, the taps of each run by
[`_run_accumulate!`](@ref). `src` is `mask · field` for the numerator at apply time and the mass field for
`invden` at plan build. On a bounded axis 1 a tap covers the columns whose source index stays in range.
"""
@inline function _banded_row_accumulate!(
    oc::AbstractVector{T}, src::AbstractMatrix, taps::BandTable, b::Integer,
    periodic_x::Bool, periodic_y::Bool, Nx::Int, Ny::Int, j::Integer,
) where {T<:AbstractFloat}
    w = taps.w
    @inbounds for s in taps.ptr[b]:(taps.ptr[b + 1] - 1)
        jj = j + taps.dj[s]
        if jj < 1 || jj > Ny
            periodic_y || continue
            jj = mod1(jj, Ny)
        end
        fc = view(src, :, jj)
        k0 = taps.start[s]
        _run_accumulate!(oc, fc, w, k0, taps.start[s + 1] - 1, taps.di0[s] - k0, periodic_x, Nx)
    end
    return oc
end

"""
    _run_accumulate!(oc, fc, w, k0, k1, d0, periodic, n) -> oc

`oc[i] += w[k] · fc[i + d0 + k]` for the taps `k0:k1` of one run, tap by tap, each a contiguous pass over
the line, wrapped where the line is periodic.
"""
@inline function _run_accumulate!(oc::AbstractVector{T}, fc::AbstractVector, w, k0::Int, k1::Int, d0::Int,
                                  periodic::Bool, n::Int) where {T}
    @inbounds for k in k0:k1
        _tap_add_range!(oc, fc, T(w[k]), d0 + k, periodic, n, 1, n)
    end
    return oc
end

# `oc[i] += wk · fc[i + d]` for `i ∈ lo:hi` whose source is on the line, or is brought back onto it where
# the line is periodic. With `|d| < n` the sources below, on and above the line are each a contiguous range
# of `i` at a constant offset.
@inline function _tap_add_range!(oc, fc, wk, d::Int, periodic::Bool, n::Int, lo::Int, hi::Int)
    @inbounds if !periodic
        @simd for i in max(lo, 1 - d):min(hi, n - d)
            oc[i] += wk * fc[i + d]
        end
    elseif abs(d) < n
        @simd for i in lo:min(hi, -d)
            oc[i] += wk * fc[i + d + n]
        end
        @simd for i in max(lo, 1 - d):min(hi, n - d)
            oc[i] += wk * fc[i + d]
        end
        @simd for i in max(lo, n - d + 1):hi
            oc[i] += wk * fc[i + d - n]
        end
    else
        @simd for i in lo:hi
            oc[i] += wk * fc[mod1(i + d, n)]
        end
    end
    return oc
end

@inline _banded_band(nbands::Int, j::Integer) = nbands == 1 ? 1 : Int(j)

"""
    _banded_unit_mass!(den, taps::BandTable, nbands, periodic_x, periodic_y) -> den

[`_banded_row_accumulate!`](@ref) over a mass field of ones, into a zeroed `den`, from the runs' weights
alone. On a periodic axis 1 every target of a row takes every tap, so the row is one sum, added in the
order the accumulation adds it. On a bounded axis 1 a target takes the part of each run whose sources stay
on the row, a difference of the run's prefix sums. Cost `O(taps)` per row on a periodic axis 1 and
`O(runs·Nx)` on a bounded one.
"""
function _banded_unit_mass!(
    den::AbstractMatrix{T}, taps::BandTable, nbands::Int, periodic_x::Bool, periodic_y::Bool,
) where {T<:AbstractFloat}
    Nx, Ny = size(den)
    w = taps.w
    cum = T[]
    @inbounds for j in 1:Ny
        b = _banded_band(nbands, j)
        acc = zero(T)
        for s in taps.ptr[b]:(taps.ptr[b + 1] - 1)
            jj = j + taps.dj[s]
            (1 <= jj <= Ny || periodic_y) || continue
            k0, k1 = taps.start[s], taps.start[s + 1] - 1
            if periodic_x
                for k in k0:k1
                    acc += T(w[k])
                end
            else
                L = k1 - k0 + 1
                resize!(cum, L + 1)
                cum[1] = zero(T)
                for t in 1:L
                    cum[t + 1] = cum[t] + T(w[k0 + t - 1])
                end
                # Tap `t` has offset `di0 + t − 1`, so target `i` takes `t` with `1 ≤ i + di0 + t − 1 ≤ Nx`.
                di0 = taps.di0[s]
                for i in 1:Nx
                    lo = max(1, 2 - i - di0)
                    hi = min(L, Nx - i - di0 + 1)
                    hi >= lo && (den[i, j] += cum[hi + 1] - cum[lo])
                end
            end
        end
        periodic_x && fill!(view(den, :, j), acc)
    end
    return den
end

"""
    _banded_build_invden(grid, taps, nbands, strategy, periodic_x, periodic_y, exterior) -> invden

The reciprocal window mass, accumulated once per plan by running [`_banded_row_accumulate!`](@ref)
over a mass field:

- `ZeroFill` counts every in-support tap whether or not its source cell is active, so its mass field
  is `1` everywhere, and adds `exterior`, the mass past a bounded edge: the kernel's full mass. An
  `exterior` vector is that full mass per row, and is the denominator itself.
- `Deformable` drops inactive cells from both sums, so its mass field is the mask, and an inactive
  target gets zero.
"""
function _banded_build_invden(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2}, taps::BandTable, nbands::Int,
    strategy::AbstractMaskStrategy, periodic_x::Bool, periodic_y::Bool,
    exterior::Union{Nothing,AbstractArray{T}},
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    zerofill = strategy isa ZeroFill
    den = zeros(T, Nx, Ny)
    if zerofill && exterior isa AbstractVector
        @inbounds for j in 1:Ny, i in 1:Nx
            den[i, j] = exterior[j]
        end
    elseif zerofill || all(FlowGeometries.Grids.mask(grid))
        _banded_unit_mass!(den, taps, nbands, periodic_x, periodic_y)
        zerofill && exterior !== nothing && (den .+= exterior)
    else
        mass = Matrix{T}(undef, Nx, Ny)
        @inbounds for j in 1:Ny, i in 1:Nx
            mass[i, j] = FlowGeometries.Grids.isactive(grid, i, j) ? one(T) : zero(T)
        end
        for j in 1:Ny
            _banded_row_accumulate!(
                view(den, :, j), mass, taps, _banded_band(nbands, j), periodic_x, periodic_y, Nx, Ny, j,
            )
        end
    end
    @inbounds for j in 1:Ny, i in 1:Nx
        keep = zerofill || FlowGeometries.Grids.isactive(grid, i, j)
        den[i, j] = keep ? _inv_mass(den[i, j]) : zero(T)
    end
    return den
end

# The `FilterFootprint` of a builder's per-tap lists `di, dj, w, ptr`. `exterior` is
# [`_banded_build_invden`](@ref)'s.
function _banded_footprint(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    di::AbstractVector{Int}, dj::AbstractVector{Int}, w::AbstractVector{T},
    ptr::AbstractVector{Int}, nbands::Int, strategy::AbstractMaskStrategy,
    kernel::Kernels.AbstractFilterKernel, scale::T;
    exterior::Union{Nothing,AbstractArray{T}} =
        strategy isa ZeroFill ? _exterior_mass(grid, kernel, scale) : nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    periodic_x = FlowGeometries.Grids.isperiodic(grid, 1)
    periodic_y = FlowGeometries.Grids.isperiodic(grid, 2)
    masked = !all(FlowGeometries.Grids.mask(grid))
    taps = _band_table(di, dj, w, ptr)
    invden = _banded_build_invden(grid, taps, nbands, strategy, periodic_x, periodic_y, exterior)
    scratch = BandedScratch(masked ? [zeros(T, Nx, Ny)] : Matrix{T}[])
    return FilterFootprint(
        taps, nbands, strategy, masked, masked || _reaches_exterior(grid, kernel, scale),
        periodic_x, periodic_y, invden, scratch,
    )
end

"""
    _banded_cartesian_rows!(di, dj, w, ptr, grid, kernel, scale, rad) -> nothing

One offset/weight band per row of a Cartesian grid uniform along `x` and stretched along `y`, appended to
`di, dj, w, ptr`. A periodic `y` tiles: a band reaches every row image inside the support, each at its
own displacement, and the apply maps a raw row offset back onto the grid with `mod1`.
"""
function _banded_cartesian_rows!(
    di::AbstractVector{Int}, dj::AbstractVector{Int}, w::AbstractVector{T}, ptr::AbstractVector{Int},
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2}, kernel::Kernels.AbstractFilterKernel, scale::T, rad::T,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    _, Ny = FlowGeometries.Grids.size_tuple(grid)
    dx = step(FlowGeometries.Grids.coordinates(grid, 1))
    y = FlowGeometries.Grids.coordinates(grid, 2)
    periodic_y = FlowGeometries.Grids.isperiodic(grid, 2) && Ny > 1
    Py = periodic_y ? T(FlowGeometries.Grids.period(grid, 2)) : zero(T)
    sy = (Ny > 1 && y[Ny] < y[1]) ? -one(T) : one(T)   # coordinate direction of increasing index
    min_dy = T(FlowGeometries.Grids.minimum_spacing(grid, 2))
    di_lim = dx > 0 ? ceil(Int, rad / dx) : 0
    dj_lim = min_dy > 0 ? ceil(Int, rad / min_dy) : 0
    for j in 1:Ny
        for ddj in -dj_lim:dj_lim
            jraw = j + ddj
            (periodic_y || 1 <= jraw <= Ny) || continue
            m = fld(jraw - 1, Ny)
            jj = jraw - m * Ny
            Δy = T(y[jj]) + sy * T(m) * Py - T(y[j])
            abs(Δy) <= rad || continue
            A = FlowGeometries.Grids.area(grid, 1, jj)
            for ddi in -di_lim:di_lim
                d = sqrt((ddi * dx)^2 + Δy^2)
                d <= rad || continue
                push!(di, ddi)
                push!(dj, ddj)
                push!(w, Kernels.kernel_weight(kernel, T(d), scale, Val(2)) * A)
            end
        end
        push!(ptr, length(di) + 1)
    end
    return nothing
end

"""
    _structured_footprint(::Tuple{UniformSpacing,Any}, grid, kernel, scale; mask_strategy = ZeroFill()) -> FilterFootprint

The banded engine, for a 2-D grid whose axis 1 is uniform: constant spacing along axis 1 makes the
footprint invariant under a shift along it, so each row of targets shares one offset/weight list. A
Cartesian grid uniform in both directions has one list for every row; otherwise each row has its own
(`nbands = Ny`), on a stretched Cartesian `y` and on any latitude axis of a sphere.
"""
function _structured_footprint(
    sp::Tuple{FlowGeometries.Axes.UniformSpacing,Any},
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    rad = Kernels.kernel_radius(kernel, scale, Val(2))
    if !(G <: FlowGeometries.Geometry.CartesianGeometry{T})
        di, dj, w, ptr = _sphere_band_rows(grid, kernel, scale, rad)
        execution = get(kwargs, :execution, ComputationalBackends.SerialBackend())
        return _banded_footprint(grid, di, dj, w, ptr, Ny, mask_strategy, kernel, scale;
                                 exterior = mask_strategy isa ZeroFill ? _exterior_mass(grid, kernel, scale, execution) : nothing)
    end

    di = Int[]
    dj = Int[]
    w = T[]
    ptr = Int[1]

    if sp[2] isa FlowGeometries.Axes.UniformSpacing
        dx = step(FlowGeometries.Grids.coordinates(grid, 1))
        dy = step(FlowGeometries.Grids.coordinates(grid, 2))
        A = FlowGeometries.Grids.area(grid, 1, 1)   # uniform Cartesian cell area
        di_lim = dx > 0 ? ceil(Int, rad / dx) : 0
        dj_lim = dy > 0 ? ceil(Int, rad / dy) : 0
        sizehint!(di, (2*di_lim + 1) * (2*dj_lim + 1))
        sizehint!(dj, (2*di_lim + 1) * (2*dj_lim + 1))
        sizehint!(w, (2*di_lim + 1) * (2*dj_lim + 1))
        for ddj in -dj_lim:dj_lim, ddi in -di_lim:di_lim
            d = sqrt((ddi * dx)^2 + (ddj * dy)^2)
            if d <= rad
                push!(di, ddi)
                push!(dj, ddj)
                push!(w, Kernels.kernel_weight(kernel, T(d), scale, Val(2)) * A)
            end
        end
        push!(ptr, length(di) + 1)
        return _banded_footprint(grid, di, dj, w, ptr, 1, mask_strategy, kernel, scale)
    else
        _banded_cartesian_rows!(di, dj, w, ptr, grid, kernel, scale, rad)
        return _banded_footprint(grid, di, dj, w, ptr, Ny, mask_strategy, kernel, scale)
    end
end

"""
    _sphere_band_rows(grid, kernel, scale, rad) -> (di, dj, w, ptr)

The taps of a sphere of uniform longitude by latitude row: row `j`'s entries `ptr[j]:ptr[j+1]-1` hold
the longitude and latitude offsets of the cells within great-circle distance `rad` of a target in row `j`
and their weights `kernel_weight(d)·area`. Uniform longitude makes them the same for every target of a
row.
"""
function _sphere_band_rows(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2}, kernel::Kernels.AbstractFilterKernel, scale::T, rad::T,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    di, dj, w, ptr, _ = _collect_rows(SphereBandRows(grid, kernel, scale, rad), T)
    return di, dj, w, ptr
end

"""
    SphereBandRows(grid, kernel, scale, rad)

The taps of a sphere of uniform longitude, one latitude row at a time ([`_row_taps!`](@ref)): each row's
window from `metric_window`, which holds every row its ball reaches (a row nearer a pole spans more longitude at
the same radius), as longitude offsets ([`_banded_lon_offsets`](@ref)) and a latitude half-width, and `havλ[k + 1]`
the `sin²(k·dλ/2)` of each longitude offset `±k`.
"""
struct SphereBandRows{
    GR, K, T<:AbstractFloat, VO<:AbstractVector{NTuple{3,UnitRange{Int}}}, VI<:AbstractVector{Int}, VT<:AbstractVector{T},
}
    grid::GR
    kernel::K
    scale::T
    rad::T
    offsets::VO
    dj_lims::VI
    havλ::VT
end

function SphereBandRows(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2}, kernel::Kernels.AbstractFilterKernel, scale::T, rad::T,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    dλ = step(FlowGeometries.Grids.coordinates(grid, 1))
    periodic_λ = FlowGeometries.Grids.isperiodic(grid, 1)
    offsets = Vector{NTuple{3,UnitRange{Int}}}(undef, Ny)
    dj_lims = Vector{Int}(undef, Ny)
    for j in 1:Ny
        dl, djl = FlowGeometries.Connectivity.metric_window(grid, (1, j), rad)
        offsets[j] = _banded_lon_offsets(dl, Nx, dλ, periodic_λ)
        dj_lims[j] = djl
    end
    havλ = [sin(T(k) * T(dλ) / T(2))^2 for k in 0:(Nx - 1)]
    return SphereBandRows(grid, kernel, scale, rad, offsets, dj_lims, havλ)
end

_nrows(r::SphereBandRows) = length(r.dj_lims)

# The widest latitude offset and the widest longitude offset any row's taps can take.
_row_dj_lim(r::SphereBandRows) = min(maximum(r.dj_lims; init = 0), _nrows(r) - 1)
_row_reach(r::SphereBandRows) = maximum(o -> maximum(q -> isempty(q) ? 0 : max(abs(first(q)), abs(last(q))), o), r.offsets; init = 0)

# An upper bound on the taps of every row, from the windows.
function _row_entries(r::SphereBandRows)
    Ny = _nrows(r)
    n = 0
    @inbounds for j in 1:Ny
        o, djl = r.offsets[j], r.dj_lims[j]
        n += (length(o[1]) + length(o[2]) + length(o[3])) * (min(Ny, j + djl) - max(1, j - djl) + 1)
    end
    return n
end

"""
    _row_taps!(f, rows, j) -> mass or nothing

Call `f(di, dj, w)` for every tap of latitude row `j`: its longitude and latitude offsets and weight, the same for
every target of the row. Returns the row's whole support mass where the taps leave part of it out (the cells
past a bounded latitude), `nothing` otherwise.
"""
function _row_taps!(f::F, r::SphereBandRows{GR,K,T}, j::Int) where {F, GR, K, T}
    grid = r.grid
    geo = FlowGeometries.Grids.grid_geometry(grid)
    lat = FlowGeometries.Grids.coordinates(grid, 2)
    Ny = _nrows(r)
    djl = r.dj_lims[j]
    for ddj in -djl:djl
        jj = j + ddj
        (1 <= jj <= Ny) || continue
        A = FlowGeometries.Grids.area(grid, 1, jj)   # spherical cell area depends only on latitude
        _row_pair_taps!(f, geo, r, j, ddj, T(lat[j]), T(lat[jj]), A)
    end
    return nothing
end

# Row `j`'s taps at latitude offset `ddj`, from target latitude `φ1` to source latitude `φ2`, each weighted by the
# source cell's measure `A`. On a sphere the haversine's part that the two latitudes fix is formed once and each
# longitude offset's from the table; on another surface the distance is FlowGeometries'.
@inline function _row_pair_taps!(
    f::F, geo::FlowGeometries.Geometry.AbstractSphericalGeometry, r::SphereBandRows{GR,K,T}, j::Int, ddj::Int,
    φ1::T, φ2::T, A,
) where {F, GR, K, T}
    hr = FlowGeometries.Geometry._hav_row(φ1, φ2)
    R = T(FlowGeometries.Geometry.radius(geo))
    @inbounds for range in r.offsets[j], ddi in range
        a = hr[1] + hr[2] * r.havλ[abs(ddi) + 1]
        d = R * (T(2) * atan(sqrt(a), sqrt(max(zero(T), one(T) - a))))
        d <= r.rad && f(ddi, ddj, Kernels.kernel_weight(r.kernel, d, r.scale, Val(2)) * A)
    end
    return nothing
end

@inline function _row_pair_taps!(f::F, geo, r::SphereBandRows{GR,K,T}, j::Int, ddj::Int, φ1::T, φ2::T, A) where {F, GR, K, T}
    dλ = step(FlowGeometries.Grids.coordinates(r.grid, 1))
    for range in r.offsets[j], ddi in range
        d = FlowGeometries.Geometry.distance(geo, SA.SVector{2,T}(zero(T), φ1), SA.SVector{2,T}(T(ddi) * dλ, φ2))
        d <= r.rad && f(ddi, ddj, Kernels.kernel_weight(r.kernel, d, r.scale, Val(2)) * A)
    end
    return nothing
end

_has_row_mass(::SphereBandRows) = false

# The rows as a concurrent worker generates them: the same object where it holds no scratch.
_row_worker(rows) = rows

# Every row's taps as the flat lists `(di, dj, w, ptr)` the banded engine groups into runs, and each row's whole
# support mass where the rows give one, `nothing` otherwise.
function _collect_rows(rows, ::Type{T}) where {T}
    n = _row_entries(rows)
    di = sizehint!(Int[], n)
    dj = sizehint!(Int[], n)
    w = sizehint!(T[], n)
    ptr = Int[1]
    mass = _has_row_mass(rows) ? Vector{T}(undef, _nrows(rows)) : nothing
    for j in 1:_nrows(rows)
        m = _row_taps!(rows, j) do ddi, ddj, wt
            push!(di, ddi)
            push!(dj, ddj)
            push!(w, wt)
        end
        mass === nothing || (mass[j] = m)
        push!(ptr, length(di) + 1)
    end
    return di, dj, w, ptr, mass
end

"""
    ListedBandRows(di, dj, w, ptr)

Taps already listed by row, `ptr[j]:ptr[j+1]-1` for row `j`, served one row at a time ([`_row_taps!`](@ref)).
"""
struct ListedBandRows{VI<:AbstractVector{Int}, VT<:AbstractVector}
    di::VI
    dj::VI
    w::VT
    ptr::VI
end

_nrows(r::ListedBandRows) = length(r.ptr) - 1
_row_dj_lim(r::ListedBandRows) = maximum(abs, r.dj; init = 0)
_row_reach(r::ListedBandRows) = maximum(abs, r.di; init = 0)
_row_entries(r::ListedBandRows) = length(r.w)
_has_row_mass(::ListedBandRows) = false

function _row_taps!(f::F, r::ListedBandRows, j::Int) where {F}
    @inbounds for q in r.ptr[j]:(r.ptr[j + 1] - 1)
        f(r.di[q], r.dj[q], r.w[q])
    end
    return nothing
end

"""
    _banded_lon_offsets(dl, Nx, dλ, periodic) -> NTuple{3,UnitRange{Int}}

The longitude offsets a latitude row's taps are drawn from, for a window of `dl` steps of `dλ`. A ring
contributes each column once: `-dl:dl` capped at one turn, the antipodal column of an even ring taken from
one side. A regional box takes the near offsets `|Δi| ≤ dl` and the far ones that come back within the
window across its seam, `|Δi|·|dλ| ≥ 2π − dl·|dλ|`, up to `Nx − 1`; longitude identifies, so the
distance between two columns is taken the short way round.
"""
@inline function _banded_lon_offsets(dl::Int, Nx::Int, dλ::T, periodic::Bool) where {T<:AbstractFloat}
    if periodic
        lim = min(dl, Nx ÷ 2)
        return (1:0, -lim:(2lim + 1 > Nx ? lim - 1 : lim), 1:0)
    end
    near = min(dl, Nx - 1)
    s = abs(dλ)
    far = dl * s >= T(2π) ? near + 1 : max(near + 1, floor(Int, (T(2π) - dl * s) / s))
    return (-(Nx - 1):-far, -near:near, far:(Nx - 1))
end

"""
    _structured_footprint(::Tuple{UniformSpacing,Any}, grid::StructuredGrid{T,<:AbstractSphericalGeometry,2}, kernel::HighOrderKernel, scale; kwargs...)

A `HighOrderKernel` on a latitude–longitude grid of uniform longitude, any latitude axis: the banded
engine with one correction per row ([`_highorder_rows`](@ref)), or the per-cell plan where the targets of
a row do not share one.
"""
_structured_footprint(
    ::Tuple{FlowGeometries.Axes.UniformSpacing,Any}, grid::FlowGeometries.Grids.StructuredGrid{T,G,2},
    kernel::Kernels.HighOrderKernel,
    scale::T; kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}} =
    _highorder_spherical_footprint(grid, kernel, scale; kwargs...)

function _highorder_spherical_footprint(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,2}, kernel::Kernels.HighOrderKernel, scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(), kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}}
    rows = _highorder_rows(grid, kernel, scale)
    rows === nothing && return _build_footprint_scattered(grid, kernel, scale; kwargs...)
    di, dj, w, ptr, mass = rows
    _, Ny = FlowGeometries.Grids.size_tuple(grid)
    return _banded_footprint(grid, di, dj, w, ptr, Ny, mask_strategy, kernel, scale; exterior = mass)
end

"""
    apply_footprint!(out, field, grid, fp, strategy, periodic_x, periodic_y)

Convolve `field` with a precomputed `fp` into `out`, applying the mask `strategy`. `out` and
`field` are 2D (a single layer). Periodicity comes from `fp`; `periodic_x`/`periodic_y` are unused.
"""
function apply_footprint!(
    out::AbstractMatrix{T},
    field::AbstractMatrix,
    grid::FlowGeometries.Grids.StructuredGrid,
    fp::FilterFootprint{T},
    strategy::AbstractMaskStrategy,
    periodic_x::Bool,
    periodic_y::Bool,
) where {T<:AbstractFloat}
    _, Ny = FlowGeometries.Grids.size_tuple(grid)
    _banded_check_strategy(fp, strategy)
    _banded_fill_source!(fp, field, grid)
    for j in 1:Ny
        apply_footprint_row!(out, field, grid, fp, strategy, periodic_x, periodic_y, j)
    end
    return out
end

# `ZeroFill` divides by the kernel's full mass and `Deformable` by the mass of the active in-domain taps.
# The two agree when every cell is active and no window leaves the grid, which is when `fp.bound` is false.
@noinline function _banded_strategy_mismatch()
    throw(ArgumentError(
        "FilterFootprint is being applied with a different mask strategy than it was built for, on a " *
        "grid where the two normalize differently: a cell is inactive, or a window reaches past a " *
        "bounded edge. Its normalization is precomputed per scale from the grid, the mask and the " *
        "strategy, so one plan cannot serve both. Rebuild the plan with the `mask_strategy` you intend " *
        "to apply with.",
    ))
end

@inline function _banded_check_strategy(fp::FilterFootprint, strategy::AbstractMaskStrategy)
    (!fp.bound || typeof(strategy) === typeof(fp.strategy)) || _banded_strategy_mismatch()
    return nothing
end

"""
    _banded_source(fp, field, slot = 1) -> src

The array the tap loop convolves: on a masked grid the scratch slot `slot`, which `_banded_fill_source!`
must have filled with `mask · field`; otherwise `field` itself.
"""
@inline _banded_source(fp::FilterFootprint, field::AbstractMatrix, slot::Int = 1) =
    fp.masked ? @inbounds(fp.scratch.masked_inputs[slot]) : field

function _banded_fill_source!(
    fp::FilterFootprint{T}, field::AbstractMatrix, grid::FlowGeometries.Grids.StructuredGrid,
    slot::Int = 1,
) where {T<:AbstractFloat}
    fp.masked || return nothing
    buf = @inbounds fp.scratch.masked_inputs[slot]
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    @inbounds for j in 1:Ny, i in 1:Nx
        buf[i, j] = FlowGeometries.Grids.isactive(grid, i, j) ? T(field[i, j]) : zero(T)
    end
    return nothing
end

"""
    prepare_row_apply!(fp, field, grid) -> nothing
    prepare_row_apply_batch!(fp, fields, grid) -> nothing

Whole-grid work an engine needs done once before any of its per-row applies run: the banded engine
materializes its `mask · field` source, one slot per field for the batch form; the other engines need
nothing. A row-parallel backend calls `apply_footprint_row!` / `apply_footprint_row_batch!` directly,
so it must call this first.

Call it before opening a parallel region, never inside one: it writes the whole buffer.
"""
prepare_row_apply!(fp, field, grid) = nothing

prepare_row_apply!(
    fp::FilterFootprint, field::AbstractMatrix, grid::FlowGeometries.Grids.StructuredGrid,
) = _banded_fill_source!(fp, field, grid)

function prepare_row_apply_batch!(fp, fields, grid)
    return nothing
end

function prepare_row_apply_batch!(
    fp::FilterFootprint, fields, grid::FlowGeometries.Grids.StructuredGrid,
)
    fp.masked || return nothing
    _banded_inputs!(fp.scratch, length(fields))
    for m in eachindex(fields)
        _banded_fill_source!(fp, @inbounds(fields[m]), grid, m)
    end
    return nothing
end

"""
    apply_footprint_row!(out, field, grid, fp, strategy, periodic_x, periodic_y, j, sc = nothing)

Fill output row `j` (`out[:, j]`) from a precomputed footprint, as one contiguous axpy per tap
scaled by the plan's `invden`. Writes every entry of `out[:, j]` and nothing else, so rows may run
concurrently once `prepare_row_apply!` has run. The banded rows query nothing, so `sc` is unused.
"""
function apply_footprint_row!(
    out::AbstractMatrix{T},
    field::AbstractMatrix,
    grid::FlowGeometries.Grids.StructuredGrid,
    fp::FilterFootprint{T},
    strategy::AbstractMaskStrategy,
    periodic_x::Bool,
    periodic_y::Bool,
    j::Integer,
    _sc = nothing,
) where {T<:AbstractFloat}
    Nx, Ny = FlowGeometries.Grids.size_tuple(grid)
    oc = view(out, :, j)
    @inbounds @simd for i in 1:Nx
        oc[i] = zero(T)
    end
    _banded_row_accumulate!(
        oc, _banded_source(fp, field), fp.taps, _banded_band(fp.nbands, j),
        fp.periodic_x, fp.periodic_y, Nx, Ny, j,
    )
    # `invden` is zero at a target the strategy skips.
    invden = fp.invden
    @inbounds @simd for i in 1:Nx
        oc[i] *= invden[i, j]
    end
    return out
end
