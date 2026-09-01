module CoarseGrainingEnergyFluxesFastSphericalHarmonicsExt

using FastSphericalHarmonics: FastSphericalHarmonics as FSH
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF
using FlowGeometries: FlowGeometries

# Spectral filtering for uniform spherical grids, via the scalar spherical-harmonic transform. The
# wavenumber of degree `l` is the Laplace–Beltrami eigenvalue `k_l = √(l(l+1))/R`, so a degree-`l`
# coefficient is scaled by `Ĝ(k_l, ℓ)`; `Ĝ(0) = 1` preserves the mean.
#
# The FSH grid is `N` colatitudes × `M = 2N−1` longitudes sampled as `F[θ, φ]`, where this package
# stores fields as `[lon, lat]` — hence the transpose in and out. The grid must be built on the FSH
# points (`FastSphericalHarmonics.sph_points(N)`): `θ_j = π(j−½)/N`, `φ_k = 2π(k−1)/M`.
#
# Masking follows the same normalized-convolution identity as the other spectral backends, with the
# `Deformable` denominator computed once at plan-build time.

"""
    SHTGridPlan

The half of a spherical-harmonic plan the filter scale does not reach: the `SphPlanCache` holding the
transform's internal FFT plans, the mask, and the grid's validated mode limits.

Only `mult` — the per-degree transfer multiplier — and the `Deformable` renormalization computed
through it depend on ℓ. Sharing the rest across a sweep matters more here than for a plain FFT: a
fresh `SphPlanCache` per scale means the transform rebuilds its internal plans the first time it sees
each one, and the node validation below re-derives and re-compares the quadrature nodes every time.

The cache is a memo table the transform populates on first use, so unlike the other spectral grid
plans this one is written to during an apply. FastSphericalHarmonics' underlying transform is itself
restricted to the root task, so a concurrent driver needs its own plan per worker in any case.
"""
struct SHTGridPlan{T<:AbstractFloat, MK} <: CGEF.Filtering.AbstractGridPlan
    cache::FSH.SphPlanCache{T}
    N::Int
    M::Int
    lmax::Int
    mmax::Int
    radius::T
    mask::MK
end

"""
    SHTScratch

The transient half of a spherical-harmonic plan: the `N × M` buffer carrying the [lon,lat] ↔ FSH [θ,φ]
transpose and, in place, the coefficients and the evaluated points; plus the `M × N` `mask · field`
staging array. One per concurrent worker.

The transpose goes through `permutedims!` into this buffer, so no `permutedims` allocation happens per
call. It is a concrete `Array` in practice — FSH's `sph_transform!`/`sph_evaluate!` require
`Array{T,2}` — while the field type itself stays free.
"""
struct SHTScratch{T<:AbstractFloat, S<:AbstractMatrix{T}, MV<:AbstractMatrix{T}} <:
       CGEF.Filtering.AbstractFilterScratch
    scratch::S        # N × M, FSH layout
    masked_input::MV  # M × N [lon,lat]; unused when mask === nothing
end

"""
    SHTFilterPlan

Cached spherical-harmonic filter plan: the per-coefficient transfer multiplier `Ĝ(k_l, ℓ)` laid out on
the FSH coefficient array (a function of degree `l` alone), the shared [`SHTGridPlan`](@ref), the
`Deformable` inverse local-mass renormalization (or `nothing`), and the [`SHTScratch`](@ref) the apply
transposes through. Built by
`plan_filter(spherical_structured_grid, kernel, scale; method = Spectral())`.
"""
struct SHTFilterPlan{
    T<:AbstractFloat, A<:AbstractMatrix{T}, GP<:SHTGridPlan{T}, R, SC<:SHTScratch{T},
} <: CGEF.Filtering.AbstractFilterPlan
    mult::A        # N × M multiplier on the spherical-harmonic coefficients
    grid_plan::GP
    invrenorm::R   # precomputed 1/filter(mask) for Deformable, or nothing (ZeroFill / fully active)
    scratch::SC
end

function _sht_grid_plan(
    grid::FlowGeometries.Grids.StructuredGrid{T,G},
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.SphericalGeometry{T}}
    M, N = size(FlowGeometries.Grids.mask(grid))   # CGEF layout is [x, y] = [longitude, latitude] = [M, N]
    M == 2N - 1 || throw(ArgumentError(
        "Spherical-harmonic filtering needs a FastSphericalHarmonics grid with M = 2N-1 longitudes " *
        "per N latitudes (got N=$N lat, M=$M lon); build the grid on `sph_points(N)`.",
    ))
    # Shape alone doesn't prove the grid sits on the actual FSH quadrature nodes — a shape-correct but
    # wrong-node grid would silently produce a meaningless transform. Check the node values themselves.
    Θ, Φ = FSH.sph_points(N)
    isapprox(FlowGeometries.Grids.coordinates(grid, 2), T(π) / 2 .- Θ; atol = 10 * eps(T)) || throw(ArgumentError(
        "The grid's latitude coordinates do not match the FastSphericalHarmonics quadrature nodes for N=$N " *
        "(expected θ = π(j-½)/N via `sph_points(N)`, lat = π/2 - θ); build the grid on `sph_points(N)`.",
    ))
    isapprox(FlowGeometries.Grids.coordinates(grid, 1), T.(Φ); atol = 10 * eps(T)) || throw(ArgumentError(
        "The grid's longitude coordinates do not match the FastSphericalHarmonics quadrature nodes for M=$M " *
        "(expected φ = 2π(k-1)/M via `sph_points(N)`); build the grid on `sph_points(N)`.",
    ))
    return SHTGridPlan(
        FSH.SphPlanCache{T}(), N, M, N - 1, M ÷ 2,
        T(FlowGeometries.Geometry.radius(FlowGeometries.Grids.grid_geometry(grid))),
        all(FlowGeometries.Grids.mask(grid)) ? nothing : FlowGeometries.Grids.mask(grid),
    )
end

# FSH works in [θ,φ] (N×M) and this package stores [lon,lat] (M×N), so the scratch carries one of each.
_sht_scratch(gp::SHTGridPlan{T}) where {T<:AbstractFloat} =
    SHTScratch(zeros(T, gp.N, gp.M), zeros(T, gp.M, gp.N))

CGEF.Filtering.spectral_grid_plan(
    ::Union{CGEF.SpectralBackends.AbstractAutoSpectralBackend, CGEF.SpectralBackends.AbstractFSHTSpectralBackend},
    grid::FlowGeometries.Grids.StructuredGrid{T,G},
    kernel::CGEF.Kernels.AbstractFilterKernel;
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.SphericalGeometry{T}} = _sht_grid_plan(grid)

CGEF.Filtering.spectral_scratch(gp::SHTGridPlan) = _sht_scratch(gp)

function CGEF.Filtering.spectral_filter_plan(
    ::Union{CGEF.SpectralBackends.AbstractAutoSpectralBackend, CGEF.SpectralBackends.AbstractFSHTSpectralBackend},
    grid::FlowGeometries.Grids.StructuredGrid{T,G},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy = CGEF.Filtering.ZeroFill(),
    backend = CGEF.ComputationalBackends.AutoBackend(),
    grid_plan::Union{Nothing,SHTGridPlan} = nothing,
    scratch::Union{Nothing,SHTScratch} = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.SphericalGeometry{T}}
    gp = grid_plan === nothing ? _sht_grid_plan(grid) : grid_plan
    sc = scratch === nothing ? _sht_scratch(gp) : scratch
    N, M, R = gp.N, gp.M, gp.radius
    cache = gp.cache

    # Mirror FastSphericalHarmonics' own coefficient-iteration (see `sph_laplace!`): the packed layout
    # stores degrees up to lmax + mmax for high |m|. The transfer value depends only on l, not m, so
    # compute it once per degree (not once per (l,m) pair, ~2l+1 times more calls) — negligible for
    # Gaussian/SharpSpectral but genuinely wasteful for TopHatKernel's Legendre-recurrence evaluation.
    lmax, mmax = gp.lmax, gp.mmax
    lmax_full = lmax + mmax
    transfer_by_l = [CGEF.Kernels.spectral_transfer_degree(kernel, l, scale, R) for l in 0:lmax_full]
    mult = ones(T, N, M)
    for l in 0:lmax_full, m in (-l):l
        if l - lmax <= abs(m) <= mmax
            mult[FSH.sph_mode(l, m)] = transfer_by_l[l+1]
        end
    end

    mask = gp.mask
    invrenorm = if mask !== nothing && mask_strategy isa CGEF.Filtering.Deformable
        # Local kernel mass over active points, `filter(mask)`, through the plan's own
        # transform/multiply/inverse pipeline. The mask is fixed for the plan, so this is built once
        # and stored inverted.
        sc.masked_input .= mask                         # [lon,lat] (M×N)
        permutedims!(sc.scratch, sc.masked_input, (2, 1))  # → FSH [θ,φ] (N×M)
        FSH.sph_transform!(sc.scratch; cache = cache)
        sc.scratch .*= mult
        FSH.sph_evaluate!(sc.scratch; cache = cache)
        renorm = zeros(T, M, N)
        permutedims!(renorm, sc.scratch, (2, 1))        # back to [lon,lat]
        threshold = T(0.01)
        ir = similar(renorm)
        @. ir = ifelse(abs(renorm) >= threshold, one(T) / renorm, zero(T))
        ir
    else
        nothing   # ZeroFill: already exactly `filter(mask .* field)`, no renormalization
    end
    return SHTFilterPlan(mult, gp, invrenorm, sc)
end

# The forward harmonic transform depends on the field alone, so a sweep runs it once and each scale only
# scales the coefficients by its own `Ĝ(k_l, ℓ)` and evaluates back to points.
CGEF.Filtering.analyze_buffer(plan::SHTFilterPlan, ::AbstractMatrix) = similar(plan.scratch.scratch)

function CGEF.Filtering.filter_analyze!(
    Ĉ::AbstractMatrix{T}, field::AbstractMatrix{T}, plan::SHTFilterPlan{T},
) where {T<:AbstractFloat}
    gp, sc = plan.grid_plan, plan.scratch
    if gp.mask === nothing
        permutedims!(Ĉ, field, (2, 1))
    else
        @. sc.masked_input = gp.mask * field
        permutedims!(Ĉ, sc.masked_input, (2, 1))
    end
    FSH.sph_transform!(Ĉ; cache = gp.cache)
    return Ĉ
end

function CGEF.Filtering.filter_synthesize!(
    out::AbstractMatrix{T}, Ĉ::AbstractMatrix{T}, plan::SHTFilterPlan{T},
) where {T<:AbstractFloat}
    # `sph_evaluate!` works in place, and `Ĉ` is reused by every later scale, so evaluate a scaled copy.
    sc = plan.scratch
    sc.scratch .= Ĉ .* plan.mult
    FSH.sph_evaluate!(sc.scratch; cache = plan.grid_plan.cache)
    permutedims!(out, sc.scratch, (2, 1))
    plan.invrenorm === nothing || (out .*= plan.invrenorm)
    return out
end

function CGEF.Filtering.filter_apply!(
    out::AbstractMatrix{T},
    field::AbstractMatrix{T},
    plan::SHTFilterPlan{T},
) where {T<:AbstractFloat}
    gp, sc = plan.grid_plan, plan.scratch
    if gp.mask === nothing
        permutedims!(sc.scratch, field, (2, 1))     # [lon, lat] (M×N) → FSH [θ, φ] (N×M), in place
    else
        @. sc.masked_input = gp.mask * field
        permutedims!(sc.scratch, sc.masked_input, (2, 1))
    end
    FSH.sph_transform!(sc.scratch; cache = gp.cache)   # in place: scratch now holds coefficients
    sc.scratch .*= plan.mult                        # Ĝ(k_l, ℓ) per coefficient
    FSH.sph_evaluate!(sc.scratch; cache = gp.cache)    # in place: scratch now holds point values
    permutedims!(out, sc.scratch, (2, 1))           # back to [lon, lat]
    plan.invrenorm === nothing || (out .*= plan.invrenorm)
    return out
end

end # module
