module CoarseGrainingEnergyFluxesFINUFFTExt

using FINUFFT: FINUFFT
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF
using FlowGeometries: FlowGeometries

# Spectral filtering for scattered Cartesian data (an `UnstructuredGrid{Cartesian}`) by nonuniform FFT:
# a quadrature estimate of the field's Fourier coefficients on a box of periods `(Lx, Ly)`, a multiply by
# the shared transfer function `Ĝ(|k|, ℓ)` (`CGEF.Kernels.spectral_transfer`), and the series evaluated
# back at the points:
#
#   type-1 (pts→modes, isign −1):  F_k = Σ_j w_j c_j e^{-i k·x_j},   w_j = A_j / (Lx·Ly)
#   multiply:                      F_k ← Ĝ(|k|, ℓ) F_k
#   type-2 (modes→pts, isign +1):  ḡ_j = Σ_k F_k e^{+i k·x_j}
#
# `A_j` is the grid's measure of point `j`, so `F_k` is the quadrature rule for the box's Fourier
# coefficient of `c`. On a uniform lattice the result is the FFTW one on the same grid, and on a periodic
# box the cells tile, a constant is kept.
#
# The filter is the circular convolution over the box. A direction the grid declares periodic takes the
# grid's period. Any other is padded as FFTW pads a bounded axis: the record, the points' extent plus
# one spacing (`N·Δx` on a uniform axis), holds `M` modes, and the box holds `2·nextprod((2,3,5), M)` at
# the same mode spacing. The field is zero beyond the record, and the wrap-around path between two
# points is at least the record long.
#
# Masking (Knutsson & Westin 1993): `ZeroFill` filters `mask·field`; `Deformable` additionally divides
# by `filter(mask)`, run through the same pipeline once per plan. A masked point contributes exactly
# zero, whatever the field holds there.

"""
    FINUFFTGridPlan

The half of a FINUFFT plan the filter scale does not reach: the persistent guru pair (type-1
points→modes, type-2 modes→points) with `finufft_setpts!` already called, the rescaled point
coordinates, the mode counts, the box periods, the quadrature weights and the mask. Only `transfer`
and the `Deformable` renormalization depend on ℓ, so the point sort runs once per grid plan.

A guru plan holds the working state of its own execution, so two tasks may not execute one
concurrently; a concurrent driver needs its own grid plan per worker.
"""
struct FINUFFTGridPlan{T<:AbstractFloat, VT<:AbstractVector{T}, P1, P2, MK} <: CGEF.Filtering.AbstractGridPlan
    X::VT        # x points scaled to [0, 2π)
    Y::VT        # y points scaled to [0, 2π)
    M::Int
    N::Int
    npts::Int
    Lx::T
    Ly::T
    plan1::P1
    plan2::P2
    weights::VT  # A_j/(Lx·Ly)
    mask::MK     # the grid's mask, or nothing when fully active
end

"""
    FINUFFTScratch

The transient half of a FINUFFT plan: the length-`npts` coefficient buffer (type-1 input, then type-2
output) and the `M × N` mode buffer. One per concurrent worker.
"""
struct FINUFFTScratch{T<:AbstractFloat, CV<:AbstractVector{Complex{T}}, CM<:AbstractMatrix{Complex{T}}} <:
       CGEF.Filtering.AbstractFilterScratch
    c_scratch::CV
    F_scratch::CM
end

"""
    FINUFFTFilterPlan

Cached scattered-data spectral filter plan: the shared [`FINUFFTGridPlan`](@ref), the precomputed
transfer-function array on the `M × N` Fourier modes, the `Deformable` inverse local-mass
renormalization (or `nothing`), and the [`FINUFFTScratch`](@ref) the apply writes through. Built by
`plan_filter(unstructured_grid, kernel, scale; method = Spectral())`.
"""
struct FINUFFTFilterPlan{
    T<:AbstractFloat, GP<:FINUFFTGridPlan{T}, A<:AbstractMatrix{T}, R, SC<:FINUFFTScratch{T},
} <: CGEF.Filtering.AbstractFilterPlan
    grid_plan::GP
    transfer::A    # Ĝ(|k|, ℓ) on the M × N CMCL-ordered mode grid
    invrenorm::R   # precomputed 1/filter(mask) for Deformable, or nothing (ZeroFill / fully active)
    scratch::SC
end

function _finufft_grid_plan(
    grid::FlowGeometries.Grids.UnstructuredGrid{T,G}; finufft_nthreads::Integer = 1,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    nth = Int(finufft_nthreads)
    nth >= 0 || throw(ArgumentError(
        "finufft_nthreads must be >= 0 (0 means FINUFFT's own default of all threads), got $nth",
    ))
    x = FlowGeometries.Grids.coordinates(grid, 1)
    y = FlowGeometries.Grids.coordinates(grid, 2)
    npts = length(x)
    npts > 0 || throw(ArgumentError("FINUFFT spectral filtering needs at least one point."))
    xmin, xmax = extrema(x)
    ymin, ymax = extrema(y)
    px = FlowGeometries.Grids.isperiodic(grid, 1)
    py = FlowGeometries.Grids.isperiodic(grid, 2)
    spanx = px ? T(FlowGeometries.Grids.period(grid, 1)) : xmax - xmin
    spany = py ? T(FlowGeometries.Grids.period(grid, 2)) : ymax - ymin

    # Mode count from the point count, `M·N ≈ npts`, split by the aspect ratio.
    aspect = spany > 0 ? spanx / spany : one(T)
    My_est = sqrt(T(npts) / aspect)
    Mx_est = T(npts) / My_est
    Mr = max(2, round(Int, Mx_est)); iseven(Mr) || (Mr += 1)
    Nr = max(2, round(Int, My_est)); iseven(Nr) || (Nr += 1)

    # A record's spacing is its extent over the rounded mode count less one, so a uniform lattice of
    # `M` points gets exactly `M·Δx`; a bounded direction's box extends it at that spacing.
    M = px ? Mr : 2 * nextprod((2, 3, 5), Mr)
    N = py ? Nr : 2 * nextprod((2, 3, 5), Nr)
    Lx = px ? spanx : M * spanx / (Mr - 1)
    Ly = py ? spany : N * spany / (Nr - 1)
    (Lx > 0 && Ly > 0) || throw(ArgumentError(
        "FINUFFT spectral filtering needs a box of positive size; the points give ($Lx, $Ly). A point set " *
        "with no extent in a direction needs `periodic` and `period` declared for it.",
    ))
    X = T(2π) .* mod.(x .- xmin, Lx) ./ Lx
    Y = T(2π) .* mod.(y .- ymin, Ly) ./ Ly
    ϵ = max(T(1e-9), eps(T) * 10)

    # `ntrans = 1`: the batch axis is the slice/batch drivers'. `nthreads` is FINUFFT's internal
    # parallelism, its spreader and its own FFTW plan.
    plan1 = FINUFFT.finufft_makeplan(1, [M, N], -1, 1, ϵ; dtype = T, nthreads = nth)
    plan2 = FINUFFT.finufft_makeplan(2, [M, N], 1, 1, ϵ; dtype = T, nthreads = nth)
    FINUFFT.finufft_setpts!(plan1, X, Y)
    FINUFFT.finufft_setpts!(plan2, X, Y)
    finalizer(FINUFFT.finufft_destroy!, plan1)
    finalizer(FINUFFT.finufft_destroy!, plan2)

    A = FlowGeometries.Grids.measure(grid)
    Asum = sum(A)
    Asum > 0 || throw(ArgumentError(
        "FINUFFT spectral filtering weights each point by the grid's measure, which sums to $Asum here.",
    ))
    weights = T.(A ./ (Lx * Ly))
    mask = all(FlowGeometries.Grids.mask(grid)) ? nothing : FlowGeometries.Grids.mask(grid)

    return FINUFFTGridPlan(X, Y, M, N, npts, Lx, Ly, plan1, plan2, weights, mask)
end

_finufft_scratch(gp::FINUFFTGridPlan{T}) where {T<:AbstractFloat} =
    FINUFFTScratch(zeros(Complex{T}, gp.npts), zeros(Complex{T}, gp.M, gp.N))

# `w .* field`, or `mask .* (w .* field)`: a `Bool` is a strong zero, so a masked point adds nothing
# even where the field is NaN.
function _load_weighted!(c, field, gp::FINUFFTGridPlan{T}) where {T}
    if gp.mask === nothing
        @. c = Complex{T}(gp.weights * field)
    else
        @. c = Complex{T}(gp.mask * (gp.weights * field))
    end
    return c
end

CGEF.Filtering.spectral_scratch(gp::FINUFFTGridPlan) = _finufft_scratch(gp)

CGEF.Filtering.spectral_grid_plan(
    ::Union{CGEF.SpectralBackends.AbstractAutoSpectralBackend, CGEF.SpectralBackends.AbstractNUFFTSpectralBackend},
    grid::FlowGeometries.Grids.UnstructuredGrid{T,G},
    kernel::CGEF.Kernels.AbstractFilterKernel;
    finufft_nthreads::Integer = 1,
    kwargs...,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}} =
    _finufft_grid_plan(grid; finufft_nthreads = finufft_nthreads)

"""
    spectral_filter_plan(...; finufft_nthreads = 1)

`finufft_nthreads` is FINUFFT's internal thread count, used by its spreader and its own FFTW plan; the
batch axis is the slice/batch drivers'.

The internal FFTW plan takes FFTW's process-global thread count, which other packages raise, and a
multi-threaded FFTW plan spawns Julia tasks on every execution, each of which allocates. The default of
1 keeps a reused plan allocation-free. `finufft_nthreads = 0` (all threads) suits one large transform;
inside a threaded slice/batch loop it nests two levels of threading.
"""
function CGEF.Filtering.spectral_filter_plan(
    ::Union{CGEF.SpectralBackends.AbstractAutoSpectralBackend, CGEF.SpectralBackends.AbstractNUFFTSpectralBackend},
    grid::FlowGeometries.Grids.UnstructuredGrid{T,G},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy = CGEF.Filtering.ZeroFill(),
    backend = CGEF.ComputationalBackends.AutoBackend(),
    finufft_nthreads::Integer = 1,
    grid_plan::Union{Nothing,FINUFFTGridPlan} = nothing,
    scratch::Union{Nothing,FINUFFTScratch} = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    gp = grid_plan === nothing ?
        _finufft_grid_plan(grid; finufft_nthreads = finufft_nthreads) : grid_plan
    sc = scratch === nothing ? _finufft_scratch(gp) : scratch
    M, N = gp.M, gp.N
    Lx, Ly = gp.Lx, gp.Ly

    nx = (-(M ÷ 2)):(M ÷ 2 - 1)
    ny = (-(N ÷ 2)):(N ÷ 2 - 1)
    transfer = T[
        CGEF.Kernels.spectral_transfer(kernel, sqrt((T(2π) * ix / Lx)^2 + (T(2π) * iy / Ly)^2), scale)
        for ix in nx, iy in ny
    ]

    bounded = !(FlowGeometries.Grids.isperiodic(grid, 1) && FlowGeometries.Grids.isperiodic(grid, 2))
    invrenorm = if mask_strategy isa CGEF.Filtering.Deformable && (gp.mask !== nothing || bounded)
        # `filter(mask)`, through the plan's own pipeline; the box beyond a bounded record is inactive.
        if gp.mask === nothing
            @. sc.c_scratch = Complex{T}(gp.weights)
        else
            @. sc.c_scratch = Complex{T}(gp.mask * gp.weights)
        end
        FINUFFT.finufft_exec!(gp.plan1, sc.c_scratch, sc.F_scratch)
        sc.F_scratch .*= transfer
        FINUFFT.finufft_exec!(gp.plan2, sc.F_scratch, sc.c_scratch)
        renorm = real.(sc.c_scratch)
        threshold = T(0.01)
        ir = similar(renorm)
        @. ir = ifelse(abs(renorm) >= threshold, one(T) / renorm, zero(T))
        ir
    else
        nothing   # ZeroFill: already exactly `filter(mask .* field)`, no renormalization
    end
    return FINUFFTFilterPlan(gp, transfer, invrenorm, sc)
end

# Analysis (pts → modes) depends on the field alone, so a sweep runs it once and each scale only applies
# its own transfer function and evaluates back to points.
CGEF.Filtering.analyze_buffer(plan::FINUFFTFilterPlan, ::AbstractVector) = similar(plan.scratch.F_scratch)

function CGEF.Filtering.filter_analyze!(
    F̂::AbstractArray, field::AbstractVector, plan::FINUFFTFilterPlan{T},
) where {T<:AbstractFloat}
    gp, sc = plan.grid_plan, plan.scratch
    _load_weighted!(sc.c_scratch, field, gp)
    FINUFFT.finufft_exec!(gp.plan1, sc.c_scratch, F̂)
    return F̂
end

function CGEF.Filtering.filter_synthesize!(
    out::AbstractVector{T}, F̂::AbstractArray, plan::FINUFFTFilterPlan{T},
) where {T<:AbstractFloat}
    # `finufft_exec!` consumes its input, and `F̂` is reused by every later scale, so the scaled copy goes
    # through the plan's own mode scratch.
    gp, sc = plan.grid_plan, plan.scratch
    sc.F_scratch .= F̂ .* plan.transfer
    FINUFFT.finufft_exec!(gp.plan2, sc.F_scratch, sc.c_scratch)
    @. out = real(sc.c_scratch)
    plan.invrenorm === nothing || (out .*= plan.invrenorm)
    return out
end

function CGEF.Filtering.filter_apply!(
    out::AbstractVector{T},
    field::AbstractVector,
    plan::FINUFFTFilterPlan{T},
) where {T<:AbstractFloat}
    gp, sc = plan.grid_plan, plan.scratch
    _load_weighted!(sc.c_scratch, field, gp)
    FINUFFT.finufft_exec!(gp.plan1, sc.c_scratch, sc.F_scratch)   # pts → modes
    sc.F_scratch .*= plan.transfer                                # Ĝ · F̂
    FINUFFT.finufft_exec!(gp.plan2, sc.F_scratch, sc.c_scratch)   # modes → pts (reuses c_scratch)
    @. out = real(sc.c_scratch)
    plan.invrenorm === nothing || (out .*= plan.invrenorm)
    return out
end

end # module
