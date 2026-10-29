# ---------------------------------------------------------------------------
# Scale-band energy decomposition (Aluie & Eyink 2009 App. 2; Germano 1992 eq. 33)
# ---------------------------------------------------------------------------

"""
    BandMaps()

Asks [`band_energies`](@ref) for the map of every band as well as its average.
"""
struct BandMaps end

"""
    band_energies(u, v, w, grid, kernel, scales; maps=nothing, filter_plans=nothing, execution=AutoBackend(), mask_strategy=ZeroFill(), method=nothing, vector_filter=nothing, transform=AutoSpectralBackend())
        -> (; bands, resolved, total, band_maps, resolved_map)

Split the kinetic energy into contributions from each scale band, using the repeated-filter
generalization of the Germano identity.

With `scales` in ascending order `ℓ₁ < ℓ₂ < … < ℓ_N`, define the repeatedly filtered fields

```
f₀ = u ,   f_n = G_{ℓ_n} * f_{n-1} ,
```

so `f_n` has had every scale below `ℓ_n` removed, successively. Band `n` holds the energy the `n`-th
application removed, which is the generalized second moment at that level:

```
k_n = ½ τ_{ℓ_n}(f_{n-1}; f_{n-1}) = ½[ (|f_{n-1}|²)‾_{ℓ_n} − |f_n|² ] ,
```

and the decomposition is exact:

```
½⟨|u|²⟩ = Σ_{n=1}^N ⟨k_n⟩ + ½⟨|f_N|²⟩ .
```

The sum telescopes because each `⟨G * x⟩ = ⟨x⟩`: the filter conserves the integral over the grid its
output is read on. Under `ZeroFill` every cell of that grid is summed, land included, and divided by
the water area, so on a periodic grid, masked or not, the identity holds to the accuracy the discrete
kernel conserves its mass. Past a bounded edge the filtered field leaves the grid, and the identity
carries a residual, `½ Σₙ (⟨(|f_{n-1}|²)‾⟩ − ⟨|f_{n-1}|²⟩)`, of order `ℓ/L`. `Deformable` renormalizes
over the active cells, which does not conserve the integral.

On a grid of non-Cartesian geometry `f_n` is the filtered velocity [`compute_Π!`](@ref) uses. On the
surface of a sphere `vector_filter` ([`Filtering.resolve_vector_filter`](@ref)) filters it: by default
the generalized convolution of Aluie (2019), which multiplies the degree-`n` part of `f` and of the
scalar `|f|²` by the same `Ĝ(n)`, its spin transforms chosen by `transform`; it reads no scalar filter
plan, and none is built for it. `Filtering.CartesianComponents()`, and every spheroid, takes the local
part of the filtered ambient Cartesian components of `f_{n-1}` and filters `|f|²` as a scalar.

The second-moment form has no cross terms between bands (Aluie & Eyink 2009). Under a scalar or
Cartesian-component convolution `k_n ≥ 0` pointwise iff the kernel is non-negative: use a positive
kernel (`TopHatKernel`, `GaussianKernel`, `SmoothHatKernel`, `HyperGaussianKernel`); a signed one such
as [`Kernels.HighOrderKernel`](@ref) can give negative band energies. The generalized convolution
filters the parts of `f` with the spectrally shifted kernels of its eqs. 59–60, which may take either
sign, so its map `k_n` may too. On the whole sphere its average is proportional to
`Σ_l (1 − Ĝ(l)²)|f̂_l|²` over the degree-`l` parts `f̂_l` of `f_{n-1}`, non-negative when every
`|Ĝ(l)| ≤ 1`, which a non-negative kernel meets.

`maps = BandMaps()` also returns the `N` per-band maps in `band_maps`, which is otherwise `nothing`;
`resolved_map` is always returned. `filter_plans` accepts a prebuilt sweep family. The maps live in the
memory of the plans' backend, whose loops compute them. A [`BandWorkspace`](@ref) holds the plans,
transforms and buffers for repeated calls on one grid, through [`band_energies!`](@ref).

Returns the per-band domain-averaged energies `bands` (length `N`), the energy left in `f_N`
(`resolved`), their sum `total`, and the corresponding pointwise maps.

# References
- Germano, M. (1992). *J. Fluid Mech.* 238, eq. (33).
- Aluie, H., & Eyink, G. L. (2009). Localness of energy cascade in hydrodynamic turbulence.
  *Phys. Fluids* 21, 115108, Appendix 2.
"""
function band_energies(
    u::AbstractArray,
    v::AbstractArray,
    w::Union{Nothing, AbstractArray},
    grid::FlowGeometries.Grids.AbstractGrid,
    kernel::Kernels.AbstractFilterKernel,
    scales::AbstractVector;
    kwargs...,
)
    return band_energies!(BandWorkspace(grid, kernel, scales, w; kwargs...), u, v, w)
end

"""
    BandWorkspace(grid, kernel, scales, w = nothing; maps = nothing, filter_plans = nothing, execution = AutoBackend(),
                  mask_strategy = ZeroFill(), method = nothing, vector_filter = nothing, transform = AutoSpectralBackend())

What [`band_energies!`](@ref) reads and writes on `grid` for `kernel` at `scales`, in ascending order: the
sweep's filter plans and those of the fields after the first band, or on a sphere the generalized
convolution's transforms; the placement; the running fields and their filtered squares; and the band
maps under `maps = BandMaps()`. `w` is the vertical velocity the calls will give, or `nothing`; only its
type is read. Keywords as for [`band_energies`](@ref).
"""
struct BandWorkspace{
    T<:AbstractFloat, K<:Kernels.AbstractFilterKernel, PL, DP, SF, P<:Placement, OG, A<:AbstractArray{T},
    BM, SM, VT <: AbstractVector{T}
}
    kernel::K
    scales::Vector{T}
    plans::PL           # the sweep's plans, or nothing under the generalized convolution
    dplans::DP          # the plans for every band after the first, read on `og`
    spin::SF            # the generalized convolution at the plans' strategy, or nothing
    at::P
    og::OG              # the grid band 2 onward is read on
    total_area::T
    f::NTuple{3,A}      # the running field, local components
    p1::NTuple{3,A}     # Cartesian: the next application; elsewhere the ambient components
    p2::NTuple{3,A}     # Cartesian: the squares; elsewhere the filtered ambient components
    p3::NTuple{3,A}     # Cartesian: the filtered squares; elsewhere `|f|²` in the first
    band_maps::BM
    scratch_map::SM
    resolved_map::A
    bands::VT
end

function BandWorkspace(
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scales::AbstractVector,
    w::Union{Nothing,AbstractArray} = nothing;
    maps::Union{Nothing,BandMaps} = nothing,
    filter_plans = nothing,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Union{Nothing, Filtering.AbstractFilterMethod} = nothing,
    vector_filter::Union{Nothing, Filtering.AbstractVectorFilter} = nothing,
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    isempty(scales) && throw(ArgumentError("band_energies needs at least one scale"))
    issorted(scales) || throw(ArgumentError(
        "band_energies needs `scales` in ascending order (finest first): band n is the energy the " *
        "n-th, progressively coarser, filter application removes. Got $scales.",
    ))
    strategy = filter_plans === nothing ? mask_strategy : Filtering.plan_strategy(first(filter_plans))
    vf = Filtering.resolve_vector_filter(grid, strategy, vector_filter)
    sweep() = Filtering.plan_filter_sweep(grid, kernel, scales;
                                          mask_strategy = mask_strategy, execution = execution, method = method,
                                          transform = transform)
    plans = _setup_plan(filter_plans, vf, sweep)
    backend = _plan_backend(plans, execution)
    at = Placement(grid; execution = backend)
    spin = _band_spin(vf, grid, at, transform, strategy)
    og, dplans = _band_derived(plans, grid, at, kernel, scales, backend, transform)
    z() = _zeros(at, T, FlowGeometries.Grids.size_tuple(grid))
    triple() = (z(), z(), z())
    band_maps = _band_maps(maps, z, scales)
    return BandWorkspace(
        kernel, T.(collect(scales)), plans, dplans, spin, at, og, active_area(grid), triple(), triple(), triple(),
        triple(), band_maps, _scratch_map(band_maps, z), z(), zeros(T, length(scales)),
    )
end

# The generalized convolution's filter where the vector filter is one, at the plans' strategy.
_band_spin(vf::Filtering.GeneralizedConvolution, grid, at, transform, strategy) =
    _spin_of(_spin_filter(grid, at, vf, transform, SpectralBackends.AutoSpectralBackend()), strategy)
_band_spin(_, _, _, _, _) = nothing

# The grid every band after the first is read on, and its plans: band 1 filters `u`, every later band the
# previous band's output, defined on the plans' output grid.
_band_derived(::Nothing, grid, at, _, _, _, _) = (at.unmasked, nothing)
function _band_derived(plans, grid, at, kernel, scales, backend, transform)
    og_host = output_grid(grid, plans[1])
    dplans = _same_cells(grid, og_host) ? plans :
        Filtering.plan_filter_sweep(og_host, kernel, scales; mask_strategy = Filtering.ZeroFill(),
                                    execution = backend, method = Filtering.plan_method(plans[1]),
                                    transform = transform)
    return _output(at, plans[1]), dplans
end

"""
    band_energies!(ws::BandWorkspace, u, v, w = nothing) -> (; bands, resolved, total, band_maps, resolved_map)

[`band_energies`](@ref) through `ws`, which allocates nothing per call beyond the result tuple: the
returned arrays are the workspace's own, overwritten by its next call. `w` is `nothing` or an array, as the
workspace was built for.
"""
function band_energies!(ws::BandWorkspace{T}, u::AbstractArray, v::AbstractArray, w::Union{Nothing,AbstractArray} = nothing) where {T}
    f, b = ws.f, ws.at.execution
    copyto!(f[1], u)
    copyto!(f[2], v)
    w === nothing ? fill!(f[3], zero(T)) : copyto!(f[3], w)
    loc = w === nothing ? (f[1], f[2]) : f
    _band_sweep!(ws, ws.spin, loc, w, b)
    return _band_result!(ws, loc, b)
end

function _band_sweep!(ws::BandWorkspace{T}, sf::SpinFilter, loc, w, b) where {T}
    _spin_bands!(ws.bands, ws.band_maps, ws.scratch_map, ws.f, w, sf, ws.kernel, ws.scales, ws.og, ws.total_area, b)
    return nothing
end

_band_sweep!(ws::BandWorkspace, ::Nothing, loc, w, b) =
    _band_sweep!(ws, FlowGeometries.Grids.grid_geometry(ws.og), loc, w, b)

# Band 1's plan filters `u` on the grid; every later band's filters the previous output on `og`, so the
# first band runs apart from the rest. Under the ambient Cartesian filter band 1 rotates `u` by the grid's
# frame plan, and every band rotates its output by `og`'s.
function _band_sweep!(ws::BandWorkspace, geo::FlowGeometries.Geometry.AbstractGeometry, loc, w, b)
    ofr = _output_frames(ws.at, ws.plans[1])
    _one_band!(ws, geo, 1, ws.plans[1], ws.at.frames, ofr, loc, w, b)
    for n in 2:length(ws.scales)
        _one_band!(ws, geo, n, ws.dplans[n], ofr, ofr, loc, w, b)
    end
    return nothing
end

# The ambient Cartesian components are filtered (Aluie 2019) and the next `f` is the local part of the
# result, as in `compute_Π!`. `|f|²` is a scalar and is filtered as one.
function _one_band!(ws::BandWorkspace{T}, ::FlowGeometries.Geometry.AbstractGeometry, n::Int, plan, fin, fout,
                    loc, w, b) where {T}
    f, P, GP, sq, og = ws.f, ws.p1, ws.p2, ws.p3[1], ws.og
    km = _band_map(ws.band_maps, ws.scratch_map, n)
    _to_ambient!(P, f[1], f[2], w === nothing ? nothing : f[3], fin, b)
    @. sq = P[1]^2 + P[2]^2 + P[3]^2
    Filtering.filter_apply_batch!((GP[1], GP[2], GP[3], km), (P[1], P[2], P[3], sq), plan)
    _ambient_to_local!(loc, GP, fout, b)
    for c in eachindex(loc)
        fc = loc[c]
        @. km -= fc * fc
    end
    mask = FlowGeometries.Grids.mask(og)
    @. km = ifelse(mask, T(0.5) * km, zero(T))
    ws.bands[n] = _area_mean(km, og, ws.total_area, b)
    return nothing
end

# One batched apply per band filters every component of `f` and its square.
function _one_band!(ws::BandWorkspace{T}, ::FlowGeometries.Geometry.CartesianGeometry, n::Int, plan, _fin, _fout,
                    loc, w, b) where {T}
    f, nxt, sqs, fsqs, og = ws.f, ws.p1, ws.p2, ws.p3, ws.og
    km = _band_map(ws.band_maps, ws.scratch_map, n)
    fill!(km, zero(T))
    for c in eachindex(loc)
        @. sqs[c] = f[c] * f[c]
    end
    outs = w === nothing ? (nxt[1], nxt[2], fsqs[1], fsqs[2]) : (nxt[1], nxt[2], nxt[3], fsqs[1], fsqs[2], fsqs[3])
    srcs = w === nothing ? (f[1], f[2], sqs[1], sqs[2]) : (f[1], f[2], f[3], sqs[1], sqs[2], sqs[3])
    Filtering.filter_apply_batch!(outs, srcs, plan)
    for c in eachindex(loc)
        # τ(f;f) = (f²)‾ − (f̄)², summed over components; the ½ is applied once at the end.
        @. km += fsqs[c] - nxt[c] * nxt[c]
    end
    mask = FlowGeometries.Grids.mask(og)
    @. km = ifelse(mask, T(0.5) * km, zero(T))
    ws.bands[n] = _area_mean(km, og, ws.total_area, b)
    for c in eachindex(loc)
        copyto!(f[c], nxt[c])
    end
    return nothing
end

# A map per band under `BandMaps`, none otherwise; band `n` is written into its map or the one scratch map.
_band_maps(::Nothing, _, _) = nothing
_band_maps(::BandMaps, z::F, scales) where {F} = [z() for _ in eachindex(scales)]
_scratch_map(::Nothing, z::F) where {F} = z()
_scratch_map(::AbstractVector, _) = nothing
@inline _band_map(::Nothing, scratch, _) = scratch
@inline _band_map(maps::AbstractVector, _, n::Int) = maps[n]

# The band energies under the generalized convolution: per band, `f` by spin 1 and `|f|²` by spin 0, with
# the radial component `f_r` (when `w` is given) as the imaginary part of the spin-0 field. Band 1 reads
# `u` on the active cells; every later band reads the previous output at every cell of `og`.
function _spin_bands!(bands, band_maps, scratch_map, f, w, sf::SpinFilter, kernel, scales, og,
                      total_area::T, b) where {T}
    C1, C0 = sf.coeffs[1], sf.coeffs[2]
    f1, f2, f3 = f
    for n in eachindex(scales)
        km = _band_map(band_maps, scratch_map, n)
        masked = n == 1
        _spin_transfer!(sf, kernel, scales[n])
        _spin_analyze!(C1, sf, 1, masked, k -> @inbounds(complex(f1[k], f2[k])))
        _spin_analyze!(C0, sf, 0, masked, _band_square(f1, f2, f3, w))
        _spin_synthesize!(sf, C0, 0)
        _spin_store_square!(km, f3, sf, w)
        _spin_synthesize!(sf, C1, 1)
        _spin_store!(sf) do k, z
            @inbounds f1[k] = real(z)
            @inbounds f2[k] = imag(z)
        end
        @. km = T(0.5) * (km - f1 * f1 - f2 * f2 - f3 * f3)
        bands[n] = _area_mean(km, og, total_area, b)
    end
    return bands
end

# `|f|²`, and with a radial component `f_r = f3` also `i f_r`, at cell `k`.
_band_square(f1, f2, _f3, ::Nothing) = k -> @inbounds(complex(f1[k]^2 + f2[k]^2))
_band_square(f1, f2, f3, ::AbstractArray) = k -> @inbounds(complex(f1[k]^2 + f2[k]^2 + f3[k]^2, f3[k]))

# `(|f|²)‾` into `km` and, with a radial component, `f̄_r` into `f3`.
_spin_store_square!(km, _f3, sf::SpinFilter, ::Nothing) =
    _spin_store!((k, z) -> @inbounds(km[k] = real(z)), sf)

function _spin_store_square!(km, f3, sf::SpinFilter, ::AbstractArray)
    _spin_store!(sf) do k, z
        @inbounds km[k] = real(z)
        @inbounds f3[k] = imag(z)
    end
    return nothing
end

# The resolved energy `½⟨|f_N|²⟩` of the final field and the returned named tuple.
function _band_result!(ws::BandWorkspace{T}, loc, b) where {T}
    resolved_map, og = ws.resolved_map, ws.og
    mask = FlowGeometries.Grids.mask(og)
    fill!(resolved_map, zero(T))
    for fc in loc
        @. resolved_map += fc * fc
    end
    @. resolved_map = ifelse(mask, T(0.5) * resolved_map, zero(T))
    resolved = _area_mean(resolved_map, og, ws.total_area, b)
    bands = ws.bands
    return (; bands, resolved, total = sum(bands) + resolved, band_maps = ws.band_maps, resolved_map)
end

band_energies(u, v, grid::FlowGeometries.Grids.AbstractGrid, kernel, scales; kwargs...) =
    band_energies(u, v, nothing, grid, kernel, scales; kwargs...)
