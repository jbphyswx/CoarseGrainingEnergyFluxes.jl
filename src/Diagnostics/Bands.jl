# ---------------------------------------------------------------------------
# Scale-band energy decomposition (Aluie & Eyink 2009 App. 2; Germano 1992 eq. 33)
# ---------------------------------------------------------------------------

"""
    band_energies(u, v, w, grid, kernel, scales; maps=false, filter_plans=nothing, execution=AutoBackend(), mask_strategy=ZeroFill(), method=nothing, vector_filter=nothing, transform=AutoSpectralBackend())
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

`maps = true` also returns the `N` per-band maps in `band_maps`, which is otherwise `nothing`;
`resolved_map` is always returned. `filter_plans` accepts a prebuilt sweep family. The maps live in the
memory of the plans' backend, whose loops compute them.

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
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scales::AbstractVector;
    maps::Bool = false,
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
    gsz = FlowGeometries.Grids.size_tuple(grid)
    has_w = w !== nothing
    total_area = active_area(grid)

    strategy = filter_plans === nothing ? mask_strategy : Filtering.plan_strategy(first(filter_plans))
    vf = Filtering.resolve_vector_filter(grid, strategy, vector_filter)
    spin = vf isa Filtering.GeneralizedConvolution
    plans = filter_plans !== nothing || spin ? filter_plans :
        Filtering.plan_filter_sweep(grid, kernel, scales;
                                    mask_strategy = mask_strategy, execution = execution, method = method,
                                    transform = transform)
    backend = plans === nothing ? execution : Filtering.plan_execution(plans[1])
    at = Placement(grid; execution = backend)
    b = at.execution

    z() = _zeros(at, T, gsz)
    band_maps = maps ? [z() for _ in eachindex(scales)] : nothing
    scratch_map = maps ? nothing : z()
    bands = zeros(T, length(scales))

    # `f` is the running repeatedly-filtered field in local components. Each band filters it and
    # `|f|²`, and the band energy is the second moment `(|f|²)‾ − |f̄|²`.
    f = (z(), z(), z())
    copyto!(f[1], u)
    copyto!(f[2], v)
    has_w && copyto!(f[3], w)
    loc = has_w ? f : (f[1], f[2])

    if spin
        sf = _spin_of(_spin_filter(grid, at, vf, transform, SpectralBackends.AutoSpectralBackend()), strategy)
        _spin_bands!(bands, band_maps, scratch_map, f, w, sf, kernel, scales, at.unmasked, total_area, b)
        return _band_result(bands, band_maps, loc, FlowGeometries.Grids.mask(at.unmasked), at.unmasked,
                            total_area, b)
    end

    # Band 1 filters `u`; every later band filters the previous band's output, defined on `og`.
    og_host = output_grid(grid, plans[1])
    og = _output(at, plans[1])
    dplans = _same_cells(grid, og_host) ? plans :
        Filtering.plan_filter_sweep(og_host, kernel, scales; mask_strategy = Filtering.ZeroFill(),
                                    execution = backend, method = Filtering.plan_method(plans[1]),
                                    transform = transform)
    mask = FlowGeometries.Grids.mask(og)

    if !(G <: FlowGeometries.Geometry.CartesianGeometry)
        # The ambient Cartesian components are filtered (Aluie 2019) and the next `f` is the local
        # part of the result, as in `compute_Π!`. `|f|²` is a scalar and is filtered as one.
        P = (z(), z(), z())
        GP = (z(), z(), z())
        sq = z()
        for n in eachindex(scales)
            km = maps ? band_maps[n] : scratch_map
            _to_ambient!(P, f[1], f[2], has_w ? f[3] : nothing, n == 1 ? at.grid : og, b)
            @. sq = P[1]^2 + P[2]^2 + P[3]^2
            Filtering.filter_apply_batch!(
                (GP[1], GP[2], GP[3], km), (P[1], P[2], P[3], sq), n == 1 ? plans[1] : dplans[n],
            )
            _ambient_to_local!(loc, GP, og, b)
            for c in eachindex(loc)
                fc = loc[c]
                @. km -= fc * fc
            end
            @. km = ifelse(mask, T(0.5) * km, zero(T))
            bands[n] = _area_mean(km, og, total_area, b)
        end
        return _band_result(bands, band_maps, loc, mask, og, total_area, b)
    end

    # `nxt` receives each next application.
    nxt = (z(), z(), z())
    # One batched apply per band filters every component of `f` and its square.
    nc = has_w ? 3 : 2
    sqs = (z(), z(), z())
    fsqs = (z(), z(), z())

    for n in eachindex(scales)
        km = maps ? band_maps[n] : scratch_map
        fill!(km, zero(T))
        for c in 1:nc
            @. sqs[c] = f[c] * f[c]
        end
        outs = nc == 2 ? (nxt[1], nxt[2], fsqs[1], fsqs[2]) :
                         (nxt[1], nxt[2], nxt[3], fsqs[1], fsqs[2], fsqs[3])
        srcs = nc == 2 ? (f[1], f[2], sqs[1], sqs[2]) :
                         (f[1], f[2], f[3], sqs[1], sqs[2], sqs[3])
        Filtering.filter_apply_batch!(outs, srcs, n == 1 ? plans[1] : dplans[n])
        for c in 1:nc
            # τ(f;f) = (f²)‾ − (f̄)², summed over components; the ½ is applied once at the end.
            @. km += fsqs[c] - nxt[c] * nxt[c]
        end
        @. km = ifelse(mask, T(0.5) * km, zero(T))
        bands[n] = _area_mean(km, og, total_area, b)
        for c in 1:nc
            copyto!(f[c], nxt[c])
        end
    end
    return _band_result(bands, band_maps, loc, mask, og, total_area, b)
end

# The band energies under the generalized convolution: per band, `f` by spin 1 and `|f|²` by spin 0, with
# the radial component `f_r` (when `w` is given) as the imaginary part of the spin-0 field. Band 1 reads
# `u` on the active cells; every later band reads the previous output at every cell of `og`.
function _spin_bands!(bands, band_maps, scratch_map, f, w, sf::SpinFilter, kernel, scales, og,
                      total_area::T, b) where {T}
    C1, C0 = sf.coeffs[1], sf.coeffs[2]
    f1, f2, f3 = f
    for n in eachindex(scales)
        km = band_maps === nothing ? scratch_map : band_maps[n]
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
function _band_result(bands, band_maps, loc, mask, grid, total_area::T, b) where {T}
    resolved_map = zero(first(loc))
    for fc in loc
        @. resolved_map += fc * fc
    end
    @. resolved_map = ifelse(mask, T(0.5) * resolved_map, zero(T))
    resolved = _area_mean(resolved_map, grid, total_area, b)
    return (; bands, resolved, total = sum(bands) + resolved, band_maps, resolved_map)
end

band_energies(u, v, grid::FlowGeometries.Grids.AbstractGrid, kernel, scales; kwargs...) =
    band_energies(u, v, nothing, grid, kernel, scales; kwargs...)
