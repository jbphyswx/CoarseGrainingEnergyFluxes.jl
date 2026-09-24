# ---------------------------------------------------------------------------
# Scale-band energy decomposition (Aluie & Eyink 2009 App. 2; Germano 1992 eq. 33)
# ---------------------------------------------------------------------------

"""
    band_energies(u, v, w, grid, kernel, scales; backend=AutoBackend(), mask_strategy=ZeroFill())
        -> (; bands, resolved, total, band_maps, resolved_map)

Split the kinetic energy into contributions from each scale band, using the **repeated-filter**
generalization of the Germano identity rather than by band-passing the velocity.

With `scales` in ASCENDING order `ℓ₁ < ℓ₂ < … < ℓ_N`, define the repeatedly filtered fields

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

On a grid of non-Cartesian geometry `f_n` is the local part of the filtered planetary Cartesian
components of `f_{n-1}` (Aluie 2019) — the filtered velocity [`compute_Π!`](@ref) uses — and `|f|²`
is filtered as a scalar.

# Why not band-pass the velocity

The obvious alternative, `u = ū₀ + Σ(ū_n − ū_{n-1})`, gives
`½⟨|u|²⟩ = ½⟨|ū₀|²⟩ + ½Σ_{n,m}⟨ū_n · ū_m⟩` — cross terms of indefinite sign, so there is no
well-defined energy at a given scale at all (Aluie & Eyink 2009). The second-moment form above has no
cross terms by construction, and `k_n ≥ 0` pointwise **iff the kernel is non-negative** — so use a
positive kernel here (`TopHatKernel`, `GaussianKernel`, `SmoothHatKernel`, `HyperGaussianKernel`); a
signed one such as [`Kernels.HighOrderKernel`](@ref) can give negative band energies.

`maps = true` additionally returns the per-band and resolved MAPS; they are `N+1` full fields and
most callers reduce them straight to the scalars below, so they are not built by default and
`band_maps` is then `nothing`. `filter_plans` accepts a prebuilt sweep family.

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
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Union{Nothing, Filtering.AbstractFilterMethod} = nothing,
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    isempty(scales) && throw(ArgumentError("band_energies needs at least one scale"))
    issorted(scales) || throw(ArgumentError(
        "band_energies needs `scales` in ascending order (finest first): band n is the energy the " *
        "n-th, progressively coarser, filter application removes. Got $scales.",
    ))
    gsz = FlowGeometries.Grids.size_tuple(grid)
    has_w = w !== nothing
    total_area = active_area(grid)

    # One family for the sweep: the grid-determined half of the engine is the same at every scale.
    plans = filter_plans === nothing ?
        Filtering.plan_filter_sweep(grid, kernel, scales;
                                    mask_strategy = mask_strategy, backend = backend, method = method) :
        filter_plans
    # Band 1 filters `u`; every later band filters the previous band's output, defined on `og`.
    og = output_grid(grid, plans[1])
    dplans = _same_cells(grid, og) ? plans :
        Filtering.plan_filter_sweep(og, kernel, scales; mask_strategy = Filtering.ZeroFill(),
                                    backend = backend, method = Filtering.plan_method(plans[1]))

    # Per-band maps are `N` full fields and most callers want only the scalars they reduce to, so they
    # are opt-in. Without them one scratch map is reused for every band.
    band_maps = maps ? [zeros(T, gsz) for _ in eachindex(scales)] : nothing
    scratch_map = maps ? nothing : zeros(T, gsz)
    bands = zeros(T, length(scales))
    mask = FlowGeometries.Grids.mask(og)

    # `f` is the running repeatedly-filtered field in local components. Each band filters it and
    # `|f|²`, and the band energy is the second moment `(|f|²)‾ − |f̄|²`.
    f = (copy(u), copy(v), has_w ? copy(w) : zeros(T, gsz))
    loc = has_w ? f : (f[1], f[2])

    if !(G <: FlowGeometries.Geometry.CartesianGeometry)
        # The velocity is filtered as its planetary Cartesian components (Aluie 2019), and the next
        # field is the local part of the result: its tangent components, with the radial one when `w`
        # is given — the filtered velocity `compute_Π!` uses. `|f|²` is a scalar and is filtered as one.
        P = ntuple(_ -> zeros(T, gsz), 3)
        GP = ntuple(_ -> zeros(T, gsz), 3)
        sq = zeros(T, gsz)
        for n in eachindex(scales)
            km = maps ? band_maps[n] : scratch_map
            _fill_planetary!(P, f[1], f[2], has_w ? f[3] : nothing, n == 1 ? grid : og)
            @. sq = P[1]^2 + P[2]^2 + P[3]^2
            Filtering.filter_apply_batch!(
                (GP[1], GP[2], GP[3], km), (P[1], P[2], P[3], sq), n == 1 ? plans[1] : dplans[n],
            )
            _planetary_to_local!(loc, GP, og)
            for c in eachindex(loc)
                fc = loc[c]
                @. km -= fc * fc
            end
            @. km = ifelse(mask, T(0.5) * km, zero(T))
            bands[n] = _area_mean(km, og, total_area)
        end
        return _band_result(bands, band_maps, loc, mask, og, total_area)
    end

    # `nxt` receives each next application.
    nxt = (zeros(T, gsz), zeros(T, gsz), zeros(T, gsz))
    # Each band filters the running field AND its square, for every component — one batched apply
    # instead of `2C` separate ones, so the geometry is walked once per band rather than `2C` times.
    nc = has_w ? 3 : 2
    sqs = ntuple(_ -> zeros(T, gsz), 3)
    fsqs = ntuple(_ -> zeros(T, gsz), 3)

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
        bands[n] = _area_mean(km, og, total_area)
        for c in 1:nc
            copyto!(f[c], nxt[c])
        end
    end
    return _band_result(bands, band_maps, loc, mask, og, total_area)
end

# The resolved energy `½⟨|f_N|²⟩` of the final field and the returned named tuple.
function _band_result(bands, band_maps, loc, mask, grid, total_area::T) where {T}
    resolved_map = zero(first(loc))
    for fc in loc
        @. resolved_map += fc * fc
    end
    @. resolved_map = ifelse(mask, T(0.5) * resolved_map, zero(T))
    resolved = _area_mean(resolved_map, grid, total_area)
    return (; bands, resolved, total = sum(bands) + resolved, band_maps, resolved_map)
end

band_energies(u, v, grid::FlowGeometries.Grids.AbstractGrid, kernel, scales; kwargs...) =
    band_energies(u, v, nothing, grid, kernel, scales; kwargs...)
