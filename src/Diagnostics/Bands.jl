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

The sum telescopes because each `⟨G * x⟩ = ⟨x⟩` — i.e. **because the filter conserves the domain
mean**. Measured, that holds to round-off on a periodic, unmasked grid (relative error 5e-16 for the
top-hat, 2e-15 for the Gaussian) and the identity is exact there.

Anywhere the footprint is truncated, it is not, and the identity carries a residual of order `ℓ/L`:
measured on the same field, **1.1e-2 relative on a BOUNDED grid** (the footprint runs off the domain
edge) and **1.1e-2 on a masked periodic grid under `ZeroFill`** (energy is smeared onto masked cells,
which report zero). `Deformable` renormalizes that leakage away and does better on a masked domain —
4.8e-4 — at the cost of the commutation property `ZeroFill` is the default for. So: read the bands as
exact on a periodic unmasked domain, and as carrying an `O(ℓ/L)` boundary residual otherwise.

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

    # `f` is the running repeatedly-filtered field; `nxt` receives each next application. `sq`/`fsq`
    # carry `|f|²` and its filtered image, which is what makes this a second moment and not a
    # band-passed velocity.
    f = (copy(u), copy(v), has_w ? copy(w) : nothing)
    nxt = (zeros(T, gsz), zeros(T, gsz), has_w ? zeros(T, gsz) : nothing)
    sq = zeros(T, gsz); fsq = zeros(T, gsz)

    # Per-band maps are `N` full fields and most callers want only the scalars they reduce to, so they
    # are opt-in. Without them one scratch map is reused for every band.
    band_maps = maps ? [zeros(T, gsz) for _ in eachindex(scales)] : nothing
    scratch_map = maps ? nothing : zeros(T, gsz)
    bands = zeros(T, length(scales))

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
        Filtering.filter_apply_batch!(outs, srcs, plans[n])
        for c in 1:nc
            # τ(f;f) = (f²)‾ − (f̄)², summed over components; the ½ is applied once at the end.
            @. km += fsqs[c] - nxt[c] * nxt[c]
        end
        mask = FlowGeometries.Grids.mask(grid)
        @. km = ifelse(mask, T(0.5) * km, zero(T))
        bands[n] = _area_mean(km, grid, total_area)
        for c in 1:nc
            copyto!(f[c], nxt[c])
        end
    end

    resolved_map = zeros(T, gsz)
    let mask = FlowGeometries.Grids.mask(grid)
        for c in 1:(has_w ? 3 : 2)
            fc = f[c]
            @. resolved_map += fc * fc
        end
        @. resolved_map = ifelse(mask, T(0.5) * resolved_map, zero(T))
    end
    resolved = _area_mean(resolved_map, grid, total_area)
    return (; bands, resolved, total = sum(bands) + resolved, band_maps, resolved_map)
end

band_energies(u, v, grid::FlowGeometries.Grids.AbstractGrid, kernel, scales; kwargs...) =
    band_energies(u, v, nothing, grid, kernel, scales; kwargs...)
