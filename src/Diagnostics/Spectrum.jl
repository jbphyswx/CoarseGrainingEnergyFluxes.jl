# ---------------------------------------------------------------------------
# Filtering Energy Spectrum E(ℓ)
# ---------------------------------------------------------------------------

"""
    active_area(grid) -> T

Total area of the active cells: the denominator of every spatial average here. A `ZeroFill` output is
summed over every cell of the grid, land included, and divided by this water area (Storer et al.
2022); a `Deformable` one over the active cells.
"""
function active_area(grid::FlowGeometries.Grids.AbstractGrid{G,T}) where {G, T<:AbstractFloat}
    total = zero(T)
    for I in CartesianIndices(FlowGeometries.Grids.size_tuple(grid))
        FlowGeometries.Grids.isactive(grid, Tuple(I)...) || continue
        total += FlowGeometries.Grids.area(grid, Tuple(I)...)
    end
    total > zero(T) || throw(ArgumentError("grid has no active cells (all masked out)"))
    return total
end

"""
    _area_mean(field, grid, total_area) -> T

`Σ field · area` over the active cells of `grid`, divided by `total_area`. Pass the
[`output_grid`](@ref) and [`active_area`](@ref) of the input grid, so every spatial average in this
module is normalized the same way.
"""
function _area_mean(
    field::AbstractArray{T}, grid::FlowGeometries.Grids.AbstractGrid, total_area::T,
) where {T<:AbstractFloat}
    acc = zero(T)
    @inbounds for I in CartesianIndices(FlowGeometries.Grids.size_tuple(grid))
        t = Tuple(I)
        FlowGeometries.Grids.isactive(grid, t...) || continue
        acc += field[I] * FlowGeometries.Grids.area(grid, t...)
    end
    return acc / total_area
end

"""
    energy_from_filtered(ws, grid, has_w, total_area) -> E(ℓ)

`E(ℓ) = ½⟨|ū_ℓ|²⟩` read from the filtered velocities already in `ws`, summed over the active cells of
`grid` (the [`output_grid`](@ref)) and divided by `total_area`. [`compute_Π!`](@ref) leaves
exactly those there, so calling this straight after it filters `u`/`v` once per scale.
"""
function energy_from_filtered(
    ws::ΠWorkspace{T}, grid::FlowGeometries.Grids.AbstractGrid, has_w::Bool, total_area::T,
) where {T<:AbstractFloat}
    return _energy_over(ws.u_filt, ws.v_filt, ws.w_filt, grid, has_w, total_area,
                        FlowGeometries.Grids.size_tuple(grid))
end

"""
    energy_from_filtered!(out, ws, grid, has_w, total_area) -> out

Per-slice `E(ℓ)` from a **batched** workspace: `out` holds one energy per slice, shaped like the
workspace's trailing axes.

`E(ℓ)` is a mean over the domain, so unlike everything else on the batch path this reduces the spatial
axes away while leaving the batch axes intact — it cannot simply broadcast. The mask and cell areas are
spatial-only, so each slice reduces against the same geometry.
"""
function energy_from_filtered!(
    out::AbstractArray{T}, ws::ΠWorkspace{T}, grid::FlowGeometries.Grids.AbstractGrid,
    has_w::Bool, total_area::T,
) where {T<:AbstractFloat}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    valR = Val(length(gsz))
    colons = ntuple(_ -> Colon(), valR)
    # `_batch_dims`, not `size(A)[R+1:end]`: slicing a tuple with a runtime range cannot infer a
    # fixed-size result and allocates on every call
    bsz = _batch_dims(ws.u_filt, valR)
    size(out) == bsz || throw(DimensionMismatch(
        "out has size $(size(out)); the workspace's batch axes are $bsz",
    ))
    # A workspace built without vertical buffers holds `nothing` there, which cannot be `view`ed — so
    # the two cases are separate branches rather than one that slices a buffer it may not have.
    (!has_w || _has_vertical(ws)) || _workspace_missing_vertical()
    @inbounds for J in CartesianIndices(size(out))
        uv = view(ws.u_filt, colons..., Tuple(J)...)
        vv = view(ws.v_filt, colons..., Tuple(J)...)
        out[J] = if has_w
            _energy_over(uv, vv, view(ws.w_filt, colons..., Tuple(J)...), grid, true, total_area, gsz)
        else
            _energy_over(uv, vv, nothing, grid, false, total_area, gsz)
        end
    end
    return out
end

# One slice's domain mean, indexed over the GRID's extent rather than the array's, so it is unaffected by
# any trailing batch axes the caller has already sliced away.
@inline function _energy_over(uf, vf, wf, grid, has_w::Bool, total_area::T, gsz::Tuple) where {T}
    e = zero(T)
    @inbounds for I in CartesianIndices(gsz)
        FlowGeometries.Grids.isactive(grid, Tuple(I)...) || continue
        v2 = uf[I]^2 + vf[I]^2
        has_w && (v2 += wf[I]^2)
        e += v2 * FlowGeometries.Grids.area(grid, Tuple(I)...)
    end
    return T(0.5) * e / total_area
end

"""
    cumulative_energy!(spectrum, u, v, w, grid, kernel, scales; workspace=nothing, backend=AutoBackend(), mask_strategy=ZeroFill())

In-place [`cumulative_energy`](@ref): writes into the caller-supplied `spectrum` vector and, when
`workspace` (a [`ΠWorkspace`](@ref) or [`EnergyWorkspace`](@ref)) is supplied, filters through its
buffers.

On a grid of non-Cartesian geometry the filtered velocity is the one [`compute_Π!`](@ref) uses: the
planetary Cartesian components are filtered and rotated back to the local frame (Aluie 2019), and
`E(ℓ)` is the energy of the tangent part (with the radial one when `w` is given), equal to what
`coarse_grain` reports.
"""
function cumulative_energy!(
    spectrum::AbstractVector{T},
    u::AbstractArray,
    v::AbstractArray,
    w::Union{Nothing, AbstractArray},
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scales::AbstractVector;
    workspace::Union{Nothing, ΠWorkspace, EnergyWorkspace} = nothing,
    filter_plans = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Filtering.AbstractFilterMethod = Filtering.RealSpace(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    size(u) == gsz || throw(DimensionMismatch("u has size $(size(u)), grid expects $gsz"))
    size(v) == gsz || throw(DimensionMismatch("v has size $(size(v)), grid expects $gsz"))
    w === nothing || size(w) == gsz || throw(DimensionMismatch("w has size $(size(w)), grid expects $gsz"))

    Nscales = length(scales)
    length(spectrum) == Nscales || throw(DimensionMismatch(
        "spectrum has length $(length(spectrum)), expected $Nscales (= length(scales))",
    ))

    # `E(ℓ)` reads only filtered velocity, so a full flux workspace is far more than this needs. One is
    # still accepted, since a sweep that already holds one should not allocate a second.
    ws = workspace === nothing ? EnergyWorkspace(grid; has_w = w !== nothing) : workspace
    _check_workspace_w(ws, w)
    u_filt, v_filt, w_filt = ws.u_filt, ws.v_filt, ws.w_filt
    planetary = !(G <: FlowGeometries.Geometry.CartesianGeometry)
    if planetary
        (ws.ux === nothing || ws.w_filt === nothing) && throw(ArgumentError(
            "this workspace has no planetary-component buffers; build it from the $(nameof(typeof(grid))) " *
            "it filters, `EnergyWorkspace(grid)`",
        ))
        pin, pout = _planetary_buffers(ws)
        _fill_planetary!(pin, u, v, w, grid)
        loc = w === nothing ? (u_filt, v_filt) : (u_filt, v_filt, w_filt)
    end

    # Dimension-generic active-cell iteration: `Tuple(I)...` splats to (i,) for a 1D UnstructuredGrid
    # or (i,j) for a 2D Structured/CurvilinearGrid, matching each grid's own `isactive`/`area` arity.
    idxs = CartesianIndices(u)
    total_area = active_area(grid)

    # Sweep through scales. When the caller (typically `coarse_grain!`, which already builds one
    # plan per scale for its own `compute_Π!` loop) supplies `filter_plans`, reuse those instead of
    # rebuilding the same footprint a second time — otherwise this becomes the dominant allocation in
    # a `coarse_grain!` sweep, since each footprint build costs far more than the rest of the loop body.
    # One family for the whole sweep, not one plan per iteration: the grid-determined half of an engine
    # (measure prefix scans, transform objects, the extended axis) is the same at every scale, and
    # building it inside the loop made it the dominant allocation of the call.
    plans = filter_plans === nothing ?
        Filtering.plan_filter_sweep(
            grid, kernel, scales;
            mask_strategy = mask_strategy, backend = backend, method = method,
        ) : filter_plans
    og = output_grid(grid, plans[1])

    for s_idx in 1:Nscales
        plan = plans[s_idx]

        # Filter velocity fields at this scale — batched (one derivation per point, not one per field).
        if planetary
            Filtering.filter_apply_batch!(pout, pin, plan)
            _planetary_to_local!(loc, pout, og)
        elseif w !== nothing
            Filtering.filter_apply_batch!((u_filt, v_filt, w_filt), (u, v, w), plan)
        else
            Filtering.filter_apply_batch!((u_filt, v_filt), (u, v), plan)
        end

        # Spatial average specific energy: E(ℓ) = 0.5 ∫ |ū_ℓ|² dA over the output grid, per unit water
        # area.
        integrated_energy = zero(T)
        for I in idxs
            if FlowGeometries.Grids.isactive(og, Tuple(I)...)
                vel2 = u_filt[I]^2 + v_filt[I]^2
                if w !== nothing
                    vel2 += w_filt[I]^2
                end
                integrated_energy += vel2 * FlowGeometries.Grids.area(grid, Tuple(I)...)
            end
        end

        spectrum[s_idx] = T(0.5) * integrated_energy / total_area
    end

    return spectrum
end

"""
    cumulative_energy(u, v, w, grid, kernel, scales; backend=AutoBackend(), mask_strategy=ZeroFill())

Cumulative coarse-grained kinetic energy `E(ℓ) = 0.5 ⟨|ū_ℓ|²⟩` at each filter scale
(Sadek & Aluie 2018, PRF, Eq. 15). This is the CUMULATIVE quantity; the filtering spectral DENSITY
(comparable to a Fourier energy spectrum) is its derivative w.r.t. filtering wavenumber — see
[`Diagnostics.filtering_spectrum`](@ref). Allocates a fresh `spectrum` vector each call; for a repeated sweep
(e.g. inside `coarse_grain!`), call [`cumulative_energy!`](@ref) directly with a reused buffer.

# Examples
```julia
scales = collect(10000.0:10000.0:100000.0)  # 10-100 km
E = cumulative_energy(u, v, nothing, grid, TopHatKernel(), scales)
# E[i] is the cumulative coarse KE at scale scales[i]
```

# References
- Sadek & Aluie (2018), *Phys. Rev. Fluids* 3, 124610 — extracting the spectrum by filtering.
"""
function cumulative_energy(
    u::AbstractArray,
    v::AbstractArray,
    w::Union{Nothing, AbstractArray},
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scales::AbstractVector;
    workspace::Union{Nothing, ΠWorkspace, EnergyWorkspace} = nothing,
    filter_plans = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Filtering.AbstractFilterMethod = Filtering.RealSpace(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    spectrum = zeros(T, length(scales))
    return cumulative_energy!(
        spectrum, u, v, w, grid, kernel, scales;
        workspace = workspace, filter_plans = filter_plans,
        backend = backend, mask_strategy = mask_strategy, method = method,
    )
end

"""
    filtering_spectrum(u, v, w, grid, kernel, scales; L=1, backend=AutoBackend(), mask_strategy=ZeroFill())
        -> (k_ℓ, Ẽ)

Filtering spectral DENSITY (Sadek & Aluie 2018, PRF, Eq. 14): the derivative of the cumulative
coarse-grained KE w.r.t. the filtering wavenumber `k_ℓ = L/ℓ`,

    Ẽ(k_ℓ) = d/dk_ℓ [ ½⟨|ū_ℓ|²⟩ ] = -(ℓ²/L) d/dℓ[ ½⟨|ū_ℓ|²⟩ ].

Unlike [`cumulative_energy`](@ref) (the cumulative quantity, Eq. 15), this is the spectral density
comparable to a Fourier energy spectrum. `scales` need not be uniform. Returns the filtering
wavenumbers `k_ℓ` and the density `Ẽ` per scale.

# The `k_ℓ = C/ℓ` convention, and why it must be stated

`L` is the region length, and `k_ℓ = L/ℓ` is the Sadek–Aluie convention: with their Fourier series
`f(x) = Σ_k f̂(k) e^{i(2π/L)k·x}`, `k` is a dimensionless index, so `L` is the domain size. The default
`L = 1` instead gives `k_ℓ = 1/ℓ`, matching Storer et al. (2022, 2023) and FlowSieve. A third
convention, `k_ℓ = 2π/ℓ` (Rivera, Aluie & Ecke 2014), is `L = 2π`.

**The choice rescales the answer.** Under `k_ℓ = C/ℓ` the density carries a Jacobian `dℓ/dk_ℓ =
-ℓ²/C`, so `Ẽ` scales as `1/C` while `k_ℓ` scales as `C`. Comparing amplitudes — or peak locations —
against a Fourier spectrum or against another code is meaningless unless the conventions match. The
cumulative energy [`cumulative_energy`](@ref) is convention-free; only the density is not.

# Limits

- **Slope ceiling.** Sadek & Aluie eq. (18): a kernel with `p` vanishing moments recovers a true
  `k^{-α}` spectrum only for `α < p + 2`, and otherwise saturates at `k^{-(p+2)}`. Both
  `TopHatKernel` and `GaussianKernel` have `p = 1`, so **the measured slope locks at `k⁻³`**. This
  bites hardest in 2-D and QG work, where the enstrophy-range target slope *is* ≈ `k⁻³`. The flux
  `Π` is unaffected — this is a limitation of the spectrum diagnostic alone.
- **Kernel admissibility.** `Ẽ(k_ℓ) ≥ 0` is guaranteed only when `d|Ĝ(k)|²/dk ≤ 0`. By default this
  function throws for a kernel that fails it; pass `policy = ForceSpectrum()` to compute it anyway.
  See [`AbstractSpectrumPolicy`](@ref) and [`Kernels.transfer_monotone`](@ref).

# References
- Sadek & Aluie (2018), *Phys. Rev. Fluids* 3, 124610.
"""
function filtering_spectrum(
    u::AbstractArray,
    v::AbstractArray,
    w::Union{Nothing, AbstractArray},
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scales::AbstractVector;
    L::Real = one(T),
    workspace::Union{Nothing, ΠWorkspace, EnergyWorkspace} = nothing,
    filter_plans = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Filtering.AbstractFilterMethod = Filtering.RealSpace(),
    policy::AbstractSpectrumPolicy = StrictSpectrum(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    compute = gate_spectrum(kernel, policy)
    kℓ = T(L) ./ T.(scales)
    # The gate first: a refused spectrum should not pay for a sweep whose result is discarded.
    compute || return kℓ, fill(T(NaN), length(kℓ))
    cum = cumulative_energy(
        u, v, w, grid, kernel, scales;
        workspace = workspace, filter_plans = filter_plans,
        backend = backend, mask_strategy = mask_strategy, method = method,
    )
    return kℓ, spectral_density(cum, kℓ)
end

"""
    spectral_density!(g, C, k) -> g

In-place [`spectral_density`](@ref): writes the non-uniform finite-difference derivative of `C`
w.r.t. `k` into the caller-supplied `g` (central in the interior, one-sided at the ends). Fills
zeros for fewer than two points.
"""
function spectral_density!(g::AbstractVector{T}, C::AbstractVector{T}, k::AbstractVector) where {T<:AbstractFloat}
    n = length(C)
    length(g) == n || throw(DimensionMismatch("g has length $(length(g)), expected $n (= length(C))"))
    n < 2 && (fill!(g, zero(T)); return g)
    @inbounds for i in 1:n
        if i == 1
            g[i] = (C[2] - C[1]) / (k[2] - k[1])
        elseif i == n
            g[i] = (C[n] - C[n-1]) / (k[n] - k[n-1])
        else
            g[i] = (C[i+1] - C[i-1]) / (k[i+1] - k[i-1])
        end
    end
    return g
end

"""
    spectral_density(C, k) -> dC/dk

Non-uniform finite-difference derivative of cumulative values `C` w.r.t. `k` (central in the
interior, one-sided at the ends). Returns zeros for fewer than two points.
"""
function spectral_density(C::AbstractVector{T}, k::AbstractVector) where {T<:AbstractFloat}
    return spectral_density!(zeros(T, length(C)), C, k)
end
