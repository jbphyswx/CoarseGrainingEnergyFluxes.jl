# ---------------------------------------------------------------------------
# Filtering Energy Spectrum E(ℓ)
# ---------------------------------------------------------------------------

"""
    active_area(grid) -> T

Total measure of the active cells, `FlowGeometries.Grids.covered_area`: the denominator of every spatial
average here. A `ZeroFill` output is summed over every cell of the grid, land included, and divided by
this water area (Storer et al. 2022); a `Deformable` one over the active cells.
"""
function active_area(grid::FlowGeometries.Grids.AbstractGrid{G,T}) where {G, T<:AbstractFloat}
    total = FlowGeometries.Grids.covered_area(grid)
    total > zero(T) || throw(ArgumentError("grid has no active cells (all masked out)"))
    return total
end

"""
    _area_mean(field, grid, total_area, b) -> T

`Σ field · measure` over the active cells of `grid`, divided by `total_area`, reduced under the
backend `b` whose memory `field` and `grid` are in. Pass the [`output_grid`](@ref) and
[`active_area`](@ref) of the input grid, so every spatial average in this module is normalized the same
way.
"""
_area_mean(field::AbstractArray{T}, grid::FlowGeometries.Grids.AbstractGrid, total_area::T, b) where {T<:AbstractFloat} =
    FlowGeometries.Grids.integrate(field, grid; execution = b) / total_area

# `Σ |c|² · measure` over the active cells of `og`, summing the squares of the components `comps` of
# the batch element that starts at linear offset `off`; `og` and the arrays are in `b`'s memory.
function _kinetic_integral(
    comps::Tuple{AbstractArray{T},Vararg{AbstractArray{T}}}, og::FlowGeometries.Grids.AbstractGrid, off::Int, b,
) where {T}
    msk = FlowGeometries.Grids.mask(og)
    m = FlowGeometries.Grids.measure(og)
    return FlowGeometries.Execution.reduce_indices(+, zero(T), length(og), b) do i
        @inbounds _cell_on(msk, i) ? _sumsq(comps, off + i) * T(m[i]) : zero(T)
    end
end

@inline _sumsq(c::NTuple{1,AbstractArray}, k::Int) = @inbounds c[1][k]^2
@inline _sumsq(c::NTuple{2,AbstractArray}, k::Int) = @inbounds c[1][k]^2 + c[2][k]^2
@inline _sumsq(c::NTuple{3,AbstractArray}, k::Int) = @inbounds (c[1][k]^2 + c[2][k]^2) + c[3][k]^2

@inline _cell_on(::FlowGeometries.Grids.AllActive, ::Int) = true
@inline _cell_on(m::AbstractArray{Bool}, i::Int) = @inbounds m[i]

# `½⟨|ū|²⟩` of the filtered velocity in `ws`, for the batch element at linear offset `off`.
@inline function _filtered_energy(ws, og, has_w::Bool, off::Int, total_area::T) where {T}
    b = ws.at.execution
    e = has_w ? _kinetic_integral(_with_w(ws.u_filt, ws.v_filt, ws.w_filt), og, off, b) :
                _kinetic_integral((ws.u_filt, ws.v_filt), og, off, b)
    return T(0.5) * e / total_area
end

@inline _with_w(u, v, w::AbstractArray) = (u, v, w)
@inline _with_w(u, v, ::Nothing) = (u, v)

"""
    energy_from_filtered(ws, plan, has_w, total_area) -> E(ℓ)

`E(ℓ) = ½⟨|ū_ℓ|²⟩` read from the filtered velocities already in `ws`, summed over the active cells of
the grid `plan`'s outputs are read on ([`output_grid`](@ref)) and divided by `total_area`. Call it after
[`compute_Π!`](@ref) with `plan` on the same `ws`, which leaves the filtered velocities there; `plan`
may be the mask strategy that call read in its place.
"""
function energy_from_filtered(
    ws::ΠWorkspace{T}, plan::_PlanOrStrategy, has_w::Bool, total_area::T,
) where {T<:AbstractFloat}
    (!has_w || _has_vertical(ws)) || _workspace_missing_vertical()
    return _filtered_energy(ws, _output(ws.at, plan), has_w, 0, total_area)
end

"""
    energy_from_filtered!(out, ws, plan, has_w, total_area) -> out

Per-slice `E(ℓ)` from a batched workspace: `out` holds one energy per slice, shaped like the
workspace's trailing axes. Every slice is averaged against the same mask and cell areas.
"""
function energy_from_filtered!(
    out::AbstractArray{T}, ws::ΠWorkspace{T}, plan::_PlanOrStrategy,
    has_w::Bool, total_area::T,
) where {T<:AbstractFloat}
    og = _output(ws.at, plan)
    gsz = FlowGeometries.Grids.size_tuple(og)
    bsz = _batch_dims(ws.u_filt, Val(length(gsz)))
    size(out) == bsz || throw(DimensionMismatch(
        "out has size $(size(out)); the workspace's batch axes are $bsz",
    ))
    (!has_w || _has_vertical(ws)) || _workspace_missing_vertical()
    n = prod(gsz)
    for (k, J) in enumerate(CartesianIndices(size(out)))
        @inbounds out[J] = _filtered_energy(ws, og, has_w, (k - 1) * n, total_area)
    end
    return out
end

"""
    cumulative_energy!(spectrum, u, v, w, grid, kernel, scales; workspace=nothing, filter_plans=nothing, execution=AutoBackend(), mask_strategy=ZeroFill(), method=RealSpace(), vector_filter=nothing, transform=AutoSpectralBackend())

In-place [`cumulative_energy`](@ref): writes into the caller-supplied `spectrum` vector and, when
`workspace` (a [`ΠWorkspace`](@ref) or [`EnergyWorkspace`](@ref), in the memory of the plans' backend)
is supplied, filters through its buffers.

On a grid of non-Cartesian geometry the filtered velocity is the one [`compute_Π!`](@ref) uses, and
`E(ℓ)` is the energy of its tangent part (with the radial one when `w` is given), equal to what
`coarse_grain` reports. On the surface of a sphere `vector_filter`
([`Filtering.resolve_vector_filter`](@ref)) filters it: by default the generalized convolution of Aluie
(2019), whose spin transforms `transform` chooses and which reads no scalar filter plan, so none is
built when `filter_plans` is not given. `Filtering.CartesianComponents()`, and every spheroid, filters
the ambient Cartesian components and rotates them back to the local frame. A given `workspace` carries
its own vector filter.
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
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Filtering.AbstractFilterMethod = Filtering.RealSpace(),
    vector_filter::Union{Nothing, Filtering.AbstractVectorFilter} = nothing,
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    size(u) == gsz || throw(DimensionMismatch("u has size $(size(u)), grid expects $gsz"))
    size(v) == gsz || throw(DimensionMismatch("v has size $(size(v)), grid expects $gsz"))
    w === nothing || size(w) == gsz || throw(DimensionMismatch("w has size $(size(w)), grid expects $gsz"))

    Nscales = length(scales)
    length(spectrum) == Nscales || throw(DimensionMismatch(
        "spectrum has length $(length(spectrum)), expected $Nscales (= length(scales))",
    ))
    Nscales == 0 && return spectrum

    strategy = filter_plans === nothing ? mask_strategy : Filtering.plan_strategy(first(filter_plans))
    sweep() = Filtering.plan_filter_sweep(
        grid, kernel, scales; mask_strategy = mask_strategy, execution = execution, method = method,
    )
    plans = filter_plans
    ws = workspace
    if ws === nothing
        vf = Filtering.resolve_vector_filter(grid, strategy, vector_filter)
        vf isa Filtering.GeneralizedConvolution || plans !== nothing || (plans = sweep())
        ws = EnergyWorkspace(
            grid; has_w = w !== nothing, vector_filter = vf, transform = transform,
            execution = plans === nothing ? execution : Filtering.plan_execution(first(plans)),
        )
    end
    _check_workspace_w(ws, w)
    total_area = active_area(grid)
    if ws.spin !== nothing
        plans === nothing || _check_placement(ws.at, first(plans))
        return _spin_cumulative_energy!(spectrum, _spin_of(ws.spin, strategy), ws, u, v, w, kernel, scales,
                                        total_area)
    end
    plans === nothing && (plans = sweep())
    _check_placement(ws.at, first(plans))
    return _cumulative_energy!(spectrum, ws, plans, u, v, w, grid, total_area)
end

function _cumulative_energy!(
    spectrum::AbstractVector{T}, ws, plans, u, v, w, grid::FlowGeometries.Grids.AbstractGrid{G,T}, total_area::T,
) where {T<:AbstractFloat, G}
    b = ws.at.execution
    u_filt, v_filt, w_filt = ws.u_filt, ws.v_filt, ws.w_filt
    ambient = !(G <: FlowGeometries.Geometry.CartesianGeometry)
    if ambient
        (ws.ux === nothing || ws.w_filt === nothing) && throw(ArgumentError(
            "this workspace has no buffers for the ambient Cartesian components; build it from the " *
            "$(nameof(typeof(grid))) it filters, `EnergyWorkspace(grid)`",
        ))
        pin, pout = _ambient_buffers(ws)
        _to_ambient!(pin, u, v, w, ws.at.grid, b)
        loc = w === nothing ? (u_filt, v_filt) : (u_filt, v_filt, w_filt)
    end

    for s_idx in eachindex(spectrum, plans)
        plan = plans[s_idx]
        og = _output(ws.at, plan)

        if ambient
            Filtering.filter_apply_batch!(pout, pin, plan)
            _ambient_to_local!(loc, pout, og, b)
        elseif w !== nothing
            Filtering.filter_apply_batch!((u_filt, v_filt, w_filt), (u, v, w), plan)
        else
            Filtering.filter_apply_batch!((u_filt, v_filt), (u, v), plan)
        end

        # Spatial average specific energy: E(ℓ) = 0.5 ∫ |ū_ℓ|² dA over the output grid, per unit water
        # area.
        integrated_energy = w === nothing ? _kinetic_integral((u_filt, v_filt), og, 0, b) :
                                            _kinetic_integral(_with_w(u_filt, v_filt, w_filt), og, 0, b)
        spectrum[s_idx] = T(0.5) * integrated_energy / total_area
    end

    return spectrum
end

# `E(ℓ)` under the generalized convolution: the velocity is analyzed once and synthesized per scale, on
# every cell of the grid.
function _spin_cumulative_energy!(
    spectrum::AbstractVector{T}, sf::SpinFilter, ws, u, v, w, kernel, scales, total_area::T,
) where {T<:AbstractFloat}
    b = ws.at.execution
    og = ws.at.unmasked
    u_filt, v_filt, w_filt = ws.u_filt, ws.v_filt, ws.w_filt
    _spin_analyze!(sf.coeffs[1], sf, 1, true, k -> @inbounds(complex(u[k], v[k])))
    w === nothing || _spin_analyze!(sf.coeffs[5], sf, 0, true, k -> @inbounds(complex(w[k])))
    for s_idx in eachindex(spectrum, scales)
        _spin_transfer!(sf, kernel, scales[s_idx])
        _spin_synthesize!(sf, sf.coeffs[1], 1)
        _spin_store!(sf) do k, z
            @inbounds u_filt[k] = real(z)
            @inbounds v_filt[k] = imag(z)
        end
        if w !== nothing
            _spin_synthesize!(sf, sf.coeffs[5], 0)
            _spin_store!(sf) do k, z
                @inbounds w_filt[k] = real(z)
            end
        end
        integrated_energy = w === nothing ? _kinetic_integral((u_filt, v_filt), og, 0, b) :
                                            _kinetic_integral(_with_w(u_filt, v_filt, w_filt), og, 0, b)
        spectrum[s_idx] = T(0.5) * integrated_energy / total_area
    end
    return spectrum
end

"""
    cumulative_energy(u, v, w, grid, kernel, scales; workspace=nothing, filter_plans=nothing, execution=AutoBackend(), mask_strategy=ZeroFill(), method=RealSpace(), vector_filter=nothing, transform=AutoSpectralBackend())

Cumulative coarse-grained kinetic energy `E(ℓ) = 0.5 ⟨|ū_ℓ|²⟩` at each filter scale
(Sadek & Aluie 2018, PRF, Eq. 15). The filtering spectral density, comparable to a Fourier energy
spectrum, is its derivative with respect to the filtering wavenumber: see
[`Diagnostics.filtering_spectrum`](@ref). Allocating form of [`cumulative_energy!`](@ref).

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
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Filtering.AbstractFilterMethod = Filtering.RealSpace(),
    vector_filter::Union{Nothing, Filtering.AbstractVectorFilter} = nothing,
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    spectrum = zeros(T, length(scales))
    return cumulative_energy!(
        spectrum, u, v, w, grid, kernel, scales;
        workspace = workspace, filter_plans = filter_plans,
        execution = execution, mask_strategy = mask_strategy, method = method,
        vector_filter = vector_filter, transform = transform,
    )
end

"""
    filtering_spectrum(u, v, w, grid, kernel, scales; L=1, workspace=nothing, filter_plans=nothing, execution=AutoBackend(), mask_strategy=ZeroFill(), method=RealSpace(), vector_filter=nothing, transform=AutoSpectralBackend(), policy=StrictSpectrum())
        -> (k_ℓ, Ẽ)

Filtering spectral density (Sadek & Aluie 2018, PRF, Eq. 14): the derivative of the cumulative
coarse-grained KE ([`cumulative_energy`](@ref), Eq. 15) with respect to the filtering wavenumber
`k_ℓ = L/ℓ`,

    Ẽ(k_ℓ) = d/dk_ℓ [ ½⟨|ū_ℓ|²⟩ ] = -(ℓ²/L) d/dℓ[ ½⟨|ū_ℓ|²⟩ ].

`scales` need not be uniform. Returns the filtering wavenumbers `k_ℓ` and the density `Ẽ` per scale.

# Wavenumber convention

`k_ℓ = L/ℓ` with `L` the region length is the Sadek–Aluie convention: in their Fourier series
`f(x) = Σ_k f̂(k) e^{i(2π/L)k·x}`, `k` is a dimensionless index. The default `L = 1` gives
`k_ℓ = 1/ℓ`, the convention of Storer et al. (2022, 2023); `L = 2π` gives `k_ℓ = 2π/ℓ` (Rivera,
Aluie & Ecke 2014).

Under `k_ℓ = C/ℓ` the density carries the Jacobian `dℓ/dk_ℓ = -ℓ²/C`, so `Ẽ` scales as `1/C` and
`k_ℓ` as `C`: amplitudes and peak locations compare across spectra only under one convention.
[`cumulative_energy`](@ref) is independent of `C`.

# Limits

- **Slope ceiling.** Sadek & Aluie eq. (18): a kernel with `p` vanishing moments recovers a
  `k^{-α}` spectrum only for `α < p + 2`, and otherwise saturates at `k^{-(p+2)}`. `TopHatKernel`
  and `GaussianKernel` have `p = 1`, so the measured slope saturates at `k⁻³`, the enstrophy-range
  slope of 2-D and QG turbulence. The flux `Π` has no such ceiling.
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
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Filtering.AbstractFilterMethod = Filtering.RealSpace(),
    vector_filter::Union{Nothing, Filtering.AbstractVectorFilter} = nothing,
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    policy::AbstractSpectrumPolicy = StrictSpectrum(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    compute = gate_spectrum(kernel, policy)
    kℓ = T(L) ./ T.(scales)
    compute || return kℓ, fill(T(NaN), length(kℓ))
    cum = cumulative_energy(
        u, v, w, grid, kernel, scales;
        workspace = workspace, filter_plans = filter_plans,
        execution = execution, mask_strategy = mask_strategy, method = method,
        vector_filter = vector_filter, transform = transform,
    )
    return kℓ, spectral_density(cum, kℓ)
end

"""
    spectral_density!(g, C, k) -> g

In-place [`spectral_density`](@ref): writes `dC/dk` into the caller-supplied `g`.
"""
function spectral_density!(g::AbstractVector{T}, C::AbstractVector{T}, k::AbstractVector) where {T<:AbstractFloat}
    n = length(C)
    length(g) == n || throw(DimensionMismatch("g has length $(length(g)), expected $n (= length(C))"))
    length(k) == n || throw(DimensionMismatch("k has length $(length(k)), expected $n (= length(C))"))
    n < 2 && return fill!(g, T(NaN))
    if n == 2
        s = T((C[2] - C[1]) / (k[2] - k[1]))
        return fill!(g, s)
    end
    @inbounds for i in 1:n
        j = clamp(i - 1, 1, n - 2)   # the three points nearest `i`: centred inside, one-sided at an end
        g[i] = _quadratic_slope(k[i], k[j], k[j+1], k[j+2], C[j], C[j+1], C[j+2])
    end
    return g
end

# Derivative at `x` of the quadratic through `(x0, y0)`, `(x1, y1)`, `(x2, y2)` (Lagrange form).
@inline _quadratic_slope(x, x0, x1, x2, y0, y1, y2) =
    y0 * ((x - x1) + (x - x2)) / ((x0 - x1) * (x0 - x2)) +
    y1 * ((x - x0) + (x - x2)) / ((x1 - x0) * (x1 - x2)) +
    y2 * ((x - x0) + (x - x1)) / ((x2 - x0) * (x2 - x1))

"""
    spectral_density(C, k) -> dC/dk

Derivative of cumulative values `C` with respect to `k`, at each `k`: the slope of the quadratic through
the three nearest points, centred in the interior and one-sided at the ends, so second order on any
spacing and exact for a quadratic `C(k)`. Two points give their secant; with fewer there is no
derivative, and the result is `NaN`.
"""
function spectral_density(C::AbstractVector{T}, k::AbstractVector) where {T<:AbstractFloat}
    return spectral_density!(zeros(T, length(C)), C, k)
end
