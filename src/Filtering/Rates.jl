# ---------------------------------------------------------------------------
# A real-space plan's derivative in its width
# ---------------------------------------------------------------------------
#
# A real-space filter is `ū = N_G/M_G`, its numerator and window mass sums of `G(d/ℓ)` times the cell measure over
# the footprint's cells. With `H(s) = −s G′(s)`, `ℓ∂_ℓ G(d/ℓ) = H(d/ℓ)`, so over the same cells
# `ℓ∂_ℓū = (N_H − ū M_H)/M_G = ρ(ū_H − ū)`, `ρ = M_H/M_G` and `ū_H` the filter with `H`. A separable kernel's tables
# give one such filter per axis (`Kernels.AxisRate`), and the terms add. With `D = ℓ∂_ℓ`, the second derivative is
# `D²ū = (D²N − ū D²M)/M − 2(Dū)(DM/M)`, `D²N` the filters of `J = −sH′(s)` (`ScaleRate(ScaleRate)`), or of the
# separable terms `J_d` and twice `H_d H_e` (nested `AxisRate`).

"""
    rate_plans(plan::PhysicalFilterPlan) -> Tuple

The plans whose filters give `plan`'s derivative in its width, `ℓ∂_ℓū = Σ_p (M_p/M)(ū_p − ū)` with `M` each plan's
[`window_mass!`](@ref): one of `Kernels.ScaleRate(kernel)` for radial weights, and for a separable kernel's tables one
of `Kernels.AxisRate(kernel, Val(d))` per axis. Each is built on `plan`'s grid, strategy, backend and engine, over the
cells `plan` sums, caching its neighbours where `plan` does and sharing its pass buffers.
"""
rate_plans(plan::PhysicalFilterPlan) = _rate_plans(plan, _rate_kernels(_weight_family(_host_footprint(plan.footprint),
                                                                                       plan.kernel), plan.kernel, plan.grid))

function _rate_plans(plan::PhysicalFilterPlan, kernels::Tuple)
    fp = _host_footprint(plan.footprint)
    kw = (; mask_strategy = plan.strategy, cache_strategy = AutoCache(), cache_byte_budget = _cache_bytes(fp),
            scratch = _pass_buffers(fp), execution = plan.execution)
    return map(kernels) do k
        _physical_plan(plan.execution, _engine_of(fp), plan.grid, k, plan.scale, plan.strategy, kw)
    end
end

# A real-space plan's derivative in its width is exact for a kernel whose profile is differentiable in distance.
rate_available(::PhysicalFilterPlan, kernel::Kernels.AbstractFilterKernel) = Kernels.has_profile_rate(kernel)
curvature_available(plan::PhysicalFilterPlan, kernel::Kernels.AbstractFilterKernel) =
    rate_available(plan, Kernels.ScaleRate(kernel))

"""
    WindowRate

A real-space plan's [`scale_rate`](@ref): its [`rate_plans`](@ref), `ρ_p = M_p/M` from the window masses, and
`B = Σ_p ρ_p`, so `ℓ∂_ℓū = A − ū·B` with `A = Σ_p ρ_p ū_p`; and `second`, `nothing` or the [`WindowSecond`](@ref)
of the second derivative.
"""
struct WindowRate{P<:Tuple, R<:Tuple, B<:AbstractArray, S}
    plans::P
    ρ::R
    B::B
    second::S
end

"""
    WindowSecond

The second derivative of a [`WindowRate`](@ref): the plans of the second-order weights and `ρ_p = c_p M_p/M`, `c_p`
the term's multiplicity, with `B = Σ_p ρ_p = D²M/M`, so `D²ū = A − ū·B − 2(Dū)·B₁` with `A = Σ_p ρ_p ū_p`, `D = ℓ∂_ℓ` and
`B₁` the first derivative's.
"""
struct WindowSecond{P<:Tuple, R<:Tuple, B<:AbstractArray}
    plans::P
    ρ::R
    B::B
end

"""
    SumRate

A nonuniform-FFT kernel sum's [`scale_rate`](@ref): the plan, its [`rate_plan`](@ref), `B = ℓ ∂_ℓM/M` and three analysis
buffers, so `ℓ∂_ℓū = A − ū·B` with `A = ℓ ∂_ℓN/M` synthesized from the field's analysis; and `second`, `nothing` or
the [`SumSecond`](@ref) of the second derivative.
"""
struct SumRate{P, Q, B<:AbstractArray, F<:NTuple{3,Any}, T<:AbstractFloat, S}
    plan::P
    rate::Q
    B::B
    F::F
    scale::T
    second::S
end

"""
    SumSecond

The second derivative of a [`SumRate`](@ref): the [`rate_plan`](@ref) of `Kernels.TransferRate(kernel)`, whose
synthesis is `∂²_ℓN/M`, and `B = D²M/M = ℓ²∂²_ℓM/M + ℓ∂_ℓM/M`, so `D²ū = A − ū·B − 2(Dū)·B₁` with
`A = ℓ²∂²_ℓN/M + ℓ∂_ℓN/M`.
"""
struct SumSecond{Q, B<:AbstractArray}
    rate::Q
    B::B
end

"""
    scale_rate(plan::PhysicalFilterPlan[, order = ScaleRates()]) -> WindowRate or SumRate

What `plan`'s derivative in its width reads beyond a field, built from the plan alone: a [`WindowRate`](@ref) for a
footprint sum, a [`SumRate`](@ref) for a nonuniform-FFT kernel sum; with `order = ScaleCurvatures()` their second
derivatives as well. Raises where [`rate_available`](@ref), or for the second derivative
[`curvature_available`](@ref), is false.
"""
function scale_rate(plan::PhysicalFilterPlan, order::AbstractScaleRates = ScaleRates())
    rate_available(plan, plan.kernel) || _no_scale_rate(plan)
    _check_order(order, plan)
    return _scale_rate(series_form(plan), plan, order)
end
scale_rate(plan::AbstractFilterPlan, ::AbstractScaleRates = ScaleRates()) = _no_scale_rate(plan)

_check_order(::ScaleRates, _plan) = nothing
_check_order(::ScaleCurvatures, plan) = curvature_available(plan, plan.kernel) ? nothing : _no_scale_rate(plan)

@noinline _no_scale_rate(plan) = throw(ArgumentError(
    "$(nameof(typeof(plan))) has no scale rate to build: they are built for a real-space plan whose kernel has a " *
    "derivative in its width (`Kernels.has_profile_rate`, or `has_transfer_rate` for a nonuniform-FFT kernel sum), " *
    "and for the second derivative one whose kernel has that too; a spectral plan synthesizes its rates from its " *
    "transfer per scale",
))

_family_rates(order::Union{ScaleRates,ScaleCurvatures}, plans) = map(p -> scale_rate(p, order), plans)

# A zeroed array of the plan's element type shaped like its grid's mask, in the plan's memory.
_mask_shaped(plan::PhysicalFilterPlan{FP,G,S,K,T}) where {FP,G,S,K,T} =
    _allocate(plan.execution, T, size(FlowGeometries.Grids.mask(plan.grid)))

function _scale_rate(::NoSeries, plan::PhysicalFilterPlan, order::AbstractScaleRates)
    family = _weight_family(_host_footprint(plan.footprint), plan.kernel)
    M = window_mass!(_mask_shaped(plan), plan)
    rp, ρ = _window_terms(plan, _rate_kernels(family, plan.kernel, plan.grid), M)
    return WindowRate(rp, ρ, reduce(+, ρ), _window_second(order, family, plan, M))
end

# The plans of `kernels` (each a `(kernel, multiplicity)` pair, or a kernel of multiplicity 1) and their
# `c M_p/M`.
function _window_terms(plan, kernels::Tuple, M)
    ps = _rate_plans(plan, map(_term_kernel, kernels))
    ρ = map((p, k) -> (Mp = window_mass!(_mask_shaped(plan), p); Mp .= _term_multiplicity(k) .* _normalized.(Mp, M)),
            ps, kernels)
    return ps, ρ
end

@inline _term_kernel(k::Kernels.AbstractFilterKernel) = k
@inline _term_kernel(k::Tuple) = first(k)
@inline _term_multiplicity(::Kernels.AbstractFilterKernel) = 1
@inline _term_multiplicity(k::Tuple) = last(k)

_window_second(::ScaleRates, _family, _plan, _M) = nothing
function _window_second(::ScaleCurvatures, family, plan, M)
    ps, ρ = _window_terms(plan, _curvature_kernels(family, plan.kernel, plan.grid), M)
    return WindowSecond(ps, ρ, reduce(+, ρ))
end

function _scale_rate(::SeriesRatio, plan::PhysicalFilterPlan, order::AbstractScaleRates)
    rate = rate_plan(plan, plan.kernel, plan.scale)
    B = filter_synthesize!(_mask_shaped(plan), mass_analysis(plan), rate)
    B .*= plan.scale
    F = ntuple(_ -> analyze_buffer(plan, _mask_shaped(plan)), Val(3))
    return SumRate(plan, rate, B, F, plan.scale, _sum_second(order, plan, B))
end

_sum_second(::ScaleRates, _plan, _B) = nothing
function _sum_second(::ScaleCurvatures, plan, B)
    rate = rate_plan(plan, Kernels.TransferRate(plan.kernel), plan.scale)
    B2 = filter_synthesize!(_mask_shaped(plan), mass_analysis(plan), rate)
    B2 .= plan.scale^2 .* B2 .+ B
    return SumSecond(rate, B2)
end

# Whether a footprint weighs cells by a radial kernel (one rate filter) or by a separable kernel's per-axis tables
# (one per axis).
struct _RadialRate end
struct _AxisRates end
_weight_family(_, _) = _RadialRate()
_weight_family(::Union{SeparableFootprint, SeparableFootprintND}, _) = _AxisRates()

_rate_kernels(::_RadialRate, kernel, _) = (Kernels.ScaleRate(kernel),)
_rate_kernels(::_AxisRates, kernel, ::FlowGeometries.Grids.StructuredGrid{T,G,N}) where {T,G,N} =
    ntuple(d -> Kernels.AxisRate(kernel, Val(d)), Val(N))

# The second derivative's terms with their multiplicities: `J` for radial weights; for a separable kernel `J_d` along
# each axis and `H_d H_e` twice for each pair.
_curvature_kernels(::_RadialRate, kernel, _) = ((Kernels.ScaleRate(Kernels.ScaleRate(kernel)), 1),)
_curvature_kernels(::_AxisRates, k, ::FlowGeometries.Grids.StructuredGrid{T,G,1}) where {T,G} =
    ((_axis_pair(k, Val(1), Val(1)), 1),)
_curvature_kernels(::_AxisRates, k, ::FlowGeometries.Grids.StructuredGrid{T,G,2}) where {T,G} =
    ((_axis_pair(k, Val(1), Val(1)), 1), (_axis_pair(k, Val(1), Val(2)), 2), (_axis_pair(k, Val(2), Val(2)), 1))
_curvature_kernels(::_AxisRates, k, ::FlowGeometries.Grids.StructuredGrid{T,G,3}) where {T,G} =
    ((_axis_pair(k, Val(1), Val(1)), 1), (_axis_pair(k, Val(1), Val(2)), 2), (_axis_pair(k, Val(1), Val(3)), 2),
     (_axis_pair(k, Val(2), Val(2)), 1), (_axis_pair(k, Val(2), Val(3)), 2), (_axis_pair(k, Val(3), Val(3)), 1))

@inline _axis_pair(k, d::Val, e::Val) = Kernels.AxisRate(Kernels.AxisRate(k, d), e)

# The engine a footprint is, so a rate plan takes the one its plan did.
_engine_of(_) = FootprintEngine()
_engine_of(::BinnedTopHatPlan) = BinnedTopHatEngine()

# The host footprint a backend's resident form wraps.
_host_footprint(fp) = fp

# A separable footprint's pass buffers, which its rate plans' builds and applies share; host arrays only.
_pass_buffers(_) = nothing
_pass_buffers(fp::SeparableFootprint) = _host_pair(fp.masked_input, fp.row_pass)
_pass_buffers(fp::SeparableFootprintND) = _host_pair(fp.masked_input, fp.scratch)
_host_pair(a::Array, b::Array) = SeparableScratch(a, b)
_host_pair(_, _) = nothing

"""
    window_mass!(m, plan::PhysicalFilterPlan) -> m

Each target's window mass, the sum of weights `plan` divides by, into `m` in the plan's memory, shaped like the
grid's mask; zero at a target the strategy leaves unfiltered. Read from the footprint where it holds it, and summed
by the engine's own loops where it forms it per target.
"""
window_mass!(m::AbstractArray, plan::PhysicalFilterPlan) = copyto!(m, _window_mass(_host_footprint(plan.footprint), plan))

_window_mass(fp::Union{FilterFootprint, FilterFootprintND}, _) = _inv_mass.(fp.invden)
_window_mass(fp::SeparableFootprint, _) = _separable_mass(fp.invrenorm, (fp.Nx_profile, fp.Ny_profile))
_window_mass(fp::SeparableFootprintND, _) = _separable_mass(fp.invrenorm, fp.profiles)

_separable_mass(inv::AbstractArray, _) = _inv_mass.(inv)
_separable_mass(::Nothing, prof::NTuple{N,AbstractVector}) where {N} =
    broadcast(*, ntuple(d -> reshape(prof[d], ntuple(e -> e == d ? length(prof[d]) : 1, Val(N))), Val(N))...)

const _PerTargetMass = Union{ScatteredFilterPlan, NDScatteredFilterPlan, NodeFilterPlan}

function _window_mass(fp::_PerTargetMass, plan::PhysicalFilterPlan{FP,G,S,K,T}) where {FP,G,S,K,T}
    dims = size(FlowGeometries.Grids.mask(plan.grid))
    m = zeros(T, dims)
    _mass_pass!(m, _MassOnly{T}(dims), fp, plan, plan.execution)
    return m
end

# On the plan's own driver where its applies run in this process's loops; serially from the host footprint where
# they run on a device or across processes.
_mass_pass!(m, unit, _, plan, ::Union{ComputationalBackends.SerialBackend, ComputationalBackends.ThreadedBackend}) =
    filter_apply!(m, unit, plan)
_mass_pass!(m, unit, fp, plan, _) = _apply_serial!(m, unit, plan.grid, fp, plan.strategy)
