# ---------------------------------------------------------------------------
# Reusable filter plans
# ---------------------------------------------------------------------------

"""
    PhysicalFilterPlan

A physical-space filter plan: the precomputed footprint, with the grid, mask strategy, kernel, scale
and resolved execution backend it was built for. `kernel` and `scale` are passed to the backend hooks,
which take them positionally; the footprint is never rebuilt from them.
"""
struct PhysicalFilterPlan{FP, G<:FlowGeometries.Grids.AbstractGrid, S<:AbstractMaskStrategy, K<:Kernels.AbstractFilterKernel, T<:AbstractFloat, B<:ComputationalBackends.AbstractExecutionBackend} <: AbstractFilterPlan
    footprint::FP   # the engine's footprint, as `prepare_workspace` returns it for `execution`
    grid::G
    strategy::S
    kernel::K
    scale::T
    execution::B
end

"""
    AbstractEngine

The real-space engine a [`plan_filter`](@ref) builds, its `engine` keyword: [`AutoEngine`](@ref), the
default, or one named. [`PaddedFFTEngine`](@ref) and [`ZonalFFTEngine`](@ref) evaluate the footprint sum by
transforms on a rectilinear grid, [`BinnedTopHatEngine`](@ref) and [`NUFFTSumEngine`](@ref) over Cartesian
points, and [`FootprintEngine`](@ref) is the grid's own `build_footprint`.
"""
abstract type AbstractEngine end

"""
    AutoEngine()

The engine [`_select_engine`](@ref) picks, once, at plan build.
"""
struct AutoEngine <: AbstractEngine end

"""
    PaddedFFTEngine()

The footprint sum of any kernel on a uniform Cartesian grid by FFTs over the padded box.
"""
struct PaddedFFTEngine <: AbstractEngine end

"""
    ZonalFFTEngine()

The footprint sum on a sphere of uniform longitude by FFTs along longitude, for weights that depend on
the longitude offset alone.
"""
struct ZonalFFTEngine <: AbstractEngine end

"""
    BinnedTopHatEngine()

The top-hat over Cartesian points by prefix sums over the point bins.
"""
struct BinnedTopHatEngine <: AbstractEngine end

"""
    NUFFTSumEngine()

The kernel sum over Cartesian points by nonuniform FFTs, for a kernel with a transfer cutoff.
"""
struct NUFFTSumEngine <: AbstractEngine end

"""
    FootprintEngine()

The grid's own real-space footprint, `build_footprint`: the banded, separable, prefix-sum, cell-by-cell or
node engine its type selects.
"""
struct FootprintEngine <: AbstractEngine end

"""
    plan_strategy(plan) -> AbstractMaskStrategy

The mask strategy `plan` filters under. Every plan type defines a method.
"""
function plan_strategy end
plan_strategy(plan::PhysicalFilterPlan) = plan.strategy

"""
    plan_method(plan) -> AbstractFilterMethod

The method a plan evaluates: `RealSpace()` for a physical-space plan, `Spectral()` for a
transfer-function plan.
"""
plan_method(::PhysicalFilterPlan) = RealSpace()
plan_method(::AbstractFilterPlan) = Spectral()

"""
    AbstractSeriesForm

What a plan's output is as a function of position, by [`series_form`](@ref): [`NoSeries`](@ref),
[`Series`](@ref) or [`SeriesRatio`](@ref).
"""
abstract type AbstractSeriesForm end

"""
    NoSeries()

A plan whose output has no series form at hand: a real-space footprint sum.
"""
struct NoSeries <: AbstractSeriesForm end

"""
    Series()

A plan whose output is `ū(x) = Σ_k w_k Ĝ(k) F̂(k) e^{ik·x}`, `F̂` the analysis `filter_analyze!` writes
([`analyze_buffer`](@ref)).
"""
struct Series <: AbstractSeriesForm end

"""
    SeriesRatio()

A plan whose output is `ū = N/M`, `N` the field's series and `M` its window mass's, both of
[`Series`](@ref)'s form: a renormalized window.
"""
struct SeriesRatio <: AbstractSeriesForm end

"""
    series_form(plan) -> AbstractSeriesForm

What `plan`'s output is; [`filter_synthesize_derivative!`](@ref) differentiates a [`Series`](@ref) or a
[`SeriesRatio`](@ref).
"""
series_form(::AbstractFilterPlan) = NoSeries()
series_form(::AbstractMaskStrategy) = NoSeries()

# A spectral plan's form by its `Deformable` renormalization: none, or the reciprocal window mass.
_ratio_form(::Nothing) = Series()
_ratio_form(::AbstractArray) = SeriesRatio()

"""
    filter_synthesize_derivative!(out, F̂, ū, plan, d) -> out

`∂ū/∂x_d` at the plan's points, `ū` the filtered field and `F̂` its field's analysis: the synthesis of
`i k_d Ĝ F̂` for a [`Series`](@ref) plan, and for a [`SeriesRatio`](@ref) one the quotient rule,
`(∂_d N − ū ∂_d M)/M`, with `∂_d M` synthesized once at plan build. A method per plan type whose
[`series_form`](@ref) is not [`NoSeries`](@ref) and whose [`series_gradient`](@ref) is `nothing`; a sphere's
plans differentiate through [`filter_synthesize_gradient!`](@ref).
"""
function filter_synthesize_derivative! end

"""
    series_gradient(plan) -> gradient state or nothing

What [`filter_synthesize_gradient!`](@ref) reads beyond `plan` itself: on a sphere the spin-1 synthesis of
the surface gradient, built here so that a plan used only to filter holds none; `nothing` for a plan whose
[`filter_synthesize_derivative!`](@ref) needs nothing more.
"""
series_gradient(_) = nothing

"""
    filter_synthesize_gradient!(g1, g2, F̂, ū, plan, gradient) -> (g1, g2)

`∂ū/∂x_1` and `∂ū/∂x_2` at the plan's points, as [`filter_synthesize_derivative!`](@ref) takes each;
`gradient` is [`series_gradient(plan)`](@ref series_gradient). On a sphere the east and north components per
unit distance, from one spin-1 synthesis of `ðū = ∂_n ū − i∂_e ū`.
"""
function filter_synthesize_gradient!(g1, g2, F̂, ū, plan, ::Nothing)
    filter_synthesize_derivative!(g1, F̂, ū, plan, 1)
    filter_synthesize_derivative!(g2, F̂, ū, plan, 2)
    return g1, g2
end

"""
    rate_plan(plan, kernel, scale) -> plan

`plan` with its transfer replaced by `∂_ℓĜ` (`Kernels.TransferRate(kernel)`) at `scale`, everything else shared:
its synthesis of an analysis `plan` writes is `∂_ℓN/M`, with `M` the window mass a renormalized `plan` divides
by (`1` otherwise), so `∂_ℓū = synth(F̂) − ū · synth(M̂)`, `M̂` the analysis of a field of ones. A method per
spectral plan type; [`rate_available`](@ref) answers where the kernel has the derivative.
"""
function rate_plan end

"""
    rate_available(plan, kernel) -> Bool

Whether [`rate_plan`](@ref) serves `plan` with `kernel`: a spectral plan whose transfer the kernel differentiates
in its width (`Kernels.has_transfer_rate` on the plane, `Kernels.has_degree_rate` on a sphere).
"""
rate_available(_plan, _kernel) = false

"""
    curvature_available(plan, kernel) -> Bool

Whether `plan`'s filter has a second derivative in its width with `kernel`: [`rate_available`](@ref) of the
first derivative's kernel, `Kernels.TransferRate(kernel)` for a spectral plan or a kernel sum and
`Kernels.ScaleRate(kernel)` for a footprint sum.
"""
curvature_available(plan, kernel::Kernels.AbstractFilterKernel) = rate_available(plan, Kernels.TransferRate(kernel))

"""
    AbstractScaleRates

[`plan_filter_sweep`](@ref)'s `scale_rates`: [`ScaleRates`](@ref) to build each real-space plan's
[`scale_rate`](@ref) with the family, [`ScaleCurvatures`](@ref) to build its second derivative as well,
[`NoScaleRates`](@ref) (the default) to leave them to the sweep.
"""
abstract type AbstractScaleRates end

"""
    NoScaleRates()

A family holding no [`scale_rate`](@ref)s; a sweep's exact filtering spectrum builds them per call.
"""
struct NoScaleRates <: AbstractScaleRates end

"""
    ScaleRates()

A family holding each plan's [`scale_rate`](@ref), built once with it. Every plan must be a real-space plan
whose kernel has a derivative in its width ([`rate_available`](@ref)).
"""
struct ScaleRates <: AbstractScaleRates end

"""
    ScaleCurvatures()

A family holding each plan's [`scale_rate`](@ref) with its second derivative in the width, which a sweep's
spectral slopes read. Every plan must be a real-space plan whose kernel has both
([`curvature_available`](@ref)).
"""
struct ScaleCurvatures <: AbstractScaleRates end

_family_rates(::NoScaleRates, _plans) = nothing

"""
    plan_execution(plan) -> AbstractExecutionBackend

The execution backend `plan` was built for, whose memory its inputs and outputs live in. Every plan
type defines a method.
"""
function plan_execution end
plan_execution(plan::PhysicalFilterPlan) = plan.execution

"""
    analysis_status(plan) -> (; converged, iterations, residual)
    analysis_status(workspace) -> (; converged, iterations, residual)

The outcome of the analysis of `plan`'s last apply. A plan whose analysis is a least-squares fit (NUFSHT's
over a spherical node set) passes on the fitting library's report, combined with its mask's fit: a fit that
stops short still writes its coefficients, and is reported here. Every other plan's analysis is exact,
`(true, 0, 0)`. A diagnostic workspace reports the worst of the [`LeastSquares`](@ref) fits its generalized
convolution has run.
"""
analysis_status(::AbstractFilterPlan) = AnalysisStatus{Float64}((true, 0, 0.0))

const AnalysisStatus{T} = @NamedTuple{converged::Bool, iterations::Int, residual::T}
_exact_status(::Type{T}) where {T} = AnalysisStatus{T}((true, 0, zero(T)))
_combine_status(a::AnalysisStatus{T}, b::AnalysisStatus{T}) where {T} =
    AnalysisStatus{T}((a.converged & b.converged, max(a.iterations, b.iterations), max(a.residual, b.residual)))

"""
    prepare_workspace(backend, grid, footprint) -> workspace

Backend hook [`plan_filter`](@ref) calls once; its result is stored as the plan's `footprint`. The
default returns `footprint` unchanged; a backend with its own memory, such as a GPU's device buffers,
returns the resident form its apply step reads.
"""
prepare_workspace(
    ::ComputationalBackends.AbstractExecutionBackend, ::FlowGeometries.Grids.AbstractGrid, fp,
) = fp

@inline function _validate_scale(scale::T) where {T<:AbstractFloat}
    isfinite(scale) && scale > zero(T) || throw(ArgumentError(
        "filter scale must be finite and positive, got $scale",
    ))
    return nothing
end

@noinline _unused_keywords(kwargs, method) = throw(ArgumentError(
    "plan_filter with method = $(nameof(typeof(method)))() takes no keyword " *
    "$(join(map(k -> "`$k`", collect(keys(kwargs))), ", "))",
))

"""
    plan_filter(grid, kernel, scale; mask_strategy = ZeroFill(), execution = AutoBackend(),
                method = RealSpace(), transform = AutoSpectralBackend(), cache_strategy = AlwaysCache(),
                grid_plan = nothing, scratch = nothing, engine = AutoEngine(), fft_options = (;)) -> AbstractFilterPlan

A reusable filter plan for `execution`, applied by `filter_apply!(out, field, plan)`: the real-space
footprint, or under `method = Spectral()` the transform plan. `engine` names the real-space engine
([`AbstractEngine`](@ref)); `AutoEngine()` takes the one with the fewest operations ([`_select_engine`](@ref)). A
spectral plan also takes `batch = nb`
for a trailing batch axis and, over a spherical node set, `nufft`, the NUFFT library NUFSHT runs, and
`analysis` ([`AbstractAnalysis`](@ref)); a keyword the method does not take raises an `ArgumentError`.
`grid_plan` and `scratch` are the parts a sweep shares; see [`plan_filter_sweep`](@ref). `fft_options` are
the planner keywords of the FFTs the plan makes (the spectral FFT filter, the padded and zonal FFT engines),
passed to the FFT library's `plan_*` as they stand, e.g. `(; flags = FFTW.MEASURE)` under FFTW.

A transform runs on 1 thread under `SerialBackend()` and on `Threads.nthreads()` under
`ThreadedBackend()`, `AutoBackend()` and `GPUBackend()`, in the device's memory under the last.
Under `MPIBackend(inner)` every rank holds the field and the output. A nonuniform transform (NUFFT,
NUFSHT) gives each rank a disjoint share of the points: the analysis `F_k = Σⱼ wⱼ cⱼ exp(−i k⋅xⱼ)` is a
sum over points, so the ranks' partial sums add to it exactly, and each rank evaluates the filtered
series at its own points. An FFT or FastSphericalHarmonics transform runs whole on every rank. Under
`DistributedBackend(inner)` the worker processes share the transform, a nonuniform one by blocks of
points and an FFT by blocks of columns and then of rows, each at `inner`'s thread count; a
FastSphericalHarmonics transform runs in the calling process.
"""
function plan_filter(
    grid::FlowGeometries.Grids.StructuredGrid{T,G},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    mask_strategy::AbstractMaskStrategy = ZeroFill(),
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    method::AbstractFilterMethod = RealSpace(),
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    cache_strategy::AbstractCacheStrategy = AlwaysCache(),
    grid_plan::Union{Nothing,AbstractGridPlan} = nothing,
    scratch::Union{Nothing,AbstractFilterScratch} = nothing,
    engine::AbstractEngine = AutoEngine(),
    fft_options::NamedTuple = (;),
    kwargs...,
) where {G<:FlowGeometries.Geometry.AbstractGeometry{T}} where {T<:AbstractFloat}
    _validate_scale(scale)
    if method isa Spectral
        return spectral_filter_plan(
            transform, grid, kernel, scale;
            mask_strategy = mask_strategy, execution = execution,
            grid_plan = grid_plan, scratch = scratch, _fft_keywords(fft_options)..., kwargs...,
        )
    end
    isempty(kwargs) || _unused_keywords(kwargs, method)
    resolved = _resolve_backend(execution, grid)
    _check_backend_compatible(grid, execution)
    return _physical_plan(
        resolved, engine, grid, kernel, scale, mask_strategy,
        (; mask_strategy = mask_strategy, cache_strategy = cache_strategy, grid_plan = grid_plan, scratch = scratch,
           execution = resolved, _fft_keywords(fft_options)...),
    )
end
