# ---------------------------------------------------------------------------
# The least-squares analysis of the spin transforms
# ---------------------------------------------------------------------------
#
# With `A` the synthesis of degrees `≤ lmax` at the cells and `W` their measures, the direct, ring and NUFSHT
# analyses are the quadrature `Aᴴ W F`. `LeastSquares` minimizes `‖F − A C‖²_W` by LSMR in the inner product of
# `W`, whose adjoint is that analysis.

"""
    spin_weights(tr) -> AbstractVector

The quadrature weights of the analysis of `tr`, each cell's measure over `R²`, laid out as its field.
"""
function spin_weights end
spin_weights(tr::DirectSpinTransforms) = tr.w
spin_weights(tr::RingSpinTransforms) = vec(tr.w)

"""
    SpinFit

What a [`LeastSquares`](@ref) analysis holds over a spin transforms' arrays: `√w` laid out as the field, a field
and a coefficient array of scratch, the unit transfer, FlowTransformBindings' LSMR workspace, the stopping
rule, and the worst outcome of the fits it has run.
"""
struct SpinFit{T<:AbstractFloat, W<:AbstractArray{T}, F<:AbstractArray, C<:AbstractArray, G<:AbstractVector{T}, WS}
    sqrtw::W
    t::F
    work::C
    unit::G
    ws::WS
    rtol::T
    maxiter::Int
    status::Base.RefValue{AnalysisStatus{T}}
end

# The synthesis `A` and the quadrature `Aᴴ W` of the transforms `tr` at spin `s`, as an LSMR operator on one
# column whose range carries the inner product of `W`.
struct _SpinFitOperator{TR<:AbstractSpinTransforms, SF<:Union{Nothing,SpinFit}}
    tr::TR
    fit::SF
    s::Int
end

_zeroed(a::AbstractArray) = fill!(a, zero(eltype(a)))

FlowTransformBindings.lsmr_ncolumns(::_SpinFitOperator) = 1
FlowTransformBindings.lsmr_allocate_domain(op::_SpinFitOperator) = _zeroed(spin_coefficients(op.tr))
FlowTransformBindings.lsmr_allocate_range(op::_SpinFitOperator) = _zeroed(spin_field(op.tr))

function FlowTransformBindings.lsmr_forward!(u, op::_SpinFitOperator{TR,<:SpinFit}, v, c, n) where {TR}
    fit = op.fit
    spin_synthesize!(fit.t, v, fit.work, fit.unit, op.tr, op.s)
    return FlowTransformBindings.colxpby!(u, fit.t, c, n, 1)
end

function FlowTransformBindings.lsmr_adjoint!(v, op::_SpinFitOperator{TR,<:SpinFit}, u, c, n) where {TR}
    fit = op.fit
    spin_analyze!(fit.work, u, op.tr, op.s)
    return FlowTransformBindings.colxpby!(v, fit.work, c, n, 1)
end

function FlowTransformBindings.lsmr_range_norm2!(out, op::_SpinFitOperator{TR,<:SpinFit}, u, n) where {TR}
    t = op.fit.t
    t .= op.fit.sqrtw .* u
    return FlowTransformBindings.colnorm2!(out, t, n, 1)
end

function SpinFit(tr::AbstractSpinTransforms, a::LeastSquares)
    w = spin_weights(tr)
    T = eltype(w)
    rtol = a.rtol === nothing ? max(T(1e-12), 100 * eps(T)) : T(a.rtol)
    ws = FlowTransformBindings.LSMRWorkspace(_SpinFitOperator(tr, nothing, 0))
    unit = fill!(spin_transfer_buffer(tr), one(T))
    return SpinFit(sqrt.(w), _zeroed(spin_field(tr)), _zeroed(spin_coefficients(tr)), unit, ws, rtol, a.maxiter,
                   Base.RefValue(_exact_status(T)))
end

"""
    spin_fit!(C, F, tr, fit, s) -> C

The spin-`s` coefficients of the field `F` into `C`: the quadrature `spin_analyze!` ([`AbstractSpinTransforms`](@ref))
where `fit` is `nothing`,
the least-squares fit of a [`SpinFit`](@ref) otherwise, its outcome folded into the fit's status.
"""
spin_fit!(C, F, tr::AbstractSpinTransforms, ::Nothing, s::Int) = spin_analyze!(C, F, tr, s)

function spin_fit!(C, F, tr::AbstractSpinTransforms, fit::SpinFit{T}, s::Int) where {T}
    r = FlowTransformBindings.lsmr!(C, _SpinFitOperator(tr, fit, s), F, fit.ws; maxiter = fit.maxiter, rtol = fit.rtol)
    fit.status[] = _combine_status(fit.status[], AnalysisStatus{T}((r.converged, r.iterations, T(r.residual))))
    return C
end

analysis_status(fit::SpinFit) = fit.status[]

"""
    spin_fit(analysis, grid, tr) -> SpinFit or nothing

The fit [`GeneralizedConvolution`](@ref)'s `analysis` takes on `grid` with the transforms `tr`, `nothing` for the
quadrature; a transform whose analysis is exact on its grid adds a method returning `nothing`.
"""
spin_fit(a::AbstractAnalysis, grid::FlowGeometries.Grids.AbstractGrid, tr::AbstractSpinTransforms) = _fit_for(a, grid, tr)

_fit_for(::Quadrature, _grid, _tr) = nothing
_fit_for(a::LeastSquares, _grid, tr) = SpinFit(tr, a)
_fit_for(a::AutoAnalysis, grid, tr) = _fit_for(resolved_analysis(a, grid), grid, tr)

"""
    resolved_analysis(analysis, grid) -> Quadrature or LeastSquares

`analysis` itself, or for [`AutoAnalysis`](@ref) the one `grid`'s sampling takes.
"""
resolved_analysis(a::AbstractAnalysis, _grid) = a
resolved_analysis(::AutoAnalysis, grid) = _auto_analysis(FlowGeometries.Grids.sampling(grid))

_auto_analysis(::Nothing) = Quadrature()
_auto_analysis(::FlowGeometries.SphericalSampling.AbstractGaussLegendreSampling) = Quadrature()
_auto_analysis(::FlowGeometries.SphericalSampling.AbstractReducedGaussianSampling) = Quadrature()
_auto_analysis(::FlowGeometries.SphericalSampling.AbstractLatLonSampling) = Quadrature()
_auto_analysis(::FlowGeometries.SphericalSampling.AbstractScatteredSphericalSampling) = Quadrature()
_auto_analysis(::FlowGeometries.SphericalSampling.AbstractSphericalSampling) = LeastSquares()
