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

"""
    _SpectrumOutputs(rates, slopes, spectrum_map, slope_map, L)

Where an exact rate route writes a scale, from the filtered velocity `ū`, `a = ℓ∂_ℓū` and `b = ℓ²∂²_ℓū`: `rates[i] =
⟨ū·∂_ℓū⟩`, `slopes[i]` the spectrum's slope `d ln Ẽ/d ln k_ℓ = −(2⟨ū·a⟩ + ⟨a·a⟩ + ⟨ū·b⟩)/⟨ū·a⟩`, and per cell
`spectrum_map` the local spectrum `−(ℓ/L) ū·a` with `slope_map` its slope `−(2ū·a + a·a + ū·b)/(ū·a)`. A destination
left `nothing` is not computed. `i` is the scale of a sweep's result, or the slice of a batch's entries at one scale.
"""
struct _SpectrumOutputs{R<:AbstractArray, S, M, P, T<:Real}
    rates::R
    slopes::S
    spectrum_map::M
    slope_map::P
    L::T
end

_SpectrumOutputs(rates::AbstractArray) = _SpectrumOutputs(rates, nothing, nothing, nothing, 1)

# The derivative order a outputs's destinations read: the second where a slope is asked for.
@inline _rate_order(outputs::_SpectrumOutputs) = _rate_order(outputs.slopes, outputs.slope_map)
@inline _rate_order(::Nothing, ::Nothing) = Filtering.ScaleRates()
@inline _rate_order(_, _) = Filtering.ScaleCurvatures()

# Scale `s`'s slice of a map with the scales along its last axis.
@inline _scale_view(::Nothing, ::Int) = nothing
@inline _scale_view(m::AbstractArray, s::Int) = selectdim(m, ndims(m), s)

# Scale `scale`'s results from `ū`, `a`, `b` over the active cells of `og`, for the field at linear offset `off`: entry
# `i` of the outputs's per-scale vectors, and the maps `m`, `p` (the outputs's at this scale, or `nothing`).
function _store_outputs!(outputs::_SpectrumOutputs, i::Int, m, p, ū, a, b, og, off::Int, scale, total_area, bk)
    T = eltype(outputs.rates)
    S1 = _dot_integral(ū, a, og, off, bk)
    outputs.rates[i] = S1 / (T(scale) * total_area)
    _store_slope!(outputs.slopes, i, S1, ū, a, b, og, off, bk)
    _store_maps!(m, p, ū, a, b, length(og), off, T(scale) / T(outputs.L), bk)
    return nothing
end

_store_slope!(::Nothing, _i, _S1, _ū, _a, _b, _og, _off, _bk) = nothing
function _store_slope!(slopes::AbstractArray, i::Int, S1, ū, a, b, og, off::Int, bk)
    S2 = _dot_integral(a, a, og, off, bk)
    S3 = _dot_integral(ū, b, og, off, bk)
    slopes[i] = -(2S1 + S2 + S3) / S1
    return nothing
end

_store_maps!(::Nothing, _p, _ū, _a, _b, _n, _off, _c, _bk) = nothing
function _store_maps!(m::AbstractArray, p, ū, a, b, n::Int, off::Int, c, bk)
    FlowGeometries.Execution.run_indices(n, bk) do k
        i = off + k
        ua = _sumdot(ū, a, i)
        @inbounds m[i] = -c * ua
        _slope_at!(p, i, ua, ū, a, b)
    end
    return nothing
end

@inline _slope_at!(::Nothing, _i, _ua, _ū, _a, _b) = nothing
@inline _slope_at!(p::AbstractArray, i::Int, ua, ū, a, b) =
    (@inbounds p[i] = -(2ua + _sumdot(a, a, i) + _sumdot(ū, b, i)) / ua; nothing)

"""
    _SpinRate

The rate route of the generalized convolution: `a = ℓ∂_ℓū` per filtered component, synthesized with `ℓ∂_ℓĜ(n)`, and
`b = ℓ²∂²_ℓū` with `ℓ²∂²_ℓĜ(n)` where a slope is asked for (`nothing` otherwise).
"""
struct _SpinRate{A<:Tuple, B}
    a::A
    b::B
end

@inline _spin_filtered(ws, ::Nothing) = (ws.u_filt, ws.v_filt)
@inline _spin_filtered(ws, ::AbstractArray) = (ws.u_filt, ws.v_filt, ws.w_filt)

@inline _second_buffers(::Filtering.ScaleRates, _x) = nothing
@inline _second_buffers(::Filtering.ScaleCurvatures, x::Tuple) = map(similar, x)

# `a` and, where the route holds them, `b` of the slice at offset `off` under the generalized convolution, whose spin-1
# (and with `w̄`, spin-0) coefficients the filter holds.
function _spin_rate_fields!(r::_SpinRate, sf::SpinFilter, kernel, scale, off::Int)
    T = eltype(sf.dtransfer)
    d = Kernels.spectral_rate_degrees!(sf.dtransfer, kernel, scale, sf.radius, sf.quadrature)
    d .*= T(scale)
    _spin_fill!(r.a, sf, d, off)
    _spin_second!(r.b, sf, kernel, scale, off)
    return nothing
end

_spin_second!(::Nothing, _sf, _kernel, _scale, _off) = nothing
function _spin_second!(b::Tuple, sf::SpinFilter, kernel, scale, off::Int)
    T = eltype(sf.dtransfer)
    d = Kernels.spectral_rate_degrees!(sf.dtransfer, Kernels.TransferRate(kernel), scale, sf.radius, sf.quadrature)
    d .*= T(scale)^2
    _spin_fill!(b, sf, d, off)
    return nothing
end

# The syntheses with the per-degree factor `d` of the filter's spin-1 coefficients into `x[1]`, `x[2]`, and of its
# spin-0 ones of `w` into `x[3]` where there are three.
function _spin_fill!(x::Tuple, sf::SpinFilter, d, off::Int)
    _spin_synthesize!(sf, sf.coeffs[1], 1, d)
    x1, x2 = x[1], x[2]
    _spin_store!(sf) do k, z
        @inbounds x1[off + k] = real(z)
        @inbounds x2[off + k] = imag(z)
    end
    _spin_fill_w!(x, sf, d, off)
    return nothing
end

_spin_fill_w!(::NTuple{2,Any}, _sf, _d, _off) = nothing
function _spin_fill_w!(x::NTuple{3,Any}, sf::SpinFilter, d, off::Int)
    _spin_synthesize!(sf, sf.coeffs[5], 0, d)
    x3 = x[3]
    _spin_store!(sf) do k, z
        @inbounds x3[off + k] = real(z)
    end
    return nothing
end

"""
    _rate_route(scale_derivative, kernel, ws, plans, u, v, w, order) -> rate route

How a sweep takes `⟨ū·∂_ℓū⟩` per scale, with the second derivative where `order` is `Filtering.ScaleCurvatures()`,
resolved once from its plans (or the mask strategy where the workspace's generalized convolution filters): from the
generalized convolution's transfer (`_SpinRate`), from a spectral family's rate plans (`_SeriesRate`), from a
real-space family's rate filters and window masses (`_WindowRate`), or by differences across the scales
(`_DifferenceRate`).
"""
_rate_route(::DifferenceScaleDerivative, _kernel, _ws, _plans, _u, _v, _w, _order) = _DifferenceRate()
_rate_route(::AutoScaleDerivative, kernel, ws, plans, u, v, w, order) =
    _or_difference(_exact_rate(kernel, ws, plans, u, v, w, order))
_rate_route(::ExactScaleDerivative, kernel, ws, plans, u, v, w, order) =
    _or_raise(_exact_rate(kernel, ws, plans, u, v, w, order), kernel, _route_plan(plans))

struct _DifferenceRate end

"""
    _SeriesRate

The rate route of a spectral plan: `∂_ℓū = synth(F̂) − ū · synth(M̂)` through each scale's `Filtering.rate_plan`,
`M̂` the analysis of a field of ones where the plan divides by a window mass (`nothing` otherwise); buffers for
`∂_ℓū` per filtered component, for `synth(M̂)`, and for the velocity's analysis where the sweep holds none; and for
the second derivative `out2`, `dM2` (`nothing` where no slope is asked for).
"""
struct _SeriesRate{O<:Tuple, A, MH, FB<:Tuple, O2, A2}
    out::O
    dM::A
    M̂::MH
    F::FB
    out2::O2
    dM2::A2
end

# A family's plans share their method, so its first stands for it.
@inline _route_plan(plans::AbstractVector) = first(plans)
@inline _route_plan(strategy) = strategy

# Whether a plan's filter has the derivative `order` asks for.
@inline _has_order(::Filtering.ScaleRates, plan, kernel) = Filtering.rate_available(plan, kernel)
@inline _has_order(::Filtering.ScaleCurvatures, plan, kernel) =
    Filtering.rate_available(plan, kernel) && Filtering.curvature_available(plan, kernel)
@inline _has_degree_order(::Filtering.ScaleRates, kernel) = Kernels.has_degree_rate(kernel)
@inline _has_degree_order(::Filtering.ScaleCurvatures, kernel) =
    Kernels.has_degree_rate(kernel) && Kernels.has_degree_rate(Kernels.TransferRate(kernel))

_exact_rate(kernel, ws, plans, u, v, w, order) =
    _exact_rate(kernel, _spin_of(ws.spin, _route_plan(plans)), ws, plans, u, v, w, order)
function _exact_rate(kernel, ::SpinFilter, ws, _plans, _u, _v, w, order)
    _has_degree_order(order, kernel) || return nothing
    filt = _spin_filtered(ws, w)
    return _SpinRate(map(similar, filt), _second_buffers(order, filt))
end
_exact_rate(_kernel, ::Nothing, _ws, ::Filtering.AbstractMaskStrategy, _u, _v, _w, _order) = nothing
_exact_rate(kernel, ::Nothing, ws, plans::AbstractVector, u, v, w, order) =
    _family_rate(Filtering.plan_method(first(plans)), kernel, ws, plans, u, v, w, order)

_family_rate(::Filtering.Spectral, kernel, ws, plans, u, v, w, order) =
    _has_order(order, first(plans), kernel) ? _series_rate(ws, first(plans), u, v, w, order) : nothing
_family_rate(::Filtering.RealSpace, kernel, ws, plans, u, v, w, order) =
    all(p -> _has_order(order, p, kernel), plans) ? _window_rate(ws, plans, kernel, u, v, w, order) : nothing

function _series_rate(ws, plan, u, v, w, order)
    ins, filt, _, _ = _rate_fields(ws, plan, u, v, w)
    one_field = fill!(similar(first(filt)), one(eltype(first(filt))))
    M̂ = _mask_analysis(_ratio_form(plan), plan, one_field)
    F = map(f -> Filtering.analyze_buffer(plan, f), ins)
    return _SeriesRate(map(similar, filt), similar(first(filt)), M̂, F, _second_buffers(order, filt),
                       _second_mass(order, M̂, first(filt)))
end

@inline _second_mass(::Filtering.ScaleRates, _M̂, _x) = nothing
@inline _second_mass(::Filtering.ScaleCurvatures, ::Nothing, ::AbstractArray) = nothing
@inline _second_mass(::Filtering.ScaleCurvatures, _M̂, x::AbstractArray) = similar(x)

@inline _ratio_form(plan) = Filtering.series_form(plan)
_mask_analysis(::Filtering.Series, _plan, _ones) = nothing
_mask_analysis(::Filtering.SeriesRatio, plan, ones) =
    Filtering.filter_analyze!(Filtering.analyze_buffer(plan, ones), ones, plan)

"""
    _WindowRate

The rate route of a real-space family: each scale's `Filtering.scale_rate`, giving `ℓ∂_ℓū = A − ū·B`, and buffers for
one rate filter's output per analyzed field, for `A` per filtered component, and for the second derivative's `A`
(`nothing` where no slope is asked for).
"""
struct _WindowRate{P<:AbstractVector, O<:Tuple, A<:Tuple, A2}
    parts::P
    out::O
    acc::A
    acc2::A2
end

function _window_rate(ws, plans, _kernel, u, v, w, order)
    _, filt, dot, _ = _rate_fields(ws, first(plans), u, v, w)
    return _WindowRate(_scale_rates(plans, order), map(similar, filt), map(similar, dot), _second_buffers(order, dot))
end

# The family's scale rates where it holds them to `order`, built here otherwise.
_scale_rates(plans::Filtering.FilterPlanFamily, order) = _held_rates(plans.rates, plans.rate_order, plans, order)
_scale_rates(plans::AbstractVector, order) = map(p -> Filtering.scale_rate(p, order), plans)
_held_rates(rates::AbstractVector, ::Filtering.ScaleCurvatures, _plans, _order) = rates
_held_rates(rates::AbstractVector, ::Filtering.ScaleRates, _plans, ::Filtering.ScaleRates) = rates
_held_rates(_rates, _held, plans, order) = map(p -> Filtering.scale_rate(p, order), plans.plans)

# `A` of `ℓ∂_ℓū = A − ū·B` into `acc`, in the frame of the filtered components `dot`; returns `B`. `out` holds a rate
# filter's output per analyzed field, turned local by the frames `fr` where those fields are ambient.
function _rate_terms!(acc, out, part::Filtering.WindowRate, ins, fr, dot, b)
    foreach(a -> fill!(a, zero(eltype(a))), acc)
    map((p, ρ) -> _add_rate!(acc, out, p, ρ, ins, fr, dot, b), part.plans, part.ρ)
    return part.B
end

function _add_rate!(acc, out, p, ρ, ins, fr, dot, b)
    Filtering.filter_apply_batch!(out, ins, p)
    foreach((a, o) -> (a .+= ρ .* o), acc, _rate_frame!(out, fr, dot, b))
    return nothing
end

function _rate_terms!(acc, out, part::Filtering.SumRate, ins::NTuple{N,Any}, fr, dot, b) where {N}
    F = ntuple(i -> part.F[i], Val(N))
    foreach((f̂, x) -> Filtering.filter_analyze!(f̂, x, part.plan), F, ins)
    foreach((o, f̂) -> Filtering.filter_synthesize!(o, f̂, part.rate), out, F)
    foreach((a, o) -> (a .= part.scale .* o), acc, _rate_frame!(out, fr, dot, b))
    return part.B
end

# `a = ℓ∂_ℓū` into `r.acc` for scale `s` and, where the route holds `r.acc2`, `b = ℓ²∂²_ℓū` there; returns the filtered
# velocity they pair with, `a` and `b`.
function _window_rate_fields!(r::_WindowRate, s::Int, ws, plan, u, v, w)
    ins, _, dot, fr = _rate_fields(ws, plan, u, v, w)
    part = r.parts[s]
    bk = ws.at.execution
    _part_rate!(r.acc, r.out, part, ins, fr, dot, bk)
    _part_second!(r.acc2, r.out, part, ins, fr, dot, r.acc, bk)
    return dot, r.acc, r.acc2
end

function _part_rate!(acc, out, part, ins, fr, dot, b)
    B = _rate_terms!(acc, out, part, ins, fr, dot, b)
    foreach((a, x) -> (a .-= x .* B), acc, dot)
    return nothing
end

# `D²ū = A₂ − ū·B₂ − 2a·B` with `D = ℓ∂_ℓ` and `B` the first derivative's, then `b = D²ū − a`.
_part_second!(::Nothing, _out, _part, _ins, _fr, _dot, _a, _b) = nothing
function _part_second!(acc2::Tuple, out, part, ins, fr, dot, a, bk)
    B2 = _second_terms!(acc2, out, part, part.second, ins, fr, dot, a, bk)
    B = part.B
    foreach((x2, x, d) -> (x2 .-= d .* B2 .+ 2 .* x .* B .+ x), acc2, a, dot)
    return nothing
end

# `A₂` into `acc2`, returning `B₂`: the second-order filters' `Σ_p ρ_p ū_p`; for the kernel sum `ℓ²∂²_ℓN/M + ℓ∂_ℓN/M`
# from the analyses the first derivative took.
function _second_terms!(acc2, out, ::Filtering.WindowRate, sec::Filtering.WindowSecond, ins, fr, dot, _a, bk)
    foreach(x -> fill!(x, zero(eltype(x))), acc2)
    map((p, ρ) -> _add_rate!(acc2, out, p, ρ, ins, fr, dot, bk), sec.plans, sec.ρ)
    return sec.B
end

function _second_terms!(acc2, out, part::Filtering.SumRate, sec::Filtering.SumSecond, ins::NTuple{N,Any}, fr, dot, a,
                        bk) where {N}
    F = ntuple(i -> part.F[i], Val(N))
    foreach((o, f̂) -> Filtering.filter_synthesize!(o, f̂, sec.rate), out, F)
    loc = _rate_frame!(out, fr, dot, bk)
    c = part.scale^2
    foreach((x2, o, x, d) -> (x2 .= c .* o .+ x .+ d .* part.B), acc2, loc, a, dot)
    return sec.B
end

"""
    _rate_fields(ws, plan, u, v, w) -> (ins, filtered, local, frames)

What a rate reads on a workspace: the fields the plan analyzes, their filtered values, the filtered velocity in
the local frame that `Ē` integrates, and the frames that turn ambient Cartesian components local (`nothing` on a
plane, where the analyzed fields are the velocity's own).
"""
_rate_fields(ws::ΠWorkspace, plan, u, v, w) = _rate_fields(ws, plan, u, v, w, ws.scratch2)
_rate_fields(ws::ΠWorkspace, _plan, u, ::Nothing, ::Nothing, ::Nothing) = ((u,), (ws.u_filt,), (ws.u_filt,), nothing)
_rate_fields(ws::ΠWorkspace, _plan, u, v, w, ::Nothing) =
    (_velocity_ins(u, v, w, w !== nothing), _velocity_outs(ws, w !== nothing), _velocity_outs(ws, w !== nothing), nothing)
_rate_fields(ws::ΠWorkspace, plan, _u, _v, w, ::AbstractArray) =
    ((ws.ux, ws.uy, ws.uz), (ws.ux_filt, ws.uy_filt, ws.uz_filt), _velocity_outs(ws, w !== nothing),
     _output_frames(ws.at, plan))
_rate_fields(ws::EnergyWorkspace, plan, u, v, w) = _rate_fields(ws, plan, u, v, w, ws.ux)
_rate_fields(ws::EnergyWorkspace, _plan, u, v, w, ::Nothing) =
    (_velocity_ins(u, v, w, w !== nothing), _energy_outs(ws, w), _energy_outs(ws, w), nothing)
function _rate_fields(ws::EnergyWorkspace, plan, _u, _v, w, ::AbstractArray)
    pin, pout = _ambient_buffers(ws)
    return (pin, pout, _energy_outs(ws, w), _output_frames(ws.at, plan))
end

@inline _energy_outs(ws, ::Nothing) = (ws.u_filt, ws.v_filt)
@inline _energy_outs(ws, ::AbstractArray) = (ws.u_filt, ws.v_filt, ws.w_filt)

@inline _or_difference(::Nothing) = _DifferenceRate()
@inline _or_difference(route) = route
@inline _or_raise(::Nothing, kernel, plan) = _no_exact_rate(kernel, plan)
@inline _or_raise(route, _, _) = route

@noinline _no_exact_rate(kernel, plan) = throw(ArgumentError(
    "scale_derivative = ExactScaleDerivative() differentiates the filter in its width, and " *
    "$(nameof(typeof(kernel))) has no such derivative on $(nameof(typeof(plan)))'s route " *
    "(Kernels.has_degree_rate, has_transfer_rate, has_profile_rate): pass `DifferenceScaleDerivative()`",
))

# Scale `scale`, the sweep's `s`-th, into the outputs after its flux, `ws` holding the filtered velocity of one slice;
# `analyzed` is the sweep's `analyze_sweep` result. On the difference route a spectrum map holds each cell's `½|ū|²`
# until the sweep ends.
_store_rate!(::Nothing, _s, ::_DifferenceRate, _ws, _plan, _kernel, _scale, _u, _v, _w, _total_area, _analyzed) = nothing
function _store_rate!(outputs::_SpectrumOutputs, s::Int, ::_DifferenceRate, ws, _plan, _kernel, _scale, _u, v, w, _total_area,
                      _analyzed)
    filt = _filtered_velocity(ws, v, w)
    _energy_map!(_scale_view(outputs.spectrum_map, s), filt, length(first(filt)), ws.at.execution)
    return nothing
end

# The filtered velocity a workspace holds: one component on a line.
@inline _filtered_velocity(ws, ::Nothing, _w) = (ws.u_filt,)
@inline _filtered_velocity(ws, _v, w) = _spin_filtered(ws, w)

function _store_rate!(outputs::_SpectrumOutputs, s::Int, r::_SpinRate, ws, plan, kernel, scale, _u, _v, w, total_area, _analyzed)
    _spin_rate_fields!(r, _spin_of(ws.spin, plan), kernel, scale, 0)
    _store_outputs!(outputs, s, _scale_view(outputs.spectrum_map, s), _scale_view(outputs.slope_map, s), _spin_filtered(ws, w), r.a,
                 r.b, ws.at.unmasked, 0, scale, total_area, ws.at.execution)
    return nothing
end

function _store_rate!(outputs::_SpectrumOutputs, s::Int, r::_SeriesRate, ws, plan, kernel, scale, u, v, w, total_area, analyzed)
    dot, a, b, og = _series_rate_fields!(r, ws, plan, kernel, scale, u, v, w,
                                         _rate_spectra!(r, analyzed, ws, plan, u, v, w, s))
    _store_outputs!(outputs, s, _scale_view(outputs.spectrum_map, s), _scale_view(outputs.slope_map, s), dot, a, b, og, 0, scale,
                 total_area, ws.at.execution)
    return nothing
end

function _store_rate!(outputs::_SpectrumOutputs, s::Int, r::_WindowRate, ws, plan, _kernel, scale, u, v, w, total_area, _analyzed)
    dot, a, b = _window_rate_fields!(r, s, ws, plan, u, v, w)
    _store_outputs!(outputs, s, _scale_view(outputs.spectrum_map, s), _scale_view(outputs.slope_map, s), dot, a, b,
                 _output(ws.at, plan), 0, scale, total_area, ws.at.execution)
    return nothing
end

# `½|ū|²` per cell into `m` over the first `n` cells of the filtered components `x`.
_energy_map!(::Nothing, _x, _n, _bk) = nothing
function _energy_map!(m::AbstractArray, x::Tuple, n::Int, bk)
    FlowGeometries.Execution.run_indices(n, bk) do k
        @inbounds m[k] = _sumsq(x, k) / 2
    end
    return nothing
end

# `a = ℓ∂_ℓū` in the local frame for the rate plan of `scale`, from the analyses `F`, and the filtered velocity it pairs
# with: `synth(F̂)` per analyzed field, turned local where those fields are ambient, less `ū · synth(M̂)` under a
# renormalized window (a scalar factor, so it commutes with the turn), times `ℓ`; where the route holds `out2`,
# `b = ℓ²∂²_ℓū` by the same with `∂²_ℓĜ` and the quotient rule's `−2∂_ℓū·∂_ℓM/M`.
function _series_rate_fields!(r::_SeriesRate, ws, plan, kernel, scale, u, v, w, F)
    rp = Filtering.rate_plan(plan, kernel, scale)
    _, _, dot, fr = _rate_fields(ws, plan, u, v, w)
    bk = ws.at.execution
    foreach((o, f̂) -> Filtering.filter_synthesize!(o, f̂, rp), r.out, F)
    loc = _rate_frame!(r.out, fr, dot, bk)
    _less_mass_rate!(loc, dot, r, rp, r.M̂)
    b = _series_second!(r.out2, r, plan, kernel, scale, F, fr, dot, loc, bk)
    foreach(x -> (x .*= scale), loc)
    return dot, loc, b, _output(ws.at, plan)
end

_series_second!(::Nothing, _r, _plan, _kernel, _scale, _F, _fr, _dot, _loc, _bk) = nothing
function _series_second!(out2::Tuple, r::_SeriesRate, plan, kernel, scale, F, fr, dot, loc, bk)
    rp2 = Filtering.rate_plan(plan, Kernels.TransferRate(kernel), scale)
    foreach((o, f̂) -> Filtering.filter_synthesize!(o, f̂, rp2), out2, F)
    loc2 = _rate_frame!(out2, fr, dot, bk)
    _less_mass_second!(loc2, loc, dot, r, rp2, r.M̂)
    c = scale^2
    foreach(x -> (x .*= c), loc2)
    return loc2
end

_less_mass_second!(_loc2, _loc, _dot, ::_SeriesRate, _rp2, ::Nothing) = nothing
function _less_mass_second!(loc2, loc, dot, r::_SeriesRate, rp2, M̂)
    Filtering.filter_synthesize!(r.dM2, M̂, rp2)
    foreach((o, x, d) -> (o .-= x .* r.dM2 .+ 2 .* d .* r.dM), loc2, dot, loc)
    return nothing
end

@inline _rate_frame!(out, ::Nothing, _dot, _b) = out
@inline _rate_frame!(out, fr, dot, b) = _rate_local!(out, fr, dot, b)
function _rate_local!(out::NTuple{3,Any}, fr, dot::Tuple, b)
    FlowGeometries.Operators.to_local!(out[1], out[2], out[3], out[1], out[2], out[3], fr; execution = b)
    return ntuple(i -> out[i], Val(length(dot)))
end

_less_mass_rate!(_loc, _dot, ::_SeriesRate, _rp, ::Nothing) = nothing
function _less_mass_rate!(loc, dot, r::_SeriesRate, rp, M̂)
    Filtering.filter_synthesize!(r.dM, M̂, rp)
    foreach((o, x) -> (o .-= x .* r.dM), loc, dot)
    return nothing
end

# The velocity's analyses: the sweep's spectra on a plane; the ambient components analyzed at the sweep's first
# scale where a `SphericalAnalysis` holds them fixed; otherwise analyzed per scale.
_rate_spectra!(_r, a::NamedTuple, _ws, _plan, _u, _v, _w, _s) = a.velocity
_rate_spectra!(r, ::SphericalAnalysis, ws, plan, u, v, w, s::Int) =
    s == 1 ? _analyze_rate_fields!(r, ws, plan, u, v, w) : r.F
_rate_spectra!(r, _analyzed, ws, plan, u, v, w, _s) = _analyze_rate_fields!(r, ws, plan, u, v, w)

function _analyze_rate_fields!(r::_SeriesRate, ws, plan, u, v, w)
    ins = first(_rate_fields(ws, plan, u, v, w))
    foreach((f̂, x) -> Filtering.filter_analyze!(f̂, x, plan), r.F, ins)
    return r.F
end

# `Σ a·b · measure` over the active cells of `og`, pairing the components of `a` and `b` in order, for the batch
# element at linear offset `off`.
function _dot_integral(a::Tuple{AbstractArray{T},Vararg{AbstractArray{T}}}, b::Tuple, og::FlowGeometries.Grids.AbstractGrid,
                       off::Int, bk) where {T}
    msk = FlowGeometries.Grids.mask(og)
    m = FlowGeometries.Grids.measure(og)
    return FlowGeometries.Execution.reduce_indices(+, zero(T), length(og), bk) do i
        @inbounds _cell_on(msk, i) ? _sumdot(a, b, off + i) * T(m[i]) : zero(T)
    end
end

@inline _sumdot(a::NTuple{1,AbstractArray}, b::NTuple{1,AbstractArray}, k::Int) = @inbounds a[1][k] * b[1][k]
@inline _sumdot(a::NTuple{2,AbstractArray}, b::NTuple{2,AbstractArray}, k::Int) = @inbounds a[1][k] * b[1][k] + a[2][k] * b[2][k]
@inline _sumdot(a::NTuple{3,AbstractArray}, b::NTuple{3,AbstractArray}, k::Int) =
    @inbounds (a[1][k] * b[1][k] + a[2][k] * b[2][k]) + a[3][k] * b[3][k]

@inline _filtered_w(_ws, ::Nothing) = nothing
@inline _filtered_w(ws, ::AbstractArray) = ws.w_filt

# The same per slice of a batch-sized workspace into a outputs of one scale's entries, one per slice in linear order, its
# maps that scale's. The flux leaves the spin coefficients of its last slice alone, so each slice's are analyzed again
# from `u`, `v`, `w`; a spectral plan analyzes and synthesizes the whole batch at once.
function _store_rates!(outputs::_SpectrumOutputs, _s, ::_DifferenceRate, ws, _plan, _kernel, _scale, _u, v, w, _total_area)
    filt = _filtered_velocity(ws, v, w)
    _energy_map!(outputs.spectrum_map, filt, length(first(filt)), ws.at.execution)
    return nothing
end

function _store_rates!(outputs::_SpectrumOutputs, _s, r::_SeriesRate, ws, plan, kernel, scale, u, v, w, total_area)
    dot, a, b, og = _series_rate_fields!(r, ws, plan, kernel, scale, u, v, w,
                                         _analyze_rate_fields!(r, ws, plan, u, v, w))
    _slice_store!(outputs, dot, a, b, og, scale, ws, total_area)
    return nothing
end

function _store_rates!(outputs::_SpectrumOutputs, s::Int, r::_WindowRate, ws, plan, _kernel, scale, u, v, w, total_area)
    dot, a, b = _window_rate_fields!(r, s, ws, plan, u, v, w)
    _slice_store!(outputs, dot, a, b, _output(ws.at, plan), scale, ws, total_area)
    return nothing
end

function _slice_store!(outputs, dot, a, b, og, scale, ws, total_area)
    n = length(og)
    for (t, off) in enumerate(0:n:(length(first(dot)) - n))
        _store_outputs!(outputs, t, outputs.spectrum_map, outputs.slope_map, dot, a, b, og, off, scale, total_area,
                     ws.at.execution)
    end
    return nothing
end

function _store_rates!(outputs::_SpectrumOutputs, _s, r::_SpinRate, ws, plan, kernel, scale, u, v, w, total_area)
    sf = _spin_of(ws.spin, plan)
    n = sf.ncells
    filt = _spin_filtered(ws, w)
    for (t, off) in enumerate(0:n:(length(u) - n))
        _spin_analyze!(sf.coeffs[1], sf, 1, true, k -> @inbounds(complex(u[off + k], v[off + k])))
        _rate_vertical_analysis!(sf, w, off)
        _spin_rate_fields!(r, sf, kernel, scale, off)
        _store_outputs!(outputs, t, outputs.spectrum_map, outputs.slope_map, filt, r.a, r.b, ws.at.unmasked, off, scale, total_area,
                     ws.at.execution)
    end
    return nothing
end

_rate_vertical_analysis!(_sf, ::Nothing, _off) = nothing
_rate_vertical_analysis!(sf::SpinFilter, w::AbstractArray, off::Int) =
    _spin_analyze!(sf.coeffs[5], sf, 0, true, k -> @inbounds(complex(w[off + k])))

"""
    filtering_density!(g, C, k, policy, rate, scales, L) -> g

The filtering spectral density at each scale: `NaN` under [`NoSpectrum`](@ref); otherwise `dC/dk` by
[`spectral_density!`](@ref) for the difference route, and `Ẽ = −(ℓ²/L)⟨ū·∂_ℓū⟩` from the rates `g` holds for an
exact one.
"""
filtering_density!(g, _C, _k, ::NoSpectrum, _rate, _scales, _L) = fill!(g, convert(eltype(g), NaN))
filtering_density!(g, C, k, ::AbstractSpectrumPolicy, rate, scales, L) = _density!(g, C, k, rate, scales, L)
_density!(g, C, k, ::_DifferenceRate, _scales, _L) = spectral_density!(g, C, k)
function _density!(g, _C, _k, _rate, scales, L)
    T = eltype(g)
    @. g = -T(scales)^2 / T(L) * g
    return g
end

"""
    finish_slopes!(outputs, g, k, policy, rate) -> outputs

A outputs's slopes and maps once its density `g` over the wavenumbers `k` is filled: `NaN` under
[`NoSpectrum`](@ref); on the difference route the slope of `ln Ẽ` in `ln k` by the three-point quadratic of
[`spectral_density!`](@ref), and the spectrum map as `dC/dk` of each cell's `½|ū|²`, its slope likewise; an exact
route stored them per scale.
"""
finish_slopes!(outputs::_SpectrumOutputs, _g, _k, ::NoSpectrum, _rate) =
    (_fill_nan!(outputs.slopes); _fill_nan!(outputs.spectrum_map); _fill_nan!(outputs.slope_map); outputs)
finish_slopes!(outputs::_SpectrumOutputs, g, k, ::AbstractSpectrumPolicy, rate) = _finish_slopes!(outputs, g, k, rate)
_finish_slopes!(outputs, _g, _k, _rate) = outputs
function _finish_slopes!(outputs, g, k, ::_DifferenceRate)
    _log_slope!(outputs.slopes, g, k)
    _map_density!(outputs.spectrum_map, outputs.slope_map, k)
    return outputs
end

_fill_nan!(::Nothing) = nothing
_fill_nan!(x::AbstractArray) = fill!(x, convert(eltype(x), NaN))

@inline _log_or_nan(x::T) where {T} = x > zero(T) ? log(x) : T(NaN)

_log_slope!(::Nothing, _g, _k) = nothing
_log_slope!(slopes, g, k) = spectral_density!(slopes, _log_or_nan.(g), log.(k))

# A map's cells' `½|ū|²` per scale into their density in `k`, and its slope in `ln k` into `p`.
_map_density!(::Nothing, _p, _k) = nothing
function _map_density!(m, p, k)
    E = copy(m)
    _density_along!(m, E, k)
    _map_slope!(p, m, k)
    return nothing
end

_map_slope!(::Nothing, _m, _k) = nothing
_map_slope!(p, m, k) = _density_along!(p, _log_or_nan.(m), log.(k))

# `dE/dk` along the last axis of `E` into `out`, by the quadratic through the three nearest scales, as
# `spectral_density!` takes it per value.
function _density_along!(out, E, k)
    d = ndims(E)
    n = size(E, d)
    n < 2 && return fill!(out, convert(eltype(out), NaN))
    sl(A, i) = selectdim(A, d, i)
    if n == 2
        s = (k[2] - k[1])
        sl(out, 1) .= (sl(E, 2) .- sl(E, 1)) ./ s
        sl(out, 2) .= sl(out, 1)
        return out
    end
    for i in 1:n
        j = clamp(i - 1, 1, n - 2)
        c0, c1, c2 = _quadratic_weights(k[i], k[j], k[j + 1], k[j + 2])
        sl(out, i) .= c0 .* sl(E, j) .+ c1 .* sl(E, j + 1) .+ c2 .* sl(E, j + 2)
    end
    return out
end

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
    _check_energy_fields(u, v, w, grid)
    Nscales = length(scales)
    length(spectrum) == Nscales || throw(DimensionMismatch(
        "spectrum has length $(length(spectrum)), expected $Nscales (= length(scales))",
    ))
    Nscales == 0 && return spectrum
    _cumulative_energy_rates!(spectrum, nothing, DifferenceScaleDerivative(), u, v, w, grid, kernel, scales,
                              workspace, filter_plans, execution, mask_strategy, method, vector_filter, transform)
    return spectrum
end

# `Ē(ℓ)` into `spectrum` and, on an exact route of `sd` with `rates` given, `⟨ū·∂_ℓū⟩` into `rates`; returns the
# rate route taken.
function _cumulative_energy_rates!(spectrum, rates, sd::AbstractScaleDerivative, u, v, w, grid, kernel, scales,
                                   workspace, filter_plans, execution, mask_strategy, method, vector_filter, transform)
    strategy = filter_plans === nothing ? mask_strategy : Filtering.plan_strategy(first(filter_plans))
    sweep() = Filtering.plan_filter_sweep(
        grid, kernel, scales; mask_strategy = mask_strategy, execution = execution, method = method,
    )
    ws, plans = _energy_setup(workspace, filter_plans, grid, strategy, w, vector_filter, transform, execution, sweep)
    _check_workspace_w(ws, w)
    total_area = active_area(grid)
    outputs = _rates_outputs(rates)
    if ws.spin !== nothing
        plans === nothing || _check_placement(ws.at, first(plans))
        route = _rates_route(rates, sd, kernel, ws, strategy, u, v, w)
        _spin_cumulative_energy!(spectrum, outputs, route, _spin_of(ws.spin, strategy), ws, u, v, w, kernel, scales,
                                 total_area)
        return route
    end
    swept = _swept(plans, sweep)
    _check_placement(ws.at, first(swept))
    route = _rates_route(rates, sd, kernel, ws, swept, u, v, w)
    _cumulative_energy!(spectrum, outputs, route, kernel, scales, ws, swept, u, v, w, grid, total_area)
    return route
end

@inline _rates_outputs(::Nothing) = nothing
@inline _rates_outputs(rates::AbstractVector) = _SpectrumOutputs(rates)
@inline _rates_route(::Nothing, _sd, _kernel, _ws, _plan, _u, _v, _w) = _DifferenceRate()
@inline _rates_route(::AbstractVector, sd, kernel, ws, plan, u, v, w) =
    _rate_route(sd, kernel, ws, plan, u, v, w, Filtering.ScaleRates())

function _check_energy_fields(u, v, w, grid)
    gsz = FlowGeometries.Grids.size_tuple(grid)
    size(u) == gsz || throw(DimensionMismatch("u has size $(size(u)), grid expects $gsz"))
    size(v) == gsz || throw(DimensionMismatch("v has size $(size(v)), grid expects $gsz"))
    w === nothing || size(w) == gsz || throw(DimensionMismatch("w has size $(size(w)), grid expects $gsz"))
    return nothing
end

# The workspace and plans a cumulative energy reads: the caller's workspace with its plans, or a new
# workspace placed by the plans, swept unless the vector filter filters through its own transforms.
function _energy_setup(
    ::Nothing, filter_plans, grid, strategy, w, vector_filter, transform, execution, sweep::F,
) where {F}
    vf = Filtering.resolve_vector_filter(grid, strategy, vector_filter)
    plans = _setup_plan(filter_plans, vf, sweep)
    ws = EnergyWorkspace(grid, (), w; vector_filter = vf, transform = transform,
                         execution = _plan_backend(plans, execution))
    return ws, plans
end
_energy_setup(ws, filter_plans, _, _, _, _, _, _, _) = (ws, filter_plans)

# A sweep's plans, `sweep()` building them where there are none.
@inline _swept(::Nothing, sweep::F) where {F} = sweep()
@inline _swept(plans, _) = plans

function _cumulative_energy!(
    spectrum::AbstractVector{T}, outputs, route, kernel, scales, ws, plans, u, v, w,
    grid::FlowGeometries.Grids.AbstractGrid{G,T}, total_area::T,
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
        _to_ambient!(pin, u, v, w, ws.at.frames, b)
        loc = w === nothing ? (u_filt, v_filt) : (u_filt, v_filt, w_filt)
    end

    for s_idx in eachindex(spectrum, plans)
        plan = plans[s_idx]
        og = _output(ws.at, plan)

        if ambient
            Filtering.filter_apply_batch!(pout, pin, plan)
            _ambient_to_local!(loc, pout, _output_frames(ws.at, plan), b)
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
        _store_rate!(outputs, s_idx, route, ws, plan, kernel, T(scales[s_idx]), u, v, w, total_area, nothing)
    end

    return spectrum
end

# `E(ℓ)` under the generalized convolution: the velocity is analyzed once and synthesized per scale, on
# every cell of the grid; on the spin rate route each scale's `⟨ū·∂_ℓū⟩` into the outputs.
function _spin_cumulative_energy!(
    spectrum::AbstractVector{T}, outputs, route, sf::SpinFilter, ws, u, v, w, kernel, scales, total_area::T,
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
        _spin_rate_at!(outputs, s_idx, route, _spin_filtered(ws, w), sf, kernel, T(scales[s_idx]), og, b, total_area)
    end
    return spectrum
end

_spin_rate_at!(_outputs, _s, ::_DifferenceRate, _filt, _sf, _kernel, _scale, _og, _b, _total_area) = nothing
function _spin_rate_at!(outputs::_SpectrumOutputs, s::Int, r::_SpinRate, filt, sf, kernel, scale, og, b, total_area)
    _spin_rate_fields!(r, sf, kernel, scale, 0)
    _store_outputs!(outputs, s, _scale_view(outputs.spectrum_map, s), _scale_view(outputs.slope_map, s), filt, r.a, r.b, og, 0,
                 scale, total_area, b)
    return nothing
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
    scale_derivative::AbstractScaleDerivative = AutoScaleDerivative(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    compute = gate_spectrum(kernel, policy)
    kℓ = T(L) ./ T.(scales)
    compute || return kℓ, fill(T(NaN), length(kℓ))
    _check_energy_fields(u, v, w, grid)
    isempty(scales) && return kℓ, T[]
    cum, Ẽ = zeros(T, length(scales)), zeros(T, length(scales))
    route = _cumulative_energy_rates!(cum, Ẽ, scale_derivative, u, v, w, grid, kernel, scales, workspace,
                                      filter_plans, execution, mask_strategy, method, vector_filter, transform)
    return kℓ, filtering_density!(Ẽ, cum, kℓ, policy, route, T.(scales), L)
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

# Derivative at `x` of the quadratic through `(x0, y0)`, `(x1, y1)`, `(x2, y2)` (Lagrange form), and its weights on
# `y0`, `y1`, `y2`.
@inline function _quadratic_slope(x, x0, x1, x2, y0, y1, y2)
    c0, c1, c2 = _quadratic_weights(x, x0, x1, x2)
    return y0 * c0 + y1 * c1 + y2 * c2
end
@inline _quadratic_weights(x, x0, x1, x2) = (((x - x1) + (x - x2)) / ((x0 - x1) * (x0 - x2)),
                                             ((x - x0) + (x - x2)) / ((x1 - x0) * (x1 - x2)),
                                             ((x - x0) + (x - x1)) / ((x2 - x0) * (x2 - x1)))

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
