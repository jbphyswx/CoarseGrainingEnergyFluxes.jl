# ---------------------------------------------------------------------------
# Strain / convergence decomposition of Π (Srinivasan, Barkan & McWilliams 2023, eq. 10)
# ---------------------------------------------------------------------------

"""
    PiStrainWorkspace(grid; execution = SerialBackend())

Scratch for [`compute_Π_strain_convergence!`](@ref), in the memory of `execution` (see
[`Placement`](@ref)): the two filtered velocities, the three stress components, the four
velocity-gradient components, the two rotation invariants and the three flux fields, plus the two shared
product buffers.
"""
struct PiStrainWorkspace{T<:AbstractFloat, A<:AbstractArray{T}, P<:Placement}
    ū::A; v̄::A
    prod::A; fbuf::A
    τuu::A; τuv::A; τvv::A
    ux::A; uy::A; vx::A; vy::A
    δ::A; α::A
    Πα::A; Πδ::A; total::A
    at::P
end

function PiStrainWorkspace(
    grid::FlowGeometries.Grids.AbstractGrid{G,T};
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
) where {T<:AbstractFloat, G}
    at = Placement(grid; execution = execution)
    gsz = FlowGeometries.Grids.size_tuple(grid)
    return PiStrainWorkspace(ntuple(_ -> _zeros(at, T, gsz), Val(16))..., at)
end

"""
    compute_Π_strain_convergence!(ws, u, v, grid, kernel, scale; filter_plan=nothing, deriv_plan=nothing, ...)
        -> (; total, strain, convergence, divergence, strain_magnitude)

In-place [`compute_Π_strain_convergence`](@ref). Returns views of `ws`'s buffers, valid until the next
call on the same workspace. With `ws`, `filter_plan` and `deriv_plan = derivative_route(grid, filter_plan, u)`
([`derivative_route`](@ref)) supplied, a repeated evaluation allocates no field.
"""
function compute_Π_strain_convergence!(
    ws::PiStrainWorkspace{T},
    u::AbstractVecOrMat,
    v::AbstractVecOrMat,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    filter_plan::Union{Nothing,Filtering.AbstractFilterPlan} = nothing,
    deriv_plan::_DerivRoute{<:AnyDerivPlan} = nothing,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    derivative::Derivatives.AbstractDerivative = Derivatives.AutoDerivative(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    _require_tangent_pair(grid, "compute_Π_strain_convergence!")
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, execution = execution) :
        filter_plan
    _check_placement(ws.at, plan)
    b = ws.at.execution
    og = _output(ws.at, plan)
    dplan = _placed(ws.at, _derivative_route(derivative, deriv_plan, plan, grid, u))

    Fu, Fv = _filtered_with_spectra!((ws.ū, ws.v̄), (u, v), plan, dplan)
    _second_moment!(ws.τuu, u, u, ws.ū, ws.ū, ws.prod, ws.fbuf, plan)
    _second_moment!(ws.τuv, u, v, ws.ū, ws.v̄, ws.prod, ws.fbuf, plan)
    _second_moment!(ws.τvv, v, v, ws.v̄, ws.v̄, ws.prod, ws.fbuf, plan)

    _grad2_route!(ws.ux, ws.uy, ws.ū, Fu, og, dplan, b, plan)
    _grad2_route!(ws.vx, ws.vy, ws.v̄, Fv, og, dplan, b, plan)

    mask = FlowGeometries.Grids.mask(og)
    @. ws.δ = ifelse(mask, ws.ux + ws.vy, zero(T))
    @. ws.α = ifelse(mask, sqrt((ws.ux - ws.vy)^2 + (ws.uy + ws.vx)^2), zero(T))
    @. ws.Πα = ifelse(mask,
        (ws.τvv - ws.τuu) * (ws.ux - ws.vy) / T(2) - ws.τuv * (ws.uy + ws.vx), zero(T))
    @. ws.Πδ = ifelse(mask, (ws.τvv + ws.τuu) * (ws.ux + ws.vy) / T(2), zero(T))
    @. ws.total = ws.Πα - ws.Πδ
    return (; total = ws.total, strain = ws.Πα, convergence = ws.Πδ,
            divergence = ws.δ, strain_magnitude = ws.α)
end

"""
    compute_Π_strain_convergence(u, v, grid, kernel, scale; execution=AutoBackend(), mask_strategy=ZeroFill())
        -> (; total, strain, convergence, divergence, strain_magnitude)

Split the 2D cross-scale flux into the two production terms of Srinivasan, Barkan & McWilliams (2023),
eq. (10), by diagonalizing the filtered strain tensor. With

```
δ̄   = ū_x + v̄_y            divergence         (rotation invariant)
σ̄_n = ū_x − v̄_y            normal strain
σ̄_s = ū_y + v̄_x            shear strain
ᾱ   = √(σ̄_n² + σ̄_s²)       strain magnitude   (rotation invariant)
```

the flux separates into

```
Π = Π_α − Π_δ ,   Π_α = (τ_vv − τ_uu) σ̄_n/2 − τ_uv σ̄_s ,   Π_δ = (τ_vv + τ_uu) δ̄/2 ,
```

with `Π_α` the deformation (shear) production, energy transferred by straining, and `Π_δ` the
convergence production, zero for a non-divergent field. Setting `δ̄ = 0` recovers Polzin (2010); the
equivalent `Π = E′(γᵖ ᾱ − δ̄)` form with `E′ = (τ_vv + τ_uu)/2` recovers Jing et al. (2017), and
`|γᵖ| ≤ 1` gives the bound `|Π_α| ≤ ᾱ E′`.

`total = Π_α − Π_δ` equals the `Π = −S̄:τ̄` of [`compute_Π!`](@ref): both expand to
`−τ_uu ū_x − τ_uv(ū_y + v̄_x) − τ_vv v̄_y`.

Returns specific flux maps (m² s⁻³) and the two rotation invariants, `divergence` = δ̄ and
`strain_magnitude` = ᾱ.

# References
- Srinivasan, K., Barkan, R., & McWilliams, J. C. (2023). A forward energy flux at submesoscales
  driven by frontogenesis. *J. Phys. Oceanogr.* 53(1), 287–305. doi:10.1175/JPO-D-22-0001.1
"""
function compute_Π_strain_convergence(
    u::AbstractVecOrMat,
    v::AbstractVecOrMat,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Filtering.AbstractFilterMethod = Filtering.RealSpace(),
    derivative::Derivatives.AbstractDerivative = Derivatives.AutoDerivative(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    size(u) == gsz || throw(DimensionMismatch("u has size $(size(u)), grid expects $gsz"))
    size(v) == gsz || throw(DimensionMismatch("v has size $(size(v)), grid expects $gsz"))
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy, execution, method)
    return compute_Π_strain_convergence!(
        PiStrainWorkspace(grid; execution = Filtering.plan_execution(plan)), u, v, grid, kernel, scale;
        filter_plan = plan, derivative = derivative,
    )
end

"""
    compute_Π_strain_convergence(u, v, grid::AbstractGrid{<:SphericalGeometry}, kernel, scale; ...)
        -> (; total, strain, convergence, divergence, strain_magnitude)

Spherical form of the strain/convergence split, on any grid resolving two tangent directions, with the
invariants of the local-frame filtered strain:

```
δ̄ = S̄_ee + S̄_nn        σ̄_n = S̄_ee − S̄_nn        σ̄_s = 2·S̄_en
```

`Π_α − Π_δ = −S̄:τ̄` holds on any metric. `S̄` carries the local frame's `tan φ/R` terms and `τ̄` is
filtered by `vector_filter` ([`Filtering.resolve_vector_filter`](@ref)), the generalized convolution by
default; both are those of [`compute_Π!`](@ref), so `total` is its flux. `workspace` is a
[`ΠWorkspace`](@ref); `vector_filter` and `transform` build it when it is not given.
"""
function compute_Π_strain_convergence(
    u::AbstractVecOrMat,
    v::AbstractVecOrMat,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    workspace::Union{Nothing, ΠWorkspace} = nothing,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    vector_filter::Union{Nothing, Filtering.AbstractVectorFilter} = nothing,
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    derivative::Derivatives.AbstractDerivative = Derivatives.AutoDerivative(),
) where {T<:AbstractFloat, G<:_CurvedSurface{T}}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    size(u) == gsz || throw(DimensionMismatch("u has size $(size(u)), grid expects $gsz"))
    size(v) == gsz || throw(DimensionMismatch("v has size $(size(v)), grid expects $gsz"))
    ws, plan = _flux_setup(
        grid, u, nothing, workspace, nothing,
        () -> Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, execution = execution);
        execution, mask_strategy, vector_filter, transform,
    )
    return compute_Π_strain_convergence!(
        ws, u, v, grid, kernel, scale;
        filter_plan = plan isa Filtering.AbstractFilterPlan ? plan : nothing, mask_strategy = mask_strategy,
        derivative = derivative,
    )
end

"""
    compute_Π_strain_convergence!(ws::ΠWorkspace, u, v, grid::AbstractGrid{<:SphericalGeometry}, kernel, scale; ...)
        -> (; total, strain, convergence, divergence, strain_magnitude)

In-place spherical [`compute_Π_strain_convergence`](@ref), over the same workspace
[`compute_Π!`](@ref) uses, whose vector filter applies. Returns views of `ws`'s buffers, valid until the
next call on it.
"""
function compute_Π_strain_convergence!(
    ws::ΠWorkspace,
    u::AbstractVecOrMat,
    v::AbstractVecOrMat,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    filter_plan::Union{Nothing,Filtering.AbstractFilterPlan} = nothing,
    deriv_plan::_DerivRoute{<:AnyDerivPlan} = nothing,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    derivative::Derivatives.AbstractDerivative = Derivatives.AutoDerivative(),
) where {T<:AbstractFloat, G<:_CurvedSurface{T}}
    _require_tangent_pair(grid, "compute_Π_strain_convergence!")
    plan = _plan_for(ws.spin, filter_plan, mask_strategy,
                     () -> Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, execution = execution))
    _check_placement(ws.at, plan)
    dplan = _placed(ws.at, _derivative_route(derivative, deriv_plan, _route_source(ws.spin, plan), grid, u))

    _fill_stress_strain!(u, v, nothing, grid, ws, plan, dplan, kernel, scale, nothing)

    # The filtered velocity and the ambient buffers are spent once the stress and strain are built, and hold
    # the outputs.
    mask = FlowGeometries.Grids.mask(_output(ws.at, plan))
    δ, α, Πα = ws.ux, ws.uy, ws.uz
    Πδ, total = ws.u_filt, ws.v_filt
    @. δ = ifelse(mask, ws.S_xx + ws.S_yy, zero(T))
    @. α = ifelse(mask, sqrt((ws.S_xx - ws.S_yy)^2 + (T(2) * ws.S_xy)^2), zero(T))
    @. Πα = ifelse(mask,
        (ws.τ_yy - ws.τ_xx) * (ws.S_xx - ws.S_yy) / T(2) - ws.τ_xy * (T(2) * ws.S_xy), zero(T))
    @. Πδ = ifelse(mask, (ws.τ_yy + ws.τ_xx) * (ws.S_xx + ws.S_yy) / T(2), zero(T))
    @. total = Πα - Πδ
    return (; total = total, strain = Πα, convergence = Πδ,
            divergence = δ, strain_magnitude = α)
end

"""
    compute_Π_decomposed(u, v, w, u_rot, v_rot, w_rot, grid::StructuredGrid{T,Cartesian,3}, kernel, scale; execution=AutoBackend(), mask_strategy=ZeroFill())
        -> (; total, rotational, cross, divergent)

The rotational/divergent split of the 2-D [`compute_Π_decomposed`](@ref) in a Cartesian volume, over
all six components of each strain and stress tensor.
"""
function compute_Π_decomposed(
    u::AbstractArray{<:Any,3},
    v::AbstractArray{<:Any,3},
    w::AbstractArray{<:Any,3},
    u_rot::AbstractArray{<:Any,3},
    v_rot::AbstractArray{<:Any,3},
    w_rot::AbstractArray{<:Any,3},
    grid::FlowGeometries.Grids.StructuredGrid{T,G,3},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    method::Filtering.AbstractFilterMethod = Filtering.RealSpace(),
    derivative::Derivatives.AbstractDerivative = Derivatives.AutoDerivative(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    size(u) == gsz || throw(DimensionMismatch("u has size $(size(u)), grid expects $gsz"))
    size(v) == gsz || throw(DimensionMismatch("v has size $(size(v)), grid expects $gsz"))
    size(w) == gsz || throw(DimensionMismatch("w has size $(size(w)), grid expects $gsz"))
    size(u_rot) == gsz || throw(DimensionMismatch("u_rot has size $(size(u_rot)), grid expects $gsz"))
    size(v_rot) == gsz || throw(DimensionMismatch("v_rot has size $(size(v_rot)), grid expects $gsz"))
    size(w_rot) == gsz || throw(DimensionMismatch("w_rot has size $(size(w_rot)), grid expects $gsz"))
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy, execution, method)
    return compute_Π_decomposed!(
        PiDecomposed3DWorkspace(grid; execution = Filtering.plan_execution(plan)), u, v, w, u_rot, v_rot,
        w_rot, grid, kernel, scale;
        filter_plan = plan, derivative = derivative,
    )
end

"""
    PiDecomposed3DWorkspace(grid; execution = SerialBackend())

Scratch for the true-3D [`compute_Π_decomposed!`](@ref), in the memory of `execution` (see
[`Placement`](@ref)): the three divergent components, the two filtered velocity triples, six product
buffers, the six independent components of each of the three stress tensors and the two strain tensors,
and the four flux fields. Tensor components are stored in the order `(xx, xy, xz, yy, yz, zz)`.
"""
struct PiDecomposed3DWorkspace{T<:AbstractFloat, A<:AbstractArray{T,3}, P<:Placement}
    div::NTuple{3,A}     # (u,v,w) − (u_rot,v_rot,w_rot)
    br::NTuple{3,A}      # filtered rotational triple
    bd::NTuple{3,A}      # filtered divergent triple
    prod::NTuple{6,A}    # the six symmetric products, then the strains' scratch
    τRR::NTuple{6,A}
    τDD::NTuple{6,A}
    τX::NTuple{6,A}
    SR::NTuple{6,A}
    SD::NTuple{6,A}
    Πrr::A; Πdd::A; Πx::A; total::A
    at::P
end

function PiDecomposed3DWorkspace(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,3};
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
) where {T<:AbstractFloat, G}
    at = Placement(grid; execution = execution)
    gsz = FlowGeometries.Grids.size_tuple(grid)
    z() = _zeros(at, T, gsz)
    t3() = (z(), z(), z())
    t6() = (t3()..., t3()...)
    return PiDecomposed3DWorkspace(
        t3(), t3(), t3(), t6(), t6(), t6(), t6(), t6(), t6(), z(), z(), z(), z(), at,
    )
end

# The six components of S̄ of a filtered velocity triple `b3` into `S`, in `_SYM3` order, the frame's
# turning included; `g` is three scratch arrays, and `grid` is held where the backend `be` runs. Under the
# series route the triple's spectra `F` give it from the filter's own output.
@inline _strain_sym3!(S::NTuple{6}, b3::NTuple{3}, grid, dplan, g::NTuple{3}, be) =
    _strain!((S[1], S[4], S[6], S[2], S[3], S[5]), b3, grid, dplan, g, be)
@inline _strain_sym3!(S::NTuple{6}, b3::NTuple{3}, grid, dplan, g::NTuple{3}, be, _F, _plan) =
    _strain_sym3!(S, b3, grid, dplan, g, be)

function _strain_sym3!(S::NTuple{6}, b3::NTuple{3}, _grid, ::SeriesStrain, g::NTuple{3}, _be, F::NTuple{3}, plan)
    _series_derivative!(S[1], F[1], b3[1], plan, 1)
    _series_derivative!(S[4], F[2], b3[2], plan, 2)
    _series_derivative!(S[6], F[3], b3[3], plan, 3)
    _series_shear!(S[2], (F[1], b3[1]), (F[2], b3[2]), 1, 2, plan, g[1], g[2])
    _series_shear!(S[3], (F[1], b3[1]), (F[3], b3[3]), 1, 3, plan, g[1], g[2])
    _series_shear!(S[5], (F[2], b3[2]), (F[3], b3[3]), 2, 3, plan, g[1], g[2])
    return nothing
end

# Nine-term symmetric contraction S_ij τ_ij at one point, over the `_SYM3` component order.
@inline _sym3_dot(S1, S2, S3, S4, S5, S6, t1, t2, t3, t4, t5, t6) =
    S1 * t1 + S4 * t4 + S6 * t6 + 2 * (S2 * t2 + S3 * t3 + S5 * t5)

# The three channels and their sum from the stored strains and stresses, zero where `mask` is false. The
# cross channel accumulates its four terms in this order, and `total` sums the stored channels, so
# `total == rotational + cross + divergent` holds to the last bit.
function _decomposed_channels!(ws, mask, ::Type{T}) where {T}
    SR, SD, τRR, τDD, τX = ws.SR, ws.SD, ws.τRR, ws.τDD, ws.τX
    @. ws.Πrr = ifelse(mask, -_sym3_dot(SR..., τRR...), zero(T))
    @. ws.Πdd = ifelse(mask, -_sym3_dot(SD..., τDD...), zero(T))
    @. ws.Πx = _sym3_dot(SR..., τDD...)
    @. ws.Πx += _sym3_dot(SD..., τRR...)
    @. ws.Πx += _sym3_dot(SR..., τX...)
    @. ws.Πx += _sym3_dot(SD..., τX...)
    @. ws.Πx = ifelse(mask, -ws.Πx, zero(T))
    @. ws.total = (ws.Πrr + ws.Πx) + ws.Πdd
    return (; total = ws.total, rotational = ws.Πrr, cross = ws.Πx, divergent = ws.Πdd)
end

"""
    compute_Π_decomposed!(ws::PiDecomposed3DWorkspace, u, v, w, u_rot, v_rot, w_rot, grid, kernel, scale;
                          filter_plan=nothing, deriv_plan=nothing, ...)
        -> (; total, rotational, cross, divergent)

In-place true-3D [`compute_Π_decomposed`](@ref). Returns views of `ws`'s buffers, valid until the next
call on the same workspace. With `ws`, `filter_plan` and `deriv_plan = derivative_route(grid, filter_plan, u)`
([`derivative_route`](@ref)) supplied, a repeated evaluation allocates no field.
"""
function compute_Π_decomposed!(
    ws::PiDecomposed3DWorkspace{T},
    u::AbstractArray{<:Any,3},
    v::AbstractArray{<:Any,3},
    w::AbstractArray{<:Any,3},
    u_rot::AbstractArray{<:Any,3},
    v_rot::AbstractArray{<:Any,3},
    w_rot::AbstractArray{<:Any,3},
    grid::FlowGeometries.Grids.StructuredGrid{T,G,3},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    filter_plan::Union{Nothing,Filtering.AbstractFilterPlan} = nothing,
    deriv_plan::_DerivRoute{<:FlowGeometries.Operators.StencilPlans} = nothing,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    derivative::Derivatives.AbstractDerivative = Derivatives.AutoDerivative(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, execution = execution) :
        filter_plan
    _check_placement(ws.at, plan)
    b = ws.at.execution
    og = _output(ws.at, plan)
    dplan = _placed(ws.at, _derivative_route(derivative, deriv_plan, plan, grid, u))

    rot = (u_rot, v_rot, w_rot)
    dv, br, bd, pr = ws.div, ws.br, ws.bd, ws.prod

    for (d, f, r) in zip(dv, (u, v, w), rot)
        @. d = f - r
    end

    F = _filtered_with_spectra!((br..., bd...), (rot..., dv...), plan, dplan)

    _pair_moments!(ws.τRR, ws.τX, ws.τDD, rot, dv, br, bd, pr, plan)

    g = (pr[1], pr[2], pr[3])
    _strain_sym3!(ws.SR, br, og, dplan, g, b, (F[1], F[2], F[3]), plan)
    _strain_sym3!(ws.SD, bd, og, dplan, g, b, (F[4], F[5], F[6]), plan)

    return _decomposed_channels!(ws, FlowGeometries.Grids.mask(og), T)
end

"""
    SphericalPiDecomposed3DWorkspace(grid; execution = SerialBackend())

Scratch for the spherical volumetric [`compute_Π_decomposed!`](@ref), in the memory of `execution` (see
[`Placement`](@ref)): the two ambient Cartesian velocity triples and their filtered forms, six product
buffers, the six components of each of the three stresses and the two strains, one local triple, and the
four flux fields.
"""
struct SphericalPiDecomposed3DWorkspace{T<:AbstractFloat, A<:AbstractArray{T,3}, P<:Placement}
    pr::NTuple{3,A}     # ambient rotational
    pd::NTuple{3,A}     # ambient divergent
    br::NTuple{3,A}
    bd::NTuple{3,A}
    prod::NTuple{6,A}
    τRR::NTuple{6,A}
    τX::NTuple{6,A}
    τDD::NTuple{6,A}
    SR::NTuple{6,A}
    SD::NTuple{6,A}
    loc::NTuple{3,A}    # one part's filtered velocity in the local frame, reused for the other
    Πrr::A; Πdd::A; Πx::A; total::A
    at::P
end

function SphericalPiDecomposed3DWorkspace(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,3};
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}}
    at = Placement(grid; execution = execution)
    gsz = FlowGeometries.Grids.size_tuple(grid)
    z() = _zeros(at, T, gsz)
    t3() = (z(), z(), z())
    t6() = (t3()..., t3()...)
    return SphericalPiDecomposed3DWorkspace(
        t3(), t3(), t3(), t3(), t6(), t6(), t6(), t6(), t6(), t6(), t3(), z(), z(), z(), z(), at,
    )
end

"""
    compute_Π_decomposed(u, v, w, u_rot, v_rot, w_rot, grid::StructuredGrid{T,<:SphericalGeometry,3}, kernel, scale; ...)
        -> (; total, rotational, cross, divergent)

The rotational/divergent split in a spherical shell. The three stresses are generalized second moments
in ambient Cartesian components rotated to the local (east, north, up) frame, and each part's strain
carries the shell's curvature terms, so the channels sum to the flux of the volumetric
[`compute_Π!`](@ref).
"""
function compute_Π_decomposed(
    u::AbstractArray{<:Any,3},
    v::AbstractArray{<:Any,3},
    w::AbstractArray{<:Any,3},
    u_rot::AbstractArray{<:Any,3},
    v_rot::AbstractArray{<:Any,3},
    w_rot::AbstractArray{<:Any,3},
    grid::FlowGeometries.Grids.StructuredGrid{T,G,3},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    derivative::Derivatives.AbstractDerivative = Derivatives.AutoDerivative(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    for (nm, a) in (("u", u), ("v", v), ("w", w),
                    ("u_rot", u_rot), ("v_rot", v_rot), ("w_rot", w_rot))
        size(a) == gsz || throw(DimensionMismatch("$nm has size $(size(a)), grid expects $gsz"))
    end
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, execution = execution)
    return compute_Π_decomposed!(
        SphericalPiDecomposed3DWorkspace(grid; execution = Filtering.plan_execution(plan)),
        u, v, w, u_rot, v_rot, w_rot, grid, kernel, scale;
        filter_plan = plan, derivative = derivative,
    )
end

"""
    compute_Π_decomposed!(ws::SphericalPiDecomposed3DWorkspace, u, v, w, u_rot, v_rot, w_rot, grid, kernel, scale; ...)

In-place spherical volumetric [`compute_Π_decomposed`](@ref). Returns views of `ws`'s buffers, valid
until the next call on the same workspace.
"""
function compute_Π_decomposed!(
    ws::SphericalPiDecomposed3DWorkspace{T},
    u::AbstractArray{<:Any,3},
    v::AbstractArray{<:Any,3},
    w::AbstractArray{<:Any,3},
    u_rot::AbstractArray{<:Any,3},
    v_rot::AbstractArray{<:Any,3},
    w_rot::AbstractArray{<:Any,3},
    grid::FlowGeometries.Grids.StructuredGrid{T,G,3},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    filter_plan::Union{Nothing,Filtering.AbstractFilterPlan} = nothing,
    deriv_plan::_DerivRoute{<:FlowGeometries.Operators.StencilPlans} = nothing,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    derivative::Derivatives.AbstractDerivative = Derivatives.AutoDerivative(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}}
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, execution = execution) :
        filter_plan
    _check_placement(ws.at, plan)
    b = ws.at.execution
    og, fr = _output(ws.at, plan), _output_frames(ws.at, plan)
    dplan = _placed(ws.at, _shell_route(_derivative_route(derivative, deriv_plan, plan, grid, u)))
    pr, pd, br, bd = ws.pr, ws.pd, ws.br, ws.bd

    _fill_tau3_velocity!(pr, u_rot, v_rot, w_rot, ws.at.grid, ws.at.frames, b)
    _fill_tau3_velocity!(pd, u, v, w, ws.at.grid, ws.at.frames, b)
    for c in 1:3
        @. pd[c] = pd[c] - pr[c]
    end

    Filtering.filter_apply_batch!((br..., bd...), (pr..., pd...), plan)
    _pair_moments!(ws.τRR, ws.τX, ws.τDD, pr, pd, br, bd, ws.prod, plan)
    _rotate_tau3_to_local!((ws.τRR, ws.τX, ws.τDD), og, fr, b)

    # The product buffers are free once the stresses are formed.
    g = (ws.prod[1], ws.prod[2], ws.prod[3])
    _localize_triple!(ws.loc, br, og, fr, b)
    _strain_sym3!(ws.SR, ws.loc, og, dplan, g, b)
    _localize_triple!(ws.loc, bd, og, fr, b)
    _strain_sym3!(ws.SD, ws.loc, og, dplan, g, b)

    return _decomposed_channels!(ws, FlowGeometries.Grids.mask(og), T)
end
