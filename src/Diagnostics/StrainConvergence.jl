# ---------------------------------------------------------------------------
# Strain / convergence decomposition of Π (Srinivasan, Barkan & McWilliams 2023, eq. 10)
# ---------------------------------------------------------------------------

"""
    PiStrainWorkspace(grid)

Scratch for [`compute_Π_strain_convergence!`](@ref): the two filtered velocities, the three stress
components, the four velocity-gradient components, the two rotation invariants and the three flux
fields, plus the two shared product buffers.
"""
struct PiStrainWorkspace{T<:AbstractFloat, A<:AbstractArray{T}}
    ū::A; v̄::A
    prod::A; fbuf::A
    τuu::A; τuv::A; τvv::A
    ux::A; uy::A; vx::A; vy::A
    δ::A; α::A
    Πα::A; Πδ::A; total::A
end

function PiStrainWorkspace(grid::FlowGeometries.Grids.AbstractGrid{G,T}) where {T<:AbstractFloat, G}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    return PiStrainWorkspace(ntuple(_ -> zeros(T, gsz), 16)...)
end

"""
    compute_Π_strain_convergence!(ws, u, v, grid, kernel, scale; filter_plan=nothing, deriv_plan=nothing, ...)
        -> (; total, strain, convergence, divergence, strain_magnitude)

In-place [`compute_Π_strain_convergence`](@ref). Returns views of `ws`'s buffers, valid until the next
call on the same workspace. With `ws` and both plans supplied, a repeated evaluation allocates nothing.
"""
function compute_Π_strain_convergence!(
    ws::PiStrainWorkspace{T},
    u::AbstractVecOrMat,
    v::AbstractVecOrMat,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    filter_plan::Union{Nothing,Filtering.AbstractFilterPlan} = nothing,
    deriv_plan::Union{Nothing,AnyDerivPlan} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    _require_tangent_pair(grid, "compute_Π_strain_convergence!")
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend) :
        filter_plan
    og = output_grid(grid, plan)
    dplan = _resolve_deriv_plan(deriv_plan, og)

    Filtering.filter_apply_batch!((ws.ū, ws.v̄), (u, v), plan)
    _second_moment!(ws.τuu, u, u, ws.ū, ws.ū, ws.prod, ws.fbuf, plan)
    _second_moment!(ws.τuv, u, v, ws.ū, ws.v̄, ws.prod, ws.fbuf, plan)
    _second_moment!(ws.τvv, v, v, ws.v̄, ws.v̄, ws.prod, ws.fbuf, plan)

    _grad2!(ws.ux, ws.uy, ws.ū, og, dplan)
    _grad2!(ws.vx, ws.vy, ws.v̄, og, dplan)

    mask = FlowGeometries.Grids.mask(og)
    # δ̄ = ū_x + v̄_y and ᾱ² = σ̄_n² + σ̄_s² are the two rotation invariants of the filtered gradient;
    # σ̄_n = ū_x − v̄_y (normal strain) and σ̄_s = ū_y + v̄_x (shear strain) are not, so they are
    # consumed inline rather than returned.
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
    compute_Π_strain_convergence(u, v, grid, kernel, scale; backend=AutoBackend(), mask_strategy=ZeroFill())
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

with `Π_α` the **deformation/shear production** — energy transferred by straining, present even in
non-divergent flow — and `Π_δ` the **convergence production**, which vanishes identically for a
non-divergent field and is the term that paper adds. Setting `δ̄ = 0` recovers Polzin (2010); the
equivalent `Π = E′(γᵖ ᾱ − δ̄)` form with `E′ = (τ_vv + τ_uu)/2` recovers Jing et al. (2017), and
`|γᵖ| ≤ 1` gives the bound `|Π_α| ≤ ᾱ E′`.

`Π_α − Π_δ` is **algebraically identical** to the direct `Π = −S̄:τ̄` that [`compute_Π!`](@ref)
computes — expanding eq. (10) collapses to `−τ_uu ū_x − τ_uv(ū_y + v̄_x) − τ_vv v̄_y`. The two are
therefore a genuine cross-check rather than a tautology: they contract different combinations of the
same four derivatives, so a sign or an ordering error in either shows up as a disagreement. The suite
asserts they match to round-off on masked and unmasked grids.

Returns specific flux maps in m² s⁻³, as [`compute_Π!`](@ref) does, plus the two rotation invariants, which are the natural axes to bin the
flux against (`divergence` = δ̄, `strain_magnitude` = ᾱ).

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
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    size(u) == gsz || throw(DimensionMismatch("u has size $(size(u)), grid expects $gsz"))
    size(v) == gsz || throw(DimensionMismatch("v has size $(size(v)), grid expects $gsz"))
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend)
    return compute_Π_strain_convergence!(
        PiStrainWorkspace(grid), u, v, grid, kernel, scale;
        filter_plan = plan, deriv_plan = _default_deriv_plan(output_grid(grid, plan)),
    )
end

"""
    compute_Π_strain_convergence(u, v, grid::AbstractGrid{<:SphericalGeometry}, kernel, scale; ...)
        -> (; total, strain, convergence, divergence, strain_magnitude)

Spherical form of the strain/convergence split, on any grid resolving two tangent directions.

The split is a statement about the filtered strain tensor, so it carries over from the plane
unchanged once that tensor is the spherical one:

```
δ̄ = S̄_ee + S̄_nn        σ̄_n = S̄_ee − S̄_nn        σ̄_s = 2·S̄_en
```

Substituting `S̄_xx = (δ̄+σ̄_n)/2`, `S̄_yy = (δ̄−σ̄_n)/2`, `S̄_xy = σ̄_s/2` into `−S̄:τ̄` collapses to
`Π_α − Π_δ` term by term, on any metric. What the sphere changes is the two tensors fed in: `S̄` picks
up the `tanφ/R` curvature terms of the local frame, and `τ̄` is formed in planetary-Cartesian
coordinates and rotated back (Aluie 2019), exactly as [`compute_Π!`](@ref) does — this shares that
code, so the two agree by construction and the suite asserts `total == compute_Π!`.

Takes a [`ΠWorkspace`](@ref): the spherical path needs the three planetary components and their six
moments, which that workspace already carries.
"""
function compute_Π_strain_convergence(
    u::AbstractVecOrMat,
    v::AbstractVecOrMat,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    workspace::Union{Nothing, ΠWorkspace} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    size(u) == gsz || throw(DimensionMismatch("u has size $(size(u)), grid expects $gsz"))
    size(v) == gsz || throw(DimensionMismatch("v has size $(size(v)), grid expects $gsz"))
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend)
    return compute_Π_strain_convergence!(
        workspace === nothing ? ΠWorkspace(grid) : workspace, u, v, grid, kernel, scale;
        filter_plan = plan, deriv_plan = _default_deriv_plan(output_grid(grid, plan)),
    )
end

"""
    compute_Π_strain_convergence!(ws::ΠWorkspace, u, v, grid::AbstractGrid{<:SphericalGeometry}, kernel, scale; ...)
        -> (; total, strain, convergence, divergence, strain_magnitude)

In-place spherical [`compute_Π_strain_convergence`](@ref), over the same workspace
[`compute_Π!`](@ref) uses. Returns views of `ws`'s buffers, valid until the next call on it.
"""
function compute_Π_strain_convergence!(
    ws::ΠWorkspace,
    u::AbstractVecOrMat,
    v::AbstractVecOrMat,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    filter_plan::Union{Nothing,Filtering.AbstractFilterPlan} = nothing,
    deriv_plan::Union{Nothing,AnyDerivPlan} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}}
    _require_tangent_pair(grid, "compute_Π_strain_convergence!")
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend) :
        filter_plan
    og = output_grid(grid, plan)
    dplan = _resolve_deriv_plan(deriv_plan, og)

    _fill_stress_strain!(u, v, nothing, grid, ws, plan, dplan, nothing)

    # `scratch`/`scratch2`/`scratch3` are spent once the stress is built, so they carry the two
    # invariants and the deformation channel; `u_filt`/`v_filt` carry the rest.
    mask = FlowGeometries.Grids.mask(og)
    δ, α, Πα = ws.scratch, ws.scratch2, ws.scratch3
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
    compute_Π_decomposed(u, v, w, u_rot, v_rot, w_rot, grid::StructuredGrid{T,Cartesian,3}, kernel, scale; backend=AutoBackend(), mask_strategy=ZeroFill())
        -> (; total, rotational, cross, divergent)

True three-dimensional analog of the 2D [`compute_Π_decomposed`](@ref) above: the same both-sides
(strain AND stress) rotational/divergent split — see that method's docstring for the derivation —
generalized to all six independent strain/stress tensor components, contracted the same way the
true-3D [`compute_Π!`](@ref) method does (nine-term symmetric contraction).
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
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    # One stencil table for every derivative below; they differ only in direction and field.
    dplan = Derivatives.StencilPlan(grid)
    gsz = FlowGeometries.Grids.size_tuple(grid)
    size(u) == gsz || throw(DimensionMismatch("u has size $(size(u)), grid expects $gsz"))
    size(v) == gsz || throw(DimensionMismatch("v has size $(size(v)), grid expects $gsz"))
    size(w) == gsz || throw(DimensionMismatch("w has size $(size(w)), grid expects $gsz"))
    size(u_rot) == gsz || throw(DimensionMismatch("u_rot has size $(size(u_rot)), grid expects $gsz"))
    size(v_rot) == gsz || throw(DimensionMismatch("v_rot has size $(size(v_rot)), grid expects $gsz"))
    size(w_rot) == gsz || throw(DimensionMismatch("w_rot has size $(size(w_rot)), grid expects $gsz"))
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend)
    return compute_Π_decomposed!(
        PiDecomposed3DWorkspace(grid), u, v, w, u_rot, v_rot, w_rot, grid, kernel, scale;
        filter_plan = plan, deriv_plan = dplan,
    )
end

"""
    PiDecomposed3DWorkspace(grid)

Scratch for the true-3D [`compute_Π_decomposed!`](@ref): the three divergent components, the two
filtered velocity triples, six product buffers, the six independent components of each of the three
stress tensors and the two strain tensors, the four flux fields, and one derivative temporary.

Tensor components are stored in the `(xx, xy, xz, yy, yz, zz)` order shared with the spherical
[`tau_decomposition!`](@ref), so one loop over that order builds a whole tensor and feeds a single
batched apply.
"""
struct PiDecomposed3DWorkspace{T<:AbstractFloat, A<:AbstractArray{T,3}}
    div::NTuple{3,A}     # (u,v,w) − (u_rot,v_rot,w_rot)
    br::NTuple{3,A}      # filtered rotational triple
    bd::NTuple{3,A}      # filtered divergent triple
    prod::NTuple{6,A}    # the six symmetric products feeding one batched apply
    τRR::NTuple{6,A}
    τDD::NTuple{6,A}
    τX::NTuple{6,A}
    SR::NTuple{6,A}
    SD::NTuple{6,A}
    Πrr::A; Πdd::A; Πx::A; total::A
    tmp::A
end

function PiDecomposed3DWorkspace(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,3},
) where {T<:AbstractFloat, G}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    z() = zeros(T, gsz)
    t(n) = ntuple(_ -> z(), n)
    return PiDecomposed3DWorkspace(
        t(3), t(3), t(3), t(6), t(6), t(6), t(6), t(6), t(6), z(), z(), z(), z(), z(),
    )
end

# Derivative along axis `d` ∈ {1,2,3}, selected at run time so the three off-diagonal strain components
# share one loop body.
@inline function _dd!(dst, f, d::Int, grid, dplan)
    d == 1 ? Derivatives.ddx!(dst, f, grid, dplan) :
    d == 2 ? Derivatives.ddy!(dst, f, grid, dplan) :
             Derivatives.ddz!(dst, f, grid, dplan)
    return nothing
end

# The six independent components of S̄ from an already-filtered velocity triple `b`, in `_SYM3` order.
# `tmp` carries the second derivative of each off-diagonal.
@inline function _strain3_into!(S, b, tmp, grid, dplan, ::Type{T}) where {T}
    Derivatives.ddx!(S[1], b[1], grid, dplan)
    Derivatives.ddy!(S[4], b[2], grid, dplan)
    Derivatives.ddz!(S[6], b[3], grid, dplan)
    for (k, i1, i2) in ((2, 1, 2), (3, 1, 3), (5, 2, 3))
        _dd!(S[k], b[i1], i2, grid, dplan)
        _dd!(tmp, b[i2], i1, grid, dplan)
        @. S[k] = T(0.5) * (S[k] + tmp)
    end
    _add_strain3_curvature!(S, b, grid, T)
    return nothing
end

# A Cartesian volume's strain is the symmetrized velocity gradient and nothing more.
@inline _add_strain3_curvature!(
    _S, _b, ::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.CartesianGeometry}, ::Type,
) = nothing

# A spherical shell, in the scale factors `h_λ = r cosφ`, `h_φ = r`, `h_r = 1`:
#
#     S_ee += w̄_r/r − v̄_n tanφ/r      S_nn += w̄_r/r      S_en += ½ ū_e tanφ/r
#     S_er −= ½ ū_e/r                  S_nr −= ½ v̄_n/r
#
# with `S_rr = ∂w̄_r/∂r` carrying no correction. The distance derivatives are already metric-scaled at
# each point's own radius, so this adds only the terms the frame's rotation contributes.
function _add_strain3_curvature!(
    S, b, grid::FlowGeometries.Grids.StructuredGrid{T2,G,3}, ::Type{T},
) where {T, T2, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T2}}
    @inbounds for I in CartesianIndices(S[1])
        i = Tuple(I)
        FlowGeometries.Grids.isactive(grid, i...) || continue
        _, φ, rk = FlowGeometries.Grids.coords(grid, i...)
        sinφ, cosφ = sincos(φ)
        tf = abs(cosφ) > T(1e-12) ? sinφ / (rk * cosφ) : zero(T)
        inv_r = one(T) / rk
        u_e = b[1][I]; v_n = b[2][I]; w_r = b[3][I]
        S[1][I] += w_r * inv_r - v_n * tf
        S[4][I] += w_r * inv_r
        S[2][I] += T(0.5) * u_e * tf
        S[3][I] -= T(0.5) * u_e * inv_r
        S[5][I] -= T(0.5) * v_n * inv_r
    end
    return nothing
end

# Nine-term symmetric contraction S_ij τ_ij at one point, over the `_SYM3` component order.
Base.@propagate_inbounds _contract_sym3(S, τ, I) =
    S[1][I] * τ[1][I] + S[4][I] * τ[4][I] + S[6][I] * τ[6][I] +
    2 * (S[2][I] * τ[2][I] + S[3][I] * τ[3][I] + S[5][I] * τ[5][I])

"""
    compute_Π_decomposed!(ws::PiDecomposed3DWorkspace, u, v, w, u_rot, v_rot, w_rot, grid, kernel, scale;
                          filter_plan=nothing, deriv_plan=nothing, ...)
        -> (; total, rotational, cross, divergent)

In-place true-3D [`compute_Π_decomposed`](@ref). Returns views of `ws`'s buffers, valid until the next
call on the same workspace. With `ws` and both plans supplied, a repeated evaluation allocates nothing.

Each stress tensor comes from one batched apply of its six symmetric products. The cross stress sums
the two orderings before filtering, which the linearity of the filter permits:

    τX_ij = M(uʳ_i, uᵈ_j) + M(uᵈ_i, uʳ_j) = (uʳ_i uᵈ_j + uᵈ_i uʳ_j)‾ − (ūʳ_i ūᵈ_j + ūᵈ_i ūʳ_j)

so it costs one product per component, and the diagonal `i = j` picks up its factor of two from the
same expression. The two strains read the filtered triples the stresses already needed, so a whole
evaluation is four batched applies over 24 fields.
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
    deriv_plan::Union{Nothing,Derivatives.StencilPlan} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend) :
        filter_plan
    dplan = deriv_plan === nothing ? Derivatives.StencilPlan(grid) : deriv_plan

    rot = (u_rot, v_rot, w_rot)
    dv, br, bd, pr = ws.div, ws.br, ws.bd, ws.prod

    # Divergent (irrotational) part is the complement of the supplied rotational part.
    for (d, f, r) in zip(dv, (u, v, w), rot)
        @. d = f - r
    end

    # Both filtered triples feed the self stresses, the cross stress AND a strain: one batch of six.
    Filtering.filter_apply_batch!((br..., bd...), (rot..., dv...), plan)

    # `τʳʳ`, `τ_X` and `τᵈᵈ` are the three generalized second moments of the rotational/divergent pair.
    # The Germano split takes the same three of the filtered velocity and its residual.
    _pair_moments!(ws.τRR, ws.τX, ws.τDD, rot, dv, br, bd, pr, plan)

    og = output_grid(grid, plan)
    _strain3_into!(ws.SR, br, ws.tmp, og, dplan, T)
    _strain3_into!(ws.SD, bd, ws.tmp, og, dplan, T)

    # One pass builds all four flux channels, so each tensor component is read once.
    mask = FlowGeometries.Grids.mask(og)
    @inbounds for I in CartesianIndices(ws.total)
        if mask[I]
            rr = _contract_sym3(ws.SR, ws.τRR, I)
            dd = _contract_sym3(ws.SD, ws.τDD, I)
            x = _contract_sym3(ws.SR, ws.τDD, I) + _contract_sym3(ws.SD, ws.τRR, I) +
                _contract_sym3(ws.SR, ws.τX, I) + _contract_sym3(ws.SD, ws.τX, I)
            # `total` is summed from the three stored channels, in their returned order, so
            # `total == rotational + cross + divergent` holds to the last bit.
            ws.Πrr[I] = -rr
            ws.Πx[I] = -x
            ws.Πdd[I] = -dd
            ws.total[I] = (ws.Πrr[I] + ws.Πx[I]) + ws.Πdd[I]
        else
            ws.Πrr[I] = zero(T); ws.Πdd[I] = zero(T); ws.Πx[I] = zero(T); ws.total[I] = zero(T)
        end
    end
    return (; total = ws.total, rotational = ws.Πrr, cross = ws.Πx, divergent = ws.Πdd)
end

"""
    SphericalPiDecomposed3DWorkspace(grid)

Scratch for the spherical volumetric [`compute_Π_decomposed!`](@ref): the two planetary-Cartesian
velocity triples and their filtered forms, six product buffers, the six components of each of the
three stresses and the two strains, one local triple, and the four flux fields.
"""
struct SphericalPiDecomposed3DWorkspace{T<:AbstractFloat, A<:AbstractArray{T,3}}
    pr::NTuple{3,A}     # planetary rotational
    pd::NTuple{3,A}     # planetary divergent
    br::NTuple{3,A}
    bd::NTuple{3,A}
    prod::NTuple{6,A}
    τRR::NTuple{6,A}
    τX::NTuple{6,A}
    τDD::NTuple{6,A}
    SR::NTuple{6,A}
    SD::NTuple{6,A}
    loc::NTuple{3,A}    # one part's filtered velocity in the local frame, reused for the other
    tmp::A
    Πrr::A; Πdd::A; Πx::A; total::A
end

function SphericalPiDecomposed3DWorkspace(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,3},
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    z() = zeros(T, gsz)
    t(n) = ntuple(_ -> z(), n)
    return SphericalPiDecomposed3DWorkspace(
        t(3), t(3), t(3), t(3), t(6), t(6), t(6), t(6), t(6), t(6), t(3), z(), z(), z(), z(), z(),
    )
end

"""
    compute_Π_decomposed(u, v, w, u_rot, v_rot, w_rot, grid::StructuredGrid{T,<:SphericalGeometry,3}, kernel, scale; ...)
        -> (; total, rotational, cross, divergent)

Spherical volumetric form of the rotational/divergent split: the same both-sides decomposition the
Cartesian volume takes, with the shell's two conventions applied. The three stresses are formed as
generalized second moments in planetary-Cartesian coordinates and rotated back to local
`(east, north, radial)`, and each part's strain carries the shell's curvature terms — so the channels
still sum to the flux the volumetric [`compute_Π!`](@ref) computes.
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
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    for (nm, a) in (("u", u), ("v", v), ("w", w),
                    ("u_rot", u_rot), ("v_rot", v_rot), ("w_rot", w_rot))
        size(a) == gsz || throw(DimensionMismatch("$nm has size $(size(a)), grid expects $gsz"))
    end
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend)
    return compute_Π_decomposed!(
        SphericalPiDecomposed3DWorkspace(grid), u, v, w, u_rot, v_rot, w_rot, grid, kernel, scale;
        filter_plan = plan, deriv_plan = Derivatives.StencilPlan(grid),
    )
end

"""
    compute_Π_decomposed!(ws::SphericalPiDecomposed3DWorkspace, u, v, w, u_rot, v_rot, w_rot, grid, kernel, scale; ...)

In-place spherical volumetric [`compute_Π_decomposed`](@ref). Four batched applies carry the whole
split: six planetary components of the two parts, then six products per stress.
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
    deriv_plan::Union{Nothing,Derivatives.StencilPlan} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}}
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend) :
        filter_plan
    dplan = deriv_plan === nothing ? Derivatives.StencilPlan(grid) : deriv_plan
    pr, pd, br, bd = ws.pr, ws.pd, ws.br, ws.bd

    # Both parts to planetary Cartesian, the divergent one as the complement of the rotational.
    _fill_tau3_velocity!(pr, u_rot, v_rot, w_rot, grid, T)
    _fill_tau3_velocity!(pd, u, v, w, grid, T)
    for c in 1:3
        @. pd[c] = pd[c] - pr[c]
    end

    Filtering.filter_apply_batch!((br..., bd...), (pr..., pd...), plan)
    _pair_moments!(ws.τRR, ws.τX, ws.τDD, pr, pd, br, bd, ws.prod, plan)
    og = output_grid(grid, plan)
    _rotate_tau3_to_local!((ws.τRR, ws.τX, ws.τDD), og)

    # The strain is linear, so each part carries its own; both are taken in the local frame.
    _localize_triple!(ws.loc, br, og, T)
    _strain3_into!(ws.SR, ws.loc, ws.tmp, og, dplan, T)
    _localize_triple!(ws.loc, bd, og, T)
    _strain3_into!(ws.SD, ws.loc, ws.tmp, og, dplan, T)

    mask = FlowGeometries.Grids.mask(og)
    @inbounds for I in CartesianIndices(ws.total)
        if mask[I]
            rr = _contract_sym3(ws.SR, ws.τRR, I)
            dd = _contract_sym3(ws.SD, ws.τDD, I)
            x = _contract_sym3(ws.SR, ws.τDD, I) + _contract_sym3(ws.SD, ws.τRR, I) +
                _contract_sym3(ws.SR, ws.τX, I) + _contract_sym3(ws.SD, ws.τX, I)
            ws.Πrr[I] = -rr
            ws.Πx[I] = -x
            ws.Πdd[I] = -dd
            ws.total[I] = (ws.Πrr[I] + ws.Πx[I]) + ws.Πdd[I]
        else
            ws.Πrr[I] = zero(T); ws.Πdd[I] = zero(T); ws.Πx[I] = zero(T); ws.total[I] = zero(T)
        end
    end
    return (; total = ws.total, rotational = ws.Πrr, cross = ws.Πx, divergent = ws.Πdd)
end
