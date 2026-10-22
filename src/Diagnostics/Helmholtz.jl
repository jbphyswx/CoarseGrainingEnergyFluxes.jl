# ---------------------------------------------------------------------------
# Rotational / divergent (Helmholtz) decomposition of the energy flux
# ---------------------------------------------------------------------------

"""
    compute_Π_decomposed(u, v, u_rot, v_rot, grid, kernel, scale; execution=AutoBackend(), mask_strategy=ZeroFill())
        -> (; total, rotational, cross, divergent)

Split the cross-scale flux `Π = −S̄:τ` on a plane into rotational–rotational `Π_RR`, divergent–divergent
`Π_DD` and interaction `Π_X` parts, the last the stimulated-cascade channel of Barkan, Srinivasan &
McWilliams (2024), by splitting both sides of the contraction.

`(u_rot, v_rot)` is the rotational (divergence-free) part of `(u, v)`, and the divergent part is the
rest, `(u, v) − (u_rot, v_rot)`. With
`u = uʳ + uᵈ` the strain is linear, `S̄ = S̄ʳ + S̄ᵈ`, and the stress bilinear,
`τ(u,u) = τʳʳ + τᵈᵈ + τ_X` with `τ_X = τ(uʳ,uᵈ) + τ(uᵈ,uʳ)`, so

    Π_RR = −S̄ʳ:τʳʳ
    Π_DD = −S̄ᵈ:τᵈᵈ
    Π_X  = −(S̄ʳ:τᵈᵈ + S̄ᵈ:τʳʳ + S̄ʳ:τ_X + S̄ᵈ:τ_X)

and `Π = Π_RR + Π_X + Π_DD` exactly, each channel built from its own terms.

Returns specific flux maps (m² s⁻³): `rotational` = Π_RR, `divergent` = Π_DD, `cross` = Π_X.
"""
function compute_Π_decomposed(
    u::AbstractVecOrMat,
    v::AbstractVecOrMat,
    u_rot::AbstractVecOrMat,
    v_rot::AbstractVecOrMat,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    size(u) == gsz || throw(DimensionMismatch("u has size $(size(u)), grid expects $gsz"))
    size(v) == gsz || throw(DimensionMismatch("v has size $(size(v)), grid expects $gsz"))
    size(u_rot) == gsz || throw(DimensionMismatch("u_rot has size $(size(u_rot)), grid expects $gsz"))
    size(v_rot) == gsz || throw(DimensionMismatch("v_rot has size $(size(v_rot)), grid expects $gsz"))
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, execution=execution)
    dplan = Derivatives.derivative_plan(output_grid(grid, plan))
    return compute_Π_decomposed!(
        PiDecomposedWorkspace(grid; execution = Filtering.plan_execution(plan)), u, v, u_rot, v_rot, grid,
        kernel, scale; filter_plan = plan, deriv_plan = dplan,
    )
end

"""
    PiDecomposedWorkspace(grid; execution = SerialBackend())

Scratch for [`compute_Π_decomposed!`](@ref), in the memory of `execution` (see [`Placement`](@ref)):
the divergent components, the four filtered means, the three stress tensors, the two strain tensors, the
four flux fields and three shared temporaries.
"""
struct PiDecomposedWorkspace{T<:AbstractFloat, A<:AbstractArray{T}, P<:Placement}
    u_div::A; v_div::A
    ūr::A; v̄r::A; ūd::A; v̄d::A
    prod::A; fbuf::A; scratch::A
    τRR_xx::A; τRR_xy::A; τRR_yy::A
    τDD_xx::A; τDD_xy::A; τDD_yy::A
    τX_xx::A;  τX_xy::A;  τX_yy::A
    SR_xx::A;  SR_xy::A;  SR_yy::A
    SD_xx::A;  SD_xy::A;  SD_yy::A
    Πrr::A; Πdd::A; Πx::A; total::A
    at::P
end

function PiDecomposedWorkspace(
    grid::FlowGeometries.Grids.AbstractGrid{G,T};
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
) where {T<:AbstractFloat, G}
    at = Placement(grid; execution = execution)
    gsz = FlowGeometries.Grids.size_tuple(grid)
    return PiDecomposedWorkspace(ntuple(_ -> _zeros(at, T, gsz), 28)..., at)
end

# The strain of the filtered tangent pair `(ā, b̄)` into `(Sxx, Sxy, Syy)`, the frame's turning included;
# `g` is two scratch arrays, and `grid` is held where the backend `be` runs.
@inline _strain_into!(Sxx, Sxy, Syy, ā, b̄, g::NTuple{2,AbstractArray}, grid, dplan, be) =
    _strain!((Sxx, Syy, Sxy), (ā, b̄), grid, dplan, g, be)

"""
    SphericalPiDecomposedWorkspace(grid; execution = SerialBackend(), vector_filter = nothing,
                                   transform = AutoSpectralBackend(), nufft = AutoSpectralBackend())

Scratch for the spherical [`compute_Π_decomposed!`](@ref), in the memory of `execution` (see
[`Placement`](@ref)): the two ambient Cartesian velocity triples and their filtered forms, six product
buffers, the six components of each of the three stresses, the local filtered velocity pair of each part,
the two strains, and the four flux fields.

On the surface of a sphere the workspace holds the transforms of the vector filter `vector_filter` names,
as [`ΠWorkspace`](@ref) does. Under the generalized convolution, the default, it holds the divergent
tangent pair in place of the ambient triples and their filtered forms, two scratch arrays for the
strains, and each stress's three local components.
"""
struct SphericalPiDecomposedWorkspace{
    T<:AbstractFloat, A<:AbstractArray{T}, R<:Tuple{Vararg{A}}, D<:Tuple{Vararg{A}}, F<:Tuple{Vararg{A}},
    W<:Tuple{Vararg{A}}, M<:Tuple{Vararg{A}}, SF, P<:Placement,
}
    pr::R               # ambient rotational
    pd::D               # ambient divergent, or the divergent tangent pair
    br::F               # filtered rotational
    bd::F               # filtered divergent
    prod::W             # the products, then the strains' scratch
    τRR::M              # (ee, en, nn) in slots 1-3
    τX::M
    τDD::M
    loc::NTuple{4,A}    # (ūʳ_e, v̄ʳ_n, ūᵈ_e, v̄ᵈ_n)
    SR::NTuple{3,A}     # (xx, xy, yy)
    SD::NTuple{3,A}
    Πrr::A; Πdd::A; Πx::A; total::A
    spin::SF            # the generalized convolution, a `SpinFilter`, or nothing
    at::P
end

function SphericalPiDecomposedWorkspace(
    grid::FlowGeometries.Grids.AbstractGrid{G,T};
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
    vector_filter::Union{Nothing,Filtering.AbstractVectorFilter} = nothing,
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    nufft::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}}
    at = Placement(grid; execution = execution)
    spin = _workspace_spin(grid, at, vector_filter, transform, nufft)
    gsz = FlowGeometries.Grids.size_tuple(grid)
    z() = _zeros(at, T, gsz)
    t(::Val{n}) where {n} = ntuple(_ -> z(), Val(n))
    spin === nothing && return SphericalPiDecomposedWorkspace(
        t(Val(3)), t(Val(3)), t(Val(3)), t(Val(3)), t(Val(6)), t(Val(6)), t(Val(6)), t(Val(6)), t(Val(4)),
        t(Val(3)), t(Val(3)), z(), z(), z(), z(), spin, at,
    )
    return SphericalPiDecomposedWorkspace(
        (), t(Val(2)), (), (), t(Val(2)), t(Val(3)), t(Val(3)), t(Val(3)), t(Val(4)),
        t(Val(3)), t(Val(3)), z(), z(), z(), z(), spin, at,
    )
end

"""
    compute_Π_decomposed(u, v, u_rot, v_rot, grid::AbstractGrid{<:SphericalGeometry}, kernel, scale; vector_filter=nothing, transform=AutoSpectralBackend(), ...)
        -> (; total, rotational, cross, divergent)

Spherical form of the rotational/divergent split, on any grid resolving two tangent directions, with
the channels and the identity `Π = Π_RR + Π_X + Π_DD` of the planar method. Each part's strain carries
the local frame's `tanφ/R` curvature terms. The stresses `τʳʳ`, `τᵈᵈ` and `τ_X` are generalized second
moments under `vector_filter` ([`Filtering.resolve_vector_filter`](@ref)): by default the generalized
convolution of Aluie (2019), which keeps each filtered part rotational or divergent (Proposition 2) and
reads no scalar filter plan, its spin transforms chosen by `transform`; `Filtering.CartesianComponents()`
forms them in ambient Cartesian components rotated to the local frame.
"""
function compute_Π_decomposed(
    u::AbstractVecOrMat,
    v::AbstractVecOrMat,
    u_rot::AbstractVecOrMat,
    v_rot::AbstractVecOrMat,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    vector_filter::Union{Nothing, Filtering.AbstractVectorFilter} = nothing,
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    for (nm, a) in (("u", u), ("v", v), ("u_rot", u_rot), ("v_rot", v_rot))
        size(a) == gsz || throw(DimensionMismatch("$nm has size $(size(a)), grid expects $gsz"))
    end
    dplan = Derivatives.derivative_plan(output_grid(grid, mask_strategy))
    vf = Filtering.resolve_vector_filter(grid, mask_strategy, vector_filter)
    if vf isa Filtering.GeneralizedConvolution
        ws = SphericalPiDecomposedWorkspace(grid; execution = execution, vector_filter = vf, transform = transform)
        return compute_Π_decomposed!(ws, u, v, u_rot, v_rot, grid, kernel, scale;
                                     mask_strategy = mask_strategy, deriv_plan = dplan)
    end
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy, execution, transform)
    return compute_Π_decomposed!(
        SphericalPiDecomposedWorkspace(grid; execution = Filtering.plan_execution(plan), vector_filter = vf),
        u, v, u_rot, v_rot, grid, kernel, scale; filter_plan = plan, deriv_plan = dplan,
    )
end

"""
    compute_Π_decomposed!(ws::SphericalPiDecomposedWorkspace, u, v, u_rot, v_rot, grid, kernel, scale; ...)
        -> (; total, rotational, cross, divergent)

In-place spherical [`compute_Π_decomposed`](@ref), by the workspace's vector filter. Returns views of
`ws`'s buffers, valid until the next call on the same workspace. The generalized convolution reads no
scalar filter plan, and none is built for it when `filter_plan` is not given.
"""
function compute_Π_decomposed!(
    ws::SphericalPiDecomposedWorkspace{T},
    u::AbstractVecOrMat,
    v::AbstractVecOrMat,
    u_rot::AbstractVecOrMat,
    v_rot::AbstractVecOrMat,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    filter_plan::Union{Nothing,Filtering.AbstractFilterPlan} = nothing,
    deriv_plan::Union{Nothing,AnyDerivPlan} = nothing,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}}
    _require_tangent_pair(grid, "compute_Π_decomposed!")
    plan = _plan_for(ws.spin, filter_plan, mask_strategy,
                     () -> Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, execution = execution))
    _check_placement(ws.at, plan)
    b = ws.at.execution
    og = _output(ws.at, plan)
    dplan = _placed(ws.at, _resolve_deriv_plan(deriv_plan, output_grid(grid, plan)))
    sf = _spin_of(ws.spin, plan)
    if sf === nothing
        _cartesian_decomposed_parts!(ws, u, v, u_rot, v_rot, plan, og, b)
    else
        _spin_decomposed_parts!(ws, sf, u, v, u_rot, v_rot, kernel, scale)
    end
    loc = ws.loc
    g = (ws.prod[1], ws.prod[2])
    _strain_into!(ws.SR[1], ws.SR[2], ws.SR[3], loc[1], loc[2], g, og, dplan, b)
    _strain_into!(ws.SD[1], ws.SD[2], ws.SD[3], loc[3], loc[4], g, og, dplan, b)

    mask = FlowGeometries.Grids.mask(og)
    SR, SD, τRR, τX, τDD = ws.SR, ws.SD, ws.τRR, ws.τX, ws.τDD
    @. ws.Πrr = ifelse(mask, -_sfs_contraction(SR[1], SR[2], SR[3], τRR[1], τRR[2], τRR[3]), zero(T))
    @. ws.Πdd = ifelse(mask, -_sfs_contraction(SD[1], SD[2], SD[3], τDD[1], τDD[2], τDD[3]), zero(T))
    @. ws.Πx = ifelse(mask, -(
        _sfs_contraction(SR[1], SR[2], SR[3], τDD[1], τDD[2], τDD[3]) +
        _sfs_contraction(SD[1], SD[2], SD[3], τRR[1], τRR[2], τRR[3]) +
        _sfs_contraction(SR[1], SR[2], SR[3], τX[1], τX[2], τX[3]) +
        _sfs_contraction(SD[1], SD[2], SD[3], τX[1], τX[2], τX[3])
    ), zero(T))
    # `total` is summed from the stored channels, in their returned order, so
    # `total == rotational + cross + divergent` holds to the last bit.
    @. ws.total = (ws.Πrr + ws.Πx) + ws.Πdd
    return (; total = ws.total, rotational = ws.Πrr, cross = ws.Πx, divergent = ws.Πdd)
end

# The three stresses and each part's local filtered pair by the ambient Cartesian components.
function _cartesian_decomposed_parts!(ws::SphericalPiDecomposedWorkspace, u, v, u_rot, v_rot, plan, og, b)
    pr, pd, br, bd, loc = ws.pr, ws.pd, ws.br, ws.bd, ws.loc
    _to_ambient!(pr, u_rot, v_rot, nothing, ws.at.grid, b)
    _to_ambient!(pd, u, v, nothing, ws.at.grid, b)
    for c in 1:3
        @. pd[c] = pd[c] - pr[c]
    end
    Filtering.filter_apply_batch!((br..., bd...), (pr..., pd...), plan)
    _sph_pair_moments!(ws.τRR, ws.τX, ws.τDD, pr, pd, br, bd, ws.prod, plan, og, b)
    _ambient_to_local!((loc[1], loc[2]), br, og, b)
    _ambient_to_local!((loc[3], loc[4]), bd, og, b)
    return nothing
end

# The same by the generalized convolution, which filters each part over the active cells.
function _spin_decomposed_parts!(ws::SphericalPiDecomposedWorkspace, sf::SpinFilter, u, v, u_rot, v_rot, kernel,
                                 scale)
    d, loc = ws.pd, ws.loc
    @. d[1] = u - u_rot
    @. d[2] = v - v_rot
    r = (u_rot, v_rot)
    r̄, d̄ = (loc[1], loc[2]), (loc[3], loc[4])
    _spin_transfer!(sf, kernel, scale)
    _spin_vector!(r̄[1], r̄[2], r[1], r[2], sf, true)
    _spin_vector!(d̄[1], d̄[2], d[1], d[2], sf, true)
    _spin_moment!(ws.τRR, sf, r, r, r̄, r̄, 1, true)
    _spin_moment!(ws.τX, sf, r, d, r̄, d̄, 2, true)
    _spin_moment!(ws.τDD, sf, d, d, d̄, d̄, 1, true)
    return nothing
end

"""
    compute_Π_decomposed!(ws, u, v, u_rot, v_rot, grid, kernel, scale; filter_plan=nothing, deriv_plan=nothing, ...)
        -> (; total, rotational, cross, divergent)

In-place [`compute_Π_decomposed`](@ref). Returns views of `ws`'s buffers, valid until the next call on
the same workspace. With `ws` and both plans supplied, a repeated evaluation allocates nothing.
"""
function compute_Π_decomposed!(
    ws::PiDecomposedWorkspace{T},
    u::AbstractVecOrMat,
    v::AbstractVecOrMat,
    u_rot::AbstractVecOrMat,
    v_rot::AbstractVecOrMat,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    filter_plan::Union{Nothing,Filtering.AbstractFilterPlan} = nothing,
    deriv_plan::Union{Nothing,AnyDerivPlan} = nothing,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    _require_tangent_pair(grid, "compute_Π_decomposed!")
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, execution = execution) :
        filter_plan
    _check_placement(ws.at, plan)
    b = ws.at.execution
    og = _output(ws.at, plan)
    dplan = _placed(ws.at, _resolve_deriv_plan(deriv_plan, output_grid(grid, plan)))

    @. ws.u_div = u - u_rot
    @. ws.v_div = v - v_rot

    Filtering.filter_apply_batch!(
        (ws.ūr, ws.v̄r, ws.ūd, ws.v̄d), (u_rot, v_rot, ws.u_div, ws.v_div), plan,
    )

    _second_moment!(ws.τRR_xx, u_rot, u_rot, ws.ūr, ws.ūr, ws.prod, ws.fbuf, plan)
    _second_moment!(ws.τRR_xy, u_rot, v_rot, ws.ūr, ws.v̄r, ws.prod, ws.fbuf, plan)
    _second_moment!(ws.τRR_yy, v_rot, v_rot, ws.v̄r, ws.v̄r, ws.prod, ws.fbuf, plan)
    _second_moment!(ws.τDD_xx, ws.u_div, ws.u_div, ws.ūd, ws.ūd, ws.prod, ws.fbuf, plan)
    _second_moment!(ws.τDD_xy, ws.u_div, ws.v_div, ws.ūd, ws.v̄d, ws.prod, ws.fbuf, plan)
    _second_moment!(ws.τDD_yy, ws.v_div, ws.v_div, ws.v̄d, ws.v̄d, ws.prod, ws.fbuf, plan)

    # Cross stress τ(rot,div) + τ(div,rot): the swap leaves a diagonal component unchanged, so it
    # doubles; the off-diagonal sums both orderings.
    _second_moment!(ws.τX_xx, u_rot, ws.u_div, ws.ūr, ws.ūd, ws.prod, ws.fbuf, plan)
    @. ws.τX_xx *= T(2)
    _second_moment!(ws.τX_yy, v_rot, ws.v_div, ws.v̄r, ws.v̄d, ws.prod, ws.fbuf, plan)
    @. ws.τX_yy *= T(2)
    _second_moment!(ws.τX_xy, u_rot, ws.v_div, ws.ūr, ws.v̄d, ws.prod, ws.fbuf, plan)
    _second_moment!(ws.scratch, ws.u_div, v_rot, ws.ūd, ws.v̄r, ws.prod, ws.fbuf, plan)
    @. ws.τX_xy += ws.scratch

    # The product buffers are free once the stresses are formed.
    _strain_into!(ws.SR_xx, ws.SR_xy, ws.SR_yy, ws.ūr, ws.v̄r, (ws.prod, ws.fbuf), og, dplan, b)
    _strain_into!(ws.SD_xx, ws.SD_xy, ws.SD_yy, ws.ūd, ws.v̄d, (ws.prod, ws.fbuf), og, dplan, b)

    mask = FlowGeometries.Grids.mask(og)
    @. ws.Πrr = ifelse(mask,
        -(ws.SR_xx * ws.τRR_xx + T(2) * ws.SR_xy * ws.τRR_xy + ws.SR_yy * ws.τRR_yy), zero(T))
    @. ws.Πdd = ifelse(mask,
        -(ws.SD_xx * ws.τDD_xx + T(2) * ws.SD_xy * ws.τDD_xy + ws.SD_yy * ws.τDD_yy), zero(T))
    @. ws.Πx = ifelse(mask, -(
        ws.SR_xx * ws.τDD_xx + T(2) * ws.SR_xy * ws.τDD_xy + ws.SR_yy * ws.τDD_yy +
        ws.SD_xx * ws.τRR_xx + T(2) * ws.SD_xy * ws.τRR_xy + ws.SD_yy * ws.τRR_yy +
        ws.SR_xx * ws.τX_xx  + T(2) * ws.SR_xy * ws.τX_xy  + ws.SR_yy * ws.τX_yy  +
        ws.SD_xx * ws.τX_xx  + T(2) * ws.SD_xy * ws.τX_xy  + ws.SD_yy * ws.τX_yy
    ), zero(T))
    @. ws.total = ws.Πrr + ws.Πx + ws.Πdd
    return (; total = ws.total, rotational = ws.Πrr, cross = ws.Πx, divergent = ws.Πdd)
end
