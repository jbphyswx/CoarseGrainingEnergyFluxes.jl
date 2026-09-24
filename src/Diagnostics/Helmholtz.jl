# ---------------------------------------------------------------------------
# Rotational / divergent (Helmholtz) decomposition of the energy flux
# ---------------------------------------------------------------------------

"""
    compute_Π_decomposed(u, v, u_rot, v_rot, grid, kernel, scale; backend=AutoBackend(), mask_strategy=ZeroFill())
        -> (; total, rotational, cross, divergent)

Split the 2D Cartesian cross-scale KE flux Π = -S̄_ij τ_ij into rotational-rotational (Π_RR),
divergent-divergent (Π_DD), and cross/interaction (Π_X — the "stimulated cascade" channel of
Barkan, Srinivasan & McWilliams 2024, JPO) parts, by decomposing **both sides** of the bilinear
contraction, not just the stress.

The Helmholtz decomposition itself is NOT recomputed here — pass the rotational (solenoidal,
divergence-free) part `(u_rot, v_rot)` from a Helmholtz solver (e.g. `HelmholtzDecomposition.jl`); the
divergent (irrotational) part is taken as the complement `(u, v) - (u_rot, v_rot)`. Writing
`u = uʳ + uᵈ`:

  - The strain S̄ is LINEAR in velocity, so it splits with **no cross term**: `S̄ = S̄ʳ + S̄ᵈ`.
  - The stress τ is BILINEAR (quadratic in velocity), so it splits into three pieces:
    `τ(u,u) = τ(uʳ,uʳ) + τ(uᵈ,uᵈ) + [τ(uʳ,uᵈ) + τ(uᵈ,uʳ)] = τʳʳ + τᵈᵈ + τ_X`.

Substituting both splits into `Π = -S̄:τ = -(S̄ʳ+S̄ᵈ):(τʳʳ+τᵈᵈ+τ_X)` and expanding the six resulting
terms into three physically named channels:

    Π_RR = -S̄ʳ:τʳʳ                                        (pure rotational-to-rotational cascade)
    Π_DD = -S̄ᵈ:τᵈᵈ                                        (pure divergent-to-divergent cascade)
    Π_X  = -(S̄ʳ:τᵈᵈ + S̄ᵈ:τʳʳ + S̄ʳ:τ_X + S̄ᵈ:τ_X)          (all rotational/divergent interaction terms)

so the channels sum **exactly** to the total flux, Π = Π_RR + Π_X + Π_DD — each piece constructed
directly (not as a residual), yet the identity holds by the same bilinearity/linearity argument.
Contracting the split stress against the *full*, undecomposed strain S̄ — a one-sided split — is only
correct when S̄ᵈ ≡ 0, and silently wrong whenever the divergent part carries strain of its own.

Returns a named tuple of specific flux maps (m² s⁻³): `rotational` = Π_RR, `divergent` = Π_DD,
`cross` = Π_X.
"""
function compute_Π_decomposed(
    u::AbstractVecOrMat,
    v::AbstractVecOrMat,
    u_rot::AbstractVecOrMat,
    v_rot::AbstractVecOrMat,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    size(u) == gsz || throw(DimensionMismatch("u has size $(size(u)), grid expects $gsz"))
    size(v) == gsz || throw(DimensionMismatch("v has size $(size(v)), grid expects $gsz"))
    size(u_rot) == gsz || throw(DimensionMismatch("u_rot has size $(size(u_rot)), grid expects $gsz"))
    size(v_rot) == gsz || throw(DimensionMismatch("v_rot has size $(size(v_rot)), grid expects $gsz"))
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy=mask_strategy, backend=backend)
    # One derivative object for every gradient below, over the cells the filtered fields are read on.
    dplan = _default_deriv_plan(output_grid(grid, plan))
    return compute_Π_decomposed!(
        PiDecomposedWorkspace(grid), u, v, u_rot, v_rot, grid, kernel, scale;
        filter_plan = plan, deriv_plan = dplan,
    )
end

"""
    PiDecomposedWorkspace(grid)

Scratch for [`compute_Π_decomposed!`](@ref): the divergent components, the four filtered means, the
three stress tensors, the two strain tensors, the four flux fields and three shared temporaries.
"""
struct PiDecomposedWorkspace{T<:AbstractFloat, A<:AbstractArray{T}}
    u_div::A; v_div::A
    ūr::A; v̄r::A; ūd::A; v̄d::A
    prod::A; fbuf::A; scratch::A
    τRR_xx::A; τRR_xy::A; τRR_yy::A
    τDD_xx::A; τDD_xy::A; τDD_yy::A
    τX_xx::A;  τX_xy::A;  τX_yy::A
    SR_xx::A;  SR_xy::A;  SR_yy::A
    SD_xx::A;  SD_xy::A;  SD_yy::A
    Πrr::A; Πdd::A; Πx::A; total::A
end

function PiDecomposedWorkspace(grid::FlowGeometries.Grids.AbstractGrid{G,T}) where {T<:AbstractFloat, G}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    z() = zeros(T, gsz)
    return PiDecomposedWorkspace(ntuple(_ -> z(), 28)...)
end

# S̄_xx = ∂ā/∂x, S̄_yy = ∂b̄/∂y, S̄_xy = ½(∂ā/∂y + ∂b̄/∂x), from an already-filtered pair, into caller
# buffers. `tmp` is scratch for the second cross derivative.
@inline function _strain_into!(Sxx, Sxy, Syy, ā, b̄, tmp, grid, dplan, ::Type{T}) where {T}
    _grad2!(Sxx, Sxy, ā, grid, dplan)     # ∂ā/∂x, ∂ā/∂y
    _grad2!(tmp, Syy, b̄, grid, dplan)     # ∂b̄/∂x, ∂b̄/∂y
    @. Sxy = T(0.5) * (Sxy + tmp)
    _add_strain_curvature!(Sxx, Sxy, ā, b̄, grid, T)
    return nothing
end

# On a plane the strain is the symmetrized velocity gradient and nothing more.
@inline _add_strain_curvature!(
    _Sxx, _Sxy, _ā, _b̄,
    ::FlowGeometries.Grids.AbstractGrid{<:FlowGeometries.Geometry.CartesianGeometry}, ::Type,
) = nothing

# On a sphere the local frame turns from point to point, and the strain in those orthogonal
# curvilinear coordinates carries the metric's own terms:
#
#     S_ee = ∂ū_e/∂x − v̄_n tanφ/R ,   S_en = ½(∂ū_e/∂y + ∂v̄_n/∂x + ū_e tanφ/R) ,   S_nn = ∂v̄_n/∂y
#
# with `∂/∂x`, `∂/∂y` the distance derivatives the gradient returns. Same factor the spherical curl
# carries, so the strain and the vorticity stay in one gauge.
function _add_strain_curvature!(
    Sxx, Sxy, ue, vn, grid::FlowGeometries.Grids.AbstractGrid{G,T2}, ::Type{T},
) where {T, T2, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T2}}
    geo = FlowGeometries.Grids.grid_geometry(grid)
    @inbounds for I in CartesianIndices(Sxx)
        i = Tuple(I)
        FlowGeometries.Grids.isactive(grid, i...) || continue
        _, φ = FlowGeometries.Grids.coords(grid, i...)
        tf = _tan_factor(geo, φ)
        Sxx[I] -= vn[I] * tf
        Sxy[I] += T(0.5) * ue[I] * tf
    end
    return nothing
end

"""
    SphericalPiDecomposedWorkspace(grid)

Scratch for the spherical [`compute_Π_decomposed!`](@ref): the two planetary-Cartesian velocity
triples and their filtered forms, six product buffers, the six components of each of the three
stresses, the local filtered velocity pair of each part, the two strains, and the four flux fields.
"""
struct SphericalPiDecomposedWorkspace{T<:AbstractFloat, A<:AbstractArray{T}}
    pr::NTuple{3,A}     # planetary rotational
    pd::NTuple{3,A}     # planetary divergent
    br::NTuple{3,A}     # filtered rotational
    bd::NTuple{3,A}     # filtered divergent
    prod::NTuple{6,A}
    τRR::NTuple{6,A}    # rotated to local in slots 1-3
    τX::NTuple{6,A}
    τDD::NTuple{6,A}
    loc::NTuple{4,A}    # (ūʳ_e, v̄ʳ_n, ūᵈ_e, v̄ᵈ_n)
    SR::NTuple{3,A}     # (xx, xy, yy)
    SD::NTuple{3,A}
    tmp::A
    Πrr::A; Πdd::A; Πx::A; total::A
end

function SphericalPiDecomposedWorkspace(
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    z() = zeros(T, gsz)
    t(n) = ntuple(_ -> z(), n)
    return SphericalPiDecomposedWorkspace(
        t(3), t(3), t(3), t(3), t(6), t(6), t(6), t(6), t(4), t(3), t(3), z(), z(), z(), z(), z(),
    )
end

"""
    compute_Π_decomposed(u, v, u_rot, v_rot, grid::AbstractGrid{<:SphericalGeometry}, kernel, scale; ...)
        -> (; total, rotational, cross, divergent)

Spherical form of the rotational/divergent split, on any grid resolving two tangent directions.

Both sides of the bilinear contraction split exactly as they do on a plane; what the sphere changes is
how each side is built. The stress is bilinear, so its three pieces `τʳʳ`, `τᵈᵈ` and `τ_X` are formed
as generalized second moments in planetary-Cartesian coordinates and rotated back to the local frame
(Aluie 2019) — the same moments the spherical [`tau_decomposition!`](@ref) takes, with the rotational
and divergent parts in place of the filtered velocity and its residual. The strain is linear, so it
splits with no cross term, and each part's strain carries the local frame's `tanφ/R` curvature terms.
"""
function compute_Π_decomposed(
    u::AbstractVecOrMat,
    v::AbstractVecOrMat,
    u_rot::AbstractVecOrMat,
    v_rot::AbstractVecOrMat,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    for (nm, a) in (("u", u), ("v", v), ("u_rot", u_rot), ("v_rot", v_rot))
        size(a) == gsz || throw(DimensionMismatch("$nm has size $(size(a)), grid expects $gsz"))
    end
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend)
    return compute_Π_decomposed!(
        SphericalPiDecomposedWorkspace(grid), u, v, u_rot, v_rot, grid, kernel, scale;
        filter_plan = plan, deriv_plan = _default_deriv_plan(output_grid(grid, plan)),
    )
end

"""
    compute_Π_decomposed!(ws::SphericalPiDecomposedWorkspace, u, v, u_rot, v_rot, grid, kernel, scale; ...)
        -> (; total, rotational, cross, divergent)

In-place spherical [`compute_Π_decomposed`](@ref). One batched apply carries the six planetary
components of both parts; three more carry the three stresses.
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
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}}
    _require_tangent_pair(grid, "compute_Π_decomposed!")
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend) :
        filter_plan
    og = output_grid(grid, plan)
    dplan = _resolve_deriv_plan(deriv_plan, og)
    geo = FlowGeometries.Grids.grid_geometry(grid)
    pr, pd, br, bd, loc = ws.pr, ws.pd, ws.br, ws.bd, ws.loc

    # The divergent part is the complement of the supplied rotational one; both go to planetary
    # Cartesian, where filtering commutes with the moment and residual operations.
    @inbounds for I in CartesianIndices(ws.total)
        i = Tuple(I)
        if FlowGeometries.Grids.isactive(grid, i...)
            λ, φ = FlowGeometries.Grids.coords(grid, i...)
            cr = FlowGeometries.Geometry.vector_to_cartesian(geo, u_rot[I], v_rot[I], λ, φ)
            cd = FlowGeometries.Geometry.vector_to_cartesian(
                geo, u[I] - u_rot[I], v[I] - v_rot[I], λ, φ,
            )
            pr[1][I] = cr[1]; pr[2][I] = cr[2]; pr[3][I] = cr[3]
            pd[1][I] = cd[1]; pd[2][I] = cd[2]; pd[3][I] = cd[3]
        else
            for c in 1:3
                pr[c][I] = zero(T); pd[c][I] = zero(T)
            end
        end
    end

    Filtering.filter_apply_batch!((br..., bd...), (pr..., pd...), plan)
    _sph_pair_moments!(ws.τRR, ws.τX, ws.τDD, pr, pd, br, bd, ws.prod, plan, og, geo)

    # Each part's filtered velocity back in the local frame, where its strain is taken.
    @inbounds for I in CartesianIndices(ws.total)
        i = Tuple(I)
        if FlowGeometries.Grids.isactive(og, i...)
            λ, φ = FlowGeometries.Grids.coords(grid, i...)
            lr = FlowGeometries.Geometry.vector_from_cartesian(geo, br[1][I], br[2][I], br[3][I], λ, φ)
            ld = FlowGeometries.Geometry.vector_from_cartesian(geo, bd[1][I], bd[2][I], bd[3][I], λ, φ)
            loc[1][I] = lr[1]; loc[2][I] = lr[2]
            loc[3][I] = ld[1]; loc[4][I] = ld[2]
        else
            for c in 1:4
                loc[c][I] = zero(T)
            end
        end
    end

    _strain_into!(ws.SR[1], ws.SR[2], ws.SR[3], loc[1], loc[2], ws.tmp, og, dplan, T)
    _strain_into!(ws.SD[1], ws.SD[2], ws.SD[3], loc[3], loc[4], ws.tmp, og, dplan, T)

    mask = FlowGeometries.Grids.mask(og)
    SR, SD, τRR, τX, τDD = ws.SR, ws.SD, ws.τRR, ws.τX, ws.τDD
    @inbounds for I in CartesianIndices(ws.total)
        if mask[I]
            rr = _sfs_contraction(SR[1][I], SR[2][I], SR[3][I], τRR[1][I], τRR[2][I], τRR[3][I])
            dd = _sfs_contraction(SD[1][I], SD[2][I], SD[3][I], τDD[1][I], τDD[2][I], τDD[3][I])
            x = _sfs_contraction(SR[1][I], SR[2][I], SR[3][I], τDD[1][I], τDD[2][I], τDD[3][I]) +
                _sfs_contraction(SD[1][I], SD[2][I], SD[3][I], τRR[1][I], τRR[2][I], τRR[3][I]) +
                _sfs_contraction(SR[1][I], SR[2][I], SR[3][I], τX[1][I], τX[2][I], τX[3][I]) +
                _sfs_contraction(SD[1][I], SD[2][I], SD[3][I], τX[1][I], τX[2][I], τX[3][I])
            # `total` is summed from the stored channels, in their returned order, so
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
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    _require_tangent_pair(grid, "compute_Π_decomposed!")
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend) :
        filter_plan
    og = output_grid(grid, plan)
    dplan = _resolve_deriv_plan(deriv_plan, og)

    # Divergent (irrotational) part is the complement of the supplied rotational part.
    @. ws.u_div = u - u_rot
    @. ws.v_div = v - v_rot

    # The four filtered means each feed a self stress, the cross stress AND a strain, so they are
    # filtered once. One batch, so a scattered engine derives each neighbourhood once for all four.
    Filtering.filter_apply_batch!(
        (ws.ūr, ws.v̄r, ws.ūd, ws.v̄d), (u_rot, v_rot, ws.u_div, ws.v_div), plan,
    )

    # Self stresses τ(a,a)_ij = ⟨a_i a_j⟩ - ā_i ā_j.
    _second_moment!(ws.τRR_xx, u_rot, u_rot, ws.ūr, ws.ūr, ws.prod, ws.fbuf, plan)
    _second_moment!(ws.τRR_xy, u_rot, v_rot, ws.ūr, ws.v̄r, ws.prod, ws.fbuf, plan)
    _second_moment!(ws.τRR_yy, v_rot, v_rot, ws.v̄r, ws.v̄r, ws.prod, ws.fbuf, plan)
    _second_moment!(ws.τDD_xx, ws.u_div, ws.u_div, ws.ūd, ws.ūd, ws.prod, ws.fbuf, plan)
    _second_moment!(ws.τDD_xy, ws.u_div, ws.v_div, ws.ūd, ws.v̄d, ws.prod, ws.fbuf, plan)
    _second_moment!(ws.τDD_yy, ws.v_div, ws.v_div, ws.v̄d, ws.v̄d, ws.prod, ws.fbuf, plan)

    # Cross stress τ(rot,div) + τ(div,rot). The diagonals are symmetric under the swap so they just
    # double; the off-diagonal genuinely needs both orderings.
    _second_moment!(ws.τX_xx, u_rot, ws.u_div, ws.ūr, ws.ūd, ws.prod, ws.fbuf, plan)
    @. ws.τX_xx *= T(2)
    _second_moment!(ws.τX_yy, v_rot, ws.v_div, ws.v̄r, ws.v̄d, ws.prod, ws.fbuf, plan)
    @. ws.τX_yy *= T(2)
    _second_moment!(ws.τX_xy, u_rot, ws.v_div, ws.ūr, ws.v̄d, ws.prod, ws.fbuf, plan)
    _second_moment!(ws.scratch, ws.u_div, v_rot, ws.ūd, ws.v̄r, ws.prod, ws.fbuf, plan)
    @. ws.τX_xy += ws.scratch

    _strain_into!(ws.SR_xx, ws.SR_xy, ws.SR_yy, ws.ūr, ws.v̄r, ws.scratch, og, dplan, T)
    _strain_into!(ws.SD_xx, ws.SD_xy, ws.SD_yy, ws.ūd, ws.v̄d, ws.scratch, og, dplan, T)

    mask = FlowGeometries.Grids.mask(og)
    @. ws.Πrr = ifelse(mask,
        -(ws.SR_xx * ws.τRR_xx + T(2) * ws.SR_xy * ws.τRR_xy + ws.SR_yy * ws.τRR_yy), zero(T))
    @. ws.Πdd = ifelse(mask,
        -(ws.SD_xx * ws.τDD_xx + T(2) * ws.SD_xy * ws.τDD_xy + ws.SD_yy * ws.τDD_yy), zero(T))
    # Masking is linear, so the four interaction contractions sum inside one `ifelse`.
    @. ws.Πx = ifelse(mask, -(
        ws.SR_xx * ws.τDD_xx + T(2) * ws.SR_xy * ws.τDD_xy + ws.SR_yy * ws.τDD_yy +
        ws.SD_xx * ws.τRR_xx + T(2) * ws.SD_xy * ws.τRR_xy + ws.SD_yy * ws.τRR_yy +
        ws.SR_xx * ws.τX_xx  + T(2) * ws.SR_xy * ws.τX_xy  + ws.SR_yy * ws.τX_yy  +
        ws.SD_xx * ws.τX_xx  + T(2) * ws.SD_xy * ws.τX_xy  + ws.SD_yy * ws.τX_yy
    ), zero(T))
    @. ws.total = ws.Πrr + ws.Πx + ws.Πdd
    return (; total = ws.total, rotational = ws.Πrr, cross = ws.Πx, divergent = ws.Πdd)
end
