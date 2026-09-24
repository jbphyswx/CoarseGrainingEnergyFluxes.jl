# ---------------------------------------------------------------------------
# Favre (density-weighted) coarse-graining — Aluie 2013
# ---------------------------------------------------------------------------

"""
    FavreWorkspace(grid)

Scratch for [`compressible_flux!`](@ref): the filtered density and pressure, the Favre velocities, the
unweighted velocities, the three Favre stress components, the two unweighted mass-flux components, the
four velocity gradients, the two pressure gradients, and the three output fields.
"""
struct FavreWorkspace{T<:AbstractFloat, A<:AbstractArray{T}}
    ρ̄::A; P̄::A
    ũ::A; ṽ::A
    ū::A; v̄::A
    τxx::A; τxy::A; τyy::A
    mx::A; my::A          # τ̄(ρ, u_j): the unweighted subscale mass flux
    ux::A; uy::A; vx::A; vy::A
    Px::A; Py::A
    prod::A; fbuf::A
    # The five raw products (ρu, ρv, ρuu, ρuv, ρvv) are all functions of the inputs alone, so they are
    # formed together and filtered in one batched apply.
    prods::NTuple{5,A}
    Π::A; Λ::A; PD::A
end

function FavreWorkspace(grid::FlowGeometries.Grids.AbstractGrid{G,T}) where {T<:AbstractFloat, G}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    z() = zeros(T, gsz)
    return FavreWorkspace(
        z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z(),
        z(), z(), ntuple(_ -> z(), 5), z(), z(), z(),
    )
end

"""
    compressible_flux!(ws, u, v, ρ, P, grid, kernel, scale; filter_plan=nothing, deriv_plan=nothing, ...)
        -> (; Π, Λ, pressure_dilatation, ρ̄, P̄, ũ, ṽ)

In-place [`compressible_flux`](@ref). Returns views of `ws`'s buffers, valid until the next call on the
same workspace. With `ws` and both plans supplied, a repeated evaluation allocates nothing.
"""
function compressible_flux!(
    ws::FavreWorkspace{T},
    u::AbstractVecOrMat,
    v::AbstractVecOrMat,
    ρ::AbstractVecOrMat,
    P::AbstractVecOrMat,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    filter_plan::Union{Nothing,Filtering.AbstractFilterPlan} = nothing,
    deriv_plan::Union{Nothing,AnyDerivPlan} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    _require_tangent_pair(grid, "compressible_flux!")
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend) :
        filter_plan
    og = output_grid(grid, plan)
    dplan = _resolve_deriv_plan(deriv_plan, og)

    # ρ̄, P̄ and the UNWEIGHTED velocities. `ū`/`v̄` are not a convenience: the budget's pressure term is
    # `P̄ ∇·ū`, with the unweighted divergence, and `τ̄(ρ,u_j)` needs `ū_j` too.
    Filtering.filter_apply_batch!((ws.ρ̄, ws.P̄, ws.ū, ws.v̄), (ρ, P, u, v), plan)

    # The five mass-weighted products, filtered in one pass. They land directly in the buffers that
    # will hold the results, and each is finished in place afterwards.
    p = ws.prods
    @. p[1] = ρ * u
    @. p[2] = ρ * v
    @. p[3] = ρ * u * u
    @. p[4] = ρ * u * v
    @. p[5] = ρ * v * v
    Filtering.filter_apply_batch!((ws.mx, ws.my, ws.τxx, ws.τxy, ws.τyy), p, plan)

    # Favre velocities ũ_i = (ρu_i)‾/ρ̄, zero where no fluid lies under the kernel.
    @. ws.ũ = _favre(ws.mx, ws.ρ̄)
    @. ws.ṽ = _favre(ws.my, ws.ρ̄)

    # ρ̄ ∂_j ũ_i = ∂_j (ρu_i)‾ − ũ_i ∂_j ρ̄. Both fields differenced here are defined at every cell, so no
    # stencil reads ũ where ρ̄ = 0. `Px`/`Py` hold ∇ρ̄ until ∇P̄ replaces it.
    _grad2!(ws.ux, ws.uy, ws.mx, og, dplan)
    _grad2!(ws.vx, ws.vy, ws.my, og, dplan)
    _grad2!(ws.Px, ws.Py, ws.ρ̄, og, dplan)
    @. ws.ux -= ws.ũ * ws.Px
    @. ws.uy -= ws.ũ * ws.Py
    @. ws.vx -= ws.ṽ * ws.Px
    @. ws.vy -= ws.ṽ * ws.Py

    # The unweighted subscale mass flux τ̄(ρ,u_i) = (ρu_i)‾ − ρ̄ū_i, which baropycnal work contracts.
    @. ws.mx -= ws.ρ̄ * ws.ū
    @. ws.my -= ws.ρ̄ * ws.v̄

    # Favre stress τ̃(u_i,u_j) = (ρu_iu_j)‾/ρ̄ − ũ_iũ_j.
    @. ws.τxx = _favre(ws.τxx, ws.ρ̄) - ws.ũ * ws.ũ
    @. ws.τxy = _favre(ws.τxy, ws.ρ̄) - ws.ũ * ws.ṽ
    @. ws.τyy = _favre(ws.τyy, ws.ρ̄) - ws.ṽ * ws.ṽ

    mask = FlowGeometries.Grids.mask(og)
    # Π = −ρ̄ ∂_j ũ_i τ̃(u_i,u_j), summed over i,j; τ̃ is symmetric so the two off-diagonals combine.
    @. ws.Π = ifelse(mask,
        -(ws.ux * ws.τxx + (ws.uy + ws.vx) * ws.τxy + ws.vy * ws.τyy), zero(T))
    # Λ = (1/ρ̄) ∂_j P̄ · τ̄(ρ,u_j) — baropycnal work.
    _grad2!(ws.Px, ws.Py, ws.P̄, og, dplan)
    @. ws.Λ = ifelse(mask, _favre(ws.Px * ws.mx + ws.Py * ws.my, ws.ρ̄), zero(T))
    # P̄ ∇·ū, with the unweighted divergence. `ws.prod`/`ws.fbuf` are free again here, and `ws.τxx` is
    # spent, so it takes the two gradient components neither term reads.
    _grad2!(ws.prod, ws.τxx, ws.ū, og, dplan)
    _grad2!(ws.τxx, ws.fbuf, ws.v̄, og, dplan)
    @. ws.PD = ifelse(mask, ws.P̄ * (ws.prod + ws.fbuf), zero(T))

    return (; Π = ws.Π, Λ = ws.Λ, pressure_dilatation = ws.PD,
            ρ̄ = ws.ρ̄, P̄ = ws.P̄, ũ = ws.ũ, ṽ = ws.ṽ)
end

"""
    compressible_flux(u, v, ρ, P, grid, kernel, scale; backend=AutoBackend(), mask_strategy=ZeroFill())
        -> (; Π, Λ, pressure_dilatation, ρ̄, P̄, ũ, ṽ)

The variable-density (Favre) cross-scale energy budget of Aluie (2013). Returns the three terms of that
budget which act on the large-scale kinetic energy `ρ̄|ũ|²/2`, plus the filtered fields they are built
from.

# Favre filtering

`f̃ ≡ (ρf)‾/ρ̄` is the density-weighted filter. It exists because it is the one that makes the filtered
continuity equation close exactly, `∂_tρ̄ + ∂_i(ρ̄ũ_i) = 0`; the unweighted filter does not. It is linear
but **does not commute with derivatives**, so the two filters are not interchangeable and the budget
genuinely needs both — which is the source of the trap below.

# The three terms

```
Π = −ρ̄ ∂_j ũ_i τ̃(u_i,u_j) ,   τ̃(u_i,u_j) = (ρu_iu_j)‾/ρ̄ − ũ_iũ_j        deformation work
Λ = (1/ρ̄) ∂_j P̄ · τ̄(ρ,u_j) ,  τ̄(ρ,u_j)   = (ρu_j)‾ − ρ̄ū_j              baropycnal work
                                                                        (τ̄ is UNWEIGHTED)
P̄ ∇·ū                                                                   pressure dilatation
```

All three are per unit volume, W m⁻³ with `ρ` in kg m⁻³ and `P` in Pa; at a constant `ρ₀`, `Π` is `ρ₀`
times the specific flux [`compute_Π!`](@ref) returns.

`Π` and `Λ` both pit a large-scale field against small-scale fluctuations, so **both transfer energy
across scales**. `P̄∇·ū` involves only large scales and cannot — it is a conversion between kinetic and
internal energy at the resolved scale, not a cascade term.

# The trap

`Λ` is frequently absorbed into the pressure term by writing it as `P̄∇·ũ` (plus a transport term) and
then dismissed as "large-scale pressure dilatation that needs no modelling". That is wrong: the budget
term is `P̄∇·ū` with the **unweighted** divergence, and writing `∇·ũ` silently destroys `Λ` — a genuine
cross-scale transfer. This implementation keeps them separate and uses `ū` for the dilatation; the
suite asserts that `Λ` is non-zero for a baroclinic configuration, so it cannot be quietly dropped.

# Masked and bounded grids

Under `ZeroFill` a masked cell and the domain exterior hold no fluid, so `ρ̄` is the fluid mass under
the kernel: the Favre fields are density-weighted means over the fluid, zero where none lies under the
kernel, and `ρ̄ ∂_j ũ_i` is formed as `∂_j (ρu_i)‾ − ũ_i ∂_j ρ̄`, which differences only fields defined
at every cell. Constant `ρ` collapses the budget to `ρ·Π` wherever the fluid fills the kernel, which
under `Deformable` is every active cell. `ρ` is read, and must be positive, on active cells only.

# Asymptotics

For a smooth field, Lees & Aluie (2019) give `Λ ≈ (C₂ℓ²/ρ̄)·c_d·[∇P̄·S̄·∇ρ̄ + ½ ω̄·(∇ρ̄ × ∇P̄)]` with
`C₂` the kernel's second moment — a strain-generation part plus a **baroclinic** part that survives
even in pure solenoidal flow. That `C₂ ≠ 0` requirement is another reason the flux framework wants a
kernel with a NON-vanishing second moment; see [`Kernels.HighOrderKernel`](@ref) for the kernels that
deliberately give it up.

# References
- Aluie, H. (2013). Scale decomposition in compressible turbulence. *Physica D* 247, 54–65.
- Lees, A., & Aluie, H. (2019). Baropycnal work: a mechanism for energy transfer across scales.
  *Fluids* 4, 92. doi:10.3390/fluids4020092
"""
function compressible_flux(
    u::AbstractVecOrMat,
    v::AbstractVecOrMat,
    ρ::AbstractVecOrMat,
    P::AbstractVecOrMat,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    for (nm, a) in (("u", u), ("v", v), ("ρ", ρ), ("P", P))
        size(a) == gsz || throw(DimensionMismatch("$nm has size $(size(a)), grid expects $gsz"))
    end
    _require_positive_density(ρ, grid)
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend)
    return compressible_flux!(
        FavreWorkspace(grid), u, v, ρ, P, grid, kernel, scale;
        filter_plan = plan, deriv_plan = _default_deriv_plan(output_grid(grid, plan)),
    )
end

"""
    SphericalFavreWorkspace(grid)

Scratch for the spherical [`compressible_flux!`](@ref): the planetary velocity, the mass flux and the
Favre velocity in those coordinates, six product buffers, the six components of the Favre stress, the
local velocity pairs, the resolved strain, the two pressure gradients and the three output fields.
"""
struct SphericalFavreWorkspace{T<:AbstractFloat, A<:AbstractArray{T}}
    p::NTuple{3,A}      # planetary velocity
    bp::NTuple{3,A}     # its unweighted filter, ū in planetary components
    m::NTuple{3,A}      # (ρu_i)‾, then the unweighted subscale mass flux τ̄(ρ,u_i)
    up::NTuple{3,A}     # Favre velocity in planetary components
    prod::NTuple{6,A}
    τ::NTuple{6,A}      # Favre stress, rotated to local in slots 1-3
    loc::NTuple{4,A}    # (ũ_e, ṽ_n, ū_e, v̄_n)
    mloc::NTuple{2,A}   # τ̄(ρ,u) in local (east, north)
    S::NTuple{3,A}      # (xx, xy, yy)
    ρ̄::A; P̄::A; Px::A; Py::A; tmp::A
    Π::A; Λ::A; PD::A
end

function SphericalFavreWorkspace(
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    z() = zeros(T, gsz)
    t(n) = ntuple(_ -> z(), n)
    return SphericalFavreWorkspace(
        t(3), t(3), t(3), t(3), t(6), t(6), t(4), t(2), t(3),
        z(), z(), z(), z(), z(), z(), z(), z(),
    )
end

"""
    compressible_flux(u, v, ρ, P, grid::AbstractGrid{<:SphericalGeometry}, kernel, scale; ...)
        -> (; Π, Λ, pressure_dilatation, ρ̄, P̄, ũ, ṽ)

Spherical form of the variable-density (Favre) budget, on any grid resolving two tangent directions.

The three terms are the ones the Cartesian method computes; each piece that carries a direction is
built in planetary-Cartesian coordinates and rotated back to the local frame, since a local
(east, north) pair filtered component-wise is not a filtered vector (Aluie 2019). That applies to the
Favre velocity `ũ`, to the Favre stress `τ̃`, and to the unweighted subscale mass flux `τ̄(ρ,u_j)` that
baropycnal work contracts against.

Two spatial operators pick up the local frame's curvature: the resolved Favre strain that `Π`
contracts, and the unweighted divergence in the pressure-dilatation term,

    ∇·ū = ∂ū_e/∂x + ∂v̄_n/∂y − v̄_n tanφ/R ,

taken here as the trace of the unweighted strain, so it carries that term by construction.
"""
function compressible_flux(
    u::AbstractVecOrMat,
    v::AbstractVecOrMat,
    ρ::AbstractVecOrMat,
    P::AbstractVecOrMat,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    for (nm, a) in (("u", u), ("v", v), ("ρ", ρ), ("P", P))
        size(a) == gsz || throw(DimensionMismatch("$nm has size $(size(a)), grid expects $gsz"))
    end
    _require_positive_density(ρ, grid)
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend)
    return compressible_flux!(
        SphericalFavreWorkspace(grid), u, v, ρ, P, grid, kernel, scale;
        filter_plan = plan, deriv_plan = _default_deriv_plan(output_grid(grid, plan)),
    )
end

"""
    compressible_flux!(ws::SphericalFavreWorkspace, u, v, ρ, P, grid, kernel, scale; ...)

In-place spherical [`compressible_flux`](@ref). Two batched applies carry the whole budget: eight
fields for the density, the pressure and both velocity forms, then the six mass-weighted products.
"""
function compressible_flux!(
    ws::SphericalFavreWorkspace{T},
    u::AbstractVecOrMat,
    v::AbstractVecOrMat,
    ρ::AbstractVecOrMat,
    P::AbstractVecOrMat,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    filter_plan::Union{Nothing,Filtering.AbstractFilterPlan} = nothing,
    deriv_plan::Union{Nothing,AnyDerivPlan} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}}
    _require_tangent_pair(grid, "compressible_flux!")
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend) :
        filter_plan
    og = output_grid(grid, plan)
    dplan = _resolve_deriv_plan(deriv_plan, og)
    geo = FlowGeometries.Grids.grid_geometry(grid)
    p, bp, m, up, pr, τ, loc = ws.p, ws.bp, ws.m, ws.up, ws.prod, ws.τ, ws.loc

    @inbounds for I in CartesianIndices(ws.Π)
        i = Tuple(I)
        if FlowGeometries.Grids.isactive(grid, i...)
            λ, φ = FlowGeometries.Grids.coords(grid, i...)
            c = FlowGeometries.Geometry.vector_to_cartesian(geo, u[I], v[I], λ, φ)
            p[1][I] = c[1]; p[2][I] = c[2]; p[3][I] = c[3]
        else
            for c in 1:3
                p[c][I] = zero(T)
            end
        end
    end

    for c in 1:3
        @. pr[c] = ρ * p[c]
    end
    Filtering.filter_apply_batch!(
        (ws.ρ̄, ws.P̄, bp[1], bp[2], bp[3], m[1], m[2], m[3]),
        (ρ, P, p[1], p[2], p[3], pr[1], pr[2], pr[3]), plan,
    )

    for (k, (i1, i2)) in enumerate(_SYM3)
        @. pr[k] = ρ * p[i1] * p[i2]
    end
    Filtering.filter_apply_batch!(τ, pr, plan)

    # Favre velocity ũ_i = (ρu_i)‾/ρ̄, read before `m` becomes the unweighted mass flux τ̄(ρ,u_i).
    for c in 1:3
        @. up[c] = _favre(m[c], ws.ρ̄)
        @. m[c] = m[c] - ws.ρ̄ * bp[c]
    end
    for (k, (i1, i2)) in enumerate(_SYM3)
        @. τ[k] = _favre(τ[k], ws.ρ̄) - up[i1] * up[i2]
    end

    # The stress to the local frame, and every vector with it.
    @inbounds for I in CartesianIndices(ws.Π)
        i = Tuple(I)
        if FlowGeometries.Grids.isactive(og, i...)
            λ, φ = FlowGeometries.Grids.coords(grid, i...)
            _store_local!(τ, I, _rotate_stress_to_local_enr(
                geo, τ[1][I], τ[2][I], τ[3][I], τ[4][I], τ[5][I], τ[6][I], λ, φ,
            ), Val(3))
            lu = FlowGeometries.Geometry.vector_from_cartesian(geo, up[1][I], up[2][I], up[3][I], λ, φ)
            lb = FlowGeometries.Geometry.vector_from_cartesian(geo, bp[1][I], bp[2][I], bp[3][I], λ, φ)
            lm = FlowGeometries.Geometry.vector_from_cartesian(geo, m[1][I], m[2][I], m[3][I], λ, φ)
            loc[1][I] = lu[1]; loc[2][I] = lu[2]
            loc[3][I] = lb[1]; loc[4][I] = lb[2]
            ws.mloc[1][I] = lm[1]; ws.mloc[2][I] = lm[2]
        else
            τ[1][I] = zero(T); τ[2][I] = zero(T); τ[3][I] = zero(T)
            for c in 1:4
                loc[c][I] = zero(T)
            end
            ws.mloc[1][I] = zero(T); ws.mloc[2][I] = zero(T)
        end
    end

    mask = FlowGeometries.Grids.mask(og)
    # Π = −ρ̄ ∂_j ũ_i τ̃_ij; τ̃ is symmetric, so only the symmetric part of the gradient survives and
    # this is −ρ̄ S̃:τ̃ with the curvature-carrying strain. The strain is linear in the velocity, so
    # ρ̄ S(ũ) = S(ρ̄ũ) − sym(ũ ⊗ ∇ρ̄), which differences only fields defined at every cell.
    @. pr[1] = ws.ρ̄ * loc[1]
    @. pr[2] = ws.ρ̄ * loc[2]
    _strain_into!(ws.S[1], ws.S[2], ws.S[3], pr[1], pr[2], ws.tmp, og, dplan, T)
    _grad2!(ws.Px, ws.Py, ws.ρ̄, og, dplan)
    @. ws.S[1] -= loc[1] * ws.Px
    @. ws.S[2] -= T(0.5) * (loc[1] * ws.Py + loc[2] * ws.Px)
    @. ws.S[3] -= loc[2] * ws.Py
    @. ws.Π = ifelse(mask, -_sfs_contraction(
        ws.S[1], ws.S[2], ws.S[3], τ[1], τ[2], τ[3]), zero(T))

    _grad2!(ws.Px, ws.Py, ws.P̄, og, dplan)
    @. ws.Λ = ifelse(mask, _favre(ws.Px * ws.mloc[1] + ws.Py * ws.mloc[2], ws.ρ̄), zero(T))

    # ∇·ū is the trace of the unweighted strain, so the frame's curvature term comes with it.
    _strain_into!(ws.S[1], ws.S[2], ws.S[3], loc[3], loc[4], ws.tmp, og, dplan, T)
    @. ws.PD = ifelse(mask, ws.P̄ * (ws.S[1] + ws.S[3]), zero(T))

    return (; Π = ws.Π, Λ = ws.Λ, pressure_dilatation = ws.PD,
            ρ̄ = ws.ρ̄, P̄ = ws.P̄, ũ = loc[1], ṽ = loc[2])
end

"""
    Favre3DWorkspace(grid)

Scratch for the true-3-D [`compressible_flux!`](@ref): the velocity triple, its unweighted filter, the
mass flux and the Favre velocity, six product buffers, the six components of the Favre stress and of
the resolved strain, the three pressure gradients and the three output fields.
"""
struct Favre3DWorkspace{T<:AbstractFloat, A<:AbstractArray{T,3}}
    p::NTuple{3,A}      # the velocity triple, planetary on a shell
    bp::NTuple{3,A}     # its unweighted filter
    m::NTuple{3,A}      # (ρu_i)‾, then the unweighted subscale mass flux τ̄(ρ,u_i)
    up::NTuple{3,A}     # Favre velocity
    prod::NTuple{6,A}
    τ::NTuple{6,A}
    S::NTuple{6,A}
    loc::NTuple{3,A}    # Favre velocity in the local frame
    mloc::NTuple{3,A}   # τ̄(ρ,u) in the local frame
    g::NTuple{3,A}      # ∇P̄
    ρ̄::A; P̄::A; tmp::A
    Π::A; Λ::A; PD::A
end

function Favre3DWorkspace(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,3},
) where {T<:AbstractFloat, G}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    z() = zeros(T, gsz)
    t(n) = ntuple(_ -> z(), n)
    return Favre3DWorkspace(
        t(3), t(3), t(3), t(3), t(6), t(6), t(6), t(3), t(3), t(3), z(), z(), z(), z(), z(), z(),
    )
end

"""
    compressible_flux(u, v, w, ρ, P, grid::StructuredGrid{T,G,3}, kernel, scale; ...)
        -> (; Π, Λ, pressure_dilatation, ρ̄, P̄, ũ, ṽ, w̃)

True three-dimensional variable-density (Favre) budget. The three terms are the ones the 2-D methods
compute, over all six independent components of the `3×3` Favre stress and the full `3×3` resolved
strain — including the genuine vertical derivatives the layer-stack path drops:

```
Π = −ρ̄ S̃_ij τ̃(u_i,u_j) ,   Λ = (1/ρ̄) ∂_j P̄ · τ̄(ρ,u_j) ,   P̄ ∇·ū
```

On a Cartesian volume the components are `(x, y, z)` as given. On a spherical shell every quantity
carrying a direction goes through planetary-Cartesian coordinates and comes back to local
`(east, north, radial)`, and the strain carries the shell's curvature terms — the convention the
true-3-D [`compute_Π!`](@ref) uses. `∇·ū` is taken as the trace of the unweighted strain, so it picks
those terms up with it.
"""
function compressible_flux(
    u::AbstractArray{<:Any,3},
    v::AbstractArray{<:Any,3},
    w::AbstractArray{<:Any,3},
    ρ::AbstractArray{<:Any,3},
    P::AbstractArray{<:Any,3},
    grid::FlowGeometries.Grids.StructuredGrid{T,G,3},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    for (nm, a) in (("u", u), ("v", v), ("w", w), ("ρ", ρ), ("P", P))
        size(a) == gsz || throw(DimensionMismatch("$nm has size $(size(a)), grid expects $gsz"))
    end
    _require_positive_density(ρ, grid)
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend)
    return compressible_flux!(
        Favre3DWorkspace(grid), u, v, w, ρ, P, grid, kernel, scale;
        filter_plan = plan, deriv_plan = Derivatives.StencilPlan(grid),
    )
end

"""
    compressible_flux!(ws::Favre3DWorkspace, u, v, w, ρ, P, grid, kernel, scale; ...)

In-place true-3-D [`compressible_flux`](@ref). Two batched applies carry the budget: eight fields for
the density, the pressure and both velocity forms, then the six mass-weighted products.
"""
function compressible_flux!(
    ws::Favre3DWorkspace{T},
    u::AbstractArray{<:Any,3},
    v::AbstractArray{<:Any,3},
    w::AbstractArray{<:Any,3},
    ρ::AbstractArray{<:Any,3},
    P::AbstractArray{<:Any,3},
    grid::FlowGeometries.Grids.StructuredGrid{T,G,3},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    filter_plan::Union{Nothing,Filtering.AbstractFilterPlan} = nothing,
    deriv_plan::Union{Nothing,Derivatives.StencilPlan} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend) :
        filter_plan
    dplan = deriv_plan === nothing ? Derivatives.StencilPlan(grid) : deriv_plan
    p, bp, m, up, pr, τ = ws.p, ws.bp, ws.m, ws.up, ws.prod, ws.τ

    _fill_tau3_velocity!(p, u, v, w, grid, T)
    for c in 1:3
        @. pr[c] = ρ * p[c]
    end
    Filtering.filter_apply_batch!(
        (ws.ρ̄, ws.P̄, bp[1], bp[2], bp[3], m[1], m[2], m[3]),
        (ρ, P, p[1], p[2], p[3], pr[1], pr[2], pr[3]), plan,
    )
    for (k, (i1, i2)) in enumerate(_SYM3)
        @. pr[k] = ρ * p[i1] * p[i2]
    end
    Filtering.filter_apply_batch!(τ, pr, plan)

    # Favre velocity ũ_i = (ρu_i)‾/ρ̄, read before `m` becomes the unweighted mass flux τ̄(ρ,u_i).
    for c in 1:3
        @. up[c] = _favre(m[c], ws.ρ̄)
        @. m[c] = m[c] - ws.ρ̄ * bp[c]
    end
    for (k, (i1, i2)) in enumerate(_SYM3)
        @. τ[k] = _favre(τ[k], ws.ρ̄) - up[i1] * up[i2]
    end

    # On a shell the stress and both vectors come back to the local frame; on a Cartesian volume they
    # are already in it.
    og = output_grid(grid, plan)
    _rotate_tau3_to_local!((τ,), og)
    _localize_triple!(ws.loc, up, og, T)
    _localize_triple!(ws.mloc, m, og, T)

    # ρ̄ S(ũ) = S(ρ̄ũ) − sym(ũ ⊗ ∇ρ̄): the strain and its curvature terms are linear in the velocity,
    # and both fields differenced here are defined at every cell. `g` holds ∇ρ̄ until ∇P̄ replaces it.
    mask = FlowGeometries.Grids.mask(og)
    ρũ = (pr[1], pr[2], pr[3])
    for c in 1:3
        @. ρũ[c] = ws.ρ̄ * ws.loc[c]
    end
    _strain3_into!(ws.S, ρũ, ws.tmp, og, dplan, T)
    Derivatives.ddx!(ws.g[1], ws.ρ̄, og, dplan)
    Derivatives.ddy!(ws.g[2], ws.ρ̄, og, dplan)
    Derivatives.ddz!(ws.g[3], ws.ρ̄, og, dplan)
    for (k, (i1, i2)) in enumerate(_SYM3)
        @. ws.S[k] -= T(0.5) * (ws.loc[i1] * ws.g[i2] + ws.loc[i2] * ws.g[i1])
    end
    @. ws.Π = ifelse(mask, -_sfs_contraction(
        ws.S[1], ws.S[2], ws.S[3], ws.S[4], ws.S[5], ws.S[6],
        τ[1], τ[2], τ[3], τ[4], τ[5], τ[6]), zero(T))

    Derivatives.ddx!(ws.g[1], ws.P̄, og, dplan)
    Derivatives.ddy!(ws.g[2], ws.P̄, og, dplan)
    Derivatives.ddz!(ws.g[3], ws.P̄, og, dplan)
    @. ws.Λ = ifelse(mask,
        _favre(ws.g[1] * ws.mloc[1] + ws.g[2] * ws.mloc[2] + ws.g[3] * ws.mloc[3], ws.ρ̄), zero(T))

    # ∇·ū is the trace of the unweighted strain, so a shell's curvature terms come with it.
    ub = (pr[4], pr[5], pr[6])
    _localize_triple!(ub, bp, og, T)
    _strain3_into!(ws.S, ub, ws.tmp, og, dplan, T)
    @. ws.PD = ifelse(mask, ws.P̄ * (ws.S[1] + ws.S[4] + ws.S[6]), zero(T))

    return (; Π = ws.Π, Λ = ws.Λ, pressure_dilatation = ws.PD,
            ρ̄ = ws.ρ̄, P̄ = ws.P̄, ũ = ws.loc[1], ṽ = ws.loc[2], w̃ = ws.loc[3])
end

# A Cartesian volume's triple is already local.
function _localize_triple!(
    dst, src, ::FlowGeometries.Grids.StructuredGrid{T2,G,3}, ::Type{T},
) where {T, T2, G<:FlowGeometries.Geometry.CartesianGeometry{T2}}
    for c in 1:3
        @. dst[c] = src[c]
    end
    return nothing
end

function _localize_triple!(
    dst, src, grid::FlowGeometries.Grids.StructuredGrid{T2,G,3}, ::Type{T},
) where {T, T2, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T2}}
    geo = FlowGeometries.Grids.grid_geometry(grid)
    @inbounds for I in CartesianIndices(dst[1])
        i = Tuple(I)
        if FlowGeometries.Grids.isactive(grid, i...)
            λ, φ, _ = FlowGeometries.Grids.coords(grid, i...)
            l = FlowGeometries.Geometry.vector_from_cartesian(
                geo, src[1][I], src[2][I], src[3][I], λ, φ,
            )
            dst[1][I] = l[1]; dst[2][I] = l[2]; dst[3][I] = l[3]
        else
            dst[1][I] = zero(T); dst[2][I] = zero(T); dst[3][I] = zero(T)
        end
    end
    return nothing
end

# A density-weighted mean `x/ρ̄`, zero where no fluid lies under the kernel (`ρ̄ = 0` there under
# `ZeroFill`, and at a masked target under `Deformable`).
@inline _favre(x, ρ̄) = ifelse(ρ̄ > zero(ρ̄), x / ρ̄, zero(x))

# ρ is read on active cells only; a masked cell's value is never filtered.
function _require_positive_density(ρ::AbstractArray, grid::FlowGeometries.Grids.AbstractGrid)
    mask = FlowGeometries.Grids.mask(grid)
    lo = typemax(float(eltype(ρ)))
    @inbounds for I in eachindex(ρ, mask)
        mask[I] && (lo = min(lo, ρ[I]))
    end
    lo > 0 || throw(ArgumentError(
        "Favre filtering divides by the filtered density, so ρ must be strictly positive on every " *
        "active cell; got a minimum of $lo.",
    ))
    return nothing
end

"""
    favre_filter!(out, tmp, f, ρ, ρ̄, plan) -> out

`f̃ = (ρf)‾/ρ̄`, given an already-filtered `ρ̄` and a scratch array `tmp`; zero where `ρ̄ = 0`. The
building block of [`compressible_flux`](@ref), exposed because a caller filtering their own tracer
Favre-style should not have to reimplement it (and get the weighting backwards).
"""
function favre_filter!(
    out::AbstractArray{T}, tmp::AbstractArray{T}, f::AbstractArray, ρ::AbstractArray,
    ρ̄::AbstractArray{T}, plan::Filtering.AbstractFilterPlan,
) where {T<:AbstractFloat}
    @. tmp = ρ * f
    Filtering.filter_apply!(out, tmp, plan)
    @. out = _favre(out, ρ̄)
    return out
end
