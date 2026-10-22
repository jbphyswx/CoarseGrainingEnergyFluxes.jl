# ---------------------------------------------------------------------------
# Favre (density-weighted) coarse-graining — Aluie 2013
# ---------------------------------------------------------------------------

"""
    FavreWorkspace(grid; execution = SerialBackend())

Scratch for [`compressible_flux!`](@ref), in the memory of `execution` (see [`Placement`](@ref)): the
filtered density and pressure, the Favre velocities, the unweighted velocities, the three Favre stress
components, the two unweighted mass-flux components, the four velocity gradients, the two pressure
gradients, scratch and product buffers, and the three output fields.
"""
struct FavreWorkspace{T<:AbstractFloat, A<:AbstractArray{T}, P<:Placement}
    ρ̄::A; P̄::A
    ũ::A; ṽ::A
    ū::A; v̄::A
    τxx::A; τxy::A; τyy::A
    mx::A; my::A          # τ̄(ρ, u_j): the unweighted subscale mass flux
    ux::A; uy::A; vx::A; vy::A
    Px::A; Py::A
    prod::A; fbuf::A
    # ρu, ρv, ρuu, ρuv, ρvv
    prods::NTuple{5,A}
    Π::A; Λ::A; PD::A
    at::P
end

function FavreWorkspace(
    grid::FlowGeometries.Grids.AbstractGrid{G,T};
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
) where {T<:AbstractFloat, G}
    at = Placement(grid; execution = execution)
    gsz = FlowGeometries.Grids.size_tuple(grid)
    z() = _zeros(at, T, gsz)
    return FavreWorkspace(
        z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z(),
        z(), z(), ntuple(_ -> z(), 5), z(), z(), z(), at,
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
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    _require_tangent_pair(grid, "compressible_flux!")
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, execution = execution) :
        filter_plan
    _check_placement(ws.at, plan)
    b = ws.at.execution
    og = _output(ws.at, plan)
    dplan = _placed(ws.at, _resolve_deriv_plan(deriv_plan, output_grid(grid, plan)))

    Filtering.filter_apply_batch!((ws.ρ̄, ws.P̄, ws.ū, ws.v̄), (ρ, P, u, v), plan)

    # The filtered products land in `mx`, `my`, `τxx`, `τxy`, `τyy`, each finished in place below.
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

    # ρ̄ ∂_j ũ_i = ∂_j (ρu_i)‾ − ũ_i ∂_j ρ̄, differencing only fields defined at every cell. `Px`/`Py`
    # hold ∇ρ̄ until ∇P̄ replaces it.
    _grad2!(ws.ux, ws.uy, ws.mx, og, dplan, b)
    _grad2!(ws.vx, ws.vy, ws.my, og, dplan, b)
    _grad2!(ws.Px, ws.Py, ws.ρ̄, og, dplan, b)
    @. ws.ux -= ws.ũ * ws.Px
    @. ws.uy -= ws.ũ * ws.Py
    @. ws.vx -= ws.ṽ * ws.Px
    @. ws.vy -= ws.ṽ * ws.Py

    # The unweighted subscale mass flux τ̄(ρ,u_i) = (ρu_i)‾ − ρ̄ū_i.
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
    _grad2!(ws.Px, ws.Py, ws.P̄, og, dplan, b)
    @. ws.Λ = ifelse(mask, _favre(ws.Px * ws.mx + ws.Py * ws.my, ws.ρ̄), zero(T))
    # P̄ ∇·ū; the spent `τxx` takes the two gradient components the divergence does not read.
    _grad2!(ws.prod, ws.τxx, ws.ū, og, dplan, b)
    _grad2!(ws.τxx, ws.fbuf, ws.v̄, og, dplan, b)
    @. ws.PD = ifelse(mask, ws.P̄ * (ws.prod + ws.fbuf), zero(T))

    return (; Π = ws.Π, Λ = ws.Λ, pressure_dilatation = ws.PD,
            ρ̄ = ws.ρ̄, P̄ = ws.P̄, ũ = ws.ũ, ṽ = ws.ṽ)
end

"""
    compressible_flux(u, v, ρ, P, grid, kernel, scale; execution=AutoBackend(), mask_strategy=ZeroFill())
        -> (; Π, Λ, pressure_dilatation, ρ̄, P̄, ũ, ṽ)

The variable-density (Favre) cross-scale energy budget of Aluie (2013). Returns the three terms of that
budget which act on the large-scale kinetic energy `ρ̄|ũ|²/2`, plus the filtered fields they are built
from.

# Favre filtering

`f̃ ≡ (ρf)‾/ρ̄` is the density-weighted filter, under which the filtered continuity equation closes
exactly, `∂_tρ̄ + ∂_i(ρ̄ũ_i) = 0`. It is linear but does not commute with derivatives, and the budget
reads both filters.

# The three terms

```
Π = −ρ̄ ∂_j ũ_i τ̃(u_i,u_j) ,   τ̃(u_i,u_j) = (ρu_iu_j)‾/ρ̄ − ũ_iũ_j        deformation work
Λ = (1/ρ̄) ∂_j P̄ · τ̄(ρ,u_j) ,  τ̄(ρ,u_j)   = (ρu_j)‾ − ρ̄ū_j              baropycnal work
                                                                        (τ̄ unweighted)
P̄ ∇·ū                                                                   pressure dilatation
```

All three are per unit volume, W m⁻³ with `ρ` in kg m⁻³ and `P` in Pa; at a constant `ρ₀`, `Π` is `ρ₀`
times the specific flux [`compute_Π!`](@ref) returns.

`Π` and `Λ` pair a large-scale field with small-scale fluctuations and transfer energy across scales.
`P̄∇·ū` involves the large scales alone: a conversion between kinetic and internal energy at the
resolved scale. The pressure term takes the unweighted divergence; the form `P̄∇·ũ` plus a transport
term absorbs `Λ`.

# Masked and bounded grids

Under `ZeroFill` a masked cell and the domain exterior hold no fluid, so `ρ̄` is the fluid mass under
the kernel: the Favre fields are density-weighted means over the fluid, zero where none lies under the
kernel. Constant `ρ` collapses the budget to `ρ·Π` wherever the fluid fills the kernel, which under
`Deformable` is every active cell. `ρ` is read, and must be positive, on active cells only.

# Asymptotics

For a smooth field, Lees & Aluie (2019) give `Λ ≈ (C₂ℓ²/ρ̄)·c_d·[∇P̄·S̄·∇ρ̄ + ½ ω̄·(∇ρ̄ × ∇P̄)]` with
`C₂` the kernel's second moment: a strain-generation part and a baroclinic part present in solenoidal
flow too. Both vanish with `C₂`, as for the kernels of [`Kernels.HighOrderKernel`](@ref).

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
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    for (nm, a) in (("u", u), ("v", v), ("ρ", ρ), ("P", P))
        size(a) == gsz || throw(DimensionMismatch("$nm has size $(size(a)), grid expects $gsz"))
    end
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, execution = execution)
    ws = FavreWorkspace(grid; execution = Filtering.plan_execution(plan))
    _require_positive_density(ρ, ws.at)
    return compressible_flux!(
        ws, u, v, ρ, P, grid, kernel, scale;
        filter_plan = plan, deriv_plan = Derivatives.derivative_plan(output_grid(grid, plan)),
    )
end

"""
    SphericalFavreWorkspace(grid; execution = SerialBackend(), vector_filter = nothing,
                            transform = AutoSpectralBackend(), nufft = AutoSpectralBackend())

Scratch for the spherical [`compressible_flux!`](@ref), in the memory of `execution` (see
[`Placement`](@ref)): the ambient Cartesian velocity, the mass flux and the Favre velocity in those
components, six product buffers, the six components of the Favre stress, the local velocity pairs, the
resolved strain, the two pressure gradients and the three output fields.

On the surface of a sphere the workspace holds the transforms of the vector filter `vector_filter` names,
as [`ΠWorkspace`](@ref) does. Under the generalized convolution, the default, it holds no ambient
triples, four product buffers and the Favre stress's three local components.
"""
struct SphericalFavreWorkspace{
    T<:AbstractFloat, A<:AbstractArray{T}, V<:Tuple{Vararg{A}}, W<:Tuple{Vararg{A}}, M<:Tuple{Vararg{A}},
    SF, P<:Placement,
}
    p::V                # ambient Cartesian velocity
    bp::V               # its unweighted filter, ū in ambient components
    m::V                # (ρu_i)‾, then the unweighted subscale mass flux τ̄(ρ,u_i)
    up::V               # Favre velocity in ambient components
    prod::W             # the products, then ρ̄ũ and the strains' scratch
    τ::M                # Favre stress, (ee, en, nn) in slots 1-3
    loc::NTuple{4,A}    # (ũ_e, ṽ_n, ū_e, v̄_n)
    mloc::NTuple{2,A}   # τ̄(ρ,u) in local (east, north)
    S::NTuple{3,A}      # (xx, xy, yy)
    ρ̄::A; P̄::A; Px::A; Py::A
    Π::A; Λ::A; PD::A
    spin::SF            # the generalized convolution, a `SpinFilter`, or nothing
    at::P
end

function SphericalFavreWorkspace(
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
    spin === nothing && return SphericalFavreWorkspace(
        t(Val(3)), t(Val(3)), t(Val(3)), t(Val(3)), t(Val(6)), t(Val(6)), t(Val(4)), t(Val(2)), t(Val(3)),
        z(), z(), z(), z(), z(), z(), z(), spin, at,
    )
    return SphericalFavreWorkspace(
        (), (), (), (), t(Val(4)), t(Val(3)), t(Val(4)), t(Val(2)), t(Val(3)),
        z(), z(), z(), z(), z(), z(), z(), spin, at,
    )
end

"""
    compressible_flux(u, v, ρ, P, grid::AbstractGrid{<:SphericalGeometry}, kernel, scale; vector_filter=nothing, transform=AutoSpectralBackend(), ...)
        -> (; Π, Λ, pressure_dilatation, ρ̄, P̄, ũ, ṽ)

Spherical form of the variable-density (Favre) budget, on any grid resolving two tangent directions,
with the three terms of the Cartesian method. The vectors `u` and `ρu` and the tensor `ρu⊗u`, from which
the Favre velocity `ũ`, the Favre stress `τ̃` and the unweighted subscale mass flux `τ̄(ρ,u_j)` are
built, are filtered by `vector_filter` ([`Filtering.resolve_vector_filter`](@ref)). By default that is
the generalized convolution of Aluie (2019), whose spin transforms `transform` chooses; `ρ̄` and `P̄` are
then their spin-0 transforms with the same `Ĝ(n)`, so the budget reads one operator, of global support,
and the filter plan is not built. `Filtering.CartesianComponents()` filters the ambient Cartesian
components and rotates them to the local frame, and the filter plan filters `ρ` and `P`. The resolved
Favre strain and the unweighted divergence carry the local frame's curvature,

    ∇·ū = ∂ū_e/∂x + ∂v̄_n/∂y − v̄_n tanφ/R ,

the divergence taken as the trace of the unweighted strain.
"""
function compressible_flux(
    u::AbstractVecOrMat,
    v::AbstractVecOrMat,
    ρ::AbstractVecOrMat,
    P::AbstractVecOrMat,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    vector_filter::Union{Nothing, Filtering.AbstractVectorFilter} = nothing,
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    for (nm, a) in (("u", u), ("v", v), ("ρ", ρ), ("P", P))
        size(a) == gsz || throw(DimensionMismatch("$nm has size $(size(a)), grid expects $gsz"))
    end
    dplan = Derivatives.derivative_plan(output_grid(grid, mask_strategy))
    vf = Filtering.resolve_vector_filter(grid, mask_strategy, vector_filter)
    if vf isa Filtering.GeneralizedConvolution
        ws = SphericalFavreWorkspace(grid; execution = execution, vector_filter = vf, transform = transform)
        _require_positive_density(ρ, ws.at)
        return compressible_flux!(ws, u, v, ρ, P, grid, kernel, scale; mask_strategy, deriv_plan = dplan)
    end
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy, execution, transform)
    ws = SphericalFavreWorkspace(grid; execution = Filtering.plan_execution(plan), vector_filter = vf)
    _require_positive_density(ρ, ws.at)
    return compressible_flux!(ws, u, v, ρ, P, grid, kernel, scale; filter_plan = plan, deriv_plan = dplan)
end

"""
    compressible_flux!(ws::SphericalFavreWorkspace, u, v, ρ, P, grid, kernel, scale; ...)

In-place spherical [`compressible_flux`](@ref), by the workspace's vector filter. Returns views of `ws`'s
buffers, valid until the next call on the same workspace. The generalized convolution reads no filter
plan, and none is built for it when `filter_plan` is not given.
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
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}}
    _require_tangent_pair(grid, "compressible_flux!")
    plan = _plan_for(ws.spin, filter_plan, mask_strategy,
                     () -> Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, execution = execution))
    _check_placement(ws.at, plan)
    b = ws.at.execution
    og = _output(ws.at, plan)
    dplan = _placed(ws.at, _resolve_deriv_plan(deriv_plan, output_grid(grid, plan)))
    pr, τ, loc = ws.prod, ws.τ, ws.loc
    sf = _spin_of(ws.spin, plan)
    if sf === nothing
        _cartesian_favre_parts!(ws, u, v, ρ, P, plan, og, b)
    else
        _spin_favre_parts!(ws, sf, u, v, ρ, P, kernel, scale)
    end

    mask = FlowGeometries.Grids.mask(og)
    # Π = −ρ̄ S(ũ):τ̃, with ρ̄ S(ũ) = S(ρ̄ũ) − sym(ũ ⊗ ∇ρ̄) differencing only fields defined at every
    # cell. The product buffers are free once the stress is filtered.
    @. pr[1] = ws.ρ̄ * loc[1]
    @. pr[2] = ws.ρ̄ * loc[2]
    g = (pr[3], pr[4])
    _strain_into!(ws.S[1], ws.S[2], ws.S[3], pr[1], pr[2], g, og, dplan, b)
    _grad2!(ws.Px, ws.Py, ws.ρ̄, og, dplan, b)
    @. ws.S[1] -= loc[1] * ws.Px
    @. ws.S[2] -= T(0.5) * (loc[1] * ws.Py + loc[2] * ws.Px)
    @. ws.S[3] -= loc[2] * ws.Py
    @. ws.Π = ifelse(mask, -_sfs_contraction(
        ws.S[1], ws.S[2], ws.S[3], τ[1], τ[2], τ[3]), zero(T))

    _grad2!(ws.Px, ws.Py, ws.P̄, og, dplan, b)
    @. ws.Λ = ifelse(mask, _favre(ws.Px * ws.mloc[1] + ws.Py * ws.mloc[2], ws.ρ̄), zero(T))

    # ∇·ū = tr S(ū), the frame's curvature term included.
    _strain_into!(ws.S[1], ws.S[2], ws.S[3], loc[3], loc[4], g, og, dplan, b)
    @. ws.PD = ifelse(mask, ws.P̄ * (ws.S[1] + ws.S[3]), zero(T))

    return (; Π = ws.Π, Λ = ws.Λ, pressure_dilatation = ws.PD,
            ρ̄ = ws.ρ̄, P̄ = ws.P̄, ũ = loc[1], ṽ = loc[2])
end

# `ρ̄`, `P̄`, the local Favre and unweighted velocity pairs, the unweighted mass flux and the local Favre
# stress, by the ambient Cartesian components.
function _cartesian_favre_parts!(ws::SphericalFavreWorkspace, u, v, ρ, P, plan, og, b)
    p, bp, m, up, pr, τ, loc = ws.p, ws.bp, ws.m, ws.up, ws.prod, ws.τ, ws.loc
    _to_ambient!(p, u, v, nothing, ws.at.grid, b)
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

    _rotate_moments_to_local!((τ,), og, Val(3), b)
    _ambient_to_local!((loc[1], loc[2]), up, og, b)
    _ambient_to_local!((loc[3], loc[4]), bp, og, b)
    _ambient_to_local!(ws.mloc, m, og, b)
    return nothing
end

# The same by the generalized convolution: `ρ` and `P` at spin 0, `u` and `ρu` at spin 1, `ρu⊗u` at spins
# 2 and 0, one `Ĝ(n)` for all.
function _spin_favre_parts!(ws::SphericalFavreWorkspace, sf::SpinFilter, u, v, ρ, P, kernel, scale)
    τ, loc, m = ws.τ, ws.loc, ws.mloc
    _spin_transfer!(sf, kernel, scale)
    _spin_scalar!(ws.ρ̄, sf, true, k -> @inbounds(ρ[k]))
    _spin_scalar!(ws.P̄, sf, true, k -> @inbounds(P[k]))
    _spin_vector!(loc[3], loc[4], u, v, sf, true)
    _spin_filter!(m[1], m[2], sf, 1, true, k -> @inbounds(ρ[k] * complex(u[k], v[k])), 0)
    _spin_tensor!(τ[1], τ[2], τ[3], sf, true, k -> @inbounds(ρ[k] * complex(u[k], v[k])^2),
                  k -> @inbounds(ρ[k] * (u[k]^2 + v[k]^2)), 0)
    # Favre velocity ũ = (ρu)‾/ρ̄, read before `m` becomes the unweighted mass flux τ̄(ρ,u).
    @. loc[1] = _favre(m[1], ws.ρ̄)
    @. loc[2] = _favre(m[2], ws.ρ̄)
    @. m[1] = m[1] - ws.ρ̄ * loc[3]
    @. m[2] = m[2] - ws.ρ̄ * loc[4]
    @. τ[1] = _favre(τ[1], ws.ρ̄) - loc[1] * loc[1]
    @. τ[2] = _favre(τ[2], ws.ρ̄) - loc[1] * loc[2]
    @. τ[3] = _favre(τ[3], ws.ρ̄) - loc[2] * loc[2]
    return nothing
end

"""
    Favre3DWorkspace(grid; execution = SerialBackend())

Scratch for the true-3-D [`compressible_flux!`](@ref), in the memory of `execution` (see
[`Placement`](@ref)): the velocity triple, its unweighted filter, the mass flux and the Favre velocity,
six product buffers, the six components of the Favre stress and of the resolved strain, the three
pressure gradients and the three output fields.
"""
struct Favre3DWorkspace{T<:AbstractFloat, A<:AbstractArray{T,3}, P<:Placement}
    p::NTuple{3,A}      # the velocity triple, in ambient Cartesian components on a shell
    bp::NTuple{3,A}     # its unweighted filter
    m::NTuple{3,A}      # (ρu_i)‾, then the unweighted subscale mass flux τ̄(ρ,u_i)
    up::NTuple{3,A}     # Favre velocity
    prod::NTuple{6,A}   # the products, then ρ̄ũ, ū and the strains' scratch
    τ::NTuple{6,A}
    S::NTuple{6,A}
    loc::NTuple{3,A}    # Favre velocity in the local frame
    mloc::NTuple{3,A}   # τ̄(ρ,u) in the local frame
    g::NTuple{3,A}      # ∇ρ̄, then ∇P̄
    ρ̄::A; P̄::A
    Π::A; Λ::A; PD::A
    at::P
end

function Favre3DWorkspace(
    grid::FlowGeometries.Grids.StructuredGrid{T,G,3};
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
) where {T<:AbstractFloat, G}
    at = Placement(grid; execution = execution)
    gsz = FlowGeometries.Grids.size_tuple(grid)
    z() = _zeros(at, T, gsz)
    t(n) = ntuple(_ -> z(), n)
    return Favre3DWorkspace(
        t(3), t(3), t(3), t(3), t(6), t(6), t(6), t(3), t(3), t(3), z(), z(), z(), z(), z(), at,
    )
end

"""
    compressible_flux(u, v, w, ρ, P, grid::StructuredGrid{T,G,3}, kernel, scale; ...)
        -> (; Π, Λ, pressure_dilatation, ρ̄, P̄, ũ, ṽ, w̃)

The variable-density (Favre) budget in a volume: the three terms of the 2-D methods over all six
components of the Favre stress and of the resolved strain, vertical derivatives included:

```
Π = −ρ̄ S̃_ij τ̃(u_i,u_j) ,   Λ = (1/ρ̄) ∂_j P̄ · τ̄(ρ,u_j) ,   P̄ ∇·ū
```

On a Cartesian volume the components are `(x, y, z)` as given. On a spherical shell every quantity
carrying a direction is filtered in ambient Cartesian components and rotated to the local
(east, north, up) frame, and the strain carries the shell's curvature terms. `∇·ū` is the trace of the
unweighted strain and carries them too.
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
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    for (nm, a) in (("u", u), ("v", v), ("w", w), ("ρ", ρ), ("P", P))
        size(a) == gsz || throw(DimensionMismatch("$nm has size $(size(a)), grid expects $gsz"))
    end
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, execution = execution)
    ws = Favre3DWorkspace(grid; execution = Filtering.plan_execution(plan))
    _require_positive_density(ρ, ws.at)
    return compressible_flux!(
        ws, u, v, w, ρ, P, grid, kernel, scale;
        filter_plan = plan, deriv_plan = Derivatives.derivative_plan(output_grid(grid, plan)),
    )
end

"""
    compressible_flux!(ws::Favre3DWorkspace, u, v, w, ρ, P, grid, kernel, scale; ...)

In-place true-3-D [`compressible_flux`](@ref). Returns views of `ws`'s buffers, valid until the next
call on the same workspace.
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
    deriv_plan::Union{Nothing,FlowGeometries.Operators.StencilPlans} = nothing,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, execution = execution) :
        filter_plan
    _check_placement(ws.at, plan)
    b = ws.at.execution
    og = _output(ws.at, plan)
    dplan = _placed(ws.at, deriv_plan === nothing ? Derivatives.derivative_plan(output_grid(grid, plan)) : deriv_plan)
    p, bp, m, up, pr, τ = ws.p, ws.bp, ws.m, ws.up, ws.prod, ws.τ

    _fill_tau3_velocity!(p, u, v, w, ws.at.grid, b)
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

    _rotate_tau3_to_local!((τ,), og, b)
    _localize_triple!(ws.loc, up, og, b)
    _localize_triple!(ws.mloc, m, og, b)

    # ρ̄ S(ũ) = S(ρ̄ũ) − sym(ũ ⊗ ∇ρ̄), differencing only fields defined at every cell. The product
    # buffers are free once the stress is filtered; `g` holds ∇ρ̄ until ∇P̄ replaces it.
    mask = FlowGeometries.Grids.mask(og)
    ρũ = (pr[1], pr[2], pr[3])
    ub = (pr[4], pr[5], pr[6])
    for c in 1:3
        @. ρũ[c] = ws.ρ̄ * ws.loc[c]
    end
    _strain_sym3!(ws.S, ρũ, og, dplan, ub, b)
    _grad3!(ws.g, ws.ρ̄, og, dplan, b)
    for (k, (i1, i2)) in enumerate(_SYM3)
        @. ws.S[k] -= T(0.5) * (ws.loc[i1] * ws.g[i2] + ws.loc[i2] * ws.g[i1])
    end
    @. ws.Π = ifelse(mask, -_sfs_contraction(
        ws.S[1], ws.S[2], ws.S[3], ws.S[4], ws.S[5], ws.S[6],
        τ[1], τ[2], τ[3], τ[4], τ[5], τ[6]), zero(T))

    _grad3!(ws.g, ws.P̄, og, dplan, b)
    @. ws.Λ = ifelse(mask,
        _favre(ws.g[1] * ws.mloc[1] + ws.g[2] * ws.mloc[2] + ws.g[3] * ws.mloc[3], ws.ρ̄), zero(T))

    # ∇·ū = tr S(ū), a shell's curvature terms included.
    _localize_triple!(ub, bp, og, b)
    _strain_sym3!(ws.S, ub, og, dplan, ρũ, b)
    @. ws.PD = ifelse(mask, ws.P̄ * (ws.S[1] + ws.S[4] + ws.S[6]), zero(T))

    return (; Π = ws.Π, Λ = ws.Λ, pressure_dilatation = ws.PD,
            ρ̄ = ws.ρ̄, P̄ = ws.P̄, ũ = ws.loc[1], ṽ = ws.loc[2], w̃ = ws.loc[3])
end

# A volume's filtered triple in its local frame, `grid` held where the backend `be` runs: a Cartesian
# volume's already is; a shell's ambient Cartesian triple is rotated to (east, north, up).
function _localize_triple!(
    dst, src, ::FlowGeometries.Grids.StructuredGrid{T,G,3}, _be,
) where {T, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    for c in 1:3
        @. dst[c] = src[c]
    end
    return nothing
end

_localize_triple!(
    dst, src, grid::FlowGeometries.Grids.StructuredGrid{T,G,3}, be,
) where {T, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}} = _ambient_to_local!(dst, src, grid, be)

# A density-weighted mean `x/ρ̄`, zero where no fluid lies under the kernel (`ρ̄ = 0` there under
# `ZeroFill`, and at a masked target under `Deformable`).
@inline _favre(x, ρ̄) = ifelse(ρ̄ > zero(ρ̄), x / ρ̄, zero(x))

# ρ is read on the active cells of the placement's grid only; a masked cell's value is never filtered.
function _require_positive_density(ρ::AbstractArray{R}, at::Placement) where {R}
    msk = FlowGeometries.Grids.mask(at.grid)
    length(ρ) == length(msk) || throw(DimensionMismatch(
        "ρ has $(length(ρ)) values for a grid of $(length(msk)) cells",
    ))
    _check_min_density(_min_active(ρ, msk, float(R), at.execution))
    return nothing
end

# The least value of `ρ` over the active cells of `msk`, as an `F`, reduced under `b`.
function _min_active(ρ::AbstractArray, msk, ::Type{F}, b) where {F}
    return FlowGeometries.Execution.reduce_indices(min, typemax(F), length(ρ), b) do i
        @inbounds _cell_on(msk, i) ? F(ρ[i]) : typemax(F)
    end
end

_check_min_density(lo) = lo > 0 ? nothing : throw(ArgumentError(
    "Favre filtering divides by the filtered density, so ρ must be strictly positive on every " *
    "active cell; got a minimum of $lo.",
))

"""
    favre_filter!(out, tmp, f, ρ, ρ̄, plan) -> out

Write the Favre filter `f̃ = (ρf)‾/ρ̄` into `out`, given the filtered density `ρ̄` and a scratch array
`tmp`; zero where `ρ̄ = 0`.
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
