# ---------------------------------------------------------------------------
# Subfilter-stress decomposition (Germano 1992): τ = Leonard + Cross + Reynolds
# ---------------------------------------------------------------------------

"""
    tau_decomposition(u, v, grid, kernel, scale; backend=AutoBackend(), mask_strategy=ZeroFill())
        -> (; L, C, R)

Split the 2D subfilter-scale stress `τ_ij = ⟨u_i u_j⟩ - ū_i ū_j` into Leonard, Cross, and Reynolds
contributions (Germano 1992, *JFM* 238, using generalized central moments so each piece is
individually Galilean-invariant). With `ū = G * u` the filtered velocity and `u' = u - ū` the
residual, and the generalized second moment `M(f, g) = (fg)‾ - f̄ ḡ`:

- Leonard  `L_ij = M(ū_i, ū_j)`            (resolved–resolved),
- Cross    `C_ij = M(ū_i, u'_j) + M(u'_i, ū_j)`,
- Reynolds `R_ij = M(u'_i, u'_j)`          (subfilter–subfilter; backscatter),

with `L + C + R = τ` exactly. Returns a named tuple of named tuples, each holding the symmetric
2D components `(; xx, xy, yy)` as arrays.
"""
function tau_decomposition(
    u::AbstractVecOrMat,
    v::AbstractVecOrMat,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    return tau_decomposition!(TauWorkspace(grid), u, v, grid, kernel, scale;
        backend = backend, mask_strategy = mask_strategy)
end

"""
    TauWorkspace(grid)

Scratch for [`tau_decomposition!`](@ref): the nine output components, the filtered fields and
residuals, and two product buffers. Allocated once and reused, so a repeated decomposition — over
timesteps, or over scales — costs no allocation after the first.
"""
struct TauWorkspace{T<:AbstractFloat, A<:AbstractArray{T}}
    ub::A; vb::A; up::A; vp::A
    ubb::A; vbb::A; upb::A; vpb::A
    prod::A; fprod::A; fprod2::A
    Lxx::A; Lxy::A; Lyy::A
    Cxx::A; Cxy::A; Cyy::A
    Rxx::A; Rxy::A; Ryy::A
end

function TauWorkspace(grid::FlowGeometries.Grids.AbstractGrid{G,T}) where {T<:AbstractFloat, G}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    z() = zeros(T, gsz)
    return TauWorkspace(z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z(),
                        z(), z(), z(), z(), z(), z(), z(), z(), z())
end

# Generalized second moment M(a,b) = (ab)‾ - ā b̄, written into `dst` through the shared product and
# filtered-product buffers. A plain function rather than a closure over the workspace: a closure that
# captured these would box them.
@inline function _second_moment!(dst, a, b, fa, fb, prod, fprod, plan)
    @. prod = a * b
    Filtering.filter_apply!(fprod, prod, plan)
    @. dst = fprod - fa * fb
    return dst
end

"""
    tau_decomposition!(ws::TauWorkspace, u, v, grid, kernel, scale; filter_plan=nothing, ...) -> (; L, C, R)

In-place [`tau_decomposition`](@ref). Writes into `ws` and returns views of its component buffers, so
the result is valid until the next call on the same workspace. Supplying `filter_plan` as well makes a
repeated decomposition allocation-free.
"""
function tau_decomposition!(
    ws::TauWorkspace{T},
    u::AbstractVecOrMat,
    v::AbstractVecOrMat,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    filter_plan::Union{Nothing,Filtering.AbstractFilterPlan} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    _require_tangent_pair(grid, "tau_decomposition!")
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend) :
        filter_plan

    Filtering.filter_apply_batch!((ws.ub, ws.vb), (u, v), plan)      # ū, v̄
    @. ws.up = u - ws.ub                                             # residuals u', v'
    @. ws.vp = v - ws.vb
    Filtering.filter_apply_batch!(                                   # ū̄, v̄̄, ū', v̄'
        (ws.ubb, ws.vbb, ws.upb, ws.vpb), (ws.ub, ws.vb, ws.up, ws.vp), plan,
    )

    _second_moment!(ws.Lxx, ws.ub, ws.ub, ws.ubb, ws.ubb, ws.prod, ws.fprod, plan)
    _second_moment!(ws.Lxy, ws.ub, ws.vb, ws.ubb, ws.vbb, ws.prod, ws.fprod, plan)
    _second_moment!(ws.Lyy, ws.vb, ws.vb, ws.vbb, ws.vbb, ws.prod, ws.fprod, plan)

    _second_moment!(ws.Cxx, ws.ub, ws.up, ws.ubb, ws.upb, ws.prod, ws.fprod, plan)
    @. ws.Cxx *= T(2)
    # The cross term is the sum of both orderings, so the second lands in `fprod2` before adding.
    _second_moment!(ws.Cxy, ws.ub, ws.vp, ws.ubb, ws.vpb, ws.prod, ws.fprod, plan)
    _second_moment!(ws.fprod2, ws.up, ws.vb, ws.upb, ws.vbb, ws.prod, ws.fprod, plan)
    @. ws.Cxy += ws.fprod2
    _second_moment!(ws.Cyy, ws.vb, ws.vp, ws.vbb, ws.vpb, ws.prod, ws.fprod, plan)
    @. ws.Cyy *= T(2)

    _second_moment!(ws.Rxx, ws.up, ws.up, ws.upb, ws.upb, ws.prod, ws.fprod, plan)
    _second_moment!(ws.Rxy, ws.up, ws.vp, ws.upb, ws.vpb, ws.prod, ws.fprod, plan)
    _second_moment!(ws.Ryy, ws.vp, ws.vp, ws.vpb, ws.vpb, ws.prod, ws.fprod, plan)

    return (
        L = (xx = ws.Lxx, xy = ws.Lxy, yy = ws.Lyy),
        C = (xx = ws.Cxx, xy = ws.Cxy, yy = ws.Cyy),
        R = (xx = ws.Rxx, xy = ws.Rxy, yy = ws.Ryy),
    )
end

"""
    tau_decomposition(u, v, grid::AbstractGrid{<:SphericalGeometry}, kernel, scale; ...) -> (; L, C, R)

Spherical counterpart of the Cartesian method above, on any grid resolving two tangent directions —
structured, curvilinear, a scattered node set or a sphere pixelization: like [`compute_Π!`](@ref)'s spherical branch,
the Leonard/Cross/Reynolds moments are formed in PLANETARY-CARTESIAN coordinates (so filtering
commutes with the moment/residual operations, Aluie 2019), then each of `L`, `C`, `R`'s resulting 3×3
symmetric tensor is rotated back to the local (east, north) frame at every grid point. `L+C+R = τ`
still holds exactly (the rotation is linear). Returns the same `(; L, C, R)` shape as the Cartesian
method — local `(; xx, xy, yy)` (≡ east-east/east-north/north-north) components.
"""
function tau_decomposition(
    u::AbstractVecOrMat,
    v::AbstractVecOrMat,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.SphericalGeometry{T}}
    return tau_decomposition!(
        Sym3TauWorkspace(grid), u, v, grid, kernel, scale;
        backend = backend, mask_strategy = mask_strategy,
    )
end

# The six independent components of a symmetric 3x3 tensor, in the order the buffers are stored.
const _SYM3 = ((1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3))

"""
    _sph_pair_moments!(Maa, Mab, Mbb, a, b, ā, b̄, pr, plan, grid, geo)

The three generalized second moments of a planetary-Cartesian velocity pair,

    M(a,a) ,   M(a,b) + M(b,a) ,   M(b,b) ,   M(x,y)_ij = (x_i y_j)‾ − x̄_i ȳ_j ,

each rotated from the planetary 3×3 into the local (east, north) frame. `ā`, `b̄` are the already
filtered triples; `pr` is six scratch buffers, so each moment costs one batched apply of six products.
The cross moment sums both orderings before filtering, which the linearity of the filter permits.

Two decompositions of the flux use this with different pairs: the Germano split feeds it the filtered
velocity and its residual, and the Helmholtz split feeds it the rotational and divergent parts.

A point's three local components are written over slots 1-3 of the same tensor, safe because all six
of that point's values are read first.
"""
function _sph_pair_moments!(Maa, Mab, Mbb, a, b, ā, b̄, pr, plan, grid, geo)
    _pair_moments!(Maa, Mab, Mbb, a, b, ā, b̄, pr, plan)
    _rotate_moments_to_local!((Maa, Mab, Mbb), grid, geo, Val(3))
    return nothing
end

"""
    _pair_moments!(Maa, Mab, Mbb, a, b, ā, b̄, pr, plan)

The three generalized second moments of a velocity pair, in whatever frame the pair is given:

    M(a,a) ,   M(a,b) + M(b,a) ,   M(b,b) ,   M(x,y)_ij = (x_i y_j)‾ − x̄_i ȳ_j .

`ā`, `b̄` are the already filtered triples and `pr` six scratch buffers, so each moment costs one
batched apply of six products. The cross moment sums both orderings before filtering, which the
linearity of the filter permits, so it costs one product per component.
"""
function _pair_moments!(Maa, Mab, Mbb, a, b, ā, b̄, pr, plan)
    for (dst, x, x̄) in ((Maa, a, ā), (Mbb, b, b̄))
        for (k, (i1, i2)) in enumerate(_SYM3)
            @. pr[k] = x[i1] * x[i2]
        end
        Filtering.filter_apply_batch!(dst, pr, plan)
        for (k, (i1, i2)) in enumerate(_SYM3)
            @. dst[k] = dst[k] - x̄[i1] * x̄[i2]
        end
    end
    for (k, (i1, i2)) in enumerate(_SYM3)
        @. pr[k] = a[i1] * b[i2] + b[i1] * a[i2]
    end
    Filtering.filter_apply_batch!(Mab, pr, plan)
    for (k, (i1, i2)) in enumerate(_SYM3)
        @. Mab[k] = Mab[k] - (ā[i1] * b̄[i2] + b̄[i1] * ā[i2])
    end
    return nothing
end

"""
    _rotate_moments_to_local!(tensors, grid, geo, ::Val{NC})

Rotate each symmetric planetary-Cartesian tensor into the local frame, in place. `NC` is how many
local components the caller keeps: three for a tangent split, six for a volume, both written over the
leading slots of the tensor they came from — safe, since all six planetary values at a point are read
before any local one is written.
"""
function _rotate_moments_to_local!(tensors, grid, geo, nc::Val)
    @inbounds for I in CartesianIndices(tensors[1][1])
        i = Tuple(I)
        FlowGeometries.Grids.isactive(grid, i...) || continue
        λ, φ = FlowGeometries.Grids.coords(grid, i...)
        for t in tensors
            l = _rotate_stress_to_local_enr(
                geo, t[1][I], t[2][I], t[3][I], t[4][I], t[5][I], t[6][I], λ, φ,
            )
            _store_local!(t, I, l, nc)
        end
    end
    return nothing
end


# The rotation keeps its component keys, so each slot names the component it takes. The tangent block
# is the one carrying no radial index, `λλ`, `λφ`, `φφ`.
Base.@propagate_inbounds function _store_local!(t, I, τ, ::Val{3})
    t[1][I] = τ.λλ; t[2][I] = τ.λφ; t[3][I] = τ.φφ
    return nothing
end

Base.@propagate_inbounds function _store_local!(t, I, τ, ::Val{6})
    t[1][I] = τ.λλ; t[2][I] = τ.λφ; t[3][I] = τ.λr
    t[4][I] = τ.φφ; t[5][I] = τ.φr; t[6][I] = τ.rr
    return nothing
end

"""
    Sym3TauWorkspace(grid)

Scratch for the three-component [`tau_decomposition!`](@ref): a velocity triple and its residual,
their filtered and double-filtered forms, six product buffers, and the six independent components of
each of `L`, `C` and `R`.

Two paths take this shape. A spherical tangent split carries the planetary-Cartesian velocity and
keeps the first three local components of each tensor; a true-3-D split carries `(u, v, w)` and keeps
all six.

Where a rotation back to the local frame happens it is done in place into the leading slots of each
tensor, since it reads a point's six planetary components and writes that point's local ones — so the
returned components alias those slots and no separate output buffers exist.
"""
struct Sym3TauWorkspace{T<:AbstractFloat, A<:AbstractArray{T}}
    up::NTuple{3,A}     # the velocity triple, overwritten in place by the residual u'
    ub::NTuple{3,A}     # ū
    ubb::NTuple{3,A}    # ū̄
    upb::NTuple{3,A}    # ū'
    prod::NTuple{6,A}   # the six symmetric products fed to one batched apply
    L::NTuple{6,A}
    C::NTuple{6,A}
    R::NTuple{6,A}
end

function Sym3TauWorkspace(grid::FlowGeometries.Grids.AbstractGrid{G,T}) where {T<:AbstractFloat, G}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    t(n) = ntuple(_ -> zeros(T, gsz), n)
    return Sym3TauWorkspace(t(3), t(3), t(3), t(3), t(6), t(6), t(6), t(6))
end

"""
    tau_decomposition!(ws::Sym3TauWorkspace, u, v, grid, kernel, scale; filter_plan=nothing, ...)

In-place spherical [`tau_decomposition`](@ref).

Like [`compute_Π!`](@ref)'s spherical branch, the Leonard/Cross/Reynolds moments are formed in
PLANETARY-CARTESIAN coordinates — where filtering commutes with the moment and residual operations
(Aluie 2019) — and each resulting symmetric 3x3 tensor is then rotated back to the local (east, north)
frame. `L + C + R = τ` still holds exactly, the rotation being linear.

Every filter here goes through a batched apply: three velocity components, then six second-filtered
fields, then six products per tensor. The cross term uses the identity

    C_ij = M(ū_i, u'_j) + M(u'_i, ū_j) = (ū_i u'_j + u'_i ū_j)‾ − (ū̄_i ū'_j + ū'_i ū̄_j)

so it costs one product per component rather than two — the filter being linear, the two orderings can
be summed before filtering instead of after.
"""
function tau_decomposition!(
    ws::Sym3TauWorkspace{T},
    u::AbstractVecOrMat,
    v::AbstractVecOrMat,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    filter_plan::Union{Nothing,Filtering.AbstractFilterPlan} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}}
    _require_tangent_pair(grid, "tau_decomposition!")
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend) :
        filter_plan
    geo = FlowGeometries.Grids.grid_geometry(grid)
    up, ub, ubb, upb, pr = ws.up, ws.ub, ws.ubb, ws.upb, ws.prod

    # Local (u, v) -> planetary Cartesian, into the buffer that will later hold the residual.
    @inbounds for I in CartesianIndices(up[1])
        i = Tuple(I)
        if FlowGeometries.Grids.isactive(grid, i...)
            λ, φ = FlowGeometries.Grids.coords(grid, i...)
            pc = FlowGeometries.Geometry.vector_to_cartesian(geo, u[I], v[I], λ, φ)
            up[1][I] = pc[1]; up[2][I] = pc[2]; up[3][I] = pc[3]
        else
            up[1][I] = zero(T); up[2][I] = zero(T); up[3][I] = zero(T)
        end
    end

    Filtering.filter_apply_batch!(ub, up, plan)                       # ū
    for c in 1:3
        @. up[c] = up[c] - ub[c]                                      # residual u', in place
    end
    Filtering.filter_apply_batch!((ubb..., upb...), (ub..., up...), plan)   # ū̄ and ū'

    _sph_pair_moments!(ws.L, ws.C, ws.R, ub, up, ubb, upb, pr, plan, grid, geo)

    return (;
        L = (xx = ws.L[1], xy = ws.L[2], yy = ws.L[3]),
        C = (xx = ws.C[1], xy = ws.C[2], yy = ws.C[3]),
        R = (xx = ws.R[1], xy = ws.R[2], yy = ws.R[3]),
    )
end

# The six components each tensor of a true-3-D split returns, in `_SYM3` order.
@inline _sym3_named(t) = (xx = t[1], xy = t[2], xz = t[3], yy = t[4], yz = t[5], zz = t[6])

"""
    tau_decomposition(u, v, w, grid::StructuredGrid{T,G,3}, kernel, scale; ...) -> (; L, C, R)

True three-dimensional Germano split: the same generalized central moments as the 2-D methods, over
all six independent components of the `3×3` subfilter stress. Each tensor comes back as
`(; xx, xy, xz, yy, yz, zz)`, and `L + C + R = τ` holds componentwise.

On a Cartesian metric the components are `(x, y, z)` as given. On a spherical volumetric shell the
moments are formed in planetary-Cartesian coordinates and rotated back, so the components are local
`(east, north, radial)` — the convention the true-3-D [`compute_Π!`](@ref) uses.
"""
function tau_decomposition(
    u::AbstractArray{<:Any,3},
    v::AbstractArray{<:Any,3},
    w::AbstractArray{<:Any,3},
    grid::FlowGeometries.Grids.StructuredGrid{T,G,3},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    for (nm, a) in (("u", u), ("v", v), ("w", w))
        size(a) == gsz || throw(DimensionMismatch("$nm has size $(size(a)), grid expects $gsz"))
    end
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend)
    return tau_decomposition!(
        Sym3TauWorkspace(grid), u, v, w, grid, kernel, scale; filter_plan = plan,
    )
end

"""
    tau_decomposition!(ws::Sym3TauWorkspace, u, v, w, grid::StructuredGrid{T,G,3}, kernel, scale; ...)

In-place true-3-D [`tau_decomposition`](@ref). Four batched applies carry the whole split: the three
velocity components, the six second-filtered fields, then six products per tensor.
"""
function tau_decomposition!(
    ws::Sym3TauWorkspace{T},
    u::AbstractArray{<:Any,3},
    v::AbstractArray{<:Any,3},
    w::AbstractArray{<:Any,3},
    grid::FlowGeometries.Grids.StructuredGrid{T,G,3},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    filter_plan::Union{Nothing,Filtering.AbstractFilterPlan} = nothing,
    backend::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, backend = backend) :
        filter_plan
    up, ub, ubb, upb, pr = ws.up, ws.ub, ws.ubb, ws.upb, ws.prod

    _fill_tau3_velocity!(up, u, v, w, grid, T)
    Filtering.filter_apply_batch!(ub, up, plan)                            # ū
    for c in 1:3
        @. up[c] = up[c] - ub[c]                                           # residual u', in place
    end
    Filtering.filter_apply_batch!((ubb..., upb...), (ub..., up...), plan)  # ū̄ and ū'
    _pair_moments!(ws.L, ws.C, ws.R, ub, up, ubb, upb, pr, plan)
    _rotate_tau3_to_local!((ws.L, ws.C, ws.R), grid)

    return (; L = _sym3_named(ws.L), C = _sym3_named(ws.C), R = _sym3_named(ws.R))
end

# A Cartesian volume's components are the ones supplied.
function _fill_tau3_velocity!(
    up, u, v, w, grid::FlowGeometries.Grids.StructuredGrid{T,G,3}, ::Type{T},
) where {T, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    @. up[1] = u
    @. up[2] = v
    @. up[3] = w
    return nothing
end

# A spherical shell's local (east, north, radial) triple is not a vector under filtering, so the
# moments are taken in planetary Cartesian and rotated back afterwards.
function _fill_tau3_velocity!(
    up, u, v, w, grid::FlowGeometries.Grids.StructuredGrid{T,G,3}, ::Type{T},
) where {T, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}}
    geo = FlowGeometries.Grids.grid_geometry(grid)
    @inbounds for I in CartesianIndices(up[1])
        i = Tuple(I)
        if FlowGeometries.Grids.isactive(grid, i...)
            λ, φ, _ = FlowGeometries.Grids.coords(grid, i...)
            pc = FlowGeometries.Geometry.vector_to_cartesian(geo, u[I], v[I], w[I], λ, φ)
            up[1][I] = pc[1]; up[2][I] = pc[2]; up[3][I] = pc[3]
        else
            up[1][I] = zero(T); up[2][I] = zero(T); up[3][I] = zero(T)
        end
    end
    return nothing
end

_rotate_tau3_to_local!(
    _tensors, ::FlowGeometries.Grids.StructuredGrid{T,G,3},
) where {T, G<:FlowGeometries.Geometry.CartesianGeometry{T}} = nothing

function _rotate_tau3_to_local!(
    tensors, grid::FlowGeometries.Grids.StructuredGrid{T,G,3},
) where {T, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}}
    geo = FlowGeometries.Grids.grid_geometry(grid)
    @inbounds for I in CartesianIndices(tensors[1][1])
        i = Tuple(I)
        FlowGeometries.Grids.isactive(grid, i...) || continue
        λ, φ, _ = FlowGeometries.Grids.coords(grid, i...)
        for t in tensors
            l = _rotate_stress_to_local_enr(
                geo, t[1][I], t[2][I], t[3][I], t[4][I], t[5][I], t[6][I], λ, φ,
            )
            _store_local!(t, I, l, Val(6))
        end
    end
    return nothing
end
