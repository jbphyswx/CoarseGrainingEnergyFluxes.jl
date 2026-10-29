# ---------------------------------------------------------------------------
# Subfilter-stress decomposition (Germano 1992): τ = Leonard + Cross + Reynolds
# ---------------------------------------------------------------------------

"""
    tau_decomposition(u, v, grid, kernel, scale; execution=AutoBackend(), mask_strategy=ZeroFill())
        -> (; L, C, R)

Split the 2D subfilter-scale stress `τ_ij = ⟨u_i u_j⟩ - ū_i ū_j` into Leonard, Cross, and Reynolds
contributions (Germano 1992, *JFM* 238); each is a generalized central moment and is Galilean-invariant.
With `ū = G * u` the filtered velocity and `u' = u - ū` the residual, and the generalized second moment
`M(f, g) = (fg)‾ - f̄ ḡ`:

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
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy, execution)
    return tau_decomposition!(TauWorkspace(grid; execution = Filtering.plan_execution(plan)), u, v, grid,
                              kernel, scale; filter_plan = plan)
end

"""
    TauWorkspace(grid; execution = SerialBackend())

Scratch for [`tau_decomposition!`](@ref), in the memory of `execution` (see [`Placement`](@ref)): the
filtered velocity and the residual, the second filter of each, three product buffers and the nine
output components.
"""
struct TauWorkspace{T<:AbstractFloat, A<:AbstractArray{T}, P<:Placement}
    ub::A; vb::A; up::A; vp::A
    ubb::A; vbb::A; upb::A; vpb::A
    prod::A; fprod::A; fprod2::A
    Lxx::A; Lxy::A; Lyy::A
    Cxx::A; Cxy::A; Cyy::A
    Rxx::A; Rxy::A; Ryy::A
    at::P
end

function TauWorkspace(
    grid::FlowGeometries.Grids.AbstractGrid{G,T};
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
) where {T<:AbstractFloat, G}
    at = Placement(grid; execution = execution)
    gsz = FlowGeometries.Grids.size_tuple(grid)
    z() = _zeros(at, T, gsz)
    return TauWorkspace(z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z(),
                        z(), z(), z(), z(), z(), z(), z(), z(), z(), at)
end

# Generalized second moment M(a,b) = (ab)‾ - ā b̄ into `dst`, given `fa = ā` and `fb = b̄`; `prod` and
# `fprod` are scratch.
@inline function _second_moment!(dst, a, b, fa, fb, prod, fprod, plan)
    @. prod = a * b
    Filtering.filter_apply!(fprod, prod, plan)
    @. dst = fprod - fa * fb
    return dst
end

"""
    tau_decomposition!(ws::TauWorkspace, u, v, grid, kernel, scale; filter_plan=nothing, derived_plan=nothing, ...) -> (; L, C, R)

In-place [`tau_decomposition`](@ref). Writes into `ws` and returns views of its component buffers, so
the result is valid until the next call on the same workspace. `filter_plan` filters `u` and `v`;
`derived_plan` filters what that produces, which under `ZeroFill` is defined over land too (see
`_derived_plan`). Supplying both makes a repeated decomposition allocation-free.
"""
function tau_decomposition!(
    ws::TauWorkspace{T},
    u::AbstractVecOrMat,
    v::AbstractVecOrMat,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    filter_plan::Union{Nothing,Filtering.AbstractFilterPlan} = nothing,
    derived_plan::Union{Nothing,Filtering.AbstractFilterPlan} = nothing,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    _require_tangent_pair(grid, "tau_decomposition!")
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, execution = execution) :
        filter_plan
    _check_placement(ws.at, plan)
    dplan = derived_plan === nothing ? _derived_plan(plan, grid, output_grid(grid, plan), kernel, scale) :
                                       derived_plan
    mask = FlowGeometries.Grids.mask(ws.at.grid)

    Filtering.filter_apply_batch!((ws.ub, ws.vb), (u, v), plan)      # ū, v̄
    @. ws.up = mask * u - ws.ub                                      # residuals u', v'; land is at rest
    @. ws.vp = mask * v - ws.vb
    Filtering.filter_apply_batch!(                                   # ū̄, v̄̄, ū', v̄'
        (ws.ubb, ws.vbb, ws.upb, ws.vpb), (ws.ub, ws.vb, ws.up, ws.vp), dplan,
    )

    _second_moment!(ws.Lxx, ws.ub, ws.ub, ws.ubb, ws.ubb, ws.prod, ws.fprod, dplan)
    _second_moment!(ws.Lxy, ws.ub, ws.vb, ws.ubb, ws.vbb, ws.prod, ws.fprod, dplan)
    _second_moment!(ws.Lyy, ws.vb, ws.vb, ws.vbb, ws.vbb, ws.prod, ws.fprod, dplan)

    _second_moment!(ws.Cxx, ws.ub, ws.up, ws.ubb, ws.upb, ws.prod, ws.fprod, dplan)
    @. ws.Cxx *= T(2)
    # C_xy sums both orderings; the swap leaves a diagonal component unchanged, so it doubles.
    _second_moment!(ws.Cxy, ws.ub, ws.vp, ws.ubb, ws.vpb, ws.prod, ws.fprod, dplan)
    _second_moment!(ws.fprod2, ws.up, ws.vb, ws.upb, ws.vbb, ws.prod, ws.fprod, dplan)
    @. ws.Cxy += ws.fprod2
    _second_moment!(ws.Cyy, ws.vb, ws.vp, ws.vbb, ws.vpb, ws.prod, ws.fprod, dplan)
    @. ws.Cyy *= T(2)

    _second_moment!(ws.Rxx, ws.up, ws.up, ws.upb, ws.upb, ws.prod, ws.fprod, dplan)
    _second_moment!(ws.Rxy, ws.up, ws.vp, ws.upb, ws.vpb, ws.prod, ws.fprod, dplan)
    _second_moment!(ws.Ryy, ws.vp, ws.vp, ws.vpb, ws.vpb, ws.prod, ws.fprod, dplan)

    return (
        L = (xx = ws.Lxx, xy = ws.Lxy, yy = ws.Lyy),
        C = (xx = ws.Cxx, xy = ws.Cxy, yy = ws.Cyy),
        R = (xx = ws.Rxx, xy = ws.Rxy, yy = ws.Ryy),
    )
end

"""
    tau_decomposition(u, v, grid::AbstractGrid{<:SphericalGeometry}, kernel, scale; vector_filter=nothing, transform=AutoSpectralBackend(), ...) -> (; L, C, R)

The Germano split on a sphere, on any grid resolving two tangent directions — structured, curvilinear, a
scattered node set or a sphere pixelization. The velocity and each product tensor are filtered by
`vector_filter` ([`Filtering.resolve_vector_filter`](@ref)): by default the generalized convolution of
Aluie (2019), whose spin transforms `transform` chooses and which reads no scalar filter plan.
`Filtering.CartesianComponents()` forms the moments in ambient Cartesian components and rotates each
symmetric 3×3 tensor linearly to the local frame. Both filters are linear, so `L + C + R = τ` holds
exactly. Each tensor comes back as its local `(; xx, xy, yy)` components: east–east, east–north and
north–north.
"""
function tau_decomposition(
    u::AbstractVecOrMat,
    v::AbstractVecOrMat,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
    vector_filter::Union{Nothing, Filtering.AbstractVectorFilter} = nothing,
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
) where {T<:AbstractFloat, G<:_CurvedSurface{T}}
    vf = Filtering.resolve_vector_filter(grid, mask_strategy, vector_filter)
    if vf isa Filtering.GeneralizedConvolution
        ws = Sym3TauWorkspace(grid; execution = execution, vector_filter = vf, transform = transform)
        return tau_decomposition!(ws, u, v, grid, kernel, scale; mask_strategy = mask_strategy)
    end
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy, execution, transform)
    return tau_decomposition!(
        Sym3TauWorkspace(grid; execution = Filtering.plan_execution(plan), vector_filter = vf), u, v, grid,
        kernel, scale; filter_plan = plan,
    )
end

# The six independent components of a symmetric 3x3 tensor, in the order the buffers are stored.
const _SYM3 = ((1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3))

"""
    _sph_pair_moments!(Maa, Mab, Mbb, a, b, ā, b̄, pr, plan, frames, be)

The three generalized second moments of an ambient Cartesian velocity pair,

    M(a,a) ,   M(a,b) + M(b,a) ,   M(b,b) ,   M(x,y)_ij = (x_i y_j)‾ − x̄_i ȳ_j ,

each rotated from the ambient 3×3 into the local (east, north) frame, over slots 1–3 of its own tensor.
`ā`, `b̄` are the filtered triples and `pr` six scratch buffers; `frames` is the frame plan of the cells
the moments are read on, held where the backend `be` runs.
"""
function _sph_pair_moments!(Maa, Mab, Mbb, a, b, ā, b̄, pr, plan, frames, be)
    _pair_moments!(Maa, Mab, Mbb, a, b, ā, b̄, pr, plan)
    _rotate_moments_to_local!((Maa, Mab, Mbb), frames, Val(3), be)
    return nothing
end

"""
    _pair_moments!(Maa, Mab, Mbb, a, b, ā, b̄, pr, plan)

The three generalized second moments of a velocity pair, in the frame the pair is given in:

    M(a,a) ,   M(a,b) + M(b,a) ,   M(b,b) ,   M(x,y)_ij = (x_i y_j)‾ − x̄_i ȳ_j .

`ā`, `b̄` are the filtered triples and `pr` six scratch buffers. The filter is linear, so the cross
moment filters the one product `a_i b_j + b_i a_j` per component.
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
    _rotate_moments_to_local!(tensors, frames, ::Val{NC}, be)

Rotate each symmetric ambient Cartesian tensor, its six components in `_SYM3` order, into the local
frame over its own leading slots: `(λλ, λφ, φφ)` for `NC = 3`, a tangent split, and
`(λλ, λφ, λr, φφ, φr, rr)` for `NC = 6`, a volume. Zero on inactive cells. Each cell reads all six
ambient components before it writes; `frames` is a frame plan held where the backend `be` runs.
"""
function _rotate_moments_to_local!(tensors, frames, ::Val{3}, be)
    for t in tensors
        FlowGeometries.Operators.tensor_to_local!(t[1], t[3], nothing, t[2], nothing, nothing,
                                                  t[1], t[4], t[6], t[2], t[3], t[5], frames; execution = be)
    end
    return nothing
end

function _rotate_moments_to_local!(tensors, frames, ::Val{6}, be)
    for t in tensors
        FlowGeometries.Operators.tensor_to_local!(t[1], t[4], t[6], t[2], t[3], t[5],
                                                  t[1], t[4], t[6], t[2], t[3], t[5], frames; execution = be)
    end
    return nothing
end

"""
    Sym3TauWorkspace(grid; execution = SerialBackend(), vector_filter = nothing, transform = AutoSpectralBackend(),
                     nufft = AutoSpectralBackend())

Scratch for the three-component [`tau_decomposition!`](@ref), in the memory of `execution` (see
[`Placement`](@ref)): a velocity triple and its residual, their filtered and double-filtered forms, six
product buffers, and the six independent components of each of `L`, `C` and `R`.

A spherical tangent split returns the first three local components of each tensor and a true-3-D split
all six; the rotation to the local frame writes over each tensor's leading slots, and the returned
components alias them. On the surface of a sphere the workspace holds the transforms of the vector filter
`vector_filter` names, as [`ΠWorkspace`](@ref) does; under the generalized convolution, the default, the
velocity has its two tangent components, no product buffers are held, and each tensor its three local
components.
"""
struct Sym3TauWorkspace{
    T<:AbstractFloat, A<:AbstractArray{T}, V<:Tuple{Vararg{A}}, W<:Tuple{Vararg{A}}, M<:Tuple{Vararg{A}},
    SF, P<:Placement,
}
    up::V       # the velocity, overwritten in place by the residual u'
    ub::V       # ū
    ubb::V      # ū̄
    upb::V      # ū'
    prod::W     # the six symmetric products fed to one batched apply
    L::M
    C::M
    R::M
    spin::SF    # the generalized convolution, a `SpinFilter`, or nothing
    at::P
end

function Sym3TauWorkspace(
    grid::FlowGeometries.Grids.AbstractGrid{G,T};
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.SerialBackend(),
    vector_filter::Union{Nothing,Filtering.AbstractVectorFilter} = nothing,
    transform::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
    nufft::SpectralBackends.AbstractSpectralBackend = SpectralBackends.AutoSpectralBackend(),
) where {T<:AbstractFloat, G}
    at = Placement(grid; execution = execution)
    spin = _workspace_spin(grid, at, vector_filter, transform, nufft)
    gsz = FlowGeometries.Grids.size_tuple(grid)
    t(::Val{n}) where {n} = ntuple(_ -> _zeros(at, T, gsz), Val(n))
    spin === nothing && return Sym3TauWorkspace(
        t(Val(3)), t(Val(3)), t(Val(3)), t(Val(3)), t(Val(6)), t(Val(6)), t(Val(6)), t(Val(6)), spin, at,
    )
    return Sym3TauWorkspace(
        t(Val(2)), t(Val(2)), t(Val(2)), t(Val(2)), t(Val(0)), t(Val(3)), t(Val(3)), t(Val(3)), spin, at,
    )
end

"""
    tau_decomposition!(ws::Sym3TauWorkspace, u, v, grid, kernel, scale; filter_plan=nothing, derived_plan=nothing, ...)
        -> (; L, C, R)

In-place spherical [`tau_decomposition`](@ref), by the workspace's vector filter. Returns views of `ws`'s
buffers, valid until the next call on the same workspace. The generalized convolution reads no scalar
filter plan, and none is built for it when `filter_plan` is not given.
"""
function tau_decomposition!(
    ws::Sym3TauWorkspace{T},
    u::AbstractVecOrMat,
    v::AbstractVecOrMat,
    grid::FlowGeometries.Grids.AbstractGrid{G,T},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    filter_plan::Union{Nothing,Filtering.AbstractFilterPlan} = nothing,
    derived_plan::Union{Nothing,Filtering.AbstractFilterPlan} = nothing,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:_CurvedSurface{T}}
    _require_tangent_pair(grid, "tau_decomposition!")
    plan = _plan_for(ws.spin, filter_plan, mask_strategy,
                     () -> Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, execution = execution))
    _check_placement(ws.at, plan)
    sf = _spin_of(ws.spin, plan)
    sf === nothing || return _spin_tau!(ws, sf, u, v, kernel, scale)
    dplan = derived_plan === nothing ? _derived_plan(plan, grid, output_grid(grid, plan), kernel, scale) :
                                       derived_plan
    b = ws.at.execution
    up, ub, ubb, upb, pr = ws.up, ws.ub, ws.ubb, ws.upb, ws.prod

    _to_ambient!(up, u, v, nothing, ws.at.frames, b)
    Filtering.filter_apply_batch!(ub, up, plan)                       # ū
    for c in 1:3
        @. up[c] = up[c] - ub[c]                                      # residual u', in place
    end
    Filtering.filter_apply_batch!((ubb..., upb...), (ub..., up...), dplan)  # ū̄ and ū'

    _sph_pair_moments!(ws.L, ws.C, ws.R, ub, up, ubb, upb, pr, dplan, _output_frames(ws.at, plan), b)
    return _tangent_split(ws)
end

@inline _tangent_split(ws::Sym3TauWorkspace) = (;
    L = (xx = ws.L[1], xy = ws.L[2], yy = ws.L[3]),
    C = (xx = ws.C[1], xy = ws.C[2], yy = ws.C[3]),
    R = (xx = ws.R[1], xy = ws.R[2], yy = ws.R[3]),
)

# The Germano split under the generalized convolution. `ū` reads the active cells; the residual
# `u' = u − ū`, with land at rest, and `ū` are defined at every cell, where their filters read them.
function _spin_tau!(ws::Sym3TauWorkspace, sf::SpinFilter, u, v, kernel, scale)
    up, ub, ubb, upb = ws.up, ws.ub, ws.ubb, ws.upb
    mask = FlowGeometries.Grids.mask(ws.at.grid)
    _spin_transfer!(sf, kernel, scale)
    _spin_vector!(ub[1], ub[2], u, v, sf, true)                    # ū
    @. up[1] = mask * u - ub[1]                                     # residual u'
    @. up[2] = mask * v - ub[2]
    _spin_vector!(ubb[1], ubb[2], ub[1], ub[2], sf, false)         # ū̄
    _spin_vector!(upb[1], upb[2], up[1], up[2], sf, false)         # ū'
    _spin_moment!(ws.L, sf, ub, ub, ubb, ubb, 1, false)
    _spin_moment!(ws.C, sf, ub, up, ubb, upb, 2, false)
    _spin_moment!(ws.R, sf, up, up, upb, upb, 1, false)
    return _tangent_split(ws)
end

# The six components each tensor of a true-3-D split returns, in `_SYM3` order.
@inline _sym3_named(t) = (xx = t[1], xy = t[2], xz = t[3], yy = t[4], yz = t[5], zz = t[6])

"""
    tau_decomposition(u, v, w, grid::StructuredGrid{T,G,3}, kernel, scale; ...) -> (; L, C, R)

True three-dimensional Germano split: the same generalized central moments as the 2-D methods, over
all six independent components of the `3×3` subfilter stress. Each tensor comes back as
`(; xx, xy, xz, yy, yz, zz)`, and `L + C + R = τ` holds componentwise.

On a Cartesian metric the components are `(x, y, z)` as given. On a spherical volumetric shell the
moments are formed in ambient Cartesian components and rotated back, so the components are local
`(east, north, radial)`.
"""
function tau_decomposition(
    u::AbstractArray{<:Any,3},
    v::AbstractArray{<:Any,3},
    w::AbstractArray{<:Any,3},
    grid::FlowGeometries.Grids.StructuredGrid{T,G,3},
    kernel::Kernels.AbstractFilterKernel,
    scale::T;
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    gsz = FlowGeometries.Grids.size_tuple(grid)
    for (nm, a) in (("u", u), ("v", v), ("w", w))
        size(a) == gsz || throw(DimensionMismatch("$nm has size $(size(a)), grid expects $gsz"))
    end
    plan = Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, execution = execution)
    return tau_decomposition!(
        Sym3TauWorkspace(grid; execution = Filtering.plan_execution(plan)), u, v, w, grid, kernel, scale;
        filter_plan = plan,
    )
end

"""
    tau_decomposition!(ws::Sym3TauWorkspace, u, v, w, grid::StructuredGrid{T,G,3}, kernel, scale; ...)
        -> (; L, C, R)

In-place true-3-D [`tau_decomposition`](@ref). Returns views of `ws`'s buffers, valid until the next
call on the same workspace.
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
    derived_plan::Union{Nothing,Filtering.AbstractFilterPlan} = nothing,
    execution::ComputationalBackends.AbstractExecutionBackend = ComputationalBackends.AutoBackend(),
    mask_strategy::Filtering.AbstractMaskStrategy = Filtering.ZeroFill(),
) where {T<:AbstractFloat, G<:FlowGeometries.Geometry.AbstractGeometry{T}}
    plan = filter_plan === nothing ?
        Filtering.plan_filter(grid, kernel, scale; mask_strategy = mask_strategy, execution = execution) :
        filter_plan
    _check_placement(ws.at, plan)
    dplan = derived_plan === nothing ? _derived_plan(plan, grid, output_grid(grid, plan), kernel, scale) :
                                       derived_plan
    b = ws.at.execution
    up, ub, ubb, upb, pr = ws.up, ws.ub, ws.ubb, ws.upb, ws.prod

    _fill_tau3_velocity!(up, u, v, w, ws.at.grid, ws.at.frames, b)
    Filtering.filter_apply_batch!(ub, up, plan)                             # ū
    for c in 1:3
        @. up[c] = up[c] - ub[c]                                            # residual u', in place
    end
    Filtering.filter_apply_batch!((ubb..., upb...), (ub..., up...), dplan)  # ū̄ and ū'
    _pair_moments!(ws.L, ws.C, ws.R, ub, up, ubb, upb, pr, dplan)
    _rotate_tau3_to_local!((ws.L, ws.C, ws.R), _output(ws.at, plan), _output_frames(ws.at, plan), b)

    return (; L = _sym3_named(ws.L), C = _sym3_named(ws.C), R = _sym3_named(ws.R))
end

# The velocity triple a volume filters, `grid` and its frame plan `frames` held where the backend `be`
# runs: the supplied components on a Cartesian volume, zero on an inactive cell; the ambient Cartesian
# components on a spherical shell.
function _fill_tau3_velocity!(
    up, u, v, w, grid::FlowGeometries.Grids.StructuredGrid{T,G,3}, _frames, _be,
) where {T, G<:FlowGeometries.Geometry.CartesianGeometry{T}}
    mask = FlowGeometries.Grids.mask(grid)
    @. up[1] = mask * u
    @. up[2] = mask * v
    @. up[3] = mask * w
    return nothing
end

_fill_tau3_velocity!(
    up, u, v, w, ::FlowGeometries.Grids.StructuredGrid{T,G,3}, frames, be,
) where {T, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}} = _to_ambient!(up, u, v, w, frames, be)

# The volume's tensors in its local frame, `_SYM3`-ordered: a Cartesian volume's already are.
_rotate_tau3_to_local!(
    _tensors, ::FlowGeometries.Grids.StructuredGrid{T,G,3}, _frames, _be,
) where {T, G<:FlowGeometries.Geometry.CartesianGeometry{T}} = nothing

_rotate_tau3_to_local!(
    tensors, ::FlowGeometries.Grids.StructuredGrid{T,G,3}, frames, be,
) where {T, G<:FlowGeometries.Geometry.AbstractSphericalGeometry{T}} =
    _rotate_moments_to_local!(tensors, frames, Val(6), be)
