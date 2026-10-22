# ---------------------------------------------------------------------------
# The generalized convolution of vectors and tensors on a sphere (Aluie 2019)
# ---------------------------------------------------------------------------
#
# A tangent vector `(u, v)` (east, north) is the spin-1 field `U = u + iv`, and a symmetric tangent tensor
# `T` its trace `T_ee + T_nn`, spin 0, and `T(m, m) = T_ee − T_nn + 2iT_en`, spin 2, with `m = e_e + ie_n`.
# The generalized convolution multiplies every degree-`n` coefficient of each by `Ĝ(n)`; see the theory
# page for the derivation from Aluie (2019), Propositions 2 and 3. Fields carrying trailing batch axes are
# filtered slice by slice, the slice at linear offset `off`.

"""
    SpinFilter

A workspace's [`Filtering.GeneralizedConvolution`](@ref) on its spherical grid: the spin transforms, a
complex field in the workspace's memory, with a host copy where the transforms run on the host and that
memory is a device's, five coefficient arrays and one to scale in, the transfer `Ĝ(n)` of the scale last
set, and the quadrature that computes it.
"""
struct SpinFilter{
    TR<:Filtering.AbstractSpinTransforms, T<:AbstractFloat, MK, IX, FD, FH, CA,
    V<:AbstractVector{T}, Q<:Kernels.LegendreQuadrature{T}, B,
}
    transforms::TR
    mask::MK                # the grid's mask in the workspace's memory
    position::IX            # a cell's position in `field`
    ncells::Int
    radius::T
    field::FD               # in the workspace's memory
    stage::FH               # the transforms' host field array, or `nothing` where they read `field`
    coeffs::NTuple{5,CA}    # a sweep's analyses, in the order of `_spin_analyze_flux!`
    work::CA
    transfer::V             # Ĝ(n), in host memory
    quadrature::Q
    execution::B
end

# One line: the transforms hold plans of C libraries.
Base.show(io::IO, sf::SpinFilter) =
    print(io, "SpinFilter(", nameof(typeof(sf.transforms)), ", ", sf.ncells, " cells, degrees 0:",
          length(sf.transfer) - 1, ")")
Base.show(io::IO, ::MIME"text/plain", sf::SpinFilter) = show(io, sf)

"""
    _spin_filter(grid, at, vector_filter, transform, nufft) -> SpinFilter or nothing

The generalized convolution on `grid` for a workspace placed at `at`, its transforms chosen by `transform`
and `nufft`, or `nothing` for [`Filtering.CartesianComponents`](@ref).
"""
_spin_filter(_grid, ::Placement, ::Filtering.CartesianComponents, _transform, _nufft) = nothing

function _spin_filter(
    grid::FlowGeometries.Grids.AbstractGrid{G,T}, at::Placement, vf::Filtering.GeneralizedConvolution,
    transform::SpectralBackends.AbstractSpectralBackend, nufft::SpectralBackends.AbstractSpectralBackend,
) where {G, T<:AbstractFloat}
    tr = Filtering.spin_transforms(transform, grid; lmax = vf.lmax, execution = at.execution,
                                   nthreads = Filtering._library_threads(at.execution), nufft = nufft)
    staged = !Filtering._host_memory(at.execution) && Filtering.spin_host(tr)
    field = staged ? _zeros(at, Complex{T}, Filtering.spin_field_size(tr)) : Filtering.spin_field(tr)
    stage = staged ? Filtering.spin_field(tr) : nothing
    coeff() = Filtering.spin_coefficients(tr)
    R = T(FlowGeometries.Geometry.radius(FlowGeometries.Grids.grid_geometry(grid)))
    return SpinFilter(
        tr, FlowGeometries.Grids.mask(at.grid), Filtering.spin_position(tr), length(FlowGeometries.Grids.mask(grid)),
        R, field, stage, ntuple(_ -> coeff(), Val(5)), coeff(), Filtering.spin_transfer_buffer(tr),
        Kernels.LegendreQuadrature(T), at.execution,
    )
end

# A workspace's filter for `vector_filter` as `Filtering.resolve_vector_filter` reads it under `ZeroFill`.
@inline _workspace_spin(grid, at::Placement, vector_filter, transform, nufft) = _spin_filter(
    grid, at, Filtering.resolve_vector_filter(grid, Filtering.ZeroFill(), vector_filter), transform, nufft,
)

# The workspace's generalized convolution, for a plan, or a mask strategy, filtering under `ZeroFill`.
@inline _spin_of(::Nothing, _plan) = nothing
@inline _spin_of(sf::SpinFilter, plan::Filtering.AbstractFilterPlan) = _spin_of(sf, Filtering.plan_strategy(plan))
@inline function _spin_of(sf::SpinFilter, strategy::Filtering.AbstractMaskStrategy)
    strategy isa Filtering.ZeroFill || _spin_needs_zerofill()
    return sf
end

# A filter plan, or the mask strategy that stands for it where a workspace's generalized convolution filters
# every field a diagnostic reads.
const _PlanOrStrategy = Union{Filtering.AbstractFilterPlan, Filtering.AbstractMaskStrategy}

"""
    _plan_for(spin, filter_plan, strategy, build) -> plan or strategy

The plan a diagnostic on a workspace whose generalized convolution is `spin` reads: `filter_plan` when
given, otherwise `strategy` where `spin` filters every field, which reads no plan, and `build()` where
`spin` is `nothing`.
"""
@inline _plan_for(spin, filter_plan, strategy, build::F) where {F} =
    filter_plan !== nothing ? filter_plan : spin === nothing ? build() : strategy

# A mask strategy holds no memory.
@inline _check_placement(::Placement, ::Filtering.AbstractMaskStrategy) = nothing

@noinline _spin_needs_zerofill() = throw(ArgumentError(
    "this workspace holds the transforms of the generalized convolution, which filters the field " *
    "extended by zero under ZeroFill; the filter plan or mask_strategy is Deformable, which is no " *
    "convolution and filters a vector by its Cartesian components. Build the workspace with " *
    "`vector_filter = Filtering.CartesianComponents()`.",
))

# `Ĝ(n)` of `kernel` at `scale` into the filter's transfer.
@inline _spin_transfer!(sf::SpinFilter, kernel::Kernels.AbstractFilterKernel, scale) =
    Kernels.spectral_transfer_degrees!(sf.transfer, kernel, scale, sf.radius, sf.quadrature)

@inline _host_field!(sf::SpinFilter) = _host_field!(sf.stage, sf.field)
@inline _host_field!(::Nothing, field) = field
@inline _host_field!(stage::AbstractArray, field) = copyto!(stage, field)

# The spin-`s` coefficients, into `C`, of the field holding `value(k)` at cell `k`; with `masked` an
# inactive cell holds zero, whatever `value` reads there.
function _spin_analyze!(C, sf::SpinFilter, s::Int, masked::Bool, value::V) where {V}
    F, pos, m = sf.field, sf.position, sf.mask
    z = zero(eltype(F))
    FlowGeometries.Execution.run_indices(sf.ncells, sf.execution) do k
        @inbounds F[pos(k)] = (masked && !m[k]) ? z : value(k)
    end
    return Filtering.spin_analyze!(C, _host_field!(sf), sf.transforms, s)
end

# Into the filter's field, the spin-`s` field whose degree-`n` coefficients are `Ĝ(n)` times those of `C`.
function _spin_synthesize!(sf::SpinFilter, C, s::Int)
    host = sf.stage === nothing ? sf.field : sf.stage
    Filtering.spin_synthesize!(host, C, sf.work, sf.transfer, sf.transforms, s)
    sf.stage === nothing || copyto!(sf.field, sf.stage)
    return nothing
end

# `sink(k, z)` at every cell `k`, `z` the value the filter's field holds there.
function _spin_store!(sink::S, sf::SpinFilter) where {S}
    F, pos = sf.field, sf.position
    FlowGeometries.Execution.run_indices(sf.ncells, sf.execution) do k
        @inbounds sink(k, F[pos(k)])
    end
    return nothing
end

# The generalized convolution of the spin-`s` field `value` into `x` (real part) and `y` (imaginary
# part) at offset `off`: a tangent vector `(a, b)` is `s = 1` with `value(k) = a + ib`, and two scalars
# `a`, `b` are `s = 0`.
function _spin_filter!(x, y, sf::SpinFilter, s::Int, masked::Bool, value::V, off::Int) where {V}
    C = sf.coeffs[1]
    _spin_analyze!(C, sf, s, masked, value)
    _spin_synthesize!(sf, C, s)
    _spin_store!(sf) do k, z
        @inbounds x[off + k] = real(z)
        @inbounds y[off + k] = imag(z)
    end
    return nothing
end

# The generalized convolution of the tangent vector `(a, b)` into `(ā, b̄)`.
@inline _spin_vector!(ā, b̄, a, b, sf::SpinFilter, masked::Bool) =
    _spin_filter!(ā, b̄, sf, 1, masked, k -> @inbounds(complex(a[k], b[k])), 0)

# The convolution of the scalar holding `value(k)` at cell `k` into `x`: spin 0, by the transforms and the
# `Ĝ(n)` of the vectors it combines with.
function _spin_scalar!(x, sf::SpinFilter, masked::Bool, value::V) where {V}
    C = sf.coeffs[1]
    _spin_analyze!(C, sf, 0, masked, k -> complex(value(k)))
    _spin_synthesize!(sf, C, 0)
    _spin_store!(sf) do k, z
        @inbounds x[k] = real(z)
    end
    return nothing
end

# The generalized convolution of a symmetric tangent tensor into `(T̄_ee, T̄_en, T̄_nn)` at offset `off`:
# its traceless part `Q(k) = T_ee − T_nn + 2iT_en` by spin 2 and its trace `tr(k) = T_ee + T_nn` by
# spin 0.
function _spin_tensor!(Tee, Ten, Tnn, sf::SpinFilter, masked::Bool, Q::FQ, tr::FT, off::Int) where {FQ, FT}
    C2, C0 = sf.coeffs[1], sf.coeffs[2]
    _spin_analyze!(C2, sf, 2, masked, Q)
    _spin_analyze!(C0, sf, 0, masked, k -> complex(tr(k)))
    _spin_synthesize!(sf, C2, 2)
    _spin_store!(sf) do k, z
        @inbounds Tee[off + k] = real(z)
        @inbounds Ten[off + k] = imag(z) / 2
    end
    _spin_synthesize!(sf, C0, 0)
    _spin_store!(sf) do k, z
        @inbounds d = Tee[off + k]
        @inbounds Tee[off + k] = (real(z) + d) / 2
        @inbounds Tnn[off + k] = (real(z) - d) / 2
    end
    return nothing
end

# `c[(sym a⊗b)‾ − sym(ā⊗b̄)]` of the tangent vectors `a = (a₁, a₂)` and `b`, given their filters `ā` and
# `b̄`, into `M[1:3]`, the (ee, en, nn) components: `c = 1` with `a = b` is the moment `M(a, a)`, `c = 2` is
# `M(a, b) + M(b, a)`. For `a⊗b`, `T(m, m) = AB` with `A = a₁ + ia₂`, and the trace is `a·b`.
function _spin_moment!(M, sf::SpinFilter, a, b, ā, b̄, c::Real, masked::Bool)
    a1, a2, b1, b2 = a[1], a[2], b[1], b[2]
    _spin_tensor!(M[1], M[2], M[3], sf, masked,
                  k -> @inbounds(c * (complex(a1[k], a2[k]) * complex(b1[k], b2[k]))),
                  k -> @inbounds(c * (a1[k] * b1[k] + a2[k] * b2[k])), 0)
    ā1, ā2, b̄1, b̄2 = ā[1], ā[2], b̄[1], b̄[2]
    @. M[1] -= c * ā1 * b̄1
    @. M[2] -= c * (ā1 * b̄2 + ā2 * b̄1) / 2
    @. M[3] -= c * ā2 * b̄2
    return nothing
end

"""
    SpinAnalysis

Marker that a workspace's [`SpinFilter`](@ref) holds a flux sweep's spin coefficients, those
`_spin_analyze_flux!` writes, so a scale synthesizes them alone.
"""
struct SpinAnalysis end

# A flux's spin coefficients of the slice at offset `off` into the filter's five arrays: `U = u + iv`
# (spin 1), `U²` (spin 2) and `|U|²` (spin 0), and with a vertical velocity `w`, `|U|² + iw²` in its place,
# `wU` (spin 1) and `w` (spin 0).
function _spin_analyze_flux!(sf::SpinFilter, u, v, w, off::Int)
    C = sf.coeffs
    _spin_analyze!(C[1], sf, 1, true, k -> @inbounds(complex(u[off + k], v[off + k])))
    _spin_analyze!(C[2], sf, 2, true, k -> @inbounds(complex(u[off + k], v[off + k])^2))
    _spin_analyze_vertical!(sf, u, v, w, off)
    return nothing
end

_spin_analyze_vertical!(sf::SpinFilter, u, v, ::Nothing, off::Int) =
    _spin_analyze!(sf.coeffs[3], sf, 0, true, k -> @inbounds(complex(u[off + k]^2 + v[off + k]^2)))

function _spin_analyze_vertical!(sf::SpinFilter, u, v, w::AbstractArray, off::Int)
    C = sf.coeffs
    _spin_analyze!(C[3], sf, 0, true, k -> @inbounds(complex(u[off + k]^2 + v[off + k]^2, w[off + k]^2)))
    _spin_analyze!(C[4], sf, 1, true, k -> @inbounds(w[off + k] * complex(u[off + k], v[off + k])))
    _spin_analyze!(C[5], sf, 0, true, k -> @inbounds(complex(w[off + k])))
    return nothing
end
