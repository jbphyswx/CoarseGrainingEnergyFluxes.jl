# ---------------------------------------------------------------------------
# The transform grid of an FFT spectral filter on a uniform Cartesian grid
# ---------------------------------------------------------------------------

"""
    _fft_layout(grid, reach) -> (; dims, P, topology, Δ, kx, ky, reach)

The transform grid of a two-direction uniform Cartesian grid: `dims` the grid's shape and `P` the
transform's, a periodic axis at its own length and a bounded one zero-padded to
[`_bounded_length`](@ref) for `reach`; the grid's topology, whose `Bounded` directions are the padded ones;
the spacing per direction; the angular wavenumbers of the real-to-complex half spectrum, `kx` over `k₁ ≥ 0`
and `ky` in FFT order; and the distance the padding puts every image of a cell beyond the grid's cells,
`Inf` with no bounded direction.
"""
function _fft_layout(grid::FlowGeometries.Grids.StructuredGrid{T}, reach::T) where {T<:AbstractFloat}
    Nx, Ny = size(FlowGeometries.Grids.mask(grid))
    px = FlowGeometries.Grids.isperiodic(grid, 1)
    py = FlowGeometries.Grids.isperiodic(grid, 2)
    dx = px ? T(FlowGeometries.Grids.period(grid, 1)) / Nx : abs(T(FlowGeometries.Grids.spacing(grid, 1)))
    dy = py ? T(FlowGeometries.Grids.period(grid, 2)) / Ny : abs(T(FlowGeometries.Grids.spacing(grid, 2)))
    Px = px ? Nx : _bounded_length(Nx, (Nx - 1) * dx, dx, reach)
    Py = py ? Ny : _bounded_length(Ny, (Ny - 1) * dy, dy, reach)
    kx = [T(2π) * i / (Px * dx) for i in 0:(Px ÷ 2)]
    ky = [T(2π) * (j < cld(Py, 2) ? j : j - Py) / (Py * dy) for j in 0:(Py - 1)]
    held = min(px ? T(Inf) : (Px - Nx + 1) * dx, py ? T(Inf) : (Py - Ny + 1) * dy)
    return (; dims = (Nx, Ny), P = (Px, Py), topology = FlowGeometries.Grids.topology(grid), Δ = (dx, dy), kx, ky,
            reach = held)
end

"""
    _bounded_length(N, extent, Δ, reach) -> Int

The transform length of a bounded axis of `N` cells `Δ` apart over `extent`: the least even 2·3·5-smooth
length of at least `2N` at which every image of a cell lies at least `reach` beyond the cells.
"""
_bounded_length(N::Int, extent::T, Δ::T, reach::T) where {T<:AbstractFloat} =
    _even_smooth_length(max(2N, ceil(Int, (extent + reach) / Δ)))

# The least even 2·3·5-smooth length of at least `n`: a real transform of an odd length (`3²5³ = 1125`) runs well
# behind its even neighbours (`2⁷3² = 1152`).
_even_smooth_length(n::Integer) = 2 * nextprod((2, 3, 5), cld(n, 2))

# The reach a grid plan pads its bounded directions for, to serve `kernel` up to `max_scale` in `D`
# dimensions; `max_scale = 0` keeps the `2N` box. The sharp kernel on a bounded direction is exact on that
# box through the FFT ([`_lattice_brick_wall`](@ref)) and refused by a nonuniform FFT.
_kernel_reach(kernel::Kernels.AbstractFilterKernel, max_scale::Real, ::Type{T}, dim::Val) where {T<:AbstractFloat} =
    T(Kernels.kernel_radius(kernel, T(max_scale), dim))
_kernel_reach(::Kernels.SharpSpectralKernel, ::Real, ::Type{T}, ::Val) where {T<:AbstractFloat} = zero(T)

# Raises unless `kernel` at `scale` reaches no farther than `held`, the distance a grid plan's padding puts
# every image of a cell beyond the grid's cells.
function _check_reach(held::T, kernel::Kernels.AbstractFilterKernel, scale::T, dim::Val) where {T<:AbstractFloat}
    r = Kernels.kernel_radius(kernel, scale, dim)
    r <= held || _past_padding(nameof(typeof(kernel)), scale, r, held)
    return nothing
end
_check_reach(::T, ::Kernels.SharpSpectralKernel, ::T, ::Val) where {T<:AbstractFloat} = nothing

@noinline _past_padding(name::Symbol, scale, r, held) = throw(ArgumentError(
    "$name at scale $scale extends $r from each cell, but this grid plan's zero padding keeps the domain's " *
    "periodic copies only $held away, so the filter would mix in cells from the opposite edge. Build the " *
    "grid plan with `max_scale = $scale` or larger (`spectral_grid_plan(…; max_scale)`), or let `plan_filter` " *
    "or `plan_filter_sweep` build it."))

# The sharp kernel over a nonuniform-FFT box with a bounded direction has no exact form, and raises.
_check_unbounded_reach(::Kernels.AbstractFilterKernel, ::Tuple) = nothing
_check_unbounded_reach(::Kernels.SharpSpectralKernel, tp::Tuple) = _padded(tp) ? _sharp_on_bounded() : nothing

@noinline _sharp_on_bounded() = throw(ArgumentError(
    "SharpSpectralKernel's spectral form has unbounded reach, so on a bounded direction its result depends on " *
    "how far the domain is zero-padded; a nonuniform-FFT filter takes it over periodic directions only. A " *
    "uniform two-direction grid takes it exactly through the FFT path (`using FFTW`). `method = RealSpace()` " *
    "gives a different filter: this kernel's real-space form cut off near 10ℓ, whose transfer function " *
    "ripples about the brick wall near k_c."))

"""
    _fft_transfer(kernel, topology, layout, scale) -> Matrix

The transfer function on `layout`'s half spectrum, on the host: `Ĝ` at each wavenumber, and for
[`Kernels.SharpSpectralKernel`](@ref) with a bounded direction [`_lattice_brick_wall`](@ref).
"""
_fft_transfer(kernel::Kernels.AbstractFilterKernel, ::Tuple, layout, scale::T) where {T<:AbstractFloat} =
    _sampled_transfer(kernel, layout, scale)
_fft_transfer(
    kernel::Kernels.SharpSpectralKernel, ::Tuple{FlowGeometries.Grids.Periodic,FlowGeometries.Grids.Periodic},
    layout, scale::T,
) where {T<:AbstractFloat} = _sampled_transfer(kernel, layout, scale)
_fft_transfer(::Kernels.SharpSpectralKernel, tp::Tuple, layout, scale::T) where {T<:AbstractFloat} =
    _lattice_brick_wall(tp, layout, scale)

_sampled_transfer(kernel, layout, scale::T) where {T} =
    T[Kernels.spectral_transfer(kernel, (kx, ky), scale) for kx in layout.kx, ky in layout.ky]

"""
    _lattice_brick_wall(topology, layout, scale) -> Matrix

The sharp kernel's transfer on `layout`'s half spectrum with a bounded direction: the transform over the
padded box of the brick wall's lattice kernel at the grid's offsets within its extent. The field is zero
past a bounded edge, so the filter is the brick wall `|k| ≤ π/ℓ` of the infinite lattice exactly, for `ℓ`
at least the spacing of each bounded direction. Methods in the AbstractFFTs extension.

Both directions bounded, the lattice kernel is `h(o) = Δx Δy k_c²/(4π) · 2J₁(k_c r)/(k_c r)`, `r` the
offset's length; with one periodic, each of its wavenumbers `k` leaves the bounded direction the
one-dimensional brick wall `|k′| ≤ a = √(k_c² − k²)`, `h(o) = Δ sin(a o Δ)/(π o Δ)`.
"""
function _lattice_brick_wall end

"""
    _fft_ik(layout) -> (ikx, iky)

`i k_x` along the first axis of the half spectrum and `i k_y` along the second, host arrays shaped to
broadcast against it, zero at `±P/2` of an even axis, entry `P ÷ 2 + 1` of both the half and the full
axis: a real field's two halves of that mode carry opposite `i k`, so its derivative there vanishes.
"""
function _fft_ik(layout)
    ik(k::AbstractVector{T}, P::Int) where {T} =
        [complex(zero(T), (iseven(P) && i == P ÷ 2 + 1) ? zero(T) : k[i]) for i in eachindex(k)]
    return (reshape(ik(layout.kx, layout.P[1]), :, 1), reshape(ik(layout.ky, layout.P[2]), 1, :))
end

# `(∂N − ū ∂M)/M` from `∂N` in `out`, the quotient rule of a renormalized window; a plan with no window
# mass differentiates its series alone.
_quotient_rule!(out, _, ::Nothing, ::Nothing, ::Int) = out
function _quotient_rule!(out, ū, invmass, dmass::Tuple, d::Int)
    dm = dmass[d]
    @. out = (out - ū * dm) * invmass
    return out
end

"""
    distributed_fft_grid_plan(grid, reach; execution::DistributedBackend, batch) -> AbstractGridPlan
    distributed_fft_filter_plan(grid_plan, kernel, scale, mask_strategy) -> AbstractFilterPlan

An FFT spectral plan whose two-direction transform is divided among the worker processes: each worker
transforms a block of columns along the first direction and a block of rows along the second, over the
transform grid [`_fft_layout`](@ref) pads for `reach`. Methods in
the extension AbstractFFTs, Distributed and SharedArrays load together; the transforms run on the FFT
provider of the workers' arrays.
"""
function distributed_fft_grid_plan end
function distributed_fft_filter_plan end

distributed_fft_grid_plan(args...; kwargs...) = throw(ArgumentError(
    "DistributedBackend is unavailable — run `using Distributed, SharedArrays` (or use SerialBackend())."))
