# ---------------------------------------------------------------------------
# Vectors and tensors on the surface of a sphere
# ---------------------------------------------------------------------------

"""
    AbstractVectorFilter

How a filter acts on a vector or a rank-2 tensor on the surface of a sphere:
[`GeneralizedConvolution`](@ref) or [`CartesianComponents`](@ref). A diagnostic takes it as its
`vector_filter` keyword; [`resolve_vector_filter`](@ref) states which applies where.
"""
abstract type AbstractVectorFilter end

"""
    AbstractAnalysis

How [`GeneralizedConvolution`](@ref) takes a field's spin-weighted coefficients: [`AutoAnalysis`](@ref),
[`Quadrature`](@ref) or [`LeastSquares`](@ref).
"""
abstract type AbstractAnalysis end

"""
    Quadrature() <: AbstractAnalysis

The coefficients by the quadrature over the grid's cells, `C_ℓm = Σ_k w_k F_k conj(ₛY_ℓm(x_k))`, `w_k` the cell's
measure over `R²`. It is exact for a field of degree at most `lmax` where the measure integrates degree `2lmax`
exactly, as Gauss–Legendre rows do to `lmax = nrings − 1`, and over a regional grid it is the integral of the
field extended by zero over the sphere.
"""
struct Quadrature <: AbstractAnalysis end

"""
    LeastSquares(; rtol = nothing, maxiter = 100) <: AbstractAnalysis

The coefficients minimizing `Σ_k w_k |F_k − Σ_ℓm C_ℓm ₛY_ℓm(x_k)|²` over the grid's cells, by FlowTransformBindings'
`lsmr!` in the inner product of the cells' measures, whose first iterate is the [`Quadrature`](@ref). It
recovers a field of degree at most `lmax` wherever the cells determine its coefficients, whatever the
quadrature's accuracy. `rtol` bounds `‖Aᴴ W r‖ / ‖Aᴴ W F‖`, `nothing` taking `max(1e-12, 100 eps(T))`; a fit
that stops at `maxiter` keeps its last iterate and reports through [`analysis_status`](@ref).
"""
struct LeastSquares{R<:Union{Nothing,Real}} <: AbstractAnalysis
    rtol::R
    maxiter::Int
end

LeastSquares(; rtol::Union{Nothing,Real} = nothing, maxiter::Integer = 100) = LeastSquares(rtol, Int(maxiter))

"""
    AutoAnalysis() <: AbstractAnalysis

[`Quadrature`](@ref) on Gauss–Legendre and reduced Gaussian samplings, where it is exact, and on grids whose
sampling may be regional (none, latitude–longitude or scattered), where it is the zero-extended field's
integral; [`LeastSquares`](@ref) on every other spherical sampling, whose quadrature is not exact at its band
limit (Fejér, Driscoll–Healy, McEwen–Wiaux, HEALPix, cubed-sphere, icosahedral, Yin–Yang). A latitude–longitude
grid over the whole sphere takes `LeastSquares()` by name.
"""
struct AutoAnalysis <: AbstractAnalysis end

"""
    GeneralizedConvolution(; lmax = nothing, analysis = AutoAnalysis()) <: AbstractVectorFilter

The convolution of vectors and rank-2 tensors on a sphere of Aluie (2019, eqs. 58 and 121): every
degree-`n` vector or tensor spherical harmonic is multiplied by the kernel's `Ĝ(n)`
([`Kernels.spectral_transfer_degrees!`](@ref)), so filtering commutes with the surface gradient,
divergence and curl (Propositions 1–3). It runs as spin-weighted transforms of the field extended by zero
over the sphere ([`spin_transforms`](@ref)): FastSphericalHarmonics' on a `Float64`
`Fejer1Sampling` grid of `2N − 1` by `N`, NUFSHT's on every other spherical grid when a NUFFT
library is loaded to run it, and otherwise the direct sums, [`RingSpinTransforms`](@ref) on a latitude–longitude `StructuredGrid` and
[`DirectSpinTransforms`](@ref) elsewhere. The theory page derives the spin-weighted form.

`lmax` is the highest degree the NUFSHT and direct transforms carry, `nothing` the grid's
[`spin_band_limit`](@ref). FastSphericalHarmonics transforms at its grid's own band limit, so a given
`lmax` selects another transform under `transform = AutoSpectralBackend()`. `analysis` is how the NUFSHT and
direct transforms take coefficients ([`AbstractAnalysis`](@ref)); FastSphericalHarmonics' transform is exact on
its grid.
"""
struct GeneralizedConvolution{L<:Union{Nothing,Int}, A<:AbstractAnalysis} <: AbstractVectorFilter
    lmax::L
    analysis::A
end

GeneralizedConvolution(; lmax::Union{Nothing,Integer} = nothing, analysis::AbstractAnalysis = AutoAnalysis()) =
    GeneralizedConvolution(lmax === nothing ? nothing : Int(lmax), analysis)

"""
    CartesianComponents <: AbstractVectorFilter

The scalar filter of each ambient Cartesian component of a vector, and of each Cartesian component of a
tensor, rotated back to the local frame (Aluie 2019, eq. 41). On the surface of a sphere it equals
[`GeneralizedConvolution`](@ref) for a tangent field with no divergence, whose Cartesian components of
degree `n` are spherical harmonics of degree `n`; the components of a divergent field's degree-`n` part
carry degrees `n ± 1`, which it filters with `Ĝ(n ± 1)`, and it gives a tangent field a radial
component, which a tangent diagnostic drops. It is how a vector is filtered under `Deformable`, which is
no convolution, and on a spheroid.
"""
struct CartesianComponents <: AbstractVectorFilter end

# Whether the generalized convolution filters `grid`'s vectors: a grid on the surface of a sphere.
@inline _sphere_surface(grid::FlowGeometries.Grids.AbstractGrid) =
    FlowGeometries.Grids.grid_geometry(grid) isa FlowGeometries.Geometry.AbstractSphericalGeometry &&
    FlowGeometries.Grids.ncoordinates(grid) == 2

"""
    resolve_vector_filter(grid, mask_strategy, vector_filter) -> AbstractVectorFilter

The filter a diagnostic applies to a vector on `grid` under `mask_strategy`. On the surface of a sphere,
`nothing` takes [`GeneralizedConvolution`](@ref) under `ZeroFill` and [`CartesianComponents`](@ref)
under `Deformable`, and a given filter is taken as given. Everywhere else it is
`CartesianComponents()`: on a plane and in a volume, whose kernel is a ball of the ambient space, the
convolution of each Cartesian component is the generalized convolution there; on a spheroid it is the
only one defined. A `GeneralizedConvolution` given under `Deformable` or on a spheroid raises an
`ArgumentError`.
"""
function resolve_vector_filter(
    grid::FlowGeometries.Grids.AbstractGrid, strategy::AbstractMaskStrategy,
    vector_filter::Union{Nothing,AbstractVectorFilter},
)
    if _sphere_surface(grid)
        vector_filter === nothing && return strategy isa ZeroFill ? GeneralizedConvolution() : CartesianComponents()
        vector_filter isa GeneralizedConvolution && !(strategy isa ZeroFill) && _no_deformable_convolution()
        return vector_filter
    end
    vector_filter isa GeneralizedConvolution &&
        FlowGeometries.Grids.grid_geometry(grid) isa FlowGeometries.Geometry.AbstractEllipsoidalGeometry &&
        _no_spheroid_convolution(grid)
    return CartesianComponents()
end

@noinline _no_deformable_convolution() = throw(ArgumentError(
    "vector_filter = GeneralizedConvolution() was given with mask_strategy = Deformable(). Deformable " *
    "renormalizes each window over its active cells, which is no convolution, so the generalized " *
    "convolution has no counterpart there; under Deformable a vector is filtered by its Cartesian " *
    "components. Pass vector_filter = CartesianComponents(), or mask_strategy = ZeroFill().",
))

@noinline _no_spheroid_convolution(grid) = throw(ArgumentError(
    "vector_filter = GeneralizedConvolution() was given on $(nameof(typeof(grid))) over a spheroid. The " *
    "generalized convolution is defined on a sphere, whose rotations carry a zonal kernel to every point; " *
    "on a spheroid pass vector_filter = CartesianComponents().",
))

# ---------------------------------------------------------------------------
# Spin-weighted transforms over a grid's cells
# ---------------------------------------------------------------------------

"""
    AbstractSpinTransforms

Spin-weighted spherical harmonic transforms of spin 0, 1 and 2 over every cell of a spherical grid, built by
[`spin_transforms`](@ref), in the grid's element type. A subtype defines

- `spin_field(tr)`: a zeroed complex field array of the kind they read and write;
- `spin_field_size(tr)`: its size;
- `spin_position(tr)`: a callable taking a cell's linear index to its position in that array;
- `spin_coefficients(tr)`: a zeroed coefficient array;
- `spin_transfer_buffer(tr)`: a zeroed host vector of one entry per degree a coefficient array carries,
  from 0, for the transfer `Ĝ`;
- `spin_analyze!(C, F, tr, s)`: the coefficients of the spin-`s` field `F` into `C`;
- `spin_synthesize!(F, C, W, Ĝ, tr, s)`: into `F`, the spin-`s` field whose degree-`n` coefficients are
  `Ĝ[n+1]` times those of `C`, with `W` a coefficient array to scale in. `C` is left as it is;
- `spin_ladder_synthesize!(F, C, W, d, tr, s, t)`: into `F`, the spin-`t` field `ð` (`t = s + 1`) or `ð̄`
  (`t = s − 1`) of the spin-`s` field whose coefficients are `C`, its degree-`n` coefficients scaled by
  `d[n+1]`.

and may define `spin_host(tr)`, whether its arrays are host memory, `true` unless it says otherwise; a
device workspace stages its fields through host copies for transforms that run there. Transforms whose
analysis is the quadrature `Aᴴ W` define [`spin_weights`](@ref), which a [`LeastSquares`](@ref) analysis reads.
"""
abstract type AbstractSpinTransforms end

function spin_field end
function spin_field_size end
function spin_position end
function spin_coefficients end
function spin_transfer_buffer end
function spin_analyze! end
function spin_synthesize! end

"""
    spin_ladder_synthesize!(F, C, W, d, tr, s, t) -> F

Into `F`, the spin-`t` field `ð` (`t = s + 1`) or `ð̄` (`t = s − 1`) of the spin-`s` field whose coefficients are
`C`, `ð` and `ð̄` the Newman–Penrose operators on the unit sphere, with its degree-`n` coefficients scaled by
`d[n + 1]`. The caller folds the ladder factor `√((n − s)(n + s + 1))` or `−√((n + s)(n − s + 1))` into `d`; each
transform applies the phases of its own harmonics. `W` is a coefficient array to scale in.
"""
function spin_ladder_synthesize! end

spin_host(::AbstractSpinTransforms) = true

"""
    spin_transforms(transform, grid; lmax = nothing, nthreads = 1, nufft = AutoSpectralBackend(),
                    execution = SerialBackend()) -> AbstractSpinTransforms

The spin transforms of [`GeneralizedConvolution`](@ref) over `grid`'s cells, for a workspace whose loops
run under the local backend `execution`, a library's on `nthreads` threads. `transform =
AutoSpectralBackend()` takes FastSphericalHarmonics on its own grids when `lmax` is `nothing`, NUFSHT on
every spherical grid where it is loaded with a NUFFT library (`nufft`, or under `Auto` NonuniformFFTs or
FINUFFT), and the direct sum otherwise. `DirectSumSpectralBackend()` takes the direct sum: [`RingSpinTransforms`](@ref) on a
latitude–longitude `StructuredGrid`, [`DirectSpinTransforms`](@ref) on any other grid. A library named that no
loaded extension provides raises an `ArgumentError`.
"""
spin_transforms(transform::SpectralBackends.AbstractSpectralBackend, grid::FlowGeometries.Grids.AbstractGrid;
                kwargs...) = _scattered_spin_transforms(transform, grid; kwargs...)

# The transforms over any spherical grid's cells: NUFSHT's extension adds its own for `Auto` and NUFSHT.
_scattered_spin_transforms(::SpectralBackends.AbstractAutoSpectralBackend, grid::FlowGeometries.Grids.AbstractGrid;
                           kwargs...) = _direct_spin_transforms(grid; kwargs...)
_scattered_spin_transforms(::SpectralBackends.AbstractDirectSumSpectralBackend, grid::FlowGeometries.Grids.AbstractGrid;
                           kwargs...) = _direct_spin_transforms(grid; kwargs...)
_scattered_spin_transforms(transform, grid; kwargs...) = _spin_library_missing(transform, grid)

_direct_spin_transforms(grid; lmax = nothing, execution = ComputationalBackends.SerialBackend(),
                        nthreads = 1, nufft = nothing) = DirectSpinTransforms(grid; lmax, execution)

@noinline _spin_library_missing(transform, grid) = throw(ArgumentError(
    "transform = $(nameof(typeof(transform)))() names a library whose spin-weighted transforms no loaded " *
    "extension provides for $(nameof(typeof(grid))): load it (`using NUFSHT`, or `using " *
    "FastSphericalHarmonics, FFTW` on a Float64 Fejer1Sampling grid of 2N − 1 longitudes by N " *
    "latitudes), or pass transform = AutoSpectralBackend() or DirectSumSpectralBackend().",
))

# A cell's centre, `cell` as `FlowGeometries.Grids.cell_at` spells it.
@inline _cell_point(grid, cell::Tuple) = FlowGeometries.Grids.coords(grid, cell...)
@inline _cell_point(grid, k::Integer) = FlowGeometries.Grids.coords(grid, k)

"""
    _spin_nodes(grid) -> (θ, φ, w)

Host vectors, in the grid's element type and its linear cell order, of every cell's colatitude, its
longitude in `[0, 2π)` and its measure over `R²`, the quadrature weight of a transform on the unit sphere.
"""
function _spin_nodes(grid::FlowGeometries.Grids.AbstractGrid{G,T}) where {G, T<:AbstractFloat}
    n = length(FlowGeometries.Grids.mask(grid))
    R = T(FlowGeometries.Geometry.radius(FlowGeometries.Grids.grid_geometry(grid)))
    m = FlowGeometries.Grids.measure(grid)
    θ, φ, w = zeros(T, n), zeros(T, n), zeros(T, n)
    for (k, c) in enumerate(FlowGeometries.Grids.cells(grid))
        p = _cell_point(grid, FlowGeometries.Grids.cell_at(grid, c))
        θ[k] = T(π) / 2 - T(p.φ)
        φ[k] = FlowGeometries.Geometry.wrap_longitude(T(p.λ))
        w[k] = T(m[k]) / R^2
    end
    return θ, φ, w
end

"""
    CellOrder()

The position of cell `k` in a field array laid out in the grid's own linear cell order: `k`.
"""
struct CellOrder end
@inline (::CellOrder)(k::Int) = k

"""
    TransposedCells(nlon, nlat)

The position of cell `k` of an `nlon × nlat` longitude–latitude grid in an `nlat × nlon` array: cell
`(i, j)`, `k = i + (j − 1)nlon`, sits at `j + (i − 1)nlat`.
"""
struct TransposedCells
    nlon::Int
    nlat::Int
end
@inline (c::TransposedCells)(k::Int) = cld(k, c.nlon) + (mod1(k, c.nlon) - 1) * c.nlat

"""
    spin_band_limit(grid) -> Int

The highest spherical-harmonic degree `grid`'s cells resolve, at least 2: the band limit of a sampling
that ties its ring count to one (`SphericalSampling.has_bandlimit`), and otherwise `⌈πR/√A⌉ − 1`, the
degree whose half wavelength is the side `√A` of the grid's largest cell.
"""
spin_band_limit(grid::FlowGeometries.Grids.AbstractGrid) =
    max(2, _band_limit(FlowGeometries.Grids.sampling(grid), grid))

_band_limit(s::FlowGeometries.SphericalSampling.AbstractSphericalSampling, grid::FlowGeometries.Grids.AbstractGrid) =
    FlowGeometries.SphericalSampling.has_bandlimit(s) ?
        FlowGeometries.SphericalSampling.bandlimit(s, _ring_count(grid)) : _cell_band_limit(grid)
_band_limit(_, grid::FlowGeometries.Grids.AbstractGrid) = _cell_band_limit(grid)

# The latitude rings of the grids whose samplings have a band limit.
_ring_count(grid::FlowGeometries.Grids.StructuredGrid) = size(FlowGeometries.Grids.mask(grid), 2)
_ring_count(grid::FlowGeometries.Grids.RingGrid) = FlowGeometries.Grids.nrings(grid)

function _cell_band_limit(grid::FlowGeometries.Grids.AbstractGrid{G,T}) where {G,T}
    R = FlowGeometries.Geometry.radius(FlowGeometries.Grids.grid_geometry(grid))
    return ceil(Int, T(π) * R / sqrt(maximum(FlowGeometries.Grids.measure(grid)))) - 1
end
