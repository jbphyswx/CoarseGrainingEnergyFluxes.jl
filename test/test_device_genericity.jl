
# The diagnostics on a device array that throws on host scalar indexing.
#
# `GPUBackend(KA.CPU())` holds plain `Array`s, so a loop that walks device memory element by element on
# the host still returns the right answer under it. A `JLArray` throws on scalar `getindex`, so here such
# a loop fails. The first testset asserts the guard is armed.
#
# Every input lives on the device and every result is compared with the serial one on the host. A map is
# the same arithmetic per cell on either side, so the two agree exactly. A domain sum is reduced in the
# device's own order: two orders of a sum of `n` terms lie within `2n·eps·Σ|xᵢ|` of each other.

# A KernelAbstractions backend whose memory is `JLArray`s. KernelAbstractions declares `synchronize` with
# no default and JLArrays supplies none for this backend, so it is defined here; the kernel-call method,
# KernelAbstractions' extension point for a backend, hands each launch to JLArrays.
struct _CGEFNoScalar <: KA.GPU end
KA.allocate(::_CGEFNoScalar, ::Type{T}, dims::Tuple) where {T} = JLArrays.JLArray{T}(undef, dims)
KA.synchronize(::_CGEFNoScalar) = nothing
# `@kernel` builds the host variant of a kernel when `isgpu` is false, and the launch below runs it on the
# host.
KA.isgpu(::_CGEFNoScalar) = false
function (k::KA.Kernel{_CGEFNoScalar, W, N, F})(args...; ndrange = nothing, workgroupsize = nothing) where {W, N, F}
    jl = KA.Kernel{JLArrays.JLBackend, W, N, F}(JLArrays.JLBackend(), k.f)
    return jl(args...; ndrange = ndrange, workgroupsize = workgroupsize)
end
# `Adapt.adapt(backend, x)` moves the arrays of `x` into the backend's memory.
Adapt.adapt_storage(::_CGEFNoScalar, a::Array) = Adapt.adapt(JLArrays.JLArray, a)
Adapt.adapt_storage(::_CGEFNoScalar, a::JLArrays.JLArray) = a

const _CGEF_DEV = CGEF.ComputationalBackends.GPUBackend(_CGEFNoScalar())
const _CGEF_HOST = CGEF.ComputationalBackends.SerialBackend()

_cgef_d(x::AbstractArray) = JLArrays.JLArray(x)
_cgef_d(::Nothing) = nothing

# A device result against the host one: each array is in device memory and equals the host's, which holds
# a nonzero value.
_cgef_same(d::AbstractArray, h::AbstractArray) = d isa JLArrays.JLArray && Array(d) == h && any(!iszero, h)
_cgef_same(d::NamedTuple, h::NamedTuple) = keys(d) == keys(h) && all(k -> _cgef_same(d[k], h[k]), keys(d))

# The bound on two orders of the area mean `Σ xᵢ Aᵢ / area` over the active cells of `og`.
_cgef_mean_atol(x, og, area) = 2 * length(x) * eps() * FG.Grids.integrate(abs.(x), og) / area

# Smooth fields on `g` from its coordinates divided by `s`, and a positive density.
function _cgef_dev_fields(g, s)
    c = [FG.Grids.coords(g, Tuple(I)...) ./ s for I in CartesianIndices(FG.Grids.size_tuple(g))]
    q(p, k) = length(p) >= k ? p[k] : 0.0
    u = [sin(q(p, 1) + 0.3) * cos(0.7 * q(p, 2)) + 0.2 * cos(q(p, 3)) for p in c]
    v = [cos(0.8 * q(p, 1)) * sin(q(p, 2) + 0.5) + 0.1 * sin(q(p, 3)) for p in c]
    w = [0.1 * sin(q(p, 2) + 2 * q(p, 1)) * cos(q(p, 3)) for p in c]
    θ = [sin(2 * q(p, 1)) * cos(q(p, 2)) + 0.5 * q(p, 3) for p in c]
    ρ = [1.0 + 0.3 * sin(q(p, 1) + q(p, 2)) for p in c]
    return u, v, w, θ, ρ
end

const _CGEF_R = 6.371e6
const _CGEF_CART = FG.Geometry.CartesianGeometry()

function _cgef_dev_cartesian()
    xs = range(0.0, 23000.0; length = 24)
    m = trues(24, 24)
    m[8:12, 6:9] .= false
    return FG.Grids.StructuredGrid(_CGEF_CART, xs, xs, m)
end

function _cgef_dev_sphere()
    m = trues(36, 13)
    m[10:14, 5:8] .= false
    return FG.Grids.StructuredGrid(FG.Geometry.SphericalGeometry(_CGEF_R), range(0.0; step = 2π / 36, length = 36),
                                   range(deg2rad(-60.0), deg2rad(60.0); length = 13), m; periodic = (true, false))
end

function _cgef_dev_shell()
    return FG.Grids.StructuredGrid(FG.Geometry.SphericalGeometry(_CGEF_R), range(0.0; step = 2π / 24, length = 24),
                                   range(deg2rad(-60.0), deg2rad(60.0); length = 9),
                                   range(_CGEF_R, _CGEF_R + 3000e3; length = 4))
end

Test.@testset "JLArray scalar indexing is disallowed" begin
    GPUArraysCore.allowscalar(false)
    a = JLArrays.JLArray([1.0, 2.0, 3.0])
    Test.@test_throws Exception a[1]
    Test.@test_throws Exception (a[2] = 0.0)
    # A whole-array copy back to the host is not scalar indexing and must still work.
    Test.@test Array(a) == [1.0, 2.0, 3.0]
    # The backend's buffers and moved arrays are device memory.
    Test.@test CGEF.Filtering._allocate(_CGEF_DEV, Float64, (2, 2)) isa JLArrays.JLArray
    Test.@test FG.Execution.on_backend(_CGEF_DEV, [1.0, 2.0]) isa JLArrays.JLArray
end

Test.@testset "compute_Π! on the device" begin
    D = CGEF.Diagnostics
    ga, th = CGEF.GaussianKernel(), CGEF.TopHatKernel()
    flux(g, k, ℓ, u, v, w, b; kw...) = D.compute_Π!(b === _CGEF_DEV ? _cgef_d(zero(u)) : zero(u),
                                                     b === _CGEF_DEV ? _cgef_d(u) : u,
                                                     b === _CGEF_DEV ? _cgef_d(v) : v,
                                                     b === _CGEF_DEV ? _cgef_d(w) : w, g, k, ℓ; execution = b, kw...)
    same(g, k, ℓ, u, v, w; kw...) =
        Test.@test _cgef_same(flux(g, k, ℓ, u, v, w, _CGEF_DEV; kw...), flux(g, k, ℓ, u, v, w, _CGEF_HOST; kw...))

    # A masked plane, each strategy, with and without a thin-layer `w`; a batch of planes.
    gc = _cgef_dev_cartesian()
    u, v, w = _cgef_dev_fields(gc, 5000.0)
    for k in (ga, th), s in (CGEF.Filtering.ZeroFill(), CGEF.Filtering.Deformable())
        same(gc, k, 4e3, u, v, nothing; mask_strategy = s)
    end
    same(gc, ga, 4e3, u, v, w)
    ub = cat(u, 2 .* v, u .* v; dims = 3)
    vb = cat(v, u, v .^ 2; dims = 3)
    same(gc, ga, 4e3, ub, vb, nothing)

    # Spheres: a regular latitude range, and a Clenshaw–Curtis sampling's latitudes, which are a vector.
    gs = _cgef_dev_sphere()
    us, vs, _ = _cgef_dev_fields(gs, 1.0)
    for k in (ga, th)
        same(gs, k, deg2rad(30.0) * _CGEF_R, us, vs, nothing)
    end
    same(gs, ga, deg2rad(30.0) * _CGEF_R, us, vs, nothing; mask_strategy = CGEF.Filtering.Deformable())
    # The in-core spin transforms run in the device's memory under its loops.
    same(gs, ga, deg2rad(30.0) * _CGEF_R, us, vs, nothing; transform = CGEF.SpectralBackends.DirectSumSpectralBackend())
    gq = FG.Connectivity.structured_grid(FG.SphericalSampling.ClenshawCurtisSampling(), 9;
                                         geometry = FG.Geometry.SphericalGeometry(_CGEF_R))
    uq, vq, _ = _cgef_dev_fields(gq, 1.0)
    same(gq, ga, deg2rad(60.0) * _CGEF_R, uq, vq, nothing)

    # A curvilinear mesh and a node set, differenced by least squares.
    n = 16
    θr = deg2rad(15.0)
    X = [1000.0 * (i * cos(θr) - 0.3j * sin(θr)) for i in 0:(n - 1), j in 0:(n - 1)]
    Y = [1000.0 * (i * sin(θr) + j * (1 + 0.3cos(θr))) for i in 0:(n - 1), j in 0:(n - 1)]
    gcv = FG.Grids.CurvilinearGrid(_CGEF_CART, X, Y, trues(n, n))
    ucv, vcv, _ = _cgef_dev_fields(gcv, 5000.0)
    same(gcv, th, 4e3, ucv, vcv, nothing)
    gn = FG.Grids.UnstructuredGrid(_CGEF_CART, vec(X), vec(Y), trues(n * n); k = 8)
    same(gn, ga, 3e3, vec(ucv), vec(vcv), nothing)

    # Volumes: a Cartesian box under both of its engines, and a spherical shell.
    x3 = range(0.0, 7000.0; length = 8)
    g3 = FG.Grids.StructuredGrid(_CGEF_CART, x3, x3, x3)
    u3, v3, w3 = _cgef_dev_fields(g3, 3000.0)
    for k in (th, ga)
        same(g3, k, 2.5e3, u3, v3, w3)
    end
    gsh = _cgef_dev_shell()
    ush, vsh, wsh = _cgef_dev_fields(gsh, (1.0, 1.0, 1e6))
    same(gsh, th, 3.0e6, ush, vsh, wsh)

    # A line.
    g1 = FG.Grids.StructuredGrid(_CGEF_CART, range(0.0, 31000.0; length = 32))
    u1, _ = _cgef_dev_fields(g1, 3000.0)
    for k in (th, ga)
        Test.@test _cgef_same(D.compute_Π!(_cgef_d(zero(u1)), _cgef_d(u1), g1, k, 4e3; execution = _CGEF_DEV),
                              D.compute_Π!(zero(u1), u1, g1, k, 4e3; execution = _CGEF_HOST))
    end

    # A workspace in host memory cannot serve a plan whose outputs are in device memory.
    pd = CGEF.Filtering.plan_filter(gc, ga, 4e3; execution = _CGEF_DEV)
    Test.@test_throws ArgumentError D.compute_Π!(_cgef_d(zero(u)), _cgef_d(u), _cgef_d(v), nothing, gc, ga, 4e3;
                                                 workspace = D.ΠWorkspace(gc), filter_plan = pd)
end

Test.@testset "Flux decompositions and budgets on the device" begin
    D = CGEF.Diagnostics
    k = CGEF.GaussianKernel()
    gc, gs = _cgef_dev_cartesian(), _cgef_dev_sphere()
    for (g, s, ℓ) in ((gc, 5000.0, 4e3), (gs, 1.0, deg2rad(30.0) * _CGEF_R))
        u, v, _, θ, ρ = _cgef_dev_fields(g, s)
        P = 1.0e5 .+ 100 .* θ
        ur, vr = 0.6 .* u, 0.4 .* v
        du, dv, dθ, dρ, dP, dur, dvr = map(_cgef_d, (u, v, θ, ρ, P, ur, vr))
        both(f) = (f(du, dv, dθ, dρ, dP, dur, dvr, _CGEF_DEV), f(u, v, θ, ρ, P, ur, vr, _CGEF_HOST))
        for (d, h) in (
            both((u, v, θ, ρ, P, ur, vr, b) -> D.tau_decomposition(u, v, g, k, ℓ; execution = b)),
            both((u, v, θ, ρ, P, ur, vr, b) -> D.compute_Π_strain_convergence(u, v, g, k, ℓ; execution = b)),
            both((u, v, θ, ρ, P, ur, vr, b) -> D.compute_Π_decomposed(u, v, ur, vr, g, k, ℓ; execution = b)),
            both((u, v, θ, ρ, P, ur, vr, b) -> (; Πθ = D.tracer_variance_flux(u, v, θ, g, k, ℓ; execution = b))),
            both((u, v, θ, ρ, P, ur, vr, b) -> (; Z = D.enstrophy_flux(u, v, g, k, ℓ; execution = b))),
            both((u, v, θ, ρ, P, ur, vr, b) -> (; ω = D.vorticity(u, v, g; execution = b))),
            both((u, v, θ, ρ, P, ur, vr, b) -> D.compressible_flux(u, v, ρ, P, g, k, ℓ; execution = b)),
        )
            Test.@test _cgef_same(d, h)
        end
    end

    # Volumes: a Cartesian box and a spherical shell.
    x3 = range(0.0, 7000.0; length = 8)
    for (g, s, ℓ) in ((FG.Grids.StructuredGrid(_CGEF_CART, x3, x3, x3), 3000.0, 2.5e3),
                      (_cgef_dev_shell(), (1.0, 1.0, 1e6), 3.0e6))
        th = CGEF.TopHatKernel()
        u, v, w, θ, ρ = _cgef_dev_fields(g, s)
        P = 1.0e5 .+ 100 .* θ
        du, dv, dw, dθ, dρ, dP = map(_cgef_d, (u, v, w, θ, ρ, P))
        Test.@test _cgef_same(D.tau_decomposition(du, dv, dw, g, th, ℓ; execution = _CGEF_DEV),
                              D.tau_decomposition(u, v, w, g, th, ℓ; execution = _CGEF_HOST))
        Test.@test _cgef_same(D.compute_Π_decomposed(du, dv, dw, 0.5 .* du, 0.3 .* dv, 0.2 .* dw, g, th, ℓ;
                                                     execution = _CGEF_DEV),
                              D.compute_Π_decomposed(u, v, w, 0.5 .* u, 0.3 .* v, 0.2 .* w, g, th, ℓ;
                                                     execution = _CGEF_HOST))
        Test.@test _cgef_same(D.tracer_variance_flux(du, dv, dw, dθ, g, th, ℓ; execution = _CGEF_DEV),
                              D.tracer_variance_flux(u, v, w, θ, g, th, ℓ; execution = _CGEF_HOST))
        Test.@test _cgef_same(D.compressible_flux(du, dv, dw, dρ, dP, g, th, ℓ; execution = _CGEF_DEV),
                              D.compressible_flux(u, v, w, ρ, P, g, th, ℓ; execution = _CGEF_HOST))
    end

    # A level stack's vertical derivative.
    gc = _cgef_dev_cartesian()
    f = [sin(0.3i + 0.2j) * cos(0.5l) for i in 1:24, j in 1:24, l in 1:6]
    Test.@test _cgef_same(CGEF.Derivatives.ddz!(_cgef_d(zero(f)), _cgef_d(f), gc, 50.0; execution = _CGEF_DEV),
                          CGEF.Derivatives.ddz!(zero(f), f, gc, 50.0; execution = _CGEF_HOST))
end

Test.@testset "Energies and sweeps on the device" begin
    D = CGEF.Diagnostics
    P = CGEF.Pipeline
    k = CGEF.GaussianKernel()
    zf = CGEF.Filtering.ZeroFill()
    x3 = range(0.0, 7000.0; length = 8)
    g3 = FG.Grids.StructuredGrid(_CGEF_CART, x3, x3, x3)
    g1 = FG.Grids.StructuredGrid(_CGEF_CART, range(0.0, 31000.0; length = 32))
    for (g, s, scales) in ((_cgef_dev_cartesian(), 5000.0, [3e3, 6e3]),
                           (_cgef_dev_sphere(), 1.0, deg2rad.([20.0, 40.0]) .* _CGEF_R),
                           (g3, 3000.0, [2e3, 4e3]), (g1, 3000.0, [3e3, 6e3]))
        u, v, w = _cgef_dev_fields(g, s)
        n = length(u)
        line = ndims(u) == 1
        ww = ndims(u) == 3 ? w : nothing
        if !line
            Ed = D.cumulative_energy(_cgef_d(u), _cgef_d(v), _cgef_d(ww), g, k, scales; execution = _CGEF_DEV)
            Eh = D.cumulative_energy(u, v, ww, g, k, scales; execution = _CGEF_HOST)
            Test.@test isapprox(Ed, Eh; rtol = 2n * eps())

            bd = D.band_energies(_cgef_d(u), _cgef_d(v), _cgef_d(ww), g, k, scales; maps = true, execution = _CGEF_DEV)
            bh = D.band_energies(u, v, ww, g, k, scales; maps = true, execution = _CGEF_HOST)
            Test.@test all(i -> _cgef_same(bd.band_maps[i], bh.band_maps[i]), eachindex(scales))
            Test.@test _cgef_same(bd.resolved_map, bh.resolved_map)
            og, area = D.output_grid(g, zf), D.active_area(g)
            Test.@test all(i -> abs(bd.bands[i] - bh.bands[i]) <= _cgef_mean_atol(bh.band_maps[i], og, area),
                           eachindex(scales))
            Test.@test abs(bd.resolved - bh.resolved) <= _cgef_mean_atol(bh.resolved_map, og, area)
        end

        rd = line ? CGEF.coarse_grain(_cgef_d(u), g; scales, kernel = k, execution = _CGEF_DEV) :
                    CGEF.coarse_grain(_cgef_d(u), _cgef_d(v), _cgef_d(ww), g; scales, kernel = k, execution = _CGEF_DEV)
        rh = line ? CGEF.coarse_grain(u, g; scales, kernel = k, execution = _CGEF_HOST) :
                    CGEF.coarse_grain(u, v, ww, g; scales, kernel = k, execution = _CGEF_HOST)
        Test.@test _cgef_same(rd.Π, rh.Π)
        Test.@test _cgef_device_energies(rd, rh, n)
    end

    # A batch of planes, every scale over the whole batch at once.
    gc = _cgef_dev_cartesian()
    u, v, _ = _cgef_dev_fields(gc, 5000.0)
    ub = cat(u, 2 .* v, u .* v; dims = 3)
    vb = cat(v, u, v .^ 2; dims = 3)
    scales = [3e3, 6e3]
    b = P.CoarseGrainBatchResult(gc, length(scales), (3,); execution = _CGEF_DEV)
    P.coarse_grain_batch!(b, _cgef_d(ub), _cgef_d(vb), gc; scales, kernel = k, execution = _CGEF_DEV)
    Test.@test b.Π isa JLArrays.JLArray
    Πb = Array(b.Π)
    for t in 1:3
        r = CGEF.coarse_grain(ub[:, :, t], vb[:, :, t], gc; scales, kernel = k, execution = _CGEF_HOST)
        Test.@test Πb[:, :, :, t] == r.Π
        Test.@test _cgef_device_energies(b.slices[t], r, length(u))
    end
end
