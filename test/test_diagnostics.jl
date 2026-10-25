
# The flux diagnostics against identities that hold on any grid, closed forms, and their definitions
# assembled from the filter and the derivatives.

const _CGEF_DF = CGEF.Filtering.Deformable()

# Fields with a prescribed kinetic-energy spectrum, from a streamfunction, `u = −∂ψ/∂y`, `v = ∂ψ/∂x`.
# `:exact` differentiates spectrally, so `E(k) ∝ k^(−α)` exactly; `:stencil` uses the three-point centred
# derivative's symbol `sin(kΔx)/Δx`, so the field is divergence-free for the diagnostics' derivative to
# round-off.
# Phases from an irrational rotation.
function _cgef_spectral_field(N, dx, α; symbol = :exact)
    k = [2π / (N * dx) * (i - 1 <= N ÷ 2 ? i - 1 : i - 1 - N) for i in 1:N]
    ψ̂ = zeros(ComplexF64, N, N)
    for j in 1:N, i in 1:N
        kmag = hypot(k[i], k[j])
        iszero(kmag) || (ψ̂[i, j] = kmag^(-(α + 3) / 2) * cis(2π * mod(0.7548776662466927i + 0.5698402909980532j, 1.0)))
    end
    ψ̂ = FFTW.fft(real(FFTW.ifft(ψ̂)))
    sym(κ) = symbol === :exact ? κ : sin(κ * dx) / dx
    û = [-im * sym(k[j]) * ψ̂[i, j] for i in 1:N, j in 1:N]
    v̂ = [im * sym(k[i]) * ψ̂[i, j] for i in 1:N, j in 1:N]
    return real(FFTW.ifft(û)), real(FFTW.ifft(v̂))
end

_cgef_logfit(x, y) = (lx = log10.(x); ly = log10.(y); sum((lx .- sum(lx) / length(lx)) .* ly) / sum((lx .- sum(lx) / length(lx)) .^ 2))

# τ the filter defines, on a Cartesian metric component by component; on a sphere from the moments of the
# ambient Cartesian components, rotated back to the local frame (Aluie 2019).
function _cgef_reference_tau(grid, u, v, plan, ::FG.Geometry.CartesianGeometry)
    ub, vb, uu, uv, vv = ntuple(_ -> similar(u), 5)
    CGEF.Filtering.filter_apply_batch!((ub, vb, uu, uv, vv), (u, v, u .* u, u .* v, v .* v), plan)
    return (uu .- ub .* ub, uv .- ub .* vb, vv .- vb .* vb)
end

function _cgef_reference_tau(grid, u, v, plan, geo::FG.Geometry.AbstractSphericalGeometry)
    sz = size(u)
    sym = ((1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3))
    p = ntuple(_ -> zeros(sz), 3)
    for I in CartesianIndices(sz)
        λ, φ = FG.Grids.coords(grid, Tuple(I)...)
        c = FG.Geometry.vector_to_cartesian(geo, u[I], v[I], λ, φ)
        p[1][I], p[2][I], p[3][I] = c[1], c[2], c[3]
    end
    pb = ntuple(_ -> zeros(sz), 3)
    mom = ntuple(_ -> zeros(sz), 6)
    CGEF.Filtering.filter_apply_batch!((pb..., mom...), (p..., ntuple(k -> p[sym[k][1]] .* p[sym[k][2]], 6)...), plan)
    τ = ntuple(_ -> zeros(sz), 3)
    for I in CartesianIndices(sz)
        λ, φ = FG.Grids.coords(grid, Tuple(I)...)
        t = ntuple(k -> mom[k][I] - pb[sym[k][1]][I] * pb[sym[k][2]][I], 6)
        l = FG.Geometry.tensor_to_local(geo, t[1], t[4], t[6], t[2], t[3], t[5], λ, φ)
        τ[1][I], τ[2][I], τ[3][I] = l.λλ, l.λφ, l.φφ
    end
    return τ
end

# On any grid: the Germano split rebuilds τ; the strain/convergence split's total is `−S̄:τ̄`; the
# Helmholtz channels sum to Π for any split and are empty but for the rotational one when the whole field
# is called rotational; and the Favre budget at constant density is `ρ·Π` with no baropycnal work. On a
# sphere `vf` names the vector filter: τ is the reference's under `CartesianComponents`, and the flux's own
# under the generalized convolution, which has no other.
function _cgef_check_identities(grid, u, v, ℓ; vf = nothing)
    k = CGEF.GaussianKernel()
    SER = CGEF.ComputationalBackends.SerialBackend()
    D = CGEF.Diagnostics
    sph = FG.Grids.grid_geometry(grid) isa FG.Geometry.AbstractSphericalGeometry
    kv = sph ? (; vector_filter = vf) : (;)
    d = D.tau_decomposition(u, v, grid, k, ℓ; kv...)
    ws = sph ? D.ΠWorkspace(grid; kv...) : D.ΠWorkspace(grid)
    Π = D.compute_Π!(zero(u), u, v, nothing, grid, k, ℓ; execution = SER, workspace = ws)
    τ = vf isa CGEF.Filtering.GeneralizedConvolution ? (ws.τ_xx, ws.τ_xy, ws.τ_yy) :
        _cgef_reference_tau(grid, u, v, CGEF.Filtering.plan_filter(grid, k, ℓ; execution = SER), FG.Grids.grid_geometry(grid))
    for (c, ref) in zip((:xx, :xy, :yy), τ)
        Test.@test _cgef_relerr(getproperty(d.L, c) .+ getproperty(d.C, c) .+ getproperty(d.R, c), ref) < 1e-10
    end
    Test.@test _cgef_relerr(D.compute_Π_strain_convergence(u, v, grid, k, ℓ; kv...).total, Π) < 1e-10
    dec = D.compute_Π_decomposed(u, v, u, v, grid, k, ℓ; kv...)
    Test.@test maximum(abs, dec.divergent) == 0 && maximum(abs, dec.cross) == 0
    Test.@test _cgef_relerr(dec.total, Π) < 1e-10
    dec2 = D.compute_Π_decomposed(u, v, 0.6 .* u, 0.6 .* v, grid, k, ℓ; kv...)
    Test.@test _cgef_relerr(dec2.total, Π) < 1e-10
    ΠD = CGEF.Diagnostics.compute_Π!(zero(u), u, v, nothing, grid, k, ℓ; execution = SER, mask_strategy = _CGEF_DF)
    fav = CGEF.Diagnostics.compressible_flux(u, v, fill(2.5, size(u)), fill(1.0e5, size(u)), grid, k, ℓ; mask_strategy = _CGEF_DF)
    Test.@test _cgef_relerr(fav.Π, 2.5 .* ΠD) < 1e-10
    Test.@test maximum(abs, fav.Λ) < 1e-12 * maximum(abs, fav.Π)
end

Test.@testset "Diagnostic identities on every grid architecture" begin
    cart = FG.Geometry.CartesianGeometry()
    n = 16
    xr = range(0.0, 1000.0 * (n - 1); length = n)
    X = [x for x in xr, y in xr]
    Y = [y for x in xr, y in xr]
    u = [sin(x / 5000) * cos(y / 6000) for x in xr, y in xr]
    v = [cos(x / 4000) * sin(y / 7000) for x in xr, y in xr]
    _cgef_check_identities(FG.Grids.StructuredGrid(cart, xr, xr), u, v, 3000.0)
    _cgef_check_identities(FG.Grids.CurvilinearGrid(cart, X, Y, trues(n, n)), u, v, 3000.0)
    _cgef_check_identities(FG.Grids.UnstructuredGrid(cart, vec(X), vec(Y), trues(n * n); k = 8), vec(u), vec(v), 3000.0)

    sgeo = FG.Geometry.SphericalGeometry()
    for grid in (FG.Grids.StructuredGrid(sgeo, range(0.0; step = 2π / 32, length = 32), range(deg2rad(-70.0), deg2rad(70.0); length = 20)),
                 FG.Grids.HEALPixGrid(sgeo, 4), FG.Grids.CubedSphereGrid(sgeo, 6), FG.Grids.IcosahedralGrid(sgeo, 4),
                 FG.Grids.RingGrid(sgeo, FG.SphericalSampling.ReducedGaussianSampling([16, 24, 32, 32, 24, 16])))
        sz = size(FG.Grids.mask(grid))
        c = [FG.Grids.coords(grid, Tuple(I)...) for I in CartesianIndices(sz)]
        for vf in (CGEF.Filtering.CartesianComponents(), CGEF.Filtering.GeneralizedConvolution())
            _cgef_check_identities(grid, [cos(p[2]) * sin(p[1]) for p in c], [cos(p[2]) * cos(p[1]) for p in c],
                                   1.3 * FG.Geometry.radius(sgeo) * sqrt(4π / prod(sz)); vf)
        end
    end
end

Test.@testset "Diagnostic identities in true 3-D" begin
    k = CGEF.GaussianKernel()
    SER = CGEF.ComputationalBackends.SerialBackend()
    sym = ((1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3))
    names = (:xx, :xy, :xz, :yy, :yz, :zz)
    n = 10
    x = range(0.0, 1000.0 * (n - 1); length = n)
    cube = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), x, x, x)
    R = 6.371e6
    lon = deg2rad.(collect(0.0:15.0:345.0))
    lat = deg2rad.(collect(-60.0:15.0:60.0))
    shell = FG.Grids.StructuredGrid(FG.Geometry.SphericalGeometry(R), lon, lat, collect(R:100e3:(R + 300e3)))
    for (grid, ℓ) in ((cube, 3000.0), (shell, 8.0e5))
        sz = FG.Grids.size_tuple(grid)
        c = [FG.Grids.coords(grid, Tuple(I)...) for I in CartesianIndices(sz)]
        u = [sin(p[1] / 4000 + p[2]) * cos(p[2] / 5000) for p in c]
        v = [cos(p[1] / 3000) * sin(p[3] / 6000 + p[2]) for p in c]
        w = [0.1 * sin(p[2] / 5000 + 2p[1]) * cos(p[3] / 4000) for p in c]
        d = CGEF.Diagnostics.tau_decomposition(u, v, w, grid, k, ℓ)
        ws = CGEF.Diagnostics.ΠWorkspace(grid; has_w = true)
        Π = CGEF.Diagnostics.compute_Π!(zeros(sz), u, v, w, grid, k, ℓ; workspace = ws, execution = SER)
        for (nm, ref) in zip(names, (ws.τ_xx, ws.τ_xy, ws.τ_xz, ws.τ_yy, ws.τ_yz, ws.τ_zz))
            Test.@test _cgef_relerr(getproperty(d.L, nm) .+ getproperty(d.C, nm) .+ getproperty(d.R, nm), ref) < 1e-9
        end
        if grid === cube
            plan = CGEF.Filtering.plan_filter(grid, k, ℓ; execution = SER)
            vel = (u, v, w)
            b = ntuple(_ -> zeros(sz), 3)
            m = ntuple(_ -> zeros(sz), 6)
            CGEF.Filtering.filter_apply_batch!((b..., m...), (vel..., ntuple(q -> vel[sym[q][1]] .* vel[sym[q][2]], 6)...), plan)
            Test.@test _cgef_relerr(ws.τ_xy, m[2] .- b[1] .* b[2]) < 1e-12
        end
        ΠD = CGEF.Diagnostics.compute_Π!(zeros(sz), u, v, w, grid, k, ℓ; execution = SER, mask_strategy = _CGEF_DF)
        fav = CGEF.Diagnostics.compressible_flux(u, v, w, fill(2.5, sz), fill(1.0e5, sz), grid, k, ℓ; mask_strategy = _CGEF_DF)
        Test.@test _cgef_relerr(fav.Π, 2.5 .* ΠD) < 1e-9
        Test.@test maximum(abs, fav.Λ) < 1e-12 * maximum(abs, fav.Π)
        dec = CGEF.Diagnostics.compute_Π_decomposed(u, v, w, u, v, w, grid, k, ℓ)
        Test.@test maximum(abs, dec.divergent) == 0 && maximum(abs, dec.cross) == 0
        Test.@test _cgef_relerr(dec.total, Π) < 1e-10
    end
end

# A normalized even kernel reproduces any linear field away from the edges, so solid-body rotation and
# pure strain filter to themselves and Π = 0. For pure strain `u = (Ωy, Ωx)` the stress is not zero:
# `τ₁₁ = τ₂₂ = Ω²σ²` with σ² the filter's own second moment, read off the filter by linearity, and
# `Π = −2Ωτ₁₂` vanishes with the cross moment.
Test.@testset "Linear fields: filtered to themselves, Π = 0, τ = Ω²σ²" begin
    N = 48
    dx = 1000.0
    xs = collect(range(-(N ÷ 2) * dx, ((N ÷ 2) - 1) * dx; length = N))
    grid = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), xs, xs)
    Ω = 1e-4
    ℓ = 6dx
    Yf = [xs[j] for i in 1:N, j in 1:N]
    for kern in (CGEF.TopHatKernel(), CGEF.GaussianKernel())
        pad = ceil(Int, CGEF.Kernels.kernel_radius(kern, ℓ) / dx) + 1
        ii = (pad + 1):(N - pad)
        plan = CGEF.Filtering.plan_filter(grid, kern, ℓ)
        flt(a) = CGEF.Filtering.filter_apply!(zeros(N, N), a, plan)
        f1 = flt(ones(N, N))
        σ² = (flt(Yf .^ 2) ./ f1 .- (flt(Yf) ./ f1) .^ 2)[N ÷ 2, N ÷ 2]
        kern isa CGEF.GaussianKernel && Test.@test σ² ≈ ℓ^2 / 12 rtol = 1e-9
        for (uf, vf) in (((x, y) -> -Ω * y, (x, y) -> Ω * x), ((x, y) -> Ω * y, (x, y) -> Ω * x))
            u = [uf(xs[i], xs[j]) for i in 1:N, j in 1:N]
            v = [vf(xs[i], xs[j]) for i in 1:N, j in 1:N]
            Test.@test maximum(abs, (flt(u) .- u)[ii, ii]) < 1e-14 * maximum(abs, u)
            Test.@test maximum(abs, (flt(v) .- v)[ii, ii]) < 1e-14 * maximum(abs, v)
            Π = CGEF.Diagnostics.compute_Π!(zeros(N, N), u, v, nothing, grid, kern, ℓ)
            Test.@test maximum(abs, Π[ii, ii]) < 1e-11 * Ω^3 * σ²
        end
        L, C, R = CGEF.Diagnostics.tau_decomposition([Ω * xs[j] for i in 1:N, j in 1:N], [Ω * xs[i] for i in 1:N, j in 1:N],
                                                     grid, kern, ℓ)
        Test.@test all(≈(Ω^2 * σ²; rtol = 1e-12), (L.xx .+ C.xx .+ R.xx)[ii, ii])
        Test.@test all(≈(Ω^2 * σ²; rtol = 1e-12), (L.yy .+ C.yy .+ R.yy)[ii, ii])
        Test.@test maximum(abs, (L.xy .+ C.xy .+ R.xy)[ii, ii]) < 1e-11 * Ω^2 * σ²
    end
end

# Rigid rotation of a shell, `u_e = Ω r cos φ`, has radial shear `Ω cos φ` and no strain, so Π = 0; the
# spherical vorticity of rotation about the polar axis is `2Ω sin φ`, half of it from the curvature term.
Test.@testset "Rigid rotation on a sphere: Π = 0 in a shell, ω = 2Ω sin φ on the surface" begin
    R = 6.371e6
    geo = FG.Geometry.SphericalGeometry(R)
    lon = deg2rad.(collect(0.0:10.0:350.0))
    lat = deg2rad.(collect(-60.0:10.0:60.0))
    r = collect(R:5000.0:(R + 20000.0))
    Ω = 7.292e-5
    u = [Ω * rk * cos(φ) for _ in lon, φ in lat, rk in r]
    zs = zeros(size(u))
    shell = FG.Grids.StructuredGrid(geo, lon, lat, r)
    Π = CGEF.Diagnostics.compute_Π!(zeros(size(u)), u, zs, zs, shell, CGEF.TopHatKernel(), 500e3; mask_strategy = _CGEF_DF)
    Test.@test maximum(abs, Π) < 1e-12 * Ω * maximum(abs, u)^2

    φr = range(deg2rad(-70.0), deg2rad(70.0); length = 20)
    surface = FG.Grids.StructuredGrid(geo, range(0.0; step = 2π / 32, length = 32), φr)
    φs = [FG.Grids.coords(surface, i, j)[2] for i in 1:32, j in 1:20]
    ω = CGEF.Diagnostics.vorticity(Ω .* R .* cos.(φs), zeros(32, 20), surface)
    # The centred latitude difference of `cos φ` leaves `Ω sin φ (1 − sin Δφ/Δφ) < Ω Δφ²/6`.
    Test.@test maximum(abs, ω[2:31, 2:19] .- 2Ω .* sin.(φs[2:31, 2:19])) < Ω * step(φr)^2 / 6
end

# With `w = 0` and `(u, v)` independent of height, the volume's six-term contraction is the plane's on
# every layer.
Test.@testset "A height-independent volume is the plane on every layer" begin
    x = collect(0.0:1.0:15.0)
    z = collect(0.0:50.0:100.0)
    cart = FG.Geometry.CartesianGeometry()
    u2 = [sin(0.4i) * cos(0.3j) for i in 1:16, j in 1:16]
    v2 = [cos(0.5i + 0.2j) for i in 1:16, j in 1:16]
    Π2 = CGEF.Diagnostics.compute_Π!(zeros(16, 16), u2, v2, nothing, FG.Grids.StructuredGrid(cart, x, x), CGEF.TopHatKernel(), 5.0)
    Π3 = CGEF.Diagnostics.compute_Π!(zeros(16, 16, 3), repeat(u2, 1, 1, 3), repeat(v2, 1, 1, 3), zeros(16, 16, 3),
                                     FG.Grids.StructuredGrid(cart, x, x, z), CGEF.TopHatKernel(), 5.0)
    Test.@test all(k -> Π3[:, :, k] ≈ Π2, 1:3)
end

# Srinivasan, Barkan & McWilliams (2023) eq. (10): `Π = Π_α + Π_δ` with `ᾱ ≥ 0` and `|Π_α| ≤ ᾱE′`,
# `E′ = (τ_uu + τ_vv)/2`; for a non-divergent field the convergence term is zero.
Test.@testset "Strain/convergence split" begin
    N = 24
    dx = 1000.0
    xs = 0.0:dx:(N - 1) * dx
    geo = FG.Geometry.CartesianGeometry()
    u = [sin(2π * 3 * i / N) * cos(2π * 2 * j / N) + 0.4cos(2π * 5 * i / N) for i in 1:N, j in 1:N]
    v = [cos(2π * 2 * i / N) * sin(2π * 4 * j / N) + 0.3sin(2π * 6 * j / N) for i in 1:N, j in 1:N]
    m = trues(N, N)
    m[6:9, 5:8] .= false
    grid = FG.Grids.StructuredGrid(geo, xs, xs, m)
    for kern in (CGEF.TopHatKernel(), CGEF.GaussianKernel()), s in (CGEF.Filtering.ZeroFill(), _CGEF_DF)
        ℓ = 4dx
        d = CGEF.Diagnostics.compute_Π_strain_convergence(u, v, grid, kern, ℓ; mask_strategy = s)
        Π = CGEF.Diagnostics.compute_Π!(zeros(N, N), u, v, nothing, grid, kern, ℓ; mask_strategy = s)
        Test.@test maximum(abs, d.total .- Π) < 1e-12 * maximum(abs, Π)
        Test.@test maximum(abs, d.strain) > 0.05 * maximum(abs, Π) && maximum(abs, d.convergence) > 0.05 * maximum(abs, Π)
        Test.@test all(>=(0), d.strain_magnitude)
        L, C, R = CGEF.Diagnostics.tau_decomposition(u, v, grid, kern, ℓ; mask_strategy = s)
        E′ = ((L.xx .+ C.xx .+ R.xx) .+ (L.yy .+ C.yy .+ R.yy)) ./ 2
        Test.@test all(I -> abs(d.strain[I]) <= abs(d.strain_magnitude[I] * E′[I]) + 1e-12 * maximum(abs, Π), CartesianIndices(Π))
        # `Deformable` zeroes an inactive target.
        s isa CGEF.Filtering.Deformable &&
            Test.@test all(f -> all(iszero, f[.!m]), (d.total, d.strain, d.convergence, d.divergence, d.strain_magnitude))
    end
    # Under `ZeroFill` land is fluid at rest, so the flux reaches over it by the kernel radius and one
    # derivative stencil and is zero beyond; the top-hat's compact support leaves a lake that far inside.
    lake = trues(N, N)
    lake[5:14, 5:14] .= false
    glake = FG.Grids.StructuredGrid(geo, xs, xs, lake)
    th = CGEF.TopHatKernel()
    dl = CGEF.Diagnostics.compute_Π_strain_convergence(u, v, glake, th, 4dx)
    dead = _cgef_beyond_reach(glake, CGEF.Kernels.kernel_radius(th, 4dx) + dx)
    Test.@test any(dead)
    Test.@test all(f -> all(iszero, f[dead]), (dl.total, dl.strain, dl.convergence, dl.divergence, dl.strain_magnitude))
    xp = range(0.0, dx * 64; length = 65)[1:64]
    us, vs = _cgef_spectral_field(64, dx, 3.0; symbol = :stencil)
    d = CGEF.Diagnostics.compute_Π_strain_convergence(us, vs, FG.Grids.StructuredGrid(geo, xp, xp; periodic = (true, true)),
                                                      CGEF.GaussianKernel(), 8dx)
    Test.@test maximum(abs, d.convergence) < 1e-11 * maximum(abs, d.strain)
end

# `Π_RR = −S̄ʳ:τʳʳ`, `Π_DD = −S̄ᵈ:τᵈᵈ`, `Π_X = −(S̄ʳ:τᵈᵈ + S̄ᵈ:τʳʳ + S̄ʳ:τ_X + S̄ᵈ:τ_X)`, assembled from the filter
# and the derivatives.
Test.@testset "Helmholtz channels against their definitions" begin
    n = 12
    ax = range(0.0, 1200.0; length = n + 1)[1:n]
    g3 = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), ax, ax, ax; periodic = (true, true, true))
    kern = CGEF.TopHatKernel()
    ℓ = 300.0
    field(a, b, c, φ) = [sinpi(a * x / 1200 + φ) * cospi(b * y / 1200) * sinpi(c * z / 1200 + φ / 2) for x in ax, y in ax, z in ax]
    u, v, w = field(2, 1, 1, 0.0), field(1, 3, 2, 0.3), field(3, 2, 1, 0.7)
    rot = (field(1, 2, 3, 0.11), field(2, 3, 1, 0.53), field(3, 1, 2, 0.91))
    dec = CGEF.Diagnostics.compute_Π_decomposed(u, v, w, rot..., g3, kern, ℓ)
    pl = CGEF.Filtering.plan_filter(g3, kern, ℓ; execution = CGEF.ComputationalBackends.SerialBackend())
    flt(f) = CGEF.Filtering.filter_apply!(zeros(n, n, n), f, pl)
    dv = (u .- rot[1], v .- rot[2], w .- rot[3])
    br, bd = map(flt, rot), map(flt, dv)
    SYM = ((1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3))
    τ(a, b, ba, bb) = ntuple(q -> flt(a[SYM[q][1]] .* b[SYM[q][2]]) .- ba[SYM[q][1]] .* bb[SYM[q][2]], 6)
    τRR, τDD = τ(rot, rot, br, br), τ(dv, dv, bd, bd)
    τX = τ(rot, dv, br, bd) .+ τ(dv, rot, bd, br)
    dd(f, d) = _cgef_dd(f, g3, d)
    strain(b) = ntuple(q -> ((i, j) = SYM[q]; i == j ? dd(b[i], i) : 0.5 .* (dd(b[i], j) .+ dd(b[j], i))), 6)
    con(S, t) = -(S[1] .* t[1] .+ S[4] .* t[4] .+ S[6] .* t[6] .+ 2 .* (S[2] .* t[2] .+ S[3] .* t[3] .+ S[5] .* t[5]))
    SR, SD = strain(br), strain(bd)
    Test.@test dec.rotational ≈ con(SR, τRR)
    Test.@test dec.divergent ≈ con(SD, τDD)
    Test.@test dec.cross ≈ con(SR, τDD) .+ con(SD, τRR) .+ con(SR, τX) .+ con(SD, τX)
end

# The tracer flux `τ_j = ⟨u_j θ⟩ − ū_j θ̄` is a vector: under `CartesianComponents` filtered as ambient
# Cartesian components and rotated back, the radial component carried in a shell.
Test.@testset "Tracer-variance flux on a sphere and in a shell against its definition" begin
    R = 6.371e6
    geo = FG.Geometry.SphericalGeometry(R)
    lon = deg2rad.(collect(0.0:10.0:350.0))
    lat = deg2rad.(collect(-60.0:10.0:60.0))
    kern = CGEF.TopHatKernel()
    for (grid, ℓ, w) in ((FG.Grids.StructuredGrid(geo, lon, lat), deg2rad(25.0) * R, nothing),
                         (FG.Grids.StructuredGrid(geo, lon, lat, collect(R:750e3:(R + 3000e3))), 3.0e6, 0.05))
        sz = FG.Grids.size_tuple(grid)
        c = [FG.Grids.coords(grid, Tuple(I)...) for I in CartesianIndices(sz)]
        θ = [sin(2p[1]) * cos(p[2]) + (length(p) == 3 ? 2 * (p[3] - R) / 3000e3 : 0.0) for p in c]
        u = [0.4 * cos(p[2]) * sin(p[1]) for p in c]
        v = [0.3 * sin(2p[2]) for p in c]
        wf = w === nothing ? nothing : [w * cos(p[1]) for p in c]
        Πθ = wf === nothing ?
            CGEF.Diagnostics.tracer_variance_flux(u, v, θ, grid, kern, ℓ; vector_filter = CGEF.Filtering.CartesianComponents()) :
            CGEF.Diagnostics.tracer_variance_flux(u, v, wf, θ, grid, kern, ℓ)
        P = [wf === nothing ? FG.Geometry.vector_to_cartesian(geo, u[I], v[I], c[I][1], c[I][2]) :
                              FG.Geometry.vector_to_cartesian(geo, u[I], v[I], wf[I], c[I][1], c[I][2]) for I in CartesianIndices(sz)]
        flt(f) = CGEF.Filtering.filter_field!(zeros(sz), f, grid, kern, ℓ)
        θb = flt(θ)
        τP = ntuple(q -> flt(getindex.(P, q) .* θ) .- flt(getindex.(P, q)) .* θb, 3)
        τL = [FG.Geometry.vector_from_cartesian(geo, τP[1][I], τP[2][I], τP[3][I], c[I][1], c[I][2]) for I in CartesianIndices(sz)]
        ref = .-(getindex.(τL, 1) .* _cgef_dd(θb, grid, 1) .+ getindex.(τL, 2) .* _cgef_dd(θb, grid, 2))
        wf === nothing || (ref .-= getindex.(τL, 3) .* _cgef_dd(θb, grid, 3))
        Test.@test maximum(abs, getindex.(τL, 1)) > 1e-4 * maximum(abs, u) * maximum(abs, θ)
        Test.@test Πθ ≈ ref
    end
end

# The enstrophy flux is Π's analogue in the same deformation gauge, `Z = −∂_jω̄ (⟨u_jω⟩ − ū_jω̄)`, with
# `ω = ∂v/∂x − ∂u/∂y`.
Test.@testset "Enstrophy flux: the deformation gauge" begin
    N = 32
    dx = 1000.0
    grid = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), 0.0:dx:(N - 1) * dx, 0.0:dx:(N - 1) * dx)
    u = [sin(2π * 3 * i / N) * cos(2π * 2 * j / N) + 0.4cos(2π * 5 * i / N) for i in 1:N, j in 1:N]
    v = [cos(2π * 2 * i / N) * sin(2π * 4 * j / N) + 0.3sin(2π * 6 * j / N) for i in 1:N, j in 1:N]
    ℓ = 6dx
    ω = CGEF.Diagnostics.vorticity(u, v, grid)
    Test.@test ω == _cgef_dd(v, grid, 1) .- _cgef_dd(u, grid, 2)
    Z = CGEF.Diagnostics.enstrophy_flux(u, v, grid, CGEF.GaussianKernel(), ℓ)
    plan = CGEF.Filtering.plan_filter(grid, CGEF.GaussianKernel(), ℓ)
    f(a) = CGEF.Filtering.filter_apply!(zeros(N, N), a, plan)
    ω̄ = f(ω)
    gx = _cgef_dd(ω̄, grid, 1)
    gy = _cgef_dd(ω̄, grid, 2)
    Test.@test maximum(abs, Z .+ (f(u .* ω) .- f(u) .* ω̄) .* gx .+ (f(v .* ω) .- f(v) .* ω̄) .* gy) < 1e-12 * maximum(abs, Z)
end

# On a masked grid the diagnostics' derivative window shrinks to the cell's active run: a linear field is
# differenced exactly wherever the run holds two cells, and a lone cell has no derivative.
Test.@testset "Derivatives on a masked grid" begin
    nx, ny = 24, 20
    m = trues(nx, ny)
    m[5:9, 6:11] .= false
    m[15, :] .= false
    m[[1, 3, 20, 21, 22, 23, 24], 3] .= false
    xs = collect(0.0:1000.0:23e3) .+ [300 * sin(1.3i) for i in 1:nx]
    ys = 0.0:800.0:(ny - 1) * 800.0
    g = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), xs, ys, m)
    lin = [2.5xs[i] - 0.7ys[j] for i in 1:nx, j in 1:ny]
    # Active cells past `I` in steps of `s`, on the bounded plane.
    reach(I, s) = (k = 0; J = I + s; while checkbounds(Bool, m, J) && m[J]; k += 1; J += s; end; k)
    for (d, slope) in ((1, 2.5), (2, -0.7))
        e = CartesianIndex(ntuple(k -> k == d ? 1 : 0, 2))
        ref = [m[I] && reach(I, e) + reach(I, -e) ≥ 1 ? slope : 0.0 for I in CartesianIndices(m)]
        Test.@test maximum(abs, _cgef_dd(lin, g, d) .- ref) < 1e-13
    end
end

# The Favre budget's terms, assembled from the filter and the derivatives: Π from the density-weighted
# stress and the product-rule strain `ρ̄∂_jũ_i = ∂_j(ρu_i)‾ − ũ_i∂_jρ̄`, the baropycnal work
# `Λ = ∇P̄·(ũ − ū)`, and the pressure dilatation `P̄∇·ū`, which is `P̄∇·ũ + Λ − ∇·(P̄(ũ − ū))`.
Test.@testset "Favre budget against its definition" begin
    N = 32
    dx = 1000.0
    xp = range(0.0, dx * N; length = N + 1)[1:N]
    grid = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), xp, xp; periodic = (true, true))
    ker = CGEF.GaussianKernel()
    ℓ = 4dx
    SER = CGEF.ComputationalBackends.SerialBackend()
    u = [sin(2π * 3 * i / N) * cos(2π * 2 * j / N) + 0.4cos(2π * 5 * i / N) for i in 1:N, j in 1:N]
    v = [cos(2π * 2 * i / N) * sin(2π * 4 * j / N) + 0.3sin(2π * 6 * j / N) for i in 1:N, j in 1:N]
    ρ = [1.0 + 0.3sin(2π * 2 * i / N) * cos(2π * 3 * j / N) for i in 1:N, j in 1:N]
    P = [10.0 + 2.0cos(2π * 3 * i / N) * sin(2π * 2 * j / N) for i in 1:N, j in 1:N]
    r = CGEF.Diagnostics.compressible_flux(u, v, ρ, P, grid, ker, ℓ; execution = SER)
    pl = CGEF.Filtering.plan_filter(grid, ker, ℓ; execution = SER)
    f(a) = CGEF.Filtering.filter_apply!(zeros(N, N), a, pl)
    d(a, j) = _cgef_dd(a, grid, j)
    ρ̄, P̄, ū, v̄ = f(ρ), f(P), f(u), f(v)
    ũ, ṽ = f(ρ .* u) ./ ρ̄, f(ρ .* v) ./ ρ̄
    Test.@test r.ρ̄ == ρ̄ && r.P̄ == P̄ && r.ũ == ũ && r.ṽ == ṽ
    A(m, ũi, j) = d(m, j) .- ũi .* d(ρ̄, j)
    τxx, τxy, τyy = f(ρ .* u .* u) ./ ρ̄ .- ũ .^ 2, f(ρ .* u .* v) ./ ρ̄ .- ũ .* ṽ, f(ρ .* v .* v) ./ ρ̄ .- ṽ .^ 2
    Test.@test r.Π == -(A(f(ρ .* u), ũ, 1) .* τxx .+ (A(f(ρ .* u), ũ, 2) .+ A(f(ρ .* v), ṽ, 1)) .* τxy .+ A(f(ρ .* v), ṽ, 2) .* τyy)
    Test.@test r.Λ == (d(P̄, 1) .* (f(ρ .* u) .- ρ̄ .* ū) .+ d(P̄, 2) .* (f(ρ .* v) .- ρ̄ .* v̄)) ./ ρ̄
    Test.@test r.pressure_dilatation == P̄ .* (d(ū, 1) .+ d(v̄, 2))
    Test.@test maximum(abs, P̄ .* (d(ũ, 1) .+ d(ṽ, 2)) .- r.pressure_dilatation) > 0.01 * maximum(abs, r.pressure_dilatation)
end

# Under `ZeroFill` land holds no fluid: ρ̄ is the fluid mass under the kernel, the Favre fields are
# density-weighted means over the fluid, zero where none lies under the kernel, and land values are never
# read. At constant density the Favre flux is ρ times the incompressible one where the kernel and the
# derivative stencil hold only fluid.
Test.@testset "Favre budget on a masked grid" begin
    N = 32
    dx = 1000.0
    xs = 0.0:dx:(N - 1) * dx
    geo = FG.Geometry.CartesianGeometry()
    m = trues(N, N)
    m[12:20, 10:18] .= false
    ker = CGEF.TopHatKernel()
    ℓ = 6dx
    rad = CGEF.Kernels.kernel_radius(ker, ℓ)
    SER = CGEF.ComputationalBackends.SerialBackend()
    u = [sin(2π * 3 * i / N) * cos(2π * 2 * j / N) + 0.4cos(2π * 5 * i / N) for i in 1:N, j in 1:N]
    v = [cos(2π * 2 * i / N) * sin(2π * 4 * j / N) + 0.3sin(2π * 6 * j / N) for i in 1:N, j in 1:N]
    ρ = [1.0 + 0.3sin(2π * 2 * i / N) * cos(2π * 3 * j / N) for i in 1:N, j in 1:N]
    P = [10.0 + 2.0cos(2π * 3 * i / N) * sin(2π * 2 * j / N) for i in 1:N, j in 1:N]
    for A in (u, v, ρ, P)
        A[.!m] .= NaN
    end
    grids = (FG.Grids.StructuredGrid(geo, xs, xs, m),
             FG.Grids.CurvilinearGrid(geo, [xs[i] for i in 1:N, j in 1:N], [xs[j] for i in 1:N, j in 1:N], m))
    for g in grids
        far = _cgef_beyond_reach(g, rad)
        Test.@test any(far)
        for s in (CGEF.Filtering.ZeroFill(), _CGEF_DF)
            r = CGEF.Diagnostics.compressible_flux(u, v, ρ, P, g, ker, ℓ; mask_strategy = s, execution = SER)
            Test.@test all(f -> all(isfinite, f), (r.Π, r.Λ, r.pressure_dilatation, r.ρ̄, r.P̄, r.ũ, r.ṽ))
            dead = s isa CGEF.Filtering.ZeroFill ? far : .!m
            Test.@test all(f -> all(iszero, f[dead]), (r.Π, r.Λ, r.pressure_dilatation, r.ũ, r.ṽ))
        end
    end
    g = grids[1]
    ρc = [m[I] ? 1.3 : NaN for I in CartesianIndices(m)]
    land = [FG.Grids.coords(g, Tuple(J)...) for J in CartesianIndices(m) if !m[J]]
    inner = [m[I] && all(x -> FG.Geometry.distance(geo, FG.Grids.coords(g, Tuple(I)...), x) > rad + dx, land) &&
             all(d -> min(I[d] - 1, N - I[d]) * dx > rad + dx, 1:2) for I in CartesianIndices(m)]
    Test.@test count(inner) > 100
    rc = CGEF.Diagnostics.compressible_flux(u, v, ρc, P, g, ker, ℓ; execution = SER)
    Πi = CGEF.Diagnostics.compute_Π!(zeros(N, N), u, v, nothing, g, ker, ℓ; execution = SER)
    Test.@test maximum(abs, rc.Π[inner] ./ 1.3 .- Πi[inner]) < 1e-12 * maximum(abs, Πi[inner])
end

# The scale-band decomposition telescopes: bands plus the resolved energy are the total, exactly where
# each filter conserves the domain integral, as `ZeroFill` does on a periodic grid, masked or not.
# Past a bounded edge the residual is the integral the filters lose there.
Test.@testset "Band energies sum to the total" begin
    N = 32
    dx = 1000.0
    u = [sin(2π * 3 * i / N) * cos(2π * 2 * j / N) + 0.4cos(2π * 7 * i / N) for i in 1:N, j in 1:N]
    v = [cos(2π * 2 * i / N) * sin(2π * 5 * j / N) + 0.3sin(2π * 6 * j / N) for i in 1:N, j in 1:N]
    scales = [3dx, 6dx, 12dx]
    geo = FG.Geometry.CartesianGeometry()
    xp = range(0.0, dx * N; length = N + 1)[1:N]
    grid = FG.Grids.StructuredGrid(geo, xp, xp; periodic = (true, true))
    Etot = 0.5 * sum(u .^ 2 .+ v .^ 2) / N^2
    for kern in (CGEF.TopHatKernel(), CGEF.GaussianKernel(), CGEF.Kernels.SmoothHatKernel())
        r = CGEF.Diagnostics.band_energies(u, v, grid, kern, scales; maps = true)
        Test.@test r.total ≈ Etot rtol = 1e-12
        # A non-negative kernel makes every band a variance, pointwise.
        Test.@test all(km -> all(>=(-1e-14 * Etot), km), r.band_maps)
    end
    rh = CGEF.Diagnostics.band_energies(u, v, grid, CGEF.Kernels.HighOrderKernel(; order = 3), [8dx, 16dx]; maps = true)
    Test.@test rh.total ≈ Etot rtol = 1e-12
    Test.@test any(km -> any(<(0), km), rh.band_maps)

    gb = FG.Grids.StructuredGrid(geo, 0.0:dx:(N - 1) * dx, 0.0:dx:(N - 1) * dx)
    k = CGEF.GaussianKernel()
    rb = CGEF.Diagnostics.band_energies(u, v, gb, k, scales)
    fu, fv = copy(u), copy(v)
    loss = 0.0
    for s in scales
        p = CGEF.Filtering.plan_filter(gb, k, s)
        sq = fu .^ 2 .+ fv .^ 2
        loss += (sum(CGEF.Filtering.filter_apply!(zeros(N, N), sq, p)) - sum(sq)) / (2N^2)
        fu = CGEF.Filtering.filter_apply!(zeros(N, N), fu, p)
        fv = CGEF.Filtering.filter_apply!(zeros(N, N), fv, p)
    end
    Test.@test abs(loss) > 1e-3 * Etot
    Test.@test rb.total - Etot ≈ loss rtol = 1e-10

    m = trues(N, N)
    m[8:13, 7:12] .= false
    gm = FG.Grids.StructuredGrid(geo, xp, xp, m; periodic = (true, true))
    Em = 0.5 * sum((u .^ 2 .+ v .^ 2)[m]) / count(m)
    for kern in (CGEF.TopHatKernel(), CGEF.GaussianKernel())
        rz = CGEF.Diagnostics.band_energies(u, v, gm, kern, scales)
        Test.@test rz.total ≈ Em rtol = 1e-12
        Test.@test abs(CGEF.Diagnostics.band_energies(u, v, gm, kern, scales; mask_strategy = _CGEF_DF).total - Em) > 1e-5 * Em
    end
end

# On a uniform periodic grid the kernel does not depend on position, so filtering commutes with the
# divergence and a divergence-free field stays divergence-free.
Test.@testset "filter ∘ div == div ∘ filter" begin
    N = 48
    dx = 1000.0
    xs = range(0.0, dx * N; length = N + 1)[1:N]
    grid = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), xs, xs; periodic = (true, true))
    u, v = _cgef_spectral_field(N, dx, 3.0; symbol = :stencil)
    div(a, b) = _cgef_dd(a, grid, 1) .+ _cgef_dd(b, grid, 2)
    Test.@test maximum(abs, div(u, v)) < 1e-12 * maximum(abs, _cgef_dd(u, grid, 1))
    w = u .+ 0.3 .* [sin(2π * 2 * i / N) for i in 1:N, j in 1:N]
    for kern in (CGEF.GaussianKernel(), CGEF.TopHatKernel()), ℓ in (8dx, 16dx)
        p = CGEF.Filtering.plan_filter(grid, kern, ℓ)
        flt(a) = CGEF.Filtering.filter_apply!(zeros(N, N), a, p)
        s = maximum(abs, _cgef_dd(flt(u), grid, 1))
        Test.@test maximum(abs, div(flt(u), flt(v))) < 1e-11 * s
        Test.@test maximum(abs, flt(div(w, v)) .- div(flt(w), flt(v))) < 1e-11 * s
    end
end

# Sadek & Aluie (2018) eq. 18: a kernel with `p` vanishing moments recovers a `k^(−α)` spectrum while
# `α < p + 2` and saturates at `k^(−(p+2))` above. The Gaussian has `p = 1`; `HighOrderKernel{3}` has
# `p = 3`, so it recovers `α = 4`, which the Gaussian cannot.
Test.@testset "Filtering spectrum slope: min(α, p + 2)" begin
    N = 256
    dx = 1000.0
    xs = range(0.0, dx * N; length = N + 1)[1:N]
    grid = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), xs, xs; periodic = (true, true))
    function slope(kern, α, lo, hi)
        scales = collect(exp10.(range(log10(lo / 1.6), log10(hi * 1.6); length = 17)))
        kℓ = 1.0 ./ scales
        u, v = _cgef_spectral_field(N, dx, α)
        Ẽ = CGEF.Diagnostics.spectral_density(CGEF.Diagnostics.cumulative_energy(u, v, nothing, grid, kern, scales), kℓ)
        w = [i for i in eachindex(scales) if lo <= scales[i] <= hi && Ẽ[i] > 0]
        Test.@test length(w) >= 6
        return _cgef_logfit(kℓ[w], Ẽ[w])
    end
    g = CGEF.GaussianKernel()
    Test.@test slope(g, 5 / 3, 8dx, 40dx) ≈ -5 / 3 atol = 0.2
    Test.@test -3.15 < slope(g, 7.0, 8dx, 40dx) < -2.75
    ho = CGEF.Kernels.HighOrderKernel(; order = 3)
    s4 = slope(ho, 4.0, 12dx, 48dx)
    Test.@test s4 ≈ -4.0 atol = 0.25
    Test.@test s4 < slope(g, 4.0, 12dx, 48dx) - 0.5
end

# On a sphere every energy diagnostic filters the velocity as the flux does, by the vector filter it names.
# Solid-body rotation about the polar axis is the degree-1 toroidal field `e_r × ∇*sin φ`, and degree 1 in
# its ambient Cartesian components, which a radial kernel maps to multiples of themselves (Funk–Hecke).
# Under `CartesianComponents` its resolved energy over `½cos²φ` is one constant across latitude away from
# a regional grid's edges; under the generalized convolution on the whole sphere it is `Ĝ(1)²`.
Test.@testset "Energies on a sphere filter the velocity as the flux does" begin
    SER = CGEF.ComputationalBackends.SerialBackend()
    CC = CGEF.Filtering.CartesianComponents()
    R = 6.371e6
    lon = range(0.0; step = 2π / 144, length = 144)
    lat = range(deg2rad(-75.0); stop = deg2rad(75.0), length = 61)
    g = FG.Grids.StructuredGrid(FG.Geometry.SphericalGeometry(R), lon, lat; periodic = (true, false))
    k = CGEF.GaussianKernel()
    scales = [600e3, 1200e3]
    u = [cos(φ) + 0.3sin(2λ) * cos(φ)^2 for λ in lon, φ in lat]
    v = [0.2cos(3λ) * cos(φ) for λ in lon, φ in lat]
    for vf in (nothing, CC)
        r = CGEF.coarse_grain(u, v, g; scales, kernel = k, execution = SER, vector_filter = vf)
        Test.@test CGEF.Diagnostics.cumulative_energy(u, v, nothing, g, k, scales; execution = SER, vector_filter = vf) ≈
                   r.cumulative_energy rtol = 1e-13
    end
    ub = [cos(φ) for λ in lon, φ in lat]
    rows = findall(φ -> abs(φ) <= deg2rad(50.0) + 1e-9, lat)
    ratio = CGEF.Diagnostics.band_energies(ub, zero(ub), g, k, [1200e3]; execution = SER, vector_filter = CC).resolved_map ./
            [0.5cos(φ)^2 for λ in lon, φ in lat]
    Test.@test maximum(ratio[:, rows]) - minimum(ratio[:, rows]) < 1e-8 * ratio[1, rows[1]]

    cc = FG.Connectivity.structured_grid(FG.SphericalSampling.ClenshawCurtisSampling(), 24; geometry = FG.Geometry.SphericalGeometry(R))
    φc = [FG.Grids.coords(cc, Tuple(I)...)[2] for I in CartesianIndices(FG.Grids.size_tuple(cc))]
    for kern in (k, CGEF.TopHatKernel())
        rmap = CGEF.Diagnostics.band_energies(cos.(φc), zero(φc), cc, kern, [1200e3]; execution = SER).resolved_map
        Test.@test rmap ≈ _cgef_legendre_transfer(kern, 1, 1200e3, R)^2 .* 0.5 .* cos.(φc) .^ 2 rtol = 1e-11
    end
end

# ---- The generalized convolution of vectors and tensors on a sphere (Aluie 2019) ----------------------

# Degree-`n` real harmonics on the unit sphere as `(n, f, fλ, fφ)`, functions of `(λ, φ)`: `f ∝ P_n^m(sin φ)
# cos mλ`, `fλ = (1/cos φ)∂f/∂λ` and `fφ = ∂f/∂φ`. `∇*f = (fλ, fφ)` and `e_r × ∇*f = (−fφ, fλ)` in
# (east, north) components.
const _CGEF_HARMONICS = (
    (1, (λ, φ) -> sin(φ), (λ, φ) -> 0.0, (λ, φ) -> cos(φ)),
    (2, (λ, φ) -> sin(φ) * cos(φ) * cos(λ), (λ, φ) -> -sin(φ) * sin(λ), (λ, φ) -> cos(2φ) * cos(λ)),
    (3, (λ, φ) -> sin(φ) * cos(φ)^2 * cos(2λ), (λ, φ) -> -2sin(φ) * cos(φ) * sin(2λ),
        (λ, φ) -> cos(φ) * (cos(φ)^2 - 2sin(φ)^2) * cos(2λ)),
    (4, (λ, φ) -> sin(φ) * cos(φ)^3 * cos(3λ), (λ, φ) -> -3sin(φ) * cos(φ)^2 * sin(3λ),
        (λ, φ) -> cos(φ)^2 * (cos(φ)^2 - 3sin(φ)^2) * cos(3λ)),
    (5, (λ, φ) -> (63sin(φ)^5 - 70sin(φ)^3 + 15sin(φ)) / 8, (λ, φ) -> 0.0,
        (λ, φ) -> cos(φ) * (315sin(φ)^4 - 210sin(φ)^2 + 15) / 8),
)

# The larger component error of `(x, y)` against `c·(u, v)`, over the field's largest component.
_cgef_vec_err(x, y, u, v, c) =
    max(maximum(abs, x .- c .* u), maximum(abs, y .- c .* v)) / max(maximum(abs, u), maximum(abs, v))

_cgef_points(g) = [FG.Grids.coords(g, Tuple(I)...) for I in CartesianIndices(FG.Grids.size_tuple(g))]

# FastSphericalHarmonics' own grid, and a Gauss–Legendre one on which NUFSHT transforms, whose measure is a
# quadrature exact to twice its band limit; each with the error its transforms reach.
_cgef_harmonic_grids(geo) = (
    (FG.Connectivity.structured_grid(FG.SphericalSampling.ClenshawCurtisSampling(), 24; geometry = geo), 1e-11),
    (FG.Connectivity.structured_grid(FG.SphericalSampling.GaussLegendreSampling(), 24; geometry = geo), 1e-8),
)

# Lemma 1: every degree-`n` vector harmonic, the gradient `∇*f` and the rotational `e_r × ∇*f`, is filtered
# to `Ĝ(n)` times itself, `Ĝ` the kernel's Legendre transform.
Test.@testset "Generalized convolution: Ĝ(n) times each degree-n vector harmonic" begin
    D = CGEF.Diagnostics
    SER = CGEF.ComputationalBackends.SerialBackend()
    R = 6.371e6
    for (g, tol) in _cgef_harmonic_grids(FG.Geometry.SphericalGeometry(R))
        sz = FG.Grids.size_tuple(g)
        pts = _cgef_points(g)
        ws = D.ΠWorkspace(g)
        for kern in (CGEF.GaussianKernel(), CGEF.TopHatKernel()), ℓ in (1.5e6, 3e6), (n, _, fλ, fφ) in _CGEF_HARMONICS
            Ĝ = _cgef_legendre_transfer(kern, n, ℓ, R)
            a = [fλ(p...) for p in pts]
            b = [fφ(p...) for p in pts]
            for (u, v) in ((a, b), (-b, a))
                D.compute_Π!(zeros(sz), u, v, nothing, g, kern, ℓ; workspace = ws, execution = SER)
                Test.@test _cgef_vec_err(ws.u_filt, ws.v_filt, u, v, Ĝ) < tol
            end
        end
    end
end

# The ambient Cartesian components of `e_r × ∇*f` are degree-`n` harmonics, so the scalar transform of
# each component is the generalized convolution there. Those of `∇*f` carry degrees `n ± 1`: with
# `G = ∇(rⁿf)` on the unit sphere, of degree `n − 1`, and `H = x f − G/(2n+1)` the degree-`(n+1)` part of
# `x f`, `∇*f = (n+1)/(2n+1)·G − n·H`, so the tangent part of the filtered gradient is
# `((n+1)Ĝ(n−1) + nĜ(n+1))/(2n+1)` times `∇*f`.
Test.@testset "CartesianComponents: equal on rotational harmonics, different on gradients" begin
    D = CGEF.Diagnostics
    SER = CGEF.ComputationalBackends.SerialBackend()
    R = 6.371e6
    cc, tol = first(_cgef_harmonic_grids(FG.Geometry.SphericalGeometry(R)))
    sz = FG.Grids.size_tuple(cc)
    pts = _cgef_points(cc)
    wc = D.ΠWorkspace(cc; vector_filter = CGEF.Filtering.CartesianComponents())
    for kern in (CGEF.GaussianKernel(), CGEF.TopHatKernel()), (n, _, fλ, fφ) in _CGEF_HARMONICS
        ℓ = 3e6
        fp = CGEF.Filtering.plan_filter(cc, kern, ℓ; method = CGEF.Filtering.Spectral(), execution = SER)
        Ĝ = _cgef_legendre_transfer(kern, n, ℓ, R)
        a = [fλ(p...) for p in pts]
        b = [fφ(p...) for p in pts]
        D.compute_Π!(zeros(sz), -b, a, nothing, cc, kern, ℓ; workspace = wc, filter_plan = fp, execution = SER)
        Test.@test _cgef_vec_err(wc.u_filt, wc.v_filt, -b, a, Ĝ) < tol
        D.compute_Π!(zeros(sz), a, b, nothing, cc, kern, ℓ; workspace = wc, filter_plan = fp, execution = SER)
        c = ((n + 1) * _cgef_legendre_transfer(kern, n - 1, ℓ, R) + n * _cgef_legendre_transfer(kern, n + 1, ℓ, R)) /
            (2n + 1)
        Test.@test _cgef_vec_err(wc.u_filt, wc.v_filt, a, b, c) < tol
    end
end

# Proposition 3: a symmetric tangent tensor's trace is filtered as a scalar and its traceless part, the
# spin-2 field `T_ee − T_nn + 2iT_en`, by `Ĝ(n)` per degree. A degree-1 velocity's `u⊗u` has a traceless
# part of degree 2 alone and a trace `|u|²` of degrees 0, 1 and 2; the E-mode `∇*∇*f − ½∇²f I` of a
# degree-3 `f`, its 45° rotation (the B-mode) and `f I` are degree 3.
Test.@testset "Generalized convolution of tensors: Ĝ(n) per degree" begin
    D = CGEF.Diagnostics
    SER = CGEF.ComputationalBackends.SerialBackend()
    R = 6.371e6
    for (g, tol) in _cgef_harmonic_grids(FG.Geometry.SphericalGeometry(R)), kern in (CGEF.GaussianKernel(), CGEF.TopHatKernel())
        ℓ = 2.5e6
        sz = FG.Grids.size_tuple(g)
        λs = [p[1] for p in _cgef_points(g)]
        φs = [p[2] for p in _cgef_points(g)]
        Ĝ1, Ĝ2, Ĝ3 = (_cgef_legendre_transfer(kern, n, ℓ, R) for n in 1:3)

        # `u = ∇*x + 0.7 e_r × ∇*z`, with `|u|² = 1 − x² + 1.4y + 0.49(1 − z²)`.
        x, y, z = cos.(φs) .* cos.(λs), cos.(φs) .* sin.(λs), sin.(φs)
        u = -sin.(λs) .- 0.7 .* cos.(φs)
        v = -sin.(φs) .* cos.(λs)
        ws = D.ΠWorkspace(g)
        D.compute_Π!(zeros(sz), u, v, nothing, g, kern, ℓ; workspace = ws, execution = SER)
        tr = 1.49 * 2 / 3 .+ Ĝ1 .* 1.4 .* y .- Ĝ2 .* ((x .^ 2 .- 1 / 3) .+ 0.49 .* (z .^ 2 .- 1 / 3))
        d = Ĝ2 .* (u .^ 2 .- v .^ 2)
        scale = maximum(u .^ 2 .+ v .^ 2)
        Test.@test maximum(abs, ws.uu_filt .- (tr .+ d) ./ 2) < tol * scale
        Test.@test maximum(abs, ws.vv_filt .- (tr .- d) ./ 2) < tol * scale
        Test.@test maximum(abs, ws.uv_filt .- Ĝ2 .* u .* v) < tol * scale

        sf = ws.spin
        D._spin_transfer!(sf, kern, ℓ)
        s = sin.(φs)
        A = (6 .* s .* cos.(φs) .^ 2 .- 4 .* s) .* cos.(2 .* λs)     # E_ee − E_nn
        B = -2 .* cos.(2 .* φs) .* sin.(2 .* λs)                      # E_en
        f3 = s .* cos.(φs) .^ 2 .* cos.(2 .* λs)
        T = (zeros(sz), zeros(sz), zeros(sz))
        for (Q, trace, want) in (((k -> complex(A[k], 2B[k])), (k -> 0.0), (A ./ 2, B, -A ./ 2)),
                                 ((k -> complex(-2B[k], A[k])), (k -> 0.0), (-B, A ./ 2, B)),
                                 ((k -> 0.0im), (k -> f3[k]), (f3 ./ 2, zero(f3), f3 ./ 2)))
            D._spin_tensor!(T..., sf, false, Q, trace, 0)
            Test.@test maximum(maximum.(abs, T .- Ĝ3 .* want)) < tol * maximum(abs, A)
        end
    end
end

# `div u + i curl u` on the unit sphere through NUFSHT's spin transforms: `u = ∇*f + e_r × ∇*g` has
# `u_θ + iu_φ = Σ λ_ℓ(f_ℓm + ig_ℓm) ₁Y_ℓm`, `λ_ℓ = √(ℓ(ℓ+1))`, so `div u + i curl u = ∇²(f + ig)` is the
# spin-0 field of the spin-1 coefficients times `−λ_ℓ`. `w` are the cells' measures on the unit sphere.
function _cgef_div_curl(u, v, θ, φ, w, lmax)
    p1 = NUFSHT.make_spin_plan(ComplexF64, θ, φ, lmax, 1)
    p0 = NUFSHT.make_spin_plan(ComplexF64, θ, φ, lmax, 0)
    C = NUFSHT.allocate_coefficients(p1)
    NUFSHT.nusht_analyze!(C, complex.(-vec(v), vec(u)), p1; weights = w)
    for l in 0:lmax
        C[l + 1, :, :] .*= -sqrt(l * (l + 1))
    end
    d = zeros(ComplexF64, length(θ))
    NUFSHT.nusht_type2_spin!(d, C, p0)
    return reshape(real.(d), size(u)), reshape(imag.(d), size(u))
end

# Proposition 1: filtering commutes with the surface divergence and curl, `div ū = (div u)‾` and
# `curl ū = (curl u)‾`, the right sides filtered as scalars.
Test.@testset "Generalized convolution commutes with div and curl" begin
    D = CGEF.Diagnostics
    SER = CGEF.ComputationalBackends.SerialBackend()
    R = 6.371e6
    g = last(first.(_cgef_harmonic_grids(FG.Geometry.SphericalGeometry(R))))
    sz = FG.Grids.size_tuple(g)
    pts = _cgef_points(g)
    θ = [π / 2 - p[2] for p in vec(pts)]
    φ = [p[1] for p in vec(pts)]
    w = collect(vec(FG.Grids.measure(g))) ./ R^2
    lmax = FG.SphericalSampling.bandlimit(FG.Grids.sampling(g), sz[2])
    a = (1.0, 0.5, -0.8, 0.3, 0.6)
    b = (0.4, -0.7, 0.2, 0.9, -0.5)
    u = sum(a[n] .* [h[3](p...) for p in pts] .- b[n] .* [h[4](p...) for p in pts] for (n, h) in enumerate(_CGEF_HARMONICS))
    v = sum(a[n] .* [h[4](p...) for p in pts] .+ b[n] .* [h[3](p...) for p in pts] for (n, h) in enumerate(_CGEF_HARMONICS))
    dv = sum(-h[1] * (h[1] + 1) * a[n] .* [h[2](p...) for p in pts] for (n, h) in enumerate(_CGEF_HARMONICS))
    cu = sum(-h[1] * (h[1] + 1) * b[n] .* [h[2](p...) for p in pts] for (n, h) in enumerate(_CGEF_HARMONICS))
    d0, c0 = _cgef_div_curl(u, v, θ, φ, w, lmax)
    Test.@test _cgef_relerr(d0, dv) < 1e-8 && _cgef_relerr(c0, cu) < 1e-8
    ws = D.ΠWorkspace(g)
    for kern in (CGEF.GaussianKernel(), CGEF.TopHatKernel()), ℓ in (1.5e6, 3e6)
        D.compute_Π!(zeros(sz), u, v, nothing, g, kern, ℓ; workspace = ws, execution = SER)
        d1, c1 = _cgef_div_curl(ws.u_filt, ws.v_filt, θ, φ, w, lmax)
        flt(f) = CGEF.Filtering.filter_field!(zeros(sz), f, g, kern, ℓ; method = CGEF.Filtering.Spectral(), execution = SER)
        Test.@test _cgef_relerr(d1, flt(dv)) < 1e-6
        Test.@test _cgef_relerr(c1, flt(cu)) < 1e-6
    end
end

# Under `ZeroFill` a regional field is the field extended by zero over the sphere: masked cells, and a
# regional grid's cells taken alone, filter as the whole sphere holding zero there.
Test.@testset "Generalized convolution of a regional field: the zero-extended sphere's" begin
    D = CGEF.Diagnostics
    SER = CGEF.ComputationalBackends.SerialBackend()
    R = 6.371e6
    geo = FG.Geometry.SphericalGeometry(R)
    vf = CGEF.Filtering.GeneralizedConvolution(; lmax = 16)
    nus = CGEF.SpectralBackends.NUFSHTSpectralBackend()
    full = FG.Connectivity.structured_grid(FG.SphericalSampling.ClenshawCurtisSampling(), 24; geometry = geo)
    pts = _cgef_points(full)
    box = [abs(p[2]) <= deg2rad(40.0) && p[1] <= deg2rad(120.0) for p in pts]
    masked = FG.Connectivity.structured_grid(FG.SphericalSampling.ClenshawCurtisSampling(), 24; geometry = geo, mask = box)
    u = [cos(p[2]) * sin(2p[1]) + 0.3sin(p[2]) for p in pts]
    v = [cos(3p[1]) * cos(p[2])^2 for p in pts]
    uz, vz = ifelse.(box, u, 0.0), ifelse.(box, v, 0.0)
    un, vn = ifelse.(box, u, NaN), ifelse.(box, v, NaN)
    region = FG.Grids.UnstructuredGrid(geo, [p[1] for p in pts[box]], [p[2] for p in pts[box]],
                                       collect(vec(FG.Grids.measure(full)))[vec(box)], trues(count(box)))
    k = CGEF.GaussianKernel()
    filtered(g, a, b; kw...) = (ws = D.EnergyWorkspace(g; vector_filter = vf, kw...);
                                D.cumulative_energy!(zeros(1), a, b, nothing, g, k, [2e6]; workspace = ws, execution = SER);
                                (ws.u_filt, ws.v_filt))
    fz = filtered(full, uz, vz; transform = nus)
    fm = filtered(masked, un, vn; transform = nus)
    fr = filtered(region, u[box], v[box])
    # The same points give the same transforms; the regional grid's own, over fewer points, agree to
    # the NUFFT's tolerance.
    Test.@test fm[1] == fz[1] && fm[2] == fz[2]
    Test.@test fr[1] ≈ fz[1][box] rtol = 1e-8
    Test.@test fr[2] ≈ fz[2][box] rtol = 1e-8
    Test.@test maximum(abs, fz[1][.!box]) > 1e-3 * maximum(abs, u)
end

# Solid-body rotation is the degree-1 rotational harmonic, filtered to `Ĝ(1)` times itself, which has no
# strain: Π = 0 to round-off over the whole sphere.
Test.@testset "Solid-body rotation under the generalized convolution: Π = 0" begin
    SER = CGEF.ComputationalBackends.SerialBackend()
    R = 6.371e6
    cc = first(first(_cgef_harmonic_grids(FG.Geometry.SphericalGeometry(R))))
    Ω = 7.292e-5
    u = [Ω * R * cos(p[2]) for p in _cgef_points(cc)]
    for kern in (CGEF.GaussianKernel(), CGEF.TopHatKernel())
        Π = CGEF.Diagnostics.compute_Π!(zero(u), u, zero(u), nothing, cc, kern, 2e6; execution = SER)
        Test.@test maximum(abs, Π) < 1e-11 * Ω * maximum(abs, u)^2
    end
end

# The in-core sums, over every cell and by rows, against NUFSHT's spin plans on the same cells and
# weights, both ways and at every spin `|s| ≤ 2`; and against closed forms on a Gauss–Legendre grid, whose
# quadrature is exact to twice its band limit: the spin-1 fields of `∇*f` and `e_r × ∇*f` and the spin-2
# E-mode of a degree-`n` `f` have coefficients of degree `n` alone, analysis inverts synthesis, and a flux
# through it scales a vector harmonic by `Ĝ(n)`.
Test.@testset "Direct-sum spin transforms against NUFSHT and closed forms" begin
    F = CGEF.Filtering
    D = CGEF.Diagnostics
    SER = CGEF.ComputationalBackends.SerialBackend()
    R = 6.371e6
    g = FG.Connectivity.structured_grid(FG.SphericalSampling.GaussLegendreSampling(), 10;
                                        geometry = FG.Geometry.SphericalGeometry(R))
    L = 9
    θ, φ, w = F._spin_nodes(g)
    for tr in (F.DirectSpinTransforms(g; lmax = L), F.RingSpinTransforms(g; lmax = L))
        M = length(θ)
        G1 = ones(L + 1)
        synth(C, s) = F.spin_synthesize!(F.spin_field(tr), C, F.spin_coefficients(tr), G1, tr, s)
        analyze(f, s) = F.spin_analyze!(F.spin_coefficients(tr), f, tr, s)
        for s in -2:2
            plan = NUFSHT.make_spin_plan(ComplexF64, θ, φ, L, s)
            f = [complex(sin(3k), cos(5k)) for k in 1:M]
            Cn = NUFSHT.allocate_coefficients(plan)
            NUFSHT.nusht_analyze!(Cn, f, plan; weights = w)
            Test.@test _cgef_relerr(vec(analyze(f, s)), vec(Cn)) < 1e-8
            C = [l >= max(abs(m), abs(s)) ? complex(cos(l + 2m), sin(3l - m)) : 0.0im for l in 0:L, m in -L:L]
            fn = zeros(ComplexF64, M)
            NUFSHT.nusht_type2_spin!(fn, reshape(C, L + 1, 2L + 1, 1), plan)
            fd = synth(C, s)
            Test.@test _cgef_relerr(fd, fn) < 1e-8
            Test.@test _cgef_relerr(analyze(fd, s), C) < 1e-11
        end

        pts = vec(_cgef_points(g))
        only_degree(C, n) = maximum(abs, C[[l != n for l in 0:L], :]) < 1e-11 * maximum(abs, C)
        for (n, _, fλ, fφ) in _CGEF_HARMONICS
            a = [fλ(p...) for p in pts]
            b = [fφ(p...) for p in pts]
            for U in (complex.(a, b), complex.(-b, a))
                C = analyze(U, 1)
                Test.@test only_degree(C, n)
                Test.@test _cgef_relerr(synth(C, 1), U) < 1e-11
            end
        end
        λs, φs = [p[1] for p in pts], [p[2] for p in pts]
        Q = complex.((6 .* sin.(φs) .* cos.(φs) .^ 2 .- 4 .* sin.(φs)) .* cos.(2 .* λs), -4 .* cos.(2 .* φs) .* sin.(2 .* λs))
        C2 = analyze(Q, 2)
        Test.@test only_degree(C2, 3)
        Test.@test _cgef_relerr(synth(C2, 2), Q) < 1e-11
    end

    ws = D.ΠWorkspace(g; transform = CGEF.SpectralBackends.DirectSumSpectralBackend())
    sz = FG.Grids.size_tuple(g)
    pg = _cgef_points(g)
    for (n, _, fλ, fφ) in _CGEF_HARMONICS, kern in (CGEF.GaussianKernel(), CGEF.TopHatKernel())
        u = [fλ(p...) for p in pg]
        v = [fφ(p...) for p in pg]
        D.compute_Π!(zeros(sz), u, v, nothing, g, kern, 3e6; workspace = ws, execution = SER)
        Test.@test _cgef_vec_err(ws.u_filt, ws.v_filt, u, v, _cgef_legendre_transfer(kern, n, 3e6, R)) < 1e-11
    end
end

# `d^ℓ_{mn}(θ)` by its explicit alternating sum in 2048-bit `BigFloat`, rounded to `Float64`.
function _cgef_wigner_big(ℓ, m, n, θ)
    return setprecision(BigFloat, 2048) do
        c, s = cos(big(θ) / 2), sin(big(θ) / 2)
        f(k) = BigFloat(factorial(big(k)))
        pre = sqrt(f(ℓ + m) * f(ℓ - m) * f(ℓ + n) * f(ℓ - n))
        Float64(sum(ifelse(isodd(k + m - n), -1, 1) * pre / (f(ℓ + n - k) * f(k) * f(ℓ - m - k) * f(k + m - n)) *
                    c^(2ℓ + n - m - 2k) * s^(2k + m - n) for k in max(0, n - m):min(ℓ + n, ℓ - m)))
    end
end

# An order `m` whose seed `d^m_{m,n} ≈ sin^m θ` lies below `Float32`'s range, at degrees past the turning
# point `ℓ sin θ = m`, where `d` is of order `ℓ^(−1/2)`: the direct synthesis of one coefficient against
# `N_ℓ d^ℓ_{m,−s}(θ) e^{imφ}` from the explicit sum, in either element type.
Test.@testset "Direct-sum spin harmonics past the floating-point range of their seeds" begin
    F = CGEF.Filtering
    colat = (0.38, 0.45, 0.52, π - 0.45)
    lon = (0.3, 1.1, 2.0, 4.0)
    L, m = 400, 120
    for T in (Float32, Float64)
        g = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(T(6.371e6)), T[λ for λ in lon],
                                      T[π / 2 - θ for θ in colat], fill(T(1e10), 4), trues(4))
        tr = F.DirectSpinTransforms(g; lmax = L)
        θ, φ, _ = F._spin_nodes(g)
        for s in (-1, 2), ℓ in (L - 40, L)
            C = F.spin_coefficients(tr)
            C[ℓ + 1, m + L + 1] = 1
            f = F.spin_synthesize!(F.spin_field(tr), C, C, ones(T, L + 1), tr, s)
            ref = [sqrt((2ℓ + 1) / 4π) * _cgef_wigner_big(ℓ, m, -s, θ[k]) * cis(m * Float64(φ[k])) for k in 1:4]
            Test.@test maximum(abs, f .- ref) < (T === Float32 ? 1e-3 : 1e-10) * maximum(abs, ref)
        end
    end
end

# Under the generalized convolution the Favre budget and the tracer flux filter their scalars at spin 0 by
# the transforms and the `Ĝ(n)` of their vectors: at constant density the budget is `ρ·Π` with no
# baropycnal work, and a uniform tracer carries no flux, both to round-off over the whole sphere.
Test.@testset "Generalized convolution: scalars and vectors through one operator" begin
    D = CGEF.Diagnostics
    SER = CGEF.ComputationalBackends.SerialBackend()
    R = 6.371e6
    cc = first(first(_cgef_harmonic_grids(FG.Geometry.SphericalGeometry(R))))
    sz = FG.Grids.size_tuple(cc)
    pts = _cgef_points(cc)
    u = [cos(p[2]) * sin(2p[1]) + 0.3sin(p[2]) for p in pts]
    v = [cos(3p[1]) * cos(p[2])^2 - 0.2sin(p[1]) for p in pts]
    θ = [sin(2p[2]) + cos(p[1]) * cos(p[2]) for p in pts]
    for kern in (CGEF.GaussianKernel(), CGEF.TopHatKernel()), ℓ in (1.5e6, 3e6)
        Π = D.compute_Π!(zero(u), u, v, nothing, cc, kern, ℓ; execution = SER)
        fav = D.compressible_flux(u, v, fill(2.5, sz), fill(1.0e5, sz), cc, kern, ℓ; execution = SER)
        Test.@test _cgef_relerr(fav.Π, 2.5 .* Π) < 1e-10
        Test.@test maximum(abs, fav.Λ) < 1e-10 * maximum(abs, fav.Π)
        Πθ = D.tracer_variance_flux(u, v, θ, cc, kern, ℓ; execution = SER)
        Πθ0 = D.tracer_variance_flux(u, v, fill(3.0, sz), cc, kern, ℓ; execution = SER)
        Test.@test maximum(abs, Πθ0) < 1e-12 * maximum(abs, Πθ)
    end
end

# The generalized convolution is defined on a sphere under `ZeroFill`: asked for under `Deformable`, or
# on a spheroid, it raises; a workspace holding it refuses a `Deformable` plan.
Test.@testset "Generalized convolution: where it is defined" begin
    D = CGEF.Diagnostics
    R = 6.371e6
    cc = first(first(_cgef_harmonic_grids(FG.Geometry.SphericalGeometry(R))))
    u = [cos(p[2]) for p in _cgef_points(cc)]
    k = CGEF.GaussianKernel()
    GC = CGEF.Filtering.GeneralizedConvolution()
    Test.@test_throws ArgumentError D.compute_Π!(zero(u), u, u, nothing, cc, k, 2e6; mask_strategy = _CGEF_DF, vector_filter = GC)
    pd = CGEF.Filtering.plan_filter(cc, k, 2e6; mask_strategy = _CGEF_DF)
    Test.@test_throws ArgumentError D.compute_Π!(zero(u), u, u, nothing, cc, k, 2e6; workspace = D.ΠWorkspace(cc), filter_plan = pd)
    Test.@test CGEF.Filtering.resolve_vector_filter(cc, _CGEF_DF, nothing) isa CGEF.Filtering.CartesianComponents
    sph = FG.Grids.StructuredGrid(FG.Geometry.SpheroidGeometry(), range(0.0; step = 2π / 16, length = 16), range(-1.0, 1.0; length = 8))
    Test.@test_throws ArgumentError CGEF.Filtering.resolve_vector_filter(sph, CGEF.Filtering.ZeroFill(), GC)
    Test.@test CGEF.Filtering.resolve_vector_filter(sph, CGEF.Filtering.ZeroFill(), nothing) isa CGEF.Filtering.CartesianComponents
end

# A `ZeroFill` sweep's energy sums `|ū|²` over every cell, land included, over the water area (Storer et
# al. 2022), and its Π over land is the flux of the zero-extended field, zero beyond the kernel's reach
# (Aluie et al. 2018). Land values are never read.
Test.@testset "coarse_grain on a masked grid" begin
    N = 24
    xs = 0.0:1.0:(N - 1)
    m = trues(N, N)
    m[8:14, 6:12] .= false
    g = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), xs, xs, m; periodic = (true, true))
    u = [m[i, j] ? sin(2π * 2 * i / N) * cos(2π * 3 * j / N) + 0.3 : NaN for i in 1:N, j in 1:N]
    v = [m[i, j] ? cos(2π * 3 * i / N) * sin(2π * 1 * j / N) - 0.2 : NaN for i in 1:N, j in 1:N]
    k = CGEF.TopHatKernel()
    scales = [2.9, 6.13]
    SER = CGEF.ComputationalBackends.SerialBackend()
    r = CGEF.coarse_grain(u, v, g; scales, kernel = k, execution = SER, spectrum = CGEF.Diagnostics.NoSpectrum())
    for (s, ℓ) in enumerate(scales)
        Test.@test r.cumulative_energy[s] ≈ 0.5 * sum(_cgef_direct_filter(g, k, ℓ, u) .^ 2 .+ _cgef_direct_filter(g, k, ℓ, v) .^ 2) / count(m) rtol = 1e-12
        Test.@test r.Π[:, :, s] == CGEF.Diagnostics.compute_Π!(zeros(N, N), u, v, nothing, g, k, ℓ; execution = SER)
    end
    far = _cgef_beyond_reach(g, CGEF.Kernels.kernel_radius(k, scales[1]) + 1.0)
    Test.@test any(far) && all(iszero, r.Π[:, :, 1][far])
    Test.@test any(!iszero, r.Π[:, :, 1][.!m .& .!far])
end

# A density is guaranteed non-negative only for a kernel whose `|Ĝ|²` is monotone; the default policy
# fills NaN for any other, and `ForceSpectrum` returns the density the cumulative energy implies. Π and
# the cumulative energy do not depend on the policy; a uniform field's cumulative energy is its energy.
Test.@testset "Spectrum policies" begin
    D = CGEF.Diagnostics
    N = 40
    xs = 0.0:1000.0:(N - 1) * 1000.0
    grid = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), xs, xs; periodic = (true, true))
    u = [sin(2π * 3 * i / N) * cos(2π * 2 * j / N) + 0.4sin(2π * 7 * i / N) for i in 1:N, j in 1:N]
    v = [cos(2π * 2 * i / N) * sin(2π * 5 * j / N) + 0.3cos(2π * 6 * j / N) for i in 1:N, j in 1:N]
    scales = collect(4e3:2e3:16e3)
    th, ga = CGEF.TopHatKernel(), CGEF.GaussianKernel()
    ref = D.spectral_density(D.cumulative_energy(u, v, nothing, grid, th, scales), 1.0 ./ scales)
    rF = Logging.with_logger(Logging.NullLogger()) do
        CGEF.coarse_grain(u, v, grid; scales, kernel = th, spectrum = D.ForceSpectrum())
    end
    rN = CGEF.coarse_grain(u, v, grid; scales, kernel = th)
    Test.@test rF.filtering_spectrum == ref
    Test.@test all(isnan, rN.filtering_spectrum)
    Test.@test rF.Π == rN.Π && rF.cumulative_energy == rN.cumulative_energy
    rg = CGEF.coarse_grain(u, v, grid; scales, kernel = ga)
    Test.@test rg.filtering_spectrum == D.filtering_spectrum(u, v, nothing, grid, ga, scales)[2]
    Test.@test all(>=(0), rg.filtering_spectrum)
    Test.@test all(E -> E ≈ 0.5 * (0.5^2 + 0.3^2), D.cumulative_energy(fill(0.5, N, N), fill(0.3, N, N), nothing, grid, th, scales))
end
