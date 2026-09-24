
# Every diagnostic reaches every architecture. The gradient each one needs comes from the grid's own
# operator — a stencil table where there are axes to difference along, a least-squares tangent-plane
# fit where there are not — so a structured grid, a curvilinear mesh, a scattered node set and a
# sphere pixelization all take the same code path through the same tensor algebra.
#
# The assertions here are identities that hold on any grid, so none of them needs a per-architecture
# reference: `L + C + R = τ` (Germano), `Π_α − Π_δ = −S̄:τ̄`, the Helmholtz channels summing to the
# total, the enstrophy flux being the tracer flux of `ω`, and the Favre budget collapsing to `ρ·Π` at
# constant density. That last one holds at every cell where the fluid fills the kernel, which under
# `Deformable` is every cell; under `ZeroFill` the domain exterior holds no fluid.

_dg_relerr(a, b) = maximum(abs, a .- b) / max(maximum(abs, b), eps())
const _DG_DF = CGEF.Filtering.Deformable()

# τ on a Cartesian metric: the components filter as they stand.
function _dg_reference_tau(grid, u, v, plan, ::FG.Geometry.CartesianGeometry)
    ub = similar(u); vb = similar(v)
    uu = similar(u); uv = similar(u); vv = similar(u)
    CGEF.Filtering.filter_apply_batch!(
        (ub, vb, uu, uv, vv), (u, v, u .* u, u .* v, v .* v), plan,
    )
    return (uu .- ub .* ub, uv .- ub .* vb, vv .- vb .* vb)
end

# τ on a sphere: a local (east, north) pair filtered component-wise is not a filtered vector, the
# basis turning from point to point (Aluie 2019). The moments are formed in planetary Cartesian and
# the resulting 3×3 tensor rotated back to the local frame.
function _dg_reference_tau(grid, u, v, plan, geo::FG.Geometry.AbstractSphericalGeometry)
    gsz = FG.Grids.size_tuple(grid)
    sym = ((1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3))
    p = ntuple(_ -> zeros(gsz), 3)
    @inbounds for I in CartesianIndices(p[1])
        i = Tuple(I)
        FG.Grids.isactive(grid, i...) || continue
        λ, φ = FG.Grids.coords(grid, i...)
        c = FG.Geometry.vector_to_cartesian(geo, u[I], v[I], λ, φ)
        p[1][I] = c[1]; p[2][I] = c[2]; p[3][I] = c[3]
    end
    pb = ntuple(_ -> zeros(gsz), 3)
    mom = ntuple(_ -> zeros(gsz), 6)
    prods = ntuple(k -> p[sym[k][1]] .* p[sym[k][2]], 6)
    CGEF.Filtering.filter_apply_batch!((pb..., mom...), (p..., prods...), plan)
    τee = zeros(gsz); τen = zeros(gsz); τnn = zeros(gsz)
    @inbounds for I in CartesianIndices(p[1])
        i = Tuple(I)
        FG.Grids.isactive(grid, i...) || continue
        λ, φ = FG.Grids.coords(grid, i...)
        t = ntuple(k -> mom[k][I] - pb[sym[k][1]][I] * pb[sym[k][2]][I], 6)
        τ = FG.Geometry.tensor_to_local(geo, t[1], t[4], t[6], t[2], t[3], t[5], λ, φ)
        τee[I] = τ.λλ; τen[I] = τ.λφ; τnn[I] = τ.φφ
    end
    return (τee, τen, τnn)
end

# The Germano split reassembles the stress the filter defines, on any metric.
function _dg_check_tau(grid, u, v, ker, ℓ)
    d = CGEF.Diagnostics.tau_decomposition(u, v, grid, ker, ℓ)
    plan = CGEF.Filtering.plan_filter(grid, ker, ℓ;
        backend = CGEF.ComputationalBackends.SerialBackend())
    τxx, τxy, τyy = _dg_reference_tau(grid, u, v, plan, FG.Grids.grid_geometry(grid))
    Test.@test _dg_relerr(d.L.xx .+ d.C.xx .+ d.R.xx, τxx) < 1e-10
    Test.@test _dg_relerr(d.L.xy .+ d.C.xy .+ d.R.xy, τxy) < 1e-10
    Test.@test _dg_relerr(d.L.yy .+ d.C.yy .+ d.R.yy, τyy) < 1e-10
end

# The enstrophy flux is the tracer flux of the vorticity, bit for bit, so the two can never drift
# into different conventions on a new architecture.
function _dg_check_enstrophy(grid, u, v, ker, ℓ)
    ω = CGEF.Diagnostics.vorticity(u, v, grid)
    Test.@test all(isfinite, ω)
    Z = CGEF.Diagnostics.enstrophy_flux(u, v, grid, ker, ℓ)
    Test.@test Z == CGEF.Diagnostics.tracer_variance_flux(u, v, ω, grid, ker, ℓ)
end


# A workspace exists so a repeated evaluation — over scales, or over timesteps — costs no field-sized
# allocation. Each of these is measured through a top-level, fully-qualified helper: inside a testset
# the arguments and the module alias are captured locals, and `@allocated` charges that capture to the
# call under test. The grids below are sized so one field is well over the bound, so a buffer that
# escaped the workspace cannot hide under it.
_dg_a_tau(ws, u, v, g, k, l, p) =
    @allocated CGEF.Diagnostics.tau_decomposition!(ws, u, v, g, k, l; filter_plan = p)
_dg_a_tau3(ws, u, v, w, g, k, l, p) =
    @allocated CGEF.Diagnostics.tau_decomposition!(ws, u, v, w, g, k, l; filter_plan = p)
_dg_a_sc(ws, u, v, g, k, l, p, d) =
    @allocated CGEF.Diagnostics.compute_Π_strain_convergence!(
        ws, u, v, g, k, l; filter_plan = p, deriv_plan = d)
_dg_a_dec(ws, u, v, ur, vr, g, k, l, p, d) =
    @allocated CGEF.Diagnostics.compute_Π_decomposed!(
        ws, u, v, ur, vr, g, k, l; filter_plan = p, deriv_plan = d)
_dg_a_dec3(ws, u, v, w, ur, vr, wr, g, k, l, p, d) =
    @allocated CGEF.Diagnostics.compute_Π_decomposed!(
        ws, u, v, w, ur, vr, wr, g, k, l; filter_plan = p, deriv_plan = d)
_dg_a_fav(ws, u, v, r, P, g, k, l, p, d) =
    @allocated CGEF.Diagnostics.compressible_flux!(
        ws, u, v, r, P, g, k, l; filter_plan = p, deriv_plan = d)
_dg_a_fav3(ws, u, v, w, r, P, g, k, l, p, d) =
    @allocated CGEF.Diagnostics.compressible_flux!(
        ws, u, v, w, r, P, g, k, l; filter_plan = p, deriv_plan = d)
_dg_a_trc(o, ws, u, v, t, g, k, l, p, d) =
    @allocated CGEF.Diagnostics.tracer_variance_flux!(
        o, ws, u, v, t, g, k, l; filter_plan = p, deriv_plan = d)
_dg_a_trc3(o, ws, u, v, w, t, g, k, l, p, d) =
    @allocated CGEF.Diagnostics.tracer_variance_flux!(
        o, ws, u, v, w, t, g, k, l; filter_plan = p, deriv_plan = d)
_dg_a_ens(o, ws, u, v, g, k, l, p, d) =
    @allocated CGEF.Diagnostics.enstrophy_flux!(
        o, ws, u, v, g, k, l; filter_plan = p, deriv_plan = d)

Test.@testset "Workspace forms allocate nothing per call" begin
    ker = CGEF.GaussianKernel()
    SER = CGEF.ComputationalBackends.SerialBackend()
    BOUND = 4096          # the returned NamedTuple and nothing field-sized

    Test.@testset "spherical tangent" begin
        nλ, nφ = 48, 32   # one field is 12,288 B, well over the bound
        sgeo = FG.Geometry.SphericalGeometry()
        lon = range(0.0; step = 2π / nλ, length = nλ)
        lat = range(deg2rad(-60.0); stop = deg2rad(60.0), length = nφ)
        grid = FG.Grids.StructuredGrid(sgeo, lon, lat, trues(nλ, nφ))
        ℓ = 0.2 * FG.Geometry.radius(sgeo)
        u = [cos(p) * sin(l) for l in lon, p in lat]
        v = [cos(p) * cos(l) for l in lon, p in lat]
        θ = [sin(2l) * cos(p) for l in lon, p in lat]
        ρ = fill(2.5, nλ, nφ); P = fill(1.0e5, nλ, nφ)
        pl = CGEF.Filtering.plan_filter(grid, ker, ℓ; backend = SER)
        dp = CGEF.Derivatives.StencilPlan(grid)
        out = zeros(nλ, nφ)

        wt = CGEF.Diagnostics.Sym3TauWorkspace(grid)
        _dg_a_tau(wt, u, v, grid, ker, ℓ, pl)
        Test.@test _dg_a_tau(wt, u, v, grid, ker, ℓ, pl) < BOUND

        wsc = CGEF.Diagnostics.ΠWorkspace(grid)
        _dg_a_sc(wsc, u, v, grid, ker, ℓ, pl, dp)
        Test.@test _dg_a_sc(wsc, u, v, grid, ker, ℓ, pl, dp) < BOUND

        wd = CGEF.Diagnostics.SphericalPiDecomposedWorkspace(grid)
        _dg_a_dec(wd, u, v, u, v, grid, ker, ℓ, pl, dp)
        Test.@test _dg_a_dec(wd, u, v, u, v, grid, ker, ℓ, pl, dp) < BOUND

        wf = CGEF.Diagnostics.SphericalFavreWorkspace(grid)
        _dg_a_fav(wf, u, v, ρ, P, grid, ker, ℓ, pl, dp)
        Test.@test _dg_a_fav(wf, u, v, ρ, P, grid, ker, ℓ, pl, dp) < BOUND

        wtr = CGEF.Diagnostics.SphericalTracerFluxWorkspace(grid)
        _dg_a_trc(out, wtr, u, v, θ, grid, ker, ℓ, pl, dp)
        Test.@test _dg_a_trc(out, wtr, u, v, θ, grid, ker, ℓ, pl, dp) < BOUND

        we = CGEF.Diagnostics.EnstrophyFluxWorkspace(grid)
        _dg_a_ens(out, we, u, v, grid, ker, ℓ, pl, dp)
        Test.@test _dg_a_ens(out, we, u, v, grid, ker, ℓ, pl, dp) < BOUND
    end

    Test.@testset "Cartesian volume" begin
        n = 12            # one field is 13,824 B
        dx = 1000.0; ℓ = 3000.0
        geom = FG.Geometry.CartesianGeometry()
        x = range(0.0, dx * (n - 1); length = n)
        grid = FG.Grids.StructuredGrid(geom, x, x, x, trues(n, n, n))
        u = [sin(i / 4) * cos(j / 5) for i in 1:n, j in 1:n, k in 1:n]
        v = [cos(i / 3) * sin(k / 6) for i in 1:n, j in 1:n, k in 1:n]
        w = [sin(j / 5) * cos(k / 4) for i in 1:n, j in 1:n, k in 1:n]
        θ = [sin(i / 6) + cos(k / 5) for i in 1:n, j in 1:n, k in 1:n]
        ρ = fill(2.5, n, n, n); P = fill(1.0e5, n, n, n)
        pl = CGEF.Filtering.plan_filter(grid, ker, ℓ; backend = SER)
        dp = CGEF.Derivatives.StencilPlan(grid)
        out = zeros(n, n, n)

        wt = CGEF.Diagnostics.Sym3TauWorkspace(grid)
        _dg_a_tau3(wt, u, v, w, grid, ker, ℓ, pl)
        Test.@test _dg_a_tau3(wt, u, v, w, grid, ker, ℓ, pl) < BOUND

        wf = CGEF.Diagnostics.Favre3DWorkspace(grid)
        _dg_a_fav3(wf, u, v, w, ρ, P, grid, ker, ℓ, pl, dp)
        Test.@test _dg_a_fav3(wf, u, v, w, ρ, P, grid, ker, ℓ, pl, dp) < BOUND

        wtr = CGEF.Diagnostics.TracerFlux3DWorkspace(grid)
        _dg_a_trc3(out, wtr, u, v, w, θ, grid, ker, ℓ, pl, dp)
        Test.@test _dg_a_trc3(out, wtr, u, v, w, θ, grid, ker, ℓ, pl, dp) < BOUND
    end

    Test.@testset "spherical shell" begin
        R = 6.371e6
        sgeo = FG.Geometry.SphericalGeometry(R)
        lon = deg2rad.(collect(0.0:15.0:345.0))
        lat = deg2rad.(collect(-60.0:15.0:60.0))
        rad = collect(R:100e3:(R + 200e3))
        grid = FG.Grids.StructuredGrid(sgeo, lon, lat, rad,
                                       trues(length(lon), length(lat), length(rad)))
        ℓ = 8.0e5
        u = [cos(p) * sin(l) for l in lon, p in lat, q in rad]
        v = [cos(p) * cos(l) for l in lon, p in lat, q in rad]
        w = [0.1 * sin(2l) * cos(p) for l in lon, p in lat, q in rad]
        pl = CGEF.Filtering.plan_filter(grid, ker, ℓ; backend = SER)
        dp = CGEF.Derivatives.StencilPlan(grid)

        wd = CGEF.Diagnostics.SphericalPiDecomposed3DWorkspace(grid)
        _dg_a_dec3(wd, u, v, w, u, v, w, grid, ker, ℓ, pl, dp)
        Test.@test _dg_a_dec3(wd, u, v, w, u, v, w, grid, ker, ℓ, pl, dp) < BOUND
    end
end


Test.@testset "Diagnostics across grid architectures: Cartesian" begin
    geom = FG.Geometry.CartesianGeometry()
    dx = 1000.0
    n = 24
    xr = range(0.0, dx * (n - 1); length = n)
    X = [x for x in xr, y in xr]
    Y = [y for x in xr, y in xr]
    u2 = [sin(x / 5000) * cos(y / 6000) for x in xr, y in xr]
    v2 = [cos(x / 4000) * sin(y / 7000) for x in xr, y in xr]
    npt = n * n
    ker = CGEF.GaussianKernel()
    ℓ = 4000.0
    SER = CGEF.ComputationalBackends.SerialBackend()

    cases = Any[
        ("StructuredGrid", FG.Grids.StructuredGrid(geom, xr, xr, trues(n, n)), u2, v2),
        ("CurvilinearGrid", FG.Grids.CurvilinearGrid(geom, X, Y, trues(n, n)), u2, v2),
        # `k` gives the node set its adjacency; without one the least-squares gradient has nothing to
        # fit, which `Derivatives.gradient_plan` refuses (asserted below).
        ("UnstructuredGrid",
         FG.Grids.UnstructuredGrid(geom, vec(X), vec(Y), trues(npt); k = 8),
         vec(u2), vec(v2)),
    ]

    for (name, grid, u, v) in cases
        Test.@testset "$name" begin
            _dg_check_tau(grid, u, v, ker, ℓ)
            _dg_check_enstrophy(grid, u, v, ker, ℓ)

            Π = zeros(size(u))
            CGEF.Diagnostics.compute_Π!(Π, u, v, nothing, grid, ker, ℓ; backend = SER)

            # Π_α − Π_δ expands to the direct −S̄:τ̄, so the two contract different combinations of
            # the same derivatives and disagree under a sign or ordering error.
            sc = CGEF.Diagnostics.compute_Π_strain_convergence(u, v, grid, ker, ℓ)
            Test.@test _dg_relerr(sc.total, Π) < 1e-10

            # Calling the whole field rotational leaves the divergent and cross channels empty.
            dec = CGEF.Diagnostics.compute_Π_decomposed(u, v, u, v, grid, ker, ℓ)
            Test.@test maximum(abs, dec.divergent) == 0
            Test.@test maximum(abs, dec.cross) == 0
            Test.@test _dg_relerr(dec.total, Π) < 1e-10

            # An arbitrary split changes the channels and leaves the total alone.
            dec2 = CGEF.Diagnostics.compute_Π_decomposed(u, v, 0.6 .* u, 0.6 .* v, grid, ker, ℓ)
            Test.@test _dg_relerr(dec2.rotational .+ dec2.cross .+ dec2.divergent, dec2.total) < 1e-12
            Test.@test _dg_relerr(dec2.total, Π) < 1e-10
            Test.@test maximum(abs, dec2.divergent) > 0

            # Constant density collapses the Favre budget: ũ = ū, τ̃ = τ, so Π_Favre = ρ·Π.
            ρ0 = 2.5
            ΠD = zeros(size(u))
            CGEF.Diagnostics.compute_Π!(ΠD, u, v, nothing, grid, ker, ℓ; backend = SER, mask_strategy = _DG_DF)
            fav = CGEF.Diagnostics.compressible_flux(
                u, v, fill(ρ0, size(u)), fill(1.0e5, size(u)), grid, ker, ℓ; mask_strategy = _DG_DF,
            )
            Test.@test _dg_relerr(fav.Π, ρ0 .* ΠD) < 1e-10
            # Uniform pressure carries no pressure gradient, so baropycnal work vanishes to the
            # round-off of the derivative operator.
            Test.@test maximum(abs, fav.Λ) < 1e-12 * maximum(abs, fav.Π)
        end
    end
end


# The true-3-D forms, on both metrics. The gates are the same identities the tangent forms use, so
# neither needs a hand-written reference: the Germano split rebuilds the stress, and the Favre budget
# collapses to `ρ·Π` at constant density — which on a shell it reaches only if its stress rotation and
# its curvature-carrying strain both match what `compute_Π!` builds.
Test.@testset "Diagnostics in true 3D" begin
    ker = CGEF.GaussianKernel()
    SER = CGEF.ComputationalBackends.SerialBackend()
    sym = ((1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3))
    names = (:xx, :xy, :xz, :yy, :yz, :zz)

    Test.@testset "Cartesian volume" begin
        n = 14; dx = 1000.0; ℓ = 4000.0
        geom = FG.Geometry.CartesianGeometry()
        x = range(0.0, dx * (n - 1); length = n)
        grid = FG.Grids.StructuredGrid(geom, x, x, x, trues(n, n, n))
        u = [sin(i / 4) * cos(j / 5) for i in 1:n, j in 1:n, k in 1:n]
        v = [cos(i / 3) * sin(k / 6) for i in 1:n, j in 1:n, k in 1:n]
        w = [sin(j / 5) * cos(k / 4) for i in 1:n, j in 1:n, k in 1:n]
        θ = [sin(i / 6) + cos(k / 5) for i in 1:n, j in 1:n, k in 1:n]

        # L + C + R rebuilds the stress the filter defines, componentwise.
        d = CGEF.Diagnostics.tau_decomposition(u, v, w, grid, ker, ℓ)
        plan = CGEF.Filtering.plan_filter(grid, ker, ℓ; backend = SER)
        vel = (u, v, w)
        b = ntuple(_ -> zeros(n, n, n), 3)
        m = ntuple(_ -> zeros(n, n, n), 6)
        CGEF.Filtering.filter_apply_batch!(
            (b..., m...),
            (vel..., ntuple(k -> vel[sym[k][1]] .* vel[sym[k][2]], 6)...), plan,
        )
        for k in 1:6
            τk = m[k] .- b[sym[k][1]] .* b[sym[k][2]]
            s = getproperty(d.L, names[k]) .+ getproperty(d.C, names[k]) .+ getproperty(d.R, names[k])
            Test.@test _dg_relerr(s, τk) < 1e-10
        end

        ΠD = zeros(n, n, n)
        CGEF.Diagnostics.compute_Π!(ΠD, u, v, w, grid, ker, ℓ; backend = SER, mask_strategy = _DG_DF)
        ρ0 = 2.5
        fav = CGEF.Diagnostics.compressible_flux(
            u, v, w, fill(ρ0, n, n, n), fill(1.0e5, n, n, n), grid, ker, ℓ; mask_strategy = _DG_DF,
        )
        Test.@test _dg_relerr(fav.Π, ρ0 .* ΠD) < 1e-10
        Test.@test maximum(abs, fav.Λ) < 1e-12 * maximum(abs, fav.Π)

        tout = zeros(n, n, n)
        CGEF.Diagnostics.tracer_variance_flux!(
            tout, CGEF.Diagnostics.TracerFlux3DWorkspace(grid), u, v, w, θ, grid, ker, ℓ,
        )
        Test.@test tout == CGEF.Diagnostics.tracer_variance_flux(u, v, w, θ, grid, ker, ℓ)
    end

    Test.@testset "Spherical shell" begin
        R = 6.371e6
        sgeo = FG.Geometry.SphericalGeometry(R)
        lon = deg2rad.(collect(0.0:15.0:345.0))
        lat = deg2rad.(collect(-60.0:15.0:60.0))
        rad = collect(R:100e3:(R + 300e3))
        gsz = (length(lon), length(lat), length(rad))
        grid = FG.Grids.StructuredGrid(sgeo, lon, lat, rad, trues(gsz...))
        ℓ = 8.0e5
        u = [cos(p) * sin(l) for l in lon, p in lat, q in rad]
        v = [cos(p) * cos(l) for l in lon, p in lat, q in rad]
        w = [0.1 * sin(2l) * cos(p) for l in lon, p in lat, q in rad]
        θ = [sin(2l) * cos(p) for l in lon, p in lat, q in rad]

        # `compute_Π!` builds the same local stress by an independently written path.
        ws = CGEF.Diagnostics.ΠWorkspace(grid; has_w = true)
        Π = zeros(gsz)
        CGEF.Diagnostics.compute_Π!(Π, u, v, w, grid, ker, ℓ; workspace = ws, backend = SER)
        d = CGEF.Diagnostics.tau_decomposition(u, v, w, grid, ker, ℓ)
        for (nm, ref) in ((:xx, ws.τ_xx), (:xy, ws.τ_xy), (:xz, ws.τ_xz),
                          (:yy, ws.τ_yy), (:yz, ws.τ_yz), (:zz, ws.τ_zz))
            s = getproperty(d.L, nm) .+ getproperty(d.C, nm) .+ getproperty(d.R, nm)
            Test.@test _dg_relerr(s, ref) < 1e-9
        end

        ρ0 = 2.5
        ΠD = zeros(gsz)
        CGEF.Diagnostics.compute_Π!(ΠD, u, v, w, grid, ker, ℓ; backend = SER, mask_strategy = _DG_DF)
        fav = CGEF.Diagnostics.compressible_flux(
            u, v, w, fill(ρ0, gsz), fill(1.0e5, gsz), grid, ker, ℓ; mask_strategy = _DG_DF,
        )
        Test.@test _dg_relerr(fav.Π, ρ0 .* ΠD) < 1e-9
        Test.@test maximum(abs, fav.Λ) < 1e-12 * maximum(abs, fav.Π)

        tout = zeros(gsz)
        CGEF.Diagnostics.tracer_variance_flux!(
            tout, CGEF.Diagnostics.TracerFlux3DWorkspace(grid), u, v, w, θ, grid, ker, ℓ,
        )
        Test.@test tout == CGEF.Diagnostics.tracer_variance_flux(u, v, w, θ, grid, ker, ℓ)

        # Helmholtz channels on the shell: naming the whole field rotational empties the other two,
        # and an arbitrary split leaves the total alone.
        dec = CGEF.Diagnostics.compute_Π_decomposed(u, v, w, u, v, w, grid, ker, ℓ)
        Test.@test maximum(abs, dec.divergent) == 0
        Test.@test maximum(abs, dec.cross) == 0
        Test.@test _dg_relerr(dec.total, Π) < 1e-10
        dec2 = CGEF.Diagnostics.compute_Π_decomposed(
            u, v, w, 0.6 .* u, 0.6 .* v, 0.6 .* w, grid, ker, ℓ,
        )
        Test.@test _dg_relerr(dec2.rotational .+ dec2.cross .+ dec2.divergent, dec2.total) < 1e-12
        Test.@test _dg_relerr(dec2.total, Π) < 1e-9
        Test.@test maximum(abs, dec2.divergent) > 0
    end
end


# A node set built with no adjacency fits nothing, so every least-squares derivative on it is zero and
# every flux contracted against that strain is zero too — the same numbers a genuinely quiescent flow
# gives. Building the plan raises.
Test.@testset "A grid with no adjacency is refused" begin
    geom = FG.Geometry.CartesianGeometry()
    dx = 1000.0
    n = 6
    npt = n * n
    xs = vec([Float64(i - 1) * dx for i in 1:n, j in 1:n])
    ys = vec([Float64(j - 1) * dx for i in 1:n, j in 1:n])
    bare = FG.Grids.UnstructuredGrid(geom, xs, ys, fill(dx^2, npt), trues(npt))
    Test.@test all(i -> isempty(FG.Grids.neighbors(bare, i)), 1:npt)
    Test.@test_throws ArgumentError CGEF.Derivatives.gradient_plan(bare)
    Test.@test_throws ArgumentError CGEF.Diagnostics.compute_Π!(
        zeros(npt), ones(npt), ones(npt), nothing, bare, CGEF.GaussianKernel(), 4000.0,
    )

    # With an adjacency the same points fit a linear field exactly.
    good = FG.Grids.UnstructuredGrid(geom, xs, ys, trues(npt); k = 8)
    gx = zeros(npt); gy = zeros(npt)
    FG.Operators.gradient!(gx, gy, 2.0 .* xs .+ 3.0 .* ys, CGEF.Derivatives.gradient_plan(good))
    Test.@test maximum(abs, gx .- 2.0) < 1e-8
    Test.@test maximum(abs, gy .- 3.0) < 1e-8
end


Test.@testset "Diagnostics across grid architectures: spherical" begin
    sgeo = FG.Geometry.SphericalGeometry()
    ker = CGEF.GaussianKernel()
    lon = range(0.0; step = 2π / 32, length = 32)
    lat = range(deg2rad(-70.0); stop = deg2rad(70.0), length = 20)

    grids = Any[
        ("StructuredGrid", FG.Grids.StructuredGrid(sgeo, lon, lat, trues(32, 20))),
        ("HEALPixGrid", FG.Grids.HEALPixGrid(sgeo, 4)),
        ("CubedSphereGrid", FG.Grids.CubedSphereGrid(sgeo, 6)),
        ("IcosahedralGrid", FG.Grids.IcosahedralGrid(sgeo, 4)),
        ("RingGrid", FG.Grids.RingGrid(
            sgeo, FG.SphericalSampling.ReducedGaussianSampling([16, 24, 32, 32, 24, 16]))),
    ]

    for (name, grid) in grids
        Test.@testset "$name" begin
            gsz = FG.Grids.size_tuple(grid)
            n = length(FG.Grids.mask(grid))
            R = FG.Geometry.radius(FG.Grids.grid_geometry(grid))
            ℓ = 1.3 * R * sqrt(4π / n)
            u = zeros(gsz); v = zeros(gsz)
            for I in CartesianIndices(u)
                λ, φ = FG.Grids.coords(grid, Tuple(I)...)
                u[I] = cos(φ) * sin(λ)
                v[I] = cos(φ) * cos(λ)
            end

            _dg_check_tau(grid, u, v, ker, ℓ)
            _dg_check_enstrophy(grid, u, v, ker, ℓ)

            Π = zeros(gsz)
            CGEF.Diagnostics.compute_Π!(Π, u, v, nothing, grid, ker, ℓ;
                backend = CGEF.ComputationalBackends.SerialBackend())

            # The strain/convergence split is a statement about the filtered strain tensor, so it
            # holds on the sphere once that tensor is the spherical one. Both sides read the same
            # `S̄` and `τ̄`, so this gates the curvature terms and the planetary stress together.
            sc = CGEF.Diagnostics.compute_Π_strain_convergence(u, v, grid, ker, ℓ)
            Test.@test _dg_relerr(sc.total, Π) < 1e-10
            Test.@test all(>=(0), sc.strain_magnitude)

            # Helmholtz channels: naming the whole field rotational empties the other two.
            dec = CGEF.Diagnostics.compute_Π_decomposed(u, v, u, v, grid, ker, ℓ)
            Test.@test maximum(abs, dec.divergent) == 0
            Test.@test maximum(abs, dec.cross) == 0
            Test.@test _dg_relerr(dec.total, Π) < 1e-10
            dec2 = CGEF.Diagnostics.compute_Π_decomposed(u, v, 0.6 .* u, 0.6 .* v, grid, ker, ℓ)
            Test.@test _dg_relerr(dec2.rotational .+ dec2.cross .+ dec2.divergent, dec2.total) < 1e-12
            Test.@test _dg_relerr(dec2.total, Π) < 1e-10
            Test.@test maximum(abs, dec2.divergent) > 0

            # Constant density collapses the Favre budget to ρ·Π, which the spherical path reaches
            # only if its stress rotation and its curvature-carrying strain both match `compute_Π!`.
            ρ0 = 2.5
            ΠD = zeros(gsz)
            CGEF.Diagnostics.compute_Π!(ΠD, u, v, nothing, grid, ker, ℓ;
                backend = CGEF.ComputationalBackends.SerialBackend(), mask_strategy = _DG_DF)
            fav = CGEF.Diagnostics.compressible_flux(
                u, v, fill(ρ0, gsz), fill(1.0e5, gsz), grid, ker, ℓ; mask_strategy = _DG_DF,
            )
            Test.@test _dg_relerr(fav.Π, ρ0 .* ΠD) < 1e-10
            Test.@test maximum(abs, fav.Λ) < 1e-12 * maximum(abs, fav.Π)

            # Each in-place form reproduces its allocating one bit for bit, so a workspace can never
            # take a different path from the one-shot call.
            let θ = sin.(2 .* u) .* v,
                tws = CGEF.Diagnostics.SphericalTracerFluxWorkspace(grid), tout = zeros(gsz),
                ews = CGEF.Diagnostics.EnstrophyFluxWorkspace(grid), Z = zeros(gsz),
                sws = CGEF.Diagnostics.ΠWorkspace(grid),
                dws = CGEF.Diagnostics.SphericalPiDecomposedWorkspace(grid),
                fws = CGEF.Diagnostics.SphericalFavreWorkspace(grid)
                CGEF.Diagnostics.tracer_variance_flux!(tout, tws, u, v, θ, grid, ker, ℓ)
                Test.@test tout == CGEF.Diagnostics.tracer_variance_flux(u, v, θ, grid, ker, ℓ)
                CGEF.Diagnostics.enstrophy_flux!(Z, ews, u, v, grid, ker, ℓ)
                Test.@test Z == CGEF.Diagnostics.enstrophy_flux(u, v, grid, ker, ℓ)
                Test.@test CGEF.Diagnostics.compute_Π_strain_convergence!(
                    sws, u, v, grid, ker, ℓ).total == sc.total
                Test.@test CGEF.Diagnostics.compute_Π_decomposed!(
                    dws, u, v, u, v, grid, ker, ℓ).total == dec.total
                Test.@test CGEF.Diagnostics.compressible_flux!(
                    fws, u, v, fill(ρ0, gsz), fill(1.0e5, gsz), grid, ker, ℓ;
                    mask_strategy = _DG_DF).Π == fav.Π
            end

            # The rotation into planetary Cartesian depends on the inputs and the grid, never on the
            # scale, so a sweep hoists it out of the scale loop. Every 2-D spherical architecture
            # reaches that, named by the count of directions it resolves.
            let ws = CGEF.Diagnostics.ΠWorkspace(grid),
                plan = CGEF.Filtering.plan_filter(grid, ker, ℓ;
                    backend = CGEF.ComputationalBackends.SerialBackend())
                Test.@test CGEF.Diagnostics.analyze_sweep(u, v, nothing, grid, ws, plan) isa
                           CGEF.Diagnostics.SphericalAnalysis
            end
        end
    end

    # The spherical curl carries the metric's own curvature term. Solid-body rotation about the polar
    # axis, `u = Ω R cosφ`, `v = 0`, has relative vorticity exactly `2Ω sinφ`. The curvature-free
    # `∂v/∂x − ∂u/∂y` gives `Ω sinφ` — half of it, an error the latitude spacing cannot explain.
    let grid = FG.Grids.StructuredGrid(sgeo, lon, lat, trues(32, 20)),
        R = FG.Geometry.radius(sgeo), Ω = 1e-5,
        φs = [FG.Grids.coords(grid, i, j)[2] for i in 1:32, j in 1:20],
        u = Ω .* R .* cos.(φs), v = zeros(32, 20),
        ω = CGEF.Diagnostics.vorticity(u, v, grid),
        ii = 2:31, jj = 2:19,
        want = 2Ω .* sin.(φs[ii, jj])
        # The residual is the φ-derivative's second-order truncation, ≈(Δφ)²/6 at this spacing.
        Test.@test maximum(abs, ω[ii, jj] .- want) < 1e-2 * maximum(abs, want)

        flat = zeros(32, 20); tmp = zeros(32, 20)
        CGEF.Derivatives.ddx!(flat, v, grid)
        CGEF.Derivatives.ddy!(tmp, u, grid)
        flat .-= tmp
        Test.@test maximum(abs, flat[ii, jj] .- want) > 0.4 * maximum(abs, want)
    end
end
