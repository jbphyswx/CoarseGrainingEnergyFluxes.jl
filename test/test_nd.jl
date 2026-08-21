
# True n-D Cartesian filtering (1D + 3D) via the general footprint engine.
Test.@testset "n-D filtering (1D + true 3D Cartesian)" begin
    # --- 1D ---
    geom1 = FG.Geometry.CartesianGeometry()
    x = collect(0.0:1.0:50.0)
    grid1 = FG.Grids.StructuredGrid(geom1, x, trues(length(x)))
    Test.@test FG.Grids.size_tuple(grid1) == (length(x),)
    # constant -> constant (normalization)
    o1 = zeros(length(x))
    CGEF.Filtering.filter_field!(o1, fill(7.0, length(x)), grid1, CGEF.TopHatKernel(), 6.0)
    Test.@test all(≈(7.0), o1[10:40])
    # sub-grid scale -> identity (only the self cell is in support)
    g1 = rand(length(x)); oi = zeros(length(x))
    CGEF.Filtering.filter_field!(oi, g1, grid1, CGEF.TopHatKernel(), 0.5)
    Test.@test oi ≈ g1

    # --- 3D (dz ≫ footprint, so only the in-plane disk contributes) ---
    geom3 = FG.Geometry.CartesianGeometry()
    x3 = collect(0.0:1.0:20.0); y3 = collect(0.0:1.0:20.0); z3 = collect(0.0:100.0:300.0)
    nx, ny, nz = length(x3), length(y3), length(z3)
    grid3 = FG.Grids.StructuredGrid(geom3, x3, y3, z3, trues(nx, ny, nz))
    Test.@test FG.Grids.size_tuple(grid3) == (nx, ny, nz)
    # constant -> constant
    o3 = zeros(nx, ny, nz)
    CGEF.Filtering.filter_field!(o3, fill(3.5, nx, ny, nz), grid3, CGEF.TopHatKernel(), 6.0)
    Test.@test all(≈(3.5), o3)

    # A z-invariant 3D field must reduce EXACTLY to the 2D filter of its slice (dz ≫ rad ⇒ no
    # vertical neighbours), validating the n-D engine against the 2D engine.
    f2d = rand(nx, ny)
    f3z = repeat(reshape(f2d, nx, ny, 1), 1, 1, nz)
    o3z = zeros(nx, ny, nz)
    CGEF.Filtering.filter_field!(o3z, f3z, grid3, CGEF.TopHatKernel(), 6.0)
    grid2 = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), x3, y3, trues(nx, ny))
    o2 = zeros(nx, ny)
    CGEF.Filtering.filter_field!(o2, f2d, grid2, CGEF.TopHatKernel(), 6.0)
    for k in 1:nz
        Test.@test o3z[:, :, k] ≈ o2
    end
end


# True 3D Cartesian energy flux Π = -ρ₀ S̄_ij τ_ij (all nine strain components).
Test.@testset "3D Cartesian energy flux" begin
    geom3 = FG.Geometry.CartesianGeometry()
    x = collect(0.0:1.0:24.0); y = collect(0.0:1.0:24.0); z = collect(0.0:50.0:150.0)
    nx, ny, nz = length(x), length(y), length(z)
    grid3 = FG.Grids.StructuredGrid(geom3, x, y, z, trues(nx, ny, nz))
    ker = CGEF.TopHatKernel(); ℓ = 5.0

    # (1) Constant velocity ⇒ zero strain ⇒ Π ≡ 0.
    Πc = zeros(nx, ny, nz)
    CGEF.Diagnostics.compute_Π!(Πc, fill(2.0, nx, ny, nz), fill(-3.0, nx, ny, nz),
                    fill(0.5, nx, ny, nz), grid3, ker, ℓ)
    Test.@test maximum(abs, Πc) < 1e-9

    # (2) z-invariant (u, v) with w = 0: the 3D six-term contraction must collapse EXACTLY to the
    # 2D three-term flux on every layer (Szz = Sxz = Syz = τxz = τyz = τzz = 0), validating the 3D
    # assembly + 3D derivatives against the established 2D path.
    u2 = rand(nx, ny) .- 0.5; v2 = rand(nx, ny) .- 0.5
    u3 = repeat(reshape(u2, nx, ny, 1), 1, 1, nz)
    v3 = repeat(reshape(v2, nx, ny, 1), 1, 1, nz)
    w3 = zeros(nx, ny, nz)
    Π3 = zeros(nx, ny, nz)
    CGEF.Diagnostics.compute_Π!(Π3, u3, v3, w3, grid3, ker, ℓ)

    grid2 = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), x, y, trues(nx, ny))
    Π2 = zeros(nx, ny)
    CGEF.Diagnostics.compute_Π!(Π2, u2, v2, nothing, grid2, ker, ℓ)
    for k in 1:nz
        Test.@test Π3[:, :, k] ≈ Π2
    end
end


# --- True-3D Cartesian pipeline: coarse_grain reuses the existing genuinely-coupled compute_Π!
# (all nine strain components), just wired through the workspace-reusing scale sweep. ---
Test.@testset "True-3D Cartesian coarse_grain pipeline" begin
    geom3 = FG.Geometry.CartesianGeometry()
    grid3 = FG.Grids.StructuredGrid(
        geom3, collect(0.0:1000.0:5000.0), collect(0.0:1000.0:5000.0), collect(0.0:500.0:2000.0),
        trues(6, 6, 5),
    )
    u3 = rand(6, 6, 5); v3 = rand(6, 6, 5); w3 = rand(6, 6, 5)
    res3 = CGEF.coarse_grain(u3, v3, w3, grid3; scales = [2000.0, 3000.0], kernel = CGEF.TopHatKernel(),
                             spectrum = CGEF.Diagnostics.NoSpectrum())
    Test.@test size(res3.Π) == (6, 6, 5, 2)
    Test.@test !any(isnan, res3.cumulative_energy)
    # The top-hat cannot carry a filtering spectral density; the density is checked on the same
    # pipeline with a kernel that can.
    Test.@test !any(isnan, CGEF.coarse_grain(u3, v3, w3, grid3; scales = [2000.0, 3000.0],
                                             kernel = CGEF.GaussianKernel()).filtering_spectrum)

    # Cross-check: coarse_grain! (in-place, reusing a workspace) matches the fresh allocation.
    ws3 = CGEF.Diagnostics.ΠWorkspace(grid3)
    res3b = CGEF.coarse_grain(u3, v3, w3, grid3; scales = [2000.0, 3000.0], kernel = CGEF.TopHatKernel(),
                              spectrum = CGEF.Diagnostics.NoSpectrum())
    CGEF.coarse_grain!(res3b, u3, v3, w3, grid3; scales = [2000.0, 3000.0], kernel = CGEF.TopHatKernel(),
                       workspace = ws3, spectrum = CGEF.Diagnostics.NoSpectrum())
    Test.@test res3b.Π ≈ res3.Π
end


# True-3D SPHERICAL volumetric support: a genuine radius axis (r = absolute distance from planet
# center, not a depth/height offset), real ∂/∂r derivatives, and the full 3x3 planetary-Cartesian
# tensor rotation (as opposed to the 2.5D layer-by-layer path, which has no radial axis at all).
Test.@testset "True-3D spherical volumetric grid + Π" begin
    R = 6.371e6
    geo = FG.Geometry.SphericalGeometry(R)
    lon = deg2rad.(collect(0.0:10.0:350.0))
    lat = deg2rad.(collect(-60.0:10.0:60.0))
    r = collect(R:5000.0:(R + 20000.0))  # 5 levels, 20 km shell
    mask = trues(length(lon), length(lat), length(r))
    grid = FG.Grids.StructuredGrid(geo, lon, lat, r, mask)
    Test.@test FG.Grids.size_tuple(grid) == (length(lon), length(lat), length(r))

    # Volume element is the genuine spherical-shell r²cosφΔλΔφΔr at each level's OWN local
    # radius, not the fixed reference R — so cell volume must grow with height at fixed (i,j).
    Test.@test FG.Grids.area(grid, 1, 7, 5) > FG.Grids.area(grid, 1, 7, 1)
    # Exact ratio check at the equator-ish band (φ index 7 is closest to 0): volumes at two
    # levels and the same (i,j) scale as (r[k]/r[k'])² (cosφ, Δλ, Δφ, Δr all cancel exactly
    # away from the domain's radial boundary, where Δr is uniform anyway on this axis).
    Test.@test FG.Grids.area(grid, 1, 7, 3) / FG.Grids.area(grid, 1, 7, 2) ≈ (r[3] / r[2])^2 rtol=1e-12

    # A single radius level is the 2D/2.5D case, not true-3D — must be rejected, not silently
    # given a wrong (area-not-volume) measure.
    Test.@test_throws ArgumentError FG.Grids.StructuredGrid(geo, lon, lat, [R], trues(length(lon), length(lat), 1))

    # Rigid-body rotation of the whole 3D shell (u_e = Ω·r·cosφ, v_n = w_r = 0) is a pure
    # rotation — zero strain rate everywhere, hence Π ≡ 0 — regardless of the genuine radial
    # shear ∂u_e/∂r = Ω·cosφ ≠ 0 this induces (unlike the 2D invariant, which has no radial
    # shear to get wrong in the first place; this is the check that actually exercises the new
    # S_er curvature-correction term).
    Ω = 7.292e-5
    u = [Ω * r[k] * cos(lat[j]) for _ in lon, j in eachindex(lat), k in eachindex(r)]
    v = zeros(length(lon), length(lat), length(r))
    w = zeros(length(lon), length(lat), length(r))
    Π = zeros(size(u))
    CGEF.Diagnostics.compute_Π!(Π, u, v, w, grid, CGEF.TopHatKernel(), 500e3)
    Test.@test maximum(abs, Π) < 1e-9 * maximum(abs, u)

    # Full pipeline: shape + finiteness.
    res = CGEF.coarse_grain(u, v, w, grid; scales = [300e3, 500e3], kernel = CGEF.TopHatKernel(),
                            spectrum = CGEF.Diagnostics.NoSpectrum())
    Test.@test size(res.Π) == (length(lon), length(lat), length(r), 2)
    Test.@test !any(isnan, res.cumulative_energy)
    Test.@test !any(isnan, CGEF.coarse_grain(u, v, w, grid; scales = [300e3, 500e3],
                                             kernel = CGEF.GaussianKernel()).filtering_spectrum)

    # Volumetric tracer-variance flux: zero for a constant tracer, and equal to the definition
    # assembled from primitives with the planetary-Cartesian rotation done explicitly. On a deep
    # shell, so the radial spacing is comparable to the horizontal one — the 20 km shell above is
    # thinner than a single filter radius, which would smooth every radial gradient to zero and leave
    # the radial term untestable.
    r_deep = collect(R:750e3:(R + 3000e3))
    grid_deep = FG.Grids.StructuredGrid(geo, lon, lat, r_deep, trues(length(lon), length(lat), length(r_deep)))
    sz = (length(lon), length(lat), length(r_deep))
    θ = [sin(2λ) * cos(φ) + 2 * (rk - R) / 3000e3 for λ in lon, φ in lat, rk in r_deep]
    ut = [0.4 * cos(φ) * sin(λ) for λ in lon, φ in lat, _ in r_deep]
    vt = [0.3 * sin(2φ) for λ in lon, φ in lat, _ in r_deep]
    wt = [0.05 * cos(λ) for λ in lon, _ in lat, _ in r_deep]
    # Wide enough that the top-hat spans several cells: at 10° resolution a radius under one cell
    # width makes the filter the identity, `τ` identically zero, and every assertion below vacuous.
    scale3 = 3.0e6
    Test.@test maximum(abs, CGEF.Diagnostics.tracer_variance_flux(
        ut, vt, wt, fill(2.5, sz), grid_deep, CGEF.TopHatKernel(), scale3)) < 1e-9

    Πθ = CGEF.Diagnostics.tracer_variance_flux(ut, vt, wt, θ, grid_deep, CGEF.TopHatKernel(), scale3)
    ux = zeros(sz); uy = zeros(sz); uz = zeros(sz)
    for I in CartesianIndices(θ)
        λ, φ = FG.Grids.coords(grid_deep, Tuple(I)...)
        p = FG.Geometry.vector_to_cartesian(geo, ut[I], vt[I], wt[I], λ, φ)
        ux[I] = p[1]; uy[I] = p[2]; uz[I] = p[3]
    end
    flt3(f) = (o = zeros(sz); CGEF.Filtering.filter_field!(o, f, grid_deep, CGEF.TopHatKernel(), scale3); o)
    θb = flt3(θ)
    τX = flt3(ux .* θ) .- flt3(ux) .* θb
    τY = flt3(uy .* θ) .- flt3(uy) .* θb
    τZ = flt3(uz .* θ) .- flt3(uz) .* θb
    τe = zeros(sz); τn = zeros(sz); τr = zeros(sz)
    for I in CartesianIndices(θ)
        λ, φ = FG.Grids.coords(grid_deep, Tuple(I)...)
        l = FG.Geometry.vector_from_cartesian(geo, τX[I], τY[I], τZ[I], λ, φ)
        τe[I] = l[1]; τn[I] = l[2]; τr[I] = l[3]
    end
    gx = zeros(sz); gy = zeros(sz); gz = zeros(sz)
    CGEF.Derivatives.ddx!(gx, θb, grid_deep)
    CGEF.Derivatives.ddy!(gy, θb, grid_deep)
    CGEF.Derivatives.ddz!(gz, θb, grid_deep)
    # The subfilter flux must actually be resolved, or the comparison below compares two zeros.
    Test.@test maximum(abs, τe) > 1e-4 * maximum(abs, ut) * maximum(abs, θ)
    Test.@test Πθ ≈ .-(τe .* gx .+ τn .* gy .+ τr .* gz)

    # The radial term is genuinely carried: dropping it changes the answer on this shell.
    Test.@test !isapprox(Πθ, .-(τe .* gx .+ τn .* gy); rtol = 1e-6)
end


# True-3D Helmholtz flux decomposition and 3D tracer-variance flux.
Test.@testset "True-3D Helmholtz flux decomposition & tracer flux" begin
    geom3 = FG.Geometry.CartesianGeometry()
    x = collect(0.0:1000.0:10e3); y = collect(0.0:1000.0:10e3); z = collect(0.0:500.0:4e3)
    grid3 = FG.Grids.StructuredGrid(geom3, x, y, z, trues(length(x), length(y), length(z)))
    kern = CGEF.TopHatKernel(); scale = 2500.0

    Lx = x[end] - x[1]; Ly = y[end] - y[1]; Lz = z[end] - z[1]
    kx = 2π * 3 / Lx; ky = 2π * 2 / Ly; kz = 2π * 2 / Lz

    # Non-divergent field via vector potential A=(0,0,ψ): u=∂ψ/∂y, v=-∂ψ/∂x, w=0 is exactly
    # non-divergent for ANY ψ(x,y) (2D streamfunction embedded in 3D, no z-dependence).
    u_rot = [ky * cos(kx * xi) * sin(ky * yj) for xi in x, yj in y, _ in z]
    v_rot = [-kx * sin(kx * xi) * cos(ky * yj) for xi in x, yj in y, _ in z]
    w_rot = zeros(length(x), length(y), length(z))

    # Irrotational field as the gradient of a scalar potential χ: curl(∇χ) ≡ 0 by construction.
    u_div = [kx * cos(kx * xi) * sin(ky * yj) * sin(kz * zk) for xi in x, yj in y, zk in z]
    v_div = [ky * sin(kx * xi) * cos(ky * yj) * sin(kz * zk) for xi in x, yj in y, zk in z]
    w_div = [kz * sin(kx * xi) * sin(ky * yj) * cos(kz * zk) for xi in x, yj in y, zk in z]

    u = u_rot .+ u_div; v = v_rot .+ v_div; w = w_rot .+ w_div

    dec = CGEF.Diagnostics.compute_Π_decomposed(u, v, w, u_rot, v_rot, w_rot, grid3, kern, scale)
    Πfull = zeros(size(u)); CGEF.Diagnostics.compute_Π!(Πfull, u, v, w, grid3, kern, scale)
    Test.@test dec.total ≈ dec.rotational .+ dec.cross .+ dec.divergent
    Test.@test dec.total ≈ Πfull

    Πr_full = zeros(size(u_rot)); CGEF.Diagnostics.compute_Π!(Πr_full, u_rot, v_rot, w_rot, grid3, kern, scale)
    dec_r = CGEF.Diagnostics.compute_Π_decomposed(u_rot, v_rot, w_rot, u_rot, v_rot, w_rot, grid3, kern, scale)
    Test.@test maximum(abs, dec_r.divergent) < 1e-10
    Test.@test maximum(abs, dec_r.cross) < 1e-10
    Test.@test dec_r.rotational ≈ Πr_full

    # 3D tracer-variance flux: constant tracer ⇒ zero gradient ⇒ zero flux.
    θ = rand(length(x), length(y), length(z))
    Πθ = CGEF.Diagnostics.tracer_variance_flux(u, v, w, θ, grid3, kern, scale)
    Test.@test all(isfinite, Πθ)
    Πθ0 = CGEF.Diagnostics.tracer_variance_flux(u, v, w, fill(2.5, size(θ)), grid3, kern, scale)
    Test.@test maximum(abs, Πθ0) < 1e-9
end
