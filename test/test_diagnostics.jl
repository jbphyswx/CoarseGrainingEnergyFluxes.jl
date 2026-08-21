
# SFS Stresses and cross-scale energy transfer (Π) calculations
Test.@testset "Diagnostics & Pipeline" begin
    # 2D rigid-body rotation u = -Ωy, v = Ωx has zero kinetic energy transfer (Π = 0)
    geom = FG.Geometry.CartesianGeometry()
    x = collect(-20000.0:2000.0:20000.0) # 21 points
    y = collect(-20000.0:2000.0:20000.0) # 21 points
    mask = trues(21, 21)
    grid = FG.Grids.StructuredGrid(geom, x, y, mask)

    Ω = 1e-4 # Coriolis frequency-like rotation rate
    u = zeros(21, 21)
    v = zeros(21, 21)
    for j in 1:21, i in 1:21
        u[i, j] = -Ω * grid.y[j]
        v[i, j] = Ω * grid.x[i]
    end

    Π = zeros(21, 21)
    CGEF.Diagnostics.compute_Π!(Π, u, v, nothing, grid, CGEF.TopHatKernel(), 10000.0)

    # Kinetic energy transfer must be zero (rigid body rotation is pure laminar cascade-free flow)
    Test.@test Π[11, 11] ≈ 0.0 atol=1e-12

    # Test Pipeline integration with unicode Π
    res = CGEF.coarse_grain(u, v, grid; scales=[10000.0], kernel=CGEF.TopHatKernel(), spectrum = CGEF.Diagnostics.NoSpectrum())
    Test.@test res.Π[:, :, 1] ≈ Π

    # Test Spherical projections and coarse graining with mixed types
    sgeom = FG.Geometry.SphericalGeometry(6371000.0)
    slon = collect(0.0:2.0:10.0)
    slat = collect(0.0:2.0:10.0)
    smask = trues(length(slon), length(slat))
    sgrid = FG.Grids.StructuredGrid(sgeom, deg2rad.(slon), deg2rad.(slat), smask)

    # Mixed-type support: the default return is a geometry-named NamedTuple, and asking for a
    # storage type gives that type back with the promoted element type.
    proj = FG.Geometry.vector_to_cartesian(sgeom, Float32(1.0), Float32(2.0), 0.1, 0.2, 0.3)
    Test.@test proj isa NamedTuple{(:x, :y, :z)}
    Test.@test FG.Geometry.vector_to_cartesian(
        SA.SVector{3,Float64}, sgeom, Float32(1.0), Float32(2.0), 0.1, 0.2, 0.3,
    ) isa SA.SVector{3, Float64}

    inv_proj = FG.Geometry.vector_from_cartesian(sgeom, Float32(1.0), 2.0, 3.0, 0.1, 0.2)
    Test.@test inv_proj isa NamedTuple{(:λ, :φ, :r)}
    Test.@test FG.Geometry.vector_from_cartesian(
        SA.SVector{3,Float64}, sgeom, Float32(1.0), 2.0, 3.0, 0.1, 0.2,
    ) isa SA.SVector{3, Float64}

    # Test coarse_grain on sphere with Float32 inputs (matching PythonCall runtime environment)
    su = fill(Float32(1.0), length(slon), length(slat))
    sv = fill(Float32(0.5), length(slon), length(slat))
    sres = CGEF.coarse_grain(su, sv, sgrid; scales=[50000.0], kernel=CGEF.GaussianKernel())
    Test.@test !any(isnan, @view sres.Π[:, :, 1])
    Test.@test !any(isnan, sres.cumulative_energy)
    Test.@test !any(isnan, sres.filtering_spectrum)
end


# Test Taylor-Green vortex for strain rate verification
Test.@testset "Taylor-Green Vortex" begin
    # Taylor-Green vortex has known analytical solutions
    # u = sin(x)cos(y), v = -cos(x)sin(y)
    # Strain rates and vorticity have exact analytical forms

    geom = FG.Geometry.CartesianGeometry()  # 0.1 unit grid spacing
    xs = collect(0.0:0.1:2π)
    ys = collect(0.0:0.1:2π)
    mask = trues(length(xs), length(ys))
    grid = FG.Grids.StructuredGrid(geom, xs, ys, mask)

    u = [sin(x) * cos(y) for x in xs, y in ys]
    v = [-cos(x) * sin(y) for x in xs, y in ys]

    # Compute derivatives
    dudx = zeros(length(xs), length(ys))
    dudy = zeros(length(xs), length(ys))
    dvdx = zeros(length(xs), length(ys))
    dvdy = zeros(length(xs), length(ys))

    CGEF.Derivatives.ddx!(dudx, u, grid)
    CGEF.Derivatives.ddy!(dudy, u, grid)
    CGEF.Derivatives.ddx!(dvdx, v, grid)
    CGEF.Derivatives.ddy!(dvdy, v, grid)

    # Check a point away from boundaries
    i, j = 10, 10
    x, y = xs[i], ys[j]

    # Analytical: ∂u/∂x = cos(x)cos(y)
    Test.@test dudx[i, j] ≈ cos(x) * cos(y) rtol=0.01

    # Analytical: ∂u/∂y = -sin(x)sin(y)
    Test.@test dudy[i, j] ≈ -sin(x) * sin(y) rtol=0.01

    # Analytical: ∂v/∂x = sin(x)sin(y)
    Test.@test dvdx[i, j] ≈ sin(x) * sin(y) rtol=0.01

    # Analytical: ∂v/∂y = -cos(x)cos(y)
    Test.@test dvdy[i, j] ≈ -cos(x) * cos(y) rtol=0.01
end


# Mathematical correctness: Rigid body rotation must have exactly Π = 0
Test.@testset "Rigid Body Rotation - Zero Energy Flux" begin
    # Rigid body rotation has no deformation, so no energy cascade
    # u = -Ωy, v = Ωx should give Π = 0 everywhere

    geom = FG.Geometry.CartesianGeometry()
    xs = collect(-50e3:1000.0:50e3)  # 101 points
    ys = collect(-50e3:1000.0:50e3)  # 101 points
    mask = trues(length(xs), length(ys))
    grid = FG.Grids.StructuredGrid(geom, xs, ys, mask)

    Ω = 1e-4  # rotation rate
    u = [-Ω * y for x in xs, y in ys]
    v = [Ω * x for x in xs, y in ys]

    # Test at multiple scales
    for scale in [5000.0, 10000.0, 20000.0]
        Π = zeros(length(xs), length(ys))
        CGEF.Diagnostics.compute_Π!(Π, u, v, nothing, grid, CGEF.TopHatKernel(), scale)

        # Check interior points (away from boundaries)
        for j in 40:60, i in 40:60
            Test.@test abs(Π[i, j]) < 1e-10  # Should be exactly zero
        end
    end
end


# Mathematical correctness: Strain rate properties
Test.@testset "Strain Rate Tensor Properties" begin
    # Strain rate tensor S_ij must be symmetric: S_ij = S_ji
    # For 2D incompressible flow: S_xx + S_yy = 0 (trace = divergence)

    geom = FG.Geometry.CartesianGeometry()
    xs = collect(0.0:1000.0:50e3)
    ys = collect(0.0:1000.0:50e3)
    mask = trues(length(xs), length(ys))
    grid = FG.Grids.StructuredGrid(geom, xs, ys, mask)

    # Create a divergent flow field
    u = [0.01 * x for x in xs, y in ys]  # Linear in x
    v = [0.01 * y for x in xs, y in ys]  # Linear in y

    # Filter the field
    u_filt = zeros(length(xs), length(ys))
    v_filt = zeros(length(xs), length(ys))
    CGEF.Filtering.filter_field!(u_filt, u, grid, CGEF.TopHatKernel(), 10000.0)
    CGEF.Filtering.filter_field!(v_filt, v, grid, CGEF.TopHatKernel(), 10000.0)

    # Compute strain rates
    S_xx = zeros(length(xs), length(ys))
    S_yy = zeros(length(xs), length(ys))
    S_xy = zeros(length(xs), length(ys))
    scratch = zeros(length(xs), length(ys))

    CGEF.Derivatives.ddx!(S_xx, u_filt, grid)
    CGEF.Derivatives.ddy!(S_yy, v_filt, grid)
    CGEF.Derivatives.ddy!(S_xy, u_filt, grid)
    CGEF.Derivatives.ddx!(scratch, v_filt, grid)
    @. S_xy = 0.5 * (S_xy + scratch)

    # Test symmetry: S_xy should equal S_yx (we only computed S_xy)
    # Test trace = divergence for incompressible flow
    #
    # This should be essentially EXACT, not just "approximately 0.02": a TopHat filter is a
    # symmetric (uniform-disk) average, which reproduces a linear field exactly at any interior
    # point with a full, untruncated footprint (odd-order terms cancel by symmetry) — and the
    # centered finite difference (`nonuniform_first_derivative`) has zero truncation error for a
    # linear function too (its leading error term involves the third derivative, which is zero
    # here). `i,j in 20:end-20` keeps a comfortable margin inside the domain relative to the
    # 5000 m footprint radius (10000/2) on a 1000 m grid (5 cells), so every point checked has a
    # full, untruncated footprint. A 50%-of-expected-value tolerance would hide any real bug in
    # either the filter or the derivative; float64 roundoff alone is ~1e-12 relative here.
    for j in 20:length(ys)-20, i in 20:length(xs)-20
        divergence = S_xx[i,j] + S_yy[i,j]
        Test.@test divergence ≈ 0.02 atol=1e-9
    end
end


# Mathematical correctness: SFS stress properties
Test.@testset "SFS Stress Tensor Properties" begin
    # τ_ij = [u_i*u_j]̄ - ū_i*ū_j must be symmetric: τ_ij = τ_ji
    # For isotropic turbulence, trace of τ should be positive (energy in SFS)

    geom = FG.Geometry.CartesianGeometry()
    x = collect(0.0:1000.0:30e3)
    y = collect(0.0:1000.0:30e3)
    mask = trues(length(x), length(y))
    grid = FG.Grids.StructuredGrid(geom, x, y, mask)

    # Create random velocity field
    u = rand(length(x), length(y))
    v = rand(length(x), length(y))

    # Filter fields
    u_filt = zeros(length(x), length(y))
    v_filt = zeros(length(x), length(y))
    CGEF.Filtering.filter_field!(u_filt, u, grid, CGEF.TopHatKernel(), 5000.0)
    CGEF.Filtering.filter_field!(v_filt, v, grid, CGEF.TopHatKernel(), 5000.0)

    # Filter products
    uu = zeros(length(x), length(y))
    uv = zeros(length(x), length(y))
    vv = zeros(length(x), length(y))
    scratch = zeros(length(x), length(y))

    @. scratch = u * u
    CGEF.Filtering.filter_field!(uu, scratch, grid, CGEF.TopHatKernel(), 5000.0)
    @. scratch = u * v
    CGEF.Filtering.filter_field!(uv, scratch, grid, CGEF.TopHatKernel(), 5000.0)
    @. scratch = v * v
    CGEF.Filtering.filter_field!(vv, scratch, grid, CGEF.TopHatKernel(), 5000.0)

    # Compute SFS stress
    τ_xx = zeros(length(x), length(y))
    τ_xy = zeros(length(x), length(y))
    τ_yy = zeros(length(x), length(y))

    @. τ_xx = uu - u_filt * u_filt
    @. τ_xy = uv - u_filt * v_filt
    @. τ_yy = vv - v_filt * v_filt

    # Test that trace of τ is positive (physical constraint for filtering)
    # τ_xx + τ_yy = [u²+v²]̄ - (ū² + v̄²) ≥ 0 by Jensen's inequality
    for j in 10:length(y)-10, i in 10:length(x)-10
        trace_τ = τ_xx[i,j] + τ_yy[i,j]
        Test.@test trace_τ >= -1e-10  # Should be non-negative
    end

    # Test symmetry: compute τ_yx and verify equals τ_xy
    scratch2 = zeros(length(x), length(y))
    @. scratch2 = v * u
    CGEF.Filtering.filter_field!(scratch, scratch2, grid, CGEF.TopHatKernel(), 5000.0)
    @. scratch2 = scratch - v_filt * u_filt  # τ_yx

    for j in 10:length(y)-10, i in 10:length(x)-10
        Test.@test τ_xy[i,j] ≈ scratch2[i,j] rtol=1e-10
    end
end


# Mathematical correctness: Π sign consistency with SFS stress and strain
Test.@testset "Energy Flux Sign Consistency" begin
    # Π = -ρ₀ * S̄_ij * τ_ij should have consistent sign based on S and τ
    # For a convergent strain with positive SFS stress, Π should be negative
    # (energy goes from resolved to sub-grid = forward cascade)

    geom = FG.Geometry.CartesianGeometry()
    xs = collect(0.0:1000.0:50e3)
    ys = collect(0.0:1000.0:50e3)
    mask = trues(length(xs), length(ys))
    grid = FG.Grids.StructuredGrid(geom, xs, ys, mask)

    # Create a simple deformation field
    # u = a*x, v = -a*y gives pure strain (convergence in x, divergence in y)
    a = 0.001
    u = [a * x for x in xs, y in ys]
    v = [-a * y for x in xs, y in ys]

    # The strain rate tensor for this field:
    # S_xx = a, S_yy = -a, S_xy = 0

    # At scale where filtering matters, we can verify:
    # - SFS stress τ should be computed correctly
    # - The sign of Π should match the sign of -S:τ

    Π = zeros(length(xs), length(ys))
    CGEF.Diagnostics.compute_Π!(Π, u, v, nothing, grid, CGEF.TopHatKernel(), 10000.0)

    # For this pure linear deformation, filtering doesn't change the field
    # (linear fields are invariant under top-hat filtering)
    # So τ should be ~0 and Π should be ~0
    for j in 20:length(ys)-20, i in 20:length(xs)-20
        Test.@test abs(Π[i,j]) < 1e-8
    end
end


# Mathematical correctness: Energy budget closure
Test.@testset "Energy Budget - Filtered vs Unfiltered" begin
    # Test that: 0.5*ρ₀*|u|² = 0.5*ρ₀*|ū|² + 0.5*ρ₀*trace(τ) + (boundary terms)
    # For periodic domains, the resolved + SFS energies should relate to total energy

    geom = FG.Geometry.CartesianGeometry()
    x = collect(0.0:1000.0:30e3)
    y = collect(0.0:1000.0:30e3)
    mask = trues(length(x), length(y))
    grid = FG.Grids.StructuredGrid(geom, x, y, mask)

    # Random velocity field
    u = rand(length(x), length(y))
    v = rand(length(x), length(y))

    ρ₀ = 1025.0

    # Compute total energy
    total_energy = 0.5 * ρ₀ * (u.^2 + v.^2)

    # Filter at some scale
    scale = 5000.0
    u_filt = zeros(length(x), length(y))
    v_filt = zeros(length(x), length(y))
    CGEF.Filtering.filter_field!(u_filt, u, grid, CGEF.TopHatKernel(), scale)
    CGEF.Filtering.filter_field!(v_filt, v, grid, CGEF.TopHatKernel(), scale)

    # Filter products for SFS stress trace
    uu_filt = zeros(length(x), length(y))
    vv_filt = zeros(length(x), length(y))
    scratch = zeros(length(x), length(y))

    @. scratch = u * u
    CGEF.Filtering.filter_field!(uu_filt, scratch, grid, CGEF.TopHatKernel(), scale)
    @. scratch = v * v
    CGEF.Filtering.filter_field!(vv_filt, scratch, grid, CGEF.TopHatKernel(), scale)

    # SFS energy = 0.5*ρ₀*trace(τ) = 0.5*ρ₀*([u²]̄ + [v²]̄ - ū² - v̄²)
    sfs_energy = zeros(length(x), length(y))
    @. sfs_energy = 0.5 * ρ₀ * (uu_filt + vv_filt - u_filt^2 - v_filt^2)

    # Resolved energy
    resolved_energy = zeros(length(x), length(y))
    @. resolved_energy = 0.5 * ρ₀ * (u_filt^2 + v_filt^2)

    # Verify: sfs_energy ≥ 0 (Jensen's inequality)
    for j in 10:length(y)-10, i in 10:length(x)-10
        Test.@test sfs_energy[i,j] >= -1e-12  # Should be non-negative
    end

    # Verify: sfs_energy + resolved_energy ≈ filtered total energy
    # ([u²]̄ + [v²]̄)/2 = ([u²+v²]̄)/2
    filtered_total = zeros(length(x), length(y))
    @. scratch = u.^2 + v.^2
    CGEF.Filtering.filter_field!(filtered_total, scratch, grid, CGEF.TopHatKernel(), scale)
    @. filtered_total = 0.5 * ρ₀ * filtered_total

    for j in 10:length(y)-10, i in 10:length(x)-10
        energy_sum = sfs_energy[i,j] + resolved_energy[i,j]
        Test.@test energy_sum ≈ filtered_total[i,j] rtol=1e-10
    end
end


# Germano subfilter-stress decomposition: L + C + R == τ (exact closure)
Test.@testset "Stress decomposition (Leonard/Cross/Reynolds)" begin
    geom = FG.Geometry.CartesianGeometry()
    x = collect(0.0:1000.0:30e3)
    y = collect(0.0:1000.0:30e3)
    grid = FG.Grids.StructuredGrid(geom, x, y, trues(length(x), length(y)))
    u = rand(length(x), length(y))
    v = rand(length(x), length(y))
    kern = CGEF.TopHatKernel()
    scale = 5000.0

    d = CGEF.Diagnostics.tau_decomposition(u, v, grid, kern, scale)

    # The workspace form is the same implementation the allocating one delegates to, so it must agree
    # exactly; with the workspace AND the plan held it also stops allocating per call.
    let ws = CGEF.Diagnostics.TauWorkspace(grid),
        pl = CGEF.Filtering.plan_filter(grid, kern, scale;
            backend = CGEF.ComputationalBackends.SerialBackend())
        dw = CGEF.Diagnostics.tau_decomposition!(ws, u, v, grid, kern, scale; filter_plan = pl)
        for blk in (:L, :C, :R), comp in (:xx, :xy, :yy)
            Test.@test getproperty(getproperty(dw, blk), comp) ==
                       getproperty(getproperty(d, blk), comp)
        end
        # Reusing the workspace must give the same answer, not accumulate into stale buffers.
        dw2 = CGEF.Diagnostics.tau_decomposition!(ws, u, v, grid, kern, scale; filter_plan = pl)
        Test.@test dw2.C.xy == dw.C.xy
        # Per-call cost is the returned NamedTuple only — no field-sized allocation.
        Test.@test (@allocated CGEF.Diagnostics.tau_decomposition!(
            ws, u, v, grid, kern, scale; filter_plan = pl)) < 4096
    end

    # Reference τ_ij = filter(u_i u_j) - ū_i ū_j with the same filter.
    ub = zeros(size(u)); vb = zeros(size(v))
    CGEF.Filtering.filter_field!(ub, u, grid, kern, scale)
    CGEF.Filtering.filter_field!(vb, v, grid, kern, scale)
    uu = zeros(size(u)); uv = zeros(size(u)); vv = zeros(size(u))
    CGEF.Filtering.filter_field!(uu, u .* u, grid, kern, scale)
    CGEF.Filtering.filter_field!(uv, u .* v, grid, kern, scale)
    CGEF.Filtering.filter_field!(vv, v .* v, grid, kern, scale)
    τxx = uu .- ub .^ 2
    τxy = uv .- ub .* vb
    τyy = vv .- vb .^ 2

    Test.@test d.L.xx .+ d.C.xx .+ d.R.xx ≈ τxx
    Test.@test d.L.xy .+ d.C.xy .+ d.R.xy ≈ τxy
    Test.@test d.L.yy .+ d.C.yy .+ d.R.yy ≈ τyy
    # Reynolds (subfilter–subfilter) stress trace is non-negative (Jensen).
    Test.@test all(d.R.xx .+ d.R.yy .>= -1e-10)
end


# Rotational/divergent (Helmholtz) decomposition of the energy flux. Uses genuine
# streamfunction/potential test fields (exactly non-divergent / exactly irrotational by
# elementary vector calculus, for ANY kx, ky) rather than an arbitrary scalar split of one
# field — that older test passed even under the (buggy) one-sided-strain implementation,
# since it degenerated to S̄ᵈ ≡ 0 and never exercised the cross-strain terms.
Test.@testset "Helmholtz flux decomposition" begin
    geom = FG.Geometry.CartesianGeometry()
    x = collect(0.0:1000.0:30e3); y = collect(0.0:1000.0:30e3)
    grid = FG.Grids.StructuredGrid(geom, x, y, trues(length(x), length(y)))
    kern = CGEF.TopHatKernel(); scale = 5000.0

    Lx = x[end] - x[1]; Ly = y[end] - y[1]
    kx = 2π / Lx; ky = 2π / Ly

    # Taylor-Green vortex from streamfunction ψ = cos(kx x) cos(ky y): u = -∂ψ/∂y, v = ∂ψ/∂x
    # is exactly non-divergent (∂u/∂x + ∂v/∂y ≡ 0) by construction.
    u_rot = [ky * cos(kx * xi) * sin(ky * yj) for xi in x, yj in y]
    v_rot = [-kx * sin(kx * xi) * cos(ky * yj) for xi in x, yj in y]

    # Sine-sine potential χ = sin(kx x) sin(ky y): u = ∂χ/∂x, v = ∂χ/∂y is exactly
    # irrotational (∂v/∂x - ∂u/∂y ≡ 0) by construction.
    u_div = [kx * cos(kx * xi) * sin(ky * yj) for xi in x, yj in y]
    v_div = [ky * sin(kx * xi) * cos(ky * yj) for xi in x, yj in y]

    u = u_rot .+ u_div
    v = v_rot .+ v_div

    dec = CGEF.Diagnostics.compute_Π_decomposed(u, v, u_rot, v_rot, grid, kern, scale)
    Πfull = zeros(size(u)); CGEF.Diagnostics.compute_Π!(Πfull, u, v, nothing, grid, kern, scale)

    # The workspace form is the implementation the allocating one delegates to, so it agrees exactly;
    # holding the workspace and both plans makes a repeated evaluation allocation-free.
    let ws = CGEF.Diagnostics.PiDecomposedWorkspace(grid),
        pl = CGEF.Filtering.plan_filter(grid, kern, scale;
            backend = CGEF.ComputationalBackends.SerialBackend()),
        dp = CGEF.Derivatives.StencilPlan(grid)
        dw = CGEF.Diagnostics.compute_Π_decomposed!(ws, u, v, u_rot, v_rot, grid, kern, scale;
            filter_plan = pl, deriv_plan = dp)
        for k in (:total, :rotational, :cross, :divergent)
            Test.@test getproperty(dw, k) == getproperty(dec, k)
        end
        # Reuse must recompute from scratch, not accumulate into the previous call's buffers.
        dw2 = CGEF.Diagnostics.compute_Π_decomposed!(ws, u, v, u_rot, v_rot, grid, kern, scale;
            filter_plan = pl, deriv_plan = dp)
        Test.@test dw2.total == dw.total
        Test.@test (@allocated CGEF.Diagnostics.compute_Π_decomposed!(
            ws, u, v, u_rot, v_rot, grid, kern, scale;
            filter_plan = pl, deriv_plan = dp)) < 4096
    end

    # (1) channels sum EXACTLY to the total, matching the standard full-flux computation.
    Test.@test dec.total ≈ dec.rotational .+ dec.cross .+ dec.divergent
    Test.@test dec.total ≈ Πfull

    # (2) purely-rotational field ⇒ divergent/cross channels vanish, rotational = full flux
    # computed on that field alone.
    Πr_full = zeros(size(u_rot)); CGEF.Diagnostics.compute_Π!(Πr_full, u_rot, v_rot, nothing, grid, kern, scale)
    dec_r = CGEF.Diagnostics.compute_Π_decomposed(u_rot, v_rot, u_rot, v_rot, grid, kern, scale)
    Test.@test maximum(abs, dec_r.divergent) < 1e-10
    Test.@test maximum(abs, dec_r.cross) < 1e-10
    Test.@test dec_r.rotational ≈ Πr_full

    # (3) purely-divergent field ⇒ rotational/cross channels vanish, divergent = full flux
    # computed on that field alone.
    Πd_full = zeros(size(u_div)); CGEF.Diagnostics.compute_Π!(Πd_full, u_div, v_div, nothing, grid, kern, scale)
    dec_d = CGEF.Diagnostics.compute_Π_decomposed(u_div, v_div, zeros(size(u_div)), zeros(size(v_div)), grid, kern, scale)
    Test.@test maximum(abs, dec_d.rotational) < 1e-10
    Test.@test maximum(abs, dec_d.cross) < 1e-10
    Test.@test dec_d.divergent ≈ Πd_full

    # (4) each channel must contract against its OWN strain component, not the full undecomposed S̄.
    # Contracting against the full strain is correct only where S̄ᵈ ≡ 0; that formula is built here
    # from primitives, and Π_RR must differ from it, since S̄ᵈ is nonzero for this mixed field.
    ub = zeros(size(u)); vb = zeros(size(v))
    CGEF.Filtering.filter_field!(ub, u, grid, kern, scale)
    CGEF.Filtering.filter_field!(vb, v, grid, kern, scale)
    Sxx_full = zeros(size(u)); CGEF.Derivatives.ddx!(Sxx_full, ub, grid)
    Syy_full = zeros(size(u)); CGEF.Derivatives.ddy!(Syy_full, vb, grid)
    p = zeros(size(u)); q = zeros(size(u))
    CGEF.Derivatives.ddy!(p, ub, grid); CGEF.Derivatives.ddx!(q, vb, grid)
    Sxy_full = 0.5 .* (p .+ q)

    urb = zeros(size(u_rot)); vrb = zeros(size(v_rot))
    CGEF.Filtering.filter_field!(urb, u_rot, grid, kern, scale)
    CGEF.Filtering.filter_field!(vrb, v_rot, grid, kern, scale)
    uu = zeros(size(u)); uv = zeros(size(u)); vv = zeros(size(u))
    CGEF.Filtering.filter_field!(uu, u_rot .* u_rot, grid, kern, scale)
    CGEF.Filtering.filter_field!(uv, u_rot .* v_rot, grid, kern, scale)
    CGEF.Filtering.filter_field!(vv, v_rot .* v_rot, grid, kern, scale)
    τRRxx = uu .- urb .^ 2; τRRxy = uv .- urb .* vrb; τRRyy = vv .- vrb .^ 2

    old_Πrr = -(Sxx_full .* τRRxx .+ 2 .* Sxy_full .* τRRxy .+ Syy_full .* τRRyy)
    Test.@test !isapprox(dec.rotational, old_Πrr)
end


# Cross-scale tracer-variance flux (scalar analog of Π).
Test.@testset "Tracer variance flux" begin
    geom = FG.Geometry.CartesianGeometry()
    x = collect(0.0:1000.0:30e3); y = collect(0.0:1000.0:30e3)
    grid = FG.Grids.StructuredGrid(geom, x, y, trues(length(x), length(y)))
    u = rand(length(x), length(y)) .- 0.5
    v = rand(length(x), length(y)) .- 0.5
    θ = rand(length(x), length(y))
    kern = CGEF.TopHatKernel(); scale = 5000.0

    # (1) constant tracer ⇒ zero gradient ⇒ zero flux.
    Πc = CGEF.Diagnostics.tracer_variance_flux(u, v, fill(2.5, size(θ)), grid, kern, scale)
    Test.@test maximum(abs, Πc) < 1e-9

    # (2) matches the explicit definition Πθ = -(τx ∂x θ̄ + τy ∂y θ̄) built from primitives.
    Πθ = CGEF.Diagnostics.tracer_variance_flux(u, v, θ, grid, kern, scale)

    # The workspace form is the implementation the allocating one delegates to, so it agrees exactly;
    # with the workspace and both plans held it stops allocating per call.
    let ws = CGEF.Diagnostics.TracerFluxWorkspace(grid),
        pl = CGEF.Filtering.plan_filter(grid, kern, scale;
            backend = CGEF.ComputationalBackends.SerialBackend()),
        dp = CGEF.Derivatives.StencilPlan(grid),
        got = zeros(size(Πθ))
        CGEF.Diagnostics.tracer_variance_flux!(got, ws, u, v, θ, grid, kern, scale;
            filter_plan = pl, deriv_plan = dp)
        Test.@test got == Πθ
        # Reusing the workspace must recompute, not accumulate into stale buffers.
        CGEF.Diagnostics.tracer_variance_flux!(got, ws, u, v, θ, grid, kern, scale;
            filter_plan = pl, deriv_plan = dp)
        Test.@test got == Πθ
        Test.@test (@allocated CGEF.Diagnostics.tracer_variance_flux!(
            got, ws, u, v, θ, grid, kern, scale; filter_plan = pl, deriv_plan = dp)) < 4096
    end
    ub = zeros(size(u)); vb = zeros(size(v)); θb = zeros(size(θ))
    CGEF.Filtering.filter_field!(ub, u, grid, kern, scale)
    CGEF.Filtering.filter_field!(vb, v, grid, kern, scale)
    CGEF.Filtering.filter_field!(θb, θ, grid, kern, scale)
    uθ = zeros(size(u)); vθ = zeros(size(u))
    CGEF.Filtering.filter_field!(uθ, u .* θ, grid, kern, scale)
    CGEF.Filtering.filter_field!(vθ, v .* θ, grid, kern, scale)
    τx = uθ .- ub .* θb; τy = vθ .- vb .* θb
    gx = zeros(size(θ)); gy = zeros(size(θ))
    CGEF.Derivatives.ddx!(gx, θb, grid); CGEF.Derivatives.ddy!(gy, θb, grid)
    ref = .-(τx .* gx .+ τy .* gy)
    Test.@test Πθ ≈ ref
end


# Spherical tracer-variance flux: the flux τ_j = ⟨u_j θ⟩ - ū_j θ̄ is a VECTOR, so it follows the same
# rotate-to-planetary-Cartesian / filter / rotate-back convention as compute_Π! rather than filtering
# the local east/north components in place.
Test.@testset "Tracer variance flux: spherical" begin
    R = 6.371e6
    sgeom = FG.Geometry.SphericalGeometry(R)
    lon = deg2rad.(collect(0.0:5.0:355.0))
    lat = deg2rad.(collect(-40.0:5.0:40.0))
    grid = FG.Grids.StructuredGrid(sgeom, lon, lat, trues(length(lon), length(lat)))
    nx, ny = length(lon), length(lat)
    u = rand(nx, ny) .- 0.5
    v = rand(nx, ny) .- 0.5
    θ = rand(nx, ny)
    kern = CGEF.TopHatKernel(); scale = deg2rad(15.0) * R

    # A constant tracer has no gradient, so the flux vanishes whatever the velocity does.
    Πc = CGEF.Diagnostics.tracer_variance_flux(u, v, fill(2.5, nx, ny), grid, kern, scale)
    Test.@test maximum(abs, Πc) < 1e-9

    # Against the definition, built from primitives with the rotation done explicitly.
    Πθ = CGEF.Diagnostics.tracer_variance_flux(u, v, θ, grid, kern, scale)
    ux = zeros(nx, ny); uy = zeros(nx, ny); uz = zeros(nx, ny)
    for j in 1:ny, i in 1:nx
        λ, φ = FG.Grids.coords(grid, i, j)
        p = FG.Geometry.vector_to_cartesian(sgeom, u[i, j], v[i, j], λ, φ)
        ux[i, j] = p[1]; uy[i, j] = p[2]; uz[i, j] = p[3]
    end
    flt(f) = (o = zeros(nx, ny); CGEF.Filtering.filter_field!(o, f, grid, kern, scale); o)
    θb = flt(θ)
    τX = flt(ux .* θ) .- flt(ux) .* θb
    τY = flt(uy .* θ) .- flt(uy) .* θb
    τZ = flt(uz .* θ) .- flt(uz) .* θb
    τe = zeros(nx, ny); τn = zeros(nx, ny)
    for j in 1:ny, i in 1:nx
        λ, φ = FG.Grids.coords(grid, i, j)
        l = FG.Geometry.vector_from_cartesian(sgeom, τX[i, j], τY[i, j], τZ[i, j], λ, φ)
        τe[i, j] = l[1]; τn[i, j] = l[2]
    end
    gx = zeros(nx, ny); gy = zeros(nx, ny)
    CGEF.Derivatives.ddx!(gx, θb, grid); CGEF.Derivatives.ddy!(gy, θb, grid)
    Test.@test Πθ ≈ .-(τe .* gx .+ τn .* gy)

    # The rotation is not cosmetic: filtering the local components in place gives a different answer,
    # so a component-wise implementation would be silently wrong rather than merely less elegant.
    τx_local = flt(u .* θ) .- flt(u) .* θb
    τy_local = flt(v .* θ) .- flt(v) .* θb
    Test.@test !isapprox(Πθ, .-(τx_local .* gx .+ τy_local .* gy); rtol = 1e-6)
end

# ---------------------------------------------------------------------------
# The filtering spectral DENSITY is only guaranteed non-negative for a kernel whose |Ĝ(k)|² is
# monotone decreasing (Sadek & Aluie 2018 eq. 21). A kernel that fails it is refused rather than
# quietly handed back a spectrum that may go negative.
# ---------------------------------------------------------------------------
Test.@testset "filtering spectrum: only kernels with monotone |Ĝ|² are admitted" begin
    K = CGEF.Kernels
    Test.@test K.transfer_monotone(CGEF.GaussianKernel())
    Test.@test K.transfer_monotone(CGEF.GaussianKernel(; α = 4))
    Test.@test K.transfer_monotone(CGEF.SharpSpectralKernel())
    Test.@test !K.transfer_monotone(CGEF.TopHatKernel())
    # No fallback method: a new kernel must state which way it goes rather than inherit a guess.
    Test.@test_throws MethodError K.transfer_monotone(nothing)

    # The top-hat's |Ĝ|² really is non-monotone — the trait is measured, not asserted.
    # (`spectral_transfer` for the top-hat lives in the SpecialFunctions extension, which the suite
    # loads.)
    let ℓ = 1.0, ks = range(0.1, 12.0; length = 4000)
        g2(kern) = [K.spectral_transfer(kern, k, ℓ)^2 for k in ks]
        rises(v) = maximum(diff(v))
        Test.@test rises(g2(CGEF.TopHatKernel())) > 1e-6      # genuinely increases somewhere
        Test.@test rises(g2(CGEF.GaussianKernel())) <= 0      # never increases
        Test.@test rises(g2(CGEF.SharpSpectralKernel())) <= 0
    end

    # HOW non-monotone, in sampling-independent terms — this is what justifies offering `ForceSpectrum`
    # at all, so it is gated rather than asserted in prose. A per-step `diff` scales with the sampling
    # and cannot stand in for it: the climb from the first zero of Ĝ = 2J₁(kR)/(kR) up to the first
    # Airy sidelobe is the physical quantity, and it is under 2% of the DC value.
    let ℓ = 1.0, ks = range(0.1, 12.0; length = 40_000)
        g2 = [K.spectral_transfer(CGEF.TopHatKernel(), k, ℓ)^2 for k in ks]
        izero = argmin(g2)                              # first zero of Ĝ
        ipeak = izero - 1 + argmax(@view g2[izero:end]) # first sidelobe past it
        Test.@test 7.5 < ks[izero] < 7.8                # measured 7.663
        Test.@test 10.1 < ks[ipeak] < 10.4              # measured 10.271
        Test.@test 0.017 < g2[ipeak] < 0.018            # measured 0.01750
        # and it never exceeds the DC value, unlike HighOrderKernel's |Ĝ|², which reaches 1.0124
        Test.@test maximum(g2) <= 1.0
    end

    geom = FG.Geometry.CartesianGeometry()
    N = 40
    xs = 0.0:1000.0:(N - 1) * 1000.0
    grid = FG.Grids.StructuredGrid(geom, xs, xs)
    # A field with real structure across the sweep, so the density is not trivially flat.
    u = [sin(2π * 3 * i / N) * cos(2π * 2 * j / N) + 0.4sin(2π * 7 * i / N) for i in 1:N, j in 1:N]
    v = [cos(2π * 2 * i / N) * sin(2π * 5 * j / N) + 0.3cos(2π * 6 * j / N) for i in 1:N, j in 1:N]
    scales = collect(4e3:2e3:16e3)

    # Refused, with a message that names the condition and the way out.
    err = try
        CGEF.Diagnostics.filtering_spectrum(u, v, nothing, grid, CGEF.TopHatKernel(), scales)
        nothing
    catch e
        e
    end
    Test.@test err isa ArgumentError
    Test.@test occursin("TopHatKernel", err.msg)
    Test.@test occursin("monotone", err.msg)
    Test.@test occursin("GaussianKernel", err.msg)

    # Admitted, and the density it returns is non-negative — the property the gate exists to protect.
    for kern in (CGEF.GaussianKernel(), CGEF.GaussianKernel(; α = 4))
        kℓ, Ẽ = CGEF.Diagnostics.filtering_spectrum(u, v, nothing, grid, kern, scales)
        Test.@test kℓ ≈ 1.0 ./ scales
        Test.@test all(isfinite, Ẽ)
        Test.@test all(>=(0), Ẽ)
    end

    # The convention is a plain rescaling: k_ℓ = L/ℓ scales the abscissa by L and the density by 1/L,
    # so a comparison against a Fourier spectrum is only meaningful once the two agree on it.
    let kern = CGEF.GaussianKernel(), Lbox = N * 1000.0
        k1, E1 = CGEF.Diagnostics.filtering_spectrum(u, v, nothing, grid, kern, scales)
        kL, EL = CGEF.Diagnostics.filtering_spectrum(u, v, nothing, grid, kern, scales; L = Lbox)
        Test.@test kL ≈ Lbox .* k1
        Test.@test EL ≈ E1 ./ Lbox
    end

    # The pipeline is gated the same way, so the headline entry point cannot bypass it.
    Test.@test_throws ArgumentError CGEF.coarse_grain(u, v, grid; scales = scales)
    Test.@test_throws ArgumentError CGEF.coarse_grain(u, v, grid; scales = scales,
                                                      kernel = CGEF.TopHatKernel())
    let b = CGEF.Pipeline.CoarseGrainBatchResult(grid, length(scales), (2,))
        Test.@test_throws ArgumentError CGEF.Pipeline.coarse_grain_batch!(
            b, randn(N, N, 2), randn(N, N, 2), grid; scales = scales,
        )
        Test.@test_throws ArgumentError CGEF.Pipeline.coarse_grain_slices!(
            [CGEF.Pipeline._allocate_result(grid, length(scales))], [u], [v], [grid]; scales = scales,
        )
    end

    # `NoSpectrum()` is the explicit opt-out: Π and the cumulative energy are computed as usual and
    # the density is left as NaN, which cannot be mistaken for a number that was computed.
    rf = CGEF.coarse_grain(u, v, grid; scales = scales, spectrum = CGEF.Diagnostics.NoSpectrum())
    Test.@test all(isfinite, rf.Π)
    Test.@test all(isfinite, rf.cumulative_energy)
    Test.@test all(isnan, rf.filtering_spectrum)
    # ... and it changes nothing else: Π matches an admissible-kernel-free reference at the same kernel.
    rg = CGEF.coarse_grain(u, v, grid; scales = scales, kernel = CGEF.GaussianKernel())
    Test.@test all(isfinite, rg.filtering_spectrum)
    Test.@test size(rg.Π) == size(rf.Π)
end

# ---------------------------------------------------------------------------
# The monotonicity condition is SUFFICIENT, not necessary, so the strict reading is the default rather
# than the only one. `ForceSpectrum` must produce exactly the density the gate was withholding — not a
# different or regularized one — and must warn without changing any other field.
# ---------------------------------------------------------------------------
Test.@testset "spectrum policy: strict / force / off" begin
    D = CGEF.Diagnostics
    geom = FG.Geometry.CartesianGeometry()
    N = 40
    xs = 0.0:1000.0:(N - 1) * 1000.0
    grid = FG.Grids.StructuredGrid(geom, xs, xs)
    u = [sin(2π * 3 * i / N) * cos(2π * 2 * j / N) + 0.4sin(2π * 7 * i / N) for i in 1:N, j in 1:N]
    v = [cos(2π * 2 * i / N) * sin(2π * 5 * j / N) + 0.3cos(2π * 6 * j / N) for i in 1:N, j in 1:N]
    scales = collect(4e3:2e3:16e3)
    th = CGEF.TopHatKernel()
    ga = CGEF.GaussianKernel()

    # The policy is a singleton-type lattice, the same idiom as AbstractMaskStrategy — not a Symbol and
    # not a Bool, so a typo is a MethodError at the call rather than a silently wrong mode.
    Test.@test D.StrictSpectrum() isa D.AbstractSpectrumPolicy
    Test.@test D.ForceSpectrum() isa D.AbstractSpectrumPolicy
    Test.@test D.NoSpectrum() isa D.AbstractSpectrumPolicy
    Test.@test_throws TypeError CGEF.coarse_grain(u, v, grid; scales = scales, spectrum = false)
    Test.@test_throws TypeError CGEF.coarse_grain(u, v, grid; scales = scales, spectrum = :force)

    # `gate_spectrum` is the whole decision: throw / warn-and-compute / skip.
    Test.@test D.gate_spectrum(ga, D.StrictSpectrum())
    Test.@test_throws ArgumentError D.gate_spectrum(th, D.StrictSpectrum())
    Test.@test !D.gate_spectrum(th, D.NoSpectrum())
    Test.@test !D.gate_spectrum(ga, D.NoSpectrum())
    Test.@test D.gate_spectrum(th, D.ForceSpectrum())

    # The strict error has to name every way forward, or it is the veto the policy exists to remove.
    err = try
        D.filtering_spectrum(u, v, nothing, grid, th, scales)
        nothing
    catch e
        e
    end
    Test.@test err isa ArgumentError
    for recourse in ("ForceSpectrum", "NoSpectrum", "GaussianKernel", "cumulative_energy")
        Test.@test occursin(recourse, err.msg)
    end

    # THE substantive assertion: forcing yields exactly the density the strict gate was withholding.
    # `spectral_density` is ungated, so the reference is computable independently of the policy — if
    # `ForceSpectrum` ever became a different computation, this catches it.
    cum = D.cumulative_energy(u, v, nothing, grid, th, scales)
    kℓ = 1.0 ./ scales
    ref = D.spectral_density(cum, kℓ)
    kf, Ẽf = Test.@test_logs (:warn, r"not monotone decreasing") match_mode = :any D.filtering_spectrum(
        u, v, nothing, grid, th, scales; policy = D.ForceSpectrum())
    Test.@test kf == kℓ
    Test.@test Ẽf == ref
    Test.@test all(isfinite, Ẽf)

    # For this field and scale range the forced density is in fact positive throughout — the concrete
    # case the strict default would have refused outright.
    Test.@test all(>(0), Ẽf)

    # A monotone kernel takes the identical path under `ForceSpectrum` and must NOT warn: the warning
    # is about the kernel, not about having asked to force.
    kg, Ẽg = Test.@test_logs min_level = Logging.Warn D.filtering_spectrum(
        u, v, nothing, grid, ga, scales; policy = D.ForceSpectrum())
    kd, Ẽd = D.filtering_spectrum(u, v, nothing, grid, ga, scales)
    Test.@test kg == kd && Ẽg == Ẽd

    # `NoSpectrum` returns the right shape rather than a short vector, so a caller indexing per scale
    # still works and simply sees NaN.
    kn, Ẽn = D.filtering_spectrum(u, v, nothing, grid, th, scales; policy = D.NoSpectrum())
    Test.@test kn == kℓ
    Test.@test length(Ẽn) == length(scales) && all(isnan, Ẽn)

    # The pipeline agrees with the diagnostic under every policy, and only `filtering_spectrum` moves:
    # Π and the cumulative energy are bit-identical across all three.
    rF = CGEF.coarse_grain(u, v, grid; scales = scales, kernel = th, spectrum = D.ForceSpectrum())
    rN = CGEF.coarse_grain(u, v, grid; scales = scales, kernel = th, spectrum = D.NoSpectrum())
    Test.@test rF.filtering_spectrum == ref
    Test.@test rF.Π == rN.Π
    Test.@test rF.cumulative_energy == rN.cumulative_energy
    Test.@test all(isnan, rN.filtering_spectrum)

    # Every batch entry point honours the policy too — the strict gate is not a property of the
    # single-slice driver, and neither is the escape from it.
    let Nt = 2, ub = repeat(u, 1, 1, Nt), vb = repeat(v, 1, 1, Nt)
        b = CGEF.Pipeline.CoarseGrainBatchResult(grid, length(scales), (Nt,))
        CGEF.Pipeline.coarse_grain_batch!(b, ub, vb, grid; scales = scales, kernel = th,
                                          spectrum = D.ForceSpectrum())
        for t in 1:Nt
            Test.@test b.slices[t].filtering_spectrum == ref
        end
        CGEF.Pipeline.coarse_grain_batch!(b, ub, vb, grid; scales = scales, kernel = th,
                                          spectrum = D.NoSpectrum())
        Test.@test all(isnan, b.filtering_spectrum)

        rs = [CGEF.Pipeline._allocate_result(grid, length(scales)) for _ in 1:Nt]
        CGEF.Pipeline.coarse_grain_slices!(rs, [u, u], [v, v], [grid, grid]; scales = scales,
                                           kernel = th, spectrum = D.ForceSpectrum())
        Test.@test all(r -> r.filtering_spectrum == ref, rs)
    end
end
