
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
    res = CGEF.coarse_grain(u, v, grid; scales=[10000.0], kernel=CGEF.TopHatKernel())
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
    sres = CGEF.coarse_grain(su, sv, sgrid; scales=[50000.0], kernel=CGEF.TopHatKernel())
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
