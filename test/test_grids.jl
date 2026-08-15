
# Grids constructor and area calculations
Test.@testset "Grids" begin
    geom = FG.Geometry.CartesianGeometry()
    x = collect(0.0:2000.0:20000.0) # 11 points
    y = collect(0.0:2000.0:10000.0) # 6 points
    mask = trues(11, 6) # fully active grid

    grid = FG.Grids.StructuredGrid(geom, x, y, mask)
    Test.@test FG.Grids.size_tuple(grid) == (11, 6)
    Test.@test FG.Grids.area(grid, 2, 2) == 2000.0 * 2000.0
    Test.@test FG.Grids.coords(grid, 2, 3) == (x = 2000.0, y = 4000.0)
    Test.@test FG.Grids.coords(SA.SVector, grid, 2, 3) == SA.SVector{2,Float64}(2000.0, 4000.0)

    # CurvilinearGrid coords bug test: verify i,j indices are used correctly
    # Create non-square grid to catch index swapping bugs
    lon_m = [Float64(i) for i in 1:10, j in 1:5]  # 10x5
    lat_m = [Float64(j*10) for i in 1:10, j in 1:5]  # y varies with j
    areas_m = ones(10, 5)
    mask_m = trues(10, 5)
    cgrid = FG.Grids.CurvilinearGrid(geom, lon_m, lat_m, areas_m, mask_m)

    # coords(i,j) should return (x[i,j], y[i,j])
    pt = FG.Grids.coords(cgrid, 5, 3)
    Test.@test pt[1] == 5.0  # x[5,3] = 5
    Test.@test pt[2] == 30.0 # y[5,3] = 30

    # This test catches the bug where y[j,j] was used instead of y[i,j]
    pt_corner = FG.Grids.coords(cgrid, 10, 5)
    Test.@test pt_corner[1] == 10.0  # x[10,5]
    Test.@test pt_corner[2] == 50.0 # y[10,5], not y[5,5]=50 vs y[10,10] error
end


# A descending coordinate axis must give the same result as its ascending mirror. Descending axes are
# ordinary — `lat = π/2 .- θ` is the FastSphericalHarmonics recipe, and depth and pressure levels are
# usually stored top-down — but a cell measure taken as the signed `x[i+1]-x[i]` goes negative on one,
# and the real-space engine's `weight_norm > threshold` gate then never passes, so every output falls
# to the "no valid neighbours" zero branch with no error raised.
#
# A measure is a magnitude and must not depend on storage order. The signed gaps `_local_spacing`
# returns are a separate thing: a derivative stencil needs the sign to orient `df/dx`.
Test.@testset "Decreasing coordinate axis: areas stay positive, filtering matches the reversed-axis grid" begin
    geom = FG.Geometry.CartesianGeometry()
    xs_inc = collect(0.0:1.0:19.0)
    xs_dec = reverse(xs_inc)              # same physical points, stored decreasing
    ys_inc = collect(0.0:1.0:14.0)
    ys_dec = reverse(ys_inc)
    mask = trues(length(xs_inc), length(ys_inc))

    grid_inc = FG.Grids.StructuredGrid(geom, xs_inc, ys_inc, mask)
    grid_dec = FG.Grids.StructuredGrid(geom, xs_dec, ys_dec, mask)
    Test.@test all(>(0), FG.Grids.measure(grid_inc))
    Test.@test all(>(0), FG.Grids.measure(grid_dec))
    Test.@test FG.Grids.measure(grid_dec) ≈ reverse(FG.Grids.measure(grid_inc); dims = (1, 2))

    field_inc = [sin(x/5) * cos(y/4) for x in xs_inc, y in ys_inc]
    field_dec = reverse(field_inc; dims = (1, 2))   # same physical field, indexed to match grid_dec
    out_inc = zeros(size(field_inc)); out_dec = zeros(size(field_dec))
    CGEF.Filtering.filter_field!(out_inc, field_inc, grid_inc, CGEF.TopHatKernel(), 4.0)
    CGEF.Filtering.filter_field!(out_dec, field_dec, grid_dec, CGEF.TopHatKernel(), 4.0)
    Test.@test maximum(abs, out_dec) > 1e-6   # a negative measure zeroes this out entirely
    Test.@test out_dec ≈ reverse(out_inc; dims = (1, 2)) atol = 1e-10

    # Spherical: the exact natural-FSH-recipe grid that surfaced this (lat = π/2 - θ, decreasing).
    sgeom = FG.Geometry.SphericalGeometry(1.0)
    θ = collect(range(0.05, π - 0.05; length = 20))
    lon = collect(range(0.0, 2π - 0.1; length = 20))
    lat_dec = π/2 .- θ                     # decreasing, mirrors the FSH recipe exactly
    sgrid = FG.Grids.StructuredGrid(sgeom, lon, lat_dec, trues(20, 20))
    Test.@test all(>(0), FG.Grids.measure(sgrid))
    sfield = [sin(3λ) * cos(2φ) for λ in lon, φ in lat_dec]
    sout = zeros(20, 20)
    CGEF.Filtering.filter_field!(sout, sfield, sgrid, CGEF.TopHatKernel(), 0.3)
    Test.@test maximum(abs, sout) > 1e-6
end


# Nonuniform-axis correctness (not just "doesn't regress on a uniform grid"): the standard
# 3-point nonuniform stencil is EXACT (to floating point) for any quadratic function of its
# input coordinate, on ANY spacing pattern — a stronger and simpler proof than a convergence sweep,
# and one a stencil using a single global Δ cannot pass.
Test.@testset "Nonuniform axes" begin
    # --- Cartesian: geometrically-stretched axis (genuinely nonuniform, no constant step) ---
    geom = FG.Geometry.CartesianGeometry() # dx/dy unused on this path; area comes from the axis
    x_nu = [0.0, 1.0, 2.5, 4.0, 7.0, 12.0, 19.0] # strictly increasing, non-constant gaps
    y_nu = [0.0, 0.5, 1.5, 3.5, 7.5]
    Nx_nu, Ny_nu = length(x_nu), length(y_nu)
    mask_nu = trues(Nx_nu, Ny_nu)
    grid_nu = FG.Grids.StructuredGrid(geom, x_nu, y_nu, mask_nu)

    # `x_nu` is a plain Vector, not a Range -- confirm this actually takes the general
    # (nonuniform-safe) footprint path, not the fast Range-only path.
    fp_nu = CGEF.Filtering.build_footprint(grid_nu, CGEF.TopHatKernel(), 2.0)
    Test.@test fp_nu isa CGEF.Filtering.PrefixSumTopHatPlan
    # ...and a Gaussian takes the SEPARABLE engine, because separability is a property of the kernel
    # rather than of the spacing: a stretched axis makes the per-axis weight depend on position as
    # well as offset, which is a wider weight table, not a different algorithm. The table's rank is
    # what distinguishes the two — a vector when the spacing is constant, a row per position here
    # (the table is position-major so that a row pass reads it along the contiguous axis).
    fp_g = CGEF.Filtering.build_footprint(grid_nu, CGEF.GaussianKernel(), 2.0)
    Test.@test fp_g isa CGEF.Filtering.SeparableGaussianFootprint
    Test.@test fp_g.gx isa AbstractMatrix
    Test.@test size(fp_g.gx, 1) == Nx_nu
    # A kernel with no separable form still takes the general scattered engine, which is what proves
    # the nonuniform axis itself is handled by the general machinery.
    Test.@test CGEF.Filtering.build_footprint(grid_nu, CGEF.SharpSpectralKernel(), 2.0) isa
               CGEF.Filtering.ScatteredFilterPlan

    # f(x,y) = x^2 + y^2: interior centered stencil is exact for a quadratic on any spacing.
    f_quad = [x_nu[i]^2 + y_nu[j]^2 for i in 1:Nx_nu, j in 1:Ny_nu]
    ∂f∂x_nu = zeros(Nx_nu, Ny_nu)
    ∂f∂y_nu = zeros(Nx_nu, Ny_nu)
    CGEF.Derivatives.ddx!(∂f∂x_nu, f_quad, grid_nu)
    CGEF.Derivatives.ddy!(∂f∂y_nu, f_quad, grid_nu)
    for i in 2:(Nx_nu-1), j in 1:Ny_nu
        Test.@test ∂f∂x_nu[i, j] ≈ 2 * x_nu[i] atol=1e-10
    end
    for i in 1:Nx_nu, j in 2:(Ny_nu-1)
        Test.@test ∂f∂y_nu[i, j] ≈ 2 * y_nu[j] atol=1e-10
    end

    # Filtering a constant field must return the constant unchanged even on a nonuniform grid
    # (this exercises the scattered/per-point footprint path end to end).
    field_const = fill(7.5, Nx_nu, Ny_nu)
    out_const = zeros(Nx_nu, Ny_nu)
    CGEF.Filtering.filter_field!(out_const, field_const, grid_nu, CGEF.TopHatKernel(), 3.0)
    Test.@test all(x -> isapprox(x, 7.5; atol=1e-10), out_const)

    # --- Spherical: nonuniform lon/lat. f(λ,φ)=λ has EXACT physical x-derivative 1/(R cosφ)
    # and f(λ,φ)=φ has EXACT physical y-derivative 1/R, on ANY spacing pattern (verified
    # algebraically: the nonuniform stencil applied to a coordinate-linear field returns the
    # exact analytic slope regardless of h_m/h_p). A single global Δ taken from `x[2]-x[1]` gets
    # this right only at the first cell.
    sgeom = FG.Geometry.SphericalGeometry(6371000.0)
    lon_s = deg2rad.([0.0, 3.0, 8.0, 20.0, 45.0, 46.0, 90.0])
    lat_s = deg2rad.([-40.0, -35.0, -20.0, 0.0, 10.0, 30.0])
    Nlon_s, Nlat_s = length(lon_s), length(lat_s)
    mask_s = trues(Nlon_s, Nlat_s)
    grid_s = FG.Grids.StructuredGrid(sgeom, lon_s, lat_s, mask_s; periodic = false)
    R = sgeom.R

    f_lon = [lon_s[i] for i in 1:Nlon_s, j in 1:Nlat_s]
    f_lat = [lat_s[j] for i in 1:Nlon_s, j in 1:Nlat_s]
    ∂flon∂x = zeros(Nlon_s, Nlat_s)
    ∂flat∂y = zeros(Nlon_s, Nlat_s)
    CGEF.Derivatives.ddx!(∂flon∂x, f_lon, grid_s)
    CGEF.Derivatives.ddy!(∂flat∂y, f_lat, grid_s)
    for i in 2:(Nlon_s-1), j in 1:Nlat_s
        Test.@test ∂flon∂x[i, j] ≈ 1 / (R * cos(lat_s[j])) rtol=1e-8
    end
    for i in 1:Nlon_s, j in 2:(Nlat_s-1)
        Test.@test ∂flat∂y[i, j] ≈ 1 / R rtol=1e-8
    end

    # Regional (non-periodic) grid: the boundary derivative must not wrap to the far edge. A
    # one-sided difference at a true domain edge uses only its interior neighbour.
    Test.@test !FG.Grids.isperiodic(grid_s, 1)
    # Perturbing the far-edge value must NOT change the near-edge (i=1) derivative on a
    # non-periodic grid -- if it wrapped, it would.
    f_lon2 = copy(f_lon)
    f_lon2[end, 3] += 1000.0 # perturb the far edge only
    ∂flon∂x2 = zeros(Nlon_s, Nlat_s)
    CGEF.Derivatives.ddx!(∂flon∂x2, f_lon2, grid_s)
    Test.@test ∂flon∂x2[1, 3] ≈ ∂flon∂x[1, 3]

    # Solid-body rotation on a NONUNIFORM spherical grid: u = U*cos(φ), v = 0. Filtering is
    # performed in planetary-Cartesian coordinates (Aluie 2019 commutativity formulation)
    # specifically so that this rigid-rotation field commutes with filtering; Π should be ≈ 0
    # regardless of axis nonuniformity.
    U = 5.0
    u_rot = [U * cos(lat_s[j]) for i in 1:Nlon_s, j in 1:Nlat_s]
    v_rot = zeros(Nlon_s, Nlat_s)
    Π_s = zeros(Nlon_s, Nlat_s)
    CGEF.Diagnostics.compute_Π!(Π_s, u_rot, v_rot, nothing, grid_s, CGEF.TopHatKernel(), 5e5)
    for i in 2:(Nlon_s-1), j in 2:(Nlat_s-1)
        Test.@test Π_s[i, j] ≈ 0.0 atol=1e-6
    end
end


Test.@testset "1D and singleton-dimension StructuredGrid" begin
    # --- Genuinely 1D Cartesian StructuredGrid: ddx! exact for a linear field, compute_Π!/
    # coarse_grain finite. ---
    cgeom = FG.Geometry.CartesianGeometry()
    x1 = collect(0.0:1000.0:10000.0)
    grid1d = FG.Grids.StructuredGrid(cgeom, x1, trues(length(x1)))
    flin = 3.0 .* x1
    dflin = zeros(length(x1))
    CGEF.Derivatives.ddx!(dflin, flin, grid1d)
    Test.@test all(x -> isapprox(x, 3.0; atol = 1e-8), dflin[2:end-1])

    u1 = rand(length(x1))
    Π1 = zeros(length(x1))
    CGEF.Diagnostics.compute_Π!(Π1, u1, grid1d, CGEF.TopHatKernel(), 3000.0)
    Test.@test all(isfinite, Π1)

    res1 = CGEF.coarse_grain(u1, grid1d; scales = [2000.0, 3000.0], kernel = CGEF.TopHatKernel())
    Test.@test size(res1.Π) == (length(x1), 2)
    Test.@test !any(isnan, res1.cumulative_energy)

    # --- Singleton-dimension StructuredGrid measure: a Cartesian grid with one axis of length 1
    # degenerates from area to the surviving axis's plain width (not zero). ---
    cgrid_singleton = FG.Grids.StructuredGrid(cgeom, x1, [500.0], trues(length(x1), 1))
    Test.@test all(i -> FG.Grids.area(cgrid_singleton, i, 1) ≈ 1000.0, 2:(length(x1)-1))

    # --- Spherical singleton-latitude (zonal transect): the measure is the EXACT arc length
    # along that circle of latitude (R cosφ Δλ), not zero and not `area_element` with a
    # placeholder Δφ (which would leave a spurious extra factor of R). ---
    sgeom2 = FG.Geometry.SphericalGeometry(6371000.0)
    λc = deg2rad.(collect(0.0:5.0:355.0))
    φ0 = deg2rad(10.0)
    sgrid_zonal = FG.Grids.StructuredGrid(sgeom2, λc, [φ0], trues(length(λc), 1); periodic = (true, false))
    exact_arclen = sgeom2.R * cos(φ0) * deg2rad(5.0)
    Test.@test all(i -> isapprox(FG.Grids.area(sgrid_zonal, i, 1), exact_arclen; rtol = 1e-12), eachindex(λc))

    # Before the fix this silently produced NaN (0/0 from a zero total area) — now finite.
    uz = rand(length(λc), 1); vz = rand(length(λc), 1)
    res_zonal = CGEF.coarse_grain(uz, vz, sgrid_zonal; scales = [5e5, 1e6], kernel = CGEF.TopHatKernel())
    Test.@test !any(isnan, res_zonal.cumulative_energy)
    Test.@test !any(isnan, res_zonal.filtering_spectrum)

    # --- Spherical singleton-longitude (meridional transect): arc length R Δφ, no cosφ factor. ---
    φc = deg2rad.(collect(-40.0:5.0:40.0))
    sgrid_merid = FG.Grids.StructuredGrid(sgeom2, [0.0], φc, trues(1, length(φc)); periodic = (false, false))
    exact_merid = sgeom2.R * deg2rad(5.0)
    Test.@test all(j -> isapprox(FG.Grids.area(sgrid_merid, 1, j), exact_merid; rtol = 1e-12), eachindex(φc))
end


# -----------------------------------------------------------------------
# Filtering a degenerate spherical grid. A transect's measure is an ARC LENGTH — `R·Δλ` along a
# parallel, `R·Δφ` along a meridian — not the `R²cosφ·Δλ·Δφ` area form with the missing differential
# replaced by a placeholder. Any engine that factorizes the measure has to reproduce the grid's own
# factors, and the reference here is weighted by `Grids.measure` directly so it cannot share the
# mistake.
# -----------------------------------------------------------------------
Test.@testset "Degenerate spherical grids: the filter weights by the grid's own measure" begin
    function measure_weighted_reference(grid, f, ker, scale)
        geo = FG.Grids.grid_geometry(grid)
        rad = CGEF.Kernels.kernel_radius(ker, scale)
        Nx, Ny = FG.Grids.size_tuple(grid)
        out = zeros(Nx, Ny)
        for j in 1:Ny, i in 1:Nx
            p = FG.Grids.coords(grid, i, j)
            ws = 0.0; wn = 0.0
            for jn in 1:Ny, in_ in 1:Nx
                d = FG.Geometry.distance(geo, p, FG.Grids.coords(grid, in_, jn))
                d <= rad || continue
                w = CGEF.Kernels.kernel_weight(ker, d, scale) * FG.Grids.measure(grid, in_, jn)
                wn += w; ws += w * f[in_, jn]
            end
            out[i, j] = wn > 1e-15 ? ws / wn : 0.0
        end
        return out
    end

    sgeo = FG.Geometry.SphericalGeometry(6.371e6)
    lat = deg2rad.(collect(-60.0:5.0:60.0))
    lon = deg2rad.(collect(0.0:5.0:355.0))
    ker = CGEF.TopHatKernel(); scale = 2.0e6

    # Meridian: one longitude. The spurious `cosφⱼ` an area-form factorization introduces varies with
    # `j`, so it does not cancel in the normalized average — it reweights the transect.
    gmer = FG.Grids.StructuredGrid(sgeo, [0.0], lat, trues(1, length(lat)))
    fmer = reshape([sin(3φ) for φ in lat], 1, length(lat))
    omer = zeros(size(fmer))
    CGEF.Filtering.filter_field!(omer, fmer, gmer, ker, scale;
        backend = CGEF.ComputationalBackends.SerialBackend())
    Test.@test maximum(abs, omer .- measure_weighted_reference(gmer, fmer, ker, scale)) < 1e-13

    # Parallel: one latitude.
    gzon = FG.Grids.StructuredGrid(sgeo, lon, [0.3], trues(length(lon), 1))
    fzon = reshape([sin(2λ) for λ in lon], length(lon), 1)
    ozon = zeros(size(fzon))
    CGEF.Filtering.filter_field!(ozon, fzon, gzon, ker, scale;
        backend = CGEF.ComputationalBackends.SerialBackend())
    Test.@test maximum(abs, ozon .- measure_weighted_reference(gzon, fzon, ker, scale)) < 1e-13

    # The factors an engine uses must be the grid's own, in every case — including the degenerate ones.
    for g in (gmer, gzon, FG.Grids.StructuredGrid(sgeo, lon, lat, trues(length(lon), length(lat))))
        wx, wy = CGEF.Filtering._rectilinear_measure_factors(g)
        Nx, Ny = FG.Grids.size_tuple(g)
        Test.@test all(
            wx[i] * wy[j] == FG.Grids.measure(g, i, j) for j in 1:Ny, i in 1:Nx
        )
    end
end
