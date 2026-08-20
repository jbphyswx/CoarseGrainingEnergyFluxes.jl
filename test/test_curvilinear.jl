
# CurvilinearGrid support built from scratch (Stage 3): WLSQ tangent-plane gradients, exact
# corner-based quadrilateral areas, real-space scattered-footprint filtering, and the full
# compute_Π!/coarse_grain pipeline, cross-checked against the StructuredGrid engine.
Test.@testset "CurvilinearGrid (WLSQ / areas / pipeline)" begin
    cart = FG.Geometry.CartesianGeometry()

    # --- HARD GATE: a "fake curvilinear" separable grid with UNIFORM (equally spaced) axes must
    # reproduce the StructuredGrid derivatives EXACTLY (to floating point) wherever both engines use
    # the same stencil. Distinct dx≠dy and a linear+quadratic field make this sensitive to any
    # axis-swap / transpose / sign error. In the interior the WLSQ normal matrix is diagonal and the
    # fit decouples per axis, reducing exactly to the centered difference the structured engine uses.
    # At a bounded end they genuinely differ, and each end is checked against its own scheme below. ---
    xs = collect(0.0:2.0:20.0)   # dx = 2
    ys = collect(0.0:5.0:30.0)   # dy = 5 (distinct spacing)
    Nx, Ny = length(xs), length(ys)
    sgrid = FG.Grids.StructuredGrid(cart, xs, ys, trues(Nx, Ny))
    x = [xs[i] for i in 1:Nx, j in 1:Ny]
    y = [ys[j] for i in 1:Nx, j in 1:Ny]
    cgrid = FG.Grids.CurvilinearGrid(cart, x, y, trues(Nx, Ny))
    dplan = FG.Connectivity.gradient_plan(cgrid)

    f = [3.0*xs[i] - 2.0*ys[j] + 0.5*xs[i]^2 + 0.25*ys[j]^2 for i in 1:Nx, j in 1:Ny]
    sx = zeros(Nx, Ny); sy = zeros(Nx, Ny); cx = zeros(Nx, Ny); cy = zeros(Nx, Ny)
    CGEF.Derivatives.ddx!(sx, f, sgrid); CGEF.Derivatives.ddy!(sy, f, sgrid)
    FG.Discretization.gradient!(cx, cy, f, dplan)
    Test.@test maximum(abs.(cx[2:end-1, :] .- sx[2:end-1, :])) < 1e-10   # hard gate (ddx vs structured)
    Test.@test maximum(abs.(cy[:, 2:end-1] .- sy[:, 2:end-1])) < 1e-10   # hard gate (ddy vs structured)

    # A three-node window is exact for a quadratic wherever it is evaluated, so the structured engine
    # is exact at the bounded ends too, not just in the interior.
    ∂f∂x = [3.0 + xs[i] for i in 1:Nx, j in 1:Ny]
    ∂f∂y = [-2.0 + 0.5 * ys[j] for i in 1:Nx, j in 1:Ny]
    Test.@test maximum(abs.(sx .- ∂f∂x)) < 1e-10
    Test.@test maximum(abs.(sy .- ∂f∂y)) < 1e-10

    # WLSQ fits a linear model to the face neighbours it has, so at an end it IS the one-sided
    # difference — first order, and exactly that value rather than approximately it.
    Test.@test cx[1, 3] ≈ (f[2, 3] - f[1, 3]) / (xs[2] - xs[1])
    Test.@test cx[Nx, 3] ≈ (f[Nx, 3] - f[Nx-1, 3]) / (xs[Nx] - xs[Nx-1])
    Test.@test cy[3, 1] ≈ (f[3, 2] - f[3, 1]) / (ys[2] - ys[1])
    Test.@test cy[3, Ny] ≈ (f[3, Ny] - f[3, Ny-1]) / (ys[Ny] - ys[Ny-1])

    # --- Non-orthogonal (sheared + rotated) curvilinear grid: WLSQ reconstructs a LINEAR field's
    # gradient exactly on ANY stencil (it cancels the leading truncation term), so a known
    # analytic gradient is recovered to floating point at every node — the specific case a
    # transposed 2×2 solve would fail, since here the normal matrix is genuinely non-diagonal
    # (Axy ≠ 0). ---
    a, b, c, d = 2.0, 0.7, -0.4, 3.0   # non-orthogonal affine index→physical map (|det| = 6.28)
    Ni, Nj = 9, 7
    slon = [a*i + b*j for i in 1:Ni, j in 1:Nj]
    slat = [c*i + d*j for i in 1:Ni, j in 1:Nj]
    shear = FG.Grids.CurvilinearGrid(cart, slon, slat, trues(Ni, Nj))
    splan = FG.Connectivity.gradient_plan(shear)
    p, q = 1.3, -2.1
    g = [p*slon[i,j] + q*slat[i,j] for i in 1:Ni, j in 1:Nj]
    gx = zeros(Ni, Nj); gy = zeros(Ni, Nj)
    FG.Discretization.gradient!(gx, gy, g, splan)
    Test.@test maximum(abs.(gx .- p)) < 1e-10
    Test.@test maximum(abs.(gy .- q)) < 1e-10

    # --- Nonuniform separable grid: WLSQ is a LINEAR reconstruction, so (per the plan) it agrees
    # with the structured nonuniform result only to within its truncation bound, NOT to floating
    # point for a quadratic. It is, however, exact for a linear field on any spacing — assert
    # that clean, honest property against both engines. ---
    xnu = [0.0, 1.0, 2.5, 4.0, 7.0, 12.0]
    ynu = [0.0, 0.5, 1.5, 3.5, 7.5]
    Nxn, Nyn = length(xnu), length(ynu)
    sgnu = FG.Grids.StructuredGrid(cart, xnu, ynu, trues(Nxn, Nyn))
    lonn = [xnu[i] for i in 1:Nxn, j in 1:Nyn]
    latn = [ynu[j] for i in 1:Nxn, j in 1:Nyn]
    cgnu = FG.Grids.CurvilinearGrid(cart, lonn, latn, trues(Nxn, Nyn))
    flin = [2.0*xnu[i] + 3.0*ynu[j] for i in 1:Nxn, j in 1:Nyn]
    gxs = zeros(Nxn, Nyn); gxc = zeros(Nxn, Nyn); gyc = zeros(Nxn, Nyn)
    CGEF.Derivatives.ddx!(gxs, flin, sgnu)
    FG.Discretization.gradient!(gxc, gyc, flin, FG.Connectivity.gradient_plan(cgnu))
    Test.@test maximum(abs.(gxc .- 2.0)) < 1e-10   # WLSQ exact for linear on nonuniform stencil
    Test.@test maximum(abs.(gyc .- 3.0)) < 1e-10
    Test.@test maximum(abs.(gxs .- 2.0)) < 1e-10   # structured also exact for linear

    # --- Exact corner-based quadrilateral areas sum to the true domain area. A sheared
    # parallelogram mesh: each cell is a parallelogram of area |ad-bc|, so the total is
    # |ad-bc|·Ncell_i·Ncell_j (the area of the enclosing parallelogram). ---
    ax, bx, cx2, dx2 = 2.0, 0.5, 0.0, 3.0   # |det| = 6
    Nci, Ncj = 5, 4
    xc = [ax*(i-1) + bx*(j-1) for i in 1:(Nci+1), j in 1:(Ncj+1)]
    yc = [cx2*(i-1) + dx2*(j-1) for i in 1:(Nci+1), j in 1:(Ncj+1)]
    cen_x = [(xc[i,j]+xc[i+1,j]+xc[i+1,j+1]+xc[i,j+1])/4 for i in 1:Nci, j in 1:Ncj]
    cen_y = [(yc[i,j]+yc[i+1,j]+yc[i+1,j+1]+yc[i,j+1])/4 for i in 1:Nci, j in 1:Ncj]
    agrid = FG.Grids.CurvilinearGrid(cart, cen_x, cen_y, trues(Nci, Ncj);
                                 x_corner=xc, y_corner=yc)
    Test.@test sum(FG.Grids.measure(agrid)) ≈ abs(ax*dx2 - bx*cx2) * Nci * Ncj
    Test.@test all(≈(6.0), FG.Grids.measure(agrid))

    # --- Spherical corner-based quadrilateral area is diagonal-invariant: splitting the SAME
    # quadrilateral along its other diagonal must give the identical area (both decompositions
    # describe the identical enclosed region) — an exact identity (zero tolerance, no limit or
    # approximation involved), unlike comparing to a lon/lat "zonal band" formula, which is a
    # DIFFERENT shape (a graticule cell's east/west edges are parallels — small circles — not the
    # great-circle arcs `_quad_area` uses), so it only coincides with the geodesic-quadrilateral
    # area in the Δφ→0 limit and was the wrong thing to compare against here. ---
    sph = FG.Geometry.SphericalGeometry(6371000.0)
    λc = deg2rad.(collect(0.0:2.0:10.0))
    φc = deg2rad.(collect(10.0:2.0:20.0))
    slonc = [λc[i] for i in 1:length(λc), j in 1:length(φc)]
    slatc = [φc[j] for i in 1:length(λc), j in 1:length(φc)]
    Ncλ, Ncφ = length(λc)-1, length(φc)-1
    scen_lon = [(slonc[i,j]+slonc[i+1,j])/2 for i in 1:Ncλ, j in 1:Ncφ]
    scen_lat = [(slatc[i,j]+slatc[i,j+1])/2 for i in 1:Ncλ, j in 1:Ncφ]
    sagrid = FG.Grids.CurvilinearGrid(sph, scen_lon, scen_lat, trues(Ncλ, Ncφ);
                                  x_corner=slonc, y_corner=slatc)
    for j in 1:Ncφ, i in 1:Ncλ
        λ1, φ1 = slonc[i,j],     slatc[i,j]
        λ2, φ2 = slonc[i+1,j],   slatc[i+1,j]
        λ3, φ3 = slonc[i+1,j+1], slatc[i+1,j+1]
        λ4, φ4 = slonc[i,j+1],   slatc[i,j+1]
        # Same quantity FlowGeometries builds the stored area from: the spherical excess of the
        # quad's two triangles, here split along the OTHER diagonal.
        uv(λ, φ) = FG.Geometry.unit_vector(Float64, (λ, φ))
        other_diag = sph.R^2 * (
            FG.Geometry.spherical_excess(uv(λ2, φ2), uv(λ3, φ3), uv(λ4, φ4)) +
            FG.Geometry.spherical_excess(uv(λ2, φ2), uv(λ4, φ4), uv(λ1, φ1))
        )
        Test.@test FG.Grids.measure(sagrid)[i,j] ≈ other_diag rtol=1e-12
    end

    # --- Real-space filtering on a curvilinear grid: a constant field is returned unchanged. ---
    cfld = fill(7.5, Nx, Ny)
    cfo = zeros(Nx, Ny)
    CGEF.Filtering.filter_field!(cfo, cfld, cgrid, CGEF.TopHatKernel(), 6.0)
    Test.@test all(x -> isapprox(x, 7.5; atol=1e-10), cfo)

    # --- Full compute_Π!/coarse_grain pipeline: on the uniform Cartesian "fake curvilinear" grid the
    # result must match the StructuredGrid pipeline on the identical coordinates wherever the two
    # derivative engines agree — the interior — since the footprints carry identical weights and
    # neighbours and the contraction is the same. The boundary ring inherits the end-stencil
    # difference gated above. ---
    uu = [sin(xs[i]/7) * cos(ys[j]/9) for i in 1:Nx, j in 1:Ny]
    vv = [cos(xs[i]/5) * sin(ys[j]/11) for i in 1:Nx, j in 1:Ny]
    Πs = zeros(Nx, Ny); Πc = zeros(Nx, Ny)
    CGEF.Diagnostics.compute_Π!(Πs, uu, vv, nothing, sgrid, CGEF.TopHatKernel(), 8.0)
    CGEF.Diagnostics.compute_Π!(Πc, uu, vv, nothing, cgrid, CGEF.TopHatKernel(), 8.0)
    Test.@test maximum(abs.(Πc[2:end-1, 2:end-1] .- Πs[2:end-1, 2:end-1])) <
               1e-9 * maximum(abs.(Πs)) + 1e-12

    res = CGEF.coarse_grain(uu, vv, cgrid; scales=[8.0, 12.0], kernel=CGEF.TopHatKernel(),
                            spectrum = false)
    Test.@test size(res.Π, 3) == 2
    Test.@test res.Π[:, :, 1] ≈ Πc
    Test.@test !any(isnan, res.cumulative_energy)
    # The top-hat cannot carry a filtering spectral density, so the finiteness of the density is
    # checked on the same pipeline with a kernel that can.
    Test.@test !any(isnan, CGEF.coarse_grain(uu, vv, cgrid; scales=[8.0, 12.0],
                                             kernel=CGEF.GaussianKernel()).filtering_spectrum)

    # --- Spherical curvilinear sanity: solid-body rotation u = U cos φ, v = 0 gives Π ≈ 0
    # (filtering is done in planetary-Cartesian coordinates, so rigid rotation commutes with the
    # filter). The residual is set by WLSQ/chord discretization, so the interior tolerance is
    # looser than the arc-length-exact StructuredGrid case. ---
    lon_s = deg2rad.([0.0, 4.0, 9.0, 15.0, 22.0, 30.0])   # moderate nonuniformity (no huge jumps)
    lat_s = deg2rad.([-20.0, -12.0, -5.0, 3.0, 12.0, 20.0])
    Nls, Nas = length(lon_s), length(lat_s)
    clon = [lon_s[i] for i in 1:Nls, j in 1:Nas]
    clat = [lat_s[j] for i in 1:Nls, j in 1:Nas]
    scurv = FG.Grids.CurvilinearGrid(sph, clon, clat, trues(Nls, Nas))
    U = 5.0
    urot = [U*cos(lat_s[j]) for i in 1:Nls, j in 1:Nas]
    vrot = zeros(Nls, Nas)
    Πsb = zeros(Nls, Nas)
    CGEF.Diagnostics.compute_Π!(Πsb, urot, vrot, nothing, scurv, CGEF.TopHatKernel(), 5e5)
    Test.@test all(isfinite, Πsb)
    for i in 2:(Nls-1), j in 2:(Nas-1)
        Test.@test abs(Πsb[i, j]) < 1e-3
    end
end
