
# Grids whose cells are named by a single index — a scattered node set and every sphere pixelization
# alike — share one real-space engine: a CSR gather over the grid's own metric ball. The engine is
# selected by the `Grids.cell_address` trait, so a layout that declares itself flat-celled gets the
# whole pipeline without naming itself anywhere in this package.

# The operator, assembled from the kernel and the grid's own distance and measure: a weighted mean over
# every cell inside the kernel radius. `ZeroFill` leaves inactive cells in the denominator and filters
# at an inactive target too; `Deformable` renormalizes over the active part of the window and zeroes an
# inactive target, so the two differ only where a window meets the mask.
function _flatcell_reference(grid, f, kernel, scale, strategy)
    geo = FG.Grids.grid_geometry(grid)
    rad = CGEF.Kernels.kernel_radius(kernel, scale)
    n = length(FG.Grids.mask(grid))
    pts = [FG.Grids.coords(Tuple, grid, i) for i in 1:n]
    meas = [FG.Grids.measure(grid, j) for j in 1:n]
    act = [FG.Grids.isactive(grid, j) for j in 1:n]
    dim = Val(FG.Grids.ncoordinates(grid))
    out = zeros(n)
    for i in 1:n
        (strategy isa CGEF.Filtering.ZeroFill || act[i]) || continue
        num = 0.0
        den = 0.0
        for j in 1:n
            d = FG.Geometry.distance(geo, pts[i], pts[j])
            d <= rad || continue
            w = CGEF.Kernels.kernel_weight(kernel, d, scale, dim) * meas[j]
            act[j] && (num += w * f[j])
            den += (strategy isa CGEF.Filtering.ZeroFill || act[j]) ? w : 0.0
        end
        out[i] = num / den
    end
    return out
end

# A filter width of about one and a third cell widths, taken from the grid's own cell count. The
# Gaussian truncates at 1.96ℓ, so the window reaches about two and a half cells and holds ~20 of them
# on every layout. A width fixed as a fraction of the planetary radius is a different number of cells
# on each of these, and on the coarsest it leaves every window holding one cell, where filtering is
# the identity and the comparisons below hold trivially.
_flatcell_scale(grid) =
    1.3 * FG.Geometry.radius(FG.Grids.grid_geometry(grid)) *
    sqrt(4π / length(FG.Grids.mask(grid)))

# Inactive within half the kernel radius of cell 1. Sized off the radius so the hole sits inside some
# active cell's window. The two mask strategies are the same operator until a window reaches an
# inactive cell.
function _flatcell_disc_mask(grid, kernel, scale)
    geo = FG.Grids.grid_geometry(grid)
    n = length(FG.Grids.mask(grid))
    cut = CGEF.Kernels.kernel_radius(kernel, scale) / 2
    p1 = FG.Grids.coords(Tuple, grid, 1)
    return [FG.Geometry.distance(geo, p1, FG.Grids.coords(Tuple, grid, i)) > cut for i in 1:n]
end

# Whether any active cell's window reaches an inactive cell — the precondition that makes the two
# strategies distinguishable.
function _flatcell_mask_in_reach(grid, kernel, scale)
    geo = FG.Grids.grid_geometry(grid)
    rad = CGEF.Kernels.kernel_radius(kernel, scale)
    n = length(FG.Grids.mask(grid))
    pts = [FG.Grids.coords(Tuple, grid, i) for i in 1:n]
    act = [FG.Grids.isactive(grid, i) for i in 1:n]
    return any(
        i -> act[i] && any(j -> !act[j] && FG.Geometry.distance(geo, pts[i], pts[j]) <= rad, 1:n),
        1:n,
    )
end

# One entry per architecture, each a function of the mask so the masked and unmasked grids come from
# the same call.
function _flatcell_builders()
    sgeo = FG.Geometry.SphericalGeometry()
    return [
        ("RingGrid", m -> FG.Grids.RingGrid(
            sgeo, FG.SphericalSampling.ReducedGaussianSampling([16, 24, 32, 32, 24, 16]); mask = m)),
        ("CubedSphereGrid", m -> FG.Grids.CubedSphereGrid(sgeo, 6; mask = m)),
        ("HEALPixGrid", m -> FG.Grids.HEALPixGrid(sgeo, 4; mask = m)),
        ("IcosahedralGrid", m -> FG.Grids.IcosahedralGrid(sgeo, 4; mask = m)),
        ("YinYangGrid", m -> FG.Grids.YinYangGrid(sgeo, 20, 10; mask = m)),
    ]
end

_flatcell_field(grid, n) = [
    (p = FG.Grids.coords(Tuple, grid, i); sin(2 * p[1]) * cos(p[2]) + 0.3cos(3 * p[1])) for i in 1:n
]


Test.@testset "Flat-cell architectures: the node engine evaluates the definition" begin
    kernel = CGEF.GaussianKernel()
    RS = CGEF.Filtering.RealSpace()
    SER = CGEF.ComputationalBackends.SerialBackend()

    for (name, build) in _flatcell_builders()
        Test.@testset "$name" begin
            for masked in (false, true)
                g0 = build(nothing)
                scale = _flatcell_scale(g0)
                grid = masked ? build(_flatcell_disc_mask(g0, kernel, scale)) : g0
                n = length(FG.Grids.mask(grid))
                f = _flatcell_field(grid, n)
                if masked
                    Test.@test _flatcell_mask_in_reach(grid, kernel, scale)
                end

                Test.@test FG.Grids.cell_address(grid) === FG.Grids.FlatCells()
                Test.@test CGEF.Filtering.build_footprint(grid, kernel, scale) isa
                           CGEF.Filtering.NodeFilterPlan

                for strategy in (CGEF.Filtering.ZeroFill(), CGEF.Filtering.Deformable())
                    plan = CGEF.Filtering.plan_filter(grid, kernel, scale;
                        method = RS, mask_strategy = strategy, backend = SER)
                    out = zeros(n)
                    CGEF.Filtering.filter_apply!(out, f, plan)
                    ref = _flatcell_reference(grid, f, kernel, scale, strategy)
                    Test.@test maximum(abs, out .- ref) < 1e-12 * maximum(abs, ref)

                    oc = zeros(n)
                    CGEF.Filtering.filter_apply!(oc, ones(n), plan)
                    act = [FG.Grids.isactive(grid, i) for i in 1:n]
                    Test.@test oc ≈ _flatcell_reference(grid, ones(n), kernel, scale, strategy) atol = 1e-12
                    # `Deformable` renormalizes over the active window, so a constant survives
                    # anywhere. `ZeroFill` keeps the inactive weight in the denominator, so it
                    # reproduces a constant exactly on a fully active grid and dilutes it within ℓ of
                    # a mask.
                    if strategy isa CGEF.Filtering.Deformable || !masked
                        Test.@test maximum(abs, oc[act] .- 1.0) < 1e-12
                    else
                        Test.@test minimum(oc[act]) < 1.0 - 1e-6
                        Test.@test maximum(oc[act]) <= 1.0 + 1e-12
                    end

                    # Every backend reproduces serial bit for bit: they differ in who walks the CSR
                    # blocks, never in what is summed or in what order.
                    for backend in (CGEF.ComputationalBackends.ThreadedBackend(),
                                    CGEF.ComputationalBackends.GPUBackend(KA.CPU()),
                                    CGEF.ComputationalBackends.DistributedBackend(),
                                    CGEF.ComputationalBackends.MPIBackend())
                        pb = CGEF.Filtering.plan_filter(grid, kernel, scale;
                            method = RS, mask_strategy = strategy, backend = backend)
                        ob = zeros(n)
                        CGEF.Filtering.filter_apply!(ob, f, pb)
                        Test.@test ob == out
                    end
                end

                # The two strategies are the same operator on a fully active grid, and separate once a
                # window reaches an inactive cell.
                let pz = CGEF.Filtering.plan_filter(grid, kernel, scale;
                        method = RS, mask_strategy = CGEF.Filtering.ZeroFill(), backend = SER),
                    pd = CGEF.Filtering.plan_filter(grid, kernel, scale;
                        method = RS, mask_strategy = CGEF.Filtering.Deformable(), backend = SER),
                    oz = zeros(n), od = zeros(n)
                    CGEF.Filtering.filter_apply!(oz, f, pz)
                    CGEF.Filtering.filter_apply!(od, f, pd)
                    Test.@test (oz == od) == !masked
                end
            end
        end
    end
end


Test.@testset "Flat-cell architectures: gradients, flux and the sweep" begin
    kernel = CGEF.GaussianKernel()
    RS = CGEF.Filtering.RealSpace()
    SER = CGEF.ComputationalBackends.SerialBackend()

    for (name, build) in _flatcell_builders()
        Test.@testset "$name" begin
            grid = build(nothing)
            n = length(FG.Grids.mask(grid))
            scale = _flatcell_scale(grid)
            pts = [FG.Grids.coords(Tuple, grid, i) for i in 1:n]

            # A formula-neighbour layout carries no stored connectivity, so the gradient plan is built
            # through the grid's own adjacency. The least-squares fit reproduces a constant exactly.
            dplan = CGEF.Derivatives.gradient_plan(grid)
            gx = zeros(n); gy = zeros(n)
            FG.Operators.gradient!(gx, gy, fill(2.5, n), dplan)
            Test.@test maximum(abs, gx) < 1e-8
            Test.@test maximum(abs, gy) < 1e-8

            u = [cos(p[2]) * sin(p[1]) for p in pts]
            v = [cos(p[2]) * cos(p[1]) for p in pts]

            Π = zeros(n)
            CGEF.Diagnostics.compute_Π!(Π, u, v, nothing, grid, kernel, scale;
                                        method = RS, backend = SER)
            Test.@test all(isfinite, Π)

            # The flux plumbing is backend-independent for the same reason the filter is.
            Πt = zeros(n)
            CGEF.Diagnostics.compute_Π!(Πt, u, v, nothing, grid, kernel, scale;
                method = RS, backend = CGEF.ComputationalBackends.ThreadedBackend())
            Test.@test Πt == Π

            r = CGEF.coarse_grain(u, v, grid; scales = [scale, 1.4scale], kernel = kernel,
                                  method = RS, spectrum = CGEF.Diagnostics.NoSpectrum())
            Test.@test size(r.Π) == (n, 2)
            Test.@test all(isfinite, r.Π)

            # Coarse graining is the compact real-space convolution on every architecture, so these
            # reach the same engine with no `method` named.
            Test.@test CGEF.Filtering._default_method(grid) isa CGEF.Filtering.RealSpace
            Test.@test CGEF.Filtering.plan_filter(grid, kernel, scale).footprint isa
                       CGEF.Filtering.NodeFilterPlan

            rep = CGEF.check_setup(grid, kernel, scale; method = RS)
            Test.@test occursin("node CSR", rep.engine)
            Test.@test rep.spacing == ()
        end
    end
end
