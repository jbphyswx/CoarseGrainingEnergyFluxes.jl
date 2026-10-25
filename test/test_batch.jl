
# Batches: every batched entry point equals the same computation slice by slice, on every backend.

_cgef_same_result(a, b) = isequal(a.Π, b.Π) && isequal(a.cumulative_energy, b.cumulative_energy) &&
                          isequal(a.filtering_spectrum, b.filtering_spectrum)

# A device reduces each energy in its own order. The summands `½|ū|²·A` are non-negative, so two orders
# of a sum of `n` of them lie within `2n·eps` of each other, relative; the spectrum is the density of the
# result's own energies.
_cgef_device_energies(a, b, n) =
    isapprox(collect(a.cumulative_energy), collect(b.cumulative_energy); rtol = 2n * eps()) &&
    isequal(collect(a.filtering_spectrum),
            CGEF.Diagnostics.spectral_density(collect(a.cumulative_energy), collect(a.wavenumber)))
_cgef_device_result(a, b, n) = isequal(a.Π, b.Π) && _cgef_device_energies(a, b, n)

Test.@testset "Pipeline batches equal the per-slice sweep" begin
    P = CGEF.Pipeline
    CB = CGEF.ComputationalBackends
    SER = CB.SerialBackend()
    N = 24
    xs = range(0.0, (N - 1) * 1000.0; length = N)
    m = trues(N, N)
    m[8:12, 6:9] .= false
    g = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), xs, xs, m)
    scales = [4e3, 8e3]
    kern = CGEF.GaussianKernel()
    sweep(u, v, grid) = CGEF.coarse_grain(u, v, grid; scales, kernel = kern, execution = SER)

    # Two trailing batch axes, on every backend.
    u = [sin(0.3i + 0.7j + t + 2z) for i in 1:N, j in 1:N, z in 1:2, t in 1:2]
    v = [cos(0.4i - 0.2j + 3t - z) for i in 1:N, j in 1:N, z in 1:2, t in 1:2]
    refs = [sweep(u[:, :, J], v[:, :, J], g) for J in CartesianIndices((2, 2))]
    for be in (SER, CB.ThreadedBackend(), CB.AutoBackend(), CB.GPUBackend(KA.CPU()), CB.DistributedBackend(), CB.MPIBackend())
        b = P.CoarseGrainBatchResult(g, length(scales), (2, 2); execution = be)
        P.coarse_grain_batch!(b, u, v, g; scales, kernel = kern, execution = be)
        same = be isa CB.GPUBackend ? ((x, y) -> _cgef_device_result(x, y, N^2)) : _cgef_same_result
        Test.@test all(i -> same(b.slices[i], refs[i]), eachindex(refs))
    end

    # A grid per slice.
    g2 = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), range(0.0, 1.5e4; length = 16), range(0.0, 1.5e4; length = 16))
    grids = [g, g2, g]
    us = [u[:, :, 1, 1], [sin(0.2i) * cos(0.3j) for i in 1:16, j in 1:16], u[:, :, 2, 1]]
    vs = [v[:, :, 1, 1], [cos(0.5i - j) for i in 1:16, j in 1:16], v[:, :, 2, 1]]
    for be in (SER, CB.ThreadedBackend())
        rr = [P._allocate_result(gg, length(scales)) for gg in grids]
        P.coarse_grain_slices!(rr, us, vs, grids; scales, kernel = kern, execution = be)
        Test.@test all(t -> _cgef_same_result(rr[t], sweep(us[t], vs[t], grids[t])), eachindex(grids))
    end

    # The vertical profile is a batch along the vertical axis.
    bp = P.coarse_grain_profile(u[:, :, :, 1], v[:, :, :, 1], g; scales, kernel = kern, execution = SER)
    Test.@test all(k -> _cgef_same_result(bp.slices[k], refs[k, 1]), 1:2)
end

# The flux of a `(Nx, Ny, Nb)` field on a two-dimensional grid is a batch of planes, on the host and on the
# device, Cartesian or spherical, with E(ℓ) per slice.
Test.@testset "compute_Π! over a batch axis equals the per-slice flux" begin
    CB = CGEF.ComputationalBackends
    D = CGEF.Diagnostics
    Nb = 3
    m = trues(24, 24)
    m[10:14, 1:3] .= false
    xs = range(0.0, 23000.0; length = 24)
    R = 6.371e6
    sm = trues(24, 11)
    sm[3:6, 4:6] .= false
    cases = ((FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), xs, xs, m), 4e3),
             (FG.Grids.StructuredGrid(FG.Geometry.SphericalGeometry(R), deg2rad.(collect(0.0:15.0:345.0)),
                                      deg2rad.(collect(-60.0:12.0:60.0)), sm), deg2rad(30.0) * R))
    for (g, ℓ) in cases, kern in (CGEF.GaussianKernel(), CGEF.TopHatKernel()), be in (CB.SerialBackend(), CB.GPUBackend(KA.CPU()))
        sz = FG.Grids.size_tuple(g)
        u = [sin(0.3i + 0.7j + b) for i in 1:sz[1], j in 1:sz[2], b in 1:Nb]
        v = [cos(0.4i - 0.2j + 2b) for i in 1:sz[1], j in 1:sz[2], b in 1:Nb]
        fp = CGEF.Filtering.plan_filter(g, kern, ℓ; execution = be)
        area = D.active_area(g)
        kw = (; filter_plan = fp, execution = be, deriv_plan = CGEF.Derivatives.derivative_plan(D.output_grid(g, fp)))
        ws1 = D.ΠWorkspace(g; execution = be)
        ref = zeros(sz..., Nb)
        refE = zeros(Nb)
        for b in 1:Nb
            D.compute_Π!(view(ref, :, :, b), u[:, :, b], v[:, :, b], nothing, g, kern, ℓ; workspace = ws1, kw...)
            refE[b] = D.energy_from_filtered(ws1, fp, false, area)
        end
        wsb = D.ΠWorkspace(g, (Nb,); execution = be)
        got = D.compute_Π!(zeros(sz..., Nb), u, v, nothing, g, kern, ℓ; workspace = wsb, kw...)
        gotE = D.energy_from_filtered!(zeros(Nb), wsb, fp, false, area)
        if be isa CB.GPUBackend && wsb.spin !== nothing
            # A device workspace runs the spin transforms' NUFFT on the library's threads, which spread in
            # a run-dependent order, so two runs agree to the rounding of those sums, one term per cell.
            n = prod(sz)
            Test.@test isapprox(got, ref; rtol = 2n * eps())
            Test.@test isapprox(gotE, refE; rtol = 2n * eps())
        else
            Test.@test got == ref
            # Each slice is reduced over the same cells, so in the same order.
            Test.@test gotE == refE
        end
    end
end

# On a device the fused batch filters every slice in one launch; on the host the batch is the slice loop.
Test.@testset "filter_apply_batched! equals the per-slice apply" begin
    CB = CGEF.ComputationalBackends
    R = 6.371e6
    lon = range(0.0, 2π * 31 / 32; length = 32)
    lat = range(-1.0, 1.0; length = 16)
    msk = trues(32, 16)
    msk[:, 1:2] .= false
    msk[10:13, :] .= false
    gcart = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), range(0.0, 3.1e4; length = 32), range(0.0, 1.5e4; length = 16))
    for (grid, kern, ℓ) in ((FG.Grids.StructuredGrid(FG.Geometry.SphericalGeometry(R), lon, lat, msk), CGEF.GaussianKernel(), 400e3),
                            (gcart, CGEF.GaussianKernel(), 8e3), (gcart, CGEF.TopHatKernel(), 8e3))
        f = [sin(0.3i + 0.5j + b) for i in 1:32, j in 1:16, b in 1:3]
        ps = CGEF.Filtering.plan_filter(grid, kern, ℓ; execution = CB.SerialBackend())
        loop = cat((CGEF.Filtering.filter_apply!(zeros(32, 16), f[:, :, b], ps) for b in 1:3)...; dims = 3)
        for b in (CB.SerialBackend(), CB.GPUBackend(KA.CPU()))
            p = CGEF.Filtering.plan_filter(grid, kern, ℓ; execution = b)
            Test.@test CGEF.Filtering.filter_apply_batched!(zeros(32, 16, 3), f, p) == loop
        end
    end
    # Independent problems, one plan each, ragged in size.
    counts = [37, 121, 64, 199]
    plans = [CGEF.Filtering.plan_filter(FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), collect(range(0.0, 1.0e4; length = n))),
                                        CGEF.GaussianKernel(), 2000.0; execution = CB.SerialBackend()) for n in counts]
    fields = [sin.(0.1 .* (1:n)) for n in counts]
    refs = [CGEF.Filtering.filter_apply!(zeros(n), fields[t], plans[t]) for (t, n) in enumerate(counts)]
    for b in (CB.SerialBackend(), CB.ThreadedBackend())
        outs = [zeros(n) for n in counts]
        CGEF.Filtering.filter_slices!(outs, fields, plans; execution = b)
        Test.@test outs == refs
    end
end

# A spectral sweep analyzes the raw fields once and synthesizes each scale; the flux is the per-scale one.
Test.@testset "A spectral sweep equals the per-scale flux" begin
    CB = CGEF.ComputationalBackends
    D = CGEF.Diagnostics
    N = 32
    xs = range(0.0, (N - 1) * 1000.0; length = N)
    m = trues(N, N)
    m[:, 1:4] .= false
    m[12:16, :] .= false
    scales = [4e3, 8e3, 16e3]
    u = [sin(0.3i + 0.7j) for i in 1:N, j in 1:N]
    v = [cos(0.4i - 0.2j) for i in 1:N, j in 1:N]
    k = CGEF.GaussianKernel()
    for mk in (trues(N, N), m), s in (CGEF.Filtering.ZeroFill(), CGEF.Filtering.Deformable())
        g = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), xs, xs, mk; periodic = (true, true))
        got = CGEF.coarse_grain(u, v, g; scales, kernel = k, method = CGEF.Filtering.Spectral(), mask_strategy = s,
                                execution = CB.SerialBackend())
        for (i, ℓ) in enumerate(scales)
            pl = CGEF.Filtering.plan_filter(g, k, ℓ; method = CGEF.Filtering.Spectral(), mask_strategy = s)
            ref = D.compute_Π!(zeros(N, N), u, v, nothing, g, k, ℓ; filter_plan = pl, execution = CB.SerialBackend(), mask_strategy = s)
            Test.@test got.Π[:, :, i] == ref
        end
    end
end
