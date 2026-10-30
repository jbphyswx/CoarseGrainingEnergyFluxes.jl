# Dispatch gates: every row of the engine table builds its plan and applies it with no runtime dispatch,
# under both mask strategies, and so do the flux entry points, given a plan or building their own. JET
# refuses pre-release Julia.

Test.@testset "JET: every engine's plan and apply, and the flux" begin
    if isempty(VERSION.prerelease)
        targets = (CGEF, CGEF.Kernels, CGEF.Filtering, CGEF.Derivatives, CGEF.Diagnostics, CGEF.Pipeline, CGEF.Geometry,
                   map(n -> Base.get_extension(CGEF, n), _CGEF_EXTENSIONS)...)
        ser = CGEF.ComputationalBackends.SerialBackend()
        for r in (_cgef_cartesian_rows()..., _cgef_spherical_rows()..., _cgef_spheroid_rows()..., _cgef_mesh_rows()...),
            s in (CGEF.Filtering.ZeroFill(), CGEF.Filtering.Deformable())
            f = _cgef_field(size(FG.Grids.mask(r.grid)))
            p = CGEF.Filtering.plan_filter(r.grid, r.kernel, r.ℓ; execution = ser, mask_strategy = s, r.kw...)
            Test.@testset "$(r.engine), $(nameof(typeof(r.kernel))), ℓ = $(r.ℓ), $(nameof(typeof(s)))" begin
                JET.@test_opt target_modules = targets CGEF.Filtering.plan_filter(
                    r.grid, r.kernel, r.ℓ; execution = ser, mask_strategy = s, r.kw...)
                JET.@test_opt target_modules = targets CGEF.Filtering.filter_apply!(zero(f), f, p)
                JET.@test_opt target_modules = targets CGEF.Filtering.filter_apply_batch!((zero(f), zero(f)), (f, f), p)
            end
        end

        geo = FG.Geometry.CartesianGeometry()
        x = 0.0:1000.0:20e3
        xv = collect(x) .+ [iseven(i) ? 4.0 : -3.0 for i in 1:21]
        f = [sin(0.3i + 0.2j) for i in 1:21, j in 1:21]
        out = zeros(21, 21)
        for (g, k) in ((FG.Grids.StructuredGrid(geo, x, x), CGEF.TopHatKernel()),
                       (FG.Grids.StructuredGrid(geo, x, x), CGEF.GaussianKernel()),
                       (FG.Grids.StructuredGrid(geo, xv, xv), CGEF.SharpSpectralKernel()))
            p = CGEF.Filtering.plan_filter(g, k, 5000.0; execution = ser)
            JET.@test_opt target_modules = targets CGEF.Diagnostics.compute_Π!(out, f, f, nothing, g, k, 5000.0;
                                                                                filter_plan = p, execution = ser)
            JET.@test_opt target_modules = targets CGEF.Diagnostics.compute_Π!(out, f, f, nothing, g, k, 5000.0;
                                                                                execution = ser)
        end

        # The flux by the filter's series, on an FFT plan and by the sphere's spin transforms with a held workspace,
        # and a held sweep of one engine taking the exact filtering spectrum from its held rate parts.
        P = FG.Grids.Periodic()
        gp = FG.Grids.StructuredGrid(geo, x, x; topology = (P, P))
        scales = [3000.0, 5000.0]
        for k in (CGEF.GaussianKernel(), CGEF.TopHatKernel())
            sp = CGEF.Filtering.plan_filter(gp, k, 5000.0; method = CGEF.Filtering.Spectral(), execution = ser)
            JET.@test_opt target_modules = targets CGEF.Diagnostics.compute_Π!(out, f, f, nothing, gp, k, 5000.0;
                                                                                filter_plan = sp, execution = ser)
        end
        for k in (CGEF.GaussianKernel(), CGEF.Kernels.SmoothHatKernel())
            fam =CGEF.Filtering.plan_filter_sweep(gp, k, scales; execution = ser, engine = CGEF.Filtering.FootprintEngine(),
                                                   scale_rates = CGEF.Filtering.ScaleRates())
            res = CGEF.Pipeline._allocate_result(gp, length(scales); execution = ser)
            JET.@test_opt target_modules = targets CGEF.coarse_grain!(res, f, f, nothing, gp; scales, kernel = k,
                                                                     filter_plans = fam, execution = ser,
                                                                     spectrum = CGEF.Diagnostics.ForceSpectrum(),
                                                                     scale_derivative = CGEF.Diagnostics.ExactScaleDerivative())
        end
        sg = FG.Connectivity.structured_grid(FG.SphericalSampling.Fejer1Sampling(), 16)
        fs = [sin(I[1] / 5) * cos(I[2] / 4) for I in CartesianIndices(FG.Grids.size_tuple(sg))]
        ws = CGEF.Diagnostics.ΠWorkspace(sg; execution = ser)
        JET.@test_opt target_modules = targets CGEF.Diagnostics.compute_Π!(zero(fs), fs, fs, nothing, sg, CGEF.GaussianKernel(),
                                                                            2.0e6; workspace = ws, execution = ser)
    end
end
