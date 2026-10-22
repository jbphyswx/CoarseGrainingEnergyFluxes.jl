
Test.@testset "Aqua" begin
    Aqua.test_all(CGEF; ambiguities = false, unbound_args = (VERSION >= v"1.12"))
end

const _CGEF_EXTENSIONS = (
    :CoarseGrainingEnergyFluxesFFTWExt, :CoarseGrainingEnergyFluxesFastSphericalHarmonicsExt,
    :CoarseGrainingEnergyFluxesNUFSHTExt, :CoarseGrainingEnergyFluxesOhMyThreadsExt,
    :CoarseGrainingEnergyFluxesDistributedExt, :CoarseGrainingEnergyFluxesKernelAbstractionsExt,
    :CoarseGrainingEnergyFluxesMPIExt,
)

Test.@testset "Explicit imports" begin
    Test.@test (EI.check_no_implicit_imports(CGEF); true)
    # The core imports FlowTransformBindings so its `__init__` runs before FastTransforms loads.
    Test.@test (EI.check_no_stale_explicit_imports(CGEF; ignore = (:FlowTransformBindings,)); true)
    for name in _CGEF_EXTENSIONS
        ext = Base.get_extension(CGEF, name)
        Test.@test ext !== nothing
        Test.@test (EI.check_no_implicit_imports(ext); true)
        Test.@test (EI.check_no_stale_explicit_imports(ext); true)
    end
end

# JET refuses pre-release Julia, so the gate runs on released versions.
Test.@testset "JET: type stability of the filter and flux entry points" begin
    if isempty(VERSION.prerelease)
        targets = (CGEF, CGEF.Kernels, CGEF.Filtering, CGEF.Derivatives, CGEF.Diagnostics, CGEF.Pipeline,
                   map(n -> Base.get_extension(CGEF, n), _CGEF_EXTENSIONS)...)
        SER = CGEF.ComputationalBackends.SerialBackend()
        geo = FG.Geometry.CartesianGeometry()
        x = 0.0:1000.0:20e3
        xv = collect(x) .+ [iseven(i) ? 4.0 : -3.0 for i in 1:21]
        f = [sin(0.3i + 0.2j) for i in 1:21, j in 1:21]
        out = zeros(21, 21)
        for (g, k) in ((FG.Grids.StructuredGrid(geo, x, x), CGEF.TopHatKernel()),
                       (FG.Grids.StructuredGrid(geo, x, x), CGEF.GaussianKernel()),
                       (FG.Grids.StructuredGrid(geo, xv, xv), CGEF.SharpSpectralKernel()))
            JET.@test_opt target_modules = targets CGEF.Filtering.plan_filter(g, k, 5000.0; execution = SER)
            p = CGEF.Filtering.plan_filter(g, k, 5000.0; execution = SER)
            JET.@test_opt target_modules = targets CGEF.Filtering.filter_apply!(out, f, p)
            JET.@test_opt target_modules = targets CGEF.Filtering.filter_apply_batch!((out, zero(out)), (f, f), p)
            # A derivative plan's type carries each direction's node count, which the axis length sets.
            dp = CGEF.Derivatives.derivative_plan(CGEF.Diagnostics.output_grid(g, p))
            JET.@test_opt target_modules = targets CGEF.Diagnostics.compute_Π!(out, f, f, nothing, g, k, 5000.0;
                                                                                filter_plan = p, deriv_plan = dp,
                                                                                execution = SER)
        end
    end
end
