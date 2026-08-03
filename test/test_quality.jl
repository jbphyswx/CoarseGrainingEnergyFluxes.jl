
# -----------------------------------------------------------------------
# Code quality / hygiene gates (house style: Aqua + ExplicitImports + JET)
# -----------------------------------------------------------------------
Test.@testset "Aqua" begin
    Aqua.test_all(CGEF; ambiguities = false, unbound_args = (VERSION >= v"1.12"))
end


Test.@testset "Explicit imports (core)" begin
    # Core module + submodules: no bare `using` (no implicit imports) and no stale explicit
    # imports — the strict qualified-import policy.
    Test.@test (EI.check_no_implicit_imports(CGEF); true)
    Test.@test (EI.check_no_stale_explicit_imports(CGEF); true)
    # Per-extension checks (each loaded backend extension must also be import-clean).
    for extname in (
        :CoarseGrainingEnergyFluxesFFTWExt,
        :CoarseGrainingEnergyFluxesFINUFFTExt,
        :CoarseGrainingEnergyFluxesFastSphericalHarmonicsExt,
        :CoarseGrainingEnergyFluxesNUFSHTExt,
        :CoarseGrainingEnergyFluxesOhMyThreadsExt,
        :CoarseGrainingEnergyFluxesDistributedExt,
        :CoarseGrainingEnergyFluxesGPUExt,
        :CoarseGrainingEnergyFluxesMPIExt,
    )
        ext = Base.get_extension(CGEF, extname)
        ext === nothing && continue
        Test.@test (EI.check_no_implicit_imports(ext); true)
        Test.@test (EI.check_no_stale_explicit_imports(ext); true)
    end
end


Test.@testset "JET type stability (hot path)" begin
    # JET tracks compiler internals and explicitly refuses to run on pre-release Julia
    # (`@test_opt` throws on nightly/rc — it loads as a no-op stub). Skip there; the type-stability
    # gate runs on every released version in CI.
    if !isempty(VERSION.prerelease)
        @info "Skipping JET type-stability checks on pre-release Julia $(VERSION)"
        Test.@test_skip true
    else
        # Analyse this package's own frames — core modules plus whichever extensions are loaded —
        # rather than a dependency's internals. A plan built with `AutoBackend` resolves to
        # `ThreadedBackend` whenever the suite runs with more than one thread, which drags
        # OhMyThreads' scheduler machinery into the analysis; its `throw_if_boxed_captures` walk is
        # dynamically dispatched by construction, and gating on code this package does not own would
        # report someone else's design decision as our regression. A dynamic dispatch made FROM our
        # frames is still attributed to our frames, so nothing of ours is hidden by this.
        jet_targets = (CGEF, CGEF.Kernels, CGEF.Filtering, CGEF.Derivatives, CGEF.Diagnostics, CGEF.Pipeline)
        for extname in (
            :CoarseGrainingEnergyFluxesFFTWExt,
            :CoarseGrainingEnergyFluxesFINUFFTExt,
            :CoarseGrainingEnergyFluxesFastSphericalHarmonicsExt,
            :CoarseGrainingEnergyFluxesNUFSHTExt,
            :CoarseGrainingEnergyFluxesOhMyThreadsExt,
            :CoarseGrainingEnergyFluxesDistributedExt,
            :CoarseGrainingEnergyFluxesGPUExt,
            :CoarseGrainingEnergyFluxesMPIExt,
        )
            ext = Base.get_extension(CGEF, extname)
            ext === nothing || (jet_targets = (jet_targets..., ext))
        end

        geom = FG.Geometry.CartesianGeometry()
        x = collect(0.0:1000.0:20e3)
        y = collect(0.0:1000.0:20e3)
        grid = FG.Grids.StructuredGrid(geom, x, y, trues(length(x), length(y)))
        field = rand(length(x), length(y))
        out = zeros(size(field))
        kern = CGEF.TopHatKernel()
        scale = 5000.0

        # Footprint build + the convolution apply must be type-stable.
        JET.@test_opt target_modules = jet_targets CGEF.Filtering.build_footprint(grid, kern, scale)
        fp = CGEF.Filtering.build_footprint(grid, kern, scale)
        JET.@test_opt target_modules = jet_targets CGEF.Filtering.apply_footprint!(out, field, grid, fp, CGEF.Filtering.Deformable(), false, false)
        JET.@test_opt target_modules = jet_targets CGEF.Filtering.apply_footprint!(out, field, grid, fp, CGEF.Filtering.ZeroFill(), false, false)

        # The serial public entry point is type-stable too.
        JET.@test_opt target_modules = jet_targets CGEF.Filtering.filter_field!(out, field, grid, kern, scale; backend = CGEF.ComputationalBackends.SerialBackend())

        # The cache-strategy dispatch arms (cached vs streaming ScatteredFilterPlan),
        # filter_apply_batch!, and the separable-Gaussian fast path must all be type-stable too.
        x_nu = collect(0.0:1000.0:20e3) .+ [iseven(i) ? 4.0 : -3.0 for i in 1:21]
        grid_nu = FG.Grids.StructuredGrid(geom, x_nu, x_nu, trues(21, 21))
        JET.@test_opt target_modules = jet_targets CGEF.Filtering.build_footprint(grid_nu, kern, scale; cache_strategy = CGEF.Filtering.AlwaysCache())
        JET.@test_opt target_modules = jet_targets CGEF.Filtering.build_footprint(grid_nu, kern, scale; cache_strategy = CGEF.Filtering.NeverCache())
        fp_cached = CGEF.Filtering.build_footprint(grid_nu, kern, scale; cache_strategy = CGEF.Filtering.AlwaysCache())
        fp_stream = CGEF.Filtering.build_footprint(grid_nu, kern, scale; cache_strategy = CGEF.Filtering.NeverCache())
        field_nu = rand(21, 21); out_nu = zeros(21, 21)
        JET.@test_opt target_modules = jet_targets CGEF.Filtering.apply_footprint!(out_nu, field_nu, grid_nu, fp_cached, CGEF.Filtering.Deformable(), false, false)
        JET.@test_opt target_modules = jet_targets CGEF.Filtering.apply_footprint!(out_nu, field_nu, grid_nu, fp_stream, CGEF.Filtering.Deformable(), false, false)

        plan_cached = CGEF.Filtering.plan_filter(grid_nu, kern, scale; cache_strategy = CGEF.Filtering.AlwaysCache())
        plan_stream = CGEF.Filtering.plan_filter(grid_nu, kern, scale; cache_strategy = CGEF.Filtering.NeverCache())
        outs_nu = (zeros(21, 21), zeros(21, 21))
        fields_nu = (rand(21, 21), rand(21, 21))
        JET.@test_opt target_modules = jet_targets CGEF.Filtering.filter_apply_batch!(outs_nu, fields_nu, plan_cached)
        JET.@test_opt target_modules = jet_targets CGEF.Filtering.filter_apply_batch!(outs_nu, fields_nu, plan_stream)

        gplan = CGEF.Filtering.plan_filter(grid, CGEF.GaussianKernel(), scale)
        Test.@test gplan.footprint isa CGEF.Filtering.SeparableGaussianFootprint
        JET.@test_opt target_modules = jet_targets CGEF.Filtering.build_footprint(grid, CGEF.GaussianKernel(), scale)
        JET.@test_opt target_modules = jet_targets CGEF.Filtering.filter_apply!(out, field, gplan)
        JET.@test_opt target_modules = jet_targets CGEF.Filtering.filter_apply_batch!(outs_nu, fields_nu, gplan)
    end
end
