
Test.@testset "Aqua" begin
    Aqua.test_all(CGEF; ambiguities = false, unbound_args = (VERSION >= v"1.12"))
end

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
