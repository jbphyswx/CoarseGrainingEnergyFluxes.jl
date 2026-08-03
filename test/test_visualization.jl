
# The CairoMakie viz functions are parent-owned stubs; without the extension loaded they must
# raise a helpful error (the real methods are exercised when `using CairoMakie`).
Test.@testset "Visualization stubs" begin
    Test.@test isdefined(CGEF, :plot_Π_map)
    Test.@test isdefined(CGEF, :plot_spectrum)
    Test.@test_throws ArgumentError CGEF.plot_Π_map(nothing, 1, nothing)
    Test.@test_throws ArgumentError CGEF.plot_spectrum(nothing)
end


# Allocation regression tests for every core hot-path method (real-space + spectral
# filter_apply!, ddx!/ddy!/ddz!, compute_Π! per grid type, coarse_grain!/cumulative_energy!'s
# repeated-sweep entry points, parallel backends) — see test_allocs.jl's own header for exactly
# what's asserted as zero vs. bounded-and-why.
