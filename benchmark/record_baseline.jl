# Local before/after snapshot of the benchmark suite, for comparing two states of the tree on the same
# machine at the same thread count.
#
#   julia --project=benchmark -t1 benchmark/record_baseline.jl before
#   julia --project=benchmark -t1 benchmark/record_baseline.jl after
#
# Output is gitignored: absolute times belong to the machine that produced them. The portable guards
# are the allocation bounds in the test suite and the ceiling ratios in `scaling.jl`.
using BenchmarkTools: BenchmarkTools

const OUT_DIR = joinpath(@__DIR__, "baseline")

# Top level, not inside `main`: `include` defines `SUITE` in a newer world age than a function body
# reading it, which Julia 1.12 warns about and will make an error.
include(joinpath(@__DIR__, "benchmarks.jl"))

function main()
    stem = isempty(ARGS) ? "snapshot" : first(ARGS)
    mkpath(OUT_DIR)
    out = joinpath(OUT_DIR, string(stem, "-t", Threads.nthreads(), ".json"))

    BenchmarkTools.tune!(SUITE)
    results = BenchmarkTools.run(SUITE; verbose = true)
    BenchmarkTools.save(out, results)

    println()
    println(rpad("entry", 62), rpad("min time", 14), rpad("allocs", 10), "memory")
    println("-"^100)
    for (keys, trial) in sort(BenchmarkTools.leaves(results); by = first)
        m = minimum(trial)
        println(rpad(join(keys, "/"), 62),
                rpad(BenchmarkTools.prettytime(BenchmarkTools.time(m)), 14),
                rpad(string(BenchmarkTools.allocs(m)), 10),
                BenchmarkTools.prettymemory(BenchmarkTools.memory(m)))
    end
    println()
    println("Compare with: judge(minimum(load(\"after-t1.json\")[1]), minimum(load(\"before-t1.json\")[1]))")
    return nothing
end

main()
