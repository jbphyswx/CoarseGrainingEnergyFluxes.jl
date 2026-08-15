# Parallel scaling harness.
#
# Reports speedup AND the machine's own ceiling for the same access pattern, so a number can be read
# as a fraction of what the hardware can actually deliver rather than against an ideal `n×` that a
# memory-bound kernel can never reach. A filtering sweep over scattered neighbours is a gather: it is
# latency-bound, and on a desktop its ceiling is roughly half the core count.
#
# Run once per thread count, e.g.
#   for t in 1 2 4 8; do julia --project=benchmark -t$t benchmark/scaling.jl; done
#
# Wall-clock lives here and never in `test/`: it is load- and GC-sensitive, and the suite must stay
# deterministic.

using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF
using FlowGeometries: FlowGeometries as FG
using ComputationalBackends: ComputationalBackends as CB
using NearestNeighbors: NearestNeighbors
using OhMyThreads: OhMyThreads
using Random: Random

const F = CGEF.Filtering

best_ms(f, reps) = minimum(begin t0 = time_ns(); f(); (time_ns() - t0) / 1e6 end for _ in 1:reps)

# --- machine ceilings -------------------------------------------------------------------------
# Compute-bound: what perfectly parallel arithmetic achieves here.
function ceiling_compute()
    f(x) = (s = 0.0; for i in 1:20_000; s += sin(x + i) * cos(x - i); end; s)
    w() = OhMyThreads.tmapreduce(f, +, 1:256; scheduler = OhMyThreads.DynamicScheduler())
    w()
    return best_ms(w, 5)
end

# Memory-bound: the gather pattern the real-space engines actually have. This is the ceiling that
# applies to filtering, not the compute one.
function ceiling_gather(; n = 600_000, k = 12)
    Random.seed!(1)
    src = randn(n); idx = rand(1:n, n * k); dst = zeros(n)
    function g!()
        OhMyThreads.tforeach(eachindex(dst); scheduler = OhMyThreads.DynamicScheduler()) do t
            s = 0.0; b = (t - 1) * k
            @inbounds for j in 1:k; s += src[idx[b + j]]; end
            @inbounds dst[t] = s
        end
    end
    g!()
    return best_ms(g!, 10)
end

# --- the workload: many independent ragged slices ----------------------------------------------
function slice_workload(; nslices = 256, dx = 1000.0, scale = 8000.0)
    Random.seed!(20260814)
    cart = FG.Geometry.CartesianGeometry()
    ker = CGEF.GaussianKernel()
    counts = rand(1232:3581, nslices)
    plans = map(counts) do n
        xs = rand(n) .* (dx * 60); ys = rand(n) .* (dx * 60)
        g = FG.Grids.UnstructuredGrid(cart, xs, ys, trues(n); k = 12, areas = fill(dx * dx, n))
        F.plan_filter(g, ker, scale; backend = CB.SerialBackend(), method = F.RealSpace())
    end
    fields = [randn(n) for n in counts]
    outs = [zeros(n) for n in counts]
    return outs, fields, plans, counts
end

function main()
    nt = Threads.nthreads()
    outs, fields, plans, counts = slice_workload()
    F.filter_slices!(outs, fields, plans; backend = CB.SerialBackend())
    F.filter_slices!(outs, fields, plans; backend = CB.ThreadedBackend())

    t_ser = best_ms(() -> F.filter_slices!(outs, fields, plans; backend = CB.SerialBackend()), 5)
    t_thr = best_ms(() -> F.filter_slices!(outs, fields, plans; backend = CB.ThreadedBackend()), 5)
    c_cmp = ceiling_compute()
    c_gat = ceiling_gather()

    println("nthreads=", nt,
            "  slices=", length(counts), "  points=", sum(counts),
            "  N∈[", minimum(counts), ",", maximum(counts), "]")
    println("  filter_slices! serial   ", round(t_ser, digits = 1), " ms")
    println("  filter_slices! threaded ", round(t_thr, digits = 1), " ms   speedup ",
            round(t_ser / t_thr, digits = 2), "x")
    println("  ceiling compute-bound   ", round(c_cmp, digits = 1), " ms")
    println("  ceiling gather-bound    ", round(c_gat, digits = 2), " ms")
    println("  -> report speedup as a fraction of the GATHER ceiling measured at the same thread",
            " count; the compute ceiling does not bound this kernel.")
    return nothing
end

main()
