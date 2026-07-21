module CoarseGrainingEnergyFluxesGPUExt

# `@kernel`/`@index`/`@Const` are imported bare (not qualified as KA.@kernel etc.) as a verified,
# necessary exception: KernelAbstractions' `@kernel` macro does AST pattern-matching on the literal
# unqualified `@Const`/`@index` syntax during its own expansion — qualifying them breaks precompilation
# with a real MethodError (confirmed directly, not assumed), so this is not a style choice.
using KernelAbstractions: KernelAbstractions as KA, @kernel, @index, @Const
using CoarseGrainingEnergyFluxes: CoarseGrainingEnergyFluxes as CGEF

# GPUBackend: apply the precomputed footprint with one GPU thread per output cell. The footprint
# (offset + weight arrays) and the mask are moved to the device once; the kernel mirrors the serial
# per-row logic exactly (so on the KernelAbstractions CPU backend it matches the serial result).
# `is_zerofill` selects the masking branch as a plain Bool (kernels can't dispatch on types).

@kernel function _cgef_filter_kernel!(
    out, @Const(field), @Const(mask), @Const(di), @Const(dj), @Const(w), @Const(ptr),
    nbands::Int, periodic::Bool, is_zerofill::Bool,
)
    i, j = @index(Global, NTuple)
    T = eltype(out)
    Nlon, Nlat = size(out)
    if i <= Nlon && j <= Nlat
        if mask[i, j]
            b = nbands == 1 ? 1 : j
            lo = ptr[b]
            hi = ptr[b+1] - 1
            ws = zero(T)
            wn = zero(T)
            for k in lo:hi
                jj = j + dj[k]
                if 1 <= jj <= Nlat
                    ii = i + di[k]
                    inbounds = true
                    if ii < 1 || ii > Nlon
                        if periodic
                            ii = mod1(ii, Nlon)
                        else
                            inbounds = false
                        end
                    end
                    if inbounds
                        active = mask[ii, jj]
                        wk = w[k]
                        if is_zerofill
                            wn += wk
                            if active
                                ws += wk * field[ii, jj]
                            end
                        elseif active
                            wn += wk
                            ws += wk * field[ii, jj]
                        end
                    end
                end
            end
            out[i, j] = wn > T(1e-15) ? ws / wn : zero(T)
        else
            out[i, j] = zero(T)
        end
    end
end

# Scattered/nonuniform-axis variant: `ii_arr`/`jj_arr` hold ABSOLUTE neighbour indices (periodic wrap
# already resolved at footprint-build time, per FilterFootprintScattered's convention), so there's no
# offset arithmetic or bounds/wrap branch here — mirrors `apply_footprint_row!`'s scattered method.
@kernel function _cgef_filter_kernel_scattered!(
    out, @Const(field), @Const(mask), @Const(ii_arr), @Const(jj_arr), @Const(w), @Const(ptr),
    is_zerofill::Bool,
)
    i, j = @index(Global, NTuple)
    T = eltype(out)
    Nlon, Nlat = size(out)
    if i <= Nlon && j <= Nlat
        if mask[i, j]
            t = i + (j - 1) * Nlon
            lo = ptr[t]
            hi = ptr[t+1] - 1
            ws = zero(T)
            wn = zero(T)
            for k in lo:hi
                ii = ii_arr[k]
                jj = jj_arr[k]
                active = mask[ii, jj]
                wk = w[k]
                if is_zerofill
                    wn += wk
                    if active
                        ws += wk * field[ii, jj]
                    end
                elseif active
                    wn += wk
                    ws += wk * field[ii, jj]
                end
            end
            out[i, j] = wn > T(1e-15) ? ws / wn : zero(T)
        else
            out[i, j] = zero(T)
        end
    end
end

move(dev, x) = (y = KA.allocate(dev, eltype(x), size(x)); copyto!(y, x); y)

# Real dispatch (not a runtime check) on which footprint type `build_footprint` returned — uniform
# axes get the offset-based kernel, nonuniform axes get the absolute-index scattered kernel.
function _run_gpu_kernel!(dev, out, field, maskd, fp::CGEF.Filtering.FilterFootprint, periodic::Bool, is_zerofill::Bool)
    Nlon, Nlat = size(out)
    di = move(dev, fp.di); dj = move(dev, fp.dj); w = move(dev, fp.w); ptr = move(dev, fp.ptr)
    _cgef_filter_kernel!(dev)(out, field, maskd, di, dj, w, ptr, fp.nbands, periodic, is_zerofill; ndrange = (Nlon, Nlat))
end

function _run_gpu_kernel!(dev, out, field, maskd, fp::CGEF.Filtering.FilterFootprintScattered, periodic::Bool, is_zerofill::Bool)
    Nlon, Nlat = size(out)
    ii = move(dev, fp.ii); jj = move(dev, fp.jj); w = move(dev, fp.w); ptr = move(dev, fp.ptr)
    _cgef_filter_kernel_scattered!(dev)(out, field, maskd, ii, jj, w, ptr, is_zerofill; ndrange = (Nlon, Nlat))
end

# Grid-generic (see the OhMyThreadsExt comment): only `grid.mask` and `isperiodic` are used below,
# both of which already work for CurvilinearGrid (isperiodic falls back to `false` — no periodicity
# flags on a curvilinear mesh — via `AbstractGrid`'s default method).
function CGEF.Filtering.gpu_filter_field!(
    gpu_backend::CGEF.Backends.GPUBackend,
    out::AbstractMatrix{T},
    field::AbstractMatrix{T},
    grid::Union{CGEF.StructuredGrid{G,T,2}, CGEF.CurvilinearGrid{T,G}},
    kernel::CGEF.Kernels.AbstractFilterKernel,
    scale::T,
    mask_strategy::CGEF.Filtering.AbstractMaskStrategy,
    workspace,
) where {T<:AbstractFloat, G<:CGEF.Geometry.AbstractGeometry{T}}
    dev = gpu_backend.backend
    # `workspace`, when supplied by a cached `PhysicalFilterPlan`, IS the already-built CPU footprint
    # — reused instead of rebuilding it every call. The per-call DEVICE upload below (mask + footprint
    # arrays) is a separate, smaller remaining inefficiency — not yet cached across calls.
    fp = workspace === nothing ? CGEF.Filtering.build_footprint(grid, kernel, scale) : workspace

    # Move the mask to the device once (out/field are assumed already on it); the footprint arrays
    # are moved inside `_run_gpu_kernel!`, dispatching on which fields `fp` actually has.
    maskd = move(dev, Array{Bool}(grid.mask))
    is_zerofill = mask_strategy isa CGEF.Filtering.ZeroFill
    _run_gpu_kernel!(dev, out, field, maskd, fp, CGEF.Grids.isperiodic(grid, 1), is_zerofill)
    KA.synchronize(dev)
    return out
end

end # module
