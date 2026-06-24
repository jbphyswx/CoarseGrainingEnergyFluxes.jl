module CoarseGrainingEnergyFluxesGPUExt

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
                        wet = mask[ii, jj]
                        wk = w[k]
                        if is_zerofill
                            wn += wk
                            if wet
                                ws += wk * field[ii, jj]
                            end
                        elseif wet
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

function CGEF.Filtering.gpu_filter_field!(
    gpu_backend::CGEF.GPUBackend,
    out::AbstractMatrix{T},
    field::AbstractMatrix{T},
    grid::CGEF.StructuredGrid{G,T},
    kernel::CGEF.AbstractFilterKernel,
    scale::T,
    mask_strategy::CGEF.AbstractMaskStrategy,
    workspace,
) where {T<:AbstractFloat, G<:CGEF.AbstractGeometry{T}}
    dev = gpu_backend.backend
    fp = CGEF.Filtering.build_footprint(grid, kernel, scale)
    Nlon, Nlat = CGEF.size_tuple(grid)

    # Move the footprint + mask to the device (out/field are assumed already on it).
    move(x) = (y = KA.allocate(dev, eltype(x), size(x)); copyto!(y, x); y)
    di = move(fp.di)
    dj = move(fp.dj)
    w = move(fp.w)
    ptr = move(fp.ptr)
    maskd = move(Array{Bool}(grid.mask))

    is_zerofill = mask_strategy isa CGEF.ZeroFill
    _cgef_filter_kernel!(dev)(
        out, field, maskd, di, dj, w, ptr, fp.nbands, CGEF.isperiodic(grid, 1), is_zerofill;
        ndrange = (Nlon, Nlat),
    )
    KA.synchronize(dev)
    return out
end

end # module
