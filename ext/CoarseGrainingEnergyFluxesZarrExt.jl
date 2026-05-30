module CoarseGrainingEnergyFluxesZarrExt

using CoarseGrainingEnergyFluxes
using Zarr
using DiskArrays

export save_result_zarr

"""
    save_result_zarr(store_path, res, grid)

Save a `CoarseGrainResult` directly into a Zarr directory store.
"""
function save_result_zarr(
    store_path::AbstractString,
    res::CoarseGrainResult{T},
    grid::StructuredGrid{G,T}
) where {T<:AbstractFloat, G}
    
    Nlon, Nlat = size_tuple(grid)
    Nscales = length(res.scales)
    
    # 1. Open or create a Zarr directory store
    store = zgroup(store_path)
    
    # 2. Write coordinate dimensions
    zcreate(store, "lon", T, (Nlon,), attrs=Dict("units" => G <: SphericalGeometry ? "radians" : "meters"))
    store["lon"][:] = grid.lon
    
    zcreate(store, "lat", T, (Nlat,), attrs=Dict("units" => G <: SphericalGeometry ? "radians" : "meters"))
    store["lat"][:] = grid.lat
    
    zcreate(store, "scale", T, (Nscales,), attrs=Dict("units" => "meters", "long_name" => "Filter scale"))
    store["scale"][:] = res.scales
    
    # 3. Write filtering energy spectrum
    zcreate(store, "energy_spectrum", T, (Nscales,), attrs=Dict("units" => "m²/s²", "long_name" => "Filtering spectrum"))
    store["energy_spectrum"][:] = res.spectrum
    
    # 4. Write multiscale 3D Pi maps: shape (Nlon, Nlat, Nscales), chunked by scales
    Pi_arr = zcreate(store, "Pi", T, (Nlon, Nlat, Nscales), chunks=(Nlon, Nlat, 1),
                     attrs=Dict("units" => "W/m³", "long_name" => "Kinetic energy flux (Π)"))
    
    for s in 1:Nscales
        Pi_arr[:, :, s] = res.Pi[s]
    end
    
    return store_path
end

end # module
