module CoarseGrainingEnergyFluxesNCDatasetsExt

using CoarseGrainingEnergyFluxes
using NCDatasets

export load_grid_netcdf, save_result_netcdf

"""
    load_grid_netcdf(filepath, lon_name, lat_name, mask_name, geom)

Load coordinate arrays and a boolean mask from a NetCDF file and construct a `StructuredGrid`.
"""
function load_grid_netcdf(
    filepath::AbstractString,
    lon_name::AbstractString,
    lat_name::AbstractString,
    mask_name::AbstractString,
    geom::AbstractGeometry{T}
) where {T<:AbstractFloat}
    
    NCDataset(filepath, "r") do ds
        lon = Array{T}(ds[lon_name][:])
        lat = Array{T}(ds[lat_name][:])
        
        # Land masks are usually saved as Byte/Int8 with values 0=land, 1=water, or similar
        raw_mask = ds[mask_name][:, :]
        mask = BitMatrix(raw_mask .> zero(eltype(raw_mask)))
        
        return StructuredGrid(geom, lon, lat, mask)
    end
end

"""
    save_result_netcdf(filepath, res, grid)

Save a `CoarseGrainResult` to a NetCDF file, including scale dimensions, spectra, and full 2D flux maps.
"""
function save_result_netcdf(
    filepath::AbstractString,
    res::CoarseGrainResult{T},
    grid::StructuredGrid{G,T}
) where {T<:AbstractFloat, G}
    
    Nlon, Nlat = size_tuple(grid)
    Nscales = length(res.scales)
    
    # Create a new NetCDF file
    NCDataset(filepath, "c") do ds
        # Define dimensions
        defDim(ds, "lon", Nlon)
        defDim(ds, "lat", Nlat)
        defDim(ds, "scale", Nscales)
        
        # Define coordinate variables
        v_lon = defVar(ds, "lon", T, ("lon",))
        v_lon[:] = grid.lon
        v_lon.attrib["units"] = G <: SphericalGeometry ? "radians" : "meters"
        
        v_lat = defVar(ds, "lat", T, ("lat",))
        v_lat[:] = grid.lat
        v_lat.attrib["units"] = G <: SphericalGeometry ? "radians" : "meters"
        
        v_scale = defVar(ds, "scale", T, ("scale",))
        v_scale[:] = res.scales
        v_scale.attrib["units"] = "meters"
        v_scale.attrib["long_name"] = "Filter scale (characteristic width ℓ)"
        
        # Define diagnostics
        v_spectrum = defVar(ds, "energy_spectrum", T, ("scale",))
        v_spectrum[:] = res.spectrum
        v_spectrum.attrib["long_name"] = "Filtering energy spectrum E(ℓ)"
        v_spectrum.attrib["units"] = "m²/s²"
        
        # Define 3D energy flux maps: (Nlon, Nlat, Nscales)
        v_Pi = defVar(ds, "Pi", T, ("lon", "lat", "scale"))
        for s in 1:Nscales
            v_Pi[:, :, s] = res.Pi[s]
        end
        v_Pi.attrib["long_name"] = "Cross-scale kinetic energy flux (Π)"
        v_Pi.attrib["units"] = "W/m³"
    end
    
    return filepath
end

end # module
