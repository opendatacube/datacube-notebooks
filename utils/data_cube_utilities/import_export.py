import time
import numpy as np
from . import dc_utilities

## Export ##

def export_slice_to_geotiff(ds, path):
    """
    Exports a single slice of an xarray.Dataset as a GeoTIFF.
    
    ds: xarray.Dataset
        The Dataset to export.
    path: str
        The path to store the exported GeoTIFF.
    """
    kwargs = dict(tif_path=path, dataset=ds.astype(np.float32), bands=list(ds.data_vars.keys()))
    if 'crs' in ds.attrs:
        kwargs['crs'] = str(ds.attrs['crs'])
    dc_utilities.write_geotiff_from_xr(**kwargs)

def export_xarray_to_geotiff(ds, path):
    """
    Exports an xarray.Dataset as individual time slices.
    
    Parameters
    ----------
    ds: xarray.Dataset
        The Dataset to export.
    path: str
        The path prefix to store the exported GeoTIFFs. For example, 'geotiffs/mydata' would result in files named like
        'mydata_2016_12_05_12_31_36.tif' within the 'geotiffs' folder.
    """
    def time_to_string(t):
        return time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime(t.astype(int)/1000000000))
    
    for t in ds.time:
        time_slice_xarray = ds.sel(time = t)
        export_slice_to_geotiff(time_slice_xarray,
                                path + "_" + time_to_string(t) + ".tif")

## End export ##