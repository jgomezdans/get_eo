"""get_eo: Earth-observation and driver data as xarray, for crop models.

Lazy access to Sentinel-2 and ECOSTRESS over an AOI, plus hourly ERA5
meteorology and SoilGrids soils::

    import get_eo as geo

    data = geo.open_aoi("field.geojson", "2022-06-01", "2022-08-31")
    data["s2"]           # Sentinel-2 L2A surface reflectance
    data["ecostress"]    # ECOSTRESS L1CT TOA radiance (bands 2, 4, 5)
    data["ecostress_lste"]  # ECOSTRESS L2T land surface temperature

    met = geo.open_era5("field.geojson", "2022-06-01", "2022-08-31")  # hourly
    soil = geo.open_soilgrids("field.geojson")   # clay, sand, ... by depth
    dem = geo.open_dem("field.geojson")          # elevation, slope, aspect

Satellite data are dask-backed; nothing is downloaded until ``.compute()``.
Meteorology and soils are small and loaded eagerly.
Use :func:`to_zarr_cache` to keep a local copy.

INFO-level progress is logged to stderr by default;
``geo.set_verbosity("WARNING")`` (or ``GET_EO_LOG=WARNING``) quietens it and
``"DEBUG"`` adds every read and GDAL's HTTP tracing.
"""

from importlib.metadata import PackageNotFoundError, version

import rioxarray  # noqa: F401  (registers the .rio accessor)

from get_eo._aggregate import aggregate_to
from get_eo._cache import open_cache, to_zarr_cache
from get_eo._dem import open_dem
from get_eo._geometry import angles_at
from get_eo._log import set_verbosity
from get_eo._meteo import open_era5
from get_eo._open import open_aoi
from get_eo._schema import validate_obs_dataset
from get_eo._soil import open_soilgrids

try:
    __version__ = version("get_eo")
except PackageNotFoundError:    # running from a source tree
    __version__ = "0.0.0"

__all__ = [
    "__version__", "open_aoi", "open_era5", "open_soilgrids", "open_dem", "aggregate_to", "angles_at",
    "to_zarr_cache", "open_cache", "validate_obs_dataset", "set_verbosity",
]
