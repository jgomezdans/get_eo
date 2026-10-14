"""US-Ne2 (Mead, Nebraska; irrigated maize-soybean rotation), 2020.

Everything a crop-model / assimilation run needs for one site-year:
Sentinel-2 reflectance, ECOSTRESS radiance, hourly ERA5 meteorology and
SoilGrids soils. Run from the repo root:  python examples/us_ne2_2020.py
"""
import get_eo as geo

SITE = (-96.4701, 41.1649)          # (lon, lat) of the flux tower
START, END = "2020-01-01", "2020-12-31"

# --- Meteorology: hourly ERA5, same variables as the AmeriFlux reader ----
met = geo.open_era5(SITE, START, END, soil_temp=True)   # source="openmeteo" (no account)
# met = geo.open_era5(SITE, START, END, source="gee")   # ERA5-Land via Earth Engine
print(met)

# --- Soils: SoilGrids 250 m, six depths, conventional units --------------
soil = geo.open_soilgrids(SITE, buffer_m=1500)
print(soil[["clay", "sand", "soc", "bdod", "phh2o"]])

# --- Terrain: Copernicus GLO-30 elevation, slope, aspect ----------------
dem = geo.open_dem(SITE, buffer_m=1500)
print(float(dem.elevation.mean()), "m a.s.l.,", float(dem.slope.mean()), "deg slope")

# --- Sentinel-2 + ECOSTRESS: lazy, on the native tile grid ---------------
# ECOSTRESS needs an Earthdata login in ~/.netrc.
obs = geo.open_aoi(SITE, START, END, buffer_m=1500)   # source="aws" by default
s2, eco = obs["s2"], obs["ecostress"]
print(s2)

# Nothing has been downloaded yet. Pull a field-mean NDVI series for the
# scenes that are clear over the field (reads only B04, B08 and SCL).
red, nir = s2.obs.sel(band="B04"), s2.obs.sel(band="B08")
ndvi = (nir - red) / (nir + red)
inside = s2.inside_aoi > 0.5
clear = s2.valid.where(inside).mean(("x", "y")).compute()
good = clear.values > 0.9
ndvi_ts = ndvi.isel(time=good).where(inside).mean(("x", "y")).compute()
print(ndvi_ts.to_series().round(2).to_string())

# S2 averaged onto the ECOSTRESS 70 m grid, ready to compare pixel by pixel
s2_70m = geo.aggregate_to(s2.isel(time=good[:s2.sizes["time"]]), eco)
print(s2_70m)

# Keep a local copy so the remote reads are paid for once
# geo.to_zarr_cache(s2.isel(time=good), "us-ne2_2020_s2.zarr")
