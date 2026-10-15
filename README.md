# get_eo

Earth-observation and driver data as `xarray`, for crop growth models and
data assimilation. Point at an AOI and a period; get lazy Sentinel-2 and
ECOSTRESS observations on their native tile grid, plus hourly ERA5
meteorology and SoilGrids soils, all with consistent names, units and
metadata.

```python
import get_eo as geo

SITE = (-96.4701, 41.1649)                      # (lon, lat); or a GeoJSON path, WKT, bbox, ...
obs  = geo.open_aoi(SITE, "2020-01-01", "2020-12-31", buffer_m=1500)
met  = geo.open_era5(SITE, "2020-01-01", "2020-12-31")      # hourly
soil = geo.open_soilgrids(SITE, buffer_m=1500)               # 6 depths, 250 m
dem  = geo.open_dem(SITE, buffer_m=1500)                     # elevation, slope, aspect, 30 m

s2, eco = obs["s2"], obs["ecostress"]           # dask-backed, nothing downloaded yet
```

A worked site-year (US-Ne2, 2020) is in [`examples/`](examples/README.md).

## Install

```bash
pip install .                 # Python >= 3.12
pip install ".[gee]"          # Earth Engine: Sentinel-2 and ERA5-Land (adds xee)
pip install ".[cdse]"         # Copernicus Data Space Sentinel-2 (adds boto3)
pip install ".[cds]"          # unmodified ERA5 from the Climate Data Store
pip install ".[all]"
```

A conda-forge environment is the easiest way to get a consistent GDAL stack
for `rasterio`.

## What it returns

### `open_aoi(aoi, start, end, sensors=("s2", "ecostress"), source="aws", ...)`

A dict of lazy `xr.Dataset`s: `s2` (L2A surface reflectance, 10 bands, 10 m),
`ecostress` (L1CT TOA radiance, TIR bands 2/4/5, 70 m) and `ecostress_lste`
(L2T land surface temperature).

* **One MGRS tile per call**, chosen as the tile that best contains the AOI
  or set with `tile="14TQL"`. Nothing is reprojected on read, so radiometry
  is untouched. The window is snapped outwards to 140 m so one ECOSTRESS
  70 m cell is exactly 7 x 7 Sentinel-2 pixels (`geo.aggregate_to`).
* **Variables**: `obs`, `obs_unc` (parametric uncertainty), `valid` (SCL
  mask), `scl`, coarse-grid `sza/saa/vza/vaa` (azimuths clockwise from true
  north), and `inside_aoi` (fraction of each pixel inside the polygon).
  The full schema is in [`docs/reframe.md`](docs/reframe.md) and enforced by
  `geo.validate_obs_dataset`.
* **Sentinel-2 `source=`**

  | `source=` | Reads | Needs |
  |-----------|-------|-------|
  | `aws` (default) | Earth Search COGs | nothing |
  | `planetary` | Planetary Computer COGs | nothing (SAS tokens fetched) |
  | `cdse` | Copernicus Data Space JP2s over S3 | `CDSE_S3_ACCESS_KEY` / `CDSE_S3_SECRET_KEY`, `[cdse]` |
  | `gee` | `S2_SR_HARMONIZED` via xee, on the same tile grid | `[gee]`, `earthengine authenticate`, `gee_project=` |

  All four give the same variables, grid and reflectance scale (GEE
  carries one mean sun/view angle per image rather than a grid).
* **ECOSTRESS** always comes from NASA Earthdata and needs
  `machine urs.earthdata.nasa.gov login ... password ...` in `~/.netrc`.

### `open_era5(aoi, start, end, source="openmeteo", co2_ppm=400.0)`

Hourly `t_air` (degC), `rh` (0-1), `wind_speed` (m/s), `solar_radiation`
(W/m2), `precipitation` (mm), `p_atm` (Pa) and `co2_ppm` on a `time`
dimension, for the grid cell at the AOI centroid.

| `source=` | Dataset | Notes |
|-----------|---------|-------|
| `openmeteo` (default) | ERA5, 0.25 deg | no account; elevation-adjusted and rounded by Open-Meteo |
| `gee` | ERA5-Land, 0.1 deg | unmodified; `[gee]` |
| `cds` | ERA5, 0.25 deg | unmodified; `[cds]` and a current-format key in `~/.cdsapirc` (`url: https://cds.climate.copernicus.eu/api`) |

Timestamps are UTC; solar radiation and precipitation are over the hour
ending at the timestamp. ERA5 has no CO2, so `co2_ppm` is a constant.

`soil_temp=True` adds `t_soil_1..4` (degC), the four ERA5 soil layers
(0-7, 7-28, 28-100 and 100-289 cm; layer 3 is the usual deep-soil boundary
condition). Each carries `depth_top_cm` / `depth_bottom_cm` attributes.

### `open_soilgrids(aoi, properties=..., depths=..., stats=("mean",))`

SoilGrids 2.0 at 250 m: clay, sand, silt, bdod, cfvo, soc, nitrogen, phh2o,
cec, ocd at 0-5, 5-15, 15-30, 30-60, 60-100 and 100-200 cm, in conventional
units, on a UTM grid aligned to multiples of 250 m (nearest neighbour).
Quantiles and uncertainty via `stats`. Small, so loaded eagerly.

### `open_dem(aoi, source="aws", res=30)`

Copernicus GLO-30 elevation (m), plus `slope` and `aspect` (degrees; aspect
is the direction the slope faces, clockwise from north) on a UTM grid
aligned to multiples of `res`, bilinear. `source="aws"` reads the public
COGs (no account); `source="gee"` uses `COPERNICUS/DEM/GLO30_2024_1`. The
two agree to a few centimetres. Use `elevation` to correct ERA5 temperature
and pressure from the grid cell to the field.

## Notes on the data

* **Sentinel-2 on AWS** already has the -1000 DN BOA offset applied in the
  Earth Search COGs for every item, whatever `earthsearch:boa_offset_applied`
  says (checked against Planetary Computer, 2019-2025), so reflectance is
  `DN / 1e4` there and `(DN - 1000) / 1e4` on Planetary and CDSE.
* **Duplicates** (reprocessed granules) are collapsed to one item per
  acquisition and tile, keeping the latest processing.
* **Broken archive files**: an item whose granule metadata is truncated at
  the source is skipped with a warning rather than failing the request.
* **Caching**: `geo.to_zarr_cache(ds, path)` / `geo.open_cache(path)`.

## Logging

get_eo reports what it is doing at INFO level by default (searches and hit
counts, which tile was chosen, remote fetches with timings, retries). Change it with:

```python
import get_eo as geo
geo.set_verbosity("WARNING")  # quieten; or set GET_EO_LOG=WARNING
geo.set_verbosity("DEBUG")    # very noisy: every window read + GDAL HTTP tracing
```

`DEBUG` is for diagnosing network, proxy and credential problems. Signed-URL
tokens (`sig=`, `token=`, `X-Amz-*`) are redacted from the output. Messages go
to the standard `get_eo` logger, so you can also attach your own handlers.

## Credentials summary

| Service | Where |
|---------|-------|
| NASA Earthdata (ECOSTRESS) | `~/.netrc` |
| CDSE S3 | env vars, or `cdse` section of `~/.get_eo/config.json` (`~/.eof/config.json` also read) |
| Earth Engine | `earthengine authenticate`; `gee_project=` or `EOF_GEE_PROJECT` / `GOOGLE_CLOUD_PROJECT` |
| CDS | `~/.cdsapirc` |

## Development

```bash
pip install -e ".[dev]"
pytest                 # offline tests
pytest -m network      # live tests (internet, Earthdata login)
```

Licence: AGPL-3.0-or-later (derived from [eof](https://github.com/profLewis/eof)).
