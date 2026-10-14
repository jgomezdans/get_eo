# Example: one site-year of drivers and observations (US-Ne2, 2020)

`us_ne2_2020.py` collects, for the AmeriFlux site US-Ne2 (Mead, Nebraska,
irrigated maize-soybean rotation) over calendar year 2020:

| What | Call | Result |
|------|------|--------|
| Hourly meteorology (ERA5) | `geo.open_era5(SITE, START, END)` | `time: 8784`, `t_air rh wind_speed solar_radiation precipitation p_atm co2_ppm` |
| Soils (SoilGrids 250 m) | `geo.open_soilgrids(SITE, buffer_m=1500)` | `depth: 6, y: 13, x: 14`, clay/sand/silt/bdod/cfvo/soc/nitrogen/phh2o/cec |
| Sentinel-2 L2A + ECOSTRESS (lazy) | `geo.open_aoi(SITE, START, END, buffer_m=1500)` | dict with `s2` (144 scenes, 10 bands, 322 x 322 px at 10 m) and `ecostress` |

```python
import get_eo as geo

SITE = (-96.4701, 41.1649)          # (lon, lat) of the flux tower
START, END = "2020-01-01", "2020-12-31"

met  = geo.open_era5(SITE, START, END)             # or source="gee" / "cds"
soil = geo.open_soilgrids(SITE, buffer_m=1500)
obs  = geo.open_aoi(SITE, START, END, buffer_m=1500)   # source="aws" | "planetary" | "cdse" | "gee"
s2, eco = obs["s2"], obs["ecostress"]

# nothing has been downloaded for s2/eco yet; this reads B04, B08 and SCL only
ndvi   = (s2.obs.sel(band="B08") - s2.obs.sel(band="B04")) / \
         (s2.obs.sel(band="B08") + s2.obs.sel(band="B04"))
inside = s2.inside_aoi > 0.5
clear  = s2.valid.where(inside).mean(("x", "y")).compute()
good   = clear.values > 0.9
ndvi_ts = ndvi.isel(time=good).where(inside).mean(("x", "y")).compute()

s2_70m = geo.aggregate_to(s2.isel(time=good), eco)   # S2 on the ECOSTRESS 70 m grid
```

## What you get

* **Meteorology**: 8784 hourly values (2020 is a leap year), same names and
  units as the AmeriFlux CSV reader. The default source is Open-Meteo's ERA5
  service (no account; elevation-adjusted and rounded, see the main README).
  Use `source="gee"` for ERA5-Land from Earth Engine, or `source="cds"` for
  unmodified ERA5.
* **Soils**: clay is about 35 % and bulk density 1.4-1.65 kg dm-3 down the
  profile, pH 6.4-7.4, with soil organic carbon falling from about 21 g kg-1
  at the surface to about 3 g kg-1 below 1 m.
* **Sentinel-2**: 144 scenes on tile 14TQL, of which 54 are clear over the
  field. Field-mean NDVI follows the crop calendar: about 0.2 in winter,
  rising from May, 0.71-0.74 through July-August, then falling to about 0.4
  by early October.
* **ECOSTRESS**: L1CT radiance (bands 2, 4, 5) and L2T LST on the same
  tile grid, so `aggregate_to` is an exact 7 x 7 block mean.

## Running it

```bash
pip install -e .          # Python >= 3.12
python examples/us_ne2_2020.py  # about 3 minutes for the full year
```

* ECOSTRESS needs an Earthdata login in `~/.netrc`
  (`machine urs.earthdata.nasa.gov login <user> password <pass>`).
* The site coordinates are the tower position to four decimals; check them
  against the AmeriFlux site page if you need the exact footprint.
* One 2020 AWS item (`S2B_14TQL_20200110_1_L2A`) has a truncated metadata
  file at the source. The reader warns and skips it, so you see 144 scenes,
  not 145.
* To keep a local copy and avoid paying for the reads again:
  `geo.to_zarr_cache(s2.isel(time=good), "us-ne2_2020_s2.zarr")`.
