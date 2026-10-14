# Handoff: lazy xarray readers in `eof` for Sentinel-2 and ECOSTRESS

Date: 2026-10-13. Base code: `eof-main` (profLewis/eof, v0.1.0).
Consumer: the CALDO "poor man's assimilation" (importance weighting
of a pre-run CALDO ensemble against S2 reflectance and ECOSTRESS
brightness temperature, per pixel).

## 1. Goal and scope

Add a lazy, xarray-based read path to `eof` for:

- **Sentinel-2 L2A** (AWS Earth Search and Planetary Computer);
- **ECOSTRESS L1CT_RAD** (tiled TOA radiance, Collection 2);
- **ECOSTRESS L2T_LSTE** (tiled LST, used only for comparison).

Both sensors return an `xr.Dataset` in one common schema (section 5).
Data stay dask-backed until `.compute()` or a Zarr write.

**Runs on a laptop.** I/O is HTTPS range requests from outside AWS.
Nothing assumes `us-west-2` or S3 credentials.

**In scope**

- STAC search, asset resolution, lazy windowed COG reads.
- Per-pixel (coarse-grid) sun and view geometry.
- QA masks, DN-to-physical conversion, an observation-uncertainty
  model, SRF metadata.
- Aggregating S2 onto the ECOSTRESS 70 m grid.
- A local Zarr cache.

**Out of scope**

- Atmospheric profiles, TOA-to-TOC correction, meteorology.
- CALDO and the observation operators.
- Multi-tile mosaics. Phase 1 is one MGRS tile per call.
- CDSE, GEE, Landsat, MODIS and VIIRS lazy paths.
- ECOSTRESS L1B_GEO swath reading (an optional later validation; see
  section 7.4).

The existing eager API (`get_s2_data`, `get_eo_data`) stays as is.
New code goes in a subpackage, `eof/xr/`.

## 2. Decisions taken (confirm before or while building)

| # | Decision | Why |
|---|----------|-----|
| D1 | Read on each sensor's **native tile grid**; never reproject on read | Both sensors use MGRS/UTM tiles; reprojection silently smooths radiometry |
| D2 | One MGRS tile per call, chosen explicitly (`tile="14TQL"`) | US-Ne1 sits near the UTM 14/15 boundary and is covered by 14TQL, 14TPL and 15TTF; picking automatically gives inconsistent grids |
| D3 | Upsample S2 20 m bands to 10 m by **pixel replication** (2x2), not interpolation | Preserves radiometry; later aggregation to 70 m averages anyway |
| D4 | Laziness via `dask.delayed` windowed `rasterio` reads, **not odc-stac or stackstac** | ECOSTRESS items have no `proj:` metadata and awkward asset keys; one transparent mechanism for both sensors |
| D5 | Geometry stored on **coarse angle grids** `(time, ay, ax)`, interpolated on demand | Full-resolution angle cubes are as large as the data |
| D6 | ECOSTRESS bands fixed to **2, 4, 5** (8.78, 10.49, 12.09 um) | Bands 1 and 3 are all-NaN in 3-band granules (most of the record since 2019) |
| D7 | ECOSTRESS view azimuth **derived from the `view_zenith` gradient** (section 7.4) | Not present in any tiled product; avoids reading L1B_GEO swaths |
| D8 | S2 view geometry is the **mean over the bands used** | Per-band geometry is overkill for one-layer SAIL; per-band VZA differences are small |
| D9 | Observation uncertainty from a simple parametric model, **not eof's 10 %** | 10 % of red reflectance over dense canopy (~0.002) would dominate the likelihood |
| D10 | New code uses **rasterio only**, never `osgeo.gdal` | pip wheels of rasterio and GDAL bundle separate libgdal copies; mixing them in one process is fragile. Prefer a conda-forge env |
| D11 | New modules use Python >= 3.10 syntax (`float \| None`) | Bump `requires-python` or gate the `xr` extra |

Decisions are fine. I'm happy to use python >= 3.12. observation geometries can be quite coarse. Errors of ~5-10 degrees are not that relevant here. 
## 3. Facts about current eof (read from source)

- **Data model.** A frozen dataclass of numpy arrays (`EOResult`),
  eagerly warped with GDAL to the cutline bbox at 10 m.
  - Angles are `(3, N)` scene means from STAC properties.
  - Time is day of year only.
  - Cache is `.npz` of int16 DN.
- **S2 bindings** (`_platform_bindings.py`) already hold the asset
  keys per platform. Reuse them:
  - AWS: `blue`, `green`, `red`, `rededge1-3`, `nir`, `nir08`,
    `swir16`, `swir22`, `scl`;
  - Planetary: `B02`...`B12`, `SCL`.
- **Reusable functions:** `_scl.SCL_MASK_VALUES` and the MGRS filter
  logic.
- **SRFs** (`data/srf.npz`) are Gaussian approximations for S2A and
  S2B only. There is nothing for S2C or ECOSTRESS.

### Defects found (worth fixing upstream too)

1. **Wrong SZA on Planetary Computer.** `_stac_reader._extract_angles`
   falls back to `s2:mean_solar_zenith` but still applies
   `sza = 90 - x`.
   - Verified: Planetary items carry `s2:mean_solar_zenith` (23.26 deg
     on 2022-07-08, tile 15TTF) and no `view:sun_elevation`, so eof
     returns 66.7 deg.
   - View zenith and azimuth default to 0, because neither platform
     provides `view:incidence_angle`.
2. **Likely double BOA offset on AWS Earth Search v1.**
   - Verified: items carry `earthsearch:boa_offset_applied: true`,
     meaning the offset is already in the COG DNs.
   - eof subtracts 1000 again whenever `s2:processing_baseline` is
     04.00 or later. Reflectance would then be about 0.1 too low and
     clipped at 0.
   - **Verify** with the test in section 9 (same date and pixel from
     AWS vs Planetary) before trusting any AWS-sourced eof output from
     2022 onwards.
3. **Unaligned output grid.** `read_and_crop_band` warps to the cutline
   bbox at 10 m without `targetAlignedPixels`. The output grid is
   therefore generally offset from the native S2 grid, and every band,
   10 m ones included, is bilinearly resampled.
4. **Uncertainty is `0.1 * reflectance`** everywhere.

## 4. Design

### 4.1 Mechanism

For each STAC item and layer, a delayed function reads one window of
one COG and returns a numpy block. Blocks become `dask.array`s through
`da.from_delayed`, then stack into `(time, band, y, x)`.

- Nothing is read at open time, except one header read per
  (tile, resolution) to get the grid.
- All items of one tile share that grid, so a single window is
  computed once and reused. Each read asserts that the source
  transform matches, and raises otherwise.

```python
import numpy as np
import rasterio
from rasterio.windows import Window


def read_window(
    href: str,
    window: tuple[int, int, int, int],
    gdal_env: dict,
    expected_transform: tuple,
    upsample: int = 1,
) -> np.ndarray:
    """Read one band window from a COG and return float32.

    Parameters
    ----------
    href : str
        HTTPS URL of the COG.
    window : tuple of int
        (row_off, col_off, n_rows, n_cols) in the file's native grid.
    gdal_env : dict
        GDAL config options (auth, curl tuning), applied inside the
        call so the function is safe in dask worker threads.
    expected_transform : tuple
        Affine coefficients (a, b, c, d, e, f) the file must have.
        Guards against items that silently sit on a different grid.
    upsample : int
        Integer pixel-replication factor (2 for S2 20 m -> 10 m).

    Returns
    -------
    np.ndarray
        (n_rows * upsample, n_cols * upsample) float32, with nodata
        mapped to NaN and the file's scale/offset applied.
    """
    row0, col0, nrow, ncol = window
    with rasterio.Env(**gdal_env), rasterio.open(href) as src:
        # Refuse to read from a grid other than the one we planned
        if tuple(src.transform)[:6] != tuple(expected_transform):
            raise ValueError(f"Unexpected grid in {href}")
        data = src.read(
            1, window=Window(col0, row0, ncol, nrow),
            boundless=True, masked=True,
        )
        scale, offset = src.scales[0], src.offsets[0]
    out = data.astype(np.float32).filled(np.nan) * scale + offset
    if upsample > 1:
        # Pixel replication: no interpolation of radiometry
        out = np.repeat(np.repeat(out, upsample, 0), upsample, 1)
    return out
```

- **Large windows.** Split them into sub-windows aligned to the COG
  internal block size (check `src.block_shapes`; typically 512 or
  1024) and assemble with `da.block`.
- **Chunks.** Default to one chunk per (time, band) when the window is
  2048 x 2048 or smaller.

### 4.2 Window snapping

- Transform the AOI (GeoJSON in EPSG:4326) to the tile CRS and take
  its bounds.
- Snap them **outwards** to multiples of 20 m from the tile origin, so
  the 10 m and 20 m windows cover exactly the same extent at an exact
  2:1 ratio. The S2 10, 20 and 60 m grids share the tile upper-left
  corner, e.g. 14TQL is (699960, 4600020) at both 10 m and 20 m
  (verified from `proj:transform`).
- If the ECOSTRESS and S2 grids turn out to be aligned (section 7.3),
  snap to 140 m (the lcm of 20 and 70) instead, so that S2 windows
  aggregate exactly onto ECOSTRESS cells.

### 4.3 Module layout (flat functions; no classes beyond dataclasses)

```
eof/xr/
  __init__.py       public API re-exports
  _env.py           gdal_env_for(source) -> dict; Earthdata auth
  _search.py        search_s2(...), search_eco(...), dedupe_s2(...)
  _assets.py        s2_asset_hrefs(item, source), eco_layer_hrefs(item),
                    parse_eco_id(item_id)
  _grid.py          tile_grid(href) -> (crs, transform, shape),
                    aoi_window(grid, aoi, snap_m) -> window
  _read.py          read_window(...), lazy_layer(...)
  _s2.py            open_s2(...), s2_angle_grids(mtd_xml_text)
  _eco.py           open_ecostress_l1ct(...), open_ecostress_lste(...),
                    eco_view_azimuth(vza, transform, crs)
  _solar.py         solar_position(time_utc, lon, lat) -> (sza, saa)
  _geometry.py      angles_at(ds, x, y) -> xr.Dataset
  _aggregate.py     aggregate_to(ds_fine, ds_coarse) -> xr.Dataset
  _uncertainty.py   s2_sigma(rho, ...), eco_sigma_radiance(L, srf, ...)
  _srf.py           load_srf_xr(sensor, platform) -> xr.DataArray
  _schema.py        validate_obs_dataset(ds) -> None (raises)
  _cache.py         to_zarr_cache(ds, path), open_cache(path)
```

**Dependencies.** Add an optional extra:
`xr = ["xarray", "dask", "rasterio", "rioxarray", "pyproj", "zarr"]`.
Add `pvlib` to `dev` only (as the reference in tests).

### 4.4 Public API

```python
def open_s2(
    aoi: str | dict,          # GeoJSON path or geometry, EPSG:4326
    start: str,               # 'YYYY-MM-DD'
    end: str,
    tile: str,                # MGRS tile, e.g. '14TQL'
    source: str = "aws",      # 'aws' | 'planetary'
    bands: tuple[str, ...] = S2_BANDS,  # B02..B12 (10 bands)
    max_cloud: float = 80.0,  # eo:cloud_cover pre-filter [%]
    snap_m: float = 20.0,     # window snapping [m]
) -> xr.Dataset: ...

def open_ecostress_l1ct(
    aoi: str | dict,
    start: str,
    end: str,
    tile: str,
    bands: tuple[int, ...] = (2, 4, 5),
    version: str = "002",     # '003' also exists; see 7.1
    day_only: bool = True,    # drop granules with SZA > 85 deg
    nedt_k: float = 0.15,     # per-band NEdT for obs_unc [K]
) -> xr.Dataset: ...

def open_ecostress_lste(aoi, start, end, tile, version="002",
                        day_only=True) -> xr.Dataset: ...

def angles_at(ds: xr.Dataset, x, y) -> xr.Dataset: ...
def aggregate_to(fine: xr.Dataset, coarse: xr.Dataset) -> xr.Dataset: ...
def to_zarr_cache(ds: xr.Dataset, path: str) -> None: ...
def open_cache(path: str) -> xr.Dataset: ...
```

## 5. Output schema (the contract CALDO relies on)

One `xr.Dataset` per sensor and product. Coordinates are pixel
centres in the tile CRS.

| Name | Dims | Dtype | Meaning |
|------|------|-------|---------|
| `obs` | time, band, y, x | float32 | S2: BOA reflectance [-]. ECO L1CT: TOA radiance [W m-2 sr-1 um-1] |
| `obs_unc` | time, band, y, x | float32 | 1-sigma, same units as `obs` |
| `valid` | time, y, x | bool | Clear land and good quality (sensor rules, sections 6-7) |
| `sza`, `saa`, `vza`, `vaa` | time, ay, ax | float32 | Degrees; azimuths clockwise from **true** north; `vaa` is the pixel-to-sensor direction |
| `x`, `y` | x / y | float64 | Pixel centres [m] |
| `ax`, `ay` | ax / ay | float64 | Angle-grid node coordinates [m] |
| `time` | time | datetime64[ns] | UTC acquisition time (granule start for ECO) |
| `band` | band | str | `B02`... or `TIR2`, `TIR4`, `TIR5` |
| `wl_centre` | band | float32 | Nominal centre wavelength [nm] |
| `platform` | time | str | `sentinel-2a/b/c` or `ecostress` |
| `item_id` | time | str | STAC id, for provenance |
| `tile` | (attr) | str | MGRS tile |
| `orbit`, `scene` | time | int | ECO only, parsed from the id |

- **Attributes:** `sensor`, `product`, `obs_kind`
  (`reflectance` | `radiance`), `units`, `source`, plus the CRS
  written with `rioxarray` (`ds.rio.write_crs(...)`).
- **SRFs** are separate: `load_srf_xr(sensor, platform)` returns a
  DataArray `(band, wavelength)`. A time step's SRF is chosen by its
  `platform` coordinate.
- **Uncertainty model.** S2:
  $\sigma^2 = \sigma_{abs}^2 + (c\,\rho)^2$, defaulting to
  $\sigma_{abs}=0.005$ and $c=0.05$, both arguments. ECO:
  $\sigma_L = \mathrm{NE\Delta T}\cdot \partial \bar B_b/\partial T$
  at the observed BT. Model-error inflation belongs downstream, not
  here.
- **ECO L2T_LSTE** uses the same coordinates, with variables `lst`,
  `lst_err`, `emis_wb`, `qc`, `height`, `valid` and the geometry
  grids. It has no `obs` or `band`.
- **Validation.** `validate_obs_dataset` checks dims, dtypes, units
  attributes, monotonic `time`, finite angle ranges and that `obs` is
  dask-backed. Every `open_*` function calls it before returning.

## 6. Sentinel-2 specifics

### 6.1 Search, asset keys and conversion

- **Search.** `pystac_client` against the existing `stac_url` values;
  filter on `tile` through the platform's MGRS property:
  - AWS: `grid:code == "MGRS-14TQL"`;
  - Planetary: `s2:mgrs_tile == "14TQL"`.
- **Duplicates.**
  - AWS ids look like `S2B_14TQL_20220708_0_L2A`; the `_0_`/`_1_`
    field is a reprocessing sequence, so keep the highest per date.
  - Planetary: keep the latest `s2:generation_time` per datatake.
- **Angle-metadata asset keys** (verified 2022-07):

  | Source | Bands | QA | Angle metadata |
  |--------|-------|----|----------------|
  | AWS | `blue`...`swir22` (eof bindings) | `scl` | `granule_metadata` (MTD_TL.xml) |
  | Planetary | `B02`...`B12` | `SCL` | `granule-metadata` |

  Planetary hrefs need signing; reuse `_source_configs._planetary_sign_href`.
  Signed URLs expire, so sign **inside** `read_window` (pass the
  unsigned href plus a `sign` flag), not at graph-build time.
- **DN to reflectance** is decided per item, as a lazy dask operation:
  - if `earthsearch:boa_offset_applied` is true, use $\rho = DN/10^4$;
  - otherwise, if baseline >= 04.00, use $\rho = (DN - 1000)/10^4$;
  - otherwise use $\rho = DN/10^4$.

  No clipping at 0: negative values are information about the noise
  floor. Mask them through `valid` if needed.
- **`valid`.** SCL (20 m, replicated to 10 m) not in
  `SCL_MASK_VALUES`, all bands finite, and DN > 0. Expose the mask
  set as an argument, since classes 2 (dark) and 7 (unclassified)
  are judgement calls.

### 6.2 Angles from MTD_TL.xml

The file is small (~1 MB). Read it eagerly at open time, in parallel,
because it is needed to build the coarse grids.

- **Sun:** `Tile_Angles/Sun_Angles_Grid/{Zenith,Azimuth}/Values_List`,
  23 x 23 nodes with `COL_STEP` and `ROW_STEP` of 5000 m. Node (0, 0)
  is the tile upper-left corner (`Tile_Geocoding/Geoposition`, ULX and
  ULY).
- **View:** `Viewing_Incidence_Angles_Grids[bandId, detectorId]`. Each
  detector grid is NaN outside its footprint.
  - Combine detectors per band: nanmean for zenith, circular nanmean
    for azimuth.
  - Then average over the bands used (D8), mapping `bandId` 0..12 to
    B01..B12, with 8 = B8A.
- **Edges.** Nodes outside the swath stay NaN. `angles_at` fills them
  by nearest neighbour before bilinear interpolation.
- **Azimuth interpolation** goes through $(\sin, \cos)$ to avoid the
  360/0 wrap.
- **Known approximation.** View azimuth jumps between odd and even
  detectors; a 5 km bilinear grid smooths across the seams. This is
  acceptable for SAIL at 70 m, but note it.
- **Frame.** MTD azimuths are relative to true north. Keep them that
  way; no grid-convergence correction is needed.

### 6.3 SRFs

eof has Gaussian S2A and S2B curves only. Add the ESA S2A, S2B and
S2C measured SRFs (from the ESA "S2 SRF" spreadsheet, document
COPE-GSEG-EOPG-TN-15-0007) through `scripts/build_srf_data.py`.
S2C is in the record from 2025.

## 7. ECOSTRESS specifics (STAC facts verified on CMR, 2022-07, US-Ne1)

### 7.1 Search

- Endpoint: `https://cmr.earthdata.nasa.gov/stac/LPCLOUD`.
- Collections: `ECO_L1CT_RAD_002` and `ECO_L2T_LSTE_002`.
  `ECO_L1CT_RAD_003` also exists; check its release notes and make
  `version` an argument.
- **Item properties are almost empty:** only `datetime`,
  `start_datetime`, `end_datetime` and `storage:schemes`. There is no
  cloud cover, no `proj:` information and no angles, so every filter
  happens after reading.
- **Ids.** For example
  `ECOv002_L1CT_RAD_22611_008_14TQL_20220702T081925_0712_02`. Parse
  with:

  ```python
  ECO_ID = (
      r"ECOv(?P<ver>\d{3})_(?P<product>L1CT_RAD|L2T_LSTE)_"
      r"(?P<orbit>\d{5})_(?P<scene>\d{3})_(?P<tile>\w{5})_"
      r"(?P<start>\d{8}T\d{6})_(?P<build>\d{4})_(?P<rev>\d{2})"
  )
  ```

- **Tile filter.** Use the parsed `tile`. A bbox at US-Ne1 returns
  both 14TQL and 14TPL granules.
- **Pairing.** L1CT and L2T granules correspond through
  (orbit, scene, tile).
- **Night granules are returned.** The example above is at 08:19 UTC,
  i.e. night at Mead. `day_only` computes SZA at the AOI centre and
  drops granules with SZA > 85 deg. Keep `False` available, since
  night LST may be useful later.

### 7.2 Assets and layers

- **Asset keys are not layer names.** They look like
  `002/<granule_id>/<granule_id>_<layer>`, with duplicates under
  `s3_002/...`. `eco_layer_hrefs(item)` must:
  - keep assets whose href starts with
    `https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/`
    and ends `.tif`;
  - set the layer name to the href stem after `<granule_id>_`;
  - ignore the `s3_*`, `thumbnail_*` and `browse` keys.
- **L1CT_RAD layers:** `radiance_1`...`radiance_5`,
  `data_quality_1`...`_5`, `cloud`, `water`. Output bands are
  `radiance_{2,4,5}` as `TIR2`, `TIR4` and `TIR5`. Assert that the
  dropped bands are all-NaN on a small window; if not (2018-2019
  five-band data), still drop them for a fixed band dimension.
- **L2T_LSTE layers:** `LST`, `LST_err`, `QC`, `EmisWB`,
  `view_zenith`, `height`, `cloud`, `water`. The tiled product has
  **no per-band emissivity**, only broadband.
- **Grid.** Each tile is 1568 x 1568 pixels at 70 m (109,760 m) in
  UTM, as COGs. Read scale and offset from the file (`src.scales`,
  `src.offsets`); don't hard-code them.
- **`valid` for L1CT:** `cloud == 0`, `water == 0`, and
  `data_quality_b` "good" for all output bands. **Read the data
  quality and cloud codes from the ECO L1C Grid/Tile User Guide V2
  tables** rather than assuming 0 means good.
- **`valid` for L2T:** decode the QC bit flags (also in the user
  guide); expose the mandatory-QA and LST-accuracy bits as arguments.
- **Geolocation.** L1CT is resampled twice by nearest neighbour
  (swath to a 0.0006 deg grid, then to the 70 m UTM tile). Expect up
  to about one pixel of misregistration. Downstream, mask pixels
  within 70 m of field edges.

### 7.3 Grid alignment with S2 (verification task, do first)

- S2 tiles span 109,800 m; ECOSTRESS tiles 109,760 m. S2 14TQL has its
  upper-left corner at (699960, 4600020).
- Read the ECOSTRESS 14TQL transform from any L1CT COG, then check
  that `(x0_eco - x0_s2) % 10 == 0` and likewise for y.
- **If aligned:** each 70 m cell is exactly 7 x 7 S2 10 m pixels, and
  `aggregate_to` is a block mean.
- **If not:** `aggregate_to` uses
  `rasterio.warp.reproject(..., resampling=Resampling.average)` onto
  the ECOSTRESS grid. Implement this general path anyway, and test
  that it reduces to the block mean when aligned.
- **Outputs of `aggregate_to`:** the NaN-aware mean of `obs`, a
  `valid_frac` variable (fraction of valid fine pixels per coarse
  cell), and `obs_unc` propagated as the standard error of the mean
  assuming independence (state this in the docstring; downstream may
  inflate it).

### 7.4 Geometry

- **SZA and SAA.** Compute them on the coarse angle grid (16-pixel
  stride, about 1.1 km) from the granule `start_datetime` and node
  lon/lat (`pyproj`).
  - Implement the NOAA solar-position algorithm in numpy in
    `_solar.py` (about 40 lines; accuracy around 0.01 deg).
  - Test it against `pvlib` (dev only).
  - The tile's scan time spread (seconds) is negligible.
- **VZA.** The L2T_LSTE `view_zenith` layer, decimated to the same
  coarse grid. Use a block median over the 16 x 16 stride to remove
  nearest-neighbour stair-steps.
- **VAA from the VZA gradient.**
  - Principle: ECOSTRESS scans across track, and VZA (unsigned) grows
    roughly linearly with distance from the ground track. The
    direction from a pixel towards the sensor is therefore the
    direction of steepest VZA *descent*.
  - On the coarse grid, with map-grid gradient
    $g=(\partial_x \mathrm{VZA}, \partial_y \mathrm{VZA})$ (x east,
    y north):
    $\mathrm{VAA}_{grid} = \operatorname{atan2}(-g_x, -g_y)$ mod 360.
  - Convert to true north by adding the meridian convergence at each
    node (`pyproj.Proj(crs).get_factors(lon, lat).meridian_convergence`).
    The **sign convention must be pinned down by a test**: transform
    two points on one true meridian and check that the corrected
    azimuth of their join is 0 or 180. At 14TQL the correction is
    about 1.5 deg.
  - Where VZA < 1 deg the direction is ill-defined and irrelevant. Set
    VAA to 0 there and record that in an attribute.
  - Optional later validation: compare against L1B_GEO
    `view_azimuth` for one granule.
- **Cost.** SZA, SAA, VZA and VAA grids are tiny, so compute them
  eagerly at open time. Only `obs`, `obs_unc` and `valid` are lazy.

### 7.5 SRFs and radiance to BT

- eof has no ECOSTRESS SRFs. Obtain the PHyTIR band SRFs from the JPL
  ECOSTRESS site or ATBD, and add them to `srf.npz` under
  `ecostress`.
- Provide `band_planck_lut(srf, t_grid)` and
  `radiance_to_bt(L, lut)` (1-D interpolation per band), used for
  `obs_unc` and as a convenience. `obs` stays as radiance; BT
  conversion for assimilation happens downstream, after atmospheric
  correction terms are available.
- **Units check:** L1CT radiance is in W m-2 sr-1 um-1. Confirm from
  the COG metadata and the user guide.

## 8. Laptop I/O: auth and GDAL settings

- **S2 on AWS:** public HTTPS COGs, no credentials.
- **S2 on Planetary:** sign the URLs (above).
- **ECOSTRESS:** HTTPS from `data.lpdaac.earthdatacloud.nasa.gov`
  works from anywhere with Earthdata Login. Use the `https://` hrefs,
  never `s3://` (those need in-region temporary credentials).
- **Primary auth: `~/.netrc` plus cookies** (the LP DAAC-recommended
  GDAL route):

  ```python
  EARTHDATA_GDAL_ENV = {
      "GDAL_HTTP_NETRC": "YES",
      "GDAL_HTTP_COOKIEFILE": "~/.eof/cookies.txt",
      "GDAL_HTTP_COOKIEJAR": "~/.eof/cookies.txt",
  }
  ```

  Expand `~` before use.
- **Alternative auth:** a bearer token from `EARTHDATA_TOKEN` (eof
  already reads it). GDAL behaviour with `Authorization` headers
  across the redirect to signed S3 URLs varies by GDAL version, so
  test it rather than assuming it works.
- **Common curl tuning** (all sources):

  ```python
  COMMON_GDAL_ENV = {
      "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
      "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.xml",
      "GDAL_HTTP_MULTIPLEX": "YES",
      "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
      "GDAL_HTTP_MAX_RETRY": "5",
      "GDAL_HTTP_RETRY_DELAY": "2",
      "VSI_CACHE": "TRUE",
  }
  ```

- **Concurrency.** Use dask's threaded scheduler with 8-16 threads for
  I/O-bound reads. Pass the env into `read_window`; don't rely on
  process-global GDAL state.

## 9. Caching and sizes

- **Remote reads repeat on every compute.** The workflow is:
  `open_*` (lazy), select, compute once, then `to_zarr_cache` writes
  locally. Analysis then uses `open_cache`. This replaces the `.npz`
  cache for the new path.
- **Zarr encoding for `obs`:** int16 with `scale_factor=1e-4` and
  `_FillValue=-32768` for S2; float32 for radiance. Compress with
  zstd. Chunks are (1, nband, 512, 512).
- **Budget** (season of about 70 S2 and about 40 ECO dates):

  | Product | Per date | Season |
  |---------|----------|--------|
  | S2 full tile, 10 bands, 10 m, int16 | ~2.4 GB | ~170 GB |
  | Same, aggregated to 70 m, float32 | ~0.1 GB | ~7 GB |
  | ECO L1CT full tile, 3 bands | ~30 MB | ~1.2 GB |

  So: keep 10 m data only for field-scale AOIs, and cache the
  70 m aggregate for tile-scale work.

## 10. Tests

### 10.1 Offline (fixtures committed under `tests/fixtures/`)

- `parse_eco_id` on both example ids above.
- `eco_layer_hrefs` on a saved L1CT and L2T item JSON: correct layer
  names, no `s3`, thumbnail or browse entries.
- `s2_angle_grids` on a saved `MTD_TL.xml`:
  - grid shape 23 x 23;
  - node (0, 0) at the tile upper-left corner;
  - mean SZA matches `Mean_Sun_Angle` within 0.1 deg.
- `solar_position` against `pvlib` at several dates and latitudes;
  max error 0.05 deg.
- `eco_view_azimuth` on a synthetic VZA field generated from a known
  track heading and side: recovers VAA within 1 deg away from nadir.
- Meridian-convergence sign test (section 7.4).
- `aoi_window`:
  - 10 m and 20 m windows cover identical extents at exactly 2:1;
  - snapping is outward;
  - an AOI not inside the tile raises.
- `read_window` replication: the `upsample=2` output equals
  `np.kron(block, ones((2, 2)))`.
- `aggregate_to`: on aligned synthetic grids it equals the 7 x 7 block
  mean, NaNs are excluded, and `valid_frac` is correct.
- DN conversion: the three branches of section 6.1.
- `validate_obs_dataset`: passes on a synthetic valid dataset and
  fails on each violated rule.

### 10.2 Network (`@pytest.mark.network`; AOI = 3 x 3 km at US-Ne1, tile 14TQL, 2022-06-01 to 2022-08-31)

- **Laziness:** after `open_s2`, `obs.data` is a `dask.array.Array`,
  and wrapping `read_window` in a counter shows zero data reads before
  `.compute()`.
- **S2 sanity:** clear-sky July reflectance over maize has NIR
  0.3-0.5 and red below 0.06.
- **AWS vs Planetary:** the same date and pixel agree within 0.005
  per band. This is the double-offset check from section 3; Planetary
  serves 14TQL too.
- **Fixed eof angle bug:** SZA from the MTD grid agrees with
  `s2:mean_solar_zenith` and with `90 - view:sun_elevation` within
  0.5 deg.
- **ECO:**
  - day granules only;
  - `band == [TIR2, TIR4, TIR5]`;
  - radiance converted to BT is within a plausible 270-330 K;
  - BT(12.09) <= BT(10.49) + 1 K on most clear pixels (the
    water-vapour ordering);
  - TOA BT vs L2T LST: correlation > 0.9 and LST - BT10.49 between
    0 and 8 K on clear pixels.
- **Grid alignment (section 7.3):** report the result; this is the
  first network test to run.

## 11. Suggested order of work

1. `_env`, `_search`, `_assets`, with offline tests. Run the grid
   alignment check (7.3) once by hand.
2. `_grid`, `_read`, plus the laziness test.
3. `_s2` (reflectance, `valid`, MTD angles), `_uncertainty`,
   `_schema`. Run the S2 network tests, including AWS vs Planetary.
4. `_solar`, `_eco` (L1CT, then L2T, VAA from gradient). Run the ECO
   network tests.
5. `_aggregate`, `_cache`, SRF additions.
6. Fix upstream defects 1-3 in the eager path, or at least document
   them in the README.

## 12. Open items

- Confirm D1-D11 (section 2), especially D2 (tile 14TQL for US-Ne1)
  and D7 (VAA from the gradient vs reading L1B_GEO).
- V003 of ECO_L1CT_RAD: what changed (calibration, geolocation), and
  whether to default to it.
- ECOSTRESS SRF source file, and S2C SRFs.
- Whether night granules are wanted later. Night thermal data
  constrain soil heat capacity and moisture rather than stomatal
  stress.
