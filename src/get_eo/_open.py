"""The one-call entry point: AOI + period -> dict of xarray Datasets."""

from __future__ import annotations

import warnings

from get_eo._aoi import resolve_aoi
from get_eo._assets import S2_BANDS, S2_SOURCES, SCL_MASK_VALUES
from get_eo._eco import (
    eco_grid, open_ecostress_l1ct, open_ecostress_lste, screen_granules,
)
from get_eo._env import check_earthdata_login, gdal_env_for
from get_eo._grid import Bounds, TileGrid, aoi_bounds
from get_eo._s2 import open_s2
from get_eo._schema import validate_obs_dataset
from get_eo._search import (
    choose_tile, s2_grid, s2_tile_of, search_eco, search_s2,
)
from get_eo._assets import parse_eco_id

_S2_NAMES = {"s2", "sentinel2", "sentinel-2"}
_ECO_NAMES = {"ecostress", "eco"}
LCM_S2_ECO = 140.0     # lcm(20 m, 70 m)


def open_aoi(
    aoi,
    start: str,
    end: str,
    sensors=("s2", "ecostress"),
    *,
    buffer_m: float | None = None,
    tile: str | None = None,
    source: str = "aws",
    max_cloud: float = 100.0,
    day_only: bool = True,
    bands=S2_BANDS,
    scl_mask=SCL_MASK_VALUES,
    eco_version: str = "002",
    include_lste: bool = True,
    chunk: int = 1024,
    gee_project: str | None = None,
) -> dict:
    """Everything available over an AOI and period, as lazy Datasets.

    Parameters
    ----------
    aoi
        GeoJSON path / dict, WKT, shapely geometry, GeoDataFrame,
        ``(w, s, e, n)`` bbox, or ``(lon, lat)`` point with ``buffer_m``.
        Always lon/lat (EPSG:4326).
    start, end : str
        'YYYY-MM-DD', inclusive.
    sensors : iterable of {'s2', 'ecostress'}
    buffer_m : float, optional
        Half-width of the square drawn around a point AOI [m].
    tile : str, optional
        MGRS tile (e.g. '14TQL'). By default the tile that best contains
        the AOI is used; every sensor is read on that tile's native grid
        and nothing is reprojected. A warning is raised if the AOI spans
        several tiles.
    source : {'aws', 'planetary', 'cdse', 'gee'}
        Where Sentinel-2 comes from. All four give the same schema, grid
        and reflectance scale.

        * ``aws``: Earth Search COGs, public.
        * ``planetary``: Planetary Computer COGs, SAS-signed automatically.
        * ``cdse``: Copernicus Data Space JP2s over S3. Needs S3 keys
          (``CDSE_S3_ACCESS_KEY`` / ``CDSE_S3_SECRET_KEY``) and
          ``pip install 'get_eo[cdse]'``. JP2 windows are slower than COGs.
        * ``gee``: Earth Engine ``S2_SR_HARMONIZED`` through xee, on the
          same native tile grid. Needs ``pip install 'get_eo[gee]'`` and
          ``earthengine authenticate``; sun/view angles are one mean value
          per image (constant coarse grid).

        ECOSTRESS always uses NASA Earthdata (needs ``~/.netrc``).
    max_cloud : float
        Scene-level ``eo:cloud_cover`` ceiling for S2 [%]. The default
        keeps everything; use the per-pixel ``valid`` mask to select.
    gee_project : str, optional
        Google Cloud project for ``source='gee'`` (default: the
        ``EOF_GEE_PROJECT`` or ``GOOGLE_CLOUD_PROJECT`` environment variable).
    day_only : bool
        Drop ECOSTRESS granules with SZA > 85 deg at the AOI centre.

    Returns
    -------
    dict[str, xr.Dataset]
        Keys ``'s2'``, ``'ecostress'`` (L1CT radiance) and
        ``'ecostress_lste'`` (L2T LST), for those that have data. All on
        the native tile grid of each sensor, snapped outwards so that S2
        and ECOSTRESS windows cover the same extent and a 70 m cell is
        exactly 7 x 7 S2 10 m pixels (see :func:`get_eo.aggregate_to`).
        Pixels outside the AOI polygon but inside the window are kept;
        ``inside_aoi`` gives the fraction of each pixel in the polygon.
    """
    want = {s.lower() for s in sensors}
    unknown = want - _S2_NAMES - _ECO_NAMES
    if unknown:
        raise ValueError(f"Unknown sensors {sorted(unknown)}; "
                         "supported: 's2', 'ecostress'")
    if source not in S2_SOURCES:
        raise ValueError(
            f"Unknown source {source!r}; supported: {', '.join(S2_SOURCES)}")
    do_s2 = bool(want & _S2_NAMES)
    do_eco = bool(want & _ECO_NAMES)
    geom = resolve_aoi(aoi, buffer_m)
    centre = (geom.centroid.x, geom.centroid.y)

    # ---- search -----------------------------------------------------------
    s2_items, eco_l1, eco_l2 = [], [], []
    if do_s2:
        s2_items = search_s2(geom, start, end, source, max_cloud, tile, gee_project)
    if do_eco:
        check_earthdata_login()
        eco_l1 = search_eco(geom, start, end, "L1CT_RAD", eco_version, tile)
        eco_l2 = search_eco(geom, start, end, "L2T_LSTE", eco_version, tile)

    # ---- tile ---------------------------------------------------------------
    grids: dict[str, TileGrid] = {}
    for it in s2_items:
        t = s2_tile_of(it)
        if t and t not in grids and (tile is None or t == tile):
            grids[t] = s2_grid(it, source)
    eco_env = gdal_env_for("earthdata") if do_eco else None
    for it in eco_l1 + eco_l2:
        t = parse_eco_id(it.id)["tile"]
        if t not in grids and (tile is None or t == tile):
            grids[t] = eco_grid(it, eco_env)
    if not grids:
        raise LookupError(
            f"No {sorted(want)} data found over the AOI between {start} "
            f"and {end}")
    chosen = tile or choose_tile(geom, grids)
    if chosen not in grids:
        raise LookupError(f"No data found on tile {chosen}")

    s2_items = [i for i in s2_items if s2_tile_of(i) == chosen]
    eco_l1 = [i for i in eco_l1 if parse_eco_id(i.id)["tile"] == chosen]
    eco_l2 = [i for i in eco_l2 if parse_eco_id(i.id)["tile"] == chosen]

    # ---- windows ------------------------------------------------------------
    s2g = grids[chosen] if s2_items else None
    ecog = None
    if eco_l1 or eco_l2:
        ecog = eco_grid((eco_l1 or eco_l2)[0], eco_env)

    b_s2 = b_eco = None
    if s2g is not None and ecog is not None:
        aligned = (abs((s2g.x0 - ecog.x0) % ecog.res) < 1e-6
                   and abs((s2g.y0 - ecog.y0) % ecog.res) < 1e-6
                   and s2g.crs == ecog.crs)
        if aligned:
            b_s2 = b_eco = aoi_bounds(geom, s2g, LCM_S2_ECO)
        else:
            warnings.warn(
                "S2 and ECOSTRESS grids are not aligned on this tile; "
                "aggregate_to will resample.", stacklevel=2)
            b_s2 = aoi_bounds(geom, s2g, 20.0)
            b_eco = aoi_bounds(geom, ecog, ecog.res)
    elif s2g is not None:
        b_s2 = aoi_bounds(geom, s2g, 20.0)
    elif ecog is not None:
        b_eco = aoi_bounds(geom, ecog, ecog.res)

    # ---- datasets -----------------------------------------------------------
    out = {}
    if s2_items:
        out["s2"] = open_s2(s2_items, source, s2g, b_s2, geom, chosen,
                            bands=tuple(bands), scl_mask=scl_mask,
                            chunk=chunk, gee_project=gee_project)
    if eco_l1:
        l1 = screen_granules(eco_l1, "radiance_4", ecog, b_eco, eco_env,
                             centre, day_only)
        if l1:
            out["ecostress"] = open_ecostress_l1ct(
                l1, eco_l2, ecog, b_eco, geom, chosen, chunk=chunk,
                version=eco_version)
    if include_lste and eco_l2:
        l2 = screen_granules(eco_l2, "LST", ecog, b_eco, eco_env,
                             centre, day_only)
        if l2:
            out["ecostress_lste"] = open_ecostress_lste(
                l2, ecog, b_eco, geom, chosen, chunk=chunk,
                version=eco_version, l2_items_all=eco_l2)

    if not out:
        raise LookupError(
            "Granules were found but none had data over the AOI "
            f"on tile {chosen}")
    for ds in out.values():
        validate_obs_dataset(ds)
    return out
