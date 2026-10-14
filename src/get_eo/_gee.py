"""Sentinel-2 L2A from Google Earth Engine, lazily, via xee.

GEE has no STAC and no asset hrefs, so this module provides the same
three things the STAC path does: a search that returns item-like objects,
the tile grid, and lazy (time, band, y, x) arrays on that grid.

The requested pixel grid is passed to xee explicitly (CRS, origin, size),
so GEE resamples nothing differently from the file-based sources: 20 m
bands are nearest-neighbour replicated to 10 m on GEE's side, which is
the same operation as ``upsample=2`` in :mod:`get_eo._read`.

``earthengine-api`` and ``xee`` are imported lazily (``pip install
'get_eo[gee]'``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import dask.array as da
import numpy as np
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

from get_eo._assets import S2_BANDS
from get_eo._geometry import circular_mean_deg, nanmean_quiet
from get_eo._grid import Bounds, TileGrid

GEE_S2 = "COPERNICUS/S2_SR_HARMONIZED"
GEE_HIGH_VOLUME = "https://earthengine-highvolume.googleapis.com"


def gee_band(band: str) -> str:
    """'B02' -> 'B2' (GEE drops the leading zero)."""
    return "B" + band[1:].lstrip("0") if band[1:2] == "0" else band


@dataclass
class GeeItem:
    """The few attributes of a pystac Item that the S2 path uses."""
    id: str
    datetime: datetime
    properties: dict
    assets: dict = field(default_factory=dict)


def _ee(need_xee: bool = True):
    try:
        import ee
        if need_xee:
            import xee  # noqa: F401  (registers the 'ee' xarray engine)
    except ImportError as err:
        raise ImportError(
            "source='gee' needs earthengine-api"
            + (" and xee" if need_xee else "")
            + ": pip install 'get_eo[gee]'") from err
    return ee


def init_ee(project: str | None = None, need_xee: bool = True):
    """Initialise Earth Engine once (high-volume endpoint) and return ``ee``."""
    ee = _ee(need_xee)
    if ee.data.is_initialized():
        return ee
    project = (project or os.environ.get("EOF_GEE_PROJECT")
               or os.environ.get("GOOGLE_CLOUD_PROJECT"))
    try:
        ee.Initialize(project=project, opt_url=GEE_HIGH_VOLUME)
    except Exception as err:
        raise EnvironmentError(
            "Google Earth Engine is not initialised. Run "
            "`earthengine authenticate` and pass gee_project=<cloud project> "
            "(or set EOF_GEE_PROJECT).") from err
    return ee


def _normalise(feature: dict) -> GeeItem:
    p = dict(feature["properties"])
    t = datetime.fromtimestamp(p["system:time_start"] / 1000, tz=timezone.utc)
    p["grid:code"] = p.get("MGRS_TILE")
    p["platform"] = str(p.get("SPACECRAFT_NAME", "sentinel-2")).lower() \
        .replace("sentinel-2", "sentinel-2")
    p["eo:cloud_cover"] = p.get("CLOUDY_PIXEL_PERCENTAGE")
    return GeeItem(id=feature["id"], datetime=t, properties=p)


def search_s2_gee(aoi: BaseGeometry, start: str, end: str,
                  max_cloud: float = 100.0, tile: str | None = None,
                  project: str | None = None) -> list[GeeItem]:
    """All S2 L2A harmonised images touching the AOI, with their tile grid.

    One extra ``getInfo`` per distinct MGRS tile reads the 10 m grid
    (``proj:code``, ``proj:transform``, ``proj:shape``) into each item.
    """
    ee = init_ee(project)
    d1 = (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)
          ).strftime("%Y-%m-%d")
    ic = (ee.ImageCollection(GEE_S2)
          .filterBounds(ee.Geometry(mapping(aoi)))
          .filterDate(start, d1))
    if max_cloud < 100:
        ic = ic.filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
    if tile:
        ic = ic.filter(ee.Filter.eq("MGRS_TILE", tile))
    feats = ic.sort("system:time_start").getInfo()["features"]
    items = [_normalise(f) for f in feats]

    grids: dict[str, dict] = {}
    for it in items:
        t = it.properties["grid:code"]
        if t not in grids:
            band = ee.Image(it.id).select("B2").getInfo()["bands"][0]
            w, h = band["dimensions"]
            grids[t] = {"proj:code": band["crs"],
                        "proj:transform": band["crs_transform"],
                        "proj:shape": [h, w]}
        it.properties.update(grids[t])
    return items


def gee_s2_arrays(items: list[GeeItem], grid: TileGrid, bounds: Bounds,
                  bands=S2_BANDS, chunk: int = 1024, project: str | None = None):
    """Lazy reflectance and SCL for ``items`` over ``bounds``.

    Returns ``(obs, scl, angles, ax, ay)`` with ``obs`` (t, band, y, x)
    float32 reflectance (NaN where masked), ``scl`` (t, y, x) uint8 (0 where
    masked), and ``angles`` a dict of (t, 2, 2) arrays: GEE gives one
    mean sun/view geometry per image, so the coarse grid is constant.
    """
    import xarray as xr

    ee = init_ee(project)
    names = [gee_band(b) for b in bands]
    # HARMONIZED already has the -1000 DN BOA offset removed.
    def prep(img):
        refl = img.select(names).toFloat().divide(10000.0)
        return (refl.addBands(img.select("SCL").toFloat())
                .copyProperties(img, ["system:time_start"]))

    ids = [it.properties["system:index"] for it in items]
    ic = (ee.ImageCollection(GEE_S2)
          .filter(ee.Filter.inList("system:index", ids))
          .select(names + ["SCL"]).map(prep))

    nrow = int(round((bounds.ymax - bounds.ymin) / grid.res))
    ncol = int(round((bounds.xmax - bounds.xmin) / grid.res))
    ds = xr.open_dataset(
        ic, engine="ee", crs=grid.crs.to_string(),
        crs_transform=(grid.res, 0.0, bounds.xmin, 0.0, -grid.res, bounds.ymax),
        shape_2d=(ncol, nrow), mask_and_scale=False,
        io_chunks={"time": 1, "X": chunk, "Y": chunk})

    times = np.array([np.datetime64(it.datetime.replace(tzinfo=None), "ns")
                      for it in items])
    got = ds["time"].values.astype("datetime64[ns]")
    if got.shape != times.shape or not np.array_equal(got, times):
        raise RuntimeError(
            "xee returned a different set or order of images than the "
            f"search ({got.size} vs {times.size})")

    ydim = next(d for d in ds["B2"].dims if d.lower() in ("y", "lat"))
    xdim = next(d for d in ds["B2"].dims if d.lower() in ("x", "lon"))
    flip = ds[ydim].values[0] < ds[ydim].values[-1]   # want north-up

    def arr(name):
        v = ds[name].transpose("time", ydim, xdim).data
        return v[:, ::-1, :] if flip else v

    obs = da.stack([arr(n) for n in names], axis=1).astype("float32")
    scl = arr("SCL")
    scl = da.where(da.isnan(scl), 0, scl).astype("uint8")

    # angles: per-image means, band-averaged like the MTD path (D8)
    def ang(prefix, circular):
        cols = np.array([[it.properties.get(f"{prefix}_{n}", np.nan)
                          for n in names] for it in items], dtype="float64")
        return (circular_mean_deg(cols, axis=1) if circular
                else nanmean_quiet(cols, axis=1))

    def const(v):
        return np.broadcast_to(np.asarray(v, "float32")[:, None, None],
                               (len(items), 2, 2)).copy()

    prop = lambda k: np.array([it.properties.get(k, np.nan) for it in items],
                              dtype="float64")
    angles = dict(
        sun_zen=const(prop("MEAN_SOLAR_ZENITH_ANGLE")),
        sun_az=const(prop("MEAN_SOLAR_AZIMUTH_ANGLE")),
        view_zen=const(ang("MEAN_INCIDENCE_ZENITH_ANGLE", False)),
        view_az=const(ang("MEAN_INCIDENCE_AZIMUTH_ANGLE", True)),
    )
    ax = np.array([bounds.xmin, bounds.xmax])
    ay = np.array([bounds.ymax, bounds.ymin])
    return obs, scl, angles, ax, ay
