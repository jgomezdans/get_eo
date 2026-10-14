"""SoilGrids 2.0 (ISRIC, 250 m) as an xarray Dataset over an AOI.

Reads windows from ISRIC's public VRTs (no account), resamples by nearest
neighbour onto a regular projected grid, and converts the stored integer
units to conventional ones. The result is small, so it is loaded eagerly.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rasterio
import xarray as xr
from affine import Affine
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from get_eo._aoi import resolve_aoi
from get_eo._grid import Bounds, aoi_mask, aoi_to_crs

SOILGRIDS_URL = ("/vsicurl/https://files.isric.org/soilgrids/latest/data/"
                 "{prop}/{prop}_{depth}_{stat}.vrt")
NODATA = -32768
DEPTHS = ("0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm")
STATS = ("mean", "Q0.05", "Q0.5", "Q0.95", "uncertainty")

# property: (divide stored value by, units after conversion, long name)
PROPERTIES = {
    "clay": (10.0, "%", "Clay content (<2 µm), mass fraction"),
    "sand": (10.0, "%", "Sand content (50-2000 µm), mass fraction"),
    "silt": (10.0, "%", "Silt content (2-50 µm), mass fraction"),
    "bdod": (100.0, "kg dm-3", "Bulk density of the fine earth fraction"),
    "cfvo": (10.0, "%", "Coarse fragments (>2 mm), volumetric fraction"),
    "soc": (10.0, "g kg-1", "Soil organic carbon, fine earth fraction"),
    "nitrogen": (100.0, "g kg-1", "Total nitrogen"),
    "phh2o": (10.0, "1", "pH in water"),
    "cec": (10.0, "cmol(c) kg-1", "Cation exchange capacity at pH 7"),
    "ocd": (10.0, "kg m-3", "Organic carbon density"),
}
DEFAULT_PROPERTIES = ("clay", "sand", "silt", "bdod", "cfvo", "soc",
                      "nitrogen", "phh2o", "cec")
_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".vrt,.tif",
            GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="2",
            GDAL_HTTP_MULTIPLEX="YES", VSI_CACHE="TRUE")


def utm_crs_for(lon: float, lat: float) -> CRS:
    zone = int((lon + 180.0) // 6.0) % 60 + 1
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def snap_bounds(xmin, ymin, xmax, ymax, res: float) -> Bounds:
    """Snap outwards to multiples of ``res`` (absolute, in the CRS)."""
    x0, y0 = np.floor(xmin / res) * res, np.floor(ymin / res) * res
    x1, y1 = np.ceil(xmax / res) * res, np.ceil(ymax / res) * res
    if x1 <= x0:
        x1 = x0 + res
    if y1 <= y0:
        y1 = y0 + res
    return Bounds(float(x0), float(y0), float(x1), float(y1))


def _read(prop, depth, stat, crs, bounds: Bounds, res, nx, ny) -> np.ndarray:
    """One property/depth/stat window as float32 in stored units (NaN =
    no data)."""
    dst = Affine(res, 0, bounds.xmin, 0, -res, bounds.ymax)
    with rasterio.Env(**_ENV):
        with rasterio.open(SOILGRIDS_URL.format(
                prop=prop, depth=depth, stat=stat)) as src:
            with WarpedVRT(src, crs=crs, transform=dst, width=nx, height=ny,
                           resampling=Resampling.nearest,
                           src_nodata=NODATA, nodata=NODATA) as vrt:
                a = vrt.read(1)
    out = a.astype("float32")
    out[a == NODATA] = np.nan
    return out


def open_soilgrids(aoi, properties=DEFAULT_PROPERTIES, depths=DEPTHS,
                   stats=("mean",), *, buffer_m: float | None = None,
                   crs=None, res: float = 250.0, workers: int = 8,
                   ) -> xr.Dataset:
    """SoilGrids properties over an AOI.

    Parameters
    ----------
    aoi
        Anything :func:`get_eo.open_aoi` accepts.
    properties : iterable of str
        Keys of ``PROPERTIES`` (clay, sand, silt, bdod, cfvo, soc,
        nitrogen, phh2o, cec, ocd).
    depths : iterable of str
        Subset of ``DEPTHS`` ('0-5cm' ... '100-200cm').
    stats : iterable of str
        'mean' (variable ``clay``) and/or 'Q0.05', 'Q0.5', 'Q0.95',
        'uncertainty' (variables ``clay_Q0.05`` ...).
    crs : CRS-like, optional
        Output CRS; default is the UTM zone of the AOI centroid.
    res : float
        Output cell size [m]; 250 is the native resolution.

    Returns
    -------
    xr.Dataset
        Variables on ``(depth, y, x)`` in conventional units (see each
        variable's ``units``), NaN where SoilGrids has no data (water,
        urban, rock). Coordinates ``depth_top_cm``, ``depth_bottom_cm``,
        ``depth_mid_cm`` and ``inside_aoi`` (fraction of the cell inside
        the polygon). Cells are aligned to multiples of ``res``.
    """
    properties, depths, stats = list(properties), list(depths), list(stats)
    for p in properties:
        if p not in PROPERTIES:
            raise ValueError(f"Unknown property {p!r}; "
                             f"choose from {sorted(PROPERTIES)}")
    for d in depths:
        if d not in DEPTHS:
            raise ValueError(f"Unknown depth {d!r}; choose from {DEPTHS}")
    for s in stats:
        if s not in STATS:
            raise ValueError(f"Unknown stat {s!r}; choose from {STATS}")

    geom = resolve_aoi(aoi, buffer_m)
    crs = CRS.from_user_input(crs) if crs is not None else utm_crs_for(
        geom.centroid.x, geom.centroid.y)
    g = aoi_to_crs(geom, crs)
    bounds = snap_bounds(*g.bounds, res)
    nx = int(round((bounds.xmax - bounds.xmin) / res))
    ny = int(round((bounds.ymax - bounds.ymin) / res))
    x, y = bounds.coords(res)

    jobs = [(p, d, s) for p in properties for s in stats for d in depths]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        arrs = dict(zip(jobs, pool.map(
            lambda j: _read(*j, crs, bounds, res, nx, ny), jobs)))

    data = {}
    for p in properties:
        div, units, long_name = PROPERTIES[p]
        for s in stats:
            stack = np.stack([arrs[(p, d, s)] for d in depths])
            # 'uncertainty' is a percentage, not in the property's units
            scale, u = (1.0, "%") if s == "uncertainty" else (div, units)
            name = p if s == "mean" else f"{p}_{s}"
            data[name] = (("depth", "y", "x"), stack / scale, dict(
                long_name=long_name + ("" if s == "mean" else f" ({s})"),
                units=u, soilgrids_stat=s))
    top = np.array([float(d.split("-")[0]) for d in depths])
    bot = np.array([float(d.split("-")[1].replace("cm", "")) for d in depths])
    ds = xr.Dataset(
        data,
        coords=dict(
            depth=depths, x=x, y=y,
            depth_top_cm=("depth", top), depth_bottom_cm=("depth", bot),
            depth_mid_cm=("depth", (top + bot) / 2.0),
            inside_aoi=(("y", "x"), aoi_mask(geom, bounds, crs, res))),
        attrs=dict(source="SoilGrids 2.0 (ISRIC)", resolution_m=res,
                   license="CC BY 4.0", resampling="nearest"))
    return ds.rio.write_crs(crs)
