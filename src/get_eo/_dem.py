"""Copernicus GLO-30 DEM (30 m) as an xarray Dataset over an AOI.

``elevation`` [m a.s.l.] plus ``slope`` and ``aspect`` [degrees] derived on
the output grid. Sources:

``aws`` (default)
    The public Copernicus DEM COGs on AWS Open Data. No account.
``gee``
    ``COPERNICUS/DEM/GLO30_2024_1`` from Earth Engine (``pip install 'get_eo[gee]'``).
"""

from __future__ import annotations

import math

import numpy as np
import rasterio
import xarray as xr
from affine import Affine
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.vrt import WarpedVRT

from get_eo._aoi import resolve_aoi
from get_eo._grid import Bounds, aoi_mask, aoi_to_crs
from get_eo._soil import snap_bounds, utm_crs_for

AWS_URL = ("https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif")
GEE_DEM = "COPERNICUS/DEM/GLO30_2024_1"
_NODATA = -9999.0
_ENV = dict(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_MAX_RETRY="5",
            GDAL_HTTP_RETRY_DELAY="2", VSI_CACHE="TRUE")


def tile_name(lat: int, lon: int) -> str:
    """Name of the 1 x 1 degree tile whose south-west corner is (lat, lon)."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return (f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_"
            f"{ew}{abs(lon):03d}_00_DEM")


def tiles_for(w: float, s: float, e: float, n: float) -> list[str]:
    """Tile names covering a lon/lat box."""
    return [tile_name(la, lo)
            for la in range(math.floor(s), math.floor(n) + 1)
            for lo in range(math.floor(w), math.floor(e) + 1)]


def slope_aspect(z: np.ndarray, res: float):
    """Slope and downslope aspect [degrees] of a north-up grid.

    Aspect is the compass direction the surface faces, clockwise from
    north (270 = faces west); NaN on flat cells. Central differences, so
    the outermost ring is only one-sided; pass a grid with a margin.
    """
    dz_dy_row, dz_dx = np.gradient(z.astype("float64"), res)
    dz_dn = -dz_dy_row                    # rows run south, so flip
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dn)))
    aspect = np.degrees(np.arctan2(-dz_dx, -dz_dn)) % 360.0
    aspect = aspect.astype("float32")
    aspect[aspect >= 360.0] = 0.0         # float32 rounding of 359.9999
    aspect[np.hypot(dz_dx, dz_dn) < 1e-9] = np.nan
    return slope.astype("float32"), aspect


def _read_aws(geom, crs, bounds: Bounds, res: float, nx: int, ny: int):
    dst = Affine(res, 0, bounds.xmin, 0, -res, bounds.ymax)
    out = np.full((ny, nx), np.nan, "float32")
    with rasterio.Env(**_ENV):
        for name in tiles_for(*geom.bounds):
            try:
                src = rasterio.open(AWS_URL.format(name=name))
            except RasterioIOError:         # no tile (open ocean)
                continue
            with src, WarpedVRT(src, crs=crs, transform=dst, width=nx,
                                height=ny, resampling=Resampling.bilinear,
                                nodata=_NODATA) as vrt:
                a = vrt.read(1)
            ok = a != _NODATA
            out[ok] = a[ok]
    return out


def _read_gee(geom, crs, bounds: Bounds, res: float, nx: int, ny: int,
              project=None):
    import io

    from get_eo._gee import init_ee
    ee = init_ee(project, need_xee=False)
    ic = ee.ImageCollection(GEE_DEM).select("DEM")
    img = (ic.mosaic().setDefaultProjection(ic.first().projection())
           .resample("bilinear"))
    raw = ee.data.computePixels({
        "expression": img, "fileFormat": "NPY", "bandIds": ["DEM"],
        "grid": {"dimensions": {"width": nx, "height": ny},
                 "affineTransform": {
                     "scaleX": res, "shearX": 0, "translateX": bounds.xmin,
                     "shearY": 0, "scaleY": -res, "translateY": bounds.ymax},
                 "crsCode": crs.to_string()}})
    return np.load(io.BytesIO(raw))["DEM"].astype("float32")


def open_dem(aoi, *, source: str = "aws", buffer_m: float | None = None,
             crs=None, res: float = 30.0, gee_project: str | None = None,
             ) -> xr.Dataset:
    """Elevation, slope and aspect over an AOI.

    Parameters
    ----------
    aoi
        Anything :func:`get_eo.open_aoi` accepts.
    source : {'aws', 'gee'}
    crs : CRS-like, optional
        Output CRS; default is the UTM zone of the AOI centroid.
    res : float
        Cell size [m]; 30 is the native resolution. Cells are aligned to
        multiples of ``res`` in the CRS.

    Returns
    -------
    xr.Dataset
        ``elevation`` [m], ``slope`` [deg], ``aspect`` [deg clockwise from
        north, direction the slope faces] on ``(y, x)``, with
        ``inside_aoi``. Bilinear resampling.
    """
    if source not in ("aws", "gee"):
        raise ValueError(f"Unknown DEM source {source!r}; use 'aws' or 'gee'")
    geom = resolve_aoi(aoi, buffer_m)
    crs = CRS.from_user_input(crs) if crs is not None else utm_crs_for(
        geom.centroid.x, geom.centroid.y)
    b = snap_bounds(*aoi_to_crs(geom, crs).bounds, res)
    # one extra ring of cells so slope/aspect are central differences
    m = Bounds(b.xmin - res, b.ymin - res, b.xmax + res, b.ymax + res)
    nx = int(round((m.xmax - m.xmin) / res))
    ny = int(round((m.ymax - m.ymin) / res))
    # geographic extent needed (AOI plus margin), for tile selection
    from shapely.geometry import box
    from pyproj import Transformer
    t = Transformer.from_crs(crs, 4326, always_xy=True)
    xs, ys = t.transform([m.xmin, m.xmax, m.xmin, m.xmax],
                         [m.ymin, m.ymin, m.ymax, m.ymax])
    ll = box(min(xs), min(ys), max(xs), max(ys))
    z = (_read_aws(ll, crs, m, res, nx, ny) if source == "aws"
         else _read_gee(ll, crs, m, res, nx, ny, gee_project))
    slope, aspect = slope_aspect(z, res)
    inner = (slice(1, -1), slice(1, -1))
    x, y = b.coords(res)
    ds = xr.Dataset(
        dict(elevation=(("y", "x"), z[inner], dict(
                long_name="Elevation above mean sea level", units="m")),
             slope=(("y", "x"), slope[inner], dict(
                long_name="Terrain slope", units="degrees")),
             aspect=(("y", "x"), aspect[inner], dict(
                long_name="Slope aspect (direction the slope faces, "
                          "clockwise from north)", units="degrees"))),
        coords=dict(x=x, y=y,
                    inside_aoi=(("y", "x"), aoi_mask(geom, b, crs, res))),
        attrs=dict(source=("Copernicus GLO-30 (AWS Open Data)" if source == "aws"
                           else f"{GEE_DEM} (Earth Engine)"),
                   resolution_m=res, resampling="bilinear",
                   license="Copernicus DEM, free for any use with attribution"))
    return ds.rio.write_crs(crs)
