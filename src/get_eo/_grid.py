"""Tile grids and AOI windows.

All windows are expressed in *metres in the tile CRS* (``Bounds``) and are
converted to a pixel window per resolution on use. A single snapped
bounds therefore covers exactly the same extent at 10, 20 and 70 m.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyproj import CRS, Transformer
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shp_transform


@dataclass(frozen=True)
class TileGrid:
    """Native grid of one tile at one resolution."""
    crs: CRS
    x0: float          # upper-left corner [m]
    y0: float
    res: float         # pixel size [m]
    nx: int
    ny: int

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax)."""
        return (self.x0, self.y0 - self.ny * self.res,
                self.x0 + self.nx * self.res, self.y0)

    @property
    def transform_coeffs(self) -> tuple[float, ...]:
        return (self.res, 0.0, self.x0, 0.0, -self.res, self.y0)


@dataclass(frozen=True)
class Bounds:
    """A snapped extent in tile CRS metres."""
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def pixel_window(self, grid: TileGrid) -> tuple[int, int, int, int]:
        """(row_off, col_off, n_rows, n_cols) on ``grid``."""
        col0 = (self.xmin - grid.x0) / grid.res
        row0 = (grid.y0 - self.ymax) / grid.res
        ncol = (self.xmax - self.xmin) / grid.res
        nrow = (self.ymax - self.ymin) / grid.res
        for v in (col0, row0, ncol, nrow):
            if abs(v - round(v)) > 1e-6:
                raise ValueError(
                    f"Bounds {self} are not aligned to a {grid.res} m grid")
        return int(round(row0)), int(round(col0)), int(round(nrow)), int(round(ncol))

    def coords(self, res: float) -> tuple[np.ndarray, np.ndarray]:
        """Pixel-centre (x, y) coordinates at ``res`` (y descending)."""
        nx = int(round((self.xmax - self.xmin) / res))
        ny = int(round((self.ymax - self.ymin) / res))
        x = self.xmin + res * (np.arange(nx) + 0.5)
        y = self.ymax - res * (np.arange(ny) + 0.5)
        return x, y


def aoi_to_crs(aoi: BaseGeometry, crs: CRS) -> BaseGeometry:
    """Project an EPSG:4326 (lon, lat) geometry to ``crs``."""
    t = Transformer.from_crs(4326, crs, always_xy=True)
    return shp_transform(t.transform, aoi)


def aoi_bounds(aoi: BaseGeometry, grid: TileGrid, snap_m: float) -> Bounds:
    """Snap the AOI's bounds *outwards* to multiples of ``snap_m`` counted
    from the tile origin, clipped to the tile.

    Raises ValueError if the AOI does not intersect the tile.
    """
    g = aoi_to_crs(aoi, grid.crs)
    tile = box(*grid.bounds)
    if not g.intersects(tile):
        raise ValueError("AOI does not intersect the tile")
    xmin, ymin, xmax, ymax = g.bounds
    gx0, gy0, gx1, gy1 = grid.bounds
    # Tile extent in whole snap cells (drop a ragged remainder at the edge)
    ncx = int(np.floor((gx1 - grid.x0) / snap_m + 1e-9))
    ncy = int(np.floor((grid.y0 - gy0) / snap_m + 1e-9))
    c0 = np.clip(np.floor((xmin - grid.x0) / snap_m), 0, ncx - 1)
    c1 = np.clip(np.ceil((xmax - grid.x0) / snap_m), c0 + 1, ncx)
    r0 = np.clip(np.floor((grid.y0 - ymax) / snap_m), 0, ncy - 1)
    r1 = np.clip(np.ceil((grid.y0 - ymin) / snap_m), r0 + 1, ncy)
    return Bounds(
        xmin=grid.x0 + c0 * snap_m, xmax=grid.x0 + c1 * snap_m,
        ymax=grid.y0 - r0 * snap_m, ymin=grid.y0 - r1 * snap_m,
    )


def aoi_mask(aoi: BaseGeometry, bounds: Bounds, crs: CRS, res: float,
             ) -> np.ndarray:
    """Fraction (0..1) of each ``res`` pixel inside the AOI polygon.

    Sub-sampled 4x4 per pixel: cheap and good enough for a mask.
    """
    from rasterio import features
    from rasterio.transform import from_origin

    g = aoi_to_crs(aoi, crs)
    nx = int(round((bounds.xmax - bounds.xmin) / res))
    ny = int(round((bounds.ymax - bounds.ymin) / res))
    ss = 4
    tr = from_origin(bounds.xmin, bounds.ymax, res / ss, res / ss)
    fine = features.rasterize([(g, 1)], out_shape=(ny * ss, nx * ss),
                              transform=tr, fill=0, dtype="uint8")
    return fine.reshape(ny, ss, nx, ss).mean(axis=(1, 3)).astype("float32")


def chunk_windows(nrow: int, ncol: int, chunk: int = 1024):
    """Yield ((r0, r1), (c0, c1)) sub-windows of at most ``chunk`` pixels."""
    rs = list(range(0, nrow, chunk)) + [nrow]
    cs = list(range(0, ncol, chunk)) + [ncol]
    return [[((rs[i], rs[i + 1]), (cs[j], cs[j + 1]))
             for j in range(len(cs) - 1)] for i in range(len(rs) - 1)]
