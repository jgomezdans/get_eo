"""STAC search, de-duplication and tile selection."""

from __future__ import annotations

import logging
import re
import warnings
from collections import defaultdict

import pystac_client
from pyproj import CRS
from shapely.geometry import box, mapping, shape
from shapely.geometry.base import BaseGeometry

from get_eo._assets import S2_GRID_KEY, parse_eco_id
from get_eo._grid import TileGrid, aoi_to_crs
from get_eo._log import timed

log = logging.getLogger(__name__)

CMR_STAC_URL = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD"
_S2_STAC = {
    "aws": "https://earth-search.aws.element84.com/v1",
    "planetary": "https://planetarycomputer.microsoft.com/api/stac/v1",
    "cdse": "https://stac.dataspace.copernicus.eu/v1",
}
_TILE_KEYS = ("grid:code", "s2:mgrs_tile", "mgrs:tile")


def _datetime_range(start: str, end: str) -> str:
    return f"{start}T00:00:00Z/{end}T23:59:59Z"


# ---------------------------------------------------------------------------
# Sentinel-2
# ---------------------------------------------------------------------------

def s2_tile_of(item) -> str | None:
    """MGRS tile name ('14TQL') of an S2 STAC item."""
    for key in _TILE_KEYS:
        v = item.properties.get(key)
        if v:
            return str(v).replace("MGRS-", "").lstrip("T")
    # AWS ids: S2B_14TQL_20220708_0_L2A
    parts = item.id.split("_")
    if len(parts) >= 3 and len(parts[1]) == 5:
        return parts[1]
    return None


def search_s2(aoi: BaseGeometry, start: str, end: str, source: str = "aws",
              max_cloud: float = 100.0, tile: str | None = None,
              gee_project: str | None = None) -> list:
    """All L2A items touching the AOI, de-duplicated.

    For ``source='gee'`` the items are :class:`get_eo._gee.GeeItem`, which
    carry the same ``id``/``datetime``/``properties`` as STAC items.
    """
    if source == "gee":
        from get_eo._gee import search_s2_gee
        return search_s2_gee(aoi, start, end, max_cloud, tile, gee_project)
    if source not in _S2_STAC:
        raise ValueError(
            f"Unknown S2 source {source!r}; supported: aws, planetary, "
            "cdse, gee")
    url = _S2_STAC[source]
    kwargs = {}
    if max_cloud < 100:
        kwargs["query"] = {"eo:cloud_cover": {"lte": max_cloud}}
    with timed(log, "S2 STAC search on %s (%s, %s..%s, max_cloud=%g)",
               source, url, start, end, max_cloud):
        client = pystac_client.Client.open(url)
        search = client.search(
            collections=["sentinel-2-l2a"], intersects=mapping(aoi),
            datetime=_datetime_range(start, end), **kwargs)
        found = list(search.items())
    items = dedupe_s2(found, source)
    log.info("S2 %s: %d items found, %d after de-duplication, tiles: %s",
             source, len(found), len(items),
             sorted({s2_tile_of(i) for i in items} - {None}) or "none")
    return items


def dedupe_s2(items: list, source: str) -> list:
    """Keep one item per acquisition and tile.

    AWS ids carry a reprocessing sequence (``_0_``, ``_1_``): keep the
    highest. Planetary can hold several processing generations of one
    datatake (also across processing baselines, which Planetary
    keeps side by side): keep the latest ``s2:generation_time``. CDSE holds one item
    per processing baseline: keep the highest ``processing:version``.
    GEE has one image per granule already.
    """
    if source == "gee":
        return sorted(items, key=lambda i: i.datetime)
    best: dict[tuple, object] = {}
    for it in items:
        tile = s2_tile_of(it)
        if source == "aws":
            platform = it.properties.get("platform")
            day = it.datetime.strftime("%Y%m%d")
            key = (platform, tile, day)
            rank = int(it.properties.get("s2:sequence", 0))
        elif source == "cdse":
            platform = it.properties.get("platform")
            key = (platform, tile, it.datetime.strftime("%Y%m%d"),
                   it.properties.get("sat:relative_orbit"))
            rank = (str(it.properties.get("processing:version", "")), it.id)
        else:
            # datatake_id ends in the baseline (..._N04.00 / ..._N05.10);
            # drop it so reprocessed granules of one datatake collide.
            dt = it.properties.get("s2:datatake_id")
            key = (re.sub(r"_N\d+\.\d+$", "", dt) if dt else it.id, tile)
            rank = it.properties.get("s2:generation_time", "")
        if key not in best or rank > best[key][0]:
            best[key] = (rank, it)
    return sorted((v[1] for v in best.values()), key=lambda i: i.datetime)


def _grid_from_file(href: str, source: str):
    """(code, transform, (ny, nx)) from the file header, for items that
    carry no ``proj:`` metadata (CDSE JP2s)."""
    import rasterio
    from get_eo._env import gdal_env_for
    log.info("item has no proj: metadata; reading grid from %s", href)
    if href.startswith("s3://"):
        href = "/vsis3/" + href[len("s3://"):]
    with rasterio.Env(**gdal_env_for(source)), rasterio.open(href) as src:
        return (src.crs.to_string(), tuple(src.transform)[:6],
                (src.height, src.width))


def s2_grid(item, source: str, res: float = 10.0) -> TileGrid:
    """Native 10 m tile grid from the item's ``proj:`` metadata.

    Falls back to reading the 10 m blue band's header when the STAC item
    has none.
    """
    p = item.properties
    if source == "gee":                       # filled in by search_s2_gee
        code, tr, (ny, nx) = p["proj:code"], p["proj:transform"], p["proj:shape"]
        return TileGrid(crs=CRS.from_user_input(code), x0=float(tr[2]),
                        y0=float(tr[5]), res=float(tr[0]), nx=int(nx),
                        ny=int(ny))
    asset = item.assets[S2_GRID_KEY[source]]
    ex = asset.extra_fields
    code = (p.get("proj:code") or ex.get("proj:code")
            or (f"EPSG:{p['proj:epsg']}" if "proj:epsg" in p else None)
            or (f"EPSG:{ex['proj:epsg']}" if "proj:epsg" in ex else None))
    tr = ex.get("proj:transform") or p.get("proj:transform")
    shape = ex.get("proj:shape") or p.get("proj:shape")
    if not (code and tr and shape):
        code, tr, shape = _grid_from_file(asset.href, source)
    ny, nx = shape
    return TileGrid(crs=CRS.from_user_input(code), x0=float(tr[2]),
                    y0=float(tr[5]), res=float(tr[0]), nx=int(nx), ny=int(ny))


# ---------------------------------------------------------------------------
# ECOSTRESS
# ---------------------------------------------------------------------------

def search_eco(aoi: BaseGeometry, start: str, end: str, product: str = "L1CT_RAD",
               version: str = "002", tile: str | None = None) -> list:
    """ECOSTRESS tiled granules (L1CT_RAD or L2T_LSTE) touching the AOI.

    Item properties carry no cloud/angle information, so the only filters
    are time, tile and footprint. Duplicate builds are collapsed.
    """
    with timed(log, "ECOSTRESS %s v%s STAC search on CMR (%s..%s)",
               product, version, start, end):
        client = pystac_client.Client.open(CMR_STAC_URL)
        search = client.search(
            collections=[f"ECO_{product}_{version}"], bbox=aoi.bounds,
            datetime=_datetime_range(start, end))
        found = list(search.items())
    best: dict[tuple, tuple] = {}
    for it in found:
        try:
            f = parse_eco_id(it.id)
        except ValueError:
            continue
        if tile and f["tile"] != tile:
            continue
        if it.geometry is not None and not shape(it.geometry).intersects(aoi):
            continue
        key = (f["orbit"], f["scene"], f["tile"])
        rank = (f["build"], f["rev"])
        if key not in best or rank > best[key][0]:
            best[key] = (rank, it)
    out = sorted((v[1] for v in best.values()),
                 key=lambda i: parse_eco_id(i.id)["start"])
    log.info("ECOSTRESS %s: %d granules found, %d kept (on AOI%s, "
             "latest build)", product, len(found), len(out),
             f", tile {tile}" if tile else "")
    return out


# ---------------------------------------------------------------------------
# Tile selection
# ---------------------------------------------------------------------------

def choose_tile(aoi: BaseGeometry, grids: dict[str, TileGrid]) -> str:
    """Pick the tile that best contains the AOI.

    Score: fraction of the AOI inside the tile, then distance of the AOI
    centroid from the tile edge (most interior wins). Deterministic, so
    the same AOI always lands on the same grid.
    """
    if not grids:
        raise ValueError("No candidate tiles")
    scored = []
    for name, g in grids.items():
        proj = aoi_to_crs(aoi, g.crs)
        tile = box(*g.bounds)
        inter = proj.intersection(tile).area
        cover = inter / proj.area if proj.area > 0 else float(tile.contains(proj.centroid))
        interior = tile.exterior.distance(proj.centroid) if tile.contains(proj.centroid) else 0.0
        scored.append((round(cover, 6), interior, name))
    scored.sort(reverse=True)
    best = scored[0]
    log.info("tile candidates (AOI cover, name): %s",
             ", ".join(f"{n} {c:.0%}" for c, _, n in scored))
    if best[0] < 1.0 - 1e-6:
        warnings.warn(
            f"AOI is not fully inside any single tile; using {best[2]} "
            f"({best[0]:.0%} of the AOI). Pass tile= to choose another.",
            stacklevel=3)
    return best[2]


def group_by_tile(items: list, tile_fn) -> dict[str, list]:
    out = defaultdict(list)
    for it in items:
        out[tile_fn(it)].append(it)
    return dict(out)
