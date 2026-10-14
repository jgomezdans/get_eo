"""Turn the many ways a user can point at a region into one geometry."""

from __future__ import annotations

from os import PathLike

from shapely.geometry import Point, box, shape
from shapely.geometry.base import BaseGeometry

from get_eo._geojson import load_geojson


def resolve_aoi(aoi, buffer_m: float | None = None) -> BaseGeometry:
    """Return the AOI as a valid shapely geometry in EPSG:4326.

    Accepted forms:

    * a shapely geometry, WKT string, or path to a GeoJSON file;
    * a GeoJSON dict (geometry, Feature or FeatureCollection);
    * anything with ``__geo_interface__`` (e.g. a GeoDataFrame);
    * ``(west, south, east, north)`` bounding box;
    * ``(lon, lat)`` point, which needs ``buffer_m`` (a square of
      half-width ``buffer_m`` metres is used).
    """
    if isinstance(aoi, (tuple, list)) and all(
            isinstance(v, (int, float)) for v in aoi):
        if len(aoi) == 4:
            w, s, e, n = aoi
            return box(w, s, e, n)
        if len(aoi) == 2:
            return _buffered_point(aoi[0], aoi[1], buffer_m)
        raise ValueError("A numeric AOI must be (w, s, e, n) or (lon, lat)")

    if isinstance(aoi, dict):
        return load_geojson(_geom_from_geojson(aoi))

    if isinstance(aoi, (str, PathLike, BaseGeometry)):
        geom = load_geojson(aoi)
    elif hasattr(aoi, "__geo_interface__"):
        gi = aoi.__geo_interface__
        geom = load_geojson(_geom_from_geojson(gi))
    else:
        raise TypeError(f"Cannot interpret AOI of type {type(aoi).__name__}")

    if isinstance(geom, Point):
        return _buffered_point(geom.x, geom.y, buffer_m)
    return geom


def _geom_from_geojson(gj: dict) -> BaseGeometry:
    if gj.get("type") == "FeatureCollection":
        from shapely.ops import unary_union
        return unary_union([shape(f["geometry"]) for f in gj["features"]])
    if gj.get("type") == "Feature":
        return shape(gj["geometry"])
    return shape(gj)


def _buffered_point(lon: float, lat: float, buffer_m: float | None):
    if not buffer_m:
        raise ValueError(
            "A point AOI needs buffer_m, e.g. aoi=(lon, lat), buffer_m=1500")
    import math
    dlat = buffer_m / 111_320.0
    dlon = buffer_m / (111_320.0 * max(math.cos(math.radians(lat)), 1e-6))
    return box(lon - dlon, lat - dlat, lon + dlon, lat + dlat)
