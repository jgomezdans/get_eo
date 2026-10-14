"""Resolve STAC items to the hrefs of the layers we need."""

from __future__ import annotations

import re

# (STAC asset key per source, S2 band id, nominal centre wavelength [nm],
#  native resolution [m])
S2_BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")

S2_BAND_INFO = {
    "B02": (490.0, 10), "B03": (560.0, 10), "B04": (665.0, 10),
    "B05": (705.0, 20), "B06": (740.0, 20), "B07": (783.0, 20),
    "B08": (842.0, 10), "B8A": (865.0, 20),
    "B11": (1610.0, 20), "B12": (2190.0, 20),
}

# S2 band id -> asset key. Planetary uses the band ids themselves.
_AWS_KEYS = {
    "B02": "blue", "B03": "green", "B04": "red", "B05": "rededge1",
    "B06": "rededge2", "B07": "rededge3", "B08": "nir", "B8A": "nir08",
    "B11": "swir16", "B12": "swir22",
}

# SCL classes to mask: NoData, Saturated, CloudShadow, CloudMedium,
# CloudHigh, Cirrus, Snow
SCL_MASK_VALUES = frozenset({0, 1, 3, 8, 9, 10, 11})

S2_SOURCES = ("aws", "planetary", "cdse", "gee")

# CDSE keys are '<band>_<res>m' at each band's native resolution.
_CDSE_KEYS = {b: f"{b}_{int(info[1])}m" for b, info in S2_BAND_INFO.items()}

S2_SCL_KEY = {"aws": "scl", "planetary": "SCL", "cdse": "SCL_20m"}
S2_MTD_KEY = {"aws": "granule_metadata", "planetary": "granule-metadata",
              "cdse": "granule_metadata"}
# Reference 10 m band whose proj:transform/shape give the tile grid.
S2_GRID_KEY = {"aws": "blue", "planetary": "B02", "cdse": "B02_10m"}

# Index of each band in MTD_TL.xml Viewing_Incidence_Angles_Grids bandId.
S2_MTD_BAND_ID = {
    "B01": 0, "B02": 1, "B03": 2, "B04": 3, "B05": 4, "B06": 5, "B07": 6,
    "B08": 7, "B8A": 8, "B09": 9, "B10": 10, "B11": 11, "B12": 12,
}

ECO_ID = re.compile(
    r"ECOv(?P<ver>\d{3})_(?P<product>L1CT_RAD|L2T_LSTE)_"
    r"(?P<orbit>\d{5})_(?P<scene>\d{3})_(?P<tile>\w{5})_"
    r"(?P<start>\d{8}T\d{6})_(?P<build>\d{4})_(?P<rev>\d{2})"
)

ECO_HTTPS_PREFIX = "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/"

# ECOSTRESS TIR bands we keep (D6) and their nominal centres [nm].
ECO_BANDS = {2: ("TIR2", 8780.0), 4: ("TIR4", 10490.0), 5: ("TIR5", 12090.0)}


def parse_eco_id(item_id: str) -> dict:
    """Split an ECOSTRESS granule id into its fields (orbit/scene as int)."""
    m = ECO_ID.fullmatch(item_id)
    if m is None:
        raise ValueError(f"Not an ECOSTRESS tiled-product id: {item_id!r}")
    d = m.groupdict()
    d["orbit"] = int(d["orbit"])
    d["scene"] = int(d["scene"])
    return d


def eco_layer_hrefs(item) -> dict[str, str]:
    """Map layer name -> https COG href for an ECOSTRESS STAC item.

    Asset keys are not layer names, so the layer is taken from the href
    stem. ``s3://``, thumbnail and browse assets are ignored.
    """
    prefix = item.id + "_"
    out = {}
    for asset in item.assets.values():
        href = asset.href
        if not (href.startswith(ECO_HTTPS_PREFIX) and href.endswith(".tif")):
            continue
        stem = href.rsplit("/", 1)[-1][:-4]
        if prefix in stem:
            out[stem.split(prefix, 1)[1]] = href
    return out


def s2_asset_hrefs(item, source: str, bands=S2_BANDS) -> dict[str, str]:
    """Map S2 band id (plus 'SCL') -> href for a STAC item."""
    if source == "gee":
        raise ValueError("GEE items have no asset hrefs")
    keys = {"aws": _AWS_KEYS, "cdse": _CDSE_KEYS}.get(
        source, {b: b for b in bands})
    out = {}
    for b in bands:
        out[b] = item.assets[keys[b]].href
    out["SCL"] = item.assets[S2_SCL_KEY[source]].href
    return out


def s2_mtd_href(item, source: str) -> str:
    return item.assets[S2_MTD_KEY[source]].href
