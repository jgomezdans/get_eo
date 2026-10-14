"""ECOSTRESS L1CT_RAD and L2T_LSTE as lazy xarray Datasets (70 m tiles)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import dask.array as da
import numpy as np
import rasterio
import xarray as xr
from pyproj import Proj, Transformer
from rasterio.windows import Window

from get_eo._assets import ECO_BANDS, eco_layer_hrefs, parse_eco_id
from get_eo._env import gdal_env_for
from get_eo._geometry import (
    block_median_quiet, fill_nearest, nanmean_quiet,
)
from get_eo._grid import Bounds, TileGrid, aoi_mask
from get_eo._read import lazy_layer, read_window
from get_eo._solar import solar_position
from get_eo._uncertainty import eco_sigma_radiance

BLOCK_PX = 16          # coarse angle grid stride (about 1.1 km)
MIN_VZA_FOR_AZIMUTH = 1.0


def eco_grid(item, env: dict) -> TileGrid:
    """Native 70 m tile grid, from one header read of any layer."""
    from pyproj import CRS
    href = next(iter(eco_layer_hrefs(item).values()))
    with rasterio.Env(**env), rasterio.open(href) as src:
        t = src.transform
        return TileGrid(crs=CRS.from_user_input(src.crs.to_wkt()),
                        x0=t.c, y0=t.f, res=t.a, nx=src.width, ny=src.height)


def eco_time(item) -> np.datetime64:
    """Granule start time (UTC) from the item properties or its id."""
    s = item.properties.get("start_datetime")
    if s:
        return np.datetime64(s.replace("Z", "").split("+")[0], "ns")
    st = parse_eco_id(item.id)["start"]       # 20220702T081925
    return np.datetime64(
        f"{st[:4]}-{st[4:6]}-{st[6:8]}T{st[9:11]}:{st[11:13]}:{st[13:15]}", "ns")


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def eco_view_azimuth(vza: np.ndarray, step_m: float, crs,
                     x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Direction from each node towards the sensor, from the VZA gradient.

    ECOSTRESS scans across track and (unsigned) VZA grows roughly linearly
    with distance from the ground track, so the direction towards the
    sensor is the direction of steepest VZA *descent*. Azimuths are
    returned clockwise from **true** north.

    Parameters
    ----------
    vza : (ny, nx) array
        View zenith [deg] on a regular grid, rows running north to south.
    step_m : float
        Node spacing [m].
    crs : pyproj-compatible CRS of the grid.
    x, y : 1-D arrays of node coordinates [m] (length nx, ny).

    Where VZA < 1 deg the direction is ill defined and set to 0.
    """
    v = fill_nearest(vza)
    if min(v.shape) < 2:
        return np.zeros_like(vza, dtype="float32")
    gy_rows, gx = np.gradient(v, step_m)          # d/d(row), d/d(col)
    gx_east = gx
    gy_north = -gy_rows                           # rows go south
    az_grid = np.degrees(np.arctan2(-gx_east, -gy_north)) % 360.0

    # grid north -> true north: add meridian convergence
    xx, yy = np.meshgrid(x, y)
    lon, lat = Transformer.from_crs(crs, 4326, always_xy=True).transform(xx, yy)
    conv = Proj(crs).get_factors(lon, lat).meridian_convergence
    az = (az_grid + conv) % 360.0
    az[np.nan_to_num(vza, nan=0.0) < MIN_VZA_FOR_AZIMUTH] = 0.0
    return az.astype("float32")


def block_nodes(grid: TileGrid, bounds: Bounds, pad_blocks: int = 1):
    """Block index ranges (and node coordinates) of the coarse angle grid.

    Blocks are ``BLOCK_PX`` pixels, aligned to the tile origin so every
    granule shares the same nodes.
    """
    b = BLOCK_PX * grid.res
    nbx, nby = grid.nx // BLOCK_PX, grid.ny // BLOCK_PX
    i0 = max(int(np.floor((bounds.xmin - grid.x0) / b)) - pad_blocks, 0)
    i1 = min(int(np.ceil((bounds.xmax - grid.x0) / b)) + pad_blocks, nbx)
    j0 = max(int(np.floor((grid.y0 - bounds.ymax) / b)) - pad_blocks, 0)
    j1 = min(int(np.ceil((grid.y0 - bounds.ymin) / b)) + pad_blocks, nby)
    ax = grid.x0 + (np.arange(i0, i1) + 0.5) * b
    ay = grid.y0 - (np.arange(j0, j1) + 0.5) * b
    return (j0, j1, i0, i1), ax, ay


def _tile_vza_blocks(href: str, grid: TileGrid, env: dict) -> np.ndarray:
    """Block-median VZA over the whole tile: (ny/16, nx/16)."""
    nby, nbx = grid.ny // BLOCK_PX, grid.nx // BLOCK_PX
    a = read_window(href, (0, 0, nby * BLOCK_PX, nbx * BLOCK_PX), env,
                    grid.transform_coeffs)
    return block_median_quiet(a, BLOCK_PX)


def _plane_fit(v: np.ndarray, x: np.ndarray, y: np.ndarray, x_c: float,
               y_c: float, sigma_m: float = 30_000.0):
    """Gaussian-weighted plane ``a + b*(x-x_c) + c*(y-y_c)`` through the
    finite nodes of ``v`` (rows <-> y, cols <-> x). Returns (a, b, c) or
    None if fewer than 6 finite nodes."""
    xx, yy = np.meshgrid(x - x_c, y - y_c)
    m = np.isfinite(v)
    if m.sum() < 6:
        return None
    w = np.exp(-0.5 * (xx[m] ** 2 + yy[m] ** 2) / sigma_m ** 2)
    A = np.column_stack([np.ones(m.sum()), xx[m], yy[m]]) * w[:, None]
    coef, *_ = np.linalg.lstsq(A, v[m] * w, rcond=None)
    return coef


def eco_geometry(items: list, l2_items: list, grid: TileGrid, bounds: Bounds,
                 env: dict) -> dict:
    """Coarse (time, ay, ax) SZA, SAA, VZA, VAA for ``items``.

    L2T ``view_zenith`` exists only where LST was produced (cloud-free,
    often 10-20 % of a tile), so VZA over a cloudy AOI is *modelled*: the
    block-median VZA of all L2T scenes of the same orbit are combined over
    the whole tile and a plane is fitted around the AOI (VZA is close to
    linear in cross-track distance). Measured VZA is used where present.
    Orbits with no L2T granule at all get NaN VZA/VAA.
    """
    blocks, ax, ay = block_nodes(grid, bounds)
    j0, j1, i0, i1 = blocks
    step = BLOCK_PX * grid.res
    lon, lat = Transformer.from_crs(grid.crs, 4326, always_xy=True).transform(
        *np.meshgrid(ax, ay))
    nby, nbx = grid.ny // BLOCK_PX, grid.nx // BLOCK_PX
    tx = grid.x0 + (np.arange(nbx) + 0.5) * step
    ty = grid.y0 - (np.arange(nby) + 0.5) * step
    xc, yc = ax.mean(), ay.mean()

    by_orbit: dict[int, list] = {}
    for it in l2_items:
        h = eco_layer_hrefs(it).get("view_zenith")
        if h:
            by_orbit.setdefault(parse_eco_id(it.id)["orbit"], []).append(h)
    orbits = list(by_orbit)

    def tile_vza(orbit):
        with ThreadPoolExecutor(max_workers=4) as pool:
            cols = list(pool.map(
                lambda h: _tile_vza_blocks(h, grid, env), by_orbit[orbit]))
        return nanmean_quiet(np.stack(cols), axis=0)

    with ThreadPoolExecutor(max_workers=4) as pool:
        tiles = dict(zip(orbits, pool.map(tile_vza, orbits)))

    nan_grid = np.full((len(ay), len(ax)), np.nan, dtype="float32")
    per_orbit = {}
    for o, V in tiles.items():
        measured = V[j0:j1, i0:i1]
        coef = _plane_fit(V, tx, ty, xc, yc)
        if coef is None:
            per_orbit[o] = (nan_grid, nan_grid)
            continue
        xx, yy = np.meshgrid(ax - xc, ay - yc)
        plane = np.maximum(coef[0] + coef[1] * xx + coef[2] * yy, 0.0)
        vza = np.where(np.isfinite(measured), measured, plane)
        vaa = eco_view_azimuth(plane.astype("float32"), step, grid.crs, ax, ay)
        vaa[vza < MIN_VZA_FOR_AZIMUTH] = 0.0
        per_orbit[o] = (vza.astype("float32"), vaa)

    sza, saa, vza, vaa = [], [], [], []
    for it in items:
        s, a = solar_position(eco_time(it), lon, lat)
        v, az = per_orbit.get(parse_eco_id(it.id)["orbit"],
                              (nan_grid, nan_grid))
        sza.append(s)
        saa.append(a)
        vza.append(v)
        vaa.append(az)
    f32 = lambda l: np.stack(l).astype("float32")   # noqa: E731
    return dict(sza=f32(sza), saa=f32(saa), vza=f32(vza), vaa=f32(vaa),
                ax=ax, ay=ay)


# ---------------------------------------------------------------------------
# Granule screening
# ---------------------------------------------------------------------------

def has_data(href: str, grid: TileGrid, bounds: Bounds, env: dict) -> bool:
    """Cheap check that a granule has any finite pixel over the window.

    Tiles are produced for every scene whose swath touches them, so many
    granules are NaN over a small AOI.
    """
    row0, col0, nrow, ncol = bounds.pixel_window(grid)
    shape = (min(nrow, 64), min(ncol, 64))
    try:
        with rasterio.Env(**env), rasterio.open(href) as src:
            a = src.read(1, window=Window(col0, row0, ncol, nrow),
                         out_shape=shape, masked=True)
        return bool(np.isfinite(a.astype("float32").filled(np.nan)).any())
    except rasterio.errors.RasterioIOError:
        return False


def screen_granules(items: list, layer: str, grid: TileGrid, bounds: Bounds,
                    env: dict, aoi_centre: tuple[float, float] | None = None,
                    day_only: bool = True, max_sza: float = 85.0) -> list:
    """Keep granules that have the layers, have data, and (optionally) are
    daytime at the AOI centre."""
    keep = []
    for it in items:
        if layer not in eco_layer_hrefs(it):
            continue
        if day_only and aoi_centre is not None:
            sza, _ = solar_position(eco_time(it), *aoi_centre)
            if float(sza) > max_sza:
                continue
        keep.append(it)
    if not keep:
        return []
    with ThreadPoolExecutor(max_workers=8) as pool:
        ok = list(pool.map(
            lambda it: has_data(eco_layer_hrefs(it)[layer], grid, bounds, env),
            keep))
    return [it for it, g in zip(keep, ok) if g]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def _common_coords(items, grid, bounds, geom, product, version):
    x, y = bounds.coords(grid.res)
    ids = [parse_eco_id(it.id) for it in items]
    return dict(
        time=np.array([eco_time(it) for it in items]),
        x=x, y=y, ax=geom["ax"], ay=geom["ay"],
        platform=("time", ["ecostress"] * len(items)),
        item_id=("time", [it.id for it in items]),
        orbit=("time", np.array([i["orbit"] for i in ids], dtype="int32")),
        scene=("time", np.array([i["scene"] for i in ids], dtype="int32")),
    )


def _angle_vars(geom):
    return {k: (("time", "ay", "ax"), geom[k]) for k in
            ("sza", "saa", "vza", "vaa")}


def open_ecostress_l1ct(items, l2_items, grid: TileGrid, bounds: Bounds, aoi,
                        tile: str, bands=(2, 4, 5), nedt_k: float = 0.15,
                        mask_water: bool = True, chunk: int = 1024,
                        version: str = "002") -> xr.Dataset:
    """Lazy TOA-radiance Dataset for already-screened L1CT ``items``."""
    env = gdal_env_for("earthdata")
    geom = eco_geometry(items, l2_items, grid, bounds, env)
    names = [ECO_BANDS[b][0] for b in bands]
    wl_um = np.array([ECO_BANDS[b][1] / 1000.0 for b in bands], dtype="float32")

    obs_t, valid_t = [], []
    for it in items:
        h = eco_layer_hrefs(it)
        rad = da.stack([lazy_layer(h[f"radiance_{b}"], grid, bounds, env,
                                   chunk=chunk) for b in bands])
        cloud = lazy_layer(h["cloud"], grid, bounds, env, raw=True,
                           dtype=np.uint8, chunk=chunk)
        ok = (cloud == 0) & da.isfinite(rad).all(axis=0)
        if mask_water:
            water = lazy_layer(h["water"], grid, bounds, env, raw=True,
                               dtype=np.uint8, chunk=chunk)
            ok = ok & (water == 0)
        for b in bands:
            dq = lazy_layer(h[f"data_quality_{b}"], grid, bounds, env,
                            raw=True, dtype=np.uint16, chunk=chunk)
            ok = ok & (dq == 0)
        obs_t.append(rad)
        valid_t.append(ok)
    obs = da.stack(obs_t)                                    # (t, b, y, x)
    unc = eco_sigma_radiance(obs, wl_um[None, :, None, None],
                             nedt_k).astype("float32")

    coords = _common_coords(items, grid, bounds, geom, "L1CT_RAD", version)
    coords["band"] = names
    coords["wl_centre"] = ("band", (wl_um * 1000.0).astype("float32"))
    ds = xr.Dataset(
        data_vars=dict(
            obs=(("time", "band", "y", "x"), obs),
            obs_unc=(("time", "band", "y", "x"), unc),
            valid=(("time", "y", "x"), da.stack(valid_t)),
            inside_aoi=(("y", "x"),
                        aoi_mask(aoi, bounds, grid.crs, grid.res)),
            **_angle_vars(geom)),
        coords=coords,
        attrs=dict(sensor="ecostress", product="L1CT_RAD", obs_kind="radiance",
                   units="W m-2 sr-1 um-1", source="earthdata", tile=tile,
                   resolution_m=grid.res, version=version,
                   obs_unc_model=f"NEdT={nedt_k} K at observed BT, "
                                 "monochromatic at band centre",
                   vaa_note="from VZA gradient; 0 where VZA < 1 deg"),
    )
    return ds.rio.write_crs(grid.crs)


def open_ecostress_lste(items, grid: TileGrid, bounds: Bounds, aoi, tile: str,
                        qa_max: int = 1, chunk: int = 1024,
                        version: str = "002",
                        l2_items_all: list | None = None) -> xr.Dataset:
    """Lazy LST Dataset (``lst, lst_err, emis_wb, qc, height, valid``).

    ``l2_items_all`` are all L2T granules of the tile (default: ``items``),
    used to model view geometry.

    ``valid`` requires cloud == 0, finite LST and the mandatory-QA field
    (QC bits 0-1: 0 best, 1 nominal, 2 cloud, 3 not produced) <= qa_max.
    """
    env = gdal_env_for("earthdata")
    geom = eco_geometry(items, l2_items_all or items, grid, bounds, env)
    layers = {"lst": "LST", "lst_err": "LST_err", "emis_wb": "EmisWB",
              "height": "height"}
    cols = {k: [] for k in layers}
    qcs, valids = [], []
    for it in items:
        h = eco_layer_hrefs(it)
        for k, layer in layers.items():
            cols[k].append(lazy_layer(h[layer], grid, bounds, env, chunk=chunk))
        qc = lazy_layer(h["QC"], grid, bounds, env, raw=True,
                        dtype=np.uint16, chunk=chunk)
        cloud = lazy_layer(h["cloud"], grid, bounds, env, raw=True,
                           dtype=np.uint8, chunk=chunk)
        qcs.append(qc)
        valids.append((cloud == 0) & ((qc & 3) <= qa_max)
                      & da.isfinite(cols["lst"][-1]))
    data = {k: (("time", "y", "x"), da.stack(v)) for k, v in cols.items()}
    data["qc"] = (("time", "y", "x"), da.stack(qcs))
    data["valid"] = (("time", "y", "x"), da.stack(valids))
    data["inside_aoi"] = (("y", "x"),
                          aoi_mask(aoi, bounds, grid.crs, grid.res))
    data.update(_angle_vars(geom))
    ds = xr.Dataset(
        data, coords=_common_coords(items, grid, bounds, geom, "L2T_LSTE",
                                    version),
        attrs=dict(sensor="ecostress", product="L2T_LSTE", obs_kind="lst",
                   units="K", source="earthdata", tile=tile,
                   resolution_m=grid.res, version=version,
                   vaa_note="from VZA gradient; 0 where VZA < 1 deg"))
    return ds.rio.write_crs(grid.crs)
