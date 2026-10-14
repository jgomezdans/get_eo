"""Aggregate a fine-grid dataset (S2, 10 m) onto a coarse one (ECO, 70 m)."""

from __future__ import annotations

import dask.array as da
import numpy as np
import xarray as xr
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject


def _is_block_aligned(fine: xr.Dataset, coarse: xr.Dataset) -> int | None:
    """Integer ratio if every coarse cell is an exact ratio x ratio block
    of fine pixels, else None."""
    fx, fy = fine.x.values, fine.y.values
    cx, cy = coarse.x.values, coarse.y.values
    if len(fx) < 2 or len(cx) < 2:
        return None
    rf, rc = abs(fx[1] - fx[0]), abs(cx[1] - cx[0])
    ratio = rc / rf
    n = int(round(ratio))
    if n < 1 or abs(ratio - n) > 1e-6:
        return None
    # coarse cell (i) must start where a fine block starts
    first_edge_x = cx[0] - rc / 2
    first_edge_y = cy[0] + rc / 2
    off_x = (first_edge_x - (fx[0] - rf / 2)) / rf
    off_y = ((fy[0] + rf / 2) - first_edge_y) / rf
    if abs(off_x - round(off_x)) > 1e-6 or abs(off_y - round(off_y)) > 1e-6:
        return None
    return n


def _block_mean(arr: da.Array, ratio: int, off_y: int, off_x: int,
                ny: int, nx: int) -> da.Array:
    """NaN-aware block sum helper: crop to the coarse extent and reshape."""
    a = arr[..., off_y:off_y + ny * ratio, off_x:off_x + nx * ratio]
    lead = a.shape[:-2]
    a = a.reshape(*lead, ny, ratio, nx, ratio)
    return a


def aggregate_to(fine: xr.Dataset, coarse: xr.Dataset) -> xr.Dataset:
    """Aggregate ``fine`` onto the pixel grid of ``coarse``.

    Returns a Dataset on ``fine``'s time axis and ``coarse``'s (x, y) with

    * ``obs``: mean of the *valid* fine pixels in each coarse cell;
    * ``valid_frac``: fraction of fine pixels in each cell that are valid;
    * ``obs_unc``: standard error of the mean, sqrt(sum(sigma^2)) / n,
      which **assumes independent errors** (downstream may inflate it);
    * ``valid``: ``valid_frac > 0``.

    When the grids are block aligned (the S2/ECOSTRESS case, ratio 7)
    this is an exact block mean. Otherwise it falls back to
    ``rasterio.warp.reproject(..., Resampling.average)`` per slice, which
    needs the fine data in memory.
    """
    ratio = _is_block_aligned(fine, coarse)
    ny, nx = coarse.sizes["y"], coarse.sizes["x"]
    valid = fine["valid"].data                             # (t, y, x)
    obs = fine["obs"].data                                 # (t, b, y, x)
    unc = fine["obs_unc"].data

    if ratio is not None:
        rf = abs(float(fine.x[1] - fine.x[0]))
        off_x = int(round(((coarse.x.values[0] - ratio * rf / 2)
                           - (fine.x.values[0] - rf / 2)) / rf))
        off_y = int(round(((fine.y.values[0] + rf / 2)
                           - (coarse.y.values[0] + ratio * rf / 2)) / rf))
        if (off_x < 0 or off_y < 0
                or off_x + nx * ratio > fine.sizes["x"]
                or off_y + ny * ratio > fine.sizes["y"]):
            raise ValueError("Coarse grid is not inside the fine grid")
        v = _block_mean(valid, ratio, off_y, off_x, ny, nx)     # t,ny,r,nx,r
        v = v & True
        n_valid = v.sum(axis=(-3, -1))
        o = _block_mean(obs, ratio, off_y, off_x, ny, nx)       # t,b,ny,r,nx,r
        vb = v[:, None]                                          # broadcast b
        o = da.where(vb, o, 0.0)
        mean = o.sum(axis=(-3, -1)) / da.maximum(n_valid[:, None], 1)
        mean = da.where(n_valid[:, None] > 0, mean, np.nan)
        u = _block_mean(unc, ratio, off_y, off_x, ny, nx)
        u2 = da.where(vb, u ** 2, 0.0).sum(axis=(-3, -1))
        se = da.sqrt(u2) / da.maximum(n_valid[:, None], 1)
        se = da.where(n_valid[:, None] > 0, se, np.nan)
        frac = n_valid / float(ratio * ratio)
        obs_c, unc_c = mean.astype("float32"), se.astype("float32")
        frac_c = frac.astype("float32")
    else:
        obs_c, unc_c, frac_c = _reproject_average(fine, coarse)

    out = xr.Dataset(
        data_vars=dict(
            obs=(("time", "band", "y", "x"), obs_c),
            obs_unc=(("time", "band", "y", "x"), unc_c),
            valid_frac=(("time", "y", "x"), frac_c),
        ),
        coords={k: v for k, v in fine.coords.items()
                if k in ("time", "band", "wl_centre", "platform", "item_id",
                         "ax", "ay")},
        attrs=dict(fine.attrs),
    )
    out = out.assign_coords(x=coarse.x, y=coarse.y)
    out["valid"] = out["valid_frac"] > 0
    for k in ("sza", "saa", "vza", "vaa"):
        if k in fine:
            out[k] = fine[k]
    if "inside_aoi" in coarse:
        out["inside_aoi"] = coarse["inside_aoi"]
    out.attrs["aggregated_to_m"] = float(abs(coarse.x[1] - coarse.x[0])) \
        if coarse.sizes["x"] > 1 else float("nan")
    out.attrs["unc_note"] = ("standard error of the mean, independent "
                             "errors assumed")
    return out.rio.write_crs(fine.rio.crs)


def _reproject_average(fine: xr.Dataset, coarse: xr.Dataset):
    """General path: rasterio average resampling (eager, in memory)."""
    fx, fy = fine.x.values, fine.y.values
    cx, cy = coarse.x.values, coarse.y.values
    rf = abs(fx[1] - fx[0])
    rc = abs(cx[1] - cx[0])
    src_tr = Affine(rf, 0, fx[0] - rf / 2, 0, -rf, fy[0] + rf / 2)
    dst_tr = Affine(rc, 0, cx[0] - rc / 2, 0, -rc, cy[0] + rc / 2)
    crs = fine.rio.crs
    shape = (len(cy), len(cx))

    def avg(a):
        out = np.full(shape, np.nan, dtype="float32")
        reproject(a.astype("float32"), out, src_transform=src_tr,
                  src_crs=crs, dst_transform=dst_tr, dst_crs=crs,
                  resampling=Resampling.average,
                  src_nodata=np.nan, dst_nodata=np.nan)
        return out

    valid = fine["valid"].values
    obs = np.where(valid[:, None], fine["obs"].values, np.nan)
    var = np.where(valid[:, None], fine["obs_unc"].values ** 2, np.nan)
    t, b = obs.shape[:2]
    mean = np.stack([[avg(obs[i, j]) for j in range(b)] for i in range(t)])
    meanvar = np.stack([[avg(var[i, j]) for j in range(b)] for i in range(t)])
    frac = np.stack([avg(valid[i].astype("float32")) for i in range(t)])
    # mean of sigma^2 / n_valid  (n_valid = frac * pixels per cell)
    n = np.maximum(frac * (rc / rf) ** 2, 1.0)[:, None]
    se = np.sqrt(meanvar / n)
    return mean, se.astype("float32"), np.nan_to_num(frac).astype("float32")
