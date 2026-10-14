"""Coarse angle grids: helpers and on-demand interpolation (D5)."""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy import ndimage

ANGLE_VARS = ("sza", "saa", "vza", "vaa")
_AZIMUTH_VARS = ("saa", "vaa")


def nanmean_quiet(a: np.ndarray, axis=0) -> np.ndarray:
    """``np.nanmean`` that returns NaN for all-NaN slices without warning
    (and without ``warnings`` state, so it is safe in threads)."""
    ok = np.isfinite(a)
    n = ok.sum(axis=axis)
    tot = np.where(ok, a, 0.0).sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, tot / np.maximum(n, 1), np.nan)


def block_median_quiet(a: np.ndarray, size: int) -> np.ndarray:
    """NaN-ignoring median over non-overlapping ``size`` x ``size`` blocks
    of a 2-D array (NaN for all-NaN blocks), without warnings."""
    ny, nx = a.shape[0] // size, a.shape[1] // size
    b = a[:ny * size, :nx * size].reshape(ny, size, nx, size)
    b = b.transpose(0, 2, 1, 3).reshape(ny, nx, size * size)
    s = np.sort(b, axis=-1)                       # NaNs sort last
    n = np.isfinite(b).sum(axis=-1)
    last = s.shape[-1] - 1
    lo = np.take_along_axis(
        s, np.clip((n - 1) // 2, 0, last)[..., None], axis=-1)[..., 0]
    hi = np.take_along_axis(
        s, np.clip(n // 2, 0, last)[..., None], axis=-1)[..., 0]
    return np.where(n > 0, 0.5 * (lo + hi), np.nan).astype(a.dtype)


def circular_mean_deg(a: np.ndarray, axis=0) -> np.ndarray:
    """NaN-aware circular mean of azimuths in degrees, in [0, 360)."""
    r = np.radians(a)
    s = nanmean_quiet(np.sin(r), axis=axis)
    c = nanmean_quiet(np.cos(r), axis=axis)
    return np.mod(np.degrees(np.arctan2(s, c)), 360.0)


def fill_nearest(a: np.ndarray) -> np.ndarray:
    """Fill NaNs in a 2-D array with the nearest finite value."""
    bad = ~np.isfinite(a)
    if not bad.any() or bad.all():
        return a
    idx = ndimage.distance_transform_edt(
        bad, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def node_slice(origin: float, step: float, lo: float, hi: float, n: int,
               ) -> slice:
    """Index slice of grid nodes ``origin + i*step`` (step may be <0)
    covering [lo, hi] plus one node of margin each side."""
    if step > 0:
        i0 = int(np.floor((lo - origin) / step)) - 1
        i1 = int(np.ceil((hi - origin) / step)) + 1
    else:
        i0 = int(np.floor((origin - hi) / -step)) - 1
        i1 = int(np.ceil((origin - lo) / -step)) + 1
    return slice(max(i0, 0), min(i1 + 1, n))


def angles_at(ds: xr.Dataset, x, y) -> xr.Dataset:
    """Interpolate the coarse angle grids to points.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with ``sza, saa, vza, vaa`` on ``(time, ay, ax)``.
    x, y : array_like
        Point coordinates in the dataset's CRS [m]. 1-D (same length) or
        scalars. For a full pixel grid, pass ``ds.x`` and ``ds.y`` with
        ``xr.DataArray`` dims ``('y', 'x')`` meshed by the caller.

    Notes
    -----
    NaN nodes (outside the S2 detector swath) are filled by nearest
    neighbour first. Azimuths are interpolated through (sin, cos) so the
    360/0 wrap is harmless. Points outside the grid take the edge value.
    """
    x = xr.DataArray(np.atleast_1d(np.asarray(x, dtype=float)), dims="point")
    y = xr.DataArray(np.atleast_1d(np.asarray(y, dtype=float)), dims="point")
    ax, ay = ds["ax"].values, ds["ay"].values
    xc = x.clip(ax.min(), ax.max())
    yc = y.clip(ay.min(), ay.max())

    out = {}
    for name in ANGLE_VARS:
        if name not in ds:
            continue
        v = ds[name]
        filled = xr.apply_ufunc(
            lambda a: np.stack([fill_nearest(s) for s in a]),
            v, input_core_dims=[["time", "ay", "ax"]],
            output_core_dims=[["time", "ay", "ax"]], vectorize=False,
        ) if v.ndim == 3 else v
        filled = filled.transpose(*v.dims)
        if name in _AZIMUTH_VARS:
            r = np.radians(filled)
            s = np.sin(r).interp(ax=xc, ay=yc)
            c = np.cos(r).interp(ax=xc, ay=yc)
            out[name] = (np.degrees(np.arctan2(s, c)) % 360.0).astype("float32")
        else:
            out[name] = filled.interp(ax=xc, ay=yc).astype("float32")
    res = xr.Dataset(out)
    res = res.assign_coords(x=("point", x.values), y=("point", y.values))
    return res
