"""Validation of the output contract (see ``docs/reframe.md`` section 5)."""

from __future__ import annotations

import dask.array as da
import numpy as np
import xarray as xr

_REQUIRED_ATTRS = ("sensor", "product", "obs_kind", "units", "source", "tile")
_ANGLES = ("sza", "saa", "vza", "vaa")


def validate_obs_dataset(ds: xr.Dataset) -> None:
    """Raise ValueError if ``ds`` breaks the observation-dataset contract.

    Checks dims, dtypes, attributes, monotonic time, dask backing of the
    big arrays, and that angle values are in range. Only the (small,
    eager) angle grids are inspected; no pixel data is read.
    """
    def fail(msg):
        raise ValueError(f"Invalid observation dataset: {msg}")

    for a in _REQUIRED_ATTRS:
        if a not in ds.attrs:
            fail(f"missing attribute {a!r}")
    for c in ("time", "x", "y", "ax", "ay", "item_id", "platform"):
        if c not in ds.coords:
            fail(f"missing coordinate {c!r}")

    t = ds["time"].values
    if t.size > 1 and not np.all(np.diff(t.astype("int64")) >= 0):
        fail("time is not monotonic")
    if "valid" not in ds:
        fail("missing 'valid'")
    if ds["valid"].dims != ("time", "y", "x") or ds["valid"].dtype != bool:
        fail("'valid' must be bool (time, y, x)")

    if "obs" in ds:
        if ds["obs"].dims != ("time", "band", "y", "x"):
            fail(f"'obs' dims are {ds['obs'].dims}")
        if ds["obs"].dtype != np.float32:
            fail("'obs' must be float32")
        if not isinstance(ds["obs"].data, da.Array):
            fail("'obs' is not dask-backed")
        if "obs_unc" not in ds or ds["obs_unc"].shape != ds["obs"].shape:
            fail("'obs_unc' missing or wrong shape")
        for c in ("band", "wl_centre"):
            if c not in ds.coords:
                fail(f"missing coordinate {c!r}")

    for name in _ANGLES:
        if name not in ds:
            fail(f"missing angle grid {name!r}")
        v = ds[name]
        if v.dims != ("time", "ay", "ax"):
            fail(f"{name!r} dims are {v.dims}")
        a = v.values
        if not np.isfinite(a).any() and name not in ("vza", "vaa"):
            fail(f"{name!r} has no finite values")
        with np.errstate(invalid="ignore"):
            if name in ("sza", "vza") and np.nanmax(a, initial=0) > 180:
                fail(f"{name!r} out of range")
            if name in ("saa", "vaa") and (
                    np.nanmin(a, initial=0) < 0 or np.nanmax(a, initial=0) > 360):
                fail(f"{name!r} out of range")
