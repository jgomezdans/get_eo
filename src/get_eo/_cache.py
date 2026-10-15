"""Local Zarr cache: pay for the remote reads once."""

from __future__ import annotations

import logging
import os
import time

import numpy as np
import xarray as xr
from numcodecs import Zstd

log = logging.getLogger(__name__)


def _encoding(ds: xr.Dataset) -> dict:
    comp = Zstd(level=3)
    enc = {}
    for name, v in ds.data_vars.items():
        if v.ndim >= 2:
            chunks = tuple(min(512, s) if d in ("x", "y") else 1
                           for d, s in zip(v.dims, v.shape))
            if "band" in v.dims:
                chunks = tuple(s if d == "band" else c
                               for d, s, c in zip(v.dims, v.shape, chunks))
            enc[name] = dict(compressor=comp, chunks=chunks)
    # S2 reflectance: int16 with scale_factor keeps ~1e-4 precision at half
    # the size. Radiance stays float32.
    if ds.attrs.get("obs_kind") == "reflectance" and "obs" in ds:
        enc["obs"].update(dtype="int16", scale_factor=1e-4, _FillValue=-32768)
    return enc


def to_zarr_cache(ds: xr.Dataset, path: str) -> None:
    """Compute ``ds`` and write it to a local Zarr store (replaces any
    existing store at ``path``)."""
    crs = ds.rio.crs
    out = ds.drop_vars([v for v in ("spatial_ref",) if v in ds.coords],
                       errors="ignore")
    out.attrs = dict(out.attrs, crs_wkt=crs.to_wkt() if crs else "")
    out.attrs = {k: (str(v) if not isinstance(v, (int, float, str)) else v)
                 for k, v in out.attrs.items()}
    # one zarr chunk == one dask chunk: (1 time, all bands, <=512 x <=512)
    out = out.chunk({d: (1 if d == "time" else -1 if d == "band" else 512)
                     for d in out.dims if d in ("time", "band", "x", "y")})
    log.info("to_zarr_cache: computing and writing %s to %s (this is where "
             "the remote reads happen)", dict(out.sizes), path)
    t0 = time.perf_counter()
    out.to_zarr(path, mode="w", encoding=_encoding(out), consolidated=True)
    size = sum(os.path.getsize(os.path.join(d, f))
               for d, _, fs in os.walk(path) for f in fs)
    log.info("to_zarr_cache: wrote %.1f MB in %.1fs", size / 1e6,
             time.perf_counter() - t0)


def open_cache(path: str) -> xr.Dataset:
    """Open a cache written by :func:`to_zarr_cache` (lazy, dask-backed)."""
    log.info("open_cache: %s", path)
    ds = xr.open_zarr(path, consolidated=True)
    wkt = ds.attrs.get("crs_wkt")
    return ds.rio.write_crs(wkt) if wkt else ds
