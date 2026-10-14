"""Lazy windowed COG reads (rasterio + dask.delayed)."""

from __future__ import annotations

import dask
import dask.array as da
import numpy as np
import rasterio
from rasterio.windows import Window

from get_eo._env import sign_planetary
from get_eo._grid import Bounds, TileGrid, chunk_windows

_SIGNERS = {"planetary": sign_planetary}


def read_window(
    href: str,
    window: tuple[int, int, int, int],
    gdal_env: dict,
    expected_transform: tuple,
    upsample: int = 1,
    scale: float | None = None,
    offset: float = 0.0,
    sign: str | None = None,
    raw: bool = False,
) -> np.ndarray:
    """Read one band window from a COG.

    Parameters
    ----------
    href : str
        HTTPS URL of the COG (unsigned; ``sign`` signs it at read time).
    window : (row_off, col_off, n_rows, n_cols)
        In the file's native grid.
    gdal_env : dict
        GDAL config options, applied inside the call so it is thread-safe.
    expected_transform : tuple
        Affine coefficients (a, b, c, d, e, f) the file must have. Guards
        against items that silently sit on a different grid.
    upsample : int
        Integer pixel-replication factor (2 for S2 20 m -> 10 m).
    scale, offset : float
        ``value = DN * scale + offset``. ``None`` uses the file's own
        scale/offset (1/0 for most products).
    sign : str or None
        Name of a URL signer ('planetary').
    raw : bool
        Return the native dtype unscaled, nodata left in place (for QA
        layers).

    Returns
    -------
    np.ndarray
        (n_rows * upsample, n_cols * upsample). float32 with nodata as NaN
        unless ``raw``.
    """
    row0, col0, nrow, ncol = window
    if sign:
        href = _SIGNERS[sign](href)
    if href.startswith("s3://"):
        # Use the GDAL path directly: rasterio would otherwise build a
        # boto3 session and ignore the endpoint given in ``gdal_env``.
        href = "/vsis3/" + href[len("s3://"):]
    with rasterio.Env(**gdal_env), rasterio.open(href) as src:
        got = tuple(src.transform)[:6]
        if not np.allclose(got, expected_transform, atol=1e-6):
            raise ValueError(
                f"Unexpected grid in {href}: {got} != {expected_transform}")
        win = Window(col0, row0, ncol, nrow)
        if raw:
            out = src.read(1, window=win)
        else:
            data = src.read(1, window=win, masked=True)
            s = src.scales[0] if scale is None else scale
            o = src.offsets[0] if scale is None else offset
            out = data.astype(np.float32).filled(np.nan) * np.float32(s) \
                + np.float32(o)
            out = out.astype(np.float32, copy=False)
    if upsample > 1:
        # Pixel replication: no interpolation of radiometry (D3)
        out = np.repeat(np.repeat(out, upsample, 0), upsample, 1)
    return out


def _do_read(*args, **kwargs):
    # Late-bound so tests can wrap/patch ``read_window``
    return read_window(*args, **kwargs)


def lazy_layer(
    href: str,
    grid: TileGrid,
    bounds: Bounds,
    gdal_env: dict,
    *,
    upsample: int = 1,
    scale: float | None = None,
    offset: float = 0.0,
    sign: str | None = None,
    raw: bool = False,
    dtype=np.float32,
    chunk: int = 1024,
) -> da.Array:
    """A dask array of one layer over ``bounds`` (nothing is read yet).

    Large windows are split into ``chunk``-pixel blocks; the default
    1024 is a multiple of the 256/512/1024 COG block sizes in use.
    """
    row0, col0, nrow, ncol = bounds.pixel_window(grid)
    expected = grid.transform_coeffs
    rows = []
    for row in chunk_windows(nrow, ncol, chunk):
        blocks = []
        for (r0, r1), (c0, c1) in row:
            d = dask.delayed(_do_read)(
                href, (row0 + r0, col0 + c0, r1 - r0, c1 - c0), gdal_env,
                expected, upsample, scale, offset, sign, raw,
                dask_key_name=None,
            )
            blocks.append(da.from_delayed(
                d, shape=((r1 - r0) * upsample, (c1 - c0) * upsample),
                dtype=dtype))
        rows.append(blocks)
    return da.block(rows)
