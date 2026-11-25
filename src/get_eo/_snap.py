"""SNAP Sentinel-2 Toolbox biophysical processor, on lazy xarray Datasets.

Python port of the S2 Toolbox L2B biophysical processor (algorithm version
2.1): one small neural network (8 reflectance bands + cos(VZA), cos(SZA),
cos(relative azimuth) -> 1 output) per variable. See the ATBD at
https://step.esa.int/docs/extra/ATBD_S2ToolBox_L2B_V1.1.pdf and the Java
sources and auxiliary data at
https://github.com/senbox-org/s2tbx (``s2tbx-biophysical``). The port
follows the Python implementation by Olli Nevalainen (Finnish
Meteorological Institute).

Differences from the SNAP/Java processor: the convex-hull input check is
not applied (only the per-band min/max definition domain is), and
no per-pixel uncertainty is produced.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from get_eo._geometry import fill_nearest

log = logging.getLogger(__name__)

_DATA = Path(__file__).parent / "_snap_data"

#: Reflectance bands the networks read, in network input order.
SNAP_BANDS = ("B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12")

#: output name -> (data folder, units, long name)
SNAP_VARIABLES = {
    "lai": ("LAI", "m2 m-2", "Leaf area index"),
    "fapar": ("FAPAR", "1", "Fraction of absorbed PAR"),
    "fcover": ("FCOVER", "1", "Fraction of vegetation cover"),
    "lai_cab": ("LAI_Cab", "ug cm-2",
                "Canopy chlorophyll content (LAI x leaf Cab)"),
    "lai_cw": ("LAI_Cw", "cm", "Canopy water content (LAI x leaf Cw)"),
}

#: Published validation RMSE of each variable (ATBD), in the output units.
SNAP_RMSE = {"fapar": 0.05, "fcover": 0.04, "lai": 0.89,
             "lai_cab": 56.0, "lai_cw": 0.03}

VEGETATION_INDICES = {
    "ndvi": ("B08", "B04"),
    "ci_red_edge": ("B07", "B05"),
    "gcc": ("B02", "B03", "B04"),
}


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------

class _Net:
    """Weights and pre/post-processing for one variable."""

    def __init__(self, folder: str):
        d = _DATA / folder

        def load(name):
            return np.loadtxt(d / f"{folder}_{name}", delimiter=",", ndmin=1)

        self.norm = load("Normalisation").reshape(-1, 2)
        self.denorm = load("Denormalisation")
        self.w1 = load("Weights_Layer1_Neurons").reshape(
            -1, self.norm.shape[0])
        self.b1 = load("Weights_Layer1_Bias").reshape(-1, 1)
        self.w2 = load("Weights_Layer2_Neurons").reshape(1, -1)
        self.b2 = load("Weights_Layer2_Bias").reshape(1, 1)
        dom = load("DefinitionDomain_MinMax").reshape(2, -1)
        self.dom_min = dom[0].reshape(-1, 1)
        self.dom_max = dom[1].reshape(-1, 1)
        self.extreme = load("ExtremeCases")      # tolerance, min, max

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Run on ``x`` of shape (11, N); returns (N,) float64 with NaN for
        invalid or out-of-range pixels."""
        x = np.asarray(x, dtype="float64")
        out = np.full(x.shape[1], np.nan)
        nb = self.dom_min.shape[0]
        with np.errstate(invalid="ignore"):
            ok = np.isfinite(x).all(axis=0)
            ok &= ((x[:nb] >= self.dom_min) & (x[:nb] <= self.dom_max)
                   ).all(axis=0)
        if not ok.any():
            return out
        xs = x[:, ok]
        lo, hi = self.norm[:, :1], self.norm[:, 1:]
        xn = 2.0 * (xs - lo) / (hi - lo) - 1.0
        h = np.tanh(self.w1 @ xn + self.b1)
        y = (self.w2 @ h + self.b2)[0]
        y = 0.5 * (y + 1.0) * (self.denorm[1] - self.denorm[0]) + self.denorm[0]
        out[ok] = self._output_range(y)
        return out

    def _output_range(self, y: np.ndarray) -> np.ndarray:
        """SNAP extreme cases: NaN beyond the tolerance, clip within it."""
        tol, lo, hi = self.extreme              # tol is negative
        y = y.copy()
        bad = (y < lo + tol) | (y > hi - tol)
        y = np.clip(y, lo, hi)
        y[bad] = np.nan
        return y


@lru_cache(maxsize=None)
def _net(folder: str) -> _Net:
    return _Net(folder)


# ---------------------------------------------------------------------------
# Geometry, evaluated per dask block so full-resolution angles never exist
# ---------------------------------------------------------------------------

def _coarse_geometry(ds: xr.Dataset) -> np.ndarray:
    """(time, 6, ay, ax): sza, vza, sin/cos(saa), sin/cos(vaa) with NaN
    nodes filled by nearest neighbour. Azimuths go through (sin, cos) so
    interpolation is safe at the 0/360 wrap."""
    def filled(name):
        return np.stack([fill_nearest(a) for a in
                         ds[name].transpose("time", "ay", "ax").values
                         .astype("float64")])
    sza, vza = filled("sza"), filled("vza")
    saa, vaa = np.radians(filled("saa")), np.radians(filled("vaa"))
    return np.stack([sza, vza, np.sin(saa), np.cos(saa),
                     np.sin(vaa), np.cos(vaa)], axis=1)


def _block_for(folder: str):
    """Block function for one variable's network (picklable closure)."""
    net = _net(folder)

    def f(obs, xs, ys, ti, **kw):
        return _block_with(net, obs, xs, ys, ti, **kw)
    return f


def _block_with(net, obs, xs, ys, ti, *, ax, ay, coarse):
    """obs (t, 8, y, x) + pixel coordinates -> network output (t, y, x)."""
    nt, _, ny, nx = obs.shape
    flip_y, flip_x = ay[0] > ay[-1], ax[0] > ax[-1]
    fy = ay[::-1] if flip_y else ay
    fx = ax[::-1] if flip_x else ax
    yy, xx = np.meshgrid(np.clip(ys, fy[0], fy[-1]),
                         np.clip(xs, fx[0], fx[-1]), indexing="ij")
    pts = np.stack([yy.ravel(), xx.ravel()], axis=-1)

    out = np.full((nt, ny, nx), np.nan, dtype="float32")
    for k in range(nt):
        g = coarse[ti[k]].transpose(1, 2, 0)            # (ay, ax, 6)
        if flip_y:
            g = g[::-1]
        if flip_x:
            g = g[:, ::-1]
        a = RegularGridInterpolator((fy, fx), g)(pts).T  # (6, N)
        cos_vz = np.cos(np.radians(a[1]))
        cos_sz = np.cos(np.radians(a[0]))
        cos_raz = a[3] * a[5] + a[2] * a[4]              # cos(saa - vaa)
        cos_raz /= np.maximum(np.hypot(a[2], a[3]) * np.hypot(a[4], a[5]),
                              1e-12)
        x = np.vstack([obs[k].reshape(len(SNAP_BANDS), -1),
                       cos_vz, cos_sz, cos_raz])
        out[k] = net(x).reshape(ny, nx)
    return out


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def _check_s2(ds: xr.Dataset, need) -> None:
    if ds.attrs.get("sensor") != "sentinel-2" or "obs" not in ds:
        raise ValueError("Expected a Sentinel-2 dataset from open_aoi()['s2']")
    missing = [b for b in need if b not in ds["band"].values]
    if missing:
        raise ValueError(
            f"Bands {missing} are not in the dataset (have "
            f"{list(ds['band'].values)}); open it with those bands.")


def _template(ds: xr.Dataset) -> xr.DataArray:
    """A (time, y, x) DataArray carrying the dataset's coordinates."""
    return ds["valid"]


def snap_biophys(ds: xr.Dataset, variables=tuple(SNAP_VARIABLES), *,
                 mask_invalid: bool = True) -> xr.Dataset:
    """SNAP biophysical variables from Sentinel-2 surface reflectance.

    Parameters
    ----------
    ds : xr.Dataset
        The ``open_aoi(...)["s2"]`` dataset (or a cached copy). Needs bands
        B03, B04, B05, B06, B07, B8A, B11, B12 and the ``sza/saa/vza/vaa``
        grids.
    variables : iterable of str
        Any of ``lai, fapar, fcover, lai_cab, lai_cw`` (case-insensitive).
    mask_invalid : bool
        Set pixels where ``ds.valid`` is False (cloud, shadow, snow, ...) to
        NaN.

    Returns
    -------
    xr.Dataset
        Lazy float32 variables on ``(time, y, x)``, on ``ds``'s grid and
        coordinates. NaN where any input is NaN or outside the networks'
        training domain, or the output is outside its valid range.
    """
    names = [v.lower() for v in
             ([variables] if isinstance(variables, str) else variables)]
    unknown = [v for v in names if v not in SNAP_VARIABLES]
    if unknown:
        raise ValueError(f"Unknown variable(s) {unknown}; choose from "
                         f"{list(SNAP_VARIABLES)}")
    _check_s2(ds, SNAP_BANDS)
    for a in ("sza", "saa", "vza", "vaa"):
        if a not in ds:
            raise ValueError(f"Dataset has no {a!r} angle grid")

    log.info("SNAP biophysical processor: %s on %d scenes", names,
             ds.sizes["time"])
    obs = ds["obs"].sel(band=list(SNAP_BANDS)).transpose(
        "time", "band", "y", "x").data
    obs = da.asarray(obs).rechunk({1: -1}).astype("float32")
    coarse = _coarse_geometry(ds)
    ax, ay = ds["ax"].values.astype("float64"), ds["ay"].values.astype("float64")
    xs = da.from_array(ds["x"].values.astype("float64"),
                       chunks=obs.chunks[3])
    ys = da.from_array(ds["y"].values.astype("float64"),
                       chunks=obs.chunks[2])
    ti = da.from_array(np.arange(ds.sizes["time"]), chunks=obs.chunks[0])
    tmpl = _template(ds)

    out = {}
    for name in names:
        folder, units, long_name = SNAP_VARIABLES[name]
        data = da.blockwise(
            _block_for(folder), "tyx", obs, "tbyx", xs, "x", ys, "y",
            ti, "t", concatenate=True, dtype="float32",
            ax=ax, ay=ay, coarse=coarse, meta=np.empty((0, 0, 0), "float32"))
        v = xr.DataArray(data, dims=("time", "y", "x"), coords=tmpl.coords)
        if mask_invalid:
            v = v.where(ds["valid"])
        v.attrs = dict(units=units, long_name=long_name,
                       algorithm="SNAP S2 Toolbox biophysical processor 2.1",
                       rmse=SNAP_RMSE[name])
        out[name] = v.astype("float32")
    res = xr.Dataset(out)
    res.attrs = {k: ds.attrs[k] for k in ("sensor", "source", "tile")
                 if k in ds.attrs}
    res.attrs["product"] = "SNAP biophysical variables"
    return res


def vegetation_indices(ds: xr.Dataset, indices=tuple(VEGETATION_INDICES), *,
                       mask_invalid: bool = True) -> xr.Dataset:
    """Lazy vegetation indices from Sentinel-2 surface reflectance.

    ``ndvi`` = (B08 - B04) / (B08 + B04); ``ci_red_edge`` = B07 / B05 - 1
    (chlorophyll index, red edge); ``gcc`` = B03 / (B02 + B03 + B04)
    (green chromatic coordinate). Pixels with a non-positive denominator
    are NaN; ``mask_invalid`` also masks on ``ds.valid``.
    """
    names = [v.lower() for v in
             ([indices] if isinstance(indices, str) else indices)]
    unknown = [v for v in names if v not in VEGETATION_INDICES]
    if unknown:
        raise ValueError(f"Unknown index {unknown}; choose from "
                         f"{list(VEGETATION_INDICES)}")
    need = sorted({b for n in names for b in VEGETATION_INDICES[n]})
    _check_s2(ds, need)

    def band(b):
        return ds["obs"].sel(band=b, drop=True)

    out = {}
    for n in names:
        if n == "ndvi":
            nir, red = band("B08"), band("B04")
            den = nir + red
            v = (nir - red) / den
        elif n == "ci_red_edge":
            b7, b5 = band("B07"), band("B05")
            den = b5
            v = b7 / b5 - 1
        else:
            b2, b3, b4 = band("B02"), band("B03"), band("B04")
            den = b2 + b3 + b4
            v = b3 / den
        v = v.where(den > 0)
        if mask_invalid:
            v = v.where(ds["valid"])
        v = v.astype("float32")
        v.attrs = dict(units="1", long_name=n)
        out[n] = v
    res = xr.Dataset(out)
    res.attrs = {k: ds.attrs[k] for k in ("sensor", "source", "tile")
                 if k in ds.attrs}
    return res
