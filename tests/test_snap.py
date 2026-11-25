"""SNAP biophysical processor and vegetation indices (offline)."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr

import get_eo as geo
from get_eo import _snap
from get_eo._snap import SNAP_BANDS, SNAP_VARIABLES, _net

ALL_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11",
             "B12"]


def _test_cases(name):
    folder = SNAP_VARIABLES[name][0]
    f = _snap._DATA / folder / f"{folder}_TestCases"
    return folder, (np.loadtxt(f, delimiter=",", ndmin=2) if f.exists() else None)


@pytest.mark.parametrize("name", ["lai", "fapar", "lai_cab"])
def test_network_matches_snap_test_cases(name):
    """Regression against the SNAP-distributed input/output pairs (FCOVER and
    Cw ship none). Expected NaN rows are SNAP's own flagged cases."""
    folder, tc = _test_cases(name)
    assert tc is not None
    got = _net(folder)(tc[:, :11].T)
    exp = tc[:, 11]
    ok = np.isfinite(exp)
    assert ok.sum() >= 20
    np.testing.assert_allclose(got[ok], exp[ok], rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("name", ["fcover", "lai_cw"])
def test_other_networks_output_in_range(name):
    folder = SNAP_VARIABLES[name][0]
    rng = np.random.default_rng(2)
    n = _net(folder)
    x = rng.uniform(n.dom_min, n.dom_max, (8, 500))
    ang = np.vstack([rng.uniform(0.9, 1, 500), rng.uniform(0.5, 1, 500),
                     rng.uniform(-1, 1, 500)])
    y = n(np.vstack([x, ang]))
    assert np.isfinite(y).any()
    assert np.nanmin(y) >= n.extreme[1] and np.nanmax(y) <= n.extreme[2]


def _s2(nt=2, ny=6, nx=7, refl=None, chunks=(1, 10, 3, 4)):
    rng = np.random.default_rng(0)
    if refl is None:
        refl = rng.uniform(0.03, 0.3, (nt, len(ALL_BANDS), ny, nx))
    obs = da.from_array(refl.astype("float32"), chunks=chunks)
    return xr.Dataset(
        dict(obs=(("time", "band", "y", "x"), obs),
             valid=(("time", "y", "x"), da.ones((nt, ny, nx), dtype=bool)),
             **{k: (("time", "ay", "ax"), np.full((nt, 2, 2), v, "float32"))
                for k, v in dict(sza=30, saa=120, vza=5, vaa=90).items()}),
        coords=dict(time=np.array(["2022-07-01", "2022-07-02"][:nt],
                                  "datetime64[ns]"),
                    x=np.arange(nx) * 10.0, y=-np.arange(ny) * 10.0,
                    ax=[0.0, 5000.0], ay=[0.0, -5000.0], band=ALL_BANDS),
        attrs=dict(sensor="sentinel-2", source="aws", tile="14TQL"))


def test_snap_biophys_lazy_shape_and_mask():
    ds = _s2()
    ds["valid"] = ds["valid"].copy(
        data=da.from_array(np.arange(2 * 6 * 7).reshape(2, 6, 7) % 2 == 0,
                           chunks=(1, 3, 4)))
    out = geo.snap_biophys(ds, ["LAI", "fapar"])
    assert isinstance(out["lai"].data, da.Array)
    assert out["lai"].dims == ("time", "y", "x") and out["lai"].dtype == "float32"
    assert out["lai"].attrs["units"] == "m2 m-2"
    r = out.compute()
    assert np.isnan(r["lai"].values[~ds["valid"].values]).all()
    assert np.isfinite(r["lai"].values[ds["valid"].values]).any()


def test_snap_biophys_matches_direct_network():
    ds = _s2()
    r = geo.snap_biophys(ds, ["lai"]).compute()
    obs = ds["obs"].sel(band=list(SNAP_BANDS)).values
    cz = lambda d: np.cos(np.radians(d))
    t, y, x = 1, 2, 3
    inp = np.concatenate([obs[t, :, y, x], [cz(5), cz(30), cz(120 - 90)]])
    ref = _net("LAI")(inp[:, None])[0]
    assert float(r["lai"][t, y, x]) == pytest.approx(ref, abs=1e-4, nan_ok=True)


def test_out_of_domain_and_nan_give_nan():
    refl = np.full((1, 10, 2, 2), 0.1)
    refl[0, 3, 0, 0] = 5.0                  # B05 far outside the domain
    refl[0, 4, 0, 1] = np.nan
    r = geo.snap_biophys(_s2(1, 2, 2, refl, (1, 10, 2, 2)), "lai").compute()
    assert np.isnan(r["lai"].values[0, 0]).all()


def test_azimuth_wrap_and_descending_grid():
    # saa 359 vs vaa 1 => relative azimuth ~ 2 deg, cos ~ 1 (not cos(358))
    ds = _s2(1)
    ds["saa"] = ds["saa"] * 0 + 359.0
    ds["vaa"] = ds["vaa"] * 0 + 1.0
    out = geo.snap_biophys(ds, "lai").compute()
    ds2 = _s2(1)
    ds2["saa"] = ds2["saa"] * 0 + 1.0
    ds2["vaa"] = ds2["vaa"] * 0 + 359.0
    np.testing.assert_allclose(
        out["lai"].values, geo.snap_biophys(ds2, "lai").compute()["lai"].values,
        atol=1e-5)


def test_errors():
    ds = _s2()
    with pytest.raises(ValueError, match="Unknown"):
        geo.snap_biophys(ds, "nope")
    with pytest.raises(ValueError, match="B8A"):
        geo.snap_biophys(ds.sel(band=[b for b in ALL_BANDS if b != "B8A"]))
    with pytest.raises(ValueError, match="Sentinel-2"):
        geo.snap_biophys(ds.assign_attrs(sensor="ecostress"))


def test_vegetation_indices():
    refl = np.zeros((1, 10, 1, 1))
    for b, v in dict(B02=0.04, B03=0.08, B04=0.05, B05=0.1, B07=0.4,
                     B08=0.45).items():
        refl[0, ALL_BANDS.index(b)] = v
    vi = geo.vegetation_indices(_s2(1, 1, 1, refl, (1, 10, 1, 1)))
    assert isinstance(vi["ndvi"].data, da.Array)
    r = vi.compute()
    assert r["ndvi"].values.item() == pytest.approx((0.45 - 0.05) / 0.5)
    assert r["ci_red_edge"].values.item() == pytest.approx(3.0)
    assert r["gcc"].values.item() == pytest.approx(0.08 / 0.17)
    with pytest.raises(ValueError, match="Unknown"):
        geo.vegetation_indices(_s2(), "evi")


def test_cache_roundtrip_of_snap_output(tmp_path):
    ds = _s2()
    ds = ds.rio.write_crs("EPSG:32632")
    ds.attrs["obs_kind"] = "reflectance"
    bio = geo.snap_biophys(ds, ["lai", "fapar"])
    p = str(tmp_path / "bio.zarr")
    geo.to_zarr_cache(bio, p)
    r = geo.open_cache(p)
    for v in ("lai", "fapar"):
        np.testing.assert_array_equal(r[v].values, bio[v].compute().values)
    assert r[v].dtype == np.float32
    # whole S2 dataset incl. int16-packed obs, then derived products from it
    p2 = str(tmp_path / "s2.zarr")
    geo.to_zarr_cache(ds, p2)
    r2 = geo.open_cache(p2)
    np.testing.assert_allclose(r2.obs.values, ds.obs.values, atol=1e-4)
    np.testing.assert_allclose(
        geo.snap_biophys(r2, "lai").lai.values,
        geo.snap_biophys(ds, "lai").lai.values, atol=0.05, equal_nan=True)
