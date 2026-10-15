"""Offline tests for get_eo (no network)."""
import numpy as np
import pytest
import rasterio
import xarray as xr
from pyproj import CRS, Proj, Transformer
from rasterio.transform import from_origin

from get_eo import _read
from get_eo._aoi import resolve_aoi
from get_eo._assets import eco_layer_hrefs, parse_eco_id
from get_eo._eco import block_nodes, eco_view_azimuth
from get_eo._geometry import angles_at
from get_eo._grid import Bounds, TileGrid, aoi_bounds
from get_eo._s2 import reflectance_offset, s2_angle_grids
from get_eo._schema import validate_obs_dataset
from get_eo._solar import solar_position
from get_eo._uncertainty import (
    eco_sigma_radiance, planck_radiance, radiance_to_bt, s2_sigma,
)

UTM14 = CRS.from_epsg(32614)
TILE_14TQL = TileGrid(UTM14, 699960.0, 4600020.0, 10.0, 10980, 10980)
ECO_14TQL = TileGrid(UTM14, 699960.0, 4600020.0, 70.0, 1568, 1568)


# --------------------------------------------------------------------- ids
@pytest.mark.parametrize("item_id,orbit,scene,tile,product", [
    ("ECOv002_L1CT_RAD_22611_008_14TQL_20220702T081925_0712_02",
     22611, 8, "14TQL", "L1CT_RAD"),
    ("ECOv002_L2T_LSTE_22706_007_15TTF_20220708T113616_0712_02",
     22706, 7, "15TTF", "L2T_LSTE"),
])
def test_parse_eco_id(item_id, orbit, scene, tile, product):
    f = parse_eco_id(item_id)
    assert (f["orbit"], f["scene"], f["tile"], f["product"]) == (
        orbit, scene, tile, product)


def test_parse_eco_id_rejects_garbage():
    with pytest.raises(ValueError):
        parse_eco_id("S2B_14TQL_20220708_0_L2A")


def test_eco_layer_hrefs(eco_item):
    h = eco_layer_hrefs(eco_item)
    assert set(h) == {"cloud", "water",
                      *(f"radiance_{i}" for i in range(1, 6)),
                      *(f"data_quality_{i}" for i in range(1, 6))}
    assert all(v.startswith("https://") and v.endswith(".tif")
               for v in h.values())


# ------------------------------------------------------------------- angles
def test_s2_angle_grids(mtd_xml):
    g = s2_angle_grids(mtd_xml)
    assert g["sun_zen"].shape == (23, 23)
    assert g["view_zen"].shape == (23, 23)
    assert (g["ulx"], g["uly"], g["step"]) == (699960.0, 4600020.0, 5000.0)
    # Mean_Sun_Angle in this file is 23.28 deg
    assert abs(np.nanmean(g["sun_zen"]) - 23.2828) < 0.1
    assert np.nanmin(g["view_zen"]) >= 0 and np.nanmax(g["view_zen"]) < 15
    assert ((g["view_az"] >= 0) & (g["view_az"] < 360) | np.isnan(g["view_az"])).all()


def test_solar_position_matches_s2_sun_grid(mtd_xml):
    """Independent check: the S2 processor's sun grid at the tile centre."""
    g = s2_angle_grids(mtd_xml)
    x = g["ulx"] + 11 * 5000.0
    y = g["uly"] - 11 * 5000.0
    lon, lat = Transformer.from_crs(UTM14, 4326, always_xy=True).transform(x, y)
    # SENSING_TIME of the tile is 17:22:31 (tile centre ~ +- seconds)
    sza, saa = solar_position(np.datetime64("2022-07-08T17:22:31"), lon, lat)
    assert abs(sza - g["sun_zen"][11, 11]) < 0.1
    assert abs(((saa - g["sun_az"][11, 11] + 180) % 360) - 180) < 0.5


def test_solar_position_vs_pvlib():
    pvlib = pytest.importorskip("pvlib")
    import pandas as pd
    cases = [("2022-03-20 12:00", 0.0, 0.0), ("2022-07-08 17:22", -96.5, 41.2),
             ("2022-12-21 09:30", 10.0, 55.0), ("2023-01-15 22:10", 140.0, -35.0)]
    for ts, lon, lat in cases:
        t = pd.Timestamp(ts, tz="UTC")
        ref = pvlib.solarposition.get_solarposition(t, lat, lon, method="nrel_numpy")
        sza, saa = solar_position(np.datetime64(t.tz_localize(None)), lon, lat)
        assert abs(sza - float(ref["zenith"].iloc[0])) < 0.05
        assert abs(((saa - float(ref["azimuth"].iloc[0]) + 180) % 360) - 180) < 0.1


def test_meridian_convergence_sign():
    """true_az = grid_az + convergence (pins the convention used in _eco)."""
    for lon in (-96.47, -101.0, -94.0):
        t = Transformer.from_crs(4326, UTM14, always_xy=True)
        x0, y0 = t.transform(lon, 41.2)
        x1, y1 = t.transform(lon, 41.3)           # along a true meridian
        grid_az = np.degrees(np.arctan2(x1 - x0, y1 - y0))
        conv = Proj(UTM14).get_factors(lon, 41.25).meridian_convergence
        assert abs((grid_az + conv + 180) % 360 - 180) < 0.02


@pytest.mark.parametrize("heading", [190.0, 350.0, 100.0])
def test_eco_view_azimuth_recovers_direction(heading):
    """VZA grows with distance from a ground track; the azimuth to the
    sensor is the direction towards the track (true north referenced)."""
    nx = ny = 30
    step = 1120.0
    x = 400000.0 + step * np.arange(nx)
    y = 4560000.0 - step * np.arange(ny)
    lon_c, lat_c = Transformer.from_crs(UTM14, 4326, always_xy=True).transform(
        x[nx // 2], y[ny // 2])
    conv = Proj(UTM14).get_factors(lon_c, lat_c).meridian_convergence
    # track heading expressed in grid azimuth (true heading - convergence)
    h = np.radians(heading - conv)
    xx, yy = np.meshgrid(x - x[nx // 2], y - y[ny // 2])
    # signed distance to the track line (through the grid centre)
    cross = xx * np.cos(h) - yy * np.sin(h)    # + to the right of the track
    off_nadir = 12.0
    vza = 0.0004 * np.abs(cross) + 0.0 * xx + off_nadir * 0 + 0.0
    vza = np.maximum(vza + 3.0 * (np.abs(cross) > 0), 0)  # keep >1 deg
    az = eco_view_azimuth(vza.astype("float32"), step, UTM14, x, y)
    # expected: towards the track = -sign(cross) * (right-hand normal)
    right = (heading + 90.0) % 360.0
    expected = np.where(cross > 0, (right + 180.0) % 360.0, right)
    far = np.abs(cross) > 3 * step
    d = (az - expected + 180) % 360 - 180
    assert np.abs(d[far]).max() < 1.0


# ------------------------------------------------------------------ windows
def test_aoi_bounds_snaps_outwards_and_aligns():
    t = Transformer.from_crs(UTM14, 4326, always_xy=True)
    # a ~3 km box well inside the tile
    xs, ys = (720000.0, 723137.0), (4580000.0, 4583011.0)
    lons, lats = t.transform(list(xs), list(ys))
    aoi = resolve_aoi((min(lons), min(lats), max(lons), max(lats)))
    for snap in (20.0, 140.0):
        b = aoi_bounds(aoi, TILE_14TQL, snap)
        assert b.xmin <= xs[0] and b.xmax >= xs[1]
        assert b.ymin <= ys[0] and b.ymax >= ys[1]
        assert ((b.xmin - TILE_14TQL.x0) / snap) % 1 == 0
        r10 = b.pixel_window(TILE_14TQL)
        # 10 m window vs ECO 70 m window cover the same extent (140 m snap)
        if snap == 140.0:
            re = b.pixel_window(ECO_14TQL)
            assert r10[2] == 7 * re[2] and r10[3] == 7 * re[3]
            assert r10[0] == 7 * re[0] and r10[1] == 7 * re[1]


def test_aoi_bounds_20m_vs_10m_ratio():
    aoi = resolve_aoi((-96.48, 41.15, -96.46, 41.18))
    b = aoi_bounds(aoi, TILE_14TQL, 20.0)
    g20 = TileGrid(UTM14, 699960.0, 4600020.0, 20.0, 5490, 5490)
    w10, w20 = b.pixel_window(TILE_14TQL), b.pixel_window(g20)
    assert tuple(2 * v for v in w20) == w10


def test_aoi_outside_tile_raises():
    with pytest.raises(ValueError):
        aoi_bounds(resolve_aoi((10.0, 10.0, 10.1, 10.1)), TILE_14TQL, 20.0)


def test_block_nodes_cover_window():
    b = Bounds(720000.0, 4580000.0, 723080.0, 4583080.0)
    (j0, j1, i0, i1), ax, ay = block_nodes(ECO_14TQL, b)
    assert ax.min() < b.xmin and ax.max() > b.xmax
    assert ay.min() < b.ymin and ay.max() > b.ymax


def test_resolve_aoi_forms(tmp_path):
    from shapely.geometry import box
    assert resolve_aoi((-1, 50, 0, 51)).bounds == (-1, 50, 0, 51)
    p = resolve_aoi((-96.47, 41.16), buffer_m=1000)
    assert 0.015 < p.bounds[2] - p.bounds[0] < 0.03
    with pytest.raises(ValueError):
        resolve_aoi((-96.47, 41.16))
    gj = {"type": "Polygon",
          "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}
    assert resolve_aoi(gj).bounds == (0, 0, 1, 1)
    assert resolve_aoi(box(0, 0, 2, 2)).bounds == (0, 0, 2, 2)
    assert resolve_aoi("POLYGON((0 0,3 0,3 3,0 3,0 0))").bounds == (0, 0, 3, 3)


# --------------------------------------------------------------------- read
@pytest.fixture
def tiny_cog(tmp_path):
    arr = np.arange(40 * 40, dtype="uint16").reshape(40, 40) + 1
    arr[0, 0] = 0                                  # nodata
    path = tmp_path / "t.tif"
    tr = from_origin(699960.0, 4600020.0, 20.0, 20.0)
    with rasterio.open(path, "w", driver="GTiff", height=40, width=40, count=1,
                       dtype="uint16", crs=UTM14, transform=tr,
                       nodata=0) as dst:
        dst.write(arr, 1)
    return str(path), arr, tr


def test_read_window_upsample_is_replication(tiny_cog):
    path, arr, tr = tiny_cog
    exp = tuple(tr)[:6]
    blk = _read.read_window(path, (4, 6, 8, 10), {}, exp, upsample=1,
                            scale=1.0, offset=0.0)
    up = _read.read_window(path, (4, 6, 8, 10), {}, exp, upsample=2,
                           scale=1.0, offset=0.0)
    np.testing.assert_array_equal(up, np.kron(blk, np.ones((2, 2))))
    np.testing.assert_array_equal(blk, arr[4:12, 6:16].astype("float32"))


def test_read_window_scale_offset_and_nodata(tiny_cog):
    path, arr, tr = tiny_cog
    out = _read.read_window(path, (0, 0, 2, 2), {}, tuple(tr)[:6],
                            scale=1e-4, offset=-0.1)
    assert np.isnan(out[0, 0])
    np.testing.assert_allclose(out[0, 1], 2 * 1e-4 - 0.1, rtol=1e-6)


def test_read_window_refuses_wrong_grid(tiny_cog):
    path, _, tr = tiny_cog
    bad = (20.0, 0.0, 700000.0, 0.0, -20.0, 4600020.0)
    with pytest.raises(ValueError, match="Unexpected grid"):
        _read.read_window(path, (0, 0, 2, 2), {}, bad)


def test_lazy_layer_is_lazy_and_correct(tiny_cog, monkeypatch):
    import dask
    import dask.array as da
    path, arr, tr = tiny_cog
    grid = TileGrid(UTM14, 699960.0, 4600020.0, 20.0, 40, 40)
    calls = []
    real = _read.read_window

    def counting(*a, **k):
        calls.append(a[1])
        return real(*a, **k)

    monkeypatch.setattr(_read, "read_window", counting)
    b = Bounds(699960.0 + 20 * 4, 4600020.0 - 20 * 20,
               699960.0 + 20 * 36, 4600020.0 - 20 * 4)
    lay = _read.lazy_layer(path, grid, b, {}, scale=1.0, offset=0.0, chunk=16)
    assert isinstance(lay, da.Array) and not calls      # nothing read yet
    with dask.config.set(scheduler="synchronous"):
        out = lay.compute()
    assert len(calls) > 1                               # chunked
    np.testing.assert_array_equal(out, arr[4:20, 4:36].astype("float32"))


# ------------------------------------------------------------- conversions
def test_reflectance_offset_branches():
    # AWS: offset is in the DNs whatever the (unreliable) flag says
    for flag in (True, False, None):
        props = {"s2:processing_baseline": "04.00"}
        if flag is not None:
            props["earthsearch:boa_offset_applied"] = flag
        assert reflectance_offset(props, "aws") == 0.0
    # Planetary: original ESA values
    assert reflectance_offset({"s2:processing_baseline": "04.00"}, "planetary") == -0.1
    assert reflectance_offset({"s2:processing_baseline": "05.10"}, "planetary") == -0.1
    assert reflectance_offset({"s2:processing_baseline": "03.01"}, "planetary") == 0.0


def test_s2_sigma():
    assert s2_sigma(0.0) == pytest.approx(0.005)
    assert s2_sigma(0.4) == pytest.approx(np.hypot(0.005, 0.02))
    assert s2_sigma(-0.01) == pytest.approx(0.005)


def test_planck_roundtrip_and_sigma():
    for wl in (8.78, 10.49, 12.09):
        for t in (270.0, 300.0, 330.0):
            assert radiance_to_bt(planck_radiance(t, wl), wl) == pytest.approx(t)
    wl = 10.49
    L = planck_radiance(300.0, wl)
    dl = planck_radiance(300.15, wl) - planck_radiance(299.85, wl)
    assert eco_sigma_radiance(L, wl, 0.15) == pytest.approx(dl / 2, rel=1e-3)


# ---------------------------------------------------------- aggregation etc
def _fine_coarse(ratio=7, nc=4, nt=2, nb=2, seed=0):
    from get_eo._aggregate import aggregate_to   # noqa: F401
    rng = np.random.default_rng(seed)
    nf = nc * ratio
    x0, y0 = 700000.0, 4600000.0
    fx = x0 + 10 * (np.arange(nf) + 0.5)
    fy = y0 - 10 * (np.arange(nf) + 0.5)
    cx = x0 + 70 * (np.arange(nc) + 0.5)
    cy = y0 - 70 * (np.arange(nc) + 0.5)
    obs = rng.random((nt, nb, nf, nf)).astype("float32")
    unc = (0.01 + 0.01 * rng.random((nt, nb, nf, nf))).astype("float32")
    valid = rng.random((nt, nf, nf)) > 0.3
    import dask.array as da
    fine = xr.Dataset(
        dict(obs=(("time", "band", "y", "x"), da.from_array(obs, chunks=-1)),
             obs_unc=(("time", "band", "y", "x"), da.from_array(unc, chunks=-1)),
             valid=(("time", "y", "x"), da.from_array(valid, chunks=-1))),
        coords=dict(time=np.arange(nt).astype("datetime64[D]"),
                    band=["a", "b"][:nb], x=fx, y=fy)).rio.write_crs(UTM14)
    coarse = xr.Dataset(coords=dict(x=cx, y=cy)).rio.write_crs(UTM14)
    return fine, coarse, obs, unc, valid


def test_aggregate_to_is_block_mean():
    from get_eo._aggregate import aggregate_to
    fine, coarse, obs, unc, valid = _fine_coarse()
    out = aggregate_to(fine, coarse).compute()
    for t in range(2):
        for iy in range(4):
            for ix in range(4):
                sl = (slice(iy * 7, iy * 7 + 7), slice(ix * 7, ix * 7 + 7))
                v = valid[t][sl]
                n = v.sum()
                assert out.valid_frac[t, iy, ix] == pytest.approx(n / 49)
                if n:
                    exp = obs[t, 0][sl][v].mean()
                    assert out.obs[t, 0, iy, ix] == pytest.approx(exp, rel=1e-5)
                    se = np.sqrt((unc[t, 0][sl][v] ** 2).sum()) / n
                    assert out.obs_unc[t, 0, iy, ix] == pytest.approx(se, rel=1e-5)
                else:
                    assert np.isnan(out.obs[t, 0, iy, ix])


def test_aggregate_reproject_path_matches_block_mean():
    from get_eo._aggregate import _reproject_average, aggregate_to
    fine, coarse, *_ = _fine_coarse(nc=3, nt=1, nb=1)
    ref = aggregate_to(fine, coarse).compute()
    m, se, fr = _reproject_average(fine.compute(), coarse)
    np.testing.assert_allclose(fr, ref.valid_frac.values, atol=1e-5)
    np.testing.assert_allclose(m, ref.obs.values, atol=1e-4)


# ------------------------------------------------------------------ schema
def _valid_ds():
    import dask.array as da
    nt, nb, ny, nx = 2, 3, 4, 5
    return xr.Dataset(
        dict(obs=(("time", "band", "y", "x"),
                  da.zeros((nt, nb, ny, nx), dtype="float32")),
             obs_unc=(("time", "band", "y", "x"),
                      da.ones((nt, nb, ny, nx), dtype="float32")),
             valid=(("time", "y", "x"), da.ones((nt, ny, nx), dtype=bool)),
             **{k: (("time", "ay", "ax"), np.full((nt, 2, 2), v, "float32"))
                for k, v in dict(sza=30, saa=120, vza=5, vaa=90).items()}),
        coords=dict(time=np.array(["2022-07-01", "2022-07-02"], "datetime64[ns]"),
                    x=np.arange(nx) * 10.0, y=np.arange(ny) * -10.0,
                    ax=[0.0, 5000.0], ay=[0.0, -5000.0],
                    band=["a", "b", "c"],
                    wl_centre=("band", np.array([1, 2, 3], "float32")),
                    platform=("time", ["p", "p"]), item_id=("time", ["i", "j"])),
        attrs=dict(sensor="s", product="p", obs_kind="reflectance", units="1",
                   source="aws", tile="14TQL"))


def test_validate_passes_and_fails():
    ds = _valid_ds()
    validate_obs_dataset(ds)
    with pytest.raises(ValueError, match="attribute"):
        validate_obs_dataset(ds.assign_attrs(units=None).drop_attrs()
                             if False else _drop(ds, attr="units"))
    with pytest.raises(ValueError, match="monotonic"):
        validate_obs_dataset(ds.isel(time=[1, 0]))
    with pytest.raises(ValueError, match="dask"):
        d = ds.copy()
        d["obs"] = d["obs"].compute()
        validate_obs_dataset(d)
    with pytest.raises(ValueError, match="out of range"):
        d = ds.copy()
        d["sza"] = d["sza"] + 200
        validate_obs_dataset(d)
    with pytest.raises(ValueError, match="valid"):
        validate_obs_dataset(ds.drop_vars("valid"))


def _drop(ds, attr):
    d = ds.copy()
    d.attrs = {k: v for k, v in ds.attrs.items() if k != attr}
    return d


def test_angles_at_interpolates_and_wraps_azimuth():
    ds = _valid_ds()
    ds["saa"] = ds["saa"] * 0 + xr.DataArray(
        np.array([[359.0, 1.0], [359.0, 1.0]], "float32"), dims=("ay", "ax"))
    out = angles_at(ds, [2500.0], [-2500.0])
    assert abs(((float(out.saa[0, 0]) + 180) % 360) - 180) < 0.1  # ~0, not 180
    assert float(out.sza[0, 0]) == pytest.approx(30.0)


def test_quiet_reductions_match_numpy():
    from get_eo._geometry import block_median_quiet, nanmean_quiet
    rng = np.random.default_rng(1)
    a = rng.random((32, 48)).astype("float32")
    a[rng.random(a.shape) < 0.3] = np.nan
    a[:16, :16] = np.nan                                  # an all-NaN block
    got = block_median_quiet(a, 16)
    ref = np.full((2, 3), np.nan, dtype="float32")
    for i in range(2):
        for j in range(3):
            blk = a[i * 16:(i + 1) * 16, j * 16:(j + 1) * 16]
            if np.isfinite(blk).any():
                ref[i, j] = np.nanmedian(blk)
    np.testing.assert_allclose(got, ref, rtol=1e-6)
    assert np.isnan(got[0, 0])
    np.testing.assert_allclose(
        nanmean_quiet(a[16:], axis=0), np.nanmean(a[16:], axis=0), rtol=1e-5)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def test_set_verbosity_logs_and_redacts():
    import io
    import logging

    import get_eo as geo
    from get_eo._log import short_href

    buf = io.StringIO()
    try:
        geo.set_verbosity("INFO", stream=buf)
        logging.getLogger("get_eo.test").info("hello %s", "x?sig=SECRET&a=1")
        out = buf.getvalue()
        assert "hello" in out and "SECRET" not in out
    finally:
        geo.set_verbosity("WARNING", stream=io.StringIO())
    assert short_href("https://h/a/b/c/d.tif?sig=SECRET") == "b/c/d.tif"
    with pytest.raises(ValueError):
        geo.set_verbosity("LOUD")
