"""Offline tests for the CDSE and GEE Sentinel-2 backends and for the
independence of get_eo from the older eof package."""
import ast
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import dask.array as da
import numpy as np
import pystac
import pytest
import xarray as xr
from pyproj import CRS

import get_eo
from get_eo import _env, _gee, _read, _search
from get_eo._assets import S2_BANDS, s2_asset_hrefs, s2_mtd_href
from get_eo._grid import Bounds, TileGrid
from get_eo._s2 import reflectance_offset

XR_DIR = Path(get_eo.__file__).parent
TILE = TileGrid(CRS.from_epsg(32614), 699960.0, 4600020.0, 10.0, 10980, 10980)


# ------------------------------------------------------------ independence
def test_does_not_import_eof():
    """get_eo is standalone: it must not depend on the eof package."""
    bad = []
    for f in XR_DIR.glob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            names = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            for n in names:
                if n == "eof" or n.startswith("eof."):
                    bad.append((f.name, n))
    assert bad == []


# -------------------------------------------------------------------- CDSE
def _cdse_item(item_id, version, day="2022-07-08", orbit=14, with_proj=True):
    ex = {"proj:transform": [10, 0, 699960, 0, -10, 4600020],
          "proj:shape": [10980, 10980], "proj:code": "EPSG:32614"} \
        if with_proj else {}
    keys = ([f"{b}_{r}m" for b, r in [("B02", 10), ("B03", 10), ("B04", 10),
             ("B05", 20), ("B06", 20), ("B07", 20), ("B08", 10),
             ("B8A", 20), ("B11", 20), ("B12", 20)]]
            + ["SCL_20m", "granule_metadata"])
    it = pystac.Item(
        id=item_id, geometry=None, bbox=None,
        datetime=datetime.fromisoformat(day + "T17:00:00+00:00"),
        properties={"platform": "sentinel-2b", "grid:code": "MGRS-14TQL",
                    "processing:version": version,
                    "sat:relative_orbit": orbit})
    for k in keys:
        it.add_asset(k, pystac.Asset(
            href=f"s3://eodata/Sentinel-2/{item_id}/{k}.jp2", extra_fields=ex))
    return it


def test_cdse_asset_keys_and_tile():
    it = _cdse_item("S2B_X_N0400", "04.00")
    h = s2_asset_hrefs(it, "cdse")
    assert h["B02"].endswith("B02_10m.jp2")
    assert h["B8A"].endswith("B8A_20m.jp2")
    assert h["SCL"].endswith("SCL_20m.jp2")
    assert s2_mtd_href(it, "cdse").endswith("granule_metadata.jp2")
    assert _search.s2_tile_of(it) == "14TQL"


def test_cdse_dedupe_keeps_latest_baseline():
    old = _cdse_item("S2B_a_N0400", "04.00")
    new = _cdse_item("S2B_a_N0510", "05.10")
    other = _cdse_item("S2B_b_N0510", "05.10", day="2022-07-13")
    out = _search.dedupe_s2([old, new, other], "cdse")
    assert [i.id for i in out] == ["S2B_a_N0510", "S2B_b_N0510"]


def test_cdse_grid_from_item_and_from_file(monkeypatch):
    g = _search.s2_grid(_cdse_item("a", "05.10"), "cdse")
    assert (g.x0, g.y0, g.res, g.nx) == (699960.0, 4600020.0, 10.0, 10980)
    calls = []
    monkeypatch.setattr(
        _search, "_grid_from_file",
        lambda href, src: calls.append(href) or (
            "EPSG:32614", (10, 0, 699960, 0, -10, 4600020), (10980, 10980)))
    g2 = _search.s2_grid(_cdse_item("a", "05.10", with_proj=False), "cdse")
    assert calls and g2 == g


def test_cdse_offset_is_the_planetary_rule():
    assert reflectance_offset({"processing:version": "05.10"}, "cdse") == -0.1
    assert reflectance_offset({"processing:version": "03.01"}, "cdse") == 0.0


def test_cdse_env_needs_keys(monkeypatch, tmp_path):
    monkeypatch.delenv("CDSE_S3_ACCESS_KEY", raising=False)
    monkeypatch.delenv("CDSE_S3_SECRET_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(EnvironmentError, match="S3 keys"):
        _env.gdal_env_for("cdse")
    monkeypatch.setenv("CDSE_S3_ACCESS_KEY", "k")
    monkeypatch.setenv("CDSE_S3_SECRET_KEY", "s")
    env = _env.gdal_env_for("cdse")
    assert env["AWS_S3_ENDPOINT"] == "eodata.dataspace.copernicus.eu"
    assert env["AWS_VIRTUAL_HOSTING"] == "FALSE"


def test_read_window_rewrites_s3_to_vsis3(monkeypatch):
    seen = []

    def fake_open(path, *a, **k):
        seen.append(path)
        raise RuntimeError("stop")

    monkeypatch.setattr(_read.rasterio, "open", fake_open)
    with pytest.raises(RuntimeError):
        _read.read_window("s3://eodata/a/b.jp2", (0, 0, 1, 1), {}, (0,) * 6)
    assert seen == ["/vsis3/eodata/a/b.jp2"]


def test_unknown_source_rejected():
    with pytest.raises(ValueError, match="Unknown source"):
        get_eo.open_aoi((-96.47, 41.16), "2022-07-01", "2022-07-02",
                        buffer_m=500, source="nope")


# --------------------------------------------------------------------- GEE
def _feature(idx, t_ms, tile="14TQL", sza=30.0):
    p = {"system:index": idx, "system:time_start": t_ms, "MGRS_TILE": tile,
         "SPACECRAFT_NAME": "Sentinel-2B", "CLOUDY_PIXEL_PERCENTAGE": 12.0,
         "MEAN_SOLAR_ZENITH_ANGLE": sza, "MEAN_SOLAR_AZIMUTH_ANGLE": 140.0}
    for b in ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"):
        p[f"MEAN_INCIDENCE_ZENITH_ANGLE_{b}"] = 4.0
        p[f"MEAN_INCIDENCE_AZIMUTH_ANGLE_{b}"] = 350.0 if b == "B2" else 10.0
    return {"id": f"COPERNICUS/S2_SR_HARMONIZED/{idx}", "properties": p}


def test_gee_band_names():
    assert [_gee.gee_band(b) for b in S2_BANDS] == [
        "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]


def test_gee_item_and_grid():
    it = _gee._normalise(_feature("20220708T1", 1657299600000))
    assert it.datetime == datetime(2022, 7, 8, 17, 0, tzinfo=timezone.utc)
    assert _search.s2_tile_of(it) == "14TQL"
    assert it.properties["platform"] == "sentinel-2b"
    it.properties.update({"proj:code": "EPSG:32614",
                          "proj:transform": [10, 0, 699960, 0, -10, 4600020],
                          "proj:shape": [10980, 10980]})
    assert _search.s2_grid(it, "gee") == TILE
    assert _search.dedupe_s2([it], "gee") == [it]


def test_gee_arrays_orientation_masking_and_angles(monkeypatch):
    """xee gives (time, Y, X) with Y possibly ascending; we need north-up."""
    items = [_gee._normalise(_feature("a", 1657299600000)),
             _gee._normalise(_feature("b", 1657386000000, sza=35.0))]
    bounds = Bounds(700000.0, 4599900.0, 700040.0, 4600000.0)   # 4 x 10 px
    ny, nx = 10, 4
    names = [_gee.gee_band(b) for b in S2_BANDS]

    rows = np.arange(ny, dtype="float32")[:, None] * np.ones((1, nx), "float32")
    data = {}
    for n in names:
        data[n] = (("time", "Y", "X"),
                   da.from_array(np.stack([rows, rows]) / 100.0, chunks=-1))
    scl = np.stack([rows, rows]).copy()
    scl[:, 0, 0] = np.nan                       # masked pixel
    data["SCL"] = (("time", "Y", "X"), da.from_array(scl, chunks=-1))
    ds = xr.Dataset(
        data, coords=dict(
            time=np.array(["2022-07-08T17:00", "2022-07-09T17:00"],
                          "datetime64[ns]"),
            Y=np.arange(ny) * 10.0 + 4599905.0,     # ascending (south-up)
            X=np.arange(nx) * 10.0 + 700005.0))

    fake_ee = types.SimpleNamespace(
        ImageCollection=lambda *_: _Chain(), Filter=_Chain())
    monkeypatch.setattr(_gee, "init_ee", lambda project=None: fake_ee)
    monkeypatch.setattr(xr, "open_dataset", lambda *a, **k: ds)

    obs, scl_o, ang, ax, ay = _gee.gee_s2_arrays(items, TILE, bounds)
    assert obs.shape == (2, 10, ny, nx) and obs.dtype == np.float32
    o = obs.compute()
    # north-up: the last source row (largest Y) is now first
    assert o[0, 0, 0, 0] == pytest.approx((ny - 1) / 100)
    assert scl_o.dtype == np.uint8
    s = scl_o.compute()
    assert s[0, -1, 0] == 0 and s[0, 0, 0] == ny - 1     # NaN -> 0 (NODATA)
    assert ang["sun_zen"].shape == (2, 2, 2)
    assert ang["sun_zen"][1, 0, 0] == 35.0
    assert ang["view_zen"][0, 0, 0] == pytest.approx(4.0)
    # circular mean of nine 10 deg and one 350 deg azimuths
    assert ang["view_az"][0, 0, 0] == pytest.approx(8.0, abs=0.5)
    assert list(ax) == [700000.0, 700040.0] and list(ay) == [4600000.0, 4599900.0]


class _Chain:
    """Accepts any chained ee call."""
    def __getattr__(self, _):
        return lambda *a, **k: self

    def __call__(self, *a, **k):
        return self


def test_gee_missing_dependency_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "ee", None)
    with pytest.raises(ImportError, match=r"get_eo\[gee\]"):
        _gee.init_ee()


# ------------------------------------------------------------- Planetary
def _pc_item(gen, baseline, tile="14TQL", datatake="GS2B_20220718T170859_028023"):
    return pystac.Item(
        id=f"S2B_MSIL2A_20220718T170859_R112_T{tile}_{gen}", geometry=None,
        bbox=None, datetime=datetime(2022, 7, 18, 17, 8, 59, tzinfo=timezone.utc),
        properties={"s2:datatake_id": f"{datatake}_N{baseline}",
                    "s2:generation_time": gen, "s2:mgrs_tile": tile})


def test_planetary_dedupe_collapses_reprocessed_baselines():
    items = [_pc_item("2022-07-19T06:07:58Z", "04.00"),
             _pc_item("2024-07-14T21:12:49Z", "05.10"),
             _pc_item("2022-07-19T05:37:34Z", "04.00", tile="15TTF"),
             _pc_item("2024-07-14T21:12:49Z", "05.10", tile="15TTF")]
    out = _search.dedupe_s2(items, "planetary")
    assert len(out) == 2
    assert {i.properties["s2:datatake_id"].rsplit("_", 1)[1] for i in out} \
        == {"N05.10"}
    # distinct datatakes on one tile are kept
    other = _pc_item("2022-07-20T00:00:00Z", "04.00",
                     datatake="GS2B_20220720T170859_028066")
    assert len(_search.dedupe_s2(items[:2] + [other], "planetary")) == 2


# ------------------------------------------------------------------ ERA5
from get_eo import _dem, _meteo, _soil   # noqa: E402


def test_era5_unit_conversion():
    c = _meteo.from_era5_raw(
        np.array(["2022-07-15T12:00"], "datetime64[ns]"),
        t2m=[300.15], d2m=[290.15], u10=[3.0], v10=[4.0], sp=[97000.0],
        tp=[0.0005], ssrd=[3.6e6])
    assert c["t_air"][0] == pytest.approx(27.0)
    assert c["wind_speed"][0] == pytest.approx(5.0)
    assert c["solar_radiation"][0] == pytest.approx(1000.0)   # J m-2 h-1 -> W m-2
    assert c["precipitation"][0] == pytest.approx(0.5)        # m -> mm
    assert c["p_atm"][0] == 97000.0
    # 27 degC with a 17 degC dew point: es(17)/es(27) = 19.4/35.7 hPa
    assert c["rh"][0] == pytest.approx(0.544, abs=0.005)


def test_era5_openmeteo_matches_csv_reader_schema(monkeypatch):
    hours = [f"2022-07-15T{h:02d}:00" for h in range(3)]
    calls = []

    def fake(url, params, tries=4):
        calls.append(params)
        return dict(latitude=41.25, longitude=-96.5, elevation=356.0, hourly=dict(
            time=hours, temperature_2m=[20.0, 21.0, 22.0],
            relative_humidity_2m=[50, 60, 101], wind_speed_10m=[1.0, 2.0, 3.0],
            shortwave_radiation=[0.0, -1.0, 300.0], precipitation=[0.0, 1.2, 0.0],
            surface_pressure=[970.0, 971.0, 972.0]))

    monkeypatch.setattr(_meteo, "_get_json", fake)
    ds = _meteo.open_era5((-96.4701, 41.1651), "2022-07-15", "2022-07-15",
                          co2_ppm=410.0)
    assert list(ds.data_vars) == list(_meteo.VARIABLES)
    assert dict(ds.sizes) == {"time": 3}
    assert ds.rh.values.tolist() == pytest.approx([0.5, 0.6, 1.0])    # clipped
    assert ds.solar_radiation.min() == 0.0
    assert ds.p_atm.values[0] == pytest.approx(97000.0)               # hPa -> Pa
    assert (ds.co2_ppm == 410.0).all()
    assert ds.t_air.attrs["units"] == "°C" and ds.rh.attrs["units"] == "1"
    assert float(ds.latitude) == 41.25
    assert calls[0]["wind_speed_unit"] == "ms" and calls[0]["timezone"] == "GMT"


def test_era5_multi_year_requests_and_bad_source(monkeypatch):
    seen = []

    def fake(url, params, tries=4):
        seen.append((params["start_date"], params["end_date"]))
        return dict(latitude=0, longitude=0, hourly=dict(
            time=[params["start_date"] + "T00:00"], temperature_2m=[1.0],
            relative_humidity_2m=[50], wind_speed_10m=[1.0],
            shortwave_radiation=[0.0], precipitation=[0.0],
            surface_pressure=[1000.0]))

    monkeypatch.setattr(_meteo, "_get_json", fake)
    ds = _meteo.open_era5((0.0, 0.0), "2021-12-30", "2022-01-02")
    assert seen == [("2021-12-30", "2021-12-31"), ("2022-01-01", "2022-01-02")]
    assert ds.sizes["time"] == 2
    with pytest.raises(ValueError, match="Unknown source"):
        _meteo.open_era5((0.0, 0.0), "2022-01-01", "2022-01-02", source="x")


# ------------------------------------------------------------------ soil
def test_soil_snap_bounds_and_utm():
    b = _soil.snap_bounds(710010.0, 4559010.0, 710240.0, 4559260.0, 250.0)
    assert (b.xmin, b.xmax, b.ymin, b.ymax) == (710000.0, 710250.0,
                                                 4559000.0, 4559500.0)
    # degenerate (point) AOI still gets one cell
    b = _soil.snap_bounds(500.0, 500.0, 500.0, 500.0, 250.0)
    assert b.xmax - b.xmin == 250.0
    assert _soil.utm_crs_for(-96.47, 41.16).to_epsg() == 32614
    assert _soil.utm_crs_for(150.0, -33.0).to_epsg() == 32756


def test_soil_scaling_and_dataset(monkeypatch):
    def fake_read(prop, depth, stat, crs, bounds, res, nx, ny):
        a = np.full((ny, nx), 350.0, "float32")        # stored units
        a[0, 0] = np.nan
        return a

    monkeypatch.setattr(_soil, "_read", fake_read)
    ds = _soil.open_soilgrids((-96.4701, 41.1651), properties=("clay", "phh2o"),
                              depths=("0-5cm", "5-15cm"),
                              stats=("mean", "uncertainty"), buffer_m=500)
    assert ds.clay.dims == ("depth", "y", "x")
    assert float(ds.clay.isel(depth=0, y=1, x=1)) == pytest.approx(35.0)  # g/kg -> %
    assert float(ds.phh2o.isel(depth=0, y=1, x=1)) == pytest.approx(35.0)
    assert float(ds.clay_uncertainty.isel(depth=0, y=1, x=1)) == 350.0   # unscaled %
    assert np.isnan(ds.clay.isel(depth=0, y=0, x=0))
    assert ds.depth_mid_cm.values.tolist() == [2.5, 10.0]
    assert ds.clay.attrs["units"] == "%"
    assert ds.rio.crs.to_epsg() == 32614
    assert float(ds.inside_aoi.max()) <= 1.0


def test_soil_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown property"):
        _soil.open_soilgrids((0.0, 0.0), properties=("gold",), buffer_m=100)
    with pytest.raises(ValueError, match="Unknown depth"):
        _soil.open_soilgrids((0.0, 0.0), depths=("0-1cm",), buffer_m=100)


def test_era5_gee_source(monkeypatch):
    import types
    from get_eo import _gee as gee

    names = _meteo._GEE_BANDS
    base = int(np.datetime64("2022-07-15T00:00", "ms").astype("int64"))
    table = [["id", "longitude", "latitude", "time"] + names]
    for h in (2, 0, 1):                       # out of order on purpose
        table.append(["x", -96.45, 41.15, base + h * 3_600_000,
                      300.15, 290.15, 3.0, 4.0, 97000.0, 0.0005, 3.6e6 * h])
    table.append(["x", -96.45, 41.15, base + 3 * 3_600_000] + [None] * 7)

    class FakeIC:
        def filterDate(self, *a): return self
        def select(self, *a): return self
        def getRegion(self, *a): return types.SimpleNamespace(
            getInfo=lambda: table)

    fake = types.SimpleNamespace(
        ImageCollection=lambda _id: FakeIC(),
        Geometry=types.SimpleNamespace(Point=lambda c: c))
    monkeypatch.setattr(gee, "init_ee", lambda project=None, need_xee=True: fake)
    ds = _meteo.open_era5((-96.4701, 41.1651), "2022-07-15", "2022-07-15",
                          source="gee")
    assert ds.sizes["time"] == 3                      # None row dropped
    assert np.all(np.diff(ds.time.values.astype("int64")) > 0)
    assert ds.t_air.values[0] == pytest.approx(27.0)
    assert ds.solar_radiation.values.tolist() == pytest.approx([0, 1000, 2000])
    assert ds.attrs["model"] == "era5_land"
    assert list(ds.data_vars) == list(_meteo.VARIABLES)


def test_unreadable_metadata_item_is_dropped(monkeypatch, mtd_xml):
    from get_eo import _s2

    items = [_cdse_item("good_a", "05.10"), _cdse_item("broken", "05.10"),
             _cdse_item("good_b", "05.10")]
    for it in items:
        it.add_asset("granule_metadata", pystac.Asset(href=f"s3://eodata/{it.id}.xml"))

    def fake_fetch(href, source):
        if "broken" in href:
            return mtd_xml[:len(mtd_xml) // 2]          # truncated
        return mtd_xml

    monkeypatch.setattr(_s2, "_fetch_mtd", fake_fetch)
    monkeypatch.setattr(_s2, "gdal_env_for", lambda s: {})
    monkeypatch.setattr(_s2.time, "sleep", lambda s: None)
    grid = TILE
    bounds = Bounds(700000.0, 4599000.0, 700280.0, 4600000.0)
    with pytest.warns(UserWarning, match="broken"):
        _, _, ang, _, _, kept = _s2._stac_arrays(
            items, "cdse", grid, bounds, ("B02", "B03"), None, 1024)
    assert [i.id for i in kept] == ["good_a", "good_b"]
    assert ang["sun_zen"].shape[0] == 2


# ----------------------------------------------------------- soil temp, DEM
def test_era5_soil_temperature_openmeteo(monkeypatch):
    seen = {}

    def fake(url, params, tries=4):
        seen.update(params)
        h = dict(time=["2022-07-15T00:00", "2022-07-15T01:00"],
                 temperature_2m=[20.0, 21.0], relative_humidity_2m=[50, 60],
                 wind_speed_10m=[1.0, 2.0], shortwave_radiation=[0.0, 0.0],
                 precipitation=[0.0, 0.0], surface_pressure=[970.0, 971.0])
        for i, n in enumerate(_meteo._OM_SOIL):
            h[n] = [10.0 + i, 11.0 + i]
        return dict(latitude=41.25, longitude=-96.5, hourly=h)

    monkeypatch.setattr(_meteo, "_get_json", fake)
    plain = _meteo.open_era5((-96.47, 41.16), "2022-07-15", "2022-07-15")
    assert not [v for v in plain.data_vars if v.startswith("t_soil")]
    ds = _meteo.open_era5((-96.47, 41.16), "2022-07-15", "2022-07-15",
                          soil_temp=True)
    assert all(n in seen["hourly"] for n in _meteo._OM_SOIL)
    assert [v for v in ds.data_vars if v.startswith("t_soil")] == [
        "t_soil_1", "t_soil_2", "t_soil_3", "t_soil_4"]
    assert ds.t_soil_3.values.tolist() == [12.0, 13.0]
    assert ds.t_soil_3.attrs["depth_top_cm"] == 28
    assert ds.t_soil_4.attrs["depth_bottom_cm"] == 289


def test_dem_tile_names():
    assert _dem.tile_name(41, -97) == "Copernicus_DSM_COG_10_N41_00_W097_00_DEM"
    assert _dem.tile_name(-5, 10) == "Copernicus_DSM_COG_10_S05_00_E010_00_DEM"
    assert len(_dem.tiles_for(-97.1, 40.9, -95.9, 41.1)) == 6


def test_slope_aspect_on_planes():
    res = 30.0
    cols = np.arange(8) * res
    rise_east = np.tile(0.1 * cols, (8, 1))                  # +10 % toward east
    s, a = _dem.slope_aspect(rise_east, res)
    assert s[4, 4] == pytest.approx(np.degrees(np.arctan(0.1)), abs=1e-4)
    assert a[4, 4] == pytest.approx(270.0, abs=1e-3)         # faces west
    rise_north = np.tile((0.1 * cols)[::-1][:, None], (1, 8))  # rows run south
    s, a = _dem.slope_aspect(rise_north, res)
    assert a[4, 4] == pytest.approx(180.0, abs=1e-3)         # faces south
    s, a = _dem.slope_aspect(np.full((5, 5), 100.0), res)
    assert np.isnan(a).all() and (s == 0).all()


def test_open_dem_grid_and_crop(monkeypatch):
    got = {}

    def fake(geom, crs, bounds, res, nx, ny):
        got.update(nx=nx, ny=ny, bounds=bounds)
        j, i = np.mgrid[0:ny, 0:nx]
        return (300.0 + 0.5 * i * res).astype("float32")      # plane rising east

    monkeypatch.setattr(_dem, "_read_aws", fake)
    ds = _dem.open_dem((-96.4701, 41.1649), buffer_m=500)
    assert ds.elevation.shape == (got["ny"] - 2, got["nx"] - 2)
    assert float(ds.aspect.mean()) == pytest.approx(270.0, abs=1e-3)
    assert float(ds.slope.mean()) == pytest.approx(
        np.degrees(np.arctan(0.5)), abs=1e-3)
    assert float(ds.x[1] - ds.x[0]) == 30.0
    assert (float(ds.x[0]) - 15.0) % 30.0 == 0.0             # aligned to res
    assert ds.rio.crs.to_epsg() == 32614
    with pytest.raises(ValueError, match="Unknown DEM source"):
        _dem.open_dem((0.0, 0.0), source="x", buffer_m=100)


def test_era5_openmeteo_soil_moisture(monkeypatch):
    seen = []

    def fake(url, params, tries=4):
        seen.append(params["hourly"])
        h = dict(time=["2022-07-15T00:00"], temperature_2m=[20.0],
                 relative_humidity_2m=[50], wind_speed_10m=[1.0],
                 shortwave_radiation=[0.0], precipitation=[0.0],
                 surface_pressure=[970.0])
        for i, name in enumerate(_meteo._OM_SWC):
            h[name] = [0.30 - 0.02 * i]
        return dict(latitude=41.25, longitude=-96.5, hourly=h)

    monkeypatch.setattr(_meteo, "_get_json", fake)
    ds = _meteo.open_era5((-96.47, 41.16), "2022-07-15", "2022-07-15",
                          soil_moisture=True)
    assert "soil_moisture_0_to_7cm" in seen[0]
    assert [float(ds[f"swc_{i}"].values[0]) for i in range(1, 5)] == pytest.approx(
        [0.30, 0.28, 0.26, 0.24])
    assert ds.swc_1.attrs["units"] == "m3 m-3"
    assert "swc_1" not in _meteo.open_era5(
        (-96.47, 41.16), "2022-07-15", "2022-07-15")
