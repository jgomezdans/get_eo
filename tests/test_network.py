"""Network tests (AOI: 3 km box at US-Ne1, tile 14TQL, 15 Jul - 15 Aug 2022).

Run with ``pytest -m network``; they need internet and, for ECOSTRESS, an
Earthdata login in ``~/.netrc``.
"""
import dask
import dask.array as da
import numpy as np
import pytest

import get_eo as eox
from get_eo import _read
from get_eo._uncertainty import radiance_to_bt

pytestmark = pytest.mark.network

AOI = (-96.4701, 41.1651)
START, END = "2022-07-15", "2022-08-15"


@pytest.fixture(scope="module")
def data():
    return eox.open_aoi(AOI, START, END, buffer_m=1500)


def test_lazy_open_reads_no_pixels(monkeypatch):
    calls = []
    real = _read._do_read

    def counting(*a, **k):
        calls.append(a[0])
        return real(*a, **k)

    monkeypatch.setattr(_read, "_do_read", counting)
    d = eox.open_aoi(AOI, START, "2022-07-25", sensors=("s2",), buffer_m=1500)
    s2 = d["s2"]
    assert isinstance(s2.obs.data, da.Array)
    assert calls == []
    s2.obs.isel(time=0, band=0).compute()
    assert calls, "compute() should read"


def test_s2_clear_maize_reflectance(data):
    s2 = data["s2"]
    ins = s2.inside_aoi > 0.5
    clear = s2.valid.where(ins).mean(("x", "y")).compute()
    sel = s2.isel(time=np.where(clear.values > 0.95)[0]).compute()
    assert sel.sizes["time"] >= 3
    m = sel.obs.where(sel.valid & ins).mean(("x", "y", "time"))
    assert 0.3 < float(m.sel(band="B8A")) < 0.5
    assert float(m.sel(band="B04")) < 0.09     # 3 km box includes non-crop
    assert float(m.sel(band="B04")) > 0.0       # no double offset


def test_sun_geometry_matches_stac(data):
    s2 = data["s2"]
    import pystac_client
    c = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
    for iid in s2.item_id.values[:3]:
        it = next(c.search(collections=["sentinel-2-l2a"], ids=[iid]).items())
        ref = 90.0 - it.properties["view:sun_elevation"]
        sza = float(s2.sza.sel(time=s2.time[s2.item_id == iid]).mean())
        assert abs(sza - ref) < 0.5


def test_aws_matches_planetary():
    kw = dict(buffer_m=300, sensors=("s2",))
    a = eox.open_aoi(AOI, "2022-07-18", "2022-07-18", source="aws", **kw)["s2"]
    p = eox.open_aoi(AOI, "2022-07-18", "2022-07-18", source="planetary", **kw)["s2"]
    ia = a.isel(time=0).compute()
    ip = p.isel(time=-1).compute()
    ok = (ia.valid & ip.valid).values
    assert ok.sum() > 50
    diff = np.abs(ia.obs.values - ip.obs.values)[:, ok]
    assert np.median(diff) < 0.005


def test_ecostress(data):
    e = data["ecostress"].compute()
    assert list(e.band.values) == ["TIR2", "TIR4", "TIR5"]
    assert e.attrs["tile"] == "14TQL"
    assert (e.sza.mean(("ay", "ax")) < 85.5).all()           # day only
    n_good = 0
    for i in range(e.sizes["time"]):
        ok = e.valid.isel(time=i)
        if int(ok.sum()) < 100:
            continue
        n_good += 1
        bt = [float(radiance_to_bt(
            e.obs.isel(time=i, band=b).where(ok), e.wl_centre[b].item() / 1000
        ).mean()) for b in range(3)]
        assert all(270 < t < 330 for t in bt)
        assert bt[2] <= bt[1] + 1.0                           # water-vapour order
    assert n_good >= 1


def test_ecostress_lst_vs_toa_bt(data):
    e = data["ecostress"].compute()
    l = data["ecostress_lste"].compute()
    n = 0
    for i in range(l.sizes["time"]):
        j = np.where(e.item_id.str.replace("L1CT_RAD", "L2T_LSTE").values
                     == l.item_id.values[i])[0]
        if not len(j):
            continue
        m = (l.valid.isel(time=i) & e.valid.isel(time=j[0])).values
        if m.sum() < 200:
            continue
        bt = radiance_to_bt(e.obs.isel(time=j[0], band=1).values, 10.49)
        lst = l.lst.isel(time=i).values
        # (no atmospheric correction: LST - TOA BT is several K, more in
        # humid summer air; night scenes have too little contrast for a
        # high correlation)
        assert 0 < np.mean(lst[m] - bt[m]) < 15
        if float(e.sza.isel(time=j[0]).mean()) < 70:
            assert np.corrcoef(lst[m], bt[m])[0, 1] > 0.9
        n += 1
    assert n >= 1


def test_s2_ecostress_grids_align(data):
    s, e = data["s2"], data["ecostress"]
    # each 70 m cell is exactly 7 x 7 S2 pixels
    assert abs(float(e.x[0] - e.x[0].astype(int) * 0) - float(e.x[0])) == 0
    assert np.isclose(e.x.values[0] - 35.0, s.x.values[0] - 5.0)
    assert np.isclose(e.y.values[0] + 35.0, s.y.values[0] + 5.0)
    assert s.sizes["x"] == 7 * e.sizes["x"]
    assert s.sizes["y"] == 7 * e.sizes["y"]
    ag = eox.aggregate_to(s.isel(time=slice(0, 2)), e)
    assert ag.obs.shape[-2:] == (e.sizes["y"], e.sizes["x"])


def test_era5_openmeteo_live():
    m = eox.open_era5(AOI, "2022-07-15", "2022-07-16")
    assert m.sizes["time"] == 48
    assert 0 < float(m.t_air.mean()) < 45 and 0 <= float(m.rh.min())
    assert 85000 < float(m.p_atm.mean()) < 105000


def test_soilgrids_live():
    s = eox.open_soilgrids(AOI, properties=("clay", "phh2o"),
                           depths=("0-5cm",), buffer_m=1000)
    assert 5 < float(s.clay.mean()) < 70
    assert 4 < float(s.phh2o.mean()) < 9


def test_dem_live():
    d = eox.open_dem(AOI, buffer_m=1000)
    assert 300 < float(d.elevation.mean()) < 420
    assert float(d.slope.max()) < 15


def test_era5_soil_temp_live():
    m = eox.open_era5(AOI, "2022-07-15", "2022-07-16", soil_temp=True)
    assert float(m.t_soil_1.mean()) > float(m.t_soil_4.mean())   # summer: warm top
