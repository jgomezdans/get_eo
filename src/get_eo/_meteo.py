"""Hourly ERA5 meteorology at a site, as an xarray Dataset.

Produces the same variables, names, units and attributes as the AmeriFlux
CSV reader used for the crop-model drivers, so ERA5 and tower data are
interchangeable::

    t_air [degC]  rh [0-1]  wind_speed [m/s]  solar_radiation [W/m2]
    precipitation [mm]  p_atm [Pa]  co2_ppm [ppm]

ERA5 is 0.25 degrees, so an AOI is represented by the grid cell nearest its
centroid and the result has a single ``time`` dimension.

Sources
-------
``openmeteo`` (default)
    The Open-Meteo historical API, which serves ERA5 (``models=era5``) or
    ERA5-Land with no account. Free for non-commercial use, CC BY 4.0.
    Open-Meteo is not the raw grid cell: it adjusts temperature and
    pressure to the elevation of the requested point (about +0.15 degC and
    +2 hPa at US-Ne1, where the cell is 356 m), and rounds values (0.1 degC,
    0.1 mm, integer % RH). Use ``cds`` for unmodified ERA5.
``gee``
    ERA5-Land hourly (``ECMWF/ERA5_LAND/HOURLY``, 0.1 degrees) from Google
    Earth Engine (``pip install 'get_eo[gee]'``; ``earthengine authenticate``).
    ERA5-Land is a land-only replay of ERA5 at higher resolution, so its
    values differ slightly from ERA5 proper. Unmodified, unrounded.
``cds``
    The Copernicus Climate Data Store ERA5 time-series dataset
    (``pip install cdsapi``; needs a current-format key in
    ``~/.cdsapirc`` and the ERA5 licence accepted).

Time convention (both sources): timestamps are UTC. ``t_air``, ``rh``,
``wind_speed`` and ``p_atm`` are instantaneous values at the timestamp;
``solar_radiation`` (mean) and ``precipitation`` (sum) are over the hour
*ending* at the timestamp (the ERA5 accumulation convention).
"""

from __future__ import annotations

import io
import logging
import os
import time
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import requests
import xarray as xr

from get_eo._log import timed

log = logging.getLogger(__name__)

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
CDS_URL = "https://cds.climate.copernicus.eu/api"
CDS_DATASET = "reanalysis-era5-single-levels-timeseries"
CDS_VARIABLES = ["2m_temperature", "2m_dewpoint_temperature",
                 "10m_u_component_of_wind", "10m_v_component_of_wind",
                 "surface_pressure", "total_precipitation",
                 "surface_solar_radiation_downwards"]
# short names used in the CDS NetCDF
_CDS_SHORT = {"2m_temperature": "t2m", "2m_dewpoint_temperature": "d2m",
              "10m_u_component_of_wind": "u10", "10m_v_component_of_wind": "v10",
              "surface_pressure": "sp", "total_precipitation": "tp",
              "surface_solar_radiation_downwards": "ssrd"}

VARIABLES = ("t_air", "rh", "wind_speed", "solar_radiation", "precipitation",
             "p_atm", "co2_ppm")

# ERA5 / ERA5-Land soil temperature layers (depth below surface [cm])
SOIL_LAYERS = {"t_soil_1": (0, 7), "t_soil_2": (7, 28),
               "t_soil_3": (28, 100), "t_soil_4": (100, 289)}
# ERA5 / ERA5-Land volumetric soil water layers, same depths [cm]
SWC_LAYERS = {"swc_1": (0, 7), "swc_2": (7, 28),
              "swc_3": (28, 100), "swc_4": (100, 289)}
_CDS_SWC = [f"volumetric_soil_water_layer_{i}" for i in range(1, 5)]
_OM_SWC = ["soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm",
           "soil_moisture_28_to_100cm", "soil_moisture_100_to_255cm"]
_CDS_SOIL = [f"soil_temperature_level_{i}" for i in range(1, 5)]
_OM_SOIL = ["soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm",
            "soil_temperature_28_to_100cm", "soil_temperature_100_to_255cm"]

_ATTRS = {
    "t_air": dict(long_name="Air temperature", units="°C",
                  standard_name="air_temperature"),
    "rh": dict(long_name="Relative humidity", units="1",
               standard_name="relative_humidity"),
    "wind_speed": dict(long_name="Wind speed", units="m/s",
                       standard_name="wind_speed"),
    "solar_radiation": dict(
        long_name="Downwelling shortwave radiation", units="W/m²",
        standard_name="surface_downwelling_shortwave_flux_in_air"),
    "precipitation": dict(long_name="Precipitation", units="mm",
                          standard_name="precipitation_amount"),
    "p_atm": dict(long_name="Atmospheric pressure", units="Pa",
                  standard_name="air_pressure"),
    "co2_ppm": dict(long_name="CO₂ mole fraction", units="ppm",
                    standard_name="mole_fraction_of_carbon_dioxide_in_air"),
}
for _k, (_t, _b) in SWC_LAYERS.items():
    _ATTRS[_k] = dict(long_name=f"Volumetric soil water, {_t}-{_b} cm",
                      units="m3 m-3", standard_name="soil_moisture_content",
                      depth_top_cm=_t, depth_bottom_cm=_b)
for _k, (_t, _b) in SOIL_LAYERS.items():
    _ATTRS[_k] = dict(long_name=f"Soil temperature, {_t}-{_b} cm",
                      units="°C", standard_name="soil_temperature",
                      depth_top_cm=_t, depth_bottom_cm=_b)


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------

def saturation_vapour_pressure(t_c):
    """Magnus formula over water [hPa] (Alduchov & Eskridge 1996)."""
    return 6.1094 * np.exp(17.625 * t_c / (t_c + 243.04))


def relative_humidity(t_c, td_c):
    """RH as a fraction from air and dew-point temperature [degC]."""
    rh = saturation_vapour_pressure(td_c) / saturation_vapour_pressure(t_c)
    return np.clip(rh, 0.0, 1.0)


def from_era5_raw(time, t2m, d2m, u10, v10, sp, tp, ssrd) -> dict:
    """ERA5 native units -> schema units.

    ``t2m``/``d2m`` [K], ``u10``/``v10`` [m/s], ``sp`` [Pa], ``tp`` [m,
    hourly accumulation], ``ssrd`` [J/m2, hourly accumulation].
    """
    t_c = np.asarray(t2m, "float64") - 273.15
    td_c = np.asarray(d2m, "float64") - 273.15
    return dict(
        time=np.asarray(time),
        t_air=t_c,
        rh=relative_humidity(t_c, td_c),
        wind_speed=np.hypot(u10, v10),
        solar_radiation=np.maximum(np.asarray(ssrd, "float64") / 3600.0, 0.0),
        precipitation=np.maximum(np.asarray(tp, "float64") * 1000.0, 0.0),
        p_atm=np.asarray(sp, "float64"),
    )


def _build(cols: dict, lat: float, lon: float, co2_ppm: float,
           attrs: dict) -> xr.Dataset:
    n = len(cols["time"])
    data = {k: ("time", np.asarray(cols[k], "float32")) for k in VARIABLES[:-1]}
    data["co2_ppm"] = ("time", np.full(n, co2_ppm, "float32"))
    for k in (*SOIL_LAYERS, *SWC_LAYERS):
        if k in cols:
            data[k] = ("time", np.asarray(cols[k], "float32"))
    ds = xr.Dataset(
        data,
        coords=dict(time=np.asarray(cols["time"], "datetime64[ns]"),
                    latitude=lat, longitude=lon),
        attrs=attrs)
    for k, a in _ATTRS.items():
        if k in ds:
            ds[k].attrs = dict(a)
    ds["co2_ppm"].attrs["comment"] = (
        "constant (ERA5 has no CO2); pass co2_ppm= to change")
    if n:
        ds.attrs["n_timesteps"] = n
        ds.attrs["time_start"] = str(ds.time.values[0])
        ds.attrs["time_end"] = str(ds.time.values[-1])
    return ds


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def _year_ranges(start: str, end: str):
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    if e < s:
        raise ValueError("end is before start")
    for y in range(s.year, e.year + 1):
        yield max(s, date(y, 1, 1)), min(e, date(y, 12, 31))


def _get_json(url, params, tries=4):
    for k in range(tries):
        log.debug("GET %s %s", url, params)
        try:
            r = requests.get(url, params=params, timeout=120)
        except requests.RequestException as err:
            log.warning("GET %s failed: %s: %s", url, type(err).__name__, err)
            raise
        log.debug("HTTP %d, %d bytes in %.1fs", r.status_code, len(r.content),
                  r.elapsed.total_seconds())
        if r.status_code == 429 and k < tries - 1:
            log.warning("%s rate-limited (HTTP 429); retry %d/%d in %ds",
                        url, k + 1, tries - 1, 2 ** (k + 1))
            time.sleep(2 ** (k + 1))
            continue
        if r.status_code != 200:
            log.error("%s: HTTP %d: %s", url, r.status_code, r.text[:300])
            raise RuntimeError(f"{url}: HTTP {r.status_code}: {r.text[:300]}")
        return r.json()


def _fetch_openmeteo(lat, lon, start, end, model, soil_temp=False,
                     soil_moisture=False):
    frames, cell = [], {}
    for a, b in _year_ranges(start, end):
        log.info("Open-Meteo %s: requesting %s..%s at (%.4f, %.4f)",
                 model, a, b, lon, lat)
        j = _get_json(OPEN_METEO_URL, dict(
            latitude=lat, longitude=lon, start_date=a.isoformat(),
            end_date=b.isoformat(), models=model, timezone="GMT",
            wind_speed_unit="ms",
            hourly="temperature_2m,relative_humidity_2m,wind_speed_10m,"
                   "shortwave_radiation,precipitation,surface_pressure"
                   + ("," + ",".join(_OM_SOIL) if soil_temp else "")
                   + ("," + ",".join(_OM_SWC) if soil_moisture else "")))
        cell = dict(latitude=j["latitude"], longitude=j["longitude"],
                    elevation=j.get("elevation"))
        log.info("Open-Meteo: %d hourly rows; grid cell (%.3f, %.3f), "
                 "elevation %s m", len(j["hourly"]["time"]), cell["longitude"],
                 cell["latitude"], cell["elevation"])
        frames.append(pd.DataFrame(j["hourly"]))
    df = pd.concat(frames, ignore_index=True)
    cols = dict(
        time=pd.to_datetime(df["time"]).values,
        t_air=df["temperature_2m"].values,
        rh=np.clip(df["relative_humidity_2m"].values / 100.0, 0.0, 1.0),
        wind_speed=df["wind_speed_10m"].values,
        solar_radiation=np.maximum(df["shortwave_radiation"].values, 0.0),
        precipitation=np.maximum(df["precipitation"].values, 0.0),
        p_atm=df["surface_pressure"].values * 100.0,        # hPa -> Pa
    )
    if soil_temp:
        for k, name in zip(SOIL_LAYERS, _OM_SOIL):
            cols[k] = df[name].values
    if soil_moisture:
        for k, name in zip(SWC_LAYERS, _OM_SWC):
            cols[k] = df[name].values
    attrs = dict(source="open-meteo", model=model,
                 note="elevation-adjusted by Open-Meteo; values rounded",
                 license="CC BY 4.0 (Open-Meteo, contains modified "
                         "Copernicus Climate Change Service information)")
    return cols, cell, attrs


def _read_cds_file(path) -> xr.Dataset:
    with open(path, "rb") as f:
        head = f.read(4)
    if head[:2] == b"PK":                       # zip of NetCDF files
        with zipfile.ZipFile(path) as z:
            parts = [xr.open_dataset(io.BytesIO(z.read(n)), engine="h5netcdf")
                     .load() for n in z.namelist() if n.endswith(".nc")]
        return xr.merge(parts, compat="override")
    return xr.open_dataset(path).load()


def _fetch_cds(lat, lon, start, end, _model, soil_temp=False,
               soil_moisture=False):
    try:
        import cdsapi
    except ImportError as err:
        raise ImportError("source='cds' needs cdsapi: pip install cdsapi") \
            from err
    import tempfile
    client = cdsapi.Client(quiet=True)
    pieces = []
    with tempfile.TemporaryDirectory() as tmp:
        for a, b in _year_ranges(start, end):
            path = os.path.join(tmp, f"era5_{a.year}.nc")
            log.info("CDS: requesting %s..%s (queued requests can take "
                     "minutes)", a, b)
            t0 = time.perf_counter()
            client.retrieve(CDS_DATASET, {
                "variable": CDS_VARIABLES + (_CDS_SOIL if soil_temp else [])
                + (_CDS_SWC if soil_moisture else []),
                "location": {"longitude": lon, "latitude": lat},
                "date": [f"{a.isoformat()}/{b.isoformat()}"],
                "data_format": "netcdf",
            }).download(path)
            log.info("CDS: downloaded %s..%s in %.0fs (%d bytes)", a, b,
                     time.perf_counter() - t0, os.path.getsize(path))
            ds = _read_cds_file(path)
            tdim = "valid_time" if "valid_time" in ds.dims else "time"
            ds = ds.squeeze(drop=True)
            pieces.append(ds.rename({tdim: "time"}) if tdim != "time" else ds)
    ds = xr.concat(pieces, dim="time")
    cols = from_era5_raw(ds["time"].values,
                         *(ds[_CDS_SHORT[v]].values for v in CDS_VARIABLES))
    if soil_temp:
        for k, name in zip(SOIL_LAYERS, _CDS_SOIL):
            short = "stl" + name[-1]
            cols[k] = np.asarray(ds[short].values, "float64") - 273.15
    if soil_moisture:
        for k, name in zip(SWC_LAYERS, _CDS_SWC):
            cols[k] = np.asarray(ds["swvl" + name[-1]].values, "float64")
    cell = dict(latitude=float(ds.get("latitude", lat)),
                longitude=float(ds.get("longitude", lon)))
    attrs = dict(source="cds", model="era5",
                 license="Contains modified Copernicus Climate Change "
                         "Service information")
    return cols, cell, attrs


GEE_ERA5L = "ECMWF/ERA5_LAND/HOURLY"
# ERA5-Land band per ERA5 variable (the *_hourly bands are de-accumulated)
_GEE_BANDS = ["temperature_2m", "dewpoint_temperature_2m",
              "u_component_of_wind_10m", "v_component_of_wind_10m",
              "surface_pressure", "total_precipitation_hourly",
              "surface_solar_radiation_downwards_hourly"]


def _month_ranges(start: str, end: str):
    s, e = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)
    for a in pd.date_range(s.replace(day=1), e, freq="MS"):
        lo, hi = max(a, s), min(a + pd.offsets.MonthBegin(1), e)
        if lo < hi:
            yield lo, hi


def _fetch_gee(lat, lon, start, end, _model, soil_temp=False,
               soil_moisture=False, project=None):
    from get_eo._gee import init_ee
    ee = init_ee(project, need_xee=False)
    soil = [f"soil_temperature_level_{i}" for i in range(1, 5)] if soil_temp else []
    swc = ([f"volumetric_soil_water_layer_{i}" for i in range(1, 5)]
           if soil_moisture else [])
    names = _GEE_BANDS + soil + swc
    pt = ee.Geometry.Point([lon, lat])
    rows, cell = [], {}
    for lo, hi in _month_ranges(start, end):
        log.info("GEE %s: requesting %s..%s", GEE_ERA5L, lo.date(), hi.date())
        ic = (ee.ImageCollection(GEE_ERA5L)
              .filterDate(lo.strftime("%Y-%m-%d"), hi.strftime("%Y-%m-%d"))
              .select(names))
        # one row per image: [id, lon, lat, time_ms, band values...]
        tab = ic.getRegion(pt, 11132).getInfo()
        head = tab[0]
        idx = {n: head.index(n) for n in ["longitude", "latitude", "time"]
               + names}
        for r in tab[1:]:
            if all(r[idx[b]] is not None for b in names):
                rows.append(r)
                cell = dict(latitude=r[idx["latitude"]],
                            longitude=r[idx["longitude"]])
    if not rows:
        raise LookupError(f"No ERA5-Land data from GEE for {start}..{end}")
    arr = {k: np.array([r[idx[k]] for r in rows], "float64") for k in idx}
    order = np.argsort(arr["time"])
    t = (arr["time"][order] * 1e6).astype("int64").astype("datetime64[ns]")
    cols = from_era5_raw(t, *(arr[b][order] for b in _GEE_BANDS))
    for k, b in zip(SOIL_LAYERS, soil):
        cols[k] = arr[b][order] - 273.15
    for k, b in zip(SWC_LAYERS, swc):
        cols[k] = arr[b][order]
    attrs = dict(source="gee", model="era5_land", dataset=GEE_ERA5L,
                 license="Contains modified Copernicus Climate Change "
                         "Service information")
    return cols, cell, attrs


_SOURCES = {"openmeteo": _fetch_openmeteo, "cds": _fetch_cds,
            "gee": _fetch_gee}


def open_era5(aoi, start: str, end: str, *, source: str = "openmeteo",
              model: str = "era5", co2_ppm: float = 400.0,
              soil_temp: bool = False, soil_moisture: bool = False,
              gee_project: str | None = None) -> xr.Dataset:
    """Hourly ERA5 meteorology for the AOI centroid.

    Parameters
    ----------
    aoi
        Anything :func:`get_eo.open_aoi` accepts, or ``(lon, lat)``.
    start, end : str
        'YYYY-MM-DD', inclusive, UTC.
    source : {'openmeteo', 'cds', 'gee'}
    model : {'era5', 'era5_land'}
        Open-Meteo only. ``era5_land`` is 0.1 degrees (land only).
    soil_temp : bool
        Also return ``t_soil_1..4`` [degC], soil temperature of the four
        ERA5 layers (0-7, 7-28, 28-100, 100-289 cm; ``t_soil_3`` is the
        usual deep-soil boundary condition). Open-Meteo labels the last
        layer 100-255 cm.
    soil_moisture : bool
        Also return ``swc_1..4`` [m3 m-3], volumetric soil water of the four
        ERA5(-Land) layers (same depths as ``soil_temp``). Instantaneous
        values at the timestamp. GEE bands
        ``volumetric_soil_water_layer_1..4``; Open-Meteo ``soil_moisture_*``
        (last layer 100-255 cm); CDS ``volumetric_soil_water_layer_1..4``.
    gee_project : str, optional
        Google Cloud project for ``source='gee'``.
    co2_ppm : float
        ERA5 has no CO2; a constant column is filled in (default 400 ppm,
        as in the AmeriFlux reader).

    Returns
    -------
    xr.Dataset
        ``time`` dimension, variables ``t_air rh wind_speed solar_radiation
        precipitation p_atm co2_ppm`` (plus ``t_soil_1..4`` if asked); scalar ``latitude``/``longitude`` of
        the ERA5 grid cell actually used.
    """
    from get_eo._aoi import resolve_aoi
    if source not in _SOURCES:
        raise ValueError(f"Unknown source {source!r}; supported: "
                         f"{', '.join(_SOURCES)}")
    if (isinstance(aoi, (tuple, list)) and len(aoi) == 2
            and all(isinstance(v, (int, float)) for v in aoi)):
        lon, lat = aoi
    else:
        c = resolve_aoi(aoi).centroid
        lon, lat = c.x, c.y
    log.info("open_era5: source=%s model=%s %s..%s at (%.4f, %.4f)%s",
             source, model, start, end, lon, lat,
             ", with soil temperature" if soil_temp else "")
    fetch = _SOURCES[source]
    kw = {"project": gee_project} if source == "gee" else {}
    cols, cell, attrs = fetch(lat, lon, start, end, model,
                              soil_temp=soil_temp,
                              soil_moisture=soil_moisture, **kw)
    attrs["requested_latitude"], attrs["requested_longitude"] = lat, lon
    if cell.get("elevation") is not None:
        attrs["grid_elevation_m"] = cell["elevation"]
    ds = _build(cols, cell["latitude"], cell["longitude"], co2_ppm, attrs)
    n_nan = int(sum(int(np.isnan(v.values).sum()) for v in ds.data_vars.values()))
    log.info("open_era5: %d hourly steps (%s..%s)%s", ds.sizes["time"],
             ds.attrs.get("time_start"), ds.attrs.get("time_end"),
             f"; {n_nan} NaN values" if n_nan else "")
    return ds
