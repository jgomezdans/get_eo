"""Solar position (NOAA / Meeus-style algorithm, numpy only).

Accuracy is around 0.01 degrees, far better than needed here.
"""

from __future__ import annotations

import numpy as np


def solar_position(time_utc, lon, lat):
    """Solar zenith and azimuth angles.

    Parameters
    ----------
    time_utc : datetime64 (scalar or array)
        UTC time.
    lon, lat : array_like
        Degrees east / north. Broadcast against ``time_utc``.

    Returns
    -------
    sza, saa : np.ndarray
        Solar zenith [deg] and azimuth [deg clockwise from true north].
    """
    t = np.asarray(time_utc, dtype="datetime64[ns]")
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    # Julian day / century
    unix_days = t.astype("datetime64[ns]").astype("int64") / 86400e9
    jd = unix_days + 2440587.5
    jc = (jd - 2451545.0) / 36525.0

    # Sun geometry
    l0 = np.mod(280.46646 + jc * (36000.76983 + jc * 0.0003032), 360.0)
    m = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    mr = np.radians(m)
    ecc = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    c = (np.sin(mr) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
         + np.sin(2 * mr) * (0.019993 - 0.000101 * jc)
         + np.sin(3 * mr) * 0.000289)
    true_lon = l0 + c
    omega = 125.04 - 1934.136 * jc
    app_lon = true_lon - 0.00569 - 0.00478 * np.sin(np.radians(omega))
    mean_obliq = 23.0 + (26.0 + (21.448 - jc * (
        46.815 + jc * (0.00059 - jc * 0.001813))) / 60.0) / 60.0
    obliq = mean_obliq + 0.00256 * np.cos(np.radians(omega))
    decl = np.degrees(np.arcsin(
        np.sin(np.radians(obliq)) * np.sin(np.radians(app_lon))))

    # Equation of time [minutes]
    y = np.tan(np.radians(obliq) / 2) ** 2
    l0r = np.radians(l0)
    eot = 4 * np.degrees(
        y * np.sin(2 * l0r) - 2 * ecc * np.sin(mr)
        + 4 * ecc * y * np.sin(mr) * np.cos(2 * l0r)
        - 0.5 * y * y * np.sin(4 * l0r) - 1.25 * ecc * ecc * np.sin(2 * mr))

    # True solar time and hour angle
    minutes = np.mod(unix_days, 1.0) * 1440.0
    tst = np.mod(minutes + eot + 4 * lon, 1440.0)
    ha = tst / 4.0 - 180.0
    ha = np.where(ha < -180, ha + 360, ha)

    latr, decr, har = np.radians(lat), np.radians(decl), np.radians(ha)
    cosz = (np.sin(latr) * np.sin(decr)
            + np.cos(latr) * np.cos(decr) * np.cos(har))
    cosz = np.clip(cosz, -1.0, 1.0)
    sza = np.degrees(np.arccos(cosz))

    # Azimuth clockwise from north
    sinz = np.sqrt(np.maximum(1.0 - cosz ** 2, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_az = (np.sin(latr) * cosz - np.sin(decr)) / (np.cos(latr) * sinz)
    az = np.degrees(np.arccos(np.clip(cos_az, -1.0, 1.0)))
    saa = np.where(ha > 0, np.mod(az + 180.0, 360.0),
                   np.mod(540.0 - az, 360.0))
    return sza, saa
