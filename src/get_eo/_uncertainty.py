"""Observation-uncertainty models and radiance <-> brightness temperature.

Radiance/BT conversion here is monochromatic at the band centre. That is
adequate for a noise model (and for a convenience BT); a proper
SRF-weighted conversion belongs with the SRFs, which are not bundled yet.
"""

from __future__ import annotations

import numpy as np

# Planck constants in units giving radiance in W m-2 sr-1 um-1, wl in um.
_C1 = 1.191042972e8     # 2 h c^2   [W um^4 m-2 sr-1]
_C2 = 1.438776877e4     # h c / k   [um K]


def s2_sigma(rho, sigma_abs: float = 0.005, c: float = 0.05):
    """sigma^2 = sigma_abs^2 + (c * rho)^2 (D9).

    Negative reflectances are treated as zero in the relative term.
    """
    rho = np.maximum(rho, 0.0)
    return np.sqrt(sigma_abs ** 2 + (c * rho) ** 2)


def planck_radiance(t_k, wl_um):
    """Spectral radiance [W m-2 sr-1 um-1] at temperature(s) ``t_k``."""
    return _C1 / (wl_um ** 5 * np.expm1(_C2 / (wl_um * t_k)))


def radiance_to_bt(rad, wl_um):
    """Brightness temperature [K] from spectral radiance."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return _C2 / (wl_um * np.log1p(_C1 / (wl_um ** 5 * rad)))


def dplanck_dt(t_k, wl_um):
    """dB/dT [W m-2 sr-1 um-1 K-1]."""
    e = np.exp(_C2 / (wl_um * t_k))
    return _C1 * _C2 * e / (wl_um ** 6 * t_k ** 2 * (e - 1.0) ** 2)


def eco_sigma_radiance(rad, wl_um, nedt_k: float = 0.15):
    """1-sigma radiance uncertainty from NEdT at the observed BT."""
    t = radiance_to_bt(rad, wl_um)
    return nedt_k * dplanck_dt(t, wl_um)
