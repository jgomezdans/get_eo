"""GDAL environment and URL signing for the lazy xarray readers.

Everything here is passed *into* each ``rasterio.Env`` rather than set
process-wide, so reads are safe in dask worker threads and sources can't
leak configuration into each other.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time

import requests

from get_eo._log import gdal_debug_env

log = logging.getLogger(__name__)

COMMON_GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.jp2,.JP2,.xml",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "VSI_CACHE": "TRUE",
}

_CONFIG_DIR = os.path.expanduser("~/.get_eo")
_COOKIE_FILE = os.path.join(_CONFIG_DIR, "earthdata_cookies.txt")

PLANETARY_SAS_URL = (
    "https://planetarycomputer.microsoft.com/api/sas/v1/token/{collection}"
)


CDSE_S3_ENDPOINT = "eodata.dataspace.copernicus.eu"


def cdse_credentials() -> tuple[str, str]:
    """CDSE S3 (access key, secret key).

    ``CDSE_S3_ACCESS_KEY`` / ``CDSE_S3_SECRET_KEY`` take priority, then the
    ``cdse`` section of ``~/.get_eo/config.json`` (``~/.eof/config.json``,
    the file the older ``eof`` package used, is read as a fallback).
    Raises EnvironmentError with set-up instructions if missing.
    """
    key = os.environ.get("CDSE_S3_ACCESS_KEY", "")
    secret = os.environ.get("CDSE_S3_SECRET_KEY", "")
    if not (key and secret):
        for path in ("~/.get_eo/config.json", "~/.eof/config.json"):
            try:
                with open(os.path.expanduser(path)) as f:
                    cfg = json.load(f).get("cdse", {})
            except (OSError, ValueError):
                continue
            key = key or cfg.get("s3_access_key", "")
            secret = secret or cfg.get("s3_secret_key", "")
    log.debug("CDSE S3 keys %s", "found" if key and secret else "NOT found")
    if not (key and secret):
        raise EnvironmentError(
            "CDSE needs S3 keys. Set CDSE_S3_ACCESS_KEY and "
            "CDSE_S3_SECRET_KEY (create them at "
            "https://eodata-s3keysmanager.dataspace.copernicus.eu), or put "
            "them in the 'cdse' section of ~/.get_eo/config.json as "
            "s3_access_key / s3_secret_key.")
    return key, secret


def cdse_s3_client():
    """boto3 S3 client for the CDSE object store (for small XML files)."""
    try:
        import boto3
    except ImportError as err:
        raise ImportError(
            "Reading CDSE metadata needs boto3: pip install 'get_eo[cdse]'"
        ) from err
    key, secret = cdse_credentials()
    return boto3.client(
        "s3", endpoint_url=f"https://{CDSE_S3_ENDPOINT}",
        aws_access_key_id=key, aws_secret_access_key=secret)


def gdal_env_for(source: str) -> dict:
    """GDAL config options for ``source``
    ('aws', 'planetary', 'cdse', 'earthdata')."""
    env = dict(COMMON_GDAL_ENV)
    env.update(gdal_debug_env())
    if source == "aws":
        env["AWS_NO_SIGN_REQUEST"] = "YES"
    elif source == "cdse":
        key, secret = cdse_credentials()
        env.update(
            AWS_S3_ENDPOINT=CDSE_S3_ENDPOINT,
            AWS_ACCESS_KEY_ID=key,
            AWS_SECRET_ACCESS_KEY=secret,
            AWS_VIRTUAL_HOSTING="FALSE",
            AWS_HTTPS="YES",
        )
    elif source == "earthdata":
        os.makedirs(os.path.dirname(_COOKIE_FILE), exist_ok=True)
        env.update(
            GDAL_HTTP_NETRC="YES",
            GDAL_HTTP_COOKIEFILE=_COOKIE_FILE,
            GDAL_HTTP_COOKIEJAR=_COOKIE_FILE,
        )
    elif source != "planetary":
        raise ValueError(f"Unknown source {source!r}")
    return env


def check_earthdata_login() -> None:
    """Raise a helpful error if no Earthdata credentials can be found."""
    netrc = os.path.expanduser("~/.netrc")
    has_netrc = False
    if os.path.exists(netrc):
        with open(netrc) as f:
            has_netrc = "urs.earthdata.nasa.gov" in f.read()
    log.debug("Earthdata credentials in ~/.netrc: %s",
              "found" if has_netrc else "NOT found")
    if not has_netrc:
        raise EnvironmentError(
            "ECOSTRESS needs NASA Earthdata Login credentials. Add this to "
            "~/.netrc (chmod 600):\n"
            "  machine urs.earthdata.nasa.gov login <user> password <pass>\n"
            "Register for free at https://urs.earthdata.nasa.gov"
        )


# Planetary Computer SAS tokens, cached per collection until near expiry.
_token_lock = threading.Lock()
_tokens: dict[str, tuple[str, float]] = {}


def _planetary_token(collection: str = "sentinel-2-l2a") -> str:
    with _token_lock:
        tok = _tokens.get(collection)
        if tok and tok[1] - time.time() > 300:
            return tok[0]
        log.info("requesting Planetary Computer SAS token for %s", collection)
        r = requests.get(PLANETARY_SAS_URL.format(collection=collection),
                         timeout=30)
        log.debug("SAS token request: HTTP %d in %.1fs", r.status_code,
                  r.elapsed.total_seconds())
        r.raise_for_status()
        j = r.json()
        # 'msft:expiry' is ISO-8601; fall back to 30 minutes if absent
        expiry = time.time() + 1800
        try:
            from datetime import datetime
            expiry = datetime.fromisoformat(
                j["msft:expiry"].replace("Z", "+00:00")).timestamp()
        except (KeyError, ValueError):
            pass
        _tokens[collection] = (j["token"], expiry)
        return j["token"]


def sign_planetary(href: str) -> str:
    """Append a SAS token to an Azure blob URL. Call at read time, not
    graph-build time: signed URLs expire."""
    return f"{href}?{_planetary_token()}"
