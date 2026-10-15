"""Logging for get_eo.

Every module logs to ``logging.getLogger(__name__)``, i.e. children of the
``get_eo`` logger, which carries a ``NullHandler`` so importing the package
prints nothing. Call :func:`set_verbosity` (or set ``GET_EO_LOG``) to see it:

* ``INFO``: what is being done and how it went (searches and hit counts,
  tile choice, granules kept, one line per remote fetch with timings).
* ``DEBUG``: also every window read, URL and retry, plus GDAL's own HTTP
  tracing (``CPL_DEBUG`` / ``CPL_CURL_VERBOSE``). Very noisy; meant for
  diagnosing network, credential and proxy problems.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from contextlib import contextmanager

ROOT = "get_eo"
ENV_VAR = "GET_EO_LOG"
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

# signed-URL secrets (Azure SAS, AWS presigned) that GDAL's tracing prints
_SECRET = re.compile(r"([?&](?:sig|token|X-Amz-Signature|X-Amz-Credential"
                     r"|X-Amz-Security-Token)=)[^&\s)\"']+", re.I)


class _RedactingFormatter(logging.Formatter):
    def format(self, record):
        return _SECRET.sub(r"\1***", super().format(record))


logging.getLogger(ROOT).addHandler(logging.NullHandler())

_handler: logging.Handler | None = None
_gdal_debug = False


def _as_level(level) -> int:
    if level is True:
        return logging.DEBUG
    if level is False or level is None:
        return logging.WARNING
    if isinstance(level, int):
        return level
    name = str(level).strip().upper()
    if name in ("VERBOSE", "TRACE"):
        return logging.DEBUG
    value = logging.getLevelName(name)
    if not isinstance(value, int):
        raise ValueError(
            f"Unknown log level {level!r}; use DEBUG, INFO, WARNING, ERROR "
            "(or True/False)")
    return value


def set_verbosity(level="INFO", *, stream=None) -> None:
    """Send get_eo's log messages to ``stream`` (default stderr).

    Parameters
    ----------
    level : str, int or bool
        ``'INFO'`` reports what get_eo does; ``'DEBUG'`` (or ``True``) is
        very noisy and also turns on GDAL's HTTP tracing, which shows each
        request GDAL makes. ``'WARNING'`` / ``False`` quiets it again.
    stream : file-like, optional
        Where to write; default ``sys.stderr``.

    Safe to call repeatedly; only one handler is ever installed. If you
    configure :mod:`logging` yourself, skip this and attach handlers to the
    ``'get_eo'`` logger.
    """
    global _handler, _gdal_debug
    lvl = _as_level(level)
    root = logging.getLogger(ROOT)
    if _handler is not None:
        root.removeHandler(_handler)
    _handler = logging.StreamHandler(stream or sys.stderr)
    _handler.setFormatter(_RedactingFormatter(_FORMAT, "%H:%M:%S"))
    root.addHandler(_handler)
    root.setLevel(lvl)
    _handler.setLevel(lvl)
    _gdal_debug = lvl <= logging.DEBUG
    # GDAL messages arrive through rasterio's logger once CPL_DEBUG is on
    rio = logging.getLogger("rasterio")
    rio.removeHandler(_handler)
    if _gdal_debug:
        rio.addHandler(_handler)
        rio.setLevel(logging.DEBUG)
    root.debug("log level %s%s", logging.getLevelName(lvl),
               " (GDAL HTTP tracing on)" if _gdal_debug else "")


def gdal_debug_env() -> dict:
    """Extra GDAL options for ``rasterio.Env`` when DEBUG logging is on."""
    if _gdal_debug:
        return {"CPL_DEBUG": "ON", "CPL_CURL_VERBOSE": "YES"}
    return {}


@contextmanager
def timed(log: logging.Logger, msg: str, *args, level: int = logging.INFO):
    """Log ``msg`` (at DEBUG) on entry and ``msg`` + elapsed time on exit.

    On failure the elapsed time and exception are logged at WARNING and the
    exception propagates.
    """
    log.debug("start: " + msg, *args)
    t0 = time.perf_counter()
    try:
        yield
    except BaseException as err:
        log.warning(msg + " FAILED after %.1fs: %s: %s", *args,
                    time.perf_counter() - t0, type(err).__name__, err)
        raise
    log.log(level, msg + " done in %.1fs", *args, time.perf_counter() - t0)


def short_href(href: str, keep: int = 3) -> str:
    """Last ``keep`` path parts of a URL, without any query string (so SAS
    tokens never reach the log)."""
    path = str(href).split("?", 1)[0]
    return "/".join(path.rstrip("/").split("/")[-keep:])


if os.environ.get(ENV_VAR):
    try:
        set_verbosity(os.environ[ENV_VAR])
    except ValueError:
        pass
