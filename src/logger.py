"""
Structured logger.

Thin wrapper around stdlib logging that adds a consistent format and
makes it easy to attach key=value metadata to any log line.

Usage:
    from src.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Detection complete", plate="SBA1234L", elapsed_ms=420)
"""

import logging
import os
from typing import Any


_LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

_configured = False


def _configure() -> None:
    global _configured
    if _configured:
        return

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
    )

    # Suppress noisy third-party loggers in production
    for noisy in ("easyocr", "torch", "PIL", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


class _StructuredAdapter(logging.LoggerAdapter):
    """Appends key=value pairs from `extra` kwargs to every log message."""

    def process(self, msg: str, kwargs: dict) -> tuple[str, dict]:
        extra_kv = kwargs.pop("extra_kv", {})
        if extra_kv:
            kv_str = "  " + "  ".join(f"{k}={v!r}" for k, v in extra_kv.items())
            msg = msg + kv_str
        return msg, kwargs

    # Convenience: allow logger.info("msg", key=val, key2=val2)
    def _log_with_kv(self, level: int, msg: str, **kv: Any) -> None:
        if self.isEnabledFor(level):
            self.log(level, msg, extra_kv=kv)

    def debug(self, msg: str, **kv: Any) -> None:      # type: ignore[override]
        self._log_with_kv(logging.DEBUG, msg, **kv)

    def info(self, msg: str, **kv: Any) -> None:       # type: ignore[override]
        self._log_with_kv(logging.INFO, msg, **kv)

    def warning(self, msg: str, **kv: Any) -> None:    # type: ignore[override]
        self._log_with_kv(logging.WARNING, msg, **kv)

    def error(self, msg: str, **kv: Any) -> None:      # type: ignore[override]
        self._log_with_kv(logging.ERROR, msg, **kv)

    def exception(self, msg: str, **kv: Any) -> None:  # type: ignore[override]
        self._log_with_kv(logging.ERROR, msg, exc_info=True, **kv)


def get_logger(name: str) -> _StructuredAdapter:
    _configure()
    base = logging.getLogger(name)
    return _StructuredAdapter(base, extra={})
