"""Centralized logging configuration.

Every module obtains its logger via :func:`get_logger` rather than calling
``logging.getLogger`` directly, so a single call to :func:`configure_logging`
(made once by the CLI entry point) controls verbosity and, for a given
experiment run, also mirrors all output to a per-experiment log file. This
guarantees that optimization traces, particle-filter diagnostics, and
convergence-failure analyses are never silently lost -- they are always
captured in the experiment's own directory, in addition to the console.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False
_ROOT_LOGGER_NAME = "dnlssm"

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: str = "INFO", log_file: Path | None = None) -> logging.Logger:
    """Configures the ``dnlssm`` root logger. Safe to call more than once.

    Parameters
    ----------
    level:
        One of ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``.
    log_file:
        If provided, all records are additionally written to this file
        (typically ``<experiment_dir>/run.log``), so the full log of a run
        is a first-class, versionable experiment artifact.
    """
    global _CONFIGURED
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    if not _CONFIGURED:
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(stream_handler)
        _CONFIGURED = True
    else:
        for handler in logger.handlers:
            handler.setLevel(level)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        # Avoid attaching duplicate file handlers if configure_logging is called
        # again (e.g. once per model-selection candidate) with the same path.
        existing_targets = {
            Path(h.baseFilename).resolve()
            for h in logger.handlers
            if isinstance(h, logging.FileHandler)
        }
        if log_file.resolve() not in existing_targets:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
            logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Returns a child logger under the shared ``dnlssm`` namespace."""
    if not name.startswith(_ROOT_LOGGER_NAME):
        name = f"{_ROOT_LOGGER_NAME}.{name}"
    return logging.getLogger(name)
