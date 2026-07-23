"""Centralized logging setup for Inno3D Inspection.

Extracted from ``main.py`` top-level. Call ``init_logging()`` once at startup.

Features:
  - File output: Inno3D_Logs/inno3d_log_YYYYMMDD_HHMMSS.txt
  - Optional console handler (``console=True``) for dev / pytest
  - Structured prefixes via standard ``logging.getLogger('SEG')`` etc.
  - Global excepthook for crash tracking
  - Does NOT redirect stdout/stderr when ``redirect_stdio=False`` (for pytest)

Layer: infra (no Qt, no VTK).
"""
from datetime import datetime
import logging
import sys

from inno3d.infra.paths import log_dir


class StreamToLogger:
    """Redirect a stream (stdout/stderr) to a Python logger."""

    def __init__(self, logger: logging.Logger, log_level: int = logging.INFO):
        self.logger = logger
        self.log_level = log_level

    def write(self, buf: str) -> None:
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self) -> None:
        pass


def _global_excepthook(exc_type, exc_value, exc_traceback):
    """Log unhandled exceptions as ERROR (crash tracking)."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.error("Uncaught exception (Crash)", exc_info=(exc_type, exc_value, exc_traceback))


_initialized = False


def init_logging(
    *,
    redirect_stdio: bool = True,
    console: bool = False,
    level: int = logging.DEBUG,
) -> str:
    """Configure application-wide logging. Returns the log file path.

    Parameters
    ----------
    redirect_stdio : bool
        If True, redirect ``sys.stdout`` / ``sys.stderr`` to the file logger.
        Set to False for pytest / CLI tools that need real stdout.
    console : bool
        If True, also add a StreamHandler for console output.
    level : int
        Root logger level.
    """
    global _initialized
    if _initialized:
        return ""

    log_filename = str(log_dir() / f"inno3d_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    handlers = [logging.FileHandler(log_filename, encoding="utf-8")]
    if console:
        handlers.append(logging.StreamHandler(sys.__stdout__))

    logging.basicConfig(
        handlers=handlers,
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Silence noisy libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    # Redirect stdout/stderr
    if redirect_stdio:
        sys.stdout = StreamToLogger(logging.getLogger("STDOUT"), logging.INFO)
        sys.stderr = StreamToLogger(logging.getLogger("STDERR"), logging.ERROR)

    # Global crash handler
    sys.excepthook = _global_excepthook

    _initialized = True
    return log_filename
