"""Lightweight, env-gated timing helpers for viewer performance profiling.

Enable logging by setting:
    INNO3D_PERF=1
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Generator, Optional

_TRUE_VALUES = {"1", "true", "yes", "on", "y"}


def perf_enabled() -> bool:
    """Return True when INNO3D_PERF is enabled."""
    value = os.getenv("INNO3D_PERF", "")
    return value.strip().lower() in _TRUE_VALUES


def _log_perf(phase: str, elapsed_ms: float, detail: str = "") -> None:
    suffix = f" | {detail}" if detail else ""
    print(f"[PERF] {phase}: {elapsed_ms:.2f} ms{suffix}")


@contextmanager
def perf_phase(phase: str, detail: str = "") -> Generator[None, None, None]:
    """Time a code block and print a [PERF] line when enabled."""
    if not perf_enabled():
        yield
        return

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _log_perf(phase=phase, elapsed_ms=elapsed_ms, detail=detail)


def perf_timed(phase: Optional[str] = None) -> Callable:
    """Decorator variant of perf_phase."""

    def _decorator(func: Callable) -> Callable:
        timed_phase = phase or func.__name__

        @wraps(func)
        def _wrapped(*args, **kwargs):
            with perf_phase(timed_phase):
                return func(*args, **kwargs)

        return _wrapped

    return _decorator
