# inno3d/services/volume_pyramid.py
# -----------------------------------------------------------------------
# Pyramid dimension math + background builder for multi-resolution cache.
# -----------------------------------------------------------------------
"""Pyramid helpers and background level builder.

Pure math helpers are covered by ``tests/unit/test_volume_pyramid.py``.
``PyramidBuilder`` writes level-0 lossless + 2× downsampled levels into a
``DiskPyramidCache`` with progressive readiness (L0 usable before higher
levels finish).
"""

from __future__ import annotations

import threading
from typing import Callable, Dict, Optional, Tuple

import numpy as np

VolumeShape = Tuple[int, int, int]
BrickCoord = Tuple[int, int, int]

ProgressCallback = Callable[[str, float], None]  # (message, fraction 0..1)


def downsample_dim(size: int, level: int) -> int:
    """Voxel count along one axis at pyramid *level* (2× downsample per level)."""
    if level < 0:
        raise ValueError(f"level must be >= 0, got {level}")
    if size <= 0:
        return 0
    return max(1, (size + (1 << level) - 1) >> level)


def level_shape(shape: VolumeShape, level: int) -> VolumeShape:
    """Full (Z, Y, X) shape at pyramid *level*."""
    return tuple(downsample_dim(int(dim), level) for dim in shape)  # type: ignore[return-value]


def num_pyramid_levels(shape: VolumeShape, min_dim: int = 1) -> int:
    """Number of usable pyramid levels until every axis is <= *min_dim*."""
    if min_dim < 1:
        raise ValueError(f"min_dim must be >= 1, got {min_dim}")
    if any(d <= 0 for d in shape):
        return 0
    level = 0
    while any(d > min_dim for d in level_shape(shape, level)):
        level += 1
    return level + 1


def brick_grid_shape(shape: VolumeShape, brick_size: int) -> VolumeShape:
    """Number of bricks along (Z, Y, X) for fixed cubic *brick_size*."""
    if brick_size <= 0:
        raise ValueError(f"brick_size must be > 0, got {brick_size}")
    return tuple((dim + brick_size - 1) // brick_size for dim in shape)  # type: ignore[return-value]


def brick_bounds(
    coord: BrickCoord,
    brick_size: int,
    volume_shape: VolumeShape,
) -> Tuple[int, int, int, int, int, int]:
    """Inclusive-start, exclusive-end (z0, y0, x0, z1, y1, x1) for one brick."""
    if brick_size <= 0:
        raise ValueError(f"brick_size must be > 0, got {brick_size}")
    bz, by, bx = coord
    z, y, x = volume_shape
    z0, y0, x0 = bz * brick_size, by * brick_size, bx * brick_size
    return z0, y0, x0, min(z0 + brick_size, z), min(y0 + brick_size, y), min(x0 + brick_size, x)


def level_downsample_factor(level: int) -> int:
    """Voxel stride between adjacent samples at *level* (2**level)."""
    if level < 0:
        raise ValueError(f"level must be >= 0, got {level}")
    return 1 << level


def downsample_2x(volume: np.ndarray) -> np.ndarray:
    """2× box downsample (mean of 2×2×2 blocks, truncated odd edges).

    Uses floor division so output matches ``level_shape(..., 1)`` for even
    dims and ``::2`` sampling semantics used by ``MemoryVolumeStore``.
    """
    if volume.ndim != 3:
        raise ValueError(f"expected 3-D volume, got shape {volume.shape}")
    # Match MemoryVolumeStore stride sampling for parity with get_slice(level=1)
    return np.ascontiguousarray(volume[::2, ::2, ::2])


class PyramidBuilder:
    """Background builder: L0 lossless + successive 2× pyramid levels."""

    def __init__(
        self,
        disk_cache,
        source,
        *,
        brick_size: int = 64,
        max_levels: Optional[int] = None,
        min_dim: int = 32,
        value_range: Optional[Tuple[float, float]] = None,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> None:
        self.disk_cache = disk_cache
        self.source = source  # ndarray or memmap-like
        self.brick_size = int(brick_size)
        self.min_dim = int(min_dim)
        shape = tuple(int(x) for x in source.shape)
        n_auto = num_pyramid_levels(shape, min_dim=self.min_dim)
        self.max_levels = int(max_levels) if max_levels is not None else n_auto
        self.max_levels = max(1, min(self.max_levels, n_auto))
        self.value_range = value_range
        self.progress_cb = progress_cb
        self._thread: Optional[threading.Thread] = None
        self._cancel = threading.Event()
        self._error: Optional[BaseException] = None
        self._ready_levels: set[int] = set()
        self._lock = threading.Lock()

    @property
    def error(self) -> Optional[BaseException]:
        return self._error

    def ready_levels(self) -> set[int]:
        with self._lock:
            return set(self._ready_levels)

    def is_level_ready(self, level: int) -> bool:
        with self._lock:
            return int(level) in self._ready_levels

    def _report(self, message: str, fraction: float) -> None:
        if self.progress_cb is not None:
            try:
                self.progress_cb(message, float(fraction))
            except Exception:
                pass

    def cancel(self) -> None:
        self._cancel.set()

    def start_background(self) -> threading.Thread:
        if self._thread is not None and self._thread.is_alive():
            return self._thread
        self._cancel.clear()
        self._thread = threading.Thread(
            target=self.build,
            name="PyramidBuilder",
            daemon=True,
        )
        self._thread.start()
        return self._thread

    def build(self) -> Dict:
        """Synchronously build pyramid; also used as thread target."""
        try:
            return self._build_impl()
        except BaseException as exc:  # noqa: BLE001 — surface to callers
            self._error = exc
            raise

    def _build_impl(self) -> Dict:
        from inno3d.infra.volume_cache import estimate_value_range

        src = self.source
        shape = tuple(int(x) for x in src.shape)
        dtype = np.dtype(src.dtype)
        levels_meta: Dict[str, dict] = {}

        # --- Level 0 (lossless copy into cache) ---
        self._report("Writing level-0 bricks...", 0.02)
        if self._cancel.is_set():
            raise RuntimeError("pyramid build cancelled")

        info0 = self.disk_cache.write_bricks_from_volume(
            0,
            src,
            brick_size=self.brick_size,
            progress_cb=lambda _lvl, done, total: self._report(
                f"Level-0 bricks {done}/{total}",
                0.02 + 0.48 * (done / max(total, 1)),
            ),
        )
        levels_meta["0"] = info0
        with self._lock:
            self._ready_levels.add(0)

        # Progressive publish: L0 usable before higher levels finish
        vmin, vmax = self.value_range if self.value_range is not None else estimate_value_range(src)
        meta = {
            "digest": self.disk_cache.identity.digest(),
            "schema": self.disk_cache.identity.schema,
            "identity": {
                "path": self.disk_cache.identity.path,
                "size": self.disk_cache.identity.size,
                "mtime_ns": self.disk_cache.identity.mtime_ns,
                "shape": list(self.disk_cache.identity.shape),
                "dtype": self.disk_cache.identity.dtype,
                "offset": self.disk_cache.identity.offset,
            },
            "value_range": [float(vmin), float(vmax)],
            "brick_size": self.brick_size,
            "levels": levels_meta,
            "complete": False,
        }
        self.disk_cache.publish(meta)

        # --- Higher levels ---
        prev = self.disk_cache.open_level_memmap(0, mode="r")
        try:
            for level in range(1, self.max_levels):
                if self._cancel.is_set():
                    raise RuntimeError("pyramid build cancelled")
                self._report(f"Building pyramid level {level}...", 0.5 + 0.5 * (level / self.max_levels))
                down = downsample_2x(prev)
                info = self.disk_cache.write_bricks_from_volume(
                    level, down, brick_size=self.brick_size
                )
                levels_meta[str(level)] = info
                with self._lock:
                    self._ready_levels.add(level)
                meta["levels"] = levels_meta
                self.disk_cache.publish(meta)
                # Cascade from newly written level
                del prev
                prev = self.disk_cache.open_level_memmap(level, mode="r")
                del down
        finally:
            del prev

        meta["complete"] = True
        self.disk_cache.publish(meta)
        self._report("Pyramid ready", 1.0)
        return meta
