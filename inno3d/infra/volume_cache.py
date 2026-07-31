# inno3d/infra/volume_cache.py
# -----------------------------------------------------------------------
# Hybrid brick + pyramid disk/RAM cache for large_volume_engine.
# -----------------------------------------------------------------------
"""Disk pyramid cache + bounded LRU RAM brick cache.

Cache identity is keyed by source path, size, mtime, dtype/shape/offset and
schema version so stale bricks are never reused after a source change.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from inno3d.infra.paths import app_runtime_dir
from inno3d.services.volume_pyramid import brick_bounds, brick_grid_shape, level_shape

CACHE_SCHEMA_VERSION = 1
DEFAULT_BRICK_SIZE = 64
DEFAULT_RAM_BUDGET_BYTES = 512 * 1024 * 1024  # 512 MiB

VolumeShape = Tuple[int, int, int]
BrickCoord = Tuple[int, int, int]
BrickKey = Tuple[int, BrickCoord]  # (level, coord)


@dataclass(frozen=True)
class CacheIdentity:
    """Stable identity for a volume source + import parameters."""

    path: str
    size: int
    mtime_ns: int
    shape: VolumeShape
    dtype: str
    offset: int = 0
    schema: int = CACHE_SCHEMA_VERSION

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def make_cache_identity(
    path: str | Path,
    shape: VolumeShape,
    dtype,
    *,
    offset: int = 0,
    size: Optional[int] = None,
    mtime_ns: Optional[int] = None,
) -> CacheIdentity:
    p = Path(path).resolve()
    st = p.stat()
    return CacheIdentity(
        path=str(p),
        size=int(size if size is not None else st.st_size),
        mtime_ns=int(mtime_ns if mtime_ns is not None else getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
        shape=tuple(int(x) for x in shape),  # type: ignore[arg-type]
        dtype=np.dtype(dtype).str,
        offset=int(offset),
        schema=CACHE_SCHEMA_VERSION,
    )


def default_cache_root() -> Path:
    """``<runtime>/Inno3D_Data/volume_cache`` — writable next to the app."""
    root = app_runtime_dir() / "Inno3D_Data" / "volume_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


class RamBrickCache:
    """Thread-safe LRU brick cache with a soft byte budget."""

    def __init__(self, max_bytes: int = DEFAULT_RAM_BUDGET_BYTES) -> None:
        self.max_bytes = max(1, int(max_bytes))
        self._lock = threading.RLock()
        self._entries: "OrderedDict[BrickKey, np.ndarray]" = OrderedDict()
        self._nbytes = 0
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def nbytes(self) -> int:
        with self._lock:
            return self._nbytes

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._nbytes = 0

    def get(self, level: int, coord: BrickCoord) -> Optional[np.ndarray]:
        key: BrickKey = (int(level), tuple(int(c) for c in coord))  # type: ignore[assignment]
        with self._lock:
            arr = self._entries.get(key)
            if arr is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return arr

    def put(self, level: int, coord: BrickCoord, brick: np.ndarray) -> None:
        key: BrickKey = (int(level), tuple(int(c) for c in coord))  # type: ignore[assignment]
        arr = np.ascontiguousarray(brick)
        cost = int(arr.nbytes)
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self._nbytes -= int(old.nbytes)
            while self._entries and (self._nbytes + cost) > self.max_bytes:
                _, evicted = self._entries.popitem(last=False)
                self._nbytes -= int(evicted.nbytes)
            if cost <= self.max_bytes:
                self._entries[key] = arr
                self._nbytes += cost


class DiskPyramidCache:
    """On-disk pyramid layout with atomic publish via ``ready.json``."""

    def __init__(self, identity: CacheIdentity, root: Optional[Path] = None) -> None:
        self.identity = identity
        self.root = Path(root) if root is not None else default_cache_root()
        self.dir = self.root / identity.digest()
        self.meta_path = self.dir / "meta.json"
        self.ready_path = self.dir / "ready.json"
        self._lock = threading.RLock()

    def ensure_dir(self) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        return self.dir

    def is_ready(self) -> bool:
        if not self.ready_path.is_file() or not self.meta_path.is_file():
            return False
        try:
            meta = self.read_meta()
        except Exception:
            return False
        return (
            meta.get("digest") == self.identity.digest()
            and meta.get("schema") == CACHE_SCHEMA_VERSION
            and bool(meta.get("levels"))
        )

    def level_ready(self, level: int) -> bool:
        if not self.is_ready() and level == 0:
            # Progressive: L0 file may exist before full publish
            return self.level_path(0).is_file()
        meta = self.read_meta() if self.meta_path.is_file() else {}
        levels = meta.get("levels") or {}
        return str(int(level)) in levels and self.level_path(level).is_file()

    def level_path(self, level: int) -> Path:
        return self.dir / f"L{int(level)}.dat"

    def read_meta(self) -> Dict[str, Any]:
        with open(self.meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def write_meta(self, meta: Dict[str, Any]) -> None:
        self.ensure_dir()
        tmp = self.meta_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        os.replace(tmp, self.meta_path)

    def publish(self, meta: Dict[str, Any]) -> None:
        """Atomically mark cache as ready after meta + level files are written."""
        self.write_meta(meta)
        ready = {
            "digest": self.identity.digest(),
            "schema": CACHE_SCHEMA_VERSION,
            "published_at": time.time(),
        }
        tmp = self.ready_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(ready, f, indent=2, sort_keys=True)
        os.replace(tmp, self.ready_path)

    def open_level_memmap(self, level: int, mode: str = "r") -> np.memmap:
        meta = self.read_meta()
        info = meta["levels"][str(int(level))]
        shape = tuple(int(x) for x in info["shape"])
        dtype = np.dtype(info["dtype"])
        path = self.level_path(level)
        if not path.is_file():
            raise FileNotFoundError(path)
        return np.memmap(str(path), dtype=dtype, mode=mode, shape=shape)

    def allocate_level(self, level: int, shape: VolumeShape, dtype) -> np.memmap:
        """Create/overwrite a level file and return a writable memmap."""
        self.ensure_dir()
        dtype = np.dtype(dtype)
        path = self.level_path(level)
        nbytes = int(np.prod(shape)) * dtype.itemsize
        # Pre-size file
        with open(path, "wb") as f:
            f.truncate(nbytes)
        return np.memmap(str(path), dtype=dtype, mode="r+", shape=tuple(int(x) for x in shape))

    def read_brick(
        self,
        level: int,
        coord: BrickCoord,
        *,
        brick_size: int = DEFAULT_BRICK_SIZE,
        ram: Optional[RamBrickCache] = None,
    ) -> np.ndarray:
        if ram is not None:
            hit = ram.get(level, coord)
            if hit is not None:
                return hit
        vol = self.open_level_memmap(level, mode="r")
        try:
            z0, y0, x0, z1, y1, x1 = brick_bounds(coord, brick_size, vol.shape)
            brick = np.array(vol[z0:z1, y0:y1, x0:x1], copy=True)
        finally:
            del vol
        if ram is not None:
            ram.put(level, coord, brick)
        return brick

    def write_bricks_from_volume(
        self,
        level: int,
        volume: np.ndarray,
        *,
        brick_size: int = DEFAULT_BRICK_SIZE,
        progress_cb=None,
    ) -> Dict[str, Any]:
        """Write a dense (or memmap) volume into ``L{level}.dat`` losslessly."""
        shape = tuple(int(x) for x in volume.shape)
        mm = self.allocate_level(level, shape, volume.dtype)
        try:
            grid = brick_grid_shape(shape, brick_size)
            total = int(np.prod(grid))
            done = 0
            for bz in range(grid[0]):
                for by in range(grid[1]):
                    for bx in range(grid[2]):
                        z0, y0, x0, z1, y1, x1 = brick_bounds((bz, by, bx), brick_size, shape)
                        mm[z0:z1, y0:y1, x0:x1] = volume[z0:z1, y0:y1, x0:x1]
                        done += 1
                        if progress_cb is not None and (done == total or done % 32 == 0):
                            progress_cb(level, done, total)
            mm.flush()
        finally:
            del mm
        return {
            "shape": list(shape),
            "dtype": np.dtype(volume.dtype).str,
            "brick_size": int(brick_size),
            "grid": list(grid),
        }


def estimate_value_range(volume: np.ndarray, max_samples: int = 2_000_000) -> Tuple[float, float]:
    """Fast min/max via strided sampling — avoids full 15 GB scans."""
    if volume.size == 0:
        return 0.0, 0.0
    if volume.size <= max_samples:
        return float(np.min(volume)), float(np.max(volume))
    step = max(1, volume.size // max_samples)
    flat = volume.ravel()
    sample = flat[::step]
    return float(np.min(sample)), float(np.max(sample))
