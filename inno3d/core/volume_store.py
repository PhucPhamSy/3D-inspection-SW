# inno3d/core/volume_store.py
# -----------------------------------------------------------------------
# Out-of-core volume access for large_volume_engine.
# Extends Composer stub: MemoryVolumeStore kept; memmap + pyramid backends.
# -----------------------------------------------------------------------
"""VolumeStore public API — metadata + slice/ROI/brick access without full materialization."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence, Tuple, Union

import numpy as np

SliceOrientation = Literal["axial", "coronal", "sagittal"]
ORIENTATIONS: Tuple[SliceOrientation, ...] = ("axial", "coronal", "sagittal")
VolumeShape = Tuple[int, int, int]  # Z, Y, X
Spacing = Tuple[float, float, float]
BrickCoord = Tuple[int, int, int]

_TRUE_VALUES = {"1", "true", "yes", "on", "y"}


def large_volume_engine_enabled(host=None) -> bool:
    """Feature flag: env, ``host.large_volume_engine``, or app settings."""
    env = os.getenv("INNO3D_LARGE_VOLUME_ENGINE", "")
    if env.strip().lower() in _TRUE_VALUES:
        return True
    if host is not None and bool(getattr(host, "large_volume_engine", False)):
        return True
    try:
        from inno3d.app.settings import get_large_volume_engine_enabled

        return get_large_volume_engine_enabled()
    except Exception:
        return False


def large_volume_byte_threshold() -> int:
    """Volumes at/above this size prefer the store path when the flag is on."""
    raw = os.getenv("INNO3D_LARGE_VOLUME_THRESHOLD_GB", "1.0")
    try:
        gb = float(raw)
    except ValueError:
        gb = 1.0
    return int(max(0.0, gb) * (1024 ** 3))


def default_3d_upload_budget_bytes() -> int:
    """Host-side budget for a single 3D VTK upload (default 512 MiB).

    Override with ``INNO3D_3D_UPLOAD_BUDGET_MB`` (integer MiB) or Help tab setting.
    """
    try:
        from inno3d.app.settings import get_3d_upload_budget_mb

        mb = float(get_3d_upload_budget_mb())
    except Exception:
        raw = os.getenv("INNO3D_3D_UPLOAD_BUDGET_MB", "512")
        try:
            mb = float(raw)
        except ValueError:
            mb = 512.0
    return int(max(16.0, mb) * (1024 ** 2))


def level_nbytes(shape: VolumeShape, dtype, level: int = 0) -> int:
    """Approx bytes for a full 3-D array at pyramid *level*."""
    from inno3d.services.volume_pyramid import level_shape

    z, y, x = level_shape(shape, int(level))
    return int(z) * int(y) * int(x) * int(np.dtype(dtype).itemsize)


def choose_3d_mip_level(
    shape: VolumeShape,
    dtype,
    budget_bytes: Optional[int] = None,
    *,
    max_level: int = 8,
    is_available=None,
) -> int:
    """Finest pyramid level whose full array fits in *budget_bytes*.

    Returns the smallest level index (0 = full-res) that fits. If nothing
    fits, returns the coarsest candidate that ``is_available`` accepts
    (or ``max_level``).
    """
    if budget_bytes is None:
        budget_bytes = default_3d_upload_budget_bytes()
    budget_bytes = int(max(1, budget_bytes))
    max_level = int(max(0, max_level))
    last_ok = max_level
    for level in range(0, max_level + 1):
        if is_available is not None and not is_available(level):
            continue
        last_ok = level
        if level_nbytes(shape, dtype, level) <= budget_bytes:
            return level
    return last_ok


@dataclass(frozen=True)
class VolumeMetadata:
    """Immutable volume descriptor exposed by VolumeStore."""

    shape: VolumeShape
    dtype: np.dtype
    spacing: Spacing
    value_range: Tuple[float, float]


class VolumeStore(ABC):
    """Out-of-core volume store.

    Contract:
    - Expose ``shape``, ``dtype``, ``spacing`` via ``metadata`` (or shortcuts).
    - ``get_slice(orientation, index, level=0)`` returns a 2-D array (H, W).
    - ``get_roi(bounds, level=0)`` returns a 3-D sub-volume for explicit workflows.
    - ``get_bricks(brick_coords, level=0)`` returns brick payloads keyed by coord.
    - Must **not** implement ``__array__`` — accidental materialization is forbidden.
    """

    @property
    @abstractmethod
    def metadata(self) -> VolumeMetadata:
        raise NotImplementedError

    @property
    def shape(self) -> VolumeShape:
        return self.metadata.shape

    @property
    def dtype(self) -> np.dtype:
        return self.metadata.dtype

    @property
    def spacing(self) -> Spacing:
        return self.metadata.spacing

    @abstractmethod
    def get_slice(
        self,
        orientation: SliceOrientation,
        index: int,
        level: int = 0,
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_roi(
        self,
        bounds: Tuple[slice, slice, slice],
        level: int = 0,
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_bricks(
        self,
        brick_coords: Sequence[BrickCoord],
        level: int = 0,
        *,
        brick_size: int = 64,
    ) -> dict[BrickCoord, np.ndarray]:
        raise NotImplementedError

    def get_level(self, level: int = 0):
        """Return a readable 3-D array/memmap for *level* (may be expensive)."""
        raise NotImplementedError

    def get_level_shape(self, level: int = 0) -> VolumeShape:
        """Shape at pyramid *level* (level 0 = full resolution)."""
        raise NotImplementedError

    def best_available_level(self, preferred: int = 0) -> int:
        """Finest available level with quality >= preferred when possible.

        Prefer *preferred* if ready; otherwise the nearest coarser ready level,
        then fall back toward level-0.
        """
        preferred = max(0, int(preferred))
        if self._level_available(preferred):
            return preferred
        # Coarser first (faster preview), then finer toward 0
        for level in range(preferred + 1, preferred + 8):
            if self._level_available(level):
                return level
        for level in range(preferred - 1, -1, -1):
            if self._level_available(level):
                return level
        return 0

    def _level_available(self, level: int) -> bool:
        return level == 0

    def sample_value_range(self, max_samples: int = 2_000_000) -> Tuple[float, float]:
        return self.metadata.value_range

    def __array__(self, dtype=None):
        raise TypeError(
            "VolumeStore must not be converted to a full numpy array; "
            "use get_slice/get_roi/get_bricks instead."
        )


def _slice_from_volume(
    vol: np.ndarray,
    orientation: SliceOrientation,
    index: int,
) -> np.ndarray:
    z, y, x = vol.shape
    if orientation == "axial":
        if not 0 <= index < z:
            raise IndexError(f"axial index {index} out of range [0, {z})")
        return np.ascontiguousarray(np.flipud(vol[index, :, :]))
    if orientation == "coronal":
        if not 0 <= index < y:
            raise IndexError(f"coronal index {index} out of range [0, {y})")
        return np.ascontiguousarray(np.flipud(vol[:, index, :]))
    if orientation == "sagittal":
        if not 0 <= index < x:
            raise IndexError(f"sagittal index {index} out of range [0, {x})")
        return np.ascontiguousarray(np.transpose(vol[:, :, index]))
    raise ValueError(f"unknown orientation: {orientation!r}")


def upsample_slice_nearest(slice_2d: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor upsample so coarse MPR previews keep level-0 geometry."""
    th, tw = target_hw
    h, w = slice_2d.shape[:2]
    if (h, w) == (th, tw):
        return slice_2d
    if h == 0 or w == 0:
        return np.zeros((th, tw), dtype=slice_2d.dtype)
    z_idx = np.minimum((np.arange(th) * h) // th, h - 1)
    y_idx = np.minimum((np.arange(tw) * w) // tw, w - 1)
    return np.ascontiguousarray(slice_2d[z_idx][:, y_idx])


def _sample_value_range_zyx(
    data: np.ndarray,
    *,
    max_samples: int = 2_000_000,
) -> Tuple[float, float]:
    """Fast approximate min/max for very large dense arrays.

    Prefers a few full XY planes near mid-Z (axial-friendly sequential reads)
    so cold memmap / page-cache warmup does not random-stride the whole file.
    Falls back to exact min/max when the volume is already small.
    """
    if data.size == 0:
        return 0.0, 0.0
    max_samples = int(max(1, max_samples))
    if int(data.size) <= max_samples:
        return float(np.min(data)), float(np.max(data))
    z, y, x = (int(v) for v in data.shape)
    plane = max(1, int(y) * int(x))
    n_planes = max(1, min(int(z), int(max_samples // plane) or 1))
    if n_planes <= 1:
        zi = int(z) // 2
        sampled = data[zi : zi + 1]
    else:
        # Evenly spaced Z planes through the stack (still mostly sequential per plane).
        zs = np.linspace(0, z - 1, num=n_planes, dtype=np.int64)
        sampled = data[zs]
    if sampled.size == 0:
        sampled = data.reshape(-1)[:1]
    return float(np.min(sampled)), float(np.max(sampled))


class MemoryVolumeStore(VolumeStore):
    """Small in-memory VolumeStore for tests and dev — not for 15 GB volumes."""

    def __init__(
        self,
        data: np.ndarray,
        spacing: Spacing = (1.0, 1.0, 1.0),
        *,
        value_range: Optional[Tuple[float, float]] = None,
        force_contiguous: bool = True,
    ) -> None:
        if data.ndim != 3:
            raise ValueError(f"expected 3-D volume, got shape {data.shape}")
        self._data = np.ascontiguousarray(data) if force_contiguous else data
        if value_range is None:
            if self._data.size > 10_000_000:
                vmin, vmax = _sample_value_range_zyx(self._data)
            else:
                vmin = float(np.min(self._data)) if self._data.size else 0.0
                vmax = float(np.max(self._data)) if self._data.size else 0.0
        else:
            vmin = float(value_range[0])
            vmax = float(value_range[1])
        self._metadata = VolumeMetadata(
            shape=tuple(int(x) for x in self._data.shape),  # type: ignore[arg-type]
            dtype=self._data.dtype,
            spacing=spacing,
            value_range=(vmin, vmax),
        )

    @property
    def metadata(self) -> VolumeMetadata:
        return self._metadata

    def _volume_at_level(self, level: int) -> np.ndarray:
        if level < 0:
            raise ValueError(f"level must be >= 0, got {level}")
        if level == 0:
            return self._data
        step = 1 << level
        return self._data[::step, ::step, ::step]

    def get_level(self, level: int = 0):
        return self._volume_at_level(level)

    def get_level_shape(self, level: int = 0) -> VolumeShape:
        from inno3d.services.volume_pyramid import level_shape

        return level_shape(self.shape, level)

    def _level_available(self, level: int) -> bool:
        return level >= 0

    def get_slice(
        self,
        orientation: SliceOrientation,
        index: int,
        level: int = 0,
    ) -> np.ndarray:
        vol = self._volume_at_level(level)
        return _slice_from_volume(vol, orientation, index)

    def get_roi(
        self,
        bounds: Tuple[slice, slice, slice],
        level: int = 0,
    ) -> np.ndarray:
        vol = self._volume_at_level(level)
        z_sl, y_sl, x_sl = bounds
        return vol[z_sl, y_sl, x_sl].copy()

    def get_bricks(
        self,
        brick_coords: Sequence[BrickCoord],
        level: int = 0,
        *,
        brick_size: int = 64,
    ) -> dict[BrickCoord, np.ndarray]:
        from inno3d.services.volume_pyramid import brick_bounds

        vol = self._volume_at_level(level)
        out: dict[BrickCoord, np.ndarray] = {}
        for coord in brick_coords:
            z0, y0, x0, z1, y1, x1 = brick_bounds(coord, brick_size, vol.shape)
            out[coord] = vol[z0:z1, y0:y1, x0:x1].copy()
        return out


class MemmapVolumeStore(VolumeStore):
    """RAW/memmap-backed store — no full RAM copy; optional disk pyramid levels."""

    def __init__(
        self,
        memmap: np.memmap,
        *,
        spacing: Spacing = (1.0, 1.0, 1.0),
        value_range: Optional[Tuple[float, float]] = None,
        source_path: Optional[str] = None,
        source_offset: int = 0,
        disk_cache=None,
        ram_cache=None,
        own_memmap: bool = True,
    ) -> None:
        if getattr(memmap, "ndim", 0) != 3:
            raise ValueError(f"expected 3-D memmap, got shape {getattr(memmap, 'shape', None)}")
        self._memmap = memmap
        self._own_memmap = own_memmap
        self._source_path = source_path
        self._source_offset = int(source_offset)
        self.disk_cache = disk_cache
        self.ram_cache = ram_cache
        self._pyramid_builder = None
        if value_range is None:
            from inno3d.infra.volume_cache import estimate_value_range

            value_range = estimate_value_range(memmap)
        self._metadata = VolumeMetadata(
            shape=tuple(int(x) for x in memmap.shape),  # type: ignore[arg-type]
            dtype=np.dtype(memmap.dtype),
            spacing=spacing,
            value_range=(float(value_range[0]), float(value_range[1])),
        )

    @classmethod
    def open_raw(
        cls,
        path: str | Path,
        shape: VolumeShape,
        dtype,
        *,
        offset: int = 0,
        spacing: Spacing = (1.0, 1.0, 1.0),
        use_disk_cache: bool = True,
        start_pyramid: bool = True,
        ram_budget_bytes: Optional[int] = None,
    ) -> "MemmapVolumeStore":
        from inno3d.infra.volume_cache import (
            DEFAULT_RAM_BUDGET_BYTES,
            DiskPyramidCache,
            RamBrickCache,
            make_cache_identity,
        )

        path = Path(path)
        dtype = np.dtype(dtype)
        expected = int(np.prod(shape)) * dtype.itemsize + int(offset)
        file_size = path.stat().st_size
        if file_size < expected:
            raise ValueError(
                f"RAW file is too small! Expected {expected} bytes for this shape "
                f"and offset, but file is {file_size} bytes."
            )
        mm = np.memmap(str(path), dtype=dtype, mode="r", offset=int(offset), shape=tuple(shape))
        disk = None
        ram = RamBrickCache(ram_budget_bytes or DEFAULT_RAM_BUDGET_BYTES)
        if use_disk_cache:
            identity = make_cache_identity(path, shape, dtype, offset=offset)
            disk = DiskPyramidCache(identity)
        store = cls(
            mm,
            spacing=spacing,
            source_path=str(path.resolve()),
            source_offset=int(offset),
            disk_cache=disk,
            ram_cache=ram,
        )
        if start_pyramid and disk is not None and not disk.is_ready():
            store.start_pyramid_build()
        elif disk is not None and disk.is_ready():
            try:
                meta = disk.read_meta()
                vr = meta.get("value_range")
                if vr and len(vr) == 2:
                    store._metadata = VolumeMetadata(
                        shape=store._metadata.shape,
                        dtype=store._metadata.dtype,
                        spacing=store._metadata.spacing,
                        value_range=(float(vr[0]), float(vr[1])),
                    )
            except Exception:
                pass
        return store

    def start_pyramid_build(self, **kwargs) -> None:
        if self.disk_cache is None:
            return
        from inno3d.services.volume_pyramid import PyramidBuilder

        builder = PyramidBuilder(
            self.disk_cache,
            self._memmap,
            value_range=self.metadata.value_range,
            **kwargs,
        )
        self._pyramid_builder = builder
        builder.start_background()

    @property
    def metadata(self) -> VolumeMetadata:
        return self._metadata

    def get_level_shape(self, level: int = 0) -> VolumeShape:
        from inno3d.services.volume_pyramid import level_shape

        return level_shape(self.shape, level)

    def _level_available(self, level: int) -> bool:
        # Source memmap always supports on-the-fly stride levels; disk pyramid is optional acceleration.
        return level >= 0

    def get_level(self, level: int = 0):
        level = int(level)
        if level < 0:
            raise ValueError(f"level must be >= 0, got {level}")
        if level == 0:
            return self._memmap
        if self.disk_cache is not None and self.disk_cache.level_ready(level):
            return self.disk_cache.open_level_memmap(level, mode="r")
        # Fallback: on-the-fly stride view (no disk pyramid yet)
        step = 1 << level
        return self._memmap[::step, ::step, ::step]

    def get_slice(
        self,
        orientation: SliceOrientation,
        index: int,
        level: int = 0,
    ) -> np.ndarray:
        level = int(level)
        if level < 0:
            raise ValueError(f"level must be >= 0, got {level}")
        # Prefer disk pyramid when ready; else stride from source memmap
        use_level = level
        if level > 0 and not self._level_available(level):
            # Progressive readiness: fall back toward available coarser/finer
            use_level = self.best_available_level(level)
        vol = self.get_level(use_level)
        try:
            return _slice_from_volume(vol, orientation, index)
        finally:
            if use_level > 0 and vol is not self._memmap:
                del vol

    def get_roi(
        self,
        bounds: Tuple[slice, slice, slice],
        level: int = 0,
    ) -> np.ndarray:
        vol = self.get_level(level)
        z_sl, y_sl, x_sl = bounds
        try:
            return np.array(vol[z_sl, y_sl, x_sl], copy=True)
        finally:
            if level > 0 and vol is not self._memmap:
                del vol

    def get_bricks(
        self,
        brick_coords: Sequence[BrickCoord],
        level: int = 0,
        *,
        brick_size: int = 64,
    ) -> dict[BrickCoord, np.ndarray]:
        from inno3d.services.volume_pyramid import brick_bounds

        out: dict[BrickCoord, np.ndarray] = {}
        if self.disk_cache is not None and self.disk_cache.level_ready(level):
            for coord in brick_coords:
                out[coord] = self.disk_cache.read_brick(
                    level, coord, brick_size=brick_size, ram=self.ram_cache
                )
            return out
        vol = self.get_level(level)
        try:
            for coord in brick_coords:
                if self.ram_cache is not None:
                    hit = self.ram_cache.get(level, coord)
                    if hit is not None:
                        out[coord] = hit
                        continue
                z0, y0, x0, z1, y1, x1 = brick_bounds(coord, brick_size, vol.shape)
                brick = np.array(vol[z0:z1, y0:y1, x0:x1], copy=True)
                if self.ram_cache is not None:
                    self.ram_cache.put(level, coord, brick)
                out[coord] = brick
            return out
        finally:
            if level > 0 and vol is not self._memmap:
                del vol

    def close(self) -> None:
        if self._pyramid_builder is not None:
            try:
                self._pyramid_builder.cancel()
            except Exception:
                pass
        if self._own_memmap:
            try:
                del self._memmap
            except Exception:
                pass


class DenseArrayStore(MemoryVolumeStore):
    """Alias clarifying fallback path for small / already-loaded volumes."""

    pass


class VolumeArrayProxy:
    """Compatibility shim so Viewer can keep ``self.volume_data`` patterns.

    Supports ``.shape``, ``.dtype``, ``.ndim``, ``.size``, indexing for slices,
    and sampled min/max. Does **not** implement ``__array__``.
    """

    def __init__(self, store: VolumeStore) -> None:
        self._store = store

    @property
    def store(self) -> VolumeStore:
        return self._store

    @property
    def shape(self) -> VolumeShape:
        return self._store.shape

    @property
    def dtype(self) -> np.dtype:
        return self._store.dtype

    @property
    def ndim(self) -> int:
        return 3

    @property
    def size(self) -> int:
        z, y, x = self.shape
        return int(z) * int(y) * int(x)

    @property
    def nbytes(self) -> int:
        return int(self.size * np.dtype(self.dtype).itemsize)

    def min(self):
        return self._store.metadata.value_range[0]

    def max(self):
        return self._store.metadata.value_range[1]

    def ravel(self):
        raise TypeError(
            "VolumeArrayProxy cannot ravel a full volume; "
            "use volume_store.sample_value_range() or get_roi()."
        )

    def copy(self):
        raise TypeError(
            "VolumeArrayProxy refuses full-volume copy; "
            "use get_roi() for explicit materialization."
        )

    def __getitem__(self, key):
        # Prefer direct memmap/array backend for numpy-style indexing
        if hasattr(self._store, "get_level"):
            try:
                vol = self._store.get_level(0)
                return vol[key]
            except Exception:
                pass
        if isinstance(self._store, MemoryVolumeStore):
            return self._store._data[key]
        raise TypeError("indexing requires a level-0 backend on the store")

    def __array__(self, dtype=None):
        raise TypeError(
            "VolumeArrayProxy must not be converted to a full numpy array; "
            "use volume_store.get_slice/get_roi instead."
        )

    def __repr__(self) -> str:
        return f"VolumeArrayProxy(shape={self.shape}, dtype={self.dtype})"


def wrap_volume_for_viewer(store: VolumeStore):
    """Return (store, volume_data_proxy) for Viewer assignment."""
    return store, VolumeArrayProxy(store)


def open_raw_volume_store(
    path: str | Path,
    shape: VolumeShape,
    dtype,
    *,
    offset: int = 0,
    spacing: Spacing = (1.0, 1.0, 1.0),
    start_pyramid: bool = True,
) -> MemmapVolumeStore:
    """Factory used by LoadVolumeThread when large_volume_engine is on."""
    return MemmapVolumeStore.open_raw(
        path,
        shape,
        dtype,
        offset=offset,
        spacing=spacing,
        start_pyramid=start_pyramid,
    )


def store_from_dense_array(
    data: np.ndarray,
    spacing: Spacing = (1.0, 1.0, 1.0),
    *,
    prefer_no_copy: bool = False,
    sampled_value_range: bool = False,
    max_samples: int = 2_000_000,
) -> MemoryVolumeStore:
    """Fallback store for TIFF/folder loads that still materialize in RAM."""
    large = int(getattr(data, "size", 0)) > 10_000_000
    value_range = None
    if sampled_value_range or large:
        value_range = _sample_value_range_zyx(data, max_samples=max_samples)
    force_contiguous = True
    if prefer_no_copy and getattr(data, "flags", None) is not None and data.flags["C_CONTIGUOUS"]:
        force_contiguous = False
    return MemoryVolumeStore(
        data,
        spacing=spacing,
        value_range=value_range,
        force_contiguous=force_contiguous,
    )
