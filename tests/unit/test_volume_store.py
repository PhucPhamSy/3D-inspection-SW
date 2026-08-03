# tests/unit/test_volume_store.py
# -----------------------------------------------------------------------
# Unit tests for inno3d/core/volume_store.py — VolumeStore public API contract.
# -----------------------------------------------------------------------
"""Tests for VolumeStore metadata, slice/ROI/brick access, and no __array__."""

import pytest
import numpy as np

pytest.importorskip("numpy")

from inno3d.core.volume_store import (
    ORIENTATIONS,
    MemoryVolumeStore,
    VolumeMetadata,
    VolumeStore,
    choose_3d_mip_level,
    default_3d_upload_budget_bytes,
    level_nbytes,
    store_from_dense_array,
)


@pytest.fixture
def sample_volume():
    """Deterministic 4×5×6 volume (Z, Y, X) for slice/orientation checks."""
    z, y, x = 4, 5, 6
    data = np.arange(z * y * x, dtype=np.uint16).reshape(z, y, x)
    return data


@pytest.fixture
def store(sample_volume):
    return MemoryVolumeStore(sample_volume, spacing=(2.0, 1.5, 1.0))


class TestVolumeStoreContract:
    """Public API surface Grok should preserve."""

    def test_orientations_tuple(self):
        assert ORIENTATIONS == ("axial", "coronal", "sagittal")

    def test_volume_store_is_abstract(self):
        with pytest.raises(TypeError):
            VolumeStore()  # type: ignore[abstract]

    def test_no_implicit_array_conversion(self, store):
        with pytest.raises(TypeError, match="must not be converted"):
            np.asarray(store)

    def test_metadata_shape_dtype_spacing(self, store, sample_volume):
        meta = store.metadata
        assert isinstance(meta, VolumeMetadata)
        assert meta.shape == sample_volume.shape
        assert meta.dtype == sample_volume.dtype
        assert store.shape == sample_volume.shape
        assert store.dtype == sample_volume.dtype
        assert store.spacing == (2.0, 1.5, 1.0)
        assert meta.spacing == (2.0, 1.5, 1.0)

    def test_value_range_from_data(self, store):
        meta = store.metadata
        assert meta.value_range == (0.0, float(store.shape[0] * store.shape[1] * store.shape[2] - 1))


class TestGetSlice:
    def test_axial_slice_shape(self, store):
        sl = store.get_slice("axial", 0)
        assert sl.shape == (store.shape[1], store.shape[2])
        assert sl.dtype == store.dtype

    def test_coronal_slice_matches_flipud(self, store, sample_volume):
        idx = 2
        sl = store.get_slice("coronal", idx)
        expected = np.flipud(sample_volume[:, idx, :])
        np.testing.assert_array_equal(sl, expected)

    def test_sagittal_slice_matches_transpose(self, store, sample_volume):
        idx = 3
        sl = store.get_slice("sagittal", idx)
        expected = np.transpose(sample_volume[:, :, idx])
        np.testing.assert_array_equal(sl, expected)

    def test_axial_index_out_of_range(self, store):
        with pytest.raises(IndexError):
            store.get_slice("axial", store.shape[0])

    def test_invalid_orientation(self, store):
        with pytest.raises(ValueError, match="unknown orientation"):
            store.get_slice("oblique", 0)  # type: ignore[arg-type]

    def test_level_one_downsampled_slice(self, store):
        full = store.get_slice("axial", 0, level=0)
        coarse = store.get_slice("axial", 0, level=1)
        assert coarse.shape == (full.shape[0] // 2 + full.shape[0] % 2,
                                 full.shape[1] // 2 + full.shape[1] % 2)


class TestGetRoi:
    def test_roi_subvolume(self, store, sample_volume):
        roi = store.get_roi((slice(1, 3), slice(0, 2), slice(2, 5)))
        expected = sample_volume[1:3, 0:2, 2:5]
        np.testing.assert_array_equal(roi, expected)

    def test_roi_at_coarse_level(self, store):
        roi = store.get_roi((slice(0, 2), slice(0, 2), slice(0, 2)), level=1)
        assert roi.ndim == 3
        assert all(d <= 2 for d in roi.shape)


class TestGetBricks:
    def test_single_brick_payload(self, store):
        bricks = store.get_bricks([(0, 0, 0)], brick_size=2)
        assert (0, 0, 0) in bricks
        brick = bricks[(0, 0, 0)]
        assert brick.shape == (2, 2, 2)

    def test_edge_brick_clipped_to_volume(self, store):
        z, y, x = store.shape
        grid = ((z + 1) // 2, (y + 1) // 2, (x + 1) // 2)
        coord = (grid[0] - 1, grid[1] - 1, grid[2] - 1)
        bricks = store.get_bricks([coord], brick_size=2)
        brick = bricks[coord]
        assert brick.size > 0
        assert all(s <= 2 for s in brick.shape)


class TestMemoryVolumeStoreInit:
    def test_rejects_non_3d(self):
        with pytest.raises(ValueError, match="3-D"):
            MemoryVolumeStore(np.zeros((2, 3)))

    def test_negative_level_rejected_via_slice(self, store):
        with pytest.raises(ValueError, match="level"):
            store.get_slice("axial", 0, level=-1)


class TestChoose3dMipLevel:
    def test_small_volume_stays_level_zero(self):
        level = choose_3d_mip_level((64, 64, 64), np.uint16, budget_bytes=512 * 1024 ** 2)
        assert level == 0

    def test_large_volume_picks_coarse_level(self):
        shape = (1400, 2000, 2000)
        assert level_nbytes(shape, np.uint16, 0) > 8 * (1024 ** 3)
        level = choose_3d_mip_level(shape, np.uint16, budget_bytes=512 * 1024 ** 2)
        assert level >= 2
        assert level_nbytes(shape, np.uint16, level) <= 512 * 1024 ** 2

    def test_1200_cube_picks_level_one_under_512mib(self):
        shape = (1200, 1200, 1200)
        assert level_nbytes(shape, np.uint16, 0) > 512 * 1024 ** 2
        level = choose_3d_mip_level(shape, np.uint16, budget_bytes=512 * 1024 ** 2)
        assert level == 1
        assert level_nbytes(shape, np.uint16, level) <= 512 * 1024 ** 2

    def test_respects_availability_callback(self):
        shape = (1024, 1024, 1024)
        level = choose_3d_mip_level(
            shape,
            np.uint16,
            budget_bytes=64 * 1024 ** 2,
            is_available=lambda lv: lv == 3,
        )
        assert level == 3

    def test_default_budget_env_helper(self, monkeypatch):
        monkeypatch.setenv("INNO3D_3D_UPLOAD_BUDGET_MB", "256")
        assert default_3d_upload_budget_bytes() == 256 * 1024 ** 2

    def test_memory_store_best_available_with_budget(self, store):
        preferred = choose_3d_mip_level(
            store.shape, store.dtype, budget_bytes=16, is_available=store._level_available
        )
        chosen = store.best_available_level(preferred)
        arr = np.ascontiguousarray(store.get_level(chosen))
        assert arr.ndim == 3
        assert chosen >= preferred or chosen == preferred
        assert arr.shape == store.get_level_shape(chosen)


class TestDenseStoreWrapOptimizations:
    def test_prefer_no_copy_reuses_contiguous_array(self, sample_volume):
        store = store_from_dense_array(sample_volume, prefer_no_copy=True)
        assert store._data is sample_volume

    def test_prefer_no_copy_copies_non_contiguous(self):
        base = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
        view = base.transpose(1, 0, 2)
        store = store_from_dense_array(view, prefer_no_copy=True)
        assert store._data is not view
        np.testing.assert_array_equal(store._data, view)

    def test_large_volume_uses_sampled_value_range(self):
        data = np.array([0, 100, 200], dtype=np.uint16).reshape(1, 1, 3)
        store = store_from_dense_array(data, sampled_value_range=True)
        assert store.metadata.value_range == (0.0, 200.0)

    def test_store_from_dense_array_copies_noncontiguous_even_with_prefer_no_copy(self):
        base = np.arange(4 * 6 * 8, dtype=np.uint16).reshape(4, 6, 8)
        noncontig = base[:, ::2, :]
        assert noncontig.flags.c_contiguous is False
        store = store_from_dense_array(
            noncontig,
            prefer_no_copy=True,
            sampled_value_range=True,
            max_samples=64,
        )
        assert store.get_level(0) is not noncontig
        np.testing.assert_array_equal(store.get_level(0), noncontig)
        assert store.metadata.value_range[0] <= store.metadata.value_range[1]
