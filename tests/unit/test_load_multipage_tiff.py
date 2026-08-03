"""Unit tests for fast multipage TIFF decode helper (Online load path)."""

import numpy as np
import pytest

pytest.importorskip("tifffile")

import tifffile

from inno3d.core.view_support import _load_multipage_tiff, _tiff_worker_counts


@pytest.fixture
def multipage_tiff(tmp_path):
    path = tmp_path / "stack.tif"
    pages = [np.full((8, 8), i, dtype=np.uint16) for i in range(6)]
    tifffile.imwrite(path, pages, compression=None, dtype=np.uint16)
    return path


class TestLoadMultipageTiff:
    def test_worker_counts_positive(self):
        decode, io = _tiff_worker_counts()
        assert decode >= 1
        assert io >= 1

    def test_loads_multipage_stack(self, multipage_tiff):
        vol = _load_multipage_tiff(multipage_tiff)
        assert vol.shape == (6, 8, 8)
        assert vol.dtype == np.uint16
        assert int(vol[3, 0, 0]) == 3

    def test_prealloc_matches_imread(self, multipage_tiff):
        expected = tifffile.imread(multipage_tiff)
        actual = _load_multipage_tiff(multipage_tiff)
        np.testing.assert_array_equal(actual, expected)

    def test_single_page_tiff_normalized_to_3d(self, tmp_path):
        path = tmp_path / "single.tif"
        tifffile.imwrite(path, np.zeros((4, 4), dtype=np.uint16))
        vol = _load_multipage_tiff(path)
        assert vol.shape == (1, 4, 4)
        assert vol.dtype == np.uint16
