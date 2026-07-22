"""Unit tests for host volume path parsing — pure logic, no Qt/GPU required."""
import pytest

from inno3d.core.wafer_context import (
    parse_host_volume_path,
    HostVolumePathInfo,
)


class TestParseHostVolumePath:
    """Test parse_host_volume_path with synthetic (non-existent) paths."""

    def test_canonical_path(self):
        """Full canonical host path parses all fields."""
        path = r"D:\data\26_07_22\LOT001_F01\W01\Chip_5_3\FOV_P5\Input.tiff"
        info = parse_host_volume_path(path)

        assert info.date_folder == "26_07_22"
        assert info.lot_foup_id == "LOT001_F01"
        assert info.lot_id == "LOT001"
        assert info.foup_id == "F01"
        assert info.wafer_id == "W01"
        assert info.chip_col == 5
        assert info.chip_row == 3
        assert info.fov_index == 5
        assert info.fov_folder == "FOV_P5"
        assert "Input.tiff" in info.input_path
        assert info.results_dir.endswith("Results")

    def test_unix_path(self):
        """Forward-slash paths work too."""
        path = "/mnt/data/26_07_22/LOT001_F01/W01/C5_R3/P5/Input.tiff"
        info = parse_host_volume_path(path)

        assert info.chip_col == 5
        assert info.chip_row == 3
        assert info.fov_index == 5

    def test_empty_path(self):
        info = parse_host_volume_path("")
        assert info.parsed_ok is False
        assert info.raw_path == ""

    def test_none_path(self):
        info = parse_host_volume_path(None)
        assert info.parsed_ok is False

    def test_short_path(self):
        """Paths with fewer hierarchy levels still parse what they can."""
        path = r"C:\Chip_7_8\FOV_P3\Input.tiff"
        info = parse_host_volume_path(path)
        assert info.chip_col == 7
        assert info.chip_row == 8
        assert info.fov_index == 3

    def test_results_dir_is_sibling(self):
        """Results dir should be a sibling of the FOV folder."""
        path = r"D:\date\Lot_F\W\Chip_1_1\FOV_P1\Input.tiff"
        info = parse_host_volume_path(path)
        assert info.results_dir.replace("\\", "/").endswith("FOV_P1/Results")

    def test_breadcrumb_format(self):
        path = r"D:\data\26_07_22\LOT001_F01\W01\Chip_5_3\FOV_P5\Input.tiff"
        info = parse_host_volume_path(path)
        bc = info.breadcrumb()
        assert "26_07_22" in bc
        assert "LOT001_F01" in bc
        assert "W01" in bc
        assert "Chip (5,3)" in bc
        assert "FOV P5" in bc

    def test_breadcrumb_empty(self):
        info = parse_host_volume_path("")
        bc = info.breadcrumb()
        assert bc == "—" or bc == ""

    def test_returns_dataclass(self):
        info = parse_host_volume_path(r"D:\x\y\z\Chip_1_1\P1\in.tiff")
        assert isinstance(info, HostVolumePathInfo)

    def test_fov_point_classes(self):
        """FovPoint dataclass basic tests."""
        from inno3d.core.wafer_context import FovPoint, BIN_OK, BIN_NG, BIN_PENDING

        p = FovPoint(index=1, judge=BIN_OK)
        assert p.is_ok() is True
        assert p.is_ng() is False
        assert p.is_pending() is False

        p2 = FovPoint(index=2, judge=BIN_NG)
        assert p2.is_ng() is True

        p3 = FovPoint(index=3, judge=BIN_PENDING)
        assert p3.is_pending() is True

    def test_fov_point_dict_roundtrip(self):
        from inno3d.core.wafer_context import FovPoint

        p = FovPoint(index=5, x_offset_mm=0.25, y_offset_mm=-0.25, judge=8)
        d = p.to_dict()
        p2 = FovPoint.from_dict(d)
        assert p2.index == p.index
        assert p2.x_offset_mm == p.x_offset_mm
        assert p2.judge == p.judge
