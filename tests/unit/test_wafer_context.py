"""Unit tests for inno3d.core.wafer_context — pure logic, no Qt/GPU required."""
import pytest
import numpy as np

from inno3d.core.wafer_context import (
    BIN_OUTSIDE, BIN_NG, BIN_PENDING, BIN_OK,
    die_inside_wafer,
    circular_bin_mask,
    _split_lot_foup,
    _parse_chip_location,
    _parse_fov_location,
)


# ─────────────────────────────────────────────
# Bin code constants
# ─────────────────────────────────────────────
class TestBinCodes:
    def test_bin_values(self):
        assert BIN_OUTSIDE == 0
        assert BIN_NG == 1
        assert BIN_PENDING == 2
        assert BIN_OK == 8

    def test_bin_distinct(self):
        codes = {BIN_OUTSIDE, BIN_NG, BIN_PENDING, BIN_OK}
        assert len(codes) == 4, "Bin codes must be distinct"


# ─────────────────────────────────────────────
# die_inside_wafer
# ─────────────────────────────────────────────
class TestDieInsideWafer:
    def test_center_is_inside(self):
        """Center die of any size map should be inside."""
        for n in (9, 14, 20):
            mid = n // 2
            assert die_inside_wafer(mid, mid, n) is True

    def test_corner_is_outside(self):
        """Corner dies should be outside the circular wafer."""
        assert die_inside_wafer(0, 0, 14) is False
        assert die_inside_wafer(13, 0, 14) is False
        assert die_inside_wafer(0, 13, 14) is False
        assert die_inside_wafer(13, 13, 14) is False

    def test_edge_midpoint(self):
        """Edge midpoints on a 14×14 map."""
        # col=7 (center column), row=0 (top edge) — normalized center:
        # cx=(7+0.5)/14-0.5 = 0.0357, cy=(0+0.5)/14-0.5 = -0.464
        # r² = 0.0013 + 0.2153 = 0.2166 < 0.24 → inside
        assert die_inside_wafer(7, 0, 14) is True
        # col=7, row=7 (center) — inside
        assert die_inside_wafer(7, 7, 14) is True
        # col=0, row=7 (left edge mid) — cx=-0.464, cy=0.036 → r²=0.217 < 0.24 → inside
        assert die_inside_wafer(0, 7, 14) is True

    def test_map_size_1(self):
        """Single die map — center is inside."""
        assert die_inside_wafer(0, 0, 1) is True

    def test_symmetry(self):
        """Map should be symmetric around center."""
        n = 14
        for r in range(n):
            for c in range(n):
                mirror_c = n - 1 - c
                mirror_r = n - 1 - r
                assert die_inside_wafer(c, r, n) == die_inside_wafer(mirror_c, mirror_r, n)


# ─────────────────────────────────────────────
# circular_bin_mask
# ─────────────────────────────────────────────
class TestCircularBinMask:
    def test_shape(self):
        mask = circular_bin_mask(14)
        assert mask.shape == (14, 14)
        assert mask.dtype == np.uint8

    def test_default_inside_bin(self):
        mask = circular_bin_mask(9)
        # Center should be PENDING
        assert mask[4, 4] == BIN_PENDING
        # Corner should be OUTSIDE
        assert mask[0, 0] == BIN_OUTSIDE

    def test_custom_inside_bin(self):
        mask = circular_bin_mask(9, inside_bin=BIN_OK)
        assert mask[4, 4] == BIN_OK

    def test_consistency_with_die_inside_wafer(self):
        """circular_bin_mask and die_inside_wafer must agree."""
        n = 14
        mask = circular_bin_mask(n)
        for r in range(n):
            for c in range(n):
                inside = die_inside_wafer(c, r, n)
                if inside:
                    assert mask[r, c] == BIN_PENDING
                else:
                    assert mask[r, c] == BIN_OUTSIDE


# ─────────────────────────────────────────────
# _split_lot_foup
# ─────────────────────────────────────────────
class TestSplitLotFoup:
    def test_normal_split(self):
        assert _split_lot_foup("LOT123_F01") == ("LOT123", "F01")

    def test_multi_underscore(self):
        """Splits on the LAST underscore."""
        assert _split_lot_foup("LOT_123_F01") == ("LOT_123", "F01")

    def test_no_underscore(self):
        assert _split_lot_foup("LOT123") == ("LOT123", "")

    def test_empty(self):
        assert _split_lot_foup("") == ("", "")
        assert _split_lot_foup(None) == ("", "")

    def test_whitespace(self):
        assert _split_lot_foup("  LOT_F  ") == ("LOT", "F")


# ─────────────────────────────────────────────
# _parse_chip_location
# ─────────────────────────────────────────────
class TestParseChipLocation:
    @pytest.mark.parametrize("name, expected", [
        ("Chip_17_16", (17, 16)),
        ("chip17_16", (17, 16)),
        ("17_16", (17, 16)),
        ("C17_R16", (17, 16)),
        ("X17Y16", (17, 16)),
        ("(17,16)", (17, 16)),
        ("17,16", (17, 16)),
        ("C5_R3", (5, 3)),
        ("Chip-5-3", (5, 3)),
    ])
    def test_various_formats(self, name, expected):
        assert _parse_chip_location(name) == expected

    def test_empty(self):
        assert _parse_chip_location("") == (0, 0)
        assert _parse_chip_location(None) == (0, 0)

    def test_no_numbers(self):
        assert _parse_chip_location("Chip") == (0, 0)


# ─────────────────────────────────────────────
# _parse_fov_location
# ─────────────────────────────────────────────
class TestParseFovLocation:
    @pytest.mark.parametrize("name, expected", [
        ("FOV_P5", 5),
        ("FOV_P05", 5),
        ("P5", 5),
        ("Point5", 5),
        ("FOV5", 5),
        ("Point_5", 5),
        ("P1", 1),
        ("P9", 9),
    ])
    def test_various_formats(self, name, expected):
        assert _parse_fov_location(name) == expected

    def test_out_of_range(self):
        assert _parse_fov_location("P0") == 0
        assert _parse_fov_location("P10") == 0

    def test_empty(self):
        assert _parse_fov_location("") == 0
        assert _parse_fov_location(None) == 0
