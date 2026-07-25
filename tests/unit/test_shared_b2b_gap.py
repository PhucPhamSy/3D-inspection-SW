"""Unit tests for shared B2B gap helpers (Viewer + Teaching SoT)."""
from inno3d.features.shared.b2b_gap_3d import (
    B2BGapOverlay,
    format_gap_label,
    layer_z_offset,
    world_spacing,
)


def test_world_spacing_defaults():
    assert world_spacing(None, None) == (1.0, 1.0, 1.0)
    assert world_spacing([0.5, 0.5, 1.0], None) == (0.5, 0.5, 1.0)
    assert world_spacing(None, (2.0, 3.0, 4.0)) == (2.0, 3.0, 4.0)


def test_layer_z_offset_row_key():
    assert layer_z_offset({"z_start": 42}) == 42
    assert layer_z_offset({"Layer": "L2"}, fallback=7) == 7


def test_layer_z_offset_bands():
    bands = [{"name": "Layer 2", "z_start": 100}, {"name": "Layer_3", "z_start": 200}]
    assert layer_z_offset({"Layer": "Layer 2"}, layer_bands=bands) == 100
    assert layer_z_offset({"Layer": "Layer 3"}, layer_bands=bands) == 200


def test_format_gap_label_lines():
    s = format_gap_label("L1", 1.234, "R0C0", "R0C1", "Right")
    assert "1.23 µm" in s
    assert "Right" in s
    assert "SRC  R0C0" in s
    assert "DST  R0C1" in s
    assert s.count("\n") == 2


def test_overlay_empty_clear():
    ov = B2BGapOverlay()
    ov.clear(None, None, render=False)
    assert ov.actors == []
    assert ov.active_stats_idx is None
