"""Unit tests for shared 3D crosshair geometry / overlay helpers."""
import numpy as np

from inno3d.features.shared.crosshair_3d import (
    AXIS_COLORS,
    Crosshair3DOverlay,
    crosshair_axis_endpoints,
)


def test_axis_colors_rgb():
    for k in ("x", "y", "z"):
        assert k in AXIS_COLORS
        assert len(AXIS_COLORS[k]) == 3


def test_endpoints_center_axis_aligned():
    # 10x20x30 volume (Z,Y,X), unit spacing, center index
    shape = (10, 20, 30)
    pos = (15.0, 10.0, 5.0)  # x,y,z indices
    ends = crosshair_axis_endpoints(pos, shape, (1.0, 1.0, 1.0), R=None)
    pc = np.array([15.0, 10.0, 5.0])
    # X line should pass through center and be parallel to +X
    p1x, p2x = ends["x"]
    mid_x = 0.5 * (p1x + p2x)
    np.testing.assert_allclose(mid_x, pc, atol=1e-9)
    dir_x = p2x - p1x
    dir_x = dir_x / np.linalg.norm(dir_x)
    np.testing.assert_allclose(np.abs(dir_x), [1.0, 0.0, 0.0], atol=1e-9)

    p1y, p2y = ends["y"]
    dir_y = p2y - p1y
    dir_y = dir_y / np.linalg.norm(dir_y)
    np.testing.assert_allclose(np.abs(dir_y), [0.0, 1.0, 0.0], atol=1e-9)

    p1z, p2z = ends["z"]
    dir_z = p2z - p1z
    dir_z = dir_z / np.linalg.norm(dir_z)
    np.testing.assert_allclose(np.abs(dir_z), [0.0, 0.0, 1.0], atol=1e-9)


def test_endpoints_respect_spacing():
    shape = (5, 5, 5)
    pos = (2.0, 2.0, 2.0)
    ends = crosshair_axis_endpoints(pos, shape, (2.0, 3.0, 4.0), R=None)
    pc = np.array([2.0 * 2.0, 2.0 * 3.0, 2.0 * 4.0])
    mid = 0.5 * (ends["x"][0] + ends["x"][1])
    np.testing.assert_allclose(mid, pc, atol=1e-9)


def test_endpoints_with_rotation():
    # 90° around Z: X→Y, Y→-X
    R = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    ends = crosshair_axis_endpoints((1, 1, 1), (3, 3, 3), (1, 1, 1), R=R)
    p1x, p2x = ends["x"]
    dir_x = p2x - p1x
    dir_x = dir_x / np.linalg.norm(dir_x)
    # Column 0 of R is [0, 1, 0]
    np.testing.assert_allclose(dir_x, [0.0, 1.0, 0.0], atol=1e-9)


def test_overlay_disabled_clears_without_ren():
    ov = Crosshair3DOverlay()
    assert ov.update(None, None, True, (0, 0, 0), (2, 2, 2), (1, 1, 1)) is False
    assert ov.update(None, None, False, (0, 0, 0), (2, 2, 2), (1, 1, 1)) is False


def test_overlay_clear_noop():
    ov = Crosshair3DOverlay()
    ov.clear(None, None, render=False)
    assert ov.actors == []
