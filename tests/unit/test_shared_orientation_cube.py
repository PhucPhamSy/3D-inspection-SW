"""Unit tests for shared interactive orientation cube helpers."""
import numpy as np

from inno3d.features.shared.orientation_cube import (
    FACE_TO_PRESET,
    apply_camera_face_preset,
    is_over_orientation_viewport,
    ray_hit_unit_cube_face,
    set_orientation_face_hover,
    theme_primary_rgb,
)


def test_face_presets_complete():
    for f in ("+X", "-X", "+Y", "-Y", "+Z", "-Z"):
        assert f in FACE_TO_PRESET


def test_ray_hit_front_face():
    # Ray from +X toward origin hits +X face
    face = ray_hit_unit_cube_face(np.array([3.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]))
    assert face == "+X"
    face_z = ray_hit_unit_cube_face(np.array([0.0, 0.0, 4.0]), np.array([0.0, 0.0, -1.0]))
    assert face_z == "+Z"


def test_viewport_hit():
    vp = (0.82, 0.0, 1.0, 0.18)
    assert is_over_orientation_viewport(900, 50, 1000, 800, vp) is True
    assert is_over_orientation_viewport(100, 400, 1000, 800, vp) is False


def test_theme_primary_rgb_tuple():
    rgb = theme_primary_rgb()
    assert len(rgb) == 3
    assert all(0.0 <= c <= 1.0 for c in rgb)


class _FakeCam:
    def __init__(self):
        self.pos = None
        self.fp = None
        self.up = None

    def SetFocalPoint(self, *a):
        self.fp = a

    def SetPosition(self, *a):
        self.pos = a

    def SetViewUp(self, *a):
        self.up = a


def test_apply_camera_face_preset():
    cam = _FakeCam()
    assert apply_camera_face_preset(cam, "+Z", center=(1, 2, 3), radius=10.0)
    assert cam.fp == (1.0, 2.0, 3.0)
    assert cam.pos[2] > cam.fp[2]
    assert cam.up == (0, 1, 0)


def test_set_hover_noop_without_actors():
    st = {
        "cube": None,
        "face_highlights": {},
        "face_text_props": {},
        "hover_face": None,
    }
    assert set_orientation_face_hover(st, "+X") is True
    assert set_orientation_face_hover(st, "+X") is False
