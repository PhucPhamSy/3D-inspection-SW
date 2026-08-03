"""Unit tests for shared MPR MES selection highlight (Viewer SoT)."""
import numpy as np

from inno3d.features.shared.mes_mpr_highlight import (
    blend_mes_selection_highlight,
    build_mes_selection_highlight_rgb,
    labeled_slice_for_orientation,
    selected_c1_labels,
)


def test_selected_c1_labels():
    assert selected_c1_labels([(1, 3), (1, 0), (2, 9), (1, 5)]) == [3, 5]
    assert selected_c1_labels(None) == []


def test_labeled_slice_orientation():
    lab = np.zeros((4, 5, 6), dtype=np.int32)
    lab[1, 2, 3] = 7
    ax = labeled_slice_for_orientation(lab, "axial", 1)
    # Axial display applies flipud (Y): storage y=2 → display row 2 when Y=5
    assert ax is not None and ax[2, 3] == 7
    # reverse_z: display slice 0 → storage Z-1; flipud moves y=0 → bottom row
    lab[-1, 0, 0] = 11
    ax_r = labeled_slice_for_orientation(lab, "axial", 0, reverse_z=True)
    assert ax_r is not None and ax_r[4, 0] == 11


def test_build_highlight_fill_and_outline():
    labeled = np.zeros((20, 20), dtype=np.int32)
    labeled[8:12, 8:12] = 4  # solid square
    hl = build_mes_selection_highlight_rgb(
        labeled, [(1, 4)], highlight_color=(0.0, 1.0, 1.0), outline_iterations=2
    )
    assert hl is not None
    assert hl.shape == (20, 20, 3)
    # Interior has cyan channel green/blue high
    assert hl[10, 10, 1] > 200 and hl[10, 10, 2] > 200
    # Outline exists outside the square
    ring = hl.copy()
    ring[8:12, 8:12] = 0
    assert np.any(ring == 255)


def test_blend_opacity():
    base = np.zeros((8, 8, 3), dtype=np.uint8)
    hl = np.zeros((8, 8, 3), dtype=np.uint8)
    hl[2:6, 2:6] = (0, 255, 255)
    out = blend_mes_selection_highlight(base, hl, opacity=0.7)
    # ~0.7 * 255 ≈ 178
    assert 150 <= int(out[4, 4, 1]) <= 200
    assert out[0, 0, 1] == 0
