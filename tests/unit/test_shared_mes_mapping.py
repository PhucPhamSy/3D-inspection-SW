"""Unit tests for shared MES mapping helpers."""
import numpy as np

from inno3d.features.shared.mes_mapping import (
    MesPickDebounce,
    centroid_nav_xyz,
    highlights_from_stats_indices,
    label_at_stat_centroid,
    label_at_volume_xyz,
    label_set_from_highlights,
    mes_bbox_for_labels,
    resolve_stat_label,
    stats_index_for_label,
)


def test_label_at_volume_xyz_direct_and_neighbor():
    labeled = np.zeros((5, 5, 5), dtype=np.int32)
    labeled[2, 2, 2] = 7
    assert label_at_volume_xyz(labeled, 2, 2, 2) == 7
    # near miss
    assert label_at_volume_xyz(labeled, 3, 2, 2, search_rad=2) == 7
    assert label_at_volume_xyz(labeled, 0, 0, 0, search_rad=1) == 0
    assert label_at_volume_xyz(None, 0, 0, 0) == 0


def test_stats_index_and_resolve_label():
    labeled = np.zeros((4, 4, 4), dtype=np.int32)
    labeled[1, 1, 1] = 3
    stats = [
        {"label": 1, "centroid_z": 0, "centroid_y": 0, "centroid_x": 0},
        {
            "label": 0,
            "centroid_z": 1,
            "centroid_y": 1,
            "centroid_x": 1,
            "z_min": 1,
            "z_max": 1,
            "y_min": 1,
            "y_max": 1,
            "x_min": 1,
            "x_max": 1,
        },
    ]
    assert stats_index_for_label(stats, 1) == 0
    assert stats_index_for_label(stats, 3, labeled=labeled) == 1
    assert resolve_stat_label(stats[1], labeled=labeled) == 3
    assert stats[1]["label"] == 3


def test_centroid_nav_and_bbox():
    st = {
        "label": 2,
        "centroid_z": 3.4,
        "centroid_y": 5.6,
        "centroid_x": 7.8,
        "z_min": 2,
        "z_max": 4,
        "y_min": 5,
        "y_max": 6,
        "x_min": 7,
        "x_max": 8,
    }
    assert centroid_nav_xyz(st, z_offset=10) == (8, 6, 13)
    bb = mes_bbox_for_labels([st], {2}, z_offset=10)
    assert bb == (12, 14, 5, 6, 7, 8)


def test_highlights_and_label_set():
    stats = [{"label": 1}, {"label": 0}, {"label": 5}]
    hl = highlights_from_stats_indices(stats, [0, 2])
    assert hl == [(1, 1), (1, 5)]
    assert label_set_from_highlights(hl) == {1, 5}


def test_debounce():
    d = MesPickDebounce(0.5)
    assert d.is_duplicate(9) is False
    assert d.is_duplicate(9) is True
    assert d.is_duplicate(10) is False
