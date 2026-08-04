"""Samsung indexing: (row, col)=(1,1) is visual top-left on axial MPR."""

from inno3d.core.bumpvoid_mes import reindex_grid_top_left


def _bump(label, cx, cy, *, layer="L1", half=2):
    return {
        "label": label,
        "layer_name": layer,
        "centroid_x": float(cx),
        "centroid_y": float(cy),
        "centroid_z": 0.0,
        "x_min": int(cx) - half,
        "x_max": int(cx) + half,
        "y_min": int(cy) - half,
        "y_max": int(cy) + half,
        "z_min": 0,
        "z_max": 1,
    }


def test_reindex_assigns_11_to_lowest_y_leftmost_x():
    """After axial flipud, low voxel-Y is top of screen → that bump is (1,1)."""
    # 2x2 grid in voxel coords (y increases downward on screen after flipud)
    stats = [
        _bump(1, cx=10, cy=10),  # top-left visually
        _bump(2, cx=30, cy=10),  # top-right
        _bump(3, cx=10, cy=40),  # bottom-left
        _bump(4, cx=30, cy=40),  # bottom-right
    ]
    out = reindex_grid_top_left(stats, voxel_x=1.0, voxel_y=1.0, per_layer=True)
    by_label = {s["label"]: s for s in out}

    assert by_label[1]["grid_row"] == 1 and by_label[1]["grid_col"] == 1
    assert by_label[2]["grid_row"] == 1 and by_label[2]["grid_col"] == 2
    assert by_label[3]["grid_row"] == 2 and by_label[3]["grid_col"] == 1
    assert by_label[4]["grid_row"] == 2 and by_label[4]["grid_col"] == 2
    assert by_label[1]["bump_id"] == "1,1"


def test_reindex_is_one_based_not_zero():
    out = reindex_grid_top_left([_bump(7, cx=5, cy=5)])
    assert out[0]["grid_row"] == 1
    assert out[0]["grid_col"] == 1


def test_reindex_per_layer_each_has_own_11():
    stats = [
        _bump(1, cx=10, cy=10, layer="A"),
        _bump(2, cx=10, cy=10, layer="B"),
    ]
    out = reindex_grid_top_left(stats, per_layer=True)
    assert all(s["grid_row"] == 1 and s["grid_col"] == 1 for s in out)
