# tests/unit/test_volume_pyramid.py
# -----------------------------------------------------------------------
# Unit tests for inno3d/services/volume_pyramid.py — pure pyramid math.
# -----------------------------------------------------------------------
"""Tests for pyramid level shapes, brick grid helpers, and downsample factors."""

import pytest

from inno3d.services.volume_pyramid import (
    brick_bounds,
    brick_grid_shape,
    downsample_dim,
    level_downsample_factor,
    level_shape,
    num_pyramid_levels,
)


# ─── downsample_dim / level_shape ─────────────────────────────────────────────

class TestDownsampleDim:
    @pytest.mark.parametrize(
        "size, level, expected",
        [
            (100, 0, 100),
            (100, 1, 50),
            (100, 2, 25),
            (101, 1, 51),
            (1, 5, 1),
            (7, 2, 2),
        ],
    )
    def test_ceiling_halving(self, size, level, expected):
        assert downsample_dim(size, level) == expected

    def test_negative_level_raises(self):
        with pytest.raises(ValueError, match="level"):
            downsample_dim(10, -1)

    def test_zero_size_returns_zero(self):
        assert downsample_dim(0, 0) == 0


class TestLevelShape:
    def test_applies_per_axis(self):
        assert level_shape((100, 200, 300), 2) == (25, 50, 75)

    def test_level_zero_identity(self):
        shape = (64, 128, 256)
        assert level_shape(shape, 0) == shape


class TestNumPyramidLevels:
    def test_single_voxel_volume(self):
        assert num_pyramid_levels((1, 1, 1)) == 1

    def test_power_of_two_cube(self):
        # 8 -> 4 -> 2 -> 1  => levels 0..3
        assert num_pyramid_levels((8, 8, 8), min_dim=1) == 4

    def test_stops_when_all_axes_at_min_dim(self):
        assert num_pyramid_levels((10, 10, 10), min_dim=2) == 4

    def test_invalid_min_dim(self):
        with pytest.raises(ValueError, match="min_dim"):
            num_pyramid_levels((8, 8, 8), min_dim=0)

    def test_empty_shape(self):
        assert num_pyramid_levels((0, 8, 8)) == 0


# ─── brick grid ───────────────────────────────────────────────────────────────

class TestBrickGridShape:
    def test_exact_division(self):
        assert brick_grid_shape((64, 128, 256), 64) == (1, 2, 4)

    def test_partial_last_brick(self):
        assert brick_grid_shape((65, 65, 65), 64) == (2, 2, 2)

    def test_invalid_brick_size(self):
        with pytest.raises(ValueError, match="brick_size"):
            brick_grid_shape((10, 10, 10), 0)


class TestBrickBounds:
    def test_origin_brick(self):
        assert brick_bounds((0, 0, 0), 64, (100, 200, 300)) == (0, 0, 0, 64, 64, 64)

    def test_edge_brick_clipped(self):
        z0, y0, x0, z1, y1, x1 = brick_bounds((1, 2, 3), 64, (100, 200, 300))
        assert (z0, y0, x0) == (64, 128, 192)
        assert (z1, y1, x1) == (100, 192, 256)

    def test_invalid_brick_size(self):
        with pytest.raises(ValueError, match="brick_size"):
            brick_bounds((0, 0, 0), -1, (10, 10, 10))


class TestLevelDownsampleFactor:
    @pytest.mark.parametrize("level, factor", [(0, 1), (1, 2), (3, 8)])
    def test_power_of_two(self, level, factor):
        assert level_downsample_factor(level) == factor

    def test_negative_level_raises(self):
        with pytest.raises(ValueError, match="level"):
            level_downsample_factor(-1)
