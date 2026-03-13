# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for FrameTransformer._build_sensor_index_lists and transform_to_vec_quat."""

import pytest
import warp as wp
from isaaclab_newton.sensors.frame_transformer.frame_transformer import FrameTransformer

from isaaclab.utils.warp.math_ops import transform_to_vec_quat


class TestTransformToVecQuat:
    """Tests for the zero-copy view split utility."""

    def test_1d_pos_quat_split(self):
        """1D array: position is first 3 floats, quaternion is last 4."""
        t = wp.zeros(3, dtype=wp.transformf, device="cpu")
        pos, quat = transform_to_vec_quat(t)
        assert pos.shape == (3,)
        assert quat.shape == (3,)
        assert pos.dtype == wp.vec3f
        assert quat.dtype == wp.quatf

    def test_2d_pos_quat_split(self):
        """2D array: shapes are (N, M) with vec3f and quatf dtypes."""
        t = wp.zeros((2, 4), dtype=wp.transformf, device="cpu")
        pos, quat = transform_to_vec_quat(t)
        assert pos.shape == (2, 4)
        assert quat.shape == (2, 4)
        assert pos.dtype == wp.vec3f
        assert quat.dtype == wp.quatf

    def test_zero_copy_1d(self):
        """Writes through pos/quat views are reflected in the original transform array."""
        t = wp.zeros(1, dtype=wp.transformf, device="cpu")
        pos, quat = transform_to_vec_quat(t)
        # Write known values through the views
        pos.numpy()[0] = (1.0, 2.0, 3.0)
        quat.numpy()[0] = (0.0, 0.0, 0.0, 1.0)
        floats = t.view(wp.float32).numpy()
        assert list(floats[0]) == pytest.approx([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])

    def test_invalid_ndim_raises(self):
        """Passing a 0D or 4D array raises ValueError."""
        with pytest.raises(ValueError):
            transform_to_vec_quat(wp.zeros((), dtype=wp.transformf, device="cpu"))


def _call(
    source_indices,
    target_per_world,
    target_frame_body_names,
    shape_labels,
    world_origin_idx,
    num_envs,
):
    return FrameTransformer._build_sensor_index_lists(
        source_indices,
        target_per_world,
        target_frame_body_names,
        shape_labels,
        world_origin_idx,
        num_envs,
    )


class TestZeroTargets:
    def test_zero_targets_shapes_refs(self):
        """0 targets: shapes/refs contain only source entries."""
        names, tgt_per_tgt, shapes, refs = _call(
            source_indices=[10, 11],
            target_per_world=[],
            target_frame_body_names=[],
            shape_labels={},
            world_origin_idx=0,
            num_envs=2,
        )
        assert shapes == [10, 11]
        assert refs == [0, 0]
        assert names == []
        assert tgt_per_tgt == []


class TestSingleTarget:
    def test_one_env_one_target(self):
        """1 env, 1 target: [src, tgt] shapes, [world_orig, src] refs."""
        names, tgt_per_tgt, shapes, refs = _call(
            source_indices=[10],
            target_per_world=[[[20]]],
            target_frame_body_names=["hand"],
            shape_labels={},
            world_origin_idx=0,
            num_envs=1,
        )
        assert shapes == [10, 20]
        assert refs == [0, 10]
        assert names == ["hand"]

    def test_two_envs_two_targets(self):
        """2 envs, 2 targets: stride-2 interleaved layout."""
        names, tgt_per_tgt, shapes, refs = _call(
            source_indices=[10, 11],
            target_per_world=[[[20], [21]], [[30], [31]]],
            target_frame_body_names=["arm", "hand"],
            shape_labels={},
            world_origin_idx=0,
            num_envs=2,
        )
        assert shapes == [10, 20, 30, 11, 21, 31]
        assert refs == [0, 10, 10, 0, 11, 11]
        assert names == ["arm", "hand"]


class TestWildcardExpansion:
    def test_wildcard_two_bodies_per_env_indices(self):
        """Wildcard: 2 bodies per env → 2 expanded target entries, correct indices."""
        # shape_labels: site index → "{body_name}/{site_label}"
        shape_labels = {20: "FL_foot/label_0", 21: "FL_foot/label_0", 22: "FR_foot/label_0", 23: "FR_foot/label_0"}
        names, tgt_per_tgt, shapes, refs = _call(
            source_indices=[10, 11],
            target_per_world=[[[20, 22], [21, 23]]],
            target_frame_body_names=["foot"],
            shape_labels=shape_labels,
            world_origin_idx=0,
            num_envs=2,
        )
        assert shapes == [10, 20, 22, 11, 21, 23]
        assert refs == [0, 10, 10, 0, 11, 11]
        # tgt_per_tgt: 2 expanded targets × 2 envs
        assert tgt_per_tgt == [[20, 21], [22, 23]]

    def test_wildcard_uses_body_names_from_shape_labels(self):
        """Wildcard: body names derived from shape_labels when n_bodies > 1."""
        shape_labels = {20: "FL_foot/label_0", 21: "FL_foot/label_0", 22: "FR_foot/label_0", 23: "FR_foot/label_0"}
        names, tgt_per_tgt, shapes, refs = _call(
            source_indices=[10, 11],
            target_per_world=[[[20, 22], [21, 23]]],
            target_frame_body_names=["foot"],
            shape_labels=shape_labels,
            world_origin_idx=0,
            num_envs=2,
        )
        assert names == ["FL_foot", "FR_foot"]

    def test_wildcard_single_body_uses_config_name(self):
        """Single body match: config name is used regardless of shape_labels."""
        names, tgt_per_tgt, shapes, refs = _call(
            source_indices=[10, 11],
            target_per_world=[[[20], [21]]],
            target_frame_body_names=["foot"],
            shape_labels={},
            world_origin_idx=0,
            num_envs=2,
        )
        assert names == ["foot"]
