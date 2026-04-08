# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for site injection, validation, and sensor index building."""

import pytest
import warp as wp
from isaaclab_newton.physics.newton_manager import NewtonManager
from isaaclab_newton.sensors.frame_transformer.frame_transformer import FrameTransformer

from isaaclab.utils.warp.math_ops import transform_to_vec_quat

# ---------------------------------------------------------------------------
# transform_to_vec_quat
# ---------------------------------------------------------------------------


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
        """Passing a 0D or 4D array raises an error."""
        with pytest.raises((ValueError, IndexError)):
            transform_to_vec_quat(wp.zeros((), dtype=wp.transformf, device="cpu"))

    def test_wrong_dtype_raises(self):
        """Passing wrong dtype raises TypeError."""
        with pytest.raises(TypeError):
            transform_to_vec_quat(wp.zeros(3, dtype=wp.vec3f, device="cpu"))


# ---------------------------------------------------------------------------
# NewtonManager._cl_inject_sites_fallback
# ---------------------------------------------------------------------------


class MockBuilder:
    """Minimal stand-in for ModelBuilder."""

    def __init__(self, body_labels: list[str]):
        self.body_label = body_labels
        self._next_idx = 0

    def add_site(self, body: int, xform: wp.transform, label: str) -> int:
        idx = self._next_idx
        self._next_idx += 1
        return idx


class TestFallbackGlobalSite:
    """Global site (body_pattern=None) must produce a (int, None) entry."""

    def setup_method(self):
        NewtonManager.clear()
        NewtonManager._builder = MockBuilder(["body0", "body1"])

    def test_global_site_entry_is_int_none_tuple(self):
        xform = wp.transform()
        NewtonManager._cl_pending_sites = {(None, tuple(xform)): ("ft_0", xform)}
        NewtonManager._cl_inject_sites_fallback()

        entry = NewtonManager._cl_site_index_map["ft_0"]
        global_idx, per_world = entry
        assert isinstance(global_idx, int)
        assert per_world is None

    def test_global_site_pending_cleared(self):
        xform = wp.transform()
        NewtonManager._cl_pending_sites = {(None, tuple(xform)): ("ft_0", xform)}
        NewtonManager._cl_inject_sites_fallback()

        assert len(NewtonManager._cl_pending_sites) == 0


class TestFallbackLocalSingleBody:
    """Single-body local site must produce a (None, [[idx]]) entry — one world."""

    def setup_method(self):
        NewtonManager.clear()
        NewtonManager._builder = MockBuilder(["Robot/base", "Robot/hand"])

    def test_single_body_entry_shape(self):
        xform = wp.transform()
        NewtonManager._cl_pending_sites = {("Robot/base", tuple(xform)): ("ft_0", xform)}
        NewtonManager._cl_inject_sites_fallback()

        entry = NewtonManager._cl_site_index_map["ft_0"]
        global_idx, per_world = entry
        assert global_idx is None
        assert isinstance(per_world, list)
        assert len(per_world) == 1  # one world
        assert len(per_world[0]) == 1  # one match
        assert isinstance(per_world[0][0], int)


class TestFallbackLocalWildcard:
    """Wildcard local site matching N bodies must produce (None, [[idx0..idxN-1]]) — one world."""

    def setup_method(self):
        NewtonManager.clear()
        NewtonManager._builder = MockBuilder(["Robot/FL_foot", "Robot/FR_foot", "Robot/RL_foot", "Robot/RR_foot"])

    def test_wildcard_entry_shape(self):
        xform = wp.transform()
        NewtonManager._cl_pending_sites = {("Robot/.*_foot", tuple(xform)): ("ft_0", xform)}
        NewtonManager._cl_inject_sites_fallback()

        entry = NewtonManager._cl_site_index_map["ft_0"]
        global_idx, per_world = entry
        assert global_idx is None
        assert len(per_world) == 1  # one world
        assert len(per_world[0]) == 4  # four bodies matched

    def test_no_match_raises(self):
        xform = wp.transform()
        NewtonManager._cl_pending_sites = {("Robot/nonexistent", tuple(xform)): ("ft_0", xform)}
        with pytest.raises(ValueError):
            NewtonManager._cl_inject_sites_fallback()


# ---------------------------------------------------------------------------
# FrameTransformer._validate_site_map
# ---------------------------------------------------------------------------


def _make_site_map(
    source_per_world: list[list[int]],
    target_per_worlds: list[list[list[int]]],
    world_origin_idx: int = 0,
) -> dict:
    m = {
        "world_origin": (world_origin_idx, None),
        "source": (None, source_per_world),
    }
    for i, pw in enumerate(target_per_worlds):
        m[f"target_{i}"] = (None, pw)
    return m


class TestSourceValidation:
    def test_valid_source_one_per_env(self):
        site_map = _make_site_map([[10], [20]], [])
        indices, _ = FrameTransformer._validate_site_map("source", "/Robot/base", [], [], site_map, num_envs=2)
        assert indices == [10, 20]

    def test_source_wrong_env_count_raises(self):
        # site map has 1 world entry but num_envs=2
        site_map = _make_site_map([[10]], [])
        with pytest.raises(ValueError, match="1 world entries.*expected 2"):
            FrameTransformer._validate_site_map("source", "/Robot/base", [], [], site_map, num_envs=2)

    def test_source_zero_in_env_raises(self):
        site_map = _make_site_map([[], [20]], [])
        with pytest.raises(ValueError, match="matched 0 bodies in env 0"):
            FrameTransformer._validate_site_map("source", "/Robot/base", [], [], site_map, num_envs=2)

    def test_source_two_in_env_raises(self):
        site_map = _make_site_map([[10, 11], [20]], [])
        with pytest.raises(ValueError, match="matched 2 bodies in env 0"):
            FrameTransformer._validate_site_map("source", "/Robot/base", [], [], site_map, num_envs=2)


class TestTargetValidation:
    def test_valid_single_target_per_env(self):
        site_map = _make_site_map([[10], [20]], [[[30], [40]]])
        _, tgt = FrameTransformer._validate_site_map(
            "source", "/Robot/base", ["target_0"], ["/Robot/hand"], site_map, num_envs=2
        )
        assert tgt[0] == [[30], [40]]

    def test_valid_wildcard_two_bodies_per_env(self):
        site_map = _make_site_map([[10], [20]], [[[30, 31], [40, 41]]])
        _, tgt = FrameTransformer._validate_site_map(
            "source", "/Robot/base", ["target_0"], ["/Robot/foot.*"], site_map, num_envs=2
        )
        assert tgt[0] == [[30, 31], [40, 41]]

    def test_target_zero_bodies_raises(self):
        site_map = _make_site_map([[10], [20]], [[[], []]])
        with pytest.raises(ValueError, match="matched no bodies"):
            FrameTransformer._validate_site_map(
                "source", "/Robot/base", ["target_0"], ["/Robot/foot.*"], site_map, num_envs=2
            )

    def test_target_non_uniform_raises(self):
        site_map = _make_site_map([[10], [20]], [[[30, 31], [40]]])
        with pytest.raises(ValueError, match="different numbers of bodies"):
            FrameTransformer._validate_site_map(
                "source", "/Robot/base", ["target_0"], ["/Robot/foot.*"], site_map, num_envs=2
            )


# ---------------------------------------------------------------------------
# FrameTransformer._build_sensor_index_lists
# ---------------------------------------------------------------------------


def _call(source_indices, target_per_world, target_frame_body_names, shape_labels, world_origin_idx, num_envs):
    return FrameTransformer._build_sensor_index_lists(
        source_indices, target_per_world, target_frame_body_names, shape_labels, world_origin_idx, num_envs
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
