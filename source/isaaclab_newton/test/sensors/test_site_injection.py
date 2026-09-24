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
    """Error paths of the zero-copy view split utility; values are covered by the frame-transformer tests."""

    def test_invalid_ndim_raises(self):
        """Passing a 4D array raises the documented ValueError rather than a Warp view error."""
        with pytest.raises(ValueError, match="ndim=4"):
            transform_to_vec_quat(wp.zeros((1, 1, 1, 1), dtype=wp.transformf, device="cpu"))

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
        NewtonManager._cl_pending_sites = {(None, False, tuple(xform)): ("ft_0", xform)}
        NewtonManager._cl_inject_sites_fallback()

        entry = NewtonManager._cl_site_index_map["ft_0"]
        global_idx, per_world = entry
        assert isinstance(global_idx, int)
        assert per_world is None
        assert len(NewtonManager._cl_pending_sites) == 0


class TestFallbackLocalWildcard:
    """Wildcard local site matching N bodies must produce (None, [[idx0..idxN-1]]) — one world."""

    def setup_method(self):
        NewtonManager.clear()
        NewtonManager._builder = MockBuilder(["Robot/FL_foot", "Robot/FR_foot", "Robot/RL_foot", "Robot/RR_foot"])

    def test_wildcard_entry_shape(self):
        xform = wp.transform()
        NewtonManager._cl_pending_sites = {("Robot/.*_foot", False, tuple(xform)): ("ft_0", xform)}
        NewtonManager._cl_inject_sites_fallback()

        entry = NewtonManager._cl_site_index_map["ft_0"]
        global_idx, per_world = entry
        assert global_idx is None
        assert len(per_world) == 1  # one world
        assert len(per_world[0]) == 4  # four bodies matched

    def test_no_match_raises(self):
        xform = wp.transform()
        NewtonManager._cl_pending_sites = {("Robot/nonexistent", False, tuple(xform)): ("ft_0", xform)}
        with pytest.raises(ValueError):
            NewtonManager._cl_inject_sites_fallback()


class TestWorldSite:
    """World-local sites are per-world, not global."""

    def setup_method(self):
        NewtonManager.clear()
        NewtonManager._builder = MockBuilder([])

    def test_world_site_reuses_label(self):
        xform = wp.transform((1.0, 2.0, 3.0), wp.quat_identity())
        label_0 = NewtonManager.cl_register_site(None, xform, per_world=True)
        label_1 = NewtonManager.cl_register_site(None, xform, per_world=True)

        assert label_0 == label_1

    def test_world_site_fallback_entry_is_local(self):
        xform = wp.transform((1.0, 2.0, 3.0), wp.quat_identity())
        label = NewtonManager.cl_register_site(None, xform, per_world=True)
        NewtonManager._cl_inject_sites_fallback()

        global_idx, per_world = NewtonManager._cl_site_index_map[label]
        assert global_idx is None
        assert isinstance(per_world, list)
        assert len(per_world) == 1
        assert len(per_world[0]) == 1

    def test_inject_sites_returns_world_sites(self):
        xform = wp.transform((1.0, 2.0, 3.0), wp.quat_identity())
        label = NewtonManager.cl_register_site(None, xform, per_world=True)
        global_sites, proto_sites, world_sites = NewtonManager._cl_inject_sites(MockBuilder([]), {})

        assert global_sites == {}
        assert proto_sites == {}
        assert world_sites[label] == xform
        assert NewtonManager._cl_pending_sites == {}


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
