# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for site injection, validation, and sensor index building."""

import pytest
import warp as wp
from isaaclab_newton.physics.newton_manager import NewtonManager
from isaaclab_newton.sensors.frame_transformer.frame_transformer import FrameTransformer
from newton import ModelBuilder

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


@pytest.mark.parametrize("replicated", [False, True])
def test_sites_bind_once_to_clone_sources_or_explicit_builder(monkeypatch, replicated):
    main = ModelBuilder()
    source = ModelBuilder() if replicated else main
    for leg in ("FL", "FR", "RL", "RR"):
        source.add_body(label=f"Robot/{leg}_foot")
    monkeypatch.setattr(NewtonManager, "_cl_pending_sites", {})
    xform = wp.transform((1.0, 2.0, 3.0), wp.quat_identity())
    global_label = NewtonManager.cl_register_site(None, xform)
    local_label = NewtonManager.cl_register_site("Robot/.*_foot", xform)
    world_label = NewtonManager.cl_register_site(None, xform, per_world=True)
    assert NewtonManager.cl_register_site(None, xform, per_world=True) == world_label

    sources = {"Robot": source} if replicated else {}
    global_sites, body_sites, world_sites = NewtonManager._cl_inject_sites(main, sources)
    assert main.shape_body[global_sites[global_label]] == -1
    indices = body_sites[id(source)][local_label]
    assert [source.shape_body[index] for index in indices] == list(range(4))
    assert [source.shape_label[index] for index in indices] == [f"{name}/{local_label}" for name in source.body_label]
    assert world_sites == {world_label: xform}
    assert not NewtonManager._cl_pending_sites

    NewtonManager.cl_register_site("Robot/nonexistent", xform)
    with pytest.raises(ValueError, match="matched no builder bodies"):
        NewtonManager._cl_inject_sites(main, sources)


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
