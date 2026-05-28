# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Pure-Python unit tests for the coupled-solver manager partitioning logic.

These tests intentionally do NOT launch :class:`isaaclab.app.AppLauncher`.
:class:`NewtonCoupledSolverManager`'s body/joint/shape partitioning and
proxy-body resolution are static class methods that operate on a Newton
:class:`newton.Model` and an :class:`isaaclab.scene.InteractiveSceneCfg`, so
they can be tested against minimal fakes without spinning up Isaac Sim.

For the end-to-end smoke (solver builds and steps), see the existing env
smoke tests on ``Isaac-Lift-Soft-Franka-v0`` in
``isaaclab_tasks/test/test_environments.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
from newton import ShapeFlags

from isaaclab.managers import SceneEntityCfg

from isaaclab_contrib.coupling.coupled_manager import NewtonCoupledSolverManager

##
# Fakes
##


@dataclass
class _FakeArray:
    """Minimal stand-in for a Newton/warp array exposing ``.numpy()``."""

    data: np.ndarray

    def numpy(self) -> np.ndarray:
        return self.data


@dataclass
class _FakeModel:
    """Minimal stand-in for :class:`newton.Model`.

    Only the attributes consulted by the partitioning/selection helpers are
    populated. Joint/shape arrays default to zero-length numpy arrays when
    there are no joints/shapes to model.
    """

    body_count: int
    body_label: list[str]
    joint_count: int = 0
    joint_child: _FakeArray = field(default_factory=lambda: _FakeArray(np.zeros(0, dtype=np.int32)))
    shape_count: int = 0
    shape_body: _FakeArray = field(default_factory=lambda: _FakeArray(np.zeros(0, dtype=np.int32)))
    shape_flags: _FakeArray = field(default_factory=lambda: _FakeArray(np.zeros(0, dtype=np.int32)))


@dataclass
class _FakeAsset:
    """Stand-in for a scene asset cfg with the ``prim_path`` field consulted by the manager."""

    prim_path: str


@dataclass
class _FakeSceneCfg:
    """Stand-in for :class:`InteractiveSceneCfg`. Asset attributes are looked up by ``getattr``."""

    robot: _FakeAsset | None = None
    other: _FakeAsset | None = None


##
# Helpers
##


def _model_with_two_bodies(
    *,
    with_shapes: bool = False,
    with_joints: bool = False,
    extra_static_shape: bool = False,
) -> _FakeModel:
    """Build a 2-body Franka-like model under ``/World/envs/env_0/Robot``.

    Args:
        with_shapes: Attach one COLLIDE_SHAPES shape to each body.
        with_joints: Attach one joint whose child is body 1.
        extra_static_shape: Add an extra shape with ``body == -1``.
    """
    body_count = 2
    body_label = [
        "/World/envs/env_0/Robot/panda_link0",
        "/World/envs/env_0/Robot/panda_hand",
    ]

    shape_body = np.zeros(0, dtype=np.int32)
    shape_flags = np.zeros(0, dtype=np.int32)
    shape_count = 0
    if with_shapes:
        owners = [0, 1]
        if extra_static_shape:
            owners.append(-1)
        shape_body = np.asarray(owners, dtype=np.int32)
        shape_flags = np.full(len(owners), int(ShapeFlags.COLLIDE_SHAPES), dtype=np.int32)
        shape_count = len(owners)

    joint_child = np.zeros(0, dtype=np.int32)
    joint_count = 0
    if with_joints:
        joint_child = np.asarray([1], dtype=np.int32)
        joint_count = 1

    return _FakeModel(
        body_count=body_count,
        body_label=body_label,
        joint_count=joint_count,
        joint_child=_FakeArray(joint_child),
        shape_count=shape_count,
        shape_body=_FakeArray(shape_body),
        shape_flags=_FakeArray(shape_flags),
    )


def _robot_scene() -> _FakeSceneCfg:
    """A scene cfg with a single ``robot`` asset at the conventional prim path."""
    return _FakeSceneCfg(robot=_FakeAsset(prim_path="/World/envs/env_.*/Robot"))


##
# _resolve_entity_to_body_ids
##


def test_resolve_entity_no_body_names_returns_all_under_asset():
    model = _model_with_two_bodies()
    body_ids = NewtonCoupledSolverManager._resolve_entity_to_body_ids(
        model,
        SceneEntityCfg("robot"),
        _robot_scene(),
        field="src_bodies",
    )
    assert body_ids == [0, 1]


def test_resolve_entity_body_names_filter_by_regex():
    """Patterns full-match against the short name (segment after last ``/``)."""
    model = _model_with_two_bodies()
    body_ids = NewtonCoupledSolverManager._resolve_entity_to_body_ids(
        model,
        SceneEntityCfg("robot", body_names=["panda_hand"]),
        _robot_scene(),
        field="proxy_bodies",
    )
    assert body_ids == [1]


def test_resolve_entity_asset_missing_on_scene_cfg_raises():
    model = _model_with_two_bodies()
    with pytest.raises(ValueError, match="not on the attached scene cfg"):
        NewtonCoupledSolverManager._resolve_entity_to_body_ids(
            model,
            SceneEntityCfg("missing_asset"),
            _robot_scene(),
            field="src_bodies",
        )


def test_resolve_entity_unmatched_body_names_raises():
    model = _model_with_two_bodies()
    with pytest.raises(ValueError, match="no bodies matching"):
        NewtonCoupledSolverManager._resolve_entity_to_body_ids(
            model,
            SceneEntityCfg("robot", body_names=["nonexistent_link"]),
            _robot_scene(),
            field="proxy_bodies",
        )


##
# _partition_model_by_entities
##


def test_partition_splits_bodies_joints_shapes():
    """Bodies, joints, and shapes are split by their assigned partition."""
    model = _model_with_two_bodies(with_shapes=True, with_joints=True, extra_static_shape=True)
    scene = _FakeSceneCfg(
        robot=_FakeAsset(prim_path="/World/envs/env_.*/Robot"),
        other=_FakeAsset(prim_path="/World/envs/env_.*/Robot"),
    )

    src_b, dst_b, src_j, dst_j, src_s, dst_s = NewtonCoupledSolverManager._partition_model_by_entities(
        model,
        src_bodies=[SceneEntityCfg("robot", body_names=["panda_link0"])],
        dst_bodies=[SceneEntityCfg("other", body_names=["panda_hand"])],
        scene_cfg=scene,
    )

    assert src_b == [0]
    assert dst_b == [1]
    # Joint 0's child is body 1 (dst partition) → the joint index lands in dst.
    assert src_j == []
    assert dst_j == [0]
    # Shape 0 → body 0 (src). Shape 1 → body 1 (dst). Shape 2 → body -1 (static, → dst).
    assert src_s == [0]
    assert dst_s == [1, 2]


def test_partition_overlapping_bodies_raises():
    model = _model_with_two_bodies()
    scene = _FakeSceneCfg(
        robot=_FakeAsset(prim_path="/World/envs/env_.*/Robot"),
        other=_FakeAsset(prim_path="/World/envs/env_.*/Robot"),
    )
    with pytest.raises(ValueError, match="match both"):
        NewtonCoupledSolverManager._partition_model_by_entities(
            model,
            src_bodies=[SceneEntityCfg("robot")],
            dst_bodies=[SceneEntityCfg("other", body_names=["panda_hand"])],
            scene_cfg=scene,
        )


def test_partition_unclaimed_bodies_raises():
    model = _model_with_two_bodies()
    with pytest.raises(ValueError, match="unclaimed"):
        NewtonCoupledSolverManager._partition_model_by_entities(
            model,
            src_bodies=[SceneEntityCfg("robot", body_names=["panda_link0"])],
            dst_bodies=[],
            scene_cfg=_robot_scene(),
        )


##
# _select_proxy_bodies
##


def test_select_proxy_bodies_filters_to_collide_shapes():
    """Only bodies with at least one ``COLLIDE_SHAPES``-flagged shape become proxies."""
    model = _model_with_two_bodies(with_shapes=True)
    # Knock out body 0's collide flag — only body 1 should remain.
    model.shape_flags = _FakeArray(np.asarray([0, int(ShapeFlags.COLLIDE_SHAPES)], dtype=np.int32))

    proxy_ids = NewtonCoupledSolverManager._select_proxy_bodies(
        model,
        proxy_bodies=[SceneEntityCfg("robot", body_names=["panda_link0", "panda_hand"])],
        scene_cfg=_robot_scene(),
    )
    assert proxy_ids == [1]


def test_select_proxy_bodies_requires_body_names():
    """For ``SceneEntityCfg`` entries — proxies must be a subset, not the whole asset."""
    model = _model_with_two_bodies(with_shapes=True)
    with pytest.raises(ValueError, match="requires `body_names`"):
        NewtonCoupledSolverManager._select_proxy_bodies(
            model,
            proxy_bodies=[SceneEntityCfg("robot")],
            scene_cfg=_robot_scene(),
        )


def test_select_proxy_bodies_empty_input_returns_empty():
    """No ``proxy_bodies`` entries → no proxies (short-circuit before model lookups)."""
    proxy_ids = NewtonCoupledSolverManager._select_proxy_bodies(
        model=_FakeModel(body_count=0, body_label=[]),
        proxy_bodies=[],
        scene_cfg=None,
    )
    assert proxy_ids == []


def test_select_proxy_bodies_deduplicates_across_entries():
    """Multiple entries matching the same body produce a single proxy entry."""
    model = _model_with_two_bodies(with_shapes=True)
    proxy_ids = NewtonCoupledSolverManager._select_proxy_bodies(
        model,
        proxy_bodies=[
            SceneEntityCfg("robot", body_names=["panda_hand"]),
            SceneEntityCfg("robot", body_names=["panda_hand"]),
        ],
        scene_cfg=_robot_scene(),
    )
    assert proxy_ids == [1]


##
# Raw prim-path strings as selectors
##


def test_resolve_string_prefix_claims_all_bodies_under_path():
    """A raw prim-path string claims every body whose label matches ``^<string>(/|$)``."""
    model = _model_with_two_bodies()
    # Asset-prefix regex over the whole robot.
    body_ids = NewtonCoupledSolverManager._resolve_entity_to_body_ids(
        model,
        spec="/World/envs/env_.*/Robot",
        scene_cfg=None,
        field="src_bodies",
    )
    assert body_ids == [0, 1]


def test_resolve_string_narrows_to_a_single_body():
    """A specific prim-path string claims only the body with that exact label."""
    model = _model_with_two_bodies()
    body_ids = NewtonCoupledSolverManager._resolve_entity_to_body_ids(
        model,
        spec="/World/envs/env_.*/Robot/panda_hand",
        scene_cfg=None,
        field="proxy_bodies",
    )
    assert body_ids == [1]


def test_resolve_string_no_matches_raises():
    """A raw prim-path string with zero matches is treated as a typo."""
    model = _model_with_two_bodies()
    with pytest.raises(ValueError, match="matched no bodies"):
        NewtonCoupledSolverManager._resolve_entity_to_body_ids(
            model,
            spec="/World/envs/env_.*/Nonexistent",
            scene_cfg=None,
            field="src_bodies",
        )


def test_partition_accepts_mixed_string_and_scene_entity():
    """``src_bodies`` / ``dst_bodies`` accept a mix of strings and ``SceneEntityCfg``."""
    model = _model_with_two_bodies(with_shapes=True, with_joints=True)
    src_b, dst_b, _, _, _, _ = NewtonCoupledSolverManager._partition_model_by_entities(
        model,
        src_bodies=[SceneEntityCfg("robot", body_names=["panda_link0"])],
        dst_bodies=["/World/envs/env_.*/Robot/panda_hand"],
        scene_cfg=_robot_scene(),
    )
    assert src_b == [0]
    assert dst_b == [1]


def test_select_proxy_bodies_accepts_string_without_body_names():
    """A raw prim-path string in ``proxy_bodies`` bypasses the ``body_names`` requirement."""
    model = _model_with_two_bodies(with_shapes=True)
    proxy_ids = NewtonCoupledSolverManager._select_proxy_bodies(
        model,
        proxy_bodies=["/World/envs/env_.*/Robot/panda_hand"],
        scene_cfg=None,
    )
    assert proxy_ids == [1]
