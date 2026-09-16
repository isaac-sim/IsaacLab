# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sim-free unit tests for the SO-101 IK layer.

The SO-101 controller is a thin subclass of the core
:class:`~isaaclab.controllers.DifferentialIKController`; the only SO-101-specific behavior is the
wrist-only orientation joint mask. The generic IK features it relies on (the ``adaptive_dls``
ik-method, per-axis orientation weighting, and null-space joint-limit avoidance) are tested in
``source/isaaclab/test/controllers/test_differential_ik_features.py``. These tests cover the mask
plus the IK-Abs env/action wiring with hand-built tensors -- no gym.make, USD, or GPU.
"""

import dataclasses
import math

import pytest
import torch

from isaaclab.controllers.differential_ik import DifferentialIKController

from isaaclab_tasks.contrib.stack.config.so101.pose_ik_controller import (
    SO101PoseIKController,
    SO101PoseIKControllerCfg,
)

# The SO-101 arm IK acts over 5 joints; the geometric Jacobian is (N, 6, 5).
_NUM_JOINTS = 5
_ID_QUAT = [0.0, 0.0, 0.0, 1.0]  # xyzw identity


def _quat_xyzw(axis: list[float], angle: float) -> list[float]:
    """Build a unit xyzw quaternion from an axis (need not be unit) and angle [rad]."""
    norm = math.sqrt(sum(a * a for a in axis)) or 1.0
    s = math.sin(angle / 2.0)
    return [axis[0] / norm * s, axis[1] / norm * s, axis[2] / norm * s, math.cos(angle / 2.0)]


def _make_controller(
    num_envs: int = 1,
    orientation_weight=1.0,
    joint_limit_avoidance_gain: float = 0.0,
    joint_limit_avoidance_margin: float = 0.3,
):
    cfg = SO101PoseIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="adaptive_dls",
        ik_params={"lambda_min": 0.05, "lambda_max": 0.2, "sigma_thresh": 0.02},
        orientation_weight=orientation_weight,
        joint_limit_avoidance_gain=joint_limit_avoidance_gain,
        joint_limit_avoidance_margin=joint_limit_avoidance_margin,
    )
    return SO101PoseIKController(cfg=cfg, num_envs=num_envs, device="cpu", num_joints=_NUM_JOINTS)


def test_orientation_joint_mask_zeros_unmasked_orientation_columns():
    """The SO-101 orientation joint mask zeros the orientation-row columns of the masked-out joints
    (so they serve position only), while the position rows and the task error are unchanged.

    This is the only SO-101-specific addition over the core controller: only ``wrist_flex`` /
    ``wrist_roll`` (the last two columns) may serve orientation, so ``shoulder_pan`` (col 0) never
    drives it and the base does not swing to track a commanded orientation.
    """
    ee_pos = torch.tensor([[0.3, 0.0, 0.2]])
    ee_quat = torch.tensor([_ID_QUAT])
    jac = torch.arange(6 * _NUM_JOINTS, dtype=torch.float32).reshape(1, 6, _NUM_JOINTS)
    cmd = torch.tensor([[0.31, 0.0, 0.2] + _quat_xyzw([0.3, 0.5, 0.8], 0.7)])

    # weight 1.0 isolates the mask effect (no per-axis scaling on top)
    c = _make_controller(orientation_weight=1.0)
    c.set_orientation_joint_mask(torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0]))  # wrist joints only
    c.set_command(cmd)
    original_jac = jac.clone()
    task_jac = jac.clone()
    task_jac[:, 3:, :3] = 0.0

    # Reference: the base solver receives a Jacobian with only wrist orientation columns.
    base = DifferentialIKController(c.cfg, num_envs=1, device="cpu", num_joints=_NUM_JOINTS)
    base.set_command(cmd)
    joint_pos = torch.zeros(1, _NUM_JOINTS)
    out = torch.empty_like(joint_pos)
    result = c.compute(ee_pos, ee_quat, jac, joint_pos, out=out)
    assert result is out
    torch.testing.assert_close(result, base.compute(ee_pos, ee_quat, task_jac, joint_pos))
    torch.testing.assert_close(jac, original_jac)


def test_mask_none_leaves_orientation_unmasked():
    """Without a mask, the SO-101 controller matches the base solver."""
    jac = torch.arange(6 * _NUM_JOINTS, dtype=torch.float32).reshape(1, 6, _NUM_JOINTS)
    c = _make_controller(orientation_weight=1.0)
    base = DifferentialIKController(c.cfg, num_envs=1, device="cpu", num_joints=_NUM_JOINTS)
    command = torch.tensor([[0.31, 0.0, 0.2] + _quat_xyzw([1.0, 0.0, 0.0], 0.5)])
    c.set_command(command)
    base.set_command(command)
    inputs = (torch.zeros(1, 3), torch.tensor([_ID_QUAT]), jac, torch.zeros(1, _NUM_JOINTS))
    torch.testing.assert_close(c.compute(*inputs), base.compute(*inputs))


def test_compute_returns_joint_targets_shape():
    """End-to-end compute (adaptive DLS + orientation weight + mask + JLA) returns one target per
    joint."""
    c = _make_controller(orientation_weight=0.5, joint_limit_avoidance_gain=0.5)
    c.set_orientation_joint_mask(torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0]))
    c.set_joint_pos_limits(torch.full((_NUM_JOINTS,), -1.0), torch.full((_NUM_JOINTS,), 1.0))
    ee_pos = torch.tensor([[0.3, 0.0, 0.2]])
    ee_quat = torch.tensor([_ID_QUAT])
    c.set_command(torch.tensor([[0.31, 0.0, 0.2] + _ID_QUAT]))
    jac = torch.zeros(1, 6, _NUM_JOINTS)
    for i in range(_NUM_JOINTS):
        jac[0, i, i] = 1.0
    out = c.compute(ee_pos, ee_quat, jac, torch.zeros(1, _NUM_JOINTS))
    assert out.shape == (1, _NUM_JOINTS)


def test_action_cfg_points_at_custom_action_and_controller():
    """The action cfg wires the custom action class and the SO-101 controller cfg."""
    pytest.importorskip("pxr")  # the action term imports UsdPhysics at module load
    from isaaclab.utils.string import string_to_callable

    from isaaclab_tasks.contrib.stack.config.so101.pose_ik_action import SO101PoseIKActionCfg
    from isaaclab_tasks.contrib.stack.config.so101.pose_ik_action_term import SO101PoseIKAction

    cfg = SO101PoseIKActionCfg(
        asset_name="robot",
        joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
        body_name="gripper",
        controller=SO101PoseIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="adaptive_dls"),
    )
    # ``class_type`` is a lazy ``{DIR}.pose_ik_action_term:SO101PoseIKAction`` string (so the cfg
    # stays importable without Kit); resolve it to confirm it points at the custom term.
    assert string_to_callable(str(cfg.class_type)) is SO101PoseIKAction
    assert isinstance(cfg.controller, SO101PoseIKControllerCfg)
    assert string_to_callable(str(cfg.controller.class_type)) is SO101PoseIKController


def test_env_cfg_arm_action_is_pose_and_ordering_matches_pipeline():
    """The IK-Abs env wires the full-pose arm action, the wrist-only orientation mask, and the
    8D action ordering."""
    pytest.importorskip("pxr")
    from isaaclab_tasks.contrib.stack.config.so101.pose_ik_action import SO101PoseIKActionCfg
    from isaaclab_tasks.contrib.stack.config.so101.stack_ik_abs_env_cfg import (
        SO101CubeStackEnvCfg,
        SO101IkActionsCfg,
    )

    cfg = SO101CubeStackEnvCfg()
    assert isinstance(cfg.actions.arm_action, SO101PoseIKActionCfg)
    controller = cfg.actions.arm_action.controller
    assert controller.command_type == "pose"
    assert controller.ik_method == "adaptive_dls"
    assert controller.orientation_joint_names == ("wrist_flex", "wrist_roll")
    assert cfg.actions.arm_action.joint_names == [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    ]
    # Field order is the positional contract with the pipeline output_order:
    # [arm(pos+quat), gripper] -> [pos_xyz, quat_xyzw, gripper].
    assert [f.name for f in dataclasses.fields(SO101IkActionsCfg)] == [
        "arm_action",
        "gripper_action",
    ]
