# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for MDP action terms."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.controllers import DifferentialIKControllerCfg, OperationalSpaceControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    AbsBinaryJointPositionActionCfg,
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
    OperationalSpaceControllerActionCfg,
)
from isaaclab.envs.mdp.actions.binary_joint_actions import AbsBinaryJointPositionAction, BinaryJointPositionAction
from isaaclab.envs.mdp.actions.task_space_actions import (
    DifferentialInverseKinematicsAction,
    OperationalSpaceControllerAction,
)

pytestmark = pytest.mark.unit

NUM_ENVS = 2
JOINT_NAMES = [f"joint{i}" for i in range(7)]


class _JointIds:
    """Stand-in for the proxy returned by ``find_joints(..., as_proxy=True)``."""

    def __init__(self, count: int):
        self.warp = torch.arange(count, dtype=torch.int32)

    def __len__(self) -> int:
        return len(self.warp)


def _make_env(joint_names: list[str] | None = None, num_envs: int = NUM_ENVS) -> SimpleNamespace:
    """Create a minimal environment whose robot resolves every joint query to ``joint_names``."""
    if joint_names is None:
        joint_names = JOINT_NAMES

    def find_joints(_names, *, as_proxy: bool = False):
        joint_ids = _JointIds(len(joint_names)) if as_proxy else list(range(len(joint_names)))
        return joint_ids, list(joint_names)

    robot = SimpleNamespace(
        num_joints=len(joint_names),
        find_joints=find_joints,
        find_bodies=lambda _name: ([len(joint_names)], ["tool"]),
        num_base_dofs=0,
        is_fixed_base=True,
    )
    sim = SimpleNamespace(get_physics_dt=lambda: 0.01)
    return SimpleNamespace(num_envs=num_envs, device="cpu", scene={"robot": robot}, sim=sim)


def _diff_ik_cfg(
    command_type: str, use_relative_mode: bool, clip: dict[str, tuple[float, float]] | None = None
) -> DifferentialInverseKinematicsActionCfg:
    return DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        body_name="tool",
        controller=DifferentialIKControllerCfg(
            command_type=command_type, use_relative_mode=use_relative_mode, ik_method="dls"
        ),
        clip=clip,
    )


def _osc_cfg(
    target_types: tuple[str, ...], impedance_mode: str, clip: dict[str, tuple[float, float]] | None = None
) -> OperationalSpaceControllerActionCfg:
    return OperationalSpaceControllerActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        body_name="tool",
        controller_cfg=OperationalSpaceControllerCfg(
            target_types=target_types,
            impedance_mode=impedance_mode,
            motion_stiffness_limits_task=(0.0, 500.0),
            motion_damping_ratio_limits_task=(0.0, 2.0),
        ),
        clip=clip,
    )


@pytest.mark.parametrize(
    "action_cls, cfg_cls",
    [
        (BinaryJointPositionAction, BinaryJointPositionActionCfg),
        (AbsBinaryJointPositionAction, AbsBinaryJointPositionActionCfg),
    ],
)
def test_binary_joint_action_clips_each_processed_joint(action_cls, cfg_cls) -> None:
    """Per-joint clip ranges cover every processed joint, not just the 1-D binary input."""
    cfg = cfg_cls(
        asset_name="robot",
        joint_names=["finger_.*"],
        open_command_expr={"finger_left": 1.0, "finger_right": 2.0},
        close_command_expr={"finger_left": -1.0, "finger_right": -2.0},
        clip={"finger_left": (-0.5, 0.5), "finger_right": (-1.5, 1.5)},
    )
    action = action_cls(cfg, _make_env(["finger_left", "finger_right"]))

    action.process_actions(torch.ones((2, 1)))

    torch.testing.assert_close(action.processed_actions, torch.tensor([[0.5, 1.5], [0.5, 1.5]]))


@pytest.mark.parametrize(
    ("command_type", "use_relative_mode", "expected"),
    [
        ("position", False, ["pos_x", "pos_y", "pos_z"]),
        ("pose", False, ["pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w"]),
    ],
)
def test_diff_ik_action_names(command_type: str, use_relative_mode: bool, expected: list[str]) -> None:
    """The action names follow the command layout and have one entry per action dimension."""
    action = DifferentialInverseKinematicsAction(_diff_ik_cfg(command_type, use_relative_mode), _make_env())
    assert action.action_names == expected
    assert len(action.action_names) == action.action_dim


def test_diff_ik_clip_matches_task_space_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clip keys select task-space components; unmatched components are left alone."""
    cfg = _diff_ik_cfg("pose", True, clip={"pos_.*": (-0.05, 0.05), "rot_z": (-0.1, 0.1)})
    action = DifferentialInverseKinematicsAction(cfg, _make_env())
    assert action.action_names == ["pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z"]
    assert len(action.action_names) == action.action_dim
    ee_pos = torch.zeros(NUM_ENVS, 3)
    ee_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(NUM_ENVS, 1)
    monkeypatch.setattr(action, "_compute_frame_pose", lambda: (ee_pos, ee_quat))

    raw = torch.tensor([[0.2, -0.2, 0.01, 0.5, -0.5, 0.5], [-0.01, 0.03, -0.3, -1.0, 1.0, -1.0]])
    expected = torch.tensor([[0.05, -0.05, 0.01, 0.5, -0.5, 0.1], [-0.01, 0.03, -0.05, -1.0, 1.0, -0.1]])

    action.process_actions(raw)

    torch.testing.assert_close(action.processed_actions, expected)


def test_diff_ik_clip_with_joint_names_raises() -> None:
    """Joint-name keys do not describe task-space components and are rejected instead of mis-indexing."""
    with pytest.raises(ValueError, match="action names"):
        DifferentialInverseKinematicsAction(_diff_ik_cfg("pose", True, clip={"joint.*": (-1.0, 1.0)}), _make_env())


def test_diff_ik_clip_quaternion_raises() -> None:
    """Clamping quaternion components does not give a valid orientation, so it is rejected."""
    with pytest.raises(ValueError, match="quaternion"):
        DifferentialInverseKinematicsAction(_diff_ik_cfg("pose", False, clip={"quat_.*": (-1.0, 1.0)}), _make_env())


def test_osc_action_names() -> None:
    """Target types contribute their components in order, followed by the impedance parameters."""
    action = OperationalSpaceControllerAction(_osc_cfg(("pose_rel", "wrench_abs"), "variable"), _make_env())
    expected = (
        ["pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z"]
        + ["force_x", "force_y", "force_z", "torque_x", "torque_y", "torque_z"]
        + [f"stiffness_{axis}" for axis in ("x", "y", "z", "rx", "ry", "rz")]
        + [f"damping_ratio_{axis}" for axis in ("x", "y", "z", "rx", "ry", "rz")]
    )
    assert action.action_names == expected
    assert len(action.action_names) == action.action_dim


def test_osc_action_names_prefixed_when_ambiguous() -> None:
    """With both pose targets, the pose names are prefixed with their target type; the impedance names are not."""
    action = OperationalSpaceControllerAction(_osc_cfg(("pose_abs", "pose_rel"), "variable"), _make_env())
    axes = ("x", "y", "z", "rx", "ry", "rz")
    assert action.action_names[:3] == ["pose_abs_pos_x", "pose_abs_pos_y", "pose_abs_pos_z"]
    assert action.action_names[7:10] == ["pose_rel_pos_x", "pose_rel_pos_y", "pose_rel_pos_z"]
    assert action.action_names[13:19] == [f"stiffness_{axis}" for axis in axes]
    assert action.action_names[19:25] == [f"damping_ratio_{axis}" for axis in axes]
    assert len(set(action.action_names)) == action.action_dim


def test_osc_clip_prefixed_quaternion_raises() -> None:
    """The quaternion components are rejected under their block prefix as well."""
    cfg = _osc_cfg(("pose_abs", "pose_rel"), "fixed", clip={"pose_abs_quat_.*": (-0.1, 0.1)})
    with pytest.raises(ValueError, match="quaternion"):
        OperationalSpaceControllerAction(cfg, _make_env())


@pytest.mark.parametrize(
    ("clip", "expected_z", "expected_stiffness"),
    [
        ({"pos_z": (-0.01, 0.01), "stiffness_.*": (0.0, 100.0)}, [0.01, -0.01], [100.0, 0.0]),
        ({}, [1.0, -1.0], [500.0, 0.0]),
    ],
)
def test_osc_clip_applies_to_processed_actions(
    clip: dict[str, tuple[float, float]], expected_z: list[float], expected_stiffness: list[float]
) -> None:
    """Scaled actions are clipped before the controller limits on the gains are applied."""
    cfg = _osc_cfg(("pose_rel",), "variable_kp", clip=clip)
    cfg.position_scale = 2.0
    cfg.stiffness_scale = 0.5
    action = OperationalSpaceControllerAction(cfg, _make_env())

    raw = torch.zeros(NUM_ENVS, action.action_dim)
    raw[:, :2] = torch.tensor([[0.02, -0.03], [-0.04, 0.05]])
    raw[0, 2] = 0.5  # pos_z above the clip range
    raw[1, 2] = -0.5
    raw[0, 6:12] = 2000.0  # scaled stiffness above both the clip range and the controller limit
    raw[1, 6:12] = -10.0  # scaled stiffness below the controller limit

    action._preprocess_actions(raw)

    processed = action.processed_actions
    torch.testing.assert_close(processed[:, 2], torch.tensor(expected_z))
    torch.testing.assert_close(processed[:, 6:12], torch.tensor(expected_stiffness)[:, None].expand(-1, 6))
    # Unclipped components retain their scaled values.
    torch.testing.assert_close(processed[:, :2], torch.tensor([[0.04, -0.06], [-0.08, 0.1]]))


def test_osc_controller_limits_bound_clip() -> None:
    """A clip range outside the controller limits cannot move the gains past them."""
    cfg = _osc_cfg(("pose_rel",), "variable_kp", clip={"stiffness_.*": (600.0, 1000.0)})
    action = OperationalSpaceControllerAction(cfg, _make_env())
    raw = torch.zeros(NUM_ENVS, action.action_dim)
    raw[0, 6:12] = 2000.0  # clipped to 1000, then limited to 500
    raw[1, 6:12] = 0.0  # raised to 600 by the clip, then limited to 500

    action._preprocess_actions(raw)

    torch.testing.assert_close(action.processed_actions[:, 6:12], torch.full((NUM_ENVS, 6), 500.0))
