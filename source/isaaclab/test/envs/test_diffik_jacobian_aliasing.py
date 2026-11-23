# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for task-space action processing, body offsets, and Jacobian aliasing (NVBug 6043099).

``DifferentialInverseKinematicsAction._compute_frame_jacobian`` historically
aliased the parent Jacobian and applied the body-offset correction in place.
When the parent Jacobian was a view onto the engine's mutable buffer, repeated
calls within a single simulation step accumulated the correction. The fix
copies the Jacobian into an owned buffer before mutating, making the method
idempotent regardless of whether ``jacobian_b`` returns a view or a copy.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import (
    DifferentialInverseKinematicsAction,
    OperationalSpaceControllerAction,
)
from isaaclab.utils import math as math_utils

pytestmark = pytest.mark.unit


class _Stub:
    """Minimal stand-in for the task-space actions' Jacobian methods.

    ``jacobian_b`` returns the backing buffer
    **without copying**, mirroring the worst case where the data layer hands out a view
    onto engine memory. The owned ``_jacobian_b`` buffer is what the fixed method must
    write into.
    """

    def __init__(self, num_envs: int, num_joints: int, body_offset_pos, body_offset_rot, backing_buffer):
        self.cfg = SimpleNamespace(body_offset=SimpleNamespace(pos=body_offset_pos, rot=body_offset_rot))
        self._offset_pos = torch.tensor(body_offset_pos, dtype=torch.float32).repeat(num_envs, 1)
        self._offset_rot = torch.tensor(body_offset_rot, dtype=torch.float32).repeat(num_envs, 1)
        self._jacobian_b = torch.zeros(num_envs, 6, num_joints)
        self._backing_buffer = backing_buffer
        # Non-trivial root and body orientations, so the offset must be rotated into the root frame.
        self._body_idx = self._ee_body_idx = 0
        root_quat_w = math_utils.quat_from_euler_xyz(torch.tensor(0.3), torch.tensor(-0.2), torch.tensor(0.5))
        body_quat_w = math_utils.quat_from_euler_xyz(torch.tensor(-1.1), torch.tensor(0.4), torch.tensor(2.0))
        self._asset = SimpleNamespace(
            data=SimpleNamespace(
                root_quat_w=SimpleNamespace(torch=root_quat_w.repeat(num_envs, 1)),
                body_quat_w=SimpleNamespace(torch=body_quat_w.repeat(num_envs, 1, 1)),
            )
        )

    @property
    def jacobian_b(self):
        return self._backing_buffer


def _make_stub(num_envs: int, num_joints: int, body_offset_pos, body_offset_rot, backing_buffer: torch.Tensor):
    return _Stub(num_envs, num_joints, body_offset_pos, body_offset_rot, backing_buffer)


def test_process_actions_applies_scale_then_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    """DiffIK actions use the configured affine transformation before reaching the controller."""
    num_envs = 2
    controller_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")
    cfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        body_name="tool",
        scale=(0.5, 2.0, -1.0),
        offset=(0.1, -0.2, 0.3),
        controller=controller_cfg,
    )
    asset = SimpleNamespace(
        find_joints=lambda _names: ([0, 1, 2], ["joint1", "joint2", "joint3"]),
        find_bodies=lambda _name: ([0], ["tool"]),
        num_joints=3,
        num_base_dofs=0,
        is_fixed_base=False,
    )
    env = SimpleNamespace(scene={"robot": asset}, num_envs=num_envs, device="cpu")
    action = DifferentialInverseKinematicsAction(cfg, env)
    ee_pos = torch.zeros(num_envs, 3)
    ee_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(num_envs, 1)
    monkeypatch.setattr(action, "_compute_frame_pose", lambda: (ee_pos, ee_quat))

    raw_actions = torch.tensor([[1.0, -2.0, 0.5], [-1.0, 0.25, -0.5]])
    expected = torch.tensor([[0.6, -4.2, -0.2], [-0.4, 0.3, 0.8]])

    action.process_actions(raw_actions)

    torch.testing.assert_close(action.raw_actions, raw_actions)
    torch.testing.assert_close(action.processed_actions, expected)
    torch.testing.assert_close(action._ik_controller.ee_pos_des, expected)


def test_compute_frame_jacobian_is_idempotent_within_step():
    """Two consecutive calls under the same state must return identical Jacobians.

    Regression for NVBug 6043099. With the buggy alias-and-mutate pattern, the
    second call returned the first-call result with the body-offset correction
    applied a second time.
    """
    num_envs, num_joints = 4, 7
    backing = torch.randn(num_envs, 6, num_joints)
    backing_snapshot = backing.clone()

    stub = _make_stub(
        num_envs,
        num_joints,
        body_offset_pos=[0.0, 0.0, 0.05],
        body_offset_rot=[1.0, 0.0, 0.0, 0.0],
        backing_buffer=backing,
    )

    compute = DifferentialInverseKinematicsAction._compute_frame_jacobian

    j1 = compute(stub).clone()
    j2 = compute(stub).clone()
    j3 = compute(stub).clone()

    torch.testing.assert_close(j1, j2)
    torch.testing.assert_close(j1, j3)
    # The backing buffer must be untouched: the fix may not corrupt the source.
    torch.testing.assert_close(backing, backing_snapshot)


@pytest.mark.parametrize(
    ("compute", "returns_jacobian"),
    [
        (DifferentialInverseKinematicsAction._compute_frame_jacobian, True),
        (OperationalSpaceControllerAction._compute_ee_jacobian, False),
    ],
    ids=["diff_ik", "osc"],
)
@pytest.mark.parametrize(
    ("root_yaw", "body_yaw"), [(0.0, math.pi / 2.0), (math.pi / 2.0, math.pi)], ids=["identity_root", "rotated_root"]
)
def test_body_offset_jacobian_uses_offset_in_root_frame(compute, returns_jacobian, root_yaw, body_yaw):
    """The offset uses root axes, and a rigid offset rotation leaves angular rows unchanged."""
    # One revolute joint about root z, with the body rotated 90 degrees about z relative to the root.
    backing = torch.tensor([[[0.0], [0.0], [0.0], [0.0], [0.0], [1.0]]])
    # A 90 degree rotation about x for the offset frame must not change the angular rows.
    offset_rot = [math.sin(math.pi / 4.0), 0.0, 0.0, math.cos(math.pi / 4.0)]
    stub = _make_stub(1, 1, [1.0, 0.0, 0.0], offset_rot, backing)
    stub._asset.data.root_quat_w.torch[:] = torch.tensor([0.0, 0.0, math.sin(root_yaw / 2.0), math.cos(root_yaw / 2.0)])
    stub._asset.data.body_quat_w.torch[:] = torch.tensor([0.0, 0.0, math.sin(body_yaw / 2.0), math.cos(body_yaw / 2.0)])
    result = compute(stub)

    # R_body_b @ (1, 0, 0) = (0, 1, 0), and z x (0, 1, 0) = (-1, 0, 0).
    expected = torch.tensor([[[-1.0], [0.0], [0.0], [0.0], [0.0], [1.0]]])
    torch.testing.assert_close(stub._jacobian_b, expected)
    if returns_jacobian:
        torch.testing.assert_close(result, expected)


@pytest.mark.parametrize(("ee_quat", "ik_offset"), [((0.0, 0.0, 0.0, 1.0), 1.0), ((0.0, 0.0, 0.0, 0.0), 0.0)])
def test_apply_actions_holds_joints_for_uninitialized_frame(ee_quat, ik_offset):
    """IK targets are applied for a valid frame pose and joints hold for all-zero quaternions."""
    device = "cpu"
    joint_pos = torch.tensor([[0.1, 0.2], [0.3, 0.4]], device=device)
    frame_pose = (torch.zeros(2, 3, device=device), torch.tensor([ee_quat] * 2, device=device))
    written = {}
    stub = SimpleNamespace(
        cfg=SimpleNamespace(controller=SimpleNamespace(joint_limit_avoidance_gain=0.0)),
        _asset=SimpleNamespace(
            data=SimpleNamespace(joint_pos=SimpleNamespace(torch=joint_pos)),
            set_joint_position_target_index=lambda target, joint_ids: written.update(target=target),
        ),
        _joint_ids=slice(None),
        _limits_injected=True,
        _compute_frame_pose=lambda: frame_pose,
        _compute_frame_jacobian=lambda: torch.zeros(2, 6, 2, device=device),
        _ik_controller=SimpleNamespace(compute=lambda ee_pos, ee_quat, jacobian, joint_pos: joint_pos + 1.0),
    )
    DifferentialInverseKinematicsAction.apply_actions(stub)
    torch.testing.assert_close(written["target"], joint_pos + ik_offset)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
