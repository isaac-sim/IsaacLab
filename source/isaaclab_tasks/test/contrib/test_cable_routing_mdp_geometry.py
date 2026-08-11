# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for physical YAM contact frames and cable-joint strain geometry."""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.contrib.cable_routing.mdp.cable_geometry import cable_relative_joint_gap
from isaaclab_tasks.contrib.cable_routing.mdp.observations import active_goal_geometry, sampled_cable_state_b
from isaaclab_tasks.contrib.cable_routing.mdp.rewards import (
    cable_near_active_peg,
    cable_stretch,
    grippers_near_cable,
)
from isaaclab_tasks.contrib.cable_routing.mdp.terminations import cable_invalid_or_out_of_bounds
from isaaclab_tasks.contrib.cable_routing.yam_frames import (
    YAM_CONTACT_FRAME_OFFSET_POS,
    yam_contact_frame_position_w,
)


def _tensor_proxy(value: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(torch=value)


def _robot(body_pos_w: torch.Tensor, body_quat_w: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(
        data=SimpleNamespace(
            body_pos_w=_tensor_proxy(body_pos_w[:, None]),
            body_quat_w=_tensor_proxy(body_quat_w[:, None]),
        )
    )


def _scene(**assets) -> dict:
    return assets


def _segment_pose(position: tuple[float, float, float], quaternion: tuple[float, float, float, float]):
    return (*position, *quaternion)


def test_yam_contact_frame_position_applies_link_rotation_to_pad_offset() -> None:
    """The physical pad midpoint follows both the link translation and rotation."""
    half_sqrt = math.sqrt(0.5)
    robot = _robot(
        torch.tensor(((1.0, 2.0, 3.0), (1.0, 2.0, 3.0))),
        torch.tensor(((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, half_sqrt, half_sqrt))),
    )

    actual = yam_contact_frame_position_w(robot, 0)

    offset = torch.tensor(YAM_CONTACT_FRAME_OFFSET_POS)
    expected = torch.stack((torch.tensor((1.0, 2.0, 3.0)) + offset, torch.tensor((1.044, 2.0, 3.1297))))
    torch.testing.assert_close(actual, expected)


def test_goal_geometry_and_gripper_reward_use_physical_pad_midpoints() -> None:
    """Policy geometry and shaping distances are measured from the inner pads."""
    identity = torch.tensor(((0.0, 0.0, 0.0, 1.0),))
    left = _robot(torch.tensor(((0.0, 0.0, 0.0),)), identity)
    right = _robot(torch.tensor(((0.2, 0.0, 0.0),)), identity)
    left_contact = yam_contact_frame_position_w(left, 0)
    right_contact = yam_contact_frame_position_w(right, 0)
    target = torch.tensor(((0.1, 0.1, 0.2),))
    cable_points = torch.stack((left_contact[0], right_contact[0]), dim=0)[None]
    cable = SimpleNamespace(
        data=SimpleNamespace(segment_pose_w=_tensor_proxy(torch.nn.functional.pad(cable_points, (0, 4))))
    )
    command = SimpleNamespace(active_peg_positions_w=target)
    env = SimpleNamespace(
        scene=_scene(cable=cable, left=left, right=right),
        command_manager=SimpleNamespace(get_term=lambda _name: command),
    )
    cable_cfg = SceneEntityCfg("cable")
    left_cfg = SceneEntityCfg("left")
    left_cfg.body_ids = [0]
    right_cfg = SceneEntityCfg("right")
    right_cfg.body_ids = [0]

    geometry = active_goal_geometry(env, "route", cable_cfg, left_cfg, right_cfg)
    reward = grippers_near_cable(env, cable_cfg, left_cfg, right_cfg, std=0.15)

    torch.testing.assert_close(geometry[:, 0:3], target - left_contact)
    torch.testing.assert_close(geometry[:, 3:6], target - right_contact)
    torch.testing.assert_close(geometry[:, -2:], torch.zeros((1, 2)))
    torch.testing.assert_close(reward, torch.ones(1))


def test_cable_relative_joint_gap_is_invariant_to_connected_bending() -> None:
    """A right-angle hinge with coincident capsule endpoints has zero strain."""
    half_sqrt = math.sqrt(0.5)
    poses = torch.tensor(
        (
            (
                _segment_pose((-0.005, 0.0, 0.0), (0.0, half_sqrt, 0.0, half_sqrt)),
                _segment_pose((0.0, 0.005, 0.0), (-half_sqrt, 0.0, 0.0, half_sqrt)),
            ),
        )
    )

    gap = cable_relative_joint_gap(poses, rest_length=0.01)
    env = SimpleNamespace(
        scene=_scene(cable=SimpleNamespace(data=SimpleNamespace(segment_pose_w=_tensor_proxy(poses))))
    )
    reward_value = cable_stretch(env, SceneEntityCfg("cable"), rest_length=0.01)

    torch.testing.assert_close(gap, torch.zeros((1, 1)), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(reward_value, torch.zeros(1), atol=1.0e-12, rtol=0.0)


def test_cable_stretch_penalizes_relative_endpoint_separation() -> None:
    """A two-millimeter joint gap on a one-centimeter segment yields 0.2 relative error."""
    half_sqrt = math.sqrt(0.5)
    poses = torch.tensor(
        (
            (
                _segment_pose((0.005, 0.0, 0.0), (0.0, half_sqrt, 0.0, half_sqrt)),
                _segment_pose((0.017, 0.0, 0.0), (0.0, half_sqrt, 0.0, half_sqrt)),
            ),
        )
    )
    env = SimpleNamespace(
        scene=_scene(cable=SimpleNamespace(data=SimpleNamespace(segment_pose_w=_tensor_proxy(poses))))
    )

    gap = cable_relative_joint_gap(poses, rest_length=0.01)
    reward_value = cable_stretch(env, SceneEntityCfg("cable"), rest_length=0.01)

    torch.testing.assert_close(gap, torch.tensor(((0.2,),)), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(reward_value, torch.tensor((0.04,)), atol=1.0e-6, rtol=0.0)


def test_cable_rewards_zero_only_nonfinite_environment_rows() -> None:
    """A terminal non-finite cable state must not poison another DDP rank's reward batch."""
    identity = torch.tensor(((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)))
    left = _robot(torch.tensor(((0.0, 0.0, 0.0), (torch.nan, 0.0, 0.0))), identity)
    right = _robot(torch.tensor(((0.02, 0.0, 0.0), (0.02, 0.0, 0.0))), identity)
    poses = torch.tensor(
        (
            (
                _segment_pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
                _segment_pose((0.0, 0.0, 0.01), (0.0, 0.0, 0.0, 1.0)),
            ),
            (
                _segment_pose((torch.nan, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
                _segment_pose((0.0, 0.0, 0.01), (0.0, 0.0, 0.0, 1.0)),
            ),
        )
    )
    cable = SimpleNamespace(data=SimpleNamespace(segment_pose_w=_tensor_proxy(poses)))
    command = SimpleNamespace(active_peg_positions_w=torch.zeros((2, 3)))
    env = SimpleNamespace(
        scene=_scene(cable=cable, left=left, right=right),
        command_manager=SimpleNamespace(get_term=lambda _name: command),
    )
    left_cfg = SceneEntityCfg("left")
    left_cfg.body_ids = [0]
    right_cfg = SceneEntityCfg("right")
    right_cfg.body_ids = [0]

    rewards = torch.stack(
        (
            cable_near_active_peg(env, "route", SceneEntityCfg("cable"), std=0.08),
            grippers_near_cable(env, SceneEntityCfg("cable"), left_cfg, right_cfg, std=0.15),
            cable_stretch(env, SceneEntityCfg("cable"), rest_length=0.01),
        ),
        dim=-1,
    )

    assert bool(torch.isfinite(rewards).all())
    assert bool((rewards[0] >= 0.0).all())
    torch.testing.assert_close(rewards[1], torch.zeros(3))


def test_invalid_cable_termination_covers_pose_orientation_and_velocity() -> None:
    """A non-finite quaternion or velocity must reset before it contaminates later observations."""
    poses = torch.zeros((3, 2, 7))
    poses[..., 6] = 1.0
    velocities = torch.zeros((3, 2, 6))
    poses[1, 0, 3] = torch.nan
    velocities[2, 1, 0] = torch.nan
    cable = SimpleNamespace(
        data=SimpleNamespace(
            segment_pose_w=_tensor_proxy(poses),
            segment_velocity_w=_tensor_proxy(velocities),
        )
    )

    class _TerminationScene(SimpleNamespace):
        def __getitem__(self, key):
            return getattr(self, key)

    env = SimpleNamespace(scene=_TerminationScene(cable=cable, env_origins=torch.zeros((3, 3))))

    invalid = cable_invalid_or_out_of_bounds(env, SceneEntityCfg("cable"))

    assert torch.equal(invalid, torch.tensor((False, True, True)))


def test_sampled_cable_state_uses_forward_tangents_and_repeats_the_final_one() -> None:
    """Sparse tangent evaluation preserves the previous full-chain observation semantics."""
    positions = torch.tensor((((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 2.0, 0.0), (1.0, 2.0, 3.0), (5.0, 2.0, 3.0)),))
    velocities = torch.arange(15, dtype=torch.float32).reshape(1, 5, 3)
    segment_pose = torch.nn.functional.pad(positions, (0, 4))
    segment_velocity = torch.nn.functional.pad(velocities, (0, 3))
    cable = SimpleNamespace(
        data=SimpleNamespace(
            segment_pose_w=_tensor_proxy(segment_pose),
            segment_velocity_w=_tensor_proxy(segment_velocity),
        )
    )

    class _ObservationScene(SimpleNamespace):
        def __getitem__(self, key):
            return getattr(self, key)

    env = SimpleNamespace(scene=_ObservationScene(cable=cable, env_origins=torch.tensor(((0.25, -0.5, 1.0),))))
    state = sampled_cable_state_b(env, SceneEntityCfg("cable"), num_samples=4).reshape(1, 4, 9)

    sample_ids = torch.tensor((0, 1, 3, 4))
    expected_tangent = torch.tensor((((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),))
    torch.testing.assert_close(state[..., :3], positions[:, sample_ids] - env.scene.env_origins[:, None])
    torch.testing.assert_close(state[..., 3:6], velocities[:, sample_ids])
    torch.testing.assert_close(state[..., 6:9], expected_tangent)
