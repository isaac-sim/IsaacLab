# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the settle-before-gripper hold in Mimic data generation.

``insert_settle_frames_before_gripper`` is pure tensor logic; the last test checks that the held frames
survive the trajectory merge that ``generate()`` executes.
"""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

import pytest
import torch

from isaaclab_mimic.datagen.data_generator import insert_settle_frames_before_gripper
from isaaclab_mimic.datagen.waypoint import Waypoint, WaypointSequence, WaypointTrajectory

GRIPPER = [-1, -1, -1, -1, 1, 1, 1, -1, -1, -1]
"""A segment that closes the gripper at frame 4 and opens it again at frame 7."""


def _sequence(gripper_values, num_dofs=1):
    """Poses whose x translation is the frame index, and gripper actions of shape (T, num_dofs)."""
    num_frames = len(gripper_values)
    poses = torch.eye(4).repeat(num_frames, 1, 1)
    poses[:, 0, 3] = torch.arange(num_frames, dtype=torch.float32)
    gripper_actions = torch.tensor(gripper_values, dtype=torch.float32).unsqueeze(1).repeat(1, num_dofs)
    return poses, gripper_actions


def _frames(poses):
    return [int(x) for x in poses[:, 0, 3].tolist()]


def test_zero_steps_returns_the_inputs_unchanged():
    poses, gripper = _sequence(GRIPPER)
    out_poses, out_gripper, out_noise = insert_settle_frames_before_gripper(poses, gripper, 0.03, num_steps=0)
    assert out_poses is poses and out_gripper is gripper and out_noise == 0.03


def test_a_sequence_without_gripper_transition_is_unchanged():
    poses, gripper = _sequence([1] * 6)
    out_poses, out_gripper, out_noise = insert_settle_frames_before_gripper(poses, gripper, 0.03, num_steps=5)
    assert out_poses is poses and out_gripper is gripper and out_noise == 0.03


def test_a_single_frame_is_unchanged():
    poses, gripper = _sequence([1])
    out_poses, _, _ = insert_settle_frames_before_gripper(poses, gripper, 0.03, num_steps=5)
    assert out_poses is poses


@pytest.mark.parametrize("num_dofs", [1, 2])
def test_hold_inserts_copies_before_every_transition(num_dofs):
    poses, gripper = _sequence(GRIPPER, num_dofs)
    new_poses, new_gripper, new_noise = insert_settle_frames_before_gripper(poses, gripper, 0.03, num_steps=3)
    assert new_poses.shape[0] == new_gripper.shape[0] == new_noise.shape[0] == 16
    # 0..3 | hold at frame 4 x3 | 4..6 | hold at frame 7 x3 | 7..9
    assert _frames(new_poses) == [0, 1, 2, 3, 4, 4, 4, 4, 5, 6, 7, 7, 7, 7, 8, 9]
    # the held frames keep the gripper action of the frame before the transition and carry no noise
    assert new_gripper[4:7].eq(gripper[3]).all() and new_gripper[10:13].eq(gripper[6]).all()
    assert new_noise[4:7].eq(0).all() and new_noise[10:13].eq(0).all()
    # the source frames are untouched, in order, and keep their noise
    kept = torch.ones(16, dtype=torch.bool)
    kept[4:7] = False
    kept[10:13] = False
    assert torch.equal(new_poses[kept], poses) and torch.equal(new_gripper[kept], gripper)
    assert new_noise[kept].eq(0.03).all()


def test_per_frame_noise_is_zeroed_on_the_held_frames_only():
    poses, gripper = _sequence(GRIPPER)
    noise = torch.arange(1, 11, dtype=torch.float32) / 100
    _, _, new_noise = insert_settle_frames_before_gripper(poses, gripper, noise, num_steps=2)
    expected = noise[:4].tolist() + [0, 0] + noise[4:7].tolist() + [0, 0] + noise[7:].tolist()
    assert new_noise.flatten().tolist() == pytest.approx(expected)


def test_a_transition_at_the_last_frame_is_held():
    poses, gripper = _sequence([-1, -1, -1, 1])
    new_poses, new_gripper, _ = insert_settle_frames_before_gripper(poses, gripper, 0.0, num_steps=2)
    assert _frames(new_poses) == [0, 1, 2, 3, 3, 3]
    assert new_gripper.flatten().tolist() == [-1, -1, -1, -1, -1, 1]


def test_the_held_frames_reach_the_executed_sequence():
    """The held frames must survive the interpolate-merge-pop that merge_eef_subtask_trajectory does."""
    poses, gripper = _sequence(GRIPPER)
    new_poses, new_gripper, new_noise = insert_settle_frames_before_gripper(poses, gripper, 0.03, num_steps=3)
    subtask_traj = WaypointTrajectory()
    subtask_traj.add_waypoint_sequence(WaypointSequence.from_poses(new_poses, new_gripper, new_noise))
    traj = WaypointTrajectory()
    traj.add_waypoint_sequence(WaypointSequence(sequence=[Waypoint(torch.eye(4), gripper[0], 0.03)]))
    traj.merge(subtask_traj, num_steps_interp=5, num_steps_fixed=0, action_noise=0.0)
    traj.pop_first()
    full = traj.get_full_sequence()
    # five interpolation frames precede the subtask segment, which then follows frame by frame
    executed = [int(waypoint.pose[0, 3]) for waypoint in full.sequence[5:]]
    assert executed == _frames(new_poses)
    held = [i for i in range(5, len(full)) if float(full[i].noise) == 0]
    assert held == [5 + 4, 5 + 5, 5 + 6, 5 + 10, 5 + 11, 5 + 12]
    assert all(int(full[i].gripper_action[0]) == (-1 if i < 5 + 10 else 1) for i in held)
