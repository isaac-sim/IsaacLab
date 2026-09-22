# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for NullSpacePostureTask on a simplified humanoid model using the Pink library directly."""

from pathlib import Path

import numpy as np
import pytest
from pink.configuration import Configuration
from pinocchio.robot_wrapper import RobotWrapper

from isaaclab.controllers.pink_ik.null_space_posture_task import NullSpacePostureTask
from isaaclab.controllers.pink_ik.pink_task_cfg import NullSpacePostureTaskCfg

pytestmark = pytest.mark.unit

_URDF_PATH = Path(__file__).parent / "simplified_test_robot.urdf"
_NUM_JOINTS = 20
_WAIST_JOINTS = ["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"]
_LEFT_HAND = "left_hand_pitch_link"
_RIGHT_HAND = "right_hand_pitch_link"


@pytest.fixture(scope="module")
def robot_configuration():
    wrapper = RobotWrapper.BuildFromURDF(str(_URDF_PATH), None, root_joint=None)
    return Configuration(wrapper.model, wrapper.data, wrapper.q0)


def _posture_task(controlled_frames: list[str], controlled_joints: list[str]) -> NullSpacePostureTask:
    task = NullSpacePostureTask(
        NullSpacePostureTaskCfg(cost=1.0, controlled_frames=controlled_frames, controlled_joints=controlled_joints)
    )
    task.set_target(np.zeros(_NUM_JOINTS))
    return task


@pytest.mark.parametrize("controlled_frames", [[_LEFT_HAND], [_LEFT_HAND, _RIGHT_HAND]])
@pytest.mark.parametrize(
    "q", [np.zeros(_NUM_JOINTS), np.array([0.5] * 5 + [0.0] * 15), np.linspace(0.1, 2.0, _NUM_JOINTS)]
)
def test_null_space_jacobian_cancels_controlled_frame_velocities(robot_configuration, controlled_frames, q):
    """Velocities projected through the null-space Jacobian produce no velocity at any controlled frame."""
    robot_configuration.q = q
    task = _posture_task(controlled_frames, _WAIST_JOINTS)
    null_space_jacobian = task.compute_jacobian(robot_configuration)

    rng = np.random.default_rng(0)
    velocities = null_space_jacobian @ (rng.standard_normal((_NUM_JOINTS, 5)) * 0.1)
    for frame in controlled_frames:
        frame_jacobian = robot_configuration.get_frame_jacobian(frame)
        np.testing.assert_allclose(frame_jacobian @ velocities, 0.0, atol=1e-7)
        # the projector annihilates the frame Jacobian rows: N J^T = 0
        np.testing.assert_allclose(null_space_jacobian @ frame_jacobian.T, 0.0, atol=1e-7)


def test_null_space_jacobian_is_identity_without_controlled_frames(robot_configuration):
    robot_configuration.q = np.linspace(0.1, 2.0, _NUM_JOINTS)
    task = _posture_task([], [])
    np.testing.assert_allclose(task.compute_jacobian(robot_configuration), np.eye(_NUM_JOINTS))


@pytest.mark.parametrize(
    "controlled_joints", [["waist_pitch_joint", "left_shoulder_pitch_joint", "left_elbow_pitch_joint"], []]
)
def test_error_is_masked_to_controlled_joints(robot_configuration, controlled_joints):
    q = np.linspace(0.1, 2.0, _NUM_JOINTS)
    robot_configuration.q = q
    task = _posture_task([_LEFT_HAND], controlled_joints)

    joint_names = robot_configuration.model.names.tolist()[1:]
    expected = np.zeros(_NUM_JOINTS)
    for name in controlled_joints:
        expected[joint_names.index(name)] = q[joint_names.index(name)]
    np.testing.assert_allclose(task.compute_error(robot_configuration), expected, atol=1e-7)


def test_target_handling(robot_configuration):
    """Errors require a target; the configuration setter captures the current joint positions."""
    task = NullSpacePostureTask(
        NullSpacePostureTaskCfg(cost=1.0, controlled_frames=[_LEFT_HAND], controlled_joints=_WAIST_JOINTS[:2])
    )
    q = np.linspace(0.1, 2.0, _NUM_JOINTS)
    robot_configuration.q = q
    with pytest.raises(ValueError, match="No posture target has been set"):
        task.compute_error(robot_configuration)

    task.set_target_from_configuration(robot_configuration)
    np.testing.assert_allclose(task.target_q, q)
    np.testing.assert_allclose(task.compute_error(robot_configuration), 0.0)
