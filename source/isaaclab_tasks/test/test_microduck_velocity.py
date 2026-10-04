# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck flat walking: deployed policy interface and reset-time BAM friction writes.

Stepping every contributed task is covered by ``test/contrib/test_contrib_environments_kitless.py``.
"""

import gymnasium as gym
import pytest

import isaaclab.sim as sim_utils
from isaaclab.actuators.newton import read_group_parameter
from isaaclab.sim import SimulationContext
from isaaclab.test.utils import DeviceScope, test_devices

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.microduck.flat_env_cfg import MICRODUCK_JOINT_NAMES
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

TASK = "IsaacContrib-Velocity-Flat-MicroDuck"

POLICY_TERMS = [
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
    "velocity_commands",
    "head_pose_commands",
    "body_pose_commands",
]
"""Actor input order of the deployed ONNX policy."""


@pytest.mark.integration
@pytest.mark.kitless
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_microduck_deploy_interface_and_friction_reset(device):
    """The deployed policy's 61/76 inputs and joint order hold, and each reset redraws BAM friction."""
    cfg = parse_env_cfg(TASK, device=device, num_envs=2)
    sim_utils.create_new_stage()
    env = gym.make(TASK, cfg=cfg).unwrapped
    try:
        obs, _ = env.reset()
        assert obs["policy"].shape == (2, 61)
        assert obs["critic"].shape == (2, 76)
        assert env.observation_manager.active_terms["policy"] == POLICY_TERMS
        assert env.action_manager.get_term("joint_pos").IO_descriptor.joint_names == MICRODUCK_JOINT_NAMES

        robot = env.scene["robot"]
        before = read_group_parameter(robot.actuators, "servos", "drive", "friction_scale").clone()
        env.reset()
        after = read_group_parameter(robot.actuators, "servos", "drive", "friction_scale")
        assert not (before == after).all()
        assert (after == after[:, :1]).all(), "one friction scale per environment"
        assert ((after >= 0.9) & (after <= 1.1)).all()
    finally:
        env.close()
        SimulationContext.clear_instance()
