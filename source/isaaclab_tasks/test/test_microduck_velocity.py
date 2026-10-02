# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deployment contract and reset/step checks for MicroDuck flat walking."""

import gymnasium as gym
import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators.newton import read_group_parameter
from isaaclab.sim import SimulationContext
from isaaclab.test.utils import DeviceScope, test_devices

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

TASK = "IsaacContrib-Velocity-Flat-MicroDuck"
JOINT_NAMES = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]
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


def test_microduck_policy_contract():
    """Keep the original walking policy's joint order and 50 Hz interface."""
    cfg = parse_env_cfg(TASK, device="cpu", num_envs=2)
    assert cfg.sim.dt * cfg.decimation == pytest.approx(0.02)
    assert cfg.actions.joint_pos.joint_names == JOINT_NAMES
    assert cfg.actions.joint_pos.preserve_order
    assert cfg.actions.joint_pos.scale == 1.0
    terms = [name for name in vars(cfg.observations.policy) if name in POLICY_TERMS]
    assert terms == POLICY_TERMS
    assert cfg.observations.policy.joint_pos.params["asset_cfg"].joint_names == JOINT_NAMES
    assert cfg.scene.terrain.terrain_type == "plane"
    assert cfg.sim.use_newton_actuators


@pytest.mark.integration
@pytest.mark.kitless
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_microduck_reset_and_step(device):
    """Resolve the policy interface on the USD and exercise reset-time BAM writes."""
    cfg = parse_env_cfg(TASK, device=device, num_envs=8)
    sim_utils.create_new_stage()
    env = gym.make(TASK, cfg=cfg).unwrapped
    try:
        obs, _ = env.reset()
        assert obs["policy"].shape == (8, 61)
        assert obs["critic"].shape == (8, 76)
        assert env.observation_manager.active_terms["policy"] == POLICY_TERMS
        robot = env.scene["robot"]
        ids = env.action_manager.get_term("joint_pos")._joint_ids
        assert [robot.joint_names[i] for i in ids] == JOINT_NAMES
        before = read_group_parameter(robot.actuators, "servos", "drive", "friction_scale").clone()
        env.reset()
        after = read_group_parameter(robot.actuators, "servos", "drive", "friction_scale")
        assert not torch.equal(before, after)
        torch.testing.assert_close(after, after[:, :1].expand_as(after))
        assert ((after >= 0.9) & (after <= 1.1)).all()
        with torch.inference_mode():
            for _ in range(64):
                obs, reward, terminated, truncated, _ = env.step(torch.randn(8, 14, device=env.device) * 0.1)
                assert all(torch.isfinite(value).all() for value in obs.values())
                assert torch.isfinite(reward).all()
        assert terminated.shape == truncated.shape == (8,)
    finally:
        env.close()
        SimulationContext.clear_instance()
