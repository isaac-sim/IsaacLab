# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck on a ball: the velocity command is a direction in a reference heading frame and drives a stepping clock."""

from isaaclab.test.utils import DeviceScope, launch_test_simulation, test_devices

from isaaclab_tasks.contrib.microduck_ball.ball_env_cfg import MicroDuckBallRollEnvCfg

ENV_CFG = MicroDuckBallRollEnvCfg()
ENV_CFG.scene.num_envs = 4
launch_test_simulation(ENV_CFG)

import pytest
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sim import SimulationContext
from isaaclab.utils import clone

STEPS = 15
"""Steps taken in each phase of the test: short enough that the robot, which receives zero actions, has not fallen."""


@pytest.mark.integration
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_reference_heading_and_clock_follow_the_command(device):
    """The reference heading turns only with the commanded yaw rate and the clock runs only while moving.

    A target velocity that turned with the robot could be met by walking in circles on a stationary ball.
    """
    cfg = clone(ENV_CFG)
    cfg.sim.device = device
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.terminations.duck_tilted = None
    cfg.terminations.duck_too_low = None
    cfg.terminations.ball_off_feet = None
    env = ManagerBasedRLEnv(cfg)
    try:
        env.reset()
        command = env.command_manager.get_term("base_velocity")
        zero_action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
        elapsed = STEPS * env.step_dt

        # no command: the reference heading and the clock stay put, and the policy sees a stopped clock
        command.vel_command_b[:] = 0.0
        start_heading, start_phase = command.reference_heading.clone(), command.gait_phase.clone()
        for _ in range(STEPS):
            env.step(zero_action)
        torch.testing.assert_close(command.reference_heading, start_heading)
        torch.testing.assert_close(command.gait_phase, start_phase)
        assert not command.gait_clock.any()

        # forward and yaw command: the reference heading turns with the yaw rate and the target velocity, which
        # points forward in the reference frame, turns with it in the world frame
        speed, yaw_rate = 0.2, 0.5
        command.vel_command_b[:] = torch.tensor([speed, 0.0, yaw_rate], device=env.device)
        start_heading, start_phase = command.reference_heading.clone(), command.gait_phase.clone()
        for _ in range(STEPS):
            env.step(zero_action)
        torch.testing.assert_close(
            command.reference_heading - start_heading, torch.full_like(start_heading, yaw_rate * elapsed)
        )
        target = command.target_velocity_w
        torch.testing.assert_close(torch.linalg.norm(target, dim=-1), torch.full_like(start_heading, speed))
        torch.testing.assert_close(
            torch.atan2(target[:, 1], target[:, 0]),
            torch.remainder(command.reference_heading + torch.pi, 2 * torch.pi) - torch.pi,
        )

        # the clock runs at the configured frequency and the observation is its sine and cosine
        frequency = (
            command.cfg.base_frequency + command.cfg.speed_frequency * speed + command.cfg.turn_frequency * yaw_rate
        )
        phase_advance = torch.remainder(command.gait_phase - start_phase, 1.0)
        torch.testing.assert_close(phase_advance, torch.full_like(phase_advance, frequency * elapsed % 1.0))
        angle = 2 * torch.pi * command.gait_phase
        torch.testing.assert_close(command.gait_clock, torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1))
    finally:
        env.close()
        SimulationContext.clear_instance()
