# Copyright (c) 2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.app import AppLauncher

HEADLESS = True

# if not AppLauncher.instance():
simulation_app = AppLauncher(headless=HEADLESS).app

"""Rest of imports follows"""

import torch

import pytest

from isaaclab.actuators import IdealPDActuatorCfg


@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize("num_joints", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_init_drive_model(num_envs, num_joints, device):
    joint_names = [f"joint_{d}" for d in range(num_joints)]
    joint_ids = [d for d in range(num_joints)]
    stiffness = 200
    damping = 10
    effort_limit = 60.0
    velocity_limit = 50
    drive_model = IdealPDActuatorCfg.DriveModelCfg(
        speed_effort_gradient=100.0,
        max_actuator_velocity=200.0,
        velocity_dependent_resistance=0.1,
    )

    actuator_cfg = IdealPDActuatorCfg(
        joint_names_expr=joint_names,
        stiffness=stiffness,
        damping=damping,
        effort_limit=effort_limit,
        velocity_limit=velocity_limit,
        drive_model=drive_model,
    )
    # assume Articulation class:
    #   - finds joints (names and ids) associate with the provided joint_names_expr

    actuator = actuator_cfg.class_type(
        actuator_cfg,
        joint_names=joint_names,
        joint_ids=joint_ids,
        num_envs=num_envs,
        device=device,
    )

    # check device and shape
    torch.testing.assert_close(
        actuator.drive_model[:, :, 2],
        drive_model.velocity_dependent_resistance * torch.ones(num_envs, num_joints, device=device),
    )
    torch.testing.assert_close(
        actuator.drive_model[:, :, 1],
        drive_model.max_actuator_velocity * torch.ones(num_envs, num_joints, device=device),
    )
    torch.testing.assert_close(
        actuator.drive_model[:, :, 0],
        drive_model.speed_effort_gradient * torch.ones(num_envs, num_joints, device=device),
    )
