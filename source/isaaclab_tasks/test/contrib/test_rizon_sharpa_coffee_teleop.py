# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tracking acquisition, bounded motion and re-clutch behavior without XR hardware."""

from types import SimpleNamespace

import torch

from isaaclab_tasks.contrib.rizon_sharpa_coffee.teleop import CoffeeTeleopController


def test_wrist_acquisition_recenter_and_tracking_loss():
    palm = torch.tensor([[0.45, -0.1, 0.2, 0.0, 0.0, 0.0, 1.0]])
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=0.01,
        palm_pose=lambda: palm,
        cfg=SimpleNamespace(grasp_reset_joint_targets=None),
    )
    controller = CoffeeTeleopController(env)
    sample = torch.zeros(1, 29)
    sample[:, :7] = torch.tensor([1.2, 0.5, 1.1, 0.0, 0.0, 0.0, 1.0])
    acquired = controller.compute(sample)
    torch.testing.assert_close(acquired[:, :7], palm)

    sample[:, 0] += 0.02
    sample[:, 7:] = 10.0
    moved = controller.compute(sample)
    torch.testing.assert_close(moved[:, :3], torch.tensor([[0.4525, -0.1, 0.2]]))
    assert torch.isfinite(moved).all()
    assert moved[:, 7:].max() < 2.0
    lost = sample.clone()
    lost[:, :7] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    torch.testing.assert_close(controller.compute(lost), moved)

    sample[:, 3:7] = torch.tensor([0.0, 0.0, 1.0, 0.0])
    rotated = controller.compute(sample)
    angle = 2.0 * torch.atan2(rotated[:, 3:6].norm(dim=-1), rotated[:, 6].abs())
    torch.testing.assert_close(angle, torch.tensor([0.012]))
    palm[:, 0] = 0.6
    controller.reset()
    torch.testing.assert_close(controller.compute(sample)[:, :7], palm)
