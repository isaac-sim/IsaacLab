# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise the lightweight backend at the stock manager environment boundary."""

import pytest
import torch
from isaaclab_experimental.real.example import make_env_cfg

from isaaclab.envs import ManagerBasedEnv

pytestmark = pytest.mark.kitless


def test_stock_manager_loop_with_deployment_session():
    """Map reordered actions to measured joints, preserve telemetry views, and reset without teleporting."""
    env = ManagerBasedEnv(make_env_cfg())
    session = env.sim.backend_manager.session
    try:
        robot = env.scene["robot"]
        position_view = robot.data.joint_pos.torch
        initial, _ = env.reset()
        torch.testing.assert_close(initial["policy"], torch.zeros(1, 4))

        # Action order is elbow/shoulder; the session order is shoulder/elbow.
        # Scale=0.5 and default offsets turn [0.6, -0.8] into targets [-0.3, 0.1].
        action = torch.tensor([[0.6, -0.8]])
        first, _ = env.step(action)
        torch.testing.assert_close(robot.data.joint_pos_target.torch, torch.tensor([[-0.3, 0.1]]))
        torch.testing.assert_close(first["policy"], torch.tensor([[-0.02, 0.02, -0.5, 0.5]]))
        torch.testing.assert_close(position_view, torch.tensor([[0.08, -0.18]]))
        assert robot.timestamp == pytest.approx(0.04)

        # Episode reset discards this unsubmitted command; it does not cancel the accepted target.
        robot.set_joint_position_target(torch.tensor([[0.5, -0.7]]))
        env.reset()
        torch.testing.assert_close(position_view, torch.tensor([[0.08, -0.18]]))
        assert robot.timestamp == pytest.approx(0.04)
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)
        torch.testing.assert_close(position_view, torch.tensor([[0.07, -0.17]]))

        for _ in range(29):
            env.step(action)
        torch.testing.assert_close(session.read().position, torch.tensor([-0.3, 0.1]))
        torch.testing.assert_close(position_view, torch.tensor([[-0.3, 0.1]]))
        assert robot.timestamp == pytest.approx(1.22)

        env.reset()
        torch.testing.assert_close(position_view, torch.tensor([[-0.3, 0.1]]))
        assert robot.timestamp == pytest.approx(1.22)
    finally:
        env.close()
    with pytest.raises(RuntimeError, match="closed"):
        session.read()
