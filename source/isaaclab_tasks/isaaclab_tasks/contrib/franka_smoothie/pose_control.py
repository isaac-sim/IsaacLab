# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TCP pose control and the scripted basket-pickup candidate used by the tap controller."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from .smoothie_env import SmoothieBlenderEnv


class PoseController:
    """Map TCP pose commands to the filtered joint actions the task's action terms expect."""

    def __init__(self, env: SmoothieBlenderEnv) -> None:
        self.env = env
        self.joint_ids = env.robot.find_joints([f"panda_joint{i}" for i in range(1, 8)], preserve_order=True)[0]
        self.controller = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            env.num_envs,
            env.device,
        )

    def compute(self, position: torch.Tensor, rotation: torch.Tensor, close: bool | torch.Tensor) -> torch.Tensor:
        """Convert world TCP position [m], XYZW rotation, and per-world gripper commands to actions."""
        env = self.env
        hand = env.robot.data.body_link_pose_w.torch[:, env.hand_id]
        jacobian = env.robot.data.body_link_jacobian_w.torch[:, env.hand_id - 1, :, self.joint_ids].clone()
        offset = env.tcp() - hand[:, :3]
        jacobian[:, :3] -= torch.bmm(math_utils.skew_symmetric_matrix(offset), jacobian[:, 3:])
        joints = env.robot.data.joint_pos.torch[:, self.joint_ids]
        self.controller.set_command(torch.cat((position, rotation), -1))
        goal = self.controller.compute(env.tcp(), hand[:, 3:], jacobian, joints)
        term = env.action_manager.get_term("arm_action")
        alpha = env.cfg.actions.arm_action.alpha
        delta = (goal - joints).clamp(-0.012, 0.012)
        raw = (delta - (1 - alpha) * term.processed_actions) / (alpha * env.cfg.actions.arm_action.scale)
        actions = torch.zeros((env.num_envs, 8), device=env.device)
        actions[:, :7] = raw.clamp(-1, 1)
        actions[:, -1] = torch.where(torch.as_tensor(close, device=env.device), -1.0, 1.0)
        return actions


class BasketExpert:
    """Scripted approach/close/lift candidate using physical contacts and policy actions.

    Success must be measured by the environment. This controller does not attach,
    teleport, or prescribe motion to the basket, and does not demonstrate pouring.
    """

    def __init__(self, env: SmoothieBlenderEnv) -> None:
        self.env = env
        self.controller = PoseController(env)
        self.position, self.rotation = env.grasp_pose()

    def compute(self, step: int | torch.Tensor) -> torch.Tensor:
        """Return actions for scalar or per-world episode steps, restarting each world at step zero."""
        env = self.env
        steps = torch.as_tensor(step, device=env.device).expand(env.num_envs)
        time_s = steps * env.step_dt
        close_step = math.ceil(7.5 / env.step_dt)
        approach = steps < close_step
        position, rotation = env.grasp_pose()
        position[:, 2] += 0.15 * (1 - ((time_s - 3.0) / 3.5).clamp(0, 1))
        self.position[approach] = position[approach]
        self.rotation[approach] = rotation[approach]
        closing = steps == close_step
        self.position[closing] = env.tcp()[closing]
        target = self.position.clone()
        target[:, 2] += 0.15 * ((time_s - 11.0) / 3.5).clamp(0, 1)
        return self.controller.compute(target, self.rotation, steps >= close_step)
