# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Local basket pose adapter preserving DLS direction before existing action limits.

Pickup retains the maintained controller. Pour and return use the same IK, TCP
Jacobian, EMA inversion and gripper commands, changing only joint-step limiting.
Final raw-action clipping remains unchanged and can still change the direction.
"""

from __future__ import annotations

import torch

from isaaclab.utils import math as math_utils

from .pose_control import PoseController

OPTIONS = {
    "algorithm": "basket_dls_uniform_joint_increment_scaling_v1",
    "scope": "measured basket pour and return only; maintained pickup unchanged",
    "maximum_joint_increment_rad": 0.012,
    "ik": "inherited PoseController DLS and TCP-offset Jacobian",
    "limiting": "delta *= min(1, 0.012 / max(abs(delta))) independently per world",
    "action_filter": "unchanged live EMA inversion followed by raw [-1, 1] clamp",
    "rationale": "Closed S2 seed 70042: elementwise joint clipping reversed the TCP correction near a singularity.",
    "physical_validation_passed": False,
}


def basket_joint_increment(delta: torch.Tensor) -> torch.Tensor:
    """Limit each world's joint increment [rad], shape [..., 7], by one positive scale.

    Zero and unsaturated inputs are preserved. The input tensor is not modified.
    """
    limit = OPTIONS["maximum_joint_increment_rad"]
    peak = delta.abs().amax(dim=-1, keepdim=True)
    return torch.where(peak > limit, (delta / peak.clamp_min(limit)) * limit, delta)


class BasketPoseController(PoseController):
    """Use the maintained pose/action mapping with uniformly limited joint steps."""

    def compute(self, position: torch.Tensor, rotation: torch.Tensor, close: bool | torch.Tensor) -> torch.Tensor:
        """Convert world TCP position [m], XYZW rotation and gripper commands to raw actions."""
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
        delta = basket_joint_increment(goal - joints)
        raw = (delta - (1 - alpha) * term.processed_actions) / (alpha * env.cfg.actions.arm_action.scale)
        actions = torch.zeros((env.num_envs, 8), device=env.device)
        actions[:, :7] = raw.clamp(-1, 1)
        actions[:, -1] = torch.where(torch.as_tensor(close, device=env.device), -1.0, 1.0)
        return actions
