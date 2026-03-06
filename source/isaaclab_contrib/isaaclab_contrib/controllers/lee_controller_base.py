# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for Lee-style geometric controllers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.assets import Multirotor

    from .lee_controller_base_cfg import LeeControllerBaseCfg


class LeeControllerBase:
    """Base class for Lee-style geometric controllers."""

    cfg: LeeControllerBaseCfg
    device: str
    robot: Multirotor

    def __init__(self, cfg: LeeControllerBaseCfg, asset: Multirotor, num_envs: int, device: str):
        """Initialize controller buffers and pre-compute aggregate inertias.

        Args:
            cfg: Controller configuration.
            asset: Multirotor asset to control.
            num_envs: Number of environments.
            device: Device to run computations on.
        """
        self.cfg = cfg
        self.robot = asset
        self.device = device
        self.num_envs = num_envs

        # Aggregate mass and inertia about the robot COM for all bodies
        root_quat_exp = self.robot.data.root_link_quat_w.unsqueeze(1).expand(num_envs, self.robot.num_bodies, 4)
        body_link_pos_delta = self.robot.data.body_link_pos_w - self.robot.data.root_pos_w.unsqueeze(1)
        self.mass, self.robot_inertia, _ = math_utils.aggregate_inertia_about_robot_com(
            self.robot.root_physx_view.get_inertias().to(device),
            self.robot.root_physx_view.get_inv_masses().to(device),
            self.robot.data.body_com_pos_b,
            self.robot.data.body_com_quat_b,
            math_utils.quat_apply_inverse(root_quat_exp, body_link_pos_delta),
            math_utils.quat_mul(math_utils.quat_inv(root_quat_exp), self.robot.data.body_link_quat_w),
        )
        # Get gravity from simulation context
        sim = sim_utils.SimulationContext.instance()
        gravity_vec = sim.cfg.gravity
        self.gravity = torch.tensor(gravity_vec, device=device, dtype=torch.float32).expand(num_envs, -1)

        # Buffers
        self.wrench_command_b = torch.zeros((num_envs, 6), device=device)  # [fx, fy, fz, tx, ty, tz]
        self.rotation_matrix_buffer = torch.zeros((num_envs, 3, 3), device=device)

    def reset(self):
        """Reset controller state for all environments."""
        self.reset_idx(env_ids=None)

    def reset_idx(self, env_ids: torch.Tensor | None):
        """Reset controller state (and optionally randomize gains) for selected environments.

        Args:
            env_ids: Tensor of environment indices, or ``None`` for all.
        """
        if env_ids is None:
            env_ids = slice(None)
        self._randomize_params(env_ids)

    def _randomize_params(self, env_ids: slice | torch.Tensor):
        """Randomize controller gains for the given environments if enabled.

        Override in subclass to implement parameter randomization.
        """
        pass

    def compute(self, command: torch.Tensor) -> torch.Tensor:
        """Compute wrench command from input command.

        Args:
            command: Input command (shape depends on controller type).

        Returns:
            (num_envs, 6) wrench command [fx, fy, fz, tx, ty, tz] in body frame.
        """
        raise NotImplementedError("Subclasses must implement compute()")
