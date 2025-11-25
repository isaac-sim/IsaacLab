# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Iterable

from isaaclab.assets import Articulation

from isaaclab_tasks.manager_based.box_pushing.box_pushing_env import BoxPushingEnv
from isaaclab_tasks.utils.mp import RawMPInterface, upgrade


class BoxPushingMPWrapper(RawMPInterface):
    """Task-specific MP wrapper exposing robot state and observation mask."""

    mp_config = {
        "ProDMP": {
            "black_box_kwargs": {
                "verbose": 1,
                "reward_aggregation": torch.sum,
            },
            "controller_kwargs": {
                "p_gains": 0.01 * torch.tensor([120.0, 120.0, 120.0, 120.0, 50.0, 30.0, 10.0]),
                "d_gains": 0.01 * torch.tensor([10.0, 10.0, 10.0, 10.0, 6.0, 5.0, 3.0]),
            },
            "trajectory_generator_kwargs": {
                "weights_scale": 0.3,
                "goal_scale": 0.3,
                "auto_scale_basis": True,
                "relative_goal": False,
                "disable_goal": False,
            },
            "basis_generator_kwargs": {
                "num_basis": 8,
                "basis_bandwidth_factor": 3,
                "num_basis_outside": 0,
                "alpha": 10,
            },
            "phase_generator_kwargs": {
                "phase_generator_type": "exp",
                "tau": 2.0,
                "alpha_phase": 3.0,
            },
        }
    }

    def __init__(self, env: BoxPushingEnv):
        super().__init__(env)
        self._device = getattr(env, "device", torch.device("cpu"))

    @property
    def context_mask(self) -> torch.Tensor:
        # policy obs is concatenated: joint_pos (7), joint_vel (7), object_pose (7), target_object_pose (7)
        mask = [True] * 7 + [False] * 7 + [True] * 7 + [True] * 7
        return torch.tensor(mask, dtype=torch.bool, device=self._device)

    @property
    def current_pos(self) -> torch.Tensor:
        scene = self.env.unwrapped.scene
        asset: Articulation = scene["robot"]
        return asset.data.joint_pos[:, :7]

    @property
    def current_vel(self) -> torch.Tensor:
        scene = self.env.unwrapped.scene
        asset: Articulation = scene["robot"]
        return asset.data.joint_vel[:, :7]

    @property
    def action_bounds(self):
        """Use normalized bounds for step actions (pre-scale)."""
        dim = getattr(self.env, "action_manager", None).total_action_dim if hasattr(self.env, "action_manager") else 7
        low = -torch.ones(dim, device=self._device)
        high = torch.ones(dim, device=self._device)
        return low, high

    def preprocessing_and_validity_callback(
        self,
        action: torch.Tensor,
        pos_traj: torch.Tensor,
        vel_traj: torch.Tensor,
        tau_bound: Iterable | None = None,
        delay_bound: Iterable | None = None,
    ):
        # Simple sanity: invalidate trajectories with NaNs or infs.
        is_valid = not (torch.isnan(pos_traj).any() or torch.isinf(pos_traj).any())
        return is_valid, pos_traj, vel_traj


def register_box_pushing_mp_env(
    reward_type: str = "Dense",
    mp_type: str = "ProDMP",
    device: str | torch.device = "cuda:0",
    base_variant: str = "step",
    mp_id: str | None = None,
    mp_config_override: dict | None = None,
    env_make_kwargs: dict | None = None,
):
    """Register a gym id for the MP variant of box pushing."""
    base_id = f"Isaac-Box-Pushing-{reward_type}-{base_variant}-Franka-v0"
    fancy_id = mp_id or f"Isaac_MP/Box-Pushing-{reward_type}-{mp_type}-Franka-v0"
    return upgrade(
        mp_id=fancy_id,
        base_id=base_id,
        mp_wrapper_cls=BoxPushingMPWrapper,
        mp_type=mp_type,
        device=device,
        mp_config_override=mp_config_override,
        env_make_kwargs=env_make_kwargs,
    )
