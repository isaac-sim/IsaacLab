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
    """Expose box-pushing robot state to the MP pipeline with task-aware defaults.

    The wrapper supplies a context mask that selects joint positions and object poses
    from the policy observation, and provides joint-level state for conditioning. It
    assumes a 7-DoF Franka arm with the robot labeled `"robot"` in the scene. The class
    ships MP defaults tuned for Franka box pushing; `registry._merge_mp_config` injects
    the runtime device so users do not need to set it manually.
    """

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
        """Select joint positions and object poses from the policy observation.

        Returns:
            torch.Tensor: Boolean mask `(28,)` marking `[q, qd, obj_pose, target_pose]`
            fields as `[1, 0, 1, 1]` chunks on `self._device`.
        """
        # policy obs is concatenated: joint_pos (7), joint_vel (7), object_pose (7), target_object_pose (7)
        mask = [True] * 7 + [False] * 7 + [True] * 7 + [True] * 7
        return torch.tensor(mask, dtype=torch.bool, device=self._device)

    @property
    def current_pos(self) -> torch.Tensor:
        """Return the 7D arm joint positions for each environment."""
        scene = self.env.unwrapped.scene
        asset: Articulation = scene["robot"]
        return asset.data.joint_pos[:, :7]

    @property
    def current_vel(self) -> torch.Tensor:
        """Return the 7D arm joint velocities for each environment."""
        scene = self.env.unwrapped.scene
        asset: Articulation = scene["robot"]
        return asset.data.joint_vel[:, :7]

    @property
    def action_bounds(self):
        """Use normalized bounds for step actions (pre-scale).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: `(low, high)` each shaped `(action_dim,)`
            with values in `[-1, 1]` on `self._device`.

        Notes:
            `BlackBoxWrapper` clamps controller outputs to these bounds before passing
            them to the env. Bounds are sized using the action manager when present to
            remain compatible with scaled or concatenated action spaces.
        """
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
        """Reject NaN/inf trajectories before rollout.

        Args:
            action (torch.Tensor): MP parameters `(batch, param_dim)` on device.
            pos_traj (torch.Tensor): Planned positions `(batch, horizon, 7)`.
            vel_traj (torch.Tensor): Planned velocities `(batch, horizon, 7)`.
            tau_bound (Iterable | None): Unused phase bounds.
            delay_bound (Iterable | None): Unused delay bounds.

        Returns:
            tuple[bool, torch.Tensor, torch.Tensor]: Validity flag and unmodified
            trajectories. Invalid plans trigger `invalid_traj_callback` with terminated
            episodes so the agent learns to avoid pathological parameters.
        """
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
    """Register a Gym id for the MP variant of box pushing.

    Args:
        reward_type (str): Reward flavor of the base env (e.g., `"Dense"`).
        mp_type (str): MP key (defaults to `"ProDMP"`) used for config merging.
        device (str | torch.device): Device propagated into all MP components.
        base_variant (str): Base env variant; `"step"` matches step-based control.
        mp_id (str | None): Optional custom Gym id; defaults to Fancy Gym naming.
        mp_config_override (dict | None): Overrides applied after wrapper defaults.
        env_make_kwargs (dict | None): Extra kwargs forwarded to base env creation.

    Returns:
        str: Registered Gym id for the MP-enabled box pushing environment.

    Notes:
        The helper delegates to `upgrade`, which builds the MP stack via `make_mp_env`.
        Device is injected during merge, so `mp_config` does not need an explicit device.
    """
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
