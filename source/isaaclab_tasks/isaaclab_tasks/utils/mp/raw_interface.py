# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch


class RawMPInterface(gym.Wrapper):
    """Base wrapper that exposes the minimal motion primitive hooks to `BlackBoxWrapper`.

    The wrapper only wires data between the MP rollout and the task; it does not change
    any behavior on its own. Subclasses must provide `context_mask`, `current_pos`, and
    `current_vel` so the MP stack can build observations and condition trajectories.
    Unless stated otherwise, methods should return torch tensors that live on the same
    device as the underlying environment buffers (usually `env.device`). Override the
    preprocessing callbacks to validate or alter trajectories before rollout without
    touching the core algorithm.
    """

    @property
    def context_mask(self) -> torch.Tensor:
        """Boolean mask over the policy observation; defaults to using all entries.

        Returns:
            torch.Tensor: 1D mask shaped `(obs_dim,)`, on `env.device` when available.

        Notes:
            Subclasses should override to select only the observation fields that are
            valid MP context. If the underlying env does not expose an observation space,
            the default returns `True` for a single element to avoid shape errors.
        """
        obs_space = getattr(self.env, "observation_space", None)
        device = getattr(self.env, "device", None)
        if obs_space is None:
            return torch.tensor([True], dtype=torch.bool, device=device)
        if isinstance(obs_space, gym.spaces.Box):
            return torch.ones(obs_space.shape[-1], dtype=torch.bool, device=device)
        return torch.tensor([True], dtype=torch.bool, device=device)

    @property
    def current_pos(self) -> float | int | torch.Tensor | tuple:
        """Current position of the controlled dofs.

        Returns:
            float | int | torch.Tensor | tuple: Position broadcastable to `(num_envs, dof)`.

        Notes:
            This is polled each MP step to clamp and condition the next action. The return
            type can be a scalar, tuple, or tensor; `BlackBoxWrapper` will broadcast it.
        """
        raise NotImplementedError

    @property
    def current_vel(self) -> float | int | torch.Tensor | tuple:
        """Current velocity of the controlled dofs.

        Returns:
            float | int | torch.Tensor | tuple: Velocity broadcastable to `(num_envs, dof)`.

        Notes:
            Same shape rules as `current_pos`. If velocities are unavailable, return zeros
            to keep MP conditioning stable.
        """
        raise NotImplementedError

    @property
    def dt(self) -> float:
        """Control interval for the MP rollout.

        Returns:
            float: Step time, preferring IsaacLab's `step_dt`/`dt` attributes; defaults to 0.02s.

        Notes:
            `BlackBoxWrapper` uses this to align trajectory time stamps with the simulator.
            The method inspects the wrapper stack so downstream wrappers can override `step_dt`.
        """
        # Prefer explicit step_dt from IsaacLab envs, otherwise fallback.
        if hasattr(self.env, "step_dt"):
            return float(self.env.step_dt)
        if hasattr(self.env, "dt"):
            return float(self.env.dt)
        return 0.02

    def preprocessing_and_validity_callback(
        self,
        action: torch.Tensor,
        pos_traj: torch.Tensor,
        vel_traj: torch.Tensor,
        tau_bound: list | None = None,
        delay_bound: list | None = None,
    ) -> tuple[bool, torch.Tensor, torch.Tensor]:
        """Validate and optionally modify the generated trajectory before rollout.

        This hook runs once per high-level MP action to reject unstable parameters or
        to project trajectories back into a safe set.

        Args:
            action (torch.Tensor): Raw MP parameters shaped `(batch, param_dim)` on `env.device`.
            pos_traj (torch.Tensor): Planned positions shaped `(batch, horizon, dof)`.
            vel_traj (torch.Tensor): Planned velocities shaped `(batch, horizon, dof)`.
            tau_bound (list | None): Phase time-scale bounds passed through from the trajectory
                generator; used only for context.
            delay_bound (list | None): Delay bounds from the phase generator; used only for context.

        Returns:
            tuple[bool, torch.Tensor, torch.Tensor]: `(is_valid, pos_traj, vel_traj)` where
            `is_valid=False` triggers `invalid_traj_callback`. The default marks every
            trajectory valid and forwards inputs unchanged.

        Side Effects:
            None. Override to log diagnostics or clamp trajectories in-place if needed.
        """
        return True, pos_traj, vel_traj

    def set_episode_arguments(
        self, action: torch.Tensor, pos_traj: torch.Tensor, vel_traj: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inject per-episode arguments or modify trajectories before rollout.

        Args:
            action (torch.Tensor): MP parameters shaped `(batch, param_dim)`.
            pos_traj (torch.Tensor): Planned positions `(batch, horizon, dof)`.
            vel_traj (torch.Tensor): Planned velocities `(batch, horizon, dof)`.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Possibly modified `(pos_traj, vel_traj)`.

        Notes:
            This runs after MP parameter clamping but before validity checks. Use it to
            stash episode metadata or to align trajectories with additional state.
        """
        return pos_traj, vel_traj

    def invalid_traj_callback(
        self,
        action: torch.Tensor,
        pos_traj: torch.Tensor,
        vel_traj: torch.Tensor,
        tau_bound: list | None,
        delay_bound: list | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Return a fallback transition when the trajectory is rejected.

        Args:
            action (torch.Tensor): Rejected MP parameters `(batch, param_dim)`.
            pos_traj (torch.Tensor): Planned positions `(batch, horizon, dof)`.
            vel_traj (torch.Tensor): Planned velocities `(batch, horizon, dof)`.
            tau_bound (list | None): Phase time-scale bounds supplied by the phase generator.
            delay_bound (list | None): Delay bounds supplied by the phase generator.

        Returns:
            tuple: `(obs, reward, terminated, truncated, info)` tensors broadcast to
            `num_envs`. The default returns zero observation and reward, sets `terminated`
            to `True` to stop the episode, and `truncated` to `False`.

        Notes:
            Override to provide task-specific recovery, e.g., penalizing invalid MPs or
            resetting the simulator. `obs` shape follows the policy observation space if
            present; otherwise `(num_envs, 1)`.
        """
        num_envs = int(pos_traj.shape[0]) if hasattr(pos_traj, "shape") else getattr(self.env, "num_envs", 1)
        obs_space = getattr(self, "observation_space", None) or getattr(self.env, "observation_space", None)
        obs_shape = (1,)
        if (
            isinstance(obs_space, gym.spaces.Dict)
            and "policy" in obs_space.spaces
            and isinstance(obs_space.spaces["policy"], gym.spaces.Box)
        ):
            obs_shape = obs_space.spaces["policy"].shape
        elif isinstance(obs_space, gym.spaces.Box):
            obs_shape = obs_space.shape

        zeros = torch.zeros((num_envs,) + tuple(obs_shape), device=pos_traj.device)
        reward = torch.zeros((num_envs,), device=pos_traj.device)
        terminated = torch.ones((num_envs,), dtype=torch.bool, device=pos_traj.device)
        truncated = torch.zeros((num_envs,), dtype=torch.bool, device=pos_traj.device)
        return zeros, reward, terminated, truncated, {}

    def episode_callback(self, action: torch.Tensor, pos_traj: torch.Tensor, vel_traj: torch.Tensor) -> tuple[bool]:
        """Split MP parameters from auxiliary parameters if needed.

        Args:
            action (torch.Tensor): MP parameters `(batch, param_dim)`.
            pos_traj (torch.Tensor): Planned positions `(batch, horizon, dof)`.
            vel_traj (torch.Tensor): Planned velocities `(batch, horizon, dof)`.

        Returns:
            tuple[bool]: A single-element tuple with `True` by default; subclasses can
            return extra flags to steer downstream logic.
        """
        return True

    def get_wrapper_attr(self, name: str, default=None):
        """Traverse the wrapper stack to retrieve an attribute.

        Args:
            name (str): Attribute to look up.
            default (Any): Value to return when the attribute is absent.

        Returns:
            Any: First matching attribute found walking through `self.env` chain.

        Notes:
            This mirrors `_find_attr_across_wrappers` in `BlackBoxWrapper` and lets MP
            utilities read attributes exposed by deeper wrappers without coupling to a
            specific wrapper order.
        """
        head = self
        while head is not None:
            if hasattr(head, name):
                return getattr(head, name)
            if not hasattr(head, "env"):
                break
            head = head.env
        return default

    @property
    def action_bounds(self):
        """Optional tuple of `(low, high)` tensors used to clamp step actions.

        Returns:
            tuple[torch.Tensor, torch.Tensor] | None: Bounds broadcastable to the step
            action shape, or `None` to fall back to `env.action_space`.

        Notes:
            `BlackBoxWrapper` clamps controller outputs against these bounds before
            stepping the env. Override to guard against unsafe torques when the env's
            `action_space` is unbounded.
        """
        return getattr(self, "_action_bounds", None)
