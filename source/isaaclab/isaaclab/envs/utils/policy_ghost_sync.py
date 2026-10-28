# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keeps ghost slices of one environment on the reference slice's terrain, spawn state and commands."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from ..manager_based_rl_env import ManagerBasedRLEnv
    from ..manager_based_rl_env_cfg import ManagerBasedRLEnvCfg

logger = logging.getLogger(__name__)


class PolicyGhostSync:
    """Synchronizes the slices of a single environment so that several policies can be compared.

    The environment is simulated with ``num_slices * slice_size`` worlds. Slice 0 is the reference; environment
    ``i`` of every other slice mirrors environment ``i`` of the reference. Whenever a reference environment resets,
    its ghost counterparts reset with it and receive the reference's spawn state. Velocity or pose commands are
    copied from the reference after every command update. Ghost environments that terminate on their own wait
    for the reference to reset.

    Only :class:`~isaaclab.envs.ManagerBasedRLEnv` is supported. Call :meth:`configure_env_cfg` on the env config
    before creating the environment, because per-environment randomization other than the reset events cannot
    be mirrored.
    """

    def __init__(self, env: ManagerBasedRLEnv, slice_size: int, num_slices: int):
        """Attach to an environment and place every slice on the reference slice's terrain.

        Args:
            env: Environment simulated with at least ``slice_size * num_slices`` worlds. Worlds beyond the slices
                are left alone: their resets are ignored, so they stay parked.
            slice_size: Number of environments per slice.
            num_slices: Number of slices, including the reference.

        Raises:
            TypeError: If the environment does not use managers.
            ValueError: If the environment has fewer worlds than the slices need.
        """
        for manager in ("command_manager", "event_manager", "scene"):
            if not hasattr(env, manager):
                raise TypeError("Policy Ghosts only support manager-based environments.")
        if slice_size < 1 or num_slices < 2 or env.num_envs < slice_size * num_slices:
            raise ValueError(
                f"Environment has {env.num_envs} envs, expected at least {num_slices} slices of {slice_size} envs."
            )
        self._env = env
        self._slice_size = slice_size
        self._num_slices = num_slices
        self._device = env.device
        self._reference_ids = torch.arange(slice_size, device=self._device)
        self._ghost_offsets = torch.arange(1, num_slices, device=self._device) * slice_size
        self._saved_terrain = self._save_terrain()
        self._sync_terrain()
        self._reset_idx_original = env._reset_idx
        self._compute_commands_original = env.command_manager.compute
        env._reset_idx = self._reset_idx
        env.command_manager.compute = self._compute_commands

    @staticmethod
    def configure_env_cfg(env_cfg: ManagerBasedRLEnvCfg) -> list[str]:
        """Remove per-environment randomization that cannot be mirrored between slices.

        Startup and interval events, curriculum terms, and observation corruption are disabled. Reset events
        are kept because the reference's reset state is copied to the ghosts.

        Args:
            env_cfg: Environment config to modify in place.

        Returns:
            Names of the disabled terms.
        """
        removed = []
        # A task may declare no events, curriculum, or observation groups at all.
        events, curriculum, observations = (
            getattr(env_cfg, name, None) for name in ("events", "curriculum", "observations")
        )
        for name, term in vars(events).items() if events is not None else ():
            if term is not None and getattr(term, "mode", None) in ("startup", "interval"):
                setattr(events, name, None)
                removed.append(f"events.{name}")
        for name, term in vars(curriculum).items() if curriculum is not None else ():
            if term is not None:
                setattr(curriculum, name, None)
                removed.append(f"curriculum.{name}")
        for group_name, group in vars(observations).items() if observations is not None else ():
            if getattr(group, "enable_corruption", False):
                group.enable_corruption = False
                removed.append(f"observations.{group_name}.enable_corruption")
        if removed:
            logger.warning("Policy Ghosts disabled per-environment randomization: %s", ", ".join(removed))
        return removed

    def env_origin_offsets(self) -> np.ndarray:
        """Return per-environment offsets [m], shape ``[num_envs, 3]``, that move each ghost onto the reference."""
        origins = self._env.scene.env_origins
        offsets = torch.zeros_like(origins)
        for ghost_start in self._ghost_offsets.tolist():
            offsets[ghost_start : ghost_start + self._slice_size] = (
                origins[: self._slice_size] - origins[ghost_start : ghost_start + self._slice_size]
            )
        return offsets.cpu().numpy().astype(np.float32)

    def close(self) -> None:
        """Restore the environment's original reset, command update, and terrain placement."""
        self._env._reset_idx = self._reset_idx_original
        self._env.command_manager.compute = self._compute_commands_original
        terrain = self._env.scene.terrain
        for name, values in self._saved_terrain.items():
            getattr(terrain, name)[: len(values)] = values

    def _save_terrain(self) -> dict[str, torch.Tensor]:
        """Copy the terrain placement of every environment, which :meth:`_sync_terrain` overwrites for the ghosts."""
        terrain = self._env.scene.terrain
        if terrain is None or getattr(terrain, "terrain_origins", None) is None:
            return {}
        return {name: getattr(terrain, name).clone() for name in ("terrain_levels", "terrain_types", "env_origins")}

    def _sync_terrain(self) -> None:
        """Place every ghost environment on the terrain tile of its reference environment."""
        terrain = self._env.scene.terrain
        if terrain is None or getattr(terrain, "terrain_origins", None) is None:
            return
        for name in ("terrain_levels", "terrain_types", "env_origins"):
            values = getattr(terrain, name)
            for ghost_start in self._ghost_offsets.tolist():
                values[ghost_start : ghost_start + self._slice_size] = values[: self._slice_size]

    def _reset_idx(self, env_ids: Any) -> None:
        """Reset reference environments together with their ghosts, then copy the reference's spawn state."""
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self._env.num_envs, device=self._device)[env_ids]
        env_ids = torch.as_tensor(env_ids, device=self._device, dtype=torch.long)
        reference = env_ids[env_ids < self._slice_size]
        if reference.numel() == 0:
            # ghost-only terminations wait for the reference to reset
            return
        ghosts = (reference[None, :] + self._ghost_offsets[:, None]).reshape(-1)
        self._reset_idx_original(torch.cat([reference, ghosts]))
        scene = self._env.scene
        state = scene.get_state(is_relative=True)
        scene.reset_to(_gather(state, reference.repeat(self._num_slices - 1)), env_ids=ghosts, is_relative=True)
        self._copy_command_state(reference)

    def _compute_commands(self, *args: Any, **kwargs: Any) -> None:
        """Update commands, then copy the reference's commands to the ghosts."""
        self._compute_commands_original(*args, **kwargs)
        self._copy_command_state(self._reference_ids)

    def _copy_command_state(self, reference: torch.Tensor) -> None:
        """Copy every per-environment tensor of each command term from the reference to its ghosts."""
        ghosts = (reference[None, :] + self._ghost_offsets[:, None]).reshape(-1)
        source = reference.repeat(self._num_slices - 1)
        manager = self._env.command_manager
        for name in manager.active_terms:
            for value in vars(manager.get_term(name)).values():
                if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == self._env.num_envs:
                    value[ghosts] = value[source]


def _gather(state: dict[str, Any], ids: torch.Tensor) -> dict[str, Any]:
    """Index every tensor of a nested scene-state dictionary by environment id."""
    return {key: _gather(value, ids) if isinstance(value, dict) else value[ids] for key, value in state.items()}
