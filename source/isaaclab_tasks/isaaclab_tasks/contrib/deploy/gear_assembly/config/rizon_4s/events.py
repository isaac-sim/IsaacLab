# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-local event terms for the Rizon 4s gear-assembly environment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import EventTermCfg, ManagerTermBase

from isaaclab_tasks.contrib.deploy.mdp.events import (
    _compose_child_pose_from_parent,
    _get_child_pose_in_parent_frame,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


_GEAR_ASSET_NAMES = ("factory_gear_small", "factory_gear_medium", "factory_gear_large")
_GEAR_KEYS = ("gear_small", "gear_medium", "gear_large")


class pin_unselected_gears_to_shafts(ManagerTermBase):
    """Keep each non-selected gear seated on its shaft.

    Static asset references, relative orientations, and write buffers are cached because this
    term runs after every control step for Newton gear-assembly environments.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the cached task event.

        Args:
            cfg: Event-term configuration.
            env: Environment instance.
        """
        super().__init__(cfg, env)

        if not hasattr(env, "_gear_type_manager"):
            raise RuntimeError(
                "Gear type manager not initialized. Configure randomize_gear_type before "
                "pin_unselected_gears_to_shafts."
            )

        self._gear_type_manager = env._gear_type_manager
        self._base = env.scene["factory_gear_base"]
        self._gears = tuple(env.scene[name] for name in _GEAR_ASSET_NAMES)
        self._all_env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        self._zero_velocity = torch.zeros((env.num_envs, 6), device=env.device)
        self._root_pose_buffers = tuple(torch.empty((env.num_envs, 7), device=env.device) for _ in self._gears)

        gear_offsets = cfg.params["gear_offsets"]
        seated_gear_z_offset = cfg.params.get("seated_gear_z_offset", 0.0)
        self._seated_offsets = torch.tensor(
            [gear_offsets[key] for key in _GEAR_KEYS], device=env.device, dtype=torch.float32
        )
        self._seated_offsets[:, 2] += seated_gear_z_offset

        base_default = self._base.data.default_root_pose.torch
        relative_quaternions = []
        for gear_index, gear in enumerate(self._gears):
            local_position = self._seated_offsets[gear_index].expand(env.num_envs, 3)
            _, relative_quaternion = _get_child_pose_in_parent_frame(
                base_default, gear.data.default_root_pose.torch, local_position
            )
            relative_quaternions.append(relative_quaternion)
        self._relative_quaternions = torch.stack(relative_quaternions)

    def _resolve_env_ids(self, env_ids: torch.Tensor | Sequence[int] | slice | None) -> torch.Tensor:
        """Return environment IDs on the event device."""
        if env_ids is None:
            return self._all_env_ids
        if isinstance(env_ids, slice):
            return self._all_env_ids[env_ids]
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | Sequence[int] | slice | None,
        gear_offsets: dict[str, Sequence[float]],
        seated_gear_z_offset: float = 0.0,
    ) -> None:
        """Write non-selected gear poses from the current base pose.

        Args:
            env: Environment instance.
            env_ids: Environment IDs to update. If ``None``, all environments are updated.
            gear_offsets: Per-gear shaft offsets in the base frame [m]. Cached at initialization.
            seated_gear_z_offset: Shaft rest-height offset [m]. Cached at initialization.
        """
        del env, gear_offsets, seated_gear_z_offset
        active_env_ids = self._resolve_env_ids(env_ids)
        selected_gear_indices = self._gear_type_manager.get_all_gear_type_indices()
        if env_ids is not None:
            selected_gear_indices = selected_gear_indices[active_env_ids]

        for gear_index, gear in enumerate(self._gears):
            gear_env_ids = active_env_ids[selected_gear_indices != gear_index]
            count = gear_env_ids.shape[0]
            gear_world_pos, gear_world_quat = _compose_child_pose_from_parent(
                self._base.data.root_link_pos_w.torch[gear_env_ids],
                self._base.data.root_link_quat_w.torch[gear_env_ids],
                self._seated_offsets[gear_index].expand(count, 3),
                self._relative_quaternions[gear_index, gear_env_ids],
            )
            root_pose = self._root_pose_buffers[gear_index][:count]
            root_pose[:, :3] = gear_world_pos
            root_pose[:, 3:] = gear_world_quat
            gear.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=gear_env_ids)
            gear.write_root_velocity_to_sim_index(root_velocity=self._zero_velocity[:count], env_ids=gear_env_ids)
