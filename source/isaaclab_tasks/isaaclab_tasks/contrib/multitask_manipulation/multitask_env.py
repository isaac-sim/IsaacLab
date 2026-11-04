# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based environment with selection-aware heterogeneous resets."""

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.warp import ProxyArray

from .selection_utils import SceneEntitySelectionCfg


class MultitaskManipulationEnv(ManagerBasedRLEnv):
    """Manager-based manipulation environment whose assets occupy partial physics views."""

    def _reset_mask(self, env_mask: ProxyArray) -> None:
        """Reset global environments through each asset's view-row mapping.

        Args:
            env_mask: Boolean mask of the global environments to reset. Shape is (num_envs,).
        """
        self.curriculum_manager.compute(env_mask=env_mask)
        for asset_name, asset in (*self.scene.articulations.items(), *self.scene.rigid_objects.items()):
            # select the asset's physics-view rows that belong to the reset environments
            asset.reset(env_mask=self._get_entity_selection(asset_name).view_row_mask(env_mask))

        if "reset" in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.cfg.decimation
            self.event_manager.apply(mode="reset", env_mask=env_mask, global_env_step_count=env_step_count)

        managers = (
            self.observation_manager,
            self.action_manager,
            self.reward_manager,
            self.curriculum_manager,
            self.command_manager,
            self.event_manager,
            self.termination_manager,
            self.recorder_manager,
        )
        self._reset_managers(env_mask, managers)

        self.episode_length_buf.masked_fill_(env_mask.torch, 0)

    def _get_entity_selection(self, asset_name: str) -> SceneEntitySelectionCfg:
        """Return the cached selection configuration for a complete scene asset."""
        cache: dict[str, SceneEntitySelectionCfg] = self.__dict__.setdefault("_scene_entity_selections", {})
        if asset_name not in cache:
            cache[asset_name] = SceneEntitySelectionCfg(asset_name)
            cache[asset_name].resolve(self.scene)
        return cache[asset_name]
