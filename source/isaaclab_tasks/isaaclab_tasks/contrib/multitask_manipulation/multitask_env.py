# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based environment with selection-aware heterogeneous resets."""

from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv

from .selection_utils import SceneEntitySelectionCfg


class MultitaskManipulationEnv(ManagerBasedRLEnv):
    """Manager-based manipulation environment whose assets occupy partial physics views."""

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        """Reset global environments through each asset's view-row mapping.

        Args:
            env_ids: Global environment IDs to reset.
        """
        global_env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        self.curriculum_manager.compute(env_ids=global_env_ids)
        for asset_name, asset in (*self.scene.articulations.items(), *self.scene.rigid_objects.items()):
            rows, _ = self._get_entity_selection(asset_name).select(global_env_ids)
            if rows.numel() > 0:
                asset.reset(rows)

        if "reset" in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.cfg.decimation
            self.event_manager.apply(mode="reset", env_ids=global_env_ids, global_env_step_count=env_step_count)

        self.extras["log"] = {}
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
        for manager in managers:
            self.extras["log"].update(manager.reset(global_env_ids))

        self.episode_length_buf[global_env_ids] = 0
        self.sim.render_context.reset_scene_state_cadence()

    def _get_entity_selection(self, asset_name: str) -> SceneEntitySelectionCfg:
        """Return the cached selection configuration for a complete scene asset."""
        cache: dict[str, SceneEntitySelectionCfg] = self.__dict__.setdefault("_scene_entity_selections", {})
        if asset_name not in cache:
            cache[asset_name] = SceneEntitySelectionCfg(asset_name)
            cache[asset_name].resolve(self.scene)
        return cache[asset_name]
