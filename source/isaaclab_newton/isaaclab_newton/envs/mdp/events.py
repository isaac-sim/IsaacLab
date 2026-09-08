# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton visual randomization event terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.visual_events import _compile_distribution
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

from ...physics.newton_manager import NewtonManager

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class randomize_visual_shape(ManagerTermBase):
    """Sample one color per selected link and write Newton shape storage on device."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        channels = cfg.params["channels"]
        if tuple(channels) != ("color",):
            raise NotImplementedError("Newton per-shape randomization currently supports only the 'color' channel.")
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        asset = env.scene[asset_cfg.name]
        if isinstance(asset_cfg.body_ids, slice):
            ids = range(asset.num_bodies)[asset_cfg.body_ids]
        else:
            ids = asset_cfg.body_ids
        body_names = tuple(asset.body_names[index] for index in ids)
        self._writer = NewtonManager.create_visual_shape_color_writer(asset, body_names)
        self._sample = _compile_distribution(channels["color"], env.device)
        self._all_env_ids = torch.arange(env.num_envs, dtype=torch.int32, device=self._writer.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
        asset_cfg: SceneEntityCfg,
        channels: dict[str, tuple | dict],
    ) -> None:
        del asset_cfg, channels
        if isinstance(env_ids, slice):
            env_ids = None
        selected = self._all_env_ids if env_ids is None else env_ids.to(device=self._writer.device, dtype=torch.int32)
        model = NewtonManager.get_model()
        if self._writer.model is not model:
            self._writer.rebind(model)
        self._writer(self._sample((len(selected), self._writer.body_count)), selected)
