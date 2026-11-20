# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms of the world-model backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .terminations import check_chunk_timing

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import EventTermCfg

    from ..physics import WorldModelManager


class reset_from_records(ManagerTermBase):
    """Start new episodes from reset records, in place of a physics task's ``reset_joints_by_*`` event.

    Samples a record per environment and publishes its start image and joint state immediately. The model's history
    is reset from that image and prompt just before the next chunk is generated.

    The world model runs once per chunk of environment steps, so environments can only reset on chunk boundaries.
    The term checks when the environment is created that decimation divides the chunk length and that
    ``episode_length_s`` is a multiple of the chunk duration, so that time-outs fall on boundaries. Termination terms
    must also fire only on boundaries, see :func:`~isaaclab_neural.mdp.at_chunk_boundary`.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        check_chunk_timing(env)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> None:
        """Reset the selected environments.

        Args:
            env: The environment.
            env_ids: Environments to reset; the event manager passes a slice for a reset of all environments.
            asset_cfg: The articulation driven by the world model.
        """
        manager: WorldModelManager = env.sim.backend_manager
        asset = env.scene[asset_cfg.name]
        selected = torch.zeros(asset.num_instances, dtype=torch.bool, device=asset.device)
        if isinstance(env_ids, slice):
            selected[env_ids] = True
        else:
            selected.index_fill_(0, env_ids, True)
        manager.reset_envs(selected)
