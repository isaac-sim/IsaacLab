# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms of the world-model backend."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import EventTermCfg

    from ..physics import WorldModelManager


class reset_from_records(ManagerTermBase):
    """Start new episodes from reset records, in place of a physics task's ``reset_joints_by_*`` event.

    Samples a record per environment, resets the world model from its start image and prompt, and writes the
    record's joint state to the articulation, so the first chunk starts from the recorded state.

    The world model runs once per chunk of environment steps, so environments can only reset on chunk boundaries.
    The term checks when the environment is created that one environment step is one model frame and that
    ``episode_length_s`` is a multiple of the chunk duration, so that time-outs fall on boundaries. Termination terms
    must also fire only on boundaries, see :func:`~isaaclab_neural.mdp.at_chunk_boundary`.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        spec = env.sim.backend_manager.spec
        if not math.isclose(env.step_dt, 1.0 / spec.fps):
            raise ValueError(
                f"One environment step must be one model frame: step_dt is {env.step_dt} s but the model runs at "
                f"{spec.fps} Hz. Use decimation = 1 and sim.dt = 1 / fps."
            )
        if env.max_episode_length % spec.chunk_length:
            raise ValueError(
                f"An episode of {env.max_episode_length} steps is not a multiple of the model chunk of "
                f"{spec.chunk_length} steps, so its time-out would fall inside a chunk."
            )

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
        ids = torch.arange(env.num_envs, device=env.device)[env_ids]
        manager.reset_envs(ids.cpu().numpy())
        position = torch.from_numpy(manager.joint_state[ids.cpu().numpy()]).to(asset.device)
        asset.write_joint_state_to_sim(position, torch.zeros_like(position), env_ids=ids)
        asset.set_joint_position_target_index(target=position, env_ids=ids)
