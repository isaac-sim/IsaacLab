# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.utils.datasets import EpisodeRecorder

if TYPE_CHECKING:
    from ... import ManagerBasedEnv


def record_initial_state(
    recorder: EpisodeRecorder, env: ManagerBasedEnv, env_ids: Sequence[int] | slice | None = None
) -> None:
    """Capture initial scene state after the caller resets the selected environments.

    Args:
        recorder: Episode storage owned by the recording workflow.
        env: Environment whose scene has been synchronized after reset.
        env_ids: Selected environments, or None for all environments.
    """
    selection = slice(None) if env_ids is None else (list(env_ids) if isinstance(env_ids, Sequence) else env_ids)

    def select(value):
        if isinstance(value, dict):
            return {key: select(item) for key, item in value.items()}
        return value[selection]

    recorder.add_to_episodes("initial_state", select(env.scene.get_state(is_relative=True)), env_ids)


def record_pre_step(recorder: EpisodeRecorder, env: ManagerBasedEnv, action: torch.Tensor) -> None:
    """Copy the policy input and requested actions before advancing the environment.

    Args:
        recorder: Episode storage owned by the recording workflow.
        env: Environment with the preceding policy observations in its observation buffer.
        action: Batched actions with shape (num_envs, action_dim).
    """
    recorder.add_to_episodes("actions", action.to(env.action_manager.action))
    recorder.add_to_episodes("obs", env.obs_buf["policy"])


def record_post_step(recorder: EpisodeRecorder, env: ManagerBasedEnv) -> None:
    """Capture scene state and processed actions after simulation and before an episode reset.

    Args:
        recorder: Episode storage owned by the recording workflow.
        env: Environment whose physical transition has completed.
    """
    recorder.add_to_episodes("states", env.scene.get_state(is_relative=True))
    actions = [env.action_manager.get_term(name).processed_actions for name in env.action_manager.active_terms]
    if actions:
        recorder.add_to_episodes("processed_actions", torch.cat(actions, dim=-1))


def get_episode_metadata(env: ManagerBasedEnv) -> dict:
    """Read task metadata or the standard simulation settings for a dataset.

    Args:
        env: Environment whose recording workflow is creating dataset storage.

    Returns:
        Task-specific metadata or simulation time-step, decimation, rendering, and batch settings.
    """
    if hasattr(env.cfg, "get_ep_meta"):
        return env.cfg.get_ep_meta()
    return {
        "sim_args": {
            "dt": env.cfg.sim.dt,
            "decimation": env.cfg.decimation,
            "render_interval": env.cfg.sim.render_interval,
            "num_envs": env.num_envs,
        }
    }
