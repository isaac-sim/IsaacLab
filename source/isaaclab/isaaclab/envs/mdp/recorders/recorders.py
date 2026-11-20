# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.utils import index_fill_
from isaaclab.utils.datasets import EpisodeRecorder

from ...common import VecEnvStepReturn

if TYPE_CHECKING:
    from ... import ManagerBasedEnv, ManagerBasedRLEnv


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


def record_step(env: ManagerBasedRLEnv, action: torch.Tensor, *, recorder: EpisodeRecorder) -> VecEnvStepReturn:
    """Step an environment with explicit action/state capture and episode export.

    This is a concrete custom step function, selected by recording workflows before the first
    step. It owns the decimation loop so capture can also be inserted between physics steps.
    Copy and adapt this function to customize capture ordering or add per-substep work.

    Args:
        env: Environment being stepped.
        action: Batched actions with shape (num_envs, action_dim).
        recorder: Episode storage owned by the calling workflow.

    Returns:
        Observations, rewards, terminated flags, truncated flags, and extras.
    """
    env.action_manager.process_action(action.to(env.device))
    record_pre_step(recorder, env, env.action_manager.action)
    # Each simulator call advances one physics step; this function owns decimation.
    env.sim.backend_manager.set_decimation(1)

    # check if we need to do rendering within the physics loop
    # note: evaluated once per step rather than once per physics substep
    is_rendering = env.sim.is_rendering

    for substep in range(env.cfg.decimation):
        env._sim_step_counter += 1
        env.action_manager.apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        # render_enabled=False skips Kit (camera/GUI); standalone visualizers still update
        if env._sim_step_counter % env.cfg.sim.render_interval == 0 and is_rendering:
            env.sim.render(skip_app_pumping=not env.render_enabled)
        env.scene.update(dt=env.physics_dt)
        # Insert per-substep capture here, with current scene buffers.

    env.episode_length_buf += 1  # step in current episode (per env)
    env.common_step_counter += 1  # total step (common for all envs)
    # -- check terminations
    env.reset_buf = env.termination_manager.compute()
    env.reset_terminated = env.termination_manager.terminated
    env.reset_time_outs = env.termination_manager.time_outs
    env.reward_buf = env.reward_manager.compute(dt=env.step_dt)

    record_post_step(recorder, env)

    # -- reset envs that terminated/timed-out and log the episode information
    reset_env_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1).int()
    if len(reset_env_ids) > 0:
        # capture the terminal observation before reset and expose it for Same-Step autoreset.
        if env.cfg.compute_final_obs:
            env.extras["final_obs"] = env.observation_manager.compute()

        success = (
            env.termination_manager.get_term("success")[reset_env_ids]
            if "success" in env.termination_manager.active_terms
            else torch.zeros_like(reset_env_ids, dtype=torch.bool)
        )
        recorder.set_success_to_episodes(reset_env_ids, success)
        recorder.export_episodes(reset_env_ids)
        env._reset_idx(reset_env_ids)

        # if sensors are added to the scene, make sure we render to reflect changes in reset
        if env.render_enabled and is_rendering and env.has_rtx_sensors and env.cfg.num_rerenders_on_reset > 0:
            for _ in range(env.cfg.num_rerenders_on_reset):
                env.sim.render()

        record_initial_state(recorder, env, reset_env_ids)

    # -- handle episode reset requested from visualizer UI controls
    if env.sim.consume_reset_request():
        # Only reset envs not already reset this step to avoid redundant resets.
        not_yet_reset = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        index_fill_(not_yet_reset, reset_env_ids, False)
        manual_reset_ids = not_yet_reset.nonzero(as_tuple=False).squeeze(-1).int()
        if len(manual_reset_ids) > 0:
            # mark as terminated so RL wrappers observe the episode boundary
            index_fill_(env.reset_terminated, manual_reset_ids, True)
            recorder.set_success_to_episodes(manual_reset_ids, torch.zeros_like(manual_reset_ids, dtype=torch.bool))
            recorder.export_episodes(manual_reset_ids)
            env._reset_idx(manual_reset_ids)
            record_initial_state(recorder, env, manual_reset_ids)

    env.command_manager.compute(dt=env.step_dt)
    if "interval" in env.event_manager.active_terms:
        env.event_manager.apply(mode="interval", dt=env.step_dt)
    # -- advance video recorders (after render and resets, before final obs)
    for video_recorder in env.video_recorders:
        video_recorder.step()
    # -- compute observations
    # note: done after reset to get the correct observations for reset envs
    env.obs_buf = env.observation_manager.compute(update_history=True)

    return env.obs_buf, env.reward_buf, env.reset_terminated, env.reset_time_outs, env.extras
