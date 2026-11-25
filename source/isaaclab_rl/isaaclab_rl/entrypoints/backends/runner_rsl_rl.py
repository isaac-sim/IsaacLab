# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runner construction shared by RSL-RL entrypoints and benchmarks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.utils import to_dict
from isaaclab.utils.array import index_fill_

from ...rsl_rl import RslRlSuperMimicRunnerCfg
from ...rsl_rl.vecenv_wrapper import RslRlShacVecEnvWrapper, RslRlVecEnvWrapper

if TYPE_CHECKING:
    from rsl_rl.runners import OnPolicyRunner

    from ...rsl_rl import RslRlBaseRunnerCfg


def step_with_final_observations(env: ManagerBasedRLEnv, action: torch.Tensor) -> VecEnvStepReturn:
    """Collect physical successor observations before resets, commands and interval events for RSL-RL."""
    env.extras.pop("final_obs", None)
    env.extras.pop("final_obs_every_step", None)
    env.action_manager.process_action(action.to(env.device))
    if env._physics_handles_actions is None:
        env._handle_physics_actions()

    # check if we need to do rendering within the physics loop
    # note: evaluated once per step rather than once per physics substep
    is_rendering = env.sim.is_rendering

    # physics-owned decimation covers all substeps in one call
    steps_per_call = env.cfg.decimation if env._physics_handles_decimation else 1
    for _ in range(env.cfg.decimation // steps_per_call):
        env._sim_step_counter += steps_per_call
        if not env._physics_handles_actions:
            env.action_manager.apply_action()
            env.scene.write_data_to_sim()
        env.sim.step(render=False)
        # render_enabled=False skips Kit (camera/GUI); standalone visualizers still update
        if env._sim_step_counter % env.cfg.sim.render_interval == 0 and is_rendering:
            env.sim.render(skip_app_pumping=not env.render_enabled)
        env.scene.update(dt=env.physics_dt * steps_per_call)

    env.episode_length_buf += 1  # step in current episode (per env)
    env.common_step_counter += 1  # total step (common for all envs)
    # -- check terminations
    env.reset_buf = env.termination_manager.compute()
    env.reset_terminated = env.termination_manager.terminated
    env.reset_time_outs = env.termination_manager.time_outs
    env.reward_buf = env.reward_manager.compute(dt=env.step_dt)

    # -- reset envs that terminated/timed-out and log the episode information
    reset_env_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1).int()
    env.extras["final_obs"] = env.observation_manager.compute()
    env.extras["final_obs_every_step"] = True
    if len(reset_env_ids) > 0:
        env._reset_idx(reset_env_ids)

        # if sensors are added to the scene, make sure we render to reflect changes in reset
        if env.render_enabled and is_rendering and env.has_rtx_sensors and env.cfg.num_rerenders_on_reset > 0:
            for _ in range(env.cfg.num_rerenders_on_reset):
                env.sim.render()

    # -- handle episode reset requested from visualizer UI controls
    manual_reset_ids = reset_env_ids[:0]
    if env.sim.consume_reset_request():
        # Only reset envs not already reset this step to avoid redundant resets.
        not_yet_reset = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        index_fill_(not_yet_reset, reset_env_ids, False)
        manual_reset_ids = not_yet_reset.nonzero(as_tuple=False).squeeze(-1).int()
        if len(manual_reset_ids) > 0:
            # mark as terminated so RL wrappers observe the episode boundary
            index_fill_(env.reset_terminated, manual_reset_ids, True)
            env._reset_idx(manual_reset_ids)

    if len(reset_env_ids) > 0 or len(manual_reset_ids) > 0:
        # Reset writes invalidate derived body poses used by observations and rewards.
        env.sim.forward()

    env.command_manager.compute(dt=env.step_dt)
    if "interval" in env.event_manager.active_terms:
        env.event_manager.apply(mode="interval", dt=env.step_dt)
    # -- advance video recorders (after render and resets, before final obs)
    for recorder in env.video_recorders:
        recorder.step()
    # -- compute observations
    # note: done after reset to get the correct observations for reset envs
    env.obs_buf = env.observation_manager.compute(update_history=True)

    return env.obs_buf, env.reward_buf, env.reset_terminated, env.reset_time_outs, env.extras


def configure_rsl_rl_env_cfg(env_cfg: Any, agent_cfg: RslRlBaseRunnerCfg, *, inference: bool = False) -> None:
    """Enable terminal observations requested by the agent before creating the environment.

    Args:
        env_cfg: Environment configuration to update in place.
        agent_cfg: Runner configuration that may require pre-reset observations.
    """
    if agent_cfg.final_obs_every_step and not isinstance(
        env_cfg, (ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg)
    ):
        raise ValueError("Every-step pre-reset observations require a ManagerBasedRLEnv, DirectRLEnv or DirectMARLEnv.")
    if agent_cfg.compute_final_obs:
        env_cfg.compute_final_obs = True
    if agent_cfg.final_obs_every_step:
        if isinstance(env_cfg, ManagerBasedRLEnvCfg):
            if env_cfg.step_fn is None:
                env_cfg.step_fn = step_with_final_observations
        else:
            env_cfg.final_obs_every_step = True

    if isinstance(agent_cfg, RslRlSuperMimicRunnerCfg):
        agent_cfg.algorithm.task_cfg.configure_env(env_cfg, agent_cfg, inference=inference)
        for term in agent_cfg.separate_goal_terms:
            setattr(env_cfg.observations.policy, term, None)
        if "history" in agent_cfg.obs_groups["actor"]:
            env_cfg.observations.policy.history_length = agent_cfg.algorithm.history_length
            env_cfg.observations.policy.flatten_history_dim = False


def wrap_rsl_rl_env(env: Any, agent_cfg: RslRlBaseRunnerCfg) -> RslRlVecEnvWrapper:
    """Adapt the simulator observations and actions to the configured runner."""
    if isinstance(agent_cfg, RslRlSuperMimicRunnerCfg):
        return RslRlShacVecEnvWrapper(
            env, agent_cfg.clip_actions, use_reference_action=agent_cfg.algorithm.use_reference_action
        )
    return RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)


def create_rsl_rl_runner(
    env: Any, agent_cfg: RslRlBaseRunnerCfg, log_dir: str | None = None, *, inference: bool = False
) -> OnPolicyRunner:
    """Construct the configured runner using native RSL-RL callable resolution.

    A cosine learning-rate schedule without ``num_schedule_updates`` spans this run's
    ``max_iterations`` times ``num_updates`` critic updates.

    Args:
        env: Wrapped RSL-RL vector environment.
        agent_cfg: Runner configuration whose ``class_name`` selects the runner.
        log_dir: Directory for logs and checkpoints, or None to disable logging.
        inference: Whether the runner only loads and runs a policy, which disables CUDA graph capture.

    Returns:
        The constructed runner.
    """
    from rsl_rl.utils import resolve_callable

    cfg = to_dict(agent_cfg)
    alg_cfg = cfg["algorithm"]
    if inference and "use_cuda_graph" in alg_cfg:
        alg_cfg["use_cuda_graph"] = False
    if alg_cfg.get("learning_rate_schedule") == "cosine" and alg_cfg.get("num_schedule_updates") is None:
        alg_cfg["num_schedule_updates"] = cfg["max_iterations"] * alg_cfg["num_updates"]
    return resolve_callable(agent_cfg.class_name)(env, cfg, log_dir=log_dir, device=agent_cfg.device)


def rsl_rl_load_cfg(runner: OnPolicyRunner, resume_optimizer_state: bool) -> dict | None:
    """Return the checkpoint selection for continued training.

    Args:
        runner: Runner whose algorithm defines the default checkpoint selection.
        resume_optimizer_state: Whether to restore the checkpoint's policy optimizer state and learning rate.

    Returns:
        Native RSL-RL load selection, or None for its ordinary checkpoint behavior.
    """
    if resume_optimizer_state:
        return None
    return {**runner.alg.get_default_load_cfg(), "optimizer": False}
