# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runner construction shared by RSL-RL entrypoints and benchmarks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from isaaclab.utils import to_dict

from ...rsl_rl import RslRlSuperMimicRunnerCfg
from ...rsl_rl.vecenv_wrapper import RslRlShacVecEnvWrapper, RslRlVecEnvWrapper

if TYPE_CHECKING:
    from rsl_rl.runners import OnPolicyRunner

    from ...rsl_rl import RslRlBaseRunnerCfg


def configure_rsl_rl_env_cfg(env_cfg: Any, agent_cfg: RslRlBaseRunnerCfg, *, inference: bool = False) -> None:
    """Enable terminal observations requested by the agent before creating the environment.

    Args:
        env_cfg: Environment configuration to update in place.
        agent_cfg: Runner configuration that may require pre-reset observations.
    """
    if agent_cfg.compute_final_obs:
        env_cfg.compute_final_obs = True

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
