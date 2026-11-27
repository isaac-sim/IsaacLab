# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL LEAPP export backend of the unified reinforcement learning entrypoint."""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)

# LEAPP traces Isaac Lab's Python tensor operations, so TorchScript is disabled before importing task
# or environment modules that compile decorated helpers at import time.
torch.jit._state.disable()

import gymnasium as gym
from rsl_rl.models import MLPModel

from isaaclab.utils.assets import retrieve_file_path

import isaaclab_tasks  # noqa: F401

from ...rsl_rl import RslRlBaseRunnerCfg, RslRlDistillationRunnerCfg, RslRlSacRunnerCfg, RslRlSuperMimicRunnerCfg
from ..common import resolve_published_checkpoint, resolve_seed
from .export_common import (
    add_common_export_args,
    finalize_export_args,
    get_checkpoint_path,
    leapp_capture,
    prepare_export_env,
    resolve_export_save_path,
    run_export,
)
from .runner_rsl_rl import configure_rsl_rl_env_cfg, create_rsl_rl_runner, wrap_rsl_rl_env


def parse_export_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse export arguments and return the remaining Hydra overrides."""
    parser = argparse.ArgumentParser(description="Export an RL agent with RSL-RL.")
    add_common_export_args(parser, agent_default="rsl_rl_cfg_entry_point")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
    parser.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder used to locate checkpoints."
    )
    return finalize_export_args(parser, argv)


def get_actor_memory_module(policy: Any) -> Any | None:
    """Return the actor-side RNN module for supported RSL-RL recurrent policies."""
    return getattr(policy, "rnn", None)


def is_actor_recurrent_policy(policy: Any) -> bool:
    """Return whether the actor policy has a supported recurrent state container."""
    return bool(getattr(policy, "is_recurrent", False) and get_actor_memory_module(policy) is not None)


def get_actor_hidden_state(policy: Any) -> Any | None:
    """Return the actor-side recurrent hidden state for supported RSL-RL policy APIs."""
    if hasattr(policy, "get_hidden_state"):
        return policy.get_hidden_state()
    memory = get_actor_memory_module(policy)
    return None if memory is None else getattr(memory, "hidden_state", None)


def set_actor_hidden_state(policy: Any, actor_hidden: Any) -> None:
    """Assign the actor-side recurrent hidden state for supported RSL-RL policy APIs."""
    memory = get_actor_memory_module(policy)
    if memory is not None:
        memory.hidden_state = actor_hidden


def ensure_actor_hidden_state_initialized(policy: Any, batch_size: int, device: Any, dtype: torch.dtype) -> Any | None:
    """Initialize and return the actor hidden state when a recurrent policy has not created it yet."""
    actor_state = get_actor_hidden_state(policy)
    if actor_state is not None:
        return actor_state
    memory = get_actor_memory_module(policy)
    if memory is None or not hasattr(memory, "rnn"):
        return None
    zeros = torch.zeros(memory.rnn.num_layers, batch_size, memory.rnn.hidden_size, device=device, dtype=dtype)
    if isinstance(memory.rnn, torch.nn.LSTM):
        actor_state = (zeros.clone(), zeros.clone())
    else:
        actor_state = zeros
    set_actor_hidden_state(policy, actor_state)
    return actor_state


def state_dict_from_actor_hidden(actor_hidden: Any) -> dict[str, torch.Tensor]:
    """Convert the actor hidden state into the named tensor mapping expected by LEAPP state APIs."""
    if actor_hidden is None:
        return {}
    if isinstance(actor_hidden, tuple):
        return {f"actor_state_{idx}": tensor for idx, tensor in enumerate(actor_hidden)}
    return {"actor_state": actor_hidden}


def actor_hidden_from_registered(registered_state: Any, original_hidden: Any) -> Any:
    """Restore the registered LEAPP state to the hidden-state structure expected by the actor memory module."""
    if isinstance(original_hidden, tuple) and not isinstance(registered_state, tuple):
        return (registered_state,)
    return registered_state


def _resolve_checkpoint(
    args_cli: argparse.Namespace, agent_cfg: RslRlBaseRunnerCfg, env_cfg: Any, log_root_path: str
) -> str | None:
    """Resolve the checkpoint to export, or None when no published checkpoint exists."""
    if args_cli.checkpoint == "pretrained":
        if isinstance(agent_cfg, (RslRlSacRunnerCfg, RslRlSuperMimicRunnerCfg)):
            raise ValueError("Published checkpoints are PPO policies and cannot be loaded by OffPolicyRunner agents.")
        return resolve_published_checkpoint("rsl_rl", args_cli.task, env_cfg)
    if args_cli.checkpoint and os.path.isdir(args_cli.checkpoint):
        return get_checkpoint_path(
            os.path.dirname(args_cli.checkpoint), os.path.basename(args_cli.checkpoint), agent_cfg.load_checkpoint
        )
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def export_rsl_rl_agent(args_cli: argparse.Namespace, env_cfg: Any, agent_cfg: RslRlBaseRunnerCfg) -> bool:
    """Export an RSL-RL agent; returns whether a graph was written."""
    if args_cli.seed is not None:
        agent_cfg.seed = resolve_seed(args_cli.seed)
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    agent_cfg.device = args_cli.device

    env_cfg.scene.num_envs = 1
    # certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    logger.info(f"Loading checkpoint search path from directory: {log_root_path}")
    resume_path = _resolve_checkpoint(args_cli, agent_cfg, env_cfg, log_root_path)
    if not resume_path:
        logger.info(f"No checkpoint found for task: {args_cli.task} in directory: {log_root_path}")
        return False
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    configure_rsl_rl_env_cfg(env_cfg, agent_cfg, inference=True)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    try:
        env = wrap_rsl_rl_env(env, agent_cfg)
        logger.info(f"Loading model checkpoint from: {resume_path}")
        runner = create_rsl_rl_runner(env, agent_cfg, inference=True)
        runner.load(resume_path, map_location=agent_cfg.device)
        # Capture portable operators instead of device-specific fused inference kernels.
        policy = runner.get_inference_policy(device="cpu")
        from rsl_rl.models import SuperMimicActor

        motion_policy = isinstance(policy, SuperMimicActor)
        if motion_policy:
            from .export_model_based import prepare_motion_export

            task = type(runner.alg.task)(env=env, device=env.unwrapped.device)
            policy = prepare_motion_export(env, policy, task=task)
        elif not isinstance(policy, MLPModel):
            raise TypeError("LEAPP export requires an MLPModel or SuperMimicActor policy.")
        obs_set = "student" if isinstance(agent_cfg, RslRlDistillationRunnerCfg) else "actor"
        required_obs_groups = set(runner.cfg["obs_groups"][obs_set])
        save_path = resolve_export_save_path(args_cli, "rsl_rl", log_dir)
        export_policy(env, policy, args_cli, env_cfg, save_path, required_obs_groups)
        if motion_policy:
            from .export_model_based import save_motion

            save_motion(env, save_path, args_cli.export_task_name or args_cli.task)
    finally:
        env.close()
    return True


def export_policy(env, policy, args_cli, env_cfg, save_path, required_obs_groups):
    """Capture one policy through the common observation and action annotations."""
    from leapp import annotate

    policy_node_name = prepare_export_env(env, args_cli, required_obs_groups=required_obs_groups)
    recurrent = is_actor_recurrent_policy(policy)

    with leapp_capture(args_cli, save_path=save_path, env_cfg=env_cfg) as num_steps:
        obs = env.reset()[0]
        for _ in range(num_steps):
            with torch.inference_mode():
                if recurrent:
                    actor_hidden = ensure_actor_hidden_state_initialized(
                        policy,
                        batch_size=env.num_envs,
                        device="cpu",
                        dtype=next(policy.parameters()).dtype,
                    )
                    registered_state = annotate.state_tensors(
                        policy_node_name, state_dict_from_actor_hidden(actor_hidden)
                    )
                    set_actor_hidden_state(policy, actor_hidden_from_registered(registered_state, actor_hidden))
                actions = policy(obs.to("cpu"))
                if recurrent:
                    annotate.update_state(
                        policy_node_name, state_dict_from_actor_hidden(get_actor_hidden_state(policy))
                    )
                obs, _, _, _ = env.step(actions.to(env.unwrapped.device))


def run(argv: list[str] | None = None) -> int:
    """Run the export backend and return a process exit code."""
    args_cli, hydra_args = parse_export_args(argv)
    if args_cli.task == "Isaac-Motion-Tracking-GenCoord-G1":
        from .export_model_based import run as export_published_motion

        return export_published_motion(argv)
    return run_export(args_cli, hydra_args, export_rsl_rl_agent)


if __name__ == "__main__":
    raise SystemExit(run())
