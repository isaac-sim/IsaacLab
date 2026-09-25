# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL LEAPP export backend of the unified reinforcement learning entrypoint."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from typing import Any

import torch

# LEAPP traces Isaac Lab's Python tensor operations, so TorchScript is disabled before importing task
# or environment modules that compile decorated helpers at import time.
torch.jit._state.disable()

import gymnasium as gym

from isaaclab.utils.assets import retrieve_file_path

import isaaclab_tasks  # noqa: F401

from ...rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    check_rsl_rl_version,
    create_rsl_rl_runner,
    handle_deprecated_rsl_rl_cfg,
)
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
    # the LEAPP runtime loads simulation modules, so import it only after the launch
    from leapp import annotate

    installed_version = check_rsl_rl_version()
    if args_cli.seed is not None:
        agent_cfg.seed = resolve_seed(args_cli.seed)
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env_cfg.scene.num_envs = 1
    # certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading checkpoint search path from directory: {log_root_path}")
    resume_path = _resolve_checkpoint(args_cli, agent_cfg, env_cfg, log_root_path)
    if not resume_path:
        print(f"[INFO] No checkpoint found for task: {args_cli.task} in directory: {log_root_path}")
        return False
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    try:
        # Export only the deployed actor or student inputs, never privileged teacher observations.
        obs_groups_cfg = getattr(agent_cfg, "obs_groups", None)
        if isinstance(obs_groups_cfg, Mapping):
            inference_group = "student" if agent_cfg.class_name == "DistillationRunner" else "actor"
            required_obs_groups = set(obs_groups_cfg.get(inference_group, ["policy"]))
        else:
            required_obs_groups = {"policy"}
        policy_node_name, patcher = prepare_export_env(env, args_cli, required_obs_groups=required_obs_groups)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner = create_rsl_rl_runner(env, agent_cfg)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        recurrent = is_actor_recurrent_policy(policy)

        save_path = resolve_export_save_path(args_cli, "rsl_rl", log_dir)
        with leapp_capture(args_cli, save_path=save_path, env_cfg=env_cfg, patcher=patcher) as num_steps:
            obs = env.reset()[0]
            for _ in range(num_steps):
                with torch.inference_mode():
                    if recurrent:
                        actor_hidden = ensure_actor_hidden_state_initialized(
                            policy,
                            batch_size=env.num_envs,
                            device=env.unwrapped.device,
                            dtype=next(policy.parameters()).dtype,
                        )
                        registered_state = annotate.state_tensors(
                            policy_node_name, state_dict_from_actor_hidden(actor_hidden)
                        )
                        set_actor_hidden_state(policy, actor_hidden_from_registered(registered_state, actor_hidden))
                    actions = policy(obs)
                    if recurrent:
                        annotate.update_state(
                            policy_node_name, state_dict_from_actor_hidden(get_actor_hidden_state(policy))
                        )
                    obs, _, _, _ = env.step(actions)
    finally:
        env.close()
    return True


def run(argv: list[str] | None = None) -> int:
    """Run the export backend and return a process exit code."""
    args_cli, hydra_args = parse_export_args(argv)
    return run_export(args_cli, hydra_args, export_rsl_rl_agent)


if __name__ == "__main__":
    raise SystemExit(run())
