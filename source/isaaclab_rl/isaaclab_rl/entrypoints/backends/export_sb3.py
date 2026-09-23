# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stable-Baselines3 LEAPP export backend of the unified reinforcement learning entrypoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch

# LEAPP traces Isaac Lab's Python tensor operations, so TorchScript is disabled before importing task
# or environment modules that compile decorated helpers at import time.
torch.jit._state.disable()

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.save_util import load_from_pkl, load_from_zip_file

from isaaclab.utils.assets import retrieve_file_path

import isaaclab_tasks.registry  # noqa: F401

from ..common import (
    CHECKPOINT_SELECTORS,
    normalize_task_name,
    resolve_checkpoint_selector,
    resolve_published_checkpoint,
)
from .export_common import (
    add_common_export_args,
    finalize_export_args,
    get_checkpoint_path,
    leapp_capture,
    prepare_export_env,
    resolve_export_save_path,
    run_export,
    state_dict_from_sequence,
    state_sequence_from_registered,
)

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    RecurrentPPO = None

_SB3_CONTRIB_HINT = (
    "Loading a recurrent SB3 checkpoint requires sb3-contrib. Install the Isaac Lab SB3 optional dependencies."
)


def parse_export_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse export arguments and return the remaining Hydra overrides."""
    parser = argparse.ArgumentParser(description="Export an RL agent with Stable-Baselines3.")
    add_common_export_args(parser, agent_default="sb3_cfg_entry_point")
    return finalize_export_args(parser, argv)


def _vec_normalize_path(checkpoint_path: str) -> Path:
    """Return the VecNormalize sidecar path used by the SB3 train and play workflows."""
    checkpoint = Path(checkpoint_path)
    stem = checkpoint.stem.replace("model", "model_vecnormalize", 1)
    return checkpoint.with_name(f"{stem}.pkl")


def _normalize_tensor(value: torch.Tensor, running_stats: Any, vec_normalize: Any) -> torch.Tensor:
    """Normalize one observation tensor with saved SB3 running statistics."""
    mean = torch.as_tensor(running_stats.mean, device=value.device, dtype=value.dtype)
    variance = torch.as_tensor(running_stats.var, device=value.device, dtype=value.dtype)
    normalized = (value - mean) / torch.sqrt(variance + vec_normalize.epsilon)
    return torch.clamp(normalized, -vec_normalize.clip_obs, vec_normalize.clip_obs)


def normalize_observation(obs: Any, vec_normalize: Any) -> Any:
    """Apply saved VecNormalize observation statistics with traceable torch operations."""
    if vec_normalize is None or not vec_normalize.norm_obs:
        return obs
    if isinstance(obs, dict):
        normalized = dict(obs)
        keys = vec_normalize.norm_obs_keys if vec_normalize.norm_obs_keys is not None else obs.keys()
        for key in keys:
            normalized[key] = _normalize_tensor(obs[key], vec_normalize.obs_rms[key], vec_normalize)
        return normalized
    return _normalize_tensor(obs, vec_normalize.obs_rms, vec_normalize)


def is_sb3_recurrent_policy(policy: Any) -> bool:
    """Return whether the SB3 policy exposes the recurrent actor interface."""
    return all(hasattr(policy, attribute) for attribute in ("lstm_actor", "lstm_hidden_state_shape", "_predict"))


def initialize_sb3_recurrent_state(policy: Any, num_envs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Create the actor LSTM hidden and cell state expected by an SB3 recurrent policy."""
    shape = policy.lstm_hidden_state_shape
    zeros = torch.zeros((shape[0], num_envs, shape[2]), device=policy.device, dtype=torch.float32)
    return (zeros.clone(), zeros.clone())


def _policy_actions(policy: Any, obs: Any, recurrent_state: tuple | None = None) -> tuple[torch.Tensor, tuple | None]:
    """Run deterministic SB3 policy inference without crossing a NumPy boundary."""
    policy.set_training_mode(False)
    if recurrent_state is None:
        if hasattr(policy, "_predict"):
            actions = policy._predict(obs, deterministic=True)
        else:
            actions = policy(obs, deterministic=True)[0]
        next_state = None
    else:
        batch_size = next(iter(obs.values())).shape[0] if isinstance(obs, dict) else obs.shape[0]
        episode_starts = torch.zeros(batch_size, device=policy.device, dtype=torch.float32)
        actions, next_state = policy._predict(
            obs, lstm_states=tuple(recurrent_state), episode_starts=episode_starts, deterministic=True
        )
    return _scale_or_clip_actions(policy, actions), next_state


def _scale_or_clip_actions(policy: Any, actions: torch.Tensor) -> torch.Tensor:
    """Match the Box-action post-processing performed by :meth:`BasePolicy.predict`."""
    action_space = policy.action_space
    if not hasattr(action_space, "low") or not hasattr(action_space, "high"):
        return actions
    low = torch.as_tensor(action_space.low, device=actions.device, dtype=actions.dtype)
    high = torch.as_tensor(action_space.high, device=actions.device, dtype=actions.dtype)
    if policy.squash_output:
        return low + 0.5 * (actions + 1.0) * (high - low)
    return torch.maximum(torch.minimum(actions, high), low)


def _load_agent(checkpoint_path: str, device: str) -> PPO:
    """Load PPO or RecurrentPPO according to the policy class stored in the checkpoint."""
    try:
        checkpoint_data, _, _ = load_from_zip_file(checkpoint_path, device=device)
    except ModuleNotFoundError as exc:
        if exc.name is not None and exc.name.startswith("sb3_contrib"):
            raise ImportError(_SB3_CONTRIB_HINT) from exc
        raise
    policy_class = checkpoint_data.get("policy_class")
    if getattr(policy_class, "__module__", "").startswith("sb3_contrib"):
        if RecurrentPPO is None:
            raise ImportError(_SB3_CONTRIB_HINT)
        return RecurrentPPO.load(checkpoint_path, device=device, print_system_info=True)
    return PPO.load(checkpoint_path, device=device, print_system_info=True)


def _resolve_checkpoint(args_cli: argparse.Namespace, env_cfg: Any) -> str | None:
    """Resolve the checkpoint to export, or None when no published checkpoint exists."""
    task_name = normalize_task_name(args_cli.task)
    if args_cli.checkpoint == "pretrained":
        return resolve_published_checkpoint("sb3", args_cli.task, env_cfg)
    log_root_path = os.path.abspath(os.path.join("logs", "sb3", task_name))
    if args_cli.checkpoint in CHECKPOINT_SELECTORS:
        return resolve_checkpoint_selector(
            log_root_path,
            args_cli.checkpoint,
            library="sb3",
            task=task_name,
            checkpoint_pattern=r"model(?:_.*)?\.zip",
            preferred_checkpoint_pattern=r"model\.zip",
            metadata={"agent": args_cli.agent},
        )
    if args_cli.checkpoint is not None:
        return retrieve_file_path(args_cli.checkpoint)
    return get_checkpoint_path(
        log_root_path, ".*", r"model_.*\.zip", sort_alpha=False, preferred_checkpoint=r"model\.zip"
    )


def export_sb3_agent(args_cli: argparse.Namespace, env_cfg: Any, agent_cfg: dict) -> bool:
    """Export a Stable-Baselines3 policy; returns whether a graph was written."""
    # concrete environment classes and the LEAPP runtime load simulation modules, so import them
    # only after launch_simulation has initialized the selected backend
    from leapp import annotate

    from isaaclab.envs import ManagerBasedRLEnv

    checkpoint_path = _resolve_checkpoint(args_cli, env_cfg)
    if not checkpoint_path:
        print(f"[INFO] No checkpoint found for task: {args_cli.task}")
        return False

    env_cfg.scene.num_envs = 1
    env_cfg.seed = agent_cfg["seed"]
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    log_dir = os.path.dirname(checkpoint_path)
    env_cfg.log_dir = log_dir

    # SB3 constructs torch.distributions.Normal even for deterministic PPO inference. Its eager argument
    # validation reduces tensor predicates to Python booleans, which a LEAPP static graph cannot represent
    # and which is unrelated to the action computation, so it is disabled only while tracing.
    previous_validate_args = torch.distributions.Distribution._validate_args
    torch.distributions.Distribution.set_default_validate_args(False)
    env = None
    try:
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        policy_node_name = prepare_export_env(env, args_cli, required_obs_groups={"policy"})
        if not isinstance(env.unwrapped, ManagerBasedRLEnv):
            raise NotImplementedError("SB3 LEAPP export currently supports manager-based environments only.")

        print(f"[INFO] Loading model checkpoint from: {checkpoint_path}")
        policy = _load_agent(checkpoint_path, device=env.unwrapped.device).policy
        vec_normalize_path = _vec_normalize_path(checkpoint_path)
        if vec_normalize_path.exists():
            print(f"[INFO] Loading saved normalization: {vec_normalize_path}")
            vec_normalize = load_from_pkl(vec_normalize_path)
        elif agent_cfg.get("normalize_input", False):
            raise FileNotFoundError(
                f"SB3 policy requires observation normalization, but no sidecar was found: {vec_normalize_path}"
            )
        else:
            vec_normalize = None

        save_path = resolve_export_save_path(args_cli, "sb3", log_dir)
        with leapp_capture(args_cli, save_path=save_path, env_cfg=env_cfg) as num_steps:
            obs = env.reset()[0]["policy"]
            recurrent_state = None
            if is_sb3_recurrent_policy(policy):
                recurrent_state = initialize_sb3_recurrent_state(policy, env.num_envs)
            for _ in range(num_steps):
                with torch.inference_mode():
                    obs = normalize_observation(obs, vec_normalize)
                    if recurrent_state is not None:
                        state_dict = state_dict_from_sequence(recurrent_state)
                        registered_state = annotate.state_tensors(policy_node_name, state_dict)
                        recurrent_state = tuple(
                            state_sequence_from_registered(registered_state, list(state_dict), recurrent_state)
                        )
                    actions, next_recurrent_state = _policy_actions(policy, obs, recurrent_state)
                    obs_dict, _, terminated, truncated, _ = env.step(actions)
                    obs = obs_dict["policy"]
                    if next_recurrent_state is not None:
                        # zero the recurrent state of environments that finished their episode
                        not_done = (~(terminated | truncated)).to(dtype=next_recurrent_state[0].dtype)
                        not_done = not_done.reshape(1, -1, 1)
                        recurrent_state = tuple(state * not_done for state in next_recurrent_state)
                        annotate.update_state(policy_node_name, state_dict_from_sequence(recurrent_state))
    finally:
        torch.distributions.Distribution.set_default_validate_args(previous_validate_args)
        if env is not None:
            env.close()
    return True


def run(argv: list[str] | None = None) -> int:
    """Run the export backend and return a process exit code."""
    args_cli, hydra_args = parse_export_args(argv)
    return run_export(args_cli, hydra_args, export_sb3_agent)


if __name__ == "__main__":
    raise SystemExit(run())
