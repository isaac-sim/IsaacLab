# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to export a checkpoint of an RL agent from Stable-Baselines3."""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from pathlib import Path

import torch

# LEAPP traces Isaac Lab's Python tensor operations, so disable TorchScript before
# importing task or environment modules that compile decorated helpers.
torch.jit._state.disable()

import gymnasium as gym
import leapp
from leapp import annotate
from stable_baselines3 import PPO
from stable_baselines3.common.save_util import load_from_pkl, load_from_zip_file

from isaaclab.app import launch_simulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.leapp import patch_env_for_export
from isaaclab.utils.leapp.utils import ensure_env_spec_id

from isaaclab_rl.entrypoints.backends.export_common import (
    add_common_export_args,
    create_graph_configs,
    finalize_export_args,
    get_checkpoint_path,
    state_dict_from_sequence,
    state_sequence_from_registered,
)
from isaaclab_rl.entrypoints.common import CHECKPOINT_SELECTORS, resolve_checkpoint_selector
from isaaclab_rl.utils.pretrained_checkpoint import (
    get_pretrained_checkpoint_backend_names,
    get_published_pretrained_checkpoint,
)

from isaaclab_tasks.utils.hydra import hydra_task_config

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    RecurrentPPO = None


def parse_export_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse export arguments and return remaining Hydra overrides."""
    parser = argparse.ArgumentParser(description="Export an RL agent with Stable-Baselines3.")
    add_common_export_args(parser, agent_default="sb3_cfg_entry_point")
    return finalize_export_args(parser, argv, agent_library="sb3")


def _vec_normalize_path(checkpoint_path: str) -> Path:
    """Return the VecNormalize sidecar path used by the SB3 train/play workflows."""
    checkpoint = Path(checkpoint_path)
    normalized_stem = checkpoint.stem.replace("model", "model_vecnormalize", 1)
    return checkpoint.with_name(f"{normalized_stem}.pkl")


def _normalize_tensor(value, running_stats, vec_normalize):
    """Normalize one observation tensor with saved SB3 running statistics."""

    mean = torch.as_tensor(running_stats.mean, device=value.device, dtype=value.dtype)
    variance = torch.as_tensor(running_stats.var, device=value.device, dtype=value.dtype)
    normalized = (value - mean) / torch.sqrt(variance + vec_normalize.epsilon)
    return torch.clamp(normalized, -vec_normalize.clip_obs, vec_normalize.clip_obs)


def normalize_observation(obs, vec_normalize):
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


def is_sb3_recurrent_policy(policy) -> bool:
    """Return whether the SB3 policy exposes the recurrent actor interface."""
    return all(hasattr(policy, attribute) for attribute in ("lstm_actor", "lstm_hidden_state_shape", "_predict"))


def initialize_sb3_recurrent_state(policy, num_envs: int):
    """Create the actor LSTM hidden and cell state expected by an SB3 recurrent policy."""

    shape = policy.lstm_hidden_state_shape
    state_shape = (shape[0], num_envs, shape[2])
    zeros = torch.zeros(state_shape, device=policy.device, dtype=torch.float32)
    return (zeros.clone(), zeros.clone())


def _policy_actions(policy, obs, recurrent_state=None):
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
            obs,
            lstm_states=tuple(recurrent_state),
            episode_starts=episode_starts,
            deterministic=True,
        )
    return _scale_or_clip_actions(policy, actions), next_state


def _scale_or_clip_actions(policy, actions):
    """Match the Box-action post-processing performed by :meth:`BasePolicy.predict`."""

    action_space = policy.action_space
    if not hasattr(action_space, "low") or not hasattr(action_space, "high"):
        return actions
    low = torch.as_tensor(action_space.low, device=actions.device, dtype=actions.dtype)
    high = torch.as_tensor(action_space.high, device=actions.device, dtype=actions.dtype)
    if policy.squash_output:
        return low + 0.5 * (actions + 1.0) * (high - low)
    return torch.maximum(torch.minimum(actions, high), low)


def _load_agent(checkpoint_path: str, device: str):
    """Load PPO or RecurrentPPO according to the policy class stored in the checkpoint."""

    try:
        checkpoint_data, _, _ = load_from_zip_file(checkpoint_path, device=device)
    except ModuleNotFoundError as e:
        if e.name is not None and e.name.startswith("sb3_contrib"):
            raise ImportError(
                "Loading a recurrent SB3 checkpoint requires sb3-contrib. "
                "Install the Isaac Lab SB3 optional dependencies."
            ) from e
        raise

    policy_class = checkpoint_data.get("policy_class")
    recurrent_checkpoint = getattr(policy_class, "__module__", "").startswith("sb3_contrib")
    if recurrent_checkpoint:
        if RecurrentPPO is None:
            raise ImportError(
                "Loading a recurrent SB3 checkpoint requires sb3-contrib. "
                "Install the Isaac Lab SB3 optional dependencies."
            )
        return RecurrentPPO.load(checkpoint_path, device=device, print_system_info=True)
    return PPO.load(checkpoint_path, device=device, print_system_info=True)


def _resolve_checkpoint(args_cli: argparse.Namespace, task_name: str, env_cfg) -> str | None:
    """Resolve the SB3 checkpoint selected by the export arguments."""

    if args_cli.checkpoint == "pretrained":
        backend_names = get_pretrained_checkpoint_backend_names(env_cfg)
        return get_published_pretrained_checkpoint("sb3", task_name, *backend_names)

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
        log_root_path,
        ".*",
        r"model_.*\.zip",
        sort_alpha=False,
        preferred_checkpoint=r"model\.zip",
    )


def export_sb3_agent(
    args_cli: argparse.Namespace,
    env_cfg,
    agent_cfg,
    simulation_app=None,
) -> bool:
    """Export a Stable-Baselines3 policy."""

    task_name = args_cli.task.split(":")[-1]
    checkpoint_task_name = task_name.replace("-Play", "")
    checkpoint_path = _resolve_checkpoint(args_cli, checkpoint_task_name, env_cfg)
    if not checkpoint_path:
        print(f"[INFO] No checkpoint found for task: {checkpoint_task_name}")
        return False

    env_cfg.scene.num_envs = 1
    env_cfg.seed = agent_cfg["seed"]
    cli_device = getattr(args_cli, "device", None)
    env_cfg.sim.device = cli_device if cli_device is not None else env_cfg.sim.device

    log_dir = os.path.dirname(checkpoint_path)
    env_cfg.log_dir = log_dir

    env = None
    leapp_started = False
    # SB3 constructs torch.distributions.Normal even for deterministic PPO
    # inference. Its eager argument validation reduces tensor predicates to
    # Python booleans, which is not representable in a LEAPP static graph and
    # is unrelated to the action computation. Disable it only while tracing
    # and restore the process-wide default before returning.
    previous_validate_args = torch.distributions.Distribution._validate_args
    torch.distributions.Distribution.set_default_validate_args(False)
    try:
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        if not isinstance(env.unwrapped, ManagerBasedRLEnv):
            if args_cli.export_method is not None:
                raise ValueError(
                    "--export_method is only supported for manager-based environments. For direct environments, "
                    "set export_with directly in the annotate.output_tensors() call instead."
                )
            raise NotImplementedError("SB3 LEAPP export currently supports manager-based environments only.")

        export_method = "onnx-dynamo" if args_cli.export_method is None else args_cli.export_method

        policy_node_name = ensure_env_spec_id(env)
        graph_name = args_cli.export_task_name if args_cli.export_task_name is not None else task_name
        patch_env_for_export(env, export_method=export_method, required_obs_groups={"policy"})

        print(f"[INFO] Loading model checkpoint from: {checkpoint_path}")
        agent = _load_agent(checkpoint_path, device=env.unwrapped.device)
        policy = agent.policy

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

        if args_cli.export_save_path is not None:
            save_path = args_cli.export_save_path
        elif args_cli.checkpoint == "pretrained":
            save_path = os.path.join(".pretrained_checkpoints", "sb3", checkpoint_task_name)
        else:
            save_path = log_dir

        leapp.start(graph_name, save_path=save_path, max_cached_io=max(args_cli.validation_steps, 2))
        leapp_started = True

        obs = env.reset()[0]["policy"]
        recurrent_state = (
            initialize_sb3_recurrent_state(policy, env.num_envs) if is_sb3_recurrent_policy(policy) else None
        )
        if simulation_app is not None:
            while not simulation_app.is_running():
                time.sleep(0.5)

        for _ in range(max(args_cli.validation_steps, 2)):
            with torch.inference_mode():
                obs = normalize_observation(obs, vec_normalize)
                if recurrent_state is not None:
                    state_tensors = state_dict_from_sequence(recurrent_state)
                    registered_state = annotate.state_tensors(
                        policy_node_name,
                        state_tensors,
                    )
                    recurrent_state = tuple(
                        state_sequence_from_registered(registered_state, list(state_tensors), recurrent_state)
                    )

                actions, next_recurrent_state = _policy_actions(policy, obs, recurrent_state)
                obs_dict, _, terminated, truncated, _ = env.step(actions)
                obs = obs_dict["policy"]

                if next_recurrent_state is not None:
                    not_done = (~(terminated | truncated)).to(dtype=next_recurrent_state[0].dtype)
                    not_done = not_done.reshape(1, -1, 1)
                    recurrent_state = tuple(state * not_done for state in next_recurrent_state)
                    annotate.update_state(policy_node_name, state_dict_from_sequence(recurrent_state))

        leapp.stop()
        leapp_started = False
        validate = args_cli.validation_steps > 0
        leapp.compile_graph(
            visualize=not args_cli.disable_graph_visualization,
            validate=validate,
            graph_configs=create_graph_configs(env_cfg),
        )
    finally:
        torch.distributions.Distribution.set_default_validate_args(previous_validate_args)
        if leapp_started:
            with contextlib.suppress(Exception):
                leapp.stop()
        if env is not None:
            env.close()

    return True


def run_export_with_hydra(args_cli: argparse.Namespace, hydra_args: list[str]) -> bool:
    """Resolve Hydra task configuration and export one SB3 policy."""

    original_argv = sys.argv
    sys.argv = [sys.argv[0]] + hydra_args
    exported = False
    try:

        @hydra_task_config(args_cli.task, args_cli.agent)
        def _main(env_cfg, agent_cfg) -> None:
            nonlocal exported
            with launch_simulation(env_cfg, args_cli):
                exported = export_sb3_agent(args_cli, env_cfg, agent_cfg)

        _main()
    finally:
        sys.argv = original_argv
    return exported


def main_cli(argv: list[str] | None = None) -> bool:
    """Run the command-line export flow."""
    args_cli, hydra_args = parse_export_args(argv)
    return run_export_with_hydra(args_cli, hydra_args)


def run(argv: list[str] | None = None) -> int:
    """Run the export backend and return a process exit code."""
    return 0 if main_cli(argv) else 1


if __name__ == "__main__":
    raise SystemExit(run())
