# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to export a checkpoint of an RL agent from Stable-Baselines3."""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from pathlib import Path

_RUNTIME_IMPORTS_LOADED = False

torch = None
leapp = None
annotate = None
gym = None
PPO = None
RecurrentPPO = None
ManagerBasedRLEnv = None
retrieve_file_path = None
patch_env_for_export = None
ensure_env_spec_id = None
get_published_pretrained_checkpoint = None
get_checkpoint_path = None
resolve_checkpoint_selector = None
load_from_pkl = None
load_from_zip_file = None
CHECKPOINT_SELECTORS = None
state_dict_from_sequence = None
state_sequence_from_registered = None


def parse_export_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse export arguments and return remaining Hydra overrides."""
    leapp_scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if leapp_scripts_dir not in sys.path:
        sys.path.insert(0, leapp_scripts_dir)
    from export_utils import add_common_export_args, finalize_export_args

    parser = argparse.ArgumentParser(description="Export an RL agent with Stable-Baselines3.")
    add_common_export_args(parser, agent_default="sb3_cfg_entry_point")
    return finalize_export_args(parser, argv, agent_library="sb3")


def _load_runtime_dependencies() -> None:
    """Import runtime dependencies after Isaac Sim has been launched."""
    global _RUNTIME_IMPORTS_LOADED
    global CHECKPOINT_SELECTORS, ManagerBasedRLEnv, PPO, RecurrentPPO, annotate, gym, leapp
    global ensure_env_spec_id, get_checkpoint_path, get_published_pretrained_checkpoint
    global load_from_pkl, load_from_zip_file, patch_env_for_export, resolve_checkpoint_selector, retrieve_file_path
    global state_dict_from_sequence, state_sequence_from_registered, torch

    if _RUNTIME_IMPORTS_LOADED:
        return

    try:
        import leapp as leapp_module
    except ImportError as e:
        raise ImportError("LEAPP package is required for policy export. Install with: pip install leapp") from e
    annotate_module = getattr(leapp_module, "annotate")

    import gymnasium as gym_module
    import torch as torch_module
    from stable_baselines3 import PPO as PPOCls
    from stable_baselines3.common.save_util import load_from_pkl as load_from_pkl_fn
    from stable_baselines3.common.save_util import load_from_zip_file as load_from_zip_file_fn

    try:
        from sb3_contrib import RecurrentPPO as RecurrentPPOCls
    except ImportError:
        RecurrentPPOCls = None

    from isaaclab.envs import ManagerBasedRLEnv as ManagerBasedRLEnvCls
    from isaaclab.utils.assets import retrieve_file_path as retrieve_file_path_fn
    from isaaclab.utils.leapp import patch_env_for_export as patch_env_for_export_fn
    from isaaclab.utils.leapp.utils import ensure_env_spec_id as ensure_env_spec_id_fn

    from isaaclab_rl.entrypoints.common import (
        CHECKPOINT_SELECTORS as CHECKPOINT_SELECTORS_VALUE,
    )
    from isaaclab_rl.entrypoints.common import (
        resolve_checkpoint_selector as resolve_checkpoint_selector_fn,
    )
    from isaaclab_rl.utils.pretrained_checkpoint import (
        get_published_pretrained_checkpoint as get_published_pretrained_checkpoint_fn,
    )

    __import__("isaaclab_tasks")
    from isaaclab_tasks.utils import get_checkpoint_path as get_checkpoint_path_fn

    leapp_scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if leapp_scripts_dir not in sys.path:
        sys.path.insert(0, leapp_scripts_dir)
    from export_utils import (  # isort: skip
        state_dict_from_sequence as state_dict_from_sequence_fn,
        state_sequence_from_registered as state_sequence_from_registered_fn,
    )

    torch = torch_module
    leapp = leapp_module
    annotate = annotate_module
    gym = gym_module
    PPO = PPOCls
    RecurrentPPO = RecurrentPPOCls
    ManagerBasedRLEnv = ManagerBasedRLEnvCls
    retrieve_file_path = retrieve_file_path_fn
    patch_env_for_export = patch_env_for_export_fn
    ensure_env_spec_id = ensure_env_spec_id_fn
    get_published_pretrained_checkpoint = get_published_pretrained_checkpoint_fn
    get_checkpoint_path = get_checkpoint_path_fn
    resolve_checkpoint_selector = resolve_checkpoint_selector_fn
    load_from_pkl = load_from_pkl_fn
    load_from_zip_file = load_from_zip_file_fn
    CHECKPOINT_SELECTORS = CHECKPOINT_SELECTORS_VALUE
    state_dict_from_sequence = state_dict_from_sequence_fn
    state_sequence_from_registered = state_sequence_from_registered_fn
    _RUNTIME_IMPORTS_LOADED = True


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


def _resolve_checkpoint(args_cli: argparse.Namespace, task_name: str) -> str | None:
    """Resolve the SB3 checkpoint selected by the export arguments."""
    if args_cli.use_pretrained_checkpoint:
        return get_published_pretrained_checkpoint("sb3", task_name)

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
    _load_runtime_dependencies()

    task_name = args_cli.task.split(":")[-1]
    checkpoint_task_name = task_name.replace("-Play", "")
    checkpoint_path = _resolve_checkpoint(args_cli, checkpoint_task_name)
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
    try:
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        if not isinstance(env.unwrapped, ManagerBasedRLEnv):
            raise NotImplementedError("SB3 LEAPP export currently supports manager-based environments only.")

        policy_node_name = ensure_env_spec_id(env)
        graph_name = args_cli.export_task_name if args_cli.export_task_name is not None else task_name
        patch_env_for_export(env, export_method=args_cli.export_method, required_obs_groups={"policy"})

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
        elif args_cli.use_pretrained_checkpoint:
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
        leapp.compile_graph(visualize=not args_cli.disable_graph_visualization, validate=validate)
    finally:
        if leapp_started:
            with contextlib.suppress(Exception):
                leapp.stop()
        if env is not None:
            env.close()

    return True


def run_export_with_hydra(args_cli: argparse.Namespace, hydra_args: list[str]) -> bool:
    """Resolve Hydra task configuration and export one SB3 policy."""
    leapp_scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if leapp_scripts_dir not in sys.path:
        sys.path.insert(0, leapp_scripts_dir)
    from export_utils import disable_torchscript_for_export

    # Must run before the imports below pull in the task modules.
    disable_torchscript_for_export()

    from isaaclab.app import launch_simulation

    from isaaclab_tasks.utils.hydra import hydra_task_config

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


if __name__ == "__main__":
    main_cli()
