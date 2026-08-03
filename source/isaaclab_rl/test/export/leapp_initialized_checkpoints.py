# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for generating backend-native initialized checkpoints for LEAPP export tests."""

from __future__ import annotations

import contextlib
import copy
import importlib.metadata as metadata
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def discover_backend_tasks(agent_cfg_entry_points: Sequence[str]) -> tuple[str, ...]:
    """Discover registered Isaac Lab tasks that expose any of the requested agent entry points."""
    import gymnasium as gym

    import isaaclab_tasks  # noqa: F401

    with contextlib.suppress(ImportError):
        import isaaclab_tasks_experimental  # noqa: F401

    task_names = set()
    for task_spec in gym.registry.values():
        if "Isaac" not in task_spec.id or "Direct" in task_spec.id:
            continue
        spec_kwargs = gym.spec(task_spec.id).kwargs
        if any(spec_kwargs.get(entry_point) is not None for entry_point in agent_cfg_entry_points):
            task_names.add(task_spec.id)
    return tuple(sorted(task_names))


def create_initialized_checkpoint(
    backend_id: str,
    task_name: str,
    args_cli: Any,
    env_cfg: Any,
    agent_cfg: Any,
    checkpoint_root: Path,
) -> Path:
    """Create a structurally valid initialized checkpoint through the requested backend's standard APIs."""
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_creators = {
        "rsl_rl": _create_rsl_rl_checkpoint,
        "rl_games": _create_rl_games_checkpoint,
        "skrl": _create_skrl_checkpoint,
        "sb3": _create_sb3_checkpoint,
    }
    if backend_id not in checkpoint_creators:
        raise ValueError(f"Unsupported LEAPP export backend: {backend_id}")

    try:
        return checkpoint_creators[backend_id](task_name, args_cli, env_cfg, agent_cfg, checkpoint_root)
    except Exception as exc:
        raise RuntimeError(f"Failed to create initialized {backend_id} checkpoint for task {task_name}.") from exc


def _set_single_env(env_cfg: Any) -> None:
    """Keep export-test checkpoint creation cheap and aligned with exporter assumptions."""
    env_cfg.scene.num_envs = 1


def _create_rsl_rl_checkpoint(
    task_name: str,
    args_cli: Any,
    env_cfg: Any,
    agent_cfg: Any,
    checkpoint_root: Path,
) -> Path:
    """Create an initialized RSL-RL checkpoint using the runner save path."""
    import gymnasium as gym
    from rsl_rl.runners import DistillationRunner, OnPolicyRunner

    from isaaclab.envs import DirectMARLEnvCfg, multi_agent_to_single_agent

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

    _set_single_env(env_cfg)
    installed_version = metadata.version("rsl-rl-lib")
    if getattr(args_cli, "seed", None) is not None:
        agent_cfg.seed = args_cli.seed
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.seed = agent_cfg.seed
    env_cfg.log_dir = str(checkpoint_root)

    env = None
    try:
        env = gym.make(task_name, cfg=env_cfg, render_mode=None)
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            env = multi_agent_to_single_agent(env)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(checkpoint_root), device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=str(checkpoint_root), device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported RSL-RL runner class: {agent_cfg.class_name}")

        checkpoint_path = checkpoint_root / "model_0.pt"
        runner.save(str(checkpoint_path), infos={"initialized_for_leapp_export_test": True})
        return checkpoint_path
    finally:
        if env is not None:
            env.close()


def _create_rl_games_checkpoint(
    task_name: str,
    args_cli: Any,
    env_cfg: Any,
    agent_cfg: dict[str, Any],
    checkpoint_root: Path,
) -> Path:
    """Create an initialized RL-Games checkpoint using its training agent save path."""
    import gymnasium as gym
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner

    from isaaclab.envs import DirectMARLEnvCfg, multi_agent_to_single_agent

    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

    agent_cfg = copy.deepcopy(agent_cfg)
    _set_single_env(env_cfg)
    agent_cfg["params"]["seed"] = getattr(args_cli, "seed", None) or agent_cfg["params"]["seed"]
    env_cfg.seed = agent_cfg["params"]["seed"]
    env_cfg.log_dir = str(checkpoint_root)

    config = agent_cfg["params"]["config"]
    config["train_dir"] = str(checkpoint_root)
    config["full_experiment_name"] = "initialized"
    rl_device = config["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    env = None
    try:
        env = gym.make(task_name, cfg=env_cfg, render_mode=None)
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            env = multi_agent_to_single_agent(env)
        env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
        )
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
        config["num_actors"] = env.unwrapped.num_envs
        # The shipped minibatch_size assumes the full training env count. RL-Games asserts
        # batch_size % minibatch_size == 0, so shrink it to the single-env batch size.
        config["minibatch_size"] = config["horizon_length"] * config["num_actors"]

        runner = Runner()
        runner.load(agent_cfg)
        agent = runner.algo_factory.create(runner.algo_name, base_name="run", params=runner.params)

        # RL-Games appends the ``.pth`` suffix itself, so save without it.
        agent.save(str(checkpoint_root / config["name"]))
        return checkpoint_root / f"{config['name']}.pth"
    finally:
        if env is not None:
            env.close()


def _create_skrl_checkpoint(
    task_name: str,
    args_cli: Any,
    env_cfg: Any,
    agent_cfg: dict[str, Any],
    checkpoint_root: Path,
) -> Path:
    """Create an initialized skrl checkpoint using the agent save path."""
    import gymnasium as gym
    from skrl.utils.runner.torch import Runner

    from isaaclab.envs import DirectMARLEnvCfg, multi_agent_to_single_agent

    from isaaclab_rl.skrl import SkrlVecEnvWrapper

    agent_cfg = copy.deepcopy(agent_cfg)
    _set_single_env(env_cfg)
    agent_cfg["seed"] = getattr(args_cli, "seed", None) or agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.log_dir = str(checkpoint_root)
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["directory"] = str(checkpoint_root)
    agent_cfg["agent"]["experiment"]["experiment_name"] = "initialized"
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    env = None
    try:
        env = gym.make(task_name, cfg=env_cfg, render_mode=None)
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg) and getattr(args_cli, "algorithm", "ppo").lower() == "ppo":
            env = multi_agent_to_single_agent(env)
        env = SkrlVecEnvWrapper(env, ml_framework="torch")

        runner = Runner(env, agent_cfg)
        checkpoint_path = checkpoint_root / "agent.pt"
        runner.agent.save(str(checkpoint_path))
        return checkpoint_path
    finally:
        if env is not None:
            env.close()


def _create_sb3_checkpoint(
    task_name: str,
    args_cli: Any,
    env_cfg: Any,
    agent_cfg: dict[str, Any],
    checkpoint_root: Path,
) -> Path:
    """Create an initialized SB3 checkpoint using PPO.save."""
    import gymnasium as gym
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    from isaaclab.envs import DirectMARLEnvCfg, ManagerBasedRLEnv, multi_agent_to_single_agent

    from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

    agent_cfg = copy.deepcopy(agent_cfg)
    _set_single_env(env_cfg)
    agent_cfg["seed"] = getattr(args_cli, "seed", None) or agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.log_dir = str(checkpoint_root)

    env = None
    try:
        env = gym.make(task_name, cfg=env_cfg, render_mode=None)
        if not isinstance(env.unwrapped, ManagerBasedRLEnv):
            raise NotImplementedError("SB3 LEAPP export currently supports manager-based environments only.")
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            env = multi_agent_to_single_agent(env)
        env = Sb3VecEnvWrapper(env, fast_variant=True)

        agent_cfg = process_sb3_cfg(agent_cfg, env_cfg.scene.num_envs)
        policy_arch = agent_cfg.pop("policy")
        agent_cfg.pop("n_timesteps", None)

        norm_keys = {"normalize_input", "normalize_value", "clip_obs"}
        norm_args = {key: agent_cfg.pop(key) for key in norm_keys if key in agent_cfg}
        if norm_args and norm_args.get("normalize_input"):
            env = VecNormalize(
                env,
                training=True,
                norm_obs=norm_args["normalize_input"],
                norm_reward=norm_args.get("normalize_value", False),
                clip_obs=norm_args.get("clip_obs", 100.0),
                gamma=agent_cfg["gamma"],
                clip_reward=np.inf,
            )

        agent = PPO(policy_arch, env, verbose=0, tensorboard_log=str(checkpoint_root), **agent_cfg)
        checkpoint_without_suffix = checkpoint_root / "model"
        agent.save(str(checkpoint_without_suffix))
        if isinstance(env, VecNormalize):
            env.save(str(checkpoint_root / "model_vecnormalize.pkl"))
        return checkpoint_root / "model.zip"
    finally:
        if env is not None:
            env.close()


def _agent_cfg_entry_point(backend_id: str) -> str:
    """Return the default Hydra agent entry point for *backend_id*."""
    return f"{backend_id}_cfg_entry_point"


def resolved_path_file(checkpoint_dir: Path) -> Path:
    """Return the file that records the created checkpoint path for one task."""
    return checkpoint_dir / "checkpoint_path.txt"


def task_checkpoint_dir(checkpoint_root: Path, backend_id: str, task_name: str) -> Path:
    """Return the per-backend/task directory used for one initialized checkpoint."""
    return checkpoint_root / backend_id / task_name


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint: create initialized checkpoints in one Kit process.

    Accepts one or more ``--spec BACKEND TASK [PRESET]`` entries and writes each
    checkpoint under ``checkpoint_root/BACKEND/TASK/``. When a per-spec preset is
    omitted, ``--preset`` is used if provided; otherwise no Hydra preset token is
    applied.

    Usage:
        python leapp_initialized_checkpoints.py --checkpoint_root /tmp/ckpt \\
            --preset newton_mjwarp \\
            --spec rsl_rl Isaac-Cartpole \\
            --spec rsl_rl IsaacContrib-Lift-Cube-Franka _
    """
    import argparse
    import os
    import sys
    import traceback
    from types import SimpleNamespace

    parser = argparse.ArgumentParser(description="Create initialized checkpoints for LEAPP export tests.")
    parser.add_argument(
        "--spec",
        nargs="+",
        action="append",
        metavar="BACKEND_TASK_PRESET",
        required=True,
        help="Backend/task pair, optionally with a preset override: BACKEND TASK [PRESET]. "
        "Use PRESET=_ to force no preset even when --preset is set. Repeat for multiple "
        "checkpoints in one Kit process.",
    )
    parser.add_argument("--checkpoint_root", required=True, type=Path)
    parser.add_argument(
        "--preset",
        default=None,
        help="Default Hydra preset for specs that do not override it.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--algorithm", type=str, default="ppo", help="skrl algorithm name.")
    args = parser.parse_args(argv)

    backend_choices = ("rsl_rl", "rl_games", "skrl", "sb3")
    specs: list[tuple[str, str, str | None]] = []
    for raw_spec in args.spec:
        if len(raw_spec) not in (2, 3):
            raise SystemExit(f"Each --spec must be 'BACKEND TASK' or 'BACKEND TASK PRESET', got: {raw_spec}")
        backend_id, task_name = raw_spec[0], raw_spec[1]
        if backend_id not in backend_choices:
            raise SystemExit(f"Unsupported backend '{backend_id}'. Choose from {backend_choices}.")
        if len(raw_spec) == 3:
            preset = None if raw_spec[2] in ("_", "") else raw_spec[2]
        else:
            preset = args.preset
        specs.append((backend_id, task_name, preset))

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    from isaaclab.app.settings_manager import get_settings_manager

    from isaaclab_tasks.utils.hydra import resolve_task_config

    get_settings_manager().set_bool("/physics/cooking/ujitsoCollisionCooking", False)
    get_settings_manager().set_bool("/isaaclab/render/rtx_sensors", False)

    cli_args = SimpleNamespace(seed=args.seed, algorithm=args.algorithm)
    failures: list[str] = []
    original_argv = sys.argv
    try:
        for backend_id, task_name, preset in specs:
            task_dir = task_checkpoint_dir(args.checkpoint_root, backend_id, task_name)
            task_dir.mkdir(parents=True, exist_ok=True)
            sys.argv = [sys.argv[0], f"presets={preset}"] if preset else [sys.argv[0]]
            try:
                env_cfg, agent_cfg = resolve_task_config(task_name, _agent_cfg_entry_point(backend_id))
                checkpoint_path = create_initialized_checkpoint(
                    backend_id,
                    task_name,
                    cli_args,
                    env_cfg,
                    agent_cfg,
                    task_dir,
                )
                resolved_path_file(task_dir).write_text(str(checkpoint_path))
            except BaseException:
                failures.append(f"{backend_id}/{task_name}")
                traceback.print_exc(file=sys.stdout)
                sys.stdout.flush()
                sys.stderr.flush()
    finally:
        sys.argv = original_argv
        if failures:
            # ``simulation_app.close()`` can exit 0; fail hard before it when any
            # checkpoint was missing so the parent subprocess call sees the error.
            print(f"[ERROR] Failed to create checkpoints for: {', '.join(failures)}", flush=True)
            os._exit(1)
        simulation_app.close()


if __name__ == "__main__":
    main()
