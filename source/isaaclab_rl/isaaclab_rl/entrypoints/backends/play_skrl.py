# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""skrl playback backend of the unified reinforcement learning entrypoint.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in a more
user-friendly way.
"""

from __future__ import annotations

import argparse
import contextlib
import os

import skrl

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.utils.seed import configure_seed

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, resolve_task_config, setup_preset_cli

from ...skrl import (
    SkrlVecEnvWrapper,
    check_skrl_version,
    import_skrl_runner,
    resolve_skrl_agent_cfg_entry_point,
    resolve_skrl_algorithm,
)
from ..common import (
    CHECKPOINT_SELECTORS,
    add_common_play_args,
    apply_env_overrides,
    apply_video_recording,
    create_isaaclab_env,
    enable_cameras_for_video,
    normalize_task_name,
    pre_launch_video_config,
    preserve_attribute,
    resolve_checkpoint_selector,
    resolve_play_task_name,
    resolve_published_checkpoint,
    resolve_seed,
    run_playback,
    set_hydra_args,
    show_run_summary,
    startup_screen,
)
from . import cli_args_skrl as cli_args

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse skrl playback arguments."""
    parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
    add_common_play_args(
        parser,
        agent_default=None,
        agent_help="Agent configuration entry point (default: the task's canonical SKRL configuration).",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path, latest/best, or pretrained.")
    cli_args.add_skrl_args(parser)
    add_launcher_args(parser)
    args_cli, hydra_args = setup_preset_cli(parser, argv)
    args_cli.task = resolve_play_task_name(args_cli.task)
    enable_cameras_for_video(args_cli)
    set_hydra_args(hydra_args)
    return args_cli


def _resolve_checkpoint(
    args_cli: argparse.Namespace, env_cfg: object, log_root_path: str, algorithm: str, metadata: dict[str, str]
) -> str | None:
    """Resolve the checkpoint to play, or None when no published checkpoint exists."""
    if args_cli.checkpoint == "pretrained":
        return resolve_published_checkpoint("skrl", args_cli.task, env_cfg)
    if args_cli.checkpoint in CHECKPOINT_SELECTORS:
        return resolve_checkpoint_selector(
            log_root_path,
            args_cli.checkpoint,
            library="skrl",
            task=normalize_task_name(args_cli.task),
            checkpoint_pattern=r".*",
            other_dirs=["checkpoints"],
            metadata=metadata,
        )
    if args_cli.checkpoint:
        return os.path.abspath(args_cli.checkpoint)
    return get_checkpoint_path(
        log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
    )


def run(argv: list[str]) -> None:
    """Play a checkpoint of a skrl agent while restoring the caller's global skrl settings."""
    args_cli = _parse_args(argv)
    with contextlib.ExitStack() as cleanup:
        if args_cli.ml_framework == "jax":
            cleanup.enter_context(preserve_attribute(skrl.config.jax, "backend"))
            skrl.config.jax.backend = "jax"
        _run(args_cli)


def _run(args_cli: argparse.Namespace) -> None:
    """Execute skrl playback with parsed arguments."""
    check_skrl_version()
    agent_cfg_entry_point = resolve_skrl_agent_cfg_entry_point(args_cli.agent, args_cli.algorithm)
    with startup_screen(args_cli, num_stages=3) as screen:
        env_cfg, agent_cfg = resolve_task_config(
            args_cli.task, agent_cfg_entry_point, play_mode=not args_cli.train_env_cfg
        )
        algorithm = resolve_skrl_algorithm(agent_cfg, args_cli.algorithm)
        pre_launch_video_config(env_cfg, args_cli)
        show_run_summary(screen, args_cli, env_cfg, library="skrl", action="play")
        screen.stage("Launching simulation")
        with launch_simulation(env_cfg, args_cli):
            runner_cls = import_skrl_runner(args_cli.ml_framework)
            apply_env_overrides(args_cli, env_cfg)
            args_cli.seed = resolve_seed(args_cli.seed)
            if args_cli.seed is not None:
                agent_cfg["seed"] = args_cli.seed
            env_cfg.seed = agent_cfg["seed"]

            experiment_cfg = agent_cfg["agent"]["experiment"]
            log_root_path = os.path.abspath(os.path.join("logs", "skrl", experiment_cfg["directory"]))
            print(f"[INFO] Loading experiment from directory: {log_root_path}")
            manifest_metadata = {
                "agent": agent_cfg_entry_point,
                "algorithm": algorithm,
                "ml_framework": args_cli.ml_framework,
            }
            resume_path = _resolve_checkpoint(args_cli, env_cfg, log_root_path, algorithm, manifest_metadata)
            if resume_path is None:
                return
            log_dir = os.path.dirname(os.path.dirname(resume_path))
            env_cfg.log_dir = log_dir
            apply_video_recording(env_cfg, log_dir, args_cli, subdir="play")

            screen.stage("Creating environment")
            env = create_isaaclab_env(
                args_cli.task,
                env_cfg,
                args_cli,
                convert_marl_to_single_agent=isinstance(env_cfg, DirectMARLEnvCfg) and algorithm == "ppo",
            )
            dt = env.unwrapped.step_dt

            screen.stage("Loading policy")
            env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
            agent_cfg["trainer"]["close_environment_at_exit"] = False
            experiment_cfg["write_interval"] = 0
            experiment_cfg["checkpoint_interval"] = 0
            runner = runner_cls(env, agent_cfg)
            # configure_seed must run after Runner() so torch determinism does not disturb its initialization
            if args_cli.deterministic:
                configure_seed(env_cfg.seed, torch_deterministic=True)
            print(f"[INFO] Loading model checkpoint from: {resume_path}")
            runner.agent.load(resume_path)
            runner.agent.enable_training_mode(False, apply_to_models=True)

            obs, _ = env.reset()
            states = env.state()

            def step() -> None:
                nonlocal obs, states
                outputs = runner.agent.act(obs, states, timestep=0, timesteps=0)
                if hasattr(env, "possible_agents"):
                    actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
                else:
                    actions = outputs[-1].get("mean_actions", outputs[0])
                obs, _, _, _, _ = env.step(actions)
                states = env.state()

            screen.close()
            run_playback(step, dt=dt, args_cli=args_cli, env_cfg=env_cfg)
            env.close()
