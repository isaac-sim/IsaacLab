# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stable-Baselines3 training backend of the unified reinforcement learning entrypoint."""

from __future__ import annotations

import argparse
import contextlib
import gc
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, LogEveryNTimesteps
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.app import add_launcher_args, launch_simulation, report_activity
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.utils.seed import configure_seed

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

from ...sb3 import Sb3VecEnvWrapper, process_sb3_cfg
from ..common import (
    CHECKPOINT_SELECTORS,
    add_common_train_args,
    apply_env_overrides,
    apply_video_recording,
    create_isaaclab_env,
    dump_train_configs,
    enable_cameras_for_video,
    pre_launch_video_config,
    resolve_checkpoint_selector,
    resolve_seed,
    set_hydra_args,
    show_run_summary,
    startup_screen,
    wrap_sensor_capture,
    write_run_manifest,
)

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


def _cleanup_pbar(*args) -> None:
    """Stop training and close the rich progress bars on Ctrl+C."""
    for obj in gc.get_objects():
        if "tqdm_rich" in type(obj).__name__:
            obj.close()
    raise KeyboardInterrupt


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse Stable-Baselines3 training arguments."""
    parser = argparse.ArgumentParser(description="Train an RL agent with Stable-Baselines3.")
    add_common_train_args(
        parser,
        agent_default="sb3_cfg_entry_point",
        agent_help="Name of the RL agent configuration entry point.",
        include_distributed=False,
    )
    parser.add_argument("--log_interval", type=int, default=100_000, help="Log data every n timesteps.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path, or latest/best.")
    parser.add_argument(
        "--keep_all_info",
        action="store_true",
        default=False,
        help="Use a slower SB3 wrapper but keep all the extra training info.",
    )
    add_launcher_args(parser)
    args_cli, hydra_args = setup_preset_cli(parser, argv)
    enable_cameras_for_video(args_cli)
    set_hydra_args(hydra_args)
    return args_cli


def _resolve_checkpoint(args_cli: argparse.Namespace, log_root_path: str) -> str | None:
    """Resolve the checkpoint to resume from, or None when training starts from scratch."""
    if args_cli.checkpoint in CHECKPOINT_SELECTORS:
        return resolve_checkpoint_selector(
            log_root_path,
            args_cli.checkpoint,
            library="sb3",
            task=args_cli.task,
            checkpoint_pattern=r"model(?:_.*)?\.zip",
            preferred_checkpoint_pattern=r"model\.zip",
            metadata={"agent": args_cli.agent},
        )
    return args_cli.checkpoint


def run(argv: list[str]) -> None:
    """Train a Stable-Baselines3 agent."""
    signal.signal(signal.SIGINT, _cleanup_pbar)
    args_cli = _parse_args(argv)
    with startup_screen(args_cli, num_stages=3) as screen:
        env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)
        pre_launch_video_config(env_cfg, args_cli)
        show_run_summary(screen, args_cli, env_cfg, library="sb3", action="train")
        screen.stage("Launching simulation")
        with launch_simulation(env_cfg, args_cli):
            apply_env_overrides(args_cli, env_cfg)
            args_cli.seed = resolve_seed(args_cli.seed)
            if args_cli.seed is not None:
                agent_cfg["seed"] = args_cli.seed
            if args_cli.max_iterations is not None:
                agent_cfg["n_timesteps"] = args_cli.max_iterations * agent_cfg["n_steps"] * env_cfg.scene.num_envs
            env_cfg.seed = agent_cfg["seed"]

            log_root_path = os.path.abspath(os.path.join("logs", "sb3", args_cli.task))
            print(f"[INFO] Logging experiment in directory: {log_root_path}")
            run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            print(f"Exact experiment name requested from command line: {run_name}")
            log_dir = os.path.join(log_root_path, run_name)
            write_run_manifest(log_dir, library="sb3", task=args_cli.task, metadata={"agent": args_cli.agent})
            dump_train_configs(log_dir, env_cfg, agent_cfg)
            (Path(log_dir) / "command.txt").write_text(" ".join(sys.orig_argv))

            checkpoint_path = _resolve_checkpoint(args_cli, log_root_path)
            agent_cfg = process_sb3_cfg(agent_cfg, env_cfg.scene.num_envs)
            policy_arch = agent_cfg.pop("policy")
            n_timesteps = agent_cfg.pop("n_timesteps")
            norm_args = {
                key: agent_cfg.pop(key)
                for key in ("normalize_input", "normalize_value", "clip_obs")
                if key in agent_cfg
            }

            env_cfg.log_dir = log_dir
            apply_video_recording(env_cfg, log_dir, args_cli)

            screen.stage("Creating environment")
            env = create_isaaclab_env(
                args_cli.task,
                env_cfg,
                args_cli,
                convert_marl_to_single_agent=isinstance(env_cfg, DirectMARLEnvCfg),
            )
            env = wrap_sensor_capture(env, log_dir, args_cli)

            screen.stage("Preparing agent")
            start_time = time.time()
            report_activity("Wrapping environment")
            env = Sb3VecEnvWrapper(env, fast_variant=not args_cli.keep_all_info)
            report_activity(None)
            if norm_args.get("normalize_input"):
                print(f"Normalizing input, {norm_args=}")
                env = VecNormalize(
                    env,
                    training=True,
                    norm_obs=norm_args["normalize_input"],
                    norm_reward=norm_args.get("normalize_value", False),
                    clip_obs=norm_args.get("clip_obs", 100.0),
                    gamma=agent_cfg["gamma"],
                    clip_reward=np.inf,
                )

            report_activity("Building policy")
            agent = PPO(policy_arch, env, verbose=1, tensorboard_log=log_dir, **agent_cfg)
            report_activity(None)
            if checkpoint_path is not None:
                agent = agent.load(checkpoint_path, env, print_system_info=True)

            # configure_seed must run after PPO construction and loading so torch determinism does not disturb
            # SB3's initialization
            if args_cli.deterministic:
                configure_seed(env_cfg.seed, torch_deterministic=True)

            callbacks = [
                CheckpointCallback(save_freq=1000, save_path=log_dir, name_prefix="model", verbose=2),
                LogEveryNTimesteps(n_steps=args_cli.log_interval),
            ]

            screen.close()
            with contextlib.suppress(KeyboardInterrupt):
                agent.learn(total_timesteps=n_timesteps, callback=callbacks, progress_bar=True, log_interval=None)

            agent.save(os.path.join(log_dir, "model"))
            print(f"Saving to:\n{os.path.join(log_dir, 'model.zip')}")
            if isinstance(env, VecNormalize):
                print("Saving normalization")
                env.save(os.path.join(log_dir, "model_vecnormalize.pkl"))
            print(f"Training time: {round(time.time() - start_time, 2)} seconds")
            env.close()
