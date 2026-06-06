# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stable-Baselines3 training logic for the unified reinforcement learning entrypoint."""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import random
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

from common import (
    add_common_train_args,
    add_isaaclab_launcher_args,
    apply_env_overrides,
    configure_io_descriptors,
    create_isaaclab_env,
    dump_train_configs,
    enable_cameras_for_video,
    set_hydra_args,
    wrap_record_video,
)

import isaaclab_tasks  # noqa: F401

logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


def _cleanup_pbar(*args):
    """Stop training and clean up rich progress bars on Ctrl+C."""
    import gc

    tqdm_objects = [obj for obj in gc.get_objects() if "tqdm" in type(obj).__name__]
    for tqdm_object in tqdm_objects:
        if "tqdm_rich" in type(tqdm_object).__name__:
            tqdm_object.close()
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
    parser.add_argument("--checkpoint", type=str, default=None, help="Continue the training from checkpoint.")
    parser.add_argument(
        "--keep_all_info",
        action="store_true",
        default=False,
        help="Use a slower SB3 wrapper but keep all the extra training info.",
    )
    from isaaclab_tasks.utils import setup_preset_cli

    add_isaaclab_launcher_args(parser)
    # setup_preset_cli registers preset-selection help text + runs parse_known_args; the
    # physics=/renderer=/presets= tokens pass through the remainder for hydra to parse later.
    args_cli, hydra_args = setup_preset_cli(parser, argv)
    enable_cameras_for_video(args_cli)
    set_hydra_args(hydra_args)
    return args_cli


def run(argv: list[str]) -> None:
    """Train a Stable-Baselines3 agent."""
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, LogEveryNTimesteps
    from stable_baselines3.common.vec_env import VecNormalize

    from isaaclab.envs import DirectMARLEnvCfg

    from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

    from isaaclab_tasks.utils import launch_simulation, resolve_task_config

    signal.signal(signal.SIGINT, _cleanup_pbar)

    args_cli = _parse_args(argv)
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)

    with launch_simulation(env_cfg, args_cli):
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)

        apply_env_overrides(args_cli, env_cfg)
        agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
        if args_cli.max_iterations is not None:
            agent_cfg["n_timesteps"] = args_cli.max_iterations * agent_cfg["n_steps"] * env_cfg.scene.num_envs

        env_cfg.seed = agent_cfg["seed"]

        run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_root_path = os.path.abspath(os.path.join("logs", "sb3", args_cli.task))
        print(f"[INFO] Logging experiment in directory: {log_root_path}")
        print(f"Exact experiment name requested from command line: {run_info}")
        log_dir = os.path.join(log_root_path, run_info)
        dump_train_configs(log_dir, env_cfg, agent_cfg)

        command = " ".join(sys.orig_argv)
        (Path(log_dir) / "command.txt").write_text(command)

        agent_cfg = process_sb3_cfg(agent_cfg, env_cfg.scene.num_envs)
        policy_arch = agent_cfg.pop("policy")
        n_timesteps = agent_cfg.pop("n_timesteps")

        configure_io_descriptors(env_cfg, args_cli, logger)
        env_cfg.log_dir = log_dir

        env = create_isaaclab_env(
            args_cli.task,
            env_cfg,
            args_cli,
            convert_marl_to_single_agent=isinstance(env_cfg, DirectMARLEnvCfg),
        )
        env = wrap_record_video(env, log_dir, args_cli)

        start_time = time.time()
        env = Sb3VecEnvWrapper(env, fast_variant=not args_cli.keep_all_info)

        norm_keys = {"normalize_input", "normalize_value", "clip_obs"}
        norm_args = {}
        for key in norm_keys:
            if key in agent_cfg:
                norm_args[key] = agent_cfg.pop(key)

        if norm_args and norm_args.get("normalize_input"):
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

        agent = PPO(policy_arch, env, verbose=1, tensorboard_log=log_dir, **agent_cfg)
        if args_cli.checkpoint is not None:
            agent = agent.load(args_cli.checkpoint, env, print_system_info=True)

        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_dir, name_prefix="model", verbose=2)
        callbacks = [checkpoint_callback, LogEveryNTimesteps(n_steps=args_cli.log_interval)]

        with contextlib.suppress(KeyboardInterrupt):
            agent.learn(
                total_timesteps=n_timesteps,
                callback=callbacks,
                progress_bar=True,
                log_interval=None,
            )

        agent.save(os.path.join(log_dir, "model"))
        print("Saving to:")
        print(os.path.join(log_dir, "model.zip"))

        if isinstance(env, VecNormalize):
            print("Saving normalization")
            env.save(os.path.join(log_dir, "model_vecnormalize.pkl"))

        print(f"Training time: {round(time.time() - start_time, 2)} seconds")
        env.close()
