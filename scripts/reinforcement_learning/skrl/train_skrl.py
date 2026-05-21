# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""skrl training logic for the unified reinforcement learning entrypoint."""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import random
import time
from datetime import datetime

from common import (
    add_common_train_args,
    add_isaaclab_launcher_args,
    apply_env_overrides,
    configure_io_descriptors,
    create_isaaclab_env,
    dump_train_configs,
    enable_cameras_for_video,
    set_hydra_args,
    validate_distributed_device,
    wrap_record_video,
)
from packaging import version

import isaaclab_tasks  # noqa: F401

logger = logging.getLogger(__name__)

SKRL_VERSION = "2.0.0"

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse skrl training arguments."""
    parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
    add_common_train_args(
        parser,
        agent_default=None,
        agent_help=(
            "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
            "--algorithm is used to determine the default agent configuration entry point."
        ),
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
    parser.add_argument(
        "--ml_framework",
        type=str,
        default="torch",
        choices=["torch", "jax"],
        help="The ML framework used for training the skrl agent.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="PPO",
        choices=["AMP", "PPO", "IPPO", "MAPPO"],
        help="The RL algorithm used for training the skrl agent.",
    )
    from isaaclab_tasks.utils import fold_preset_tokens, setup_preset_cli

    add_isaaclab_launcher_args(parser)
    # setup_preset_cli registers preset-selection help text + runs parse_known_args;
    # fold_preset_tokens rewrites typed selectors (physics=, renderer=, presets=) post-argparse.
    args_cli, hydra_args = setup_preset_cli(parser, argv)
    enable_cameras_for_video(args_cli)
    set_hydra_args(fold_preset_tokens(hydra_args))
    return args_cli


def _resolve_agent_entry_point(args_cli: argparse.Namespace) -> tuple[str, str]:
    """Resolve the skrl agent entry point and algorithm from CLI arguments."""
    if args_cli.agent is None:
        algorithm = args_cli.algorithm.lower()
        agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
    else:
        agent_cfg_entry_point = args_cli.agent
        algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()
    return agent_cfg_entry_point, algorithm


def run(argv: list[str]) -> None:
    """Train a skrl agent."""
    import skrl

    from isaaclab.envs import DirectMARLEnvCfg
    from isaaclab.utils.assets import retrieve_file_path

    from isaaclab_rl.skrl import SkrlVecEnvWrapper

    from isaaclab_tasks.utils import launch_simulation, resolve_task_config

    args_cli = _parse_args(argv)

    if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
        skrl.logger.error(
            f"Unsupported skrl version: {skrl.__version__}. "
            f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
        )
        raise SystemExit(1)

    agent_cfg_entry_point, algorithm = _resolve_agent_entry_point(args_cli)
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, agent_cfg_entry_point)

    with launch_simulation(env_cfg, args_cli):
        if args_cli.ml_framework.startswith("torch"):
            from skrl.utils.runner.torch import Runner
        elif args_cli.ml_framework.startswith("jax"):
            from skrl.utils.runner.jax import Runner

        apply_env_overrides(args_cli, env_cfg)
        validate_distributed_device(args_cli)

        if args_cli.distributed:
            global_rank = int(os.getenv("RANK", "0"))

        if args_cli.max_iterations:
            agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
        agent_cfg["trainer"]["close_environment_at_exit"] = False

        if args_cli.ml_framework.startswith("jax"):
            skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)

        agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
        if args_cli.distributed:
            agent_cfg["seed"] = agent_cfg["seed"] + global_rank
        env_cfg.seed = agent_cfg["seed"]

        log_root_path = os.path.abspath(os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"]))
        print(f"[INFO] Logging experiment in directory: {log_root_path}")
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
        print(f"Exact experiment name requested from command line: {log_dir}")
        if agent_cfg["agent"]["experiment"]["experiment_name"]:
            log_dir += f"_{agent_cfg['agent']['experiment']['experiment_name']}"
        agent_cfg["agent"]["experiment"]["directory"] = log_root_path
        agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
        log_dir = os.path.join(log_root_path, log_dir)

        dump_train_configs(log_dir, env_cfg, agent_cfg)

        resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

        configure_io_descriptors(env_cfg, args_cli, logger)
        env_cfg.log_dir = log_dir

        env = create_isaaclab_env(
            args_cli.task,
            env_cfg,
            args_cli,
            convert_marl_to_single_agent=isinstance(env_cfg, DirectMARLEnvCfg) and algorithm in ["ppo"],
        )
        env = wrap_record_video(env, log_dir, args_cli)

        start_time = time.time()
        env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
        runner = Runner(env, agent_cfg)

        if resume_path:
            print(f"[INFO] Loading model checkpoint from: {resume_path}")
            runner.agent.load(resume_path)

        try:
            runner.run()
            print(f"Training time: {round(time.time() - start_time, 2)} seconds")

            total_timesteps = agent_cfg["trainer"]["timesteps"]
            os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
            runner.agent.write_checkpoint(timestep=total_timesteps, timesteps=total_timesteps)
            print(f"[INFO] Saved final agent checkpoint to: {log_dir}/checkpoints")
            env.close()
        except KeyboardInterrupt:
            pass
