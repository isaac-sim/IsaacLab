# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TorchRL training logic for the unified reinforcement learning entrypoint."""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import random
import time
from datetime import datetime

from isaaclab.app import add_launcher_args

from isaaclab_rl.entrypoints.common import (
    add_common_train_args,
    apply_env_overrides,
    apply_video_recording,
    configure_io_descriptors,
    create_isaaclab_env,
    dump_train_configs,
    enable_cameras_for_video,
    pre_launch_video_config,
    set_hydra_args,
    show_run_summary,
    startup_screen,
    wrap_training_capture,
    write_run_manifest,
)

import isaaclab_tasks  # noqa: F401

logger = logging.getLogger(__name__)

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse TorchRL training arguments."""
    from isaaclab_tasks.utils import setup_preset_cli

    parser = argparse.ArgumentParser(description="Train an RL agent with TorchRL.")
    add_common_train_args(
        parser,
        agent_default="torchrl_cfg_entry_point",
        agent_help="Name of the RL agent configuration entry point.",
        include_distributed=False,
    )
    add_launcher_args(parser)
    args_cli, hydra_args = setup_preset_cli(parser, argv, agent_library="torchrl")
    enable_cameras_for_video(args_cli)
    set_hydra_args(hydra_args)
    return args_cli


def run(argv: list[str]) -> None:
    """Train a PPO agent with TorchRL."""
    from isaaclab.app import launch_simulation
    from isaaclab.envs import DirectMARLEnvCfg
    from isaaclab.utils.seed import configure_seed

    from isaaclab_tasks.utils import resolve_task_config

    args_cli = _parse_args(argv)
    with startup_screen(args_cli, num_stages=2) as screen:
        env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)
        pre_launch_video_config(env_cfg, args_cli=args_cli)
        show_run_summary(screen, args_cli, env_cfg, library="torchrl", action="train")
        screen.stage("Launching simulation")
        with launch_simulation(env_cfg, args_cli):
            # imported after the task config is resolved so preset errors surface even without torchrl installed
            from isaaclab_rl.torchrl import IsaacLabTorchRLWrapper, train_ppo

            apply_env_overrides(args_cli, env_cfg)
            if args_cli.seed is not None:
                agent_cfg.seed = args_cli.seed if args_cli.seed != -1 else random.randint(0, 10000)
            if args_cli.max_iterations is not None:
                agent_cfg.max_iterations = args_cli.max_iterations
            agent_cfg.device = env_cfg.sim.device
            env_cfg.seed = agent_cfg.seed
            # terminal observations let the value estimator bootstrap correctly on time-outs
            env_cfg.compute_final_obs = True

            run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + (
                f"_{agent_cfg.run_name}" if agent_cfg.run_name else ""
            )
            log_dir = os.path.abspath(os.path.join("logs", "torchrl", agent_cfg.experiment_name, run_name))
            print(f"[INFO] Logging experiment in directory: {log_dir}")
            write_run_manifest(log_dir, library="torchrl", task=args_cli.task, metadata={"agent": args_cli.agent})
            dump_train_configs(log_dir, env_cfg, agent_cfg)
            configure_io_descriptors(env_cfg, args_cli, logger)
            env_cfg.log_dir = log_dir
            apply_video_recording(env_cfg, log_dir, args_cli)

            screen.stage("Creating environment")
            env = create_isaaclab_env(
                args_cli.task,
                env_cfg,
                args_cli,
                convert_marl_to_single_agent=isinstance(env_cfg, DirectMARLEnvCfg),
            )
            env = IsaacLabTorchRLWrapper(
                wrap_training_capture(env, log_dir, args_cli), clip_actions=agent_cfg.clip_actions
            )
            if args_cli.deterministic:
                configure_seed(env_cfg.seed, torch_deterministic=True)
            screen.close()

            start_time = time.time()
            with contextlib.suppress(KeyboardInterrupt):
                train_ppo(env, agent_cfg, log_dir)
            print(f"Training time: {round(time.time() - start_time, 2)} seconds")
            env.close()
