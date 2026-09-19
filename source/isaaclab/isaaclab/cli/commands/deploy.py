# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Commands for deploying exported policies."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from isaaclab.app import AppLauncher, launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import resolve_task_config


def command_deploy_leapp(argv: list[str] | None = None) -> int:
    """Deploy a LEAPP pipeline in an Isaac Lab simulation.

    Args:
        argv: Command-line arguments excluding the executable and ``leapp deploy`` tokens.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Deploy a LEAPP-exported policy in simulation.",
        prog=f"{Path(sys.argv[0]).name} leapp deploy",
    )
    parser.add_argument("--task", required=True, help="Name of the registered Isaac Lab task.")
    parser.add_argument("--pipeline", required=True, help="Path to the exported LEAPP YAML pipeline description.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for the environment.")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Maximum policy steps to run, or -1 to continue until the visualizer closes.",
    )
    AppLauncher.add_app_launcher_args(parser)

    if argv is None:
        argv = sys.argv[1:]
    args_cli, hydra_args = parser.parse_known_args(AppLauncher._fuse_kit_args(argv))

    original_argv = sys.argv
    sys.argv = [original_argv[0]] + hydra_args
    env = None
    try:
        task_name = args_cli.task.split(":")[-1]
        env_cfg, _ = resolve_task_config(task_name, "", play_mode=True)

        if args_cli.seed is not None:
            env_cfg.seed = args_cli.seed
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device

        with launch_simulation(env_cfg, args_cli):
            try:
                from isaaclab.envs import LeappDeploymentEnv

                env = LeappDeploymentEnv(
                    env_cfg,
                    args_cli.pipeline,
                    controller_owned_write_handlers=LeappDeploymentEnv.simulated_controller_owned_write_handlers(),
                )

                if getattr(args_cli, "headless", False):
                    print(
                        "[WARN]: Running deploy without a viewport. This happens when headless mode is active, "
                        "including the default case where no visualizer was selected. The policy may be "
                        "stepping normally, but no viewport will appear unless you specify the "
                        "`--visualizer` field."
                    )

                print(f"[INFO]: Deploying task '{task_name}' with LEAPP pipeline: {args_cli.pipeline}")
                print(
                    f"[INFO]: Num envs: {env.num_envs}, decimation: {env.cfg.decimation}, step_dt: {env.step_dt:.4f}s"
                )

                env.reset()
                step_count = 0
                with torch.inference_mode():
                    while env.sim.is_headless_or_exist_active_visualizer() and (
                        args_cli.max_steps < 0 or step_count < args_cli.max_steps
                    ):
                        env.step()
                        step_count += 1
            finally:
                if env is not None:
                    env.close()
                    env = None
    except KeyboardInterrupt:
        return 0
    finally:
        sys.argv = original_argv

    return 0
