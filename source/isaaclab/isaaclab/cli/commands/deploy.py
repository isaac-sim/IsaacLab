# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Commands for deploying exported policies."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def command_deploy_leapp(argv: list[str] | None = None) -> int:
    """Deploy a LEAPP pipeline in an Isaac Lab simulation.

    Args:
        argv: Command-line arguments excluding the executable and ``deploy_leapp`` tokens.

    Returns:
        Process exit code.
    """
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(
        description="Deploy a LEAPP-exported policy in simulation.",
        prog=f"{Path(sys.argv[0]).name} deploy_leapp",
    )
    parser.add_argument("--task", required=True, help="Name of the registered Isaac Lab task.")
    parser.add_argument("--pipeline", required=True, help="Path to the exported LEAPP YAML pipeline description.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for the environment.")
    AppLauncher.add_app_launcher_args(parser)

    if argv is None:
        argv = sys.argv[1:]
    args_cli, hydra_args = parser.parse_known_args(AppLauncher._fuse_kit_args(argv))

    original_argv = sys.argv
    sys.argv = [original_argv[0]] + hydra_args
    simulation_app = None
    env = None
    try:
        simulation_app = AppLauncher(args_cli).app

        import torch

        from isaaclab.envs import LeappDeploymentEnv

        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils.hydra import resolve_task_config

        task_name = args_cli.task.split(":")[-1]
        env_cfg, _ = resolve_task_config(task_name, "")

        if args_cli.seed is not None:
            env_cfg.seed = args_cli.seed
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device

        env = LeappDeploymentEnv(env_cfg, args_cli.pipeline)

        if getattr(args_cli, "headless", False):
            print(
                "[WARN]: Running deploy without a viewport. This happens when headless mode is active, "
                "including the default case where no visualizer was selected. The policy may be "
                "stepping normally, but no viewport will appear unless you specify the "
                "`--visualizer` field."
            )

        print(f"[INFO]: Deploying task '{task_name}' with LEAPP pipeline: {args_cli.pipeline}")
        print(f"[INFO]: Num envs: {env.num_envs}, decimation: {env.cfg.decimation}, step_dt: {env.step_dt:.4f}s")

        env.reset()
        with torch.inference_mode():
            while simulation_app.is_running():
                env.step()
    except KeyboardInterrupt:
        return 0
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            if simulation_app is not None:
                simulation_app.close()
            sys.argv = original_argv

    return 0
