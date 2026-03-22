# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deploy a LEAPP-exported policy in an Isaac Lab simulation.

Usage::

    ./isaaclab.sh -p scripts/reinforcement_learning/deploy.py \
        --task Isaac-Velocity-Flat-Anymal-B-v0 \
        --leapp_model .pretrained_checkpoints/rsl_rl/Isaac-Velocity-Flat-Anymal-B-v0/Isaac-Velocity-Flat-Anymal-B-v0/Isaac-Velocity-Flat-Anymal-B-v0.yaml \
        --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Deploy a LEAPP-exported policy in simulation.")
parser.add_argument("--task", type=str, required=True, help="Name of the registered Isaac Lab task.")
parser.add_argument("--leapp_model", type=str, required=True, help="Path to the LEAPP .yaml pipeline description.")
parser.add_argument("--seed", type=int, default=None, help="Seed for the environment.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from isaaclab.envs.direct_deployment_env import DirectDeploymentEnv

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


def main():
    # ── Load env config from gym registry ─────────────────────────
    task_name = args_cli.task.split(":")[-1]
    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")

    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # ── Create deploy env ─────────────────────────────────────────
    env = DirectDeploymentEnv(env_cfg, args_cli.leapp_model)

    print(f"[INFO]: Deploying task '{task_name}' with LEAPP model: {args_cli.leapp_model}")
    print(f"[INFO]: Num envs: {env.num_envs}, decimation: {env.cfg.decimation}, step_dt: {env.step_dt:.4f}s")

    # ── Run loop ──────────────────────────────────────────────────
    env.reset()
    with torch.inference_mode():
        while simulation_app.is_running():
            env.step()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
