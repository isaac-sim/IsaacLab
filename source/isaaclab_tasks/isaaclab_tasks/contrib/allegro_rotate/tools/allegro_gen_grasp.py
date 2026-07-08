# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate Allegro grasp cache with a two-stage cache/train flow."""

from __future__ import annotations

import argparse
import sys
import time

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config


parser = argparse.ArgumentParser(description="Generate Allegro in-hand rotate grasp cache.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=16384, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Inhand-Rotate-Grasp-Allegro-v0", help="Task name.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment.")
parser.add_argument("--max_cache_rows", type=int, default=None, help="Stop after saving this many grasp states.")
parser.add_argument("--output", type=str, default=None, help="Output cache path or scale-suffixed prefix.")
parser.add_argument("--probe_init_pose", action="store_true", default=False, help="Hold the init pose for GUI tuning.")
parser.add_argument(
    "--probe_drop_hold_steps",
    type=int,
    default=None,
    help="Probe mode only: keep a failed/fallen grasp visible for this many env steps before reset.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Throttle stepping to env.step_dt.")
parser.add_argument("--max_steps", type=int, default=None, help="Maximum env steps before exit; default runs until cache is saved.")
parser.add_argument("--print_interval", type=int, default=None, help="Override grasp status print interval in env steps.")
parser.add_argument("--reset_dof_pos_noise", type=float, default=None, help="Override reset joint-position noise.")
parser.add_argument("--reset_position_noise", type=float, default=None, help="Override reset object-position noise.")
parser.add_argument(
    "--object_reset_z_offset",
    type=float,
    default=None,
    help="Override object reset Z offset in meters before generating cache rows.",
)
parser.add_argument(
    "--object_reset_pos_offset",
    type=float,
    nargs=3,
    metavar=("DX", "DY", "DZ"),
    default=None,
    help="Override object reset XYZ offset in meters before generating cache rows.",
)
parser.add_argument(
    "--object_center_offset",
    type=float,
    nargs=3,
    metavar=("DX", "DY", "DZ"),
    default=None,
    help="Offset added to fixed pinch-center object placement.",
)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args


def main() -> None:
    torch.manual_seed(args_cli.seed)

    env_cfg, _ = resolve_task_config(args_cli.task, "")
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        env_cfg.seed = args_cli.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        if args_cli.max_cache_rows is not None:
            env_cfg.grasp_cache_target = args_cli.max_cache_rows
        if args_cli.output is not None:
            env_cfg.grasp_output_path = args_cli.output
        if args_cli.probe_init_pose:
            env_cfg.grasp_probe_init_pose = True
            env_cfg.reset_dof_pos_noise = 0.0
            env_cfg.reset_position_noise = 0.0
            env_cfg.grasp_cache_status_interval = 30
        if args_cli.probe_drop_hold_steps is not None:
            env_cfg.grasp_probe_drop_hold_steps = args_cli.probe_drop_hold_steps
        if args_cli.print_interval is not None:
            env_cfg.grasp_cache_status_interval = args_cli.print_interval
        if args_cli.reset_dof_pos_noise is not None:
            env_cfg.reset_dof_pos_noise = args_cli.reset_dof_pos_noise
        if args_cli.reset_position_noise is not None:
            env_cfg.reset_position_noise = args_cli.reset_position_noise
        if args_cli.object_reset_z_offset is not None:
            env_cfg.object_reset_z_offset = args_cli.object_reset_z_offset
            print(f"[INFO] Overriding object reset Z offset: {env_cfg.object_reset_z_offset:+.4f} m")
        if args_cli.object_reset_pos_offset is not None:
            env_cfg.object_reset_pos_offset = tuple(args_cli.object_reset_pos_offset)
            print(f"[INFO] Overriding object reset XYZ offset: {env_cfg.object_reset_pos_offset}")
        if args_cli.object_center_offset is not None:
            env_cfg.object_pinch_center_offset = tuple(args_cli.object_center_offset)
            env_cfg.object_fingertip_center_offset = tuple(args_cli.object_center_offset)
        if args_cli.disable_fabric:
            env_cfg.sim.use_fabric = False

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        try:
            env.reset()
            step = 0
            while args_cli.max_steps is None or step < args_cli.max_steps:
                start_time = time.time()
                with torch.no_grad():
                    actions = env.unwrapped.zero_actions()
                    env.step(actions)
                step += 1
                sleep_time = env.unwrapped.step_dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0.0:
                    time.sleep(sleep_time)
        finally:
            env.close()


if __name__ == "__main__":
    main()
