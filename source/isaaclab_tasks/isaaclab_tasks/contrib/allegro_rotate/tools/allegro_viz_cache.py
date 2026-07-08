# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize Allegro rotate cache resets without an RL policy.

This is a cache/debug viewer: it loads the Allegro rotate environment from a
saved grasp cache and repeatedly applies zero actions, so the hand holds the
cached joint target instead of being driven by an untrained policy.
"""

from __future__ import annotations

import argparse
import sys
import time

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config


parser = argparse.ArgumentParser(description="Visualize Allegro grasp-cache resets with zero actions.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to visualize.")
parser.add_argument("--task", type=str, default="Isaac-Inhand-Rotate-Allegro-v0", help="Task name.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment.")
parser.add_argument("--cache", type=str, required=True, help="Path to Allegro grasp cache .npy file.")
parser.add_argument(
    "--object_reset_z_offset",
    type=float,
    default=None,
    help="Override Allegro object reset Z offset in meters. Defaults to the task cfg value.",
)
parser.add_argument(
    "--object_reset_pos_offset",
    type=float,
    nargs=3,
    metavar=("DX", "DY", "DZ"),
    default=None,
    help="Override Allegro cache object reset XYZ offset in meters.",
)
parser.add_argument("--steps", type=int, default=2400, help="Maximum number of zero-action simulation steps.")
parser.add_argument("--print_interval", type=int, default=60, help="Print status every N steps.")
parser.add_argument("--reset_interval", type=int, default=0, help="Force reset every N steps; 0 disables forced resets.")
parser.add_argument("--real-time", action="store_true", default=False, help="Throttle stepping to env.step_dt.")
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args


def _metric(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().mean().cpu().item())
    return float(value)


def _print_status(env_u, step: int) -> None:
    log = env_u.extras.get("log", {})
    fields = []
    for key in (
        "rotate/mean_fingertip_dist",
        "rotate/top2_contact_force",
        "rotate/contact_count",
        "rotate/thumb_force",
        "rotate/object_pos_diff",
        "rotate/drop_rate",
    ):
        if key in log:
            fields.append(f"{key.split('/')[-1]}={_metric(log[key]):.4f}")
    object_pos = getattr(env_u, "object_pos", None)
    if object_pos is not None:
        mean_pos = object_pos.detach().mean(dim=0).cpu().tolist()
        fields.append(f"object=({mean_pos[0]:+.4f},{mean_pos[1]:+.4f},{mean_pos[2]:+.4f})")
    print(f"[viz_cache] step={step} " + ", ".join(fields))


def main() -> None:
    torch.manual_seed(args_cli.seed)

    env_cfg, _ = resolve_task_config(args_cli.task, "")
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        env_cfg.seed = args_cli.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env_cfg.grasp_cache_path = args_cli.cache
        env_cfg.require_grasp_cache = True
        if args_cli.object_reset_z_offset is not None:
            env_cfg.cache_object_reset_z_offset = args_cli.object_reset_z_offset
            print(f"[INFO] Overriding cache object reset Z offset: {env_cfg.cache_object_reset_z_offset:+.4f} m")
        if args_cli.object_reset_pos_offset is not None:
            env_cfg.cache_object_reset_pos_offset = tuple(args_cli.object_reset_pos_offset)
            print(f"[INFO] Overriding cache object reset XYZ offset: {env_cfg.cache_object_reset_pos_offset}")
        if args_cli.disable_fabric:
            env_cfg.sim.use_fabric = False

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        env_u = env.unwrapped
        try:
            env.reset()
            for step in range(args_cli.steps):
                start_time = time.time()
                if args_cli.reset_interval > 0 and step > 0 and step % args_cli.reset_interval == 0:
                    env.reset()
                with torch.no_grad():
                    obs, rew, terminated, truncated, info = env.step(env_u.zero_actions())
                if step % args_cli.print_interval == 0:
                    _print_status(env_u, step)
                sleep_time = env_u.step_dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0.0:
                    time.sleep(sleep_time)
        finally:
            env.close()


if __name__ == "__main__":
    main()
