# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Probe OpenArm joint sign conventions in Isaac Sim, one joint at a time.

Sets a single joint to a small explicit target value, bypassing the task's action
manager entirely, and holds it there so you can watch, in the viewport, which
physical direction a POSITIVE joint value corresponds to. Use this to fill in the
`sign` values in calibration.json (used by mirror_bridge.py / replay_sim_dataset.py /
safe_probe.py in ~/lerobot_openarm) by comparing against the real arm's actual
direction for the same nudge (via safe_probe.py --joint N --step <small value>).

This script never touches real hardware -- sim only.

Usage:
  ./isaaclab.sh -p scripts/tools/probe_joint_sign.py --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0
Then follow the interactive prompts: pick a joint number, watch which way it turns,
note it down, press Enter to reset it, move to the next joint.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Probe OpenArm joint sign conventions in Isaac Sim.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--delta", type=float, default=0.3, help="Radians to move for visualization (comfortably visible; sim-only, no hardware risk).")
parser.add_argument("--hold_time", type=float, default=2.0, help="Seconds to hold the nudged position before prompting.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import re

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

JOINT_NAME_PATTERNS = [
    r"openarm_left_joint[1-7]",
    r"openarm_right_joint[1-7]",
]


def drive_and_render(env, robot, indices, targets, duration_s: float, sim_dt: float):
    steps = max(1, int(duration_s / sim_dt))
    for _ in range(steps):
        if not simulation_app.is_running():
            break
        robot.set_joint_position_target(targets.unsqueeze(0), joint_ids=indices)
        env.scene.write_data_to_sim()
        env.sim.step()
        env.scene.update(sim_dt)
        env.sim.render()


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    robot = env.scene["robot"]
    env.sim.reset()
    env.reset()

    all_names = robot.data.joint_names
    pattern = re.compile("|".join(f"(?:{p})" for p in JOINT_NAME_PATTERNS))
    indices = [i for i, name in enumerate(all_names) if pattern.fullmatch(name)]
    names = [all_names[i] for i in indices]

    print("\n=== Joint Sign Probe ===")
    print("Available joints:")
    for k, name in enumerate(names):
        print(f"  {k}: {name}")
    print(f"\nEach selection sets that joint to +{args_cli.delta} rad and holds for {args_cli.hold_time}s.")
    print("Watch the viewport, note which physical direction it rotated, then press Enter to reset it to 0.")
    print("Type 'q' to quit.\n")

    sim_dt = env.sim.get_physics_dt()
    targets = torch.zeros(len(indices), device=env.device)

    with torch.inference_mode():
        while simulation_app.is_running():
            sel = input("Joint number to nudge (or 'q' to quit): ").strip()
            if sel.lower() == "q":
                break
            try:
                k = int(sel)
                name = names[k]
            except (ValueError, IndexError):
                print(f"Invalid selection -- enter a number from 0 to {len(names) - 1}.")
                continue

            print(f"Setting {name} to +{args_cli.delta} rad -- watch the viewport now...")
            targets[k] = args_cli.delta
            drive_and_render(env, robot, indices, targets, args_cli.hold_time, sim_dt)

            input(f"Which way did {name} rotate for a POSITIVE value? Note it down. Press Enter to reset it to 0...")
            targets[k] = 0.0
            drive_and_render(env, robot, indices, targets, 1.0, sim_dt)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
