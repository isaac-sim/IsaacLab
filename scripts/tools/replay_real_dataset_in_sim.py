# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Replay a REAL-recorded OpenArm episode (raw motor .pos values) inside Isaac Sim,
by inverse-mapping it through calibration.json back into sim joint angles.

This is the reverse direction of lerobot_openarm/replay_hf_sim_episode.py (which
drives real hardware from a sim-recorded episode) -- here a REAL-recorded episode
drives the SIM viewport instead. Purpose: isolate whether Phase 0 calibration
itself is correct from every hardware variable (motor gain, gripper response
speed, CAN timing) that could otherwise muddy the picture, by testing purely the
numeric sign/offset/gripper mapping in the opposite direction, in sim, where
"did it go to a sensible place" can be checked by eye with no hardware risk.

If the sim robot reproduces a sensible pick-up motion here, the calibration
mapping is validated end-to-end and any remaining "looks different when replayed
on real hardware from the OTHER (sim-recorded) dataset" is almost certainly that
dataset's own scene/cube layout not matching your physical setup -- not a
mapping bug.

This script never touches real hardware -- sim only, read-only trajectory
playback via direct joint-position targets (bypassing the task's IK action
manager entirely, same approach as probe_joint_sign.py).

Usage:
  1. Extract a real dataset episode's action trajectory to JSON first, in the
     lerobot_openarm venv (needs `lerobot` to read the HF dataset):
       python3 -c "
       import json
       from lerobot.datasets.lerobot_dataset import LeRobotDataset
       from lerobot.utils.constants import ACTION
       dataset = LeRobotDataset('ethanCSL/0422_stanley_red_cube', episodes=[0], revision='main')
       names = dataset.features[ACTION]['names']
       actions = dataset.select_columns(ACTION)
       traj = [{n: float(actions[i][ACTION][j]) for j, n in enumerate(names)} for i in range(dataset.num_frames)]
       json.dump(traj, open('real_episode_0.json', 'w'))
       "
  2. ./isaaclab.sh -p scripts/tools/replay_real_dataset_in_sim.py \\
         --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0 \\
         --trajectory real_episode_0.json --calibration calibration.json
"""

import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Replay a real-recorded episode's raw motor values inside Isaac Sim via inverse calibration mapping."
)
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument(
    "--trajectory", type=str, required=True,
    help="Path to a JSON list of {'LJ1.pos': rad, ...} frames extracted from a real dataset.",
)
parser.add_argument(
    "--calibration", type=str, required=True,
    help="Path to calibration.json (see lerobot_openarm/calibration.example.json).",
)
parser.add_argument("--playback-hz", type=float, default=30.0, help="Rate to step through the recorded trajectory.")
parser.add_argument("--max-steps", type=int, default=None, help="Only replay the first N frames -- use for a quick look.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import re
import time

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

ARM_JOINT_KEYS = [f"joint{i}" for i in range(1, 8)]
GRIPPER_SIM_OPEN = 0.044  # radians, matches BinaryJointPositionActionCfg open_command_expr in sim
GRIPPER_SIM_CLOSED = 0.0

ARM_JOINT_PATTERNS = [r"openarm_left_joint[1-7]", r"openarm_right_joint[1-7]"]
FINGER_JOINT_PATTERNS = [r"openarm_left_finger_joint.*", r"openarm_right_finger_joint.*"]


def load_calibration(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def real_raw_to_sim_arm(real_val: float, sign: int, offset: float) -> float:
    """Inverse of sign*sim_val + offset -- recovers the sim joint angle that
    produced a given real motor reading."""
    return (real_val - offset) / sign


def real_raw_to_sim_gripper(raw: float, open_raw: float, closed_raw: float) -> float:
    """Inverse of gripper_sim_to_raw()/gripper_cmd_to_raw() -- maps a real raw
    gripper reading back to the 0 (closed) .. GRIPPER_SIM_OPEN (open) sim
    finger-joint range."""
    frac = max(0.0, min(1.0, (raw - closed_raw) / (open_raw - closed_raw)))
    return GRIPPER_SIM_CLOSED + frac * (GRIPPER_SIM_OPEN - GRIPPER_SIM_CLOSED)


def resolve_joints(robot, patterns: list[str]) -> tuple[list[int], list[str]]:
    all_names = robot.data.joint_names
    pattern = re.compile("|".join(f"(?:{p})" for p in patterns))
    indices = [i for i, name in enumerate(all_names) if pattern.fullmatch(name)]
    names = [all_names[i] for i in indices]
    return indices, names


def frame_to_sim_targets(frame: dict, calib: dict, arm_names: list[str], finger_names: list[str]) -> dict:
    """Map one frame of real {"LJ1.pos": rad, ...} data to {sim_joint_name: target}."""
    targets = {}
    for side, prefix in (("left", "L"), ("right", "R")):
        sec = calib[side]
        for n, jkey in enumerate(ARM_JOINT_KEYS, start=1):
            sim_name = f"openarm_{side}_joint{n}"
            if sim_name in arm_names:
                sign = sec["sign"][jkey]
                offset = sec["offset_rad"][jkey]
                targets[sim_name] = real_raw_to_sim_arm(frame[f"{prefix}J{n}.pos"], sign, offset)
        grip = sec["gripper"]
        sim_grip_val = real_raw_to_sim_gripper(frame[f"{prefix}J8.pos"], grip["open_raw"], grip["closed_raw"])
        for fname in finger_names:
            if fname.startswith(f"openarm_{side}_finger"):
                targets[fname] = sim_grip_val
    return targets


def main():
    calib = load_calibration(args_cli.calibration)
    with open(args_cli.trajectory) as f:
        trajectory = json.load(f)
    if args_cli.max_steps is not None:
        trajectory = trajectory[: args_cli.max_steps]
    print(f"Loaded {len(trajectory)} frames from {args_cli.trajectory}")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    robot = env.scene["robot"]
    env.sim.reset()
    env.reset()

    arm_indices, arm_names = resolve_joints(robot, ARM_JOINT_PATTERNS)
    finger_indices, finger_names = resolve_joints(robot, FINGER_JOINT_PATTERNS)
    all_indices = arm_indices + finger_indices
    all_names = arm_names + finger_names
    print(f"Driving {len(all_names)} sim joints: {all_names}")

    sim_dt = env.sim.get_physics_dt()
    dt = 1.0 / args_cli.playback_hz
    steps_per_frame = max(1, round(dt / sim_dt))

    print("Playing back real-recorded episode in sim (inverse-calibration-mapped). Ctrl+C to stop.")
    with torch.inference_mode():
        for step_idx, frame in enumerate(trajectory):
            if not simulation_app.is_running():
                break
            loop_start = time.time()

            targets_by_name = frame_to_sim_targets(frame, calib, arm_names, finger_names)
            targets = torch.tensor([targets_by_name[n] for n in all_names], device=env.device)

            for _ in range(steps_per_frame):
                robot.set_joint_position_target(targets.unsqueeze(0), joint_ids=all_indices)
                env.scene.write_data_to_sim()
                env.sim.step()
                env.scene.update(sim_dt)
                env.sim.render()

            if step_idx % 20 == 0:
                print(f"  frame {step_idx}/{len(trajectory)}")

            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    print("Playback complete.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
