# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Headset-free Snap Circuits preview with deterministic GR1 hand motion."""

from __future__ import annotations

import argparse
import math
import time
import traceback

import warp as wp

wp.config.enable_backward = False

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="IsaacContrib-PickPlace-GR1T2-SnapCircuits-Abs")
parser.add_argument("--motion", action=argparse.BooleanOptionalAction, default=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(vars(args_cli), enable_cameras=True)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab_teleop.spectator_rtsp import EgocentricSpectatorRtsp  # noqa: E402

import omni.kit.app

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main() -> None:
    """Run the production scene with a safe, repeatable bimanual motion."""
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=True)
    env_cfg.episode_length_s = 3600.0
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    spectator = EgocentricSpectatorRtsp(env)
    # Ensure the timeline is live so OnPlaybackTick can create/feed the stream.
    env.sim.play()
    env.sim.forward()
    idle = torch.tensor(env_cfg.idle_action, dtype=torch.float32, device=env.device).unsqueeze(0)
    started_at = time.monotonic()

    print("[INFO] Desktop preview is running; scripted GR1 wrists and fingers should move in RTSP.")
    while simulation_app.is_running():
        action = idle.clone()
        if args_cli.motion:
            phase = 2.0 * math.pi * 0.18 * (time.monotonic() - started_at)
            sweep = 0.055 * math.sin(phase)
            lift = 0.035 * (1.0 - math.cos(phase))
            grip = 0.7 * (0.5 + 0.5 * math.sin(phase - math.pi / 2.0))

            # Absolute wrist positions; quaternions remain at the task's proven idle values.
            action[:, 0] += sweep
            action[:, 2] += lift
            action[:, 7] -= sweep
            action[:, 9] += lift
            # The remaining 22 values are the left/right GR1 finger targets.
            # Flexion is negative for the four fingers and proximal thumb yaw,
            # but positive for thumb pitch/distal joints.
            action[:, 14:24] = -grip
            action[:, 24:28] = -grip
            action[:, 28] = grip
            action[:, 29:33] = -grip
            action[:, 33:] = grip

        spectator.update()
        env.step(action)
        # Headless mode has no Kit visualizer to do this for us. Synchronize
        # PhysX into Fabric, then pump one Kit frame so the RTSP render product
        # consumes the exact transforms that the XR renderer will consume.
        env.sim.forward()
        play_flag = env.sim.get_setting("/app/player/playSimulations")
        env.sim.set_setting("/app/player/playSimulations", False)
        omni.kit.app.get_app().update()
        env.sim.set_setting("/app/player/playSimulations", bool(play_flag))

    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
