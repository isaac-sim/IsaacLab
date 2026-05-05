# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CI/automation entry point for replaying captured teleop sessions.

This is the non-interactive counterpart to ``teleop_se3_agent.py``. It builds
a teleop environment, attaches a teleop device, schedules a replay driver,
and pumps the simulation loop until the replay completes and the application
exits. The user-journey teleop script remains ``teleop_se3_agent.py``.

The current implementation drives playback through Kit's OpenXR XCR backend
and the legacy native XR ``handtracking`` device. The script is structured so
that the replay-driver call site and device selection are the only pieces
that need to change when migrating to a different replay backend in the
future (e.g. an Isaac Teleop ``TeleopSession`` running in replay mode).
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description=(
        "Replay a captured teleop session against an Isaac Lab environment. "
        "CI/automation entry point; for interactive teleoperation see teleop_se3_agent.py."
    )
)
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--replay_file",
    type=str,
    required=True,
    help="Absolute path to the recorded teleop session to replay.",
)
parser.add_argument(
    "--replay_start_delay_s",
    type=float,
    default=120.0,
    help="Seconds to wait after the environment is up before starting replay (default: 120.0).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""


import asyncio
import logging

import gymnasium as gym
import torch

from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs import ManagerBasedRLEnvCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

logger = logging.getLogger(__name__)

_LEGACY_DEVICE_NAME = "handtracking"


def _prepare_env_cfg(task: str, num_envs: int, device: str) -> ManagerBasedRLEnvCfg:
    """Build and tweak an env config suitable for non-interactive replay."""
    env_cfg = parse_env_cfg(task, device=device, num_envs=num_envs)
    env_cfg.env_name = task
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise ValueError(
            "teleop_replay_agent only supports ManagerBasedRLEnv environments. "
            f"Received environment config type: {type(env_cfg).__name__}"
        )
    env_cfg.terminations.time_out = None
    env_cfg = remove_camera_configs(env_cfg)
    env_cfg.sim.render.antialiasing_mode = "DLSS"
    return env_cfg


def _create_replay_teleop_device(env_cfg: ManagerBasedRLEnvCfg, task: str) -> object:
    """Instantiate the teleop device used during replay.

    Today this returns the legacy native XR ``handtracking`` device because the
    XCR backend replays through Kit's OpenXR runtime, which is the surface
    that device consumes. When migrating to a ``TeleopSession``-driven replay
    backend, swap this for an ``IsaacTeleopDevice`` configured in replay mode.
    """
    if not hasattr(env_cfg, "teleop_devices") or _LEGACY_DEVICE_NAME not in env_cfg.teleop_devices.devices:
        raise ValueError(
            f"Task '{task}' does not expose a teleop device named '{_LEGACY_DEVICE_NAME}'. "
            "Use a task whose env config defines that legacy device, "
            "or update _create_replay_teleop_device to use a different backend."
        )
    teleop_interface = create_teleop_device(_LEGACY_DEVICE_NAME, env_cfg.teleop_devices.devices, callbacks={})
    if teleop_interface is None:
        raise RuntimeError(f"Failed to create '{_LEGACY_DEVICE_NAME}' teleop device for task '{task}'.")
    return teleop_interface


def _schedule_replay_driver(replay_file: str, start_delay_s: float) -> None:
    """Schedule the replay driver coroutine on the running asyncio loop.

    Today this drives Kit's OpenXR XCR backend. To migrate to a different
    replay backend (e.g. ``TeleopSession`` running in replay mode), replace
    this call with the equivalent driver hook -- this is the only XCR-specific
    site outside the device-creation helper above.
    """
    from isaaclab_teleop.automation import XcrReplayConfig, start_xcr_replay

    asyncio.ensure_future(start_xcr_replay(XcrReplayConfig(replay_file=replay_file, start_delay_s=start_delay_s)))


def main() -> None:
    """Replay a captured teleop session against an Isaac Lab environment.

    Builds the env, attaches a replay teleop device, schedules the replay
    driver as a background task, and runs the standard teleop step loop
    until the application is closed (driver-issued ``post_quit``, Kit
    shutdown, or operator interrupt).
    """
    env_cfg = _prepare_env_cfg(args_cli.task, args_cli.num_envs, args_cli.device)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    teleop_interface = _create_replay_teleop_device(env_cfg, args_cli.task)
    print(f"Using teleop device: {teleop_interface}")

    env.reset()
    teleop_interface.reset()

    print(f"Replay agent started; replay will begin in {args_cli.replay_start_delay_s:.1f} seconds.")
    _schedule_replay_driver(args_cli.replay_file, args_cli.replay_start_delay_s)

    while simulation_app.is_running():
        try:
            with torch.inference_mode():
                action = teleop_interface.advance()
                if action is None:
                    env.sim.render()
                    continue
                actions = action.repeat(env.num_envs, 1)
                env.step(actions)
        except Exception as e:
            logger.error(f"Error during simulation step: {e}")
            break

    env.close()
    print("Environment closed")


if __name__ == "__main__":
    main()
    simulation_app.update()
    simulation_app.close()
