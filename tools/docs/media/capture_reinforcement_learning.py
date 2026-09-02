# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capture helpers for OVRTX-rendered reinforcement-learning documentation media."""

from __future__ import annotations

import dataclasses
import os
import sys

import gymnasium as gym
from isaaclab_visualizers.newton import NewtonRTXVisualizerCfg

from isaaclab.envs import VideoRecorderCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.core.velocity.config.anymal_d.flat_env_cfg import AnymalDFlatEnvCfg


@configclass
class AnymalDFlatCaptureCfg(AnymalDFlatEnvCfg):
    """Anymal-D flat-terrain configuration for a fixed OVRTX progression recording."""

    def __post_init__(self):
        super().__post_init__()

        command_cfg = self.commands.base_velocity
        command_cfg.heading_command = False
        command_cfg.rel_standing_envs = 0.0
        command_cfg.rel_heading_envs = 0.0
        command_cfg.debug_vis = False
        command_cfg.ranges.lin_vel_x = (0.0, 0.0)
        command_cfg.ranges.lin_vel_y = (0.0, 0.0)
        command_cfg.ranges.ang_vel_z = (0.6, 0.6)
        command_cfg.ranges.heading = (0.0, 0.0)

        _configure_capture(self)


def configure_playback() -> list[str]:
    """Register the fixed Anymal-D capture config and preserve Hydra arguments."""
    task = _argument_value("--task")
    if task != "Isaac-Velocity-Flat-AnymalD":
        raise ValueError(f"No reinforcement-learning capture configuration is registered for task {task!r}.")
    gym.spec(task).kwargs["env_cfg_entry_point"] = "capture_reinforcement_learning:AnymalDFlatCaptureCfg"
    return sys.argv[1:]


def _configure_capture(env_cfg: object):
    """Configure every simulation preset before Hydra selects the physics backend."""
    sim_presets = getattr(env_cfg, "sim")
    if isinstance(sim_presets, SimulationCfg):
        _configure_resolved_sim(sim_presets)
    else:
        for field in dataclasses.fields(sim_presets):
            sim_cfg = getattr(sim_presets, field.name)
            if isinstance(sim_cfg, SimulationCfg):
                _configure_resolved_sim(sim_cfg)

    output_dir = os.environ.get("RL_PROGRESS_VIDEO_DIR")
    if output_dir:
        env_cfg.video_recorders = [
            VideoRecorderCfg(
                source="visualizer:newton_rtx",
                output_dir=output_dir,
                output_filename_prefix=os.environ.get("RL_PROGRESS_VIDEO_PREFIX", "progress"),
                fps=50,
            )
        ]


def _configure_resolved_sim(sim_cfg: SimulationCfg):
    """Attach a headless 480 by 270 Newton RTX/OVRTX visualizer."""
    sim_cfg.visualizer_cfgs = [
        NewtonRTXVisualizerCfg(
            eye=(2.3, 2.0, 1.5),
            lookat=(0.0, 0.0, 0.45),
            focal_length=40.0,
            window_width=480,
            window_height=270,
            headless=True,
            rtx_environment="studio",
        )
    ]


def _argument_value(name: str) -> str | None:
    """Return a command-line option value from either supported argparse form."""
    for index, argument in enumerate(sys.argv[1:]):
        if argument == name and index + 2 <= len(sys.argv) - 1:
            return sys.argv[index + 2]
        if argument.startswith(f"{name}="):
            return argument.split("=", 1)[1]
    return None
