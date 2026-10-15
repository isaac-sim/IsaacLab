# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capture helpers for OVRTX-rendered quickstart media."""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

import gymnasium as gym
import torch
from isaaclab_visualizers.newton import NewtonRTXVisualizerCfg

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, VideoRecorderCfg
from isaaclab.managers import EventTermCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import WindowCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.core.cabinet.config.franka.joint_pos_env_cfg import FrankaCabinetEnvCfg
from isaaclab_tasks.core.cartpole.cartpole_manager_env_cfg import CartpoleEnvCfg
from isaaclab_tasks.core.lift.config.kuka_allegro.kuka_allegro_env_cfg import KukaAllegroLiftEnvCfg
from isaaclab_tasks.core.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

_TASK_CONFIGS = {
    "Isaac-Cartpole": "capture_quickstart:CartpoleCaptureCfg",
    "Isaac-Lift-KukaAllegro": "capture_quickstart:KukaAllegroCaptureCfg",
    "Isaac-Open-Drawer-Franka": "capture_quickstart:FrankaCabinetCaptureCfg",
    "Isaac-Velocity-Rough-G1": "capture_quickstart:G1RoughCaptureCfg",
}

_FRANKA_EYE = (-1.6, 1.7, 1.6)
_FRANKA_LOOKAT = (0.4, 0.0, 0.55)
_G1_EYE_OFFSET = (2.1, -2.1, 1.1)
_G1_TARGET_HEIGHT = 0.7


@configclass
class CartpoleCaptureCfg(CartpoleEnvCfg):
    """Cartpole configuration with a compact OVRTX recording viewport."""

    def __post_init__(self):
        super().__post_init__()
        _configure_capture(self, eye=(4.5, -0.4, 4.5), lookat=(0.0, 0.0, 2.3), focal_length=32.0)


@configclass
class G1RoughCaptureCfg(G1RoughEnvCfg):
    """G1 rough-terrain configuration with a following OVRTX recording viewport."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.debug_vis = False
        _configure_capture(
            self,
            eye=(_G1_EYE_OFFSET[0], _G1_EYE_OFFSET[1], _G1_EYE_OFFSET[2] + _G1_TARGET_HEIGHT),
            lookat=(0.0, 0.0, _G1_TARGET_HEIGHT),
            focal_length=24.0,
        )
        self.events.follow_camera = EventTermCfg(
            func=_follow_g1_camera, mode="interval", interval_range_s=(0.0, 0.0), is_global_time=True
        )
        self.events.reset_camera = EventTermCfg(func=_follow_g1_camera, mode="reset")


@configclass
class KukaAllegroCaptureCfg(KukaAllegroLiftEnvCfg):
    """Kuka Allegro lift configuration with a compact OVRTX recording viewport."""

    def __post_init__(self):
        super().__post_init__()
        _configure_capture(self, eye=(-2.3, -0.8, 1.2), lookat=(0.0, 0.0, 0.4), focal_length=30.0)


@configclass
class FrankaCabinetCaptureCfg(FrankaCabinetEnvCfg):
    """Franka cabinet configuration with a compact OVRTX recording viewport."""

    def __post_init__(self):
        super().__post_init__()
        _configure_capture(self, eye=_FRANKA_EYE, lookat=_FRANKA_LOOKAT, focal_length=24.0)


def configure_playback() -> list[str]:
    """Register the task-specific capture config and preserve Hydra arguments."""
    task = _argument_value("--task")
    if task not in _TASK_CONFIGS:
        raise ValueError(f"No quickstart capture configuration is registered for task {task!r}.")
    gym.spec(task).kwargs["env_cfg_entry_point"] = _TASK_CONFIGS[task]
    return sys.argv[1:]


def main():
    """Record a zero- or random-agent clip using the quickstart capture settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("zero", "random"), required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--video-length", type=int, default=360)
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["newton_rtx"])
    args, hydra_args = setup_preset_cli(parser)
    sys.argv = [sys.argv[0], *hydra_args]

    env_cfg, _ = resolve_task_config(args.task, "", play_mode=True)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = 42
    _configure_resolved_sim(env_cfg.sim, eye=_FRANKA_EYE, lookat=_FRANKA_LOOKAT, focal_length=24.0)
    env_cfg.video_recorders = [
        VideoRecorderCfg(
            source="viz:newton_rtx",
            output_dir=args.output_dir,
            output_filename_prefix=args.policy,
            video_length=args.video_length,
            fps=60,
        )
    ]

    torch.manual_seed(42)
    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg)
        env.reset(seed=42)
        zero_actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        for _ in range(args.video_length):
            with torch.inference_mode():
                actions = zero_actions if args.policy == "zero" else 2 * torch.rand_like(zero_actions) - 1
                env.step(actions)
        env.close()


def _configure_capture(
    env_cfg: ManagerBasedRLEnvCfg,
    eye: tuple[float, float, float],
    lookat: tuple[float, float, float],
    focal_length: float,
):
    """Configure every simulation preset before Hydra resolves the selected backend."""
    sim_presets = getattr(env_cfg, "sim")
    if isinstance(sim_presets, SimulationCfg):
        _configure_resolved_sim(sim_presets, eye, lookat, focal_length)
    else:
        for field in dataclasses.fields(sim_presets):
            sim_cfg = getattr(sim_presets, field.name)
            if isinstance(sim_cfg, SimulationCfg):
                _configure_resolved_sim(sim_cfg, eye, lookat, focal_length)

    output_dir = os.environ.get("QUICKSTART_VIDEO_DIR")
    if output_dir:
        env_cfg.video_recorders = [
            VideoRecorderCfg(
                source="viz:newton_rtx",
                output_dir=output_dir,
                output_filename_prefix=os.environ.get("QUICKSTART_VIDEO_PREFIX", "clip"),
                fps=60,
            )
        ]


def _configure_resolved_sim(
    sim_cfg: SimulationCfg,
    eye: tuple[float, float, float],
    lookat: tuple[float, float, float],
    focal_length: float,
):
    """Attach a headless 1280 by 720 OVRTX visualizer with task-specific framing."""
    sim_cfg.visualizer_cfgs = [
        NewtonRTXVisualizerCfg(
            eye=eye,
            lookat=lookat,
            focal_length=focal_length,
            window=WindowCfg(size=(1280, 720)),
            headless=True,
            render_settings={"omni:rtx:quality": ("Int", 100)},
        )
    ]


def _follow_g1_camera(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None) -> None:
    """Follow G1 while tracking its height on uneven terrain."""
    position = env.scene["robot"].data.root_pos_w[0]
    target = (float(position[0]), float(position[1]), float(position[2]))
    eye = tuple(target[axis] + _G1_EYE_OFFSET[axis] for axis in range(3))
    env.sim.set_camera_view(eye, target)


def _argument_value(name: str) -> str | None:
    """Return a command-line option value from either supported argparse form."""
    for index, argument in enumerate(sys.argv[1:]):
        if argument == name and index + 2 <= len(sys.argv) - 1:
            return sys.argv[index + 2]
        if argument.startswith(f"{name}="):
            return argument.split("=", 1)[1]
    return None


if __name__ == "__main__":
    main()
