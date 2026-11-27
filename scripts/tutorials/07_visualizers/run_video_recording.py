# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tutorial: recording video from visualizers and scene sensors.

This script demonstrates three progressively richer recording configurations,
all using the Shadow Hand cube-reorientation task
(``Isaac-Reorient-Cube-Shadow-Camera-Direct``).

Example 1 — Newton viewport (simplest)
    One clip from the Newton interactive viewport, showing 4 parallel environments.

    .. code-block:: bash

        uv run python scripts/tutorials/07_visualizers/run_video_recording.py \
            --example 1 --num_envs 4 --viz newton_rtx

Example 2 — scene sensor only, headless
    One clip captured directly from the scene's tiled-camera sensor.
    No visualizer window opens; the sensor renders offline.

    .. code-block:: bash

        uv run python scripts/tutorials/07_visualizers/run_video_recording.py \
            --example 2 --num_envs 16

Example 3 — Newton RTX viewport + tiled grid + scene sensor
    Three independent clip streams recorded simultaneously.

    .. code-block:: bash

        uv run python scripts/tutorials/07_visualizers/run_video_recording.py \
            --example 3 --num_envs 4 --viz newton_rtx

Clips are written to ``videos/recording_tutorial/example_<N>/`` in the working directory.
Examples 1 and 2 each demonstrate one recording source; Example 3 combines all of them.
``--viz`` selects the visualizers to record from; the configs below only set their camera and panels.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys

import gymnasium as gym
import torch

from isaaclab.visualizers import ImageViewCfg, VisualizerCfg, WindowCfg

import isaaclab_tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VIDEO_LENGTH = 100  # env steps per clip
_NUM_STEPS = 115  # slightly more than _VIDEO_LENGTH so the clip flushes cleanly

_TASK_SHADOW = "Isaac-Reorient-Cube-Shadow-Camera-Direct"

# Viewport camera: positioned to show a 2×2 grid of Shadow Hand environments.
# env_spacing=1.0 with 4 envs → envs centered at ±0.5 in x and y.
# Cube spawns at ~(0, -0.39, 0.6) per env; wrist cylinder is the landmark at the top.
_SHADOW_EYE = (0.0, -2.2, 1.8)
_SHADOW_LOOKAT = (0.0, -0.1, 0.4)
_SHADOW_ENV_SPACING = 1.0

# Skip the first few steps so the scene has settled before recording starts.
_RENDER_STEP_OFFSET = 5


def _output_dir(example: int) -> str:
    return os.path.join("videos", "recording_tutorial", f"example_{example}")


def _shadow_env_cfg(num_envs: int, env_spacing: float = _SHADOW_ENV_SPACING):
    """Build a base Shadow Hand camera env cfg shared by all examples."""
    overrides = list(sys.argv[1:])
    if not any(arg.startswith("renderer=") for arg in overrides):
        overrides.append("renderer=rtx")
    env_cfg, _ = resolve_task_config(_TASK_SHADOW, "", overrides=(*overrides, "env.scene.tiled_camera=rgb"))
    env_cfg.scene.tiled_camera.height = 256
    env_cfg.scene.tiled_camera.width = 256
    env_cfg.scene.num_envs = num_envs
    env_cfg.scene.env_spacing = env_spacing
    return env_cfg


# ---------------------------------------------------------------------------
# Per-example environment config builders
# ---------------------------------------------------------------------------


def _build_env_cfg_example_1(num_envs: int, legacy_kit: bool = False):
    """Shadow Hand + Newton viewport: one clip from the interactive viewport."""
    from isaaclab_visualizers.kit import KitBackendCfg
    from isaaclab_visualizers.newton import NewtonGLBackendCfg

    env_cfg = _shadow_env_cfg(num_envs)
    env_cfg.sim.visualizer_cfgs = [
        VisualizerCfg(
            eye=_SHADOW_EYE,
            lookat=_SHADOW_LOOKAT,
            backend=KitBackendCfg() if legacy_kit else NewtonGLBackendCfg(initial_renderer="rtx"),
        )
    ]

    out = _output_dir(1)
    env_cfg.video_recorders = [
        VideoRecorderCfg(
            source="viz:kit" if legacy_kit else "viz:newton",
            output_dir=out,
            output_filename_prefix="kit_viewport" if legacy_kit else "newton_viewport",
            video_length=_VIDEO_LENGTH,
            fps=30,
            step_offset=_RENDER_STEP_OFFSET,
        ),
    ]
    return env_cfg, _TASK_SHADOW


def _build_env_cfg_example_2(num_envs: int):
    """Shadow Hand + headless: scene tiled-camera sensor clip only."""
    env_cfg = _shadow_env_cfg(num_envs, env_spacing=2.0)

    out = _output_dir(2)
    env_cfg.video_recorders = [
        VideoRecorderCfg(
            source="sensor:tiled_camera",
            output_dir=out,
            output_filename_prefix="sensor",
            video_length=_VIDEO_LENGTH,
            fps=30,
        ),
    ]
    return env_cfg, _TASK_SHADOW


def _build_env_cfg_example_3(num_envs: int, legacy_kit: bool = False):
    """Record the Newton RTX view, a tiled scene-camera view, and one sensor image together."""
    from isaaclab_visualizers.kit import KitBackendCfg
    from isaaclab_visualizers.newton import NewtonGLBackendCfg

    env_cfg = _shadow_env_cfg(num_envs)
    tiled_view = ImageViewCfg(source="tiled_camera", envs=tuple(range(min(num_envs, 16))))
    env_cfg.sim.visualizer_cfgs = [
        VisualizerCfg(
            eye=_SHADOW_EYE,
            lookat=_SHADOW_LOOKAT,
            streaming_view=True,
            streaming_envs=min(num_envs, 16),
            window=WindowCfg(size=(1280, 720)),
            focal_length=25.0,
            backend=NewtonGLBackendCfg(initial_renderer="rtx"),
        )
    ]
    streams = [
        ("viz:newton", "newton_viewport", _RENDER_STEP_OFFSET),
        (tiled_view, "tiled_camera_view", _RENDER_STEP_OFFSET),
        ("sensor:tiled_camera", "sensor", 0),
    ]
    if legacy_kit:
        # Keep the original four streams and filenames for explicitly requested Kit runs.
        newton_view = env_cfg.sim.visualizer_cfgs[0].replace(streaming_view=False, backend=NewtonGLBackendCfg())
        env_cfg.sim.visualizer_cfgs = [
            VisualizerCfg(
                eye=_SHADOW_EYE,
                lookat=_SHADOW_LOOKAT,
                streaming_view=True,
                streaming_envs=min(num_envs, 16),
                backend=KitBackendCfg(),
            ),
            newton_view,
        ]
        streams = [
            ("viz:kit", "kit_viewport", _RENDER_STEP_OFFSET),
            ("viz:kit:streaming_view", "tiled_kit_viewport", _RENDER_STEP_OFFSET),
            ("viz:newton", "newton_viewport", 0),
            ("sensor:tiled_camera", "sensor", 0),
        ]
    env_cfg.video_recorders = [
        VideoRecorderCfg(
            source=source,
            output_dir=_output_dir(3),
            output_filename_prefix=prefix,
            video_length=_VIDEO_LENGTH,
            fps=30,
            step_offset=offset,
        )
        for source, prefix, offset in streams
    ]
    return env_cfg, _TASK_SHADOW


_BUILDERS = {
    1: _build_env_cfg_example_1,
    2: _build_env_cfg_example_2,
    3: _build_env_cfg_example_3,
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Video recording tutorial for Isaac Lab environments.")
parser.add_argument(
    "--example", type=int, default=1, choices=[1, 2, 3], help="Which recording example to run (1, 2, or 3)."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


def main():
    """Run the selected video recording example."""
    defaults = {1: 4, 2: 16, 3: 4}
    num_envs = args_cli.num_envs if args_cli.num_envs is not None else defaults[args_cli.example]
    visualizers = next((arg.split("=", 1)[1].split(",") for arg in hydra_args if arg.startswith("visualizer=")), None)
    if args_cli.example == 2:
        env_cfg, task = _BUILDERS[args_cli.example](num_envs)
    else:
        env_cfg, task = _BUILDERS[args_cli.example](num_envs, legacy_kit=bool(visualizers and "kit" in visualizers))
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # The builders configure recording views after task resolution. Reapply the requested
    # visualizer selection, including newton_rtx and none, when launching those views.
    with launch_simulation(env_cfg, {**vars(args_cli), "visualizer": visualizers}):
        env = gym.make(task, cfg=env_cfg)

        out = _output_dir(args_cli.example)
        print(f"[INFO]: Running Example {args_cli.example} — clips → {out}/")
        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")

        print("[INFO]: Setup complete.")
        env.reset()
        for _ in range(_NUM_STEPS):
            with torch.inference_mode():
                actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
                env.step(actions)

        env.close()
        print(f"[INFO]: Done. Clips written to {out}/")


if __name__ == "__main__":
    main()
