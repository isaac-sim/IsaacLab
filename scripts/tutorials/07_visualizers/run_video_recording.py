# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tutorial: recording video from visualizers and scene sensors.

This script demonstrates three progressively richer recording configurations,
all using the Shadow Hand cube-reorientation task
(``Isaac-Reorient-Cube-Shadow-Camera-Direct``).

Example 1 — Kit viewport (simplest)
    One clip from the Kit interactive viewport, showing 4 parallel environments.

    .. code-block:: bash

        uv run python scripts/tutorials/07_visualizers/run_video_recording.py \
            --example 1 --num_envs 4

Example 2 — scene sensor only, headless
    One clip captured directly from the scene's tiled-camera sensor.
    No visualizer window opens; the sensor renders offline.

    .. code-block:: bash

        uv run python scripts/tutorials/07_visualizers/run_video_recording.py \
            --example 2 --num_envs 16

Example 3 — Kit viewport + Kit tiled grid + Newton viewport + scene sensor
    Four independent clip streams recorded simultaneously.

    .. code-block:: bash

        uv run python scripts/tutorials/07_visualizers/run_video_recording.py \
            --example 3 --num_envs 4

Clips are written to ``videos/recording_tutorial/example_<N>/`` in the working directory.
Examples 1 and 2 each demonstrate one recording source; Example 3 combines all of them.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

from isaaclab_tasks.utils import setup_preset_cli

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VIDEO_LENGTH = 100  # env steps per clip
_NUM_STEPS = 115  # slightly more than _VIDEO_LENGTH so the clip flushes cleanly

_TASK_SHADOW = "Isaac-Reorient-Cube-Shadow-Camera-Direct"

# Kit viewport camera: positioned to show a 2×2 grid of Shadow Hand environments.
# env_spacing=1.0 with 4 envs → envs centered at ±0.5 in x and y.
# Cube spawns at ~(0, -0.39, 0.6) per env; wrist cylinder is the landmark at the top.
_SHADOW_EYE = (0.0, -2.2, 1.8)
_SHADOW_LOOKAT = (0.0, -0.1, 0.4)
_SHADOW_ENV_SPACING = 1.0

# Tiled camera eye offset from each robot root for generated per-env cameras.
_SHADOW_TILED_EYE = (0.0, 0.35, 0.8)

# Skip the first few steps so the RTX renderer has warmed up before recording starts.
_KIT_STEP_OFFSET = 5


def _output_dir(example: int) -> str:
    return os.path.join("videos", "recording_tutorial", f"example_{example}")


def _shadow_env_cfg(num_envs: int, env_spacing: float = _SHADOW_ENV_SPACING):
    """Build a base Shadow Hand camera env cfg shared by all examples."""
    from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_direct_camera_env_cfg import ShadowHandCameraEnvCfg

    env_cfg = ShadowHandCameraEnvCfg()
    env_cfg.tiled_camera = env_cfg.tiled_camera.rgb
    env_cfg.tiled_camera.renderer_cfg = env_cfg.tiled_camera.renderer_cfg.default
    env_cfg.tiled_camera.height = 256
    env_cfg.tiled_camera.width = 256
    env_cfg.scene.num_envs = num_envs
    env_cfg.scene.env_spacing = env_spacing
    return env_cfg


# ---------------------------------------------------------------------------
# Per-example environment config builders
# ---------------------------------------------------------------------------


def _build_env_cfg_example_1(num_envs: int):
    """Shadow Hand + Kit viewport: one clip from the interactive viewport."""
    from isaaclab_visualizers.kit import KitVisualizerCfg

    env_cfg = _shadow_env_cfg(num_envs)
    env_cfg.sim.physics = env_cfg.sim.physics.default

    env_cfg.sim.visualizer_cfgs = [KitVisualizerCfg(eye=_SHADOW_EYE, lookat=_SHADOW_LOOKAT)]

    out = _output_dir(1)
    env_cfg.video_recorders = [
        VideoRecorderCfg(
            source="visualizer:kit",
            output_dir=out,
            output_filename_prefix="kit_viewport",
            video_length=_VIDEO_LENGTH,
            fps=30,
            step_offset=_KIT_STEP_OFFSET,
        ),
    ]
    return env_cfg, _TASK_SHADOW


def _build_env_cfg_example_2(num_envs: int):
    """Shadow Hand + headless: scene tiled-camera sensor clip only."""
    env_cfg = _shadow_env_cfg(num_envs, env_spacing=2.0)
    env_cfg.sim.physics = env_cfg.sim.physics.default
    env_cfg.sim.visualizer_cfgs = []  # no interactive visualizer

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


def _build_env_cfg_example_3(num_envs: int):
    """Shadow Hand + Kit viewport + Kit tiled grid + Newton viewport + sensor: four simultaneous streams.

    Note: ``source='visualizer:newton'`` captures the full Newton GL window. When
    ``tiled_cam_view=True`` is set on :class:`~isaaclab_visualizers.newton.NewtonVisualizerCfg`,
    the GL window displays the tiled per-environment camera panel, so this effectively records
    a Newton tiled view without a separate ``render_tiled_rgb_array()`` call.
    """
    from isaaclab_visualizers.kit import KitVisualizerCfg
    from isaaclab_visualizers.newton import NewtonVisualizerCfg

    env_cfg = _shadow_env_cfg(num_envs)
    env_cfg.sim.physics = env_cfg.sim.physics.default

    kit_cfg = KitVisualizerCfg(
        eye=_SHADOW_EYE,
        lookat=_SHADOW_LOOKAT,
        tiled_cam_view=True,
        tiled_cam_num=min(num_envs, 16),
        # Reuse the existing scene camera sensor so the tiled panel shows
        # the same RTX-rendered views as source="sensor:tiled_camera".
        tiled_cam_prim_path="/World/envs/env_.*/Camera",
    )
    newton_cfg = NewtonVisualizerCfg(
        eye=_SHADOW_EYE,
        lookat=_SHADOW_LOOKAT,
        window_width=1280,
        window_height=720,
        focal_length=25.0,
    )
    env_cfg.sim.visualizer_cfgs = [kit_cfg, newton_cfg]

    out = _output_dir(3)
    env_cfg.video_recorders = [
        VideoRecorderCfg(
            source="visualizer:kit",
            output_dir=out,
            output_filename_prefix="kit_viewport",
            video_length=_VIDEO_LENGTH,
            fps=30,
            step_offset=_KIT_STEP_OFFSET,
        ),
        VideoRecorderCfg(
            source="visualizer:kit:tiled",
            output_dir=out,
            output_filename_prefix="tiled_kit_viewport",
            video_length=_VIDEO_LENGTH,
            fps=30,
            step_offset=_KIT_STEP_OFFSET,
        ),
        VideoRecorderCfg(
            source="visualizer:newton",
            output_dir=out,
            output_filename_prefix="newton_viewport",
            video_length=_VIDEO_LENGTH,
            fps=30,
        ),
        VideoRecorderCfg(
            source="sensor:tiled_camera",
            output_dir=out,
            output_filename_prefix="sensor",
            video_length=_VIDEO_LENGTH,
            fps=30,
        ),
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
    env_cfg, task = _BUILDERS[args_cli.example](num_envs)
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Examples 1 and 3 record from the Kit viewport via omni.replicator, which requires
    # camera rendering support.  Force it here for visualizer-only recording.
    if args_cli.example in (1, 3):
        args_cli.enable_cameras = True

    with launch_simulation(env_cfg, args_cli):
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
