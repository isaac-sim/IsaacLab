# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted IK control of the Rizon–Sharpa coffee scene.

Run ``uv run isaaclab demo rizon-sharpa-coffee --water --visualizer newton_gl``.
Use ``--visualizer none --headless`` for a physics-only rollout.
"""

from __future__ import annotations

import argparse
import math
import os
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
import warp as wp

from isaaclab.app import add_launcher_args, launch_simulation

wp.config.enable_backward = False


def main() -> None:
    args, parser = _parse_args()
    video_stride = args.control_hz // args.video_fps
    cfg = _make_env_cfg(args, video_stride)
    if args.headless:
        # Kit's launcher resolves its app window from HEADLESS and the visualizer selection.
        os.environ["HEADLESS"] = "1"
    with launch_simulation(cfg, args), ExitStack() as stack:
        from isaaclab_tasks.contrib.rizon_sharpa_coffee.demo_env import CoffeeDemoEnv
        from isaaclab_tasks.contrib.rizon_sharpa_coffee.rendering import CoffeeRendering
        from isaaclab_tasks.contrib.rizon_sharpa_coffee.scripted import CoffeeScoopDemo

        # Launch owns the final display and recording producers, including Kit
        # added for livestreaming. Resolve render allocation before scene construction.
        cfg.sim.physics.load_visual_shapes = bool(cfg.sim.visualizer_cfgs)
        cfg.grains_per_particle = args.grains_per_particle if cfg.sim.visualizer_cfgs else 0
        if args.keep_open and not any(not visualizer.headless for visualizer in cfg.sim.visualizer_cfgs):
            parser.error("--keep_open requires a desktop visualizer.")
        env = CoffeeDemoEnv(cfg)
        stack.callback(env.close)
        env.reset()
        if cfg.sim.visualizer_cfgs and cfg.grains_per_particle:
            env.rendering = CoffeeRendering(env, voxel_size=cfg.voxel_size)
        if cfg.sim.visualizer_cfgs:
            env.sim.render()
        controller = CoffeeScoopDemo(env)
        camera = (
            _SpoonCamera(env.sim.visualizers, show_cup=cfg.water_enabled)
            if cfg.sim.visualizer_cfgs and args.camera == "follow"
            else None
        )
        step = 0
        with torch.inference_mode():
            while args.keep_open or step < args.max_steps:
                if not env.sim.is_running():
                    break
                if env.sim.consume_reset_request():
                    env.reset()
                    controller.reset()
                    step = 0
                    if camera is not None:
                        camera.reset()
                if controller.finished or step >= args.max_steps:
                    env.sim.render()
                    continue
                action = controller.compute()
                if camera is not None:
                    camera.update(
                        env.scene["spoon"].data.root_link_pos_w.torch[0].tolist(),
                        controller.phase,
                        env.step_dt,
                    )
                env.step(action)
                if not env.is_state_finite().all():
                    raise RuntimeError("Coffee rollout produced nonfinite robot or material positions.")
                if controller.finished:
                    fraction = env.delivered_fraction()[0]
                    delivered_mass = fraction * len(cfg.scene.coffee.spawn.positions) * cfg.scene.coffee.spawn.mass
                    if fraction < cfg.min_delivered_fraction or delivered_mass < cfg.min_delivered_mass:
                        raise RuntimeError("The arm completed its path, but insufficient coffee reached the cup.")
                    if not args.keep_open:
                        break
                    print("Scoop sequence complete. Use the viewer Reset control to run it again.", flush=True)
                step += 1


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--water",
        action="store_true",
        help="Fill the cup, dissolve wet coffee into liquid, diffuse its colour and stir.",
    )
    parser.add_argument("--voxel_size", type=float, default=0.004)
    parser.add_argument(
        "--bowl_scale",
        type=float,
        default=0.65,
        help="Horizontal scale of the source bowl (>= 0.4); the coffee fill is capped independently.",
    )
    parser.add_argument(
        "--physics_hz",
        type=int,
        choices=(100, 120, 200, 400, 800),
        default=None,
        help="Coupling rate; defaults to 400 Hz for dry 2 mm or wet 3+ mm scripts, otherwise 800 Hz.",
    )
    parser.add_argument("--grains_per_particle", type=int, choices=(0, 1, 3, 5, 10, 20, 64, 128), default=1)
    parser.add_argument("--render_samples", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=8000)
    parser.add_argument("--camera", choices=("follow", "fixed"), default="follow")
    parser.add_argument(
        "--backdrop", choices=("kitchen", "studio"), default="kitchen", help="Kit and RTX render backdrop."
    )
    parser.add_argument("--video", action="store_true", help="Record an MP4 at 1x simulation speed.")
    parser.add_argument("--video_fps", type=int, choices=(25, 30, 50, 60, 100), default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--disable_cuda_graph", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--keep_open", action="store_true", help="Keep the desktop viewer open after the scripted cycle."
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    add_launcher_args(parser)
    parser.set_defaults(visualizer=None)
    args = parser.parse_args()
    if args.physics_hz is None:
        fast_script = (not args.water and args.voxel_size == 0.002) or (args.water and args.voxel_size >= 0.003)
        args.physics_hz = 400 if fast_script else 800
    args.control_hz = 60 if args.physics_hz == 120 else 100
    if args.video_fps is None:
        args.video_fps = 30 if args.physics_hz == 120 else 25
    if args.control_hz % args.video_fps:
        parser.error("--video_fps must divide the control rate (60 Hz at 120 Hz physics, otherwise 100 Hz).")
    if args.visualizer is None:
        args.visualizer = ["newton_rtx" if args.video else "newton_gl"]
    if any(name not in ("newton_gl", "newton_rtx", "kit", "none") for name in args.visualizer):
        parser.error("This demo's grain/surface presentation supports newton_gl, newton_rtx, kit or none.")
    if args.keep_open and args.headless:
        parser.error("--keep_open requires a desktop viewer.")
    if args.render_samples < 1:
        parser.error("--render_samples must be positive.")
    return args, parser


def _make_env_cfg(args, video_stride):
    from isaaclab_physx.renderers import IsaacRtxRendererGlobalSettingsCfg
    from isaaclab_visualizers.newton import NewtonGLBackendCfg

    from isaaclab.visualizers import VisualizerCfg, WindowCfg

    from isaaclab_tasks.contrib.rizon_sharpa_coffee.demo_env_cfg import (
        CoffeeDemoEnvCfg,
        CoffeeKitBackendCfg,
        CoffeeRTXVisualizerCfg,
    )

    cfg = CoffeeDemoEnvCfg(water_enabled=args.water)
    cfg.seed = 7
    cfg.voxel_size = args.voxel_size
    cfg.bowl_scale = args.bowl_scale
    cfg.coffee_fill_scale = min(cfg.coffee_fill_scale, cfg.bowl_scale)
    cfg.sim.dt = 1.0 / args.physics_hz
    cfg.decimation = args.physics_hz // args.control_hz
    cfg.sim.render_interval = 2 * cfg.decimation
    # The robot and material keep independent integration rates.
    rigid_hz = 840 if args.physics_hz == 120 else 800
    cfg.sim.physics.solver_cfg.entries[0].substeps = rigid_hz // args.physics_hz
    cfg.sim.physics.use_cuda_graph = not args.disable_cuda_graph
    media = cfg.sim.physics.solver_cfg.entries[1].solver_cfg
    if not args.water and cfg.voxel_size == 0.002:
        # Bound padded grid work for this scripted fill. Other resolutions,
        # and water keep the conservative automatic budget.
        media.max_active_cell_count = 98304
    if args.video:
        from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

        cfg.sim.render_interval = video_stride * cfg.decimation
        cfg.video_recorders = [
            VideoRecorderCfg(
                source="viz",
                output_dir=str(args.output_dir or "logs/coffee_video"),
                output_filename_prefix="coffee",
                video_length=args.max_steps,
                frame_stride=video_stride,
                fps=args.video_fps,
            )
        ]
    view = VisualizerCfg(
        headless=args.headless,
        window=WindowCfg(size=(args.width, args.height)),
        eye=(0.95, -0.65, 0.66),
        lookat=(0.52, 0.02, 0.15),
        focal_length=24.0,
        background_color=(0.008, 0.014, 0.022),
    )
    cfg.sim.visualizer_cfgs = [
        view.replace(
            backend=NewtonGLBackendCfg(
                class_type="isaaclab_tasks.contrib.rizon_sharpa_coffee.rendering:CoffeeGLVisualizer",
                show_particles=args.grains_per_particle == 0,
                particle_color=(0.095, 0.029, 0.010) if not args.water else None,
                enable_picking=False,
                enable_sky=False,
            )
        ),
        view.replace(
            backend=CoffeeRTXVisualizerCfg(
                backdrop=args.backdrop,
                render_settings={
                    "omni:rtx:rtpt:spp": ("Int", args.render_samples),
                    "omni:rtx:rtpt:maxBounces": ("UInt", 8),
                    "omni:rtx:rtpt:extraSpecularAndTransmissiveBounces": ("UInt", 4),
                    "omni:rtx:directLighting:sampledLighting:denoisingTechnique": ("Token", "NRD ReLAX"),
                    "omni:rtx:reflections:denoiser:enabled": ("Bool", True),
                    "omni:rtx:indirectDiffuse:denoiser:enabled": ("Bool", True),
                },
            )
        ),
        view.replace(
            enable_markers=False,
            enable_live_plots=False,
            backend=CoffeeKitBackendCfg(
                backdrop=args.backdrop,
                render_settings=IsaacRtxRendererGlobalSettingsCfg(
                    enable_translucency=True,
                    max_bounces=12,
                    split_glass=False,
                    carb_settings={
                        "/rtx/rtpt/spp": args.render_samples,
                        "/rtx/rtpt/extraSpecularAndTransmissiveBounces": 8,
                    },
                ),
            ),
        ),
    ]
    if args.backdrop == "kitchen":
        for visualizer in cfg.sim.visualizer_cfgs[1:]:
            visualizer.eye, visualizer.lookat = (0.86, -0.61, 0.70), (0.53, -0.03, 0.10)
    return cfg


class _SpoonCamera:
    """Ease framing through the demonstration and clear the arm when stirring."""

    def __init__(self, visualizers, *, show_cup: bool = False):
        self._visualizers = visualizers
        self._initial_eye = np.array(visualizers[0].cfg.eye, dtype=float)
        self._initial_target = np.array(visualizers[0].cfg.lookat, dtype=float)
        self._offset = self._initial_eye - self._initial_target
        self._azimuth = math.atan2(self._offset[1], self._offset[0])
        self._show_cup = show_cup
        self.reset()

    def reset(self):
        self._eye = self._initial_eye.copy()
        self._target = self._initial_target.copy()
        distance = np.linalg.norm(self._offset)
        elevation = math.asin(self._offset[2] / distance)
        # Three cascaded filters keep position, velocity and acceleration
        # continuous even when the scripted phase changes its framing target.
        self._framing = np.tile((*self._target, math.log(distance), elevation, self._azimuth), (3, 1))
        self._stirring = False
        for visualizer in self._visualizers:
            visualizer.set_camera_view(self._eye, self._target)

    def update(self, spoon_position, phase, dt):
        from isaaclab_tasks.contrib.rizon_sharpa_coffee.assets import BOWL_POSITION, CUP_HEIGHT, CUP_POSITION

        spoon = np.asarray(spoon_position)
        # Bias toward the handle so the thumb/index pinch stays in the shot.
        focus = spoon + (-0.045, 0.0, 0.025)
        target = 0.35 * focus + 0.65 * np.array((0.52, 0.02, 0.13))
        distance = 0.85
        elevation = math.radians(48.0)
        azimuth = self._azimuth
        if phase.startswith(("approach", "scoop", "settle")):
            target = 0.25 * focus + 0.75 * (np.array(BOWL_POSITION) + (-0.025, 0.0, 0.075))
            distance = 0.68
            elevation = math.radians(52.0)
        if phase.startswith("stir"):
            self._stirring = True
        if self._show_cup and (self._stirring or phase.startswith(("transfer", "tip", "empty"))):
            target = np.array(CUP_POSITION) + (-0.015, 0.0, CUP_HEIGHT * (0.9 if self._stirring else 0.75))
            distance = 0.33 if self._stirring else 0.50
            elevation = math.radians(35.0 if self._stirring else 60.0)
            if self._stirring:
                # Move right toward a front view of the fingers and liquid,
                # keeping the wrist above the close-up.
                angle = math.radians(-30.0) - self._azimuth
                azimuth += math.atan2(math.sin(angle), math.cos(angle))
        # Filter log distance for a gradual dolly at both wide and close views.
        goal = np.array((*target, math.log(distance), elevation, azimuth))
        errors = self._framing - goal
        elapsed = dt / 0.70
        decay = math.exp(-elapsed)
        self._framing[0] = goal + decay * errors[0]
        self._framing[1] = goal + decay * (errors[1] + elapsed * errors[0])
        self._framing[2] = goal + decay * (errors[2] + elapsed * errors[1] + 0.5 * elapsed**2 * errors[0])
        self._target = self._framing[2, :3].copy()
        distance = math.exp(self._framing[2, 3])
        elevation = self._framing[2, 4]
        azimuth = self._framing[2, 5]
        horizontal = distance * math.cos(elevation)
        self._eye = self._target + (
            horizontal * math.cos(azimuth),
            horizontal * math.sin(azimuth),
            distance * math.sin(elevation),
        )
        for visualizer in self._visualizers:
            visualizer.set_camera_view(self._eye, self._target)


if __name__ == "__main__":
    main()
