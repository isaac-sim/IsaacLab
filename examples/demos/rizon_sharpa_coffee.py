# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted IK or Apple Vision Pro control of the Rizon–Sharpa coffee scene.

Run ``uv run isaaclab demo rizon-sharpa-coffee --water --visualizer newton_gl``.
Use ``--visualizer none --headless`` for a physics-only rollout.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import warp as wp

from isaaclab.app import add_launcher_args, launch_simulation

wp.config.enable_backward = False


def main() -> None:
    args, parser, ffmpeg = _parse_args()
    video_stride = 100 // args.video_fps
    cfg, visualizers = _make_env_cfg(args, parser, video_stride)
    with launch_simulation(cfg, args), ExitStack() as stack:
        from isaaclab_tasks.contrib.rizon_sharpa_coffee.coffee_env import CoffeeEnv
        from isaaclab_tasks.contrib.rizon_sharpa_coffee.presentation import CoffeePresentation
        from isaaclab_tasks.contrib.rizon_sharpa_coffee.scripted import CoffeeScoopDemo

        env = CoffeeEnv(cfg)
        stack.callback(env.close)
        env.reset()
        if visualizers and cfg.grains_per_particle:
            env.presentation = CoffeePresentation(
                env, voxel_size=cfg.voxel_size, surface_smoothing_time=args.surface_smoothing_time
            )
        if visualizers:
            env.sim.render()
        controller = CoffeeScoopDemo(env) if args.mode == "scripted" else None
        camera = (
            _SpoonCamera(env.sim.visualizers, show_cup=cfg.water_enabled)
            if visualizers and args.camera == "follow"
            else None
        )
        device = None
        if args.mode == "avp":
            from isaaclab_teleop import CLOUDXR_AVP_ENV, IsaacTeleopCfg, create_isaac_teleop_device, poll_control_events

            from isaaclab.devices.openxr import XrCfg

            from isaaclab_tasks.contrib.rizon_sharpa_coffee.teleop import CoffeeTeleopController, build_teleop_pipeline

            device = stack.enter_context(
                create_isaac_teleop_device(
                    IsaacTeleopCfg(
                        pipeline_builder=build_teleop_pipeline,
                        xr_cfg=XrCfg(anchor_pos=tuple(args.anchor_pos)),
                        sim_device=env.device,
                    ),
                    cloudxr_env_file=args.cloudxr_env or CLOUDXR_AVP_ENV,
                    use_kit_xr_bridge=False,
                )
            )
            device.reset()
            device.request_start()
            teleop_controller = CoffeeTeleopController(env)
        recording = stack.enter_context(_DemoRecording(env, args, ffmpeg, video_stride))
        step = 0
        with torch.inference_mode():
            while step < args.max_steps:
                if not env.sim.is_running():
                    break
                if device is not None:
                    action = device.advance()
                    events = poll_control_events(device)
                    if events.should_reset:
                        env.reset()
                        device.reset()
                        teleop_controller.reset()
                        if camera is not None:
                            camera.reset()
                    if action is None or events.is_active is False:
                        env.sim.render()
                        continue
                    action = teleop_controller.compute(action)
                else:
                    action = controller.compute()
                if camera is not None:
                    camera.update(
                        env.scene["spoon"].data.root_link_pos_w.torch[0].tolist(),
                        controller.phase if controller is not None else "teleoperation",
                        env.step_dt,
                    )
                observations, reward, terminated, truncated, _ = env.step(env.actions_from_targets(action))
                if not torch.isfinite(observations["policy"]).all() or not torch.isfinite(reward).all():
                    raise RuntimeError("Coffee rollout produced a nonfinite observation or reward.")
                if bool(terminated.any()) or bool(truncated.any()):
                    if controller is not None:
                        reasons = [
                            name
                            for name in env.termination_manager.active_terms
                            if bool(env.termination_manager.get_term(name).any())
                        ]
                        if not reasons:
                            # Desktop Reset is an episode boundary, not a failed simulation.
                            controller.reset()
                            if camera is not None:
                                camera.reset()
                            recording.previous_phase = None
                            step += 1
                            continue
                        raise RuntimeError(f"Scripted rollout ended at step {step}: {', '.join(reasons)}.")
                    device.reset()
                    teleop_controller.reset()
                if env.presentation is not None:
                    env.presentation.update(env.step_dt)
                phase = str(controller.phase) if controller is not None else "teleoperation"
                recording.update(step, phase)
                if controller is not None and controller.finished:
                    if env.particle_fractions()[0, 2] < cfg.success_fraction:
                        raise RuntimeError("The arm completed its path, but insufficient coffee reached the cup.")
                    recording.close()
                    # Capture the complete sequence; desktop Reset can replay it independently.
                    if not args.keep_open:
                        break
                    print("Scoop sequence complete. Use the viewer Reset control to run it again.", flush=True)
                    if not _wait_for_reset(env, controller):
                        break
                    if camera is not None:
                        camera.reset()
                    recording.previous_phase = None
                step += 1
        recording.close()
        print("Final coffee fractions [bowl, spoon, cup, spill]:", env.particle_fractions()[0].tolist(), flush=True)
        if args.keep_open:
            while env.sim.is_running():
                env.sim.render()


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("scripted", "avp"), default="scripted")
    parser.add_argument(
        "--water",
        action="store_true",
        help="Fill the cup, dissolve wet coffee into liquid, diffuse its colour and stir.",
    )
    parser.add_argument(
        "--water_fill_height", type=float, default=None, help="Override the task's water seed height [m]."
    )
    parser.add_argument("--voxel_size", type=float, default=0.002)
    parser.add_argument(
        "--spoon_margin", type=float, default=None, help="Override the broad spoon collision margin [m]."
    )
    parser.add_argument(
        "--spoon_rim_margin_scale", type=float, default=1.0 / 9.0, help="Rim margin relative to the broad spoon faces."
    )
    parser.add_argument(
        "--coffee_wetting_time",
        type=float,
        default=0.12,
        help="Accumulated water contact before coffee becomes liquid [s].",
    )
    parser.add_argument(
        "--physics_hz",
        type=int,
        choices=(100, 200, 400, 800),
        default=None,
        help="Coupling rate; defaults to 400 Hz for dry 2 mm or wet 3+ mm mounted scripts, otherwise 800 Hz.",
    )
    parser.add_argument(
        "--rigid_hz", type=int, choices=(400, 800), default=800, help="Mounted robot integration rate [Hz]."
    )
    parser.add_argument("--mpm_substeps", type=int, default=None, help="MPM substeps per physics tick.")
    parser.add_argument(
        "--mpm_iterations",
        type=int,
        default=None,
        help="Maximum MPM solver iterations; default 500 with water, 100 dry.",
    )
    parser.add_argument(
        "--spoon_proxy_mass_scale",
        type=float,
        default=None,
        help="Contact-grasp MPM inertia multiplier; default 20. The rigid spoon remains 25 g.",
    )
    parser.add_argument(
        "--mpm_grid_capacity",
        type=int,
        default=None,
        help=(
            "Active-cell budget; default 98304 for the dry 2 mm script, otherwise automatic."
            " Zero uses the full particle budget."
        ),
    )
    parser.add_argument("--grains_per_particle", type=int, choices=(0, 1, 5, 10, 20, 64, 128), default=5)
    parser.add_argument("--render_samples", type=int, default=4)
    parser.add_argument(
        "--surface_smoothing_time",
        type=float,
        default=0.06,
        help="Liquid render-field smoothing time [s]; zero disables.",
    )
    parser.add_argument("--max_steps", type=int, default=8000)
    parser.add_argument("--camera", choices=("follow", "fixed"), default="follow")
    parser.add_argument("--video", action="store_true", help="Record an MP4 at 1x simulation speed.")
    parser.add_argument("--video_fps", type=int, choices=(25, 50, 100), default=25)
    parser.add_argument("--video_crf", type=int, default=14, help="H.264 quality from 0 to 51; lower is better.")
    parser.add_argument(
        "--video_render_scale", type=int, choices=(1, 2), default=1, help="Render scale before downsampling the MP4."
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--disable_cuda_graph", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--keep_open", action="store_true", help="Keep the desktop viewer open after the scripted cycle."
    )
    parser.add_argument("--cloudxr_env", type=str, default=None)
    parser.add_argument("--grasp_reset", type=str, default=None, help="Load an authored, named-joint grasp reset JSON.")
    parser.add_argument("--grasp_mode", choices=("mounted", "contact"), default="mounted")
    parser.add_argument("--anchor_pos", type=float, nargs=3, default=(0.0, 0.0, -0.55))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    add_launcher_args(parser)
    parser.set_defaults(visualizer=None)
    args = parser.parse_args()
    if args.physics_hz is None:
        fast_script = (
            args.mode == "scripted"
            and args.grasp_mode == "mounted"
            and ((not args.water and args.voxel_size == 0.002) or (args.water and args.voxel_size >= 0.003))
        )
        args.physics_hz = 400 if fast_script else 800
    if args.visualizer is None:
        args.visualizer = ["newton_rtx" if args.video else "newton_gl"]
    if args.grasp_mode == "mounted" and args.rigid_hz < args.physics_hz:
        parser.error("--rigid_hz must be at least --physics_hz.")
    if args.grasp_mode == "contact" and args.rigid_hz != 800:
        parser.error("--rigid_hz applies only to mounted tools; contact grasping uses 1600 Hz.")
    if args.spoon_margin is not None and (not math.isfinite(args.spoon_margin) or args.spoon_margin < 0.0):
        parser.error("--spoon_margin must be finite and nonnegative.")
    if args.mpm_grid_capacity is not None and args.mpm_grid_capacity < 0:
        parser.error("--mpm_grid_capacity must be nonnegative.")
    if args.keep_open and args.headless:
        parser.error("--keep_open requires a desktop viewer.")
    if args.render_samples < 1:
        parser.error("--render_samples must be positive.")
    if not 0 <= args.video_crf <= 51:
        parser.error("--video_crf must be between 0 and 51.")
    if not math.isfinite(args.surface_smoothing_time) or args.surface_smoothing_time < 0:
        parser.error("--surface_smoothing_time must be finite and nonnegative.")
    if args.mpm_substeps is not None and args.mpm_substeps < 1:
        parser.error("--mpm_substeps must be positive.")
    if args.mpm_iterations is not None and args.mpm_iterations < 1:
        parser.error("--mpm_iterations must be positive.")
    if args.spoon_proxy_mass_scale is not None and (
        not math.isfinite(args.spoon_proxy_mass_scale) or args.spoon_proxy_mass_scale <= 0.0
    ):
        parser.error("--spoon_proxy_mass_scale must be finite and positive.")
    if args.video and args.output_dir is None:
        args.output_dir = Path("logs/coffee_video") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ffmpeg = _find_encoder(args.output_dir)
    if args.video and ffmpeg is None:
        parser.error("--video requires FFmpeg; install the video extra or a system FFmpeg encoder.")
    return args, parser, ffmpeg


def _make_env_cfg(args, parser, video_stride):
    from isaaclab_tasks.contrib.rizon_sharpa_coffee.coffee_env_cfg import CoffeeEnvCfg

    cfg = CoffeeEnvCfg(water_enabled=args.water)
    cfg.seed = 7
    cfg.sim.device = args.device
    cfg.voxel_size = args.voxel_size
    if args.spoon_margin is not None:
        cfg.spoon_margin_voxels = args.spoon_margin / cfg.voxel_size
    cfg.spoon_rim_margin_scale = args.spoon_rim_margin_scale
    cfg.sim.dt = 1.0 / args.physics_hz
    cfg.decimation = args.physics_hz // 100
    cfg.sim.render_interval = 2 * cfg.decimation
    # The robot and material keep independent integration rates.
    cfg.sim.physics.solver_cfg.entries[0].substeps = args.rigid_hz // args.physics_hz
    if args.spoon_proxy_mass_scale is not None:
        cfg.spoon_proxy_mass_scale = args.spoon_proxy_mass_scale
    if args.water_fill_height is not None:
        cfg.water_fill_height = args.water_fill_height
    cfg.coffee_wetting_time = args.coffee_wetting_time
    cfg.grasp_reset_path = args.grasp_reset
    cfg.grasp_mode = args.grasp_mode
    cfg.sim.physics.use_cuda_graph = not args.disable_cuda_graph
    media = cfg.sim.physics.solver_cfg.entries[1].solver_cfg
    if args.mpm_iterations is not None:
        media.max_iterations = args.mpm_iterations
    if args.mpm_grid_capacity:
        media.max_active_cell_count = args.mpm_grid_capacity
    elif args.mpm_grid_capacity is None and args.mode == "scripted" and not args.water and cfg.voxel_size == 0.002:
        # Bound padded grid work for this scripted fill. Other resolutions,
        # water and live teleoperation keep the conservative automatic budget.
        media.max_active_cell_count = 98304
    if args.mpm_substeps is not None:
        cfg.sim.physics.solver_cfg.entries[1].substeps = args.mpm_substeps
    cfg.terminations.success = None
    cfg.episode_length_s = max(40.0, args.max_steps * cfg.sim.dt * cfg.decimation + 1.0)
    if args.video:
        # Render at the recording cadence, independently of the physics tick rate.
        cfg.sim.render_interval = video_stride * cfg.decimation
    visualizers = _make_visualizers(args, parser)
    cfg.sim.visualizer_cfgs = visualizers
    if args.keep_open and not visualizers:
        parser.error("--keep_open requires a desktop visualizer.")
    if args.video and not visualizers:
        parser.error("--video requires a GL, RTX or Kit visualizer; --headless recording is supported.")
    if args.video and any(visualizer.visualizer_type == "kit" for visualizer in visualizers):
        args.enable_cameras = True
    cfg.sim.physics.load_visual_shapes = bool(visualizers)
    cfg.grains_per_particle = args.grains_per_particle if visualizers else 0
    return cfg, visualizers


def _make_visualizers(args, parser):
    from isaaclab_physx.renderers import IsaacRtxRendererCfg, IsaacRtxRendererGlobalSettingsCfg
    from isaaclab_visualizers.kit import KitVisualizerCfg
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg

    from isaaclab_tasks.contrib.rizon_sharpa_coffee.presentation import CoffeeGLVisualizer, CoffeeRTXVisualizer

    visualizers = []
    camera_cfg = dict(
        headless=args.headless,
        window_width=args.width * (args.video_render_scale if args.video else 1),
        window_height=args.height * (args.video_render_scale if args.video else 1),
        eye=(0.95, -0.65, 0.66),
        lookat=(0.52, 0.02, 0.15),
        focal_length=24.0,
        background_color=(0.008, 0.014, 0.022),
    )
    for name in args.visualizer or []:
        if name == "none":
            continue
        if name not in ("newton_gl", "newton_rtx", "kit"):
            parser.error("This demo's grain/surface presentation supports newton_gl, newton_rtx or kit.")
        if name == "kit":
            visualizers.append(
                KitVisualizerCfg(
                    class_type="isaaclab_tasks.contrib.rizon_sharpa_coffee.kit_presentation:CoffeeKitVisualizer",
                    streaming_cam_renderer_cfg=IsaacRtxRendererCfg(
                        global_settings=IsaacRtxRendererGlobalSettingsCfg(
                            enable_translucency=True,
                            max_bounces=12,
                            split_glass=False,
                            carb_settings={
                                "/rtx/rtpt/spp": args.render_samples,
                                "/rtx/rtpt/extraSpecularAndTransmissiveBounces": 8,
                            },
                        )
                    ),
                    enable_markers=False,
                    enable_live_plots=False,
                    **camera_cfg,
                )
            )
            continue
        common = dict(
            show_particles=args.grains_per_particle == 0,
            particle_color=(0.095, 0.029, 0.010) if not args.water else None,
            enable_picking=False,
            enable_sky=False,
            **camera_cfg,
        )
        visualizers.append(
            NewtonRTXVisualizerCfg(
                class_type=CoffeeRTXVisualizer,
                rtx_environment="studio",
                render_settings={
                    "omni:rtx:rtpt:spp": ("Int", args.render_samples),
                    "omni:rtx:rtpt:maxBounces": ("UInt", 8),
                    "omni:rtx:rtpt:extraSpecularAndTransmissiveBounces": ("UInt", 4),
                    "omni:rtx:directLighting:sampledLighting:denoisingTechnique": ("Token", "NRD ReLAX"),
                    "omni:rtx:reflections:denoiser:enabled": ("Bool", True),
                    "omni:rtx:indirectDiffuse:denoiser:enabled": ("Bool", True),
                },
                **common,
            )
            if name == "newton_rtx"
            else NewtonGLVisualizerCfg(class_type=CoffeeGLVisualizer, **common)
        )
    return visualizers


def _find_encoder(output):
    ffmpeg = shutil.which("ffmpeg")
    if output is not None and ffmpeg is None:
        try:
            from imageio_ffmpeg import get_ffmpeg_exe

            ffmpeg = get_ffmpeg_exe()
        except ImportError:
            print("Video encoding needs the video extra; saving PNG captures and metrics.", flush=True)
    return ffmpeg


def _wait_for_reset(env, controller) -> bool:
    """Keep the finished scene visible and rearm its controller on desktop Reset."""
    while env.sim.is_running():
        env.sim.render()
        if env.sim.consume_reset_request():
            env.reset()
            controller.reset()
            return True
    return False


def _close_video(video: subprocess.Popen) -> None:
    if not video.stdin.closed:
        video.stdin.close()
    if video.wait() != 0:
        raise RuntimeError("ffmpeg failed while encoding the coffee demo.")


def _write_metrics(output: Path, metrics: list) -> None:
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


class _SpoonCamera:
    """Follow the tool, with elevated cup close-ups during pouring and stirring."""

    def __init__(self, visualizers, *, show_cup: bool = False):
        self._visualizers = visualizers
        self._initial_eye = np.array(visualizers[0].cfg.eye, dtype=float)
        self._initial_target = np.array(visualizers[0].cfg.lookat, dtype=float)
        self._offset = self._initial_eye - self._initial_target
        self._show_cup = show_cup
        self.reset()

    def reset(self):
        self._eye = self._initial_eye.copy()
        self._target = self._initial_target.copy()
        self._cup_revealed = False
        for visualizer in self._visualizers:
            visualizer.set_camera_view(self._eye, self._target)

    def update(self, spoon_position, phase, dt):
        from isaaclab_tasks.contrib.rizon_sharpa_coffee.assets import BOWL_POSITION, CUP_POSITION

        spoon = np.asarray(spoon_position)
        # Bias toward the handle so the thumb/index pinch stays in the shot.
        focus = spoon + (-0.045, 0.0, 0.025)
        target = 0.45 * focus + 0.55 * np.array((0.52, 0.02, 0.11))
        scale = 1.0
        if phase.startswith("scoop"):
            proximity = np.clip((0.20 - spoon[2]) / 0.13, 0.0, 1.0)
            blend = proximity * proximity * (3.0 - 2.0 * proximity)
            pickup_target = 0.7 * focus + 0.3 * (np.array(BOWL_POSITION) + (-0.025, 0.0, 0.055))
            target = (1.0 - blend) * target + blend * pickup_target
            scale -= 0.30 * blend
        elif phase.startswith(("pour_height", "tip", "empty", "stir")):
            target = 0.55 * focus + 0.45 * (np.array(CUP_POSITION) + (-0.015, 0.0, 0.055))
            scale = 0.85
        if self._show_cup and phase == "empty_1":
            self._cup_revealed = True
        offset = self._offset
        if self._cup_revealed:
            # Raise and gently orbit toward +X at nearly constant distance.
            # The steeper view clears the near rim during the second scoop.
            offset = offset + (0.05, 0.30, 0.22)
        if self._show_cup and phase.startswith(("pour_height", "tip", "empty", "stir")):
            # View from -X/+Y at 55 degrees, clearing the hand along the full
            # scoop path. Keep the opening and falling grains large in frame.
            offset = np.array((-0.36772, 0.36772, 0.74270))
            target = np.array(CUP_POSITION) + (0.0, 0.0, 0.075)
            scale = 0.42
            if phase.startswith("stir"):
                target = np.array(CUP_POSITION) + (0.0, 0.0, 0.052)
                scale = 0.32
        # Time-based damping keeps motion smooth regardless of rendering speed.
        weight = -math.expm1(-dt / 0.65)
        self._target += weight * (target - self._target)
        self._eye += weight * (target + scale * offset - self._eye)
        for visualizer in self._visualizers:
            visualizer.set_camera_view(self._eye, self._target)


class _DemoRecording(ExitStack):
    """Own streamed video, snapshots and rollout logs, including partial failures."""

    def __init__(self, env, args, ffmpeg: str | None, stride: int):
        super().__init__()
        self._env = env
        self._output = args.output_dir
        self._ffmpeg = ffmpeg
        self._crf = args.video_crf
        self._stride = stride
        self._size = tuple((size + 1) // 2 * 2 for size in (args.width, args.height))
        self._fps = 1.0 / (stride * env.step_dt)
        self._video = None
        self._metrics = []
        self.previous_phase = None
        cfg = env.cfg
        self._coffee_mass_g = len(cfg.scene.coffee.spawn.positions) * cfg.scene.coffee.spawn.mass * 1000.0
        if self._output is not None:
            self._output.mkdir(parents=True, exist_ok=True)
            metadata = dict(
                schema_version=2,
                task="IsaacContrib-Rizon-Sharpa-Coffee",
                arguments=vars(args),
                simulation=cfg.sim.to_dict(),
                seed=cfg.seed,
                voxel_size_m=cfg.voxel_size,
                spoon_margin_m=cfg.voxel_size * cfg.spoon_margin_voxels,
                spoon_rim_margin_m=cfg.voxel_size * cfg.spoon_margin_voxels * cfg.spoon_rim_margin_scale,
                coffee_fill_height_m=cfg.coffee_fill_height,
                coffee_particles=len(cfg.scene.coffee.spawn.positions),
                coffee_mass_g=self._coffee_mass_g,
                water_particles=len(cfg.scene.water.spawn.positions) if cfg.water_enabled else 0,
                water_fill_height_m=cfg.water_fill_height if cfg.water_enabled else None,
                control_dt_s=env.step_dt,
                policy_actions=env.action_manager.total_action_dim,
                video_fps=self._fps,
                video_stride_steps=stride,
                video_timebase="simulation",
            )
            (self._output / "run.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
            self._poses = self.enter_context((self._output / "spoon_poses.jsonl").open("w", buffering=1))
            self.callback(_write_metrics, self._output, self._metrics)

    def __exit__(self, *exc_info):
        """Flush captures once; a desktop replay can continue without overwriting them."""
        try:
            return super().__exit__(*exc_info)
        finally:
            self._output = None
            self._video = None

    def update(self, step: int, phase: str) -> None:
        """Log measured progress and capture only at the recording cadence."""
        env, output = self._env, self._output
        new_phase = phase != self.previous_phase
        if output is not None and step % 5 == 0:
            pose = env.scene["spoon"].data.root_link_pose_w.torch.tolist()
            self._poses.write(json.dumps(dict(step=step, phase=phase, pose=pose)) + "\n")
        if step % 100 == 0 or new_phase:
            fractions = env.particle_fractions()[0].tolist()
            row = dict(
                step=step,
                time=(step + 1) * env.step_dt,
                phase=phase,
                source=fractions[0],
                spoon=fractions[1],
                cup=fractions[2],
                spill=fractions[3],
                carried_g=fractions[1] * self._coffee_mass_g,
                delivered_g=fractions[2] * self._coffee_mass_g,
            )
            if env.cfg.water_enabled:
                row["dissolved_coffee_g"] = 1000.0 * env.dissolution.converted_mass_kg
                if env.presentation is not None:
                    row["visual_tracer_g"] = 1000.0 * env.presentation.tracer_mass_kg
            if output is not None:
                self._metrics.append(row)
            print(json.dumps(row), flush=True)
        if output is not None and env.sim.visualizers and (step % self._stride == 0 or new_phase):
            self._capture_frame(step, phase, new_phase)
        self.previous_phase = phase

    def _capture_frame(self, step: int, phase: str, new_phase: bool) -> None:
        from PIL import Image

        frame = self._env.sim.visualizers[0].render_rgb_array()
        if frame is None:
            return
        image = Image.fromarray(frame[..., :3])
        encode_frame = step % self._stride == 0
        if self._video is None and self._ffmpeg is not None and encode_frame:
            self._video = subprocess.Popen(
                [
                    self._ffmpeg,
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{self._size[0]}x{self._size[1]}",
                    "-r",
                    str(self._fps),
                    "-i",
                    "-",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "slow",
                    "-tune",
                    "grain",
                    "-crf",
                    str(self._crf),
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(self._output / "coffee.mp4"),
                ],
                stdin=subprocess.PIPE,
            )
            self.callback(_close_video, self._video)
        if self._video is not None and encode_frame:
            # Desktop resizing must not change the raw stream's declared dimensions.
            video_frame = image if image.size == self._size else image.resize(self._size, Image.Resampling.LANCZOS)
            self._video.stdin.write(video_frame.tobytes())
        if new_phase:
            image.save(self._output / f"{step:04d}_{phase}.png")
        if new_phase or step % 100 == 0:
            preview = self._output / "latest.tmp.png"
            image.save(preview)
            preview.replace(self._output / "latest.png")


if __name__ == "__main__":
    main()
