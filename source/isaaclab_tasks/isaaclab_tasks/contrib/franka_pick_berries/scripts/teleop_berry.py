# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the berry workcell, scripted contact validation, or continuous-grasp teleop."""

import argparse
import json
import math
import shutil
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path

import warp as wp

wp.config.enable_backward = False

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--berry",
    choices=["raspberry", "blackberry", "blueberry", "strawberry", "all"],
    default="raspberry",
)
parser.add_argument(
    "--target_berry",
    choices=["raspberry", "blackberry", "blueberry", "strawberry"],
    default="raspberry",
    help="Scripted pick/squash/place and close-up target when --berry all is selected",
)
parser.add_argument(
    "--count",
    type=int,
    choices=[1, 2, 3],
    default=1,
    help="Berries of the selected species in the punnet; scripted modes handle the first",
)
parser.add_argument(
    "--layout_seed",
    type=int,
    default=0,
    help="Seed of the random punnet layout of several berries (EBC); reset replays it",
)
parser.add_argument(
    "--fixed_layout",
    action="store_true",
    help="Place several berries in a fixed, unrotated layout instead of the random one",
)
parser.add_argument(
    "--tissue_solver",
    choices=["explicit", "implicit"],
    default="explicit",
    help="Explicit MLS-MPM with frictional finger pads, or Newton's implicit MPM with clamped grasping",
)
parser.add_argument(
    "--mode",
    choices=["gamepad", "keyboard", "idle", "pick", "place", "squash", "sort"],
    default="gamepad",
)
parser.add_argument(
    "--steps", type=int, default=0, help="0 runs until the window closes, or a headless scripted run ends"
)
parser.add_argument(
    "--loop", action="store_true", help="Reset and repeat scripted pick/place/squash/sort demonstrations"
)
parser.add_argument(
    "--motion_speed",
    type=float,
    default=2.0,
    help="Arm motion multiplier (default: 2; 1 restores previous speed); grasp speed is unchanged",
)
parser.add_argument("--no_render", action="store_true")
parser.add_argument(
    "--background",
    choices=["studio", "ebc"],
    default="studio",
    help="Studio surface (default), or an EBC lab table with punnet, glass bowl and reject dish",
)
parser.add_argument(
    "--view",
    choices=["auto", "berry", "workcell", "scene", "director"],
    default="auto",
    help=(
        "Initial camera; auto shows the plate/bowl for EBC and a close-up for studio. director (sort mode) moves"
        " between the workcell and a close-up of the berry being handled"
    ),
)
parser.add_argument(
    "--physics_resolution",
    choices=["full", "half"],
    default="full",
    help="full: original tissue; half: about half the physics particles, unchanged visual Gaussians",
)
parser.add_argument(
    "--pick_gap", type=float, default=None, help="Scripted pick opening [m]; default: 70%% of the berry width"
)
asset_selection = parser.add_mutually_exclusive_group()
asset_selection.add_argument(
    "--asset", type=Path, help="Use a specific self-contained berry USD/USDZ instead of the default asset"
)
asset_selection.add_argument(
    "--asset_version",
    choices=["v1", "v2", "v3"],
    default="v1",
    help="v1: original <berry>.usdz (default); v2: repaired <berry>_v2.usdz; v3: <berry>_v3.usdz",
)
parser.add_argument(
    "--sh_rotation",
    choices=["on", "off"],
    default="on",
    help="Rotate SH shading with live material deformation (default: on); physics and bruise tint are unchanged",
)
parser.add_argument(
    "--hide_interior",
    action="store_true",
    help="Hide interior Gaussians for appearance comparisons; tissue physics is unchanged",
)
parser.add_argument(
    "--sync_render",
    action="store_true",
    help="Disable physics/render overlap for comparison",
)
parser.add_argument(
    "--capture_every",
    type=int,
    default=30,
    help="Image capture interval; 0 disables screenshots",
)
parser.add_argument("--window", action="store_true", help="Show the scripted benchmark in a real window")
parser.add_argument("--partitions", type=int, choices=[1, 4], default=1)
parser.add_argument(
    "--rtpt_spp",
    type=int,
    help="Explicit realtime path-tracing samples per pixel (e.g. 4 for captures)",
)
parser.add_argument("--output", type=Path)
parser.add_argument(
    "--print_metrics",
    action="store_true",
    help="Print detailed berry metrics every 30 steps (quiet by default; --output still saves them)",
)
parser.add_argument(
    "--video",
    action="store_true",
    help="Record output/demo.mp4 at simulation time (adds capture/encoding cost)",
)
parser.add_argument("--gamepad_device", help="Linux joystick path, e.g. /dev/input/js0")
parser.add_argument(
    "--verify_reset",
    action="store_true",
    help="Check exact tissue/contact restoration after the run",
)
parser.add_argument(
    "--verify_render",
    action="store_true",
    help="Read back and check all native Gaussian attributes after the run",
)
parser.add_argument(
    "--f_stop",
    type=float,
    help="Depth of field with this camera f-number in --view director close shots (e.g. 64; smaller blurs more)",
)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument(
    "--tracy",
    action="store_true",
    help="Profile with Tracy: renderer, physics and task zones on port 8086 (see the README)",
)
parser.add_argument(
    "--tracy_sync",
    action="store_true",
    help="With --tracy, time GPU work per zone: synchronizes each zone and disables the physics CUDA graph",
)
parser.add_argument(
    "--tracy_setting",
    action="append",
    default=[],
    metavar="SETTING",
    help="With --tracy, an extra Carbonite renderer setting such as --tracy_setting=--/app/profilerMask=7 (repeatable)",
)
add_launcher_args(parser)
parser.set_defaults(device="cuda:0", visualizer=[], headless=True)
args = parser.parse_args()
if not math.isfinite(args.motion_speed) or args.motion_speed <= 0:
    parser.error("--motion_speed must be finite and positive")
if args.berry == "all" and args.asset:
    parser.error("--asset requires a single --berry; use --asset_version for all berries")
if args.count > 1 and args.berry == "all":
    parser.error("--count requires a single --berry species")
if args.layout_seed < 0:
    parser.error("--layout_seed must be nonnegative")
if args.mode == "place" and args.background != "ebc":
    parser.error("--mode place requires --background ebc (punnet and receiving bowl)")
if args.mode == "sort" and (args.count != 3 or args.background != "ebc" or args.berry == "all"):
    parser.error("--mode sort requires --count 3 berries of one species and --background ebc")
if args.view == "director" and args.mode != "sort":
    parser.error("--view director requires --mode sort")
if args.loop and args.mode not in ("pick", "place", "squash", "sort"):
    parser.error("--loop requires a scripted pick, place, squash or sort mode")
if args.rtpt_spp is not None and args.rtpt_spp < 1:
    parser.error("--rtpt_spp must be positive")
if args.pick_gap is not None and not 0 < args.pick_gap <= 0.08:
    parser.error("--pick_gap must be in (0, 0.08] m")
if args.steps < 0 or args.width <= 0 or args.height <= 0 or args.capture_every < 0:
    parser.error("Steps/capture interval must be nonnegative and image size positive")
if args.mode == "keyboard" and args.no_render:
    parser.error("Keyboard mode requires its interactive window")
if args.tracy_setting and not (args.tracy or args.tracy_sync):
    parser.error("--tracy_setting requires --tracy")
if (args.tracy or args.tracy_sync) and args.no_render:
    parser.error("--tracy requires rendering: the Tracy client ships with the OVRTX renderer")
if args.verify_render and args.no_render:
    parser.error("--verify_render requires rendering")
if args.video and (args.no_render or args.output is None or shutil.which("ffmpeg") is None):
    parser.error("--video requires rendering, --output, and ffmpeg on PATH")
if args.video and (args.width % 2 or args.height % 2):
    parser.error("Video dimensions must be even")
if args.output and args.output.exists() and any(args.output.iterdir()):
    parser.error("Output directory is not empty; choose a new directory to preserve earlier runs")
if args.video and (args.output / "demo.mp4").exists():
    parser.error("Refusing to overwrite an existing demo.mp4; choose a new output directory")
if args.mode == "gamepad":
    devices = [Path(args.gamepad_device)] if args.gamepad_device else list(Path("/dev/input").glob("js*"))
    if not any(p.exists() for p in devices):
        parser.error("No Linux joystick found. Connect a controller or use --mode keyboard")

if not args.no_render:
    from isaaclab_tasks.contrib.franka_pick_berries.rendering.settings import require_live_gaussian_renderer

    require_live_gaussian_renderer()

from isaaclab_tasks.contrib.franka_pick_berries import profiling  # noqa: E402
from isaaclab_tasks.contrib.franka_pick_berries.pick_berries_env_cfg import BerryPickEnvCfg  # noqa: E402

cfg = BerryPickEnvCfg(background=args.background)
if args.background == "ebc" and not args.no_render:
    from isaaclab.utils.assets import retrieve_file_path

    from isaaclab_tasks.contrib.franka_pick_berries.scene.background import ebc_background_path

    retrieve_file_path(ebc_background_path())
cfg.berry = args.berry
cfg.target_berry = args.target_berry
cfg.berry_count = args.count
cfg.tissue_solver = args.tissue_solver
cfg.randomize_layout = not args.fixed_layout
cfg.layout_seed = args.layout_seed
cfg.berry_asset_path = str(args.asset.resolve()) if args.asset else None
cfg.berry_asset_version = args.asset_version
cfg.physics_resolution = args.physics_resolution
cfg.sim.device = args.device
if args.tracy or args.tracy_sync:
    profiling.enable(sync=args.tracy_sync, settings=args.tracy_setting)
    if args.tracy_sync:
        # Run the solvers from Python on every step, so that their zones appear.
        cfg.sim.physics.use_cuda_graph = False
with launch_simulation(cfg, args), ExitStack() as resources:
    import gymnasium as gym
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation

    from isaaclab_tasks.contrib.franka_pick_berries.control.gamepad import BerryGamepad, BerryGamepadCfg
    from isaaclab_tasks.contrib.franka_pick_berries.control.motion import pause_speed, scripted_motion_time
    from isaaclab_tasks.contrib.franka_pick_berries.physics.coupling import coupled_solver
    from isaaclab_tasks.contrib.franka_pick_berries.scene.tableware import BOWL

    env = gym.make("IsaacContrib-Pick-Berry-Franka-IK-Rel-Newton", cfg=cfg).unwrapped
    resources.callback(env.close)
    if profiling.enabled():
        profiling.instrument(env.sim, "step", "physics", 0x4C8BF5)
        profiling.instrument_solvers(coupled_solver())
    env.reset()
    viewer = None
    if not args.no_render:
        from isaaclab_tasks.contrib.franka_pick_berries.rendering.viewer import BerryViewer

        viewer = BerryViewer(
            env,
            width=args.width,
            height=args.height,
            headless=not args.window and args.mode in ("idle", "pick", "place", "squash", "sort"),
            pipeline=not args.sync_render,
            partitions=args.partitions,
            rtpt_spp=args.rtpt_spp,
            hide_interior=args.hide_interior,
            sh_rotation=args.sh_rotation == "on",
            view=args.view,
            f_stop=args.f_stop,
        )
        resources.callback(viewer.close)
    controller = BerryGamepad(BerryGamepadCfg(device=args.gamepad_device)) if args.mode == "gamepad" else None
    if controller is not None:
        resources.callback(controller.close)
        print(controller, flush=True)
        controller.add_callback("R", lambda: (env.reset(), controller.reset()))
    action = torch.zeros((1, 7))
    action[0, 6] = 1
    rows, timings = [], []
    step = 0
    script_step = 0
    reset_held = False
    reset_verified = None
    render_verified = None
    video = None
    scripted_center = None
    scripted_gap = args.pick_gap
    scripted_opening = 0.08
    # Scripted arm commands per control step: 8 mm and 0.03 rad at the default speed 2, scaled with it so that the
    # arm keeps up with faster plans; above it, pauses shorten too.
    step_limit, turn_limit = 0.004 * args.motion_speed, 0.015 * args.motion_speed
    pauses = pause_speed(args.motion_speed)
    sorter = sort_result = director = None
    if args.mode == "sort":
        from isaaclab_tasks.contrib.franka_pick_berries.control.sorting import BerrySortSequence, sorting_result

        if args.view == "director":
            from isaaclab_tasks.contrib.franka_pick_berries.rendering.cinematography import SortCinematography

            # For a video: an establishing shot before the sequence, and the crush and the first gentle grasp at
            # the gripper's original pace, which shows their deformation longer.
            sorter = BerrySortSequence(args.motion_speed, args.pick_gap, start_delay=3.0, slow_closing=(0, 1))
            director = SortCinematography()
        else:
            sorter = BerrySortSequence(args.motion_speed, args.pick_gap)
    if args.video:
        args.output.mkdir(parents=True, exist_ok=True)
        video = subprocess.Popen(
            [
                "ffmpeg",
                "-v",
                "error",
                "-n",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{args.width}x{args.height}",
                "-r",
                "30",
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(args.output / "demo.mp4"),
            ],
            stdin=subprocess.PIPE,
        )
    try:
        while (not args.steps or step < args.steps) and (viewer is None or viewer.is_running()):
            started = time.perf_counter()
            script_time = scripted_motion_time(script_step / 30, args.mode, args.motion_speed, pauses)
            finished = sorter.complete if sorter is not None else script_time >= (32 if args.mode == "place" else 18)
            if (
                finished
                and viewer is None
                and not args.steps
                and not args.loop
                and args.mode in ("pick", "place", "squash", "sort")
            ):
                break  # Without a window or a step count, a scripted run ends with its script.
            if args.loop and finished:
                if sorter is not None:
                    sort_result = sorting_result(env.berries)
                    sorter.reset()
                    if director is not None:
                        director.reset()
                env.reset()
                script_step = 0
                scripted_center = None
                scripted_gap = args.pick_gap
                scripted_opening = 0.08
            if viewer is not None:
                reset_key = viewer.is_key_down("R")
                if viewer.reset_requested or (reset_key and not reset_held):
                    env.reset()
                    if sorter is not None:
                        sorter.reset()
                        sort_result = None
                    if director is not None:
                        director.reset()
                    script_step = 0
                    scripted_center = None
                    scripted_gap = args.pick_gap
                    scripted_opening = 0.08
                    viewer.aperture = 0.08
                    viewer.reset_requested = False
                    if controller is not None:
                        controller.reset()
                reset_held = reset_key
                if viewer.is_paused():
                    viewer.draw(step / 30)
                    time.sleep(1 / 30)
                    continue
            if controller is not None:
                action[0] = controller.advance()
            elif sorter is not None:
                arm = env.action_manager.get_term("arm_action")
                pos, quat = arm._compute_frame_pose()
                current = pos[0].cpu().numpy()
                target, aperture = sorter.command(script_step / 30, env.berries, current)
                # The viewer and the per-step metrics follow the berry being handled.
                env.berry = list(env.berries.values())[sorter.index]
                if viewer is not None:
                    viewer.status = sorter.phase
                    if director is not None:
                        director.update(viewer, sorter, env.berries, current, 1 / 30)
                    else:
                        viewer.follow(env.berry)
                action[0, :3] = torch.from_numpy(np.clip((target - current) * 0.6, -step_limit, step_limit))
                current_rot = Rotation.from_quat(quat[0].cpu().numpy())
                desired = Rotation.from_euler("x", np.pi)
                action[0, 3:6] = torch.from_numpy(
                    np.clip((desired * current_rot.inv()).as_rotvec() * 0.2, -turn_limit, turn_limit)
                )
                action[0, 6] = aperture / 0.04 - 1
            elif args.mode in ("pick", "place", "squash"):
                arm = env.action_manager.get_term("arm_action")
                pos, quat = arm._compute_frame_pose()
                t = scripted_motion_time(script_step / 30, args.mode, args.motion_speed, pauses)
                if scripted_center is None and t >= 1:
                    tissue = env.berry.positions()
                    scripted_center = (tissue.min(0) + tissue.max(0)) / 2 + env.berry.offset
                    if len(env.berries) > 1:
                        # Pre-shape above the punnet so open fingers miss adjacent berries.
                        scripted_opening = float(np.clip(np.ptp(tissue[:, 1]) + 0.008, 0.02, 0.08))
                    if scripted_gap is None:
                        scripted_gap = float(np.clip(np.ptp(tissue[:, 1]) * 0.7, 0.001, 0.08))
                target = np.array([0.48, 0, 0.09])
                if scripted_center is not None:
                    target[:2] = scripted_center[:2]
                # With the hand facing down, pad centers sit 3.6 mm above the TCP.
                grasp_z = max(0.005, float(scripted_center[2]) - 0.0036) if scripted_center is not None else 0.0055
                target[2] = 0.09 - (0.09 - grasp_z) * np.clip((t - 1) / 3, 0, 1)
                if args.mode == "pick":
                    target[2] += 0.05 * np.clip((t - 9) / 3, 0, 1)
                elif args.mode == "place":
                    target[2] += 0.09 * np.clip((t - 9) / 3, 0, 1)
                    target[:2] += (np.array(BOWL[:2]) - target[:2]) * np.clip((t - 13) / 6, 0, 1)
                    target[2] += (0.035 - grasp_z - 0.09) * np.clip((t - 20) / 3, 0, 1)
                    if t > 26:
                        target[2] += 0.08 * np.clip((t - 26) / 3, 0, 1)
                current = pos[0].cpu().numpy()
                action[0, :3] = torch.from_numpy(np.clip((target - current) * 0.6, -step_limit, step_limit))
                current_rot = Rotation.from_quat(quat[0].cpu().numpy())
                desired = Rotation.from_euler("x", np.pi)
                action[0, 3:6] = torch.from_numpy(
                    np.clip((desired * current_rot.inv()).as_rotvec() * 0.2, -turn_limit, turn_limit)
                )
                gap = (scripted_gap if scripted_gap is not None else 0.02) if args.mode in ("pick", "place") else 0.001
                aperture = scripted_opening + (gap - scripted_opening) * np.clip((t - 4) / 4, 0, 1)
                release = {"pick": 14, "place": 24, "squash": 12}[args.mode]
                if t > release:
                    aperture += (0.08 - gap) * np.clip((t - release) / 2, 0, 1)
                action[0, 6] = aperture / 0.04 - 1
            elif args.mode == "keyboard":
                action.zero_()
                for axis, pair in enumerate(("WS", "AD", "QE", "ZX", "TG", "CV")):
                    action[0, axis] = (0.0015 if axis < 3 else 0.02) * (
                        int(viewer.is_key_down(pair[0])) - int(viewer.is_key_down(pair[1]))
                    )
                viewer.aperture = float(
                    np.clip(
                        viewer.aperture + 0.012 / 30 * (int(viewer.is_key_down("J")) - int(viewer.is_key_down("K"))),
                        0,
                        0.08,
                    )
                )
                action[0, 6] = viewer.aperture / 0.04 - 1
            if args.mode in ("gamepad", "keyboard"):
                action[0, :6] *= args.motion_speed
            with profiling.zone("env: step"):
                env.step(action)
            script_step += 1
            simulated = time.perf_counter()
            if viewer is not None:
                with profiling.zone("viewer: draw", 0xFBBC04):
                    viewer.draw(step / 30)
            ended = time.perf_counter()
            timings.append(
                {
                    "physics_ms": (simulated - started) * 1000,
                    "frame_ms": (ended - started) * 1000,
                }
            )
            if video is not None:
                video.stdin.write(np.ascontiguousarray(viewer.capture_image()[..., :3]).tobytes())
            if (args.output or args.print_metrics) and (step % 30 == 0 or step == args.steps - 1):
                row = {
                    "step": step,
                    **env.berry.metrics(),
                    "aperture_command_m": float((action[0, 6] + 1) * 0.04),
                }
                row["tcp_m"] = env.action_manager.get_term("arm_action")._compute_frame_pose()[0][0].tolist()
                robot = env.scene["robot"]
                fingers = robot.data.body_pos_w.torch[0, robot.find_bodies("panda_(left|right)finger")[0]]
                row["finger_separation_m"] = float(torch.linalg.norm(fingers[0] - fingers[1]))
                if sorter is not None:
                    row["sort_phase"] = sorter.phase
                    row["sort_grip_adjustment_m"] = sorter.grip_adjustment
                    row["sort_holding"] = sorter.holding
                if args.output:
                    row["berries"] = {name: berry.metrics() for name, berry in env.berries.items()}
                    rows.append(row)
                if args.print_metrics:
                    print(json.dumps(row), flush=True)
            if (
                viewer is not None
                and args.output
                and step > 0
                and args.capture_every
                and (step % args.capture_every == 0 or step == args.steps - 1)
            ):
                args.output.mkdir(parents=True, exist_ok=True)
                viewer.save_screenshot(str(args.output / f"frame-{step:05d}.png"))
            if args.mode in ("keyboard", "gamepad"):
                time.sleep(max(0, 1 / 30 - (time.perf_counter() - started)))
            profiling.frame_mark()
            step += 1
        if sorter is not None and sorter.complete:
            sort_result = sorting_result(env.berries)
            print(f"Sorting {'passed' if sort_result['passed'] else 'failed'}", flush=True)
        if args.verify_render:
            render_verified = viewer.verify_geometry()
            print(
                "Native renderer verification passed: all dynamic arrays, opacity, and full SH3",
                flush=True,
            )
        if args.verify_reset:
            env.reset()
            for berry in env.berries.values():
                np.testing.assert_allclose(berry.positions(), berry.rest, atol=1e-6)
                np.testing.assert_array_equal(berry.velocities(), 0.0)
            np.testing.assert_allclose(
                env.action_manager.get_term("gripper_action").processed_actions.cpu().numpy(),
                0.04,
            )
            reset_verified = True
            print(
                "Reset verification passed: tissue positions, velocities and aperture",
                flush=True,
            )
    finally:
        if args.output:
            args.output.mkdir(parents=True, exist_ok=True)
            report = {
                "berry": args.berry,
                "target_berry": env.berry.profile["berry"],
                "berries": {
                    name: {
                        "asset": str(berry.usd_path),
                        "position_m": berry.offset.tolist(),
                        "material": {
                            "tissue_solver": args.tissue_solver,
                            "density_kg_m3": berry.spec.material.density,
                            "young_modulus_pa": berry.spec.material.young_modulus,
                            "poisson_ratio": berry.spec.material.poisson_ratio,
                        },
                        "physics_resolution": berry.resolution,
                        "voxel_size_m": berry.spec.voxel_size,
                        "gaussians": len(berry.asset["xyz"]),
                        "physical_particles": len(berry.rest),
                    }
                    for name, berry in env.berries.items()
                },
                "asset": str(env.berry.usd_path),
                "asset_version": None if args.asset else args.asset_version,
                "physics_resolution": args.physics_resolution,
                "scripted_pick_gap_m": scripted_gap,
                "mode": args.mode,
                "sorting_result": sort_result,
                "motion_speed": args.motion_speed,
                "loop": args.loop,
                "steps": step,
                "rendered": viewer is not None,
                "background": args.background,
                "background_asset": viewer.background_path if viewer is not None else None,
                "initial_view": args.view,
                "hide_interior": args.hide_interior,
                "sh_rotation_initial": args.sh_rotation,
                "sh_rotation_final": ("on" if viewer.sh_rotation else "off") if viewer is not None else None,
                "width": args.width,
                "height": args.height,
                "pipeline": not args.sync_render,
                "window": args.window or args.mode in ("gamepad", "keyboard"),
                "partitions": args.partitions,
                "rtpt_spp_override": args.rtpt_spp,
                "sampling_overrides": viewer.sampling_overrides if viewer is not None else {},
                "video": args.video,
                "timing_excludes_capture_and_pacing": True,
                "reset_verified": reset_verified,
                "render_verified": render_verified,
                "rows": rows,
                "timings": timings,
                "gaussians": sum(len(berry.asset["xyz"]) for berry in env.berries.values()),
                "physical_particles": sum(len(berry.rest) for berry in env.berries.values()),
            }
            if len(timings) > 10:
                steady = [r["frame_ms"] for r in timings[10:]]
                report["mean_frame_ms_after_warmup"] = float(np.mean(steady))
                report["p95_frame_ms_after_warmup"] = float(np.percentile(steady, 95))
                report["mean_compute_fps_after_warmup"] = 1000 / float(np.mean(steady))
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if video is not None:
            video.stdin.close()
            if video.wait() != 0:
                raise RuntimeError("ffmpeg failed; the video is incomplete")
