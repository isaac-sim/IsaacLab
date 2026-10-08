# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pick deformable raspberries with a Franka: teleoperate it, or watch it sort three berries.

The berries' tissue is simulated with the material point method (MPM) on Newton, and rendered as 3D Gaussians that
follow it. See the task README for the commands of the demo and of its video.
"""

import argparse
import time
from contextlib import ExitStack
from pathlib import Path

import warp as wp

wp.config.enable_backward = False

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument(
    "--mode",
    choices=["teleop_gamepad", "teleop_keyboard", "scripted_sorting"],
    default="teleop_keyboard",
    help="Teleoperate the robot with a gamepad or the keyboard, or watch it sort three berries by itself",
)
parser.add_argument(
    "--num_berries", type=int, choices=[1, 2, 3], help="Raspberries in the punnet (default: 1; scripted_sorting: 3)"
)
parser.add_argument("--layout_seed", type=int, default=0, help="Seed of the random punnet layout; reset replays it")
parser.add_argument("--fixed_layout", action="store_true", help="Place several berries side by side, unrotated")
parser.add_argument(
    "--tissue_resolution",
    choices=["full", "half"],
    default="full",
    help="Tissue particles: all, or half for speed; the Gaussians are unchanged",
)
parser.add_argument(
    "--tissue_solver",
    choices=["explicit", "implicit"],
    default="explicit",
    help="Explicit MLS-MPM with frictional finger pads, or Newton's implicit MPM with clamped grasping",
)
parser.add_argument(
    "--arm_speed", type=float, default=2.0, help="Arm speed multiplier; the gripper keeps its gentle closing pace"
)
parser.add_argument(
    "--camera",
    choices=["punnet_and_bowl", "berry_closeup", "room", "film_director"],
    default="punnet_and_bowl",
    help="Fixed views, a close-up that follows the handled berry, or (scripted_sorting) a shot-by-shot film director",
)
parser.add_argument(
    "--bowl_material", choices=["glass", "porcelain"], default="glass", help="Material of the receiving bowl"
)
parser.add_argument("--repeat", action="store_true", help="scripted_sorting: reset and sort again when it ends")
parser.add_argument(
    "--video", type=Path, help="scripted_sorting: record this MP4 file offscreen at 30 frames/s, then exit"
)
parser.add_argument("--width", type=int, default=1280, help="Image width [px]")
parser.add_argument("--height", type=int, default=720, help="Image height [px]")
parser.add_argument(
    "--samples_per_pixel", type=int, help="Path-tracing samples per pixel (the renderer's default if omitted)"
)
parser.add_argument("--f_stop", type=float, help="film_director: camera f-number of the close shots' depth of field")
parser.add_argument("--gamepad_device", help="Linux joystick path, e.g. /dev/input/js0")
add_launcher_args(parser)
parser.set_defaults(device="cuda:0", visualizer=[], headless=True)
args = parser.parse_args()
sorting = args.mode == "scripted_sorting"
args.num_berries = args.num_berries or (3 if sorting else 1)
if sorting and args.num_berries != 3:
    parser.error("scripted_sorting sorts three berries: use --num_berries 3")
if not sorting and (args.camera == "film_director" or args.video or args.repeat):
    parser.error("--camera film_director, --video and --repeat apply to --mode scripted_sorting")
if not args.arm_speed > 0:
    parser.error("--arm_speed must be positive")
if args.mode == "teleop_gamepad" and not any(
    p.exists() for p in ([Path(args.gamepad_device)] if args.gamepad_device else Path("/dev/input").glob("js*"))
):
    parser.error("No Linux joystick found. Connect a controller or use --mode teleop_keyboard")

from isaaclab_tasks.contrib.franka_pick_berries.rendering.renderer_version import require_live_gaussian_renderer  # noqa

require_live_gaussian_renderer()

from isaaclab_tasks.contrib.franka_pick_berries.pick_berries_env_cfg import BerryPickEnvCfg  # noqa: E402

cfg = BerryPickEnvCfg()
cfg.num_berries = args.num_berries
cfg.randomize_layout = not args.fixed_layout
cfg.layout_seed = args.layout_seed
cfg.tissue_resolution = args.tissue_resolution
cfg.tissue_solver = args.tissue_solver
cfg.sim.device = args.device

with launch_simulation(cfg, args), ExitStack() as resources:
    import gymnasium as gym

    from isaaclab_tasks.contrib.franka_pick_berries.control.gamepad import BerryGamepad, BerryGamepadCfg
    from isaaclab_tasks.contrib.franka_pick_berries.control.keyboard import keyboard_action
    from isaaclab_tasks.contrib.franka_pick_berries.control.sorting_sequence import (
        SortingSequence,
        evaluate_sorting,
        sorting_summary,
    )
    from isaaclab_tasks.contrib.franka_pick_berries.rendering.camera_director import SortingCameraDirector
    from isaaclab_tasks.contrib.franka_pick_berries.rendering.video_writer import VideoWriter
    from isaaclab_tasks.contrib.franka_pick_berries.rendering.viewer import BerryViewer

    env = gym.make("IsaacContrib-Pick-Berry-Franka-IK-Rel-Newton", cfg=cfg).unwrapped
    resources.callback(env.close)
    env.reset()
    viewer = BerryViewer(
        env,
        width=args.width,
        height=args.height,
        headless=args.video is not None,
        samples_per_pixel=args.samples_per_pixel,
        camera=args.camera,
        f_stop=args.f_stop,
        bowl_material=args.bowl_material,
    )
    resources.callback(viewer.close)

    gamepad = sorter = director = video = None
    if args.mode == "teleop_gamepad":
        gamepad = BerryGamepad(BerryGamepadCfg(device=args.gamepad_device))
        resources.callback(gamepad.close)
        print(gamepad, flush=True)
        gamepad.add_callback("R", lambda: setattr(viewer, "reset_requested", True))
    elif sorting and args.camera == "film_director":
        # A film opens on the room before the sequence starts, and shows the crush and the first gentle grasp at the
        # gripper's original closing pace, which shows their deformation longer.
        sorter = SortingSequence(args.arm_speed, start_delay=3.0, slow_closing=(0, 1))
        director = SortingCameraDirector()
    elif sorting:
        sorter = SortingSequence(args.arm_speed)
    if args.video:
        video = VideoWriter(args.video, args.width, args.height)
        resources.callback(video.close)

    def reset():
        env.reset()
        viewer.keyboard_aperture = 0.08
        for part in (sorter, director, gamepad):
            if part is not None:
                part.reset()

    # One control step is 1/30 s of simulated time: four 120 Hz physics steps.
    step = sequence_step = 0
    while viewer.is_running():
        started = time.perf_counter()
        if viewer.reset_requested:
            viewer.reset_requested = False
            reset()
            sequence_step = 0
        if viewer.is_paused():
            viewer.draw(step / 30)
            time.sleep(1 / 30)
            continue

        if sorter is not None:
            if sorter.complete and args.repeat:
                print(sorting_summary(evaluate_sorting(env.berries)), flush=True)
                reset()
                sequence_step = 0
            action = sorter.action(env, sequence_step / 30)
            # The close-up and the director follow the berry being handled.
            env.handled_berry = list(env.berries.values())[sorter.index]
            viewer.status_text = sorter.phase
            if director is not None:
                director.update(viewer, sorter, env.berries, sorter.tcp, 1 / 30)
            else:
                viewer.set_handled_berry(env.handled_berry)
        else:
            action = gamepad.advance() if gamepad is not None else keyboard_action(viewer)
            action[0, :6] *= args.arm_speed

        env.step(action)
        viewer.draw(step / 30)
        step += 1
        sequence_step += 1

        if video is not None:
            video.write(viewer.capture_image())
            # A film ends with the director's last shot; a plain recording with the sequence.
            if director.finished if director is not None else sorter.complete:
                break
        elif sorter is None:
            # Teleoperation runs in real time when the machine keeps up: 30 control steps per second.
            time.sleep(max(0.0, 1 / 30 - (time.perf_counter() - started)))

    if sorter is not None and sorter.complete:
        print(sorting_summary(evaluate_sorting(env.berries)), flush=True)
