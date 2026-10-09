# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pick deformable raspberries with a Franka: teleoperate it, or watch the scripted demo.

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
    choices=["teleop_gamepad", "teleop_keyboard", "scripted_demo"],
    default="teleop_keyboard",
    help="Teleoperate the robot with a gamepad or the keyboard, or watch the scripted demo: crush one berry, place two",
)
parser.add_argument(
    "--num_berries", type=int, choices=[1, 2, 3], help="Raspberries in the punnet (default: 1; scripted_demo: 3)"
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
    choices=["punnet_and_bowl", "berry_closeup", "room", "scripted_camera"],
    default="punnet_and_bowl",
    help="Fixed views, a close-up that follows the handled berry, or (scripted_demo) a shot-by-shot scripted camera",
)
parser.add_argument(
    "--bowl_material", choices=["glass", "porcelain"], default="glass", help="Material of the receiving bowl"
)
parser.add_argument("--repeat", action="store_true", help="scripted_demo: reset and run it again when it ends")
parser.add_argument(
    "--save_frames",
    type=Path,
    metavar="DIR",
    help="scripted_demo: render offscreen and save each frame (1/30 s of simulated time) as DIR/NNNNN.png, then exit",
)
parser.add_argument("--width", type=int, default=1280, help="Image width [px]")
parser.add_argument("--height", type=int, default=720, help="Image height [px]")
parser.add_argument(
    "--samples_per_pixel", type=int, help="Path-tracing samples per pixel (the renderer's default if omitted)"
)
parser.add_argument("--f_stop", type=float, help="scripted_camera: camera f-number of the close shots' depth of field")
parser.add_argument("--gamepad_device", help="Linux joystick path, e.g. /dev/input/js0")
add_launcher_args(parser)
parser.set_defaults(device="cuda:0", visualizer=[], headless=True)
args = parser.parse_args()
scripted_demo = args.mode == "scripted_demo"
args.num_berries = args.num_berries or (3 if scripted_demo else 1)
if scripted_demo and args.num_berries != 3:
    parser.error("scripted_demo handles three berries: use --num_berries 3")
if not scripted_demo and (args.camera == "scripted_camera" or args.save_frames or args.repeat):
    parser.error("--camera scripted_camera, --save_frames and --repeat apply to --mode scripted_demo")
if args.save_frames and args.save_frames.exists() and any(args.save_frames.iterdir()):
    parser.error(f"{args.save_frames} is not empty; choose a new directory for the frames")
if not args.arm_speed > 0:
    parser.error("--arm_speed must be positive")
if args.mode == "teleop_gamepad" and not any(
    p.exists() for p in ([Path(args.gamepad_device)] if args.gamepad_device else Path("/dev/input").glob("js*"))
):
    parser.error("No Linux joystick found. Connect a controller or use --mode teleop_keyboard")

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
    from PIL import Image

    from isaaclab_tasks.contrib.franka_pick_berries.control.scripted_demo import (
        ScriptedDemo,
        demo_summary,
        evaluate_demo,
    )
    from isaaclab_tasks.contrib.franka_pick_berries.control.teleop_devices import (
        BerryGamepad,
        BerryGamepadCfg,
        keyboard_action,
        view_to_robot,
    )
    from isaaclab_tasks.contrib.franka_pick_berries.rendering.scripted_camera_demo import ScriptedCameraDemo
    from isaaclab_tasks.contrib.franka_pick_berries.rendering.viewer import BerryViewer

    env = gym.make("IsaacContrib-Pick-Berry-Franka-IK-Rel-Newton", cfg=cfg).unwrapped
    resources.callback(env.close)
    env.reset()
    viewer = BerryViewer(
        env,
        width=args.width,
        height=args.height,
        headless=args.save_frames is not None,
        samples_per_pixel=args.samples_per_pixel,
        camera=args.camera,
        f_stop=args.f_stop,
        bowl_material=args.bowl_material,
    )
    resources.callback(viewer.close)

    gamepad = demo = scripted_camera = None
    if args.mode == "teleop_gamepad":
        gamepad = BerryGamepad(BerryGamepadCfg(device=args.gamepad_device))
        resources.callback(gamepad.close)
        print(gamepad, flush=True)
        gamepad.add_callback("R", lambda: setattr(viewer, "reset_requested", True))
    elif scripted_demo and args.camera == "scripted_camera":
        # A film opens on the room before the sequence starts, and shows the crush and the first gentle grasp at the
        # gripper's original closing pace, which shows their deformation longer.
        demo = ScriptedDemo(args.arm_speed, start_delay=3.0, slow_closing=(0, 1))
        scripted_camera = ScriptedCameraDemo()
    elif scripted_demo:
        demo = ScriptedDemo(args.arm_speed)
    if args.save_frames:
        args.save_frames.mkdir(parents=True, exist_ok=True)

    def reset():
        env.reset()
        viewer.keyboard_aperture = 0.08
        for part in (demo, scripted_camera, gamepad):
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

        if demo is not None:
            if demo.complete and args.repeat:
                print(demo_summary(evaluate_demo(env.berries)), flush=True)
                reset()
                sequence_step = 0
            action = demo.action(env, sequence_step / 30)
            # The close-up and the scripted camera follow the berry being handled.
            env.handled_berry = list(env.berries.values())[demo.index]
            viewer.status_text = demo.phase
            if scripted_camera is not None:
                scripted_camera.update(viewer, demo, env.berries, demo.tcp, 1 / 30)
            else:
                viewer.set_handled_berry(env.handled_berry)
        else:
            action = gamepad.advance().reshape(1, -1) if gamepad is not None else keyboard_action(viewer)
            action = view_to_robot(action, viewer.camera.yaw)
            action[0, :6] *= args.arm_speed

        env.step(action)
        viewer.draw(step / 30)
        step += 1
        sequence_step += 1

        if args.save_frames:
            Image.fromarray(viewer.capture_image()[..., :3]).save(args.save_frames / f"{step:05d}.png")
            # A film ends with the scripted camera's last shot; a plain recording with the sequence.
            if scripted_camera.finished if scripted_camera is not None else demo.complete:
                break
        elif demo is None:
            # Teleoperation runs in real time when the machine keeps up: 30 control steps per second.
            time.sleep(max(0.0, 1 / 30 - (time.perf_counter() - started)))

    if demo is not None and demo.complete:
        print(demo_summary(evaluate_demo(env.berries)), flush=True)
