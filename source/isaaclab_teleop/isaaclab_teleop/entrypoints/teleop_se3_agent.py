# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run teleoperation with Isaac Lab manipulation environments.

Supports multiple input devices (e.g., keyboard, spacemouse, gamepad) and devices
configured within the environment (including OpenXR-based hand tracking or motion
controllers).

This script supports two teleoperation stacks:
1. Native Isaac Lab teleop stack (via teleop_devices in env_cfg)
2. Isaac Capture-based stack (via isaac_teleop in env_cfg)

The script automatically detects which stack to use based on the environment config.
"""

import argparse
import logging
import sys
from collections.abc import Callable
from contextlib import ExitStack

import gymnasium as gym
import torch
from isaaclab_physx.renderers import IsaacRtxRendererGlobalSettingsCfg

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.cli import load_external_tasks
from isaaclab.devices import DeviceCfg
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.string import string_to_callable
from isaaclab.utils.version import has_kit

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.core.lift import mdp
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

from .. import (
    CLOUDXR_AVP_ENV,
    CLOUDXR_JS_ENV,
    CLOUDXR_STANDALONE_ENV,
    IsaacTeleopCfg,
    se3_gamepad_teleop_cfg,
    se3_keyboard_teleop_cfg,
    se3_spacemouse_teleop_cfg,
)
from ..device_selection import create_teleop_device, select_teleop_cfg

logger = logging.getLogger(__name__)

args_cli: argparse.Namespace
"""Parsed command line, set by :func:`_parse_args`."""


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser, including the simulation launcher arguments."""
    parser = argparse.ArgumentParser(description="Teleoperation for Isaac Lab environments.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
    parser.add_argument(
        "--teleop_device",
        type=str,
        default=None,
        help=(
            "Teleop device name: an entry of env_cfg.teleop_devices.devices, or a built-in device (keyboard,"
            " spacemouse, gamepad). When omitted, the Isaac Capture pipeline is used if configured in the env,"
            " otherwise keyboard. Keyboard, gamepad and spacemouse run through Isaac Capture (the teleop extra)."
        ),
    )
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
    parser.add_argument(
        "--cloudxr_env",
        type=str,
        default=None,
        help=(
            "Path to a CloudXR .env file, or a shorthand: 'cloudxrjs' (Quest/Pico), 'avp' (Apple Vision Pro),"
            " or 'standalone' (headless, no XR client). Set to 'none' to disable CloudXR auto-launch entirely."
            " When unset, defaults to 'cloudxrjs' with --xr and 'standalone' without --xr."
        ),
    )
    parser.add_argument(
        "--auto_launch_cloudxr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-launch the CloudXR runtime when --cloudxr_env is set. Use --no-auto_launch_cloudxr to disable.",
    )
    parser.add_argument(
        "--enable_debug_visualization",
        action="store_true",
        default=False,
        help="Enable hand joint and controller aim debug visualization at session start (Isaac Capture only).",
    )
    parser.add_argument(
        "--external_callback",
        default=None,
        help="Fully qualified path to an externally defined callback.",
    )
    parser.add_argument(
        "--disable_external_cameras",
        action="store_true",
        default=False,
        help=(
            "Disable external camera rendering. External cameras render by default for teleoperation;"
            " pass this flag to strip camera sensors from the environment (e.g. to reduce GPU contention"
            " and improve XR performance)."
        ),
    )

    add_launcher_args(parser)
    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse ``argv`` (default ``sys.argv[1:]``) into :data:`args_cli`; Hydra task overrides stay in ``sys.argv``."""
    global args_cli
    if argv is not None:
        sys.argv = [sys.argv[0], *argv]
    parser = _build_parser()
    args_cli, hydra_args = setup_preset_cli(parser)

    # Enable external camera rendering by default (``--disable_external_cameras`` turns it off). The
    # ``--enable_cameras`` CLI flag was removed in Isaac Lab 3.0 (see #6656), so pass the intent to
    # the launcher; this selects a camera-rendering experience that provides RTX/DLSS support.
    # Everywhere else we read ``args_cli.disable_external_cameras`` directly.
    args_cli.enable_cameras = not args_cli.disable_external_cameras

    # Call an external callback if requested.
    if args_cli.external_callback:
        remaining = string_to_callable(args_cli.external_callback, separator=".")()
        if remaining is not None:
            _, hydra_args = setup_preset_cli(parser, remaining)
    sys.argv = [sys.argv[0]] + hydra_args

    # Resolve CLI profile names once; runtime construction receives a path or None.
    if args_cli.cloudxr_env is None:
        args_cli.cloudxr_env = "cloudxrjs" if args_cli.xr else "standalone"
    if not args_cli.cloudxr_env.strip() or args_cli.cloudxr_env.lower() == "none":
        args_cli.cloudxr_env = None
    else:
        args_cli.cloudxr_env = {
            "cloudxrjs": CLOUDXR_JS_ENV,
            "avp": CLOUDXR_AVP_ENV,
            "standalone": CLOUDXR_STANDALONE_ENV,
        }.get(args_cli.cloudxr_env.lower(), args_cli.cloudxr_env)
    return args_cli


def main() -> None:  # noqa: C901
    """
    Run teleoperation with an Isaac Lab manipulation environment.

    Creates the environment, sets up teleoperation interfaces and callbacks,
    and runs the main simulation loop until the application is closed.

    Returns:
        None
    """
    # Resolve the task configuration through Hydra so CLI presets are applied.
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.env_name = args_cli.task
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise ValueError(
            "Teleoperation is only supported for ManagerBasedRLEnv environments. "
            f"Received environment config type: {type(env_cfg).__name__}"
        )
    # modify configuration
    env_cfg.terminations.time_out = None
    if "Lift" in args_cli.task:
        # set the resampling time range to large number to avoid resampling
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        # add termination condition for reaching the goal otherwise the environment won't reset
        env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)

    teleop_cfg = select_teleop_cfg(
        env_cfg,
        args_cli.teleop_device,
        {
            "keyboard": se3_keyboard_teleop_cfg(
                pos_sensitivity=0.05 * args_cli.sensitivity,
                rot_sensitivity=0.05 * args_cli.sensitivity,
                sim_device=args_cli.device,
            ),
            "spacemouse": se3_spacemouse_teleop_cfg(
                pos_sensitivity=0.05 * args_cli.sensitivity,
                rot_sensitivity=0.05 * args_cli.sensitivity,
                sim_device=args_cli.device,
            ),
            "gamepad": se3_gamepad_teleop_cfg(
                pos_sensitivity=0.1 * args_cli.sensitivity,
                rot_sensitivity=0.1 * args_cli.sensitivity,
                sim_device=args_cli.device,
            ),
        },
        args_cli.device,
    )
    if isinstance(teleop_cfg, IsaacTeleopCfg):
        env_cfg.isaac_teleop = teleop_cfg

    # XR-rendering setup (camera removal + DLSS) is only needed for the Kit XR
    # path. Without --xr, Isaac Capture runs standalone (I/O only) and renders
    # normally, so gate on --xr alone.
    if args_cli.xr:
        # Keep camera configs when external cameras are enabled (defaulted on); otherwise
        # strip them so the XR headset view is the sole render product.
        if args_cli.disable_external_cameras:
            env_cfg = remove_camera_configs(env_cfg)

    with launch_simulation(env_cfg, args_cli), ExitStack() as cleanup:
        run_teleoperation(env_cfg, teleop_cfg, cleanup)


def run_teleoperation(
    env_cfg: ManagerBasedRLEnvCfg, teleop_cfg: IsaacTeleopCfg | DeviceCfg, cleanup: ExitStack
) -> None:  # noqa: C901
    """Create the environment and teleop device, then run the teleoperation loop until the app is closed."""
    from .. import XrCameraFeedSession, poll_control_events

    use_isaac_teleop = isinstance(teleop_cfg, IsaacTeleopCfg)

    camera_feed_session = XrCameraFeedSession.prepare(
        env_cfg,
        enabled=args_cli.xr and use_isaac_teleop,
        camera_rendering_enabled=not args_cli.disable_external_cameras,
    )
    cleanup.callback(camera_feed_session.close)

    # Apply the RTX/DLSS global settings when Kit runs an RTX render pipeline (Kit visualizer,
    # external cameras, or XR).
    if has_kit() and (not args_cli.disable_external_cameras or args_cli.kit_visualizer or args_cli.xr):
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import apply_isaac_rtx_global_settings

        apply_isaac_rtx_global_settings(
            IsaacRtxRendererGlobalSettingsCfg(
                antialiasing_mode="DLSS",
                carb_settings=(
                    {"/rtx/dldenoiser/responsiveDenoising": True}
                    if camera_feed_session.requires_responsive_denoising
                    else None
                ),
            ),
        )

    try:
        # create environment
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        # check environment name (for reach , we don't allow the gripper)
        if "Reach" in args_cli.task:
            logger.warning(
                f"The environment '{args_cli.task}' does not support gripper control. The device command will be"
                " ignored."
            )
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        return

    # Flags for controlling teleoperation flow
    should_reset_recording_instance = False
    teleoperation_active = True

    # Callback handlers
    def reset_recording_instance() -> None:
        """
        Reset the environment to its initial state.

        Sets a flag to reset the environment on the next simulation step.

        Returns:
            None
        """
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("Reset triggered - Environment will reset on next step")

    def start_teleoperation() -> None:
        """
        Activate teleoperation control of the robot.

        Enables the application of teleoperation commands to the environment.

        Returns:
            None
        """
        nonlocal teleoperation_active
        teleoperation_active = True
        print("Teleoperation activated")

    def stop_teleoperation() -> None:
        """
        Deactivate teleoperation control of the robot.

        Disables the application of teleoperation commands to the environment.

        Returns:
            None
        """
        nonlocal teleoperation_active
        teleoperation_active = False
        print("Teleoperation deactivated")

    # Create device config if not already in env_cfg
    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "R": reset_recording_instance,
        "START": start_teleoperation,
        "STOP": stop_teleoperation,
        "RESET": reset_recording_instance,
    }

    # For XR devices (hand tracking or Isaac Capture), default to inactive. Without
    # --xr, teleop is started locally (see ``request_start`` below) rather than by
    # a headset, so it still begins running -- but it flows through the same state
    # machine, so keyboard/host pause/resume keeps working.
    if use_isaac_teleop or args_cli.xr:
        teleoperation_active = env_cfg.isaac_teleop.teleoperation_active_default if use_isaac_teleop else False
    else:
        # Always active for other devices
        teleoperation_active = True

    try:
        teleop_interface = create_teleop_device(
            teleop_cfg,
            env_prim_path=env.scene.env_prim_paths[0],
            sim_device=args_cli.device,
            callbacks=teleoperation_callbacks,
            visualizers=env.sim.visualizers,
            cloudxr_env_file=args_cli.cloudxr_env,
            auto_launch_cloudxr=args_cli.auto_launch_cloudxr,
            enable_debug_visualization=args_cli.enable_debug_visualization,
            use_kit_xr_bridge=args_cli.xr,
            haptic_cfg=getattr(env_cfg, "haptic_feedback", None),
        )
    except Exception as e:
        logger.error(f"Failed to create teleop device: {e}")
        env.close()
        return

    print(f"Using teleop device: {teleop_interface}")

    haptic_driver = None
    if use_isaac_teleop:
        from .. import create_haptic_feedback_driver

        haptic_driver = create_haptic_feedback_driver(env.unwrapped, teleop_interface, env_cfg)

    if use_isaac_teleop:
        # the device reads these from the focused visualizer window
        print("Isaac Capture control keys: [B] start/resume  [P] pause  [R] reset")

    def run_loop():
        """Inner function to run the teleop loop with access to nonlocal variables."""
        nonlocal should_reset_recording_instance, teleoperation_active

        # reset environment
        env.reset()
        teleop_interface.reset()

        # Without --xr there is no headset to send START, so start locally ([B]/[P] can
        # still pause/resume). The reset() above is a host reset (a pure pulse), so it does
        # not cancel this start.
        if use_isaac_teleop and not args_cli.xr:
            teleop_interface.request_start()

        stack_name = "Isaac Capture" if use_isaac_teleop else "native"
        print(f"{stack_name} teleoperation started. Press 'R' to reset the environment.")

        # simulate environment
        while env.sim.is_running():
            try:
                # run everything in inference mode
                with torch.inference_mode():
                    # get device command
                    action = teleop_interface.advance()

                    if use_isaac_teleop:
                        ctrl = poll_control_events(teleop_interface)
                        if ctrl.is_active is not None:
                            teleoperation_active = ctrl.is_active
                        if ctrl.should_reset:
                            should_reset_recording_instance = True

                    # action is None when Isaac Capture session hasn't started yet
                    # (e.g. waiting for user to click "Start AR")
                    if action is None or not teleoperation_active:
                        env.sim.render()
                        if haptic_driver is not None:
                            haptic_driver.stop()
                    else:
                        # process actions
                        actions = action.repeat(env.num_envs, 1)
                        # apply actions
                        env.step(actions)
                        # render controller haptics from post-step contact forces
                        if haptic_driver is not None:
                            haptic_driver.update()

                    if should_reset_recording_instance:
                        env.reset()
                        teleop_interface.reset()
                        camera_feed_session.refresh()
                        should_reset_recording_instance = False
                        print("Environment reset complete")
            except Exception as e:
                logger.error(f"Error during simulation step: {e}")
                break

    # Run the teleoperation loop
    # Isaac Capture requires a context manager, native devices don't
    if use_isaac_teleop:
        with teleop_interface, camera_feed_session.bind(env):
            run_loop()
    else:
        with camera_feed_session.bind(env):
            run_loop()

    # close the simulator
    env.close()
    print("Environment closed")


def run_teleop_se3_agent_cli(argv: list[str] | None = None) -> int:
    """Run interactive teleoperation from the command line.

    Args:
        argv: Command-line arguments, excluding the program name. Defaults to ``sys.argv[1:]``.
            ``sys.argv`` is restored on return.

    Returns:
        The process exit code, 0 once the session ends.
    """
    # tasks from installed downstream projects, including for the script and ``isaaclab teleop``
    load_external_tasks()
    original_argv = sys.argv
    try:
        _parse_args(argv)
        main()
    finally:
        sys.argv = original_argv
    return 0
