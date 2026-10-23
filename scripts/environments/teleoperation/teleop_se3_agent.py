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

"""Parse CLI first so we can decide whether to launch Isaac Sim Kit."""

# Isaac Lab does not use Warp autodiff; skipping adjoint codegen roughly halves the
# time spent building kernels on a cold kernel cache.
import warp as wp

wp.config.enable_backward = False

import argparse
import sys
from collections.abc import Callable
from contextlib import ExitStack

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils.string import string_to_callable

from isaaclab_tasks.utils import setup_preset_cli

# add argparse arguments
parser = argparse.ArgumentParser(description="Teleoperation for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default=None,
    help=(
        "Legacy teleop device name. When omitted, the Isaac Capture pipeline is used if configured in the env,"
        " otherwise keyboard is used as fallback. When explicitly provided, the script uses the legacy"
        " teleop_devices path and looks up this name in env_cfg.teleop_devices.devices."
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

# append simulation launcher cli args
add_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = setup_preset_cli(parser)

# Enable external camera rendering by default (``--disable_external_cameras`` turns it off). The
# ``--enable_cameras`` CLI flag was removed in Isaac Lab 3.0 (see #6656), so pass the intent to
# the launcher; this selects a camera-rendering experience that provides RTX/DLSS support.
# Everywhere else we read ``args_cli.disable_external_cameras`` directly.
args_cli.enable_cameras = not args_cli.disable_external_cameras
# the teleop input devices and the RTX settings use Kit APIs directly
args_cli.require_kit = True

# Call an external callback if requested.
if args_cli.external_callback:
    remaining = string_to_callable(args_cli.external_callback, separator=".")()
    if remaining is not None:
        _, hydra_args = setup_preset_cli(parser, remaining)
sys.argv = [sys.argv[0]] + hydra_args

"""Rest everything follows."""


import logging

import gymnasium as gym
import torch
from isaaclab_physx.renderers import IsaacRtxRendererGlobalSettingsCfg
from isaaclab_teleop import CLOUDXR_AVP_ENV, CLOUDXR_JS_ENV, CLOUDXR_STANDALONE_ENV

from isaaclab.devices import Se3GamepadCfg, Se3KeyboardCfg, Se3SpaceMouseCfg
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import instantiate

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.core.lift import mdp
from isaaclab_tasks.utils import resolve_task_config

logger = logging.getLogger(__name__)

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

    # When --teleop_device is explicitly provided, use the legacy teleop_devices path
    # even if isaac_teleop is configured. Otherwise prefer isaac_teleop when available.
    teleop_device_explicitly_set = args_cli.teleop_device is not None
    use_isaac_teleop = (
        not teleop_device_explicitly_set and hasattr(env_cfg, "isaac_teleop") and env_cfg.isaac_teleop is not None
    )

    # XR-rendering setup (camera removal + DLSS) is only needed for the Kit XR
    # path. Without --xr, Isaac Capture runs standalone (I/O only) and renders
    # normally, so gate on --xr alone.
    if args_cli.xr:
        # Keep camera configs when external cameras are enabled (defaulted on); otherwise
        # strip them so the XR headset view is the sole render product.
        if args_cli.disable_external_cameras:
            env_cfg = remove_camera_configs(env_cfg)

    with launch_simulation(env_cfg, args_cli), ExitStack() as cleanup:
        run_teleoperation(env_cfg, use_isaac_teleop, cleanup)


def run_teleoperation(env_cfg: ManagerBasedRLEnvCfg, use_isaac_teleop: bool, cleanup: ExitStack) -> None:  # noqa: C901
    """Create the environment and teleop device, then run the teleoperation loop until the app is closed."""
    from isaaclab_physx.renderers.isaac_rtx_renderer_utils import apply_isaac_rtx_global_settings
    from isaaclab_teleop import XrCameraFeedSession

    camera_feed_session = XrCameraFeedSession.prepare(
        env_cfg,
        enabled=args_cli.xr and use_isaac_teleop,
        camera_rendering_enabled=not args_cli.disable_external_cameras,
    )
    cleanup.callback(camera_feed_session.close)

    # Apply the RTX/DLSS global settings when an RTX render pipeline will run (Kit visualizer,
    # external cameras, or XR).
    if not args_cli.disable_external_cameras or args_cli.kit_visualizer or args_cli.xr:
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
        if use_isaac_teleop:
            from isaaclab_teleop import create_isaac_teleop_device, poll_control_events

            teleop_interface = create_isaac_teleop_device(
                env_cfg.isaac_teleop,
                sim_device=args_cli.device,
                callbacks=teleoperation_callbacks,
                cloudxr_env_file=args_cli.cloudxr_env,
                auto_launch_cloudxr=args_cli.auto_launch_cloudxr,
                enable_debug_visualization=args_cli.enable_debug_visualization,
                use_kit_xr_bridge=args_cli.xr,
                haptic_cfg=getattr(env_cfg, "haptic_feedback", None),
            )

        else:
            # Task-declared devices take precedence only when explicitly selected.
            if args_cli.teleop_device is not None and args_cli.teleop_device in env_cfg.teleop_devices.devices:
                teleop_interface = create_teleop_device(
                    args_cli.teleop_device, env_cfg.teleop_devices.devices, teleoperation_callbacks
                )
            else:
                device_name = "keyboard" if args_cli.teleop_device is None else args_cli.teleop_device.lower()
                device_cfgs = {
                    "keyboard": Se3KeyboardCfg(pos_sensitivity=0.05, rot_sensitivity=0.05),
                    "spacemouse": Se3SpaceMouseCfg(pos_sensitivity=0.05, rot_sensitivity=0.05),
                    "gamepad": Se3GamepadCfg(pos_sensitivity=0.1, rot_sensitivity=0.1),
                }
                if device_name not in device_cfgs:
                    logger.error(
                        f"--teleop_device={args_cli.teleop_device} was passed but no matching entry exists in"
                        " env_cfg.teleop_devices and it is not a built-in device name. Either remove"
                        " --teleop_device to use the Isaac Capture pipeline, or add a"
                        f" '{args_cli.teleop_device}' entry under teleop_devices in the environment config."
                        " Built-in devices: keyboard, spacemouse, gamepad."
                    )
                    env.close()
                    return
                device_cfg = device_cfgs[device_name]
                device_cfg.pos_sensitivity *= args_cli.sensitivity
                device_cfg.rot_sensitivity *= args_cli.sensitivity
                teleop_interface = instantiate(device_cfg)
                for key, callback in teleoperation_callbacks.items():
                    try:
                        teleop_interface.add_callback(key, callback)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Failed to add callback for key {key}: {e}")
    except Exception as e:
        logger.error(f"Failed to create teleop device: {e}")
        env.close()
        return

    print(f"Using teleop device: {teleop_interface}")

    haptic_driver = None
    if use_isaac_teleop:
        from isaaclab_teleop import create_haptic_feedback_driver

        haptic_driver = create_haptic_feedback_driver(env.unwrapped, teleop_interface, env_cfg)

    # Retain the keyboard so its input subscription lives for the whole session.
    if use_isaac_teleop and env.sim.has_gui and not env.sim.get_setting("/isaaclab/xr/auto_start"):
        try:
            control_keyboard = instantiate(Se3KeyboardCfg(pos_sensitivity=0.0, rot_sensitivity=0.0))
            control_keyboard.add_callback("B", teleop_interface.request_start)
            control_keyboard.add_callback("P", teleop_interface.request_stop)
            # Inject a reset pulse; the loop owns the environment reset.
            control_keyboard.add_callback("R", lambda: teleop_interface.reset(pause=True))
            print("Isaac Capture control keys: [B] start/resume  [P] pause  [R] reset")
        except Exception as e:
            logger.warning(f"Control keyboard unavailable ({e}); teleop still auto-starts without --xr")
            control_keyboard = None

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


if __name__ == "__main__":
    # run the main function
    main()
