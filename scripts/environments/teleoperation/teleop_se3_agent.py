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
2. IsaacTeleop-based stack (via isaac_teleop in env_cfg)

The script automatically detects which stack to use based on the environment config.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from collections.abc import Callable

from isaaclab.app import AppLauncher
from isaaclab.utils.string import list_intersection, string_to_callable

# add argparse arguments
parser = argparse.ArgumentParser(description="Teleoperation for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default=None,
    help=(
        "Legacy teleop device name. When omitted, the IsaacTeleop pipeline is used if configured in the env,"
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
    help="Enable hand joint and controller aim debug visualization at session start (IsaacTeleop only).",
)
parser.add_argument(
    "--external_callback",
    default=None,
    help="Fully qualified path to an externally defined callback.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, remaining_args = parser.parse_known_args()

app_launcher_args = vars(args_cli)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

# Call an external callback if requested.
remaining_args_env_registration = None
if args_cli.external_callback:
    external_callback_function = string_to_callable(args_cli.external_callback, separator=".")
    remaining_args_env_registration = external_callback_function()

# Error on unrecognized arguments.
unrecognized_args = list_intersection(remaining_args, remaining_args_env_registration)
if unrecognized_args:
    parser.error(f"unrecognized arguments: {' '.join(unrecognized_args)}")

"""Rest everything follows."""


import logging

import gymnasium as gym
import torch
from isaaclab_physx.renderers import IsaacRtxRendererGlobalSettingsCfg
from isaaclab_physx.renderers.isaac_rtx_renderer_utils import (
    apply_isaac_rtx_global_settings,
)

from isaaclab.devices import Se3Gamepad, Se3GamepadCfg, Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.core.lift import mdp
from isaaclab_tasks.utils import parse_env_cfg

logger = logging.getLogger(__name__)

_CLOUDXR_ENV_SHORTHANDS: dict[str, str] = {}


def _resolve_cloudxr_env(value: str | None, xr_enabled: bool = False) -> str | None:
    """Resolve ``--cloudxr_env`` shorthands to absolute ``.env`` file paths.

    Accepts ``"cloudxrjs"`` (Quest/Pico), ``"avp"`` (Apple Vision Pro),
    ``"standalone"`` (headless, no XR client), ``"none"`` (disable), or an
    arbitrary file path. When *value* is ``None`` (flag unset), defaults to
    ``"cloudxrjs"`` when *xr_enabled* else ``"standalone"`` -- so a run without
    ``--xr`` uses the clientless headless profile.
    """
    if value is None:
        value = "cloudxrjs" if xr_enabled else "standalone"
    if value.strip() == "" or value.lower() == "none":
        return None
    if not _CLOUDXR_ENV_SHORTHANDS:
        from isaaclab_teleop import CLOUDXR_AVP_ENV, CLOUDXR_JS_ENV, CLOUDXR_STANDALONE_ENV

        _CLOUDXR_ENV_SHORTHANDS["cloudxrjs"] = CLOUDXR_JS_ENV
        _CLOUDXR_ENV_SHORTHANDS["avp"] = CLOUDXR_AVP_ENV
        _CLOUDXR_ENV_SHORTHANDS["standalone"] = CLOUDXR_STANDALONE_ENV
    return _CLOUDXR_ENV_SHORTHANDS.get(value.lower(), value)


def _rtx_rendering_requested(args: argparse.Namespace) -> bool:
    """Return whether the CLI selects a renderer that actually drives RTX rendering.

    The RTX/DLSS global settings (and the ``omni.replicator`` extension they configure)
    are only meaningful when something renders through RTX. That happens when the Kit
    visualizer is enabled (``--viz kit``), when external cameras are rendered
    (``--enable_cameras``), or in XR mode (``--xr``, which drives the Kit XR pipeline).
    A pure-headless session selects none of these and renders nothing.

    This intentionally reads the CLI intent rather than any Kit/carb runtime state so the
    check keeps working as these scripts grow support for other renderers and kitless runs.
    """
    visualizers = getattr(args, "visualizer", None) or []
    return bool(getattr(args, "enable_cameras", False)) or ("kit" in visualizers) or bool(getattr(args, "xr", False))


def _create_builtin_device(device_name: str, sensitivity: float) -> object | None:
    """Create a built-in teleop device by name, or return None if unrecognized."""
    name = device_name.lower()
    if name == "keyboard":
        return Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity))
    elif name == "spacemouse":
        return Se3SpaceMouse(Se3SpaceMouseCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity))
    elif name == "gamepad":
        return Se3Gamepad(Se3GamepadCfg(pos_sensitivity=0.1 * sensitivity, rot_sensitivity=0.1 * sensitivity))
    return None


def _make_haptic_io(env, teleop_interface, env_cfg, use_isaac_teleop: bool):
    """Return ``(update, stop)`` callables driving controller haptics, or no-ops.

    Keeps haptics opt-in without branching in the main loop: both callables are
    no-ops unless the active device is an IsaacTeleop device and the env declares
    a ``haptic_feedback`` config. ``update`` renders the current contact force;
    ``stop`` zeroes it so a stale pulse does not persist while teleop is paused.
    """
    noop = lambda: None  # noqa: E731
    if not use_isaac_teleop:
        return noop, noop
    from isaaclab_teleop import create_haptic_feedback_driver

    driver = create_haptic_feedback_driver(env.unwrapped, teleop_interface, env_cfg)
    if driver is None:
        return noop, noop
    return driver.update, driver.stop


def _make_control_keyboard(teleop_interface, use_isaac_teleop: bool, headless: bool):
    """Create an optional keyboard for headset-free IsaacTeleop control.

    Binds ``B`` / ``P`` / ``R`` to start-resume / pause / reset so a user can drive
    the teleop state machine without an XR headset. Keys are captured through the Kit
    app window, so this returns ``None`` when running headless or when IsaacTeleop is
    not the active stack (a headless run still auto-starts teleop). ``R`` calls
    :meth:`~isaaclab_teleop.IsaacTeleopDevice.reset`, which injects a single RESET
    pulse that the loop's control-event handler turns into one environment reset
    (binding it straight to the reset callback would reset the env twice). The
    returned device must be kept referenced by the caller so its carb input
    subscription survives.
    """
    if not use_isaac_teleop or headless:
        return None
    try:
        keyboard = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.0, rot_sensitivity=0.0))
        keyboard.add_callback("B", teleop_interface.request_start)
        keyboard.add_callback("P", teleop_interface.request_stop)
        keyboard.add_callback("R", teleop_interface.reset)
        print("IsaacTeleop control keys: [B] start/resume  [P] pause  [R] reset")
        return keyboard
    except Exception as e:
        logger.warning(f"Control keyboard unavailable ({e}); teleop still auto-starts without --xr")
        return None


def _auto_start_teleop(teleop_interface, use_isaac_teleop: bool, xr: bool) -> None:
    """Start IsaacTeleop locally when no XR headset will send START.

    Without ``--xr`` there is no client to START the session, so drive its state
    machine to RUNNING via :meth:`~isaaclab_teleop.IsaacTeleopDevice.request_start`.
    The command flows through the normal state machine, so keyboard pause/resume
    still works afterwards. No-op with ``--xr`` or for non-IsaacTeleop devices.
    """
    if use_isaac_teleop and not xr:
        teleop_interface.request_start()


def main() -> None:  # noqa: C901
    """
    Run teleoperation with an Isaac Lab manipulation environment.

    Creates the environment, sets up teleoperation interfaces and callbacks,
    and runs the main simulation loop until the application is closed.

    Returns:
        None
    """
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
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
    # path. Without --xr, IsaacTeleop runs standalone (I/O only) and renders
    # normally, so gate on --xr alone.
    if args_cli.xr:
        env_cfg = remove_camera_configs(env_cfg)
    # Apply the RTX/DLSS global settings only when an RTX render pipeline will actually run
    # (Kit visualizer, external cameras, or XR). Applying them pulls in ``omni.replicator``,
    # which is not loaded in a pure-headless run (e.g. headless IsaacTeleop I/O), where a
    # ``ModuleNotFoundError`` would otherwise abort startup.
    if _rtx_rendering_requested(args_cli):
        apply_isaac_rtx_global_settings(
            IsaacRtxRendererGlobalSettingsCfg(antialiasing_mode="DLSS"),
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
        simulation_app.close()
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

    # For XR devices (hand tracking or IsaacTeleop), default to inactive. Without
    # --xr, teleop is started locally (see ``request_start`` below) rather than by
    # a headset, so it still begins running -- but it flows through the same state
    # machine, so keyboard/host pause/resume keeps working.
    if use_isaac_teleop or args_cli.xr:
        teleoperation_active = env_cfg.isaac_teleop.teleoperation_active_default if use_isaac_teleop else False
    else:
        # Always active for other devices
        teleoperation_active = True

    # Create teleop device based on configuration
    teleop_interface = None

    try:
        if use_isaac_teleop:
            from isaaclab_teleop import create_isaac_teleop_device, poll_control_events

            teleop_interface = create_isaac_teleop_device(
                env_cfg.isaac_teleop,
                sim_device=args_cli.device,
                callbacks=teleoperation_callbacks,
                cloudxr_env_file=_resolve_cloudxr_env(args_cli.cloudxr_env, args_cli.xr),
                auto_launch_cloudxr=args_cli.auto_launch_cloudxr,
                enable_debug_visualization=args_cli.enable_debug_visualization,
                use_kit_xr_bridge=args_cli.xr,
                haptic_cfg=getattr(env_cfg, "haptic_feedback", None),
            )

        elif teleop_device_explicitly_set:
            device_name = args_cli.teleop_device
            if hasattr(env_cfg, "teleop_devices") and device_name in env_cfg.teleop_devices.devices:
                teleop_interface = create_teleop_device(
                    device_name, env_cfg.teleop_devices.devices, teleoperation_callbacks
                )
            else:
                teleop_interface = _create_builtin_device(device_name, args_cli.sensitivity)
                if teleop_interface is None:
                    logger.error(
                        f"--teleop_device={device_name} was passed but no matching entry exists in"
                        " env_cfg.teleop_devices and it is not a built-in device name. Either remove"
                        " --teleop_device to use the IsaacTeleop pipeline, or add a"
                        f" '{device_name}' entry under teleop_devices in the environment config."
                        " Built-in devices: keyboard, spacemouse, gamepad."
                    )
                    env.close()
                    simulation_app.close()
                    return
                for key, callback in teleoperation_callbacks.items():
                    try:
                        teleop_interface.add_callback(key, callback)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Failed to add callback for key {key}: {e}")
        else:
            # No --teleop_device and no isaac_teleop: fall back to keyboard
            sensitivity = args_cli.sensitivity
            teleop_interface = Se3Keyboard(
                Se3KeyboardCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
            )
            for key, callback in teleoperation_callbacks.items():
                try:
                    teleop_interface.add_callback(key, callback)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to add callback for key {key}: {e}")
    except Exception as e:
        logger.error(f"Failed to create teleop device: {e}")
        env.close()
        simulation_app.close()
        return

    if teleop_interface is None:
        logger.error("Failed to create teleop interface")
        env.close()
        simulation_app.close()
        return

    print(f"Using teleop device: {teleop_interface}")

    # Optional controller haptics: no-ops unless the env declares a
    # ``haptic_feedback`` config and the device can render it (IsaacTeleop).
    haptic_update, haptic_stop = _make_haptic_io(env, teleop_interface, env_cfg, use_isaac_teleop)

    # Optional keyboard for headset-free IsaacTeleop control. Kept in a local so its
    # carb input subscription is not garbage-collected; a headless run auto-starts
    # (in ``run_loop``) without it.
    control_keyboard = _make_control_keyboard(teleop_interface, use_isaac_teleop, args_cli.headless)  # noqa: F841

    def run_loop():
        """Inner function to run the teleop loop with access to nonlocal variables."""
        nonlocal should_reset_recording_instance, teleoperation_active

        # reset environment
        env.reset()
        teleop_interface.reset()

        # Without --xr there is no headset to send START, so start locally; the
        # [B]/[P] keys can still pause/resume afterwards.
        _auto_start_teleop(teleop_interface, use_isaac_teleop, args_cli.xr)

        stack_name = "IsaacTeleop" if use_isaac_teleop else "native"
        print(f"{stack_name} teleoperation started. Press 'R' to reset the environment.")

        # simulate environment
        while simulation_app.is_running():
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

                    # action is None when IsaacTeleop session hasn't started yet
                    # (e.g. waiting for user to click "Start AR")
                    if action is None:
                        env.sim.render()
                        haptic_stop()
                    elif teleoperation_active:
                        # process actions
                        actions = action.repeat(env.num_envs, 1)
                        # apply actions
                        env.step(actions)
                        # render controller haptics from post-step contact forces
                        haptic_update()
                    else:
                        env.sim.render()
                        # not stepping: zero haptics so a paused grip stops buzzing
                        haptic_stop()

                    if should_reset_recording_instance:
                        env.reset()
                        teleop_interface.reset()
                        should_reset_recording_instance = False
                        print("Environment reset complete")
            except Exception as e:
                logger.error(f"Error during simulation step: {e}")
                break

    # Run the teleoperation loop
    # IsaacTeleop requires a context manager, native devices don't
    if use_isaac_teleop:
        with teleop_interface:
            run_loop()
    else:
        run_loop()

    # close the simulator
    env.close()
    print("Environment closed")


if __name__ == "__main__":
    # run the main function
    main()
    # env.close() already closes the USD stage via sim.clear_instance().
    # Pump the event loop so the viewport processes closure, then close the app.
    simulation_app.update()
    simulation_app.close()
