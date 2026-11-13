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
import math
import signal
import sys
import time
from collections.abc import Callable

import gymnasium as gym
import torch
from isaaclab_physx.renderers import IsaacRtxRendererGlobalSettingsCfg

from isaaclab.app import add_launcher_args, launch_simulation, validate_launch
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
    OvLibrariesXrCameraCfg,
    se3_gamepad_teleop_cfg,
    se3_keyboard_teleop_cfg,
    se3_spacemouse_teleop_cfg,
)
from ..action_policy import initial_hand_tracking_issue, prepare_teleop_action
from ..device_selection import create_teleop_device, select_teleop_cfg
from ._ovxr_launch import (
    add_ovxr_presentation_args,
    prepare_ovxr_launch,
    reject_ovxr_runtime_conflicts,
    resolve_ovxr_pip_cameras,
    validate_ovxr_presentation_args,
)
from ._ovxr_presentation import OvxrPresentation, admit_reset_publication, update_idle_presentation

logger = logging.getLogger(__name__)

_RENDERER_ATTACH_TIMEOUT_S = 180.0
"""Allow cold RTX preparation before failing attachment; client connection waits separately."""

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
        "--step_hz",
        type=float,
        default=0.0,
        help="Maximum host-control rate with --ovxr; 0 leaves active control unthrottled. XR rendering is independent.",
    )
    parser.add_argument(
        "--cloudxr_env",
        type=str,
        default=None,
        help=(
            "Path to a CloudXR .env file, or a shorthand: 'cloudxrjs' (Quest/Pico), 'avp' (Apple Vision Pro),"
            " or 'standalone' (headless, no XR client). Set to 'none' to disable CloudXR auto-launch entirely."
            " When unset, defaults to 'cloudxrjs' with --xr or --ovxr and 'standalone' otherwise."
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
    parser.add_argument(
        "--ovxr",
        dest="using_ovxr",
        action="store_true",
        default=False,
        help="Use the OVXR OVRTX/OVStage/OVXR path instead of Kit XR.",
    )
    parser.add_argument(
        "--ovxr_session_mode",
        choices=("ar", "vr"),
        default="ar",
        help="OVXR session mode. Defaults to 'ar'; pass 'vr' for an opaque background.",
    )
    parser.add_argument(
        "--ovrtx_active_cuda_gpus",
        type=_cuda_index_list,
        default=None,
        help="Comma-separated CUDA indices for OVRTX rendering, e.g. 0. Does not select physics or camera outputs.",
    )
    parser.add_argument(
        "--geometry_publication_device",
        type=_cuda_device,
        default=None,
        metavar="CUDA_DEVICE",
        help=(
            "Destination CUDA device for completed OVStage geometry publication, e.g. cuda:0. "
            "Adds copy work for geometry on another GPU; unset preserves renderer configuration. "
            "Requires --ovxr; does not select physics or rendering placement."
        ),
    )
    add_ovxr_presentation_args(parser)

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
    if args_cli.xr and args_cli.using_ovxr:
        parser.error("--xr and --ovxr are mutually exclusive OpenXR frame-loop owners")
    if not math.isfinite(args_cli.step_hz) or args_cli.step_hz < 0.0:
        parser.error("--step_hz must be finite and non-negative; 0 disables active-control pacing")
    if args_cli.step_hz > 0.0 and not args_cli.using_ovxr:
        parser.error("--step_hz requires --ovxr")
    if not args_cli.using_ovxr and (
        args_cli.geometry_publication_device is not None or args_cli.ovrtx_active_cuda_gpus is not None
    ):
        parser.error("--geometry_publication_device and --ovrtx_active_cuda_gpus require --ovxr")
    validate_ovxr_presentation_args(parser, args_cli, using_ovxr=args_cli.using_ovxr)

    # Call an external callback if requested.
    if args_cli.external_callback:
        remaining = string_to_callable(args_cli.external_callback, separator=".")()
        if remaining is not None:
            _, hydra_args = setup_preset_cli(parser, remaining)
    sys.argv = [sys.argv[0]] + hydra_args

    # Resolve CLI profile names once; runtime construction receives a path or None.
    if args_cli.cloudxr_env is None:
        args_cli.cloudxr_env = "cloudxrjs" if args_cli.xr or args_cli.using_ovxr else "standalone"
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
    env_cfg, teleop_cfg, use_isaac_teleop, selected_cameras = _resolve_env_cfg()
    with launch_simulation(env_cfg, args_cli, cleanup_context=not use_isaac_teleop):
        run_teleoperation(env_cfg, teleop_cfg, use_isaac_teleop, selected_cameras)


def _resolve_env_cfg() -> tuple[
    ManagerBasedRLEnvCfg, IsaacTeleopCfg | DeviceCfg, bool, tuple[OvLibrariesXrCameraCfg, ...]
]:
    """Resolve task configuration and teleop ownership before runtime launch."""
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

    use_isaac_teleop = isinstance(teleop_cfg, IsaacTeleopCfg)
    if args_cli.using_ovxr and not use_isaac_teleop:
        raise ValueError("--ovxr requires a task configured with env_cfg.isaac_teleop")
    # XR-rendering setup (camera removal + DLSS) is only needed for the Kit XR
    # path. Without --xr, Isaac Capture runs standalone (I/O only) and renders
    # normally, so gate on --xr alone.
    if args_cli.xr:
        # Keep camera configs when external cameras are enabled (defaulted on); otherwise
        # strip them so the XR headset view is the sole render product.
        if args_cli.disable_external_cameras:
            env_cfg = remove_camera_configs(env_cfg)
    selected_cameras = ()
    if args_cli.using_ovxr:
        # Reject an inconsistent GPU placement before launch.
        renderer_cfg = _ovxr_renderer_cfg(env_cfg.sim.device)
        if args_cli.ovxr_pip_camera is not None:
            # Resolve the panels before launch preparation removes the other cameras, so that an
            # invalid selection or task layout fails before launch.
            selected_cameras = resolve_ovxr_pip_cameras(env_cfg, args_cli, renderer_cfg)
        env_cfg = prepare_ovxr_launch(
            env_cfg, args_cli.task, args_cli.num_envs, keep=tuple(args_cli.ovxr_pip_camera or ())
        )
    # A task without a physics selection keeps None: SimulationContext defaults it to PhysxCfg.
    configs, kit_sources = validate_launch(env_cfg, vars(args_cli).copy())
    if args_cli.using_ovxr:
        reject_ovxr_runtime_conflicts(configs, kit_sources)
    if use_isaac_teleop and teleop_cfg.prepare_environment is not None:
        teleop_cfg.prepare_environment(env_cfg)
    return env_cfg, teleop_cfg, use_isaac_teleop, selected_cameras


def _pace_ovxr_control_loop(
    using_ovxr: bool,
    action_applied: bool,
    *,
    iteration_started_s: float,
    action_period_s: float,
    step_hz: float = 0.0,
    wait_for_shutdown: Callable[[float], bool] | None = None,
) -> None:
    """Limit host control independently of the XR frame clock.

    Credit elapsed input, physics and publication work; an overrun adds no wait
    or catch-up steps. With no explicit limit, active control remains unpaced
    and paused polling retains the environment's action interval.
    """
    if not using_ovxr or (action_applied and step_hz == 0.0):
        return
    period_s = 1.0 / step_hz if step_hz > 0.0 else action_period_s
    remaining_s = period_s - (time.perf_counter() - iteration_started_s)
    if remaining_s > 0.0:
        if wait_for_shutdown is None:
            time.sleep(remaining_s)
        else:
            wait_for_shutdown(remaining_s)


def _cuda_index_list(value: str) -> str:
    """Validate ``--ovrtx_active_cuda_gpus`` at parse time."""
    from isaaclab_ov.renderers import parse_active_cuda_gpus

    try:
        parse_active_cuda_gpus(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return value


def _cuda_device(value: str) -> str:
    """Validate ``--geometry_publication_device`` at parse time."""
    from isaaclab_ov.renderers import parse_cuda_device_index

    try:
        parse_cuda_device_index(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return value


def _ovxr_renderer_cfg(sim_device: str):
    """Build the one OVRTX configuration shared by the OVXR session and its PiP cameras.

    The geometry publication device must be a rendering GPU: one of ``--ovrtx_active_cuda_gpus``,
    or the simulation GPU when the renderer follows it. Otherwise the copies would land on a GPU
    that does not render. The configuration rejects a publication GPU outside an explicit set;
    this resolves the simulation GPU so the launch fails before the app starts in the other cases.
    Without a simulation GPU, the rendering GPUs must be explicit.
    """
    from isaaclab_ov.renderers import OVRTXRendererCfg, parse_cuda_device_index

    publication = args_cli.geometry_publication_device
    if publication is not None and args_cli.ovrtx_active_cuda_gpus is None:
        if not sim_device.startswith("cuda"):
            raise ValueError(
                f"--geometry_publication_device {publication} requires --ovrtx_active_cuda_gpus when physics runs"
                f" on {sim_device!r}: there is no simulation GPU for rendering to follow."
            )
        # Only this path needs the session module, which loads the OVRTX renderer before launch.
        from ..ovxr_session import resolve_cuda_device

        rendering = parse_cuda_device_index(resolve_cuda_device(sim_device))
        if parse_cuda_device_index(publication) != rendering:
            raise ValueError(
                f"--geometry_publication_device {publication} is not a rendering GPU {[rendering]}; select the"
                " simulation GPU or list the rendering GPUs with --ovrtx_active_cuda_gpus."
            )
    return OVRTXRendererCfg(
        active_cuda_gpus=args_cli.ovrtx_active_cuda_gpus,
        geometry_publication_device=publication,
    )


def _apply_teleop_frame(env, action, issue: str | None, haptic_driver, *, render: bool = True) -> tuple[bool, bool]:
    """Step one valid action, or hold the scene while an unsafe tracker frame is rejected.

    ``render`` is false under OVXR, which owns rendering and republishes the unchanged scene.
    Returns whether an action was applied and whether the environment auto-reset.
    """
    if issue is not None:
        if render:
            env.sim.render()
        if haptic_driver is not None:
            haptic_driver.stop()
        return False, False
    # process actions
    actions = action.repeat(env.num_envs, 1)
    # apply actions
    _, _, terminated, truncated, _ = env.step(actions)
    auto_reset = bool(torch.any(terminated) or torch.any(truncated))
    # render controller haptics from post-step contact forces
    if haptic_driver is not None:
        haptic_driver.update()
    return True, auto_reset


def _restore_ovxr_shutdown_handlers(previous_handlers: dict) -> None:
    for signum, handler in previous_handlers.items():
        signal.signal(signum, handler)


def _close_teleop_resources(
    env,
    camera_feed_session,
    presentation: OvxrPresentation | None,
    previous_shutdown_handlers,
    teleop_interface=None,
    *,
    previous_context=None,
) -> None:
    """Preserve the first failure and drain XR before freeing its borrowed environment."""
    failure = sys.exception()
    cleanup_failure = None
    xr_drained = True
    camera_drained = True

    def report_cleanup_error(error: Exception) -> None:
        nonlocal cleanup_failure
        logger.exception("Teleoperation cleanup failed")
        if failure is not None:
            failure.add_note(f"Cleanup also failed: {error!r}")
        elif cleanup_failure is None:
            cleanup_failure = error
        else:
            cleanup_failure.add_note(f"Cleanup also failed: {error!r}")

    try:
        input_drain_failed = getattr(teleop_interface, "session_cleanup_failed", False) is True
        xr_drained = not input_drain_failed
        if presentation is not None:
            # Releases the session only after its input device drained.
            try:
                presentation.close()
            except Exception as error:
                xr_drained = False
                report_cleanup_error(error)
        elif input_drain_failed:
            report_cleanup_error(
                RuntimeError("Isaac Capture cleanup failed; borrowed XR/environment drain is unproved")
            )
        if not input_drain_failed:
            # A successfully bound feed exits before Capture. This is also the
            # cleanup path for preparation that never reached bind().
            try:
                camera_feed_session.close()
            except Exception as error:
                camera_drained = False
                report_cleanup_error(error)
        if env is not None and xr_drained and camera_drained:
            if presentation is None or not presentation.session.async_worker_owns_environment:
                try:
                    env.close()
                except Exception as error:
                    report_cleanup_error(error)
        elif env is None and xr_drained and camera_drained:
            from isaaclab.sim import SimulationContext

            if presentation is None or not presentation.session.async_worker_owns_environment:
                context = SimulationContext.instance()
                if context is not None and context is not previous_context:
                    try:
                        SimulationContext.clear_instance()
                    except Exception as error:
                        report_cleanup_error(error)
    finally:
        # Keep repeated SIGINT/SIGTERM on the owner throughout this cleanup attempt.
        try:
            _restore_ovxr_shutdown_handlers(previous_shutdown_handlers)
        except Exception as error:
            report_cleanup_error(error)
    if cleanup_failure is not None:
        raise cleanup_failure
    if env is not None and xr_drained and camera_drained:
        print("Environment closed")


def run_teleoperation(  # noqa: C901
    env_cfg: ManagerBasedRLEnvCfg,
    teleop_cfg: IsaacTeleopCfg | DeviceCfg,
    use_isaac_teleop: bool,
    selected_cameras: tuple[OvLibrariesXrCameraCfg, ...] = (),
) -> None:
    """Create the environment and teleop device, then run the teleoperation loop until the app is closed."""

    from isaaclab.sim import SimulationContext

    from .. import XrCameraFeedSession, poll_control_events

    previous_context = SimulationContext.instance()
    camera_feed_session = XrCameraFeedSession.prepare(
        env_cfg,
        enabled=args_cli.xr and use_isaac_teleop,
        camera_rendering_enabled=not args_cli.disable_external_cameras and not args_cli.using_ovxr,
    )

    env = None
    teleop_interface = None
    presentation = None
    previous_shutdown_handlers = {}
    shutting_down = False
    try:
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

        if args_cli.using_ovxr:
            presentation = OvxrPresentation.prepare(
                env_cfg,
                args_cli,
                application_name=f"isaaclab-ovxr-teleop-{args_cli.task}",
                auxiliary_cameras=selected_cameras,
                renderer_cfg=_ovxr_renderer_cfg(env_cfg.sim.device),
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
                cloudxr_env_file=(None if args_cli.using_ovxr else args_cli.cloudxr_env),
                auto_launch_cloudxr=args_cli.auto_launch_cloudxr and not args_cli.using_ovxr,
                enable_debug_visualization=args_cli.enable_debug_visualization,
                use_kit_xr_bridge=args_cli.xr,
                external_oxr_handles_provider=(
                    presentation.session.external_oxr_handles if presentation is not None else None
                ),
                haptic_cfg=getattr(env_cfg, "haptic_feedback", None),
            )
        except Exception as e:
            logger.error(f"Failed to create teleop device: {e}")
            return

        if presentation is not None:
            presentation.input_device = teleop_interface
        print(f"Using teleop device: {teleop_interface}")

        haptic_driver = None
        if use_isaac_teleop:
            from .. import create_haptic_feedback_driver

            haptic_driver = create_haptic_feedback_driver(env.unwrapped, teleop_interface, env_cfg)

        require_initial_bimanual_tracking = (
            env_cfg.isaac_teleop.require_initial_bimanual_hand_tracking if use_isaac_teleop else False
        )
        initialized_hand_tracking: set[str] = set()
        skipped_tracking_frames = 0
        action_conditioner = env_cfg.isaac_teleop.action_conditioner if use_isaac_teleop else None
        action_validator = env_cfg.isaac_teleop.action_validator if use_isaac_teleop else None
        action_state: dict = {}

        if use_isaac_teleop:
            # the device reads these from the focused visualizer window
            print("Isaac Capture control keys: [B] start/resume  [P] pause  [R] reset")

        def run_loop():
            """Inner function to run the teleop loop with access to nonlocal variables."""
            nonlocal should_reset_recording_instance, skipped_tracking_frames, teleoperation_active

            # reset environment
            env.reset()
            teleop_interface.reset()
            initialized_hand_tracking.clear()
            action_state.clear()
            if presentation is not None:
                presentation.mark_changed(env, reset=True)

            # Without --xr there is no headset to send START, so start locally ([B]/[P] can
            # still pause/resume). The reset() above is a host reset (a pure pulse), so it does
            # not cancel this start.
            if use_isaac_teleop and not (args_cli.xr or args_cli.using_ovxr):
                teleop_interface.request_start()

            stack_name = "Isaac Capture" if use_isaac_teleop else "native"
            print(f"{stack_name} teleoperation started. Press 'R' to reset the environment.")

            # simulate environment
            while env.sim.is_running() and not (presentation is not None and presentation.session.shutdown_requested):
                iteration_started_s = time.perf_counter() if presentation is not None else 0.0
                try:
                    # run everything in inference mode
                    with torch.inference_mode():
                        # Reset invalidates the tracking anchor. Publish the new epoch
                        # before reading input, retaining force across deferred admissions.
                        if not admit_reset_publication(env, presentation):
                            continue
                        # get device command
                        target_frame_prim_path = (
                            teleop_interface.target_frame_prim_path
                            if presentation is not None and use_isaac_teleop
                            else None
                        )
                        if target_frame_prim_path is None:
                            action = teleop_interface.advance()
                        else:
                            target_T_world = presentation.session.get_target_frame_T_world(target_frame_prim_path)
                            action = teleop_interface.advance(target_T_world=target_T_world)

                        if use_isaac_teleop:
                            ctrl = poll_control_events(teleop_interface)
                            if ctrl.is_active is not None:
                                teleoperation_active = ctrl.is_active
                            if ctrl.should_reset:
                                should_reset_recording_instance = True

                        if not teleoperation_active:
                            action_state.clear()

                        auto_reset = False
                        # action is None when Isaac Capture session hasn't started yet
                        # (e.g. waiting for user to click "Start AR")
                        if action is None or not teleoperation_active:
                            update_idle_presentation(env, presentation, report_timing=False)
                            if haptic_driver is not None:
                                haptic_driver.stop()
                            action_applied = False
                        else:
                            issue = initial_hand_tracking_issue(
                                teleop_interface,
                                initialized_hand_tracking,
                                required=require_initial_bimanual_tracking,
                            )
                            if issue is None:
                                action, issue = prepare_teleop_action(
                                    env, action, action_conditioner, action_validator, action_state
                                )
                            action_applied, auto_reset = _apply_teleop_frame(
                                env, action, issue, haptic_driver, render=presentation is None
                            )
                            if issue is not None:
                                skipped_tracking_frames += 1
                                if skipped_tracking_frames <= 5 or skipped_tracking_frames % 200 == 0:
                                    logger.warning(
                                        "Rendering without a physics step for unsafe teleop frame (%s); skipped=%d",
                                        issue,
                                        skipped_tracking_frames,
                                    )

                        reset_applied = should_reset_recording_instance or auto_reset
                        if reset_applied:
                            if should_reset_recording_instance:
                                env.reset()
                                teleop_interface.reset()
                                if use_isaac_teleop:
                                    teleop_interface.notify_reset()
                            elif use_isaac_teleop:
                                # env.step already reset the robot. Notify the input graph
                                # without echoing another operator/environment reset.
                                teleop_interface.notify_reset()
                            else:
                                teleop_interface.reset()
                            initialized_hand_tracking.clear()
                            action_state.clear()
                            camera_feed_session.refresh()
                            should_reset_recording_instance = False
                            print("Environment reset complete")

                        if presentation is not None:
                            presentation.publish(env, changed=action_applied, reset=reset_applied)

                        _pace_ovxr_control_loop(
                            presentation is not None,
                            action_applied,
                            iteration_started_s=iteration_started_s,
                            action_period_s=env.step_dt,
                            step_hz=args_cli.step_hz,
                            wait_for_shutdown=(
                                presentation.session.wait_for_shutdown if presentation is not None else None
                            ),
                        )
                except Exception as e:
                    logger.error(f"Error during simulation step: {e}")
                    if presentation is not None:
                        raise
                    break

        # Install after environment/device initialization because embedded native
        # libraries may replace inherited signal dispositions during startup.
        if presentation is not None:

            def request_shutdown(_signum, _frame) -> None:
                nonlocal shutting_down
                if not presentation.started and not shutting_down:
                    shutting_down = True
                    raise KeyboardInterrupt
                presentation.session.request_shutdown()

            previous_shutdown_handlers = {
                sig: signal.signal(sig, request_shutdown) for sig in (signal.SIGINT, signal.SIGTERM)
            }
        # Run the teleoperation loop
        # Isaac Capture requires a context manager, native devices don't
        if use_isaac_teleop:
            if presentation is not None:
                # Capture releases borrowed handles before the outer cleanup
                # closes OVXR and then the environment.
                with teleop_interface, camera_feed_session.bind(env):
                    if presentation.start(
                        env,
                        timeout_s=_RENDERER_ATTACH_TIMEOUT_S,
                        required_openxr_extensions=teleop_interface.required_openxr_extensions,
                    ):
                        run_loop()
            else:
                with teleop_interface, camera_feed_session.bind(env):
                    run_loop()
        else:
            with camera_feed_session.bind(env):
                run_loop()
    finally:
        shutting_down = True
        _close_teleop_resources(
            env,
            camera_feed_session,
            presentation,
            previous_shutdown_handlers,
            teleop_interface,
            previous_context=previous_context,
        )


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
