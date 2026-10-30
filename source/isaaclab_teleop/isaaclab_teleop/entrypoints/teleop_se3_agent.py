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
import os
import signal
import sys
import time
from collections.abc import Callable

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation, validate_launch
from isaaclab.cli import load_external_tasks
from isaaclab.devices import DeviceCfg
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
from ..action_policy import (
    initial_hand_tracking_issue as _required_hand_tracking_issue,
)
from ..action_policy import prepare_teleop_action
from ..device_selection import create_teleop_device, select_teleop_cfg

logger = logging.getLogger(__name__)

_RENDERER_ATTACH_TIMEOUT_S = 180.0
"""Allow cold RTX preparation before failing attachment; client connection waits separately."""

args_cli: argparse.Namespace
"""Parsed command line, set by :func:`_parse_args`."""


def _build_parser() -> argparse.ArgumentParser:
    """Build workflow and simulation launcher arguments."""
    parser = argparse.ArgumentParser(description="Teleoperation for Isaac Lab environments.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
    parser.add_argument(
        "--teleop_device",
        type=str,
        default=None,
        help=(
            "Teleop device name: a task device or keyboard, spacemouse, gamepad. When omitted, use the task's"
            " Isaac Capture pipeline, otherwise keyboard. Built-in devices run through Isaac Capture."
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
        type=str,
        default=None,
        help="Comma-separated CUDA indices for OVRTX rendering, e.g. 0. Does not select physics or camera outputs.",
    )
    parser.add_argument(
        "--geometry_publication_device",
        type=str,
        default=None,
        metavar="CUDA_DEVICE",
        help=(
            "Destination CUDA device for completed OVStage geometry publication, e.g. cuda:0. "
            "Adds copy work for geometry on another GPU; unset preserves renderer configuration. "
            "Requires --ovxr; does not select physics or rendering placement."
        ),
    )
    parser.add_argument(
        "--ovxr_desktop_mirror",
        action="store_true",
        default=False,
        help="Show the OVXR desktop mirror (enabled by default; --headless disables it).",
    )
    parser.add_argument("--headless", action="store_true", help="Disable the OVXR desktop mirror.")
    parser.add_argument(
        "--ovxr_desktop_mirror_fps",
        type=float,
        default=30.0,
        help="Maximum desktop-mirror update rate (default: 30 Hz).",
    )
    parser.add_argument(
        "--ovxr_pip_camera",
        type=str,
        nargs="+",
        default=None,
        help="Show up to three task-configured cameras as application-owned XR panels.",
    )
    parser.add_argument(
        "--ovxr_pip_fps",
        type=float,
        default=30.0,
        help="Maximum OVXR PiP update rate (default: 30 Hz).",
    )
    parser.add_argument(
        "--ovxr_pip_head_locked",
        action="store_true",
        help="Place the application camera quad in the OpenXR VIEW space.",
    )
    parser.add_argument(
        "--ovxr_pip_view_position",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Optional VIEW-space camera panel center in meters; requires --ovxr_pip_head_locked.",
    )

    add_launcher_args(parser)
    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse workflow arguments, leaving task overrides in ``sys.argv``."""
    global args_cli
    if argv is not None:
        sys.argv = [sys.argv[0], *argv]
    parser = _build_parser()
    args_cli, hydra_args = setup_preset_cli(parser)
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
    if args_cli.ovxr_desktop_mirror and not args_cli.using_ovxr:
        parser.error("--ovxr_desktop_mirror requires --ovxr")
    args_cli.ovxr_desktop_mirror = args_cli.using_ovxr and not args_cli.headless
    if args_cli.ovxr_pip_camera is not None and not args_cli.using_ovxr:
        parser.error("--ovxr_pip_camera requires --ovxr")
    if args_cli.ovxr_pip_camera is not None and args_cli.disable_external_cameras:
        parser.error("--ovxr_pip_camera cannot be combined with --disable_external_cameras")
    if not math.isfinite(args_cli.ovxr_pip_fps) or args_cli.ovxr_pip_fps <= 0.0:
        parser.error("--ovxr_pip_fps must be finite and positive")
    if args_cli.ovxr_pip_view_position is not None:
        if not args_cli.ovxr_pip_head_locked or args_cli.ovxr_pip_camera is None:
            parser.error("--ovxr_pip_view_position requires a head-locked PiP camera")
        if not all(math.isfinite(value) for value in args_cli.ovxr_pip_view_position):
            parser.error("--ovxr_pip_view_position must be finite")
    if not math.isfinite(args_cli.ovxr_desktop_mirror_fps) or args_cli.ovxr_desktop_mirror_fps <= 0:
        parser.error("--ovxr_desktop_mirror_fps must be finite and positive")

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


def _rtx_rendering_requested(args: argparse.Namespace) -> bool:
    """Return whether the CLI selects a renderer that actually drives RTX rendering.

    The RTX/DLSS global settings are only meaningful when something renders through RTX.
    That happens when the Kit visualizer is enabled (``--viz kit``), when external cameras
    are rendered (on by default; see ``--disable_external_cameras``), or in XR mode (``--xr``).
    A pure-headless session with none of these renders nothing.

    This reads the resolved namespace intent rather than any Kit/carb runtime state so the
    check keeps working as these scripts grow support for other rendering backends.
    """
    visualizers = getattr(args, "visualizer", None) or []
    external_cameras = not getattr(args, "disable_external_cameras", False)
    return external_cameras or ("kit" in visualizers) or bool(getattr(args, "xr", False))


def _make_haptic_io(env, teleop_interface, env_cfg, use_isaac_teleop: bool):
    """Return ``(update, stop)`` callables driving controller haptics, or no-ops.

    Keeps haptics opt-in without branching in the main loop: both callables are
    no-ops unless the active device is an Isaac Capture device and the env declares
    a ``haptic_feedback`` config. ``update`` renders the current contact force;
    ``stop`` zeroes it so a stale pulse does not persist while teleop is paused.
    """
    noop = lambda: None  # noqa: E731
    if not use_isaac_teleop:
        return noop, noop
    from .. import create_haptic_feedback_driver

    driver = create_haptic_feedback_driver(env.unwrapped, teleop_interface, env_cfg)
    if driver is None:
        return noop, noop
    return driver.update, driver.stop


def _apply_teleop_frame(env, action, issue: str | None, haptic_update, haptic_stop) -> bool:
    """Step one valid action, or render-only while an unsafe tracker frame is held."""
    if issue is not None:
        env.sim.render()
        haptic_stop()
        return False
    env.step(action.repeat(env.num_envs, 1))
    haptic_update()
    return True


def _ovxr_publication_identity(
    physics_step_id: int, *, reset_generation: int | None = None
) -> int | tuple[str, int, int]:
    """Return a dedup identity that also represents a reset at an unchanged physics step."""
    if reset_generation is None:
        return physics_step_id
    return ("reset", reset_generation, physics_step_id)


def _resolve_env_cfg() -> tuple[
    ManagerBasedRLEnvCfg, IsaacTeleopCfg | DeviceCfg, bool, list[tuple[str, object, object]]
]:
    """Resolve task configuration and teleop ownership before runtime launch."""
    overrides = list(sys.argv[1:])
    env_cfg, _ = resolve_task_config(args_cli.task, "", overrides=overrides)
    if args_cli.device is not None:
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
    if args_cli.using_ovxr and args_cli.num_envs != 1:
        raise ValueError("--ovxr currently requires --num_envs 1")
    if args_cli.using_ovxr:
        stage_mode = os.environ.get("ISAAC_LAB_OVRTX_USE_OVSTAGE", "1")
        if stage_mode.strip() != "1":
            raise ValueError("--ovxr requires OVStage (ISAAC_LAB_OVRTX_USE_OVSTAGE=1)")

    pip_selection = []
    keep_cameras: tuple[str, ...] = ()
    if args_cli.ovxr_pip_camera is not None:
        from isaaclab_ov.renderers import OVRTXRendererCfg

        from ..camera_feed import select_ovxr_camera_feeds

        names = args_cli.ovxr_pip_camera
        if len(names) > 1 and args_cli.ovxr_pip_view_position is not None:
            raise ValueError("A PiP position override requires one camera; multiple panels use task layout")
        renderer_cfg = OVRTXRendererCfg(
            active_cuda_gpus=args_cli.ovrtx_active_cuda_gpus,
            geometry_publication_device=args_cli.geometry_publication_device,
        )
        pip_selection = select_ovxr_camera_feeds(env_cfg, names, renderer_cfg)
        keep_cameras = tuple(names)

    # The leased OVRTX renderer is the only render path in OVXR. Strip
    # every task camera before the launcher scans the config.
    if args_cli.using_ovxr or args_cli.disable_external_cameras:
        from .. import remove_camera_configs

        env_cfg = remove_camera_configs(env_cfg, keep=keep_cameras)

    if env_cfg.sim.physics is None:
        from isaaclab.physics import PhysicsCfg

        env_cfg.sim.physics = PhysicsCfg()
    configs, kit_sources = validate_launch(env_cfg, vars(args_cli).copy())
    if args_cli.using_ovxr:
        from isaaclab.visualizers import VisualizerBackendCfg

        if kit_sources or any(isinstance(cfg, VisualizerBackendCfg) for cfg in configs):
            raise ValueError("--ovxr requires no Kit runtime or local visualizer")
    if use_isaac_teleop and teleop_cfg.prepare_environment is not None:
        teleop_cfg.prepare_environment(env_cfg)
    return env_cfg, teleop_cfg, use_isaac_teleop, pip_selection


def _install_ovxr_shutdown_handlers(ov_libraries_xr_session) -> dict:
    """Make owner-script signals request the same orderly shutdown as OVUI."""
    if ov_libraries_xr_session is None:
        return {}

    def request_shutdown(_signum, _frame) -> None:
        ov_libraries_xr_session.request_shutdown()

    return {signum: signal.signal(signum, request_shutdown) for signum in (signal.SIGINT, signal.SIGTERM)}


def _restore_ovxr_shutdown_handlers(previous_handlers: dict) -> None:
    for signum, handler in previous_handlers.items():
        signal.signal(signum, handler)


def _close_teleop_resources(
    env,
    camera_feed_session,
    ov_libraries_xr_session,
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
        if input_drain_failed:
            xr_drained = False
            report_cleanup_error(
                RuntimeError("Isaac Capture cleanup failed; borrowed XR/environment drain is unproved")
            )
            if ov_libraries_xr_session is not None:
                try:
                    ov_libraries_xr_session.request_shutdown()
                except Exception as error:
                    report_cleanup_error(error)
        elif ov_libraries_xr_session is not None:
            try:
                ov_libraries_xr_session.close()
            except Exception as error:
                xr_drained = False
                report_cleanup_error(error)
        if not input_drain_failed:
            # A successfully bound feed exits before Capture. This is also the
            # cleanup path for preparation that never reached bind().
            try:
                camera_feed_session.close()
            except Exception as error:
                camera_drained = False
                report_cleanup_error(error)
        if env is not None and xr_drained and camera_drained:
            if ov_libraries_xr_session is None or not ov_libraries_xr_session.async_worker_owns_environment:
                try:
                    env.close()
                except Exception as error:
                    report_cleanup_error(error)
        elif env is None and xr_drained and camera_drained:
            from isaaclab.sim import SimulationContext

            if ov_libraries_xr_session is None or not ov_libraries_xr_session.async_worker_owns_environment:
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
    pip_selection: list[tuple[str, object, object]],
) -> None:
    """Run teleoperation through the selected Kit or OVXR runtime."""

    from isaaclab.sim import SimulationContext

    from .. import XrCameraFeedSession

    previous_context = SimulationContext.instance()
    camera_feed_session = XrCameraFeedSession.prepare(
        env_cfg,
        enabled=args_cli.xr and use_isaac_teleop,
        camera_rendering_enabled=not args_cli.disable_external_cameras and not args_cli.using_ovxr,
    )

    env = None
    teleop_interface = None
    ov_libraries_xr_session = None
    previous_shutdown_handlers = {}
    try:
        # Apply the RTX/DLSS global settings when an RTX render pipeline will run (Kit visualizer,
        # external cameras, or XR). ``apply_isaac_rtx_global_settings`` uses ``omni.replicator``,
        # which some experiences do not preload, so ensure it is loaded first.
        if has_kit() and _rtx_rendering_requested(args_cli):
            from isaaclab_physx.renderers import IsaacRtxRendererGlobalSettingsCfg
            from isaaclab_physx.renderers.isaac_rtx_renderer_utils import apply_isaac_rtx_global_settings

            # The renderer's public helper loads Replicator when its settings need it.
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
            from isaaclab_ov.renderers import OVRTXRendererCfg

            from .. import OvLibrariesXrConfig, OvLibrariesXrSession
            from ..camera_feed import resolve_ovxr_camera_configs
            from ..isaac_teleop_cfg import XrCameraFeedLayoutCfg

            layout = env_cfg.isaac_teleop.xr_camera_feed_layout
            if args_cli.ovxr_pip_head_locked:
                layout = (layout or XrCameraFeedLayoutCfg()).replace(placement="head_locked")
            selected_cameras = resolve_ovxr_camera_configs(
                pip_selection,
                layout,
                max_fps=args_cli.ovxr_pip_fps,
                position_override=args_cli.ovxr_pip_view_position,
            )
            ov_libraries_xr_session = OvLibrariesXrSession.prepare(
                env_cfg.sim,
                OvLibrariesXrConfig(
                    application_name=f"isaaclab-ovxr-teleop-{args_cli.task}",
                    session_mode=args_cli.ovxr_session_mode,
                    cloudxr_env_file=args_cli.cloudxr_env,
                    auto_launch_cloudxr=args_cli.auto_launch_cloudxr,
                    desktop_mirror_enabled=args_cli.ovxr_desktop_mirror,
                    desktop_mirror_max_fps=args_cli.ovxr_desktop_mirror_fps,
                    auxiliary_cameras=selected_cameras,
                ),
                renderer_cfg=(
                    selected_cameras[0].camera_cfg.renderer_cfg
                    if selected_cameras
                    else OVRTXRendererCfg(
                        active_cuda_gpus=args_cli.ovrtx_active_cuda_gpus,
                        geometry_publication_device=args_cli.geometry_publication_device,
                    )
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

        except Exception:
            logger.exception("Failed to create environment")
            raise

        # Flags for controlling teleoperation flow
        should_reset_recording_instance = False
        teleoperation_active = True
        reset_publication_generation = 0

        # Callback handlers
        def reset_recording_instance() -> None:
            """Request an environment reset at the next simulation step."""
            nonlocal should_reset_recording_instance
            should_reset_recording_instance = True
            print("Reset triggered - Environment will reset on next step")

        def start_teleoperation() -> None:
            """Resume applying teleoperation commands to the environment."""
            nonlocal teleoperation_active
            teleoperation_active = True
            print("Teleoperation activated")

        def stop_teleoperation() -> None:
            """Pause applying teleoperation commands to the environment."""
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

        # Use the upstream device factory for Capture and native inputs alike.
        from .. import poll_control_events

        try:
            teleop_interface = create_teleop_device(
                teleop_cfg,
                env_prim_path=env.scene.env_prim_paths[0],
                sim_device=env.sim.device,
                callbacks=teleoperation_callbacks,
                visualizers=env.sim.visualizers,
                cloudxr_env_file=(None if args_cli.using_ovxr else args_cli.cloudxr_env),
                auto_launch_cloudxr=args_cli.auto_launch_cloudxr and not args_cli.using_ovxr,
                enable_debug_visualization=args_cli.enable_debug_visualization,
                use_kit_xr_bridge=args_cli.xr,
                external_oxr_handles_provider=(
                    ov_libraries_xr_session.external_oxr_handles if ov_libraries_xr_session is not None else None
                ),
                haptic_cfg=getattr(env_cfg, "haptic_feedback", None),
            )
        except Exception:
            logger.exception("Failed to create teleop device")
            raise

        if teleop_interface is None:
            logger.error("Failed to create teleop interface")
            raise RuntimeError("Failed to create teleop interface")

        print(f"Using teleop device: {teleop_interface}")

        # Optional controller haptics: no-ops unless the env declares a
        # ``haptic_feedback`` config and the device can render it (Isaac Capture).
        haptic_update, haptic_stop = _make_haptic_io(env, teleop_interface, env_cfg, use_isaac_teleop)

        require_initial_bimanual_tracking = (
            env_cfg.isaac_teleop.require_initial_bimanual_hand_tracking if use_isaac_teleop else False
        )
        initialized_hand_tracking: set[str] = set()
        skipped_tracking_frames = 0
        action_conditioner = env_cfg.isaac_teleop.action_conditioner if use_isaac_teleop else None
        action_validator = env_cfg.isaac_teleop.action_validator if use_isaac_teleop else None
        action_state: dict = {}

        if use_isaac_teleop:
            print("Isaac Capture control keys: [B] start/resume  [P] pause  [R] reset")

        def run_loop():
            """Inner function to run the teleop loop with access to nonlocal variables."""
            nonlocal should_reset_recording_instance, skipped_tracking_frames, teleoperation_active
            nonlocal reset_publication_generation

            # reset environment
            env.reset()
            teleop_interface.reset()
            initialized_hand_tracking.clear()
            action_state.clear()
            pending_publication_identity = (
                _ovxr_publication_identity(
                    env.sim.get_physics_step_count(), reset_generation=reset_publication_generation
                )
                if ov_libraries_xr_session is not None
                else None
            )
            pending_publication_force = ov_libraries_xr_session is not None

            # Without --xr there is no headset to send START, so start locally ([B]/[P] can
            # still pause/resume). The reset() above is a host reset (a pure pulse), so it does
            # not cancel this start.
            if use_isaac_teleop and not (args_cli.xr or args_cli.using_ovxr):
                teleop_interface.request_start()

            stack_name = "Isaac Capture" if use_isaac_teleop else "native"
            print(f"{stack_name} teleoperation started. Press 'R' to reset the environment.")

            # simulate environment
            while env.sim.is_running() and not (
                ov_libraries_xr_session is not None and ov_libraries_xr_session.shutdown_requested
            ):
                iteration_started_s = time.perf_counter() if ov_libraries_xr_session is not None else 0.0
                try:
                    # run everything in inference mode
                    with torch.inference_mode():
                        # Reset invalidates the tracking anchor. Publish the new epoch
                        # before reading input, retaining force across deferred admissions.
                        if pending_publication_force:
                            admitted = ov_libraries_xr_session.request_render(pending_publication_identity, force=True)
                            if admitted is None:
                                ov_libraries_xr_session.wait_for_shutdown(env.step_dt)
                                continue
                            pending_publication_identity = None
                            pending_publication_force = False
                        # get device command
                        target_frame_prim_path = (
                            teleop_interface.target_frame_prim_path
                            if ov_libraries_xr_session is not None and use_isaac_teleop
                            else None
                        )
                        if target_frame_prim_path is None:
                            action = teleop_interface.advance()
                        else:
                            target_T_world = ov_libraries_xr_session.get_target_frame_T_world(target_frame_prim_path)
                            action = teleop_interface.advance(target_T_world=target_T_world)

                        if use_isaac_teleop:
                            ctrl = poll_control_events(teleop_interface)
                            if ctrl.is_active is not None:
                                teleoperation_active = ctrl.is_active
                            if ctrl.should_reset:
                                should_reset_recording_instance = True

                        if not teleoperation_active:
                            action_state.clear()

                        # action is None when Isaac Capture session hasn't started yet
                        # (e.g. waiting for user to click "Start AR")
                        if action is None:
                            if ov_libraries_xr_session is None:
                                env.sim.render()
                            haptic_stop()
                            action_applied = False
                        elif teleoperation_active:
                            issue = (
                                _required_hand_tracking_issue(teleop_interface, initialized_hand_tracking)
                                if require_initial_bimanual_tracking
                                else None
                            )
                            if issue is None:
                                action, issue = prepare_teleop_action(
                                    env, action, action_conditioner, action_validator, action_state
                                )
                            action_applied = _apply_teleop_frame(env, action, issue, haptic_update, haptic_stop)
                            if issue is not None:
                                skipped_tracking_frames += 1
                                if skipped_tracking_frames <= 5 or skipped_tracking_frames % 200 == 0:
                                    logger.warning(
                                        "Rendering without a physics step for unsafe teleop frame (%s); skipped=%d",
                                        issue,
                                        skipped_tracking_frames,
                                    )
                        else:
                            action_state.clear()
                            if ov_libraries_xr_session is None:
                                env.sim.render()
                            # not stepping: zero haptics so a paused grip stops buzzing
                            haptic_stop()
                            action_applied = False

                        state_changed = action_applied
                        force_publication = False
                        publication_identity = env.sim.get_physics_step_count()
                        if should_reset_recording_instance:
                            env.reset()
                            teleop_interface.reset()
                            initialized_hand_tracking.clear()
                            action_state.clear()
                            camera_feed_session.refresh()
                            should_reset_recording_instance = False
                            if ov_libraries_xr_session is not None:
                                state_changed = True
                                reset_publication_generation += 1
                                force_publication = True
                                publication_identity = _ovxr_publication_identity(
                                    env.sim.get_physics_step_count(), reset_generation=reset_publication_generation
                                )
                            print("Environment reset complete")

                        if ov_libraries_xr_session is not None:
                            if state_changed:
                                pending_publication_identity = publication_identity
                                pending_publication_force = force_publication
                            if pending_publication_identity is not None:
                                admitted = ov_libraries_xr_session.request_render(
                                    pending_publication_identity, force=pending_publication_force
                                )
                                if admitted is not None:
                                    pending_publication_identity = None
                                    pending_publication_force = False

                        if ov_libraries_xr_session is not None:
                            ov_libraries_xr_session.report_host_timing(
                                env.sim.get_physics_step_count(), env.sim.get_physics_dt()
                            )
                        _pace_ovxr_control_loop(
                            ov_libraries_xr_session is not None,
                            action_applied,
                            iteration_started_s=iteration_started_s,
                            action_period_s=env.step_dt,
                            step_hz=args_cli.step_hz,
                            wait_for_shutdown=(
                                ov_libraries_xr_session.wait_for_shutdown
                                if ov_libraries_xr_session is not None
                                else None
                            ),
                        )
                except Exception as e:
                    logger.exception("Error during simulation step: %s", e)
                    raise

        # Install after environment/device initialization because embedded native
        # libraries may replace inherited signal dispositions during startup.
        previous_shutdown_handlers = _install_ovxr_shutdown_handlers(ov_libraries_xr_session)
        # IsaacTeleop requires a context manager, native devices do not.
        if use_isaac_teleop:
            if ov_libraries_xr_session is not None:
                # Capture releases borrowed handles before the outer cleanup
                # closes OVXR and then the environment.
                with teleop_interface, camera_feed_session.bind(env):
                    render_worker_started = ov_libraries_xr_session.start_render_worker(
                        env,
                        tracking_anchor_path=teleop_interface.xr_render_anchor_path,
                        required_openxr_extensions=teleop_interface.required_openxr_extensions,
                        hidden_visual_prims=env_cfg.isaac_teleop.xr_cfg.hidden_visual_prims,
                        excluded_visual_meshes=env_cfg.isaac_teleop.xr_cfg.excluded_visual_meshes,
                        external_anchor_state=teleop_interface.external_anchor_state,
                        timeout_s=_RENDERER_ATTACH_TIMEOUT_S,
                    )
                    if render_worker_started is not False:
                        run_loop()
            else:
                with teleop_interface, camera_feed_session.bind(env):
                    run_loop()
        else:
            with camera_feed_session.bind(env):
                run_loop()
    finally:
        _close_teleop_resources(
            env,
            camera_feed_session,
            ov_libraries_xr_session,
            previous_shutdown_handlers,
            teleop_interface,
            previous_context=previous_context,
        )


def main() -> None:
    """Resolve the task and run teleoperation with its selected runtime."""
    resolved_env_cfg, selected_teleop_cfg, uses_isaac_teleop, pip_selection = _resolve_env_cfg()
    with launch_simulation(resolved_env_cfg, args_cli, cleanup_context=not args_cli.using_ovxr):
        run_teleoperation(resolved_env_cfg, selected_teleop_cfg, uses_isaac_teleop, pip_selection)


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
