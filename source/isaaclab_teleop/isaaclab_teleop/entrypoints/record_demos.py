# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Script to record demonstrations with Isaac Lab environments using human teleoperation.

This script allows users to record demonstrations operated by human teleoperation for a specified task.
The recorded demonstrations are stored as episodes in a hdf5 file. Users can specify the task, teleoperation
device, dataset directory, and environment stepping rate through command-line arguments.

This script supports two teleoperation stacks:
1. Native Isaac Lab teleop stack (via teleop_devices in env_cfg)
2. Isaac Capture-based stack (via isaac_teleop in env_cfg)

The script automatically detects which stack to use based on the environment config.

required arguments:
    --task                    Name of the task.

optional arguments:
    -h, --help                Show this help message and exit
    --teleop_device           Teleop device name: an env teleop_devices entry, or keyboard /
                              spacemouse. When omitted, Isaac Capture is used if configured,
                              otherwise keyboard. Keyboard and spacemouse run through Isaac Capture
                              (the teleop extra).
    --dataset_file            File path to export recorded demos. (default: "./datasets/dataset.hdf5")
    --step_hz                 Environment stepping rate in Hz. (default: 30)
    --num_demos               Number of demonstrations to record. (default: 0)
    --num_success_steps       Number of continuous steps with task success for concluding a demo as successful.
                              (default: 10)
"""

import argparse
import contextlib
import importlib
import importlib.util
import logging
import math
import os
import sys
import time
from typing import TYPE_CHECKING

import gymnasium as gym
import torch
from isaaclab_isaacsim.renderers import IsaacRtxRendererGlobalSettingsCfg

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.cli import load_external_tasks
from isaaclab.devices import DeviceCfg
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.envs.utils._manual_success import _prepare_success_term
from isaaclab.managers import CommandTermCfg, DatasetExportMode
from isaaclab.utils.string import string_to_callable
from isaaclab.utils.version import has_kit

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from .. import (
    CLOUDXR_AVP_ENV,
    CLOUDXR_JS_ENV,
    CLOUDXR_STANDALONE_ENV,
    IsaacTeleopCfg,
    OvLibrariesXrCameraFeedCfg,
    se3_keyboard_teleop_cfg,
    se3_spacemouse_teleop_cfg,
)
from ..action_policy import initial_hand_tracking_issue, prepare_teleop_action
from ..device_selection import create_teleop_device, select_teleop_cfg
from ._ovxr_launch import (
    add_ovxr_presentation_args,
    cuda_index_list,
    ovxr_renderer_cfg,
    prepare_ovxr_launch,
    reject_ovxr_runtime_conflicts,
    resolve_ovxr_camera_feeds,
    validate_ovxr_presentation_args,
)
from ._ovxr_presentation import OvxrPresentation, admit_reset_publication, update_idle_presentation
from ._ovxr_runtime import (
    RENDERER_ATTACH_TIMEOUT_S,
    OvxrShutdownSignals,
    close_teleop_resources,
    hold_teleop_frame,
    pace_ovxr_control_loop,
    requires_hard_sim_reset,
    warn_rejected_teleop_frame,
)

if TYPE_CHECKING:
    from .. import XrCameraFeedSession
    from ._instruction_display import InstructionDisplay

# Mimic task variants (``*-Mimic-v0``) are registered when isaaclab_mimic is installed.
if importlib.util.find_spec("isaaclab_mimic") is not None:
    importlib.import_module("isaaclab_mimic.envs")

logger = logging.getLogger(__name__)

args_cli: argparse.Namespace
"""Parsed command line, set by :func:`_parse_args`."""


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser, including the simulation launcher arguments."""
    parser = argparse.ArgumentParser(description="Record demonstrations for Isaac Lab environments.")
    parser.add_argument("--task", type=str, required=True, help="Name of the task.")
    parser.add_argument(
        "--teleop_device",
        type=str,
        default=None,
        help=(
            "Teleop device name: an entry of env_cfg.teleop_devices.devices, or a built-in device (keyboard,"
            " spacemouse). When omitted, the Isaac Capture pipeline is used if configured in the env, otherwise"
            " keyboard. Keyboard and spacemouse run through Isaac Capture (the teleop extra)."
        ),
    )
    parser.add_argument(
        "--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos."
    )
    parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")
    parser.add_argument(
        "--num_demos", type=int, default=0, help="Number of demonstrations to record. Set to 0 for infinite."
    )
    parser.add_argument(
        "--num_success_steps",
        type=int,
        default=10,
        help="Number of continuous steps with task success for concluding a demo as successful. Default is 10.",
    )
    parser.add_argument(
        "--reset_sim_buffer_each_episode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Call env.sim.reset() before the initial episode and between recording attempts."
            " Use --no-reset_sim_buffer_each_episode to preserve simulation buffers. With --ovxr, kitless"
            " physics keeps its simulator, as in MCAP replay."
        ),
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
        "--mcap_record_path",
        type=str,
        default=None,
        help=(
            "Debug-only: write the live Isaac Capture session to this MCAP file (one continuous file for the"
            " whole run). Intended for pairing with teleop_replay_agent.py in CI -- NOT a data-generation"
            " format. MCAPs produced here lack per-episode segmentation, world-frame anchor state, env reset"
            " state, and have no public Python decoder. For data-gen workflows use the HDF5 dataset path"
            " (default). Ignored when the Isaac Capture stack is not in use."
        ),
    )

    parser.add_argument(
        "--enable_debug_visualization",
        action="store_true",
        default=False,
        help="Enable hand joint and controller aim debug visualization at session start (Isaac Capture only).",
    )
    parser.add_argument(
        "--external_callback", default=None, help="Fully qualified path to an externally defined callback."
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
        "--ovxr_step_hz",
        type=float,
        default=0.0,
        help="Maximum host-control rate with --ovxr; 0 leaves active control unthrottled. XR rendering is independent.",
    )
    parser.add_argument(
        "--ovrtx_active_cuda_gpus",
        type=cuda_index_list,
        default=None,
        help="Comma-separated CUDA indices for OVRTX rendering, e.g. 0. Does not select physics or camera outputs.",
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

    # Validate required arguments
    if args_cli.task is None:
        parser.error("--task is required")

    # Enable external camera rendering by default (``--disable_external_cameras`` turns it off). The
    # ``--enable_cameras`` CLI flag was removed in Isaac Lab 3.0 (see #6656), so pass the intent to
    # the launcher; this selects a camera-rendering experience that provides RTX/DLSS support.
    # Everywhere else we read ``args_cli.disable_external_cameras`` directly.
    args_cli.enable_cameras = not args_cli.disable_external_cameras
    if args_cli.xr and args_cli.using_ovxr:
        parser.error("--xr and --ovxr are mutually exclusive OpenXR frame-loop owners")
    if not math.isfinite(args_cli.ovxr_step_hz) or args_cli.ovxr_step_hz < 0.0:
        parser.error("--ovxr_step_hz must be finite and non-negative; 0 disables active-control pacing")
    if args_cli.ovxr_step_hz > 0.0 and not args_cli.using_ovxr:
        parser.error("--ovxr_step_hz requires --ovxr")
    if not args_cli.using_ovxr and args_cli.ovrtx_active_cuda_gpus is not None:
        parser.error("--ovrtx_active_cuda_gpus requires --ovxr")
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


# CommandTerm samples its timer with uniform_, which requires finite bounds.
_RECORDING_COMMAND_RESAMPLING_TIME_S = 1.0e9


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz: int):
        """Initialize a RateLimiter with specified frequency.

        Args:
            hz: Frequency to enforce in Hertz.
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env: gym.Env):
        """Attempt to sleep at the specified rate in hz.

        Args:
            env: Environment to render during sleep periods.
        """
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def setup_output_directories() -> tuple[str, str]:
    """Set up output directories for saving demonstrations.

    Creates the output directory if it doesn't exist and extracts the file name
    from the dataset file path.

    Returns:
        tuple[str, str]: A tuple containing:
            - output_dir: The directory path where the dataset will be saved
            - output_file_name: The filename (without extension) for the dataset
    """
    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]

    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    return output_dir, output_file_name


def create_environment_config(
    output_dir: str, output_file_name: str
) -> tuple[
    ManagerBasedRLEnvCfg | DirectRLEnvCfg,
    object | None,
    IsaacTeleopCfg | DeviceCfg,
    tuple[OvLibrariesXrCameraFeedCfg, ...],
]:
    """Create and configure the environment configuration.

    Parses the environment configuration and makes necessary adjustments for demo recording.
    Extracts the success termination function and configures the recorder manager.

    Args:
        output_dir: Directory where recorded demonstrations will be saved
        output_file_name: Name of the file to store the demonstrations

    Returns:
        tuple: A tuple containing:
            - env_cfg: The configured environment configuration
            - success_term: The success termination object or None if not available
            - teleop_cfg: The resolved Capture or native device configuration
            - selected_cameras: The resolved OVXR PiP panels, empty without ``--ovxr_camera_feed``

    Raises:
        SystemExit: If parsing the environment configuration fails (the error is logged and the process exits).
        ValueError: With ``--ovxr``, if the task has no Isaac Capture pipeline or its launch conflicts with OVXR.
    """
    # Resolve the task configuration through Hydra so CLI presets are applied.
    try:
        env_cfg, _ = resolve_task_config(args_cli.task, "")
        env_cfg.sim.device = args_cli.device
        env_cfg.scene.num_envs = 1
        env_cfg.env_name = args_cli.task.split(":")[-1]
    except Exception as e:
        logger.error(f"Failed to parse environment configuration: {e}")
        exit(1)

    teleop_cfg = select_teleop_cfg(
        env_cfg,
        args_cli.teleop_device,
        {
            "keyboard": se3_keyboard_teleop_cfg(pos_sensitivity=0.2, rot_sensitivity=0.5, sim_device=args_cli.device),
            "spacemouse": se3_spacemouse_teleop_cfg(
                pos_sensitivity=0.2, rot_sensitivity=0.5, sim_device=args_cli.device
            ),
        },
        args_cli.device,
    )
    if isinstance(teleop_cfg, IsaacTeleopCfg):
        env_cfg.isaac_teleop = teleop_cfg
    elif args_cli.using_ovxr:
        raise ValueError("--ovxr requires a task configured with env_cfg.isaac_teleop")

    from isaaclab.app import validate_launch

    if args_cli.xr and args_cli.disable_external_cameras:
        env_cfg = remove_camera_configs(env_cfg)
    selected_cameras = ()
    if args_cli.using_ovxr:
        if args_cli.ovxr_camera_feed is not None:
            # Resolve the panels before launch preparation removes the other cameras, so that an
            # invalid selection or task layout fails before launch.
            selected_cameras = resolve_ovxr_camera_feeds(
                env_cfg, args_cli, ovxr_renderer_cfg(args_cli.ovrtx_active_cuda_gpus)
            )
        env_cfg = prepare_ovxr_launch(
            env_cfg, args_cli.task, env_cfg.scene.num_envs, keep=tuple(args_cli.ovxr_camera_feed or ())
        )

    if isinstance(teleop_cfg, IsaacTeleopCfg):
        if args_cli.using_ovxr or teleop_cfg.prepare_environment is not None:
            # Resolve automatic backend choices that the preparation hook depends on.
            configs, kit_sources = validate_launch(env_cfg, vars(args_cli).copy())
            if args_cli.using_ovxr:
                reject_ovxr_runtime_conflicts(configs, kit_sources)
        # Preparation may select the task's action policy, which the recording loop applies.
        if teleop_cfg.prepare_environment is not None:
            teleop_cfg.prepare_environment(env_cfg)

    # Extract the success condition for manual evaluation in the main loop. Keep
    # an inert term registered under the same name so rewards and other manager
    # terms that reference "success" can still resolve it during initialization.
    success_term = _prepare_success_term(env_cfg)
    if success_term is None:
        logger.warning(
            "No success termination term was found in the environment."
            " Will not be able to mark recorded demos as successful."
        )

    # modify configuration such that the environment runs indefinitely until
    # the goal is reached or other termination conditions are met
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    # Keep each demonstration's goals fixed; command generators still sample
    # fresh goals when the environment resets for the next recording attempt.
    if env_cfg.commands is not None:
        command_cfgs = (
            env_cfg.commands.values() if isinstance(env_cfg.commands, dict) else vars(env_cfg.commands).values()
        )
        for command_cfg in command_cfgs:
            if isinstance(command_cfg, CommandTermCfg):
                command_cfg.resampling_time_range = (
                    _RECORDING_COMMAND_RESAMPLING_TIME_S,
                    _RECORDING_COMMAND_RESAMPLING_TIME_S,
                )

    demo_recorder_cfg_entry_point = gym.spec(args_cli.task.split(":")[-1]).kwargs.get("demo_recorder_cfg_entry_point")
    if demo_recorder_cfg_entry_point is None:
        env_cfg.recorders = ActionStateRecorderManagerCfg()
    else:
        env_cfg.recorders = load_cfg_from_registry(args_cli.task, "demo_recorder_cfg_entry_point")
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    return env_cfg, success_term, teleop_cfg, selected_cameras


def setup_ui(label_text: str, env: gym.Env) -> "InstructionDisplay":
    """Set up the user interface elements.

    Creates instruction display and UI window with labels for showing information
    to the user during demonstration recording.

    Args:
        label_text: Text to display showing current recording status
        env: The environment instance for which UI is being created

    Returns:
        InstructionDisplay: The configured instruction display object
    """
    from ._instruction_display import InstructionDisplay

    instruction_display = InstructionDisplay(args_cli.xr)
    if not args_cli.xr and has_kit():
        import omni.ui as ui

        from isaaclab.envs.ui import EmptyWindow

        window = EmptyWindow(env, "Instruction")
        with window.ui_window_elements["main_vstack"]:
            demo_label = ui.Label(label_text)
            subtask_label = ui.Label("")
            instruction_display.set_labels(subtask_label, demo_label)

    return instruction_display


def process_success_condition(env: gym.Env, success_term: object | None, success_step_count: int) -> tuple[int, bool]:
    """Process the success condition for the current step.

    Checks if the environment has met the success condition for the required
    number of consecutive steps. Marks the episode as successful if criteria are met.

    Args:
        env: The environment instance to check
        success_term: The success termination object or None if not available
        success_step_count: Current count of consecutive successful steps

    Returns:
        tuple[int, bool]: A tuple containing:
            - updated success_step_count: The updated count of consecutive successful steps
            - success_reset_needed: Boolean indicating if reset is needed due to success
    """
    if success_term is None:
        return success_step_count, False

    if bool(success_term.func(env, **success_term.params)[0]):
        success_step_count += 1
        if success_step_count >= args_cli.num_success_steps:
            env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
            env.recorder_manager.set_success_to_episodes(
                [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
            )
            env.recorder_manager.export_episodes([0])
            print("Success condition met! Recording completed.")
            return success_step_count, True
    else:
        success_step_count = 0

    return success_step_count, False


def _reset_sim_buffer(env: gym.Env) -> bool:
    """Return whether to rebuild the simulator before a recording attempt.

    With ``--ovxr`` this follows the replay agent: kitless physics keeps its solver and the stage
    leased to the XR render worker.
    """
    return args_cli.reset_sim_buffer_each_episode and (
        not args_cli.using_ovxr or requires_hard_sim_reset(env.scene.physics_backend)
    )


def handle_reset(
    env: gym.Env,
    success_step_count: int,
    instruction_display: "InstructionDisplay",
    label_text: str,
    teleop_interface: object | None = None,
) -> int:
    """Handle resetting the environment.

    Resets the environment, recorder manager, teleop device, and related
    state variables.  Updates the instruction display with current status.

    Args:
        env: The environment instance to reset.
        success_step_count: Current count of consecutive successful steps.
        instruction_display: The display object to update.
        label_text: Text to display showing current recording status.
        teleop_interface: Optional teleop device to reset (resets XR anchor
            and retargeter cross-step state).

    Returns:
        Reset success step count (0).
    """
    print("Resetting environment...")
    if _reset_sim_buffer(env):
        env.sim.reset()
    env.recorder_manager.reset()
    env.reset()
    if teleop_interface is not None and hasattr(teleop_interface, "reset"):
        teleop_interface.reset()
    if teleop_interface is not None and hasattr(teleop_interface, "notify_reset"):
        # An operator RESET was already processed by the control channel, so reset() cannot
        # reset retargeters. This does, and withholds actions computed before env.reset().
        teleop_interface.notify_reset()
    success_step_count = 0
    instruction_display.show_demo(label_text)
    return success_step_count


def notify_teleop_of_auto_reset(step_result: tuple, teleop_interface: object) -> bool:
    """Report a reset that environment 0 performed inside ``env.step`` to the teleop input.

    The teleop anchor and target frame follow environment 0, so only its termination or truncation
    restarts the input pipeline. The environment is not reset again.

    Args:
        step_result: The ``(obs, reward, terminated, truncated, extras)`` tuple from ``env.step``.
        teleop_interface: The teleop device. Devices without ``notify_reset`` are left unchanged.

    Returns:
        Whether the device was notified.
    """
    _, _, terminated, truncated, _ = step_result
    if not hasattr(teleop_interface, "notify_reset") or not bool(terminated[0] or truncated[0]):
        return False
    teleop_interface.notify_reset()
    return True


def run_simulation_loop(  # noqa: C901
    env: gym.Env,
    success_term: object | None,
    rate_limiter: RateLimiter | None,
    camera_feed_session: "XrCameraFeedSession",
    teleop_cfg: IsaacTeleopCfg | DeviceCfg,
    *,
    presentation: OvxrPresentation | None = None,
    shutdown_signals: OvxrShutdownSignals | None = None,
) -> int:
    """Run the main simulation loop for collecting demonstrations.

    Sets up callback functions for teleop device, initializes the UI,
    and runs the main loop that processes user inputs and environment steps.
    Records demonstrations when success conditions are met.

    Args:
        env: The environment instance
        success_term: The success termination object or None if not available
        rate_limiter: Optional rate limiter to control simulation speed
        camera_feed_session: Shared XR camera-feed lifecycle
        teleop_cfg: The resolved Capture or native device configuration.
        presentation: The OVXR presentation with ``--ovxr``. The caller closes it.
        shutdown_signals: Signal owner installed for ``presentation``. The caller restores it.

    Returns:
        int: Number of successful demonstrations recorded
    """
    use_isaac_teleop = isinstance(teleop_cfg, IsaacTeleopCfg)
    current_recorded_demo_count = 0
    success_step_count = 0
    should_reset_recording_instance = False
    # For Isaac Capture or XR, default to inactive until START is triggered. Without
    # --xr or --ovxr, recording is started locally (see ``request_start`` below) instead
    # of by a headset; it flows through the same state machine so keyboard/host
    # pause/resume keeps working.
    running_recording_instance = not (args_cli.xr or args_cli.using_ovxr or use_isaac_teleop)

    # Callback closures for the teleop device
    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        if success_step_count > 0:
            print(
                "Manual reset ignored. Success has fired and post-success steps are still recording. Please wait for"
                " the automatic reset."
            )
            return
        should_reset_recording_instance = True
        print("Recording instance reset requested")

    def start_recording_instance():
        nonlocal running_recording_instance
        running_recording_instance = True
        print("Recording started")

    def stop_recording_instance():
        nonlocal running_recording_instance
        running_recording_instance = False
        print("Recording paused")

    # Set up teleoperation callbacks.  For Isaac Capture the primary control
    # path is poll_control_events(); these callbacks are bridged automatically
    # and also serve native (keyboard / spacemouse) devices.
    teleoperation_callbacks = {
        "R": reset_recording_instance,
        "START": start_recording_instance,
        "STOP": stop_recording_instance,
        "RESET": reset_recording_instance,
    }

    try:
        teleop_interface = create_teleop_device(
            teleop_cfg,
            env_prim_path=env.scene.env_prim_paths[0],
            sim_device=args_cli.device,
            callbacks=teleoperation_callbacks,
            visualizers=env.sim.visualizers,
            # The OVXR session owns CloudXR and lends its OpenXR handles to the input device.
            cloudxr_env_file=(None if args_cli.using_ovxr else args_cli.cloudxr_env),
            auto_launch_cloudxr=args_cli.auto_launch_cloudxr and not args_cli.using_ovxr,
            use_kit_xr_bridge=args_cli.xr,
            external_oxr_handles_provider=(
                presentation.session.external_oxr_handles if presentation is not None else None
            ),
            mcap_record_path=args_cli.mcap_record_path,
            enable_debug_visualization=args_cli.enable_debug_visualization,
            haptic_cfg=getattr(env.cfg, "haptic_feedback", None),
        )
    except Exception:
        # Cleanup keeps this error; the command must not report success.
        logger.exception("Failed to create teleop device")
        raise
    if presentation is not None:
        presentation.input_device = teleop_interface
    if use_isaac_teleop and args_cli.mcap_record_path is not None:
        logger.info("Recording live Isaac Capture session to MCAP (debug-only): %s", args_cli.mcap_record_path)

    haptic_driver = None
    if use_isaac_teleop:
        from .. import create_haptic_feedback_driver

        haptic_driver = create_haptic_feedback_driver(env.unwrapped, teleop_interface, env.cfg)

    require_initial_bimanual_tracking = teleop_cfg.require_initial_bimanual_hand_tracking if use_isaac_teleop else False
    initialized_hand_tracking: set[str] = set()
    skipped_tracking_frames = 0
    action_conditioner = teleop_cfg.action_conditioner if use_isaac_teleop else None
    action_validator = teleop_cfg.action_validator if use_isaac_teleop else None
    action_state: dict = {}

    if use_isaac_teleop and not args_cli.using_ovxr:
        # the device reads these from the focused visualizer window, which OVXR does not open
        print("Isaac Capture control keys: [B] start/resume  [P] pause  [R] reset")

    label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."
    instruction_display = setup_ui(label_text, env)

    def ovxr_shutdown_requested() -> bool:
        return presentation is not None and presentation.session.shutdown_requested

    def pace_ovxr(iteration_started_s: float, action_applied: bool) -> None:
        """Pace OVXR host control as the live agent does; other paths keep their own pacing."""
        if presentation is not None:
            pace_ovxr_control_loop(
                True,
                action_applied,
                iteration_started_s=iteration_started_s,
                action_period_s=env.step_dt,
                step_hz=args_cli.ovxr_step_hz,
                wait_for_shutdown=presentation.session.wait_for_shutdown,
                clock=time.perf_counter,
            )

    def inner_loop():  # noqa: C901
        """Inner loop function with access to nonlocal variables."""
        nonlocal current_recorded_demo_count, success_step_count, should_reset_recording_instance
        nonlocal running_recording_instance, label_text, skipped_tracking_frames

        # Reset before starting
        if _reset_sim_buffer(env):
            env.sim.reset()
        env.reset()
        teleop_interface.reset()
        initialized_hand_tracking.clear()
        action_state.clear()
        if presentation is not None:
            presentation.mark_changed(env, reset=True)

        # Without --xr or --ovxr there is no headset to send START, so drive the Isaac Capture
        # state machine to RUNNING locally ([B]/[P] can still pause/resume). The reset()
        # above is a host reset (a pure pulse), so it does not cancel this start.
        if use_isaac_teleop and not (args_cli.xr or args_cli.using_ovxr):
            teleop_interface.request_start()

        subtasks = {}
        stack_name = "Isaac Capture" if use_isaac_teleop else "native"
        print(f"{stack_name} recording started.")

        if use_isaac_teleop:
            from .. import poll_control_events

        from ._instruction_display import show_subtask_instructions

        with contextlib.suppress(KeyboardInterrupt), torch.inference_mode(), camera_feed_session.bind(env):
            while env.sim.is_running() and not ovxr_shutdown_requested():
                iteration_started_s = time.perf_counter() if presentation is not None else 0.0
                # Reset invalidates the tracking anchor. Publish the new epoch
                # before reading input, retaining force across deferred admissions.
                if not admit_reset_publication(env, presentation):
                    continue
                # Get teleop command (may be None while waiting for session start)
                target_frame_prim_path = (
                    teleop_interface.target_frame_prim_path if presentation is not None and use_isaac_teleop else None
                )
                if target_frame_prim_path is None:
                    action = teleop_interface.advance()
                else:
                    target_T_world = presentation.session.get_target_frame_T_world(target_frame_prim_path)
                    action = teleop_interface.advance(target_T_world=target_T_world)

                if use_isaac_teleop:
                    ctrl = poll_control_events(teleop_interface)
                    if ctrl.is_active is not None:
                        running_recording_instance = ctrl.is_active
                    if ctrl.should_reset:
                        if success_step_count > 0:
                            # same guard as the R callback: keep a demo whose success is still being recorded
                            reset_recording_instance()
                        else:
                            should_reset_recording_instance = True

                if not running_recording_instance:
                    action_state.clear()

                if action is None:
                    update_idle_presentation(env, presentation)
                    if haptic_driver is not None:
                        haptic_driver.stop()
                    pace_ovxr(iteration_started_s, action_applied=False)
                    continue

                # Apply the task's action policy before stepping, as the live agent does.
                issue = None
                if running_recording_instance:
                    issue = initial_hand_tracking_issue(
                        teleop_interface, initialized_hand_tracking, required=require_initial_bimanual_tracking
                    )
                    if issue is None:
                        action, issue = prepare_teleop_action(
                            env, action, action_conditioner, action_validator, action_state
                        )

                # Perform action on environment
                action_applied = running_recording_instance and issue is None
                auto_reset = False
                if action_applied:
                    # Expand to batch dimension and compute actions based on environment
                    obv = env.step(action.repeat(env.num_envs, 1))
                    auto_reset = notify_teleop_of_auto_reset(obv, teleop_interface)
                    if auto_reset:
                        camera_feed_session.refresh()
                    # render controller haptics from post-step contact forces
                    if haptic_driver is not None:
                        haptic_driver.update()
                    if subtasks is not None:
                        if subtasks == {}:
                            subtasks = obv[0].get("subtask_terms")
                        elif subtasks:
                            show_subtask_instructions(instruction_display, subtasks, obv, env.cfg)
                elif issue is not None:
                    hold_teleop_frame(env, haptic_driver, render=presentation is None)
                    skipped_tracking_frames += 1
                    warn_rejected_teleop_frame(issue, skipped_tracking_frames)
                else:
                    # The publication after this frame reports its host timing.
                    update_idle_presentation(env, presentation, report_timing=False)
                    # not stepping: zero haptics so a paused grip stops buzzing
                    if haptic_driver is not None:
                        haptic_driver.stop()

                # Check for success condition
                success_step_count_new, success_reset_needed = process_success_condition(
                    env, success_term, success_step_count
                )
                success_step_count = success_step_count_new
                if success_reset_needed:
                    should_reset_recording_instance = True

                # Update demo count if it has changed
                if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                    current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                    label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."
                    print(label_text)

                # Check if we've reached the desired number of demos
                if (
                    args_cli.num_demos > 0
                    and env.recorder_manager.exported_successful_episode_count >= args_cli.num_demos
                ):
                    label_text = f"All {current_recorded_demo_count} demonstrations recorded.\nExiting the app."
                    instruction_display.show_demo(label_text)
                    print(label_text)
                    target_time = time.time() + 0.8
                    while time.time() < target_time and not ovxr_shutdown_requested():
                        if rate_limiter:
                            rate_limiter.sleep(env)
                        else:
                            iteration_started_s = time.perf_counter() if presentation is not None else 0.0
                            update_idle_presentation(env, presentation)
                            pace_ovxr(iteration_started_s, action_applied=False)
                    break

                # Handle reset if requested
                reset_applied = should_reset_recording_instance or auto_reset
                if should_reset_recording_instance:
                    success_step_count = handle_reset(
                        env, success_step_count, instruction_display, label_text, teleop_interface
                    )
                    camera_feed_session.refresh()
                    should_reset_recording_instance = False
                if reset_applied:
                    initialized_hand_tracking.clear()
                    action_state.clear()

                if presentation is not None:
                    presentation.publish(env, changed=action_applied, reset=reset_applied)

                # Check if simulation is stopped
                if env.sim.is_stopped():
                    break

                # Rate limiting
                if rate_limiter:
                    rate_limiter.sleep(env)
                pace_ovxr(iteration_started_s, action_applied)

    # Install after environment/device initialization because embedded native
    # libraries may replace inherited signal dispositions during startup.
    if presentation is not None:
        shutdown_signals.install(presentation)
    # Run the loop with or without context manager based on stack
    if use_isaac_teleop:
        with teleop_interface:
            # Capture releases borrowed handles before the caller closes OVXR and then the environment.
            if presentation is None or presentation.start(
                env,
                timeout_s=RENDERER_ATTACH_TIMEOUT_S,
                required_openxr_extensions=teleop_interface.required_openxr_extensions,
            ):
                inner_loop()
    else:
        inner_loop()

    return current_recorded_demo_count


def main() -> None:
    """Collect demonstrations from the environment using teleop interfaces.

    Main function that orchestrates the entire process:
    1. Sets up rate limiting based on configuration
    2. Creates output directories for saving demonstrations
    3. Configures the environment
    4. Runs the simulation loop to collect demonstrations
    5. Cleans up resources when done

    Raises:
        Exception: Propagates exceptions from any of the called functions
    """
    # Set up output directories
    output_dir, output_file_name = setup_output_directories()

    # Create and configure environment
    env_cfg, success_term, teleop_cfg, selected_cameras = create_environment_config(output_dir, output_file_name)

    with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
        record_demos(env_cfg, success_term, teleop_cfg, cleanup, selected_cameras=selected_cameras)


def record_demos(
    env_cfg: ManagerBasedRLEnvCfg,
    success_term: object | None,
    teleop_cfg: IsaacTeleopCfg | DeviceCfg,
    cleanup: contextlib.ExitStack,
    *,
    selected_cameras: tuple[OvLibrariesXrCameraFeedCfg, ...] = (),
) -> None:
    """Create the environment and record demonstrations until the session ends."""
    from .. import XrCameraFeedSession

    use_isaac_teleop = isinstance(teleop_cfg, IsaacTeleopCfg)

    # Apply the RTX/DLSS global settings when Kit runs an RTX render pipeline (Kit visualizer,
    # external cameras, or XR). OVXR renders without Kit.
    if (
        not args_cli.using_ovxr
        and has_kit()
        and (not args_cli.disable_external_cameras or args_cli.kit_visualizer or args_cli.xr)
    ):
        from isaaclab_isaacsim.renderers.isaac_rtx_renderer_utils import apply_isaac_rtx_global_settings

        apply_isaac_rtx_global_settings(
            IsaacRtxRendererGlobalSettingsCfg(antialiasing_mode="DLSS"),
        )

    camera_feed_session = XrCameraFeedSession.prepare(
        env_cfg,
        enabled=args_cli.xr and use_isaac_teleop,
        camera_rendering_enabled=not args_cli.disable_external_cameras and not args_cli.using_ovxr,
    )
    if args_cli.using_ovxr:
        current_recorded_demo_count = _record_ovxr_demos(
            env_cfg, success_term, teleop_cfg, camera_feed_session, selected_cameras
        )
        print(f"Recording session completed with {current_recorded_demo_count} successful demonstrations")
        print(f"Demonstrations saved to: {args_cli.dataset_file}")
        return
    cleanup.callback(camera_feed_session.close)
    if camera_feed_session.requires_responsive_denoising:
        from isaaclab_isaacsim.renderers.isaac_rtx_renderer_utils import apply_isaac_rtx_global_settings

        apply_isaac_rtx_global_settings(
            IsaacRtxRendererGlobalSettingsCfg(
                carb_settings={"/rtx/dldenoiser/responsiveDenoising": True},
            )
        )

    # With --xr, rate limiting is achieved via OpenXR and the XR visualization
    # manager is installed. Without --xr (including standalone Isaac Capture I/O),
    # fall back to the software rate limiter and skip the XR viz stack.
    if args_cli.xr:
        rate_limiter = None
        from isaaclab.ui.xr_widgets import TeleopVisualizationManager, XRVisualization

        # Assign the teleop visualization manager to the visualization system
        XRVisualization.assign_manager(TeleopVisualizationManager)
    else:
        rate_limiter = RateLimiter(args_cli.step_hz)

    # Create environment
    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        exit(1)

    # Run simulation loop
    current_recorded_demo_count = run_simulation_loop(env, success_term, rate_limiter, camera_feed_session, teleop_cfg)

    # Clean up
    env.close()
    print(f"Recording session completed with {current_recorded_demo_count} successful demonstrations")
    print(f"Demonstrations saved to: {args_cli.dataset_file}")


def _record_ovxr_demos(
    env_cfg: ManagerBasedRLEnvCfg,
    success_term: object | None,
    teleop_cfg: IsaacTeleopCfg,
    camera_feed_session: "XrCameraFeedSession",
    selected_cameras: tuple[OvLibrariesXrCameraFeedCfg, ...],
) -> int:
    """Record through an OVXR session prepared before the environment and drained before its release.

    OVXR paces host control instead of :class:`RateLimiter`, and the Kit XR visualization manager is not used.

    Returns:
        The number of successful demonstrations recorded.
    """
    from isaaclab.sim import SimulationContext

    previous_context = SimulationContext.instance()
    env = None
    presentation = None
    shutdown_signals = OvxrShutdownSignals()
    try:
        presentation = OvxrPresentation.prepare(
            env_cfg,
            args_cli,
            application_name=f"isaaclab-ovxr-record-{args_cli.task}",
            camera_feeds=selected_cameras,
            # Selected panels already carry this configuration from launch resolution.
            renderer_cfg=None if selected_cameras else ovxr_renderer_cfg(args_cli.ovrtx_active_cuda_gpus),
        )
        try:
            env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        except Exception:
            # Cleanup below keeps this error; the command must not report success.
            logger.exception("Failed to create environment")
            raise
        return run_simulation_loop(
            env,
            success_term,
            None,
            camera_feed_session,
            teleop_cfg,
            presentation=presentation,
            shutdown_signals=shutdown_signals,
        )
    finally:
        shutdown_signals.shutting_down = True
        close_teleop_resources(
            env,
            camera_feed_session,
            presentation,
            shutdown_signals.previous_handlers,
            presentation.input_device if presentation is not None else None,
            previous_context=previous_context,
        )


def run_record_demos_cli(argv: list[str] | None = None) -> int:
    """Record teleoperated demonstrations from the command line.

    Args:
        argv: Command-line arguments, excluding the program name. Defaults to ``sys.argv[1:]``.
            ``sys.argv`` is restored on return.

    Returns:
        The process exit code, 0 once recording ends.
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
