# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Argparse helpers for configuring Isaac Lab simulation launch options."""

from __future__ import annotations

import argparse
import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "APPLAUNCHER_CFG_INFO",
    "SIM_APP_CFG_TYPES",
    "ExplicitAction",
    "ExplicitTrueAction",
    "add_app_launcher_args",
    "check_argparser_config_params",
    "parse_visualizer_csv",
]


class ExplicitAction(argparse.Action):
    """Custom action to track if an argument was explicitly passed by the user."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, f"{self.dest}_explicit", True)


class ExplicitTrueAction(argparse.Action):
    """Custom action to track explicit use of boolean flags."""

    def __init__(self, option_strings, dest, default=False, required=False, help=None):
        super().__init__(
            option_strings=option_strings, dest=dest, nargs=0, default=default, required=required, help=help
        )

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, True)
        setattr(namespace, f"{self.dest}_explicit", True)


APPLAUNCHER_CFG_INFO: dict[str, tuple[list[type], Any]] = {
    "headless": ([bool], False),
    "livestream": ([int], -1),
    "enable_cameras": ([bool], False),
    "xr": ([bool], False),
    "device": ([str], "cuda:0"),
    "experience": ([str], ""),
    "deterministic": ([bool], False),
    "rendering_mode": ([str], "balanced"),
    "max_visible_envs": ([int, type(None)], None),
}
"""Arguments added by :func:`add_app_launcher_args` with expected types and defaults."""

SIM_APP_CFG_TYPES: dict[str, list[type]] = {
    "headless": [bool],
    "hide_ui": [bool, type(None)],
    "active_gpu": [int, type(None)],
    "physics_gpu": [int],
    "multi_gpu": [bool],
    "sync_loads": [bool],
    "width": [int],
    "height": [int],
    "window_width": [int],
    "window_height": [int],
    "display_options": [int],
    "subdiv_refinement_level": [int],
    "renderer": [str],
    "anti_aliasing": [int],
    "samples_per_pixel_per_frame": [int],
    "denoiser": [bool],
    "max_bounces": [int],
    "max_specular_transmission_bounces": [int],
    "max_volume_bounces": [int],
    "open_usd": [str, type(None)],
    "livesync_usd": [str, type(None)],
    "fast_shutdown": [bool],
    "experience": [str],
}
"""Known SimulationApp config argument types used for parser conflict checks."""


def _parse_visualizer_csv(value: str) -> list[str] | None:
    """Parse visualizer list from a single comma-delimited CLI token."""
    valid = {"kit", "newton", "rerun", "viser", "none"}
    token = (value or "").strip()
    if not token:
        raise argparse.ArgumentTypeError(
            "Invalid --visualizer value: empty string. Use a comma-separated list, e.g. --viz kit,newton."
        )
    if " " in token:
        raise argparse.ArgumentTypeError(
            "Invalid --visualizer value: spaces are not allowed. "
            "Use a comma-separated list without spaces, e.g. --viz kit,newton,rerun,viser."
        )

    names = [item.strip().lower() for item in token.split(",")]
    if any(not name for name in names):
        raise argparse.ArgumentTypeError(
            "Invalid --visualizer value: empty visualizer entry detected. "
            "Use a comma-separated list without empty items."
        )
    invalid = [name for name in names if name not in valid]
    if invalid:
        invalid_names = ", ".join(invalid)
        valid_names = ", ".join(sorted(valid))
        raise argparse.ArgumentTypeError(
            f"Invalid --visualizer value(s): {invalid_names}. Valid options: {valid_names}."
        )
    if "none" in names:
        if len(names) > 1:
            raise argparse.ArgumentTypeError(
                "Invalid --visualizer value: 'none' cannot be combined with other visualizer types."
            )
        return None
    # De-duplicate while preserving order.
    return list(dict.fromkeys(names))


parse_visualizer_csv = _parse_visualizer_csv


def check_argparser_config_params(config: dict) -> None:
    """Check parser parameters for AppLauncher and SimulationApp name conflicts."""
    applauncher_keys = set(APPLAUNCHER_CFG_INFO.keys())
    for key in config:
        if key in applauncher_keys:
            raise ValueError(
                f"The passed ArgParser object already has the field {key!r}. This field will be added by "
                "`AppLauncher.add_app_launcher_args()`, and should not be added directly. Please remove the "
                "argument or rename it to a non-conflicting name."
            )

    simulationapp_keys = set(SIM_APP_CFG_TYPES.keys())
    for key, value in config.items():
        if key in simulationapp_keys:
            given_type = type(value)
            expected_types = SIM_APP_CFG_TYPES[key]
            if given_type not in set(expected_types):
                raise ValueError(
                    f"Invalid value type for the argument {key!r}: {given_type}. Expected one of "
                    f"{expected_types}, if intended to be ingested by the SimulationApp object. Please "
                    "change the type if this intended for the SimulationApp or change the name of the "
                    "argument to avoid name conflicts."
                )
            logger.info("The argument %s will be used to configure the SimulationApp.", key)


def add_app_launcher_args(parser: argparse.ArgumentParser) -> None:
    """Add AppLauncher command-line arguments to an existing argument parser."""
    parser_help = None
    if len(parser._actions) > 0 and isinstance(parser._actions[0], argparse._HelpAction):  # type: ignore
        parser_help = parser._actions[0]
        parser._option_string_actions.pop("-h")
        parser._option_string_actions.pop("--help")

    known, _ = parser.parse_known_args()
    config = vars(known)
    if len(config) == 0:
        logger.warning(
            "[WARN][AppLauncher]: There are no arguments attached to the ArgumentParser object. "
            "If you have your own arguments, please load your own arguments before calling the "
            "`AppLauncher.add_app_launcher_args` method. This allows the method to check the validity "
            "of the arguments and perform checks for argument names."
        )
    else:
        check_argparser_config_params(config)

    arg_group = parser.add_argument_group(
        "app_launcher arguments",
        description="Arguments for the AppLauncher. For more details, please check the documentation.",
    )
    arg_group.add_argument(
        "--headless",
        action=ExplicitTrueAction,
        default=APPLAUNCHER_CFG_INFO["headless"][1],
        help=(
            "[DEPRECATED] Disable visualizers and force headless mode (display off). "
            "Omit '--viz' for default headless, or use '--viz none' to force-disable visualizers."
        ),
    )
    arg_group.add_argument(
        "--livestream",
        type=int,
        default=APPLAUNCHER_CFG_INFO["livestream"][1],
        choices={0, 1, 2},
        help="Force enable livestreaming. Mapping corresponds to that for the `LIVESTREAM` environment variable.",
    )
    arg_group.add_argument(
        "--enable_cameras",
        action="store_true",
        default=APPLAUNCHER_CFG_INFO["enable_cameras"][1],
        help="Enable camera sensors and relevant extension dependencies.",
    )
    arg_group.add_argument(
        "--xr",
        action="store_true",
        default=APPLAUNCHER_CFG_INFO["xr"][1],
        help="Enable XR mode for VR/AR applications.",
    )
    arg_group.add_argument(
        "--device",
        type=str,
        action=ExplicitAction,
        default=APPLAUNCHER_CFG_INFO["device"][1],
        help="The device to run the simulation on. Can be \"cpu\", \"cuda\", \"cuda:N\", where N is the device ID",
    )
    arg_group.add_argument(
        "--visualizer",
        "--viz",
        type=parse_visualizer_csv,
        action=ExplicitAction,
        default=None,
        help="Visualizer backends to enable as CSV (e.g., kit,newton,rerun,viser).",
    )
    arg_group.add_argument("--cpu", action="store_true", help=argparse.SUPPRESS)
    arg_group.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose-level log output from the SimulationApp.",
    )
    arg_group.add_argument(
        "--info",
        action="store_true",
        help="Enable info-level log output from the SimulationApp.",
    )
    arg_group.add_argument(
        "--experience",
        type=str,
        default="",
        help=(
            "The experience file to load when launching the SimulationApp. If an empty string is provided, "
            "the experience file is determined based on the headless flag. If a relative path is provided, "
            "it is resolved relative to the `apps` folder in Isaac Sim and Isaac Lab (in that order)."
        ),
    )
    arg_group.add_argument(
        "--deterministic",
        action="store_true",
        default=APPLAUNCHER_CFG_INFO["deterministic"][1],
        help="After startup, apply RTX/RTPT settings for reproducible rendering (see AppLauncher docs).",
    )
    arg_group.add_argument(
        "--rendering_mode",
        type=str,
        action=ExplicitAction,
        choices={"performance", "balanced", "quality"},
        help=(
            "Sets the rendering mode. Preset settings files can be found in apps/rendering_modes. "
            "Can be \"performance\", \"balanced\", or \"quality\". Individual settings can be overwritten by using "
            "the RenderCfg class."
        ),
    )
    arg_group.add_argument(
        "--kit_args",
        type=str,
        default="",
        help=(
            "Command line arguments for Omniverse Kit as a string separated by a space delimiter. "
            "Example usage: --kit_args \"--ext-folder=/path/to/ext1 --ext-folder=/path/to/ext2\""
        ),
    )
    arg_group.add_argument(
        "--anim_recording_enabled",
        action="store_true",
        help="Enable recording time-sampled USD animations from IsaacLab PhysX simulations.",
    )
    arg_group.add_argument(
        "--anim_recording_start_time",
        type=float,
        default=0,
        help="Set time that animation recording begins playing. If not set, the recording will start from the beginning.",
    )
    arg_group.add_argument(
        "--anim_recording_stop_time",
        type=float,
        default=10,
        help="Set time that animation recording stops playing. If the process is shutdown before the stop time is exceeded, then the animation is not recorded.",
    )
    arg_group.add_argument(
        "--max_visible_envs",
        type=int,
        default=argparse.SUPPRESS,
        help="When set, caps the nums of envs shown in the launched visualizers.",
    )

    if parser_help is not None:
        parser._option_string_actions["-h"] = parser_help
        parser._option_string_actions["--help"] = parser_help
