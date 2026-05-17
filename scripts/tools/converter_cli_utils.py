# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared CLI helpers for asset converter scripts."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from isaaclab.sim.converters._importer_api import _is_standalone_importer_package_available

_IMPORTER_RUNTIME_MODULES = {
    "mjcf": "mujoco_usd_converter",
    "urdf": "urdf_usd_converter",
}


def parse_visualizer_csv(value: str) -> list[str]:
    """Parse visualizer list from a single comma-delimited CLI token."""
    valid = {"kit", "newton", "none", "rerun", "viser"}
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
        raise argparse.ArgumentTypeError(
            f"Invalid --visualizer value(s): {', '.join(invalid)}. Valid options: {', '.join(sorted(valid))}."
        )
    return list(dict.fromkeys(names))


def parse_converter_cli_args(parser: argparse.ArgumentParser) -> tuple[argparse.Namespace, object | None]:
    """Parse converter CLI arguments and launch Kit only when requested or required.

    Args:
        parser: The converter-specific argument parser.

    Returns:
        Parsed CLI arguments and an optional launched ``SimulationApp``.
    """
    standalone_importer_available = _is_standalone_importer_package_available()
    launch_kit = not standalone_importer_available or _should_launch_kit(_preparse_app_args())

    if launch_kit and _is_help_requested() and not standalone_importer_available:
        _add_kitless_app_launcher_args(parser)
        return parser.parse_args(), None

    if launch_kit:
        try:
            from isaaclab.app import AppLauncher
        except ImportError as exc:
            raise ImportError(
                "Launching Omniverse Kit requires the full Isaac Sim package. Omit '--viz kit' for kitless "
                "conversion with the 'isaacsim-asset-isolated' package."
            ) from exc
        AppLauncher.add_app_launcher_args(parser)
        args_cli = parser.parse_args()
        return args_cli, AppLauncher(args_cli).app

    _add_kitless_app_launcher_args(parser)
    return parser.parse_args(), None


def ensure_standalone_importer_runtime(importer_kind: str) -> None:
    """Verify that the standalone importer runtime can load without aborting Python.

    Args:
        importer_kind: The importer runtime to check.

    Raises:
        RuntimeError: If the standalone runtime cannot be imported in a subprocess.
    """
    runtime_module = _IMPORTER_RUNTIME_MODULES[importer_kind]
    try:
        result = subprocess.run(
            [sys.executable, "-c", f"import usdex.core; import {runtime_module}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"The standalone {importer_kind.upper()} importer runtime check timed out after {exc.timeout} seconds."
        ) from exc
    if result.returncode == 0:
        return

    details = (result.stderr or result.stdout).strip()
    raise RuntimeError(
        f"The standalone {importer_kind.upper()} importer runtime is installed but cannot load in this environment. "
        "This usually means the OpenUSD Python bindings from 'usd-core' and 'usd-exchange' are mismatched. "
        "Reinstall 'usd-exchange' after installing IsaacLab and 'isaacsim-asset-isolated', or use a kitless "
        "environment without 'usd-core'."
        f"\nRuntime check exited with code {result.returncode}:\n{details}"
    )


def should_open_stage_with_kit(args_cli: argparse.Namespace) -> bool:
    """Return True when the converter should open the generated stage in Kit."""
    visualizers = getattr(args_cli, "visualizer", None) or []
    if isinstance(visualizers, str):
        visualizers = parse_visualizer_csv(visualizers)
    livestream = getattr(args_cli, "livestream", -1)
    if livestream < 0:
        livestream = _read_int_env("LIVESTREAM", 0)
    return ("kit" in visualizers and not getattr(args_cli, "headless", False)) or livestream in {1, 2}


def _add_kitless_app_launcher_args(parser: argparse.ArgumentParser) -> None:
    """Add AppLauncher-compatible arguments accepted during kitless conversion."""
    arg_group = parser.add_argument_group(
        "app_launcher arguments",
        description="Arguments accepted for compatibility. Kit-only options require full Isaac Sim.",
    )
    arg_group.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help=(
            "[DEPRECATED] Disable visualizers and force headless mode. Conversion is headless by default unless "
            "'--viz kit' is passed."
        ),
    )
    arg_group.add_argument(
        "--livestream",
        type=int,
        default=-1,
        choices={0, 1, 2},
        help="Force enable livestreaming. Values 1 and 2 require full Isaac Sim.",
    )
    arg_group.add_argument(
        "--enable_cameras",
        action="store_true",
        default=False,
        help="Enable camera sensors and relevant extension dependencies. Requires full Isaac Sim.",
    )
    arg_group.add_argument("--xr", action="store_true", default=False, help="Enable XR mode. Requires full Isaac Sim.")
    arg_group.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help='Accepted for compatibility with AppLauncher. Can be "cpu", "cuda", or "cuda:N".',
    )
    arg_group.add_argument(
        "--visualizer",
        "--viz",
        type=parse_visualizer_csv,
        default=None,
        help="Visualizer backends to enable as CSV. Use 'kit' to open Omniverse Kit after conversion.",
    )
    arg_group.add_argument("--verbose", action="store_true", help="Accepted for compatibility with AppLauncher.")
    arg_group.add_argument("--info", action="store_true", help="Accepted for compatibility with AppLauncher.")
    arg_group.add_argument(
        "--experience",
        type=str,
        default="",
        help="The experience file to load when launching Kit. Requires full Isaac Sim.",
    )
    arg_group.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="Accepted for compatibility with AppLauncher.",
    )
    arg_group.add_argument(
        "--rendering_mode",
        type=str,
        choices={"performance", "balanced", "quality"},
        default="balanced",
        help="Accepted for compatibility with AppLauncher.",
    )
    arg_group.add_argument(
        "--kit_args",
        type=str,
        default="",
        help="Command line arguments for Omniverse Kit. Requires full Isaac Sim.",
    )
    arg_group.add_argument(
        "--anim_recording_enabled",
        action="store_true",
        default=False,
        help="Enable USD animation recording. Requires full Isaac Sim.",
    )
    arg_group.add_argument("--anim_recording_start_time", type=float, default=0)
    arg_group.add_argument("--anim_recording_stop_time", type=float, default=10)
    arg_group.add_argument("--max_visible_envs", type=int, default=None)


def _preparse_app_args() -> argparse.Namespace:
    """Pre-parse AppLauncher arguments that determine whether Kit is required."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--visualizer", "--viz", type=parse_visualizer_csv, default=None)
    parser.add_argument("--livestream", type=int, default=-1, choices={0, 1, 2})
    parser.add_argument("--enable_cameras", action="store_true", default=False)
    parser.add_argument("--experience", type=str, default="")
    parser.add_argument("--kit_args", type=str, default="")
    parser.add_argument("--xr", action="store_true", default=False)
    parser.add_argument("--anim_recording_enabled", action="store_true", default=False)
    return parser.parse_known_args()[0]


def _should_launch_kit(app_args: argparse.Namespace) -> bool:
    """Return True when CLI arguments or environment variables request Kit."""
    visualizers = app_args.visualizer or []
    livestream = app_args.livestream if app_args.livestream >= 0 else _read_int_env("LIVESTREAM", 0)
    enable_cameras = app_args.enable_cameras or _read_int_env("ENABLE_CAMERAS", 0) == 1
    xr = app_args.xr or _read_int_env("XR", 0) == 1
    return (
        "kit" in visualizers
        or livestream in {1, 2}
        or enable_cameras
        or bool(app_args.experience)
        or bool(app_args.kit_args)
        or xr
        or app_args.anim_recording_enabled
    )


def _read_int_env(name: str, default: int) -> int:
    """Read an integer environment variable with a default fallback."""
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _is_help_requested() -> bool:
    """Return True if the current command only needs argparse help."""
    return "-h" in sys.argv[1:] or "--help" in sys.argv[1:]
