# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch checks shared by the live and replay OVXR entrypoints."""

from __future__ import annotations

import argparse
import math
from typing import Any


def require_kitless_physics(env_cfg: Any) -> None:
    """Reject a task whose physics selection needs the Kit runtime before an OVXR launch.

    A task that leaves ``sim.physics`` unset selects Isaac Sim PhysX, which needs Kit.
    The kitless OVXR session cannot run it, so fail before launch with the selector to use.

    Args:
        env_cfg: Resolved environment configuration.

    Raises:
        ValueError: If ``env_cfg.sim.physics`` is ``None``.
    """
    if env_cfg.sim.physics is None:
        raise ValueError(
            "--ovxr requires a kitless physics preset, but the task leaves sim.physics unset (Isaac Sim"
            " PhysX through Kit). Select one, for example physics=ovphysx or physics=newton_mjwarp."
        )


def add_ovxr_presentation_args(parser: argparse.ArgumentParser) -> None:
    """Add the OVXR camera-panel and desktop-mirror options shared by live teleoperation and replay."""
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
        help="Place the application camera quads in the OpenXR VIEW space, overriding the task layout.",
    )
    parser.add_argument(
        "--ovxr_pip_view_position",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Optional VIEW-space camera panel center in meters; requires --ovxr_pip_head_locked.",
    )


def validate_ovxr_presentation_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace, *, using_ovxr: bool
) -> None:
    """Validate the shared OVXR presentation options and resolve the desktop-mirror default.

    Args:
        parser: Parser used to report errors.
        args: Parsed namespace; ``ovxr_desktop_mirror`` is set to the resolved selection.
        using_ovxr: Whether the command selected ``--ovxr``.
    """
    if args.ovxr_desktop_mirror and not using_ovxr:
        parser.error("--ovxr_desktop_mirror requires --ovxr")
    if args.ovxr_pip_camera is not None and not using_ovxr:
        parser.error("--ovxr_pip_camera requires --ovxr")
    if (args.ovxr_pip_head_locked or args.ovxr_pip_view_position is not None) and not using_ovxr:
        parser.error("OVXR presentation options require --ovxr")
    args.ovxr_desktop_mirror = using_ovxr and not args.headless
    if args.ovxr_pip_camera is not None and getattr(args, "disable_external_cameras", False):
        parser.error("--ovxr_pip_camera cannot be combined with --disable_external_cameras")
    if not math.isfinite(args.ovxr_pip_fps) or args.ovxr_pip_fps <= 0.0:
        parser.error("--ovxr_pip_fps must be finite and positive")
    if args.ovxr_pip_view_position is not None:
        if not args.ovxr_pip_head_locked or args.ovxr_pip_camera is None:
            parser.error("--ovxr_pip_view_position requires a head-locked PiP camera")
        if len(args.ovxr_pip_camera) > 1:
            parser.error("A PiP position override requires one camera; multiple panels use task layout")
        if not all(math.isfinite(value) for value in args.ovxr_pip_view_position):
            parser.error("--ovxr_pip_view_position must be finite")
    if not math.isfinite(args.ovxr_desktop_mirror_fps) or args.ovxr_desktop_mirror_fps <= 0:
        parser.error("--ovxr_desktop_mirror_fps must be finite and positive")


def ovxr_camera_layout(env_cfg: Any, head_locked: bool) -> Any:
    """Return the task's camera-panel layout, forced to head-locked placement when requested."""
    from ..isaac_teleop_cfg import XrCameraFeedLayoutCfg

    layout = env_cfg.isaac_teleop.xr_camera_feed_layout
    if head_locked:
        layout = (layout or XrCameraFeedLayoutCfg()).replace(placement="head_locked")
    return layout
