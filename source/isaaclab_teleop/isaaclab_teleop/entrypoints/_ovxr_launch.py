# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch checks shared by the live and replay OVXR entrypoints."""

from __future__ import annotations

import argparse
import math
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ..xr_cfg import remove_camera_configs

if TYPE_CHECKING:
    from ..ovxr_cfg import OvLibrariesXrCameraCfg


def prepare_ovxr_launch(env_cfg: Any, task: str, num_envs: int, *, keep: tuple[str, ...] = ()) -> Any:
    """Check an OVXR launch before runtime startup and remove the task's camera sensors.

    Args:
        env_cfg: Resolved environment configuration.
        task: Task name, for the error message.
        num_envs: Requested number of environments.
        keep: Camera sensor names to retain for the OVXR camera panels.

    Returns:
        The configuration without camera sensors other than ``keep``.

    Raises:
        ValueError: If ``num_envs`` is not 1, ``ISAAC_LAB_OVRTX_USE_OVSTAGE`` is set to a value
            other than ``1``, or the task needs Kit physics (see :func:`require_kitless_physics`).
    """
    if num_envs != 1:
        raise ValueError("--ovxr currently requires --num_envs 1")
    if os.environ.get("ISAAC_LAB_OVRTX_USE_OVSTAGE", "1").strip() != "1":
        raise ValueError("--ovxr requires OVStage (ISAAC_LAB_OVRTX_USE_OVSTAGE=1)")
    require_kitless_physics(env_cfg, task)
    return remove_camera_configs(env_cfg, keep=keep)


def require_kitless_physics(env_cfg: Any, task: str) -> None:
    """Reject a task whose physics selection needs the Kit runtime before an OVXR launch.

    A task that leaves ``sim.physics`` unset selects Isaac Sim PhysX, which needs Kit.
    The kitless OVXR session cannot run it, so fail before launch. Such a task normally
    declares no physics preset, so the error points at ``--xr`` or the task configuration
    rather than at a ``physics=`` selector.

    Args:
        env_cfg: Resolved environment configuration.
        task: Task name, for the error message.

    Raises:
        ValueError: If ``env_cfg.sim.physics`` is ``None``.
    """
    if env_cfg.sim.physics is None:
        raise ValueError(
            f"--ovxr needs a task that selects kitless physics (OvPhysxCfg or NewtonCfg), but task '{task}' leaves"
            " sim.physics unset, which runs Isaac Sim PhysX through Kit. Run this task with --xr instead, or give"
            " its configuration a kitless physics preset."
        )


def add_ovxr_presentation_args(parser: argparse.ArgumentParser) -> None:
    """Add the OVXR camera-panel and desktop-mirror options shared by live teleoperation and replay."""
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
    """Validate the shared OVXR presentation options.

    Args:
        parser: Parser used to report errors.
        args: Parsed namespace.
        using_ovxr: Whether the command selected ``--ovxr``.
    """
    if args.ovxr_pip_camera is not None and not using_ovxr:
        parser.error("--ovxr_pip_camera requires --ovxr")
    if (args.ovxr_pip_head_locked or args.ovxr_pip_view_position is not None) and not using_ovxr:
        parser.error("OVXR presentation options require --ovxr")
    if args.headless and not using_ovxr:
        parser.error("--headless requires --ovxr; Kit runs headless unless --visualizer is given")
    if args.ovxr_pip_camera is not None and args.disable_external_cameras:
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


def resolve_ovxr_pip_cameras(
    env_cfg: Any, args: argparse.Namespace, renderer_cfg: Any
) -> tuple[OvLibrariesXrCameraCfg, ...]:
    """Select the requested PiP cameras and resolve their OVXR panels before launch.

    Selection assigns ``renderer_cfg`` to each selected camera, so the OVXR session and its panels
    share one OVRTX configuration. Resolution creates no native resources, so an invalid selection or
    task layout fails before the simulation launches.

    Args:
        env_cfg: Resolved environment configuration.
        args: Parsed namespace with the shared OVXR presentation options.
        renderer_cfg: OVRTX configuration shared by the OVXR session and its PiP cameras.

    Returns:
        The resolved panels in selection order, or an empty tuple without ``--ovxr_pip_camera``.

    Raises:
        ValueError: If a selected camera or its feed is invalid, or if the task layout is not
            head-locked and ``--ovxr_pip_head_locked`` is not given.
    """
    if args.ovxr_pip_camera is None:
        return ()

    from ..camera_feed import resolve_ovxr_camera_configs, select_ovxr_camera_feeds
    from ..isaac_teleop_cfg import XrCameraFeedLayoutCfg

    selected = select_ovxr_camera_feeds(env_cfg, args.ovxr_pip_camera, renderer_cfg)
    layout = env_cfg.isaac_teleop.xr_camera_feed_layout or XrCameraFeedLayoutCfg()
    if args.ovxr_pip_head_locked:
        layout = layout.replace(placement="head_locked")
    elif layout.placement != "head_locked":
        raise ValueError(
            f"OVXR camera panels are head-locked, but the task places them with {layout.placement!r} placement."
            " Pass --ovxr_pip_head_locked to place them in the OpenXR VIEW space."
        )
    return resolve_ovxr_camera_configs(
        selected, layout, max_fps=args.ovxr_pip_fps, position_override=args.ovxr_pip_view_position
    )


def reject_ovxr_runtime_conflicts(configs: Sequence[Any], kit_sources: Sequence[str]) -> None:
    """Reject validated launch components that conflict with an OVXR-owned frame loop.

    Args:
        configs: Runtime configurations returned by :func:`~isaaclab.app.validate_launch`.
        kit_sources: Components that :func:`~isaaclab.app.validate_launch` found to require Kit.

    Raises:
        ValueError: If a component requires Kit or a local visualizer is configured.
    """
    from isaaclab.visualizers import VisualizerBackendCfg

    if kit_sources or any(isinstance(cfg, VisualizerBackendCfg) for cfg in configs):
        raise ValueError("--ovxr requires no Kit runtime or local visualizer")
