# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch checks shared by the live, recording and replay OVXR entrypoints."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ..xr_cfg import XrAnchorRotationMode, remove_camera_configs

if TYPE_CHECKING:
    from ..ovxr_cfg import OvLibrariesXrCameraFeedCfg
    from ..ovxr_renderer import OvxrRendererCfg


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
        ValueError: If ``num_envs`` is not 1, or the task needs Kit physics
            (see :func:`require_kitless_physics`).
    """
    if num_envs != 1:
        raise ValueError("--ovxr currently requires --num_envs 1")
    viewers = env_cfg.sim.visualizer_cfgs
    if not isinstance(viewers, list):
        viewers = [viewers]
    backends = [
        backend
        for viewer in viewers
        if viewer is not None
        for backend in (viewer.backend if isinstance(viewer.backend, list) else [viewer.backend])
    ]
    reject_ovxr_runtime_conflicts(backends, ())
    require_kitless_physics(env_cfg, task)
    if env_cfg.isaac_teleop is not None:
        xr_cfg = env_cfg.isaac_teleop.xr_cfg
        if xr_cfg.anchor_prim_path is not None and xr_cfg.anchor_rotation_mode == XrAnchorRotationMode.CUSTOM:
            raise ValueError("OVXR dynamic CUSTOM anchors are unsupported: the callback requires a previous head pose")
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
    """Add the OVXR camera-panel and desktop-mirror options shared by live teleoperation, recording and replay."""
    parser.add_argument(
        "--ovxr_desktop_mirror",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show the OVXR desktop mirror (disabled by default).",
    )
    parser.add_argument(
        "--ovxr_desktop_mirror_fps",
        type=float,
        default=30.0,
        help="Maximum desktop-mirror update rate (default: 30 Hz).",
    )
    parser.add_argument(
        "--ovxr_camera_feed",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Show up to three task-configured cameras as application-owned XR panels. Place Hydra"
            " overrides such as presets=rgb before this option or after another option."
        ),
    )
    parser.add_argument(
        "--ovxr_camera_feed_fps",
        type=float,
        default=30.0,
        help="Maximum OVXR camera feed update rate (default: 30 Hz).",
    )


def cuda_index_list(value: str) -> str:
    """Validate ``--ovrtx_active_cuda_gpus`` at parse time."""
    from ..ovxr_gpu_publication import parse_active_cuda_gpus

    try:
        parse_active_cuda_gpus(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return value


def ovxr_renderer_cfg(active_cuda_gpus: str | None) -> OvxrRendererCfg:
    """Build the OVXR renderer configuration shared by the OVXR session and its camera panels."""
    from ..ovxr_renderer import OvxrRendererCfg

    return OvxrRendererCfg(active_cuda_gpus=active_cuda_gpus)


def validate_ovxr_presentation_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace, *, using_ovxr: bool
) -> None:
    """Validate the shared OVXR presentation options.

    Args:
        parser: Parser used to report errors.
        args: Parsed namespace.
        using_ovxr: Whether the command selected ``--ovxr``.
    """
    if args.ovxr_camera_feed is not None and not using_ovxr:
        parser.error("--ovxr_camera_feed requires --ovxr")
    if args.ovxr_camera_feed is not None:
        # The option takes every following bare token, including Hydra overrides placed after it.
        overrides = [name for name in args.ovxr_camera_feed if "=" in name]
        if overrides:
            parser.error(
                f"--ovxr_camera_feed takes camera names, not overrides {overrides}; place overrides such as"
                " presets=rgb before --ovxr_camera_feed or after another option"
            )
    if args.ovxr_desktop_mirror and not using_ovxr:
        parser.error("--ovxr_desktop_mirror requires --ovxr")
    if args.ovxr_camera_feed is not None and args.disable_external_cameras:
        parser.error("--ovxr_camera_feed cannot be combined with --disable_external_cameras")
    if not math.isfinite(args.ovxr_camera_feed_fps) or args.ovxr_camera_feed_fps <= 0.0:
        parser.error("--ovxr_camera_feed_fps must be finite and positive")
    if not math.isfinite(args.ovxr_desktop_mirror_fps) or args.ovxr_desktop_mirror_fps <= 0:
        parser.error("--ovxr_desktop_mirror_fps must be finite and positive")


def resolve_ovxr_camera_feeds(
    env_cfg: Any, args: argparse.Namespace, renderer_cfg: Any
) -> tuple[OvLibrariesXrCameraFeedCfg, ...]:
    """Select the requested camera feeds and resolve their OVXR panels before launch.

    Selection assigns ``renderer_cfg`` to each selected camera, so the OVXR session and its panels
    share one OVXR renderer configuration. Resolution creates no native resources, so an invalid selection or
    task layout fails before the simulation launches.

    OVXR shows camera feeds head-locked only. The task's ``xr_camera_feed_layout`` supplies the panel
    size, gap and distance; its ``placement`` is not used, nor is the world pose of a ``"world"`` layout.

    Args:
        env_cfg: Resolved environment configuration.
        args: Parsed namespace with the shared OVXR presentation options.
        renderer_cfg: OVXR renderer configuration shared by the OVXR session and its camera feeds.

    Returns:
        The resolved panels in selection order, or an empty tuple without ``--ovxr_camera_feed``.

    Raises:
        ValueError: If a selected camera or its feed is invalid, or if the task's
            ``xr_camera_feed_layout`` cannot be placed head-locked.
    """
    if args.ovxr_camera_feed is None:
        return ()

    from ..camera_feed import resolve_ovxr_camera_configs, select_ovxr_camera_feeds
    from ..isaac_teleop_cfg import XrCameraFeedLayoutCfg

    selected = select_ovxr_camera_feeds(env_cfg, args.ovxr_camera_feed, renderer_cfg)
    layout = env_cfg.isaac_teleop.xr_camera_feed_layout or XrCameraFeedLayoutCfg()
    try:
        return resolve_ovxr_camera_configs(
            selected, layout.replace(placement="head_locked"), max_fps=args.ovxr_camera_feed_fps
        )
    except ValueError as exc:
        if layout.placement != "world":
            raise
        # "world" placement leaves distance_m unused and unvalidated; head-locked placement needs it.
        raise ValueError(
            "OVXR shows camera feeds head-locked, but the task's xr_camera_feed_layout uses 'world' placement and"
            f" cannot be shown that way: {exc}"
        ) from exc


def reject_ovxr_runtime_conflicts(configs: Sequence[Any], kit_sources: Sequence[str]) -> None:
    """Reject validated launch components that conflict with an OVXR-owned frame loop.

    Args:
        configs: Runtime configurations returned by :func:`~isaaclab.app.validate_launch`.
        kit_sources: Components that :func:`~isaaclab.app.validate_launch` found to require Kit.

    Raises:
        ValueError: If a component requires Kit or a local visualizer is configured.
    """
    from isaaclab.visualizers import VisualizerBackendCfg

    visualizers = [cfg for cfg in configs if isinstance(cfg, VisualizerBackendCfg)]
    if any(cfg.visualizer_type == "kit" for cfg in visualizers):
        raise ValueError("--ovxr does not support the deprecated Kit visualizer; remove the Kit --viz selection")
    if kit_sources:
        raise ValueError(f"--ovxr requires kitless runtime components; Kit is required by: {', '.join(kit_sources)}")
    if visualizers:
        raise ValueError("Local visualizers are not currently supported with --ovxr; remove the --viz selection")
