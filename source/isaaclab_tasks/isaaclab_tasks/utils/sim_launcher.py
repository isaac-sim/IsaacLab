# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for detecting and launching the appropriate simulation backend."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

from isaaclab.physics.physics_manager_cfg import PhysicsCfg
from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.sensors.camera.camera_cfg import CameraCfg

logger = logging.getLogger(__name__)


def add_launcher_args(parser: argparse.ArgumentParser) -> None:
    """Add simulation-launcher CLI arguments (``--headless``, ``--device``, etc.) to *parser*.

    Delegates to :meth:`AppLauncher.add_app_launcher_args` so that user scripts
    do not need to import ``AppLauncher`` directly.
    """
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)


def _scan_config(cfg, predicates: list[Callable[[Any], bool]]) -> list[bool]:
    """Recursively walk *cfg* and evaluate each predicate on every node.

    Returns a list of booleans, one per predicate, where ``True`` means at
    least one node in the config tree satisfied that predicate.  Once a
    predicate is satisfied it is no longer evaluated (short-circuit).
    """
    results = [False] * len(predicates)
    visited: set[int] = set()

    def _visit(node):
        if all(results):
            return
        node_id = id(node)
        if node_id in visited:
            return
        visited.add(node_id)

        for i, pred in enumerate(predicates):
            if not results[i] and pred(node):
                results[i] = True

        try:
            children = vars(node)
        except TypeError:
            return
        for child in children.values():
            if child is None or isinstance(child, (int, float, str, bool)):
                continue
            _visit(child)

    _visit(cfg)
    return results


def _is_kitless_physics(node) -> bool:
    """True when the node is a kitless physics config (Newton or OvPhysX)."""
    return isinstance(node, PhysicsCfg) and type(node).__name__ in ("NewtonCfg", "OvPhysxCfg")


def _get_visualizer_types(launcher_args: argparse.Namespace | dict | None) -> set[str]:
    """Extract requested visualizer type names from launcher args."""
    if isinstance(launcher_args, argparse.Namespace):
        visualizers = getattr(launcher_args, "visualizer", None)
    elif isinstance(launcher_args, dict):
        visualizers = launcher_args.get("visualizer")
    else:
        return set()
    if not visualizers:
        return set()
    return {str(v).strip().lower() for v in visualizers if str(v).strip()}


def _is_kit_camera(node) -> bool:
    """True for a CameraCfg whose renderer requires Kit (not Newton)."""
    if not isinstance(node, CameraCfg):
        return False
    renderer_cfg = getattr(node, "renderer_cfg", None)
    if renderer_cfg is None:
        return True
    if isinstance(renderer_cfg, RendererCfg):
        return renderer_cfg.renderer_type in ("default", "isaac_rtx")
    return True


def compute_kit_requirements(
    env_cfg,
    launcher_args: argparse.Namespace | dict | None = None,
) -> tuple[bool, bool, set[str]]:
    """Compute whether Kit is needed and related flags.

    Uses the same logic as :func:`launch_simulation` to decide whether Isaac Sim
    Kit must be launched.

    Args:
        env_cfg: Resolved environment config (e.g. from :func:`resolve_task_config`).
        launcher_args: Optional CLI args; if ``visualizer=kit`` is set, needs_kit is True.

    Returns:
        (needs_kit, has_kit_cameras, visualizer_types)
    """
    is_kitless, has_kit_cameras = _scan_config(env_cfg, [_is_kitless_physics, _is_kit_camera])
    needs_kit = has_kit_cameras or not is_kitless
    visualizer_types = _get_visualizer_types(launcher_args)
    if "kit" in visualizer_types:
        needs_kit = True
    return needs_kit, has_kit_cameras, visualizer_types


@contextmanager
def launch_simulation(
    env_cfg,
    launcher_args: argparse.Namespace | dict | None = None,
) -> Generator[None, None, None]:
    """Context manager that launches the appropriate simulation runtime for *env_cfg*.

    * Recursively scans the config tree to decide whether Isaac Sim Kit is needed.
    * Auto-enables ``enable_cameras`` when the scene contains camera sensors
      that use a Kit renderer (not Newton).
    * For Kit-based backends, launches ``AppLauncher`` and calls ``app.close()`` on exit.
    * For kitless backends (e.g. Newton with Newton Warp renderer only), this is a no-op.
    * For Newton Physics + RTX Renderer (with Kit cameras): Kit is launched
      so that RTX can run; Newton syncs its state to the USD stage each step for rendering.

    Example::

        with launch_simulation(env_cfg, args_cli):
            main()
    """
    needs_kit, has_kit_cameras, visualizer_types = compute_kit_requirements(env_cfg, launcher_args)

    if needs_kit and has_kit_cameras:
        if isinstance(launcher_args, argparse.Namespace):
            if not getattr(launcher_args, "enable_cameras", False):
                logger.info("Auto-enabling cameras: scene contains camera sensors with a Kit renderer.")
                launcher_args.enable_cameras = True
        elif isinstance(launcher_args, dict):
            if not launcher_args.get("enable_cameras", False):
                logger.info("Auto-enabling cameras: scene contains camera sensors with a Kit renderer.")
                launcher_args["enable_cameras"] = True

    close_fn: Any = None

    if needs_kit:
        from isaaclab.app import AppLauncher

        app_launcher = AppLauncher(launcher_args)
        close_fn = app_launcher.app.close
    elif visualizer_types:
        # Newton path without Kit: AppLauncher is skipped, so manually store the visualizer
        # selection in SettingsManager (works in standalone mode via plain dict) so that
        # SimulationContext._get_cli_visualizer_types() can find it.
        from isaaclab.app.settings_manager import get_settings_manager

        visualizer_str = " ".join(sorted(visualizer_types))
        get_settings_manager().set_string("/isaaclab/visualizer/types", visualizer_str)

    try:
        yield
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        if close_fn is not None:
            close_fn()
