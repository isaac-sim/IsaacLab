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


def _is_newton_physics(node) -> bool:
    """True when the node is a Newton physics config (Kit is not required)."""
    return isinstance(node, PhysicsCfg) and type(node).__name__ == "NewtonCfg"


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
    if isinstance(visualizers, str):
        # CLI now uses comma-delimited syntax: --visualizer kit,newton,rerun
        visualizers = [token.strip() for token in visualizers.split(",")]
    return {str(v).strip().lower() for v in visualizers if str(v).strip()}


def _compute_visualizer_intent(env_cfg) -> dict[str, bool]:
    """Compute upstream visualizer intent from ``env_cfg.sim.visualizer_cfgs``."""
    sim_cfg = getattr(env_cfg, "sim", None)
    visualizer_cfgs = getattr(sim_cfg, "visualizer_cfgs", None)
    if visualizer_cfgs is None:
        return {"has_any_visualizers": False, "has_kit_visualizer": False}

    cfg_list = visualizer_cfgs if isinstance(visualizer_cfgs, list) else [visualizer_cfgs]
    cfg_list = [cfg for cfg in cfg_list if cfg is not None]
    has_any = len(cfg_list) > 0
    has_kit = any(getattr(cfg, "visualizer_type", None) == "kit" for cfg in cfg_list)
    return {"has_any_visualizers": has_any, "has_kit_visualizer": has_kit}


def _set_visualizer_intent_on_launcher_args(
    launcher_args: argparse.Namespace | dict | None, visualizer_intent: dict[str, bool]
) -> None:
    """Attach visualizer intent to launcher args when possible."""
    if launcher_args is None:
        return
    if isinstance(launcher_args, argparse.Namespace):
        setattr(launcher_args, "visualizer_intent", visualizer_intent)
    elif isinstance(launcher_args, dict):
        launcher_args["visualizer_intent"] = visualizer_intent


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
        launcher_args: Optional CLI args; if ``--visualizer`` includes ``kit``, needs_kit is True.

    Returns:
        (needs_kit, has_kit_cameras, visualizer_types)
    """
    is_newton, has_kit_cameras = _scan_config(env_cfg, [_is_newton_physics, _is_kit_camera])
    needs_kit = has_kit_cameras or not is_newton
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
    visualizer_intent = _compute_visualizer_intent(env_cfg)
    _set_visualizer_intent_on_launcher_args(launcher_args, visualizer_intent)

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
        # check if Isaac Sim is installed
        import importlib.util

        if importlib.util.find_spec("omni.kit") is None:
            logger.error(
                "\n[ERROR] Isaac Sim is not installed or not found on PYTHONPATH.\n"
                "\n"
                "  This environment requires Isaac Sim and Omniverse Kit.\n"
                "    PhysX backend and Kit visualizer currently requires Isaac Sim.\n"
                "\n"
                "  To fix this, ensure Isaac Sim is installed and available in the current environment.\n"
                "\n"
                "  See https://isaac-sim.github.io/IsaacLab/main/source/setup/installation for details.\n"
            )
            raise SystemExit(1)

        # If the simulation app is not launched, we launch it.
        from isaaclab.utils import has_kit

        if not has_kit():
            from isaaclab.app import AppLauncher

            app_launcher = AppLauncher(launcher_args)
            close_fn = app_launcher.app.close
    elif visualizer_types:
        # Newton path without Kit: AppLauncher is skipped, so manually store the visualizer
        # selection in SettingsManager (works in standalone mode via plain dict) so that
        # SimulationContext._get_cli_visualizer_types() can find it.
        from isaaclab.app.settings_manager import get_settings_manager

        disable_all = "none" in visualizer_types
        active_types = [] if disable_all else sorted(visualizer_types)
        visualizer_str = " ".join(active_types)
        settings = get_settings_manager()
        settings.set_string("/isaaclab/visualizer/types", visualizer_str)
        settings.set_bool("/isaaclab/visualizer/explicit", True)
        settings.set_bool("/isaaclab/visualizer/disable_all", disable_all)

    try:
        yield
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        if close_fn is not None:
            close_fn()
