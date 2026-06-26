# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for detecting and launching the appropriate simulation backend.

The flow is intentionally simple: walk the config tree **once** to collect every
signal we care about (physics backend, OVRTX renderer, Kit cameras, visualizer
intent) into a :class:`Scan`, then decide how to launch from that scan.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from isaaclab_newton.physics import NewtonCfg
from isaaclab_ovphysx.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.physics.physics_manager_cfg import PhysicsCfg
from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.sensors.camera.camera_cfg import CameraCfg

logger = logging.getLogger(__name__)

# Class names of the kitless physics backends (Newton, OvPhysX). Matched by exact
# name so subclasses with distinct names (e.g. ``DeformableNewtonCfg``) opt out.
_KITLESS_PHYSICS_CFGS = ("NewtonCfg", "OvPhysxCfg")


def add_launcher_args(parser: argparse.ArgumentParser) -> None:
    """Add simulation-launcher CLI arguments (``--headless``, ``--device``, etc.) to *parser*.

    Delegates to :meth:`AppLauncher.add_app_launcher_args` so that user scripts
    do not need to import ``AppLauncher`` directly.
    """
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)


def make_physics_cfg(physics_cfg_str: str) -> PhysicsCfg:
    """Build a concrete physics config for the requested backend.

    Args:
        physics_cfg_str: Backend selector: ``"physx"``, ``"newton_mjwarp"``, or ``"ovphysx"``.

    Returns:
        A new physics config instance for the requested backend.

    Raises:
        ValueError: If *physics_cfg_str* does not name a known backend.
    """
    if physics_cfg_str == "physx":
        return PhysxCfg()
    if physics_cfg_str == "newton_mjwarp":
        return NewtonCfg()
    if physics_cfg_str == "ovphysx":
        return OvPhysxCfg()
    raise ValueError(f"Invalid physics config: {physics_cfg_str!r} (expected 'physx', 'newton_mjwarp', or 'ovphysx').")


"""
Node Predicates.
"""


def _is_ovrtx_renderer(node) -> bool:
    """True when the node is an OVRTX renderer config."""
    return isinstance(node, RendererCfg) and getattr(node, "renderer_type", None) == "ovrtx"


def _is_kit_camera(node) -> bool:
    """True for a CameraCfg whose renderer requires Kit (not Newton)."""
    if not isinstance(node, CameraCfg):
        return False
    renderer_cfg = getattr(node, "renderer_cfg", None)
    if renderer_cfg is None:
        return True
    if isinstance(renderer_cfg, RendererCfg):
        return renderer_cfg.renderer_type in ("default", "isaac_rtx")
    # PresetCfg renderers (e.g. MultiBackendRendererCfg) are resolved during
    # environment construction once the physics backend is known; assume they
    # match the backend, so not necessarily Kit.
    from isaaclab_tasks.utils import PresetCfg

    return not isinstance(renderer_cfg, PresetCfg)


"""
Launcher Argument Helpers.
"""


def _get_arg(launcher_args: argparse.Namespace | dict | None, key: str, default: Any = None) -> Any:
    """Read *key* from launcher args, whether a namespace, dict, or ``None``."""
    if isinstance(launcher_args, argparse.Namespace):
        return getattr(launcher_args, key, default)
    if isinstance(launcher_args, dict):
        return launcher_args.get(key, default)
    return default


def _set_arg(launcher_args: argparse.Namespace | dict | None, key: str, value: Any) -> None:
    """Write *key* on launcher args when it is a namespace or dict."""
    if isinstance(launcher_args, argparse.Namespace):
        setattr(launcher_args, key, value)
    elif isinstance(launcher_args, dict):
        launcher_args[key] = value


def _get_visualizer_types(launcher_args: argparse.Namespace | dict | None) -> set[str]:
    """Extract requested visualizer type names (``--visualizer kit,newton,...``)."""
    visualizers = _get_arg(launcher_args, "visualizer")
    if not visualizers:
        return set()
    if isinstance(visualizers, str):
        visualizers = visualizers.split(",")
    return {str(v).strip().lower() for v in visualizers if str(v).strip()}


def _get_livestream_mode(launcher_args: argparse.Namespace | dict | None) -> int:
    """Return effective livestream mode using the same CLI-over-env precedence as AppLauncher."""
    livestream_arg = _get_arg(launcher_args, "livestream", -1)
    if livestream_arg is not None and int(livestream_arg) >= 0:
        return int(livestream_arg)
    return int(os.environ.get("LIVESTREAM", 0))


def _ensure_livestream_kit_visualizer(launcher_args: argparse.Namespace | dict | None) -> None:
    """Request the Kit visualizer when livestreaming needs a video-producing viewport."""
    if launcher_args is None or _get_livestream_mode(launcher_args) == 0:
        return

    visualizer_explicit = bool(_get_arg(launcher_args, "visualizer_explicit", False))
    visualizers = _get_arg(launcher_args, "visualizer")
    if visualizer_explicit and (visualizers is None or "none" in _get_visualizer_types(launcher_args)):
        raise ValueError("Livestreaming requires the Kit visualizer. Remove '--viz none' or pass '--viz kit'.")

    visualizer_types = _get_visualizer_types(launcher_args)
    if "kit" in visualizer_types:
        return

    requested_visualizers = []
    if visualizers:
        requested_visualizers = (
            [visualizer.strip() for visualizer in visualizers.split(",")]
            if isinstance(visualizers, str)
            else [str(visualizer).strip() for visualizer in visualizers if str(visualizer).strip()]
        )
    requested_visualizers.append("kit")
    _set_arg(launcher_args, "visualizer", requested_visualizers)


def _get_visualizer_intent(cfg) -> dict[str, bool]:
    """Compute upstream visualizer intent from ``cfg.sim.visualizer_cfgs``."""
    visualizer_cfgs = getattr(getattr(cfg, "sim", None), "visualizer_cfgs", None)
    if visualizer_cfgs is None:
        return {"has_any_visualizers": False, "has_kit_visualizer": False}
    cfgs = visualizer_cfgs if isinstance(visualizer_cfgs, list) else [visualizer_cfgs]
    cfgs = [c for c in cfgs if c is not None]
    return {
        "has_any_visualizers": len(cfgs) > 0,
        "has_kit_visualizer": any(getattr(c, "visualizer_type", None) == "kit" for c in cfgs),
    }


"""
The Single Scan.
"""


@dataclass
class Scan:
    """Signals gathered from one walk of the config tree (see :func:`scan`).

    Every field is a plain snapshot computed during that single walk; nothing here
    is recomputed or mutated afterwards. ``needs_kit`` is the headline decision: a
    Kit-renderer camera or non-kitless physics requires Kit (the launcher additionally
    forces Kit when ``--visualizer kit`` is requested).
    """

    resolved_physics_cfg: PhysicsCfg | None  # first physics config in walk order (post --physics override)
    effective_cfg: Any  # the input config, or its replacement when the config itself was an overridden physics config
    visualizer_intent: dict[str, bool]
    has_ovrtx: bool
    has_kit_camera: bool
    has_kit_physics: bool  # PhysX (Kit-based)
    has_kitless_physics: bool  # Newton or OvPhysX
    has_ovphysx_physics: bool
    needs_kit: bool


def scan(cfg, physics_str: str | None = None) -> Scan:
    """Walk *cfg* once, collecting all launch signals and applying ``--physics``.

    When *physics_str* is set, every physics config is replaced by the requested
    backend (see :func:`make_physics_cfg`): nested configs in place, a root config
    via :attr:`Scan.effective_cfg` (it cannot be mutated in place).
    """
    physics_cfgs: list[PhysicsCfg] = []
    effective_cfg: Any = cfg
    has_ovrtx = False
    has_kit_camera = False
    visited: set[int] = set()

    def visit(node, parent, attr):
        nonlocal effective_cfg, has_ovrtx, has_kit_camera
        if id(node) in visited:
            return
        visited.add(id(node))

        if isinstance(node, PhysicsCfg):
            if physics_str:
                node = make_physics_cfg(physics_str)
                if parent is not None:
                    setattr(parent, attr, node)
                else:
                    effective_cfg = node
            physics_cfgs.append(node)
        elif _is_ovrtx_renderer(node):
            has_ovrtx = True
        elif _is_kit_camera(node):
            has_kit_camera = True

        try:
            children = vars(node)
        except TypeError:
            return
        for name, child in children.items():
            if child is None or isinstance(child, (int, float, str, bool)):
                continue
            visit(child, node, name)

    visit(cfg, None, None)

    names = [type(pcfg).__name__ for pcfg in physics_cfgs]
    has_kitless_physics = any(name in _KITLESS_PHYSICS_CFGS for name in names)
    return Scan(
        resolved_physics_cfg=physics_cfgs[0] if physics_cfgs else None,
        effective_cfg=effective_cfg,
        visualizer_intent=_get_visualizer_intent(cfg),
        has_ovrtx=has_ovrtx,
        has_kit_camera=has_kit_camera,
        has_kit_physics="PhysxCfg" in names,
        has_kitless_physics=has_kitless_physics,
        has_ovphysx_physics="OvPhysxCfg" in names,
        needs_kit=has_kit_camera or not has_kitless_physics,
    )


"""
Launch Decisions (derived purely from a scan).
"""


def _validate_runtime(scan: Scan, launcher_args: argparse.Namespace | dict | None) -> None:
    """Raise if *scan*'s physics/renderer/visualizer combination is unsupported.

    OVRTX is kitless and cannot share a process with Kit-based runtimes (PhysX physics
    or the Kit visualizer); OvPhysX physics likewise cannot run with the Kit visualizer.
    """
    visualizer_types = _get_visualizer_types(launcher_args)
    has_kit_visualizer = "kit" in visualizer_types or scan.visualizer_intent["has_kit_visualizer"]

    if scan.has_ovphysx_physics and has_kit_visualizer:
        raise ValueError(
            "Invalid backend combination: OvPhysX physics (`OvPhysxCfg`) is kitless and cannot be used together "
            'with the Kit visualizer (`--visualizer kit` / `visualizer_type="kit"`). Use a kitless visualizer '
            "such as `--visualizer newton`, `--visualizer rerun`, or `--visualizer viser`, or omit the visualizer "
            "argument for headless execution."
        )

    if not scan.has_ovrtx or (not scan.has_kit_physics and not has_kit_visualizer):
        return

    sources = []
    if scan.has_kit_physics:
        sources.append("Isaac Sim PhysX physics (`PhysxCfg`)")
    if has_kit_visualizer:
        sources.append('the Kit visualizer (`--visualizer kit` / `visualizer_type="kit"`)')

    raise ValueError(
        "Invalid backend combination: the OVRTX renderer (`OVRTXRendererCfg`,"
        ' `renderer_type="ovrtx"`) is a kitless renderer and cannot be used together'
        f" with Isaac Sim / Kit ({' and '.join(sources)}).\n"
        "\n"
        "To fix this, pick one of the following supported combinations:\n"
        "  * Keep Isaac Sim / Kit and switch the renderer:\n"
        "      presets=isaacsim_rtx_renderer\n"
        "    (uses `IsaacRtxRendererCfg`, the Kit-compatible renderer.)\n"
        "  * Keep the OVRTX renderer and switch to a kitless physics backend\n"
        "    (and avoid `--visualizer kit`):\n"
        "      presets=newton_mjwarp,ovrtx_renderer\n"
    )


def _resolve_distributed_device(cfg, launcher_args: argparse.Namespace | dict | None) -> None:
    """Set ``cfg.sim.device`` for distributed training, mirroring AppLauncher's fallback.

    When ``--distributed`` restricts each process to one GPU, ``local_rank`` may exceed
    the visible device count. The Kit path later overwrites this with ``AppLauncher.device``;
    the kitless path relies solely on this value.
    """
    if not _get_arg(launcher_args, "distributed", False):
        return

    import torch

    local_rank = int(os.getenv("LOCAL_RANK", "0")) + int(os.getenv("JAX_LOCAL_RANK", "0"))
    num_visible_gpus = torch.cuda.device_count()
    # Compare against the local device count (not WORLD_SIZE) so multi-node runs work.
    device_str = f"cuda:{local_rank}" if local_rank < num_visible_gpus else "cuda:0"

    sim_cfg = getattr(cfg, "sim", None)
    if sim_cfg is not None:
        sim_cfg.device = device_str
    torch.cuda.set_device(device_str)
    logger.info(
        "Distributed device resolved to %s (local_rank=%d, visible_gpus=%d)",
        device_str,
        local_rank,
        num_visible_gpus,
    )


@contextmanager
def launch_simulation(
    cfg,
    launcher_args: argparse.Namespace | dict | None = None,
) -> Generator[PhysicsCfg | None, None, None]:
    """Context manager that launches the appropriate simulation runtime for *cfg*.

    Walks the config tree once (resolving ``--physics``, validating the
    physics/renderer/visualizer combination, and deciding whether Isaac Sim Kit is
    needed), then launches ``AppLauncher`` for Kit-based backends (closed on exit) or
    does nothing for kitless ones. Cameras are auto-enabled for Kit-renderer sensors.

    Yields the resolved physics config, so a script can pass a bare placeholder and
    pick the backend from the command line::

        with launch_simulation(PhysicsCfg(), args_cli) as physics_cfg:
            sim = SimulationContext(SimulationCfg(physics=physics_cfg))

    Callers that do not need the value simply omit ``as``.
    """
    # The single walk: collect every signal and apply the --physics override.
    config_scan = scan(cfg, _get_arg(launcher_args, "physics"))
    effective_cfg = config_scan.effective_cfg
    physics_cfg = config_scan.resolved_physics_cfg
    _ensure_livestream_kit_visualizer(launcher_args)
    visualizer_types = _get_visualizer_types(launcher_args)

    # ovrtx + Kit visualizer share conflicting RTX hydra libraries under different USD
    # namespaces; loading both in one process crashes the dynamic linker. Fail early
    # with a targeted hint (_validate_runtime covers the broader ovrtx-vs-Kit cases).
    if "kit" in visualizer_types and config_scan.has_ovrtx:
        raise ValueError(
            "[launch_simulation] '--visualizer kit' is incompatible with 'ovrtx_renderer'. "
            "Both Kit (Isaac Sim) and ovrtx ship conflicting RTX hydra libraries "
            "(librtx.hydra.so, liblegacy.hydra.so) compiled against different USD namespaces, "
            "which causes a dynamic-linker crash when loaded into the same process. "
            "Use '--visualizer newton' instead, which is fully compatible with ovrtx presets."
        )

    _validate_runtime(config_scan, launcher_args)
    needs_kit = config_scan.needs_kit or "kit" in visualizer_types
    _set_arg(launcher_args, "visualizer_intent", config_scan.visualizer_intent)

    if needs_kit and config_scan.has_kit_camera and launcher_args is not None:
        if not _get_arg(launcher_args, "enable_cameras", False):
            logger.info("Auto-enabling cameras: scene contains camera sensors with a Kit renderer.")
            _set_arg(launcher_args, "enable_cameras", True)

    # Resolve distributed device early, before AppLauncher or physics init.
    _resolve_distributed_device(effective_cfg, launcher_args)

    visualizer_explicit_none = _get_arg(launcher_args, "visualizer") is None and _get_arg(
        launcher_args, "visualizer_explicit", False
    )

    close_fn: Any = None
    if needs_kit:
        _ensure_isaac_sim_available()
        from isaaclab.utils import has_kit

        if not has_kit():
            from isaaclab.app import AppLauncher

            app_launcher = AppLauncher(launcher_args)
            # AppLauncher may refine the device choice; propagate its final value,
            # intentionally overwriting the value set by _resolve_distributed_device.
            sim_cfg = getattr(effective_cfg, "sim", None)
            if sim_cfg is not None and hasattr(app_launcher, "device"):
                sim_cfg.device = app_launcher.device
            close_fn = app_launcher.app.close
    elif visualizer_types or visualizer_explicit_none:
        # Kitless path: AppLauncher is skipped, so persist the visualizer selection in
        # SettingsManager so SimulationContext._get_cli_visualizer_types() can find it.
        from isaaclab.app import AppLauncher

        disable_all = visualizer_explicit_none or "none" in visualizer_types
        base = vars(launcher_args) if isinstance(launcher_args, argparse.Namespace) else launcher_args
        if base is not None:
            AppLauncher.sync_visualizer_cli_settings_to_carb(
                {**base, "visualizer_explicit": True, "visualizer_disable_all": disable_all}
            )

    try:
        yield physics_cfg
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        if close_fn is not None:
            close_fn()


def _ensure_isaac_sim_available() -> None:
    """Raise ``SystemExit`` with an actionable hint when Isaac Sim / Kit is missing."""
    import importlib.util

    if importlib.util.find_spec("omni.kit") is not None:
        return

    isaaclab_path = os.environ.get("ISAACLAB_PATH")
    local_sim = os.path.join(isaaclab_path, "_isaac_sim") if isaaclab_path else None
    extra_hint = ""
    if local_sim and os.path.isdir(local_sim):
        launcher, source = ("isaaclab.bat", f'call "{local_sim}\\setup_conda_env.bat"')
        if sys.platform != "win32":
            launcher, source = ("./isaaclab.sh", f'source "{local_sim}/setup_conda_env.sh"')
        extra_hint = (
            f"  Found a local Isaac Sim at {local_sim} but its environment is not active.\n"
            f"  Either run via `{launcher} ...` (which sources the Isaac Sim env automatically),\n"
            f"  or in your current shell run:\n"
            f"    {source}\n"
        )

    logger.error(
        "\n[ERROR] Isaac Sim is not installed or not found on PYTHONPATH.\n"
        "\n"
        "  This environment requires Isaac Sim and Omniverse Kit.\n"
        "    PhysX backend and Kit visualizer currently requires Isaac Sim.\n"
        "\n"
        f"{extra_hint}"
        "  To fix this, ensure Isaac Sim is installed and available in the current environment.\n"
        "\n"
        "  See https://isaac-sim.github.io/IsaacLab/main/source/setup/installation for details.\n"
    )
    raise SystemExit(1)
