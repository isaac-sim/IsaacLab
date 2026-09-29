# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for detecting and launching the appropriate simulation backend.

The flow is intentionally simple: walk the config tree **once** to collect its
signals into a :class:`Scan`, resolve the Kit runtime sources from that scan and
the launcher inputs, then validate and launch.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field, is_dataclass
from types import SimpleNamespace
from typing import Any

from isaaclab_newton.physics import NewtonCfg, VBDSolverCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_ov.renderers import OVRTXRendererCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg
from isaaclab_visualizers.kit import KitVisualizerCfg

from ..physics.physics_manager_cfg import PhysicsCfg, PhysxAutoCfg, _resolve_physx_auto_cfg
from ..renderers.renderer_cfg import RendererCfg
from ..sensors.camera.camera_cfg import CameraCfg
from ..sim.simulation_cfg import SimulationCfg
from ..utils.device import set_cuda_device
from ..utils.presets import resolve_presets
from ..utils.string import string_to_callable
from ..visualizers import VisualizerCfg
from .logging_utils import apply_python_logging_level

logger = logging.getLogger(__name__)

_KIT_LAUNCHER = "isaaclab_physx.app:KitLauncher"
"""Launcher for Kit needs that no config names, e.g. a default-renderer camera or ``require_kit``."""


class SimulationLauncher:
    """Starts the process runtime a resolved simulation config requires, and stops it.

    A backend package subclasses this and names the subclass in its config's ``launcher_type``.
    :func:`launch_simulation` constructs it, which starts the runtime, and calls :meth:`close` when the
    simulation ends.
    """

    device: str | None = None
    """Simulation device chosen by the runtime, or None to keep the config's device."""

    def __init__(self, launcher_args: argparse.Namespace | dict | None = None):
        """Start the runtime.

        Args:
            launcher_args: Parsed launcher arguments.
        """

    def close(self, exit_code: int = 0) -> None:
        """Stop the runtime.

        Args:
            exit_code: Exit status of the simulation, for runtimes that end the process.
        """


def add_launcher_args(parser: argparse.ArgumentParser) -> None:
    """Add simulation-launcher CLI arguments (``--device``, ``--viz``, etc.) to *parser*.

    The arguments are defined by the Kit launcher, which consumes most of them.
    """
    string_to_callable(_KIT_LAUNCHER).add_launcher_args(parser)


def fuse_kit_args(argv: list[str]) -> list[str]:
    """Fuse ``["--kit_args", "<option-like value>"]`` pairs into single ``--kit_args=<value>`` tokens.

    Argparse rejects a value token that itself looks like an option (starts with ``-`` and contains
    no space) with "expected one argument", and Kit arguments always start with ``--``. Fusing the
    pair into the ``=``-attached form before parsing makes the documented space-separated form work
    for a single Kit argument. All other forms pass through unchanged.

    Args:
        argv: Command-line tokens, excluding the program name.

    Returns:
        Tokens with any affected pair replaced by one fused token.
    """
    fused: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        next_token = argv[index + 1] if index + 1 < len(argv) else None
        if token == "--kit_args" and next_token is not None and next_token.startswith("-") and " " not in next_token:
            fused.append(f"--kit_args={next_token}")
            index += 2
        else:
            fused.append(token)
            index += 1
    return fused


def _make_physics_cfg(physics_cfg_str: str) -> PhysicsCfg:
    """Build a physics config for the requested backend.

    Args:
        physics_cfg_str: Backend selector: ``"physx"``, ``"isaacsim_physx"``,
            ``"newton_mjwarp"``, ``"newton_vbd"``, or ``"ovphysx"``. The
            ``"physx"`` selector is automatic: it resolves to Isaac Sim PhysX
            when Kit is required, and to OvPhysX otherwise.

    Returns:
        A new physics config instance for the requested backend.

    Raises:
        ValueError: If *physics_cfg_str* does not name a known backend.
    """
    if physics_cfg_str == "physx":
        return PhysxAutoCfg(isaacsim_physx=PhysxCfg(), ovphysx=OvPhysxCfg())
    if physics_cfg_str == "isaacsim_physx":
        return PhysxCfg()
    if physics_cfg_str == "newton_mjwarp":
        return NewtonCfg()
    if physics_cfg_str == "newton_vbd":
        return NewtonCfg(solver_cfg=VBDSolverCfg())
    if physics_cfg_str == "ovphysx":
        return OvPhysxCfg()
    raise ValueError(
        f"Invalid physics config: {physics_cfg_str!r} "
        "(expected 'physx', 'isaacsim_physx', 'newton_mjwarp', 'newton_vbd', or 'ovphysx')."
    )


"""
Logging.
"""


def _resolve_python_logging_level(args: dict) -> int:
    """Return the level for ``--verbose`` / ``--info`` (also read from ``sys.argv``), else the current root level."""
    if args.get("verbose", False) or "--verbose" in sys.argv:
        return logging.DEBUG
    if args.get("info", False) or "--info" in sys.argv:
        return logging.INFO
    level = logging.getLogger().getEffectiveLevel()
    return logging.WARNING if level == logging.NOTSET else level


"""
The Single Scan.
"""


@dataclass
class Scan:
    """Resolved configurations and runtime requirements returned by :func:`scan`."""

    resolved_physics_cfg: PhysicsCfg | None  # first physics config in walk order (post --physics override)
    effective_cfg: Any  # the input config, or its replacement when the config itself was an overridden physics config
    visualizer_cfgs: list[VisualizerCfg]
    has_ovrtx: bool
    has_kit_camera: bool
    has_kit_physics: bool  # PhysX (Kit-based)
    has_ovphysx_physics: bool
    needs_kit: bool
    launcher_types: list[str] = field(default_factory=list)  # named by the physics and renderer configs
    simulation_cfg: SimulationCfg | None = None
    """Simulation config found in the tree, used to apply the runtime's device choice."""


def _refresh_physics_scan_flags(config_scan: Scan, concrete_physics_cfgs: list[PhysicsCfg], has_physics: bool) -> None:
    """Refresh physics-derived launch signals from concrete physics configs."""
    config_scan.has_kit_physics = any(isinstance(cfg, PhysxCfg) for cfg in concrete_physics_cfgs)
    config_scan.has_ovphysx_physics = any(isinstance(cfg, OvPhysxCfg) for cfg in concrete_physics_cfgs)
    config_scan.needs_kit = config_scan.has_kit_camera or config_scan.has_kit_physics or not has_physics


def _resolve_launch_cfg(cfg, launcher_args: dict):
    """Compose presets and streaming requirements before choosing the runtime."""
    max_visible = launcher_args.get("max_visible_envs")
    if max_visible is not None and max_visible < 0:
        raise ValueError("--max_visible_envs must be non-negative.")
    visualizer = launcher_args.get("visualizer")
    explicit = launcher_args.get("visualizer_explicit", False)
    livestream = launcher_args.get("livestream", -1)
    if livestream is None or int(livestream) < 0:
        livestream = os.environ.get("LIVESTREAM", 0)
    livestream = int(livestream)
    if livestream not in (0, 1, 2):
        raise ValueError(f"Invalid livestream mode: {livestream}. Expected 0 (disabled), 1, or 2.")
    launcher_args["livestream"] = livestream
    if cfg is None:
        cfg = SimulationCfg()
    selectors = {}
    if visualizer is not None or explicit:
        names = [name.strip() for name in visualizer.split(",")] if isinstance(visualizer, str) else visualizer
        names = names or ["none"]
        selectors[lambda value: isinstance(value, VisualizerCfg)] = names[0] if len(names) == 1 else names
    cfg = resolve_presets(cfg, selectors=selectors)
    if selectors:
        launcher_args.pop("visualizer", None)
        launcher_args.pop("visualizer_explicit", None)
    return cfg


def scan(cfg, launcher_args: argparse.Namespace | dict | None = None) -> Scan:
    """Resolve presets and collect runtime requirements from a config tree.

    Walk dataclass and namespace fields, dictionaries, lists, and tuples. Backend overrides
    replace nested configs in place; a replaced root is returned in :attr:`Scan.effective_cfg`.
    Automatic PhysX and RTX choices resolve after the walk, using all declared runtime needs.
    """
    launcher_args = {} if launcher_args is None else launcher_args
    launcher_args = vars(launcher_args) if isinstance(launcher_args, argparse.Namespace) else launcher_args
    root = {"cfg": _resolve_launch_cfg(cfg, launcher_args)}
    max_visible = launcher_args.get("max_visible_envs")

    physics_str = launcher_args.get("physics")
    physics_cfgs: list[PhysicsCfg] = []
    concrete_physics_cfgs: list[PhysicsCfg] = []
    simulation_cfg: SimulationCfg | None = None
    visualizer_cfgs: list[VisualizerCfg] = []
    has_ovrtx = False
    has_kit_camera = False
    auto_rtx_locations: list[tuple[Any, Any, bool]] = []  # (parent, key, is_cam_renderer) for each auto RTX placeholder
    auto_physx_locations: list[tuple[PhysicsCfg, Any, Any, bool]] = []  # (node, parent, key, is_first_physics)
    launcher_types: list[str] = []
    visited: set[int] = set()

    def visit(node, parent, key):
        nonlocal simulation_cfg, has_ovrtx, has_kit_camera
        if callable(node) or (isinstance(parent, SimulationCfg) and key == "default_visualizer_cfg"):
            return
        if not is_dataclass(node) and not isinstance(node, (dict, list, tuple, argparse.Namespace, SimpleNamespace)):
            return
        owner = parent if isinstance(parent, (dict, list, tuple)) else vars(parent)
        if isinstance(node, RendererCfg) and node.renderer_type == "auto_rtx":
            auto_rtx_locations.append((owner, key, isinstance(parent, CameraCfg) and key == "renderer_cfg"))

        if id(node) in visited:
            return
        visited.add(id(node))

        if isinstance(node, SimulationCfg):
            simulation_cfg = node
            if node.physics is None and physics_str:
                node.physics = PhysicsCfg()
            if launcher_args["livestream"] or launcher_args.get("xr", False):
                viewers = node.visualizer_cfgs
                viewers = viewers if isinstance(viewers, list) else [viewers] if viewers else []
                if launcher_args["livestream"] and node.visualizer_cfgs == []:
                    raise ValueError("Livestreaming requires the Kit visualizer; visualizer=none disables it.")
                if not any(viewer.launcher_type == _KIT_LAUNCHER for viewer in viewers):
                    node.visualizer_cfgs = [*viewers, KitVisualizerCfg(headless=bool(launcher_args.get("xr", False)))]
        elif isinstance(node, PhysicsCfg):
            if physics_str:
                owner[key] = node = _make_physics_cfg(physics_str)
            physics_cfgs.append(node)
            if isinstance(node, PhysxAutoCfg):
                auto_physx_locations.append((node, owner, key, len(physics_cfgs) == 1))
                return
            concrete_physics_cfgs.append(node)
        elif isinstance(node, RendererCfg):
            has_ovrtx |= node.renderer_type == "ovrtx"
        elif isinstance(node, CameraCfg):
            renderer = node.renderer_cfg
            has_kit_camera |= renderer is None or renderer.renderer_type in ("default", "isaac_rtx")
        elif isinstance(node, VisualizerCfg):
            visualizer_cfgs.append(node)
            has_ovrtx |= node.launcher_type == OVRTXRendererCfg.launcher_type
            if max_visible is not None:
                node.max_visible_envs = max_visible
        if isinstance(node, (PhysicsCfg, RendererCfg, VisualizerCfg)) and (launcher_type := node.launcher_type):
            launcher_types.append(launcher_type)

        fields = node if isinstance(node, (dict, list, tuple)) else vars(node)
        children = enumerate(fields) if isinstance(fields, (list, tuple)) else fields.items()
        for name, child in children:
            visit(child, node, name)

    visit(root["cfg"], root, "cfg")

    has_physics = bool(physics_cfgs)
    config_scan = Scan(
        resolved_physics_cfg=physics_cfgs[0] if physics_cfgs else None,
        effective_cfg=root["cfg"],
        visualizer_cfgs=visualizer_cfgs,
        has_ovrtx=has_ovrtx,
        has_kit_camera=has_kit_camera,
        has_kit_physics=False,
        has_ovphysx_physics=False,
        needs_kit=False,
        launcher_types=launcher_types,
        simulation_cfg=simulation_cfg,
    )
    _refresh_physics_scan_flags(config_scan, concrete_physics_cfgs, has_physics)

    # RTX selection depends on the resolved physics backend.
    if auto_physx_locations:
        use_isaac_sim = bool(_get_kit_runtime_sources(config_scan, launcher_args))
        for node, owner, key, is_first_physics in auto_physx_locations:
            physics_cfg = _resolve_physx_auto_cfg(node, use_isaac_sim)
            concrete_physics_cfgs.append(physics_cfg)
            if launcher_type := physics_cfg.launcher_type:
                launcher_types.append(launcher_type)
            owner[key] = physics_cfg
            if is_first_physics:
                config_scan.resolved_physics_cfg = physics_cfg

        _refresh_physics_scan_flags(config_scan, concrete_physics_cfgs, has_physics)

    # Resolve recorded auto RTX renderer placeholders.
    use_isaac_sim = bool(_get_kit_runtime_sources(config_scan, launcher_args))
    renderer_factory = IsaacRtxRendererCfg if use_isaac_sim else OVRTXRendererCfg

    # Resolve every auto RTX placeholder in place, tracking camera renderers that may require Kit.
    for owner, key, is_camera in auto_rtx_locations:
        if owner is root:
            raise ValueError("Automatic RTX renderer placeholders cannot be resolved as the root config.")
        renderer_cfg = renderer_factory()
        if launcher_type := renderer_cfg.launcher_type:
            launcher_types.append(launcher_type)
        owner[key] = renderer_cfg
        config_scan.has_kit_camera |= use_isaac_sim and is_camera

    # Update the scan with the resolved auto RTX renderer type and Kit-camera status.
    config_scan.has_ovrtx |= not use_isaac_sim and bool(auto_rtx_locations)
    _refresh_physics_scan_flags(config_scan, concrete_physics_cfgs, has_physics)
    config_scan.effective_cfg = root["cfg"]

    return config_scan


"""
Launch Decisions (derived purely from a scan).
"""


def _get_kit_runtime_sources(config_scan: Scan, launcher_args: dict) -> tuple[str, ...]:
    """Return the config and launcher components that require Isaac Sim / Kit."""
    kit_sources = []
    if config_scan.has_kit_physics:
        kit_sources.append("Isaac Sim PhysX physics (`PhysxCfg`)")
    if config_scan.has_kit_camera:
        kit_sources.append('a Kit-based renderer (`IsaacRtxRendererCfg`, `renderer_type="isaac_rtx"`)')
    if any(cfg.launcher_type == _KIT_LAUNCHER for cfg in config_scan.visualizer_cfgs):
        kit_sources.append('the Kit visualizer (`--visualizer kit` / `visualizer_type="kit"`)')
    if launcher_args.get("experience"):
        kit_sources.append("an explicit Kit experience")
    if launcher_args.get("livestream", 0) > 0:
        kit_sources.append("livestreaming")
    if launcher_args.get("require_kit", False):
        kit_sources.append("the caller's explicit Kit requirement")
    if config_scan.needs_kit and not kit_sources:
        kit_sources.append("the default Isaac Sim / Kit runtime")

    return tuple(kit_sources)


def _validate_runtime(scan: Scan, kit_sources: tuple[str, ...]) -> None:
    """Raise if the scan and resolved Kit sources form an unsupported combination.

    OVRTX is kitless and cannot share a process with Kit-based runtimes (PhysX physics
    or any other Kit source); OvPhysX physics is subject to the same process boundary.
    """
    if scan.has_ovphysx_physics and kit_sources:
        # both would initialize Carbonite in one process; without this guard OvPhysX fails
        # deep inside its own library with "Failed to initialize Carbonite and load PhysX plugins"
        raise ValueError(
            "Invalid backend combination: OvPhysX physics (`OvPhysxCfg`) is kitless and cannot be used together "
            f"with Isaac Sim / Kit ({', '.join(kit_sources)}).\n"
            "\n"
            "To fix this, pick one of the following supported combinations:\n"
            "  * Keep OvPhysX physics and switch to a kitless renderer/visualizer:\n"
            "      use `OvPhysxCfg` with `OVRTXRendererCfg`\n"
            "    (and use `--visualizer newton`, `--visualizer rerun`, or `--visualizer viser`, or omit\n"
            "    the visualizer argument for headless execution.)\n"
            "  * Keep Isaac Sim / Kit and switch to a Kit-compatible physics backend:\n"
            "      use `PhysxCfg` with `IsaacRtxRendererCfg`\n"
        )

    if not scan.has_ovrtx or not kit_sources:
        return

    raise ValueError(
        "Invalid backend combination: the OVRTX runtime (`OVRTXRendererCfg` or the"
        " `newton_rtx` visualizer) cannot be used together"
        f" with Isaac Sim / Kit ({', '.join(kit_sources)}).\n"
        "\n"
        "To fix this, pick one of the following supported combinations:\n"
        "  * Keep Isaac Sim / Kit and switch the renderer:\n"
        "      use `IsaacRtxRendererCfg`, the Kit-compatible renderer\n"
        "  * Keep OVRTX (`OVRTXRendererCfg` or `--visualizer newton_rtx`) and remove every Kit source\n"
    )


def _resolve_distributed_device(sim_cfg: SimulationCfg | None, launcher_args: dict) -> None:
    """Set the simulation and launcher devices for distributed training.

    When ``--distributed`` restricts each process to one GPU, ``local_rank`` may exceed
    the visible device count, so the process falls back to the one GPU it can see.
    """
    if not launcher_args.get("distributed", False):
        return

    import torch

    local_rank = int(os.getenv("LOCAL_RANK", "0")) + int(os.getenv("JAX_LOCAL_RANK", "0"))
    num_visible_gpus = torch.cuda.device_count()
    # Compare against the local device count (not WORLD_SIZE) so multi-node runs work.
    device_str = f"cuda:{local_rank}" if local_rank < num_visible_gpus else "cuda:0"

    if sim_cfg is not None:
        sim_cfg.device = device_str
    launcher_args["device"] = device_str
    set_cuda_device(device_str)
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
    needed), then starts the launcher each required runtime's config names (closed on exit) or
    does nothing for kitless ones. Cameras are auto-enabled for Kit-renderer sensors.

    Resolve viewer and physics choices on the same config used to construct the simulation::

        sim_cfg = SimulationCfg()
        with launch_simulation(sim_cfg, args_cli):
            sim = SimulationContext(sim_cfg)

    Yields the resolved physics config when a caller needs to adjust solver settings before construction.

    Args:
        cfg: Config tree to scan for backend, renderer, and sensor requirements.
        launcher_args: Parsed launcher arguments, typically the script's ``args_cli``. Besides the
            arguments added by :func:`add_launcher_args`, the following keys are read when a script
            contributes them:

            * ``physics``: Backend selector applied to every physics config in *cfg*, see
              :func:`_make_physics_cfg`.
            * ``require_kit``: Whether the caller needs Kit for a reason *cfg* cannot express, e.g.
              a tool that reaches a Kit-only extension API. This is additive -- it can only turn a
              kitless launch into a Kit one, never the reverse, so a config that already needs Kit
              still launches it when the key is absent or ``False``.
    """
    launcher_args = {} if launcher_args is None else launcher_args
    launcher_args = vars(launcher_args) if isinstance(launcher_args, argparse.Namespace) else launcher_args

    # The single walk: collect every signal, apply the --physics override, and
    # resolve the automatic PhysX and RTX placeholders.
    config_scan = scan(cfg, launcher_args)
    sim_cfg = config_scan.simulation_cfg
    physics_cfg = config_scan.resolved_physics_cfg

    kit_sources = _get_kit_runtime_sources(config_scan, launcher_args)
    _validate_runtime(config_scan, kit_sources)
    needs_kit = bool(kit_sources)
    kit_viewers = [cfg for cfg in config_scan.visualizer_cfgs if cfg.launcher_type == _KIT_LAUNCHER]
    launcher_args["kit_visualizer"] = any(not cfg.headless for cfg in kit_viewers)

    # Honor --verbose / --info; the Kit launcher re-applies this level once Kit has started.
    apply_python_logging_level(_resolve_python_logging_level(launcher_args))

    if needs_kit and (config_scan.has_kit_camera or any(cfg.streaming_view for cfg in kit_viewers)):
        if not launcher_args.get("enable_cameras", False):
            logger.info(
                "Auto-enabling camera rendering because the scene contains Kit camera sensors "
                "or a Kit visualizer with streaming_view=True."
            )
            launcher_args["enable_cameras"] = True

    # Resolve distributed device early, before any launcher or physics init.
    _resolve_distributed_device(sim_cfg, launcher_args)

    # Configs name their runtime; add Kit for requirements without a config-owned launcher.
    launcher_types = [_KIT_LAUNCHER] if needs_kit else []
    launcher_types += config_scan.launcher_types
    launchers = [string_to_callable(launcher_type)(launcher_args) for launcher_type in dict.fromkeys(launcher_types)]
    for launcher in launchers:
        # the runtime may refine the device choice made by _resolve_distributed_device
        if sim_cfg is not None and launcher.device is not None:
            sim_cfg.device = launcher.device

    exit_code = 0
    try:
        # The import stays after the Kit launch decision. With no selected profile this is a
        # no-op; with one, it installs process-wide OmniClient routing before user code runs.
        from ..utils.assets import configure_storage_profile

        configure_storage_profile()
        yield physics_cfg
    except KeyboardInterrupt:
        exit_code = 130
        raise
    except SystemExit as exc:
        # keep the status of ``sys.exit(n)`` in the block; Kit would otherwise exit with 0
        exit_code = exc.code if isinstance(exc.code, int) else int(exc.code is not None)
        raise
    except Exception:
        exit_code = 1
        import traceback

        traceback.print_exc()
        raise
    finally:
        for launcher in reversed(launchers):
            launcher.close(exit_code)
