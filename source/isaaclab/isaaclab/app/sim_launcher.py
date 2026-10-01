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
import traceback
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
from isaaclab_newton.physics import NewtonCfg, VBDSolverCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_ov.renderers import OVRTXRendererCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg

from ..physics.physics_manager_cfg import PhysicsCfg, PhysxAutoCfg, _resolve_physx_auto_cfg
from ..renderers.renderer_cfg import RendererCfg
from ..sensors.camera.camera_cfg import CameraCfg
from ..sim.simulation_cfg import SimulationCfg
from ..utils.assets import configure_storage_profile
from ..utils.device import set_cuda_device
from ..utils.string import string_to_callable
from ..visualizers.visualizer_cfg import VisualizerCfg, parse_visualizer_csv, resolve_visualizer_cfgs
from .logging_utils import apply_python_logging_level
from .settings_manager import get_settings_manager

logger = logging.getLogger(__name__)

_KIT_LAUNCHER = "isaaclab_physx.app:KitLauncher"
"""Launcher for Kit needs that no config names, e.g. a default-renderer camera or ``--viz kit``."""


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
Node Predicates.
"""


def _is_kit_camera(node) -> bool:
    """True for a CameraCfg whose renderer requires Kit (not Newton)."""
    if not isinstance(node, CameraCfg):
        return False
    renderer_cfg = node.renderer_cfg
    if renderer_cfg is None:
        return True
    if not isinstance(renderer_cfg, RendererCfg):
        raise TypeError(
            f"CameraCfg.renderer_cfg must be a concrete RendererCfg or None, got {type(renderer_cfg).__name__}."
        )
    if renderer_cfg.renderer_type == "auto_rtx":
        # ``auto_rtx`` is resolved after the initial scan once physics and
        # visualizer intent are known; ie. it may become OVRTX for a kitless run.
        return False
    return renderer_cfg.renderer_type in ("default", "isaac_rtx")


"""
Launcher Argument Helpers.
"""


def _resolve_launcher_args(args: dict) -> None:
    """Resolve the livestream mode and the visualizer selection in place, once for every consumer.

    Writes ``livestream`` (the effective mode: ``--livestream`` when set (>= 0), else the ``LIVESTREAM``
    environment variable) and ``visualizer``: canonical names, an empty list when ``--viz none`` disabled
    all visualizers, or None when no visualizer was requested. Livestreaming adds the Kit visualizer, whose
    viewport produces the stream. Resolving twice gives the same result.
    """
    livestream = args.get("livestream", -1)
    if livestream is None or int(livestream) < 0:
        livestream = os.environ.get("LIVESTREAM", 0)
    livestream = int(livestream)
    if livestream not in (0, 1, 2):
        raise ValueError(f"Invalid livestream mode: {livestream}. Expected 0 (disabled), 1, or 2.")
    max_visible_envs = args.get("max_visible_envs")
    if max_visible_envs is not None and int(max_visible_envs) < 0:
        raise ValueError(f"Invalid value for --max_visible_envs: {max_visible_envs}. Expected non-negative int.")
    visualizers = args.get("visualizer")
    if visualizers:
        try:
            visualizers = parse_visualizer_csv(visualizers)
        except argparse.ArgumentTypeError as error:
            raise ValueError(str(error)) from error
    if livestream > 0:
        if visualizers == []:
            raise ValueError("Livestreaming requires the Kit visualizer. Remove '--viz none' or pass '--viz kit'.")
        if "kit" not in (visualizers or []):
            visualizers = [*(visualizers or []), "kit"]
    args["livestream"] = livestream
    args["visualizer"] = visualizers


def _resolve_python_logging_level(args: dict) -> int:
    """Return the level for ``--verbose`` / ``--info`` (also read from ``sys.argv``), else the current root level."""
    if args.get("verbose", False) or "--verbose" in sys.argv:
        return logging.DEBUG
    if args.get("info", False) or "--info" in sys.argv:
        return logging.INFO
    level = logging.getLogger().getEffectiveLevel()
    return logging.WARNING if level == logging.NOTSET else level


def _get_visualizer_intent(visualizer_cfgs: list[VisualizerCfg], args: dict) -> dict[str, bool]:
    """Compute the intent of the config's visualizers, OR-ed with a caller's ``visualizer_intent``.

    An explicit ``--visualizer`` selection overrides whether the config's Kit visualizer, and its streaming
    view, is used.
    """
    kit_cfgs = [cfg for cfg in visualizer_cfgs if cfg.visualizer_type == "kit"]
    if args["visualizer"] is not None:
        has_kit_visualizer = "kit" in args["visualizer"]
        # the selection drops a configured Kit visualizer it does not name
        kit_cfgs = kit_cfgs if has_kit_visualizer else []
    else:
        caller_intent = args.get("visualizer_intent") or {}
        has_kit_visualizer = bool(kit_cfgs) or bool(caller_intent.get("has_kit_visualizer"))
    return {
        "has_kit_visualizer": has_kit_visualizer,
        "has_kit_streaming_view": any(cfg.streaming_view for cfg in kit_cfgs),
    }


"""
The Single Scan.
"""


@dataclass
class Scan:
    """Signals gathered from one walk of the config tree (see :func:`scan`).

    Every field starts as a plain snapshot computed during that single walk.
    Automatic PhysX configurations and RTX placeholders are also recorded so
    launch-time resolution can update the physics- and renderer-related fields
    without traversing the config tree again. ``needs_kit`` is the headline launch
    decision after automatic selections are resolved: a Kit-renderer camera or Isaac
    Sim PhysX requires Kit (the launcher additionally forces Kit when
    ``--visualizer kit`` is requested).
    """

    resolved_physics_cfg: PhysicsCfg | None  # first physics config in walk order (post --physics override)
    effective_cfg: Any  # the input config, or its replacement when the config itself was an overridden physics config
    sim_cfg: SimulationCfg | None  # first simulation config in walk order, e.g. an env config's ``sim``
    visualizer_intent: dict[str, bool]
    has_ovrtx: bool
    has_kit_camera: bool
    has_kit_physics: bool  # PhysX (Kit-based)
    has_ovphysx_physics: bool
    needs_kit: bool
    launcher_types: list[str] = field(default_factory=list)  # named by the physics and renderer configs


def _refresh_physics_scan_flags(config_scan: Scan, concrete_physics_cfgs: list[PhysicsCfg], has_physics: bool) -> None:
    """Refresh physics-derived launch signals from concrete physics configs."""
    names = [type(pcfg).__name__ for pcfg in concrete_physics_cfgs]
    config_scan.has_kit_physics = "PhysxCfg" in names
    config_scan.has_ovphysx_physics = "OvPhysxCfg" in names
    config_scan.needs_kit = config_scan.has_kit_camera or config_scan.has_kit_physics or not has_physics


def scan(cfg, launcher_args: argparse.Namespace | dict | None = None) -> Scan:
    """Walk *cfg* once, collecting all launch signals and applying ``--physics``.

    When the ``physics`` key is present in *launcher_args*, every physics config is
    replaced by the requested backend (see :func:`launch_simulation`): nested configs
    in place, a root config via :attr:`Scan.effective_cfg` (it cannot be mutated in
    place). Automatic PhysX configurations and RTX
    renderer placeholders (``renderer_type="auto_rtx"``) are also resolved
    at this stage using the full *launcher_args* context.

    The walk mutates *cfg* in place, and resolving a placeholder consumes it, so
    a second walk of the same config observes the same signals and reaches the
    same launch decision.
    """
    args = vars(launcher_args) if isinstance(launcher_args, argparse.Namespace) else launcher_args
    args = {} if args is None else args
    # Livestreaming implies a Kit visualizer; make that visible to auto RTX resolution.
    _resolve_launcher_args(args)

    physics_str = args.get("physics")
    physics_cfgs: list[PhysicsCfg] = []
    concrete_physics_cfgs: list[PhysicsCfg] = []
    effective_cfg: Any = cfg
    sim_cfg: SimulationCfg | None = None
    has_ovrtx = "newton_rtx" in (args["visualizer"] or ())
    has_auto_rtx = False
    has_auto_physx = False
    has_kit_camera = False
    auto_rtx_locations: list[tuple[Any, Any, bool]] = []  # (parent, key, is_cam_renderer) for each auto RTX placeholder
    auto_physx_locations: list[tuple[PhysicsCfg, Any, Any, bool]] = []  # (node, parent, key, is_first_physics)
    launcher_types: list[str] = []
    visualizer_cfgs: list[VisualizerCfg] = []
    visited: set[int] = set()

    def add_launcher_type(node: PhysicsCfg | RendererCfg):
        if node.launcher_type:
            launcher_types.append(node.launcher_type)

    def visit(node, parent, key):
        nonlocal effective_cfg, sim_cfg, has_ovrtx, has_auto_rtx, has_auto_physx, has_kit_camera
        if isinstance(node, RendererCfg) and node.renderer_type == "auto_rtx":
            has_auto_rtx = True
            auto_rtx_locations.append((parent, key, isinstance(parent, CameraCfg) and key == "renderer_cfg"))

        if id(node) in visited:
            return
        visited.add(id(node))

        if isinstance(node, PhysicsCfg):
            if physics_str:
                node = _make_physics_cfg(physics_str)
                if parent is not None:
                    setattr(parent, key, node)
                else:
                    effective_cfg = node
            physics_cfgs.append(node)
            if isinstance(node, PhysxAutoCfg):
                has_auto_physx = True
                auto_physx_locations.append((node, parent, key, len(physics_cfgs) == 1))
                return
            else:
                concrete_physics_cfgs.append(node)
        elif isinstance(node, VisualizerCfg):
            visualizer_cfgs.append(node)
        elif isinstance(node, SimulationCfg):
            sim_cfg = sim_cfg or node
        elif isinstance(node, RendererCfg) and node.renderer_type == "ovrtx":
            has_ovrtx = True
        elif _is_kit_camera(node):
            has_kit_camera = True
        if isinstance(node, (PhysicsCfg, RendererCfg)):
            add_launcher_type(node)

        try:
            children = vars(node)
        except TypeError:
            return
        for name, child in children.items():
            if child is None or isinstance(child, (int, float, str, bool)):
                continue
            if isinstance(child, (list, tuple)):
                # sequences are not walked, except to collect visualizers, e.g. ``SimulationCfg.visualizer_cfgs``
                visualizer_cfgs.extend(item for item in child if isinstance(item, VisualizerCfg))
                continue
            visit(child, node, name)

    visit(cfg, None, None)
    if args["visualizer"] is None:
        # an explicit --visualizer selection drops configured visualizers it does not name, as for Kit
        has_ovrtx = has_ovrtx or any(cfg.visualizer_type == "newton_rtx" for cfg in visualizer_cfgs)

    has_physics = bool(physics_cfgs)
    config_scan = Scan(
        resolved_physics_cfg=physics_cfgs[0] if physics_cfgs else None,
        effective_cfg=effective_cfg,
        sim_cfg=sim_cfg,
        visualizer_intent=_get_visualizer_intent(visualizer_cfgs, args),
        has_ovrtx=has_ovrtx,
        has_kit_camera=has_kit_camera,
        has_kit_physics=False,
        has_ovphysx_physics=False,
        needs_kit=False,
        launcher_types=launcher_types,
    )
    _refresh_physics_scan_flags(config_scan, concrete_physics_cfgs, has_physics)

    # RTX selection depends on the resolved physics backend.
    if has_auto_physx:
        use_isaac_sim = bool(_get_kit_runtime_sources(config_scan, args))
        for node, parent, key, is_first_physics in auto_physx_locations:
            physics_cfg = _resolve_physx_auto_cfg(node, use_isaac_sim)
            concrete_physics_cfgs.append(physics_cfg)
            add_launcher_type(physics_cfg)
            if parent is None:
                effective_cfg = physics_cfg
                config_scan.effective_cfg = physics_cfg
            else:
                setattr(parent, key, physics_cfg)
            if is_first_physics:
                config_scan.resolved_physics_cfg = physics_cfg

        _refresh_physics_scan_flags(config_scan, concrete_physics_cfgs, has_physics)

    # Resolve recorded auto RTX renderer placeholders.
    if not has_auto_rtx:
        return config_scan

    use_isaac_sim = bool(_get_kit_runtime_sources(config_scan, args))
    renderer_factory = IsaacRtxRendererCfg if use_isaac_sim else OVRTXRendererCfg

    # Resolve every auto RTX placeholder in place, tracking camera renderers that may require Kit.
    has_auto_camera = False
    for location in auto_rtx_locations:
        if location[0] is None:
            raise ValueError("Automatic RTX renderer placeholders cannot be resolved as the root config.")
        renderer_cfg = renderer_factory()
        add_launcher_type(renderer_cfg)
        setattr(location[0], location[1], renderer_cfg)
        has_auto_camera = has_auto_camera or location[2]

    # Update the scan with the resolved auto RTX renderer type and Kit-camera status.
    if use_isaac_sim:
        config_scan.has_kit_camera = config_scan.has_kit_camera or has_auto_camera
    else:
        config_scan.has_ovrtx = True
    _refresh_physics_scan_flags(config_scan, concrete_physics_cfgs, has_physics)

    return config_scan


"""
Launch Decisions (derived purely from a scan).
"""


def _get_kit_runtime_sources(config_scan: Scan, args: dict) -> tuple[str, ...]:
    """Return the config and launcher components that require Isaac Sim / Kit; *args* are resolved by :func:`scan`."""
    kit_sources = []
    if config_scan.has_kit_physics:
        kit_sources.append("Isaac Sim PhysX physics (`PhysxCfg`)")
    if config_scan.has_kit_camera:
        kit_sources.append('a Kit-based renderer (`IsaacRtxRendererCfg`, `renderer_type="isaac_rtx"`)')
    if config_scan.visualizer_intent["has_kit_visualizer"]:
        kit_sources.append('the Kit visualizer (`--visualizer kit` / `visualizer_type="kit"`)')
    if args.get("experience", ""):
        kit_sources.append("an explicit Kit experience")
    if args.get("livestream", 0) > 0:
        kit_sources.append("livestreaming")
    if args.get("require_kit", False):
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


def _resolve_distributed_device(args: dict) -> None:
    """Set the launcher ``device`` to this rank's GPU for distributed training.

    When ``--distributed`` restricts each process to one GPU, ``local_rank`` may exceed
    the visible device count, so the process falls back to the one GPU it can see.
    """
    if not args.get("distributed", False):
        return

    local_rank = int(os.getenv("LOCAL_RANK", "0")) + int(os.getenv("JAX_LOCAL_RANK", "0"))
    num_visible_gpus = torch.cuda.device_count()
    # Compare against the local device count (not WORLD_SIZE) so multi-node runs work.
    device_str = f"cuda:{local_rank}" if local_rank < num_visible_gpus else "cuda:0"

    args["device"] = device_str
    set_cuda_device(device_str)
    logger.info(
        "Distributed device resolved to %s (local_rank=%d, visible_gpus=%d)",
        device_str,
        local_rank,
        num_visible_gpus,
    )


def _resolve_device(sim_cfg, args: dict, launchers: list[SimulationLauncher]) -> None:
    """Write the run's device to ``sim_cfg.device`` and the ``device`` launcher argument, once.

    Starts from the ``device`` launcher argument resolved before launch; a started runtime may refine it
    (e.g. XR selects the CPU), and a bare ``"cuda"`` is pinned to the physics GPU index.
    """
    device = args.get("device")
    for launcher in launchers:
        device = launcher.device or device
    if device == "cuda":
        cuda_device = get_settings_manager().get("/physics/cudaDevice")
        device = f"cuda:{max(0, int(cuda_device) if cuda_device is not None else 0)}"
    if device is None:
        return
    args["device"] = device
    if sim_cfg is not None:
        sim_cfg.device = device


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

    The run's visualizers and device are decided here, once: they are written to the
    :class:`~isaaclab.sim.SimulationCfg` in *cfg* (``visualizer_cfgs`` and ``device``), which
    every later consumer reads.

    Yields the resolved physics config, so a script can pass a bare placeholder and
    pick the backend from the command line::

        with launch_simulation(PhysicsCfg(), args_cli) as physics_cfg:
            sim = SimulationContext(SimulationCfg(physics=physics_cfg))

    Callers that do not need the value simply omit ``as``.

    Args:
        cfg: Config tree to scan for backend, renderer, and sensor requirements.
        launcher_args: Parsed launcher arguments, typically the script's ``args_cli``. Besides the
            arguments added by :func:`add_launcher_args`, the following keys are read when a script
            contributes them:

            * ``physics``: Backend selector applied to every physics config in *cfg*: ``"physx"``
              (Isaac Sim PhysX when Kit is required, else OvPhysX), ``"isaacsim_physx"``,
              ``"newton_mjwarp"``, ``"newton_vbd"``, or ``"ovphysx"``.
            * ``require_kit``: Whether the caller needs Kit for a reason *cfg* cannot express, e.g.
              a tool that reaches a Kit-only extension API. This is additive -- it can only turn a
              kitless launch into a Kit one, never the reverse, so a config that already needs Kit
              still launches it when the key is absent or ``False``.
            * ``visualizer_intent``: ``{"has_kit_visualizer": True}`` requests the Kit visualizer when *cfg*
              configures none and no ``visualizer`` selection is given.
    """
    # writes to ``args`` reach the caller's namespace or dict
    args = vars(launcher_args) if isinstance(launcher_args, argparse.Namespace) else launcher_args
    args = {} if args is None else args

    # The single walk: collect every signal, apply the --physics override, and
    # resolve the automatic PhysX and RTX placeholders.
    config_scan = scan(cfg, args)
    physics_cfg = config_scan.resolved_physics_cfg

    kit_sources = _get_kit_runtime_sources(config_scan, args)
    _validate_runtime(config_scan, kit_sources)
    needs_kit = bool(kit_sources)
    args["kit_visualizer"] = config_scan.visualizer_intent["has_kit_visualizer"]

    # Honor --verbose / --info; the Kit launcher re-applies this level once Kit has started.
    apply_python_logging_level(_resolve_python_logging_level(args))

    if needs_kit and (config_scan.has_kit_camera or config_scan.visualizer_intent.get("has_kit_streaming_view")):
        if not args.get("enable_cameras", False):
            logger.info(
                "Auto-enabling camera rendering because the scene contains Kit camera sensors "
                "or a Kit visualizer with streaming_view=True."
            )
            args["enable_cameras"] = True

    # The SimulationCfg the simulation is built from, e.g. an env config's ``sim``; a physics config or None
    # holds none.
    sim_cfg = config_scan.sim_cfg

    # Resolve the device before any launcher or physics init: --device, else this rank's GPU, else the config's.
    _resolve_distributed_device(args)
    if sim_cfg is not None:
        args["device"] = args.get("device") or sim_cfg.device
        # Decide the visualizers once, into the SimulationCfg.
        sim_cfg.visualizer_cfgs = resolve_visualizer_cfgs(
            sim_cfg.visualizer_cfgs, args["visualizer"], args.get("max_visible_envs")
        )

    # Start the launchers the resolved config names, plus Kit and OVRTX for needs that no config names
    # (e.g. a default-renderer camera, ``--viz kit`` or ``--viz newton_rtx``); Kit starts first.
    launcher_types = [_KIT_LAUNCHER] if needs_kit else []
    launcher_types += config_scan.launcher_types
    if config_scan.has_ovrtx:
        # validated above: OVRTX never shares the process with Kit
        launcher_types.append(OVRTXRendererCfg.launcher_type)
    launchers = [string_to_callable(launcher_type)(args) for launcher_type in dict.fromkeys(launcher_types)]
    # after the launchers, so a started Kit already backs the settings
    _resolve_device(sim_cfg, args, launchers)
    # A config without a SimulationCfg (e.g. a bare physics config) leaves the selection for the
    # SimulationContext built after launch; otherwise clear one left by an earlier launch. ``none``
    # round-trips through ``parse_visualizer_csv`` as ``--viz none``; empty means no selection.
    visualizers = None if sim_cfg is not None else args.get("visualizer")
    max_visible_envs = None if sim_cfg is not None else args.get("max_visible_envs")
    settings = get_settings_manager()
    settings.set("/isaaclab/visualizer/types", "" if visualizers is None else ",".join(visualizers) or "none")
    settings.set("/isaaclab/visualizer/max_visible_envs", -1 if max_visible_envs is None else int(max_visible_envs))

    exit_code = 0
    try:
        # With no selected profile this is a no-op; with one, it installs process-wide OmniClient
        # routing before user code runs.
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
        traceback.print_exc()
        raise
    finally:
        for launcher in reversed(launchers):
            launcher.close(exit_code)
