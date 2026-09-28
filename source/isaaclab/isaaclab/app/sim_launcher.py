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
import warnings
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from isaaclab_newton.physics import NewtonCfg, VBDSolverCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_ov.renderers import OVRTXRendererCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg

from ..physics.physics_manager_cfg import PhysicsCfg, PhysxAutoCfg, _resolve_physx_auto_cfg
from ..renderers.renderer_cfg import RendererCfg
from ..sensors.camera.camera_cfg import CameraCfg
from ..utils.device import set_cuda_device
from ..utils.string import string_to_callable
from .logging_utils import apply_python_logging_level, resolve_python_logging_level
from .settings_manager import get_settings_manager

logger = logging.getLogger(__name__)

_VISUALIZER_TYPES = ("kit", "newton_gl", "newton_rtx", "rerun", "viser", "none")
_VISUALIZER_ALIASES = {"newton": "newton_gl"}
"""Deprecated ``--visualizer`` names and their replacements."""

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
    string_to_callable(_KIT_LAUNCHER).add_app_launcher_args(parser)


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


def make_physics_cfg(physics_cfg_str: str) -> PhysicsCfg:
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


def _is_ovrtx_renderer(node) -> bool:
    """True when the node is an OVRTX renderer config."""
    return isinstance(node, RendererCfg) and getattr(node, "renderer_type", None) == "ovrtx"


def _is_auto_rtx_renderer(node) -> bool:
    """True when the node is an automatic RTX renderer placeholder."""
    return getattr(node, "renderer_type", None) == "auto_rtx"


def _is_auto_physx_physics(node) -> bool:
    """True when the node is an automatic PhysX placeholder."""
    return isinstance(node, PhysxAutoCfg)


def _is_kit_camera(node) -> bool:
    """True for a CameraCfg whose renderer requires Kit (not Newton)."""
    if not isinstance(node, CameraCfg):
        return False
    renderer_cfg = getattr(node, "renderer_cfg", None)
    if renderer_cfg is None:
        return True
    if _is_auto_rtx_renderer(renderer_cfg):
        # ``auto_rtx`` is resolved after the initial scan once physics and
        # visualizer intent are known; ie. it may become OVRTX for a kitless run.
        return False
    if not isinstance(renderer_cfg, RendererCfg):
        raise TypeError(
            f"CameraCfg.renderer_cfg must be a concrete RendererCfg or None, got {type(renderer_cfg).__name__}."
        )
    return renderer_cfg.renderer_type in ("default", "isaac_rtx")


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


def _parse_visualizer_csv(value: str) -> list[str] | None:
    """Parse the ``--visualizer`` comma-separated list into canonical names; ``none`` yields None."""
    token = (value or "").strip()
    if not token:
        raise argparse.ArgumentTypeError(
            "Invalid --visualizer value: empty string. Use a comma-separated list, e.g. --viz kit,newton_gl."
        )
    if " " in token:
        raise argparse.ArgumentTypeError(
            "Invalid --visualizer value: spaces are not allowed. "
            "Use a comma-separated list without spaces, e.g. --viz kit,newton_gl,rerun,viser."
        )
    names = [item.strip().lower() for item in token.split(",")]
    if any(not name for name in names):
        raise argparse.ArgumentTypeError(
            "Invalid --visualizer value: empty visualizer entry detected. "
            "Use a comma-separated list without empty items."
        )
    invalid = [name for name in names if name not in _VISUALIZER_TYPES and name not in _VISUALIZER_ALIASES]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Invalid --visualizer value(s): {', '.join(invalid)}. "
            f"Valid options: {', '.join(sorted(_VISUALIZER_TYPES))}."
        )
    for name in names:
        if name in _VISUALIZER_ALIASES:
            warnings.warn(
                f"--viz '{name}' is deprecated. Use '--viz {_VISUALIZER_ALIASES[name]}' instead.",
                DeprecationWarning,
                stacklevel=3,
            )
    names = [_VISUALIZER_ALIASES.get(name, name) for name in names]
    if "none" in names:
        if len(names) > 1:
            raise argparse.ArgumentTypeError(
                "Invalid --visualizer value: 'none' cannot be combined with other visualizer types."
            )
        return None
    return list(dict.fromkeys(names))


def _get_livestream_mode(launcher_args: argparse.Namespace | dict | None) -> int:
    """Return the livestream mode: ``--livestream`` when set (>= 0), else the ``LIVESTREAM`` environment variable."""
    livestream = _get_arg(launcher_args, "livestream", -1)
    if livestream is None or int(livestream) < 0:
        livestream = os.environ.get("LIVESTREAM", 0)
    livestream = int(livestream)
    if livestream not in (0, 1, 2):
        raise ValueError(f"Invalid livestream mode: {livestream}. Expected 0 (disabled), 1, or 2.")
    return livestream


def _normalize_launcher_args(launcher_args: argparse.Namespace | dict | None) -> None:
    """Resolve the livestream mode and the visualizer selection in place, once for every consumer.

    Writes ``livestream`` (the effective mode), ``visualizer`` (canonical names, or None when all are
    disabled), ``visualizer_explicit`` and ``visualizer_disable_all``. Livestreaming adds the Kit
    visualizer, whose viewport produces the stream. Normalizing twice gives the same result.
    """
    if launcher_args is None:
        return
    livestream = _get_livestream_mode(launcher_args)
    visualizers = _get_arg(launcher_args, "visualizer")
    explicit = bool(_get_arg(launcher_args, "visualizer_explicit", False)) or visualizers is not None
    if visualizers and not isinstance(visualizers, str):
        visualizers = ",".join(str(visualizer).strip() for visualizer in visualizers)
    if visualizers:
        try:
            visualizers = _parse_visualizer_csv(visualizers)
        except argparse.ArgumentTypeError as error:
            raise ValueError(str(error)) from error
    disable_all = explicit and visualizers is None
    if livestream > 0:
        if disable_all:
            raise ValueError("Livestreaming requires the Kit visualizer. Remove '--viz none' or pass '--viz kit'.")
        if "kit" not in (visualizers or []):
            visualizers = [*(visualizers or []), "kit"]
        explicit = True
    _set_arg(launcher_args, "livestream", livestream)
    _set_arg(launcher_args, "visualizer", visualizers)
    _set_arg(launcher_args, "visualizer_explicit", explicit)
    _set_arg(launcher_args, "visualizer_disable_all", disable_all)


def _sync_visualizer_cli_settings(launcher_args: argparse.Namespace | dict) -> None:
    """Write the normalized visualizer selection and ``--max_visible_envs`` to the settings."""
    max_visible_envs = _get_arg(launcher_args, "max_visible_envs")
    if max_visible_envs is not None and int(max_visible_envs) < 0:
        raise ValueError(f"Invalid value for --max_visible_envs: {max_visible_envs}. Expected non-negative int.")
    settings = get_settings_manager()
    settings.set_string("/isaaclab/visualizer/types", " ".join(_get_arg(launcher_args, "visualizer") or ()))
    settings.set_bool("/isaaclab/visualizer/explicit", bool(_get_arg(launcher_args, "visualizer_explicit", False)))
    settings.set_bool(
        "/isaaclab/visualizer/disable_all", bool(_get_arg(launcher_args, "visualizer_disable_all", False))
    )
    # Sentinel: ``-1`` means ``--max_visible_envs`` was not passed (see ``SimulationContext``).
    settings.set_int("/isaaclab/visualizer/max_visible_envs", -1 if max_visible_envs is None else int(max_visible_envs))


def _get_visualizer_intent(cfg, launcher_args: argparse.Namespace | dict | None) -> dict[str, bool]:
    """Compute the visualizer intent of ``cfg.sim.visualizer_cfgs``, OR-ed with a caller's ``visualizer_intent``."""
    # Accept both env_cfg (has .sim.visualizer_cfgs) and a bare SimulationCfg
    # (has .visualizer_cfgs directly).
    sim = getattr(cfg, "sim", None)
    visualizer_cfgs = getattr(sim, "visualizer_cfgs", None) or getattr(cfg, "visualizer_cfgs", None)
    if visualizer_cfgs is None:
        visualizer_cfgs = []
    cfgs = visualizer_cfgs if isinstance(visualizer_cfgs, list) else [visualizer_cfgs]
    kit_cfgs = [c for c in cfgs if getattr(c, "visualizer_type", None) == "kit"]
    caller_intent = _get_arg(launcher_args, "visualizer_intent") or {}
    return {
        "has_kit_visualizer": bool(kit_cfgs) or bool(caller_intent.get("has_kit_visualizer")),
        "has_kit_streaming_view": any(bool(getattr(c, "streaming_view", False)) for c in kit_cfgs)
        or bool(caller_intent.get("has_kit_streaming_view")),
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
    replaced by the requested backend (see :func:`make_physics_cfg`): nested configs
    in place, a root config via :attr:`Scan.effective_cfg` (it cannot be mutated in
    place). Automatic PhysX configurations and RTX
    renderer placeholders (``renderer_type="auto_rtx"``) are also resolved
    at this stage using the full *launcher_args* context.

    The walk mutates *cfg* in place, and resolving a placeholder consumes it, so
    a second walk of the same config observes the same signals and reaches the
    same launch decision.
    """
    # Livestreaming implies a Kit visualizer; make that visible to auto RTX resolution.
    _normalize_launcher_args(launcher_args)

    physics_str = _get_arg(launcher_args, "physics", None)
    physics_cfgs: list[PhysicsCfg] = []
    concrete_physics_cfgs: list[PhysicsCfg] = []
    effective_cfg: Any = cfg
    has_ovrtx = "newton_rtx" in (_get_arg(launcher_args, "visualizer") or ())
    has_auto_rtx = False
    has_auto_physx = False
    has_kit_camera = False
    auto_rtx_locations: list[tuple[Any, Any, bool]] = []  # (parent, key, is_cam_renderer) for each auto RTX placeholder
    auto_physx_locations: list[tuple[PhysicsCfg, Any, Any, bool]] = []  # (node, parent, key, is_first_physics)
    launcher_types: list[str] = []
    visited: set[int] = set()

    def add_launcher_type(node):
        if getattr(node, "launcher_type", None):
            launcher_types.append(node.launcher_type)

    def visit(node, parent, key):
        nonlocal effective_cfg, has_ovrtx, has_auto_rtx, has_auto_physx, has_kit_camera
        if _is_auto_rtx_renderer(node):
            has_auto_rtx = True
            auto_rtx_locations.append((parent, key, isinstance(parent, CameraCfg) and key == "renderer_cfg"))

        if id(node) in visited:
            return
        visited.add(id(node))

        if isinstance(node, PhysicsCfg):
            if physics_str:
                node = make_physics_cfg(physics_str)
                if parent is not None:
                    setattr(parent, key, node)
                else:
                    effective_cfg = node
            physics_cfgs.append(node)
            if _is_auto_physx_physics(node):
                has_auto_physx = True
                auto_physx_locations.append((node, parent, key, len(physics_cfgs) == 1))
                return
            else:
                concrete_physics_cfgs.append(node)
        elif _is_ovrtx_renderer(node) or getattr(node, "visualizer_type", None) == "newton_rtx":
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
            visit(child, node, name)

    visit(cfg, None, None)

    has_physics = bool(physics_cfgs)
    config_scan = Scan(
        resolved_physics_cfg=physics_cfgs[0] if physics_cfgs else None,
        effective_cfg=effective_cfg,
        visualizer_intent=_get_visualizer_intent(cfg, launcher_args),
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
        use_isaac_sim = bool(_get_kit_runtime_sources(config_scan, launcher_args))
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

    use_isaac_sim = bool(_get_kit_runtime_sources(config_scan, launcher_args))
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


def _has_kit_visualizer(config_scan: Scan, launcher_args: argparse.Namespace | dict | None) -> bool:
    """Return whether the run uses the Kit visualizer; an explicit CLI selection overrides the config."""
    if _get_arg(launcher_args, "visualizer_explicit", False):
        return "kit" in (_get_arg(launcher_args, "visualizer") or ())
    return config_scan.visualizer_intent["has_kit_visualizer"]


def _get_kit_runtime_sources(config_scan: Scan, launcher_args: argparse.Namespace | dict | None) -> tuple[str, ...]:
    """Return the config and launcher components that require Isaac Sim / Kit."""
    kit_sources = []
    if config_scan.has_kit_physics:
        kit_sources.append("Isaac Sim PhysX physics (`PhysxCfg`)")
    if config_scan.has_kit_camera:
        kit_sources.append('a Kit-based renderer (`IsaacRtxRendererCfg`, `renderer_type="isaac_rtx"`)')
    if _has_kit_visualizer(config_scan, launcher_args):
        kit_sources.append('the Kit visualizer (`--visualizer kit` / `visualizer_type="kit"`)')
    if _get_arg(launcher_args, "experience", ""):
        kit_sources.append("an explicit Kit experience")
    if _get_livestream_mode(launcher_args) > 0:
        kit_sources.append("livestreaming")
    if _get_arg(launcher_args, "require_kit", False):
        kit_sources.append("the caller's explicit Kit requirement")
    if config_scan.needs_kit and not kit_sources:
        kit_sources.append("the default Isaac Sim / Kit runtime")

    return tuple(kit_sources)


def _format_runtime_sources(sources: tuple[str, ...]) -> str:
    """Format runtime sources as a readable list."""
    if len(sources) < 3:
        return " and ".join(sources)
    return f"{', '.join(sources[:-1])}, and {sources[-1]}"


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
            f"with Isaac Sim / Kit ({_format_runtime_sources(kit_sources)}).\n"
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
        f" with Isaac Sim / Kit ({_format_runtime_sources(kit_sources)}).\n"
        "\n"
        "To fix this, pick one of the following supported combinations:\n"
        "  * Keep Isaac Sim / Kit and switch the renderer:\n"
        "      use `IsaacRtxRendererCfg`, the Kit-compatible renderer\n"
        "  * Keep OVRTX (`OVRTXRendererCfg` or `--visualizer newton_rtx`) and remove every Kit source\n"
    )


def _resolve_distributed_device(cfg, launcher_args: argparse.Namespace | dict | None) -> None:
    """Set ``cfg.sim.device`` and the launcher ``device`` for distributed training.

    When ``--distributed`` restricts each process to one GPU, ``local_rank`` may exceed
    the visible device count, so the process falls back to the one GPU it can see.
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
    _set_arg(launcher_args, "device", device_str)
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

            * ``physics``: Backend selector applied to every physics config in *cfg*, see
              :func:`make_physics_cfg`.
            * ``require_kit``: Whether the caller needs Kit for a reason *cfg* cannot express, e.g.
              a tool that reaches a Kit-only extension API. This is additive -- it can only turn a
              kitless launch into a Kit one, never the reverse, so a config that already needs Kit
              still launches it when the key is absent or ``False``.
            * ``visualizer_intent``: Visualizer intent the config cannot express, e.g.
              ``{"has_kit_visualizer": True}``; it is combined with the intent of *cfg*.
    """
    if launcher_args is None:
        launcher_args = {}

    # The single walk: collect every signal, apply the --physics override, and
    # resolve the automatic PhysX and RTX placeholders.
    config_scan = scan(cfg, launcher_args)
    effective_cfg = config_scan.effective_cfg
    physics_cfg = config_scan.resolved_physics_cfg

    kit_sources = _get_kit_runtime_sources(config_scan, launcher_args)
    _validate_runtime(config_scan, kit_sources)
    needs_kit = bool(kit_sources)
    _set_arg(launcher_args, "kit_visualizer", _has_kit_visualizer(config_scan, launcher_args))

    # Honor --verbose / --info; the Kit launcher re-applies this level once Kit has started.
    apply_python_logging_level(resolve_python_logging_level(launcher_args))

    if needs_kit and (config_scan.has_kit_camera or config_scan.visualizer_intent.get("has_kit_streaming_view")):
        if not _get_arg(launcher_args, "enable_cameras", False):
            logger.info(
                "Auto-enabling camera rendering because the scene contains Kit camera sensors "
                "or a Kit visualizer with streaming_view=True."
            )
            _set_arg(launcher_args, "enable_cameras", True)

    # Resolve distributed device early, before any launcher or physics init.
    _resolve_distributed_device(effective_cfg, launcher_args)

    # Start the launchers the resolved config names, plus Kit and OVRTX for needs that no config names
    # (e.g. a default-renderer camera, ``--viz kit`` or ``--viz newton_rtx``); Kit starts first.
    launcher_types = [_KIT_LAUNCHER] if needs_kit else []
    launcher_types += config_scan.launcher_types
    if config_scan.has_ovrtx:
        # validated above: OVRTX never shares the process with Kit
        launcher_types.append(OVRTXRendererCfg.launcher_type)
    launchers = [string_to_callable(launcher_type)(launcher_args) for launcher_type in dict.fromkeys(launcher_types)]
    for launcher in launchers:
        # the runtime may refine the device choice made by _resolve_distributed_device
        sim_cfg = getattr(effective_cfg, "sim", None)
        if sim_cfg is not None and launcher.device is not None:
            sim_cfg.device = launcher.device
    # after the launchers, so a started Kit already backs the settings
    _sync_visualizer_cli_settings(launcher_args)

    exit_code = 0
    try:
        # The import stays after the Kit launch decision. With no selected profile this is a
        # no-op; with one, it installs process-wide OmniClient routing before user code runs.
        from ..utils.assets import configure_storage_profile

        configure_storage_profile()
        yield physics_cfg
    except Exception:
        exit_code = 1
        import traceback

        traceback.print_exc()
        raise
    finally:
        for launcher in reversed(launchers):
            launcher.close(exit_code)
