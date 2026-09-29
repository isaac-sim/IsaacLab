# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resolve configured consumers, then launch their required process runtimes."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from dataclasses import is_dataclass
from types import SimpleNamespace
from typing import Any

from isaaclab_newton.physics import NewtonCfg, VBDSolverCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_ov.renderers import OVRTXRendererCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg
from isaaclab_visualizers.kit import KitVisualizerCfg
from isaaclab_visualizers.presets import MultiBackendVisualizerCfg

from ..physics.physics_manager_cfg import PhysicsCfg, PhysxAutoCfg, _resolve_physx_auto_cfg
from ..renderers.renderer_cfg import RendererCfg
from ..sensors.camera.camera_cfg import CameraCfg
from ..sim.simulation_cfg import SimulationCfg
from ..utils.device import set_cuda_device
from ..utils.presets import PresetCfg, resolve_presets
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


def _iter_runtime_configs(cfg: Any) -> Iterator[Any]:
    """Visit active config objects once, excluding inactive backend choices and viewer defaults."""
    visited = set()

    def visit(node):
        if callable(node) or id(node) in visited:
            return
        visited.add(id(node))
        if is_dataclass(node) or isinstance(node, (argparse.Namespace, SimpleNamespace)):
            if isinstance(node, (SimulationCfg, PhysicsCfg, RendererCfg, VisualizerCfg, CameraCfg)):
                yield node
            if isinstance(node, (PhysicsCfg, RendererCfg, VisualizerCfg)):
                return
            children = (value for name, value in vars(node).items() if name != "default_visualizer_cfg")
        elif isinstance(node, dict):
            children = node.values()
        elif isinstance(node, (list, tuple)):
            children = node
        else:
            return
        for child in children:
            yield from visit(child)

    yield from visit(cfg)


def _livestream_mode(args: dict) -> int:
    """Read the CLI/environment streaming choice without changing the caller's arguments."""
    mode = args.get("livestream", -1)
    mode = int(os.environ.get("LIVESTREAM", 0) if mode is None or int(mode) < 0 else mode)
    if mode not in (0, 1, 2):
        raise ValueError(f"Invalid livestream mode: {mode}. Expected 0 (disabled), 1, or 2.")
    return mode


def _kit_runtime_sources(configs: tuple, args: dict) -> tuple[str, ...]:
    """Name the declared requirements that select Kit, before resolving automatic backends."""
    sources = []
    if not any(isinstance(cfg, PhysicsCfg) for cfg in configs):
        sources.append("the default Isaac Sim / Kit runtime")
    for cfg in configs:
        if isinstance(cfg, PhysxCfg) or isinstance(cfg, PhysxAutoCfg) and cfg.ovphysx is None:
            sources.append("Isaac Sim PhysX physics (PhysxCfg)")
        elif isinstance(cfg, CameraCfg) and (
            cfg.renderer_cfg is None or cfg.renderer_cfg.renderer_type in ("default", "isaac_rtx")
        ):
            sources.append("a Kit-based renderer (IsaacRtxRendererCfg)")
        elif isinstance(cfg, (PhysicsCfg, RendererCfg, VisualizerCfg)) and cfg.launcher_type == _KIT_LAUNCHER:
            sources.append("the Kit visualizer" if isinstance(cfg, VisualizerCfg) else type(cfg).__name__)
    for enabled, source in (
        (args.get("experience"), "an explicit Kit experience"),
        (_livestream_mode(args), "livestreaming"),
        (args.get("require_kit"), "the caller's explicit Kit requirement"),
    ):
        if enabled:
            sources.append(source)
    return tuple(dict.fromkeys(sources))


def _replace_configs(cfg: Any, replacements: dict[int, Any]) -> Any:
    """Replace selected config objects in place, preserving aliases and tuple-held references."""
    resolved = {}

    def replace(node):
        if id(node) in resolved:
            return resolved[id(node)]
        if id(node) in replacements:
            replacement = replacements[id(node)]
            if replacement is not node:
                resolved[id(node)] = replace(replacement)
                return resolved[id(node)]
        if callable(node) or isinstance(node, (PhysicsCfg, RendererCfg, VisualizerCfg)):
            return node
        if is_dataclass(node) or isinstance(node, (argparse.Namespace, SimpleNamespace)):
            fields = vars(node)
        elif isinstance(node, (dict, list)):
            fields = node
        elif isinstance(node, tuple):
            replacement = tuple(replace(child) for child in node)
            resolved[id(node)] = replacement
            return replacement
        else:
            return node
        resolved[id(node)] = node
        for name, child in enumerate(fields) if isinstance(fields, list) else fields.items():
            if name != "default_visualizer_cfg":
                fields[name] = replace(child)
        return node

    return replace(cfg)


def resolve_simulation_cfg(cfg: Any, launcher_args: argparse.Namespace | dict | None = None) -> Any:
    """Apply CLI choices and resolve automatic backends before starting any runtime.

    Config objects are updated in place; return the selected config when the root itself changes.
    Launcher arguments are read-only. Calling this again preserves selected visualizer settings.

    Args:
        cfg: Simulation config or a tree containing physics, renderer, and visualizer configs.
        launcher_args: Parsed launcher arguments.

    Returns:
        The config tree with concrete backend and visualizer choices.
    """
    args = vars(launcher_args) if isinstance(launcher_args, argparse.Namespace) else launcher_args or {}
    max_visible = args.get("max_visible_envs")
    if max_visible is not None and max_visible < 0:
        raise ValueError("--max_visible_envs must be non-negative.")
    livestream = _livestream_mode(args)
    cfg = SimulationCfg() if cfg is None else cfg
    physics = _make_physics_cfg(args["physics"]) if args.get("physics") else None
    selection = args.get("visualizer")
    if isinstance(selection, str):
        selection = [name.strip() for name in selection.split(",")]

    for node in _iter_runtime_configs(cfg):
        if not isinstance(node, SimulationCfg):
            continue
        if node.physics is None and physics is not None:
            node.physics = physics
        if args.get("device"):
            node.device = args["device"]
        if selection is not None or args.get("visualizer_explicit", False):
            choices = node.visualizer_cfgs
            if not isinstance(choices, PresetCfg):
                # CLI choices override concrete configs too; preserve settings of reselected viewers.
                choices = MultiBackendVisualizerCfg()
                viewers = node.visualizer_cfgs
                for viewer in viewers if isinstance(viewers, list) else [viewers] if viewers else []:
                    vars(choices)[viewer.visualizer_type] = viewer
            node.visualizer_cfgs = resolve_presets(choices, overrides={"": selection or "none"})
        else:
            node.visualizer_cfgs = resolve_presets(node.visualizer_cfgs)
        if livestream or args.get("xr", False):
            viewers = node.visualizer_cfgs
            if livestream and viewers == []:
                raise ValueError("Livestreaming requires the Kit visualizer; visualizer=none disables it.")
            viewers = viewers if isinstance(viewers, list) else [viewers] if viewers else []
            if not any(viewer.launcher_type == _KIT_LAUNCHER for viewer in viewers):
                node.visualizer_cfgs = [*viewers, KitVisualizerCfg(headless=bool(args.get("xr", False)))]

    cfg = resolve_presets(cfg)
    configs = tuple(_iter_runtime_configs(cfg))
    replacements = {id(node): physics for node in configs if isinstance(node, PhysicsCfg) and physics is not None}
    configs = tuple(replacements.get(id(node), node) for node in configs)
    kit_sources = _kit_runtime_sources(configs, args)
    for node in configs:
        if isinstance(node, PhysxAutoCfg):
            replacements[id(node)] = _resolve_physx_auto_cfg(node, bool(kit_sources))
        elif isinstance(node, RendererCfg) and node.renderer_type == "auto_rtx":
            replacements[id(node)] = IsaacRtxRendererCfg() if kit_sources else OVRTXRendererCfg()
        elif isinstance(node, VisualizerCfg) and max_visible is not None:
            node.max_visible_envs = max_visible

    cfg = _replace_configs(cfg, replacements)
    if kit_sources:
        for node in _iter_runtime_configs(cfg):
            if isinstance(node, OvPhysxCfg):
                conflict = "OvPhysX physics (OvPhysxCfg)"
            elif isinstance(node, RendererCfg | VisualizerCfg) and node.launcher_type == OVRTXRendererCfg.launcher_type:
                conflict = "the OVRTX runtime (OVRTXRendererCfg or the newton_rtx visualizer)"
            else:
                continue
            raise ValueError(
                f"Invalid backend combination: {conflict} cannot be used together with Isaac Sim / Kit "
                f"({', '.join(kit_sources)}). Use Kit-compatible physics with IsaacRtxRendererCfg, "
                "or remove the Kit requirements and use a kitless renderer/visualizer."
            )
    return cfg


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

    Resolves configured consumers before starting their required runtimes. Each runtime starts
    once and closes on exit. Cameras are auto-enabled for Kit-renderer sensors.

    Resolve viewer and physics choices on the same config used to construct the simulation::

        sim_cfg = SimulationCfg()
        with launch_simulation(sim_cfg, args_cli):
            sim = SimulationContext(sim_cfg)

    Yields the resolved physics config when a caller needs to adjust solver settings before construction.

    Args:
        cfg: Config tree declaring backend, renderer, and sensor requirements.
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
    cfg = resolve_simulation_cfg(cfg, launcher_args)
    configs = tuple(_iter_runtime_configs(cfg))
    sim_cfg = next((node for node in configs if isinstance(node, SimulationCfg)), None)
    physics_cfg = next((node for node in configs if isinstance(node, PhysicsCfg)), None)
    kit_viewers = [node for node in configs if isinstance(node, KitVisualizerCfg)]
    launcher_args["livestream"] = _livestream_mode(launcher_args)
    launcher_args["kit_visualizer"] = any(not viewer.headless for viewer in kit_viewers)
    launcher_args["enable_cameras"] = (
        launcher_args.get("enable_cameras", False)
        or any(
            isinstance(node, CameraCfg)
            and (node.renderer_cfg is None or node.renderer_cfg.renderer_type in ("default", "isaac_rtx"))
            for node in configs
        )
        or any(viewer.streaming_view for viewer in kit_viewers)
    )

    apply_python_logging_level(_resolve_python_logging_level(launcher_args))
    _resolve_distributed_device(sim_cfg, launcher_args)

    launcher_types = [_KIT_LAUNCHER] if _kit_runtime_sources(configs, launcher_args) else []
    launcher_types += [
        node.launcher_type
        for node in configs
        if isinstance(node, (PhysicsCfg, RendererCfg, VisualizerCfg)) and node.launcher_type
    ]
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
