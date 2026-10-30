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
import traceback
import warnings
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import is_dataclass
from types import SimpleNamespace
from typing import Any

import torch
from isaaclab_newton.physics import NewtonCfg, VBDSolverCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_ov.renderers import OVRTXRendererCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg
from isaaclab_visualizers.kit import KitBackendCfg
from isaaclab_visualizers.newton import NewtonGLBackendCfg, NewtonRTXBackendCfg
from isaaclab_visualizers.rerun import RerunBackendCfg
from isaaclab_visualizers.viser import ViserBackendCfg

from ..envs.utils.video_recorder_cfg import CAPTURE_VISUALIZER_TYPES, VideoRecorderCfg, parse_video_source
from ..physics.physics_manager_cfg import PhysicsCfg, PhysxAutoCfg, _resolve_physx_auto_cfg
from ..renderers.renderer_cfg import RendererCfg
from ..scene.interactive_scene_cfg import InteractiveSceneCfg
from ..sensors.camera.camera_cfg import CameraCfg
from ..sim.simulation_cfg import SimulationCfg
from ..sim.spawners.sensors.sensors_cfg import PinholeCameraCfg
from ..utils.assets import configure_storage_profile
from ..utils.device import set_cuda_device
from ..utils.images import sensor_key_for_gt_type
from ..utils.math import create_rotation_matrix_from_view, quat_from_matrix
from ..utils.string import string_to_callable
from ..visualizers import ImageViewCfg, PerspectiveCameraCfg, SceneCameraCfg, VisualizerBackendCfg, VisualizerCfg
from ..visualizers.visualizer_cfg import VISUALIZER_ALIASES, migrate_visualizer_cfgs
from .logging_utils import apply_python_logging_level, ensure_console_handlers, resolve_python_logging_level
from .settings_manager import get_settings_manager

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


def _walk_runtime_configs(cfg: Any, replacements: dict[int, Any] | None = None) -> tuple[Any, ...]:
    """Collect runtime configs once per object, preserving aliases when replacing automatic backends."""
    replacements = {} if replacements is None else replacements
    visited = {}
    configs = []

    def visit(node):
        if id(node) in visited:
            return visited[id(node)]
        if id(node) in replacements and replacements[id(node)] is not node:
            visited[id(node)] = visit(replacements[id(node)])
            return visited[id(node)]
        visited[id(node)] = node
        if callable(node):
            return node
        if isinstance(node, (PhysicsCfg, RendererCfg, VisualizerBackendCfg)):
            configs.append(node)
            return node
        if isinstance(node, (SimulationCfg, InteractiveSceneCfg, VisualizerCfg, CameraCfg, VideoRecorderCfg)):
            configs.append(node)
        if isinstance(node, tuple):
            children = tuple(visit(child) for child in node)
            if any(old is not new for old, new in zip(node, children, strict=True)):
                visited[id(node)] = children
            return visited[id(node)]
        if is_dataclass(node) or isinstance(node, (argparse.Namespace, SimpleNamespace)):
            children = vars(node)
            items = ((name, value) for name, value in children.items() if not name.startswith("_"))
        elif isinstance(node, (Mapping, list)):
            children = node
            items = enumerate(node) if isinstance(node, list) else node.items()
        else:
            return node
        for name, child in items:
            replacement = visit(child)
            if replacement is not child:
                children[name] = replacement
        return node

    visit(cfg)
    return tuple(configs)


def validate_launch(  # noqa: C901
    cfg, launcher_args: argparse.Namespace | dict | None = None
) -> tuple[tuple[Any, ...], tuple[str, ...]]:
    """Prepare configured consumers and reject incompatible runtimes without starting them.

    Applies launcher selections and resolves automatic physics/renderer configurations in place.
    Task-run enumeration and :func:`launch_simulation` use this same preparation step.

    Args:
        cfg: Configuration tree declaring physics, renderers, visualizers, and recorders.
        launcher_args: Parsed launcher arguments; derived runtime options are written back.

    Returns:
        Resolved runtime configurations and the components requiring Isaac Sim / Kit.

    Raises:
        ValueError: If selections are invalid or configured runtimes cannot share one process.
    """
    launcher_args = {} if launcher_args is None else launcher_args
    launcher_args = vars(launcher_args) if isinstance(launcher_args, argparse.Namespace) else launcher_args
    max_visible = launcher_args.get("max_visible_envs")
    if max_visible is not None and max_visible < 0:
        raise ValueError("--max_visible_envs must be non-negative.")
    livestream = launcher_args.get("livestream", -1)
    livestream = int(os.environ.get("LIVESTREAM", 0) if livestream is None or int(livestream) < 0 else livestream)
    if livestream not in (0, 1, 2):
        raise ValueError(f"Invalid livestream mode: {livestream}. Expected 0 (disabled), 1, or 2.")
    cfg = SimulationCfg() if cfg is None else cfg
    physics = None
    if launcher_args.get("physics"):
        physics_options = {
            "physx": lambda: PhysxAutoCfg(isaacsim_physx=PhysxCfg(), ovphysx=OvPhysxCfg()),
            "isaacsim_physx": PhysxCfg,
            "ovphysx": OvPhysxCfg,
            "newton_mjwarp": NewtonCfg,
            "newton_vbd": lambda: NewtonCfg(solver_cfg=VBDSolverCfg()),
        }
        physics = physics_options[launcher_args["physics"]]()
    visualizer_types = {
        "kit": KitBackendCfg,
        "newton_gl": NewtonGLBackendCfg,
        "newton_rtx": NewtonRTXBackendCfg,
        "rerun": RerunBackendCfg,
        "viser": ViserBackendCfg,
    }
    selection = launcher_args.get("visualizer")
    if isinstance(selection, str):
        selection = selection.split(",")
    if selection is not None:
        for name in selection:
            if name in VISUALIZER_ALIASES:
                warnings.warn(
                    f"Visualizer '{name}' is deprecated. Use '{VISUALIZER_ALIASES[name]}' instead.",
                    FutureWarning,
                    stacklevel=3,
                )
        selection = list(dict.fromkeys(VISUALIZER_ALIASES.get(name, name) for name in selection))
        if selection == ["none"]:
            selection = []
        if any(name not in visualizer_types for name in selection):
            raise ValueError(f"Invalid visualizer selection: {selection}. Expected one of {tuple(visualizer_types)}.")
        if livestream and not selection:
            raise ValueError("Livestreaming requires the Kit visualizer; visualizer=none disables it.")

    configs = _walk_runtime_configs(cfg)
    configured = []
    simulations = [node for node in configs if isinstance(node, SimulationCfg)]
    for node in simulations:
        if node.physics is None and physics is not None:
            node.physics = physics
        if launcher_args.get("device"):
            node.device = launcher_args["device"]
        node.visualizer_cfgs = migrate_visualizer_cfgs(node.visualizer_cfgs, node.default_visualizer_cfg)
        node.default_visualizer_cfg = None
        declarations = node.visualizer_cfgs or [VisualizerCfg()]
        viewers = []
        for viewer in node.visualizer_cfgs:
            if isinstance(viewer.backend, list) and viewer.backend:
                viewers.extend(viewer.replace(backend=backend) for backend in viewer.backend)
            elif not viewer.backend and selection:
                viewers.extend(viewer.replace(backend=visualizer_types[name]()) for name in selection)
            else:
                viewers.append(viewer)
        node.visualizer_cfgs = viewers
        configured.extend(node.visualizer_cfgs)
        template = node.visualizer_cfgs[0] if node.visualizer_cfgs else VisualizerCfg()
        if selection is not None:
            viewers = []
            for name in selection:
                viewers.extend(
                    [
                        viewer
                        for viewer in node.visualizer_cfgs
                        if viewer.backend and viewer.backend.visualizer_type == name
                    ]
                    or [viewer.replace(backend=visualizer_types[name]()) for viewer in declarations]
                )
            node.visualizer_cfgs = viewers
        if livestream or launcher_args.get("xr", False):
            if not any(isinstance(viewer.backend, KitBackendCfg) for viewer in node.visualizer_cfgs):
                node.visualizer_cfgs.append(
                    template.replace(backend=KitBackendCfg(), headless=bool(launcher_args.get("xr", False)))
                )
        node.visualizer_cfgs = [viewer for viewer in node.visualizer_cfgs if viewer.backend]

    viewers = [viewer for sim in simulations for viewer in sim.visualizer_cfgs]
    for recorder in (node for node in configs if isinstance(node, VideoRecorderCfg)):
        if isinstance(recorder.source, ImageViewCfg):
            if isinstance(recorder.source.source, str):
                continue
            owners = [viewer for viewer in configured if viewer.view is recorder.source]
            if len(owners) != 1:
                raise ValueError("A recorded perspective view requires exactly one configured visualizer.")
            producer = next((viewer for viewer in viewers if viewer.view is recorder.source), owners[0])
            if not producer.backend:
                producer = producer.replace(backend=NewtonGLBackendCfg())
        else:
            kind, name, channel = parse_video_source(recorder.source)
            if kind == "sensor":
                continue
            name = name or next(
                (v.backend.visualizer_type for v in viewers if v.backend.visualizer_type in CAPTURE_VISUALIZER_TYPES),
                "newton_gl",
            )
            if name not in CAPTURE_VISUALIZER_TYPES:
                raise ValueError(
                    f"Cannot record video source {recorder.source!r}: the {name!r} visualizer streams to a viewer and"
                    " has no frame capture. Use 'viz:kit', 'viz:newton_gl', 'viz:newton_rtx', or 'sensor:<name>'."
                )
            recorder.source = f"viz:{name}:{channel}" if channel else f"viz:{name}"
            producer = next(
                (
                    viewer
                    for viewer in viewers + configured
                    if viewer.backend and viewer.backend.visualizer_type == name
                ),
                None,
            )
            if producer is None:
                template = configured[0] if configured else VisualizerCfg()
                producer = template.replace(backend=visualizer_types[name]())
            if channel == "streaming_view":
                producer.streaming_view = True
            if name == "kit":
                launcher_args["enable_cameras"] = True
        if not any(viewer is producer for viewer in viewers):
            producer = producer.replace(headless=True)
            simulations[0].visualizer_cfgs.append(producer)
            viewers.append(producer)

    # Declare RTX perspective cameras before scene construction; runtime viewers only borrow their output.
    scene_cfg = next((node for node in configs if isinstance(node, InteractiveSceneCfg)), None)
    for viewer_index, visualizer in enumerate(viewers):
        if not isinstance(visualizer.backend, NewtonRTXBackendCfg) or scene_cfg is None:
            continue
        sources = (
            [visualizer.view.source] if visualizer.view is not None else list(visualizer.cameras or [
                PerspectiveCameraCfg(eye=visualizer.eye, lookat=visualizer.lookat, focal_length=visualizer.focal_length)
            ])
        )  # fmt: skip
        for camera_index, source in enumerate(sources):
            if not isinstance(source, PerspectiveCameraCfg):
                continue
            name = f"viewer_camera_{viewer_index}_{camera_index}"
            width, height = visualizer.window.size
            channels = visualizer.view.channels if visualizer.view is not None else visualizer.streaming_gt_types
            rotation = quat_from_matrix(create_rotation_matrix_from_view(
                torch.tensor([source.eye]), torch.tensor([source.lookat]), "Z", device="cpu"
            ))[0].tolist()  # fmt: skip
            camera = CameraCfg(
                prim_path=f"/World/{name}", width=width, height=height,
                data_types=["rgba" if channel == "rgb" else sensor_key_for_gt_type(channel) for channel in channels],
                spawn=PinholeCameraCfg(
                    focal_length=source.focal_length, vertical_aperture=15.2908,
                    horizontal_aperture=15.2908 * width / height,
                ),
                offset=CameraCfg.OffsetCfg(pos=source.eye, rot=tuple(rotation), convention="opengl"),
                renderer_cfg=visualizer.backend.renderer_cfg, background_color=visualizer.background_color,
                render_settings={"omni:rtx:quality": ("Int", 0), **visualizer.backend.render_settings},
            )  # fmt: skip
            vars(scene_cfg)[name] = camera
            if visualizer.view is not None:
                visualizer.view.source = name
            else:
                sources[camera_index] = SceneCameraCfg(prim_path=camera.prim_path)
        if visualizer.view is None:
            visualizer.cameras = sources
            visualizer.streaming_view = True

    configs = _walk_runtime_configs(cfg)
    replacements = {id(node): physics for node in configs if isinstance(node, PhysicsCfg) and physics is not None}
    configs = tuple(replacements.get(id(node), node) for node in configs)
    # Decide the process runtime before resolving automatic physics and renderer backends.
    kit_sources = []
    launcher_args["kit_visualizer"] = False
    launcher_args.setdefault("enable_cameras", False)
    if not any(isinstance(node, PhysicsCfg) for node in configs):
        kit_sources.append("the default Isaac Sim / Kit runtime")
    for node in configs:
        if isinstance(node, PhysxAutoCfg) and node.ovphysx is None:
            kit_sources.append("Isaac Sim PhysX physics (PhysxCfg)")
        elif isinstance(node, CameraCfg) and (
            node.renderer_cfg is None or node.renderer_cfg.renderer_type in ("default", "isaac_rtx")
        ):
            kit_sources.append("a Kit-based renderer (IsaacRtxRendererCfg)")
        elif isinstance(node, (PhysicsCfg, RendererCfg, VisualizerBackendCfg)) and node.launcher_type == _KIT_LAUNCHER:
            kit_sources.append("the Kit visualizer" if isinstance(node, VisualizerBackendCfg) else type(node).__name__)
        elif isinstance(node, VisualizerCfg) and isinstance(node.backend, KitBackendCfg):
            launcher_args["kit_visualizer"] |= not node.headless
            launcher_args["enable_cameras"] |= node.streaming_view or node.headless
    for enabled, source in (
        (launcher_args.get("experience"), "an explicit Kit experience"),
        (livestream, "livestreaming"),
        (launcher_args.get("require_kit"), "the caller's explicit Kit requirement"),
    ):
        if enabled:
            kit_sources.append(source)
    kit_sources = tuple(dict.fromkeys(kit_sources))
    for node in configs:
        if isinstance(node, PhysxAutoCfg):
            replacements[id(node)] = _resolve_physx_auto_cfg(node, bool(kit_sources))
        elif isinstance(node, RendererCfg) and node.renderer_type == "auto_rtx":
            replacements[id(node)] = IsaacRtxRendererCfg() if kit_sources else OVRTXRendererCfg()
        elif isinstance(node, VisualizerCfg) and max_visible is not None:
            node.max_visible_envs = max_visible

    configs = _walk_runtime_configs(cfg, replacements)
    if kit_sources:
        for node in configs:
            if isinstance(node, OvPhysxCfg):
                conflict = "OvPhysX physics (OvPhysxCfg)"
            elif isinstance(node, (OVRTXRendererCfg, NewtonRTXBackendCfg)):
                conflict = "the OVRTX runtime (OVRTXRendererCfg or the newton_rtx visualizer)"
            else:
                continue
            raise ValueError(
                f"Invalid backend combination: {conflict} cannot be used together with Isaac Sim / Kit "
                f"({', '.join(kit_sources)}). Use Kit-compatible physics with IsaacRtxRendererCfg, "
                "or remove the Kit requirements and use a kitless renderer/visualizer."
            )
    launcher_args["livestream"] = livestream
    launcher_args["enable_cameras"] |= any(
        isinstance(node, CameraCfg)
        and (node.renderer_cfg is None or node.renderer_cfg.renderer_type in ("default", "isaac_rtx"))
        for node in configs
    )

    return configs, kit_sources


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
        cfg: Concrete config tree declaring backend, renderer, and sensor requirements.
        launcher_args: Parsed launcher arguments, typically the script's ``args_cli``. Besides the
            arguments added by :func:`add_launcher_args`, the following keys are read when a script
            contributes them:

            * ``physics``: Backend selector applied to every physics config in *cfg*.
            * ``require_kit``: Whether the caller needs Kit for a reason *cfg* cannot express, e.g.
              a tool that reaches a Kit-only extension API. This is additive -- it can only turn a
              kitless launch into a Kit one, never the reverse, so a config that already needs Kit
              still launches it when the key is absent or ``False``.
    """
    launcher_args = {} if launcher_args is None else launcher_args
    launcher_args = vars(launcher_args) if isinstance(launcher_args, argparse.Namespace) else launcher_args
    configs, kit_sources = validate_launch(cfg, launcher_args)
    sim_cfg = next((node for node in configs if isinstance(node, SimulationCfg)), None)
    physics_cfg = next((node for node in configs if isinstance(node, PhysicsCfg)), None)

    python_logging_level = resolve_python_logging_level(launcher_args)
    apply_python_logging_level(python_logging_level)
    ensure_console_handlers(python_logging_level)

    # Resolve the device before any launcher or physics init: --device, else this rank's GPU, else the config's.
    if launcher_args.get("distributed", False):
        local_rank = int(os.getenv("LOCAL_RANK", "0")) + int(os.getenv("JAX_LOCAL_RANK", "0"))
        num_visible_gpus = torch.cuda.device_count()
        # A launcher may mask each rank to one GPU; otherwise use its local device index.
        launcher_args["device"] = f"cuda:{local_rank}" if local_rank < num_visible_gpus else "cuda:0"
        set_cuda_device(launcher_args["device"])
        logger.info("Distributed device resolved to %s (local_rank=%d)", launcher_args["device"], local_rank)
    if (
        sys.platform == "darwin"
        and launcher_args.get("device") == "cpu"
        and not launcher_args.get("device_explicit", False)
    ):
        # the --device default on macOS, which has no CUDA
        logger.warning("No --device given; using 'cpu' because macOS has no CUDA.")
    if sim_cfg is not None:
        launcher_args["device"] = launcher_args.get("device") or sim_cfg.device

    launcher_types = [_KIT_LAUNCHER] if kit_sources else []
    launcher_types += [
        node.launcher_type
        for node in configs
        if isinstance(node, (PhysicsCfg, RendererCfg, VisualizerBackendCfg)) and node.launcher_type
    ]
    launchers = [string_to_callable(launcher_type)(launcher_args) for launcher_type in dict.fromkeys(launcher_types)]
    # after the launchers, so a started Kit already backs the settings
    device = launcher_args.get("device")
    for launcher in launchers:
        device = launcher.device or device
    if device is not None and device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Device '{device}' requires CUDA, which is not available; pass --device cpu.")
    if device == "cuda":
        cuda_device = get_settings_manager().get("/physics/cudaDevice")
        device = f"cuda:{max(0, int(cuda_device) if cuda_device is not None else 0)}"
    if device is not None:
        launcher_args["device"] = device
        if sim_cfg is not None:
            sim_cfg.device = device

    from ..sim.simulation_context import SimulationContext

    previous_context = SimulationContext.instance()
    exit_code = 0
    try:
        # With no selected profile this is a no-op; with one, it installs process-wide OmniClient
        # routing before user code runs.
        try:
            configure_storage_profile()
            yield physics_cfg
        finally:
            # Release runtime consumers before a launcher can terminate the process.
            context = SimulationContext.instance()
            if context is not None and context is not previous_context:
                SimulationContext.clear_instance()
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
