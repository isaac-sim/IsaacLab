# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for detecting and launching the appropriate simulation backend."""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

from isaaclab.app.settings_manager import get_settings_manager
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


def _is_kit_physics(node) -> bool:
    """True when the node is a Kit-required physics config (Isaac Sim PhysX)."""
    return isinstance(node, PhysicsCfg) and type(node).__name__ == "PhysxCfg"


def _is_ovphysx_physics(node) -> bool:
    """True when the node is an OvPhysX physics config."""
    return isinstance(node, PhysicsCfg) and type(node).__name__ == "OvPhysxCfg"


def _is_ovrtx_renderer(node) -> bool:
    """True when the node is an OVRTX renderer config."""
    return isinstance(node, RendererCfg) and getattr(node, "renderer_type", None) == "ovrtx"


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


def _get_launcher_arg(launcher_args: argparse.Namespace | dict | None, name: str, default=None):
    """Read *name* from argparse namespace or dict launcher args."""
    if isinstance(launcher_args, argparse.Namespace):
        return getattr(launcher_args, name, default)
    if isinstance(launcher_args, dict):
        return launcher_args.get(name, default)
    return default


def _set_launcher_arg(launcher_args: argparse.Namespace | dict | None, name: str, value) -> None:
    """Write *name* to argparse namespace or dict launcher args."""
    if isinstance(launcher_args, argparse.Namespace):
        setattr(launcher_args, name, value)
    elif isinstance(launcher_args, dict):
        launcher_args[name] = value


def _get_livestream_mode(launcher_args: argparse.Namespace | dict | None) -> int:
    """Return effective livestream mode using the same CLI-over-env precedence as AppLauncher."""
    livestream_arg = _get_launcher_arg(launcher_args, "livestream", -1)
    if livestream_arg is not None and int(livestream_arg) >= 0:
        return int(livestream_arg)
    return int(os.environ.get("LIVESTREAM", 0))


def _ensure_livestream_kit_visualizer(launcher_args: argparse.Namespace | dict | None) -> None:
    """Request the Kit visualizer when livestreaming needs a video-producing viewport."""
    if launcher_args is None or _get_livestream_mode(launcher_args) == 0:
        return

    visualizer_explicit = bool(_get_launcher_arg(launcher_args, "visualizer_explicit", False))
    visualizers = _get_launcher_arg(launcher_args, "visualizer")
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
    _set_launcher_arg(launcher_args, "visualizer", requested_visualizers)


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
    # PresetCfg renderers (e.g. MultiBackendRendererCfg) are resolved during
    # environment construction when the physics backend is known (see
    # resolve_task_config and preset resolution in presets.py).  At this
    # stage we assume they will match the physics backend, so not
    # necessarily Kit.
    from isaaclab_tasks.utils import PresetCfg

    if isinstance(renderer_cfg, PresetCfg):
        return False
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
    is_kitless, has_kit_cameras = _scan_config(env_cfg, [_is_kitless_physics, _is_kit_camera])
    needs_kit = has_kit_cameras or not is_kitless
    visualizer_types = _get_visualizer_types(launcher_args)
    if "kit" in visualizer_types:
        needs_kit = True
    return needs_kit, has_kit_cameras, visualizer_types


def validate_runtime_compatibility(
    env_cfg,
    launcher_args: argparse.Namespace | dict | None = None,
) -> None:
    """Validate that the resolved physics, renderer, and visualizer combination is supported.

    The OVRTX renderer (``OVRTXRendererCfg``, ``renderer_type="ovrtx"``) is a kitless
    renderer that runs without Isaac Sim / Omniverse Kit. Combining it with Kit-based
    runtimes — Isaac Sim PhysX physics (``PhysxCfg``) or the Kit visualizer
    (``--visualizer kit`` / a ``visualizer_cfgs`` entry with ``visualizer_type="kit"``) —
    is unsupported. OvPhysX physics (``OvPhysxCfg``) also cannot share a process with
    the Kit visualizer.

    Args:
        env_cfg: Resolved environment config (e.g. from :func:`resolve_task_config`).
        launcher_args: Optional CLI args. Inspected for ``--visualizer kit``.

    Raises:
        ValueError: If the OVRTX renderer is combined with Kit-based physics or the
            Kit visualizer, or if OvPhysX physics is combined with the Kit visualizer.
    """
    has_kit_physics, has_ovrtx_renderer, has_ovphysx_physics = _scan_config(
        env_cfg, [_is_kit_physics, _is_ovrtx_renderer, _is_ovphysx_physics]
    )
    visualizer_intent = _compute_visualizer_intent(env_cfg)
    visualizer_types = _get_visualizer_types(launcher_args)
    has_kit_visualizer = "kit" in visualizer_types or visualizer_intent.get("has_kit_visualizer", False)

    if has_ovphysx_physics and has_kit_visualizer:
        raise ValueError(
            "Invalid backend combination: OvPhysX physics (`OvPhysxCfg`) is kitless and cannot be used together "
            'with the Kit visualizer (`--visualizer kit` / `visualizer_type="kit"`). Use a kitless visualizer '
            "such as `--visualizer newton`, `--visualizer rerun`, or `--visualizer viser`, or omit the visualizer "
            "argument for headless execution."
        )

    if not has_ovrtx_renderer:
        return

    if not has_kit_physics and not has_kit_visualizer:
        return

    sources = []
    if has_kit_physics:
        sources.append("Isaac Sim PhysX physics (`PhysxCfg`)")
    if has_kit_visualizer:
        sources.append('the Kit visualizer (`--visualizer kit` / `visualizer_type="kit"`)')
    sources_text = " and ".join(sources)

    raise ValueError(
        "Invalid backend combination: the OVRTX renderer (`OVRTXRendererCfg`,"
        ' `renderer_type="ovrtx"`) is a kitless renderer and cannot be used together'
        f" with Isaac Sim / Kit ({sources_text}).\n"
        "\n"
        "To fix this, pick one of the following supported combinations:\n"
        "  * Keep Isaac Sim / Kit and switch the renderer:\n"
        "      presets=isaacsim_rtx_renderer\n"
        "    (uses `IsaacRtxRendererCfg`, the Kit-compatible renderer.)\n"
        "  * Keep the OVRTX renderer and switch to a kitless physics backend\n"
        "    (and avoid `--visualizer kit`):\n"
        "      presets=newton_mjwarp,ovrtx_renderer\n"
    )


def _resolve_distributed_device(
    env_cfg,
    launcher_args: argparse.Namespace | dict | None,
) -> None:
    """Set ``env_cfg.sim.device`` for distributed training.

    When ``--distributed`` is active and CUDA_VISIBLE_DEVICES restricts each
    process to a single GPU, ``local_rank`` may exceed the visible device count.
    This helper applies the same fallback logic used by :class:`AppLauncher` so
    that **training scripts do not need their own device-resolution code**.

    For the Kit path, :func:`launch_simulation` additionally propagates
    ``AppLauncher.device`` after creation; this function handles the early
    (pre-AppLauncher) and kitless cases.
    """
    distributed = False
    if isinstance(launcher_args, argparse.Namespace):
        distributed = getattr(launcher_args, "distributed", False)
    elif isinstance(launcher_args, dict):
        distributed = launcher_args.get("distributed", False)

    if not distributed:
        return

    import os

    import torch

    local_rank = int(os.getenv("LOCAL_RANK", "0")) + int(os.getenv("JAX_LOCAL_RANK", "0"))
    num_visible_gpus = torch.cuda.device_count()

    # Compare local_rank against device_count (not WORLD_SIZE) so that
    # multi-node setups work correctly: WORLD_SIZE is global across all
    # nodes, but device_count is local.
    if local_rank < num_visible_gpus:
        device_str = f"cuda:{local_rank}"
    else:
        device_str = "cuda:0"

    sim_cfg = getattr(env_cfg, "sim", None)
    if sim_cfg is not None:
        sim_cfg.device = device_str

    # Set CUDA device early so physics backends that allocate on the
    # "current" device during init get the correct GPU. For the Kit path,
    # AppLauncher._resolve_device_settings will call set_device again with
    # the same value, which is harmless. For the kitless Newton path, this
    # is the only place it gets set.
    torch.cuda.set_device(device_str)

    logger.info(
        "Distributed device resolved to %s (local_rank=%d, visible_gpus=%d)",
        device_str,
        local_rank,
        num_visible_gpus,
    )


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
    # When --visualizer kit is explicitly requested alongside an ovrtx preset, fail early.
    # ovrtx and Kit ship the same RTX hydra libraries under conflicting USD namespaces;
    # loading both in the same process causes a dynamic-linker crash.  Use
    # --visualizer newton instead, which is compatible with ovrtx presets.
    early_visualizer_types = _get_visualizer_types(launcher_args)
    has_ovrtx_renderer = _scan_config(env_cfg, [_is_ovrtx_renderer])[0]
    if "kit" in early_visualizer_types and has_ovrtx_renderer:
        raise ValueError(
            "[launch_simulation] '--visualizer kit' is incompatible with 'ovrtx_renderer'. "
            "Both Kit (Isaac Sim) and ovrtx ship conflicting RTX hydra libraries "
            "(librtx.hydra.so, liblegacy.hydra.so) compiled against different USD namespaces, "
            "which causes a dynamic-linker crash when loaded into the same process. "
            "Use '--visualizer newton' instead, which is fully compatible with ovrtx presets."
        )

    validate_runtime_compatibility(env_cfg, launcher_args)
    _ensure_livestream_kit_visualizer(launcher_args)
    needs_kit, has_kit_cameras, visualizer_types = compute_kit_requirements(env_cfg, launcher_args)
    visualizer_intent = _compute_visualizer_intent(env_cfg)
    has_kit_visualizer = "kit" in visualizer_types or visualizer_intent.get("has_kit_visualizer", False)
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

    # Resolve distributed device early, before AppLauncher or physics init.
    _resolve_distributed_device(env_cfg, launcher_args)

    visualizer_explicit_none = False
    if isinstance(launcher_args, argparse.Namespace):
        visualizer_explicit_none = getattr(launcher_args, "visualizer", None) is None and getattr(
            launcher_args, "visualizer_explicit", False
        )
    elif isinstance(launcher_args, dict):
        visualizer_explicit_none = launcher_args.get("visualizer") is None and launcher_args.get(
            "visualizer_explicit", False
        )

    if needs_kit:
        # check if Isaac Sim is installed
        import importlib.util

        if importlib.util.find_spec("omni.kit") is None:
            # Print a more obvious hint when a local _isaac_sim symlink
            # exists but its env wasn't sourced (typical on Win11 + conda
            # when activate.d hooks didn't fire, e.g. under `conda run`).
            import os
            import sys

            isaaclab_path = os.environ.get("ISAACLAB_PATH")
            local_sim = os.path.join(isaaclab_path, "_isaac_sim") if isaaclab_path else None
            extra_hint = ""
            if local_sim and os.path.isdir(local_sim):
                if sys.platform == "win32":
                    extra_hint = (
                        f"  Found a local Isaac Sim at {local_sim} but its environment is not active.\n"
                        f"  Either run via `isaaclab.bat ...` (which now sources setup_conda_env.bat\n"
                        f"  automatically), or in your current shell run:\n"
                        f'    call "{local_sim}\\setup_conda_env.bat"\n'
                    )
                else:
                    extra_hint = (
                        f"  Found a local Isaac Sim at {local_sim} but its environment is not active.\n"
                        f"  Either run via `./isaaclab.sh ...` (which now sources setup_conda_env.sh\n"
                        f"  automatically), or in your current shell run:\n"
                        f'    source "{local_sim}/setup_conda_env.sh"\n'
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

        # If the simulation app is not launched, we launch it.
        from isaaclab.utils import has_kit

        if not has_kit():
            from isaaclab.app import AppLauncher

            app_launcher = AppLauncher(launcher_args)
            # AppLauncher may refine the device choice (e.g. Kit-specific
            # overrides), so propagate its final value to env_cfg.  This
            # intentionally overwrites the earlier value set by
            # _resolve_distributed_device.
            sim_cfg = getattr(env_cfg, "sim", None)
            if sim_cfg is not None and hasattr(app_launcher, "device"):
                sim_cfg.device = app_launcher.device
            close_fn = app_launcher.app.close

    settings = get_settings_manager()
    settings.set_bool("/isaaclab/runtime/needs_kit", bool(needs_kit))
    settings.set_bool("/isaaclab/runtime/has_kit_cameras", bool(has_kit_cameras))
    settings.set_bool("/isaaclab/runtime/has_kit_visualizer", bool(has_kit_visualizer))
    settings.set_bool("/isaaclab/runtime/has_ovrtx_renderer", bool(has_ovrtx_renderer))

    if not needs_kit and (visualizer_types or visualizer_explicit_none):
        # Newton path without Kit: AppLauncher is skipped, so manually store the visualizer
        # selection in SettingsManager (works in standalone mode via plain dict) so that
        # SimulationContext._get_cli_visualizer_types() can find it.
        from isaaclab.app import AppLauncher

        disable_all = visualizer_explicit_none or "none" in visualizer_types
        if isinstance(launcher_args, argparse.Namespace):
            AppLauncher.sync_visualizer_cli_settings_to_carb(
                {**vars(launcher_args), "visualizer_explicit": True, "visualizer_disable_all": disable_all}
            )
        elif isinstance(launcher_args, dict):
            AppLauncher.sync_visualizer_cli_settings_to_carb(
                {**launcher_args, "visualizer_explicit": True, "visualizer_disable_all": disable_all}
            )

    try:
        yield
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        if close_fn is not None:
            close_fn()
