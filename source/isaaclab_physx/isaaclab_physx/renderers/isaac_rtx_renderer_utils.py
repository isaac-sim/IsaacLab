# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for Isaac RTX renderer integration."""

from __future__ import annotations

import json
from typing import Any, Protocol

import numpy as np
import torch

from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.app.settings_manager import get_settings_manager
from isaaclab.utils.array import convert_to_torch
from isaaclab.utils.version import get_isaac_sim_version

RTX_DISABLE_COLOR_RENDER_SETTING = "/rtx/sdg/force/disableColorRender"
RTX_SENSORS_SETTING = "/isaaclab/render/rtx_sensors"
SIMPLE_SHADING_AOV = "SimpleShadingSD"
SIMPLE_SHADING_MODE_SETTING = "/rtx/sdg/simpleShading/mode"
SIMPLE_SHADING_MODES = {
    "simple_shading_constant_diffuse": 0,
    "simple_shading_diffuse_mdl": 1,
    "simple_shading_full_mdl": 2,
}
_SUPPORTED_FAST_TYPES = frozenset({
    "distance_to_camera",
    "distance_to_image_plane",
    "depth",
    "albedo",
})


def apply_rtx_sensors_setup(data_types: list[str]) -> None:
    """Set RTX sensors flag and apply version-specific setup.

    Sets /isaaclab/render/rtx_sensors to True so SimulationContext enables rendering.
    Logs warnings for Isaac Sim < 6.0 when albedo or simple shading types are requested.
    """
    import logging

    logger = logging.getLogger(__name__)
    settings = get_settings_manager()
    settings.set_bool(RTX_SENSORS_SETTING, True)

    if get_isaac_sim_version().major < 6:
        if "albedo" in data_types:
            logger.warning(
                "Albedo annotator is only supported in Isaac Sim 6.0+. The albedo data type will be ignored."
            )
        if any(dt in SIMPLE_SHADING_MODES for dt in data_types):
            logger.warning(
                "Simple shading annotators are only supported in Isaac Sim 6.0+. The simple shading data types"
                " will be ignored."
            )


def apply_simple_shading_mode(data_types: list[str]) -> None:
    """Set RTX simple shading mode if requested in data types."""
    requested = [dt for dt in data_types if dt in SIMPLE_SHADING_MODES]
    if not requested:
        return
    if len(requested) > 1:
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(
            "Multiple simple shading modes requested (%s). Using '%s' only.",
            requested,
            requested[0],
        )
    settings = get_settings_manager()
    settings.set_int(SIMPLE_SHADING_MODE_SETTING, SIMPLE_SHADING_MODES[requested[0]])


def apply_rtx_disable_color_render(data_types: list[str]) -> None:
    """Set RTX disableColorRender for fast path when only depth/albedo requested.

    Isaac Sim 6.0+ only. Sets True when all data types are depth/albedo; otherwise False.
    If GUI is enabled, always False so viewport is not black.
    """
    if get_isaac_sim_version().major < 6:
        return
    settings = get_settings_manager()
    if settings.get("/isaaclab/has_gui"):
        settings.set_bool(RTX_DISABLE_COLOR_RENDER_SETTING, False)
    elif all(dt in _SUPPORTED_FAST_TYPES for dt in data_types):
        settings.set_bool(RTX_DISABLE_COLOR_RENDER_SETTING, True)
    else:
        settings.set_bool(RTX_DISABLE_COLOR_RENDER_SETTING, False)


# Module-level dedup stamp: tracks the last (sim instance, physics step) at
# which Kit's ``app.update()`` was pumped.  Keyed on ``id(sim)`` so that a
# new ``SimulationContext`` (e.g. in a new test) automatically invalidates
# any stale stamp from a previous instance.
_last_render_update_key: tuple[int, int] = (0, -1)


def ensure_isaac_rtx_render_update() -> None:
    """Ensure the Isaac RTX renderer has been pumped for the current physics step.

    This keeps the Kit-specific ``app.update()`` logic inside the renderers
    package rather than in the backend-agnostic ``SimulationContext``.

    Safe to call from multiple ``Camera`` / ``TiledCamera`` instances per step —
    only the first call triggers ``app.update()``.  Subsequent calls are no-ops
    because the module-level ``_last_render_update_key`` already matches the
    current ``(id(sim), step_count)`` pair.

    The key is a ``(sim_instance_id, step_count)`` tuple so that creating a new
    ``SimulationContext`` (e.g. in a subsequent test) automatically invalidates
    any stale stamp left over from a previous instance.

    No-op conditions:
        * Already called this step (dedup across camera instances).
        * A visualizer already pumps ``app.update()`` (e.g. KitVisualizer).
        * Rendering is not active.
    """
    global _last_render_update_key

    sim = sim_utils.SimulationContext.instance()
    if sim is None:
        return

    key = (id(sim), sim._physics_step_count)
    if _last_render_update_key == key:
        return  # Already pumped this step (by another camera or a visualizer)

    # If a visualizer already pumps the Kit app loop, mark as done and skip.
    if any(viz.pumps_app_update() for viz in sim.visualizers):
        _last_render_update_key = key
        return

    if not sim.is_rendering:
        return

    # Sync physics results → Fabric so RTX sees updated positions.
    # physics_manager.step() only runs simulate()/fetch_results() and does NOT
    # call _update_fabric(), so without this the render would lag one frame behind.
    sim.physics_manager.forward()

    import omni.kit.app

    sim.set_setting("/app/player/playSimulations", False)
    omni.kit.app.get_app().update()
    sim.set_setting("/app/player/playSimulations", True)

    _last_render_update_key = key


# --- Per-render-product Camera (IsaacRTXSpecific) utilities ---


class _PerRenderProductCfgProtocol(Protocol):
    """Protocol for camera config used by per-render-product setup."""

    data_types: list[str]
    semantic_filter: str | list[str]
    colorize_semantic_segmentation: bool
    colorize_instance_segmentation: bool
    colorize_instance_id_segmentation: bool
    semantic_segmentation_mapping: dict
    depth_clipping_behavior: str
    width: int
    height: int
    spawn: Any  # has .clipping_range


def create_per_render_product_annotators(
    cfg: _PerRenderProductCfgProtocol,
    device: str,
    view: Any,
    sensor_prims: list,
) -> tuple[list[str], dict[str, list]]:
    """Create Replicator render products and annotators for per-camera rendering.

    Returns:
        Tuple of (render_product_paths, rep_registry).
    """
    import omni.replicator.core as rep
    from omni.syntheticdata.scripts.SyntheticData import SyntheticData

    apply_simple_shading_mode(list(cfg.data_types))
    apply_rtx_disable_color_render(list(cfg.data_types))

    render_product_paths: list[str] = []
    rep_registry: dict[str, list] = {name: [] for name in cfg.data_types}

    for cam_prim in view.prims:
        cam_prim_path = cam_prim.GetPath().pathString
        if not cam_prim.IsA(UsdGeom.Camera):
            raise RuntimeError(f"Prim at path '{cam_prim_path}' is not a Camera.")
        sensor_prims.append(UsdGeom.Camera(cam_prim))

        render_prod_path = rep.create.render_product(
            cam_prim_path, resolution=(cfg.width, cfg.height)
        )
        if not isinstance(render_prod_path, str):
            render_prod_path = render_prod_path.path
        render_product_paths.append(render_prod_path)

        if isinstance(cfg.semantic_filter, list):
            semantic_filter_predicate = ":*; ".join(cfg.semantic_filter) + ":*"
        elif isinstance(cfg.semantic_filter, str):
            semantic_filter_predicate = cfg.semantic_filter
        else:
            raise ValueError(
                f"Semantic types must be a list or a string. Received: {cfg.semantic_filter}."
            )
        SyntheticData.Get().set_instance_mapping_semantic_filter(semantic_filter_predicate)

        if "cuda" in device:
            device_name = device.split(":")[0]
        else:
            device_name = "cpu"

        for name in cfg.data_types:
            if name == "semantic_segmentation":
                init_params = {
                    "colorize": cfg.colorize_semantic_segmentation,
                    "mapping": json.dumps(cfg.semantic_segmentation_mapping),
                }
            elif name == "instance_segmentation_fast":
                init_params = {"colorize": cfg.colorize_instance_segmentation}
            elif name == "instance_id_segmentation_fast":
                init_params = {"colorize": cfg.colorize_instance_id_segmentation}
            else:
                init_params = None

            if name == "albedo":
                rep.AnnotatorRegistry.register_annotator_from_aov(
                    aov="DiffuseAlbedoSD", output_data_type=np.uint8, output_channels=4
                )
            if name in SIMPLE_SHADING_MODES:
                rep.AnnotatorRegistry.register_annotator_from_aov(
                    aov=SIMPLE_SHADING_AOV, output_data_type=np.uint8, output_channels=4
                )

            simple_shading_cases = {key: SIMPLE_SHADING_AOV for key in SIMPLE_SHADING_MODES}
            special_cases = {
                "rgba": "rgb",
                "depth": "distance_to_image_plane",
                "albedo": "DiffuseAlbedoSD",
                **simple_shading_cases,
            }
            annotator_name = special_cases.get(name, name)
            rep_annotator = rep.AnnotatorRegistry.get_annotator(
                annotator_name, init_params, device=device_name
            )
            rep_annotator.attach(render_prod_path)
            rep_registry[name].append(rep_annotator)

    return render_product_paths, rep_registry


def process_annotator_output(
    name: str,
    output: Any,
    cfg: _PerRenderProductCfgProtocol,
    device: str,
) -> tuple[torch.Tensor, dict | None]:
    """Process raw annotator output into tensor and info."""
    if isinstance(output, dict):
        data = output["data"]
        info = output["info"]
    else:
        data = output
        info = None
    data = convert_to_torch(data, device=device)

    height, width = cfg.height, cfg.width
    if name == "semantic_segmentation":
        if cfg.colorize_semantic_segmentation:
            data = data.view(torch.uint8).reshape(height, width, -1)
        else:
            data = data.view(height, width, 1)
    elif name == "instance_segmentation_fast":
        if cfg.colorize_instance_segmentation:
            data = data.view(torch.uint8).reshape(height, width, -1)
        else:
            data = data.view(height, width, 1)
    elif name == "instance_id_segmentation_fast":
        if cfg.colorize_instance_id_segmentation:
            data = data.view(torch.uint8).reshape(height, width, -1)
        else:
            data = data.view(height, width, 1)
    elif name in ("distance_to_camera", "distance_to_image_plane", "depth"):
        data = data.view(height, width, 1)
    elif name in ("rgb", "normals"):
        data = data[..., :3]
    elif name == "motion_vectors":
        data = data[..., :2]
    elif name in SIMPLE_SHADING_MODES:
        data = data[..., :3]

    return data, info


def apply_depth_clipping_to_output(
    name: str,
    data_output: dict[str, torch.Tensor],
    cfg: _PerRenderProductCfgProtocol,
) -> None:
    """Apply depth clipping to output buffers (mutates data_output in place)."""
    if name not in ("distance_to_camera", "distance_to_image_plane"):
        return
    clipping_range = cfg.spawn.clipping_range
    if name == "distance_to_camera":
        data_output[name][data_output[name] > clipping_range[1]] = torch.inf
    if cfg.depth_clipping_behavior != "none":
        data_output[name][torch.isinf(data_output[name])] = (
            0.0 if cfg.depth_clipping_behavior == "zero" else clipping_range[1]
        )


def cleanup_per_render_product_annotators(
    rep_registry: dict[str, list],
    render_product_paths: list[str],
) -> None:
    """Detach annotators from render products."""
    for _, annotators in rep_registry.items():
        for annotator, render_product_path in zip(annotators, render_product_paths):
            annotator.detach([render_product_path])


# --- IsaacRTXSpecific: per-render-product Camera backend ---


def create_isaac_rtx_backend(
    cfg: _PerRenderProductCfgProtocol,
    device: str,
    view: Any,
    sensor_prims: list,
) -> "IsaacRTXSpecific":
    """Create and setup Isaac RTX backend for per-render-product Camera."""
    backend = IsaacRTXSpecific(cfg=cfg, device=device, view=view, sensor_prims=sensor_prims)
    backend.setup()
    return backend


class IsaacRTXSpecific:
    """Isaac RTX backend for per-render-product Camera.

    Handles Replicator annotator setup, render products, and annotator output
    processing for the non-tiled Camera class.
    """

    def __init__(
        self,
        cfg: _PerRenderProductCfgProtocol,
        device: str,
        view: Any,
        sensor_prims: list,
    ):
        """Initialize the backend.

        Args:
            cfg: Camera configuration (CameraCfg or compatible).
            device: Device string (e.g. "cuda:0", "cpu").
            view: XformPrimView over camera prims.
            sensor_prims: List to populate with UsdGeom.Camera prims during setup.
        """
        self._cfg = cfg
        self._device = device
        self._view = view
        self._sensor_prims = sensor_prims
        self._render_product_paths: list[str] = []
        self._rep_registry: dict[str, list] = {}

    @property
    def render_product_paths(self) -> list[str]:
        """Paths of render products for the cameras."""
        return self._render_product_paths

    @property
    def rep_registry(self) -> dict[str, list]:
        """Registry of annotators by data type name."""
        return self._rep_registry

    def setup(self) -> None:
        """Create replicator render products and annotators."""
        self._render_product_paths, self._rep_registry = create_per_render_product_annotators(
            cfg=self._cfg,
            device=self._device,
            view=self._view,
            sensor_prims=self._sensor_prims,
        )

    def update_annotator_buffers(
        self,
        env_ids: torch.Tensor,
        all_indices: torch.Tensor,
        data_output: dict,
        data_info: list,
    ) -> None:
        """Pump RTX renderer and read annotator data into camera buffers."""
        ensure_isaac_rtx_render_update()

        if len(data_output) == 0:
            self._create_annotator_data(all_indices, data_output, data_info)
        else:
            for name, annotators in self._rep_registry.items():
                for index in env_ids:
                    output = annotators[index].get_data()
                    data, info = process_annotator_output(
                        name, output, self._cfg, self._device
                    )
                    data_output[name][index] = data
                    data_info[index][name] = info
                apply_depth_clipping_to_output(name, data_output, self._cfg)

    def _create_annotator_data(
        self,
        all_indices: torch.Tensor,
        data_output: dict,
        data_info: list,
    ) -> None:
        """Create buffers from annotator data (first-time allocation)."""
        for name, annotators in self._rep_registry.items():
            data_all_cameras = []
            for index in all_indices:
                output = annotators[index].get_data()
                data, info = process_annotator_output(
                    name, output, self._cfg, self._device
                )
                data_all_cameras.append(data)
                data_info[index][name] = info
            data_output[name] = torch.stack(data_all_cameras, dim=0)
            apply_depth_clipping_to_output(name, data_output, self._cfg)

    def cleanup(self) -> None:
        """Detach annotators from render products."""
        cleanup_per_render_product_annotators(
            self._rep_registry, self._render_product_paths
        )
