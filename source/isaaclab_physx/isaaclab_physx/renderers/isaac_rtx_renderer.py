# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac RTX renderer using Omniverse Replicator for tiled camera rendering."""

from __future__ import annotations

import json
import math
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import warp as wp

from isaaclab.app.settings_manager import get_settings_manager
from isaaclab.utils.warp.kernels import reshape_tiled_image

from .isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update

if TYPE_CHECKING:
    from isaaclab.sensors import SensorBase

    from .isaac_rtx_renderer_cfg import IsaacRtxRendererCfg

# Constants from Camera (SIMPLE_SHADING_MODES, etc.) - avoid circular import
SIMPLE_SHADING_AOV = "SimpleShadingSD"
SIMPLE_SHADING_MODES = {
    "simple_shading_constant_diffuse": 0,
    "simple_shading_diffuse_mdl": 1,
    "simple_shading_full_mdl": 2,
}
SIMPLE_SHADING_MODE_SETTING = "/rtx/sdg/simpleShading/mode"


@dataclass
class IsaacRtxRenderData:
    """Render data for Isaac RTX renderer."""

    annotators: dict[str, Any]
    render_product_paths: list[str]
    output_data: dict[str, torch.Tensor] | None = None
    sensor: SensorBase | None = None


class IsaacRtxRenderer:
    """Isaac RTX backend using Omniverse Replicator for tiled camera rendering.

    Requires Isaac Sim.
    """

    def __init__(self, cfg: IsaacRtxRendererCfg):
        self.cfg = cfg

    def create_render_data(self, sensor: SensorBase) -> IsaacRtxRenderData:
        """Create render product and annotators for the tiled camera.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.create_render_data`."""
        import omni.replicator.core as rep
        from pxr import UsdGeom

        # Get camera prim paths from sensor view
        view = sensor._view
        cam_prim_paths = []
        for cam_prim in view.prims:
            cam_prim_path = cam_prim.GetPath().pathString
            if not cam_prim.IsA(UsdGeom.Camera):
                raise RuntimeError(f"Prim at path '{cam_prim_path}' is not a Camera.")
            cam_prim_paths.append(cam_prim_path)

        # Create replicator tiled render product
        rp = rep.create.render_product_tiled(
            cameras=cam_prim_paths, tile_resolution=(sensor.cfg.width, sensor.cfg.height)
        )
        render_product_paths = [rp.path]

        # Register simple shading if needed
        if any(data_type in SIMPLE_SHADING_MODES for data_type in sensor.cfg.data_types):
            rep.AnnotatorRegistry.register_annotator_from_aov(
                aov=SIMPLE_SHADING_AOV, output_data_type=np.uint8, output_channels=4
            )
            # Set simple shading mode (if requested) before rendering
            simple_shading_mode = self._resolve_simple_shading_mode(sensor)
            if simple_shading_mode is not None:
                get_settings_manager().set_int(SIMPLE_SHADING_MODE_SETTING, simple_shading_mode)

        # Define annotators based on requested data types
        annotators = {}
        for annotator_type in sensor.cfg.data_types:
            if annotator_type == "rgba" or annotator_type == "rgb":
                annotator = rep.AnnotatorRegistry.get_annotator("rgb", device=sensor.device, do_array_copy=False)
                annotators["rgba"] = annotator
            elif annotator_type == "albedo":
                # TODO: this is a temporary solution because replicator has not exposed the annotator yet
                # once it's exposed, we can remove this
                rep.AnnotatorRegistry.register_annotator_from_aov(
                    aov="DiffuseAlbedoSD", output_data_type=np.uint8, output_channels=4
                )
                annotator = rep.AnnotatorRegistry.get_annotator(
                    "DiffuseAlbedoSD", device=sensor.device, do_array_copy=False
                )
                annotators["albedo"] = annotator
            elif annotator_type in SIMPLE_SHADING_MODES:
                annotator = rep.AnnotatorRegistry.get_annotator(
                    SIMPLE_SHADING_AOV, device=sensor.device, do_array_copy=False
                )
                annotators[annotator_type] = annotator
            elif annotator_type == "depth" or annotator_type == "distance_to_image_plane":
                # keep depth for backwards compatibility
                annotator = rep.AnnotatorRegistry.get_annotator(
                    "distance_to_image_plane", device=sensor.device, do_array_copy=False
                )
                annotators[annotator_type] = annotator
            # note: we are verbose here to make it easier to understand the code.
            #   if colorize is true, the data is mapped to colors and a uint8 4 channel image is returned.
            #   if colorize is false, the data is returned as a uint32 image with ids as values.
            else:
                init_params = None
                if annotator_type == "semantic_segmentation":
                    init_params = {
                        "colorize": sensor.cfg.colorize_semantic_segmentation,
                        "mapping": json.dumps(sensor.cfg.semantic_segmentation_mapping),
                    }
                elif annotator_type == "instance_segmentation_fast":
                    init_params = {"colorize": sensor.cfg.colorize_instance_segmentation}
                elif annotator_type == "instance_id_segmentation_fast":
                    init_params = {"colorize": sensor.cfg.colorize_instance_id_segmentation}

                annotator = rep.AnnotatorRegistry.get_annotator(
                    annotator_type, init_params, device=sensor.device, do_array_copy=False
                )
                annotators[annotator_type] = annotator

        # Attach annotators to render product
        for annotator in annotators.values():
            annotator.attach(render_product_paths)

        # Currently camera owns the renderer and render data. By holding full
        # reference of the sensor, we create a circular reference between the
        # sensor and the render data. Weak reference ensures proper garbage
        # collection.
        return IsaacRtxRenderData(
            annotators=annotators,
            render_product_paths=render_product_paths,
            sensor=weakref.ref(sensor),
        )

    def _resolve_simple_shading_mode(self, sensor: SensorBase) -> int | None:
        """Resolve the requested simple shading mode from data types."""
        requested = [dt for dt in sensor.cfg.data_types if dt in SIMPLE_SHADING_MODES]
        if not requested:
            return None
        return SIMPLE_SHADING_MODES[requested[0]]

    def set_outputs(self, render_data: IsaacRtxRenderData, output_data: dict[str, torch.Tensor]):
        """Store reference to output buffers for writing during render.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.set_outputs`."""
        render_data.output_data = output_data

    def update_transforms(self) -> None:
        """No-op for Isaac RTX - uses USD scene directly.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.update_transforms`."""
        pass

    def update_camera(
        self,
        render_data: IsaacRtxRenderData,
        positions: torch.Tensor,
        orientations: torch.Tensor,
        intrinsics: torch.Tensor,
    ):
        """No-op for Replicator - uses USD camera prims directly.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.update_camera`."""
        pass

    def render(self, render_data: IsaacRtxRenderData):
        """Extract data from annotators and write to output buffers.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.render`."""
        sensor = render_data.sensor() if render_data.sensor else None
        output_data = render_data.output_data
        if output_data is None or sensor is None:
            return

        # Ensure the RTX renderer has been pumped so annotator buffers are fresh.
        # This is a no-op if another camera instance already triggered the update
        # for the current physics step, or if a visualizer already pumped it.
        ensure_isaac_rtx_render_update()

        view_count = sensor._view.count
        cfg = sensor.cfg

        def tiling_grid_shape():
            cols = math.ceil(math.sqrt(view_count))
            rows = math.ceil(view_count / cols)
            return (cols, rows)

        num_tiles_x = tiling_grid_shape()[0]

        # Extract the flattened image buffer
        for data_type, annotator in render_data.annotators.items():
            # check whether returned data is a dict (used for segmentation)
            output = annotator.get_data()
            if isinstance(output, dict):
                tiled_data_buffer = output["data"]
                if hasattr(sensor, "_data") and sensor._data is not None:
                    sensor._data.info[data_type] = output["info"]
            else:
                tiled_data_buffer = output

            # convert data buffer to warp array
            if isinstance(tiled_data_buffer, np.ndarray):
                # Let warp infer the dtype from numpy array instead of hardcoding uint8
                # Different annotators return different dtypes: RGB(uint8), depth(float32), segmentation(uint32)
                tiled_data_buffer = wp.array(tiled_data_buffer, device=sensor.device)
            else:
                tiled_data_buffer = tiled_data_buffer.to(device=sensor.device)

            # process data for different segmentation types
            # Note: Replicator returns raw buffers of dtype uint32 for segmentation types
            #   so we need to convert them to uint8 4 channel images for colorized types
            if (
                (data_type == "semantic_segmentation" and cfg.colorize_semantic_segmentation)
                or (data_type == "instance_segmentation_fast" and cfg.colorize_instance_segmentation)
                or (data_type == "instance_id_segmentation_fast" and cfg.colorize_instance_id_segmentation)
            ):
                tiled_data_buffer = wp.array(
                    ptr=tiled_data_buffer.ptr, shape=(*tiled_data_buffer.shape, 4), dtype=wp.uint8, device=sensor.device
                )

            # For motion vectors, use specialized kernel that reads 4 channels but only writes 2
            # Note: Not doing this breaks the alignment of the data (check: https://github.com/isaac-sim/IsaacLab/issues/2003)
            if data_type == "motion_vectors":
                tiled_data_buffer = tiled_data_buffer[:, :, :2].contiguous()

            # For normals, we only require the first three channels of the tiled buffer
            # Note: Not doing this breaks the alignment of the data (check: https://github.com/isaac-sim/IsaacLab/issues/4239)
            if data_type == "normals":
                tiled_data_buffer = tiled_data_buffer[:, :, :3].contiguous()
            if data_type in SIMPLE_SHADING_MODES:
                tiled_data_buffer = tiled_data_buffer[:, :, :3].contiguous()

            wp.launch(
                kernel=reshape_tiled_image,
                dim=(view_count, cfg.height, cfg.width),
                inputs=[
                    tiled_data_buffer.flatten(),
                    wp.from_torch(output_data[data_type]),
                    *list(output_data[data_type].shape[1:]),
                    num_tiles_x,
                ],
                device=sensor.device,
            )

            # alias rgb as first 3 channels of rgba
            if data_type == "rgba" and "rgb" in cfg.data_types:
                output_data["rgb"] = output_data["rgba"][..., :3]

            # NOTE: The `distance_to_camera` annotator returns the distance to the camera optical center.
            #       However, the replicator depth clipping is applied w.r.t. to the image plane which may result
            #       in values larger than the clipping range in the output. We apply an additional clipping to
            #       ensure values are within the clipping range for all the annotators.
            if data_type == "distance_to_camera":
                output_data[data_type][output_data[data_type] > cfg.spawn.clipping_range[1]] = torch.inf

            # apply defined clipping behavior
            if (
                data_type in ("distance_to_camera", "distance_to_image_plane", "depth")
                and cfg.depth_clipping_behavior != "none"
            ):
                output_data[data_type][torch.isinf(output_data[data_type])] = (
                    0.0 if cfg.depth_clipping_behavior == "zero" else cfg.spawn.clipping_range[1]
                )

    def write_output(self, render_data: IsaacRtxRenderData, output_name: str, output_data: torch.Tensor):
        """No-op for Isaac RTX - all outputs written in render().
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.write_output`."""
        pass

    def cleanup(self, render_data: IsaacRtxRenderData | None):
        """Detach annotators from render product.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.cleanup`."""
        if render_data:
            for annotator in render_data.annotators.values():
                annotator.detach(render_data.render_product_paths)
            render_data.sensor = None
