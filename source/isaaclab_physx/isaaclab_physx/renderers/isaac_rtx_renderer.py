# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac RTX renderer using Omniverse Replicator for tiled camera rendering."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import warp as wp
from packaging import version

from pxr import Sdf

from isaaclab.app.settings_manager import get_settings_manager
from isaaclab.renderers import BaseRenderer, RenderBufferKind, RenderBufferSpec
from isaaclab.renderers.camera_render_spec import CameraRenderSpec
from isaaclab.utils.version import get_isaac_sim_version
from isaaclab.utils.warp.kernels import reshape_tiled_image

from .isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update, ensure_rtx_hydra_engine_attached

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from isaaclab.sensors.camera.camera_data import CameraData

from .isaac_rtx_renderer_cfg import IsaacRtxRendererCfg

# RTX simple-shading constants.
#
# Simple shading is driven by Kit's RTX "Minimal" render mode via the
# ``/rtx/minimal/mode`` carb setting (key ``omni:rtx:minimal:mode``), with
# integer values:
#   0 = No Rendering (black output; only other AOVs are produced)
#   1 = Constant Diffuse (single constant color for all surfaces)
#   2 = Texture Diffuse  (diffuse shading using texture colors)
#   3 = Diffuse/Glossy/Emission (full material shading)
#
# The public data-type names we expose (``simple_shading_*``) are kept stable
# for backwards compatibility and map onto the Kit integer values below.
SIMPLE_SHADING_AOV = "SimpleShadingSD"
SIMPLE_SHADING_MODES = {
    "simple_shading_constant_diffuse": 1,
    "simple_shading_diffuse_mdl": 2,
    "simple_shading_full_mdl": 3,
}
SIMPLE_SHADING_MODE_SETTING = "/rtx/minimal/mode"


def _camera_semantic_filter_predicate(semantic_filter: str | list[str]) -> str:
    """Build the instance-mapping semantics predicate from :attr:`isaaclab.sensors.camera.CameraCfg.semantic_filter`.

    Replicator's semantic/instance segmentation annotators consume this via the synthetic-data pipeline.
    """
    if isinstance(semantic_filter, list):
        return ":*; ".join(semantic_filter) + ":*"
    return semantic_filter


@dataclass
class IsaacRtxRenderData:
    """Render data for Isaac RTX renderer."""

    annotators: dict[str, Any]
    render_product_paths: list[str]
    output_data: dict[str, torch.Tensor] | None = None
    spec: CameraRenderSpec | None = None
    renderer_info: dict[str, Any] = field(default_factory=dict)


class IsaacRtxRenderer(BaseRenderer):
    """Isaac RTX backend using Omniverse Replicator for tiled camera rendering.

    Requires Isaac Sim.
    """

    def __init__(self, cfg: IsaacRtxRendererCfg):
        self.cfg = cfg
        # RTX rendering requires the app to be launched with ``--enable_cameras``.
        if not get_settings_manager().get("/isaaclab/cameras_enabled"):
            raise RuntimeError(
                "A camera was spawned without the --enable_cameras flag. Please use --enable_cameras to enable"
                " rendering."
            )
        ensure_rtx_hydra_engine_attached()
        # ``/isaaclab/render/rtx_sensors`` is owned by ``Camera.__init__`` (must be set pre-``sim.reset()``).

    def supported_output_types(self) -> dict[RenderBufferKind, RenderBufferSpec]:
        """Publish the per-output Replicator layout this RTX backend writes.

        ``ALBEDO`` and the three ``SIMPLE_SHADING_*`` outputs require Isaac Sim 6.0+
        and are omitted on older versions. The three segmentation outputs report
        ``RenderBufferSpec(4, uint8)`` when the matching ``self.cfg.colorize_*`` flag is
        set, otherwise ``RenderBufferSpec(1, int32)``.
        """
        sim_major = get_isaac_sim_version().major

        specs: dict[RenderBufferKind, RenderBufferSpec] = {
            # Replicator's native layout for color output is rgba/uint8;
            # ``Camera`` aliases ``rgb`` as a view into ``rgba`` storage.
            RenderBufferKind.RGBA: RenderBufferSpec(4, torch.uint8),
            RenderBufferKind.RGB: RenderBufferSpec(3, torch.uint8),
            RenderBufferKind.DEPTH: RenderBufferSpec(1, torch.float32),
            RenderBufferKind.DISTANCE_TO_IMAGE_PLANE: RenderBufferSpec(1, torch.float32),
            RenderBufferKind.DISTANCE_TO_CAMERA: RenderBufferSpec(1, torch.float32),
            RenderBufferKind.NORMALS: RenderBufferSpec(3, torch.float32),
            RenderBufferKind.MOTION_VECTORS: RenderBufferSpec(2, torch.float32),
        }

        if sim_major >= 6:
            specs[RenderBufferKind.ALBEDO] = RenderBufferSpec(4, torch.uint8)
            for shading_type in SIMPLE_SHADING_MODES:
                specs[RenderBufferKind(shading_type)] = RenderBufferSpec(3, torch.uint8)

        seg_specs = (
            (RenderBufferKind.SEMANTIC_SEGMENTATION, self.cfg.colorize_semantic_segmentation),
            (RenderBufferKind.INSTANCE_SEGMENTATION_FAST, self.cfg.colorize_instance_segmentation),
            (RenderBufferKind.INSTANCE_ID_SEGMENTATION_FAST, self.cfg.colorize_instance_id_segmentation),
        )
        for name, colorize in seg_specs:
            specs[name] = RenderBufferSpec(4, torch.uint8) if colorize else RenderBufferSpec(1, torch.int32)

        return specs

    def prepare_stage(self, stage: Any, num_envs: int) -> None:
        """No-op for Isaac RTX - uses USD scene directly without export.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.prepare_stage`."""
        pass

    def create_render_data(self, spec: CameraRenderSpec) -> IsaacRtxRenderData:
        """Create render product and annotators for the tiled camera.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.create_render_data`."""
        import omni.replicator.core as rep
        from omni.syntheticdata import SyntheticData
        from pxr import UsdGeom

        from isaaclab.sim.utils.stage import get_current_stage

        settings = get_settings_manager()
        isaac_sim_version = get_isaac_sim_version()

        if isaac_sim_version.major >= 6:
            needs_color_render = "rgb" in spec.cfg.data_types or "rgba" in spec.cfg.data_types
            if not needs_color_render:
                settings.set_bool("/rtx/sdg/force/disableColorRender", True)
            if settings.get("/isaaclab/has_gui"):
                settings.set_bool("/rtx/sdg/force/disableColorRender", False)
        else:
            if "albedo" in spec.cfg.data_types:
                logger.warning(
                    "Albedo annotator is only supported in Isaac Sim 6.0+. The albedo data type will be ignored."
                )
            if any(dt in SIMPLE_SHADING_MODES for dt in spec.cfg.data_types):
                logger.warning(
                    "Simple shading annotators are only supported in Isaac Sim 6.0+."
                    " The simple shading data types will be ignored."
                )

        # HACK: Isaac Sim 4.5 has a bug in Camera that breaks segmentation
        # outputs for instanceable assets. Disable instancing as a workaround.
        stage = get_current_stage()
        if isaac_sim_version == version.parse("4.5") and (
            "semantic_segmentation" in spec.cfg.data_types or "instance_segmentation_fast" in spec.cfg.data_types
        ):
            logger.warning(
                "Isaac Sim 4.5 introduced a bug in Camera when outputting instance and semantic"
                " segmentation outputs for instanceable assets. As a workaround, the instanceable flag on assets"
                " will be disabled in the current workflow and may lead to longer load times and increased memory"
                " usage."
            )
            with Sdf.ChangeBlock():
                for prim in stage.Traverse():
                    prim.SetInstanceable(False)

        # Get camera prim paths from sensor view
        cam_prim_paths = list(spec.camera_prim_paths)
        for cam_prim_path in cam_prim_paths:
            cam_prim = stage.GetPrimAtPath(cam_prim_path)
            if not cam_prim.IsA(UsdGeom.Camera):
                raise RuntimeError(f"Prim at path '{cam_prim_path}' is not a Camera.")

        # Create replicator tiled render product
        rp = rep.create.render_product_tiled(cameras=cam_prim_paths, tile_resolution=(spec.cfg.width, spec.cfg.height))
        render_product_paths = [rp.path]

        # Synthetic-data instance mapping filter for segmentation; before annotator attach.
        SyntheticData.Get().set_instance_mapping_semantic_filter(
            _camera_semantic_filter_predicate(self.cfg.semantic_filter)
        )

        # Register simple shading if needed
        if any(data_type in SIMPLE_SHADING_MODES for data_type in spec.cfg.data_types):
            rep.AnnotatorRegistry.register_annotator_from_aov(
                aov=SIMPLE_SHADING_AOV, output_data_type=np.uint8, output_channels=4
            )
            # Set simple shading mode (if requested) before rendering
            simple_shading_mode = self._resolve_simple_shading_mode(spec)
            if simple_shading_mode is not None:
                get_settings_manager().set_int(SIMPLE_SHADING_MODE_SETTING, simple_shading_mode)

        # Define annotators based on requested data types
        annotators = {}
        for annotator_type in spec.cfg.data_types:
            if annotator_type == "rgba" or annotator_type == "rgb":
                annotator = rep.AnnotatorRegistry.get_annotator("rgb", device=spec.device, do_array_copy=False)
                annotators["rgba"] = annotator
            elif annotator_type == "albedo":
                # TODO: this is a temporary solution because replicator has not exposed the annotator yet
                # once it's exposed, we can remove this
                rep.AnnotatorRegistry.register_annotator_from_aov(
                    aov="DiffuseAlbedoSD", output_data_type=np.uint8, output_channels=4
                )
                annotator = rep.AnnotatorRegistry.get_annotator(
                    "DiffuseAlbedoSD", device=spec.device, do_array_copy=False
                )
                annotators["albedo"] = annotator
            elif annotator_type in SIMPLE_SHADING_MODES:
                annotator = rep.AnnotatorRegistry.get_annotator(
                    SIMPLE_SHADING_AOV, device=spec.device, do_array_copy=False
                )
                annotators[annotator_type] = annotator
            elif annotator_type == "depth" or annotator_type == "distance_to_image_plane":
                # keep depth for backwards compatibility
                annotator = rep.AnnotatorRegistry.get_annotator(
                    "distance_to_image_plane", device=spec.device, do_array_copy=False
                )
                annotators[annotator_type] = annotator
            # note: we are verbose here to make it easier to understand the code.
            #   if colorize is true, the data is mapped to colors and a uint8 4 channel image is returned.
            #   if colorize is false, the data is returned as a uint32 image with ids as values.
            else:
                init_params = None
                if annotator_type == "semantic_segmentation":
                    init_params = {
                        "colorize": self.cfg.colorize_semantic_segmentation,
                        "mapping": json.dumps(self.cfg.semantic_segmentation_mapping),
                    }
                elif annotator_type == "instance_segmentation_fast":
                    init_params = {"colorize": self.cfg.colorize_instance_segmentation}
                elif annotator_type == "instance_id_segmentation_fast":
                    init_params = {"colorize": self.cfg.colorize_instance_id_segmentation}

                annotator = rep.AnnotatorRegistry.get_annotator(
                    annotator_type, init_params, device=spec.device, do_array_copy=False
                )
                annotators[annotator_type] = annotator

        # Attach annotators to render product
        for annotator in annotators.values():
            annotator.attach(render_product_paths)

        return IsaacRtxRenderData(
            annotators=annotators,
            render_product_paths=render_product_paths,
            spec=spec,
        )

    def _resolve_simple_shading_mode(self, spec: CameraRenderSpec) -> int | None:
        """Resolve the requested simple shading mode from data types."""
        requested = [dt for dt in spec.cfg.data_types if dt in SIMPLE_SHADING_MODES]
        if not requested:
            return None
        if len(requested) > 1:
            logger.warning(
                "Multiple simple shading modes requested (%s). Using '%s' only.",
                requested,
                requested[0],
            )
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
        spec = render_data.spec
        output_data = render_data.output_data
        if output_data is None or spec is None:
            return

        # Ensure the RTX renderer has been pumped so annotator buffers are fresh.
        # This is a no-op if another camera instance already triggered the update
        # for the current physics step, or if a visualizer already pumped it.
        ensure_isaac_rtx_render_update()

        view_count = spec.view_count
        cfg = spec.cfg
        device = spec.device

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
                render_data.renderer_info[data_type] = output["info"]
            else:
                tiled_data_buffer = output

            # convert data buffer to warp array
            if isinstance(tiled_data_buffer, np.ndarray):
                # Let warp infer the dtype from numpy array instead of hardcoding uint8
                # Different annotators return different dtypes: RGB(uint8), depth(float32), segmentation(uint32)
                tiled_data_buffer = wp.array(tiled_data_buffer, device=device)
            else:
                tiled_data_buffer = tiled_data_buffer.to(device=device)

            # process data for different segmentation types
            # Note: Replicator returns raw buffers of dtype uint32 for segmentation types
            #   so we need to convert them to uint8 4 channel images for colorized types
            if (
                (data_type == "semantic_segmentation" and self.cfg.colorize_semantic_segmentation)
                or (data_type == "instance_segmentation_fast" and self.cfg.colorize_instance_segmentation)
                or (data_type == "instance_id_segmentation_fast" and self.cfg.colorize_instance_id_segmentation)
            ):
                tiled_data_buffer = wp.array(
                    ptr=tiled_data_buffer.ptr, shape=(*tiled_data_buffer.shape, 4), dtype=wp.uint8, device=device
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
                device=device,
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
                and self.cfg.depth_clipping_behavior != "none"
            ):
                output_data[data_type][torch.isinf(output_data[data_type])] = (
                    0.0 if self.cfg.depth_clipping_behavior == "zero" else cfg.spawn.clipping_range[1]
                )

    def read_output(self, render_data: IsaacRtxRenderData, camera_data: CameraData) -> None:
        """Populate per-output metadata collected during render(). Pixel data already written in render().
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.read_output`."""
        for output_name, info in render_data.renderer_info.items():
            if info is not None:
                camera_data.info[output_name] = info

    def cleanup(self, render_data: IsaacRtxRenderData | None):
        """Detach annotators from render product.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.cleanup`."""
        if render_data:
            for annotator in render_data.annotators.values():
                annotator.detach(render_data.render_product_paths)
            render_data.spec = None
