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
from typing import TYPE_CHECKING, Any, NoReturn

import numpy as np
import warp as wp
from packaging import version

from pxr import Sdf, Usd, UsdGeom

from isaaclab.app.settings_manager import get_settings_manager
from isaaclab.renderers import BaseRenderer, RenderBufferKind, RenderBufferSpec
from isaaclab.renderers.camera_render_spec import CameraRenderSpec
from isaaclab.utils.version import get_isaac_sim_version
from isaaclab.utils.warp.kernels import reshape_tiled_image
from isaaclab.utils.warp.warp_math import clamp_depth_to_inf_wp, replace_inf_depth_wp

from .isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update, ensure_rtx_hydra_engine_attached

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from isaaclab_ppisp import PpispPipeline

    from isaaclab.sensors.camera.camera_data import CameraData
    from isaaclab.utils.warp import ProxyArray

from .isaac_rtx_renderer_cfg import IsaacRtxRendererCfg

_PPISP_IMPORT_ERROR_MESSAGE = (
    "isaaclab_ppisp is required when CameraCfg.isp_cfg is set. "
    "Install Isaac Lab with the 'all' extra (`pip install isaaclab[all]`) or install the "
    "isaaclab-ppisp extension from the Isaac Lab source checkout."
)


def _raise_missing_ppisp_error(exc: ModuleNotFoundError) -> NoReturn:
    # Only translate missing isaaclab_ppisp imports into the optional-dependency hint;
    # unrelated missing modules should surface unchanged for easier debugging.
    if exc.name != "isaaclab_ppisp" and not (exc.name and exc.name.startswith("isaaclab_ppisp.")):
        raise exc
    raise ModuleNotFoundError(_PPISP_IMPORT_ERROR_MESSAGE, name="isaaclab_ppisp") from exc


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
    output_data: dict[str, ProxyArray] | None = None
    spec: CameraRenderSpec | None = None
    renderer_info: dict[str, Any] = field(default_factory=dict)
    ppisp_pipeline: PpispPipeline | None = None
    """Post-render PPISP pipeline composed when ``spec.cfg.isp_cfg`` is set."""
    _hdr_scratch_wp: wp.array | None = None
    """Internal HDR scratch buffer allocated when the user did not request
    ``"rgb_hdr"`` in ``data_types`` but the PPISP pipeline still needs
    somewhere to receive the HDR AOV before LDR conversion."""


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

    def prepare_cameras(self, stage: Any, spec: CameraRenderSpec) -> None:
        """Resolve the camera's PPISP cfg and apply RTX-specific USD overrides.

        When ``spec.cfg.isp_cfg`` is set, resolves it (sentinel discovery +
        normalization) via :func:`isaaclab_ppisp.resolve_and_normalize` so
        :mod:`isaaclab` does not need to know about PPISP. Then pins
        ``exposure:*`` to neutral and applies ``OmniRtxCameraExposureAPI_1`` so
        RTX's physical-camera exposure model does not compound on top of the
        ISP. Without an ISP, the camera prim's authored exposure is left alone.
        """
        if not spec.camera_prim_paths or spec.cfg.isp_cfg is None:
            return
        try:
            from isaaclab_ppisp import apply_rtx_exposure_overrides, resolve_and_normalize
        except ModuleNotFoundError as exc:
            _raise_missing_ppisp_error(exc)

        spec.cfg.isp_cfg = resolve_and_normalize(spec.cfg.isp_cfg, stage, spec.camera_prim_paths[0])
        if spec.cfg.isp_cfg is None:
            return
        apply_rtx_exposure_overrides(stage, list(spec.camera_prim_paths))

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
            RenderBufferKind.RGBA: RenderBufferSpec(4, wp.uint8),
            RenderBufferKind.RGB: RenderBufferSpec(3, wp.uint8),
            RenderBufferKind.RGB_HDR: RenderBufferSpec(3, wp.float32),
            RenderBufferKind.DEPTH: RenderBufferSpec(1, wp.float32),
            RenderBufferKind.DISTANCE_TO_IMAGE_PLANE: RenderBufferSpec(1, wp.float32),
            RenderBufferKind.DISTANCE_TO_CAMERA: RenderBufferSpec(1, wp.float32),
            RenderBufferKind.NORMALS: RenderBufferSpec(3, wp.float32),
            RenderBufferKind.MOTION_VECTORS: RenderBufferSpec(2, wp.float32),
        }

        if sim_major >= 6:
            specs[RenderBufferKind.ALBEDO] = RenderBufferSpec(4, wp.uint8)
            for shading_type in SIMPLE_SHADING_MODES:
                specs[RenderBufferKind(shading_type)] = RenderBufferSpec(3, wp.uint8)

        seg_specs = (
            (RenderBufferKind.SEMANTIC_SEGMENTATION, self.cfg.colorize_semantic_segmentation),
            (RenderBufferKind.INSTANCE_SEGMENTATION_FAST, self.cfg.colorize_instance_segmentation),
            (RenderBufferKind.INSTANCE_ID_SEGMENTATION_FAST, self.cfg.colorize_instance_id_segmentation),
        )
        for name, colorize in seg_specs:
            specs[name] = RenderBufferSpec(4, wp.uint8) if colorize else RenderBufferSpec(1, wp.int32)

        return specs

    def prepare_stage(self, stage: Usd.Stage, num_envs: int) -> None:
        """Author per-env ``omni:scenePartition`` attributes for RTX cull-by-env rendering.

        For each ``/World/envs/env_{i}`` root, writes the inheriting primvar
        ``primvars:omni:scenePartition`` (token ``env_{i}``) on the root and the matching
        non-primvar ``omni:scenePartition`` token on every :class:`UsdGeom.Camera` descendant.
        RTX honors primvar inheritance, so the env-root primvar propagates to all descendant
        geometry and isolates each env's render tile.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.prepare_stage`."""
        root_layer = stage.GetRootLayer()
        token_type = Sdf.ValueTypeNames.Token
        with Sdf.ChangeBlock():
            for env_idx in range(num_envs):
                env_prim = stage.GetPrimAtPath(f"/World/envs/env_{env_idx}")
                if not env_prim.IsValid():
                    continue
                token = f"env_{env_idx}"
                for prim in Usd.PrimRange(env_prim):
                    if prim == env_prim:
                        attr_path = prim.GetPath().AppendProperty("primvars:omni:scenePartition")
                    elif prim.IsA(UsdGeom.Camera):
                        attr_path = prim.GetPath().AppendProperty("omni:scenePartition")
                    else:
                        continue
                    # Idempotent: a different renderer backend sharing this stage may have already
                    # authored this attribute. Re-creating an existing spec raises, so only create
                    # it when absent, then (re)assign the per-env token either way.
                    attr_spec = root_layer.GetAttributeAtPath(attr_path)
                    if attr_spec is None:
                        Sdf.JustCreatePrimAttributeInLayer(
                            root_layer, attr_path, token_type, Sdf.VariabilityUniform, True
                        )
                        attr_spec = root_layer.GetAttributeAtPath(attr_path)
                    attr_spec.default = token

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
            needs_color_render = any(
                data_type in spec.cfg.data_types for data_type in ("rgb", "rgba", str(RenderBufferKind.RGB_HDR))
            )
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

        needs_hdr_color = str(RenderBufferKind.RGB_HDR) in spec.cfg.data_types or (
            spec.cfg.isp_cfg is not None and any(data_type in ("rgb", "rgba") for data_type in spec.cfg.data_types)
        )
        if needs_hdr_color:
            rep.AnnotatorRegistry.register_annotator_from_aov(
                aov="HdrColor", output_data_type=np.float32, output_channels=4
            )

        # Define annotators based on requested data types
        annotators = {}
        for annotator_type in spec.cfg.data_types:
            if annotator_type == "rgba" or annotator_type == "rgb":
                if spec.cfg.isp_cfg is not None:
                    if str(RenderBufferKind.RGB_HDR) not in annotators:
                        annotator = rep.AnnotatorRegistry.get_annotator(
                            "HdrColor", device=spec.device, do_array_copy=False
                        )
                        annotators[str(RenderBufferKind.RGB_HDR)] = annotator
                else:
                    annotator = rep.AnnotatorRegistry.get_annotator("rgb", device=spec.device, do_array_copy=False)
                    annotators["rgba"] = annotator
            elif annotator_type == str(RenderBufferKind.RGB_HDR):
                if str(RenderBufferKind.RGB_HDR) not in annotators:
                    annotator = rep.AnnotatorRegistry.get_annotator("HdrColor", device=spec.device, do_array_copy=False)
                    annotators[str(RenderBufferKind.RGB_HDR)] = annotator
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

        ppisp_pipeline = None
        if spec.cfg.isp_cfg is not None:
            try:
                from isaaclab_ppisp import PpispPipeline
            except ModuleNotFoundError as exc:
                _raise_missing_ppisp_error(exc)

            ppisp_pipeline = PpispPipeline(spec.cfg.isp_cfg, stage=stage)

        return IsaacRtxRenderData(
            annotators=annotators,
            render_product_paths=render_product_paths,
            spec=spec,
            ppisp_pipeline=ppisp_pipeline,
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

    def set_outputs(self, render_data: IsaacRtxRenderData, output_data: dict[str, ProxyArray]):
        """Store reference to output buffers for writing during render.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.set_outputs`."""
        if render_data.ppisp_pipeline is not None and str(RenderBufferKind.RGBA) not in output_data:
            raise ValueError(
                "Isaac RTX renderer ISP requires 'rgba' (or 'rgb', which aliases into rgba) as the"
                " LDR output destination, but neither was provided. Add 'rgb' or 'rgba' to"
                " Camera.cfg.data_types when isp_cfg is set."
            )
        render_data.output_data = output_data
        # Allocate an internal HDR scratch buffer when PPISP is composed but
        # the user did not request the raw HDR AOV in ``data_types`` — the
        # PPISP kernel still needs somewhere to receive the HDR annotator
        # output before LDR conversion.
        if render_data.ppisp_pipeline is not None and str(RenderBufferKind.RGB_HDR) not in output_data:
            spec = render_data.spec
            assert spec is not None
            hdr_spec = self.supported_output_types()[RenderBufferKind.RGB_HDR]
            assert hdr_spec.dtype is wp.float32
            render_data._hdr_scratch_wp = wp.zeros(
                (spec.num_instances, spec.cfg.height, spec.cfg.width, hdr_spec.channels),
                dtype=wp.float32,
                device=spec.device,
            )

    def update_transforms(self) -> None:
        """No-op for Isaac RTX - uses USD scene directly.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.update_transforms`."""
        pass

    def update_camera(
        self,
        render_data: IsaacRtxRenderData,
        positions: ProxyArray,
        orientations: ProxyArray,
        intrinsics: ProxyArray,
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
            if data_type == str(RenderBufferKind.RGB_HDR):
                tiled_data_buffer = tiled_data_buffer[:, :, :3].contiguous()

            # The HDR annotator's destination is the user-visible ``output_data["rgb_hdr"]``
            # when they requested it explicitly; otherwise the renderer's internal
            # scratch buffer that the PPISP pipeline reads.
            if data_type == str(RenderBufferKind.RGB_HDR) and data_type not in output_data:
                assert render_data._hdr_scratch_wp is not None
                buf_wp = render_data._hdr_scratch_wp
            else:
                buf_wp = output_data[data_type].warp
            wp.launch(
                kernel=reshape_tiled_image,
                dim=(view_count, cfg.height, cfg.width),
                inputs=[
                    tiled_data_buffer.flatten(),
                    buf_wp,
                    *list(buf_wp.shape[1:]),
                    num_tiles_x,
                ],
                device=device,
            )

            # rgb is a strided warp view into rgba set up in CameraData.allocate();
            # no per-frame alias assignment needed.

            # NOTE: The `distance_to_camera` annotator returns the distance to the camera optical center.
            #       However, the replicator depth clipping is applied w.r.t. to the image plane which may result
            #       in values larger than the clipping range in the output. We apply an additional clipping to
            #       ensure values are within the clipping range for all the annotators.
            if data_type == "distance_to_camera":
                clamp_depth_to_inf_wp(buf_wp, cfg.spawn.clipping_range[1], device=device)

            # apply defined clipping behavior
            if (
                data_type in ("distance_to_camera", "distance_to_image_plane", "depth")
                and self.cfg.depth_clipping_behavior != "none"
            ):
                replacement = 0.0 if self.cfg.depth_clipping_behavior == "zero" else cfg.spawn.clipping_range[1]
                replace_inf_depth_wp(buf_wp, replacement, device=device)

        # Post-render PPISP: HDR scene-linear → LDR RGBA. The camera enforces
        # that ``rgba`` (or ``rgb`` aliasing into it) is present when an ISP is
        # configured, so writing to ``output_data["rgba"]`` is safe.
        if render_data.ppisp_pipeline is not None:
            hdr_proxy = output_data.get(str(RenderBufferKind.RGB_HDR))
            hdr_source = hdr_proxy.warp if hdr_proxy is not None else render_data._hdr_scratch_wp
            render_data.ppisp_pipeline.apply(hdr_source, output_data[str(RenderBufferKind.RGBA)].warp)

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
