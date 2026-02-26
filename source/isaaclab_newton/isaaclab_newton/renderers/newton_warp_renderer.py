# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton Warp renderer for tiled camera rendering."""

from __future__ import annotations

import logging
import math
import re
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING

import newton
import torch
import warp as wp

from isaaclab.sim import SimulationContext
from isaaclab.utils.math import convert_camera_frame_orientation_convention
from isaaclab.visualizers import VisualizerCfg

from .newton_warp_renderer_cfg import NewtonWarpRendererCfg

if TYPE_CHECKING:
    from isaaclab.sensors import SensorBase
    from isaaclab.sim.scene_data_providers import SceneDataProvider

logger = logging.getLogger(__name__)

# Scene variant keys are typically "<width>x<height>..." (e.g. 64x64rgb)
_SCENE_KEY_RESOLUTION_PATTERN = re.compile(r"^(\d+)x(\d+)")


def _resolution_from_scene_variant() -> tuple[int | None, int | None]:
    """Try to get width and height from the selected env.scene variant."""
    try:
        from hydra.core.hydra_config import HydraConfig

        cfg = HydraConfig.get()
        if cfg is None:
            return (None, None)
        choices = getattr(getattr(cfg, "runtime", None), "choices", None)
        if not isinstance(choices, dict):
            return (None, None)
        scene_key = choices.get("env.scene") or choices.get("env/scene")
        if not isinstance(scene_key, str):
            return (None, None)
        m = _SCENE_KEY_RESOLUTION_PATTERN.match(scene_key.strip())
        if not m:
            return (None, None)
        return (int(m.group(1)), int(m.group(2)))
    except Exception:  # noqa: BLE001
        return (None, None)


def _world_count(render_context) -> int | None:
    """Newton may use num_worlds; upstream IsaacLab may use world_count."""
    return getattr(render_context, "world_count", None) or getattr(render_context, "num_worlds", None)


class RenderData:
    class OutputNames:
        RGB = "rgb"
        RGBA = "rgba"
        ALBEDO = "albedo"
        DEPTH = "depth"
        NORMALS = "normals"
        INSTANCE_SEGMENTATION = "instance_segmentation_fast"

    @dataclass
    class CameraOutputs:
        color_image: wp.array(dtype=wp.uint32, ndim=4) = None
        albedo_image: wp.array(dtype=wp.uint32, ndim=4) = None
        depth_image: wp.array(dtype=wp.float32, ndim=4) = None
        normals_image: wp.array(dtype=wp.vec3f, ndim=4) = None
        instance_segmentation_image: wp.array(dtype=wp.uint32, ndim=4) = None

    def __init__(self, render_context: newton.sensors.SensorTiledCamera.RenderContext, sensor: SensorBase):
        self.render_context = render_context
        self.sensor = weakref.ref(sensor)
        self.num_cameras = 1
        self._world_count: int | None = _world_count(render_context)
        self._output_ndim = 4  # 4D (n,nc,H,W); prebundle Newton uses 3D (n,nc,H*W)
        self._outputs_3d: dict | None = None  # lazy when _output_ndim==3

        self.camera_rays: wp.array(dtype=wp.vec3f, ndim=4) = None
        self.camera_transforms: wp.array(dtype=wp.transformf, ndim=2) = None
        self.outputs = RenderData.CameraOutputs()
        self.width = getattr(sensor.cfg, "width", 100)
        self.height = getattr(sensor.cfg, "height", 100)

    def set_outputs(self, output_data: dict[str, torch.Tensor]):
        for output_name, tensor_data in output_data.items():
            if output_name == RenderData.OutputNames.RGBA:
                self.outputs.color_image = self._from_torch(tensor_data, dtype=wp.uint32)
            elif output_name == RenderData.OutputNames.ALBEDO:
                self.outputs.albedo_image = self._from_torch(tensor_data, dtype=wp.uint32)
            elif output_name == RenderData.OutputNames.DEPTH:
                self.outputs.depth_image = self._from_torch(tensor_data, dtype=wp.float32)
            elif output_name == RenderData.OutputNames.NORMALS:
                self.outputs.normals_image = self._from_torch(tensor_data, dtype=wp.vec3f)
            elif output_name == RenderData.OutputNames.INSTANCE_SEGMENTATION:
                self.outputs.instance_segmentation_image = self._from_torch(tensor_data, dtype=wp.uint32)
            elif output_name == RenderData.OutputNames.RGB:
                pass
            else:
                logger.warning(f"NewtonWarpRenderer - output type {output_name} is not yet supported")

    def get_output(self, output_name: str) -> wp.array:
        if output_name == RenderData.OutputNames.RGBA:
            return self.outputs.color_image
        elif output_name == RenderData.OutputNames.ALBEDO:
            return self.outputs.albedo_image
        elif output_name == RenderData.OutputNames.DEPTH:
            return self.outputs.depth_image
        elif output_name == RenderData.OutputNames.NORMALS:
            return self.outputs.normals_image
        elif output_name == RenderData.OutputNames.INSTANCE_SEGMENTATION:
            return self.outputs.instance_segmentation_image
        return None

    def _ensure_outputs_3d(self):
        """Allocate 3D buffers (n, nc, H*W) for prebundle Newton and copy 4D -> 3D."""
        if self._outputs_3d is not None:
            return
        n = self._world_count or 1
        nc = self.num_cameras
        h, w = self.height, self.width
        device = getattr(self.render_context, "device", None) or wp.get_device("cuda:0")
        self._outputs_3d = {}
        if self.outputs.color_image is not None:
            self._outputs_3d["color"] = wp.empty((n, nc, h * w), dtype=wp.uint32, device=device)
        if self.outputs.albedo_image is not None:
            self._outputs_3d["albedo"] = wp.empty((n, nc, h * w), dtype=wp.uint32, device=device)
        if self.outputs.depth_image is not None:
            self._outputs_3d["depth"] = wp.empty((n, nc, h * w), dtype=wp.float32, device=device)
        if self.outputs.normals_image is not None:
            self._outputs_3d["normal"] = wp.empty((n, nc, h * w), dtype=wp.vec3f, device=device)
        if self.outputs.instance_segmentation_image is not None:
            self._outputs_3d["shape_index"] = wp.empty((n, nc, h * w), dtype=wp.uint32, device=device)

    def _copy_4d_to_3d(self):
        """Copy 4D outputs to 3D buffers for prebundle render."""
        n = self._world_count or 1
        nc = self.num_cameras
        h, w = self.height, self.width
        dim = n * nc * h * w
        inp = [n, nc, w, h]
        if self.outputs.color_image is not None:
            wp.launch(
                RenderData._copy_4d_to_3d_uint32,
                dim=dim,
                inputs=[self.outputs.color_image, self._outputs_3d["color"]] + inp,
                device=self.outputs.color_image.device,
            )
        if self.outputs.albedo_image is not None:
            wp.launch(
                RenderData._copy_4d_to_3d_uint32,
                dim=dim,
                inputs=[self.outputs.albedo_image, self._outputs_3d["albedo"]] + inp,
                device=self.outputs.albedo_image.device,
            )
        if self.outputs.depth_image is not None:
            wp.launch(
                RenderData._copy_4d_to_3d_float,
                dim=dim,
                inputs=[self.outputs.depth_image, self._outputs_3d["depth"]] + inp,
                device=self.outputs.depth_image.device,
            )
        if self.outputs.normals_image is not None:
            wp.launch(
                RenderData._copy_4d_to_3d_vec3,
                dim=dim,
                inputs=[self.outputs.normals_image, self._outputs_3d["normal"]] + inp,
                device=self.outputs.normals_image.device,
            )
        if self.outputs.instance_segmentation_image is not None:
            wp.launch(
                RenderData._copy_4d_to_3d_uint32,
                dim=dim,
                inputs=[
                    self.outputs.instance_segmentation_image,
                    self._outputs_3d["shape_index"],
                ]
                + inp,
                device=self.outputs.instance_segmentation_image.device,
            )

    def _copy_3d_to_4d(self):
        """Copy 3D buffers back to 4D outputs after prebundle render."""
        n = self._world_count or 1
        nc = self.num_cameras
        h, w = self.height, self.width
        dim = n * nc * h * w
        inp = [n, nc, w, h]
        if self.outputs.color_image is not None:
            wp.launch(
                RenderData._copy_3d_to_4d_uint32,
                dim=dim,
                inputs=[self._outputs_3d["color"], self.outputs.color_image] + inp,
                device=self.outputs.color_image.device,
            )
        if self.outputs.albedo_image is not None:
            wp.launch(
                RenderData._copy_3d_to_4d_uint32,
                dim=dim,
                inputs=[self._outputs_3d["albedo"], self.outputs.albedo_image] + inp,
                device=self.outputs.albedo_image.device,
            )
        if self.outputs.depth_image is not None:
            wp.launch(
                RenderData._copy_3d_to_4d_float,
                dim=dim,
                inputs=[self._outputs_3d["depth"], self.outputs.depth_image] + inp,
                device=self.outputs.depth_image.device,
            )
        if self.outputs.normals_image is not None:
            wp.launch(
                RenderData._copy_3d_to_4d_vec3,
                dim=dim,
                inputs=[self._outputs_3d["normal"], self.outputs.normals_image] + inp,
                device=self.outputs.normals_image.device,
            )
        if self.outputs.instance_segmentation_image is not None:
            wp.launch(
                RenderData._copy_3d_to_4d_uint32,
                dim=dim,
                inputs=[
                    self._outputs_3d["shape_index"],
                    self.outputs.instance_segmentation_image,
                ]
                + inp,
                device=self.outputs.instance_segmentation_image.device,
            )

    def update(self, positions: torch.Tensor, orientations: torch.Tensor, intrinsics: torch.Tensor):
        converted_orientations = convert_camera_frame_orientation_convention(
            orientations, origin="world", target="opengl"
        )
        self._world_count = positions.shape[0]
        device = getattr(self.render_context, "device", None) or wp.get_device("cuda:0")
        self.camera_transforms = wp.empty((1, self._world_count), dtype=wp.transformf, device=device)
        wp.launch(
            RenderData._update_transforms,
            self._world_count,
            [positions, converted_orientations, self.camera_transforms],
        )

        if self.render_context is not None and self._world_count is not None:
            utils = self.render_context.utils
            if intrinsics is not None:
                first_focal_length = intrinsics[:, 1, 1][0:1]
                fov_radians_all = 2.0 * torch.atan(self.height / (2.0 * first_focal_length))
            else:
                fov_radians_all = torch.tensor(
                    [math.radians(60.0)], device=positions.device, dtype=torch.float32
                )
            fov_wp = wp.from_torch(fov_radians_all, dtype=wp.float32)
            try:
                self.camera_rays = utils.compute_pinhole_camera_rays(
                    self.width, self.height, fov_wp
                )
            except TypeError:
                self.camera_rays = utils.compute_pinhole_camera_rays(fov_wp)

    def _from_torch(self, tensor: torch.Tensor, dtype) -> wp.array:
        n = self._world_count or getattr(self.render_context, "world_count", tensor.shape[0])
        torch_array = wp.from_torch(tensor)
        if tensor.is_contiguous():
            return wp.array(
                ptr=torch_array.ptr,
                dtype=dtype,
                shape=(n, self.num_cameras, self.height, self.width),
                device=torch_array.device,
                copy=False,
            )

        logger.warning("NewtonWarpRenderer - torch output array is non-contiguous")
        return wp.zeros(
            (n, self.num_cameras, self.height, self.width),
            dtype=dtype,
            device=torch_array.device,
        )

    @wp.kernel
    def _update_transforms(
        positions: wp.array(dtype=wp.vec3f),
        orientations: wp.array(dtype=wp.quatf),
        output: wp.array(dtype=wp.transformf, ndim=2),
    ):
        tid = wp.tid()
        output[0, tid] = wp.transformf(positions[tid], orientations[tid])

    @staticmethod
    @wp.kernel
    def _copy_4d_to_3d_uint32(
        src: wp.array(dtype=wp.uint32, ndim=4),
        dst: wp.array(dtype=wp.uint32, ndim=3),
        n: wp.int32,
        nc: wp.int32,
        width: wp.int32,
        height: wp.int32,
    ):
        tid = wp.tid()
        pixels_per_view = width * height
        idx = tid % pixels_per_view
        j = (tid // pixels_per_view) % nc
        i = tid // (pixels_per_view * nc)
        py, px = idx // width, idx % width
        dst[i, j, idx] = src[i, j, py, px]

    @staticmethod
    @wp.kernel
    def _copy_3d_to_4d_uint32(
        src: wp.array(dtype=wp.uint32, ndim=3),
        dst: wp.array(dtype=wp.uint32, ndim=4),
        n: wp.int32,
        nc: wp.int32,
        width: wp.int32,
        height: wp.int32,
    ):
        tid = wp.tid()
        pixels_per_view = width * height
        idx = tid % pixels_per_view
        j = (tid // pixels_per_view) % nc
        i = tid // (pixels_per_view * nc)
        py, px = idx // width, idx % width
        dst[i, j, py, px] = src[i, j, idx]

    @staticmethod
    @wp.kernel
    def _copy_4d_to_3d_float(
        src: wp.array(dtype=wp.float32, ndim=4),
        dst: wp.array(dtype=wp.float32, ndim=3),
        n: wp.int32,
        nc: wp.int32,
        width: wp.int32,
        height: wp.int32,
    ):
        tid = wp.tid()
        pixels_per_view = width * height
        idx = tid % pixels_per_view
        j = (tid // pixels_per_view) % nc
        i = tid // (pixels_per_view * nc)
        py, px = idx // width, idx % width
        dst[i, j, idx] = src[i, j, py, px]

    @staticmethod
    @wp.kernel
    def _copy_3d_to_4d_float(
        src: wp.array(dtype=wp.float32, ndim=3),
        dst: wp.array(dtype=wp.float32, ndim=4),
        n: wp.int32,
        nc: wp.int32,
        width: wp.int32,
        height: wp.int32,
    ):
        tid = wp.tid()
        pixels_per_view = width * height
        idx = tid % pixels_per_view
        j = (tid // pixels_per_view) % nc
        i = tid // (pixels_per_view * nc)
        py, px = idx // width, idx % width
        dst[i, j, py, px] = src[i, j, idx]

    @staticmethod
    @wp.kernel
    def _copy_4d_to_3d_vec3(
        src: wp.array(dtype=wp.vec3f, ndim=4),
        dst: wp.array(dtype=wp.vec3f, ndim=3),
        n: wp.int32,
        nc: wp.int32,
        width: wp.int32,
        height: wp.int32,
    ):
        tid = wp.tid()
        pixels_per_view = width * height
        idx = tid % pixels_per_view
        j = (tid // pixels_per_view) % nc
        i = tid // (pixels_per_view * nc)
        py, px = idx // width, idx % width
        dst[i, j, idx] = src[i, j, py, px]

    @staticmethod
    @wp.kernel
    def _copy_3d_to_4d_vec3(
        src: wp.array(dtype=wp.vec3f, ndim=3),
        dst: wp.array(dtype=wp.vec3f, ndim=4),
        n: wp.int32,
        nc: wp.int32,
        width: wp.int32,
        height: wp.int32,
    ):
        tid = wp.tid()
        pixels_per_view = width * height
        idx = tid % pixels_per_view
        j = (tid // pixels_per_view) % nc
        i = tid // (pixels_per_view * nc)
        py, px = idx // width, idx % width
        dst[i, j, py, px] = src[i, j, idx]


class NewtonWarpRenderer:
    """Newton Warp backend for tiled camera rendering"""

    RenderData = RenderData

    def __init__(self, cfg: NewtonWarpRendererCfg):
        self.cfg = cfg
        self._newton_sensor = None  # created lazily in _get_newton_sensor()

    def _get_newton_sensor(self, width: int, height: int, num_cameras: int = 1):
        """Create Newton SensorTiledCamera once we have width/height. Supports (model) and (model, num_cameras, width, height) APIs."""
        if self._newton_sensor is not None:
            return self._newton_sensor
        model = self.get_scene_data_provider().get_newton_model()
        if model is None:
            raise RuntimeError(
                "NewtonWarpRenderer: get_newton_model() returned None. Ensure scene data provider is set up for Newton."
            )
        try:
            self._newton_sensor = newton.sensors.SensorTiledCamera(model)
        except TypeError:
            self._newton_sensor = newton.sensors.SensorTiledCamera(
                model, num_cameras, width, height
            )
        return self._newton_sensor

    @property
    def newton_sensor(self):
        """Newton sensor; valid after create_render_data() has been called."""
        return self._newton_sensor

    def create_render_data(self, sensor: SensorBase) -> RenderData:
        """Create render data for the Newton tiled camera."""
        w_from_variant, h_from_variant = _resolution_from_scene_variant()
        width = w_from_variant if w_from_variant is not None else getattr(sensor.cfg, "width", 64)
        height = h_from_variant if h_from_variant is not None else getattr(sensor.cfg, "height", 64)
        num_cameras = getattr(self.cfg, "num_cameras", 1) if self.cfg else 1
        newton_sensor = self._get_newton_sensor(width, height, num_cameras)
        return RenderData(newton_sensor.render_context, sensor)

    def set_outputs(self, render_data: RenderData, output_data: dict[str, torch.Tensor]):
        """Store output buffers. See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.set_outputs`."""
        render_data.set_outputs(output_data)

    def reset(self):
        """Sync Newton state from PhysX after env reset. Called by TiledCamera.reset()."""
        self.update_transforms()

    def update_transforms(self):
        """Sync Newton scene state before rendering.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.update_transforms`."""
        SimulationContext.instance().update_scene_data_provider(True)

    def update_camera(
        self, render_data: RenderData, positions: torch.Tensor, orientations: torch.Tensor, intrinsics: torch.Tensor
    ):
        """Update camera poses and intrinsics.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.update_camera`."""
        render_data.update(positions, orientations, intrinsics)

    def render(self, render_data: RenderData):
        """Render and write to output buffers. Try 4D first; on Newton conflict fall back to 3D (n, nc, H*W)."""
        self._get_newton_sensor(render_data.width, render_data.height)
        provider = self.get_scene_data_provider()
        state = provider.get_newton_state()
        transforms = render_data.camera_transforms
        rays = render_data.camera_rays

        if render_data._output_ndim == 3:
            render_data._copy_4d_to_3d()
            self._newton_sensor.render(
                state,
                transforms,
                rays,
                color_image=render_data._outputs_3d.get("color"),
                albedo_image=render_data._outputs_3d.get("albedo"),
                depth_image=render_data._outputs_3d.get("depth"),
                normal_image=render_data._outputs_3d.get("normal"),
                shape_index_image=render_data._outputs_3d.get("shape_index"),
            )
            render_data._copy_3d_to_4d()
            return

        try:
            self._newton_sensor.render(
                state,
                transforms,
                rays,
                color_image=render_data.outputs.color_image,
                albedo_image=render_data.outputs.albedo_image,
                depth_image=render_data.outputs.depth_image,
                normal_image=render_data.outputs.normals_image,
                shape_index_image=render_data.outputs.instance_segmentation_image,
            )
        except RuntimeError as e:
            if "3 dimension" in str(e) or "expects an array with 3" in str(e):
                logger.info(
                    "NewtonWarpRenderer: Newton expects 3D outputs (prebundle); using 3D buffers (n, nc, H*W)."
                )
                render_data._output_ndim = 3
                render_data._ensure_outputs_3d()
                render_data._copy_4d_to_3d()
                self._newton_sensor.render(
                    state,
                    transforms,
                    rays,
                    color_image=render_data._outputs_3d.get("color"),
                    albedo_image=render_data._outputs_3d.get("albedo"),
                    depth_image=render_data._outputs_3d.get("depth"),
                    normal_image=render_data._outputs_3d.get("normal"),
                    shape_index_image=render_data._outputs_3d.get("shape_index"),
                )
                render_data._copy_3d_to_4d()
            else:
                raise

    def write_output(self, render_data: RenderData, output_name: str, output_data: torch.Tensor):
        """Copy a specific output to the given buffer.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.write_output`."""
        image_data = render_data.get_output(output_name)
        if image_data is not None:
            if image_data.ptr != output_data.data_ptr():
                wp.copy(wp.from_torch(output_data), image_data)

    def cleanup(self, render_data: RenderData | None):
        """Release resources. No-op for Newton Warp.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.cleanup`."""
        if render_data:
            render_data.sensor = None

    def get_scene_data_provider(self) -> SceneDataProvider:
        return SimulationContext.instance().initialize_scene_data_provider([VisualizerCfg(visualizer_type="newton")])
