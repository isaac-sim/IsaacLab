# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Newton Warp renderer — plug-and-chug style (PR #4608).

Usage:
    renderer = NewtonWarpRenderer()
    tiled_camera = TiledCamera(tiled_camera_cfg, renderer)

Requires Newton from git (e.g. 35657fc) with 4D API; install via isaaclab -i (setup.py).
"""

from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import newton
import torch
import warp as wp

from isaaclab.sim import SimulationContext
from isaaclab.utils.math import convert_camera_frame_orientation_convention

from .camera_renderer import Renderer

if TYPE_CHECKING:
    from isaaclab.sensors import SensorBase

    from ..sim.scene_data_providers import SceneDataProvider


logger = logging.getLogger(__name__)

# Scene variant keys are typically "<width>x<height>..." (e.g. 64x64rgb, 128x128warp_rgb)
_SCENE_KEY_RESOLUTION_PATTERN = re.compile(r"^(\d+)x(\d+)")


def _resolution_from_scene_variant() -> tuple[int | None, int | None]:
    """Try to get width and height from the selected env.scene variant (primary source of truth).
    Returns (width, height) if the variant key parses, else (None, None); caller should fall back to sensor config.
    """
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


class _NewtonVizCfg:
    """Minimal config so PhysxSceneDataProvider enables Newton sync."""
    visualizer_type = "newton"


def _world_count(render_context) -> int:
    """Newton uses num_worlds; upstream IsaacLab may use world_count."""
    return getattr(render_context, "world_count", None) or getattr(render_context, "num_worlds", 1)


class RenderData:
    class OutputNames:
        RGB = "rgb"
        RGBA = "rgba"
        ALBEDO = "albedo"
        DEPTH = "depth"
        DISTANCE_TO_IMAGE_PLANE = "distance_to_image_plane"
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
        self.sensor = sensor
        self.num_cameras = 1
        self._world_count = _world_count(render_context)
        self._output_ndim = 4  # 4D (n,nc,H,W); prebundle Newton uses 3D (n,nc,H*W)
        self._outputs_3d = None  # lazy: dict of 3D wp arrays when _output_ndim==3

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
            elif output_name in (RenderData.OutputNames.DEPTH, RenderData.OutputNames.DISTANCE_TO_IMAGE_PLANE):
                self.outputs.depth_image = self._from_torch(tensor_data, dtype=wp.float32)
            elif output_name == RenderData.OutputNames.NORMALS:
                self.outputs.normals_image = self._from_torch(tensor_data, dtype=wp.vec3f)
            elif output_name == RenderData.OutputNames.INSTANCE_SEGMENTATION:
                self.outputs.instance_segmentation_image = self._from_torch(tensor_data, dtype=wp.uint32)
            elif output_name == RenderData.OutputNames.RGB:
                pass
            else:
                logger.warning("NewtonWarpRenderer - output type %s is not yet supported", output_name)

    def get_output(self, output_name: str) -> wp.array:
        if output_name == RenderData.OutputNames.RGBA:
            return self.outputs.color_image
        if output_name == RenderData.OutputNames.ALBEDO:
            return self.outputs.albedo_image
        if output_name in (RenderData.OutputNames.DEPTH, RenderData.OutputNames.DISTANCE_TO_IMAGE_PLANE):
            return self.outputs.depth_image
        if output_name == RenderData.OutputNames.NORMALS:
            return self.outputs.normals_image
        if output_name == RenderData.OutputNames.INSTANCE_SEGMENTATION:
            return self.outputs.instance_segmentation_image
        return None

    def _ensure_outputs_3d(self):
        """Allocate 3D buffers (n, nc, H*W) for prebundle Newton and copy 4D -> 3D."""
        if self._outputs_3d is not None:
            return
        n, nc = self._world_count, self.num_cameras
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
        n, nc = self._world_count, self.num_cameras
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
        n, nc = self._world_count, self.num_cameras
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
        n = self._world_count
        device = getattr(self.render_context, "device", None) or wp.get_device("cuda:0")
        self.camera_transforms = wp.empty((1, n), dtype=wp.transformf, device=device)
        wp.launch(
            RenderData._update_transforms,
            n,
            [positions, converted_orientations, self.camera_transforms],
        )
        if self.render_context is not None:
            if intrinsics is not None:
                first_focal_length = intrinsics[:, 1, 1][0:1]
                fov_radians_all = 2.0 * torch.atan(self.height / (2.0 * first_focal_length))
            else:
                # Default ~60° vertical FOV when intrinsics not yet available (e.g. before _update_intrinsic_matrices)
                fov_radians_all = torch.tensor(
                    [math.radians(60.0)], device=positions.device, dtype=torch.float32
                )
            fov_wp = wp.from_torch(fov_radians_all, dtype=wp.float32)
            try:
                self.camera_rays = self.render_context.utils.compute_pinhole_camera_rays(
                    self.width, self.height, fov_wp
                )
            except TypeError:
                # Some Newton versions: compute_pinhole_camera_rays(fov_only); width/height from context
                self.camera_rays = self.render_context.utils.compute_pinhole_camera_rays(fov_wp)

    def _from_torch(self, tensor: torch.Tensor, dtype) -> wp.array:
        torch_array = wp.from_torch(tensor)
        n, nc = self._world_count, self.num_cameras
        if tensor.is_contiguous():
            return wp.array(
                ptr=torch_array.ptr,
                dtype=dtype,
                shape=(n, nc, self.height, self.width),
                device=torch_array.device,
                copy=False,
            )
        logger.warning("NewtonWarpRenderer - torch output array is non-contiguous")
        return wp.zeros(
            (n, nc, self.height, self.width),
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


class NewtonWarpRenderer(Renderer):
    """Newton Warp renderer: plug-and-chug with TiledCamera(cfg, renderer=NewtonWarpRenderer())."""

    RenderData = RenderData

    def __init__(self):
        self._scene_data_provider = self._create_scene_data_provider()
        self._newton_sensor = None  # created in _get_newton_sensor() when we have width/height

    def _get_newton_sensor(self, width: int, height: int, num_cameras: int = 1):
        """Create Newton SensorTiledCamera once we have width/height (from camera cfg). Supports both (model) and (model, num_cameras, width, height) APIs."""
        if self._newton_sensor is not None:
            return self._newton_sensor
        model = self._scene_data_provider.get_newton_model()
        if model is None:
            raise RuntimeError("NewtonWarpRenderer: get_newton_model() returned None. Ensure PhysxSceneDataProvider is set up for Newton.")
        try:
            self._newton_sensor = newton.sensors.SensorTiledCamera(model)
        except TypeError:
            self._newton_sensor = newton.sensors.SensorTiledCamera(model, num_cameras, width, height)
        return self._newton_sensor

    @property
    def newton_sensor(self):
        """Newton sensor; valid after create_render_data() has been called."""
        return self._newton_sensor

    def _create_scene_data_provider(self) -> SceneDataProvider:
        sim = SimulationContext.instance()
        if getattr(sim, "_scene_data_provider", None) is not None:
            return sim._scene_data_provider
        from ..sim.scene_data_providers import PhysxSceneDataProvider
        import isaaclab.sim as isaaclab_sim
        stage = isaaclab_sim.get_current_stage()
        provider = PhysxSceneDataProvider([_NewtonVizCfg()], stage, sim)
        sim._scene_data_provider = provider
        return provider

    def create_render_data(self, sensor: SensorBase) -> RenderData:
        # Prefer width/height from the scene variant (e.g. env.scene=64x64rgb); fall back to sensor config
        w_from_variant, h_from_variant = _resolution_from_scene_variant()
        width = w_from_variant if w_from_variant is not None else getattr(sensor.cfg, "width", 64)
        height = h_from_variant if h_from_variant is not None else getattr(sensor.cfg, "height", 64)
        num_cameras = 1  # one camera per world; Newton 4D is (num_worlds, num_cameras, H, W)
        newton_sensor = self._get_newton_sensor(width, height, num_cameras)
        return RenderData(newton_sensor.render_context, sensor)

    def set_outputs(self, render_data: RenderData, output_data: dict[str, torch.Tensor]):
        render_data.set_outputs(output_data)

    def update_transforms(self):
        self._scene_data_provider.update()

    def reset(self):
        """Sync Newton state from PhysX after env reset. Called by TiledCamera.reset()."""
        self.update_transforms()

    def update_camera(
        self,
        render_data: RenderData,
        positions: torch.Tensor,
        orientations: torch.Tensor,
        intrinsics: torch.Tensor,
    ):
        render_data.update(positions, orientations, intrinsics)

    def render(self, render_data: RenderData):
        self._get_newton_sensor(render_data.width, render_data.height)
        if render_data._output_ndim == 3:
            render_data._copy_4d_to_3d()
            self._newton_sensor.render(
                self._scene_data_provider.get_newton_state(),
                render_data.camera_transforms,
                render_data.camera_rays,
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
                self._scene_data_provider.get_newton_state(),
                render_data.camera_transforms,
                render_data.camera_rays,
                color_image=render_data.outputs.color_image,
                albedo_image=render_data.outputs.albedo_image,
                depth_image=render_data.outputs.depth_image,
                normal_image=render_data.outputs.normals_image,
                shape_index_image=render_data.outputs.instance_segmentation_image,
            )
        except RuntimeError as e:
            if "3 dimension" in str(e) or "expects an array with 3" in str(e):
                logger.info(
                    "NewtonWarpRenderer: Newton expects 3D outputs (prebundle); using 3D buffers."
                )
                render_data._output_ndim = 3
                render_data._ensure_outputs_3d()
                render_data._copy_4d_to_3d()
                self._newton_sensor.render(
                    self._scene_data_provider.get_newton_state(),
                    render_data.camera_transforms,
                    render_data.camera_rays,
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
        image_data = render_data.get_output(output_name)
        if image_data is not None and image_data.ptr != output_data.data_ptr():
            wp.copy(wp.from_torch(output_data), image_data)

    def rgba_to_rgb_channels(self) -> str:
        """Return 'rgba' (use :3 for rgb). Newton 4D API outputs RGBA."""
        return "rgba"


def save_data(camera, filename: str):
    """Save the current Newton Warp color buffer to a PNG (Daniela's approach).

    Uses the renderer's flatten_color_image_to_rgba so the saved image matches
    what Newton outputs. Call after a render; camera must be a TiledCamera
    using NewtonWarpRenderer.

    Args:
        camera: TiledCamera instance (must have ._renderer and ._render_data set).
        filename: Path for the PNG (e.g. "path/to/frame.png").
    """
    if not isinstance(camera._renderer, NewtonWarpRenderer):
        return
    render_data = getattr(camera, "_render_data", None)
    if not isinstance(render_data, NewtonWarpRenderer.RenderData):
        return
    # Prebundle Newton expects 3D (n, nc, H*W); our 4D is (n, nc, H, W). Use 3D buffer when we have it.
    color_image = (
        render_data._outputs_3d.get("color")
        if getattr(render_data, "_output_ndim", 4) == 3 and getattr(render_data, "_outputs_3d", None)
        else render_data.outputs.color_image
    )
    if color_image is None:
        return
    color_data = camera._renderer.newton_sensor.render_context.utils.flatten_color_image_to_rgba(
        color_image
    )
    from PIL import Image

    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    arr = color_data.numpy() if hasattr(color_data, "numpy") else color_data
    Image.fromarray(arr).save(filename)
