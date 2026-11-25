# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVRTX Renderer implementation.

How it fits together
--------------------
- **ovrtx_renderer.py** (this file): Orchestrates USD loading/cloning, camera and object
  bindings, and output buffers, borrowing native handles from OVRTXBackend. Each frame it:
  updates camera/object transforms (using kernels), steps the renderer, then extracts
  tiles from the tiled framebuffer (kernels). Under asynchronous rendering, each camera keeps
  its own pending ``step_async`` render operation and publishes it at the next ``read_output``.
  The ovrtx binding write operations retain their input buffers until ovrtx has consumed them.

- **ovrtx_renderer_kernels.py**: Warp GPU kernels for OVRTX rendering pipeline.

- **ovrtx_usd.py**: USD helpers for OVRTX: render var config, camera injection, etc.
"""

from __future__ import annotations

import contextlib
import ctypes
import dataclasses
import itertools
import logging
import math
import os
import sys
import tempfile
import warnings
import weakref
from builtins import ExceptionGroup
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

logger = logging.getLogger(__name__)

import numpy as np
import ovstage
import torch
import warp as wp

import isaaclab.utils.warp  # noqa: F401  # initializes Warp runtime

# The ovrtx C library links to its own version of the USD libraries. Having
# the pxr Python package available can cause the C library to load an
# incompatible version of libusd, potentially leading to undefined behavior.
# By setting OVRTX_SKIP_USD_CHECK, we prevent the C library from loading the pxr Python package.
os.environ["OVRTX_SKIP_USD_CHECK"] = "1"


from ovrtx import (
    AttributeFilterMode,
    BindingFlag,
    DataAccess,
    Device,
    FilterKind,
    PrimMode,
    Renderer,
    RendererConfig,
    RenderProductSetOutputs,
    Semantic,
    TextureStreamingMode,
)
from PIL import Image

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

from isaaclab.markers.visualization_markers_cfg import VisualizationMarkersCfg
from isaaclab.renderers import BaseRenderer, RenderBufferKind, RenderBufferSpec
from isaaclab.scene_data import SceneDataFormat
from isaaclab.sensors.camera.camera_data import CameraData
from isaaclab.sim import PreviewSurfaceCfg, SimulationContext, use_stage
from isaaclab.utils.warp import ProxyArray

from isaaclab_ov.cloner import ovrtx_replicate
from isaaclab_ov.renderers.ovrtx_annotator_utils import (
    IdentifierMap,
    RgbaColor,
    build_instance_id_to_labels_and_semantics,
    build_semantic_id_to_labels,
    decode_semantic_id_map,
    decode_stable_id_map,
    decode_stable_id_semantic_id_map,
)
from isaaclab_ov.renderers.ovrtx_lidar import copy_tiled_point_cloud
from isaaclab_ov.renderers.ovrtx_renderer_cfg import OVRTXBackendCfg, OVRTXRendererCfg
from isaaclab_ov.renderers.ovrtx_renderer_kernels import (
    accumulate_mesh_normals,
    create_camera_transforms_kernel,
    create_lidar_transforms_kernel,
    create_marker_transforms_kernel,
    extract_all_tiles_kernel,
    generate_random_colors_from_ids_kernel,
    hide_environment_bodies_kernel,
    normalize_mesh_normals,
    remap_semantic_colors_kernel,
)
from isaaclab_ov.renderers.ovrtx_shader_cache import redirect_shader_cache
from isaaclab_ov.renderers.ovrtx_usd import (
    _RTX_MINIMAL_MODES,
    build_lidar_render_product_as_string,
    build_render_product_as_string,
    render_var_prim_paths_by_source,
)
from isaaclab_ov.renderers.visual_materials import OVRTXVisualMaterialWriter
from isaaclab_ov.stage import (
    export_stage_to_string,
    points_tensor_from_warp,
    xform_tensor_from_numpy,
    xform_tensor_from_warp,
)

if TYPE_CHECKING:
    from ovrtx import AttributeBinding, FrameOutput, Operation, PendingFetch, RenderVarOutput

    from isaaclab.renderers.base_renderer import VisualMaterialBatch
    from isaaclab.sensors import LidarData

    from isaaclab_ov.renderers.ovrtx_lidar_cfg import OVRTXLidarCfg
    from isaaclab_ov.stage import OvstageBackend

from isaaclab.renderers.camera_render_spec import CameraRenderSpec

_RENDER_DELTA_TIME = 1.0 / 60.0
_FAILED_FRAME_TEXT = "failed to produce a frame"
_FAILED_FRAME_NOTE = (
    "A tiled image above a renderer buffer limit is one cause of a failed frame."
    " Set OVRTXRendererCfg.max_envs_per_render_product to an integer to split the camera sensors of a renderer"
    " over several render products with smaller tiled images. Use the same value on every OVRTX renderer"
    ' configuration of the task. The value "auto" searches for this split after a failed first frame, but only'
    " with one camera path per environment."
)
_MAX_SMALLER_LAYOUTS = 4
_PER_PRODUCT_ATTRIBUTES = frozenset({"camera", "orderedVars", "resolution", "usd-path"})
_GPU_CAMERA_TRANSFORMS_ATTRIBUTE = "omni:rtx:scene:hydra:readGPUCameraTransformsFromFabricInRenderDelegate"
_LDR_COLOR_VAR = "LdrColor"
# TODO: Remove with the below-the-scene hiding workaround in OVRTXRenderer.set_environment_visibility.
_HIDDEN_DROP = wp.float64(1.0e6)
"""Distance [m] below the scene to which the bodies of hidden environments are moved."""
_CAMERA_INTRINSIC_ATTRIBUTES = (
    "focalLength",
    "horizontalAperture",
    "verticalAperture",
    "horizontalApertureOffset",
    "verticalApertureOffset",
)
_HDR_COLOR_VAR = "HdrColor"
_ALBEDO_VAR = "DiffuseAlbedoSD"
_NORMALS_VAR = "NormalSD"
_MOTION_VECTORS_VAR = "TargetMotionSD"
_SEMANTIC_SEGMENTATION_VAR = "SemanticSegmentation"
_INSTANCE_SEGMENTATION_VAR = "NonStableInstanceSegmentation"
_SEMANTIC_ID_MAP_VAR = "SemanticIdMap"
_STABLE_ID_MAP_VAR = "StableIdMap"
_STABLE_ID_SEMANTIC_ID_MAP_VAR = "StableIdSemanticIdMap"

# Map render vars needed to decode the instance-segmentation info dicts.
_INSTANCE_SEGMENTATION_MAP_VARS = (_STABLE_ID_SEMANTIC_ID_MAP_VAR, _STABLE_ID_MAP_VAR, _SEMANTIC_ID_MAP_VAR)

# Maps depth render vars to compatible output buffers.
_DEPTH_VAR_BUFFER_KEYS: dict[str, tuple[str, ...]] = {
    "DistanceToImagePlaneSD": ("depth", "distance_to_image_plane"),
    "DistanceToCameraSD": ("distance_to_camera",),
}


# Opts Linux out of the host wait, onto the same GPU-side ordering every other platform uses.
# See :meth:`OVRTXRenderer._map_render_var_to_dlpack`.
_DISABLE_LINUX_CUDA_CPU_SYNC_ENV = "ISAAC_LAB_OVRTX_DISABLE_LINUX_CUDA_CPU_SYNC"


def _tile_columns(num_envs: int) -> int:
    """Return the number of tile columns in the atlas of ``num_envs`` environments."""
    return math.ceil(math.sqrt(num_envs))


def _gpu_side_render_var_sync_enabled() -> bool:
    """Return whether a render-var mapping is ordered by a GPU-side wait rather than a host wait.

    See :meth:`OVRTXRenderer._map_render_var_to_dlpack` for why Linux is the exception, and
    :data:`_DISABLE_LINUX_CUDA_CPU_SYNC_ENV` for opting out of it.

    Raises:
        ValueError: If the environment variable is set to anything other than ``0`` or ``1``.
    """
    if not sys.platform.startswith("linux"):
        return True
    value = os.environ.get(_DISABLE_LINUX_CUDA_CPU_SYNC_ENV, "0").strip()
    if value not in {"0", "1"}:
        raise ValueError(
            f"Invalid value for environment variable `{_DISABLE_LINUX_CUDA_CPU_SYNC_ENV}`: {value}. Expected 0 or 1."
        )
    return value == "1"


def _write_file(output_dir: Path, file_name: str, content: str) -> None:
    """Write a UTF-8 debug dump, creating its directory if needed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / file_name
    output_path.write_text(content, encoding="utf-8")
    logger.info("Wrote USD file: %s", output_path)


def _decode_segmentation_map(render_var: RenderVarOutput, decoder: Callable[[IdentifierMap], Any]) -> Any:
    """Decode OVRTX 0.7 label tensors before their mapping is released."""
    with render_var.map(device=Device.CPU) as mapping:
        return decoder({name: np.from_dlpack(mapping[name]) for name in mapping.keys()})


def _copy_label_map(id_map: IdentifierMap) -> dict[str, np.ndarray]:
    """Return a host copy of the tensors of a label map. The copy stays valid after the mapping is released."""
    return {name: tensor.copy() for name, tensor in id_map.items()}


def _label_map_matches(
    id_map: IdentifierMap, first: IdentifierMap, decoder: Callable[[IdentifierMap], Any], expected: Any
) -> bool:
    """Return whether ``id_map`` decodes to ``expected``, the decoded map of the tensors ``first``.

    Equal tensors give equal decoded maps. Decoding is slow, so the function decodes only a map with other tensors.
    """
    same_tensors = id_map.keys() == first.keys() and all(
        tensor.dtype == first[name].dtype and np.array_equal(tensor, first[name]) for name, tensor in id_map.items()
    )
    return same_tensors or decoder(id_map) == expected


def _write_combined_stage(output_dir: Path, render_product_usd: str) -> None:
    """Write the scene and render product prims in one debug layer, preserving scene metadata."""
    scene_layer = Sdf.Layer.OpenAsAnonymous(str(output_dir / "ovrtx_renderer_stage.usda"))
    render_layer = Sdf.Layer.CreateAnonymous("render_product.usda")
    render_layer.ImportFromString(render_product_usd)
    for prim in render_layer.rootPrims:
        Sdf.CopySpec(render_layer, prim.path, scene_layer, prim.path)
    _write_file(output_dir, "ovrtx_renderer_stage.usda", scene_layer.ExportToString())


def _resolve_env_light_partitions(
    stage: Usd.Stage,
    env_paths: Sequence[str],
    *,
    copies: Sequence[tuple[str, Sequence[str]]] = (),
) -> dict[str, str]:
    """Resolve scene-partition tokens for environment-owned lights and their copies."""
    partitions = {Sdf.Path(path): f"env_{index}" for index, path in enumerate(env_paths)}
    roots: dict[Sdf.Path, list[Sdf.Path]] = {path: [] for path in partitions if stage.GetPrimAtPath(path)}
    for source, destinations in copies:
        source_path = Sdf.Path(source)
        if not stage.GetPrimAtPath(source_path):
            continue
        roots.setdefault(source_path, []).extend(map(Sdf.Path, destinations))

    assignments: dict[str, str] = {}
    has_instance_proxy = False
    for source_path, destinations in roots.items():
        for prim in Usd.PrimRange(stage.GetPrimAtPath(source_path), Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdLux.LightAPI):
                continue
            path = prim.GetPath()
            targets = (path, *(path.ReplacePrefix(source_path, destination) for destination in destinations))
            resolved = []
            for target in targets:
                ancestor = target
                while ancestor != Sdf.Path.absoluteRootPath:
                    if ancestor in partitions:
                        resolved.append((str(target), partitions[ancestor]))
                        break
                    ancestor = ancestor.GetParentPath()
            if prim.IsInstanceProxy():
                if resolved:
                    has_instance_proxy = True
            else:
                assignments.update(resolved)

    if has_instance_proxy:
        logger.warning(
            "Lights inside USD instances cannot be assigned to environment scene partitions and may affect other"
            " environments."
        )
    return assignments


class OVRTXBackend:
    """Own one native renderer, independently of its camera products and optional OVStage."""

    def __init__(self, cfg: OVRTXBackendCfg):
        self.cfg = cfg
        self.attached = False
        self.render_products: set[str] = set()
        # Camera sensors of every renderer on this engine without a rendered frame, and their renderers.
        # The split search can replace their render products until a step renders.
        self.unrendered: dict[OVRTXCameraRenderData, OVRTXRenderer] = {}
        # Resolve the wheel's native dependency here, including callers without a viewer.
        dependency = Path(ovstage.__file__).parent / "bin/plugins/libosdCPU.so.3.6.0"
        if dependency.exists():
            with contextlib.suppress(OSError):
                ctypes.CDLL(str(dependency))
        # Select the simulation's CUDA device for rendering, matching Kit's ``activeCudaGpus`` setting.
        sim_device = wp.get_device(cfg.device)
        native_cfg = RendererConfig(
            active_cuda_gpus=str(sim_device.ordinal) if sim_device.is_cuda else None,
            log_file_path=cfg.log_file_path,
            log_level=cfg.log_level,
            motion_bvh=cfg.motion_bvh,
            read_gpu_transforms=cfg.read_gpu_transforms,
            keep_system_alive=True,
            suppress_deprecation_warnings=True,
            texture_streaming_mode=TextureStreamingMode.SYNCHRONOUS,
        )
        # Redirection may initialize the native library and must use the same settings.
        redirect_shader_cache(native_cfg)
        self.next_camera_id = 0
        if not cfg.use_ovstage:
            # Prepared by clone dispatch, consumed after camera overrides are authored.
            self.clone_copies: list[tuple[str, list[str]]] = []
            self.clone_env_paths: list[str] = []
            self.population_env_paths: list[str] = []
            self.clone_positions: np.ndarray | None = None
        self.renderer = Renderer(native_cfg)

    def detach(self) -> None:
        """Release the borrowed stage before it is rebuilt or destroyed."""
        if self.attached:
            self.renderer.detach_ovstage()
            self.attached = False

    def close(self) -> None:
        """Destroy the engine; the simulation's backend registry releases any borrowed stage afterward."""
        if self.renderer is not None:
            # The SDK destroys bindings and detaches only if attached, including partial initialization.
            self.renderer.destroy()
            self.renderer = None
            self.attached = False


def _split_priority(data: OVRTXCameraRenderData) -> int:
    """Return the pixels of the first and largest atlas of a camera sensor times its samples per pixel.

    The largest path tracer buffer grows with both, so the split search halves the sensor with the highest value first.
    """
    product = data.products[0]
    spp = int(data.spec.cfg.render_settings.get("omni:rtx:rtpt:spp", ("Int", 1))[1])
    return product.num_cols * data.width * product.num_rows * data.height * spp


@dataclasses.dataclass(frozen=True)
class _SplitPlan:
    """Next layouts of the camera sensors without a rendered frame after a failed step.

    :func:`_plan_split` decides the plan without a change to any sensor or scene.
    :meth:`OVRTXRenderer._apply_split_plan` then changes the sensors and the scene.
    """

    max_envs_per_render_product: tuple[tuple[OVRTXCameraRenderData, int | None], ...] = ()
    """Each sensor with its next number of environments per render product, in the order of the engine.
    ``None`` means that the search gives up on the sensor."""
    halved: OVRTXCameraRenderData | None = None
    """Sensor with half as many environments per render product. ``None`` if the search gives up or if the plan is
    empty. The caller then raises the error of the failed step."""


def _plan_split(cameras: Sequence[OVRTXCameraRenderData], error: RuntimeError, num_smaller_layouts: int) -> _SplitPlan:
    """Plan the next layouts of the camera sensors without a rendered frame after a failed step.

    OVRTX reports a failed frame only in its error text, so the plan is empty for any other error. A candidate has
    ``searches_split`` and more than one environment in its first render product. ``searches_split`` needs
    ``"auto"`` for :attr:`~isaaclab_ov.renderers.OVRTXRendererCfg.max_envs_per_render_product` and one camera path
    per environment. Of the candidates, the one with the highest :func:`_split_priority` gets half as many
    environments per render product. The other sensors keep their split.

    Without a candidate, or after ``_MAX_SMALLER_LAYOUTS`` smaller layouts, the plan halves no sensor. The search
    then gives up on every sensor with ``searches_split``. Each failed layout stays in the scene until the sensor
    cleanup, so ``_MAX_SMALLER_LAYOUTS`` limits the smaller layouts of one render batch.

    Args:
        cameras: Camera sensors without a rendered frame, in the order of the engine.
        error: Error of the failed step.
        num_smaller_layouts: Number of smaller layouts that this render batch tried before ``error``.
    """
    if _FAILED_FRAME_TEXT not in str(error):
        return _SplitPlan()
    # The first render product of a sensor has the most environments.
    candidates = [data for data in cameras if data.searches_split and data.products[0].num_envs > 1]
    halved = max(candidates, key=_split_priority) if candidates and num_smaller_layouts < _MAX_SMALLER_LAYOUTS else None

    def next_max_envs(data: OVRTXCameraRenderData) -> int | None:
        if halved is None and data.searches_split:
            return None
        num_envs = data.products[0].num_envs
        return math.ceil(num_envs / 2) if data is halved else num_envs

    return _SplitPlan(tuple((data, next_max_envs(data)) for data in cameras), halved)


@dataclasses.dataclass(frozen=True)
class OVRTXRenderProduct:
    """One render product of a camera sensor. It renders a contiguous range of environments into one tiled atlas.

    One atlas for all environments of a sensor can be too large for the renderer, the driver or the GPU memory.
    So a sensor can have several render products.
    :attr:`~isaaclab_ov.renderers.OVRTXRendererCfg.max_envs_per_render_product` sets the environments of one render
    product.
    """

    path: str
    """Absolute prim path of the ``RenderProduct``."""
    camera_paths: list[str]
    """Camera prim paths of these environments, in the environment order of the sensor."""
    env_offset: int
    """Index of the first environment of this render product in the sensor."""
    num_envs: int
    """Number of environments of this render product."""
    num_cols: int
    """Number of tile columns in the atlas of this product, ``ceil(sqrt(num_envs))``."""
    num_rows: int
    """Number of tile rows in the atlas of this product, ``ceil(num_envs / num_cols)``."""


class OVRTXCameraRenderData:
    """Owns one camera sensor's native resources and borrowed output buffers.

    Each render product renders a contiguous range of environments into its own tiled atlas. All of them use
    the RenderVars in the scope of the sensor. Until the first frame renders, the split search can change them.
    """

    def __init__(
        self,
        spec: CameraRenderSpec,
        render_scope_name: str,
        camera_paths: list[str],
        max_envs_per_render_product: int | Literal["auto", "all"],
    ):
        """Create render data for a camera in its assigned render scope.

        Args:
            spec: Camera render specification.
            render_scope_name: Root scope containing this camera's render products and RenderVars.
            camera_paths: Concrete camera paths in sensor row order.
            max_envs_per_render_product: Maximum number of environments in one render product, see
                :attr:`~isaaclab_ov.renderers.OVRTXRendererCfg.max_envs_per_render_product`.
        """
        self.spec = spec
        self.camera_paths = camera_paths
        self.camera_xform_binding: AttributeBinding[wp.array] | ovstage.Query | None = None
        self.resources = contextlib.ExitStack()
        self.width = spec.cfg.width
        self.height = spec.cfg.height
        self.num_envs = spec.num_instances
        self.data_types = spec.cfg.data_types
        max_envs = max_envs_per_render_product
        # Whether the split search may halve the environments per render product of this sensor.
        self.searches_split = max_envs == "auto" and len(camera_paths) == self.num_envs
        if max_envs == "auto" and not self.searches_split:
            # A split needs one camera path per environment. create_render_data completes the paths of a camera below
            # an environment root, so a different number is unexpected.
            logger.warning(
                "Camera %s has %d camera path(s) for %d environments, so the split search is off for it."
                ' Set OVRTXRendererCfg.max_envs_per_render_product="all" for its renderer to turn this warning off.',
                spec.cfg.prim_path,
                len(camera_paths),
                self.num_envs,
            )
        # The error that ended the split search before the first frame of this sensor. It is the failed frame if the
        # search gave up, or a failed scene write. Later renders raise at once, with this error as the cause.
        self.first_frame_error: Exception | None = None
        # "auto" and "all" both start with one render product for all environments.
        self._set_layout(render_scope_name, max_envs if isinstance(max_envs, int) else self.num_envs)
        self.warp_buffers: dict[str, wp.array] = {}
        self.intrinsic_bindings: list[AttributeBinding] = []
        self.camera_writes = _AsyncWriteBuffers[wp.array]()
        self.pending: tuple[Operation[PendingFetch[RenderProductSetOutputs]], dict | None] | None = None
        self.ready: tuple[Operation[PendingFetch[RenderProductSetOutputs]], dict | None] | None = None
        self.capture: dict[str, ProxyArray] = {}
        # Per-output metadata collected during render() and copied into CameraData.info by read_output().
        # Populated for "semantic_segmentation" (with an "idToLabels" mapping) and
        # "instance_segmentation" (with "idToLabels" and "idToSemantics" mappings).
        self.renderer_info: dict[str, Any] = {}

    @property
    def render_product_paths(self) -> tuple[str, ...]:
        """Paths of every render product of this sensor, in environment order."""
        return tuple(product.path for product in self.products)

    @property
    def render_product_path(self) -> str:
        """Path of the render product of a camera sensor with one render product.

        .. deprecated:: 9.3.0
            Use :attr:`OVRTXRenderProduct.path` of each entry of :attr:`products`.

        Raises:
            RuntimeError: If the camera sensor has several render products.
        """
        return self._single_product("render_product_path").path

    @property
    def num_cols(self) -> int:
        """Number of tile columns in the tiled image of a camera sensor with one render product.

        .. deprecated:: 9.3.0
            Use :attr:`OVRTXRenderProduct.num_cols` of each entry of :attr:`products`.

        Raises:
            RuntimeError: If the camera sensor has several render products.
        """
        return self._single_product("num_cols").num_cols

    @property
    def num_rows(self) -> int:
        """Number of tile rows in the tiled image of a camera sensor with one render product.

        .. deprecated:: 9.3.0
            Use :attr:`OVRTXRenderProduct.num_rows` of each entry of :attr:`products`.

        Raises:
            RuntimeError: If the camera sensor has several render products.
        """
        return self._single_product("num_rows").num_rows

    def cleanup(self) -> None:
        """Release this camera's native resources and buffers. Safe to call repeatedly.

        Resources are released in reverse acquisition order, before the shared renderer or stage
        is closed. The render product paths remain as identifiers for renderer bookkeeping.
        """
        try:
            with self.resources:
                self.camera_writes.close()
        finally:
            self.pending = self.ready = None
            self.capture = {}
            self.camera_xform_binding = None
            self.intrinsic_bindings.clear()
            self.warp_buffers.clear()
            self.renderer_info.clear()

    def _set_layout(self, render_scope_name: str, max_envs: int) -> None:
        """Divide the environments over render products of at most ``max_envs`` environments each.

        Args:
            render_scope_name: Root scope of the render products and RenderVars.
            max_envs: Maximum number of environments in one render product, at least 1.

        Raises:
            ValueError: If the sensor needs several render products but lacks one camera path per environment.
        """
        self.render_scope_name = render_scope_name
        self.render_var_keys = render_var_prim_paths_by_source(render_scope_name)
        envs_per_product = min(max_envs, self.num_envs)
        count = math.ceil(self.num_envs / envs_per_product)
        if count > 1 and len(self.camera_paths) != self.num_envs:
            raise ValueError(
                "OVRTXRendererCfg.max_envs_per_render_product needs one camera path per environment, got"
                f" {len(self.camera_paths)} camera path(s) for {self.num_envs} environments."
                ' Use "all" for the renderer of this sensor.'
            )
        self.products: list[OVRTXRenderProduct] = []
        for k in range(count):
            start, end = k * envs_per_product, min((k + 1) * envs_per_product, self.num_envs)
            envs = end - start
            cols = _tile_columns(envs)
            # One render product keeps the plain name, so "auto", "all" and a limit of at least num_envs author
            # the same layer.
            name = "RenderProduct" if count == 1 else f"RenderProduct_{k}"
            self.products.append(
                OVRTXRenderProduct(
                    path=f"/{render_scope_name}/{name}",
                    camera_paths=self.camera_paths[start:end],
                    env_offset=start,
                    num_envs=envs,
                    num_cols=cols,
                    num_rows=math.ceil(envs / cols),
                )
            )

    def _single_product(self, name: str) -> OVRTXRenderProduct:
        """Return the only render product for the deprecated property ``name``.

        A split camera sensor raises, because the property would describe only a part of its environments.
        """
        warnings.warn(
            f"OVRTXCameraRenderData.{name} is deprecated. Use OVRTXCameraRenderData.products instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        if len(self.products) > 1:
            raise RuntimeError(
                f"OVRTXCameraRenderData.{name} needs a camera sensor with one render product, but this sensor has"
                f" {len(self.products)}. Use OVRTXCameraRenderData.products instead."
            )
        return self.products[0]


class OVRTXLidarRenderData:
    """Native resources and capture interval for one tiled lidar."""

    def __init__(self, scope: str, sample_dt: float, lidar_paths: list[str]):
        self.lidar_paths = lidar_paths
        self.render_product_path = f"/{scope}/RenderProduct"
        self.point_cloud_path = f"/{scope}/Vars/PointCloud"
        self.xform_binding: AttributeBinding[wp.array] | ovstage.Query | None = None
        self.resources = contextlib.ExitStack()
        self.pose_writes = _AsyncWriteBuffers[wp.array]()
        self.sample_dt = sample_dt

    def cleanup(self) -> None:
        """Release the pose writes before releasing the binding."""
        with self.resources:
            self.pose_writes.close()
        self.xform_binding = None


class OVRTXRenderer(BaseRenderer):
    """OVRTX Renderer implementation using the ovrtx library.

    This renderer uses the ovrtx library for high-fidelity RTX-based rendering,
    providing ray-traced rendering capabilities for Isaac Lab environments.
    """

    cfg: OVRTXRendererCfg

    def supported_output_types(self) -> dict[RenderBufferKind, RenderBufferSpec]:
        """Publish the per-output layout this OVRTX backend writes.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.supported_output_types`."""
        return self.cfg.supported_output_types()

    def __init__(self, cfg: OVRTXRendererCfg, *, sim: SimulationContext):
        super().__init__(cfg, sim=sim)
        self._warp_device = wp.get_device(self._sim.device)
        self._device = str(self._warp_device)
        self._lidars: list[OVRTXLidarRenderData] = []
        self._camera_render_data: list[OVRTXCameraRenderData] = []
        self.meshes: dict[str, OVRTXMesh] = {}
        self.markers: dict[Any, OVRTXMarkers] = {}
        self.marker_groups: Iterable = ()
        self.geometry_callbacks: dict[Any, Callable[[OVRTXRenderer], None]] = {}
        self.geometry_resources = contextlib.ExitStack()
        self._sdp = self._sim.get_scene_data_provider()
        self._transforms_timestamp = -1
        self._transform_writes = _AsyncWriteBuffers(SceneDataFormat.TransposedMatrix44d() for _ in range(2))
        self._object_scales: wp.array | None = None
        self._geometry_paths: list[str] = []
        self._visible_envs: np.ndarray | None = None
        self._hidden_marker_rows: torch.Tensor | None = None
        self._hidden_envs: wp.array | None = None
        self._body_envs: wp.array | None = None
        self._hidden_matrices: dict[int, wp.array] = {}
        self._geometry_offsets: dict[str, int] = {}
        self._geometry_writes = _AsyncWriteBuffers[wp.array]()
        self._geometry_timestamp = -1
        self._initialized_scene = False
        # Flat scratch buffers for colorized segmentation, one per output, sized for the largest atlas.
        self._output_id_color_buffers: dict[str, wp.array] = {}
        self._semantic_palette_key: tuple[tuple[RgbaColor, RgbaColor], ...] = ()
        self._semantic_palette_buffers: tuple[wp.array, wp.array] | None = None
        self._visual_material_writer_ref: weakref.ReferenceType[OVRTXVisualMaterialWriter] | None = None

        # Clone preparation resolves scene ownership before acquiring native resources.
        self._use_ovstage = False
        self._needs_transform_initialization = False
        self.scene: OvstageBackend | OVRTXBackend | None = None
        self.backend: OVRTXBackend | None = None
        """Native engine borrowed from the simulation registry after clone preparation."""
        self._object_xform_binding: AttributeBinding[wp.array] | ovstage.Query | None = None
        self._geometry_points_binding: AttributeBinding[list[wp.array]] | ovstage.Query | None = None
        self._bindings = contextlib.ExitStack()
        self._env_paths: Sequence[str] = ()

    @property
    def visual_material_writer(self):
        """Return a writer factory once this renderer has a scene consumer."""
        return self._create_visual_material_writer if self._initialized_scene else None

    def _create_visual_material_writer(self, batches: tuple[VisualMaterialBatch, ...]) -> OVRTXVisualMaterialWriter:
        """Create the detached-scene material writer after scene population."""
        writer = OVRTXVisualMaterialWriter(self, batches)
        self._visual_material_writer_ref = weakref.ref(writer)
        return writer

    def prepare_stage(self, stage: Usd.Stage, num_envs: int) -> None:
        """Load, clone, and bind the shared scene before any camera products are registered."""
        logger.info("Preparing stage (%d envs)...", num_envs)
        env_paths = self._env_paths = self.scene.clone_env_paths
        light_partitions = _resolve_env_light_partitions(stage, env_paths, copies=self.scene.clone_copies)
        if self.cfg.temp_usd_dir is not None or not self._use_ovstage or not self.scene.ordinal:
            if self.cfg.temp_usd_dir is not None:
                _write_file(Path(self.cfg.temp_usd_dir), "pre_ovrtx_renderer_stage.usda", stage.ExportToString())
            # Physics population needs fully composed instance collider data.
            usd = export_stage_to_string(
                stage, self.scene.clone_env_paths, source_paths=tuple(source for source, _ in self.scene.clone_copies),
                keep_env_roots=False, env_paths=self.scene.population_env_paths,
                preserve_references=not (self._use_ovstage and self.scene.cfg.populate_physics),
            )  # fmt: skip
            if not self._use_ovstage:
                # Author the camera mode before loading the stage: changing it later
                # leaves cached camera poses stale until another scene update.
                settings_path = Sdf.Path(stage.GetMetadata("renderSettingsPrimPath") or "/Render/IsaacLabSettings")
                layer = Sdf.Layer.CreateAnonymous()
                layer.ImportFromString(usd)
                attribute = layer.GetAttributeAtPath(settings_path.AppendProperty(_GPU_CAMERA_TRANSFORMS_ATTRIBUTE))
                if attribute is None:
                    attribute = Sdf.AttributeSpec(
                        layer.GetPrimAtPath(settings_path), _GPU_CAMERA_TRANSFORMS_ATTRIBUTE, Sdf.ValueTypeNames.Bool
                    )
                attribute.default = False
                usd = layer.ExportToString()
            if self.cfg.temp_usd_dir is not None:
                _write_file(Path(self.cfg.temp_usd_dir), "ovrtx_renderer_stage.usda", usd)
            if self._use_ovstage:
                self.scene.populate(usd)
            else:
                self.backend.renderer.open_usd_from_string(usd)
                ovrtx_replicate(self.backend.renderer, self.scene.clone_copies, env_paths, self.scene.clone_positions)
        partition_writes = (
            (env_paths, "primvars:omni:scenePartition", [f"env_{i}" for i in range(len(env_paths))]),
            (list(light_partitions), "omni:scenePartition", list(light_partitions.values())),
        )
        for paths, attribute_name, tokens in partition_writes:
            if not paths:
                continue
            if self._use_ovstage:
                with self.scene.query(paths) as query:
                    self.scene.stage.write_attribute(
                        query, attribute_name, ordinal=self.scene.ordinal,
                        tensors=np.array(
                            [self.scene.paths.intern_token(token).raw_id for token in tokens], dtype=np.uint64
                        ),
                        is_array=False, semantic=ovstage.AttributeSemantic.TOKEN_ID,
                    ).wait()  # fmt: skip
            else:
                self.backend.renderer.write_attribute(
                    paths, attribute_name, tokens,
                    semantic=Semantic.TOKEN_STRING, prim_mode=PrimMode.CREATE_NEW,
                )  # fmt: skip
        self._setup_xform_bindings(stage)
        self._setup_geometry_bindings()
        if self._use_ovstage:
            # OVRTX 0.7's GPU camera cache mixes tiled views and misses pose updates.
            # Disable only camera reads before registering products (and before attaching
            # an OVStage), retaining GPU instance transforms for large articulated scenes.
            settings_path = str(stage.GetMetadata("renderSettingsPrimPath") or "/Render/IsaacLabSettings")
            cpu_cameras = np.array([False], dtype=np.bool_)
            with self.scene.query([settings_path]) as query:
                self.scene.stage.write_attribute(
                    query, _GPU_CAMERA_TRANSFORMS_ATTRIBUTE, ordinal=self.scene.ordinal,
                    tensors=cpu_cameras, is_array=False,
                ).wait()  # fmt: skip
        if self._use_ovstage:
            self.scene.commit()
            if not self.backend.attached:
                self.backend.renderer.attach_ovstage(self.scene.stage)
                self.backend.attached = True
                self.scene.resources.callback(self.backend.detach)
            self.scene.resources.callback(self._sim.render_context.reset_stage_prepare_flag)
            self.scene.resources.callback(self.close)
        self._initialized_scene = True
        self._needs_transform_initialization = not self._use_ovstage and self.backend.cfg.read_gpu_transforms

    def _setup_xform_bindings(self, stage: Usd.Stage) -> None:
        """Bind SDP body paths and preserve their authored scales when writing world poses."""
        object_paths = self._sdp.backend.transform_paths
        if not object_paths:
            return
        # SDP poses omit scale. Include native copies that do not exist on the authored USD stage.
        scales_by_path = {}
        xform_cache = UsdGeom.XformCache()
        for root, targets in self.scene.clone_copies:
            for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
                if not prim.IsA(UsdGeom.Xformable):
                    continue
                scale = tuple(map(float, Gf.Transform(xform_cache.GetLocalToWorldTransform(prim)).GetScale()))
                if not all(math.isclose(axis, 1.0, rel_tol=1e-6, abs_tol=1e-6) for axis in scale):
                    path = str(prim.GetPath())
                    scales_by_path[path] = scale
                    scales_by_path.update((target + path[len(root) :], scale) for target in targets)
        self._object_scales = None
        # No override lets renderers share SDP's publication instead of multiplying by ones.
        if any(path in scales_by_path for path in object_paths):
            scales = [scales_by_path.get(path, (1.0, 1.0, 1.0)) for path in object_paths]
            self._object_scales = wp.array(scales, dtype=wp.vec3f, device=self._device)
        reset_xforms = np.ones(len(object_paths), dtype=np.bool_)
        if self._use_ovstage:
            self._object_xform_binding = self._bindings.enter_context(self.scene.query(object_paths))
            self.scene.stage.write_attribute(
                self._object_xform_binding,
                "omni:resetXformStack",
                ordinal=self.scene.ordinal,
                tensors=reset_xforms,
                is_array=False,
            ).wait()
        else:
            self._object_xform_binding = self.backend.renderer.bind_attribute(
                prim_paths=object_paths,
                attribute_name="omni:xform",
                semantic=Semantic.XFORM_MAT4x4,
                prim_mode=PrimMode.EXISTING_ONLY,
            )
            if self._object_xform_binding is None:
                raise RuntimeError("Failed to create OVRTX object bindings")
            self._bindings.callback(self._object_xform_binding.unbind)
            self.backend.renderer.write_attribute(
                prim_paths=object_paths, attribute_name="omni:resetXformStack", tensor=reset_xforms
            )

    def _setup_geometry_bindings(self) -> None:
        """Bind SDP's world-space point prims without applying inherited transforms again."""
        points = self._sdp.get_geometry_points()
        self._geometry_paths = list(points)
        if not self._geometry_paths:
            return
        prim_count = len(self._geometry_paths)
        reset_xforms = np.ones(prim_count, dtype=np.bool_)
        identity_xforms = np.tile(np.eye(4, dtype=np.float64), (prim_count, 1, 1))
        if self._use_ovstage:
            self._geometry_points_binding = self._bindings.enter_context(self.scene.query(self._geometry_paths))
            self.scene.stage.write_attribute(
                self._geometry_points_binding,
                "omni:resetXformStack",
                ordinal=self.scene.ordinal,
                tensors=reset_xforms,
                is_array=False,
            ).wait()
            self.scene.stage.write_attribute(
                self._geometry_points_binding,
                "omni:xform",
                ordinal=self.scene.ordinal,
                tensors=xform_tensor_from_numpy(identity_xforms),
                is_array=False,
                semantic=ovstage.AttributeSemantic.MATRIX,
            ).wait()
        else:
            if self.cfg.async_rendering:
                count = 0
                for path, array in points.items():
                    self._geometry_offsets[path] = count
                    count += len(array)
                self._geometry_writes = _AsyncWriteBuffers(
                    wp.empty(count, wp.vec3f, device=self._device) for _ in range(2)
                )
            self.backend.renderer.write_attribute(
                prim_paths=self._geometry_paths,
                attribute_name="omni:resetXformStack",
                tensor=reset_xforms,
                prim_mode=PrimMode.MUST_EXIST,
            )
            self.backend.renderer.write_attribute(
                prim_paths=self._geometry_paths,
                attribute_name="omni:xform",
                tensor=identity_xforms,
                semantic=Semantic.XFORM_MAT4x4,
                prim_mode=PrimMode.MUST_EXIST,
            )
            self._geometry_points_binding = self.backend.renderer.bind_array_attribute(
                prim_paths=self._geometry_paths,
                attribute_name="points",
                dtype=np.float32,
                shape=(3,),
                prim_mode=PrimMode.MUST_EXIST,
                flags=BindingFlag.OPTIMIZE,
            )
            if self._geometry_points_binding is None:
                raise RuntimeError("Failed to create OVRTX geometry point bindings")
            self._bindings.callback(self._geometry_points_binding.unbind)

    def _resolve_environment_sensor_paths(
        self, prim_paths: list[str], count: int, *, environment_order: bool
    ) -> tuple[list[str], list[str]]:
        """Expand a cloned sensor to one prim per environment and return its partition tokens.

        Native cloning may leave only the source prim in USD. The suffix below its environment
        root is applied to every path in the clone plan. A shorter list is completed that way.
        ``environment_order`` also replaces a complete list with the clone-plan order when it
        names the same prims, so tile ``i`` is environment ``i``. Otherwise a complete list keeps
        its supplied order. Paths outside a cloned environment are unchanged and have no tokens.
        """
        env_partitions = {Sdf.Path(path): f"env_{i}" for i, path in enumerate(self.scene.clone_env_paths)}
        source_root = next((root for root in Sdf.Path(prim_paths[0]).GetPrefixes() if root in env_partitions), None)
        if source_root is None:
            return prim_paths, []
        suffix = prim_paths[0][len(str(source_root)) :]
        ordered = [path + suffix for path in self.scene.clone_env_paths]
        if len(prim_paths) < count or (environment_order and set(ordered) == set(prim_paths)):
            prim_paths = ordered
        partitions = [
            next(env_partitions[root] for root in Sdf.Path(path).GetPrefixes() if root in env_partitions)
            for path in prim_paths
        ]
        return prim_paths, partitions

    def create_lidar_render_data(
        self, cfg: OVRTXLidarCfg, lidar_paths: list[str], num_envs: int
    ) -> OVRTXLidarRenderData:
        """Register a tiled lidar product with one prim per environment."""
        prim_path = cfg.prim_path
        if not lidar_paths:
            raise RuntimeError(f"OVRTX lidar '{prim_path}' expected {num_envs} prims in environment order, found 0.")

        lidar_paths, lidar_partitions = self._resolve_environment_sensor_paths(
            lidar_paths, num_envs, environment_order=True
        )

        if len(lidar_paths) != num_envs:
            raise RuntimeError(
                f"OVRTX lidar '{prim_path}' expected {num_envs} prims in environment order, found {len(lidar_paths)}."
            )

        scope = f"RenderLidar_{self.scene.next_camera_id}"
        update_period = float(cfg.update_period)
        render_data = OVRTXLidarRenderData(scope, update_period or self._sim.get_physics_dt(), lidar_paths)
        product_path = render_data.render_product_path

        usd = build_lidar_render_product_as_string(
            render_scope_name=scope,
            lidar_path=lidar_paths[0],
            device_id=self._warp_device.ordinal,
        )

        try:
            if self._use_ovstage:
                reference = ovstage.population.add_usd_reference_from_string(self.scene.stage, usd, f"/{scope}")
                render_data.resources.callback(self._remove_usd_reference, reference)

                ovstage.population.apply_usd_changes(self.scene.stage, ordinal=self.scene.ordinal)

                with self.scene.query([product_path]) as query:
                    self.scene.stage.write_attribute(
                        query,
                        "camera",
                        ordinal=self.scene.ordinal,
                        tensors=np.array(
                            [self.scene.paths.intern_path(path).raw_id for path in lidar_paths],
                            dtype=np.uint64,
                        ),
                        is_array=True,
                        semantic=ovstage.AttributeSemantic.RELATIONSHIP_PATH_ID,
                    ).wait()

                render_data.xform_binding = render_data.resources.enter_context(self.scene.query(lidar_paths))

                self.scene.stage.write_attribute(
                    render_data.xform_binding,
                    "omni:resetXformStack",
                    ordinal=self.scene.ordinal,
                    tensors=np.full(num_envs, True, dtype=np.bool_),
                    is_array=False,
                ).wait()

                if lidar_partitions:
                    self.scene.stage.write_attribute(
                        render_data.xform_binding,
                        "omni:scenePartition",
                        ordinal=self.scene.ordinal,
                        tensors=np.array(
                            [self.scene.paths.intern_token(partition).raw_id for partition in lidar_partitions],
                            dtype=np.uint64,
                        ),
                        is_array=False,
                        semantic=ovstage.AttributeSemantic.TOKEN_ID,
                    ).wait()
            else:
                reference = self.backend.renderer.add_usd_reference_from_string(usd, f"/{scope}")
                render_data.resources.callback(self.backend.renderer.remove_usd, reference)

                self.backend.renderer.write_array_attribute(
                    prim_paths=[product_path],
                    attribute_name="camera",
                    tensors=[lidar_paths],
                )

                render_data.xform_binding = self.backend.renderer.bind_attribute(
                    prim_paths=lidar_paths,
                    attribute_name="omni:xform",
                    semantic=Semantic.XFORM_MAT4x4,
                    prim_mode=PrimMode.EXISTING_ONLY,
                )
                render_data.resources.callback(render_data.xform_binding.unbind)

                self.backend.renderer.write_attribute(
                    prim_paths=lidar_paths,
                    attribute_name="omni:resetXformStack",
                    tensor=np.full(num_envs, True, dtype=np.bool_),
                )

                if lidar_partitions:
                    self.backend.renderer.write_attribute(
                        lidar_paths,
                        "omni:scenePartition",
                        lidar_partitions,
                        semantic=Semantic.TOKEN_STRING,
                    )

            render_data.pose_writes = _AsyncWriteBuffers(
                wp.empty(num_envs, wp.mat44d, device=self._device) for _ in range(1)
            )
        except Exception:
            render_data.cleanup()
            raise

        self.scene.next_camera_id += 1
        self._lidars.append(render_data)
        return render_data

    @contextlib.contextmanager
    def _map_lidar_point_cloud(self, point_cloud, device: Device) -> Iterator[Any]:
        """Map a lidar point cloud for reads and release it after the consumer returns.

        CUDA maps wait for rendering on the consuming Warp stream, or on the host on Linux unless
        :data:`_DISABLE_LINUX_CUDA_CPU_SYNC_ENV` is ``1``. Unmapping records that stream so native
        release waits for queued copies. CPU maps use the native context manager.

        Args:
            point_cloud: OVRTX point-cloud render variable.
            device: ``Device.CUDA`` or ``Device.CPU``.
        """
        if device is not Device.CUDA:
            with point_cloud.map(device=device, sync_stream=0) as mapped:
                if mapped.wait_event is not None:
                    mapped.wait()
                yield mapped
            return

        gpu_side_sync = _gpu_side_render_var_sync_enabled()
        stream = self._warp_device.stream.cuda_stream or 1
        mapping = point_cloud.map(device=Device.CUDA, sync_stream=stream if gpu_side_sync else 0)
        try:
            if not gpu_side_sync:
                mapping.wait()
            yield mapping
        finally:
            mapping.unmap(stream=stream)

    def render_lidar(self, render_data: OVRTXLidarRenderData, lidar_data: LidarData) -> None:
        """Capture one tiled lidar product and publish its point cloud.

        The capture submits only this lidar. Another lidar's pose is written when that lidar
        captures, so including it here would replace its buffers with a scan from a stale pose.
        Camera products stay on the camera step. A lidar needs the point cloud immediately, so this
        submission waits.
        """
        if not self._initialized_scene:
            raise RuntimeError("Scene not initialized. Call initialize() first.")

        lidars = self._lidars
        if self.backend is None or self.backend.renderer is None:
            raise RuntimeError(f"Cannot capture {render_data.render_product_path!r}: OvRTX renderer is unavailable.")
        if render_data not in lidars:
            raise RuntimeError(f"OvRTX lidar product {render_data.render_product_path!r} is not registered.")

        delta_time = render_data.sample_dt
        if delta_time <= 0.0:
            delta_time = _RENDER_DELTA_TIME

        # The first step initializes GPU transforms from authored poses and must not advance any
        # lidar clock. The capture itself includes only the lidar whose pose was just written.
        products = {data.render_product_path for data in lidars}
        requested = {render_data.render_product_path}

        material_writer = self._visual_material_writer_ref() if self._visual_material_writer_ref is not None else None
        try:
            if material_writer is not None:
                material_writer.publish()
            if self._needs_transform_initialization:
                self.backend.renderer.step(render_products=products, delta_time=0.0)
                self._needs_transform_initialization = False
            if self._use_ovstage:
                ordinal = self.scene.commit()
            else:
                result = self.backend.renderer.step(render_products=requested, delta_time=delta_time)
        finally:
            if material_writer is not None:
                primary_active = sys.exc_info()[0] is not None
                try:
                    material_writer.drain()
                except Exception as exc:
                    if not primary_active:
                        raise
                    logger.warning("Error draining material writes after a failed render: %s", exc, exc_info=True)

        if self._use_ovstage:
            result = self.backend.renderer.step(render_products=requested, delta_time=delta_time, ordinal=ordinal)

        self._publish_lidar_output(render_data, result, lidar_data)

    def _publish_lidar_output(
        self, render_data: OVRTXLidarRenderData, products: RenderProductSetOutputs, lidar_data: LidarData
    ) -> None:
        """Split one lidar product's point cloud into ``lidar_data``.

        A missing frame or point-cloud variable raises. If the CUDA map is unavailable,
        retry on the CPU and propagate any fallback readback error.
        """
        if render_data.render_product_path not in products or not products[render_data.render_product_path].frames:
            raise RuntimeError(f"OVRTX returned no frame for lidar product {render_data.render_product_path!r}.")

        frames = products[render_data.render_product_path].frames
        frame = frames[-1]
        point_cloud = frame.render_vars.get(render_data.point_cloud_path)
        if point_cloud is None:
            raise RuntimeError(
                f"OVRTX lidar frame has no point cloud at {render_data.point_cloud_path!r}. "
                f"Available render vars: {list(frame.render_vars)}."
            )

        try:
            self._copy_mapped_lidar_point_cloud(point_cloud, lidar_data, Device.CUDA)
        except Exception as exc:
            message = str(exc).lower()
            if not ("failed to map" in message or "unpublish" in message or "point cloud was empty" in message):
                raise

            logger.info("Retrying OvRTX lidar point-cloud readback on the CPU: %s", exc)
            self._copy_mapped_lidar_point_cloud(point_cloud, lidar_data, Device.CPU)

    def _copy_mapped_lidar_point_cloud(self, point_cloud, lidar_data: LidarData, device: Device) -> None:
        """Map one point cloud and copy it into the lidar buffers.

        A CUDA map that produces no coordinates raises so the caller can fall back to the CPU map.
        The dense CUDA copy is queued on the consuming Warp stream. Unmapping records that stream so
        the mapped views stay valid until the copy completes.
        """

        def mapped_point_count(mapped) -> int:
            """Return the coordinate count without exporting the tensor, or zero when it is unusable."""
            try:
                if "Coordinates" not in mapped:
                    return 0
                shape = mapped["Coordinates"].shape
            except (TypeError, KeyError, RuntimeError, AttributeError):
                return 0

            count = 1
            for dim in shape:
                count *= int(dim)
            return count

        with self._map_lidar_point_cloud(point_cloud, device) as mapped:
            if device is Device.CUDA and mapped_point_count(mapped) == 0:
                raise RuntimeError("OVRTX lidar CUDA point cloud was empty")

            copy_tiled_point_cloud(mapped, lidar_data.points.torch, lidar_data.intensity.torch, lidar_data.valid.torch)

    def update_lidar(
        self,
        render_data: OVRTXLidarRenderData,
        positions: ProxyArray,
        orientations: ProxyArray,
    ) -> None:
        """Update lidar poses using the lidar's reusable conversion buffers."""
        binding = render_data.xform_binding
        if binding is None:
            return
        stream = self._warp_device.stream
        matrices = render_data.pose_writes.acquire(stream)
        wp.launch(
            create_lidar_transforms_kernel, len(matrices), [positions, orientations, matrices], device=self._device
        )
        if self._use_ovstage:
            self.scene.stage.write_attribute(
                binding,
                "omni:xform",
                ordinal=self.scene.ordinal,
                tensors=xform_tensor_from_warp(matrices),
                is_array=False,
                semantic=ovstage.AttributeSemantic.MATRIX,
                cuda_stream=stream.cuda_stream or 1,
            ).wait()
        else:
            operation = render_data.pose_writes.submit(binding, matrices, stream)
            if not self.cfg.async_rendering:
                operation.wait()

    def cleanup_lidar(self, render_data: OVRTXLidarRenderData | None) -> None:
        """Release one lidar's native resources."""
        if render_data in self._lidars:
            self._lidars.remove(render_data)
            render_data.cleanup()

    def create_render_data(self, spec: CameraRenderSpec) -> OVRTXCameraRenderData:
        """Register the render products and bindings of a camera sensor in the prepared scene."""
        if str(wp.get_device(spec.device)) != self._device:
            raise ValueError("Cameras sharing an OVRTX renderer must use the simulation device.")
        camera_paths = list(spec.camera_prim_paths)
        env_partitions = {Sdf.Path(path): f"env_{i}" for i, path in enumerate(self.scene.clone_env_paths)}
        source_root = next((root for root in Sdf.Path(camera_paths[0]).GetPrefixes() if root in env_partitions), None)
        camera_partitions = []
        if source_root is not None:
            # Native cloning may leave only the source camera in USD. Complete paths retain their row order.
            if len(camera_paths) < spec.num_instances:
                suffix = camera_paths[0][len(str(source_root)) :]
                camera_paths = [path + suffix for path in self.scene.clone_env_paths]
            camera_partitions = [
                next(env_partitions[root] for root in Sdf.Path(path).GetPrefixes() if root in env_partitions)
                for path in camera_paths
            ]
        render_data = OVRTXCameraRenderData(
            spec, f"RenderCamera_{self.scene.next_camera_id}", camera_paths, self.cfg.max_envs_per_render_product
        )
        try:
            self._add_render_products(render_data, spec)
            if self._use_ovstage:
                render_data.camera_xform_binding = render_data.resources.enter_context(self.scene.query(camera_paths))
                self.scene.stage.write_attribute(
                    render_data.camera_xform_binding,
                    "omni:resetXformStack",
                    ordinal=self.scene.ordinal,
                    tensors=np.full(spec.num_instances, True, dtype=np.bool_),
                    is_array=False,
                ).wait()
                if camera_partitions:
                    self.scene.stage.write_attribute(
                        render_data.camera_xform_binding,
                        "omni:scenePartition",
                        ordinal=self.scene.ordinal,
                        tensors=np.array(
                            [self.scene.paths.intern_token(partition).raw_id for partition in camera_partitions],
                            dtype=np.uint64,
                        ),
                        is_array=False,
                        semantic=ovstage.AttributeSemantic.TOKEN_ID,
                    ).wait()
            else:
                render_data.camera_xform_binding = self.backend.renderer.bind_attribute(
                    prim_paths=camera_paths,
                    attribute_name="omni:xform",
                    semantic=Semantic.XFORM_MAT4x4,
                    prim_mode=PrimMode.EXISTING_ONLY,
                )
                render_data.resources.callback(render_data.camera_xform_binding.unbind)
                self.backend.renderer.write_attribute(
                    prim_paths=camera_paths,
                    attribute_name="omni:resetXformStack",
                    tensor=np.full(spec.num_instances, True, dtype=np.bool_),
                )
                if camera_partitions:
                    self.backend.renderer.write_attribute(
                        camera_paths,
                        "omni:scenePartition",
                        camera_partitions,
                        semantic=Semantic.TOKEN_STRING,
                    )
            render_data.camera_writes = _AsyncWriteBuffers(
                wp.empty(spec.num_instances, wp.mat44d, device=self._device)
                for _ in range(2 if self.cfg.async_rendering and not self._use_ovstage else 1)
            )
            if not self._use_ovstage:
                for name in _CAMERA_INTRINSIC_ATTRIBUTES:
                    binding = self.backend.renderer.bind_attribute(
                        prim_paths=camera_paths, attribute_name=name, dtype="float32",
                        prim_mode=PrimMode.EXISTING_ONLY, flags=BindingFlag.OPTIMIZE,
                    )  # fmt: skip
                    render_data.resources.callback(binding.unbind)
                    render_data.intrinsic_bindings.append(binding)
        except Exception:
            render_data.cleanup()
            raise
        self.scene.next_camera_id += 1
        self._camera_render_data.append(render_data)
        self.backend.unrendered[render_data] = self
        render_data.resources.callback(self.backend.unrendered.pop, render_data, None)
        return render_data

    def set_outputs(self, render_data: OVRTXCameraRenderData, output_data: dict[str, ProxyArray]) -> None:
        """Register pre-allocated warp output buffers for rendering.

        Each :class:`~isaaclab.utils.warp.ProxyArray` already carries the correct warp
        dtype from :meth:`~isaaclab.sensors.camera.CameraData.allocate`; store
        the underlying warp array directly. ``rgb`` is excluded because it is a
        non-contiguous strided view into ``rgba`` and is updated automatically.

        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.set_outputs`.
        """
        render_data.warp_buffers = {
            name: proxy.warp for name, proxy in output_data.items() if name != str(RenderBufferKind.RGB)
        }

    def prepare_capture(self, render_data: OVRTXCameraRenderData, camera_data: CameraData, frame: ProxyArray) -> None:
        """Snapshot delayed-image metadata without changing live camera state."""
        if self.cfg.async_rendering and not self._use_ovstage:
            render_data.capture = {
                name: ProxyArray(wp.clone(value.warp))
                for name, value in (
                    ("pos_w", camera_data.pos_w),
                    ("quat_w_world", camera_data.quat_w_world),
                    ("intrinsic_matrices", camera_data.intrinsic_matrices),
                    ("frame", frame),
                )
            }

    def read_output(
        self,
        render_data: OVRTXCameraRenderData,
        camera_data: CameraData | None = None,
    ) -> None:
        """Publish the pixels of the render products of this sensor, and copy matching metadata when requested."""
        if self.cfg.async_rendering and not self._use_ovstage and render_data.ready is None:
            return
        capture = {}
        if render_data.ready is not None:
            operation, capture = render_data.ready
            if capture is None:
                render_data.ready = None
                return
            try:
                result = operation.wait().fetch()
            except Exception as error:
                self._note_failed_frame(error)
                raise
            self._process_render_products((render_data,), result)
            if render_data.ready is render_data.pending:
                # Retain the native operation, but do not publish this priming image twice.
                render_data.pending = (operation, None)
            render_data.ready = None
        if camera_data is not None:
            assert camera_data.info is not None, "CameraData.info should be created in CameraData.allocate"
            for output_name in camera_data.info:
                info = render_data.renderer_info.get(output_name)
                camera_data.info[output_name] = {**(info or {}), "capture": capture} if capture else info

    @contextlib.contextmanager
    def _map_render_var_to_dlpack(self, render_var: RenderVarOutput) -> Iterator[wp.array]:
        """Map ``render_var`` for CUDA reads and yield it as a Warp array.

        Wait for rendering on the consuming stream, or on the host on Linux unless
        :data:`_DISABLE_LINUX_CUDA_CPU_SYNC_ENV` is ``1``. Consume the zero-copy view inside the
        context. Unmapping records the consuming stream; native release waits for its queued
        reads and the last view to be dropped.

        Args:
            render_var: OVRTX ``RenderVarOutput`` to map (looked up from ``frame.render_vars``).

        Yields:
            The render var's contents as a Warp array, valid for the duration of the context.
        """
        gpu_side_sync = _gpu_side_render_var_sync_enabled()
        # OVRTX uses 0 for no synchronization and 1 for Torch's legacy default stream (CUDA handle 0).
        stream = self._warp_device.stream.cuda_stream or 1
        mapping = render_var.map(device=Device.CUDA, sync_stream=stream if gpu_side_sync else 0)
        try:
            if not gpu_side_sync:
                mapping.wait()
            yield wp.from_dlpack(mapping)
        finally:
            mapping.unmap(stream=stream)

    def _process_segmentation(
        self, render_data: OVRTXCameraRenderData, frame: FrameOutput, output_buffers: dict[str, wp.array]
    ) -> None:
        """Extract the segmentation pixels of one render product."""
        semantic, instance = "semantic_segmentation", "instance_segmentation"
        if semantic not in output_buffers and instance not in output_buffers:
            return

        for source, key, colorize in (
            (_SEMANTIC_SEGMENTATION_VAR, semantic, self.cfg.colorize_semantic_segmentation),
            (_INSTANCE_SEGMENTATION_VAR, instance, self.cfg.colorize_instance_segmentation),
        ):
            render_var = frame.render_vars.get(render_data.render_var_keys[source])
            if key not in output_buffers or render_var is None:
                continue
            with self._map_render_var_to_dlpack(render_var) as tiled_data:
                if tiled_data.dtype != wp.uint32:
                    continue
                if colorize:
                    color_buffer = self._output_id_color_buffers.get(key)
                    if color_buffer is None or color_buffer.size < tiled_data.size:
                        color_buffer = wp.zeros(tiled_data.size, dtype=wp.uint32, device=self._device)
                        self._output_id_color_buffers[key] = color_buffer
                    color_buffer = color_buffer[: tiled_data.size].reshape(tiled_data.shape)
                    wp.launch(
                        generate_random_colors_from_ids_kernel,
                        tiled_data.shape,
                        inputs=[tiled_data, color_buffer],
                        device=self._device,
                    )
                    colors = wp.to_torch(color_buffer).view(torch.uint8).reshape(*tiled_data.shape[:2], 4)
                    self._launch_extract_all_tiles(
                        render_data, wp.from_torch(colors, dtype=wp.uint8), output_buffers[key]
                    )
                else:
                    # Keep uint32 in Warp; torch-to-Warp conversion does not support that dtype.
                    if tiled_data.ndim == 2:
                        tiled_data = tiled_data.reshape((*tiled_data.shape, 1))
                    self._launch_extract_all_tiles(render_data, tiled_data, output_buffers[key])

    def _prepare_semantic_palette(
        self, id_to_labels: dict[RgbaColor, dict[str, str]]
    ) -> dict[RgbaColor, dict[str, str]]:
        """Prepare the pixel replacements and matching metadata for the configured class palette."""
        palette = self.cfg.semantic_segmentation_mapping
        mapped = {}
        replacements = []
        for color, semantics in id_to_labels.items():
            target = next(
                (
                    tuple(palette[f"{kind}:{label}"])
                    for kind, labels in semantics.items()
                    for label in labels.split(",")
                    if f"{kind}:{label}" in palette
                ),
                color,
            )
            mapped[target] = semantics
            if target != color:
                replacements.append((color, target))
        key = tuple(replacements)
        if key:
            if key != self._semantic_palette_key:
                source = np.asarray([color for color, _ in key], dtype=np.uint8).view(np.uint32).reshape(-1)
                target = np.asarray([color for _, color in key], dtype=np.uint8)
                self._semantic_palette_buffers = (
                    wp.array(source, dtype=wp.uint32, device=self._device),
                    wp.array(target, dtype=wp.uint8, device=self._device),
                )
        self._semantic_palette_key = key
        return mapped

    def _process_segmentation_metadata(
        self, render_data: OVRTXCameraRenderData, frames: Sequence[FrameOutput], output_buffers: dict[str, wp.array]
    ) -> None:
        """Decode the label maps of the sensor into ``render_data.renderer_info``, once per step.

        A label map lists the labels of the whole scene, not of one camera. So the method uses the maps of
        the first product for every product of the sensor. It checks that the other products return the same
        maps, because a wrong label for an ID would go unnoticed in training.

        Raises:
            RuntimeError: If ``instance_segmentation`` is requested and a map is missing from a frame, or if
                two render products of the sensor return different label maps, or only one of them returns a map.
        """
        semantic, instance = "semantic_segmentation", "instance_segmentation"
        if semantic not in output_buffers and instance not in output_buffers:
            return
        if instance in output_buffers:
            for frame in frames:
                missing = [
                    key
                    for key in _INSTANCE_SEGMENTATION_MAP_VARS
                    if frame.render_vars.get(render_data.render_var_keys[key]) is None
                ]
                if missing:
                    raise RuntimeError(
                        f"instance_segmentation was requested but the following render vars are missing from the "
                        f"OVRTX frame: {missing}. Available vars: {list(frame.render_vars)}"
                    )
        frame = frames[0]
        maps = {key: frame.render_vars.get(render_data.render_var_keys[key]) for key in _INSTANCE_SEGMENTATION_MAP_VARS}
        first_tensors: dict[str, dict[str, np.ndarray]] = {}

        def decode(key: str, decoder: Callable[[IdentifierMap], Any]) -> Any:
            if len(frames) == 1:
                return _decode_segmentation_map(maps[key], decoder)
            first_tensors[key], decoded = _decode_segmentation_map(
                maps[key], lambda id_map: (_copy_label_map(id_map), decoder(id_map))
            )
            return decoded

        labels = None
        if maps[_SEMANTIC_ID_MAP_VAR] is not None:
            labels = decode(_SEMANTIC_ID_MAP_VAR, decode_semantic_id_map)
        if semantic in output_buffers and labels is not None:
            id_to_labels = build_semantic_id_to_labels(
                labels, colorize=self.cfg.colorize_semantic_segmentation, device=self._device
            )
            if self.cfg.colorize_semantic_segmentation and self.cfg.semantic_segmentation_mapping:
                id_to_labels = self._prepare_semantic_palette(id_to_labels)
            render_data.renderer_info[semantic] = {"idToLabels": id_to_labels}
        if instance in output_buffers:
            instances = decode(_STABLE_ID_SEMANTIC_ID_MAP_VAR, decode_stable_id_semantic_id_map)
            paths = decode(_STABLE_ID_MAP_VAR, decode_stable_id_map)
            id_to_labels, id_to_semantics = build_instance_id_to_labels_and_semantics(
                instances, paths, labels, colorize=self.cfg.colorize_instance_segmentation, device=self._device
            )
            render_data.renderer_info[instance] = {"idToLabels": id_to_labels, "idToSemantics": id_to_semantics}
        if len(frames) == 1:
            return
        used = [(_SEMANTIC_ID_MAP_VAR, decode_semantic_id_map, labels)]
        if instance in output_buffers:
            used += [
                (_STABLE_ID_SEMANTIC_ID_MAP_VAR, decode_stable_id_semantic_id_map, instances),
                (_STABLE_ID_MAP_VAR, decode_stable_id_map, paths),
            ]
        for key, decoder, expected in used:
            first = first_tensors.get(key)
            for product, other in zip(render_data.products[1:], frames[1:], strict=True):
                render_var = other.render_vars.get(render_data.render_var_keys[key])
                if first is None or render_var is None:
                    # A map that only some render products return is a mismatch too.
                    same = first is None and render_var is None
                else:
                    same = _decode_segmentation_map(
                        render_var, lambda id_map: _label_map_matches(id_map, first, decoder, expected)
                    )
                if not same:
                    raise RuntimeError(
                        f"OVRTX returned a different {key} for render product {product.path!r} than for"
                        f" {render_data.products[0].path!r}. The segmentation labels of a split camera sensor"
                        " assume one map for the whole scene. Set OVRTXRendererCfg.max_envs_per_render_product"
                        ' to "all" to render this sensor through one render product, if its buffers stay below'
                        " the renderer limit."
                    )

    def _launch_extract_all_tiles(
        self, render_data: OVRTXCameraRenderData, tiled_buffer: wp.array, output_buffer: wp.array
    ) -> None:
        """Launch ``extract_all_tiles_kernel`` for one tiled/output buffer pair.

        This is the only place that should launch ``extract_all_tiles_kernel``: it validates that
        ``output_buffer`` cannot read past the end of ``tiled_buffer`` (the kernel derives its per-thread
        channel loop bound from ``output_buffer``'s last dimension) before every launch, so callers cannot
        accidentally skip the check.

        Args:
            render_data: OVRTX render data for the current frame.
            tiled_buffer: 3D array of shape (H, W, C) holding all tiles packed into one buffer.
            output_buffer: 4D array of shape (N, H, W, C) to receive the tiles of the N environments in
                ``tiled_buffer``, with C no greater than ``tiled_buffer``'s channel count.

        Raises:
            ValueError: If ``output_buffer``'s channel count exceeds ``tiled_buffer``'s.
        """
        tiled_channels = tiled_buffer.shape[-1]
        output_channels = output_buffer.shape[-1]
        if output_channels > tiled_channels:
            raise ValueError(
                f"Output buffer has {output_channels} channels but the tiled buffer only has {tiled_channels};"
                " extract_all_tiles_kernel would read out of bounds."
            )

        wp.launch(
            kernel=extract_all_tiles_kernel,
            dim=(output_buffer.shape[0], render_data.height, render_data.width),
            inputs=[
                tiled_buffer,
                output_buffer,
                _tile_columns(output_buffer.shape[0]),
                render_data.width,
                render_data.height,
            ],
            device=self._device,
        )

    def _process_render_frames(
        self, render_data: OVRTXCameraRenderData, frames: Sequence[FrameOutput], output_buffers: dict[str, wp.array]
    ) -> None:
        """Extract the frame of each render product into the environment range of that product.

        It checks and decodes the label maps of the sensor first. So a missing map raises before the method writes
        any output.
        """
        # Reset per-output metadata so it is a snapshot of this frame only. Unlike pixel AOVs (always
        # present), metadata like the semantic ``idToLabels`` is only repopulated below when its render var
        # is available, so without this a missing SemanticIdMap on a later frame would leave a stale mapping.
        render_data.renderer_info.clear()
        self._process_segmentation_metadata(render_data, frames, output_buffers)
        for product, frame in zip(render_data.products, frames, strict=True):
            buffers = output_buffers
            if product.num_envs != render_data.num_envs:
                stop = product.env_offset + product.num_envs
                buffers = {key: buffer[product.env_offset : stop] for key, buffer in output_buffers.items()}
            self._process_render_frame(render_data, frame, buffers)
        if (
            self.cfg.colorize_semantic_segmentation
            and self.cfg.semantic_segmentation_mapping
            and "semantic_segmentation" in render_data.renderer_info
            and self._semantic_palette_key
        ):
            image = output_buffers["semantic_segmentation"]
            wp.launch(
                remap_semantic_colors_kernel,
                dim=image.shape[:3],
                inputs=[image, *self._semantic_palette_buffers],
                device=self._device,
            )

    def _process_render_frame(
        self, render_data: OVRTXCameraRenderData, frame: FrameOutput, output_buffers: dict[str, wp.array]
    ) -> None:
        """Extract RGB, depth, albedo, and semantic from a single render frame into output_buffers."""
        ldr_color = frame.render_vars.get(render_data.render_var_keys[_LDR_COLOR_VAR])
        if ldr_color is not None:
            buffer_key = None

            if "rgba" in output_buffers:
                buffer_key = "rgba"
            else:
                # The output buffers must contain only one simple shading data type at most after resolution of the data
                # types during creation of the output buffers (OVRTXCameraRenderData._create_warp_buffers).
                for dt in _RTX_MINIMAL_MODES:
                    if dt in output_buffers:
                        buffer_key = dt
                        break

            if buffer_key is not None:
                with self._map_render_var_to_dlpack(ldr_color) as tiled_data:
                    self._launch_extract_all_tiles(render_data, tiled_data, output_buffers[buffer_key])

        for depth_var, buffer_keys in _DEPTH_VAR_BUFFER_KEYS.items():
            depth_render_var = frame.render_vars.get(render_data.render_var_keys[depth_var])
            if depth_render_var is None:
                continue
            if not any(buffer_key in output_buffers for buffer_key in buffer_keys):
                continue
            with self._map_render_var_to_dlpack(depth_render_var) as tiled_depth_data:
                if tiled_depth_data.dtype == wp.uint32:
                    tiled_depth_data = wp.from_torch(
                        wp.to_torch(tiled_depth_data).view(torch.float32), dtype=wp.float32
                    )
                for depth_type in buffer_keys:
                    if depth_type in output_buffers:
                        self._launch_extract_all_tiles(render_data, tiled_depth_data, output_buffers[depth_type])

        albedo_var = frame.render_vars.get(render_data.render_var_keys[_ALBEDO_VAR])
        if albedo_var is not None and "albedo" in output_buffers:
            with self._map_render_var_to_dlpack(albedo_var) as tiled_albedo_data:
                self._launch_extract_all_tiles(render_data, tiled_albedo_data, output_buffers["albedo"])

        hdr_color = frame.render_vars.get(render_data.render_var_keys[_HDR_COLOR_VAR])
        if hdr_color is not None and "rgb_hdr" in output_buffers:
            with self._map_render_var_to_dlpack(hdr_color) as tiled_hdr_data:
                # OVRTX can fall back from deviceIds to automatic assignment; HDR consumers need the output device.
                output_device = str(output_buffers["rgb_hdr"].device)
                if str(tiled_hdr_data.device) != output_device:
                    tiled_hdr_data = wp.clone(tiled_hdr_data, device=output_device)
                if tiled_hdr_data.dtype not in (wp.float16, wp.float32):
                    raise TypeError(f"Unsupported OVRTX HdrColor dtype: {tiled_hdr_data.dtype}.")
                self._launch_extract_all_tiles(render_data, tiled_hdr_data, output_buffers["rgb_hdr"])

        self._process_segmentation(render_data, frame, output_buffers)

        normals_var = frame.render_vars.get(render_data.render_var_keys[_NORMALS_VAR])
        if normals_var is not None and "normals" in output_buffers:
            with self._map_render_var_to_dlpack(normals_var) as tiled_normals_data:
                self._launch_extract_all_tiles(render_data, tiled_normals_data, output_buffers["normals"])

        # For motion vectors, extract only the first two (u, v) channels from the tiled buffer.
        # Note: mirrors the Isaac RTX renderer's handling of the "TargetMotionSD" AOV
        # (check: https://github.com/isaac-sim/IsaacLab/issues/2003).
        motion_var = frame.render_vars.get(render_data.render_var_keys[_MOTION_VECTORS_VAR])
        if motion_var is not None and "motion_vectors" in output_buffers:
            with self._map_render_var_to_dlpack(motion_var) as tiled_motion_vectors_data:
                self._launch_extract_all_tiles(render_data, tiled_motion_vectors_data, output_buffers["motion_vectors"])

    def drain_pending_renders(self, cameras: Sequence[OVRTXCameraRenderData] | None = None) -> list[Exception]:
        """Complete native work without publishing observations into any camera."""
        cameras = self._camera_render_data if cameras is None else cameras
        operations = {frame[0] for data in cameras for frame in (data.pending, data.ready) if frame is not None}
        errors = []
        for operation in operations:
            try:
                operation.wait().fetch()
            except Exception as error:
                self._note_failed_frame(error)
                errors.append(error)
        return errors

    def _note_failed_frame(self, error: Exception, num_smaller_layouts: int = 0) -> None:
        """Add a note about render product splits to an error about a failed frame.

        The renderer error about a failed frame does not name the renderer setting that can fix a tiled atlas above
        a buffer limit. The note names it and lists the camera sensors of this renderer with several environments in
        one render product. If the split search gave up, the same note gives the number of smaller layouts that it
        tried. Other errors and errors that already have the note do not change.

        Args:
            error: Error of the failed frame.
            num_smaller_layouts: Number of smaller layouts that the split search tried before it gave up.
        """
        notes = getattr(error, "__notes__", ())
        if _FAILED_FRAME_TEXT not in str(error) or any(note.startswith(_FAILED_FRAME_NOTE) for note in notes):
            return
        # The first render product of a sensor has the most environments.
        cameras = [
            f"{data.camera_paths[0]!r} ({data.products[0].num_envs} environments)"
            for data in self._camera_render_data
            if data.products[0].num_envs > 1
        ]
        note = _FAILED_FRAME_NOTE
        if num_smaller_layouts:
            note += (
                f" The first frame also failed with {num_smaller_layouts} smaller render product layout(s)"
                " of this search. Smaller render products do not fix a lack of GPU memory. The renderer log shows"
                " the cause."
            )
        if cameras:
            note += f" Camera sensors with several environments in one render product: {', '.join(cameras)}."
        error.add_note(note)

    def reset(self, render_data: OVRTXCameraRenderData, env_ids: Sequence[int] | None = None) -> None:
        """Retire pre-reset frames. The next capture primes every tiled render product of this sensor.

        Pending captures cover every environment tile, so they are not separable per environment
        and ``env_ids`` is not consulted.
        """
        del env_ids
        errors = self.drain_pending_renders((render_data,))
        render_data.pending = render_data.ready = None
        render_data.capture = {}
        if errors:
            raise ExceptionGroup("OVRTX renders failed during camera reset", errors)

    def _process_render_products(
        self, render_data: Sequence[OVRTXCameraRenderData], products: RenderProductSetOutputs
    ) -> None:
        """Populate camera outputs only after every requested product returned a frame."""
        for data in render_data:
            for path in data.render_product_paths:
                if path not in products or not products[path].frames:
                    raise RuntimeError(f"OVRTX returned no frame for render product {path!r}.")
        for data in render_data:
            frames = []
            for path in data.render_product_paths:
                # DLSS-generated images have earlier capture times than the real frame. Keep all
                # camera outputs on the latest capture without assuming the returned list's order.
                captures = products[path].frames
                frames.append(captures[0] if len(captures) == 1 else max(captures, key=lambda frame: frame.end_time))
            self._process_render_frames(data, frames, data.warp_buffers)

    def set_environment_visibility(self, visible: Sequence[bool]) -> None:
        """Show or hide whole cloned environments' bodies and markers in every camera that renders the scene.

        Writing the USD ``visibility`` of a cloned environment corrupts the transforms of its bodies once it is shown
        again, so hidden bodies are instead moved far below the scene when their poses are published.

        .. todo::
            Hiding by moving bodies below the scene is a workaround. Find why writing ``visibility`` on a cloned
            environment corrupts its body transforms after it is shown again, and hide with ``visibility`` instead.

        Args:
            visible: Per-environment flag, in environment order.
        """
        num_envs = len(self._env_paths)
        if len(visible) != num_envs:
            raise ValueError(f"Expected {num_envs} visibility flags, got {len(visible)}.")
        self._visible_envs = np.asarray(visible, dtype=bool)
        self._hidden_marker_rows = None
        hidden = np.zeros(num_envs + 1, dtype=np.uint8)  # The last entry stands for bodies outside any environment.
        hidden[:num_envs] = ~self._visible_envs
        self._hidden_envs = wp.array(hidden, dtype=wp.uint8, device=self._device) if hidden.any() else None
        self._transforms_timestamp = -1  # Publish the poses again even when the simulation has not stepped.

    def _visible_marker_rows(self, marker: Any) -> tuple:
        """Return the marker's translations, orientations, scales, and indices with hidden markers moved away.

        A batch whose size divides by the environment count is laid out environment-major; other batches are global.
        The batch keeps its size, so the marker instances are never rebuilt when the visible environments change;
        hidden environments' markers are moved far below the scene, as their bodies are.
        """
        tensors = (marker.translations, marker.orientations, marker.scales, marker.marker_indices)
        mask, num_envs, count = self._visible_envs, len(self._env_paths), len(marker.translations)
        if mask is None or mask.all() or num_envs == 0 or count % num_envs:
            return tensors
        hidden = self._hidden_marker_rows
        if hidden is None or hidden.device != marker.translations.device:
            hidden = self._hidden_marker_rows = torch.as_tensor(~mask, device=marker.translations.device)
        translations = marker.translations.reshape(num_envs, count // num_envs, -1).clone()
        translations[hidden, :, 2] -= float(_HIDDEN_DROP)
        return (translations.reshape(marker.translations.shape), *tensors[1:])

    def _body_environment_ids(self) -> wp.array:
        """Return each published body's environment index, or ``num_envs`` for a body outside any environment."""
        paths = self._sdp.backend.transform_paths
        if self._body_envs is None or len(self._body_envs) != len(paths):
            roots = {root: i for i, root in enumerate(self._env_paths)}
            outside = len(self._env_paths)
            env_ids = []
            for path in paths:
                env_id, end = outside, path.rfind("/")
                while end > 0:  # Walk up the prim path until it reaches an environment root.
                    env_id = roots.get(path[:end], outside)
                    if env_id != outside:
                        break
                    end = path.rfind("/", 0, end)
                env_ids.append(env_id)
            self._body_envs = wp.array(env_ids, dtype=wp.int32, device=self._device)
        return self._body_envs

    def _hide_environment_bodies(self, matrices: wp.array, stream: wp.Stream) -> wp.array:
        """Return the body poses with those of hidden environments moved out of view."""
        if self._hidden_envs is None:
            return matrices
        body_envs = self._body_environment_ids()
        # A published buffer may be shared with other consumers, so hidden poses are written to a copy. The copy is
        # kept per source buffer: an asynchronous write still reads it until the source buffer is acquired again.
        hidden = self._hidden_matrices.get(matrices.ptr)
        if hidden is None or len(hidden) != len(matrices):
            hidden = self._hidden_matrices[matrices.ptr] = wp.empty(len(matrices), dtype=wp.mat44d, device=self._device)
        wp.launch(
            hide_environment_bodies_kernel,
            len(matrices),
            [matrices, body_envs, self._hidden_envs, _HIDDEN_DROP, hidden],
            device=self._device,
            stream=stream,
        )
        return hidden

    def update_transforms(self) -> None:
        """Write changed SDP transforms to OVRTX."""
        binding = self._object_xform_binding
        if binding is None:
            return
        timestamp = self._sdp.backend.transforms_timestamp
        if self._transforms_timestamp == timestamp:
            return
        stream = self._warp_device.stream
        asynchronous = self.cfg.async_rendering and not self._use_ovstage
        output = self._transform_writes.acquire(stream)
        if not self._sdp.get_transforms(output, allow_passthrough=not asynchronous, scales=self._object_scales):
            return
        matrices = self._hide_environment_bodies(output.matrices, stream)
        if self._use_ovstage:
            # The write waits for consumption; the producing CUDA stream orders access to the
            # borrowed buffer.
            self.scene.stage.write_attribute(
                binding,
                "omni:xform",
                ordinal=self.scene.ordinal,
                tensors=xform_tensor_from_warp(matrices),
                is_array=False,
                semantic=ovstage.AttributeSemantic.MATRIX,
                cuda_stream=stream.cuda_stream or 1,
            ).wait()
        elif asynchronous:
            self._transform_writes.submit(binding, matrices, stream)
        else:
            binding.write(matrices, data_access=DataAccess.ASYNC, cuda_stream=stream.cuda_stream or 1)
        self._transforms_timestamp = timestamp

    def update_geometries(self) -> None:
        """Write changed SDP geometry to OVRTX."""
        for callback in self.geometry_callbacks.values():
            callback(self)
        for marker in tuple(self.markers):
            if marker not in self.marker_groups:
                self.markers.pop(marker).close()
        for marker in self.marker_groups:
            if marker.translations is None:
                continue
            if marker not in self.markers:
                self.markers[marker] = OVRTXMarkers(self, marker.cfg)
            self.markers[marker].update(
                *self._visible_marker_rows(marker),
                visible=marker.is_visible(),
            )  # fmt: skip
        binding = self._geometry_points_binding
        if binding is None:
            return
        stream = self._warp_device.stream
        asynchronous = self.cfg.async_rendering and not self._use_ovstage
        if asynchronous:
            output = self._geometry_writes.acquire(stream)
            self._sdp.get_geometry_points(output=output, offsets=self._geometry_offsets)
            offsets = list(self._geometry_offsets.values())
            points = [output[start:end] for start, end in zip(offsets, [*offsets[1:], len(output)], strict=True)]
        else:
            points = self._sdp.get_geometry_points()
            points = [points[path] for path in self._geometry_paths]
        timestamp = self._sdp.backend.geometry_timestamp
        if self._geometry_timestamp == timestamp:
            return
        if self._use_ovstage:
            self.scene.stage.write_attribute(
                binding,
                "points",
                ordinal=self.scene.ordinal,
                tensors=[points_tensor_from_warp(array) for array in points],
                is_array=True,
                semantic=ovstage.AttributeSemantic.POINT,
                cuda_stream=stream.cuda_stream or 1,
            ).wait()
        elif asynchronous:
            self._geometry_writes.submit(binding, points, stream)
        else:
            binding.write(points, data_access=DataAccess.ASYNC, cuda_stream=stream.cuda_stream or 1)
        self._geometry_timestamp = timestamp

    def update_camera(
        self,
        render_data: OVRTXCameraRenderData,
        positions: ProxyArray,
        orientations: ProxyArray,
        intrinsics: ProxyArray,
    ) -> None:
        """Update camera poses using the camera's reusable conversion buffers."""
        binding = render_data.camera_xform_binding
        if binding is None:
            return
        stream = self._warp_device.stream
        matrices = render_data.camera_writes.acquire(stream)
        wp.launch(
            create_camera_transforms_kernel, len(matrices), [positions, orientations, matrices], device=self._device
        )
        if self._use_ovstage:
            self.scene.stage.write_attribute(
                binding,
                "omni:xform",
                ordinal=self.scene.ordinal,
                tensors=xform_tensor_from_warp(matrices),
                is_array=False,
                semantic=ovstage.AttributeSemantic.MATRIX,
                cuda_stream=stream.cuda_stream or 1,
            ).wait()
        else:
            operation = render_data.camera_writes.submit(binding, matrices, stream)
            if not self.cfg.async_rendering:
                operation.wait()

    def update_camera_intrinsics(self, render_data: OVRTXCameraRenderData, intrinsics: wp.array, parameters: wp.array):
        """Publish calibration columns from GPU memory into the renderer-owned scene."""
        # A runtime calibration change is a scene write for this camera's bindings, so its
        # in-flight renders must finish first. Other cameras keep their pipelines.
        errors = self.drain_pending_renders((render_data,))
        if errors:
            raise ExceptionGroup("OVRTX renders failed before calibration update", errors)
        stream = wp.get_stream(parameters.device).cuda_stream or 1
        if self._use_ovstage:
            self.scene.stage.write_attributes(
                render_data.camera_xform_binding,
                [
                    ovstage.WriteDesc(attribute=name, tensors=parameters[row], is_array=False, cuda_stream=stream)
                    for row, name in enumerate(_CAMERA_INTRINSIC_ATTRIBUTES)
                ],
                ordinal=self.scene.ordinal,
            ).wait()
        else:
            operations = []
            try:
                for row, binding in enumerate(render_data.intrinsic_bindings):
                    operations.append(
                        binding.write_async(parameters[row], data_access=DataAccess.ASYNC, cuda_stream=stream)
                    )
            finally:
                for operation in operations:
                    operation.wait()

    def render(self, render_data: OVRTXCameraRenderData) -> None:
        """Submit one camera capture; :meth:`read_output` publishes the available image."""
        self.render_batch((render_data,))

    def log_mesh(self, name: str, *args, **kwargs) -> None:
        """Publish a runtime mesh through the same scene used by every camera."""
        if name not in self.meshes:
            self.meshes[name] = OVRTXMesh(self, f"/IsaacLabMeshes/{name.lstrip('/')}")
        self.meshes[name].update(*args, **kwargs)

    def render_batch(self, render_data: Sequence[OVRTXCameraRenderData]) -> None:
        """Render all requested camera products in one native submission.

        Every registered render product is submitted, not just the requested ones: OVRTX
        batches products within a step, and naming only a subset makes each product that
        re-enters the set on a later step far more expensive. Only ``render_data`` is read
        back, so products left out of the request keep their previous outputs.

        If the first frame fails, the split search halves the environments per render product of a camera sensor
        whose renderer has ``max_envs_per_render_product = "auto"`` and renders the frame again. A warning then gives
        the value that skips the search. See :func:`_plan_split` and :meth:`_apply_split_plan`.

        Args:
            render_data: Cameras whose poses and output buffers have been prepared. An empty
                sequence performs no work.

        Raises:
            RuntimeError: If the scene is uninitialized, if a requested product returns no frame, or if the
                first frame fails and the search gives up. A camera whose search failed raises in every later call.
                The cause is the failed frame or the failed scene write.
            Exception: If a scene write of the search fails.
        """
        if not render_data:
            return
        if not self._initialized_scene:
            raise RuntimeError("Scene not initialized. Call initialize() first.")
        failed = next((data for data in render_data if data.first_frame_error is not None), None)
        if failed is not None:
            raise RuntimeError(
                f"Camera {failed.spec.cfg.prim_path} renders no frames. Its first frame failed, and the search of the"
                " OVRTX renderer for a render product split (OVRTXRendererCfg.max_envs_per_render_product) ended"
                " without a layout that renders. The cause gives the reason."
            ) from failed.first_frame_error
        if self.backend.renderer is None or not self._camera_render_data:
            return
        # Each pass renders once. After a failed first frame, the plan moves the sensors without a rendered frame to
        # new render products, which the next pass renders. Each plan halves at most one sensor. After
        # _MAX_SMALLER_LAYOUTS smaller layouts in this render batch, the plan halves none, so the loop ends with a
        # frame or an error.
        for num_smaller_layouts in itertools.count():
            try:
                self._render_batch_once(render_data)
                break
            except RuntimeError as error:
                plan = _plan_split(list(self.backend.unrendered), error, num_smaller_layouts)
                # A plan without a halved sensor still moves the other sensors, so that no step renders a failed
                # render product again. The loop then raises the error.
                self._apply_split_plan(plan, error)
                if plan.halved is None:
                    self._note_failed_frame(error, num_smaller_layouts)
                    raise
        for data in self.backend.unrendered:
            if data.searches_split and len(data.products) > 1:
                logger.warning(
                    "Camera %s: the first frame failed with one render product. It now uses %d render products."
                    " Set OVRTXRendererCfg.max_envs_per_render_product=%d on every OVRTX renderer configuration of"
                    " the task to skip this search.",
                    data.spec.cfg.prim_path,
                    len(data.products),
                    data.products[0].num_envs,  # The first render product has the most environments.
                )
        self.backend.unrendered.clear()

    def _apply_split_plan(self, plan: _SplitPlan, error: RuntimeError) -> None:
        """Move the camera sensors of ``plan`` to new render products, or end their search.

        A failed step fails for every sensor on the engine, and OVRTX can abort when it renders a failed render
        product again. So each sensor of the plan gets a new scope, and its old render products leave the steps. The
        new render products get the attributes of the first old one, see :meth:`_copy_render_product_attributes`.
        A sensor that the search gives up on renders no frames.

        A failed scene write also ends the search. The sensor of that write and the sensors that the plan did not
        move yet then render no frames. Sensors that the plan moved before that write keep their new render products.

        Args:
            plan: Next layouts from :func:`_plan_split`. An empty plan changes nothing.
            error: Error of the failed step.

        Raises:
            Exception: If a scene write fails.
        """
        cameras = self.backend.unrendered
        # First remove every failed render product from the set that each step renders. If a scene
        # write below fails, the set then holds no failed render product.
        for data, _ in plan.max_envs_per_render_product:
            self.backend.render_products.difference_update(data.render_product_paths)
        for index, (data, max_envs) in enumerate(plan.max_envs_per_render_product):
            renderer = cameras.get(data)
            if renderer is None:
                # A cleanup during this search removed the sensor. Its render products already left the steps.
                continue
            if max_envs is None:
                # The search gave up on this sensor. Another step cannot render it, so it leaves the steps
                # for good, and a later render of it raises at once. Its old scope stays until the cleanup.
                data.first_frame_error = error
                cameras.pop(data)
                continue
            failed = data.products[0]
            try:
                data._set_layout(f"RenderCamera_{renderer.scene.next_camera_id}", max_envs)
                renderer.scene.next_camera_id += 1
                renderer._add_render_products(data, data.spec)
                renderer._copy_render_product_attributes(failed.path, data.render_product_paths)
            except Exception as write_error:
                # This sensor and the ones after it leave the steps for good. Their next renders then name this error,
                # not a missing frame. A failed copy can leave new render products of this sensor in the steps.
                for rest, _ in plan.max_envs_per_render_product[index:]:
                    rest.first_frame_error = write_error
                    cameras.pop(rest, None)
                    self.backend.render_products.difference_update(rest.render_product_paths)
                raise

    def _copy_render_product_attributes(self, source: str, targets: Sequence[str]) -> None:
        """Copy the attributes of the render product ``source`` to the render products ``targets``.

        A script can write render settings directly onto a render product, for example the samples per pixel after
        the reset. The split search builds new render products from the camera configuration, so without this copy,
        they lose such settings. The attributes in ``_PER_PRODUCT_ATTRIBUTES`` keep the values of the new layout,
        because each render product has its own cameras, render variables, resolution and path.

        The copy also writes the attributes that nobody changed. The runtime reports no difference between an
        authored value and a schema default, so the copy cannot leave these out.
        """
        if self._use_ovstage:
            stage, paths = self.scene.stage, self.scene.paths
            writes = []
            # The prims of the failed layout and their settings are sealed, because the failed step committed them.
            everything = ovstage.OrdinalRange.latest(ovstage.ORDINAL_MAX)
            with (
                self.scene.query([source]) as query,
                stage.read_attributes(query, [paths.intern_token("*")], everything) as read,
            ):
                for group in read.groups():
                    with group:
                        for column in group.columns:
                            if paths.token_to_string(column.attribute) in _PER_PRODUCT_ATTRIBUTES:
                                continue
                            # Each host copy holds the data row of the source prim. DLPack keeps a half as float16.
                            row = column.data_row_index(0)
                            if column.is_array:
                                # A read gives the components of each element as an axis, but an array write takes
                                # them only as lanes. Without the conversion, a float3[] arrives as a float[].
                                value = np.array(np.from_dlpack(column.dlpack(row)))
                                tensors = [ovstage.typed_tensor_to_write(ovstage.make_dltensor(value))]
                            else:
                                tensors = np.array(np.from_dlpack(column.dlpack(0))[row : row + 1])
                            writes.append(
                                ovstage.WriteDesc(
                                    column.attribute,
                                    tensors,
                                    column.is_array,
                                    index_map=[0] * len(targets),
                                    semantic=column.semantic,
                                )
                            )
            # A write that adds attributes to the targets can change the borrowed data of an open read. So the copy
            # writes the host copies after the release.
            if writes:
                with self.scene.query(targets) as query:
                    stage.write_attributes(query, writes, ordinal=self.scene.ordinal).wait()
        else:
            renderer = self.backend.renderer
            render_products = renderer.query_prims(
                require_all=[(FilterKind.PRIM_TYPE, "RenderProduct")], attribute_filter_mode=AttributeFilterMode.ALL
            )
            for name, info in render_products[source].items():
                # A tag is the name of an applied schema and has no value.
                if name in _PER_PRODUCT_ATTRIBUTES or info.semantic == Semantic.TAG:
                    continue
                if info.is_array:
                    try:
                        (value,) = renderer.read_array_attribute(name, [source]).values()
                    except RuntimeError:
                        # OVRTX raises for an array without elements.
                        # The new render products then keep their own value.
                        continue
                    renderer.write_array_attribute(list(targets), name, [value] * len(targets), semantic=info.semantic)
                else:
                    value = np.from_dlpack(renderer.read_attribute(name, [source]))
                    renderer.write_attribute(
                        list(targets), name, np.repeat(value, len(targets), axis=0), semantic=info.semantic
                    )

    def _render_batch_once(self, render_data: Sequence[OVRTXCameraRenderData]) -> None:
        """Render one step with every registered render product, and read back the images of ``render_data``."""
        products = self.backend.render_products.copy()
        material_writer = self._visual_material_writer_ref() if self._visual_material_writer_ref is not None else None
        try:
            if material_writer is not None:
                material_writer.publish()
            if self._needs_transform_initialization:
                # OVRTX 0.6 initializes GPU world transforms on its first step using
                # authored poses. Flush that initialization before capturing runtime
                # camera poses, without advancing the sensor simulation clock.
                self.backend.renderer.step(render_products=products, delta_time=0.0)
                self._needs_transform_initialization = False
            if self._use_ovstage:
                ordinal = self.scene.commit()
            elif self.cfg.async_rendering:
                unread = {data.ready[0] for data in render_data if data.ready is not None}
                operation = self.backend.renderer.step_async(render_products=products, delta_time=_RENDER_DELTA_TIME)
                if self.backend.unrendered:
                    # The wait happens only until every sensor on the engine has a rendered frame.
                    # A failed frame then raises here, before a camera stores the failed operation.
                    # Only render_batch can replace the failed render products, so the error must raise in it.
                    operation.wait()
                for data in render_data:
                    previous = data.pending
                    data.pending = (operation, data.capture)
                    data.ready = previous if previous is not None else data.pending
                # Repeated submissions may replace unread results, but must still complete them.
                for previous in unread:
                    previous.wait().fetch()
            else:
                result = self.backend.renderer.step(render_products=products, delta_time=_RENDER_DELTA_TIME)
                self._process_render_products(render_data, result)
        finally:
            if material_writer is not None:
                # When another exception is already propagating, log the drain failure instead of
                # replacing it. Losing it silently would hide a failed material write.
                primary_active = sys.exc_info()[0] is not None
                try:
                    material_writer.drain()
                except Exception as e:
                    if not primary_active:
                        raise
                    logger.warning("Error draining material writes after a failed render: %s", e, exc_info=True)

        if self._use_ovstage:
            # Stage writes and material reads finish before rendering. Consume only after advancing
            # the ordinal so a failed readback cannot leave later writes at the sealed floor.
            result = self.backend.renderer.step(
                render_products=products, delta_time=_RENDER_DELTA_TIME, ordinal=ordinal
            )
            self._process_render_products(render_data, result)

    def cleanup(self, render_data: OVRTXCameraRenderData | None) -> None:
        """Release the render data's buffers. See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.cleanup`.

        Each camera owns its render products and pose binding. Scene and physics bindings remain alive
        until :meth:`close`, so other cameras can continue rendering.
        """
        if render_data is None:
            return
        try:
            self.reset(render_data)
        finally:
            try:
                render_data.cleanup()
            finally:
                if render_data in self._camera_render_data:
                    self._camera_render_data.remove(render_data)

    def _add_render_products(self, render_data: OVRTXCameraRenderData, spec: CameraRenderSpec) -> None:
        """Add the render products of ``render_data`` to the scene, and render them in every later step."""
        scope = render_data.render_scope_name
        usd = build_render_product_as_string(
            spec, render_data, device_id=self._warp_device.ordinal, enable_shadows=self.cfg.enable_shadows
        )
        # Registration mutates the shared scene; finish renders using its existing products first.
        errors = self.drain_pending_renders()
        if errors:
            raise ExceptionGroup("OVRTX renders failed before render product registration", errors)
        if self.cfg.temp_usd_dir is not None:
            _write_combined_stage(Path(self.cfg.temp_usd_dir), usd)
        if self._use_ovstage:
            reference = ovstage.population.add_usd_reference_from_string(self.scene.stage, usd, f"/{scope}")
            render_data.resources.callback(self._remove_usd_reference, reference)
            ovstage.population.apply_usd_changes(self.scene.stage, ordinal=self.scene.ordinal)
            # USD references drop external camera targets; author the relationship in Fabric.
            for product in render_data.products:
                with self.scene.query([product.path]) as query:
                    self.scene.stage.write_attribute(
                        query,
                        "camera",
                        ordinal=self.scene.ordinal,
                        tensors=np.array(
                            [self.scene.paths.intern_path(p).raw_id for p in product.camera_paths], dtype=np.uint64
                        ),
                        is_array=True,
                        semantic=ovstage.AttributeSemantic.RELATIONSHIP_PATH_ID,
                    ).wait()
        else:
            reference = self.backend.renderer.add_usd_reference_from_string(usd, f"/{scope}")
            render_data.resources.callback(self.backend.renderer.remove_usd, reference)
            # One camera write for several products can give them wrong images, so write each product on its own.
            for product in render_data.products:
                self.backend.renderer.write_array_attribute(
                    prim_paths=[product.path],
                    attribute_name="camera",
                    tensors=[product.camera_paths],
                )
        for path in render_data.render_product_paths:
            self.backend.render_products.add(path)
            render_data.resources.callback(self.backend.render_products.discard, path)

    def _remove_usd_reference(self, reference: int) -> None:
        """Publish removal of a private USD reference at the shared stage's current ordinal."""
        ovstage.population.remove_usd(self.scene.stage, reference)
        ovstage.population.apply_usd_changes(self.scene.stage, ordinal=self.scene.ordinal)

    def close(self) -> None:
        """Complete borrowed-buffer reads and release bindings; the registry owns the native engine."""
        try:
            with contextlib.ExitStack() as resources:
                resources.callback(self._bindings.close)
                resources.callback(self._transform_writes.close)
                resources.callback(self._geometry_writes.close)
                resources.callback(self.geometry_resources.close)
                for geometry in (*self.meshes.values(), *self.markers.values()):
                    resources.callback(geometry.close)
                for lidar in tuple(self._lidars):
                    resources.callback(self.cleanup_lidar, lidar)
                for render_data in tuple(self._camera_render_data):
                    resources.callback(self.cleanup, render_data)
        finally:
            self._object_xform_binding = self._geometry_points_binding = None
            self._object_scales = None
            self._transform_writes = _AsyncWriteBuffers(SceneDataFormat.TransposedMatrix44d() for _ in range(2))
            self._geometry_offsets.clear()
            self._geometry_paths = []
            self._geometry_timestamp = -1
            self._transforms_timestamp = -1
            self._output_id_color_buffers.clear()
            self._semantic_palette_key = ()
            self._semantic_palette_buffers = None
            self._initialized_scene = False
            self._visual_material_writer_ref = None
            self.meshes.clear()
            self.markers.clear()
            self.geometry_callbacks.clear()


class OVRTXMesh:
    """Own a runtime mesh reference and update its vertices directly from device arrays."""

    def __init__(self, renderer: OVRTXRenderer, path: str):
        self.renderer = renderer
        self.path = path
        self.resources = contextlib.ExitStack()
        self.stage = None
        self._native_resources = self.resources.enter_context(contextlib.ExitStack())
        self._appearance = (True, (0.7, 0.7, 0.7), 0.5, 0.0, 1.0)
        self._texture = None
        self._face_counts = None
        self._normals = None

    def update(
        self,
        points: wp.array,
        indices: wp.array,
        normals: wp.array | None = None,
        uvs: wp.array | None = None,
        texture: np.ndarray | str | None = None,
        hidden: bool = False,
        backface_culling: bool = True,
        color: tuple | None = None,
        roughness: float | None = None,
        metallic: float | None = None,
        dynamic: bool = False,
        opacity: float | None = None,
    ) -> None:
        """Publish GPU geometry and apply explicitly supplied appearance changes."""
        indices = indices.view(wp.int32)
        errors = self.renderer.drain_pending_renders()
        if errors:
            raise ExceptionGroup("OVRTX renders failed before mesh update", errors)
        if normals is None:
            if self._normals is None or len(self._normals) != len(points):
                self._normals = wp.empty_like(points)
            self._normals.zero_()
            wp.launch(
                accumulate_mesh_normals,
                len(indices) // 3,
                [points, indices, self._normals],
                device=points.device,
            )
            wp.launch(normalize_mesh_normals, len(points), [self._normals], device=points.device)
            normals = self._normals
        if self._face_counts is None or len(self._face_counts) != len(indices) // 3:
            self._face_counts = wp.full(len(indices) // 3, 3, dtype=wp.int32, device=indices.device)
        if color is not None:
            color = tuple(float(channel) for channel in color[:3])  # Callers pass NumPy rows or sequences.
        appearance = tuple(
            old if value is None else value
            for old, value in zip(
                self._appearance, (backface_culling, color, roughness, metallic, opacity), strict=True
            )
        )
        populate = self.stage is None or appearance != self._appearance
        if self.stage is None:
            self.stage = Usd.Stage.CreateInMemory()
            mesh = UsdGeom.Mesh.Define(self.stage, "/Mesh")
            self.stage.SetDefaultPrim(mesh.GetPrim())
            UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("omni:scenePartition", Sdf.ValueTypeNames.Token).Set("viewer")
            mesh.CreatePointsAttr(points.numpy())
            mesh.CreateNormalsAttr(normals.numpy())
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
            mesh.CreateFaceVertexIndicesAttr(indices.numpy())
            mesh.CreateFaceVertexCountsAttr(self._face_counts.numpy())
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateDoubleSidedAttr(not backface_culling)
            material_cfg = PreviewSurfaceCfg()
            with use_stage(self.stage):
                material_cfg.func("/Mesh/Material", material_cfg)
            material = UsdShade.Material.Get(self.stage, "/Mesh/Material")
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        mesh = UsdGeom.Mesh.Get(self.stage, "/Mesh")
        material = UsdShade.Material.Get(self.stage, "/Mesh/Material")
        surface = material.ComputeSurfaceSource()[0]
        if populate:
            mesh.CreateDoubleSidedAttr(not appearance[0])
            for name, value in zip(("diffuseColor", "roughness", "metallic", "opacity"), appearance[1:], strict=True):
                surface.GetInput(name).Set(Gf.Vec3f(*value) if name == "diffuseColor" else value)
            self._appearance = appearance
        if uvs is not None and not UsdGeom.PrimvarsAPI(mesh).HasPrimvar("st"):
            UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
            ).Set(uvs.numpy())
            populate = True
        if texture is not None and not np.array_equal(texture, self._texture):
            if isinstance(texture, np.ndarray):
                directory = self.resources.enter_context(tempfile.TemporaryDirectory(prefix="isaaclab-mesh-"))
                texture_path = Path(directory) / "texture.png"
                pixels = (np.clip(texture, 0.0, 1.0) * 255).astype(np.uint8) if texture.dtype.kind == "f" else texture
                Image.fromarray(pixels).save(texture_path)
                self._texture = texture.copy()
            else:
                texture_path = Path(texture).absolute()
                self._texture = texture
            reader = UsdShade.Shader.Define(self.stage, "/Mesh/Material/Primvar")
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
            sampler = UsdShade.Shader.Define(self.stage, "/Mesh/Material/Texture")
            sampler.CreateIdAttr("UsdUVTexture")
            sampler.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture_path))
            sampler.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
            sampler.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            surface.GetInput("diffuseColor").ConnectToSource(sampler.ConnectableAPI(), "rgb")
            populate = True
        if populate:
            if self._texture is not None:
                sampler = UsdShade.Shader.Get(self.stage, "/Mesh/Material/Texture")
                sampler.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set((*appearance[1], 1.0))
            self._native_resources.close()
            usd = self.stage.GetRootLayer().ExportToString()
            if self.renderer._use_ovstage:
                reference = ovstage.population.add_usd_reference_from_string(self.renderer.scene.stage, usd, self.path)
                self._native_resources.callback(self.renderer._remove_usd_reference, reference)
                ovstage.population.apply_usd_changes(self.renderer.scene.stage, ordinal=self.renderer.scene.ordinal)
                self.query = self._native_resources.enter_context(self.renderer.scene.query([self.path]))
            else:
                reference = self.renderer.backend.renderer.add_usd_reference_from_string(usd, self.path)
                self._native_resources.callback(self.renderer.backend.renderer.remove_usd, reference)
        attributes = {"points": points, "normals": normals}
        if uvs is not None:
            attributes["primvars:st"] = uvs
        if dynamic:
            attributes.update(faceVertexIndices=indices, faceVertexCounts=self._face_counts)
        stream = wp.get_stream(points.device).cuda_stream or 1
        for name, values in attributes.items():
            if self.renderer._use_ovstage:
                semantic = {
                    "points": ovstage.AttributeSemantic.POINT,
                    "normals": ovstage.AttributeSemantic.NORMAL,
                    "primvars:st": ovstage.AttributeSemantic.TEXTURE_COORDINATE,
                }.get(name, ovstage.AttributeSemantic.NONE)
                if name in ("points", "normals"):
                    values = points_tensor_from_warp(values)
                elif name == "primvars:st":
                    values = ovstage.make_dltensor(
                        values, dtype=ovstage.DLDataType(code=ovstage.DLDataTypeCode.kDLFloat, bits=32, lanes=2)
                    )
                self.renderer.scene.stage.write_attribute(
                    self.query,
                    name,
                    ordinal=self.renderer.scene.ordinal,
                    tensors=[values],
                    is_array=True,
                    semantic=semantic,
                    cuda_stream=stream,
                ).wait()
            else:
                self.renderer.backend.renderer.write_array_attribute(
                    [self.path], name, [values], data_access=DataAccess.ASYNC, cuda_stream=stream
                )
        token = "invisible" if hidden or not len(points) else "inherited"
        if self.renderer._use_ovstage:
            self.renderer.scene.stage.write_attribute(
                self.query,
                "visibility",
                ordinal=self.renderer.scene.ordinal,
                tensors=np.array([self.renderer.scene.paths.intern_token(token).raw_id], dtype=np.uint64),
                is_array=False,
                semantic=ovstage.AttributeSemantic.TOKEN_ID,
            ).wait()
        else:
            self.renderer.backend.renderer.write_attribute(
                [self.path], "visibility", [token], semantic=Semantic.TOKEN_STRING
            )

    def close(self) -> None:
        """Release the mesh reference while the shared renderer remains alive."""
        with self.resources:
            errors = self.renderer.drain_pending_renders()
            if errors:
                raise ExceptionGroup("OVRTX renders failed before mesh cleanup", errors)


class OVRTXMarkers:
    """Shared USD marker instances with GPU transform bindings."""

    def __init__(self, renderer: OVRTXRenderer, cfg: VisualizationMarkersCfg):
        self._renderer = renderer
        self._resources = contextlib.ExitStack()
        self._path = f"{cfg.prim_path}_{id(self)}"
        self._visible = True
        self._source_count = -1
        self._layout = None
        self._transforms = self._selection = None
        self._stage = Usd.Stage.CreateInMemory()
        root = UsdGeom.Xform.Define(self._stage, "/Markers")
        UsdGeom.PrimvarsAPI(root).CreatePrimvar("omni:scenePartition", Sdf.ValueTypeNames.Token).Set("viewer")
        root.GetVisibilityAttr().Set("inherited")
        self._stage.SetDefaultPrim(root.GetPrim())
        with use_stage(self._stage):
            for name, prototype in cfg.markers.items():
                prim = prototype.func(f"/Markers/Prototypes/{name}", prototype)
                prim.SetSpecifier(Sdf.SpecifierClass)
        self._prototypes = tuple(f"/Markers/Prototypes/{name}" for name in cfg.markers)
        self._prototype_transforms = np.array(
            [UsdGeom.Xformable(self._stage.GetPrimAtPath(path)).GetLocalTransformation() for path in self._prototypes]
        )
        self._local_transforms = None

    def update(
        self,
        translations: torch.Tensor,
        orientations: torch.Tensor | None,
        scales: torch.Tensor | None,
        marker_indices: torch.Tensor | None,
        *,
        visible: bool,
    ) -> None:
        """Project the marker owner's current state into GPU-bound USD instances."""
        renderer = self._renderer
        count = len(translations)
        count_changed = count != self._source_count
        if count_changed:
            self._instance_ids = wp.array(np.arange(count, dtype=np.int32), device=renderer._device)
            self._transforms = wp.empty(count, dtype=wp.mat44d, device=renderer._device)
            self._identity = wp.full(count, wp.quat_identity(), dtype=wp.quat, device=renderer._device)
            self._unit_scale = wp.ones(count, dtype=wp.vec3, device=renderer._device)
            self._source_count = count
        if marker_indices is not self._selection or self._layout is None or count_changed:
            indices = (0,) * count if marker_indices is None else tuple(marker_indices.tolist())
            if len(indices) != count or any(i < 0 or i >= len(self._prototypes) for i in indices):
                raise ValueError("Each marker must select a configured prototype.")
            self._selection = marker_indices
        else:
            indices = self._layout
        token = "inherited" if visible else "invisible"
        UsdGeom.Imageable(self._stage.GetDefaultPrim()).GetVisibilityAttr().Set(token)
        if indices != self._layout:
            errors = renderer.drain_pending_renders()
            if errors:
                raise ExceptionGroup("OVRTX renders failed before marker geometry update", errors)
            self._resources.close()
            self._stage.RemovePrim("/Markers/Instances")
            self._local_transforms = wp.array(
                self._prototype_transforms[list(indices)], dtype=wp.mat44d, device=renderer._device
            )
            paths = []
            for index, prototype in enumerate(indices):
                path = f"/Markers/Instances/marker_{index}"
                prim = self._stage.DefinePrim(path)
                prim.GetReferences().AddInternalReference(self._prototypes[prototype])
                prim.SetInstanceable(True)
                prim.CreateAttribute("omni:resetXformStack", Sdf.ValueTypeNames.Bool).Set(True)
                paths.append(self._path + path[len("/Markers") :])
            if indices:
                usd = self._stage.GetRootLayer().ExportToString()
                with self._resources:
                    if renderer._use_ovstage:
                        reference = ovstage.population.add_usd_reference_from_string(
                            renderer.scene.stage, usd, self._path
                        )
                        self._resources.callback(renderer._remove_usd_reference, reference)
                        ovstage.population.apply_usd_changes(renderer.scene.stage, ordinal=renderer.scene.ordinal)
                        self._root_query = self._resources.enter_context(renderer.scene.query([self._path]))
                        self._query = self._resources.enter_context(renderer.scene.query(paths))
                    else:
                        reference = renderer.backend.renderer.add_usd_reference_from_string(usd, self._path)
                        self._resources.callback(renderer.backend.renderer.remove_usd, reference)
                        self._binding = renderer.backend.renderer.bind_attribute(
                            paths, "omni:xform", semantic=Semantic.XFORM_MAT4x4
                        )
                        self._resources.callback(self._binding.unbind)
                    self._resources = self._resources.pop_all()
            self._layout = indices
        if not indices:
            return
        if visible != self._visible:
            if renderer._use_ovstage:
                renderer.scene.stage.write_attribute(
                    self._root_query,
                    "visibility",
                    ordinal=renderer.scene.ordinal,
                    tensors=np.array([renderer.scene.paths.intern_token(token).raw_id], dtype=np.uint64),
                    is_array=False,
                    semantic=ovstage.AttributeSemantic.TOKEN_ID,
                ).wait()
            else:
                renderer.backend.renderer.write_attribute(
                    [self._path], "visibility", [token], semantic=Semantic.TOKEN_STRING
                )
            self._visible = visible
        stream = wp.stream_from_torch(torch.cuda.current_stream(translations.device))
        wp.launch(
            create_marker_transforms_kernel,
            len(indices),
            [
                wp.from_torch(translations.contiguous(), wp.vec3),
                self._identity if orientations is None else wp.from_torch(orientations.contiguous(), wp.quat),
                self._unit_scale if scales is None else wp.from_torch(scales.contiguous(), wp.vec3),
                self._local_transforms,
                self._instance_ids,
                self._transforms,
            ],
            device=renderer._device,
            stream=stream,
        )
        if renderer._use_ovstage:
            renderer.scene.stage.write_attribute(
                self._query,
                "omni:xform",
                ordinal=renderer.scene.ordinal,
                tensors=xform_tensor_from_warp(self._transforms),
                is_array=False,
                semantic=ovstage.AttributeSemantic.MATRIX,
                cuda_stream=stream.cuda_stream or 1,
            ).wait()
        else:
            self._binding.write(self._transforms, data_access=DataAccess.ASYNC, cuda_stream=stream.cuda_stream or 1)

    def close(self) -> None:
        """Release bindings and geometry before the renderer's native scene closes."""
        if self._renderer is None:
            return
        renderer, self._renderer = self._renderer, None
        with self._resources:
            errors = renderer.drain_pending_renders()
            if errors:
                raise ExceptionGroup("OVRTX renders failed before marker cleanup", errors)
        self._transforms = self._selection = self._stage = None
        self._local_transforms = None


_BufferT = TypeVar("_BufferT", wp.array, SceneDataFormat.TransposedMatrix44d)


class _AsyncWriteBuffers(Generic[_BufferT]):
    """Retain GPU buffers until native writes finish; reuse them in submission order."""

    def __init__(self, buffers: Iterable[_BufferT] = ()):
        self._writes: deque[tuple[_BufferT, Operation[bool] | None, wp.Stream | None]] = deque(
            (buffer, None, None) for buffer in buffers
        )

    def acquire(self, stream: wp.Stream) -> _BufferT:
        """Wait until the next buffer is writable on the given stream."""
        buffer, operation, producer = self._writes[0]
        if operation is not None:
            operation.wait()
            if producer != stream:
                stream.wait_stream(producer)
        return buffer

    def submit(
        self, binding: AttributeBinding, values: wp.array | list[wp.array], stream: wp.Stream
    ) -> Operation[bool]:
        """Submit a write and retain its inputs; a failed submission does not advance the buffers."""
        operation = binding.write_async(values, data_access=DataAccess.ASYNC, cuda_stream=stream.cuda_stream or 1)
        self._writes[0] = (self._writes[0][0], operation, stream)
        self._writes.rotate(-1)
        return operation

    def close(self) -> None:
        """Complete every pending write before releasing storage, including on failure."""
        with contextlib.ExitStack() as writes:
            writes.callback(self._writes.clear)
            for _, operation, producer in self._writes:
                if operation is not None:
                    writes.callback(wp.synchronize_stream, producer)
                    writes.callback(operation.wait)
