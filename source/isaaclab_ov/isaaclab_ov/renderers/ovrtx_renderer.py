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
import logging
import math
import os
import sys
import weakref
from builtins import ExceptionGroup
from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, cast

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


try:
    from ovrtx import (
        BindingFlag,
        DataAccess,
        Device,
        PrimMode,
        Renderer,
        RendererConfig,
        RenderProductSetOutputs,
        Semantic,
        TextureStreamingMode,
    )
except ModuleNotFoundError as exc:
    if exc.name != "ovrtx":
        raise
    raise ModuleNotFoundError(
        "The OVRTX renderer requires the optional 'ovrtx' runtime wheel, which is not installed. "
        "Run your command with: uv run --extra ovrtx <command> "
        "(or, manually: python -m pip install 'ovrtx==0.5.0.377615')."
    ) from exc

from isaaclab.renderers import BaseRenderer, RenderBufferKind, RenderBufferSpec
from isaaclab.scene_data import SceneDataFormat
from isaaclab.sim import SimulationContext
from isaaclab.utils.warp import ProxyArray
from isaaclab.utils.warp.warp_math import convert_camera_frame_orientation_convention_wp

from isaaclab_ov.cloner import OvPhysxReplicateContext, ovrtx_replicate
from isaaclab_ov.renderers.ovrtx_annotator_utils import (
    build_instance_id_to_labels_and_semantics,
    build_semantic_id_to_labels,
    decode_semantic_id_map,
    decode_stable_id_map,
    decode_stable_id_semantic_id_map,
)
from isaaclab_ov.renderers.ovrtx_compat import OVRTX_VERSION, uses_prim_path_render_vars
from isaaclab_ov.renderers.ovrtx_renderer_cfg import OVRTXBackendCfg, OVRTXRendererCfg
from isaaclab_ov.renderers.ovrtx_renderer_kernels import (
    create_camera_transforms_kernel,
    extract_all_tiles_kernel,
    generate_random_colors_from_ids_kernel,
)
from isaaclab_ov.renderers.ovrtx_shader_cache import redirect_shader_cache
from isaaclab_ov.renderers.ovrtx_usd import (
    _RTX_MINIMAL_MODES,
    build_render_product_as_string,
    export_stage_to_string,
    render_var_prim_names_by_source,
)
from isaaclab_ov.renderers.visual_materials import OVRTXVisualMaterialWriter
from isaaclab_ov.stage import (
    OvstageBackendCfg,
    points_tensor_from_warp,
    xform_tensor_from_numpy,
    xform_tensor_from_warp,
)

if TYPE_CHECKING:
    from isaaclab_ppisp import PpispPipeline
    from ovrtx import AttributeBinding, Operation, PendingFetch, RenderProductSetOutputs

    from isaaclab.renderers.base_renderer import VisualMaterialBatch
    from isaaclab.sensors.camera.camera_data import CameraData

from isaaclab.renderers.camera_render_spec import CameraRenderSpec

_RENDER_VAR_PRIM_NAMES = render_var_prim_names_by_source()
_RENDER_DELTA_TIME = 1.0 / 60.0
_LDR_COLOR_VAR = "LdrColor"
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


_PPISP_IMPORT_ERROR_MESSAGE = (
    "isaaclab_ppisp is required when CameraCfg.isp_cfg is set. "
    "It ships with the Isaac Lab wheel (`pip install isaaclab`); otherwise install the "
    "isaaclab-ppisp extension from the Isaac Lab source checkout."
)
_READ_GPU_TRANSFORMS_ENV = "ISAAC_LAB_OVRTX_READ_GPU_TRANSFORMS"


# Runtime environment variable used to enable the ovstage code path for ovrtx.
_USE_OVSTAGE_ENV = "ISAAC_LAB_OVRTX_USE_OVSTAGE"


# Opts Linux out of the host wait, onto the same GPU-side ordering every other platform uses.
# See :meth:`OVRTXRenderer._map_render_var_to_dlpack`.
_DISABLE_LINUX_CUDA_CPU_SYNC_ENV = "ISAAC_LAB_OVRTX_DISABLE_LINUX_CUDA_CPU_SYNC"


def ovrtx_use_ovstage_enabled() -> bool:
    """Return whether the ovstage scene-ownership path should be used.

    Enabled by ``ISAAC_LAB_OVRTX_USE_OVSTAGE=1``. With OVPhysX, this selects one shared
    physics and rendering stage. Native cloning remains the default.

    Raises:
        ValueError: If the environment variable is set to anything other than ``0`` or ``1``.
    """
    value = os.environ.get(_USE_OVSTAGE_ENV, "0").strip()
    if value not in {"0", "1"}:
        raise ValueError(f"Invalid value for environment variable `{_USE_OVSTAGE_ENV}`: {value}. Expected 0 or 1.")
    return value == "1"


def _raise_missing_ppisp_error(exc: ModuleNotFoundError) -> NoReturn:
    # Only translate missing isaaclab_ppisp imports into the optional-dependency hint;
    # unrelated missing modules should surface unchanged for easier debugging.
    if exc.name != "isaaclab_ppisp" and not (exc.name and exc.name.startswith("isaaclab_ppisp.")):
        raise exc
    raise ModuleNotFoundError(_PPISP_IMPORT_ERROR_MESSAGE, name="isaaclab_ppisp") from exc


def _read_gpu_transforms_enabled() -> bool:
    """Return whether OVRTX should read GPU transforms from its internal transform cache."""
    value = os.environ.get(_READ_GPU_TRANSFORMS_ENV, "1").strip()
    if value not in {"0", "1"}:
        raise ValueError(
            f"Invalid value for environment variable `{_READ_GPU_TRANSFORMS_ENV}`: {value}. Expected 0 or 1."
        )
    return value == "1"


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


def _write_combined_stage(output_dir: Path, scene_usd: str, render_product_usd: str) -> None:
    """Write the scene and render product prims in one debug layer, preserving scene metadata."""
    from pxr import Sdf

    scene_layer = Sdf.Layer.CreateAnonymous("scene.usda")
    scene_layer.ImportFromString(scene_usd)
    render_layer = Sdf.Layer.CreateAnonymous("render_product.usda")
    render_layer.ImportFromString(render_product_usd)
    for prim in render_layer.rootPrims:
        Sdf.CopySpec(render_layer, prim.path, scene_layer, prim.path)
    _write_file(output_dir, "ovrtx_renderer_stage.usda", scene_layer.ExportToString())


class OVRTXBackend:
    """Own one native renderer, independently of its camera products and optional OVStage."""

    def __init__(self, cfg: OVRTXBackendCfg):
        # Resolve the wheel's native dependency here, including callers without a viewer.
        dependency = Path(ovstage.__file__).parent / "bin/plugins/libosdCPU.so.3.6.0"
        if dependency.exists():
            with contextlib.suppress(OSError):
                ctypes.CDLL(str(dependency))
        native_cfg = RendererConfig(
            log_file_path=cfg.log_file_path,
            log_level=cfg.log_level,
            read_gpu_transforms=cfg.read_gpu_transforms,
            keep_system_alive=True,
            suppress_deprecation_warnings=True,
            texture_streaming_mode=TextureStreamingMode.SYNCHRONOUS,
        )
        # Redirection may initialize the native library and must use the same settings.
        redirect_shader_cache(native_cfg)
        self.next_camera_id = 0
        self.attached = False
        if not cfg.use_ovstage:
            # Prepared by clone dispatch, consumed after camera overrides are authored.
            self.clone_copies: list[tuple[str, list[str]]] = []
            self.clone_env_paths: list[str] = []
            self.population_env_paths: list[str] = []
            self.clone_positions: np.ndarray | None = None
        self.renderer = Renderer(native_cfg)

    def close(self) -> None:
        """Destroy the engine; the registry releases any borrowed stage afterward."""
        if self.renderer is not None:
            # The SDK destroys bindings and detaches only if attached, including partial initialization.
            self.renderer.destroy()
            self.renderer = None


class OVRTXCameraRenderData:
    """Owns one camera sensor's native resources and Warp output buffers."""

    def __init__(self, spec: CameraRenderSpec, device, render_scope_name: str):
        """Create render data for a camera in its assigned render scope.

        Args:
            spec: Camera render specification.
            device: Rendering device.
            render_scope_name: Root scope containing this camera's render product and RenderVars.
        """
        self.render_scope_name = render_scope_name
        self.render_product_name = "RenderProduct"
        self.render_product_path = f"/{render_scope_name}/{self.render_product_name}"
        self.render_var_keys: dict[str, str] = (
            {source: f"/{render_scope_name}/Vars/{name}" for source, name in _RENDER_VAR_PRIM_NAMES.items()}
            if uses_prim_path_render_vars(OVRTX_VERSION)
            else {source: source for source in _RENDER_VAR_PRIM_NAMES}
        )
        self.camera_xform_binding = None
        self.camera_xform_query = None
        self.resources = contextlib.ExitStack()
        self.width = spec.cfg.width
        self.height = spec.cfg.height
        self.num_envs = spec.num_instances
        self.data_types = spec.cfg.data_types if spec.cfg.data_types else ["rgb"]
        self.num_cols = math.ceil(math.sqrt(self.num_envs))
        self.num_rows = math.ceil(self.num_envs / self.num_cols)
        self.warp_buffers: dict[str, wp.array] = {}
        self.intrinsic_bindings: list[AttributeBinding] = []
        self.camera_writes = _AsyncWriteBuffers()
        self.pending: tuple[Operation[PendingFetch[RenderProductSetOutputs]], dict | None] | None = None
        self.ready: tuple[Operation[PendingFetch[RenderProductSetOutputs]], dict | None] | None = None
        self.capture: dict[str, ProxyArray] = {}
        # Per-output metadata collected during render() and copied into CameraData.info by read_output().
        # Populated for "semantic_segmentation" (with an "idToLabels" mapping) and
        # "instance_segmentation" (with "idToLabels" and "idToSemantics" mappings).
        self.renderer_info: dict[str, Any] = {}
        # Post-render PPISP pipeline composed when ``spec.cfg.isp_cfg`` is set.
        # ``isp_cfg`` is already fully normalized by ``prepare_cameras`` by the time it reaches here.
        self.ppisp_pipeline: PpispPipeline | None = None
        if spec.cfg.isp_cfg is not None:
            try:
                from isaaclab_ppisp import PpispPipeline
            except ModuleNotFoundError as exc:
                _raise_missing_ppisp_error(exc)

            self.ppisp_pipeline = PpispPipeline(spec.cfg.isp_cfg)

    def cleanup(self) -> None:
        """Release this camera's native resources and buffers. Safe to call repeatedly.

        Resources are released in reverse acquisition order, before the shared renderer or stage
        is closed. The product path remains as an identifier for renderer bookkeeping.
        """
        try:
            with self.resources:
                self.camera_writes.close()
        finally:
            self.pending = self.ready = None
            self.capture = {}
            self.camera_xform_binding = None
            self.camera_xform_query = None
            self.intrinsic_bindings.clear()
            self.warp_buffers.clear()
            self.renderer_info.clear()
            self.ppisp_pipeline = None


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

    def __init__(self, cfg: OVRTXRendererCfg):
        self.cfg = cfg
        self._device = "cuda:0"  # default; overridden by create_render_data(spec)
        # Resolved by create_render_data(spec); every render-product device id and CUDA sync stream
        # derives from this one cached device so a bare "cuda" cannot be re-interpreted per call site.
        self._warp_device: wp.Device | None = None
        self._camera_render_data: list[OVRTXCameraRenderData] = []
        self._sdp = SimulationContext.instance().get_scene_data_provider()
        self._transforms_timestamp = -1
        self._transform_writes = _AsyncWriteBuffers(SceneDataFormat.TransposedMatrix44d() for _ in range(2))
        self._object_scales: wp.array | None = None
        self._object_scales_by_path: dict[str, tuple[float, float, float]] = {}
        self._geometry_paths: list[str] = []
        self._geometry_offsets: dict[str, int] = {}
        self._geometry_writes = _AsyncWriteBuffers()
        self._geometry_timestamp = -1
        self._initialized_scene = False
        self._exported_usd_string: str | None = None
        self._output_id_color_buffers: dict[str, wp.array] = {}
        self._visual_material_writer_ref: weakref.ReferenceType[OVRTXVisualMaterialWriter] | None = None

        # Selected once at construction so every operation below sees a stable path for the
        # lifetime of the renderer, even if the environment variable changes mid-process.
        sim = SimulationContext.instance()
        self._use_ovstage = ovrtx_use_ovstage_enabled()
        # TODO: Uncomment after requiring an OVPhysX release with batched cold binding and direct
        # path lookups, and verifying startup parity (release version TBD; 0.6.3 is affected).
        # self._use_ovstage |= sim.physics_manager.clone_context_type is OvPhysxReplicateContext
        shared_physics = self._use_ovstage and sim.physics_manager.clone_context_type is OvPhysxReplicateContext
        if shared_physics and sim.physics_manager.backend is not None and sim.physics_manager.backend.scene is None:
            raise RuntimeError("Configure OVRTX cameras before the first OVPhysX reset to share its OVStage.")
        if cfg.async_rendering and self._use_ovstage:
            logger.warning("Asynchronous OVRTX rendering is unsupported with ovstage; rendering synchronously.")
        backend_cfg = OVRTXBackendCfg(
            scene_key=None if shared_physics else cfg,
            log_file_path=cfg.log_file_path,
            log_level=cfg.log_level,
            use_ovstage=self._use_ovstage,
            read_gpu_transforms=_read_gpu_transforms_enabled(),
        )
        if shared_physics and any(
            isinstance(other, OVRTXBackendCfg) and other != backend_cfg for other, _ in sim._backend_registry
        ):
            raise ValueError(
                "A shared OVStage requires one OVRTX engine; use the same native logging/transform settings."
            )
        # Register the stage first so teardown destroys its borrowing engine before the stage.
        domains = ovstage.PopulationDomain.ALL if shared_physics else ovstage.PopulationDomain.RENDERING
        self.scene = (
            sim.get_or_create_backend(
                OvstageBackendCfg(
                    scene_key=None if shared_physics else backend_cfg,
                    population_domains=domains,
                )
            )
            if self._use_ovstage
            else None
        )
        self.backend: OVRTXBackend = sim.get_or_create_backend(backend_cfg)
        """Native engine borrowed from the simulation registry."""
        if self.scene is None:
            self.scene = self.backend
        self._object_xform_binding = self._geometry_points_binding = None
        self._object_xform_query = self._object_paths_list = None
        self._geometry_points_query = self._geometry_paths_list = None

    def visual_material_writer(self, batches: tuple[VisualMaterialBatch, ...]) -> OVRTXVisualMaterialWriter:
        """Create the detached-scene material writer after scene population."""
        if not self._initialized_scene:
            raise RuntimeError("OVRTX must ingest its detached scene before material writes are compiled.")
        writer = OVRTXVisualMaterialWriter(self, batches)
        self._visual_material_writer_ref = weakref.ref(writer)
        return writer

    def prepare_cameras(self, stage: Any, spec: CameraRenderSpec) -> None:
        """Resolve the camera's PPISP cfg and apply OVRTX-specific USD overrides.

        When ``spec.cfg.isp_cfg`` is set, resolves it (sentinel discovery +
        normalization) via :func:`isaaclab_ppisp.resolve_and_normalize` so
        :mod:`isaaclab` does not need to know about PPISP. Then pins
        ``exposure:*`` to neutral and applies ``OmniRtxCameraExposureAPI_1`` so
        the RTX exposure model OVRTX embeds does not compound on top of the
        ISP. Without an ISP, the camera prim's authored exposure is left alone.
        """
        if spec.cfg.isp_cfg is None:
            return
        try:
            from isaaclab_ppisp import apply_rtx_exposure_overrides, resolve_and_normalize
        except ModuleNotFoundError as exc:
            _raise_missing_ppisp_error(exc)

        camera_prim_path = spec.camera_prim_paths[0] if spec.camera_prim_paths else None
        spec.cfg.isp_cfg = resolve_and_normalize(spec.cfg.isp_cfg, stage, camera_prim_path)
        if spec.cfg.isp_cfg is None or not spec.camera_prim_paths:
            return
        apply_rtx_exposure_overrides(stage, list(spec.camera_prim_paths))

    def prepare_stage(self, stage: Any, num_envs: int) -> None:
        """Prepare the USD stage for OVRTX before :meth:`create_render_data`.

        Capture composed scales and export unpopulated scenes for :meth:`create_render_data`.
        """
        if stage is None:
            return

        # If temp_usd_dir is set, write the pre-ovrtx stage to a temporary file.
        if self.cfg.temp_usd_dir is not None:
            _write_file(Path(self.cfg.temp_usd_dir), "pre_ovrtx_renderer_stage.usda", stage.ExportToString())

        logger.info("Preparing stage (%d envs)...", num_envs)

        # Composed scales must be read while the full stage is still live, before export trims it.
        self._capture_object_scales(stage)

        # Populate backend-owned frames; native replication applies plan poses after copying prototypes.
        sources = tuple(source for source, _ in self.scene.clone_copies)
        if not self._use_ovstage or not self.scene.ordinal or self.cfg.temp_usd_dir is not None:
            self._exported_usd_string = export_stage_to_string(
                stage, num_envs, source_paths=sources, keep_env_roots=False, env_paths=self.scene.population_env_paths
            )

    def _capture_object_scales(self, stage: Any) -> None:
        """Record composed world scales beneath the routed sources and their prepared destinations.

        The per-frame object transform write rebuilds each body's matrix from an SDP
        pose, which carries only translation and rotation, so any scale authored on the
        USD prim is lost once that write lands. Capturing the composed scale here, while the full
        stage is still live, preserves it in the body binding order.

        Only non-unit scales are stored. Native instance paths include repeated assets whose
        destinations OVRTX creates after the host stage is exported.

        Args:
            stage: The live USD stage, before per-environment trimming and export.
        """
        self._object_scales_by_path.clear()

        from pxr import Gf, Usd, UsdGeom

        xform_cache = UsdGeom.XformCache()
        for root, targets in self.scene.clone_copies:
            for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
                if not prim.IsA(UsdGeom.Xformable):
                    continue
                scale = tuple(map(float, Gf.Transform(xform_cache.GetLocalToWorldTransform(prim)).GetScale()))
                if not all(math.isclose(axis, 1.0, rel_tol=1e-6, abs_tol=1e-6) for axis in scale):
                    path = str(prim.GetPath())
                    self._object_scales_by_path[path] = scale
                    self._object_scales_by_path.update((target + path[len(root) :], scale) for target in targets)

    def _setup_xform_bindings(self) -> None:
        """Bind SDP body paths and their composed scales in the scene's representation."""
        object_paths = self._sdp.backend.transform_paths
        if not object_paths:
            return
        reset_xforms = np.ones(len(object_paths), dtype=np.bool_)
        if self._use_ovstage:
            self._object_paths_list = self.scene.paths.create_path_list_from_strings(object_paths)
            self._object_xform_query = self.scene.stage.query_from_path_list(self._object_paths_list)
            self.scene.stage.write_attribute(
                self._object_xform_query,
                "omni:resetXformStack",
                ordinal=self.scene.ordinal,
                tensors=reset_xforms,
                is_array=False,
            ).wait()
            binding = self._object_xform_query
        else:
            self._object_xform_binding = self.backend.renderer.bind_attribute(
                prim_paths=object_paths,
                attribute_name="omni:xform",
                semantic=Semantic.XFORM_MAT4x4,
                prim_mode=PrimMode.EXISTING_ONLY,
            )
            self.backend.renderer.write_attribute(
                prim_paths=object_paths, attribute_name="omni:resetXformStack", tensor=reset_xforms
            )
            binding = self._object_xform_binding
        if binding is None:
            raise RuntimeError("Failed to create OVRTX object bindings")
        scales = [self._object_scales_by_path.get(path, (1.0, 1.0, 1.0)) for path in object_paths]
        self._object_scales = wp.array(scales, dtype=wp.vec3f, device=self._device)

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
            self._geometry_paths_list = self.scene.paths.create_path_list_from_strings(self._geometry_paths)
            self._geometry_points_query = self.scene.stage.query_from_path_list(self._geometry_paths_list)
            if self._geometry_points_query is None:
                raise RuntimeError("Failed to create OVRTX geometry point bindings")
            self.scene.stage.write_attribute(
                self._geometry_points_query,
                "omni:resetXformStack",
                ordinal=self.scene.ordinal,
                tensors=reset_xforms,
                is_array=False,
            ).wait()
            self.scene.stage.write_attribute(
                self._geometry_points_query,
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

    def create_render_data(self, spec: CameraRenderSpec) -> OVRTXCameraRenderData:
        """Create OVRTX-specific RenderData with GPU buffers.

        Performs OVRTX initialization (stage export, USD load, bindings) on first call,
        matching the interface of Isaac RTX and Newton Warp which need no separate initialize().
        """
        source = spec.camera_prim_paths[0]
        prefix = "/World/envs/env_0/"
        if not source.startswith(prefix) or source == prefix:
            raise ValueError(f"OVRTX cameras must be under {prefix}, got {source!r}.")
        camera_paths = [f"/World/envs/env_{i}/{source[len(prefix) :]}" for i in range(spec.num_instances)]
        # Normalize aliases such as "cuda" before comparing cameras sharing this renderer.
        warp_device = wp.get_device(spec.device)
        if self._initialized_scene and str(warp_device) != self._device:
            raise ValueError("Cameras sharing an OVRTX renderer must use the same device.")
        self._warp_device = warp_device
        self._device = str(warp_device)
        render_data = OVRTXCameraRenderData(
            spec, self._device, render_scope_name=f"RenderCamera_{self.scene.next_camera_id}"
        )
        try:
            first_camera = not self._initialized_scene
            if first_camera:
                if self._exported_usd_string is None and not (self._use_ovstage and self.scene.ordinal):
                    raise RuntimeError("Expected an exported USD string from stage")
                env_paths = self.scene.clone_env_paths
                tokens = [f"env_{i}" for i in range(len(env_paths))]
                if self._use_ovstage:
                    if not self.scene.ordinal:
                        self.scene.populate(self._exported_usd_string)
                    if env_paths:
                        paths = self.scene.paths.create_path_list_from_strings(env_paths)
                        try:
                            with self.scene.stage.query_from_path_list(paths) as query:
                                self.scene.stage.write_attribute(
                                    query,
                                    "primvars:omni:scenePartition",
                                    ordinal=self.scene.ordinal,
                                    tensors=np.array(
                                        [self.scene.paths.intern_token(t) for t in tokens], dtype=np.uint64
                                    ),
                                    is_array=False,
                                    semantic=ovstage.AttributeSemantic.TOKEN_ID,
                                ).wait()
                        finally:
                            self.scene.paths.destroy_path_list(paths)
                else:
                    self.backend.renderer.open_usd_from_string(self._exported_usd_string)
                    ovrtx_replicate(
                        self.backend.renderer,
                        self.scene.clone_copies,
                        env_paths,
                        self.scene.clone_positions,
                    )
                    if env_paths:
                        self.backend.renderer.write_attribute(
                            env_paths, "primvars:omni:scenePartition", tokens, semantic=Semantic.TOKEN_STRING
                        )
                self._setup_xform_bindings()
                self._setup_geometry_bindings()
            self._register_camera(spec, render_data, camera_paths)
            if first_camera:
                if self._use_ovstage:
                    self.scene.commit()
                    if not self.backend.attached:
                        self.backend.renderer.attach_ovstage(self.scene.stage)
                        self.backend.attached = True
                self._exported_usd_string = None
                self._initialized_scene = True
            render_data.camera_writes = _AsyncWriteBuffers(
                tuple(wp.empty(spec.num_instances, dtype, device=self._device) for dtype in (wp.quatf, wp.mat44d))
                for _ in range(2 if self.cfg.async_rendering and not self._use_ovstage else 1)
            )
            if not self._use_ovstage:
                for name in _CAMERA_INTRINSIC_ATTRIBUTES:
                    binding = self.backend.renderer.bind_attribute(
                        prim_paths=camera_paths,
                        attribute_name=name,
                        dtype="float32",
                        prim_mode=PrimMode.EXISTING_ONLY,
                        flags=BindingFlag.OPTIMIZE,
                    )
                    render_data.resources.callback(binding.unbind)
                    render_data.intrinsic_bindings.append(binding)
        except Exception:
            render_data.cleanup()
            raise
        self.scene.next_camera_id += 1
        self._camera_render_data.append(render_data)
        return render_data

    def _register_camera(
        self, spec: CameraRenderSpec, render_data: OVRTXCameraRenderData, camera_paths: list[str]
    ) -> None:
        """Add a tiled product and its camera bindings after scene population and cloning."""
        # Registration mutates the shared scene (a new product and camera bindings), so any
        # in-flight render must finish first.
        errors = self.drain_pending_renders()
        if errors:
            raise ExceptionGroup("OVRTX renders failed before camera registration", errors)
        scope = render_data.render_scope_name
        product_path = render_data.render_product_path
        usd = build_render_product_as_string(
            spec,
            render_data,
            device_id=self._warp_device.ordinal,
            enable_shadows=self.cfg.enable_shadows,
        )
        if self.cfg.temp_usd_dir is not None and self._exported_usd_string is not None:
            _write_combined_stage(Path(self.cfg.temp_usd_dir), self._exported_usd_string, usd)
        if self._use_ovstage:
            reference = ovstage.population.add_usd_reference_from_string(self.scene.stage, usd, f"/{scope}")
            render_data.resources.callback(self._remove_camera_reference, reference)
            ovstage.population.apply_usd_changes(self.scene.stage, ordinal=self.scene.ordinal)
            product_paths = self.scene.paths.create_path_list_from_strings([product_path])
            try:
                with self.scene.stage.query_from_path_list(product_paths) as query:
                    # USD references drop external camera targets; author the relationship in Fabric.
                    self.scene.stage.write_attribute(
                        query,
                        "camera",
                        ordinal=self.scene.ordinal,
                        tensors=np.array([self.scene.paths.intern_path(p) for p in camera_paths], dtype=np.uint64),
                        is_array=True,
                        semantic=ovstage.AttributeSemantic.RELATIONSHIP_PATH_ID,
                    ).wait()
            finally:
                self.scene.paths.destroy_path_list(product_paths)
            camera_paths_list = self.scene.paths.create_path_list_from_strings(camera_paths)
            render_data.resources.callback(self.scene.paths.destroy_path_list, camera_paths_list)
            render_data.camera_xform_query = render_data.resources.enter_context(
                self.scene.stage.query_from_path_list(camera_paths_list)
            )
            self.scene.stage.write_attribute(
                render_data.camera_xform_query,
                "omni:resetXformStack",
                ordinal=self.scene.ordinal,
                tensors=np.full(spec.num_instances, True, dtype=np.bool_),
                is_array=False,
            ).wait()
            self.scene.stage.write_attribute(
                render_data.camera_xform_query,
                "omni:scenePartition",
                ordinal=self.scene.ordinal,
                tensors=np.array(
                    [self.scene.paths.intern_token(f"env_{i}") for i in range(spec.num_instances)], dtype=np.uint64
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
                tensors=[camera_paths],
            )
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
            self.backend.renderer.write_attribute(
                camera_paths,
                "omni:scenePartition",
                [f"env_{i}" for i in range(spec.num_instances)],
                semantic=Semantic.TOKEN_STRING,
            )

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
        # When PPISP is composed but the user did not request the raw HDR AOV,
        # allocate an internal HDR scratch buffer under "rgb_hdr" so both the
        # HdrColor extractor and PPISP dispatch can use the same buffer map.
        if render_data.ppisp_pipeline is not None and str(RenderBufferKind.RGB_HDR) not in render_data.warp_buffers:
            ref_proxy = next(iter(output_data.values()))
            render_data.warp_buffers[str(RenderBufferKind.RGB_HDR)] = wp.zeros(
                (render_data.num_envs, render_data.height, render_data.width, 3),
                dtype=wp.float32,
                device=ref_proxy.device,
            )
        if render_data.ppisp_pipeline is not None:
            if str(RenderBufferKind.RGBA) not in render_data.warp_buffers:
                raise ValueError(
                    "OVRTX renderer ISP requires 'rgba' (or 'rgb', which aliases into rgba) as the"
                    " LDR output destination, but neither was provided. Add 'rgb' or 'rgba' to"
                    " Camera.cfg.data_types when isp_cfg is set."
                )

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
        camera_data: CameraData,
    ) -> None:
        """Publish only this camera's image and its matching metadata."""
        assert camera_data.info is not None, "CameraData.info should be created in CameraData.allocate"
        if self.cfg.async_rendering and not self._use_ovstage and render_data.ready is None:
            return
        capture = {}
        if render_data.ready is not None:
            operation, capture = render_data.ready
            if capture is None:
                render_data.ready = None
                return
            self._process_render_products((render_data,), operation.wait().fetch())
            if render_data.ready is render_data.pending:
                # Retain the native operation, but do not publish this priming image twice.
                render_data.pending = (operation, None)
            render_data.ready = None
        for output_name in camera_data.info:
            info = render_data.renderer_info.get(output_name)
            camera_data.info[output_name] = {**(info or {}), "capture": capture} if capture else info

    def _generate_random_colors_from_ids(self, input_ids: wp.array, output_colors: wp.array | None) -> wp.array:
        """Generate pseudo-random RGBA colors from uint32 IDs into a reusable output buffer.

        Args:
            input_ids: 3-D uint32 Warp array of shape (H, W, 1).
            output_colors: Existing color buffer to reuse, or None to allocate a new one.

        Returns:
            Color buffer containing the generated colors.
        """

        # Lazily allocate, and re-allocate if the shape changes.
        if output_colors is None or output_colors.shape != input_ids.shape:
            output_colors = wp.zeros(shape=input_ids.shape, dtype=wp.uint32, device=self._device)

        wp.launch(
            kernel=generate_random_colors_from_ids_kernel,
            dim=input_ids.shape,
            inputs=[input_ids, output_colors],
            device=self._device,
        )
        return output_colors

    @contextlib.contextmanager
    def _map_render_var_to_dlpack(self, render_var: Any) -> Iterator[wp.array]:
        """Map ``render_var`` for CUDA reads and yield it as a Warp array.

        The render is still in flight when the mapping returns, so reading it has to be ordered
        against render completion. Normally that is a ``cudaStreamWaitEvent`` on the Warp stream the
        consuming kernels run on, which is the ordering the OVRTX API is designed around.

        On Linux that GPU-side wait measures substantially slower end to end, so the mapping is
        instead requested with no GPU-side barrier and the calling thread blocks on the
        render-completion event. Setting :data:`_DISABLE_LINUX_CUDA_CPU_SYNC_ENV` to ``1`` puts
        Linux back on the GPU-side wait; it is an escape hatch for platforms where that trade-off
        no longer holds, and is worth re-measuring before being relied on.

        Note that ``sync_stream=0`` is OVRTX's "no sync" sentinel, *not* the NULL CUDA stream: the
        field encodes ``0=no sync, 1=default stream, >1=specific stream``, so omitting the argument
        entirely means ``1``, not ``0``.

        The yielded array is a zero-copy view of the mapped memory and is only valid inside the
        ``with`` block -- the mapping is released on exit.

        Args:
            render_var: OVRTX ``RenderVarOutput`` to map (looked up from ``frame.render_vars``).

        Yields:
            The render var's contents as a Warp array, valid for the duration of the context.
        """
        gpu_side_sync = _gpu_side_render_var_sync_enabled()
        sync_stream = self._warp_device.stream.cuda_stream if gpu_side_sync else 0
        with render_var.map(device=Device.CUDA, sync_stream=sync_stream) as mapping:
            if not gpu_side_sync:
                mapping.wait()
            yield wp.from_dlpack(mapping)

    def _process_id_segmentation_render_var(
        self,
        render_data: OVRTXCameraRenderData,
        frame,
        output_buffers: dict,
        render_var_key: str,
        buffer_key: str,
        colorize: bool,
    ) -> None:
        """Extract a uint32 ID-segmentation render var into ``output_buffers[buffer_key]``.

        Shared by ``semantic_segmentation`` (``SemanticSegmentation``) and ``instance_segmentation``
        (``NonStableInstanceSegmentation``), which only differ in the source render var, the destination buffer,
        and whether to colorize.

        Args:
            render_data: OVRTX render data for the current frame.
            frame: OVRTX frame holding the mapped render vars.
            output_buffers: Destination warp buffers, keyed by data type.
            render_var_key: Render-var source name.
            buffer_key: Data type key into ``output_buffers``.
            colorize: If True, IDs are mapped to RGBA colors; otherwise raw uint32 IDs are copied.
        """
        render_var = frame.render_vars.get(render_data.render_var_keys[render_var_key])
        if render_var is None or buffer_key not in output_buffers:
            return

        with self._map_render_var_to_dlpack(render_var) as tiled_data:
            if tiled_data.dtype != wp.uint32:
                return

            if colorize:
                color_buffer = self._generate_random_colors_from_ids(
                    tiled_data, self._output_id_color_buffers.get(buffer_key)
                )
                self._output_id_color_buffers[buffer_key] = color_buffer

                colors_torch = wp.to_torch(color_buffer)
                colors_uint8 = colors_torch.view(torch.uint8)
                if colors_torch.dim() == 2:
                    h, w = colors_torch.shape
                    colors_uint8 = colors_uint8.reshape(h, w, 4)
                tiled_data = wp.from_torch(colors_uint8, dtype=wp.uint8)
                self._extract_rgba_tiles(render_data, tiled_data, output_buffers, buffer_key)
            else:
                # Non-colorized: ensure (TH, TW, 1) shape for the uint32 extraction kernel. Reshape the warp
                # array directly instead of round-tripping through torch, which raises on ``torch.uint32``
                # (newer torch exposes the dtype but ``wp.from_torch`` still rejects it).
                if tiled_data.ndim == 2:
                    tiled_data = tiled_data.reshape((*tiled_data.shape, 1))
                self._launch_extract_all_tiles(render_data, tiled_data, output_buffers[buffer_key])

    def _process_semantic_id_map(self, render_data: OVRTXCameraRenderData, frame) -> None:
        """Decode the ``SemanticIdMap`` render var into ``render_data.renderer_info["semantic_segmentation"]``.

        Populates an ``"idToLabels"`` mapping compatible with Isaac RTX / Replicator: keys are the raw semantic
        IDs (``colorize_semantic_segmentation=False``) or the RGBA color tuples the segmentation buffer uses
        (``colorize_semantic_segmentation=True``); values are ``{semantic_type: label}`` dicts. The reserved
        BACKGROUND (ID 0) and UNLABELLED (ID 1) entries are always included.

        Args:
            render_data: OVRTX render data for the current frame.
            frame: OVRTX frame holding the mapped render vars.
        """
        semantic_id_map = frame.render_vars.get(render_data.render_var_keys[_SEMANTIC_ID_MAP_VAR])
        if semantic_id_map is None:
            return

        with semantic_id_map.map(device=Device.CPU) as mapping:
            labels_by_id = decode_semantic_id_map(np.from_dlpack(mapping))

        render_data.renderer_info["semantic_segmentation"] = {
            "idToLabels": build_semantic_id_to_labels(
                labels_by_id, colorize=self.cfg.colorize_semantic_segmentation, device=self._device
            )
        }

    def _process_instance_segmentation_maps(self, render_data: OVRTXCameraRenderData, frame) -> None:
        """Decode the instance-segmentation map render vars into ``renderer_info["instance_segmentation"]``.

        An *instance pixel ID* is a compact integer that the renderer assigns to each visible object instance.
        Every pixel in the segmentation buffer holds the ID of the instance rendered at that location; the same
        ID maps to the same object across the entire frame.  ID 0 is reserved for BACKGROUND (no geometry), and
        ID 1 for UNLABELLED (geometry with no semantic annotation).  All other IDs are dynamically assigned per
        frame.

        Populates ``"idToLabels"`` (instance pixel ID -> USD prim path) and ``"idToSemantics"`` (instance pixel
        ID -> ``{semantic_type: label}``) compatible with Isaac RTX / Replicator. Resolving both requires all
        three map render vars — ``StableIdSemanticIdMap`` (pixel ID -> stable ID + semantic ID), ``StableIdMap``
        (stable ID -> prim path), and ``SemanticIdMap`` (semantic ID -> label). Keys are the raw pixel IDs
        (``colorize_instance_segmentation=False``) or the RGBA color tuples the segmentation buffer uses
        (``colorize_instance_segmentation=True``); the reserved BACKGROUND (ID 0) and UNLABELLED (ID 1) entries
        are always included.

        Raises:
            RuntimeError: If any of the three required render vars is absent from ``frame``.

        Args:
            render_data: OVRTX render data for the current frame.
            frame: OVRTX frame holding the mapped render vars.
        """
        resolved = {
            key: frame.render_vars.get(render_data.render_var_keys[key]) for key in _INSTANCE_SEGMENTATION_MAP_VARS
        }
        missing = [key for key, render_var in resolved.items() if render_var is None]
        if missing:
            raise RuntimeError(
                f"instance_segmentation was requested but the following render vars are missing from the "
                f"OVRTX frame: {missing}. Available vars: {list(frame.render_vars.keys())}"
            )

        with resolved[_STABLE_ID_SEMANTIC_ID_MAP_VAR].map(device=Device.CPU) as mapping:
            stable_id_semantic_id_map = decode_stable_id_semantic_id_map(np.from_dlpack(mapping))
        with resolved[_STABLE_ID_MAP_VAR].map(device=Device.CPU) as mapping:
            stable_id_to_path = decode_stable_id_map(np.from_dlpack(mapping))
        with resolved[_SEMANTIC_ID_MAP_VAR].map(device=Device.CPU) as mapping:
            semantic_id_to_labels = decode_semantic_id_map(np.from_dlpack(mapping))

        id_to_labels, id_to_semantics = build_instance_id_to_labels_and_semantics(
            stable_id_semantic_id_map,
            stable_id_to_path,
            semantic_id_to_labels,
            colorize=self.cfg.colorize_instance_segmentation,
            device=self._device,
        )
        render_data.renderer_info["instance_segmentation"] = {
            "idToLabels": id_to_labels,
            "idToSemantics": id_to_semantics,
        }

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
            output_buffer: 4D array of shape (num_envs, H, W, C) to receive the per-env tiles, with C no
                greater than ``tiled_buffer``'s channel count.

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
            dim=(render_data.num_envs, render_data.height, render_data.width),
            inputs=[
                tiled_buffer,
                output_buffer,
                render_data.num_cols,
                render_data.width,
                render_data.height,
            ],
            device=self._device,
        )

    def _extract_rgba_tiles(
        self, render_data: OVRTXCameraRenderData, tiled_data: wp.array, output_buffers: dict, buffer_key: str
    ) -> None:
        """Extract per-env RGBA tiles from tiled buffer into output_buffers (single kernel launch)."""
        output_buffer = output_buffers[buffer_key]
        num_channels = output_buffer.shape[-1]
        if num_channels not in (3, 4):
            raise ValueError(f"Expected RGB (3 channels) or RGBA (4 channels), got {num_channels}")

        self._launch_extract_all_tiles(render_data, tiled_data, output_buffer)

    def _prepare_ppisp_hdr_source(
        self, render_data: OVRTXCameraRenderData, tiled_data: wp.array, output_buffers: dict
    ) -> wp.array:
        """Return the PPISP HdrColor source on the output buffer device."""
        if render_data.ppisp_pipeline is None:
            return tiled_data

        output_device = str(output_buffers[str(RenderBufferKind.RGB_HDR)].device)
        if str(tiled_data.device) == output_device:
            return tiled_data

        # The render product pins ``deviceIds`` to this renderer's CUDA device, so the mapping
        # normally lands on the output device already. This stays as a fallback for the case OVRTX
        # reports as "deviceIds ... not in the active device set" and falls back to automatic
        # assignment.
        return wp.clone(tiled_data, device=output_device)

    def _process_render_frame(self, render_data: OVRTXCameraRenderData, frame, output_buffers: dict) -> None:
        """Extract RGB, depth, albedo, and semantic from a single render frame into output_buffers."""
        # Reset per-output metadata so it is a snapshot of this frame only. Unlike pixel AOVs (always
        # present), metadata like the semantic ``idToLabels`` is only repopulated below when its render var
        # is available, so without this a missing SemanticIdMap on a later frame would leave a stale mapping.
        render_data.renderer_info.clear()

        ldr_color = frame.render_vars.get(render_data.render_var_keys[_LDR_COLOR_VAR])
        if ldr_color is not None:
            buffer_key = None

            if render_data.ppisp_pipeline is None and "rgba" in output_buffers:
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
                    self._extract_rgba_tiles(render_data, tiled_data, output_buffers, buffer_key)

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
                self._extract_rgba_tiles(render_data, tiled_albedo_data, output_buffers, "albedo")

        hdr_color = frame.render_vars.get(render_data.render_var_keys[_HDR_COLOR_VAR])
        if hdr_color is not None and "rgb_hdr" in output_buffers:
            with self._map_render_var_to_dlpack(hdr_color) as tiled_hdr_data:
                tiled_hdr_data = self._prepare_ppisp_hdr_source(render_data, tiled_hdr_data, output_buffers)
                if tiled_hdr_data.dtype not in (wp.float16, wp.float32):
                    raise TypeError(f"Unsupported OVRTX HdrColor dtype: {tiled_hdr_data.dtype}.")
                self._launch_extract_all_tiles(render_data, tiled_hdr_data, output_buffers["rgb_hdr"])

        self._process_id_segmentation_render_var(
            render_data,
            frame,
            output_buffers,
            _SEMANTIC_SEGMENTATION_VAR,
            "semantic_segmentation",
            self.cfg.colorize_semantic_segmentation,
        )
        # Decode the SemanticIdMap into camera.data.info["semantic_segmentation"]["idToLabels"].
        if "semantic_segmentation" in output_buffers:
            self._process_semantic_id_map(render_data, frame)

        self._process_id_segmentation_render_var(
            render_data,
            frame,
            output_buffers,
            _INSTANCE_SEGMENTATION_VAR,
            "instance_segmentation",
            self.cfg.colorize_instance_segmentation,
        )
        # Decode the StableIdSemanticIdMap/StableIdMap/SemanticIdMap trio into
        # camera.data.info["instance_segmentation"]["idToLabels"] and ["idToSemantics"].
        if "instance_segmentation" in output_buffers:
            self._process_instance_segmentation_maps(render_data, frame)

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
                errors.append(error)
        return errors

    def reset(self, render_data: OVRTXCameraRenderData, env_ids: Sequence[int] | None = None) -> None:
        """Retire pre-reset frames. The next capture primes this whole tiled product.

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
            if data.render_product_path not in products or not products[data.render_product_path].frames:
                raise RuntimeError(f"OVRTX returned no frame for render product {data.render_product_path!r}.")
        for data in render_data:
            self._process_render_frame(
                data,
                products[data.render_product_path].frames[0],
                data.warp_buffers,
            )

            # Post-render PPISP uses each camera's own HDR source and RGBA destination.
            if data.ppisp_pipeline is not None:
                data.ppisp_pipeline.apply(
                    data.warp_buffers[str(RenderBufferKind.RGB_HDR)],
                    data.warp_buffers[str(RenderBufferKind.RGBA)],
                )

    def update_transforms(self) -> None:
        """Write changed SDP transforms to OVRTX."""
        binding = self._object_xform_query if self._use_ovstage else self._object_xform_binding
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
        matrices = output.matrices
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
                cuda_stream=self._warp_device.stream.cuda_stream,
            ).wait()
        elif asynchronous:
            self._transform_writes.submit(binding, matrices, stream)
        else:
            binding.write(matrices, data_access=DataAccess.ASYNC, cuda_stream=stream.cuda_stream)
        self._transforms_timestamp = timestamp

    def update_geometries(self) -> None:
        """Write changed SDP geometry to OVRTX."""
        binding = self._geometry_points_query if self._use_ovstage else self._geometry_points_binding
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
                cuda_stream=self._warp_device.stream.cuda_stream,
            ).wait()
        elif asynchronous:
            self._geometry_writes.submit(binding, points, stream)
        else:
            binding.write(cast(Any, points), data_access=DataAccess.ASYNC, cuda_stream=stream.cuda_stream)
        self._geometry_timestamp = timestamp

    def update_camera(
        self,
        render_data: OVRTXCameraRenderData,
        positions: ProxyArray,
        orientations: ProxyArray,
        intrinsics: ProxyArray,
    ) -> None:
        """Update camera poses using the camera's reusable conversion buffers."""
        binding = render_data.camera_xform_query if self._use_ovstage else render_data.camera_xform_binding
        if binding is None:
            return
        stream = self._warp_device.stream
        converted, matrices = render_data.camera_writes.acquire(stream)
        convert_camera_frame_orientation_convention_wp(orientations, converted, "world", "opengl", device=self._device)
        wp.launch(
            create_camera_transforms_kernel, len(matrices), inputs=[positions, converted, matrices], device=self._device
        )
        if self._use_ovstage:
            self.scene.stage.write_attribute(
                binding,
                "omni:xform",
                ordinal=self.scene.ordinal,
                tensors=xform_tensor_from_warp(matrices),
                is_array=False,
                semantic=ovstage.AttributeSemantic.MATRIX,
                cuda_stream=stream.cuda_stream,
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
        stream = wp.get_stream(parameters.device).cuda_stream
        if self._use_ovstage:
            self.scene.stage.write_attributes(
                render_data.camera_xform_query,
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

    def render_batch(self, render_data: Sequence[OVRTXCameraRenderData]) -> None:
        """Render all requested camera products in one native submission.

        Every registered render product is submitted, not just the requested ones: OVRTX
        batches products within a step, and naming only a subset makes each product that
        re-enters the set on a later step far more expensive. Only ``render_data`` is read
        back, so products left out of the request keep their previous outputs.

        Args:
            render_data: Cameras whose poses and output buffers have been prepared. An empty
                sequence performs no work.

        Raises:
            RuntimeError: If the scene is uninitialized or a requested product returns no frame.
        """
        if not render_data:
            return
        if not self._initialized_scene:
            raise RuntimeError("Scene not initialized. Call initialize() first.")
        if self.backend.renderer is None or not self._camera_render_data:
            return
        products = {camera.render_product_path for camera in self._camera_render_data}
        material_writer = self._visual_material_writer_ref() if self._visual_material_writer_ref is not None else None
        try:
            if material_writer is not None:
                material_writer.publish()
            if self._use_ovstage:
                ordinal = self.scene.commit()
            elif self.cfg.async_rendering:
                unread = {data.ready[0] for data in render_data if data.ready is not None}
                operation = self.backend.renderer.step_async(render_products=products, delta_time=_RENDER_DELTA_TIME)
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

        Each camera owns its product and pose binding. Scene and physics bindings remain alive
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

    def _remove_camera_reference(self, reference: int) -> None:
        """Publish removal of a camera's USD reference at the shared stage's current ordinal."""
        ovstage.population.remove_usd(self.scene.stage, reference)
        ovstage.population.apply_usd_changes(self.scene.stage, ordinal=self.scene.ordinal)

    def close(self) -> None:
        """Complete borrowed-buffer reads and release bindings; the registry owns the native engine."""
        try:
            with contextlib.ExitStack() as resources:
                resources.callback(self._close_ovstage if self._use_ovstage else self._close_legacy)
                for render_data in tuple(self._camera_render_data):
                    resources.callback(self.cleanup, render_data)
                resources.callback(self._transform_writes.close)
                resources.callback(self._geometry_writes.close)
        finally:
            self._transform_writes = _AsyncWriteBuffers(SceneDataFormat.TransposedMatrix44d() for _ in range(2))
            self._geometry_offsets.clear()
            self._geometry_paths = []
            self._geometry_timestamp = -1
            self._transforms_timestamp = -1
            self._output_id_color_buffers.clear()
            self._initialized_scene = False
            self._visual_material_writer_ref = None

    def _close_legacy(self) -> None:
        """Release the renderer's tensor bindings. See :meth:`close`."""

        # Unbind before tearing down renderer
        def _safe_unbind(binding, name: str) -> None:
            if binding is None:
                return
            try:
                binding.unbind()
            except Exception as e:
                if "destroyed" not in str(e).lower():
                    logger.warning("Error unbinding %s: %s", name, e)

        _safe_unbind(self._object_xform_binding, "object transforms")
        self._object_xform_binding = None
        _safe_unbind(self._geometry_points_binding, "geometry points")
        self._geometry_points_binding = None

    def _close_ovstage(self) -> None:
        """Release the renderer's stage queries and path lists. See :meth:`close`."""

        def _safe_release_query(query, name: str) -> None:
            if query is None or self.scene.stage is None:
                return
            try:
                self.scene.stage.release_query(query).wait()
            except Exception as e:
                if "destroyed" not in str(e).lower():
                    logger.warning("Error releasing %s query: %s", name, e)

        def _safe_destroy_path_list(path_list, name: str) -> None:
            if path_list is None or self.scene.paths is None:
                return
            try:
                self.scene.paths.destroy_path_list(path_list)
            except Exception as e:
                if "destroyed" not in str(e).lower():
                    logger.warning("Error destroying %s path list: %s", name, e)

        _safe_release_query(self._object_xform_query, "object transforms")
        self._object_xform_query = None
        _safe_destroy_path_list(self._object_paths_list, "object paths")
        self._object_paths_list = None
        _safe_release_query(self._geometry_points_query, "geometry points")
        self._geometry_points_query = None
        _safe_destroy_path_list(self._geometry_paths_list, "geometry paths")
        self._geometry_paths_list = None

        self._object_scales = None
        self._object_scales_by_path = {}


class _AsyncWriteBuffers:
    """Retain GPU buffers until native writes finish; reuse them in submission order."""

    def __init__(self, buffers: Iterable[Any] = ()):
        self._writes = deque((buffer, None, None) for buffer in buffers)

    def acquire(self, stream: wp.Stream) -> Any:
        """Wait until the next buffer is writable on the given stream."""
        buffer, operation, producer = self._writes[0]
        if operation is not None:
            operation.wait()
            if producer != stream:
                stream.wait_stream(producer)
        return buffer

    def submit(self, binding: AttributeBinding, values: Any, stream: wp.Stream) -> Operation:
        """Submit a write and retain its inputs; a failed submission does not advance the buffers."""
        operation = binding.write_async(values, data_access=DataAccess.ASYNC, cuda_stream=stream.cuda_stream)
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
