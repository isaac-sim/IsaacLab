# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native engine, markers, visual materials and camera panels of the OVXR renderer's OVStage scene.

These follow the OVStage path of Isaac Lab's OVRTX renderer
(:mod:`isaaclab_ov.renderers.ovrtx_renderer` and :mod:`isaaclab_ov.renderers.visual_materials`):
the scene is written to the renderer's own OVStage, and the OVXR session renders it.
"""

from __future__ import annotations

import contextlib
import ctypes
import itertools
import logging
import os
import threading
import weakref
from collections.abc import Sequence
from dataclasses import MISSING
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import ovstage
import torch
import warp as wp

# The ovrtx C library links to its own USD libraries; keep it from loading the pxr Python package.
os.environ["OVRTX_SKIP_USD_CHECK"] = "1"

from isaaclab_ov.renderers.ovrtx_shader_cache import redirect_shader_cache
from isaaclab_ov.stage import xform_tensor_from_warp
from ovrtx import Renderer, RendererConfig, TextureStreamingMode

from pxr import Sdf, Usd, UsdGeom, UsdLux

from isaaclab.markers.visualization_markers_cfg import VisualizationMarkersCfg
from isaaclab.sim import BackendCfg, use_stage
from isaaclab.utils import configclass

from .ovxr_gpu_publication import parse_active_cuda_gpus

if TYPE_CHECKING:
    from isaaclab_ov.stage import OvstageBackend

    from isaaclab.renderers.base_renderer import VisualMaterialBatch
    from isaaclab.renderers.camera_render_spec import CameraRenderSpec

    from .ovxr_renderer import OvxrRenderer, OvxrRendererCfg

logger = logging.getLogger(__name__)


# Matrix layouts the native renderer reads, as in :mod:`isaaclab_ov.renderers.ovrtx_renderer_kernels`.


@wp.kernel(enable_backward=False)
def _marker_transforms_kernel(
    positions: wp.array(dtype=wp.vec3),
    orientations: wp.array(dtype=wp.quat),
    scales: wp.array(dtype=wp.vec3),
    local_transforms: wp.array(dtype=wp.mat44d),
    instance_ids: wp.array(dtype=wp.int32),
    transforms: wp.array(dtype=wp.mat44d),
):
    """Compose marker poses and scales in the native renderer's matrix layout."""
    i = wp.tid()
    source = instance_ids[i]
    position, orientation, scale = positions[source], orientations[source], scales[source]
    p = wp.vec3d(wp.float64(position[0]), wp.float64(position[1]), wp.float64(position[2]))
    q = wp.quatd(
        wp.float64(orientation[0]), wp.float64(orientation[1]), wp.float64(orientation[2]), wp.float64(orientation[3])
    )
    s = wp.vec3d(wp.float64(scale[0]), wp.float64(scale[1]), wp.float64(scale[2]))
    transforms[i] = local_transforms[i] * wp.transpose(wp.transform_compose(p, q, s))


@wp.kernel(enable_backward=False)
def _camera_transforms_kernel(
    positions: wp.array(dtype=wp.vec3),
    orientations: wp.array(dtype=wp.quatf),
    transforms: wp.array(dtype=wp.mat44d),
):
    """Convert world-frame camera poses to OVRTX's OpenGL row-vector matrices in one pass."""
    i = wp.tid()
    orientation = orientations[i] * wp.quatf(0.5, -0.5, -0.5, 0.5)
    transforms[i] = wp.mat44d(wp.transpose(wp.transform_to_matrix(wp.transformf(positions[i], orientation))))


# OVRTX selects rendering GPUs once per process. None is unselected; an empty tuple requests native GPU
# selection for a CPU simulation.
_initial_gpu_selection: tuple[int, ...] | None = None
_initialization_lock = threading.Lock()


@configclass
class _OvxrEngineCfg(BackendCfg):
    """Native OVRTX engine of one OVXR renderer, which attaches to that renderer's own OVStage."""

    class_type: type[_OvxrEngine] | str = "{DIR}.ovxr_scene:_OvxrEngine"
    scene_key: OvxrRendererCfg = MISSING
    """Configuration of the OVXR renderer that owns the engine's OVStage."""
    log_file_path: str = MISSING
    """Native renderer log destination."""
    log_level: str = MISSING
    """Native renderer log level."""
    device: str = MISSING
    """Normalized simulation device: ``cpu`` or ``cuda:<index>``."""
    active_cuda_gpus: str | None = None
    """Native rendering GPUs, such as ``"0,1"``, or ``None`` to render on the simulation device.

    A CPU simulation without a selection uses native GPU selection.
    """


class _OvxrEngine:
    """Own one native renderer, which reads its GPU transform cache and renders on the selected GPUs.

    The first engine fixes the rendering GPUs for the process, including after teardown or failed
    startup; a later engine that resolves to another selection is rejected before native startup.
    """

    def __init__(self, cfg: _OvxrEngineCfg):
        self.cfg = cfg
        self.attached = False
        self.render_products: set[str] = set()
        # Resolve the wheel's native dependency here, including callers without a viewer.
        dependency = Path(ovstage.__file__).parent / "bin/plugins/libosdCPU.so.3.6.0"
        if dependency.exists():
            with contextlib.suppress(OSError):
                ctypes.CDLL(str(dependency))
        # Select the simulation's CUDA device for rendering, matching Kit's ``activeCudaGpus`` setting.
        # An explicit ``active_cuda_gpus`` selection takes precedence.
        sim_device = wp.get_device(cfg.device)
        active_cuda_gpus = (
            cfg.active_cuda_gpus
            if cfg.active_cuda_gpus is not None
            else str(sim_device.ordinal)
            if sim_device.is_cuda
            else None
        )
        selection = parse_active_cuda_gpus(active_cuda_gpus) if active_cuda_gpus is not None else ()
        native_cfg = RendererConfig(
            active_cuda_gpus=active_cuda_gpus,
            log_file_path=cfg.log_file_path,
            log_level=cfg.log_level,
            read_gpu_transforms=True,
            keep_system_alive=True,
            suppress_deprecation_warnings=True,
            texture_streaming_mode=TextureStreamingMode.SYNCHRONOUS,
        )
        global _initial_gpu_selection
        with _initialization_lock:
            initial = _initial_gpu_selection
            if initial is not None and selection != initial:
                source = (
                    "from active_cuda_gpus"
                    if cfg.active_cuda_gpus is not None
                    else f"for simulation device {cfg.device!r}, since active_cuda_gpus is unset"
                )
                raise ValueError(
                    "OVRTX selects rendering GPUs once per process: an OVXR renderer already uses"
                    f" {initial or 'the native default'}, but this one requests"
                    f" {selection or 'the native default'} {source}. Every OVXR renderer in a process must"
                    " resolve to the same GPUs; start a new process to select different rendering GPUs."
                )
            visible = wp.get_cuda_device_count()
            if any(index >= visible for index in selection):
                # Reject before the selection is reserved for the process.
                raise ValueError(
                    f"active_cuda_gpus {active_cuda_gpus!r} names a CUDA device this process cannot see"
                    f" ({visible} visible)."
                )
            # Cache redirection and failed renderer construction can retain native system state. Reserve
            # the selection before either call; keep_system_alive also preserves it after close.
            _initial_gpu_selection = selection
            # Redirection may initialize the native library and must use the same settings.
            redirect_shader_cache(native_cfg)
            self.renderer = Renderer(native_cfg)

    def detach(self) -> None:
        """Release the borrowed stage before it is rebuilt or destroyed."""
        if self.attached:
            self.renderer.detach_ovstage()
            self.attached = False

    def close(self) -> None:
        """Destroy the engine; the simulation's backend registry releases the OVStage afterward."""
        if self.renderer is not None:
            # The SDK destroys bindings and detaches only if attached, including partial initialization.
            self.renderer.destroy()
            self.renderer = None
            self.attached = False


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


class _OvxrPanelRenderData:
    """One camera panel's render product, pose binding and pose buffer in the OVXR renderer's OVStage.

    The fields that :func:`~isaaclab_ov.renderers.ovrtx_usd.build_render_product_as_string` reads follow
    the OVRTX renderer's camera render data; a panel is one camera view.
    """

    def __init__(self, spec: CameraRenderSpec, render_scope_name: str, device: str):
        """Create render data for a panel camera in its assigned render scope.

        Args:
            spec: Camera render specification of the panel's single view.
            render_scope_name: Root scope containing the panel's render product and RenderVars.
            device: Simulation device, which holds the panel's pose buffer.
        """
        self.render_scope_name = render_scope_name
        self.render_product_path = f"/{render_scope_name}/RenderProduct"
        self.camera_paths = list(spec.camera_prim_paths)
        self.width = spec.cfg.width
        self.height = spec.cfg.height
        # The panel's one render product, with an atlas of one tile.
        self.products = [
            SimpleNamespace(path=self.render_product_path, camera_paths=self.camera_paths, num_cols=1, num_rows=1)
        ]
        self.camera_xform_binding: ovstage.Query | None = None
        self.camera_transforms = wp.empty(len(self.camera_paths), dtype=wp.mat44d, device=device)
        self.resources = contextlib.ExitStack()
        """Release the product and its binding, in reverse acquisition order."""

    def cleanup(self) -> None:
        """Release the panel's product and pose binding. Safe to call repeatedly."""
        try:
            self.resources.close()
        finally:
            self.camera_xform_binding = None


def _remove_usd_reference(scene: OvstageBackend, reference: int) -> None:
    """Publish removal of a private USD reference at the stage's current ordinal."""
    ovstage.population.remove_usd(scene.stage, reference)
    ovstage.population.apply_usd_changes(scene.stage, ordinal=scene.ordinal)


class _OvxrMarkers:
    """Shared USD marker instances whose poses are written to the OVXR renderer's OVStage."""

    def __init__(self, renderer: OvxrRenderer, cfg: VisualizationMarkersCfg):
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
        """Project the marker owner's current state into USD instances at the scene's ordinal."""
        renderer = self._renderer
        scene = renderer.scene
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
                    reference = ovstage.population.add_usd_reference_from_string(scene.stage, usd, self._path)
                    self._resources.callback(_remove_usd_reference, scene, reference)
                    ovstage.population.apply_usd_changes(scene.stage, ordinal=scene.ordinal)
                    self._root_query = self._resources.enter_context(scene.query([self._path]))
                    self._query = self._resources.enter_context(scene.query(paths))
                    self._resources = self._resources.pop_all()
            self._layout = indices
        if not indices:
            return
        if visible != self._visible:
            scene.stage.write_attribute(
                self._root_query,
                "visibility",
                ordinal=scene.ordinal,
                tensors=np.array([scene.paths.intern_token(token).raw_id], dtype=np.uint64),
                is_array=False,
                semantic=ovstage.AttributeSemantic.TOKEN_ID,
            ).wait()
            self._visible = visible
        # CPU marker state is converted and written on the host without a CUDA stream.
        stream = wp.stream_from_torch(torch.cuda.current_stream(translations.device)) if translations.is_cuda else None
        wp.launch(
            _marker_transforms_kernel,
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
        scene.stage.write_attribute(
            self._query,
            "omni:xform",
            ordinal=scene.ordinal,
            tensors=xform_tensor_from_warp(self._transforms),
            is_array=False,
            semantic=ovstage.AttributeSemantic.MATRIX,
            cuda_stream=(stream.cuda_stream or 1) if stream is not None else None,
        ).wait()

    def close(self) -> None:
        """Release queries and geometry before the renderer's OVStage closes."""
        if self._renderer is None:
            return
        self._renderer = None
        self._resources.close()
        self._transforms = self._selection = self._stage = None
        self._local_transforms = None


class _OvxrVisualMaterialWriter:
    """Publish dirty visual-material device buffers into the OVXR renderer's OVStage.

    The writer may exist before the Stage: environment reset finalizes material consumers before the
    session's render worker prepares the deferred Stage, so shader addresses compile on the first
    publication after population.
    """

    def __init__(self, renderer: OvxrRenderer, batches: tuple[VisualMaterialBatch, ...]):
        self._renderer_ref = weakref.ref(renderer)
        self._buffers: dict[str, torch.Tensor] = {}
        self._dirty_channels: set[str] = set()
        self._operations: tuple[Any, ...] = ()
        self._addresses: list[tuple[str, Any, str, slice]] = []
        self._pending_groups: list[tuple[str, str, list[str], slice]] = []
        self._resources = contextlib.ExitStack()

        groups = []
        device = None
        for batch in batches:
            values = batch.values.detach()
            if values.dtype != torch.float32 or not (
                values.ndim == 1 or (values.ndim == 2 and values.shape[1] in (2, 3))
            ):
                raise TypeError(
                    f"OVRTX visual-material channel {batch.channel!r} requires float, float2, or float3; "
                    f"got dtype={values.dtype}, shape={tuple(values.shape)}."
                )
            if not values.is_cuda or (device is not None and values.device != device):
                raise RuntimeError("OVRTX visual-material attributes must reside on one CUDA device.")
            device = values.device
            self._buffers[batch.channel] = values
            start = 0
            for input_name, input_group in itertools.groupby(batch.input_names):
                end = start + sum(1 for _ in input_group)
                rows = slice(start, end)
                groups.append((batch.channel, f"inputs:{input_name}", list(batch.shader_paths[rows]), rows))
                start = end
        self._device = str(device)
        self._event = wp.Event(device=self._device)
        if renderer._initialized_scene:
            self._compile_addresses(renderer, groups)
        else:
            self._pending_groups = groups

    @property
    def has_pending_updates(self) -> bool:
        """Whether material changes require a scene publication even without a physics step."""
        return bool(self._dirty_channels)

    def __call__(self, material_offsets: dict[str, Any] | None = None, env_ids: Any | None = None) -> None:
        """Mark channels dirty; each dirty channel's full device buffer is copied at publication."""
        del env_ids
        selected = self._buffers if material_offsets is None else material_offsets
        self._dirty_channels.update(selected)
        wp.record_event(self._event)

    def publish(self) -> None:
        """Submit dirty buffers at the scene's ordinal after ordering their producer streams."""
        channels = self._dirty_channels
        if not channels:
            return
        renderer = self._renderer_ref()
        if self._pending_groups:
            if not renderer._initialized_scene:
                raise RuntimeError("Visual materials require a populated renderer scene before publication.")
            self._compile_addresses(renderer, self._pending_groups)
            self._pending_groups = []
        scene = renderer.scene
        operations = []
        try:
            for channel, address, attribute_name, rows in self._addresses:
                if channel not in channels:
                    continue
                operations.append(
                    scene.stage.write_attribute(
                        address,
                        attribute_name,
                        ordinal=scene.ordinal,
                        tensors=self._buffers[channel][rows],
                        is_array=False,
                        cuda_event=self._event.cuda_event,
                    )
                )
        finally:
            self._operations = tuple(operations)
        channels.clear()

    def drain(self) -> None:
        """Complete submitted writes before their scene buffers may change.

        Every op is waited even when an earlier one fails. Skipping the rest would leave their
        writes pending with no remaining reference to wait on.
        """
        operations, self._operations = self._operations, ()
        errors = []
        for operation in operations:
            try:
                operation.wait()
            except Exception as e:
                errors.append(e)
        if errors:
            raise RuntimeError(f"{len(errors)} OVRTX material write(s) failed to complete") from errors[0]

    def close(self) -> None:
        """Drain writes and release every compiled scene query."""
        try:
            with self._resources:
                self.drain()
        finally:
            self._addresses.clear()
            self._pending_groups = []
            self._dirty_channels.clear()
            self._buffers.clear()

    def _compile_addresses(self, renderer: OvxrRenderer, groups: list[tuple[str, str, list[str], slice]]) -> None:
        """Query each channel's shader inputs in the renderer's populated scene."""
        addresses = []
        with contextlib.ExitStack() as resources:
            for channel, attribute_name, shader_paths, rows in groups:
                address = resources.enter_context(renderer.scene.query(shader_paths))
                addresses.append((channel, address, attribute_name, rows))
            self._resources.enter_context(resources.pop_all())
        self._addresses.extend(addresses)
