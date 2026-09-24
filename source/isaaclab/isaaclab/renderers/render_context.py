# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulation-scoped rendering state."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
import warp as wp

from ..sensors.camera.camera_data import CameraData
from .base_renderer import BaseRenderer, VisualMaterialBatch
from .renderer_cfg import RendererCfg

if TYPE_CHECKING:
    from ..sim import BackendCfg

logger = logging.getLogger(__name__)


def __getattr__(name: str) -> Any:
    if name == "RENDER_PROFILE_SCOPE":
        from ..benchmark.stepping import RENDER_PROFILE_SCOPE

        warnings.warn(
            "isaaclab.renderers.render_context.RENDER_PROFILE_SCOPE is deprecated; "
            "use isaaclab.benchmark.stepping.RENDER_PROFILE_SCOPE instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return RENDER_PROFILE_SCOPE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@wp.kernel(enable_backward=False)
def _write_material(
    values: wp.array(dtype=Any, ndim=2),
    offsets: wp.array(dtype=wp.int32),
    env_ids: wp.array(dtype=wp.int32),
    output: wp.array(dtype=Any),
):
    material, env = wp.tid()
    row = offsets[material] + env_ids[env]
    output[row] = values[material, env]


_MATERIAL_WRITES = {
    (): wp.float32,
    (2,): wp.vec2f,
    (3,): wp.vec3f,
}


class RenderContext:
    """Orchestrate simulation-owned renderers and own flat runtime material buffers.

    Renderer instances are borrowed from the simulation's backend registry. SDP owns transform
    freshness, including pose writes that do not advance the physics-step counter.
    """

    __slots__ = (
        "clone_contexts",
        "_backend_registry",
        "_physics_initialized",
        "_prepared_renderer_ids",
        "_prepared_num_envs",
        "_last_geometry_update_step",
        "_visual_materials",
        "_visual_material_batches",
        "_visual_material_batches_by_channel",
        "_visual_material_batch_views",
        "_visual_material_writers",
        "_visual_material_selections",
        "_visual_material_env_ids",
        "_consumers_finalized",
    )

    def __init__(self, backend_registry: list[tuple[BackendCfg, Any]]) -> None:
        self.clone_contexts: set[type | str] = set()
        """Scene representations declared by camera renderers and visualizers before cloning."""
        self._backend_registry = backend_registry
        self._physics_initialized: bool = False  # Set to True after the first PHYSICS_READY callback fires.
        self._prepared_renderer_ids: set[int] = set()
        self._prepared_num_envs: int | None = None
        self._last_geometry_update_step: int | None = None  # Physics step of the last renderer geometry update.
        self._visual_materials: list[Any] = []
        self._visual_material_batches: tuple[VisualMaterialBatch, ...] = ()
        self._visual_material_batches_by_channel: dict[str, VisualMaterialBatch] = {}
        self._visual_material_batch_views: dict[str, wp.array] = {}
        self._visual_material_writers: tuple[Any, ...] = ()
        self._visual_material_selections: dict[tuple[str, tuple[int, ...]], tuple[torch.Tensor, wp.array]] = {}
        self._visual_material_env_ids: dict[tuple[torch.device, int], tuple[torch.Tensor, wp.array]] = {}
        self._consumers_finalized = False

    @property
    def _renderer_entries(self) -> tuple[tuple[RendererCfg, BaseRenderer], ...]:
        return tuple((cfg, resource) for cfg, resource in self._backend_registry if isinstance(cfg, RendererCfg))

    @property
    def renderer_types(self) -> tuple[str, ...]:
        """Return the registered camera renderer types."""
        return tuple(cfg.renderer_type for cfg, _renderer in self._renderer_entries)

    def validate_renderer_cfg(self, cfg: RendererCfg) -> None:
        """Reject late registration and conflicting global settings before renderer construction."""
        if self._consumers_finalized and self._visual_material_batches:
            raise RuntimeError("Renderers must be registered before rendering consumers are finalized.")
        if cfg.renderer_type != "isaac_rtx":
            return
        for stored_cfg, _renderer in self._renderer_entries:
            if stored_cfg.renderer_type != "isaac_rtx":
                continue
            if stored_cfg.global_settings != cfg.global_settings:
                raise ValueError(
                    "Isaac RTX global settings differ across camera renderer configs. "
                    "These settings are process-global; configure the same "
                    "IsaacRtxRendererCfg.global_settings for every Isaac RTX camera."
                )

    def register_renderer(self, cfg: RendererCfg, renderer: BaseRenderer) -> None:
        """Include a newly registry-owned renderer in cloning and post-physics initialization."""
        self.clone_contexts.update(cfg.cloning_contexts)
        self._last_geometry_update_step = None
        if self._physics_initialized:
            renderer.initialize()

    def ensure_initialize(self) -> None:
        """Idempotent call fired after PHYSICS_READY callback."""
        if self._physics_initialized:
            return
        self._physics_initialized = True
        for _cfg, renderer in self._renderer_entries:
            renderer.initialize()

    def register_visual_material(self, material: Any) -> None:
        """Register one initialized material asset for flat channel composition."""
        if any(registered is material for registered in self._visual_materials):
            return
        if self._consumers_finalized:
            raise RuntimeError("Visual materials must initialize before rendering consumers are finalized.")
        self._visual_materials.append(material)

    def finalize_consumers(self, visualizers: list[Any], *, rebuild: bool = False) -> None:
        """Compose material buffers and create backend writers at the post-reset lifecycle point."""
        if self._consumers_finalized and not rebuild:
            return

        old_writers, self._visual_material_writers = self._visual_material_writers, ()
        self._consumers_finalized = False
        close_error = None
        for writer in old_writers:
            try:
                writer.close()
            except Exception as exc:  # noqa: BLE001 - close every writer before reporting failure
                close_error = close_error or exc
        if close_error is not None:
            raise RuntimeError("Failed to close a visual-material writer during rebuild.") from close_error
        batches = []
        channels = {channel for material in self._visual_materials for channel in material.channels}
        for channel in sorted(channels):
            rows = sorted(
                (
                    (
                        material,
                        material._material_paths,
                        material._shader_paths,
                        material._input_names[channel],
                        material._values[channel],
                    )
                    for material in self._visual_materials
                    if channel in material.channels
                ),
                key=lambda row: row[3],
            )
            values = torch.cat([row[4] for row in rows])
            material_paths = tuple(path for row in rows for path in row[1])
            shader_paths = tuple(path for row in rows for path in row[2])
            input_names = tuple(row[3] for row in rows for _ in row[1])
            batches.append(VisualMaterialBatch(channel, material_paths, shader_paths, input_names, values))
            offset = 0
            for material, paths, _shader_paths, _input_name, _material_values in rows:
                end = offset + len(paths)
                material._values[channel] = values[offset:end]
                material._offsets[channel] = offset
                offset = end
        self._visual_material_batches = tuple(batches)
        self._visual_material_batches_by_channel = {batch.channel: batch for batch in batches}
        self._visual_material_batch_views = {
            batch.channel: wp.from_torch(batch.values, dtype=_MATERIAL_WRITES[tuple(batch.values.shape[1:])])
            for batch in batches
        }
        self._visual_material_selections.clear()
        self._visual_material_env_ids.clear()
        factories = []
        consumers = (*visualizers, *(renderer for _cfg, renderer in self._renderer_entries))
        for consumer in consumers:
            factory = consumer.visual_material_writer
            if factory is not None and factory not in factories:
                factories.append(factory)
        writers = []
        try:
            if batches:
                device = batches[0].values.device
                stream = wp.stream_from_torch(torch.cuda.current_stream(device)) if device.type == "cuda" else None
                with wp.ScopedStream(stream, sync_enter=False):
                    for factory in factories:
                        writers.append(factory(self._visual_material_batches))
                    for writer in writers:
                        writer()
        except Exception:
            for writer in writers:
                writer.close()
            raise
        self._visual_material_writers = tuple(writers)
        self._consumers_finalized = True

    def write_visual_materials(
        self, materials: list[Any], channels: dict[str, torch.Tensor], env_ids: torch.Tensor | None = None
    ) -> None:
        """Update selected rows and dispatch the already-compiled backend writers."""
        if not materials or not channels:
            return
        if not self._consumers_finalized:
            raise RuntimeError("Visual materials can only be written after simulation reset.")
        per_env = materials[0].is_per_env
        if not per_env and env_ids is not None:
            raise ValueError("env_ids is only valid for per-environment materials.")

        device = next(iter(self._visual_material_batches_by_channel.values())).values.device
        count = materials[0].num_instances if per_env else 1
        if env_ids is None:
            env_key = (device, count)
            selected = self._visual_material_env_ids.get(env_key)
            if selected is None:
                env_tensor = torch.arange(count, dtype=torch.int32, device=device)
                selected = (env_tensor, wp.from_torch(env_tensor, dtype=wp.int32))
                self._visual_material_env_ids[env_key] = selected
        else:
            env_tensor = env_ids.to(device=device, dtype=torch.int32)
            selected = (env_tensor, wp.from_torch(env_tensor, dtype=wp.int32))

        stream = wp.stream_from_torch(torch.cuda.current_stream(device)) if device.type == "cuda" else None
        with wp.ScopedStream(stream, sync_enter=False):
            material_offsets = {}
            material_key = tuple(id(material) for material in materials)
            for channel, values in channels.items():
                batch = self._visual_material_batches_by_channel[channel]
                key = (channel, material_key)
                offsets = self._visual_material_selections.get(key)
                if offsets is None:
                    offset_tensor = torch.tensor(
                        [material._offsets[channel] for material in materials],
                        dtype=torch.int32,
                        device=batch.values.device,
                    )
                    offsets = (offset_tensor, wp.from_torch(offset_tensor, dtype=wp.int32))
                    self._visual_material_selections[key] = offsets
                trailing = tuple(batch.values.shape[1:])
                expected = (len(materials), len(selected[0]), *trailing)
                values = values.detach().to(device=batch.values.device, dtype=torch.float32)
                if not per_env:
                    values = values.unsqueeze(1)
                if tuple(values.shape) != expected:
                    raise ValueError(
                        f"Channel {channel!r} values must have shape {expected}; got {tuple(values.shape)}."
                    )
                dtype = _MATERIAL_WRITES[trailing]
                wp.launch(
                    _write_material,
                    dim=(len(materials), len(selected[0])),
                    inputs=[
                        wp.from_torch(values, dtype=dtype),
                        offsets[1],
                        selected[1],
                        self._visual_material_batch_views[channel],
                    ],
                    device=str(batch.values.device),
                )
                material_offsets[channel] = offsets[1]
            for writer in self._visual_material_writers:
                writer(material_offsets, selected[1])

    def ensure_prepare_stage(self, stage: Any, num_envs: int) -> None:
        """Call :meth:`BaseRenderer.prepare_stage` for each registered backend (once per backend).

        If a new backend is added after the first :meth:`prepare_stage` call, this method ensures
        that new backend is prepared for the same ``stage`` and ``num_envs`` when the camera
        that owns it is initialized.

        Args:
            stage: USD stage passed to each backend.
            num_envs: Environment count.

        Raises:
            RuntimeError: If no renderer is registered, or ``num_envs`` disagrees with
                a value already used for a prepared backend in this context.
        """
        if not self._renderer_entries:
            raise RuntimeError("A renderer must be registered before ensure_prepare_stage.")
        if self._prepared_num_envs is not None and self._prepared_num_envs != num_envs:
            raise RuntimeError(
                "RenderContext prepare_stage was used with a different num_envs "
                f"({self._prepared_num_envs} vs {num_envs})."
            )
        for _cfg, renderer in self._renderer_entries:
            rid = id(renderer)
            if rid not in self._prepared_renderer_ids:
                renderer.prepare_stage(stage, num_envs)
                self._prepared_renderer_ids.add(rid)
        if self._prepared_num_envs is None:
            self._prepared_num_envs = num_envs

    def update_scene_state(self, physics_step_count: int) -> None:
        """Publish physics state and refresh renderers through SDP's producer versions.

        Transforms follow SDP freshness; geometry updates retain their once-per-step cadence.
        """
        for _cfg, renderer in self._renderer_entries:
            renderer.update_transforms()
            if self._last_geometry_update_step != physics_step_count:
                renderer.update_geometries()
        self._last_geometry_update_step = physics_step_count

    def render_into_camera(
        self,
        renderer: BaseRenderer,
        render_data: Any,
        camera_data: CameraData,
        physics_step_count: int,
    ) -> None:
        """Sync scene state and capture one camera through :meth:`render_into_cameras`."""
        self.render_into_cameras([(renderer, render_data, camera_data)], physics_step_count)

    def render_into_cameras(
        self,
        requests: Sequence[tuple[BaseRenderer, Any, CameraData]],
        physics_step_count: int,
    ) -> None:
        """Render prepared cameras in batches grouped by renderer instance.

        Camera poses must be updated before this call. Requests are used only for this
        submission; the context does not retain cameras or manage sensor timing.

        Args:
            requests: Tuples of renderer, renderer-specific render data, and output camera data.
                An empty sequence performs no work.
            physics_step_count: Current physics step for shared scene synchronization.
        """
        if not requests:
            return

        self.update_scene_state(physics_step_count)

        groups: dict[int, tuple[BaseRenderer, list[tuple[Any, CameraData]]]] = {}
        for renderer, render_data, camera_data in requests:
            groups.setdefault(id(renderer), (renderer, []))[1].append((render_data, camera_data))

        for renderer, cameras in groups.values():
            renderer.render_batch([render_data for render_data, _ in cameras])
            for render_data, camera_data in cameras:
                renderer.read_output(render_data, camera_data)

    def reset_stage_prepare_flag(self) -> None:
        """Allow :meth:`ensure_prepare_stage` to run ``prepare_stage`` again (e.g. a new USD stage)."""
        self._prepared_renderer_ids.clear()
        self._prepared_num_envs = None

    def reset_scene_state_cadence(self) -> None:
        """Invalidate geometry updates after resets that do not advance the physics step."""
        self._last_geometry_update_step = None

    def close(self) -> None:
        """Release material writers and lifecycle bookkeeping, not registry-owned renderers.

        Raises:
            RuntimeError: If a material writer failed to close.
        """
        errors: list[Exception] = []
        for writer in self._visual_material_writers:
            try:
                writer.close()
            except Exception as exc:  # noqa: BLE001 - reported after every resource is closed
                logger.error("Error closing visual-material writer: %s", exc)
                errors.append(exc)
        self.clone_contexts.clear()
        self._prepared_renderer_ids.clear()
        self._prepared_num_envs = None
        self._last_geometry_update_step = None
        self._physics_initialized = False
        self._visual_materials.clear()
        self._visual_material_batches = ()
        self._visual_material_batches_by_channel.clear()
        self._visual_material_batch_views.clear()
        self._visual_material_writers = ()
        self._visual_material_selections.clear()
        self._visual_material_env_ids.clear()
        self._consumers_finalized = False

        if errors:
            # TODO: Use ExceptionGroup when ruff target-version is bumped to py311+
            raise RuntimeError(f"{len(errors)} material writer(s) failed to close") from errors[0]
