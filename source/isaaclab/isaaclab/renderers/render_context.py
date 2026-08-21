# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulation-scoped rendering state."""

from __future__ import annotations

import logging
from typing import Any, cast

import torch
import warp as wp

from isaaclab.app.logging_utils import force_log_level
from isaaclab.sensors.camera.camera_data import CameraData

from .base_renderer import BaseRenderer, VisualMaterialBatch
from .renderer import Renderer
from .renderer_cfg import RendererCfg

logger = logging.getLogger(__name__)


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
    """Own camera renderers and flat runtime material buffers for one simulation.

    A camera reuses a backend when a prior camera registered a config equal under ``==`` (value
    equality) and the same concrete ``RendererCfg`` subclass. A distinct ``RendererCfg`` that
    maps to a different implementation (e.g. Isaac RTX vs Newton) produces another backend; each
    has :meth:`BaseRenderer.prepare_stage` run before use.

    :meth:`update_scene_state` is invoked at most once per :meth:`get_physics_step_count` for the
    context;
    """

    __slots__ = (
        "_renderer_entries",
        "_physics_initialized",
        "_prepared_renderer_ids",
        "_prepared_num_envs",
        "_last_scene_state_step",
        "_visual_materials",
        "_visual_material_batches",
        "_visual_material_batches_by_channel",
        "_visual_material_batch_views",
        "_visual_material_writers",
        "_visual_material_selections",
        "_visual_material_env_ids",
        "_consumers_finalized",
    )

    def __init__(self) -> None:
        self._renderer_entries: list[tuple[RendererCfg, BaseRenderer]] = []
        self._physics_initialized: bool = False  # Set to True after the first PHYSICS_READY callback fires.
        self._prepared_renderer_ids: set[int] = set()
        self._prepared_num_envs: int | None = None
        self._last_scene_state_step: int | None = None
        self._visual_materials: list[Any] = []
        self._visual_material_batches: tuple[VisualMaterialBatch, ...] = ()
        self._visual_material_batches_by_channel: dict[str, VisualMaterialBatch] = {}
        self._visual_material_batch_views: dict[str, wp.array] = {}
        self._visual_material_writers: tuple[Any, ...] = ()
        self._visual_material_selections: dict[tuple[str, tuple[int, ...]], tuple[torch.Tensor, wp.array]] = {}
        self._visual_material_env_ids: dict[tuple[torch.device, int], tuple[torch.Tensor, wp.array]] = {}
        self._consumers_finalized = False

    def _check_global_settings_compatible(self, cfg: RendererCfg) -> None:
        """Reject conflicting process-global renderer settings."""
        if getattr(cfg, "renderer_type", None) != "isaac_rtx" or not hasattr(cfg, "global_settings"):
            return
        for stored_cfg, _renderer in self._renderer_entries:
            if getattr(stored_cfg, "renderer_type", None) != "isaac_rtx" or not hasattr(stored_cfg, "global_settings"):
                continue
            if stored_cfg.global_settings != cfg.global_settings:
                raise ValueError(
                    "Isaac RTX global settings differ across camera renderer configs. "
                    "These settings are process-global; configure the same "
                    "IsaacRtxRendererCfg.global_settings for every Isaac RTX camera."
                )

    @property
    def renderer_types(self) -> tuple[str, ...]:
        """Return the registered camera renderer types."""
        return tuple(cfg.renderer_type for cfg, _renderer in self._renderer_entries)

    def get_renderer(self, cfg: RendererCfg) -> BaseRenderer:
        """Return a backend for this configuration, reusing a matching instance if present.

        Lookups use ``==`` and concrete ``RendererCfg`` type, so :func:`hash` is not used (configs
        are typically not hashable).

        Args:
            cfg: Renderer configuration from the initializing camera.

        Returns:
            A shared or newly created renderer backend.
        """
        self._check_global_settings_compatible(cfg)
        for stored_cfg, r in self._renderer_entries:
            if type(stored_cfg) is type(cfg) and stored_cfg == cfg:
                return r
        if self._consumers_finalized and self._visual_material_batches:
            raise RuntimeError("Renderers must be registered before rendering consumers are finalized.")
        new_renderer = cast(BaseRenderer, Renderer(cfg))  # type: ignore[misc]
        self._renderer_entries.append((cfg, new_renderer))
        with force_log_level(logging.INFO):
            logger.info("Created new renderer for simulation: %s", type(new_renderer).__name__)
        if self._physics_initialized:
            new_renderer.initialize()
        return new_renderer

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
            RuntimeError: If :meth:`get_renderer` was never called, or ``num_envs`` disagrees with
                a value already used for a prepared backend in this context.
        """
        if not self._renderer_entries:
            raise RuntimeError("get_renderer must be called at least once before ensure_prepare_stage.")
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
        """Update scene state on all backends (at most once per step).

        Invokes :meth:`BaseRenderer.update_transforms` and then
        :meth:`BaseRenderer.update_geometries` on each registered renderer.
        """
        if not self._renderer_entries:
            return

        if self._last_scene_state_step == physics_step_count:
            return

        for _cfg, renderer in self._renderer_entries:
            renderer.update_transforms()
            renderer.update_geometries()

        self._last_scene_state_step = physics_step_count

    def render_into_camera(
        self,
        renderer: BaseRenderer,
        render_data: Any,
        camera_data: CameraData,
        physics_step_count: int,
    ) -> None:
        """Sync scene state, render, and read outputs into ``camera_data``."""
        self.update_scene_state(physics_step_count)
        renderer.render(render_data)
        renderer.read_output(render_data, camera_data)

    def reset_stage_prepare_flag(self) -> None:
        """Allow :meth:`ensure_prepare_stage` to run ``prepare_stage`` again (e.g. a new USD stage)."""
        self._prepared_renderer_ids.clear()
        self._prepared_num_envs = None

    def reset_scene_state_cadence(self) -> None:
        """Clear per-step scene state update dedupe (e.g. a long pause with no physics)."""
        self._last_scene_state_step = None

    def close(self) -> None:
        """Close every registered backend and drop it from this context.

        Called from :meth:`~isaaclab.sim.simulation_context.SimulationContext.clear_instance` after
        cameras have released their render data and before the stage is torn down, so
        :meth:`BaseRenderer.close` runs while the stage is still alive. A backend that raises does
        not prevent the others from closing; the failure is reported once every backend has been
        given the chance. Idempotent.

        Raises:
            RuntimeError: If any backend's :meth:`BaseRenderer.close` raised.
        """
        errors: list[Exception] = []
        for writer in self._visual_material_writers:
            try:
                writer.close()
            except Exception as exc:  # noqa: BLE001 - reported after every resource is closed
                logger.error("Error closing visual-material writer: %s", exc)
                errors.append(exc)
        for _cfg, renderer in self._renderer_entries:
            try:
                renderer.close()
            except Exception as exc:  # noqa: BLE001 - re-raised below once every backend is closed
                logger.error("Error closing renderer %s: %s", type(renderer).__name__, exc)
                errors.append(exc)
        self._renderer_entries.clear()
        self._prepared_renderer_ids.clear()
        self._prepared_num_envs = None
        self._last_scene_state_step = None
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
            raise RuntimeError(f"{len(errors)} renderer(s) failed to close") from errors[0]
