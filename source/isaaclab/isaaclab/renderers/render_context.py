# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulation-scoped renderers for camera sensors."""

from __future__ import annotations

import logging
from typing import Any, cast

from isaaclab.sensors.camera.camera_data import CameraData

from .base_renderer import BaseRenderer
from .renderer import Renderer
from .renderer_cfg import RendererCfg

logger = logging.getLogger(__name__)


class RenderContext:
    """Holds :class:`BaseRenderer` instances for all :class:`Camera` sensors in a simulation.

    A camera reuses a backend when a prior camera registered a config equal under ``==`` (value
    equality) and the same concrete ``RendererCfg`` subclass. A distinct ``RendererCfg`` that
    maps to a different implementation (e.g. Isaac RTX vs Newton) produces another backend; each
    has :meth:`BaseRenderer.prepare_stage` run before use.

    :meth:`update_transforms` is invoked at most once per :meth:`get_physics_step_count` for the
    context;
    """

    __slots__ = (
        "_renderer_entries",
        "_prepared_renderer_ids",
        "_prepared_num_envs",
        "_last_transforms_step",
    )

    def __init__(self) -> None:
        self._renderer_entries: list[tuple[RendererCfg, BaseRenderer]] = []
        self._prepared_renderer_ids: set[int] = set()
        self._prepared_num_envs: int | None = None
        self._last_transforms_step: int | None = None

    def get_renderer(self, cfg: RendererCfg) -> BaseRenderer:
        """Return a backend for this configuration, reusing a matching instance if present.

        Lookups use ``==`` and concrete ``RendererCfg`` type, so :func:`hash` is not used (configs
        are typically not hashable).

        Args:
            cfg: Renderer configuration from the initializing camera.

        Returns:
            A shared or newly created renderer backend.
        """
        for stored_cfg, r in self._renderer_entries:
            if type(stored_cfg) is type(cfg) and stored_cfg == cfg:
                return r
        new_renderer = cast(BaseRenderer, Renderer(cfg))  # type: ignore[misc]
        self._renderer_entries.append((cfg, new_renderer))
        logger.info(
            "Created new renderer for simulation: %s",
            type(new_renderer).__name__,
        )
        return new_renderer

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

    def update_transforms(self, physics_step_count: int) -> None:
        """Call :meth:`BaseRenderer.update_transforms` on all backends (at most once per step)."""
        if not self._renderer_entries:
            return
        if self._last_transforms_step == physics_step_count:
            return
        for _cfg, renderer in self._renderer_entries:
            renderer.update_transforms()
        self._last_transforms_step = physics_step_count

    def render_into_camera(
        self,
        renderer: BaseRenderer,
        render_data: Any,
        camera_data: CameraData,
        physics_step_count: int,
    ) -> None:
        """Sync scene transforms, render, and read outputs into ``camera_data``."""
        self.update_transforms(physics_step_count)
        renderer.render(render_data)
        renderer.read_output(render_data, camera_data)

    def reset_stage_prepare_flag(self) -> None:
        """Allow :meth:`ensure_prepare_stage` to run ``prepare_stage`` again (e.g. a new USD stage)."""
        self._prepared_renderer_ids.clear()
        self._prepared_num_envs = None

    def reset_transform_cadence(self) -> None:
        """Clear per-step transform dedupe (e.g. a long pause with no physics)."""
        self._last_transforms_step = None
