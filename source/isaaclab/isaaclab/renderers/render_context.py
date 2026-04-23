# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulation-scoped shared renderer for camera sensors."""

from __future__ import annotations

import logging
from typing import Any, cast

from isaaclab.sensors.camera.camera_data import CameraData

from .base_renderer import BaseRenderer
from .renderer import Renderer
from .renderer_cfg import RendererCfg

logger = logging.getLogger(__name__)


def renderer_cfgs_compatible(a: RendererCfg, b: RendererCfg) -> bool:
    """Return True if two camera renderer configs may share one BaseRenderer.

    Args:
        a: Renderer configuration from the first camera using this context.
        b: Renderer configuration from another camera.

    Returns:
        Whether both configs use the same concrete class and ``renderer_type``.
    """
    if type(a) is not type(b):
        return False
    return getattr(a, "renderer_type", None) == getattr(b, "renderer_type", None)


class RenderContext:
    """Owns one Renderer / BaseRenderer for all scene cameras.

    Every Camera with a compatible ``renderer_cfg`` shares the same backend.
    ``prepare_stage`` runs once. For backends with
    :attr:`~isaaclab.renderers.base_renderer.BaseRenderer.uses_global_scene_transform_sync`,
    ``update_transforms`` runs at most once per physics step (see ``physics_step_count``).

    Mixing incompatible ``renderer_cfg`` in one simulation raises RuntimeError.
    """

    __slots__ = (
        "_renderer",
        "_canonical_cfg",
        "_stage_prepared",
        "_prepared_num_envs",
        "_last_transforms_step",
    )

    def __init__(self) -> None:
        self._renderer: BaseRenderer | None = None
        self._canonical_cfg: RendererCfg | None = None
        self._stage_prepared: bool = False
        self._prepared_num_envs: int | None = None
        self._last_transforms_step: int | None = None

    @property
    def renderer(self) -> BaseRenderer | None:
        """Shared backend, or None if no camera requested a renderer yet."""
        return self._renderer

    def get_renderer(self, cfg: RendererCfg) -> BaseRenderer:
        """Return the shared BaseRenderer, creating it on first use.

        Args:
            cfg: Renderer configuration from the initializing camera.

        Returns:
            Shared renderer backend.

        Raises:
            RuntimeError: If cfg is incompatible with an existing shared renderer.
        """
        if self._renderer is None:
            self._canonical_cfg = cfg
            # Renderer.__new__ returns a BaseRenderer implementation.
            self._renderer = cast(BaseRenderer, Renderer(cfg))  # type: ignore[misc]
            logger.info(
                "Created shared simulation renderer: %s",
                type(self._renderer).__name__,
            )
            return self._renderer
        if self._canonical_cfg is None or not renderer_cfgs_compatible(
            self._canonical_cfg, cfg
        ):
            ex_t = type(self._canonical_cfg).__name__
            ex_r = getattr(self._canonical_cfg, "renderer_type", None)
            rq_t = type(cfg).__name__
            rq_r = getattr(cfg, "renderer_type", None)
            raise RuntimeError(
                "All Camera sensors must use the same concrete renderer configuration "
                "class and renderer_type when sharing the simulation renderer. "
                f"Existing: {ex_t} ({ex_r!r}); this camera requested: {rq_t} ({rq_r!r})."
            )
        return self._renderer

    def ensure_prepare_stage(self, stage: Any, num_envs: int) -> None:
        """Call BaseRenderer.prepare_stage once for this simulation.

        Args:
            stage: USD stage passed to the backend.
            num_envs: Environment count passed to the backend.

        Raises:
            RuntimeError: If get_renderer was never called, or num_envs disagrees
                with a previous successful prepare_stage.
        """
        if self._renderer is None:
            raise RuntimeError("get_renderer must be called before ensure_prepare_stage.")
        if not self._stage_prepared:
            self._renderer.prepare_stage(stage, num_envs)
            self._stage_prepared = True
            self._prepared_num_envs = num_envs
            return
        if self._prepared_num_envs != num_envs:
            raise RuntimeError(
                "Shared renderer prepare_stage was already called with a different "
                f"num_envs ({self._prepared_num_envs} vs {num_envs})."
            )

    def maybe_update_transforms(self, physics_step_count: int) -> None:
        """Call update_transforms at most once per physics step when needed.

        Isaac RTX uses a no-op; Newton and OVRTX sync shared scene state.

        Args:
            physics_step_count: Monotonic counter from SimulationContext (see
                get_physics_step_count).
        """
        if self._renderer is None:
            return
        if not self._renderer.uses_global_scene_transform_sync:
            self._renderer.update_transforms()
            return
        if self._last_transforms_step == physics_step_count:
            return
        self._renderer.update_transforms()
        self._last_transforms_step = physics_step_count

    def render_into_camera(
        self,
        renderer: BaseRenderer,
        render_data: Any,
        camera_data: CameraData,
        physics_step_count: int,
    ) -> None:
        """Sync scene transforms (if needed), render, and copy outputs into ``camera_data``."""
        self.maybe_update_transforms(physics_step_count)
        renderer.render(render_data)
        renderer.read_output(render_data, camera_data)

    def reset_stage_prepare_flag(self) -> None:
        """Allow ensure_prepare_stage to run prepare_stage again (e.g. new USD stage)."""
        self._stage_prepared = False
        self._prepared_num_envs = None

    def reset_transform_cadence(self) -> None:
        """Clear per-step transform dedupe (e.g. after a long pause with no physics)."""
        self._last_transforms_step = None
