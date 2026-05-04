# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :class:`~isaaclab.renderers.render_context.RenderContext`."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch

from isaaclab.renderers.base_renderer import BaseRenderer
from isaaclab.renderers.output_contract import RenderBufferKind, RenderBufferSpec
from isaaclab.renderers.render_context import RenderContext
from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.sensors.camera.camera_data import CameraData

pytest.importorskip("isaaclab_physx")
pytest.importorskip("isaaclab_newton")
pytest.importorskip("isaaclab_ov")

from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg


class _FakeBackend(BaseRenderer):
    """Test double for :class:`BaseRenderer`; does not load PhysX/Newton/OV renderer classes."""

    __slots__ = ("_prepare_hits", "_update_transforms_hits", "_event_log")

    def __init__(
        self,
        *,
        prepare_hits: list[int] | None = None,
        update_transforms_hits: list[int] | None = None,
        event_log: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._prepare_hits = prepare_hits
        self._update_transforms_hits = update_transforms_hits
        self._event_log = event_log

    def supported_output_types(self) -> dict[RenderBufferKind, RenderBufferSpec]:
        return {}

    def prepare_stage(self, stage: Any, num_envs: int) -> None:
        if self._prepare_hits is not None:
            self._prepare_hits.append(1)

    def create_render_data(self, spec: Any) -> Any:
        return object()

    def set_outputs(self, render_data: Any, output_data: dict[str, torch.Tensor]) -> None:
        pass

    def update_transforms(self) -> None:
        if self._update_transforms_hits is not None:
            self._update_transforms_hits.append(1)
        if self._event_log is not None:
            self._event_log.append("ut")

    def update_camera(
        self,
        render_data: Any,
        positions: torch.Tensor,
        orientations: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> None:
        pass

    def render(self, render_data: Any) -> None:
        if self._event_log is not None:
            self._event_log.append("render")

    def read_output(self, render_data: Any, camera_data: CameraData) -> None:
        if self._event_log is not None:
            self._event_log.append("read")

    def cleanup(self, render_data: Any) -> None:
        pass


def _set_entries(ctx: RenderContext, *cfg_backend_pairs: tuple[RendererCfg, BaseRenderer]) -> None:
    ctx._renderer_entries = list(cfg_backend_pairs)  # type: ignore[assignment]  # noqa: SLF001


@pytest.fixture(autouse=True)
def _patch_renderer_factory() -> Generator[None, None, None]:
    """Never construct :class:`~isaaclab.renderers.renderer.Renderer` (real backends) in this module."""

    with patch(
        "isaaclab.renderers.render_context.Renderer",
        side_effect=lambda *_args, **_kwargs: _FakeBackend(),
    ):
        yield


def test_get_renderer_returns_equal_cfg_singleton():
    ctx = RenderContext()
    cfg = IsaacRtxRendererCfg()
    r1 = ctx.get_renderer(cfg)
    r2 = ctx.get_renderer(cfg)
    assert r1 is r2


def test_get_renderer_two_different_concrete_types_coexist():
    """Different renderer_cfg concrete classes register distinct backends (no error)."""

    ctx = RenderContext()
    rtx = ctx.get_renderer(IsaacRtxRendererCfg())
    nw = ctx.get_renderer(NewtonWarpRendererCfg())
    assert rtx is not nw


def test_ensure_prepare_stage_idempotent():
    """Second ``ensure_prepare_stage`` with same args does not call ``prepare_stage`` again."""

    ctx = RenderContext()
    prepares: list[int] = []
    cfg = IsaacRtxRendererCfg()
    _set_entries(ctx, (cfg, _FakeBackend(prepare_hits=prepares)))

    ctx.ensure_prepare_stage(None, 4)
    ctx.ensure_prepare_stage(None, 4)
    assert len(prepares) == 1


def test_ensure_prepare_stage_num_envs_mismatch():
    ctx = RenderContext()
    cfg = IsaacRtxRendererCfg()
    _set_entries(ctx, (cfg, _FakeBackend()))

    ctx.ensure_prepare_stage(None, 4)
    with pytest.raises(RuntimeError, match="different num_envs"):
        ctx.ensure_prepare_stage(None, 8)


def test_update_transforms_dedupes_per_physics_step():
    """All backends' update_transforms run once per physics step index."""

    ctx = RenderContext()
    hits: list[int] = []
    cfg = NewtonWarpRendererCfg()
    _set_entries(ctx, (cfg, _FakeBackend(update_transforms_hits=hits)))

    ctx.update_transforms(1)
    ctx.update_transforms(1)
    assert len(hits) == 1

    ctx.update_transforms(2)
    assert len(hits) == 2


def test_render_into_camera_calls_update_render_read_order():
    """render_into_camera runs update_transforms then render then read_output; dedupes UT per step."""
    ctx = RenderContext()
    events: list[str] = []
    cfg = IsaacRtxRendererCfg()
    fake = _FakeBackend(event_log=events)
    _set_entries(ctx, (cfg, fake))

    rd = object()
    cam_data = CameraData()
    ctx.render_into_camera(cast(BaseRenderer, fake), rd, cam_data, physics_step_count=1)
    assert events == ["ut", "render", "read"]

    ctx.render_into_camera(cast(BaseRenderer, fake), rd, cam_data, physics_step_count=1)
    assert events == ["ut", "render", "read", "render", "read"]


def test_reset_stage_prepare_flag_allows_second_prepare_stage():
    """After reset_stage_prepare_flag, ensure_prepare_stage invokes prepare_stage again."""
    ctx = RenderContext()
    prepares: list[int] = []
    cfg = IsaacRtxRendererCfg()
    _set_entries(ctx, (cfg, _FakeBackend(prepare_hits=prepares)))

    ctx.ensure_prepare_stage(None, 4)
    assert len(prepares) == 1
    ctx.ensure_prepare_stage(None, 4)
    assert len(prepares) == 1

    ctx.reset_stage_prepare_flag()
    ctx.ensure_prepare_stage(None, 4)
    assert len(prepares) == 2


def test_reset_transform_cadence_allows_repeat_update_transforms_same_step():
    """reset_transform_cadence clears step dedupe so the same physics_step_count can sync again."""
    ctx = RenderContext()
    hits: list[int] = []
    cfg = IsaacRtxRendererCfg()
    _set_entries(ctx, (cfg, _FakeBackend(update_transforms_hits=hits)))

    ctx.update_transforms(1)
    assert len(hits) == 1
    ctx.update_transforms(1)
    assert len(hits) == 1

    ctx.reset_transform_cadence()
    ctx.update_transforms(1)
    assert len(hits) == 2
