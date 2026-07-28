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

pytestmark = [pytest.mark.integration, pytest.mark.rendering]


class _FakeBackend(BaseRenderer):
    """Test double for :class:`BaseRenderer`; does not load PhysX/Newton/OV renderer classes."""

    __slots__ = (
        "_prepare_hits",
        "_update_transforms_hits",
        "_update_geometries_hits",
        "_event_log",
        "_cleanup_log",
        "_cleanup_raises",
    )

    def __init__(
        self,
        *,
        prepare_hits: list[int] | None = None,
        update_transforms_hits: list[int] | None = None,
        update_geometries_hits: list[int] | None = None,
        event_log: list[str] | None = None,
        cleanup_log: list[Any] | None = None,
        cleanup_raises: bool = False,
    ) -> None:
        super().__init__()
        self._prepare_hits = prepare_hits
        self._update_transforms_hits = update_transforms_hits
        self._update_geometries_hits = update_geometries_hits
        self._event_log = event_log
        self._cleanup_log = cleanup_log
        self._cleanup_raises = cleanup_raises

    def supported_output_types(self) -> dict[RenderBufferKind, RenderBufferSpec]:
        return {}

    def prepare_stage(self, stage: Any, num_envs: int) -> None:
        if self._prepare_hits is not None:
            self._prepare_hits.append(1)

    def create_render_data(self, spec: Any) -> Any:
        return object()

    def set_outputs(self, render_data: Any, output_data: Any) -> None:
        pass

    def update_transforms(self) -> None:
        if self._update_transforms_hits is not None:
            self._update_transforms_hits.append(1)
        if self._event_log is not None:
            self._event_log.append("ut")

    def update_geometries(self) -> None:
        if self._update_geometries_hits is not None:
            self._update_geometries_hits.append(1)
        if self._event_log is not None:
            self._event_log.append("geo")

    def update_camera(self, render_data: Any, positions: Any, orientations: Any, intrinsics: Any) -> None:
        pass

    def render(self, render_data: Any) -> None:
        if self._event_log is not None:
            self._event_log.append("render")

    def read_output(self, render_data: Any, camera_data: CameraData) -> None:
        if self._event_log is not None:
            self._event_log.append("read")

    def cleanup(self, render_data: Any) -> None:
        if self._cleanup_log is not None:
            self._cleanup_log.append(render_data)
        if self._cleanup_raises:
            raise RuntimeError("backend cleanup failed")


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


def test_update_scene_state_dedupes_per_physics_step():
    """All backends' scene state hooks run once per physics step index."""

    ctx = RenderContext()
    transform_hits: list[int] = []
    geometry_hits: list[int] = []
    cfg = NewtonWarpRendererCfg()
    _set_entries(
        ctx,
        (
            cfg,
            _FakeBackend(update_transforms_hits=transform_hits, update_geometries_hits=geometry_hits),
        ),
    )

    ctx.update_scene_state(1)
    ctx.update_scene_state(1)
    assert len(transform_hits) == 1
    assert len(geometry_hits) == 1

    ctx.update_scene_state(2)
    assert len(transform_hits) == 2
    assert len(geometry_hits) == 2


def test_render_into_camera_calls_update_render_read_order():
    """render_into_camera runs scene sync then render then read_output; dedupes sync per step."""
    ctx = RenderContext()
    events: list[str] = []
    cfg = IsaacRtxRendererCfg()
    fake = _FakeBackend(event_log=events)
    _set_entries(ctx, (cfg, fake))

    rd = object()
    cam_data = CameraData()
    ctx.render_into_camera(cast(BaseRenderer, fake), rd, cam_data, physics_step_count=1)
    assert events == ["ut", "geo", "render", "read"]

    ctx.render_into_camera(cast(BaseRenderer, fake), rd, cam_data, physics_step_count=1)
    assert events == ["ut", "geo", "render", "read", "render", "read"]


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


def test_reset_scene_state_cadence_allows_repeat_update_scene_state_same_step():
    """reset_scene_state_cadence clears step dedupe so the same physics_step_count can update again."""
    ctx = RenderContext()
    hits: list[int] = []
    cfg = IsaacRtxRendererCfg()
    _set_entries(ctx, (cfg, _FakeBackend(update_transforms_hits=hits)))

    ctx.update_scene_state(1)
    assert len(hits) == 1
    ctx.update_scene_state(1)
    assert len(hits) == 1

    ctx.reset_scene_state_cadence()
    ctx.update_scene_state(1)
    assert len(hits) == 2


def test_close_releases_render_data_created_through_context():
    """close releases every render data handed out by create_render_data."""
    ctx = RenderContext()
    cleaned: list[Any] = []
    backend = _FakeBackend(cleanup_log=cleaned)
    _set_entries(ctx, (IsaacRtxRendererCfg(), backend))

    first = ctx.create_render_data(backend, cast(Any, object()))
    second = ctx.create_render_data(backend, cast(Any, object()))

    ctx.close()

    assert cleaned == [first, second]


def test_close_releases_render_data_while_a_reference_survives():
    """close releases render data even when the creating owner is still referenced.

    Regression test for the shutdown crash where a camera kept in a task attribute (in
    addition to the scene's own entry) never reached a zero reference count, so its
    finalizer never ran and its renderer resources stayed registered until the app tore
    down. Release must be driven by :meth:`RenderContext.close`, not by reference counting.
    """
    ctx = RenderContext()
    cleaned: list[Any] = []
    backend = _FakeBackend(cleanup_log=cleaned)
    _set_entries(ctx, (IsaacRtxRendererCfg(), backend))

    render_data = ctx.create_render_data(backend, cast(Any, object()))
    surviving_owner = {"camera_like": render_data}  # stands in for the extra reference

    ctx.close()

    assert cleaned == [render_data]
    assert surviving_owner["camera_like"] is render_data


def test_close_is_idempotent():
    """A second close does not release the same render data twice."""
    ctx = RenderContext()
    cleaned: list[Any] = []
    backend = _FakeBackend(cleanup_log=cleaned)
    _set_entries(ctx, (IsaacRtxRendererCfg(), backend))
    ctx.create_render_data(backend, cast(Any, object()))

    ctx.close()
    ctx.close()

    assert len(cleaned) == 1


def test_close_continues_after_a_backend_raises():
    """One backend failing to clean up does not strand the others."""
    ctx = RenderContext()
    cleaned: list[Any] = []
    failing = _FakeBackend(cleanup_raises=True)
    healthy = _FakeBackend(cleanup_log=cleaned)
    _set_entries(ctx, (IsaacRtxRendererCfg(), failing), (NewtonWarpRendererCfg(), healthy))

    ctx.create_render_data(failing, cast(Any, object()))
    healthy_data = ctx.create_render_data(healthy, cast(Any, object()))

    ctx.close()

    assert cleaned == [healthy_data]


def test_close_drops_registered_backends():
    """close clears the backend registry so a stale renderer cannot outlive the stage."""
    ctx = RenderContext()
    backend = _FakeBackend()
    _set_entries(ctx, (IsaacRtxRendererCfg(), backend))

    ctx.close()

    assert ctx._renderer_entries == []  # noqa: SLF001


def test_close_does_not_release_render_data_already_released_by_its_owner():
    """A handle the camera released on its own is not released a second time by close.

    A camera releases its render data when physics stops
    (``Camera._invalidate_initialize_callback``) and reports it via
    :meth:`RenderContext.release_render_data`. Without that, the context would still hold the
    handle and hand it to ``cleanup`` again at teardown -- releasing it twice.
    """
    ctx = RenderContext()
    cleaned: list[Any] = []
    backend = _FakeBackend(cleanup_log=cleaned)
    _set_entries(ctx, (IsaacRtxRendererCfg(), backend))
    owned = ctx.create_render_data(backend, cast(Any, object()))
    still_held = ctx.create_render_data(backend, cast(Any, object()))

    ctx.release_render_data(owned)  # the camera released this one itself
    ctx.close()

    assert cleaned == [still_held]
