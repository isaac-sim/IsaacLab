# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :class:`~isaaclab.renderers.render_context.RenderContext`."""

from __future__ import annotations

from typing import cast

import pytest

from isaaclab.renderers.base_renderer import BaseRenderer
from isaaclab.renderers.render_context import RenderContext
from isaaclab.sensors.camera.camera_data import CameraData
from isaaclab_physx.renderers import IsaacRtxRendererCfg


def test_get_renderer_returns_equal_cfg_singleton():
    ctx = RenderContext()
    cfg = IsaacRtxRendererCfg()
    r1 = ctx.get_renderer(cfg)
    r2 = ctx.get_renderer(cfg)
    assert r1 is r2
    assert ctx.renderer is r1
    assert len(ctx.renderers) == 1


def test_get_renderer_two_different_concrete_types_coexist():
    """Different renderer_cfg concrete classes register distinct backends (no error)."""
    pytest.importorskip("isaaclab_newton")
    from isaaclab_newton.renderers import NewtonWarpRendererCfg

    ctx = RenderContext()
    rtx = ctx.get_renderer(IsaacRtxRendererCfg())
    nw = ctx.get_renderer(NewtonWarpRendererCfg())
    assert rtx is not nw
    assert len(ctx.renderers) == 2
    assert ctx.renderer is None


def test_ensure_prepare_stage_idempotent():
    ctx = RenderContext()
    ctx.get_renderer(IsaacRtxRendererCfg())
    ctx.ensure_prepare_stage(None, 4)
    ctx.ensure_prepare_stage(None, 4)


def test_ensure_prepare_stage_num_envs_mismatch():
    ctx = RenderContext()
    ctx.get_renderer(IsaacRtxRendererCfg())
    ctx.ensure_prepare_stage(None, 4)
    with pytest.raises(RuntimeError, match="different num_envs"):
        ctx.ensure_prepare_stage(None, 8)


def test_update_transforms_dedupes_per_physics_step():
    """All backends' update_transforms run once per physics step index."""
    pytest.importorskip("isaaclab_newton")
    from isaaclab_newton.renderers import NewtonWarpRendererCfg

    ctx = RenderContext()
    calls: list[int] = []

    class FakeNewton:
        def prepare_stage(self, stage, num_envs):
            pass

        def create_render_data(self, spec):
            return object()

        def set_outputs(self, render_data, output_data):
            pass

        def update_camera(self, render_data, positions, orientations, intrinsics):
            pass

        def render(self, render_data):
            pass

        def read_output(self, render_data, camera_data):
            pass

        def cleanup(self, render_data):
            pass

        def update_transforms(self):
            calls.append(1)

    FakeNewton.__name__ = "NewtonWarpRenderer"
    cfg = NewtonWarpRendererCfg()
    ctx._renderer_entries = [(cfg, FakeNewton())]  # type: ignore[assignment]  # noqa: SLF001

    ctx.update_transforms(1)
    ctx.update_transforms(1)
    assert len(calls) == 1

    ctx.update_transforms(2)
    assert len(calls) == 2


def test_render_into_camera_calls_update_render_read_order():
    """render_into_camera runs update_transforms then render then read_output; dedupes UT per step."""
    ctx = RenderContext()
    events: list[str] = []

    class FakeRenderer:
        def prepare_stage(self, stage, num_envs):
            pass

        def create_render_data(self, spec):
            return object()

        def set_outputs(self, render_data, output_data):
            pass

        def update_camera(self, render_data, positions, orientations, intrinsics):
            pass

        def render(self, render_data):
            events.append("render")

        def read_output(self, render_data, camera_data):
            events.append("read")

        def cleanup(self, render_data):
            pass

        def update_transforms(self):
            events.append("ut")

    cfg = IsaacRtxRendererCfg()
    fake = FakeRenderer()
    ctx._renderer_entries = [(cfg, fake)]  # type: ignore[assignment]  # noqa: SLF001

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

    class FakeRenderer:
        def prepare_stage(self, stage, num_envs):
            prepares.append(1)

        def create_render_data(self, spec):
            return object()

        def set_outputs(self, render_data, output_data):
            pass

        def update_camera(self, render_data, positions, orientations, intrinsics):
            pass

        def render(self, render_data):
            pass

        def read_output(self, render_data, camera_data):
            pass

        def cleanup(self, render_data):
            pass

        def update_transforms(self):
            pass

    cfg = IsaacRtxRendererCfg()
    ctx._renderer_entries = [(cfg, FakeRenderer())]  # type: ignore[assignment]  # noqa: SLF001

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
    calls: list[int] = []

    class FakeRenderer:
        def prepare_stage(self, stage, num_envs):
            pass

        def create_render_data(self, spec):
            return object()

        def set_outputs(self, render_data, output_data):
            pass

        def update_camera(self, render_data, positions, orientations, intrinsics):
            pass

        def render(self, render_data):
            pass

        def read_output(self, render_data, camera_data):
            pass

        def cleanup(self, render_data):
            pass

        def update_transforms(self):
            calls.append(1)

    cfg = IsaacRtxRendererCfg()
    ctx._renderer_entries = [(cfg, FakeRenderer())]  # type: ignore[assignment]  # noqa: SLF001

    ctx.update_transforms(1)
    assert len(calls) == 1
    ctx.update_transforms(1)
    assert len(calls) == 1

    ctx.reset_transform_cadence()
    ctx.update_transforms(1)
    assert len(calls) == 2
