# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :class:`~isaaclab.renderers.render_context.RenderContext`."""

from __future__ import annotations

import pytest

from isaaclab.renderers.render_context import (
    RenderContext,
)
from isaaclab.renderers.renderer_cfg import RendererCfg
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
