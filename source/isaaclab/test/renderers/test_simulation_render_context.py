# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for RenderContext."""

from __future__ import annotations

import pytest

from isaaclab.renderers.render_context import (
    RenderContext,
    renderer_cfgs_compatible,
)
from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg


def test_renderer_cfgs_compatible_same_class():
    a = IsaacRtxRendererCfg()
    b = IsaacRtxRendererCfg()
    assert renderer_cfgs_compatible(a, b)


def test_renderer_cfgs_compatible_different_class():
    a = IsaacRtxRendererCfg()
    b = RendererCfg()
    assert not renderer_cfgs_compatible(a, b)


def test_get_renderer_returns_singleton():
    ctx = RenderContext()
    cfg = IsaacRtxRendererCfg()
    r1 = ctx.get_renderer(cfg)
    r2 = ctx.get_renderer(cfg)
    assert r1 is r2
    assert ctx.renderer is r1


def test_get_renderer_rejects_incompatible_cfg():
    pytest.importorskip("isaaclab_newton")
    from isaaclab_newton.renderers import NewtonWarpRendererCfg

    ctx = RenderContext()
    ctx.get_renderer(IsaacRtxRendererCfg())
    with pytest.raises(RuntimeError, match="same concrete renderer"):
        ctx.get_renderer(NewtonWarpRendererCfg())


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


def test_maybe_update_transforms_dedupes_newton():
    """NewtonWarpRenderer.update_transforms runs once per physics step index."""
    pytest.importorskip("isaaclab_newton")
    from isaaclab_newton.renderers import NewtonWarpRendererCfg

    ctx = RenderContext()
    calls: list[int] = []

    class FakeNewton:
        @property
        def uses_global_scene_transform_sync(self) -> bool:
            return True

        def prepare_stage(self, stage, num_envs):
            pass

        def _create_render_data_impl(self, spec):
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
    ctx._renderer = FakeNewton()  # type: ignore[assignment]  # noqa: SLF001
    ctx._canonical_cfg = NewtonWarpRendererCfg()  # noqa: SLF001

    ctx.maybe_update_transforms(1)
    ctx.maybe_update_transforms(1)
    assert len(calls) == 1

    ctx.maybe_update_transforms(2)
    assert len(calls) == 2
