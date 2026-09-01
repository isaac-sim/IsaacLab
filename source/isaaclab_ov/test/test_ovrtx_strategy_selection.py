# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for OVRTX render strategy selection."""

from __future__ import annotations

import importlib.util

import pytest

_REQUIRED_MODULES = ("isaaclab_ov", "ovrtx")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = [
    pytest.mark.isaacsim_ci,
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}",
    ),
]

if not _MISSING_MODULES:
    from isaaclab_ov.renderers.ovrtx_renderer import _resolve_render_strategy
    from isaaclab_ov.renderers.ovrtx_renderer_cfg import OVRTXRendererCfg
    from isaaclab_ov.renderers.ovrtx_renderer_strategies import _AsyncRenderStrategy, _SyncRenderStrategy

    from isaaclab.renderers import ASYNC_RENDERING_ENV_VAR


@pytest.fixture()
def async_cfg(monkeypatch) -> OVRTXRendererCfg:
    monkeypatch.delenv(ASYNC_RENDERING_ENV_VAR, raising=False)
    return OVRTXRendererCfg(async_rendering=True)


@pytest.fixture()
def sync_cfg(monkeypatch) -> OVRTXRendererCfg:
    monkeypatch.delenv(ASYNC_RENDERING_ENV_VAR, raising=False)
    return OVRTXRendererCfg(async_rendering=False)


def test_async_selected_when_enabled(async_cfg):
    assert isinstance(_resolve_render_strategy(async_cfg), _AsyncRenderStrategy)


def test_sync_selected_when_disabled(sync_cfg):
    assert isinstance(_resolve_render_strategy(sync_cfg), _SyncRenderStrategy)


def test_env_override_enables_async(sync_cfg, monkeypatch):
    monkeypatch.setenv(ASYNC_RENDERING_ENV_VAR, "1")

    assert isinstance(_resolve_render_strategy(sync_cfg), _AsyncRenderStrategy)


def test_env_override_disables_async(async_cfg, monkeypatch):
    monkeypatch.setenv(ASYNC_RENDERING_ENV_VAR, "0")

    assert isinstance(_resolve_render_strategy(async_cfg), _SyncRenderStrategy)


def test_async_pipelines_exactly_one_frame(async_cfg):
    """One frame of latency: the queue holds that render plus the one being enqueued.

    Deeper queues are deliberately unsupported — the ovstage path cannot sustain them because its
    scene writes drain in-flight renders, and the legacy path has not measured a benefit.
    """
    strategy = _resolve_render_strategy(async_cfg)

    assert isinstance(strategy, _AsyncRenderStrategy)
    assert strategy._render_queue_depth == 2


def test_multi_frame_latency_is_rejected(monkeypatch):
    """Frame counts above one are refused explicitly rather than silently truncated."""
    monkeypatch.delenv(ASYNC_RENDERING_ENV_VAR, raising=False)

    with pytest.raises(ValueError, match="not supported yet"):
        _resolve_render_strategy(OVRTXRendererCfg(async_rendering=3))


def test_multi_frame_env_override_is_ignored(async_cfg, monkeypatch):
    """A frame count in the env var is not a boolean spelling, so it is ignored with a warning."""
    monkeypatch.setenv(ASYNC_RENDERING_ENV_VAR, "3")

    assert isinstance(_resolve_render_strategy(async_cfg), _AsyncRenderStrategy)


def test_invalid_env_override_falls_back_to_cfg(async_cfg, monkeypatch):
    monkeypatch.setenv(ASYNC_RENDERING_ENV_VAR, "banana")

    assert isinstance(_resolve_render_strategy(async_cfg), _AsyncRenderStrategy)


def test_negative_value_is_rejected(monkeypatch):
    monkeypatch.delenv(ASYNC_RENDERING_ENV_VAR, raising=False)

    with pytest.raises(ValueError):
        _resolve_render_strategy(OVRTXRendererCfg(async_rendering=-1))


def test_ovstage_forces_sync_and_warns(async_cfg, caplog):
    """The ovstage path renders synchronously for now; a requested async must warn, not silently apply."""
    with caplog.at_level("WARNING"):
        strategy = _resolve_render_strategy(async_cfg, use_ovstage=True)

    assert isinstance(strategy, _SyncRenderStrategy)
    assert any("ovstage" in record.message for record in caplog.records)


def test_ovstage_sync_is_silent(sync_cfg, caplog):
    """No warning when async was never requested on the ovstage path."""
    with caplog.at_level("WARNING"):
        strategy = _resolve_render_strategy(sync_cfg, use_ovstage=True)

    assert isinstance(strategy, _SyncRenderStrategy)
    assert not caplog.records
