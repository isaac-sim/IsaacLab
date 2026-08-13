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


def test_zero_frames_selects_sync(monkeypatch):
    monkeypatch.delenv(ASYNC_RENDERING_ENV_VAR, raising=False)

    assert isinstance(_resolve_render_strategy(OVRTXRendererCfg(async_rendering=0)), _SyncRenderStrategy)


@pytest.mark.parametrize("frames", [1, 2, 5])
def test_frame_count_sets_queue_depth(frames, monkeypatch):
    """A frame count is latency, so the queue holds one more render than that."""
    monkeypatch.delenv(ASYNC_RENDERING_ENV_VAR, raising=False)

    strategy = _resolve_render_strategy(OVRTXRendererCfg(async_rendering=frames))

    assert isinstance(strategy, _AsyncRenderStrategy)
    assert strategy._render_queue_depth == frames + 1


def test_bool_matches_one_frame(monkeypatch):
    """``True`` is the same request as ``1``."""
    monkeypatch.delenv(ASYNC_RENDERING_ENV_VAR, raising=False)

    assert (
        _resolve_render_strategy(OVRTXRendererCfg(async_rendering=True))._render_queue_depth
        == _resolve_render_strategy(OVRTXRendererCfg(async_rendering=1))._render_queue_depth
    )


def test_env_override_sets_frame_count(sync_cfg, monkeypatch):
    monkeypatch.setenv(ASYNC_RENDERING_ENV_VAR, "3")

    assert _resolve_render_strategy(sync_cfg)._render_queue_depth == 4


def test_invalid_env_override_falls_back_to_cfg(async_cfg, monkeypatch):
    monkeypatch.setenv(ASYNC_RENDERING_ENV_VAR, "banana")

    assert isinstance(_resolve_render_strategy(async_cfg), _AsyncRenderStrategy)


def test_negative_frame_count_is_rejected(monkeypatch):
    monkeypatch.delenv(ASYNC_RENDERING_ENV_VAR, raising=False)

    with pytest.raises(ValueError):
        _resolve_render_strategy(OVRTXRendererCfg(async_rendering=-1))
