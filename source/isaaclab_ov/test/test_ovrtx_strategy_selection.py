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


@pytest.fixture()
def async_cfg(monkeypatch) -> OVRTXRendererCfg:
    monkeypatch.delenv("OVRTX_ASYNC_RENDERING", raising=False)
    return OVRTXRendererCfg(async_rendering=True)


@pytest.fixture()
def sync_cfg(monkeypatch) -> OVRTXRendererCfg:
    monkeypatch.delenv("OVRTX_ASYNC_RENDERING", raising=False)
    return OVRTXRendererCfg(async_rendering=False)


def test_async_selected_when_enabled(async_cfg):
    assert isinstance(_resolve_render_strategy(async_cfg), _AsyncRenderStrategy)


def test_sync_selected_when_disabled(sync_cfg):
    assert isinstance(_resolve_render_strategy(sync_cfg), _SyncRenderStrategy)


def test_env_override_enables_async(sync_cfg, monkeypatch):
    monkeypatch.setenv("OVRTX_ASYNC_RENDERING", "1")

    assert isinstance(_resolve_render_strategy(sync_cfg), _AsyncRenderStrategy)


def test_env_override_disables_async(async_cfg, monkeypatch):
    monkeypatch.setenv("OVRTX_ASYNC_RENDERING", "0")

    assert isinstance(_resolve_render_strategy(async_cfg), _SyncRenderStrategy)
