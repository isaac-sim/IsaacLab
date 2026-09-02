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


def test_strategy_matches_cfg_and_env(monkeypatch):
    """The cfg selects the strategy, and the environment variable overrides the cfg."""
    assert isinstance(_resolve_render_strategy(OVRTXRendererCfg(async_rendering=True)), _AsyncRenderStrategy)
    assert isinstance(_resolve_render_strategy(OVRTXRendererCfg(async_rendering=False)), _SyncRenderStrategy)

    monkeypatch.setenv(ASYNC_RENDERING_ENV_VAR, "1")
    assert isinstance(_resolve_render_strategy(OVRTXRendererCfg(async_rendering=False)), _AsyncRenderStrategy)


def test_ovstage_forces_sync_and_warns_only_when_async_requested(caplog):
    """The ovstage path renders synchronously for now. A requested async must warn, not silently apply."""
    with caplog.at_level("WARNING"):
        strategy = _resolve_render_strategy(OVRTXRendererCfg(async_rendering=False), use_ovstage=True)
    assert isinstance(strategy, _SyncRenderStrategy)
    assert not caplog.records

    with caplog.at_level("WARNING"):
        strategy = _resolve_render_strategy(OVRTXRendererCfg(async_rendering=True), use_ovstage=True)
    assert isinstance(strategy, _SyncRenderStrategy)
    assert any("ovstage" in record.message for record in caplog.records)
