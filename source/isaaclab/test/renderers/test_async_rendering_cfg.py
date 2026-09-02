# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for resolving the renderer-agnostic ``async_rendering`` setting."""

import logging

import pytest

from isaaclab.renderers import (
    ASYNC_RENDERING_ENV_VAR,
    RendererCfg,
    resolve_async_rendering_enabled,
    warn_unsupported_async_rendering,
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv(ASYNC_RENDERING_ENV_VAR, raising=False)


def test_cfg_enables_async_rendering():
    assert resolve_async_rendering_enabled(RendererCfg()) is False
    assert resolve_async_rendering_enabled(RendererCfg(async_rendering=True)) is True


@pytest.mark.parametrize("raw,expected", [("1", True), ("false", False), (" on ", True)])
def test_env_var_overrides_cfg(raw, expected, monkeypatch):
    """The variable wins over the configuration and accepts the usual boolean spellings."""
    monkeypatch.setenv(ASYNC_RENDERING_ENV_VAR, raw)

    assert resolve_async_rendering_enabled(RendererCfg(async_rendering=not expected)) is expected


@pytest.mark.parametrize("cfg_value", [2, None], ids=["multi-frame", "hydra-null"])
def test_invalid_cfg_value_is_rejected(cfg_value, monkeypatch):
    """Multi-frame latency and non-boolean values raise, even when the env var overrides the cfg."""
    monkeypatch.setenv(ASYNC_RENDERING_ENV_VAR, "1")

    with pytest.raises(ValueError):
        resolve_async_rendering_enabled(RendererCfg(async_rendering=cfg_value))


def test_invalid_env_var_is_rejected(monkeypatch):
    """A typo raises instead of silently degrading an async run to a synchronous one."""
    monkeypatch.setenv(ASYNC_RENDERING_ENV_VAR, "banana")

    with pytest.raises(ValueError):
        resolve_async_rendering_enabled(RendererCfg())


def test_unsupported_renderer_warns_only_when_requested(caplog):
    with caplog.at_level(logging.WARNING):
        warn_unsupported_async_rendering(RendererCfg(), "newton_warp")
    assert caplog.text == ""

    with caplog.at_level(logging.WARNING):
        warn_unsupported_async_rendering(RendererCfg(async_rendering=True), "newton_warp")
    assert "newton_warp" in caplog.text
