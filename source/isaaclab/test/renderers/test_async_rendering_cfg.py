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
    async_rendering_enabled_from_env,
    resolve_async_rendering_enabled,
    warn_unsupported_async_rendering,
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv(ASYNC_RENDERING_ENV_VAR, raising=False)


@pytest.mark.parametrize("value,expected", [(False, False), (0, False), (True, True), (1, True)])
def test_cfg_value_resolves_to_flag(value, expected):
    """Boolean spellings and their 0/1 integer shorthands resolve to the flag."""
    assert resolve_async_rendering_enabled(RendererCfg(async_rendering=value)) is expected


def test_default_is_synchronous():
    assert resolve_async_rendering_enabled(RendererCfg()) is False


def test_multi_frame_latency_is_rejected():
    """Frame counts above one are refused explicitly: multi-frame latency is future work."""
    with pytest.raises(ValueError, match="not supported yet"):
        resolve_async_rendering_enabled(RendererCfg(async_rendering=4))


def test_negative_value_is_rejected():
    with pytest.raises(ValueError):
        resolve_async_rendering_enabled(RendererCfg(async_rendering=-1))


@pytest.mark.parametrize(
    "raw,expected",
    [("0", False), ("false", False), ("off", False), ("1", True), ("true", True), ("yes", True), (" on ", True)],
)
def test_env_var_spellings(raw, expected, monkeypatch):
    monkeypatch.setenv(ASYNC_RENDERING_ENV_VAR, raw)

    assert async_rendering_enabled_from_env() is expected


@pytest.mark.parametrize("raw", ["", "   ", "banana", "1.5", "-1", "3"])
def test_unusable_env_var_is_ignored(raw, monkeypatch):
    """Non-boolean spellings are ignored with a warning — including frame counts above one."""
    monkeypatch.setenv(ASYNC_RENDERING_ENV_VAR, raw)

    assert async_rendering_enabled_from_env() is None


def test_env_var_overrides_cfg(monkeypatch):
    monkeypatch.setenv(ASYNC_RENDERING_ENV_VAR, "1")

    assert resolve_async_rendering_enabled(RendererCfg(async_rendering=False)) is True


def test_env_var_can_disable_cfg(monkeypatch):
    monkeypatch.setenv(ASYNC_RENDERING_ENV_VAR, "0")

    assert resolve_async_rendering_enabled(RendererCfg(async_rendering=True)) is False


def test_invalid_env_var_leaves_cfg_in_effect(monkeypatch, caplog):
    monkeypatch.setenv(ASYNC_RENDERING_ENV_VAR, "banana")

    with caplog.at_level(logging.WARNING):
        assert resolve_async_rendering_enabled(RendererCfg(async_rendering=True)) is True
    assert ASYNC_RENDERING_ENV_VAR in caplog.text


def test_unsupported_renderer_warns_when_requested(caplog):
    with caplog.at_level(logging.WARNING):
        warn_unsupported_async_rendering(RendererCfg(async_rendering=True), "newton_warp")

    assert "newton_warp" in caplog.text
    assert "not implemented" in caplog.text


def test_unsupported_renderer_is_silent_when_synchronous(caplog):
    with caplog.at_level(logging.WARNING):
        warn_unsupported_async_rendering(RendererCfg(async_rendering=False), "newton_warp")

    assert caplog.text == ""


def test_unsupported_renderer_warns_for_env_request(monkeypatch, caplog):
    """The env var reaches renderers that cannot honor it, so it must warn too."""
    monkeypatch.setenv(ASYNC_RENDERING_ENV_VAR, "1")

    with caplog.at_level(logging.WARNING):
        warn_unsupported_async_rendering(RendererCfg(), "isaac_rtx")

    assert "isaac_rtx" in caplog.text
