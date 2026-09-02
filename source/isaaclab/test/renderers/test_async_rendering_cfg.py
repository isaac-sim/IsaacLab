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


@pytest.mark.parametrize("value", [None, 1.0], ids=["none", "float"])
def test_untyped_cfg_value_raises_the_documented_error(value):
    """Values outside bool/int/str (e.g. a Hydra ``null``) must raise ValueError, not AttributeError."""
    with pytest.raises(ValueError, match="boolean spelling"):
        resolve_async_rendering_enabled(RendererCfg(async_rendering=value))


@pytest.mark.parametrize(
    "raw,expected",
    [("0", False), ("false", False), ("off", False), ("1", True), ("true", True), ("yes", True), (" on ", True)],
)
def test_env_var_spellings(raw, expected, monkeypatch):
    monkeypatch.setenv(ASYNC_RENDERING_ENV_VAR, raw)

    assert resolve_async_rendering_enabled(RendererCfg(async_rendering=not expected)) is expected


@pytest.mark.parametrize("raw", ["", "   "])
def test_empty_env_var_means_unset(raw, monkeypatch):
    """An empty variable leaves the configured value in effect."""
    monkeypatch.setenv(ASYNC_RENDERING_ENV_VAR, raw)

    assert resolve_async_rendering_enabled(RendererCfg(async_rendering=True)) is True


@pytest.mark.parametrize("raw", ["banana", "1.5", "-1", "3", "yes please"])
def test_invalid_env_var_raises(raw, monkeypatch):
    """Invalid spellings raise. A silently ignored typo would degrade an async run to sync."""
    monkeypatch.setenv(ASYNC_RENDERING_ENV_VAR, raw)

    with pytest.raises(ValueError):
        resolve_async_rendering_enabled(RendererCfg())


def test_env_var_overrides_cfg(monkeypatch):
    monkeypatch.setenv(ASYNC_RENDERING_ENV_VAR, "1")

    assert resolve_async_rendering_enabled(RendererCfg(async_rendering=False)) is True


def test_env_var_can_disable_cfg(monkeypatch):
    monkeypatch.setenv(ASYNC_RENDERING_ENV_VAR, "0")

    assert resolve_async_rendering_enabled(RendererCfg(async_rendering=True)) is False


def test_invalid_cfg_raises_despite_env_override(monkeypatch):
    """The cfg value is validated even when the env var overrides it, so failures are not env-dependent."""
    monkeypatch.setenv(ASYNC_RENDERING_ENV_VAR, "1")

    with pytest.raises(ValueError, match="not supported yet"):
        resolve_async_rendering_enabled(RendererCfg(async_rendering=3))


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
