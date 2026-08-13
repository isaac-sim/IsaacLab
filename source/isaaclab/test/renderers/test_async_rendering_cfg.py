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
    async_rendering_frames_from_env,
    resolve_async_rendering_frames,
    warn_unsupported_async_rendering,
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv(ASYNC_RENDERING_ENV_VAR, raising=False)


@pytest.mark.parametrize(
    "value,expected_frames",
    [(False, 0), (0, 0), (True, 1), (1, 1), (4, 4)],
)
def test_cfg_value_resolves_to_frames(value, expected_frames):
    """Booleans collapse onto the 0/1 frame counts they are shorthand for."""
    assert resolve_async_rendering_frames(RendererCfg(async_rendering=value)) == expected_frames


def test_default_is_synchronous():
    assert resolve_async_rendering_frames(RendererCfg()) == 0


def test_negative_frame_count_is_rejected():
    with pytest.raises(ValueError):
        resolve_async_rendering_frames(RendererCfg(async_rendering=-1))


@pytest.mark.parametrize(
    "raw,expected_frames",
    [("0", 0), ("false", 0), ("off", 0), ("1", 1), ("true", 1), ("yes", 1), ("3", 3), (" 2 ", 2)],
)
def test_env_var_spellings(raw, expected_frames, monkeypatch):
    monkeypatch.setenv(ASYNC_RENDERING_ENV_VAR, raw)

    assert async_rendering_frames_from_env() == expected_frames


@pytest.mark.parametrize("raw", ["", "   ", "banana", "1.5", "-1"])
def test_unusable_env_var_is_ignored(raw, monkeypatch):
    monkeypatch.setenv(ASYNC_RENDERING_ENV_VAR, raw)

    assert async_rendering_frames_from_env() is None


def test_env_var_overrides_cfg(monkeypatch):
    monkeypatch.setenv(ASYNC_RENDERING_ENV_VAR, "2")

    assert resolve_async_rendering_frames(RendererCfg(async_rendering=False)) == 2


def test_env_var_can_disable_cfg(monkeypatch):
    monkeypatch.setenv(ASYNC_RENDERING_ENV_VAR, "0")

    assert resolve_async_rendering_frames(RendererCfg(async_rendering=5)) == 0


def test_invalid_env_var_leaves_cfg_in_effect(monkeypatch, caplog):
    monkeypatch.setenv(ASYNC_RENDERING_ENV_VAR, "banana")

    with caplog.at_level(logging.WARNING):
        assert resolve_async_rendering_frames(RendererCfg(async_rendering=2)) == 2
    assert ASYNC_RENDERING_ENV_VAR in caplog.text


def test_unsupported_renderer_warns_when_requested(caplog):
    with caplog.at_level(logging.WARNING):
        warn_unsupported_async_rendering(RendererCfg(async_rendering=1), "newton_warp")

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
