# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the OVRTX driver shader-cache redirect.

The redirect runs before renderer construction and its only observable effect is
which directory the driver writes to, so a broken redirect otherwise surfaces as
nothing louder than a slow test. These cover that boundary with a fake settings
applier: no GPU, no ovrtx runtime, no renderer.
"""

import pytest
from isaaclab_ov.renderers import ovrtx_shader_cache
from isaaclab_ov.renderers.ovrtx_shader_cache import (
    SHADER_CACHE_PATH_ENV,
    apply_shader_cache_settings,
    redirect_shader_cache,
)

_CACHE_PATH = "/tmp/isaaclab-ovrtx-kitless-cache"

# Only ever forwarded to the patched applier factory, so its contents never matter.
_CONFIG = object()


class _RecordingApplier:
    """Settings applier that records what it was handed and can reject one setting."""

    def __init__(self, reject: str | None = None):
        self.applied: list[str] = []
        self._reject = reject

    def __call__(self, setting: str) -> bool:
        self.applied.append(setting)
        return self._reject is None or self._reject not in setting


def test_apply_settings_redirects_both_driver_caches():
    """Both driver cache settings must be pointed at the requested directory."""
    applier = _RecordingApplier()

    apply_shader_cache_settings(applier, _CACHE_PATH)

    assert applier.applied == [
        f"--/rtx/shaderDb/driverShaderCachePath={_CACHE_PATH}",
        f"--/rtx/shaderDb/driverAppShaderCachePath={_CACHE_PATH}",
    ]


@pytest.mark.parametrize("rejected", ["driverShaderCachePath", "driverAppShaderCachePath"])
def test_apply_settings_raises_when_a_setting_is_rejected(rejected, caplog):
    """A rejected setting must fail loudly, and must not be reported as a working redirect."""
    applier = _RecordingApplier(reject=rejected)

    with caplog.at_level("INFO", logger=ovrtx_shader_cache.__name__):
        with pytest.raises(RuntimeError, match=rejected):
            apply_shader_cache_settings(applier, _CACHE_PATH)

    assert "redirected" not in caplog.text


def test_redirect_is_skipped_when_env_var_is_unset(monkeypatch):
    """Without the env var the renderer must not touch settings at all."""
    monkeypatch.delenv(SHADER_CACHE_PATH_ENV, raising=False)
    monkeypatch.setattr(
        ovrtx_shader_cache,
        "_acquire_settings_applier",
        lambda config: pytest.fail("settings must not be queried when the env var is unset"),
    )

    redirect_shader_cache(_CONFIG)


def test_redirect_applies_settings_with_the_renderer_config(monkeypatch):
    """The requested path must reach both settings, and the renderer's config the applier.

    The applier factory is what initializes the ovrtx library, and initialization
    runs once per process, so a config dropped here is a config the renderer never
    gets - losing its log sink and level with no other symptom.
    """
    applier = _RecordingApplier()
    seen = []
    monkeypatch.setenv(SHADER_CACHE_PATH_ENV, _CACHE_PATH)
    monkeypatch.setattr(
        ovrtx_shader_cache,
        "_acquire_settings_applier",
        lambda config: seen.append(config) or applier,
    )

    redirect_shader_cache(_CONFIG)

    assert seen == [_CONFIG]
    assert [setting.split("=", 1)[1] for setting in applier.applied] == [_CACHE_PATH, _CACHE_PATH]


def test_redirect_warns_when_settings_extension_is_unavailable(monkeypatch, caplog):
    """A runtime without the settings extension degrades to a warning, not a failure."""
    monkeypatch.setenv(SHADER_CACHE_PATH_ENV, _CACHE_PATH)
    monkeypatch.setattr(ovrtx_shader_cache, "_acquire_settings_applier", lambda config: None)

    with caplog.at_level("WARNING", logger=ovrtx_shader_cache.__name__):
        redirect_shader_cache(_CONFIG)

    assert SHADER_CACHE_PATH_ENV in caplog.text
