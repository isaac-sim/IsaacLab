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

from types import SimpleNamespace

import pytest
from isaaclab_ov.renderers import ovrtx_shader_cache
from isaaclab_ov.renderers.ovrtx_shader_cache import SHADER_CACHE_PATH_ENV, redirect_shader_cache

_CACHE_PATH = "/tmp/isaaclab-ovrtx-kitless-cache"


class _RecordingApplier:
    """Settings applier that records what it was handed and can reject one setting."""

    def __init__(self, reject: str | None = None):
        self.applied: list[str] = []
        self._reject = reject

    def __call__(self, setting: str) -> bool:
        self.applied.append(setting)
        return self._reject is None or self._reject not in setting


@pytest.mark.parametrize("rejected", ["driverShaderCachePath", "driverAppShaderCachePath"])
def test_redirect_raises_when_a_setting_is_rejected(rejected, monkeypatch, caplog):
    """A rejected setting must fail loudly, and must not be reported as a working redirect."""
    monkeypatch.setenv(SHADER_CACHE_PATH_ENV, _CACHE_PATH)
    monkeypatch.setattr(ovrtx_shader_cache, "_acquire_settings_applier", lambda: _RecordingApplier(reject=rejected))

    with caplog.at_level("INFO", logger=ovrtx_shader_cache.__name__):
        with pytest.raises(RuntimeError, match=rejected):
            redirect_shader_cache()

    assert "redirected" not in caplog.text


def test_redirect_is_skipped_when_env_var_is_unset(monkeypatch):
    """Without the env var the renderer must not touch settings at all."""
    monkeypatch.delenv(SHADER_CACHE_PATH_ENV, raising=False)
    monkeypatch.setattr(
        ovrtx_shader_cache,
        "_acquire_settings_applier",
        lambda: pytest.fail("settings must not be queried when the env var is unset"),
    )

    redirect_shader_cache()


def test_redirect_applies_the_requested_path_to_both_settings(monkeypatch):
    """The requested path must reach both shader-cache settings."""
    applier = _RecordingApplier()
    monkeypatch.setenv(SHADER_CACHE_PATH_ENV, _CACHE_PATH)
    monkeypatch.setattr(ovrtx_shader_cache, "_acquire_settings_applier", lambda: applier)

    redirect_shader_cache()

    assert applier.applied == [
        f"--/rtx/shaderDb/driverShaderCachePath={_CACHE_PATH}",
        f"--/rtx/shaderDb/driverAppShaderCachePath={_CACHE_PATH}",
    ]


def test_redirect_raises_when_settings_extension_is_unavailable(monkeypatch):
    """A runtime without the settings extension must fail loudly.

    Silently recompiling at the default cache path would leave CI reporting the
    restore as a hit even though nothing was redirected.
    """
    monkeypatch.setenv(SHADER_CACHE_PATH_ENV, _CACHE_PATH)
    monkeypatch.setattr(ovrtx_shader_cache, "_acquire_settings_applier", lambda: None)

    with pytest.raises(RuntimeError, match=SHADER_CACHE_PATH_ENV):
        redirect_shader_cache()


def test_settings_applier_loads_ovrtx_without_initializing_it(monkeypatch):
    """The redirect only needs the settings extension; initialization and its config stay with the renderer."""
    bindings = pytest.importorskip("ovrtx._src.bindings")
    loaded = []

    class _QueryExtension:
        def __call__(self, name, out):
            return SimpleNamespace(status=bindings.OVRTX_API_SUCCESS + 1)

    loader = SimpleNamespace(
        _lib=None,
        _load_library=lambda paths: loaded.append(paths) or SimpleNamespace(ovrtx_query_extension=_QueryExtension()),
        create_bindings=lambda config: pytest.fail("the redirect must not initialize ovrtx"),
    )
    monkeypatch.setattr(bindings, "_ovrtx_loader", loader)
    monkeypatch.setattr(bindings, "OVRTX_LIBRARY_PATH_HINT", None)

    assert ovrtx_shader_cache._acquire_settings_applier() is None
    assert loaded == [None]
    assert loader._lib is None
