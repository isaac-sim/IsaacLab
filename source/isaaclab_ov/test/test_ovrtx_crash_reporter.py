# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the OVRTX crash-report upload settings.

Crash upload has no observable effect until something crashes, and its settings
are process-local, so a shim wired to the wrong process or the wrong config
reports nothing at all. These cover that boundary with a fake settings applier:
no GPU, no ovrtx runtime, no renderer.
"""

import pytest
from isaaclab_ov.renderers import ovrtx_crash_reporter
from isaaclab_ov.renderers.ovrtx_crash_reporter import (
    CRASH_REPORT_PRODUCT,
    CRASH_UPLOAD_ENV,
    apply_crash_report_settings,
    crash_report_settings,
    enable_crash_upload,
)

_OVRTX_VERSION = "0.4.1.364340"

# Only ever forwarded to the patched applier factory, so its contents never matter.
_CONFIG = object()


class _RecordingApplier:
    """Settings applier that records what it was handed."""

    def __init__(self):
        self.applied: list[str] = []

    def __call__(self, setting: str) -> None:
        self.applied.append(setting)


def test_settings_enable_upload_and_carry_the_runtime_version():
    """The reporter must be switched on, addressed, and stamped with the running ovrtx version.

    Spelled out rather than derived from the module: these strings are the contract
    the crash-report service is read against, so a typo in one is the whole feature.
    """
    assert crash_report_settings(_OVRTX_VERSION) == (
        "--/crashreporter/enabled=true",
        '--/crashreporter/url="https://services.nvidia.com/submit"',
        '--/crashreporter/product="Omniverse.ovrtx"',
        f'--/crashreporter/version="{_OVRTX_VERSION}"',
        "--/crashreporter/preserveDump=true",
        "--/crashreporter/gatherUserStory=false",
        "--/crashreporter/devOnlyOverridePrivacyAndForceUpload=true",
        "--/crashreporter/alwaysUpload=true",
    )


def test_every_setting_is_applied(caplog):
    """Applying a subset would leave the reporter half-configured and silent."""
    applier = _RecordingApplier()

    with caplog.at_level("INFO", logger=ovrtx_crash_reporter.__name__):
        apply_crash_report_settings(applier, _OVRTX_VERSION)

    assert applier.applied == list(crash_report_settings(_OVRTX_VERSION))
    assert CRASH_REPORT_PRODUCT in caplog.text


def test_upload_is_skipped_when_env_var_is_unset(monkeypatch):
    """Without the env var the renderer must not touch settings at all."""
    monkeypatch.delenv(CRASH_UPLOAD_ENV, raising=False)
    monkeypatch.setattr(
        ovrtx_crash_reporter,
        "_acquire_settings_applier",
        lambda config: pytest.fail("settings must not be applied when the env var is unset"),
    )

    enable_crash_upload(_CONFIG)


def test_upload_applies_settings_with_the_renderer_config(monkeypatch):
    """The applier factory must see the renderer's config, and every setting must land.

    The factory is what can initialize the ovrtx library, and initialization runs
    once per process, so a config dropped here is a config the renderer never gets -
    losing its log sink and level with no other symptom.
    """
    applier = _RecordingApplier()
    seen = []
    monkeypatch.setenv(CRASH_UPLOAD_ENV, "1")
    monkeypatch.setattr(ovrtx_crash_reporter, "version", lambda name: _OVRTX_VERSION)
    monkeypatch.setattr(
        ovrtx_crash_reporter,
        "_acquire_settings_applier",
        lambda config: seen.append(config) or applier,
    )

    enable_crash_upload(_CONFIG)

    assert seen == [_CONFIG]
    assert applier.applied == list(crash_report_settings(_OVRTX_VERSION))


def test_upload_raises_when_ovrtx_extensions_is_missing(monkeypatch):
    """An explicit request that cannot be honoured must fail loudly.

    The variable is only set once the wheelhouse installed the package, so skipping
    silently would leave a run looking like it uploads dumps it never collected.
    """
    monkeypatch.setenv(CRASH_UPLOAD_ENV, "1")
    monkeypatch.setattr(ovrtx_crash_reporter, "_acquire_settings_applier", lambda config: None)

    with pytest.raises(RuntimeError, match=CRASH_UPLOAD_ENV):
        enable_crash_upload(_CONFIG)
