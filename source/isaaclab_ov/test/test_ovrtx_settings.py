# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for forwarding Isaac Lab RTX settings into the OVRTX runtime."""

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
    from isaaclab.app.settings_manager import get_settings_manager  # noqa: E402
    from isaaclab_ov.renderers import ovrtx_settings  # noqa: E402


@pytest.fixture
def settings_manager():
    """Provide the settings manager singleton with its recorded ``/rtx/`` settings restored after."""
    manager = get_settings_manager()
    saved = manager.get_with_prefix("/rtx/")
    yield manager
    for path in manager.get_with_prefix("/rtx/"):
        if path not in saved:
            manager.set(path, None)
    for path, value in saved.items():
        manager.set(path, value)


def test_bools_are_formatted_as_words_not_digits():
    """Carbonite auto-types ``0``/``1`` as integers, so bools must be spelled out."""
    assert ovrtx_settings._format_value(False) == "false"
    assert ovrtx_settings._format_value(True) == "true"
    assert ovrtx_settings._format_value(3) == "3"


def test_empty_settings_are_a_no_op():
    """No settings means the ovrtx library is never touched."""
    assert ovrtx_settings.apply_carb_settings({}) is True


def test_gaussian_tonemapping_setting_reaches_ovrtx(settings_manager):
    """A ``/rtx/`` setting recorded before the renderer exists is queued as a Kit-style token.

    This is the path :class:`~isaaclab.sensors.camera.Camera` relies on to disable Gaussian
    tonemapping for ISP/HDR outputs on the kit-less OVRTX backend.
    """
    settings_manager.set_bool("/rtx/rtpt/gaussian/skipTonemapping/enabled", False)

    applied: list[dict[str, object]] = []
    original = ovrtx_settings.apply_carb_settings
    ovrtx_settings.apply_carb_settings = applied.append  # type: ignore[assignment]
    try:
        ovrtx_settings.apply_pending_rtx_settings()
    finally:
        ovrtx_settings.apply_carb_settings = original  # type: ignore[assignment]

    assert applied == [{"/rtx/rtpt/gaussian/skipTonemapping/enabled": False}]


def test_only_rtx_settings_are_forwarded(settings_manager):
    """Isaac Lab's own settings are not RTX settings and must not be sent to the RTX runtime."""
    settings_manager.set_bool("/isaaclab/render/rtx_sensors", True)
    settings_manager.set_bool("/rtx/rtpt/gaussian/skipTonemapping/enabled", False)

    forwarded = settings_manager.get_with_prefix("/rtx/")

    assert "/isaaclab/render/rtx_sensors" not in forwarded
    assert forwarded["/rtx/rtpt/gaussian/skipTonemapping/enabled"] is False


def test_settings_extension_is_reachable():
    """The internal ``ovrtx.settings.apply_settings`` extension still exists in the installed ovrtx.

    Guards the unsupported entry point this module depends on: if a future ovrtx drops or renames it,
    this fails here rather than silently reverting Gaussian output to display-referred tonemapping.
    Only looks the extension up, so no setting is queued into the process-global OVRTX runtime.
    """
    assert ovrtx_settings.query_apply_settings_fn() is not None
