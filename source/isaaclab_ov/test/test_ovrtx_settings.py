# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for forwarding Isaac Lab RTX settings into the OVRTX runtime."""

from __future__ import annotations

import importlib.util
import logging

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
    from isaaclab_ov.renderers import ovrtx_settings  # noqa: E402

    from isaaclab.app.settings_manager import get_settings_manager  # noqa: E402


@pytest.fixture
def settings_manager():
    """Provide the settings manager singleton with the settings a test records removed after.

    The manager is a process-wide singleton and has no delete API — ``set(path, None)`` would leave
    the path behind with a None value — so its standalone storage is snapshotted and restored
    wholesale to keep a test's writes out of the tests that follow.
    """
    manager = get_settings_manager()
    saved = dict(manager._standalone_settings)
    yield manager
    manager._standalone_settings.clear()
    manager._standalone_settings.update(saved)


@pytest.fixture
def unflushed_settings(monkeypatch):
    """Reset the module's record of what OVRTX already latched.

    :func:`~isaaclab_ov.renderers.ovrtx_settings.apply_pending_rtx_settings` only forwards once per
    process, so without this a test would see whatever an earlier test in the same process flushed.
    """
    monkeypatch.setattr(ovrtx_settings, "_flushed_settings", None)


def test_bools_are_formatted_as_words_not_digits():
    """Carbonite auto-types ``0``/``1`` as integers, so bools must be spelled out."""
    assert ovrtx_settings._format_value(False) == "false"
    assert ovrtx_settings._format_value(True) == "true"
    assert ovrtx_settings._format_value(3) == "3"


def test_empty_settings_are_a_no_op():
    """No settings means the ovrtx library is never touched."""
    assert ovrtx_settings.apply_carb_settings({}) is True


def test_gaussian_tonemapping_setting_reaches_ovrtx(settings_manager, unflushed_settings):
    """A ``/rtx/`` setting recorded before the renderer exists is queued, and only that setting.

    This is the path :class:`~isaaclab.sensors.camera.Camera` relies on to disable Gaussian
    tonemapping for ISP/HDR outputs on the kit-less OVRTX backend. Isaac Lab's own settings share
    the manager but are not RTX settings, so they must not be sent to the RTX runtime.
    """
    settings_manager.set_bool("/isaaclab/render/rtx_sensors", True)
    settings_manager.set_bool("/rtx/rtpt/gaussian/skipTonemapping/enabled", False)

    applied: list[dict[str, object]] = []
    original = ovrtx_settings.apply_carb_settings
    ovrtx_settings.apply_carb_settings = applied.append  # type: ignore[assignment]
    try:
        ovrtx_settings.apply_pending_rtx_settings()
    finally:
        ovrtx_settings.apply_carb_settings = original  # type: ignore[assignment]

    assert applied == [{"/rtx/rtpt/gaussian/skipTonemapping/enabled": False}]


def test_settings_recorded_after_the_first_flush_are_reported_as_dropped(settings_manager, unflushed_settings, caplog):
    """A setting a later sensor records cannot reach ovrtx, so it must be reported, not dropped quietly.

    OVRTX latches its settings while the first renderer is built, so a camera constructed after that
    — a second :class:`~isaaclab.sim.SimulationContext` in one process, for instance — would
    otherwise render display-referred Gaussian radiance with no indication why.
    """
    applied: list[dict[str, object]] = []
    original = ovrtx_settings.apply_carb_settings
    ovrtx_settings.apply_carb_settings = applied.append  # type: ignore[assignment]
    try:
        ovrtx_settings.apply_pending_rtx_settings()
        # A later sensor records the setting only now, after the first renderer latched its values.
        settings_manager.set_bool("/rtx/rtpt/gaussian/skipTonemapping/enabled", False)
        with caplog.at_level(logging.WARNING, logger=ovrtx_settings.logger.name):
            ovrtx_settings.apply_pending_rtx_settings()
    finally:
        ovrtx_settings.apply_carb_settings = original  # type: ignore[assignment]

    # The second call must not pretend to apply anything: ovrtx would ignore it.
    assert applied == [{}]
    assert "/rtx/rtpt/gaussian/skipTonemapping/enabled" in caplog.text


def test_unchanged_settings_are_not_reported_as_dropped(settings_manager, unflushed_settings, caplog):
    """Re-forwarding the settings that were already latched is a no-op, not a warning.

    Every renderer constructor calls this, so an unchanged second call — the common case — must stay
    quiet or the warning would fire on every well-behaved multi-renderer run.
    """
    settings_manager.set_bool("/rtx/rtpt/gaussian/skipTonemapping/enabled", False)

    original = ovrtx_settings.apply_carb_settings
    ovrtx_settings.apply_carb_settings = lambda settings: True  # type: ignore[assignment]
    try:
        ovrtx_settings.apply_pending_rtx_settings()
        with caplog.at_level(logging.WARNING, logger=ovrtx_settings.logger.name):
            ovrtx_settings.apply_pending_rtx_settings()
    finally:
        ovrtx_settings.apply_carb_settings = original  # type: ignore[assignment]

    assert caplog.text == ""


def test_settings_extension_is_reachable():
    """The internal ``ovrtx.settings.apply_settings`` extension still exists in the installed ovrtx.

    Guards the unsupported entry point this module depends on: if a future ovrtx drops or renames it,
    this fails here rather than silently reverting Gaussian output to display-referred tonemapping.
    Only looks the extension up, so no setting is queued into the process-global OVRTX runtime.
    """
    assert ovrtx_settings.query_apply_settings_fn() is not None
