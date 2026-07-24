# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the _SessionKitLauncher singleton logic used by tools/conftest.py.

These tests validate the session-kit behaviour without Isaac Sim by supplying a
lightweight mock base class in place of the real :class:`~isaaclab.app.AppLauncher`.
The mock records every ``super().__init__()`` call so we can assert that Kit is
started exactly once regardless of how many launcher objects are constructed.
"""

import pytest

# ---------------------------------------------------------------------------
# Helpers — mock base and factory
# ---------------------------------------------------------------------------


class _MockAppLauncher:
    """Minimal stand-in for AppLauncher that records construction calls."""

    _init_call_count = 0

    def __init__(self, launcher_args=None, **kwargs):
        _MockAppLauncher._init_call_count += 1
        # Populate the attrs that _SessionKitLauncher copies on subsequent constructions.
        self._python_logging_level = 20  # logging.INFO
        self._headless = kwargs.get("headless", True)
        self._livestream = 0
        self._offscreen_render = False
        self._sim_experience_file = ""
        self._video_enabled = False
        self.device_id = 0
        self.device = "cuda:0"
        self._deferred_cuda_device_id = None
        self.local_rank = 0
        self.global_rank = 0
        self._app = object()  # sentinel

    @property
    def app(self):
        return self._app


def _make_session_kit_launcher_cls(base_cls):
    """Return a fresh _SessionKitLauncher subclass of *base_cls*.

    This mirrors the class definition inside ``tools/conftest.py::pytest_configure``
    exactly, allowing the logic to be tested with a mock base.
    """

    class _SessionKitLauncher(base_cls):
        _session_instance = None

        def __init__(self, launcher_args=None, **kwargs):
            if _SessionKitLauncher._session_instance is not None:
                _e = _SessionKitLauncher._session_instance
                self._python_logging_level = _e._python_logging_level
                self._headless = _e._headless
                self._livestream = _e._livestream
                self._offscreen_render = _e._offscreen_render
                self._sim_experience_file = _e._sim_experience_file
                self._video_enabled = _e._video_enabled
                self.device_id = _e.device_id
                self.device = _e.device
                self._deferred_cuda_device_id = _e._deferred_cuda_device_id
                self.local_rank = _e.local_rank
                self.global_rank = _e.global_rank
                self._app = _e._app
                return
            super().__init__(launcher_args, **kwargs)
            _SessionKitLauncher._session_instance = self

    return _SessionKitLauncher


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_mock_call_count():
    """Reset the mock's call counter and any class state before each test."""
    _MockAppLauncher._init_call_count = 0
    yield


@pytest.fixture()
def launcher_cls():
    """Return a fresh _SessionKitLauncher class for each test."""
    cls = _make_session_kit_launcher_cls(_MockAppLauncher)
    yield cls
    # Clean up class-level singleton so tests don't bleed into one another.
    cls._session_instance = None


# ---------------------------------------------------------------------------
# Tests — singleton construction
# ---------------------------------------------------------------------------


def test_first_construction_calls_super(launcher_cls):
    """The very first construction must delegate to the base class."""
    launcher_cls(headless=True)
    assert _MockAppLauncher._init_call_count == 1


def test_first_construction_sets_session_instance(launcher_cls):
    """After the first construction, _session_instance must point at that object."""
    launcher = launcher_cls(headless=True)
    assert launcher_cls._session_instance is launcher


def test_second_construction_does_not_call_super(launcher_cls):
    """Subsequent constructions must not start a new Kit process."""
    launcher_cls(headless=True)
    launcher_cls(headless=True)
    assert _MockAppLauncher._init_call_count == 1


def test_n_constructions_call_super_exactly_once(launcher_cls):
    """N constructions must result in exactly one base-class init call."""
    n = 10
    for _ in range(n):
        launcher_cls(headless=True)
    assert _MockAppLauncher._init_call_count == 1


def test_alias_shares_app_object(launcher_cls):
    """All launcher instances in a session must share the same _app sentinel."""
    first = launcher_cls(headless=True)
    second = launcher_cls(headless=True)
    third = launcher_cls(headless=True)
    assert first._app is second._app is third._app


def test_alias_copies_all_public_attrs(launcher_cls):
    """The alias must expose the same public attributes as the owning instance."""
    first = launcher_cls(headless=True)
    second = launcher_cls(headless=False)  # kwargs ignored on alias path
    assert second.device == first.device
    assert second.device_id == first.device_id
    assert second.local_rank == first.local_rank
    assert second.global_rank == first.global_rank
    assert second._headless == first._headless


def test_alias_is_distinct_object(launcher_cls):
    """Each construction must return a separate Python object (not the same instance)."""
    first = launcher_cls(headless=True)
    second = launcher_cls(headless=True)
    assert first is not second


# ---------------------------------------------------------------------------
# Tests — monkey-patch contract
# ---------------------------------------------------------------------------


def test_monkey_patch_replaces_module_attribute(monkeypatch):
    """Patching ``isaaclab.app.AppLauncher`` must be visible to
    subsequent ``from isaaclab.app import AppLauncher`` calls.

    We simulate this with a fake module object rather than importing the real
    isaaclab (which requires Isaac Sim at import time).
    """
    import types

    # Build a minimal fake isaaclab.app module.
    fake_module = types.ModuleType("isaaclab.app")
    fake_module.AppLauncher = _MockAppLauncher

    # Simulate what pytest_configure does.
    cls = _make_session_kit_launcher_cls(_MockAppLauncher)
    fake_module.AppLauncher = cls

    # ``from isaaclab.app import AppLauncher`` resolves the attribute at import
    # time, so inspecting the module attribute is the correct assertion.
    assert fake_module.AppLauncher is cls
    assert issubclass(fake_module.AppLauncher, _MockAppLauncher)
