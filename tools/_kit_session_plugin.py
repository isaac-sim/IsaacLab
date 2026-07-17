# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pytest plugin for Kit session reuse across a batch of test files.

Loaded via ``-p _kit_session_plugin`` in batch subprocess commands built by
``tools/conftest.py``.  It should never be loaded in single-file runs.

The module starts a single Kit / SimulationApp at **import time** — before
pytest begins collecting test modules.  This means all subsequent
``from isaaclab.* import ...`` statements in collected test files land in an
already-running Kit process.

``AppLauncher.__init__`` is then patched so that every module-level
``simulation_app = AppLauncher(...).app`` call in a collected test file
silently reuses the shared launcher instead of starting a second
SimulationApp.

``set ISAACLAB_BATCH_CAMERAS=1`` in the subprocess environment to start Kit
with camera support enabled (used for batches that contain camera tests).
"""

from __future__ import annotations

import os

import pytest

# ---------------------------------------------------------------------------
# Start Kit at import time (before pytest collection)
# ---------------------------------------------------------------------------

from isaaclab.app import AppLauncher as _AppLauncher

_enable_cameras = os.environ.get("ISAACLAB_BATCH_CAMERAS") == "1"
_launcher = _AppLauncher(headless=True, enable_cameras=_enable_cameras)

# Patch so every subsequent AppLauncher() call reuses the shared session.
_original_init = _AppLauncher.__init__


def _shared_init(self, launcher_args=None, **kwargs):
    self.__dict__.update(_launcher.__dict__)


_AppLauncher.__init__ = _shared_init

# Prevent test teardown from closing the shared SimulationApp.
_launcher.app.close = lambda *a, **kw: None


# ---------------------------------------------------------------------------
# Kit state reset between files
# ---------------------------------------------------------------------------


def _reset_kit_state() -> None:
    """Reset Kit/USD state between test files in a shared session."""
    try:
        import omni.timeline

        omni.timeline.get_timeline_interface().stop()
    except Exception:
        pass

    try:
        import omni.usd

        omni.usd.get_context().new_stage()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _kit_session():
    """Hold the shared Kit session alive for the entire batch."""
    yield _launcher


@pytest.fixture(scope="module", autouse=True)
def _kit_module_fence():
    """Reset Kit/USD state after each test file completes."""
    yield
    _reset_kit_state()
