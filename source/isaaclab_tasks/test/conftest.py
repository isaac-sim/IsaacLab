# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pytest configuration for the isaaclab_tasks test suite.

Adds this directory to ``sys.path`` so tests located in the ``core/`` and ``contrib/``
sub-directories can import the shared helpers (``env_test_utils``, ``rendering_test_utils``)
that live at the test-suite root.
"""

import contextlib
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_OVRTX_LOG_PATH = os.path.join(tempfile.gettempdir(), "ovrtx_renderer.log")
"""Where the OVRTX renderer writes its log.

Mirrors the default of :attr:`~isaaclab_ov.renderers.OVRTXRendererCfg.log_file_path`, spelled out here so
this file does not import ``isaaclab_ov`` into every test in the suite.
"""


def _ovrtx_log_size() -> int:
    """Return the current size of the OVRTX log in bytes, or 0 when it does not exist yet."""
    try:
        return os.path.getsize(_OVRTX_LOG_PATH)
    except OSError:
        return 0


@pytest.fixture(autouse=True)
def _echo_ovrtx_log(request):
    """Replay whatever the OVRTX renderer appended to its log during the test.

    A no-op for tests that never build an OVRTX renderer, since nothing is written then. OVRTX runs with
    ``keep_system_alive=True`` and holds the log open for the lifetime of the process, so only the byte
    range written during this test is replayed. A log that shrank was reopened, and is replayed from the
    beginning instead.
    """
    start = _ovrtx_log_size()
    yield
    if _ovrtx_log_size() < start:
        start = 0
    with contextlib.suppress(OSError), open(_OVRTX_LOG_PATH, encoding="utf-8", errors="replace") as handle:
        handle.seek(start)
        if chunk := handle.read():
            print(f"\n----- OVRTX renderer log: {request.node.name} -----\n{chunk}", end="")


@pytest.fixture()
def enable_scene_partition(monkeypatch):
    """Set ``ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION=1`` for the duration of one test."""
    monkeypatch.setenv("ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION", "1")


@pytest.fixture()
def ovstage_variant(request, monkeypatch):
    """Select the indirectly parametrized OVRTX stage path."""
    if request.param == "ovstage":
        monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", "1")
    else:
        # Clear explicitly rather than relying on the variable being unset. An ambient
        # ISAAC_LAB_OVRTX_USE_OVSTAGE=1 would otherwise make both variants exercise the ovstage
        # path, silently dropping legacy coverage while still reporting two passing variants.
        monkeypatch.delenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", raising=False)
    return request.param
