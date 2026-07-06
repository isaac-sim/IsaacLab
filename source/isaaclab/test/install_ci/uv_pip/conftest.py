# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared fixtures for the ``uv_pip`` install_ci tests.

These tests install isaaclab from a pre-built wheel via ``uv pip install``. The
wheel is supplied by the runner (``tools/run_install_ci.py --build-wheel`` builds
it, ``--wheel <path>`` accepts a pre-built one) and surfaced to tests through the
``ISAACLAB_WHEEL`` env var / the ``wheel_path`` session fixture in the parent
conftest. The ``wheel`` fixture below is the strict variant: it fails the test if
no wheel was provided, rather than silently building one or skipping.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def wheel(wheel_path: Path | None) -> Path:
    """Pre-built isaaclab wheel for the uv_pip tests; errors out when missing.

    The wheel must be supplied by the test runner. Build it on-demand with
    ``tools/run_install_ci.py docker --build-wheel``, or pass an existing path
    via ``--wheel <path>`` / ``ISAACLAB_WHEEL``.
    """
    if wheel_path is None:
        pytest.fail(
            "uv_pip tests require a pre-built isaaclab wheel. Run"
            " `tools/run_install_ci.py docker --build-wheel` (or `--wheel <path>`),"
            " or set ISAACLAB_WHEEL=<path>."
        )
    return wheel_path
