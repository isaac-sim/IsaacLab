# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify that AppLauncher preserves the exit status of unhandled exceptions."""

import subprocess
import sys

import pytest

_TRIGGER_UNHANDLED_EXCEPTION = "--trigger-unhandled-exception"


def _raise_after_app_launcher_initialization() -> None:
    from isaaclab.app import AppLauncher

    AppLauncher(headless=True, device="cpu")
    raise RuntimeError("intentional AppLauncher failure")


@pytest.mark.integration
def test_unhandled_exception_exits_with_failure():
    """Verify that AppLauncher shutdown does not replace an exception's exit status with zero."""
    result = subprocess.run(
        [sys.executable, __file__, _TRIGGER_UNHANDLED_EXCEPTION],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert "RuntimeError: intentional AppLauncher failure" in result.stderr


if __name__ == "__main__" and _TRIGGER_UNHANDLED_EXCEPTION in sys.argv:
    _raise_after_app_launcher_initialization()
