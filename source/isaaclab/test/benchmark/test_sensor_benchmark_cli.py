# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for sensor micro-benchmark argument contracts."""

import subprocess
import sys
from pathlib import Path

import pytest

_REPOSITORY_ROOT = Path(__file__).parents[4]
_BACKENDS = ("isaaclab_physx", "isaaclab_newton", "isaaclab_ovphysx")


def _script_args(backend: str, script: str, *args: str) -> list[str]:
    command = [
        sys.executable,
        str(_REPOSITORY_ROOT / "source" / backend / "benchmark" / "sensors" / f"benchmark_{script}.py"),
    ]
    command.extend(args)
    return command


@pytest.mark.parametrize("backend", _BACKENDS)
def test_ray_caster_entrypoint_rejects_unknown_terrain_workload(backend: str) -> None:
    """An unknown terrain workload should fail cleanly before simulator startup."""
    result = subprocess.run(
        _script_args(backend, "ray_caster", "--terrain", "unknown"),
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 2
    assert "invalid choice" in result.stderr
