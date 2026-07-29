# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for retained sensor micro-benchmark argument contracts."""

import subprocess
import sys
from pathlib import Path

import pytest

_REPOSITORY_ROOT = Path(__file__).parents[4]
_BACKENDS = ("isaaclab_physx", "isaaclab_newton", "isaaclab_ovphysx")
_SCRIPTS = ("contact_sensor", "frame_transformer", "imu_pva", "joint_wrench", "ray_caster")


def _script_args(backend: str, script: str, *args: str) -> list[str]:
    command = [
        sys.executable,
        str(_REPOSITORY_ROOT / "source" / backend / "benchmark" / "sensors" / f"benchmark_{script}.py"),
    ]
    if script == "imu_pva":
        command.extend(["--sensor", "imu"])
    command.extend(args)
    return command


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("script", _SCRIPTS)
@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("--num_envs", "0"), "must be greater than zero"),
        (("--num_steps", "0"), "must be greater than zero"),
        (("--warmup_steps", "-1"), "must be non-negative"),
        (("--physics_variant", "unknown"), "invalid choice"),
    ],
)
def test_sensor_entrypoint_rejects_invalid_common_arguments(
    backend: str, script: str, args: tuple[str, str], message: str
) -> None:
    """Invalid common arguments should fail cleanly before simulator startup."""
    result = subprocess.run(
        _script_args(backend, script, *args),
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 2
    assert message in result.stderr


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
