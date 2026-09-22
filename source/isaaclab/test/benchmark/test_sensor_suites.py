# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the sensor micro-benchmark argument and workload helpers."""

import argparse
import math
import subprocess
import sys
from pathlib import Path

import pytest

from isaaclab.benchmark.sensor_suites import add_sensor_benchmark_args, rough_terrain_size

pytestmark = pytest.mark.benchmark

_REPOSITORY_ROOT = Path(__file__).parents[4]
_BACKENDS = ("isaaclab_physx", "isaaclab_newton", "isaaclab_ov")


def _sensor_parser(add_device: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_sensor_benchmark_args(
        parser,
        physics_variants=("newton_mjwarp", "newton_kamino"),
        default_physics_variant="newton_mjwarp",
        add_device=add_device,
    )
    return parser


def test_sensor_argument_helper_registers_one_consistent_common_contract() -> None:
    args = _sensor_parser(add_device=True).parse_args(
        [
            "--physics_variant",
            "newton_kamino",
            "--num_envs",
            "8",
            "--num_steps",
            "3",
            "--warmup_steps",
            "2",
            "--label",
            "candidate",
            "--output_path",
            "results",
            "--benchmark_formatter",
            "json",
            "--device",
            "cpu",
        ]
    )
    assert vars(args) == {
        "physics_variant": "newton_kamino",
        "num_envs": 8,
        "num_steps": 3,
        "warmup_steps": 2,
        "label": "candidate",
        "output_path": "results",
        "benchmark_formatter": "json",
        "device": "cpu",
    }
    assert "device" not in vars(_sensor_parser(add_device=False).parse_args([]))


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("--num_envs", "0"), "must be greater than zero"),
        (("--num_steps", "0"), "must be greater than zero"),
        (("--warmup_steps", "-1"), "must be non-negative"),
        (("--physics_variant", "unknown"), "invalid choice"),
    ],
)
def test_sensor_argument_helper_rejects_invalid_common_arguments(args: tuple[str, str], message: str, capsys) -> None:
    with pytest.raises(SystemExit):
        _sensor_parser(add_device=False).parse_args(args)
    assert message in capsys.readouterr().err


def test_rough_terrain_covers_every_environment_ray_grid() -> None:
    num_envs, env_spacing, ray_grid_size = 4096, 2.0, 1.0
    columns = math.ceil(math.sqrt(num_envs))
    rows = math.ceil(num_envs / columns)

    assert (
        rough_terrain_size(num_envs, env_spacing, ray_grid_size)
        >= max(columns - 1, rows - 1) * env_spacing + ray_grid_size
    )
    # Small smoke workloads keep a useful minimum surface.
    assert rough_terrain_size(1, 2.0, 1.0) == 3.0


def test_ray_caster_entrypoints_reject_unknown_terrain_workload_before_startup() -> None:
    """Every backend script validates its arguments before launching the simulator."""
    processes = {
        backend: subprocess.Popen(
            [
                sys.executable,
                str(_REPOSITORY_ROOT / "source" / backend / "benchmark" / "sensors" / "benchmark_ray_caster.py"),
                "--terrain",
                "unknown",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        for backend in _BACKENDS
    }

    for backend, process in processes.items():
        _, stderr = process.communicate(timeout=120)
        assert process.returncode == 2, (backend, stderr)
        assert "invalid choice" in stderr, backend
