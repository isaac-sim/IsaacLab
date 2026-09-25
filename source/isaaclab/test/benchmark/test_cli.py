# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for benchmark CLI value validation."""

import argparse

import pytest

from isaaclab.benchmark._cli import parse_non_negative_int, parse_positive_int, validate_warmup_steps
from isaaclab.benchmark.sensor_suites import add_sensor_benchmark_args


@pytest.mark.parametrize("argument_type", [parse_non_negative_int, parse_positive_int])
def test_integer_parsers_report_native_conversion_errors(argument_type, capsys):
    parser = argparse.ArgumentParser()
    parser.add_argument("--value", type=argument_type)

    with pytest.raises(SystemExit):
        parser.parse_args(["--value", "invalid"])

    assert f"argument --value: invalid {argument_type.__name__} value: 'invalid'" in capsys.readouterr().err


@pytest.mark.parametrize(("warmup_steps", "available_steps"), [(0, 1), (15, 16)])
def test_validate_warmup_steps_accepts_a_remaining_sample(warmup_steps: int, available_steps: int):
    validate_warmup_steps(warmup_steps, available_steps)


@pytest.mark.parametrize(("warmup_steps", "available_steps"), [(1, 1), (17, 16)])
def test_validate_warmup_steps_rejects_exhausted_workload(warmup_steps: int, available_steps: int):
    with pytest.raises(ValueError, match="must be less than resolved training environment steps"):
        validate_warmup_steps(warmup_steps, available_steps)


def test_sensor_argument_helper_registers_one_consistent_common_contract() -> None:
    """Every sensor script should share the same workload and output arguments."""
    parser = argparse.ArgumentParser()
    add_sensor_benchmark_args(
        parser,
        physics_variants=("newton_mjwarp", "newton_kamino"),
        default_physics_variant="newton_mjwarp",
        add_device=True,
    )

    args = parser.parse_args(
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
    """Invalid common sensor arguments should fail through argparse."""
    parser = argparse.ArgumentParser()
    add_sensor_benchmark_args(
        parser,
        physics_variants=("newton_mjwarp", "newton_kamino"),
        default_physics_variant="newton_mjwarp",
        add_device=False,
    )

    with pytest.raises(SystemExit):
        parser.parse_args(args)

    assert message in capsys.readouterr().err
