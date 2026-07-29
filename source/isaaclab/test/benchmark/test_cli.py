# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for benchmark CLI value validation."""

import argparse

import pytest

from isaaclab.benchmark._cli import parse_non_negative_int, parse_positive_int, validate_warmup_steps


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
