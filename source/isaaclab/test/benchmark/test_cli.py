# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the argument helpers shared by the benchmark scripts."""

import argparse

import pytest

from isaaclab.benchmark._cli import (
    add_benchmark_output_args,
    add_play_args,
    parse_non_negative_int,
    parse_positive_int,
    validate_warmup_steps,
)

pytestmark = pytest.mark.benchmark


@pytest.mark.parametrize("argument_type", [parse_non_negative_int, parse_positive_int])
def test_integer_parsers_report_native_conversion_errors(argument_type, capsys):
    parser = argparse.ArgumentParser()
    parser.add_argument("--value", type=argument_type)

    with pytest.raises(SystemExit):
        parser.parse_args(["--value", "invalid"])

    assert f"argument --value: invalid {argument_type.__name__} value: 'invalid'" in capsys.readouterr().err


def test_integer_parsers_enforce_bounds():
    assert parse_non_negative_int("0") == 0 and parse_positive_int("1") == 1
    with pytest.raises(argparse.ArgumentTypeError, match="non-negative"):
        parse_non_negative_int("-1")
    with pytest.raises(argparse.ArgumentTypeError, match="greater than zero"):
        parse_positive_int("0")


@pytest.mark.parametrize(
    ("warmup_steps", "available_steps", "valid"), [(0, 1, True), (15, 16, True), (1, 1, False), (17, 16, False)]
)
def test_validate_warmup_steps_requires_a_remaining_sample(warmup_steps: int, available_steps: int, valid: bool):
    if valid:
        validate_warmup_steps(warmup_steps, available_steps)
    else:
        with pytest.raises(ValueError, match="must be less than resolved training environment steps"):
            validate_warmup_steps(warmup_steps, available_steps)


def test_benchmark_output_args_share_defaults_and_optional_learning_curve_args():
    parser = argparse.ArgumentParser()
    add_benchmark_output_args(parser)
    assert vars(parser.parse_args([])) == {
        "output_path": ".",
        "measure_sync_step": False,
        "warmup_steps": 1,
        "benchmark_formatter": "schema",
    }

    training = argparse.ArgumentParser()
    add_benchmark_output_args(training, include_learning_args=True)
    args = training.parse_args(
        ["--warmup_steps", "0", "--benchmark_formatter", "schema,omniperf", "--ema_alpha", "0.5", "--no_series"]
    )
    assert (args.warmup_steps, args.benchmark_formatter, args.ema_alpha, args.no_series) == (
        0,
        "schema,omniperf",
        0.5,
        True,
    )
    with pytest.raises(SystemExit):
        training.parse_args(["--warmup_steps", "-1"])


def test_play_args_relax_task_only_when_help_is_requested(capsys):
    parser = argparse.ArgumentParser()
    add_play_args(parser, ["--task", "T"], agent_default="agent_entry", agent_help="Agent entry point.")
    args = parser.parse_args(["--task", "T", "--num_steps", "5", "--video"])
    assert (args.task, args.num_steps, args.video, args.agent, args.checkpoint) == ("T", 5, True, "agent_entry", None)
    with pytest.raises(SystemExit):
        parser.parse_args([])

    help_parser = argparse.ArgumentParser()
    add_play_args(help_parser, ["--help"], agent_default=None, agent_help="Agent entry point.")
    with pytest.raises(SystemExit) as exc_info:
        help_parser.parse_args(["--help"])
    assert exc_info.value.code == 0
    assert "Agent entry point." in capsys.readouterr().out
