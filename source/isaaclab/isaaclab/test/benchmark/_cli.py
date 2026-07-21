# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command-line argument utilities for benchmark scripts."""

import argparse


def parse_non_negative_int(value: str) -> int:
    """Parse a non-negative integer command-line argument."""
    parsed_value = _parse_int(value)
    if parsed_value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed_value


def parse_positive_int(value: str) -> int:
    """Parse a positive integer command-line argument."""
    parsed_value = _parse_int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed_value


def _parse_int(value: str) -> int:
    """Parse an integer and report conversion errors through argparse."""
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
