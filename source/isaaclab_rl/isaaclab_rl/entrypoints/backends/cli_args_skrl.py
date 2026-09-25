# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command-line arguments shared by the skrl entrypoints."""

from __future__ import annotations

import argparse


def add_skrl_args(parser: argparse.ArgumentParser) -> None:
    """Add skrl arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    arg_group = parser.add_argument_group("skrl", description="Arguments for skrl agent.")
    arg_group.add_argument(
        "--ml_framework",
        type=str,
        default="torch",
        choices=["torch", "jax"],
        help="The ML framework used for training the skrl agent.",
    )
    arg_group.add_argument(
        "--algorithm",
        type=str,
        default=None,
        choices=["AMP", "PPO", "IPPO", "MAPPO"],
        help="Optional algorithm selector; with --agent, the resolved agent.class must match.",
    )
