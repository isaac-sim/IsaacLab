# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command line argument utilities for RLinf integration with IsaacLab."""

from __future__ import annotations

import argparse


def add_rlinf_args(parser: argparse.ArgumentParser) -> None:
    """Add RLinf arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group("rlinf", description="Arguments for RLinf agent.")
    # -- config arguments
    arg_group.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to the RLinf configuration directory (for Hydra).",
    )
    arg_group.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Name of the RLinf configuration file (without .yaml extension).",
    )
    # -- load arguments
    arg_group.add_argument("--resume_dir", type=str, default=None, help="Directory to resume training from.")
    # -- training arguments
    arg_group.add_argument(
        "--only_eval", action="store_true", default=False, help="Only run evaluation without training."
    )
