# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command-line token helpers for the launcher arguments.

This module imports nothing heavy, so CLI dispatchers can use it before choosing a backend.
"""

from __future__ import annotations


def fuse_kit_args(argv: list[str]) -> list[str]:
    """Fuse ``["--kit_args", "<option-like value>"]`` pairs into single ``--kit_args=<value>`` tokens.

    Argparse rejects a value token that itself looks like an option (starts with ``-`` and contains
    no space) with "expected one argument", and Kit arguments always start with ``--``. Fusing the
    pair into the ``=``-attached form before parsing makes the documented space-separated form work
    for a single Kit argument. All other forms pass through unchanged.

    Args:
        argv: Command-line tokens, excluding the program name.

    Returns:
        Tokens with any affected pair replaced by one fused token.
    """
    fused: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        next_token = argv[index + 1] if index + 1 < len(argv) else None
        if token == "--kit_args" and next_token is not None and next_token.startswith("-") and " " not in next_token:
            fused.append(f"--kit_args={next_token}")
            index += 2
        else:
            fused.append(token)
            index += 1
    return fused
