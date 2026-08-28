# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Map missing imports back to the optional extra that provides them."""

from __future__ import annotations

import os
import sys

EXTRA_FOR_MODULE: dict[str, str] = {
    "isaacsim": "isaacsim",
    "leapp": "leapp",
    "moviepy": "video",
    "ovphysx": "ovphysx",
    "ovrtx": "ovrtx",
    "ovstage": "ov",
    "pytetwild": "tetrahedralization",
    "rerun": "rerun",
    "rl_games": "rl-games",
    "robomimic": "mimic",
    "skrl": "skrl",
    "stable_baselines3": "sb3",
    "viser": "viser",
}
"""Top-level import name to the optional extra that provides it.

Deliberate omissions: ``rsl_rl`` (``rsl-rl-lib`` is a base dependency, so it is never
missing because an extra was skipped) and ``rlinf`` (the ``rlinf`` extra installs that
framework's dependencies, not the framework itself, so the hint would point at the
wrong remedy).
"""


def missing_extra_hint(module: str, command: str | None = None) -> str | None:
    """Return an install hint when a module is provided by an optional extra.

    Args:
        module: Top-level module name that failed to import.
        command: Command to suggest re-running, without the installer prefix. Defaults to
            the current ``isaaclab`` invocation.

    Returns:
        An actionable message naming the extra and how to install it, or None when the
        module is not provided by any extra.
    """
    extra = EXTRA_FOR_MODULE.get(module)
    if extra is None:
        return None
    if command is None:
        command = f"isaaclab {' '.join(sys.argv[1:])}".strip()
    # uv run exports these; a conda or plain-pip install exports neither.
    if os.environ.get("UV_RUN_RECURSION_DEPTH") or os.environ.get("UV"):
        install = f"uv run --extra {extra} {command}"
    else:
        install = f'pip install "isaaclab-dev[{extra}]"'
    return f"{module} is not installed. It is provided by the '{extra}' extra:\n  {install}"
