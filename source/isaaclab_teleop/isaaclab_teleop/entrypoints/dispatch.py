# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dispatch ``isaaclab teleop`` command lines to their workflow."""

from __future__ import annotations

import argparse
import importlib
import sys

# workflow name -> (module, entrypoint); a module is imported only when its workflow runs
_WORKFLOWS = {
    "run": ("teleop_se3_agent", "run_teleop_se3_agent_cli"),
    "record": ("record_demos", "run_record_demos_cli"),
    "replay": ("replay_demos", "run_replay_demos_cli"),
}


def run_cli(argv: list[str] | None = None) -> int:
    """Run a teleoperation workflow: live teleoperation, demonstration recording or demonstration replay.

    Args:
        argv: The workflow name (``run``, ``record`` or ``replay``) followed by its arguments, excluding the
            program name. Defaults to ``sys.argv[1:]``.

    Returns:
        The workflow's process exit code.
    """
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(prog="isaaclab teleop", description="Run an Isaac Lab teleoperation workflow.")
    parser.add_argument("command", choices=tuple(_WORKFLOWS), help="Teleoperation workflow to run.")
    if not argv or argv[0] in ("-h", "--help"):
        parser.parse_args(argv)
    command = parser.parse_args(argv[:1]).command
    module_name, entrypoint = _WORKFLOWS[command]
    module = importlib.import_module(f".{module_name}", __package__)
    return getattr(module, entrypoint)(argv[1:])
