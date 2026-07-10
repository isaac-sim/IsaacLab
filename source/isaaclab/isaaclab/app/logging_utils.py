# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-agnostic resolution and application of the Python logging level.

The logging intent expressed by the ``--verbose`` / ``--info`` CLI arguments must be
honored by every simulation backend, not just the Kit-based one. Keeping the resolution
here (rather than inside :class:`~isaaclab.app.AppLauncher`) lets the kitless launch path
apply the same level without constructing Kit.
"""

from __future__ import annotations

import argparse
import logging
import sys


def resolve_python_logging_level(launcher_args: argparse.Namespace | dict | None = None) -> int:
    """Resolve the intended Python logging level from launcher args and ``sys.argv``.

    ``--verbose`` maps to :data:`logging.DEBUG` and ``--info`` to :data:`logging.INFO`.
    When neither is requested, the current effective level of the root logger is kept
    (falling back to :data:`logging.WARNING` when unset).

    Args:
        launcher_args: Parsed launcher arguments as a namespace or dict. Defaults to None,
            which is treated as an empty set of arguments.

    Returns:
        The resolved logging level.
    """
    if launcher_args is None:
        args = {}
    elif isinstance(launcher_args, argparse.Namespace):
        args = launcher_args.__dict__
    else:
        args = launcher_args

    if args.get("verbose", False) or "--verbose" in sys.argv:
        return logging.DEBUG
    if args.get("info", False) or "--info" in sys.argv:
        return logging.INFO

    level = logging.getLogger().getEffectiveLevel()
    return logging.WARNING if level == logging.NOTSET else level


def apply_python_logging_level(level: int) -> None:
    """Apply a Python logging level to the root logger and its handlers.

    Args:
        level: The logging level to apply.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)
