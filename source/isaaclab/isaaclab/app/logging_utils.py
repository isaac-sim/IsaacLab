# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-agnostic application of the Python logging level.

The logging intent expressed by the ``--verbose`` / ``--info`` CLI arguments must be
honored by every simulation backend, not just the Kit-based one. Keeping the application
here (rather than inside :class:`~isaaclab_physx.app.KitLauncher`) lets the kitless launch path
apply the same level without constructing Kit.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager


@contextmanager
def force_log_level(level: int):
    """Context manager that temporarily lowers the root logger and its handlers to *level*, then restores them.

    Args:
        level: The logging level to enforce inside the block (e.g. :data:`logging.INFO`).
    """
    root = logging.getLogger()
    saved_root = root.level
    saved_handlers = [(h, h.level) for h in root.handlers]
    root.setLevel(level)
    for h, _ in saved_handlers:
        h.setLevel(level)
    try:
        yield
    finally:
        root.setLevel(saved_root)
        for h, saved in saved_handlers:
            h.setLevel(saved)


def apply_python_logging_level(level: int) -> None:
    """Apply a Python logging level to the root logger and its handlers.

    Args:
        level: The logging level to apply.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)
