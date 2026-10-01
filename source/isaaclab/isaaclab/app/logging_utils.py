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
import sys
from contextlib import contextmanager

_INFO_HANDLER_NAME = "isaaclab_info_stream"
_FALLBACK_HANDLER_NAME = "isaaclab_fallback_stream"


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


def resolve_python_logging_level(args: dict | None = None) -> int:
    """Return the level requested with ``--verbose`` / ``--info``, also read from ``sys.argv``.

    The root logger's level is not used because :func:`ensure_console_handlers` lowers it to INFO.

    Args:
        args: Parsed launcher arguments. Defaults to None, in which case only ``sys.argv`` is inspected.

    Returns:
        :data:`logging.DEBUG` for ``--verbose``, :data:`logging.INFO` for ``--info``, otherwise
        :data:`logging.WARNING`.
    """
    args = {} if args is None else args
    if args.get("verbose", False) or "--verbose" in sys.argv:
        return logging.DEBUG
    if args.get("info", False) or "--info" in sys.argv:
        return logging.INFO
    return logging.WARNING


def ensure_console_handlers(level: int, fallback: bool = True) -> None:
    """Print Isaac Lab INFO records and warnings on the console.

    Isaac Lab reports progress with ``logger.info`` on ``isaaclab*`` loggers. At the default WARNING level
    these records would be dropped, so this adds a stdout handler that prints only Isaac Lab INFO records
    and lowers the root logger to INFO so they are created. Other root handlers keep the level set by
    :func:`apply_python_logging_level`.

    Python prints warnings through its last-resort handler only while no handler is configured, which
    stops being true once the INFO handler exists. With ``fallback`` enabled and no other root handler, a
    stderr handler prints records at ``level`` and above instead. Pass ``fallback=False`` once another
    handler prints warnings, such as Kit's log bridge; an existing fallback handler is then removed.

    Calling this again is safe: each handler is added once.

    Args:
        level: The requested Python logging level, e.g. from :func:`resolve_python_logging_level`.
        fallback: Whether to print warnings on stderr when no other handler is configured.
    """
    root = logging.getLogger()
    handlers = {handler.name: handler for handler in root.handlers}
    if not fallback and _FALLBACK_HANDLER_NAME in handlers:
        root.removeHandler(handlers.pop(_FALLBACK_HANDLER_NAME))
    if level > logging.WARNING:
        return
    info_handler = handlers.get(_INFO_HANDLER_NAME)
    if info_handler is None:
        info_handler = logging.StreamHandler(sys.stdout)
        info_handler.name = _INFO_HANDLER_NAME
        info_handler.addFilter(_is_isaaclab_info)
        info_handler.setFormatter(logging.Formatter("[INFO]: %(message)s"))
        root.addHandler(info_handler)
    # apply_python_logging_level() sets every root handler to the requested level; this one stays at INFO
    info_handler.setLevel(logging.INFO)
    has_other_handlers = any(name not in (_INFO_HANDLER_NAME, _FALLBACK_HANDLER_NAME) for name in handlers)
    if fallback and not has_other_handlers:
        fallback_handler = handlers.get(_FALLBACK_HANDLER_NAME)
        if fallback_handler is None:
            fallback_handler = logging.StreamHandler(sys.stderr)
            fallback_handler.name = _FALLBACK_HANDLER_NAME
            fallback_handler.addFilter(lambda record: not _is_isaaclab_info(record))
            fallback_handler.setFormatter(logging.Formatter("[%(levelname)s]: %(message)s"))
            root.addHandler(fallback_handler)
        fallback_handler.setLevel(level)
    root.setLevel(min(level, logging.INFO))


def configure_console_logging(args: dict | None = None) -> int:
    """Apply the ``--verbose`` / ``--info`` level and set up the console handlers for an entry point.

    Command-line entry points call this first so that messages logged before a simulation runtime
    starts reach the console.

    Args:
        args: Parsed launcher arguments. Defaults to None, in which case only ``sys.argv`` is inspected.

    Returns:
        The applied logging level.
    """
    level = resolve_python_logging_level(args)
    apply_python_logging_level(level)
    ensure_console_handlers(level)
    return level


def _is_isaaclab_info(record: logging.LogRecord) -> bool:
    """Return whether *record* is an INFO record from an Isaac Lab logger."""
    return record.levelno == logging.INFO and record.name.startswith("isaaclab")
