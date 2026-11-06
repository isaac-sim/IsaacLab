# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for interacting with Kit extensions."""

from typing import Any


def enable_extension(extension_name: str) -> None:
    """Enable a Kit extension immediately when it is not already enabled.

    Args:
        extension_name: Name of the extension to enable.

    Raises:
        RuntimeError: If no Kit application is running.
    """
    extension_manager = _get_extension_manager()
    if not extension_manager.is_extension_enabled(extension_name):
        extension_manager.set_extension_enabled_immediate(extension_name, True)


def disable_extension(extension_name: str) -> None:
    """Disable a Kit extension immediately when it is enabled.

    Args:
        extension_name: Name of the extension to disable.

    Raises:
        RuntimeError: If no Kit application is running.
    """
    extension_manager = _get_extension_manager()
    if extension_manager.is_extension_enabled(extension_name):
        extension_manager.set_extension_enabled_immediate(extension_name, False)


def get_extension_path(extension_name: str) -> str:
    """Return the filesystem path of an enabled Kit extension.

    Args:
        extension_name: Name of the enabled extension.

    Returns:
        Filesystem path of the extension.

    Raises:
        RuntimeError: If no Kit application is running.
    """
    extension_manager = _get_extension_manager()
    extension_id = extension_manager.get_enabled_extension_id(extension_name)
    return extension_manager.get_extension_path(extension_id)


def _get_extension_manager() -> Any:
    """Return Kit's extension manager from a running application."""
    import omni.kit.app  # noqa: PLC0415

    try:
        app = omni.kit.app.get_app()
    except RuntimeError as exc:
        raise RuntimeError("Cannot manage Kit extensions because no Kit application is running.") from exc
    if app is None:
        raise RuntimeError("Cannot manage Kit extensions because no Kit application is running.")
    return app.get_extension_manager()
