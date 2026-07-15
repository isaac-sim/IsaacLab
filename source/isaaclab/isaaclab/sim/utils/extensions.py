# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for interacting with Kit extensions."""


def enable_extension(extension_name: str) -> None:
    """Enable a Kit extension immediately when it is not already enabled.

    Args:
        extension_name: Name of the extension to enable.
    """
    import omni.kit.app  # noqa: PLC0415

    extension_manager = omni.kit.app.get_app().get_extension_manager()
    if not extension_manager.is_extension_enabled(extension_name):
        extension_manager.set_extension_enabled_immediate(extension_name, True)


def get_extension_path(extension_name: str) -> str:
    """Return the filesystem path of an enabled Kit extension.

    Args:
        extension_name: Name of the enabled extension.

    Returns:
        Filesystem path of the extension.
    """
    import omni.kit.app  # noqa: PLC0415

    extension_manager = omni.kit.app.get_app().get_extension_manager()
    extension_id = extension_manager.get_enabled_extension_id(extension_name)
    return extension_manager.get_extension_path(extension_id)
