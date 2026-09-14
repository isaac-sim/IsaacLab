# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Construction boundary used by the isolated fixed-scene export worker."""

from collections.abc import Callable
from contextvars import ContextVar

# Only the export process installs a callback. Normal task construction is unchanged.
_scene_export_callback: ContextVar[Callable | None] = ContextVar("scene_export_callback", default=None)


def capture_before_events(env) -> None:
    """Visit the complete scene before EventManager construction or event execution."""
    callback = _scene_export_callback.get()
    if callback is not None:
        callback(env)
