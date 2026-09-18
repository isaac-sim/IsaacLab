# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Internal PhysX tensor-view wrapper for pose-write tracking."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class _PoseTrackingView:
    """Delegate to a native PhysX view and track successful pose writes."""

    def __init__(self, native_view: Any, on_pose_write: Callable[[], None]):
        self._native_view = native_view
        self._on_pose_write = on_pose_write

    def __getattr__(self, name: str) -> Any:
        return getattr(self._native_view, name)

    def set_root_transforms(self, *args, **kwargs) -> Any:
        """Set articulation root transforms and record the successful mutation."""
        result = self._native_view.set_root_transforms(*args, **kwargs)
        self._on_pose_write()
        return result

    def set_transforms(self, *args, **kwargs) -> Any:
        """Set rigid-body transforms and record the successful mutation."""
        result = self._native_view.set_transforms(*args, **kwargs)
        self._on_pose_write()
        return result
