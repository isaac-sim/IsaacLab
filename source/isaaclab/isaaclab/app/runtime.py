# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Application operations that core code needs but only a runtime such as Kit provides.

The defaults do nothing, which is correct for kitless runs. A launcher that starts an application
installs its implementation with :func:`set_runtime`, so core modules never import the application.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import threading

    from pxr import Usd


class Runtime:
    """Operations of the running application; the base class is the kitless no-op runtime."""

    def update(self) -> None:
        """Run one application update."""

    def attach_stage(self, stage: Usd.Stage) -> None:
        """Show *stage* in the application's USD context, where its extensions discover it."""

    def close_stage(self) -> None:
        """Close the stage of the application's USD context."""

    def share_stage_context(self, context: threading.local) -> None:
        """Share Isaac Lab's thread-local current-stage *context* with the application's stage helpers."""

    def show_stage(self, usd_path: str) -> None:
        """Open *usd_path* in the application viewport and block until the application is closed."""
        raise RuntimeError("Showing a stage in a viewport requires a running Kit application.")


_runtime = Runtime()


def get_runtime() -> Runtime:
    """Return the installed application runtime."""
    return _runtime


def set_runtime(runtime: Runtime) -> None:
    """Install the application runtime that core operations delegate to.

    Args:
        runtime: The runtime of the started application.
    """
    global _runtime
    _runtime = runtime
