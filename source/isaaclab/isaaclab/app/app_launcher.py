# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated adapter that starts Isaac Sim / Kit through :func:`~isaaclab.app.launch_simulation`."""

from __future__ import annotations

import argparse
import contextlib
import logging
from typing import Any

from .sim_launcher import add_launcher_args, launch_simulation

logger = logging.getLogger(__name__)


_RUNTIME = contextlib.ExitStack()
"""Runtimes the adapter started; held for the process so they outlive the ``AppLauncher`` object."""


class _LaunchedApp:
    """The running Kit app; ``close()`` stops the runtime the adapter started."""

    def __init__(self, app: Any):
        self._app = app

    def close(self, *args, **kwargs) -> None:
        _RUNTIME.close()

    def is_exiting(self) -> bool:
        return not self._app.is_running()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._app, name)


class AppLauncher:
    """Start Isaac Sim / Kit for scripts that have not moved to :func:`~isaaclab.app.launch_simulation`.

    .. deprecated::
        Use :func:`~isaaclab.app.add_launcher_args` and ``with launch_simulation(cfg, args_cli):``, which
        start only the runtime the config needs.
    """

    def __init__(self, launcher_args: argparse.Namespace | dict | None = None, **kwargs):
        logger.warning("AppLauncher is deprecated; use isaaclab.app.launch_simulation.")
        # update the caller's namespace in place, as scripts read resolved values such as ``headless`` from it
        args = vars(launcher_args) if isinstance(launcher_args, argparse.Namespace) else dict(launcher_args or {})
        args.update(kwargs, require_kit=True)
        # Kit cannot share the process with the OVRTX visualizer; drop it as Kit did when it failed to load it
        if "newton_rtx" in (args.get("visualizer") or []):
            logger.warning("AppLauncher starts Kit, which cannot run the 'newton_rtx' visualizer; skipping it.")
            args["visualizer"] = [v for v in args["visualizer"] if v != "newton_rtx"]
        _RUNTIME.enter_context(launch_simulation(None, args))

        import omni.kit.app

        self.app = _LaunchedApp(omni.kit.app.get_app())

    @staticmethod
    def add_app_launcher_args(parser: argparse.ArgumentParser) -> None:
        """Add the launcher arguments, see :func:`~isaaclab.app.add_launcher_args`."""
        add_launcher_args(parser)
