# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared CLI runtime for asset converter scripts."""

from __future__ import annotations

import argparse
import contextlib
import logging
import os

import isaaclab.app.app_launcher as app_launcher_module
from isaaclab.app import AppLauncher

from ._importer_api import ImporterKind, ImporterProvider

logger = logging.getLogger(__name__)


class ConverterCli:
    """Parse converter arguments and select a Kit or standalone importer runtime."""

    @classmethod
    def parse_args(
        cls, parser: argparse.ArgumentParser, importer_kind: ImporterKind
    ) -> tuple[argparse.Namespace, object | None]:
        """Parse converter arguments, starting Kit only when available.

        Full Isaac Sim installations retain the established extension-backed path: Kit is
        launched (headless unless ``--viz kit`` requests the viewport preview) and provides
        the importer modules. If Isaac Sim is absent, conversion runs directly through the
        standalone importer distribution.

        Args:
            parser: Parser containing converter-specific arguments.
            importer_kind: Importer runtime required by the converter.

        Returns:
            Parsed arguments and the optional running ``SimulationApp``.

        Raises:
            ImportError: If the requested runtime is unavailable or cannot be loaded.
        """
        parser.add_argument(
            "--viz",
            type=str,
            default="none",
            choices=["kit", "none"],
            help=(
                "Open the converted stage in the Kit viewport after conversion. "
                "Requires the full Isaac Sim package."
            ),
        )
        args_cli = parser.parse_args()

        if app_launcher_module.SimulationApp is not None:
            # Kit provides the importer modules; the launcher is configured internally so the
            # converter CLI stays limited to converter arguments.
            launcher_args = {"visualizer": ["kit"]} if args_cli.viz == "kit" else {}
            return args_cli, AppLauncher(launcher_args).app

        if args_cli.viz == "kit":
            raise ImportError(
                "The Kit viewport preview requires the full Isaac Sim package. Install Isaac Sim "
                "or omit '--viz kit' for kitless conversion."
            )
        if os.environ.get("LIVESTREAM", "0") not in ("", "0"):
            logger.warning(
                "The LIVESTREAM environment variable is set, but livestreaming requires the full "
                "Isaac Sim package. Kitless conversion proceeds without streaming."
            )
        if not ImporterProvider.is_standalone_available():
            raise ImportError(
                "Asset conversion requires either the full Isaac Sim package or the standalone "
                f"{ImporterProvider.standalone_distribution!r} distribution."
            )

        ImporterProvider.validate_standalone_runtime(importer_kind)
        return args_cli, None

    @classmethod
    def maybe_preview(cls, args_cli: argparse.Namespace, simulation_app: object | None, usd_path: str) -> None:
        """Open the generated stage in the Kit viewport when preview was requested.

        Blocks in the Kit update loop until the window is closed (or the livestream client
        disconnects). Returns immediately when Kit is not running or no preview was
        requested.

        Args:
            args_cli: Parsed converter arguments.
            simulation_app: Running ``SimulationApp``, when Kit was launched.
            usd_path: Path of the generated USD file to open.
        """
        if simulation_app is None or not cls._should_open_stage(args_cli):
            return
        import omni.kit.app
        import omni.usd

        omni.usd.get_context().open_stage(usd_path)
        app = omni.kit.app.get_app_interface()
        with contextlib.suppress(KeyboardInterrupt):
            while app.is_running():
                app.update()

    @classmethod
    def _should_open_stage(cls, args_cli: argparse.Namespace) -> bool:
        """Return whether a preview was requested and survived launcher resolution."""
        # the LIVESTREAM environment variable is honored by the AppLauncher, so it counts as
        # a remote-preview request even though the converter CLI exposes no livestream flag
        try:
            livestream = int(os.environ.get("LIVESTREAM", 0))
        except ValueError:
            livestream = 0
        requested = getattr(args_cli, "viz", None) == "kit" or livestream in {1, 2}
        if not requested:
            return False
        # the resolved app state can drop the request (e.g. HEADLESS=1 disables all
        # visualizers) — without a window or livestream the update loop would spin
        # with no way to exit
        import carb

        settings = carb.settings.get_settings()
        return bool(settings.get("/app/window/enabled")) or bool(settings.get("/app/livestream/enabled"))
