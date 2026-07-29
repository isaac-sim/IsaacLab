# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared CLI runtime for asset converter scripts."""

from __future__ import annotations

import argparse
import logging

from isaaclab.app import AppLauncher

from ._importer_api import ImporterKind, ImporterProvider

logger = logging.getLogger(__name__)


class ConverterCli:
    """Process bootstrap for the converter tool scripts.

    Contributes the ``--viz`` flag, decides once per process whether conversion runs through
    Kit or the standalone importer distribution, and offers a kitless post-conversion preview.
    Backend loading is owned by :class:`ImporterProvider`.

    Two related predicates answer different questions and are not interchangeable:
    :func:`~isaaclab.utils.version.has_kit` reports whether Kit is *running* (how
    :class:`ImporterProvider` adapts inside Kit), while :meth:`~isaaclab.app.AppLauncher.is_available`
    reports whether Kit is *launchable* (the decision made here, before any launch — when
    ``has_kit()`` is still ``False`` by definition).
    """

    @classmethod
    def parse_args(
        cls, parser: argparse.ArgumentParser, importer_kind: ImporterKind
    ) -> tuple[argparse.Namespace, object | None]:
        """Parse converter arguments, starting Kit only when available.

        Full Isaac Sim installations retain the established extension-backed path: Kit is launched
        (headless) and provides the importer modules. If Isaac Sim is absent, conversion runs
        through the standalone importer distribution. The optional ``--viz`` preview is handled
        after conversion by :meth:`preview` and never affects the conversion runtime, except
        that ``--viz kit`` is rejected here (before converting) when Isaac Sim is absent.

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
            nargs="?",
            const="auto",
            default="none",
            choices=["none", "auto", "kit", "newton", "rerun", "viser"],
            help=(
                "Preview the converted asset after conversion. Pass --viz on its own to use the "
                "backend that fits the runtime: the Kit viewport with a full Isaac Sim "
                "installation, otherwise the kitless Newton viewer. Name a backend to choose it "
                "explicitly -- 'kit' needs Isaac Sim, while 'newton', 'rerun', and 'viser' are "
                "kitless and only available when converting without Isaac Sim."
            ),
        )
        args_cli = parser.parse_args()

        if AppLauncher.is_available():
            if args_cli.viz == "auto":
                args_cli.viz = "kit"
            if args_cli.viz not in ("none", "kit"):
                # Conversion runs through Kit whenever Isaac Sim is installed, and a kitless
                # visualizer cannot share that process. Reject the combination instead of
                # converting and then silently skipping the preview that was asked for.
                parser.error(
                    f"--viz {args_cli.viz} is a kitless preview and cannot be used with Isaac Sim "
                    "installed. Use --viz kit, or run the converter in an environment without "
                    "Isaac Sim to preview with newton, rerun, or viser."
                )
            # Isaac Sim provides the importer modules through Kit; the launcher stays internal so
            # the converter CLI is limited to converter arguments. '--viz kit' additionally needs
            # the Kit visualizer, otherwise the app resolves without a viewport and the preview
            # would pump an invisible window.
            launcher_args = {"visualizer": ["kit"]} if args_cli.viz == "kit" else {}
            return args_cli, AppLauncher(launcher_args).app

        if args_cli.viz == "auto":
            # the Newton viewer ships with the base install, unlike rerun and viser
            args_cli.viz = "newton"

        if args_cli.viz == "kit":
            # fail before converting: the Kit viewport needs a full Isaac Sim installation
            parser.error(
                "--viz kit requires a full Isaac Sim installation, which was not found. Use "
                "--viz newton, --viz rerun, or --viz viser for a kitless preview."
            )

        if not ImporterProvider.is_standalone_available():
            raise ImportError(
                "Asset conversion requires either the full Isaac Sim package or the standalone "
                f"{ImporterProvider._STANDALONE_DIST!r} distribution."
            )

        ImporterProvider.validate_standalone_runtime(importer_kind)
        return args_cli, None

    @classmethod
    def preview(cls, args_cli: argparse.Namespace, simulation_app: object | None, usd_path: str) -> None:
        """Open the converted asset in the requested visualizer when a preview was requested.

        ``--viz kit`` displays the asset in the viewport of the Kit app that ran the conversion.
        ``--viz newton`` / ``rerun`` / ``viser`` launch a kitless
        :class:`~isaaclab.sim.SimulationContext` with the requested visualizer. Blocks in the
        visualizer render loop until the window is closed.

        Args:
            args_cli: Parsed converter arguments.
            simulation_app: Running ``SimulationApp`` when conversion used Isaac Sim, else ``None``.
            usd_path: Path of the generated USD file to open.
        """
        visualizer = getattr(args_cli, "viz", "none")
        if visualizer == "none":
            return

        # The asset is already converted and written at this point, so a preview that cannot run
        # is reported and skipped rather than failing the conversion.
        try:
            if visualizer == "kit":
                # ``parse_args`` rejects '--viz kit' without Isaac Sim, so Kit is running here.
                if not AppLauncher.has_gui():
                    logger.warning(
                        "Skipping the Kit preview: the app resolved without a GUI, so there is no"
                        " viewport to display the converted asset in."
                    )
                    return
                from isaaclab.sim.utils import show_stage_in_viewport  # noqa: PLC0415

                show_stage_in_viewport(usd_path)
                return
            # ``parse_args`` rejects the kitless visualizers when Isaac Sim is installed, so no Kit
            # app is running here.
            cls._preview_kitless(usd_path, visualizer)
        except Exception:
            logger.exception("Preview with the '%s' visualizer failed; the converted asset is at %s.", visualizer, usd_path)

    @classmethod
    def _preview_kitless(cls, usd_path: str, visualizer: str) -> None:
        """Open a converted USD asset in a kitless Isaac Lab visualizer.

        Reuses :func:`~isaaclab.app.launch_simulation` and
        :class:`~isaaclab.sim.SimulationContext`, so the asset renders through whichever kitless
        visualizer backend is requested with no backend-specific code: the Newton physics backend
        ingests the USD stage and every visualizer reads the shared scene data. Physics is not
        stepped -- the asset is shown in its imported pose until the visualizer window is closed.

        Args:
            usd_path: Path of the converted USD file to display.
            visualizer: Kitless visualizer backend to open the asset in, e.g. ``"newton"``.
        """
        import torch  # noqa: PLC0415

        import isaaclab.sim as sim_utils  # noqa: PLC0415
        from isaaclab.app import launch_simulation  # noqa: PLC0415
        from isaaclab.physics import PhysicsCfg  # noqa: PLC0415

        # Request the chosen kitless visualizer on the Newton backend; ``launch_simulation`` reads
        # the keys it needs and falls back to the launcher defaults for the rest. The default
        # device is 'cuda:0', so pick the CPU on hosts without a CUDA device.
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        launcher_args = {"physics": "newton_mjwarp", "visualizer": [visualizer], "device": device}

        with launch_simulation(cfg=PhysicsCfg(), launcher_args=launcher_args) as physics_cfg:
            sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=device, physics=physics_cfg))

            # add a light so the asset is visible, then reference the converted USD at the origin
            light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
            light_cfg.func("/World/Light", light_cfg)
            asset_cfg = sim_utils.UsdFileCfg(usd_path=usd_path)
            asset_cfg.func("/World/ConvertedAsset", asset_cfg)

            sim.reset()

            # nothing to display if no visualizer backend was created (e.g. package not installed)
            if not sim.visualizers:
                logger.warning(
                    "No '%s' visualizer available for preview (is 'isaaclab_visualizers[%s]' installed?)."
                    " Skipping viewport preview.",
                    visualizer,
                    visualizer,
                )
                return

            # Render the imported pose (without stepping physics) until the visualizer window closes.
            # Checked per visualizer rather than through
            # :meth:`~isaaclab.sim.SimulationContext.is_headless_or_exist_active_visualizer`: that
            # predicate also reports ``True`` for an empty visualizer list (headless stepping), and
            # ``render`` drops visualizers once they close or fail, so the preview would never exit.
            while any(viz.is_running() and not viz.is_closed for viz in sim.visualizers):
                sim.render()
