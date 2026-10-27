# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton GL selector for packaged demos and examples."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .programs import ProgramSpec
    from .sim import SimulationContext
    from .visualizers import BaseVisualizer


class ProgramBrowser:
    """List packaged programs in Newton GL windows and remember the one the user picks.

    Picking a program or requesting a reset stops the current simulation. After the program
    cleans up, the launcher runs the selection or restarts the current program in the same window.
    """

    def __init__(self, catalogs: dict[str, tuple[ProgramSpec, ...]]) -> None:
        """Initialize the browser.

        Args:
            catalogs: Programs keyed by the CLI command that runs them, e.g. ``"demo"``. Programs
                that cannot run with Newton GL or are missing optional modules are left out.
        """
        self.selected: tuple[str, ProgramSpec] | None = None
        """CLI command and program picked by the user, or None."""
        self.restart_requested = False
        """Whether the current program should restart after its simulation closes."""
        self._catalogs = {
            command: tuple(
                program for program in catalog if program.newton_gl_args is not None and not program.missing_modules()
            )
            for command, catalog in catalogs.items()
        }
        self._attached: set[int] = set()

    def attach(self, sim: SimulationContext) -> None:
        """Add the program list to every Newton GL window of a simulation.

        Args:
            sim: Simulation whose visualizers receive the list. Windows that already show it are skipped.
        """
        for visualizer in sim.visualizers:
            if visualizer.cfg.backend.visualizer_type == "newton_gl" and id(visualizer) not in self._attached:
                self._attached.add(id(visualizer))
                visualizer.register_ui_callback(self._render_callback(sim), position="panel")
                visualizer.register_ui_callback(self._restart_callback(sim, visualizer), position="free")

    def _render_callback(self, sim: SimulationContext) -> Callable[[Any], None]:
        """Return the ImGui callback that draws the program list in one window."""

        def render(imgui: Any) -> None:
            imgui.set_next_item_open(True, imgui.Cond_.appearing)
            if not imgui.collapsing_header("Isaac Lab Programs"):
                return
            for command, catalog in self._catalogs.items():
                if command == "demo":
                    imgui.set_next_item_open(True, imgui.Cond_.appearing)
                if imgui.tree_node(command.title() + "s"):
                    for program in catalog:
                        clicked, _ = imgui.selectable(
                            f"{program.name.replace('-', ' ').title()}##{command}:{program.name}", False
                        )
                        if clicked and self.selected is None and not self.restart_requested:
                            self.selected = command, program
                            sim.stop()
                        if imgui.is_item_hovered():
                            imgui.set_tooltip(program.summary)
                    imgui.tree_pop()

        return render

    def _restart_callback(self, sim: SimulationContext, visualizer: BaseVisualizer) -> Callable[[Any], None]:
        """Observe toolbar reset even when the program panel is hidden."""

        def render(_imgui: Any) -> None:
            if self.selected is None and not self.restart_requested and visualizer.is_reset_requested():
                self.restart_requested = True
                sim.consume_reset_request()
                sim.stop()

        return render
