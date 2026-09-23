# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton GL selector for packaged demos and examples."""

from __future__ import annotations

from isaaclab._programs import DEMOS, EXAMPLES, ProgramSpec

_active = False
_selected: tuple[str, ProgramSpec] | None = None


def start_program() -> None:
    """Enable the selector for the current packaged program."""
    global _active, _selected
    _active = True
    _selected = None


def finish_program() -> tuple[str, ProgramSpec] | None:
    """Disable the selector and return a requested program switch."""
    global _active, _selected
    _active = False
    selected, _selected = _selected, None
    return selected


def register_newton_browser(viewer: object) -> None:
    """Add GL-compatible installed programs to a Newton GL viewer.

    Args:
        viewer: Newton viewer with a UI callback registry.
    """
    if not _active:
        return
    catalogs = tuple(
        (
            command,
            tuple(program for program in catalog if program.newton_gl_args is not None and not program.missing_modules()),
        )
        for command, catalog in (("demo", DEMOS), ("example", EXAMPLES))
    )

    def render(imgui: object) -> None:
        global _selected
        imgui.set_next_item_open(True, imgui.Cond_.appearing)
        if not imgui.collapsing_header("Isaac Lab Programs"):
            return
        for command, catalog in catalogs:
            if command == "demo":
                imgui.set_next_item_open(True, imgui.Cond_.appearing)
            if imgui.tree_node(command.title() + "s"):
                for program in catalog:
                    clicked, _ = imgui.selectable(f"{program.name.replace('-', ' ').title()}##{command}:{program.name}", False)
                    if clicked and _selected is None:
                        _selected = command, program
                        viewer._program_switch_requested = True
                    if imgui.is_item_hovered():
                        imgui.set_tooltip(program.summary)
                imgui.tree_pop()

    viewer.register_ui_callback(render, position="panel")
