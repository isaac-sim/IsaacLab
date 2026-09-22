# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Catalog and launcher for programs distributed with Isaac Lab."""

from __future__ import annotations

import argparse
import runpy
import sys
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path


@dataclass(frozen=True)
class ProgramSpec:
    """Metadata for one packaged program."""

    name: str
    module: str
    summary: str
    extras: tuple[str, ...] = ()
    required_modules: tuple[str, ...] = ()

    def uvx_command(self, command: str) -> str:
        """Return the command for running the program from a released package."""
        if not self.extras:
            return f"uvx isaaclab {command} {self.name}"
        extras = ",".join(self.extras)
        return f"uvx --from 'isaaclab[{extras}]' isaaclab {command} {self.name}"

    def missing_modules(self) -> tuple[str, ...]:
        """Return required Python modules that are unavailable."""
        return tuple(module for module in self.required_modules if find_spec(module) is None)


_ISAACSIM = {"extras": ("isaacsim",), "required_modules": ("isaacsim",)}
_TETRAHEDRALIZATION = {"extras": ("tetrahedralization",), "required_modules": ("pytetwild",)}
_TELEOP = {"extras": ("teleop",), "required_modules": ("isaaclab_teleop", "websockets")}


DEMOS = (
    ProgramSpec("zoo", "isaaclab.demos.zoo", "Explore Isaac Lab robots and simulation features."),
    ProgramSpec(
        "h1-locomotion",
        "isaaclab.demos.h1_locomotion",
        "Control a trained H1 locomotion policy.",
        **_ISAACSIM,
    ),
    ProgramSpec(
        "pick-and-place",
        "isaaclab.demos.pick_and_place",
        "Interactively pick and place a cube.",
        **_ISAACSIM,
    ),
    ProgramSpec(
        "newton-dominoes",
        "isaaclab.demos.newton_viewer_dominoes",
        "Interact with Newton XPBD dominoes.",
    ),
    ProgramSpec(
        "newton-block-and-tackle",
        "isaaclab.demos.newton_viewer_block_and_tackle",
        "Interact with a Newton VBD block-and-tackle scene.",
    ),
    ProgramSpec(
        "snowball-smash",
        "isaaclab.demos.mpm.snowball_smash",
        "Smash rigid crates with MPM snowballs.",
    ),
    ProgramSpec("teapot-fill", "isaaclab.demos.mpm.teapot_fill", "Fill and pour a teapot with MPM fluid."),
)


EXAMPLES = (
    ProgramSpec(
        "bin-packing",
        "isaaclab.examples.bin_packing",
        "Clone heterogeneous randomized bin layouts.",
        **_ISAACSIM,
    ),
    ProgramSpec("cables", "isaaclab.examples.cables", "Simulate colliding cables with Newton VBD."),
    ProgramSpec(
        "deformables",
        "isaaclab.examples.deformables",
        "Compare deformable objects across backends.",
        **_TETRAHEDRALIZATION,
    ),
    ProgramSpec(
        "heterogeneous-scene",
        "isaaclab.examples.heterogeneous_scene",
        "Compose heterogeneous task scenes.",
        **_ISAACSIM,
    ),
    ProgramSpec("markers", "isaaclab.examples.markers", "Render reusable visualization markers.", **_ISAACSIM),
    ProgramSpec("multi-asset", "isaaclab.examples.multi_asset", "Spawn different assets across cloned environments."),
    ProgramSpec(
        "procedural-terrain",
        "isaaclab.examples.procedural_terrain",
        "Generate procedural terrain meshes.",
        **_ISAACSIM,
    ),
    ProgramSpec(
        "visual-color-randomization",
        "isaaclab.examples.visual_color_randomization",
        "Randomize visual materials on cloned assets.",
    ),
    ProgramSpec(
        "mpm-granular",
        "isaaclab.examples.mpm.newton_mpm_granular",
        "Drop granular MPM material on obstacles.",
    ),
    ProgramSpec(
        "mpm-two-way-coupling",
        "isaaclab.examples.mpm.newton_mpm_twoway_coupling",
        "Couple MPM sand with rigid bodies.",
    ),
    ProgramSpec(
        "camera",
        "isaaclab.examples.sensors.cameras",
        "Capture data from several camera configurations.",
        **_ISAACSIM,
    ),
    ProgramSpec(
        "contact-sensor",
        "isaaclab.examples.sensors.contact_sensor",
        "Inspect robot contact measurements.",
    ),
    ProgramSpec(
        "frame-transformer",
        "isaaclab.examples.sensors.frame_transformer_sensor",
        "Track transforms between robot frames.",
        **_ISAACSIM,
    ),
    ProgramSpec(
        "imu",
        "isaaclab.examples.sensors.imu_sensor",
        "Inspect inertial measurements.",
        **_ISAACSIM,
    ),
    ProgramSpec(
        "multi-mesh-ray-caster",
        "isaaclab.examples.sensors.multi_mesh_raycaster",
        "Cast rays against several dynamic meshes.",
    ),
    ProgramSpec(
        "multi-mesh-ray-caster-camera",
        "isaaclab.examples.sensors.multi_mesh_raycaster_camera",
        "Render depth and normals with a multi-mesh ray caster.",
        **_ISAACSIM,
    ),
    ProgramSpec(
        "newton-raycast",
        "isaaclab.examples.sensors.newton_raycast",
        "Raycast against static or moving Newton geometry.",
    ),
    ProgramSpec(
        "pva",
        "isaaclab.examples.sensors.pva_sensor",
        "Inspect pose, velocity, and acceleration data.",
        **_ISAACSIM,
    ),
    ProgramSpec(
        "ray-caster",
        "isaaclab.examples.sensors.raycaster_sensor",
        "Inspect a lidar-style ray caster.",
        **_ISAACSIM,
    ),
    ProgramSpec(
        "arl-robot-1",
        "isaaclab_contrib.examples.arl_robot_1",
        "Fly ARL Robot 1 with its position controller.",
        **_ISAACSIM,
    ),
    ProgramSpec(
        "haply-teleoperation",
        "isaaclab_teleop.examples.haply_teleoperation",
        "Teleoperate a Franka with Haply hardware.",
        **_TELEOP,
    ),
    ProgramSpec(
        "ppisp-camera",
        "isaaclab_ppisp.examples.ppisp_camera",
        "Compare PPISP camera renderers.",
        **_ISAACSIM,
    ),
    ProgramSpec(
        "tactile-sensor",
        "isaaclab_contrib.examples.tacsl_sensor",
        "Inspect camera and force-field tactile data.",
        **_ISAACSIM,
    ),
)


def run_program(command: str, program: ProgramSpec, args: list[str] | None = None) -> None:
    """Run a packaged program as its ``__main__`` module.

    Args:
        command: CLI command that owns the program.
        program: Program to run.
        args: Arguments forwarded to the program.

    Raises:
        ModuleNotFoundError: If an optional dependency is unavailable.
    """
    missing_modules = program.missing_modules()
    if missing_modules:
        missing = ", ".join(missing_modules)
        raise ModuleNotFoundError(
            f"{command} {program.name!r} requires missing module(s): {missing}. Run: {program.uvx_command(command)}"
        )

    original_argv = sys.argv
    try:
        sys.argv = [f"isaaclab {command} {program.name}", *(args or [])]
        runpy.run_module(program.module, run_name="__main__")
    finally:
        sys.argv = original_argv


def run_program_cli(command: str, catalog: tuple[ProgramSpec, ...], args: list[str] | None = None) -> None:
    """List or run programs from a catalog.

    Args:
        command: Singular CLI command name.
        catalog: Programs available through the command.
        args: Command-line arguments. Uses ``sys.argv`` when omitted.
    """
    parser = argparse.ArgumentParser(
        description=f"Run a packaged Isaac Lab {command}.",
        prog=f"{Path(sys.argv[0]).name} {command}",
    )
    parser.add_argument("name", nargs="?", help=f"{command.title()} name, or 'list' to show the catalog.")
    if args is None:
        args = sys.argv[1:]
    if args and args[0] in ("-h", "--help"):
        parser.parse_args(args)
    parsed_args = parser.parse_args(args[:1])

    if parsed_args.name in (None, "list"):
        if len(args) > 1:
            parser.error("the list command does not accept additional arguments")
        name_width = max(len(program.name) for program in catalog)
        for program in catalog:
            install_hint = f"  [{program.uvx_command(command)}]" if program.extras else ""
            print(f"{program.name:<{name_width}}  {program.summary}{install_hint}")
        return

    programs_by_name = {program.name: program for program in catalog}
    try:
        program = programs_by_name[parsed_args.name]
    except KeyError:
        parser.error(f"unknown {command} {parsed_args.name!r}; run '{parser.prog} list' to see available {command}s")

    missing_modules = program.missing_modules()
    if missing_modules:
        missing = ", ".join(missing_modules)
        parser.error(
            f"{command} {program.name!r} requires missing module(s): {missing}. Run: {program.uvx_command(command)}"
        )
    run_program(command, program, args[1:])
