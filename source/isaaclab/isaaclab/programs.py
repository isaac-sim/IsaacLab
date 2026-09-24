# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Catalog and launcher for programs distributed with Isaac Lab."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType

from isaaclab.paths import ISAACLAB_ROOT


@dataclass(frozen=True)
class ProgramSpec:
    """Metadata for one packaged program."""

    name: str
    relative_path: str
    summary: str
    extras: tuple[str, ...] = ()
    required_modules: tuple[str, ...] = ()
    newton_gl_args: tuple[str, ...] | None = ()

    def uvx_command(self, command: str) -> str:
        """Return the command for running the program from a released package."""
        if not self.extras:
            return f"uvx isaaclab {command} {self.name}"
        extras = ",".join(self.extras)
        return f"uvx --from 'isaaclab[{extras}]' isaaclab {command} {self.name}"

    def missing_modules(self) -> tuple[str, ...]:
        """Return required Python modules that are unavailable."""
        return tuple(module for module in self.required_modules if find_spec(module) is None)

    @property
    def path(self) -> Path:
        """Return the program path in a checkout or installed wheel."""
        directory, relative_path = self.relative_path.split("/", 1)
        return _program_root(directory) / relative_path


def _program_root(directory: str) -> Path:
    """Return the root containing executable programs."""
    installed_root = Path(__file__).resolve().parent / directory
    if installed_root.is_dir():
        return installed_root
    return ISAACLAB_ROOT / directory


def _run_script(path: Path) -> None:
    """Run a script as ``__main__`` without replacing ``sys.argv[0]``."""
    module = ModuleType("__main__")
    module.__file__ = str(path)
    module.__package__ = None
    module.__spec__ = None
    original_main = sys.modules.get("__main__")
    try:
        sys.modules["__main__"] = module
        exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    finally:
        if original_main is None:
            sys.modules.pop("__main__", None)
        else:
            sys.modules["__main__"] = original_main


_RESET_CALLBACK_NAME = "isaaclab.programs"
"""Name of the simulation reset callback installed while a program runs."""

_ISAACSIM = {"extras": ("isaacsim",), "required_modules": ("isaacsim",)}
_TETRAHEDRALIZATION = {"extras": ("tetrahedralization",), "required_modules": ("pytetwild",)}
_TELEOP = {"extras": ("teleop",), "required_modules": ("isaaclab_teleop", "websockets")}


DEMOS = (
    ProgramSpec("zoo", "examples/demos/zoo.py", "Explore Isaac Lab robots and simulation features."),
    ProgramSpec("h1-locomotion", "examples/demos/h1_locomotion.py", "Control a trained H1 locomotion policy."),
    ProgramSpec(
        "pick-and-place",
        "examples/demos/pick_and_place.py",
        "Interactively pick and place a cube.",
        **_ISAACSIM,
        newton_gl_args=None,
    ),
    ProgramSpec(
        "newton-block-and-tackle",
        "examples/demos/newton_viewer_block_and_tackle.py",
        "Interact with a Newton VBD block-and-tackle scene.",
    ),
    ProgramSpec(
        "snowball-smash",
        "examples/demos/snowball_smash.py",
        "Smash rigid crates with MPM snowballs.",
    ),
    ProgramSpec("teapot-fill", "examples/demos/teapot_fill.py", "Fill and pour a teapot with MPM fluid."),
)


EXAMPLES = (
    ProgramSpec(
        "bin-packing",
        "examples/bin_packing.py",
        "Clone heterogeneous randomized bin layouts.",
        **_ISAACSIM,
    ),
    ProgramSpec("cables", "examples/cables.py", "Simulate colliding cables with Newton VBD."),
    ProgramSpec(
        "deformables",
        "examples/deformables.py",
        "Compare deformable objects across backends.",
        **_TETRAHEDRALIZATION,
        newton_gl_args=("--physics", "newton_vbd"),
    ),
    ProgramSpec(
        "heterogeneous-scene",
        "examples/heterogeneous_scene.py",
        "Compose heterogeneous task scenes.",
        **_ISAACSIM,
    ),
    ProgramSpec("markers", "examples/markers.py", "Render reusable visualization markers.", **_ISAACSIM),
    ProgramSpec("multi-asset", "examples/multi_asset.py", "Spawn different assets across cloned environments."),
    ProgramSpec("newton-dominoes", "examples/newton_viewer_dominoes.py", "Interact with Newton XPBD dominoes."),
    ProgramSpec(
        "procedural-terrain",
        "examples/procedural_terrain.py",
        "Generate procedural terrain meshes.",
        **_ISAACSIM,
    ),
    ProgramSpec(
        "visual-color-randomization",
        "examples/visual_color_randomization.py",
        "Randomize visual materials on cloned assets.",
    ),
    ProgramSpec(
        "mpm-granular",
        "examples/mpm/newton_mpm_granular.py",
        "Drop granular MPM material on obstacles.",
    ),
    ProgramSpec(
        "mpm-two-way-coupling",
        "examples/mpm/newton_mpm_twoway_coupling.py",
        "Couple MPM sand with rigid bodies.",
    ),
    ProgramSpec(
        "camera",
        "examples/sensors/cameras.py",
        "Capture data from several camera configurations.",
        **_ISAACSIM,
    ),
    ProgramSpec(
        "contact-sensor",
        "examples/sensors/contact_sensor.py",
        "Inspect robot contact measurements.",
    ),
    ProgramSpec(
        "frame-transformer",
        "examples/sensors/frame_transformer_sensor.py",
        "Track transforms between robot frames.",
        **_ISAACSIM,
    ),
    ProgramSpec(
        "imu",
        "examples/sensors/imu_sensor.py",
        "Inspect inertial measurements.",
        **_ISAACSIM,
    ),
    ProgramSpec(
        "multi-mesh-ray-caster",
        "examples/sensors/multi_mesh_raycaster.py",
        "Cast rays against several dynamic meshes.",
    ),
    ProgramSpec(
        "multi-mesh-ray-caster-camera",
        "examples/sensors/multi_mesh_raycaster_camera.py",
        "Render depth and normals with a multi-mesh ray caster.",
        **_ISAACSIM,
    ),
    ProgramSpec(
        "newton-raycast",
        "examples/sensors/newton_raycast.py",
        "Raycast against static or moving Newton geometry.",
    ),
    ProgramSpec(
        "pva",
        "examples/sensors/pva_sensor.py",
        "Inspect pose, velocity, and acceleration data.",
        **_ISAACSIM,
    ),
    ProgramSpec(
        "ray-caster",
        "examples/sensors/raycaster_sensor.py",
        "Inspect a lidar-style ray caster.",
        **_ISAACSIM,
    ),
    ProgramSpec(
        "arl-robot-1",
        "examples/arl_robot_1.py",
        "Fly ARL Robot 1 with its position controller.",
        **_ISAACSIM,
    ),
    ProgramSpec(
        "haply-teleoperation",
        "examples/haply_teleoperation.py",
        "Teleoperate a Franka with Haply hardware.",
        **_TELEOP,
        newton_gl_args=None,
    ),
    ProgramSpec(
        "ppisp-camera",
        "examples/sensors/ppisp_camera.py",
        "Compare PPISP camera renderers.",
        **_ISAACSIM,
        newton_gl_args=None,
    ),
    ProgramSpec(
        "tactile-sensor",
        "examples/sensors/tacsl_sensor.py",
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
        FileNotFoundError: If the program script is unavailable.
        ModuleNotFoundError: If an optional dependency is unavailable.
    """
    missing_modules = program.missing_modules()
    if missing_modules:
        missing = ", ".join(missing_modules)
        raise ModuleNotFoundError(
            f"{command} {program.name!r} requires missing module(s): {missing}. Run: {program.uvx_command(command)}"
        )
    path = program.path
    if not path.is_file():
        raise FileNotFoundError(f"{command} {program.name!r} is not installed at {path}")

    import warp as wp

    # Programs only run inference. Warp fixes this option when each kernel module is imported, so set it
    # before anything below imports one; it also matches the kernel cache keys produced by the test suites.
    wp.config.enable_backward = False

    from isaaclab.program_browser import ProgramBrowser

    browser = ProgramBrowser({"demo": DEMOS, "example": EXAMPLES})
    original_argv = sys.argv
    try:
        forwarded_args = list(args or [])
        sys.argv = [f"isaaclab {command} {program.name}", *forwarded_args]
        if any(arg in ("-h", "--help") for arg in forwarded_args):
            _run_script(path)
        else:
            from isaaclab.app.loading_screen import LoadingScreen
            from isaaclab.sim import SimulationContext

            verbose = any(arg in ("--info", "--verbose") for arg in forwarded_args)
            with LoadingScreen(1, enabled=False if verbose else None) as screen:
                screen.summary(
                    f"Isaac Lab · {command}",
                    {"Program": program.name, "Description": program.summary},
                )
                screen.stage("Launching simulation")

                def on_reset(sim: SimulationContext) -> None:
                    # Hand the console to the program once its simulation is ready.
                    screen.close()
                    browser.attach(sim)

                SimulationContext.add_reset_callback(_RESET_CALLBACK_NAME, on_reset)
                try:
                    _run_script(path)
                finally:
                    SimulationContext.remove_reset_callback(_RESET_CALLBACK_NAME)
                screen.close()
    finally:
        sys.argv = original_argv
    if browser.selected is not None:
        next_command, next_program = browser.selected
        sys.stdout.flush()
        sys.stderr.flush()
        os.execv(
            sys.executable,
            [
                sys.executable,
                "-m",
                "isaaclab",
                next_command,
                next_program.name,
                *(next_program.newton_gl_args or ()),
                "--viz",
                "newton_gl",
            ],
        )


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
