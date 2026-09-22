# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Catalog and launcher for the demos distributed with Isaac Lab."""

from __future__ import annotations

import runpy
import sys
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

from isaaclab.paths import ISAACLAB_ROOT


@dataclass(frozen=True)
class DemoSpec:
    """Metadata for one packaged demonstration."""

    name: str
    relative_path: str
    summary: str
    category: str = "showcase"
    extras: tuple[str, ...] = ()
    required_modules: tuple[str, ...] = ()

    @property
    def uvx_command(self) -> str:
        """Return the complete command for a released Isaac Lab package."""
        if not self.extras:
            return f"uvx isaaclab demo {self.name}"
        extras = ",".join(self.extras)
        return f"uvx --from 'isaaclab[{extras}]' isaaclab demo {self.name}"

    def missing_modules(self) -> tuple[str, ...]:
        """Return required Python modules that are unavailable."""
        return tuple(module for module in self.required_modules if find_spec(module) is None)

    @property
    def path(self) -> Path:
        """Return the demo script path in a checkout or installed wheel."""
        return _demo_root() / self.relative_path


def _demo_root() -> Path:
    """Return the root containing executable demo scripts."""
    installed_root = Path(__file__).resolve().parent / "_demos"
    if installed_root.is_dir():
        return installed_root
    return ISAACLAB_ROOT / "demos"


_ISAACSIM = {"extras": ("isaacsim",), "required_modules": ("isaacsim",)}
_TETRAHEDRALIZATION = {"extras": ("tetrahedralization",), "required_modules": ("pytetwild",)}
_TELEOP = {"extras": ("teleop",), "required_modules": ("websockets",)}


_DEMOS = (
    DemoSpec("arl-robot-1", "arl_robot_1.py", "Fly ARL Robot 1 with its position controller.", **_ISAACSIM),
    DemoSpec("arms", "arms.py", "Spawn and command several single-arm manipulators."),
    DemoSpec("bin-packing", "bin_packing.py", "Clone heterogeneous randomized bin layouts.", **_ISAACSIM),
    DemoSpec("bipeds", "bipeds.py", "Spawn a collection of biped robots."),
    DemoSpec("cables", "cables.py", "Simulate colliding cables with Newton VBD."),
    DemoSpec("deformables", "deformables.py", "Compare deformable objects across backends.", **_TETRAHEDRALIZATION),
    DemoSpec("h1-locomotion", "h1_locomotion.py", "Control a trained H1 locomotion policy.", **_ISAACSIM),
    DemoSpec("hands", "hands.py", "Command dexterous robot hands."),
    DemoSpec(
        "haply-teleoperation",
        "haply_teleoperation.py",
        "Teleoperate a Franka with Haply hardware.",
        **_TELEOP,
    ),
    DemoSpec(
        "heterogeneous-scene",
        "heterogeneous_scene.py",
        "Compose heterogeneous task scenes.",
        **_ISAACSIM,
    ),
    DemoSpec("markers", "markers.py", "Render reusable visualization markers.", **_ISAACSIM),
    DemoSpec("multi-asset", "multi_asset.py", "Spawn different assets across cloned environments."),
    DemoSpec(
        "newton-block-and-tackle",
        "newton_viewer_block_and_tackle.py",
        "Interact with a Newton VBD block-and-tackle scene.",
    ),
    DemoSpec("newton-dominoes", "newton_viewer_dominoes.py", "Interact with Newton XPBD dominoes."),
    DemoSpec("pick-and-place", "pick_and_place.py", "Interactively pick and place a cube.", **_ISAACSIM),
    DemoSpec("procedural-terrain", "procedural_terrain.py", "Generate procedural terrain meshes.", **_ISAACSIM),
    DemoSpec("quadcopter", "quadcopter.py", "Spawn and command a quadcopter."),
    DemoSpec("quadrupeds", "quadrupeds.py", "Spawn and command several quadruped robots."),
    DemoSpec(
        "visual-color-randomization",
        "visual_color_randomization.py",
        "Randomize visual materials on cloned assets.",
    ),
    DemoSpec("mpm-granular", "mpm/newton_mpm_granular.py", "Drop granular MPM material on obstacles.", "mpm"),
    DemoSpec(
        "mpm-two-way-coupling",
        "mpm/newton_mpm_twoway_coupling.py",
        "Couple MPM sand with rigid bodies.",
        "mpm",
    ),
    DemoSpec("snowball-smash", "mpm/snowball_smash.py", "Smash rigid crates with MPM snowballs.", "mpm"),
    DemoSpec("teapot-fill", "mpm/teapot_fill.py", "Fill and pour a teapot with MPM fluid.", "mpm"),
    DemoSpec("camera", "sensors/cameras.py", "Capture data from several camera configurations.", "sensor", **_ISAACSIM),
    DemoSpec("contact-sensor", "sensors/contact_sensor.py", "Inspect robot contact measurements.", "sensor"),
    DemoSpec(
        "frame-transformer",
        "sensors/frame_transformer_sensor.py",
        "Track transforms between robot frames.",
        "sensor",
        **_ISAACSIM,
    ),
    DemoSpec("imu", "sensors/imu_sensor.py", "Inspect inertial measurements.", "sensor", **_ISAACSIM),
    DemoSpec(
        "multi-mesh-ray-caster",
        "sensors/multi_mesh_raycaster.py",
        "Cast rays against several dynamic meshes.",
        "sensor",
    ),
    DemoSpec(
        "multi-mesh-ray-caster-camera",
        "sensors/multi_mesh_raycaster_camera.py",
        "Render depth and normals with a multi-mesh ray caster.",
        "sensor",
        **_ISAACSIM,
    ),
    DemoSpec(
        "newton-raycast-heightfield",
        "sensors/newton_raycast_heightfield.py",
        "Raycast against a Newton height field.",
        "sensor",
    ),
    DemoSpec(
        "newton-raycast-moving-geometry",
        "sensors/newton_raycast_moving_geometry.py",
        "Raycast against moving Newton geometry.",
        "sensor",
    ),
    DemoSpec("ppisp-camera", "sensors/ppisp_camera.py", "Compare PPISP camera renderers.", "sensor"),
    DemoSpec("pva", "sensors/pva_sensor.py", "Inspect pose, velocity, and acceleration data.", "sensor", **_ISAACSIM),
    DemoSpec("ray-caster", "sensors/raycaster_sensor.py", "Inspect a lidar-style ray caster.", "sensor", **_ISAACSIM),
    DemoSpec(
        "tactile-sensor",
        "sensors/tacsl_sensor.py",
        "Inspect camera and force-field tactile data.",
        "sensor",
        **_ISAACSIM,
    ),
)
_DEMOS_BY_NAME = {demo.name: demo for demo in _DEMOS}


def list_demos() -> tuple[DemoSpec, ...]:
    """Return all public demos in display order."""
    return _DEMOS


def get_demo(name: str) -> DemoSpec:
    """Return the demo registered under ``name``.

    Args:
        name: Public kebab-case demo name.

    Raises:
        KeyError: If no demo has the requested name.
    """
    return _DEMOS_BY_NAME[name]


def run_demo(name: str, args: list[str] | None = None) -> None:
    """Run a packaged demo as its ``__main__`` module.

    Args:
        name: Public kebab-case demo name.
        args: Arguments forwarded to the demo.
    """
    demo = get_demo(name)
    missing_modules = demo.missing_modules()
    if missing_modules:
        missing = ", ".join(missing_modules)
        raise ModuleNotFoundError(f"demo {name!r} requires missing module(s): {missing}. Run: {demo.uvx_command}")
    if not demo.path.is_file():
        raise FileNotFoundError(f"demo {name!r} is not installed at {demo.path}")
    original_argv = sys.argv
    try:
        sys.argv = [f"isaaclab demo {name}", *(args or [])]
        runpy.run_path(str(demo.path), run_name="__main__")
    finally:
        sys.argv = original_argv
