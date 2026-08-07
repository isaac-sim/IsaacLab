# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Utility to convert a MJCF into USD format.

MuJoCo XML Format (MJCF) is an XML file format used in MuJoCo to describe all elements of a robot.
For more information, see: http://www.mujoco.org/book/XMLreference.html

This script uses the MJCF importer API (``isaacsim.asset.importer.mjcf``) from Isaac Sim or its standalone
wheel to convert a MJCF asset into USD format. It is designed as a convenience script for command-line use.
For more information on the MJCF importer, see the documentation for the extension:
https://docs.isaacsim.omniverse.nvidia.com/latest/robot_setup/ext_isaacsim_asset_importer_mjcf.html


positional arguments:
  input               The path to the input MJCF file.
  output              The path to store the USD file.

optional arguments:
  -h, --help                Show this help message and exit
  --merge_mesh              Merge meshes where possible to optimize the model. (default: False)
  --collision_from_visuals  Generate collision geometry from visual geometries. (default: False)
  --collision_type          Type of collision geometry to use. (default: "Convex Hull")
  --self_collision          Activate self-collisions between links. (default: False)
  --import_physics_scene    Import the physics scene from the MJCF file. (default: False)

The standard launcher arguments are also accepted. In particular, ``--viz`` previews the converted
asset: ``--viz kit`` opens it in the Isaac Sim viewport, while ``--viz newton`` (or ``rerun`` /
``viser``) opens it kitlessly. Run with ``--help`` for the full list.

"""

"""Parse CLI first so we can decide whether to launch Isaac Sim Kit."""

import argparse
from importlib import metadata

from isaaclab.app import AppLauncher, add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Utility to convert a MJCF into USD format.")
parser.add_argument("input", type=str, help="The path to the input MJCF file.")
parser.add_argument("output", type=str, help="The path to store the USD file.")
parser.add_argument(
    "--merge_mesh",
    "--merge-mesh",
    action="store_true",
    default=False,
    help="Merge meshes where possible to optimize the model.",
)
parser.add_argument(
    "--collision_from_visuals",
    "--collision-from-visuals",
    action="store_true",
    default=False,
    help="Generate collision geometry from visual geometries.",
)
parser.add_argument(
    "--collision_type",
    "--collision-type",
    type=str,
    default="Convex Hull",
    choices=["Convex Hull", "Convex Decomposition", "Bounding Sphere", "Bounding Cube"],
    help='Type of collision geometry to use. Defaults to "Convex Hull".',
)
parser.add_argument(
    "--self_collision",
    "--self-collision",
    action="store_true",
    default=False,
    help="Activate self-collisions between links of the articulation.",
)
parser.add_argument(
    "--import_physics_scene",
    "--import-physics-scene",
    action="store_true",
    default=False,
    help="Import the physics scene (worldbody, defaults) from the MJCF file.",
)
add_launcher_args(parser)
args_cli = parser.parse_args()

# The MJCF importer ships as a Kit extension unless the standalone importer wheel is installed, so
# Kit is only required when the wheel is absent. With the wheel present the conversion runs kitlessly
# and the kitless visualizers can host the preview.
try:
    metadata.distribution("isaacsim-asset-isolated")
    args_cli.require_kit = False
except metadata.PackageNotFoundError:
    args_cli.require_kit = True

# Report the missing importer before converting anything. Without this the launcher reports only
# that Isaac Sim is absent, which does not mention the wheel that would make this run kitlessly.
if args_cli.require_kit and not AppLauncher.is_available():
    raise ImportError(
        "MJCF conversion requires either the full Isaac Sim runtime or the standalone"
        " 'isaacsim-asset-isolated' importer wheel, but neither is installed."
    )

import os  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.physics import PhysicsCfg  # noqa: E402
from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg  # noqa: E402
from isaaclab.utils.assets import check_file_path  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402


def preview(usd_path: str, physics_cfg: PhysicsCfg) -> None:
    """Open the converted asset in the visualizer selected on the command line.

    Args:
        usd_path: Path of the generated USD file to display.
        physics_cfg: Physics config resolved by :func:`~isaaclab.app.launch_simulation`.
    """
    visualizers = args_cli.visualizer or []
    if not visualizers:
        return

    if "kit" in visualizers:
        # a Kit app that resolved without a GUI has no viewport to display the asset in
        if AppLauncher.has_gui():
            sim_utils.show_stage_in_viewport(usd_path)
        return

    # Kitless preview: the physics backend ingests the USD stage and every visualizer renders the
    # shared scene data, so no backend-specific code is needed here. Physics is not stepped -- the
    # asset is shown in its imported pose until the visualizer window is closed.
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device, physics=physics_cfg))
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)
    asset_cfg = sim_utils.UsdFileCfg(usd_path=usd_path)
    asset_cfg.func("/World/ConvertedAsset", asset_cfg)
    sim.reset()

    # Checked per visualizer rather than through ``SimulationContext.is_headless_or_exist_active_visualizer``:
    # that predicate also reports True for an empty visualizer list (headless stepping), and ``render``
    # drops visualizers once they close, so the preview would never exit.
    while any(viz.is_running() and not viz.is_closed for viz in sim.visualizers):
        sim.render()


def main():
    # check valid file path
    mjcf_path = args_cli.input
    if not os.path.isabs(mjcf_path):
        mjcf_path = os.path.abspath(mjcf_path)
    if not check_file_path(mjcf_path):
        raise ValueError(f"Invalid file path: {mjcf_path}")
    # create destination path
    dest_path = args_cli.output
    if not os.path.isabs(dest_path):
        dest_path = os.path.abspath(dest_path)

    # create the converter configuration
    mjcf_converter_cfg = MjcfConverterCfg(
        asset_path=mjcf_path,
        usd_dir=os.path.dirname(dest_path),
        force_usd_conversion=True,
        merge_mesh=args_cli.merge_mesh,
        collision_from_visuals=args_cli.collision_from_visuals,
        collision_type=args_cli.collision_type,
        self_collision=args_cli.self_collision,
        import_physics_scene=args_cli.import_physics_scene,
    )

    # Print info
    print("-" * 80)
    print("-" * 80)
    print(f"Input MJCF file: {mjcf_path}")
    print("MJCF importer config:")
    print_dict(mjcf_converter_cfg.to_dict(), nesting=0)
    print("-" * 80)
    print("-" * 80)

    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        # Create mjcf converter and import the file
        mjcf_converter = MjcfConverter(mjcf_converter_cfg)
        # print output
        print("MJCF importer output:")
        print(f"Generated USD file: {mjcf_converter.usd_path}")
        print("-" * 80)
        print("-" * 80)

        preview(mjcf_converter.usd_path, physics_cfg)


if __name__ == "__main__":
    main()
