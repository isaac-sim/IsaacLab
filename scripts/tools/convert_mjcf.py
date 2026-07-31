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
  --viz [BACKEND]           Preview the converted asset; bare --viz picks the backend that fits
                            the runtime (kit, newton, rerun, viser). (default: no preview)
"""

import argparse
import os
import sys
import traceback

from isaaclab.sim.converters._converter_cli import ConverterCli


def _create_parser() -> argparse.ArgumentParser:
    """Create the MJCF converter argument parser."""
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
    return parser


args_cli, simulation_app = ConverterCli.parse_args(_create_parser(), "mjcf")

from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg  # noqa: E402
from isaaclab.utils.assets import check_file_path  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402


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

    # Create mjcf converter and import the file
    mjcf_converter = MjcfConverter(mjcf_converter_cfg)
    # print output
    print("MJCF importer output:")
    print(f"Generated USD file: {mjcf_converter.usd_path}")
    print("-" * 80)
    print("-" * 80)

    # Open the converted asset in a kitless visualizer (newton / rerun / viser) when requested.
    ConverterCli.preview(args_cli, simulation_app, mjcf_converter.usd_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        # Kit's shutdown hooks override the interpreter exit status, so force a failure code.
        # os._exit skips interpreter shutdown, so flush first or the diagnostics printed
        # above are lost whenever stdout is redirected (the CI case).
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    # close sim app
    if simulation_app is not None:
        simulation_app.close()
