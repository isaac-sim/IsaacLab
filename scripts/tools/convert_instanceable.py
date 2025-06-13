# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Utility to bulk convert URDFs or mesh files into instanceable USD format.

Unified Robot Description Format (URDF) is an XML file format used in ROS to describe all elements of
a robot. For more information, see: http://wiki.ros.org/urdf

This script uses the URDF importer extension from Isaac Sim (``omni.isaac.urdf_importer``) to convert a
URDF asset into USD format. It is designed as a convenience script for command-line use. For more
information on the URDF importer, see the documentation for the extension:
https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/ext_omni_isaac_urdf.html


positional arguments:
  input               The path to the input directory containing URDFs and Meshes.
  output              The path to directory to store the instanceable files.

optional arguments:
  -h, --help                Show this help message and exit
  --conversion-type         Select file type to convert, urdf or mesh. (default: urdf)
  --merge-joints            Consolidate links that are connected by fixed joints. (default: False)
  --fix-base                Fix the base to where it is imported. (default: False)
  --make-instanceable       Make the asset instanceable for efficient cloning. (default: False)

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Utility to convert a URDF or mesh into an Instanceable asset.")
parser.add_argument("input", type=str, help="The path to the input directory.")
parser.add_argument("output", type=str, help="The path to directory to store converted instanceable files.")
parser.add_argument(
    "--conversion-type", type=str, default="both", help="Select file type to convert, urdf, mesh, or both."
)
parser.add_argument(
    "--merge-joints",
    action="store_true",
    default=False,
    help="Consolidate links that are connected by fixed joints.",
)
parser.add_argument("--fix-base", action="store_true", default=False, help="Fix the base to where it is imported.")
parser.add_argument(
    "--make-instanceable",
    action="store_true",
    default=True,
    help="Make the asset instanceable for efficient cloning.",
)
parser.add_argument(
    "--collision-approximation",
    type=str,
    default="convexDecomposition",
    choices=["convexDecomposition", "convexHull", "none"],
    help=(
        'The method used for approximating collision mesh. Set to "none" '
        "to not add a collision mesh to the converted mesh."
    ),
)
parser.add_argument(
    "--mass",
    type=float,
    default=None,
    help="The mass (in kg) to assign to the converted asset. If not provided, then no mass is added.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg, UrdfConverter, UrdfConverterCfg
from isaaclab.sim.schemas import schemas_cfg


def main():

    # Define conversion time given
    conversion_type = args_cli.conversion_type.lower()
    # Warning if conversion type input is not valid
    if conversion_type != "urdf" and conversion_type != "mesh" and conversion_type != "both":
        raise Warning("Conversion type is not valid, please select either 'urdf', 'mesh', or 'both'.")

    if not os.path.exists(args_cli.input):
        print(f"Error: The directory {args_cli.input} does not exist.")

    # For each file and subsequent sub-directory
    for root, dirs, files in os.walk(args_cli.input):
        # For each file
        for filename in files:
            # Check for URDF extensions
            if (conversion_type == "urdf" or conversion_type == "both") and filename.lower().endswith(".urdf"):
                # URDF converter call
                urdf_converter_cfg = UrdfConverterCfg(
                    asset_path=f"{root}/{filename}",
                    usd_dir=f"{args_cli.output}/{filename[:-5]}",
                    usd_file_name=f"{filename[:-5]}.usd",
                    fix_base=args_cli.fix_base,
                    merge_fixed_joints=args_cli.merge_joints,
                    force_usd_conversion=True,
                    make_instanceable=args_cli.make_instanceable,
                )
                # Create Urdf converter and import the file
                urdf_converter = UrdfConverter(urdf_converter_cfg)
                print(f"Generated USD file: {urdf_converter.usd_path}")

            elif (conversion_type == "mesh" or conversion_type == "both") and (
                filename.lower().endswith(".fbx")
                or filename.lower().endswith(".obj")
                or filename.lower().endswith(".dae")
                or filename.lower().endswith(".stl")
            ):
                # Mass properties
                if args_cli.mass is not None:
                    mass_props = schemas_cfg.MassPropertiesCfg(mass=args_cli.mass)
                    rigid_props = schemas_cfg.RigidBodyPropertiesCfg()
                else:
                    mass_props = None
                    rigid_props = None

                # Collision properties
                collision_props = schemas_cfg.CollisionPropertiesCfg(
                    collision_enabled=args_cli.collision_approximation != "none"
                )
                # Mesh converter call
                mesh_converter_cfg = MeshConverterCfg(
                    mass_props=mass_props,
                    rigid_props=rigid_props,
                    collision_props=collision_props,
                    asset_path=f"{root}/{filename}",
                    force_usd_conversion=True,
                    usd_dir=f"{args_cli.output}/{filename[:-4]}",
                    usd_file_name=f"{filename[:-4]}.usd",
                    make_instanceable=args_cli.make_instanceable,
                    collision_approximation=args_cli.collision_approximation,
                )
                # Create mesh converter and import the file
                mesh_converter = MeshConverter(mesh_converter_cfg)
                print(f"Generated USD file: {mesh_converter.usd_path}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
