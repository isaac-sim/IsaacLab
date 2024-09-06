# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Utility to convert a MJCF into USD format.

MuJoCo XML Format (MJCF) is an XML file format used in MuJoCo to describe all elements of a robot. For more information, see: http://www.mujoco.org/book/XMLreference.html

This script uses the MJCF importer extension from Isaac Sim (``omni.isaac.mjcf_importer``) to convert a MJCF asset into USD format. It is designed as a convenience script for command-line use. For more information on the MJCF importer, see the documentation for the extension:
https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/ext_omni_isaac_mjcf.html


positional arguments:
  input               The path to the input URDF file.
  output              The path to store the USD file.

optional arguments:
  -h, --help                Show this help message and exit
  --fix-base                Fix the base to where it is imported. (default: False)
  --make-instanceable       Make the asset instanceable for efficient cloning. (default: False)

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Utility to convert a MJCF into USD format.")
parser.add_argument("input", type=str, help="The path to the input MJCF file.")
parser.add_argument("output", type=str, help="The path to store the USD file.")
parser.add_argument("--fix-base", action="store_true", default=False, help="Fix the base to where it is imported.")
parser.add_argument(
    "--make-instanceable",
    action="store_true",
    default=False,
    help="Make the asset instanceable for efficient cloning.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import os

import carb
import omni.isaac.core.utils.stage as stage_utils
import omni.kit.app

from omni.isaac.lab.sim.converters import MjcfConverter, MjcfConverterCfg
from omni.isaac.lab.utils.assets import check_file_path
from omni.isaac.lab.utils.dict import print_dict


























if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
