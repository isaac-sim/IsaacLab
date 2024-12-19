# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Convert a mesh file to `.obj` using blender.

This file processes a given dae mesh file and saves the resulting mesh file in obj format.

It needs to be called using the python packaged with blender, i.e.:

    blender --background --python blender_obj.py -- -in_file FILE -out_file FILE

For more information: https://docs.blender.org/api/current/index.html

The script was tested on Blender 3.2 on Ubuntu 20.04LTS.
"""

import bpy
import os
import sys


def parse_cli_args():
    """Parse the input command line arguments."""
    import argparse

    # get the args passed to blender after "--", all of which are ignored by
    # blender so scripts may receive their own arguments
    argv = sys.argv

    if "--" not in argv:
        argv = []  # as if no args are passed
    else:
        argv = argv[argv.index("--") + 1 :]  # get all args after "--"

    # When --help or no args are given, print this help
    usage_text = (
        f"Run blender in background mode with this script:\n\tblender --background --python {__file__} -- [options]"
    )
    parser = argparse.ArgumentParser(description=usage_text)
    # Add arguments
    parser.add_argument("-i", "--in_file", metavar="FILE", type=str, required=True, help="Path to input OBJ file.")
    parser.add_argument("-o", "--out_file", metavar="FILE", type=str, required=True, help="Path to output OBJ file.")
    args = parser.parse_args(argv)
    # Check if any arguments provided
    if not argv or not args.in_file or not args.out_file:
        parser.print_help()
        return None
    # return arguments
    return args


def convert_to_obj(in_file: str, out_file: str, save_usd: bool = False):
    """Convert a mesh file to `.obj` using blender.

    Args:
        in_file: Input mesh file to process.
        out_file: Path to store output obj file.
    """
    # check valid input file
    if not os.path.exists(in_file):
        raise FileNotFoundError(in_file)
    # add ending of file format
    if not out_file.endswith(".obj"):
        out_file += ".obj"
    # create directory if it doesn't exist for destination file
    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
    # reset scene to empty
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # load object into scene
    if in_file.endswith(".dae"):
        bpy.ops.wm.collada_import(filepath=in_file)
    elif in_file.endswith(".stl") or in_file.endswith(".STL"):
        bpy.ops.import_mesh.stl(filepath=in_file)
    else:
        raise ValueError(f"Input file not in dae/stl format: {in_file}")
    # convert to obj format and store with z up
    # TODO: Read the convention from dae file instead of manually fixing it.
    # Reference: https://docs.blender.org/api/2.79/bpy.ops.export_scene.html
    bpy.ops.export_scene.obj(
        filepath=out_file, check_existing=False, axis_forward="Y", axis_up="Z", global_scale=1, path_mode="RELATIVE"
    )
    # save it as usd as well
    if save_usd:
        out_file = out_file.replace("obj", "usd")
        bpy.ops.wm.usd_export(filepath=out_file, check_existing=False)


if __name__ == "__main__":
    # read arguments
    cli_args = parse_cli_args()
    # check CLI args
    if cli_args is None:
        sys.exit()
    # process via blender
    convert_to_obj(cli_args.in_file, cli_args.out_file)
