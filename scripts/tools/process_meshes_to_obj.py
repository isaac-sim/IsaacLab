# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert all mesh files to `.obj` in given folders."""

import argparse
import os
import shutil
import subprocess

# Constants
# Path to blender
BLENDER_EXE_PATH = shutil.which("blender")


def parse_cli_args():
    """Parse the input command line arguments."""
    # add argparse arguments
    parser = argparse.ArgumentParser("Utility to convert all mesh files to `.obj` in given folders.")
    parser.add_argument("input_dir", type=str, help="The input directory from which to load meshes.")
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        help="The output directory to save converted meshes into. Default is same as input directory.",
    )
    args_cli = parser.parse_args()
    # resolve output directory
    if args_cli.output_dir is None:
        args_cli.output_dir = args_cli.input_dir
    # return arguments
    return args_cli


def run_blender_convert2obj(in_file: str, out_file: str):
    """Calls the python script using `subprocess` to perform processing of mesh file.

    Args:
        in_file: Input mesh file.
        out_file: Output obj file.
    """
    # resolve for python file
    tools_dirname = os.path.dirname(os.path.abspath(__file__))
    script_file = os.path.join(tools_dirname, "blender_obj.py")
    # complete command
    command_exe = f"{BLENDER_EXE_PATH} --background --python {script_file} -- -i {in_file} -o {out_file}"
    # break command into list
    command_exe_list = command_exe.split(" ")
    # run command
    subprocess.run(command_exe_list)


def convert_meshes(source_folders: list[str], destination_folders: list[str]):
    """Processes all mesh files of supported format into OBJ file using blender.

    Args:
        source_folders: List of directories to search for meshes.
        destination_folders: List of directories to dump converted files.
    """
    # create folder for corresponding destination
    for folder in destination_folders:
        os.makedirs(folder, exist_ok=True)
    # iterate over each folder
    for in_folder, out_folder in zip(source_folders, destination_folders):
        # extract all dae files in the directory
        mesh_filenames = [f for f in os.listdir(in_folder) if f.endswith("dae")]
        mesh_filenames += [f for f in os.listdir(in_folder) if f.endswith("stl")]
        mesh_filenames += [f for f in os.listdir(in_folder) if f.endswith("STL")]
        # print status
        print(f"Found {len(mesh_filenames)} files to process in directory: {in_folder}")
        # iterate over each OBJ file
        for mesh_file in mesh_filenames:
            # extract meshname
            mesh_name = os.path.splitext(mesh_file)[0]
            # complete path of input and output files
            in_file_path = os.path.join(in_folder, mesh_file)
            out_file_path = os.path.join(out_folder, mesh_name + ".obj")
            # perform blender processing
            print("Processing: ", in_file_path)
            run_blender_convert2obj(in_file_path, out_file_path)


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_cli_args()
    # Run conversion
    convert_meshes([args.input_dir], [args.output_dir])
