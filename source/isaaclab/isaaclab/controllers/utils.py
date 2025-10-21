# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helper functions for Isaac Lab controllers.

This module provides utility functions to help with controller implementations.
"""

import os

from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.asset.exporter.urdf")

import nvidia.srl.tools.logger as logger
import omni.log
from nvidia.srl.from_usd.to_urdf import UsdToUrdf


def convert_usd_to_urdf(usd_path: str, output_path: str, force_conversion: bool = True) -> tuple[str, str]:
    """Convert a USD file to URDF format.

    Args:
        usd_path: Path to the USD file to convert.
        output_path: Directory to save the converted URDF and mesh files.
        force_conversion: Whether to force the conversion even if the URDF and mesh files already exist.
    Returns:
        A tuple containing the paths to the URDF file and the mesh directory.
    """
    usd_to_urdf_kwargs = {
        "node_names_to_remove": None,
        "edge_names_to_remove": None,
        "root": None,
        "parent_link_is_body_1": None,
        "log_level": logger.level_from_name("ERROR"),
    }

    urdf_output_dir = os.path.join(output_path, "urdf")
    urdf_file_name = os.path.basename(usd_path).split(".")[0] + ".urdf"
    urdf_output_path = urdf_output_dir + "/" + urdf_file_name
    urdf_meshes_output_dir = os.path.join(output_path, "meshes")

    if not os.path.exists(urdf_output_path) or not os.path.exists(urdf_meshes_output_dir) or force_conversion:
        usd_to_urdf = UsdToUrdf.init_from_file(usd_path, **usd_to_urdf_kwargs)
        os.makedirs(urdf_output_dir, exist_ok=True)
        os.makedirs(urdf_meshes_output_dir, exist_ok=True)

        output_path = usd_to_urdf.save_to_file(
            urdf_output_path=urdf_output_path,
            visualize_collision_meshes=False,
            mesh_dir=urdf_meshes_output_dir,
            mesh_path_prefix="",
        )

        # The current version of the usd to urdf converter creates "inf" effort,
        # This has to be replaced with a max value for the urdf to be valid
        # Open the file for reading and writing
        with open(urdf_output_path) as file:
            # Read the content of the file
            content = file.read()

        # Replace all occurrences of 'inf' with '0'
        content = content.replace("inf", "0.")

        # Open the file again to write the modified content
        with open(urdf_output_path, "w") as file:
            # Write the modified content back to the file
            file.write(content)
    return urdf_output_path, urdf_meshes_output_dir


def change_revolute_to_fixed(urdf_path: str, fixed_joints: list[str], verbose: bool = False):
    """Change revolute joints to fixed joints in a URDF file.

    This function modifies a URDF file by changing specified revolute joints to fixed joints.
    This is useful when you want to disable certain joints in a robot model.

    Args:
        urdf_path: Path to the URDF file to modify.
        fixed_joints: List of joint names to convert from revolute to fixed.
        verbose: Whether to print information about the changes being made.
    """
    with open(urdf_path) as file:
        content = file.read()

    for joint in fixed_joints:
        old_str = f'<joint name="{joint}" type="revolute">'
        new_str = f'<joint name="{joint}" type="fixed">'
        if verbose:
            omni.log.warn(f"Replacing {joint} with fixed joint")
            omni.log.warn(old_str)
            omni.log.warn(new_str)
            if old_str not in content:
                omni.log.warn(f"Error: Could not find revolute joint named '{joint}' in URDF file")
        content = content.replace(old_str, new_str)

    with open(urdf_path, "w") as file:
        file.write(content)
